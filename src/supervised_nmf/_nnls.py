"""Guarded multiple-right-hand-side NNLS for fixed SPA bases.

The block-principal-pivot strategy follows Kim and Park (2011),
doi:10.1137/110821172, and its public nonnegfac-python reference. The
implementation groups equal passive sets and reuses their Cholesky factors.
The public SciPy solver is the reference and the numerical fallback.

Third-party reference notice:
Copyright (c) 2014, Nokia Corporation
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Nokia Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NOKIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from numbers import Integral
import math

import numpy as np
from scipy import linalg
from scipy.optimize import nnls


# The acceleration guard is intentionally much stricter than SPA's modelling
# condition_limit. Normal equations square the condition number; at 1e4 the
# worst-case amplification of float64 roundoff is already about 2e-8.
_BATCH_CONDITION_LIMIT = 1e4
# QR avoids forming normal equations, but still needs a bounded admission
# domain and original-design qualification; this is not a forward-error bound.
_QR_CONDITION_LIMIT = 1e6
_AUTO_MIN_ROWS = 32
_FACTOR_CACHE_LIMIT = 128
# A bounded numerical domain, not a claim that problems outside it are
# mathematically unsolvable. Binary scaling leaves NNLS coefficients unchanged.
_MIN_EXPONENT = -450
_MAX_EXPONENT = 450
_MAX_SCALED_DIMENSION = 2**32


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _scipy_rows(design, rows, maxiter):
    result = np.empty((len(rows), design.shape[1]), dtype=np.float64)
    for index, row in enumerate(rows):
        result[index] = nnls(design, row, maxiter=maxiter)[0]
    return result


def _eligible_design(design):
    if design.shape[0] < design.shape[1] or np.any(design < 0):
        return False
    try:
        condition = np.linalg.cond(design)
    except np.linalg.LinAlgError:
        return False
    return bool(np.isfinite(condition) and condition <= _BATCH_CONDITION_LIMIT)


def _passive_solve(gram, cross, passive, factors):
    """Solve shared passive sets together; factor storage is block-local."""
    result = np.zeros_like(cross)
    packed = np.packbits(passive.T, axis=1)
    patterns, first, membership = np.unique(
        packed, axis=0, return_index=True, return_inverse=True
    )
    for group, example in enumerate(first):
        indices = np.flatnonzero(passive[:, example])
        if not indices.size:
            continue
        columns = np.flatnonzero(membership == group)
        key = patterns[group].tobytes()
        factor = factors.get(key)
        if factor is None:
            factor = linalg.cho_factor(gram[np.ix_(indices, indices)], check_finite=False)
            if len(factors) >= _FACTOR_CACHE_LIMIT:
                factors.clear()
            factors[key] = factor
        result[np.ix_(indices, columns)] = linalg.cho_solve(
            factor, cross[np.ix_(indices, columns)], check_finite=False
        )
    return result


def _batched_block(design, targets, gram, maxiter):
    """Return a candidate and per-RHS convergence flags, never certify it here."""
    cross = design.T @ targets
    if not np.isfinite(cross).all():
        raise FloatingPointError("NNLS products overflowed.")
    rank, rhs = cross.shape
    coefficients = np.zeros_like(cross)
    passive = np.zeros((rank, rhs), dtype=bool)
    gradient = -cross
    # Coordinatewise scale, unlike a single RHS scale, does not hide small
    # columns behind a large one. No absolute unit floor is introduced.
    tolerance = 1e-12 * np.maximum(np.abs(cross), np.finfo(float).tiny)
    best_count = np.full(rhs, rank + 1, dtype=int)
    retries = np.full(rhs, 3, dtype=int)
    factors = {}
    for iteration in range(maxiter + 1):
        primal_tolerance = 64 * np.finfo(float).eps * np.maximum(
            np.max(np.abs(coefficients), axis=0), np.finfo(float).tiny
        )
        violations = (passive & (coefficients < -primal_tolerance)) | (
            ~passive & (gradient < -tolerance)
        )
        counts = violations.sum(axis=0)
        pending = np.flatnonzero(counts)
        if not pending.size or iteration == maxiter:
            break
        for column in pending:
            if counts[column] < best_count[column]:
                best_count[column], retries[column] = counts[column], 3
                passive[violations[:, column], column] ^= True
            elif retries[column] > 0:
                retries[column] -= 1
                passive[violations[:, column], column] ^= True
            else:
                passive[np.flatnonzero(violations[:, column])[-1], column] ^= True
        solved = _passive_solve(gram, cross[:, pending], passive[:, pending], factors)
        coefficients[:, pending] = solved
        gradient[:, pending] = gram @ solved - cross[:, pending]
    return coefficients, counts == 0


def _qualified_rows(design, targets, coefficients):
    """Check finite feasible solutions and KKT against ORIGINAL design/targets.

    A residual workspace is limited to one block, never all input rows. The
    tolerance is coordinatewise and bounds matrix-product roundoff separately
    from the 1e-12 stopping tolerance. Negative coefficients are not silently
    clipped: a row with even a small negative value uses the reference solver.
    """
    feasible = np.isfinite(coefficients).all(axis=0) & (coefficients >= 0).all(axis=0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        fitted = design @ coefficients
        residual = fitted - targets
        gradient = design.T @ residual
        cross_scale = design.T @ np.abs(targets)
        roundoff_scale = design.T @ (np.abs(fitted) + np.abs(targets))
        tolerance = (1e-12 * cross_scale
                     + 64 * np.finfo(float).eps * max(design.shape) * roundoff_scale)
        projected = np.where((coefficients > 0) | (gradient < 0), gradient, 0.0)
        finite = (np.isfinite(gradient).all(axis=0)
                  & np.isfinite(tolerance).all(axis=0)
                  & np.isfinite(residual).all(axis=0))
        kkt = (np.abs(projected) <= tolerance).all(axis=0)
        # Underflow of a nonzero nonnegative RHS must not certify a zero
        # solution merely because both products rounded down to zero.
        active_rhs = np.any(targets != 0, axis=0)
        representable = ~active_rhs | np.any(cross_scale >= np.finfo(float).tiny, axis=0)
    return feasible & finite & kkt & representable


def project_nnls(design, rows, *, solver="auto", maxiter=None, batch_size=256):
    """Solve ``min_{z>=0} ||design @ z - row||`` for each input row.

    ``auto`` batches at least 32 rows. ``batched`` requests batching even for
    smaller inputs, but retains the same numerical guards and SciPy fallback.
    Batching is restricted to nonnegative, full-column-rank designs with
    condition <= 1e4 and nonnegative RHS blocks. Small explicit iteration caps
    (< 3 * rank) use SciPy directly so its failure semantics are preserved.
    Failed block solves or row-level KKT checks fall back to public SciPy with
    the original maxiter; reference failures always propagate.

    ``qr`` attempts a call-local economy QR for at least 32 rows and a
    nonnegative design with height >= 2 * rank and condition(R) <= 1e6.
    Reduced systems use public SciPy NNLS and original-space qualification.
    Ineligible calls or failed blocks use the existing ``auto`` path with
    the original iteration cap. A short failed tail can therefore use SciPy
    where a larger ``auto`` call would have treated it as a batched tail.

    Extreme magnitudes use exactly reversible common binary scaling of the
    design and each independent RHS. Rows with the same scale share a solve;
    eligibility never depends on unrelated query rows. Inputs whose nonzero
    exponent span cannot fit the supported range raise FloatingPointError.
    This does not change conditioning or certify arbitrary extreme problems.

    Returns a float64 array of shape (len(rows), design.shape[1]). Inputs are
    never modified. This internal helper also permits signed input via SciPy.
    """
    if not isinstance(solver, str) or solver not in ("auto", "scipy", "batched", "qr"):
        raise ValueError("solver must be 'auto', 'scipy', 'batched', or 'qr'.")
    batch_size = _positive_integer(batch_size, "batch_size")
    if maxiter is not None:
        maxiter = _positive_integer(maxiter, "maxiter")
    if np.iscomplexobj(design) or np.iscomplexobj(rows):
        raise ValueError("NNLS inputs must be real-valued.")
    design = np.asarray(design, dtype=np.float64, order="C")
    rows = np.asarray(rows, dtype=np.float64)
    if design.ndim != 2 or rows.ndim != 2:
        raise ValueError("design and rows must be two-dimensional.")
    if min(design.shape) < 1 or rows.shape[1] != design.shape[0]:
        raise ValueError("NNLS input shapes are incompatible or empty.")
    if not np.isfinite(design).all() or not np.isfinite(rows).all():
        raise ValueError("NNLS inputs must contain only finite values.")
    if not len(rows):
        return np.empty((0, design.shape[1]), dtype=np.float64)
    if max(design.shape + rows.shape) > _MAX_SCALED_DIMENSION:
        raise FloatingPointError("NNLS dimensions exceed the supported numerical range.")
    design_bounds = _magnitude_bounds(design)
    bounds = _combine_bounds(design_bounds, _magnitude_bounds(rows))
    if _within_range(bounds):
        result = _project_unscaled(design, rows, solver, maxiter, batch_size)
    else:
        prepared = _prepare_scale_groups(design, rows, design_bounds)
        result = np.empty((len(rows), design.shape[1]), dtype=np.float64)
        for indices, scaled_design, scaled_rows in prepared:
            result[indices] = _project_unscaled(
                scaled_design, scaled_rows, solver, maxiter, batch_size
            )
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError("NNLS returned nonfinite or negative coefficients.")
    return result


def _magnitude_bounds(values):
    """Inclusive frexp exponents of nonzero finite input values."""
    absolute = np.abs(values)
    maximum = float(np.max(absolute, initial=0.0))
    if maximum == 0:
        return None
    minimum = float(np.min(absolute, where=absolute != 0, initial=np.inf))
    return math.frexp(minimum)[1], math.frexp(maximum)[1]


def _combine_bounds(*bounds):
    present = [bound for bound in bounds if bound is not None]
    if not present:
        return None
    return min(bound[0] for bound in present), max(bound[1] for bound in present)


def _within_range(bounds):
    return bounds is None or (_MIN_EXPONENT <= bounds[0] and bounds[1] <= _MAX_EXPONENT)


def _scale_exponent(bounds):
    if bounds is None or _within_range(bounds):
        return 0
    lower, upper = bounds[1] - _MAX_EXPONENT, bounds[0] - _MIN_EXPONENT
    if lower > upper:
        raise FloatingPointError(
            "NNLS input exponent span exceeds the supported scaling range; "
            "this is not a claim that the mathematical problem is unsolvable."
        )
    return min(max(bounds[1], lower), upper)


def _binary_scale(design, rows, exponent):
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        scaled_design = np.ldexp(design, -exponent)
        scaled_rows = np.ldexp(rows, -exponent)
        if (not np.array_equal(np.ldexp(scaled_design, exponent), design)
                or not np.array_equal(np.ldexp(scaled_rows, exponent), rows)):
            raise FloatingPointError("NNLS binary scaling is not exactly reversible.")
    if not _within_range(_combine_bounds(
            _magnitude_bounds(scaled_design), _magnitude_bounds(scaled_rows))):
        raise FloatingPointError("NNLS scaled inputs exceed the supported range.")
    return scaled_design, scaled_rows


def _prepare_scale_groups(design, rows, design_bounds):
    """Validate all independent RHS groups before any numerical solve."""
    membership = {}
    for index, row in enumerate(rows):
        exponent = _scale_exponent(_combine_bounds(design_bounds, _magnitude_bounds(row)))
        membership.setdefault(exponent, []).append(index)
    prepared = []
    for exponent, members in sorted(membership.items()):
        indices = np.asarray(members, dtype=np.intp)
        if exponent:
            scaled_design, scaled_rows = _binary_scale(design, rows[indices], exponent)
        else:
            scaled_design, scaled_rows = design, rows[indices]
        prepared.append((indices, scaled_design, scaled_rows))
    return prepared


def _project_unscaled(design, rows, solver, maxiter, batch_size):
    """Existing guarded solver on validated, ordinary-scale arrays."""
    if solver == "qr":
        return _project_qr(design, rows, maxiter, batch_size)
    rank = design.shape[1]
    if (solver == "scipy" or not len(rows)
            or (solver == "auto" and len(rows) < _AUTO_MIN_ROWS)
            or (maxiter is not None and maxiter < 3 * rank)
            or not _eligible_design(design)):
        return _scipy_rows(design, rows, maxiter)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gram = design.T @ design
    if not np.isfinite(gram).all() or np.any(np.diag(gram) < np.finfo(float).tiny):
        return _scipy_rows(design, rows, maxiter)
    answer = np.empty((len(rows), rank), dtype=np.float64)
    # An acceleration attempt has a bounded budget even when callers grant
    # SciPy a much larger cap. Exhausted rows receive the original SciPy cap.
    cap = 3 * rank if maxiter is None else min(maxiter, 3 * rank)
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        if np.any(block < 0):
            answer[start:start + len(block)] = _scipy_rows(design, block, maxiter)
            continue
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate, converged = _batched_block(design, block.T, gram, cap)
            qualified = converged & _qualified_rows(design, block.T, candidate)
        except (np.linalg.LinAlgError, FloatingPointError):
            answer[start:start + len(block)] = _scipy_rows(design, block, maxiter)
            continue
        answer[start:start + len(block)] = candidate.T
        if not qualified.all():
            # Advanced indexing writes into the owned output, never input.
            failing = start + np.flatnonzero(~qualified)
            answer[failing] = _scipy_rows(design, block[~qualified], maxiter)
    return answer


def _project_qr(design, rows, maxiter, batch_size):
    """Try orthogonal reduction; retain the current solver as block fallback.

    For A = QR, ||Az-b||^2 = ||Rz-Q.T b||^2 + ||(I-Q Q.T)b||^2.
    Variables and their nonnegativity constraint are unchanged. R and Q.T b
    can be signed, so the reduced problem uses public SciPy, never BPP.
    Factors live only for this validated scale-group call.
    """
    rank = design.shape[1]
    if (len(rows) < _AUTO_MIN_ROWS or design.shape[0] < 2 * rank
            or (maxiter is not None and maxiter < 3 * rank)
            or np.any(design < 0)):
        return _project_unscaled(design, rows, "auto", maxiter, batch_size)
    failures = (np.linalg.LinAlgError, FloatingPointError, RuntimeError)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            q, r = linalg.qr(design, mode="economic", pivoting=False,
                             overwrite_a=False, check_finite=False)
            r = np.ascontiguousarray(r)
            if not np.isfinite(q).all() or not np.isfinite(r).all():
                raise FloatingPointError("NNLS QR factors are nonfinite.")
            condition = float(np.linalg.cond(r))
    except failures:
        return _project_unscaled(design, rows, "auto", maxiter, batch_size)
    if not np.isfinite(condition) or condition > _QR_CONDITION_LIMIT:
        return _project_unscaled(design, rows, "auto", maxiter, batch_size)

    answer = np.empty((len(rows), rank), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        if np.any(block < 0):
            answer[start:start + len(block)] = _project_unscaled(
                design, block, "auto", maxiter, batch_size)
            continue
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                reduced_rows = block @ q
                if not np.isfinite(reduced_rows).all():
                    raise FloatingPointError("NNLS reduced targets are nonfinite.")
            candidate = _scipy_rows(r, reduced_rows, maxiter)
            qualified = _qualified_rows(design, block.T, candidate.T)
        except failures:
            answer[start:start + len(block)] = _project_unscaled(
                design, block, "auto", maxiter, batch_size)
            continue
        if not qualified.all():
            # Retry the whole rejected block, including a possible short tail.
            # Fallback errors propagate with the caller's original cap.
            answer[start:start + len(block)] = _project_unscaled(
                design, block, "auto", maxiter, batch_size)
            continue
        answer[start:start + len(block)] = candidate
    return answer
