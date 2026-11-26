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
import scipy
from scipy import linalg
from scipy.optimize import nnls

from ._aocl import _uses_native_product, nnls_matmul


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
_MAX_QUALIFICATION_VALUE = 2.0**_MAX_EXPONENT


_QR_ORIGINAL_QR = linalg.qr
_QR_ORIGINAL_COND = np.linalg.cond


def _discard_qr_preparation():
    return None


class _QRPrepared(tuple):
    """Immutable process-local preparation; serialization deliberately drops it."""

    __slots__ = ()

    def __new__(cls, design, q, r, context):
        if (type(design) is not np.ndarray or type(q) is not np.ndarray
                or type(r) is not np.ndarray or design.dtype != np.float64
                or q.dtype != np.float64 or r.dtype != np.float64
                or not design.flags.c_contiguous or not r.flags.c_contiguous
                or not (q.flags.f_contiguous or q.flags.c_contiguous)
                or q.shape != design.shape or r.shape != (design.shape[1],) * 2):
            raise ValueError("Unsupported prepared QR storage.")
        if q.strides == (q.shape[1] * 8, 8):
            order = "C"
        elif q.strides == (8, q.shape[0] * 8):
            order = "F"
        else:
            raise ValueError("Unsupported prepared Q strides.")
        return tuple.__new__(cls, (design.shape, design.tobytes(order="C"),
            q.tobytes(order=order), r.tobytes(order="C"), order, context))

    def matches(self, design, context):
        if design.shape != self[0] or context != self[5]:
            return False
        saved = np.frombuffer(self[1], dtype=np.uint64).reshape(self[0])
        return np.array_equal(design.view(np.uint64), saved)

    def factors(self):
        shape, _, q, r, order, _ = self
        return (np.frombuffer(q, dtype=np.float64).reshape(shape, order=order),
                np.frombuffer(r, dtype=np.float64).reshape((shape[1], shape[1])))

    def _transport_payload(self):
        """Export immutable factors for the existing call-scoped transport."""
        shape, basis, q, r, order, context = self
        # Keep already immutable small buffers immutable through pickle.
        # Larger buffers retain the existing Joblib read-only mapping.
        buffers = tuple(value if len(value) <= 1048576 else
                        np.frombuffer(value, dtype=np.uint8) for value in (basis, q, r))
        return (shape,) + buffers + (order, context[2])

    @staticmethod
    def _from_checked_transport(shape, basis, q, r, order, context):
        """Build from fields already checked by the process admission path."""
        return tuple.__new__(_QRPrepared, (shape, basis, q, r, order, context))

    def __reduce__(self):
        return _discard_qr_preparation, ()


class _QRFitCapture:
    __slots__ = ("entry",)

    def __init__(self):
        self.entry = None


def _qr_context():
    # This bounded implementation was checked for silent successful preparation.
    # Future runtimes and replaced entry callables retain current QR behavior.
    if (np.__version__ != "2.5.4" or scipy.__version__ != "1.18.1"
            or linalg.qr is not _QR_ORIGINAL_QR
            or np.linalg.cond is not _QR_ORIGINAL_COND
            or np.geterr() != {"divide": "warn", "over": "warn",
                               "under": "ignore", "invalid": "warn"}
            or np.geterrcall() is not None):
        return None
    from ._projection import _qr_native_context
    return _qr_native_context()


def _qr_lookup(design, prepared, capture, owned):
    """Bind the exact snapshot checked and used by this one solve."""
    if type(prepared) is not _QRPrepared and type(capture) is not _QRFitCapture:
        return design, None, None
    original = design
    try:
        context = _qr_context()
        if context is None or type(design) is not np.ndarray:
            return original, None, None
        if owned is not True:
            design = np.array(design, dtype=np.float64, order="C", copy=True)
        if type(prepared) is _QRPrepared and prepared.matches(design, context):
            return design, prepared.factors(), context
        return design, None, context
    except Exception:
        # Only new optional context/snapshot/storage work is covered here.
        # The required original QR and reference solve are outside this catch.
        return original, None, None


def _qr_capture_positive(capture, design, q, r, context):
    if context is not None and type(capture) is _QRFitCapture:
        try:
            capture.entry = _QRPrepared(design, q, r, context)
        except Exception:
            capture.entry = None


def _prepare_qr_entry(design, *, _prepared=None, _owned=False, _allow_prepare=True):
    """Prepare or reuse parent factors without replacing a required projection."""
    try:
        context = _qr_context()
        if context is None or type(design) is not np.ndarray:
            return None
        if (_owned is True and type(_prepared) is _QRPrepared
                and design.dtype == np.float64 and design.flags.c_contiguous
                and design.flags.owndata and _prepared.matches(design, context)):
            return _prepared
        if _allow_prepare is False:
            return None
        if not (_owned is True and design.dtype == np.float64
                and design.flags.c_contiguous and design.flags.owndata):
            design = np.array(design, dtype=np.float64, order="C", copy=True)
        if (design.ndim != 2 or min(design.shape) < 1
                or design.shape[0] < 2 * design.shape[1]
                or max(design.shape) > _MAX_SCALED_DIMENSION
                or not np.isfinite(design).all() or np.any(design < 0)
                or not _within_range(_magnitude_bounds(design))):
            return None
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            q, r = linalg.qr(design, mode="economic", pivoting=False,
                             overwrite_a=False, check_finite=False)
            r = np.ascontiguousarray(r)
            if not np.isfinite(q).all() or not np.isfinite(r).all():
                return None
            condition = float(np.linalg.cond(r))
        if not np.isfinite(condition) or condition > _QR_CONDITION_LIMIT:
            return None
        return _QRPrepared(design, q, r, context)
    except Exception:
        # Parent preparation is optional. Failure leaves required projection
        # to the original worker route and does not cache a negative result.
        return None



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


def _batched_block(design, targets, gram, maxiter, *, matmul=np.matmul):
    """Return candidate, per-RHS convergence flags and the unchanged cross."""
    cross = matmul(design.T, targets)
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
        if iteration == 0:
            # Every pending column improves on its initial best count, rank + 1.
            best_count[pending] = counts[pending]
            passive[:] = violations
        else:
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
        gradient[:, pending] = matmul(gram, solved) - cross[:, pending]
    return coefficients, counts == 0, cross


def _qualified_rows(design, targets, coefficients, *, matmul=np.matmul, cross_scale=None):
    """Check finite feasible solutions and KKT against ORIGINAL design/targets.

    A residual workspace is limited to one block, never all input rows. The
    tolerance is coordinatewise and bounds matrix-product roundoff separately
    from the 1e-12 stopping tolerance. Negative coefficients are not silently
    clipped: a row with even a small negative value uses the reference solver.
    A supplied cross_scale belongs to this exact admitted nonnegative BPP
    block. Reuse requires the same target layout as the absolute-value product.
    """
    feasible = np.isfinite(coefficients).all(axis=0) & (coefficients >= 0).all(axis=0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        fitted = matmul(design, coefficients)
        residual = fitted - targets
        gradient = matmul(design.T, residual)
        if cross_scale is None or not _matches_absolute_layout(targets):
            cross_scale = matmul(design.T, np.abs(targets))
        # The original test adds a nonnegative roundoff allowance. If every
        # feasible column already passes the smaller tolerance, we only need
        # to establish that the original allowance would have been finite.
        absolute_projected = None
        if (_uses_native_product(matmul, np)
                and type(design) is np.ndarray and type(targets) is np.ndarray
                and type(coefficients) is np.ndarray and type(cross_scale) is np.ndarray
                and design.size and design.dtype == targets.dtype == coefficients.dtype == cross_scale.dtype == np.float64):
            projected = np.where((coefficients > 0) | (gradient < 0), gradient, 0.0)
            lower = 1e-12 * cross_scale
            absolute_projected = np.abs(projected)
            within = (absolute_projected <= lower).all(axis=0)
            # On this domain every feasible-column product is bounded by
            # 2**32 * 2**450 * (2 * 2**450), comfortably inside float64.
            if ((within | ~feasible).all() and np.isfinite(lower).all()
                    and max(design.shape) <= _MAX_SCALED_DIMENSION and design.min() >= 0
                    and design.max() <= _MAX_QUALIFICATION_VALUE and fitted.max(initial=0.) <= _MAX_QUALIFICATION_VALUE
                    and targets.min(initial=0.) >= 0 and targets.max(initial=0.) <= _MAX_QUALIFICATION_VALUE):
                active_rhs = np.any(targets != 0, axis=0)
                return feasible & (~active_rhs | np.any(cross_scale >= np.finfo(float).tiny, axis=0))
        roundoff_input = np.abs(fitted)
        roundoff_input += np.abs(targets)
        roundoff_scale = matmul(design.T, roundoff_input)
        if absolute_projected is None:
            tolerance = (1e-12 * cross_scale
                         + 64 * np.finfo(float).eps * max(design.shape) * roundoff_scale)
            projected = np.where((coefficients > 0) | (gradient < 0), gradient, 0.0)
        else:
            tolerance = lower + 64 * np.finfo(float).eps * max(design.shape) * roundoff_scale
        finite = (np.isfinite(gradient).all(axis=0)
                  & np.isfinite(tolerance).all(axis=0)
                  & np.isfinite(residual).all(axis=0))
        if absolute_projected is None:
            absolute_projected = np.abs(projected)
        kkt = (absolute_projected <= tolerance).all(axis=0)
        # Underflow of a nonzero nonnegative RHS must not certify a zero
        # solution merely because both products rounded down to zero.
        active_rhs = np.any(targets != 0, axis=0)
        representable = ~active_rhs | np.any(cross_scale >= np.finfo(float).tiny, axis=0)
    return feasible & finite & kkt & representable


def _matches_absolute_layout(targets):
    # np.abs makes an aligned, compact array. A differently strided operand
    # can choose another native reduction and change a KKT boundary decision.
    if type(targets) is not np.ndarray or not targets.flags.aligned:
        return False
    height, width = targets.shape
    item = targets.itemsize
    if targets.flags.c_contiguous:
        return targets.strides == (width * item, item)
    return targets.flags.f_contiguous and targets.strides == (item, height * item)


def project_nnls(design, rows, *, solver="auto", maxiter=None, batch_size=256,
                 _qr_prepared=None, _qr_capture=None, _qr_owned=False):
    """Solve ``min_{z>=0} ||design @ z - row||`` for each input row.

    ``auto`` batches at least 32 rows. ``batched`` requests batching even for
    smaller inputs, but retains the same numerical guards and SciPy fallback.
    Batching is restricted to nonnegative, full-column-rank designs with
    condition <= 1e4 and nonnegative RHS blocks. Small explicit iteration caps
    (< 3 * rank) use SciPy directly so its failure semantics are preserved.
    Failed block solves or row-level KKT checks fall back to public SciPy with
    the original maxiter; reference failures always propagate.

    ``qr`` attempts an economy QR for at least 32 rows and a
    nonnegative design with height >= 2 * rank and condition(R) <= 1e6.
    Private prepared factors require fresh basis and context checks.
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
    matmul = nnls_matmul() if solver != "scipy" else np.matmul
    if _within_range(bounds):
        if _qr_prepared is not None or _qr_capture is not None:
            result = _project_unscaled(design, rows, solver, maxiter, batch_size,
                matmul=matmul, _qr_prepared=_qr_prepared,
                _qr_capture=_qr_capture, _qr_owned=_qr_owned)
        else:
            result = _project_unscaled(design, rows, solver, maxiter, batch_size, matmul=matmul)
    else:
        prepared = _prepare_scale_groups(design, rows, design_bounds)
        result = np.empty((len(rows), design.shape[1]), dtype=np.float64)
        for indices, scaled_design, scaled_rows in prepared:
            result[indices] = _project_unscaled(
                scaled_design, scaled_rows, solver, maxiter, batch_size, matmul=matmul
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


def _project_unscaled(design, rows, solver, maxiter, batch_size, *, matmul=np.matmul,
                      _qr_prepared=None, _qr_capture=None, _qr_owned=False):
    """Existing guarded solver on validated, ordinary-scale arrays."""
    if solver == "qr":
        if _qr_prepared is not None or _qr_capture is not None:
            return _project_qr(design, rows, maxiter, batch_size, matmul=matmul,
                _qr_prepared=_qr_prepared, _qr_capture=_qr_capture, _qr_owned=_qr_owned)
        return _project_qr(design, rows, maxiter, batch_size, matmul=matmul)
    rank = design.shape[1]
    if (solver == "scipy" or not len(rows)
            or (solver == "auto" and len(rows) < _AUTO_MIN_ROWS)
            or (maxiter is not None and maxiter < 3 * rank)
            or not _eligible_design(design)):
        return _scipy_rows(design, rows, maxiter)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gram = matmul(design.T, design)
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
            targets = block.T
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate, converged, cross = _batched_block(
                    design, targets, gram, cap, matmul=matmul)
            qualified = converged & _qualified_rows(
                design, targets, candidate, matmul=matmul, cross_scale=cross)
        except (np.linalg.LinAlgError, FloatingPointError):
            answer[start:start + len(block)] = _scipy_rows(design, block, maxiter)
            continue
        answer[start:start + len(block)] = candidate.T
        if not qualified.all():
            # Advanced indexing writes into the owned output, never input.
            failing = start + np.flatnonzero(~qualified)
            answer[failing] = _scipy_rows(design, block[~qualified], maxiter)
    return answer


def _project_qr(design, rows, maxiter, batch_size, *, matmul=np.matmul,
                _qr_prepared=None, _qr_capture=None, _qr_owned=False):
    """Try orthogonal reduction; retain the current solver as block fallback.

    For A = QR, ||Az-b||^2 = ||Rz-Q.T b||^2 + ||(I-Q Q.T)b||^2.
    Variables and their nonnegativity constraint are unchanged. R and Q.T b
    can be signed, so the reduced problem uses public SciPy, never BPP.
    Prepared factors require fresh basis and context checks before reuse.
    """
    rank = design.shape[1]
    if (len(rows) < _AUTO_MIN_ROWS or design.shape[0] < 2 * rank
            or (maxiter is not None and maxiter < 3 * rank)
            or np.any(design < 0)):
        return _project_unscaled(design, rows, "auto", maxiter, batch_size, matmul=matmul)
    cached = context = None
    if _qr_prepared is not None or _qr_capture is not None:
        design, cached, context = _qr_lookup(
            design, _qr_prepared, _qr_capture, _qr_owned)
    failures = (np.linalg.LinAlgError, FloatingPointError, RuntimeError)
    if cached is None:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                q, r = linalg.qr(design, mode="economic", pivoting=False,
                                 overwrite_a=False, check_finite=False)
                r = np.ascontiguousarray(r)
                if not np.isfinite(q).all() or not np.isfinite(r).all():
                    raise FloatingPointError("NNLS QR factors are nonfinite.")
                condition = float(np.linalg.cond(r))
        except failures:
            return _project_unscaled(design, rows, "auto", maxiter, batch_size, matmul=matmul)
        if not np.isfinite(condition) or condition > _QR_CONDITION_LIMIT:
            return _project_unscaled(design, rows, "auto", maxiter, batch_size, matmul=matmul)
    else:
        q, r = cached
    if type(_qr_capture) is _QRFitCapture:
        if cached is not None and type(_qr_prepared) is _QRPrepared:
            _qr_capture.entry = _qr_prepared
        else:
            _qr_capture_positive(_qr_capture, design, q, r, context)

    answer = np.empty((len(rows), rank), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        if np.any(block < 0):
            answer[start:start + len(block)] = _project_unscaled(
                design, block, "auto", maxiter, batch_size, matmul=matmul)
            continue
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                reduced_rows = matmul(block, q)
                if not np.isfinite(reduced_rows).all():
                    raise FloatingPointError("NNLS reduced targets are nonfinite.")
            candidate = _scipy_rows(r, reduced_rows, maxiter)
            qualified = _qualified_rows(design, block.T, candidate.T, matmul=matmul)
        except failures:
            answer[start:start + len(block)] = _project_unscaled(
                design, block, "auto", maxiter, batch_size, matmul=matmul)
            continue
        if not qualified.all():
            # Retry the whole rejected block, including a possible short tail.
            # Fallback errors propagate with the caller's original cap.
            answer[start:start + len(block)] = _project_unscaled(
                design, block, "auto", maxiter, batch_size, matmul=matmul)
            continue
        answer[start:start + len(block)] = candidate
    return answer
