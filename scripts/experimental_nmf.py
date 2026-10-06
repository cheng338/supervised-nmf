"""Experimental ANLS-BPP comparator for Chang Cheng's supervised NMF pipeline.

This is not a change to the package's CD/SPA implementation or default.
Algorithm: Jingu Kim and Haesun Park, "Fast Nonnegative Matrix Factorization:
An Active-Set-Like Method and Comparisons", SIAM J. Scientific Computing
33(6), 3261--3281 (2011). The block-pivot/three-retry/single-pivot strategy
follows their public reference implementation, with new validation, grouped
rank-deficient least-squares handling, scale-aware tolerances, sklearn API,
and explicit convergence diagnostics. Reference:
https://github.com/kimjingu/nonnegfac-python/blob/master/nonnegfac/nnls.py

Third-party reference notice (pipeline/research authorship remains Chang Cheng):
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

from numbers import Integral, Real
import warnings

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition._nmf import _initialize_nmf
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array, check_is_fitted


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _tolerance(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return float(value)


def _array(value, *, nonnegative=False):
    result = check_array(value, dtype=np.float64, ensure_2d=True, order="C")
    if nonnegative and np.any(result < 0):
        raise ValueError("NMF requires nonnegative input.")
    return result


def _pg_values(W, H, grad_W, grad_H):
    projected_W = np.where((W > 0) | (grad_W < 0), grad_W, 0.0)
    projected_H = np.where((H > 0) | (grad_H < 0), grad_H, 0.0)
    pg = float(np.hypot(linalg.norm(projected_W), linalg.norm(projected_H)))
    total = float(np.hypot(linalg.norm(grad_W), linalg.norm(grad_H)))
    return {"projected_gradient_norm": pg, "gradient_norm": total,
            "relative_projected_gradient": pg / total if total else 0.0,
            "primal_violation": float(max(0.0, -W.min(), -H.min()))}


def projected_gradient_kkt(X, W, H):
    """Descriptive full-NMF KKT diagnostics, including a canonical factor copy.

    Objective is .5*||X-WH||_F**2, without penalties. The projected gradient
    retains a gradient entry when its variable is positive or gradient negative.
    Relative values divide by the full gradient norm at the SAME factors, not
    the initial gradient used by a solver. This is not a shared stopping rule.
    Raw gradients depend on reciprocal W/H scaling. Canonical diagnostics use
    unit-L2 rows of H with compensating W columns; zero H rows are left alone.
    Inputs and classifier geometry are never modified. SPA does not optimize
    unrestricted joint NMF, so these values are NOT SPA eligibility criteria.
    """
    X, W, H = _array(X, nonnegative=True), _array(W), _array(H)
    if W.shape[0] != X.shape[0] or H.shape[1] != X.shape[1] or W.shape[1] != H.shape[0]:
        raise ValueError("Incompatible X, W and H shapes.")
    residual = W @ H - X
    raw = _pg_values(W, H, residual @ H.T, W.T @ residual)
    scales = linalg.norm(H, axis=1)
    scales = np.where(scales > 0, scales, 1.0)
    canonical_W, canonical_H = W * scales, H / scales[:, None]
    canonical = _pg_values(canonical_W, canonical_H,
                           residual @ canonical_H.T, canonical_W.T @ residual)
    error, norm_X = float(linalg.norm(residual)), float(linalg.norm(X))
    return {"raw": raw, "canonical": canonical, "reconstruction_error": error,
            "relative_reconstruction_error": error / norm_X if norm_X else (0.0 if error == 0 else None),
            "objective": 0.5 * error ** 2, "dtype": "float64",
            "convention": "full unregularized Frobenius NMF; descriptive only; canonical H row L2=1 on copies"}


def _minimum_norm_lstsq(A, B):
    """Original-design minimum-norm LS via matched F2PY DGELSD wrappers.

    SciPy 1.18's batched ``linalg.lstsq(..., lapack_driver='gelsd')``
    dispatcher was the recorded location of a reproducible Windows/AOCL
    access violation; direct DGELSD passed replay of the captured inputs.
    Calling DGELSD directly preserves its SVD algorithm and float64-epsilon
    rank cutoff while bypassing that dispatcher. This also follows SciPy's
    pre-1.18 dispatch structure. No ridge, normal-equation pseudoinverse,
    changed rank threshold, or alternate least-squares driver is used.

    The owned Fortran copies prevent LAPACK overwrites of training arrays;
    B is zero-padded to max(m,n) for underdetermined designs. Query and solve
    use the SAME get_lapack_funcs dispatch, including its integer ABI.
    """
    A, B = _array(A), _array(B)
    m, n = A.shape
    if B.shape[0] != m:
        raise ValueError("Least-squares A and B must have equal row counts.")
    design = np.array(A, dtype=np.float64, order="F", copy=True)
    target = np.zeros((max(m, n), B.shape[1]), dtype=np.float64, order="F")
    target[:m] = B
    solve, query = get_lapack_funcs(("gelsd", "gelsd_lwork"), (design, target))
    cutoff = np.finfo(np.float64).eps
    work, integer_work, info = query(m, n, B.shape[1], cond=cutoff)
    if info != 0:
        raise linalg.LinAlgError(f"DGELSD workspace query failed with info={info}.")
    if (not np.isfinite(work) or work < 1 or not np.isfinite(integer_work)
            or integer_work < 1 or int(integer_work) != integer_work):
        raise linalg.LinAlgError("DGELSD returned an invalid workspace size.")
    work_size = int(np.ceil(work))
    integer_dtype = getattr(solve, "int_dtype", None)
    if integer_dtype is not None:
        integer_limit = np.iinfo(integer_dtype).max
        if max(m, n, B.shape[1], work_size, int(integer_work)) > integer_limit:
            raise linalg.LinAlgError("DGELSD dimensions/workspaces exceed its LAPACK integer range.")
    solution, singular_values, rank, info = solve(
        design, target, work_size, int(integer_work), cond=cutoff,
        overwrite_a=1, overwrite_b=1)
    if info < 0:
        raise linalg.LinAlgError(f"DGELSD rejected argument {-info}.")
    if info > 0:
        raise linalg.LinAlgError("DGELSD singular-value decomposition did not converge.")
    answer = solution[:n].copy()
    if (answer.shape != (n, B.shape[1]) or not np.isfinite(answer).all()
            or not np.isfinite(singular_values).all() or not 0 <= rank <= min(m, n)):
        raise linalg.LinAlgError("DGELSD returned invalid solution diagnostics.")
    return answer


def _passive_solve(A, B, gram, cross, passive):
    """One solve per shared passive pattern; no ridge or alternate NNLS solver."""
    result = np.zeros_like(cross)
    packed = np.packbits(passive.T, axis=1)
    _, first, membership = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    direct_solves = 0
    for group, example in enumerate(first):
        indices = np.flatnonzero(passive[:, example])
        if not indices.size:
            continue
        columns = np.flatnonzero(membership == group)
        subsystem = gram[np.ix_(indices, indices)]
        # Squaring the design condition in normal equations is unsafe for a
        # nearly singular passive set. Solve original-design least squares,
        # which also gives a deterministic minimum-norm singular solution.
        condition = np.linalg.cond(subsystem)
        if not np.isfinite(condition) or condition > 1e10:
            solution = _minimum_norm_lstsq(A[:, indices], B[:, columns])
            direct_solves += 1
        else:
            solution = linalg.solve(subsystem, cross[np.ix_(indices, columns)],
                                    assume_a="pos", check_finite=False)
        result[np.ix_(indices, columns)] = solution
    return result, int(len(first)), direct_solves


def nnls_block_pivot(A, B, *, init=None, max_iter=None, tol=1e-10):
    """Solve min_X>=0 .5*||A X-B||_F**2 by grouped block principal pivoting.

    Dense A and multiple-column B may be signed. Returns (X, info), where info
    includes success, iterations, pivot counts and independently recomputed
    original-design KKT residuals. Degenerate solutions need not be unique.
    Default inner cap is max(100,5*n_columns_A); it is NOT the NMF outer cap.
    Warm starts specify the initial passive set, not an unverified solution.
    On numerical nonconvergence a feasible clipped candidate and success=False
    are returned WITH ConvergenceWarning. No alternate NNLS algorithm is used.
    """
    A, B = _array(A), _array(B)
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have equal row counts.")
    tol = _tolerance(tol, "tol")
    count, rhs = A.shape[1], B.shape[1]
    cap = max(100, 5 * count) if max_iter is None else _positive_integer(max_iter, "max_iter")
    gram, cross = A.T @ A, A.T @ B
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise ValueError("NNLS products overflowed; rescale the input.")
    passive = np.zeros((count, rhs), dtype=bool)
    solves = direct_solves = 0
    if init is not None:
        initial = _array(init, nonnegative=True)
        if initial.shape != cross.shape:
            raise ValueError("init has the wrong coefficient shape.")
        passive = initial > 0
        coefficients, solves, direct_solves = _passive_solve(A, B, gram, cross, passive)
    else:
        coefficients = np.zeros_like(cross)
    gradient = gram @ coefficients - cross
    # Relative to the zero-solution gradient for each RHS; no arbitrary unit
    # floor, so rescaling observations does not silently change qualification.
    gradient_scale = np.maximum(np.max(np.abs(cross), axis=0), np.finfo(float).tiny)
    gradient_tolerance = tol * gradient_scale
    best_count = np.full(rhs, count + 1, dtype=int)
    retries = np.full(rhs, 3, dtype=int)
    backups = iterations = 0
    for iteration in range(cap + 1):
        coefficient_scale = np.maximum(np.max(np.abs(coefficients), axis=0), np.finfo(float).tiny)
        coefficient_tolerance = 64 * np.finfo(float).eps * coefficient_scale
        bad_primal = passive & (coefficients < -coefficient_tolerance)
        bad_dual = ~passive & (gradient < -gradient_tolerance)
        violation = bad_primal | bad_dual
        counts = violation.sum(axis=0)
        pending = np.flatnonzero(counts)
        if not pending.size or iteration == cap:
            break
        for column in pending:
            if counts[column] < best_count[column]:
                best_count[column], retries[column] = counts[column], 3
                passive[violation[:, column], column] ^= True
            elif retries[column] > 0:
                retries[column] -= 1
                passive[violation[:, column], column] ^= True
            else:
                index = np.flatnonzero(violation[:, column])[-1]
                passive[index, column] ^= True
                backups += 1
        solved, groups, direct = _passive_solve(A, B[:, pending], gram,
                                               cross[:, pending], passive[:, pending])
        coefficients[:, pending] = solved
        solves += groups
        direct_solves += direct
        gradient[:, pending] = gram @ solved - cross[:, pending]
        iterations = iteration + 1
    # Check the feasible returned candidate against ORIGINAL A/B, not just the
    # potentially cancellation-prone normal equations or passive-set status.
    unclipped_primal = float(max(0.0, -coefficients.min()))
    coefficients = np.maximum(coefficients, 0.0)
    residual = A @ coefficients - B
    gradient = A.T @ residual
    projected = np.where((coefficients > 0) | (gradient < 0), gradient, 0.0)
    relative_by_rhs = np.max(np.abs(projected), axis=0) / gradient_scale
    relative = float(relative_by_rhs.max())
    # A small roundoff allowance is separate from user tolerance.
    allowance = tol + 256 * np.finfo(float).eps * max(A.shape)
    success = bool(np.isfinite(coefficients).all() and np.isfinite(relative)
                   and not pending.size and relative <= allowance)
    info = {"success": success, "n_iter": iterations, "max_iter": cap,
            "grouped_solves": solves, "direct_lstsq_groups": direct_solves,
            "single_pivots": backups, "projected_gradient_relative": relative,
            "kkt_tolerance": allowance, "unclipped_primal_violation": unclipped_primal,
            "primal_violation": 0.0, "dual_violation": float(max(0.0, -gradient.min())),
            "complementarity_max": float(np.max(np.abs(coefficients * gradient))),
            "residual_norm": float(linalg.norm(residual))}
    if not success:
        warnings.warn("ANLS-BPP inner NNLS failed its iteration/KKT qualification; "
                      f"iterations={iterations}/{cap}, relative projected gradient={relative:.3g}.",
                      ConvergenceWarning, stacklevel=2)
    return coefficients, info


class ANLSBPP(TransformerMixin, BaseEstimator):
    """Experimental dense, unregularized Frobenius NMF using ANLS-BPP.

    Float64 NNDSVDa initialization uses the installed sklearn initializer (its
    version is recorded by the study). No factor normalization is performed.
    ``max_iter`` counts OUTER alternating NNLS sweeps. Each inner solve has its
    own ``nnls_max_iter`` and stricter KKT qualification. Outer convergence is
    projected-gradient norm <= tol times its initial norm, not MU's objective
    improvement rule. All failure/cap paths warn or raise; no silent fallback.
    The common external diagnostic is descriptive, not this stopping criterion.
    """

    def __init__(self, n_components=2, *, max_iter=1000, tol=1e-4,
                 random_state=0, nnls_max_iter=None, nnls_tol=1e-10):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.nnls_max_iter = nnls_max_iter
        self.nnls_tol = nnls_tol

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)
        if fit_params:
            raise TypeError("ANLSBPP does not accept extra fit parameters.")
        X = _array(X, nonnegative=True)
        rank = _positive_integer(self.n_components, "n_components")
        if rank > min(X.shape):
            raise ValueError("n_components exceeds min(n_samples, n_features).")
        cap = _positive_integer(self.max_iter, "max_iter")
        tol = _tolerance(self.tol, "tol", allow_zero=True)
        _tolerance(self.nnls_tol, "nnls_tol")
        if self.nnls_max_iter is not None:
            _positive_integer(self.nnls_max_iter, "nnls_max_iter")
        W, H = _initialize_nmf(X, rank, init="nndsvda", random_state=self.random_state)
        grad_W = W @ (H @ H.T) - X @ H.T
        grad_H = (W.T @ W) @ H - W.T @ X
        initial_pg = _pg_values(W, H, grad_W, grad_H)["projected_gradient_norm"]
        initial_error = float(linalg.norm(X - W @ H))
        history = [0.5 * initial_error ** 2]
        pg_history = [initial_pg]
        inner_summary = {"solves": 0, "pivot_iterations": 0, "direct_lstsq_groups": 0,
                         "single_pivots": 0, "failed_solves": 0,
                         "maximum_relative_projected_gradient": 0.0}
        converged = initial_pg == 0
        reason = "initial_stationarity" if converged else "outer_iteration_limit"
        iterations = 0
        for iteration in range(cap):
            if converged:
                break
            next_W, info_W = nnls_block_pivot(H.T, X.T, init=W.T,
                                              max_iter=self.nnls_max_iter, tol=self.nnls_tol)
            if info_W["success"]:
                next_H, info_H = nnls_block_pivot(next_W.T, X, init=H,
                                                  max_iter=self.nnls_max_iter, tol=self.nnls_tol)
                records = (info_W, info_H)
            else:
                info_H = {"success": False}
                records = (info_W,)
            for record in records:
                inner_summary["solves"] += 1
                inner_summary["pivot_iterations"] += record["n_iter"]
                inner_summary["direct_lstsq_groups"] += record["direct_lstsq_groups"]
                inner_summary["single_pivots"] += record["single_pivots"]
                inner_summary["failed_solves"] += int(not record["success"])
                inner_summary["maximum_relative_projected_gradient"] = max(
                    inner_summary["maximum_relative_projected_gradient"], record["projected_gradient_relative"])
            if not info_W["success"] or not info_H["success"]:
                reason = "inner_nnls_failure"
                break
            new_W, new_H = next_W.T, next_H
            residual = new_W @ new_H - X
            objective = 0.5 * float(linalg.norm(residual)) ** 2
            monotonic_allowance = 256 * np.finfo(float).eps * max(1.0, history[0])
            if not np.isfinite(objective) or objective > history[-1] + monotonic_allowance:
                reason = "objective_increase_or_nonfinite"
                break
            W, H = new_W, new_H
            iterations = iteration + 1
            history.append(objective)
            current_pg = _pg_values(W, H, residual @ H.T, W.T @ residual)["projected_gradient_norm"]
            pg_history.append(current_pg)
            converged = current_pg <= tol * initial_pg
            if converged:
                reason = "relative_projected_gradient"
        self.n_features_in_, self.n_components_ = X.shape[1], rank
        self.components_, self.training_coordinates_ = H.copy(), W.copy()
        self.n_iter_, self.converged_ = iterations, bool(converged)
        self.convergence_reason_ = reason
        self.objective_history_, self.projected_gradient_history_ = history, pg_history
        self.initial_projected_gradient_ = initial_pg
        self.relative_projected_gradient_ = pg_history[-1] / initial_pg if initial_pg else 0.0
        self.reconstruction_err_ = float(linalg.norm(X - W @ H))
        self.nnls_diagnostics_, self.computation_dtype_ = inner_summary, "float64"
        if not converged:
            warnings.warn(f"ANLS-BPP did not converge: {reason}; outer sweeps={iterations}/{cap}, "
                          f"relative initial projected gradient={self.relative_projected_gradient_:.3g}.",
                          ConvergenceWarning, stacklevel=2)
        return W.copy()

    def transform(self, X):
        check_is_fitted(self, "components_")
        self.last_transform_converged_ = False
        X = _array(X, nonnegative=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Transform feature count does not match the fitted basis.")
        coefficients, info = nnls_block_pivot(self.components_.T, X.T,
                                              max_iter=self.nnls_max_iter, tol=self.nnls_tol)
        self.last_transform_converged_ = bool(info["success"])
        self.last_transform_diagnostics_ = info
        return coefficients.T

