"""Observation-specific MU projection for Chang Cheng's fitted NMF models."""

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ._mu import _standard_mu_basis, transform_mu


def _ordinary_rows(x, h, max_iter, tol):
    """MU updates with independent initializers and residual stopping tests.

    Each row uses its own mean and its own initial residual. A computed zero
    initial residual returns the initializer; it is not a symbolic optimality
    certificate. Matrix-kernel rounding can still depend on the active rows.
    """
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        gram = h @ h.T
        numerator = x @ h.T
        initializers = np.sqrt(np.mean(x, axis=1) / len(h))
        w = np.broadcast_to(initializers[:, None], (len(x), len(h))).copy()
        residual = x - w @ h
        initial = np.sqrt(np.einsum("ij,ij->i", residual, residual))
        previous = initial.copy()
        iterations = np.zeros(len(x), dtype=np.int64)
        active = initial != 0
        # Match the ordinary MU recurrence's treatment of zero denominators.
        epsilon = np.finfo(np.float32).eps
        for iteration in range(1, max_iter + 1):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            current = w[indices]
            denominator = current @ gram
            denominator[denominator == 0] = epsilon
            current *= numerator[indices] / denominator
            if not np.isfinite(current).all() or np.any(current < 0):
                raise FloatingPointError("Rowwise MU produced invalid coordinates.")
            w[indices] = current
            iterations[indices] = iteration
            if tol > 0 and iteration % 10 == 0:
                residual = x[indices] - current @ h
                error = np.sqrt(np.einsum("ij,ij->i", residual, residual))
                stopped = (previous[indices] - error) / initial[indices] < tol
                active[indices[stopped]] = False
                previous[indices] = error
    return w, iterations


def _native_rows(x, h, max_iter, tol):
    """Return ordinary results and mark rows requiring range recovery.

    Marked rows have placeholder zeros and iteration count -1. The caller must
    recover them in the original input dtype before exposing any result.
    """
    output = np.zeros((len(x), len(h)), dtype=x.dtype)
    iterations = np.zeros(len(x), dtype=np.int64)
    nonzero = np.any(x != 0, axis=1)
    limits = np.finfo(x.dtype)
    small = math.sqrt(float(limits.tiny))
    large = math.sqrt(float(limits.max)) / (4 * max(x.shape[1], len(h)))
    risky = ((np.any((x > 0) & (x < small), axis=1)
              | np.any(x > large, axis=1)) & nonzero)
    if np.any((h > 0) & (h < small)) or np.any(h > large):
        risky = nonzero.copy()
    indices = np.flatnonzero(nonzero & ~risky)
    if len(indices):
        try:
            values, counts = _ordinary_rows(x[indices], h, max_iter, tol)
            output[indices], iterations[indices] = values, counts
        except FloatingPointError:
            # One difficult row must not send all others into range recovery.
            for index in indices:
                try:
                    values, counts = _ordinary_rows(x[index:index + 1], h, max_iter, tol)
                    output[index], iterations[index] = values[0], counts[0]
                except FloatingPointError:
                    risky[index] = True
    iterations[risky] = -1
    return output, iterations, risky


def _recover_rows(decomposition, x, output, recovery):
    """Keep the established range and rounding behavior for each original row."""
    for index in np.flatnonzero(recovery):
        # Numerical warnings and nonnumerical errors remain visible.
        output[index] = transform_mu(decomposition, x[index:index + 1])[0]


def transform_mu_rowwise(decomposition, selected):
    """Project a standard fitted MU model with a separate stop for each row.

    The input must be finite, nonnegative, and have the fitted basis's dtype.
    The classifier performs any mixed-dtype widening before this helper.
    Ordinary float32 projections use float64 working arithmetic and return
    float32. Tiny query rows, subnormal or out-of-range outputs and failed
    double arithmetic retain the original per-row range recovery. Tiny
    float32 basis tails alone do not force ordinary queries into serial solves.

    This changes the finite-tolerance encoding, not the fixed-basis Frobenius
    objective. It removes shared stopping, not all floating-point dependence
    on matrix shape. No inputs or fitted-model arrays are modified.
    """
    h = _standard_mu_basis(decomposition, selected)
    if h is None:
        raise ValueError(
            "mu_projection='rowwise' requires a standard fitted, unregularized "
            "Frobenius MU model with matching float32 or float64 input and basis."
        )
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Rowwise MU input must be finite and nonnegative.")
    if not np.any(selected):
        return np.zeros((len(selected), len(h)), dtype=selected.dtype)
    max_iter, tol = decomposition.max_iter, decomposition.tol
    if selected.dtype == np.float64:
        output, iterations, recovery = _native_rows(selected, h, max_iter, tol)
        _recover_rows(decomposition, selected, output, recovery)
    else:
        limits = np.finfo(np.float32)
        small = math.sqrt(float(limits.tiny))
        large = math.sqrt(float(limits.max)) / (4 * max(selected.shape[1], len(h)))
        original = np.any((selected > 0) & (selected < small), axis=1)
        # Rank-one range recovery preserves exact subnormal rounding ties.
        if len(h) == 1 and (np.any((h > 0) & (h < small)) or np.any(h > large)):
            original[:] = True
        output = np.zeros((len(selected), len(h)), dtype=np.float32)
        iterations = np.zeros(len(selected), dtype=np.int64)
        indices = np.flatnonzero(~original)
        if len(indices):
            wide, counts, recovery = _native_rows(
                selected[indices].astype(np.float64), h.astype(np.float64), max_iter, tol)
            retry = (recovery | ~np.isfinite(wide).all(axis=1)
                     | np.any(wide < 0, axis=1) | np.any(wide > limits.max, axis=1)
                     | np.any((wide > 0) & (wide < limits.tiny), axis=1))
            good = indices[~retry]
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                output[good] = wide[~retry].astype(np.float32)
            iterations[good] = counts[~retry]
            original[indices[retry]] = True
        for index in np.flatnonzero(original):
            row = selected[index:index + 1]
            values, counts, recovery = _native_rows(row, h, max_iter, tol)
            _recover_rows(decomposition, row, values, recovery)
            output[index], iterations[index] = values[0], counts[0]
    capped = int(np.sum(iterations == max_iter))
    if tol > 0 and capped:
        warnings.warn(
            f"Maximum number of iterations {max_iter} reached for {capped} rows "
            "during rowwise MU projection. Increase it to improve convergence.",
            ConvergenceWarning, stacklevel=2)
    return output
