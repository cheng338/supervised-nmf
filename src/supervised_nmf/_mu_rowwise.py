"""Observation-specific MU projection for Chang Cheng's fitted NMF models."""

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ._mu import (_CappedPrefix, _SuffixSetup, _standard_mu_basis,
                  _try_log_row_recovery, transform_mu)
from ._mu_dispatch import QUALIFIED_VERSION, stock_dot

if QUALIFIED_VERSION:
    from ._mu_updates import check_stop, recover_update, retry_matmul


_SUFFIX_CACHE_TRANSFORM = transform_mu
_SUFFIX_SETUP_TRY = _try_log_row_recovery


def _direct_row_errors(x, w, h):
    residual = x - w @ h
    return np.sqrt(np.einsum("ij,ij->i", residual, residual))


def _row_denominator(current, gram, numerator, tiny, epsilon, flagged, iteration, max_iter, tol):
    """Repair active tiny products before the ordinary zero substitution."""
    if tiny is None:
        denominator = current @ gram
        denominator[denominator == 0] = epsilon
        return denominator, False
    try:
        denominator = current @ gram
    except FloatingPointError as error:
        # A later first exception can affect rows already stopped. Retain
        # ordered per-row recovery in that case and preserve cap warnings.
        if tol > 0 and (max_iter <= 10 or (not flagged and iteration > 10)):
            raise
        return retry_matmul(current, gram, numerator, tiny, error), True
    small = denominator < tiny
    if small.any():
        recover_update(current, numerator, gram, denominator, small)
        denominator[denominator == 0] = epsilon
    return denominator, flagged


def _gram_range(values):
    # This deliberately excludes extreme/subnormal arithmetic. Positive products,
    # their sums, and possible direct residual squares remain in ordinary range
    # under the dimensions admitted below. Zero entries are allowed.
    return (np.all(values <= 2.0 ** 200)
            and np.all((values == 0) | (values >= 2.0 ** -200)))


def _gram_error_interval(current, numerator, gram, xnorm, features):
    """Conservative experimental envelope; direct residual remains the fallback.

    The envelope accounts for positive dot-product accumulation and cancellation
    in the quadratic residual, plus direct-residual rounding. Its constant is
    intentionally generous. Qualification must compare actual stop counts and
    outputs: this prototype is not a formal guarantee for every BLAS reduction.
    """
    cross = np.einsum("ij,ij->i", current, numerator)
    quadratic = np.einsum("ij,ij->i", current @ gram, current)
    scale = xnorm + 2 * cross + quadratic
    squared = xnorm - 2 * cross + quadratic
    factor = 64 * np.finfo(np.float64).eps * (features + 2 * current.shape[1] + 8)
    bound = factor * scale
    # A zero lower bound still permits a decision if the full interval clears
    # the stopping threshold. A nonpositive squared upper bound requires a direct residual.
    squared_high = squared + bound
    valid = squared_high > 0
    direct_rounding = factor * np.sqrt(scale)
    low = np.maximum(0, np.sqrt(np.maximum(squared - bound, 0)) - direct_rounding)
    high = np.sqrt(np.maximum(squared_high, 0)) + direct_rounding
    return low, high, valid


def _ordinary_rows_cached(x, h, max_iter, tol):
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
        tiny = np.finfo(x.dtype).tiny if QUALIFIED_VERSION and stock_dot() else None
        flagged = False
        for block_start in range(1, max_iter + 1, 10):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            # Membership changes only at the ten-update stopping check. Reuse
            # these gathered rows until then, without changing the MU updates.
            current = w[indices]
            active_numerator = numerator[indices]
            block_end = min(block_start + 9, max_iter)
            for iteration in range(block_start, block_end + 1):
                denominator, flagged = _row_denominator(
                    current, gram, active_numerator, tiny, epsilon, flagged, iteration, max_iter, tol)
                current *= active_numerator / denominator
                if not np.isfinite(current).all() or np.any(current < 0):
                    raise FloatingPointError("Rowwise MU produced invalid coordinates.")
            w[indices] = current
            iterations[indices] = block_end
            if tol > 0 and block_end % 10 == 0:
                residual = x[indices] - current @ h
                error = np.sqrt(np.einsum("ij,ij->i", residual, residual))
                ratio = (previous[indices] - error) / initial[indices]
                if flagged:
                    check_stop(previous[indices], error, initial[indices], ratio, tol, x[indices], len(h))
                stopped = ratio < tol
                active[indices[stopped]] = False
                previous[indices] = error
    if flagged and tol > 0 and np.any(iterations == max_iter):
        raise FloatingPointError("Retain capped rowwise warning order")
    return w, iterations


def _ordinary_rows_gram(x, h, max_iter, tol):
    """Cached MU updates with a guarded experimental Gram stopping shortcut.

    Initial norms and uncertain stopping decisions use the existing direct
    residual. The previous entire active block is retained so uncertainty can
    reconstruct both sides of the previous/current error comparison with the
    corresponding matrix shapes. Updates and independent stopping policy remain
    unchanged; the shortcut's floating-point qualification is empirical.
    """
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        gram = h @ h.T
        numerator = x @ h.T
        initializers = np.sqrt(np.mean(x, axis=1) / len(h))
        w = np.broadcast_to(initializers[:, None], (len(x), len(h))).copy()
        initial = _direct_row_errors(x, w, h)
        previous = initial.copy()
        previous_low = initial.copy()
        previous_high = initial.copy()
        previous_indices = previous_coordinates = None
        iterations = np.zeros(len(x), dtype=np.int64)
        active = initial != 0
        epsilon = np.finfo(np.float32).eps
        tiny = np.finfo(x.dtype).tiny if QUALIFIED_VERSION and stock_dot() else None
        flagged = False
        # Small/narrow, non-double and extreme domains retain the direct path.
        eligible = (tol > 0 and x.dtype == np.float64 and h.dtype == np.float64
                    and 64 <= x.shape[1] <= 2 ** 20 and len(h) <= 2 ** 12
                    and _gram_range(x) and _gram_range(h))
        xnorm = np.einsum("ij,ij->i", x, x) if eligible else None
        for block_start in range(1, max_iter + 1, 10):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            current = w[indices]
            active_numerator = numerator[indices]
            block_end = min(block_start + 9, max_iter)
            for iteration in range(block_start, block_end + 1):
                denominator, flagged = _row_denominator(
                    current, gram, active_numerator, tiny, epsilon, flagged, iteration, max_iter, tol)
                current *= active_numerator / denominator
                if not np.isfinite(current).all() or np.any(current < 0):
                    raise FloatingPointError("Rowwise MU produced invalid coordinates.")
            w[indices] = current
            iterations[indices] = block_end
            if tol > 0 and block_end % 10 == 0:
                fallback = True
                if eligible and not flagged and _gram_range(current):
                    try:
                        low, high, valid = _gram_error_interval(
                            current, active_numerator, gram, xnorm[indices], x.shape[1])
                        threshold = tol * initial[indices]
                        rounding = (8 * np.finfo(np.float64).eps
                                    * (previous_high[indices] + high + threshold))
                        # Enclose the direct comparison including its subtract/divide
                        # rounding. Ambiguity falls back for the entire active block.
                        stopped = previous_high[indices] - low < threshold - rounding
                        continued = previous_low[indices] - high > threshold + rounding
                        # For a continuing row whose interval reaches zero, store a direct error
                        # if its upper endpoint is still near or above the stopping threshold.
                        usable = (low > 0) | (high < threshold - rounding)
                        fallback = not np.all(valid & (stopped | (continued & usable)))
                    except FloatingPointError:
                        # The shortcut introduces arithmetic such as tol*initial.
                        # Its range failure must not change the established solver
                        # recovery path or narrow the valid finite-tol domain.
                        fallback = True
                if fallback:
                    error = _direct_row_errors(x[indices], current, h)
                    if np.any(previous_low[indices] != previous_high[indices]):
                        # Restore the previous error using its original active
                        # matrix shape, rather than a new subset-shaped GEMM.
                        old_error = _direct_row_errors(
                            x[previous_indices], previous_coordinates, h)
                        previous[indices] = old_error[np.searchsorted(previous_indices, indices)]
                    ratio = (previous[indices] - error) / initial[indices]
                    if flagged:
                        check_stop(previous[indices], error, initial[indices], ratio, tol, x[indices], len(h))
                    stopped = ratio < tol
                    low = high = error
                else:
                    error = (low + high) / 2
                active[indices[stopped]] = False
                previous[indices] = error
                previous_low[indices], previous_high[indices] = low, high
                previous_indices, previous_coordinates = indices, current.copy()
    if flagged and tol > 0 and np.any(iterations == max_iter):
        raise FloatingPointError("Retain capped rowwise warning order")
    return w, iterations


def _ordinary_rows_single(x, h, max_iter, tol):
    """The ordinary one-row recurrence without active-row gather/scatter work."""
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        gram = h @ h.T
        numerator = x @ h.T
        initializers = np.sqrt(np.mean(x, axis=1) / len(h))
        current = np.broadcast_to(initializers[:, None], (1, len(h))).copy()
        residual = x - current @ h
        initial = np.sqrt(np.einsum("ij,ij->i", residual, residual))
        previous = initial.copy()
        if initial[0] == 0:
            return current, np.zeros(1, dtype=np.int64)
        epsilon = np.finfo(np.float32).eps
        tiny = np.finfo(x.dtype).tiny if QUALIFIED_VERSION and stock_dot() else None
        flagged = False
        for block_start in range(1, max_iter + 1, 10):
            block_end = min(block_start + 9, max_iter)
            for iteration in range(block_start, block_end + 1):
                denominator, flagged = _row_denominator(
                    current, gram, numerator, tiny, epsilon, flagged, iteration, max_iter, tol)
                current *= numerator / denominator
                if not np.isfinite(current).all() or np.any(current < 0):
                    raise FloatingPointError("Rowwise MU produced invalid coordinates.")
            if tol > 0 and block_end % 10 == 0:
                residual = x - current @ h
                error = np.sqrt(np.einsum("ij,ij->i", residual, residual))
                ratio = (previous - error) / initial
                if flagged:
                    check_stop(previous, error, initial, ratio, tol, x, len(h))
                stopped = ratio < tol
                if stopped[0]:
                    break
                previous = error
    if flagged and tol > 0 and block_end == max_iter:
        raise FloatingPointError("Retain capped rowwise warning order")
    return current, np.full(1, block_end, dtype=np.int64)



def _ordinary_rows(x, h, max_iter, tol):
    """Dispatch once; provisional size gates do not change numerical admission."""
    if len(x) == 1:
        return _ordinary_rows_single(x, h, max_iter, tol)
    if (len(x) >= 128 and 512 <= x.shape[1] <= 2 ** 20 and len(h) <= 64
            and tol > 0 and x.dtype == np.float64 and h.dtype == np.float64
            and _gram_range(x) and _gram_range(h)):
        # Retain the reviewed Gram implementation's own eligibility and direct
        # fallback checks. No recurrence state crosses between the two loops.
        return _ordinary_rows_gram(x, h, max_iter, tol)
    return _ordinary_rows_cached(x, h, max_iter, tol)

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
    indices = np.flatnonzero(recovery)
    setup = None
    if x.dtype == np.float64 and len(indices) >= 2:
        # Resolve the original callable before evaluating its arguments.
        attempt = _try_log_row_recovery
        selected = x[indices]
        basis = decomposition.components_
        max_iter, tol = decomposition.max_iter, decomposition.tol
        if (attempt is _SUFFIX_SETUP_TRY
                and _try_log_row_recovery is _SUFFIX_SETUP_TRY
                and transform_mu is _SUFFIX_CACHE_TRANSFORM
                and type(x) is np.ndarray and type(selected) is np.ndarray
                and type(basis) is np.ndarray and basis.ndim == 2
                and basis.dtype == np.float64
                and type(output) is np.ndarray and output.dtype == x.dtype
                and output.shape == (len(x), len(basis))):
            values = attempt(
                selected, basis, max_iter, tol, _retain_capped_prefix=True,
                _return_suffix_setup=True)
        else:
            values = attempt(
                selected, basis, max_iter, tol, _retain_capped_prefix=True)
        del attempt, selected, basis, max_iter, tol
        if type(values) is _SuffixSetup:
            setup = values.cache
            values = values.result
        if isinstance(values, _CappedPrefix):
            # A validated silent prefix remains private. The first capped row
            # and every later recovery row use the original callback-aware path.
            output[indices[:values.count]] = values.coordinates
            indices = indices[values.count:]
        elif values is not None:
            output[indices] = values
            return
    # Accepted-vector and standalone singleton calls allocate no context.
    # A genuine vector cap may hand off setup even for its final one-row
    # suffix; scalar eligibility and warning/native invalidation stay fresh.
    cache = setup
    setup = None
    if cache is None and x.dtype == np.float64 and len(indices) >= 2:
        try:
            cache = {}
        except MemoryError:
            pass
    for index in indices:
        # Preserve two-argument replacements installed by callbacks.
        if cache is not None and transform_mu is _SUFFIX_CACHE_TRANSFORM:
            output[index] = transform_mu(
                decomposition, x[index:index + 1], _basis_cache=cache)[0]
        else:
            if cache is not None:
                cache.clear()
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
