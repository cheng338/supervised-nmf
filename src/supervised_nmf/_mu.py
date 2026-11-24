"""Numerical-range handling for Chang Cheng's fixed-basis MU projection."""

import math
from copy import copy
from collections import namedtuple
from fractions import Fraction
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from ._mu_dispatch import initial_or_restored_transform, native_projection


_CappedPrefix = namedtuple("_CappedPrefix", "count coordinates")
_SuffixSetup = namedtuple("_SuffixSetup", "result cache")


_NMF_TRANSFORM = vars(NMF).get("transform")
_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _integer(value, *, positive=False):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and (not positive or value > 0))


def _real(value, *, nonnegative=False):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value)) and (not nonnegative or value >= 0)
    except (TypeError, ValueError, OverflowError):
        return False


def _zero(value):
    return _real(value) and value == 0


def _standard_mu_basis(decomposition, selected):
    """Qualify the common fitted domain; otherwise keep public sklearn handling.

    The caller has already validated the full input and selected fitted features.
    These eligibility checks preserve warnings and errors for altered estimators;
    they do not replace sklearn's general parameter validation.
    """
    if type(decomposition) is not NMF:
        return None
    solver = getattr(decomposition, "solver", None)
    if not isinstance(solver, str) or solver != "mu":
        return None
    if (not isinstance(selected, np.ndarray) or selected.ndim != 2 or
            selected.shape[0] == 0 or selected.shape[1] == 0 or selected.dtype not in _DTYPES):
        return None
    if ("transform" in vars(decomposition) or
            not initial_or_restored_transform(vars(type(decomposition)).get("transform"))):
        return None
    if not hasattr(decomposition, "init") or hasattr(decomposition, "feature_names_in_"):
        return None
    initialization = decomposition.init
    if ((initialization is not None and not isinstance(initialization, str))
            or initialization == "nndsvd"):
        return None
    beta = getattr(decomposition, "beta_loss", None)
    frobenius = ((isinstance(beta, str) and beta == "frobenius")
                 or (_real(beta) and beta == 2))
    if not frobenius or not _zero(getattr(decomposition, "alpha_W", None)):
        return None
    alpha_h = getattr(decomposition, "alpha_H", None)
    if not ((isinstance(alpha_h, str) and alpha_h == "same") or _zero(alpha_h)):
        return None
    if not _zero(getattr(decomposition, "l1_ratio", None)):
        return None
    if not _integer(getattr(decomposition, "max_iter", None), positive=True):
        return None
    if not _real(getattr(decomposition, "tol", None), nonnegative=True):
        return None
    verbosity = getattr(decomposition, "verbose", None)
    if not _integer(verbosity) or verbosity != 0:
        return None
    basis = getattr(decomposition, "components_", None)
    if (type(basis) not in (np.ndarray, np.memmap) or basis.ndim != 2 or
            basis.shape[0] == 0 or basis.shape[1] != selected.shape[1] or
            basis.dtype != selected.dtype or basis.dtype not in _DTYPES):
        return None
    rank = basis.shape[0]
    fitted_rank = getattr(decomposition, "n_components_", None)
    requested_rank = getattr(decomposition, "n_components", None)
    fitted_features = getattr(decomposition, "n_features_in_", None)
    if (not _integer(fitted_rank, positive=True) or fitted_rank != rank or
            not _integer(requested_rank, positive=True) or requested_rank != rank or
            not _integer(fitted_features, positive=True) or fitted_features != selected.shape[1]):
        return None
    if not np.isfinite(basis).all() or np.any(basis < 0) or not np.any(basis):
        return None
    return basis


def _logs(values):
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    np.log(values, out=result, where=values > 0)
    return result


def _centered_logs(values, *, per_row):
    mantissa, exponent = np.frexp(values)
    nonzero = values > 0
    if per_row:
        base = np.max(exponent, axis=1, where=nonzero, initial=np.iinfo(exponent.dtype).min)
        base[~nonzero.any(axis=1)] = 0
        centered = _logs(mantissa) + (exponent - base[:, None]) * math.log(2.0)
    else:
        base = int(np.max(exponent, where=nonzero, initial=np.iinfo(exponent.dtype).min))
        centered = _logs(mantissa) + (exponent - base) * math.log(2.0)
    return centered, base


def _log_sum(values, axis=None):
    # Negligible relative summands can round to zero during logsumexp's exp
    # contraction. Stored input and coordinate logs are never thresholded.
    with np.errstate(under="ignore"):
        return logsumexp(values, axis=axis)


def _log_cross_reference(left, right):
    """Log of a nonnegative product; no three-dimensional product workspace."""
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for column in range(len(right)):
        result[:, column] = _log_sum(left + right[column], axis=1)
    return result


def _log_cross_panel(left, right):
    """Batch log products with an optional owned-workspace reduction."""
    if (type(left) is np.ndarray and type(right) is np.ndarray
            and left.ndim == right.ndim == 2
            and left.dtype == right.dtype == np.dtype(np.float64)
            and left.shape[1] == right.shape[1]
            and left.flags.c_contiguous and _owned_log_sum_compatible()):
        result = np.empty((len(left), len(right)), dtype=np.float64)
        budget = 65536 if right.flags.c_contiguous else 262144
        panel_width = max(1, budget // max(1, left.size))
        for start in range(0, len(right), panel_width):
            stop = min(start + panel_width, len(right))
            products = np.empty((stop - start, len(left), left.shape[1]),
                                dtype=np.float64)
            np.add(left[None, :, :], right[start:stop, None, :], out=products)
            result[:, start:stop] = _owned_log_sum(products).T
        return result

    result = np.empty((len(left), len(right)), dtype=np.float64)
    budget = 65536 if right.flags.c_contiguous else 262144
    panel_width = max(1, budget // max(1, left.size))
    for start in range(0, len(right), panel_width):
        stop = min(start + panel_width, len(right))
        products = np.empty((stop - start, len(left), left.shape[1]),
                            dtype=np.float64)
        np.add(left[None, :, :], right[start:stop, None, :], out=products)
        result[:, start:stop] = _log_sum(products, axis=2).T
    return result


def _log_cross(left, right):
    if not left.flags.c_contiguous:
        return _log_cross_reference(left, right)
    try:
        with np.errstate(all="raise"):
            return _log_cross_panel(left, right)
    except FloatingPointError:
        return _log_cross_reference(left, right)



def _log_error_reference(log_x, log_w, log_h, row_shifts):
    total = -np.inf
    for column in range(log_h.shape[1]):
        fitted = _log_sum(log_w + log_h[:, column], axis=1)
        observed = log_x[:, column]
        observed_finite, fitted_finite = np.isfinite(observed), np.isfinite(fitted)
        residual = np.full(len(log_x), -np.inf)
        observed_only = observed_finite & ~fitted_finite
        fitted_only = fitted_finite & ~observed_finite
        residual[observed_only] = observed[observed_only]
        residual[fitted_only] = fitted[fitted_only]
        different = observed_finite & fitted_finite & (observed != fitted)
        gap = np.abs(observed[different] - fitted[different])
        residual[different] = (np.maximum(observed[different], fitted[different])
                               + np.log(-np.expm1(-gap)))
        with np.errstate(under="ignore"):
            total = np.logaddexp(total, _log_sum(2 * (residual + row_shifts)))
    return 0.5 * float(total)


def _log_error_panel(log_x, log_w, log_h, row_shifts):
    """Evaluate unchanged per-feature residuals in bounded feature panels.

    The rank axis remains the last contiguous axis, followed by the original
    row reduction. Accumulate the feature totals in their original order.
    Paneling changes dispatch/allocation, not the objective or stopping test.
    """
    total = -np.inf
    rows, rank = log_w.shape
    panel_width = max(1, 262144 // max(1, rows * rank))
    for start in range(0, log_h.shape[1], panel_width):
        stop = min(start + panel_width, log_h.shape[1])
        products = np.empty((stop - start, rows, rank), dtype=np.float64)
        np.add(log_w[None, :, :], log_h[:, start:stop].T[:, None, :],
               out=products)
        fitted = _log_sum(products, axis=2)
        observed = log_x[:, start:stop].T
        observed_finite, fitted_finite = np.isfinite(observed), np.isfinite(fitted)
        residual = np.full(fitted.shape, -np.inf)
        observed_only = observed_finite & ~fitted_finite
        fitted_only = fitted_finite & ~observed_finite
        residual[observed_only] = observed[observed_only]
        residual[fitted_only] = fitted[fitted_only]
        different = observed_finite & fitted_finite & (observed != fitted)
        gap = np.abs(observed[different] - fitted[different])
        residual[different] = (np.maximum(observed[different], fitted[different])
                               + np.log(-np.expm1(-gap)))
        with np.errstate(under="ignore"):
            columns = _log_sum(2 * (residual + row_shifts[None, :]), axis=1)
            for value in columns:
                total = np.logaddexp(total, value)
    return 0.5 * float(total)


def _log_error(log_x, log_w, log_h, row_shifts):
    if not log_w.flags.c_contiguous:
        return _log_error_reference(log_x, log_w, log_h, row_shifts)
    # Exceptional arithmetic replays the scalar implementation under the
    # caller's original errstate, preserving its first error, warning or error
    # callback. No process-global warning filter is changed on the fast path.
    try:
        with np.errstate(all="raise"):
            return _log_error_panel(log_x, log_w, log_h, row_shifts)
    except FloatingPointError:
        return _log_error_reference(log_x, log_w, log_h, row_shifts)



def _stopped(previous, current, initial, tolerance):
    if current >= previous:
        return True
    decrement = 1.0 if current == -np.inf else -math.expm1(current - previous)
    return previous - initial + math.log(decrement) < math.log(tolerance)


def _restore(log_w, dtype, binary_shifts):
    limits = np.finfo(dtype)
    result = np.zeros(log_w.shape, dtype=dtype)
    # Rounding a final positive coefficient below the output dtype's range is
    # legitimate. Never apply this rounding to intermediate MU coordinates.
    with np.errstate(under="ignore"):
        for index in zip(*np.nonzero(np.isfinite(log_w))):
            logarithm = float(log_w[index])
            exponent = math.floor(logarithm / math.log(2.0))
            mantissa = math.exp(logarithm - exponent * math.log(2.0))
            try:
                value = math.ldexp(mantissa, exponent + int(binary_shifts[index[0]]))
            except OverflowError as error:
                raise FloatingPointError("MU coordinates exceed the output dtype range.") from error
            if value > float(limits.max):
                raise FloatingPointError("MU coordinates exceed the output dtype range.")
            result[index] = value
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError("MU coordinates cannot be represented in the output dtype.")
    return result


def _scaled_dot(left, right):
    positive = (left > 0) & (right > 0)
    if not positive.any():
        return 0.0, 0
    lm, le = np.frexp(left[positive])
    rm, re = np.frexp(right[positive])
    exponents = le + re
    shift = int(exponents.max())
    return math.fsum(math.ldexp(float(m), int(e) - shift)
                     for m, e in zip(lm * rm, exponents)), shift


def _rank_one_output(x, h):
    # The first rank-one MU update is the exact NNLS closed form. Evaluate
    # its products in binary-scaled coordinates to preserve subnormal ties.
    basis = np.asarray(h[0], dtype=np.float64)
    denominator, denominator_shift = _scaled_dot(basis, basis)
    result = np.zeros((len(x), 1), dtype=x.dtype)
    limits = np.finfo(x.dtype)
    exact_basis = None
    with np.errstate(under="ignore"):
        for index, row in enumerate(x):
            numerator, shift = _scaled_dot(np.asarray(row, dtype=np.float64), basis)
            if numerator == 0:
                continue
            try:
                value = math.ldexp(numerator / denominator, shift - denominator_shift)
            except OverflowError as error:
                raise FloatingPointError("MU coordinates exceed the output dtype range.") from error
            if value <= 2 * float(limits.tiny):
                # Tiny terms can decide which side of a subnormal midpoint
                # the exact rank-one solution occupies. Evaluate only these
                # exceptional outputs as exact ratios of the stored dyadic inputs.
                if exact_basis is None:
                    exact_basis = [Fraction.from_float(float(v)) for v in basis]
                    exact_denominator = sum(v * v for v in exact_basis)
                exact_cross = sum(Fraction.from_float(float(v)) * b
                                  for v, b in zip(row, exact_basis))
                ratio = exact_cross / exact_denominator
                binade = ratio.numerator.bit_length() - ratio.denominator.bit_length()
                if ((ratio.numerator < ratio.denominator << binade) if binade >= 0
                        else (ratio.numerator << -binade < ratio.denominator)):
                    binade -= 1
                exponent = max(int(limits.minexp), binade) - int(limits.nmant)
                units, remainder = divmod(ratio.numerator << -exponent, ratio.denominator)
                if 2 * remainder > ratio.denominator or (
                        2 * remainder == ratio.denominator and units % 2):
                    units += 1
                value = math.ldexp(float(units), exponent)
            if value > float(limits.max):
                raise FloatingPointError("MU coordinates exceed the output dtype range.")
            result[index, 0] = value
    return result


def _row_log_basis(h, cache):
    """Lazily share successful invariant setup within one public rowwise call."""
    if cache is None:
        return _centered_logs(h.astype(np.float64, copy=False), per_row=False)
    previous = cache.get("basis")
    try:
        reusable = (previous is not None and previous[0] is h
                    and previous[1].dtype == h.dtype
                    and previous[1].shape == h.shape
                    and previous[2] == h.strides
                    and np.array_equal(previous[1], h))
    except MemoryError:
        cache.clear()
        previous = None
        return _centered_logs(h.astype(np.float64, copy=False), per_row=False)
    if reusable:
        return previous[3]
    # A warning callback could have replaced or edited H between rows. Reuse
    # never crosses such a change, nor any failed setup calculation.
    cache.clear()
    try:
        with np.errstate(all="raise"):
            value = _centered_logs(h.astype(np.float64, copy=False), per_row=False)
    except FloatingPointError:
        return _centered_logs(h.astype(np.float64, copy=False), per_row=False)
    try:
        cache["basis"] = (h, h.copy(order="K"), h.strides, value)
    except MemoryError:
        # The owned snapshot is optional. The required centered basis
        # was already computed and remains valid for this scalar row.
        cache.clear()
    return value


def _row_log_invariant(cache, key, calculate):
    if cache is None or "basis" not in cache:
        return calculate()
    if key in cache:
        return cache[key]
    try:
        with np.errstate(all="raise"):
            value = calculate()
    except FloatingPointError:
        return calculate()
    try:
        cache[key] = value
    except MemoryError:
        cache.clear()
    return value


def _project_log_mu(x, h, max_iter, tol, *, _basis_cache=None, emit_warning=True):
    """Frobenius MU with fixed H and positive products represented by logs.

    A uniform initializer and the every-ten-iterations relative-improvement
    criterion match the ordinary recurrence in exact arithmetic. Logarithmic
    arithmetic changes rounding, not the objective; final coordinates keep the
    input dtype. Initial residual equality is numerical, not a symbolic proof.
    Rank-one zero-tolerance recovery uses the existing closed-form restoration
    directly, without logarithmic setup or updates.
    """
    if len(h) == 1 and tol == 0:
        # Rank-one restoration already supplies the final result. With no
        # stopping test, logarithmic setup and updates cannot affect it.
        return _rank_one_output(x, h)
    log_x, row_exponents = _centered_logs(x.astype(np.float64, copy=False), per_row=True)
    if _basis_cache is None:
        log_h, basis_exponent = _centered_logs(
            h.astype(np.float64, copy=False), per_row=False)
    else:
        log_h, basis_exponent = _row_log_basis(h, _basis_cache)
    row_shifts = row_exponents * math.log(2.0)
    basis_shift = basis_exponent * math.log(2.0)
    log_mean = (float(_log_sum(log_x + row_shifts[:, None]))
                - math.log(x.shape[0]) - math.log(x.shape[1]))
    # Q_i = exp(-a_i) X_i, B = exp(-b) H, Z_i = exp(b-a_i) W_i.
    # Centering avoids repeatedly adding and cancelling extreme log offsets.
    # Restore row shifts inside the objective to retain global stopping.
    log_w = np.broadcast_to(
        0.5 * (log_mean - math.log(len(h))) + basis_shift - row_shifts[:, None],
        (len(x), len(h))).copy()
    initial = _log_error(log_x, log_w, log_h, row_shifts)
    if initial == -np.inf:
        return (_rank_one_output(x, h) if len(h) == 1
                else _restore(log_w, x.dtype, row_exponents - basis_exponent))
    previous = initial
    numerator = _log_cross(log_x, log_h)
    gram = (_log_cross(log_h, log_h) if _basis_cache is None
            else _row_log_invariant(
                _basis_cache, "gram", lambda: _log_cross(log_h, log_h)))
    for iteration in range(1, max_iter + 1):
        if iteration == 1:
            first = (_log_sum(gram, axis=0)[None, :] if _basis_cache is None
                     else _row_log_invariant(
                         _basis_cache, "first_denominator",
                         lambda: _log_sum(gram, axis=0)[None, :]))
            denominator = first + np.zeros((len(x), 1))
        else:
            denominator = _log_cross(log_w, gram.T)
        active = np.isfinite(log_w) & np.isfinite(numerator)
        if not np.isfinite(denominator[active]).all():
            raise FloatingPointError("MU has a positive numerator with an undefined denominator.")
        updated = np.full(log_w.shape, -np.inf)
        if iteration == 1:
            # The first MU update cancels its common positive initializer.
            updated[active] = numerator[active] - denominator[active]
        else:
            updated[active] = log_w[active] + numerator[active] - denominator[active]
        if not np.isfinite(updated[active]).all():
            raise FloatingPointError("MU logarithmic coordinates became nonfinite.")
        log_w = updated
        if tol > 0 and iteration % 10 == 0:
            current = _log_error(log_x, log_w, log_h, row_shifts)
            if _stopped(previous, current, initial, tol):
                break
            previous = current
    if iteration == max_iter and tol > 0:
        if _basis_cache is not None:
            _basis_cache.clear()
        if emit_warning:
            warnings.warn(f"Maximum number of iterations {max_iter} reached. "
                          "Increase it to improve convergence.", ConvergenceWarning, stacklevel=2)
    return (_rank_one_output(x, h) if len(h) == 1
            else _restore(log_w, x.dtype, row_exponents - basis_exponent))


def _log_row_errors(log_x, log_w, log_h, row_shifts):
    """Independent singleton log residuals, with feature order unchanged.

    Rows are an outer vectorization axis, never a reduction axis. The trailing
    singleton reduction retains the original one-row logsumexp operation.
    Identical operation order alone is not a universal libm bitwise guarantee.
    """
    rows, rank = log_w.shape
    total = np.full(rows, -np.inf, dtype=np.float64)
    panel_width = max(1, 262144 // max(1, rows * rank))
    for start in range(0, log_h.shape[1], panel_width):
        stop = min(start + panel_width, log_h.shape[1])
        products = np.empty((stop - start, rows, rank), dtype=np.float64)
        np.add(log_w[None, :, :], log_h[:, start:stop].T[:, None, :],
               out=products)
        fitted = _log_sum(products, axis=2)
        observed = log_x[:, start:stop].T
        observed_finite, fitted_finite = np.isfinite(observed), np.isfinite(fitted)
        residual = np.full(fitted.shape, -np.inf)
        observed_only = observed_finite & ~fitted_finite
        fitted_only = fitted_finite & ~observed_finite
        residual[observed_only] = observed[observed_only]
        residual[fitted_only] = fitted[fitted_only]
        different = observed_finite & fitted_finite & (observed != fitted)
        gap = np.abs(observed[different] - fitted[different])
        residual[different] = (np.maximum(observed[different], fitted[different])
                               + np.log(-np.expm1(-gap)))
        with np.errstate(under="ignore"):
            terms = 2 * (residual + row_shifts[None, :])
            columns = _log_sum(terms[:, :, None], axis=2)
            for values in columns:
                total = np.logaddexp(total, values)
    errors = 0.5 * total
    if np.any(np.isnan(errors)) or np.any(np.isposinf(errors)):
        raise FloatingPointError("Independent MU log residuals became invalid.")
    return errors


def _handoff_log_setup(result, x, h, log_h, basis_exponent, gram,
                       first_denominator, max_iter, tol):
    """Offer only successful silent vector setup to the first scalar row.

    There are no caller callbacks between this qualified vector calculation
    and its first scalar replay. The scalar entry still checks its live model,
    bindings, basis identity/layout and owned snapshot before consuming setup.
    Warning/native boundaries keep the ordinary cache-clearing rules.
    """
    try:
        if (type(x) is not np.ndarray or type(h) is not np.ndarray
                or x.dtype != np.float64 or h.dtype != np.float64
                or type(max_iter) is not int or max_iter <= 10
                or (type(tol) is not int and type(tol) is not float) or tol <= 0
                or not all(actual is original for actual, original in zip(
                    (_standard_mu_basis, _centered_logs, _logs, _log_sum,
                     _log_cross, _log_cross_panel, _log_cross_reference,
                     _owned_log_sum, _owned_log_sum_compatible,
                     _log_error, _log_error_panel, _log_error_reference,
                     _stopped, _restore, _row_log_basis, _row_log_invariant,
                     _project_log_mu, _project_scaled_float64_mu),
                    _SUFFIX_CACHE_FUNCTIONS))
                or not all(actual is original for actual, original in zip(
                    (_project_log_mu_rows, _try_log_row_recovery,
                     _log_row_errors, _handoff_log_setup, _SuffixSetup),
                    _SUFFIX_SETUP_FUNCTIONS))):
            return result
        cache = {"basis": (h, h.copy(order="K"), h.strides,
                           (log_h, basis_exponent)),
                 "gram": gram, "first_denominator": first_denominator}
        return _SuffixSetup(result, cache)
    except MemoryError:
        # Metadata/snapshot ownership is optional. Keep the v5 prefix/decline
        # result and let its scalar suffix prepare its own setup normally.
        return result


def _project_log_mu_rows(x, h, max_iter, tol, *, _retain_capped_prefix=False,
                         _return_suffix_setup=False):
    """Speculate on independent log solves; publish no partial coordinates.

    The caller supplies a qualified double-precision log-only recovery cohort.
    Every initializer retains the scalar solver's one-row reduction. Only the
    independent update/residual operations share array dispatch. By default any
    capped row invalidates the whole attempt before warnings/results are exposed.
    The private opt-in may return a restored ordered prefix before the first cap;
    the caller must replay that first capped row and every later row serially.
    With plain built-in controls, opt-in speculation ends at the last stopping
    checkpoint strictly before the cap. Still-active rows must be replayed; their
    discarded final updates cannot change an already-stopped prefix.
    """
    log_h, basis_exponent = _centered_logs(h, per_row=False)
    basis_shift = basis_exponent * math.log(2.0)
    log_x = np.empty(x.shape, dtype=np.float64)
    row_exponents = np.empty(len(x), dtype=np.int64)
    row_shifts = np.empty(len(x), dtype=np.float64)
    log_w = np.empty((len(x), len(h)), dtype=np.float64)
    for index in range(len(x)):
        row, exponents = _centered_logs(x[index:index + 1], per_row=True)
        shifts = exponents * math.log(2.0)
        log_mean = (float(_log_sum(row + shifts[:, None]))
                    - math.log(1) - math.log(x.shape[1]))
        initializer = np.broadcast_to(
            0.5 * (log_mean - math.log(len(h))) + basis_shift - shifts[:, None],
            (1, len(h))).copy()
        log_x[index] = row[0]
        row_exponents[index] = exponents[0]
        row_shifts[index] = shifts[0]
        log_w[index] = initializer[0]
    initial = _log_row_errors(log_x, log_w, log_h, row_shifts)
    previous = initial.copy()
    iterations = np.zeros(len(x), dtype=np.int64)
    active_rows = initial != -np.inf
    indices = np.flatnonzero(active_rows)
    if len(indices):
        numerator = np.full(log_w.shape, -np.inf)
        numerator[indices] = _log_cross(log_x[indices], log_h)
        gram = _log_cross(log_h, log_h)
        first_denominator = _log_sum(gram, axis=0)[None, :]
        precap = (_retain_capped_prefix is True and type(max_iter) is int
                  and (type(tol) is int or type(tol) is float)
                  and max_iter > 10 and tol > 0)
        iteration_limit = ((max_iter - 1) // 10) * 10 if precap else max_iter
        for block_start in range(1, iteration_limit + 1, 10):
            indices = np.flatnonzero(active_rows)
            if not len(indices):
                break
            current = log_w[indices]
            active_numerator = numerator[indices]
            block_end = min(block_start + 9, iteration_limit)
            for iteration in range(block_start, block_end + 1):
                denominator = (first_denominator + np.zeros((len(indices), 1))
                               if iteration == 1 else _log_cross(current, gram.T))
                supported = np.isfinite(current) & np.isfinite(active_numerator)
                if not np.isfinite(denominator[supported]).all():
                    raise FloatingPointError(
                        "MU has a positive numerator with an undefined denominator.")
                updated = np.full(current.shape, -np.inf)
                if iteration == 1:
                    updated[supported] = active_numerator[supported] - denominator[supported]
                else:
                    updated[supported] = (current[supported] + active_numerator[supported]
                                          - denominator[supported])
                if not np.isfinite(updated[supported]).all():
                    raise FloatingPointError("MU logarithmic coordinates became nonfinite.")
                current = updated
            log_w[indices] = current
            iterations[indices] = block_end
            if tol > 0 and block_end % 10 == 0:
                errors = _log_row_errors(log_x[indices], current, log_h, row_shifts[indices])
                for offset, index in enumerate(indices):
                    if _stopped(float(previous[index]), float(errors[offset]),
                                float(initial[index]), tol):
                        active_rows[index] = False
                previous[indices] = errors
        # Active rows have not stopped strictly before the cap. Their original
        # scalar replay is authoritative, including a stop exactly at max_iter.
        # Keep executed iteration counts intact; this is a separate cap mask.
        capped_rows = active_rows if precap else None
        if tol > 0 and np.any(capped_rows if precap else iterations == max_iter):
            if not _retain_capped_prefix:
                return None
            # The scalar solver warns at the first capped row, including a
            # stop exactly at max_iter. Nothing at or beyond it is reusable.
            count = int(np.flatnonzero(capped_rows if precap else iterations == max_iter)[0])
            if count == 0:
                if (_return_suffix_setup is True
                        and _handoff_log_setup is _SUFFIX_SETUP_FUNCTIONS[3]):
                    return _handoff_log_setup(None, x, h, log_h, basis_exponent,
                        gram, first_denominator, max_iter, tol)
                return None
            prefix = _restore(log_w[:count], x.dtype,
                              row_exponents[:count] - basis_exponent)
            if (prefix.shape != (count, len(h)) or prefix.dtype != x.dtype
                    or not np.isfinite(prefix).all() or np.any(prefix < 0)):
                raise FloatingPointError("Independent MU log coordinates are invalid.")
            result = _CappedPrefix(count, prefix)
            if (_return_suffix_setup is True
                    and _handoff_log_setup is _SUFFIX_SETUP_FUNCTIONS[3]):
                return _handoff_log_setup(result, x, h, log_h, basis_exponent,
                    gram, first_denominator, max_iter, tol)
            return result
    result = _restore(log_w, x.dtype, row_exponents - basis_exponent)
    if (result.shape != (len(x), len(h)) or result.dtype != x.dtype
            or not np.isfinite(result).all() or np.any(result < 0)):
        raise FloatingPointError("Independent MU log coordinates are invalid.")
    return result


def _try_log_row_recovery(x, h, max_iter, tol, *, _retain_capped_prefix=False,
                          _return_suffix_setup=False):
    """Attempt only a double cohort whose original rows must reach log MU.

    Tiny positive H bypasses ordinary sklearn in transform_mu. Its singleton
    scaling bounds below are independent of X: an empty H interval proves that
    adding either X bound cannot make coupled scaling admissible. Tiny X skips
    that failed scaling attempt too. No other original recovery route is moved.
    """
    if (x.dtype != np.float64 or h.dtype != np.float64 or len(x) < 2
            or len(h) <= 1 or (tol > 0 and max_iter <= 10)):
        return None
    small = math.sqrt(float(np.finfo(np.float64).tiny))
    if not np.any((h > 0) & (h < small)):
        return None
    positive_h = h[h > 0]
    h_min = math.frexp(float(positive_h.min()))[1] - 1
    h_max = math.frexp(float(positive_h.max()))[1] - 1
    shape_bits = max(1, (max(x.shape[1], h.size, len(h)) - 1).bit_length())
    lower_h = max(0, -500 - h_min)
    upper_h = 240 - shape_bits - h_max
    if lower_h <= upper_h:
        return None
    # No global warning filters or caller callbacks run during speculation.
    # The established helpers still ignore their intentional negligible-term
    # underflow. Any other arithmetic failure replays the original row order.
    try:
        with np.errstate(all="raise"):
            if _retain_capped_prefix:
                if (_return_suffix_setup is True
                        and type(x) is np.ndarray and type(h) is np.ndarray
                        and _project_log_mu_rows is _SUFFIX_SETUP_FUNCTIONS[0]):
                    return _project_log_mu_rows(
                        x, h, max_iter, tol, _retain_capped_prefix=True,
                        _return_suffix_setup=True)
                return _project_log_mu_rows(
                    x, h, max_iter, tol, _retain_capped_prefix=True)
            return _project_log_mu_rows(x, h, max_iter, tol)
    except (FloatingPointError, OverflowError):
        return None


def _project_float64_mu(decomposition, selected, basis):
    """Retry float32 arithmetic in float64 without changing the fitted model."""
    widened = copy(decomposition)
    widened.components_ = basis.astype(np.float64)
    with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
        result = native_projection(widened, selected.astype(np.float64))
    if (result.shape != (len(selected), basis.shape[0])
            or not np.isfinite(result).all() or np.any(result < 0)
            or np.any(result > np.finfo(selected.dtype).max)):
        raise FloatingPointError("MU coordinates exceed the output dtype range.")
    if basis.shape[0] == 1:
        # A tiny exact term can decide a float32 subnormal rounding tie even
        # when double arithmetic loses it. Keep the exact rank-one restoration.
        return _rank_one_output(selected, basis)
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        result = result.astype(selected.dtype)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise FloatingPointError("MU coordinates cannot be represented in the output dtype.")
    return result


def _project_scaled_float64_mu(decomposition, selected, basis, *, _basis_cache=None):
    """Retry a bounded float64 problem using coupled power-of-two scaling.

    H' = 2**k H and X' = 2**(2k) X imply W' = 2**k W, including
    sklearn's uniform initializer. The MU recurrence and relative-error stop
    are unchanged in exact arithmetic. These conservative admission bounds
    and product checks are not a certificate for every later floating-point
    operation; a reported numerical failure still needs logarithmic recovery.
    """
    positive_x = selected[selected > 0]
    positive_h = basis[basis > 0]
    x_min = math.frexp(float(positive_x.min()))[1] - 1
    x_max = math.frexp(float(positive_x.max()))[1] - 1
    h_min = math.frexp(float(positive_h.min()))[1] - 1
    h_max = math.frexp(float(positive_h.max()))[1] - 1
    del positive_x, positive_h
    # Keep stored positives at least 2**-500, with headroom for dense products
    # and residual squares. The shape margin grows with the contraction size.
    shape_bits = max(1, (max(selected.size, basis.size, len(basis)) - 1).bit_length())
    lower = max(0, -500 - h_min, math.ceil((-500 - x_min) / 2))
    upper = min(240 - shape_bits - h_max,
                math.floor((400 - shape_bits - x_max) / 2))
    if lower > upper:
        raise FloatingPointError("No admissible coupled MU scaling exponent.")
    exponent = int(lower)
    with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
        scaled_x = np.ldexp(selected, 2 * exponent)
        scaled_h = np.ldexp(basis, exponent)
        if (not np.array_equal(np.ldexp(scaled_x, -2 * exponent), selected)
                or not np.array_equal(np.ldexp(scaled_h, -exponent), basis)):
            raise FloatingPointError("MU scaling is not exactly reversible.")
        gram = scaled_h @ scaled_h.T
        cross = scaled_x @ scaled_h.T
    # BLAS need not report every underflow through np.errstate. Independently
    # check support so a positive contraction cannot silently become zero.
    gram_support = (scaled_h > 0) @ (scaled_h.T > 0)
    cross_support = (scaled_x > 0) @ (scaled_h.T > 0)
    if np.any(gram_support & (gram == 0)) or np.any(cross_support & (cross == 0)):
        raise FloatingPointError("A positive MU contraction was rounded to zero.")
    for product in (gram, cross):
        if (not np.isfinite(product).all() or np.any(product < 0)
                or np.any((product > 0) & (product < np.finfo(np.float64).tiny))):
            raise FloatingPointError("MU scaling produced an unsafe contraction.")
    # These are admission checks, not solver caches. Release their temporary
    # arrays before the public transform allocates its own products.
    del gram, cross, gram_support, cross_support, product
    temporary = copy(decomposition)
    temporary.components_ = scaled_h
    # Native transform can warn and invoke caller callbacks. Strict
    # admission failures above this boundary are silent and may reuse H.
    if _basis_cache is not None:
        _basis_cache.clear()
    with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
        result = native_projection(temporary, scaled_x)
    if (result.shape != (len(selected), len(basis))
            or not np.isfinite(result).all() or np.any(result < 0)):
        raise FloatingPointError("Scaled MU coordinates are invalid.")
    if len(basis) == 1:
        return _rank_one_output(selected, basis)
    with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
        restored = np.ldexp(result, -exponent)
    if (not np.isfinite(restored).all() or np.any(restored < 0)
            or np.any((result > 0) & (restored == 0))):
        raise FloatingPointError("MU coordinates cannot be restored safely.")
    return restored


def _make_owned_log_sum():
    """Capture an optional in-place reduction for a qualified implementation.

    The panel supplies fresh scratch arrays. Unknown numerical implementations
    keep the original reduction; these checks do not constrain installation.
    """
    import hashlib
    import importlib
    from pathlib import Path
    import sys
    from types import CodeType, FunctionType

    QUALIFIED_PAIRS = (("2.5.4", "1.18.1"), ("2.5.2", "1.18.0"))
    SCIPY_SOURCE = (
        "7da908306d6382cd8214628b24f0f753ea035378ff6285276828b7af33c5b101"
    )
    AT_SOURCE = "313c2b0e3719cfff3efbef1adfb4a40a802850d3254a3a9ce34818beed7c9346"
    VALUE_BOUND = 2.0 ** 60
    COUNT_BOUND = 2 ** 53
    mu = sys.modules[__name__]
    np = mu.np
    scipy = importlib.import_module("scipy")
    versions = (getattr(np, "__version__", None), getattr(scipy, "__version__", None))
    qualified_pair = all(type(v) is str for v in versions) and versions in QUALIFIED_PAIRS
    # A diagnostic snapshot, not mutable authority for subsequent admission.
    contract = {
        "scope": (
            "owned native-float64 C products with finite bounded row maxima; "
            "mixed negative infinity admitted"
        ),
        "numpy": versions[0],
        "scipy": versions[1],
        "qualified_numpy_scipy_pairs": QUALIFIED_PAIRS,
        "required_numpy": versions[0] if qualified_pair else None,
        "required_scipy": versions[1] if qualified_pair else None,
        "normalized_scipy_source_required": SCIPY_SOURCE,
        "row_max_abs_bound": VALUE_BOUND,
        "feature_count_bound": COUNT_BOUND,
        "admitted_implementation": False,
        "reason": None,
    }
    def matches_definitions(module, source, path, signatures, *, owner=None):
        # Compile the fingerprinted source without running decorators or imports.
        code = compile(source, str(path), "exec", dont_inherit=True)
        definitions = {c.co_name: c for c in code.co_consts if type(c) is CodeType}
        if owner is not None:
            code = definitions[owner.__name__]
            definitions = {c.co_name: c for c in code.co_consts if type(c) is CodeType}
        namespace = vars(module if owner is None else owner)
        for name, defaults, keywords in signatures:
            function = namespace.get(name)
            if (type(function) is not FunctionType or function.__globals__ is not vars(module)
                    or function.__code__ != definitions.get(name)):
                return False
            actual = function.__defaults__
            if defaults is None:
                if actual is not None:
                    return False
            elif (type(actual) is not tuple or len(actual) != len(defaults)
                    or any(a is not b for a, b in zip(actual, defaults))):
                return False
            actual = function.__kwdefaults__
            if keywords is None:
                if actual is not None:
                    return False
            elif (type(actual) is not dict or len(actual) != len(keywords)
                    or any(actual.get(key) is not value for key, value in keywords.items())):
                return False
        return True

    def regular_source(module):
        path = Path(module.__file__).resolve()
        if not path.is_file():
            raise OSError("optional numerical source is not a regular file")
        return path, path.read_bytes()

    implementation = xp = None
    try:
        if not qualified_pair:
            contract["reason"] = "unqualified numerical version"
        else:
            implementation = importlib.import_module("scipy.special._logsumexp")
            xp = importlib.import_module("scipy._external.array_api_compat.numpy")
            scipy_path, scipy_bytes = regular_source(implementation)
            contract.update(
                scipy_source_path=str(scipy_path),
                scipy_raw_sha256=hashlib.sha256(scipy_bytes).hexdigest(),
                scipy_normalized_sha256=hashlib.sha256(
                    scipy_bytes.replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
            if (contract["scipy_normalized_sha256"] != SCIPY_SOURCE
                    or mu.logsumexp is not implementation.logsumexp):
                contract["reason"] = "unqualified source or logsumexp binding"
            else:
                # The owned arithmetic omits these calls from SciPy's
                # public reduction. An instrumented export must reach SciPy
                # even if it was replaced before this package was imported.
                array_api = importlib.import_module("scipy._lib._array_api")
                extra = implementation.xpx
                at_module = importlib.import_module("scipy._external.array_api_extra._lib._at")
                delegation = importlib.import_module("scipy._external.array_api_extra._delegation")
                direct_numpy = ("max", "min", "sum", "where", "log1p", "log", "all", "isfinite",
                                "isdtype", "exp", "abs", "squeeze", "inf", "nan",
                                "atleast_1d", "result_type")
                at_path, at_bytes = regular_source(at_module)
                if hashlib.sha256(at_bytes.replace(b"\r\n", b"\n")).hexdigest() != AT_SOURCE:
                    raise ValueError("unqualified SciPy indexed-update source")
                sums = (("logsumexp", (None, None, False, False), None),
                        ("_logsumexp", None, None),
                        ("_elements_and_indices_with_max_real", None, {"axis": -1}))
                updates = (("__init__", (at_module._undef,), None),
                           ("__getitem__", None, None), ("set", (None, None), None))
                if (all(getattr(xp, name) is getattr(np, name) for name in direct_numpy)
                        and type(xp.inf) is float and xp.inf == math.inf
                        and type(xp.nan) is float and math.isnan(xp.nan)
                        and all(getattr(xp, name) is getattr(xp._aliases, name)
                                for name in ("sign", "astype", "asarray"))
                        # SciPy promotes the input and copies it for its first indexed update.
                        and np.iterable is np.lib._function_base_impl.iterable
                        and np.asanyarray is np._core.multiarray.asanyarray
                        and np.array is np._core.multiarray.array
                        and implementation.np is np and np.errstate is np._core._ufunc_config.errstate
                        and all(getattr(np, name) is getattr(np._core.umath, name)
                                for name in ("add", "subtract", "exp", "sign"))
                        and all(getattr(implementation, name) is getattr(array_api, name)
                                for name in ("array_namespace", "xp_promote", "xp_size", "xpx"))
                        and extra.at is at_module.at and type(extra.at) is type
                        and extra.atleast_nd is delegation.atleast_nd
                        and matches_definitions(implementation, scipy_bytes, scipy_path, sums)
                        and matches_definitions(at_module, at_bytes, at_path, updates, owner=extra.at)):
                    contract["admitted_implementation"] = True
                else:
                    contract["reason"] = "changed SciPy reduction binding"
    except (ImportError, OSError, AttributeError, KeyError, TypeError, ValueError, RuntimeError, SyntaxError) as error:
        contract["reason"] = "optional implementation unavailable: " + type(error).__name__

    admitted = bool(contract["admitted_implementation"])
    watched = []
    functions = []
    if admitted:
        declarations = [
            (mu, ("np", "logsumexp", "_log_sum", "_log_cross",
                  "_log_cross_reference", "_log_cross_panel")),
            (implementation, (
                "logsumexp", "_logsumexp", "_elements_and_indices_with_max_real",
                "array_namespace", "xp_promote", "xp_size", "xpx", "np",
            )),
            (extra, ("at", "atleast_nd")),
            (extra.at, ("__init__", "__getitem__", "set")),
            (np, ("empty", "add", "subtract", "exp", "sign", "errstate", "ndarray",
                  "dtype", "float64", "inf", "iterable", "asanyarray", "array")),
            (xp, (*direct_numpy, "astype", "sign", "asarray")),
        ]
        try:
            for owner, names in declarations:
                for name in names:
                    namespace = vars(owner)
                    value = namespace.get(name)
                    if value is None:
                        raise AttributeError(f"{owner.__name__}.{name} is unavailable")
                    watched.append((namespace, name, value))
                    if type(value) is FunctionType:
                        keywords = value.__kwdefaults__
                        if keywords is not None and type(keywords) is not dict:
                            raise TypeError("unexpected keyword defaults")
                        functions.append((value, value.__code__, value.__defaults__, keywords,
                                          tuple(keywords.items()) if keywords is not None else ()))
        except (AttributeError, TypeError) as error:
            admitted = contract["admitted_implementation"] = False
            contract["reason"] = "optional numerical binding unavailable: " + str(error)

    def compatible():
        return (admitted
                and all(namespace.get(name) is original for namespace, name, original in watched)
                and all(fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is keywords
                        and (keywords is None or (len(keywords) == len(items)
                             and all(k in keywords and keywords[k] is v for k, v in items)))
                        for fn, code, defaults, keywords, items in functions))

    def owned_sum(products):
        # Only the panel's fresh allocation enters this private closure.
        # Complete every decline check before changing the workspace.
        if (not compatible() or type(products) is not np.ndarray
                or products.dtype != np.dtype(np.float64) or products.ndim != 3
                or not products.flags.c_contiguous or not products.flags.owndata
                or not products.flags.writeable or products.base is not None
                or not products.size or not 1 <= products.shape[2] <= COUNT_BOUND
                or implementation.array_namespace(products) is not xp):
            return mu._log_sum(products, axis=2)
        a_max, at_max = implementation._elements_and_indices_with_max_real(
            products, axis=2, xp=xp
        )
        # A finite maximum guarantees a finite contributor. Other entries may
        # be -inf; NaN, +inf and all--inf rows decline without mutation.
        if (not bool(xp.all(xp.isfinite(a_max)))
                or not bool(xp.all(a_max >= -VALUE_BOUND))
                or not bool(xp.all(a_max <= VALUE_BOUND))):
            return mu._log_sum(products, axis=2)
        # Keep SciPy's float64 mask conversion and last-axis sum, including
        # every tied maximum. The count bound makes the multiplicity exact.
        m = xp.sum(xp.astype(at_max, products.dtype), axis=2, keepdims=True,
                   dtype=products.dtype)
        with np.errstate(under="ignore"):
            products[at_max] = -np.inf
            np.subtract(products, a_max, out=products)
            np.exp(products, out=products)
            total = xp.sum(products, axis=2, keepdims=True, dtype=products.dtype)
            s = xp.where(total == 0, total, total / m)
            # Here m > 0 and s >= 0, so the sign/absolute-value branches are
            # identities. Preserve the qualified arithmetic association.
            out = (xp.log1p(s) + xp.log(m)) + a_max
        if not bool(xp.all(xp.isfinite(out))):
            # The outer dispatcher replays untouched left/right operands.
            # Never pass the mutated scratch array to the original reducer.
            raise FloatingPointError(
                "Qualified in-place logsumexp produced nonfinite output"
            )
        return out[:, :, 0]

    owned_sum.implementation_contract = dict(contract)
    return owned_sum, compatible

_owned_log_sum, _owned_log_sum_compatible = _make_owned_log_sum()


# A warning callback can replace private helpers as well as the model.
_SUFFIX_CACHE_FUNCTIONS = (
    _standard_mu_basis, _centered_logs, _logs, _log_sum,
    _log_cross, _log_cross_panel, _log_cross_reference,
    _owned_log_sum, _owned_log_sum_compatible,
    _log_error, _log_error_panel, _log_error_reference,
    _stopped, _restore, _row_log_basis, _row_log_invariant,
    _project_log_mu, _project_scaled_float64_mu,
)


def transform_mu(decomposition, selected, *, _basis_cache=None):
    """Preserve ordinary projection and correct qualified MU range failures."""
    basis = _standard_mu_basis(decomposition, selected)
    if basis is None:
        if _basis_cache is not None:
            _basis_cache.clear()
        return decomposition.transform(selected)
    if _basis_cache is not None and (
            type(selected) is not np.ndarray or type(basis) is not np.ndarray
            or selected.dtype != np.float64 or len(basis) <= 1
            or type(decomposition.max_iter) is not int
            or type(decomposition.tol) not in (int, float)
            or not all(actual is original for actual, original in zip(
                (_standard_mu_basis, _centered_logs, _logs, _log_sum,
                 _log_cross, _log_cross_panel, _log_cross_reference,
                 _owned_log_sum, _owned_log_sum_compatible,
                 _log_error, _log_error_panel, _log_error_reference,
                 _stopped, _restore, _row_log_basis, _row_log_invariant,
                 _project_log_mu, _project_scaled_float64_mu),
                _SUFFIX_CACHE_FUNCTIONS))):
        _basis_cache.clear()
        _basis_cache = None
    if not np.any(selected):
        return np.zeros((len(selected), basis.shape[0]), dtype=selected.dtype)
    threshold = math.sqrt(float(np.finfo(selected.dtype).tiny))
    small_input = np.any((selected > 0) & (selected < threshold))
    small_basis = np.any((basis > 0) & (basis < threshold))
    if not small_input and not small_basis:
        if _basis_cache is not None:
            _basis_cache.clear()
        try:
            with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
                return native_projection(decomposition, selected)
        except FloatingPointError:
            pass
    if selected.dtype == np.float32 and not small_input:
        try:
            return _project_float64_mu(decomposition, selected, basis)
        except FloatingPointError:
            pass
    if selected.dtype == np.float64 and small_basis and not small_input:
        try:
            if _basis_cache is not None:
                return _project_scaled_float64_mu(
                    decomposition, selected, basis, _basis_cache=_basis_cache)
            return _project_scaled_float64_mu(decomposition, selected, basis)
        except FloatingPointError:
            pass
    if _basis_cache is not None:
        return _project_log_mu(
            selected, basis, decomposition.max_iter, decomposition.tol,
            _basis_cache=_basis_cache)
    return _project_log_mu(selected, basis, decomposition.max_iter, decomposition.tol)


_SUFFIX_SETUP_FUNCTIONS = (
    _project_log_mu_rows, _try_log_row_recovery,
    _log_row_errors, _handoff_log_setup, _SuffixSetup,
)
