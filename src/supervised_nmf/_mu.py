"""Numerical-range handling for Chang Cheng's fixed-basis MU projection."""

import math
from copy import copy
from fractions import Fraction
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


_NMF_TRANSFORM = NMF.transform
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
            type(decomposition).transform is not _NMF_TRANSFORM):
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


def _log_cross(left, right):
    """Log of a nonnegative product; no three-dimensional product workspace."""
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for column in range(len(right)):
        result[:, column] = _log_sum(left + right[column], axis=1)
    return result


def _log_error(log_x, log_w, log_h, row_shifts):
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


def _project_log_mu(x, h, max_iter, tol):
    """Frobenius MU with fixed H and positive products represented by logs.

    A uniform initializer and the every-ten-iterations relative-improvement
    criterion match the ordinary recurrence in exact arithmetic. Logarithmic
    arithmetic changes rounding, not the objective; final coordinates keep the
    input dtype. Initial residual equality is numerical, not a symbolic proof.
    """
    log_x, row_exponents = _centered_logs(x.astype(np.float64, copy=False), per_row=True)
    log_h, basis_exponent = _centered_logs(h.astype(np.float64, copy=False), per_row=False)
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
    numerator, gram = _log_cross(log_x, log_h), _log_cross(log_h, log_h)
    for iteration in range(1, max_iter + 1):
        denominator = (_log_sum(gram, axis=0)[None, :] + np.zeros((len(x), 1))
                       if iteration == 1 else _log_cross(log_w, gram.T))
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
        warnings.warn(f"Maximum number of iterations {max_iter} reached. "
                      "Increase it to improve convergence.", ConvergenceWarning, stacklevel=2)
    return (_rank_one_output(x, h) if len(h) == 1
            else _restore(log_w, x.dtype, row_exponents - basis_exponent))


def _project_float64_mu(decomposition, selected, basis):
    """Retry float32 arithmetic in float64 without changing the fitted model."""
    widened = copy(decomposition)
    widened.components_ = basis.astype(np.float64)
    with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
        result = widened.transform(selected.astype(np.float64))
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


def _project_scaled_float64_mu(decomposition, selected, basis):
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
    with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
        result = temporary.transform(scaled_x)
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


def transform_mu(decomposition, selected):
    """Preserve ordinary projection and correct qualified MU range failures."""
    basis = _standard_mu_basis(decomposition, selected)
    if basis is None:
        return decomposition.transform(selected)
    if not np.any(selected):
        return np.zeros((len(selected), basis.shape[0]), dtype=selected.dtype)
    threshold = math.sqrt(float(np.finfo(selected.dtype).tiny))
    small_input = np.any((selected > 0) & (selected < threshold))
    small_basis = np.any((basis > 0) & (basis < threshold))
    if not small_input and not small_basis:
        try:
            with np.errstate(invalid="raise", divide="raise", over="raise", under="raise"):
                return decomposition.transform(selected)
        except FloatingPointError:
            pass
    if selected.dtype == np.float32 and not small_input:
        try:
            return _project_float64_mu(decomposition, selected, basis)
        except FloatingPointError:
            pass
    if selected.dtype == np.float64 and small_basis and not small_input:
        try:
            return _project_scaled_float64_mu(decomposition, selected, basis)
        except FloatingPointError:
            pass
    return _project_log_mu(selected, basis, decomposition.max_iter, decomposition.tol)
