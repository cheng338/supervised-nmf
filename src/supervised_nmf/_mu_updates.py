"""Fixed-basis Frobenius MU updates and range recovery for dense coordinates."""

import math
from operator import matmul
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning


def _check_operands(current, gram, numerator, error=None):
    # Input flushing can lose even a large product of a subnormal and a large
    # normal. Inspect storage bits so denormals-are-zero mode cannot hide it.
    unsigned = np.uint32 if current.dtype == np.float32 else np.uint64
    normal_bits = 1 << np.finfo(current.dtype).nmant
    for operand in (current, gram, numerator):
        bits = operand.view(unsigned)
        if np.any((bits != 0) & (bits < normal_bits)):
            if error is not None:
                raise error
            raise FloatingPointError("Stored subnormal operand alongside active tiny denominator")


def recover_update(current, numerator, gram, denominator, small, *, error=None):
    """Reconstruct each live tiny denominator from the same old coordinate row."""
    signed = np.int32 if current.dtype == np.float32 else np.int64
    active = (current.view(signed) > 0) & (numerator.view(signed) > 0)
    rows, cols = np.nonzero(small & active)
    if not len(rows):
        if error is not None:
            raise error
        return
    if not np.isfinite(denominator).all() or np.any(denominator < 0):
        if error is not None:
            raise error
        raise FloatingPointError("Invalid denominator alongside active tiny denominator")
    if error is None:
        _check_operands(current, gram, numerator)
    limits = np.finfo(current.dtype)
    # At most rank terms smaller than tiny were lost from a sibling. Bound
    # their sum to below eps/16 of each active normal denominator.
    near = 16.0 * len(gram) * float(limits.tiny) / float(limits.eps)
    if np.any(active & ~small & (denominator < near)):
        if error is not None:
            raise error
        raise FloatingPointError("Insufficient normal sibling denominator headroom")

    from . import _mu

    values = []
    for i, j in zip(rows, cols):
        dm, de = _mu._scaled_dot(current[i], gram[:, j])
        if dm == 0 or not math.isfinite(dm):
            raise FloatingPointError("Cannot reconstruct a positive MU denominator")
        wm, we = math.frexp(float(current[i, j]))
        nm, ne = math.frexp(float(numerator[i, j]))
        try:
            value = math.ldexp(wm * nm / dm, we + ne - de)
        except OverflowError as exc:
            raise FloatingPointError("MU coordinate exceeds output range") from exc
        if not limits.tiny <= value <= limits.max:
            raise FloatingPointError("MU coordinate needs range recovery")
        values.append(value)
    # Commit after all repairs succeed. The ordinary divide/multiply then
    # leaves each repaired coordinate in place with an exact multiplier of 1.
    denominator[rows, cols] = numerator[rows, cols]
    current[rows, cols] = values


def _retry_denominator(current, gram, numerator, tiny, error, operation, message):
    if str(error) != message:
        raise error
    _check_operands(current, gram, numerator, error)
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        denominator = operation(current, gram)
    recover_update(current, numerator, gram, denominator, denominator < tiny, error=error)
    denominator[denominator == 0] = np.finfo(np.float32).eps
    return denominator


def retry_matmul(current, gram, numerator, tiny, error):
    """Retry a flagged rowwise contraction with the original @ operation."""
    return _retry_denominator(current, gram, numerator, tiny, error, matmul,
                              "underflow encountered in matmul")


def check_stop(previous, error, initial, ratio, tol, x, rank):
    """Allow a recovered cohort to stop only outside its rounding allowance."""
    # A flagged cohort formerly ran as singletons, whose GEMMs can round
    # differently. ||WH|| <= ||X|| + ||X-WH|| bounds the change in the ratio.
    features = x.shape[1]
    observation = np.sqrt(features) * np.max(x, axis=1).astype(np.float64)
    scale = (2 * observation + previous.astype(np.float64) + error) / initial + abs(tol)
    allowance = 16.0 * (features + rank + 8) * np.finfo(x.dtype).eps * scale
    if np.any(np.abs(ratio - tol) <= allowance):
        raise FloatingPointError("Ambiguous rowwise stop after a flagged denominator")


def _joint_error(x, w, h):
    """Evaluate the native residual norm, retaining its intermediate rounding."""
    residual = (x - np.dot(w, h)).ravel(order="K")
    try:
        squared = np.dot(residual, residual)
    except FloatingPointError as error:
        if str(error) != "underflow encountered in dot":
            raise
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            squared = np.dot(residual, residual)
            limits = np.finfo(residual.dtype)
            # Bound all potentially flushed terms, then verify the rounded
            # norm against a power-of-two rescaling of the same reduction.
            floor = float(limits.tiny) * (16.0 * residual.size / float(limits.eps))
            if not math.isfinite(squared) or squared < floor:
                raise FloatingPointError("Insufficient normal residual headroom")
            value = np.sqrt((squared / 2.0) * 2)
            _, exponent = math.frexp(float(np.max(np.abs(residual))))
            scaled = np.ldexp(residual, -exponent)
            reference = np.sqrt((np.dot(scaled, scaled) / 2.0) * 2)
            if value != np.ldexp(reference, exponent):
                raise FloatingPointError("Ambiguous residual rounding")
        rescued = True
    else:
        value = np.sqrt((squared / 2.0) * 2)
        rescued = False
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite MU residual")
    return value, rescued


def _dense_admission(initial_weight, numerator, gram):
    """Test whether every supported row has normal active denominators."""
    # With Gram entries and positive numerators in [1/B, B], the largest
    # supported weight contributes to every denominator. Its next weight is
    # at least 1/(16 B^3), and each weight is at most 8 B^2 since its own
    # diagonal contributes to the denominator. The resulting denominators
    # and multipliers fit within [1/(64 B^5), 64 B^5].
    limits = np.finfo(numerator.dtype)
    bound = 2.0 ** ((limits.maxexp - 8) // 8)
    lower = 1.0 / bound
    if (len(gram) > min(bound, 2 ** (limits.nmant - 2))
            or not lower <= initial_weight <= bound
            or gram.min() < lower or gram.max() > bound or numerator.max() > bound):
        return False
    return np.min(numerator, where=numerator > 0, initial=np.inf) >= lower


def joint(x, h, max_iter, tol, *, rank=None, emit_warning=True):
    """Project a batch with one initializer and a joint residual stopping test."""
    initial_weight = np.sqrt(x.mean() / (len(h) if rank is None else rank))
    if not np.isfinite(initial_weight) or initial_weight == 0:
        raise FloatingPointError("Unrepresentable MU initializer")
    w = np.full((len(x), len(h)), initial_weight, dtype=x.dtype)
    initial, initial_rescued = _joint_error(x, w, h)
    previous, previous_rescued = initial, initial_rescued
    numerator = x @ h.T
    gram = np.dot(h, h.T)
    gram_max, numerator_max = gram.max(), numerator.max()
    if not np.isfinite(gram_max) or not np.isfinite(numerator_max):
        raise FloatingPointError("Nonfinite MU input product")
    limits = np.finfo(x.dtype)
    tiny = limits.tiny
    epsilon = np.finfo(np.float32).eps
    # Initially D_j <= rank * initial_weight * max(G). On later updates,
    # D_j includes w_j G_jj, giving w_j <= 8 N_j/G_jj. Outside this bound
    # inspect each returned product for worker-local overflow.
    bound = 2.0 ** (limits.maxexp // 8)
    headroom = (len(h) <= bound and initial_weight <= bound and gram_max <= bound
                and numerator_max <= bound and gram.diagonal().min() >= 1.0 / bound)
    dense_safe = _dense_admission(initial_weight, numerator, gram)
    if dense_safe:
        unsupported = np.flatnonzero(~numerator.any(axis=1))
    dot_rescued = False
    for iteration in range(1, max_iter + 1):
        try:
            denominator = np.dot(w, gram)
        except FloatingPointError as error:
            if dense_safe:
                raise
            denominator = _retry_denominator(w, gram, numerator, tiny, error, np.dot,
                                             "underflow encountered in dot")
            dot_rescued = True
        else:
            if not headroom and not np.isfinite(denominator.max()):
                raise FloatingPointError("Nonfinite MU denominator")
            if dense_safe:
                if unsupported.size:
                    denominator[unsupported] = epsilon
            else:
                small = denominator < tiny
                if small.any():
                    recover_update(w, numerator, gram, denominator, small)
                    denominator[denominator == 0] = epsilon
        np.divide(numerator, denominator, out=denominator)
        w *= denominator
        if tol > 0 and iteration % 10 == 0:
            current, current_rescued = _joint_error(x, w, h)
            ratio = (previous - current) / initial
            if dot_rescued or initial_rescued or previous_rescued or current_rescued:
                scale = float(previous) / float(initial) + float(current) / float(initial) + abs(tol)
                allowance = (16.0 * x.size * float(limits.eps)) * scale
                if abs(float(ratio) - tol) <= allowance:
                    raise FloatingPointError("Ambiguous rescued MU stopping comparison")
            if ratio < tol:
                break
            previous, previous_rescued = current, current_rescued
    if emit_warning and iteration == max_iter and tol > 0:
        warnings.warn(f"Maximum number of iterations {max_iter} reached. "
                      "Increase it to improve convergence.", ConvergenceWarning)
    return w
