"""Range certificates and operand snapshots for fixed-basis Frobenius MU."""

from math import ceil, floor, frexp, isfinite, log2

import numpy as np


def _bounded_products(x_max, rank_bits, size_bits, gram_min, gram_max, num_min, num_max, limits):
    """Bound active denominators and update factors in binary exponents."""
    x_exp = frexp(x_max)[1]
    gram_lo, gram_hi = frexp(gram_min)[1] - 3, frexp(gram_max)[1] + 2
    num_lo, num_hi = frexp(num_min)[1] - 3, frexp(num_max)[1] + 2
    sum_lo = x_exp - 1 - size_bits - rank_bits
    initial_lo, initial_hi = sum_lo // 2 - 2, (x_exp + 1) // 2 + 2
    max_weight_lo = min(initial_lo, num_lo - rank_bits - gram_hi - 4)
    weight_hi = max(initial_hi, num_hi - gram_lo + 3)
    den_lo = max_weight_lo + gram_lo - 2
    den_hi = rank_bits + weight_hi + gram_hi + 2
    return (min(sum_lo, max_weight_lo, den_lo, num_lo - den_hi - 2) >= limits.minexp + 2
            and max(x_exp + size_bits, weight_hi, den_hi, num_hi - den_lo + 2) <= limits.maxexp - 2)


def _iteration_cutoff(x, h, gram, start, gram_min, gram_max, num_min, num_max, limits):
    """Largest iteration certified by either lower bound on positive weights."""
    if max(x.size, h.size) * float(limits.eps) >= 1 / 256:
        return None
    if (min(gram_min, num_min) < float(limits.tiny) * (4 * h.shape[1])
            or not all(map(isfinite, (gram_min, gram_max, num_min, num_max, start))) or start <= 0):
        return None
    rank_bits, size_bits = (len(h) - 1).bit_length(), (x.size - 1).bit_length()
    x_max = float(x.max())
    if not _bounded_products(x_max, rank_bits, size_bits, gram_min, gram_max, num_min, num_max, limits):
        return None

    # A support mismatch on the first row certifies a nonzero residual.
    col_max = h.max(axis=0)
    covered = col_max > 0
    if not np.any((x[0] > 0) != covered):
        return None
    h_min = float(np.min(h, where=h > 0, initial=np.inf))
    col_min = float(np.min(col_max, where=covered, initial=np.inf))
    x_min = float(np.min(x, where=x > 0, initial=np.inf))
    log_num_min, log_num_max = log2(num_min), log2(num_max)
    log_gram_min, log_gram_max = log2(gram_min), log2(gram_max)
    log_start, log_rank = log2(start), log2(len(h))
    weight_upper = max(log_start, log_num_max - log2(float(gram.diagonal().min())))
    residual_upper = max(ceil(log2(x_max)),
                         ceil(weight_upper) + ceil(log2(float(col_max.max()))) + rank_bits + 3)
    norm_upper = residual_upper + (size_bits + 1) // 2 + 2
    floor_limit = int(limits.minexp) + 4
    min_residual = max(-((-floor_limit) // 2), floor_limit + norm_upper + int(limits.nmant))
    if (2 * norm_upper > limits.maxexp - 4
            or floor(log2(x_min)) - limits.nmant - 2 < min_residual):
        return None
    min_weight = max(floor_limit, floor_limit - floor(log2(min(h_min, gram_min))),
                     min_residual - floor(log2(col_min)) + 2)

    # One inequality bounds each weight's loss per step. The other uses the
    # row's total weight after the first update. Round losses down to eighths
    # of a binary exponent before inverting the two integer inequalities.
    step = floor(8 * min(0., log_num_min - log_rank - weight_upper - log_gram_max - 1.))
    first = floor(8 * min(0., log_num_min - log_rank - log_start - log_gram_max - 1.))
    later = floor(8 * min(0., log_num_min + log_gram_min - log_num_max - log_gram_max - .5))
    headroom = 8 * (floor(log_start) - 2 - min_weight)
    # The Gram diagonal is bounded by its maximum, num_min <= num_max,
    # and rank >= 1. Thus step <= -8 and later <= -4.
    return max(0, headroom // (-step), 1 + (headroom + first) // (-later))


def qualification(x, h, rank, *, bounded_prefix=True):
    """Return True for all iterations, a finite safe count, or None."""
    limits = np.finfo(x.dtype)
    if max(x.size, h.size) > 2 ** (limits.nmant - 3):
        return None
    with np.errstate(all="ignore"):
        gram = np.dot(h, h.T)
        numerator = x @ h.T
        start = np.sqrt(x.mean() / rank)
    bound = 2. ** ((limits.maxexp - 16) // 8)
    lo, hi = 2. / bound, bound / 2.

    # Independent cliques in the Gram support obey the same common-weight
    # bound as a dense Gram. The public-transform path only uses this proof.
    num_max = float(numerator.max())
    gram_max = float(gram.max())
    if (lo <= start <= hi and len(h) <= hi and num_max <= hi and gram_max <= hi
            and np.min(numerator, where=numerator > 0, initial=np.inf) >= lo
            and np.min(gram, where=gram > 0, initial=np.inf) >= lo):
        support = gram > 0
        if np.all(support) or np.array_equal(support @ support, support):
            return True
        return None
    if bounded_prefix:
        return _iteration_cutoff(x, h, gram, start, float(gram.min()), gram_max,
                                 float(numerator.min()), num_max, limits)
    return None


def capture(values):
    """Own the values while preserving strides, reduction layout and alignment."""
    if not values.size:
        return values.copy(order="K")
    strides = values.strides
    span = values.itemsize + sum((n - 1) * abs(step) for n, step in zip(values.shape, strides))
    if span > 4 * values.nbytes + 64:
        # Compact wide gaps, retaining padded axes and the stride modulo a
        # cache line. Making a padded BLAS reduction contiguous can change a
        # stopping decision or a nearest-neighbor tie.
        original_extent = extent = values.itemsize
        packed = [0] * values.ndim
        for axis in sorted(range(values.ndim), key=lambda i: abs(strides[i])):
            stride, size = strides[axis], values.shape[axis]
            if not stride or size == 1:
                continue
            distance = (values.itemsize if abs(stride) == values.itemsize
                        else max(extent, 2 * values.itemsize))
            if abs(stride) > original_extent:
                distance = max(distance, extent + values.itemsize)
            distance += (abs(stride) - distance) % 64
            packed[axis] = distance if stride > 0 else -distance
            original_extent += abs(stride) * (size - 1)
            extent += distance * (size - 1)
        strides, span = tuple(packed), extent
    lower = sum(min(0, (n - 1) * step) for n, step in zip(values.shape, strides))
    storage = np.empty(span + 63, dtype=np.uint8)
    offset = -lower + (values.ctypes.data - storage.ctypes.data + lower) % 64
    owned = np.ndarray(values.shape, dtype=values.dtype, buffer=storage, offset=offset, strides=strides)
    np.copyto(owned, values)
    return owned
