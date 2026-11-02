"""Package-side NNLS workaround for AOCL's double-GEMM fringe overread.

AOCL environments use AOCL for NumPy. Other loaded libraries do not override
that environment policy. This helper changes operands, never the native library.
"""

import numpy as np
from numpy import ndarray as _ndarray


def nnls_matmul():
    """Select once per NNLS call; ordinary backends retain NumPy's matmul.

    Discovery is reused by the projection controller and renewed after fork.
    NumPy and SciPy are already imported before discovery. Runtime libraries
    should be configured at process startup, as for projection thread control.
    """
    # Import at call time: _projection itself imports _nnls.
    from ._projection import _native_controller

    for pool in _native_controller().info():
        if (pool.get("user_api") == "blas"
                and pool.get("internal_api") == "blis"
                and (str(pool.get("version", "")).startswith("AOCL-BLAS ")
                     or pool.get("prefix") == "aocl-libblis")):
            return padded_matmul
    return np.matmul


def padded_matmul(left, right):
    """Multiply validated float64 NNLS arrays with real outer-dimension padding.

    For (M,K) @ (K,N), enlarge M and N to multiples of eight, leave K unchanged,
    and crop the product. This preserves the original product in exact
    arithmetic. Added entries are zeros, not unused storage hiding an overread.
    The inspected single-thread Zen3 path cannot form its faulty 5-by-3 tile
    from these outer dimensions. Other native paths have no universal guarantee.
    """
    m, k = left.shape
    n = right.shape[1]
    # Shapes are nonnegative integers; clear the low bits to round up by eight.
    mp, npad = (m + 7) & ~7, (n + 7) & ~7
    if mp == m:
        if npad == n:
            return np.matmul(left, right)
        if (type(left) is _ndarray and left.flags.c_contiguous
                and left.dtype == np.float64 and left.flags.aligned
                and left.strides == (k * 8, 8)):
            b = np.zeros((k, npad), dtype=np.float64)
            b[..., :n] = right
            product = np.matmul(left, b)
            return np.array(product[:m, :n], order="C", copy=True)
    elif (npad == n and type(right) is _ndarray and right.flags.c_contiguous
            and right.dtype == np.float64 and right.flags.aligned
            and right.strides == (n * 8, 8)):
        a = np.zeros((mp, k), dtype=np.float64)
        a[:m] = left
        product = np.matmul(a, right)
        return np.array(product[:m, :n], order="C", copy=True)
    if (mp != m and npad != n and m >= 8 and n >= 8
            and type(left) is _ndarray and type(right) is _ndarray
            and left.dtype == np.float64 and right.dtype == np.float64
            and left.flags.aligned and right.flags.aligned
            and right.flags.c_contiguous and right.strides == (n * 8, 8)):
        # Keep the original full padded left; split only protected N regions.
        nc = n & ~7
        a = np.zeros((mp, k), dtype=np.float64)
        b = np.zeros((k, 8), dtype=np.float64)
        a[:m] = left
        b[:, :n - nc] = right[:, nc:]
        result = np.empty((m, n), dtype=np.float64)
        result[:, :nc] = np.matmul(a, right[:, :nc])[:m]
        result[:, nc:] = np.matmul(a, b)[:m, :n - nc]
        return result
    # Ineligible operands retain the original allocation and population order.
    a = np.zeros((mp, k), dtype=np.float64)
    b = np.zeros((k, npad), dtype=np.float64)
    a[:m] = left
    b[..., :n] = right
    product = np.matmul(a, b)
    return np.array(product[:m, :n], order="C", copy=True)


class _PreparedRightOperand:
    """Own one fixed validated RHS in its existing padded native geometry.

    The caller must bind this object to a single immutable logical operand.
    This is not an array-identity cache: later changing operands must never be
    substituted. Only the right preparation is retained; left preparation and
    the owned C output remain identical to ``padded_matmul``.
    """

    def __init__(self, right):
        self._k, self._n = right.shape
        self._npad = (self._n + 7) // 8 * 8
        self._right = np.zeros((self._k, self._npad), dtype=np.float64)
        self._right[:, :self._n] = right

    def matmul(self, left):
        m, k = left.shape
        if k != self._k:
            raise ValueError("Prepared RHS has a different inner dimension.")
        mp = (m + 7) // 8 * 8
        if (mp, self._npad) == (m, self._n):
            raise ValueError("Prepared RHS requires a genuinely padded product.")
        a = np.zeros((mp, k), dtype=np.float64)
        a[:m, :] = left
        product = np.matmul(a, self._right)
        return np.array(product[:m, :self._n], order="C", copy=True)
