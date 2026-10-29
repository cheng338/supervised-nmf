"""Package-side NNLS workaround for AOCL's double-GEMM fringe overread.

AOCL environments use AOCL for NumPy. Other loaded libraries do not override
that environment policy. This helper changes operands, never the native library.
"""

import numpy as np


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
    mp, npad = (m + 7) // 8 * 8, (n + 7) // 8 * 8
    if (mp, npad) == (m, n):
        return np.matmul(left, right)
    a = np.zeros((mp, k), dtype=np.float64)
    b = np.zeros((k, npad), dtype=np.float64)
    a[:m, :] = left
    b[:, :n] = right
    product = np.matmul(a, b)
    return np.array(product[:m, :n], order="C", copy=True)
