"""Reported reconstruction norms for Chang Cheng's supervised NMF package.

This helper does not drive optimization, stopping or numerical qualification.
"""

import math
import sys

import numpy as np

from ._projection import _native_controller


_BLOCK_ROWS = 128
_MAX_BLOCK_ELEMENTS = 162112 * 4096


def _use_blocks(X, coordinates, components, *, extended=False):
    # Keep small, high-rank, strided and multithreaded workloads on their
    # existing GEMM path. No copy is made merely to enter this acceleration.
    if not all(isinstance(array, np.ndarray) and array.dtype == np.float64
               and array.ndim == 2
               and (array.flags.c_contiguous or array.flags.f_contiguous)
               for array in (X, coordinates, components)):
        return False
    n, p = X.shape
    rank = coordinates.shape[1]
    max_rows, max_features = (162112, 4608) if extended else (32768, 4096)
    if not (8192 <= n <= max_rows and 1024 <= p <= max_features and 1 <= rank <= 32
            and coordinates.shape[0] == n and components.shape == (rank, p)):
        return False
    # Preserve the qualified pure-backend paths. Mixed admission below is
    # limited to the measured MKL/AOCL inventory and aligned C-order operands.
    pools = [pool for pool in _native_controller().info() if pool.get("user_api") == "blas"]
    if not pools or any(pool.get("num_threads") != 1 for pool in pools):
        return False
    if all(pool.get("internal_api") == "mkl" for pool in pools):
        return not extended or (sys.platform == "linux" and len(pools) == 1)
    if sys.platform != "win32" or not all(
            array.flags.c_contiguous for array in (X, coordinates, components)):
        return False
    aocl = [pool for pool in pools
            if pool.get("internal_api") == "blis"
            and pool.get("prefix") == "aocl-libblis"
            and pool.get("version") == "AOCL-BLAS 5.3.0 Build 20260511"]
    if len(pools) == 1:
        return not extended and len(aocl) == 1
    # The chosen row blocks, including their tails, have aligned outer
    # dimensions. No copying or padding is introduced to gain eligibility.
    return (len(pools) == 2 and len(aocl) == 1
            and sum(pool.get("internal_api") == "mkl" for pool in pools) == 1
            and aocl[0].get("architecture") == "zen3"
            and n % 8 == 0 and p % 8 == 0)


def _blocked_error(X, coordinates, components, block_rows=_BLOCK_ROWS):
    norms = []
    for start in range(0, X.shape[0], block_rows):
        stop = min(start + block_rows, X.shape[0])
        residual = coordinates[start:stop] @ components
        residual -= X[start:stop]
        norms.append(float(np.linalg.norm(residual)))
    return math.hypot(*norms)


def reconstruction_error(X, coordinates, components):
    """Return a report-only Frobenius norm without modifying supplied arrays.

    Qualified float64 arrays use row or column blocks. Matrix products and
    norm reductions can round differently from the full expression.
    The reported value is a diagnostic of the fitted factors.
    """
    if _use_blocks(X, coordinates, components):
        return _blocked_error(X, coordinates, components)
    # Qualify the larger shapes independently of the established block path.
    if (type(X) is np.ndarray and X.ndim == 2 and (X.shape[0] > 32768 or X.shape[1] > 4096)
            and type(coordinates) is np.ndarray and type(components) is np.ndarray
            and X.size <= _MAX_BLOCK_ELEMENTS
            and _use_blocks(X, coordinates, components, extended=True)
            and np.geterr() == {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}):
        try:
            with np.errstate(over="raise", invalid="raise"):
                # At rank <= 32 this leaves room for dot-product rounding even
                # when a large reconstruction almost exactly matches the data.
                bound = 2.**240
                if (coordinates.min() >= -bound and coordinates.max() <= bound
                        and components.min() >= -bound and components.max() <= bound):
                    if X.flags.c_contiguous:
                        rows = 32 if sys.platform == "win32" else _BLOCK_ROWS
                        value = _blocked_error(X, coordinates, components, rows)
                    else:
                        if coordinates.shape[1] >= 20 and X.shape[0] > 32768:
                            columns = 32 if X.shape[0] < 131072 else 128
                        else:
                            columns = max(8, min(128, (2**18 // X.shape[0]) // 8 * 8))
                        value = _blocked_error(X.T, components.T, coordinates.T, columns)
                    # Replay if one full sum of squared residuals could overflow.
                    if value < 2.**511:
                        return value
        except FloatingPointError:
            pass
    return float(np.linalg.norm(X - coordinates @ components))
