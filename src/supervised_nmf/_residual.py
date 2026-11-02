"""Reported reconstruction norms for Chang Cheng's supervised NMF package.

This helper does not drive optimization, stopping or numerical qualification.
"""

import math
import sys

import numpy as np

from ._projection import _native_controller


_BLOCK_ROWS = 128


def _use_blocks(X, coordinates, components):
    # Keep small, high-rank, strided and multithreaded workloads on their
    # existing GEMM path. No copy is made merely to enter this acceleration.
    if not all(isinstance(array, np.ndarray) and array.dtype == np.float64
               and array.ndim == 2
               and (array.flags.c_contiguous or array.flags.f_contiguous)
               for array in (X, coordinates, components)):
        return False
    n, p = X.shape
    rank = coordinates.shape[1]
    if not (8192 <= n <= 32768 and 1024 <= p <= 4096 and 1 <= rank <= 32
            and coordinates.shape[0] == n and components.shape == (rank, p)):
        return False
    # Preserve the qualified pure-backend paths. Mixed admission below is
    # limited to the measured MKL/AOCL inventory and aligned C-order operands.
    pools = [pool for pool in _native_controller().info() if pool.get("user_api") == "blas"]
    if not pools or any(pool.get("num_threads") != 1 for pool in pools):
        return False
    if all(pool.get("internal_api") == "mkl" for pool in pools):
        return True
    if sys.platform != "win32" or not all(
            array.flags.c_contiguous for array in (X, coordinates, components)):
        return False
    aocl = [pool for pool in pools
            if pool.get("internal_api") == "blis"
            and pool.get("prefix") == "aocl-libblis"
            and pool.get("version") == "AOCL-BLAS 5.3.0 Build 20260511"]
    if len(pools) == 1:
        return len(aocl) == 1
    # Every 128-row block, including its tail, then has aligned outer
    # dimensions. No copying or padding is introduced to gain eligibility.
    return (len(pools) == 2 and len(aocl) == 1
            and sum(pool.get("internal_api") == "mkl" for pool in pools) == 1
            and aocl[0].get("architecture") == "zen3"
            and n % 8 == 0 and p % 8 == 0)


def _blocked_error(X, coordinates, components):
    norms = []
    for start in range(0, X.shape[0], _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, X.shape[0])
        residual = coordinates[start:stop] @ components
        residual -= X[start:stop]
        norms.append(float(np.linalg.norm(residual)))
    return math.hypot(*norms)


def reconstruction_error(X, coordinates, components):
    """Return a report-only Frobenius norm without modifying supplied arrays.

    The bounded float64 path partitions rows and combines block norms. GEMM
    rounding may differ from one full product; this is not an exact-arithmetic
    residual or a convergence certificate. Other inputs keep the full product.
    """
    if _use_blocks(X, coordinates, components):
        return _blocked_error(X, coordinates, components)
    return float(np.linalg.norm(X - coordinates @ components))
