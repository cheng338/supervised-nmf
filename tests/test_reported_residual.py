"""Report-only residual routing and arithmetic contracts. Author: Chang Cheng."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import _residual


MKL = {"user_api": "blas", "internal_api": "mkl", "num_threads": 1}
AOCL = {"user_api": "blas", "internal_api": "blis", "prefix": "aocl-libblis",
        "version": "AOCL-BLAS 5.3.0 Build 20260511", "num_threads": 1}


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def native_pools(monkeypatch):
    def configure(pools, platform="win32"):
        monkeypatch.setattr(_residual.sys, "platform", platform)
        monkeypatch.setattr(_residual, "_native_controller",
                            lambda: SimpleNamespace(info=lambda: pools))
    return configure


@pytest.fixture
def eligible_arrays():
    # Dispatch inspects metadata only; these arrays are never used in GEMM.
    return (np.empty((8192, 1024)), np.empty((8192, 32)), np.empty((32, 1024)))


@pytest.mark.parametrize("order", ["C", "F"])
def test_block_arithmetic_preserves_inputs_and_known_residual(order):
    W = np.full((301, 7), .5, order=order)
    H = np.full((7, 39), .5, order=order)
    X = np.full((301, 39), 2., order=order)
    arrays = (X, W, H)
    before = [a.copy() for a in arrays]
    for array in arrays:
        array.setflags(write=False)
    expected = .25 * math.sqrt(X.size)
    assert _residual._blocked_error(*arrays) == pytest.approx(expected, rel=0., abs=5e-13)
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("delta", [0., 2.**-40])
def test_exact_and_near_zero_block_residual(delta):
    W = np.full((129, 2), .5)
    H = np.ones((2, 7))
    X = np.full((129, 7), 1. + delta)
    expected = delta * math.sqrt(X.size)
    assert _residual._blocked_error(X, W, H) == pytest.approx(expected, rel=1e-14, abs=0.)


@pytest.mark.parametrize("exponent", [-500, 500])
def test_finite_extreme_nonzero_block_residual(exponent):
    # Powers of two give an independent residual without product cancellation.
    # Squared residuals remain representable at both tested scales.
    scale = math.ldexp(1., exponent)
    W = np.ones((257, 1))
    H = np.full((1, 3), scale)
    X = np.full((257, 3), 2. * scale)
    expected = scale * math.sqrt(X.size)
    actual = _residual._blocked_error(X, W, H)
    assert math.isfinite(actual) and actual > 0
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_small_and_float32_inputs_keep_original_expression(monkeypatch, dtype):
    rng = np.random.default_rng(734)
    W = rng.random((31, 3)).astype(dtype)
    H = rng.random((3, 17)).astype(dtype)
    X = (W @ H).copy()

    def unexpected_block(*args):
        pytest.fail("Small inputs must retain the full expression")

    monkeypatch.setattr(_residual, "_blocked_error", unexpected_block)
    assert _residual.reconstruction_error(X, W, H) == float(np.linalg.norm(X - W @ H))


@pytest.mark.parametrize("pools, expected", [
    pytest.param([MKL], True, id="mkl-one"),
    pytest.param([dict(MKL, num_threads=20)], False, id="mkl-twenty"),
    pytest.param([MKL, MKL.copy()], True, id="two-mkl-one"),
    pytest.param([MKL, dict(MKL, num_threads=20)], False, id="mixed-mkl-budgets"),
    pytest.param([AOCL], True, id="qualified-aocl"),
    pytest.param([dict(AOCL, num_threads=8)], False, id="aocl-eight"),
    pytest.param([AOCL, AOCL.copy()], False, id="two-aocl-libraries"),
    pytest.param([MKL, AOCL], False, id="mixed-mkl-aocl"),
    pytest.param([dict(AOCL, prefix="libblis")], False, id="generic-blis-prefix"),
    pytest.param([dict(MKL, internal_api="blis")], False, id="generic-blis"),
    pytest.param([dict(AOCL, internal_api="openblas")], False, id="wrong-internal-api"),
    pytest.param([dict(MKL, internal_api="openblas")], False, id="openblas"),
    pytest.param([MKL, dict(MKL, internal_api="openblas")], False, id="mixed-mkl-openblas"),
    pytest.param([dict(MKL, internal_api="unknown")], False, id="unknown-backend"),
    pytest.param([{"user_api": "blas", "num_threads": 1}], False, id="missing-backend"),
    pytest.param([AOCL, {"user_api": "blas", "num_threads": 1}], False, id="mixed-aocl-unknown"),
    pytest.param([MKL, {"user_api": "blas", "num_threads": 1}], False, id="mixed-mkl-unknown"),
    pytest.param([{"user_api": "blas", "internal_api": "mkl"}], False, id="missing-budget"),
    pytest.param([dict(AOCL, num_threads=None)], False, id="missing-aocl-budget"),
    pytest.param([], False, id="no-libraries"),
    pytest.param([{"user_api": "openmp", "num_threads": 1}], False, id="openmp-only"),
    pytest.param([MKL, {"user_api": "openmp", "num_threads": 20}], True, id="mkl-and-openmp"),
    pytest.param([AOCL, {"user_api": "openmp", "num_threads": 8}], True, id="aocl-and-openmp"),
])
def test_dispatch_backend_and_native_budget(native_pools, eligible_arrays, pools, expected):
    native_pools(pools)
    assert _residual._use_blocks(*eligible_arrays) is expected


@pytest.mark.parametrize("version", [None, "", "5.3.0", "AOCL-BLAS 5.2.2 Build 20250606",
    "AOCL-BLAS 5.3.0 Build 20260510", "AOCL-BLAS 5.3.0 Build 20260512",
    "AOCL-BLAS 5.3.0 Build 20260511 extra", " AOCL-BLAS 5.3.0 Build 20260511"])
def test_aocl_requires_exact_qualified_version(native_pools, eligible_arrays, version):
    native_pools([dict(AOCL, version=version)])
    assert not _residual._use_blocks(*eligible_arrays)


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("pool", [MKL, AOCL], ids=["mkl", "aocl"])
def test_platform_scope_preserves_mkl(native_pools, eligible_arrays, platform, pool):
    native_pools([pool], platform=platform)
    assert _residual._use_blocks(*eligible_arrays) is (pool is MKL or platform == "win32")


@pytest.mark.parametrize("index", [0, 1, 2], ids=["data", "coordinates", "components"])
@pytest.mark.parametrize("pool", [MKL, AOCL], ids=["mkl", "aocl"])
def test_mkl_accepts_fortran_but_aocl_requires_all_c(native_pools, eligible_arrays, index, pool):
    native_pools([pool])
    arrays = list(eligible_arrays)
    arrays[index] = np.empty(arrays[index].shape, order="F")
    assert _residual._use_blocks(*arrays) is (pool is MKL)


@pytest.mark.parametrize("index", [0, 1, 2], ids=["data", "coordinates", "components"])
@pytest.mark.parametrize("pool", [MKL, AOCL], ids=["mkl", "aocl"])
def test_each_input_must_be_float64(native_pools, eligible_arrays, index, pool):
    native_pools([pool])
    arrays = list(eligible_arrays)
    arrays[index] = np.empty(arrays[index].shape, dtype=np.float32)
    assert not _residual._use_blocks(*arrays)


@pytest.mark.parametrize("n,p,rank,expected", [
    (8191, 1024, 1, False), (8192, 1024, 1, True),
    (32768, 1024, 1, True), (32769, 1024, 1, False),
    (8192, 1023, 1, False), (8192, 4096, 1, True), (8192, 4097, 1, False),
    (8192, 1024, 0, False), (8192, 1024, 32, True), (8192, 1024, 33, False),
])
@pytest.mark.parametrize("pool", [MKL, AOCL], ids=["mkl", "aocl"])
def test_dispatch_dimension_boundaries(native_pools, pool, n, p, rank, expected):
    native_pools([pool])
    arrays = (np.empty((n, p)), np.empty((n, rank)), np.empty((rank, p)))
    assert _residual._use_blocks(*arrays) is expected


@pytest.mark.parametrize("index", [0, 1, 2], ids=["data", "coordinates", "components"])
@pytest.mark.parametrize("pool", [MKL, AOCL], ids=["mkl", "aocl"])
def test_valid_shape_strided_arrays_keep_original_path(native_pools, eligible_arrays, index, pool):
    native_pools([pool])
    arrays = list(eligible_arrays)
    rows, columns = arrays[index].shape
    arrays[index] = np.empty((rows, 2 * columns))[:, ::2]
    assert not _residual._use_blocks(*arrays)


@pytest.mark.parametrize("pool", [MKL, AOCL], ids=["mkl", "aocl"])
def test_mismatched_or_nonmatrix_inputs_do_not_enter_blocks(native_pools, eligible_arrays, pool):
    native_pools([pool])
    X, W, H = eligible_arrays
    assert not _residual._use_blocks(X, W[:-1], H)
    assert not _residual._use_blocks(X, W, H[:, :-1])
    assert not _residual._use_blocks(X, W[:, 0], H)
    assert not _residual._use_blocks([[0.]], W, H)


def test_qualified_aocl_route_passes_original_readonly_arrays(monkeypatch, native_pools, eligible_arrays):
    native_pools([AOCL])
    for array in eligible_arrays:
        array.setflags(write=False)
    received = []

    def block_report(*arrays):
        received.append(arrays)
        return 13.

    monkeypatch.setattr(_residual, "_blocked_error", block_report)
    assert _residual.reconstruction_error(*eligible_arrays) == 13.
    assert len(received) == 1
    assert all(actual is original for actual, original in zip(received[0], eligible_arrays))
