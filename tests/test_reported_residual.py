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


_MIXED_AOCL = dict(AOCL, architecture="zen3")
_REPORT_ERRORS = dict(divide="warn", over="warn", invalid="warn", under="ignore")


@pytest.fixture(scope="module")
def extended_metadata_arrays():
    # Real, contiguous extended geometry. No test computes on this payload.
    return np.empty((32776, 1024)), np.empty((32776, 8)), np.empty((8, 1024))


@pytest.mark.parametrize("platform,pools,expected", [
    ("linux", [MKL], True),
    ("linux", [MKL, MKL.copy()], False),
    ("linux", [dict(MKL, num_threads=2)], False),
    ("linux", [MKL, _MIXED_AOCL], False),
    ("darwin", [MKL], False),
    ("win32", [MKL], False),
    ("win32", [_MIXED_AOCL], False),
    ("win32", [MKL, _MIXED_AOCL], True),
    ("win32", [_MIXED_AOCL, MKL], True),
    ("win32", [MKL, _MIXED_AOCL, MKL.copy()], False),
    ("win32", [MKL, dict(_MIXED_AOCL, architecture="zen4")], False),
    ("win32", [MKL, dict(_MIXED_AOCL, version="other")], False),
    ("win32", [MKL, dict(_MIXED_AOCL, num_threads=2)], False),
    ("linux", [dict(MKL, internal_api="openblas")], False),
])
def test_extended_backend_scope(native_pools, extended_metadata_arrays, platform, pools, expected):
    native_pools(pools, platform=platform)
    assert not _residual._use_blocks(*extended_metadata_arrays)
    assert _residual._use_blocks(*extended_metadata_arrays, extended=True) is expected


@pytest.mark.parametrize("platform,pools", [("win32", [MKL, _MIXED_AOCL]), ("linux", [MKL])])
def test_extended_input_layout_and_type(native_pools, extended_metadata_arrays, platform, pools):
    native_pools(pools, platform=platform)
    x, w, h = extended_metadata_arrays
    for index, array in enumerate((x, w, h)):
        operands = [x, w, h]
        operands[index] = array.reshape(array.shape[::-1]).T
        assert _residual._use_blocks(*operands, extended=True) is (platform == "linux")
        operands[index] = array[:, ::-1]
        assert not _residual._use_blocks(*operands, extended=True)
        operands[index] = np.ndarray(array.shape, dtype=np.float32, buffer=array)
        assert not _residual._use_blocks(*operands, extended=True)
    assert _residual._use_blocks(x[:-1], w[:-1], h, extended=True) is (platform == "linux")
    # Transpose-based views above preserve dimensions. This wide C-order view
    # isolates the feature alignment rule while keeping row alignment intact.
    wide = x.ravel()[:8192 * 4097].reshape(8192, 4097)
    assert _residual._use_blocks(wide, w[:8192], np.empty((8, 4097)),
                                 extended=True) is (platform == "linux")


@pytest.mark.parametrize("n,p,rank,expected", [
    (8191, 4097, 1, False), (8192, 4097, 1, True),
    (32769, 1023, 1, False), (32769, 1024, 1, True),
    (162112, 4096, 32, True), (162113, 4096, 32, False),
    (8192, 4608, 32, True), (8192, 4609, 32, False),
    (32769, 1024, 0, False), (32769, 1024, 33, False),
])
def test_extended_dimension_boundaries(native_pools, n, p, rank, expected):
    from unittest.mock import Mock

    # Only shape, dtype, and flags enter this predicate. Represent the largest
    # shapes without reserving several GiB of address space in each CI worker.
    def matrix(shape):
        return Mock(spec=np.ndarray, shape=shape, ndim=2, dtype=np.dtype("float64"),
                    flags=SimpleNamespace(c_contiguous=True, f_contiguous=False))

    native_pools([MKL], platform="linux")
    arrays = (matrix((n, p)), matrix((n, rank)), matrix((rank, p)))
    assert _residual._use_blocks(*arrays, extended=True) is expected


@pytest.fixture
def scaled_extended_report(monkeypatch):
    # Keep one real outer dimension above its old ceiling and shrink the other.
    # Exercise the production report, its arithmetic guards and its real tiles.
    monkeypatch.setattr(_residual, "_use_blocks", lambda *arrays, extended=False: extended)
    return _residual.reconstruction_error


@pytest.fixture
def report_norm_shapes(monkeypatch):
    original_norm = np.linalg.norm
    shapes = []

    def observed_norm(array):
        shapes.append(array.shape)
        return original_norm(array)

    monkeypatch.setattr(np.linalg, "norm", observed_norm)
    return shapes


@pytest.mark.parametrize("platform,order,width", [
    ("win32", "C", 32), ("linux", "C", 128), ("linux", "F", 128),
])
def test_extended_tiles_and_tails_match_exact_rational_residual(
        monkeypatch, scaled_extended_report, report_norm_shapes, platform, order, width):
    monkeypatch.setattr(_residual, "sys", SimpleNamespace(platform=platform))
    n, p = 136, 4104
    i = np.arange(n) % 5 - 2
    j = np.arange(p) % 7 - 3
    w = np.column_stack((i, 2 - i, np.full(n, 3))) / 8
    h = np.vstack((j + 1, 2 * j, 4 - j)) / 16
    # Expanding these three inner products in integers gives the exact target.
    numerator = i[:, None] * (1 - j) + j + 12
    delta = 2.**-40
    x = np.array(numerator / 128 + delta, order=order)
    x[-8:, :] += 2 * delta
    x[:, -8:] += 4 * delta
    expected_square = (n - 8) * (p - 8) + 9 * 8 * (p - 8) + 25 * (n - 8) * 8 + 49 * 64
    originals = [a.copy() for a in (x, w, h)]
    for a in (x, w, h):
        a.flags.writeable = False

    with np.errstate(**_REPORT_ERRORS):
        observed = scaled_extended_report(x, w, h)
    assert observed == pytest.approx(delta * math.sqrt(expected_square), rel=5e-15, abs=0)
    outer, inner = (n, p) if order == "C" else (p, n)
    assert report_norm_shapes == [(width, inner)] * (outer // width) + [(8, inner)]
    for actual, original in zip((x, w, h), originals):
        np.testing.assert_array_equal(actual, original)
        assert not actual.flags.writeable


@pytest.mark.parametrize("n,rank,width", [
    (32769, 19, 8), (32769, 20, 32), (131071, 20, 32), (131072, 20, 128),
])
def test_tall_fortran_tile_crossover(monkeypatch, scaled_extended_report, n, rank, width):
    monkeypatch.setattr(_residual, "sys", SimpleNamespace(platform="linux"))
    x = np.empty((n, 3), order="F")
    w = np.zeros((n, rank))
    h = np.zeros((rank, 3))

    def selected(target, left, right, tile_width):
        assert tile_width == width
        for view, original in zip((target, left, right), (x, h, w)):
            assert np.shares_memory(view, original)
            assert view.shape == original.T.shape
        return 19.

    monkeypatch.setattr(_residual, "_blocked_error", selected)
    with np.errstate(**_REPORT_ERRORS):
        assert scaled_extended_report(x, w, h) == 19.


def test_extended_size_cap(monkeypatch, scaled_extended_report, report_norm_shapes):
    assert _residual._MAX_BLOCK_ELEMENTS == 162112 * 4096
    x, w, h = np.zeros((136, 4104)), np.zeros((136, 1)), np.zeros((1, 4104))
    with np.errstate(**_REPORT_ERRORS):
        for limit, full in [(x.size, False), (x.size - 1, True)]:
            monkeypatch.setattr(_residual, "_MAX_BLOCK_ELEMENTS", limit)
            report_norm_shapes.clear()
            assert scaled_extended_report(x, w, h) == 0
            assert (x.shape in report_norm_shapes) is full


@pytest.mark.parametrize("index", [0, 1, 2], ids=["data", "coordinates", "components"])
def test_extended_ndarray_subclasses_keep_full_expression(
        scaled_extended_report, report_norm_shapes, index):
    class ArraySubclass(np.ndarray):
        pass

    arrays = [np.full((136, 4104), 6.), np.ones((136, 2)), np.full((2, 4104), 3.)]
    arrays[index] = arrays[index].view(ArraySubclass)
    with np.errstate(**_REPORT_ERRORS):
        assert scaled_extended_report(*arrays) == 0
    assert report_norm_shapes == [arrays[0].shape]


@pytest.mark.parametrize("index,sign", [(1, -1), (1, 1), (2, -1), (2, 1)])
def test_extended_factor_guard_includes_both_endpoints(
        scaled_extended_report, report_norm_shapes, index, sign):
    arrays = [np.zeros((136, 4104)), np.zeros((136, 1)), np.zeros((1, 4104))]
    bound = 2.**240
    with np.errstate(**_REPORT_ERRORS):
        for magnitude, full in [(bound, False), (math.nextafter(bound, math.inf), True)]:
            arrays[index][0, 0] = sign * magnitude
            report_norm_shapes.clear()
            assert scaled_extended_report(*arrays) == 0
            assert (arrays[0].shape in report_norm_shapes) is full


@pytest.mark.parametrize("value,replay", [(math.nextafter(2.**511, 0), False), (2.**511, True)])
def test_extended_aggregate_replay_boundary(scaled_extended_report, report_norm_shapes, value, replay):
    x = np.zeros((32776, 2))
    x[0, 0] = value
    with np.errstate(**_REPORT_ERRORS):
        assert scaled_extended_report(x, np.zeros((len(x), 1)), np.zeros((1, 2))) == value
    assert (x.shape in report_norm_shapes) is replay


@pytest.mark.parametrize("case,mode", [
    ("distributed-overflow", "default"), ("split-inf-nan", "default"),
    ("signaling-nan", "default"), ("signaling-nan", "warnings-as-errors"),
    ("signaling-nan", "under-raise"), ("signaling-nan", "invalid-raise"),
    ("signaling-nan", "callback"),
])
def test_extended_exceptional_results_replay_the_numpy_expression(
        scaled_extended_report, report_norm_shapes, case, mode):
    import warnings

    x = np.zeros((32776, 2))
    w, h = np.zeros((len(x), 1)), np.zeros((1, 2))
    if case == "distributed-overflow":
        # Each tile has at most one finite square of 2**511. Their sum is 2**1024.
        x[:512:128, 0] = 2.**511
    elif case == "split-inf-nan":
        x[0, 0], x[128, 0] = np.inf, np.nan
    else:
        # A signaling NaN raises invalid on the first arithmetic operation.
        x.view(np.uint64)[128, 0] = 0x7ff0000000000001
    errors = _REPORT_ERRORS.copy()
    if mode == "under-raise":
        errors["under"] = "raise"
    elif mode == "invalid-raise":
        errors["invalid"] = "raise"
    elif mode == "callback":
        errors["invalid"] = "call"

    def observe(report):
        callbacks = []

        def handler(message, flag):
            callbacks.append((message, flag))

        with np.errstate(**errors, call=handler), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error" if mode == "warnings-as-errors" else "always", RuntimeWarning)
            try:
                outcome = ("value", float(report(x, w, h)).hex())
            except (RuntimeWarning, FloatingPointError) as exc:
                outcome = (type(exc).__name__, str(exc))
            assert np.geterr() == errors
            assert np.geterrcall() is handler
        return outcome, [(item.category, str(item.message)) for item in caught], callbacks

    reference = observe(lambda x, w, h: np.linalg.norm(x - w @ h))
    report_norm_shapes.clear()
    assert observe(scaled_extended_report) == reference
    if case == "distributed-overflow":
        assert reference[0] == ("value", "inf")
    if case == "split-inf-nan":
        assert reference[0] == ("value", "nan")
    if mode in ("default", "warnings-as-errors"):
        assert any(shape != x.shape for shape in report_norm_shapes)
    else:
        assert all(shape == x.shape for shape in report_norm_shapes)
    if reference[0][0] == "value":
        assert report_norm_shapes[-1] == x.shape
