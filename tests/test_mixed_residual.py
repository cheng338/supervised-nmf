"""Metadata contracts for the narrowly qualified mixed MKL/AOCL residual route."""

from types import SimpleNamespace

import numpy as np
import pytest

from supervised_nmf import _residual


MKL = {"user_api": "blas", "internal_api": "mkl", "num_threads": 1}
AOCL = {"user_api": "blas", "internal_api": "blis", "prefix": "aocl-libblis",
        "version": "AOCL-BLAS 5.3.0 Build 20260511", "architecture": "zen3",
        "num_threads": 1}
OPENBLAS = {"user_api": "blas", "internal_api": "openblas", "num_threads": 1}
UNKNOWN = {"user_api": "blas", "internal_api": "unknown", "num_threads": 1}


def _arrays(n=8192, p=1024, rank=8, order="C"):
    # These uninitialized arrays are inspected only for metadata, never values.
    return (np.empty((n, p), order=order), np.empty((n, rank), order=order),
            np.empty((rank, p), order=order))


@pytest.fixture
def native_pools(monkeypatch):
    def configure(pools, platform="win32"):
        monkeypatch.setattr(_residual.sys, "platform", platform)
        monkeypatch.setattr(_residual, "_native_controller",
                            lambda: SimpleNamespace(info=lambda: pools))
    return configure


@pytest.fixture
def eligible_arrays():
    return _arrays()


@pytest.mark.parametrize("reverse", [False, True], ids=["mkl-first", "aocl-first"])
@pytest.mark.parametrize("openmp", [False, True], ids=["blas-only", "extra-openmp"])
def test_exact_mixed_pair_accepts_both_orders_and_ignores_openmp(native_pools, eligible_arrays, reverse, openmp):
    pools = [MKL.copy(), AOCL.copy()]
    if reverse:
        pools.reverse()
    if openmp:
        pools.insert(1, {"user_api": "openmp", "internal_api": "openmp", "num_threads": 8})
    native_pools(pools)
    assert _residual._use_blocks(*eligible_arrays)


@pytest.mark.parametrize("role,key,value", [
    pytest.param("aocl", "architecture", "zen2", id="zen2"),
    pytest.param("aocl", "architecture", "zen4", id="zen4"),
    pytest.param("aocl", "architecture", None, id="null-architecture"),
    pytest.param("aocl", "architecture", "Zen3", id="architecture-case"),
    pytest.param("aocl", "prefix", "libblis", id="generic-prefix"),
    pytest.param("aocl", "prefix", None, id="null-prefix"),
    pytest.param("aocl", "version", "AOCL-BLAS 5.3.0 Build20260511", id="build-format"),
    pytest.param("aocl", "version", "AOCL-BLAS 5.3.0 Build 20260510", id="different-build"),
    pytest.param("aocl", "version", "AOCL-BLAS 5.3.0 Build 20260511 extra", id="build-suffix"),
    pytest.param("aocl", "version", None, id="null-build"),
    pytest.param("aocl", "internal_api", "openblas", id="aocl-openblas"),
    pytest.param("aocl", "internal_api", "unknown", id="aocl-unknown-api"),
    pytest.param("aocl", "internal_api", None, id="aocl-null-api"),
    pytest.param("aocl", "num_threads", 0, id="aocl-zero-threads"),
    pytest.param("aocl", "num_threads", 2, id="aocl-two-threads"),
    pytest.param("aocl", "num_threads", None, id="aocl-null-threads"),
    pytest.param("mkl", "internal_api", "openblas", id="mkl-openblas"),
    pytest.param("mkl", "internal_api", "unknown", id="mkl-unknown-api"),
    pytest.param("mkl", "internal_api", None, id="mkl-null-api"),
    pytest.param("mkl", "num_threads", 0, id="mkl-zero-threads"),
    pytest.param("mkl", "num_threads", 2, id="mkl-two-threads"),
    pytest.param("mkl", "num_threads", None, id="mkl-null-threads"),
])
def test_mixed_pair_requires_exact_runtime_metadata(native_pools, eligible_arrays, role, key, value):
    records = {"mkl": MKL.copy(), "aocl": AOCL.copy()}
    records[role][key] = value
    native_pools([records["mkl"], records["aocl"]])
    assert not _residual._use_blocks(*eligible_arrays)


@pytest.mark.parametrize("role,key", [
    ("aocl", "internal_api"), ("aocl", "prefix"), ("aocl", "version"),
    ("aocl", "architecture"), ("aocl", "num_threads"),
    ("mkl", "internal_api"), ("mkl", "num_threads"),
])
def test_mixed_pair_rejects_missing_required_metadata(native_pools, eligible_arrays, role, key):
    records = {"mkl": MKL.copy(), "aocl": AOCL.copy()}
    records[role].pop(key)
    native_pools([records["aocl"], records["mkl"]])
    assert not _residual._use_blocks(*eligible_arrays)


@pytest.mark.parametrize("pools", [
    pytest.param([AOCL, AOCL.copy()], id="two-aocl"),
    pytest.param([MKL, AOCL, MKL.copy()], id="extra-mkl"),
    pytest.param([AOCL, MKL, AOCL.copy()], id="extra-aocl"),
    pytest.param([AOCL, OPENBLAS], id="aocl-openblas"),
    pytest.param([MKL, AOCL, OPENBLAS], id="extra-openblas"),
    pytest.param([AOCL, UNKNOWN], id="aocl-unknown"),
    pytest.param([MKL, AOCL, UNKNOWN], id="extra-unknown"),
    pytest.param([AOCL, {"user_api": "blas", "num_threads": 1}], id="unidentified-blas"),
])
def test_mixed_pair_does_not_generalize_to_other_pool_sets(native_pools, eligible_arrays, pools):
    native_pools(pools)
    assert not _residual._use_blocks(*eligible_arrays)


@pytest.mark.parametrize("platform", ["linux", "darwin", "cygwin"])
def test_mixed_pair_is_windows_only(native_pools, eligible_arrays, platform):
    native_pools([MKL, AOCL], platform=platform)
    assert not _residual._use_blocks(*eligible_arrays)


@pytest.mark.parametrize("index", [0, 1, 2], ids=["data", "coordinates", "components"])
@pytest.mark.parametrize("violation", ["fortran", "strided", "float32", "one-dimensional", "three-dimensional", "array-like"])
def test_mixed_pair_requires_each_input_to_be_c_float64_matrix(native_pools, eligible_arrays, index, violation):
    native_pools([MKL, AOCL])
    arrays = list(eligible_arrays)
    rows, columns = arrays[index].shape
    if violation == "fortran":
        arrays[index] = np.empty((rows, columns), order="F")
    elif violation == "strided":
        arrays[index] = np.empty((rows, 2 * columns))[:, ::2]
    elif violation == "float32":
        arrays[index] = np.empty((rows, columns), dtype=np.float32)
    elif violation == "one-dimensional":
        arrays[index] = np.empty(rows * columns)
    elif violation == "three-dimensional":
        arrays[index] = np.empty((rows, columns, 1))
    else:
        arrays[index] = [[0.]]
    assert not _residual._use_blocks(*arrays)


@pytest.mark.parametrize("mismatch", ["coordinate-rows", "component-rank", "component-features"])
def test_mixed_pair_requires_matching_shapes(native_pools, eligible_arrays, mismatch):
    native_pools([MKL, AOCL])
    x, w, h = eligible_arrays
    if mismatch == "coordinate-rows":
        w = np.empty((len(x) - 1, w.shape[1]))
    elif mismatch == "component-rank":
        h = np.empty((h.shape[0] - 1, h.shape[1]))
    else:
        h = np.empty((h.shape[0], h.shape[1] - 1))
    assert not _residual._use_blocks(x, w, h)


@pytest.mark.parametrize("n,p,rank,expected", [
    pytest.param(8192, 1024, 1, True, id="lower-bounds"),
    pytest.param(8192, 1024, 32, True, id="maximum-rank"),
    pytest.param(8200, 1032, 1, True, id="aligned-eight-row-tail-r1"),
    pytest.param(8200, 1032, 32, True, id="aligned-eight-row-tail-r32"),
    pytest.param(32768, 1024, 32, True, id="maximum-rows"),
    pytest.param(8192, 4096, 32, True, id="maximum-features"),
    pytest.param(8184, 1024, 1, False, id="aligned-too-few-rows"),
    pytest.param(32776, 1024, 1, False, id="aligned-too-many-rows"),
    pytest.param(8192, 1016, 1, False, id="aligned-too-few-features"),
    pytest.param(8192, 4104, 1, False, id="aligned-too-many-features"),
    pytest.param(8192, 1024, 0, False, id="zero-rank"),
    pytest.param(8192, 1024, 33, False, id="rank-above-bound"),
    pytest.param(8197, 1024, 8, False, id="unaligned-rows"),
    pytest.param(8192, 1027, 8, False, id="unaligned-features"),
    pytest.param(8197, 1027, 8, False, id="both-unaligned"),
    pytest.param(8201, 1032, 8, False, id="unaligned-short-tail"),
    pytest.param(8200, 1033, 8, False, id="unaligned-feature-tail"),
])
def test_mixed_pair_keeps_bounds_and_requires_aligned_outer_dimensions(native_pools, n, p, rank, expected):
    native_pools([AOCL, MKL])
    assert _residual._use_blocks(*_arrays(n, p, rank)) is expected


@pytest.mark.parametrize("architecture", ["absent", None, "zen2", "zen4"])
@pytest.mark.parametrize("n,p", [(8192, 1024), (8197, 1027)], ids=["aligned", "unaligned"])
def test_sole_aocl_keeps_legacy_architecture_and_alignment_policy(native_pools, architecture, n, p):
    pool = AOCL.copy()
    if architecture == "absent":
        pool.pop("architecture")
    else:
        pool["architecture"] = architecture
    native_pools([pool])
    # Metadata only: preserving the old decision does not qualify this arithmetic.
    assert _residual._use_blocks(*_arrays(n, p))


@pytest.mark.parametrize("pools", [[MKL], [MKL, MKL.copy()]], ids=["one-mkl", "two-mkl"])
@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_all_mkl_keeps_legacy_layout_platform_and_alignment_policy(native_pools, pools, platform, order):
    native_pools(pools, platform=platform)
    assert _residual._use_blocks(*_arrays(8197, 1027, order=order))


@pytest.mark.parametrize("reverse", [False, True], ids=["mkl-first", "aocl-first"])
def test_qualified_mixed_public_route_passes_original_readonly_arrays(monkeypatch, native_pools, eligible_arrays, reverse):
    pools = [MKL, AOCL]
    native_pools(pools[::-1] if reverse else pools)
    for array in eligible_arrays:
        array.setflags(write=False)
    received = []

    def sentinel(*arrays):
        received.append(arrays)
        return 17.

    monkeypatch.setattr(_residual, "_blocked_error", sentinel)
    assert _residual.reconstruction_error(*eligible_arrays) == 17.
    assert len(received) == 1
    assert all(actual is original for actual, original in zip(received[0], eligible_arrays))
    assert all(not array.flags.writeable for array in eligible_arrays)
