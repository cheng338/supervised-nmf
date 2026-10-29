"""AOCL-only NNLS product dispatch and original-problem equivalence.

Backend fixtures exercise routing policy; they do not establish native safety
for unmeasured AOCL builds, architectures, or thread counts.
"""

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import _aocl, _nnls, _projection


AOCL = {
    "user_api": "blas", "internal_api": "blis", "prefix": "aocl-libblis",
    "version": "AOCL-BLAS 5.3.0 Build 20260511", "architecture": "zen3",
    "num_threads": 1,
}
MKL = {"user_api": "blas", "internal_api": "mkl", "prefix": "mkl_rt"}


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def native_pools(monkeypatch):
    controller = MagicMock()
    controller.info.return_value = [AOCL.copy()]
    monkeypatch.setattr(_projection, "_native_controller", lambda: controller)
    return controller


@pytest.mark.parametrize("pools,enabled", [
    pytest.param([AOCL], True, id="aocl-53"),
    pytest.param([dict(AOCL, version="AOCL-BLAS 5.2.0 Build 20260104")], True,
                 id="aocl-52"),
    pytest.param([dict(AOCL, version="AOCL-BLAS 9.0", prefix="libblis")], True,
                 id="version-identifies-aocl"),
    pytest.param([dict(AOCL, version=None)], True, id="prefix-identifies-aocl"),
    pytest.param([dict(AOCL, architecture="other", num_threads=20)], True,
                 id="policy-not-limited-to-measured-dispatch"),
    pytest.param([MKL, AOCL], True, id="user-policy-aocl-present-with-mkl"),
    pytest.param([AOCL, AOCL.copy()], True, id="multiple-aocl-libraries"),
    pytest.param([AOCL, {"user_api": "openmp", "num_threads": 20}], True,
                 id="aocl-with-openmp"),
    pytest.param([MKL], False, id="mkl"),
    pytest.param([MKL, MKL.copy()], False, id="multiple-mkl-libraries"),
    pytest.param([dict(MKL, internal_api="openblas")], False, id="openblas"),
    pytest.param([dict(AOCL, prefix="libblis", version="2.0")], False,
                 id="generic-blis"),
    pytest.param([dict(AOCL, prefix="libblis", version=None)], False,
                 id="unidentified-blis"),
    pytest.param([dict(AOCL, internal_api="mkl")], False, id="require-blis-api"),
    pytest.param([dict(AOCL, user_api="openmp")], False, id="require-blas-api"),
    pytest.param([{}], False, id="missing-metadata"),
    pytest.param([], False, id="no-pools"),
])
def test_dispatch_uses_positive_loaded_aocl_identity(native_pools, pools, enabled):
    native_pools.info.return_value = pools
    selected = _aocl.nnls_matmul()
    assert selected is (_aocl.padded_matmul if enabled else np.matmul)


def test_dispatch_rechecks_live_pool_metadata_per_solve(native_pools):
    native_pools.info.side_effect = [[MKL], [AOCL], [MKL]]
    assert _aocl.nnls_matmul() is np.matmul
    assert _aocl.nnls_matmul() is _aocl.padded_matmul
    assert _aocl.nnls_matmul() is np.matmul
    assert native_pools.info.call_count == 3


@pytest.mark.parametrize("m,k,n", [
    (19, 513, 143), (143, 257, 19), (5, 17, 3), (8, 13, 9),
    (9, 13, 8), (16, 13, 24), (1, 11, 1),
])
def test_padding_preserves_product_and_readonly_strided_inputs(m, k, n):
    rng = np.random.default_rng(9029)
    left_storage = rng.normal(size=(m, 2 * k))
    right_storage = rng.normal(size=(2 * k, n))
    left = left_storage[:, ::2]
    right = right_storage[::2, :]
    before = (left_storage.copy(), right_storage.copy())
    expected = np.einsum("ik,kj->ij", left, right, optimize=False)
    left.setflags(write=False)
    right.setflags(write=False)

    result = _aocl.padded_matmul(left, right)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.shape == (m, n)
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, left)
    assert not np.shares_memory(result, right)
    np.testing.assert_array_equal(left_storage, before[0])
    np.testing.assert_array_equal(right_storage, before[1])


def test_padding_changes_true_outer_dimensions_and_zero_fills():
    left = np.arange(19 * 13, dtype=float).reshape(19, 13) / 16
    right = np.arange(13 * 11, dtype=float).reshape(13, 11) / 32
    original_matmul = np.matmul
    calls = []

    def observed(a, b, *args, **kwargs):
        calls.append((a.copy(), b.copy()))
        return original_matmul(a, b, *args, **kwargs)

    with patch.object(_aocl.np, "matmul", side_effect=observed):
        result = _aocl.padded_matmul(left, right)

    assert len(calls) == 1
    padded_left, padded_right = calls[0]
    assert padded_left.shape == (24, 13)
    assert padded_right.shape == (13, 16)
    np.testing.assert_array_equal(padded_left[:19], left)
    np.testing.assert_array_equal(padded_right[:, :11], right)
    assert np.count_nonzero(padded_left[19:]) == 0
    assert np.count_nonzero(padded_right[:, 11:]) == 0
    np.testing.assert_array_equal(result, np.einsum("ik,kj->ij", left, right))


def test_aligned_product_avoids_padding_allocations():
    left = np.ones((16, 13))
    right = np.ones((13, 24))
    with patch.object(_aocl.np, "zeros", side_effect=AssertionError("Padding allocated")):
        result = _aocl.padded_matmul(left, right)
    np.testing.assert_array_equal(result, np.full((16, 24), 13.))


def known_problem():
    # Disjoint column supports give an independent exact NNLS solution.
    design = np.vstack([np.tile(np.eye(3), (4, 1)), np.zeros((1, 3))])
    truth = (np.arange(47 * 3).reshape(47, 3) % 17) / 8
    rows = np.einsum("ij,kj->ik", truth, design, optimize=False)
    return design, rows, truth


def test_auto_and_qr_route_every_explicit_product_through_selected_callable(
        native_pools, monkeypatch):
    original_product = _aocl.padded_matmul
    sites = set()

    def recorded(left, right):
        frame = inspect.currentframe().f_back
        try:
            sites.add((frame.f_code.co_name, frame.f_lineno))
        finally:
            del frame
        return original_product(left, right)

    monkeypatch.setattr(_aocl, "padded_matmul", recorded)
    design, rows, truth = known_problem()
    design.setflags(write=False)
    rows.setflags(write=False)
    for solver in ("auto", "qr"):
        result = _nnls.project_nnls(design, rows, solver=solver, batch_size=17)
        np.testing.assert_allclose(result, truth, rtol=1e-12, atol=1e-12)
    assert native_pools.info.call_count == 2
    # Seven BPP/qualification sites plus QR reduction must all use the chosen
    # product; exercising just the original failing cross product is not enough.
    assert {name: sum(site[0] == name for site in sites) for name, _ in sites} == {
        "_project_unscaled": 1, "_batched_block": 2,
        "_qualified_rows": 4, "_project_qr": 1,
    }


@pytest.mark.parametrize("solver", ["auto", "qr"])
@pytest.mark.parametrize("exponent", [-700, 700])
def test_selected_product_survives_exact_binary_scaling(native_pools, solver, exponent):
    design, rows, truth = known_problem()
    design = np.ldexp(design, exponent)
    rows = np.ldexp(rows, exponent)
    before = (design.copy(), rows.copy())
    with patch.object(_aocl, "padded_matmul", wraps=_aocl.padded_matmul) as product:
        result = _nnls.project_nnls(design, rows, solver=solver, batch_size=17)
    assert product.called
    assert native_pools.info.call_count == 1
    np.testing.assert_allclose(result, truth, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(design, before[0])
    np.testing.assert_array_equal(rows, before[1])


@pytest.mark.parametrize("solver", ["auto", "qr"])
def test_mkl_solve_never_enters_padding(native_pools, solver):
    native_pools.info.return_value = [MKL]
    design, rows, truth = known_problem()
    with patch.object(_aocl, "padded_matmul", side_effect=AssertionError("AOCL path on MKL")):
        assert _aocl.nnls_matmul() is np.matmul
        result = _nnls.project_nnls(design, rows, solver=solver, batch_size=17)
    np.testing.assert_allclose(result, truth, rtol=1e-12, atol=1e-12)


def test_qr_failure_keeps_selected_product_during_auto_fallback(native_pools):
    design, rows, truth = known_problem()
    with patch.object(_nnls.linalg, "qr", side_effect=np.linalg.LinAlgError("QR failed")), \
            patch.object(_aocl, "padded_matmul", wraps=_aocl.padded_matmul) as product:
        result = _nnls.project_nnls(design, rows, solver="qr", batch_size=17)
    assert product.called
    assert native_pools.info.call_count == 1
    np.testing.assert_allclose(result, truth, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("solver,count", [("scipy", 47), ("auto", 7)])
def test_existing_reference_paths_do_not_enter_padding(native_pools, solver, count):
    design, rows, truth = known_problem()
    with patch.object(_aocl, "padded_matmul", side_effect=AssertionError("Product on reference path")):
        result = _nnls.project_nnls(design, rows[:count], solver=solver)
    np.testing.assert_allclose(result, truth[:count], rtol=1e-12, atol=1e-12)
