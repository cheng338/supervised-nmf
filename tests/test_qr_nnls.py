"""Guarded QR projection contracts. Author: Chang Cheng."""

from unittest.mock import patch

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import _nnls


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def dyadic_problem(count=64):
    # Disjoint column supports give independently known, unique coefficients.
    design = np.tile(np.eye(4), (4, 1)) * np.array([1., 1.25, 1.5, 1.75])
    expected = (np.arange(count * 4).reshape(count, 4) % 17) / 8
    return design, expected @ design.T, expected


def test_qr_solves_known_active_constraints_on_original_design():
    design = np.array([[1., 0.], [0., 1.], [1., 1.], [1., 0.]])
    rows = np.tile([[0., 1., 0., 0.], [1., 0., 0., 0.],
                    [1., 1., 2., 1.], [0., 0., 0., 0.]], (8, 1))
    # For the first two rows the inactive coordinate has positive gradient.
    expected = np.tile([[0., .5], [1 / 3, 0.], [1., 1.], [0., 0.]], (8, 1))
    with patch.object(_nnls.linalg, "qr", wraps=_nnls.linalg.qr) as factor, \
            patch.object(_nnls, "_scipy_rows", wraps=_nnls._scipy_rows) as solve, \
            patch.object(_nnls, "_qualified_rows", wraps=_nnls._qualified_rows) as qualify:
        result = _nnls.project_nnls(design, rows, solver="qr", batch_size=13)
    assert factor.call_count == 1
    assert [call.args[0].shape for call in solve.call_args_list] == [(2, 2)] * 3
    assert [call.args[1].shape[1] for call in qualify.call_args_list] == [13, 13, 6]
    for call in qualify.call_args_list:
        np.testing.assert_array_equal(call.args[0], design)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    gradient = (result @ design.T - rows) @ design
    assert np.all(result >= 0)
    assert np.all(gradient >= -1e-12)
    np.testing.assert_allclose(result * gradient, 0, atol=1e-12)


@pytest.mark.parametrize("count", [31, 32, 33, 255, 256, 257])
def test_row_and_block_boundaries_have_known_solution(count):
    design, rows, expected = dyadic_problem(count)
    with patch.object(_nnls.linalg, "qr", wraps=_nnls.linalg.qr) as factor:
        actual = _nnls.project_nnls(design, rows, solver="qr", batch_size=256)
    assert factor.call_count == int(count >= 32)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("solver", ["auto", "scipy", "batched"])
def test_existing_selectors_do_not_enter_qr(solver):
    design, rows, expected = dyadic_problem()
    with patch.object(_nnls, "_project_qr", side_effect=AssertionError("QR entered")):
        actual = _nnls.project_nnls(design, rows, solver=solver)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["short", "not_tall", "underdetermined", "signed_design",
                                  "rank_deficient", "ill_conditioned", "signed_rows"])
def test_guards_retain_current_auto_results(case):
    design, rows, _ = dyadic_problem()
    if case == "short":
        rows = rows[:31]
    elif case == "not_tall":
        design, rows = design[:7], rows[:, :7]
    elif case == "underdetermined":
        design, rows = design[:3], rows[:, :3]
    elif case == "signed_design":
        design = -design
    elif case == "rank_deficient":
        design[:, 1] = design[:, 0]
    elif case == "ill_conditioned":
        design[:, 1] *= 1e-7
    elif case == "signed_rows":
        rows -= 1
    expected = _nnls.project_nnls(design, rows, maxiter=1000)
    with patch.object(_nnls, "_scipy_rows", wraps=_nnls._scipy_rows) as solve:
        actual = _nnls.project_nnls(design, rows, solver="qr", maxiter=1000)
    assert all(call.args[0].shape == design.shape for call in solve.call_args_list)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("condition,admitted", [(1e6, True), (1e6 + 1, False),
                                               (np.inf, False), (np.nan, False)])
def test_condition_boundary_is_inclusive_and_fails_closed(condition, admitted):
    design, rows, expected = dyadic_problem()
    with patch.object(_nnls.np.linalg, "cond", return_value=condition), \
            patch.object(_nnls, "_scipy_rows", wraps=_nnls._scipy_rows) as solve:
        actual = _nnls.project_nnls(design, rows, solver="qr")
    assert any(call.args[0].shape == (4, 4) for call in solve.call_args_list) == admitted
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_expanded_condition_domain_solves_dyadic_column_scaling():
    design, _, expected = dyadic_problem()
    design[:, 0] *= 2. ** -18
    rows = expected @ design.T
    with patch.object(_nnls, "_scipy_rows", wraps=_nnls._scipy_rows) as solve:
        actual = _nnls.project_nnls(design, rows, solver="qr")
    assert all(call.args[0].shape == (4, 4) for call in solve.call_args_list)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("defect", ["runtime", "floating", "linalg", "zero", "negative", "nan", "inf"])
def test_reduced_solver_failures_and_bad_coefficients_fall_back(defect):
    design, rows, expected = dyadic_problem()
    reference = _nnls.project_nnls(design, rows, maxiter=1234)
    original = _nnls.nnls

    def corrupted(matrix, row, maxiter=None):
        if matrix.shape != (4, 4):
            return original(matrix, row, maxiter=maxiter)
        assert maxiter == 1234
        errors = {"runtime": RuntimeError, "floating": FloatingPointError,
                  "linalg": np.linalg.LinAlgError}
        if defect in errors:
            raise errors[defect]("Reduced solve failed")
        fill = {"zero": 0., "negative": -1., "nan": np.nan, "inf": np.inf}[defect]
        return np.full(4, fill), 0.

    with patch.object(_nnls, "nnls", side_effect=corrupted), \
            patch.object(_nnls, "_project_unscaled", wraps=_nnls._project_unscaled) as route:
        result = _nnls.project_nnls(design, rows, solver="qr", maxiter=1234)
    assert any(call.args[2:] == ("auto", 1234, 256) for call in route.call_args_list)
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("failure", [np.linalg.LinAlgError, FloatingPointError, RuntimeError])
def test_factorization_failure_falls_back(failure):
    design, rows, _ = dyadic_problem()
    expected = _nnls.project_nnls(design, rows)
    with patch.object(_nnls.linalg, "qr", side_effect=failure("QR failed")):
        actual = _nnls.project_nnls(design, rows, solver="qr")
    np.testing.assert_array_equal(actual, expected)


def test_short_tail_fallback_uses_original_design_and_original_cap():
    design, rows, expected = dyadic_problem(257)
    original = _nnls.nnls
    reduced_calls = 0

    def corrupt_last(matrix, row, maxiter=None):
        nonlocal reduced_calls
        if matrix.shape == (4, 4):
            reduced_calls += 1
            if reduced_calls == 257:
                return np.zeros(4), 0.
        return original(matrix, row, maxiter=maxiter)

    with patch.object(_nnls, "nnls", side_effect=corrupt_last), \
            patch.object(_nnls, "_scipy_rows", wraps=_nnls._scipy_rows) as solve:
        actual = _nnls.project_nnls(design, rows, solver="qr", maxiter=1234)
    assert [(call.args[0].shape, len(call.args[1]), call.args[2])
            for call in solve.call_args_list] == [((4, 4), 256, 1234),
                                                 ((4, 4), 1, 1234),
                                                 ((16, 4), 1, 1234)]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_reference_failure_after_reduced_failure_is_not_suppressed():
    design, rows, _ = dyadic_problem()

    def fail(matrix, row, maxiter=None):
        assert maxiter == 1234
        raise RuntimeError("Reduced failure" if matrix.shape == (4, 4) else "Reference failure")

    with patch.object(_nnls, "nnls", side_effect=fail), \
            patch.object(_nnls, "_eligible_design", return_value=False):
        with pytest.raises(RuntimeError, match="Reference failure"):
            _nnls.project_nnls(design, rows, solver="qr", maxiter=1234)


def test_small_iteration_cap_preserves_public_failure():
    design, rows, _ = dyadic_problem()
    # Four nonzero, orthogonal coordinates require more than one NNLS step.
    rows[:] = design.sum(axis=1)
    with patch.object(_nnls.linalg, "qr", side_effect=AssertionError("QR entered")):
        for solver in ("auto", "qr"):
            with pytest.raises(RuntimeError, match="iterations"):
                _nnls.project_nnls(design, rows, solver=solver, maxiter=1)


@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize("readonly", [False, True])
def test_input_layout_and_ownership(order, readonly):
    design, rows, expected = dyadic_problem()
    if order == "strided":
        design = np.repeat(design, 2, axis=0)[::2]
        rows = np.repeat(rows, 2, axis=1)[:, ::2]
    else:
        design, rows = np.array(design, order=order), np.array(rows, order=order)
    before = design.copy(), rows.copy()
    if readonly:
        design.setflags(write=False)
        rows.setflags(write=False)
    actual = _nnls.project_nnls(design, rows, solver="qr")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for array, original in zip((design, rows), before):
        np.testing.assert_array_equal(array, original)
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("exponent", [-700, 0, 700])
def test_common_binary_scale_preserves_known_coefficients(exponent):
    design, rows, expected = dyadic_problem()
    actual = _nnls.project_nnls(np.ldexp(design, exponent), np.ldexp(rows, exponent), solver="qr")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_repeated_calls_and_changed_basis_do_not_reuse_stale_factors():
    design, rows, expected = dyadic_problem()
    with patch.object(_nnls.linalg, "qr", wraps=_nnls.linalg.qr) as factor:
        first = _nnls.project_nnls(design, rows, solver="qr")
        repeated = _nnls.project_nnls(design, rows, solver="qr")
        changed_query = _nnls.project_nnls(design, rows * 2, solver="qr")
        changed_basis = _nnls.project_nnls(design * 2, rows, solver="qr")
    assert factor.call_count == 4
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(changed_query, expected * 2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(changed_basis, expected / 2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["complex", "nan", "width", "empty_design", "cap", "batch", "range"])
def test_validation_precedes_numerical_dispatch(case):
    design, rows, _ = dyadic_problem()
    controls = {}
    error = ValueError
    if case == "complex":
        design = design.astype(complex)
    elif case == "nan":
        rows[0, 0] = np.nan
    elif case == "width":
        rows = rows[:, :-1]
    elif case == "empty_design":
        design, rows = design[:0], rows[:, :0]
    elif case == "cap":
        controls["maxiter"] = True
    elif case == "batch":
        controls["batch_size"] = 0
    elif case == "range":
        rows[0, 0] = np.nextafter(0., 1.)
        error = FloatingPointError
    with patch.object(_nnls, "_project_qr", side_effect=AssertionError("QR entered")):
        with pytest.raises(error):
            _nnls.project_nnls(design, rows, solver="qr", **controls)


def test_empty_and_float32_inputs_keep_public_output_contract():
    design, rows, expected = dyadic_problem()
    empty = _nnls.project_nnls(design, rows[:0], solver="qr")
    assert empty.shape == (0, 4) and empty.dtype == np.float64
    actual = _nnls.project_nnls(design.astype(np.float32), rows.astype(np.float32), solver="qr")
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
