"""Qualification temporaries preserve the original-space product contract.

All products use independent reference arithmetic, never raw native GEMM.
These contracts complement existing solver/public-workflow numerical tests.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import _aocl, _nnls

assert Path(_nnls.__file__).resolve() == ROOT / "src/supervised_nmf/_nnls.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"
VARIANT = "T-ADD"

assert _nnls.nnls_matmul is _aocl.nnls_matmul


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def reference_product(left, right):
    return np.array(np.einsum("ik,kj->ij", left, right, optimize=False),
                    dtype=np.float64, order="C", copy=True)


def design_matrix():
    return np.array([[1., 0.], [0., 1.], [2., 0.], [0., 2.]])


def ordinary_problem():
    design = design_matrix()
    truth = np.array([[1., 0., 0., 1.], [0., 2., 0., 1.]])
    targets = reference_product(design, truth)
    coefficients = truth.copy()
    coefficients[0, 2] = -0.125  # Feasibility must reject this column.
    coefficients[:, 3] = 0.0   # The nonzero RHS fails original-space KKT.
    return design, targets, coefficients, np.array([True, True, False, False])


def layout(values, kind, readonly):
    if kind == "fortran":
        result = np.array(values, order="F", copy=True)
    elif kind == "strided":
        storage = np.empty((2 * values.shape[0], 2 * values.shape[1]))
        result = storage[::2, ::2]
        result[:] = values
    else:
        result = np.array(values, order="C", copy=True)
    if readonly:
        result.setflags(write=False)
    return result


def snapshot(value):
    return (value.shape, value.strides, value.dtype, value.flags.writeable,
            value.tobytes(order="C"))


def exercise(monkeypatch, design, targets, coefficients, expected):
    before = [snapshot(value) for value in (design, targets, coefficients)]
    absolute_calls, products, events = [], [], []

    class ObservedNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def abs(self, value, *args, **kwargs):
            label = ("targets" if value is targets else
                     "fitted" if products and value is products[0]["result"]
                     else "projected")
            events.append(("abs", label))
            result = np.abs(value, *args, **kwargs)
            absolute_calls.append({"input": value, "result": result,
                                   "initial_value": result.copy()})
            return result

    def product(left, right):
        events.append(("product", len(products)))
        result = reference_product(left, right)
        products.append({"left": left, "right": right,
                         "left_before": snapshot(left), "right_before": snapshot(right),
                         "result": result, "result_before": snapshot(result)})
        return result

    with monkeypatch.context() as patch:
        # Rebind only this module's NumPy name, not the NumPy module itself.
        patch.setattr(_nnls, "np", ObservedNumpy())
        actual = _nnls._qualified_rows(design, targets, coefficients, matmul=product)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.bool_ and actual.shape == (targets.shape[1],)
    assert before == [snapshot(value) for value in (design, targets, coefficients)]
    assert len(products) == 4
    fitted, gradient, cross_scale, roundoff_scale = products
    assert fitted["left"] is design and fitted["right"] is coefficients
    for call in products[1:]:
        np.testing.assert_array_equal(call["left"], design.T)
    np.testing.assert_array_equal(gradient["right"], fitted["result"] - targets)
    np.testing.assert_array_equal(cross_scale["right"], np.abs(targets))
    np.testing.assert_array_equal(roundoff_scale["right"],
                                  np.abs(fitted["result"]) + np.abs(targets))


    target_abs = [call for call in absolute_calls if call["input"] is targets]
    fitted_abs = [call for call in absolute_calls if call["input"] is fitted["result"]]
    assert len(fitted_abs) == 1
    assert len(target_abs) == (1 if VARIANT == "T-ABS" else 2)
    absolute_targets = target_abs[0]["result"]
    absolute_fitted = fitted_abs[0]["result"]
    assert cross_scale["right"] is absolute_targets
    assert absolute_fitted.flags.owndata and absolute_fitted.base is None
    for original in (design, targets, coefficients, fitted["result"], absolute_targets):
        assert not np.shares_memory(absolute_fitted, original)
    for call in target_abs:
        assert call["result"].flags.owndata
        np.testing.assert_array_equal(call["result"], call["initial_value"])
        assert not np.signbit(call["result"]).any()
    if VARIANT == "T-ABS":
        assert roundoff_scale["right"] is not absolute_fitted
        assert not np.shares_memory(roundoff_scale["right"], absolute_fitted)
        assert not np.shares_memory(roundoff_scale["right"], absolute_targets)
        np.testing.assert_array_equal(absolute_fitted, fitted_abs[0]["initial_value"])
        expected_events = [
            ("product", 0), ("product", 1), ("abs", "targets"),
            ("product", 2), ("abs", "fitted"), ("product", 3), ("abs", "projected")]
    else:
        assert roundoff_scale["right"] is absolute_fitted
        assert target_abs[0]["result"] is not target_abs[1]["result"]
        assert not np.shares_memory(target_abs[0]["result"], target_abs[1]["result"])
        np.testing.assert_array_equal(
            absolute_fitted, fitted_abs[0]["initial_value"] + target_abs[1]["result"])
        expected_events = [
            ("product", 0), ("product", 1), ("abs", "targets"), ("product", 2),
            ("abs", "fitted"), ("abs", "targets"), ("product", 3), ("abs", "projected")]
    assert events == expected_events
    # Later use of shared absolute targets must not overwrite earlier operands.
    for call in products:
        assert snapshot(call["left"]) == call["left_before"]
        assert snapshot(call["right"]) == call["right_before"]
        assert snapshot(call["result"]) == call["result_before"]


@pytest.mark.parametrize("kind,readonly", [
    ("c", False), ("fortran", False), ("strided", False),
    ("c", True), ("fortran", True),
])
def test_isolated_temporary_change_preserves_products_masks_and_inputs(
        monkeypatch, kind, readonly):
    design, targets, coefficients, expected = ordinary_problem()
    exercise(monkeypatch, *(layout(value, kind, readonly)
             for value in (design, targets, coefficients)), expected)


@pytest.mark.parametrize("case", ["zero_rhs", "zero_design", "empty_rhs", "tiny_rhs", "signed_zero"])
def test_zero_empty_and_tiny_qualification_contracts(monkeypatch, case):
    if case == "tiny_rhs":
        design = np.eye(2)
        tiny = np.nextafter(0.0, 1.0)
        coefficients = np.array([[tiny, 0.], [tiny, 0.]])
        targets = coefficients.copy()
        # The exact subnormal solution still fails the existing representability
        # safeguard; the zero RHS remains valid. Do not weaken that safeguard.
        expected = np.array([False, True])
    else:
        design = design_matrix()
        columns = 0 if case == "empty_rhs" else 2
        targets = np.zeros((4, columns))
        coefficients = np.zeros((2, columns))
        expected = np.ones(columns, dtype=bool)
        if case == "zero_design":
            design[:] = 0.0
            targets[:, 0] = 1.0
            expected[0] = False
        elif case == "signed_zero":
            targets[::2, 0] = -0.0
            coefficients[0, 0] = -0.0
    targets.setflags(write=False)
    exercise(monkeypatch, design, targets, coefficients, expected)


@pytest.mark.parametrize("magnitude,expected_mask", [
    (np.finfo(float).tiny / 2, False),
    (np.finfo(float).tiny, True),
    (np.ldexp(1.0, 400), True),
    (np.ldexp(1.0, 1022), True),
    (np.ldexp(1.0, 1023), False),
])
def test_roundoff_range_and_representability_are_unchanged(
        monkeypatch, magnitude, expected_mask):
    # An exact identity solution still needs the existing finite-roundoff and
    # normal representability guards. Overflow in abs(fitted)+abs(targets)
    # rejects qualification even though the fitted value itself is finite.
    design = np.eye(2)
    targets = np.full((2, 1), magnitude)
    coefficients = targets.copy()
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exercise(monkeypatch, design, targets, coefficients,
                 np.array([expected_mask]))


def test_signed_targets_with_exact_cancellation_keep_original_space_kkt(monkeypatch):
    design = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    targets = np.array([[2., -1.], [4., 3.], [-2., -1.], [-4., -3.]])
    coefficients = np.zeros((2, 2))
    coefficients[0, 0] = -0.0
    # Both column sums are zero or negative for each nonnegative coordinate,
    # so z=0 is optimal; absolute target scales remain strictly positive.
    exercise(monkeypatch, design, targets, coefficients, np.array([True, True]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficient_is_never_qualified(monkeypatch, bad):
    design = np.eye(2)
    targets = np.zeros((2, 1))
    coefficients = np.array([[bad], [0.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        exercise(monkeypatch, design, targets, coefficients, np.array([False]))


@pytest.mark.parametrize("reference_raises", [False, True])
def test_row_qualification_failure_keeps_original_fallback_cap(
        monkeypatch, reference_raises):
    design, targets, candidate, _ = ordinary_problem()
    rows = targets.T.copy()
    before = rows.copy()
    truth = rows[:, :2].copy()
    calls = []
    monkeypatch.setattr(_nnls, "_eligible_design", lambda value: True)

    def block(current_design, current_targets, gram, cap, *, matmul):
        assert current_design is design and cap == 6
        np.testing.assert_array_equal(current_targets, targets)
        return candidate.copy(), np.ones(4, dtype=bool), reference_product(design.T, targets)

    def scipy(current_design, rejected_rows, cap):
        assert current_design is design
        calls.append((rejected_rows.copy(), cap))
        if reference_raises:
            raise RuntimeError("reference iteration cap marker")
        return rejected_rows[:, :2].copy()

    monkeypatch.setattr(_nnls, "_batched_block", block)
    monkeypatch.setattr(_nnls, "_scipy_rows", scipy)
    if reference_raises:
        with pytest.raises(RuntimeError, match="reference iteration cap marker"):
            _nnls._project_unscaled(
                design, rows, "batched", 13, 4, matmul=reference_product)
    else:
        actual = _nnls._project_unscaled(
            design, rows, "batched", 13, 4, matmul=reference_product)
        np.testing.assert_array_equal(actual, truth)
        assert actual.flags.owndata and not np.shares_memory(actual, rows)
    assert len(calls) == 1 and calls[0][1] == 13
    np.testing.assert_array_equal(calls[0][0], rows[[2, 3]])
    np.testing.assert_array_equal(rows, before)


def test_qualification_product_error_keeps_whole_block_fallback(monkeypatch):
    design, targets, candidate, _ = ordinary_problem()
    rows = targets.T.copy()
    count = 0
    fallback = []
    monkeypatch.setattr(_nnls, "_eligible_design", lambda value: True)
    monkeypatch.setattr(
        _nnls, "_batched_block",
        lambda *args, **kwargs: (candidate.copy(), np.ones(4, dtype=bool),
                               reference_product(args[0].T, args[1])))

    def product(left, right):
        nonlocal count
        count += 1
        if count == 2:  # Gram succeeds; the qualification fitted product fails.
            raise FloatingPointError("qualification product marker")
        return reference_product(left, right)

    def scipy(current_design, current_rows, cap):
        fallback.append((current_rows.copy(), cap))
        return current_rows[:, :2].copy()

    monkeypatch.setattr(_nnls, "_scipy_rows", scipy)
    actual = _nnls._project_unscaled(design, rows, "batched", 13, 4, matmul=product)
    assert count == 2 and len(fallback) == 1 and fallback[0][1] == 13
    np.testing.assert_array_equal(fallback[0][0], rows)
    np.testing.assert_array_equal(actual, rows[:, :2])


def test_qr_rejection_preserves_whole_block_retry_and_original_cap(monkeypatch):
    design = np.vstack([np.eye(2), np.zeros((2, 2))])
    truth = np.tile(np.array([[1., 2.], [3., 0.]]), (16, 1))
    rows = reference_product(design, truth.T).T.copy()
    before = rows.copy()
    retries = []
    monkeypatch.setattr(_nnls.linalg, "qr", lambda *a, **k: (design.copy(), np.eye(2)))
    monkeypatch.setattr(_nnls.np.linalg, "cond", lambda value: 1.0)

    def reduced(current_design, reduced_rows, cap):
        assert cap == 13
        np.testing.assert_array_equal(current_design, np.eye(2))
        result = reduced_rows.copy()
        result[3, 0] = -1.0
        return result

    def retry(current_design, current_rows, solver, cap, batch_size, *, matmul):
        assert current_design is design
        retries.append((current_rows.copy(), solver, cap, batch_size))
        return current_rows[:, :2].copy()

    monkeypatch.setattr(_nnls, "_scipy_rows", reduced)
    monkeypatch.setattr(_nnls, "_project_unscaled", retry)
    actual = _nnls._project_qr(design, rows, 13, 32, matmul=reference_product)
    np.testing.assert_array_equal(actual, truth)
    assert len(retries) == 1 and retries[0][1:] == ("auto", 13, 32)
    np.testing.assert_array_equal(retries[0][0], rows)
    np.testing.assert_array_equal(rows, before)
