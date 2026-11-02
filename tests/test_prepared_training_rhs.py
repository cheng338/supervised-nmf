"""Prepared ANLS training RHS: native geometry, role lifetime and solver contracts.

Reference routing uses einsum without optimization, so it never supplies an
unprotected native GEMM oracle. Lifecycle stubs test control flow only; real
NNLS and complete scientific comparisons are separate checks.
"""
import gc
import inspect
from pathlib import Path
import pickle
import sys
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"


def reference(left, right):
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def layout(values, kind):
    if kind == "f":
        result = np.array(values, order="F", copy=True)
    elif kind == "strided":
        storage = np.empty((values.shape[0] * 2, values.shape[1] * 2))
        result = storage[::2, ::2]
        result[:] = values
    else:
        result = np.array(values, order="C", copy=True)
    result.setflags(write=False)
    return result


def metadata(value):
    return value.shape, value.strides, value.dtype, value.flags.writeable


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def safe_padding(monkeypatch):
    """Observe exact native arguments, while substituting scalar-style math."""
    calls = []

    class NumericalProxy:
        def __getattr__(self, name):
            return getattr(np, name)

        def matmul(self, left, right):
            output = reference(left, right)
            calls.append((left, right, output))
            return output

    monkeypatch.setattr(_aocl, "np", NumericalProxy())
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: _aocl.padded_matmul)
    return calls


@pytest.fixture
def prepared_observer(monkeypatch):
    events = []
    original = experimental._PreparedRightOperand

    def prepare(right):
        result = original(right)
        events.append({"values": right.copy(), "input_metadata": metadata(right),
                       "object": weakref.ref(result), "buffer": weakref.ref(result._right)})
        return result

    monkeypatch.setattr(experimental, "_PreparedRightOperand", prepare)
    return events


def unit_initializer(data, rank, **kwargs):
    assert rank == 1
    return np.ones((len(data), 1)), np.ones((1, data.shape[1]))


def problem():
    return np.array([[1., 0., 2.], [0., 2., 1.], [2., 1., 0.], [1., 2., 3.]])


def inner_info(success=True):
    return {"success": success, "n_iter": 0, "direct_lstsq_groups": 0,
            "single_pivots": 0, "projected_gradient_relative": 0.0}


@pytest.mark.parametrize("m,k,n", [(1, 7, 1), (7, 5, 8), (8, 5, 7),
                                   (9, 13, 9), (15, 3, 17), (17, 9, 15)])
@pytest.mark.parametrize("kind", ["c", "f", "strided"])
def test_prepared_native_arguments_equal_generic_padding_and_reuse_only_rhs(
        safe_padding, m, k, n, kind):
    left = layout((np.arange(m * k).reshape(m, k) % 19 - 9.) / 8, kind)
    right = layout((np.arange(k * n).reshape(k, n) % 13 - 6.) / 4, kind)
    originals = left.copy(), right.copy(), metadata(left), metadata(right)
    expected = _aocl.padded_matmul(left, right)
    # The generic two-product branch does not change prepared full padding.
    split = m >= 8 and n >= 8 and m % 8 and n % 8 and kind == "c"
    assert len(safe_padding) == (2 if split else 1)
    generic_left = np.zeros((((m + 7) // 8) * 8, k))
    generic_right = np.zeros((k, ((n + 7) // 8) * 8))
    generic_left[:m] = left
    generic_right[:, :n] = right
    prepared = _aocl._PreparedRightOperand(right)
    first = prepared.matmul(left)
    second_left = np.array(left * 2, order="F")
    second = prepared.matmul(second_left)
    actual_left, actual_right, actual_output = safe_padding[-2]
    later_left, later_right, later_output = safe_padding[-1]

    np.testing.assert_array_equal(actual_left, generic_left)
    np.testing.assert_array_equal(actual_right, generic_right)
    assert actual_left.shape == (((m + 7) // 8) * 8, k)
    assert actual_right.shape == (k, ((n + 7) // 8) * 8)
    assert actual_left.dtype == actual_right.dtype == np.float64
    assert actual_left.flags.c_contiguous and actual_right.flags.c_contiguous
    assert actual_right.flags.owndata and not np.shares_memory(actual_right, right)
    assert actual_right is later_right is prepared._right
    assert actual_left is not later_left
    assert not np.shares_memory(actual_left, later_left)
    np.testing.assert_array_equal(later_left[:m], second_left)
    assert np.count_nonzero(actual_left[m:]) == 0
    assert np.count_nonzero(actual_right[:, n:]) == 0
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, reference(second_left, right))
    for result, native in ((first, actual_output), (second, later_output)):
        assert result.flags.c_contiguous and result.flags.owndata
        assert not np.shares_memory(result, native)
        assert not np.shares_memory(result, prepared._right)
    first[:] = -99
    np.testing.assert_array_equal(second, reference(second_left, right))
    np.testing.assert_array_equal(left, originals[0])
    np.testing.assert_array_equal(right, originals[1])
    assert metadata(left) == originals[2] and metadata(right) == originals[3]


@pytest.mark.parametrize("m,k,n", [(1, 7, 1), (7, 13, 8), (8, 13, 7),
                                   (9, 13, 9), (15, 17, 17), (17, 17, 15)])
@pytest.mark.parametrize("exponent", [-200, 0, 200])
def test_native_prepared_product_matches_existing_protected_product_exactly(m, k, n, exponent):
    rng = np.random.default_rng(55103 + m + n)
    left = layout(np.ldexp(rng.integers(-7, 8, (m, k)).astype(float), exponent), "f")
    right = layout(np.ldexp(rng.integers(-7, 8, (k, n)).astype(float), exponent), "strided")
    prepared = _aocl._PreparedRightOperand(right)
    for current_left in (left, -left):
        expected = _aocl.padded_matmul(current_left, right)
        actual = prepared.matmul(current_left)
        np.testing.assert_array_equal(actual, expected)
        assert actual.tobytes(order="C") == expected.tobytes(order="C")
        np.testing.assert_array_equal(actual, reference(current_left, right))
        assert np.isfinite(actual).all()


def test_private_prepared_operation_rejects_wrong_k_and_fully_aligned_calls(safe_padding):
    prepared = _aocl._PreparedRightOperand(np.ones((5, 8)))
    with pytest.raises(ValueError, match="inner dimension"):
        prepared.matmul(np.ones((7, 6)))
    with pytest.raises(ValueError, match="genuinely padded"):
        prepared.matmul(np.ones((8, 5)))
    assert safe_padding == []


def test_roles_bind_distinct_targets_and_update_left_values(safe_padding, prepared_observer):
    design = np.array([[1., 0.], [0., 1.], [2., 0.], [0., 2.], [0., 0.]])
    first_truth = np.array([[1., 2., 0.], [3., 0., 1.]])
    first_target = reference(design, first_truth)
    second_target = first_target + reference(design, np.ones_like(first_truth))
    roles = [experimental._FixedTrainingRHS(value) for value in (first_target, second_target)]
    for role, targets in zip(roles, (first_target, second_target)):
        for current_design in (design, design * 2):
            result, info = role.solve(current_design)
            expected, expected_info = experimental.nnls_block_pivot(current_design, targets)
            np.testing.assert_array_equal(result, expected)
            assert info == expected_info and info["success"]
    assert len(prepared_observer) == 2
    np.testing.assert_array_equal(prepared_observer[0]["values"], first_target)
    np.testing.assert_array_equal(prepared_observer[1]["values"], second_target)
    assert roles[0]._prepared is not roles[1]._prepared
    assert roles[0]._targets is first_target and roles[1]._targets is second_target


def test_each_solve_honors_backend_change_and_invalidates_preparation(
        monkeypatch, safe_padding, prepared_observer):
    design = np.ones((5, 1))
    target = np.tile(np.array([[1., 2., 3.]]), (5, 1))
    role = experimental._FixedTrainingRHS(target)
    operations = iter([_aocl.padded_matmul, reference, _aocl.padded_matmul])
    selections = []

    def select():
        chosen = next(operations)
        selections.append(chosen)
        return chosen

    monkeypatch.setattr(experimental, "nnls_matmul", select)
    first, _ = role.solve(design)
    original = role._prepared
    second, _ = role.solve(design)
    assert role._prepared is None
    third, _ = role.solve(design)
    assert role._prepared is not original
    assert len(selections) == 3 and len(prepared_observer) == 2
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)


def test_fully_aligned_cross_never_prepares(safe_padding, prepared_observer):
    design = np.eye(8)
    targets = np.arange(64, dtype=float).reshape(8, 8)
    role = experimental._FixedTrainingRHS(targets)
    result, info = role.solve(design)
    assert info["success"]
    np.testing.assert_array_equal(result, targets)
    assert role._prepared is None and prepared_observer == []


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "tol", "cap"])
def test_validation_precedes_preparation(monkeypatch, safe_padding, prepared_observer, failure):
    targets = np.ones((5, 3))
    if failure == "nonfinite":
        targets[0, 0] = np.nan
    role = experimental._FixedTrainingRHS(targets)
    design = np.ones((4 if failure == "shape" else 5, 1))
    kwargs = {"tol": 0} if failure == "tol" else {"max_iter": 0} if failure == "cap" else {}
    with pytest.raises(ValueError):
        role.solve(design, **kwargs)
    assert role._prepared is None and prepared_observer == []


def test_overflow_still_rejected_after_prepared_cross(safe_padding, prepared_observer):
    role = experimental._FixedTrainingRHS(np.full((5, 3), 1e308))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="products overflowed"):
            role.solve(np.full((5, 1), 1e308))
    assert len(prepared_observer) == 1


@pytest.mark.parametrize("kind", ["c", "f", "strided"])
def test_real_fit_prepares_two_roles_and_validates_one_reused_c_transpose(
        monkeypatch, safe_padding, prepared_observer, kind):
    data = layout(problem(), kind)
    original_data, original_metadata = data.copy(), metadata(data)
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    original_array = experimental._array
    transposes = []

    def observed_array(value, **kwargs):
        result = original_array(value, **kwargs)
        if value.shape == data.T.shape:
            transposes.append((value, result))
        return result

    monkeypatch.setattr(experimental, "_array", observed_array)
    model = experimental.ANLSBPP(1, max_iter=3, tol=0)
    with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
        model.fit_transform(data)
    assert model.n_iter_ == 3 and model.nnls_diagnostics_["failed_solves"] == 0
    assert len(prepared_observer) == 2 and len(transposes) == 3
    np.testing.assert_array_equal(prepared_observer[0]["values"], data.T)
    np.testing.assert_array_equal(prepared_observer[1]["values"], data)
    # The caller now prepares C-order storage once; ordinary target validation
    # remains on every W solve and returns that same already-normalized array.
    target = transposes[0][0]
    assert target.flags.c_contiguous and not np.shares_memory(target, data)
    assert all(before is after is target for before, after in transposes)
    np.testing.assert_array_equal(data, original_data)
    assert metadata(data) == original_metadata
    assert not any(isinstance(value, (experimental._FixedTrainingRHS, _aocl._PreparedRightOperand))
                   for value in vars(model).values())


def test_initial_stationarity_and_pre_solve_errors_do_not_prepare(
        monkeypatch, safe_padding, prepared_observer):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    model = experimental.ANLSBPP(1, max_iter=3)
    result = model.fit_transform(np.ones((4, 3)))
    assert model.converged_ and model.n_iter_ == 0
    assert model.convergence_reason_ == "initial_stationarity"
    np.testing.assert_array_equal(result, np.ones((4, 1)))
    assert prepared_observer == []
    with pytest.raises(ValueError):
        experimental.ANLSBPP(1, max_iter=0).fit_transform(problem())
    assert prepared_observer == []


@pytest.mark.parametrize("stop", ["one_sweep", "w_failure", "h_failure", "objective", "exception"])
def test_outer_exit_paths_prepare_only_reached_roles_and_keep_accepted_state(
        monkeypatch, safe_padding, prepared_observer, stop):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    entered = []

    def lifecycle_step(self, design, *, init=None, **kwargs):
        # Invoke the real role-specific cross preparation, then a documented
        # control-flow stub; this is not an NNLS numerical oracle.
        normalized_design = experimental._array(design)
        normalized_target = experimental._array(self._targets)
        self._cross_product(normalized_design.T, normalized_target, experimental.nnls_matmul())
        entered.append(self)
        if stop == "exception":
            raise RuntimeError("role failure marker")
        success = not ((stop == "w_failure" and len(entered) == 1)
                       or (stop == "h_failure" and len(entered) == 2))
        coefficients = np.array(init, order="C", copy=True)
        if stop == "objective" and len(entered) == 2:
            coefficients *= 100
        return coefficients, inner_info(success)

    monkeypatch.setattr(experimental._FixedTrainingRHS, "solve", lifecycle_step)
    model = experimental.ANLSBPP(1, max_iter=1 if stop == "one_sweep" else 3, tol=0)
    if stop == "exception":
        with pytest.raises(RuntimeError, match="role failure marker"):
            model.fit_transform(problem())
        assert len(entered) == len(prepared_observer) == 1
        return
    reason = ("outer_iteration_limit" if stop == "one_sweep" else
              "objective_increase_or_nonfinite" if stop == "objective" else "inner_nnls_failure")
    with pytest.warns(ConvergenceWarning, match=reason):
        result = model.fit_transform(problem())
    expected_count = 1 if stop == "w_failure" else 2
    assert len(entered) == len(prepared_observer) == expected_count
    assert model.n_iter_ == (1 if stop == "one_sweep" else 0)
    assert model.convergence_reason_ == reason
    np.testing.assert_array_equal(result, np.ones((4, 1)))
    np.testing.assert_array_equal(model.components_, np.ones((1, 3)))


def test_refits_have_new_buffers_and_no_cached_state_in_clone_or_pickle(
        monkeypatch, safe_padding, prepared_observer):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    model = experimental.ANLSBPP(1, max_iter=1, tol=0)
    for data in (problem(), problem() + .5):
        with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
            model.fit_transform(data)
    assert len(prepared_observer) == 4
    np.testing.assert_array_equal(prepared_observer[2]["values"], (problem() + .5).T)
    np.testing.assert_array_equal(prepared_observer[3]["values"], problem() + .5)
    restored = pickle.loads(pickle.dumps(model))
    assert vars(restored).keys() == vars(model).keys()
    assert clone(model).get_params() == model.get_params()
    assert all("prepared" not in name and "rhs" not in name for name in vars(model))
    # The native spy retains arrays for argument inspection; object lifetime is
    # independent of those references. Drop them before checking buffer lifetime.
    safe_padding.clear()
    gc.collect()
    assert all(event["object"]() is None and event["buffer"]() is None
               for event in prepared_observer)


def test_standalone_transform_diagnostics_and_ordinary_backend_do_not_prepare(
        monkeypatch, safe_padding, prepared_observer):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: reference)
    model = experimental.ANLSBPP(1, max_iter=1, tol=0)
    with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
        model.fit_transform(problem())
    assert prepared_observer == []
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: _aocl.padded_matmul)
    model.transform(problem())
    experimental.projected_gradient_kkt(problem(), model.training_coordinates_, model.components_)
    experimental.nnls_block_pivot(np.ones((4, 1)), problem())
    assert prepared_observer == []


def test_prepared_buffers_release_after_fit_exception(
        monkeypatch, safe_padding, prepared_observer):
    monkeypatch.setattr(experimental, "_initialize_nmf", unit_initializer)

    def failing_solve(self, design, *, init=None, **kwargs):
        normalized = experimental._array(self._targets)
        self._cross_product(experimental._array(design).T, normalized,
                            experimental.nnls_matmul())
        raise RuntimeError("lifetime marker")

    monkeypatch.setattr(experimental._FixedTrainingRHS, "solve", failing_solve)

    def run_and_drop_traceback():
        try:
            experimental.ANLSBPP(1).fit_transform(problem())
        except RuntimeError as error:
            assert str(error) == "lifetime marker"
        else:
            raise AssertionError("Expected the deliberate failure")

    run_and_drop_traceback()
    assert len(prepared_observer) == 1
    safe_padding.clear()
    gc.collect()
    assert prepared_observer[0]["object"]() is None
    assert prepared_observer[0]["buffer"]() is None


def test_public_signature_and_warning_origin_preserved(monkeypatch, safe_padding):
    signature = inspect.signature(experimental.nnls_block_pivot)
    assert list(signature.parameters) == ["A", "B", "init", "max_iter", "tol"]
    assert all(signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
               for name in ("init", "max_iter", "tol"))
    design = np.array([[1., 1.], [0., 1.]])
    targets = np.array([[1., 2.], [3., 6.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = experimental.nnls_block_pivot(design, targets, max_iter=1)
    assert isinstance(result, tuple) and len(result) == 2
    assert result[1]["success"] is False
    assert set(result[1]) == {"success", "n_iter", "max_iter", "grouped_solves",
                             "direct_lstsq_groups", "single_pivots", "projected_gradient_relative",
                             "kkt_tolerance", "unclipped_primal_violation", "primal_violation",
                             "dual_violation", "complementarity_max", "residual_norm"}
    relevant = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    assert len(relevant) == 1 and Path(relevant[0].filename).resolve() == Path(__file__).resolve()
    role = experimental._FixedTrainingRHS(targets)
    with warnings.catch_warnings(record=True) as role_warnings:
        warnings.simplefilter("always")
        prepared_result = role.solve(design, max_iter=1)
    np.testing.assert_array_equal(prepared_result[0], result[0])
    assert prepared_result[1] == result[1]
    relevant = [item for item in role_warnings if issubclass(item.category, ConvergenceWarning)]
    assert len(relevant) == 1 and Path(relevant[0].filename).resolve() == Path(__file__).resolve()
