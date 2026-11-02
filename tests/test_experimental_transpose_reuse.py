"""Fit-local C transpose reuse together with prepared native RHS roles.

Lifecycle stubs deliberately hold factors fixed and do not claim NNLS accuracy.
Real-solver checks retain ordinary validation; separate qualified numerical and
real-data comparisons establish arithmetic equivalence. Every explicit product
in these tests uses einsum reference arithmetic, including the padded route.
"""
import gc
from pathlib import Path
import pickle
import sys
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"
assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"


def reference(left, right):
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def problem():
    return np.array([[1., 0., 2.], [0., 2., 1.], [2., 1., 0.], [1., 2., 3.]])


def metadata(value):
    return type(value), value.shape, value.strides, value.dtype, value.flags.writeable


class ArraySubclass(np.ndarray):
    pass


def layout(kind, tmp_path):
    values = problem()
    if kind == "fortran":
        data = np.array(values, order="F")
    elif kind == "strided":
        storage = np.zeros((8, 6))
        data = storage[::2, ::2]
        data[:] = values
    elif kind == "integer":
        data = values.astype(np.int32)
    elif kind == "float32":
        data = values.astype(np.float32)
    elif kind == "subclass":
        data = values.view(ArraySubclass)
    elif kind == "memmap":
        path = tmp_path / "readonly-training.npy"
        np.save(path, values)
        data = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        data = values.copy()
    if kind in ("readonly", "fortran", "strided"):
        data.setflags(write=False)
    return data


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(params=["ordinary", "padded"])
def routed_setup(monkeypatch, request):
    observed = {"route": request.param, "transpose": [], "normalized": [], "native_rhs": []}

    class ObservedNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def ascontiguousarray(self, value, *args, **kwargs):
            result = np.ascontiguousarray(value, *args, **kwargs)
            observed["transpose"].append(weakref.ref(result))
            return result

    class SafeNativeNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def matmul(self, left, right):
            return reference(left, right)

    def initialize(data, rank, **kwargs):
        assert rank == 1
        observed["normalized"].append(data)
        return np.ones((len(data), 1)), np.ones((1, data.shape[1]))

    original_preparation = experimental._PreparedRightOperand

    def prepare(right):
        result = original_preparation(right)
        observed["native_rhs"].append((weakref.ref(result), weakref.ref(result._right)))
        return result

    monkeypatch.setattr(experimental, "np", ObservedNumpy())
    monkeypatch.setattr(_aocl, "np", SafeNativeNumpy())
    operation = _aocl.padded_matmul if request.param == "padded" else reference
    monkeypatch.setattr(experimental, "nnls_matmul", lambda: operation)
    monkeypatch.setattr(experimental, "_initialize_nmf", initialize)
    monkeypatch.setattr(experimental, "_PreparedRightOperand", prepare)
    return observed


def install_lifecycle_step(monkeypatch, calls, *, failure=None, retain=True):
    """Exercise both dispatch routes and real cross preparation, not NNLS math."""
    def step(design, targets, *, init=None, cross_rhs=None, **kwargs):
        assert init is not None
        normalized_design = experimental._array(design)
        normalized_target = experimental._array(targets)
        if cross_rhs is not None:
            cross_rhs._cross_product(normalized_design.T, normalized_target,
                                     experimental.nnls_matmul())
        calls.append(targets if retain else weakref.ref(targets))
        if failure == "raise":
            raise RuntimeError("transpose lifecycle marker")
        info = {"success": failure != "w_failure", "n_iter": 0,
                "direct_lstsq_groups": 0, "single_pivots": 0,
                "projected_gradient_relative": 0.0}
        return np.array(init, order="C", copy=True), info
    monkeypatch.setattr(experimental, "_nnls_block_pivot", step)


@pytest.mark.parametrize("kind", ["c", "fortran", "strided", "readonly", "integer",
                                 "float32", "subclass", "memmap"])
def test_one_c_target_reaches_every_w_solve_and_preserves_input(
        monkeypatch, routed_setup, kind, tmp_path):
    data = layout(kind, tmp_path)
    original, original_metadata = np.array(data, copy=True), metadata(data)
    calls = []
    install_lifecycle_step(monkeypatch, calls)
    model = experimental.ANLSBPP(1, max_iter=3, tol=0)
    with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
        model.fit_transform(data)
    assert model.n_iter_ == 3 and len(calls) == 6
    assert len(routed_setup["transpose"]) == 1
    normalized, prepared = routed_setup["normalized"][0], calls[0]
    assert prepared is routed_setup["transpose"][0]()
    assert prepared.dtype == np.float64 and prepared.flags.c_contiguous
    assert not np.shares_memory(prepared, normalized)
    np.testing.assert_array_equal(prepared, normalized.T)
    assert all(target is prepared for target in calls[::2])
    assert all(target is normalized for target in calls[1::2])
    assert len(routed_setup["native_rhs"]) == (2 if routed_setup["route"] == "padded" else 0)
    np.testing.assert_array_equal(data, original)
    assert metadata(data) == original_metadata
    assert all(value is not prepared for value in vars(model).values())


@pytest.mark.parametrize("routed_setup", ["ordinary"], indirect=True)
def test_real_public_w_solver_validates_the_same_c_target_on_every_solve(
        monkeypatch, routed_setup):
    original_core, original_array = experimental._nnls_block_pivot, experimental._array
    active, targets, validations = {"target": None}, [], []

    def checked_array(value, **kwargs):
        result = original_array(value, **kwargs)
        if value is active["target"]:
            validations.append((value, result))
        return result

    def core(design, target, **kwargs):
        is_w = target.shape == problem().T.shape
        active["target"] = target if is_w else None
        if is_w:
            targets.append(target)
        try:
            return original_core(design, target, **kwargs)
        finally:
            active["target"] = None

    monkeypatch.setattr(experimental, "_array", checked_array)
    monkeypatch.setattr(experimental, "_nnls_block_pivot", core)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = experimental.ANLSBPP(1, max_iter=3, tol=0)
        model.fit_transform(problem())
    assert all(issubclass(item.category, ConvergenceWarning) for item in caught)
    assert len(targets) >= 2 and len(validations) == len(targets)
    assert len(routed_setup["transpose"]) == 1
    assert all(target is targets[0] for target in targets)
    assert all(before is after is targets[0] for before, after in validations)
    assert model.nnls_diagnostics_["failed_solves"] == 0


def test_stationarity_and_early_errors_do_not_prepare_transpose(monkeypatch, routed_setup):
    def unexpected(*args, **kwargs):
        raise AssertionError("No inner solve is expected")
    monkeypatch.setattr(experimental, "_nnls_block_pivot", unexpected)
    model = experimental.ANLSBPP(1, max_iter=3)
    weights = model.fit_transform(np.ones((4, 3)))
    assert model.n_iter_ == 0 and model.converged_
    assert model.convergence_reason_ == "initial_stationarity"
    np.testing.assert_array_equal(weights, np.ones((4, 1)))
    with pytest.raises(ValueError):
        experimental.ANLSBPP(1, max_iter=0).fit_transform(problem())
    def fail_initializer(*args, **kwargs):
        raise RuntimeError("initializer marker")
    monkeypatch.setattr(experimental, "_initialize_nmf", fail_initializer)
    with pytest.raises(RuntimeError, match="initializer marker"):
        experimental.ANLSBPP(1).fit_transform(problem())
    assert routed_setup["transpose"] == routed_setup["native_rhs"] == []


def test_failed_w_update_prepares_only_reached_training_storage(monkeypatch, routed_setup):
    calls = []
    install_lifecycle_step(monkeypatch, calls, failure="w_failure")
    model = experimental.ANLSBPP(1, max_iter=3)
    with pytest.warns(ConvergenceWarning, match="inner_nnls_failure"):
        model.fit_transform(problem())
    assert len(calls) == len(routed_setup["transpose"]) == 1
    assert len(routed_setup["native_rhs"]) == (1 if routed_setup["route"] == "padded" else 0)
    assert model.n_iter_ == 0 and model.nnls_diagnostics_["failed_solves"] == 1


def test_refit_and_pickle_do_not_keep_transpose_or_native_cache(monkeypatch, routed_setup):
    calls = []
    install_lifecycle_step(monkeypatch, calls)
    model = experimental.ANLSBPP(1, max_iter=1, tol=0)
    for data in (problem(), problem() + .5):
        with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
            model.fit_transform(data)
    assert len(routed_setup["transpose"]) == 2 and len(calls) == 4
    assert calls[0] is not calls[2]
    np.testing.assert_array_equal(calls[0], problem().T)
    np.testing.assert_array_equal(calls[2], (problem() + .5).T)
    assert all(value is not calls[0] and value is not calls[2] for value in vars(model).values())
    assert all("transpose" not in name and "rhs" not in name for name in vars(model))
    restored = pickle.loads(pickle.dumps(model))
    assert vars(restored).keys() == vars(model).keys()
    assert restored.get_params() == model.get_params()
    np.testing.assert_array_equal(restored.components_, model.components_)
    np.testing.assert_array_equal(restored.training_coordinates_, model.training_coordinates_)


@pytest.mark.parametrize("failure", [None, "raise"])
def test_fit_local_transpose_and_native_buffers_release_on_return_or_exception(
        monkeypatch, routed_setup, failure):
    calls = []
    install_lifecycle_step(monkeypatch, calls, failure=failure, retain=False)
    def run_and_drop_traceback():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                experimental.ANLSBPP(1, max_iter=1, tol=0).fit_transform(problem())
        except RuntimeError as error:
            assert failure == "raise" and str(error) == "transpose lifecycle marker"
        else:
            assert failure is None
    run_and_drop_traceback()
    gc.collect()
    assert len(routed_setup["transpose"]) == 1
    assert routed_setup["transpose"][0]() is None and calls[0]() is None
    assert all(role() is None and buffer() is None for role, buffer in routed_setup["native_rhs"])


@pytest.mark.parametrize("shape", [(1, 3), (4, 1)])
def test_singleton_target_reuses_existing_contiguous_storage(monkeypatch, routed_setup, shape):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape) + 2
    original = data.copy()
    calls = []
    install_lifecycle_step(monkeypatch, calls)
    with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
        experimental.ANLSBPP(1, max_iter=1, tol=0).fit_transform(data)
    assert len(routed_setup["transpose"]) == 1 and len(calls) == 2
    assert calls[0].flags.c_contiguous
    assert np.shares_memory(calls[0], routed_setup["normalized"][0])
    np.testing.assert_array_equal(calls[0], original.T)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("routed_setup", ["padded"], indirect=True)
def test_fit_role_retains_c_target_while_selector_drops_and_rebuilds_native_cache(
        monkeypatch, routed_setup):
    operations = [_aocl.padded_matmul, _aocl.padded_matmul, _aocl.padded_matmul,
                  reference, reference, _aocl.padded_matmul, _aocl.padded_matmul]
    selected, updates = [], []
    def select():
        operation = operations[len(selected)]
        selected.append(operation)
        return operation
    original_core = experimental._nnls_block_pivot
    def core(design, target, *, cross_rhs=None, **kwargs):
        result = original_core(design, target, cross_rhs=cross_rhs, **kwargs)
        if target.shape == problem().T.shape:
            updates.append((target, cross_rhs, cross_rhs._prepared))
        return result
    monkeypatch.setattr(experimental, "nnls_matmul", select)
    monkeypatch.setattr(experimental, "_nnls_block_pivot", core)
    with pytest.warns(ConvergenceWarning, match="outer_iteration_limit"):
        model = experimental.ANLSBPP(1, max_iter=3, tol=0)
        model.fit_transform(problem())
    assert len(selected) == 7 and len(updates) == 3
    assert model.n_iter_ == 3 and model.nnls_diagnostics_["failed_solves"] == 0
    assert len(routed_setup["transpose"]) == 1
    assert all(target is updates[0][0] and role is updates[0][1] for target, role, _ in updates)
    assert updates[0][2] is not None and updates[1][2] is None
    assert updates[2][2] is not None and updates[2][2] is not updates[0][2]
    assert len(routed_setup["native_rhs"]) == 4
    np.testing.assert_array_equal(updates[0][0], problem().T)
