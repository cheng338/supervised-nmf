"""Caller-owned experimental projection receipts and unchanged fitted state."""

from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import inspect
import json
from pathlib import Path
import pickle
import sys
from threading import Event, local
import warnings

import numpy as np
import pytest
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.estimator_checks import check_dict_unchanged
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import experimental_nmf as experimental

assert Path(experimental.__file__).resolve() == ROOT / "scripts/experimental_nmf.py"


@pytest.fixture(autouse=True)
def one_native_thread():
    # The main thread sets process-wide native controls before any workers exist.
    with threadpool_limits(limits=1):
        yield


def prepared(basis=None, *, cap=None, legacy=False):
    basis = np.array([[1., 0.], [1., 1.]]) if basis is None else np.array(basis, dtype=float)
    model = experimental.ANLSBPP(len(basis), nnls_max_iter=cap)
    model.components_ = basis.copy()
    model.n_features_in_, model.n_components_ = basis.shape[1], basis.shape[0]
    model.training_coordinates_ = np.zeros((3, len(basis)))
    model.converged_ = True
    model.nnls_diagnostics_ = {"solves": 17, "failed_solves": 0}
    if legacy:
        model.last_transform_converged_ = False
        model.last_transform_diagnostics_ = {"success": True, "old_receipt": "historical"}
    return model


def frozen(model):
    return pickle.dumps(model, protocol=5)


def capture(model, query):
    with experimental.capture_transform_diagnostics(model) as receipt:
        result = model.transform(query)
    return result, receipt


def assert_success(receipt):
    assert receipt["performed"] is True
    assert receipt["converged"] is True
    assert receipt["diagnostics"]["success"] is True
    assert receipt["error"] is None
    json.dumps(receipt, allow_nan=False)


def assert_failure(receipt, error_type, text=None):
    assert receipt["performed"] is True
    assert receipt["converged"] is False
    assert receipt["diagnostics"] is None
    assert receipt["error"].startswith(error_type.__name__ + ":")
    if text:
        assert text in receipt["error"]
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("basis", [np.eye(2), [[1., 0.], [1., 1.]], [[1., 2.], [2., 4.]]],
                         ids=["identity", "nonorthogonal", "dependent"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_capture_matches_independent_nnls_and_retains_prior_receipts(basis, dtype, layout):
    model = prepared(basis, legacy=True)
    x = np.array([[1., 3.], [2., 6.], [0., 0.]], dtype=dtype, order="F")
    if layout == "strided":
        padded = np.empty((3, 4), dtype=dtype)
        padded[:, ::2] = x
        x = padded[:, ::2]
    x.setflags(write=False)
    state, original = frozen(model), x.copy()
    expected = np.array([nnls(model.components_.T, row.astype(float))[0] for row in x])
    actual, first = capture(model, x)
    assert_success(first)
    # Reconstruct through einsum so the reference itself is backend-independent.
    product = lambda weights: np.einsum("ik,kj->ij", weights, model.components_, optimize=False)
    np.testing.assert_allclose(product(actual), product(expected), rtol=1e-11, atol=1e-11)
    if np.linalg.matrix_rank(model.components_) == 2:
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    saved = deepcopy(first)
    again, second = capture(model, x * .5)
    assert_success(second)
    assert first is not second and first["diagnostics"] is not second["diagnostics"]
    np.testing.assert_allclose(again, actual * .5, rtol=1e-11, atol=1e-11)
    assert first == saved
    assert frozen(model) == state
    np.testing.assert_array_equal(x, original)
    assert not x.flags.writeable


def test_no_capture_leaves_state_unchanged_and_does_not_retain_arrays():
    model = prepared()
    state = frozen(model)
    x = np.array([[1., 3.], [2., 6.]])
    result = model.transform(x)
    assert frozen(model) == state
    assert not any(name.startswith("last_transform_") for name in vars(model))
    assert result.shape == (2, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        model.transform(-x)
    assert frozen(model) == state


@pytest.mark.parametrize("query", [
    np.array([[-1., 3.]]), np.array([[np.nan, 3.]]), np.array([[np.inf, 3.]]),
    np.ones((2, 3)), np.array([1., 3.]), np.empty((0, 2)), np.empty((2, 0)),
    np.array([[1. + 1j, 3.]]), np.array([["not", "numeric"]]), 1.,
], ids=["negative", "nan", "infinity", "width", "one_dimensional", "empty_rows",
        "empty_columns", "complex", "nonnumeric", "scalar"])
def test_input_failure_has_current_receipt_and_recovery(query):
    model = prepared(legacy=True)
    valid = np.array([[1., 3.], [2., 6.]])
    expected, first = capture(model, valid)
    prior = deepcopy(first)
    state = frozen(model)
    with experimental.capture_transform_diagnostics(model) as failed:
        with pytest.raises(ValueError):
            model.transform(query)
    assert_failure(failed, ValueError)
    assert frozen(model) == state and first == prior
    restored, recovered = capture(model, valid)
    assert_success(recovered)
    np.testing.assert_array_equal(restored, expected)
    assert failed["diagnostics"] is None and first == prior


@pytest.mark.parametrize("name,value", [
    ("nnls_max_iter", 0), ("nnls_max_iter", -1), ("nnls_max_iter", True),
    ("nnls_max_iter", 1.5), ("nnls_tol", 0.), ("nnls_tol", -1.),
    ("nnls_tol", np.nan), ("nnls_tol", np.inf),
])
def test_invalid_solver_settings_are_current_failure_without_mutation(name, value):
    model = prepared(legacy=True)
    model.set_params(**{name: value})
    state = frozen(model)
    with experimental.capture_transform_diagnostics(model) as failed:
        with pytest.raises(ValueError):
            model.transform(np.array([[1., 3.]]))
    assert_failure(failed, ValueError)
    assert frozen(model) == state


@pytest.mark.parametrize("error", [RuntimeError("controlled solve"), KeyboardInterrupt("controlled interrupt")])
def test_solver_exception_and_interrupt_are_not_swallowed(monkeypatch, error):
    model = prepared(legacy=True)
    state = frozen(model)
    original = experimental.nnls_block_pivot

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(experimental, "nnls_block_pivot", fail)
    with experimental.capture_transform_diagnostics(model) as failed:
        with pytest.raises(type(error)) as caught:
            model.transform(np.array([[1., 3.]]))
    assert caught.value is error
    assert_failure(failed, type(error), str(error))
    assert frozen(model) == state
    monkeypatch.setattr(experimental, "nnls_block_pivot", original)
    _, recovered = capture(model, np.array([[1., 3.]]))
    assert_success(recovered)
    assert failed["diagnostics"] is None


def test_real_finite_product_overflow_preserves_model_and_receipt_boundary():
    model = prepared(legacy=True)
    state = frozen(model)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with experimental.capture_transform_diagnostics(model) as failed:
            with pytest.raises(ValueError, match="overflowed"):
                model.transform(np.array([[1e308, 1e308]]))
    assert_failure(failed, ValueError, "overflowed")
    assert frozen(model) == state


@pytest.mark.parametrize("policy", ["always", "ignore", "error"])
def test_real_cap_warning_policies_and_attribution(policy):
    model = prepared(cap=1, legacy=True)
    state = frozen(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, ConvergenceWarning)
        with experimental.capture_transform_diagnostics(model) as receipt:
            if policy == "error":
                with pytest.raises(ConvergenceWarning, match="inner NNLS failed"):
                    model.transform(np.array([[1., 3.], [2., 6.]]))
            else:
                values = model.transform(np.array([[1., 3.], [2., 6.]]))
    assert frozen(model) == state
    assert receipt["performed"] and receipt["converged"] is False
    if policy == "error":
        assert_failure(receipt, ConvergenceWarning, "inner NNLS failed")
    else:
        assert receipt["error"] is None and receipt["diagnostics"]["success"] is False
        assert values.shape == (2, 2) and np.all(values >= 0)
    if policy == "always":
        assert len(caught) == 1 and caught[0].category is ConvergenceWarning
        # The existing solver wrapper uses stacklevel=3, which attributes its
        # warning to the NNLS call in transform, not a diagnostic wrapper.
        assert Path(caught[0].filename).resolve() == Path(experimental.__file__).resolve()
        source, start = inspect.getsourcelines(inspect.unwrap(experimental.ANLSBPP.transform))
        assert start <= caught[0].lineno < start + len(source)
    else:
        assert not caught


def test_empty_nonowner_and_nonexperimental_capture_are_unknown():
    model, other = prepared(), prepared()
    expected = {"performed": False, "converged": None, "diagnostics": None, "error": None}
    with experimental.capture_transform_diagnostics(model) as record:
        assert record == expected
        other.transform(np.array([[1., 3.]]))
        assert record == expected
    assert record == expected
    with experimental.capture_transform_diagnostics(object()) as unsupported:
        assert unsupported is None


def test_unfitted_failure_is_recorded_without_creating_model_state():
    model = experimental.ANLSBPP(2)
    before = frozen(model)
    with experimental.capture_transform_diagnostics(model) as receipt:
        with pytest.raises(NotFittedError):
            model.transform(np.array([[1., 3.]]))
    assert_failure(receipt, NotFittedError)
    assert frozen(model) == before


@pytest.mark.parametrize("same_owner", [False, True])
def test_nested_capture_restores_outer_after_inner_exception(same_owner):
    model = prepared()
    other = model if same_owner else prepared()
    with experimental.capture_transform_diagnostics(model) as outer:
        before = deepcopy(outer)
        with pytest.raises(ValueError):
            with experimental.capture_transform_diagnostics(other) as inner:
                other.transform(np.array([[-1., 3.]]))
        assert outer == before
        assert_failure(inner, ValueError)
        model.transform(np.array([[1., 3.]]))
    assert_success(outer)
    assert_failure(inner, ValueError)


def test_capture_scope_restores_after_caller_raises_and_records_stay_detached():
    model = prepared()
    with pytest.raises(RuntimeError, match="caller"):
        with experimental.capture_transform_diagnostics(model) as record:
            model.transform(np.array([[1., 3.]]))
            raise RuntimeError("caller")
    assert_success(record)
    saved = deepcopy(record)
    model.transform(np.zeros((1, 2)))
    assert record == saved
    json.loads(json.dumps(record, allow_nan=False))


def test_legacy_pickle_clone_and_active_capture_do_not_transport_context():
    original = prepared(legacy=True)
    model = pickle.loads(pickle.dumps(original))
    before = frozen(model)
    with experimental.capture_transform_diagnostics(model) as record:
        restored = pickle.loads(pickle.dumps(model))
        restored.transform(np.array([[1., 3.]]))
        assert record["performed"] is False
        model.transform(np.array([[1., 3.]]))
    assert_success(record)
    assert frozen(model) == before
    assert frozen(restored) == before
    cloned = clone(model)
    assert not hasattr(cloned, "components_")
    assert not any(name.startswith("last_transform_") for name in vars(cloned))
    assert cloned.get_params() == model.get_params()


@pytest.mark.parametrize("refit", ["fit", "fit_transform", "invalid"])
def test_refit_preserves_existing_reset_semantics(refit):
    model = prepared(legacy=True)
    data = np.arange(1., 7.)[:, None] * np.array([[1., 2.]])
    model.set_params(n_components=1)
    if refit == "invalid":
        with pytest.raises(ValueError, match="nonnegative"):
            model.fit(-data)
        assert not hasattr(model, "components_")
    else:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            getattr(model, refit)(data)
        with experimental.capture_transform_diagnostics(model) as record:
            result = model.transform(data)
        assert_success(record)
        assert result.shape == (6, 1)
    assert not any(name.startswith("last_transform_") for name in vars(model))


def test_exact_upstream_estimator_state_check_passes():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        check_dict_unchanged("ANLSBPP", experimental.ANLSBPP(2, max_iter=20))


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("outcomes", ["success_success", "success_cap", "success_invalid"])
def test_two_callers_retain_own_receipts_after_both_calls_finish(monkeypatch, order, outcomes):
    model = prepared(cap=1 if outcomes == "success_cap" else None, legacy=True)
    state = frozen(model)
    queries = {"a": np.zeros((1, 2)), "b": np.array([[1., 3.]])}
    if outcomes == "success_invalid":
        queries["b"] = np.array([[-1., 3.]])
    expected = {}
    for caller, query in queries.items():
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with experimental.capture_transform_diagnostics(model) as record:
                try:
                    values = model.transform(query)
                    error = None
                except ValueError as exc:
                    values, error = None, str(exc)
        expected[caller] = (values, deepcopy(record), error)
    assert frozen(model) == state
    local_call = local()
    entered = {name: Event() for name in queries}
    finish_gate = {name: Event() for name in queries}
    returned = {name: Event() for name in queries}
    read_gate = Event()
    original = experimental.nnls_block_pivot

    def controlled_solver(*args, **kwargs):
        name = local_call.name
        # Real solver and its warning contract are unchanged. Filters were set
        # before workers; no warnings context is opened inside a worker.
        values, info = original(*args, **kwargs)
        entered[name].set()
        if not finish_gate[name].wait(10):
            raise TimeoutError("Solver-return gate")
        return values, info

    def invoke(name):
        local_call.name = name
        with experimental.capture_transform_diagnostics(model) as record:
            try:
                values = model.transform(queries[name])
                error = None
            except ValueError as exc:
                values, error = None, str(exc)
                entered[name].set()
            finally:
                returned[name].set()
        if not read_gate.wait(10):
            raise TimeoutError("After-return receipt gate")
        return values, deepcopy(record), error

    monkeypatch.setattr(experimental, "nnls_block_pivot", controlled_solver)
    futures = []
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(invoke, name) for name in ("a", "b")]
        assert all(event.wait(10) for event in entered.values())
        for name in order:
            finish_gate[name].set()
            assert returned[name].wait(10)
        read_gate.set()
        results = [future.result(timeout=10) for future in futures]
    finally:
        for event in finish_gate.values():
            event.set()
        read_gate.set()
        done, pending = wait(futures, timeout=10)
        pool.shutdown(wait=not pending, cancel_futures=True)
        assert not pending
    for name, actual in zip(("a", "b"), results):
        expected_values, expected_record, expected_error = expected[name]
        values, record, error = actual
        assert error == expected_error
        assert record == expected_record
        if values is not None:
            np.testing.assert_array_equal(values, expected_values)
    assert results[0][1] is not results[1][1]
    assert frozen(model) == state
