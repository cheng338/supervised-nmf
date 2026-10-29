"""Prediction state, diagnostic ownership and direct warning contracts."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import pickle
from threading import Event
import warnings

import numpy as np
import pytest
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection


@pytest.fixture
def data():
    rng = np.random.default_rng(932)
    weights = rng.uniform(.1, 2, (48, 3))
    basis = np.column_stack((np.eye(3), rng.uniform(.1, .5, (3, 9))))
    X = weights @ basis + rng.uniform(0, .001, (48, 12))
    y = np.where(weights[:, 0] > np.median(weights[:, 0]), "yes", "no")
    return X, y


def fitted(data, policy="spa", head="knn", dtype=np.float64):
    X, y = data
    return NMFClassifier(
        solver="mu" if policy == "rowwise" else policy,
        mu_projection="rowwise" if policy == "rowwise" else "batch",
        classifier=head, screening="none", n_components=3, n_neighbors=3,
        random_state=21, max_iter=150, tol=0,
        logistic_max_iter=2000, svm_max_iter=5000,
    ).fit(X.astype(dtype), y)


@pytest.mark.parametrize("policy", ["spa", "cd", "mu", "rowwise"])
@pytest.mark.parametrize("head", ["knn", "logistic", "linear_svm"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_inference_preserves_model_and_decomposition(data, policy, head, dtype):
    model = fitted(data, policy, head, dtype)
    state = dict(vars(model))
    decomposition = pickle.dumps(model.decomposition_)
    arrays = {key: value.copy() for key, value in vars(model).items()
              if isinstance(value, np.ndarray)}
    X, y = data
    queries = (X[:8].astype(dtype), X[:8].astype(np.float64 if dtype == np.float32 else np.float32),
               np.zeros((8, X.shape[1]), dtype=dtype))
    methods = ("transform", "predict", "predict_proba", "decision_function",
               "predict_with_probabilities", "predict_with_decision_function")
    for query in queries:
        for method in methods:
            if hasattr(model, method):
                with _capture_projection(model) as record:
                    getattr(model, method)(query)
                assert record["last_transform_performed"] is True
                assert record["last_transform_converged"] is True
                assert record["last_transform_error"] is None
        model.score(query, y[:8])
    assert vars(model).keys() == state.keys()
    assert all(vars(model)[key] is value for key, value in state.items())
    assert pickle.dumps(model.decomposition_) == decomposition
    for key, value in arrays.items():
        np.testing.assert_array_equal(getattr(model, key), value)
    assert not any(key.startswith("last_transform_") for key in vars(model))


def test_nested_records_restore_outer_owner_and_do_not_leak(data):
    model = fitted(data)
    other = deepcopy(model)
    X, _ = data
    with _capture_projection(model) as outer:
        with _capture_projection(model) as inner:
            with pytest.raises(ValueError):
                model.transform(X[:, :-1])
        assert inner["last_transform_converged"] is False
        assert outer["last_transform_performed"] is False
        other.transform(X)
        assert outer["last_transform_performed"] is False
        model.transform(X)
    assert outer["last_transform_converged"] is True
    saved = dict(outer)
    with pytest.raises(ValueError):
        model.transform(X[:, :-1])
    assert outer == saved
    with _capture_projection(model) as empty:
        pass
    assert empty["last_transform_performed"] is False
    assert empty["last_transform_converged"] is None


def test_concurrent_success_and_error_have_separate_records(data):
    model = fitted(data)
    X, _ = data
    entered, failed = Event(), Event()
    original = model.decomposition_.transform

    def controlled(query):
        entered.set()
        assert failed.wait(10)
        return original(query)

    model.decomposition_.transform = controlled

    def success():
        with _capture_projection(model) as record:
            model.predict(X[:6])
        return record

    def failure():
        assert entered.wait(10)
        try:
            with _capture_projection(model) as record:
                with pytest.raises(ValueError):
                    model.predict(X[:6, :-1])
            return record
        finally:
            failed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        good, bad = pool.submit(success), pool.submit(failure)
        good, bad = good.result(timeout=20), bad.result(timeout=20)
    assert good["last_transform_converged"] is True
    assert good["last_transform_error"] is None
    assert bad["last_transform_converged"] is False
    assert "expecting" in bad["last_transform_error"]
    assert not any(key.startswith("last_transform_") for key in vars(model))


@pytest.mark.parametrize("category", [ConvergenceWarning, RuntimeWarning])
@pytest.mark.parametrize("action", ["always", "ignore", "error"])
@pytest.mark.parametrize("solver_error", [False, True])
def test_direct_warning_and_exception_precedence(data, category, action, solver_error):
    model = fitted(data)
    X, _ = data

    def operation(query):
        warnings.warn("projection warning", category)
        if solver_error:
            raise ValueError("original solver failure")
        return np.zeros((len(query), 3))

    model.decomposition_.transform = operation
    expected_error = None
    if action == "error" and category is RuntimeWarning:
        expected_error = RuntimeWarning
    elif solver_error:
        expected_error = ValueError
    elif action == "error":
        expected_error = category
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        with _capture_projection(model) as record:
            if expected_error is None:
                model.transform(X)
            else:
                with pytest.raises(expected_error):
                    model.transform(X)
    assert bool(record["last_transform_error"]) == (expected_error is not None)
    assert record["last_transform_converged"] == (
        expected_error is None and category is not ConvergenceWarning)
    assert record["last_transform_warnings"] == (
        ("projection warning",) if category is ConvergenceWarning else ())
    assert len(caught) == (1 if action == "always" else 0)


def _process_prediction(model, X):
    with _capture_projection(model) as record:
        result = model.predict_with_probabilities(X)
    return result, record, any(key.startswith("last_transform_") for key in vars(model))


@pytest.mark.parametrize("policy", ["spa", "cd", "mu", "rowwise"])
def test_pickle_clone_and_process_calls_do_not_carry_diagnostic_context(data, policy):
    model = fitted(data, policy)
    X, _ = data
    with _capture_projection(model) as parent:
        restored = pickle.loads(pickle.dumps(model))
        assert clone(model).get_params() == model.get_params()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=2)(delayed(_process_prediction)(restored, X[:8]) for _ in range(2))
    assert parent["last_transform_performed"] is False
    expected = model.predict_with_probabilities(X[:8])
    for actual, record, has_model_diagnostics in results:
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        assert record["last_transform_converged"] is True
        assert not has_model_diagnostics
