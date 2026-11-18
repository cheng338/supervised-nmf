"""Current-model persistence, copying, cloning and process transport.

Current-model contracts only. These tests do not require older states to be
rejected, repaired, or assigned a newly inferred fitted policy.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
import hashlib
import io
import multiprocessing
import os
import pickle

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_info, threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def data():
    rng = np.random.default_rng(20261105)
    weights = rng.uniform(.2, 2, (48, 3))
    basis = np.column_stack((np.eye(3), rng.uniform(.1, .7, (3, 9))))
    X = weights @ basis + rng.uniform(0, .001, (48, 12))
    y = (weights[:, 0] > np.median(weights[:, 0])).astype(int)
    query = np.vstack((X[:7], np.zeros((1, 12)), X[[0, 0]]))
    return X, y, query


def _new(kind):
    if kind == 'classifier':
        return NMFClassifier(n_components=3, screening='none', n_neighbors=3,
                             knn_normalization='sd', max_iter=1000, random_state=17)
    if kind == 'spa':
        return FeatureSPA(n_components=3)
    return Pipeline([
        ('representation', Pipeline([('spa', FeatureSPA(n_components=3))])),
        ('knn', KNeighborsClassifier(n_neighbors=3)),
    ])


def _parameters(value):
    """Compare nested estimator settings, not estimator object identities."""
    if isinstance(value, BaseEstimator):
        return (type(value).__module__, type(value).__qualname__,
                _parameters(value.get_params(deep=False)))
    if isinstance(value, dict):
        return {key: _parameters(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_parameters(item) for item in value)
    return value


def _arrays(value, path='', seen=None):
    if isinstance(value, np.ndarray):
        return {path: (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())}
    seen = set() if seen is None else seen
    if id(value) in seen:
        return {}
    seen.add(id(value))
    if isinstance(value, BaseEstimator):
        items = vars(value).items()
    elif isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    else:
        return {}
    result = {}
    for key, item in items:
        result.update(_arrays(item, path + '/' + str(key), seen))
    return result


def _policies(model):
    names = ('training_coordinates_mode_', 'training_projection_performed_',
             'knn_normalization_', 'mu_projection_', 'decision_threshold_',
             'positive_label_', 'positive_class_index_')
    return {key: getattr(model, key) for key in names if hasattr(model, key)}


def _outputs(model, query):
    names = ('transform', 'predict', 'predict_proba', 'decision_function',
             'predict_with_probabilities', 'predict_with_decision_function')
    return {name: getattr(model, name)(query) for name in names if hasattr(model, name)}


def _assert_outputs(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if isinstance(actual[name], tuple):
            assert len(actual[name]) == len(expected[name])
        pairs = zip(actual[name], expected[name]) if isinstance(actual[name], tuple) else [(actual[name], expected[name])]
        for value, reference in pairs:
            assert np.isfinite(value).all()
            np.testing.assert_array_equal(value, reference)


def _transport(model, operation):
    if operation == 'pickle':
        return pickle.loads(pickle.dumps(model, protocol=5))
    if operation == 'joblib':
        stream = io.BytesIO()
        joblib.dump(model, stream, compress=0)
        stream.seek(0)
        return joblib.load(stream)
    return copy(model) if operation == 'copy' else deepcopy(model)


def _assert_unfitted(model):
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            _assert_unfitted(step)


@pytest.mark.parametrize('kind', ['classifier', 'pipeline'])
@pytest.mark.parametrize('operation', ['pickle', 'joblib', 'copy', 'deepcopy'])
def test_current_transports_preserve_learned_state_and_pending_settings(data, kind, operation):
    X, y, query = data
    model = _new(kind).fit(X, y)
    expected = _outputs(model, query)
    learned, policies = _arrays(model), _policies(model)
    if kind == 'classifier':
        model.set_params(solver='cd', classifier='linear_svm', knn_normalization='none',
                         training_coordinates='fit_transform')
    parameters = _parameters(model)
    restored = _transport(model, operation)
    assert _parameters(restored) == parameters
    assert _policies(restored) == policies
    assert _arrays(restored) == learned
    _assert_outputs(_outputs(restored, query), expected)
    assert _arrays(restored) == learned
    assert _policies(restored) == policies


@pytest.mark.parametrize('kind', ['classifier', 'spa'])
@pytest.mark.parametrize('failed_refit', [False, True])
def test_current_unfitted_and_failed_states_do_not_resurrect(data, kind, failed_refit):
    X, y, _ = data
    model = _new(kind)
    if failed_refit:
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.fit(np.full_like(X, np.nan), y)
    restored = _transport(model, 'pickle')
    assert _parameters(restored) == _parameters(model)
    _assert_unfitted(restored)
    _assert_unfitted(clone(restored))


@pytest.mark.parametrize('kind', ['classifier', 'pipeline'])
def test_current_loaded_clone_and_refit_use_constructor_settings(data, kind):
    X, y, query = data
    model = _new(kind).fit(X, y)
    if kind == 'classifier':
        model.set_params(solver='cd', classifier='linear_svm', knn_normalization='none',
                         training_coordinates='fit_transform')
    restored = _transport(model, 'pickle')
    fresh = clone(restored)
    assert _parameters(fresh) == _parameters(restored)
    _assert_unfitted(fresh)
    restored.fit(X, y)
    fresh.fit(X, y)
    _assert_outputs(_outputs(restored, query), _outputs(fresh, query))
    if kind == 'classifier':
        assert restored.decomposition_.solver == 'cd'
        assert restored.training_coordinates_mode_ == 'fit_transform'
        assert restored.knn_normalization_ is None
        assert hasattr(restored, 'decision_function')
        assert not hasattr(restored, 'predict_proba')


def _process_outputs(model, query):
    with threadpool_limits(limits=1):
        result = _outputs(model, query)
        return os.getpid(), result, _policies(model), [pool['num_threads'] for pool in threadpool_info()]


@pytest.mark.parametrize('kind', ['classifier', 'pipeline'])
def test_current_model_survives_spawn_process_transport(data, kind, monkeypatch):
    X, y, query = data
    model = _new(kind).fit(X, y)
    expected = _outputs(model, query)
    for name in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS'):
        monkeypatch.setenv(name, '1')
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        pid, actual, policies, pools = pool.submit(_process_outputs, model, query).result(timeout=60)
    assert pid != os.getpid()
    assert all(count == 1 for count in pools)
    assert policies == _policies(model)
    _assert_outputs(actual, expected)
