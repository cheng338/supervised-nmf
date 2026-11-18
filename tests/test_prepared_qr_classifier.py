"""Actual fitted-model lifecycle and classifier contracts with QR reuse."""
from copy import copy, deepcopy
import gc
import io
import pickle
import weakref

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from supervised_nmf import FeatureSPA, NMFClassifier, _projection
from supervised_nmf.classifier import _capture_projection
from tests._prepared_qr_support import (
    assert_coordinates, observed_calls, problem, qr_state, reference,
)


@pytest.fixture
def fitted_data():
    owner, query, _ = problem(rows=32)
    rng = np.random.default_rng(812403)
    weights = np.vstack((2 * np.eye(32), rng.uniform(.05, 1, (64, 32))))
    train = weights @ owner
    labels = (weights[:, 0] > np.median(weights[:, 0])).astype(int)
    return train, labels, query


def classifier():
    return NMFClassifier(n_components=32, solver='spa', screening='none',
                         classifier='knn', n_neighbors=3, knn_normalization='none',
                         projection_solver='qr', projection_workers=1,
                         projection_native_threads=1, projection_batch_size=256,
                         max_iter=10000, random_state=17)


def test_actual_classifier_uses_current_basis_and_preserves_inference_state(qr_state, fitted_data):
    train, labels, query = fitted_data
    originals = [a.copy() for a in fitted_data]
    model = classifier().fit(train, labels)
    owner = model.decomposition_.components_
    assert (_projection._qr_registry_get(owner) is not None) is qr_state
    expected_train = reference(owner, train)
    expected_query = reference(owner, query)
    native = clone(model.classifier_).fit(expected_train, labels)
    before = pickle.dumps(model, protocol=5)
    with observed_calls() as calls, _capture_projection(model) as metadata:
        actual = model.predict_with_probabilities(query)
    assert calls['qr'] == int(not qr_state)
    assert metadata['last_transform_performed'] is True
    assert metadata['last_transform_converged'] is True
    assert metadata['last_transform_error'] is None
    assert metadata['last_transform_warnings'] == ()
    np.testing.assert_array_equal(actual[0], native.predict(expected_query))
    np.testing.assert_array_equal(actual[1], native.predict_proba(expected_query))
    assert_coordinates(model.transform(query), expected_query)
    assert pickle.dumps(model, protocol=5) == before
    for a, b in zip(fitted_data, originals):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('operation', ['copy', 'deepcopy', 'pickle', 'joblib'])
def test_model_transport_preserves_outputs_without_persisting_preparation(qr_state, fitted_data, operation):
    train, labels, query = fitted_data
    model = classifier().fit(train, labels)
    expected = model.predict_with_probabilities(query)
    owner = model.decomposition_.components_
    entry = _projection._qr_registry_get(owner)
    if operation == 'copy':
        restored = copy(model)
    elif operation == 'deepcopy':
        restored = deepcopy(model)
    elif operation == 'pickle':
        restored = pickle.loads(pickle.dumps(model, protocol=5))
    else:
        stream = io.BytesIO()
        joblib.dump(model, stream, compress=0)
        stream.seek(0)
        restored = joblib.load(stream)
    restored_owner = restored.decomposition_.components_
    assert (restored_owner is owner) is (operation == 'copy')
    if operation != 'copy':
        assert _projection._qr_registry_get(restored_owner) is None
    else:
        assert _projection._qr_registry_get(restored_owner) is entry
    before = pickle.dumps(restored, protocol=5)
    with observed_calls() as calls:
        actual = restored.predict_with_probabilities(query)
    assert calls['qr'] == (int(not qr_state) if operation == 'copy' else 1)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert pickle.dumps(restored, protocol=5) == before
    assert restored.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        clone(restored).predict(query)


@pytest.mark.parametrize('failed_first', [False, True])
def test_refit_cannot_restore_or_reuse_the_previous_basis(qr_state, fitted_data, failed_first):
    train, labels, query = fitted_data
    model = classifier().fit(train, labels)
    old_owner = model.decomposition_.components_
    old_basis = old_owner.copy()
    if failed_first:
        invalid = train.copy()
        invalid[0, 0] = np.nan
        with pytest.raises(ValueError):
            model.fit(invalid, labels)
        with pytest.raises(NotFittedError):
            model.predict(query)
    changed = train.copy()
    changed[:, -1] *= 1.125
    model.fit(changed, labels)
    owner = model.decomposition_.components_
    assert owner is not old_owner
    assert not np.array_equal(owner, old_basis)
    expected_coordinates = reference(owner, query)
    assert_coordinates(model.transform(query), expected_coordinates)
    fresh = clone(model).fit(changed, labels)
    for actual, expected in zip(model.predict_with_probabilities(query), fresh.predict_with_probabilities(query)):
        np.testing.assert_array_equal(actual, expected)


def test_cache_does_not_keep_actual_model_or_query_alive(qr_state, fitted_data):
    train, labels, query = fitted_data
    model = classifier().fit(train, labels)
    owner = model.decomposition_.components_
    own_query = query.copy()
    model.predict_with_probabilities(own_query)
    owner_key = id(owner)
    refs = [weakref.ref(value) for value in (model, owner, own_query)]
    del model, owner, own_query
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert owner_key not in _projection._qr_registry


def test_actual_fit_only_then_projection_retains_fitted_arrays(qr_state, fitted_data):
    train, _, query = fitted_data
    model = FeatureSPA(32, projection_solver='qr', projection_native_threads=1,
                       nnls_max_iter=10000).fit(train)
    owner = model.components_
    assert _projection._qr_registry_get(owner) is None
    expected = reference(owner, query)
    before = pickle.dumps(model, protocol=5)
    with observed_calls() as first:
        actual = model.transform(query)
    assert first['qr'] == 1
    assert_coordinates(actual, expected)
    assert pickle.dumps(model, protocol=5) == before
    with observed_calls() as again:
        assert_coordinates(model.transform(query), expected)
    assert again['qr'] == int(not qr_state)
