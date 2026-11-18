"""Required prediction policies preserve fitted decisions and caller boundaries."""
from copy import deepcopy
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection
from supervised_nmf.evaluation import evaluate_classifier
from supervised_nmf import study


@pytest.fixture(autouse=True)
def native_one():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(20261106)
    weights = rng.uniform(.2, 2, (48, 2))
    basis = np.column_stack((np.eye(2), rng.uniform(.1, .6, (2, 8))))
    X = weights @ basis + rng.uniform(0, .0001, (48, 10))
    y = (weights[:, 0] > np.median(weights[:, 0])).astype(int)
    return X, y, np.vstack((X[:8], np.zeros((1, 10)), X[[0, 0]]))


def estimator(**kwargs):
    parameters = dict(n_components=2, screening='none', n_neighbors=3,
                      random_state=17, max_iter=2000)
    parameters.update(kwargs)
    return NMFClassifier(**parameters)


@pytest.fixture(scope='module')
def models(data):
    X, y, _ = data
    settings = {
        'knn': {},
        'logistic': dict(classifier='logistic', decision_threshold=.37, positive_label=0),
        'svm': dict(classifier='linear_svm'),
        'rowwise': dict(solver='mu', mu_projection='rowwise'),
    }
    with threadpool_limits(limits=1):
        return {name: estimator(**options).fit(X, y) for name, options in settings.items()}


def assert_unfitted(model):
    assert not any(name.endswith('_') for name in vars(model))


@pytest.mark.parametrize('kind', ['knn', 'rowwise'])
def test_missing_mu_policy_never_uses_pending_constructor(models, data, kind):
    model = deepcopy(models[kind])
    query = data[2]
    model.set_params(mu_projection='batch' if kind == 'rowwise' else 'rowwise')
    del model.mu_projection_
    basis = model.components_.copy()
    decomposition = pickle.dumps(model.decomposition_)
    for method in ('transform', 'predict', 'predict_proba', 'predict_with_probabilities'):
        with _capture_projection(model) as record:
            with pytest.raises(AttributeError, match='mu_projection_'):
                getattr(model, method)(query)
        assert record['last_transform_performed'] is True
        assert record['last_transform_converged'] is False
        assert record['last_transform_warnings'] == ()
        assert 'mu_projection_' in record['last_transform_error']
    np.testing.assert_array_equal(model.components_, basis)
    assert pickle.dumps(model.decomposition_) == decomposition
    assert 'mu_projection_' not in vars(model)
    # Support selection does not consume the projection policy.
    assert model.get_support().all()


@pytest.mark.parametrize('kind', ['knn', 'logistic', 'svm'])
def test_missing_threshold_changes_label_consumers_not_raw_scores(models, data, kind):
    model = deepcopy(models[kind])
    X, y, query = data
    raw_methods = [name for name in ('transform', 'predict_proba', 'decision_function')
                   if hasattr(model, name)]
    expected = {name: getattr(model, name)(query) for name in raw_methods}
    model.set_params(decision_threshold=.91, positive_label=1)
    del model.decision_threshold_
    for name in raw_methods:
        np.testing.assert_array_equal(getattr(model, name)(query), expected[name])
    for name in ('predict', 'predict_with_probabilities', 'predict_with_decision_function'):
        if hasattr(model, name):
            with pytest.raises(AttributeError, match='decision_threshold_'):
                getattr(model, name)(query)
    with pytest.raises(AttributeError, match='decision_threshold_'):
        model.score(X, y)
    report = evaluate_classifier(model, X, y)
    assert report['status'] == 'failed'
    assert report['error_type'] == 'AttributeError'
    assert 'decision_threshold_' in report['error']
    assert 'decision_threshold' not in report['diagnostics']
    # Projection succeeded; the missing label policy is a later failure.
    assert report['diagnostics']['last_transform_performed'] is True
    assert report['diagnostics']['last_transform_converged'] is True
    assert report['diagnostics']['last_transform_error'] is None
    assert 'predictions' not in report and 'metrics' not in report
    assert 'decision_threshold_' not in vars(model)


@pytest.mark.parametrize('kind', ['knn', 'logistic'])
def test_positive_index_is_needed_only_by_explicit_threshold(models, data, kind):
    model = deepcopy(models[kind])
    query = data[2]
    expected = model.predict(query)
    del model.positive_class_index_
    assert np.isfinite(model.predict_proba(query)).all()
    if kind == 'knn':
        np.testing.assert_array_equal(model.predict(query), expected)
        np.testing.assert_array_equal(model.predict_with_probabilities(query)[0], expected)
    else:
        for name in ('predict', 'predict_with_probabilities', 'predict_with_decision_function'):
            with pytest.raises(AttributeError, match='positive_class_index_'):
                getattr(model, name)(query)


@pytest.mark.parametrize('field', ['decision_threshold', 'positive_label'])
def test_missing_constructor_does_not_replace_fitted_policy_and_refit_clears(models, data, field):
    model = deepcopy(models['logistic'])
    X, y, query = data
    expected = model.predict_with_probabilities(query)
    delattr(model, field)
    restored = pickle.loads(pickle.dumps(model))
    actual = restored.predict_with_probabilities(query)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert field not in vars(restored)
    invalid_data = deepcopy(restored)
    with pytest.raises(AttributeError, match=field):
        restored.fit(X, y)
    assert_unfitted(restored)
    with pytest.raises(NotFittedError):
        restored.predict(query)
    assert_unfitted(pickle.loads(pickle.dumps(restored)))
    # Threshold is read before X validation, positive_label after _binary_data.
    error = AttributeError if field == 'decision_threshold' else ValueError
    with pytest.raises(error):
        invalid_data.fit(np.full_like(X, np.nan), y)
    assert_unfitted(invalid_data)


def test_current_refit_reestablishes_missing_fitted_fields(models, data):
    X, y, query = data
    model = deepcopy(models['knn'])
    model.set_params(decision_threshold=.37, positive_label=0)
    del model.mu_projection_
    del model.decision_threshold_
    fresh = clone(model)
    assert_unfitted(fresh)
    model.fit(X, y)
    fresh.fit(X, y)
    assert model.mu_projection_ is None and model.decision_threshold_ == .37
    assert model.positive_label_ == 0 and model.positive_class_index_ == 0
    for a, b in zip(model.predict_with_probabilities(query), fresh.predict_with_probabilities(query)):
        np.testing.assert_array_equal(a, b)


def test_optional_metadata_and_custom_transform_remain_optional(data):
    X, y, query = data
    head = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    model = NMFClassifier(positive_label='queued\0', decision_threshold=.9, mu_projection='rowwise')
    model.classifier_, model.classes_ = head, head.classes_
    model.decision_threshold_, model.positive_class_index_ = None, 1
    model.transform = Mock(side_effect=lambda rows: rows)
    # No positive_label_ or mu_projection_: custom transform bypasses projection.
    expected = head.predict(query)
    np.testing.assert_array_equal(model.predict(query), expected)
    model.transform.assert_called_once_with(query)
    model.transform.reset_mock()
    np.testing.assert_array_equal(model.predict_with_probabilities(query)[0], expected)
    model.transform.assert_called_once_with(query)
    report = evaluate_classifier(model, X, y)
    assert report['status'] == 'valid'
    assert 'positive_label' not in report['diagnostics']
    assert 'mu_projection' not in report['diagnostics']
    generic = evaluate_classifier(DummyClassifier(strategy='most_frequent').fit(X, y), X, y)
    assert generic['status'] == 'valid'
    assert all(name not in generic['diagnostics'] for name in (
        'decision_threshold', 'positive_label', 'positive_class_index', 'mu_projection'))


def test_own_classes_absence_retains_existing_operation_boundary(models, data):
    model = deepcopy(models['knn'])
    query = data[2]
    expected = model.predict(query)
    del model.classes_
    np.testing.assert_array_equal(model.predict(query), expected)
    assert np.isfinite(model.transform(query)).all()
    assert np.isfinite(model.predict_proba(query)).all()
    with pytest.raises(AttributeError, match='classes_'):
        model.predict_with_probabilities(query)


@pytest.mark.parametrize('field', ['mu_projection_', 'decision_threshold_'])
def test_input_and_label_errors_keep_precedence_over_missing_policy(models, data, field):
    model = deepcopy(models['knn'])
    query = data[2]
    delattr(model, field)
    for malformed in (query[:, :-1], -query):
        with pytest.raises(ValueError):
            model.predict(malformed)
    model.positive_label_ = 'bad\0'
    with pytest.raises(ValueError, match='NUL'):
        model.predict(query)


@pytest.mark.parametrize('field', ['mu_projection_', 'decision_threshold_', 'positive_class_index_'])
def test_shared_study_never_substitutes_missing_fresh_fitted_policy(data, tmp_path, monkeypatch, field):
    X, y, _ = data
    original = NMFClassifier._fit
    fitted_instances = []

    def fit_then_remove(instance, *args, **kwargs):
        result = original(instance, *args, **kwargs)
        delattr(instance, field)
        fitted_instances.append(instance)
        return result

    def forbid_query(*args, **kwargs):
        raise AssertionError('Missing policy must fail before a shared neighbor query')

    # Keep exact NMFClassifier type: a subclass would not enter the shared path.
    monkeypatch.setattr(NMFClassifier, '_fit', fit_then_remove)
    monkeypatch.setattr(study._KNNQueryReuse, 'predict', forbid_query)
    report = study.calibration_knn_grid(estimator(decision_threshold=.37, positive_label=0),
        {'n_neighbors': [1, 3]}, X, y, cv=2, seed=17, threads=1,
        checkpoint_dir=tmp_path / 'shared')
    assert len(fitted_instances) == 2
    assert all(field not in vars(model) for model in fitted_instances)
    assert report['selected_parameters'] is None
    assert all(not row['valid'] for row in report['candidates'])
    assert len(report['representation_folds']) == 2
    for record in report['representation_folds']:
        if field == 'mu_projection_':
            assert record['status'] == 'failed' and record['failure_stage'] == 'projection'
            assert record['error_type'] == 'AttributeError' and field in record['error']
            assert not record['neighbors']
        else:
            assert record['status'] == 'valid'
            assert len(record['neighbors']) == 2
            for result in record['neighbors'].values():
                assert result['status'] == 'failed'
                assert result['error_type'] == 'AttributeError' and field in result['error']
