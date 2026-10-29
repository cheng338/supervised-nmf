"""Classifier and transformer interface contracts. Author: Chang Cheng."""

from copy import deepcopy
import pickle

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone, is_classifier
from sklearn.model_selection import StratifiedKFold, check_cv, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags
from sklearn.utils.estimator_checks import estimator_checks_generator

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(417)
    y = np.arange(36) % 2
    X = rng.uniform(.1, 1.2, (len(y), 8))
    X[:, :3] += y[:, None]
    query = rng.uniform(.1, 1.7, (9, X.shape[1]))
    query[0] = 0.
    return X, y, query


def model(**overrides):
    parameters = dict(n_components=2, screening="none", n_neighbors=3,
                      random_state=17, max_iter=2000, tol=1e-3)
    parameters.update(overrides)
    return NMFClassifier(**parameters)


def test_tags_preserve_classifier_identity_and_are_independent():
    estimator = model()
    before = deepcopy(vars(estimator))
    first, second = get_tags(estimator), get_tags(estimator)
    assert is_classifier(estimator)
    assert first.estimator_type == "classifier"
    assert first.classifier_tags is not None
    assert first.target_tags.required
    assert first.transformer_tags is not None
    assert first.transformer_tags.preserves_dtype == ["float64"]
    assert first is not second
    first.target_tags.required = False
    first.transformer_tags.preserves_dtype.append("mutation_probe")
    assert get_tags(estimator).target_tags.required
    assert get_tags(estimator).transformer_tags.preserves_dtype == ["float64"]
    assert vars(estimator) == before


@pytest.mark.parametrize("solver", ["spa", "cd", "mu"])
@pytest.mark.parametrize("classifier", ["knn", "logistic", "linear_svm"])
@pytest.mark.parametrize("training_coordinates", ["auto", "fit_transform", "transform"])
def test_fit_transform_uses_the_fitted_projection(binary_data, solver, classifier,
                                                training_coordinates):
    X, y, query = binary_data
    estimator = model(solver=solver, classifier=classifier,
                      training_coordinates=training_coordinates)
    reference = clone(estimator).fit(X, y)
    transformed = estimator.fit_transform(X, y)
    assert isinstance(transformed, np.ndarray)
    assert transformed.dtype == X.dtype == np.dtype("float64")
    np.testing.assert_array_equal(transformed, reference.transform(X))
    np.testing.assert_array_equal(transformed, estimator.transform(X))
    np.testing.assert_array_equal(estimator.training_coordinates_, reference.training_coordinates_)
    np.testing.assert_array_equal(estimator.predict(query), reference.predict(query))
    if classifier == "linear_svm":
        np.testing.assert_array_equal(estimator.decision_function(query), reference.decision_function(query))
    else:
        np.testing.assert_array_equal(estimator.predict_proba(query), reference.predict_proba(query))
    assert estimator.training_coordinates_mode_ == reference.training_coordinates_mode_


def test_legacy_spa_training_coordinates_are_not_the_transformer_return(binary_data):
    X, y, _ = binary_data
    estimator = model(training_coordinates="fit_transform")
    transformed = estimator.fit_transform(X, y)
    np.testing.assert_array_equal(estimator.training_coordinates_, estimator.decomposition_.fit_coordinates_)
    np.testing.assert_array_equal(estimator.classifier_._fit_X, estimator.training_coordinates_)
    np.testing.assert_array_equal(transformed, estimator.transform(X))
    assert not np.allclose(transformed, estimator.training_coordinates_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_fit_transform_requires_training_labels(binary_data, explicit_none):
    X, _, _ = binary_data
    estimator = model()
    with pytest.raises(TypeError):
        if explicit_none:
            estimator.fit_transform(X, y=None)
        else:
            estimator.fit_transform(X)
    assert not hasattr(estimator, "classifier_")


@pytest.mark.parametrize("routing", [False, True])
def test_fit_transform_rejects_unsupported_fit_parameters(binary_data, routing):
    X, y, _ = binary_data
    estimator = model()
    with config_context(enable_metadata_routing=routing):
        with pytest.raises(TypeError, match="sample_weight"):
            estimator.fit_transform(X, y, sample_weight=np.ones(len(y)))
    assert not hasattr(estimator, "classifier_")


def test_clone_pickle_and_projection_diagnostics_preserve_the_interface(binary_data):
    X, y, query = binary_data
    estimator = model()
    with _capture_projection(estimator) as training_record:
        transformed = estimator.fit_transform(X, y)
    assert training_record is not None
    assert training_record["last_transform_performed"]
    assert training_record["last_transform_error"] is None
    np.testing.assert_array_equal(transformed, estimator.transform(X))
    expected = estimator.predict(query)
    restored = pickle.loads(pickle.dumps(estimator))
    assert is_classifier(restored)
    assert get_tags(restored).transformer_tags is not None
    np.testing.assert_array_equal(restored.predict(query), expected)
    unfitted = clone(restored)
    assert unfitted.get_params() == estimator.get_params()
    assert not hasattr(unfitted, "classifier_")
    before = set(vars(restored))
    with _capture_projection(restored) as record:
        restored.predict(query)
    assert record is not None and record["last_transform_performed"]
    assert record["last_transform_error"] is None
    assert record["last_transform_seconds"] >= 0
    assert set(vars(restored)) == before


def test_classifier_pipeline_and_cross_validation_keep_classifier_semantics(binary_data):
    X, y, query = binary_data
    estimator = model()
    fitted = clone(estimator).fit(X, y)
    pipeline = Pipeline([("classifier", clone(estimator))]).fit(X, y)
    assert is_classifier(pipeline)
    assert isinstance(check_cv(3, y, classifier=is_classifier(estimator)), StratifiedKFold)
    np.testing.assert_array_equal(pipeline.predict(query), fitted.predict(query))
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
    expected = cross_val_score(estimator, X, y, cv=folds, scoring="balanced_accuracy", n_jobs=1)
    actual = cross_val_score(pipeline, X, y, cv=folds, scoring="balanced_accuracy", n_jobs=1)
    np.testing.assert_array_equal(actual, expected)


def test_feature_pipeline_receives_projected_coordinates_in_legacy_mode(binary_data):
    X, y, query = binary_data
    features = model(training_coordinates="fit_transform")
    reference = clone(features).fit(X, y)
    head = KNeighborsClassifier(n_neighbors=3).fit(reference.transform(X), y)
    pipeline = Pipeline([("features", features), ("head", KNeighborsClassifier(n_neighbors=3))]).fit(X, y)
    np.testing.assert_array_equal(pipeline.named_steps["head"]._fit_X, reference.transform(X))
    np.testing.assert_array_equal(pipeline.predict(query), head.predict(reference.transform(query)))


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("output", ["default", "pandas", "polars"])
def test_global_output_configuration_does_not_add_container_support(binary_data, routing, output):
    X, y, query = binary_data
    expected = model().fit(X, y)
    estimator = model()
    # The classifier does not expose feature names or a container-output API.
    # These contexts must not require optional pandas/polars dependencies.
    assert not hasattr(estimator, "set_output")
    with config_context(enable_metadata_routing=routing, transform_output=output):
        transformed = estimator.fit_transform(X, y)
        queried = estimator.transform(query)
        prediction = estimator.predict(query)
    assert isinstance(transformed, np.ndarray) and isinstance(queried, np.ndarray)
    np.testing.assert_array_equal(transformed, expected.transform(X))
    np.testing.assert_array_equal(queried, expected.transform(query))
    np.testing.assert_array_equal(prediction, expected.predict(query))


def test_estimator_check_discovery_includes_both_interfaces():
    names = []
    for _, check in estimator_checks_generator(model()):
        while hasattr(check, "func"):
            check = check.func
        names.append(check.__name__)
    assert "check_transformer_general" in names
    assert "check_classifiers_train" in names
    # Discovery is the regression here; individual checks may still report the
    # other requirements, including validation error-message formats.
