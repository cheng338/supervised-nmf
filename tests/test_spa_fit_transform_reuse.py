"""Reuse fitted SPA coordinates without bypassing custom transformer behavior."""

from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier
from supervised_nmf.classifier import _capture_projection


@pytest.fixture(autouse=True)
def one_numerical_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(417)
    y = np.arange(36) % 2
    X = rng.uniform(.1, 1.2, (len(y), 8))
    X[:, :3] += y[:, None]
    query = rng.uniform(.1, 1.7, (9, X.shape[1]))
    query[0] = 0.
    return X, y, query


def model(cls=NMFClassifier, **changes):
    parameters = dict(solver="spa", n_components=2, screening="none",
                      n_neighbors=3, random_state=17, max_iter=2000, tol=1e-3)
    parameters.update(changes)
    return cls(**parameters)


def independent_projection(estimator, X):
    basis = estimator.selected_components_
    return np.vstack([nnls(basis.T, row, maxiter=2000)[0] for row in X])


@pytest.mark.parametrize("mode", ["auto", "fit_transform", "transform"])
@pytest.mark.parametrize("head", ["knn", "logistic", "linear_svm"])
def test_stock_spa_fit_transform_projects_once_for_every_mode_and_head(binary_data, mode, head):
    X, y, _ = binary_data
    estimator = model(training_coordinates=mode, classifier=head)
    original_transform = FeatureSPA.transform
    calls = []

    def capture_transform(spa, supplied):
        calls.append(np.asarray(supplied).copy())
        return original_transform(spa, supplied)

    with patch.object(FeatureSPA, "transform", capture_transform):
        with _capture_projection(estimator) as record:
            actual = estimator.fit_transform(X, y)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], X)
    np.testing.assert_allclose(actual, independent_projection(estimator, X), rtol=2e-11, atol=2e-11)
    np.testing.assert_array_equal(actual, estimator.training_coordinates_)
    assert estimator.training_coordinates == mode
    assert estimator.training_coordinates_mode_ == "transform"
    assert estimator.training_projection_performed_
    assert estimator.training_projection_converged_
    assert record is not None
    assert not record["last_transform_performed"]
    assert record["last_transform_converged"] is None
    assert record["last_transform_error"] is None


def test_returned_coordinates_do_not_alias_saved_coordinates_or_knn(binary_data):
    X, y, query = binary_data
    estimator = model()
    actual = estimator.fit_transform(X, y)
    stored = estimator.training_coordinates_.copy()
    head_input = estimator.classifier_._fit_X.copy()
    anchors = estimator.decomposition_.fit_coordinates_.copy()
    basis = estimator.selected_components_.copy()
    expected_predictions = estimator.predict(query)
    for fitted_array in (estimator.training_coordinates_, estimator.classifier_._fit_X,
                         estimator.decomposition_.fit_coordinates_, estimator.selected_components_):
        assert not np.shares_memory(actual, fitted_array)
    actual[:] = 0.
    np.testing.assert_array_equal(estimator.training_coordinates_, stored)
    np.testing.assert_array_equal(estimator.classifier_._fit_X, head_input)
    np.testing.assert_array_equal(estimator.decomposition_.fit_coordinates_, anchors)
    np.testing.assert_array_equal(estimator.selected_components_, basis)
    np.testing.assert_array_equal(estimator.predict(query), expected_predictions)


class PlainSubclass(NMFClassifier):
    pass


class ShiftedSubclass(NMFClassifier):
    def transform(self, X):
        return super().transform(X) + 7.


@pytest.mark.parametrize("cls,offset", [(PlainSubclass, 0.), (ShiftedSubclass, 7.)])
def test_subclasses_keep_their_fit_then_transform_behavior(binary_data, cls, offset):
    X, y, _ = binary_data
    estimator = model(cls=cls)
    original_transform = FeatureSPA.transform
    calls = []

    def capture_transform(spa, supplied):
        calls.append(len(supplied))
        return original_transform(spa, supplied)

    with patch.object(FeatureSPA, "transform", capture_transform):
        actual = estimator.fit_transform(X, y)
    assert calls == [len(X), len(X)]
    expected = independent_projection(estimator, X) + offset
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_instance_transform_override_is_not_bypassed(binary_data):
    X, y, _ = binary_data
    estimator = model()
    transformed_inputs = []

    def shifted_transform(instance, supplied):
        transformed_inputs.append(np.asarray(supplied).copy())
        return NMFClassifier.transform(instance, supplied) + 4.

    with patch.object(estimator, "transform", MethodType(shifted_transform, estimator)):
        actual = estimator.fit_transform(X, y)
    assert len(transformed_inputs) == 1
    np.testing.assert_array_equal(transformed_inputs[0], X)
    np.testing.assert_allclose(actual, independent_projection(estimator, X) + 4., rtol=2e-11, atol=2e-11)


def test_failed_training_projection_does_not_return_previous_cached_coordinates(binary_data):
    X, y, _ = binary_data
    estimator = model().fit(X, y)
    with patch.object(FeatureSPA, "transform", side_effect=RuntimeError("training projection failed")) as project:
        with pytest.raises(RuntimeError, match="training projection failed"):
            estimator.fit_transform(X, y)
    assert project.call_count == 1
    assert not hasattr(estimator, "classifier_")
    assert not hasattr(estimator, "training_coordinates_")


def test_custom_transform_failure_is_propagated(binary_data):
    X, y, _ = binary_data
    estimator = model()
    with patch.object(estimator, "transform", side_effect=RuntimeError("custom transform failed")) as transform:
        with pytest.raises(RuntimeError, match="custom transform failed"):
            estimator.fit_transform(X, y)
    assert transform.call_count == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "strided_readonly"])
def test_reused_coordinates_preserve_spa_dtype_and_input_storage(binary_data, dtype, layout):
    X, y, _ = binary_data
    if layout == "strided_readonly":
        owner = np.full((len(X), 2 * X.shape[1]), -7., dtype=dtype)
        supplied = owner[:, ::2]
        supplied[:] = X
        supplied.setflags(write=False)
    else:
        owner = np.array(X, dtype=dtype, order="C")
        supplied = owner
    before = owner.copy()
    writeable = supplied.flags.writeable
    estimator = model()
    actual = estimator.fit_transform(supplied, y)
    assert actual.dtype == np.dtype("float64")
    np.testing.assert_allclose(actual, independent_projection(estimator, supplied), rtol=2e-11, atol=2e-11)
    np.testing.assert_array_equal(actual, estimator.training_coordinates_)
    np.testing.assert_array_equal(owner, before)
    assert supplied.flags.writeable == writeable
