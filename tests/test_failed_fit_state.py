"""Failed estimator fits discard newly published learned state."""
from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

import supervised_nmf.decomposition as decomposition
import supervised_nmf.screening as screening
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.screening import BinaryScreening


@pytest.fixture
def data():
    X = np.array([[1., 0., 1., 2.], [2., 1., 3., 1.],
                  [1., 2., 2., 3.], [4., 1., 2., 2.],
                  [3., 4., 1., 4.], [2., 3., 4., 1.]])
    return X, np.array([0, 0, 0, 1, 1, 1])


def _model(kind):
    return (FeatureSPA(n_components=2) if kind == "spa"
            else BinaryScreening(score_threshold=0.))


def _learned(model):
    return {name: value for name, value in vars(model).items()
            if name.endswith("_") and not name.startswith("__")}


def _assert_unfitted(model, X):
    assert _learned(model) == {}
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.transform(X)
    if isinstance(model, BinaryScreening):
        with pytest.raises(NotFittedError):
            model.get_support()
        with pytest.raises(NotFittedError):
            model.get_feature_names_out()
    else:
        with pytest.raises(NotFittedError):
            model.inverse_transform(np.ones((2, 2)))


def _assert_retry(model, X, y):
    reference = clone(model).fit(X, y)
    assert model.fit(X, y) is model
    np.testing.assert_array_equal(model.transform(X), reference.transform(X))


class _NumpyFailure:
    """Inject an allocation failure only into the screening module's calls."""
    def __init__(self, name, occurrence, error):
        self.name = name
        self.occurrence = occurrence
        self.error = error
        self.calls = 0

    def __getattr__(self, name):
        original = getattr(np, name)
        if name != self.name:
            return original

        def call(*args, **kwargs):
            self.calls += 1
            if self.calls == self.occurrence:
                raise self.error
            return original(*args, **kwargs)
        return call


@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError, FloatingPointError])
def test_spa_diagnostic_failure_clears_state(monkeypatch, data, refit, error_type):
    X, y = data
    before = X.copy(), y.copy()
    model = _model("spa")
    if refit:
        model.fit(X, y)
        model.old_marker_ = "discard after failed refit"
    parameters = model.get_params().copy()
    error = error_type("diagnostic failed after basis fitting")

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(decomposition, "reconstruction_error", fail)
        with pytest.raises(error_type) as caught:
            model.fit(X, y)
    assert caught.value is error
    assert model.get_params() == parameters
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(y, before[1])
    _assert_unfitted(model, X)
    _assert_retry(model, X, y)


@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("name,occurrence", [
    pytest.param("flatnonzero", 1, id="selected-indices"),
    pytest.param("flatnonzero", 2, id="constant-indices"),
    pytest.param("isinf", 1, id="infinite-score-counter"),
    pytest.param("isinf", 3, id="overflow-counter"),
])
def test_screening_finalization_failure_clears_state(monkeypatch, data, refit,
                                                    name, occurrence):
    X, y = data
    before = X.copy(), y.copy()
    model = _model("screen")
    if refit:
        model.fit(X, y)
        model.old_marker_ = "discard after failed refit"
    parameters = model.get_params().copy()
    error = MemoryError("screening finalization allocation")
    failure = _NumpyFailure(name, occurrence, error)
    with monkeypatch.context() as patch:
        patch.setattr(screening, "np", failure)
        with pytest.raises(MemoryError) as caught:
            model.fit(X, y)
    assert caught.value is error
    assert failure.calls == occurrence
    assert model.get_params() == parameters
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(y, before[1])
    _assert_unfitted(model, X)
    _assert_retry(model, X, y)


@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("boundary", ["middle-copy", "last-copy", "index", "count"])
def test_statistics_reuse_failure_clears_recipient(monkeypatch, data, refit, boundary):
    X, y = data
    donor = _model("screen").fit(X, y)
    donor.extra_learned_ = {"nested": [1, 2]}
    saved = deepcopy(_learned(donor))
    model = _model("screen")
    if refit:
        model.fit(X, y)
        model.old_marker_ = "discard after failed refit"
    parameters = model.get_params().copy()
    error = MemoryError("screening statistics reuse failed")
    original_copy = screening.deepcopy
    calls = 0
    fail_at = len(saved) // 2 if boundary == "middle-copy" else len(saved)

    def copy_or_fail(value):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise error
        return original_copy(value)

    class SumFailure(np.ndarray):
        def sum(self, *args, **kwargs):
            raise error

    original_support = model._support_from_statistics

    def support(*args, **kwargs):
        return original_support(*args, **kwargs).view(SumFailure)

    with monkeypatch.context() as patch:
        if boundary.endswith("copy"):
            patch.setattr(screening, "deepcopy", copy_or_fail)
        elif boundary == "index":
            patch.setattr(screening, "np", _NumpyFailure("flatnonzero", 1, error))
        else:
            patch.setattr(model, "_support_from_statistics", support)
        with pytest.raises(MemoryError) as caught:
            model._fit_from_statistics(donor)
    assert caught.value is error
    assert model.get_params() == parameters
    _assert_unfitted(model, X)
    assert saved.keys() == _learned(donor).keys()
    for name, value in saved.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(donor, name), value)
        else:
            assert getattr(donor, name) == value
    assert model._fit_from_statistics(donor) is model
    np.testing.assert_array_equal(model.transform(X), donor.transform(X))
    assert model.extra_learned_ == donor.extra_learned_
    assert model.extra_learned_ is not donor.extra_learned_


class _AssignmentFailure:
    def __setattr__(self, name, value):
        if name == getattr(self, "_fail_name", None):
            object.__setattr__(self, "_fail_name", None)
            raise self._fail_error
        super().__setattr__(name, value)


class _AssignmentSPA(_AssignmentFailure, FeatureSPA):
    pass


class _AssignmentScreen(_AssignmentFailure, BinaryScreening):
    pass


@pytest.mark.parametrize("kind,name", [
    ("spa", "components_"), ("spa", "n_iter_"),
    ("screen", "support_"), ("screen", "n_classwise_constant_features_"),
])
def test_failed_learned_attribute_assignment_clears_state(data, kind, name):
    X, y = data
    model = (_AssignmentSPA(n_components=2) if kind == "spa"
             else _AssignmentScreen(score_threshold=0.))
    model.fit(X, y)
    error = RuntimeError("learned attribute publication failed")
    model._fail_name, model._fail_error = name, error
    with pytest.raises(RuntimeError) as caught:
        model.fit(X, y)
    assert caught.value is error
    _assert_unfitted(model, X)
    _assert_retry(model, X, y)


def test_statistics_reuse_keeps_independent_per_attribute_copies(data):
    X, y = data
    donor = _model("screen").fit(X, y)
    shared = []
    donor.extra_a_ = [shared, shared]
    donor.extra_b_ = donor.extra_a_
    model = _model("screen")._fit_from_statistics(donor)
    assert model.extra_a_ is not donor.extra_a_
    assert model.extra_b_ is not donor.extra_b_
    assert model.extra_a_ is not model.extra_b_
    assert model.extra_a_[0] is model.extra_a_[1]
    assert model.extra_b_[0] is model.extra_b_[1]
    assert model.extra_a_[0] is not model.extra_b_[0]
    model.extra_a_[0].append("recipient only")
    assert shared == []
    for name, value in _learned(donor).items():
        if isinstance(value, np.ndarray):
            assert not np.shares_memory(value, getattr(model, name))


@pytest.mark.parametrize("kind", ["spa", "screen"])
@pytest.mark.parametrize("refit", [False, True])
def test_pipeline_rejects_partial_child_after_failed_fit(monkeypatch, data, kind, refit):
    X, y = data
    pipeline = Pipeline([("transform", _model(kind)),
                         ("classifier", KNeighborsClassifier(n_neighbors=3))])
    if refit:
        pipeline.fit(X, y)
    error = MemoryError("late child fit failure")

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        if kind == "spa":
            patch.setattr(decomposition, "reconstruction_error", fail)
        else:
            patch.setattr(screening, "np", _NumpyFailure("flatnonzero", 1, error))
        with pytest.raises(MemoryError) as caught:
            pipeline.fit(X, y)
    assert caught.value is error
    _assert_unfitted(pipeline.named_steps["transform"], X)
    with pytest.raises(NotFittedError):
        pipeline.predict(X)
    with pytest.raises(NotFittedError):
        pipeline.predict_proba(X)
    reference = clone(pipeline).fit(X, y)
    assert pipeline.fit(X, y) is pipeline
    np.testing.assert_array_equal(pipeline.predict(X), reference.predict(X))
    np.testing.assert_array_equal(pipeline.predict_proba(X), reference.predict_proba(X))


@pytest.mark.parametrize("kind", ["spa", "screen"])
def test_later_transform_failure_keeps_successful_fit(monkeypatch, data, kind):
    X, y = data
    model = _model(kind)
    error = RuntimeError("transform failed after fit completed")
    saved = {}

    def fail(*args, **kwargs):
        saved.update(_learned(model))
        raise error

    with monkeypatch.context() as patch:
        if kind == "spa":
            patch.setattr(decomposition, "project_rows", fail)
        else:
            patch.setattr(screening, "_nonnegative_array", fail)
        with pytest.raises(RuntimeError) as caught:
            model.fit_transform(X, y)
    assert caught.value is error
    assert saved
    assert _learned(model).keys() == saved.keys()
    assert all(_learned(model)[name] is value for name, value in saved.items())
    reference = _model(kind).fit_transform(X, y)
    np.testing.assert_array_equal(model.transform(X), reference)
