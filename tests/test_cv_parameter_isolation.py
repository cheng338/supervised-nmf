"""Estimator-valued parameter grids must not carry fitted state across folds."""
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf.evaluation import calibration_search


FIT_EVENTS = []


class RememberingHead(ClassifierMixin, BaseEstimator):
    """A warm-start classifier exposing exactly which rows its state contains."""

    def __init__(self, label=0):
        self.label = label

    def fit(self, X, y):
        previous = set(getattr(self, "observed_rows_", set()))
        current = set(np.asarray(X[:, 0], dtype=int))
        FIT_EVENTS.append({"object": self, "before": previous, "training": current})
        self.observed_rows_ = previous | current
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.full(len(X), self.label, dtype=int)


class RememberingTransformer(TransformerMixin, BaseEstimator):
    """Expose stale transformation state without depending on prediction scores."""

    def __init__(self, gain=1.):
        self.gain = gain

    def fit(self, X, y=None):
        previous = set(getattr(self, "observed_rows_", set()))
        current = set(np.asarray(X[:, 0], dtype=int))
        FIT_EVENTS.append({"object": self, "before": previous, "training": current})
        self.observed_rows_ = previous | current
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X) * self.gain


@pytest.fixture(autouse=True)
def isolated_threads_and_events():
    FIT_EVENTS.clear()
    with threadpool_limits(limits=1):
        yield
    FIT_EVENTS.clear()


def row_fixture():
    X = np.column_stack([np.arange(60), np.linspace(.1, 1., 60)])
    y = np.arange(60) % 2
    folds = list(StratifiedKFold(3, shuffle=True, random_state=23).split(X, y))
    return X, y, folds


@pytest.mark.parametrize("role", ["head", "transformer"])
@pytest.mark.parametrize("prefitted", [False, True])
def test_grid_replacement_has_fresh_state_per_fold_and_final_refit(role, prefitted):
    X, y, folds = row_fixture()
    if role == "head":
        replacement = RememberingHead(label=0)
        model = Pipeline([("head", DummyClassifier(strategy="most_frequent"))])
        grid = {"head": [replacement]}
        parameter = "label"
        original_value = 0
    else:
        replacement = RememberingTransformer(gain=1.)
        model = Pipeline([("features", RememberingTransformer()),
                          ("head", DummyClassifier(strategy="most_frequent"))])
        grid = {"features": [replacement], "features__gain": [2., 3.]}
        parameter = "gain"
        original_value = 1.
    if prefitted:
        replacement.fit(np.array([[900., 0.], [901., 1.]]), np.array([0, 1]))
    original_state = set(getattr(replacement, "observed_rows_", set()))
    FIT_EVENTS.clear()
    result = calibration_search(model, grid, X, y, cv=3, folds=folds, refit=True)
    candidates = 1 if role == "head" else 2
    assert len(FIT_EVENTS) == candidates * 3 + 1
    assert all(not event["before"] for event in FIT_EVENTS), "Previous fit state entered another fold/refit"
    assert len({id(event["object"]) for event in FIT_EVENTS}) == len(FIT_EVENTS)
    for index, event in enumerate(FIT_EVENTS[:-1]):
        train, validation = folds[index % 3]
        assert event["training"] == set(X[train, 0].astype(int))
        assert not event["training"] & set(X[validation, 0].astype(int))
    assert FIT_EVENTS[-1]["training"] == set(X[:, 0].astype(int))
    assert set(getattr(replacement, "observed_rows_", set())) == original_state
    assert getattr(replacement, parameter) == original_value
    if not prefitted:
        assert not hasattr(replacement, "observed_rows_")
    final_step = result.estimator.named_steps["head" if role == "head" else "features"]
    assert final_step is not replacement
    assert not result.report["is_fallback"]


def random_forest_fixture():
    rng = np.random.default_rng(20261103)
    X = rng.normal(size=(120, 6))
    y = rng.permutation(np.tile([0, 1], 60))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=23).split(X, y))
    forest = RandomForestClassifier(n_estimators=7, bootstrap=False, max_features=None,
                                    warm_start=True, random_state=17, n_jobs=1)
    return X, y, folds, forest


def test_warm_start_replacement_matches_sklearn_gridsearch_cv():
    X, y, folds, forest = random_forest_fixture()
    model = Pipeline([("head", DummyClassifier())])
    with warnings.catch_warnings(record=True):
        result = calibration_search(model, {"head": [forest]}, X, y, cv=3,
                                    folds=folds, refit=False)
    reference = GridSearchCV(model, {"head": [clone(forest)]}, cv=folds,
                             scoring="accuracy", refit=False, n_jobs=1).fit(X, y)
    expected = [reference.cv_results_[f"split{i}_test_score"][0] for i in range(3)]
    candidate = result.report["candidates"][0]
    assert candidate["valid"]
    np.testing.assert_array_equal([fold["score"] for fold in candidate["folds"]], expected)
    assert not hasattr(forest, "estimators_"), "The search fitted the caller's replacement object"


def test_scalar_grid_and_base_estimator_remain_isolated():
    X, y, folds = row_fixture()
    model = Pipeline([("head", RememberingHead())])
    result = calibration_search(model, {"head__label": [0, 1]}, X, y,
                                cv=3, folds=folds, refit=True)
    assert len(FIT_EVENTS) == 7
    assert all(not event["before"] for event in FIT_EVENTS)
    assert len({id(event["object"]) for event in FIT_EVENTS}) == 7
    assert not hasattr(model.named_steps["head"], "observed_rows_")
    assert result.report["selected_parameters"] == {"head__label": 0}
