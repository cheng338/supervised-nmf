"""Replacement estimators must be isolated across shared CV folds and stages."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid


FIT_EVENTS = []


class AccumulatingTargetFeature(TransformerMixin, BaseEstimator):
    """A valid warm-start supervised transformer exposing stale fitted state.

    Its single feature is the known training label for a sample ID, or zero
    for a new ID. Independent CV fits cannot know a validation sample's label.
    Warm starting the same object across folds would expose earlier-fit labels.
    """

    def __init__(self, gain=1.0, warm_start=True):
        self.gain = gain
        self.warm_start = warm_start

    def fit(self, X, y):
        incoming = tuple(np.asarray(X[:, 0], dtype=int))
        old = dict(getattr(self, "known_", {})) if self.warm_start else {}
        FIT_EVENTS.append({"old": set(old), "incoming": set(incoming)})
        self.known_ = old
        self.known_.update(zip(incoming, np.asarray(y, dtype=int)))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return (self.gain * np.array([self.known_.get(int(row), 0)
                                     for row in X[:, 0]], dtype=float))[:, None]


def fixture():
    rng = np.random.default_rng(20261103)
    X = np.column_stack([np.arange(36), rng.uniform(.1, 1., (36, 3))])
    y = np.arange(36) % 2
    groups = np.repeat(np.arange(18), 2)
    folds = [(np.flatnonzero(groups % 3 != fold), np.flatnonzero(groups % 3 == fold))
             for fold in range(3)]
    return X, y, groups, folds


def test_shared_replacement_warm_start_has_fresh_fold_state_and_resumes(tmp_path):
    X, y, groups, folds = fixture()
    replacement = AccumulatingTargetFeature()
    estimator = Pipeline([("feature", AccumulatingTargetFeature()),
                          ("knn", KNeighborsClassifier())])
    grid = {"feature": [replacement], "knn__n_neighbors": [1, 3]}
    expected = {}
    with threadpool_limits(limits=1):
        for k in (1, 3):
            expected[k] = []
            for train, valid in folds:
                independent = clone(estimator).set_params(
                    feature=clone(replacement), knn__n_neighbors=k).fit(X[train], y[train])
                expected[k].append(accuracy_score(y[valid], independent.predict(X[valid])))
        FIT_EVENTS.clear()
        result = calibration_knn_grid(estimator, grid, X, y, cv=3, groups=groups,
                                      folds=folds, data_ids=np.arange(len(y)),
                                      checkpoint_dir=tmp_path / "shared")
    # The fixture itself has an exactly known fresh-fit result.
    assert expected == {1: [.5, .5, .5], 3: [.5, .5, .5]}
    assert len(FIT_EVENTS) == 3
    assert all(not event["old"] for event in FIT_EVENTS), FIT_EVENTS
    assert [event["incoming"] for event in FIT_EVENTS] == [set(train) for train, _ in folds]
    assert not hasattr(replacement, "known_")
    assert not hasattr(estimator.named_steps["feature"], "known_")
    for candidate in result["candidates"]:
        assert candidate["valid"]
        assert [row["score"] for row in candidate["folds"]] == expected[candidate["parameters"]["knn__n_neighbors"]]
    assert result["new_representation_fits"] == 3
    FIT_EVENTS.clear()
    resumed = calibration_knn_grid(estimator, grid, X, y, cv=3, groups=groups,
                                  folds=folds, data_ids=np.arange(len(y)),
                                  checkpoint_dir=tmp_path / "shared")
    assert not FIT_EVENTS
    assert resumed["reused_representation_folds"] == 3
    assert resumed["new_representation_fits"] == 0
    assert resumed["selected_mean_score"] == result["selected_mean_score"]


def test_shared_invalid_preflight_does_not_mutate_nested_replacement(tmp_path):
    X, y, groups, folds = fixture()
    replacement = AccumulatingTargetFeature(gain=1.)
    estimator = Pipeline([("feature", AccumulatingTargetFeature()),
                          ("knn", KNeighborsClassifier())])
    FIT_EVENTS.clear()
    with pytest.raises(ValueError, match="n_neighbors"):
        calibration_knn_grid(estimator, {"feature": [replacement], "feature__gain": [3.],
                                        "knn__n_neighbors": [0]}, X, y, cv=3,
                             groups=groups, folds=folds, checkpoint_dir=tmp_path)
    assert replacement.gain == 1.
    assert not FIT_EVENTS
    assert not hasattr(replacement, "known_")


def test_rank_search_preserves_replacement_estimators_across_real_expansion(tmp_path):
    X, y, groups, folds = fixture()
    # Remove the deliberately explicit sample-ID column for this ordinary PCA fit.
    X = X[:, 1:]
    replacement = PCA(n_components=None, svd_solver="full", random_state=817)
    estimator = Pipeline([("pca", PCA(svd_solver="full", random_state=817)),
                          ("knn", KNeighborsClassifier())])
    grid = {"pca": [replacement], "pca__n_components": [1], "knn__n_neighbors": [1, 3]}
    with threadpool_limits(limits=1):
        result = calibration_rank_search(estimator, grid, X, y, cv=3, groups=groups,
                                         folds=folds, data_ids=np.arange(len(y)),
                                         max_expansions=1, max_rank=2,
                                         checkpoint_dir=tmp_path / "rank")
    assert replacement.n_components is None
    assert not hasattr(replacement, "components_")
    assert result["expansions_performed"] == 1
    assert len(result["stages"]) == 2
    assert all(row["valid"] for row in result["candidates"])
    assert {row["parameters"]["pca__n_components"] for row in result["candidates"]} == {1, 2}
    assert not result["refit_performed"]
    with threadpool_limits(limits=1):
        resumed = calibration_rank_search(estimator, grid, X, y, cv=3, groups=groups,
                                          folds=folds, data_ids=np.arange(len(y)),
                                          max_expansions=1, max_rank=2,
                                          checkpoint_dir=tmp_path / "rank")
    assert resumed["new_representation_fits"] == 0
    assert resumed["reused_representation_folds"] == 6
    assert resumed["selected_parameters"] == result["selected_parameters"]
