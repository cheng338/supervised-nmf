"""Candidate identity must distinguish configurations independently of repr."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf.evaluation import calibration_search
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid


class ColumnPicker(TransformerMixin, BaseEstimator):
    def __init__(self, column=0):
        self.column = column

    def __repr__(self):
        # Repr is for display; sklearn does not require it to encode parameters.
        return "ColumnPicker()"

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X[:, [self.column]]


class ArrayColumnPicker(TransformerMixin, BaseEstimator):
    def __init__(self, configuration):
        self.configuration = configuration

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X[:, [int(self.configuration[600])]]


class AddressColumnPicker(ColumnPicker):
    def __repr__(self):
        return f"AddressColumnPicker(column={self.column}, address={id(self)})"


def fixture(kind="constant"):
    y = np.tile([0, 1], 30)
    X = np.column_stack([y, np.zeros(len(y))]).astype(float)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=817).split(X, y))
    if kind == "constant":
        choices = [ColumnPicker(0), ColumnPicker(1)]
    elif kind == "address":
        choices = [AddressColumnPicker(0), AddressColumnPicker(1)]
    else:
        first, second = np.zeros(1200), np.zeros(1200)
        second[600] = 1
        choices = [ArrayColumnPicker(first), ArrayColumnPicker(second)]
    base = Pipeline([("features", clone(choices[0])), ("knn", KNeighborsClassifier())])
    return X, y, folds, base, choices


def independent_scores(base, choices, X, y, folds):
    rows = []
    for params in ParameterGrid({"features": choices, "knn__n_neighbors": [1, 3]}):
        scores = []
        for train, valid in folds:
            fitted = clone(base).set_params(**clone(params, safe=False)).fit(X[train], y[train])
            scores.append(accuracy_score(y[valid], fitted.predict(X[valid])))
        rows.append(scores)
    return rows


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("kind", ["constant", "array"])
def test_shared_search_does_not_merge_repr_collisions(tmp_path, kind):
    X, y, folds, base, choices = fixture(kind)
    assert repr(choices[0]) == repr(choices[1])
    expected = independent_scores(base, choices, X, y, folds)
    assert expected == [[1.] * 3, [1.] * 3, [.5] * 3, [.5] * 3]
    report = calibration_knn_grid(base, {"features": choices, "knn__n_neighbors": [1, 3]},
                                  X, y, cv=3, folds=folds, checkpoint_dir=tmp_path)
    assert report["unique_representations"] == 2
    assert report["new_representation_fits"] == 6
    for row, reference in zip(report["candidates"], expected):
        assert row["valid"]
        np.testing.assert_array_equal([fold["score"] for fold in row["folds"]], reference)


def run_search(api, base, replacement, X, y, folds, path):
    grid = {"features": [replacement], "knn__n_neighbors": [1]}
    if api == "generic":
        return calibration_search(base, grid, X, y, cv=3, folds=folds, refit=False,
                                  checkpoint_path=path / "generic.json").report
    if api == "shared":
        return calibration_knn_grid(base, grid, X, y, cv=3, folds=folds,
                                     checkpoint_dir=path / "shared")
    base = Pipeline([("features", clone(base.named_steps["features"])),
                     ("pca", PCA(n_components=1, svd_solver="full", random_state=817)),
                     ("knn", KNeighborsClassifier())])
    grid["pca__n_components"] = [1]
    return calibration_rank_search(base, grid, X, y, cv=3, folds=folds,
                                    max_expansions=0, max_rank=1,
                                    checkpoint_dir=path / "rank")


@pytest.mark.parametrize("api", ["generic", "shared", "rank"])
def test_changed_constructor_values_reject_same_repr_checkpoint(tmp_path, api):
    X, y, folds, base, choices = fixture()
    first = run_search(api, base, choices[0], X, y, folds, tmp_path)
    assert first["selected_mean_score"] == 1.
    with pytest.raises(ValueError, match="incompatible"):
        run_search(api, base, choices[1], X, y, folds, tmp_path)


@pytest.mark.parametrize("api", ["generic", "shared", "rank"])
def test_equivalent_new_objects_can_resume_checkpoint(tmp_path, api):
    X, y, folds, base, choices = fixture()
    first = run_search(api, base, choices[0], X, y, folds, tmp_path)
    resumed = run_search(api, base, clone(choices[0]), X, y, folds, tmp_path)
    assert resumed["selected_mean_score"] == first["selected_mean_score"]
    assert resumed["fingerprint"] == first["fingerprint"]
    if api != "generic":
        assert resumed["new_representation_fits"] == 0
        assert resumed["reused_representation_folds"] == 3


@pytest.mark.parametrize("api", ["generic", "shared", "rank"])
def test_address_bearing_repr_does_not_prevent_equivalent_resume(tmp_path, api):
    X, y, folds, base, choices = fixture("address")
    equivalent = clone(choices[0])
    assert repr(equivalent) != repr(choices[0])
    assert equivalent.get_params() == choices[0].get_params()
    first = run_search(api, base, choices[0], X, y, folds, tmp_path)
    resumed = run_search(api, base, equivalent, X, y, folds, tmp_path)
    assert resumed["selected_mean_score"] == first["selected_mean_score"]
    assert resumed["fingerprint"] == first["fingerprint"]
    if api != "generic":
        assert resumed["new_representation_fits"] == 0
        assert resumed["reused_representation_folds"] == 3
