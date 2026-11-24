"""Recovery with a private split stream or an estimator that spawns child streams."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import ExtraTreeClassifier

from supervised_nmf import study
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid


class ChildMap(TransformerMixin, BaseEstimator):
    def __init__(self, generator):
        self.generator = generator

    def fit(self, X, y=None):
        child = self.generator.spawn(1)[0]
        self.components_ = child.normal(size=(X.shape[1], 4))
        return self

    def transform(self, X):
        return X @ self.components_


@pytest.fixture
def data():
    rng = np.random.RandomState(94)
    return rng.uniform(size=(60, 5)), rng.permutation(np.arange(60) % 2)


def test_equivalent_private_split_rng_resumes_and_advanced_state_refuses(tmp_path, data):
    X, y = data
    path = tmp_path / "cv.json"

    def search(seed):
        return calibration_search(ExtraTreeClassifier(random_state=61), {"max_depth": [2, 4]},
                                  X, y, seed=seed, cv=3, refit=False, threads=1,
                                  checkpoint_path=path).report

    first = search(np.random.RandomState(19))
    second = search(np.random.RandomState(19))
    assert first["candidates"] == second["candidates"]
    saved = path.read_bytes()
    advanced = np.random.RandomState(19)
    advanced.random_sample()
    with pytest.raises(ValueError, match="incompatible"):
        search(advanced)
    assert path.read_bytes() == saved


@pytest.mark.parametrize("api", ["shared", "rank"])
def test_child_cursor_in_study_checkpoint(tmp_path, data, api):
    X, y = data

    def search(generator):
        pipeline = Pipeline([("map", ChildMap(generator)), ("pca", PCA(random_state=19)),
                             ("knn", KNeighborsClassifier(n_jobs=1))])
        grid = {"pca__n_components": [2], "knn__n_neighbors": [1, 3]}
        kwargs = dict(seed=9, threads=1, cv=3, checkpoint_dir=tmp_path)
        if api == "rank":
            return calibration_rank_search(pipeline, grid, X, y, max_expansions=0, **kwargs)["stages"][0]
        return calibration_knn_grid(pipeline, grid, X, y, **kwargs)

    first = search(np.random.default_rng(31))
    assert all(row["valid"] for row in first["candidates"])
    if np.__version__.split(".", 1)[0] == "2":
        second = search(np.random.default_rng(31))
        assert [r["mean_score"] for r in second["candidates"]] == [
            r["mean_score"] for r in first["candidates"]]
    else:
        with pytest.raises(ValueError):
            search(np.random.default_rng(31))
    files = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    advanced = np.random.default_rng(31)
    advanced.spawn(1)
    with pytest.raises(ValueError):
        search(advanced)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == files


@pytest.mark.parametrize("missing_earlier_fold", [False, True])
def test_changed_rng_in_expanded_rank_is_checked_before_reusing_earlier_stage(tmp_path, missing_earlier_fold):
    class Interrupted(BaseException):
        pass

    data_rng = np.random.RandomState(7421)
    X = data_rng.normal(size=(90, 6))
    y = (X[:, 0] * X[:, 1] - .6 * X[:, 2] + data_rng.normal(size=len(X)) * .4 > 0).astype(int)

    def run():
        private = np.random.RandomState(51)

        def project(X):
            if X.shape[1] > 2:
                return X * private.uniform(.1, 1.9, size=X.shape[1])
            return X

        estimator = Pipeline([("pca", PCA(random_state=23)), ("map", FunctionTransformer(project)),
                              ("head", KNeighborsClassifier(n_jobs=1))])
        return calibration_rank_search(estimator, {"pca__n_components": [2], "head__n_neighbors": [3]},
                                       X, y, max_expansions=1, checkpoint_dir=tmp_path, cv=3, seed=53, threads=1)

    publish = study._write_receipt

    def stop_in_expansion(file, record):
        publish(file, record)
        if file.parent == tmp_path / "stage-001/folds":
            raise Interrupted

    with patch.object(study, "_write_receipt", stop_in_expansion), pytest.raises(Interrupted):
        run()
    if missing_earlier_fold:
        next((tmp_path / "stage-000/folds").glob("*.json")).unlink()
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    reason = ("captured random source" if np.__version__.split(".", 1)[0] == "2"
              else "Cannot verify the fit generator in this saved rank search")
    with patch.object(Pipeline, "fit") as fits, patch.object(study, "_write_receipt") as writes:
        with pytest.raises(ValueError, match=reason):
            run()
        fits.assert_not_called()
        writes.assert_not_called()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == original
