"""Column and transformer output contracts for fresh and resumed searches."""

import json
import numpy as np
import pytest
from sklearn import config_context, set_config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier, evaluation, study
from supervised_nmf.search import calibration_rank_search


@pytest.fixture
def problem():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(181)
    y = np.arange(30) % 2
    X = rng.uniform(0.01, 0.2, (len(y), 4))
    X[:, 0] += 5 * y
    X[:, 1] += 5 * (1 - y)
    return pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"]), y


def coordinate_pipeline(mode=None):
    spa = FeatureSPA(n_components=2).set_output(transform=mode)
    selector = ColumnTransformer([("component", "passthrough", ["featurespa0"])])
    return Pipeline([("spa", spa), ("select", selector), ("knn", KNeighborsClassifier(n_neighbors=1))])


def search(method, directory, model, X, y, validation):
    grid = {"spa__n_components": [2], "knn__n_neighbors": [1]}
    if method == "comparison":
        result = evaluation.run_comparison(
            {"method": (model, grid)}, X, y, validation, y, output_dir=directory, seed=42)
        return not result["models"]["method"]["is_fallback"]
    if method == "calibration":
        result = evaluation.calibration_search(
            model, grid, X, y, checkpoint_path=directory / "cv.json", seed=42, cv=2)
        return not result.report["is_fallback"]
    options = dict(checkpoint_dir=directory, seed=42, cv=2)
    if method == "rank":
        result = calibration_rank_search(model, grid, X, y, max_expansions=0, **options)
    else:
        result = study.calibration_knn_grid(model, grid, X, y, **options)
    return result["selected_mean_score"] is not None


def saved_files(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("method", ["calibration", "grid", "rank", "comparison"])
def test_renamed_columns_invalidate_checkpoint_with_identical_values(problem, tmp_path, method):
    frame, y = problem
    model = coordinate_pipeline("pandas")
    assert search(method, tmp_path, model, frame, y, frame)
    saved = saved_files(tmp_path)
    renamed = frame.rename(columns={"alpha": "renamed"})
    with pytest.raises(ValueError, match="incompatible"):
        search(method, tmp_path, model, renamed, y, renamed)
    assert saved_files(tmp_path) == saved
    assert search(method, tmp_path, model, frame, y, frame)


@pytest.mark.parametrize(("solver", "mu_projection", "normalization"), [("cd", "auto", "none"), ("mu", "batch", "sd")])
def test_named_nmf_grid_matches_ordinary_calibration_in_pandas_output(problem, tmp_path, solver, mu_projection,
                                                                     normalization):
    frame, y = problem
    model = NMFClassifier(solver=solver, mu_projection=mu_projection, n_components=2, screening="absolute",
                          score_threshold=0, random_state=7, max_iter=3000, tol=1e-3, knn_normalization=normalization)
    grid = {"n_neighbors": [1, 3]}
    with config_context(transform_output="pandas"):
        ordinary = evaluation.calibration_search(model, grid, frame, y, cv=2, seed=42, refit=False)
        shared = study.calibration_knn_grid(model, grid, frame, y, cv=2, seed=42, checkpoint_dir=tmp_path)
    for expected, actual in zip(ordinary.report["candidates"], shared["candidates"], strict=True):
        assert expected["valid"] and actual["valid"]
        assert [fold["score"] for fold in actual["folds"]] == [fold["score"] for fold in expected["folds"]]


@pytest.mark.parametrize("method", ["calibration", "grid", "rank", "comparison"])
@pytest.mark.parametrize(("setting", "first", "second"), [
    ("instance", "default", "pandas"), ("global", "pandas", "default")])
def test_output_mode_changes_fresh_results_and_invalidates_reuse(problem, tmp_path, method, setting, first, second):
    frame, y = problem
    X = frame if setting == "instance" else frame.to_numpy()
    # The global case includes array training and a validation-only named frame.
    def invoke(path, mode):
        path.mkdir(exist_ok=True)
        with config_context(transform_output=mode if setting == "global" else "default"):
            model = coordinate_pipeline(mode if setting == "instance" else None)
            return search(method, path, model, X, y, frame)

    saved = tmp_path / "saved"
    assert invoke(saved, first) == (first == "pandas")
    before = saved_files(saved)
    with pytest.raises(ValueError, match="incompatible"):
        invoke(saved, second)
    assert saved_files(saved) == before
    assert invoke(tmp_path / "fresh", second) == (second == "pandas")
    assert invoke(saved, first) == (first == "pandas")


@pytest.mark.parametrize("modes", [("default", "pandas"), ("pandas", "default")])
def test_grid_replacements_have_separate_representation_folds(problem, tmp_path, modes):
    frame, y = problem
    variants = [FeatureSPA(n_components=2).set_output(transform=mode) for mode in modes]
    grid = {"spa": variants, "knn__n_neighbors": [1, 2]}
    with config_context(transform_output="default"):
        result = study.calibration_knn_grid(coordinate_pipeline(), grid, frame, y,
                                            cv=2, seed=42, checkpoint_dir=tmp_path)
    assert result["unique_representations"] == 2
    assert result["new_representation_fits"] == 4
    assert [row["valid"] for row in result["candidates"]] == [mode == "pandas" for _ in (1, 2) for mode in modes]


def test_partial_output_override_mismatch_keeps_committed_fold(problem, tmp_path, monkeypatch):
    frame, y = problem
    path = tmp_path / "cv.json"
    original_write = evaluation.write_json

    class FoldCommitted(BaseException):
        pass

    def stop_after_fold(destination, report):
        original_write(destination, report)
        if any(candidate["folds"] for candidate in report["candidates"]):
            raise FoldCommitted

    def invoke(mode):
        return evaluation.calibration_search(coordinate_pipeline(mode), {}, frame, y, cv=2,
                                             checkpoint_path=path)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "write_json", stop_after_fold)
        with pytest.raises(FoldCommitted):
            invoke("pandas")
    saved = path.read_bytes()
    fold, = json.loads(saved)["candidates"][0]["folds"]
    assert fold["status"] == "valid"
    with pytest.raises(ValueError, match="incompatible"):
        invoke("default")
    assert path.read_bytes() == saved
    completed = invoke("pandas")
    assert completed.report["candidates"][0]["folds"][0] == fold
    assert completed.report["candidates"][0]["valid"]


class SetOutputDuringFit(ClassifierMixin, BaseEstimator):
    def fit(self, X, y=None):
        set_config(transform_output="pandas")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


@pytest.mark.parametrize("method", ["calibration", "grid"])
def test_fit_changing_global_output_cannot_publish_a_fold(problem, tmp_path, method, monkeypatch):
    frame, y = problem
    with config_context(transform_output="default"), pytest.raises(ValueError, match="transform output|runtime changed"):
        if method == "calibration":
            evaluation.calibration_search(SetOutputDuringFit(), {}, frame, y, cv=2, checkpoint_path=tmp_path / "cv.json")
        else:
            original = study._run_representation_fold

            def change_output_after_fit(*args, **kwargs):
                original(*args, **kwargs)
                set_config(transform_output="pandas")

            monkeypatch.setattr(study, "_run_representation_fold", change_output_after_fit)
            study.calibration_knn_grid(KNeighborsClassifier(n_neighbors=1), {}, frame, y, cv=2,
                                       checkpoint_dir=tmp_path)
    if method == "calibration":
        assert json.loads((tmp_path / "cv.json").read_bytes())["candidates"][0]["folds"] == []
    else:
        assert not list((tmp_path / "folds").glob("*.json"))


@pytest.mark.parametrize(("global_mode", "local_mode"), [("polars", "default"), ("default", "pandas")])
def test_prepared_screen_preserves_candidate_output_and_names(problem, tmp_path, global_mode, local_mode,
                                                             monkeypatch):
    pd = pytest.importorskip("pandas")
    X, y = problem
    screen = BinaryScreening(score_threshold=0).set_output(transform=local_mode)
    model = Pipeline([("screen", screen), ("knn", KNeighborsClassifier(n_neighbors=1))])
    with config_context(transform_output=global_mode):
        control = evaluation.calibration_search(model, {}, X, y, cv=2, refit=False)
        assert control.report["candidates"][0]["valid"]
        observed = []
        original_transform = study._PreparedScreening.transform

        def record_output(self, data):
            result = original_transform(self, data)
            observed.append((type(result), self.get_feature_names_out().tolist()))
            if local_mode == "pandas":
                assert result.columns.tolist() == self.get_feature_names_out().tolist()
            return result

        monkeypatch.setattr(study._PreparedScreening, "transform", record_output)
        shared = study.calibration_knn_grid(model, {}, X, y, cv=2, checkpoint_dir=tmp_path)
    scores = [f["score"] for f in control.report["candidates"][0]["folds"]]
    assert [f["score"] for f in shared["candidates"][0]["folds"]] == scores
    assert observed and all(cls is (pd.DataFrame if local_mode == "pandas" else np.ndarray)
                            and names == X.columns.tolist() for cls, names in observed)
