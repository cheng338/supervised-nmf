"""Explicit scoring contracts for shared representation grids.

The imbalanced fixture has analytically different accuracy/balanced-accuracy
winners. It exercises actual KNN predictions rather than replacing scorers.
"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf import study
from supervised_nmf._neighbors import _KNNQueryReuse


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def imbalanced():
    # Each fold has eighteen negatives and two positives. Repeating each
    # observation satisfies the minimum two-per-class training support while
    # preserving the analytically known nearest-neighbor predictions.
    # In the first validation direction, k=1 identifies both positives plus
    # incorrectly labels six negatives as positive: accuracy=.7, balanced
    # accuracy=5/6. In reverse, k=1 is perfect. k=9 always predicts zero:
    # accuracy=.9, BA=.5 in both folds.
    first = np.r_[np.arange(9), 100.0]
    second = np.r_[np.arange(6), [100.1, 100.2, 100.3], 100.0]
    X = np.repeat(np.r_[first, second], 2)[:, None]
    y = np.repeat(np.tile(np.r_[np.zeros(9, dtype=int), 1], 2), 2)
    folds = [(np.arange(20), np.arange(20, 40)),
             (np.arange(20, 40), np.arange(20))]
    return X, y, folds


def run_shared(imbalanced, directory, *, scoring=None, grid=None):
    X, y, folds = imbalanced
    kwargs = {} if scoring is None else {"scoring": scoring}
    return calibration_knn_grid(
        KNeighborsClassifier(algorithm="brute"),
        {"n_neighbors": [1, 9]} if grid is None else grid,
        X, y, cv=2, folds=folds, checkpoint_dir=directory, **kwargs,
    )


def test_default_accuracy_and_explicit_balanced_accuracy_have_different_winners(imbalanced, tmp_path):
    ordinary = run_shared(imbalanced, tmp_path / "accuracy")
    balanced = run_shared(imbalanced, tmp_path / "balanced", scoring="balanced_accuracy")
    assert ordinary["scoring"] == "accuracy"
    assert ordinary["selected_parameters"] == {"n_neighbors": 9}
    assert ordinary["selected_mean_score"] == pytest.approx(.9)
    assert ordinary["selected_mean_accuracy"] == pytest.approx(.9)
    assert "selected_mean_balanced_accuracy" not in ordinary
    assert [row["mean_accuracy"] for row in ordinary["candidates"]] == pytest.approx([.85, .9])
    assert balanced["selected_parameters"] == {"n_neighbors": 1}
    assert balanced["selected_mean_score"] == pytest.approx(11 / 12)
    assert balanced["selected_mean_balanced_accuracy"] == pytest.approx(11 / 12)
    assert "selected_mean_accuracy" not in balanced
    assert [row["mean_balanced_accuracy"] for row in balanced["candidates"]] == pytest.approx([11 / 12, .5])
    for report in (ordinary, balanced):
        metric = report["scoring"]
        other = "balanced_accuracy" if metric == "accuracy" else "accuracy"
        assert report["identity"]["scoring"] == metric
        assert report["identity"]["calibration"]["scoring"] == metric
        for candidate in report["candidates"]:
            assert candidate["mean_score"] == candidate["mean_" + metric]
            assert "mean_" + other not in candidate
            for fold in candidate["folds"]:
                assert fold["score"] == fold[metric]
                assert other not in fold


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_shared_grid_matches_ordinary_predictions_and_metric(imbalanced, tmp_path, monkeypatch, scoring):
    X, y, folds = imbalanced
    direct_predictions, reused_predictions = [], []
    original_predict = KNeighborsClassifier.predict
    original_reuse = _KNNQueryReuse.predict

    def observe_predict(self, rows):
        result = original_predict(self, rows)
        direct_predictions.append((self.n_neighbors, tuple(rows[:, 0]), tuple(result)))
        return result

    with monkeypatch.context() as context:
        context.setattr(KNeighborsClassifier, "predict", observe_predict)
        ordinary = calibration_search(KNeighborsClassifier(algorithm="brute"),
                                      {"n_neighbors": [1, 9]}, X, y, cv=2,
                                      folds=folds, scoring=scoring, refit=False).report

    def observe_reuse(self, k, **kwargs):
        result = original_reuse(self, k, **kwargs)
        reused_predictions.append((k, tuple(self.query[:, 0]), tuple(result[0])))
        return result

    monkeypatch.setattr(_KNNQueryReuse, "predict", observe_reuse)
    shared = run_shared(imbalanced, tmp_path / scoring, scoring=scoring)
    assert sorted(direct_predictions) == sorted(reused_predictions)
    assert shared["selected_parameters"] == ordinary["selected_parameters"]
    assert [row["mean_score"] for row in shared["candidates"]] == pytest.approx(
        [row["mean_score"] for row in ordinary["candidates"]])
    assert shared["unique_representation_fits"] == 2


def test_same_metric_resumes_without_fits_and_changed_metric_is_rejected(imbalanced, tmp_path, monkeypatch):
    directory = tmp_path / "resume"
    initial = run_shared(imbalanced, directory)

    def no_fitting(*args, **kwargs):
        raise AssertionError("No new representation work is allowed on complete resume.")

    monkeypatch.setattr(study, "_run_representation_fold", no_fitting)
    resumed = run_shared(imbalanced, directory, scoring="accuracy")
    assert resumed["new_representation_fits"] == 0
    assert resumed["reused_representation_folds"] == 2
    assert resumed["compute_seconds"] == initial["compute_seconds"]
    with pytest.raises(ValueError, match="incompatible"):
        run_shared(imbalanced, directory, scoring="balanced_accuracy")


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_failed_neighbor_candidate_never_ranked_using_partial_folds(imbalanced, tmp_path, monkeypatch, scoring):
    original = _KNNQueryReuse.predict

    def fail_second_fold(self, k, **kwargs):
        if k == 9 and np.max(self.query[:, 0]) == 100.0:
            raise ValueError("Deliberate one-fold query failure")
        return original(self, k, **kwargs)

    monkeypatch.setattr(_KNNQueryReuse, "predict", fail_second_fold)
    report = run_shared(imbalanced, tmp_path / scoring, scoring=scoring)
    failed = report["candidates"][1]
    assert not failed["valid"]
    assert failed["invalid_fold_count"] == 1
    assert failed["mean_score"] is None
    assert failed["mean_" + scoring] is None
    assert {row["status"] for row in failed["folds"]} == {"valid", "failed"}
    assert report["selected_parameters"] == {"n_neighbors": 1}
    assert report["ranked_candidate_indices"] == [0]


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_all_failed_candidates_keep_selected_score_unavailable(imbalanced, tmp_path, scoring):
    report = run_shared(imbalanced, tmp_path / scoring, scoring=scoring,
                        grid={"n_neighbors": [21]})
    assert report["selected_parameters"] is None
    assert report["selected_mean_score"] is None
    assert report["selected_mean_" + scoring] is None
    assert not report["candidates"][0]["valid"]
    assert report["candidates"][0]["invalid_fold_count"] == 2


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_scoring_reuses_same_two_projections_per_representation_fold(tmp_path, monkeypatch, scoring):
    X = np.column_stack((np.arange(1, 25, dtype=float), np.arange(1, 25, dtype=float) * 2))
    y = np.tile([0, 1], 12)
    original = FeatureSPA.transform
    observed = []

    def observe(self, rows):
        observed.append(rows.shape)
        return original(self, rows)

    monkeypatch.setattr(FeatureSPA, "transform", observe)
    report = calibration_knn_grid(
        NMFClassifier(screening="none", n_components=1, random_state=3),
        {"n_neighbors": [1, 3]}, X, y, cv=2, scoring=scoring,
        checkpoint_dir=tmp_path / scoring,
    )
    assert report["unique_representation_fits"] == 2
    # Per fold: the default SPA classifier projects its training rows once;
    # the shared evaluator projects validation rows once for both KNN sizes.
    assert observed == [(12, 2)] * 4
    assert all(candidate["valid"] for candidate in report["candidates"])


@pytest.mark.parametrize("scoring", [None, "roc_auc", "Accuracy", 1, ["accuracy"]])
def test_invalid_metric_fails_before_runtime_or_fit(imbalanced, tmp_path, monkeypatch, scoring):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid scoring must be rejected before runtime inspection.")

    monkeypatch.setattr(study, "_runtime_identity", forbidden)
    X, y, folds = imbalanced
    with pytest.raises(ValueError, match="scoring"):
        calibration_knn_grid(KNeighborsClassifier(), {"n_neighbors": [1]},
                             X, y, cv=2, folds=folds, scoring=scoring,
                             checkpoint_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_shared_grid_warns_once_and_marks_inactive_cutoff_refinement(tmp_path):
    X = np.column_stack((np.arange(1, 25, dtype=float), np.arange(1, 25, dtype=float) * 2))
    y = np.tile([0, 1], 12)
    with pytest.warns(UserWarning, match="Inactive screening grid") as caught:
        report = calibration_knn_grid(
            NMFClassifier(screening="none", n_components=1, random_state=3),
            {"score_threshold": [1, 2], "n_neighbors": [1]}, X, y, cv=2,
            checkpoint_dir=tmp_path / "inactive",
        )
    assert len(caught) == 1
    assert report["screening_diagnostics"]["inactive_parameters"]
    assert report["refinement"]["parameters"]["score_threshold"]["status"] == "inactive_for_selected_screening_mode"
    assert len(report["candidates"]) == 2
    assert report["unique_representation_fits"] == 4
