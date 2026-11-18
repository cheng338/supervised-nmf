"""Impossible shared-kNN folds skip work without dropping declared evidence."""

from contextlib import ExitStack
from unittest.mock import patch
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import NMFClassifier, study
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.screening import BinaryScreening


NEIGHBOR_ERROR = "n_neighbors exceeds the number of observations in this training fold."
CUSTOM_WARNING = "Controlled preprocessing warning."
CUSTOM_ERROR = "Controlled validation projection failure."


@pytest.fixture
def problem():
    random = np.random.default_rng(812)
    y = np.arange(24) % 2
    coefficients = random.uniform(0.1, 1.0, (24, 2))
    coefficients[:, 0] += y
    basis = np.column_stack([np.eye(2), random.uniform(0.1, 1.0, (2, 6))])
    return coefficients @ basis, y


def model(layout):
    if layout == "raw":
        return KNeighborsClassifier(), "n_neighbors"
    if layout == "nmf":
        return NMFClassifier(solver="spa", screening="absolute", score_threshold=0,
                             n_components=2, random_state=7, max_iter=500), "n_neighbors"
    steps = [("screen", BinaryScreening(screening="absolute", score_threshold=0))]
    if layout == "pca":
        steps.append(("pca", PCA(n_components=2, svd_solver="full", random_state=7)))
    steps.append(("knn", KNeighborsClassifier()))
    return Pipeline(steps), "knn__n_neighbors"


def run_grid(problem, path, estimator, grid, order, **kwargs):
    X, y = problem
    return study.calibration_knn_grid(estimator, grid, X, y, cv=3, seed=7,
                                      threads=1, checkpoint_dir=path,
                                      execution_order=order, **kwargs)


def assert_neighbor_failure(row):
    assert row["status"] == "failed"
    assert row["error_type"] == "ValueError"
    assert row["error"] == NEIGHBOR_ERROR
    assert row["failure_stage"] == "neighbor_feasibility"
    assert row["preflight_rejected"] is True
    assert row["warnings"] == []
    assert row.get("score") is None
    assert row.get("diagnostics", {}) == {}


def assert_zero_work(report):
    assert report["state"] == "cv_completed"
    assert report["selected_parameters"] is None
    assert report["selected_mean_score"] is None
    assert report["ranked_candidate_indices"] == []
    assert report["new_representation_fits"] == 0
    assert report["unique_representation_fits"] == 0
    assert report["screening_statistics_fits"] == 0
    assert report["preflight_rejected_representation_folds"] == 3
    assert report["tuning_seconds"] == 0
    assert all(value == 0 for value in report["compute_seconds"].values())
    assert len(report["representation_folds"]) == 3
    for row in report["representation_folds"]:
        assert_neighbor_failure(row)
        assert row["state"] == "completed"
        assert row["representation_fit_performed"] is False
        assert row["screening_statistics_fit_performed"] is False
        assert row["neighbors"] == {}
        assert row["neighbor_queries"] == []
        for field in ("representation_fit_seconds", "projection_seconds",
                      "screening_statistics_seconds", "screening_selection_seconds",
                      "neighbor_reuse_setup_seconds", "neighbor_query_seconds"):
            assert row[field] == 0
    for candidate in report["candidates"]:
        assert candidate["valid"] is False
        assert candidate["mean_score"] is None
        assert candidate["invalid_fold_count"] == 3
        assert len(candidate["folds"]) == 3
        for row in candidate["folds"]:
            assert_neighbor_failure(row)
            assert row["fit_seconds"] == row["prediction_seconds"] == 0


@pytest.mark.parametrize("order", ["representation", "fold"])
@pytest.mark.parametrize("layout", ["raw", "screen", "pca", "nmf"])
def test_all_infeasible_neighbors_skip_work_and_keep_records(problem, tmp_path, layout, order):
    estimator, parameter = model(layout)
    X, y = problem
    original_X, original_y = X.copy(), y.copy()
    # Explicit call assertions also catch exceptions swallowed into failure rows.
    targets = [(BinaryScreening, "_fit_statistics"), (PCA, "fit_transform"),
               (FeatureSPA, "fit_transform"), (NMFClassifier, "fit"),
               (NMFClassifier, "_fit_with_screening"), (KNeighborsClassifier, "fit"),
               (study, "_stored_training_representation"),
               (study, "_validation_representation")]
    with ExitStack() as stack:
        observers = [stack.enter_context(patch.object(owner, name,
                     side_effect=AssertionError("Infeasible folds must skip numerical work")))
                     for owner, name in targets]
        report = run_grid(problem, tmp_path, estimator, {parameter: [99, 100]}, order)
    for observer in observers:
        observer.assert_not_called()
    assert [row["parameters"][parameter] for row in report["candidates"]] == [99, 100]
    assert_zero_work(report)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("order", ["representation", "fold"])
@pytest.mark.parametrize("layout", ["raw", "screen", "pca", "nmf"])
def test_mixed_neighbors_keep_feasible_scores_and_equality_boundary(problem, tmp_path, layout, order):
    estimator, parameter = model(layout)
    with patch.object(study, "_validation_representation",
                      wraps=study._validation_representation) as projection:
        report = run_grid(problem, tmp_path / "mixed", estimator, {parameter: [1, 16, 99]}, order)
    assert projection.call_count == 3
    assert {len(row["training_indices"]) for row in report["identity"]["folds"]} == {16}
    assert report["unique_representation_fits"] == 3
    assert report["preflight_rejected_representation_folds"] == 0
    assert all(row["status"] == "valid" for row in report["representation_folds"])
    assert [row["valid"] for row in report["candidates"]] == [True, True, False]
    for row in report["candidates"][-1]["folds"]:
        assert row["error"] == NEIGHBOR_ERROR
        assert row["fit_seconds"] == row["prediction_seconds"] == 0
        assert not row.get("preflight_rejected", False)
    reference = run_grid(problem, tmp_path / "valid", estimator, {parameter: [1, 16]}, order)
    assert report["selected_parameters"] == reference["selected_parameters"]
    assert report["selected_mean_score"] == reference["selected_mean_score"]
    for actual, expected in zip(report["candidates"][:2], reference["candidates"]):
        assert actual["parameters"] == expected["parameters"]
        assert actual["mean_score"] == expected["mean_score"]
        for actual_fold, expected_fold in zip(actual["folds"], expected["folds"]):
            for field in ("status", "score", "warnings"):
                assert actual_fold[field] == expected_fold[field]
            # Diagnostics contain elapsed fit/projection times as well as results.
            for field in ("n_selected_features", "selected_feature_indices", "n_iter",
                          "converged", "last_transform_converged", "last_transform_error",
                          "decomposition", "fitted_classifier_type"):
                assert actual_fold["diagnostics"].get(field) == expected_fold["diagnostics"].get(field)


@pytest.mark.parametrize("order", ["representation", "fold"])
@pytest.mark.parametrize("layout", ["pca", "nmf"])
@pytest.mark.parametrize("fold_kind", ["explicit", "grouped"])
def test_uneven_folds_retain_feasible_scores_and_diagnostics(problem, tmp_path, layout, order, fold_kind):
    estimator, parameter = model(layout)
    indices = np.arange(len(problem[1]))
    if fold_kind == "explicit":
        parts = [indices[:8], indices[8:14], indices[14:]]
        options = {"folds": [(np.setdiff1d(indices, part), part) for part in parts]}
    else:
        options = {"groups": np.repeat(np.arange(7), [2, 2, 2, 2, 2, 4, 10])}
    with patch.object(study, "_validation_representation",
                      wraps=study._validation_representation) as projection:
        report = run_grid(problem, tmp_path, estimator, {parameter: [16]}, order, **options)
    training_sizes = [len(row["training_indices"]) for row in report["identity"]["folds"]]
    assert sum(size >= 16 for size in training_sizes) == 2
    assert projection.call_count == 2
    assert report["unique_representation_fits"] == 2
    assert report["screening_statistics_fits"] == 2
    assert report["preflight_rejected_representation_folds"] == 1
    assert len(report["candidates"]) == 1
    candidate = report["candidates"][0]
    assert candidate["parameters"][parameter] == 16
    assert candidate["valid"] is False
    assert candidate["mean_score"] is None
    assert candidate["invalid_fold_count"] == 1
    assert report["selected_parameters"] is None
    assert len(candidate["folds"]) == 3
    for size, row in zip(training_sizes, candidate["folds"]):
        if size < 16:
            assert_neighbor_failure(row)
        else:
            assert row["status"] == "valid"
            assert 0 <= row["score"] <= 1
            assert row["diagnostics"]["n_selected_features"] > 0
            assert row["diagnostics"]["selected_feature_indices"]
    if fold_kind == "grouped":
        for row in report["identity"]["folds"]:
            groups = options["groups"]
            assert not set(groups[row["training_indices"]]) & set(groups[row["validation_indices"]])


class _ObservedTransform(TransformerMixin, BaseEstimator):
    events = []

    def __init__(self, mode):
        self.mode = mode

    def fit(self, X, y=None):
        self.events.append("fit")
        self.n_features_in_ = X.shape[1]
        if self.mode == "warning":
            warnings.warn(CUSTOM_WARNING, RuntimeWarning)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return X

    def transform(self, X):
        self.events.append("transform")
        if self.mode == "projection_failure":
            raise ValueError(CUSTOM_ERROR)
        return X


@pytest.mark.parametrize("order", ["representation", "fold"])
@pytest.mark.parametrize("mode", ["warning", "projection_failure"])
@pytest.mark.parametrize("neighbors", [[99], [1, 99]], ids=["all_infeasible", "mixed"])
def test_custom_failure_precedence_depends_on_any_feasible_neighbor(problem, tmp_path, order, mode, neighbors):
    _ObservedTransform.events.clear()
    estimator = Pipeline([("custom", _ObservedTransform(mode)), ("knn", KNeighborsClassifier())])
    report = run_grid(problem, tmp_path, estimator, {"knn__n_neighbors": neighbors}, order)
    if neighbors == [99]:
        assert _ObservedTransform.events == []
        assert_zero_work(report)
        return
    assert _ObservedTransform.events.count("fit") == 3
    assert _ObservedTransform.events.count("transform") == 3
    assert report["unique_representation_fits"] == 3
    assert report["preflight_rejected_representation_folds"] == 0
    if mode == "projection_failure":
        assert report["selected_parameters"] is None
        for candidate in report["candidates"]:
            assert candidate["valid"] is False
            for row in candidate["folds"]:
                assert row["failure_stage"] == "projection"
                assert row["error_type"] == "ValueError"
                assert row["error"] == CUSTOM_ERROR
    else:
        assert [row["valid"] for row in report["candidates"]] == [True, False]
        for candidate in report["candidates"]:
            for row in candidate["folds"]:
                assert {"category": "RuntimeWarning", "message": CUSTOM_WARNING} in row["warnings"]
        assert all(row["error"] == NEIGHBOR_ERROR for row in report["candidates"][1]["folds"])
