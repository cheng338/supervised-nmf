"""Interactions between released 0.4.3 fixes and the November 3 API changes."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier, _nnls
from supervised_nmf.evaluation import calibration_search, evaluate_classifier
from supervised_nmf.study import calibration_knn_grid


@pytest.fixture(autouse=True)
def one_numerical_thread():
    with threadpool_limits(limits=1):
        yield


def grouped_problem():
    rows = np.arange(16, dtype=float)
    y = np.tile([0, 0, 0, 1], 4)
    X = np.column_stack([30 * y + rows / 100, 2 + np.sin(rows), 2 + np.cos(rows)])
    left, right = np.arange(8), np.arange(8, 16)
    return X, y, [(left, right), (right, left)]


@pytest.mark.parametrize("runner", ["general", "shared"])
@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_extended_metadata_resume_preserves_structural_candidates_and_scoring(tmp_path, runner, scoring):
    X, y, folds = grouped_problem()
    replacements = [PCA(n_components=rank, svd_solver="full", random_state=7) for rank in (1, 2)]
    model = Pipeline([("pca", PCA(random_state=7)), ("knn", KNeighborsClassifier(n_neighbors=3))])
    settings = dict(cv=2, threads=1, seed=7, folds=folds,
                    groups=np.repeat(np.arange(4, dtype=np.longdouble), 4),
                    data_ids=np.arange(16, dtype=np.longdouble) + np.longdouble("0.5"))

    def search(metric):
        if runner == "general":
            return calibration_search(model, {"pca": replacements}, X, y, scoring=metric,
                                      refit=False, checkpoint_path=tmp_path / "cv.json", **settings).report
        return calibration_knn_grid(model, {"pca": replacements}, X, y, scoring=metric,
                                    checkpoint_dir=tmp_path / "shared", **settings)

    first = search(scoring)
    assert all(row["valid"] for row in first["candidates"])
    identity = first["identity"] if runner == "general" else first["identity"]["calibration"]
    assert identity["identity_schema"] == 2
    assert identity["structural_configuration"]
    assert identity["implementation_source"]
    assert identity["scoring"] == scoring
    assert all(row["mean_score"] == row["mean_" + scoring] for row in first["candidates"])
    assert all(not hasattr(value, "components_") for value in replacements)
    if runner == "shared":
        assert first["unique_representations"] == 2
        assert first["new_representation_fits"] == 4
    with (patch.object(PCA, "fit", side_effect=AssertionError("Resume refitted PCA")) as fit,
          patch.object(PCA, "fit_transform", side_effect=AssertionError("Resume refitted PCA")) as transform,
          patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("Resume refitted KNN")) as head_fit):
        resumed = search(scoring)
        fit.assert_not_called()
        transform.assert_not_called()
        head_fit.assert_not_called()
    assert resumed["fingerprint"] == first["fingerprint"]
    assert resumed["candidates"] == first["candidates"]
    if runner == "shared":
        assert resumed["reused_representation_folds"] == 4
        assert resumed["new_representation_fits"] == 0
    other_metric = "balanced_accuracy" if scoring == "accuracy" else "accuracy"
    with pytest.raises(ValueError, match="incompatible"):
        search(other_metric)


def test_spa_fit_transform_projects_once_and_uses_released_batched_cross_reuse():
    rng = np.random.default_rng(419)
    coordinates = rng.uniform(.1, 1, (64, 3))
    basis = np.column_stack([np.eye(3), rng.uniform(.1, 1, (3, 37))])
    X, y = coordinates @ basis, np.arange(64) % 2
    model = NMFClassifier(screening="none", n_components=3, n_neighbors=3,
                          projection_solver="batched", random_state=7, max_iter=1000)
    original_transform, original_qualification = FeatureSPA.transform, _nnls._qualified_rows
    projected, qualification_crosses = [], []

    def transform(estimator, values):
        projected.append(len(values))
        return original_transform(estimator, values)

    def qualify(design, targets, coefficients, **kwargs):
        qualification_crosses.append(kwargs.get("cross_scale"))
        return original_qualification(design, targets, coefficients, **kwargs)

    with (patch.object(FeatureSPA, "transform", transform),
          patch.object(_nnls, "_qualified_rows", qualify)):
        actual = model.fit_transform(X, y)
    assert projected == [len(X)]
    assert qualification_crosses and all(value is not None for value in qualification_crosses)
    expected = np.vstack([nnls(model.selected_components_.T, row, maxiter=1000)[0] for row in X])
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(actual, model.training_coordinates_)
    assert not np.shares_memory(actual, model.training_coordinates_)


def test_exact_mixed_numeric_overlap_rejected_before_spa_joint_prediction():
    X = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 1.], [1., 1.]])
    y = np.arange(len(X)) % 2
    model = NMFClassifier(screening="none", n_components=2, n_neighbors=2,
                          weights="distance", random_state=7).fit(X, y)
    training = np.array([np.float64(2**53), *range(100, 105)], dtype=object)
    evaluation = np.array([2**53, *range(200, 205)], dtype=object)
    with (patch.object(model, "transform", side_effect=AssertionError("Overlap reached projection")) as transform,
          patch.object(model, "predict_with_probabilities", side_effect=AssertionError("Overlap reached prediction")) as predict):
        with pytest.raises(ValueError, match="sample IDs must not overlap"):
            evaluate_classifier(model, X, y, sample_ids=evaluation, train_sample_ids=training)
        transform.assert_not_called()
        predict.assert_not_called()


def test_released_refinement_rules_combine_with_inactive_screening_diagnostics():
    X, y, folds = grouped_problem()
    model = Pipeline([("screen", BinaryScreening(score_threshold=0)),
                      ("knn", KNeighborsClassifier(n_neighbors=3))])
    # Failed None/NumPy-bool candidates remain declared alternatives, while an
    # inactive percentile must not receive a recommendation to expand its grid.
    grid = [{"screen__score_threshold": [cutoff], "screen__percentile": [percentile]}
            for cutoff, percentile in [(0., 10.), (2., 20.), (None, 10.), (False, 10.), (np.bool_(False), 10.)]]
    with pytest.warns(UserWarning, match="Inactive screening"):
        report = calibration_search(model, grid, X, y, cv=2, folds=folds, refit=False).report
    assert report["selected_candidate_index"] == 0
    assert report["candidates"][2]["valid"] is False
    refinement = report["refinement"]["parameters"]
    assert refinement["screen__score_threshold"]["values"] == [0., 2.]
    assert refinement["screen__score_threshold"]["includes_none"] is True
    assert refinement["screen__score_threshold"]["status"] == "lower_boundary_requires_expansion_or_constraint"
    assert refinement["screen__percentile"]["status"] == "inactive_for_selected_screening_mode"


def test_weighted_tie_fallback_reports_fitted_policy_with_one_projection():
    X = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 1.], [1., 1.]])
    y = np.arange(len(X)) % 2
    model = NMFClassifier(screening="none", n_components=2, n_neighbors=2,
                          weights="distance", positive_label=0, random_state=7).fit(X, y)
    # Pending constructor changes must not alter the saved decision policy.
    model.set_params(decision_threshold=.9, positive_label=1, mu_projection="rowwise")
    with (patch.object(model, "transform", wraps=model.transform) as transform,
          patch.object(model.classifier_, "predict", wraps=model.classifier_.predict) as native_predict):
        report = evaluate_classifier(model, X, y)
    assert report["status"] == "valid"
    assert transform.call_count == 1
    assert native_predict.call_count == 1
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], np.zeros(len(X), dtype=int))
    np.testing.assert_array_equal(report["predictions"]["positive_class_probability"], np.full(len(X), .5))
    assert {key: report["diagnostics"][key] for key in
            ("decision_threshold", "positive_label", "positive_class_index", "mu_projection")} == {
                "decision_threshold": None, "positive_label": 0,
                "positive_class_index": 0, "mu_projection": None}
