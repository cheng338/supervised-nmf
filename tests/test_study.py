"""Equivalence, leakage and cache checks for explicit shared-KNN grids."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.screening import BinaryScreening
from supervised_nmf.study import calibration_knn_grid


class SharedKNNGridTests(unittest.TestCase):
    def setUp(self):
        random = np.random.RandomState(23)
        self.y = np.arange(80) % 2
        coefficients = random.uniform(0.1, 1.0, (80, 2))
        coefficients[:, 0] += self.y
        basis = np.column_stack([np.eye(2), random.uniform(0.1, 1.0, (2, 10))])
        self.X = (coefficients @ basis + random.uniform(0, 0.005, (80, 12))).astype(np.float32)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def shared(self, estimator, grid, name="shared", **kwargs):
        return calibration_knn_grid(estimator, grid, self.X, self.y,
                                    checkpoint_dir=Path(self.temp.name) / name, seed=19, **kwargs)

    def assert_equivalent(self, estimator, grid, name):
        original_predict = KNeighborsClassifier.predict
        observed = {}

        def capture(classifier, X):
            predictions = original_predict(classifier, X)
            key = (classifier.n_neighbors, classifier._fit_X.shape[1])
            observed.setdefault(key, []).append(predictions.copy())
            return predictions

        with patch.object(KNeighborsClassifier, "predict", capture):
            ordinary = calibration_search(estimator, grid, self.X, self.y, cv=5, seed=19, refit=False).report
        ordinary_predictions = observed
        observed = {}
        with patch.object(KNeighborsClassifier, "predict", capture):
            shared = self.shared(estimator, grid, name)
        self.assertEqual(set(observed), set(ordinary_predictions))
        for key, folds in ordinary_predictions.items():
            self.assertEqual(len(folds), len(observed[key]))
            for expected, actual in zip(folds, observed[key]):
                np.testing.assert_array_equal(expected, actual)
        self.assertEqual(len(ordinary["candidates"]), len(shared["candidates"]))
        for expected, actual in zip(ordinary["candidates"], shared["candidates"]):
            self.assertEqual(expected["parameters"], actual["parameters"])
            self.assertTrue(expected["valid"], expected)
            self.assertTrue(actual["valid"], actual)
            self.assertEqual(expected["mean_balanced_accuracy"], actual["mean_balanced_accuracy"])
            self.assertEqual([fold["balanced_accuracy"] for fold in expected["folds"]],
                             [fold["balanced_accuracy"] for fold in actual["folds"]])
            for expected_fold, actual_fold in zip(expected["folds"], actual["folds"]):
                for field in ("n_iter", "converged", "last_transform_converged",
                              "n_selected_features", "selected_feature_indices"):
                    self.assertEqual(expected_fold["diagnostics"].get(field), actual_fold["diagnostics"].get(field))
        self.assertEqual(shared["selected_parameters"], ordinary["selected_parameters"])
        self.assertFalse(shared["refit_performed"])
        self.assertFalse(shared["defaults_frozen"])
        return shared

    def test_cd_float32_matches_ordinary_grid_and_retains_coordinates(self):
        estimator = NMFClassifier(solver="cd", screening="percentile", percentile=100,
                                  n_components=2, random_state=7, max_iter=3000, tol=1e-3)
        result = self.assert_equivalent(estimator, {"n_neighbors": [1, 3, 5]}, "cd")
        self.assertEqual(result["unique_representation_fits"], 5)
        self.assertEqual(result["new_representation_fits"], 5)
        self.assertTrue(all(fold["training_representation_dtype"] == "float32" for fold in result["representation_folds"]))
        self.assertEqual(result["tuning_seconds"], sum(result["compute_seconds"].values()))

    def test_spa_matches_ordinary_grid(self):
        estimator = NMFClassifier(solver="spa", screening="percentile", percentile=100,
                                  n_components=2, random_state=7, weights="distance", max_iter=1000)
        result = self.assert_equivalent(estimator, {"n_neighbors": [1, 3]}, "spa")
        self.assertEqual(result["unique_representation_fits"], 5)

    def test_mu_float32_matches_ordinary_grid_and_retains_coordinates(self):
        estimator = NMFClassifier(solver="mu", screening="absolute", score_threshold=1.0,
                                  n_components=2, random_state=7, max_iter=3000, tol=1e-3)
        result = self.assert_equivalent(estimator, {"n_neighbors": [1, 3, 5]}, "mu")
        self.assertEqual(result["unique_representation_fits"], 5)
        self.assertEqual(result["new_representation_fits"], 5)
        self.assertTrue(all(fold["training_representation_dtype"] == "float32"
                            for fold in result["representation_folds"]))
        self.assertEqual(result["tuning_seconds"], sum(result["compute_seconds"].values()))

    def test_pca_pipeline_uses_original_fit_transform_coordinates(self):
        estimator = Pipeline([("screen", BinaryScreening(percentile=100)),
                              ("pca", PCA(n_components=2, svd_solver="full", random_state=7, whiten=True)),
                              ("knn", KNeighborsClassifier(weights="distance"))])
        grid = {"pca__n_components": [2, 3], "knn__n_neighbors": [1, 3]}
        result = self.assert_equivalent(estimator, grid, "pca")
        self.assertEqual(result["unique_representation_fits"], 10)
        original_transform = PCA.transform
        transformed_shapes = []

        def observe(instance, X):
            transformed_shapes.append(X.shape)
            return original_transform(instance, X)

        with patch.object(PCA, "transform", observe):
            self.shared(estimator, {"knn__n_neighbors": [1, 3, 5]}, "pca-count")
        self.assertEqual(len(transformed_shapes), 5)
        self.assertTrue(all(shape[0] == 16 for shape in transformed_shapes))

    def test_screening_only_and_raw_knn_match_ordinary_grid(self):
        estimator = Pipeline([("screen", BinaryScreening(percentile=50)),
                              ("knn", KNeighborsClassifier())])
        self.assert_equivalent(estimator, {"knn__n_neighbors": [1, 3, 5]}, "screen")
        self.assert_equivalent(KNeighborsClassifier(weights="distance"), {"n_neighbors": [1, 3, 5]}, "raw")

    def test_duplicate_points_and_distance_ties_preserve_each_prediction(self):
        self.X = np.repeat(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32), 20, axis=0)
        self.y = np.tile([0, 1], 40)
        self.assert_equivalent(KNeighborsClassifier(weights="distance"), {"n_neighbors": [1, 3, 40]}, "ties")

    def test_screening_is_fit_on_each_training_fold_once_without_refit(self):
        estimator = Pipeline([("screen", BinaryScreening(percentile=50)),
                              ("knn", KNeighborsClassifier())])
        observed = []
        original_fit = BinaryScreening.fit

        def observe(instance, X, y):
            observed.append((X.copy(), y.copy()))
            return original_fit(instance, X, y)

        with patch.object(BinaryScreening, "fit", observe):
            result = self.shared(estimator, {"knn__n_neighbors": [1, 3, 5]})
        self.assertEqual(len(observed), 5)
        for (X, y), fold in zip(observed, result["identity"]["folds"]):
            np.testing.assert_array_equal(X, self.X[fold["training_indices"]])
            np.testing.assert_array_equal(y, self.y[fold["training_indices"]])
            self.assertFalse(set(fold["training_indices"]) & set(fold["validation_indices"]))

    def test_representation_failures_invalidate_every_corresponding_neighbor_candidate(self):
        estimator = NMFClassifier(solver="cd", screening="absolute", score_threshold=1e20,
                                  n_components=2, random_state=7)
        result = self.shared(estimator, {"n_neighbors": [1, 3]})
        for candidate in result["candidates"]:
            self.assertFalse(candidate["valid"])
            self.assertEqual(candidate["invalid_fold_count"], 5)
            self.assertIsNone(candidate["mean_balanced_accuracy"])
        self.assertIsNone(result["selected_parameters"])
        self.assertFalse(result["is_fallback"])

    def test_nonconvergence_and_impossible_k_are_never_success_averaged(self):
        estimator = NMFClassifier(solver="cd", screening="none", n_components=2,
                                  random_state=7, max_iter=1, tol=1e-10)
        result = self.shared(estimator, {"n_neighbors": [1, 3]}, "nonconverged")
        self.assertTrue(all(not row["valid"] for row in result["candidates"]))
        self.assertTrue(any(row["warnings"] for row in result["representation_folds"]))
        raw = self.shared(KNeighborsClassifier(), {"n_neighbors": [1, 1000]}, "large-k")
        self.assertTrue(raw["candidates"][0]["valid"])
        self.assertFalse(raw["candidates"][1]["valid"])
        self.assertIsNone(raw["candidates"][1]["mean_balanced_accuracy"])

    def test_projection_warning_invalidates_all_neighbors_for_that_representation(self):
        estimator = NMFClassifier(solver="spa", screening="none", n_components=2, random_state=7)
        original_transform = NMFClassifier.transform

        def warn_on_projection(instance, X):
            coordinates = original_transform(instance, X)
            warnings.warn("test projection nonconvergence", ConvergenceWarning)
            return coordinates

        with patch.object(NMFClassifier, "transform", warn_on_projection):
            result = self.shared(estimator, {"n_neighbors": [1, 3]})
        self.assertTrue(all(not row["valid"] for row in result["candidates"]))
        self.assertTrue(all(row["status"] == "nonconverged" for row in result["representation_folds"]))
        self.assertTrue(all(not row["neighbors"] for row in result["representation_folds"]))

    def test_resume_reuses_folds_and_preserves_shared_compute_totals(self):
        estimator = Pipeline([("screen", BinaryScreening(percentile=50)),
                              ("knn", KNeighborsClassifier())])
        first = self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        with patch.object(BinaryScreening, "fit", side_effect=AssertionError("Completed fold should not refit")):
            resumed = self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertEqual(resumed["reused_representation_folds"], 5)
        self.assertEqual(first["compute_seconds"], resumed["compute_seconds"])
        self.assertEqual(first["candidates"], resumed["candidates"])

    def test_incompatible_grid_data_and_corrupt_receipts_are_rejected(self):
        estimator = KNeighborsClassifier()
        self.shared(estimator, {"n_neighbors": [1, 3]})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.shared(estimator, {"n_neighbors": [1, 5]})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            calibration_knn_grid(estimator, {"n_neighbors": [1, 3]}, self.X + 1, self.y,
                                 seed=19, checkpoint_dir=Path(self.temp.name) / "shared")
        path = next((Path(self.temp.name) / "shared" / "folds").glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["payload"]["neighbors"]["1"]["balanced_accuracy"] = 0.123
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.shared(estimator, {"n_neighbors": [1, 3]})

    def test_interrupted_fold_keeps_completed_folds_and_failed_attempt(self):
        estimator = Pipeline([("screen", BinaryScreening(percentile=50)),
                              ("knn", KNeighborsClassifier())])
        original_fit = BinaryScreening.fit
        calls = []

        def interrupt_third(instance, X, y):
            calls.append(X.shape[0])
            if len(calls) == 3:
                raise KeyboardInterrupt("test interruption")
            return original_fit(instance, X, y)

        with patch.object(BinaryScreening, "fit", interrupt_third), self.assertRaises(KeyboardInterrupt):
            self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        attempts = [json.loads(path.read_text())["payload"]
                    for path in (Path(self.temp.name) / "shared" / "attempts").glob("*.json")]
        self.assertEqual(sum(row["state"] == "interrupted" for row in attempts), 1)
        result = self.shared(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(result["new_representation_fits"], 3)
        self.assertEqual(result["reused_representation_folds"], 2)
        self.assertEqual(result["unique_representation_fits"], 5)

    def test_unsupported_classifier_knobs_and_unseeded_representation_are_rejected(self):
        for estimator, grid, pattern in (
            (DummyClassifier(), {}, "Use NMFClassifier"),
            (KNeighborsClassifier(), {"weights": ["uniform", "distance"]}, "Only n_neighbors"),
            (Pipeline([("knn", KNeighborsClassifier())]), {"knn__metric": ["euclidean"]}, "Only n_neighbors"),
            (NMFClassifier(solver="cd", screening="none"), {"n_neighbors": [1, 3]}, "fixed integer seed"),
        ):
            with self.subTest(estimator=type(estimator).__name__), self.assertRaisesRegex((ValueError, TypeError), pattern):
                self.shared(estimator, grid)


if __name__ == "__main__":
    unittest.main()
