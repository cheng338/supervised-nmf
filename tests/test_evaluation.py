"""Check leakage boundaries, numerical rejection, and audit preservation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

from supervised_nmf.evaluation import (
    calibration_search, comparison_models, evaluate_classifier, provenance, run_comparison, save_data_manifest, support_recovery,
)


FIT_HISTORY = []


class AuditedClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic estimator exposing exact training rows to these tests."""

    def __init__(self, reject_missing_zero=False, convergence_warning=False, fail_full_size=None):
        self.reject_missing_zero = reject_missing_zero
        self.convergence_warning = convergence_warning
        self.fail_full_size = fail_full_size

    def fit(self, X, y):
        FIT_HISTORY.append(np.asarray(X[:, 0], dtype=int).tolist())
        if self.reject_missing_zero and 0 not in X[:, 0]:
            raise ValueError("Deliberately invalid in one fold")
        if self.fail_full_size == len(X):
            raise ValueError("Deliberately invalid at final refit")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if self.convergence_warning:
            warnings.warn("Deliberately unconverged", ConvergenceWarning)
        return self

    def predict(self, X):
        return self.classes_[(X[:, 0].astype(int) % 2)]

    def predict_proba(self, X):
        index = X[:, 0].astype(int) % 2
        return np.column_stack([1 - index, index]).astype(float)


class CountingTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.transform_calls_ = 0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        self.transform_calls_ += 1
        return X


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        FIT_HISTORY.clear()
        self.X = np.column_stack([np.arange(60), np.ones(60)])
        self.y = np.arange(60) % 2

    def test_folds_have_no_validation_rows_in_training(self):
        result = calibration_search(AuditedClassifier(), {}, self.X, self.y)
        self.assertEqual(len(FIT_HISTORY), 6)
        for observed, split in zip(FIT_HISTORY[:5], result.report["folds"]):
            self.assertEqual(observed, split["training_indices"])
            self.assertFalse(set(observed) & set(split["validation_indices"]))
        self.assertEqual(FIT_HISTORY[-1], list(range(60)))
        self.assertEqual(result.report["candidates"][0]["mean_balanced_accuracy"], 1.0)

    def test_single_failed_fold_disqualifies_candidate(self):
        result = calibration_search(AuditedClassifier(), {"reject_missing_zero": [True, False]}, self.X, self.y)
        failed, valid = result.report["candidates"]
        self.assertEqual(failed["invalid_fold_count"], 1)
        self.assertIsNone(failed["mean_balanced_accuracy"])
        self.assertFalse(failed["valid"])
        self.assertTrue(valid["valid"])
        self.assertEqual(result.report["selected_candidate_index"], 1)
        self.assertFalse(result.report["is_fallback"])

    def test_mu_comparison_grids_tune_and_reject_truncated_fits(self):
        random = np.random.RandomState(23)
        y = np.arange(80) % 2
        coefficients = random.uniform(.1, 1, (80, 2))
        coefficients[:, 0] += y
        basis = np.column_stack([np.eye(2), random.uniform(.1, 1, (2, 10))])
        X = (coefficients @ basis + random.uniform(0, .005, (80, 12))).astype(np.float32)
        models = comparison_models(random_state=7, quick=True, max_iter=3000, tol=1e-3)
        for screening in ("absolute", "percentile"):
            with self.subTest(screening=screening):
                estimator, grid = models[f"screened_mu_{screening}"]
                result = calibration_search(estimator, {**grid, "max_iter": [1, 3000]},
                                            X, y, cv=3, threads=1)
                failed, valid = result.report["candidates"]
                self.assertFalse(failed["valid"])
                self.assertEqual(failed["invalid_fold_count"], 3)
                self.assertIsNone(failed["mean_balanced_accuracy"])
                self.assertTrue(valid["valid"], valid)
                self.assertFalse(result.report["is_fallback"])
                self.assertEqual(result.estimator.solver, "mu")
                self.assertEqual(result.estimator.max_iter, 3000)
                self.assertTrue(result.estimator.converged_)
                for fold in valid["folds"]:
                    self.assertTrue(fold["diagnostics"]["converged"])
                    self.assertTrue(fold["diagnostics"]["last_transform_converged"])
                    self.assertGreaterEqual(fold["diagnostics"]["fit_times"]["factorization"], 0)
                evaluated = evaluate_classifier(result.estimator, X[:10] * 1.05, y[:10], threads=1)
                self.assertEqual(evaluated["status"], "valid")
                self.assertGreaterEqual(evaluated["prediction_seconds"], 0)
                self.assertGreaterEqual(result.report["tuning_seconds"], 0)
                self.assertGreaterEqual(result.report["final_fit_seconds"], 0)

    def test_nonconverged_candidates_produce_flagged_training_only_fallback(self):
        result = calibration_search(AuditedClassifier(convergence_warning=True), {}, self.X, self.y)
        self.assertTrue(result.report["is_fallback"])
        self.assertIsInstance(result.estimator, DummyClassifier)
        self.assertIsNone(result.report["selected_parameters"])
        candidate = result.report["candidates"][0]
        self.assertEqual(candidate["invalid_fold_count"], 5)
        self.assertEqual(candidate["folds"][0]["warnings"][0]["category"], "ConvergenceWarning")
        self.assertFalse(result.report["defaults_frozen"])

    def test_final_fit_failure_is_retained_and_cannot_silently_win(self):
        result = calibration_search(AuditedClassifier(fail_full_size=60), {}, self.X, self.y)
        self.assertTrue(result.report["candidates"][0]["valid"])
        self.assertTrue(result.report["is_fallback"])
        self.assertEqual(result.report["final_fit_attempts"][0]["status"], "failed")
        self.assertEqual(result.report["final_fit_attempts"][1]["status"], "fallback")

    def test_resume_preserves_folds_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            first = calibration_search(AuditedClassifier(), {}, self.X, self.y, checkpoint_path=path)
            second = calibration_search(AuditedClassifier(), {}, self.X, self.y, checkpoint_path=path)
            self.assertEqual(len(FIT_HISTORY), 7)  # second invocation only repeats final refit
            self.assertEqual(first.report["candidates"], second.report["candidates"])
            self.assertEqual(len(second.report["final_fit_attempts"]), 2)
            self.assertEqual(json.loads(path.read_text())["state"], "completed")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                calibration_search(AuditedClassifier(), {}, self.X + 1, self.y, checkpoint_path=path)

    def test_completed_comparison_resumes_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            models = {"audited": (AuditedClassifier(), {})}
            first = run_comparison(models, self.X, self.y, self.X, self.y, output_dir=directory)
            history_length = len(FIT_HISTORY)
            second = run_comparison(models, self.X, self.y, self.X, self.y, output_dir=directory)
            self.assertEqual(len(FIT_HISTORY), history_length)
            self.assertEqual(first, second)

    def test_imbalanced_metrics_reveal_majority_only_classifier(self):
        y = np.array([0] * 54 + [1] * 6)
        model = DummyClassifier(strategy="most_frequent").fit(self.X, y)
        result = evaluate_classifier(model, self.X, y)
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.9)
        self.assertAlmostEqual(result["metrics"]["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["metrics"]["roc_auc"], 0.5)
        self.assertAlmostEqual(result["metrics"]["average_precision"], 0.1)
        self.assertEqual(result["metrics"]["recall"], 0.0)
        self.assertEqual(result["metrics"]["confusion_matrix"], [[54, 0], [6, 0]])

    def test_arbitrary_binary_labels_and_prediction_ids(self):
        y = np.where(self.y == 0, "negative", "positive")
        model = AuditedClassifier().fit(self.X, y)
        ids = [f"row-{i}" for i in range(len(y))]
        result = evaluate_classifier(model, self.X, y, sample_ids=ids)
        self.assertEqual(result["metrics"]["positive_class"], "positive")
        self.assertEqual(result["metrics"]["roc_auc"], 1.0)
        self.assertEqual(result["predictions"]["sample_ids"], ids)

    def test_one_class_evaluation_does_not_report_misleading_ranking_metrics(self):
        model = AuditedClassifier().fit(self.X, self.y)
        result = evaluate_classifier(model, self.X[::2], self.y[::2])
        self.assertIsNone(result["metrics"]["balanced_accuracy"])
        self.assertIsNone(result["metrics"]["roc_auc"])
        self.assertIn("undefined_reason", result["metrics"])

    def test_insufficient_minority_samples_raise_before_search(self):
        with self.assertRaisesRegex(ValueError, "Every class"):
            calibration_search(AuditedClassifier(), {}, self.X, np.array([0] * 57 + [1] * 3))
        self.assertEqual(FIT_HISTORY, [])

    def test_support_recovery_distinguishes_recall_from_exact_recovery(self):
        from sklearn.pipeline import Pipeline
        from supervised_nmf.screening import BinaryScreening
        rng = np.random.RandomState(0)
        X = np.column_stack([10 * self.y + rng.uniform(0, 0.1, 60),
                             rng.uniform(0, 1, 60), rng.uniform(0, 1, 60), rng.uniform(0, 1, 60)])
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=25)),
                              ("classifier", DummyClassifier(strategy="most_frequent"))]).fit(X, self.y)
        recovery = support_recovery(estimator, np.array([True, True, False, False]))
        self.assertEqual(recovery["support_recall"], 0.5)
        self.assertEqual(recovery["support_precision"], 1.0)
        self.assertFalse(recovery["full_support_retained"])
        self.assertFalse(recovery["exact_support_recovered"])

    def test_all_features_retained_is_explicitly_unscreened(self):
        model = AuditedClassifier().fit(self.X, self.y)
        result = support_recovery(model, np.array([True, False]))
        self.assertFalse(result["screening_applied"])
        self.assertEqual(result["support_recall"], 1.0)
        self.assertEqual(result["support_precision"], 0.5)
        self.assertTrue(result["full_support_retained"])
        self.assertFalse(result["exact_support_recovered"])

    def test_incompatible_manifest_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            save_data_manifest(path, {"ids": ["a", "b"]})
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "incompatible"):
                save_data_manifest(path, {"ids": ["a", "c"]})
            self.assertEqual(path.read_bytes(), original)

    def test_runtime_change_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv.json"
            calibration_search(AuditedClassifier(), {}, self.X, self.y, checkpoint_path=path)
            environment = provenance()
            environment["python"] = "different-python-version"
            with patch("supervised_nmf.evaluation.provenance", return_value=environment):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    calibration_search(AuditedClassifier(), {}, self.X, self.y, checkpoint_path=path)

    def test_knn_pipeline_uses_one_projection_for_labels_and_scores(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        model = Pipeline([("projection", CountingTransformer()),
                          ("knn", KNeighborsClassifier(n_neighbors=3))]).fit(self.X, self.y)
        expected_labels = model.predict(self.X)
        expected_scores = model.predict_proba(self.X)[:, 1]
        model.named_steps["projection"].transform_calls_ = 0
        result = evaluate_classifier(model, self.X, self.y)
        self.assertEqual(model.named_steps["projection"].transform_calls_, 1)
        np.testing.assert_array_equal(result["predictions"]["predicted_labels"], expected_labels)
        np.testing.assert_allclose(result["predictions"]["positive_class_probability"], expected_scores)
        self.assertEqual(result["prediction_operation"], "joint_label_and_probability_prediction_one_projection")

    def test_cv_only_calibration_does_not_refit_or_substitute_an_estimator(self):
        result = calibration_search(AuditedClassifier(), {}, self.X, self.y, refit=False)
        self.assertIsNone(result.estimator)
        self.assertFalse(result.report["refit_performed"])
        self.assertFalse(result.report["is_fallback"])
        self.assertEqual(len(FIT_HISTORY), 5)
        self.assertTrue(all(len(rows) == 48 for rows in FIT_HISTORY))

    def test_checkpoint_integrity_rejects_modified_fold_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv.json"
            calibration_search(AuditedClassifier(), {}, self.X, self.y,
                               checkpoint_path=path, refit=False)
            saved = json.loads(path.read_text())
            saved["candidates"][0]["folds"][0]["balanced_accuracy"] = 0.125
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "integrity"):
                calibration_search(AuditedClassifier(), {}, self.X, self.y,
                                   checkpoint_path=path, refit=False)
            self.assertEqual(len(FIT_HISTORY), 5)


if __name__ == "__main__":
    unittest.main()
