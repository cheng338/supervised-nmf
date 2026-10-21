"""Real-margin evaluation and comparison contracts, authored by Chang Cheng."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import (
    calibration_search, comparison_models, evaluate_classifier, format_evaluation_summary,
)


class MarginOnly(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        return X[:, 0]

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class ProbabilityAndCombinedMargin(MarginOnly):
    """An external head with both score APIs but no combined probability API."""

    def predict_proba(self, X):
        positive = 1.0 / (1.0 + np.exp(X[:, 0]))
        return np.column_stack((1.0 - positive, positive))

    def predict_with_decision_function(self, X):
        return self.predict(X), self.decision_function(X)


class CountTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.calls_ = 0
        return self

    def transform(self, X):
        self.calls_ += 1
        return X


class MarginEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-4., 1.], [-2., 1.], [-.2, 1.], [0., 1.], [1., 1.], [5., 1.]])
        self.y = np.array(["negative", "negative", "positive", "negative", "positive", "positive"])

    def test_unbounded_margins_are_used_for_ranking_and_never_called_probabilities(self):
        model = MarginOnly().fit(self.X, self.y)
        report = evaluate_classifier(model, self.X, self.y)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["ranking_score_type"], "decision_margin")
        self.assertIsNone(report["predictions"]["positive_class_probability"])
        np.testing.assert_array_equal(report["predictions"]["positive_class_decision_margin"], self.X[:, 0])
        self.assertEqual(report["metrics"]["roc_auc"], roc_auc_score(self.y == "positive", self.X[:, 0]))
        self.assertEqual(report["metrics"]["average_precision"],
                         average_precision_score(self.y == "positive", self.X[:, 0]))
        self.assertIn("probabilities are unavailable", format_evaluation_summary(report))
        self.assertEqual(report["predictions"]["predicted_labels"][3], "negative")

    def test_available_probabilities_take_priority_over_external_combined_margins(self):
        model = ProbabilityAndCombinedMargin().fit(self.X, self.y)
        expected_probability = model.predict_proba(self.X)[:, 1]
        expected_labels = model.predict(self.X)
        with patch.object(model, "predict_with_decision_function",
                          side_effect=AssertionError("A probability head must keep probability ranking.")), \
                patch.object(model, "predict_proba", wraps=model.predict_proba) as probabilities:
            report = evaluate_classifier(model, self.X, self.y)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["ranking_score_type"], "probability")
        self.assertEqual(report["prediction_operation"], "predict_plus_predict_proba_calls")
        probabilities.assert_called_once()
        self.assertIsNone(report["predictions"]["positive_class_decision_margin"])
        np.testing.assert_array_equal(report["predictions"]["positive_class_probability"],
                                      expected_probability)
        np.testing.assert_array_equal(report["predictions"]["predicted_labels"], expected_labels)
        self.assertEqual(report["metrics"]["roc_auc"],
                         roc_auc_score(self.y == "positive", expected_probability))
        self.assertEqual(report["metrics"]["average_precision"],
                         average_precision_score(self.y == "positive", expected_probability))

    def test_diagnostics_record_fitted_head_and_metric_after_parameter_changes(self):
        random = np.random.default_rng(420)
        coordinates = random.uniform(.2, 1., size=(48, 2))
        y = np.where(np.arange(48) % 2, "present", "absent")
        coordinates[y == "present", 0] += 1.
        X = coordinates @ np.array([[1., 0., .2, .6], [0., 1., .7, .1]])
        for classifier, expected_metric, head_type in (
                ("knn", "sd", "sklearn.neighbors._classification.KNeighborsClassifier"),
                ("linear_svm", None, "sklearn.svm._classes.LinearSVC")):
            with self.subTest(classifier=classifier):
                model = NMFClassifier(
                    n_components=2, screening="none", classifier=classifier,
                    knn_normalization="sd", random_state=11).fit(X, y)
                model.set_params(classifier="logistic", knn_normalization="none",
                                 training_coordinates="fit_transform")
                report = evaluate_classifier(model, X[:12], y[:12])
                self.assertEqual(report["status"], "valid")
                diagnostics = report["diagnostics"]
                self.assertEqual(diagnostics["knn_normalization"], expected_metric)
                self.assertEqual(diagnostics["training_coordinates_mode"], "transform")
                self.assertEqual(diagnostics["fitted_classifier_type"], head_type)
                self.assertEqual(report["ranking_score_type"],
                                 "probability" if classifier == "knn" else "decision_margin")

    def test_nonfinite_and_multicolumn_binary_margins_fail_explicitly(self):
        model = MarginOnly().fit(self.X, self.y)
        labels = model.predict(self.X)
        for margins in (np.full(len(self.X), np.nan), np.full(len(self.X), np.inf),
                        np.zeros((len(self.X), 2))):
            with self.subTest(shape=margins.shape), patch.object(model, "predict", return_value=labels), \
                    patch.object(model, "decision_function", return_value=margins):
                report = evaluate_classifier(model, self.X, self.y)
                self.assertEqual(report["status"], "failed")
                self.assertIn("Decision margins", report["error"])

    def test_pipeline_projects_once_and_keeps_linear_svc_label_polarity(self):
        model = Pipeline([("coordinates", CountTransform()),
                          ("linear_svm", LinearSVC(dual=False, max_iter=20000))]).fit(self.X, self.y)
        expected = model.predict(self.X)
        model.named_steps["coordinates"].calls_ = 0
        report = evaluate_classifier(model, self.X, self.y)
        self.assertEqual(model.named_steps["coordinates"].calls_, 1)
        np.testing.assert_array_equal(report["predictions"]["predicted_labels"], expected)
        self.assertEqual(report["prediction_operation"], "joint_label_and_margin_prediction_one_projection")

    def test_svm_comparators_have_independent_c_grids_and_no_hidden_scaler(self):
        models = comparison_models(classifier="linear_svm", random_state=7)
        self.assertEqual(len(models), 13)
        self.assertTrue(set(models).isdisjoint(comparison_models()))
        self.assertTrue(set(models).isdisjoint(comparison_models(classifier="logistic")))
        for name, (model, grid) in models.items():
            if "majority" in name:
                continue
            key = "svm_C" if isinstance(model, NMFClassifier) else "linear_svm__C"
            self.assertEqual(grid[key], [.01, .1, 1., 10., 100.])
            if isinstance(model, Pipeline):
                self.assertNotIn("scale", model.named_steps)
                self.assertFalse(model.named_steps["linear_svm"].dual)
        raw, _ = models["raw_linear_svm"]
        fitted = clone(raw).fit(self.X, self.y)
        self.assertEqual(evaluate_classifier(fitted, self.X, self.y)["ranking_score_type"], "decision_margin")

    def test_svm_calibration_preserves_classifier_warnings(self):
        rng = np.random.default_rng(390)
        X, y = rng.normal(size=(100, 20)), np.arange(100) % 2
        estimator, _ = comparison_models(classifier="linear_svm", svm_max_iter=1)["raw_linear_svm"]
        result = calibration_search(estimator, {"linear_svm__C": [1.]}, X, y, cv=3,
                                    seed=19, refit=False)
        self.assertFalse(result.report["candidates"][0]["valid"])
        self.assertTrue(all(row["status"] == "nonconverged"
                            for row in result.report["candidates"][0]["folds"]))


if __name__ == "__main__":
    unittest.main()
