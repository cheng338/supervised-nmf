"""Accuracy objectives, compatibility, and reporting at the public API."""

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import BinaryScreening
from supervised_nmf.evaluation import calibration_search, run_comparison


class MetricChoiceClassifier(ClassifierMixin, BaseEstimator):
    """Two fixed decision rules whose accuracy and mean recall disagree."""

    fits = 0
    predictions = 0

    def __init__(self, rule="majority", fail_one_fold=False, nonconverged=False,
                 fail_majority_refit=False):
        self.rule = rule
        self.fail_one_fold = fail_one_fold
        self.nonconverged = nonconverged
        self.fail_majority_refit = fail_majority_refit

    def fit(self, X, y):
        type(self).fits += 1
        if self.fail_one_fold and 0 not in X[:, 0]:
            raise ValueError("Deliberate failure in the fold excluding row zero")
        if self.fail_majority_refit and len(X) == 50 and self.rule == "majority":
            raise ValueError("Deliberate final majority refit failure")
        if self.nonconverged:
            warnings.warn("Deliberately unconverged", ConvergenceWarning)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        type(self).predictions += 1
        index = np.zeros(len(X), dtype=int) if self.rule == "majority" else X[:, 1].astype(int)
        return self.classes_[index]


class CalibrationScoringTests(unittest.TestCase):
    def setUp(self):
        # Every validation fold has eight negatives and two positives.
        # Majority: accuracy .8, mean recall .5.
        # Sensitive: accuracy .6, mean recall .75.
        self.X = np.column_stack([np.arange(50), np.tile([0, 0, 1, 1, 1], 10)])
        self.y = np.tile([0, 0, 0, 0, 1], 10)
        rows = np.arange(50)
        self.folds = [(rows[(rows < start) | (rows >= start + 10)], rows[start:start + 10])
                      for start in range(0, 50, 10)]
        self.grid = {"rule": ["majority", "sensitive"]}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        MetricChoiceClassifier.fits = MetricChoiceClassifier.predictions = 0

    def search(self, **kwargs):
        estimator = kwargs.pop("estimator", MetricChoiceClassifier())
        return calibration_search(estimator, self.grid, self.X, self.y, folds=self.folds,
                                  refit=kwargs.pop("refit", False), **kwargs)

    def test_default_accuracy_and_explicit_balanced_accuracy_select_different_rules(self):
        accuracy = self.search().report
        balanced = self.search(scoring="balanced_accuracy").report
        self.assertEqual(accuracy["selected_parameters"], {"rule": "majority"})
        self.assertEqual(balanced["selected_parameters"], {"rule": "sensitive"})
        self.assertAlmostEqual(accuracy["selected_mean_score"], .8)
        self.assertAlmostEqual(balanced["selected_mean_score"], .75)
        for report, metric, expected in ((accuracy, "accuracy", [.8, .6]),
                                         (balanced, "balanced_accuracy", [.5, .75])):
            self.assertEqual(report["scoring"], metric)
            self.assertEqual(report["identity"]["scoring"], metric)
            other = "balanced_accuracy" if metric == "accuracy" else "accuracy"
            for row, score in zip(report["candidates"], expected):
                self.assertAlmostEqual(row["mean_score"], score)
                self.assertEqual(row["mean_" + metric], row["mean_score"])
                self.assertNotIn("mean_" + other, row)
                for fold in row["folds"]:
                    self.assertEqual(fold["score"], fold[metric])
                    self.assertNotIn(other, fold)
        self.assertEqual(MetricChoiceClassifier.fits, 20)
        self.assertEqual(MetricChoiceClassifier.predictions, 20)

    def test_default_does_not_switch_on_severe_imbalance(self):
        X = np.column_stack([np.arange(1000), np.zeros(1000)])
        y = np.zeros(1000, dtype=int)
        y[::200] = 1
        report = calibration_search(MetricChoiceClassifier(), {}, X, y, refit=False).report
        self.assertEqual(report["scoring"], "accuracy")
        self.assertAlmostEqual(report["selected_mean_score"], .995)

    def test_invalid_scorer_rejected_before_fitting_or_creating_checkpoint(self):
        for scorer in (None, "roc_auc", "Accuracy", {}, lambda y, p: 1):
            with self.subTest(scoring=scorer):
                path = Path(self.temp.name) / "invalid.json"
                with self.assertRaisesRegex(ValueError, "scoring"):
                    self.search(scoring=scorer, checkpoint_path=path)
                self.assertFalse(path.exists())
        self.assertEqual(MetricChoiceClassifier.fits, 0)

    def test_checkpoint_resume_binds_scoring_and_preserves_original(self):
        path = Path(self.temp.name) / "cv.json"
        first = self.search(checkpoint_path=path).report
        second = self.search(checkpoint_path=path, scoring="accuracy").report
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(MetricChoiceClassifier.fits, 10)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.search(checkpoint_path=path, scoring="balanced_accuracy")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(MetricChoiceClassifier.fits, 10)

    def test_every_fold_must_qualify_for_either_objective(self):
        for metric in ("accuracy", "balanced_accuracy"):
            for failure in ("fail_one_fold", "nonconverged"):
                with self.subTest(metric=metric, failure=failure):
                    report = self.search(scoring=metric,
                                         estimator=MetricChoiceClassifier(**{failure: True})).report
                    self.assertIsNone(report["selected_parameters"])
                    self.assertIsNone(report["selected_mean_score"])
                    for row in report["candidates"]:
                        self.assertFalse(row["valid"])
                        self.assertIsNone(row["mean_score"])
                        self.assertIsNone(row["mean_" + metric])

    def test_refit_failure_uses_next_candidate_under_selected_objective(self):
        result = self.search(refit=True, estimator=MetricChoiceClassifier(fail_majority_refit=True))
        self.assertEqual(result.report["selected_parameters"], {"rule": "sensitive"})
        self.assertAlmostEqual(result.report["selected_mean_score"], .6)
        self.assertEqual([row["status"] for row in result.report["final_fit_attempts"]],
                         ["failed", "valid"])
        self.assertEqual(result.estimator.rule, "sensitive")
        self.assertFalse(result.report["is_fallback"])

    def test_comparison_records_objective_and_separate_evaluation_metrics(self):
        models = {"rules": (MetricChoiceClassifier(), self.grid)}
        report = run_comparison(models, self.X, self.y, self.X, self.y,
                                output_dir=self.temp.name)
        self.assertEqual(report["scoring"], "accuracy")
        row = report["models"]["rules"]
        self.assertEqual(row["scoring"], "accuracy")
        self.assertAlmostEqual(row["selected_mean_score"], .8)
        self.assertEqual(row["selected_parameters"], {"rule": "majority"})
        self.assertAlmostEqual(row["evaluation"]["metrics"]["accuracy"], .8)
        self.assertAlmostEqual(row["evaluation"]["metrics"]["balanced_accuracy"], .5)
        count = MetricChoiceClassifier.fits
        repeated = run_comparison(models, self.X, self.y, self.X, self.y,
                                  output_dir=self.temp.name, scoring="accuracy")
        self.assertEqual(report, repeated)
        self.assertEqual(MetricChoiceClassifier.fits, count)
        before = (Path(self.temp.name) / "rules-evaluation.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            run_comparison(models, self.X, self.y, self.X, self.y,
                           output_dir=self.temp.name, scoring="balanced_accuracy")
        self.assertEqual((Path(self.temp.name) / "rules-evaluation.json").read_bytes(), before)
        self.assertEqual(json.loads(before)["scoring"], "accuracy")

    def test_inactive_grid_warning_and_refinement_do_not_change_screening(self):
        model = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=100)),
                          ("knn", KNeighborsClassifier(n_neighbors=1))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = calibration_search(model, {"screen__score_threshold": [0, 10]},
                                        self.X, self.y, folds=self.folds, refit=False).report
        messages = [str(w.message) for w in caught if "Inactive screening grid" in str(w.message)]
        self.assertEqual(len(messages), 1)
        diagnostics = report["screening_diagnostics"]
        self.assertEqual(diagnostics["inactive_parameters"][0]["candidate_indices"], [0, 1])
        self.assertEqual(diagnostics["candidates"][0]["screens"][0]["mode"], "percentile")
        self.assertEqual(report["refinement"]["parameters"]["screen__score_threshold"]["status"],
                         "inactive_for_selected_screening_mode")
        self.assertEqual([f["score"] for f in report["candidates"][0]["folds"]],
                         [f["score"] for f in report["candidates"][1]["folds"]])
        self.assertEqual(model.named_steps["screen"].screening, "percentile")


if __name__ == "__main__":
    unittest.main()
