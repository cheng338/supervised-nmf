"""Protect subject boundaries and descriptive reporting in Chang Cheng's studies."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from supervised_nmf.evaluation import calibration_search, evaluate_classifier, format_evaluation_summary
from supervised_nmf.validation import build_cv_folds, validate_metadata


class GroupValidationTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(30), 2)
        self.y = np.tile(["absent", "present"], 30)
        self.X = np.column_stack([np.arange(60), self.y == "present"]).astype(float)
        self.ids = np.array([f"image-{i}" for i in range(60)])
        self.folds = [(np.flatnonzero(self.groups % 5 != i), np.flatnonzero(self.groups % 5 == i))
                      for i in range(5)]

    def test_ordinary_folds_preserve_previous_sklearn_splits(self):
        expected = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=7).split(self.X, self.y))
        actual = build_cv_folds(self.X, self.y, cv=3, seed=7)
        for (a, b), (c, d) in zip(expected, actual):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_automatic_group_folds_are_exhaustive_and_group_disjoint(self):
        for groups in (self.groups, np.array([f"person-{i}" for i in self.groups])):
            with self.subTest(dtype=groups.dtype):
                folds = build_cv_folds(self.X, self.y, cv=3, groups=groups)
                np.testing.assert_array_equal(np.sort(np.concatenate([b for _, b in folds])), np.arange(60))
                for train, valid in folds:
                    self.assertFalse(set(groups[train]) & set(groups[valid]))
                    self.assertEqual(len(np.unique(self.y[train])), 2)
                    self.assertEqual(len(np.unique(self.y[valid])), 2)

    def test_custom_dict_folds_preserve_direct_model_scores_and_predictions(self):
        dictionaries = [{"training_indices": a.tolist(), "validation_indices": b.tolist()} for a, b in self.folds]
        base = KNeighborsClassifier(n_neighbors=3)
        result = calibration_search(base, {}, self.X, self.y, groups=self.groups,
                                    data_ids=self.ids, folds=dictionaries)
        actual = result.report["candidates"][0]["folds"]
        for index, (train, valid) in enumerate(self.folds):
            direct = clone(base).fit(self.X[train], self.y[train])
            expected = balanced_accuracy_score(self.y[valid], direct.predict(self.X[valid]))
            self.assertEqual(actual[index]["balanced_accuracy"], expected)
        full = clone(base).fit(self.X, self.y)
        np.testing.assert_array_equal(result.estimator.predict(self.X), full.predict(self.X))
        self.assertEqual(result.report["identity"]["groups"], self.groups.tolist())
        self.assertEqual(result.report["identity"]["data_ids"], self.ids.tolist())

    def test_invalid_metadata_and_folds_raise_before_fit(self):
        corruptions = []
        for name, replacement in (("floating_indices", self.folds[0][0].astype(float)),
                                  ("boolean_indices", self.folds[0][0].astype(bool)),
                                  ("duplicate_indices", np.append(self.folds[0][0], self.folds[0][0][0])),
                                  ("incomplete_complement", self.folds[0][0][1:]),
                                  ("negative_index", np.append(self.folds[0][0], -1))):
            folds = [(a.copy(), b.copy()) for a, b in self.folds]
            folds[0] = replacement, folds[0][1]
            corruptions.append((name, {"folds": folds, "groups": self.groups}))
        repeated = [self.folds[0]] * 5
        corruptions.append(("nonpartition", {"folds": repeated, "groups": self.groups}))
        crossing = self.groups.copy()
        crossing[2] = crossing[0]
        corruptions.append(("group_leakage", {"folds": self.folds, "groups": crossing}))
        corruptions.append(("too_few_folds", {"folds": self.folds[:4]}))
        for name, groups in (("missing", [None, *self.groups[1:]]),
                             ("nonfinite", [np.nan, *self.groups[1:]]),
                             ("blank", ["", *map(str, self.groups[1:])]),
                             ("short", self.groups[:-1]), ("matrix", self.groups[:, None]),
                             ("mixed_types", ["first", *self.groups[1:]])):
            corruptions.append((name, {"groups": groups}))
        corruptions.append(("duplicate_data_id", {"data_ids": ["same"] * 60}))
        for name, options in corruptions:
            with self.subTest(case=name), patch.object(DummyClassifier, "fit", side_effect=AssertionError("Unexpected fit")):
                with self.assertRaises(ValueError):
                    calibration_search(DummyClassifier(), {}, self.X, self.y, **options)

    def test_missing_class_and_insufficient_screening_support_rejected(self):
        y = self.groups % 5 == 0
        with self.assertRaisesRegex(ValueError, "both classes|two samples"):
            build_cv_folds(self.X, y, groups=self.groups, folds=self.folds)
        with self.assertRaisesRegex(ValueError, "two samples"):
            build_cv_folds(self.X[:4], self.y[:4], cv=2)
        with self.assertRaisesRegex(ValueError, "distinct groups"):
            build_cv_folds(self.X, self.y, cv=5, groups=np.zeros(60))

    def test_checkpoint_binds_groups_ids_and_exact_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv.json"
            options = dict(groups=self.groups, data_ids=self.ids, folds=self.folds, refit=False, checkpoint_path=path)
            initial = calibration_search(DummyClassifier(), {}, self.X, self.y, **options)
            with patch.object(DummyClassifier, "fit", side_effect=AssertionError("Unexpected refit")):
                resumed = calibration_search(DummyClassifier(), {}, self.X, self.y, **options)
            self.assertEqual(initial.report["candidates"], resumed.report["candidates"])
            original_bytes = path.read_bytes()
            changes = [{"groups": self.groups + 100},
                       {"data_ids": np.array([f"new-{i}" for i in range(60)])},
                       {"folds": self.folds[::-1]}]
            for change in changes:
                with self.subTest(change=next(iter(change))), self.assertRaisesRegex(ValueError, "incompatible"):
                    calibration_search(DummyClassifier(), {}, self.X, self.y, **{**options, **change})
                self.assertEqual(path.read_bytes(), original_bytes)
            receipt = json.loads(path.read_text())
            self.assertEqual(len(receipt["identity"]["folds"]), 5)

    def test_subject_memorization_is_not_mistaken_for_new_subject_accuracy(self):
        # Identical images of a person can be memorized in row-wise splits.
        groups = np.repeat(np.arange(30), 2)
        y = np.repeat(np.arange(30) % 2, 2)
        X = np.eye(30)[groups]
        row_folds = [(np.arange(0, 60, 2), np.arange(1, 60, 2)),
                     (np.arange(1, 60, 2), np.arange(0, 60, 2))]
        with self.assertRaisesRegex(ValueError, "groups must not overlap"):
            build_cv_folds(X, y, cv=2, groups=groups, folds=row_folds)
        ordinary = build_cv_folds(X, y, cv=2, folds=row_folds)
        for train, valid in ordinary:
            model = KNeighborsClassifier(n_neighbors=1).fit(X[train], y[train])
            self.assertEqual(balanced_accuracy_score(y[valid], model.predict(X[valid])), 1.0)


class EvaluationReportingTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(20, dtype=float)[:, None]
        self.train_y = np.array(["absent"] * 18 + ["present"] * 2)
        self.y = np.array(["absent"] * 2 + ["present"] * 18)
        self.model = DummyClassifier(strategy="most_frequent").fit(self.X, self.train_y)

    def test_training_majority_is_not_reselected_on_evaluation_split(self):
        report = evaluate_classifier(self.model, self.X, self.y, train_y=self.train_y)
        baseline = report["training_majority_reference"]
        self.assertEqual(baseline["label"], "absent")
        self.assertEqual(baseline["accuracy"], .1)
        self.assertEqual(baseline["balanced_accuracy"], .5)
        self.assertEqual(report["metrics"]["per_class"][1]["recall"], 0)
        self.assertTrue(report["support_diagnostics"]["low_class_support"])
        self.assertFalse(report["support_diagnostics"]["meets_support_criteria"])
        absent = evaluate_classifier(self.model, self.X, self.y)
        self.assertFalse(absent["training_majority_reference"]["available"])

    def test_support_threshold_boundary_and_subject_counts(self):
        groups = np.repeat(np.arange(4), 5)
        for count, low_support in ((9, True), (10, False)):
            with self.subTest(minority=count):
                y = np.array(["absent"] * count + ["present"] * (20 - count))
                result = evaluate_classifier(self.model, self.X, y, groups=groups, train_groups=groups + 100,
                                             train_y=self.train_y)
                diagnostics = result["support_diagnostics"]
                self.assertEqual(diagnostics["low_class_support"], low_support)
                self.assertEqual(diagnostics["group_count"], 4)
                self.assertTrue(diagnostics["low_group_support"])
                self.assertTrue(diagnostics["group_overlap_checked"])
                self.assertEqual(result["status"], "valid")

    def test_overlap_or_invalid_metadata_rejected_before_prediction(self):
        ids = np.arange(20)
        options = [dict(sample_ids=ids, train_sample_ids=ids),
                   dict(groups=ids, train_groups=ids),
                   dict(train_sample_ids=ids), dict(train_groups=ids),
                   dict(groups=[None] * 20), dict(sample_ids=np.zeros(20)),
                   dict(train_y=["wrong"] * 20), dict(train_y=self.train_y, train_groups=ids[:-1]),
                   dict(min_class_support=0), dict(min_groups=True)]
        for metadata in options:
            with self.subTest(metadata=next(iter(metadata))), patch.object(self.model, "predict", side_effect=AssertionError("Unexpected prediction")):
                with self.assertRaises(ValueError):
                    evaluate_classifier(self.model, self.X, self.y, **metadata)

    def test_valid_metadata_preserves_predictions_and_probabilities(self):
        direct = evaluate_classifier(self.model, self.X, self.y)
        enriched = evaluate_classifier(self.model, self.X, self.y, sample_ids=np.arange(20),
                                       train_sample_ids=np.arange(20) + 100, groups=np.arange(20),
                                       train_groups=np.arange(20) + 100, train_y=self.train_y)
        self.assertEqual(enriched["predictions"], direct["predictions"])
        self.assertEqual(enriched["metrics"], direct["metrics"])
        self.assertTrue(enriched["support_diagnostics"]["sample_overlap_checked"])

    def test_absent_class_keeps_undefined_balanced_accuracy_explicit(self):
        result = evaluate_classifier(self.model, self.X, np.repeat("present", 20), train_y=self.train_y)
        self.assertIsNone(result["metrics"]["balanced_accuracy"])
        self.assertIsNone(result["training_majority_reference"]["balanced_accuracy"])
        self.assertTrue(result["support_diagnostics"]["low_class_support"])

    def test_summary_surfaces_support_stage_timings_and_known_convergence(self):
        self.model.converged_ = False
        self.model.convergence_warnings_ = ("Fit iteration limit reached",)
        self.model.factorization_converged_ = False
        self.model.factorization_convergence_warnings_ = ("Factorization limit reached",)
        self.model.fit_times_ = {"factorization": .125, "classifier": .001}
        evaluated = evaluate_classifier(self.model, self.X, self.y, train_y=self.train_y,
                                         groups=np.repeat(np.arange(4), 5))
        text = format_evaluation_summary({"evaluation": evaluated, "tuning_seconds": 12.5, "final_fit_seconds": .2})
        self.assertIn("provisional_nonconvergence", text)
        self.assertIn("Low class support", text)
        self.assertIn("Low subject support", text)
        self.assertIn("Training-majority reference", text)
        self.assertIn("| Factorization | failed |", text)
        self.assertIn("Fit iteration limit reached", text)
        self.assertIn("| Tuning | 12.5 |", text)
        self.assertIn("| Fit: factorization | 0.125 |", text)
        self.assertIn("subject overlap was not checked", text)


if __name__ == "__main__":
    unittest.main()
