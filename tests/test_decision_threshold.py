"""Persisted decision policies and SPA projection-control propagation."""

import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier
from supervised_nmf.classifier import _capture_projection
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.study import calibration_knn_grid


class DecisionThresholdTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        rng = np.random.default_rng(44)
        self.X = rng.uniform(.05, 1.5, (60, 8))
        self.y = np.where(np.arange(60) % 2, "present", "absent")
        self.X[self.y == "present", :2] += .7
        self.query = rng.uniform(.1, 1.8, (12, 8))

    def model(self, **kwargs):
        parameters = dict(n_components=2, screening="none", random_state=17,
                          projection_solver="scipy")
        parameters.update(kwargs)
        return NMFClassifier(**parameters)

    def test_fixed_policy_matches_external_rule_for_both_classifiers_and_labels(self):
        for classifier in ("knn", "logistic"):
            for positive in (None, "present", "absent"):
                with self.subTest(classifier=classifier, positive=positive):
                    model = self.model(classifier=classifier, decision_threshold=.37,
                                       positive_label=positive).fit(self.X, self.y)
                    probabilities = model.predict_proba(self.query)
                    index = 0 if positive == "absent" else 1
                    expected = model.classes_[np.where(probabilities[:, index] > .37,
                                                       index, 1 - index)]
                    np.testing.assert_array_equal(model.predict(self.query), expected)
                    with patch.object(model.decomposition_, "transform",
                                      wraps=model.decomposition_.transform) as projection:
                        predicted, combined = model.predict_with_probabilities(self.query)
                    self.assertEqual(projection.call_count, 1)
                    np.testing.assert_array_equal(predicted, expected)
                    np.testing.assert_array_equal(combined, probabilities)
                    self.assertEqual(model.positive_label_, model.classes_[index])
                    self.assertEqual(model.decision_threshold_, .37)

    def test_strict_equality_and_none_keep_distinct_logistic_decision_rules(self):
        for threshold, expected in ((None, "present"), (.5, "absent")):
            model = self.model(classifier="logistic", decision_threshold=threshold).fit(self.X, self.y)
            logistic = model.classifier_.named_steps["logistic"]
            logistic.coef_[:] = 0
            logistic.intercept_[:] = 1e-20
            labels, probabilities = model.predict_with_probabilities(self.query)
            np.testing.assert_array_equal(probabilities, np.full((12, 2), .5))
            np.testing.assert_array_equal(labels, np.full(12, expected))
            np.testing.assert_array_equal(model.predict(self.query), labels)

    def test_knn_half_probability_and_reversed_positive_class(self):
        model = self.model(n_neighbors=60, decision_threshold=.5,
                           positive_label="absent").fit(self.X, self.y)
        labels, probabilities = model.predict_with_probabilities(self.query)
        np.testing.assert_array_equal(probabilities, np.full((12, 2), .5))
        np.testing.assert_array_equal(labels, np.full(12, "present"))
        ordinary = self.model(n_neighbors=60).fit(self.X, self.y)
        np.testing.assert_array_equal(ordinary.predict(self.query), np.full(12, "absent"))
        np.testing.assert_array_equal(ordinary.predict_proba(self.query), probabilities)

    def test_endpoint_policies_arbitrary_numeric_labels(self):
        y = np.where(self.y == "present", -7, 13)
        for threshold, expected in ((np.nextafter(0.0, -np.inf), -7), (1, 13)):
            model = self.model(decision_threshold=threshold, positive_label=-7).fit(self.X, y)
            np.testing.assert_array_equal(model.predict(self.query), np.full(12, expected))
            self.assertEqual(model.positive_class_index_, 0)

    def test_threshold_does_not_change_probabilities_or_learned_representation(self):
        ordinary = self.model(classifier="logistic").fit(self.X, self.y)
        thresholded = self.model(classifier="logistic", decision_threshold=.12).fit(self.X, self.y)
        np.testing.assert_array_equal(ordinary.components_, thresholded.components_)
        np.testing.assert_array_equal(ordinary.training_coordinates_, thresholded.training_coordinates_)
        np.testing.assert_array_equal(ordinary.predict_proba(self.query), thresholded.predict_proba(self.query))

    def test_clone_serialization_and_fitted_policy_frozen_until_refit(self):
        model = self.model(decision_threshold=0.37, positive_label="absent").fit(self.X, self.y)
        expected = model.predict(self.query)
        cloned = clone(model)
        self.assertEqual(cloned.get_params(), model.get_params())
        with self.assertRaises(NotFittedError):
            cloned.predict(self.query)
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        for restored in (pickle.loads(pickle.dumps(model)), joblib.load(stream)):
            np.testing.assert_array_equal(restored.predict(self.query), expected)
            self.assertEqual(restored.decision_threshold_, .37)
            self.assertEqual(restored.positive_label_, "absent")
        model.set_params(decision_threshold=1., positive_label="present")
        np.testing.assert_array_equal(model.predict(self.query), expected)
        self.assertEqual(model.decision_threshold_, .37)
        model.fit(self.X, self.y)
        self.assertEqual(model.decision_threshold_, 1.)
        self.assertEqual(model.positive_label_, "present")
        np.testing.assert_array_equal(model.predict(self.query), np.full(12, "absent"))
        model.set_params(decision_threshold=None, positive_label=None).fit(self.X, self.y)
        self.assertIsNone(model.decision_threshold_)

    def test_invalid_policy_fails_before_factorization(self):
        invalid = [dict(decision_threshold=value) for value in
                   (-1., -1e-10, np.nan, np.inf, -np.inf, 1.01, True, np.bool_(False), "0.5", [.5])]
        invalid += [dict(positive_label=value) for value in
                    ("missing", ["present"], {"present": 1}, np.nan)]
        for options in invalid:
            with self.subTest(options=options), patch.object(FeatureSPA, "fit_transform") as factorization:
                with self.assertRaises(ValueError):
                    self.model(**options).fit(self.X, self.y)
                factorization.assert_not_called()
        for y in (np.full(60, "present"), np.arange(60) % 3):
            with self.assertRaisesRegex(ValueError, "exactly two classes"):
                self.model(decision_threshold=.3).fit(self.X, y)

    def test_failed_refit_clears_all_fitted_policy_and_model_state(self):
        model = self.model(decision_threshold=.37).fit(self.X, self.y)
        model.set_params(positive_label="missing")
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)
        self.assertFalse(any(name.endswith("_") for name in vars(model)))
        with self.assertRaises(NotFittedError):
            model.predict(self.query)
        model.set_params(positive_label=None).fit(self.X, self.y)
        # Also guard exceptions after factorization, rather than only validation errors.
        with patch("supervised_nmf.classifier.KNeighborsClassifier.fit", side_effect=RuntimeError("failed head")):
            with self.assertRaisesRegex(RuntimeError, "failed head"):
                model.fit(self.X, self.y)
        self.assertFalse(any(name.endswith("_") for name in vars(model)))

    def test_legacy_state_missing_new_attributes_keeps_ordinary_prediction(self):
        model = self.model(classifier="logistic").fit(self.X, self.y)
        expected = model.predict(self.query)
        for name in ("decision_threshold", "positive_label", "decision_threshold_", "positive_label_",
                     "positive_class_index_", "projection_solver", "projection_workers",
                     "projection_batch_size", "projection_native_threads"):
            delattr(model, name)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(self.query), expected)
        labels, _ = restored.predict_with_probabilities(self.query)
        np.testing.assert_array_equal(labels, expected)
        self.assertEqual(restored.get_params()["projection_solver"], "scipy")
        self.assertEqual(clone(restored).get_params(), restored.get_params())
        restored.fit(self.X, self.y)
        self.assertIsNone(restored.decision_threshold_)
        self.assertEqual(restored.decomposition_.projection_solver, "scipy")

    def test_spa_projection_controls_forwarded_and_validated_for_all_solvers(self):
        model = self.model(projection_solver="batched", projection_batch_size=16).fit(self.X, self.y)
        for parameter in ("projection_solver", "projection_workers", "projection_batch_size", "projection_native_threads"):
            self.assertEqual(getattr(model, parameter), getattr(model.decomposition_, parameter))
        np.testing.assert_allclose(model.predict_proba(self.query),
                                   self.model().fit(self.X, self.y).predict_proba(self.query), atol=1e-12)
        invalid = ({"projection_solver": "other"}, {"projection_workers": 0},
                   {"projection_batch_size": 1.2}, {"projection_native_threads": True})
        for solver in ("spa", "cd", "mu"):
            for options in invalid:
                with self.subTest(solver=solver, options=options), self.assertRaises(ValueError):
                    self.model(solver=solver, **options).fit(self.X, self.y)

    def test_execution_controls_change_without_refit_but_do_not_change_basis(self):
        model = self.model().fit(self.X, self.y)
        basis = model.components_.copy()
        fitted_solver = model.decomposition_.projection_solver
        fitted_batch_size = model.decomposition_.projection_batch_size
        expected = model.predict_proba(self.query)
        model.set_params(projection_solver="batched", projection_batch_size=7)
        np.testing.assert_allclose(model.predict_proba(self.query), expected, atol=1e-12)
        self.assertEqual(model.decomposition_.projection_solver, fitted_solver)
        self.assertEqual(model.decomposition_.projection_batch_size, fitted_batch_size)
        np.testing.assert_array_equal(model.components_, basis)
        model.set_params(projection_workers=0)
        with self.assertRaisesRegex(ValueError, "projection_workers"), _capture_projection(model) as diagnostic:
            model.predict(self.query)
        self.assertFalse(diagnostic["last_transform_converged"])
        self.assertIn("projection_workers", diagnostic["last_transform_error"])
        model.set_params(projection_workers=1)
        np.testing.assert_allclose(model.predict_proba(self.query), expected, atol=1e-12)

    def test_shared_knn_calibration_honors_fixed_threshold_and_positive_label(self):
        for positive_label in ("present", "absent"):
            model = self.model(decision_threshold=.3, positive_label=positive_label)
            grid = {"n_neighbors": [3, 5]}
            ordinary = calibration_search(model, grid, self.X, self.y, cv=3, seed=19,
                                          refit=False).report
            with tempfile.TemporaryDirectory() as directory:
                shared = calibration_knn_grid(model, grid, self.X, self.y, cv=3, seed=19,
                                              checkpoint_dir=Path(directory) / "checkpoint")
            for reference, actual in zip(ordinary["candidates"], shared["candidates"]):
                self.assertTrue(reference["valid"] and actual["valid"])
                self.assertEqual(reference["mean_balanced_accuracy"], actual["mean_balanced_accuracy"])
                self.assertEqual([fold["balanced_accuracy"] for fold in reference["folds"]],
                                 [fold["balanced_accuracy"] for fold in actual["folds"]])
        for parameter, values in (("decision_threshold", [.3, .7]), ("positive_label", ["present", "absent"])):
            with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "Only n_neighbors"):
                calibration_knn_grid(self.model(), {parameter: values}, self.X, self.y,
                                     cv=3, checkpoint_dir=Path(directory))


if __name__ == "__main__":
    unittest.main()
