"""Stage reporting follows the classifier's resolved training-coordinate policy."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import NMF, PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import profile_celeba_stages as profiler
from supervised_nmf.classifier import NMFClassifier
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.screening import BinaryScreening


class StageProfilerTests(unittest.TestCase):
    def setUp(self):
        limit = threadpool_limits(limits=1)
        self.addCleanup(limit.restore_original_limits)
        # Ordinary small fixture shared with the item 5 investigation.
        rng = np.random.default_rng(831)
        basis = np.full((2, 12), 0.1)
        basis[0, ::2] += 2
        basis[1, 1::2] += 2
        latent = rng.uniform(0.3, 2, (72, 2))
        self.X = np.column_stack((latent @ basis, np.zeros((72, 2))))
        self.y = (latent[:, 0] > np.median(latent[:, 0])).astype(int)
        query_latent = np.asarray([
            [.4, .7], [.6, 1.5], [1.8, .8], [1.9, 1.7],
            [.3, 1.1], [1.7, .4], [.5, 1.8], [1.9, 1.],
        ])
        self.query = np.column_stack((query_latent @ basis, np.zeros((8, 2))))
        self.query_y = (query_latent[:, 0] > np.median(latent[:, 0])).astype(int)
        self.ids = [f"query-{i}" for i in range(len(self.query))]
        self.inputs = (self.X, self.y, self.query, self.query_y)
        self.original_inputs = tuple(value.copy() for value in self.inputs)

    def tearDown(self):
        for actual, original in zip(self.inputs, self.original_inputs):
            np.testing.assert_array_equal(actual, original)

    @staticmethod
    def model(**changes):
        parameters = dict(
            solver="spa", training_coordinates="auto", screening="none",
            n_components=2, n_neighbors=3, weights="distance", max_iter=3000,
            tol=1e-4, random_state=17,
        )
        parameters.update(changes)
        return NMFClassifier(**parameters)

    def profile(self, model, reference=None):
        # Failure/held cases must never need a prediction reference.
        return profiler.profile_selected_estimator(
            model, self.X, self.y, self.query, self.query_y,
            reference={} if reference is None else reference, sample_ids=self.ids,
        )

    def reference(self, model):
        fitted = clone(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted.fit(self.X, self.y)
        self.assertFalse(profiler._nonconverged(fitted, caught))
        result = profiler.evaluate_classifier(
            fitted, self.query, self.query_y, threads=1, sample_ids=self.ids,
        )
        self.assertEqual(result["status"], "valid", result)
        return result, fitted

    def assert_unreached(self, record, *stages):
        for stage in stages:
            with self.subTest(stage=stage):
                self.assertEqual(record["instrumented_method_calls"][stage], 0)
                self.assertIsNone(record["timing"][stage])

    def assert_absent(self, record, *stages):
        for stage in stages:
            with self.subTest(stage=stage):
                self.assertEqual(record["instrumented_method_calls"][stage], 0)
                self.assertEqual(record["timing"][stage], 0.0)

    def assert_measured(self, record, *stages):
        for stage in stages:
            with self.subTest(stage=stage):
                self.assertEqual(record["instrumented_method_calls"][stage], 1)
                seconds = record["timing"][stage]
                self.assertIsInstance(seconds, float)
                self.assertTrue(np.isfinite(seconds))
                self.assertGreaterEqual(seconds, 0.0)

    def assert_exact_predictions(self, record, reference):
        self.assertEqual(record["status"], "valid", record)
        self.assertTrue(record["exact_prediction_match"])
        for field in ("sample_ids", "true_labels", "predicted_labels",
                      "positive_class_probability"):
            np.testing.assert_array_equal(
                record["evaluation"]["predictions"][field],
                reference["predictions"][field],
            )

    def test_invalid_rank_reports_unreached_projection_for_all_coordinate_policies(self):
        cases = (
            ("spa", "auto", True), ("spa", "fit_transform", False), ("spa", "transform", True),
            ("cd", "auto", False), ("cd", "fit_transform", False), ("cd", "transform", True),
            ("mu", "auto", False), ("mu", "fit_transform", False), ("mu", "transform", True),
        )
        for solver, mode, projection_required in cases:
            with self.subTest(solver=solver, coordinates=mode):
                model = self.model(solver=solver, training_coordinates=mode, n_components=100)
                record = self.profile(model)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["error_type"], "ValueError")
                self.assertIn("n_components", record["error"])
                self.assertIsNone(record["evaluation"])
                self.assert_unreached(
                    record, "decomposition_fit_seconds", "classifier_fit_seconds",
                    "decomposition_projection_seconds", "classifier_prediction_seconds",
                )
                assertion = self.assert_unreached if projection_required else self.assert_absent
                assertion(record, "training_projection_seconds")
                self.assert_absent(record, "screening_fit_seconds", "screening_prediction_seconds")
                self.assertFalse(hasattr(model, "classifier_"))

    def test_spa_auto_screening_failure_leaves_projection_unavailable(self):
        def fail_screening(*args, **kwargs):
            raise RuntimeError("screening stage fixture")

        model = self.model(screening="absolute", score_threshold=0)
        with patch.object(BinaryScreening, "fit_transform", fail_screening):
            record = self.profile(model)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "screening stage fixture")
        self.assert_measured(record, "screening_fit_seconds")
        self.assert_unreached(record, "decomposition_fit_seconds", "training_projection_seconds",
                              "classifier_fit_seconds", "decomposition_projection_seconds",
                              "classifier_prediction_seconds")
        self.assert_absent(record, "screening_prediction_seconds")

    def test_spa_auto_factorization_failure_leaves_projection_unavailable(self):
        def fail_factorization(*args, **kwargs):
            raise RuntimeError("factorization stage fixture")

        with patch.object(FeatureSPA, "fit_transform", fail_factorization):
            record = self.profile(self.model())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "factorization stage fixture")
        self.assert_measured(record, "decomposition_fit_seconds")
        self.assert_unreached(record, "training_projection_seconds", "classifier_fit_seconds",
                              "decomposition_projection_seconds", "classifier_prediction_seconds")

    def test_failed_training_projection_keeps_attempt_timing_and_restores_methods(self):
        methods = [(BinaryScreening, "fit_transform"), (BinaryScreening, "transform")]
        methods += [(cls, method) for cls in (NMF, FeatureSPA, PCA)
                    for method in ("fit_transform", "transform")]
        methods += [(cls, method) for cls in (KNeighborsClassifier, DummyClassifier)
                    for method in ("fit", "predict_proba")]
        methods += [(DummyClassifier, "predict")]
        originals = {(cls, method): getattr(cls, method) for cls, method in methods}

        def fail_projection(*args, **kwargs):
            raise RuntimeError("training projection stage fixture")

        with patch.object(FeatureSPA, "transform", fail_projection):
            patched = {(cls, method): getattr(cls, method) for cls, method in methods}
            record = self.profile(self.model())
            for (cls, method), original in patched.items():
                self.assertIs(getattr(cls, method), original)
        for (cls, method), original in originals.items():
            self.assertIs(getattr(cls, method), original)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "training projection stage fixture")
        self.assert_measured(record, "decomposition_fit_seconds", "training_projection_seconds")
        self.assert_unreached(record, "classifier_fit_seconds", "decomposition_projection_seconds",
                              "classifier_prediction_seconds")

    def test_completed_training_projection_keeps_timing_after_classifier_failure(self):
        def fail_classifier(*args, **kwargs):
            raise RuntimeError("classifier stage fixture")

        with patch.object(KNeighborsClassifier, "fit", fail_classifier):
            record = self.profile(self.model())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "classifier stage fixture")
        self.assert_measured(record, "decomposition_fit_seconds", "training_projection_seconds",
                             "classifier_fit_seconds")
        self.assert_unreached(record, "decomposition_projection_seconds", "classifier_prediction_seconds")

    def test_successful_spa_auto_and_explicit_transform_reproduce_predictions(self):
        references = []
        for mode in ("auto", "transform"):
            with self.subTest(coordinates=mode):
                model = self.model(training_coordinates=mode)
                reference, fitted = self.reference(model)
                self.assertEqual(fitted.training_coordinates_mode_, "transform")
                self.assertTrue(fitted.training_projection_performed_)
                record = self.profile(model, reference)
                self.assert_exact_predictions(record, reference)
                self.assert_measured(
                    record, "decomposition_fit_seconds", "training_projection_seconds",
                    "classifier_fit_seconds", "decomposition_projection_seconds",
                    "classifier_prediction_seconds",
                )
                self.assert_absent(record, "screening_fit_seconds", "screening_prediction_seconds")
                self.assertFalse(hasattr(model, "classifier_"))
                references.append(reference["predictions"])
        for field in ("predicted_labels", "positive_class_probability"):
            np.testing.assert_array_equal(references[0][field], references[1][field])

    def test_legacy_and_cd_mu_auto_skip_training_projection(self):
        for solver, mode in (("spa", "fit_transform"), ("cd", "auto"),
                             ("cd", "fit_transform"), ("mu", "auto"),
                             ("mu", "fit_transform")):
            with self.subTest(solver=solver, coordinates=mode):
                model = self.model(solver=solver, training_coordinates=mode)
                reference, fitted = self.reference(model)
                self.assertFalse(fitted.training_projection_performed_)
                self.assertEqual(fitted.training_coordinates_mode_, "fit_transform")
                record = self.profile(model, reference)
                self.assert_exact_predictions(record, reference)
                self.assert_absent(record, "training_projection_seconds")
                self.assert_measured(record, "decomposition_fit_seconds", "classifier_fit_seconds",
                                     "decomposition_projection_seconds", "classifier_prediction_seconds")

    def test_nonconverged_training_projection_suppresses_prediction(self):
        original_transform = FeatureSPA.transform

        def warn_and_project(instance, *args, **kwargs):
            warnings.warn("training projection warning fixture", ConvergenceWarning)
            return original_transform(instance, *args, **kwargs)

        with patch.object(FeatureSPA, "transform", warn_and_project), patch.object(
                profiler, "evaluate_classifier", side_effect=AssertionError("prediction must be held")) as evaluate:
            record = self.profile(self.model())
        evaluate.assert_not_called()
        self.assertEqual(record["status"], "nonconverged")
        self.assertIsNone(record["evaluation"])
        self.assertIsNone(record["timing"]["prediction_seconds"])
        self.assertFalse(record["diagnostics"]["training_projection_converged"])
        self.assertTrue(record["fit_warnings"])
        self.assert_measured(record, "decomposition_fit_seconds", "training_projection_seconds",
                             "classifier_fit_seconds")
        self.assert_unreached(record, "decomposition_projection_seconds", "classifier_prediction_seconds")

    def test_raw_screened_and_pca_controls_preserve_structural_zeros(self):
        classifier_stages = {"classifier_fit_seconds", "classifier_prediction_seconds"}
        screening_stages = {"screening_fit_seconds", "screening_prediction_seconds"}
        decomposition_stages = {"decomposition_fit_seconds", "decomposition_projection_seconds"}
        cases = (
            ("raw", KNeighborsClassifier(n_neighbors=3, weights="distance"), classifier_stages),
            ("screened", Pipeline([
                ("screen", BinaryScreening(screening="absolute", score_threshold=0)),
                ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance")),
            ]), classifier_stages | screening_stages),
            ("pca", Pipeline([
                ("screen", BinaryScreening(screening="absolute", score_threshold=0)),
                ("pca", PCA(n_components=2, svd_solver="full", random_state=17)),
                ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance")),
            ]), classifier_stages | screening_stages | decomposition_stages),
        )
        for name, model, measured in cases:
            with self.subTest(model=name):
                reference, _ = self.reference(model)
                record = self.profile(model, reference)
                self.assert_exact_predictions(record, reference)
                self.assert_measured(record, *sorted(measured))
                self.assert_absent(record, *sorted(set(profiler.STAGES) - measured))


if __name__ == "__main__":
    unittest.main()
