"""Reference equivalence, fold isolation and qualification for logistic classifiers."""

from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier
from supervised_nmf.evaluation import calibration_search, comparison_models, evaluate_classifier
from supervised_nmf.study import calibration_knn_grid


class LogisticClassifierTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(limits=1)
        self.addCleanup(self.limits.restore_original_limits)
        random = np.random.RandomState(23)
        self.y = np.arange(80) % 2
        coefficients = random.uniform(.1, 1, (80, 2))
        coefficients[:, 0] += self.y
        basis = np.column_stack([np.eye(2), random.uniform(.1, 1, (2, 10))])
        self.X = (coefficients @ basis + random.uniform(0, .005, (80, 12))).astype(np.float32)

    def model(self, **kwargs):
        settings = dict(classifier="logistic", n_components=2, score_threshold=0,
                        random_state=7, max_iter=3000, tol=1e-3)
        settings.update(kwargs)
        return NMFClassifier(**settings)

    def test_all_solvers_and_coordinate_conventions_match_explicit_reference(self):
        labels = np.where(self.y, "present", "absent")
        for solver in ("spa", "cd", "mu"):
            for coordinates in ("fit_transform", "transform"):
                with self.subTest(solver=solver, coordinates=coordinates):
                    model = self.model(solver=solver, training_coordinates=coordinates,
                                       logistic_C=.3, logistic_class_weight="balanced").fit(self.X, labels)
                    screen = BinaryScreening(screening="absolute", score_threshold=0)
                    selected = screen.fit_transform(self.X, labels)
                    representation = (FeatureSPA(n_components=2, nnls_max_iter=3000) if solver == "spa"
                        else NMF(n_components=2, init="nndsvda", solver=solver, beta_loss="frobenius",
                                 max_iter=3000, tol=1e-3, random_state=7))
                    W = representation.fit_transform(selected)
                    if coordinates == "transform":
                        W = representation.transform(selected)
                    classifier = Pipeline([("scale", StandardScaler()),
                        ("logistic", LogisticRegression(C=.3, solver="lbfgs", max_iter=1000,
                            tol=1e-4, class_weight="balanced", random_state=7))]).fit(W, labels)
                    incoming = self.X[:12] * 1.07
                    projected = representation.transform(screen.transform(incoming))
                    predicted, probabilities = model.predict_with_probabilities(incoming)
                    np.testing.assert_array_equal(predicted, classifier.predict(projected))
                    np.testing.assert_allclose(probabilities, classifier.predict_proba(projected), rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(model.training_coordinates_, W)
                    np.testing.assert_allclose(model.classifier_.named_steps["scale"].mean_, W.mean(axis=0, dtype=np.float64))
                    np.testing.assert_array_equal(model.classifier_n_iter_, classifier.named_steps["logistic"].n_iter_)
                    self.assertTrue(model.converged_ and model.classifier_converged_)

    def test_prediction_preserves_basis_screen_and_scaler(self):
        model = self.model().fit(self.X, self.y)
        scale = model.classifier_.named_steps["scale"]
        before = [model.selected_components_.copy(), model.get_support(), scale.mean_.copy(),
                  scale.var_.copy(), model.training_coordinates_.copy()]
        evaluated = evaluate_classifier(model, self.X * 3, self.y)
        self.assertEqual(evaluated["status"], "valid")
        for expected, actual in zip(before, [model.selected_components_, model.get_support(), scale.mean_,
                                            scale.var_, model.training_coordinates_]):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(scale.n_samples_seen_, len(self.X))
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in model.fit_times_.values()))
        self.assertGreaterEqual(model.fit_times_["total"], sum(value for key, value in model.fit_times_.items() if key != "total"))
        self.assertEqual(evaluated["diagnostics"]["classifier_converged"], True)

    def test_clone_pickle_and_refit_switching_classifier(self):
        model = self.model(logistic_C=.1, logistic_class_weight={0: 1., 1: 2.}).fit(self.X, self.y)
        copied = clone(model)
        self.assertEqual(copied.get_params(), model.get_params())
        self.assertFalse(hasattr(copied, "classifier_"))
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(self.X), model.predict(self.X))
        np.testing.assert_allclose(restored.predict_proba(self.X), model.predict_proba(self.X))
        basis = model.selected_components_.copy()
        model.set_params(classifier="knn").fit(self.X, self.y)
        self.assertIsInstance(model.classifier_, KNeighborsClassifier)
        self.assertIsNone(model.classifier_n_iter_)
        np.testing.assert_array_equal(model.selected_components_, basis)
        model.set_params(classifier="unknown")
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_classifier_parameters_do_not_change_representation(self):
        first = self.model(logistic_C=.01).fit(self.X, self.y)
        second = self.model(logistic_C=100).fit(self.X, self.y)
        np.testing.assert_array_equal(first.selected_components_, second.selected_components_)
        np.testing.assert_array_equal(first.training_coordinates_, second.training_coordinates_)
        self.assertFalse(np.allclose(first.classifier_.named_steps["logistic"].coef_,
                                     second.classifier_.named_steps["logistic"].coef_))

    def test_joint_prediction_matches_decision_rule_even_when_probability_rounds_to_half(self):
        model = self.model().fit(self.X, self.y)
        logistic = model.classifier_.named_steps["logistic"]
        logistic.coef_[:] = 0
        for intercept, expected in ((0., 0), (1e-20, 1), (-1e-20, 0)):
            logistic.intercept_[:] = intercept
            with patch.object(model.decomposition_, "transform", wraps=model.decomposition_.transform) as projection:
                labels, probabilities = model.predict_with_probabilities(self.X[:3])
                self.assertEqual(projection.call_count, 1)
            np.testing.assert_array_equal(probabilities, np.full((3, 2), .5))
            np.testing.assert_array_equal(labels, np.full(3, expected))
            np.testing.assert_array_equal(labels, model.predict(self.X[:3]))

    def test_invalid_logistic_settings_fail_before_factorization(self):
        invalid = ({"classifier": "other"}, {"classifier": None}, {"logistic_C": 0},
                   {"logistic_C": float("nan")}, {"logistic_C": True},
                   {"logistic_max_iter": 0}, {"logistic_max_iter": 1.5}, {"logistic_max_iter": True},
                   {"logistic_tol": 0}, {"logistic_tol": float("inf")},
                   {"logistic_class_weight": "unsupported"}, {"logistic_class_weight": {0: -1}})
        for parameters in invalid:
            with self.subTest(parameters=parameters), patch.object(FeatureSPA, "fit_transform") as fit:
                with self.assertRaises(ValueError):
                    self.model(**parameters).fit(self.X, self.y)
                fit.assert_not_called()
        # Neighbors are irrelevant to logistic regression and remain checked for KNN.
        self.model(n_neighbors=10000).fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "n_neighbors"):
            self.model(classifier="knn", n_neighbors=10000).fit(self.X, self.y)

    def test_nonconverged_logistic_fit_is_recorded_and_cannot_win_calibration(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            truncated = self.model(logistic_max_iter=1).fit(self.X, self.y)
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))
        self.assertFalse(truncated.classifier_converged_ or truncated.converged_)
        self.assertTrue(truncated.classifier_convergence_warnings_)
        self.assertEqual(evaluate_classifier(truncated, self.X, self.y)["status"], "provisional_nonconvergence")
        search = calibration_search(self.model(), {"logistic_max_iter": [1, 1000]}, self.X, self.y, cv=3)
        failed, valid = search.report["candidates"]
        self.assertFalse(failed["valid"])
        self.assertEqual(failed["invalid_fold_count"], 3)
        self.assertIsNone(failed["mean_balanced_accuracy"])
        self.assertTrue(valid["valid"])
        self.assertFalse(search.report["is_fallback"])
        self.assertEqual(search.estimator.logistic_max_iter, 1000)
        self.assertTrue(all(not f["diagnostics"]["classifier_converged"] for f in failed["folds"]))

    def test_screen_and_logistic_scaling_fit_only_within_training_folds(self):
        seen_screen, seen_scaler = [], []
        screen_fit, scale_fit = BinaryScreening.fit, StandardScaler.fit

        def capture_screen(instance, X, y):
            seen_screen.append(X.copy())
            return screen_fit(instance, X, y)

        def capture_scale(instance, X, y=None, **kwargs):
            seen_scaler.append(X.copy())
            return scale_fit(instance, X, y, **kwargs)

        with patch.object(BinaryScreening, "fit", capture_screen), patch.object(StandardScaler, "fit", capture_scale):
            search = calibration_search(self.model(), {"logistic_C": [.1, 1]}, self.X, self.y,
                                        cv=3, seed=11, refit=False)
        self.assertEqual(len(seen_screen), 6)
        self.assertEqual(len(seen_scaler), 6)
        for index, (screen_rows, scale_rows) in enumerate(zip(seen_screen, seen_scaler)):
            training = search.report["folds"][index % 3]["training_indices"]
            np.testing.assert_array_equal(screen_rows, self.X[training])
            reference = self.model().fit(self.X[training], self.y[training])
            np.testing.assert_array_equal(scale_rows, reference.training_coordinates_)
            self.assertLess(len(scale_rows), len(self.X))

    def test_comparison_names_and_parameters_preserve_knn_and_add_logistic(self):
        knn = comparison_models()
        logistic = comparison_models(classifier="logistic")
        self.assertEqual(len(knn), 13)
        self.assertEqual(sum(len(ParameterGrid(grid)) for _, grid in knn.values()), 118)
        self.assertFalse(set(knn) & set(logistic))
        self.assertIn("raw_logistic", logistic)
        self.assertIn("screened_pca_absolute_logistic", logistic)
        self.assertIn("screening_only_absolute_logistic", logistic)
        for name, (model, grid) in logistic.items():
            if "majority" in name:
                continue
            C_key = "logistic_C" if isinstance(model, NMFClassifier) else "logistic__C"
            self.assertEqual(grid[C_key], [.01, .1, 1, 10, 100])
            self.assertFalse(any("n_neighbors" in key for key in grid))
            if isinstance(model, NMFClassifier):
                self.assertEqual(model.classifier, "logistic")
            else:
                self.assertIsInstance(model.named_steps["scale"], StandardScaler)
                self.assertIsInstance(model.named_steps["logistic"], LogisticRegression)
        with self.assertRaises(ValueError):
            comparison_models(classifier="unknown")

    def test_raw_logistic_baseline_matches_explicit_pipeline_and_scales_inside_folds(self):
        estimator, _ = comparison_models(classifier="logistic", random_state=7)["raw_logistic"]
        fitted = clone(estimator).fit(self.X, self.y)
        reference = Pipeline([("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1, max_iter=1000, tol=1e-4, solver="lbfgs", random_state=7))]).fit(self.X, self.y)
        np.testing.assert_array_equal(fitted.predict(self.X), reference.predict(self.X))
        np.testing.assert_allclose(fitted.predict_proba(self.X), reference.predict_proba(self.X))
        original_fit, observed = StandardScaler.fit, []

        def capture(instance, X, y=None, **kwargs):
            observed.append(X.copy())
            return original_fit(instance, X, y, **kwargs)

        with patch.object(StandardScaler, "fit", capture):
            calibrated = calibration_search(estimator, {"logistic__C": [.1]}, self.X, self.y,
                                            cv=3, seed=11, refit=False)
        for rows, fold in zip(observed, calibrated.report["folds"]):
            np.testing.assert_array_equal(rows, self.X[fold["training_indices"]])
        self.assertEqual(len(observed), 3)
        with patch.object(fitted.named_steps["scale"], "transform", wraps=fitted.named_steps["scale"].transform) as transform:
            evaluated = evaluate_classifier(fitted, self.X, self.y)
            self.assertEqual(transform.call_count, 1)
        self.assertEqual(evaluated["status"], "valid")
        self.assertIsNotNone(evaluated["metrics"]["roc_auc"])

    def test_standalone_logistic_baseline_retains_convergence_evidence(self):
        model, _ = comparison_models(classifier="logistic", logistic_max_iter=1)["raw_logistic"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(self.X, self.y)
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))
        logistic = model.named_steps["logistic"]
        self.assertFalse(logistic.converged_)
        self.assertTrue(logistic.convergence_warnings_)
        report = evaluate_classifier(model, self.X, self.y)
        self.assertEqual(report["status"], "provisional_nonconvergence")
        self.assertFalse(report["diagnostics"]["classifier_converged"])
        self.assertTrue(report["diagnostics"]["classifier_convergence_warnings"])
        model.set_params(logistic__max_iter=1000).fit(self.X, self.y)
        self.assertTrue(logistic.converged_)
        self.assertEqual(logistic.convergence_warnings_, ())
        self.assertEqual(evaluate_classifier(model, self.X, self.y)["status"], "valid")
        with self.assertRaises(ValueError):
            model.fit(self.X, np.zeros(len(self.y), dtype=int))
        self.assertFalse(hasattr(logistic, "converged_"))
        self.assertFalse(hasattr(logistic, "convergence_warnings_"))
        with self.assertRaises(NotFittedError):
            logistic.predict(self.X)

    def test_shared_knn_grid_rejects_logistic_estimators_and_classifier_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            for estimator, grid, error in (
                    (self.model(), {"logistic_C": [.1, 1]}, TypeError),
                    (self.model(classifier="knn"), {"classifier": ["logistic"]}, ValueError),
                    (self.model(classifier="knn"), {"logistic_C": [.1, 1]}, ValueError)):
                with self.subTest(grid=grid), self.assertRaises(error):
                    calibration_knn_grid(estimator, grid, self.X, self.y, cv=3,
                                         checkpoint_dir=Path(directory) / "checkpoint")
            self.assertFalse((Path(directory) / "checkpoint").exists())


if __name__ == "__main__":
    unittest.main()
