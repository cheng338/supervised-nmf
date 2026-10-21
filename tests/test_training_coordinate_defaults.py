"""Classifier defaults preserve one fitted SPA map and explicit legacy modes."""
import pickle
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier


class TrainingCoordinateDefaultTests(unittest.TestCase):
    def setUp(self):
        limit = threadpool_limits(1)
        self.addCleanup(limit.restore_original_limits)
        rng = np.random.default_rng(817)
        basis = np.hstack([np.eye(3), rng.uniform(.02, .8, (3, 12))])
        coordinates = rng.uniform(.05, 2, (80, 3))
        self.y = ((coordinates[:, 0] + coordinates[:, 1]) > 2).astype(int)
        self.X = coordinates @ basis + rng.uniform(0, .25, (80, 15))
        self.query = rng.uniform(.05, 2, (19, 3)) @ basis + rng.uniform(0, .25, (19, 15))

    def model(self, **changes):
        params = dict(n_components=3, screening="none", n_neighbors=5,
                      random_state=13, max_iter=10000, tol=1e-4,
                      logistic_C=.7, logistic_tol=1e-8, logistic_max_iter=2000)
        params.update(changes)
        return NMFClassifier(**params)

    @staticmethod
    def projected(basis, X):
        return np.vstack([nnls(basis.T, row, maxiter=10000)[0] for row in X])

    def test_spa_default_knn_uses_independent_fixed_basis_nnls_for_both_inputs(self):
        model = self.model().fit(self.X, self.y)
        basis = model.selected_components_.copy()
        expected_train = self.projected(basis, self.X)
        expected_query = self.projected(basis, self.query)
        self.assertEqual(model.get_params()["training_coordinates"], "auto")
        self.assertEqual(model.training_coordinates_mode_, "transform")
        self.assertTrue(model.training_projection_performed_)
        self.assertGreater(np.linalg.norm(expected_train - model.decomposition_.fit_coordinates_), .001)
        np.testing.assert_allclose(model.training_coordinates_, expected_train, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(model.classifier_._fit_X, expected_train, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(model.transform(self.query), expected_query, atol=2e-10, rtol=2e-10)
        reference = KNeighborsClassifier(n_neighbors=5).fit(expected_train, self.y)
        np.testing.assert_array_equal(model.predict(self.query), reference.predict(expected_query))
        np.testing.assert_allclose(model.predict_proba(self.query), reference.predict_proba(expected_query))
        np.testing.assert_array_equal(model.selected_components_, basis)
        np.testing.assert_array_equal(model.decomposition_.components_, basis)
        self.assertAlmostEqual(model.reconstruction_err_, np.linalg.norm(self.X - expected_train @ basis))
        self.assertAlmostEqual(model.factorization_reconstruction_err_,
                               np.linalg.norm(self.X - model.decomposition_.fit_coordinates_ @ basis))

    def test_spa_default_logistic_scaler_and_classifier_receive_projected_training_rows(self):
        original_fit = StandardScaler.fit
        supplied_rows = []

        def capture_fit(scaler, X, *args, **kwargs):
            supplied_rows.append(np.asarray(X).copy())
            return original_fit(scaler, X, *args, **kwargs)

        with patch.object(StandardScaler, "fit", capture_fit):
            model = self.model(classifier="logistic").fit(self.X, self.y)
        expected_train = self.projected(model.selected_components_, self.X)
        expected_query = self.projected(model.selected_components_, self.query)
        self.assertEqual(len(supplied_rows), 1)
        np.testing.assert_allclose(supplied_rows[0], expected_train, atol=2e-10, rtol=2e-10)
        reference = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=.7, solver="lbfgs", tol=1e-8,
                                             max_iter=2000, random_state=13)),
        ]).fit(expected_train, self.y)
        np.testing.assert_allclose(model.classifier_.named_steps["scale"].mean_, expected_train.mean(axis=0))
        np.testing.assert_allclose(model.classifier_.named_steps["logistic"].coef_,
                                   reference.named_steps["logistic"].coef_, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(model.predict_proba(self.query), reference.predict_proba(expected_query),
                                   atol=2e-8, rtol=2e-8)

    def test_explicit_legacy_spa_retains_anchor_coordinates_and_lowlevel_contract(self):
        spa = FeatureSPA(3, nnls_max_iter=10000)
        anchors = spa.fit_transform(self.X)
        np.testing.assert_array_equal(anchors, self.X[:, spa.anchor_indices_])
        self.assertGreater(np.linalg.norm(spa.transform(self.X) - anchors), .001)
        legacy = self.model(training_coordinates="fit_transform").fit(self.X, self.y)
        self.assertEqual(legacy.training_coordinates_mode_, "fit_transform")
        self.assertFalse(legacy.training_projection_performed_)
        self.assertIsNone(legacy.training_projection_converged_)
        np.testing.assert_array_equal(legacy.training_coordinates_, anchors)
        reference = KNeighborsClassifier(n_neighbors=5).fit(anchors, self.y)
        np.testing.assert_array_equal(legacy.predict(self.query), reference.predict(spa.transform(self.query)))
        np.testing.assert_array_equal(legacy.selected_components_, spa.components_)

    def test_explicit_transform_matches_spa_auto(self):
        automatic = self.model().fit(self.X, self.y)
        explicit = self.model(training_coordinates="transform").fit(self.X, self.y)
        np.testing.assert_array_equal(automatic.training_coordinates_, explicit.training_coordinates_)
        np.testing.assert_array_equal(automatic.predict_proba(self.query), explicit.predict_proba(self.query))

    def test_cd_and_mu_auto_keep_historical_sklearn_training_coordinates(self):
        for solver in ("cd", "mu"):
            with self.subTest(solver=solver):
                reference = NMF(n_components=3, init="nndsvda", solver=solver,
                                beta_loss="frobenius", max_iter=10000, tol=1e-4, random_state=13)
                expected = reference.fit_transform(self.X)
                for mode in ("auto", "fit_transform"):
                    model = self.model(solver=solver, training_coordinates=mode).fit(self.X, self.y)
                    self.assertEqual(model.training_coordinates_mode_, "fit_transform")
                    self.assertFalse(model.training_projection_performed_)
                    self.assertIsNone(model.training_projection_converged_)
                    np.testing.assert_array_equal(model.training_coordinates_, expected)
                    np.testing.assert_array_equal(model.selected_components_, reference.components_)
                    classifier = KNeighborsClassifier(n_neighbors=5).fit(expected, self.y)
                    np.testing.assert_array_equal(model.predict_proba(self.query),
                                                  classifier.predict_proba(reference.transform(self.query)))

    def test_explicit_transform_remains_supported_for_cd_and_mu(self):
        for solver in ("cd", "mu"):
            with self.subTest(solver=solver):
                model = self.model(solver=solver, training_coordinates="transform").fit(self.X, self.y)
                self.assertEqual(model.training_coordinates_mode_, "transform")
                self.assertTrue(model.training_projection_performed_)
                np.testing.assert_array_equal(model.training_coordinates_, model.decomposition_.transform(self.X))

    def test_clone_and_solver_refit_resolve_auto_without_mutating_constructor(self):
        model = self.model().fit(self.X, self.y)
        copied = clone(model)
        self.assertEqual(copied.training_coordinates, "auto")
        self.assertFalse(hasattr(copied, "training_coordinates_mode_"))
        with self.assertRaises(NotFittedError):
            copied.predict(self.query)
        model.set_params(solver="cd").fit(self.X, self.y)
        self.assertEqual(model.training_coordinates, "auto")
        self.assertEqual(model.training_coordinates_mode_, "fit_transform")
        model.set_params(solver="spa").fit(self.X, self.y)
        self.assertEqual(model.training_coordinates_mode_, "transform")

    def test_parameter_change_waits_for_refit_but_projection_execution_is_mutable(self):
        model = self.model().fit(self.X, self.y)
        original_coordinates = model.training_coordinates_.copy()
        original_scores = model.predict_proba(self.query)
        model.set_params(training_coordinates="fit_transform", projection_solver="scipy")
        np.testing.assert_array_equal(model.training_coordinates_, original_coordinates)
        np.testing.assert_allclose(model.predict_proba(self.query), original_scores)
        self.assertEqual(model.training_coordinates_mode_, "transform")
        self.assertEqual(model.decomposition_.projection_solver, "scipy")
        model.fit(self.X, self.y)
        self.assertEqual(model.training_coordinates_mode_, "fit_transform")
        np.testing.assert_array_equal(model.training_coordinates_, model.decomposition_.fit_coordinates_)

    def test_invalid_modes_fail_cleanly_and_clear_fitted_state(self):
        model = self.model().fit(self.X, self.y)
        for invalid in (None, "anchor", "AUTO", 0, True, [], {}):
            with self.subTest(mode=invalid):
                model.set_params(training_coordinates=invalid)
                with self.assertRaisesRegex(ValueError, "training_coordinates must be"):
                    model.fit(self.X, self.y)
                self.assertFalse(hasattr(model, "training_coordinates_mode_"))
                self.assertFalse(hasattr(model, "training_coordinates_"))
                with self.assertRaises(NotFittedError):
                    model.predict(self.query)

    def test_pickles_preserve_explicit_modes_and_restore_actual_legacy_fit_metadata(self):
        for mode in ("auto", "fit_transform", "transform"):
            with self.subTest(mode=mode):
                model = self.model(training_coordinates=mode).fit(self.X, self.y)
                expected = model.predict_proba(self.query)
                resolved = model.training_coordinates_mode_
                restored = pickle.loads(pickle.dumps(model))
                self.assertEqual(restored.training_coordinates, mode)
                self.assertEqual(restored.training_coordinates_mode_, resolved)
                np.testing.assert_array_equal(restored.predict_proba(self.query), expected)
                # Emulate a pre-mode serialized model after constructor-only
                # set_params: saved fitting metadata must win over that change.
                del model.training_coordinates_mode_
                model.set_params(training_coordinates="fit_transform" if resolved == "transform" else "transform")
                restored = pickle.loads(pickle.dumps(model))
                self.assertEqual(restored.training_coordinates_mode_, resolved)
                np.testing.assert_array_equal(restored.predict_proba(self.query), expected)

    def test_pickle_without_coordinate_parameter_restores_historical_policy(self):
        legacy = self.model(training_coordinates="fit_transform").fit(self.X, self.y)
        expected = legacy.predict_proba(self.query)
        del legacy.training_coordinates
        del legacy.training_coordinates_mode_
        del legacy.training_projection_performed_
        restored = pickle.loads(pickle.dumps(legacy))
        self.assertEqual(restored.get_params()["training_coordinates"], "fit_transform")
        self.assertEqual(restored.training_coordinates_mode_, "fit_transform")
        np.testing.assert_array_equal(restored.predict_proba(self.query), expected)
        self.assertEqual(clone(restored).training_coordinates, "fit_transform")

    def test_default_projection_warning_is_forwarded_and_reported(self):
        original_transform = FeatureSPA.transform

        def warn_and_project(spa, X):
            warnings.warn("training projection warning fixture", ConvergenceWarning)
            return original_transform(spa, X)

        with patch.object(FeatureSPA, "transform", warn_and_project), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self.model().fit(self.X, self.y)
        self.assertTrue(any("training projection warning fixture" in str(item.message) for item in caught))
        self.assertTrue(model.training_projection_performed_)
        self.assertFalse(model.training_projection_converged_)
        self.assertFalse(model.converged_)
        self.assertEqual(model.training_projection_convergence_warnings_, ("training projection warning fixture",))
        self.assertTrue(model.factorization_converged_)

    def test_training_nnls_is_in_fit_timing_and_failure_leaves_no_usable_model(self):
        clock = [0.0]

        def tick():
            clock[0] += .01
            return clock[0]

        original_transform = FeatureSPA.transform

        def timed_project(spa, X):
            clock[0] += 7.0
            return original_transform(spa, X)

        with patch("supervised_nmf.classifier.time.perf_counter", tick), patch.object(FeatureSPA, "transform", timed_project):
            model = self.model().fit(self.X, self.y)
        self.assertGreaterEqual(model.fit_times_["training_projection"], 7.0)
        self.assertLess(model.fit_times_["factorization"], 7.0)
        self.assertGreaterEqual(model.fit_times_["total"], sum(value for name, value in model.fit_times_.items() if name != "total"))
        with patch.object(FeatureSPA, "transform", side_effect=RuntimeError("projection failed")):
            with self.assertRaisesRegex(RuntimeError, "projection failed"):
                model.fit(self.X, self.y)
        self.assertFalse(hasattr(model, "training_coordinates_mode_"))
        self.assertFalse(hasattr(model, "classifier_"))


if __name__ == "__main__":
    unittest.main()
