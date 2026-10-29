"""Numerical and leakage checks for Chang Cheng's supervised NMF estimators."""

import pickle
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.stats import ttest_ind
from sklearn.base import clone, is_classifier
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf.classifier import NMFClassifier, _capture_projection, _record_warnings
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.screening import BinaryScreening, _percentile_support


class CoreTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls._threads.restore_original_limits()

    def setUp(self):
        rng = np.random.default_rng(52)
        self.y = np.repeat(["negative", "positive"], 24)
        self.X = rng.gamma(2, 1, size=(48, 12))
        self.X[24:, :3] += 3


class ScreeningTests(CoreTestCase):
    def test_pooled_t_and_anova_scores(self):
        selector = BinaryScreening(percentile=100).fit(self.X, self.y)
        expected_f = f_classif(self.X, self.y)[0]
        expected_t = -ttest_ind(self.X[:24], self.X[24:], equal_var=True, axis=0).statistic
        np.testing.assert_allclose(selector.scores_, expected_f, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(selector.t_scores_, expected_t, rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(selector.classes_, ["negative", "positive"])

    def test_percentile_agrees_with_sklearn(self):
        for percentile in (1, 10, 25, 33, 75, 100):
            expected = SelectPercentile(f_classif, percentile=percentile).fit(self.X, self.y)
            actual = BinaryScreening(percentile=percentile).fit(self.X, self.y)
            np.testing.assert_array_equal(actual.get_support(), expected.get_support())

    def test_percentile_rounding_and_ties(self):
        for scores in (np.arange(13.0), np.array([0., 1., 1., 1., 2., 3., 3.]), np.ones(7)):
            for percentile in (0, 0.1, 10, 25, 50, 75, 99, 100):
                sklearn_selector = SelectPercentile(percentile=percentile)
                sklearn_selector.scores_ = scores
                expected = sklearn_selector.get_support()
                np.testing.assert_array_equal(_percentile_support(scores, np.ones(scores.size, bool), percentile), expected)

    def test_inclusive_absolute_t_threshold(self):
        probe = BinaryScreening(percentile=100).fit(self.X, self.y)
        index = int(np.argsort(np.abs(probe.t_scores_))[-2])
        threshold = abs(probe.t_scores_[index])
        exact = BinaryScreening(screening="absolute", score_threshold=threshold).fit(self.X, self.y)
        above = BinaryScreening(screening="absolute", score_threshold=np.nextafter(threshold, np.inf)).fit(self.X, self.y)
        self.assertTrue(exact.get_support()[index])
        self.assertFalse(above.get_support()[index])

    def test_constant_and_classwise_constant_features(self):
        X = np.column_stack([np.ones(48), np.r_[np.full(24, .1), np.full(24, .3)], self.X[:, 0]])
        selector = BinaryScreening(screening="absolute", score_threshold=0).fit(X, self.y)
        np.testing.assert_array_equal(selector.get_support(), [False, True, True])
        self.assertEqual(selector.n_constant_features_, 1)
        self.assertEqual(selector.n_infinite_scores_, 1)
        self.assertEqual(selector.n_classwise_constant_features_, 1)
        self.assertTrue(np.isposinf(selector.t_scores_[1]))
        percentile = BinaryScreening(percentile=40).fit(X, self.y)
        np.testing.assert_array_equal(percentile.get_support(), [False, True, False])

    def test_statistic_scale_invariance_and_no_input_mutation(self):
        original = self.X.copy()
        base = BinaryScreening(percentile=100).fit(self.X, self.y)
        scales = np.geomspace(1e-200, 1e200, self.X.shape[1])
        rescaled = BinaryScreening(percentile=100).fit(self.X * scales, self.y)
        np.testing.assert_allclose(base.t_scores_, rescaled.t_scores_, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(self.X, original)
        np.testing.assert_array_equal(base.transform(self.X), original)

    def test_rejects_invalid_data_and_empty_selection(self):
        for X, y in ((np.ones_like(self.X), self.y), (-self.X, self.y),
                     (self.X, np.zeros(48)), (self.X, np.r_[np.zeros(47), 1]),
                     (self.X, np.tile([0, 1, 2], 16))):
            with self.assertRaises(ValueError):
                BinaryScreening().fit(X, y)
        with self.assertRaisesRegex(ValueError, "zero features"):
            BinaryScreening(percentile=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            BinaryScreening(percentile=100).fit(self.X * np.nan, self.y)


class SPATests(CoreTestCase):
    def separable(self):
        rng = np.random.default_rng(83)
        W = rng.uniform(.1, 2, size=(30, 3))
        H = np.column_stack([np.diag([8., .2, 3.]), rng.uniform(.2, 4, size=(3, 8))])
        return W, H, W @ H

    def test_exact_feature_separability_and_fixed_basis_prediction(self):
        W, H, X = self.separable()
        estimator = FeatureSPA(n_components=3)
        train = estimator.fit_transform(X)
        self.assertEqual(set(estimator.anchor_indices_), {0, 1, 2})
        np.testing.assert_allclose(train @ estimator.components_, X, rtol=1e-10, atol=1e-10)
        components = estimator.components_.copy()
        rng = np.random.default_rng(91)
        X_new = rng.uniform(.1, 2, size=(9, 3)) @ H
        prediction = estimator.transform(X_new)
        np.testing.assert_allclose(estimator.inverse_transform(prediction), X_new, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(estimator.components_, components)
        np.testing.assert_allclose(estimator.transform(X), train, rtol=1e-10, atol=1e-10)

    def test_zero_features_and_deterministic_anchors(self):
        _, _, X = self.separable()
        X = np.column_stack([np.zeros(X.shape[0]), X])
        first = FeatureSPA(3).fit(X)
        second = FeatureSPA(3).fit(X)
        self.assertNotIn(0, first.anchor_indices_)
        np.testing.assert_array_equal(first.anchor_indices_, second.anchor_indices_)
        np.testing.assert_array_equal(first.components_[:, 0], np.zeros(3))

    def test_rank_deficiency_and_invalid_component_counts(self):
        rank_one = np.outer(np.arange(1, 6), np.arange(1, 5))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            FeatureSPA(2).fit(rank_one)
        for rank in (0, 1.5, 5, True):
            with self.assertRaises(ValueError):
                FeatureSPA(rank).fit(rank_one)
        with self.assertRaises(ValueError):
            FeatureSPA(1).fit(np.zeros((4, 3)))

    def test_noisy_fit_and_transform_coordinate_conventions(self):
        _, _, X = self.separable()
        X += np.random.default_rng(6).uniform(0, .05, X.shape)
        estimator = FeatureSPA(3)
        anchor_coordinates = estimator.fit_transform(X)
        projected = estimator.transform(X)
        self.assertGreater(np.linalg.norm(anchor_coordinates - projected), 1e-6)
        self.assertLessEqual(np.linalg.norm(X - projected @ estimator.components_), estimator.reconstruction_err_ + 1e-10)


class ClassifierTests(CoreTestCase):
    def model(self, **kwargs):
        options = dict(solver="cd", screening="percentile", percentile=50, n_components=2,
                       n_neighbors=3, max_iter=2000, random_state=10)
        options.update(kwargs)
        return NMFClassifier(**options)

    def test_selected_defaults_equal_explicit_method_and_serialize(self):
        parameters = NMFClassifier().get_params()
        self.assertEqual((parameters["solver"], parameters["screening"], parameters["max_iter"]),
                         ("spa", "absolute", 20000))
        settings = {"score_threshold": 2.0, "n_components": 2, "n_neighbors": 3, "random_state": 10}
        default = NMFClassifier(**settings).fit(self.X, self.y)
        explicit = NMFClassifier(solver="spa", screening="absolute", **settings).fit(self.X, self.y)
        np.testing.assert_array_equal(default.get_support(), explicit.get_support())
        np.testing.assert_array_equal(default.training_coordinates_, explicit.training_coordinates_)
        np.testing.assert_array_equal(default.selected_components_, explicit.selected_components_)
        np.testing.assert_array_equal(default.components_, explicit.components_)
        incoming = self.X * 1.07
        labels, probabilities = default.predict_with_probabilities(incoming)
        expected_labels, expected_probabilities = explicit.predict_with_probabilities(incoming)
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(probabilities, expected_probabilities)
        copied = clone(default)
        self.assertEqual(copied.get_params(), default.get_params())
        self.assertFalse(hasattr(copied, "classifier_"))
        restored = pickle.loads(pickle.dumps(default))
        self.assertEqual(restored.get_params(), default.get_params())
        restored_labels, restored_probabilities = restored.predict_with_probabilities(incoming)
        np.testing.assert_array_equal(labels, restored_labels)
        np.testing.assert_array_equal(probabilities, restored_probabilities)

    def test_explicit_method_and_numerical_overrides_remain_available(self):
        model = NMFClassifier(percentile=50, n_neighbors=3, random_state=10)
        model.set_params(solver="cd", screening="absolute", score_threshold=0, max_iter=2000)
        copied = clone(model)
        self.assertEqual(copied.get_params(), model.get_params())
        copied.fit(self.X, self.y)
        self.assertIsInstance(copied.decomposition_, NMF)
        self.assertEqual(copied.selector_.screening, "absolute")
        self.assertEqual(copied.decomposition_.max_iter, 2000)
        copied.set_params(solver="spa", screening="none", max_iter=37).fit(self.X, self.y)
        self.assertIsNone(copied.selector_)
        self.assertIsInstance(copied.decomposition_, FeatureSPA)
        self.assertEqual(copied.decomposition_.nnls_max_iter, 37)
        self.assertEqual(copied.get_params()["max_iter"], 37)

    def test_default_numerical_cap_reaches_all_solvers(self):
        for solver in ("cd", "spa", "mu"):
            model = NMFClassifier(solver=solver, score_threshold=2.0, n_neighbors=3, random_state=10).fit(self.X, self.y)
            cap = model.decomposition_.nnls_max_iter if solver == "spa" else model.decomposition_.max_iter
            self.assertEqual(cap, 20000)
            self.assertEqual(model.tol, 1e-4)

    def test_invalid_explicit_method_values_do_not_use_defaults(self):
        for solver in (None, "invalid"):
            with self.assertRaisesRegex(ValueError, "solver"):
                NMFClassifier(solver=solver).fit(self.X, self.y)
        for screening in (None, "invalid"):
            with self.assertRaisesRegex(ValueError, "screening"):
                NMFClassifier(screening=screening).fit(self.X, self.y)

    def test_classification_serialization_and_fixed_basis(self):
        for solver in ("cd", "spa", "mu"):
            model = self.model(solver=solver).fit(self.X, self.y)
            self.assertTrue(is_classifier(model))
            components = model.components_.copy()
            fitted_basis = model.decomposition_.components_.copy()
            training_coordinates = model.training_coordinates_.copy()
            score_before = model.selector_.scores_.copy()
            predicted = model.predict(self.X)
            proba = model.predict_proba(self.X)
            np.testing.assert_allclose(proba.sum(axis=1), np.ones(48))
            np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], predicted)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(restored.predict(self.X), predicted)
            with _capture_projection(model) as diagnostic:
                model.predict(self.X * 1.1)
            np.testing.assert_array_equal(model.components_, components)
            np.testing.assert_array_equal(model.decomposition_.components_, fitted_basis)
            np.testing.assert_array_equal(model.training_coordinates_, training_coordinates)
            np.testing.assert_array_equal(model.selector_.scores_, score_before)
            self.assertEqual(model.components_.shape, (2, 12))
            np.testing.assert_array_equal(model.components_[:, ~model.get_support()], 0)
            self.assertGreaterEqual(model.fit_times_["total"], model.fit_times_["factorization"])
            self.assertTrue(all(np.isfinite(seconds) and seconds >= 0 for seconds in model.fit_times_.values()))
            self.assertGreaterEqual(diagnostic["last_transform_seconds"], 0)
            self.assertFalse(hasattr(model, "last_transform_seconds_"))
            copied = clone(model)
            self.assertEqual(copied.get_params(), model.get_params())
            self.assertFalse(hasattr(copied, "classifier_"))

    def test_screening_fits_only_inside_cv_training_folds(self):
        cv = StratifiedKFold(3, shuffle=True, random_state=12)
        folds = list(cv.split(self.X, self.y))
        for solver in ("cd", "mu"):
            with self.subTest(solver=solver):
                result = cross_validate(self.model(solver=solver), self.X, self.y, cv=folds,
                                        return_estimator=True, error_score="raise")
                for (train, _), model in zip(folds, result["estimator"]):
                    expected = BinaryScreening(percentile=50).fit(self.X[train], self.y[train])
                    self.assertEqual(model.selector_.n_samples_seen_, len(train))
                    np.testing.assert_allclose(model.selector_.scores_, expected.scores_)
                    np.testing.assert_array_equal(model.selected_features_, expected.selected_features_)

    def test_combined_prediction_projects_once(self):
        model = self.model().fit(self.X, self.y)
        expected_labels = model.predict(self.X)
        expected_probabilities = model.predict_proba(self.X)
        with patch.object(model, "transform", wraps=model.transform) as transform:
            labels, probabilities = model.predict_with_probabilities(self.X)
        transform.assert_called_once()
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_allclose(probabilities, expected_probabilities)

    def test_training_reprojection_is_explicit(self):
        for solver in ("cd", "spa", "mu"):
            with self.subTest(solver=solver):
                model = self.model(solver=solver, training_coordinates="transform").fit(self.X, self.y)
                np.testing.assert_allclose(model.training_coordinates_, model.transform(self.X))
                expected_error = np.linalg.norm(self.X[:, model.selected_features_] - model.training_coordinates_ @ model.selected_components_)
                self.assertAlmostEqual(model.reconstruction_err_, expected_error)

    def test_mu_matches_independent_screening_nmf_knn_pipeline(self):
        X = self.X.astype(np.float32)
        # Compare pipeline composition with matching query layouts. Independent
        # projection accuracy across layouts is checked against a closed form below.
        incoming = np.asfortranarray((self.X[:11] * 1.07 + .03).astype(np.float32))
        for screening in ("absolute", "percentile", "none"):
            with self.subTest(screening=screening):
                steps = []
                if screening != "none":
                    steps.append(("screen", BinaryScreening(screening=screening, percentile=50,
                                                            score_threshold=2.0)))
                steps.extend([
                    ("nmf", NMF(n_components=2, init="nndsvda", solver="mu", beta_loss="frobenius",
                                max_iter=2000, tol=1e-4, random_state=10)),
                    ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance")),
                ])
                reference = Pipeline(steps).fit(X, self.y)
                model = self.model(solver="mu", screening=screening, weights="distance").fit(X, self.y)
                expected_coordinates = reference.named_steps["knn"]._fit_X
                np.testing.assert_array_equal(model.training_coordinates_, expected_coordinates)
                np.testing.assert_array_equal(model.selected_components_, reference.named_steps["nmf"].components_)
                np.testing.assert_array_equal(model.predict(incoming), reference.predict(incoming))
                np.testing.assert_allclose(model.predict_proba(incoming), reference.predict_proba(incoming),
                                           rtol=1e-6, atol=1e-7)
                self.assertEqual(model.training_coordinates_.dtype, np.float32)
                self.assertEqual(model.n_iter_, reference.named_steps["nmf"].n_iter_)
                self.assertAlmostEqual(model.reconstruction_err_, reference.named_steps["nmf"].reconstruction_err_)

    def test_mu_rank_one_projection_matches_closed_form_across_layouts(self):
        # The fixed nonzero rank-one basis has a unique NNLS projection.
        # Its scale comes from the fit; joint NMF factors are not unique.
        for dtype in (np.float32, np.float64):
            basis = np.array([[1, 2, 0, 4]], dtype=dtype)
            coefficients = np.array([[.2], [.5], [1.2], [2.], [3.], [4.]], dtype=dtype)
            training = coefficients @ basis
            labels = np.repeat([0, 1], 3)
            model = self.model(solver="mu", screening="none", n_components=1,
                               n_neighbors=1).fit(training, labels)
            fitted_basis = model.decomposition_.components_.copy()
            fitted_coordinates = model.training_coordinates_.copy()
            h = fitted_basis.astype(np.float64)
            squared_norm = float((h @ h.T)[0, 0])
            self.assertTrue(np.isfinite(squared_norm) and squared_norm > 0)
            query = np.array([[0, 0, 0, 0], [1e-8, 2e-8, 0, 4e-8],
                              [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 7, 0],
                              [1, 2, 3, 4]], dtype=dtype)
            expected = np.maximum(query.astype(np.float64) @ h.T / squared_norm, 0)
            backing = np.zeros((len(query), 2 * query.shape[1]), dtype=dtype)
            backing[:, ::2] = query
            readonly = query.copy()
            readonly.flags.writeable = False
            query_before = query.copy()
            layouts = {"C": query, "F": np.asfortranarray(query),
                       "strided": backing[:, ::2], "readonly": readonly}
            for layout, incoming in layouts.items():
                with self.subTest(dtype=np.dtype(dtype).name, layout=layout):
                    np.testing.assert_array_equal(incoming, query_before)
                    actual = model.transform(incoming)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertTrue((actual >= 0).all())
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                    np.testing.assert_array_equal(incoming, query_before)
                    np.testing.assert_array_equal(model.decomposition_.components_, fitted_basis)
                    np.testing.assert_array_equal(model.training_coordinates_, fitted_coordinates)

    def test_original_float32_cd_computations_preserved(self):
        X = self.X.astype(np.float32)
        model = self.model().fit(X, self.y)
        screened = model.selector_.transform(X)
        self.assertEqual(screened.dtype, np.float32)
        self.assertEqual(model.training_coordinates_.dtype, np.float32)
        self.assertEqual(model.decomposition_.components_.dtype, np.float32)
        self.assertEqual(model.selector_.scores_.dtype, np.float64)
        original = NMF(n_components=2, init="nndsvda", solver="cd", max_iter=2000, random_state=10)
        expected = original.fit_transform(screened)
        np.testing.assert_array_equal(model.training_coordinates_, expected)
        np.testing.assert_array_equal(model.decomposition_.components_, original.components_)

    def test_convergence_warnings_are_recorded_and_emitted(self):
        for solver in ("cd", "mu"):
            with self.subTest(solver=solver), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = self.model(solver=solver, max_iter=1).fit(self.X, self.y)
                with _capture_projection(model) as diagnostic:
                    model.transform(self.X)
                self.assertFalse(model.converged_)
                self.assertFalse(diagnostic["last_transform_converged"])
                self.assertTrue(model.convergence_warnings_)
                self.assertTrue(diagnostic["last_transform_warnings"])
                self.assertTrue(any(issubclass(w.category, ConvergenceWarning) for w in caught))

    def test_warnings_preceding_solver_failure_remain_visible(self):
        def failed_operation():
            warnings.warn("Failure diagnostic", ConvergenceWarning)
            raise ValueError("Solver failed")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "Solver failed"):
                _record_warnings(failed_operation)
        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0].message), "Failure diagnostic")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(ValueError, "Solver failed"):
                _record_warnings(failed_operation)

    def test_refit_state_and_failed_refit(self):
        model = self.model().fit(self.X, self.y)
        model.set_params(screening="none")
        model.fit(self.X[:, :5], self.y)
        self.assertIsNone(model.selector_)
        self.assertEqual(model.components_.shape, (2, 5))
        with self.assertRaises(ValueError):
            model.fit(-self.X, self.y)
        with self.assertRaises(NotFittedError):
            model.predict(self.X)

    def test_rank_and_feature_mismatch_errors(self):
        with self.assertRaisesRegex(ValueError, "selected features"):
            self.model(n_components=10, percentile=10).fit(self.X, self.y)
        model = self.model().fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "features"):
            model.predict(self.X[:, :5])


if __name__ == "__main__":
    unittest.main()
