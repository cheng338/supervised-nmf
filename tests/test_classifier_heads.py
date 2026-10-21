"""Author: Chang Cheng. Fitted coordinate metrics and genuine linear-SVM margins."""
import pickle
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier
from supervised_nmf.evaluation import calibration_search


class ClassifierHeadTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        random = np.random.default_rng(133)
        self.y = np.where(np.arange(80) % 2, "present", "absent")
        W = random.uniform(.1, 1, (80, 2))
        W[:, 0] += (self.y == "present") * .7
        H = np.column_stack([np.diag([.4, 3.]), random.uniform(.1, 1, (2, 10))])
        self.X = (W @ H + random.uniform(0, .002, (80, 12))).astype(np.float32)
        self.query = self.X[:13] * 1.17

    def model(self, **settings):
        parameters = dict(n_components=2, screening="none", random_state=11,
                          max_iter=4000, tol=1e-3, n_neighbors=5,
                          projection_solver="scipy")
        parameters.update(settings)
        return NMFClassifier(**parameters)

    def test_sd_head_matches_explicit_fitted_pipeline_for_each_solver(self):
        for solver in ("spa", "cd", "mu"):
            for weights in ("uniform", "distance"):
                with self.subTest(solver=solver, weights=weights):
                    model = self.model(solver=solver, weights=weights,
                                       knn_normalization="sd").fit(self.X, self.y)
                    reference = Pipeline([
                        ("scale", StandardScaler(with_mean=False)),
                        ("knn", KNeighborsClassifier(n_neighbors=5, weights=weights)),
                    ]).fit(model.training_coordinates_, self.y)
                    raw_query = model.transform(self.query)
                    labels, probabilities = model.predict_with_probabilities(self.query)
                    np.testing.assert_array_equal(labels, reference.predict(raw_query))
                    np.testing.assert_allclose(probabilities, reference.predict_proba(raw_query),
                                               rtol=1e-12, atol=1e-12)
                    self.assertEqual(model.knn_normalization_, "sd")
                    self.assertFalse(model.classifier_.named_steps["scale"].with_mean)
                    np.testing.assert_array_equal(
                        model.classifier_.named_steps["knn"]._fit_X,
                        reference.named_steps["knn"]._fit_X)

    def test_sd_does_not_change_raw_representation_or_refit_on_queries(self):
        ordinary = self.model().fit(self.X, self.y)
        scaled = self.model(knn_normalization="sd").fit(self.X, self.y)
        self.assertIsInstance(ordinary.classifier_, KNeighborsClassifier)
        self.assertEqual(ordinary.knn_normalization_, "none")
        np.testing.assert_array_equal(ordinary.training_coordinates_, scaled.training_coordinates_)
        np.testing.assert_array_equal(ordinary.components_, scaled.components_)
        np.testing.assert_array_equal(ordinary.transform(self.query), scaled.transform(self.query))
        scaler = scaled.classifier_.named_steps["scale"]
        before = [scaler.mean_.copy(), scaler.var_.copy(), scaler.scale_.copy()]
        with patch.object(scaler, "fit", side_effect=AssertionError("query must not fit a scaler")), \
                patch.object(scaler, "transform", wraps=scaler.transform) as scaling, \
                patch.object(scaled.decomposition_, "transform",
                             wraps=scaled.decomposition_.transform) as projection:
            labels, probabilities = scaled.predict_with_probabilities(self.query * 17)
            self.assertEqual(scaling.call_count, 1)
            self.assertEqual(projection.call_count, 1)
        np.testing.assert_array_equal(labels, scaled.classes_[probabilities.argmax(axis=1)])
        for expected, actual in zip(before, [scaler.mean_, scaler.var_, scaler.scale_]):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(scaler.n_samples_seen_, len(self.y))

    def test_sd_constant_coordinates_and_diagonal_rescaling_contract(self):
        # Head-only invariants use an explicit fixed raw coordinate map.
        train = np.array([[1., 1., 3.], [2., 1., 6.], [4., 1., 2.], [5., 1., 8.]])
        query = np.array([[1.5, 1., 4.], [4.5, 1., 6.]])
        labels = np.array(["a", "a", "z", "z"])
        model = self.model(knn_normalization="sd").fit(self.X, self.y)
        head = model.classifier_
        head.set_params(knn__n_neighbors=3).fit(train, labels)
        self.assertEqual(head.named_steps["scale"].scale_[1], 1.)
        np.testing.assert_array_equal(head.named_steps["knn"]._fit_X[:, 1], np.ones(4))
        # Nonconstant components cancel a positive factor gauge; the unchanged
        # constant column contributes zero pairwise distance in both models.
        diagonal = np.array([8., 1., .25])
        rescaled = clone(head).fit(train * diagonal, labels)
        np.testing.assert_array_equal(head.predict(query), rescaled.predict(query * diagonal))
        np.testing.assert_allclose(head.predict_proba(query),
                                   rescaled.predict_proba(query * diagonal), atol=1e-15)

    def test_sd_scaler_fits_only_training_fold_coordinates(self):
        seen = []
        original_fit = StandardScaler.fit

        def capture(instance, X, y=None, **kwargs):
            seen.append(np.array(X, copy=True))
            return original_fit(instance, X, y, **kwargs)

        with patch.object(StandardScaler, "fit", capture):
            search = calibration_search(self.model(knn_normalization="sd"),
                                        {"n_neighbors": [3]}, self.X, self.y,
                                        cv=3, seed=7, refit=False)
        self.assertEqual(len(seen), 3)
        for rows, fold in zip(seen, search.report["folds"]):
            training = fold["training_indices"]
            reference = self.model(knn_normalization="sd").fit(self.X[training], self.y[training])
            np.testing.assert_array_equal(rows, reference.training_coordinates_)
            self.assertLess(len(rows), len(self.X))

    def test_sd_threshold_ties_and_fitted_policy_survive_parameter_changes(self):
        model = self.model(knn_normalization="sd", n_neighbors=80,
                           decision_threshold=.5, positive_label="absent").fit(self.X, self.y)
        labels, probabilities = model.predict_with_probabilities(self.query)
        np.testing.assert_array_equal(probabilities, np.full((len(self.query), 2), .5))
        np.testing.assert_array_equal(labels, np.full(len(self.query), "present"))
        model.set_params(knn_normalization="none", classifier="linear_svm")
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertFalse(hasattr(model, "decision_function"))
        self.assertEqual(model.knn_normalization_, "sd")
        np.testing.assert_array_equal(model.predict(self.query), labels)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.knn_normalization_, "sd")
        np.testing.assert_array_equal(restored.predict(self.query), labels)
        self.assertEqual(clone(restored).get_params(), restored.get_params())

    def test_logistic_ignores_knn_metric_and_scales_only_once(self):
        ordinary = self.model(classifier="logistic").fit(self.X, self.y)
        unused = self.model(classifier="logistic", knn_normalization="sd").fit(self.X, self.y)
        self.assertIsNone(unused.knn_normalization_)
        self.assertEqual(tuple(unused.classifier_.named_steps), ("scale", "logistic"))
        self.assertTrue(unused.classifier_.named_steps["scale"].with_mean)
        np.testing.assert_array_equal(ordinary.predict(self.query), unused.predict(self.query))
        np.testing.assert_array_equal(ordinary.predict_proba(self.query), unused.predict_proba(self.query))
        scaler = unused.classifier_.named_steps["scale"]
        with patch.object(scaler, "transform", wraps=scaler.transform) as scaling:
            unused.predict_with_probabilities(self.query)
            self.assertEqual(scaling.call_count, 1)

    def test_svm_matches_explicit_primal_reference_with_unscaled_coordinates(self):
        labels = np.where(self.y == "present", -7, 13)
        for solver in ("spa", "cd", "mu"):
            with self.subTest(solver=solver):
                model = self.model(classifier="linear_svm", solver=solver,
                                   svm_C=.3, svm_class_weight="balanced",
                                   positive_label=-7).fit(self.X, labels)
                reference = LinearSVC(C=.3, penalty="l2", loss="squared_hinge", dual=False,
                                      max_iter=20000, tol=1e-5, class_weight="balanced",
                                      random_state=11).fit(model.training_coordinates_, labels)
                query = model.transform(self.query)
                predicted, margin = model.predict_with_decision_function(self.query)
                np.testing.assert_array_equal(predicted, reference.predict(query))
                np.testing.assert_allclose(margin, reference.decision_function(query),
                                           rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(predicted, model.classes_[(margin > 0).astype(int)])
                self.assertIsInstance(model.classifier_, LinearSVC)
                self.assertFalse(model.classifier_.dual)
                self.assertIsNone(model.knn_normalization_)
                self.assertEqual(model.classifier_n_iter_, reference.n_iter_)
                self.assertTrue(model.classifier_converged_)
                self.assertFalse(hasattr(model, "predict_proba"))
                self.assertFalse(hasattr(model, "predict_with_probabilities"))

    def test_svm_zero_tie_margin_polarity_and_single_projection(self):
        model = self.model(classifier="linear_svm", positive_label="absent").fit(self.X, self.y)
        model.classifier_.coef_[:] = 0
        for intercept, expected in ((0., "absent"), (1e-20, "present"), (-1e-20, "absent")):
            with self.subTest(intercept=intercept):
                model.classifier_.intercept_[:] = intercept
                with patch.object(model.decomposition_, "transform",
                                  wraps=model.decomposition_.transform) as projection:
                    labels, margin = model.predict_with_decision_function(self.query)
                    self.assertEqual(projection.call_count, 1)
                np.testing.assert_array_equal(labels, np.full(len(self.query), expected))
                np.testing.assert_array_equal(margin, np.full(len(self.query), intercept))
                np.testing.assert_array_equal(labels, model.predict(self.query))
                np.testing.assert_array_equal(margin, model.decision_function(self.query))

    def test_probability_capability_and_margin_capability_follow_fitted_head(self):
        model = self.model(classifier="linear_svm")
        self.assertFalse(hasattr(model, "predict_proba"))
        self.assertFalse(hasattr(model, "predict_with_probabilities"))
        with self.assertRaises(NotFittedError):
            model.decision_function(self.query)
        model.fit(self.X, self.y)
        expected = model.predict(self.query)
        model.set_params(classifier="knn")
        self.assertFalse(hasattr(model, "predict_proba"))
        self.assertTrue(hasattr(model, "decision_function"))
        np.testing.assert_array_equal(model.predict(self.query), expected)
        model.fit(self.X, self.y)
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertFalse(hasattr(model, "decision_function"))
        self.assertFalse(hasattr(model, "predict_with_decision_function"))
        for classifier in ("knn", "logistic", "linear_svm"):
            for normalization in ("none", "sd"):
                original = self.model(classifier=classifier, knn_normalization=normalization)
                self.assertEqual(clone(original).get_params(), original.get_params())

    def test_combined_logistic_margin_keeps_explicit_probability_policy(self):
        model = self.model(classifier="logistic", decision_threshold=.37,
                           positive_label="absent").fit(self.X, self.y)
        scaler = model.classifier_.named_steps["scale"]
        with patch.object(model.decomposition_, "transform",
                          wraps=model.decomposition_.transform) as projection, \
                patch.object(scaler, "transform", wraps=scaler.transform) as scaling:
            labels, margin = model.predict_with_decision_function(self.query)
            self.assertEqual(projection.call_count, 1)
            self.assertEqual(scaling.call_count, 1)
        np.testing.assert_array_equal(labels, model.predict(self.query))
        np.testing.assert_array_equal(margin, model.decision_function(self.query))
        np.testing.assert_array_equal(
            labels, model.classes_[np.where(model.predict_proba(self.query)[:, 0] > .37, 0, 1)])

    def test_invalid_new_head_settings_fail_before_factorization(self):
        invalid = [
            {"knn_normalization": "l2"}, {"knn_normalization": None},
            {"knn_normalization": ["sd"]},
            *({"classifier": "linear_svm", "decision_threshold": value}
              for value in (0., .5, 1., np.nextafter(0., -np.inf))),
            *({"classifier": "linear_svm", key: value} for key, value in (
                ("svm_C", 0), ("svm_C", np.nan), ("svm_C", True),
                ("svm_tol", 0), ("svm_tol", np.inf), ("svm_tol", np.bool_(True)),
                ("svm_max_iter", 0), ("svm_max_iter", 1.5), ("svm_max_iter", True),
                ("svm_class_weight", "unsupported"), ("svm_class_weight", {"absent": -1}),
                ("svm_class_weight", {"present": np.nan}))),
        ]
        for settings in invalid:
            with self.subTest(settings=settings), patch.object(FeatureSPA, "fit_transform") as factorization:
                with self.assertRaises(ValueError):
                    self.model(**settings).fit(self.X, self.y)
                factorization.assert_not_called()
        # Inactive solver-specific knobs must not silently influence another head.
        self.model(svm_C=-1).fit(self.X, self.y)
        self.model(classifier="linear_svm", logistic_C=-1, n_neighbors=1000).fit(self.X, self.y)

    def test_svm_warnings_and_failed_refit_clear_state(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self.model(classifier="linear_svm", svm_max_iter=1).fit(self.X, self.y)
        self.assertTrue(any(issubclass(w.category, ConvergenceWarning) for w in caught))
        self.assertFalse(model.classifier_converged_)
        self.assertFalse(model.converged_)
        self.assertTrue(model.classifier_convergence_warnings_)
        self.assertEqual(model.classifier_n_iter_, 1)
        model.set_params(svm_max_iter=20000).fit(self.X, self.y)
        self.assertTrue(model.classifier_converged_)
        model.set_params(decision_threshold=.5)
        with self.assertRaisesRegex(ValueError, "requires probabilities"):
            model.fit(self.X, self.y)
        self.assertFalse(any(key.endswith("_") for key in vars(model)))
        with self.assertRaises(NotFittedError):
            model.predict(self.query)

    def test_new_head_pickle_and_legacy_missing_parameters(self):
        for settings in ({"classifier": "linear_svm"}, {"knn_normalization": "sd"},
                         {"classifier": "logistic"}, {}):
            with self.subTest(settings=settings):
                model = self.model(**settings).fit(self.X, self.y)
                expected = model.predict(self.query)
                restored = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(restored.predict(self.query), expected)
                self.assertEqual(restored.get_params(), model.get_params())
        # A genuine old head had neither SD nor SVM settings. Constructor
        # restoration must not refit its representation or change its policy.
        for classifier in ("knn", "logistic"):
            model = self.model(classifier=classifier).fit(self.X, self.y)
            expected = model.predict(self.query)
            for name in ("knn_normalization", "knn_normalization_", "svm_C", "svm_tol",
                         "svm_max_iter", "svm_class_weight"):
                delattr(model, name)
            restored = pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(restored.predict(self.query), expected)
            self.assertEqual(restored.knn_normalization, "none")
            self.assertEqual(restored.svm_max_iter, 20000)
            self.assertEqual(clone(restored).get_params(), restored.get_params())


if __name__ == "__main__":
    unittest.main()
