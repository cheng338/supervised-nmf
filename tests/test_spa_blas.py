"""Numerical regression checks for SPA's direct-BLAS/layout optimization.

The reference below retains the original NumPy rank-one projection expression
and public SciPy NNLS calls. Stable fixtures must select identical anchors;
nearly tied choices are checked through their reconstruction as well. Float64
roundoff tolerances are 2e-11 for ordinary systems, and 2e-7 for coefficients
in the deliberately ill-conditioned fixture (condition number about 1e5).
Neither tolerance relaxes the production solver or its qualification rules.
"""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import nnls
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier


def original_numpy_reference(X, rank, *, normalize=True, maxiter=500):
    """Independent pre-optimization anchor/projection path on valid fixtures."""
    data = np.asarray(X, dtype=np.float64)
    scales = data.sum(axis=0) if normalize else np.ones(data.shape[1])
    normalized = np.divide(data, scales, out=np.zeros_like(data), where=scales > 0)
    residual = normalized.copy()
    anchors, directions, relative_gaps = [], [], []
    for _ in range(rank):
        lengths = np.linalg.norm(residual, axis=0)
        lengths[anchors] = -1
        anchor = int(np.argmax(lengths))
        ordered = np.sort(lengths)
        relative_gaps.append(float((ordered[-1] - ordered[-2]) / ordered[-1]))
        direction = residual[:, anchor] / lengths[anchor]
        for previous in directions:
            direction -= previous * np.dot(previous, direction)
        direction /= np.linalg.norm(direction)
        anchors.append(anchor)
        directions.append(direction)
        residual -= np.outer(direction, direction @ residual)
    coordinates = data[:, anchors].copy()
    components = np.column_stack([
        nnls(coordinates, data[:, column], maxiter=maxiter)[0]
        for column in range(data.shape[1])
    ])
    return np.asarray(anchors), coordinates, components, scales, relative_gaps


def original_nnls_transform(components, X, maxiter=500):
    return np.vstack([
        nnls(components.T, row, maxiter=maxiter)[0]
        for row in np.asarray(X, dtype=np.float64)
    ])


class SPABlasEquivalenceTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)

    @staticmethod
    def separable(seed=813):
        rng = np.random.default_rng(seed)
        coordinates = rng.uniform(.1, 1.4, (47, 3))
        basis = np.column_stack([np.diag([1.0, 1.8, .7]), rng.uniform(.05, .3, (3, 14))])
        incoming = rng.uniform(.1, 1.4, (13, 3)) @ basis
        return coordinates @ basis, incoming

    def assert_reference_fit(self, X, incoming, rank, *, normalize=True, rtol=2e-11):
        indices, coordinates, components, scales, gaps = original_numpy_reference(
            X, rank, normalize=normalize)
        model = FeatureSPA(rank, normalize=normalize, nnls_max_iter=500)
        fitted = model.fit_transform(X)
        np.testing.assert_array_equal(model.anchor_indices_, indices)
        np.testing.assert_array_equal(fitted, coordinates)
        np.testing.assert_allclose(model.column_scales_, scales, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(model.components_, components, rtol=rtol, atol=rtol)
        expected = original_nnls_transform(components, incoming)
        transformed = model.transform(incoming)
        np.testing.assert_allclose(transformed, expected, rtol=rtol, atol=rtol)
        np.testing.assert_allclose(model.inverse_transform(transformed), expected @ components,
                                   rtol=rtol, atol=rtol)
        self.assertAlmostEqual(model.reconstruction_err_,
                               float(np.linalg.norm(np.asarray(X) - coordinates @ components)), places=10)
        self.assertEqual(fitted.dtype, np.float64)
        self.assertEqual(model.components_.dtype, np.float64)
        self.assertEqual(transformed.dtype, np.float64)
        return model, gaps

    def test_random_nonnegative_matches_original_kernel_with_stable_anchors(self):
        for seed in (4, 17, 103):
            rng = np.random.default_rng(seed)
            X = rng.gamma(1.5, 1.0, (41, 23))
            incoming = rng.gamma(1.5, 1.0, (9, 23))
            for normalize in (False, True):
                for dtype in (np.float32, np.float64):
                    with self.subTest(seed=seed, normalize=normalize, dtype=dtype):
                        _, gaps = self.assert_reference_fit(X.astype(dtype), incoming.astype(dtype),
                                                            4, normalize=normalize)
                        # The fixture explicitly separates anchor maxima from
                        # rounding-scale ties; identity is meaningful here.
                        self.assertGreater(min(gaps), 1e-6)

    def test_separable_reconstruction_and_zero_columns_match_original(self):
        X, incoming = self.separable()
        X = np.column_stack([np.zeros(len(X)), X, np.zeros(len(X))])
        incoming = np.column_stack([np.zeros(len(incoming)), incoming, np.zeros(len(incoming))])
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                model, _ = self.assert_reference_fit(X, incoming, 3, normalize=normalize)
                self.assertEqual(set(model.anchor_indices_), {1, 2, 3})
                np.testing.assert_allclose(model.fit_coordinates_ @ model.components_, X,
                                           rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(model.inverse_transform(model.transform(incoming)), incoming,
                                           rtol=2e-11, atol=2e-11)
                np.testing.assert_array_equal(model.components_[:, [0, -1]], np.zeros((3, 2)))

    def test_exact_and_near_ties_preserve_reconstruction(self):
        # Distinct orthogonal rays with tied/near-tied norms. Mixture columns
        # make the fitted representation test nontrivial without ambiguity in
        # the span if the order of rounding-scale tied anchors changes.
        for delta in (0.0, 8 * np.finfo(float).eps, 1e-10):
            for normalize in (False, True):
                with self.subTest(delta=delta, normalize=normalize):
                    rays = np.zeros((9, 2))
                    rays[0, 0], rays[1, 1] = 1., 1. + delta
                    weights = np.array([[1., 0., .3, .15, 0.], [0., 1., .2, .4, 0.]])
                    X = rays @ weights
                    ref_indices, ref_W, ref_H, _, _ = original_numpy_reference(X, 2, normalize=normalize)
                    model = FeatureSPA(2, normalize=normalize).fit(X)
                    self.assertEqual(set(model.anchor_indices_), {0, 1})
                    if delta == 0:
                        # Exact equality retains np.argmax's first-index rule.
                        np.testing.assert_array_equal(model.anchor_indices_, ref_indices)
                    np.testing.assert_allclose(model.fit_coordinates_ @ model.components_, ref_W @ ref_H,
                                               rtol=2e-11, atol=2e-11)
                    incoming = np.array([[.2, .7], [.9, .1], [0., 0.]]) @ weights
                    np.testing.assert_allclose(model.inverse_transform(model.transform(incoming)), incoming,
                                               rtol=2e-11, atol=2e-11)

    def test_ill_conditioned_allowed_fixture_and_condition_rejection(self):
        rng = np.random.default_rng(123)
        first = rng.uniform(.8, 1.2, 39)
        rays = np.column_stack([first, first + 1e-4 * rng.uniform(0., 1., 39)])
        mixing = np.array([[1., 0., .25, .7, .1], [0., 1., .6, .2, .3]])
        X = rays @ mixing
        incoming = rng.uniform(.2, 1., (6, 2)) @ mixing
        model, _ = self.assert_reference_fit(X, incoming, 2, rtol=2e-7)
        self.assertGreater(model.original_anchor_condition_, 1e4)
        np.testing.assert_allclose(model.fit_coordinates_ @ model.components_, X,
                                   rtol=2e-9, atol=2e-9)
        with self.assertRaisesRegex(ValueError, "condition_limit"):
            FeatureSPA(2, condition_limit=100).fit(X)
        rank_one = np.outer(np.arange(1., 9.), np.arange(1., 6.))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            FeatureSPA(2).fit(rank_one)

    def test_c_f_readonly_and_strided_inputs_do_not_mutate_data_or_model(self):
        X, incoming = self.separable(14)
        for order in ("C", "F"):
            for readonly in (False, True):
                for dtype in (np.float32, np.float64):
                    with self.subTest(order=order, readonly=readonly, dtype=dtype):
                        fit_input = np.array(X, order=order, dtype=dtype)
                        fit_input.setflags(write=not readonly)
                        before_fit = fit_input.copy()
                        model, _ = self.assert_reference_fit(fit_input, incoming, 3)
                        np.testing.assert_array_equal(fit_input, before_fit)
                        self.assertEqual(fit_input.flags.writeable, not readonly)
                        # Stress the fitted basis too: transform must not let
                        # an overwrite-capable numerical routine change it.
                        model.components_ = np.array(model.components_, order=order)
                        model.components_.setflags(write=not readonly)
                        state = {name: value.copy() for name, value in vars(model).items()
                                 if isinstance(value, np.ndarray)}
                        buffer = np.empty((len(incoming), incoming.shape[1] * 2), dtype=dtype)
                        buffer[:, ::2], buffer[:, 1::2] = incoming, 17
                        view = buffer[:, ::2]
                        view.setflags(write=not readonly)
                        before_buffer = buffer.copy()
                        actual = model.transform(view)
                        expected = original_nnls_transform(state["components_"], view)
                        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
                        np.testing.assert_array_equal(buffer, before_buffer)
                        for name, original in state.items():
                            np.testing.assert_array_equal(getattr(model, name), original, err_msg=name)
                        self.assertEqual(model.components_.flags.writeable, not readonly)
                        self.assertEqual(view.flags.writeable, not readonly)

    def test_nnls_iteration_cap_is_forwarded_and_failures_remain_visible(self):
        X, incoming = self.separable(67)
        with patch("supervised_nmf.decomposition.nnls", wraps=nnls) as observed:
            model = FeatureSPA(3, nnls_max_iter=37).fit(X)
            self.assertEqual(observed.call_count, X.shape[1])
            self.assertTrue(all(call.kwargs["maxiter"] == 37 for call in observed.call_args_list))
            observed.reset_mock()
            model.transform(incoming)
            self.assertEqual(observed.call_count, len(incoming))
            self.assertTrue(all(call.kwargs["maxiter"] == 37 for call in observed.call_args_list))
        # Genuine public-NNLS exhaustion: the mixture needs three active
        # coefficients. This checks error propagation without a mocked solver.
        mixture = np.column_stack([np.eye(3), np.ones(3)])
        with self.assertRaisesRegex(RuntimeError, "iterations"):
            FeatureSPA(3, nnls_max_iter=1).fit(mixture)
        model = FeatureSPA(3, nnls_max_iter=20).fit(np.eye(3))
        original = model.components_.copy()
        model.set_params(nnls_max_iter=1)
        with self.assertRaisesRegex(RuntimeError, "iterations"):
            model.transform(np.ones((1, 3)))
        np.testing.assert_array_equal(model.components_, original)
        with self.assertRaisesRegex(RuntimeError, "iterations"):
            model.fit(mixture)
        with self.assertRaises(NotFittedError):
            model.transform(np.ones((1, 4)))

    def test_knn_and_logistic_predictions_match_independent_original_pipeline(self):
        rng = np.random.default_rng(819)
        y = np.where(np.arange(84) % 2, "present", "absent")
        coefficients = rng.uniform(.1, .8, (84, 3))
        coefficients[:, 0] += (y == "present") * .85
        basis = np.column_stack([np.diag([1., 1.5, .7]), rng.uniform(.1, .4, (3, 13))])
        X = (coefficients @ basis + rng.uniform(0, .001, (84, 16))).astype(np.float32)
        incoming = (rng.uniform(.1, 1.7, (17, 3)) @ basis).astype(np.float32)
        screen = BinaryScreening("absolute", score_threshold=0).fit(X, y)
        selected = screen.transform(X)
        _, reference_train, reference_basis, _, _ = original_numpy_reference(selected, 3)
        reference_test = original_nnls_transform(reference_basis, screen.transform(incoming))
        for coordinates in ("fit_transform", "transform"):
            W = (reference_train if coordinates == "fit_transform"
                 else original_nnls_transform(reference_basis, selected))
            for classifier in ("knn", "logistic"):
                with self.subTest(coordinates=coordinates, classifier=classifier):
                    model = NMFClassifier(n_components=3, solver="spa", screening="absolute", score_threshold=0,
                                          max_iter=500, training_coordinates=coordinates, classifier=classifier,
                                          n_neighbors=5, logistic_C=.3, logistic_tol=1e-8,
                                          logistic_max_iter=2000, random_state=4).fit(X, y)
                    if classifier == "knn":
                        downstream = KNeighborsClassifier(n_neighbors=5).fit(W, y)
                    else:
                        downstream = Pipeline([
                            ("scale", StandardScaler()),
                            ("logistic", LogisticRegression(C=.3, solver="lbfgs", tol=1e-8,
                                                             max_iter=2000, random_state=4)),
                        ]).fit(W, y)
                    predicted, probabilities = model.predict_with_probabilities(incoming)
                    np.testing.assert_array_equal(predicted, downstream.predict(reference_test))
                    np.testing.assert_allclose(probabilities, downstream.predict_proba(reference_test),
                                               rtol=2e-8, atol=2e-9)
                    np.testing.assert_allclose(model.training_coordinates_, W, rtol=2e-11, atol=2e-11)
                    self.assertTrue(model.converged_)


if __name__ == "__main__":
    unittest.main()
