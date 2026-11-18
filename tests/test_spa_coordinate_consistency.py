"""SPA transformer coordinates use the same fixed-basis map for every input."""

import unittest

import numpy as np
from scipy.optimize import nnls
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA


class SPACoordinateConsistencyTests(unittest.TestCase):
    def setUp(self):
        limit = threadpool_limits(1)
        self.addCleanup(limit.restore_original_limits)
        rng = np.random.default_rng(817)
        basis = np.hstack([np.eye(3), rng.uniform(.02, .8, (3, 12))])
        coordinates = rng.uniform(.05, 2, (80, 3))
        self.y = ((coordinates[:, 0] + coordinates[:, 1]) > 2).astype(int)
        self.X = coordinates @ basis + rng.uniform(0, .25, (80, 15))
        self.query = rng.uniform(.05, 2, (19, 3)) @ basis + rng.uniform(0, .25, (19, 15))

    @staticmethod
    def projected(basis, X):
        return np.vstack([nnls(basis.T, row, maxiter=10000)[0] for row in X])

    def test_noisy_fit_transform_matches_independent_nnls_and_fit_then_transform(self):
        for solver in ("auto", "scipy"):
            with self.subTest(projection_solver=solver):
                model = FeatureSPA(3, nnls_max_iter=10000, projection_solver=solver)
                projected = model.fit_transform(self.X, self.y)
                expected = self.projected(model.components_, self.X)
                separately_fitted = FeatureSPA(3, nnls_max_iter=10000, projection_solver=solver).fit(self.X)
                np.testing.assert_allclose(projected, expected, rtol=2e-10, atol=2e-10)
                np.testing.assert_array_equal(projected, model.transform(self.X))
                np.testing.assert_array_equal(projected, separately_fitted.transform(self.X))
                self.assertGreater(np.linalg.norm(projected - model.fit_coordinates_), .001)
                self.assertLessEqual(
                    np.linalg.norm(self.X - projected @ model.components_),
                    model.reconstruction_err_ + 1e-10,
                )

    def test_projection_preserves_anchor_basis_objective_and_saved_factorization(self):
        reference = FeatureSPA(3, nnls_max_iter=10000).fit(self.X)
        expected_anchors = reference.anchor_indices_.copy()
        expected_coordinates = reference.fit_coordinates_.copy()
        expected_basis = reference.components_.copy()
        expected_error = reference.reconstruction_err_
        independent_basis = np.column_stack([
            nnls(expected_coordinates, self.X[:, index], maxiter=10000)[0]
            for index in range(self.X.shape[1])
        ])
        model = FeatureSPA(3, nnls_max_iter=10000)
        projected = model.fit_transform(self.X)
        np.testing.assert_array_equal(model.anchor_indices_, expected_anchors)
        np.testing.assert_array_equal(model.fit_coordinates_, self.X[:, expected_anchors])
        np.testing.assert_array_equal(model.components_, expected_basis)
        np.testing.assert_allclose(model.components_, independent_basis, rtol=2e-10, atol=2e-10)
        self.assertEqual(model.reconstruction_err_, expected_error)
        self.assertAlmostEqual(expected_error, np.linalg.norm(self.X - expected_coordinates @ expected_basis))
        model.transform(self.query)
        np.testing.assert_array_equal(model.fit_coordinates_, expected_coordinates)
        np.testing.assert_array_equal(model.components_, expected_basis)
        self.assertFalse(np.shares_memory(projected, model.fit_coordinates_))
        projected[:] = 0
        np.testing.assert_array_equal(model.fit_coordinates_, expected_coordinates)
        np.testing.assert_array_equal(model.components_, expected_basis)

    def test_generic_pipeline_trains_knn_on_fixed_basis_nnls_coordinates(self):
        pipeline = Pipeline([
            ("spa", FeatureSPA(3, nnls_max_iter=10000)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]).fit(self.X, self.y)
        spa = pipeline.named_steps["spa"]
        expected_train = self.projected(spa.components_, self.X)
        expected_query = self.projected(spa.components_, self.query)
        actual_train = pipeline.named_steps["knn"]._fit_X
        np.testing.assert_allclose(actual_train, expected_train, rtol=2e-10, atol=2e-10)
        self.assertGreater(np.linalg.norm(actual_train - spa.fit_coordinates_), .001)
        reference = KNeighborsClassifier(n_neighbors=5).fit(expected_train, self.y)
        np.testing.assert_array_equal(pipeline.predict(self.query), reference.predict(expected_query))
        np.testing.assert_allclose(pipeline.predict_proba(self.query), reference.predict_proba(expected_query))


if __name__ == "__main__":
    unittest.main()
