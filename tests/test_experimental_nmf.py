"""Independent small NNLS oracles and experimental ANLS-BPP API checks."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experimental_nmf import ANLSBPP, nnls_block_pivot, projected_gradient_kkt, _minimum_norm_lstsq
import experimental_nmf


class BlockPivotTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.addCleanup(self.threads.restore_original_limits)

    def oracle(self, A, B, **kwargs):
        X, info = nnls_block_pivot(A, B, **kwargs)
        self.assertTrue(info["success"], info)
        self.assertTrue(np.all(X >= 0))
        expected = np.column_stack([nnls(A, column, maxiter=10000)[0] for column in B.T])
        # Coefficients need not agree for rank-deficient designs.
        actual_error = np.linalg.norm(A @ X - B, axis=0)
        oracle_error = np.linalg.norm(A @ expected - B, axis=0)
        np.testing.assert_allclose(actual_error, oracle_error, rtol=1e-8, atol=1e-9)
        gradient = A.T @ (A @ X - B)
        scale = max(1.0, np.max(np.abs(A.T @ B)))
        self.assertGreaterEqual(gradient.min(), -1e-8 * scale)
        self.assertLessEqual(np.max(np.abs(X * gradient)), 1e-7 * scale * max(1.0, X.max()))
        return X, info

    def test_independent_oracle_many_signed_and_nonnegative_rhs(self):
        for seed in range(12):
            random = np.random.RandomState(seed)
            A, B = random.normal(size=(18, 7)), random.normal(size=(18, 9))
            with self.subTest(seed=seed):
                self.oracle(A, B)
                self.oracle(abs(A), abs(B))

    def test_rank_deficient_duplicate_and_zero_columns(self):
        random = np.random.RandomState(52)
        base = random.uniform(size=(16, 3))
        A = np.column_stack((base, base[:, 0], 2 * base[:, 1], np.zeros(16)))
        _, info = self.oracle(A, random.uniform(size=(16, 8)))
        self.assertGreater(info["direct_lstsq_groups"], 0)

    def test_nearly_dependent_columns_original_design_check(self):
        random = np.random.RandomState(12)
        base = random.normal(size=(18, 2))
        A = np.column_stack((base, base[:, 0] + 1e-8 * random.normal(size=18)))
        self.oracle(A, random.normal(size=(18, 5)))

    def test_underdetermined_nonunique_problem(self):
        random = np.random.RandomState(46)
        A = random.uniform(size=(4, 9))
        self.oracle(A, random.uniform(size=(4, 7)))

    def test_zero_design_zero_rhs_and_negative_rhs(self):
        self.oracle(np.zeros((5, 3)), np.ones((5, 2)))
        self.oracle(np.ones((5, 3)), np.zeros((5, 2)))
        X, _ = self.oracle(np.ones((5, 3)), -np.ones((5, 2)))
        np.testing.assert_array_equal(X, 0)

    def test_warm_start_and_scale_equivariance(self):
        random = np.random.RandomState(3)
        A, B = random.uniform(size=(15, 5)), random.uniform(size=(15, 4))
        X, _ = self.oracle(A, B)
        for scale in (1e-7, 1e7):
            actual, _ = self.oracle(A * scale, B * scale, init=X)
            np.testing.assert_allclose(actual, X, rtol=1e-7, atol=1e-9)

    def test_inner_cap_warns_and_does_not_claim_success(self):
        random = np.random.RandomState(10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, info = nnls_block_pivot(random.normal(size=(20, 9)),
                                      random.normal(size=(20, 8)), max_iter=1)
        self.assertFalse(info["success"])
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))

    def test_original_design_kkt_rejects_inaccurate_passive_solution(self):
        # A feasible positive answer can still have an unacceptable gradient.
        # Simulate a faulty numerical solve, and require the independent final
        # check to reject it even though no passive/nonpassive sign is wrong.
        actual_solve = experimental_nmf._passive_solve
        def inaccurate(*args):
            values, groups, direct = actual_solve(*args)
            return values * 1.01, groups, direct
        with patch.object(experimental_nmf, "_passive_solve", inaccurate):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, info = nnls_block_pivot(np.eye(3), np.ones((3, 2)))
        self.assertFalse(info["success"])
        self.assertGreater(info["projected_gradient_relative"], info["kkt_tolerance"])
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))

    def test_invalid_inputs(self):
        for arguments in ((np.ones((3, 2)), np.ones((4, 1))),
                          (np.array([[np.nan]]), np.ones((1, 1)))):
            with self.assertRaises(ValueError):
                nnls_block_pivot(*arguments)
        with self.assertRaises(ValueError):
            nnls_block_pivot(np.eye(2), np.ones((2, 1)), tol=0)

    def test_direct_gelsd_matches_lstsq_and_does_not_mutate(self):
        random = np.random.RandomState(71)
        for m, n, rhs in ((15, 4, 7), (4, 9, 3), (8, 5, 1)):
            A, B = random.normal(size=(m, n)), random.normal(size=(m, rhs))
            A[:, -1] = A[:, 0]
            old_A, old_B = A.copy(), B.copy()
            expected = linalg.lstsq(A, B, cond=np.finfo(float).eps, lapack_driver="gelsd")[0]
            # A mistaken reintroduction of SciPy's batched dispatcher fails.
            with patch.object(experimental_nmf.linalg, "lstsq", side_effect=AssertionError("batched dispatch forbidden")):
                actual = _minimum_norm_lstsq(A, B)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
            np.testing.assert_array_equal(A, old_A)
            np.testing.assert_array_equal(B, old_B)

    def test_direct_gelsd_checks_workspace_and_info(self):
        A, B = np.eye(2), np.ones((2, 1))
        for work, integer_work, info in ((10, 2, -1), (np.nan, 2, 0), (10, 0, 0)):
            with patch.object(experimental_nmf, "get_lapack_funcs", return_value=(None, lambda *a, **k: (work, integer_work, info))):
                with self.assertRaises(linalg.LinAlgError):
                    _minimum_norm_lstsq(A, B)
        for info in (-3, 1):
            def bad_solve(*args, **kwargs):
                self.assertEqual(kwargs["cond"], np.finfo(float).eps)
                return np.ones((2, 1)), np.ones(2), 2, info
            with patch.object(experimental_nmf, "get_lapack_funcs", return_value=(bad_solve, lambda *a, **k: (10.0, 2, 0))):
                with self.assertRaises(linalg.LinAlgError):
                    _minimum_norm_lstsq(A, B)
        def unused_solve(*args, **kwargs):
            raise AssertionError("Invalid workspace must be rejected before LAPACK")
        unused_solve.int_dtype = np.dtype("int32")
        with patch.object(experimental_nmf, "get_lapack_funcs", return_value=(unused_solve, lambda *a, **k: (2.0**32, 2, 0))):
            with self.assertRaises(linalg.LinAlgError):
                _minimum_norm_lstsq(A, B)


class ANLSBPPTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.addCleanup(self.threads.restore_original_limits)
        random = np.random.RandomState(56)
        self.X = (random.uniform(.1, 1, (35, 2)) @ random.uniform(.1, 1, (2, 11)))

    def test_monotone_objective_and_independent_transform_oracle(self):
        model = ANLSBPP(2, max_iter=1000, tol=1e-4, random_state=42)
        W = model.fit_transform(self.X.astype(np.float32))
        self.assertTrue(model.converged_, model.convergence_reason_)
        self.assertEqual(W.dtype, np.float64)
        self.assertTrue(np.all(np.diff(model.objective_history_) <= 1e-10))
        self.assertLessEqual(model.relative_projected_gradient_, model.tol)
        np.testing.assert_array_equal(W, model.training_coordinates_)
        H_before = model.components_.copy()
        probe = self.X[::4] * 1.1
        actual = model.transform(probe)
        expected = np.vstack([nnls(H_before.T, row)[0] for row in probe])
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
        np.testing.assert_array_equal(model.components_, H_before)
        self.assertTrue(model.last_transform_converged_)

    def test_pipeline_clone_and_global_random_state_unchanged(self):
        np.random.seed(491)
        before = np.random.get_state()
        prototype = Pipeline([("nmf", ANLSBPP(2, random_state=8)),
                              ("knn", KNeighborsClassifier(n_neighbors=3))])
        fitted = clone(prototype).fit(self.X, np.arange(len(self.X)) % 2)
        self.assertEqual(fitted.predict(self.X[:4]).shape, (4,))
        self.assertFalse(hasattr(prototype.named_steps["nmf"], "components_"))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_outer_cap_and_inner_failure_emit_warning_through_pipeline(self):
        random = np.random.RandomState(10)
        noisy = random.uniform(size=self.X.shape)
        for params in ({"max_iter": 1, "tol": 1e-12}, {"nnls_max_iter": 1}):
            model = Pipeline([("nmf", ANLSBPP(4, **params)),
                              ("knn", KNeighborsClassifier(n_neighbors=1))])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(noisy, np.arange(len(noisy)) % 2)
            self.assertFalse(model.named_steps["nmf"].converged_)
            self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))

    def test_zero_data_and_invalid_refit_clear_fitted_state(self):
        model = ANLSBPP(2).fit(np.zeros((6, 4)))
        self.assertTrue(model.converged_)
        np.testing.assert_array_equal(model.transform(np.zeros((3, 4))), 0)
        with self.assertRaises(ValueError):
            model.fit(-self.X)
        self.assertFalse(hasattr(model, "components_"))

    def test_transform_cap_emits_warning_and_preserves_basis(self):
        random = np.random.RandomState(10)
        model = ANLSBPP(4, max_iter=1000).fit(random.uniform(size=(35, 11)))
        model.set_params(nnls_max_iter=1)
        basis = model.components_.copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.transform(random.uniform(size=(40, 11)))
        self.assertFalse(model.last_transform_converged_)
        self.assertTrue(any(issubclass(item.category, ConvergenceWarning) for item in caught))
        np.testing.assert_array_equal(model.components_, basis)

    def test_canonical_diagnostic_invariance_without_mutation(self):
        random = np.random.RandomState(2)
        W, H = random.uniform(size=(35, 2)), random.uniform(size=(2, 11))
        before_W, before_H = W.copy(), H.copy()
        first = projected_gradient_kkt(self.X, W, H)
        scales = np.array([.01, 100.0])
        other = projected_gradient_kkt(self.X, W * scales, H / scales[:, None])
        for key in first["canonical"]:
            self.assertAlmostEqual(first["canonical"][key], other["canonical"][key], places=9)
        self.assertNotAlmostEqual(first["raw"]["projected_gradient_norm"],
                                  other["raw"]["projected_gradient_norm"])
        np.testing.assert_array_equal(W, before_W)
        np.testing.assert_array_equal(H, before_H)
        exact = projected_gradient_kkt(W @ H, W, H)
        self.assertEqual(exact["raw"]["projected_gradient_norm"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
