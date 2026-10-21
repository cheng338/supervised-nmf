"""Represented-input oracles for Chang Cheng's screening and NNLS repairs."""

from decimal import Decimal, localcontext
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from supervised_nmf import _nnls
from supervised_nmf.screening import BinaryScreening


def decimal_t(column, labels):
    """Pooled t on the exact binary values supplied, not ideal pre-rounded data."""
    with localcontext() as context:
        context.prec = 400
        groups = [[Decimal.from_float(float(v)) for v in column[labels == label]]
                  for label in (0, 1)]
        means = [sum(group) / len(group) for group in groups]
        variances = [sum((v - mean)**2 for v in group) / (len(group) - 1)
                     for group, mean in zip(groups, means)]
        pooled = sum((len(group) - 1) * variance
                     for group, variance in zip(groups, variances)) / (len(column) - 2)
        difference = means[1] - means[0]
        if pooled == 0:
            return 0.0 if difference == 0 else float(Decimal("Infinity").copy_sign(difference))
        error = (pooled * (Decimal(1) / len(groups[0]) + Decimal(1) / len(groups[1]))).sqrt()
        return float(difference / error)


class NumericalRangeRepairs(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)

    def test_screening_repairs_exact_represented_offset_and_cutoff(self):
        labels = np.repeat([0, 1], 24)
        for offset, step in ((1e16, 2.0), (2.0**500, np.spacing(2.0**500))):
            signal = offset + np.arange(48) * step
            X = np.column_stack([signal, signal + np.arange(48) * step, labels])
            expected = np.array([decimal_t(X[:, j], labels) for j in range(2)])
            self.assertTrue((expected < 12).all())
            for layout in ("C", "F"):
                values = np.array(X, order=layout)
                before = values.copy()
                values.setflags(write=False)
                fitted = BinaryScreening(screening="absolute", score_threshold=12).fit(values, labels)
                np.testing.assert_allclose(fitted.t_scores_[:2], expected, rtol=1e-12)
                np.testing.assert_array_equal(fitted.support_, [False, False, True])
                np.testing.assert_array_equal(fitted.transform(values), values[:, [2]])
                np.testing.assert_array_equal(values, before)
                np.testing.assert_array_equal(fitted.statistic_offsets_, values.min(axis=0))

    def test_screening_permutation_constants_extremes_and_dtype(self):
        labels = np.repeat([0, 1], 24)
        for dtype, offset, step in ((np.float32, 2.0**20, .125),
                                     (np.float64, 2.0**500, np.spacing(2.0**500))):
            X = np.column_stack([offset + np.arange(48) * step, np.ones(48), labels]).astype(dtype)
            fitted = BinaryScreening(screening="absolute", score_threshold=0).fit(X, labels)
            order = np.random.default_rng(730).permutation(48)
            permuted = BinaryScreening(screening="absolute", score_threshold=0).fit(X[order], labels[order])
            np.testing.assert_allclose(fitted.t_scores_, permuted.t_scores_, rtol=1e-12)
            np.testing.assert_array_equal(fitted.support_, [True, False, True])
            self.assertEqual(fitted.transform(X).dtype, dtype)
        maximum = np.finfo(np.float64).max
        X = np.array([[maximum / 8], [maximum / 4], [maximum / 2], [maximum]])
        labels = np.array([0, 0, 1, 1])
        fitted = BinaryScreening(screening="absolute", score_threshold=0).fit(X, labels)
        self.assertAlmostEqual(fitted.t_scores_[0], decimal_t(X[:, 0], labels), places=12)

    def test_screening_underflow_remains_visible_and_clears_prior_fit(self):
        model = BinaryScreening(screening="absolute", score_threshold=0)
        labels = np.array([0, 0, 1, 1])
        model.fit(np.array([[1.], [2.], [3.], [4.]]), labels)
        with self.assertRaisesRegex(ValueError, "underflow"):
            model.fit(np.array([[0.], [1e-200], [1.], [1.]]), labels)
        self.assertFalse(hasattr(model, "support_"))

    def test_common_binary_scale_recovers_known_coefficients_and_decisions(self):
        design = np.r_[np.eye(3), np.zeros((2, 3))]
        truth = np.array([[0., .25, 2.], [1., 0., .5], [.5, 2., 0.]])
        truth = np.tile(truth, (15, 1))
        for exponent in (-1000, -700, 700, 1000):
            A = np.ldexp(design, exponent)
            X = np.ldexp(truth @ design.T, exponent)
            before = (A.copy(), X.copy())
            A.setflags(write=False)
            X.setflags(write=False)
            for solver in ("auto", "scipy", "batched"):
                with self.subTest(exponent=exponent, solver=solver):
                    actual = _nnls.project_nnls(A, X, solver=solver)
                    np.testing.assert_allclose(actual, truth, rtol=1e-12, atol=0)
                    np.testing.assert_array_equal(actual[:, 0] > .75, truth[:, 0] > .75)
            np.testing.assert_array_equal(A, before[0])
            np.testing.assert_array_equal(X, before[1])

    def test_mixed_query_scales_are_rowwise_and_batch_order_invariant(self):
        A = np.eye(2)
        X = np.vstack([np.full((17, 2), np.ldexp(1., -700)),
                       np.full((17, 2), np.ldexp(1., 700)), np.zeros((2, 2))])
        order = np.random.default_rng(731).permutation(len(X))
        for solver in ("auto", "scipy", "batched"):
            together = _nnls.project_nnls(A, X, solver=solver)
            separately = np.vstack([_nnls.project_nnls(A, row[None], solver=solver) for row in X])
            permuted = _nnls.project_nnls(A, X[order], solver=solver)
            np.testing.assert_allclose(together, X, rtol=1e-12, atol=0)
            np.testing.assert_allclose(separately, X, rtol=1e-12, atol=0)
            np.testing.assert_allclose(permuted, X[order], rtol=1e-12, atol=0)

    def test_unsupported_span_fails_before_any_group_solve(self):
        X = np.array([[1., 2.], [np.ldexp(1., -700), np.ldexp(1., 700)]])
        with patch.object(_nnls, "_project_unscaled") as solve:
            with self.assertRaisesRegex(FloatingPointError, "not.*unsolvable"):
                _nnls.project_nnls(np.eye(2), X)
            solve.assert_not_called()

    def test_ordinary_scipy_is_unchanged_and_scaled_low_cap_still_fails(self):
        rng = np.random.default_rng(23)
        A = rng.uniform(0, 2, (37, 7))
        X = rng.uniform(0, 3, (83, 37))
        expected = np.array([nnls(A, row)[0] for row in X])
        with patch.object(_nnls, "_prepare_scale_groups", side_effect=AssertionError("Ordinary grouping")):
            actual = _nnls.project_nnls(A, X, solver="scipy")
        np.testing.assert_array_equal(actual, expected)
        for exponent in (0, 700, -700):
            with self.subTest(exponent=exponent), self.assertRaises(RuntimeError):
                _nnls.project_nnls(np.ldexp(A, exponent), np.ldexp(X, exponent), maxiter=1)

    def test_scaled_signed_design_uses_public_problem(self):
        A = np.array([[1., -.25], [.25, 1.], [-.5, .125]])
        truth = np.tile([[.25, 2.], [1., .5]], (20, 1))
        X = truth @ A.T
        actual = _nnls.project_nnls(np.ldexp(A, 700), np.ldexp(X, 700), solver="batched")
        np.testing.assert_allclose(actual, truth, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
