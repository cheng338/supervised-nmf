"""Numerical, fallback, and resource-bound regressions for SPA projection."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from supervised_nmf import _nnls


class BatchedNNLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def setUp(self):
        rng = np.random.default_rng(23)
        self.design = rng.uniform(0, 2, size=(37, 7))
        self.rows = rng.uniform(0, 3, size=(83, 37))

    def reference(self, design=None, rows=None, maxiter=None):
        design = self.design if design is None else design
        rows = self.rows if rows is None else rows
        return np.asarray([nnls(design, row, maxiter=maxiter)[0] for row in rows])

    def assert_solution(self, design, rows, actual):
        expected = self.reference(design, rows)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())
        residual = actual @ design.T - rows
        reference_residual = expected @ design.T - rows
        np.testing.assert_allclose(np.linalg.norm(residual, axis=1),
                                   np.linalg.norm(reference_residual, axis=1),
                                   rtol=1e-9, atol=1e-9)
        gradient = residual @ design
        projected = np.where((actual > 0) | (gradient < 0), gradient, 0)
        scale = np.maximum(np.max(np.abs(rows @ design), axis=1), np.finfo(float).tiny)
        self.assertTrue((np.max(np.abs(projected), axis=1) / scale < 1e-9).all())

    def test_batched_agrees_across_nonnegative_active_sets_and_shapes(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            rank, features = 1 + seed % 8, 25 + seed
            design = rng.gamma(1.3, 1, size=(features, rank))
            rows = rng.gamma(.6, 2, size=(39, features))
            with self.subTest(seed=seed):
                actual = _nnls.project_nnls(design, rows, solver="batched", batch_size=17)
                self.assert_solution(design, rows, actual)

    def test_common_case_uses_batch_and_never_mutates_even_read_only_views(self):
        design = np.asfortranarray(self.design)
        rows = self.rows[::-1]
        design_before, rows_before = design.copy(), rows.copy()
        design.setflags(write=False)
        rows.setflags(write=False)
        with patch.object(_nnls, "nnls", side_effect=AssertionError("Unexpected fallback")):
            actual = _nnls.project_nnls(design, rows)
        self.assert_solution(design, rows, actual)
        np.testing.assert_array_equal(design, design_before)
        np.testing.assert_array_equal(rows, rows_before)

    def test_small_auto_and_explicit_scipy_are_exact_reference(self):
        for solver, rows in (("auto", self.rows[:31]), ("scipy", self.rows)):
            with self.subTest(solver=solver), patch.object(
                    _nnls, "_batched_block", side_effect=AssertionError("Batch used")):
                actual = _nnls.project_nnls(self.design, rows, solver=solver)
                np.testing.assert_array_equal(actual, self.reference(rows=rows))

    def test_auto_starts_at_32_rows(self):
        with patch.object(_nnls, "_batched_block", wraps=_nnls._batched_block) as batch:
            _nnls.project_nnls(self.design, self.rows[:32])
            self.assertEqual(batch.call_count, 1)

    def test_exact_fit_zero_rows_zero_features_and_float32(self):
        rng = np.random.default_rng(231)
        design = np.r_[np.eye(5), np.zeros((3, 5))].astype(np.float32)
        coordinates = rng.uniform(size=(40, 5)).astype(np.float32)
        coordinates[::3] = 0
        rows = coordinates @ design.T
        actual = _nnls.project_nnls(design, rows, solver="batched")
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, coordinates)
        empty = _nnls.project_nnls(design, rows[:0])
        self.assertEqual(empty.shape, (0, 5))

    def test_global_and_moderate_column_scaling(self):
        for factor in (1e-8, 1.0, 1e8):
            for scales in (np.ones(7), np.geomspace(.1, 10, 7)):
                with self.subTest(factor=factor, scales=scales.tolist()):
                    design = self.design * scales * factor
                    rows = self.rows * factor
                    actual = _nnls.project_nnls(design, rows, solver="batched")
                    np.testing.assert_allclose(actual, self.reference(design, rows),
                                               rtol=1e-7, atol=1e-9)

    def test_rank_deficient_underdetermined_signed_and_ill_conditioned_use_scipy(self):
        rng = np.random.default_rng(311)
        cases = [
            np.column_stack([self.design[:, :2], self.design[:, 0], np.zeros(37)]),
            rng.uniform(size=(3, 7)),
            self.design - 1,
            self.design * np.geomspace(1e-8, 1e8, 7),
            np.diag([1.0, 1e-5]),
        ]
        for design in cases:
            rows = rng.uniform(size=(40, len(design)))
            with self.subTest(shape=design.shape), patch.object(
                    _nnls, "_batched_block", side_effect=AssertionError("Batch used")):
                actual = _nnls.project_nnls(design, rows, solver="batched")
                np.testing.assert_array_equal(actual, self.reference(design, rows))
        signed_rows = self.rows - 2
        with patch.object(_nnls, "_batched_block", side_effect=AssertionError("Batch used")):
            np.testing.assert_array_equal(
                _nnls.project_nnls(self.design, signed_rows, solver="batched"),
                self.reference(rows=signed_rows))

    def test_underflow_and_overflow_design_products_delegate_to_reference(self):
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale), patch.object(
                    _nnls, "nnls", return_value=(np.ones(7), 0)) as reference, patch.object(
                    _nnls, "_batched_block", side_effect=AssertionError("Batch used")):
                result = _nnls.project_nnls(self.design * scale, self.rows, solver="batched")
                self.assertEqual(reference.call_count, len(self.rows))
                np.testing.assert_array_equal(result, np.ones((len(self.rows), 7)))

    def test_batch_and_original_design_checks_are_bounded_by_block_size(self):
        with patch.object(_nnls, "_batched_block", wraps=_nnls._batched_block) as batch, \
                patch.object(_nnls, "_qualified_rows", wraps=_nnls._qualified_rows) as qualify:
            actual = _nnls.project_nnls(self.design, self.rows, solver="batched", batch_size=13)
        self.assert_solution(self.design, self.rows, actual)
        self.assertEqual([call.args[1].shape[1] for call in batch.call_args_list],
                         [13, 13, 13, 13, 13, 13, 5])
        self.assertEqual([call.args[1].shape[1] for call in qualify.call_args_list],
                         [13, 13, 13, 13, 13, 13, 5])

    def test_shared_passive_pattern_reuses_factor(self):
        gram = self.design.T @ self.design
        cross = self.design.T @ self.rows[:4].T
        passive = np.ones_like(cross, dtype=bool)
        factors = {}
        with patch.object(_nnls.linalg, "cho_factor", wraps=_nnls.linalg.cho_factor) as factor:
            first = _nnls._passive_solve(gram, cross, passive, factors)
            second = _nnls._passive_solve(gram, cross * 2, passive, factors)
        self.assertEqual(factor.call_count, 1)
        np.testing.assert_allclose(second, first * 2, rtol=1e-14, atol=1e-14)

    def test_forged_success_requires_finite_nonnegative_original_design_kkt(self):
        reference = self.reference()
        defects = [lambda x: x.__setitem__((0, 0), -1e-20),
                   lambda x: x.__setitem__((0, 0), np.nan),
                   lambda x: x.__setitem__((0, 0), np.inf),
                   lambda x: x.__setitem__((slice(None), 0), 0),
                   lambda x: x.__setitem__((slice(None), 0), 100)]
        for defect in defects:
            coefficients = reference.T.copy()
            defect(coefficients)
            with self.subTest(defect=defect), patch.object(
                    _nnls, "_batched_block", return_value=(coefficients, np.ones(83, bool))), \
                    patch.object(_nnls, "nnls", wraps=nnls) as fallback:
                actual = _nnls.project_nnls(self.design, self.rows, solver="batched")
                self.assertEqual(fallback.call_count, 1)
                np.testing.assert_array_equal(actual, reference)

    def test_nonconverged_rows_fallback_even_when_candidate_passes_kkt(self):
        expected = self.reference()
        converged = np.ones(83, bool)
        converged[[0, 7]] = False
        with patch.object(_nnls, "_batched_block", return_value=(expected.T, converged)), \
                patch.object(_nnls, "nnls", wraps=nnls) as fallback:
            actual = _nnls.project_nnls(self.design, self.rows, solver="batched", maxiter=100)
        self.assertEqual(fallback.call_count, 2)
        self.assertTrue(all(call.kwargs["maxiter"] == 100 for call in fallback.call_args_list))
        np.testing.assert_array_equal(actual, expected)

    def test_failed_block_falls_back_and_reference_failure_propagates(self):
        for failure in (np.linalg.LinAlgError("factor"), FloatingPointError("overflow")):
            with self.subTest(failure=failure), patch.object(
                    _nnls, "_batched_block", side_effect=failure):
                np.testing.assert_array_equal(
                    _nnls.project_nnls(self.design, self.rows, solver="batched"), self.reference())
        with patch.object(_nnls, "_batched_block", side_effect=np.linalg.LinAlgError()), \
                patch.object(_nnls, "nnls", side_effect=RuntimeError("Maximum number")):
            with self.assertRaisesRegex(RuntimeError, "Maximum number"):
                _nnls.project_nnls(self.design, self.rows, solver="batched")

    def test_small_iteration_cap_routes_to_reference_and_preserves_failure(self):
        with patch.object(_nnls, "_batched_block", side_effect=AssertionError("Batch used")), \
                patch.object(_nnls, "nnls", wraps=nnls) as reference:
            try:
                expected = self.reference(maxiter=1)
            except RuntimeError as error:
                with self.assertRaisesRegex(RuntimeError, str(error)):
                    _nnls.project_nnls(self.design, self.rows, solver="batched", maxiter=1)
            else:
                np.testing.assert_array_equal(
                    _nnls.project_nnls(self.design, self.rows, solver="batched", maxiter=1), expected)
            self.assertTrue(reference.called)
            self.assertTrue(all(call.kwargs["maxiter"] == 1 for call in reference.call_args_list))

    def test_default_and_explicit_batch_iteration_caps(self):
        original = _nnls._batched_block
        for cap, expected in ((None, 21), (21, 21), (33, 21), (20000, 21)):
            with self.subTest(cap=cap), patch.object(_nnls, "_batched_block", wraps=original) as batch:
                _nnls.project_nnls(self.design, self.rows, solver="batched", maxiter=cap)
                self.assertEqual(batch.call_args.args[3], expected)

    def test_invalid_inputs(self):
        for value in (0, -1, 1.5, True, np.bool_(False)):
            for name in ("batch_size", "maxiter"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    _nnls.project_nnls(self.design, self.rows, **{name: value})
        for arguments in ((self.design.ravel(), self.rows), (self.design, self.rows.ravel()),
                          (self.design, self.rows[:, :-1]), (np.empty((0, 7)), np.empty((3, 0))),
                          (np.empty((37, 0)), self.rows),
                          (self.design.astype(complex), self.rows),
                          (self.design, self.rows.astype(complex))):
            with self.subTest(shapes=[x.shape for x in arguments]), self.assertRaises(ValueError):
                _nnls.project_nnls(*arguments)
        for invalid in (np.nan, np.inf, -np.inf):
            design, rows = self.design.copy(), self.rows.copy()
            design[0, 0] = invalid
            rows[0, 0] = invalid
            for args in ((design, self.rows), (self.design, rows)):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                    _nnls.project_nnls(*args)
        with self.assertRaisesRegex(ValueError, "solver"):
            _nnls.project_nnls(self.design, self.rows, solver="fast")


if __name__ == "__main__":
    unittest.main()
