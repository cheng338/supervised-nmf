"""Report-only residual routing and arithmetic contracts. Author: Chang Cheng."""

import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from supervised_nmf import _residual


class ReportedResidualTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)

    def test_block_arithmetic_preserves_inputs_and_known_residual(self):
        rng = np.random.default_rng(733)
        W = rng.uniform(.1, .9, (301, 7))
        H = rng.uniform(.1, .9, (7, 39))
        # Exactly representable control with a known nonzero residual.
        W.fill(.5)
        H.fill(.5)
        X = np.full((301, 39), 2.)
        expected = .25 * np.sqrt(X.size)
        for order in ("C", "F"):
            arrays = [np.array(a, order=order) for a in (X, W, H)]
            before = [a.copy() for a in arrays]
            for array in arrays:
                array.setflags(write=False)
            actual = _residual._blocked_error(*arrays)
            self.assertAlmostEqual(actual, expected, places=12)
            for array, original in zip(arrays, before):
                np.testing.assert_array_equal(array, original)

    def test_small_and_float32_inputs_keep_original_expression(self):
        rng = np.random.default_rng(734)
        for dtype in (np.float32, np.float64):
            W = rng.random((31, 3)).astype(dtype)
            H = rng.random((3, 17)).astype(dtype)
            X = (W @ H).copy()
            with patch.object(_residual, "_blocked_error", side_effect=AssertionError("Blocked")):
                actual = _residual.reconstruction_error(X, W, H)
            self.assertEqual(actual, float(np.linalg.norm(X - W @ H)))

    def test_dispatch_respects_shape_layout_dtype_backend_and_native_budget(self):
        # Eligibility reads metadata only. Empty allocations do not run GEMM.
        X = np.empty((8192, 1024), dtype=np.float64)
        W = np.empty((8192, 32), dtype=np.float64)
        H = np.empty((32, 1024), dtype=np.float64)
        with patch.object(_residual, "_native_controller") as controller:
            mkl = {"user_api": "blas", "internal_api": "mkl", "num_threads": 1}
            controller.return_value.info.return_value = [mkl]
            self.assertTrue(_residual._use_blocks(X, W, H))
            self.assertTrue(_residual._use_blocks(np.empty(X.shape, order="F"), W, H))
            self.assertFalse(_residual._use_blocks(X[:-1], W[:-1], H))
            self.assertFalse(_residual._use_blocks(X[:, ::2], W, H[:, ::2]))
            self.assertFalse(_residual._use_blocks(np.empty(X.shape, dtype=np.float32), W, H))
            self.assertFalse(_residual._use_blocks(X, np.empty((8192, 33)), np.empty((33, 1024))))
            cases = [
                ("mkl-one", [mkl], True),
                ("mkl-twenty", [dict(mkl, num_threads=20)], False),
                ("two-mkl-one", [mkl, mkl.copy()], True),
                ("mixed-mkl-thread-budgets", [mkl, dict(mkl, num_threads=20)], False),
                ("aocl", [dict(mkl, internal_api="blis", prefix="aocl-libblis")], False),
                ("blis", [dict(mkl, internal_api="blis")], False),
                ("openblas", [dict(mkl, internal_api="openblas")], False),
                ("mixed-backends", [mkl, dict(mkl, internal_api="openblas")], False),
                ("unknown-backend", [dict(mkl, internal_api="unknown")], False),
                ("missing-backend", [{"user_api": "blas", "num_threads": 1}], False),
                ("mixed-known-unknown", [mkl, {"user_api": "blas", "num_threads": 1}], False),
                ("missing-budget", [{"user_api": "blas", "internal_api": "mkl"}], False),
                ("no-libraries", [], False),
                ("openmp-only", [{"user_api": "openmp", "num_threads": 1}], False),
                ("mkl-and-openmp", [mkl, {"user_api": "openmp", "num_threads": 20}], True),
            ]
            for name, pools, expected in cases:
                with self.subTest(backend=name):
                    controller.return_value.info.return_value = pools
                    self.assertEqual(_residual._use_blocks(X, W, H), expected)


if __name__ == "__main__":
    unittest.main()
