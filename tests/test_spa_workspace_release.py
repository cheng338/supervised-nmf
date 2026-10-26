"""Check SPA's temporary-storage lifetime at the basis-fitting boundary.

These tests observe arrays through numerical calls, keeping weak references
only. They do not depend on allocator/RSS behavior or inspect fit's locals.
Numerical equivalence and failed-refit behavior live in test_spa_blas.py.
"""

import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA
from supervised_nmf import decomposition


class SPAWorkspaceReleaseTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)

    def assert_workspaces_released(self, rank, dtype, order, normalize):
        rng = np.random.default_rng(914)
        coordinates = rng.uniform(.1, 1.1, (23, rank))
        mixing = np.column_stack([
            np.eye(rank), rng.uniform(.02, .15, (rank, 7))
        ])
        X = np.array(coordinates @ mixing, dtype=dtype, order=order)
        original_X = X.copy()
        references = {name: [] for name in (
            "normalized", "residual", "direction", "projection", "norms"
        )}
        original_divide = decomposition.np.divide
        original_norm = decomposition.np.linalg.norm
        original_dgemv = decomposition.dgemv
        original_fit_basis = decomposition._fit_basis
        basis_calls = 0

        def remember_storage(name, array):
            # Observing a transient view alone would pass while its full-size
            # backing array remained live. Follow ndarray bases to their owner.
            while isinstance(array.base, np.ndarray):
                array = array.base
            references[name].append(weakref.ref(array))

        def observed_divide(*args, **kwargs):
            result = original_divide(*args, **kwargs)
            if isinstance(result, np.ndarray) and result.shape == X.shape:
                remember_storage("normalized", result)
            return result

        def observed_norm(array, *args, **kwargs):
            result = original_norm(array, *args, **kwargs)
            if (isinstance(result, np.ndarray)
                    and np.shape(array) == X.shape
                    and result.shape == (X.shape[1],)):
                remember_storage("norms", result)
            return result

        def observed_dgemv(alpha, array, vector, *args, **kwargs):
            remember_storage("residual", array)
            remember_storage("direction", vector)
            result = original_dgemv(alpha, array, vector, *args, **kwargs)
            remember_storage("projection", result)
            return result

        def observed_fit_basis(design, target, *args, **kwargs):
            nonlocal basis_calls
            if basis_calls == 0:
                self.assertEqual(len(references["normalized"]), 1)
                for name in ("residual", "direction", "projection"):
                    self.assertEqual(len(references[name]), rank, name)
                self.assertEqual(len(references["norms"]), rank + 1)
                # Collection makes the test independent of immediate reference
                # counting, while live local/loop references still fail it.
                gc.collect()
                alive = [
                    name for name, values in references.items()
                    if any(reference() is not None for reference in values)
                ]
                self.assertEqual(alive, [], f"SPA workspaces still live: {alive}")
                self.assertEqual(design.shape, (X.shape[0], rank))
                self.assertTrue(np.isfinite(design).all())
            basis_calls += 1
            return original_fit_basis(design, target, *args, **kwargs)

        # Plain replacement functions are deliberate: a recording Mock would
        # hold the ndarray arguments alive and invalidate the lifetime check.
        with patch.object(decomposition.np, "divide", new=observed_divide), \
                patch.object(decomposition.np.linalg, "norm", new=observed_norm), \
                patch.object(decomposition, "dgemv", new=observed_dgemv), \
                patch.object(decomposition, "_fit_basis", new=observed_fit_basis):
            model = FeatureSPA(rank, normalize=normalize).fit(X)
        self.assertEqual(basis_calls, 1)
        np.testing.assert_array_equal(X, original_X)
        np.testing.assert_array_equal(model.fit_coordinates_, X[:, model.anchor_indices_])
        np.testing.assert_allclose(
            model.fit_coordinates_ @ model.components_, X, rtol=2e-6, atol=2e-7
        )

    def test_rank_one_releases_workspaces_without_previous_direction(self):
        for dtype in (np.float32, np.float64):
            for order in ("C", "F"):
                for normalize in (False, True):
                    with self.subTest(dtype=dtype, order=order, normalize=normalize):
                        self.assert_workspaces_released(1, dtype, order, normalize)

    def test_multiple_anchors_release_every_direction_and_backing_array(self):
        for dtype in (np.float32, np.float64):
            for order in ("C", "F"):
                for normalize in (False, True):
                    with self.subTest(dtype=dtype, order=order, normalize=normalize):
                        self.assert_workspaces_released(3, dtype, order, normalize)


if __name__ == "__main__":
    unittest.main()
