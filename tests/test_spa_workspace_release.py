"""Check SPA's selection workspaces, condition checks and callback behavior.

These tests observe arrays through numerical calls, keeping weak references
only. They do not depend on allocator/RSS behavior or inspect fit's locals.
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

    @staticmethod
    def separable(rank, dtype, order):
        rng = np.random.default_rng(914)
        coordinates = rng.uniform(.1, 1.1, (23, rank))
        mixing = np.column_stack([
            np.eye(rank), rng.uniform(.02, .15, (rank, 7))
        ])
        return np.array(coordinates @ mixing, dtype=dtype, order=order)

    def assert_workspaces_released(self, X, model, *, reuse_normalized=False):
        rank = model.n_components
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
            self.assertFalse(np.shares_memory(array, X))
            if model.normalize:
                normalized = references["normalized"][0]()
                self.assertIsNotNone(normalized)
                self.assertEqual(np.shares_memory(array, normalized), reuse_normalized)
            remember_storage("direction", vector)
            result = original_dgemv(alpha, array, vector, *args, **kwargs)
            remember_storage("projection", result)
            return result

        def observed_fit_basis(design, target, *args, **kwargs):
            nonlocal basis_calls
            if basis_calls == 0:
                self.assertEqual(len(references["normalized"]), int(model.normalize))
                for name in ("residual", "direction", "projection"):
                    self.assertEqual(len(references[name]), rank, name)
                self.assertEqual(len(references["norms"]), rank)
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
        with np.errstate(under="ignore"), \
                patch.object(decomposition.np, "divide", new=observed_divide), \
                patch.object(decomposition.np.linalg, "norm", new=observed_norm), \
                patch.object(decomposition, "dgemv", new=observed_dgemv), \
                patch.object(decomposition, "_fit_basis", new=observed_fit_basis):
            model.fit(X)
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
                        X = self.separable(1, dtype, order)
                        self.assert_workspaces_released(X, FeatureSPA(1, normalize=normalize))

    def test_multiple_anchors_release_every_direction_and_backing_array(self):
        for dtype in (np.float32, np.float64):
            for order in ("C", "F"):
                for normalize in (False, True):
                    with self.subTest(dtype=dtype, order=order, normalize=normalize):
                        X = self.separable(3, dtype, order)
                        self.assert_workspaces_released(X, FeatureSPA(3, normalize=normalize))

    def test_large_normalized_fit_reuses_residual_and_checks_both_anchor_scales(self):
        # A duplicate anchor ties at the first pivot. The other ray has four
        # equal entries, so its normalized L2 norm is exactly one half.
        for amplitude, original_condition in ((2., 1.), (8., 4.)):
            with self.subTest(amplitude=amplitude):
                owner = np.full((65, 130), -7.)
                X = owner[::-1, ::2]
                X[:] = 0.
                X[0, 0] = X[0, 2] = amplitude
                X[1:5, 1] = 1.
                X.setflags(write=False)
                before = owner.copy()
                limit = max(2., original_condition)

                model = FeatureSPA(2, condition_limit=limit)
                self.assert_workspaces_released(X, model, reuse_normalized=True)
                np.testing.assert_array_equal(model.anchor_indices_, [0, 1])
                self.assertEqual(model.anchor_condition_, 2.)
                self.assertEqual(model.original_anchor_condition_, original_condition)
                for value in vars(model).values():
                    if isinstance(value, np.ndarray):
                        self.assertFalse(np.shares_memory(value, owner))

                model.condition_limit = np.nextafter(limit, 0.)
                with np.errstate(under="ignore"), self.assertRaisesRegex(ValueError, "SPA anchors exceed"):
                    model.fit(X)
                self.assertFalse(any(key.endswith("_") for key in vars(model)))
                np.testing.assert_array_equal(owner, before)
                self.assertFalse(X.flags.writeable)

    def test_underflow_callback_preserves_selection_snapshot_and_first_pivot_sweep(self):
        X = np.zeros((65, 65))
        X[0, 0], X[1, 1] = 8., 1.
        X[1, 0] = np.finfo(float).smallest_subnormal
        events = []
        sweeps = []
        original_norm = np.linalg.norm

        def edit_input(event, flag):
            events.append(event)
            X[0, 0] = 2.

        def observed_norm(array, *args, **kwargs):
            if array.shape == X.shape and kwargs.get("axis") == 0:
                sweeps.append(len(events))
            return original_norm(array, *args, **kwargs)

        self.addCleanup(np.seterrcall, np.geterrcall())
        np.seterrcall(edit_input)
        with np.errstate(under="call"), patch.object(np.linalg, "norm", new=observed_norm):
            model = FeatureSPA(2).fit(X)
        self.assertTrue(events)
        self.assertEqual(set(events), {"underflow"})
        self.assertEqual(len(sweeps), 3)
        self.assertGreater(sweeps[0], 0)
        np.testing.assert_array_equal(model.anchor_indices_, [0, 1])
        self.assertEqual(model.column_scales_[0], 8.)
        self.assertEqual(model.anchor_condition_, 1.)
        self.assertEqual(model.original_anchor_condition_, 2.)
        np.testing.assert_array_equal(model.fit_coordinates_, X[:, [0, 1]])

    def test_callback_setting_rank_to_zero_preserves_empty_condition_error(self):
        tiny = np.finfo(float).smallest_subnormal
        X = np.array([[8., 0., .25], [tiny, 1., .125], [tiny, 0., 0.]])
        model = FeatureSPA(2).fit(np.eye(2))
        events = []

        def clear_rank(event, flag):
            events.append(event)
            model.n_components = 0

        self.addCleanup(np.seterrcall, np.geterrcall())
        np.seterrcall(clear_rank)
        with np.errstate(under="call"), self.assertRaisesRegex(np.linalg.LinAlgError, "empty arrays"):
            model.fit(X)
        self.assertEqual(events, ["underflow"])
        self.assertFalse(any(key.endswith("_") for key in vars(model)))


if __name__ == "__main__":
    unittest.main()
