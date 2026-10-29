"""Public projection execution, persistence, and compatibility regressions.

The independent reference is the public SciPy row-wise NNLS operation used
before batched projection. Process checks execute in a fresh interpreter so
that temporary worker pools cannot affect the rest of the test suite.
"""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_info, threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier
from supervised_nmf import _projection


PROJECTION_PARAMETERS = {
    "projection_solver": "auto",
    "projection_workers": 1,
    "projection_batch_size": 256,
    "projection_native_threads": 1,
}


def _data():
    rng = np.random.default_rng(2026100911)
    basis = np.column_stack([np.eye(4), rng.uniform(.05, .7, (4, 21))])
    weights = rng.uniform(.1, 2, (64, 4))
    train = weights @ basis + rng.uniform(0, .01, (64, 25))
    query = rng.uniform(.1, 2, (43, 4)) @ basis
    query[1] = 0
    query[10] = query[9]
    labels = np.where(weights[:, 0] > np.median(weights[:, 0]), "yes", "no")
    return train, query, labels


def _reference(model, query):
    design = np.ascontiguousarray(model.components_.T)
    return np.vstack([nnls(design, row, maxiter=model.nnls_max_iter)[0]
                      for row in query])


class ProjectionExecutionTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        self.train, self.query, self.labels = _data()

    def test_public_defaults_and_clone_preserve_projection_controls(self):
        for estimator in (FeatureSPA(), NMFClassifier()):
            with self.subTest(estimator=type(estimator).__name__):
                params = estimator.get_params()
                for name, value in PROJECTION_PARAMETERS.items():
                    self.assertEqual(params[name], value)
                estimator.set_params(projection_solver="scipy", projection_workers=2,
                                     projection_batch_size=17, projection_native_threads=2)
                copied = clone(estimator)
                self.assertEqual(copied.get_params(), estimator.get_params())
                self.assertFalse(hasattr(copied, "components_"))

    def test_all_backends_match_public_scipy_on_unseen_rows(self):
        model = FeatureSPA(4).fit(self.train)
        expected = _reference(model, self.query)
        original_basis = model.components_.copy()
        original_query = self.query.copy()
        for solver in ("scipy", "auto", "batched"):
            for batch_size in (1, 7, 256):
                with self.subTest(solver=solver, batch_size=batch_size):
                    model.set_params(projection_solver=solver,
                                     projection_batch_size=batch_size)
                    actual = model.transform(self.query)
                    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
                    np.testing.assert_array_equal(actual[1], np.zeros(4))
                    np.testing.assert_allclose(actual[9], actual[10], rtol=1e-12, atol=1e-12)
                    self.assertEqual(actual.dtype, np.float64)
                    self.assertTrue(np.all(actual >= 0))
        np.testing.assert_array_equal(model.components_, original_basis)
        np.testing.assert_array_equal(self.query, original_query)

    def test_fitting_is_unchanged_by_projection_backend(self):
        serial = FeatureSPA(4, projection_solver="scipy").fit(self.train)
        for solver in ("auto", "batched"):
            with self.subTest(solver=solver):
                model = FeatureSPA(4, projection_solver=solver).fit(self.train)
                np.testing.assert_array_equal(model.anchor_indices_, serial.anchor_indices_)
                np.testing.assert_array_equal(model.components_, serial.components_)
                np.testing.assert_array_equal(model.fit_coordinates_, serial.fit_coordinates_)

    def test_pickle_roundtrip_and_clone_after_transform(self):
        model = FeatureSPA(4, projection_solver="batched",
                           projection_batch_size=11).fit(self.train)
        expected = model.transform(self.query)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.get_params(), model.get_params())
        np.testing.assert_array_equal(restored.transform(self.query), expected)
        copied = clone(model)
        self.assertEqual(copied.get_params(), model.get_params())
        with self.assertRaises(NotFittedError):
            copied.transform(self.query)

    def test_old_pickle_without_execution_controls_retains_scipy(self):
        model = FeatureSPA(4, projection_solver="scipy").fit(self.train)
        expected = _reference(model, self.query)
        for name in PROJECTION_PARAMETERS:
            delattr(model, name)
        restored = pickle.loads(pickle.dumps(model))
        # Exact equality distinguishes the unchanged row-wise reference from
        # numerically close grouped normal-equation solves on this fixture.
        np.testing.assert_array_equal(restored.transform(self.query), expected)
        self.assertEqual(restored.get_params()["projection_solver"], "scipy")
        self.assertEqual(clone(restored).projection_solver, "scipy")

    def test_native_library_discovery_is_reused_between_queries(self):
        model = FeatureSPA(4).fit(self.train)
        with patch.object(_projection, "_controller", None), \
                patch.object(_projection, "_controller_pid", None), \
                patch.object(_projection, "ThreadpoolController",
                             wraps=_projection.ThreadpoolController) as discover:
            first = model.transform(self.query[:1])
            second = model.transform(self.query[:1])
            self.assertEqual(discover.call_count, 1)
            np.testing.assert_array_equal(first, second)

    def test_invalid_controls_are_rejected_at_fit_and_after_set_params(self):
        invalid = [
            ("projection_solver", None), ("projection_solver", "threading"),
            ("projection_workers", 0), ("projection_workers", -1),
            ("projection_workers", True), ("projection_workers", 1.5),
            ("projection_batch_size", 0), ("projection_batch_size", False),
            ("projection_batch_size", 1.5), ("projection_native_threads", 0),
            ("projection_native_threads", True), ("projection_native_threads", 1.5),
        ]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    FeatureSPA(4, **{name: value}).fit(self.train)
                model = FeatureSPA(4).fit(self.train)
                model.set_params(**{name: value})
                with self.assertRaisesRegex(ValueError, name):
                    model.transform(self.query)
                with self.assertRaisesRegex(ValueError, name):
                    model.fit(self.train)
                with self.assertRaises(NotFittedError):
                    model.transform(self.query)

    def test_scipy_iteration_errors_propagate_after_set_params(self):
        model = FeatureSPA(3, projection_solver="scipy", nnls_max_iter=20).fit(np.eye(3))
        basis = model.components_.copy()
        model.set_params(nnls_max_iter=1)
        with self.assertRaisesRegex(RuntimeError, "iterations"):
            model.transform(np.ones((2, 3)))
        np.testing.assert_array_equal(model.components_, basis)

    def test_small_queries_do_not_construct_a_process_pool(self):
        model = FeatureSPA(4, projection_workers=2).fit(self.train)
        with patch("supervised_nmf._projection.Parallel",
                   side_effect=AssertionError("small query must stay in the calling process")):
            for count, batch_size in ((1, 256), (43, 7), (511, 256), (600, 400)):
                with self.subTest(count=count, batch_size=batch_size):
                    incoming = np.resize(self.query, (count, self.query.shape[1]))
                    model.set_params(projection_batch_size=batch_size)
                    expected = _reference(model, incoming)
                    actual = model.transform(incoming)
                    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)

    def test_classifier_forwarding_and_fixed_head_decisions(self):
        for classifier in ("knn", "logistic"):
            for training_coordinates in ("fit_transform", "transform"):
                with self.subTest(classifier=classifier, coordinates=training_coordinates):
                    options = dict(n_components=4, screening="none", classifier=classifier,
                                   n_neighbors=5, training_coordinates=training_coordinates,
                                   projection_batch_size=7, projection_native_threads=1)
                    serial = NMFClassifier(projection_solver="scipy", **options).fit(self.train, self.labels)
                    batch = NMFClassifier(projection_solver="batched", **options).fit(self.train, self.labels)
                    self.assertEqual(batch.decomposition_.projection_solver, "batched")
                    self.assertEqual(batch.decomposition_.projection_batch_size, 7)
                    np.testing.assert_allclose(batch.transform(self.query), serial.transform(self.query),
                                               rtol=1e-7, atol=1e-9)
                    np.testing.assert_array_equal(batch.predict(self.query), serial.predict(self.query))
                    np.testing.assert_allclose(batch.predict_proba(self.query), serial.predict_proba(self.query),
                                               rtol=1e-7, atol=1e-9)

    def test_classifier_execution_controls_can_change_without_refitting(self):
        model = NMFClassifier(n_components=4, screening="none",
                              projection_solver="scipy").fit(self.train, self.labels)
        expected = model.predict_proba(self.query)
        basis = model.decomposition_.components_.copy()
        head = model.classifier_
        fitted_controls = {name: getattr(model.decomposition_, name) for name in PROJECTION_PARAMETERS}
        original_transform = FeatureSPA.transform
        observed_controls = []

        def observe_projection(decomposition, incoming):
            observed_controls.append({name: getattr(decomposition, name) for name in PROJECTION_PARAMETERS})
            self.assertIsNot(decomposition, model.decomposition_)
            return original_transform(decomposition, incoming)

        model.set_params(projection_solver="batched", projection_workers=2,
                         projection_batch_size=7, projection_native_threads=1)
        with patch.object(FeatureSPA, "transform", autospec=True, side_effect=observe_projection):
            np.testing.assert_allclose(model.predict_proba(self.query), expected, rtol=1e-7, atol=1e-9)
        self.assertEqual(observed_controls, [{name: getattr(model, name) for name in PROJECTION_PARAMETERS}])
        self.assertEqual({name: getattr(model.decomposition_, name) for name in PROJECTION_PARAMETERS}, fitted_controls)
        np.testing.assert_array_equal(model.decomposition_.components_, basis)
        self.assertIs(model.classifier_, head)
        model.set_params(projection_workers=0)
        with self.assertRaisesRegex(ValueError, "projection_workers"):
            model.predict(self.query)
        # Repairing execution controls must not require refitting either part.
        model.set_params(projection_workers=1)
        np.testing.assert_allclose(model.predict_proba(self.query), expected, rtol=1e-7, atol=1e-9)
        self.assertIs(model.classifier_, head)

    def test_native_thread_limit_restored_after_success_and_failure(self):
        def pools():
            return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}

        with threadpool_limits(limits=2):
            model = FeatureSPA(3, projection_native_threads=1, projection_solver="scipy",
                               nnls_max_iter=20).fit(np.eye(3))
            before = pools()
            model.transform(np.ones((2, 3)))
            self.assertEqual(pools(), before)
            model.set_params(nnls_max_iter=1)
            with self.assertRaisesRegex(RuntimeError, "iterations"):
                model.transform(np.ones((2, 3)))
            self.assertEqual(pools(), before)

    def test_process_pool_reuse_serialization_and_errors_in_fresh_interpreter(self):
        # Only the test subprocess repairs this workspace's renamed executable
        # path. Production code must work with ordinary interpreter installs
        # and must not rewrite interpreter globals for this local packaging.
        script = textwrap.dedent(r'''
            import json, os, pickle, sys
            from pathlib import Path
            if os.name == "nt" and not Path(sys._base_executable).is_file():
                base = Path(sys.base_prefix) / "python.exe"
                if not base.is_file():
                    raise RuntimeError("Test could not find actual base Python executable")
                sys._base_executable = str(base)
            import numpy as np
            from joblib.externals.loky import reusable_executor
            from threadpoolctl import threadpool_limits
            from supervised_nmf import FeatureSPA
            rng = np.random.default_rng(931)
            train = rng.gamma(2, 1, (48, 19))
            query = rng.gamma(2, 1, (641, 19))
            records = []
            with threadpool_limits(limits=1):
                model = FeatureSPA(4, projection_solver="scipy", projection_batch_size=128).fit(train)
                expected = model.transform(query)
                model.set_params(projection_workers=2)
                first = model.transform(query)
                np.testing.assert_array_equal(first, expected)
                executor = reusable_executor._executor
                assert executor is not None
                initial_pids = set(executor._processes)
                assert len(initial_pids) == 2, initial_pids
                assert os.getpid() not in initial_pids
                second = model.transform(query[::-1])
                np.testing.assert_array_equal(second, expected[::-1])
                assert reusable_executor._executor is executor
                assert set(executor._processes) == initial_pids
                restored = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(restored.transform(query), expected)
                assert reusable_executor._executor is executor
                # A changed fitted basis must not read a previous model's
                # cached numerical design from the persistent worker pool.
                replacement = FeatureSPA(3, projection_solver="scipy", projection_workers=2,
                                         projection_batch_size=128).fit(train[:, :11])
                alternative = replacement.transform(query[:, :11])
                replacement.set_params(projection_workers=1)
                np.testing.assert_array_equal(alternative, replacement.transform(query[:, :11]))
                for solver in ("auto", "batched"):
                    model.set_params(projection_solver=solver)
                    np.testing.assert_allclose(model.transform(query), expected, rtol=1e-7, atol=1e-9)
                    records.append(solver)
                failing = FeatureSPA(3, projection_solver="scipy", projection_workers=2,
                                     projection_batch_size=128, nnls_max_iter=20).fit(np.eye(3))
                failing.set_params(nnls_max_iter=1)
                try:
                    failing.transform(np.ones((641, 3)))
                except RuntimeError as error:
                    assert "iterations" in str(error).lower(), str(error)
                else:
                    raise AssertionError("child NNLS iteration failure disappeared")
                # An unsuccessful job must not leave estimator-side pool
                # state that prevents the next independent query.
                failing.set_params(nnls_max_iter=20)
                np.testing.assert_array_equal(failing.transform(np.ones((641, 3))), np.ones((641, 3)))
                payload = dict(processes=len(initial_pids), reused=True, solvers=records,
                               serialized=True, iteration_failure_visible=True)
            current = reusable_executor._executor
            if current is not None:
                current.shutdown(wait=True)
            print(json.dumps(payload))
        ''')
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["processes"], 2)
        self.assertTrue(payload["reused"])
        self.assertTrue(payload["serialized"])
        self.assertTrue(payload["iteration_failure_visible"])
        self.assertEqual(payload["solvers"], ["auto", "batched"])


if __name__ == "__main__":
    unittest.main()
