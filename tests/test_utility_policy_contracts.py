"""Public MU profiling and logistic prediction contracts."""
from copy import copy
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from supervised_nmf import BinaryScreening, NMFClassifier
from supervised_nmf import classifier as classifier_module
from _thread_import_state import preserve_thread_environment

with preserve_thread_environment():
    from scripts import profile_celeba_stages as profiler


class LimitedCase(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(1)
        self.addCleanup(self.limits.restore_original_limits)


class ProfilerPolicyTests(LimitedCase):
    def fixture(self):
        rng = np.random.default_rng(831)
        h = np.full((2, 12), .1); h[0, ::2] += 2; h[1, 1::2] += 2
        latent = rng.uniform(.3, 2, (72, 2))
        x = latent @ h
        y = (latent[:, 0] > np.median(latent[:, 0])).astype(int)
        q = np.asarray([[.4, .7], [.6, 1.5], [1.8, .8], [1.9, 1.7]]) @ h
        return x, y, q

    def model(self, **kwargs):
        params = dict(solver='mu', screening='none', n_components=2, n_neighbors=3,
                      weights='distance', max_iter=3000, tol=1e-4, random_state=17)
        return NMFClassifier(**dict(params, **kwargs))

    def test_native_transform_identity_and_dispatch_restored_on_exit(self):
        original = NMF.transform
        dispatch = {n: getattr(classifier_module, n) for n in ('transform_mu', 'transform_mu_rowwise')}
        with profiler._StageProfiler():
            self.assertIs(NMF.transform, original)
            for name, function in dispatch.items():
                self.assertIsNot(getattr(classifier_module, name), function)
        self.assertIs(NMF.transform, original)
        for name, function in dispatch.items():
            self.assertIs(getattr(classifier_module, name), function)

    def test_all_mu_policies_keep_coordinates_predictions_and_single_stage_counts(self):
        x, y, q = self.fixture()
        for policy in ('auto', 'batch', 'rowwise'):
            for training in ('fit_transform', 'transform'):
                with self.subTest(policy=policy, training=training):
                    settings = self.model(mu_projection=policy, training_coordinates=training)
                    plain = clone(settings).fit(x, y)
                    expected_w = plain.transform(q)
                    expected_labels, expected_p = plain.predict_with_probabilities(q)
                    with profiler._StageProfiler() as meter:
                        measured = clone(settings).fit(x, y)
                        meter.phase = 'prediction'
                        actual_labels, actual_p = measured.predict_with_probabilities(q)
                    np.testing.assert_array_equal(measured.training_coordinates_, plain.training_coordinates_)
                    np.testing.assert_array_equal(measured.transform(q), expected_w)
                    np.testing.assert_array_equal(actual_labels, expected_labels)
                    np.testing.assert_array_equal(actual_p, expected_p)
                    self.assertEqual(measured.mu_projection_, plain.mu_projection_)
                    self.assertEqual(meter.calls['training_projection_seconds'], int(training == 'transform'))
                    self.assertEqual(meter.calls['decomposition_projection_seconds'], 1)

    def test_rowwise_error_and_warning_are_unmodified_and_wrappers_restore(self):
        x, y, q = self.fixture()
        model = self.model().fit(x, y)
        original = classifier_module.transform_mu_rowwise
        def fail(*args, **kwargs):
            warnings.warn('dispatch warning sentinel', RuntimeWarning)
            raise RuntimeError('dispatch error sentinel')
        with patch.object(classifier_module, 'transform_mu_rowwise', fail):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with profiler._StageProfiler() as meter:
                    meter.phase = 'prediction'
                    with self.assertRaisesRegex(RuntimeError, 'dispatch error sentinel'):
                        model.transform(q)
                self.assertIs(classifier_module.transform_mu_rowwise, fail)
            self.assertEqual([(w.category, str(w.message)) for w in caught],
                             [(RuntimeWarning, 'dispatch warning sentinel')])
            self.assertEqual(meter.calls['decomposition_projection_seconds'], 1)
        self.assertIs(classifier_module.transform_mu_rowwise, original)

    def test_external_transform_override_keeps_ordinary_rowwise_rejection(self):
        x, y, q = self.fixture()
        model = self.model().fit(x, y)
        original = NMF.transform
        def external(instance, *args, **kwargs):
            return original(instance, *args, **kwargs)
        with patch.object(NMF, 'transform', external):
            with self.assertRaises(ValueError) as plain:
                model.transform(q)
            with profiler._StageProfiler() as meter:
                meter.phase = 'prediction'
                self.assertIs(NMF.transform, external)
                with self.assertRaises(ValueError) as measured:
                    model.transform(q)
            self.assertEqual(str(measured.exception), str(plain.exception))
            self.assertIs(NMF.transform, external)
        self.assertIs(NMF.transform, original)


class LogisticDefaultPolicyTests(LimitedCase):
    def fixture(self):
        rng = np.random.RandomState(23)
        y = np.arange(80) % 2
        w = rng.uniform(.1, 1, (80, 2)); w[:, 0] += y
        h = np.column_stack([np.eye(2), rng.uniform(.1, 1, (2, 10))])
        # Use native float64 for the independent per-row sklearn reference;
        # float32's intentional double-working rowwise policy is covered by the
        # existing MU policy suite, rather than falsely equated bit for bit.
        x = w @ h + rng.uniform(0, .005, (80, 12))
        return x, np.where(y, 'present', 'absent')

    def model(self, **kwargs):
        return NMFClassifier(**dict(dict(classifier='logistic', solver='mu', n_components=2,
                                        score_threshold=0, random_state=7, max_iter=3000, tol=1e-3,
                                        logistic_C=.3, logistic_class_weight='balanced'), **kwargs))

    def test_omitted_default_equals_explicit_rowwise_for_both_training_policies(self):
        x, y = self.fixture()
        for mode in ('fit_transform', 'transform'):
            with self.subTest(training=mode):
                default = self.model(training_coordinates=mode).fit(x, y)
                explicit = self.model(training_coordinates=mode, mu_projection='rowwise').fit(x, y)
                self.assertEqual(default.mu_projection, 'auto')
                self.assertEqual(default.mu_projection_, 'rowwise')
                np.testing.assert_array_equal(default.training_coordinates_, explicit.training_coordinates_)
                for q in (x[:12]*1.07, np.zeros((2, x.shape[1]))):
                    a, ap = default.predict_with_probabilities(q)
                    b, bp = explicit.predict_with_probabilities(q)
                    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(ap, bp)

    def test_rowwise_logistic_matches_independent_native_single_row_reference(self):
        x, y = self.fixture()
        for mode in ('fit_transform', 'transform'):
            with self.subTest(training=mode):
                screen = BinaryScreening(screening='absolute', score_threshold=0)
                selected = screen.fit_transform(x, y)
                representation = NMF(n_components=2, init='nndsvda', solver='mu', beta_loss='frobenius',
                                     max_iter=3000, tol=1e-3, random_state=7)
                w = representation.fit_transform(selected)
                if mode == 'transform':
                    w = np.vstack([copy(representation).transform(row[None]) for row in selected])
                head = Pipeline([('scale', StandardScaler()),
                    ('logistic', LogisticRegression(C=.3, solver='lbfgs', max_iter=1000, tol=1e-4,
                                                   class_weight='balanced', random_state=7))]).fit(w, y)
                q = x[:12] * 1.07
                encoded = np.vstack([copy(representation).transform(row[None]) for row in screen.transform(q)])
                model = self.model(training_coordinates=mode).fit(x, y)
                actual_labels, actual_p = model.predict_with_probabilities(q)
                np.testing.assert_allclose(model.training_coordinates_, w, rtol=5e-6, atol=1e-12)
                np.testing.assert_allclose(model.transform(q), encoded, rtol=5e-6, atol=1e-12)
                np.testing.assert_array_equal(actual_labels, head.predict(encoded))
                np.testing.assert_allclose(actual_p, head.predict_proba(encoded), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(model.classifier_.named_steps['scale'].mean_,
                                           w.mean(axis=0, dtype=np.float64), rtol=5e-6, atol=1e-12)
