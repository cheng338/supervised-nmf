"""Author: Chang Cheng. Public-decision contracts for shared neighbor queries."""

import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf._neighbors import _KNNQueryReuse


class NeighborReuseTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        random = np.random.RandomState(20261020)
        self.X = random.normal(size=(83, 6))
        self.Q = random.normal(size=(19, 6))

    def models(self, X, y, ks=(1, 3, 5, 18), **settings):
        return {k: KNeighborsClassifier(n_neighbors=k, n_jobs=1, **settings).fit(X, y)
                for k in ks}

    def compare(self, X, Q, y, ks, **settings):
        before = X.copy(), Q.copy(), y.copy()
        models = self.models(X, y, ks, **settings)
        reuse = _KNNQueryReuse(models, Q)
        outcomes = {}
        for k, model in models.items():
            labels, diagnostic, _ = reuse.predict(k)
            probability, _, _ = reuse.predict(k, probabilities=True)
            np.testing.assert_array_equal(labels, model.predict(Q))
            np.testing.assert_allclose(probability, model.predict_proba(Q), rtol=1e-12, atol=1e-14)
            outcomes[k] = diagnostic
        for original, actual in zip(before, (X, Q, y)):
            np.testing.assert_array_equal(original, actual)
        return reuse, outcomes

    def test_labels_dtypes_backends_and_backend_transition(self):
        for dtype in (np.float32, np.float64):
            for label_kind in ("binary", "multiclass"):
                y = (np.where(np.arange(len(self.X)) % 2, -7, 23) if label_kind == "binary"
                     else np.asarray(["class_z", "class_a", "class_m"])[np.arange(len(self.X)) % 3])
                for algorithm in ("auto", "brute", "kd_tree", "ball_tree"):
                    for weights in ("uniform", "distance"):
                        with self.subTest(dtype=dtype, labels=label_kind,
                                          algorithm=algorithm, weights=weights):
                            self.compare(self.X.astype(dtype), self.Q.astype(dtype), y,
                                         [1, 3, 5, 18, 42, 83],
                                         algorithm=algorithm, weights=weights)

    def test_original_weighted_multiclass_near_tie_uses_whole_batch(self):
        X = np.repeat(self.X[:12], 3, axis=0)
        Q = np.r_[self.X[:4], self.Q[:7]]
        y = np.asarray(["class_z", "class_a", "class_m"])[np.arange(len(X)) % 3]
        for algorithm in ("auto", "brute"):
            with self.subTest(algorithm=algorithm):
                models = self.models(X, y, [1, 3, 5, 18, 36],
                                     algorithm=algorithm, weights="distance")
                reference = models[18].predict(Q)
                reuse = _KNNQueryReuse(models, Q)
                with patch.object(models[18], "predict", wraps=models[18].predict) as public:
                    actual, metadata, _ = reuse.predict(18)
                np.testing.assert_array_equal(actual, reference)
                self.assertFalse(metadata["accelerated"])
                self.assertGreater(metadata["near_vote_rows"], 0)
                public.assert_called_once()
                self.assertIs(public.call_args.args[0], Q)
                self.assertEqual(metadata["public_fallback_query_rows"], len(Q))

    def test_binary_near_half_zero_and_duplicate_contracts(self):
        for dtype in (np.float32, np.float64):
            epsilon = np.finfo(dtype).eps
            fixtures = []
            for multiplier in (-64, -1, 0, 1, 64):
                X = np.array([[-1., 0.], [1. + multiplier * epsilon, 0.],
                              [3., 0.], [-5., 0.], [9., 0.], [-13., 0.]], dtype=dtype)
                Q = np.array([[0., 0.], [epsilon, 0.], [-epsilon, 0.]], dtype=dtype)
                fixtures.append((X, Q, np.array(["a", "z", "z", "a", "a", "z"])))
            ratio = 2.**40
            fixtures.append((np.array([[1., 0.], [-ratio, 0.], [2 * ratio, 0.],
                                       [-4 * ratio, 0.]], dtype=dtype),
                             np.zeros((2, 2), dtype=dtype), np.array(["a", "z", "z", "a"])))
            fixtures.append((np.array([[0., 0.], [0., 0.], [1., 0.], [-1., 0.],
                                       [2., 0.], [-3., 0.]], dtype=dtype),
                             np.array([[0., 0.], [.5, 0.]], dtype=dtype),
                             np.array(["a", "z", "a", "z", "z", "a"])))
            for X, Q, y in fixtures:
                for algorithm in ("brute", "kd_tree", "ball_tree"):
                    with self.subTest(dtype=dtype, algorithm=algorithm, X=X.tolist()):
                        models = self.models(X, y, [1, 2, 3, len(X)],
                                             algorithm=algorithm, weights="distance")
                        reuse = _KNNQueryReuse(models, Q)
                        for k, model in models.items():
                            expected = model.predict_proba(Q)
                            for positive_index in (0, 1):
                                # Includes non-half exact probability boundaries;
                                # threshold guard must preserve the fitted rule.
                                for threshold in (0., .5, 1., float(expected[0, positive_index])):
                                    actual, _, _ = reuse.predict(
                                        k, probabilities=True, decision_threshold=threshold,
                                        positive_index=positive_index)
                                    np.testing.assert_array_equal(
                                        actual[:, positive_index] > threshold,
                                        expected[:, positive_index] > threshold)

    def test_shared_query_once_and_distinct_geometry_not_merged(self):
        y = np.zeros(len(self.X), dtype=int)
        models = self.models(self.X, y, algorithm="brute", weights="uniform")
        reuse = _KNNQueryReuse(models, self.Q)
        with patch.object(models[18], "kneighbors", wraps=models[18].kneighbors) as query:
            for k in models:
                _, metadata, _ = reuse.predict(k)
                self.assertTrue(metadata["accelerated"])
            self.assertEqual(query.call_count, 1)
            self.assertEqual(query.call_args.kwargs["n_neighbors"], 19)
        self.assertEqual(len(reuse.query_records()), 1)
        self.assertGreaterEqual(reuse.query_seconds, 0.)
        copied = self.models(self.X.copy(), y, [3], algorithm="brute", weights="uniform")
        mixed = _KNNQueryReuse({1: models[1], 3: copied[3]}, self.Q)
        self.assertEqual(len(mixed.groups), 2)
        self.assertTrue(all(len(group["ks"]) == 1 for group in mixed.groups))

    def test_callable_metric_sparse_and_unknown_backend_use_public_calls(self):
        y = np.arange(len(self.X)) % 2
        cases = [
            (self.X, self.Q, {"weights": lambda distances: np.ones_like(distances)}),
            (self.X, self.Q, {"metric": "manhattan"}),
            (sparse.csr_matrix(self.X), sparse.csr_matrix(self.Q), {}),
            (sparse.csr_array(self.X), sparse.csr_array(self.Q), {}),
        ]
        for X, Q, settings in cases:
            models = self.models(X, y, [1, 3], **settings)
            reuse = _KNNQueryReuse(models, Q)
            for k, model in models.items():
                expected = model.predict(Q)
                with patch.object(model, "predict", wraps=model.predict) as public:
                    actual, metadata, _ = reuse.predict(k)
                np.testing.assert_array_equal(actual, expected)
                public.assert_called_once()
                self.assertFalse(metadata["shared_query_attempted"])
                self.assertEqual(metadata["public_fallback_query_rows"], Q.shape[0])
                self.assertEqual(reuse.query_seconds, 0.)
        models = self.models(self.X, y, [1, 3], algorithm="brute")
        models[1]._fit_method = "future_backend"
        reuse = _KNNQueryReuse(models, self.Q)
        with patch.object(models[1], "predict", return_value=np.ones(len(self.Q))) as public:
            _, metadata, _ = reuse.predict(1)
        self.assertEqual(metadata["reason"], "unknown_neighbor_backend")
        public.assert_called_once()

    def test_query_failure_falls_back_per_k_and_retains_warnings(self):
        y = np.arange(len(self.X)) % 2
        models = self.models(self.X, y, [1, 3], algorithm="brute", weights="distance")
        expected = {k: model.predict(self.Q) for k, model in models.items()}
        reuse = _KNNQueryReuse(models, self.Q)
        original = models[3].kneighbors

        def fail_shared(*args, **kwargs):
            if kwargs.get("n_neighbors") == 4:
                warnings.warn("shared query diagnostic", RuntimeWarning)
                raise RuntimeError("only the optimization query failed")
            return original(*args, **kwargs)

        with patch.object(models[3], "kneighbors", fail_shared):
            for k in models:
                actual, metadata, caught = reuse.predict(k)
                np.testing.assert_array_equal(actual, expected[k])
                self.assertEqual(metadata["reason"], "shared_query_failed")
                self.assertEqual(len(caught), 1)
        record = reuse.query_records()[0]
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(record["warnings"][0]["message"], "shared query diagnostic")

    def test_invalid_k_public_failure_does_not_disable_valid_k(self):
        y = np.arange(len(self.X)) % 2
        models = self.models(self.X, y, [1, 3, len(y) + 1], algorithm="brute")
        reuse = _KNNQueryReuse(models, self.Q)
        for k in (1, 3):
            np.testing.assert_array_equal(reuse.predict(k)[0], models[k].predict(self.Q))
        with self.assertRaises(ValueError):
            reuse.predict(len(y) + 1)


if __name__ == "__main__":
    unittest.main()
