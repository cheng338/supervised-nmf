"""Fold-local screening reuse, rank feasibility, and subject checkpoint tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from supervised_nmf import NMFClassifier
from supervised_nmf.decomposition import FeatureSPA
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.screening import BinaryScreening
from supervised_nmf.study import calibration_knn_grid


class StudyImprovementTests(unittest.TestCase):
    def setUp(self):
        random = np.random.RandomState(81)
        self.y = np.tile([0, 1], 30)
        self.X = random.uniform(0.1, 1.0, (60, 8)).astype(np.float32)
        self.X[:, :3] += self.y[:, None]
        self.groups = np.repeat(np.arange(15), 4)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_grid(self, estimator, grid, name="run", **kwargs):
        return calibration_knn_grid(estimator, grid, self.X, self.y, cv=3,
                                    seed=11, checkpoint_dir=Path(self.temp.name) / name,
                                    **kwargs)

    def test_reused_statistics_match_direct_masks_including_constant_and_infinite(self):
        X = np.column_stack([self.X, np.ones(60), self.y])
        base = BinaryScreening(screening="absolute", score_threshold=0).fit(X, self.y)
        for mode, values in (("absolute", [0, 1, 3, 1e10]), ("percentile", [20, 50, 100])):
            for value in values:
                template = BinaryScreening(screening=mode, **{
                    "score_threshold" if mode == "absolute" else "percentile": value})
                direct = clone(template).fit(X, self.y)
                cached = clone(template)._fit_from_statistics(base)
                for attribute in ("support_", "scores_", "t_scores_", "selected_features_",
                                  "class_counts_", "statistic_scales_", "scaled_standard_errors_"):
                    np.testing.assert_array_equal(getattr(direct, attribute), getattr(cached, attribute))
                np.testing.assert_array_equal(direct.transform(X), cached.transform(X))
                self.assertFalse(cached.support_[-2])
                self.assertTrue(cached.support_[-1])
        # Deriving another mask never changes the reference's all-feature mask.
        self.assertEqual(base.n_features_selected_, X.shape[1] - 1)

    def test_statistics_fit_once_per_training_fold_across_cutoffs_and_ranks(self):
        observed = []
        original_fit = BinaryScreening.fit

        def capture(selector, X, y):
            observed.append((X.copy(), y.copy()))
            return original_fit(selector, X, y)

        with patch.object(BinaryScreening, "fit", capture):
            report = self.run_grid(NMFClassifier(random_state=3), {
                "n_components": [1, 2], "score_threshold": [0, 1], "n_neighbors": [1, 3]})
        self.assertEqual(len(observed), 3)
        self.assertEqual(report["screening_statistics_fits"], 3)
        self.assertEqual(report["unique_representation_fits"], 12)
        for (X, y), fold in zip(observed, report["identity"]["folds"]):
            np.testing.assert_array_equal(X, self.X[fold["training_indices"]])
            np.testing.assert_array_equal(y, self.y[fold["training_indices"]])
        self.assertEqual(report["tuning_seconds"], sum(report["compute_seconds"].values()))

    def test_empty_screens_and_impossible_nmf_ranks_never_factorize(self):
        for solver, owner in (("spa", FeatureSPA), ("cd", NMF), ("mu", NMF)):
            for threshold, rank, stage in ((1e20, 1, "screening"), (0, 9, "rank_feasibility")):
                with self.subTest(solver=solver, stage=stage):
                    with patch.object(owner, "fit_transform", side_effect=AssertionError("must not factorize")):
                        report = self.run_grid(NMFClassifier(solver=solver, random_state=3),
                                               {"n_components": [rank], "score_threshold": [threshold]},
                                               name=f"{solver}-{stage}")
                    self.assertEqual(report["unique_representation_fits"], 0)
                    self.assertEqual(report["preflight_rejected_representation_folds"], 3)
                    self.assertFalse(report["candidates"][0]["valid"])
                    self.assertTrue(all(row["failure_stage"] == stage
                                        for row in report["candidates"][0]["folds"]))
                    self.assertTrue(all(row["error_type"] == "ValueError"
                                        for row in report["candidates"][0]["folds"]))

    def test_impossible_pca_rank_never_factorizes(self):
        estimator = Pipeline([("screen", BinaryScreening(percentile=100)),
                              ("pca", PCA(n_components=9, svd_solver="full", random_state=3)),
                              ("knn", KNeighborsClassifier())])
        with patch.object(PCA, "fit_transform", side_effect=AssertionError("must not factorize")):
            report = self.run_grid(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(report["unique_representation_fits"], 0)
        self.assertTrue(all(row["failure_stage"] == "rank_feasibility"
                            for row in report["representation_folds"]))

    def test_valid_predictions_match_naive_across_cutoffs_ranks_and_solvers(self):
        models = [(solver, NMFClassifier(solver=solver, random_state=3, max_iter=20000, tol=1e-3),
                   {"n_components": [2, 3], "score_threshold": [0, 1], "n_neighbors": [1, 3]})
                  for solver in ("spa", "cd", "mu")]
        models.append(("pca", Pipeline([
            ("screen", BinaryScreening(screening="absolute")),
            ("pca", PCA(svd_solver="full", random_state=3, whiten=True)),
            ("knn", KNeighborsClassifier())]),
            {"pca__n_components": [1, 2], "screen__score_threshold": [0, 1], "knn__n_neighbors": [1, 3]}))
        original_predict = KNeighborsClassifier.predict
        for name, estimator, grid in models:
            observed = []

            def capture(knn, X):
                prediction = original_predict(knn, X)
                observed.append((knn.n_neighbors, knn._fit_X.copy(), X.copy(), prediction.copy()))
                return prediction

            with patch.object(KNeighborsClassifier, "predict", capture):
                ordinary = calibration_search(estimator, grid, self.X, self.y, cv=3,
                                               seed=11, refit=False).report
            expected = observed
            observed = []
            with patch.object(KNeighborsClassifier, "predict", capture):
                shared = self.run_grid(estimator, grid, name=name)
            # Shared iteration is representation/fold/k instead of candidate/fold.
            def keyed(rows):
                return sorted((k, a.shape, a.tobytes(), b.tobytes(), p.tobytes()) for k, a, b, p in rows)
            self.assertEqual(keyed(expected), keyed(observed), name)
            self.assertTrue(all(row["valid"] for row in shared["candidates"]), name)
            self.assertEqual([row["mean_balanced_accuracy"] for row in ordinary["candidates"]],
                             [row["mean_balanced_accuracy"] for row in shared["candidates"]])

    def test_unknown_preprocessing_pipeline_keeps_ordinary_fit_semantics(self):
        # This scale changes screening inputs; the shortcut must not screen raw X.
        estimator = Pipeline([("scale", StandardScaler(with_mean=False)),
                              ("screen", BinaryScreening(percentile=50)),
                              ("knn", KNeighborsClassifier())])
        ordinary = calibration_search(estimator, {"knn__n_neighbors": [1, 3]}, self.X, self.y,
                                      cv=3, seed=11, refit=False).report
        shared = self.run_grid(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(shared["screening_statistics_fits"], 0)
        self.assertEqual([row["mean_balanced_accuracy"] for row in ordinary["candidates"]],
                         [row["mean_balanced_accuracy"] for row in shared["candidates"]])

    def test_group_and_row_id_checkpoint_identity_and_complete_resume(self):
        estimator = NMFClassifier(random_state=3)
        ids = np.array([f"sample-{i}" for i in range(len(self.y))])
        report = self.run_grid(estimator, {"score_threshold": [0, 1], "n_components": [1, 2]},
                               groups=self.groups, data_ids=ids)
        folds = [(row["training_indices"], row["validation_indices"])
                 for row in report["identity"]["folds"]]
        for train, valid in folds:
            self.assertFalse(set(self.groups[train]) & set(self.groups[valid]))
        with patch.object(BinaryScreening, "fit", side_effect=AssertionError("complete resume must not refit")):
            resumed = self.run_grid(estimator, {"score_threshold": [0, 1], "n_components": [1, 2]},
                                    groups=self.groups, data_ids=ids, folds=folds)
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertEqual(resumed["compute_seconds"], report["compute_seconds"])
        for metadata in ({"groups": self.groups + 100, "data_ids": ids},
                         {"groups": self.groups, "data_ids": ids[::-1]}):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.run_grid(estimator, {"score_threshold": [0, 1], "n_components": [1, 2]},
                              folds=folds, **metadata)


if __name__ == "__main__":
    unittest.main()
