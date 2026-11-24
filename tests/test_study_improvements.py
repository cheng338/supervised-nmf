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
from supervised_nmf._neighbors import _KNNQueryReuse


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

    def test_missing_knn_training_state_resumes_from_last_completed_fold(self):
        from supervised_nmf import study

        estimator = KNeighborsClassifier()
        grid = {"n_neighbors": [1, 3, len(self.y)]}
        reference = calibration_search(estimator, grid, self.X, self.y, cv=3, seed=11,
                                       threads=1, refit=False).report
        read_training = study._stored_training_representation
        reads = 0

        def lose_second_matrix(fitted, kind, *args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 2:
                del fitted._fit_X
            return read_training(fitted, kind, *args, **kwargs)

        with patch.object(study, "_stored_training_representation", lose_second_matrix):
            with self.assertRaises(study._KNNTrainingStateCompatibilityError) as caught:
                self.run_grid(estimator, grid)

        self.assertEqual(reads, 2)
        directory = Path(self.temp.name) / "run"
        committed, = (directory / "folds").glob("*.json")
        saved = committed.read_bytes()
        attempts = [study._read_receipt(path) for path in (directory / "attempts").glob("*.json")]
        self.assertCountEqual([entry["state"] for entry in attempts], ["completed", "failed"])
        failed, = [entry for entry in attempts if entry["state"] == "failed"]
        self.assertEqual(failed["attempt_error_type"], type(caught.exception).__name__)
        self.assertFalse((directory / "folds" / (failed["key"] + ".json")).exists())
        self.assertFalse((directory / "report.json").exists())

        restored = self.run_grid(estimator, grid)
        self.assertEqual(restored["reused_representation_folds"], 1)
        self.assertEqual(restored["new_representation_fits"], 2)
        self.assertEqual(committed.read_bytes(), saved)
        self.assertEqual(restored["selected_parameters"], reference["selected_parameters"])
        for actual, expected in zip(restored["candidates"], reference["candidates"], strict=True):
            self.assertEqual(actual["parameters"], expected["parameters"])
            self.assertEqual(actual["valid"], expected["valid"])
            self.assertEqual([row.get("score") for row in actual["folds"]],
                             [row.get("score") for row in expected["folds"]])

        with patch.object(study, "_stored_training_representation", side_effect=AssertionError) as reader:
            completed = self.run_grid(estimator, grid)
        reader.assert_not_called()
        self.assertEqual(completed["reused_representation_folds"], 3)
        self.assertEqual(completed["new_representation_fits"], 0)
        self.assertEqual(completed["candidates"], restored["candidates"])
        self.assertEqual(completed["ranked_candidate_indices"], restored["ranked_candidate_indices"])

    def test_malformed_knn_width_preserves_compatibility_error_on_abort(self):
        from supervised_nmf import study

        estimator = KNeighborsClassifier()
        grid = {"n_neighbors": [1, 3]}
        read_training = study._stored_training_representation

        def malformed_width(fitted, kind):
            fitted.n_features_in_ = "unavailable"
            return read_training(fitted, kind)

        with patch.object(study, "_stored_training_representation", malformed_width):
            with self.assertRaisesRegex(study._KNNTrainingStateCompatibilityError,
                                        "n_features_in_='unavailable'") as caught:
                self.run_grid(estimator, grid)

        directory = Path(self.temp.name) / "run"
        attempt, = (directory / "attempts").glob("*.json")
        failed = study._read_receipt(attempt)
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["attempt_error_type"], type(caught.exception).__name__)
        self.assertFalse(list((directory / "folds").glob("*.json")))
        self.assertFalse((directory / "report.json").exists())

        resumed = self.run_grid(estimator, grid)
        self.assertEqual(resumed["reused_representation_folds"], 0)
        self.assertEqual(resumed["new_representation_fits"], 3)
        self.assertTrue(all(candidate["valid"] for candidate in resumed["candidates"]))

    def test_native_knn_fit_failure_stays_a_completed_fold(self):
        estimator = KNeighborsClassifier(leaf_size=0)
        grid = {"n_neighbors": [1, 3]}
        report = self.run_grid(estimator, grid)
        self.assertTrue(all(row["failure_stage"] == "fit" and row["status"] == "failed"
                            for row in report["representation_folds"]))
        self.assertFalse(report["ranked_candidate_indices"])
        resumed = self.run_grid(estimator, grid)
        self.assertEqual(resumed["reused_representation_folds"], 3)
        self.assertEqual(resumed["new_representation_fits"], 0)
        self.assertEqual(resumed["candidates"], report["candidates"])

    def test_optional_kd_metadata_missing_on_one_real_head(self):
        from supervised_nmf import study

        X = self.X[:40].copy()
        query = self.X[40:]
        models = {k: KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree").fit(X, self.y[:40])
                  for k in (1, 3, 5)}
        expected = models[3].predict(query)
        del models[3].effective_metric_params_
        # Stock tree prediction uses the built index for this explicit query.
        np.testing.assert_array_equal(models[3].predict(query), expected)
        self.assertIsNone(study._kd_training_snapshot({1: models[1], 3: models[3]}, X))
        snapshot = study._kd_training_snapshot(models, X)
        np.testing.assert_array_equal(snapshot, X)
        reuse = _KNNQueryReuse(models, query, _head_fit_input=X)
        self.assertIsNone(reuse.by_k[3][0])
        predictions, metadata, _ = reuse.predict(3)
        self.assertFalse(metadata["shared_query_attempted"])
        np.testing.assert_array_equal(predictions, expected)

    def test_knn_reader_requires_usable_storage_and_preserves_fitted_carriers(self):
        from joblib import dump, load
        from scipy.sparse import csr_matrix
        from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
        from supervised_nmf import study

        class ShapeOnly:
            shape = self.X.shape
            dtype = self.X.dtype

            def __array__(self, *args, **kwargs):
                raise AssertionError("A shape and a dtype do not establish usable storage")

        class MisreportedDtype(np.ndarray):
            @property
            def dtype(self):
                return np.dtype("float64")

        index_path = Path(self.temp.name) / "sparse-index.joblib"
        dump(NearestNeighbors().fit(csr_matrix(self.X)), index_path)
        mapped_index = load(index_path, mmap_mode="r")
        for array in (mapped_index._fit_X.data, mapped_index._fit_X.indices, mapped_index._fit_X.indptr):
            if isinstance(array, np.memmap):
                self.addCleanup(array._mmap.close)
        self.assertIsInstance(mapped_index._fit_X.data, np.memmap)
        for source in (self.X, csr_matrix(self.X), KDTree(self.X), BallTree(self.X),
                       NearestNeighbors().fit(self.X), mapped_index):
            with self.subTest(source=type(source).__name__):
                head = KNeighborsClassifier().fit(source, self.y)
                matrix, template = study._stored_training_representation(Pipeline([("knn", head)]), "pipeline")
                self.assertIs(template, head)
                if isinstance(source, (KDTree, BallTree)):
                    fitted_values = np.asarray(head._fit_X)
                    self.assertTrue(np.shares_memory(matrix, fitted_values))
                    self.assertEqual(matrix.dtype, fitted_values.dtype)
                    np.testing.assert_array_equal(matrix, fitted_values)
                else:
                    self.assertIs(matrix, head._fit_X)

        head = KNeighborsClassifier().fit(self.X, self.y)
        strings = np.full(self.X.shape, "invalid")
        for unusable in (strings, strings.view(MisreportedDtype), self.X.astype(object),
                         self.X.astype(complex), ShapeOnly()):
            with self.subTest(carrier=type(unusable).__name__, dtype=unusable.dtype):
                with patch.object(head, "_fit_X", unusable):
                    with self.assertRaises(study._KNNTrainingStateCompatibilityError):
                        study._stored_training_representation(head, "raw")

        def broken_count(model):
            raise RuntimeError("Fitted metadata is unreadable")

        with patch.object(KNeighborsClassifier, "n_samples_fit_", property(broken_count), create=True):
            with self.assertRaises(study._KNNTrainingStateCompatibilityError) as caught:
                study._stored_training_representation(head, "raw")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_nonfinite_training_values_remain_ordinary_fold_failures(self):
        from supervised_nmf import study

        read = study._stored_training_representation
        for name, value in (("nan", np.nan), ("inf", np.inf)):
            with self.subTest(value=name):
                def nonfinite(fitted, kind, *args, **kwargs):
                    matrix = fitted._fit_X.copy()
                    matrix[0, 0] = value
                    with patch.object(fitted, "_fit_X", matrix):
                        return read(fitted, kind, *args, **kwargs)

                with patch.object(study, "_stored_training_representation", nonfinite):
                    report = self.run_grid(KNeighborsClassifier(), {"n_neighbors": [1]}, name=name)
                for row in report["representation_folds"]:
                    self.assertEqual(row["status"], "failed")
                    self.assertEqual(row["failure_stage"], "projection")
                    self.assertEqual(row["error_type"], "ValueError")
                    self.assertIn("nonfinite coordinates", row["error"])

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
        estimator = Pipeline([("screen", BinaryScreening(screening="percentile", percentile=100)),
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
        original_shared_predict = _KNNQueryReuse.predict
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

            def capture_shared(reuse, k, **kwargs):
                result = original_shared_predict(reuse, k, **kwargs)
                observed.append((k, reuse.models[k]._fit_X.copy(),
                                 reuse.query.copy(), result[0].copy()))
                return result

            with patch.object(_KNNQueryReuse, "predict", capture_shared):
                shared = self.run_grid(estimator, grid, name=name)
            # Shared iteration is representation/fold/k instead of candidate/fold.
            def keyed(rows):
                return sorted((k, a.shape, a.tobytes(), b.tobytes(), p.tobytes()) for k, a, b, p in rows)
            self.assertEqual(keyed(expected), keyed(observed), name)
            self.assertTrue(all(row["valid"] for row in shared["candidates"]), name)
            self.assertEqual([row["mean_score"] for row in ordinary["candidates"]],
                             [row["mean_score"] for row in shared["candidates"]])

    def test_unknown_preprocessing_pipeline_keeps_ordinary_fit_semantics(self):
        # This scale changes screening inputs; the shortcut must not screen raw X.
        estimator = Pipeline([("scale", StandardScaler(with_mean=False)),
                              ("screen", BinaryScreening(screening="percentile", percentile=50)),
                              ("knn", KNeighborsClassifier())])
        ordinary = calibration_search(estimator, {"knn__n_neighbors": [1, 3]}, self.X, self.y,
                                      cv=3, seed=11, refit=False).report
        shared = self.run_grid(estimator, {"knn__n_neighbors": [1, 3]})
        self.assertEqual(shared["screening_statistics_fits"], 0)
        self.assertEqual([row["mean_score"] for row in ordinary["candidates"]],
                         [row["mean_score"] for row in shared["candidates"]])

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
