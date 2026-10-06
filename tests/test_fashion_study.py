"""Synthetic checks for Chang Cheng's training-only Fashion benchmark runner."""

from copy import deepcopy
import gc
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fashion_mnist_study as runner
from supervised_nmf.evaluation import _digest, evaluate_classifier, write_json
from supervised_nmf.study import _read_receipt, _write_receipt


class FashionStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fashion-study-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        random = np.random.RandomState(71)
        self.y = np.repeat(np.array([0, 6], dtype=np.uint8), 60)
        self.X = random.uniform(0.1, 0.3, size=(120, 16)).astype(np.float32)
        self.X[:, :5] += (self.y[:, None] == 6).astype(np.float32) * 0.5
        self.X[:, -1] = 0
        np.save(self.cache / "X_train.npy", self.X, allow_pickle=False)
        np.save(self.cache / "y_train.npy", self.y, allow_pickle=False)
        write_json(self.cache / "manifest.json", {"official_test_loaded": False, "fixture": True,
            "arrays": {name: {"sha256": runner._file_hash(self.cache / name), "shape": list(array.shape), "dtype": str(array.dtype)}
                       for name, array in (("X_train.npy", self.X), ("y_train.npy", self.y))}})
        self.protocol = {"schema_version": 1, "author": "Chang Cheng", "dataset": str(self.cache),
            "cv": 5, "native_threads": 1, "workers": 1, "validation_size_per_pair": 20,
            "validation_seed": 104729, "train_sizes": [40], "seeds": [43],
            "pairs": [{"id": "tshirt-shirt", "negative": 0, "positive": 6}],
            "grid": {"n_components": [2], "score_threshold": [0, 1000000], "n_neighbors": [3, 1]},
            "methods": ["screened_spa_absolute"], "max_iter": 2000, "tol": 1e-4,
            "profile_repeats": 3, "profile_order_seed": 42043}
        self.protocol_path = self.root / "protocol.json"
        self.output = self.root / "run"
        write_json(self.protocol_path, self.protocol)
        self.shape_patch = patch.object(runner, "EXPECTED_SHAPE", self.X.shape)
        self.shape_patch.start()
        self.addCleanup(self.shape_patch.stop)
        self.addCleanup(gc.collect)  # release traceback/mock-held Windows mmap handles before cleanup

    def initialize(self):
        return runner.initialize(self.protocol_path, self.output)

    def complete(self, config):
        records = [runner._worker((str(self.output), plan, chunk, config["sources"]))
                   for plan in config["plans"] for chunk in plan["chunks"]]
        return records, runner.aggregate_cv(self.output, config, require_complete=True)

    def test_fixed_balanced_validation_and_nested_training_samples(self):
        protocol = deepcopy(self.protocol)
        protocol.update(train_sizes=[20, 80, 40], seeds=[43, 44, 45])
        first = runner.make_splits(self.y, protocol)
        self.assertEqual(first, runner.make_splits(self.y, protocol))
        pair = first["tshirt-shirt"]
        validation = pair["validation_indices"]
        self.assertEqual(np.bincount((self.y[validation] == 6).astype(int)).tolist(), [10, 10])
        for seed in protocol["seeds"]:
            large = pair["training_indices"][f"n80-seed{seed}"]
            self.assertFalse(set(large) & set(validation))
            for size in protocol["train_sizes"]:
                actual = pair["training_indices"][f"n{size}-seed{seed}"]
                self.assertEqual(actual, large[:size])
                self.assertEqual(np.bincount((self.y[actual] == 6).astype(int)).tolist(), [size // 2] * 2)
                self.assertEqual(pair["training_indices_sha256"][f"n{size}-seed{seed}"], _digest(actual))
        self.assertNotEqual(pair["training_indices"]["n80-seed43"], pair["training_indices"]["n80-seed44"])

    def test_training_fold_preflight_retains_declared_and_excluded_grids(self):
        config = self.initialize()
        plan = config["plans"][0]
        self.assertEqual(len(plan["declared_candidates"]), 4)
        self.assertEqual(len(plan["candidates"]), 2)
        self.assertEqual(len(plan["chunks"]), 1)
        self.assertEqual(plan["chunks"][0]["grid"][0]["n_neighbors"], [3, 1])
        excluded = plan["excluded_representations"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["candidate_indices"], [2, 3])
        self.assertTrue(all(row["selected_features"] == 0 and row["training_size"] == 32 for row in excluded[0]["folds"]))
        record = config["preflight"][plan["job"]["array_key"]]
        self.assertTrue(all(row["screens"]["0"]["features"] == 15 for row in record["folds"]))
        excessive = runner._plan(plan["job"], {**self.protocol["grid"], "n_components": [16]}, record)
        self.assertFalse(excessive["chunks"])
        self.assertEqual(len(excessive["declared_candidates"]), 4)

    def test_cv_isolates_validation_shares_neighbors_and_resumes(self):
        config = self.initialize()
        load = runner._load_arrays
        def training_only(record):
            self.assertNotEqual(Path(record["path"]).name, "validation")
            return load(record)
        with patch.object(runner, "_load_arrays", training_only):
            records, first = self.complete(config)
            files = list(self.output.glob("*/c*/folds/*.json"))
            before = {path: path.read_bytes() for path in files}
            resumed, second = self.complete(self.initialize())
        self.assertEqual(sum(row["new_representation_fits"] for row in records), 20)
        self.assertEqual(sum(row["new_representation_fits"] for row in resumed), 0)
        self.assertEqual(sum(row["reused_representation_folds"] for row in resumed), 20)
        self.assertEqual(before, {path: path.read_bytes() for path in files})
        for job_id, report in first["jobs"].items():
            valid = [row for row in report["candidates"] if row["valid"]]
            self.assertTrue(valid)
            self.assertEqual(report["selected_candidate"], min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])))
            self.assertEqual(report["compute_seconds"], second["jobs"][job_id]["compute_seconds"])

    def test_no_validation_load_before_every_cv_chunk_finishes(self):
        config = self.initialize()
        with patch.object(runner, "_load_arrays", side_effect=AssertionError("Early validation load")):
            with self.assertRaisesRegex(ValueError, "All protocol CV"):
                runner.evaluate(self.output, config)
        self.assertFalse((self.output / "validation-gate.json").exists())

    def test_spawned_workers_finish_and_resume_without_new_chunks(self):
        config = self.initialize()
        result = runner.run_cv(self.output, config, workers=2)
        self.assertEqual(result["state"], "cv_completed")
        self.assertEqual(result["completed_chunks_this_call"], 4)
        resumed = runner.run_cv(self.output, self.initialize(), workers=2)
        self.assertEqual(resumed["completed_chunks_this_call"], 0)

    def test_serial_exact_profiles_metrics_and_resume(self):
        config = self.initialize()
        self.complete(config)
        result = runner.evaluate(self.output, config)
        self.assertEqual(result["valid_evaluations"], 4)
        self.assertEqual(result["models_with_valid_profiling"], 4)
        records = {}
        for plan in config["plans"]:
            path = self.output / plan["job"]["id"] / "evaluation.json"
            record = _read_receipt(path)
            records[path] = path.read_bytes()
            self.assertEqual(len(record["profiling"]), 3)
            self.assertTrue(all(row["exact_prediction_match"] for row in record["profiling"]))
            if plan["job"]["method"] == "screened_spa_absolute":
                self.assertEqual(record["training_stationarity"]["status"], "valid")
                self.assertTrue({"raw", "canonical"} <= set(record["training_stationarity"]["diagnostics"]))
            self.assertEqual(record["profile_medians_seconds"]["fit_plus_prediction_seconds"], float(np.median([
                row["timing"]["total_fit_seconds"] + row["timing"]["prediction_seconds"] for row in record["profiling"]])))
            self.assertTrue({"accuracy", "balanced_accuracy", "roc_auc", "average_precision", "confusion_matrix", "per_class"}
                            <= set(record["evaluation"]["metrics"]))
        with patch.object(runner, "_profile", side_effect=AssertionError("Repeated completed profile")):
            self.assertEqual(result, runner.evaluate(self.output, self.initialize()))
        self.assertEqual(records, {path: path.read_bytes() for path in records})

    def test_changed_protocol_data_and_corrupt_receipts_rejected(self):
        config = self.initialize()
        changed = deepcopy(self.protocol)
        changed["tol"] = 1e-5
        write_json(self.protocol_path, changed)
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            self.initialize()
        write_json(self.protocol_path, self.protocol)
        self.complete(config)
        plan = config["plans"][0]
        path = self.output / plan["job"]["id"] / "c0000" / "report.json"
        report = _read_receipt(path)
        report["candidates"][0]["mean_balanced_accuracy"] = 0.12345
        _write_receipt(path, report)
        with self.assertRaisesRegex(ValueError, "aggregate score differs"):
            runner.aggregate_cv(self.output, config)
        self.X[0, 0] += 0.01
        np.save(self.cache / "X_train.npy", self.X, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.initialize()

    def test_interrupted_profiling_reuses_original_and_completed_repeat(self):
        config = self.initialize()
        self.complete(config)
        original_profile = runner._profile
        calls = []
        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt("synthetic interruption")
            return original_profile(*args, **kwargs)
        with patch.object(runner, "_profile", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                runner.evaluate(self.output, config)
        first = self.output / config["evaluation_order"][0]
        saved = {path: path.read_bytes() for path in (first / "original-evaluation.json", first / "profile-0.json")}
        with patch.object(runner, "_profile", wraps=original_profile) as profiled:
            result = runner.evaluate(self.output, self.initialize())
        self.assertEqual(result["models_with_valid_profiling"], 4)
        self.assertEqual(profiled.call_count, 11)
        profiled.reset_mock()
        self.assertEqual(saved, {path: path.read_bytes() for path in saved})

    def test_numerical_nonconvergence_is_invalid_and_not_replaced(self):
        self.protocol.update(methods=["screened_mu_absolute"], max_iter=1)
        self.protocol["grid"]["score_threshold"] = [0]
        write_json(self.protocol_path, self.protocol)
        config = self.initialize()
        _, cv = self.complete(config)
        selected = cv["jobs"][config["plans"][0]["job"]["id"]]
        self.assertIsNone(selected["selected_parameters"])
        self.assertTrue(all(row["invalid_fold_count"] == 5 and not row["valid"] for row in selected["candidates"]))
        self.assertTrue(all(fold["status"] == "nonconverged" for row in selected["candidates"] for fold in row["folds"]))
        self.assertFalse(selected["is_fallback"])

    def test_zero_eligible_representations_are_held_without_fallback(self):
        self.protocol["grid"].update(n_components=[20], score_threshold=[1000000])
        write_json(self.protocol_path, self.protocol)
        config = self.initialize()
        self.complete(config)
        result = runner.evaluate(self.output, config)
        self.assertEqual(result["valid_evaluations"], 1)  # raw KNN still has a valid representation
        for plan in config["plans"]:
            if plan["job"]["method"] != "raw_knn":
                record = _read_receipt(self.output / plan["job"]["id"] / "evaluation.json")
                self.assertEqual(record["evaluation"]["status"], "held")
                self.assertFalse(record["is_fallback"])

    def test_mu_pipeline_local_profile_reproduces_original(self):
        job = {"method": "screened_mu_absolute", "seed": 43, "max_iter": 2000, "tol": 1e-4}
        model = runner._estimator(job).set_params(nmf__n_components=2, screen__score_threshold=0, knn__n_neighbors=3)
        y = (self.y == 6).astype(np.uint8)
        with threadpool_limits(limits=1):
            model.fit(self.X, y)
            original = evaluate_classifier(model, self.X[:12], y[:12], threads=1, sample_ids=list(range(12)))
            profile = runner._profile(model, self.X, y, self.X[:12], y[:12], reference=original, sample_ids=list(range(12)))
        self.assertEqual(profile["status"], "valid")
        self.assertTrue(profile["exact_prediction_match"])
        self.assertEqual(profile["stage_calls"]["decomposition_fit_seconds"], 1)
        self.assertEqual(profile["stage_calls"]["decomposition_projection_seconds"], 1)
        self.assertEqual(profile["diagnostics"]["basis_dtype"], "float32")

    def test_bpp_factory_profile_and_diagnostics_preserve_coordinates(self):
        job = {"method": "screened_anls_bpp_absolute", "seed": 43, "max_iter": 20000, "tol": 1e-4,
               "anls_max_iter": 1000, "anls_nnls_max_iter": None, "anls_nnls_tol": 1e-10}
        model = runner._estimator(job).set_params(nmf__n_components=2, screen__score_threshold=0, knn__n_neighbors=3)
        self.assertEqual(model.named_steps["nmf"].max_iter, 1000)
        self.assertEqual(model.named_steps["nmf"].nnls_tol, 1e-10)
        y = (self.y == 6).astype(np.uint8)
        with threadpool_limits(limits=1):
            model.fit(self.X, y)
            H = model.named_steps["nmf"].components_.copy()
            W = model.named_steps["knn"]._fit_X.copy()
            stationarity = runner._training_stationarity(model, self.X)
            original = evaluate_classifier(model, self.X[:12], y[:12], threads=1, sample_ids=list(range(12)))
            profile = runner._profile(model, self.X, y, self.X[:12], y[:12], reference=original, sample_ids=list(range(12)))
        np.testing.assert_array_equal(model.named_steps["nmf"].components_, H)
        np.testing.assert_array_equal(model.named_steps["knn"]._fit_X, W)
        self.assertEqual(stationarity["status"], "valid")
        self.assertEqual(profile["status"], "valid")
        self.assertTrue(profile["exact_prediction_match"])
        self.assertEqual(profile["stage_calls"]["decomposition_fit_seconds"], 1)
        self.assertEqual(profile["stage_calls"]["decomposition_projection_seconds"], 1)
        self.assertEqual(profile["diagnostics"]["basis_dtype"], "float64")
        model.named_steps["nmf"].converged_ = False
        self.assertTrue(runner._model_nonconverged(model, []))


if __name__ == "__main__":
    unittest.main()
