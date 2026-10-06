"""Check explicit-study coverage, training isolation, strict resume and profiling."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import celeba_default_study as runner
from supervised_nmf.evaluation import _digest, data_fingerprint, write_json
from supervised_nmf.study import _read_receipt, _write_receipt


class DefaultStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nmf-default-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        random = np.random.RandomState(23)
        manifest = {"author": "Chang Cheng", "seed": 42, "official_test_loaded": False, "splits": {}}
        for split, n, partition in (("train", 50, 0), ("validation", 20, 1)):
            y = np.arange(n, dtype=np.int64) % 2
            X = random.uniform(0.1, 0.3, (n, 4096)).astype(np.float32)
            X[:, :100] += (y[:, None] * 0.4).astype(np.float32)
            np.save(self.cache / f"{split}-X.npy", X, allow_pickle=False)
            np.save(self.cache / f"{split}-Smiling-y.npy", y, allow_pickle=False)
            manifest["splits"][split] = {"metadata": {"partition_code": partition},
                                        "image_ids": [f"{split}-{i}.jpg" for i in range(n)],
                                        "attributes": {"Smiling": {"class_counts": [n // 2, n // 2],
                                                                    "data_sha256": data_fingerprint(X, y)}}}
        write_json(self.cache / "manifest.json", manifest)
        self.job = {"id": "spa-pct", "cache": str(self.cache), "attribute": "Smiling", "seed": 42,
                    "method": "screened_spa_percentile", "max_iter": 2000, "tol": 1e-4,
                    "grid": [{"n_components": [2], "percentile": [2], "n_neighbors": [3, 1]},
                             {"n_components": [2, 3], "percentile": [2], "n_neighbors": [1]}]}
        self.protocol = {"schema_version": 1, "author": "Chang Cheng", "cv": 5, "native_threads": 1,
                         "profile_repeats": 3, "jobs": [self.job]}
        self.protocol_path = self.root / "protocol.json"
        write_json(self.protocol_path, self.protocol)
        self.output = self.root / "run"

    def initialize(self):
        return runner.initialize(self.protocol_path, self.output)

    def complete(self, configuration):
        results = []
        for plan in configuration["plans"]:
            for chunk in plan["chunks"]:
                results.append(runner._worker((str(self.output), plan, chunk, configuration["sources"])))
        return results, runner.aggregate_cv(self.output, configuration, require_complete=True)

    def test_boxes_share_only_exact_representations_and_canonical_ties(self):
        config = self.initialize()
        plan = config["plans"][0]
        self.assertEqual(plan["duplicate_box_candidates_removed"], 1)
        self.assertEqual(len(plan["candidates"]), 3)
        self.assertEqual(len(plan["chunks"]), 2)
        self.assertEqual([row["n_neighbors"] for row in plan["candidates"]], [3, 1, 1])
        workers, summary = self.complete(config)
        self.assertEqual(sum(row["new_representation_fits"] for row in workers), 10)
        report = summary["jobs"]["spa-pct"]
        self.assertEqual([row["parameters"] for row in report["candidates"]], plan["candidates"])
        valid = [row for row in report["candidates"] if row["valid"]]
        self.assertTrue(valid)
        self.assertEqual(report["selected_candidate"], min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])))
        self.assertEqual(report["tuning_compute_seconds"], sum(report["compute_seconds"].values()))

    def test_cv_does_not_open_validation_arrays_and_resume_preserves_fold_receipts(self):
        original_load = np.load
        def training_only(path, *args, **kwargs):
            self.assertNotIn("validation-", Path(path).name)
            return original_load(path, *args, **kwargs)
        with patch.object(runner.np, "load", training_only):
            config = self.initialize()
            _, first = self.complete(config)
            files = list(self.output.glob("*/c*/folds/*.json"))
            before = {path: path.read_bytes() for path in files}
            workers, second = self.complete(self.initialize())
        self.assertEqual(len(before), 10)
        self.assertEqual(before, {path: path.read_bytes() for path in files})
        self.assertEqual(sum(row["new_representation_fits"] for row in workers), 0)
        self.assertEqual(sum(row["reused_representation_folds"] for row in workers), 10)
        self.assertEqual(first["jobs"]["spa-pct"]["selected_parameters"], second["jobs"]["spa-pct"]["selected_parameters"])
        self.assertEqual(first["jobs"]["spa-pct"]["compute_seconds"], second["jobs"]["spa-pct"]["compute_seconds"])

    def test_evaluation_requires_all_cv_before_reading_validation(self):
        config = self.initialize()
        with patch.object(runner, "_cache_identity", side_effect=AssertionError("Validation accessed too soon")):
            with self.assertRaisesRegex(ValueError, "All protocol CV jobs"):
                runner.evaluate(self.output, config)
        self.assertFalse((self.output / "comparison.json").exists())

    def test_three_serial_profiles_reproduce_predictions_and_evaluation_resume(self):
        config = self.initialize()
        self.complete(config)
        result = runner.evaluate(self.output, config)
        self.assertEqual(result["valid_evaluations"], 1)
        self.assertEqual(result["exact_profiles"], 3)
        path = self.output / "spa-pct" / "evaluation.json"
        record = _read_receipt(path)
        self.assertTrue(record["profiling_valid"])
        self.assertEqual(len(record["profiling"]), 3)
        expected = float(np.median([row["timing"]["total_fit_seconds"] + row["timing"]["prediction_seconds"]
                                    for row in record["profiling"]]))
        self.assertEqual(record["profile_medians_seconds"]["fit_plus_prediction_seconds"], expected)
        before = path.read_bytes()
        with patch.object(runner, "profile_selected_estimator", side_effect=AssertionError("Refitted saved result")):
            resumed = runner.evaluate(self.output, self.initialize())
        self.assertEqual(resumed, result)
        self.assertEqual(before, path.read_bytes())

    def test_configuration_and_cache_mismatch_rejected(self):
        self.initialize()
        changed = deepcopy(self.protocol)
        changed["jobs"][0]["tol"] = 1e-5
        write_json(self.protocol_path, changed)
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            self.initialize()
        write_json(self.protocol_path, self.protocol)
        X = np.load(self.cache / "train-X.npy")
        X[0, 0] += 0.01
        np.save(self.cache / "train-X.npy", X, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.initialize()

    def test_corrupt_or_fabricated_cv_score_rejected(self):
        config = self.initialize()
        self.complete(config)
        path = self.output / "spa-pct" / "c0000" / "report.json"
        report = _read_receipt(path)
        report["candidates"][0]["mean_balanced_accuracy"] = 0.314159
        _write_receipt(path, report)
        with self.assertRaisesRegex(ValueError, "aggregate score differs"):
            runner.aggregate_cv(self.output, config)
        envelope = json.loads(path.read_text())
        envelope["sha256"] = "corrupt"
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "integrity check"):
            runner.aggregate_cv(self.output, config)

    def test_infeasible_candidate_remains_failed_without_fallback(self):
        self.protocol["jobs"][0]["grid"] = [{"n_components": [5000], "percentile": [2], "n_neighbors": [1, 3]}]
        write_json(self.protocol_path, self.protocol)
        config = self.initialize()
        _, summary = self.complete(config)
        report = summary["jobs"]["spa-pct"]
        self.assertEqual(len(report["candidates"]), 2)
        self.assertTrue(all(not row["valid"] and row["invalid_fold_count"] == 5 for row in report["candidates"]))
        self.assertIsNone(report["selected_parameters"])
        result = runner.evaluate(self.output, config)
        self.assertEqual(result["valid_evaluations"], 0)
        held = _read_receipt(self.output / "spa-pct" / "evaluation.json")
        self.assertEqual(held["evaluation"]["status"], "held")
        self.assertFalse(held["is_fallback"])
        self.assertEqual(held["profiling"], [])


if __name__ == "__main__":
    unittest.main()
