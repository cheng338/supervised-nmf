"""Matched CelebA adapter checks. Author: Chang Cheng. No NMF fits or real data."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_snmf_test_celeba_four_solver", ROOT / "scripts/celeba_four_solver_study.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def protocol_fixture(cache):
    return {"schema_version": 1, "author": "Chang Cheng", "cache": str(cache),
            "cache_manifest_sha256": runner.shared._file_hash(cache / "manifest.json"),
            "attributes": list(runner.ATTRIBUTES), "methods": list(runner.METHODS),
            "seed": 43, "n_train": 30, "n_validation": 12, "cv": 5, "native_threads": 1,
            "workers": 1, "max_iter": 20000, "tol": 1e-4, "profile_repeats": 3,
            "anls_max_iter": 1000, "anls_nnls_max_iter": None, "anls_nnls_tol": 1e-10,
            "profile_order_seed": 42043, "grid": {"n_components": [2], "score_threshold": [0], "n_neighbors": [1, 3]},
            "extra_representations": {}, "screening": "absolute"}


def cache_fixture(cache):
    cache.mkdir()
    random = np.random.RandomState(7)
    manifest = {"seed": 43, "official_test_loaded": False, "array_sha256": {}, "splits": {}}
    for split, size, partition in (("train", 30, 0), ("validation", 12, 1)):
        X = random.uniform(.1, .9, size=(size, runner.N_FEATURES)).astype(np.float32)
        np.save(cache / f"{split}-X.npy", X)
        description = {"image_ids": [f"{split}-{i}.jpg" for i in range(size)],
                       "metadata": {"partition_code": partition}, "attributes": {}}
        for attribute in runner.ATTRIBUTES:
            y = (np.arange(size) % (2 if attribute == "Smiling" else 3) == 0).astype(np.int64)
            name = f"{split}-{attribute}-y.npy"
            np.save(cache / name, y)
            description["attributes"][attribute] = {"class_counts": np.bincount(y, minlength=2).tolist(),
                                                    "data_sha256": runner.data_fingerprint(X, y)}
            manifest["array_sha256"][name] = runner.shared._file_hash(cache / name)
        manifest["array_sha256"][f"{split}-X.npy"] = runner.shared._file_hash(cache / f"{split}-X.npy")
        manifest["splits"][split] = description
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def fake_preflight(thresholds, *, features=4096):
    return {"fold_indices_sha256": "synthetic-folds", "folds": [
        {"fold": fold, "training_size": 6400, "screens": {
            str(threshold): {"features": features, "mask_sha256": "fixture"} for threshold in thresholds}}
        for fold in range(5)]}


class CelebaFourSolverTests(unittest.TestCase):
    def test_common_grid_and_incumbents_have_identical_budgets(self):
        grid = {"n_components": [2, 5, 10, 20, 40, 80], "score_threshold": [0, 1, 2, 4, 8, 16, 32],
                "n_neighbors": [1, 2, 3, 5, 9, 15, 25, 41, 65, 81]}
        extras = [{"n_components": 12, "score_threshold": 9.899494936611665},
                  {"n_components": 80, "score_threshold": 11.313708498984761}]
        masks = fake_preflight(grid["score_threshold"] + [row["score_threshold"] for row in extras])
        normalized = []
        for method in runner.METHODS:
            plan = runner.plan_with_extras({"method": method}, grid, extras, masks)
            self.assertEqual(len(plan["declared_candidates"]), 440)
            self.assertEqual(len(plan["chunks"]), 44)
            self.assertEqual([row["index"] for row in plan["candidates"]], list(range(440)))
            normalized.append([{key.rsplit("__", 1)[-1]: value for key, value in row.items()}
                               for row in plan["declared_candidates"]])
        self.assertTrue(all(rows == normalized[0] for rows in normalized))

    def test_duplicate_extra_pairs_deduplicate_without_reordering(self):
        grid = {"n_components": [2], "score_threshold": [0], "n_neighbors": [1, 3]}
        extras = [{"n_components": 2, "score_threshold": 0}, {"n_components": 3, "score_threshold": 0},
                  {"n_components": 3, "score_threshold": 0}]
        plan = runner.plan_with_extras({"method": runner.METHODS[0]}, grid, extras, fake_preflight([0]))
        self.assertEqual(len(plan["declared_candidates"]), 4)
        self.assertEqual([chunk["candidate_indices"] for chunk in plan["chunks"]], [[0, 1], [2, 3]])

    def test_dimensional_exclusions_preserve_declared_candidate_indices(self):
        grid = {"n_components": [2, 5], "score_threshold": [0], "n_neighbors": [1, 3]}
        plan = runner.plan_with_extras({"method": runner.METHODS[0]}, grid, [], fake_preflight([0], features=3))
        self.assertEqual(len(plan["declared_candidates"]), 4)
        self.assertEqual([row["index"] for row in plan["candidates"]], [0, 1])
        self.assertEqual(plan["excluded_representations"][0]["candidate_indices"], [2, 3])
        self.assertEqual(len(plan["excluded_representations"][0]["folds"]), 5)

    def test_initialize_and_resume_never_open_validation_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_fixture(root / "cache")
            protocol = protocol_fixture(root / "cache")
            source = root / "protocol.json"
            source.write_text(json.dumps(protocol), encoding="utf-8")
            real_load, real_hash = runner.np.load, runner.shared._file_hash
            def guarded_load(path, *args, **kwargs):
                self.assertNotIn("validation-", str(path))
                return real_load(path, *args, **kwargs)
            def guarded_hash(path):
                self.assertNotIn("validation-", str(path))
                return real_hash(path)
            with patch.object(runner.np, "load", side_effect=guarded_load), patch.object(runner.shared, "_file_hash", side_effect=guarded_hash):
                first = runner.initialize(source, root / "run")
                second = runner.initialize(source, root / "run")
            self.assertEqual(first, second)
            self.assertEqual(len(first["plans"]), 8)
            self.assertEqual(len(first["evaluation_order"]), 8)
            self.assertIn(str(ROOT / "scripts/celeba_four_solver_study.py"), first["sources"]["extra_sources_sha256"])
            for plan in first["plans"]:
                self.assertEqual(plan["job"]["training_array_record"]["filenames"]["X"], "train-X.npy")
                self.assertEqual(plan["job"]["anls_max_iter"], 1000)
                self.assertIsNone(plan["job"]["anls_nnls_max_iter"])
                self.assertEqual(plan["job"]["anls_nnls_tol"], 1e-10)
            self.assertEqual(first["evaluation_role"], "official_celeba_validation_natural_class_proportions")

    def test_incomplete_cv_blocks_validation_reads(self):
        with patch.object(runner.shared, "aggregate_cv", side_effect=ValueError("incomplete")), \
             patch.object(runner, "_verify_values") as verify, patch.object(runner.shared, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.evaluate("unused", {"arrays": {}})
            verify.assert_not_called()
            evaluate.assert_not_called()

    def test_completed_gate_checks_validation_before_shared_evaluation(self):
        events = []
        configuration = {"arrays": {"smiling/train": {"fixture": "train"}, "smiling/validation": {"fixture": "validation"}}}
        with patch.object(runner.shared, "aggregate_cv", side_effect=lambda *a, **k: events.append("gate")), \
             patch.object(runner, "_verify_values", side_effect=lambda *a, **k: events.append("validation")), \
             patch.object(runner.shared, "evaluate", side_effect=lambda *a, **k: events.append("evaluate")):
            runner.evaluate("unused", configuration)
        self.assertEqual(events, ["gate", "validation", "evaluate"])

    def test_changed_training_arrays_and_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            manifest = cache_fixture(cache)
            protocol = protocol_fixture(cache)
            record = runner._array_record(cache, manifest, "train", "Smiling")
            X = np.load(cache / "train-X.npy")
            X[0, 0] += np.float32(.01)
            np.save(cache / "train-X.npy", X)
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner._verify_values(record, training=True)
            manifest["seed"] = 44
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                runner._cache_manifest(protocol)

    def test_protocol_rejects_missing_solver_or_wrong_screening(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache_fixture(cache)
            original = protocol_fixture(cache)
            runner.validate_protocol(original)
            for key, value in (("methods", original["methods"][:3]), ("screening", "percentile"), ("profile_repeats", 1)):
                protocol = {**deepcopy(original), key: value}
                with self.assertRaises(ValueError):
                    runner.validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
