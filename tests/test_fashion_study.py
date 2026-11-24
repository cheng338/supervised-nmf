"""Synthetic checks for Chang Cheng's training-only Fashion benchmark runner."""

from copy import deepcopy
from datetime import datetime, timezone
import gc
from io import BytesIO
from pathlib import Path
from shutil import copytree
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import BadZipFile

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fashion_mnist_study as runner
from supervised_nmf.evaluation import _digest, evaluate_classifier, write_json
from supervised_nmf.study import _cv_report_pin, _read_receipt, _write_receipt


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

    def test_failed_fashion_worker_allows_cache_replacement(self):
        config = self.initialize()
        plan = config["plans"][0]
        plan["job"]["training_data_sha256"] = "0" * 64
        feature_file = Path(plan["job"]["cache"]) / "X.npy"
        expected = np.load(feature_file, allow_pickle=False)[:1]
        replacement = self.root / "replacement.npy"
        np.save(replacement, np.zeros_like(expected))
        fingerprint = runner.data_fingerprint
        view = None

        def keep_view(X, y):
            nonlocal view
            view = X[:1]
            return fingerprint(X, y)

        with patch.object(runner, "data_fingerprint", keep_view):
            try:
                runner._worker((str(self.output), plan, plan["chunks"][0], config["sources"]))
            except ValueError as error:
                self.assertIn("Worker training data differ", str(error))
                retained = error.__traceback__
            else:
                self.fail("Worker accepted a different training fingerprint")
        try:
            self.assertIsNotNone(retained)
            replacement.replace(feature_file)
            self.assertIsNotNone(view)
            self.assertFalse(view.flags.writeable)
            owner = view
            while isinstance(owner, np.ndarray):
                mapping = getattr(owner, "_mmap", None)
                if mapping is not None:
                    self.assertFalse(mapping.closed, "Worker closed a map still backing the view")
                owner = owner.base
            np.testing.assert_array_equal(view, expected)
        finally:
            del retained, view

    @unittest.skipUnless(sys.platform == "win32", "Windows worker file ownership")
    def test_worker_releases_archive_under_npy_with_live_traceback(self):
        config = self.initialize()
        plan = config["plans"][0]
        chunk = plan["chunks"][0]
        with BytesIO() as archive:
            np.savez(archive, values=[1, 2])
            encoded = archive.getvalue()
        archives = (("npz", encoded, ValueError), ("truncated_zip", encoded[:6], BadZipFile))

        for filename in ("X.npy", "y.npy"):
            for kind, contents, expected_error in archives:
                with self.subTest(filename=filename, kind=kind):
                    cache = copytree(plan["job"]["cache"], self.root / f"{filename}-{kind}")
                    job = {**plan["job"], "cache": str(cache)}
                    output = cache / "run"
                    target = cache / filename
                    original = target.read_bytes()
                    replacement = cache / "replacement.npy"
                    replacement.write_bytes(original)
                    target.write_bytes(contents)

                    try:
                        runner._worker((str(output), {"job": job}, chunk, config["sources"]))
                    except Exception as error:
                        held = error
                    else:
                        self.fail("Worker accepted an archive in the training cache")
                    try:
                        self.assertIsNotNone(held.__traceback__)
                        self.assertFalse((output / job["id"] / chunk["id"]).exists())
                        replacement.replace(target)
                        self.assertEqual(target.read_bytes(), original)
                        self.assertIsInstance(held, expected_error)
                        if kind == "npz":
                            self.assertIn(filename, str(held))
                    finally:
                        del held

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
        def training_only(record, **options):
            self.assertNotEqual(Path(record["path"]).name, "validation")
            return load(record, **options)
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

    def test_rebuilt_report_retains_original_worker_pool_order(self):
        config = self.initialize()
        plan = config["plans"][0]
        chunk = plan["chunks"][0]
        work = (str(self.output), plan, chunk, config["sources"])
        directory = self.output / plan["job"]["id"] / chunk["id"]
        report = directory / "report.json"
        pools = [{"internal_api": "mkl", "num_threads": 1}, {"internal_api": "openmp", "num_threads": 1}]
        with patch("supervised_nmf.study.threadpool_info", return_value=pools):
            runner._worker(work)
        self.assertEqual(_read_receipt(report)["active_threadpools"], pools)
        pin = _cv_report_pin(directory)
        original = report.read_bytes()
        report.unlink()
        with patch("supervised_nmf.study.threadpool_info", return_value=pools[::-1]):
            replay = runner._worker(work, reuse_only=True, _frozen_report=pin)
        self.assertEqual(replay["new_representation_fits"], 0)
        self.assertEqual(replay["reused_representation_folds"], 5)
        self.assertEqual(report.read_bytes(), original)

    def test_spawned_workers_finish_and_resume_without_new_chunks(self):
        config = self.initialize()
        argv = ["fashion", "--protocol", str(self.protocol_path), "--output", str(self.output), "--mode", "all"]
        with (patch.object(sys, "argv", argv),
              patch.object(runner, "ProcessPoolExecutor", side_effect=KeyboardInterrupt),
              self.assertRaises(KeyboardInterrupt)):
            runner.main()
        first, = [_read_receipt(path) for path in (self.output / "cv-timing").glob("*.json")]
        self.assertEqual(first["state"], "interrupted")
        self.assertGreater(first["cv_wall_seconds_this_invocation"], 0.)
        result = runner.run_cv(self.output, config, workers=2)
        self.assertEqual(result["state"], "cv_completed")
        self.assertEqual(result["completed_chunks_this_call"], 4)
        self.assertEqual(result["cv_wall_seconds_cumulative"], first["cv_wall_seconds_this_invocation"]
                         + result["cv_wall_seconds_this_invocation"])
        def evaluate_after_cv(*unused):
            latest = max((_read_receipt(path) for path in (self.output / "cv-timing").glob("*.json")),
                         key=lambda row: row["started_utc"])
            self.assertEqual(latest["state"], "completed")
            self.assertEqual(latest["cv_wall_seconds_this_invocation"], 0.)
            raise RuntimeError("evaluation interruption")
        with (patch.object(sys, "argv", argv), patch.object(runner, "evaluate", evaluate_after_cv),
              self.assertRaisesRegex(RuntimeError, "evaluation interruption")):
            runner.main()
        resumed = runner.run_cv(self.output, self.initialize(), workers=2)
        self.assertEqual(resumed["completed_chunks_this_call"], 0)
        self.assertEqual(resumed["cv_wall_seconds_this_invocation"], 0.)
        self.assertEqual(resumed["cv_wall_seconds_cumulative"], result["cv_wall_seconds_cumulative"])

        with (patch.object(runner, "_verify_arrays", side_effect=RuntimeError("after gate")),
              self.assertRaisesRegex(RuntimeError, "after gate")):
            runner.evaluate(self.output, config)
        plan = config["plans"][0]
        directory = self.output / plan["job"]["id"] / plan["chunks"][0]["id"]
        report = directory / "report.json"
        original, gate = report.read_bytes(), (self.output / "validation-gate.json").read_bytes()
        report.unlink()
        replay = runner.run_cv(self.output, config, workers=1)
        self.assertEqual(replay["cv_wall_seconds_this_invocation"], 0.)
        self.assertEqual(report.read_bytes(), original)
        self.assertEqual((self.output / "validation-gate.json").read_bytes(), gate)
        evaluate_argv = argv[:-1] + ["evaluate"]
        report.unlink()
        with patch.object(sys, "argv", evaluate_argv):
            runner.main()
        self.assertEqual(report.read_bytes(), original)
        self.assertEqual((self.output / "validation-gate.json").read_bytes(), gate)
        contents = _read_receipt(report)
        # A valid checksum on an incomplete derived report cannot authorize refitting.
        fold = contents["representation_folds"].pop()
        (directory / "folds" / f"{fold['key']}.json").unlink()
        attempts = set((directory / "attempts").iterdir())
        with (patch.object(sys, "argv", evaluate_argv), patch.object(runner, "aggregate_cv") as summarize,
              self.assertRaisesRegex(ValueError, "fresh CV is closed")):
            runner.main()
        summarize.assert_not_called()
        _write_receipt(report, contents)
        with self.assertRaisesRegex(ValueError, "fresh CV is closed"):
            runner.run_cv(self.output, config, workers=1)
        self.assertEqual(set((directory / "attempts").iterdir()), attempts)
        self.assertFalse((directory / "folds" / f"{fold['key']}.json").exists())

    def test_source_inventory_is_stable_and_restores_thread_limits(self):
        expected = runner._sources()
        with threadpool_limits(limits=2):
            before = {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}
            self.assertEqual(runner._sources(), expected)
            after = {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}
            self.assertEqual(after, before)
        self.assertEqual(runner._sources(), expected)

    def test_source_guard_rejects_changed_library_and_package_identities(self):
        expected = runner._sources()
        self.assertTrue(expected["runtime"]["numerical_runtime"])
        altered_runtime = deepcopy(expected)
        altered_runtime["runtime"]["numerical_runtime"][0]["version"] = "different-runtime"
        altered_source = deepcopy(expected)
        name = next(iter(altered_source["package_source_sha256"]))
        altered_source["package_source_sha256"][name] = "0" * 64
        for altered in (altered_runtime, altered_source):
            with self.subTest(identity=altered):
                with self.assertRaisesRegex(ValueError, "Source or numerical runtime changed"):
                    runner._check_source(altered)

    def test_serial_exact_profiles_metrics_and_resume(self):
        config = self.initialize()
        self.complete(config)
        stationarity = runner._training_stationarity
        between_phases = []
        def observe_stationarity(*args):
            between_phases.append(datetime.now(timezone.utc).isoformat())
            return stationarity(*args)
        with patch.object(runner, "_training_stationarity", observe_stationarity):
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
            middle = between_phases[config["evaluation_order"].index(plan["job"]["id"])]
            self.assertLessEqual(record["fit_execution"]["finished_utc"], middle)
            self.assertLessEqual(middle, record["prediction_execution"]["started_utc"])
            for row in [record, *record["profiling"]]:
                self.assertEqual(row["fit_execution"]["seconds"],
                                 row["final_fit_seconds"] if row is record else row["timing"]["total_fit_seconds"])
                self.assertLessEqual(row["fit_execution"]["finished_utc"], row["prediction_execution"]["started_utc"])
                self.assertEqual(row["prediction_execution"]["seconds"], row["evaluation"]["prediction_seconds"])
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

    def test_rechecksummed_identity_cannot_change_schema_scoring_or_configuration(self):
        config = self.initialize()
        self.complete(config)
        path = self.output / config["plans"][0]["job"]["id"] / "c0000" / "report.json"
        original = _read_receipt(path)
        self.assertEqual(original["scoring"], "balanced_accuracy")
        for target, field, value in (
                ("report", "scoring", "accuracy"),
                ("identity", "scoring", "accuracy"),
                ("calibration", "scoring", "accuracy"),
                ("identity", "schema_version", 3),
                ("calibration", "identity_schema", 1),
                ("calibration", "structural_configuration", {}),
                ("calibration", "implementation_source", {})):
            with self.subTest(target=target, field=field):
                report = deepcopy(original)
                identity = report["identity"]
                records = {"report": report, "identity": identity,
                           "calibration": identity["calibration"]}
                records[target][field] = value
                report["fingerprint"] = _digest(identity)
                _write_receipt(path.parent / "identity.json", identity)
                _write_receipt(path, report)
                with self.assertRaisesRegex(ValueError, "CV checkpoint identity or grid coverage differs"):
                    runner.aggregate_cv(self.output, config)
        _write_receipt(path.parent / "identity.json", original["identity"])
        _write_receipt(path, original)
        runner.aggregate_cv(self.output, config, require_complete=True)

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
        self.assertEqual(profile["stage_calls"]["training_projection_seconds"], 0)
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
        self.assertEqual(profile["stage_calls"]["training_projection_seconds"], 0)
        self.assertEqual(profile["stage_calls"]["decomposition_projection_seconds"], 1)
        self.assertEqual(profile["diagnostics"]["basis_dtype"], "float64")
        model.named_steps["nmf"].converged_ = False
        self.assertTrue(runner._model_nonconverged(model, []))


    def test_cd_pipeline_local_profile_counts_projection_once(self):
        job = {"method": "screened_mu_absolute", "seed": 43, "max_iter": 2000, "tol": 1e-4}
        model = runner._estimator(job).set_params(
            nmf__solver="cd", nmf__n_components=2, screen__score_threshold=0, knn__n_neighbors=3)
        y = (self.y == 6).astype(np.uint8)
        original_transform = runner.NMF.transform
        with threadpool_limits(limits=1):
            model.fit(self.X, y)
            original = evaluate_classifier(model, self.X[:12], y[:12], threads=1, sample_ids=list(range(12)))
            profile = runner._profile(model, self.X, y, self.X[:12], y[:12],
                                      reference=original, sample_ids=list(range(12)))
        self.assertIs(runner.NMF.transform, original_transform)
        self.assertEqual(profile["status"], "valid", profile)
        self.assertTrue(profile["exact_prediction_match"])
        self.assertEqual(profile["stage_calls"]["decomposition_fit_seconds"], 1)
        self.assertEqual(profile["stage_calls"]["training_projection_seconds"], 0)
        self.assertEqual(profile["stage_calls"]["decomposition_projection_seconds"], 1)

    def test_classifier_mu_profile_keeps_native_transform_identity(self):
        from supervised_nmf import classifier as classifier_module
        from supervised_nmf.classifier import NMFClassifier

        model = NMFClassifier(
            solver="mu", mu_projection="batch", training_coordinates="transform",
            screening="absolute", score_threshold=0, n_components=2, n_neighbors=3,
            max_iter=2000, tol=1e-4, random_state=43)
        y = (self.y == 6).astype(np.uint8)
        original_transform = runner.NMF.transform
        dispatch_code = classifier_module.transform_mu.__code__
        observed = []
        previous = sys.getprofile()
        self.assertIsNone(previous)

        def observe(frame, event, value):
            if event == "call" and frame.f_code is dispatch_code:
                observed.append(runner.NMF.transform is original_transform)

        with threadpool_limits(limits=1):
            model.fit(self.X, y)
            original = evaluate_classifier(model, self.X[:12], y[:12], threads=1, sample_ids=list(range(12)))
            try:
                sys.setprofile(observe)
                profile = runner._profile(model, self.X, y, self.X[:12], y[:12],
                                          reference=original, sample_ids=list(range(12)))
            finally:
                sys.setprofile(previous)
        self.assertIs(runner.NMF.transform, original_transform)
        self.assertEqual(observed, [True, True])  # training and query dispatch
        self.assertEqual(profile["status"], "valid", profile)
        self.assertTrue(profile["exact_prediction_match"])
        self.assertEqual(profile["instrumented_method_calls"]["decomposition_fit_seconds"], 1)
        self.assertEqual(profile["instrumented_method_calls"]["training_projection_seconds"], 1)
        self.assertEqual(profile["instrumented_method_calls"]["decomposition_projection_seconds"], 1)

    def test_pipeline_projection_errors_restore_profiled_methods(self):
        for solver in ("mu", "cd", "anls_bpp"):
            with self.subTest(solver=solver), threadpool_limits(limits=1):
                job = {"method": "screened_anls_bpp_absolute" if solver == "anls_bpp" else "screened_mu_absolute",
                       "seed": 43, "max_iter": 2000, "tol": 1e-4, "anls_max_iter": 1000}
                model = runner._estimator(job).set_params(
                    nmf__n_components=2, screen__score_threshold=0, knn__n_neighbors=3)
                if solver == "cd":
                    model.set_params(nmf__solver="cd")
                y = (self.y == 6).astype(np.uint8)
                model.fit(self.X, y)
                original = evaluate_classifier(model, self.X[:12], y[:12], threads=1, sample_ids=list(range(12)))
                decomposition_type = type(model.named_steps["nmf"])
                watched = [(cls, method) for cls in {runner.NMF, decomposition_type}
                           for method in ("fit_transform", "transform")]
                originals = {(cls, method): getattr(cls, method) for cls, method in watched}

                def fail_projection(*args, **kwargs):
                    raise RuntimeError("required projection error")

                with patch.object(decomposition_type, "transform", fail_projection):
                    patched = {(cls, method): getattr(cls, method) for cls, method in watched}
                    profile = runner._profile(model, self.X, y, self.X[:12], y[:12],
                                              reference=original, sample_ids=list(range(12)))
                    for (cls, method), expected in patched.items():
                        self.assertIs(getattr(cls, method), expected)
                for (cls, method), expected in originals.items():
                    self.assertIs(getattr(cls, method), expected)
                self.assertEqual(profile["status"], "invalid_reproduction", profile)
                self.assertFalse(profile["exact_prediction_match"])
                self.assertEqual(profile["evaluation"]["status"], "failed")
                self.assertEqual(profile["evaluation"]["error_type"], "RuntimeError")
                self.assertEqual(profile["evaluation"]["error"], "required projection error")
                self.assertEqual(profile["stage_calls"]["decomposition_projection_seconds"], 1)

if __name__ == "__main__":
    unittest.main()
