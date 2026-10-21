"""Receipt migration and launch policy checks. Author: Chang Cheng.

These fixtures perform no estimator fits, native crash replays or data downloads.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_snmf_test_recovery", ROOT / "scripts/prepare_four_solver_recovery.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
shared = recovery.shared
orchestration = recovery.orchestration
orchestration.suppress_crash_dialogs()
report_spec = importlib.util.spec_from_file_location("_snmf_recovery_report", ROOT / "scripts/report_four_solver_benchmarks.py")
reporter = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(reporter)


def fixture(root, *, failed=False):
    old, new = root / "old", root / "new"
    old.mkdir()
    new.mkdir()
    X, y = np.ones((20, 4), dtype=np.float32), np.tile([0, 1], 10)
    folds = [{"training_indices": train.tolist(), "validation_indices": validation.tolist()}
             for train, validation in StratifiedKFold(5, shuffle=True, random_state=43).split(X, y)]
    job = {"id": "task-cd", "method": "screened_cd_absolute", "pair": {"id": "task", "negative": 0, "positive": 1},
           "seed": 43, "n_train": 20, "max_iter": 20000, "tol": .0001, "array_key": "train",
           "training_data_sha256": shared.data_fingerprint(X, y), "cache": str(old / "arrays")}
    estimator = shared._estimator(job)
    runtime = shared._calibration_runtime(estimator, 1)
    preflight = {"fold_indices_sha256": recovery._digest(folds), "folds": [
        {"fold": fold, "training_size": 16, "screens": {"0": {"features": 4}, "1": {"features": 4}}} for fold in range(5)]}
    grid = {"n_components": [2], "score_threshold": [0, 1], "n_neighbors": [1, 3]}
    plan = shared._plan(job, grid, preflight)
    bpp = shared._plan({**job, "id": "task-bpp", "method": recovery.BPP}, grid, preflight)
    config = {"sources": {"package_source_sha256": runtime["package_source"]}, "splits": {"task": {"train": list(range(20))}},
              "plans": [plan, bpp]}
    for chunk_index, chunk in enumerate(plan["chunks"]):
        directory = old / job["id"] / chunk["id"]
        directory.mkdir(parents=True)
        candidates = list(recovery.ParameterGrid(chunk["grid"]))
        calibration = {"estimator": "supervised_nmf.classifier.NMFClassifier", "parameters": shared._jsonable(estimator.get_params()),
            "candidates": candidates, "data_sha256": job["training_data_sha256"], "cv": 5, "seed": 43, "threads": 1,
            "refit": False, "python": runtime["python"], "platform": runtime["platform"], "numerical_runtime": runtime["numerical_libraries"],
            "software": runtime["versions"], "source_sha256": runtime["package_source"]}
        identity = {"schema_version": 1, "method": "explicit_grid_shared_knn_representation", "calibration": calibration,
                    "runtime": runtime, "folds": folds, "neighbor_parameter": "n_neighbors", "representation_backend": "nmf"}
        fingerprint = recovery._digest(identity)
        recovery._write_receipt(directory / "identity.json", identity)
        parameters = {key: value for key, value in candidates[0].items() if key != "n_neighbors"}
        representation = recovery._digest(parameters)
        receipts = []
        for fold in range(5 if chunk_index == 0 else 2):
            key = recovery._digest({"identity": fingerprint, "representation": representation, "fold": fold})
            record = {"fold": fold, "identity": fingerprint, "key": key, "parameters": parameters,
                      "representation_key": representation, "state": "completed", "status": "valid", "warnings": [],
                      "representation_fit_seconds": 1., "projection_seconds": .5, "representation_fit_performed": True,
                      "neighbors": {str(k): {"n_neighbors": k, "status": "valid", "warnings": [], "fit_seconds": .125,
                                             "prediction_seconds": .25, "balanced_accuracy": .75} for k in (1, 3)}}
            if failed and fold == 0:
                record.update(status="nonconverged", warnings=[{"category": "ConvergenceWarning", "message": "fixture"}], neighbors={})
            recovery._write_receipt(directory / "folds" / f"{key}.json", record)
            recovery._write_receipt(directory / "attempts" / f"completed-{fold}.json", record)
            receipts.append(record)
        if chunk_index == 1:
            key = recovery._digest({"identity": fingerprint, "representation": representation, "fold": 2})
            recovery._write_receipt(directory / "attempts/running.json", {"fold": 2, "identity": fingerprint,
                "key": key, "parameters": parameters, "representation_key": representation, "state": "running"})
            continue
        candidate_rows = []
        for index, params in enumerate(candidates):
            rows = []
            for source in receipts:
                if source["status"] == "valid":
                    row = deepcopy(source["neighbors"][str(params["n_neighbors"])])
                else:
                    row = {"status": source["status"], "fit_seconds": 0., "prediction_seconds": 0., "warnings": source["warnings"]}
                row.update(fold=source["fold"], shared_representation_key=source["key"], diagnostics={})
                rows.append(row)
            valid = all(row["status"] == "valid" for row in rows)
            candidate_rows.append({"index": index, "parameters": params, "folds": rows, "valid": valid,
                "mean_balanced_accuracy": .75 if valid else None, "fold_score_std": 0. if valid else None,
                "invalid_fold_count": 0 if valid else 1})
        costs = {"representation_fit_seconds": 5., "validation_projection_seconds": 2.5,
                 "knn_fit_seconds": sum(item["fit_seconds"] for row in receipts for item in row["neighbors"].values()),
                 "knn_prediction_seconds": sum(item["prediction_seconds"] for row in receipts for item in row["neighbors"].values())}
        report = {"identity": identity, "fingerprint": fingerprint, "state": "cv_completed", "refit_performed": False,
                  "is_fallback": False, "representation_folds": receipts, "candidates": candidate_rows, "compute_seconds": costs,
                  "tuning_seconds": sum(costs.values())}
        recovery._write_receipt(directory / "report.json", report)
    # Invalid historical BPP bytes deliberately must never be read or inherited.
    bpp_dir = old / "task-bpp/c0000"
    bpp_dir.mkdir(parents=True)
    (bpp_dir / "report.json").write_text("obsolete BPP implementation; not a receipt", encoding="utf-8")
    child = deepcopy(config)
    child["plans"][0]["job"]["cache"] = str(new / "arrays")
    return old, new, config, child


class RecoveryTests(unittest.TestCase):
    def test_copy_is_byte_identical_preserves_partial_folds_and_excludes_all_bpp(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            result = recovery.migrate_checkpoints(old, new, parent, child, {})
            self.assertEqual(result["counts"]["completed_reports"], 1)
            self.assertEqual(result["counts"]["committed_folds"], 7)
            self.assertEqual(result["counts"]["partial_chunk_committed_folds"], 2)
            self.assertEqual(result["counts"]["historical_running_attempts"], 1)
            self.assertFalse((new / "task-bpp").exists())
            self.assertEqual(result["excluded_bpp"][0]["old_completed_reports"], 1)
            self.assertEqual(result["inherited_cv_compute_seconds"], 7 * 2.25)
            for item in result["copied_files"]:
                self.assertEqual(Path(item["source"]).read_bytes(), Path(item["destination"]).read_bytes())
            again = recovery.migrate_checkpoints(old, new, parent, child, {})
            self.assertEqual(result, again)

    def test_numerically_invalid_receipts_are_preserved_as_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory), failed=True)
            recovery.migrate_checkpoints(old, new, parent, child, {})
            report = recovery._read_receipt(new / "task-cd/c0000/report.json")
            self.assertTrue(all(not row["valid"] for row in report["candidates"]))
            self.assertTrue(all(row["mean_balanced_accuracy"] is None for row in report["candidates"]))

    def test_corrupt_receipt_rejected_before_any_checkpoint_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            path = next((old / "task-cd/c0001/folds").glob("*.json"))
            path.write_text('{"payload": {}, "sha256": "bad"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                recovery.migrate_checkpoints(old, new, parent, child, {})
            self.assertFalse((new / "task-cd").exists())

    def test_rechecksummed_fold_tampering_cannot_match_original_report(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            path = next((old / "task-cd/c0000/folds").glob("*.json"))
            record = recovery._read_receipt(path)
            record["neighbors"]["1"]["balanced_accuracy"] = .5
            recovery._write_receipt(path, record)
            with self.assertRaisesRegex(ValueError, "fold evidence"):
                recovery.migrate_checkpoints(old, new, parent, child, {})

    def test_changed_estimator_runtime_is_not_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            path = old / "task-cd/c0000/identity.json"
            identity = recovery._read_receipt(path)
            identity["runtime"]["estimator_source"]["supervised_nmf.classifier.NMFClassifier"] = "tampered"
            recovery._write_receipt(path, identity)
            with self.assertRaisesRegex(ValueError, "estimator source"):
                recovery.migrate_checkpoints(old, new, parent, child, {})

    def test_changed_package_source_or_grid_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            changed = deepcopy(child)
            changed["sources"]["package_source_sha256"] = {"different": "implementation"}
            with self.assertRaisesRegex(ValueError, "Package implementation"):
                recovery.migrate_checkpoints(old, new, parent, changed, {})
            changed = deepcopy(child)
            changed["plans"][0]["chunks"][0]["grid"][0]["n_neighbors"] = [1]
            with self.assertRaisesRegex(ValueError, "grid"):
                recovery.migrate_checkpoints(old, new, parent, changed, {})

    def test_different_existing_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            path = new / "task-cd/c0000/identity.json"
            path.parent.mkdir(parents=True)
            path.write_text("keep this evidence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Destination receipt differs"):
                recovery.migrate_checkpoints(old, new, parent, child, {})
            self.assertEqual(path.read_text(), "keep this evidence")

    def test_destination_bpp_checkpoint_or_unknown_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, parent, child = fixture(Path(directory))
            (new / "task-bpp/c0000").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "BPP checkpoints"):
                recovery.migrate_checkpoints(old, new, parent, child, {})
            parent["plans"][0]["job"]["method"] = child["plans"][0]["job"]["method"] = "unreviewed_solver"
            with self.assertRaisesRegex(ValueError, "allowlist"):
                recovery.migrate_checkpoints(old, new, parent, child, {})

    def test_orchestrator_passes_selected_protocol_paths_to_every_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fashion, celeba = root / "fashion-v3.json", root / "celeba-v2.json"
            with patch.object(orchestration.sys, "prefix", str(root / "py12_df_aocl")), \
                 patch.object(orchestration, "freeze_run", return_value={"files": {}}) as freeze, \
                 patch.object(orchestration.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as launch:
                orchestration.run(root, fashion, celeba)
            freeze.assert_called_once_with(root.resolve(), fashion, celeba)
            commands = [call.args[0] for call in launch.call_args_list]
            self.assertEqual(len(commands), 5)
            self.assertEqual([commands[index][commands[index].index("--protocol") + 1] for index in range(4)],
                             [str(path.resolve()) for path in (fashion, celeba, fashion, celeba)])
            self.assertTrue(all(call.kwargs["env"]["PYTHONFAULTHANDLER"] == "1" for call in launch.call_args_list))
            state = recovery.read_record(root / "status.json", {}, envelope=False)
            self.assertEqual(state["error_dialog_policy"], orchestration.ERROR_DIALOG_POLICY)

    def test_process_error_policy_preserves_flags_and_enables_faulthandler(self):
        if orchestration.os.name != "nt":
            self.skipTest("Windows process policy")
        import ctypes
        before = ctypes.windll.kernel32.GetErrorMode()
        orchestration.suppress_crash_dialogs()
        after = ctypes.windll.kernel32.GetErrorMode()
        self.assertEqual(after & before, before)
        self.assertEqual(after & 3, 3)
        self.assertEqual(orchestration.os.environ["PYTHONFAULTHANDLER"], "1")

    def test_report_reads_versioned_frozen_protocol_and_rejects_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source/examples/fashion-v3.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"version": 3}', encoding="utf-8")
            expected = orchestration.sha256(path)
            configuration = {"protocol": {"version": 3}, "protocol_file_sha256": expected}
            (root / "source-freeze.json").write_text(json.dumps({"protocol_paths": {"fashion": "examples/fashion-v3.json"},
                "files": {"examples/fashion-v3.json": expected}}), encoding="utf-8")
            self.assertEqual(reporter.frozen_protocol(reporter.Inputs(), root, "fashion", configuration, "not-used.json"), {"version": 3})
            path.write_text('{"version": 4}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot checksum"):
                reporter.frozen_protocol(reporter.Inputs(), root, "fashion", configuration, "not-used.json")

    def test_report_binds_inherited_receipt_bytes_and_excludes_bpp(self):
        with tempfile.TemporaryDirectory() as directory:
            # write_report resolves the run before calling this helper.
            root = Path(directory).resolve()
            destination = root / "fashion/task-cd/c0000/identity.json"
            destination.parent.mkdir(parents=True)
            destination.write_text('{"saved": "original bytes"}', encoding="utf-8")
            freeze = root / "source-freeze.json"
            freeze.write_text('{}', encoding="utf-8")
            configuration = {"plans": [{"job": {"id": "task-cd", "method": "screened_cd_absolute"}}]}
            value = {"state": "prepared_no_models_fitted", "bpp_receipts_reused": False, "official_test_loaded": False,
                     "validation_outcomes_read": False, "child_configuration_sha256": recovery._digest(configuration),
                     "child_source_freeze_sha256": orchestration.sha256(freeze),
                     "copied_files": [{"destination": str(destination), "sha256": orchestration.sha256(destination)}]}
            recovery._write_receipt(root / "recovery-manifest.json", value)
            inputs = reporter.Inputs()
            inputs.read(freeze, receipt=False)
            self.assertEqual(reporter.recovery_provenance(inputs, root, configuration), value)
            destination.write_text('changed', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                reporter.recovery_provenance(inputs, root, configuration)


if __name__ == "__main__":
    unittest.main()
