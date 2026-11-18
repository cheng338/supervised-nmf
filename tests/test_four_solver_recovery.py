"""Historical recovery provenance, orchestration and exact current resume.

The retired migration writer is not imported or required by these checks.
"""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

import supervised_nmf.study as study
from supervised_nmf.evaluation import _digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fashion_mnist_study as shared
import run_four_solver_benchmarks as orchestration

orchestration.suppress_crash_dialogs()
report_spec = importlib.util.spec_from_file_location("_snmf_recovery_report", ROOT / "scripts/report_four_solver_benchmarks.py")
reporter = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(reporter)


class RecoveryProvenanceTests(unittest.TestCase):
    def test_orchestrator_passes_selected_protocol_paths_to_every_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fashion, celeba = root / "fashion-v3.json", root / "celeba-v2.json"
            with patch.object(orchestration.sys, "prefix", str(root / "renamed_current_env")), \
                 patch.object(orchestration, "freeze_run", return_value={"files": {}}) as freeze, \
                 patch.object(orchestration.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as launch:
                orchestration.run(root, fashion, celeba)
            freeze.assert_called_once_with(root.resolve(), fashion, celeba)
            commands = [call.args[0] for call in launch.call_args_list]
            self.assertEqual(len(commands), 5)
            self.assertTrue(all(command[0] == sys.executable for command in commands))
            self.assertEqual([commands[index][commands[index].index("--protocol") + 1] for index in range(4)],
                             [str(path.resolve()) for path in (fashion, celeba, fashion, celeba)])
            self.assertTrue(all(call.kwargs["env"]["PYTHONFAULTHANDLER"] == "1" for call in launch.call_args_list))
            state = json.loads((root / "status.json").read_text(encoding="utf-8"))
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
                     "validation_outcomes_read": False, "child_configuration_sha256": _digest(configuration),
                     "child_source_freeze_sha256": orchestration.sha256(freeze),
                     "copied_files": [{"destination": str(destination), "sha256": orchestration.sha256(destination)}]}
            study._write_receipt(root / "recovery-manifest.json", value)
            inputs = reporter.Inputs()
            inputs.read(freeze, receipt=False)
            self.assertEqual(reporter.recovery_provenance(inputs, root, configuration), value)
            destination.write_text('changed', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                reporter.recovery_provenance(inputs, root, configuration)

    def test_source_freeze_does_not_require_retired_writer(self):
        with tempfile.TemporaryDirectory(prefix="freeze-") as directory:
            root = Path(directory)
            files = ["src/supervised_nmf/__init__.py", "scripts/run_four_solver_benchmarks.py",
                     "scripts/fashion_mnist_study.py", "scripts/celeba_four_solver_study.py",
                     "scripts/experimental_nmf.py", "scripts/report_four_solver_benchmarks.py",
                     "scripts/profile_celeba_stages.py", "scripts/prepare_fashion_mnist.py",
                     "scripts/check_four_solver_training.py", "docs/fashion-mnist-protocol.md",
                     "docs/nmf-solver-comparison.md", "docs/numerical-repair-20261004.md"]
            for relative in files:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            fashion, celeba = root / "examples/fashion.json", root / "examples/celeba.json"
            fashion.parent.mkdir()
            fashion.write_text('{"fixture": "fashion"}', encoding="utf-8")
            celeba.write_text('{"fixture": "celeba"}', encoding="utf-8")
            output = root / "run"
            with patch.object(orchestration, "ROOT", root), patch.object(
                    orchestration, "__file__", str(root / "scripts/run_four_solver_benchmarks.py")):
                freeze = orchestration.freeze_run(output, fashion, celeba)
                self.assertEqual(orchestration.freeze_run(output, fashion, celeba), freeze)
            self.assertNotIn("scripts/prepare_four_solver_recovery.py", freeze["files"])
            self.assertFalse((root / "scripts/prepare_four_solver_recovery.py").exists())
            for relative, expected in freeze["files"].items():
                self.assertEqual(orchestration.sha256(output / "source" / relative), expected)


class CurrentResumeTests(unittest.TestCase):
    @staticmethod
    def snapshot(root):
        return {path.relative_to(root): path.read_bytes() for path in root.rglob("*")
                if path.is_file() and path.name not in {".orchestrator.lock", ".study.lock"}}

    def test_current_worker_receipts_resume_without_fits(self):
        with tempfile.TemporaryDirectory(prefix="resume-") as directory:
            root = Path(directory)
            X = np.arange(80, dtype=np.float32).reshape(20, 4) / 80
            y = np.tile([0, 1], 10).astype(np.uint8)
            arrays = shared._save_arrays(root / "a", X, y)
            job = {"id": "raw", "method": "raw_knn", "seed": 43, "n_train": len(y),
                   "max_iter": 100, "tol": .001, "cache": arrays["path"],
                   "training_data_sha256": arrays["data_sha256"]}
            folds = [{"training_indices": train.tolist(), "validation_indices": validation.tolist()}
                     for train, validation in StratifiedKFold(5, shuffle=True, random_state=43).split(X, y)]
            preflight = {"fold_indices_sha256": _digest(folds), "folds": [
                {"fold": fold, "training_size": 16, "screens": {}} for fold in range(5)]}
            plan = shared._plan(job, {"n_neighbors": [1, 3]}, preflight)
            parent = root / "p"
            sources = shared._sources()
            first = shared._worker((parent, plan, plan["chunks"][0], sources))
            self.assertEqual(first["new_representation_fits"], 5)
            checkpoint = parent / "raw/c0000"
            original_report = study._read_receipt(checkpoint / "report.json")
            original_bytes = self.snapshot(checkpoint)
            relocated = root / "r"
            shutil.copytree(parent / "raw", relocated / "raw")
            with patch.object(study, "_run_representation_fold", side_effect=AssertionError(
                    "Completed folds must not be recomputed")) as fold_fit, patch.object(
                    KNeighborsClassifier, "fit", side_effect=AssertionError("No new KNN fit")) as knn_fit:
                resumed = shared._worker((relocated, plan, plan["chunks"][0], sources))
            fold_fit.assert_not_called()
            knn_fit.assert_not_called()
            self.assertEqual(resumed["new_representation_fits"], 0)
            self.assertEqual(resumed["reused_representation_folds"], 5)
            resumed_report = study._read_receipt(relocated / "raw/c0000/report.json")
            self.assertEqual(resumed_report["candidates"], original_report["candidates"])
            self.assertEqual(resumed_report["compute_seconds"], original_report["compute_seconds"])
            self.assertEqual(self.snapshot(relocated / "raw/c0000/folds"),
                             self.snapshot(checkpoint / "folds"))
            self.assertEqual(self.snapshot(checkpoint), original_bytes)


if __name__ == "__main__":
    unittest.main()
