"""Historical recovery provenance, orchestration and exact current resume.

The retired migration writer is not imported or required by these checks.
"""

from contextlib import contextmanager
import importlib.util
import io
import json
import os
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


@contextmanager
def freeze_sources(root, fashion_name="fashion.json"):
    files = ["src/supervised_nmf/__init__.py", "scripts/run_four_solver_benchmarks.py",
             "scripts/fashion_mnist_study.py", "scripts/celeba_four_solver_study.py",
             "scripts/experimental_nmf.py", "scripts/report_four_solver_benchmarks.py",
             "scripts/_audit_identity.py",
             "scripts/profile_celeba_stages.py", "scripts/prepare_fashion_mnist.py",
             "scripts/check_four_solver_training.py", "research/fashion-mnist-protocol.md",
             "research/nmf-solver-comparison.md", "research/numerical-repair-20261004.md"]
    for relative in files:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    fashion, celeba = root / "examples" / fashion_name, root / "examples/celeba.json"
    fashion.parent.mkdir()
    fashion.write_text('{"fixture": "fashion"}', encoding="utf-8")
    celeba.write_text('{"fixture": "celeba"}', encoding="utf-8")
    with patch.object(orchestration, "ROOT", root), patch.object(
            orchestration, "__file__", str(root / "scripts/../scripts/run_four_solver_benchmarks.py")):
        yield fashion, celeba


def file_digests(directory):
    return {path.relative_to(directory).as_posix(): orchestration.sha256(path)
            for path in directory.rglob("*") if path.is_file()}


@contextmanager
def forbid_file_writes():
    original_open = io.open

    def read_only(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"Archive preflight opened {path} in mode {mode!r}")
        return original_open(path, mode, *args, **kwargs)

    with patch("builtins.open", read_only), patch("io.open", read_only):
        yield


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

    def test_report_reads_both_protocols_after_whole_run_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            output = root.parent / "run"
            relocated = root.parent / "copied-run"
            with freeze_sources(root) as (fashion, celeba):
                freeze = orchestration.freeze_run(output, fashion, celeba)
                shutil.copytree(output, relocated)
                for role, source in (("fashion", fashion), ("celeba", celeba)):
                    with self.subTest(role=role):
                        protocol = json.loads(source.read_bytes())
                        configuration = {"protocol": protocol, "protocol_file_sha256": orchestration.sha256(source)}
                        self.assertEqual(
                            reporter.frozen_protocol(reporter.Inputs(), relocated, role, configuration, "unused.json"),
                            protocol,
                        )
                        archived = relocated / "source" / freeze["protocol_paths"][role]
                        self.assertEqual(archived.read_bytes(), source.read_bytes())

    def test_archive_root_uses_the_literal_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            output = root.parent / "run"
            other = output / "Source"
            other.mkdir(parents=True)
            marker = other / "existing.txt"
            marker.write_bytes(b"previous partial data")
            manifest = output / "source-freeze.json"
            with freeze_sources(root) as (fashion, celeba):
                if (output / "source").exists():
                    with self.assertRaises(FileExistsError):
                        orchestration.freeze_run(output, fashion, celeba)
                    self.assertFalse(manifest.exists())
                    self.assertEqual(os.listdir(other), ["existing.txt"])
                else:
                    freeze = orchestration.freeze_run(output, fashion, celeba)
                    self.assertIn("source", os.listdir(output))
                    self.assertFalse(other.samefile(output / "source"))
                    self.assertEqual(orchestration.freeze_run(output, fashion, celeba), freeze)
                self.assertEqual(marker.read_bytes(), b"previous partial data")

    def test_protocols_belong_to_the_physical_checkout_on_fresh_and_resumed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "Checkout"
            fresh = root.parent / "new-run"
            complete = root.parent / "frozen-run"
            with freeze_sources(root) as (fashion, celeba):
                other = root.with_name("checkout")
                if other.exists():
                    other = root.with_name("OtherCheckout")
                other_protocol = other / fashion.relative_to(root)
                other_protocol.parent.mkdir(parents=True)
                # Equal data distinguishes selection of the wrong file from ordinary content drift.
                other_protocol.write_bytes(fashion.read_bytes())
                self.assertFalse(other_protocol.samefile(fashion))
                original = orchestration.freeze_run(complete, fashion, fashion)
                record = complete / "source-freeze.json"
                before = record.read_bytes()
                for output in (fresh, complete):
                    with self.subTest(output=output.name), self.assertRaises(ValueError):
                        orchestration.freeze_run(output, fashion, other_protocol)
                self.assertFalse((fresh / "source-freeze.json").exists())
                self.assertEqual(record.read_bytes(), before)
                self.assertEqual(original["protocol_paths"]["fashion"], original["protocol_paths"]["celeba"])
                self.assertEqual(orchestration.freeze_run(complete, fashion, fashion), original)

    def test_archive_destination_cannot_be_a_different_selected_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            output = root / "run"
            with freeze_sources(root) as (fashion, celeba):
                # Select the first protocol's archive destination as the second protocol.
                second = output / "source" / fashion.relative_to(root)
                second.parent.mkdir(parents=True)
                second.write_bytes(celeba.read_bytes())
                original = file_digests(root)
                with forbid_file_writes(), self.assertRaises(shutil.SameFileError):
                    orchestration.freeze_run(output, fashion, second)
                self.assertFalse((output / "source-freeze.json").exists())
                self.assertEqual(file_digests(root), original)

    def test_missing_source_is_reported_before_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            with freeze_sources(root) as (fashion, celeba):
                missing_protocol = root / "examples/missing.json"
                missing_doc = root / "research/numerical-repair-20261004.md"
                for missing, selected in ((missing_protocol, missing_protocol), (missing_doc, celeba)):
                    with self.subTest(missing=missing.name):
                        if missing == missing_doc:
                            missing_doc.unlink()
                        output = root.parent / missing.stem
                        original = file_digests(root)
                        with forbid_file_writes(), self.assertRaises(FileNotFoundError) as failure:
                            orchestration.freeze_run(output, fashion, selected)
                        self.assertEqual(Path(failure.exception.filename), missing)
                        self.assertFalse((output / "source-freeze.json").exists())
                        self.assertEqual(file_digests(output), {})
                        self.assertEqual(file_digests(root), original)

    def test_archive_preserves_literal_nested_directory_and_filename_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            with freeze_sources(root) as (fashion, celeba):
                for component, spelling in (("directory", "Examples/fashion.json"), ("file", "examples/Fashion.json")):
                    with self.subTest(component=component):
                        output = root.parent / component
                        archive = output / "source"
                        marker = archive / spelling
                        marker.parent.mkdir(parents=True)
                        marker.write_bytes(b"unpublished partial copy")
                        canonical = archive / fashion.relative_to(root)
                        if canonical.exists():
                            with self.assertRaises(FileExistsError):
                                orchestration.freeze_run(output, fashion, celeba)
                            self.assertFalse((output / "source-freeze.json").exists())
                        else:
                            orchestration.freeze_run(output, fashion, celeba)
                            self.assertIn("examples", os.listdir(archive))
                            self.assertIn(fashion.name, os.listdir(canonical.parent))
                            self.assertEqual(canonical.read_bytes(), fashion.read_bytes())
                        self.assertEqual(marker.read_bytes(), b"unpublished partial copy")

    def test_redirected_archive_directory_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            output = root.parent / "run"
            target = root.parent / "other"
            target.mkdir()
            marker = target / "existing.txt"
            marker.write_bytes(b"keep")
            link = output / "source/examples"
            link.parent.mkdir(parents=True)
            try:
                if os.name == "nt":
                    import _winapi
                    _winapi.CreateJunction(str(target), str(link))
                else:
                    link.symlink_to(target, target_is_directory=True)
            except (AttributeError, ImportError, NotImplementedError, OSError) as error:
                self.skipTest(f"Directory redirection unavailable in temporary storage: {error}")
            with freeze_sources(root) as (fashion, celeba):
                with forbid_file_writes(), self.assertRaises(FileExistsError):
                    orchestration.freeze_run(output, fashion, celeba)
                self.assertFalse((output / "source-freeze.json").exists())
                self.assertEqual(list(target.iterdir()), [marker])
                self.assertEqual(marker.read_bytes(), b"keep")

    def test_retry_completes_the_same_partial_archive_and_verifies_every_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkout"
            output = root.parent / "run"
            archive = output / "source"
            manifest = output / "source-freeze.json"
            with freeze_sources(root) as (fashion, celeba):
                expected = file_digests(root)
                stale = archive / fashion.relative_to(root)
                stale.parent.mkdir(parents=True)
                stale.write_bytes(b"unfinished protocol copy")
                real_copystat = shutil.copystat

                def interrupt_copy(source, destination, **kwargs):
                    if source == celeba:
                        raise OSError("simulated copy interruption")
                    return real_copystat(source, destination, **kwargs)

                with patch.object(shutil, "copystat", side_effect=interrupt_copy):
                    with self.assertRaisesRegex(OSError, "simulated copy interruption"):
                        orchestration.freeze_run(output, fashion, celeba)
                self.assertFalse(manifest.exists())
                self.assertEqual((archive / celeba.relative_to(root)).read_bytes(), celeba.read_bytes())
                self.assertEqual(stale.read_bytes(), b"unfinished protocol copy")

                freeze = orchestration.freeze_run(output, fashion, celeba)
                self.assertEqual(json.loads(manifest.read_bytes()), freeze)
                self.assertEqual(freeze["files"], expected)
                self.assertEqual(file_digests(archive), expected)
                self.assertEqual(file_digests(root), expected)

    def test_shared_hardlink_does_not_identify_its_sibling_checkout_as_this_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "Checkout"
            other = root.with_name("checkout")
            with freeze_sources(root) as (fashion, celeba):
                if other.exists():
                    self.skipTest("Case-distinct sibling checkouts require case-sensitive temporary storage")
                shared_protocol = other / fashion.relative_to(root)
                shared_protocol.parent.mkdir(parents=True)
                try:
                    shared_protocol.hardlink_to(fashion)
                except (NotImplementedError, OSError) as error:
                    self.skipTest(f"Hardlinks unavailable in temporary storage: {error}")
                self.assertFalse(other.samefile(root))
                self.assertTrue(shared_protocol.samefile(fashion))
                original = file_digests(root)
                fresh, complete = root.parent / "new-run", root.parent / "frozen-run"
                orchestration.freeze_run(complete, fashion, fashion)
                published = file_digests(complete)
                for output in (fresh, complete):
                    with self.subTest(output=output.name), forbid_file_writes(), self.assertRaises(ValueError):
                        orchestration.freeze_run(output, fashion, shared_protocol)
                self.assertFalse((fresh / "source-freeze.json").exists())
                self.assertEqual(file_digests(complete), published)
                self.assertEqual(file_digests(root), original)

    def test_source_freeze_does_not_require_retired_writer(self):
        with tempfile.TemporaryDirectory(prefix="freeze-") as directory:
            root = Path(directory).resolve()
            output = root / "run"
            with freeze_sources(root) as (fashion, celeba):
                freeze = orchestration.freeze_run(output, fashion, celeba)
                self.assertEqual(orchestration.freeze_run(output, fashion, celeba), freeze)
                helper = root / "scripts/_audit_identity.py"
                self.assertEqual(freeze["files"]["scripts/_audit_identity.py"], orchestration.sha256(helper))
                self.assertEqual((output / "source/scripts/_audit_identity.py").read_bytes(), helper.read_bytes())
                helper.write_bytes(helper.read_bytes() + b"\n# changed for restart check\n")
                with self.assertRaisesRegex(RuntimeError, "Frozen source/environment changed"):
                    orchestration.freeze_run(output, fashion, celeba)

                # Change the helper while the new freeze is being created.
                def record_edit():
                    helper.write_bytes(helper.read_bytes() + b"\n# edit during freeze\n")
                    return "2026-01-01T00:00:00+00:00"

                second = root / "second-run"
                with patch.object(orchestration, "now", side_effect=record_edit):
                    second_freeze = orchestration.freeze_run(second, fashion, celeba)
                self.assertEqual(second_freeze["files"]["scripts/_audit_identity.py"],
                                 orchestration.sha256(second / "source/scripts/_audit_identity.py"))
            self.assertIn("scripts/run_four_solver_benchmarks.py", freeze["files"])
            self.assertNotIn("scripts/prepare_four_solver_recovery.py", freeze["files"])
            self.assertFalse((root / "scripts/prepare_four_solver_recovery.py").exists())
            for relative, expected in freeze["files"].items():
                self.assertEqual(orchestration.sha256(output / "source" / relative), expected)

    def test_source_freeze_refuses_to_overwrite_its_own_source(self):
        with tempfile.TemporaryDirectory(prefix="freeze-") as directory:
            output = Path(directory).resolve()
            root = output / "source"
            manifest = output / "source-freeze.json"
            with freeze_sources(root) as (fashion, celeba):
                original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                real_open = io.open

                def guard_open(path, mode="r", *args, **kwargs):
                    if any(flag in mode for flag in "wax+") and Path(path).resolve() in original:
                        self.fail(f"Freeze opened a source file for writing: {path}")
                    return real_open(path, mode, *args, **kwargs)

                self.assertFalse(manifest.exists())
                with patch("builtins.open", guard_open), patch("io.open", guard_open):
                    with self.assertRaises(shutil.SameFileError):
                        orchestration.freeze_run(output, fashion, celeba)
                self.assertFalse(manifest.exists())
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content, str(path))

    @unittest.skipUnless(os.name == "posix", "A backslash is a literal filename character on POSIX")
    def test_source_freeze_preserves_literal_backslash_on_restart(self):
        with tempfile.TemporaryDirectory(prefix="freeze-") as directory:
            root = Path(directory).resolve()
            output = root / "run"
            key = r"examples/fashion\custom.json"
            with freeze_sources(root, r"fashion\custom.json") as (fashion, celeba):
                orchestration.freeze_run(output, fashion, celeba)
                manifest = output / "source-freeze.json"
                before = manifest.read_bytes()
                freeze = json.loads(before)
                self.assertEqual(freeze["protocol_paths"]["fashion"], key)
                self.assertEqual(freeze["files"][key], orchestration.sha256(fashion))
                self.assertNotIn("examples/fashion/custom.json", freeze["files"])
                self.assertEqual((output / "source" / key).read_bytes(), fashion.read_bytes())
                self.assertEqual(orchestration.freeze_run(output, fashion, celeba), freeze)
                self.assertEqual(manifest.read_bytes(), before)


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
