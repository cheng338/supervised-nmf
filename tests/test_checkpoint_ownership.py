"""Ownership contracts, without benchmark/timing assertions.

Cross-process schedules and abrupt-exit evidence live in the separate batch probe.
These tests use source-backed sklearn estimators under the normal identity guard.
"""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from supervised_nmf import evaluation, study


class CheckpointOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / "nested" / "checkpoint.json"
        self.X = np.random.default_rng(713).uniform(.2, 2., size=(24, 4))
        self.y = np.tile([0, 1], 12)
        self.X[:, 0] += self.y * 3

    def calibrate(self, path=None, *, refit=False, neighbors=1):
        return evaluation.calibration_search(
            KNeighborsClassifier(n_neighbors=neighbors), {"n_neighbors": [neighbors]},
            self.X, self.y, cv=3, seed=9, threads=1, refit=refit,
            checkpoint_path=path, scoring="accuracy")

    def assert_owned(self, path):
        with self.assertRaisesRegex(RuntimeError, "writing"):
            with evaluation._checkpoint_file(path):
                self.fail("A second owner entered the checkpoint transaction")

    def marker(self, path):
        return path.with_name("." + path.name + ".lock")

    def test_none_context_creates_no_files(self):
        before = list(self.root.rglob("*"))
        with evaluation._checkpoint_file(None) as path:
            self.assertIsNone(path)
        self.assertEqual(before, list(self.root.rglob("*")))

    def test_same_file_excluded_different_files_independent(self):
        other = self.path.with_name("other.json")
        with evaluation._checkpoint_file(self.path):
            self.assert_owned(self.path)
            with evaluation._checkpoint_file(other):
                self.assert_owned(other)
        self.assertTrue(self.marker(self.path).is_file())
        self.assertTrue(self.marker(other).is_file())
        with evaluation._checkpoint_file(self.path):
            pass

    def test_equivalent_parent_paths_share_ownership(self):
        alias = self.root / "nested" / ".." / "nested" / "checkpoint.json"
        with evaluation._checkpoint_file(self.path):
            self.assert_owned(alias)

    def test_threads_cannot_share_same_checkpoint(self):
        outcomes = []

        def contender():
            try:
                with evaluation._checkpoint_file(self.path):
                    outcomes.append("entered")
            except BaseException as error:
                outcomes.append(error)

        with evaluation._checkpoint_file(self.path):
            thread = threading.Thread(target=contender, daemon=True)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "Nonblocking ownership must return promptly")
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], RuntimeError)

    def test_keyboard_interrupt_releases_without_deleting_marker(self):
        with self.assertRaises(KeyboardInterrupt):
            with evaluation._checkpoint_file(self.path):
                raise KeyboardInterrupt("controlled")
        marker = self.marker(self.path)
        retained = marker.read_bytes()
        self.assertTrue(retained)
        with evaluation._checkpoint_file(self.path):
            pass
        self.assertEqual(marker.read_bytes(), retained)

    def test_calibration_contender_rejected_before_checkpoint_read_fit_or_write(self):
        self.path.parent.mkdir()
        self.path.write_text("not valid JSON", encoding="utf-8")
        real_read = Path.read_text
        reads = []

        def observed_read(path, *args, **kwargs):
            if path == self.path:
                reads.append(path)
            return real_read(path, *args, **kwargs)

        with evaluation._checkpoint_file(self.path):
            with patch.object(Path, "read_text", observed_read), \
                    patch.object(KNeighborsClassifier, "fit") as fit, \
                    patch.object(evaluation, "write_json") as write:
                with self.assertRaisesRegex(RuntimeError, "writing"):
                    self.calibrate(self.path)
                fit.assert_not_called()
                write.assert_not_called()
        self.assertEqual(reads, [])
        self.assertEqual(self.path.read_text(), "not valid JSON")

    def test_calibration_owns_every_fit_and_write_through_final_return(self):
        real_fit, real_write = KNeighborsClassifier.fit, evaluation.write_json
        fits, states = [], []

        def fit(model, X, y):
            self.assert_owned(self.path)
            fits.append(len(X))
            return real_fit(model, X, y)

        def write(path, payload):
            self.assert_owned(self.path)
            states.append(payload["state"])
            return real_write(path, payload)

        with patch.object(KNeighborsClassifier, "fit", fit), patch.object(evaluation, "write_json", write):
            result = self.calibrate(self.path, refit=True)
        self.assertFalse(result.report["is_fallback"])
        self.assertEqual(fits, [16, 16, 16, 24])
        self.assertEqual(states[-1], "completed")
        self.assertIn("refitting", states)
        with evaluation._checkpoint_file(self.path):
            pass

    def test_fallback_fit_remains_owned(self):
        real_fit = DummyClassifier.fit
        observed = []

        def fit(model, X, y, *args, **kwargs):
            self.assert_owned(self.path)
            observed.append(len(X))
            return real_fit(model, X, y, *args, **kwargs)

        with patch.object(DummyClassifier, "fit", fit):
            result = self.calibrate(self.path, refit=True, neighbors=99)
        self.assertTrue(result.report["is_fallback"])
        self.assertEqual(observed, [24])
        self.assertEqual(result.report["state"], "completed")
        with evaluation._checkpoint_file(self.path):
            pass

    def test_completed_resume_and_no_checkpoint_keep_fit_contract(self):
        real_fit = KNeighborsClassifier.fit
        calls = []

        def fit(model, X, y):
            calls.append(len(X))
            return real_fit(model, X, y)

        with patch.object(KNeighborsClassifier, "fit", fit):
            first = self.calibrate(self.path)
            self.assertEqual(len(calls), 3)
            resumed = self.calibrate(self.path)
            self.assertEqual(len(calls), 3)
            files = sorted(str(p) for p in self.root.rglob("*"))
            no_checkpoint = self.calibrate()
            self.assertEqual(len(calls), 6)
        self.assertEqual(files, sorted(str(p) for p in self.root.rglob("*")))
        self.assertEqual(first.report["selected_parameters"], resumed.report["selected_parameters"])
        self.assertEqual(first.report["selected_mean_score"], resumed.report["selected_mean_score"])
        self.assertEqual(first.report["selected_mean_score"], no_checkpoint.report["selected_mean_score"])

    def test_partial_resume_preserves_committed_fold_and_releases_owner(self):
        real_write, real_fit = evaluation.write_json, KNeighborsClassifier.fit

        def write(path, payload):
            real_write(path, payload)
            if sum(len(candidate["folds"]) for candidate in payload["candidates"]) == 1:
                raise KeyboardInterrupt("controlled one-fold interruption")

        with patch.object(evaluation, "write_json", write):
            with self.assertRaises(KeyboardInterrupt):
                self.calibrate(self.path)
        first = json.loads(self.path.read_text())["candidates"][0]["folds"][0]
        calls = []

        def fit(model, X, y):
            calls.append(len(X))
            return real_fit(model, X, y)

        with patch.object(KNeighborsClassifier, "fit", fit):
            result = self.calibrate(self.path)
        self.assertEqual(calls, [16, 16])
        self.assertEqual(first, result.report["candidates"][0]["folds"][0])
        self.assertEqual(result.report["state"], "cv_completed")

    def test_rejected_resume_does_not_keep_ownership(self):
        self.calibrate(self.path)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.calibrate(self.path, neighbors=3)
        self.assertEqual(before, self.path.read_bytes())
        with evaluation._checkpoint_file(self.path):
            pass
        saved = json.loads(before)
        saved["tuning_seconds"] += 1
        self.path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.calibrate(self.path)
        with evaluation._checkpoint_file(self.path):
            pass

    def test_manifest_create_verify_and_mismatch_are_owned(self):
        real_write, real_read = evaluation.write_json, Path.read_text
        observed = []

        def write(path, value):
            self.assert_owned(self.path)
            observed.append("write")
            real_write(path, value)

        def read(path, *args, **kwargs):
            if path == self.path:
                self.assert_owned(self.path)
                observed.append("read")
            return real_read(path, *args, **kwargs)

        with patch.object(evaluation, "write_json", write), patch.object(Path, "read_text", read):
            evaluation.save_data_manifest(self.path, {"split": "A"})
            evaluation.save_data_manifest(self.path, {"split": "A"})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                evaluation.save_data_manifest(self.path, {"split": "B"})
        self.assertEqual(observed, ["write", "read", "read"])
        self.assertEqual(json.loads(self.path.read_text()), {"split": "A"})
        with evaluation._checkpoint_file(self.path):
            pass

    def test_manifest_contender_rejected_before_read_or_write(self):
        self.path.parent.mkdir()
        self.path.write_text("not valid JSON", encoding="utf-8")
        with evaluation._checkpoint_file(self.path):
            with patch.object(evaluation, "write_json") as write, patch.object(Path, "read_text") as read:
                with self.assertRaisesRegex(RuntimeError, "writing"):
                    evaluation.save_data_manifest(self.path, {"split": "A"})
                read.assert_not_called()
                write.assert_not_called()

    def test_study_lock_retains_marker_and_context_contract(self):
        directory = self.root / "study"
        directory.mkdir()
        marker = directory / ".study.lock"
        with study._checkpoint_lock(directory):
            with self.assertRaisesRegex(RuntimeError, "writing"):
                with study._checkpoint_lock(directory):
                    self.fail("Study owner must be exclusive")
        content = marker.read_bytes()
        with self.assertRaises(KeyboardInterrupt):
            with study._checkpoint_lock(directory):
                raise KeyboardInterrupt("controlled")
        with study._checkpoint_lock(directory):
            pass
        self.assertEqual(marker.read_bytes(), content)
        self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
