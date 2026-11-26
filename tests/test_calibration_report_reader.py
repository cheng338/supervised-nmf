"""Read completed calibration reports without replaying the original search."""

from copy import deepcopy
import builtins
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from supervised_nmf import evaluation


def write_search(path, depths, *, scoring="accuracy", refit=False):
    # The callback advances its captured RNG while preserving the design matrix.
    rng = np.random.RandomState(17)

    def transform(X):
        rng.random_sample()
        return X

    estimator = make_pipeline(FunctionTransformer(transform), DecisionTreeClassifier(random_state=7))
    y = np.tile([0, 1], 8)
    return evaluation.calibration_search(
        estimator, {"decisiontreeclassifier__max_depth": depths}, y[:, None], y,
        cv=2, seed=3, refit=refit, scoring=scoring, checkpoint_path=path,
    )


def seal(report):
    report["checkpoint_sha256"] = evaluation._digest({
        key: value for key, value in report.items() if key != "checkpoint_sha256"
    })


class CalibrationReportReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.json"
            write_search(path, [0, 1, 2])
            cls.original = json.loads(path.read_bytes())

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "archived.json"

    def save(self, report):
        self.path.write_text(json.dumps(report), encoding="utf-8")

    def test_archive_returns_exact_report_without_reexecuting_or_writing(self):
        self.assertFalse(self.original["resume_eligible"])
        self.assertEqual([row["valid"] for row in self.original["candidates"]], [False, True, True])
        self.assertEqual(self.original["selected_candidate_index"], 1)
        self.save(self.original)
        original_bytes = self.path.read_bytes()
        with (
            patch.object(builtins, "open", wraps=builtins.open) as file_open,
            patch.object(io, "open", wraps=io.open) as io_open,
            patch.object(evaluation, "calibration_search", side_effect=AssertionError("search replayed")),
            patch.object(evaluation, "_search_identity", side_effect=AssertionError("identity rebuilt")),
            patch.object(evaluation, "build_cv_folds", side_effect=AssertionError("folds rebuilt")),
            patch.object(DecisionTreeClassifier, "fit", side_effect=AssertionError("estimator refit")),
            patch.object(evaluation, "write_json", side_effect=AssertionError("archive rewritten")),
        ):
            self.assertEqual(evaluation.load_calibration_report(self.path), self.original)
        opens = [call for call in file_open.call_args_list + io_open.call_args_list
                 if call.args and call.args[0] in (self.path, str(self.path))]
        self.assertEqual(len(opens), 1)
        mode = opens[0].args[1] if len(opens[0].args) > 1 else opens[0].kwargs.get("mode", "r")
        self.assertIn(mode, ("r", "rb", "rt"))
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_balanced_accuracy_and_no_eligible_candidate_are_readable(self):
        for depths, selected in (([1], 0), ([0, -1], None)):
            with self.subTest(depths=depths):
                path = self.path.with_name(f"search-{depths[0]}.json")
                write_search(path, depths, scoring="balanced_accuracy")
                report = json.loads(path.read_bytes())
                self.assertEqual(report["selected_candidate_index"], selected)
                self.assertFalse(report["is_fallback"])
                self.assertEqual(evaluation.load_calibration_report(str(path)), report)

    def test_nonconverged_folds_and_nonzero_sample_standard_deviation(self):
        y = np.array([0] * 6 + [1] * 2 + [0] * 5 + [1] * 3)
        first, second = np.arange(8), np.arange(8, 16)
        evaluation.calibration_search(
            LogisticRegression(random_state=11, solver="lbfgs", tol=1e-9),
            {"max_iter": [1, 100]}, np.ones((len(y), 1)), y,
            cv=2, folds=[(first, second), (second, first)], refit=False, checkpoint_path=self.path,
        )
        report = json.loads(self.path.read_bytes())
        unconverged, valid = report["candidates"]
        self.assertEqual([fold["status"] for fold in unconverged["folds"]], ["nonconverged"] * 2)
        self.assertIsNone(unconverged["mean_score"])
        self.assertEqual(report["selected_candidate_index"], 1)
        self.assertAlmostEqual(valid["fold_score_std"], np.std([0.625, 0.75], ddof=1))
        self.assertEqual(evaluation.load_calibration_report(self.path), report)

    def test_completed_final_fit_fallback_is_outside_cv_only_contract(self):
        write_search(self.path, [0], refit=True)
        report = json.loads(self.path.read_bytes())
        self.assertTrue(report["is_fallback"])
        self.assertEqual(report["final_fit_attempts"][-1]["status"], "fallback")
        with self.assertRaises(ValueError):
            evaluation.load_calibration_report(self.path)

    def test_rehashed_contradictions_and_invalid_shapes(self):
        changes = (
            (("schema_version",), 1),
            (("state",), "calibrating"),
            (("refit_performed",), True),
            (("final_fit_attempts",), [{"status": "fallback"}]),
            (("final_fit_seconds",), 0.1),
            (("final_fit_seconds",), False),
            (("identity",), []),
            (("candidates",), {}),
            (("candidates", 0), None),
            (("candidates", 0, "valid"), True),
            (("candidates", 0, "invalid_fold_count"), 1),
            (("candidates", 1, "folds"), []),
            (("candidates", 1, "folds", 0), None),
            (("candidates", 1, "folds", 1, "fold"), 0),
            (("candidates", 1, "folds", 0, "accuracy"), 0.25),
            (("candidates", 1, "folds", 0, "score"), True),
            (("candidates", 2, "fold_score_std"), 0.25),
            (("candidates", 2, "index"), 1),
            (("selected_mean_score",), 0.5),
        )
        for keys, value in changes:
            with self.subTest(keys=keys):
                report = deepcopy(self.original)
                node = report
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                seal(report)
                self.save(report)
                with self.assertRaises(ValueError):
                    evaluation.load_calibration_report(self.path)

    def test_rehashed_aggregate_and_tie_winner_must_follow_saved_folds(self):
        for corruption in ("mean", "winner", "parameters"):
            with self.subTest(corruption=corruption):
                report = deepcopy(self.original)
                row = report["candidates"][2]
                if corruption == "mean":
                    row.update(mean_score=0.75, mean_accuracy=0.75)
                elif corruption == "winner":
                    report.update(selected_candidate_index=2, selected_parameters=row["parameters"])
                else:
                    row["parameters"]["decisiontreeclassifier__max_depth"] = 3
                seal(report)
                self.save(report)
                with self.assertRaises(ValueError):
                    evaluation.load_calibration_report(self.path)

    def test_stale_checksums_and_strict_json(self):
        report = deepcopy(self.original)
        report["checkpoint_sha256"] = "0" * 64
        original_json = json.dumps(self.original)
        invalid = [
            json.dumps(report), original_json[:-1], "[]", "null",
            original_json.replace('"schema_version":', '"schema_version": 2, "schema_version":', 1),
            original_json.replace('"score":', '"score": null, "score":', 1),
        ]
        for token in ("NaN", "Infinity", "-Infinity", "1e9999"):
            # The writer's digest normalizes nonfinite floats to null.
            report = dict(self.original, extra_value=None)
            seal(report)
            invalid.append(json.dumps(report).replace('"extra_value": null', f'"extra_value": {token}'))
        for index, payload in enumerate(invalid):
            with self.subTest(index=index):
                self.path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluation.load_calibration_report(self.path)

    def test_partition_identity_and_checksum_are_independent_checks(self):
        for corruption in ("identity", "split", "bool_index", "overlap", "refit_plan"):
            with self.subTest(corruption=corruption):
                report = deepcopy(self.original)
                identity = report["identity"]
                if corruption == "identity":
                    identity["seed"] += 1
                elif corruption == "refit_plan":
                    identity["refit"] = True
                elif corruption == "split":
                    report["folds"][0]["training_indices"].reverse()
                else:
                    train = report["folds"][0]["training_indices"]
                    train[0] = True if corruption == "bool_index" else report["folds"][0]["validation_indices"][0]
                    identity["folds"][0][0] = train.copy()
                if corruption != "identity":
                    report["fingerprint"] = evaluation._digest(identity)
                seal(report)
                self.save(report)
                with self.assertRaises(ValueError):
                    evaluation.load_calibration_report(self.path)

    def test_historical_large_integer_encodings(self):
        value = -(1 << 2100)
        for encoding in (str(value), hex(value)):
            with self.subTest(encoding=encoding[:6]):
                report = deepcopy(self.original)
                identity = report["identity"]
                config = {k["value"]: v for k, v in identity["structural_configuration"]["items"]}
                config["candidates"]["items"][0]["items"][0][1]["value"] = encoding
                report["candidates"][0]["parameters"]["decisiontreeclassifier__max_depth"] = value
                report["refinement"]["parameters"]["decisiontreeclassifier__max_depth"]["values"][0] = value
                report["fingerprint"] = evaluation._digest(identity)
                # A version-6 report still contains the original numeric axis.
                seal(report)
                self.save(report)
                with self.assertRaisesRegex(ValueError, "Refinement"):
                    evaluation.load_calibration_report(self.path)
                report["refinement_axis_inputs"]["decisiontreeclassifier__max_depth"]["coordinates"][0] = value
                seal(report)
                self.save(report)
                self.assertEqual(evaluation.load_calibration_report(self.path), report)
                identity["fold_evidence_version"] = 5
                del report["refinement_axis_inputs"]
                report["fingerprint"] = evaluation._digest(identity)
                seal(report)
                self.save(report)
                self.assertEqual(evaluation.load_calibration_report(self.path), report)

    def test_numpy_integer_grid_is_bound_to_declaration(self):
        write_search(self.path, [np.int64(1), np.int64(2)])
        report = evaluation.load_calibration_report(self.path)
        report["candidates"][1]["parameters"]["decisiontreeclassifier__max_depth"] = 3
        seal(report)
        self.save(report)
        with self.assertRaises(ValueError):
            evaluation.load_calibration_report(self.path)

    def test_missing_file_remains_a_file_error(self):
        with self.assertRaises(FileNotFoundError):
            evaluation.load_calibration_report(self.path)

