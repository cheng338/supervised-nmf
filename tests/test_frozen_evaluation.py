"""No-fit tests for the default-freeze and official-test access gates."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_celeba_frozen_default as final
from supervised_nmf.evaluation import _digest
from supervised_nmf.study import _write_receipt


class FrozenEvaluationGates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nmf-freeze-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_incomplete_gate_cannot_call_test_loader(self):
        freeze = {"job_ids": [f"j{i}" for i in range(8)]}
        gate = {"freeze_sha256": _digest(freeze), "job_ids": freeze["job_ids"],
                "records": {f"j{i}": f"hash{i}" for i in range(7)}}
        _write_receipt(self.root / "validation-gate.json", gate)
        with patch.object(final, "_assert_sources"), patch.object(final, "_assert_annotation_files"), \
                patch.object(final, "load_celeba", side_effect=AssertionError("Test image access")) as loader:
            with self.assertRaisesRegex(ValueError, "eight-model validation gate"):
                final.load_frozen_test(self.root, freeze, gate, [])
        loader.assert_not_called()

    def test_validation_mismatch_aborts_before_test_cache(self):
        with patch.object(final, "_write_new"), patch.object(final, "provenance", return_value={}), \
                patch.object(final, "revalidate_models", side_effect=ValueError("Validation differs")), \
                patch.object(final, "load_frozen_test", side_effect=AssertionError("Test cache access")) as loader:
            with self.assertRaisesRegex(ValueError, "Validation differs"):
                final.evaluate_frozen(self.root, {}, [])
        loader.assert_not_called()

    def test_different_freeze_cannot_replace_committed_plan(self):
        path = self.root / "freeze.json"
        original = {"winner": "screened_spa_percentile", "parameters": {"n_components": 10}}
        final._write_new(path, original, receipt=False)
        before = path.read_bytes()
        final._write_new(path, original, receipt=False)
        self.assertEqual(before, path.read_bytes())
        with self.assertRaisesRegex(ValueError, "immutable artifact differs"):
            final._write_new(path, {**original, "winner": "screened_cd_absolute"}, receipt=False)
        self.assertEqual(before, path.read_bytes())

    def test_model_hash_checked_before_unpickling_and_paths_confined(self):
        path = self.root / "models" / "model.pkl"
        path.parent.mkdir()
        path.write_bytes(b"not a pickle; must never reach the unpickler")
        record = {"freeze_sha256": _digest({}), "status": "valid", "model_file": "models/model.pkl",
                  "model_sha256": "wrong"}
        with patch.object(final, "_assert_sources"), patch.object(final.pickle, "load") as loader:
            with self.assertRaisesRegex(ValueError, "hash check"):
                final._load_model(self.root, record, {})
        loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "outside this run"):
            final._model_path(self.root, "../foreign.pkl")

    def test_only_recorded_default_source_file_may_change(self):
        old = {"classifier.py": "old", "evaluation.py": "same"}
        self.assertEqual(final._default_only_source_change(old, {**old, "classifier.py": "new"}), ["classifier.py"])
        self.assertEqual(final._default_only_source_change(old, {"classifier.py": "new", "evaluation.py": "new"}), ["classifier.py", "evaluation.py"])
        for new in (old, {**old, "evaluation.py": "new"}, {**old, "classifier.py": "new", "new.py": "new"}):
            with self.assertRaisesRegex(ValueError, "Only classifier.py"):
                final._default_only_source_change(old, new)

    def test_stale_decision_evidence_or_conflicting_winner_rejected(self):
        evidence = self.root / "evidence.json"
        evidence.write_text("original")
        decision = {"winner_method": "screened_spa_percentile",
                    "primary": {"winner": "screened_spa_percentile", "status": "selected"},
                    "input_sha256": {str(evidence): final._sha(evidence)}}
        path = self.root / "decision.json"
        final.write_json(path, decision)
        self.assertEqual(final._decision(path)[1], "screened_spa_percentile")
        evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "input evidence changed"):
            final._decision(path)
        decision["winner_method"] = "screened_cd_absolute"
        final.write_json(path, decision)
        with self.assertRaisesRegex(ValueError, "one operationally qualified"):
            final._decision(path)

    def test_final_annotations_match_stage2_even_when_paths_relocated(self):
        attributes, partitions = self.root / "attrs.csv", self.root / "partitions.txt"
        attributes.write_text("sample labels")
        partitions.write_text("sample official partitions")
        manifest = {"preparation_identity": {"image_directory": str(self.root),
                    "source_files": {"attributes": {"path": "old location", "sha256": final._sha(attributes)},
                                     "partitions": {"path": "old location", "sha256": final._sha(partitions)}}}}
        checked = final._data_sources(manifest, self.root, attributes, partitions)
        self.assertEqual(checked["attributes"]["path"], str(attributes.resolve()))
        partitions.write_text("changed split")
        with self.assertRaisesRegex(ValueError, "data definition exactly"):
            final._data_sources(manifest, self.root, attributes, partitions)

    def test_freeze_refuses_before_recorded_default_is_installed(self):
        config = {}
        decision = {"configuration_sha256": _digest(config)}
        unset = Mock()
        unset.get_params.return_value = {"solver": None, "screening": None, "max_iter": 400, "tol": 1e-4}
        with patch.object(final, "_stage2", return_value=(config, [])), \
                patch.object(final, "_decision", return_value=(decision, "screened_spa_percentile")), \
                patch.object(final, "NMFClassifier", return_value=unset), patch.object(final, "load_celeba") as loader:
            with self.assertRaisesRegex(ValueError, "Install the recorded method default"):
                final.prepare_freeze("unused", "unused", self.root / "freeze.json", "unused", "unused", "unused")
        loader.assert_not_called()
        self.assertFalse((self.root / "freeze.json").exists())

    def test_stage2_must_contain_exactly_all_eight_models(self):
        config = {"plans": []}
        _write_receipt(self.root / "configuration.json", config)
        _write_receipt(self.root / "comparison.json", {"state": "development_evaluation_completed",
                      "configuration_sha256": _digest(config), "official_test_loaded": False})
        with self.assertRaisesRegex(ValueError, "exactly four methods"):
            final._stage2(self.root)

    def test_saved_predictions_are_reused_without_prediction_or_refit(self):
        freeze = {"winner_method": "screened_spa_percentile"}
        records, chosen = {}, []
        test = {"attributes": {attribute: {"data_sha256": attribute} for attribute in final.ATTRIBUTES}}
        for attribute in final.ATTRIBUTES:
            for method in final.METHODS:
                jobid = f"{attribute}-{method}"
                chosen.append({"job": {"id": jobid, "attribute": attribute, "method": method}})
                records[jobid] = {"model_sha256": jobid}
                identity = {"freeze_sha256": _digest(freeze), "job_id": jobid, "model_sha256": jobid,
                            "test_manifest_sha256": _digest(test), "test_data_sha256": attribute}
                _write_receipt(self.root / "evaluations" / f"{jobid}.json", {"identity": identity})
        before = {path: path.read_bytes() for path in (self.root / "evaluations").glob("*.json")}
        with patch.object(final, "provenance", return_value={}), patch.object(final, "_assert_sources"), \
                patch.object(final, "revalidate_models", return_value=({}, records)), \
                patch.object(final, "load_frozen_test", return_value=(test, None)), \
                patch.object(final, "evaluate_classifier", side_effect=AssertionError("Repeated prediction")) as predictor, \
                patch.object(final, "_load_model", side_effect=AssertionError("Unnecessary model load")) as loader:
            summary = final.evaluate_frozen(self.root, freeze, chosen)
        self.assertEqual(len(summary["models"]), 8)
        predictor.assert_not_called()
        loader.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_interrupted_exact_plan_prediction_retries_without_new_selection(self):
        freeze = {"winner_method": "screened_spa_percentile"}
        chosen, records = [], {}
        test = {"image_ids": ["test-a", "test-b"],
                "attributes": {attribute: {"data_sha256": attribute, "class_counts": [1, 1]}
                               for attribute in final.ATTRIBUTES}}
        for attribute in final.ATTRIBUTES:
            for method in final.METHODS:
                jobid = f"{attribute}-{method}"
                chosen.append({"job": {"id": jobid, "attribute": attribute, "method": method},
                               "manifest": {"splits": {"train": {"attributes": {attribute: {"class_counts": [1, 1]}}}}}})
                records[jobid] = {"model_sha256": jobid, "model_parameters": {}, "fit_seconds": .1}
        result = {"status": "valid", "metrics": {}}
        with patch.object(final, "provenance", return_value={}), patch.object(final, "_assert_sources"), \
                patch.object(final, "revalidate_models", return_value=({}, records)), \
                patch.object(final, "load_frozen_test", return_value=(test, None)), \
                patch.object(final, "_load_model", return_value=object()), \
                patch.object(final.np, "load", return_value=np.array([0, 1])), patch("builtins.print"), \
                patch.object(final, "evaluate_classifier", side_effect=[KeyboardInterrupt(), *([result] * 8)]) as predictor:
            with self.assertRaises(KeyboardInterrupt):
                final.evaluate_frozen(self.root, freeze, chosen)
            self.assertFalse((self.root / "comparison.json").exists())
            completed = final.evaluate_frozen(self.root, freeze, chosen)
            final.evaluate_frozen(self.root, freeze, chosen)
        self.assertEqual(predictor.call_count, 9)
        self.assertEqual(len(completed["models"]), 8)
        attempts = [final._read_receipt(path) for path in (self.root / "attempts").glob("*.json")]
        self.assertEqual(sum(row["state"] == "interrupted" for row in attempts), 1)
        self.assertEqual(sum(row["state"] == "completed" for row in attempts), 8)
        self.assertEqual({row["freeze_sha256"] for row in attempts}, {_digest(freeze)})

    def test_one_of_eight_probability_mismatches_leaves_gate_closed(self):
        prediction = {"sample_ids": ["v-a", "v-b"], "true_labels": [0, 1],
                      "predicted_labels": [0, 1], "positive_class_probability": [.2, .8]}
        manifest = {"splits": {"validation": {"image_ids": prediction["sample_ids"]}}}
        chosen = []
        for attribute in final.ATTRIBUTES:
            for method in final.METHODS:
                chosen.append({"job": {"id": f"{attribute}-{method}", "method": method, "attribute": attribute},
                               "manifest": manifest, "evaluation_file_sha256": "stage2-reference",
                               "evaluation": {"selected_parameters": {}, "evaluation": {"predictions": prediction}}})
        freeze = {"training_cache": str(self.root / "unused"), "job_ids": [row["job"]["id"] for row in chosen]}
        model = Mock()
        model.get_params.return_value = {}
        evaluations = [{"status": "valid", "predictions": prediction} for _ in range(7)]
        evaluations.append({"status": "valid", "predictions": {**prediction, "positive_class_probability": [.2, .81]}})
        with patch.object(final, "_assert_sources"), patch.object(final, "_cache_identity", return_value={"manifest": manifest}), \
                patch.object(final.np, "load", return_value=np.array([0, 1])), \
                patch.object(final, "comparison_models", return_value={method: (model, {}) for method in final.METHODS}), \
                patch.object(final, "_nonconverged", return_value=False), patch.object(final, "_diagnostics", return_value={}), \
                patch.object(final, "evaluate_classifier", side_effect=evaluations), \
                patch.object(final.pickle, "dump", side_effect=lambda value, stream, protocol: stream.write(b"mock model")), \
                patch.object(final, "load_celeba") as loader, patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "validation predictions differ"):
                final.revalidate_models(self.root, freeze, chosen)
        self.assertEqual(model.fit.call_count, 8)  # All calls are mocks: no numerical fit occurs.
        self.assertEqual(len(list((self.root / "revalidation").glob("*.json"))), 7)
        self.assertFalse((self.root / "validation-gate.json").exists())
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
