"""Synthetic default-selection regressions. Author: Chang Cheng.

No estimator fits, private data, saved study outcomes or ignored protocol files
are required. The literal decision fixture preserves the version-1 rule.
"""

from copy import deepcopy
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tempfile
import unittest
from unittest.mock import patch

from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_supervised_nmf_test_default_selector", ROOT / "scripts/select_celeba_default.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
METHODS = list(selector.LABELS)
PROTOCOL = {
    "schema_version": 1, "author": "Chang Cheng", "protocol_id": "synthetic-version-1-rule-fixture",
    "methods_in_final_tie_order": METHODS,
    "attributes": ["Smiling", "Wearing_Necktie"], "training_sizes": [2000, 4000, 8000], "training_seeds": [43, 44, 45],
    "sampling": {"validation_sample_count": 4000, "validation_seed": 42},
    "numerical_configuration": {"max_iter": 20000, "tol": .0001},
    "stage_2": {"method_attribute_configuration_jobs": 72, "n_neighbors": [1, 2, 3, 5, 9, 15, 25, 41, 65, 81]},
    "timing": {"additional_repeats_per_selected_model": 3, "primary_cost_closeness_ratio": 1.05, "order_seed": 42043},
    "decision_rule": {"macro_regret_tolerance": .01, "attribute_regret_tolerance": .02, "numerical_comparison_slack": 1e-12},
    "sensitivity": {"macro_margins": [0, .005, .01, .02], "attribute_margin_multiplier": 2,
                    "prediction_multipliers": [.1, 1, 10, 100]},
}


def rows_for(scores=None, costs=None, tuning=None):
    scores = scores or {method: (.8, .7) for method in METHODS}
    costs = costs or {method: 1 for method in METHODS}
    tuning = tuning or {method: 10 for method in METHODS}
    rows = []
    for n, seed, attribute, method in itertools.product(PROTOCOL["training_sizes"], PROTOCOL["training_seeds"], PROTOCOL["attributes"], METHODS):
        ba = scores[method][PROTOCOL["attributes"].index(attribute)]
        rows.append({"jobid": f"{n}-{seed}-{attribute}-{method}", "n_train": n, "seed": seed, "attribute": attribute,
                     "method": method, "qualified": True, "issues": [], "metrics": {key: ba for key in (*selector.METRICS, "prevalence")},
                     "tuning_compute_seconds": tuning[method], "timing_pairs": [{"fit": costs[method] * .8, "prediction": costs[method] * .2,
                     "stages": {"total_fit_seconds": costs[method] * .8, "prediction_seconds": costs[method] * .2}} for _ in range(3)],
                     "cv_failed_candidates": []})
    return rows


def write(path, payload, receipt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"payload": payload, "sha256": selector.digest(payload)} if receipt else payload
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def build_fixture(run, protocol_path):
    jobs, plans, caches = [], [], {}
    ids = [f"validation-{i}" for i in range(4000)]
    validation = {"image_ids": ids, "metadata": {"partition_code": 1, "random_state": 42},
                  "attributes": {attribute: {"data_sha256": selector.digest(attribute), "class_counts": [2000, 2000]}
                                 for attribute in PROTOCOL["attributes"]}}
    for n, seed in itertools.product(PROTOCOL["training_sizes"], PROTOCOL["training_seeds"]):
        cache = str((run / f"cache-{n}-{seed}").resolve())
        manifest = {"official_test_loaded": False, "seed": seed,
                    "splits": {"train": {"image_ids": [f"train-{seed}-{i}" for i in range(n)], "metadata": {"partition_code": 0},
                               "attributes": {a: {"data_sha256": selector.digest([n, seed, a]), "class_counts": [n // 2, n // 2]} for a in PROTOCOL["attributes"]}},
                               "validation": validation}}
        caches[cache] = {"manifest": manifest, "manifest_sha256": selector.digest(manifest)}
        for attribute, method in itertools.product(PROTOCOL["attributes"], METHODS):
            screening = "percentile" if method.endswith("percentile") else "score_threshold"
            screen_value = 2 if screening == "percentile" else 2 * math.sqrt(n / 4000)
            grid = [{"n_components": [rank], screening: [screen_value], "n_neighbors": PROTOCOL["stage_2"]["n_neighbors"]} for rank in [2, 3, 4]]
            job = {"id": f"n{n}-s{seed}-{attribute}-{method}", "cache": cache, "seed": seed, "attribute": attribute, "method": method,
                   "grid": grid, "max_iter": 20000, "tol": .0001, "n_train": n}
            candidates = list(ParameterGrid(grid))
            chunks = [{"id": f"c{i:04d}", "candidate_indices": list(range(i * 10, (i + 1) * 10)), "grid": [box]} for i, box in enumerate(grid)]
            jobs.append(job)
            plans.append({"job": job, "candidates": candidates, "chunks": chunks})
    choices = {attribute: {method: {"pairs": [{"parameters": {"n_components": rank,
               "percentile" if method.endswith("percentile") else "score_threshold": 2}} for rank in [2, 3, 4]]}
               for method in METHODS} for attribute in PROTOCOL["attributes"]}
    rng, blocks, ordered_plans = random.Random(42043), [], []
    indexed = {(plan["job"]["n_train"], plan["job"]["seed"], plan["job"]["attribute"], plan["job"]["method"]): plan for plan in plans}
    for n, seed, attribute in itertools.product(PROTOCOL["training_sizes"], PROTOCOL["training_seeds"], PROTOCOL["attributes"]):
        order = list(METHODS)
        rng.shuffle(order)
        blocks.append({"n_train": n, "seed": seed, "attribute": attribute, "method_order": order})
        ordered_plans.extend(indexed[n, seed, attribute, method] for method in order)
    plans = ordered_plans
    jobs = [plan["job"] for plan in plans]
    declared = {"schema_version": 1, "author": "Chang Cheng", "cv": 5, "native_threads": 1, "profile_repeats": 3,
                "decision_protocol_sha256": selector.hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "jobs": jobs,
                "training_only_shortlists": choices,
                "timing_order": {"seed": 42043, "blocks": blocks, "three_profiles_are_consecutive_within_each_method": True}}
    configuration = {"protocol": declared, "plans": plans, "caches": caches, "official_test_loaded": False,
                     "sources": {"package_source_sha256": {"synthetic.py": "fixture"}}}
    identity = selector.digest(configuration)
    write(run / "configuration.json", configuration, True)
    truth = [0] * 2000 + [1] * 2000
    predicted = truth[:]
    probability = [.1] * 2000 + [.9] * 2000
    metrics = {"balanced_accuracy": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
               "average_precision": 1.0, "roc_auc": 1.0, "positive_class": 1,
               "confusion_matrix_labels": [0, 1], "confusion_matrix": [[2000, 0], [0, 2000]]}
    summaries, evaluations = {}, {}
    for plan in plans:
        job = plan["job"]
        directory = run / job["id"]
        rows, receipts = [], {}
        for chunk in plan["chunks"]:
            candidates = [{"index": i, "parameters": point, "valid": True, "mean_balanced_accuracy": 1.0,
                           "folds": [{"fold": fold, "status": "valid", "balanced_accuracy": 1.0} for fold in range(5)]}
                          for i, point in enumerate(ParameterGrid(chunk["grid"]))]
            calibration = {"candidates": list(ParameterGrid(chunk["grid"])), "cv": 5, "seed": job["seed"], "threads": 1, "refit": False,
                           "data_sha256": caches[job["cache"]]["manifest"]["splits"]["train"]["attributes"][job["attribute"]]["data_sha256"],
                           "source_sha256": configuration["sources"]["package_source_sha256"],
                           "parameters": {"max_iter": job["max_iter"], "tol": job["tol"], "random_state": job["seed"],
                                          "solver": "cd" if "_cd_" in job["method"] else "spa",
                                          "screening": "percentile" if job["method"].endswith("percentile") else "absolute"}}
            chunk_identity = {"calibration": calibration}
            report = {"state": "cv_completed", "identity": chunk_identity, "fingerprint": selector.digest(chunk_identity),
                      "candidates": candidates, "compute_seconds": {"representation_fit": 1.0, "classifier_prediction": 1.0}}
            receipts[chunk["id"]] = selector.digest(report)
            write(directory / chunk["id"] / "report.json", report, True)
            write(directory / chunk["id"] / "identity.json", chunk_identity, True)
            rows.extend({**row, "index": index, "chunk": chunk["id"]} for index, row in zip(chunk["candidate_indices"], candidates))
        summary = {"author": "Chang Cheng", "job": job, "configuration_sha256": identity, "state": "cv_completed", "candidates": rows,
                   "selected_candidate": rows[0], "selected_parameters": rows[0]["parameters"], "cv_balanced_accuracy": 1.0,
                   "compute_seconds": {"representation_fit": 3.0, "classifier_prediction": 3.0}, "tuning_compute_seconds": 6.0,
                   "report_receipts": receipts, "official_test_loaded": False}
        summaries[job["id"]] = summary
        write(directory / "calibration-summary.json", summary, True)
        evaluation = {"status": "valid", "metrics": metrics, "prediction_seconds": .1, "warnings": [],
                      "data_sha256": validation["attributes"][job["attribute"]]["data_sha256"],
                      "predictions": {"sample_ids": ids, "true_labels": truth, "predicted_labels": predicted,
                                      "positive_class_probability": probability}}
        profiles = [{"repeat": repeat, "status": "valid", "exact_prediction_match": True, "evaluation": evaluation,
                     "timing": {"total_fit_seconds": fit, "prediction_seconds": .1}, "fit_warnings": []}
                    for repeat, fit in enumerate([1.0, 1.1, .9])]
        result = {"author": "Chang Cheng", "job": job, "identity": {"configuration_sha256": identity, "job_id": job["id"],
                  "selected_candidate_sha256": selector.digest(rows[0]), "validation_data_sha256": evaluation["data_sha256"]},
                  "jobid": job["id"], "attribute": job["attribute"], "method": job["method"], "seed": job["seed"],
                  "n_train": len(caches[job["cache"]]["manifest"]["splits"]["train"]["image_ids"]),
                  "selected_parameters": summary["selected_parameters"], "cv_balanced_accuracy": 1.0, "tuning_compute_seconds": 6.0,
                  "official_test_loaded": False, "is_fallback": False, "final_fit_seconds": 1.0, "evaluation": evaluation,
                  "profiling": profiles, "profiling_valid": True,
                  "profile_medians_seconds": {"total_fit_seconds": 1.0, "prediction_seconds": .1, "fit_plus_prediction_seconds": 1.1}}
        evaluations[job["id"]] = result
        write(directory / "evaluation.json", result, True)
    write(run / "cv-summary.json", {"state": "cv_completed", "configuration_sha256": identity, "jobs": summaries,
                                   "missing_chunks": [], "official_test_loaded": False}, True)
    write(run / "comparison.json", {"state": "development_evaluation_completed", "configuration_sha256": identity,
                                   "jobs": evaluations, "official_test_loaded": False}, True)
    return configuration, summaries, evaluations


class DecisionTests(unittest.TestCase):
    def test_accuracy_gate_rejects_fast_but_inferior(self):
        scores = {method: (.8, .7) for method in METHODS}
        scores[METHODS[1]] = (.77, .67)
        costs = dict.fromkeys(METHODS, 2)
        costs[METHODS[1]] = .1
        result = selector.choose(rows_for(scores, costs), PROTOCOL)
        self.assertEqual(result["winner"], METHODS[0])
        self.assertNotIn(METHODS[1], result["accuracy_acceptable_methods"])

    def test_attribute_gate_protects_rare_attribute(self):
        scores = dict.fromkeys(METHODS, (.8, .7))
        scores[METHODS[1]] = (.83, .67)
        result = selector.choose(rows_for(scores), PROTOCOL)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["winner"], METHODS[0])

    def test_empty_gate_minimum_worst_regret(self):
        scores = {METHODS[0]: (.9, .5), METHODS[1]: (.7, .7), METHODS[2]: (.5, .9), METHODS[3]: (.65, .65)}
        result = selector.choose(rows_for(scores), PROTOCOL)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["winner"], METHODS[1])

    def test_zero_margin_fallback_has_no_division_by_zero(self):
        scores = {METHODS[0]: (.9, .5), METHODS[1]: (.7, .7), METHODS[2]: (.5, .9), METHODS[3]: (.65, .65)}
        result = selector.choose(rows_for(scores), PROTOCOL, macro_margin=0)
        self.assertEqual(result["winner"], METHODS[1])

    def test_five_percent_band_and_secondary_tuning(self):
        costs = {METHODS[0]: 1, METHODS[1]: 1.05, METHODS[2]: 2, METHODS[3]: 2}
        tuning = {METHODS[0]: 10, METHODS[1]: 1, METHODS[2]: 1, METHODS[3]: 1}
        self.assertEqual(selector.choose(rows_for(costs=costs, tuning=tuning), PROTOCOL)["winner"], METHODS[1])
        costs[METHODS[1]] = 1.05001
        self.assertEqual(selector.choose(rows_for(costs=costs, tuning=tuning), PROTOCOL)["winner"], METHODS[0])

    def test_macro_then_stable_tie_order(self):
        scores = dict.fromkeys(METHODS, (.8, .7))
        scores[METHODS[2]] = (.801, .701)
        self.assertEqual(selector.choose(rows_for(scores), PROTOCOL)["winner"], METHODS[2])
        self.assertEqual(selector.choose(rows_for(), PROTOCOL)["winner"], METHODS[0])

    def test_one_invalid_case_excludes_entire_method(self):
        rows = rows_for()
        rows[0]["qualified"] = False
        result = selector.choose(rows, PROTOCOL)
        self.assertNotIn(METHODS[0], result["qualified_methods"])
        self.assertEqual(result["winner"], METHODS[1])
        for row in rows:
            row["qualified"] = False
        self.assertIsNone(selector.choose(rows, PROTOCOL)["winner"])

    def test_repeated_medians_use_paired_sums(self):
        row = rows_for()[0]
        row["timing_pairs"] = [{"fit": 1, "prediction": 100}, {"fit": 2, "prediction": 2}, {"fit": 100, "prediction": 1}]
        self.assertEqual(selector.case_cost(row), 101)
        self.assertNotEqual(selector.case_cost(row), statistics.median([1, 2, 100]) + statistics.median([100, 2, 1]))
        self.assertEqual(selector.case_cost(row, 10), 110)

    def test_geometric_cost_not_arithmetic_cost(self):
        rows = rows_for(costs={METHODS[0]: 1, METHODS[1]: 10, METHODS[2]: 20, METHODS[3]: 20})
        method_zero = [row for row in rows if row["method"] == METHODS[0]]
        method_zero[-1]["timing_pairs"] = [{"fit": 800, "prediction": 200} for _ in range(3)]
        self.assertGreater(statistics.fmean(selector.case_cost(row) for row in method_zero), 10)
        self.assertEqual(selector.choose(rows, PROTOCOL)["winner"], METHODS[0])

    def test_numerical_warning_invalidates_claimed_valid_profile(self):
        predictions = dict.fromkeys(selector.PREDICTION_FIELDS, [0, 1])
        evaluation = {"status": "valid", "predictions": predictions, "prediction_seconds": 1}
        profiles = [{"repeat": i, "status": "valid", "exact_prediction_match": True, "evaluation": deepcopy(evaluation),
                     "timing": {"total_fit_seconds": 1, "prediction_seconds": 1},
                     "fit_warnings": [{"category": "ConvergenceWarning", "message": "cap"}] if i == 1 else []} for i in range(3)]
        result = {"evaluation": evaluation, "profiling": profiles, "profiling_valid": True}
        self.assertIn("nonconvergence", selector.profile_qualification(result, 3)[0][0])

    def test_any_fit_or_evaluation_warning_invalidates_timing(self):
        predictions = dict.fromkeys(selector.PREDICTION_FIELDS, [0, 1])
        evaluation = {"status": "valid", "predictions": predictions, "prediction_seconds": 1}
        clean = [{"repeat": i, "status": "valid", "exact_prediction_match": True, "evaluation": deepcopy(evaluation),
                  "timing": {"total_fit_seconds": 1, "prediction_seconds": 1}, "fit_warnings": []} for i in range(3)]
        for where in ("fit", "evaluation"):
            profiles = deepcopy(clean)
            warning = [{"category": "UserWarning", "message": "synthetic non-convergence-independent warning"}]
            if where == "fit":
                profiles[1]["fit_warnings"] = warning
            else:
                profiles[1]["evaluation"]["warnings"] = warning
            issues, _ = selector.profile_qualification({"evaluation": evaluation, "profiling": profiles, "profiling_valid": True}, 3)
            self.assertIn("warnings invalidate timing", issues[0])

    def test_missing_and_duplicate_cases_rejected(self):
        rows = rows_for()
        with self.assertRaisesRegex(ValueError, "complete decision cases"):
            selector.choose(rows[1:], PROTOCOL)
        with self.assertRaisesRegex(ValueError, "Duplicate decision cases"):
            selector.choose(rows + [rows[0]], PROTOCOL)

    def test_profiles_checked_not_merely_trusted(self):
        predictions = dict.fromkeys(selector.PREDICTION_FIELDS, [0, 1])
        evaluation = {"status": "valid", "predictions": predictions, "prediction_seconds": 1}
        profiles = [{"repeat": i, "status": "valid", "exact_prediction_match": True, "evaluation": deepcopy(evaluation),
                     "timing": {"total_fit_seconds": 1, "prediction_seconds": 1}} for i in range(3)]
        result = {"evaluation": evaluation, "profiling": profiles, "profiling_valid": True}
        self.assertEqual(selector.profile_qualification(result, 3)[0], [])
        profiles[1]["evaluation"]["predictions"]["positive_class_probability"] = [0, .9]
        self.assertIn("independently checked", selector.profile_qualification(result, 3)[0][0])
        profiles.pop()
        self.assertIn("Missing", selector.profile_qualification(result, 3)[0][0])

    def test_sensitivity_does_not_replace_primary(self):
        report = selector.summarize(rows_for(), PROTOCOL)
        self.assertEqual(len(report["sensitivity"]), 32)
        self.assertEqual(len(report["leave_one_out"]), 6)
        self.assertEqual(report["primary"]["winner"], METHODS[0])

    def test_markdown_table_method_labels_escape_pipes(self):
        costs = {method: 1 for method in METHODS}
        costs[METHODS[1]] = .5
        report = selector.summarize(rows_for(costs=costs), PROTOCOL)
        report.update(source_run="synthetic fixture", decision_protocol_sha256="synthetic fixture")
        text = selector.markdown(report)
        self.assertIn("CD + absolute \\|t\\|", text)
        width = None
        for line in text.splitlines():
            if not line.startswith("|"):
                width = None
                continue
            count = len(re.findall(r"(?<!\\)\|", line)) - 1
            if width is None:
                width = count
            self.assertEqual(count, width, line)


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.protocol_path = Path(cls.temporary.name) / "synthetic-decision-protocol.json"
        write(cls.protocol_path, PROTOCOL)
        cls.fixture_run = Path(cls.temporary.name) / "fixture"
        cls.configuration, cls.summaries, cls.evaluations = build_fixture(cls.fixture_run, cls.protocol_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_72_case_report_and_plot(self):
        output = Path(self.temporary.name) / "report"
        has_matplotlib = importlib.util.find_spec("matplotlib") is not None
        with patch.dict(os.environ, {"MPLCONFIGDIR": str(Path(self.temporary.name) / "matplotlib")}):
            result = selector.write_report(self.fixture_run, self.protocol_path, output, chart=has_matplotlib)
        self.assertEqual(result["winner_method"], METHODS[0])
        self.assertEqual(len(result["cases"]), 72)
        self.assertEqual(len((output / "cases.csv").read_text().splitlines()), 73)
        if has_matplotlib:
            self.assertTrue((output / "accuracy-cost.png").stat().st_size > 1000)
        self.assertTrue(all(row["qualified_cases"] == 18 for row in result["methods"]))

    def test_missing_declared_case_blocks_before_winner(self):
        path = self.fixture_run / "configuration.json"
        original = path.read_bytes()
        broken = deepcopy(self.configuration)
        broken["plans"].pop()
        write(path, broken, True)
        try:
            with self.assertRaisesRegex(ValueError, "72 declared jobs"):
                selector.load_cases(self.fixture_run, self.protocol_path)
        finally:
            path.write_bytes(original)

    def test_checksum_tamper_rejected(self):
        jobid = self.configuration["plans"][0]["job"]["id"]
        path = self.fixture_run / jobid / "evaluation.json"
        original = path.read_bytes()
        value = json.loads(original)
        value["payload"]["tuning_compute_seconds"] = 999
        path.write_text(json.dumps(value))
        try:
            with self.assertRaisesRegex(ValueError, "checksum"):
                selector.load_cases(self.fixture_run, self.protocol_path)
        finally:
            path.write_bytes(original)

    def test_wrong_selected_candidate_digest_rejected(self):
        jobid = self.configuration["plans"][0]["job"]["id"]
        path = self.fixture_run / jobid / "evaluation.json"
        original = path.read_bytes()
        value = deepcopy(self.evaluations[jobid])
        value["identity"]["selected_candidate_sha256"] = "wrong"
        write(path, value, True)
        try:
            with self.assertRaisesRegex(ValueError, "not bound"):
                selector.load_cases(self.fixture_run, self.protocol_path)
        finally:
            path.write_bytes(original)

    def test_metrics_recomputed(self):
        jobid = self.configuration["plans"][0]["job"]["id"]
        evaluation = deepcopy(self.evaluations[jobid]["evaluation"])
        evaluation["metrics"]["balanced_accuracy"] = .9
        cache = self.configuration["plans"][0]["job"]["cache"]
        validation = self.configuration["caches"][cache]["manifest"]["splits"]["validation"]
        with self.assertRaisesRegex(ValueError, "balanced_accuracy differs"):
            selector.checked_metrics(evaluation, validation, "Smiling")

    def test_transferred_rank_must_match_shortlist(self):
        declared = deepcopy(self.configuration["protocol"])
        originals = {job["id"]: deepcopy(job) for job in declared["jobs"]}
        first = declared["jobs"][0]
        originals[first["id"]]["grid"][0]["n_components"] = [99]
        with self.assertRaisesRegex(ValueError, "exactly transfer"):
            selector.check_transfer_and_timing(declared, originals, PROTOCOL)

    def test_absolute_cutoff_must_use_literal_size_scaling(self):
        declared = deepcopy(self.configuration["protocol"])
        originals = {job["id"]: deepcopy(job) for job in declared["jobs"]}
        job = next(job for job in declared["jobs"] if job["n_train"] == 2000 and job["method"].endswith("absolute"))
        originals[job["id"]]["grid"][0]["score_threshold"] = [2]
        with self.assertRaisesRegex(ValueError, "exactly transfer"):
            selector.check_transfer_and_timing(declared, originals, PROTOCOL)

    def test_timing_metadata_must_reproduce_seed(self):
        declared = deepcopy(self.configuration["protocol"])
        originals = {job["id"]: deepcopy(job) for job in declared["jobs"]}
        declared["timing_order"]["blocks"][0]["method_order"].reverse()
        with self.assertRaisesRegex(ValueError, "frozen randomization seed"):
            selector.check_transfer_and_timing(declared, originals, PROTOCOL)

    def test_actual_job_order_must_follow_timing_blocks(self):
        declared = deepcopy(self.configuration["protocol"])
        originals = {job["id"]: deepcopy(job) for job in declared["jobs"]}
        declared["jobs"][0], declared["jobs"][1] = declared["jobs"][1], declared["jobs"][0]
        with self.assertRaisesRegex(ValueError, "Actual job order"):
            selector.check_transfer_and_timing(declared, originals, PROTOCOL)

    def make_preflight(self):
        root = Path(self.temporary.name) / "preflight"
        original = deepcopy(self.configuration["protocol"])
        original_path, audit_path = root / "original.json", root / "audit.json"
        write(original_path, original)
        actual, masks, audits = deepcopy(original), {}, []
        for job in actual["jobs"]:
            kept, excluded = [], []
            for point in selector.expanded_candidates(job["grid"]):
                setting = ({"screening": "percentile", "percentile": point["percentile"]} if "percentile" in point
                           else {"screening": "absolute", "score_threshold": point["score_threshold"]})
                mask_key = json.dumps([job["cache"], job["attribute"], job["seed"], setting], sort_keys=True, separators=(",", ":"))
                masks[mask_key] = [{"fold": fold, "selected_features": 3, "n_train": 1600} for fold in range(5)]
                if point["n_components"] <= 3:
                    kept.append({key: [value] for key, value in point.items()})
                else:
                    excluded.append({"parameters": point, "mask_key": mask_key, "reason": "rank exceeds retained-feature dimension"})
            job.update(grid=kept, original_declared_candidates=30, declared_candidates=20,
                       excluded_infeasible_candidates=10, nominal_representation_folds=10)
            audits.append({"id": job["id"], "kept_candidates": 20, "excluded": excluded})
        original_sha = selector.hashlib.sha256(original_path.read_bytes()).hexdigest()
        audit = {"input_protocol_sha256": original_sha, "training_only": True, "decomposition_or_classifier_fits": 0,
                 "official_test_loaded": False, "jobs": audits, "fold_masks": masks}
        write(audit_path, audit)
        actual["preflight"] = {"input_protocol_path": str(original_path), "input_protocol_sha256": original_sha,
                               "audit_path": str(audit_path), "audit_sha256": selector.hashlib.sha256(audit_path.read_bytes()).hexdigest()}
        return actual, audit, audit_path

    def test_preflight_proves_complete_original_and_feasible_grids(self):
        actual, _, _ = self.make_preflight()
        jobs, exclusions = selector.check_preflight(actual, selector.Inputs())
        self.assertEqual(len(jobs), 72)
        self.assertEqual(sum(map(len, exclusions.values())), 720)
        self.assertTrue(all(len(selector.expanded_candidates(job["grid"])) == 30 for job in jobs.values()))

    def test_preflight_omission_without_dimensional_reason_rejected(self):
        actual, audit, path = self.make_preflight()
        for folds in audit["fold_masks"].values():
            for fold in folds:
                fold["selected_features"] = 100
        write(path, audit)
        actual["preflight"]["audit_sha256"] = selector.hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "dimensional proof"):
            selector.check_preflight(actual, selector.Inputs())

    def test_preflight_checksum_changed_rejected(self):
        actual, audit, path = self.make_preflight()
        audit["training_only"] = False
        write(path, audit)
        with self.assertRaisesRegex(ValueError, "Preflight proof identity"):
            selector.check_preflight(actual, selector.Inputs())


if __name__ == "__main__":
    unittest.main()
