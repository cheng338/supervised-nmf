"""Pure four-solver reporting checks. Author: Chang Cheng. No estimator fits."""

from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_snmf_test_four_solver_report", ROOT / "scripts/report_four_solver_benchmarks.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
METHODS = reporter.METHODS
CD, SPA, MU, BPP = METHODS
RULE = {"candidates": list(METHODS), "global_regret_limit": .01, "per_dataset_regret_limit": .01,
        "per_task_regret_limit": .02, "relative_cost_tie_band": .05, "numerical_comparison_slack": 1e-12,
        "sensitivities": {"global_and_dataset_regret_limits": [.005, .01, .02], "per_task_limit_multiplier": 2,
                          "prediction_cost_multipliers": [.1, 1, 10, 100]}}


def rows_for(scores=None, costs=None):
    scores = scores or {method: {"Fashion-MNIST": .8, "CelebA": .8} for method in METHODS}
    costs = costs or dict(zip(METHODS, (2, 1, 3, 4)))
    rows = []
    for dataset, tasks, configs in (("Fashion-MNIST", ["tshirt-shirt", "pullover-coat", "sneaker-boot"],
                                    list(itertools.product([2000, 8000], [43, 44, 45]))),
                                   ("CelebA", ["smiling", "wearing_necktie"], [(8000, 43)])):
        for task, (n_train, seed), method in itertools.product(tasks, configs, METHODS):
            score, cost = scores[method][dataset], costs[method]
            if isinstance(cost, dict):
                cost = cost[dataset]
            rows.append({"dataset": dataset, "task": task, "n_train": n_train, "seed": seed, "method": method,
                         "jobid": f"{dataset}-{task}-{n_train}-{seed}-{method}", "qualified": True, "issues": [],
                         "metrics": {key: score for key in (*reporter.METRICS, "negative_recall", "prevalence")},
                         "timing_pairs": [{"fit": .8 * cost, "prediction": .2 * cost,
                                           "stages": {"total_fit_seconds": .8 * cost, "prediction_seconds": .2 * cost}} for _ in range(3)],
                         "tuning_compute_seconds": 100, "cv_failed_candidates": 0, "excluded_candidate_count": 0,
                         "refinement": {"parameters": {}}, "diagnostics": {}})
    return rows


def evaluation_fixture():
    return {"status": "valid", "warnings": [], "predictions": {"sample_ids": ["a", "b", "c", "d"],
            "true_labels": [0, 0, 1, 1], "predicted_labels": [0, 0, 1, 1], "positive_class_probability": [.1, .2, .8, .9]},
            "metrics": {**{key: 1.0 for key in reporter.METRICS}, "confusion_matrix": [[2, 0], [0, 2]],
                        "confusion_matrix_labels": [0, 1], "positive_class": 1}}


def receipt_fixture(root):
    """Produce coherent synthetic receipts, never fit an estimator or read real data."""
    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"payload": payload, "sha256": reporter.digest(payload)}), encoding="utf-8")
    protocol = {"methods": list(METHODS), "attributes": ["Smiling", "Wearing_Necktie"], "n_train": 20,
                "seed": 43, "profile_repeats": 3, "profile_order_seed": 42043, "max_iter": 20000, "tol": .0001,
                "anls_max_iter": 1000, "anls_nnls_max_iter": None, "anls_nnls_tol": 1e-10,
                "grid": {"n_components": [2], "score_threshold": [0], "n_neighbors": [1]}}
    cache = root / "labels"
    cache.mkdir()
    np.save(cache / "y.npy", np.array([0, 0, 1, 1]))
    label_hash = reporter.hashlib.sha256((cache / "y.npy").read_bytes()).hexdigest()
    folds = [{"training_indices": [i for i in range(20) if i % 5 != fold],
              "validation_indices": [i for i in range(20) if i % 5 == fold]} for fold in range(5)]
    configuration = {"protocol": protocol, "official_test_loaded": False, "sources": {"package_source_sha256": {"source": "fixture"}},
                     "arrays": {}, "splits": {}, "plans": [], "preflight": {}}
    for attribute in protocol["attributes"]:
        task = attribute.lower()
        configuration["arrays"][f"{task}/validation"] = {"path": str(cache), "shape": [4, 3], "class_counts": [2, 2],
                    "files_sha256": {"y.npy": label_hash}, "data_sha256": "validation-fixture"}
        configuration["splits"][task] = {"validation_indices": ["a", "b", "c", "d"]}
        configuration["preflight"][task] = {"fold_indices_sha256": reporter.digest(folds), "folds": [
            {"fold": fold, "training_size": 16, "screens": {"0": {"features": 3}}} for fold in range(5)]}
        for method in METHODS:
            direct = method in METHODS[:2]
            keys = ("n_components", "score_threshold", "n_neighbors") if direct else ("nmf__n_components", "screen__score_threshold", "knn__n_neighbors")
            params = dict(zip(keys, (2, 0, 1)))
            job = {"id": f"{task}-{method}", "pair": {"id": task, "negative": 0, "positive": 1}, "attribute": attribute,
                   "method": method, "n_train": 20, "seed": 43, "array_key": task, "training_data_sha256": "training-fixture",
                   **{key: protocol[key] for key in ("max_iter", "tol", "anls_max_iter", "anls_nnls_max_iter", "anls_nnls_tol")}}
            configuration["plans"].append({"job": job, "declared_candidates": [params], "candidates": [{"index": 0, "parameters": params}],
                "chunks": [{"id": "c0000", "candidate_indices": [0], "grid": [{key: [value] for key, value in params.items()}]}],
                "excluded_representations": [], "fold_indices_sha256": reporter.digest(folds)})
    configuration["evaluation_order"] = [configuration["plans"][index]["job"]["id"]
        for index in np.random.RandomState(42043).permutation(8)]
    identity = reporter.digest(configuration)
    save(root / "configuration.json", configuration)
    summaries = {}
    for plan in configuration["plans"]:
        job, method = plan["job"], plan["job"]["method"]
        direct = method in METHODS[:2]
        if direct:
            params = {"solver": "cd" if method == CD else "spa", "screening": "absolute", "max_iter": 20000,
                      "tol": .0001, "random_state": 43, "weights": "uniform", "training_coordinates": "fit_transform"}
        else:
            params = {"screen__screening": "absolute", "nmf__max_iter": 20000 if method == MU else 1000,
                      "nmf__tol": .0001, "nmf__random_state": 43, "knn__weights": "uniform", "knn__n_jobs": 1}
            params.update({"nmf__solver": "mu", "nmf__init": "nndsvda", "nmf__beta_loss": "frobenius"} if method == MU
                          else {"nmf": "ANLSBPP()", "nmf__nnls_max_iter": None, "nmf__nnls_tol": 1e-10})
        checkpoint = {"folds": folds, "calibration": {"parameters": params, "estimator": "supervised_nmf.classifier.NMFClassifier" if direct else "sklearn.pipeline.Pipeline",
            "data_sha256": "training-fixture", "cv": 5, "seed": 43, "threads": 1, "refit": False,
            "candidates": plan["declared_candidates"], "source_sha256": configuration["sources"]["package_source_sha256"]}}
        candidate = {"index": 0, "parameters": plan["declared_candidates"][0], "valid": True, "mean_balanced_accuracy": .8,
                     "folds": [{"fold": fold, "status": "valid", "balanced_accuracy": .8, "warnings": []} for fold in range(5)]}
        chunk = {"identity": checkpoint, "fingerprint": reporter.digest(checkpoint), "state": "cv_completed", "candidates": [candidate], "compute_seconds": {"representation": 1.0}}
        selected = {**candidate, "chunk": "c0000"}
        summary = {"job": job, "configuration_sha256": identity, "state": "cv_completed", "declared_candidates": plan["declared_candidates"],
            "excluded_representations": [], "report_receipts": {"c0000": reporter.digest(chunk)}, "candidates": [selected], "selected_candidate": selected,
            "selected_parameters": selected["parameters"], "cv_balanced_accuracy": .8, "compute_seconds": {"representation": 1.0}, "tuning_compute_seconds": 1.0,
            "refinement": {"parameters": {}}}
        summaries[job["id"]] = summary
        save(root / job["id"] / "c0000" / "report.json", chunk)
        save(root / job["id"] / "c0000" / "identity.json", checkpoint)
        save(root / job["id"] / "calibration-summary.json", summary)
    cv = {"state": "cv_completed", "configuration_sha256": identity, "missing_chunks": [], "jobs": summaries}
    gate = {"configuration_sha256": identity, "cv_sha256": reporter.digest(cv), "evaluation_order": configuration["evaluation_order"], "all_cv_completed": True}
    save(root / "cv-summary.json", cv)
    save(root / "validation-gate.json", gate)
    records = {}
    for plan in configuration["plans"]:
        job, summary = plan["job"], summaries[plan["job"]["id"]]
        record = {"job": job, "jobid": job["id"], "method": job["method"], "n_train": 20, "seed": 43,
            "identity": {"gate_sha256": reporter.digest(gate), "job_id": job["id"], "selected_candidate_sha256": reporter.digest(summary["selected_candidate"]),
                         "validation_data_sha256": "validation-fixture"},
            "selected_parameters": summary["selected_parameters"], "cv_balanced_accuracy": .8, "is_fallback": False, "official_test_loaded": False,
            "tuning_compute_seconds": 1., "final_fit_seconds": 1., "final_fit_warnings": [], "diagnostics": {}, "evaluation": evaluation_fixture(),
            "profiling": [], "profiling_valid": False, "profile_medians_seconds": {}}
        save(root / job["id"] / "original-evaluation.json", record)
        original_digest = reporter.digest(record)
        for repeat in range(3):
            profile = {"status": "valid", "repeat": repeat, "exact_prediction_match": True, "fit_warnings": [], "evaluation": evaluation_fixture(),
                       "timing": {"total_fit_seconds": 1., "prediction_seconds": .1},
                       "identity": {"original_evaluation_sha256": original_digest, "repeat": repeat}}
            record["profiling"].append(profile)
            save(root / job["id"] / f"profile-{repeat}.json", profile)
        record.update(profiling_valid=True, profile_medians_seconds={"fit_plus_prediction_seconds": 1.1})
        records[job["id"]] = record
        save(root / job["id"] / "evaluation.json", record)
    save(root / "comparison.json", {"state": "development_evaluation_completed", "configuration_sha256": identity,
                                   "official_test_loaded": False, "jobs": records})
    return configuration


class FourSolverRuleTests(unittest.TestCase):
    def test_matched_baselines_exclude_other_sizes_and_seeds(self):
        rows = rows_for()
        baselines = ["screened_pca_absolute", "screening_only_absolute", "raw_knn"]
        protocol = {"baseline_train_size": 8000, "baseline_seed": 43, "baseline_methods": baselines,
                    "pairs": [{"id": task} for task in ("tshirt-shirt", "pullover-coat", "sneaker-boot")]}
        for row in rows:
            if row["dataset"] == "Fashion-MNIST":
                row["metrics"]["balanced_accuracy"] = .9 if row["n_train"] == 8000 and row["seed"] == 43 else .1
        for template in [row for row in rows if row["dataset"] == "Fashion-MNIST" and row["method"] == CD
                         and row["n_train"] == 8000 and row["seed"] == 43]:
            for method in baselines:
                row = deepcopy(template)
                row.update(method=method, jobid=template["jobid"].replace(CD, method))
                rows.append(row)
        table = reporter.matched_fashion_baselines(rows, protocol)
        self.assertEqual(len(table["rows"]), 7)
        for row in table["rows"]:
            self.assertEqual(row["case_count"], 3)
            self.assertAlmostEqual(row["metrics_descriptive_available_task_means"]["balanced_accuracy"], .9)
            self.assertEqual(row["qualified_cases"], 3)
            self.assertTrue(all("-8000-43-" in job for job in row["jobids"]))
        # Missing baselines remain an evidence error, never silently fewer tasks.
        with self.assertRaisesRegex(ValueError, "lacks"):
            reporter.matched_fashion_baselines(rows[:-1], protocol)

    def test_datasets_have_equal_weight_despite_18_vs_2_cases(self):
        scores = {method: {"Fashion-MNIST": .9, "CelebA": .5} for method in METHODS}
        decision = reporter.choose(rows_for(scores), RULE)
        self.assertAlmostEqual(decision["scores"][CD]["global"], .7)
        weights = decision["case_weights"]
        self.assertAlmostEqual(sum(row["weight"] for row in weights if row["case"][0] == "CelebA"), .5)
        self.assertAlmostEqual(sum(row["weight"] for row in weights if row["case"][0] == "Fashion-MNIST"), .5)

    def test_geometric_cost_uses_same_dataset_weights(self):
        costs = {CD: {"Fashion-MNIST": 2, "CelebA": 8}, SPA: 1, MU: 3, BPP: 4}
        decision = reporter.choose(rows_for(costs=costs), RULE)
        self.assertAlmostEqual(decision["scores"][CD]["normalized_cost"], 4)

    def test_dataset_gate_blocks_small_global_regret(self):
        scores = {method: {"Fashion-MNIST": .8, "CelebA": .8} for method in METHODS}
        scores[SPA] = {"Fashion-MNIST": .82, "CelebA": .785}
        decision = reporter.choose(rows_for(scores), RULE)
        self.assertLess(decision["scores"][SPA]["global_regret"], .01)
        self.assertFalse(decision["scores"][SPA]["accuracy_acceptable"])
        self.assertNotIn(SPA, decision["accuracy_acceptable_methods"])

    def test_task_gate_blocks_compensating_task_scores(self):
        rows = rows_for()
        for row in rows:
            if row["method"] == SPA and row["dataset"] == "Fashion-MNIST":
                row["metrics"]["balanced_accuracy"] = .83 if row["task"] != "tshirt-shirt" else .755
        decision = reporter.choose(rows, RULE)
        self.assertFalse(decision["scores"][SPA]["accuracy_acceptable"])
        self.assertGreater(decision["scores"][SPA]["task_regrets"]["Fashion-MNIST/tshirt-shirt"], .02)

    def test_five_percent_cost_band_prefers_ba_not_tuning(self):
        scores = {method: {"Fashion-MNIST": .8, "CelebA": .8} for method in METHODS}
        scores[CD] = {"Fashion-MNIST": .805, "CelebA": .805}
        rows = rows_for(scores, {CD: 1.04, SPA: 1, MU: 4, BPP: 5})
        for row in rows:
            if row["method"] == CD:
                row["tuning_compute_seconds"] = 10000
        decision = reporter.choose(rows, RULE)
        self.assertEqual(decision["methods_within_cost_band"], [CD, SPA])
        self.assertEqual(decision["winner"], CD)

    def test_empty_gate_uses_normalized_worst_regret_then_cost(self):
        scores = {CD: {"Fashion-MNIST": .9, "CelebA": .7}, SPA: {"Fashion-MNIST": .7, "CelebA": .9},
                  MU: {"Fashion-MNIST": .79, "CelebA": .79}, BPP: {"Fashion-MNIST": .78, "CelebA": .78}}
        decision = reporter.choose(rows_for(scores), RULE)
        self.assertTrue(decision["fallback_used"])
        self.assertEqual(decision["winner"], MU)

    def test_any_unqualified_case_disqualifies_solver(self):
        rows = rows_for()
        next(row for row in rows if row["method"] == SPA)["qualified"] = False
        decision = reporter.choose(rows, RULE)
        self.assertNotIn(SPA, decision["eligible_methods"])
        self.assertEqual(decision["winner"], CD)

    def test_no_eligible_solver_and_duplicate_cases(self):
        rows = rows_for()
        for row in rows:
            row["qualified"] = False
        self.assertEqual(reporter.choose(rows, RULE)["status"], "blocked_no_eligible_method")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reporter.choose(rows + [deepcopy(rows[0])], RULE)

    def test_missing_solver_case_rejected_instead_of_partial_mean(self):
        rows = rows_for()
        with self.assertRaisesRegex(ValueError, "coverage"):
            reporter.choose(rows[:-1], RULE)

    def test_seed43_omission_cannot_silently_drop_celeba(self):
        report = reporter.describe(rows_for(), RULE)
        omission = next(row for row in report["omissions"] if row["omitted_field"] == "seed" and row["omitted_value"] == 43)
        self.assertEqual(omission["status"], "skipped_missing_dataset_scope")
        self.assertEqual(len(report["sensitivity"]), 24)

    def test_pair_median_not_sum_of_stage_medians(self):
        row = {"timing_pairs": [{"fit": 1, "prediction": 100}, {"fit": 100, "prediction": 1}, {"fit": 2, "prediction": 4}]}
        self.assertEqual(reporter.case_cost(row), 101)

    def test_prediction_metrics_recomputed_and_tampering_rejected(self):
        evaluation = evaluation_fixture()
        result = reporter.verify_metrics(evaluation, ["a", "b", "c", "d"])
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertEqual(result["prevalence"], .5)
        evaluation["metrics"]["average_precision"] = .9
        with self.assertRaisesRegex(ValueError, "average_precision"):
            reporter.verify_metrics(evaluation, ["a", "b", "c", "d"])

    def test_profile_warnings_and_false_exactness_disqualify(self):
        evaluation = evaluation_fixture()
        profiles = [{"status": "valid", "repeat": index, "exact_prediction_match": True, "fit_warnings": [],
                     "evaluation": deepcopy(evaluation), "timing": {"total_fit_seconds": 1, "prediction_seconds": .1}}
                    for index in range(3)]
        self.assertEqual(reporter.profile_issues({"profiling": profiles}, evaluation)[0], [])
        profiles[0]["fit_warnings"] = [{"category": "RuntimeWarning", "message": "fixture"}]
        profiles[1]["evaluation"]["predictions"]["positive_class_probability"][0] = .11
        issues, pairs = reporter.profile_issues({"profiling": profiles}, evaluation)
        self.assertEqual(len(issues), 2)
        self.assertEqual(len(pairs), 3)

    def test_receipt_checksum_and_input_change_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            payload = {"value": 1}
            path.write_text(json.dumps({"payload": payload, "sha256": reporter.digest(payload)}), encoding="utf-8")
            inputs = reporter.Inputs()
            self.assertEqual(inputs.read(path), payload)
            path.write_text(json.dumps({"payload": {"value": 2}, "sha256": reporter.digest(payload)}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                inputs.verify()
            with self.assertRaisesRegex(ValueError, "checksum"):
                reporter.Inputs().read(path)

    def test_complete_receipt_graph_metrics_profiles_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = receipt_fixture(root)
            inputs = reporter.Inputs()
            rows, actual, executions = reporter.load_dataset(root, "CelebA", inputs)
            self.assertEqual(actual, configuration)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["qualified"] for row in rows))
            self.assertTrue(all(row["metrics"]["balanced_accuracy"] == 1 for row in rows))
            self.assertEqual(executions, [])
            self.assertEqual(inputs.missing, [])
            self.assertGreater(len(inputs.hashes), 65)
            inputs.verify()
            # A separately damaged profile must fail even if comparison remains intact.
            profile = root / configuration["plans"][0]["job"]["id"] / "profile-1.json"
            wrapped = json.loads(profile.read_text())
            wrapped["payload"]["timing"]["prediction_seconds"] = .2
            wrapped["sha256"] = reporter.digest(wrapped["payload"])
            profile.write_text(json.dumps(wrapped))
            with self.assertRaisesRegex(ValueError, "Profile receipt"):
                reporter.load_dataset(root, "CelebA", reporter.Inputs())

    def test_grid_and_solver_settings_are_independently_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            configuration = receipt_fixture(Path(directory))
            plan = deepcopy(configuration["plans"][0])
            plan["declared_candidates"][0]["n_components"] = 3
            with self.assertRaisesRegex(ValueError, "Declared grid"):
                reporter.verify_plan(plan, configuration)
            calibration = json.loads((Path(directory) / plan["job"]["id"] / "c0000" / "identity.json").read_text())["payload"]["calibration"]
            calibration["parameters"]["max_iter"] = 100
            with self.assertRaisesRegex(ValueError, "numerical settings"):
                reporter.verify_solver_parameters(calibration, plan["job"])


if __name__ == "__main__":
    unittest.main()
