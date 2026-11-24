"""Four-solver reporting checks and a tiny current calibration receipt fixture.

Author: Chang Cheng. Historical decision fixtures remain unchanged.
"""

from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import ParameterGrid

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


def four_solver_drivers():
    spec = importlib.util.spec_from_file_location("_timed_celeba_four_solver", ROOT / "scripts/celeba_four_solver_study.py")
    celeba = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(celeba)
    return celeba.shared, celeba


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


class CurrentSchemaReceiptTests(unittest.TestCase):
    def test_current_calibration_receipts_and_rechecksummed_tampering(self):
        from scripts.fashion_mnist_study import _estimator
        from supervised_nmf.study import calibration_knn_grid

        def save(path, payload):
            path.write_text(json.dumps({"payload": payload, "sha256": reporter.digest(payload)}), encoding="utf-8")

        rng = np.random.default_rng(735)
        y = np.arange(40) % 2
        X = rng.uniform(.1, 1, (40, 8)) + y[:, None] * .2
        grid = {"n_components": [2], "score_threshold": [0], "n_neighbors": [3]}
        job = {"id": "current-spa", "method": SPA, "seed": 7, "max_iter": 1000, "tol": 1e-4}
        points = list(ParameterGrid(grid))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = calibration_knn_grid(_estimator(job), grid, X, y, cv=5, seed=7,
                                          scoring="balanced_accuracy", checkpoint_dir=directory / "c0000")
            calibration = report["identity"]["calibration"]
            job["training_data_sha256"] = calibration["data_sha256"]
            configuration = {"sources": {"package_source_sha256": calibration["source_sha256"]}}
            plan = {"job": job, "declared_candidates": points, "excluded_representations": [],
                "candidates": [{"index": 0, "parameters": points[0]}],
                "chunks": [{"id": "c0000", "candidate_indices": [0], "grid": grid}],
                "fold_indices_sha256": reporter.digest(report["identity"]["folds"])}
            candidates = [{**row, "chunk": "c0000"} for row in report["candidates"]]
            self.assertTrue(candidates[0]["valid"])
            summary = {"state": "cv_completed", "job": job, "configuration_sha256": reporter.digest(configuration),
                "declared_candidates": points, "excluded_representations": [],
                "candidates": candidates, "selected_candidate": candidates[0], "selected_parameters": points[0],
                "cv_balanced_accuracy": candidates[0]["mean_balanced_accuracy"],
                "compute_seconds": report["compute_seconds"], "tuning_compute_seconds": sum(report["compute_seconds"].values()),
                "report_receipts": {"c0000": reporter.digest(report)}}
            self.assertEqual(reporter.verify_cv(reporter.Inputs(), directory, plan, configuration, summary), candidates[0])
            relocated = deepcopy(report)
            def relocate(value):
                if isinstance(value, dict):
                    if set(value) == {"path", "sha256"}:
                        value["path"] = "/archive/relocated/source/" + value["path"].rsplit("/", 1)[-1]
                    else:
                        for item in value.values():
                            relocate(item)
                elif isinstance(value, list):
                    for item in value:
                        relocate(item)
            relocated_calibration = relocated["identity"]["calibration"]
            relocate(relocated_calibration["structural_configuration"])
            relocated_calibration["implementation_source"] = {
                key.split("@", 1)[0] + "@/archive/relocated/source/" + key.rsplit("/", 1)[-1]: value
                for key, value in relocated_calibration["implementation_source"].items()}
            relocated["fingerprint"] = reporter.digest(relocated["identity"])
            relocated_summary = deepcopy(summary)
            relocated_summary["report_receipts"]["c0000"] = reporter.digest(relocated)
            save(directory / "c0000" / "report.json", relocated)
            save(directory / "c0000" / "identity.json", relocated["identity"])
            self.assertEqual(reporter.verify_cv(reporter.Inputs(), directory, plan, configuration, relocated_summary), candidates[0])

            def republish(changed):
                # Keep checksums and independent summary copies consistent so
                # rejection must reach the receipt contract under test.
                changed["fingerprint"] = reporter.digest(changed["identity"])
                changed_summary = deepcopy(summary)
                rows = [{**row, "chunk": "c0000"} for row in changed["candidates"]]
                self.assertEqual(len(rows), 1)
                changed_summary.update(candidates=rows, selected_candidate=rows[0],
                    selected_parameters=rows[0]["parameters"],
                    cv_balanced_accuracy=rows[0]["mean_balanced_accuracy"])
                changed_summary["report_receipts"]["c0000"] = reporter.digest(changed)
                save(directory / "c0000" / "report.json", changed)
                save(directory / "c0000" / "identity.json", changed["identity"])
                return changed_summary

            missing = object()
            different_fold = .25 if report["candidates"][0]["folds"][0]["score"] != .25 else .75
            different_mean = .25 if report["selected_mean_score"] != .25 else .75
            for target, field, value, message in (
                    ("report", "scoring", "accuracy", "scoring"),
                    ("identity", "scoring", "accuracy", "scoring"),
                    ("calibration", "scoring", "accuracy", "scoring"),
                    ("calibration", "identity_schema", 1, "schema"),
                    ("report", "schema_version", 4.0, "schema"),
                    ("identity", "schema_version", 4.0, "schema"),
                    ("calibration", "identity_schema", 2.0, "schema"),
                    ("report", "refit_performed", 0, "final refit"),
                    ("report", "refit_performed", True, "final refit"),
                    ("report", "refit_performed", missing, "final refit"),
                    ("candidate", "valid", 1, "qualification must be boolean"),
                    ("candidate", "valid", 1.0, "qualification must be boolean"),
                    ("fold", "score", different_fold, "score aliases"),
                    ("fold", "score", missing, "score aliases"),
                    ("candidate", "mean_score", different_mean, "score aliases"),
                    ("candidate", "mean_score", missing, "score aliases"),
                    ("report", "selected_mean_score", different_mean, "score aliases"),
                    ("report", "selected_mean_score", missing, "score aliases"),
                    ("report", "selected_mean_balanced_accuracy", different_mean, "score aliases"),
                    ("report", "selected_mean_balanced_accuracy", missing, "score aliases"),
                    ("calibration", "structural_configuration", {}, "identity"),
                    ("calibration", "implementation_source", {}, "source")):
                with self.subTest(target=target, field=field, value="missing" if value is missing else value):
                    broken = deepcopy(report)
                    records = {"report": broken, "identity": broken["identity"],
                               "calibration": broken["identity"]["calibration"],
                               "candidate": broken["candidates"][0],
                               "fold": broken["candidates"][0]["folds"][0]}
                    if value is missing:
                        records[target].pop(field)
                    else:
                        records[target][field] = value
                    changed_summary = republish(broken)
                    with self.assertRaisesRegex(ValueError, message):
                        reporter.verify_cv(reporter.Inputs(), directory, plan, configuration, changed_summary)

            for score in (True, "0.8", 2.0):
                with self.subTest(coherent_score=score):
                    broken = deepcopy(report)
                    row = broken["candidates"][0]
                    for fold in row["folds"]:
                        fold.update(score=score, balanced_accuracy=score)
                    # Numeric means preserve the arithmetic for Boolean/string
                    # fold values; equal aliases alone must not admit them.
                    row.update(mean_score=float(score), mean_balanced_accuracy=float(score))
                    broken.update(selected_mean_score=float(score), selected_mean_balanced_accuracy=float(score))
                    changed_summary = republish(broken)
                    with self.assertRaisesRegex(ValueError, "finite number in"):
                        reporter.verify_cv(reporter.Inputs(), directory, plan, configuration, changed_summary)
            restored_summary = republish(deepcopy(report))
            self.assertEqual(reporter.verify_cv(reporter.Inputs(), directory, plan, configuration, restored_summary), candidates[0])

    def test_structural_solver_settings_cover_all_four_methods(self):
        from scripts.fashion_mnist_study import _estimator
        from supervised_nmf._identity import configuration_identity, implementation_sources

        for method in METHODS:
            with self.subTest(method=method):
                job = {"method": method, "seed": 7, "max_iter": 1000, "tol": 1e-4,
                       "anls_max_iter": 200, "anls_nnls_max_iter": 500, "anls_nnls_tol": 1e-10}
                estimator = _estimator(job)
                direct = method in METHODS[:2]
                candidates = [{"n_components" if direct else "nmf__n_components": 2,
                               "score_threshold" if direct else "screen__score_threshold": 0,
                               "n_neighbors" if direct else "knn__n_neighbors": 3}]
                calibration = {"identity_schema": 2, "scoring": "balanced_accuracy",
                    "structural_configuration": configuration_identity({"estimator": estimator, "candidates": candidates}),
                    "implementation_source": implementation_sources(estimator, candidates)}
                reporter.verify_solver_parameters(calibration, job, candidates)
                with self.assertRaisesRegex(ValueError, "identity"):
                    reporter.verify_solver_parameters(calibration, {**job, "tol": 1e-3}, candidates)


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

    def test_cv_wall_history_follows_registered_order_and_audits_staged_attempt(self):
        from supervised_nmf.study import _write_receipt

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            journal = run / "cv-timing"
            index = journal / "attempts.index"
            config = reporter.digest({"dataset": "CelebA"})
            first = {"attempt_id": "a" * 32, "configuration_sha256": config, "previous_sha256": None,
                     "state": "completed", "started_utc": "2999-01-01", "cv_wall_seconds_this_invocation": 3.}
            second = {**first, "attempt_id": "b" * 32, "previous_sha256": reporter.digest(first),
                      "state": "admitted", "started_utc": "2000-01-01", "cv_wall_seconds_this_invocation": 0.}
            finished, staged = journal / (first["attempt_id"] + ".json"), journal / ".starting"
            ids = [first["attempt_id"], second["attempt_id"]]
            catalog = {"attempts": ids, "terminal_sha256": None}
            _write_receipt(finished, first)
            _write_receipt(staged, second)
            _write_receipt(index, catalog)
            inputs = reporter.Inputs()
            self.assertEqual(reporter.cv_wall_history(run, config, inputs, []), [first, second])
            self.assertEqual(set(inputs.hashes), {str(p.resolve()) for p in (index, finished, staged)})
            with self.assertRaises(ValueError):
                reporter.cv_wall_history(run, "another configuration", reporter.Inputs(), [])
            _write_receipt(index, {**catalog, "attempts": ids + ids[-1:]})
            with self.assertRaisesRegex(ValueError, "changed"):
                inputs.verify()
            with self.assertRaises(ValueError):
                reporter.cv_wall_history(run, config, reporter.Inputs(), [])
            _write_receipt(index, catalog)
            duplicate = journal / "copy.json"
            duplicate.write_bytes(finished.read_bytes())
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                reporter.cv_wall_history(run, config, reporter.Inputs(), [])
            duplicate.unlink()
            staged.unlink()
            with self.assertRaisesRegex(ValueError, "Missing registered"):
                reporter.cv_wall_history(run, config, reporter.Inputs(), [])
            index.unlink()
            with self.assertRaisesRegex(ValueError, "Missing attempt index"):
                reporter.cv_wall_history(run, config, reporter.Inputs(), [])

    def test_cv_wall_history_handles_historical_archives_and_missing_timed_evidence(self):
        from supervised_nmf.study import _write_receipt

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            configuration = receipt_fixture(run)
            identity = reporter.digest(configuration)
            self.assertEqual(reporter.cv_wall_history(run, identity, reporter.Inputs(), []), [])
            _write_receipt(run / "executions" / "cv.json", {"configuration_sha256": identity, "mode": "cv",
                           "cv": {"cv_wall_seconds_this_invocation": 2.}})
            _, _, executions = reporter.load_dataset(run, "CelebA", reporter.Inputs())
            with self.assertRaisesRegex(ValueError, "Missing CV timing history"):
                reporter.cv_wall_history(run, identity, reporter.Inputs(), executions)
            row = {"configuration_sha256": identity, "started_utc": "2025-01-01",
                   "cv_wall_seconds_this_invocation": 2.}
            _write_receipt(run / "cv-timing" / "old.json", row)
            self.assertEqual(reporter.cv_wall_history(run, identity, reporter.Inputs(), executions), [row])
            with self.assertRaisesRegex(ValueError, "Missing CV timing history"):
                reporter.cv_wall_history(run, identity, reporter.Inputs(), executions + executions)
            with self.assertRaisesRegex(ValueError, "another configuration"):
                reporter.cv_wall_history(run, "another configuration", reporter.Inputs(), executions)

    def test_both_cli_writers_bind_completed_cv_and_zero_work_replay_to_their_receipts(self):
        from supervised_nmf import study

        shared, celeba = four_solver_drivers()
        configuration = {"sources": {}, "protocol": {"workers": 1}}
        identity = reporter.digest(configuration)
        for driver in (shared, celeba):
            with self.subTest(driver=driver.__name__), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                run = Path(tmp) / "run"
                argv = ["study", "--mode", "cv", "--output", str(run), "--protocol", "unused"]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(driver, "initialize", return_value=configuration))
                check = stack.enter_context(patch.object(shared, "_check_source", side_effect=ValueError("before CV")))
                # Rejection before CV admission creates a command record without a timing attempt.
                with self.assertRaisesRegex(ValueError, "before CV"):
                    driver.main()
                def executions():
                    return reporter.read_executions(run, identity, reporter.Inputs())
                def history():
                    return reporter.cv_wall_history(run, identity, reporter.Inputs(), executions())
                self.assertEqual(history(), [])
                check.side_effect = None
                chunk = (str(run), {"job": {"id": "case"}}, {"id": "chunk", "grid": {}}, {})
                work = stack.enter_context(patch.object(shared, "_protocol_cv_chunks", return_value=([chunk], [run / "chunk"])))
                stack.enter_context(patch.object(shared, "_cv_work_pending", return_value=True))
                stack.enter_context(patch.object(shared, "aggregate_cv", return_value={"state": "cv_completed", "missing_chunks": []}))
                pool = stack.enter_context(patch.object(shared, "ProcessPoolExecutor"))
                completed = Future()
                completed.set_result({"job": "case", "chunk": "chunk", "valid_candidates": 1, "candidates": 1})
                pool.return_value.__enter__.return_value.submit.return_value = completed
                # Save the genuine in-flight version of each numbered receipt.
                running = {}
                write = study._write_receipt
                def save(path, row):
                    write(path, row)
                    if path.suffix == ".json" and path.parent.name == "cv-timing" and row["state"] == "running":
                        running[row["attempt_id"]] = path.read_bytes()
                stack.enter_context(patch.object(study, "_write_receipt", side_effect=save))
                driver.main()
                index = run / "cv-timing" / "attempts.index"
                original_index = index.read_bytes()
                first, = history()
                self.assertGreater(first["cv_wall_seconds_this_invocation"], 0.)
                work.return_value = ([], [])
                driver.main()
                self.assertEqual(pool.call_count, 1)
                first, replay = history()
                self.assertEqual(replay["cv_wall_seconds_this_invocation"], 0.)
                commands = {row["cv_timing_attempt_id"]: row for row in executions() if row["state"] == "completed"}
                self.assertEqual(set(commands), {first["attempt_id"], replay["attempt_id"]})
                for receipt in (first, replay):
                    self.assertEqual(commands[receipt["attempt_id"]]["cv_timing_sha256"], reporter.digest(receipt))
                with self.assertRaisesRegex(ValueError, "timing receipt"):
                    reporter.cv_wall_history(Path(tmp) / "missing", identity, reporter.Inputs(), executions())
                sibling = Path(tmp) / "sibling"
                with study._cv_wall_attempt(sibling, identity, active=False, workers=1, native_threads=1):
                    pass
                with self.assertRaisesRegex(ValueError, "timing receipt"):
                    reporter.cv_wall_history(sibling, identity, reporter.Inputs(), executions())
                tail = index.parent / (replay["attempt_id"] + ".json")
                tail.write_bytes(running[replay["attempt_id"]])
                with self.assertRaises(ValueError):
                    history()
                tail.unlink()
                index.write_bytes(original_index)
                with self.assertRaisesRegex(ValueError, "timing receipt"):
                    history()

    def test_failed_commands_preserve_the_cv_phase_outcome_and_measured_wall(self):
        from supervised_nmf import study

        shared, celeba = four_solver_drivers()
        configuration = {"sources": {}, "protocol": {"workers": 1}}
        identity = reporter.digest(configuration)
        for driver in (shared, celeba):
            with self.subTest(driver=driver.__name__), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                run = Path(tmp) / "run"
                argv = ["study", "--mode", "cv", "--output", str(run), "--protocol", "unused"]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(driver, "initialize", return_value=configuration))
                stack.enter_context(patch.object(shared, "_check_source"))
                chunks = stack.enter_context(patch.object(shared, "_protocol_cv_chunks", return_value=([], [])))
                stack.enter_context(patch.object(shared, "_cv_work_pending", return_value=True))
                aggregate = stack.enter_context(patch.object(shared, "aggregate_cv", return_value={
                    "state": "cv_completed", "missing_chunks": []}))
                pool = stack.enter_context(patch.object(shared, "ProcessPoolExecutor"))
                submit = pool.return_value.__enter__.return_value.submit
                driver.main()  # A genuine earlier, complete journal.
                chunk = (str(run), {"job": {"id": "case"}}, {"id": "chunk", "grid": {}}, {})
                chunks.return_value = ([chunk], [run / "chunk"])
                cases = [("worker", RuntimeError("worker"), "failed", None),
                         ("interrupt", KeyboardInterrupt(), "interrupted", None),
                         ("interrupt-then-io", KeyboardInterrupt(), "interrupted", OSError("index publication")),
                         ("worker-then-interrupt", RuntimeError("worker"), "failed", KeyboardInterrupt()),
                         ("aggregate", ValueError("aggregate"), "completed", None)]
                if driver is shared:
                    cases.append(("evaluate", ValueError("evaluate"), "completed", None))
                for stage, error, cv_state, index_error in cases:
                    with self.subTest(stage=stage), ExitStack() as call:
                        prior = Path(tmp) / stage
                        shutil.copytree(run / "cv-timing", prior / "cv-timing")
                        future = Future()
                        if cv_state != "completed":
                            future.set_exception(error)
                        else:
                            future.set_result({"job": "case", "chunk": "chunk", "valid_candidates": 1, "candidates": 1})
                        submit.return_value = future
                        aggregate.side_effect = error if stage == "aggregate" else None
                        if stage == "evaluate":
                            argv[2] = "all"
                            stack.enter_context(patch.object(shared, "evaluate", side_effect=error))
                        if index_error is not None:
                            write_cv = study._write_receipt

                            def fail_terminal_index(path, payload):
                                if path.name == "attempts.index" and payload["terminal_sha256"] is not None:
                                    raise index_error
                                write_cv(path, payload)

                            call.enter_context(patch.object(study, "_write_receipt", side_effect=fail_terminal_index))
                        command_error = index_error or error
                        with self.assertRaises(type(command_error)):
                            driver.main()
                        commands = reporter.read_executions(run, identity, reporter.Inputs())
                        history = reporter.cv_wall_history(run, identity, reporter.Inputs(), commands)
                        receipt = history[-1]
                        command, = (c for c in commands if c["cv_timing_attempt_id"] == receipt["attempt_id"])
                        self.assertEqual(command["state"], "interrupted" if isinstance(command_error, KeyboardInterrupt) else "failed")
                        self.assertEqual(command["cv_timing_state"], cv_state)
                        self.assertEqual(receipt["state"], cv_state)
                        self.assertGreater(receipt["cv_wall_seconds_this_invocation"], 0.)
                        self.assertEqual(command["cv_timing_sha256"], reporter.digest(receipt))
                        if cv_state != "completed":
                            with self.assertRaisesRegex(ValueError, "timing receipt"):
                                reporter.cv_wall_history(run, identity, reporter.Inputs(), [{**command, "state": "completed"}])
                        with self.assertRaisesRegex(ValueError, "timing receipt"):
                            reporter.cv_wall_history(prior, identity, reporter.Inputs(), commands)

                # A hard stop can leave only the initially published running command.
                # The journal still includes a terminal CV attempt that has no independent witness.
                write = driver._write_receipt
                def stop_before_command_receipt(path, payload):
                    if path.parent.name == "executions" and payload["state"] != "running":
                        raise OSError("command receipt was not published")
                    write(path, payload)
                stack.enter_context(patch.object(driver, "_write_receipt", side_effect=stop_before_command_receipt))
                with self.assertRaisesRegex(OSError, "not published"):
                    driver.main()
                commands = reporter.read_executions(run, identity, reporter.Inputs())
                running, = (row for row in commands if row["state"] == "running")
                self.assertNotIn("cv_timing_attempt_id", running)
                attempts = reporter.cv_wall_history(run, identity, reporter.Inputs(), commands)
                self.assertEqual(len(attempts), len(commands))

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
