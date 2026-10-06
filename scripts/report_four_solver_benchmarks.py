"""Verify and summarize saved four-solver development benchmarks.

Author: Chang Cheng. This reporter fits no models, reads no images/test data,
and does not install a default. It applies the frozen prospective decision rule
to matched Fashion-MNIST and CelebA evidence, retaining unsuccessful cases.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("screened_cd_absolute", "screened_spa_absolute", "screened_mu_absolute", "screened_anls_bpp_absolute")
LABELS = dict(zip(METHODS, ("CD", "SPA", "MU", "ANLS-BPP")))
METRICS = ("balanced_accuracy", "accuracy", "roc_auc", "average_precision", "precision", "recall", "f1")
PREDICTION_FIELDS = ("sample_ids", "true_labels", "predicted_labels", "positive_class_probability")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value, label, *, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"Invalid {label}.")
    return float(value)


def close(actual, expected, label, tolerance=1e-10):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), f"{label} differs from saved evidence.")


class Inputs:
    def __init__(self):
        self.hashes, self.missing = {}, []

    def read(self, path, *, receipt=True, required=True):
        path = Path(path).resolve()
        if not path.exists():
            self.missing.append(str(path))
            require(not required, f"Missing required evidence: {path}")
            return None
        raw = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if receipt:
            require(set(value) == {"payload", "sha256"} and digest(value["payload"]) == value["sha256"],
                    f"Invalid receipt checksum: {path}")
            value = value["payload"]
        return value

    def labels(self, record):
        name = record.get("filenames", {}).get("y", "y.npy")
        path = (Path(record["path"]) / name).resolve()
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        require(raw_hash == record["files_sha256"][name], "Saved validation label checksum differs.")
        self.hashes[str(path)] = raw_hash
        values = np.load(path, allow_pickle=False)
        require(values.shape == (record["shape"][0],), "Saved validation label shape differs.")
        return values.tolist()

    def verify(self):
        for path, expected in self.hashes.items():
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"Evidence changed while reporting: {path}")


def nonconvergence(value):
    if isinstance(value, dict):
        return value.get("category") == "ConvergenceWarning" or any(
            (key.rstrip("_").endswith("converged") and item is False) or nonconvergence(item)
            for key, item in value.items())
    return isinstance(value, list) and any(nonconvergence(item) for item in value)


def verify_metrics(evaluation, sample_ids, true_labels=None):
    predictions = evaluation.get("predictions")
    require(isinstance(predictions, dict), "Valid evaluation lacks predictions.")
    require(predictions["sample_ids"] == sample_ids and len(set(map(str, sample_ids))) == len(sample_ids),
            "Validation image/row IDs differ or repeat.")
    y = np.asarray(predictions["true_labels"])
    predicted = np.asarray(predictions["predicted_labels"])
    probability = np.asarray(predictions["positive_class_probability"], dtype=float)
    require(y.shape == predicted.shape == probability.shape == (len(sample_ids),)
            and np.all((y == 0) | (y == 1)) and len(np.unique(y)) == 2
            and np.all((predicted == 0) | (predicted == 1)) and np.isfinite(probability).all()
            and np.all((probability >= 0) & (probability <= 1)), "Invalid binary predictions or probabilities.")
    if true_labels is not None:
        require(np.array_equal(y, true_labels), "Validation truth differs across methods or saved label cache.")
    metrics = {"balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
               "accuracy": float(accuracy_score(y, predicted)), "roc_auc": float(roc_auc_score(y, probability)),
               "average_precision": float(average_precision_score(y, probability)),
               "precision": float(precision_score(y, predicted, zero_division=0)),
               "recall": float(recall_score(y, predicted, zero_division=0)),
               "f1": float(f1_score(y, predicted, zero_division=0)), "prevalence": float(np.mean(y)),
               "negative_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
               "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}
    for key in METRICS:
        close(evaluation["metrics"][key], metrics[key], f"Validation {key}")
    require(evaluation["metrics"]["confusion_matrix"] == metrics["confusion_matrix"]
            and evaluation["metrics"]["confusion_matrix_labels"] == [0, 1]
            and evaluation["metrics"]["positive_class"] == 1, "Class orientation/confusion counts differ.")
    return metrics


def profile_issues(record, evaluation):
    issues, pairs = [], []
    profiles = record.get("profiling", [])
    if len(profiles) != 3 or [p.get("repeat") for p in profiles] != [0, 1, 2]:
        issues.append("three ordered profiling repeats are missing")
    for index, profile in enumerate(profiles):
        reproduced = profile.get("evaluation") or {}
        exact = (bool(evaluation.get("predictions")) and all(
            (reproduced.get("predictions") or {}).get(key) == evaluation["predictions"].get(key)
            for key in PREDICTION_FIELDS))
        if (profile.get("status") != "valid" or not profile.get("exact_prediction_match") or not exact
                or reproduced.get("status") != "valid" or profile.get("fit_warnings")
                or reproduced.get("warnings") or nonconvergence(profile)):
            issues.append(f"profile {index} is unqualified, warned, nonconverged or not exact")
        timing = profile.get("timing", {})
        try:
            fit = finite(timing.get("total_fit_seconds"), "profile total fit", positive=True)
            prediction = finite(timing.get("prediction_seconds"), "profile prediction", positive=True)
            pairs.append({"fit": fit, "prediction": prediction, "stages": timing})
        except ValueError:
            issues.append(f"profile {index} lacks finite positive total timing")
    return issues, pairs


def verify_plan(plan, configuration):
    """Reconstruct grid order and dimensional exclusions from frozen preflight evidence."""
    job, protocol = plan["job"], configuration["protocol"]
    for key in ("max_iter", "tol", "anls_max_iter", "anls_nnls_max_iter", "anls_nnls_tol"):
        require(job[key] == protocol[key], f"Job numerical setting differs: {key}")
    method = job["method"]
    direct = method in METHODS[:2]
    rank_key = "n_components" if direct else ("nmf__n_components" if method in METHODS else "pca__n_components")
    screen_key = "score_threshold" if direct else "screen__score_threshold"
    neighbor_key = "n_neighbors" if direct or method == "raw_knn" else "knn__n_neighbors"
    grid = protocol["grid"]
    ranks = grid["n_components"] if method in METHODS or method == "screened_pca_absolute" else [None]
    thresholds = grid["score_threshold"] if method != "raw_knn" else [None]
    representations = list(itertools.product(ranks, thresholds))
    representations.extend((pair["n_components"], pair["score_threshold"]) for pair in
                           protocol.get("extra_representations", {}).get(job.get("attribute"), []))
    representations = list(dict.fromkeys(representations))
    preflight = configuration["preflight"][job["array_key"]]
    require([fold["fold"] for fold in preflight["folds"]] == list(range(5)), "Preflight lacks five training folds.")
    declared, candidates, chunks, excluded = [], [], [], []
    for rank, threshold in representations:
        representation = {key: value for key, value in ((rank_key, rank), (screen_key, threshold)) if value is not None}
        parameters = [{**representation, neighbor_key: k} for k in grid["n_neighbors"]]
        indices = list(range(len(declared), len(declared) + len(parameters)))
        declared.extend(parameters)
        violations = []
        for fold in preflight["folds"]:
            count = fold["screens"][str(threshold)]["features"] if threshold is not None else None
            if count == 0 or (rank is not None and rank > min(count, fold["training_size"])):
                violations.append({"fold": fold["fold"], "selected_features": count, "training_size": fold["training_size"]})
        if violations:
            excluded.append({"parameters": representation, "candidate_indices": indices,
                             "reason": "training_fold_screen_or_rank_dimension", "folds": violations})
        else:
            candidates.extend({"index": index, "parameters": params} for index, params in zip(indices, parameters))
            chunks.append({"id": f"c{len(chunks):04d}", "candidate_indices": indices,
                           "grid": [{**{key: [value] for key, value in representation.items()}, neighbor_key: grid["n_neighbors"]}]})
    expected = {"job": job, "declared_candidates": declared, "candidates": candidates, "chunks": chunks,
                "excluded_representations": excluded, "fold_indices_sha256": preflight["fold_indices_sha256"]}
    require(plan == expected, "Declared grid, canonical order, extras or training-only dimensional exclusions differ.")


def verify_solver_parameters(calibration, job):
    params, method = calibration["parameters"], job["method"]
    if method in METHODS[:2]:
        expected = {"solver": "cd" if method == METHODS[0] else "spa", "screening": "absolute",
                    "max_iter": job["max_iter"], "tol": job["tol"], "random_state": job["seed"],
                    "training_coordinates": "fit_transform", "weights": "uniform"}
        require(calibration["estimator"] == "supervised_nmf.classifier.NMFClassifier", "Wrong direct estimator class.")
    elif method in METHODS[2:]:
        expected = {"screen__screening": "absolute", "nmf__max_iter": job["max_iter"] if method == METHODS[2] else job["anls_max_iter"],
                    "nmf__tol": job["tol"], "nmf__random_state": job["seed"], "knn__weights": "uniform", "knn__n_jobs": 1}
        if method == METHODS[2]:
            expected.update(nmf__solver="mu", nmf__init="nndsvda", nmf__beta_loss="frobenius")
        else:
            expected.update(nmf__nnls_max_iter=job["anls_nnls_max_iter"], nmf__nnls_tol=job["anls_nnls_tol"])
            require(str(params.get("nmf", "")).startswith("ANLSBPP("), "Wrong ANLS decomposition class.")
        require(calibration["estimator"] == "sklearn.pipeline.Pipeline", "Wrong pipeline estimator class.")
    else:
        expected = {}
    require(all(key in params and params[key] == value for key, value in expected.items()), "CV solver/numerical settings differ from job.")


def verify_cv(inputs, directory, plan, configuration, summary):
    job, identity = plan["job"], digest(configuration)
    require(summary["job"] == job and summary["configuration_sha256"] == identity and summary["state"] == "cv_completed",
            "CV summary job/configuration differs.")
    require(summary["declared_candidates"] == plan["declared_candidates"]
            and summary["excluded_representations"] == plan["excluded_representations"], "CV declared grid/exclusions differ.")
    all_rows, costs = {}, defaultdict(float)
    for chunk in plan["chunks"]:
        path = directory / chunk["id"]
        report = inputs.read(path / "report.json")
        checkpoint_identity = inputs.read(path / "identity.json")
        require(report["identity"] == checkpoint_identity and report["fingerprint"] == digest(checkpoint_identity)
                and summary["report_receipts"][chunk["id"]] == digest(report)
                and report["state"] == "cv_completed", "CV chunk receipt/identity mismatch.")
        calibration = checkpoint_identity["calibration"]
        verify_solver_parameters(calibration, job)
        expected = list(ParameterGrid(chunk["grid"]))
        require(calibration["data_sha256"] == job["training_data_sha256"] and calibration["cv"] == 5
                and calibration["seed"] == job["seed"] and calibration["threads"] == 1 and calibration["refit"] is False
                and calibration["candidates"] == expected and calibration["source_sha256"] == configuration["sources"]["package_source_sha256"]
                and digest(checkpoint_identity["folds"]) == plan["fold_indices_sha256"], "CV data, folds, source or grid differs.")
        require([row["parameters"] for row in report["candidates"]] == expected, "CV chunk coverage differs.")
        for index, row in zip(chunk["candidate_indices"], report["candidates"]):
            require(index not in all_rows, "CV candidate duplicated across chunks.")
            all_rows[index] = {**row, "index": index, "chunk": chunk["id"]}
        for name, value in report["compute_seconds"].items():
            costs[name] += finite(value, f"CV compute {name}")
    candidates = [all_rows[index] for index in sorted(all_rows)]
    require(candidates == summary["candidates"] and [(r["index"], r["parameters"]) for r in candidates]
            == [(r["index"], r["parameters"]) for r in plan["candidates"]], "CV summary does not cover the complete feasible grid.")
    for row in candidates:
        folds = row["folds"]
        require([fold["fold"] for fold in folds] == list(range(5)), "CV candidate lacks five folds.")
        valid = all(fold["status"] == "valid" for fold in folds)
        require(row["valid"] == valid, "CV qualification differs from five fold statuses.")
        if valid:
            require(not nonconvergence(folds), "Valid CV candidate records nonconvergence.")
            close(row["mean_balanced_accuracy"], float(np.mean([fold["balanced_accuracy"] for fold in folds])), "CV mean")
        else:
            require(row["mean_balanced_accuracy"] is None, "Failed CV candidate averaged only successful folds.")
    valid = [row for row in candidates if row["valid"]]
    selected = min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])) if valid else None
    require(summary["selected_candidate"] == selected and summary["selected_parameters"] == (selected["parameters"] if selected else None)
            and summary["cv_balanced_accuracy"] == (selected["mean_balanced_accuracy"] if selected else None), "CV winner or canonical tie order differs.")
    require(set(costs) == set(summary["compute_seconds"]), "CV compute fields differ.")
    for name, value in costs.items():
        close(summary["compute_seconds"][name], value, f"CV {name}", 1e-8)
    close(summary["tuning_compute_seconds"], sum(costs.values()), "CV total compute", 1e-8)
    return selected


def expected_cases(configuration, dataset):
    protocol = configuration["protocol"]
    if dataset == "Fashion-MNIST":
        return {(pair["id"], n, seed, method) for pair in protocol["pairs"] for n in protocol["train_sizes"]
                for seed in protocol["seeds"] for method in METHODS}
    return {(attribute.lower(), protocol["n_train"], protocol["seed"], method)
            for attribute in protocol["attributes"] for method in METHODS}


def load_dataset(run, dataset, inputs):
    configuration = inputs.read(run / "configuration.json")
    identity = digest(configuration)
    require(configuration.get("official_test_loaded") is False, "Study must exclude official-test data.")
    comparison = inputs.read(run / "comparison.json", required=False)
    cv = inputs.read(run / "cv-summary.json")
    gate = inputs.read(run / "validation-gate.json", required=False)
    plans = configuration["plans"]
    ids = [plan["job"]["id"] for plan in plans]
    require(len(ids) == len(set(ids)), "Duplicate job IDs.")
    cases = [(p["job"]["pair"]["id"], p["job"]["n_train"], p["job"]["seed"], p["job"]["method"])
             for p in plans if p["job"]["method"] in METHODS]
    require(len(cases) == len(set(cases)) and set(cases) == expected_cases(configuration, dataset), "Required solver case coverage differs.")
    require(configuration["protocol"]["methods"] == list(METHODS) and configuration["protocol"]["profile_repeats"] == 3,
            "Four solvers and three profiles must be declared.")
    require(cv["state"] == "cv_completed" and cv["configuration_sha256"] == identity and not cv["missing_chunks"]
            and set(cv["jobs"]) == set(ids), "CV is incomplete or mismatched.")
    if comparison is not None:
        require(comparison["state"] == "development_evaluation_completed" and comparison["configuration_sha256"] == identity
                and comparison["official_test_loaded"] is False and set(comparison["jobs"]) == set(ids), "Global comparison coverage/identity differs.")
    if gate is not None:
        require(gate == {"configuration_sha256": identity, "cv_sha256": digest(cv),
                         "evaluation_order": configuration["evaluation_order"], "all_cv_completed": True}, "Validation gate differs from complete CV.")
    require(len(configuration["evaluation_order"]) == len(ids) and set(configuration["evaluation_order"]) == set(ids), "Evaluation order coverage differs.")
    order = np.random.RandomState(configuration["protocol"]["profile_order_seed"]).permutation(len(plans))
    require(configuration["evaluation_order"] == [plans[index]["job"]["id"] for index in order], "Seeded serial evaluation order differs.")
    rows, truths = [], {}
    for plan in plans:
        job = plan["job"]
        verify_plan(plan, configuration)
        directory = run / job["id"]
        summary = inputs.read(directory / "calibration-summary.json")
        require(summary == cv["jobs"][job["id"]], "Global and individual CV summaries differ.")
        selected = verify_cv(inputs, directory, plan, configuration, summary)
        record = inputs.read(directory / "evaluation.json", required=False)
        row = {"dataset": dataset, "task": job["pair"]["id"], "method": job["method"], "jobid": job["id"],
               "n_train": job["n_train"], "seed": job["seed"], "qualified": False, "issues": [], "metrics": None,
               "selected_parameters": summary["selected_parameters"], "cv_balanced_accuracy": summary["cv_balanced_accuracy"],
               "tuning_compute_seconds": summary["tuning_compute_seconds"], "cv_candidate_count": len(summary["candidates"]),
               "cv_failed_candidates": sum(not r["valid"] for r in summary["candidates"]),
               "excluded_candidate_count": len(plan["declared_candidates"]) - len(plan["candidates"]),
               "refinement": summary["refinement"], "timing_pairs": [], "evaluation_status": "missing",
               "source_receipt": str((directory / "evaluation.json").resolve()), "diagnostics": {}}
        if record is None:
            row["issues"].append("missing selected evaluation receipt")
            rows.append(row)
            continue
        if comparison is not None:
            require(record == comparison["jobs"][job["id"]], "Global and individual evaluation receipts differ.")
        validation = configuration["arrays"][f"{job['pair']['id']}/validation"]
        expected_identity = {"gate_sha256": digest(gate) if gate is not None else record["identity"]["gate_sha256"],
                             "job_id": job["id"], "selected_candidate_sha256": digest(selected),
                             "validation_data_sha256": validation["data_sha256"]}
        require(record["identity"] == expected_identity and record["job"] == job and record["jobid"] == job["id"]
                and record["method"] == job["method"] and record["seed"] == job["seed"] and record["n_train"] == job["n_train"]
                and record["selected_parameters"] == summary["selected_parameters"]
                and record["cv_balanced_accuracy"] == summary["cv_balanced_accuracy"] and record.get("is_fallback") is False
                and record.get("official_test_loaded") is False, "Selected evaluation identity/parameters differ or fallback used.")
        close(record["tuning_compute_seconds"], summary["tuning_compute_seconds"], "Selected tuning compute", 1e-8)
        evaluation = record["evaluation"]
        row.update(evaluation_status=evaluation["status"], final_fit_seconds=record["final_fit_seconds"],
                   original_prediction_seconds=evaluation.get("prediction_seconds"), diagnostics=record.get("diagnostics", {}),
                   training_stationarity=record.get("training_stationarity", {"status": "unrecorded"}))
        ids_for_task = configuration["splits"][job["pair"]["id"]]["validation_indices"]
        if evaluation.get("predictions"):
            if job["pair"]["id"] not in truths:
                truths[job["pair"]["id"]] = inputs.labels(validation)
            row["metrics"] = verify_metrics(evaluation, ids_for_task, truths.get(job["pair"]["id"]))
            truths[job["pair"]["id"]] = evaluation["predictions"]["true_labels"]
            require(np.bincount(np.asarray(truths[job["pair"]["id"]]), minlength=2).tolist() == validation["class_counts"],
                    "Validation class counts differ from configuration.")
        if selected is None:
            row["issues"].append("no valid five-fold CV candidate")
        elif any(fold.get("warnings") for fold in selected["folds"]):
            row["issues"].append("selected CV candidate has warnings")
        if evaluation["status"] != "valid" or record.get("final_fit_warnings") or evaluation.get("warnings") or nonconvergence(record.get("diagnostics", {})):
            row["issues"].append("original selected fit/evaluation is unqualified or warned")
        if selected is not None:
            original = inputs.read(directory / "original-evaluation.json", required=False)
            if original is not None:
                require(all(original.get(key) == record.get(key) for key in ("identity", "job", "selected_parameters", "evaluation",
                            "final_fit_seconds", "final_fit_warnings", "diagnostics", "training_stationarity")), "Original evaluation differs from completed record.")
                for index, profile in enumerate(record["profiling"]):
                    separate = inputs.read(directory / f"profile-{index}.json", required=False)
                    if separate is not None:
                        require(separate == profile and profile["identity"] == {"original_evaluation_sha256": digest(original), "repeat": index},
                                "Profile receipt or original-fit binding differs.")
        issues, pairs = profile_issues(record, evaluation)
        row["issues"].extend(issues)
        row["timing_pairs"] = pairs
        row["qualified"] = not row["issues"]
        require(bool(record["profiling_valid"]) == row["qualified"], "Recorded profiling qualification differs from independent checks.")
        if row["qualified"]:
            close(record["profile_medians_seconds"]["fit_plus_prediction_seconds"], case_cost(row), "Paired timing median")
        rows.append(row)
    executions = []
    for path in sorted((run / "executions").glob("*.json")):
        record = inputs.read(path)
        require(record["configuration_sha256"] == identity, "Execution belongs to another configuration.")
        executions.append({key: record.get(key) for key in ("mode", "state", "started_at", "finished_at", "workers",
                                                           "native_threads", "current_call_wall_seconds")})
    return rows, configuration, executions


def case_key(row):
    return row["dataset"], row["task"], row["n_train"], row["seed"]


def weights_for(keys):
    datasets = sorted({key[0] for key in keys})
    weights = {}
    for dataset in datasets:
        tasks = sorted({key[1] for key in keys if key[0] == dataset})
        for task in tasks:
            local = [key for key in keys if key[:2] == (dataset, task)]
            for key in local:
                weights[key] = 1 / (len(datasets) * len(tasks) * len(local))
    return weights


def case_cost(row, multiplier=1, include_tuning=False):
    value = statistics.median(pair["fit"] + multiplier * pair["prediction"] for pair in row["timing_pairs"])
    return value + (row["tuning_compute_seconds"] if include_tuning else 0)


def choose(rows, rule, *, margin=None, prediction_multiplier=1, include_tuning=False, allowed_methods=None):
    rows = [row for row in rows if row["method"] in METHODS]
    require(rows, "No solver cases.")
    indexed = {(row["method"], case_key(row)): row for row in rows}
    require(len(indexed) == len(rows), "Duplicate solver decision case.")
    sets = {method: {case_key(row) for row in rows if row["method"] == method} for method in METHODS}
    require(all(sets[method] == sets[METHODS[0]] for method in METHODS), "Solver case coverage is unequal.")
    keys = sorted(sets[METHODS[0]])
    weight = weights_for(keys)
    eligible = [method for method in METHODS if all(indexed[method, key]["qualified"] for key in keys)
                and (allowed_methods is None or method in allowed_methods)]
    margin = rule["global_regret_limit"] if margin is None else margin
    dataset_margin = rule["per_dataset_regret_limit"] if margin == rule["global_regret_limit"] else margin
    task_margin = rule["per_task_regret_limit"] if margin == rule["global_regret_limit"] else margin * rule["sensitivities"]["per_task_limit_multiplier"]
    slack = rule.get("numerical_comparison_slack", 1e-12)
    result = {"winner": None, "eligible_methods": eligible, "global_margin": margin, "dataset_margin": dataset_margin,
              "task_margin": task_margin, "prediction_multiplier": prediction_multiplier, "include_tuning": include_tuning,
              "scores": {}, "fallback_used": False, "case_weights": [{"case": list(key), "weight": weight[key]} for key in keys]}
    if not eligible:
        return {**result, "status": "blocked_no_eligible_method"}
    for method in eligible:
        task_values = {f"{dataset}/{task}": statistics.fmean(indexed[method, key]["metrics"]["balanced_accuracy"]
                       for key in keys if key[:2] == (dataset, task)) for dataset, task in sorted({key[:2] for key in keys})}
        dataset_values = {dataset: statistics.fmean(value for name, value in task_values.items() if name.startswith(dataset + "/"))
                          for dataset in sorted({key[0] for key in keys})}
        result["scores"][method] = {"global": statistics.fmean(dataset_values.values()), "datasets": dataset_values, "tasks": task_values}
    scores = result["scores"]
    acceptable = []
    for method, score in scores.items():
        global_regret = max(s["global"] for s in scores.values()) - score["global"]
        dataset_regrets = {name: max(s["datasets"][name] for s in scores.values()) - value for name, value in score["datasets"].items()}
        task_regrets = {name: max(s["tasks"][name] for s in scores.values()) - value for name, value in score["tasks"].items()}
        normalized = [global_regret / margin, *[r / dataset_margin for r in dataset_regrets.values()],
                      *[r / task_margin for r in task_regrets.values()]]
        passed = global_regret <= margin + slack and all(r <= dataset_margin + slack for r in dataset_regrets.values()) \
                 and all(r <= task_margin + slack for r in task_regrets.values())
        score.update(global_regret=global_regret, dataset_regrets=dataset_regrets, task_regrets=task_regrets,
                     maximum_normalized_regret=max(normalized), accuracy_acceptable=passed)
        if passed:
            acceptable.append(method)
        score["normalized_cost"] = math.exp(sum(weight[key] * math.log(
            case_cost(indexed[method, key], prediction_multiplier, include_tuning)
            / min(case_cost(indexed[other, key], prediction_multiplier, include_tuning) for other in eligible)) for key in keys))
        score["geometric_workload_seconds"] = math.exp(sum(weight[key] * math.log(
            case_cost(indexed[method, key], prediction_multiplier, include_tuning)) for key in keys))
    result["accuracy_acceptable_methods"] = acceptable[:]
    if not acceptable:
        result["fallback_used"] = True
        regret = min(score["maximum_normalized_regret"] for score in scores.values())
        retained = [method for method in eligible if scores[method]["maximum_normalized_regret"] <= regret + slack]
        cost = min(scores[method]["normalized_cost"] for method in retained)
        retained = [method for method in retained if scores[method]["normalized_cost"] <= cost + slack]
        result["methods_within_cost_band"] = retained[:]
    else:
        cost = min(scores[method]["normalized_cost"] for method in acceptable)
        retained = [method for method in acceptable if scores[method]["normalized_cost"] <=
                    cost * (1 + rule["relative_cost_tie_band"]) + slack]
        result["methods_within_cost_band"] = retained[:]
        ba = max(scores[method]["global"] for method in retained)
        retained = [method for method in retained if scores[method]["global"] >= ba - slack]
        cost = min(scores[method]["normalized_cost"] for method in retained)
        retained = [method for method in retained if scores[method]["normalized_cost"] <= cost + slack]
    return {**result, "status": "selected_for_audit", "winner": next(method for method in METHODS if method in retained)}


def describe(rows, rule):
    primary = choose(rows, rule)
    sensitivities = [choose(rows, rule, margin=margin, prediction_multiplier=multiplier, include_tuning=include)
                     for include in (False, True) for margin in rule["sensitivities"]["global_and_dataset_regret_limits"]
                     for multiplier in rule["sensitivities"]["prediction_cost_multipliers"]]
    omissions = []
    solver_rows = [row for row in rows if row["method"] in METHODS]
    fields = [("dataset", sorted({r["dataset"] for r in solver_rows})),
              ("task", sorted({(r["dataset"], r["task"]) for r in solver_rows})),
              ("seed", sorted({r["seed"] for r in solver_rows}))]
    for field, values in fields:
        for value in values:
            subset = [r for r in solver_rows if ((r["dataset"], r["task"]) if field == "task" else r[field]) != value]
            record = {"omitted_field": field, "omitted_value": value}
            if not subset or (field != "dataset" and {r["dataset"] for r in subset} != {r["dataset"] for r in solver_rows}):
                record.update(status="skipped_missing_dataset_scope", reason="Omission removes every primary case of a dataset; not a comparable two-dataset sensitivity.")
            else:
                record.update(status="computed", decision=choose(subset, rule, allowed_methods=primary["eligible_methods"]))
            omissions.append(record)
    aggregates = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in dict.fromkeys(r["method"] for r in rows if r["dataset"] == dataset):
            subset = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            measured = [r for r in subset if r["metrics"]]
            qualified = [r for r in subset if r["qualified"]]
            stationarity = [r["training_stationarity"]["diagnostics"] for r in subset
                            if r.get("training_stationarity", {}).get("status") == "valid"]
            stages = sorted({key for r in qualified for pair in r["timing_pairs"] for key in pair["stages"]})
            aggregates.append({"dataset": dataset, "method": method, "case_count": len(subset), "qualified_cases": len(qualified),
                               "metrics_descriptive_available_case_means": {key: statistics.fmean(r["metrics"][key] for r in measured)
                                                                          if measured else None for key in (*METRICS, "negative_recall", "prevalence")},
                               "mean_case_median_fit_plus_prediction_seconds": statistics.fmean(case_cost(r) for r in qualified) if qualified else None,
                               "stage_means_of_case_medians": {key: statistics.fmean(statistics.median(pair["stages"][key] for pair in r["timing_pairs"])
                                    for r in qualified) if qualified and all(all(pair["stages"].get(key) is not None for pair in r["timing_pairs"])
                                    for r in qualified) else None for key in stages},
                               "recorded_cv_compute_seconds": sum(r["tuning_compute_seconds"] for r in subset),
                               "failed_cv_candidates": sum(r["cv_failed_candidates"] for r in subset),
                               "preflight_excluded_candidates": sum(r["excluded_candidate_count"] for r in subset),
                               "stationarity_recorded_cases": len(stationarity),
                               "stationarity_descriptive_means": {
                                   "raw_relative_projected_gradient": statistics.fmean(value["raw"]["relative_projected_gradient"] for value in stationarity) if stationarity else None,
                                   "canonical_relative_projected_gradient": statistics.fmean(value["canonical"]["relative_projected_gradient"] for value in stationarity) if stationarity else None,
                                   "relative_reconstruction_error": statistics.fmean(value["relative_reconstruction_error"] for value in stationarity) if stationarity else None},
                               "recorded_stopping_definitions": sorted({r.get("diagnostics", {}).get("stopping_criterion") for r in subset
                                                                      if r.get("diagnostics", {}).get("stopping_criterion")}),
                               "recorded_basis_coordinate_dtypes": sorted({(r.get("diagnostics", {}).get("basis_dtype", "unrecorded"),
                                                                 r.get("diagnostics", {}).get("training_coordinates_dtype", "unrecorded")) for r in subset})})
    return {"primary": primary, "sensitivity": sensitivities, "omissions": omissions, "aggregates": aggregates, "cases": rows}


def number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def matched_fashion_baselines(rows, protocol):
    """Compare all methods on precisely the baseline's declared size and seed."""
    size, seed = protocol["baseline_train_size"], protocol["baseline_seed"]
    tasks = {pair["id"] for pair in protocol["pairs"]}
    methods = [*METHODS, *protocol["baseline_methods"]]
    selected = [row for row in rows if row["dataset"] == "Fashion-MNIST"
                and row["n_train"] == size and row["seed"] == seed and row["method"] in methods]
    summaries = []
    for method in methods:
        local = [row for row in selected if row["method"] == method]
        require(len(local) == len(tasks) and {row["task"] for row in local} == tasks,
                "Matched Fashion baseline comparison lacks a declared method/task case.")
        measured, qualified = [row for row in local if row["metrics"]], [row for row in local if row["qualified"]]
        summaries.append({"method": method, "case_count": len(local), "metrics_available_cases": len(measured),
                          "qualified_cases": len(qualified), "jobids": [row["jobid"] for row in local],
                          "metrics_descriptive_available_task_means": {
                              key: statistics.fmean(row["metrics"][key] for row in measured) if measured else None
                              for key in METRICS},
                          "mean_case_median_fit_plus_prediction_seconds": statistics.fmean(case_cost(row) for row in qualified)
                          if qualified else None})
    return {"n_train": size, "seed": seed, "tasks": sorted(tasks), "rows": summaries,
            "role": "Matched configuration descriptive baseline comparison; baselines do not enter the four-solver decision."}


def markdown(report):
    primary = report["primary"]
    winner = primary["winner"]
    lines = ["# Four-solver development comparison", "", "Author: Chang Cheng.", "",
             f"**Decision status: {primary['status']}. Candidate: {LABELS.get(winner, 'none')}.**", "",
             "Absolute pooled-t screening is fixed. The reporter does not change the installed default. "
             "A MU/ANLS-BPP winner requires public-API integration and validation; a CD/SPA winner still requires final audit. "
             "All statements concern the declared binary image tasks, not universal algorithm superiority.", ""]
    if winner:
        score = primary["scores"][winner]
        lines += [f"Equal-dataset balanced accuracy: {number(score['global'])}; global regret: {100*score['global_regret']:.4f} percentage points; "
                  f"normalized weighted geometric workload cost: {number(score['normalized_cost'])}.", "",
                  "The gate permits 1 BA percentage point globally and per dataset and 2 per task. Within the eligible accuracy set, "
                  "the 5% cost band precedes higher global BA, lower cost and declared solver order. "
                  + ("**No solver passed every accuracy gate; minimum normalized worst-regret fallback selected a compromise.**"
                     if primary["fallback_used"] else "The selected solver passed the declared accuracy gates."), ""]
    lines += ["## Metrics and operational qualification", "",
              "Dataset rows are available-case descriptive means; only complete, warning-free cases enter decision eligibility. "
              "The primary rule weights datasets equally, then tasks and size/seed cases. Fashion NMF rows average 18 cases "
              "across both training sizes and all three seeds; each Fashion baseline row covers only three tasks at its one "
              "declared size/seed. Those broader rows are not a matched baseline comparison. Baselines are context only; "
              "use the following matched table for direct comparisons.", "",
              "| Dataset | Solver / method | Qualified cases | BA | Accuracy | AUC | AP | Positive recall | Negative recall | Precision | F1 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["aggregates"]:
        metrics = row["metrics_descriptive_available_case_means"]
        lines.append(f"| {row['dataset']} | {LABELS.get(row['method'], row['method'])} | {row['qualified_cases']}/{row['case_count']} | "
                     + " | ".join(number(metrics[key]) for key in ("balanced_accuracy", "accuracy", "roc_auc", "average_precision", "recall", "negative_recall", "precision", "f1")) + " |")
    matched = report["matched_fashion_baselines"]
    labels = {**LABELS, "screened_pca_absolute": "PCA + absolute screening", "screening_only_absolute": "Absolute screening + KNN", "raw_knn": "Raw KNN"}
    lines += ["", "## Matched Fashion-MNIST baseline comparison", "",
              f"All seven methods below use {matched['n_train']:,} training observations, seed {matched['seed']}, the same "
              f"{len(matched['tasks'])} binary tasks and each task's fixed validation sample. Each metric is the equal-task mean "
              "of available outcomes; counts expose held cases. Fit + predict is the mean of qualified task-level paired "
              "timing medians. This descriptive table does not change solver-selection weights or add baselines to the decision.", "",
              "| Method | Metrics / tasks | Qualified / tasks | BA | Accuracy | AUC | AP | Positive recall | Precision | F1 | Fit + predict (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in matched["rows"]:
        metrics = row["metrics_descriptive_available_task_means"]
        lines.append(f"| {labels.get(row['method'],row['method'])} | {row['metrics_available_cases']}/{row['case_count']} | "
                     f"{row['qualified_cases']}/{row['case_count']} | "
                     + " | ".join(number(metrics[key]) for key in ("balanced_accuracy", "accuracy", "roc_auc", "average_precision", "recall", "precision", "f1"))
                     + f" | {number(row['mean_case_median_fit_plus_prediction_seconds'],3)} |")
    lines += ["", "## Timing", "",
              "Serial costs use the median of three per-repeat total-fit plus prediction sums, on each dataset's declared validation batch "
              "(Fashion-MNIST 2,000; CelebA 4,000). Stage columns are separate means of case medians; nested stages must not be added "
              "again. CV compute is a sum of worker wall timers, not serial runtime or CPU time. Whole-process peak memory is unmeasured.", "",
              "| Dataset | Method | Mean paired median (s) | MF/PCA fit (s) | KNN fit (s) | Total fit (s) | Predict (s) | CV compute (s) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["aggregates"]:
        stages = row["stage_means_of_case_medians"]
        lines.append(f"| {row['dataset']} | {LABELS.get(row['method'], row['method'])} | {number(row['mean_case_median_fit_plus_prediction_seconds'],3)} | "
                     + " | ".join(number(stages.get(key), 3) for key in ("decomposition_fit_seconds", "classifier_fit_seconds", "total_fit_seconds", "prediction_seconds"))
                     + f" | {number(row['recorded_cv_compute_seconds'],2)} |")
    lines += ["", "Recorded command invocations (wall time includes command overhead; repeated/resumed calls remain separate):", ""]
    for dataset, executions in report["executions"].items():
        for row in executions:
            lines.append(f"- {dataset}, {row['mode']}, {row['state']}: {number(row['current_call_wall_seconds'],3)} s; {row['workers']} worker(s), native threads {row['native_threads']}.")
    if report.get("recovery"):
        recovery = report["recovery"]
        counts = recovery["counts"]
        lines += ["", "### Numerical-repair continuation", "",
                  recovery["amendment_reason"], "",
                  f"This versioned run inherits {counts.get('completed_reports',0):,} unchanged-method completed CV reports "
                  f"and {counts.get('partial_chunk_committed_folds',0)} committed folds from unfinished chunks. Original receipt bytes "
                  "and source provenance are preserved. All BPP candidates are rerun with the repaired implementation; old BPP "
                  "results remain historical diagnostics and do not enter this decision.", "",
                  f"Inherited recorded non-BPP compute: {number(recovery['inherited_cv_compute_seconds'],3)} s, already included "
                  "once in the method CV totals above. Do not add it again. Current invocation wall times exclude prior execution. "
                  "Prior-run wall records below include interrupted work where timing was saved; they are not additive method "
                  "compute or isolated solver-speed measurements. Uncommitted interrupted compute is unknown. Old BPP work and "
                  "repair qualification are additional historical overhead.", ""]
        for prior in recovery["parent_execution_receipts"]:
            record = prior["record"]
            lines.append(f"- Prior {record.get('mode','unknown')} invocation, {record.get('state','unknown')}: "
                         f"{number(record.get('current_call_wall_seconds'),3)} s, started {record.get('started_at','unrecorded')}. "
                         "Missing wall time is unmeasured, not zero.")
    lines += ["", "## Sensitivity", "",
              "Descriptive recalculations do not replace the primary rule. Including CV compute combines parallel worker-wall sums "
              "with serial deployment cost and is labeled separately. Prediction multipliers model repeated validation-batch workloads; "
              "they are not additional measured runtimes or a claim that one larger batch scales linearly.", "",
              "| Global/dataset margin (pp) | Prediction multiplier | Add tuning | Candidate | Fallback |",
              "| --- | ---: | --- | --- | --- |"]
    for row in report["sensitivity"]:
        lines.append(f"| {100*row['global_margin']:g} | {row['prediction_multiplier']:g} | {'yes' if row['include_tuning'] else 'no'} | "
                     f"{LABELS.get(row['winner'],'none')} | {'yes' if row['fallback_used'] else 'no'} |")
    lines += ["", "Omission checks keep primary operational eligibility fixed:", ""]
    for row in report["omissions"]:
        outcome = LABELS.get(row.get("decision", {}).get("winner"), "none") if row["status"] == "computed" else row["reason"]
        lines.append(f"- Omit {row['omitted_field']}={row['omitted_value']}: {outcome}.")
    lines += ["", "## Numerical diagnostics", "",
              "These are descriptive diagnostics at the stored training coordinates and basis. Relative projected gradient (PG) "
              "divides final PG by the final full-gradient norm. Canonical copies give each basis row unit L2 norm and compensate "
              "the corresponding coordinate column; fitted classifier geometry is unchanged. These values are not the algorithms' "
              "stopping ratios, a common convergence certificate, or proof of global optimality. SPA is anchor constrained and "
              "does not optimize the joint NMF objective. Iteration caps, tolerances and factor dtypes differ by algorithm; "
              "equal nominal tolerance does not imply equal optimization error. Diagnostics are timed separately from fit and prediction.", "",
              "| Dataset | Method | Recorded cases | Raw relative PG | Canonical relative PG | Relative reconstruction error | Basis / coordinate dtype |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in report["aggregates"]:
        diagnostic = row["stationarity_descriptive_means"]
        dtypes = ", ".join(" / ".join(pair) for pair in row["recorded_basis_coordinate_dtypes"])
        lines.append(f"| {row['dataset']} | {LABELS.get(row['method'],row['method'])} | {row['stationarity_recorded_cases']}/{row['case_count']} | "
                     + " | ".join(number(diagnostic[key], 6) for key in ("raw_relative_projected_gradient", "canonical_relative_projected_gradient", "relative_reconstruction_error"))
                     + f" | {dtypes} |")
    lines += ["", "## Held cases, boundaries and diagnostic limits", ""]
    for row in report["aggregates"]:
        lines.append(f"- {row['dataset']} / {LABELS.get(row['method'],row['method'])}: {row['failed_cv_candidates']} failed CV candidates; "
                     f"{row['preflight_excluded_candidates']} dimensional exclusions; {row['case_count']-row['qualified_cases']} unqualified selected cases.")
    for row in report["cases"]:
        if row["issues"]:
            lines.append(f"- {row['jobid']}: {'; '.join(row['issues'])}.")
    boundary_rows = sum(any("boundary" in item.get("status", "") or "single_value" in item.get("status", "")
                            for item in row["refinement"].get("parameters", {}).values()) for row in report["cases"])
    lines += ["", f"{boundary_rows} selected jobs have declared-grid boundary/single-value flags. Full selected parameters, CV scores, "
              "boundary diagnostics, numerical/KKT diagnostics where recorded, warning statuses and stage timings are retained in JSON. "
              "An interior grid point is not proof of completed tuning. KKT diagnostics assess numerical stationarity, not global "
              "optimality, factor recovery or classifier consistency.", "",
              "Fashion-MNIST uses official-training development subsets; its official test data are not used. CelebA uses official "
              "training/validation only. Its matched eight-job block supplies every solver's primary contribution; old CD/SPA "
              "18-case studies remain historical context. Overlapping samples and reused validation sets are not independent "
              "replications. Person-level CelebA separation is unverified. No new final-test result enters this decision.", "",
              "`report.json` binds every read input with SHA-256 hashes. Any missing evidence blocks the primary decision. "
              "The report is a reviewable candidate selection and does not change the public package default.", ""]
    return "\n".join(lines)


def frozen_protocol(inputs, run, dataset, configuration, fallback):
    """Read the exact version snapshotted by this run, including recovery runs."""
    freeze_path = run / "source-freeze.json"
    if freeze_path.exists():
        freeze = inputs.read(freeze_path, receipt=False)
        relative = freeze.get("protocol_paths", {}).get(dataset, fallback)
        path = (run / "source" / relative).resolve()
        require(path.is_relative_to((run / "source").resolve()), "Frozen protocol path escapes source snapshot.")
        protocol = inputs.read(path, receipt=False)
        require(inputs.hashes[str(path)] == freeze["files"][relative], "Frozen protocol snapshot checksum differs.")
    else:
        path = (ROOT / fallback).resolve()
        protocol = inputs.read(path, receipt=False)
    require(protocol == configuration["protocol"] and inputs.hashes[str(path)] == configuration["protocol_file_sha256"],
            "Frozen protocol and configured decision rule differ.")
    return protocol


def recovery_provenance(inputs, run, configuration):
    path = run / "recovery-manifest.json"
    if not path.exists():
        return None
    recovery = inputs.read(path)
    require(recovery["state"] == "prepared_no_models_fitted" and recovery["bpp_receipts_reused"] is False
            and recovery["official_test_loaded"] is False and recovery["validation_outcomes_read"] is False
            and recovery["child_configuration_sha256"] == digest(configuration)
            and recovery["child_source_freeze_sha256"] == inputs.hashes[str((run / "source-freeze.json").resolve())],
            "Recovery manifest does not bind this configuration/freeze or reused BPP evidence.")
    reusable = {*METHODS[:3], "screened_pca_absolute", "screening_only_absolute", "raw_knn"}
    for record in recovery["copied_files"]:
        destination = Path(record["destination"]).resolve()
        require(destination.is_relative_to((run / "fashion").resolve()), "Inherited receipt escapes this run.")
        jobid = destination.relative_to(run / "fashion").parts[0]
        require(any(plan["job"]["id"] == jobid and plan["job"]["method"] in reusable for plan in configuration["plans"]),
                "Inherited receipt is not an unchanged-method checkpoint.")
        value = hashlib.sha256(destination.read_bytes()).hexdigest()
        require(value == record["sha256"], "Inherited receipt bytes changed after recovery.")
        inputs.hashes[str(destination)] = value
    return recovery


def write_report(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / "report"
    inputs = Inputs()
    rows, configurations, executions = [], {}, {}
    for directory, dataset in (("fashion", "Fashion-MNIST"), ("celeba", "CelebA")):
        local, configurations[dataset], executions[dataset] = load_dataset(run / directory, dataset, inputs)
        rows.extend(local)
    fashion_protocol = configurations["Fashion-MNIST"]["protocol"]
    rule = fashion_protocol["cross_dataset_decision"]
    require(rule["candidates"] == list(METHODS), "Unexpected decision candidates.")
    frozen_protocol(inputs, run, "fashion", configurations["Fashion-MNIST"], "examples/fashion_mnist_solver_protocol_v2.json")
    frozen_protocol(inputs, run, "celeba", configurations["CelebA"], rule["primary_celeba_protocol"])
    recovery = recovery_provenance(inputs, run, configurations["Fashion-MNIST"])
    require(Counter(row["method"] for row in rows if row["dataset"] == "Fashion-MNIST" and row["method"] in METHODS)
            == Counter({method: 18 for method in METHODS})
            and Counter(row["method"] for row in rows if row["dataset"] == "CelebA") == Counter({method: 2 for method in METHODS}),
            "Primary decision requires 18 Fashion and two matched CelebA cases per solver.")
    report = describe(rows, rule)
    report["matched_fashion_baselines"] = matched_fashion_baselines(rows, fashion_protocol)
    if inputs.missing:
        report["primary"].update(status="blocked_incomplete_evidence", winner=None)
    report.update(schema_version=1, author="Chang Cheng", created_utc=datetime.now(timezone.utc).isoformat(),
                  source_run=str(run), winner_method=report["primary"]["winner"], decision_rule=rule,
                  decision_protocol_sha256=configurations["Fashion-MNIST"]["protocol_file_sha256"],
                  input_sha256=inputs.hashes, missing_evidence=inputs.missing, executions=executions,
                  recovery=recovery,
                  reporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  package_defaults_modified=False, final_test_evaluated=False,
                  remaining_audits=["Independent evidence/decision audit", "Public API integration if MU or ANLS-BPP is selected",
                                    "Frozen final-test evaluation only after source/default decisions", "Broader data-domain and identity-generalization validation"])
    inputs.verify()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = write_report(args.run, args.output)
    print(json.dumps({"status": report["primary"]["status"], "winner_method": report["winner_method"],
                      "report": str((args.output or args.run / "report") / "report.md")}, indent=2))
    return 0 if report["winner_method"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
