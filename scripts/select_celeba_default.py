"""Apply the frozen CelebA practical-default rule to complete saved results.

Author: Chang Cheng. This script does not fit estimators, propose parameters,
change package defaults, or read images. It verifies receipts and saved predictions,
then records one decision and its descriptive sensitivity analyses.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid


LABELS = {
    "screened_cd_percentile": "CD + percentile",
    "screened_cd_absolute": "CD + absolute |t|",
    "screened_spa_percentile": "SPA + percentile",
    "screened_spa_absolute": "SPA + absolute |t|",
}
ROOT = Path(__file__).resolve().parents[1]
METRICS = ("balanced_accuracy", "accuracy", "precision", "recall", "f1", "average_precision", "roc_auc")
PREDICTION_FIELDS = ("sample_ids", "true_labels", "predicted_labels", "positive_class_probability")


def digest(value):
    """Match the runner's receipt representation, including default separators."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, positive=False, unit_interval=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"{label} must be a finite number.")
    require(value > 0 if positive else value >= 0, f"{label} has an invalid sign.")
    require(not unit_interval or value <= 1, f"{label} must be at most one.")
    return float(value)


def same_number(actual, expected, label, tolerance=1e-12):
    finite(actual, label)
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), f"{label} differs from its evidence.")


def has_nonconvergence(value):
    if isinstance(value, dict):
        if value.get("category") == "ConvergenceWarning":
            return True
        return any((key.rstrip("_").endswith("converged") and item is False) or has_nonconvergence(item)
                   for key, item in value.items())
    return isinstance(value, list) and any(has_nonconvergence(item) for item in value)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, *, receipt=False):
        path = Path(path).resolve()
        raw = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()

        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, f"Duplicate JSON key {key!r}: {path}")
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError(f"Nonfinite JSON constant {value}: {path}")

        value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
        if receipt:
            require(isinstance(value, dict) and "payload" in value and value.get("sha256") == digest(value["payload"]),
                    f"Receipt checksum failed: {path}")
            value = value["payload"]
        return value

    def verify_unchanged(self):
        for name, expected in self.hashes.items():
            require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected,
                    f"Input changed during reporting: {name}")


def check_protocol(protocol):
    require(protocol.get("schema_version") == 1 and protocol.get("author") == "Chang Cheng", "Unsupported decision protocol.")
    require(protocol.get("methods_in_final_tie_order") == list(LABELS), "Unexpected method set/tie order.")
    require(protocol.get("attributes") == ["Smiling", "Wearing_Necktie"], "Unexpected attribute set.")
    require(protocol.get("training_sizes") == [2000, 4000, 8000] and protocol.get("training_seeds") == [43, 44, 45],
            "This selector requires the frozen nine configurations.")
    require(protocol["sampling"]["validation_sample_count"] == 4000, "The frozen workload requires 4,000 validation images.")
    require(protocol["stage_2"]["method_attribute_configuration_jobs"] == 72, "The frozen study requires 72 jobs.")
    require(protocol["timing"]["additional_repeats_per_selected_model"] == 3, "Three profiling repeats are required.")
    for field in ("macro_regret_tolerance", "attribute_regret_tolerance", "numerical_comparison_slack"):
        finite(protocol["decision_rule"][field], field, positive=True)
    finite(protocol["timing"]["primary_cost_closeness_ratio"], "cost band", positive=True)


def checked_metrics(evaluation, validation, attribute, expected_truth=None):
    saved = evaluation["predictions"]
    count = len(validation["image_ids"])
    require(saved["sample_ids"] == validation["image_ids"], "Prediction image IDs differ from the fixed validation allocation.")
    arrays = {name: np.asarray(saved[name]) for name in PREDICTION_FIELDS[1:]}
    y, predicted, probability = (arrays[name] for name in PREDICTION_FIELDS[1:])
    require(all(array.shape == (count,) for array in arrays.values()), "Invalid prediction array length.")
    require(np.isin(y, [0, 1]).all() and np.isin(predicted, [0, 1]).all(), "Invalid binary predictions or labels.")
    require(probability.dtype.kind in "iuf" and np.isfinite(probability).all()
            and np.all((probability >= 0) & (probability <= 1)), "Invalid saved probabilities.")
    require(np.bincount(y.astype(int), minlength=2).tolist() == validation["attributes"][attribute]["class_counts"],
            "Saved truth counts differ from the validation manifest.")
    require(len(np.unique(y)) == 2, "Both validation classes are required.")
    if expected_truth is not None:
        require(saved["true_labels"] == expected_truth, "Validation truth changed across methods or configurations.")
    measured = {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "prevalence": float(np.mean(y)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
    for metric in METRICS:
        same_number(evaluation["metrics"][metric], measured[metric], metric)
    require(evaluation["metrics"].get("positive_class") == 1
            and evaluation["metrics"].get("confusion_matrix_labels") == [0, 1]
            and evaluation["metrics"]["confusion_matrix"] == measured["confusion_matrix"], "Confusion matrix/label convention differs.")
    return measured


def expanded_candidates(grid):
    result, seen = [], set()
    for point in ParameterGrid(grid):
        key = digest(point)
        if key not in seen:
            result.append(point)
            seen.add(key)
    return result


def check_cv_summary(summary, plan, configuration, inputs, directory):
    configuration_digest = digest(configuration)
    require(summary.get("state") == "cv_completed" and summary.get("job") == plan["job"]
            and summary.get("configuration_sha256") == configuration_digest
            and summary.get("official_test_loaded") is False, "CV summary identity differs.")
    candidates = summary["candidates"]
    require([row["index"] for row in candidates] == list(range(len(plan["candidates"])))
            and [row["parameters"] for row in candidates] == plan["candidates"], "CV candidate coverage differs.")
    chunk_rows, costs = {}, Counter()
    for chunk in plan["chunks"]:
        report = inputs.read(directory / chunk["id"] / "report.json", receipt=True)
        require(summary["report_receipts"][chunk["id"]] == digest(report)
                and report["state"] == "cv_completed" and report["fingerprint"] == digest(report["identity"]),
                "CV chunk identity/checksum differs.")
        require(report["identity"] == inputs.read(directory / chunk["id"] / "identity.json", receipt=True),
                "CV chunk checkpoint identity differs.")
        expected = list(ParameterGrid(chunk["grid"]))
        calibration = report["identity"]["calibration"]
        job = plan["job"]
        train = configuration["caches"][job["cache"]]["manifest"]["splits"]["train"]
        require(calibration["candidates"] == expected and calibration["cv"] == 5
                and calibration["seed"] == job["seed"] and calibration["threads"] == 1
                and calibration["refit"] is False
                and calibration["data_sha256"] == train["attributes"][job["attribute"]]["data_sha256"]
                and calibration["source_sha256"] == configuration["sources"]["package_source_sha256"],
                "CV chunk calibration is not bound to the declared data, folds or source.")
        parameters = calibration["parameters"]
        require(parameters["max_iter"] == job["max_iter"] and parameters["tol"] == job["tol"]
                and parameters["random_state"] == job["seed"]
                and parameters["solver"] == ("cd" if "_cd_" in job["method"] else "spa")
                and parameters["screening"] == ("percentile" if job["method"].endswith("percentile") else "absolute"),
                "CV estimator numerical/method settings differ from the job.")
        require([row["parameters"] for row in report["candidates"]] == expected, "Chunk does not cover its grid.")
        require([plan["candidates"][index] for index in chunk["candidate_indices"]] == expected,
                "Chunk canonical candidate indices differ.")
        for index, row in zip(chunk["candidate_indices"], report["candidates"]):
            require(index not in chunk_rows, "Candidate occurs in multiple chunks.")
            chunk_rows[index] = {**row, "index": index, "chunk": chunk["id"]}
        for name, value in report["compute_seconds"].items():
            costs[name] += finite(value, f"CV compute/{name}")
    require([chunk_rows[index] for index in sorted(chunk_rows)] == candidates, "CV summary differs from committed chunk rows.")
    require(set(costs) == set(summary["compute_seconds"]), "CV cost fields differ.")
    for name, value in costs.items():
        same_number(summary["compute_seconds"][name], value, f"CV compute/{name}", tolerance=1e-8)
    same_number(summary["tuning_compute_seconds"], sum(costs.values()), "CV tuning compute", tolerance=1e-8)
    failures = []
    for row in candidates:
        require([fold["fold"] for fold in row["folds"]] == list(range(5)), "A candidate is missing a CV fold.")
        valid = all(fold["status"] == "valid" for fold in row["folds"])
        require(row["valid"] is valid, "Candidate qualification differs from its five folds.")
        if valid:
            require(not any(has_nonconvergence(fold) for fold in row["folds"]), "A supposedly valid CV fold records nonconvergence.")
            scores = [finite(fold["balanced_accuracy"], "fold score", unit_interval=True) for fold in row["folds"]]
            same_number(row["mean_balanced_accuracy"], float(np.mean(scores)), "candidate mean")
        else:
            require(row["mean_balanced_accuracy"] is None, "Failed candidate must not average successful folds.")
            failures.append({"parameters": row["parameters"], "index": row["index"], "chunk": row["chunk"],
                             "folds": [fold for fold in row["folds"] if fold["status"] != "valid"]})
    eligible = [row for row in candidates if row["valid"]]
    selected = min(eligible, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])) if eligible else None
    require(summary["selected_candidate"] == selected, "Selected candidate is not the canonical CV winner.")
    require(summary["selected_parameters"] == (selected["parameters"] if selected else None)
            and summary["cv_balanced_accuracy"] == (selected["mean_balanced_accuracy"] if selected else None),
            "Selected parameters or CV score differ.")
    return failures


def check_preflight(declared, inputs):
    """Recover the complete original grid and verify every dimensional exclusion."""
    if "preflight" not in declared:
        return {job["id"]: job for job in declared["jobs"]}, {}
    provenance = declared["preflight"]
    original_path, audit_path = Path(provenance["input_protocol_path"]), Path(provenance["audit_path"])
    original = inputs.read(original_path)
    audit = inputs.read(audit_path)
    require(inputs.hashes[str(original_path.resolve())] == provenance["input_protocol_sha256"]
            and inputs.hashes[str(audit_path.resolve())] == provenance["audit_sha256"]
            and audit["input_protocol_sha256"] == provenance["input_protocol_sha256"]
            and audit["training_only"] is True and audit["decomposition_or_classifier_fits"] == 0
            and audit["official_test_loaded"] is False, "Preflight proof identity or training-only role differs.")
    require({key: value for key, value in original.items() if key != "jobs"}
            == {key: value for key, value in declared.items() if key not in {"jobs", "preflight"}},
            "Preflight changed fields outside the declared grids.")
    original_jobs = {job["id"]: job for job in original["jobs"]}
    audits = {job["id"]: job for job in audit["jobs"]}
    require(len(original_jobs) == len(audits) == len(original["jobs"]) == len(audit["jobs"]) == 72,
            "Preflight original/proof jobs are missing or duplicated.")
    exclusions = {}
    for job in declared["jobs"]:
        old, evidence = original_jobs[job["id"]], audits[job["id"]]
        changed_keys = {"grid", "original_declared_candidates", "declared_candidates", "excluded_infeasible_candidates", "nominal_representation_folds"}
        require({key: value for key, value in old.items() if key not in changed_keys}
                == {key: value for key, value in job.items() if key not in changed_keys}, "Preflight changed a job's non-grid configuration.")
        original_points, kept = expanded_candidates(old["grid"]), expanded_candidates(job["grid"])
        expected_kept, expected_excluded = [], []
        for point in original_points:
            setting = ({"screening": "percentile", "percentile": point["percentile"]} if "percentile" in point
                       else {"screening": "absolute", "score_threshold": point["score_threshold"]})
            mask_key = json.dumps([old["cache"], old["attribute"], old["seed"], setting], sort_keys=True, separators=(",", ":"))
            folds = audit["fold_masks"][mask_key]
            require([fold["fold"] for fold in folds] == list(range(5)), "Preflight lacks five fold masks.")
            feasible = all(fold["selected_features"] > 0
                           and point["n_components"] <= min(fold["selected_features"], fold["n_train"])
                           and point["n_neighbors"] <= fold["n_train"] for fold in folds)
            if feasible:
                expected_kept.append(point)
            else:
                expected_excluded.append((point, mask_key))
        require(kept == expected_kept and [(item["parameters"], item["mask_key"]) for item in evidence["excluded"]] == expected_excluded
                and all(item.get("reason") for item in evidence["excluded"]), "Preflight omissions do not match their dimensional proof.")
        require(job["original_declared_candidates"] == len(original_points)
                and job["declared_candidates"] == evidence["kept_candidates"] == len(kept)
                and job["excluded_infeasible_candidates"] == len(expected_excluded), "Preflight candidate counts differ.")
        exclusions[job["id"]] = evidence["excluded"]
    return original_jobs, exclusions


def check_transfer_and_timing(declared, original_jobs, protocol):
    """Independently reproduce the frozen shortlist transfer and timing order."""
    choices = declared["training_only_shortlists"]
    methods, attributes = protocol["methods_in_final_tie_order"], protocol["attributes"]
    require(set(choices) == set(attributes) and all(set(choices[attribute]) == set(methods) for attribute in attributes),
            "Training-only shortlists do not cover the exact methods and attributes.")
    for attribute in attributes:
        for method in methods:
            pairs = choices[attribute][method]["pairs"]
            screen = "percentile" if method.endswith("percentile") else "score_threshold"
            require(len(pairs) == 3 and all(set(pair["parameters"]) == {"n_components", screen} for pair in pairs)
                    and len({digest(pair["parameters"]) for pair in pairs}) == 3,
                    "A shortlist does not contain three distinct rank/screening pairs.")
    expected_blocks, expected_order = [], []
    rng = random.Random(protocol["timing"]["order_seed"])
    for n, seed, attribute in itertools.product(protocol["training_sizes"], protocol["training_seeds"], attributes):
        order = list(methods)
        rng.shuffle(order)
        expected_blocks.append({"n_train": n, "seed": seed, "attribute": attribute, "method_order": order})
        expected_order.extend((n, seed, attribute, method) for method in order)
    timing = declared["timing_order"]
    require(timing.get("seed") == protocol["timing"]["order_seed"] and timing.get("blocks") == expected_blocks
            and timing.get("three_profiles_are_consecutive_within_each_method") is True,
            "Timing block order does not reproduce the frozen randomization seed.")
    actual_order = [(job["n_train"], job["seed"], job["attribute"], job["method"]) for job in declared["jobs"]]
    require(actual_order == expected_order, "Actual job order differs from the frozen randomized timing blocks.")
    for job in declared["jobs"]:
        original = original_jobs[job["id"]]
        require((original["n_train"], original["seed"], original["attribute"], original["method"])
                == (job["n_train"], job["seed"], job["attribute"], job["method"]), "Preflight changed a job's transfer configuration.")
        expected_grid = []
        for pair in choices[job["attribute"]][job["method"]]["pairs"]:
            parameters = dict(pair["parameters"])
            if "score_threshold" in parameters:
                parameters["score_threshold"] *= math.sqrt(job["n_train"] / 4000)
            expected_grid.append({**{name: [value] for name, value in parameters.items()},
                                  "n_neighbors": protocol["stage_2"]["n_neighbors"]})
        require(expanded_candidates(original["grid"]) == expanded_candidates(expected_grid),
                "Original grid does not exactly transfer its training-only shortlist and size-scaled absolute cutoff.")


def profile_qualification(result, repeats):
    issues, pairs = [], []
    profiles = result.get("profiling", [])
    if len(profiles) != repeats or [item.get("repeat") for item in profiles] != list(range(repeats)):
        issues.append("Missing, duplicate or unordered profiling repeats.")
    reference = result.get("evaluation", {}).get("predictions")
    for number, profile in enumerate(profiles):
        if profile.get("status") != "valid" or profile.get("exact_prediction_match") is not True:
            issues.append(f"Profile {number}: {profile.get('status')}; {profile.get('reason', profile.get('error', 'exact match not qualified'))}")
            continue
        if has_nonconvergence(profile):
            issues.append(f"Profile {number}: numerical diagnostics or warnings record nonconvergence.")
            continue
        if profile.get("fit_warnings") or (profile.get("evaluation") or {}).get("warnings"):
            issues.append(f"Profile {number}: fit/evaluation warnings invalidate timing under the frozen protocol.")
            continue
        evaluation = profile.get("evaluation") or {}
        if evaluation.get("status") != "valid" or not reference or any(
                evaluation.get("predictions", {}).get(field) != reference[field] for field in PREDICTION_FIELDS):
            issues.append(f"Profile {number}: independently checked predictions/probabilities do not exactly match.")
            continue
        timing = profile["timing"]
        try:
            fit = finite(timing.get("total_fit_seconds"), "profile fit", positive=True)
            prediction = finite(timing.get("prediction_seconds"), "profile prediction", positive=True)
            same_number(evaluation["prediction_seconds"], prediction, "profile prediction timer")
            for name, value in timing.items():
                if value is not None:
                    finite(value, name)
        except (KeyError, ValueError) as error:
            issues.append(f"Profile {number}: {error}")
            continue
        pairs.append({"fit": fit, "prediction": prediction, "stages": timing})
    if result.get("profiling_valid") is not True:
        issues.append("Runner did not qualify all profiling repeats.")
    if not issues:
        for name, value in result.get("profile_medians_seconds", {}).items():
            if name == "fit_plus_prediction_seconds":
                expected = statistics.median(pair["fit"] + pair["prediction"] for pair in pairs)
            else:
                values = [pair["stages"].get(name) for pair in pairs]
                expected = statistics.median(values) if all(item is not None for item in values) else None
            if expected is None:
                require(value is None, "Unavailable profile stage has an invented median.")
            else:
                same_number(value, expected, f"profile median/{name}")
    return issues, pairs


def load_cases(run, protocol_path):
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    inputs = Inputs()
    protocol = inputs.read(protocol_path)
    check_protocol(protocol)
    protocol_sha = inputs.hashes[str(protocol_path)]
    configuration = inputs.read(run / "configuration.json", receipt=True)
    identity = digest(configuration)
    declared = configuration["protocol"]
    require(declared.get("decision_protocol_sha256") == protocol_sha, "Run is not bound to this frozen decision protocol.")
    require(declared.get("cv") == 5 and declared.get("native_threads") == 1 and declared.get("profile_repeats") == 3
            and configuration.get("official_test_loaded") is False, "Run numerical/evaluation protocol differs.")
    plans = configuration["plans"]
    require(len(plans) == len(declared["jobs"]) == 72, "Exactly 72 declared jobs are required before selection.")
    require([plan["job"]["id"] for plan in plans] == [job["id"] for job in declared["jobs"]],
            "Actual evaluation/timing plan order differs from the frozen job order.")
    declared_by_id = {job["id"]: job for job in declared["jobs"]}
    require(len(declared_by_id) == 72, "Duplicate protocol job IDs.")
    original_jobs, preflight_exclusions = check_preflight(declared, inputs)
    check_transfer_and_timing(declared, original_jobs, protocol)
    expected_cases = set(itertools.product(protocol["training_sizes"], protocol["training_seeds"],
                                          protocol["attributes"], protocol["methods_in_final_tie_order"]))
    seen_cases, seen_ids, cache_for_config, validation_reference = set(), set(), {}, None
    cv_report = inputs.read(run / "cv-summary.json", receipt=True)
    require(cv_report.get("state") == "cv_completed" and not cv_report.get("missing_chunks")
            and cv_report.get("configuration_sha256") == identity
            and set(cv_report["jobs"]) == set(declared_by_id), "Full declared CV is not complete.")
    rows, truths = [], {}
    for plan in plans:
        job = plan["job"]
        jobid = job["id"]
        require(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", jobid) and jobid not in seen_ids, "Invalid or duplicate plan job ID.")
        seen_ids.add(jobid)
        declared_job = dict(declared_by_id[jobid])
        cache_path = Path(declared_job["cache"])
        declared_job["cache"] = str((ROOT / cache_path).resolve() if not cache_path.is_absolute() else cache_path.resolve())
        require(declared_job == job, "Plan job differs from the declared protocol.")
        require(job["max_iter"] == protocol["numerical_configuration"]["max_iter"]
                and job["tol"] == protocol["numerical_configuration"]["tol"], "Job numerical cap/tolerance differs.")
        require(plan["candidates"] == expanded_candidates(job["grid"]), "Plan's candidates differ from its explicit grid.")
        representations = {}
        for point in expanded_candidates(original_jobs[jobid]["grid"]):
            pair = {name: value for name, value in point.items() if name != "n_neighbors"}
            representations.setdefault(digest(pair), set()).add(point["n_neighbors"])
        require(len(representations) == 3 and all(sorted(values) == protocol["stage_2"]["n_neighbors"]
                for values in representations.values()), "Original preflight grid does not have the frozen three-pair/equal-k budget.")
        cache = configuration["caches"][job["cache"]]
        manifest = cache["manifest"]
        require(cache["manifest_sha256"] == digest(manifest) and manifest.get("official_test_loaded") is False,
                "Cache manifest identity or test-exclusion differs.")
        train, validation = manifest["splits"]["train"], manifest["splits"]["validation"]
        n = len(train["image_ids"])
        require(job.get("n_train", n) == n, "Declared training size differs from its cache.")
        case = (n, job["seed"], job["attribute"], job["method"])
        require(case in expected_cases and case not in seen_cases, "Missing/duplicate/unexpected method-configuration case.")
        seen_cases.add(case)
        require(manifest["seed"] == job["seed"] and train["metadata"]["partition_code"] == 0
                and validation["metadata"]["partition_code"] == 1
                and validation["metadata"]["random_state"] == protocol["sampling"]["validation_seed"],
                "Sampling seed or official partition differs.")
        require(len(set(train["image_ids"])) == n and len(validation["image_ids"]) == 4000
                and len(set(validation["image_ids"])) == 4000
                and not set(train["image_ids"]).intersection(validation["image_ids"]), "Invalid data IDs or split overlap.")
        if validation_reference is None:
            validation_reference = validation
        require(validation == validation_reference, "The fixed validation allocation differs across jobs.")
        sample_key = (n, job["seed"])
        require(sample_key not in cache_for_config or cache_for_config[sample_key] == job["cache"],
                "Methods/attributes do not share the same training cache.")
        cache_for_config[sample_key] = job["cache"]
        directory = run / jobid
        summary = inputs.read(directory / "calibration-summary.json", receipt=True)
        require(cv_report["jobs"][jobid] == summary, "Global CV summary differs from its job receipt.")
        failures = check_cv_summary(summary, plan, configuration, inputs, directory)
        result = inputs.read(directory / "evaluation.json", receipt=True)
        expected_identity = {"configuration_sha256": identity, "job_id": jobid,
                             "selected_candidate_sha256": digest(summary["selected_candidate"]),
                             "validation_data_sha256": validation["attributes"][job["attribute"]]["data_sha256"]}
        require(result.get("identity") == expected_identity and result.get("job") == job
                and result.get("jobid") == jobid and result.get("attribute") == job["attribute"]
                and result.get("method") == job["method"] and result.get("seed") == job["seed"]
                and result.get("n_train") == n and result.get("official_test_loaded") is False,
                "Evaluation is not bound to the declared job/candidate/data.")
        for name in ("selected_parameters", "cv_balanced_accuracy", "tuning_compute_seconds"):
            require(result[name] == summary[name], f"Evaluation {name} differs from training selection.")
        tuning = finite(result["tuning_compute_seconds"], "tuning compute", positive=True)
        evaluation = result["evaluation"]
        issues, metrics = [], None
        if evaluation["status"] != "valid":
            issues.append(f"Evaluation {evaluation['status']}: {evaluation.get('reason', evaluation.get('error', ''))}")
        elif result.get("is_fallback") is not False or summary["selected_candidate"] is None:
            issues.append("Fallback or unqualified training selection.")
        else:
            require(evaluation["data_sha256"] == expected_identity["validation_data_sha256"], "Evaluation data fingerprint differs.")
            metrics = checked_metrics(evaluation, validation, job["attribute"], truths.get(job["attribute"]))
            truths[job["attribute"]] = evaluation["predictions"]["true_labels"]
        if has_nonconvergence({key: result.get(key) for key in ("diagnostics", "final_fit_warnings", "evaluation")}):
            issues.append("Selected fit or evaluation diagnostics record nonconvergence.")
        profile_issues, pairs = profile_qualification(result, 3)
        issues.extend(profile_issues)
        rows.append({"jobid": jobid, "n_train": n, "seed": job["seed"], "attribute": job["attribute"],
                     "method": job["method"], "qualified": not issues, "issues": issues,
                     "evaluation_status": evaluation["status"], "metrics": metrics,
                     "selected_parameters": result["selected_parameters"], "cv_balanced_accuracy": result["cv_balanced_accuracy"],
                     "tuning_compute_seconds": tuning, "original_final_fit_seconds": result["final_fit_seconds"],
                     "original_prediction_seconds": evaluation.get("prediction_seconds"), "timing_pairs": pairs,
                     "profiling_statuses": [{"repeat": item.get("repeat"), "status": item.get("status"),
                                             "exact_prediction_match": item.get("exact_prediction_match"),
                                             "reason": item.get("reason", item.get("error")),
                                             "fit_warnings": item.get("fit_warnings", [])}
                                            for item in result["profiling"]],
                     "final_fit_warnings": result.get("final_fit_warnings", []),
                     "evaluation_warnings": evaluation.get("warnings", []),
                     "cv_failed_candidates": failures,
                     "preflight_excluded_candidates": preflight_exclusions.get(jobid, []),
                     "source_receipt": str(directory / "evaluation.json")})
    require(seen_cases == expected_cases, "All 72 distinct expected cases must be present before selection.")
    for seed in protocol["training_seeds"]:
        allocated = [set(configuration["caches"][cache_for_config[n, seed]]["manifest"]["splits"]["train"]["image_ids"])
                     for n in protocol["training_sizes"]]
        require(all(small <= large for small, large in zip(allocated, allocated[1:])), "Training sizes are not nested within seed.")
    comparison = inputs.read(run / "comparison.json", receipt=True)
    require(comparison.get("state") == "development_evaluation_completed"
            and comparison.get("configuration_sha256") == identity
            and comparison.get("official_test_loaded") is False
            and set(comparison["jobs"]) == seen_ids, "Completed comparison coverage/identity differs.")
    for row in rows:
        require(comparison["jobs"][row["jobid"]] == inputs.read(row["source_receipt"], receipt=True),
                "Global comparison differs from its evaluation receipts.")
    inputs.verify_unchanged()
    return rows, protocol, inputs, configuration


def geometric_mean(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(value) and value > 0 for value in values), "Geometric mean requires positive finite costs.")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def case_cost(row, multiplier=1, include_tuning=False):
    cost = statistics.median(pair["fit"] + multiplier * pair["prediction"] for pair in row["timing_pairs"])
    return cost + (row["tuning_compute_seconds"] if include_tuning else 0)


def choose(rows, protocol, *, macro_margin=None, prediction_multiplier=1, include_tuning=False, allowed_methods=None):
    """Pure implementation of the frozen rule; no outcome-dependent adjustment."""
    methods, attributes = protocol["methods_in_final_tie_order"], protocol["attributes"]
    slack = protocol["decision_rule"]["numerical_comparison_slack"]
    macro_margin = protocol["decision_rule"]["macro_regret_tolerance"] if macro_margin is None else macro_margin
    attribute_margin = (protocol["decision_rule"]["attribute_regret_tolerance"] if macro_margin == protocol["decision_rule"]["macro_regret_tolerance"]
                        else macro_margin * protocol["sensitivity"]["attribute_margin_multiplier"])
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    case_sets = {method: {(row["n_train"], row["seed"], row["attribute"]) for row in subset} for method, subset in by_method.items()}
    require(all(len(case_sets[method]) == len(by_method[method]) for method in methods), "Duplicate decision cases.")
    require(all(case_sets[method] == case_sets[methods[0]] for method in methods) and bool(case_sets[methods[0]]),
            "Methods do not share complete decision cases.")
    eligible = [method for method in methods if all(row["qualified"] for row in by_method[method])
                and (allowed_methods is None or method in allowed_methods)]
    result = {"winner": None, "qualified_methods": eligible, "macro_margin": macro_margin,
              "attribute_margin": attribute_margin, "prediction_multiplier": prediction_multiplier,
              "include_parallel_tuning_in_workload": include_tuning, "scores": {}, "fallback_used": False}
    if not eligible:
        return {**result, "status": "blocked_no_operationally_qualified_method",
                "reason": "No method qualified in every required case; repair and record numerical/execution problems before choosing a default."}
    for method in eligible:
        attribute_means = {attribute: statistics.fmean(row["metrics"]["balanced_accuracy"] for row in by_method[method]
                                                     if row["attribute"] == attribute) for attribute in attributes}
        result["scores"][method] = {"attributes": attribute_means, "macro": statistics.fmean(attribute_means.values())}
    scores = result["scores"]
    macro_best = max(value["macro"] for value in scores.values())
    attribute_best = {attribute: max(value["attributes"][attribute] for value in scores.values()) for attribute in attributes}
    acceptable = []
    for method, value in scores.items():
        value["macro_regret"] = macro_best - value["macro"]
        value["attribute_regrets"] = {attribute: attribute_best[attribute] - value["attributes"][attribute] for attribute in attributes}
        regrets = [value["macro_regret"], *value["attribute_regrets"].values()]
        value["maximum_regret"] = max(regrets)
        value["normalized_worst_regret"] = (max(value["macro_regret"] / macro_margin,
                                                   *(regret / attribute_margin for regret in value["attribute_regrets"].values()))
                                               if macro_margin > 0 else max(regrets))
        value["accuracy_acceptable"] = value["macro_regret"] <= macro_margin + slack and all(
            regret <= attribute_margin + slack for regret in value["attribute_regrets"].values())
        if value["accuracy_acceptable"]:
            acceptable.append(method)
    result["accuracy_acceptable_methods"] = acceptable[:]
    if not acceptable:
        result["fallback_used"] = True
        regret = min(value["normalized_worst_regret"] for value in scores.values())
        acceptable = [method for method in eligible if scores[method]["normalized_worst_regret"] <= regret + slack]
    result["methods_before_cost"] = acceptable
    costs, tuning = {}, {}
    for method in eligible:
        for row in by_method[method]:
            case = (row["n_train"], row["seed"], row["attribute"])
            costs[method, case] = case_cost(row, prediction_multiplier, include_tuning)
            tuning[method, case] = row["tuning_compute_seconds"]
    for method in eligible:
        cases = sorted(case_sets[method])
        scores[method]["normalized_serial_cost"] = geometric_mean(
            costs[method, case] / min(costs[other, case] for other in eligible) for case in cases)
        scores[method]["normalized_tuning_compute"] = geometric_mean(
            tuning[method, case] / min(tuning[other, case] for other in eligible) for case in cases)
        scores[method]["geometric_workload_seconds"] = geometric_mean(costs[method, case] for case in cases)
    cheapest = min(scores[method]["normalized_serial_cost"] for method in acceptable)
    close = [method for method in acceptable if scores[method]["normalized_serial_cost"] <=
             cheapest * protocol["timing"]["primary_cost_closeness_ratio"] + slack]
    result["methods_within_cost_band"] = close[:]
    best_tuning = min(scores[method]["normalized_tuning_compute"] for method in close)
    close = [method for method in close if scores[method]["normalized_tuning_compute"] <= best_tuning + slack]
    result["methods_after_tuning_tie_break"] = close[:]
    best_macro = max(scores[method]["macro"] for method in close)
    close = [method for method in close if scores[method]["macro"] >= best_macro - slack]
    winner = next(method for method in methods if method in close)
    return {**result, "winner": winner, "status": "selected", "winning_solver": "cd" if "_cd_" in winner else "spa",
            "winning_screening": "percentile" if winner.endswith("percentile") else "absolute"}


def summarize(rows, protocol):
    primary = choose(rows, protocol)
    aggregates = []
    for method in protocol["methods_in_final_tie_order"]:
        subset = [row for row in rows if row["method"] == method]
        measured = [row for row in subset if row["metrics"] is not None]
        valid = [row for row in subset if row["qualified"]]
        attributes = {}
        for attribute in protocol["attributes"]:
            selected = [row for row in measured if row["attribute"] == attribute]
            attributes[attribute] = {"evaluated_cases": len(selected), **{
                metric: statistics.fmean(row["metrics"][metric] for row in selected) if selected else None
                for metric in (*METRICS, "prevalence")}}
        stages = sorted({name for row in valid for pair in row["timing_pairs"] for name in pair["stages"]})
        stage_means = {}
        for name in stages:
            medians = [statistics.median(pair["stages"][name] for pair in row["timing_pairs"])
                       for row in valid if all(pair["stages"].get(name) is not None for pair in row["timing_pairs"])]
            stage_means[name] = {"cases": len(medians), "mean_case_median": statistics.fmean(medians) if medians else None}
        aggregates.append({"method": method, "label": LABELS[method], "qualified_cases": len(valid),
                           "expected_cases": len(subset), "evaluated_cases": len(measured),
                           "operationally_qualified": method in primary["qualified_methods"], "attributes": attributes,
                           "descriptive_macro_ba": statistics.fmean(value["balanced_accuracy"] for value in attributes.values())
                           if all(value["balanced_accuracy"] is not None for value in attributes.values()) else None,
                           "score": primary["scores"].get(method), "stage_timing_means": stage_means,
                           "mean_case_fit_plus_prediction_seconds": statistics.fmean(case_cost(row) for row in valid) if valid else None,
                           "tuning_compute_seconds_sum": sum(row["tuning_compute_seconds"] for row in subset),
                           "cv_failed_candidate_count": sum(len(row["cv_failed_candidates"]) for row in subset),
                           "preflight_excluded_candidate_count": sum(len(row.get("preflight_excluded_candidates", [])) for row in subset),
                           "exclusions": [{"jobid": row["jobid"], "issues": row["issues"]} for row in subset if not row["qualified"]]})
    sensitivities = [choose(rows, protocol, macro_margin=margin, prediction_multiplier=multiplier, include_tuning=include)
                     for include in (False, True) for margin in protocol["sensitivity"]["macro_margins"]
                     for multiplier in protocol["sensitivity"]["prediction_multipliers"]]
    omissions = []
    for field, values in (("n_train", protocol["training_sizes"]), ("seed", protocol["training_seeds"])):
        for value in values:
            decision = choose([row for row in rows if row[field] != value], protocol,
                              allowed_methods=primary["qualified_methods"])
            omissions.append({"omitted_field": field, "omitted_value": value, "decision": decision})
    paired = []
    for left, right in itertools.combinations(primary["qualified_methods"], 2):
        indexed = {(row["method"], row["n_train"], row["seed"], row["attribute"]): row for row in rows}
        differences = [{"n_train": n, "seed": seed, "attribute": attribute,
                        "balanced_accuracy_difference": indexed[left, n, seed, attribute]["metrics"]["balanced_accuracy"]
                        - indexed[right, n, seed, attribute]["metrics"]["balanced_accuracy"]}
                       for n, seed, attribute in itertools.product(protocol["training_sizes"], protocol["training_seeds"], protocol["attributes"])]
        values = [row["balanced_accuracy_difference"] for row in differences]
        paired.append({"left": left, "right": right, "mean_difference": statistics.fmean(values),
                       "minimum": min(values), "maximum": max(values), "differences": differences,
                       "left_better_cases": sum(value > 1e-12 for value in values),
                       "near_tied_cases": sum(abs(value) <= 1e-12 for value in values)})
    return {"primary": primary, "methods": aggregates, "sensitivity": sensitivities,
            "leave_one_out": omissions, "paired_descriptive_differences": paired, "cases": rows}


def number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def table_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report):
    decision = report["primary"]
    winner = decision["winner"]
    lines = ["# CelebA general-default decision", "", "Author: Chang Cheng.", ""]
    if winner:
        lines += [f"**Selected general combination: {LABELS[winner]}.**", "",
                  "This selects the solver and screening rule; rank, screening amount and KNN neighbors remain tunable.", "",
                  f"Decision path: {'minimum-worst-regret fallback' if decision['fallback_used'] else 'practical accuracy gate'}, "
                  "then serial cost, the 5% timing band, recorded tuning cost, macro balanced accuracy and fixed method order.", "",
                  f"Eligible before cost: {', '.join(LABELS[name] for name in decision['methods_before_cost'])}. "
                  f"Within the cost band: {', '.join(LABELS[name] for name in decision['methods_within_cost_band'])}.", ""]
        selected = decision["scores"][winner]
        lines += [f"Selected macro BA: {number(selected['macro'])}; macro regret: {number(selected['macro_regret'])}. "
                  f"Attribute regrets: {', '.join(f'{key} {number(value)}' for key, value in selected['attribute_regrets'].items())}.", ""]
        if decision["fallback_used"]:
            lines += ["**No combination met every practical accuracy margin.** The declared minimum-worst-regret rule made "
                      "the compromise; it does not establish accuracy equivalence.", ""]
    else:
        lines += ["**No operationally qualified default can be selected.**", "", decision["reason"], ""]
    lines += ["## Accuracy and qualification", "",
              "Means weight the nine configurations equally within each attribute and the two attributes equally. "
              "Unqualified methods remain visible; their available-case means are descriptive only.", "",
              "| Combination | Qualified / 18 | Macro BA | Smiling BA | Necktie BA | Smiling recall / AP | Necktie recall / AP |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["methods"]:
        smiling, necktie = (row["attributes"][name] for name in ("Smiling", "Wearing_Necktie"))
        lines.append(f"| {table_cell(row['label'])} | {row['qualified_cases']} / 18 | {number(row['descriptive_macro_ba'])} | "
                     f"{number(smiling['balanced_accuracy'])} | {number(necktie['balanced_accuracy'])} | "
                     f"{number(smiling['recall'])} / {number(smiling['average_precision'])} | "
                     f"{number(necktie['recall'])} / {number(necktie['average_precision'])} |")
    lines += ["", "## Time", "",
              "Primary workload: one fit plus one 4,000-image prediction. Each case uses the median of three "
              "per-repeat sums. Components below are means of independently calculated case medians; they are "
              "nested and must not be added to total fit/prediction. Parallel CV worker-wall sums are secondary "
              "study compute, not serial runtime, CPU time or a speedup estimate.", "",
              "| Combination | Normalized serial cost | Mean case fit + predict (s) | MF fit (s) | KNN fit (s) | Total fit (s) | Predict (s) | CV compute sum (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["methods"]:
        stages = row["stage_timing_means"]
        stage = lambda key: number(stages.get(key, {}).get("mean_case_median"), 3)
        lines.append(f"| {table_cell(row['label'])} | {number((row['score'] or {}).get('normalized_serial_cost'), 3)} | "
                     f"{number(row['mean_case_fit_plus_prediction_seconds'], 3)} | {stage('decomposition_fit_seconds')} | "
                     f"{stage('classifier_fit_seconds')} | {stage('total_fit_seconds')} | {stage('prediction_seconds')} | "
                     f"{number(row['tuning_compute_seconds_sum'], 2)} |")
    lines += ["", "## Sensitivity", "",
              "These descriptive recalculations do not change the primary winner. Adding tuning mixes parallel "
              "worker-wall compute with the serial workload and is explicitly a sensitivity only.", "",
              "| Macro / attribute margin | Prediction multiplier | Added tuning | Selected combination | Fallback |",
              "| --- | ---: | --- | --- | --- |"]
    for row in report["sensitivity"]:
        lines.append(f"| {row['macro_margin']:.3f} / {row['attribute_margin']:.3f} | {row['prediction_multiplier']:g} | "
                     f"{'yes' if row['include_parallel_tuning_in_workload'] else 'no'} | {table_cell(LABELS.get(row['winner'], 'None qualified'))} | "
                     f"{'yes' if row['fallback_used'] else 'no'} |")
    lines += ["", "Leave-one-size/seed checks keep the original operationally qualified methods:", ""]
    for row in report["leave_one_out"]:
        lines.append(f"- Omit {row['omitted_field']}={row['omitted_value']}: {LABELS.get(row['decision']['winner'], 'None qualified')}.")
    lines += ["", "## Exclusions and failed attempts", ""]
    for row in report["methods"]:
        lines.append(f"- {row['label']}: {row['cv_failed_candidate_count']} failed CV candidates "
                     f"(candidate counts include shared-representation duplicates across k); "
                     f"{row['preflight_excluded_candidate_count']} documented preflight exclusions; {len(row['exclusions'])} unqualified final cases.")
        for exclusion in row["exclusions"]:
            lines.append(f"  - {exclusion['jobid']}: {'; '.join(exclusion['issues'])}")
    lines += ["", "## Scope and audit trail", "",
              "The same 4,000 validation images are reused; training allocations overlap and sizes are nested. "
              "These are development configurations, not independent datasets. No statistical superiority, "
              "equivalence, unseen-person generalization, theoretical recovery or global tuning optimum is claimed. "
              "Initial development results were available before this follow-up protocol. Final-test "
              "evaluation has not been performed by this selector. No package source/default was changed.", "",
              "`decision.json` contains all 72 cases, selected parameters, CV scores, failure details, timing repeats, "
              "paired descriptive differences and input hashes. `cases.csv` supplies the flat case table. "
              "Peak whole-process memory is unmeasured. Committed failed tuning work is included; unmeasured "
              "interrupted attempts and restart overhead are not fabricated.", "",
              f"Run: `{report['source_run']}`.", "",
              f"Decision protocol SHA-256: `{report['decision_protocol_sha256']}`.", ""]
    return "\n".join(lines)


def draw_chart(report, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for row, color in zip(report["methods"], colors):
        if not row["operationally_qualified"]:
            continue
        selected = row["method"] == report["primary"]["winner"]
        ax.scatter(row["score"]["geometric_workload_seconds"], row["score"]["macro"],
                   color=color, marker="*" if selected else "o", s=200 if selected else 70,
                   label=row["label"] + (" (selected)" if selected else ""))
    ax.set(xlabel="Geometric mean serial fit + 4,000-image prediction (seconds)",
           ylabel="Macro balanced accuracy", title="Practical accuracy and cost across 18 development cases")
    ax.set_xscale("log")
    ax.grid(alpha=.2)
    if report["primary"]["qualified_methods"]:
        ax.legend(loc="best", fontsize=9)
    fig.savefig(destination, dpi=200, metadata={"Software": "Chang Cheng supervised NMF study"})
    plt.close(fig)


def write_report(run, protocol_path, output, *, chart=True):
    rows, protocol, inputs, configuration = load_cases(run, protocol_path)
    report = summarize(rows, protocol)
    report.update(schema_version=1, author="Chang Cheng", created_utc=datetime.now(timezone.utc).isoformat(),
                  winner_method=report["primary"]["winner"],
                  source_run=str(Path(run).resolve()), configuration_sha256=digest(configuration),
                  decision_protocol_sha256=inputs.hashes[str(Path(protocol_path).resolve())], decision_protocol=protocol,
                  selector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256=inputs.hashes, final_test_evaluated=False, package_defaults_modified=False,
                  interpretation="One scoped practical decision; overlapping development configurations are not independent datasets.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["jobid", "n_train", "seed", "attribute", "method", "qualified", "evaluation_status", "issues", "selected_parameters",
              "cv_balanced_accuracy", *METRICS, "prevalence", "tn", "fp", "fn", "tp", "median_fit_plus_prediction_seconds",
              "tuning_compute_seconds", "original_final_fit_seconds", "original_prediction_seconds", "cv_failed_candidates"]
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat.update(row["metrics"] or {})
            flat.pop("confusion_matrix", None)
            if row["metrics"]:
                matrix = row["metrics"]["confusion_matrix"]
                flat.update(tn=matrix[0][0], fp=matrix[0][1], fn=matrix[1][0], tp=matrix[1][1])
            flat["issues"] = "; ".join(row["issues"])
            flat["selected_parameters"] = json.dumps(row["selected_parameters"], sort_keys=True)
            flat["cv_failed_candidates"] = len(row["cv_failed_candidates"])
            flat["median_fit_plus_prediction_seconds"] = case_cost(row) if row["qualified"] else None
            writer.writerow(flat)
    if chart:
        draw_chart(report, output / "accuracy-cost.png")
    inputs.verify_unchanged()
    (output / "decision.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "decision.md").write_text(markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--decision-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/default-selection/decision"))
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()
    report = write_report(args.run, args.decision_protocol, args.output, chart=not args.no_chart)
    print(json.dumps({"status": report["primary"]["status"], "selected_combination": report["primary"]["winner"],
                      "report": str(args.output.resolve() / "decision.md")}, indent=2))
    return 0 if report["primary"]["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
