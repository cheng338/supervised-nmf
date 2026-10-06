"""Run an explicit, resumable CelebA comparison for general-default selection.

Authorship: Chang Cheng. The method remains the dissertation-derived screened
NMF classifier. This runner adds no adaptive search or default-selection rule.
CV reads training arrays only; development evaluation is a separate serial
command, permitted only after every declared job has completed all five folds.
Each representation is fitted once per fold and shared across its declared k.

    python scripts/celeba_default_study.py --protocol protocol.json --output RUN --mode cv --workers 8
    python scripts/celeba_default_study.py --protocol protocol.json --output RUN --mode evaluate

Protocol v1: {"schema_version": 1, "author": "Chang Cheng", "cv": 5,
"native_threads": 1, "jobs": [{"id": "seed42-smiling-cd-percentile",
"cache": "data/celeba-development-4000-seed42", "seed": 42,
"attribute": "Smiling", "method": "screened_cd_percentile", "max_iter": 10000,
"tol": 0.0001, "grid": [{"n_components": [10, 15], "percentile": [2, 3],
"n_neighbors": [15, 25, 41]}]}]}. Cache paths are relative to the repository.
Overlapping boxes are deduplicated in first-occurrence ParameterGrid order.
Numerically invalid candidates remain in the results and cannot be selected.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import re
import sys
import time
import uuid
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits

from supervised_nmf.evaluation import (
    _diagnostics, _digest, _jsonable, _nonconverged, _warning_records,
    comparison_models, data_fingerprint, evaluate_classifier, provenance,
    refinement_diagnostics, write_json,
)
from supervised_nmf.study import _checkpoint_lock, _read_receipt, _write_receipt, calibration_knn_grid
from profile_celeba_stages import _runtime_identity, profile_selected_estimator


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_identity():
    environment = provenance(1)
    files = [Path(__file__), ROOT / "scripts" / "profile_celeba_stages.py"]
    return {"package_source_sha256": environment["source_sha256"],
            "runner_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "runtime": _runtime_identity(environment)}


def _check_source(expected):
    if _source_identity() != expected:
        raise ValueError("Source or numerical runtime changed; preserve this run and start another directory.")


def _safe_name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError(f"{label} must use 1-80 letters, digits, underscores or hyphens.")
    return value


def _integer(value, label, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {label}: {value!r}.")
    return int(value)


def _estimator(job):
    return comparison_models(random_state=job["seed"], max_iter=job["max_iter"], tol=job["tol"])[job["method"]][0]


def expand_job(job):
    """Build exact candidate and representation lists without adding settings."""
    job = deepcopy(job)
    _safe_name(job["id"], "Job id")
    _safe_name(job["attribute"], "Attribute")
    job["seed"] = _integer(job["seed"], "seed", 0, 2 ** 32 - 1)
    job["max_iter"] = _integer(job["max_iter"], "max_iter")
    if isinstance(job["tol"], bool) or not isinstance(job["tol"], Real) or not np.isfinite(job["tol"]) or job["tol"] < 0:
        raise ValueError("tol must be finite and nonnegative.")
    models = comparison_models(random_state=job["seed"], max_iter=job["max_iter"], tol=job["tol"])
    if job["method"] not in models or job["method"] == "majority_class":
        raise ValueError("Job method must be a supported tunable comparison_models method.")
    template, allowed = models[job["method"]]
    if not isinstance(job["grid"], list) or not job["grid"] or any(not isinstance(box, dict) for box in job["grid"]):
        raise ValueError("grid must be a nonempty list of ParameterGrid dictionaries.")
    if any(set(box) != set(allowed) for box in job["grid"]):
        raise ValueError(f"Each box must explicitly supply exactly these grid keys: {sorted(allowed)}")
    neighbor_key = next(key for key in allowed if key.rsplit("__", 1)[-1] == "n_neighbors")
    candidates, seen, duplicates = [], set(), 0
    for parameters in ParameterGrid(job["grid"]):
        for key, value in parameters.items():
            short = key.rsplit("__", 1)[-1]
            if short in {"n_components", "n_neighbors"}:
                _integer(value, key)
            elif isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric.")
            elif short == "percentile" and not 0 < value <= 100:
                raise ValueError("percentile must be in (0, 100].")
            elif short == "score_threshold" and value < 0:
                raise ValueError("score_threshold must be nonnegative.")
        clone(template).set_params(**parameters)
        key = _digest(parameters)
        if key in seen:
            duplicates += 1
        else:
            candidates.append(parameters)
            seen.add(key)
    if not candidates:
        raise ValueError("Each job must contain a candidate.")
    representations = {}
    for index, parameters in enumerate(candidates):
        representation = {key: value for key, value in parameters.items() if key != neighbor_key}
        key = _digest(representation)
        entry = representations.setdefault(key, {"parameters": representation, "candidate_indices": [], "neighbors": []})
        entry["candidate_indices"].append(index)
        entry["neighbors"].append(parameters[neighbor_key])
    chunks = []
    for index, representation in enumerate(representations.values()):
        chunks.append({"id": f"c{index:04d}", "candidate_indices": representation["candidate_indices"],
                       "grid": [{**{key: [value] for key, value in representation["parameters"].items()},
                                 neighbor_key: representation["neighbors"]}]})
    cache = Path(job["cache"])
    job["cache"] = str((ROOT / cache).resolve() if not cache.is_absolute() else cache.resolve())
    return {"job": job, "candidates": candidates, "duplicate_box_candidates_removed": duplicates, "chunks": chunks}


def _cache_identity(cache, jobs, *, validation=False):
    """Verify values and manifest; CV never opens validation arrays."""
    cache = Path(cache)
    manifest = _read_json(cache / "manifest.json")
    if manifest.get("official_test_loaded") is not False:
        raise ValueError("Cache must explicitly exclude official test data.")
    if any(manifest["seed"] != job["seed"] for job in jobs):
        raise ValueError("Job seed must match the cache training seed.")
    for split, partition in (("train", 0), ("validation", 1)):
        record = manifest["splits"][split]
        ids = record["image_ids"]
        if not ids or len(ids) != len(set(ids)) or record["metadata"]["partition_code"] != partition:
            raise ValueError("Cached image IDs or official partition differ from the required split.")
    if set(manifest["splits"]["train"]["image_ids"]) & set(manifest["splits"]["validation"]["image_ids"]):
        raise ValueError("Training and validation image IDs overlap.")
    split = "validation" if validation else "train"
    record = manifest["splits"][split]
    X = np.load(cache / f"{split}-X.npy", mmap_mode="r", allow_pickle=False)
    if X.shape != (len(record["image_ids"]), 4096) or X.dtype != np.float32 or not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
        raise ValueError("Cache must contain finite original 64x64 float32 images in [0, 1].")
    for attribute in sorted({job["attribute"] for job in jobs}):
        y = np.load(cache / f"{split}-{attribute}-y.npy", allow_pickle=False)
        counts = np.bincount(y, minlength=2) if y.dtype.kind in "iu" and np.all((y == 0) | (y == 1)) else None
        expected = record["attributes"][attribute]
        if (y.shape != (len(X),) or counts is None or len(counts) != 2
                or counts.tolist() != expected["class_counts"]
                or data_fingerprint(X, y) != expected["data_sha256"]):
            raise ValueError(f"Cached values, labels, counts or fingerprint differ for {split}/{attribute}.")
        if split == "train" and min(counts) < 5:
            raise ValueError("Both training classes require at least five observations.")
    return {"manifest_sha256": _digest(manifest), "manifest": manifest}


def initialize(protocol_path, output):
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = _read_json(protocol_path)
    if (protocol.get("schema_version") != 1 or protocol.get("author") != "Chang Cheng"
            or protocol.get("cv") != 5 or protocol.get("native_threads") != 1):
        raise ValueError("Protocol v1 requires Chang Cheng authorship, five folds and one native thread.")
    if not isinstance(protocol.get("jobs"), list) or not protocol["jobs"]:
        raise ValueError("Protocol must contain jobs.")
    _integer(protocol.get("profile_repeats", 1), "profile_repeats")
    plans = [expand_job(job) for job in protocol["jobs"]]
    if len({plan["job"]["id"] for plan in plans}) != len(plans):
        raise ValueError("Job ids must be unique.")
    caches = {}
    for cache in sorted({plan["job"]["cache"] for plan in plans}):
        caches[cache] = _cache_identity(cache, [plan["job"] for plan in plans if plan["job"]["cache"] == cache])
    identity = {"schema_version": 1, "protocol": protocol,
                "protocol_file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "sources": _source_identity(), "caches": caches, "plans": plans,
                "official_test_loaded": False, "maximum_workers": 8, "native_threads": 1}
    configuration = output / "configuration.json"
    if configuration.exists():
        if _read_receipt(configuration) != identity:
            raise ValueError("Resume configuration differs in protocol, data, source or runtime.")
    else:
        if any((output / plan["job"]["id"]).exists() for plan in plans):
            raise ValueError("Unidentified job output exists without a configuration; choose a new directory.")
        _write_receipt(configuration, identity)
        write_json(output / "protocol.json", protocol)
        write_json(output / "provenance.json", provenance(1))
    return identity


def _worker(work):
    output, plan, chunk, expected_sources = work
    _check_source(expected_sources)
    job = plan["job"]
    X = np.load(Path(job["cache"]) / "train-X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(Path(job["cache"]) / f"train-{job['attribute']}-y.npy", allow_pickle=False)
    report = calibration_knn_grid(_estimator(job), chunk["grid"], X, y, cv=5, seed=job["seed"], threads=1,
                                  checkpoint_dir=Path(output) / job["id"] / chunk["id"])
    _check_source(expected_sources)
    return {"job": job["id"], "chunk": chunk["id"], "candidates": len(report["candidates"]),
            "valid_candidates": sum(row["valid"] for row in report["candidates"]),
            "new_representation_fits": report["new_representation_fits"],
            "reused_representation_folds": report["reused_representation_folds"], "worker_pid": os.getpid()}


def aggregate_cv(output, configuration, *, require_complete=False):
    """Check complete grid coverage and select by training CV with canonical ties."""
    output = Path(output)
    _check_source(configuration["sources"])
    summaries, missing = {}, []
    for plan in configuration["plans"]:
        job = plan["job"]
        paths = [output / job["id"] / chunk["id"] / "report.json" for chunk in plan["chunks"]]
        absent = [str(path.relative_to(output)) for path in paths if not path.exists()]
        if absent:
            missing.extend(absent)
            continue
        candidates, costs = [], {}
        manifest = configuration["caches"][job["cache"]]["manifest"]
        X = np.load(Path(job["cache"]) / "train-X.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(Path(job["cache"]) / f"train-{job['attribute']}-y.npy", allow_pickle=False)
        expected_folds = _jsonable([{"training_indices": a, "validation_indices": b}
                                   for a, b in StratifiedKFold(5, shuffle=True, random_state=job["seed"]).split(X, y)])
        report_receipts = {}
        for chunk, path in zip(plan["chunks"], paths):
            report = _read_receipt(path)
            identity = report["identity"]
            if _read_receipt(path.parent / "identity.json") != identity:
                raise ValueError("CV report does not match its original checkpoint identity.")
            calibration = identity["calibration"]
            expected_candidates = list(ParameterGrid(chunk["grid"]))
            if (report["state"] != "cv_completed" or report["fingerprint"] != _digest(identity)
                    or calibration["data_sha256"] != manifest["splits"]["train"]["attributes"][job["attribute"]]["data_sha256"]
                    or calibration["candidates"] != expected_candidates or calibration["cv"] != 5
                    or calibration["seed"] != job["seed"] or calibration["threads"] != 1 or calibration["refit"] is not False
                    or calibration["parameters"] != _jsonable(_estimator(job).get_params(deep=True))
                    or calibration["source_sha256"] != configuration["sources"]["package_source_sha256"]
                    or identity["folds"] != expected_folds):
                raise ValueError(f"CV report identity differs: {path}")
            if [row["parameters"] for row in report["candidates"]] != expected_candidates:
                raise ValueError("CV report does not contain its exact declared candidates.")
            canonical = {_digest(plan["candidates"][index]): index for index in chunk["candidate_indices"]}
            for item in report["candidates"]:
                item = deepcopy(item)
                if ([fold["fold"] for fold in item["folds"]] != list(range(5))
                        or item["valid"] != all(fold["status"] == "valid" for fold in item["folds"])):
                    raise ValueError("CV candidate does not contain five qualified folds.")
                scores = [fold["balanced_accuracy"] for fold in item["folds"]] if item["valid"] else []
                if item["mean_balanced_accuracy"] != (float(np.mean(scores)) if scores else None):
                    raise ValueError("CV candidate aggregate score differs from its folds.")
                item.update(index=canonical[_digest(item["parameters"])], chunk=chunk["id"])
                candidates.append(item)
            for name, value in report["compute_seconds"].items():
                costs[name] = costs.get(name, 0.0) + value
            report_receipts[chunk["id"]] = _digest(report)
        candidates.sort(key=lambda row: row["index"])
        if [row["index"] for row in candidates] != list(range(len(plan["candidates"]))):
            raise ValueError("Completed chunks do not cover the exact deduplicated job grid.")
        valid = [row for row in candidates if row["valid"]]
        winner = min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])) if valid else None
        summary = {"author": "Chang Cheng", "job": job, "configuration_sha256": _digest(configuration),
                   "state": "cv_completed", "candidates": candidates, "selected_candidate": winner,
                   "selected_parameters": winner["parameters"] if winner else None,
                   "cv_balanced_accuracy": winner["mean_balanced_accuracy"] if winner else None,
                   "compute_seconds": costs, "tuning_compute_seconds": sum(costs.values()),
                   "report_receipts": report_receipts,
                   "refinement": refinement_diagnostics(plan["candidates"], winner["parameters"] if winner else None),
                   "duplicate_box_candidates_removed": plan["duplicate_box_candidates_removed"],
                   "selection_role": "highest_complete_training_cv_score_canonical_first_occurrence_tie_break",
                   "timing_note": "Shared representation fit includes its initial carrier KNN fit, counted once; it is not an MF-only timer. Committed compute excludes I/O and lost interrupted work.",
                   "defaults_frozen": False, "official_test_loaded": False}
        _write_receipt(output / job["id"] / "calibration-summary.json", summary)
        summaries[job["id"]] = summary
    if require_complete and missing:
        raise ValueError(f"All protocol CV jobs must finish before validation; {len(missing)} chunks are missing.")
    report = {"author": "Chang Cheng", "configuration_sha256": _digest(configuration),
              "state": "cv_completed" if not missing else "cv_incomplete", "jobs": summaries,
              "summary_paths": {job_id: str(Path(job_id) / "calibration-summary.json") for job_id in summaries},
              "missing_chunks": missing, "official_test_loaded": False, "defaults_frozen": False}
    _write_receipt(output / "cv-summary.json", report)
    return report


def run_cv(output, configuration, *, workers=8, job_ids=None):
    workers = _integer(workers, "workers", 1, 8)
    chosen = set(job_ids or [plan["job"]["id"] for plan in configuration["plans"]])
    if not chosen <= {plan["job"]["id"] for plan in configuration["plans"]}:
        raise ValueError("--jobs contains an unknown protocol job id.")
    work = [(str(output), plan, chunk, configuration["sources"]) for plan in configuration["plans"]
            if plan["job"]["id"] in chosen for chunk in plan["chunks"]]
    receipts = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, item): item for item in work}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            receipts.append(result)
            print(f"CV {index}/{len(work)} {result['job']} {result['chunk']} "
                  f"valid={result['valid_candidates']}/{result['candidates']} "
                  f"new={result['new_representation_fits']} reused={result['reused_representation_folds']}", flush=True)
    summary = aggregate_cv(output, configuration)
    return {"state": summary["state"], "completed_chunks": receipts, "missing_chunks": summary["missing_chunks"]}


def evaluate(output, configuration):
    """Serial independent final fits plus exact-reproduction stage profiling."""
    output = Path(output)
    cv = aggregate_cv(output, configuration, require_complete=True)
    for cache, saved in configuration["caches"].items():
        actual = _cache_identity(cache, [plan["job"] for plan in configuration["plans"] if plan["job"]["cache"] == cache], validation=True)
        if actual != saved:
            raise ValueError("Validation cache manifest changed after CV.")
    results = {}
    for plan in configuration["plans"]:
        _check_source(configuration["sources"])
        job, selected = plan["job"], cv["jobs"][plan["job"]["id"]]
        identity = {"configuration_sha256": _digest(configuration), "job_id": job["id"],
                    "selected_candidate_sha256": _digest(selected["selected_candidate"]),
                    "validation_data_sha256": configuration["caches"][job["cache"]]["manifest"]["splits"]["validation"]["attributes"][job["attribute"]]["data_sha256"]}
        path = output / job["id"] / "evaluation.json"
        if path.exists():
            result = _read_receipt(path)
            if result["identity"] != identity:
                raise ValueError("Saved selected-model evaluation identity differs.")
            results[job["id"]] = result
            continue
        result = {"author": "Chang Cheng", "job": job, "identity": identity,
                  "jobid": job["id"], "attribute": job["attribute"], "method": job["method"], "seed": job["seed"],
                  "n_train": len(configuration["caches"][job["cache"]]["manifest"]["splits"]["train"]["image_ids"]),
                  "selected_parameters": selected["selected_parameters"],
                  "cv_balanced_accuracy": selected["cv_balanced_accuracy"],
                  "tuning_compute_seconds": selected["tuning_compute_seconds"],
                  "evaluation_role": "development_validation_used_for_default_selection_not_final_test",
                  "is_fallback": False, "defaults_frozen": False, "official_test_loaded": False,
                  "final_fit_seconds": None, "profiling": [], "profile_medians_seconds": {}, "profiling_valid": False}
        if selected["selected_parameters"] is None:
            result["evaluation"] = {"status": "held", "reason": "No candidate qualified in all five training folds."}
        else:
            cache = Path(job["cache"])
            X = np.load(cache / "train-X.npy", mmap_mode="r", allow_pickle=False)
            y = np.load(cache / f"train-{job['attribute']}-y.npy", allow_pickle=False)
            Xv = np.load(cache / "validation-X.npy", mmap_mode="r", allow_pickle=False)
            yv = np.load(cache / f"validation-{job['attribute']}-y.npy", allow_pickle=False)
            ids = configuration["caches"][job["cache"]]["manifest"]["splits"]["validation"]["image_ids"]
            template = _estimator(job).set_params(**selected["selected_parameters"])
            model, caught = clone(template), []
            with threadpool_limits(limits=1):
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(X, y)
                    result["final_fit_seconds"] = time.perf_counter() - start
                    result["diagnostics"] = _diagnostics(model)
                    if _nonconverged(model, caught):
                        result["evaluation"] = {"status": "held", "reason": "Selected final fit did not converge."}
                    else:
                        result["evaluation"] = evaluate_classifier(model, Xv, yv, threads=1, sample_ids=ids)
                except Exception as error:
                    result["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
                finally:
                    if result["final_fit_seconds"] is None:
                        result["final_fit_seconds"] = time.perf_counter() - start
                    result["final_fit_warnings"] = _warning_records(caught)
                if result["evaluation"]["status"] == "valid":
                    for repeat in range(configuration["protocol"].get("profile_repeats", 1)):
                        profile = profile_selected_estimator(template, X, y, Xv, yv,
                                                             reference=result["evaluation"], sample_ids=ids)
                        profile["repeat"] = repeat
                        result["profiling"].append(profile)
                    result["profiling_valid"] = all(item["status"] == "valid" and item["exact_prediction_match"]
                                                    for item in result["profiling"])
                    if result["profiling_valid"]:
                        for stage in result["profiling"][0]["timing"]:
                            values = [item["timing"][stage] for item in result["profiling"]]
                            result["profile_medians_seconds"][stage] = float(np.median(values)) if all(value is not None for value in values) else None
                        result["profile_medians_seconds"]["fit_plus_prediction_seconds"] = float(np.median([
                            item["timing"]["total_fit_seconds"] + item["timing"]["prediction_seconds"]
                            for item in result["profiling"]]))
                    result["profile_timing_note"] = "Additional serial fit and prediction; exact original labels and probabilities required. Component timers are nested within total fit/predict. Do not add this fit cost to original fitting or CV costs."
        _check_source(configuration["sources"])
        _write_receipt(path, result)
        results[job["id"]] = result
        print(f"Evaluation {len(results)}/{len(configuration['plans'])} {job['id']} "
              f"status={result['evaluation']['status']} exact_profiles={sum(item['exact_prediction_match'] for item in result['profiling'])}/{len(result['profiling'])}", flush=True)
    report = {"author": "Chang Cheng", "state": "development_evaluation_completed", "configuration_sha256": _digest(configuration),
              "jobs": results, "defaults_frozen": False, "official_test_loaded": False,
              "evaluation_role": "development_validation_used_for_default_selection_not_final_test"}
    _write_receipt(output / "comparison.json", report)
    return {"state": report["state"], "evaluations": len(results),
            "valid_evaluations": sum(row["evaluation"]["status"] == "valid" for row in results.values()),
            "exact_profiles": sum(item.get("exact_prediction_match", False) for row in results.values() for item in row["profiling"]),
            "models_with_valid_profiling": sum(row["profiling_valid"] for row in results.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Exact resumable run directory.")
    parser.add_argument("--mode", choices=("cv", "evaluate"), default="cv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jobs", nargs="+", help="CV only: execute these ids from the immutable full protocol.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8; each worker uses one numerical thread")
    if args.mode == "evaluate" and args.jobs:
        parser.error("Evaluation requires all jobs and does not support --jobs")
    args.output.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(args.output):
        configuration = initialize(args.protocol, args.output)
        attempt_path = args.output / "executions" / f"{uuid.uuid4().hex}.json"
        attempt = {"started_at": _now(), "mode": args.mode, "workers": args.workers if args.mode == "cv" else 1,
                   "native_threads": 1, "requested_job_ids": args.jobs,
                   "configuration_sha256": _digest(configuration), "state": "running"}
        _write_receipt(attempt_path, attempt)
        start = time.perf_counter()
        try:
            result = (run_cv(args.output, configuration, workers=args.workers, job_ids=args.jobs)
                      if args.mode == "cv" else evaluate(args.output, configuration))
            attempt.update(state="completed", result=result)
        except BaseException as error:
            attempt.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            attempt.update(finished_at=_now(), current_call_wall_seconds=time.perf_counter() - start)
            _write_receipt(attempt_path, attempt)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
