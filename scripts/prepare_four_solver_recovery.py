"""Prepare a versioned numerical-repair run without fitting models.

Author: Chang Cheng. Only unchanged CD/SPA/MU and baseline CV receipts may be
inherited. ANLS-BPP always starts again after its implementation changes. Original
receipt bytes and the original source freeze are never rewritten.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.model_selection import ParameterGrid

import fashion_mnist_study as shared
import run_four_solver_benchmarks as orchestration
from supervised_nmf.evaluation import _digest, _jsonable
from supervised_nmf.study import _checkpoint_lock, _layout, _read_receipt, _write_receipt

REUSABLE_METHODS = frozenset({"screened_cd_absolute", "screened_spa_absolute", "screened_mu_absolute",
                             "screened_pca_absolute", "screening_only_absolute", "raw_knn"})
BPP = "screened_anls_bpp_absolute"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_record(path, inputs, *, envelope=True):
    path = Path(path).resolve()
    inputs[str(path)] = orchestration.sha256(path)
    return _read_receipt(path) if envelope else json.loads(path.read_text(encoding="utf-8"))


def fold_cost(record):
    values = [record["representation_fit_seconds"], record["projection_seconds"]]
    values.extend(item[key] for item in record["neighbors"].values() for key in ("fit_seconds", "prediction_seconds"))
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value >= 0 for value in values), "Invalid recorded compute time.")
    return float(sum(values))


def validate_identity(identity, plan, chunk, configuration, estimator, runtime):
    job = plan["job"]
    kind, neighbor, _ = _layout(estimator)
    candidates = list(ParameterGrid(chunk["grid"]))
    calibration = {"estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
                   "parameters": _jsonable(estimator.get_params(deep=True)), "candidates": candidates,
                   "data_sha256": job["training_data_sha256"], "cv": 5, "seed": job["seed"], "threads": 1,
                   "refit": False, "python": runtime["python"], "platform": runtime["platform"],
                   "numerical_runtime": runtime["numerical_libraries"], "software": runtime["versions"],
                   "source_sha256": configuration["sources"]["package_source_sha256"]}
    require(_digest(identity["folds"]) == plan["fold_indices_sha256"], "Inherited fold assignments differ.")
    expected = {"schema_version": 1, "method": "explicit_grid_shared_knn_representation", "calibration": calibration,
                "runtime": runtime, "folds": identity["folds"], "neighbor_parameter": neighbor, "representation_backend": kind}
    require(identity == expected, "Inherited data, grid, numerical settings, estimator source or runtime differ.")
    require(len(identity["folds"]) == 5, "Expected five fold assignments.")
    return neighbor, candidates


def validate_chunk(directory, plan, chunk, configuration, inputs, estimator, runtime):
    """Validate all inherited envelopes, keys, fold contents and report arithmetic."""
    identity = read_record(directory / "identity.json", inputs)
    neighbor, candidates = validate_identity(identity, plan, chunk, configuration, estimator, runtime)
    fingerprint = _digest(identity)
    groups = {}
    for parameters in candidates:
        representation = {key: value for key, value in parameters.items() if key != neighbor}
        group = groups.setdefault(_digest(representation), {"parameters": representation, "neighbors": []})
        group["neighbors"].append(int(parameters[neighbor]))
    expected_keys = {_digest({"identity": fingerprint, "representation": representation, "fold": fold}):
                     (representation, fold) for representation in groups for fold in range(5)}
    records, receipt_files = {}, {directory / "identity.json"}
    running_attempts = 0
    for category in ("folds", "attempts"):
        for path in sorted((directory / category).glob("*.json")):
            record = read_record(path, inputs)
            require(record["key"] in expected_keys and record["identity"] == fingerprint, "Inherited fold/attempt belongs to another identity.")
            representation, fold = expected_keys[record["key"]]
            require(record["fold"] == fold and record["representation_key"] == representation
                    and record["parameters"] == groups[representation]["parameters"], "Inherited representation/fold differs.")
            if category == "folds":
                require(path.stem == record["key"] and record["state"] == "completed", "Fold receipt is not terminal or has wrong filename.")
                require(record["status"] in {"valid", "nonconverged", "failed"}, "Unknown inherited fold qualification.")
                if record["status"] == "valid":
                    require(set(record["neighbors"]) == {str(value) for value in groups[representation]["neighbors"]},
                            "Inherited fold lacks a declared neighbor result.")
                fold_cost(record)
                records[record["key"]] = record
            else:
                require(record["state"] in {"running", "completed", "interrupted", "failed"}, "Unknown historical attempt state.")
                running_attempts += int(record["state"] == "running")
            receipt_files.add(path)
    report_path = directory / "report.json"
    report = None
    if report_path.exists():
        report = read_record(report_path, inputs)
        receipt_files.add(report_path)
        require(set(records) == set(expected_keys), "Completed report lacks committed folds.")
        ordered = [records[key] for key in expected_keys]
        require(report["identity"] == identity and report["fingerprint"] == fingerprint and report["state"] == "cv_completed"
                and report["representation_folds"] == ordered and report["refit_performed"] is False and report["is_fallback"] is False,
                "Inherited report does not match committed fold evidence.")
        reconstructed = []
        for index, parameters in enumerate(candidates):
            representation = _digest({key: value for key, value in parameters.items() if key != neighbor})
            rows = []
            for fold in range(5):
                key = _digest({"identity": fingerprint, "representation": representation, "fold": fold})
                source = records[key]
                if source["status"] == "valid":
                    row = deepcopy(source["neighbors"][str(parameters[neighbor])])
                    row["warnings"] = source["warnings"] + row["warnings"]
                else:
                    row = {"status": source["status"], "fit_seconds": 0.0, "prediction_seconds": 0.0, "warnings": source["warnings"]}
                    row.update({key: source[key] for key in ("error", "error_type") if key in source})
                row.update(fold=fold, shared_representation_key=source["key"], diagnostics=source.get("diagnostics", {}))
                rows.append(row)
            valid = all(row["status"] == "valid" for row in rows)
            scores = [row["balanced_accuracy"] for row in rows] if valid else []
            reconstructed.append({"index": index, "parameters": parameters, "folds": rows, "valid": valid,
                                  "mean_balanced_accuracy": float(np.mean(scores)) if valid else None,
                                  "fold_score_std": float(np.std(scores, ddof=1)) if valid else None,
                                  "invalid_fold_count": sum(row["status"] != "valid" for row in rows)})
        require(report["candidates"] == reconstructed, "Inherited report candidate scores differ from fold receipts.")
        costs = {"representation_fit_seconds": sum(row["representation_fit_seconds"] for row in ordered),
                 "validation_projection_seconds": sum(row["projection_seconds"] for row in ordered),
                 "knn_fit_seconds": sum(item["fit_seconds"] for row in ordered for item in row["neighbors"].values()),
                 "knn_prediction_seconds": sum(item["prediction_seconds"] for row in ordered for item in row["neighbors"].values())}
        require(report["compute_seconds"] == costs and report["tuning_seconds"] == sum(costs.values()),
                "Inherited report compute accounting differs from folds.")
    # Lock files are operational and never migrated. Uncommitted temporary files
    # stay in the original run; only checked JSON receipts can be reused.
    ignored = [str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() and path not in receipt_files]
    return {"files": sorted(receipt_files), "ignored_files": ignored, "completed_report": report is not None,
            "committed_folds": len(records), "historical_running_attempts": running_attempts,
            "compute_seconds": sum(fold_cost(row) for row in records.values())}


def migrate_checkpoints(parent_fashion, child_fashion, parent, child, inputs):
    """Plan and verify everything before copying any reusable checkpoint file."""
    parent_fashion, child_fashion = Path(parent_fashion), Path(child_fashion)
    require(parent["sources"]["package_source_sha256"] == child["sources"]["package_source_sha256"],
            "Package implementation changed; unchanged-method receipt migration is not valid.")
    require(parent["splits"] == child["splits"], "Dataset splits or class orientation changed.")
    old_plans = {plan["job"]["id"]: plan for plan in parent["plans"]}
    require(len(old_plans) == len(parent["plans"]), "Duplicate parent job IDs.")
    require({plan["job"]["id"] for plan in child["plans"]} == set(old_plans), "Parent/child job coverage differs.")
    checked, excluded_bpp, runtime_cache = [], [], {}
    for plan in child["plans"]:
        job, old = plan["job"], old_plans[plan["job"]["id"]]
        require(job["method"] == old["job"]["method"], "A job changed solver identity.")
        if job["method"] == BPP:
            require(not any((child_fashion / job["id"]).glob("c*")), "BPP checkpoints may not be inherited into the repaired run.")
            excluded_bpp.append({"job": job["id"], "declared_chunks": len(plan["chunks"]),
                                 "old_completed_reports": sum((parent_fashion / job["id"] / chunk["id"] / "report.json").exists()
                                                              for chunk in old["chunks"])})
            continue
        require(job["method"] in REUSABLE_METHODS, "Method is not on the explicit unchanged-method allowlist.")
        require({key: value for key, value in plan.items() if key != "job"} == {key: value for key, value in old.items() if key != "job"},
                "An unchanged-method grid, canonical order, exclusion or fold assignment changed.")
        for key in ("pair", "seed", "method", "n_train", "max_iter", "tol", "training_data_sha256"):
            require(job[key] == old["job"][key], f"Unchanged-method job setting differs: {key}")
        estimator = shared._estimator(job)
        runtime_key = (job["method"], job["seed"])
        if runtime_key not in runtime_cache:
            runtime_cache[runtime_key] = shared._calibration_runtime(estimator, 1)
        for chunk in plan["chunks"]:
            source = parent_fashion / job["id"] / chunk["id"]
            target = child_fashion / job["id"] / chunk["id"]
            if not source.exists():
                require(not target.exists(), "Unexplained destination checkpoint exists.")
                continue
            with _checkpoint_lock(source):
                evidence = validate_chunk(source, plan, chunk, child, inputs, estimator, runtime_cache[runtime_key])
            expected_names = {path.relative_to(source) for path in evidence["files"]}
            if target.exists():
                actual_names = {path.relative_to(target) for path in target.rglob("*") if path.is_file() and path.name != ".study.lock"}
                require(actual_names <= expected_names, "Destination contains results not supplied by this migration.")
            for path in evidence["files"]:
                destination = target / path.relative_to(source)
                require(not destination.exists() or orchestration.sha256(destination) == inputs[str(path.resolve())],
                        "Destination receipt differs; no existing result will be overwritten.")
            checked.append({"job": job["id"], "method": job["method"], "chunk": chunk["id"], "source": source,
                            "target": target, **evidence})
    files, counts, compute = [], Counter(), defaultdict(float)
    for evidence in checked:
        for source in evidence["files"]:
            destination = evidence["target"] / source.relative_to(evidence["source"])
            expected_hash = inputs[str(source.resolve())]
            require(orchestration.sha256(source) == expected_hash, "Parent receipt changed during migration.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            require(orchestration.sha256(destination) == expected_hash, "Copied receipt checksum differs.")
            files.append({"source": str(source.resolve()), "destination": str(destination.resolve()), "sha256": expected_hash})
        counts["copied_chunks"] += 1
        counts["completed_reports"] += int(evidence["completed_report"])
        counts["committed_folds"] += evidence["committed_folds"]
        counts["historical_running_attempts"] += evidence["historical_running_attempts"]
        if not evidence["completed_report"]:
            counts["partial_chunk_committed_folds"] += evidence["committed_folds"]
        compute[evidence["method"]] += evidence["compute_seconds"]
    return {"counts": dict(counts), "copied_files": files, "inherited_cv_compute_seconds_by_method": dict(compute),
            "inherited_cv_compute_seconds": sum(compute.values()), "excluded_bpp": excluded_bpp,
            "ignored_uncommitted_or_lock_files": [{"job": item["job"], "chunk": item["chunk"], "files": item["ignored_files"]}
                                                  for item in checked if item["ignored_files"]]}


def prepare(parent_run, output, fashion_protocol, celeba_protocol, reason):
    require(Path(sys.prefix).name == "py12_df_aocl", "This historical recovery study requires the py12_df_aocl environment.")
    parent_run, output = Path(parent_run).resolve(), Path(output).resolve()
    require(parent_run != output and not output.is_relative_to(parent_run) and not parent_run.is_relative_to(output),
            "Parent and child runs must be separate nonnested directories.")
    require(bool(reason.strip()), "Record the concrete numerical-repair reason.")
    started, inputs = time.perf_counter(), {}
    with orchestration.output_lock(parent_run), orchestration.output_lock(output):
        require(not (output / "status.json").exists(), "Production already started; use orchestration resume rather than preparation.")
        parent_freeze = read_record(parent_run / "source-freeze.json", inputs, envelope=False)
        for relative, expected in parent_freeze["files"].items():
            path = parent_run / "source" / relative
            require(orchestration.sha256(path) == expected, "Original frozen source snapshot differs.")
            inputs[str(path.resolve())] = expected
        parent = read_record(parent_run / "fashion/configuration.json", inputs)
        require(parent.get("official_test_loaded") is False, "Original run must exclude official-test data.")
        child_freeze = orchestration.freeze_run(output, fashion_protocol, celeba_protocol)
        child = shared.initialize(fashion_protocol, output / "fashion")
        evidence = migrate_checkpoints(parent_run / "fashion", output / "fashion", parent, child, inputs)
        executions = [{"path": str(path.resolve()), "record": read_record(path, inputs)}
                      for path in sorted((parent_run / "fashion/executions").glob("*.json"))]
        history = []
        for path in [parent_run / "status.json", *sorted(parent_run.glob("pause-*.json"))]:
            if path.exists():
                history.append({"path": str(path.resolve()), "record": read_record(path, inputs, envelope=False)})
        for path in parent_run.glob("fashion-cv-attempt*.log"):
            inputs[str(path.resolve())] = orchestration.sha256(path)
        for path, expected in inputs.items():
            require(orchestration.sha256(path) == expected, "Parent evidence changed during recovery preparation.")
        shared._check_source(child["sources"])
        require(orchestration.freeze_run(output, fashion_protocol, celeba_protocol) == child_freeze, "Child sources changed during preparation.")
        report = {"schema_version": 1, "author": "Chang Cheng", "state": "prepared_no_models_fitted", "created_utc": orchestration.now(),
                  "parent_run": str(parent_run), "child_run": str(output), "amendment_reason": reason,
                  "parent_source_freeze_sha256": inputs[str(parent_run / "source-freeze.json")],
                  "parent_configuration_sha256": _digest(parent), "child_configuration_sha256": _digest(child),
                  "child_source_freeze_sha256": orchestration.sha256(output / "source-freeze.json"),
                  "child_configuration_file_sha256": orchestration.sha256(output / "fashion/configuration.json"),
                  "child_protocol_files_sha256": {str(Path(path).resolve()): orchestration.sha256(path) for path in (fashion_protocol, celeba_protocol)},
                  "changed_frozen_files": {name: {"before": parent_freeze["files"].get(name), "after": child_freeze["files"].get(name)}
                                           for name in sorted(set(parent_freeze["files"]) | set(child_freeze["files"]))
                                           if parent_freeze["files"].get(name) != child_freeze["files"].get(name)},
                  "parent_execution_receipts": executions, "parent_orchestration_and_pause_records": history,
                  "input_files_sha256": inputs, "preparation_wall_seconds": time.perf_counter() - started,
                  "official_test_loaded": False, "validation_outcomes_read": False, "bpp_receipts_reused": False,
                  "timing_interpretation": "Inherited non-BPP committed fold compute is already included once in final CV totals; do not add it again. New invocation wall time excludes prior runs. Parent invocation/pause evidence includes interrupted execution where recorded, and is separate from method compute. Old BPP work is historical repair overhead, excluded from repaired BPP method cost. Uncommitted interrupted fold compute is unknown. Preparation performs validation/copy/preflight only, not model fitting.",
                  **evidence}
        _write_receipt(output / "recovery-manifest.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fashion-protocol", type=Path, required=True)
    parser.add_argument("--celeba-protocol", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    result = prepare(args.parent_run, args.output, args.fashion_protocol, args.celeba_protocol, args.reason)
    print(json.dumps({"state": result["state"], "counts": result["counts"], "bpp_receipts_reused": False,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
