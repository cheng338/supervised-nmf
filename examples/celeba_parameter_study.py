"""Explicit CelebA parameter comparisons for Chang Cheng's classification package.

This executes a declared grid, with representation reuse across KNN neighbor
counts within each training fold. It supplies no automatic search policy or
package-default decision. All reported performance is development validation;
the official test partition is never loaded. Refine the declared grids in new
runs after examining training CV evidence. Keep every earlier run.

Completed results are reused by default. Optional fitted-model recovery is
restricted to your own trusted local study and uses joblib/pickle state.
It recovers a committed fit before evaluation, not an interrupted solver.
"""

from __future__ import annotations


import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervised_nmf.datasets import _attribute_labels, load_celeba
from supervised_nmf.evaluation import (
    _diagnostics, _digest, _file_lock, _nonconverged, _warning_records, comparison_models,
    data_fingerprint, evaluate_classifier, new_run_directory, provenance,
    refinement_diagnostics, save_data_manifest, write_json,
)
from supervised_nmf._completed_reuse import _try_completed
from supervised_nmf.study import calibration_knn_grid, _reuse_calibration_knn_grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_store
if (getattr(recovery_store, "__file__", None) is None
        or Path(recovery_store.__file__).resolve()
        != Path(__file__).resolve().with_name("recovery_store.py").resolve()):
    raise ImportError("The study requires its sibling recovery_store.py; "
                      "restart with the intended checkout on the import path.")
if getattr(recovery_store, "_SOURCE_SHA256_AT_IMPORT", None) is None:
    raise ImportError("recovery_store.py lacks an import source pin; "
                      "restart the interpreter before continuing this study.")

ATTRIBUTES = ("Smiling", "Wearing_Necktie")
_INITIALIZATION_DEFAULT_HELPER = recovery_store.active_runtime_identity


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prepare_cache(directory, images, attributes, partitions, n_train, n_validation, seed):
    """Cache the original float32 preprocessing and both labels for identical IDs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        mapping = {name: _attribute_labels(Path(attributes), name) for name in ATTRIBUTES}
        manifest = {"author": "Chang Cheng", "seed": seed, "official_test_loaded": False,
                    "identity_overlap_checked": False, "splits": {}}
        for split, size in (("train", n_train), ("validation", n_validation)):
            data = load_celeba(images, attributes, partitions, attribute="Smiling", split=split,
                               max_samples=size, random_state=seed, image_size=(64, 64))
            np.save(directory / f"{split}-X.npy", data.data, allow_pickle=False)
            record = {"metadata": data.metadata, "image_ids": data.image_ids.tolist(), "attributes": {}}
            for attribute in ATTRIBUTES:
                y = np.array([mapping[attribute][name] for name in data.image_ids], dtype=np.int64)
                np.save(directory / f"{split}-{attribute}-y.npy", y, allow_pickle=False)
                record["attributes"][attribute] = {
                    "class_counts": np.bincount(y, minlength=2).tolist(),
                    "data_sha256": data_fingerprint(data.data, y)}
            manifest["splits"][split] = record
        write_json(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["seed"] != seed or manifest["official_test_loaded"]:
        raise ValueError("Incompatible cache seed or test access.")
    for split, count in (("train", n_train), ("validation", n_validation)):
        record = manifest["splits"][split]
        if len(record["image_ids"]) != count or record["metadata"]["partition_code"] != (0 if split == "train" else 1):
            raise ValueError("Cached sample count or official partition differs.")
        X = np.load(directory / f"{split}-X.npy", mmap_mode="r", allow_pickle=False)
        if X.shape != (count, 4096) or X.dtype != np.float32:
            raise ValueError("Cached preprocessing differs from the 64x64 float32 representation.")
        for attribute in ATTRIBUTES:
            y = np.load(directory / f"{split}-{attribute}-y.npy", allow_pickle=False)
            if data_fingerprint(X, y) != record["attributes"][attribute]["data_sha256"]:
                raise ValueError("Cached data failed their fingerprint check.")
    if set(manifest["splits"]["train"]["image_ids"]) & set(manifest["splits"]["validation"]["image_ids"]):
        raise ValueError("Training and validation image IDs overlap.")
    return manifest


def study_models(settings, seed):
    models = comparison_models(random_state=seed, max_iter=settings["max_iter"], tol=settings["tol"])
    for name, (estimator, grid) in models.items():
        models[name] = (estimator, {parameter: settings[parameter.rsplit("__", 1)[-1]] for parameter in grid})
    return models


def chunks_for_models(models):
    """Partition an explicit grid by screening value for balanced worker jobs."""
    chunks = []
    for name, (_, grid) in models.items():
        if name == "majority_class":
            continue
        screening = next((key for key in grid if key.rsplit("__", 1)[-1] in {"percentile", "score_threshold"}), None)
        values = grid[screening] if screening else [None]
        for index, value in enumerate(values):
            chunk = deepcopy(grid)
            if screening:
                chunk[screening] = [value]
            chunks.append((name, f"chunk-{index:03d}", chunk))
    return chunks


def _worker(job, *, reuse_only=False, _reuse_environment=None):
    cache, output, attribute, name, chunk_id, grid, settings, seed, threads, script_hash = job
    if _reuse_environment is not None:
        _reuse_environment(threads)
    if _source_hash() != script_hash:
        raise ValueError("Experiment script changed during this run.")
    X = np.load(Path(cache) / "train-X.npy", mmap_mode="r", allow_pickle=False)
    try:
        y = np.load(Path(cache) / f"train-{attribute}-y.npy", allow_pickle=False)
        estimator = study_models(settings, seed)[name][0]
        calibrate = _reuse_calibration_knn_grid if reuse_only else calibration_knn_grid
        options = {} if _reuse_environment is None else {"_reuse_environment": _reuse_environment}
        report = calibrate(estimator, grid, X, y, cv=5, seed=seed, threads=threads,
                                      scoring="balanced_accuracy",
                                      checkpoint_dir=Path(output) / attribute / name / chunk_id, **options)
    finally:
        # Linux completed reuse owns this mapping, including on errors.
        if sys.platform == "linux":
            X._mmap.close()
    if report is None:
        return None
    return {"attribute": attribute, "method": name, "chunk": chunk_id,
            "candidates": len(report["candidates"]),
            "valid_candidates": sum(item["valid"] for item in report["candidates"]),
            "unique_representation_fits": report["unique_representation_fits"],
            "tuning_compute_seconds": report["tuning_seconds"], "worker_pid": os.getpid()}


def _read_report(path):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if envelope["sha256"] != _digest(envelope["payload"]):
        raise ValueError(f"Invalid report checksum: {path}")
    return envelope["payload"]


def _declared_reports(output, models):
    declared = {name: [] for name in models}
    for name, chunk_id, _ in chunks_for_models(models):
        declared[name].append(chunk_id)
    paths = {}
    for attribute in ATTRIBUTES:
        for name in models:
            directory = Path(output) / attribute / name
            expected = [directory / chunk / "report.json" for chunk in declared[name]]
            missing = [str(path) for path in expected if not path.is_file()]
            extra = sorted(set(directory.glob("chunk-*/report.json")) - set(expected))
            if extra:
                raise ValueError("Unexpected CV reports; preserve and review: "
                                 + ", ".join(str(path) for path in extra))
            if missing:
                raise ValueError("Missing declared CV reports: " + ", ".join(missing))
            paths[attribute, name] = expected
    return paths


def _package_source_hashes():
    """Exactly provenance's source inventory, without unrelated runtime probes."""
    from supervised_nmf import evaluation as evaluation_module
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(evaluation_module.__file__).parent.glob("*.py"))}


def _ordinary_final_fit(result, estimator, parameters, X, y, Xv, yv, manifest, threads, details):
    """Existing behavior with no-op boundaries and no recovery fields."""
    if parameters is None:
        result["evaluation"] = {"status": "held", "reason": "No candidate passed all five numerical checks."}
        return result
    fitted = clone(estimator).set_params(**parameters)
    recovery_store._hook("fit-start", details)
    start = time.perf_counter()
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted.fit(X, y)
            result["final_fit_seconds"] = time.perf_counter() - start
            result["diagnostics"] = _diagnostics(fitted)
            recovery_store._hook("fit-complete", details)
            if _nonconverged(fitted, caught):
                result["evaluation"] = {"status": "held", "reason": "Selected final fit did not converge."}
            else:
                recovery_store._hook("evaluation-start", details)
                result["evaluation"] = evaluate_classifier(fitted, Xv, yv, threads=threads,
                    sample_ids=manifest["splits"]["validation"]["image_ids"])
                recovery_store._hook("evaluation-complete", details)
        except Exception as error:
            result["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        finally:
            if result["final_fit_seconds"] == 0.0:
                result["final_fit_seconds"] = time.perf_counter() - start
    result["final_fit_warnings"] = _warning_records(caught)
    return result


def _publish_final_result(result_path, result, details):
    recovery_store._hook("before-result-commit", details)
    write_json(result_path, {"payload": result, "sha256": _digest(result)})
    recovery_store._hook("result-commit", details)


def _recoverable_final_fit(result, directory, result_path, estimator, parameters, X, y, Xv, yv,
                           manifest, threads, configuration, runtime_cache):
    policy = recovery_store.validate_policy(configuration.get("final_fit_recovery", "off"))
    details = {"attribute": result["attribute"], "method": result["method"]}
    if policy == "off":
        result = _ordinary_final_fit(result, estimator, parameters, X, y, Xv, yv, manifest, threads, details)
        _publish_final_result(result_path, result, details)
        return result
    initialization = runtime_cache.setdefault("initialization_cache", {})
    base_identity = {"evaluation_identity": result["evaluation_identity"],
                     "run_identity": result["run_identity"], **details}
    codec_context = None
    if result["method"] == "raw_knn":
        # main has fingerprint-validated this cache. Keep the original ordered
        # observations scoped to this call, without a second matrix hash per model.
        cache_X, cache_y = np.asarray(X).view(), np.asarray(y).view()
        cache_X.flags.writeable = False
        cache_y.flags.writeable = False
        codec_context = {"X": cache_X, "y": cache_y, "identity": {
            **base_identity, "manifest_sha256": configuration["data_manifest_sha256"],
            "training_data_sha256": manifest["splits"]["train"]["attributes"][result["attribute"]]["data_sha256"]}}
    accounting = {"policy": policy, "fit_reused": False, "outcome_reused": False, "new_fit_seconds": 0.0,
                  "fit_performed": False, "original_fit_performed": parameters is not None,
                  "scope": "invocation that committed final result; completed-result reuse returns historical record",
                  "identity_seconds": 0.0,
                  "checkpoint_seconds": 0.0, "verification_seconds": 0.0,
                  "model_load_seconds": 0.0, "head_reconstruction_seconds": 0.0,
                  "reconstructed_knn_heads": 0, "cache_reference_restored": False,
                  "cache_binding_seconds": 0.0, "model_bytes": 0}
    def verify_identity():
        identity_start = time.perf_counter()
        try:
            if (_source_hash() != configuration["script_sha256"]
                    or _package_source_hashes() != configuration["source_sha256"]):
                raise recovery_store.RecoveryError("Source or active runtime changed during final fitting/evaluation")
            if recovery_store.active_runtime_identity is _INITIALIZATION_DEFAULT_HELPER:
                observed = recovery_store.active_runtime_identity(
                    threads, file_cache=runtime_cache.setdefault("file_cache", {}),
                    _initialization_cache=initialization)
            else:
                initialization.clear()
                observed = recovery_store.active_runtime_identity(
                    threads, file_cache=runtime_cache.setdefault("file_cache", {}))
            recovery_store.validate_loaded_sources(observed["recovery_sources"])
            if "identity" not in runtime_cache:
                runtime_cache["identity"] = observed
            elif observed != runtime_cache["identity"]:
                raise recovery_store.RecoveryError("Source or active runtime changed during final fitting/evaluation")
        except BaseException:
            initialization.clear()
            raise
        finally:
            accounting["identity_seconds"] += time.perf_counter() - identity_start

    # Refresh each enabled method before outcome reads, including methods after
    # an earlier result populated the invocation-local native-file hash cache.
    verify_identity()
    store = recovery_store.RecoveryStore(directory / ".fit-recovery",
        lambda: {**base_identity, "runtime": runtime_cache["identity"]},
        policy=policy, attempt_identity=base_identity)
    outcome = store.load_outcome()
    attempt = store.start_attempt("recover" if outcome else "fit")
    fitted, fit_ready, committed = None, False, outcome is not None
    fit_metadata = None
    try:
        if outcome is not None:
            stored = outcome["manifest"]["metadata"]
            if any(stored.get(key) != result.get(key) for key in
                   ("method", "attribute", "run_identity", "evaluation_identity", "selected_parameters")):
                raise recovery_store.RecoveryError("Committed fit metadata differs from selected final fit")
            result = deepcopy(stored)
            fit_metadata = deepcopy(stored)
            accounting["outcome_reused"] = True
            accounting["fit_reused"] = outcome["manifest"]["kind"] == "fitted"
            accounting["verification_seconds"] = outcome["verify_seconds"]
            accounting["generation"] = outcome["generation"].name
            accounting["model_bytes"] = sum(item["bytes"] for item in outcome["manifest"]["files"].values())
            if outcome["manifest"]["kind"] == "fitted":
                # Recheck immediately before the owned model loader, after the
                # potentially lengthy committed-payload integrity checks.
                verify_identity()
                with threadpool_limits(limits=threads):
                    fitted, accounting["model_load_seconds"], stats = store.restore_model(
                        outcome, codec_context=codec_context)
                accounting["head_reconstruction_seconds"] = stats.get("reconstruction_seconds", 0.0)
                accounting["reconstructed_knn_heads"] = stats.get("reconstructed_knn_heads", 0)
                accounting["cache_reference_restored"] = stats.get("cache_reference_restored", False)
                accounting["cache_binding_seconds"] = stats.get("cache_binding_seconds", 0.0)
                fit_ready = True
            store.record(attempt, "recovered", accounting)
        elif parameters is None:
            result["evaluation"] = {"status": "held", "reason": "No candidate passed all five numerical checks."}
        else:
            fitted = clone(estimator).set_params(**parameters)
            accounting["fit_performed"] = True
            recovery_store._hook("fit-start", details)
            start = time.perf_counter()
            with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fitted.fit(X, y)
                    result["final_fit_seconds"] = time.perf_counter() - start
                    result["diagnostics"] = _diagnostics(fitted)
                    if _nonconverged(fitted, caught):
                        result["evaluation"] = {"status": "held", "reason": "Selected final fit did not converge."}
                    else:
                        fit_ready = True
                except Exception as error:
                    initialization.clear()
                    result["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
                finally:
                    if result["final_fit_seconds"] == 0.0:
                        result["final_fit_seconds"] = time.perf_counter() - start
            result["final_fit_warnings"] = _warning_records(caught)
            accounting["new_fit_seconds"] = result["final_fit_seconds"]
            fit_metadata = deepcopy(result)
            store.record(attempt, "fit-complete", result)
            recovery_store._hook("fit-complete", details)

        # Persistence exceptions are outside numerical fit/evaluation handlers.
        if not committed:
            verify_identity()
            published = store.commit(fit_metadata or result, model=fitted if fit_ready else None,
                                     details=details, codec_context=codec_context)
            accounting.update(published)
            committed = True
            store.record(attempt, "outcome-committed", published)
        if fit_ready:
            recovery_store._hook("evaluation-start", details)
            with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result["evaluation"] = evaluate_classifier(fitted, Xv, yv, threads=threads,
                        sample_ids=manifest["splits"]["validation"]["image_ids"])
                except Exception as error:
                    initialization.clear()
                    result["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            result["final_fit_warnings"] = result.get("final_fit_warnings", []) + _warning_records(caught)
            evaluation = result["evaluation"]
            store.record(attempt, "evaluation-complete", {
                "status": evaluation["status"],
                "prediction_seconds": evaluation.get("prediction_seconds"),
                "prediction_count": len(evaluation.get("predictions", {}).get("predicted_labels", [])),
                "warning_count": len(evaluation.get("warnings", [])),
                "error_type": evaluation.get("error_type")})
            recovery_store._hook("evaluation-complete", details)
        verify_identity()
        result["recovery"] = accounting
        store.record(attempt, "result-ready", {
            "status": result["evaluation"]["status"], "path": result_path.name})
        _publish_final_result(result_path, result, details)
        store.record(attempt, "result-committed", {"path": result_path.name})
        return result
    except BaseException as error:
        initialization.clear()
        try:
            store.record(attempt, "interrupted", {"error_type": type(error).__name__, "error": str(error)})
        except BaseException as evidence_error:
            error.add_note(f"Could not record interruption: {type(evidence_error).__name__}: {evidence_error}")
        raise


def aggregate_and_evaluate(output, cache, models, manifest, threads, configuration):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with _file_lock(output / ".final-evaluation.lock",
                    f"Another process or thread is evaluating this study: {output}"):
        with ExitStack() as final_guards:
            guard_entered = False
            def ensure_final_guard():
                nonlocal guard_entered
                if not guard_entered:
                    waited = final_guards.enter_context(recovery_store._final_aggregation_guard())
                    guard_entered = True
                    if waited and configuration.get("final_fit_recovery", "off") == "off":
                        if (_source_hash() != configuration["script_sha256"]
                                or _package_source_hashes() != configuration["source_sha256"]):
                            raise ValueError("Source changed before final fitting; preserve this run and start a new one.")
            return _aggregate_and_evaluate(output, cache, models, manifest, threads, configuration,
                                           _ensure_final_guard=ensure_final_guard)


def _aggregate_and_evaluate(output, cache, models, manifest, threads, configuration, *,
                            _ensure_final_guard=None):
    recovery_store.validate_policy(configuration.get('final_fit_recovery', 'off'))
    runtime_cache = {"file_cache": {}}
    reports = _declared_reports(output, models)
    summaries = {}
    X = np.load(Path(cache) / "train-X.npy", mmap_mode="r", allow_pickle=False)
    Xv = np.load(Path(cache) / "validation-X.npy", mmap_mode="r", allow_pickle=False)
    for attribute in ATTRIBUTES:
        y = np.load(Path(cache) / f"train-{attribute}-y.npy", allow_pickle=False)
        yv = np.load(Path(cache) / f"validation-{attribute}-y.npy", allow_pickle=False)
        results = {}
        for name, (estimator, grid) in models.items():
            if (_source_hash() != configuration["script_sha256"]
                    or _package_source_hashes() != configuration["source_sha256"]):
                raise ValueError("Source changed before final fitting; preserve this run and start a new one.")
            directory = Path(output) / attribute / name
            candidates, compute = [], 0.0
            for chunk in reports[attribute, name]:
                report = _read_report(chunk)
                if report["identity"]["calibration"]["data_sha256"] != manifest["splits"]["train"]["attributes"][attribute]["data_sha256"]:
                    raise ValueError("CV report belongs to different training data or attribute labels.")
                for candidate in report["candidates"]:
                    candidate["chunk"] = chunk.parent.name
                    candidates.append(candidate)
                compute += report["tuning_seconds"]
            canonical_indices = {_digest(params): index for index, params in enumerate(ParameterGrid(grid))}
            for candidate in candidates:
                candidate["index"] = canonical_indices[_digest(candidate["parameters"])]
            candidates.sort(key=lambda row: row["index"])
            if name != "majority_class" and [row["index"] for row in candidates] != list(range(len(canonical_indices))):
                raise ValueError("CV chunks do not cover the complete declared parameter grid exactly once.")
            valid = [row for row in candidates if row["valid"]]
            winner = min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])) if valid else None
            parameters = {} if name == "majority_class" else winner["parameters"] if winner else None
            calibration = {"candidates": candidates, "selected_parameters": parameters,
                           "cv_balanced_accuracy": winner["mean_balanced_accuracy"] if winner else None,
                           "tuning_compute_seconds": compute, "tuning_complete": False, "defaults_frozen": False,
                           "refinement": refinement_diagnostics([row["parameters"] for row in candidates], parameters)}
            write_json(directory / "calibration-summary.json", calibration)
            evaluation_identity = _digest({"run": configuration, "attribute": attribute,
                                           "method": name, "selected_candidate": winner})
            result_path = directory / "development-evaluation.json"
            if result_path.exists():
                result = _read_report(result_path)
                if (result["selected_parameters"] != parameters
                        or result["run_identity"] != _digest(configuration)
                        or result["evaluation_identity"] != evaluation_identity
                        or result["method"] != name or result["attribute"] != attribute):
                    raise ValueError("Saved development evaluation differs from the selected CV parameters.")
                results[name] = result
                continue
            result = {"method": name, "attribute": attribute, "run_identity": _digest(configuration),
                      "evaluation_identity": evaluation_identity,
                      "selected_parameters": parameters,
                      "cv_balanced_accuracy": calibration["cv_balanced_accuracy"],
                      "tuning_compute_seconds": compute, "final_fit_seconds": 0.0,
                      "tuning_complete": False, "defaults_frozen": False, "is_fallback": False,
                      "evaluation_role": "development_validation_not_final_test"}
            if _ensure_final_guard is not None:
                _ensure_final_guard()
            result = _recoverable_final_fit(result, directory, result_path, estimator, parameters,
                X, y, Xv, yv, manifest, threads, configuration, runtime_cache)
            results[name] = result
            print(attribute, name, result["evaluation"]["status"], result["evaluation"].get("metrics", {}), flush=True)
        summaries[attribute] = {"models": results, "training_class_counts": np.bincount(y).tolist(),
                                "validation_class_counts": np.bincount(yv).tolist()}
    return summaries


def _completed_or_spawned(jobs, workers):
    def describe(job):
        _, output, attribute, name, chunk_id, grid, settings, seed, threads, _ = job
        return study_models(settings, seed)[name][0], grid, Path(output) / attribute / name / chunk_id, threads

    def invoke(job, environment):
        return _worker(job, reuse_only=True, _reuse_environment=environment)

    completed = _try_completed(jobs, workers, describe=describe, invoke=invoke)
    if completed is not None:
        yield from completed
        return
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for future in as_completed(futures):
                yield future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--attributes", type=Path)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--data-cache", type=Path, required=True,
                        help="Existing cache takes precedence over image paths; values are fingerprint-verified.")
    parser.add_argument("--grid-json", type=Path, default=Path(__file__).with_name("celeba_starting_grid.json"))
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-validation", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--final-fit-recovery", choices=recovery_store.POLICIES, default="off",
                        help="off (default) reuses completed results only; all also saves fitted state before evaluation. "
                             "A supported raw KNN head references the same verified training cache without refitting. "
                             "Use all only for your own trusted local study: recovery loads joblib/pickle state. "
                             "Do not resume untrusted or received model files; integrity checks do not establish trust. "
                             "The choice is saved and must match on resume.")
    args = parser.parse_args()
    try:
        recovery_store.validate_policy(args.final_fit_recovery)
    except ValueError as error:
        parser.error(str(error))
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    if not (args.data_cache / "manifest.json").exists() and not all((args.dir, args.attributes, args.partitions)):
        parser.error("Creating a cache requires --dir, --attributes and --partitions")
    settings = json.loads(args.grid_json.read_text(encoding="utf-8"))
    manifest = prepare_cache(args.data_cache, args.dir, args.attributes, args.partitions,
                              args.max_train, args.max_validation, args.seed)
    models = study_models(settings, args.seed)
    config = {"final_fit_recovery": args.final_fit_recovery, "grid": settings, "seed": args.seed, "workers": args.workers, "threads": args.threads,
              "data_manifest_sha256": _digest(manifest), "script_sha256": _source_hash(),
              "source_sha256": provenance(args.threads)["source_sha256"], "cv": 5,
              "selection_metric": "balanced_accuracy", "ties": "first candidate in sklearn ParameterGrid order",
              "tuning_complete": False, "defaults_frozen": False}
    output = args.resume or new_run_directory(args.output, "celeba-parameter-study")
    save_data_manifest(output / "configuration.json", config)
    save_data_manifest(output / "data.json", manifest)
    if not args.resume:
        write_json(output / "provenance.json", provenance(args.threads))
    chunks = chunks_for_models(models)
    jobs = [(str(args.data_cache.resolve()), str(output.resolve()), attribute, name, chunk_id, grid,
             settings, args.seed, args.threads, config["script_sha256"])
            for name, chunk_id, grid in chunks for attribute in ATTRIBUTES]
    print(f"Run: {output.resolve()}\nExplicit training-CV chunks: {len(jobs)}; workers={args.workers}, threads={args.threads}", flush=True)
    start = time.perf_counter()
    completed = []
    for record in _completed_or_spawned(jobs, args.workers):
        completed.append(record)
        write_json(output / "progress.json", {"completed_chunks": len(completed), "total_chunks": len(jobs), "chunks": completed})
        print(f"{len(completed)}/{len(jobs)} {record['attribute']} {record['method']} {record['chunk']}: "
              f"{record['valid_candidates']}/{record['candidates']} valid", flush=True)
    calibration_wall_seconds = time.perf_counter() - start
    results = aggregate_and_evaluate(output, args.data_cache, models, manifest, args.threads, config)
    write_json(output / "comparison.json", {"author": "Chang Cheng", "attributes": results,
        "calibration_call_wall_seconds": calibration_wall_seconds,
        "tuning_complete": False, "defaults_frozen": False, "official_test_loaded": False,
        "timing_note": "Chunk compute times count shared fits once; simultaneous worker times are not wall time.",
        "replications": 1, "role": "explicit_grid_development_comparison"})
    print(f"Development comparison saved: {output.resolve()}\nGrid refinement, numerical sensitivity and replications remain required.", flush=True)


if __name__ == "__main__":
    main()
