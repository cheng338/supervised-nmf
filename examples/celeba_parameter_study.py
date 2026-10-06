"""Explicit CelebA parameter comparisons for Chang Cheng's classification package.

This executes a declared grid, with representation reuse across KNN neighbor
counts within each training fold. It supplies no automatic search policy or
package-default decision. All reported performance is development validation;
the official test partition is never loaded. Refine the declared grids in new
runs after examining training CV evidence. Keep every earlier run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
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
    _diagnostics, _digest, _nonconverged, _warning_records, comparison_models,
    data_fingerprint, evaluate_classifier, new_run_directory, provenance,
    refinement_diagnostics, save_data_manifest, write_json,
)
from supervised_nmf.study import calibration_knn_grid

ATTRIBUTES = ("Smiling", "Wearing_Necktie")


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


def _worker(job):
    cache, output, attribute, name, chunk_id, grid, settings, seed, threads, script_hash = job
    if _source_hash() != script_hash:
        raise ValueError("Experiment script changed during this run.")
    X = np.load(Path(cache) / "train-X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(Path(cache) / f"train-{attribute}-y.npy", allow_pickle=False)
    estimator = study_models(settings, seed)[name][0]
    report = calibration_knn_grid(estimator, grid, X, y, cv=5, seed=seed, threads=threads,
                                  checkpoint_dir=Path(output) / attribute / name / chunk_id)
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


def aggregate_and_evaluate(output, cache, models, manifest, threads, configuration):
    summaries = {}
    X = np.load(Path(cache) / "train-X.npy", mmap_mode="r", allow_pickle=False)
    Xv = np.load(Path(cache) / "validation-X.npy", mmap_mode="r", allow_pickle=False)
    for attribute in ATTRIBUTES:
        y = np.load(Path(cache) / f"train-{attribute}-y.npy", allow_pickle=False)
        yv = np.load(Path(cache) / f"validation-{attribute}-y.npy", allow_pickle=False)
        results = {}
        for name, (estimator, grid) in models.items():
            if (_source_hash() != configuration["script_sha256"]
                    or provenance(threads)["source_sha256"] != configuration["source_sha256"]):
                raise ValueError("Source changed before final fitting; preserve this run and start a new one.")
            directory = Path(output) / attribute / name
            candidates, compute = [], 0.0
            for chunk in sorted(directory.glob("chunk-*/report.json")):
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
            if parameters is None:
                result["evaluation"] = {"status": "held", "reason": "No candidate passed all five numerical checks."}
            else:
                fitted = clone(estimator).set_params(**parameters)
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
                            result["evaluation"] = evaluate_classifier(fitted, Xv, yv, threads=threads,
                                sample_ids=manifest["splits"]["validation"]["image_ids"])
                    except Exception as error:
                        result["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
                    finally:
                        if result["final_fit_seconds"] == 0.0:
                            result["final_fit_seconds"] = time.perf_counter() - start
                result["final_fit_warnings"] = _warning_records(caught)
            write_json(result_path, {"payload": result, "sha256": _digest(result)})
            results[name] = result
            print(attribute, name, result["evaluation"]["status"], result["evaluation"].get("metrics", {}), flush=True)
        summaries[attribute] = {"models": results, "training_class_counts": np.bincount(y).tolist(),
                                "validation_class_counts": np.bincount(yv).tolist()}
    return summaries


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
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    if not (args.data_cache / "manifest.json").exists() and not all((args.dir, args.attributes, args.partitions)):
        parser.error("Creating a cache requires --dir, --attributes and --partitions")
    settings = json.loads(args.grid_json.read_text(encoding="utf-8"))
    manifest = prepare_cache(args.data_cache, args.dir, args.attributes, args.partitions,
                              args.max_train, args.max_validation, args.seed)
    models = study_models(settings, args.seed)
    config = {"grid": settings, "seed": args.seed, "workers": args.workers, "threads": args.threads,
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
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
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
