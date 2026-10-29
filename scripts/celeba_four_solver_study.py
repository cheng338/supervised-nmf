"""Matched four-solver CelebA extension. Author: Chang Cheng.

Reuse the Fashion-MNIST study's explicit-grid, checkpoint, solver, numerical and
profiling utilities. CV opens only the existing official-training arrays. The
separate evaluation command verifies the official-validation arrays after every
declared CV chunk is terminal. No official-test files are accessed.

The parent orchestration completes both datasets' CV before either evaluation.
Historical CelebA runs are input provenance only and are never modified here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from numbers import Real
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import fashion_mnist_study as shared
from supervised_nmf.evaluation import _digest, data_fingerprint, provenance, write_json
from supervised_nmf.study import _checkpoint_lock, _read_receipt, _write_receipt

METHODS = ("screened_cd_absolute", "screened_spa_absolute", "screened_mu_absolute", "screened_anls_bpp_absolute")
ATTRIBUTES = ("Smiling", "Wearing_Necktie")
N_FEATURES = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol):
    _require(protocol.get("schema_version") == 1 and protocol.get("author") == "Chang Cheng",
             "Protocol requires schema 1 and Chang Cheng authorship.")
    _require(protocol.get("cv") == 5 and protocol.get("native_threads") == 1,
             "Protocol requires five folds and one native thread.")
    _require(protocol.get("screening", "absolute") == "absolute", "Absolute screening is fixed.")
    _require(protocol.get("attributes") == list(ATTRIBUTES), "Declare both CelebA attributes in canonical order.")
    _require(protocol.get("methods") == list(METHODS), "Declare all four absolute-screening solvers in canonical order.")
    shared._integer(protocol["seed"], "seed", 0, 2**32 - 1)
    shared._integer(protocol["n_train"], "n_train", 10)
    shared._integer(protocol["n_validation"], "n_validation", 2)
    shared._integer(protocol.get("workers", 8), "workers", 1, 8)
    shared._integer(protocol["max_iter"], "max_iter")
    shared._integer(protocol["anls_max_iter"], "anls_max_iter")
    if protocol["anls_nnls_max_iter"] is not None:
        shared._integer(protocol["anls_nnls_max_iter"], "anls_nnls_max_iter")
    _require(isinstance(protocol["anls_nnls_tol"], Real) and not isinstance(protocol["anls_nnls_tol"], bool)
             and np.isfinite(protocol["anls_nnls_tol"]) and protocol["anls_nnls_tol"] > 0,
             "ANLS inner NNLS tolerance must be finite and positive.")
    shared._integer(protocol.get("profile_order_seed", 42043), "profile_order_seed", 0, 2**32 - 1)
    _require(protocol.get("profile_repeats") == 3, "Three profiling repeats are required.")
    _require(isinstance(protocol["tol"], Real) and not isinstance(protocol["tol"], bool)
             and np.isfinite(protocol["tol"]) and protocol["tol"] >= 0, "Invalid numerical tolerance.")
    _require(set(protocol["grid"]) == {"n_components", "score_threshold", "n_neighbors"}, "Invalid grid keys.")
    for key, values in protocol["grid"].items():
        _require(isinstance(values, list) and values and len(set(values)) == len(values), "Grid values must be nonempty and distinct.")
        for value in values:
            if key == "score_threshold":
                _require(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value) and value >= 0,
                         "Absolute thresholds must be finite and nonnegative.")
            else:
                shared._integer(value, key)
    extras = protocol.get("extra_representations", {})
    _require(isinstance(extras, dict) and set(extras) <= set(ATTRIBUTES), "Unknown extra-representation attribute.")
    for values in extras.values():
        _require(isinstance(values, list), "Extra representations must be a list.")
        for pair in values:
            _require(set(pair) == {"n_components", "score_threshold"}, "Extra representations must supply exactly rank and cutoff.")
            shared._integer(pair["n_components"], "extra rank")
            value = pair["score_threshold"]
            _require(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value) and value >= 0,
                     "Extra absolute thresholds must be finite and nonnegative.")
    _require(isinstance(protocol.get("cache_manifest_sha256"), str) and len(protocol["cache_manifest_sha256"]) == 64,
             "Pin the existing CelebA cache manifest SHA-256.")


def plan_with_extras(job, grid, extras, preflight):
    """Same explicit representations for every solver; retain canonical ties."""
    boxes = [grid, *[{"n_components": [row["n_components"]], "score_threshold": [row["score_threshold"]],
                     "n_neighbors": grid["n_neighbors"]} for row in extras]]
    result = {"job": job, "declared_candidates": [], "candidates": [], "chunks": [],
              "excluded_representations": [], "fold_indices_sha256": preflight["fold_indices_sha256"]}
    canonical, feasible_seen, representation_seen, excluded_seen = {}, set(), set(), set()
    for box in boxes:
        piece = shared._plan(job, box, preflight)
        indices = {}
        for old_index, point in enumerate(piece["declared_candidates"]):
            fingerprint = _digest(point)
            if fingerprint not in canonical:
                canonical[fingerprint] = len(result["declared_candidates"])
                result["declared_candidates"].append(point)
            indices[old_index] = canonical[fingerprint]
        for row in piece["candidates"]:
            index = indices[row["index"]]
            if index not in feasible_seen:
                result["candidates"].append({**row, "index": index})
                feasible_seen.add(index)
        for chunk in piece["chunks"]:
            mapped = [indices[index] for index in chunk["candidate_indices"]]
            key = tuple(mapped)
            if key not in representation_seen:
                result["chunks"].append({**chunk, "id": f"c{len(result['chunks']):04d}", "candidate_indices": mapped})
                representation_seen.add(key)
        for excluded in piece["excluded_representations"]:
            mapped = [indices[index] for index in excluded["candidate_indices"]]
            key = tuple(mapped)
            if key not in excluded_seen:
                result["excluded_representations"].append({**excluded, "candidate_indices": mapped})
                excluded_seen.add(key)
    result["candidates"].sort(key=lambda row: row["index"])
    return result


def _cache_manifest(protocol):
    cache = Path(protocol["cache"])
    cache = (ROOT / cache).resolve() if not cache.is_absolute() else cache.resolve()
    path = cache / "manifest.json"
    _require(shared._file_hash(path) == protocol["cache_manifest_sha256"], "CelebA manifest checksum changed.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("official_test_loaded") is False, "Cache must exclude official test data.")
    _require(manifest["seed"] == protocol["seed"], "Cache training seed differs.")
    for split, partition in (("train", 0), ("validation", 1)):
        record = manifest["splits"][split]
        ids = record["image_ids"]
        _require(ids and len(ids) == len(set(ids)) and record["metadata"]["partition_code"] == partition,
                 "Image IDs or official partition code are invalid.")
        _require(set(ATTRIBUTES) <= set(record["attributes"]), "Cache omits a required attribute.")
    train_ids = manifest["splits"]["train"]["image_ids"]
    validation_ids = manifest["splits"]["validation"]["image_ids"]
    _require(len(train_ids) == protocol["n_train"], "Training count differs from protocol.")
    _require(not set(train_ids) & set(validation_ids), "Training and validation IDs overlap.")
    _require(len(validation_ids) == protocol["n_validation"], "Validation count differs from protocol.")
    return cache, manifest


def _array_record(cache, manifest, split, attribute):
    names = {"X": f"{split}-X.npy", "y": f"{split}-{attribute}-y.npy"}
    description = manifest["splits"][split]
    evidence = description["attributes"][attribute]
    return {"path": str(cache), "filenames": names, "data_sha256": evidence["data_sha256"],
            "files_sha256": {name: manifest["array_sha256"][name] for name in names.values()},
            "shape": [len(description["image_ids"]), N_FEATURES], "class_counts": evidence["class_counts"]}


def _verify_values(record, *, training):
    shared._verify_arrays(record)
    X, y = shared._load_arrays(record)
    _require(list(X.shape) == record["shape"] and X.dtype == np.float32 and np.isfinite(X).all()
             and X.min() >= 0 and X.max() <= 1, "Expected finite original 64x64 float32 pixels in [0, 1].")
    _require(y.shape == (len(X),) and y.dtype.kind in "iu" and np.all((y == 0) | (y == 1)), "Invalid binary label array.")
    counts = np.bincount(y, minlength=2).tolist()
    _require(counts == record["class_counts"] and data_fingerprint(X, y) == record["data_sha256"],
             "Array fingerprint or class counts differ from manifest.")
    if training:
        _require(min(counts) >= 5, "Each training class needs at least five rows.")
    return X, y


def initialize(protocol_path, output):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    cache, manifest = _cache_manifest(protocol)
    sources = shared._sources(extra_paths=[Path(__file__)])
    header = {"schema_version": 1, "author": "Chang Cheng", "protocol": protocol,
              "protocol_file_sha256": shared._file_hash(protocol_path), "sources": sources,
              "dataset": {"name": "CelebA", "cache": str(cache), "manifest": manifest,
                          "manifest_file_sha256": protocol["cache_manifest_sha256"]},
              "official_test_loaded": False, "evaluation_role": "official_celeba_validation_natural_class_proportions",
              "old_cd_spa_runs_role": "historical context only; not mixed with this matched four-solver contribution"}
    arrays, splits, preflights, plans = {}, {}, {}, []
    start = time.perf_counter()
    for attribute in ATTRIBUTES:
        pair_id = attribute.lower()
        key = f"{pair_id}/n{protocol['n_train']}-seed{protocol['seed']}"
        arrays[key] = _array_record(cache, manifest, "train", attribute)
        arrays[f"{pair_id}/validation"] = _array_record(cache, manifest, "validation", attribute)
        X, y = _verify_values(arrays[key], training=True)
        extras = protocol.get("extra_representations", {}).get(attribute, [])
        thresholds = list(protocol["grid"]["score_threshold"])
        thresholds.extend(row["score_threshold"] for row in extras if row["score_threshold"] not in thresholds)
        preflights[key] = shared._preflight(X, y, protocol["seed"], thresholds)
        splits[pair_id] = {"attribute": attribute, "training_indices": manifest["splits"]["train"]["image_ids"],
                           "validation_indices": manifest["splits"]["validation"]["image_ids"],
                           "source_partition_codes": {"train": 0, "validation": 1}}
        for method in METHODS:
            job = {"id": f"{pair_id}-n{protocol['n_train']}-seed{protocol['seed']}-{method}",
                   "attribute": attribute, "pair": {"id": pair_id, "negative": 0, "positive": 1},
                   "seed": protocol["seed"], "method": method, "n_train": protocol["n_train"],
                   "array_key": key, "cache": str(cache), "training_array_record": arrays[key],
                   "max_iter": protocol["max_iter"], "tol": protocol["tol"],
                   "anls_max_iter": protocol["anls_max_iter"], "anls_nnls_max_iter": protocol["anls_nnls_max_iter"],
                   "anls_nnls_tol": protocol["anls_nnls_tol"],
                   "training_data_sha256": arrays[key]["data_sha256"]}
            plans.append(plan_with_extras(job, protocol["grid"], extras, preflights[key]))
    order = np.random.RandomState(protocol.get("profile_order_seed", 42043)).permutation(len(plans))
    configuration = {**header, "arrays": arrays, "splits": splits, "preflight": preflights, "plans": plans,
                     "evaluation_order": [plans[index]["job"]["id"] for index in order]}
    shared._check_source(sources)
    path = output / "configuration.json"
    if path.exists():
        _require(_read_receipt(path) == configuration, "Resume configuration differs in protocol, data, source, folds or runtime.")
    else:
        _require(not any(output.glob("*/c*/identity.json")) and not (output / "comparison.json").exists(),
                 "Unidentified old results exist; choose a new directory.")
        output.mkdir(parents=True, exist_ok=True)
        _write_receipt(path, configuration)
        _write_receipt(output / "preflight.json", {"author": "Chang Cheng", "configuration_sha256": _digest(configuration),
                       "elapsed_seconds": time.perf_counter() - start, "training_folds_only": True,
                       "records": preflights, "excluded": {p["job"]["id"]: p["excluded_representations"] for p in plans}})
        write_json(output / "protocol.json", protocol)
        write_json(output / "provenance.json", provenance(1))
    return configuration


def evaluate(output, configuration):
    # No validation array is opened until the complete local CV gate passes.
    shared.aggregate_cv(output, configuration, require_complete=True)
    for key, record in configuration["arrays"].items():
        if key.endswith("/validation"):
            _verify_values(record, training=False)
    return shared.evaluate(output, configuration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cv", "evaluate"), default="cv")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(args.output):
        configuration = initialize(args.protocol, args.output)
        path = args.output / "executions" / f"{uuid.uuid4().hex}.json"
        start = time.perf_counter()
        attempt = {"author": "Chang Cheng", "started_at": datetime.now(timezone.utc).isoformat(),
                   "mode": args.mode, "state": "running", "configuration_sha256": _digest(configuration),
                   "native_threads": 1, "workers": 1 if args.mode == "evaluate" else
                   (args.workers or configuration["protocol"].get("workers", 8)), "official_test_loaded": False}
        _write_receipt(path, attempt)
        try:
            attempt[args.mode] = (shared.run_cv(args.output, configuration, workers=args.workers)
                                 if args.mode == "cv" else evaluate(args.output, configuration))
            attempt["state"] = "completed"
        except BaseException as error:
            attempt.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            attempt.update(current_call_wall_seconds=time.perf_counter() - start,
                           finished_at=datetime.now(timezone.utc).isoformat())
            _write_receipt(path, attempt)
        print(json.dumps({"state": attempt["state"], args.mode: attempt[args.mode]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
