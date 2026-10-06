"""Explicit, resumable Fashion-MNIST development benchmark by Chang Cheng.

Only the official training arrays are read. Five-fold calibration of each
representation is shared across its declared neighbor counts; all CV must finish
before serial development-validation fits and three exact-reproduction profiles.
Use --mode cv to inspect boundary flags, then --mode evaluate. A changed grid
requires a new protocol/output directory. No package default is selected here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
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
from sklearn.decomposition import NMF
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf.evaluation import (
    _diagnostics, _digest, _jsonable, _nonconverged, _warning_records,
    comparison_models, data_fingerprint, evaluate_classifier, provenance,
    refinement_diagnostics, write_json,
)
from supervised_nmf.screening import BinaryScreening
from supervised_nmf.study import (
    _checkpoint_lock, _read_receipt, _runtime_identity as _calibration_runtime,
    _write_receipt, calibration_knn_grid,
)
from profile_celeba_stages import _compare_predictions, _runtime_identity, _StageProfiler, profile_selected_estimator

EXPECTED_SHAPE = (60000, 784)
NMF_METHODS = ("screened_cd_absolute", "screened_spa_absolute")
EXPERIMENTAL_METHODS = ("screened_mu_absolute", "screened_anls_bpp_absolute")
BASELINE_METHODS = ("screened_pca_absolute", "screening_only_absolute", "raw_knn")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, label, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {label}: {value!r}.")
    return int(value)


def _sources(extra_paths=()):
    environment = provenance(1)
    paths = [Path(__file__), ROOT / "scripts" / "profile_celeba_stages.py"]
    experimental = ROOT / "scripts" / "experimental_nmf.py"
    if experimental.exists():
        paths.append(experimental)
    return {"package_source_sha256": environment["source_sha256"],
            "runner_source_sha256": {p.name: _file_hash(p) for p in paths},
            "extra_sources_sha256": {str(Path(p).resolve()): _file_hash(p) for p in extra_paths},
            "runtime": _runtime_identity(environment)}


def _check_source(expected):
    if _sources(expected.get("extra_sources_sha256", {})) != expected:
        raise ValueError("Source or numerical runtime changed; start a new run directory.")


def _validate_protocol(protocol):
    if (protocol.get("schema_version") != 1 or protocol.get("author") != "Chang Cheng"
            or protocol.get("cv") != 5 or protocol.get("native_threads") != 1):
        raise ValueError("Protocol requires schema 1, Chang Cheng authorship, five folds and native_threads=1.")
    _integer(protocol.get("workers", 8), "workers", 1, 8)
    methods = protocol.get("methods", list(NMF_METHODS))
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or not set(methods) <= set(NMF_METHODS + EXPERIMENTAL_METHODS)):
        raise ValueError("Methods must be distinct explicit absolute-screened NMF solvers.")
    if protocol.get("screening", "absolute") != "absolute" or protocol.get("training_coordinates", "fit_transform") != "fit_transform":
        raise ValueError("This benchmark uses absolute screening and original fit_transform training coordinates.")
    if protocol.get("baseline_methods", list(BASELINE_METHODS)) != list(BASELINE_METHODS):
        raise ValueError("The declared baseline set must contain PCA, screening-only, and raw KNN.")
    _integer(protocol.get("profile_repeats", protocol.get("profiles", 3)), "profile_repeats")
    _integer(protocol.get("profile_order_seed", 42043), "profile_order_seed", 0, 2**32 - 1)
    _integer(protocol["validation_seed"], "validation_seed", 0, 2**32 - 1)
    size = _integer(protocol["validation_size_per_pair"], "validation_size_per_pair", 2)
    if size % 2:
        raise ValueError("Validation size must be even for exact class balance.")
    for field in ("train_sizes", "seeds"):
        if not isinstance(protocol[field], list) or not protocol[field] or len(set(protocol[field])) != len(protocol[field]):
            raise ValueError(f"{field} must be nonempty and distinct.")
    for size in protocol["train_sizes"]:
        _integer(size, "training size", 10)
        if size % 2:
            raise ValueError("Training sizes must be even for exact class balance.")
    for seed in protocol["seeds"]:
        _integer(seed, "seed", 0, 2**32 - 1)
    _integer(protocol["max_iter"], "max_iter")
    _integer(protocol.get("anls_max_iter", 1000), "anls_max_iter")
    if protocol.get("anls_nnls_max_iter") is not None:
        _integer(protocol["anls_nnls_max_iter"], "anls_nnls_max_iter")
    inner_tolerance = protocol.get("anls_nnls_tol", 1e-10)
    if isinstance(inner_tolerance, bool) or not isinstance(inner_tolerance, Real) or not np.isfinite(inner_tolerance) or inner_tolerance <= 0:
        raise ValueError("anls_nnls_tol must be finite and positive.")
    if isinstance(protocol["tol"], bool) or not isinstance(protocol["tol"], Real) or not np.isfinite(protocol["tol"]) or protocol["tol"] < 0:
        raise ValueError("tol must be finite and nonnegative.")
    if set(protocol["grid"]) != {"n_components", "score_threshold", "n_neighbors"}:
        raise ValueError("Grid must explicitly contain ranks, absolute thresholds and neighbor counts.")
    for key, values in protocol["grid"].items():
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} grid must be nonempty and distinct.")
        for value in values:
            if key != "score_threshold":
                _integer(value, key)
            elif isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError("Absolute thresholds must be finite and nonnegative.")
    ids = []
    if not isinstance(protocol["pairs"], list) or not protocol["pairs"]:
        raise ValueError("Declare at least one class pair.")
    for pair in protocol["pairs"]:
        if not isinstance(pair["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", pair["id"]):
            raise ValueError("Pair IDs must be short path-safe names.")
        ids.append(pair["id"])
        for key in ("negative", "positive"):
            _integer(pair[key], key, 0, 9)
        if pair["negative"] == pair["positive"]:
            raise ValueError("Pair classes must differ.")
    if len(set(ids)) != len(ids):
        raise ValueError("Pair IDs must be unique.")
    if (protocol.get("baseline_train_size", max(protocol["train_sizes"])) not in protocol["train_sizes"]
            or protocol.get("baseline_seed", protocol["seeds"][0]) not in protocol["seeds"]):
        raise ValueError("Baseline configuration must be one of the declared sizes/seeds.")


def make_splits(y, protocol):
    """Fixed balanced validation; exact-prefix, balanced training sets per seed."""
    result = {}
    for pair in protocol["pairs"]:
        random = np.random.RandomState(protocol["validation_seed"])
        half = protocol["validation_size_per_pair"] // 2
        validation, remaining = [], []
        for label in (pair["negative"], pair["positive"]):
            indices = random.permutation(np.flatnonzero(y == label))
            if len(indices) < half + max(protocol["train_sizes"]) // 2:
                raise ValueError(f"Insufficient official-training rows for pair {pair['id']}.")
            validation.append(indices[:half])
            remaining.append(np.sort(indices[half:]))
        split = {"validation_indices": np.column_stack(validation).ravel().tolist(), "training_indices": {}}
        for seed in protocol["seeds"]:
            random = np.random.RandomState(seed)
            orders = [random.permutation(indices) for indices in remaining]
            for size in protocol["train_sizes"]:
                split["training_indices"][f"n{size}-seed{seed}"] = np.column_stack([row[:size // 2] for row in orders]).ravel().tolist()
        split["validation_indices_sha256"] = _digest(split["validation_indices"])
        split["training_indices_sha256"] = {key: _digest(value) for key, value in split["training_indices"].items()}
        result[pair["id"]] = split
    return result


def _dataset(protocol):
    path = Path(protocol["dataset"])
    cache = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if protocol.get("dataset_manifest_sha256", _file_hash(manifest_path)) != _file_hash(manifest_path):
        raise ValueError("Dataset manifest differs from its protocol-bound checksum.")
    if manifest.get("official_test_loaded") is not False:
        raise ValueError("Dataset manifest must explicitly exclude official test data.")
    # Deliberately never enumerate/open official-test arrays.
    X = np.load(cache / "X_train.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(cache / "y_train.npy", allow_pickle=False)
    if (X.shape != EXPECTED_SHAPE or X.dtype != np.float32 or not np.isfinite(X).all()
            or X.min() < 0 or X.max() > 1 or y.shape != (len(X),) or y.dtype != np.uint8
            or np.any(y > 9)):
        raise ValueError("Expected official training pixels as float32 [0,1] and uint8 labels 0..9.")
    identity = {"cache": str(cache), "manifest": manifest, "manifest_file_sha256": _file_hash(manifest_path),
                "files_sha256": {name: _file_hash(cache / name) for name in ("X_train.npy", "y_train.npy")},
                "shape": list(X.shape), "dtype": str(X.dtype), "class_counts": np.bincount(y, minlength=10).tolist()}
    for name, array in (("X_train.npy", X), ("y_train.npy", y)):
        record = manifest.get("arrays", {}).get(name, {})
        if (record.get("sha256") != identity["files_sha256"][name]
                or record.get("shape") != list(array.shape) or record.get("dtype") != str(array.dtype)):
            raise ValueError("Training arrays differ from their manifest checksum, shape or dtype.")
    if EXPECTED_SHAPE == (60000, 784) and identity["class_counts"] != [6000] * 10:
        raise ValueError("Official Fashion-MNIST training must contain exactly 6,000 rows per class.")
    if manifest.get("class_counts", identity["class_counts"]) != identity["class_counts"]:
        raise ValueError("Training label counts differ from the manifest.")
    return X, y, identity


def _save_arrays(path, X, y):
    path.mkdir(parents=True, exist_ok=True)
    for name, array in (("X.npy", X), ("y.npy", y)):
        target = path / name
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, target)
    return {"path": str(path.resolve()), "data_sha256": data_fingerprint(X, y),
            "files_sha256": {name: _file_hash(path / name) for name in ("X.npy", "y.npy")},
            "shape": list(X.shape), "class_counts": np.bincount(y, minlength=2).tolist()}


def _load_arrays(record):
    filenames = record.get("filenames", {"X": "X.npy", "y": "y.npy"})
    return (np.load(Path(record["path"]) / filenames["X"], mmap_mode="r", allow_pickle=False),
            np.load(Path(record["path"]) / filenames["y"], allow_pickle=False))


def _verify_arrays(record):
    if any(_file_hash(Path(record["path"]) / name) != value for name, value in record["files_sha256"].items()):
        raise ValueError("Materialized training/validation cache checksum changed.")


def _preflight(X, y, seed, thresholds):
    """Count masks on training folds only; never score or read held-out pixels."""
    folds, rows = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)), []
    with threadpool_limits(limits=1):
        for fold, (train, validation) in enumerate(folds):
            try:
                selector = BinaryScreening(screening="absolute", score_threshold=0).fit(X[train], y[train])
            except ValueError as error:
                if "Screening selected zero features" not in str(error):
                    raise
                selector = None
            screens = {}
            for threshold in thresholds:
                mask = (selector.get_support() & (np.abs(selector.t_scores_) >= threshold)
                        if selector is not None else np.zeros(X.shape[1], dtype=bool))
                screens[str(threshold)] = {"features": int(mask.sum()), "mask_sha256": _digest(mask.tolist())}
            rows.append({"fold": fold, "training_size": len(train), "screens": screens})
    return {"folds": rows, "fold_indices_sha256": _digest(_jsonable([
        {"training_indices": train, "validation_indices": validation} for train, validation in folds]))}


def _plan(job, grid, preflight):
    method = job["method"]
    is_nmf = method in NMF_METHODS
    rank_key = "n_components" if is_nmf else ("nmf__n_components" if method in EXPERIMENTAL_METHODS else "pca__n_components")
    screen_key = "score_threshold" if is_nmf else "screen__score_threshold"
    neighbor_key = "n_neighbors" if is_nmf or method == "raw_knn" else "knn__n_neighbors"
    ranks = grid["n_components"] if is_nmf or method in EXPERIMENTAL_METHODS or method == "screened_pca_absolute" else [None]
    thresholds = grid["score_threshold"] if method != "raw_knn" else [None]
    candidates, chunks, excluded, declared = [], [], [], []
    for rank in ranks:
        for threshold in thresholds:
            representation = {}
            if rank is not None:
                representation[rank_key] = rank
            if threshold is not None:
                representation[screen_key] = threshold
            parameters = [{**representation, neighbor_key: k} for k in grid["n_neighbors"]]
            indices = list(range(len(declared), len(declared) + len(parameters)))
            declared.extend(parameters)
            violations = []
            for fold in preflight["folds"]:
                count = fold["screens"][str(threshold)]["features"] if threshold is not None else None
                if count == 0 or (rank is not None and rank > min(count, fold["training_size"])):
                    violations.append({"fold": fold["fold"], "selected_features": count,
                                       "training_size": fold["training_size"]})
            if violations:
                excluded.append({"parameters": representation, "candidate_indices": indices,
                                 "reason": "training_fold_screen_or_rank_dimension", "folds": violations})
                continue
            candidates.extend({"index": index, "parameters": item} for index, item in zip(indices, parameters))
            chunks.append({"id": f"c{len(chunks):04d}", "candidate_indices": indices,
                           "grid": [{**{key: [value] for key, value in representation.items()},
                                     neighbor_key: grid["n_neighbors"]}]})
    return {"job": job, "declared_candidates": declared, "candidates": candidates,
            "chunks": chunks, "excluded_representations": excluded,
            "fold_indices_sha256": preflight["fold_indices_sha256"]}


def initialize(protocol_path, output):
    protocol_path, output = Path(protocol_path), Path(output).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    X, y, dataset = _dataset(protocol)
    header = {"schema_version": 1, "protocol": protocol, "protocol_file_sha256": _file_hash(protocol_path),
              "sources": _sources(), "dataset": dataset, "official_test_loaded": False}
    path = output / "configuration.json"
    if path.exists():
        saved = _read_receipt(path)
        if any(saved[key] != value for key, value in header.items()):
            raise ValueError("Resume configuration differs in protocol, data, source or runtime.")
        for record in saved["arrays"].values():
            _verify_arrays(record)
        return saved
    if any(output.glob("*/c*/identity.json")) or (output / "comparison.json").exists():
        raise ValueError("Existing results have no matching configuration; use a new directory.")
    started = time.perf_counter()
    splits, arrays, preflights, plans = make_splits(y, protocol), {}, {}, []
    baseline_size = protocol.get("baseline_train_size", max(protocol["train_sizes"]))
    baseline_seed = protocol.get("baseline_seed", protocol["seeds"][0])
    for pair in protocol["pairs"]:
        pair_id = pair["id"]
        indices = splits[pair_id]["validation_indices"]
        arrays[f"{pair_id}/validation"] = _save_arrays(output / "arrays" / pair_id / "validation",
                                                      X[indices], (y[indices] == pair["positive"]).astype(np.uint8))
        for size in protocol["train_sizes"]:
            for seed in protocol["seeds"]:
                key = f"{pair_id}/n{size}-seed{seed}"
                indices = splits[pair_id]["training_indices"][f"n{size}-seed{seed}"]
                Xt, yt = X[indices], (y[indices] == pair["positive"]).astype(np.uint8)
                arrays[key] = _save_arrays(output / "arrays" / key, Xt, yt)
                preflights[key] = _preflight(Xt, yt, seed, protocol["grid"]["score_threshold"])
                methods = list(protocol.get("methods", NMF_METHODS))
                if size == baseline_size and seed == baseline_seed:
                    methods.extend(BASELINE_METHODS)
                for method in methods:
                    job = {"id": f"{pair_id}-n{size}-seed{seed}-{method}", "pair": pair,
                           "seed": seed, "method": method, "n_train": size, "array_key": key,
                           "cache": arrays[key]["path"], "max_iter": protocol["max_iter"], "tol": protocol["tol"],
                           "anls_max_iter": protocol.get("anls_max_iter", 1000),
                           "anls_nnls_max_iter": protocol.get("anls_nnls_max_iter"),
                           "anls_nnls_tol": protocol.get("anls_nnls_tol", 1e-10),
                           "training_data_sha256": arrays[key]["data_sha256"]}
                    plans.append(_plan(job, protocol["grid"], preflights[key]))
    order = np.random.RandomState(protocol.get("profile_order_seed", 42043)).permutation(len(plans))
    configuration = {**header, "splits": splits, "arrays": arrays, "preflight": preflights,
                     "plans": plans, "evaluation_order": [plans[i]["job"]["id"] for i in order]}
    _write_receipt(path, configuration)
    _write_receipt(output / "preflight.json", {"author": "Chang Cheng", "configuration_sha256": _digest(configuration),
                   "elapsed_seconds": time.perf_counter() - started, "training_folds_only": True,
                   "records": preflights, "excluded": {p["job"]["id"]: p["excluded_representations"] for p in plans}})
    write_json(output / "protocol.json", protocol)
    write_json(output / "provenance.json", provenance(1))
    return configuration


def _estimator(job):
    if job["method"] in EXPERIMENTAL_METHODS:
        if job["method"] == "screened_mu_absolute":
            decomposition = NMF(solver="mu", init="nndsvda", random_state=job["seed"],
                                max_iter=job["max_iter"], tol=job["tol"])
        else:
            from experimental_nmf import ANLSBPP
            decomposition = ANLSBPP(random_state=job["seed"], max_iter=job.get("anls_max_iter", 1000), tol=job["tol"],
                                    nnls_max_iter=job.get("anls_nnls_max_iter"), nnls_tol=job.get("anls_nnls_tol", 1e-10))
        return Pipeline([("screen", BinaryScreening(screening="absolute")), ("nmf", decomposition),
                         ("knn", KNeighborsClassifier(n_jobs=1))])
    return comparison_models(random_state=job["seed"], max_iter=job["max_iter"], tol=job["tol"])[job["method"]][0]


def _worker(work):
    output, plan, chunk, sources = work
    _check_source(sources)
    job = plan["job"]
    X, y = _load_arrays(job.get("training_array_record", {"path": job["cache"]}))
    if data_fingerprint(X, y) != job["training_data_sha256"]:
        raise ValueError("Worker training data differ from the frozen configuration.")
    report = calibration_knn_grid(_estimator(job), chunk["grid"], X, y, cv=5, seed=job["seed"], threads=1,
                                  checkpoint_dir=Path(output) / job["id"] / chunk["id"])
    _check_source(sources)
    return {"job": job["id"], "chunk": chunk["id"], "valid_candidates": sum(row["valid"] for row in report["candidates"]),
            "candidates": len(report["candidates"]), "new_representation_fits": report["new_representation_fits"],
            "reused_representation_folds": report["reused_representation_folds"], "worker_pid": os.getpid()}


def aggregate_cv(output, configuration, *, require_complete=False):
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
        canonical = {_digest(row["parameters"]): row["index"] for row in plan["candidates"]}
        candidates, costs, receipts = [], {}, {}
        expected_runtime = _calibration_runtime(_estimator(job), 1)
        for chunk, path in zip(plan["chunks"], paths):
            report = _read_receipt(path)
            identity = report["identity"]
            calibration = identity["calibration"]
            expected = list(ParameterGrid(chunk["grid"]))
            if (_read_receipt(path.parent / "identity.json") != identity or report["state"] != "cv_completed"
                    or report["fingerprint"] != _digest(identity) or calibration["data_sha256"] != job["training_data_sha256"]
                    or calibration["candidates"] != expected or calibration["cv"] != 5 or calibration["seed"] != job["seed"]
                    or calibration["threads"] != 1 or calibration["refit"] is not False
                    or calibration["parameters"] != _jsonable(_estimator(job).get_params(deep=True))
                    or calibration["source_sha256"] != configuration["sources"]["package_source_sha256"]
                    or identity["runtime"] != expected_runtime
                    or _digest(identity["folds"]) != plan["fold_indices_sha256"]
                    or [row["parameters"] for row in report["candidates"]] != expected):
                raise ValueError(f"CV checkpoint identity or grid coverage differs: {path}")
            for row in report["candidates"]:
                row = deepcopy(row)
                if ([fold["fold"] for fold in row["folds"]] != list(range(5))
                        or row["valid"] != all(fold["status"] == "valid" for fold in row["folds"])):
                    raise ValueError("CV candidate must contain five qualified folds.")
                score = float(np.mean([fold["balanced_accuracy"] for fold in row["folds"]])) if row["valid"] else None
                if row["mean_balanced_accuracy"] != score:
                    raise ValueError("CV aggregate score differs from its folds.")
                row.update(index=canonical[_digest(row["parameters"])], chunk=chunk["id"])
                candidates.append(row)
            for key, value in report["compute_seconds"].items():
                costs[key] = costs.get(key, 0.0) + value
            receipts[chunk["id"]] = _digest(report)
        candidates.sort(key=lambda row: row["index"])
        if [row["index"] for row in candidates] != [row["index"] for row in plan["candidates"]]:
            raise ValueError("CV coverage differs from the feasible declared grid.")
        valid = [row for row in candidates if row["valid"]]
        winner = min(valid, key=lambda row: (-row["mean_balanced_accuracy"], row["index"])) if valid else None
        summaries[job["id"]] = {"author": "Chang Cheng", "job": job, "configuration_sha256": _digest(configuration),
            "state": "cv_completed", "candidates": candidates, "declared_candidates": plan["declared_candidates"],
            "excluded_representations": plan["excluded_representations"], "selected_candidate": winner,
            "selected_parameters": winner["parameters"] if winner else None,
            "cv_balanced_accuracy": winner["mean_balanced_accuracy"] if winner else None,
            "compute_seconds": costs, "tuning_compute_seconds": sum(costs.values()), "report_receipts": receipts,
            "refinement": refinement_diagnostics(plan["declared_candidates"], winner["parameters"] if winner else None),
            "selection_rule": "highest_complete_valid_five_fold_balanced_accuracy_then_declared_rank_threshold_k_index",
            "timing_note": "Summed worker wall timers count each shared representation once, not CPU time; exclude I/O and interrupted work.",
            "official_test_loaded": False, "is_fallback": False}
        _write_receipt(output / job["id"] / "calibration-summary.json", summaries[job["id"]])
    if require_complete and missing:
        raise ValueError(f"All protocol CV must finish before validation; {len(missing)} chunks are missing.")
    result = {"author": "Chang Cheng", "configuration_sha256": _digest(configuration),
              "state": "cv_incomplete" if missing else "cv_completed", "jobs": summaries, "missing_chunks": missing,
              "official_test_loaded": False, "validation_evaluated": False}
    _write_receipt(output / "cv-summary.json", result)
    return result


def run_cv(output, configuration, *, workers=None):
    workers = _integer(workers if workers is not None else configuration["protocol"].get("workers", 8), "workers", 1, 8)
    _check_source(configuration["sources"])
    work = [(str(output), plan, chunk, configuration["sources"])
            for plan in configuration["plans"] for chunk in plan["chunks"]
            if not (Path(output) / plan["job"]["id"] / chunk["id"] / "report.json").exists()]
    completed = []
    # Spawn prevents inheriting numerical threads/state. Each fit also enters native1.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(_worker, item) for item in work]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed.append(result)
            print(f"CV {index}/{len(work)} {result['job']} {result['chunk']} valid={result['valid_candidates']}/{result['candidates']}", flush=True)
    summary = aggregate_cv(output, configuration)
    return {"state": summary["state"], "completed_chunks_this_call": len(completed), "missing_chunks": summary["missing_chunks"]}


def _profile_medians(records):
    output = {}
    for key in records[0]["timing"]:
        values = [row["timing"][key] for row in records]
        output[key] = float(np.median(values)) if all(value is not None for value in values) else None
    output["fit_plus_prediction_seconds"] = float(np.median([
        row["timing"]["total_fit_seconds"] + row["timing"]["prediction_seconds"] for row in records]))
    return output


def _profile(estimator, X, y, Xv, yv, *, reference, sample_ids):
    """Extend unchanged stage instrumentation to the two experimental pipelines."""
    if not (isinstance(estimator, Pipeline) and "nmf" in estimator.named_steps):
        return profile_selected_estimator(estimator, X, y, Xv, yv, reference=reference, sample_ids=sample_ids)
    model, caught = clone(estimator), []
    record = {"status": "failed", "exact_prediction_match": False,
              "timing": {"total_fit_seconds": None, "prediction_seconds": None}, "evaluation": None}
    with threadpool_limits(limits=1), _StageProfiler() as meter:
        decomposition_type = type(estimator.named_steps["nmf"])
        if decomposition_type is not NMF:
            meter._wrap(decomposition_type, "fit_transform", {"fit": "decomposition_fit_seconds"})
            meter._wrap(decomposition_type, "transform", {"fit": "training_projection_seconds",
                                                         "prediction": "decomposition_projection_seconds"})
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X, y)
            record["timing"]["total_fit_seconds"] = time.perf_counter() - start
            record["diagnostics"] = _model_diagnostics(model)
            if _model_nonconverged(model, caught):
                record["status"] = "nonconverged"
            else:
                meter.phase = "prediction"
                reproduced = evaluate_classifier(model, Xv, yv, threads=1, sample_ids=sample_ids)
                record["evaluation"] = reproduced
                record["diagnostics"] = _model_diagnostics(model)
                record["timing"]["prediction_seconds"] = reproduced["prediction_seconds"]
                comparison = _compare_predictions(reference["predictions"], reproduced.get("predictions", {}))
                record["prediction_comparison"] = comparison
                record["exact_prediction_match"] = comparison["exact_match"]
                record["status"] = ("valid" if reproduced["status"] == "valid" and comparison["exact_match"]
                                    and not _model_nonconverged(model, []) else "invalid_reproduction")
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            if record["timing"]["total_fit_seconds"] is None:
                record["timing"]["total_fit_seconds"] = time.perf_counter() - start
            record["fit_warnings"] = _warning_records(caught)
            record["timing"].update(meter.seconds)
            record["stage_calls"] = meter.calls
    expected = {"screening_fit_seconds", "decomposition_fit_seconds", "classifier_fit_seconds",
                "screening_prediction_seconds", "decomposition_projection_seconds", "classifier_prediction_seconds"}
    if record["status"] == "valid" and any(record["stage_calls"][key] < 1 for key in expected):
        record.update(status="failed_instrumentation", reason="An expected original stage method was not observed.")
    return record


def _model_diagnostics(model):
    output = _diagnostics(model)
    decomposition = model.named_steps.get("nmf") if isinstance(model, Pipeline) else getattr(model, "decomposition_", None)
    if decomposition is not None:
        output["decomposition"] = {key: _jsonable(getattr(decomposition, key)) for key in (
            "n_iter_", "converged_", "last_transform_converged_", "reconstruction_err_", "solver", "tol", "max_iter",
            "nnls_tol", "nnls_max_iter", "convergence_reason_", "relative_projected_gradient_",
            "initial_projected_gradient_", "nnls_diagnostics_", "last_transform_diagnostics_", "computation_dtype_")
            if hasattr(decomposition, key)}
        if hasattr(decomposition, "components_"):
            output["basis_dtype"] = str(decomposition.components_.dtype)
        solver = getattr(model, "solver", getattr(decomposition, "solver", "anls_bpp"))
        output["solver"] = solver
        output["stopping_criterion"] = {
            "cd": "sklearn accumulated coordinate projected-gradient violation divided by its first-sweep value <= tol",
            "mu": "sklearn decrease in Frobenius reconstruction norm over ten iterations divided by initialization norm < tol",
            "spa": "requested feature-anchor count, followed by per-row scipy NNLS; classifier tol is not a SPA stopping tolerance",
            "anls_bpp": "full projected-gradient norm <= tol times its initialization norm; each inner NNLS separately qualified",
        }.get(solver, "See estimator source and recorded parameters.")
        coordinates = model.named_steps["knn"]._fit_X if isinstance(model, Pipeline) else model.training_coordinates_
        output["training_coordinates_dtype"] = str(coordinates.dtype)
    return output


def _model_nonconverged(model, caught):
    """Inspect fitted pipeline steps as well as their required warning contract."""
    return _nonconverged(model, caught) or (isinstance(model, Pipeline) and any(
        _nonconverged(step, []) for _, step in model.steps))


def _warning_free_profile(profile):
    return not profile.get("fit_warnings") and not (profile.get("evaluation") or {}).get("warnings")


def _training_stationarity(model, X):
    """Common descriptive KKT diagnostics; never alter classifier coordinates."""
    if isinstance(model, Pipeline):
        if "nmf" not in model.named_steps:
            return {"status": "not_applicable"}
        W, H = model.named_steps["knn"]._fit_X, model.named_steps["nmf"].components_
        selector = model.named_steps["screen"]
    else:
        if not hasattr(model, "selected_components_"):
            return {"status": "not_applicable"}
        W, H, selector = model.training_coordinates_, model.selected_components_, model.selector_
    started = time.perf_counter()
    from experimental_nmf import projected_gradient_kkt
    try:
        result = projected_gradient_kkt(selector.transform(X), W, H)
        return {"status": "valid", "diagnostics": result,
                "diagnostic_seconds": time.perf_counter() - started,
                "role": "Descriptive stationarity at stored training coordinates and selected basis; not a common stopping certificate, tuning criterion, or added fit/predict cost. SPA does not optimize the joint NMF objective."}
    except Exception as error:
        return {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                "diagnostic_seconds": time.perf_counter() - started}


def evaluate(output, configuration):
    output = Path(output)
    cv = aggregate_cv(output, configuration, require_complete=True)
    # Freeze exact CV evidence before any validation scoring, including resume.
    gate = {"configuration_sha256": _digest(configuration), "cv_sha256": _digest(cv),
            "evaluation_order": configuration["evaluation_order"], "all_cv_completed": True}
    gate_path = output / "validation-gate.json"
    if gate_path.exists() and _read_receipt(gate_path) != gate:
        raise ValueError("Frozen CV evidence changed after validation started.")
    _write_receipt(gate_path, gate)
    for record in configuration["arrays"].values():
        _verify_arrays(record)
    plans = {plan["job"]["id"]: plan for plan in configuration["plans"]}
    results = {}
    for job_id in configuration["evaluation_order"]:
        _check_source(configuration["sources"])
        job, selected = plans[job_id]["job"], cv["jobs"][job_id]
        validation = configuration["arrays"][f"{job['pair']['id']}/validation"]
        ids = configuration["splits"][job["pair"]["id"]]["validation_indices"]
        identity = {"gate_sha256": _digest(gate), "job_id": job_id,
                    "selected_candidate_sha256": _digest(selected["selected_candidate"]),
                    "validation_data_sha256": validation["data_sha256"]}
        path = output / job_id / "evaluation.json"
        if path.exists():
            result = _read_receipt(path)
            if result["identity"] != identity:
                raise ValueError("Saved evaluation identity differs.")
            results[job_id] = result
            continue
        result = {"author": "Chang Cheng", "identity": identity, "job": job, "jobid": job_id,
                  "pair": job["pair"], "method": job["method"], "seed": job["seed"], "n_train": job["n_train"],
                  "selected_parameters": selected["selected_parameters"], "cv_balanced_accuracy": selected["cv_balanced_accuracy"],
                  "tuning_compute_seconds": selected["tuning_compute_seconds"], "final_fit_seconds": None,
                  "selected_cv_warning_free": bool(selected["selected_candidate"] is not None and all(
                      not fold.get("warnings") for fold in selected["selected_candidate"]["folds"])),
                  "profiling": [], "profiling_valid": False, "profile_medians_seconds": {}, "is_fallback": False,
                  "official_test_loaded": False, "evaluation_role": configuration.get(
                      "evaluation_role", "balanced_development_holdout_from_official_training")}
        if selected["selected_parameters"] is None:
            result["evaluation"] = {"status": "held", "reason": "No candidate qualified in all five training folds."}
        else:
            X, y = _load_arrays(configuration["arrays"][job["array_key"]])
            Xv, yv = _load_arrays(validation)
            template = _estimator(job).set_params(**selected["selected_parameters"])
            original_path = output / job_id / "original-evaluation.json"
            with threadpool_limits(limits=1):
                if original_path.exists():
                    result = _read_receipt(original_path)
                    if result["identity"] != identity:
                        raise ValueError("Saved original selected fit identity differs.")
                else:
                    model, caught = clone(template), []
                    start = time.perf_counter()
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(X, y)
                        result["final_fit_seconds"] = time.perf_counter() - start
                        result["diagnostics"] = _model_diagnostics(model)
                        result["training_stationarity"] = _training_stationarity(model, X)
                        result["evaluation"] = ({"status": "held", "reason": "Selected final fit did not converge."}
                                                if _model_nonconverged(model, caught)
                                                else evaluate_classifier(model, Xv, yv, threads=1, sample_ids=ids))
                        result["diagnostics"] = _model_diagnostics(model)
                        if result["evaluation"]["status"] == "valid" and _model_nonconverged(model, []):
                            result["evaluation"]["status"] = "provisional_nonconvergence"
                    except Exception as error:
                        result["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
                    finally:
                        if result["final_fit_seconds"] is None:
                            result["final_fit_seconds"] = time.perf_counter() - start
                        result["final_fit_warnings"] = _warning_records(caught)
                    _check_source(configuration["sources"])
                    _write_receipt(original_path, result)
                original_sha256 = _digest(result)
                if result["evaluation"]["status"] == "valid":
                    repeats = configuration["protocol"].get("profile_repeats", configuration["protocol"].get("profiles", 3))
                    for repeat in range(repeats):
                        profile_path = output / job_id / f"profile-{repeat}.json"
                        profile_identity = {"original_evaluation_sha256": original_sha256, "repeat": repeat}
                        if profile_path.exists():
                            profile = _read_receipt(profile_path)
                            if profile["identity"] != profile_identity:
                                raise ValueError("Saved serial profile identity differs.")
                        else:
                            profile = _profile(template, X, y, Xv, yv, reference=result["evaluation"], sample_ids=ids)
                            profile.update(identity=profile_identity, repeat=repeat)
                            _check_source(configuration["sources"])
                            _write_receipt(profile_path, profile)
                        result["profiling"].append(profile)
                    result["profiling_valid"] = bool(result["selected_cv_warning_free"] and not result["final_fit_warnings"]
                        and not result["evaluation"].get("warnings") and all(
                            row["status"] == "valid" and row["exact_prediction_match"] and _warning_free_profile(row)
                            for row in result["profiling"]))
                    if result["profiling_valid"]:
                        result["profile_medians_seconds"] = _profile_medians(result["profiling"])
                    result["profile_timing_note"] = "Three additional serial fits in seeded job order; nested stage timers, exact saved labels/probabilities, median of each fit+prediction sum."
        _check_source(configuration["sources"])
        _write_receipt(path, result)
        results[job_id] = result
        print(f"Validation {len(results)}/{len(plans)} {job_id}: {result['evaluation']['status']} profiles={len(result['profiling'])}", flush=True)
    report = {"author": "Chang Cheng", "configuration_sha256": _digest(configuration), "state": "development_evaluation_completed",
              "jobs": results, "official_test_loaded": False, "package_defaults_modified": False}
    _write_receipt(output / "comparison.json", report)
    return {"state": report["state"], "evaluations": len(results),
            "valid_evaluations": sum(row["evaluation"]["status"] == "valid" for row in results.values()),
            "models_with_valid_profiling": sum(row["profiling_valid"] for row in results.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("cv", "evaluate", "all"), default="cv")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(args.output):
        configuration = initialize(args.protocol, args.output)
        attempt_path = args.output / "executions" / f"{uuid.uuid4().hex}.json"
        start = time.perf_counter()
        attempt = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "state": "running",
                   "configuration_sha256": _digest(configuration), "native_threads": 1,
                   "workers": 1 if args.mode == "evaluate" else (args.workers or configuration["protocol"].get("workers", 8)),
                   "evaluation_workers": 1}
        _write_receipt(attempt_path, attempt)
        try:
            if args.mode in {"cv", "all"}:
                attempt["cv"] = run_cv(args.output, configuration, workers=args.workers)
            if args.mode in {"evaluate", "all"}:
                attempt["evaluation"] = evaluate(args.output, configuration)
            attempt["state"] = "completed"
        except BaseException as error:
            attempt.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            attempt.update(current_call_wall_seconds=time.perf_counter() - start,
                           finished_at=datetime.now(timezone.utc).isoformat())
            _write_receipt(attempt_path, attempt)
        print(json.dumps({key: attempt[key] for key in ("state", "cv", "evaluation") if key in attempt}, indent=2), flush=True)


if __name__ == "__main__":
    main()
