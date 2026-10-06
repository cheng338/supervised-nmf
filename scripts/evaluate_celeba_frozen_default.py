"""One frozen CelebA official-test comparison after installing the default.

Author: Chang Cheng. This script never chooses parameters or a default. It first
reproduces the Stage 2 development predictions with the installed package source
for all four methods, both attributes, n_train=8000, seed=43. Only after all eight
models pass exact label/probability checks may it decode official-test images.
Test predictions are saved once and compatible interrupted work can resume.

Prepare the immutable final plan after default installation (no images decoded):
  python scripts/evaluate_celeba_frozen_default.py --prepare-freeze --comparison RUN \
    --decision decision.json --freeze freeze.json --images IMAGES \
    --attributes ATTRIBUTES --partitions PARTITIONS
Execute or resume the identical frozen comparison:
  python scripts/evaluate_celeba_frozen_default.py --comparison RUN \
    --decision decision.json --freeze freeze.json --images IMAGES \
    --attributes ATTRIBUTES --partitions PARTITIONS --output FINAL_RUN

Model pickle files are produced locally by this script. Loading one requires its
own checksummed receipt, matching freeze/source identity, and matching file hash.
Do not use this runner with untrusted local artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import uuid
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import _attribute_labels, _partitions, load_celeba
from supervised_nmf.evaluation import (
    _diagnostics, _digest, _nonconverged, _warning_records, comparison_models,
    data_fingerprint, evaluate_classifier, provenance, write_json,
)
from supervised_nmf.study import _checkpoint_lock, _read_receipt, _write_receipt
from celeba_default_study import _cache_identity
from profile_celeba_stages import _compare_predictions, _runtime_identity

ATTRIBUTES = ("Smiling", "Wearing_Necktie")
METHODS = ("screened_cd_percentile", "screened_cd_absolute", "screened_spa_percentile", "screened_spa_absolute")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(4 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _path(path):
    path = Path(path)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _write_new(path, value, *, receipt=True):
    """Preserve committed artifacts; identical recovery does not rewrite them."""
    path = Path(path)
    if path.exists():
        old = _read_receipt(path) if receipt else _read(path)
        if old != value:
            raise ValueError(f"Existing immutable artifact differs: {path}")
    elif receipt:
        _write_receipt(path, value)
    else:
        write_json(path, value)


def _source():
    environment = provenance(1)
    return {"package_source_sha256": environment["source_sha256"],
            "evaluator_sha256": _sha(__file__),
            "helper_source_sha256": {name: _sha(ROOT / "scripts" / name)
                                     for name in ("celeba_default_study.py", "profile_celeba_stages.py")},
            "runtime": _runtime_identity(environment)}


def _decision(path):
    value = _read(path)
    if "payload" in value:
        value = _read_receipt(path)
    winner = value.get("winner_method", value.get("primary", {}).get("winner"))
    if (winner not in METHODS or value.get("primary", {}).get("status") != "selected"
            or winner != value["primary"].get("winner")):
        raise ValueError("The frozen decision must identify one operationally qualified method.")
    if not isinstance(value.get("input_sha256"), dict) or not value["input_sha256"]:
        raise ValueError("Decision must bind its complete input evidence by file hashes.")
    for filename, expected in value["input_sha256"].items():
        if _sha(filename) != expected:
            raise ValueError(f"Decision input evidence changed: {filename}")
    return value, winner


def _default_only_source_change(old, current):
    """All algorithms outside the classifier API must retain Stage 2 source."""
    changed = sorted(name for name in set(old) | set(current) if old.get(name) != current.get(name))
    if (set(old) != set(current) or "classifier.py" not in changed
            or not set(changed) <= {"classifier.py", "evaluation.py"}):
        raise ValueError("Only classifier.py and optionally evaluation.py may change for the installed numerical defaults.")
    return changed


def _data_sources(manifest, images, attributes, partitions):
    original = manifest["preparation_identity"]
    if _path(images) != _path(original["image_directory"]):
        raise ValueError("Final images must use the Stage 2 local image directory.")
    records = {}
    for name, filename in (("attributes", attributes), ("partitions", partitions)):
        source = {"path": str(_path(filename)), "sha256": _sha(_path(filename))}
        if source["sha256"] != original["source_files"][name]["sha256"]:
            raise ValueError("Final annotations must match the Stage 2 data definition exactly.")
        records[name] = source
    return records


def _stage2(comparison):
    """Verify the eight predeclared model records and their training selection."""
    comparison = _path(comparison)
    config = _read_receipt(comparison / "configuration.json")
    summary = _read_receipt(comparison / "comparison.json")
    if (summary.get("state") != "development_evaluation_completed"
            or summary.get("configuration_sha256") != _digest(config)
            or summary.get("official_test_loaded") is not False):
        raise ValueError("Stage 2 comparison is incomplete, mismatched or used official test data.")
    chosen, pairs = [], []
    for plan in config["plans"]:
        job = plan["job"]
        manifest = config["caches"][job["cache"]]["manifest"]
        n_train = len(manifest["splits"]["train"]["image_ids"])
        if n_train != 8000 or job["seed"] != 43:
            continue
        if job["method"] not in METHODS or job["attribute"] not in ATTRIBUTES:
            raise ValueError("Unexpected method or attribute in the final predeclared block.")
        record = _read_receipt(comparison / job["id"] / "evaluation.json")
        cv = _read_receipt(comparison / job["id"] / "calibration-summary.json")
        selected = cv["selected_candidate"]
        validation = manifest["splits"]["validation"]
        if (summary["jobs"].get(job["id"]) != record or record["job"] != job
                or record["identity"]["configuration_sha256"] != _digest(config)
                or cv["configuration_sha256"] != _digest(config) or cv["job"] != job
                or selected is None or not selected["valid"]
                or record["selected_parameters"] != selected["parameters"]
                or record["identity"]["selected_candidate_sha256"] != _digest(selected)
                or record["evaluation"]["status"] != "valid" or not record.get("profiling_valid")
                or len(record.get("profiling", [])) != 3
                or not all(item["status"] == "valid" and item["exact_prediction_match"] for item in record["profiling"])
                or len(validation["image_ids"]) != 4000
                or record["evaluation"]["predictions"]["sample_ids"] != validation["image_ids"]
                or record["evaluation"]["data_sha256"] != validation["attributes"][job["attribute"]]["data_sha256"]
                or job["max_iter"] != 20000 or job["tol"] != 1e-4):
            raise ValueError(f"Stage 2 selected-model evidence is not qualified: {job['id']}")
        chosen.append({"job": job, "manifest": manifest, "evaluation": record,
                       "evaluation_file_sha256": _sha(comparison / job["id"] / "evaluation.json"),
                       "cv_summary_file_sha256": _sha(comparison / job["id"] / "calibration-summary.json")})
        pairs.append((job["attribute"], job["method"]))
    if len(chosen) != 8 or set(pairs) != {(attribute, method) for attribute in ATTRIBUTES for method in METHODS}:
        raise ValueError("Final evaluation requires exactly four methods x two attributes at n8000/seed43.")
    if len({row["job"]["cache"] for row in chosen}) != 1:
        raise ValueError("All final models must use the same saved training/validation images.")
    return config, chosen


def prepare_freeze(comparison, decision_path, freeze_path, images, attributes, partitions):
    """Materialize a complete plan without fitting models or decoding images."""
    comparison, decision_path = _path(comparison), _path(decision_path)
    config, chosen = _stage2(comparison)
    decision, winner = _decision(decision_path)
    if decision.get("configuration_sha256") != _digest(config):
        raise ValueError("Decision does not belong to this Stage 2 configuration.")
    defaults = NMFClassifier().get_params()
    if (winner != f"screened_{defaults['solver']}_{defaults['screening']}"
            or defaults["max_iter"] != 20000 or defaults["tol"] != 1e-4):
        raise ValueError("Install the recorded method default and qualified numerical cap before freezing the final test.")
    current = _source()
    if current["runtime"] != config["sources"]["runtime"]:
        raise ValueError("Final evaluation must use the same numerical runtime as Stage 2.")
    changed = _default_only_source_change(config["sources"]["package_source_sha256"], current["package_source_sha256"])
    sources = _data_sources(chosen[0]["manifest"], images, attributes, partitions)
    values = {"schema_version": 1, "author": "Chang Cheng", "decision_frozen": True,
              "winner_method": winner, "decision_file_sha256": _sha(decision_path),
              "stage2_configuration_sha256": _digest(config),
              "stage2_comparison_file_sha256": _sha(comparison / "comparison.json"),
              "stage2_package_source_sha256": config["sources"]["package_source_sha256"],
              "changed_package_source_files": changed,
              "stage2_run": str(comparison), "decision_path": str(decision_path),
              "training_cache": chosen[0]["job"]["cache"], "n_train": 8000, "seed": 43,
              "validation_count": 4000, "test_count": 19962,
              "job_ids": [row["job"]["id"] for row in chosen],
              "stage2_model_evidence": {row["job"]["id"]: {
                  "evaluation_file_sha256": row["evaluation_file_sha256"],
                  "cv_summary_file_sha256": row["cv_summary_file_sha256"],
                  "parameters": row["evaluation"]["selected_parameters"],
                  "max_iter": row["job"]["max_iter"], "tol": row["job"]["tol"]} for row in chosen},
              **current, "installed_constructor_defaults": defaults,
              "images": str(_path(images)),
              "attributes": sources["attributes"], "partitions": sources["partitions"],
              "native_threads": 1, "default_change_from_test": False,
              "validation_gate": "all_8_models_exact_labels_probabilities_ids_before_test_image_decode",
              "test_scope": "all_official_test_images_same_preprocessing_no_sampling_no_balancing",
              "recovery_policy": "same_freeze_model_and_data_only; retain_interrupted_attempts_no_new_selection"}
    _write_new(freeze_path, values, receipt=False)
    return values


def _validate_freeze(freeze_path, comparison, decision_path, images, attributes, partitions):
    freeze = _read(freeze_path)
    # Reconstructing the plan is read-only for an existing identical freeze.
    expected = prepare_freeze(comparison, decision_path, freeze_path, images, attributes, partitions)
    if freeze != expected:
        raise ValueError("Final plan differs from the frozen configuration.")
    return freeze


def _assert_sources(freeze):
    current = _source()
    if current != {key: freeze[key] for key in current}:
        raise ValueError("Package, runner, helpers or numerical runtime changed after final freeze.")


def _assert_annotation_files(freeze):
    for name in ("attributes", "partitions"):
        if _sha(freeze[name]["path"]) != freeze[name]["sha256"]:
            raise ValueError("Attribute or official-partition annotations changed after freeze.")


def _model_path(output, relative):
    path = (Path(output) / relative).resolve()
    if path.parent != (Path(output) / "models").resolve() or path.suffix != ".pkl":
        raise ValueError("Model checkpoint path is outside this run's models directory.")
    return path


def _load_model(output, record, freeze):
    if record["freeze_sha256"] != _digest(freeze) or record["status"] != "valid":
        raise ValueError("Model checkpoint is not qualified for the exact frozen source/decision.")
    _assert_sources(freeze)
    path = _model_path(output, record["model_file"])
    if _sha(path) != record["model_sha256"]:
        raise ValueError("Locally generated model checkpoint failed its hash check.")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    if type(model) is not NMFClassifier or model.get_params() != record["model_parameters"]:
        raise ValueError("Restored model type or explicit parameters differ.")
    return model


def _record_attempt(output, phase, job_id, freeze, extra=None):
    path = Path(output) / "attempts" / f"{phase}-{job_id}-{uuid.uuid4().hex}.json"
    payload = {"phase": phase, "job_id": job_id, "freeze_sha256": _digest(freeze),
               "state": "running", "started_at": _now(), **(extra or {})}
    _write_receipt(path, payload)
    return path, payload


def revalidate_models(output, freeze, chosen):
    """Fit/recover all eight models and close the gate only after exact checks."""
    output = Path(output)
    jobs = [row["job"] for row in chosen]
    cache = Path(freeze["training_cache"])
    checked = _cache_identity(cache, jobs)
    _cache_identity(cache, jobs, validation=True)
    if checked["manifest"] != chosen[0]["manifest"]:
        raise ValueError("Training/validation cache differs from the Stage 2 source evidence.")
    X = np.load(cache / "train-X.npy", mmap_mode="r", allow_pickle=False)
    Xv = np.load(cache / "validation-X.npy", mmap_mode="r", allow_pickle=False)
    records = {}
    for source in chosen:
        _assert_sources(freeze)
        job = source["job"]
        path = output / "revalidation" / f"{job['id']}.json"
        if path.exists():
            record = _read_receipt(path)
            if (record["job_id"] != job["id"] or record["stage2_evaluation_file_sha256"] != source["evaluation_file_sha256"]
                    or record["model_parameters"] != comparison_models(random_state=43, max_iter=20000, tol=1e-4)[job["method"]][0].set_params(**source["evaluation"]["selected_parameters"]).get_params()
                    or record["validation_evaluation"]["status"] != "valid"
                    or not _compare_predictions(source["evaluation"]["evaluation"]["predictions"], record["validation_evaluation"]["predictions"])["exact_match"]):
                raise ValueError("Saved revalidation record differs from its declared Stage 2 model.")
            _load_model(output, record, freeze)
            records[job["id"]] = record
            continue
        attempt_path, attempt = _record_attempt(output, "revalidate", job["id"], freeze)
        start = time.perf_counter()
        try:
            y = np.load(cache / f"train-{job['attribute']}-y.npy", allow_pickle=False)
            yv = np.load(cache / f"validation-{job['attribute']}-y.npy", allow_pickle=False)
            model = comparison_models(random_state=43, max_iter=20000, tol=1e-4)[job["method"]][0]
            model.set_params(**source["evaluation"]["selected_parameters"])
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_start = time.perf_counter()
                model.fit(X, y)
                fit_seconds = time.perf_counter() - fit_start
            fit_warnings = _warning_records(caught)
            if _nonconverged(model, caught):
                raise ValueError("Final-source training fit did not converge; official-test gate remains closed.")
            evaluation = evaluate_classifier(model, Xv, yv, threads=1,
                                             sample_ids=source["manifest"]["splits"]["validation"]["image_ids"])
            check = _compare_predictions(source["evaluation"]["evaluation"]["predictions"], evaluation.get("predictions", {}))
            if evaluation["status"] != "valid" or not check["exact_match"]:
                raise ValueError("Final-source validation predictions differ; official-test gate remains closed.")
            _assert_sources(freeze)
            relative = str(Path("models") / f"{job['id']}-{uuid.uuid4().hex}.pkl")
            model_path = _model_path(output, relative)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with model_path.open("xb") as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
            record = {"author": "Chang Cheng", "status": "valid", "freeze_sha256": _digest(freeze),
                      "job_id": job["id"], "method": job["method"], "attribute": job["attribute"],
                      "stage2_evaluation_file_sha256": source["evaluation_file_sha256"],
                      "model_file": relative, "model_sha256": _sha(model_path), "model_parameters": model.get_params(),
                      "fit_seconds": fit_seconds, "fit_warnings": fit_warnings,
                      "diagnostics": _diagnostics(model), "validation_evaluation": evaluation,
                      "reproducibility": check, "official_test_loaded": False}
            _write_new(path, record)
            records[job["id"]] = record
            attempt["state"] = "completed"
            print(f"Revalidation {len(records)}/8 {job['id']}: exact labels and probabilities", flush=True)
        except BaseException as error:
            attempt.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            attempt.update(finished_at=_now(), wall_seconds=time.perf_counter() - start)
            _write_receipt(attempt_path, attempt)
    gate = {"author": "Chang Cheng", "state": "all_eight_models_revalidated_before_test_decode",
            "freeze_sha256": _digest(freeze), "job_ids": freeze["job_ids"],
            "records": {job_id: _digest(record) for job_id, record in records.items()},
            "models": {job_id: record["model_sha256"] for job_id, record in records.items()}}
    _write_new(output / "validation-gate.json", gate)
    return gate, records


def _save_array(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def load_frozen_test(output, freeze, gate, chosen):
    """The only image-decoding entry point, reachable after the complete gate."""
    output = Path(output)
    _assert_sources(freeze)
    _assert_annotation_files(freeze)
    if (_read_receipt(output / "validation-gate.json") != gate or gate["freeze_sha256"] != _digest(freeze)
            or gate["job_ids"] != freeze["job_ids"] or len(gate["records"]) != 8):
        raise ValueError("Official-test decoding requires the exact completed eight-model validation gate.")
    directory = output / "test-cache"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        attempt_path, attempt = _record_attempt(output, "test-cache", "all", freeze, {"gate_sha256": _digest(gate)})
        start = time.perf_counter()
        try:
            data = load_celeba(freeze["images"], freeze["attributes"]["path"], freeze["partitions"]["path"],
                               attribute="Smiling", split="test", max_samples=None, random_state=43, image_size=(64, 64))
            ids = data.image_ids.tolist()
            if len(ids) != 19962 or data.metadata["partition_code"] != 2 or len(set(ids)) != len(ids):
                raise ValueError("Expected all 19,962 official test images, once each.")
            development_ids = set(chosen[0]["manifest"]["splits"]["train"]["image_ids"]) | set(chosen[0]["manifest"]["splits"]["validation"]["image_ids"])
            if set(ids) & development_ids:
                raise ValueError("Official test IDs overlap the frozen development images.")
            labels = {attribute: _attribute_labels(Path(freeze["attributes"]["path"]), attribute) for attribute in ATTRIBUTES}
            manifest = {"author": "Chang Cheng", "freeze_sha256": _digest(freeze), "gate_sha256": _digest(gate),
                        "metadata": data.metadata, "image_ids": ids, "attributes": {}, "files": {},
                        "official_test_loaded": True, "decode_wall_seconds": time.perf_counter() - start}
            _save_array(directory / "X.npy", data.data)
            manifest["files"]["X.npy"] = _sha(directory / "X.npy")
            for attribute in ATTRIBUTES:
                y = np.asarray([labels[attribute][image_id] for image_id in ids], dtype=np.int64)
                name = f"{attribute}-y.npy"
                _save_array(directory / name, y)
                manifest["files"][name] = _sha(directory / name)
                manifest["attributes"][attribute] = {"class_counts": np.bincount(y, minlength=2).tolist(),
                                                     "data_sha256": data_fingerprint(data.data, y)}
            _assert_sources(freeze)
            _assert_annotation_files(freeze)
            _write_new(manifest_path, manifest)
            attempt["state"] = "completed"
        except BaseException as error:
            attempt.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            attempt.update(finished_at=_now(), wall_seconds=time.perf_counter() - start)
            _write_receipt(attempt_path, attempt)
    manifest = _read_receipt(manifest_path)
    if manifest["freeze_sha256"] != _digest(freeze) or manifest["gate_sha256"] != _digest(gate):
        raise ValueError("Official-test cache belongs to a different frozen plan or gate.")
    expected_ids = sorted(identifier for identifier, code in _partitions(Path(freeze["partitions"]["path"])).items() if code == 2)
    if (manifest["image_ids"] != expected_ids or len(expected_ids) != 19962
            or manifest["metadata"]["partition_code"] != 2
            or set(manifest["files"]) != {"X.npy", "Smiling-y.npy", "Wearing_Necktie-y.npy"}):
        raise ValueError("Official-test cache must cover exactly the official partition in sorted order.")
    for name, expected in manifest["files"].items():
        if name not in {"X.npy", "Smiling-y.npy", "Wearing_Necktie-y.npy"} or _sha(directory / name) != expected:
            raise ValueError("Official-test cache failed its file integrity check.")
    X = np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False)
    if X.shape != (19962, 4096) or X.dtype != np.float32 or not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
        raise ValueError("Official-test preprocessing differs from the frozen specification.")
    for attribute in ATTRIBUTES:
        y = np.load(directory / f"{attribute}-y.npy", allow_pickle=False)
        if data_fingerprint(X, y) != manifest["attributes"][attribute]["data_sha256"]:
            raise ValueError("Official-test values/labels differ from their manifest.")
    return manifest, X


def evaluate_frozen(output, freeze, chosen):
    output = Path(output)
    _write_new(output / "freeze.json", freeze, receipt=False)
    _write_new(output / "source-provenance.json", provenance(1))
    gate, records = revalidate_models(output, freeze, chosen)
    test, X = load_frozen_test(output, freeze, gate, chosen)
    results = {}
    for source in chosen:
        _assert_sources(freeze)
        job = source["job"]
        record = records[job["id"]]
        identity = {"freeze_sha256": _digest(freeze), "job_id": job["id"], "model_sha256": record["model_sha256"],
                    "test_manifest_sha256": _digest(test), "test_data_sha256": test["attributes"][job["attribute"]]["data_sha256"]}
        result_path = output / "evaluations" / f"{job['id']}.json"
        if result_path.exists():
            result = _read_receipt(result_path)
            if result["identity"] != identity:
                raise ValueError("Saved official-test evaluation belongs to another model, data or frozen decision.")
            results[job["id"]] = result
            continue
        attempt_path, attempt = _record_attempt(output, "test-prediction", job["id"], freeze, identity)
        start = time.perf_counter()
        try:
            model = _load_model(output, record, freeze)
            y = np.load(output / "test-cache" / f"{job['attribute']}-y.npy", allow_pickle=False)
            evaluation = evaluate_classifier(model, X, y, threads=1, sample_ids=test["image_ids"])
            result = {"author": "Chang Cheng", "identity": identity, "method": job["method"], "attribute": job["attribute"],
                      "selected_default": job["method"] == freeze["winner_method"],
                      "model_parameters": record["model_parameters"], "fit_seconds": record["fit_seconds"],
                      "validation_reproduced_exactly": True, "evaluation": evaluation,
                      "training_class_counts": source["manifest"]["splits"]["train"]["attributes"][job["attribute"]]["class_counts"],
                      "test_class_counts": test["attributes"][job["attribute"]]["class_counts"],
                      "evaluation_role": "frozen_official_test_comparison_no_selection_or_retuning",
                      "default_changed_from_test": False, "official_test_loaded": True}
            _assert_sources(freeze)
            _write_new(result_path, result)
            results[job["id"]] = result
            attempt["state"] = "completed"
            print(f"Official test {len(results)}/8 {job['attribute']} {job['method']}: "
                  f"{evaluation['status']} {json.dumps(evaluation.get('metrics', {}), sort_keys=True)}", flush=True)
        except BaseException as error:
            attempt.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            attempt.update(finished_at=_now(), wall_seconds=time.perf_counter() - start)
            _write_receipt(attempt_path, attempt)
    summary = {"author": "Chang Cheng", "state": "frozen_official_test_comparison_completed",
               "freeze_sha256": _digest(freeze), "winner_method": freeze["winner_method"], "models": results,
               "test_count": 19962, "official_test_loaded": True, "default_changed_from_test": False,
               "selection_or_retuning_performed": False,
               "timing_note": "Each numerical fit/prediction used one native thread. Revalidation fits are additional fits, separate from Stage 2 timing. Interrupted identical attempts remain recorded and may be recovered."}
    _write_new(output / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-freeze", action="store_true")
    args = parser.parse_args()
    if args.prepare_freeze:
        freeze = prepare_freeze(args.comparison, args.decision, args.freeze, args.images, args.attributes, args.partitions)
        print(json.dumps({"freeze": str(args.freeze.resolve()), "sha256": _sha(args.freeze),
                          "winner_method": freeze["winner_method"], "model_count": len(freeze["job_ids"]),
                          "official_test_loaded": False}, indent=2))
        return
    if args.output is None:
        parser.error("--output is required when executing the frozen comparison")
    freeze = _validate_freeze(args.freeze, args.comparison, args.decision, args.images, args.attributes, args.partitions)
    _, chosen = _stage2(args.comparison)
    args.output.mkdir(parents=True, exist_ok=True)
    with _checkpoint_lock(args.output), threadpool_limits(limits=1):
        summary = evaluate_frozen(args.output, freeze, chosen)
    print(json.dumps({"output": str(args.output.resolve()), "state": summary["state"],
                      "winner_method": summary["winner_method"], "official_test_loaded": True,
                      "default_changed_from_test": False}, indent=2))


if __name__ == "__main__":
    main()
