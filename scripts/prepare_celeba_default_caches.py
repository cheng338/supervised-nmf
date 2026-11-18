"""Prepare immutable nested CelebA training caches with one fixed validation set.

Author: Chang Cheng. Images are decoded once at the largest size for each seed.
Smaller samples exactly match the current load_celeba sampling rule, including
sorted image order. This script prepares data only; it never fits a model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits

from supervised_nmf.datasets import _attribute_labels, _partitions, load_celeba
from supervised_nmf.evaluation import _file_lock, data_fingerprint, provenance

ATTRIBUTES = ("Smiling", "Wearing_Necktie")
SIZES = (2000, 4000, 8000)
SEEDS = (43, 44, 45)
ARRAY_NAMES = tuple(f"{split}-{suffix}.npy" for split in ("train", "validation")
                    for suffix in ("X", "Smiling-y", "Wearing_Necktie-y"))
POLICY = (
    "For each training seed, independently reproduce sorted RandomState(seed).choice("
    "sorted official-training population size, n, replace=False) for each n. Verify "
    "2000 subset 4000 subset 8000. Decode the sorted n8000 IDs with load_celeba once, "
    "then map each smaller sample's global indices into n8000 rows; never take the "
    "first n sorted n8000 rows. Copy the unchanged existing seed42 validation arrays."
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """Publish complete JSON under the preparation lock; never replace a record."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=".j-", dir=path.parent, text=True)
    temporary = Path(name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary.rename(path)


def _confined(path, output):
    """Check exact paths before directory publication or preservation moves."""
    path, output = Path(path), Path(output).resolve()
    resolved = path.resolve()
    if resolved == output or not resolved.is_relative_to(output) or resolved != path or path.is_symlink():
        raise ValueError(f"Cache preparation path must remain directly within its output tree: {path}")
    return path


def _publish_cache(cache, directory, output):
    cache, directory = _confined(cache, output), _confined(directory, output)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Cache destination already exists: {directory}")
    # The output-scoped lock coordinates all participating preparers. Staging
    # and destination share a filesystem; no large-array copy is introduced.
    cache.rename(directory)


def _new_stage(directory, identity, output):
    stage = _confined(Path(tempfile.mkdtemp(prefix=".p-", dir=output)), output)
    write_new(stage / "identity.json", {"staging_schema": 1, "target": directory.name,
                                       "preparation_identity": identity})
    (stage / "c").mkdir(exist_ok=False)
    return stage


def _find_complete_stage(directory, identity, expected_ids, validation, expected_labels, output):
    """Reuse a complete owned stage; leave incomplete attempts unchanged."""
    expected_marker = {"staging_schema": 1, "target": directory.name, "preparation_identity": identity}
    for stage in sorted(output.glob(".p-*")):
        stage = _confined(stage, output)
        if not stage.is_dir():
            continue
        marker, cache = _confined(stage / "identity.json", output), _confined(stage / "c", output)
        if not marker.is_file() or not cache.is_dir():
            continue
        try:
            owned = read(marker) == expected_marker
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not owned:
            continue
        manifest_path = _confined(cache / "manifest.json", output)
        if not manifest_path.is_file():
            continue
        try:
            read(manifest_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        # A parseable complete manifest is evidence to verify, not permission to
        # silently rebuild a corrupt or identity-mismatched completed stage.
        manifest = validate_cache(cache, identity, expected_ids, validation, expected_labels)
        return stage, manifest
    return None


def _preserve_legacy_incomplete(directory, output, recover_incomplete):
    """Only absent/broken manifests qualify for explicit preserve-and-retry."""
    directory = _confined(directory, output)
    if not directory.is_dir():
        raise ValueError(f"Cache destination is not a directory: {directory}")
    manifest_path = _confined(directory / "manifest.json", output)
    incomplete = not manifest_path.is_file()
    if not incomplete:
        try:
            read(manifest_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            incomplete = True
    if not incomplete:
        return None
    if not recover_incomplete:
        raise ValueError(f"Incomplete cache exists: {directory}. Preserve and inspect it, or explicitly "
                         "use --recover-incomplete to retain this directory under a new name and rebuild.")
    retained = _confined(Path(tempfile.mkdtemp(prefix=".r-", dir=output)), output)
    write_new(retained / "origin.json", {"target": directory.name,
                                        "reason": "legacy_missing_or_unreadable_manifest"})
    preserved = _confined(retained / "c", output)
    _publish_cache(directory, preserved, output)
    return preserved



def _receipt_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _write_training_receipt(stage, directory, identity, image_ids, metadata, array_sha256):
    """Commit only after the largest training .npy has closed and been hashed."""
    payload = {"payload_schema": 1, "target": directory.name, "preparation_identity": identity,
               "image_ids": image_ids, "metadata": metadata,
               "shape": [len(image_ids), 4096], "dtype": "float32", "array_sha256": array_sha256}
    write_new(stage / "training.json", {"payload": payload, "sha256": _receipt_digest(payload)})


def _find_training_payload(directory, identity, expected_ids, output):
    """Verify a committed largest-training payload; never trust shape alone."""
    marker_expected = {"staging_schema": 1, "target": directory.name, "preparation_identity": identity}
    for stage in sorted(output.glob(".p-*")):
        stage = _confined(stage, output)
        if not stage.is_dir():
            continue
        marker = _confined(stage / "identity.json", output)
        if not marker.is_file():
            continue
        try:
            owned = read(marker) == marker_expected
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not owned:
            continue
        cache = _confined(stage / "c", output)
        if not cache.exists():
            # Successful publication consumes the entire payload directory.
            continue
        if not cache.is_dir():
            raise ValueError("Training payload cache is not a directory")
        receipt_path = _confined(stage / "training.json", output)
        if not receipt_path.is_file():
            continue
        receipt = read(receipt_path)
        payload = receipt["payload"]
        if receipt["sha256"] != _receipt_digest(payload):
            raise ValueError("Training payload receipt checksum changed")
        if (payload["payload_schema"] != 1 or payload["target"] != directory.name
                or payload["preparation_identity"] != identity or payload["image_ids"] != expected_ids
                or payload["shape"] != [len(expected_ids), 4096] or payload["dtype"] != "float32"):
            raise ValueError("Training payload identity, sample, or array contract changed")
        expected_metadata = {"attribute": "Smiling", "split": "train", "partition_code": 0,
            "partition_size": identity["training_population_size"], "random_state": identity["training_seed"],
            "max_samples": identity["training_size"], "image_size": [64, 64],
            "sampling": "uniform without replacement, no class balancing",
            "preprocessing": "Pillow L grayscale, resize, flatten, float32 / 255",
            "attributes_path": identity["source_files"]["attributes"]["path"],
            "partitions_path": identity["source_files"]["partitions"]["path"]}
        if payload["metadata"] != expected_metadata:
            raise ValueError("Training payload loader metadata changed")
        array = _confined(stage / "c" / "train-X.npy", output)
        if sha(array) != payload["array_sha256"]:
            raise ValueError("Training payload array checksum changed")
        values = np.load(array, mmap_mode="r", allow_pickle=False)
        try:
            if values.shape != (len(expected_ids), 4096) or values.dtype != np.float32:
                raise ValueError("Training payload has wrong shape or dtype")
        finally:
            _close_owned_array(values)
        return stage, payload
    return None


def source_identity(attributes, partitions):
    return {"attributes": {"path": str(attributes), "sha256": sha(attributes)},
            "partitions": {"path": str(partitions), "sha256": sha(partitions)}}


def _close_owned_array(values):
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


def validate_cache(directory, identity, expected_ids, validation, expected_labels):
    """Validate a committed cache without modifying any existing cache file."""
    manifest = read(directory / "manifest.json")
    if manifest["preparation_identity"] != identity:
        raise ValueError(f"Incompatible immutable cache identity: {directory}")
    if manifest["seed"] != identity["training_seed"] or manifest["official_test_loaded"]:
        raise ValueError(f"Unexpected seed or test access in {directory}")
    if manifest["splits"]["validation"] != validation:
        raise ValueError(f"Validation metadata changed in {directory}")
    if manifest["splits"]["train"]["image_ids"] != expected_ids:
        raise ValueError(f"Training sample or ordering changed in {directory}")
    if set(expected_ids) & set(validation["image_ids"]):
        raise ValueError("Training and fixed validation IDs overlap")
    if set(manifest["array_sha256"]) != set(ARRAY_NAMES):
        raise ValueError(f"Unexpected cache file set in {directory}")
    for name, expected in manifest["array_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Array checksum changed: {directory / name}")
    for split in ("train", "validation"):
        record = manifest["splits"][split]
        values = np.load(directory / f"{split}-X.npy", mmap_mode="r", allow_pickle=False)
        try:
            if values.shape != (len(record["image_ids"]), 4096) or values.dtype != np.float32:
                raise ValueError(f"Wrong shape or dtype in {directory}/{split}")
            if record["metadata"]["partition_code"] != (0 if split == "train" else 1):
                raise ValueError("Unexpected official partition code")
            for attribute in ATTRIBUTES:
                labels = np.load(directory / f"{split}-{attribute}-y.npy", allow_pickle=False)
                if not np.array_equal(labels, expected_labels[split, attribute]):
                    raise ValueError(f"Wrong labels in {directory}/{split}/{attribute}")
                facts = record["attributes"][attribute]
                if (data_fingerprint(values, labels) != facts["data_sha256"]
                        or np.bincount(labels, minlength=2).tolist() != facts["class_counts"]):
                    raise ValueError(f"Data fingerprint or class counts changed in {directory}")
        finally:
            _close_owned_array(values)
            del values
    return manifest


def prepare(images, attributes, partitions, validation_cache, output, *, recover_incomplete=False):
    if type(recover_incomplete) is not bool:
        raise TypeError("recover_incomplete must be a boolean.")
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the repository workspace")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".cache-preparation.lock"
    _confined(marker, output)
    with _file_lock(marker, f"Another process or thread is preparing this cache output: {output}"):
        return _prepare(images, attributes, partitions, validation_cache, output,
                        recover_incomplete=recover_incomplete)


def _prepare(images, attributes, partitions, validation_cache, output, *, recover_incomplete):
    start = time.perf_counter()
    images, attributes, partitions = (Path(path).resolve() for path in (images, attributes, partitions))
    validation_cache, output = Path(validation_cache).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the repository workspace")
    output.mkdir(parents=True, exist_ok=True)
    original_manifest = read(validation_cache / "manifest.json")
    validation = deepcopy(original_manifest["splits"]["validation"])
    if (original_manifest["official_test_loaded"] or original_manifest["seed"] != 42
            or validation["metadata"]["random_state"] != 42
            or validation["metadata"]["partition_code"] != 1 or len(validation["image_ids"]) != 4000
            or len(set(validation["image_ids"])) != 4000):
        raise ValueError("Expected the existing 4000-row official-validation seed42 cache")
    sources = source_identity(attributes, partitions)
    partition_map = _partitions(partitions)
    training_population = sorted(name for name, code in partition_map.items() if code == 0)
    if len(training_population) < max(SIZES):
        raise ValueError("Official training partition is too small")
    if not all(partition_map[name] == 1 for name in validation["image_ids"]):
        raise ValueError("Fixed validation IDs are outside the official validation partition")
    label_map = {attribute: _attribute_labels(attributes, attribute) for attribute in ATTRIBUTES}
    for attribute in ATTRIBUTES:
        if not all(name in label_map[attribute] for name in training_population):
            raise ValueError(f"Missing official-training labels for {attribute}")
    validation_files = {name: sha(validation_cache / name) for name in ARRAY_NAMES if name.startswith("validation-")}
    validation_values = np.load(validation_cache / "validation-X.npy", mmap_mode="r", allow_pickle=False)
    validation_labels = {}
    try:
        if validation_values.shape != (4000, 4096) or validation_values.dtype != np.float32:
            raise ValueError("Original validation preprocessing differs")
        for attribute in ATTRIBUTES:
            labels = np.load(validation_cache / f"validation-{attribute}-y.npy", allow_pickle=False)
            mapped = np.array([label_map[attribute][name] for name in validation["image_ids"]], dtype=np.int64)
            if (not np.array_equal(labels, mapped)
                    or data_fingerprint(validation_values, labels) != validation["attributes"][attribute]["data_sha256"]):
                raise ValueError(f"Original validation data/labels failed checks for {attribute}")
            validation_labels[attribute] = labels
    finally:
        _close_owned_array(validation_values)
        del validation_values
    common = {"schema_version": 1, "construction": POLICY, "source_files": sources,
              "image_directory": str(images), "validation_seed": 42, "validation_size": 4000,
              "original_validation_manifest_sha256": sha(validation_cache / "manifest.json"),
              "validation_array_sha256": validation_files,
              "loader_source_sha256": sha(ROOT / "src/supervised_nmf/datasets.py"),
              "numpy_version": np.__version__, "pillow_version": importlib.metadata.version("Pillow")}
    selected = {}
    summaries = []
    decode_calls = 0
    preserved_incomplete = []
    recovered_stages = []
    recovered_training_payloads = []
    for seed in SEEDS:
        choices = {size: np.sort(np.random.RandomState(seed).choice(len(training_population), size=size, replace=False))
                   for size in SIZES}
        for smaller, larger in zip(SIZES, SIZES[1:]):
            if not np.isin(choices[smaller], choices[larger]).all():
                raise ValueError("Current RandomState sampling is not nested; do not silently change the policy")
        selected[seed] = {size: [training_population[index] for index in choices[size]] for size in SIZES}
        largest_values = None
        largest_metadata = None
        decode_seconds = None
        try:
            for size in reversed(SIZES):
                expected_ids = selected[seed][size]
                directory = output / f"n{size}-seed{seed}"
                identity = {**common, "training_seed": seed, "training_size": size,
                            "training_population_size": len(training_population),
                            "global_selected_indices_sha256": hashlib.sha256(choices[size].tobytes()).hexdigest()}
                expected_labels = {("validation", attribute): validation_labels[attribute] for attribute in ATTRIBUTES}
                expected_labels.update({("train", attribute): np.array([label_map[attribute][name] for name in expected_ids], dtype=np.int64)
                                        for attribute in ATTRIBUTES})
                recovered_manifest = None
                if directory.exists():
                    preserved = _preserve_legacy_incomplete(directory, output, recover_incomplete)
                    if preserved is not None:
                        preserved_incomplete.append({"cache": directory.relative_to(ROOT).as_posix(),
                                                     "preserved": preserved.relative_to(ROOT).as_posix()})
                if not directory.exists():
                    recovered = _find_complete_stage(directory, identity, expected_ids, validation, expected_labels, output)
                    if recovered is not None:
                        stage, recovered_manifest = recovered
                        _publish_cache(stage / "c", directory, output)
                        recovered_stages.append({"cache": directory.relative_to(ROOT).as_posix(),
                                                 "stage": stage.relative_to(ROOT).as_posix()})
                if directory.exists():
                    manifest = (recovered_manifest if recovered_manifest is not None else
                                validate_cache(directory, identity, expected_ids, validation, expected_labels))
                    action = "recovered_complete_stage" if recovered_manifest is not None else "verified_existing"
                    if size == max(SIZES):
                        largest_values = np.load(directory / "train-X.npy", mmap_mode="r", allow_pickle=False)
                        largest_metadata = deepcopy(manifest["splits"]["train"]["metadata"])
                else:
                    payload_source = None
                    payload_sha = None
                    recovered_payload = None
                    if size == max(SIZES):
                        recovered_payload = _find_training_payload(directory, identity, expected_ids, output)
                    if recovered_payload is not None:
                        payload_stage, payload = recovered_payload
                        payload_source = payload_stage / "c" / "train-X.npy"
                        payload_sha = payload["array_sha256"]
                        largest_values = np.load(payload_source, mmap_mode="r", allow_pickle=False)
                        largest_metadata = deepcopy(payload["metadata"])
                        recovered_training_payloads.append({"cache": directory.relative_to(ROOT).as_posix(),
                            "stage": payload_stage.relative_to(ROOT).as_posix()})
                    elif size == max(SIZES):
                        print(f"Decoding {size} official-training images once for seed {seed}", flush=True)
                        tick = time.perf_counter()
                        loaded = load_celeba(images, attributes, partitions, attribute="Smiling", split="train",
                                             max_samples=size, random_state=seed, image_size=(64, 64))
                        decode_seconds = time.perf_counter() - tick
                        decode_calls += 1
                        if loaded.image_ids.tolist() != expected_ids:
                            raise ValueError("Loader sample differs from independently computed sample")
                        largest_values, largest_metadata = loaded.data, deepcopy(loaded.metadata)
                        del loaded
                    rows = np.searchsorted(choices[max(SIZES)], choices[size])
                    if not np.array_equal(choices[max(SIZES)][rows], choices[size]):
                        raise ValueError("Subsample mapping failed")
                    values = largest_values if size == max(SIZES) else largest_values[rows]
                    if source_identity(attributes, partitions) != sources:
                        raise ValueError("Source annotations/partitions changed during preparation")
                    stage = _new_stage(directory, identity, output)
                    cache = stage / "c"
                    if payload_source is None:
                        np.save(cache / "train-X.npy", values, allow_pickle=False)
                    else:
                        # Recovery only: preserve the old failed stage byte-for-byte.
                        shutil.copy2(payload_source, cache / "train-X.npy")
                    metadata = deepcopy(largest_metadata)
                    metadata.update(max_samples=size, random_state=seed)
                    if size == max(SIZES):
                        if payload_sha is None:
                            payload_sha = sha(cache / "train-X.npy")
                        _write_training_receipt(stage, directory, identity, expected_ids, metadata, payload_sha)
                    train = {"metadata": metadata, "image_ids": expected_ids, "attributes": {}}
                    for attribute in ATTRIBUTES:
                        labels = expected_labels["train", attribute]
                        np.save(cache / f"train-{attribute}-y.npy", labels, allow_pickle=False)
                        train["attributes"][attribute] = {"class_counts": np.bincount(labels, minlength=2).tolist(),
                                                          "data_sha256": data_fingerprint(values, labels)}
                    for name, expected in validation_files.items():
                        shutil.copy2(validation_cache / name, cache / name)
                        if sha(cache / name) != expected:
                            raise ValueError("Validation copy checksum mismatch")
                    manifest = {"author": "Chang Cheng", "seed": seed, "official_test_loaded": False,
                        "identity_overlap_checked": False, "splits": {"train": train, "validation": validation},
                        "created_utc": datetime.now(timezone.utc).isoformat(), "preparation_identity": identity,
                        "preparation": {"script_sha256": sha(Path(__file__)), "native_threads": 1, "processes": 1,
                            "direct_training_decode": size == max(SIZES) and recovered_payload is None,
                            "shared_largest_training_cache": f"n{max(SIZES)}-seed{seed}",
                            "direct_decode_seconds": decode_seconds if size == max(SIZES) else None,
                            "smaller_rows_match_independent_load_celeba_sampling": True,
                            "validation_copy_mode": "independent byte-identical file copies; no hardlinks",
                            "image_provenance_note": "Image pixels are fingerprinted after the unchanged loader preprocessing; raw JPEG file hashes are not collected."},
                        "array_sha256": {name: (payload_sha if name == "train-X.npy" and payload_sha is not None
                                                    else sha(cache / name)) for name in ARRAY_NAMES}}
                    write_new(cache / "manifest.json", manifest)
                    validate_cache(cache, identity, expected_ids, validation, expected_labels)
                    _publish_cache(cache, directory, output)
                    action = "created"
                    del values
                summary = {"seed": seed, "n_train": size, "cache": directory.relative_to(ROOT).as_posix(),
                           "manifest_sha256": sha(directory / "manifest.json"), "action": action,
                           "training_class_counts": {a: manifest["splits"]["train"]["attributes"][a]["class_counts"] for a in ATTRIBUTES},
                           "validation_class_counts": {a: validation["attributes"][a]["class_counts"] for a in ATTRIBUTES},
                           "training_validation_overlap": 0}
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
        finally:
            _close_owned_array(largest_values)
            largest_values = None
    if source_identity(attributes, partitions) != sources:
        raise ValueError("Source metadata changed during preparation")
    for name, expected in validation_files.items():
        if sha(validation_cache / name) != expected:
            raise ValueError("Original validation cache changed during preparation")
    overlaps = [{"size": size, "seeds": [left, right], "shared_training_ids": len(set(selected[left][size]) & set(selected[right][size]))}
                for index, left in enumerate(SEEDS) for right in SEEDS[index + 1:] for size in SIZES]
    report = {"author": "Chang Cheng", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "policy": POLICY, "seeds": list(SEEDS), "training_sizes": list(SIZES),
              "validation_size": 4000, "validation_seed": 42, "official_test_loaded": False,
              "all_within_seed_samples_nested": True, "source_files": sources,
              "preparation_script_sha256": sha(Path(__file__)), "runtime": provenance(1),
              "new_largest_sample_decode_calls": decode_calls, "elapsed_seconds": time.perf_counter() - start,
              "preserved_incomplete_caches": preserved_incomplete, "recovered_complete_stages": recovered_stages,
              "recovered_training_payloads": recovered_training_payloads,
              "array_bytes": sum((ROOT / row["cache"] / name).stat().st_size for row in summaries for name in ARRAY_NAMES),
              "caches": summaries, "cross_seed_training_overlaps": overlaps,
              "interpretation": "Matched nested training samples preserve natural prevalence. Different seeds may share training observations; every cache uses the same validation sample. This prepares data, not independent validation replications or model results."}
    output_report = output / f"preparation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    write_new(output_report, report)
    print(json.dumps({"report": str(output_report), "caches": len(summaries), "new_decode_calls": decode_calls,
                      "elapsed_seconds": report["elapsed_seconds"], "array_bytes": report["array_bytes"],
                      "cross_seed_training_overlaps": overlaps}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, default=ROOT / "data/celeba-development-4000-seed42")
    parser.add_argument("--output", type=Path, default=ROOT / "data/default-selection")
    parser.add_argument("--recover-incomplete", action="store_true",
                        help="Preserve legacy caches with absent/broken manifests under unique names, then rebuild; never repair committed integrity failures.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        prepare(args.images, args.attributes, args.partitions, args.validation_cache, args.output,
                recover_incomplete=args.recover_incomplete)


if __name__ == "__main__":
    main()
