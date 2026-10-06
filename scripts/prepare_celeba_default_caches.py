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
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from threadpoolctl import threadpool_limits

from supervised_nmf.datasets import _attribute_labels, _partitions, load_celeba
from supervised_nmf.evaluation import _digest, data_fingerprint, provenance

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
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity(attributes, partitions):
    return {"attributes": {"path": str(attributes), "sha256": sha(attributes)},
            "partitions": {"path": str(partitions), "sha256": sha(partitions)}}


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
    return manifest


def prepare(images, attributes, partitions, validation_cache, output):
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
    if validation_values.shape != (4000, 4096) or validation_values.dtype != np.float32:
        raise ValueError("Original validation preprocessing differs")
    validation_labels = {}
    for attribute in ATTRIBUTES:
        labels = np.load(validation_cache / f"validation-{attribute}-y.npy", allow_pickle=False)
        mapped = np.array([label_map[attribute][name] for name in validation["image_ids"]], dtype=np.int64)
        if (not np.array_equal(labels, mapped)
                or data_fingerprint(validation_values, labels) != validation["attributes"][attribute]["data_sha256"]):
            raise ValueError(f"Original validation data/labels failed checks for {attribute}")
        validation_labels[attribute] = labels
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
        for size in reversed(SIZES):
            expected_ids = selected[seed][size]
            directory = output / f"n{size}-seed{seed}"
            identity = {**common, "training_seed": seed, "training_size": size,
                        "training_population_size": len(training_population),
                        "global_selected_indices_sha256": hashlib.sha256(choices[size].tobytes()).hexdigest()}
            expected_labels = {("validation", attribute): validation_labels[attribute] for attribute in ATTRIBUTES}
            expected_labels.update({("train", attribute): np.array([label_map[attribute][name] for name in expected_ids], dtype=np.int64)
                                    for attribute in ATTRIBUTES})
            if directory.exists():
                if not (directory / "manifest.json").is_file():
                    raise ValueError(f"Incomplete cache exists; preserve and inspect it before choosing another output: {directory}")
                manifest = validate_cache(directory, identity, expected_ids, validation, expected_labels)
                action = "verified_existing"
                if size == max(SIZES):
                    largest_values = np.load(directory / "train-X.npy", mmap_mode="r", allow_pickle=False)
                    largest_metadata = deepcopy(manifest["splits"]["train"]["metadata"])
            else:
                if size == max(SIZES):
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
                directory.mkdir(exist_ok=False)
                np.save(directory / "train-X.npy", values, allow_pickle=False)
                metadata = deepcopy(largest_metadata)
                metadata.update(max_samples=size, random_state=seed)
                train = {"metadata": metadata, "image_ids": expected_ids, "attributes": {}}
                for attribute in ATTRIBUTES:
                    labels = expected_labels["train", attribute]
                    np.save(directory / f"train-{attribute}-y.npy", labels, allow_pickle=False)
                    train["attributes"][attribute] = {"class_counts": np.bincount(labels, minlength=2).tolist(),
                                                      "data_sha256": data_fingerprint(values, labels)}
                for name, expected in validation_files.items():
                    shutil.copy2(validation_cache / name, directory / name)
                    if sha(directory / name) != expected:
                        raise ValueError("Validation copy checksum mismatch")
                manifest = {"author": "Chang Cheng", "seed": seed, "official_test_loaded": False,
                    "identity_overlap_checked": False, "splits": {"train": train, "validation": validation},
                    "created_utc": datetime.now(timezone.utc).isoformat(), "preparation_identity": identity,
                    "preparation": {"script_sha256": sha(Path(__file__)), "native_threads": 1, "processes": 1,
                        "direct_training_decode": size == max(SIZES),
                        "shared_largest_training_cache": f"n{max(SIZES)}-seed{seed}",
                        "direct_decode_seconds": decode_seconds if size == max(SIZES) else None,
                        "smaller_rows_match_independent_load_celeba_sampling": True,
                        "validation_copy_mode": "independent byte-identical file copies; no hardlinks",
                        "image_provenance_note": "Image pixels are fingerprinted after the unchanged loader preprocessing; raw JPEG file hashes are not collected."},
                    "array_sha256": {name: sha(directory / name) for name in ARRAY_NAMES}}
                write_new(directory / "manifest.json", manifest)
                validate_cache(directory, identity, expected_ids, validation, expected_labels)
                action = "created"
                del values
            summary = {"seed": seed, "n_train": size, "cache": directory.relative_to(ROOT).as_posix(),
                       "manifest_sha256": sha(directory / "manifest.json"), "action": action,
                       "training_class_counts": {a: manifest["splits"]["train"]["attributes"][a]["class_counts"] for a in ATTRIBUTES},
                       "validation_class_counts": {a: validation["attributes"][a]["class_counts"] for a in ATTRIBUTES},
                       "training_validation_overlap": 0}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        del largest_values
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
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        prepare(args.images, args.attributes, args.partitions, args.validation_cache, args.output)


if __name__ == "__main__":
    main()
