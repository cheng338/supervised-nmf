"""Download verified Fashion-MNIST training data for Chang Cheng's study.

Only the official training archive is downloaded or decoded. The official test
files are deliberately absent from this preparation command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
ARCHIVES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archives = []
    for name, md5 in ARCHIVES.items():
        path = destination / name
        if not path.exists():
            temporary = destination / (name + ".part")
            with urllib.request.urlopen(SOURCE + name, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if digest(temporary, "md5") != md5:
                raise ValueError(f"Official archive MD5 mismatch: {name}")
            temporary.replace(path)
        if digest(path, "md5") != md5:
            raise ValueError(f"Existing archive MD5 mismatch: {name}")
        archives.append({"filename": name, "url": SOURCE + name, "md5": md5,
                         "sha256": digest(path), "bytes": path.stat().st_size})
    with gzip.open(destination / "train-images-idx3-ubyte.gz", "rb") as stream:
        header = struct.unpack(">IIII", stream.read(16))
        if header != (2051, 60000, 28, 28):
            raise ValueError(f"Unexpected image IDX header: {header}")
        pixels = np.frombuffer(stream.read(), dtype=np.uint8)
    if pixels.size != 60000 * 784:
        raise ValueError("Image payload length mismatch.")
    with gzip.open(destination / "train-labels-idx1-ubyte.gz", "rb") as stream:
        header = struct.unpack(">II", stream.read(8))
        if header != (2049, 60000):
            raise ValueError(f"Unexpected label IDX header: {header}")
        labels = np.frombuffer(stream.read(), dtype=np.uint8)
    if labels.size != 60000 or not np.array_equal(np.bincount(labels), np.full(10, 6000)):
        raise ValueError("Unexpected label count or class balance.")
    arrays = {"X_train.npy": pixels.reshape(60000, 784).astype(np.float32) / np.float32(255),
              "y_train.npy": labels}
    for name, values in arrays.items():
        path = destination / name
        if path.exists():
            if not np.array_equal(np.load(path, allow_pickle=False), values):
                raise ValueError(f"Existing cache differs from official archive: {name}")
        else:
            np.save(path, values, allow_pickle=False)
    manifest = {
        "schema_version": 1, "author": "Chang Cheng", "dataset": "Fashion-MNIST",
        "dataset_authors": ["Han Xiao", "Kashif Rasul", "Roland Vollgraf"],
        "citation": "https://arxiv.org/abs/1708.07747",
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_loaded": False, "official_test_downloaded": False,
        "archives": archives, "preprocessing": "Native 28x28 pixels flattened row-major; float32 / 255; no learned scaling, resizing, centering, or augmentation.",
        "shape": [60000, 784], "class_counts": np.bincount(labels).tolist(),
        "arrays": {name: {"sha256": digest(destination / name), "shape": list(values.shape),
                           "dtype": str(values.dtype)} for name, values in arrays.items()},
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "prepared_utc"} != {k: v for k, v in manifest.items() if k != "prepared_utc"}:
            raise ValueError("Existing cache manifest differs; preserve it and use a new destination.")
        manifest = old
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "cache": str(destination), "manifest_sha256": digest(manifest_path),
                      "official_test_loaded": False, "shape": manifest["shape"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "fashion-mnist-training")
    prepare(parser.parse_args().output)
