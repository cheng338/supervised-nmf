"""Small, reproducible simulation generators and a local-only CelebA reader.

The reader never downloads data. Simulation correlation parameters describe
Gaussian noise *before* clipping; observed correlations and moments will differ.
"""

from __future__ import annotations

import csv
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from sklearn.utils import Bunch


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or value < low or (high is not None and value > high):
        interval = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"{name} must be finite and in {interval}")
    return value


def _rng(random_state):
    # Deliberately do not use np.random.seed or accept a mutable RNG instance:
    # these helpers leave the caller's global/random-generator state untouched.
    if random_state is not None:
        random_state = _integer(random_state, "random_state", 0)
        if random_state > 2**32 - 1:
            raise ValueError("random_state must be <= 2**32 - 1")
    return np.random.RandomState(random_state)


def make_original_simulation(
    n_samples=6000, n_features=25000, signal_ratio=0.01, random_state=42
):
    """Reproduce the historical ``sim.py`` data using a local RandomState.

    Labels are zero for the first floor(n_samples / 2) rows and one thereafter.
    The first int(n_features * signal_ratio) columns are 1 + 2*y + N(0, .8²).
    Remaining columns are N(3, 9²). Concatenation, clipping at zero and float32
    conversion follow the original draw order exactly. Defaults allocate a
    large dataset; use smaller dimensions for development.

    Returns a Bunch with data, target, a boolean signal_support mask, and
    metadata. Reproduction refers to generated observations, not fitted-model
    results, which also depend on library versions and numerical settings.
    """
    n_samples = _integer(n_samples, "n_samples", 2)
    n_features = _integer(n_features, "n_features")
    signal_ratio = _real(signal_ratio, "signal_ratio", 0, 1)
    rng = _rng(random_state)
    target = np.zeros(n_samples, dtype=np.int32)
    target[n_samples // 2 :] = 1
    n_signal = int(n_features * signal_ratio)
    signal = 1.0 + (target * 2.0)[:, None] + rng.normal(
        0, 0.8, (n_samples, n_signal)
    )
    background = rng.normal(3.0, 9.0, (n_samples, n_features - n_signal))
    raw = np.hstack([signal, background])
    clipped_fraction = float(np.mean(raw < 0))
    data = np.clip(raw, a_min=0.0, a_max=None).astype(np.float32)
    support = np.arange(n_features) < n_signal
    return Bunch(
        data=data,
        target=target,
        signal_support=support,
        metadata={
            "generator": "original_occluded_additive",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_signal": n_signal,
            "signal_ratio": signal_ratio,
            "random_state": random_state,
            "signal_strength": 2.0,
            "signal_base_mean": 1.0,
            "signal_noise_std": 0.8,
            "background_mean": 3.0,
            "background_noise_std": 9.0,
            "correlation": 0.0,
            "clipping": "maximum(raw, 0), then float32",
            "clipped_fraction": clipped_fraction,
            "moment_scale": "specified means and standard deviations are pre-clipping",
        },
    )


def make_simulation(
    n_samples,
    n_features,
    n_signal,
    signal_strength,
    correlation,
    random_state,
    scenario="additive",
    *,
    correlation_scope="within_blocks",
):
    """Generate an additive or interaction-only classification problem.

    The first n_signal columns are the designated generating support. Each
    feature's standardized noise is sqrt(correlation)*g +
    sqrt(1-correlation)*e_j, with independent standard normal g and e_j.
    By default, correlation_scope='within_blocks' gives signal and background
    separate independent common factors g: noise correlation is correlation
    within either block and zero between them. 'all_features' shares one g
    across all columns for a signal/background-correlated stress case. These
    are pre-clipping noise correlations, not observed feature correlations.
    Signal columns have baseline 1 and noise SD .8, background columns baseline
    3 and noise SD 9. All values are clipped at zero and returned as float32.
    signal_strength is the raw shift delta; nominal_signal_to_noise metadata
    reports delta / .8, not a post-clipping standardized effect size.

    Additive labels have balanced counts (up to one row), randomly permuted;
    their signal shift is signal_strength*y. For XOR, independent fair bits
    a,b generate y=a XOR b, and two nonempty blocks of signal columns receive
    shifts signal_strength*a and signal_strength*b. Every individual feature
    has the same population distribution in both classes; the two blocks are
    jointly informative for positive signal_strength. Finite samples can have
    marginal associations. Zero strength defines a null problem, so the
    designated support then need not contain predictive information.
    """
    n_samples = _integer(n_samples, "n_samples", 2)
    n_features = _integer(n_features, "n_features")
    n_signal = _integer(n_signal, "n_signal", 0)
    if n_signal > n_features:
        raise ValueError("n_signal must not exceed n_features")
    signal_strength = _real(signal_strength, "signal_strength", 0)
    correlation = _real(correlation, "correlation", 0, 1)
    if scenario not in ("additive", "xor"):
        raise ValueError("scenario must be 'additive' or 'xor'")
    if correlation_scope not in ("within_blocks", "all_features"):
        raise ValueError("correlation_scope must be 'within_blocks' or 'all_features'")
    if scenario == "xor" and n_signal < 2:
        raise ValueError("XOR requires n_signal >= 2")
    rng = _rng(random_state)
    if scenario == "additive":
        target = np.zeros(n_samples, dtype=np.int32)
        target[n_samples // 2 :] = 1
        target = rng.permutation(target)
        shift = signal_strength * target[:, None]
        block_boundary = None
    else:
        bits = rng.randint(0, 2, size=(n_samples, 2))
        target = np.bitwise_xor(bits[:, 0], bits[:, 1]).astype(np.int32)
        block_boundary = n_signal // 2
        shift = signal_strength * np.concatenate(
            [
                np.repeat(bits[:, :1], block_boundary, axis=1),
                np.repeat(bits[:, 1:], n_signal - block_boundary, axis=1),
            ],
            axis=1,
        )
    if correlation_scope == "within_blocks":
        block_factors = rng.normal(size=(n_samples, 2))
        common = np.empty((n_samples, n_features))
        common[:, :n_signal] = block_factors[:, :1]
        common[:, n_signal:] = block_factors[:, 1:]
    else:
        common = rng.normal(size=(n_samples, 1))
    noise = np.sqrt(correlation) * common + np.sqrt(1 - correlation) * rng.normal(
        size=(n_samples, n_features)
    )
    raw = 3.0 + 9.0 * noise
    raw[:, :n_signal] = 1.0 + shift + 0.8 * noise[:, :n_signal]
    clipped_fraction = float(np.mean(raw < 0))
    data = np.maximum(raw, 0).astype(np.float32)
    return Bunch(
        data=data,
        target=target,
        signal_support=np.arange(n_features) < n_signal,
        metadata={
            "generator": "controlled_simulation",
            "scenario": scenario,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_signal": n_signal,
            "signal_strength": signal_strength,
            "nominal_signal_to_noise": signal_strength / 0.8,
            "correlation": correlation,
            "correlation_scope": correlation_scope,
            "correlation_definition": (
                "Gaussian noise correlation within signal/background blocks; zero between blocks; before clipping"
                if correlation_scope == "within_blocks" else
                "Gaussian noise correlation across all columns, before clipping"
            ),
            "signal_base_mean": 1.0,
            "signal_noise_std": 0.8,
            "background_mean": 3.0,
            "background_noise_std": 9.0,
            "random_state": random_state,
            "xor_block_boundary": block_boundary,
            "clipping": "maximum(raw, 0), then float32",
            "clipped_fraction": clipped_fraction,
            "moment_scale": "specified means and standard deviations are pre-clipping",
        },
    )


def _image_id(value, source):
    value = value.strip()
    if not value or value in (".", "..") or any(c in value for c in ("/", "\\", ":")):
        raise ValueError(f"Invalid image identifier {value!r} in {source}")
    return value


def _attribute_labels(path, attribute):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        stream.seek(0)
        if path.suffix.lower() == ".csv" or "," in first:
            reader = csv.reader(stream)
            header = next(reader, [])
            header = [name.strip() for name in header]
            if not header or header[0] not in ("image_id", "filename", "file_name", "image"):
                raise ValueError("Attribute CSV must start with an image_id or filename column")
            rows = list(reader)
            expected = None
        else:
            try:
                expected = int(stream.readline().strip())
            except ValueError as exc:
                raise ValueError("Official attribute text must begin with its row count") from exc
            if expected < 1:
                raise ValueError("Attribute row count must be positive")
            header = ["image_id"] + stream.readline().split()
            rows = [line.split() for line in stream if line.strip()]
        if len(header) != len(set(header)):
            raise ValueError("Duplicate attribute names")
        if attribute not in header[1:]:
            raise ValueError(f"Attribute {attribute!r} is missing from {path}")
        index = header.index(attribute)
        labels = {}
        for row_number, row in enumerate(rows, start=1):
            if len(row) != len(header):
                raise ValueError(f"Malformed attribute row {row_number} in {path}")
            identifier = _image_id(row[0], path)
            if identifier in labels:
                raise ValueError(f"Duplicate image identifier {identifier!r} in attributes")
            label = row[index].strip()
            if label not in ("-1", "0", "1"):
                raise ValueError(f"Invalid binary label {label!r} for {identifier}")
            labels[identifier] = int(label == "1")
        if expected is not None and len(labels) != expected:
            raise ValueError("Attribute row count does not match the official text header")
        if not labels:
            raise ValueError("Attribute file is empty")
        return labels


def _partitions(path):
    partitions = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = line.split()
            if len(row) != 2 or row[1] not in ("0", "1", "2"):
                raise ValueError(f"Invalid official partition row {line_number} in {path}")
            identifier = _image_id(row[0], path)
            if identifier in partitions:
                raise ValueError(f"Duplicate image identifier {identifier!r} in partitions")
            partitions[identifier] = int(row[1])
    if not partitions:
        raise ValueError("Partition file is empty")
    return partitions


def load_celeba(
    data_dir,
    attributes_path,
    partitions_path,
    attribute="Smiling",
    split="train",
    max_samples=None,
    random_state=0,
    image_size=(64, 64),
):
    """Load local CelebA images using their official partition and image IDs.

    Attributes may be official count/header/whitespace text or a CSV whose
    first column is image_id/filename. Labels -1 and 0 map to 0; 1 maps to 1.
    partitions_path must contain official ``filename partition`` rows, where
    0=train, 1=validation, 2=test. Partition metadata are required.

    Selection is uniform without replacement within the requested partition,
    without balancing classes; a subsample's realized prevalence may vary.
    All rows of the requested partition must have labels and image files,
    including rows not selected by max_samples. Other partitions' images need
    not be downloaded. Images follow the original Pillow ``convert('L').resize``
    preprocessing, then flatten and divide by 255 as float32. image_size is
    (width, height). Pillow is imported only when this function loads images.
    """
    split_codes = {"train": 0, "validation": 1, "val": 1, "test": 2}
    if split not in split_codes:
        raise ValueError("split must be 'train', 'validation'/'val', or 'test'")
    if max_samples is not None:
        max_samples = _integer(max_samples, "max_samples")
    try:
        if len(image_size) != 2:
            raise ValueError("image_size must contain width and height")
        size = tuple(_integer(value, "image_size dimension") for value in image_size)
    except TypeError as exc:
        raise ValueError("image_size must contain width and height") from exc
    rng = _rng(random_state)
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {data_dir}")
    attributes_path = Path(attributes_path)
    partitions_path = Path(partitions_path)
    labels = _attribute_labels(attributes_path, attribute)
    partitions = _partitions(partitions_path)
    identifiers = sorted(key for key, code in partitions.items() if code == split_codes[split])
    if not identifiers:
        raise ValueError(f"No images in requested partition {split!r}")
    for identifier in identifiers:
        if identifier not in labels:
            raise ValueError(f"Missing attribute label for partition image {identifier!r}")
        if not (data_dir / identifier).is_file():
            raise FileNotFoundError(f"Missing partition image: {data_dir / identifier}")
    partition_size = len(identifiers)
    if max_samples is not None and max_samples < partition_size:
        chosen = np.sort(rng.choice(partition_size, size=max_samples, replace=False))
        identifiers = [identifiers[index] for index in chosen]
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Loading CelebA images requires the optional Pillow dependency") from exc
    data = np.empty((len(identifiers), size[0] * size[1]), dtype=np.float32)
    for index, identifier in enumerate(identifiers):
        with Image.open(data_dir / identifier) as image:
            data[index] = np.asarray(image.convert("L").resize(size), dtype=np.float32).ravel() / 255.0
    return Bunch(
        data=data,
        target=np.array([labels[identifier] for identifier in identifiers], dtype=np.int64),
        image_ids=np.asarray(identifiers),
        metadata={
            "attribute": attribute,
            "split": "validation" if split == "val" else split,
            "partition_code": split_codes[split],
            "partition_size": partition_size,
            "random_state": random_state,
            "max_samples": max_samples,
            "image_size": size,
            "sampling": "uniform without replacement, no class balancing",
            "preprocessing": "Pillow L grayscale, resize, flatten, float32 / 255",
            "attributes_path": str(attributes_path),
            "partitions_path": str(partitions_path),
        },
    )
