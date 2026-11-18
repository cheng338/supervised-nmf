"""Joblib adapter for fitted checkpoints from a trusted local CelebA study.

This is a pickle-based loader: checksums and declared root types do NOT authenticate files
or make untrusted objects safe. Use only the caller's own generated study files
after its transaction, source, data and exact runtime checks. Never use this arm
to import a received/downloaded model. The runner enables this format through
its explicit all policy; the default is off.
"""
from __future__ import annotations

import hashlib
import io
import pickle
import numpy as np
import sklearn
from pathlib import Path
import re
import time

import joblib
from sklearn.decomposition import NMF, PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier

_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


CODEC = "celeba-final-fit-joblib-v1"
PAYLOAD_NAME = "model.joblib"
REGISTRY = {
    "nmf_classifier": NMFClassifier, "feature_spa": FeatureSPA,
    "binary_screening": BinaryScreening, "sklearn_nmf": NMF,
    "sklearn_pca": PCA, "sklearn_pipeline": Pipeline,
    "sklearn_dummy": DummyClassifier, "sklearn_knn": KNeighborsClassifier,
}
TYPE_NAMES = {value: key for key, value in REGISTRY.items()}
SUPPORT = {"trust": "owned-local-study-only", "format": "joblib-pickle",
           "knn_policy": "restore-saved-native-state", "protocol": 5, "compression": 0}


class CodecError(ValueError):
    """Owned checkpoint does not match this adapter's declared format."""


def _require(condition, message):
    if not condition:
        raise CodecError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _payload_hook(name, details):
    """Inert complete-file observation; this is not a partial-write boundary."""


def _open_payload_writer(path):
    """Ordinary writer; isolated tests may observe and flush actual write calls."""
    return Path(path).open("xb")


def _dump_complete(model, payload_dir):
    """Write only a new owned payload; leave partial evidence on failure."""
    _require(type(model) in TYPE_NAMES, "Root model is not a declared runner estimator.")
    directory = Path(payload_dir)
    _require(not directory.is_symlink(), "Owned payload directory cannot be a link.")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    _require(not any(directory.iterdir()), "Owned payload directory must be empty.")
    path = directory / PAYLOAD_NAME
    with _open_payload_writer(path) as stream:
        joblib.dump(model, stream, compress=0, protocol=5)
    _payload_hook("payload-written", {"path": str(path), "bytes": path.stat().st_size})
    return {"codec": CODEC, "root_class": TYPE_NAMES[type(model)],
            "joblib_version": joblib.__version__, "support": dict(SUPPORT),
            "payload": {"file": PAYLOAD_NAME, "file_bytes": path.stat().st_size, "sha256": _sha(path)}}


def _load_complete(payload_dir, metadata, *, stats=None):
    """Load a trusted own-run model after the store verifies its pinned identity.

    The final root-type check is a support assertion after loading, not a sandbox.
    No training or native KNN reconstruction is performed by this adapter.
    """
    started = time.perf_counter()
    _require(type(metadata) is dict and set(metadata) == {
        "codec", "root_class", "joblib_version", "support", "payload"}, "Invalid owned checkpoint metadata.")
    _require(metadata["codec"] == CODEC and metadata["support"] == SUPPORT,
             "Unsupported owned checkpoint format or policy.")
    _require(type(metadata["root_class"]) is str and metadata["root_class"] in REGISTRY,
             "Unsupported declared root estimator.")
    _require(metadata["joblib_version"] == joblib.__version__, "Joblib runtime version differs.")
    record = metadata["payload"]
    _require(type(record) is dict and set(record) == {"file", "file_bytes", "sha256"},
             "Invalid owned payload manifest.")
    _require(record["file"] == PAYLOAD_NAME and type(record["file_bytes"]) is int
             and record["file_bytes"] > 0 and type(record["sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
             "Invalid owned payload filename, size or checksum.")
    directory = Path(payload_dir)
    _require(not directory.is_symlink() and directory.is_dir(), "Owned payload directory is missing or linked.")
    directory = directory.resolve()
    path = directory / PAYLOAD_NAME
    _require(not path.is_symlink() and path.is_file(), "Owned payload must be a regular file.")
    _require(path.stat().st_size == record["file_bytes"] and _sha(path) == record["sha256"],
             "Owned payload failed its size or integrity check.")
    model = joblib.load(path, mmap_mode=None)
    _require(type(model) is REGISTRY[metadata["root_class"]], "Loaded root differs from its declared estimator.")
    if stats is not None:
        stats.update(reconstruction_seconds=0.0, reconstructed_knn_heads=0,
                     codec_load_seconds=time.perf_counter() - started)
    return model


def _cache_context(context, *, require_readonly=True):
    """Inspect a caller-verified cache handle without a second matrix hash.

    The runner main verifies actual cache fingerprints before aggregation.
    Independent callers must provide the same verified, immutable data boundary;
    declarations in this context do not authenticate arbitrary input arrays.
    """
    _require(type(context) is dict and set(context) == {"X", "y", "identity"},
             "Raw KNN recovery requires its verified training-cache context.")
    X, y, identity = context["X"], context["y"], context["identity"]
    keys = {"method", "attribute", "run_identity", "evaluation_identity",
            "manifest_sha256", "training_data_sha256"}
    _require(type(identity) is dict and set(identity) == keys
             and identity["method"] == "raw_knn"
             and type(identity["attribute"]) is str and bool(identity["attribute"]),
             "Raw KNN cache identity is incomplete or belongs to another method.")
    _require(all(type(identity[key]) is str and re.fullmatch(r"[0-9a-f]{64}", identity[key]) is not None
                 for key in keys - {"method", "attribute"}), "Invalid raw KNN cache identity digest.")
    _require(type(X) in {np.ndarray, np.memmap} and type(y) in {np.ndarray, np.memmap}
             and X.ndim == 2 and y.ndim == 1 and len(X) == len(y),
             "Raw KNN cache arrays must be ordered observations and labels.")
    _require(not require_readonly or (not X.flags.writeable and not y.flags.writeable),
             "Raw KNN recovery cache arrays must be read-only.")
    return X, y, dict(identity)

def _training_record(cache):
    X, y, _ = cache
    return {"shape": list(X.shape), "dtype": X.dtype.str, "strides": list(X.strides),
            "readonly": True, "y_shape": list(y.shape), "y_dtype": y.dtype.str}

def _check_training_record(record, cache):
    expected = _training_record(cache)
    _require(type(record) is dict and set(record) == set(expected), "Invalid raw KNN training metadata.")
    _require(all(type(record[name]) is list and len(record[name]) == size
                 and all(type(value) is int for value in record[name])
                 for name, size in (("shape", 2), ("strides", 2), ("y_shape", 1)))
             and type(record["dtype"]) is str and type(record["y_dtype"]) is str
             and record["readonly"] is True, "Invalid raw KNN training metadata types.")
    _require(record == expected, "Raw KNN recovery cache array layout differs.")

# Public protocol-5 buffers preserve the complete native model state.
# No model field is located, removed, replaced, or inferred from internals.
RAW_CODEC = "celeba-final-raw-knn-public-buffer-v1"
BUFFER_NAME = "model.pickle"
RAW_SUPPORT = {"trust": "owned-local-study-only", "format": "pickle5-cache-buffer",
               "knn_policy": "restore-saved-native-state", "protocol": 5,
               "external_buffer_policy": "one-exact-readonly-training-X-span"}


def _supported_raw_parameters(model):
    parameters = model.get_params(deep=False)
    wanted = {"algorithm": "auto", "leaf_size": 30, "metric": "minkowski", "metric_params": None,
              "n_jobs": 1, "p": 2, "weights": "uniform"}
    return (type(parameters) is dict and set(parameters) == set(wanted) | {"n_neighbors"}
            and all(type(parameters[key]) is type(value) and parameters[key] == value
                    for key, value in wanted.items())
            and type(parameters["n_neighbors"]) is int and parameters["n_neighbors"] > 0)


def _buffer_eligible(model, cache):
    if type(model) is not KNeighborsClassifier or cache is None:
        return False
    X, y, _ = cache
    return (_supported_raw_parameters(model) and not X.flags.writeable and not y.flags.writeable
            and X.dtype == np.float32 and X.dtype.isnative and X.flags.c_contiguous
            and X.shape[1] == 4096 and len(X) >= 2 and y.dtype == np.int64
            and np.array_equal(np.unique(y), np.array([0, 1], dtype=y.dtype)))


def _buffer_record(cache):
    return {"source": "training-X", "nbytes": cache[0].nbytes, "readonly": True}


def _raw_cache_buffer(cache):
    owned = np.array(cache[0], copy=True, order="C")
    owned.flags.writeable = False
    return memoryview(owned).cast("B")


def _write_buffer_pickle(payload_dir, encoded):
    directory = Path(payload_dir)
    _require(not directory.is_symlink(), "Owned payload directory cannot be a link.")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    _require(not any(directory.iterdir()), "Owned payload directory must be empty.")
    path = directory / BUFFER_NAME
    with _open_payload_writer(path) as stream:
        view = encoded.getbuffer()
        try:
            _require(stream.write(view) == view.nbytes, "Owned buffer payload write was incomplete.")
        finally:
            view.release()
    _payload_hook("payload-written", {"path": str(path), "bytes": path.stat().st_size})
    return {"file": BUFFER_NAME, "file_bytes": path.stat().st_size, "sha256": _sha(path)}


def _checked_buffer_path(payload_dir, record):
    _require(type(record) is dict and set(record) == {"file", "file_bytes", "sha256"},
             "Invalid owned buffer payload metadata.")
    _require(record["file"] == BUFFER_NAME and type(record["file_bytes"]) is int
             and record["file_bytes"] > 0 and type(record["sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
             "Invalid owned buffer filename, size or checksum.")
    directory = Path(payload_dir)
    _require(not directory.is_symlink() and directory.is_dir(), "Owned payload directory is missing or linked.")
    path = directory.resolve() / BUFFER_NAME
    _require(not path.is_symlink() and path.is_file(), "Owned buffer payload must be a regular file.")
    _require(path.stat().st_size == record["file_bytes"] and _sha(path) == record["sha256"],
             "Owned buffer payload failed its size or integrity check.")
    return path


def dump_model(model, payload_dir, *, context=None):
    """Serialize complete native state, with one verified cache buffer.

    Valid unsupported layouts and zero/multiple exact-span matches use the
    complete joblib representation. The discovery stream exists only in memory
    until the one-buffer contract is established; fallback leaves no partial
    files. No equal-value or arbitrary-model training provenance is inferred.
    """
    cache = _cache_context(context, require_readonly=False) if context is not None else None
    if not _buffer_eligible(model, cache):
        return _dump_complete(model, payload_dir)
    X = cache[0]
    address = X.__array_interface__["data"][0]
    external = []
    class UseCompleteRepresentation(Exception):
        """Private discovery decision, never a serialization error."""
    def choose_buffer(buffer):
        raw = buffer.raw()
        try:
            if (raw.readonly and raw.nbytes == X.nbytes
                    and np.frombuffer(raw, dtype=np.uint8).__array_interface__["data"][0] == address):
                external.append(_buffer_record(cache))
                if len(external) > 1:
                    raise UseCompleteRepresentation
                return False
            if raw.nbytes >= X.nbytes:
                raise UseCompleteRepresentation
            return True
        finally:
            raw.release()
    with io.BytesIO() as encoded:
        try:
            pickle.dump(model, encoded, protocol=5, buffer_callback=choose_buffer)
        except UseCompleteRepresentation:
            return _dump_complete(model, payload_dir)
        if len(external) != 1:
            return _dump_complete(model, payload_dir)
        payload = _write_buffer_pickle(payload_dir, encoded)
    return {"codec": RAW_CODEC, "root_class": "sklearn_knn", "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__, "support": dict(RAW_SUPPORT),
            "cache_identity": cache[2], "training": _training_record(cache),
            "external_buffers": external, "payload": payload}


def load_model(payload_dir, metadata, *, stats=None, context=None):
    """Restore complete native state through public pickle buffers, without fit.

    This owned-pickle format retains the verified immutable-cache lifetime
    requirement. Committed invalid state is rejected, never reinterpreted as a
    supported layout or silently rebuilt.
    """
    if not (type(metadata) is dict and metadata.get("codec") == RAW_CODEC):
        return _load_complete(payload_dir, metadata, stats=stats)
    started = time.perf_counter()
    _require(set(metadata) == {"codec", "root_class", "sklearn_version", "numpy_version", "support",
                              "cache_identity", "training", "external_buffers", "payload"},
             "Invalid committed public-buffer metadata.")
    _require(metadata["root_class"] == "sklearn_knn" and metadata["sklearn_version"] == sklearn.__version__
             and metadata["numpy_version"] == np.__version__ and metadata["support"] == RAW_SUPPORT,
             "Incompatible public-buffer type, runtime or policy.")
    cache = _cache_context(context)
    _require(metadata["cache_identity"] == cache[2], "Public-buffer training-cache identity differs.")
    _check_training_record(metadata["training"], cache)
    X, y, _ = cache
    _require(X.dtype == np.float32 and X.dtype.isnative and X.flags.c_contiguous and X.shape[1] == 4096
             and len(X) >= 2 and y.dtype == np.int64
             and np.array_equal(np.unique(y), np.array([0, 1], dtype=y.dtype)),
             "Unsupported committed public-buffer cache values or layout.")
    declarations = metadata["external_buffers"]
    _require(type(declarations) is list and len(declarations) == 1
             and type(declarations[0]) is dict and set(declarations[0]) == {"source", "nbytes", "readonly"}
             and type(declarations[0]["nbytes"]) is int and declarations[0]["readonly"] is True
             and declarations[0] == _buffer_record(cache), "Invalid declared external buffer.")
    path = _checked_buffer_path(payload_dir, metadata["payload"])
    consumed = []
    preparation_seconds = 0.0
    def buffers():
        nonlocal preparation_seconds
        began = time.perf_counter()
        raw = _raw_cache_buffer(cache)
        preparation_seconds = time.perf_counter() - began
        consumed.append(True)
        yield raw
    try:
        with path.open("rb") as stream:
            model = pickle.load(stream, buffers=buffers())
            _require(stream.read(1) == b"", "Owned buffer payload contains trailing data.")
    except CodecError:
        raise
    except Exception as error:
        raise CodecError(f"Cannot restore owned public-buffer model: {error}") from error
    _require(consumed == [True], "Owned payload did not consume its declared external buffer.")
    _require(type(model) is KNeighborsClassifier and _supported_raw_parameters(model),
             "Loaded root or public parameters differ from the supported raw estimator.")
    if stats is not None:
        stats.update(reconstruction_seconds=0.0, reconstructed_knn_heads=0,
                     cache_reference_restored=False, cache_binding_seconds=preparation_seconds,
                     buffer_preparation_seconds=preparation_seconds, buffer_preparation="copy",
                     codec_load_seconds=time.perf_counter()-started)
    return model


dump = dump_model
load = load_model
