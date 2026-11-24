"""Auditable classification comparisons derived from Chang Cheng's scripts.

Model selection uses only training folds. A candidate with even one failed or
nonconverged fold has no aggregate score and cannot win by averaging successes.
The reports deliberately leave refinement and package-default decisions open.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
import errno
import fnmatch
import hashlib
import importlib.metadata
from importlib.machinery import SOURCE_SUFFIXES
from inspect import getattr_static
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import re
import sys
import time
from types import FunctionType, MethodType
import uuid
import warnings

import numpy as np
from joblib import effective_n_jobs, parallel_config
from joblib.parallel import get_active_backend
from sklearn import get_config
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted, check_random_state
from threadpoolctl import threadpool_info, threadpool_limits

from .classifier import (NMFClassifier, _capture_projection, _record_warnings, _uses_projection_diagnostics,
                         _knn_probabilities_preserve_labels)
from .decomposition import FeatureSPA
from ._calibration import _score_fields, _score_predictions, _validate_scoring
from ._feature_names import _input_feature_names, _named_frame_backend, _named_matrix
from ._identity import _share_calibration_observation, _source_bytes, configuration_identity, implementation_sources
from ._labels import _check_labeled_data, _label_values, _validate_class_label, _validate_class_labels
from ._neighbors import _is_integer_index
from ._search_diagnostics import mark_inactive_refinement, screening_grid_diagnostics
from .screening import BinaryScreening, _clear_fitted_attributes
from .validation import (
    _NATIVE_ID_TYPES, _exact_metadata_promotion, _native_float_key_set, _metadata_keys, _metadata_overlap,
    _rank_count, _validate_metadata_snapshot, _validated_unique_metadata,
    _validate_refit, _validate_threads, build_cv_folds, validate_metadata,
)


# Capture the stock scalar class independently of the mutable public alias.
_LONGDOUBLE_SCALAR_TYPE = np.dtype("longdouble").type

class _TrackedLogisticRegression(LogisticRegression):
    """Retain fit convergence evidence for later standalone evaluation."""

    def fit(self, X, y, sample_weight=None):
        for attribute in ("convergence_warnings_", "converged_"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        fit = super().fit
        try:
            result, messages = _record_warnings(lambda: fit(X, y, sample_weight=sample_weight))
        except BaseException:
            _clear_fitted_attributes(self)
            raise
        self.convergence_warnings_ = messages
        self.converged_ = not messages
        return result


class _TrackedLinearSVC(LinearSVC):
    """Retain linear-SVM convergence evidence for standalone evaluation."""

    def fit(self, X, y, sample_weight=None):
        _clear_fitted_attributes(self)
        fit = super().fit
        try:
            result, messages = _record_warnings(lambda: fit(X, y, sample_weight=sample_weight))
        except BaseException:
            _clear_fitted_attributes(self)
            raise
        self.convergence_warnings_ = messages
        self.converged_ = not messages
        return result


def _extended_float_json(value):
    """Encode an unboxable NumPy float exactly, without narrowing its range."""
    if not np.isfinite(value):
        return None
    numerator, denominator = value.as_integer_ratio()
    if numerator == 0:
        significand = "-0" if np.signbit(value) else "0"
        exponent = 0
    else:
        # A binary float has a power-of-two denominator. Remove powers of two
        # from the numerator too, keeping even extreme values compact.
        magnitude = abs(numerator)
        shift = (magnitude & -magnitude).bit_length() - 1
        significand = str(numerator >> shift)
        exponent = shift - (denominator.bit_length() - 1)
    return {"__supervised_nmf_scalar__": "longdouble-v1",
            "significand": significand, "exponent": exponent}


def _jsonable(value):
    value_type = type(value)
    if value is None or value_type is str or value_type is int or value_type is bool:
        return value
    if value_type is float:
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        item = value if value_type is _LONGDOUBLE_SCALAR_TYPE else value.item()
        if not isinstance(item, np.generic):
            return _jsonable(item)
        # longdouble/clongdouble cannot always unbox to a Python scalar.
        # Recursing through .item() again would never terminate.
        if isinstance(item, np.floating):
            return _extended_float_json(item)
        if isinstance(item, np.complexfloating):
            return {"__supervised_nmf_scalar__": "clongdouble-v1",
                    "real": _extended_float_json(item.real),
                    "imag": _extended_float_json(item.imag)}
        return repr(item)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value_type is list and len(value) >= 16 and type(next(iter(value), None)) is int:
        return [v if type(v) is int else _jsonable(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return repr(value)


@contextmanager
def _file_lock(marker_path, busy_message):
    """Own a persistent marker through its OS lock, never through its existence."""
    with Path(marker_path).open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise RuntimeError(busy_message) from error
            raise
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _checkpoint_file(path):
    """Own one checkpoint destination for its complete read/modify/write cycle."""
    if path is None:
        yield None
        return
    path = Path(path)
    # Normalize parent aliases without changing final-symlink replace semantics.
    path = path.parent.resolve() / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(f".{path.name}.lock")
    with _file_lock(marker, f"Another process or thread is writing this checkpoint: {path}"):
        yield path


def write_json(path, report):
    """Atomically replace a complete report; concurrent replacements are not merged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(report), indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    temporary = None
    try:
        for _ in range(10):
            candidate = path.parent / f".snmf-{uuid.uuid4().hex}.tmp"
            try:
                stream = candidate.open("x", encoding="utf-8")
            except FileExistsError:
                continue
            temporary = candidate
            break
        else:
            raise FileExistsError(f"Could not create a unique temporary file for {path}")
        with stream:
            stream.write(payload)
        # Close before replacement, retaining the existing bounded Windows retries.
        # Never delete the old valid checkpoint first.
        for attempt in range(6):
            try:
                temporary.replace(path)
                temporary = None
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * 2 ** attempt)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Preserve the original I/O error if cleanup is also denied.
                pass


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def save_data_manifest(path, manifest):
    """Create or verify a manifest while rejecting a competing writer promptly."""
    path = Path(path)
    with _checkpoint_file(path) as path:
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != _jsonable(manifest):
                raise ValueError("Existing data manifest is incompatible; create a new run directory.")
        else:
            write_json(path, manifest)


def data_fingerprint(X, y):
    """Hash values, ordering, shape, dtype and recognized ordered column names."""
    feature_names = _input_feature_names(X)
    return _data_fingerprint(X, y, feature_names)


def _data_fingerprint(X, y, feature_names):
    arrays = np.asarray(X), np.asarray(y)
    _validate_class_labels(y, arrays[1])
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        if array.dtype.hasobject or array.dtype.kind == "T":
            digest.update(json.dumps(_jsonable(array), sort_keys=True).encode())
        else:
            contiguous = np.ascontiguousarray(array)
            digest.update(memoryview(contiguous).cast("B"))
    value_digest = digest.hexdigest()
    if feature_names is None:
        return value_digest
    return _digest(["ordered-feature-names-v2", value_digest, _schema_digest(feature_names)])


def _schema_digest(names):
    if names is None:
        return None
    # JSON escaping alone conflates an astral code point with the two literal
    # Python surrogate code points that spell its UTF-16 encoding.
    digest = hashlib.sha256()
    for name in names:
        encoded = str.encode(name, "utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _check_matching_columns(expected, observed):
    # NumPy can strip trailing NULs when it infers a string dtype for a tuple.
    if expected is not None and observed is not None and tuple(expected) != tuple(observed):
        raise ValueError("Feature names must match the training columns in the same order.")


def _active_steps(model):
    return [step for _, step in model.steps if step is not None
            and not (type(step) is str and step == "passthrough")]


def _has_input_override(model):
    return any(name in getattr(model, "__dict__", {}) for name in (
        "fit", "fit_transform", "transform", "predict", "predict_proba",
        "decision_function", "predict_with_probabilities", "predict_with_decision_function",
        "kneighbors"))


def _numeric_schema_estimator(model, *, fitted=False):
    """Can the raw consumer accept a canonical numeric table without changing callbacks?"""
    if model is None:
        return True
    if type(model) is str:
        return model in ("drop", "passthrough")
    if _has_input_override(model):
        return False
    if type(model) is SimpleImputer:
        return type(model.strategy) is str
    if type(model) is Pipeline:
        steps = _active_steps(model)
        if not steps or not _numeric_schema_estimator(steps[0], fitted=fitted):
            return False
        # An ordinary numeric transform returns the same positional matrix to
        # downstream estimators and callbacks, including unlisted stock heads.
        # A named output can expose its changed column labels to a callback.
        if len(steps) == 1:
            return True
        first = steps[0]
        output = getattr(first, "_sklearn_output_config", {}).get("transform", get_config()["transform_output"])
        return ((type(first) is not Pipeline and output == "default")
                or all(_numeric_schema_estimator(step, fitted=fitted) for step in steps[1:]))
    if type(model) is ColumnTransformer:
        # ColumnTransformer resolves a callable to its chosen columns during
        # fit. Transform uses those stored columns and does not call it again.
        if fitted and any(_has_pipeline_input_override(step) for _, step, _ in getattr(model, "transformers_", ())):
            return False
        return (_numeric_schema_estimator(model.remainder, fitted=fitted)
                and all((fitted or not callable(columns)) and _numeric_schema_estimator(step, fitted=fitted)
                        for _, step, columns in model.transformers))
    return type(model) in (NMFClassifier, BinaryScreening, FeatureSPA, NMF, PCA, StandardScaler, MinMaxScaler,
                           KNeighborsClassifier, LogisticRegression, LinearSVC,
                           _TrackedLogisticRegression, _TrackedLinearSVC, DummyClassifier)


def _fitted_input_names(model):
    # Pipeline's public property proxies steps[0], which may simply pass
    # through X. The first active stage owns the raw input's actual schema.
    while isinstance(model, Pipeline):
        model = next(iter(_active_steps(model)), None)
    return getattr(model, "feature_names_in_", None)


def _has_named_pipeline_stage(model):
    if isinstance(model, ColumnTransformer):
        # A fitted selector can require a DataFrame even when all its output
        # and the downstream classifier's input are positional.
        return getattr(model, "feature_names_in_", None) is not None
    if not isinstance(model, Pipeline):
        return False
    steps = _active_steps(model)
    # A schema on the first raw consumer is checked explicitly. Look for
    # additional names that can only be checked after actual transformation.
    return bool(steps) and (_has_named_pipeline_stage(steps[0]) or any(
        _fitted_input_names(step) is not None or _has_named_pipeline_stage(step) for step in steps[1:]))


def _has_pipeline_input_override(model):
    if _has_input_override(model):
        return True
    if type(model) is ColumnTransformer:
        return any(_has_pipeline_input_override(step) for _, step, _ in getattr(model, "transformers_", ()))
    if not isinstance(model, Pipeline):
        return False
    if any(getattr_static(type(model), name, None) is not getattr_static(Pipeline, name, None)
           for name in ("predict", "predict_proba", "decision_function")):
        return True
    return any(_has_pipeline_input_override(step) for step in _active_steps(model))


def provenance(threads=1):
    """Record software, hardware, package code, and requested native thread limits.

    The source digest map includes adjacent bytecode modules when their
    Python source is absent. Ordinary interpreter cache directories are ignored.
    """
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "joblib", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Path.glob suppresses directory errors, which could admit an empty source identity.
    module_file = Path(__file__)
    package_dir = module_file.parent
    entries = os.listdir(package_dir)
    names = [name for suffix in SOURCE_SUFFIXES for name in fnmatch.filter(entries, "*" + suffix)]
    bytecode = fnmatch.filter(entries, "*.pyc")
    if bytecode:
        # Only an exact source spelling establishes shadowing on a
        # case-sensitive directory, including such directories on Windows.
        source_names = set(names)
        names.extend(name for name in bytecode
                     if package_dir / name == module_file
                     or not any(name[:-4] + suffix in source_names for suffix in SOURCE_SUFFIXES))
    sources = sorted(package_dir / name for name in names)
    suffix = os.path.normcase(module_file.suffix)
    if (suffix in SOURCE_SUFFIXES or suffix == ".pyc") and module_file.name not in names:
        if not any(module_file.samefile(path) for path in sources):
            raise FileNotFoundError(errno.ENOENT, "Running module is missing from the package source manifest", module_file)
    source_hashes = {p.name: hashlib.sha256(_source_bytes(p)).hexdigest() for p in sources}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "versions": versions, "source_sha256": source_hashes,
            "native_threads_requested": threads, "threadpools": threadpool_info()}


def _initialized_provenance(threads):
    # A first BLAS operation can load a previously unseen OpenMP library.
    # Stabilize that inventory before freezing it, without using study data.
    with threadpool_limits(limits=threads):
        probe = np.eye(4)
        probe @ probe
        return provenance(threads)


def new_run_directory(root, prefix="development"):
    """Create a unique run directory; historical runs are never overwritten."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _warning_records(caught):
    return [{"category": w.category.__name__, "message": str(w.message)}
            for w in caught]


def _diagnostics(estimator, projection=None):
    result = {}
    call_local = _uses_projection_diagnostics(estimator)
    for attribute in ("n_iter_", "converged_", "convergence_warnings_",
                      "last_transform_converged_", "last_transform_warnings_", "last_transform_seconds_", "last_transform_error_",
                      "factorization_converged_", "factorization_convergence_warnings_",
                      "training_projection_converged_", "training_projection_convergence_warnings_", "training_projection_performed_",
                      "classifier_n_iter_", "classifier_converged_", "classifier_convergence_warnings_",
                      "knn_normalization_", "training_coordinates_mode_",
                      "decision_threshold_", "positive_label_", "positive_class_index_", "mu_projection_",
                      "fit_times_", "n_features_in_"):
        if attribute.startswith("last_transform_") and call_local:
            continue
        if hasattr(estimator, attribute):
            result[attribute.rstrip("_")] = _jsonable(getattr(estimator, attribute))
    if call_local:
        # Fit-only records have no prediction to qualify. Old serialized models
        # may retain obsolete last-call fields; never reuse those as evidence.
        result.update(_jsonable(projection) if projection is not None else {
            "last_transform_performed": False, "last_transform_converged": None,
            "last_transform_warnings": [], "last_transform_error": None,
            "last_transform_seconds": 0.0})
    # Constructor settings may have changed since fit. Identify the actual
    # fitted head, including an NMF head's optional scaler pipeline.
    head = estimator.steps[-1][1] if isinstance(estimator, Pipeline) and estimator.steps else estimator
    head = getattr(head, "classifier_", head)
    if isinstance(head, Pipeline) and head.steps:
        head = head.steps[-1][1]
    if hasattr(head, "classes_"):
        result["fitted_classifier_type"] = f"{type(head).__module__}.{type(head).__qualname__}"
    # Estimators and comparison Pipelines expose their selectors differently.
    selector = getattr(estimator, "selector_", None)
    if selector is None:
        selector = getattr(estimator, "screening_", None)
    if hasattr(estimator, "named_steps"):
        selector = estimator.named_steps.get("screen", selector)
    if isinstance(estimator, Pipeline) and estimator.steps:
        final_classifier = estimator.steps[-1][1]
        if hasattr(final_classifier, "n_iter_"):
            result["classifier_n_iter"] = _jsonable(final_classifier.n_iter_)
        if hasattr(final_classifier, "converged_"):
            result["classifier_converged"] = bool(final_classifier.converged_)
        if hasattr(final_classifier, "convergence_warnings_"):
            result["classifier_convergence_warnings"] = _jsonable(final_classifier.convergence_warnings_)
    if selector is not None and hasattr(selector, "support_"):
        indices = np.flatnonzero(selector.get_support())
        result["n_selected_features"] = len(indices)
        result["selected_feature_indices"] = indices.tolist()
        for name in ("n_constant_features_", "n_infinite_scores_"):
            if hasattr(selector, name):
                result[name.rstrip("_")] = int(getattr(selector, name))
    elif hasattr(estimator, "n_features_in_"):
        result["n_selected_features"] = int(estimator.n_features_in_)
    decomposition = getattr(estimator, "decomposition_", None)
    if decomposition is not None:
        result["decomposition"] = {
            attribute.rstrip("_"): _jsonable(getattr(decomposition, attribute))
            for attribute in ("n_iter_", "reconstruction_err_", "converged_",
                              "anchor_condition_", "original_anchor_condition_", "basis_condition_")
            if hasattr(decomposition, attribute)
        }
    return result


def _nonconverged(estimator, caught, projection=None):
    final_classifier = estimator.steps[-1][1] if isinstance(estimator, Pipeline) and estimator.steps else None
    if _uses_projection_diagnostics(estimator):
        projection_failed = (projection is not None
                             and projection.get("last_transform_converged") is False)
    else:
        projection_failed = not bool(getattr(estimator, "last_transform_converged_", True))
    return (any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or not bool(getattr(estimator, "converged_", True))
            or not bool(getattr(estimator, "classifier_converged_", True))
            or not bool(getattr(final_classifier, "converged_", True))
            or projection_failed)


def support_recovery(estimator, signal_support):
    """Keep empirical signal-selection recovery distinct from classification.

    No separability or population-support guarantee follows from these counts.
    For an unscreened method every input feature is retained; this is explicitly
    labeled and has no interpretation as successful feature identification.
    """
    truth = np.asarray(signal_support)
    if truth.dtype != np.bool_ or truth.ndim != 1:
        raise ValueError("signal_support must be a one-dimensional boolean mask.")
    selector = getattr(estimator, "selector_", None)
    if hasattr(estimator, "named_steps"):
        selector = estimator.named_steps.get("screen", selector)
    if selector is not None and hasattr(selector, "support_"):
        retained = np.asarray(selector.get_support(), dtype=bool)
    else:
        retained = np.ones(int(estimator.n_features_in_), dtype=bool)
    if len(retained) != len(truth):
        raise ValueError("signal_support does not match the fitted input feature count.")
    true_selected = int(np.sum(retained & truth))
    false_selected = int(np.sum(retained & ~truth))
    total_signal = int(np.sum(truth))
    total_selected = int(np.sum(retained))
    return {"screening_applied": selector is not None,
            "true_signal_features": total_signal, "selected_features": total_selected,
            "true_selections": true_selected, "false_selections": false_selected,
            "missed_signal_features": total_signal - true_selected,
            "support_precision": true_selected / total_selected if total_selected else None,
            "support_recall": true_selected / total_signal if total_signal else None,
            "full_support_retained": bool(np.all(retained[truth])),
            "exact_support_recovered": bool(np.array_equal(retained, truth))}


def _stock_split_rng(rng):
    return (type(rng) is np.random.RandomState and any(type(rng._bit_generator) is stock for stock in
            (np.random.MT19937, np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64)))


def _private_split_rng(seed):
    return _stock_split_rng(seed) and seed._bit_generator is not np.random.get_bit_generator()


def _uses_global_rng(estimator, candidates, limits):
    if "ambient fit RNG" in limits:
        # Even if cloning isolates each fitted estimator, it copies the state
        # at that fold's position in the process stream. Callbacks can retain
        # the original object and advance it directly.
        return True
    if "implicit RNG" not in limits:
        return False
    cls = type(estimator)
    if cls is not DummyClassifier and cls is not NMFClassifier:
        # Recognize the exact public class if it was already imported.
        tree = sys.modules.get("sklearn.tree")
        if tree is None or cls is not vars(tree).get("ExtraTreeClassifier"):
            return True
    # Exempt only ordinary configurations whose stock fit and prediction paths
    # do not consult the ambient RNG. A grid can replace individual settings.
    parameters = vars(estimator)
    for options in (parameters, *candidates):
        if any(type(key) is not str or key not in parameters or
               (value is not None and not any(type(value) is kind for kind in (int, float, bool, str)))
               for key, value in options.items()):
            return True
    if all(type(p.get("random_state", estimator.random_state)) is int for p in candidates):
        return False
    if cls is not DummyClassifier and cls is not NMFClassifier:
        return True
    if type(estimator) is DummyClassifier:
        return any(p.get("strategy", estimator.strategy) not in ("most_frequent", "prior", "constant")
                   for p in candidates)
    return any(p.get("solver", estimator.solver) != "spa" or
               p.get("classifier", estimator.classifier) != "knn" for p in candidates)


def _ambient_mt_token():
    """Identify the legacy, nonspawning NumPy singleton without advancing it."""
    engine = np.random.get_bit_generator()
    if type(engine) is not np.random.MT19937:
        return None
    if getattr(engine, "seed_seq", getattr(engine, "_seed_seq", 1)) is not None:
        return None
    _, keys, position, cached, gaussian = np.random.get_state()
    token = hashlib.sha256(keys.astype("<u4", copy=False).tobytes())
    token.update(position.to_bytes(4, "little"))
    token.update(bytes([cached]))
    token.update(np.asarray(gaussian, dtype="<f8").tobytes())
    return token.hexdigest()


def _output_library_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _named_input_identity():
    backend = _named_frame_backend()
    return {"container": backend.__name__, "version": backend.__version__}


def _check_transform_output(expected):
    if get_config()["transform_output"] != expected:
        raise ValueError("Scikit-learn transform output changed during the search or comparison.")


def _search_identity(estimator, candidates, X, y, cv, seed, threads, refit=True,
                     *, groups=None, folds=None, data_ids=None, scoring="accuracy",
                     _environment=None, _observation=None, _split_limits=None, _fit_rng_dependencies=None,
                     _feature_names=None):
    environment = _initialized_provenance(threads) if _environment is None else _environment(threads)
    runtime_keys = ("user_api", "internal_api", "prefix", "filepath", "version", "threading_layer", "architecture")
    numerical_runtime = sorted([{key: library.get(key) for key in runtime_keys}
                                for library in environment["threadpools"]],
                               key=lambda library: library.get("filepath") or "")
    nonstandard_text = [] if _observation is not None else None
    feature_names = _input_feature_names(X) if _feature_names is None else _feature_names
    transform_output = get_config()["transform_output"]
    output_libraries = {transform_output} if transform_output in ("pandas", "polars") else set()
    identity = {"identity_schema": 2,
            "estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            # Candidate reports retain display descriptions. Keep those out of
            # identity: repr may omit parameters or contain process addresses.
            "data_sha256": _data_fingerprint(X, y, feature_names),
            "structural_configuration": configuration_identity({"estimator": estimator, "candidates": candidates},
                                                              _nonstandard_text=nonstandard_text,
                                                              _split_limits=_split_limits, _split_seed=seed,
                                                              _fit_rng_dependencies=_fit_rng_dependencies,
                                                              _output_libraries=output_libraries),
            "implementation_source": implementation_sources(estimator, candidates, _observation=_observation,
                                                             _nonstandard_text=nonstandard_text, _split_seed=seed),
            "cv": cv, "seed": seed, "threads": threads, "refit": refit,
            "sklearn_skip_parameter_validation": get_config()["skip_parameter_validation"],
            "sklearn_transform_output": transform_output,
            "scoring": _validate_scoring(scoring),
            "groups": _jsonable(groups), "folds": _jsonable(folds), "data_ids": _jsonable(data_ids),
            "python": environment["python"], "platform": environment["platform"],
            "numerical_runtime": numerical_runtime,
            "software": environment["versions"],
            "source_sha256": environment["source_sha256"]}
    if feature_names is not None:
        identity["named_input"] = _named_input_identity()
    if output_libraries:
        identity["output_library_versions"] = {name: _output_library_version(name)
                                              for name in sorted(output_libraries)}
    if _observation is not None and not nonstandard_text:
        _share_calibration_observation(identity["structural_configuration"], _observation)
    return identity


def _rank_coordinate(value):
    """Return an exact comparable coordinate when the setting supplies one."""
    if type(value) is _LONGDOUBLE_SCALAR_TYPE:
        return _metadata_keys([value])[0] if np.isfinite(value) else None
    try:
        if _is_integer_index(value):
            return _rank_count(value)
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if math.isfinite(number) else None
    except (MemoryError, RecursionError):
        raise
    except Exception:
        # Advisory bounds must not discard a completed fit when the parameter's
        # own conversion method refuses to provide a numerical coordinate.
        return None
    return None


def _wide_builtin_integer(value):
    """Recognize stock integers outside binary64's consecutive integer range."""
    cls = type(value)
    return ((cls is int or cls is np.int64 or cls is np.uint64 or cls is np.longlong or cls is np.ulonglong)
            and int(value).bit_length() > 53)


def refinement_diagnostics(candidates, selected_params, *, _rank_coordinates=None):
    """Flag range boundaries and interior intervals; this is not completed tuning."""
    if selected_params is None:
        return {"status": "pending_valid_candidate", "parameters": {}}
    numeric_types = (int, float, np.number)
    boolean_types = (bool, np.bool_)
    output = {}
    for parameter, selected in selected_params.items():
        if _rank_coordinates and parameter in _rank_coordinates:
            axis = _rank_coordinates[parameter]
            if axis is None or (coordinate := _rank_coordinate(axis[1])) is None:
                output[parameter] = {"status": "categorical", "selected": selected}
                continue
            declared, chosen = axis
            selected = coordinate if _is_integer_index(chosen) else chosen
            includes_none = any(value is None for value in declared)
            bounds = {}
            for value in declared:
                point = coordinate if value is chosen else _rank_coordinate(value)
                if point is None:
                    continue
                if type(value) is _LONGDOUBLE_SCALAR_TYPE and type(point) is not float:
                    # Keep wide integers and fractions in the compact extended-float encoding.
                    display = value
                elif type(point) is int:
                    display = point if point.bit_length() > 53 else float(point)
                else:
                    display = point
                bounds.setdefault(point, display)
            points = sorted(bounds)
            values = [bounds[p] for p in points]
            if len(values) > 1:
                location = points.index(coordinate)
        else:
            values = [c[parameter] for c in candidates if parameter in c]
            if not isinstance(selected, numeric_types) or isinstance(selected, boolean_types):
                output[parameter] = {"status": "categorical", "selected": selected}
                continue
            includes_none = any(value is None for value in values)
            values = sorted(set(int(v) if _wide_builtin_integer(v) else float(v) for v in values
                                if isinstance(v, numeric_types) and not isinstance(v, boolean_types)))
            if len(values) > 1:
                location = values.index(int(selected) if _wide_builtin_integer(selected) else float(selected))
        if len(values) < 2:
            output[parameter] = {"status": "single_value_requires_exploration", "selected": selected}
        else:
            status = ("lower_boundary_requires_expansion_or_constraint" if location == 0
                      else "upper_boundary_requires_expansion_or_constraint" if location == len(values) - 1
                      else "interior_interval_requires_refinement")
            output[parameter] = {"status": status, "selected": selected, "values": values}
        if includes_none:
            output[parameter]["includes_none"] = True
    return {"status": "pending_refinement_and_replication", "parameters": output}


class CalibrationFallbackWarning(UserWarning):
    """A calibration search substituted a training-majority classifier."""


@dataclass
class CalibrationResult:
    estimator: object
    report: dict
    # Positional fits keep their source labels, or an empty tuple for unnamed input.
    _positional_input_columns: tuple | None = field(default=None, repr=False, compare=False)


@dataclass
class _ComparisonRecovery:
    unverified_prefix: bool
    loaded_checkpoint: bool = False
    serial_final: bool = False
    batch: _ComparisonBatch | None = None
    refit: Callable[[], CalibrationResult] | None = None


@dataclass
class _ComparisonBatch:
    locks: ExitStack
    pending: deque[Callable[[], None]]
    root: Path | None = None
    studies: dict = field(default_factory=dict)
    evaluations: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    closed: bool = False

    def declare(self, root, directories):
        self.root = Path(root).resolve()
        self.studies = {Path(d).resolve().relative_to(self.root).as_posix(): {} for d in directories}
        seal = self.root / "comparison-cv-closed.json"
        if not seal.exists():
            if (self.root / "replications.json").exists() or any(
                    _has_serial_finals(self.root / key) for key in self.studies):
                raise ValueError("Existing comparison folds are incompatible or lack their closed CV record.")
            return
        envelope = json.loads(seal.read_bytes())
        saved = envelope["studies"]
        if envelope.get("sha256") != _digest(saved) or set(saved) != set(self.studies):
            raise ValueError("The serial comparison CV seal is incompatible with these study directories.")
        # Inspect all replications, including those processed later in the
        # launcher, before any partition, CV checkpoint or evaluation write.
        for key, pins in saved.items():
            directory = self.root / key
            evaluations = self.evaluations[key] = {}
            if pins:
                _check_sealed_partition(directory, pins)
            for name, pin in pins.items():
                path = directory / f"{name}-cv.json"
                if not path.is_file():
                    raise ValueError(f"Missing or changed CV checkpoint for {name} after final admission.")
                self.locks.enter_context(_checkpoint_file(path))
                report = json.loads(path.read_bytes())
                if _cv_pin(report) != pin["cv"]:
                    raise ValueError(f"Missing or changed CV checkpoint for {name} after final admission.")
                evaluation_path = directory / f"{name}-evaluation.json"
                if evaluation_path.exists():
                    record = json.loads(evaluation_path.read_bytes())
                    if (not isinstance(record, dict) or report["state"] != "completed"
                            or record.get("model") != name or record.get("fold_plan_sha256") != pin["partition"]
                            or record.get("cv_checkpoint_sha256") != report["checkpoint_sha256"]
                            or record.get("evaluation_sha256") != _digest({
                                k: v for k, v in record.items() if k != "evaluation_sha256"})):
                        raise ValueError(f"Serial evaluation failed its checkpoint or integrity check: {evaluation_path}")
                    evaluations[name] = record
        self.studies = saved
        self.closed = True

    def register(self, directory, models):
        if self.root is None:
            self.declare(directory, [directory])
        key = directory.resolve().relative_to(self.root).as_posix()
        if key not in self.studies or (self.closed and set(models) != set(self.studies[key])):
            raise ValueError("Existing comparison folds are incompatible with these methods or directories.")
        return self.studies[key], self.evaluations.get(key, {})

def _cv_pin(report):
    if report.get("checkpoint_sha256") != _digest({k: v for k, v in report.items() if k != "checkpoint_sha256"}):
        raise ValueError("The sealed CV checkpoint failed its integrity check.")
    folds = set(range(len(report["folds"])))
    if not folds or any(len(c["folds"]) != len(folds) or {f["fold"] for f in c["folds"]} != folds
                        for c in report["candidates"]):
        raise ValueError("The sealed CV checkpoint has incomplete folds.")
    return _digest({key: report.get(key) for key in (
        "fingerprint", "folds", "candidates", "cv_timing", "tuning_seconds", "tuning_seconds_complete",
        "active_threadpools", "global_rng_prefix_changed", "resume_eligible")})


def _check_sealed_partition(directory, pins):
    plan = directory / "comparison-folds.json"
    receipt = directory / "comparison-folds-receipt.json"
    if plan.is_file() and receipt.is_file():
        payload = json.loads(plan.read_bytes())
        checksum = payload.pop("sha256", None)
        if (checksum == _digest(payload) and json.loads(receipt.read_bytes()) == {"fold_plan_sha256": checksum}
                and all(pin["partition"] == checksum for pin in pins.values())):
            return
    raise ValueError("The sealed comparison fold plan or its receipt is missing or changed.")


def _has_serial_finals(directory):
    if ((directory / "comparison.json").exists() or (directory / "comparison-cv-closed.json").exists()
            or any(directory.glob("*-evaluation.json"))):
        return True
    for path in directory.glob("*-cv.json"):
        report = json.loads(path.read_bytes())
        if report.get("final_fit_attempts") or report.get("state") in {"refitting", "completed"}:
            return True
    return False


@contextmanager
def _serial_comparison_batch(*, _host_reserved=False, root=None, directories=()):
    """Finish CV for every submitted comparison before consuming its final calls."""
    with ExitStack() as locks:
        if not _host_reserved:
            locks.enter_context(_timing_study_lock())
        batch = _ComparisonBatch(locks, deque())
        if root is not None:
            batch.declare(root, directories)
        yield batch
        for check in batch.checks:
            check()
        if not batch.closed and batch.pending:
            write_json(batch.root / "comparison-cv-closed.json",
                       {"studies": batch.studies, "sha256": _digest(batch.studies)})
        while batch.pending:
            batch.pending.popleft()()


def _cv_joblib_settings():
    backend, jobs = get_active_backend()
    return {"backend": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "requested_n_jobs": jobs, "effective_n_jobs": effective_n_jobs(None),
            "inner_max_num_threads": getattr(backend, "inner_max_num_threads", None)}


class _SerialTimingError(RuntimeError):
    """The requested serial measurements cannot be established."""


@contextmanager
def _serial_timing_phase(record):
    """Record the process, effective numerical limits and measured call interval."""
    record["verified"] = False
    with ExitStack() as limits:
        try:
            limits.enter_context(parallel_config(n_jobs=1))
            limits.enter_context(threadpool_limits(limits=1))
            pools = threadpool_info()
            workers = effective_n_jobs(None)
        except Exception as error:
            raise _SerialTimingError("Could not establish the serial timing limits.") from error
        if not pools:
            raise _SerialTimingError("No native thread pool was visible for the serial timing check.")
        if any(pool["num_threads"] != 1 for pool in pools):
            raise _SerialTimingError("A native thread pool did not accept the serial timing limit.")
        if workers != 1:
            raise _SerialTimingError("Joblib did not accept the serial timing limit.")
        record.update(application_workers=workers, native_threads=1, threadpools=pools,
                      host=platform.node(), pid=os.getpid(),
                      started_utc=datetime.now(timezone.utc).isoformat())
        elapsed = None

        def finish():
            nonlocal elapsed
            if elapsed is None:
                elapsed = time.perf_counter() - start
                record.update(seconds=elapsed, finished_utc=datetime.now(timezone.utc).isoformat())
            return elapsed

        start = time.perf_counter()
        try:
            yield finish
        finally:
            finish()
            # Failed attempts are published too. Check late-loaded libraries
            # before a model exception can be treated as a scientific failure.
            try:
                final_pools = threadpool_info()
            except Exception as error:
                raise _SerialTimingError("Could not verify the final native thread limits.") from error
            if final_pools != pools:
                record["final_threadpools"] = final_pools
            if not final_pools or any(pool["num_threads"] != 1 for pool in final_pools):
                raise _SerialTimingError("The final native thread limits could not be confirmed.")
            record["verified"] = True


def _serial_worker_overrides(model):
    # None inherits the Joblib context. Recent sklearn releases also expose
    # a string sentinel for deprecated n_jobs parameters that have no effect.
    try:
        return {name: 1 for name, value in model.get_params().items()
                if name.rsplit("__", 1)[-1] in {"n_jobs", "projection_workers", "projection_native_threads"}
                and isinstance(value, Integral) and value != 1}
    except Exception as error:
        raise _SerialTimingError("Could not inspect the estimator's worker settings.") from error


def _check_serial_workers(model):
    overrides = _serial_worker_overrides(model)
    if overrides:
        raise _SerialTimingError(f"Serial timing requires one worker and one native thread for {', '.join(overrides)}.")


def _private_input(array):
    """Copy input for an independent call, preserving table metadata or array flags."""
    if not isinstance(array, np.ndarray):
        return deepcopy(array)
    private = array.copy(order="K")
    if not array.flags.writeable:
        private.flags.writeable = False
    return private


def _serial_comparison_fit(model, X, y, attempt):
    # Change only the final clone. Explicit worker options take precedence
    # over an outer Joblib context and the native threadpool controller.
    overrides = _serial_worker_overrides(model)
    if overrides:
        try:
            model = model.set_params(**overrides)
        except Exception as error:
            raise _SerialTimingError("Could not set the estimator's serial timing parameters.") from error
        _check_serial_workers(model)
    # A transformer can change its input even on a failed fit. Every retry
    # and subsequent method starts with the same training values as its CV.
    X, y = _private_input(X), _private_input(y)
    attempt["fit_timing"] = {"worker_parameter_overrides": overrides}
    with _serial_timing_phase(attempt["fit_timing"]):
        return model.fit(X, y)


def calibration_search(estimator, param_grid, X, y, *, cv=5, seed=0,
                       threads=1, checkpoint_path=None, refit=True,
                       groups=None, folds=None, data_ids=None, scoring="accuracy",
                       _comparison_recovery=None):
    """Calibrate on training data with complete, resumable per-fold records.

    Defaults are five folds, seed zero and one native numerical thread.
    Candidate/fold loops are serial: ``threads`` limits supported native pools,
    not CV workers. Separate SPA projection controls can override that limit
    or launch processes, so it is not a global concurrency ceiling. ``seed``
    controls generated folds; fix estimator random states separately.

    ``threads`` accepts positive Python/NumPy integer scalars representable by
    the native C-int interface; Boolean and temporal scalars are not counts.
    ``refit`` accepts Python/NumPy Boolean scalars, not numeric 0/1 or arrays.
    These controls are checked before fold generation or checkpoint activity.
    For generated folds, ``seed`` retains sklearn's random-state contract,
    including None and RandomState. Use a fixed integer for reproducible runs.
    With explicit folds, seed does not generate splits but remains in identity;
    changing it can therefore prevent resume even when the folds are unchanged.

    Both fit and held-out-fold prediction warnings are retained. Convergence
    warnings invalidate that candidate. If a numerically valid CV winner fails
    full-training refit, try the next valid candidate and retain every attempt.
    If no candidate qualifies, a standalone search emits
    ``CalibrationFallbackWarning`` and returns a training-majority classifier
    with ``report["is_fallback"]`` set to True.
    With ``refit=False``, only CV is performed and the returned estimator is
    None. This supports staged tuning without intermediate full-data fits.
    Supply one subject ID per row in ``groups`` for group-disjoint CV, or pass
    exhaustive custom ``folds`` accepted by :func:`build_cv_folds`. Optional
    unique ``data_ids``, groups and the exact folds are bound to checkpoints.
    ``scoring`` is ``'accuracy'`` (the default) or ``'balanced_accuracy'``.
    The latter preserves the historical selection objective. Choose the
    objective before tuning; it never switches automatically with class
    proportions and does not change the estimator's decision rule.
    For named tables, the selected refit reports its input container. Known
    numeric estimators and stock pipelines receive validated numeric values
    with the ordered column names. Custom estimators and nonstandard pipelines
    receive positional arrays, as in an array-based search. Supply columns
    in training order when predicting with those returned estimators.
    Explicitly gridded inactive screening parameters produce a warning and
    report diagnostics; selection modes and candidate grids remain unchanged.

    A checkpoint destination has one cooperating writer for the entire call,
    including refit. A concurrent call using the same destination raises
    RuntimeError; different checkpoint files remain independent. The adjacent
    ``.<filename>.lock`` marker persists after release and must not be removed
    or replaced. Its presence alone does not mean the checkpoint is busy.
    With ``checkpoint_path=None``, no checkpoint files or locks are created.

    Keep callable parameters fixed across fits. For a random transformation,
    give its estimator an integer seed, learn the map in fit, and reuse that
    map in transform. A restart skips the draws belonging to completed folds.
    At fold publication, a checkpoint records the first observed change to
    configuration structure or a captured NumPy RNG. The current run can
    finish. Reusing marked folds for fitting requires a fresh search, even
    for a bookkeeping-only change. To read a compatible completed CV-only
    report, reconstruct the estimator and grid as they were at entry. A callback
    instance mutated by an earlier call can fail the fingerprint check. A change
    confined to the final refit leaves clean folds available for another refit
    from the original configuration.

    The entry fingerprint includes the contents of ordinary NumPy arrays and
    scalars. The fold check also observes supported RNG states, captured record
    scalar flags and the contents of their supported backing arrays. For other
    non-object arrays it observes dtype and shape. Same-dtype edits
    to numbers, fixed-width strings, dates and non-object structured values
    can escape detection when array shape is preserved. The check can also miss
    equal-length byte replacements, mutable globals and state shared across models.
    """
    threads = _validate_threads(threads)
    refit = _validate_refit(refit)
    scoring = _validate_scoring(scoring)
    raw_y = y
    feature_names = _input_feature_names(X)
    X, y, original_y = _check_labeled_data(X, y)
    classes, counts = np.unique(y, return_counts=True)
    _validate_class_labels(raw_y, y, classes=classes)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    split_state = None
    if _private_split_rng(seed):
        split_state = _digest(configuration_identity(seed.get_state(legacy=False)))
    folds = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    cv = len(folds)
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("At least one parameter candidate is required.")
    entry_observation = [] if checkpoint_path is not None else None
    split_limits = []
    identity = _search_identity(estimator, candidates, X, original_y, cv, seed, threads, refit=refit,
                                groups=groups, folds=folds, data_ids=data_ids, scoring=scoring,
                                _observation=entry_observation, _split_limits=split_limits,
                                _feature_names=feature_names)
    if type(estimator) is NMFClassifier:
        y = original_y
    # Identity must validate custom cloning before diagnostics clone anything.
    screening_diagnostics = screening_grid_diagnostics(estimator, candidates)
    uses_global = _uses_global_rng(estimator, candidates, split_limits)
    ambient_entry = _ambient_mt_token() if uses_global else None
    if uses_global:
        identity["global_rng_entry"] = ambient_entry
    if (split_state is not None and "unverified fit RNG" not in split_limits
            and (not uses_global or ambient_entry is not None)):
        identity["seed"] = split_state
        identity["split_seed_kind"] = "numpy.random.RandomState"
    if _comparison_recovery is not None and _comparison_recovery.unverified_prefix:
        identity["unverified_comparison_prefix"] = True
    serial_final = _comparison_recovery is not None and _comparison_recovery.serial_final
    if serial_final:
        identity["serial_final"] = True
        identity["cv_joblib"] = _cv_joblib_settings()
    fingerprint = _digest(identity)
    writer = _checkpoint_file(checkpoint_path)
    closed_cv = serial_final and _comparison_recovery.batch.closed
    if serial_final:
        # Retain writer ownership over the gap between this CV and its refit.
        writer = nullcontext(Path(checkpoint_path) if closed_cv else _comparison_recovery.batch.locks.enter_context(writer))
    with writer as path:
        loaded_checkpoint = path is not None and path.exists()
        if closed_cv and not loaded_checkpoint:
            raise ValueError("Missing CV checkpoint after final admission.")
        if _comparison_recovery is not None:
            _comparison_recovery.loaded_checkpoint = loaded_checkpoint
        if loaded_checkpoint:
            if _comparison_recovery is not None and _comparison_recovery.unverified_prefix:
                raise ValueError("Cannot reuse folds after an unverifiable comparison RNG prefix.")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("fingerprint") != fingerprint:
                raise ValueError("Checkpoint is incompatible with these data, settings, software, or source.")
            saved_digest = report.get("checkpoint_sha256")
            if saved_digest != _digest({k: v for k, v in report.items() if k != "checkpoint_sha256"}):
                raise ValueError("Checkpoint content failed its integrity check.")
            if "unverified scalar owner" in split_limits:
                raise ValueError("Cannot verify the backing storage of a captured NumPy scalar in this checkpoint.")
            if "unverified fit RNG" in split_limits:
                raise ValueError("Checkpoint reuse requires an estimator random state with verifiable clone behavior.")
            if uses_global and ambient_entry is None:
                raise ValueError("Checkpoint cannot identify the estimator's global random state; use an integer estimator seed.")
            if uses_global and report.get("global_rng_prefix_changed"):
                if not refit and report.get("state") == "cv_completed":
                    return CalibrationResult(None, report)
                # After the last saved fold, a scalar grid needs only score aggregation.
                complete = set(range(cv))
                if (refit or any({f["fold"] for f in c["folds"]} != complete for c in report["candidates"])
                        or any(type(k) is not str or
                               (v is not None and not any(type(v) is t for t in (int, float, bool, str)))
                               for p in candidates for k,v in p.items())):
                    raise ValueError("Checkpoint cannot reproduce the estimator's global random state; use an integer estimator seed.")
            if report.get("resume_eligible") is False:
                if not refit and report.get("state") == "cv_completed":
                    return CalibrationResult(None, report)
                raise ValueError("Checkpoint is ineligible for resume. " + report["resume_ineligibility_reason"])
        else:
            report = {"schema_version": 2, "author": "Chang Cheng", "fingerprint": fingerprint,
                      "scoring": scoring, "screening_diagnostics": screening_diagnostics,
                      "identity": identity, "provenance": provenance(threads),
                      "class_labels": _jsonable(classes), "class_counts": counts.tolist(),
                      "folds": [{"training_indices": a.tolist(), "validation_indices": b.tolist()}
                                for a, b in folds],
                      "candidates": [{"index": i, "parameters": _jsonable(p), "folds": []}
                                     for i, p in enumerate(candidates)],
                      "tuning_seconds": 0.0, "final_fit_seconds": 0.0, "final_fit_attempts": []}
        previous_seconds = report["tuning_seconds"]
        start = time.perf_counter()
        cv_interval = None
        if serial_final and not closed_cv:
            report.setdefault("tuning_seconds_complete", True)
            for interval in report.get("cv_timing", []):
                if interval.get("state") == "running":
                    interval["state"] = "unknown_after_last_checkpoint"
                    report["tuning_seconds_complete"] = False

        def checkpoint():
            if path is not None:
                report["checkpoint_sha256"] = _digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
                write_json(path, report)

        def record_cv_elapsed():
            elapsed = time.perf_counter() - start
            report["tuning_seconds"] = previous_seconds + elapsed
            if cv_interval is not None:
                cv_interval.update(seconds=elapsed, finished_utc=datetime.now(timezone.utc).isoformat())

        @contextmanager
        def cv_work():
            if cv_interval is None:
                yield
                return
            checkpoint()  # A process death may leave only the measured subtotal at this boundary.
            try:
                yield
            except BaseException as error:
                cv_interval.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                                   error_type=type(error).__name__)
                # Interrupted user code can advance RNGs or captured parameters
                # even though this fold did not produce a reusable score.
                if ambient_entry is not None and _ambient_mt_token() != ambient_entry:
                    report["global_rng_prefix_changed"] = True
                if entry_observation is not None and check_configuration(after_fold=True):
                    report.update(resume_eligible=False, resume_ineligibility_reason=(
                        "Estimator or candidate configuration changed during interrupted calibration."))
                raise
            else:
                cv_interval["state"] = "completed"
            finally:
                record_cv_elapsed()
                checkpoint()

        previous_files = {}

        def check_configuration(*, after_fold=False):
            """Check source on every call and report observed drift at fold publication."""
            if path is not None:
                _check_transform_output(identity["sklearn_transform_output"])
            if (path is not None and
                    get_config()["skip_parameter_validation"] != identity["sklearn_skip_parameter_validation"]):
                raise ValueError("Scikit-learn parameter validation changed during calibration.")
            observation = [] if after_fold and entry_observation is not None else None
            sources = implementation_sources(estimator, candidates, _previous_files=previous_files,
                                             _observation=observation, _split_seed=seed)
            if sources != identity["implementation_source"]:
                raise ValueError("Estimator implementation source changed during calibration.")
            return observation is not None and observation != entry_observation

        if not closed_cv:
            report["state"] = "calibrating"
            checkpoint()
        with threadpool_limits(limits=threads):
            remaining_folds = any(len(c["folds"]) < cv for c in report["candidates"])
            if closed_cv and remaining_folds:
                raise ValueError("The closed serial comparison requires all original CV folds.")
            if not serial_final or remaining_folds:
                report["active_threadpools"] = threadpool_info()
            if serial_final and remaining_folds:
                cv_interval = {"state": "running", "seconds": 0.0, "joblib": identity["cv_joblib"], "native_threads": threads,
                               "threadpools": report["active_threadpools"], "host": platform.node(),
                               "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
                report.setdefault("cv_timing", []).append(cv_interval)
                cv_interval["finished_utc"] = cv_interval["started_utc"]
                start = time.perf_counter()
            with cv_work():
                for params, candidate in zip(candidates, report["candidates"]):
                    completed = {f["fold"] for f in candidate["folds"]}
                    for fold, (training, validation) in enumerate(folds):
                        if fold in completed:
                            continue
                        check_configuration()
                        record = {"fold": fold, "fit_seconds": 0.0, "prediction_seconds": 0.0}
                        fitted = None
                        projection = None
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            try:
                                # Grid entries may themselves be estimators. Clone them
                                # too so replacement steps cannot retain another fold's
                                # fitted state or mutate the caller's candidate objects.
                                fitted = clone(estimator).set_params(**clone(params, safe=False))
                                names = feature_names if feature_names is not None and _numeric_schema_estimator(fitted) else None
                                fit_start = time.perf_counter()
                                try:
                                    fitted.fit(_named_matrix(X[training], names), y[training])
                                finally:
                                    record["fit_seconds"] = time.perf_counter() - fit_start
                                inference_start = time.perf_counter()
                                try:
                                    with _capture_projection(fitted) as projection:
                                        predictions = fitted.predict(_named_matrix(X[validation], names))
                                finally:
                                    record["prediction_seconds"] = time.perf_counter() - inference_start
                                score = _score_predictions(scoring, y[validation], predictions)
                                record.update(_score_fields(scoring, score))
                                record["status"] = "nonconverged" if _nonconverged(fitted, caught, projection) else "valid"
                            except Exception as error:
                                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                        record["warnings"] = _warning_records(caught)
                        if entry_observation is None:
                            check_configuration()
                        if fitted is not None:
                            record["diagnostics"] = _diagnostics(fitted, projection)
                        # Diagnostics may call user code. Observe after they finish,
                        # immediately before publishing this fold and its marker.
                        if entry_observation is not None and check_configuration(after_fold=True):
                            index = candidate["index"]
                            report.update(
                                resume_eligible=False,
                                resume_ineligibility_candidate_index=index,
                                resume_ineligibility_fold=fold,
                                resume_ineligibility_reason=(
                                    "Estimator or candidate configuration changed during calibration "
                                    f"(candidate {index}, fold {fold}). Repeat the entire search using a new "
                                    "checkpoint path or comparison output directory."
                                ),
                            )
                            entry_observation = None
                        if (path is not None and ambient_entry is not None and not report.get("global_rng_prefix_changed")
                                and _ambient_mt_token() != ambient_entry):
                            report["global_rng_prefix_changed"] = True
                        candidate["folds"].append(record)
                        record_cv_elapsed()
                        checkpoint()
                    valid = len(candidate["folds"]) == cv and all(f["status"] == "valid" for f in candidate["folds"])
                    scores = [f["score"] for f in candidate["folds"]] if valid else []
                    candidate["valid"] = valid
                    candidate.update(_score_fields(scoring, float(np.mean(scores)) if valid else None, mean=True))
                    candidate["fold_score_std"] = float(np.std(scores, ddof=1)) if valid else None
                    candidate["invalid_fold_count"] = sum(f["status"] != "valid" for f in candidate["folds"])
                    if not closed_cv:
                        checkpoint()
            if not serial_final:
                record_cv_elapsed()
            ranked = sorted((c for c in report["candidates"] if c["valid"]),
                            key=lambda c: (-c["mean_score"], c["index"]))
            if not refit:
                selected = ranked[0] if ranked else None
                report.update(state="cv_completed", is_fallback=False, refit_performed=False,
                              selected_candidate_index=None if selected is None else selected["index"],
                              selected_parameters=None if selected is None else selected["parameters"],
                              selected_mean_score=None if selected is None else selected["mean_score"],
                              tuning_complete=False, defaults_frozen=False,
                              refinement=mark_inactive_refinement(
                                  refinement_diagnostics(candidates, None if selected is None else selected["parameters"]),
                                  screening_diagnostics, None if selected is None else selected["index"]))
                checkpoint()
                return CalibrationResult(None, report)

            def fit_selected():
                cost_field = "cumulative_final_fit_seconds" if serial_final else "final_fit_seconds"
                if serial_final:
                    report.setdefault(cost_field, 0.0)
                    report.setdefault("cumulative_final_fit_seconds_complete", True)
                    pending = report.pop("pending_final_fit", None)
                    if pending is not None:
                        pending.update(status="interrupted_before_publication",
                                       seconds=pending.get("fit_timing", {}).get("seconds"))
                        if pending["seconds"] is None:
                            report["cumulative_final_fit_seconds_complete"] = False
                        else:
                            report[cost_field] += pending["seconds"]
                        report["final_fit_attempts"].append(pending)

                def measured_fit(model, labels, attempt, names):
                    pending = report["pending_final_fit"] = dict(attempt, entered_utc=datetime.now(timezone.utc).isoformat())
                    checkpoint()
                    try:
                        return _serial_comparison_fit(model, _named_matrix(X, names), labels, attempt)
                    except BaseException as error:
                        if isinstance(error, _SerialTimingError) or not isinstance(error, Exception):
                            receipt = attempt.get("fit_timing", {})
                            if "started_utc" in receipt:
                                status = "timing_verification_failed" if isinstance(error, _SerialTimingError) else "interrupted"
                                seconds = receipt.get("seconds")
                                attempt.update(status=status, error_type=type(error).__name__, error=str(error), seconds=seconds)
                                if seconds is None:
                                    report["cumulative_final_fit_seconds_complete"] = False
                                else:
                                    report[cost_field] += seconds
                                report["final_fit_attempts"].append(attempt)
                            elif not isinstance(error, Exception):
                                attempt.update(status="interrupted_before_fit", error_type=type(error).__name__, seconds=None)
                                report["final_fit_attempts"].append(attempt)
                            report.pop("pending_final_fit")
                        raise
                    finally:
                        # Preserve a completed measured call even if the worker
                        # dies while collecting diagnostics before publishing it.
                        if attempt.get("fit_timing", {}).get("verified"):
                            pending["fit_timing"] = attempt["fit_timing"]
                        checkpoint()

                selected = None
                fitted = None
                report["state"] = "refitting"
                checkpoint()
                for candidate in ranked:
                    check_configuration()
                    attempt = {"candidate_index": candidate["index"], "parameters": candidate["parameters"]}
                    attempt_start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            parameters = clone(candidates[candidate["index"]], safe=False)
                            model = clone(estimator).set_params(**parameters)
                            names = feature_names if feature_names is not None and _numeric_schema_estimator(model) else None
                            if feature_names is not None:
                                attempt["input_container"] = "named_numeric_frame" if names is not None else "positional_array"
                            if serial_final:
                                model = measured_fit(model, y, attempt, names)
                            else:
                                model = model.fit(_named_matrix(X, names), y)
                            attempt["diagnostics"] = _diagnostics(model)
                            attempt["status"] = "nonconverged" if _nonconverged(model, caught) else "valid"
                            if attempt["status"] == "valid":
                                fitted, selected = model, candidate
                        except Exception as error:
                            if serial_final and isinstance(error, _SerialTimingError):
                                raise
                            attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
                    attempt["warnings"] = _warning_records(caught)
                    check_configuration()
                    attempt["seconds"] = (attempt.get("fit_timing", {}).get("seconds") if serial_final
                                          else time.perf_counter() - attempt_start)
                    if attempt["seconds"] is not None:
                        report[cost_field] += attempt["seconds"]
                    report["final_fit_attempts"].append(attempt)
                    report.pop("pending_final_fit", None)
                    checkpoint()
                    if fitted is not None:
                        break
                if fitted is None:
                    fallback_start = time.perf_counter()
                    fitted = DummyClassifier(strategy="most_frequent")
                    attempt = {"status": "fallback", "reason": "No candidate passed all folds and final refit."}
                    if feature_names is not None:
                        attempt["input_container"] = "named_numeric_frame"
                    if serial_final:
                        measured_fit(fitted, _label_values(y), attempt, feature_names)
                        attempt["seconds"] = attempt["fit_timing"]["seconds"]
                    else:
                        fitted = fitted.fit(_named_matrix(X, feature_names), _label_values(y))
                        attempt["seconds"] = time.perf_counter() - fallback_start
                    report[cost_field] += attempt["seconds"]
                    report["final_fit_attempts"].append(attempt)
                    report.pop("pending_final_fit", None)
                if serial_final:
                    report["final_fit_seconds"] = attempt["seconds"]
                return fitted, selected

            def complete(fitted, selected):
                positional_columns = () if feature_names is None else None
                if feature_names is not None:
                    container = report["final_fit_attempts"][-1]["input_container"]
                    report["refit_input_container"] = container
                    if container == "positional_array":
                        positional_columns = tuple(feature_names)
                        report["refit_evaluation_schema"] = "calibration"
                    else:
                        report["refit_evaluation_schema"] = "native"
                report.update(state="completed", is_fallback=selected is None,
                              selected_candidate_index=None if selected is None else selected["index"],
                              selected_parameters=None if selected is None else selected["parameters"],
                              selected_mean_score=None if selected is None else selected["mean_score"],
                              tuning_complete=False, defaults_frozen=False,
                              refinement=mark_inactive_refinement(
                                  refinement_diagnostics(candidates, None if selected is None else selected["parameters"]),
                                  screening_diagnostics, None if selected is None else selected["index"]))
                checkpoint()
                return CalibrationResult(fitted, report, positional_columns)

            if serial_final:
                _comparison_recovery.refit = lambda: complete(*fit_selected())
                if not closed_cv:
                    report["state"] = "cv_completed"
                    checkpoint()
                return CalibrationResult(None, report)
            fitted, selected = fit_selected()
        result = complete(fitted, selected)
    if _comparison_recovery is None and result.report["is_fallback"]:
        warnings.warn(
            "No calibration candidate passed every fold and the full-training refit. "
            "Using a training-majority fallback (DummyClassifier). "
            "See report['candidates'] and report['final_fit_attempts'] for details.",
            CalibrationFallbackWarning, stacklevel=2,
        )
    return result


def _unique_native_groups(values, *, return_inverse=False):
    """Reuse exact uniform finite native groups without a sort.

    Only already validated owned numeric arrays reach this private helper.
    Nonuniform and empty arrays retain NumPy's original unique operation.
    Zero inverse codes preserve the original indexing/counting semantics.
    """
    if values.size and values[0] == values[-1] and np.all(values == values[0]):
        unique = values[:1].copy()
        if return_inverse:
            return unique, np.zeros(values.size, dtype=np.intp)
        return unique
    return np.unique(values, return_inverse=return_inverse)


def _evaluation_metadata(y, classes, *, sample_ids, groups, train_sample_ids,
                         train_groups, train_y, min_class_support, min_groups):
    """Validate supplied split metadata before any held-out prediction."""
    for name, value in (("min_class_support", min_class_support), ("min_groups", min_groups)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    sample_ids, sample_id_identity = _validated_unique_metadata(sample_ids, len(y), name="sample_ids")
    groups, group_native = _validate_metadata_snapshot(groups, len(y), name="groups")
    train_count = None
    if train_y is not None:
        raw_train_y = train_y
        train_y = np.asarray(train_y)
        if train_y.ndim != 1 or not len(train_y):
            raise ValueError("train_y must contain both fitted binary classes in a one-dimensional array.")
        _validate_class_labels(raw_train_y, train_y)
        train_y = _label_values(train_y)
        if not np.all(np.isin(train_y, classes)) or len(np.unique(train_y)) != 2:
            raise ValueError("train_y must contain both fitted binary classes in a one-dimensional array.")
        train_count = len(train_y)
    for name, values in (("train_sample_ids", train_sample_ids), ("train_groups", train_groups)):
        if values is not None:
            array = np.asarray(values, dtype=object)
            if array.ndim != 1:
                raise ValueError(f"{name} must match rows as a one-dimensional array.")
            if train_count is None:
                train_count = len(array)
    if train_count == 0:
        raise ValueError("Training metadata must not be empty.")
    train_sample_ids, train_sample_id_identity = _validated_unique_metadata(train_sample_ids, train_count, name="train_sample_ids")
    train_groups, train_group_native = _validate_metadata_snapshot(train_groups, train_count, name="train_groups")
    if train_sample_ids is not None:
        if sample_ids is None:
            raise ValueError("sample_ids are required when train_sample_ids are supplied.")
        # Each validated array has one ID family. Compare without coercing IDs.
        if isinstance(train_sample_ids[0], str) != isinstance(sample_ids[0], str):
            raise ValueError("Training and evaluation sample IDs must use the same identifier type (numeric or string).")
        if _metadata_overlap(train_sample_ids, train_sample_id_identity,
                             sample_ids, sample_id_identity):
            raise ValueError("Training and evaluation sample IDs must not overlap.")
    # Preserve the original point of group work: sample-ID rejection has
    # already completed. All grouping uses the snapshots validated above.
    group_unique = train_group_unique = None
    group_inverse = None
    mixed_group_disjoint = False
    mixed_group_overlap = None
    native_groups = (group_native is not None
                     and group_native.dtype.type in _NATIVE_ID_TYPES
                     and (train_groups is None
                          or (train_group_native is not None
                              and train_group_native.dtype.type in _NATIVE_ID_TYPES)))
    # Bound the complete canonical workload; pure primitive fast sets stay.
    if (native_groups
            and (group_native.dtype.type is np.longdouble
                 or (train_group_native is not None
                     and train_group_native.dtype.type is np.longdouble))
            and len(group_native) + (0 if train_group_native is None
                                     else len(train_group_native)) <= 8
            and type(y) is np.ndarray and y.ndim == 1 and y.dtype.kind in "biufUS"
            and type(classes) is np.ndarray and classes.ndim == 1
            and classes.dtype.kind in "biufUS"):
        native_groups = False
    if native_groups and group_native.dtype.type is np.longdouble:
        group_unique, group_inverse = _unique_native_groups(group_native, return_inverse=True)
        if (train_groups is not None
                and (group_native.dtype == train_group_native.dtype
                     or train_group_native.dtype.type is np.longdouble)):
            train_group_unique = _unique_native_groups(train_group_native)
        group_keys = group_set = train_group_set = None
        if train_groups is not None and train_group_unique is None:
            train_group_set = set(train_groups.tolist())
    elif native_groups:
        # Exact primitive snapshots convert to Python int/float objects.
        group_keys = groups
        group_set = set(group_keys.tolist())
        if (train_groups is not None
                and group_native.dtype != train_group_native.dtype
                and train_group_native.dtype.type is np.longdouble):
            train_group_unique = _unique_native_groups(train_group_native)
            train_group_set = None
        else:
            train_group_set = None if train_groups is None else set(train_groups.tolist())
    else:
        # Keep the complete original key order for mixed types and subclasses.
        group_keys = None if groups is None else _metadata_keys(groups)
        group_set = None if group_keys is None else set(group_keys.tolist())
        train_group_set = None if train_groups is None else set(_metadata_keys(train_groups).tolist())
    if (native_groups and train_groups is not None
            and group_native.dtype != train_group_native.dtype
            and (group_unique is not None or train_group_unique is not None)):
        # Canonicalize only endpoints first. Python numeric keys compare
        # exactly across formats; native arrays never undergo mixed coercion.
        query_bounds = (None if not group_unique.size else
                        _metadata_keys(np.asarray(group_unique[[0, -1]], dtype=object))) if group_unique is not None else (
                            None if not group_set else (min(group_set), max(group_set)))
        training_bounds = (None if not train_group_unique.size else
                           _metadata_keys(np.asarray(train_group_unique[[0, -1]], dtype=object))) if train_group_unique is not None else (
                               None if not train_group_set else (min(train_group_set), max(train_group_set)))
        if query_bounds is not None and training_bounds is not None:
            mixed_group_disjoint = bool(query_bounds[1] < training_bounds[0]
                                        or training_bounds[1] < query_bounds[0])
        if not mixed_group_disjoint:
            comparison_dtype = _exact_metadata_promotion(
                group_native.dtype, train_group_native.dtype,
                query_bounds, training_bounds)
            if comparison_dtype is not None:
                # Only comparison values widen. Native counts, inverse codes,
                # original report objects and caller storage remain unchanged.
                query_comparison = (group_unique if group_unique is not None
                                    else np.unique(group_native))
                training_comparison = (train_group_unique if train_group_unique is not None
                                       else np.unique(train_group_native))
                query_comparison = query_comparison.astype(comparison_dtype, copy=False)
                training_comparison = training_comparison.astype(comparison_dtype, copy=False)
                # Group promotion is complete here; two-field identities
                # deliberately bypass the sample-only promotion route.
                mixed_group_overlap = _metadata_overlap(
                    groups, (query_comparison, None),
                    train_groups, (training_comparison, None))
                del query_comparison, training_comparison
            else:
                if group_set is None:
                    group_set = _native_float_key_set(group_unique)
                    if group_set is None:
                        group_set = set(_metadata_keys(np.asarray(group_unique, dtype=object)).tolist())
                if train_group_set is None:
                    train_group_set = _native_float_key_set(train_group_unique)
                    if train_group_set is None:
                        train_group_set = set(_metadata_keys(np.asarray(train_group_unique, dtype=object)).tolist())
    if train_groups is not None:
        if groups is None:
            raise ValueError("groups are required when train_groups are supplied.")
        if isinstance(train_groups[0], str) != isinstance(groups[0], str):
            raise ValueError("Training and evaluation groups must use the same identifier type (numeric or string).")
        if (mixed_group_overlap if mixed_group_overlap is not None else
                not mixed_group_disjoint and (_metadata_overlap(train_groups, (train_group_unique, train_group_set),
                                      groups, (group_unique, group_set))
                    if group_unique is not None else train_group_set & group_set)):
            raise ValueError("Training and evaluation groups must not overlap.")
    counts = [int(np.sum(y == label)) for label in classes]
    group_count = (len(group_unique) if group_unique is not None
                   else None if group_set is None else len(group_set))
    low_class_support = min(counts) < min_class_support
    low_group_support = group_count is not None and group_count < min_groups
    metadata = {
        "sample_count": len(y), "class_labels": _jsonable(classes), "class_counts": counts,
        "min_class_support": int(min_class_support), "low_class_support": bool(low_class_support),
        "group_count": group_count, "min_groups": int(min_groups),
        "low_group_support": bool(low_group_support),
        "meets_support_criteria": not (low_class_support or low_group_support),
        "support_criteria_note": "Descriptive reporting thresholds, not a test of statistical independence or significance.",
        "sample_overlap_checked": train_sample_ids is not None,
        "group_overlap_checked": train_groups is not None,
        "training_sample_count": train_count,
        "training_group_count": (None if train_groups is None else
                                 len(train_group_unique) if train_group_unique is not None
                                 else len(train_group_set)),
    }
    if groups is not None:
        if group_unique is not None:
            metadata["per_class_group_count"] = [
                int(np.count_nonzero(np.bincount(group_inverse[y == label],
                                                minlength=len(group_unique))))
                for label in classes]
        else:
            metadata["per_class_group_count"] = [len(set(group_keys[y == label].tolist())) for label in classes]
    if train_y is None:
        baseline = {"available": False, "reason": "Training labels were not supplied; majority class was not inferred from evaluation labels."}
    else:
        training_counts = [int(np.sum(train_y == label)) for label in classes]
        majority = classes[int(np.argmax(training_counts))]
        baseline = {"available": True, "label": _jsonable(majority),
                    "training_class_counts": training_counts, "class_labels": _jsonable(classes),
                    "tie_rule": "First fitted class when training class counts tie.",
                    "accuracy": float(np.mean(y == majority)),
                    "balanced_accuracy": 0.5 if min(counts) > 0 else None}
    return sample_ids, groups, metadata, baseline


def _evaluation_pipeline_can_share(estimator):
    """Also admit subclasses inheriting Pipeline's public prediction methods."""
    if Pipeline not in type(estimator).__mro__:
        return False
    missing = object()
    for name in ("predict", "predict_proba", "decision_function",
                 "__getattribute__", "__getattr__", "__dict__"):
        if getattr_static(estimator, name, missing) is not getattr_static(Pipeline, name, missing):
            return False
    return _evaluation_head_has_plain_prediction_access(estimator.steps[-1][1])


def _evaluation_head_has_plain_prediction_access(head):
    """Decline dynamic lookup/descriptor policies without evaluating them.

    Existing adapter families' custom methods/callables delegate directly.
    Dynamic score descriptors retain the whole Pipeline lookup/call sequence.
    """
    if not any(base in type(head).__mro__
               for base in (KNeighborsClassifier, LogisticRegression, LinearSVC)):
        return False
    if (getattr_static(type(head), "__getattribute__", object.__getattribute__)
            is not object.__getattribute__
            or getattr_static(type(head), "__getattr__", None) is not None):
        return False
    for name in ("predict", "predict_proba", "decision_function"):
        member = getattr_static(head, name, None)
        if (member is not None and type(member) not in (FunctionType, MethodType)
                and getattr_static(type(member), "__get__", None) is not None):
            return False
    return True


def _evaluation_ordinary_numpy_error_policy():
    """Retain native operations for explicit error and callback policies."""
    return (np.geterr() == {"divide": "warn", "over": "warn",
                            "under": "ignore", "invalid": "warn"}
            and np.geterrcall() is None)


def _evaluation_knn_can_reuse(head, coordinates):
    """A narrow stock KNN boundary; callbacks keep native call order."""
    return (type(head) is KNeighborsClassifier
            and not any(name in vars(head) for name in
                        ("predict", "predict_proba", "kneighbors"))
            and type(head.weights) is str and head.weights in ("uniform", "distance")
            and type(head.metric) is str and head.metric != "pyfunc"
            and type(coordinates) in (np.ndarray, np.memmap)
            and _evaluation_ordinary_numpy_error_policy())


def _evaluation_svc_can_reuse(head):
    """Keep native binary SVC thresholding only for inherited stock methods."""
    if LinearSVC not in type(head).__mro__:
        return False
    missing = object()
    return (all(getattr_static(head, name, missing) is getattr_static(LinearSVC, name, missing)
                for name in ("predict", "decision_function", "predict_proba"))
            and _evaluation_ordinary_numpy_error_policy())


def evaluate_classifier(estimator, X, y, *, threads=1, sample_ids=None,
                        groups=None, train_sample_ids=None, train_groups=None,
                        train_y=None, min_class_support=10, min_groups=5, _serial_timing=None):
    """Evaluate a fitted classifier or the fitted model in a CalibrationResult.

    Passing the result retains the input column order for custom classifiers
    fitted on positional arrays. With unnamed training input, evaluation uses
    validation columns by position. Use that result with its original fit.
    If its calibration report flags a majority fallback, the evaluation report
    carries ``is_fallback=True``. A bare fitted estimator has no such provenance.

    Optional ``train_y`` defines a training-only majority reference. Subject IDs
    in ``groups`` provide subject counts; with ``train_groups`` they also enforce
    disjoint training/evaluation subjects. Likewise, ``train_sample_ids`` require
    explicit evaluation ``sample_ids`` and reject shared rows. Each paired
    training/evaluation ID array must use the same numeric or string identifier
    type; integer and floating-point IDs are compatible. Metadata is checked
    before prediction. Support thresholds add descriptive flags only;
    they do not alter predictions, scores, numerical status or model selection.
    Ranking metrics use probabilities when available, otherwise genuine binary
    decision margins oriented toward ``classes_[1]``. Margins are recorded
    separately and are never converted into purported probabilities.

    Explicit combined prediction methods keep their own contract. Eligible
    pipelines share preprocessing once; custom outer prediction methods use
    the supplied pipeline APIs. Stock KNN and linear-SVM score reuse preserves
    native decisions, with native calls for uncertain weighted votes, custom
    prediction behavior and explicit NumPy error/handler policies. Warnings
    describe operations actually performed; invalid scores still fail, but
    their default-warning order or duplication may differ from two calls.

    ``threads`` accepts positive Python/NumPy integer scalars within the native
    C-int range. Boolean, temporal, floating-point and array values are rejected
    before prediction or native-runtime initialization.
    """
    threads = _validate_threads(threads)
    positional_names = None
    calibration_metadata = {}
    if isinstance(estimator, CalibrationResult):
        if estimator.report.get("is_fallback") is True:
            calibration_metadata["is_fallback"] = True
        positional_names = estimator._positional_input_columns
        estimator = estimator.estimator
        if estimator is None:
            raise ValueError("The calibration result has no fitted estimator.")
    feature_names = _input_feature_names(X)
    if feature_names is not None:
        if positional_names is None:
            original_table = X
            expected_names = _fitted_input_names(estimator)
            _check_matching_columns(expected_names, feature_names)
        elif positional_names:
            _check_matching_columns(positional_names, feature_names)
    raw_y = y
    X, y, original_y = _check_labeled_data(X, y)
    _validate_class_labels(raw_y, y)
    raw_classes = estimator.classes_
    classes = np.asarray(raw_classes)
    _validate_class_labels(raw_classes, classes)
    classes = _label_values(classes)
    _validate_class_label(getattr(estimator, "positive_label_", None))
    if len(classes) != 2 or not np.all(np.isin(y, classes)):
        raise ValueError("Evaluation labels must belong to the fitted binary classes.")
    sample_ids, groups, support_diagnostics, majority_reference = _evaluation_metadata(
        y, classes, sample_ids=sample_ids, groups=groups, train_sample_ids=train_sample_ids,
        train_groups=train_groups, train_y=train_y, min_class_support=min_class_support,
        min_groups=min_groups)
    positive = classes[1]
    source_X = X
    if feature_names is not None and positional_names is None:
        if _numeric_schema_estimator(estimator, fitted=True):
            X = _named_matrix(X, feature_names)
        elif expected_names is not None or _has_named_pipeline_stage(estimator):
            # Custom consumers can observe the original index and dtypes,
            # including through a transformer configured for table output.
            # A downstream stage may also own names created by a transform.
            if not _has_pipeline_input_override(estimator):
                X = original_table
            elif expected_names is None:
                raise ValueError("Cannot validate raw feature names for this overridden pipeline.")
        else:
            raise ValueError("Cannot verify feature names for a positional estimator. "
                             "Pass its CalibrationResult or an array in training order.")
    score_X = X
    if _serial_timing is not None:
        _check_serial_workers(estimator)
    # A reconstructed frame may share validated numeric storage. Native table
    # transforms can also edit the caller's columns, index, or values in place.
    private_calls = _serial_timing is not None or feature_names is not None
    if private_calls:
        X = _private_input(X)
        # Combined predictions and shared pipelines consume one raw query.
        # Separate public calls may both contain in-place transformations.
        if not (_evaluation_pipeline_can_share(estimator) or
                (type(estimator) is NMFClassifier and
                 (hasattr(estimator, "predict_with_probabilities") or
                  hasattr(estimator, "predict_with_decision_function")))):
            score_X = _private_input(score_X)
    start = time.perf_counter()
    limits = threadpool_limits(limits=threads) if _serial_timing is None else _serial_timing_phase(_serial_timing)
    with (limits as finish_serial_timing, warnings.catch_warnings(record=True) as caught,
          _capture_projection(estimator) as projection):
        warnings.simplefilter("always")
        try:
            probability = margin = None
            if hasattr(estimator, "predict_with_probabilities"):
                predicted, all_probabilities = estimator.predict_with_probabilities(X)
                probability = all_probabilities[:, 1]
                prediction_operation = "joint_label_and_probability_prediction_one_projection"
            elif (hasattr(estimator, "predict_with_decision_function")
                  and not hasattr(estimator, "predict_proba")):
                predicted, margin = estimator.predict_with_decision_function(X)
                prediction_operation = "joint_label_and_margin_prediction_one_projection"
            elif _evaluation_pipeline_can_share(estimator):
                check_is_fitted(estimator)
                projected = X
                for _, _, transform in estimator._iter(with_final=False):
                    projected = transform.transform(projected)
                head = estimator.steps[-1][1]
                if _evaluation_knn_can_reuse(head, projected):
                    all_probabilities = head.predict_proba(projected)
                    predicted = (head.classes_[all_probabilities.argmax(axis=1)]
                                 if _knn_probabilities_preserve_labels(head, projected, all_probabilities)
                                 else head.predict(projected))
                    probability = all_probabilities[:, 1]
                    prediction_operation = "joint_label_and_probability_prediction_one_projection"
                elif _evaluation_svc_can_reuse(head):
                    margin = head.decision_function(projected)
                    predicted = head.classes_[(margin > 0).astype(np.intp)]
                    prediction_operation = "joint_label_and_margin_prediction_one_projection"
                else:
                    # Projection may itself be stateful. Reuse its exact output
                    # for both head calls while isolating custom in-place heads.
                    labels_input = projected
                    if (private_calls and
                            (type(head) not in (LogisticRegression, _TrackedLogisticRegression) or
                             any(name in vars(head) for name in ("predict", "predict_proba", "decision_function")))):
                        labels_input = _private_input(projected)
                    predicted = head.predict(labels_input)
                    if hasattr(estimator, "predict_proba"):
                        probability = head.predict_proba(projected)[:, 1]
                        prediction_operation = "joint_label_and_probability_prediction_one_projection"
                    elif hasattr(estimator, "decision_function"):
                        margin = head.decision_function(projected)
                        prediction_operation = "joint_label_and_margin_prediction_one_projection"
                    else:
                        prediction_operation = "predict_only"
            elif _evaluation_knn_can_reuse(estimator, X):
                all_probabilities = estimator.predict_proba(X)
                predicted = (estimator.classes_[all_probabilities.argmax(axis=1)]
                             if _knn_probabilities_preserve_labels(estimator, X, all_probabilities)
                             else estimator.predict(X))
                probability = all_probabilities[:, 1]
                prediction_operation = "joint_label_and_probability_prediction_no_projection"
            else:
                predicted = estimator.predict(X)
                if hasattr(estimator, "predict_proba"):
                    probability = estimator.predict_proba(score_X)[:, 1]
                    prediction_operation = "predict_plus_predict_proba_calls"
                elif hasattr(estimator, "decision_function"):
                    margin = estimator.decision_function(score_X)
                    prediction_operation = "predict_plus_decision_function_calls"
                else:
                    prediction_operation = "predict_only"
            if _serial_timing is not None:
                finish_serial_timing()
            if probability is not None and (not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1))):
                raise ValueError("Nonfinite or invalid predicted probabilities.")
            if margin is not None:
                margin = np.asarray(margin)
                if margin.shape != (len(y),) or not np.all(np.isfinite(margin)):
                    raise ValueError("Decision margins must be a finite one-dimensional binary score.")
            predicted = _label_values(_validate_class_labels(predicted))
        except Exception as error:
            if _serial_timing is not None:
                finish_serial_timing()
            return {**calibration_metadata, "status": "failed", "error_type": type(error).__name__, "error": str(error),
                    "prediction_seconds": (time.perf_counter() - start if _serial_timing is None
                                           else _serial_timing["seconds"]),
                    "warnings": _warning_records(caught), "diagnostics": _diagnostics(estimator, projection),
                    "support_diagnostics": support_diagnostics, "training_majority_reference": majority_reference,
                    "data_sha256": _data_fingerprint(source_X, original_y, feature_names)}
    elapsed = time.perf_counter() - start if _serial_timing is None else _serial_timing["seconds"]
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=classes,
                                                                     zero_division=0)
    both_classes = len(np.unique(y)) == 2
    auc = ap = None
    ranking_score = probability if probability is not None else margin
    score_type = "probability" if probability is not None else "decision_margin" if margin is not None else None
    if ranking_score is not None:
        if both_classes:
            auc = float(roc_auc_score(y == positive, ranking_score))
            ap = float(average_precision_score(y == positive, ranking_score))
    metrics = {"accuracy": float(accuracy_score(y, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both_classes else None,
               "roc_auc": auc, "average_precision": ap,
               "precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]),
               "positive_class": _jsonable(positive),
               "confusion_matrix": confusion_matrix(y, predicted, labels=classes).tolist(),
               "confusion_matrix_labels": _jsonable(classes),
               "per_class": [{"label": _jsonable(label), "precision": float(precision[i]),
                              "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                             for i, label in enumerate(classes)]}
    if not both_classes:
        metrics["undefined_reason"] = "Evaluation split contains only one class; balanced accuracy and ranking metrics omitted."
    return {**calibration_metadata,
            "status": "provisional_nonconvergence" if _nonconverged(estimator, caught, projection) else "valid",
            "metrics": metrics, "prediction_seconds": elapsed, "prediction_operation": prediction_operation,
            "ranking_score_type": score_type,
            "warnings": _warning_records(caught),
            "diagnostics": _diagnostics(estimator, projection),
            "data_sha256": _data_fingerprint(source_X, original_y, feature_names),
            "support_diagnostics": support_diagnostics, "training_majority_reference": majority_reference,
            "evaluation_groups": _jsonable(groups),
            "predictions": {"sample_ids": _jsonable(np.arange(len(y)) if sample_ids is None else sample_ids),
                            "true_labels": _jsonable(y), "predicted_labels": _jsonable(predicted),
                            "positive_class_probability": _jsonable(probability),
                            "positive_class_decision_margin": _jsonable(margin)}}


def format_evaluation_summary(report):
    """Format class balance, metrics, stage timings and convergence.

    Accept an evaluation report, or a comparison record containing ``evaluation``
    and its tuning/final-fit times. Missing diagnostics remain explicitly absent.
    Timing totals are reported as recorded, never inferred by adding components.
    """
    evaluation = report.get("evaluation", report)
    lines = [f"Evaluation status: {evaluation.get('status', 'unavailable')}."]
    if report.get("is_fallback", evaluation.get("is_fallback")):
        lines.insert(0, "Calibration used a training-majority fallback (DummyClassifier).")
    if "error" in evaluation:
        lines.append(f"Error: {evaluation['error']}")

    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    def count(value):
        return "unavailable" if value is None else str(value)

    metrics = evaluation.get("metrics", {})
    support = evaluation.get("support_diagnostics", {})
    reference = evaluation.get("training_majority_reference", {})
    if metrics:
        lines += ["", "| Metric | Value |", "| --- | --- |"]
        for key in ("accuracy", "balanced_accuracy", "roc_auc", "average_precision"):
            lines.append(f"| {key.replace('_', ' ').capitalize()} | {number(metrics.get(key))} |")
        if evaluation.get("ranking_score_type") == "decision_margin":
            lines += ["", "Ranking metrics use decision margins; probabilities are unavailable."]
        if metrics.get("undefined_reason"):
            lines += ["", metrics["undefined_reason"]]
    class_rows = metrics.get("per_class", [])
    if not class_rows:
        class_rows = [{"label": label, "support": count} for label, count in
                      zip(support.get("class_labels", []), support.get("class_counts", []))]
    if class_rows:
        training_counts = dict(zip(reference.get("class_labels", []),
                                   reference.get("training_class_counts", []))) if reference.get("available") else {}
        lines += ["", "| Class | Precision | Recall | Evaluation count | Training count |",
                  "| --- | --- | --- | --- | --- |"]
        for row in class_rows:
            label = str(row["label"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {number(row.get('precision'))} | {number(row.get('recall'))} | "
                         f"{count(row.get('support'))} | {count(training_counts.get(row['label']))} |")
    if reference.get("available"):
        lines += ["", f"Training-majority reference (class {reference['label']}): "
                  f"accuracy {number(reference['accuracy'])}; balanced accuracy {number(reference['balanced_accuracy'])}."]
    else:
        lines += ["", "Training-majority reference: unavailable; training labels were not supplied."]
    if support:
        groups = support.get("group_count")
        lines.append(f"Evaluation support: {support['sample_count']} rows; "
                     f"{'subject count unavailable' if groups is None else str(groups) + ' distinct subjects'}.")
        if support.get("low_class_support"):
            lines.append(f"Low class support: at least one class has fewer than {support['min_class_support']} rows.")
        if support.get("low_group_support"):
            lines.append(f"Low subject support: fewer than {support['min_groups']} distinct subjects.")
        if not support.get("group_overlap_checked"):
            lines.append("Training/evaluation subject overlap was not checked: paired subject metadata was not supplied.")
    diagnostics = evaluation.get("diagnostics", {})
    phases = (("Overall fit", "converged", "convergence_warnings"),
              ("Factorization", "factorization_converged", "factorization_convergence_warnings"),
              ("Training projection", "training_projection_converged", "training_projection_convergence_warnings"),
              ("Classifier fit", "classifier_converged", "classifier_convergence_warnings"),
              ("Latest projection", "last_transform_converged", "last_transform_warnings"))
    lines += ["", "| Numerical phase | Convergence |", "| --- | --- |"]
    phase_warnings = []
    for label, flag, warning_key in phases:
        value = diagnostics.get(flag)
        state = "unavailable" if value is None else "passed" if value else "failed"
        if flag == "training_projection_converged" and diagnostics.get("training_projection_performed") is False:
            state = "not performed"
        if flag == "last_transform_converged" and diagnostics.get("last_transform_performed") is False:
            state = "not performed"
        lines.append(f"| {label} | {state} |")
        for message in diagnostics.get(warning_key, ()):
            phase_warnings.append(f"{label} warning: {message}")
    if phase_warnings:
        lines += ["", *phase_warnings]
    if diagnostics.get("last_transform_error"):
        lines += ["", f"Latest projection error: {diagnostics['last_transform_error']}"]
    lines += ["", "| Timed stage | Seconds |", "| --- | --- |"]
    if "cumulative_final_fit_seconds" in report:
        cumulative_label = ("Final fitting (measured subtotal, incomplete)"
                            if report.get("cumulative_final_fit_seconds_complete") is False
                            else "Final fitting (all recorded attempts)")
        fits = [("Final fitting (evaluated model)", report.get("final_fit_seconds")),
                (cumulative_label, report["cumulative_final_fit_seconds"])]
        fit_note = "The cumulative fit time includes measured durations for the evaluated fit, earlier refits and unsuccessful attempts."
    else:
        fits = [("Final fitting (recorded attempts)", report.get("final_fit_seconds"))]
        fit_note = "Final fitting can include unsuccessful refit attempts."
    tuning_label = "Tuning (measured subtotal, incomplete)" if report.get("tuning_seconds_complete") is False else "Tuning"
    for label, value in [(tuning_label, report.get("tuning_seconds")), *fits,
                         ("Prediction", evaluation.get("prediction_seconds"))]:
        lines.append(f"| {label} | {number(value)} |")
    fit_times = diagnostics.get("fit_times", {})
    stages = ("screening", "factorization", "training_projection", "classifier", "total")
    for label in stages:
        lines.append(f"| Fit: {label.replace('_', ' ')} | {number(fit_times.get(label))} |")
    for label, value in fit_times.items():
        if label in stages:
            continue
        lines.append(f"| Fit: {label.replace('_', ' ')} | {number(value)} |")
    lines += ["", fit_note + " Fit rows describe the "
              "fitted estimator; its total includes its component stages. Prediction is the measured "
              "evaluation operation. Do not add these overlapping totals or infer total time by "
              "summing stage medians or concurrent worker times. Individual-operation latency "
              "and batch throughput are separate measurements."]
    return "\n".join(lines) + "\n"


@contextmanager
def _timing_study_lock():
    """Reserve this user and host for one participating timing study, including CV."""
    # A shared home can span machines. Different environments and output
    # directories on the same machine must still contend for the same lock.
    configured = os.environ.get("SUPERVISED_NMF_TIMING_LOCK")
    if configured is None:
        marker = Path.home() / f".supervised-nmf-timing-{platform.node().lower()}.lock"
    else:
        marker = Path(configured)
        if not marker.is_absolute():
            raise ValueError("SUPERVISED_NMF_TIMING_LOCK requires an absolute lock-file path.")
    with _file_lock(marker, f"Another timing study is active on this host ({marker})."):
        yield


def comparison_models(*, random_state=0, quick=False, max_iter=20000, tol=1e-4,
                      classifier="knn", logistic_max_iter=1000,
                      logistic_tol=1e-4, logistic_class_weight=None,
                      svm_max_iter=20000, svm_tol=1e-5, svm_class_weight=None):
    """Create independent CD/SPA/MU, PCA, screening-only and original baseline grids.

    Grids are deliberately modest starting grids. Caller-supplied JSON can expand
    them, and diagnostics identify unfinished boundaries/interior refinement.
    Absolute screening thresholds the computed |t|. Percentile screening ranks it.
    ``classifier='knn'`` preserves the historical method names and grids.
    ``classifier='logistic'`` uses standardized representations and L2 logistic
    regression, with distinct names so both mappings can be combined. Scaling,
    screening and decompositions are fitted independently within each CV fold.
    ``classifier='linear_svm'`` uses primal squared-hinge L2 linear SVMs and
    genuine margins. It introduces no implicit coordinate scaling.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .classifier import NMFClassifier
    from .screening import BinaryScreening

    if classifier not in {"knn", "logistic", "linear_svm"}:
        raise ValueError("classifier must be 'knn', 'logistic', or 'linear_svm'.")
    ranks = [2] if quick else [2, 4]
    neighbors = [3] if quick else [3, 5, 7]
    percentages = [25.0] if quick else [10.0, 25.0]
    thresholds = [1.0] if quick else [1.0, 2.0]
    regularization = [1.0] if quick else [0.01, 0.1, 1.0, 10.0, 100.0]
    suffix = "" if classifier == "knn" else f"_{classifier}"
    if classifier == "logistic":
        nmf_options = {"classifier": classifier, "logistic_max_iter": logistic_max_iter,
                       "logistic_tol": logistic_tol, "logistic_class_weight": logistic_class_weight}
    elif classifier == "linear_svm":
        nmf_options = {"classifier": classifier, "svm_max_iter": svm_max_iter,
                       "svm_tol": svm_tol, "svm_class_weight": svm_class_weight}
    else:
        nmf_options = {}
    regularization_parameter = "logistic_C" if classifier == "logistic" else "svm_C"
    nmf_classifier_grid = ({"n_neighbors": neighbors} if classifier == "knn"
                           else {regularization_parameter: regularization})
    pipeline_classifier_grid = ({"knn__n_neighbors": neighbors} if classifier == "knn"
                                else {f"{classifier}__C": regularization})

    def classifier_steps():
        if classifier == "knn":
            return [("knn", KNeighborsClassifier(n_jobs=1))]
        if classifier == "linear_svm":
            return [("linear_svm", _TrackedLinearSVC(
                loss="squared_hinge", penalty="l2", dual=False, max_iter=svm_max_iter,
                tol=svm_tol, class_weight=svm_class_weight, random_state=random_state))]
        return [("scale", StandardScaler()),
                ("logistic", _TrackedLogisticRegression(solver="lbfgs", max_iter=logistic_max_iter,
                                                         tol=logistic_tol, class_weight=logistic_class_weight,
                                                         random_state=random_state))]

    models = {}
    for screening in ("percentile", "absolute"):
        screen_parameter = "percentile" if screening == "percentile" else "score_threshold"
        screen_values = percentages if screening == "percentile" else thresholds
        for solver in ("cd", "spa", "mu"):
            models[f"screened_{solver}_{screening}{suffix}"] = (
                NMFClassifier(solver=solver, screening=screening, random_state=random_state,
                              max_iter=max_iter, tol=tol, **nmf_options),
                {screen_parameter: screen_values, "n_components": ranks, **nmf_classifier_grid})
        screen = BinaryScreening(screening=screening)
        models[f"screened_pca_{screening}{suffix}"] = (
            Pipeline([("screen", screen), ("pca", PCA(svd_solver="full", random_state=random_state))]
                     + classifier_steps()),
            {f"screen__{screen_parameter}": screen_values, "pca__n_components": ranks,
             **pipeline_classifier_grid})
        models[f"screening_only_{screening}{suffix}"] = (
            Pipeline([("screen", clone(screen))] + classifier_steps()),
            {f"screen__{screen_parameter}": screen_values, **pipeline_classifier_grid})
    if classifier == "knn":
        models["raw_knn"] = (KNeighborsClassifier(n_jobs=1), {"n_neighbors": neighbors})
    elif classifier == "logistic":
        models["raw_logistic"] = (Pipeline(classifier_steps()), {"logistic__C": regularization})
    else:
        models["raw_linear_svm"] = (Pipeline(classifier_steps()), {"linear_svm__C": regularization})
    models[f"majority_class{suffix}"] = (DummyClassifier(strategy="most_frequent"), {})
    models[f"unscreened_cd_nmf{suffix}"] = (
        NMFClassifier(solver="cd", screening="none", random_state=random_state,
                      max_iter=max_iter, tol=tol, **nmf_options),
        {"n_components": ranks, **nmf_classifier_grid})
    return models


_COMPARISON_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _uses_skipped_rng(dependencies, skipped):
    """Identify streams or overlapping buffers retained by a skipped callback."""
    for value, _ in dependencies:
        if id(value) in skipped:
            return True
        if type(value) is np.ndarray and any(
            type(previous) is np.ndarray and np.may_share_memory(value, previous)
            for previous in skipped.values()
        ):
            return True
    return False


def _comparison_has_mutable_parameters(configuration):
    """Check actual estimator and grid values in the already captured identity."""
    def immutable(value):
        kind = value["kind"]
        if kind == "estimator":
            return all(immutable(setting) for _, setting in value["parameters"]["items"])
        if kind == "tuple":
            return all(immutable(item) for item in value["items"])
        if kind == "numpy_scalar":
            # A structured scalar can be a writable view into an RNG's pool.
            dtype = value["dtype"]
            return dtype["str"][1] not in ("V", "O") and not dtype.get("metadata")
        return kind in ("none", "bool", "int", "float", "complex", "str", "bytes")

    # The outer dictionaries and candidate list were synthesized for this
    # search. Only their parameter values can belong to the caller's RNG.
    entries = {key["value"]: value for key, value in configuration["items"]}
    return not immutable(entries["estimator"]) or any(
        not immutable(setting) for candidate in entries["candidates"]["items"]
        for _, setting in candidate["items"])


def _comparison_has_results(directory):
    # Only files whose names the comparison writer can produce are evidence.
    for suffix in ("-cv.json", "-evaluation.json"):
        for path in directory.glob("*" + suffix):
            if _COMPARISON_NAME.fullmatch(path.name[:-len(suffix)]) and path.is_file():
                return True
    summary = directory / "comparison.json"
    if not summary.exists():
        return False
    saved = json.loads(summary.read_text(encoding="utf-8"))
    return not (isinstance(saved, dict) and saved.get("models") == {}
                and saved.get("phase") == "development_comparison"
                and saved.get("fold_plan_sha256") is None)


def _comparison_folds(directory, requested, X, y, seed):
    """Commit a single partition before any model starts calibrating."""
    path = directory / "comparison-folds.json"
    receipt = directory / "comparison-folds-receipt.json"
    # Calibration accepts a one-column target. Leave its usual warning to
    # calibration itself rather than issuing it once more during split setup.
    labels = y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
    with _checkpoint_file(path) as path:
        if path.exists():
            plan = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(plan, dict) or type(plan.get("schema_version")) is not int
                    or plan["schema_version"] != 1):
                raise ValueError("Invalid comparison fold plan.")
            checksum = plan.pop("sha256", None)
            if checksum != _digest(plan):
                raise ValueError("Comparison fold plan failed its integrity check.")
            if _digest(plan.get("identity")) != _digest(requested):
                raise ValueError("Existing comparison folds are incompatible; create a new run directory.")
            validation = plan.get("validation_indices")
            if not isinstance(validation, list) or len(validation) != requested["cv"]:
                raise ValueError("Invalid comparison fold plan.")
            folds = None
        else:
            if receipt.exists() or _comparison_has_results(directory):
                raise ValueError("Existing comparison records lack their fold plan; create a new run directory.")
            checked_X, checked_y, _ = _check_labeled_data(X, labels)
            folds = build_cv_folds(checked_X, checked_y, cv=requested["cv"], seed=seed)
            validation = [valid.tolist() for _, valid in folds]
            plan = {"schema_version": 1, "identity": requested,
                    "validation_indices": validation}
            checksum = _digest(plan)
            write_json(path, dict(plan, sha256=checksum))
        # A valid plan copied from another run has a valid *internal* checksum.
        # Record the committed digest independently, including for partial runs
        # and callers appending new model names before loading older results.
        expected_receipt = {"fold_plan_sha256": checksum}
        if receipt.exists():
            if json.loads(receipt.read_text(encoding="utf-8")) != expected_receipt:
                raise ValueError("Comparison fold plan does not match its original receipt.")
        else:
            if folds is None and _comparison_has_results(directory):
                raise ValueError("Existing comparison records lack their fold receipt; create a new run directory.")
            # A prior call may have committed its plan and stopped here. No
            # model can start before the receipt is published under this lock.
            write_json(receipt, expected_receipt)
    return folds, validation, checksum


def run_comparison(models, X_train, y_train, X_validation, y_validation, *,
                   output_dir, seed=0, threads=1, validation_ids=None, signal_support=None,
                   scoring="accuracy", serial_final=False, _batch=None):
    """Save each model independently; resume only exactly compatible results.

    Validation is development data, never evidence that defaults are finalized.
    No test-set loading, default selection, or claimed reproduction occurs here.
    ``scoring`` chooses the common training-CV objective for all methods;
    development evaluation still reports its full set of classification metrics.

    With ``serial_final=True``, all models finish CV before any final fit.
    CV uses the requested ``threads`` and ambient Joblib configuration, and final
    fits and held-out predictions run with one Joblib worker and one native
    thread. The final estimator clone uses one for explicit ``n_jobs``,
    ``projection_workers`` and ``projection_native_threads`` settings. Other
    estimators must respect these controls for a comparable timing. Keep
    competing jobs off the host. The CV checkpoint records the settings,
    process, native pools and wall-clock interval of each final fit. The
    evaluation record includes the corresponding prediction evidence.
    The selected fit time excludes unsuccessful attempts and earlier refits
    after interruption. The cumulative final fit time includes all attempts.
    Resume a serial study with its original ordered methods and CV settings.

    ``threads`` follows the same integer-count contract as calibration and
    evaluation. The generated-fold ``seed`` follows sklearn's random-state
    contract, including None and RandomState; both controls are checked before
    output creation. Seed preflight does not draw random numbers. Completed
    compatible results retain the existing reuse policy; use a fixed integer
    seed for reproducible new comparisons. Each nonempty comparison saves one
    common partition in comparison-folds.json for all models. Ordinary
    comparisons also support new models appended on resume. With a split seed of None or a mutable
    RandomState, later models can produce different results from earlier
    releases, which drew folds for each model. Keep the partition, its receipt,
    and the per-model records together. The directory reserves comparison.json
    and files named <model>-cv.json or <model>-evaluation.json for legal model
    names. Each call writes comparison.json for the models supplied in that
    call. Appending only B preserves A's per-model files and lists only B in
    the summary. Supply both A and B for a combined summary. Finish older
    jobs with their original source and environment, or start a new comparison
    in a fresh directory after upgrading.

    Give each model an independent integer seed. Loading an archived model
    skips its fits and predictions, including any external RNG draws. Models
    depending on NumPy's global RNG cannot follow a skipped model that
    may have advanced it. Restoring the partition also skips the splitter's
    draws, so a model sharing its mutable RNG requires a fresh comparison.
    A custom RandomState subclass or bit generator can change other inputs
    during splitting. Comparisons using one must run in a fresh directory.
    A later model using a stream or overlapping buffer captured by a skipped
    callback also requires a fresh comparison. A view with an unknown backing
    object, dtype metadata, and other opaque retained state can hide a shared
    RNG or buffer on either side of a model boundary. When an earlier model
    or a mutable splitter could affect that state, the later model's folds
    and evaluation require a fresh comparison, even if read on their own.
    Other mutable callback state shared across models can still affect results
    when appended.
    A model with unverifiable RNG clone or child-stream behavior can run
    only when the comparison has executed the split and every preceding model
    from scratch.
    An existing compatible evaluation remains readable when
    its calibration observed a change during a fold. To load it, reconstruct
    the estimator and grid as they were at calibration entry. A callback
    instance mutated by an earlier call can fail the fingerprint check.
    To recreate the fitted estimator after a marked fold, repeat the search
    in a new comparison output directory.
    """
    threads = _validate_threads(threads)
    if not isinstance(serial_final, (bool, np.bool_)):
        raise ValueError("serial_final must be a Boolean.")
    if serial_final and _batch is None:
        with _serial_comparison_batch() as batch:
            return run_comparison(models, X_train, y_train, X_validation, y_validation,
                                  output_dir=output_dir, seed=seed, threads=threads,
                                  validation_ids=validation_ids, signal_support=signal_support,
                                  scoring=scoring, serial_final=True, _batch=batch)
    # Validate the consumed generic seed without drawing from mutable/global RNGs.
    # Keep sklearn's existing None/RandomState contract and the original identity.
    custom_split = not _stock_split_rng(check_random_state(seed))
    scoring = _validate_scoring(scoring)
    y_train = _validate_class_labels(y_train)
    y_validation = _validate_class_labels(y_validation)
    transform_output = get_config()["transform_output"]
    train_names = _input_feature_names(X_train)
    validation_names = _input_feature_names(X_validation)
    _check_matching_columns(train_names, validation_names)
    named_input = _named_input_identity() if train_names is not None or validation_names is not None else None
    output_dir = Path(output_dir)
    if serial_final:
        model_pins, saved_evaluations = _batch.register(output_dir, models)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    fold_digest = None
    restored_plan = False
    independent_split = isinstance(seed, (int, np.integer)) or _private_split_rng(seed)
    skipped_global_model = False
    skipped_captures = {}
    # A noninteger splitter advances an RNG that later opaque state may retain.
    earlier_retained_state = not isinstance(seed, (int, np.integer))
    earlier_unverified_state = False
    loaded_model_checkpoint = False
    skipped_cv = []

    def publish(name, comparison_fingerprint, result_path, skip_validation, calibrated, recovery):
        _check_transform_output(transform_output)
        if serial_final:
            calibrated = recovery.refit()
        record = {"model": name, "comparison_fingerprint": comparison_fingerprint,
                  "fold_plan_sha256": fold_digest,
                  "scoring": scoring,
                  "selected_mean_score": calibrated.report["selected_mean_score"],
                  "is_fallback": calibrated.report["is_fallback"],
                  "selected_parameters": calibrated.report["selected_parameters"],
                  "tuning_seconds": calibrated.report["tuning_seconds"],
                  "final_fit_seconds": calibrated.report["final_fit_seconds"],
                  "tuning_complete": False, "defaults_frozen": False,
                  "evaluation_role": "development_validation",
                  "refinement": calibrated.report["refinement"]}
        if serial_final:
            record["cumulative_final_fit_seconds"] = calibrated.report["cumulative_final_fit_seconds"]
            record["cumulative_final_fit_seconds_complete"] = calibrated.report["cumulative_final_fit_seconds_complete"]
            record["tuning_seconds_complete"] = calibrated.report["tuning_seconds_complete"]
            record["cv_timing"] = calibrated.report["cv_timing"]
            attempts = calibrated.report["final_fit_attempts"]
            record["fit_timing"] = attempts[-1]["fit_timing"]
            # A killed process may have started fitting before it could save its clock.
            record["first_final_fit_started_utc"] = next(
                (a.get("fit_timing", {}).get("started_utc") for a in attempts
                 if "started_utc" in a.get("fit_timing", {}) or a["status"] == "interrupted_before_publication"), None)
        if calibrated.report.get("resume_eligible") is False:
            for field in ("resume_eligible", "resume_ineligibility_candidate_index",
                          "resume_ineligibility_fold", "resume_ineligibility_reason"):
                record["calibration_" + field] = calibrated.report[field]
        if signal_support is not None and not record["is_fallback"]:
            record["support_recovery"] = support_recovery(calibrated.estimator, signal_support)
        if "refit_input_container" in calibrated.report:
            record["refit_input_container"] = calibrated.report["refit_input_container"]
            record["refit_evaluation_schema"] = calibrated.report["refit_evaluation_schema"]
        prediction_options = {}
        if serial_final:
            record["prediction_timing"] = {}
            prediction_options["_serial_timing"] = record["prediction_timing"]
        try:
            evaluation_model = (calibrated if validation_names is not None and calibrated._positional_input_columns is not None
                                else calibrated.estimator)
            record["evaluation"] = evaluate_classifier(evaluation_model, X_validation, y_validation,
                                                        threads=1 if serial_final else threads,
                                                        sample_ids=validation_ids, train_y=y_train,
                                                        **prediction_options)
        except Exception as error:
            if serial_final and isinstance(error, _SerialTimingError):
                raise
            record["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            if serial_final and record["prediction_timing"].get("verified"):
                record["evaluation"]["prediction_seconds"] = record["prediction_timing"]["seconds"]
        else:
            evaluation = record["evaluation"]
            if record["is_fallback"] and evaluation_model is calibrated and "is_fallback" in evaluation:
                # Comparison records already carry this calibration flag at the top level.
                record["evaluation"] = {key: value for key, value in evaluation.items() if key != "is_fallback"}
        if get_config()["skip_parameter_validation"] != skip_validation:
            raise ValueError("Scikit-learn parameter validation changed during comparison.")
        _check_transform_output(transform_output)
        if serial_final:
            record["cv_checkpoint_sha256"] = calibrated.report["checkpoint_sha256"]
            record["evaluation_sha256"] = _digest(record)
        write_json(result_path, record)
        results[name] = record

    def verify_skipped_cv(name, uses_global, retains_state, unverified_state, mutable, rng_dependencies):
        # A later method's CV also precedes this refit. Its skipped callbacks
        # or RNG draws can therefore matter in either direction on recovery.
        for other_name, global_rng, retained, unverified, captures in skipped_cv:
            if name == other_name:
                continue
            if ((global_rng and uses_global) or _uses_skipped_rng(rng_dependencies, captures)
                    or (unverified and (retains_state or mutable)) or (unverified_state and retained)):
                raise ValueError("Cannot reproduce a final fit after skipping another model's CV RNG or callback.")

    for name, (estimator, grid) in models.items():
        if not _COMPARISON_NAME.fullmatch(name):
            raise ValueError("Model names must contain only letters, digits, underscore or hyphen.")
        candidates = list(ParameterGrid(grid))
        rng_limits = []
        rng_dependencies = []
        identity = _search_identity(estimator, candidates, X_train, y_train, 5, seed, threads,
                                    scoring=scoring, _split_limits=rng_limits,
                                    _fit_rng_dependencies=rng_dependencies)
        if serial_final:
            identity["serial_final"] = True
            identity["cv_joblib"] = _cv_joblib_settings()
        uses_global = _uses_global_rng(estimator, candidates, rng_limits)
        # A hidden owner can occur on either side (or both). A bare captured
        # array, for example, can be a seed pool with no visible Generator.
        # Record uncertainty during fresh execution as well: a later reader
        # can omit the predecessor when requesting the saved model alone.
        unverified_state = ("unverified captured state" in rng_limits or
                            "unverified fit RNG" in rng_limits)
        retains_state = unverified_state or uses_global or any(captured for _, captured in rng_dependencies)
        unverified_prefix = custom_split or (earlier_retained_state and unverified_state)
        if earlier_unverified_state:
            unverified_prefix |= retains_state or _comparison_has_mutable_parameters(
                identity["structural_configuration"])
        if unverified_prefix:
            identity["unverified_comparison_prefix"] = True
        shares_split = "shared split RNG" in rng_limits
        if shares_split:
            identity["comparison_shares_split_rng"] = True
        if uses_global:
            identity["comparison_global_rng_entry"] = _ambient_mt_token()
            if not independent_split:
                # The first model's entry precedes the split. Later entries
                # follow it, and an equal token at a different boundary does
                # not establish the same RNG position at the first fit.
                identity["comparison_first_model"] = not results
        validation_data = data_fingerprint(X_validation, y_validation)
        partition_request = _jsonable({key: identity[key] for key in
                                       ("cv", "seed", "data_sha256", "software", "source_sha256",
                                        "sklearn_skip_parameter_validation", "sklearn_transform_output")})
        if named_input is not None:
            partition_request["validation_feature_names_sha256"] = _schema_digest(validation_names)
            partition_request["named_input"] = named_input
        _check_transform_output(transform_output)
        if serial_final:
            partition_request["serial_final"] = True
            partition_request["ordered_models"] = list(models)
            partition_request["cv_joblib"] = identity["cv_joblib"]
            partition_request["threads"] = threads
        if custom_split:
            partition_request["custom_split_rng"] = True
        created_plan = False
        if fold_digest is None:
            first_request_digest = _digest(partition_request)
            folds, held_out, fold_digest = _comparison_folds(output_dir, partition_request, X_train, y_train, seed)
            created_plan = folds is not None
            restored_plan = folds is None
        elif _digest(partition_request) != first_request_digest:
            raise ValueError("Training data or software changed during comparison.")
        configuration = identity.pop("structural_configuration")
        comparison_context = _digest({"calibration": identity, "fold_plan_sha256": fold_digest,
                                      "validation": validation_data, "validation_ids": validation_ids,
                                      "signal_support": signal_support})
        result_path = output_dir / f"{name}-evaluation.json"
        reuse = name in saved_evaluations if serial_final else result_path.exists()
        cv_path = output_dir / f"{name}-cv.json"
        saved_model = reuse or cv_path.is_file()
        if custom_split and (restored_plan or saved_model):
            raise ValueError("Cannot replay the effects of a custom split RNG from a saved comparison.")
        if (restored_plan or saved_model or loaded_model_checkpoint) and "unverified scalar owner" in rng_limits:
            raise ValueError("Cannot verify the backing storage of a captured NumPy scalar in this saved comparison.")
        if (restored_plan or saved_model or loaded_model_checkpoint) and "unverified fit RNG" in rng_limits:
            raise ValueError("Cannot verify the fit generator in this saved comparison.")
        if uses_global and saved_model and identity["comparison_global_rng_entry"] is None:
            raise ValueError("Cannot verify NumPy global randomness in this comparison.")
        if restored_plan and shares_split:
            raise ValueError("Restoring the fold plan skips draws from the model's shared split RNG. "
                             "Start a new comparison or use independent integer seeds.")
        if skipped_global_model and uses_global:
            raise ValueError("An earlier archived model may have advanced the global RNG. "
                             "Start a new comparison or use an explicit estimator seed.")
        if skipped_captures and _uses_skipped_rng(rng_dependencies, skipped_captures):
            raise ValueError("An earlier archived callback may have changed this model's RNG or buffer. "
                             "Start a new comparison or use independent model seeds.")
        if unverified_prefix and (restored_plan or saved_model or loaded_model_checkpoint):
            raise ValueError("Cannot verify the independence of captured state and another model's RNG.")
        if restored_plan and uses_global:
            if not independent_split and (not reuse or results):
                raise ValueError("Restoring the fold plan skips draws that may advance the global RNG. "
                                 "Start a new comparison or use an independent split seed.")
            if not saved_model and not results:
                raise ValueError("Cannot verify global RNG state for a new model appended to a saved comparison. "
                                 "Start a new comparison or use an explicit estimator seed.")
        if reuse:
            if created_plan:
                # A concurrent caller published this model after our initial
                # directory check. Compare its configuration after our split.
                configuration = configuration_identity({"estimator": estimator, "candidates": candidates})
        else:
            if folds is None:
                # The released splitter orders training rows by original index.
                rows = np.arange(len(y_train))
                folds = [(np.setdiff1d(rows, valid), valid) for valid in held_out]
            cv_seed = repr(seed) if _private_split_rng(seed) else seed
            if shares_split:
                # Explicit folds bypass the splitter in calibration. Bind the
                # original ownership to the CV record for a later partial read.
                cv_seed = {"shared_split_rng": repr(seed)}
            recovery = _ComparisonRecovery(unverified_prefix, serial_final=serial_final, batch=_batch)
            calibrated = calibration_search(estimator, grid, X_train, y_train, cv=5, seed=cv_seed,
                                            threads=threads, checkpoint_path=cv_path,
                                            scoring=scoring, folds=folds,
                                            _comparison_recovery=recovery)
            if serial_final and not _batch.closed:
                model_pins[name] = {"partition": fold_digest, "cv": _cv_pin(calibrated.report)}
            # Calibration captures estimator and grid state after fold selection
            # and before fitting. Its generator may also be the split generator.
            configuration = calibrated.report["identity"]["structural_configuration"]
        comparison_fingerprint = _digest([comparison_context, configuration])
        earlier_retained_state |= retains_state
        earlier_unverified_state |= unverified_state
        # A competing caller can publish partial CV after our path check.
        # Calibration reports what it actually loaded under its writer lock.
        if reuse or recovery.loaded_checkpoint:
            loaded_model_checkpoint = True
            skipped_global_model |= uses_global
            skipped_captures.update((id(value), value) for value, captured in rng_dependencies if captured)
            if serial_final:
                skipped_cv.append((name, uses_global, retains_state, unverified_state,
                                   {id(value): value for value, captured in rng_dependencies if captured}))
        if reuse:
            _check_transform_output(transform_output)
            existing = saved_evaluations[name] if serial_final else json.loads(result_path.read_text(encoding="utf-8"))
            if (existing.get("comparison_fingerprint") != comparison_fingerprint
                    or existing.get("fold_plan_sha256") != fold_digest):
                raise ValueError(f"Existing {name} results are incompatible; create a new run directory.")
            results[name] = existing
            continue
        publish_model = partial(publish, name, comparison_fingerprint, result_path,
                                identity["sklearn_skip_parameter_validation"], calibrated, recovery)
        if serial_final:
            results[name] = None  # Preserve model order while waiting for the remaining CV.
            _batch.checks.append(partial(verify_skipped_cv, name, uses_global, retains_state,
                                          unverified_state, _comparison_has_mutable_parameters(configuration),
                                          rng_dependencies))
            _batch.pending.append(publish_model)
        else:
            publish_model()
    summary = {"author": "Chang Cheng", "phase": "development_comparison",
               "scoring": scoring,
               "fold_plan_sha256": fold_digest,
               "defaults_frozen": False, "tuning_complete": False,
               "remaining_work": ["Expand unconstrained boundary optima and refine interior intervals.",
                                  "Repeat experiments and verify numerical convergence.",
                                  "Compare empirical data before deciding package defaults.",
                                  "Freeze settings before a separate final-test evaluation."],
               "models": results}
    if serial_final:
        _batch.pending.append(partial(write_json, output_dir / "comparison.json", summary))
    else:
        write_json(output_dir / "comparison.json", summary)
    return summary
