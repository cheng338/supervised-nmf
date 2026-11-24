"""Execute explicit classification grids while sharing fold-local representations.

Author: Chang Cheng. This is a computation-saving implementation of an explicit
grid, not a parameter-proposal policy. Screening, factorization and projection
are fitted inside the same training folds for every declared neighbor count.
No full-training refit, held-out evaluation, or package-default decision occurs.
"""

from collections import ChainMap, UserDict, UserList, defaultdict, deque
from contextlib import ExitStack, contextmanager, nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from inspect import signature
from itertools import chain, islice
import json
import math
import multiprocessing
from numbers import Integral
from pathlib import Path
import re
import time
from types import (BuiltinMethodType, CellType, FunctionType, MappingProxyType, MethodType,
                   MethodWrapperType, ModuleType)
import uuid
import warnings

import numpy as np
from scipy import sparse
from sklearn import get_config
from sklearn.base import clone
try:
    from sklearn.callback import CallbackSupportMixin as _CallbackSupportMixin
except ImportError:  # Older scikit-learn versions have no fit callback API.
    _CallbackSupportMixin = ()
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from threadpoolctl import threadpool_info, threadpool_limits

from ._completed_reuse import _ReuseUnavailable, _quiet_record
from .classifier import NMFClassifier, _capture_projection

_STUDY_LABEL_CONVERSION = NMFClassifier._labels_from_probabilities
from ._calibration import _validate_scoring, _score_predictions, _score_fields
from ._feature_names import _input_feature_names
from ._identity import configuration_identity, implementation_sources
from ._labels import _check_labeled_data, _label_values, _validate_class_labels
from ._neighbors import _KNNQueryReuse, _has_euclidean_metric, _is_integer_index
from ._search_diagnostics import screening_grid_diagnostics, mark_inactive_refinement
from .decomposition import FeatureSPA
from .screening import BinaryScreening
from .validation import _validate_fixed_seed, _validate_threads, build_cv_folds, validate_metadata
from .evaluation import (
    _check_transform_output,
    _diagnostics, _digest, _file_lock, _initialized_provenance, _jsonable, _nonconverged, _search_identity,
    _named_matrix, _numeric_schema_estimator, _serial_timing_phase, _warning_records, provenance,
    refinement_diagnostics, write_json,
)


def _write_receipt(path, payload):
    write_json(path, {"payload": payload, "sha256": _digest(payload)})


def _read_receipt(path):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or "payload" not in envelope or envelope.get("sha256") != _digest(envelope["payload"]):
        raise ValueError(f"Study checkpoint failed its integrity check: {path}")
    return envelope["payload"]


def _time_study_fit(model, X, y, result, *, profile=False):
    """Use the same interval for the UTC execution receipt and fit duration."""
    execution = result["fit_execution"] = {}
    with _serial_timing_phase(execution) as finish:
        try:
            model.fit(X, y)
        finally:
            if profile:
                result["timing"]["total_fit_seconds"] = finish()
            else:
                result["final_fit_seconds"] = finish()


def _study_fold_keys(identity, grid):
    fingerprint = _digest(identity)
    representations = {_digest(configuration_identity({
        key: value for key, value in params.items() if key != identity["neighbor_parameter"]}))
        for params in ParameterGrid(grid)}
    for representation in representations:
        for fold in range(len(identity["folds"])):
            yield _digest({"identity": fingerprint, "representation": representation, "fold": fold}), representation, fold


def _cv_work_pending(directories, grids):
    """Resolve the ordinary study grids against their authoritative fold keys."""
    try:
        for directory, grid in zip(directories, grids, strict=True):
            identity = _read_receipt(directory / "identity.json")
            if any(not (directory / "folds" / f"{key}.json").is_file()
                   for key, _, _ in _study_fold_keys(identity, grid)):
                return True
        return False
    except (OSError, ValueError, KeyError, TypeError):
        return True


_CV_CALL_FIELDS = ("current_call_wall_seconds", "new_representation_fits", "reused_representation_folds",
                   "active_threadpools")


def _cv_report_pin(directory):
    report = _read_receipt(directory / "report.json")
    return {"sha256": _digest(report), "call": {name: report[name] for name in _CV_CALL_FIELDS}}


def _seal_study_cv(output, configuration, directories):
    """Pin complete chunk reports before any final fit, including its recoverable state."""
    output = Path(output)
    path = output / "final-cv.json"
    chunks = {directory.relative_to(output).as_posix(): _cv_report_pin(directory) for directory in directories}
    payload = {"configuration_sha256": _digest(configuration), "chunks": chunks}
    if path.exists():
        if _read_receipt(path) != payload:
            raise ValueError("CV evidence changed after final evaluation started.")
    else:
        _write_receipt(path, payload)


def _frozen_study_cv(output, configuration, directories):
    """Detect final-stage archives even if the original CV seal was removed."""
    output = Path(output)
    path = output / "final-cv.json"
    if not path.exists():
        patterns = ("comparison.json", "validation-gate.json", "*/evaluation.json", "*/original-evaluation.json",
                    "*/profile-*.json", "profiles/*/profiling.json", "*/*/development-evaluation.json", "*/*/.fit-recovery")
        if any(any(output.glob(pattern)) for pattern in patterns):
            # Existing reports still permit strict replay if the seal itself
            # was lost. Recovering an absent aggregate needs its original pin.
            if not all((directory / "report.json").is_file() for directory in directories):
                raise ValueError("Final evaluation evidence exists without the original CV report or seal; fresh CV is closed.")
            return [_cv_report_pin(directory) for directory in directories]
        return None
    payload = _read_receipt(path)
    if payload["configuration_sha256"] != _digest(configuration):
        raise ValueError("The final CV seal belongs to another study configuration.")
    return [payload["chunks"][directory.relative_to(output).as_posix()] for directory in directories]


def _evaluation_cv_repairs(work, directories, pins):
    """Verify every committed fold before rebuilding a derived report or opening validation."""
    repairs, repair_pins = [], []
    for index, (item, directory) in enumerate(zip(work, directories, strict=True)):
        chunk = item[2]
        pin = None if pins is None else pins[index]
        try:
            identity = _read_receipt(directory / "identity.json")
            fingerprint = _digest(identity)
            report_path = directory / "report.json"
            report = _read_receipt(report_path) if report_path.is_file() else None
            if report is not None:
                if pin is not None and _digest(report) != pin["sha256"]:
                    raise ValueError(f"CV report differs from its frozen evidence; fresh CV is closed: {report_path}")
                if report["identity"] != identity or report["fingerprint"] != fingerprint:
                    raise ValueError(f"CV report identity differs from its checkpoint: {report_path}")
                rows = {row["key"]: row for row in report["representation_folds"]}
                if len(rows) != len(report["representation_folds"]):
                    raise ValueError(f"Duplicate CV fold in {report_path}")
            for key, representation, fold in _study_fold_keys(identity, chunk["grid"]):
                saved = _read_receipt(directory / "folds" / f"{key}.json")
                if (saved["key"] != key or saved["identity"] != fingerprint or saved["state"] != "completed"
                        or saved["representation_key"] != representation or saved["fold"] != fold
                        or (report is not None and rows.pop(key, None) != saved)):
                    raise ValueError(f"CV fold differs from the declared checkpoint: {directory}, fold {fold}")
            if report is None:
                repairs.append(item)
                repair_pins.append(pin)
            elif rows:
                raise ValueError(f"Unexpected CV folds in {report_path}")
        except FileNotFoundError as error:
            raise ValueError(f"All protocol CV jobs must have complete fold receipts before validation; "
                             f"fresh CV is closed during evaluation: {error.filename}") from error
    return repairs, repair_pins


def _protocol_cv_chunks(output, configuration):
    chunks = [(str(output), plan, chunk, configuration["sources"])
              for plan in configuration["plans"] for chunk in plan["chunks"]]
    directories = [Path(output) / plan["job"]["id"] / chunk["id"] for _, plan, chunk, _ in chunks]
    return chunks, directories


def _replay_frozen_cv(work, pins, worker, workers):
    """A clean child validates each original grid and can only reuse committed folds."""
    if work:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(worker, item, reuse_only=True, _frozen_report=pin)
                       for item, pin in zip(work, pins, strict=True)]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    raise ValueError("A frozen CV fold or identity is missing or incompatible; fresh CV is closed.")
                yield result


def _validated_cv_wall_receipts(output, config_sha256, *, read_receipt=_read_receipt):
    """Read attempts in admission order, including a start interrupted during publication."""
    directory = Path(output) / "cv-timing"
    index = directory / "attempts.index"
    catalog = read_receipt(index) if index.exists() else {"attempts": [], "terminal_sha256": None}
    if not isinstance(catalog, dict) or set(catalog) != {"attempts", "terminal_sha256"}:
        raise ValueError("Invalid CV timing attempt index.")
    ids = catalog["attempts"]
    if (not isinstance(ids, list) or not all(isinstance(s, str) and re.fullmatch(r"[0-9a-f]{32}", s) for s in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid CV timing attempt index.")
    paths = {path.stem: path for path in directory.glob("*.json")}
    if paths.keys() - set(ids):
        raise ValueError("Unregistered CV timing receipt. Preserve the original study output.")
    rows = []
    previous = None
    for i, attempt in enumerate(ids):
        path = paths.get(attempt)
        if path is None:
            path = directory / ".starting"
            if i != len(ids) - 1 or not path.exists():
                raise ValueError(f"Missing registered CV timing receipt: {attempt}")
        row = read_receipt(path)
        if (row.get("attempt_id") != attempt or row.get("configuration_sha256") != config_sha256
                or row.get("previous_sha256") != previous
                or (path.name == ".starting" and row.get("state") != "admitted")):
            raise ValueError(f"CV timing receipt differs from its registered attempt: {path}")
        rows.append((path, row))
        previous = _digest(row)
    if catalog["terminal_sha256"] is not None and catalog["terminal_sha256"] != previous:
        raise ValueError("The last CV timing receipt differs from its terminal record.")
    return rows


@contextmanager
def _cv_wall_attempt(output, config_sha256, *, active, workers, native_threads, preexisting_work=False,
                     seal_sha256=None):
    """Journal one coordinator's CV interval while its caller holds the study reservation.

    A completed replay adds zero. A killed coordinator leaves a running receipt
    with unknown duration, so a later invocation can report the measured subtotal.
    """
    directory = Path(output) / "cv-timing"
    receipts = _validated_cv_wall_receipts(output, config_sha256)
    previous = [row for _, row in receipts]
    if previous and previous[-1]["seal_sha256"] is not None and previous[-1]["seal_sha256"] != seal_sha256:
        raise ValueError("CV timing history belongs to an existing final-stage seal.")
    staged = directory / ".starting"
    if receipts and receipts[-1][0] == staged:
        staged.replace(directory / (previous[-1]["attempt_id"] + ".json"))
    known = math.fsum(row["cv_wall_seconds_this_invocation"] for row in previous
                      if row["cv_wall_seconds_this_invocation"] is not None)
    incomplete = (not previous and preexisting_work) or any(
        row["cv_wall_seconds_this_invocation"] is None or row["prior_wall_time_unknown"]
        for row in previous)
    attempt = uuid.uuid4().hex
    path = directory / (attempt + ".json")
    record = {"attempt_id": attempt, "previous_sha256": _digest(previous[-1]) if previous else None,
              "seal_sha256": seal_sha256, "configuration_sha256": config_sha256, "state": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "workers": workers, "native_threads": native_threads, "cv_work_pending": active,
              "prior_wall_time_unknown": incomplete,
              "cv_wall_seconds_this_invocation": None if active else 0.0}
    # Reserve this exact attempt before work starts. An interrupted reservation
    # either never entered the index or still has a recoverable start receipt.
    _write_receipt(staged, {**record, "state": "admitted", "cv_wall_seconds_this_invocation": 0.0})
    index = directory / "attempts.index"
    catalog = {"attempts": [row["attempt_id"] for row in previous] + [attempt], "terminal_sha256": None}
    _write_receipt(index, catalog)
    staged.replace(path)
    _write_receipt(path, record)
    start = time.perf_counter()
    try:
        yield record
    except BaseException as error:
        record.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    else:
        record["state"] = "completed"
    finally:
        elapsed = time.perf_counter() - start if active else 0.0
        recorded = known + elapsed
        terminal = dict(record, finished_utc=datetime.now(timezone.utc).isoformat(),
                        cv_wall_seconds_this_invocation=elapsed, cv_wall_seconds_recorded=recorded,
                        cv_wall_seconds_cumulative=None if incomplete else recorded)
        _write_receipt(path, terminal)
        # The enclosing command can now cite a durable terminal receipt even
        # if publishing its independent index fingerprint is interrupted.
        record.update(terminal)
        _write_receipt(index, {**catalog, "terminal_sha256": _digest(record)})


def _seal_final_settings(output, settings_hashes):
    """Commit the exact selections before the first final call while holding the study lock."""
    path = Path(output) / "final-start.json"
    if path.exists():
        if _read_receipt(path)["settings_sha256"] != settings_hashes:
            raise ValueError("Frozen settings changed after final evaluation began.")
        return
    _write_receipt(path, {"settings_sha256": settings_hashes, "started_utc": datetime.now(timezone.utc).isoformat()})


@contextmanager
def _checkpoint_lock(directory):
    """Coordinate cache writers without deleting files left by interrupted runs."""
    with _file_lock(Path(directory) / ".study.lock",
                    "Another process is writing this study checkpoint directory."):
        yield


def _layout(estimator):
    if type(estimator) is NMFClassifier:
        if estimator.classifier != "knn":
            raise TypeError("calibration_knn_grid requires classifier='knn'; use calibration_search for other classifiers.")
        return "nmf", "n_neighbors", {"weights", "classifier", "logistic_C", "logistic_max_iter",
                                       "logistic_tol", "logistic_class_weight",
                                       "knn_normalization", "svm_C", "svm_max_iter",
                                       "svm_tol", "svm_class_weight",
                                       "decision_threshold", "positive_label"}
    if type(estimator) is KNeighborsClassifier:
        return "raw", "n_neighbors", set(estimator.get_params()) - {"n_neighbors"}
    if type(estimator) is Pipeline and type(estimator.steps[-1][1]) is KNeighborsClassifier:
        name, classifier = estimator.steps[-1]
        forbidden = {name, "steps"} | {f"{name}__{key}" for key in classifier.get_params() if key != "n_neighbors"}
        return "pipeline", f"{name}__n_neighbors", forbidden
    raise TypeError("Use NMFClassifier with classifier='knn', a standard Pipeline ending in KNeighborsClassifier, or raw KNeighborsClassifier.")


def _rank_axis_kind(estimator, candidates, key):
    """Identify a numeric rank axis, including grids that replace an entire step."""
    if type(estimator) is NMFClassifier:
        return "rank" if key == "n_components" else None
    if type(estimator) is not Pipeline or not key.endswith("__n_components"):
        return None
    step = key.rsplit("__", 1)[0]
    if step not in estimator.named_steps:
        return None
    original = estimator.named_steps[step]
    owners = set()
    for params in candidates:
        if key in params:
            owner = type(params.get(step, original))
            owners.add(owner is PCA or owner is NMF or owner is FeatureSPA)
            if len(owners) > 1:
                return "mixed"
    return "rank" if True in owners else None


def _rank_coordinates(estimator, candidates, selected):
    axes = {}
    for key, value in (selected or {}).items():
        kind = _rank_axis_kind(estimator, candidates, key)
        if kind == "rank":
            axes[key] = ([params[key] for params in candidates if key in params], value)
        elif kind == "mixed":
            axes[key] = None  # The same parameter name describes different quantities.
    return axes


def _runtime_identity(estimator, threads, candidates=(), *, _environment=None, _configuration=None):
    info = _initialized_provenance(threads) if _environment is None else _environment(threads)
    runtime_keys = ("user_api", "internal_api", "prefix", "filepath", "version", "threading_layer", "architecture")
    libraries = sorted([{key: pool.get(key) for key in runtime_keys} for pool in info["threadpools"]],
                       key=lambda row: row.get("filepath") or "")
    if _configuration is None:
        sources = implementation_sources(estimator, candidates)
    else:
        sources = {}
        _configuration.append(configuration_identity({"estimator": estimator, "candidates": candidates},
                                                      _source_records=sources))
    return {"python": info["python"], "platform": info["platform"], "versions": info["versions"],
            "sklearn_skip_parameter_validation": get_config()["skip_parameter_validation"],
            "sklearn_transform_output": get_config()["transform_output"],
            "package_source": info["source_sha256"], "estimator_source": sources, "numerical_libraries": libraries}


def _fixed_randomness(estimator):
    # Sharing one randomized fit across k values requires a declared common seed.
    for name, value in estimator.get_params(deep=True).items():
        if name == "random_state" or name.endswith("__random_state"):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= 2 ** 32 - 1:
                raise ValueError(f"{name} must be a fixed integer seed for exact shared-representation comparisons.")


class _KNNTrainingStateCompatibilityError(RuntimeError):
    """The original KNN training state cannot be shared across neighbor counts."""


def _fit_callback_parameters(value):
    """Read the public sklearn constructor parameters, including delegated methods."""
    get_params = getattr(value, "get_params", None)
    if not callable(get_params):
        return None
    try:
        parameters = get_params(deep=False)
    except TypeError as error:
        # An unrelated API can use the same name (e.g. email.message.Message).
        # Propagate exceptions from a method that accepts the sklearn call.
        if error.__traceback__.tb_next is not None:
            raise
        try:
            method_signature = signature(get_params, follow_wrapped=False)
        except (TypeError, ValueError):
            method_signature = None
        if method_signature is not None:
            try:
                method_signature.bind(deep=False)
            except TypeError:
                return None
        raise
    if issubclass(type(parameters), dict):
        return dict.values(parameters)
    return None


def _fit_callback_children(value):
    """Expose configuration references without executing a transform or inspecting arbitrary state."""
    kind = type(value)
    if issubclass(kind, dict):
        if sparse.issparse(value):  # A DOK matrix's numeric entries are data.
            return None
        members = chain(dict.keys(value), dict.values(value))
        if issubclass(kind, defaultdict):
            members = chain(members, (defaultdict.default_factory.__get__(value),))
        return members
    if kind is MappingProxyType:
        return chain(value.keys(), value.values())
    if issubclass(kind, (ChainMap, UserDict, UserList)):
        # Inspect the documented backing store, including shadowed ChainMap entries.
        storage = value.maps if issubclass(kind, ChainMap) else value.data
        return (storage,)
    for container in (tuple, list, set, frozenset, deque):
        if issubclass(kind, container):
            return container.__iter__(value)
    if kind is slice:
        return (value.start, value.stop, value.step)
    if issubclass(kind, partial):
        if kind is partial:
            return (value.func, value.args, value.keywords)
        # A subclass may override attribute access while partial still calls its stored target.
        return (partial.func.__get__(value), partial.args.__get__(value), partial.keywords.__get__(value))
    if kind is MethodType:
        return (value.__self__, value.__func__)
    if kind is BuiltinMethodType or kind is MethodWrapperType:
        return (value.__self__,)
    if kind is FunctionType:
        return (value.__defaults__, value.__kwdefaults__, value.__closure__)
    if kind is CellType:
        try:
            return (value.cell_contents,)
        except ValueError:  # An empty closure cell holds no object.
            return None
    if issubclass(kind, (np.ndarray, np.generic)):
        array = np.asarray(value)
        dtype = array.dtype
        if dtype.names and dtype.hasobject:
            return chain((dtype,), (array[name] for name in dtype.names if dtype[name].hasobject))
        if dtype.kind == "O":
            return chain((dtype,), array.flat)
        if dtype.names or dtype.metadata is not None or dtype.subdtype or dtype.kind == "T":
            return (dtype,)
        return None
    if issubclass(kind, np.dtype):
        if value.kind == "T":  # A NumPy StringDType can retain a missing-value sentinel.
            return (value.metadata, getattr(value, "na_object", None))
        return (value.metadata, value.fields, value.subdtype)
    return None


def _has_registered_fit_callbacks(estimator):
    if not _CallbackSupportMixin:
        return False
    seen = {}

    def visit(value):
        kind = type(value)
        if (value is None or kind is str or kind is bytes or kind is int or kind is bool
                or kind is float or kind is complex or issubclass(kind, (type, ModuleType))):
            return False
        if id(value) in seen:
            return False
        if issubclass(kind, _CallbackSupportMixin) and getattr(value, "_skl_callbacks", None):
            return True
        # A carrier can also declare constructor parameters beyond its contents.
        parameters = _fit_callback_parameters(value)
        children = _fit_callback_children(value)
        if parameters is None and children is None:
            return False
        # Keep visited carriers alive: get_params and array field access may
        # produce temporary objects whose addresses Python could otherwise reuse.
        seen[id(value)] = value
        return any(visit(child) for child in chain(parameters or (), children or ()))

    return visit(estimator)


@contextmanager
def _observe_knn_grid_fit(estimator, observer):
    """Observe one clone's native _fit while keeping its copies independently usable."""
    state = vars(estimator)
    native = getattr(estimator, "_fit", None)
    if not callable(native) or "_fit" in state or "__getstate__" in state:
        raise _KNNTrainingStateCompatibilityError("Cannot identify the native fit used by shared KNN calibration.")

    def observed(*args, **kwargs):
        return observer(native, *args, **kwargs)

    def remove_bindings(mapping):
        if mapping.get("_fit") is observed:
            del mapping["_fit"]
        if mapping.get("__getstate__") is copied_state:
            del mapping["__getstate__"]

    def copied_state():
        copied = type(estimator).__getstate__(estimator).copy()
        remove_bindings(copied)
        return copied

    state["_fit"] = observed
    state["__getstate__"] = copied_state
    try:
        yield
    finally:
        remove_bindings(state)


def _fit_pipeline_for_shared_knn(pipeline, X, y):
    """Identify the native KNN input during the single shareable representation fit."""
    head = None
    calls = 0
    adopted_index = False
    labels_match = False

    def fit_head(native, X, y=None, *args, **kwargs):
        nonlocal calls, adopted_index, labels_match
        calls += 1
        result = native(X, y, *args, **kwargs)
        labels_match = y is fold_labels or np.array_equal(y, fold_labels)
        if type(X) is not np.ndarray:
            try:
                from sklearn.neighbors import BallTree, KDTree
                from sklearn.neighbors._base import NeighborsBase
                adopted_index = isinstance(X, (KDTree, BallTree, NeighborsBase))
            except Exception as error:
                raise _KNNTrainingStateCompatibilityError("Cannot identify the fitted KNN input.") from error
        return result

    def preprocess(native, *args, **kwargs):
        nonlocal head
        values = native(*args, **kwargs)
        head = pipeline.steps[-1][1]
        if type(head) is not KNeighborsClassifier:
            raise _KNNTrainingStateCompatibilityError("Shared calibration requires a stock KNN classifier.")
        hooks.enter_context(_observe_knn_grid_fit(head, fit_head))
        return values

    fold_labels = y
    with ExitStack() as hooks:
        hooks.enter_context(_observe_knn_grid_fit(pipeline, preprocess))
        pipeline.fit(X, y)
    if head is not pipeline.steps[-1][1] or calls != 1 or not labels_match:
        raise _KNNTrainingStateCompatibilityError(
            "Shared calibration could not establish one final KNN fit on the fold labels. "
            "Use calibration_search with refit=False to fit each neighbor count independently.")
    return adopted_index


def _stored_training_representation(fitted, kind):
    """Recover the exact training representation used by the original head fit."""
    if kind == "nmf":
        if type(fitted.classifier_) is not Pipeline:
            return fitted.training_coordinates_, fitted.classifier_
        steps = fitted.classifier_.steps
        if (len(steps) != 2 or type(steps[0][1]) is not StandardScaler
                or type(steps[-1][1]) is not KNeighborsClassifier):
            raise _KNNTrainingStateCompatibilityError("Unsupported fitted NMF KNN coordinate pipeline.")
        knn = steps[-1][1]
    else:
        knn = fitted.steps[-1][1] if kind == "pipeline" else fitted

    # A second training transform can change the numerical representation.
    # Use the fitted matrix directly; for a buffer obtain a single no-copy view.
    cause = None
    detail = "could not read the fitted matrix or its dimensions"
    try:
        coordinates = knn._fit_X
        if type(coordinates) is not np.ndarray:
            if sparse.issparse(coordinates):
                data = coordinates.data
                if not isinstance(data, np.ndarray) or np.ndarray.dtype.__get__(data) != coordinates.dtype:
                    raise TypeError("Unusable fitted sparse coordinate storage.")
            elif isinstance(coordinates, np.ndarray):
                # Bypass subclass overrides of dtype, shape, and view.
                coordinates = np.ndarray.view(coordinates, type=np.ndarray)
            else:
                coordinates = np.asarray(memoryview(coordinates))
                if coordinates.flags.owndata:
                    raise ValueError("The training buffer requires a copy.")
        shape, dtype = coordinates.shape, coordinates.dtype
        rows = knn.n_samples_fit_
        missing = object()
        width = getattr(knn, "n_features_in_", missing)
        # sklearn can retain temporal dtypes from a pipeline transformer.
        if (dtype.kind in "biufmM" and isinstance(shape, (tuple, list)) and len(shape) == 2
                and all(_is_integer_index(n) and n > 0 for n in shape)
                and _is_integer_index(rows) and shape[0] == rows
                and ((width is missing and kind == "pipeline")
                     or (_is_integer_index(width) and shape[1] == width))):
            return coordinates, knn
        declared_width = "missing" if width is missing else repr(width)
        detail = (f"shape={shape!r}, dtype={dtype}, n_samples_fit_={rows!r}, "
                  f"n_features_in_={declared_width}")
    except Exception as error:
        # A dependency can replace an attribute with a property or an opaque
        # exporter. Its read failure must also leave this fold uncommitted.
        cause = error
    from sklearn import __version__
    raise _KNNTrainingStateCompatibilityError(
        f"scikit-learn {__version__} did not provide usable fitted KNN training coordinates "
        f"for the {kind} layout ({detail}).") from cause


def _validation_representation(fitted, kind, X):
    if kind == "nmf":
        coordinates = fitted.transform(X)
        if type(fitted.classifier_) is Pipeline:
            return fitted.classifier_[:-1].transform(coordinates)
        return coordinates
    if kind == "pipeline":
        # Pipeline.predict checks the complete model, then transforms each
        # active intermediate step. A fit_transform-only step can be valid here.
        check_is_fitted(fitted)
        for _, transformer in islice(fitted.steps, len(fitted.steps) - 1):
            if transformer is not None and transformer != "passthrough":
                X = transformer.transform(X)
    return X


def _finite(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(np.isfinite(values).all())


def _index_query_values(query):
    """Obtain numeric values for the fold report while preserving the classifier's query."""
    if type(query) is np.ndarray and query.ndim == 2 and query.dtype.kind in "biuf":
        return query
    # sklearn handles sparse formats, nullable columns and numeric object arrays.
    # Each selected head emits its own conversion warnings on the original input.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return check_array(query, accept_sparse="csr", ensure_all_finite=False,
                           ensure_min_samples=0, ensure_min_features=0)


class _PreparedScreening(BinaryScreening):
    """Carry a fold-local screen through a standard uncached Pipeline fit."""

    def __init__(self, reference):
        self.reference = reference
        super().__init__(**reference.get_params(deep=False))

    def fit(self, X, y):
        names = _input_feature_names(X)
        self._fit_from_statistics(self.reference)
        if names is not None:
            self.feature_names_in_ = names
        return self


def _screening_template(estimator, kind):
    if kind == "nmf" and estimator.screening in {"percentile", "absolute"}:
        return BinaryScreening(estimator.screening, estimator.percentile, estimator.score_threshold)
    # Arbitrary pipelines may change values, sample order, metadata or fitting
    # semantics. Only the two exact standard layouts below use this shortcut.
    if kind == "pipeline" and estimator.memory is None:
        steps = estimator.steps
        if (len(steps) in {2, 3} and type(steps[0][1]) is BinaryScreening
                and (len(steps) == 2 or type(steps[1][1]) is PCA)):
            return steps[0][1]
    return None


def _plain_order_parameters(value, depth=0):
    """Limit reordering to ordinary constructor data without stateful hooks."""
    if depth > 16:
        return False
    if value is None or type(value) in {bool, int, float, complex, str, bytes}:
        return True
    if isinstance(value, np.generic):
        return np.dtype(type(value)).type is type(value) and not value.dtype.hasobject
    if type(value) in {list, tuple}:
        return all(_plain_order_parameters(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(_plain_order_parameters(key, depth + 1)
                   and _plain_order_parameters(item, depth + 1)
                   for key, item in value.items())
    return False


def _scalar_independent_template(estimator):
    """Recognize fixed pipelines whose parameters cannot share a record's storage."""
    if type(estimator) is not Pipeline or type(estimator.steps) is not list:
        return False

    def primitive_settings(model, omit=None):
        def plain(values):
            return all(any(type(value) is cls for cls in (type(None), bool, int, float, str))
                       for name, value in values.items() if name != omit)

        # Check instance overrides before calling a stock parameter accessor.
        return plain(vars(model)) and plain(model.get_params(deep=False))

    if not primitive_settings(estimator, omit="steps"):
        return False
    for entry in estimator.steps:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
            return False
        step = entry[1]
        if not any(type(step) is cls for cls in (FunctionTransformer, StandardScaler, PCA, KNeighborsClassifier)):
            return False
        if not primitive_settings(step):
            return False
    return True


def _fold_order_eligible(configured, kind):
    """Recognize order-independent layouts; an unknown setting keeps old order."""
    try:
        if kind == "nmf":
            if not _plain_order_parameters(configured.get_params(deep=False)):
                return False
            if type(configured.weights) is not str or configured.weights not in {"uniform", "distance"}:
                return False
        elif kind == "pipeline":
            steps = configured.steps
            if (len(steps) not in {2, 3} or type(steps[0][1]) is not BinaryScreening
                    or type(steps[-1][1]) is not KNeighborsClassifier
                    or (len(steps) == 3 and type(steps[1][1]) is not PCA)):
                return False
            shell = {name: value for name, value in configured.get_params(deep=False).items()
                     if name != "steps"}
            if not _plain_order_parameters(shell) or getattr(configured, "transform_input", None) is not None:
                return False
            if not all(type(name) is str and _plain_order_parameters(part.get_params(deep=False))
                       for name, part in steps):
                return False
            head = steps[-1][1]
            if not (type(head.weights) is str and head.weights in {"uniform", "distance"}
                    and type(head.metric) is str and head.metric in {"euclidean", "minkowski"}
                    and (head.metric == "euclidean" or head.p == 2)
                    and (head.metric_params is None
                         or (type(head.metric_params) is dict and not head.metric_params))):
                return False
        else:
            return False
        # Check plain values first: rejected parameters must not gain extra
        # comparisons, validation hooks or other observable side effects.
        template = _screening_template(configured, kind)
        if template is None:
            return False
        template._validate_settings()
        return True
    except Exception:
        # Eligibility must not move an existing fit failure into preflight.
        return False


def _check_rank_feasibility(estimator, kind, n_train, n_features):
    if kind == "nmf":
        rank = estimator.n_components
        if (isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral)
                or not 1 <= rank <= min(n_train, n_features)):
            raise ValueError("n_components must be between 1 and min(training samples, selected features).")
    elif kind == "pipeline":
        steps = estimator.steps
        standard = ((len(steps) == 2 and type(steps[0][1]) is PCA)
                    or (len(steps) == 3 and type(steps[0][1]) is BinaryScreening
                        and type(steps[1][1]) is PCA))
        if standard:
            pca = steps[-2][1]
            rank = pca.n_components
            if isinstance(rank, Integral) and not isinstance(rank, (bool, np.bool_)):
                limit = min(n_train, n_features)
                if rank > limit or (pca.svd_solver == "arpack" and rank >= limit):
                    raise ValueError("PCA n_components exceeds the limit for this training fold's selected features and solver.")


class _ScreeningCache:
    """Bound to one call's immutable data and materialized training folds."""

    def __init__(self):
        self.statistics = {}
        self.selectors = {}

    def prepare(self, template, fold, X, y, train, record):
        # Validate the actual requested screen even when another cutoff already
        # populated this fold's cache.
        template._validate_settings()
        if fold not in self.statistics:
            start = time.perf_counter()
            try:
                base = BinaryScreening(screening="absolute", score_threshold=0.0).fit(X[train], y[train])
                self.statistics[fold] = base
            except Exception as error:
                self.statistics[fold] = error
            finally:
                record["screening_statistics_seconds"] = time.perf_counter() - start
                record["screening_statistics_fit_performed"] = True
        base = self.statistics[fold]
        if isinstance(base, Exception):
            raise base
        key = (fold, _digest(template.get_params(deep=False)))
        if key not in self.selectors:
            start = time.perf_counter()
            try:
                self.selectors[key] = clone(template)._fit_from_statistics(base)
            except Exception as error:
                self.selectors[key] = error
            finally:
                record["screening_selection_seconds"] = time.perf_counter() - start
        selector = self.selectors[key]
        if isinstance(selector, Exception):
            raise selector
        return selector



class StatisticsRecoveryError(ValueError):
    """Authoritative statistics evidence or its current source is invalid."""


class _PersistentScreeningCache(_ScreeningCache):
    """Static operation-owned extension; no global patch or dynamic class."""

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.events = {}

    def prepare(self, template, fold, X, y, train, record, *, caught):
        template._validate_settings()
        new_statistics = fold not in self.statistics
        if new_statistics:
            event = None
            restored = self.store.load(fold, record, X.shape[1], len(train))
            if restored is not None:
                base, event = restored
            else:
                warning_count = len(caught)
                start = time.perf_counter()
                try:
                    base = BinaryScreening(screening='absolute', score_threshold=0.0).fit(X[train], y[train])
                    self.statistics[fold] = base
                except Exception as error:
                    self.statistics[fold] = error
                finally:
                    record['screening_statistics_seconds'] = time.perf_counter() - start
                    record['screening_statistics_fit_performed'] = True
                    self.store.current['actual_statistics_fits'] += 1
                base = self.statistics[fold]
                if type(base) is BinaryScreening:
                    event = self.store.publish(fold, base, record['screening_statistics_seconds'], record,
                                               silent=len(caught) == warning_count)
        else:
            base = self.statistics[fold]
            event = self.events.get(fold)
        if isinstance(base, Exception):
            raise base
        if event is not None:
            try:
                if new_statistics:
                    if restored is not None:
                        self.statistics[fold] = base
                    self.events[fold] = event
                record['screening_statistics_event_id'] = event['event_id']
                record['screening_statistics_event_reused'] = not record['screening_statistics_fit_performed']
            except Exception as error:
                raise StatisticsRecoveryError('Cannot link the committed statistics event to the fold') from error
        key = (fold, _digest(template.get_params(deep=False)))
        if key not in self.selectors:
            start = time.perf_counter()
            try:
                self.selectors[key] = clone(template)._fit_from_statistics(base)
            except Exception as error:
                self.selectors[key] = error
            finally:
                record['screening_selection_seconds'] = time.perf_counter() - start
        selector = self.selectors[key]
        if isinstance(selector, Exception):
            raise selector
        return selector

    def close(self):
        self.statistics.clear()
        self.selectors.clear()
        self.events.clear()
        self.store.close()
        self.store = None


def _check_captured_folds(directory, fingerprint):
    # Skipping a committed fold also skips its recorded callback state change.
    for path in (directory / "folds").glob("*.json"):
        saved = _read_receipt(path)
        if saved.get("identity") == fingerprint and saved.get("captured_configuration_changed"):
            raise ValueError("A captured random source or callback configuration changed during a saved fold. "
                             "Repeat the study in a fresh directory.")


def _preflight_statistics_receipts(directory, fingerprint, representation_keys, cv, store):
    """Reject invalid prior event consumers before extending a partial study."""
    folds_directory = Path(directory) / "folds"
    try:
        mode = folds_directory.stat().st_mode
    except FileNotFoundError:
        return
    from stat import S_ISDIR
    if not S_ISDIR(mode):
        raise StatisticsRecoveryError("Study fold receipts require a directory")
    saved_receipts = []
    for representation_key in representation_keys:
        for fold in range(cv):
            key = _digest({"identity": fingerprint, "representation": representation_key, "fold": fold})
            path = folds_directory / f"{key}.json"
            try:
                record = _read_receipt(path)
            except FileNotFoundError:
                continue
            if record.get("identity") != fingerprint or record.get("key") != key or record.get("state") != "completed":
                raise ValueError("The representation checkpoint belongs to another run or is incomplete.")
            saved_receipts.append(record)
    store.validate_consumers(saved_receipts)


def _kd_training_snapshot(models, coordinates):
    """Remember the coordinates used for fresh trees that could share a query."""
    if (type(coordinates) is not np.ndarray or not coordinates.flags.c_contiguous
            or coordinates.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        return None
    eligible = sum(type(method := getattr(model, "_fit_method", None)) is str and method == "kd_tree"
                   and getattr(model, "_fit_X", None) is coordinates
                   and type(getattr(model, "weights", None)) is str and model.weights in {"uniform", "distance"}
                   and _has_euclidean_metric(model)
                   for model in models.values())
    if eligible < 2:
        return None
    try:
        return coordinates.copy()
    except MemoryError:
        return None


def _unchanged_kd_coordinates(snapshot, coordinates):
    if (snapshot is None or coordinates.shape != snapshot.shape or coordinates.dtype != snapshot.dtype
            or not coordinates.flags.c_contiguous):
        return False
    # Compare float32/64 bit patterns without special treatment of signed zeros.
    bits = np.uint32 if snapshot.itemsize == 4 else np.uint64
    try:
        return np.array_equal(snapshot.view(bits), coordinates.view(bits))
    except MemoryError:
        return False


def _fit_neighbor_heads(knn_template, head_training_input, training_labels, neighbors, n_train, record):
    models, fit_warnings = {}, {}
    for neighbors_count in neighbors:
        result = {"n_neighbors": neighbors_count, "fit_seconds": 0.0, "prediction_seconds": 0.0}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                classifier = clone(knn_template).set_params(n_neighbors=neighbors_count)
                if neighbors_count > n_train:
                    raise ValueError("n_neighbors exceeds the number of observations in this training fold.")
                start = time.perf_counter()
                try:
                    classifier.fit(head_training_input, training_labels)
                finally:
                    result["fit_seconds"] = time.perf_counter() - start
                models[neighbors_count] = classifier
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, error=str(error))
        fit_warnings[neighbors_count] = list(caught)
        result["warnings"] = _warning_records(caught)
        record["neighbors"][str(neighbors_count)] = result
    return models, fit_warnings


def _run_representation_fold(estimator, kind, neighbor_parameter, representation_parameters,
                             neighbors, X, y, train, validation, record, *, screening_cache=None,
                             scoring="accuracy", _feature_names=None):
    fitted = None
    projection = None
    caught = []
    record.update(representation_fit_seconds=0.0, projection_seconds=0.0,
                  representation_fit_performed=False, neighbors={},
                  neighbor_reuse_setup_seconds=0.0, neighbor_query_seconds=0.0,
                  neighbor_queries=[],
                  screening_statistics_seconds=0.0, screening_selection_seconds=0.0,
                  screening_statistics_fit_performed=False, preflight_rejected=False)
    if neighbors and min(neighbors) > len(train):
        record.update(status="failed", error_type="ValueError",
                      error="n_neighbors exceeds the number of observations in this training fold.",
                      failure_stage="neighbor_feasibility", preflight_rejected=True,
                      warnings=[])
        return
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fitted = clone(estimator).set_params(**clone(representation_parameters, safe=False))
                # This initial KNN fit is only the carrier for the representation;
                # its small cost is retained in the shared fit timer exactly once.
                fitted.set_params(**{neighbor_parameter: 1})
                selected = None
                template = _screening_template(fitted, kind)
                failure_stage = "screening"
                if template is not None and screening_cache is not None:
                    if isinstance(screening_cache, _PersistentScreeningCache):
                        selected = screening_cache.prepare(template, record["fold"], X, y, train, record, caught=caught)
                    else:
                        selected = screening_cache.prepare(template, record["fold"], X, y, train, record)
                    record["screening_cache_used"] = True
                    record["n_selected_features"] = selected.n_features_selected_
                failure_stage = "rank_feasibility"
                # For unknown preprocessing layouts, leave all validation to
                # their ordinary fit: the input dimension may legitimately change.
                if kind == "nmf" or selected is not None or (kind == "pipeline" and len(fitted.steps) == 2):
                    _check_rank_feasibility(fitted, kind, len(train),
                                            selected.n_features_selected_ if selected is not None else X.shape[1])
                names = (_feature_names if _feature_names is not None and kind == "pipeline"
                         and _numeric_schema_estimator(fitted) else None)
                if selected is not None and kind == "pipeline":
                    name, screen = fitted.steps[0]
                    prepared = _PreparedScreening(selected)
                    # Statistics depend on numeric values. Output mode belongs
                    # to this candidate, even when several cutoffs share a mask.
                    mode = getattr(screen, "_sklearn_output_config", {}).get("transform")
                    fitted.steps[0] = (name, prepared.set_output(transform=mode))
                failure_stage = "fit"
                record["representation_fit_performed"] = True
                start = time.perf_counter()
                try:
                    training_labels = y[train]
                    if selected is not None and kind == "nmf":
                        fitted._fit_with_screening(X[train], training_labels, selected)
                    elif kind == "pipeline":
                        adopted_index = _fit_pipeline_for_shared_knn(
                            fitted, _named_matrix(X[train], names), training_labels)
                    else:
                        fitted.fit(_named_matrix(X[train], names), training_labels)
                finally:
                    record["representation_fit_seconds"] = time.perf_counter() - start
                training_coordinates, knn_template = _stored_training_representation(fitted, kind)
                head_training_input = training_coordinates
                if kind == "pipeline" and adopted_index:
                    # A fitted index can carry a different metric or tree. A
                    # numeric fit must choose its own backend for each k.
                    head_training_input = knn_template
                numeric_pipeline = kind == "pipeline" and head_training_input is not knn_template
                snapshot = None
                if kind == "pipeline":
                    # Build each head before validation can edit the transformer's
                    # training coordinates or labels. Each count chooses its native
                    # backend, and an adopted index retains its original tree.
                    models, fit_warnings = _fit_neighbor_heads(
                        knn_template, head_training_input, training_labels, neighbors, len(train), record)
                    # A numeric refit uses the validated array. Keep the input column
                    # names sklearn recorded on the original fit for query validation.
                    start = time.perf_counter()
                    if hasattr(knn_template, "feature_names_in_"):
                        for model in models.values():
                            model.feature_names_in_ = knn_template.feature_names_in_.copy()
                    elif numeric_pipeline:
                        snapshot = _kd_training_snapshot(models, head_training_input)
                    record["neighbor_reuse_setup_seconds"] = time.perf_counter() - start
                failure_stage = "projection"
                start = time.perf_counter()
                try:
                    with _capture_projection(fitted) as projection:
                        validation_coordinates = _validation_representation(
                            fitted, kind, _named_matrix(X[validation], names))
                finally:
                    record["projection_seconds"] = time.perf_counter() - start
                validation_values = (_index_query_values(validation_coordinates)
                                     if kind == "pipeline" else validation_coordinates)
                if not _finite(training_coordinates) or not _finite(validation_values):
                    raise ValueError("The fitted representation contains nonfinite coordinates.")
                record["training_representation_dtype"] = str(training_coordinates.dtype)
                record["validation_representation_dtype"] = str(validation_values.dtype)
                record["representation_shape"] = list(training_coordinates.shape)
                record["status"] = "nonconverged" if _nonconverged(fitted, caught, projection) else "valid"
            except _KNNTrainingStateCompatibilityError:
                # Diagnostics can read the same unusable fitted state. Keep the
                # original error on the failed attempt and its fold retryable.
                fitted = None
                raise
            except StatisticsRecoveryError:
                raise
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error),
                              failure_stage=locals().get("failure_stage", "configuration"),
                              preflight_rejected=not record["representation_fit_performed"])
    finally:
        record["warnings"] = _warning_records(caught)
        if fitted is not None:
            record["diagnostics"] = _diagnostics(fitted, projection)
    if record["status"] != "valid":
        # Each head inherits the shared outcome and keeps its own fit work.
        for result in record["neighbors"].values():
            result["status"] = record["status"]
            for field in ("error", "error_type", "failure_stage"):
                if field in record:
                    result[field] = record[field]
                else:
                    result.pop(field, None)
        return
    if kind != "pipeline":
        models, fit_warnings = _fit_neighbor_heads(
            knn_template, head_training_input, y[train], neighbors, len(train), record)
    start = time.perf_counter()
    # This loop owns all head calls and keeps one policy/threshold per fold.
    # Generic private-adapter callers retain the original full-query behavior.
    study_calls = None
    if (models and len(train) in models and type(scoring) is str
            and (kind != "nmf" or (type(fitted) is NMFClassifier
                 and "_labels_from_probabilities" not in fitted.__dict__
                 and getattr(fitted._labels_from_probabilities, "__func__", None)
                     is _STUDY_LABEL_CONVERSION))):
        study_threshold = fitted.decision_threshold_ if kind == "nmf" else None
        study_positive_index = fitted.positive_class_index_ if study_threshold is not None else None
        study_calls = tuple((value, study_threshold is not None, study_threshold, study_positive_index)
                            for value in models)
    verified_input = head_training_input
    native_input_validation = numeric_pipeline and (
        hasattr(knn_template, "feature_names_in_") or type(validation_coordinates) is not np.ndarray)
    if numeric_pipeline and not _unchanged_kd_coordinates(snapshot, head_training_input):
        # KDTree pruning bounds were built before validation. Brute heads read
        # the current coordinates, and BallTree already uses the selected width.
        verified_input = None
    del snapshot
    reuse = _KNNQueryReuse(models, validation_coordinates, _study_call_contract=study_calls,
                           _head_fit_input=verified_input, _query_values=validation_values,
                           _native_input_validation=native_input_validation)
    record["neighbor_reuse_setup_seconds"] += time.perf_counter() - start
    for neighbors_count, classifier in models.items():
        result = record["neighbors"][str(neighbors_count)]
        shared_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                threshold = fitted.decision_threshold_ if kind == "nmf" else None
                positive_index = fitted.positive_class_index_ if threshold is not None else None
                before_shared = reuse.query_seconds
                start = time.perf_counter()
                try:
                    predictions, metadata, shared_warnings = reuse.predict(
                        neighbors_count, probabilities=threshold is not None,
                        decision_threshold=threshold, positive_index=positive_index)
                    result["neighbor_query_reuse"] = metadata
                    if threshold is not None:
                        predictions = fitted._labels_from_probabilities(predictions)
                finally:
                    # Shared queries are charged once at fold level, not to the
                    # first k visited or repeatedly to each candidate.
                    result["prediction_seconds"] = max(
                        0.0, time.perf_counter() - start - (reuse.query_seconds - before_shared))
                score = _score_predictions(scoring, y[validation], predictions)
                all_warnings = fit_warnings[neighbors_count] + list(caught) + shared_warnings
                result.update(status="nonconverged" if _nonconverged(classifier, all_warnings) else "valid",
                              **_score_fields(scoring, score))
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, error=str(error))
        result["warnings"] = _warning_records(fit_warnings[neighbors_count] + list(caught) + shared_warnings)
    record["neighbor_query_seconds"] = reuse.query_seconds
    record["neighbor_queries"] = reuse.query_records()


def calibration_knn_grid(estimator, param_grid, X, y, *, cv=5, seed=0, threads=1,
                         checkpoint_dir, groups=None, folds=None, data_ids=None,
                         scoring="accuracy", execution_order="representation",
                         screening_statistics_recovery="off"):
    """Evaluate an explicit grid, sharing fold-local screening and representations.

    Defaults are five folds, seed zero and one native numerical thread.
    Fold/group loops are serial: ``threads`` limits supported native pools,
    not CV workers. Separate SPA projection controls can override that limit
    or launch processes, so it is not a global concurrency ceiling. ``seed``
    controls generated folds; fix representation random states separately.

    ``threads`` accepts positive Python/NumPy integer scalars within the native
    C-int range. ``seed`` requires a fixed Python/NumPy integer in 0..2**32-1,
    including with explicit folds. Boolean and temporal scalars are rejected
    for both controls, which are normalized before runtime or file activity.

    The return value is a JSON-compatible report containing ordinary complete
    candidate/fold records, provisional CV ranking and refinement diagnostics.
    Classifier settings other than n_neighbors must be fixed on the estimator;
    including them in the grid is rejected. Representation random_state settings
    must be fixed integer seeds. Only standard supported estimator types are
    accepted because representation independence from k is essential.
    Candidates with registered scikit-learn fit callbacks in their estimator
    configuration require independent fits with
    ``calibration_search(..., refit=False, checkpoint_path=None)``.
    Other classifiers use the general-purpose ``calibration_search`` instead.
    ``scoring`` is ``"accuracy"`` (the default) or ``"balanced_accuracy"``.
    Class proportions do not automatically change the requested metric.
    Candidate ranking uses the mean of complete fold scores, with ties retaining
    grid order. The metric is part of checkpoint identity; changing it requires
    a separate checkpoint directory. Scores reuse the existing predictions.
    ``groups`` keeps subjects together; ``folds`` supplies a validated complete
    CV partition. Optional unique ``data_ids`` bind checkpoints to sample IDs.
    Class labels must not end in a NUL codepoint; groups and data IDs retain
    their separate metadata validation rules.
    Standard NMFClassifier and screen/PCA/KNN layouts reuse screening statistics
    across cutoffs and ranks. Arbitrary pipelines keep their ordinary fit path
    when at least one requested neighbor count is feasible for the fold.
    Impossible ranks and empty screens are explicit failed candidates, never
    silently reduced ranks or changed cutoffs.

    If every requested neighbor count exceeds a fold's training size, record
    ``failure_stage="neighbor_feasibility"`` before screening, fitting or
    projection. Skipped work has zero recorded cost and no fitted diagnostics
    or warnings; neighbor infeasibility takes precedence over errors from those
    unexecuted operations. Equality with the training size remains allowed.
    Keep all candidate/fold records, including executed feasible folds when a
    candidate is impossible in another fold; every fold must be valid to qualify.

    Completed representation/fold receipts, including numerical failures, are
    reused only for the identical data, folds, grid, parameters, code and runtime.
    No estimator is serialized. An interrupted representation fold is retried;
    its earlier attempt remains recorded. Shared computation appears once in
    compute_seconds, never once per neighbor candidate. Candidate fit_seconds
    accounts for its KNN fit only; its shared_representation_key links the rest.
    Supported Euclidean heads also reuse neighbor queries across k. Uncertain
    batches and unsupported metrics/weights retain the public selected-k path.
    Shared query/setup costs are separate from each head's voting/fallback time.

    ``execution_order="representation"`` (default) finishes each representation's
    folds first. Opt-in ``"fold"`` processes eligible representations together
    within each fold, reducing repeated screening in some resumed multi-rank
    studies. This changes which candidates finish early and is not always faster.
    Only recognized screened layouts with ordinary parameters and builtin heads
    reorder; unsupported configurations and single-representation grids retain
    the default order. The report's ``execution_order`` records ``requested``,
    ``resolved`` and ``fallback_reason``. Final reports retain canonical grid/fold
    order. The requested choice is fixed in checkpoint identity even on fallback;
    changing it requires a new directory. Execution order alone does not persist
    fitted state or change the existing committed-work accounting.

    ``screening_statistics_recovery="off"`` (default) keeps screening statistics
    in memory for this call only. Opt-in ``"early"`` also commits successful
    fold-local screening statistics before selector/representation fitting and
    before the representation/fold receipt. A later call can restore those
    statistics after interruption; the unfinished representation fit is still
    retried. No estimator, representation, or neighbor model is serialized.
    This option uses the required ``checkpoint_dir`` and is part of checkpoint
    identity. Changing it requires a separate directory; existing checkpoints
    must also match the current source and numerical runtime.

    Persistence requires every grid configuration to have a recognized standard
    NMFClassifier or exact BinaryScreening[/PCA]/KNeighborsClassifier layout,
    ordinary parameter values, and supported builtin neighbor settings. It also
    requires NumPy's default error policy (divide/overflow/invalid warn,
    underflow ignore), no NumPy error callback, and sklearn assume_finite=False.
    Only successful statistics that emit no warning and have a supported array
    encoding are published. Other configurations retain ordinary computation;
    strict error policies and callbacks are not bypassed by restoring statistics.
    The additional ``screening_statistics_recovery`` report records eligibility,
    bypass reasons, publication failures, and current-call recovery activity.
    Eligibility alone does not imply that each fold produced a durable event.

    Each immutable event has an authoritative checksummed JSON record and a
    separate NPZ payload of learned arrays. Publishing the JSON record under the
    existing checkpoint lock commits the event. Temporary files and payloads
    without a JSON commit are not recovery authority. Missing or corrupt metadata
    referenced by an existing expected fold receipt aborts before new attempts
    or fold fitting. Corrupt committed metadata, identity mismatches, and
    source/runtime changes are errors, not permission to silently restart.
    When valid metadata survives but its learned payload is missing or unusable,
    recovery tries another valid event or recomputes statistics and publishes a
    new event. The original event and its historical cost are retained. Completed
    studies validate metadata and consumer references without opening unused
    learned payloads; this does not certify those payloads for a later restore.
    Atomic publication handles process interruption, not guaranteed power-loss
    durability. No checkpoint migration or automatic artifact deletion occurs.

    Original fold flags, compute_seconds and tuning_seconds retain committed
    receipt accounting. The recovery report separately exposes unique durable
    event cost, including events not yet consumed by a receipt, and union totals
    that count event-backed receipt work only once. Restored statistics are not
    reported as newly computed. Uncommitted attempts remain separate evidence.
    Early recovery adds metadata validation, serialization and I/O and can slow
    an uninterrupted study; it is intended to reduce repeated screening after
    interruption, not to guarantee ordinary-run speed. The default does not load
    the recovery codec or read event artifacts, but normal source-provenance
    checks still hash all package modules, including this implementation.
    """

    return _calibration_knn_grid(
        estimator, param_grid, X, y, cv=cv, seed=seed, threads=threads,
        checkpoint_dir=checkpoint_dir, groups=groups, folds=folds, data_ids=data_ids,
        scoring=scoring, execution_order=execution_order,
        _screening_statistics_recovery=screening_statistics_recovery)


def _calibrate_worker_cache(make_estimator, param_grid, feature_file, label_file, *, cv, seed,
                            threads, scoring, checkpoint_dir, reuse_only, _reuse_environment, _frozen_report=None):
    """Calibrate a study-owned file snapshot without a second copy of ordinary dense X."""
    mapped = np.load(feature_file, mmap_mode="r", allow_pickle=False)
    try:
        y = np.load(label_file, allow_pickle=False)
        estimator = make_estimator()
        started = time.perf_counter()
        snapshot = np.array(mapped, copy=True, order="K", subok=False)
    finally:
        del mapped
    snapshot.setflags(write=False)
    return _calibration_knn_grid(estimator, param_grid, snapshot, y, cv=cv, seed=seed,
                                 threads=threads, scoring=scoring, checkpoint_dir=checkpoint_dir,
                                 _reuse_only=reuse_only, _reuse_environment=_reuse_environment,
                                 _owned_snapshot=snapshot, _started_at=started, _frozen_report=_frozen_report)


def _calibration_knn_grid(estimator, param_grid, X, y, *, cv=5, seed=0, threads=1,
                         checkpoint_dir, groups=None, folds=None, data_ids=None,
                         scoring="accuracy", execution_order="representation", _reuse_only=False, _reuse_environment=None,
                         _screening_statistics_recovery="off", _owned_snapshot=None, _started_at=None, _frozen_report=None,
                         _feature_names=None):
    start_call = time.perf_counter() if _started_at is None else _started_at
    if _frozen_report is not None:
        _reuse_only = True
    if (type(_screening_statistics_recovery) not in (str, np.str_)
            or _screening_statistics_recovery not in {"off", "early"}):
        raise ValueError("screening_statistics_recovery must be 'off' or 'early'.")
    _screening_statistics_recovery = str(_screening_statistics_recovery)
    if _reuse_environment is not None:
        if not _reuse_only:
            raise ValueError("Read-only environment requires private completed reuse.")
        _reuse_environment(threads)
    if (type(execution_order) not in (str, np.str_)
            or execution_order not in {"representation", "fold"}):
        raise ValueError("execution_order must be 'representation' or 'fold'.")
    execution_order = str(execution_order)
    threads = _validate_threads(threads)
    seed = _validate_fixed_seed(seed)
    scoring = _validate_scoring(scoring)
    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral) or cv < 2:
        raise ValueError("cv must be an integer >= 2.")
    cv = int(cv)
    raw_y = y
    if _feature_names is None:
        _feature_names = _input_feature_names(X)
    X, y, original_y = _check_labeled_data(X, y)
    classes, counts = np.unique(y, return_counts=True)
    _validate_class_labels(raw_y, y, classes=classes)
    if len(classes) != 2 or counts.min() < cv:
        raise ValueError("Both binary classes must contain at least cv observations.")
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("The grid must declare at least one candidate.")
    # A custom clone hook could return a different, ordinary estimator and
    # erase the original implementation/state before identity is established.
    rng_limits = []
    implementation_sources(estimator, candidates, _split_limits=rng_limits)
    estimator = clone(estimator)
    kind, neighbor_parameter, forbidden = _layout(estimator)
    screening_diagnostics = screening_grid_diagnostics(estimator, candidates)
    rep_groups = {}
    opaque_declarations = {}
    candidate_representation_keys = []
    all_fold_order_eligible = execution_order == "fold"
    all_statistics_eligible = _screening_statistics_recovery == "early"
    unsafe_fit_rng = "unverified fit RNG" in rng_limits or "ambient fit RNG" in rng_limits
    captured_scalar_storage = "captured scalar storage" in rng_limits
    unknown_scalar_storage = "unverified scalar owner" in rng_limits
    watch_rng = "captured fit RNG" in rng_limits
    watch_captured_state = watch_rng or (captured_scalar_storage and not unknown_scalar_storage)
    for index, params in enumerate(candidates):
        if set(params) & forbidden:
            raise ValueError("Only n_neighbors may vary among classifier settings; fix other classifier knobs on the estimator.")
        configured = clone(estimator).set_params(**clone(params, safe=False))
        if _layout(configured)[:2] != (kind, neighbor_parameter):
            raise ValueError("Grid entries cannot change the classifier layout.")
        # At k=1 a callback can appear inert and still change a selected-k fit.
        # Check each effective configuration before reading or writing receipts.
        if _has_registered_fit_callbacks(configured):
            raise _KNNTrainingStateCompatibilityError(
                "Shared KNN calibration cannot reuse a configuration with registered scikit-learn fit callbacks. "
                "Use calibration_search(..., refit=False, checkpoint_path=None) "
                "to fit each neighbor count independently.")
        _fixed_randomness(configured)
        if not unsafe_fit_rng and "implicit RNG" in rng_limits:
            effective_limits = []
            configuration_identity(configured, _split_limits=effective_limits)
            unsafe_fit_rng = "implicit RNG" in effective_limits or "unverified fit RNG" in effective_limits
        if all_statistics_eligible:
            all_statistics_eligible = _fold_order_eligible(configured, kind)
        if all_fold_order_eligible:
            all_fold_order_eligible = _fold_order_eligible(configured, kind)
        neighbors_count = configured.get_params()[neighbor_parameter]
        if not _is_integer_index(neighbors_count) or neighbors_count < 1:
            raise ValueError("Every declared n_neighbors value must be a positive integer.")
        representation = {name: value for name, value in params.items() if name != neighbor_parameter}
        scalar_limits = []
        if captured_scalar_storage:
            # A grid callback may share a scalar owner with a callback fixed
            # on the template. Preserve that relationship when grouping fits.
            target = representation if _scalar_independent_template(estimator) else (estimator, representation)
            configuration = configuration_identity(target, _split_limits=scalar_limits)
        else:
            configuration = configuration_identity(representation)
        if "unverified dtype" in rng_limits or "unverified scalar owner" in scalar_limits:
            # ParameterGrid holds these objects alive across all neighbor
            # counts. Record the first candidate index for a declaration so
            # distinct opaque scalar implementations never share a fit.
            declaration = tuple((name, id(value)) for name, value in representation.items())
            first_index = opaque_declarations.setdefault(declaration, index)
            configuration = {"configuration": configuration, "declared_at": first_index}
        key = _digest(configuration)
        candidate_representation_keys.append(key)
        group = rep_groups.setdefault(key, {"parameters": representation, "candidate_indices": [], "neighbors": []})
        group["candidate_indices"].append(index)
        if int(neighbors_count) not in group["neighbors"]:
            group["neighbors"].append(int(neighbors_count))
    order = {"requested": execution_order, "resolved": "representation", "fallback_reason": None}
    if execution_order == "fold":
        if len(rep_groups) == 1:
            order["fallback_reason"] = "single_representation"
        elif not all_fold_order_eligible:
            order["fallback_reason"] = "unsupported_configuration"
        else:
            order["resolved"] = "fold"
    if X is not _owned_snapshot:
        X = np.array(X, copy=True)
    original_y = np.array(original_y, copy=True)
    X.setflags(write=False)
    original_y.setflags(write=False)
    y = _label_values(original_y)
    y.setflags(write=False)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    folds = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    runtime = _runtime_identity(estimator, threads, candidates, _environment=_reuse_environment)
    identity = _jsonable({"schema_version": 4, "method": "explicit_grid_shared_knn_representation",
                         "calibration": _search_identity(estimator, candidates, X, original_y, cv, seed, threads,
                                                         refit=False, groups=groups, folds=folds, data_ids=data_ids,
                                                         scoring=scoring, _environment=_reuse_environment,
                                                         _feature_names=_feature_names),
                         "scoring": scoring,
                         "execution_order": order,
                         "runtime": runtime, "folds": [{"training_indices": a, "validation_indices": b} for a, b in folds],
                         "neighbor_parameter": neighbor_parameter, "representation_backend": kind})
    if _screening_statistics_recovery == "early":
        identity["screening_statistics_recovery"] = {"requested": "early", "schema": 1}
    previous_configuration = identity["calibration"]["structural_configuration"]
    fingerprint = _digest(identity)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    screening_cache = _ScreeningCache()
    reused = new_fits = 0
    with _checkpoint_lock(directory):
        manifest = directory / "identity.json"
        if watch_captured_state:
            _check_captured_folds(directory, fingerprint)
        if manifest.exists():
            saved_identity = _read_receipt(manifest)
            if _reuse_only and isinstance(saved_identity, dict) and saved_identity.get("runtime") != runtime:
                # A caller may have loaded extra native libraries. The clean
                # worker remains authoritative; this is only a scheduling choice.
                return None
            if saved_identity != identity:
                raise ValueError("Study checkpoint is incompatible with these data, grid, folds, settings, source or runtime.")
            if unknown_scalar_storage:
                raise ValueError("Cannot verify the backing storage of a captured NumPy scalar in this saved study.")
            if unsafe_fit_rng:
                raise ValueError("Cannot verify implicit or unsupported fit randomness in this saved study. "
                                 "Use fixed estimator seeds for checkpoint recovery.")
        else:
            if _reuse_only:
                return None
            if unknown_scalar_storage and (any((directory / "folds").glob("*.json")) or (directory / "report.json").exists()):
                raise ValueError("A saved study with unverifiable scalar backing storage requires its original identity.")
            if unsafe_fit_rng and (any((directory / "folds").glob("*.json")) or (directory / "report.json").exists()):
                raise ValueError("A saved study with implicit or unsupported fit randomness requires its original identity.")
            if _screening_statistics_recovery == "early":
                from ._screening_recovery import StatisticsEventStore
                if StatisticsEventStore.committed_paths(directory / "statistics-events"):
                    raise StatisticsRecoveryError("Committed statistics events require the original study identity")
            _write_receipt(manifest, identity)
        if _screening_statistics_recovery == "early":
            from ._screening_recovery import StatisticsEventStore
            store = StatisticsEventStore(directory, fingerprint, runtime, estimator,
                                         candidates, threads, _runtime_identity,
                                         layout_eligible=all_statistics_eligible,
                                         fold_shapes=tuple((X.shape[1], len(train)) for train, _ in folds))
            screening_cache = _PersistentScreeningCache(store)
        try:
            if isinstance(screening_cache, _PersistentScreeningCache):
                _preflight_statistics_receipts(directory, fingerprint, rep_groups, cv, screening_cache.store)
            with (nullcontext() if _reuse_environment is not None else threadpool_limits(limits=threads)):
                active_pools = threadpool_info()
                if order["resolved"] == "fold":
                    schedule = ((key, group, ((fold, pair),))
                                for fold, pair in enumerate(folds) for key, group in rep_groups.items())
                else:
                    schedule = ((key, group, enumerate(folds)) for key, group in rep_groups.items())
                for representation_key, group, scheduled_folds in schedule:
                    for fold, (train, validation) in scheduled_folds:
                        key = _digest({"identity": fingerprint, "representation": representation_key, "fold": fold})
                        path = directory / "folds" / f"{key}.json"
                        if _runtime_identity(estimator, threads, candidates, _environment=_reuse_environment) != runtime:
                            if _reuse_only:
                                return None
                            raise ValueError("Source or numerical runtime changed during explicit-grid calibration.")
                        if path.exists():
                            record = _read_receipt(path)
                            if record.get("identity") != fingerprint or record.get("key") != key or record.get("state") != "completed":
                                raise ValueError("The representation checkpoint belongs to another run or is incomplete.")
                            if _reuse_environment is not None and not _quiet_record(record, len(validation)):
                                raise _ReuseUnavailable("nonfinite_or_unsupported_saved_score")
                            reused += 1
                        else:
                            if _reuse_only:
                                return None
                            record = {"identity": fingerprint, "key": key, "representation_key": representation_key,
                                      "parameters": group["parameters"], "fold": fold, "state": "running"}
                            attempt = directory / "attempts" / f"{uuid.uuid4().hex}.json"
                            _write_receipt(attempt, record)
                            try:
                                _run_representation_fold(estimator, kind, neighbor_parameter, group["parameters"],
                                                         group["neighbors"], X, y, train, validation, record,
                                                         screening_cache=screening_cache, scoring=scoring,
                                                         **({"_feature_names": _feature_names} if _feature_names is not None else {}))
                                snapshot = [] if watch_captured_state and not watch_rng else None
                                current_runtime = _runtime_identity(estimator, threads, candidates, _environment=_reuse_environment,
                                                                    _configuration=snapshot)
                                if current_runtime != runtime:
                                    raise ValueError("Source or numerical runtime changed during a representation fit.")
                                if watch_captured_state:
                                    current = (configuration_identity({"estimator": estimator, "candidates": candidates})
                                               if watch_rng else snapshot[0])
                                    if current != previous_configuration:
                                        record["captured_configuration_changed"] = True
                                    previous_configuration = current
                                record["state"] = "completed"
                                _write_receipt(path, record)
                                _write_receipt(attempt, record)
                            except BaseException as error:
                                record.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                                              attempt_error_type=type(error).__name__, attempt_error=str(error))
                                try:
                                    _write_receipt(attempt, record)
                                except Exception as logging_error:
                                    error.add_note(f"Saving the interrupted attempt also failed: {logging_error}")
                                raise
                            new_fits += int(record["representation_fit_performed"])
                        receipts.append(record)
            if order["resolved"] == "fold":
                representation_order = {key: index for index, key in enumerate(rep_groups)}
                receipts.sort(key=lambda row: (representation_order[row["representation_key"]], row["fold"]))
            by_representation_fold = {(row["representation_key"], row["fold"]): row for row in receipts}
            candidate_reports = []
            for index, (params, representation_key) in enumerate(zip(candidates, candidate_representation_keys)):
                configured = clone(estimator).set_params(**clone(params, safe=False))
                neighbors_count = int(configured.get_params()[neighbor_parameter])
                rows = []
                for fold in range(cv):
                    shared = by_representation_fold[representation_key, fold]
                    if shared["status"] == "valid":
                        row = deepcopy(shared["neighbors"][str(neighbors_count)])
                        row["warnings"] = shared["warnings"] + row["warnings"]
                    else:
                        head = shared["neighbors"].get(str(neighbors_count), {})
                        row = {"status": shared["status"], "fit_seconds": head.get("fit_seconds", 0.0),
                               "prediction_seconds": 0.0, "warnings": shared["warnings"]}
                        for field in ("error", "error_type", "failure_stage", "preflight_rejected"):
                            if field in shared:
                                row[field] = shared[field]
                    row.update(fold=fold, shared_representation_key=shared["key"],
                               diagnostics=shared.get("diagnostics", {}))
                    rows.append(row)
                valid = all(row["status"] == "valid" for row in rows)
                scores = [row["score"] for row in rows] if valid else []
                candidate_reports.append({"index": index, "parameters": params, "folds": rows, "valid": valid,
                                          **_score_fields(scoring, float(np.mean(scores)) if valid else None, mean=True),
                                          "fold_score_std": float(np.std(scores, ddof=1)) if valid else None,
                                          "invalid_fold_count": sum(row["status"] != "valid" for row in rows)})
            ranked = sorted((row["index"] for row in candidate_reports if row["valid"]),
                            key=lambda index: (-candidate_reports[index]["mean_score"], index))
            selected = candidates[ranked[0]] if ranked else None
            selected_mean = candidate_reports[ranked[0]]["mean_score"] if ranked else None
            # JSON sorts neighbor keys; sum the same sequence for live and reloaded folds.
            neighbor_timings = [row["neighbors"][key] for row in receipts for key in sorted(row["neighbors"])]
            costs = {"screening_statistics_seconds": sum(row["screening_statistics_seconds"] for row in receipts),
                     "screening_selection_seconds": sum(row["screening_selection_seconds"] for row in receipts),
                     "representation_fit_seconds": sum(row["representation_fit_seconds"] for row in receipts),
                     "validation_projection_seconds": sum(row["projection_seconds"] for row in receipts),
                     "knn_fit_seconds": sum(item["fit_seconds"] for item in neighbor_timings),
                     "knn_reuse_setup_seconds": sum(row["neighbor_reuse_setup_seconds"] for row in receipts),
                     "knn_shared_query_seconds": sum(row["neighbor_query_seconds"] for row in receipts),
                     "knn_prediction_seconds": sum(item["prediction_seconds"] for item in neighbor_timings)}
            report = {"schema_version": 4, "author": "Chang Cheng", "state": "cv_completed", "fingerprint": fingerprint,
                      "identity": identity, "active_threadpools": active_pools, "candidates": candidate_reports,
                      "scoring": scoring, "screening_diagnostics": screening_diagnostics,
                      "execution_order": order,
                      "ranked_candidate_indices": ranked, "selected_parameters": selected,
                      "selected_mean_score": selected_mean, f"selected_mean_{scoring}": selected_mean,
                      "selection_role": "provisional_highest_complete_training_cv_score_in_declared_grid",
                      "refinement": mark_inactive_refinement(
                          refinement_diagnostics(candidates, selected,
                                                 _rank_coordinates=_rank_coordinates(estimator, candidates, selected)),
                          screening_diagnostics,
                          ranked[0] if ranked else None),
                      "representation_folds": receipts, "unique_representations": len(rep_groups),
                      "screening_statistics_fits": sum(row["screening_statistics_fit_performed"] for row in receipts),
                      "preflight_rejected_representation_folds": sum(row["preflight_rejected"] for row in receipts),
                      "unique_representation_fits": sum(row["representation_fit_performed"] for row in receipts),
                      "new_representation_fits": new_fits, "reused_representation_folds": reused,
                      "compute_seconds": costs, "tuning_seconds": sum(costs.values()),
                      "current_call_wall_seconds": time.perf_counter() - start_call,
                      "timing_note": "Screen statistics, mask selection, representation fits, projection and shared neighbor-query/setup work are counted once when computed. Candidate fit times contain only their KNN fits; candidate prediction times contain voting/guard/public-fallback work excluding shared queries. Resume totals retain historical completed compute, not newly performed work. A partial resume may recompute in-memory screens. Recorded numerical compute excludes checkpoint I/O and uncommitted interrupted attempts, which remain in attempts/.",
                      "refit_performed": False, "final_fit_seconds": 0.0, "is_fallback": False,
                      "tuning_complete": False, "defaults_frozen": False}
            if isinstance(screening_cache, _PersistentScreeningCache):
                report["screening_statistics_recovery"] = screening_cache.store.summary(receipts, costs)
            report = _jsonable(report)
            if _reuse_environment is not None:
                _reuse_environment(threads)
            _check_transform_output(runtime["sklearn_transform_output"])
            report_path = directory / "report.json"
            if _frozen_report is not None:
                # The original worker's counters and observed native inventory
                # belong on disk. A replay can discover the same libraries in a different order.
                original = {**report, **_frozen_report["call"]}
                if _digest(original) != _frozen_report["sha256"]:
                    raise ValueError("Reaggregated CV folds differ from the frozen report.")
                if report_path.exists():
                    if _read_receipt(report_path) != original:
                        raise ValueError("Saved CV report differs from the frozen report.")
                else:
                    _write_receipt(report_path, original)
            else:
                _write_receipt(report_path, report)
            return report
        finally:
            if isinstance(screening_cache, _PersistentScreeningCache):
                screening_cache.close()
