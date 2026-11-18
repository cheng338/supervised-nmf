"""Execute explicit classification grids while sharing fold-local representations.

Author: Chang Cheng. This is a computation-saving implementation of an explicit
grid, not a parameter-proposal policy. Screening, factorization and projection
are fitted inside the same training folds for every declared neighbor count.
No full-training refit, held-out evaluation, or package-default decision occurs.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path
import time
import uuid
import warnings

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from ._completed_reuse import _ReuseUnavailable, _quiet_record
from .classifier import NMFClassifier, _capture_projection

_STUDY_LABEL_CONVERSION = NMFClassifier._labels_from_probabilities
from ._calibration import _validate_scoring, _score_predictions, _score_fields
from ._identity import configuration_identity, implementation_sources
from ._labels import _check_labeled_data, _sklearn_labels, _validate_class_labels
from ._neighbors import _KNNQueryReuse
from ._search_diagnostics import screening_grid_diagnostics, mark_inactive_refinement
from .screening import BinaryScreening
from .validation import _validate_fixed_seed, _validate_threads, build_cv_folds, validate_metadata
from .evaluation import (
    _diagnostics, _digest, _file_lock, _initialized_provenance, _jsonable, _nonconverged, _search_identity,
    _warning_records, provenance, refinement_diagnostics, write_json,
)


def _write_receipt(path, payload):
    write_json(path, {"payload": payload, "sha256": _digest(payload)})


def _read_receipt(path):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or "payload" not in envelope or envelope.get("sha256") != _digest(envelope["payload"]):
        raise ValueError(f"Study checkpoint failed its integrity check: {path}")
    return envelope["payload"]


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


def _runtime_identity(estimator, threads, candidates=(), *, _environment=None):
    info = _initialized_provenance(threads) if _environment is None else _environment(threads)
    runtime_keys = ("user_api", "internal_api", "prefix", "filepath", "version", "threading_layer", "architecture")
    libraries = sorted([{key: pool.get(key) for key in runtime_keys} for pool in info["threadpools"]],
                       key=lambda row: row.get("filepath") or "")
    sources = implementation_sources(estimator, candidates)
    return {"python": info["python"], "platform": info["platform"], "versions": info["versions"],
            "package_source": info["source_sha256"], "estimator_source": sources, "numerical_libraries": libraries}


def _fixed_randomness(estimator):
    # Sharing one randomized fit across k values requires a declared common seed.
    for name, value in estimator.get_params(deep=True).items():
        if name == "random_state" or name.endswith("__random_state"):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= 2 ** 32 - 1:
                raise ValueError(f"{name} must be a fixed integer seed for exact shared-representation comparisons.")


def _stored_training_representation(fitted, kind):
    """Isolate the version-checked sklearn private-state adapter.

    A pipeline's fit_transform result can differ numerically from transforming
    its training data again (notably PCA). KNN._fit_X is the exact representation
    that its original fit used. Never replace it with pipeline.transform(X).
    NMFClassifier explicitly exposes its original training_coordinates_.
    Equivalence tests cover CD/MU float32, SPA, PCA, screening-only and raw KNN.
    """
    if kind == "nmf":
        if type(fitted.classifier_) is Pipeline:
            steps = fitted.classifier_.steps
            if (len(steps) != 2 or type(steps[0][1]) is not StandardScaler
                    or type(steps[-1][1]) is not KNeighborsClassifier):
                raise RuntimeError("Unsupported fitted KNN coordinate pipeline.")
            knn = steps[-1][1]
            return knn._fit_X, knn
        return fitted.training_coordinates_, fitted.classifier_
    knn = fitted.steps[-1][1] if kind == "pipeline" else fitted
    if not hasattr(knn, "_fit_X"):
        raise RuntimeError("This sklearn version does not expose the tested KNN training-representation adapter.")
    return knn._fit_X, knn


def _validation_representation(fitted, kind, X):
    if kind == "nmf":
        coordinates = fitted.transform(X)
        if type(fitted.classifier_) is Pipeline:
            return fitted.classifier_[:-1].transform(coordinates)
        return coordinates
    if kind == "pipeline" and len(fitted.steps) > 1:
        return fitted[:-1].transform(X)
    return X


def _finite(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(np.isfinite(values).all())


class _PreparedScreening(BinaryScreening):
    """Carry a fold-local screen through a standard uncached Pipeline fit."""

    def __init__(self, reference):
        self.reference = reference
        super().__init__(**reference.get_params(deep=False))

    def fit(self, X, y):
        return self._fit_from_statistics(self.reference)


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
        if fold not in self.statistics:
            restored = self.store.load(fold, record, X.shape[1], len(train))
            if restored is not None:
                self.statistics[fold], self.events[fold] = restored
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
                    if event is not None:
                        self.events[fold] = event
        base = self.statistics[fold]
        if isinstance(base, Exception):
            raise base
        event = self.events.get(fold)
        if event is not None:
            record['screening_statistics_event_id'] = event['event_id']
            record['screening_statistics_event_reused'] = not record['screening_statistics_fit_performed']
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


def _run_representation_fold(estimator, kind, neighbor_parameter, representation_parameters,
                             neighbors, X, y, train, validation, record, *, screening_cache=None,
                             scoring="accuracy"):
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
                if selected is not None and kind == "pipeline":
                    name, _ = fitted.steps[0]
                    fitted.steps[0] = (name, _PreparedScreening(selected))
                failure_stage = "fit"
                record["representation_fit_performed"] = True
                start = time.perf_counter()
                try:
                    if selected is not None and kind == "nmf":
                        fitted._fit_with_screening(X[train], y[train], selected)
                    else:
                        fitted.fit(X[train], y[train])
                finally:
                    record["representation_fit_seconds"] = time.perf_counter() - start
                training_coordinates, knn_template = _stored_training_representation(fitted, kind)
                failure_stage = "projection"
                start = time.perf_counter()
                try:
                    with _capture_projection(fitted) as projection:
                        validation_coordinates = _validation_representation(fitted, kind, X[validation])
                finally:
                    record["projection_seconds"] = time.perf_counter() - start
                if not _finite(training_coordinates) or not _finite(validation_coordinates):
                    raise ValueError("The fitted representation contains nonfinite coordinates.")
                record["training_representation_dtype"] = str(training_coordinates.dtype)
                record["validation_representation_dtype"] = str(validation_coordinates.dtype)
                record["representation_shape"] = list(training_coordinates.shape)
                record["status"] = "nonconverged" if _nonconverged(fitted, caught, projection) else "valid"
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
        return
    models, fit_warnings = {}, {}
    for neighbors_count in neighbors:
        result = {"n_neighbors": neighbors_count, "fit_seconds": 0.0, "prediction_seconds": 0.0}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                classifier = clone(knn_template).set_params(n_neighbors=neighbors_count)
                if neighbors_count > len(train):
                    raise ValueError("n_neighbors exceeds the number of observations in this training fold.")
                start = time.perf_counter()
                try:
                    classifier.fit(training_coordinates, y[train])
                finally:
                    result["fit_seconds"] = time.perf_counter() - start
                models[neighbors_count] = classifier
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, error=str(error))
        fit_warnings[neighbors_count] = list(caught)
        result["warnings"] = _warning_records(caught)
        record["neighbors"][str(neighbors_count)] = result
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
    reuse = _KNNQueryReuse(models, validation_coordinates, _study_call_contract=study_calls)
    record["neighbor_reuse_setup_seconds"] = time.perf_counter() - start
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


def _reuse_calibration_knn_grid(estimator, param_grid, X, y, *, cv=5, seed=0, threads=1,
                         checkpoint_dir, groups=None, folds=None, data_ids=None,
                         scoring="accuracy", execution_order="representation", _reuse_environment=None,
                         _screening_statistics_recovery="off"):
    """Validate/rebuild completed CV; return None before any missing work is fitted."""
    return _calibration_knn_grid(
        estimator, param_grid, X, y, cv=cv, seed=seed, threads=threads,
        checkpoint_dir=checkpoint_dir, groups=groups, folds=folds, data_ids=data_ids,
        scoring=scoring, execution_order=execution_order, _reuse_only=True,
        _reuse_environment=_reuse_environment,
        _screening_statistics_recovery=_screening_statistics_recovery)


def _calibration_knn_grid(estimator, param_grid, X, y, *, cv=5, seed=0, threads=1,
                         checkpoint_dir, groups=None, folds=None, data_ids=None,
                         scoring="accuracy", execution_order="representation", _reuse_only=False, _reuse_environment=None,
                         _screening_statistics_recovery="off"):
    start_call = time.perf_counter()
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
    X, y = _check_labeled_data(X, y)
    classes, counts = np.unique(y, return_counts=True)
    _validate_class_labels(raw_y, y, classes=classes)
    if len(classes) != 2 or counts.min() < cv:
        raise ValueError("Both binary classes must contain at least cv observations.")
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("The grid must declare at least one candidate.")
    # A custom clone hook could return a different, ordinary estimator and
    # erase the original implementation/state before identity is established.
    implementation_sources(estimator, candidates)
    estimator = clone(estimator)
    kind, neighbor_parameter, forbidden = _layout(estimator)
    screening_diagnostics = screening_grid_diagnostics(estimator, candidates)
    rep_groups = {}
    all_fold_order_eligible = execution_order == "fold"
    all_statistics_eligible = _screening_statistics_recovery == "early"
    for index, params in enumerate(candidates):
        if set(params) & forbidden:
            raise ValueError("Only n_neighbors may vary among classifier settings; fix other classifier knobs on the estimator.")
        configured = clone(estimator).set_params(**clone(params, safe=False))
        if _layout(configured)[:2] != (kind, neighbor_parameter):
            raise ValueError("Grid entries cannot change the classifier layout.")
        _fixed_randomness(configured)
        if all_statistics_eligible:
            all_statistics_eligible = _fold_order_eligible(configured, kind)
        if all_fold_order_eligible:
            all_fold_order_eligible = _fold_order_eligible(configured, kind)
        neighbors_count = configured.get_params()[neighbor_parameter]
        if isinstance(neighbors_count, (bool, np.bool_)) or not isinstance(neighbors_count, Integral) or neighbors_count < 1:
            raise ValueError("Every declared n_neighbors value must be a positive integer.")
        representation = {name: value for name, value in params.items() if name != neighbor_parameter}
        key = _digest(configuration_identity(representation))
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
    X, y = np.array(X, copy=True), np.array(y, copy=True)
    X.setflags(write=False)
    y.setflags(write=False)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    folds = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    runtime = _runtime_identity(estimator, threads, candidates, _environment=_reuse_environment)
    identity = _jsonable({"schema_version": 4, "method": "explicit_grid_shared_knn_representation",
                         "calibration": _search_identity(estimator, candidates, X, y, cv, seed, threads,
                                                         refit=False, groups=groups, folds=folds, data_ids=data_ids,
                                                         scoring=scoring, _environment=_reuse_environment),
                         "scoring": scoring,
                         "execution_order": order,
                         "runtime": runtime, "folds": [{"training_indices": a, "validation_indices": b} for a, b in folds],
                         "neighbor_parameter": neighbor_parameter, "representation_backend": kind})
    if _screening_statistics_recovery == "early":
        identity["screening_statistics_recovery"] = {"requested": "early", "schema": 1}
    fingerprint = _digest(identity)
    y = _sklearn_labels(y)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    screening_cache = _ScreeningCache()
    reused = new_fits = 0
    with _checkpoint_lock(directory):
        manifest = directory / "identity.json"
        if manifest.exists():
            saved_identity = _read_receipt(manifest)
            if _reuse_only and isinstance(saved_identity, dict) and saved_identity.get("runtime") != runtime:
                # A caller may have loaded extra native libraries. The clean
                # worker remains authoritative; this is only a scheduling choice.
                return None
            if saved_identity != identity:
                raise ValueError("Study checkpoint is incompatible with these data, grid, folds, settings, source or runtime.")
        else:
            if _reuse_only:
                return None
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
                                                         screening_cache=screening_cache, scoring=scoring)
                                if _runtime_identity(estimator, threads, candidates, _environment=_reuse_environment) != runtime:
                                    raise ValueError("Source or numerical runtime changed during a representation fit.")
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
            for index, params in enumerate(candidates):
                representation = {name: value for name, value in params.items() if name != neighbor_parameter}
                representation_key = _digest(configuration_identity(representation))
                configured = clone(estimator).set_params(**clone(params, safe=False))
                neighbors_count = int(configured.get_params()[neighbor_parameter])
                rows = []
                for fold in range(cv):
                    shared = by_representation_fold[representation_key, fold]
                    if shared["status"] == "valid":
                        row = deepcopy(shared["neighbors"][str(neighbors_count)])
                        row["warnings"] = shared["warnings"] + row["warnings"]
                    else:
                        row = {"status": shared["status"], "fit_seconds": 0.0, "prediction_seconds": 0.0,
                               "warnings": shared["warnings"]}
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
            costs = {"screening_statistics_seconds": sum(row["screening_statistics_seconds"] for row in receipts),
                     "screening_selection_seconds": sum(row["screening_selection_seconds"] for row in receipts),
                     "representation_fit_seconds": sum(row["representation_fit_seconds"] for row in receipts),
                     "validation_projection_seconds": sum(row["projection_seconds"] for row in receipts),
                     "knn_fit_seconds": sum(item["fit_seconds"] for row in receipts for item in row["neighbors"].values()),
                     "knn_reuse_setup_seconds": sum(row["neighbor_reuse_setup_seconds"] for row in receipts),
                     "knn_shared_query_seconds": sum(row["neighbor_query_seconds"] for row in receipts),
                     "knn_prediction_seconds": sum(item["prediction_seconds"] for row in receipts for item in row["neighbors"].values())}
            report = {"schema_version": 4, "author": "Chang Cheng", "state": "cv_completed", "fingerprint": fingerprint,
                      "identity": identity, "active_threadpools": active_pools, "candidates": candidate_reports,
                      "scoring": scoring, "screening_diagnostics": screening_diagnostics,
                      "execution_order": order,
                      "ranked_candidate_indices": ranked, "selected_parameters": selected,
                      "selected_mean_score": selected_mean, f"selected_mean_{scoring}": selected_mean,
                      "selection_role": "provisional_highest_complete_training_cv_score_in_declared_grid",
                      "refinement": mark_inactive_refinement(
                          refinement_diagnostics(candidates, selected), screening_diagnostics,
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
            _write_receipt(directory / "report.json", report)
            return report
        finally:
            if isinstance(screening_cache, _PersistentScreeningCache):
                screening_cache.close()
