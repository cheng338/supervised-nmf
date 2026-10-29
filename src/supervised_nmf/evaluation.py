"""Auditable classification comparisons derived from Chang Cheng's scripts.

Model selection uses only training folds. A candidate with even one failed or
nonconverged fold has no aggregate score and cannot win by averaging successes.
The reports deliberately leave refinement and package-default decisions open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time
import uuid
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_X_y
from threadpoolctl import threadpool_info, threadpool_limits

from .classifier import _capture_projection, _record_warnings, _uses_projection_diagnostics
from .screening import _clear_fitted_attributes
from .validation import build_cv_folds, validate_metadata


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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return repr(value)


def write_json(path, report):
    """Atomically replace an explicitly supplied report, preserving valid JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    # Windows indexers/antivirus can briefly hold an existing checkpoint open.
    # Keep atomic replacement; never delete the old valid checkpoint first.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def save_data_manifest(path, manifest):
    """Create a manifest or verify it unchanged before resuming historical work."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != _jsonable(manifest):
            raise ValueError("Existing data manifest is incompatible; create a new run directory.")
    else:
        write_json(path, manifest)


def data_fingerprint(X, y):
    """Hash data values, ordering, shape, and dtype without publishing raw data."""
    digest = hashlib.sha256()
    for array in (np.asarray(X), np.asarray(y)):
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        if array.dtype.hasobject:
            digest.update(json.dumps(_jsonable(array), sort_keys=True).encode())
        else:
            contiguous = np.ascontiguousarray(array)
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def provenance(threads=1):
    """Record software, hardware, source, and requested native thread limits."""
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "joblib", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "versions": versions, "source_sha256": source_hashes,
            "native_threads_requested": threads, "threadpools": threadpool_info()}


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


def _search_identity(estimator, candidates, X, y, cv, seed, threads, refit=True,
                     *, groups=None, folds=None, data_ids=None):
    environment = provenance(threads)
    runtime_keys = ("user_api", "internal_api", "prefix", "filepath", "version", "threading_layer", "architecture")
    numerical_runtime = sorted([{key: library.get(key) for key in runtime_keys}
                                for library in environment["threadpools"]],
                               key=lambda library: library.get("filepath") or "")
    return {"estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            "parameters": _jsonable(estimator.get_params(deep=True)),
            "candidates": _jsonable(candidates), "data_sha256": data_fingerprint(X, y),
            "cv": cv, "seed": seed, "threads": threads, "refit": refit,
            "groups": _jsonable(groups), "folds": _jsonable(folds), "data_ids": _jsonable(data_ids),
            "python": environment["python"], "platform": environment["platform"],
            "numerical_runtime": numerical_runtime,
            "software": environment["versions"],
            "source_sha256": environment["source_sha256"]}


def refinement_diagnostics(candidates, selected_params):
    """Flag range boundaries and interior intervals; this is not completed tuning."""
    if selected_params is None:
        return {"status": "pending_valid_candidate", "parameters": {}}
    output = {}
    for parameter, selected in selected_params.items():
        values = [c[parameter] for c in candidates if parameter in c]
        if isinstance(selected, (int, float, np.number)) and not isinstance(selected, bool):
            values = sorted(set(float(v) for v in values if isinstance(v, (int, float, np.number))))
            if len(values) < 2:
                output[parameter] = {"status": "single_value_requires_exploration", "selected": selected}
            else:
                location = values.index(float(selected))
                status = ("lower_boundary_requires_expansion_or_constraint" if location == 0
                          else "upper_boundary_requires_expansion_or_constraint" if location == len(values) - 1
                          else "interior_interval_requires_refinement")
                output[parameter] = {"status": status, "selected": selected, "values": values}
        else:
            output[parameter] = {"status": "categorical", "selected": selected}
    return {"status": "pending_refinement_and_replication", "parameters": output}


@dataclass
class CalibrationResult:
    estimator: object
    report: dict


def calibration_search(estimator, param_grid, X, y, *, cv=5, seed=0,
                       threads=1, checkpoint_path=None, refit=True,
                       groups=None, folds=None, data_ids=None):
    """Calibrate on training data with complete, resumable per-fold records.

    Both fit and held-out-fold prediction warnings are retained. Convergence
    warnings invalidate that candidate. If a numerically valid CV winner fails
    full-training refit, try the next valid candidate and retain every attempt.
    If none can be fitted, a training-only majority-class fallback is returned
    with ``is_fallback=True``. It is never presented as the requested method.
    With ``refit=False``, only CV is performed and the returned estimator is
    None. This supports staged tuning without intermediate full-data fits.
    Supply one subject ID per row in ``groups`` for group-disjoint CV, or pass
    exhaustive custom ``folds`` accepted by :func:`build_cv_folds`. Optional
    unique ``data_ids``, groups and the exact folds are bound to checkpoints.
    """
    X, y = check_X_y(X, y)
    classes, counts = np.unique(y, return_counts=True)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    folds = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    cv = len(folds)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    if not isinstance(refit, bool):
        raise ValueError("refit must be boolean.")
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("At least one parameter candidate is required.")
    identity = _search_identity(estimator, candidates, X, y, cv, seed, threads, refit=refit,
                                groups=groups, folds=folds, data_ids=data_ids)
    fingerprint = _digest(identity)
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    if path is not None and path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("fingerprint") != fingerprint:
            raise ValueError("Checkpoint is incompatible with these data, settings, software, or source.")
        saved_digest = report.get("checkpoint_sha256")
        if saved_digest != _digest({k: v for k, v in report.items() if k != "checkpoint_sha256"}):
            raise ValueError("Checkpoint content failed its integrity check.")
    else:
        report = {"schema_version": 1, "author": "Chang Cheng", "fingerprint": fingerprint,
                  "identity": identity, "provenance": provenance(threads),
                  "class_labels": _jsonable(classes), "class_counts": counts.tolist(),
                  "folds": [{"training_indices": a.tolist(), "validation_indices": b.tolist()}
                            for a, b in folds],
                  "candidates": [{"index": i, "parameters": _jsonable(p), "folds": []}
                                 for i, p in enumerate(candidates)],
                  "tuning_seconds": 0.0, "final_fit_seconds": 0.0, "final_fit_attempts": []}
    previous_seconds = report["tuning_seconds"]
    start = time.perf_counter()

    def checkpoint():
        if path is not None:
            report["checkpoint_sha256"] = _digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
            write_json(path, report)

    report["state"] = "calibrating"
    checkpoint()
    with threadpool_limits(limits=threads):
        report["active_threadpools"] = threadpool_info()
        for params, candidate in zip(candidates, report["candidates"]):
            completed = {f["fold"] for f in candidate["folds"]}
            for fold, (training, validation) in enumerate(folds):
                if fold in completed:
                    continue
                record = {"fold": fold, "fit_seconds": 0.0, "prediction_seconds": 0.0}
                fitted = None
                projection = None
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        fitted = clone(estimator).set_params(**params)
                        fit_start = time.perf_counter()
                        try:
                            fitted.fit(X[training], y[training])
                        finally:
                            record["fit_seconds"] = time.perf_counter() - fit_start
                        inference_start = time.perf_counter()
                        try:
                            with _capture_projection(fitted) as projection:
                                predictions = fitted.predict(X[validation])
                        finally:
                            record["prediction_seconds"] = time.perf_counter() - inference_start
                        score = float(balanced_accuracy_score(y[validation], predictions))
                        if not math.isfinite(score):
                            raise ValueError("Nonfinite calibration score.")
                        record["balanced_accuracy"] = score
                        record["status"] = "nonconverged" if _nonconverged(fitted, caught, projection) else "valid"
                    except Exception as error:
                        record.update(status="failed", error_type=type(error).__name__, error=str(error))
                record["warnings"] = _warning_records(caught)
                if fitted is not None:
                    record["diagnostics"] = _diagnostics(fitted, projection)
                candidate["folds"].append(record)
                report["tuning_seconds"] = previous_seconds + time.perf_counter() - start
                checkpoint()
            valid = len(candidate["folds"]) == cv and all(f["status"] == "valid" for f in candidate["folds"])
            scores = [f["balanced_accuracy"] for f in candidate["folds"]] if valid else []
            candidate["valid"] = valid
            candidate["mean_balanced_accuracy"] = float(np.mean(scores)) if valid else None
            candidate["fold_score_std"] = float(np.std(scores, ddof=1)) if valid else None
            candidate["invalid_fold_count"] = sum(f["status"] != "valid" for f in candidate["folds"])
            checkpoint()
        report["tuning_seconds"] = previous_seconds + time.perf_counter() - start
        ranked = sorted((c for c in report["candidates"] if c["valid"]),
                        key=lambda c: (-c["mean_balanced_accuracy"], c["index"]))
        if not refit:
            selected = ranked[0] if ranked else None
            report.update(state="cv_completed", is_fallback=False, refit_performed=False,
                          selected_candidate_index=None if selected is None else selected["index"],
                          selected_parameters=None if selected is None else selected["parameters"],
                          tuning_complete=False, defaults_frozen=False,
                          refinement=refinement_diagnostics(candidates, None if selected is None else selected["parameters"]))
            checkpoint()
            return CalibrationResult(None, report)
        selected = None
        fitted = None
        report["state"] = "refitting"
        checkpoint()
        for candidate in ranked:
            attempt = {"candidate_index": candidate["index"], "parameters": candidate["parameters"]}
            attempt_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    model = clone(estimator).set_params(**candidates[candidate["index"]]).fit(X, y)
                    attempt["diagnostics"] = _diagnostics(model)
                    attempt["status"] = "nonconverged" if _nonconverged(model, caught) else "valid"
                    if attempt["status"] == "valid":
                        fitted, selected = model, candidate
                except Exception as error:
                    attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
            attempt["warnings"] = _warning_records(caught)
            attempt["seconds"] = time.perf_counter() - attempt_start
            report["final_fit_seconds"] += attempt["seconds"]
            report["final_fit_attempts"].append(attempt)
            checkpoint()
            if fitted is not None:
                break
        if fitted is None:
            fallback_start = time.perf_counter()
            fitted = DummyClassifier(strategy="most_frequent").fit(X, y)
            seconds = time.perf_counter() - fallback_start
            report["final_fit_seconds"] += seconds
            report["final_fit_attempts"].append({"status": "fallback", "seconds": seconds,
                                                "reason": "No candidate passed all folds and final refit."})
    report.update(state="completed", is_fallback=selected is None,
                  selected_candidate_index=None if selected is None else selected["index"],
                  selected_parameters=None if selected is None else selected["parameters"],
                  tuning_complete=False, defaults_frozen=False,
                  refinement=refinement_diagnostics(candidates, None if selected is None else selected["parameters"]))
    checkpoint()
    return CalibrationResult(fitted, report)


def _evaluation_metadata(y, classes, *, sample_ids, groups, train_sample_ids,
                         train_groups, train_y, min_class_support, min_groups):
    """Validate supplied split metadata before any held-out prediction."""
    for name, value in (("min_class_support", min_class_support), ("min_groups", min_groups)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    sample_ids = validate_metadata(sample_ids, len(y), name="sample_ids", unique=True)
    groups = validate_metadata(groups, len(y), name="groups")
    train_count = None
    if train_y is not None:
        train_y = np.asarray(train_y)
        if (train_y.ndim != 1 or not len(train_y)
                or not np.all(np.isin(train_y, classes)) or len(np.unique(train_y)) != 2):
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
    train_sample_ids = validate_metadata(train_sample_ids, train_count, name="train_sample_ids", unique=True)
    train_groups = validate_metadata(train_groups, train_count, name="train_groups")
    if train_sample_ids is not None:
        if sample_ids is None:
            raise ValueError("sample_ids are required when train_sample_ids are supplied.")
        if set(train_sample_ids.tolist()) & set(sample_ids.tolist()):
            raise ValueError("Training and evaluation sample IDs must not overlap.")
    if train_groups is not None:
        if groups is None:
            raise ValueError("groups are required when train_groups are supplied.")
        if set(train_groups.tolist()) & set(groups.tolist()):
            raise ValueError("Training and evaluation groups must not overlap.")
    counts = [int(np.sum(y == label)) for label in classes]
    group_count = None if groups is None else len(set(groups.tolist()))
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
        "training_group_count": None if train_groups is None else len(set(train_groups.tolist())),
    }
    if groups is not None:
        metadata["per_class_group_count"] = [len(set(groups[y == label].tolist())) for label in classes]
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


def evaluate_classifier(estimator, X, y, *, threads=1, sample_ids=None,
                        groups=None, train_sample_ids=None, train_groups=None,
                        train_y=None, min_class_support=10, min_groups=5):
    """Evaluate a fitted classifier without fitting or threshold selection.

    Optional ``train_y`` defines a training-only majority reference. Subject IDs
    in ``groups`` provide subject counts; with ``train_groups`` they also enforce
    disjoint training/evaluation subjects. Likewise, ``train_sample_ids`` require
    explicit evaluation ``sample_ids`` and reject shared rows. Metadata is
    checked before prediction. Support thresholds add descriptive flags only;
    they do not alter predictions, scores, numerical status or model selection.
    Ranking metrics use probabilities when available, otherwise genuine binary
    decision margins oriented toward ``classes_[1]``. Margins are recorded
    separately and are never converted into purported probabilities.
    """
    X, y = check_X_y(X, y)
    classes = np.asarray(estimator.classes_)
    if len(classes) != 2 or not np.all(np.isin(y, classes)):
        raise ValueError("Evaluation labels must belong to the fitted binary classes.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    sample_ids, groups, support_diagnostics, majority_reference = _evaluation_metadata(
        y, classes, sample_ids=sample_ids, groups=groups, train_sample_ids=train_sample_ids,
        train_groups=train_groups, train_y=train_y, min_class_support=min_class_support,
        min_groups=min_groups)
    positive = classes[1]
    start = time.perf_counter()
    with (threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught,
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
            elif isinstance(estimator, Pipeline) and isinstance(estimator.steps[-1][1], KNeighborsClassifier):
                projected = estimator[:-1].transform(X) if len(estimator.steps) > 1 else X
                knn = estimator.steps[-1][1]
                all_probabilities = knn.predict_proba(projected)
                predicted = knn.classes_[np.argmax(all_probabilities, axis=1)]
                probability = all_probabilities[:, 1]
                prediction_operation = "joint_label_and_probability_prediction_one_projection"
            elif isinstance(estimator, KNeighborsClassifier):
                all_probabilities = estimator.predict_proba(X)
                predicted = estimator.classes_[np.argmax(all_probabilities, axis=1)]
                probability = all_probabilities[:, 1]
                prediction_operation = "joint_label_and_probability_prediction_no_projection"
            elif isinstance(estimator, Pipeline) and isinstance(estimator.steps[-1][1], LogisticRegression):
                projected = estimator[:-1].transform(X) if len(estimator.steps) > 1 else X
                logistic = estimator.steps[-1][1]
                # Keep sklearn's decision-score tie rule. Probability rounding
                # can produce a 0.5 tie for a small nonzero decision score.
                predicted = logistic.predict(projected)
                probability = logistic.predict_proba(projected)[:, 1]
                prediction_operation = "joint_label_and_probability_prediction_one_projection"
            elif isinstance(estimator, Pipeline) and isinstance(estimator.steps[-1][1], LinearSVC):
                projected = estimator[:-1].transform(X) if len(estimator.steps) > 1 else X
                svm = estimator.steps[-1][1]
                margin = svm.decision_function(projected)
                predicted = svm.classes_[(margin > 0).astype(int)]
                prediction_operation = "joint_label_and_margin_prediction_one_projection"
            else:
                predicted = estimator.predict(X)
                if hasattr(estimator, "predict_proba"):
                    probability = estimator.predict_proba(X)[:, 1]
                    prediction_operation = "predict_plus_predict_proba_calls"
                elif hasattr(estimator, "decision_function"):
                    margin = estimator.decision_function(X)
                    prediction_operation = "predict_plus_decision_function_calls"
                else:
                    prediction_operation = "predict_only"
            if probability is not None and (not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1))):
                raise ValueError("Nonfinite or invalid predicted probabilities.")
            if margin is not None:
                margin = np.asarray(margin)
                if margin.shape != (len(y),) or not np.all(np.isfinite(margin)):
                    raise ValueError("Decision margins must be a finite one-dimensional binary score.")
        except Exception as error:
            return {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                    "prediction_seconds": time.perf_counter() - start,
                    "warnings": _warning_records(caught), "diagnostics": _diagnostics(estimator, projection),
                    "support_diagnostics": support_diagnostics, "training_majority_reference": majority_reference,
                    "data_sha256": data_fingerprint(X, y)}
    elapsed = time.perf_counter() - start
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
    return {"status": "provisional_nonconvergence" if _nonconverged(estimator, caught, projection) else "valid",
            "metrics": metrics, "prediction_seconds": elapsed, "prediction_operation": prediction_operation,
            "ranking_score_type": score_type,
            "warnings": _warning_records(caught),
            "diagnostics": _diagnostics(estimator, projection), "data_sha256": data_fingerprint(X, y),
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
    for label, value in (("Tuning", report.get("tuning_seconds")),
                         ("Final fitting (recorded attempts)", report.get("final_fit_seconds")),
                         ("Prediction", evaluation.get("prediction_seconds"))):
        lines.append(f"| {label} | {number(value)} |")
    fit_times = diagnostics.get("fit_times", {})
    stages = ("screening", "factorization", "training_projection", "classifier", "total")
    for label in stages:
        lines.append(f"| Fit: {label.replace('_', ' ')} | {number(fit_times.get(label))} |")
    for label, value in fit_times.items():
        if label in stages:
            continue
        lines.append(f"| Fit: {label.replace('_', ' ')} | {number(value)} |")
    lines += ["", "Final fitting can include unsuccessful refit attempts. Fit rows describe the "
              "fitted estimator; its total includes its component stages. Prediction is the measured "
              "evaluation operation. Do not add these overlapping totals or infer total time by "
              "summing stage medians or concurrent worker times. Individual-operation latency "
              "and batch throughput are separate measurements."]
    return "\n".join(lines) + "\n"


def comparison_models(*, random_state=0, quick=False, max_iter=20000, tol=1e-4,
                      classifier="knn", logistic_max_iter=1000,
                      logistic_tol=1e-4, logistic_class_weight=None,
                      svm_max_iter=20000, svm_tol=1e-5, svm_class_weight=None):
    """Create independent CD/SPA/MU, PCA, screening-only and original baseline grids.

    Grids are deliberately modest starting grids. Caller-supplied JSON can expand
    them, and diagnostics identify unfinished boundaries/interior refinement.
    Absolute screening uses |t|, not the F statistic used for percentile ranking.
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


def run_comparison(models, X_train, y_train, X_validation, y_validation, *,
                   output_dir, seed=0, threads=1, validation_ids=None, signal_support=None):
    """Save each model independently; resume only exactly compatible results.

    Validation is development data, never evidence that defaults are finalized.
    No test-set loading, default selection, or claimed reproduction occurs here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (estimator, grid) in models.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Model names must contain only letters, digits, underscore or hyphen.")
        identity = _search_identity(estimator, list(ParameterGrid(grid)), X_train, y_train, 5, seed, threads)
        comparison_fingerprint = _digest({"calibration": identity,
                                          "validation": data_fingerprint(X_validation, y_validation),
                                          "validation_ids": validation_ids, "signal_support": signal_support})
        result_path = output_dir / f"{name}-evaluation.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("comparison_fingerprint") != comparison_fingerprint:
                raise ValueError(f"Existing {name} results are incompatible; create a new run directory.")
            results[name] = existing
            continue
        calibrated = calibration_search(estimator, grid, X_train, y_train, cv=5, seed=seed,
                                        threads=threads, checkpoint_path=output_dir / f"{name}-cv.json")
        record = {"model": name, "comparison_fingerprint": comparison_fingerprint,
                  "is_fallback": calibrated.report["is_fallback"],
                  "selected_parameters": calibrated.report["selected_parameters"],
                  "tuning_seconds": calibrated.report["tuning_seconds"],
                  "final_fit_seconds": calibrated.report["final_fit_seconds"],
                  "tuning_complete": False, "defaults_frozen": False,
                  "evaluation_role": "development_validation",
                  "refinement": calibrated.report["refinement"]}
        if signal_support is not None and not record["is_fallback"]:
            record["support_recovery"] = support_recovery(calibrated.estimator, signal_support)
        try:
            record["evaluation"] = evaluate_classifier(calibrated.estimator, X_validation, y_validation,
                                                        threads=threads, sample_ids=validation_ids,
                                                        train_y=y_train)
        except Exception as error:
            record["evaluation"] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        write_json(result_path, record)
        results[name] = record
    summary = {"author": "Chang Cheng", "phase": "development_comparison",
               "defaults_frozen": False, "tuning_complete": False,
               "remaining_work": ["Expand unconstrained boundary optima and refine interior intervals.",
                                  "Repeat experiments and verify numerical convergence.",
                                  "Compare empirical data before deciding package defaults.",
                                  "Freeze settings before a separate final-test evaluation."],
               "models": results}
    write_json(output_dir / "comparison.json", summary)
    return summary
