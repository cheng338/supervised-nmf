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
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y
from threadpoolctl import threadpool_info, threadpool_limits


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
    for package in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "Pillow"):
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


def _diagnostics(estimator):
    result = {}
    for attribute in ("n_iter_", "converged_", "last_transform_converged_",
                      "fit_times_", "n_features_in_"):
        if hasattr(estimator, attribute):
            result[attribute.rstrip("_")] = _jsonable(getattr(estimator, attribute))
    # Estimators and comparison Pipelines expose their selectors differently.
    selector = getattr(estimator, "selector_", None)
    if selector is None:
        selector = getattr(estimator, "screening_", None)
    if hasattr(estimator, "named_steps"):
        selector = estimator.named_steps.get("screen", selector)
    if selector is not None and hasattr(selector, "support_"):
        indices = np.flatnonzero(selector.get_support())
        result["n_selected_features"] = len(indices)
        result["selected_feature_indices"] = indices.tolist()
        for name in ("n_constant_features_", "n_infinite_scores_"):
            if hasattr(selector, name):
                result[name.rstrip("_")] = int(getattr(selector, name))
    elif hasattr(estimator, "n_features_in_"):
        result["n_selected_features"] = int(estimator.n_features_in_)
    return result


def _nonconverged(estimator, caught):
    return (any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or not bool(getattr(estimator, "converged_", True))
            or not bool(getattr(estimator, "last_transform_converged_", True)))


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


def _search_identity(estimator, candidates, X, y, cv, seed, threads, refit=True):
    environment = provenance(threads)
    runtime_keys = ("user_api", "internal_api", "prefix", "filepath", "version", "threading_layer", "architecture")
    numerical_runtime = sorted([{key: library.get(key) for key in runtime_keys}
                                for library in environment["threadpools"]],
                               key=lambda library: library.get("filepath") or "")
    return {"estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            "parameters": _jsonable(estimator.get_params(deep=True)),
            "candidates": _jsonable(candidates), "data_sha256": data_fingerprint(X, y),
            "cv": cv, "seed": seed, "threads": threads, "refit": refit,
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
                       threads=1, checkpoint_path=None, refit=True):
    """Calibrate on training data with complete, resumable per-fold records.

    Both fit and held-out-fold prediction warnings are retained. Convergence
    warnings invalidate that candidate. If a numerically valid CV winner fails
    full-training refit, try the next valid candidate and retain every attempt.
    If none can be fitted, a training-only majority-class fallback is returned
    with ``is_fallback=True``. It is never presented as the requested method.
    With ``refit=False``, only CV is performed and the returned estimator is
    None. This supports staged tuning without intermediate full-data fits.
    """
    X, y = check_X_y(X, y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Calibration currently requires exactly two classes.")
    if not isinstance(cv, int) or cv < 2 or counts.min() < cv:
        raise ValueError("Every class must have at least cv samples; cv must be >= 2.")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    if not isinstance(refit, bool):
        raise ValueError("refit must be boolean.")
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("At least one parameter candidate is required.")
    identity = _search_identity(estimator, candidates, X, y, cv, seed, threads, refit=refit)
    fingerprint = _digest(identity)
    folds = list(StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed).split(X, y))
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
                            predictions = fitted.predict(X[validation])
                        finally:
                            record["prediction_seconds"] = time.perf_counter() - inference_start
                        score = float(balanced_accuracy_score(y[validation], predictions))
                        if not math.isfinite(score):
                            raise ValueError("Nonfinite calibration score.")
                        record["balanced_accuracy"] = score
                        record["status"] = "nonconverged" if _nonconverged(fitted, caught) else "valid"
                    except Exception as error:
                        record.update(status="failed", error_type=type(error).__name__, error=str(error))
                record["warnings"] = _warning_records(caught)
                if fitted is not None:
                    record["diagnostics"] = _diagnostics(fitted)
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


def evaluate_classifier(estimator, X, y, *, threads=1, sample_ids=None):
    """Evaluate a fitted classifier; no fitting or threshold selection occurs."""
    X, y = check_X_y(X, y)
    classes = np.asarray(estimator.classes_)
    if len(classes) != 2 or not np.all(np.isin(y, classes)):
        raise ValueError("Evaluation labels must belong to the fitted binary classes.")
    if sample_ids is not None and len(sample_ids) != len(y):
        raise ValueError("sample_ids must match evaluation rows.")
    positive = classes[1]
    start = time.perf_counter()
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if hasattr(estimator, "predict_with_probabilities"):
                predicted, all_probabilities = estimator.predict_with_probabilities(X)
                probability = all_probabilities[:, 1]
                prediction_operation = "joint_label_and_probability_prediction_one_projection"
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
            else:
                predicted = estimator.predict(X)
                probability = estimator.predict_proba(X)[:, 1] if hasattr(estimator, "predict_proba") else None
                prediction_operation = "predict_plus_predict_proba_calls"
            if probability is not None and (not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1))):
                raise ValueError("Nonfinite or invalid predicted probabilities.")
        except Exception as error:
            return {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                    "prediction_seconds": time.perf_counter() - start,
                    "warnings": _warning_records(caught), "diagnostics": _diagnostics(estimator),
                    "data_sha256": data_fingerprint(X, y)}
    elapsed = time.perf_counter() - start
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=classes,
                                                                 zero_division=0)
    both_classes = len(np.unique(y)) == 2
    auc = ap = None
    if probability is not None:
        if both_classes:
            auc = float(roc_auc_score(y == positive, probability))
            ap = float(average_precision_score(y == positive, probability))
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
    return {"status": "provisional_nonconvergence" if _nonconverged(estimator, caught) else "valid",
            "metrics": metrics, "prediction_seconds": elapsed, "prediction_operation": prediction_operation,
            "warnings": _warning_records(caught),
            "diagnostics": _diagnostics(estimator), "data_sha256": data_fingerprint(X, y),
            "predictions": {"sample_ids": _jsonable(np.arange(len(y)) if sample_ids is None else sample_ids),
                            "true_labels": _jsonable(y), "predicted_labels": _jsonable(predicted),
                            "positive_class_probability": _jsonable(probability)}}


def comparison_models(*, random_state=0, quick=False, max_iter=20000, tol=1e-4):
    """Create independent CD/SPA/MU, PCA, screening-only and original baseline grids.

    Grids are deliberately modest starting grids. Caller-supplied JSON can expand
    them, and diagnostics identify unfinished boundaries/interior refinement.
    Absolute screening uses |t|, not the F statistic used for percentile ranking.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from .classifier import NMFClassifier
    from .screening import BinaryScreening

    ranks = [2] if quick else [2, 4]
    neighbors = [3] if quick else [3, 5, 7]
    percentages = [25.0] if quick else [10.0, 25.0]
    thresholds = [1.0] if quick else [1.0, 2.0]
    models = {}
    for screening in ("percentile", "absolute"):
        screen_parameter = "percentile" if screening == "percentile" else "score_threshold"
        screen_values = percentages if screening == "percentile" else thresholds
        for solver in ("cd", "spa", "mu"):
            models[f"screened_{solver}_{screening}"] = (
                NMFClassifier(solver=solver, screening=screening, random_state=random_state,
                              max_iter=max_iter, tol=tol),
                {screen_parameter: screen_values, "n_components": ranks, "n_neighbors": neighbors})
        screen = BinaryScreening(screening=screening)
        models[f"screened_pca_{screening}"] = (
            Pipeline([("screen", screen), ("pca", PCA(svd_solver="full", random_state=random_state)),
                      ("knn", KNeighborsClassifier(n_jobs=1))]),
            {f"screen__{screen_parameter}": screen_values, "pca__n_components": ranks,
             "knn__n_neighbors": neighbors})
        models[f"screening_only_{screening}"] = (
            Pipeline([("screen", clone(screen)), ("knn", KNeighborsClassifier(n_jobs=1))]),
            {f"screen__{screen_parameter}": screen_values, "knn__n_neighbors": neighbors})
    models["raw_knn"] = (KNeighborsClassifier(n_jobs=1), {"n_neighbors": neighbors})
    models["majority_class"] = (DummyClassifier(strategy="most_frequent"), {})
    models["unscreened_cd_nmf"] = (
        NMFClassifier(solver="cd", screening="none", random_state=random_state,
                      max_iter=max_iter, tol=tol),
        {"n_components": ranks, "n_neighbors": neighbors})
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
                                                        threads=threads, sample_ids=validation_ids)
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
