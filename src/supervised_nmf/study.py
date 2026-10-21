"""Execute explicit classification grids while sharing fold-local representations.

Author: Chang Cheng. This is a computation-saving implementation of an explicit
grid, not a parameter-proposal policy. Screening, factorization and projection
are fitted inside the same training folds for every declared neighbor count.
No full-training refit, held-out evaluation, or package-default decision occurs.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
from numbers import Integral
import os
from pathlib import Path
import time
import uuid
import warnings

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from threadpoolctl import threadpool_info, threadpool_limits

from .classifier import NMFClassifier
from ._neighbors import _KNNQueryReuse
from .screening import BinaryScreening
from .validation import build_cv_folds, validate_metadata
from .evaluation import (
    _diagnostics, _digest, _jsonable, _nonconverged, _search_identity,
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
    with (Path(directory) / ".study.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        locked = False
        try:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another process is writing this study checkpoint directory.") from error
            locked = True
            yield
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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


def _runtime_identity(estimator, threads):
    info = provenance(threads)
    runtime_keys = ("user_api", "internal_api", "prefix", "filepath", "version", "threading_layer", "architecture")
    libraries = sorted([{key: pool.get(key) for key in runtime_keys} for pool in info["threadpools"]],
                       key=lambda row: row.get("filepath") or "")
    sources = {}
    for obj in (estimator, *estimator.get_params(deep=True).values()):
        if not hasattr(obj, "get_params"):
            continue
        filename = inspect.getsourcefile(type(obj))
        if filename and Path(filename).is_file():
            sources[f"{type(obj).__module__}.{type(obj).__qualname__}"] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
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


def _run_representation_fold(estimator, kind, neighbor_parameter, representation_parameters,
                             neighbors, X, y, train, validation, record, *, screening_cache=None):
    fitted = None
    caught = []
    record.update(representation_fit_seconds=0.0, projection_seconds=0.0,
                  representation_fit_performed=False, neighbors={},
                  neighbor_reuse_setup_seconds=0.0, neighbor_query_seconds=0.0,
                  neighbor_queries=[],
                  screening_statistics_seconds=0.0, screening_selection_seconds=0.0,
                  screening_statistics_fit_performed=False, preflight_rejected=False)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fitted = clone(estimator).set_params(**representation_parameters)
                # This initial KNN fit is only the carrier for the representation;
                # its small cost is retained in the shared fit timer exactly once.
                fitted.set_params(**{neighbor_parameter: 1})
                selected = None
                template = _screening_template(fitted, kind)
                failure_stage = "screening"
                if template is not None and screening_cache is not None:
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
                    validation_coordinates = _validation_representation(fitted, kind, X[validation])
                finally:
                    record["projection_seconds"] = time.perf_counter() - start
                if not _finite(training_coordinates) or not _finite(validation_coordinates):
                    raise ValueError("The fitted representation contains nonfinite coordinates.")
                record["training_representation_dtype"] = str(training_coordinates.dtype)
                record["validation_representation_dtype"] = str(validation_coordinates.dtype)
                record["representation_shape"] = list(training_coordinates.shape)
                record["status"] = "nonconverged" if _nonconverged(fitted, caught) else "valid"
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error),
                              failure_stage=locals().get("failure_stage", "configuration"),
                              preflight_rejected=not record["representation_fit_performed"])
    finally:
        record["warnings"] = _warning_records(caught)
        if fitted is not None:
            record["diagnostics"] = _diagnostics(fitted)
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
    reuse = _KNNQueryReuse(models, validation_coordinates)
    record["neighbor_reuse_setup_seconds"] = time.perf_counter() - start
    for neighbors_count, classifier in models.items():
        result = record["neighbors"][str(neighbors_count)]
        shared_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                threshold = getattr(fitted, "decision_threshold_", None) if kind == "nmf" else None
                positive_index = getattr(fitted, "positive_class_index_", None) if threshold is not None else None
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
                score = float(balanced_accuracy_score(y[validation], predictions))
                if not np.isfinite(score):
                    raise ValueError("The calibration score is not finite.")
                all_warnings = fit_warnings[neighbors_count] + list(caught) + shared_warnings
                result.update(status="nonconverged" if _nonconverged(classifier, all_warnings) else "valid",
                              balanced_accuracy=score)
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, error=str(error))
        result["warnings"] = _warning_records(fit_warnings[neighbors_count] + list(caught) + shared_warnings)
    record["neighbor_query_seconds"] = reuse.query_seconds
    record["neighbor_queries"] = reuse.query_records()


def calibration_knn_grid(estimator, param_grid, X, y, *, cv=5, seed=0, threads=1,
                         checkpoint_dir, groups=None, folds=None, data_ids=None):
    """Evaluate an explicit grid, sharing fold-local screening and representations.

    The return value is a JSON-compatible report containing ordinary complete
    candidate/fold records, provisional CV ranking and refinement diagnostics.
    Classifier settings other than n_neighbors must be fixed on the estimator;
    including them in the grid is rejected. Representation random_state settings
    must be fixed integer seeds. Only standard supported estimator types are
    accepted because representation independence from k is essential.
    Other classifiers use the general-purpose ``calibration_search`` instead.
    ``groups`` keeps subjects together; ``folds`` supplies a validated complete
    CV partition. Optional unique ``data_ids`` bind checkpoints to sample IDs.
    Standard NMFClassifier and screen/PCA/KNN layouts reuse screening statistics
    across cutoffs and ranks. Arbitrary pipelines keep their ordinary fit path.
    Impossible ranks and empty screens are explicit failed candidates, never
    silently reduced ranks or changed cutoffs.

    Completed representation/fold receipts, including numerical failures, are
    reused only for the identical data, folds, grid, parameters, code and runtime.
    No estimator is serialized. An interrupted representation fold is retried;
    its earlier attempt remains recorded. Shared computation appears once in
    compute_seconds, never once per neighbor candidate. Candidate fit_seconds
    accounts for its KNN fit only; its shared_representation_key links the rest.
    Supported Euclidean heads also reuse neighbor queries across k. Uncertain
    batches and unsupported metrics/weights retain the public selected-k path.
    Shared query/setup costs are separate from each head's voting/fallback time.
    """
    start_call = time.perf_counter()
    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral) or cv < 2:
        raise ValueError("cv must be an integer >= 2.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("seed must be a fixed integer between 0 and 2**32 - 1.")
    if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    cv, seed, threads = int(cv), int(seed), int(threads)
    X, y = check_X_y(X, y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts.min() < cv:
        raise ValueError("Both binary classes must contain at least cv observations.")
    estimator = clone(estimator)
    kind, neighbor_parameter, forbidden = _layout(estimator)
    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        raise ValueError("The grid must declare at least one candidate.")
    rep_groups = {}
    for index, params in enumerate(candidates):
        if set(params) & forbidden:
            raise ValueError("Only n_neighbors may vary among classifier settings; fix other classifier knobs on the estimator.")
        configured = clone(estimator).set_params(**params)
        if _layout(configured)[:2] != (kind, neighbor_parameter):
            raise ValueError("Grid entries cannot change the classifier layout.")
        _fixed_randomness(configured)
        neighbors_count = configured.get_params()[neighbor_parameter]
        if isinstance(neighbors_count, (bool, np.bool_)) or not isinstance(neighbors_count, Integral) or neighbors_count < 1:
            raise ValueError("Every declared n_neighbors value must be a positive integer.")
        representation = {name: value for name, value in params.items() if name != neighbor_parameter}
        key = _digest(representation)
        group = rep_groups.setdefault(key, {"parameters": representation, "candidate_indices": [], "neighbors": []})
        group["candidate_indices"].append(index)
        if int(neighbors_count) not in group["neighbors"]:
            group["neighbors"].append(int(neighbors_count))
    X, y = np.array(X, copy=True), np.array(y, copy=True)
    X.setflags(write=False)
    y.setflags(write=False)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    folds = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    runtime = _runtime_identity(estimator, threads)
    identity = _jsonable({"schema_version": 3, "method": "explicit_grid_shared_knn_representation",
                         "calibration": _search_identity(estimator, candidates, X, y, cv, seed, threads,
                                                         refit=False, groups=groups, folds=folds, data_ids=data_ids),
                         "runtime": runtime, "folds": [{"training_indices": a, "validation_indices": b} for a, b in folds],
                         "neighbor_parameter": neighbor_parameter, "representation_backend": kind})
    fingerprint = _digest(identity)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    screening_cache = _ScreeningCache()
    reused = new_fits = 0
    with _checkpoint_lock(directory):
        manifest = directory / "identity.json"
        if manifest.exists():
            if _read_receipt(manifest) != identity:
                raise ValueError("Study checkpoint is incompatible with these data, grid, folds, settings, source or runtime.")
        else:
            _write_receipt(manifest, identity)
        with threadpool_limits(limits=threads):
            active_pools = threadpool_info()
            for representation_key, group in rep_groups.items():
                for fold, (train, validation) in enumerate(folds):
                    key = _digest({"identity": fingerprint, "representation": representation_key, "fold": fold})
                    path = directory / "folds" / f"{key}.json"
                    if _runtime_identity(estimator, threads) != runtime:
                        raise ValueError("Source or numerical runtime changed during explicit-grid calibration.")
                    if path.exists():
                        record = _read_receipt(path)
                        if record.get("identity") != fingerprint or record.get("key") != key or record.get("state") != "completed":
                            raise ValueError("The representation checkpoint belongs to another run or is incomplete.")
                        reused += 1
                    else:
                        record = {"identity": fingerprint, "key": key, "representation_key": representation_key,
                                  "parameters": group["parameters"], "fold": fold, "state": "running"}
                        attempt = directory / "attempts" / f"{uuid.uuid4().hex}.json"
                        _write_receipt(attempt, record)
                        try:
                            _run_representation_fold(estimator, kind, neighbor_parameter, group["parameters"],
                                                     group["neighbors"], X, y, train, validation, record,
                                                     screening_cache=screening_cache)
                            if _runtime_identity(estimator, threads) != runtime:
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
        by_representation_fold = {(row["representation_key"], row["fold"]): row for row in receipts}
        candidate_reports = []
        for index, params in enumerate(candidates):
            representation = {name: value for name, value in params.items() if name != neighbor_parameter}
            representation_key = _digest(representation)
            neighbors_count = int(clone(estimator).set_params(**params).get_params()[neighbor_parameter])
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
            scores = [row["balanced_accuracy"] for row in rows] if valid else []
            candidate_reports.append({"index": index, "parameters": params, "folds": rows, "valid": valid,
                                      "mean_balanced_accuracy": float(np.mean(scores)) if valid else None,
                                      "fold_score_std": float(np.std(scores, ddof=1)) if valid else None,
                                      "invalid_fold_count": sum(row["status"] != "valid" for row in rows)})
        ranked = sorted((row["index"] for row in candidate_reports if row["valid"]),
                        key=lambda index: (-candidate_reports[index]["mean_balanced_accuracy"], index))
        selected = candidates[ranked[0]] if ranked else None
        costs = {"screening_statistics_seconds": sum(row["screening_statistics_seconds"] for row in receipts),
                 "screening_selection_seconds": sum(row["screening_selection_seconds"] for row in receipts),
                 "representation_fit_seconds": sum(row["representation_fit_seconds"] for row in receipts),
                 "validation_projection_seconds": sum(row["projection_seconds"] for row in receipts),
                 "knn_fit_seconds": sum(item["fit_seconds"] for row in receipts for item in row["neighbors"].values()),
                 "knn_reuse_setup_seconds": sum(row["neighbor_reuse_setup_seconds"] for row in receipts),
                 "knn_shared_query_seconds": sum(row["neighbor_query_seconds"] for row in receipts),
                 "knn_prediction_seconds": sum(item["prediction_seconds"] for row in receipts for item in row["neighbors"].values())}
        report = {"schema_version": 3, "author": "Chang Cheng", "state": "cv_completed", "fingerprint": fingerprint,
                  "identity": identity, "active_threadpools": active_pools, "candidates": candidate_reports,
                  "ranked_candidate_indices": ranked, "selected_parameters": selected,
                  "selection_role": "provisional_highest_complete_training_cv_score_in_declared_grid",
                  "refinement": refinement_diagnostics(candidates, selected),
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
        report = _jsonable(report)
        _write_receipt(directory / "report.json", report)
        return report
