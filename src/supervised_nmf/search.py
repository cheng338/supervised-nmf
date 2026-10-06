"""Bounded training-only rank exploration, authored by Chang Cheng.

This optional search extends one coordinate of a declared classification grid.
It does not change estimator defaults or claim joint parameter optimality.
"""

from copy import deepcopy
from numbers import Integral
from pathlib import Path
import time

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y

from .classifier import NMFClassifier
from .decomposition import FeatureSPA
from .evaluation import _digest, _jsonable, _search_identity, refinement_diagnostics
from .screening import BinaryScreening
from .study import (
    _checkpoint_lock, _read_receipt, _runtime_identity, _write_receipt,
    calibration_knn_grid,
)
from .validation import build_cv_folds, validate_metadata


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _rank_parameter(estimator, candidates, parameter):
    declared = set().union(*(set(candidate) for candidate in candidates))
    if parameter is None:
        choices = sorted(key for key in declared if key == "n_components" or key.endswith("__n_components"))
        if len(choices) != 1:
            raise ValueError("Declare exactly one n_components parameter or supply rank_parameter explicitly.")
        parameter = choices[0]
    if not isinstance(parameter, str) or parameter not in estimator.get_params(deep=True):
        raise ValueError("rank_parameter must name an estimator parameter.")
    if parameter != "n_components" and not parameter.endswith("__n_components"):
        raise ValueError("rank_parameter must be n_components or a pipeline step's n_components.")
    for candidate in candidates:
        if parameter not in candidate:
            raise ValueError("Every initial candidate must explicitly declare rank_parameter.")
        _positive_integer(candidate[parameter], "Every declared rank")
    if type(estimator) is NMFClassifier and parameter == "n_components":
        return parameter
    if type(estimator) is Pipeline:
        step_name, attribute = parameter.rsplit("__", 1)
        if attribute == "n_components" and type(estimator.named_steps.get(step_name)) in (PCA, NMF, FeatureSPA):
            return parameter
    raise TypeError("Rank expansion supports NMFClassifier or a PCA, NMF, or FeatureSPA pipeline step.")


def _physical_limit(estimator, parameters, parameter, candidate, X, folds):
    """Use only dimensions observed in the winner's own training folds."""
    configured = clone(estimator).set_params(**parameters)
    known_feature_bound = type(configured) is NMFClassifier
    factorizer = configured
    if type(configured) is Pipeline:
        step_name = parameter.rsplit("__", 1)[0]
        step_index = [name for name, _ in configured.steps].index(step_name)
        factorizer = configured.named_steps[step_name]
        # Arbitrary preceding transformers can increase feature dimension. In
        # that case only the row bound is known without another fitted model.
        known_feature_bound = all(type(step) is BinaryScreening for _, step in configured.steps[:step_index])
    limits = []
    evidence = []
    for (train, _), record in zip(folds, candidate["folds"]):
        dimensions = [len(train)]
        selected = record.get("diagnostics", {}).get("n_selected_features")
        if known_feature_bound:
            dimensions.append(X.shape[1])
            if selected is not None:
                dimensions.append(int(selected))
        bound = min(dimensions)
        if type(factorizer) is PCA and factorizer.svd_solver == "arpack":
            bound -= 1
        limits.append(bound)
        evidence.append({"training_rows": len(train), "selected_features": selected,
                         "feature_bound_used": known_feature_bound, "maximum_rank": bound})
    return min(limits), evidence


def calibration_rank_search(estimator, param_grid, X, y, *, rank_parameter=None,
                            cv=5, seed=0, threads=1, checkpoint_dir,
                            max_expansions=2, max_rank=None, groups=None,
                            folds=None, data_ids=None):
    """Extend an upper-boundary rank winner using frozen training folds only.

    First evaluate the explicit grid with :func:`calibration_knn_grid`. When
    its qualified winner uses the largest declared rank, evaluate one doubled
    rank at that winner's *other unchanged parameters*. Clamp only new ranks
    to the fold dimensions and optional ``max_rank``. Strictly better complete
    CV scores replace the incumbent; ties retain the earlier candidate.

    ``max_expansions`` counts additional one-candidate stages, not the initial
    grid. Explicit initial candidates are never silently removed or clamped.
    Every stage uses the same validated folds, groups and sample identifiers.
    Immutable identities bind checkpoints to all data, settings, package code
    and numerical runtime. Resuming requires the identical invocation.

    ``search_complete`` refers only to this bounded upper-rank policy. It is
    false after a user cap, budget exhaustion or an unqualified expansion, and is never
    a claim of global or joint optimality. The report retains stage candidate
    records, numerical failures and other parameter-boundary diagnostics. No
    full-training fit or held-out prediction occurs. Only the KNN estimators
    supported by the shared-grid evaluator are accepted.
    """
    start = time.perf_counter()
    max_expansions = _positive_integer(max_expansions, "max_expansions", minimum=0)
    threads = _positive_integer(threads, "threads")
    if max_rank is not None:
        max_rank = _positive_integer(max_rank, "max_rank")
    estimator = clone(estimator)
    candidates = list(ParameterGrid(param_grid))
    parameter = _rank_parameter(estimator, candidates, rank_parameter)
    initial_maximum = max(int(params[parameter]) for params in candidates)
    if max_rank is not None and initial_maximum > max_rank:
        raise ValueError("max_rank must not exclude any explicit initial candidate.")
    X, y = check_X_y(X, y)
    X, y = np.array(X, copy=True), np.array(y, copy=True)
    X.setflags(write=False)
    y.setflags(write=False)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    pairs = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    runtime = _runtime_identity(estimator, threads)
    identity = _jsonable({
        "schema_version": 1, "method": "bounded_coordinate_rank_expansion",
        "calibration": _search_identity(estimator, candidates, X, y, cv, seed, threads,
                                         refit=False, groups=groups, folds=pairs, data_ids=data_ids),
        "runtime": runtime, "rank_parameter": parameter,
        "max_expansions": max_expansions, "max_rank": max_rank,
        "proposal": "double_incumbent_rank_with_other_parameters_fixed",
        "tie_policy": "retain_incumbent_then_initial_parameter_grid_order",
    })
    fingerprint = _digest(identity)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stages, all_candidates = [], []
    incumbent = None
    rank_limit = None
    bounds = []
    highest_attempted = initial_maximum
    expansions = 0
    stage_grid = param_grid
    with _checkpoint_lock(directory):
        manifest = directory / "identity.json"
        if manifest.exists():
            if _read_receipt(manifest) != identity:
                raise ValueError("Rank-search checkpoint is incompatible with these data, folds, settings, source or runtime.")
        else:
            _write_receipt(manifest, identity)
        while True:
            if _runtime_identity(estimator, threads) != runtime:
                raise ValueError("Source or numerical runtime changed during rank search.")
            stage = calibration_knn_grid(
                estimator, stage_grid, X, y, cv=cv, seed=seed, threads=threads,
                checkpoint_dir=directory / f"stage-{len(stages):03d}",
                groups=groups, folds=pairs, data_ids=data_ids,
            )
            stages.append(stage)
            new_valid = []
            for row in stage["candidates"]:
                row = deepcopy(row)
                row.update(index=len(all_candidates), stage=len(stages) - 1)
                all_candidates.append(row)
                if row["valid"]:
                    new_valid.append(row)
                    if incumbent is None or row["mean_balanced_accuracy"] > incumbent["mean_balanced_accuracy"]:
                        incumbent = row
            if incumbent is None:
                reason, complete = "no_qualified_initial_candidate", False
                break
            rank_limit, bounds = _physical_limit(estimator, incumbent["parameters"], parameter,
                                                 incumbent, X, pairs)
            current_rank = int(incumbent["parameters"][parameter])
            if not new_valid:
                reason, complete = "no_qualified_expansion_candidate", False
                break
            if current_rank < highest_attempted:
                higher_qualified = any(row["valid"] and row["parameters"][parameter] > current_rank
                                       for row in all_candidates)
                reason, complete = (("winner_below_upper_rank_boundary", True) if higher_qualified else
                                    ("unqualified_upper_rank_candidates", False))
                break
            if current_rank >= rank_limit:
                reason, complete = "physical_rank_limit", True
                break
            if max_rank is not None and current_rank >= max_rank:
                reason, complete = "user_rank_limit", False
                break
            if expansions >= max_expansions:
                reason, complete = "expansion_budget_exhausted", False
                break
            proposed = min(current_rank * 2, rank_limit)
            if max_rank is not None:
                proposed = min(proposed, max_rank)
            parameters = dict(incumbent["parameters"], **{parameter: proposed})
            stage_grid = {key: [value] for key, value in parameters.items()}
            highest_attempted = proposed
            expansions += 1
        report = _jsonable({
            "schema_version": 1, "author": "Chang Cheng", "state": "cv_completed",
            "fingerprint": fingerprint, "identity": identity, "stages": stages,
            "candidates": all_candidates,
            "selected_parameters": incumbent["parameters"] if incumbent else None,
            "selected_candidate_index": incumbent["index"] if incumbent else None,
            "selected_mean_balanced_accuracy": incumbent["mean_balanced_accuracy"] if incumbent else None,
            "selection_role": "provisional_highest_complete_training_cv_score_among_evaluated_candidates",
            "rank_parameter": parameter, "expansions_performed": expansions,
            "largest_attempted_rank": highest_attempted, "physical_rank_limit": rank_limit,
            "physical_limit_evidence": bounds, "stop_reason": reason, "search_complete": complete,
            "unresolved_upper_rank_boundary": not complete,
            "refinement": refinement_diagnostics([row["parameters"] for row in all_candidates],
                                                  incumbent["parameters"] if incumbent else None),
            "search_scope": "initial_explicit_grid_then_one_rank_coordinate_with_other_parameters_fixed",
            "global_optimum_claimed": False, "tuning_complete": False, "defaults_frozen": False,
            "refit_performed": False, "final_fit_seconds": 0.0,
            "tuning_seconds": sum(stage["tuning_seconds"] for stage in stages),
            "new_representation_fits": sum(stage["new_representation_fits"] for stage in stages),
            "reused_representation_folds": sum(stage["reused_representation_folds"] for stage in stages),
            "current_call_wall_seconds": time.perf_counter() - start,
        })
        _write_receipt(directory / "report.json", report)
        return report
