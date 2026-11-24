"""Bounded training-only rank exploration, authored by Chang Cheng.

This optional search extends one coordinate of a declared classification grid.
It does not change estimator defaults or claim joint parameter optimality.
"""

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from .classifier import NMFClassifier
from ._calibration import _validate_scoring
from ._feature_names import _input_feature_names
from ._search_diagnostics import mark_inactive_refinement
from .evaluation import _check_transform_output, _digest, _jsonable, _search_identity, refinement_diagnostics
from ._identity import implementation_sources
from ._labels import _check_labeled_data, _validate_class_labels
from ._neighbors import _is_integer_index
from .screening import BinaryScreening
from .study import (
    _calibration_knn_grid, _check_captured_folds, _checkpoint_lock, _rank_axis_kind, _read_receipt, _runtime_identity,
    _write_receipt, calibration_knn_grid,
)
from .validation import _rank_count, _validate_fixed_seed, _validate_threads, build_cv_folds, validate_metadata


def _positive_integer(value, name, *, minimum=1):
    if _is_integer_index(value):
        try:
            count = _rank_count(value)
        except (MemoryError, RecursionError):
            raise
        except Exception as error:
            raise ValueError(f"{name} must identify one integer count. {error}") from error
        if count >= minimum:
            return count
    raise ValueError(f"{name} must be an integer >= {minimum}.")


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
    ranks = []
    for candidate in candidates:
        if parameter not in candidate:
            raise ValueError("Every initial candidate must explicitly declare rank_parameter.")
        ranks.append(_positive_integer(candidate[parameter], "Every declared rank"))
    if _rank_axis_kind(estimator, candidates, parameter) == "rank":
        return parameter, ranks
    raise TypeError("Rank expansion supports NMFClassifier or a PCA, NMF, or FeatureSPA pipeline step.")


def _physical_limit(estimator, parameters, parameter, candidate, X, folds):
    """Use only dimensions observed in the winner's own training folds."""
    configured = clone(estimator).set_params(**clone(parameters, safe=False))
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
                            folds=None, data_ids=None, scoring="accuracy", _require_completed=False):
    """Extend an upper-boundary rank winner using frozen training folds only.

    First evaluate the explicit grid with :func:`calibration_knn_grid`. When
    its qualified winner uses the largest declared rank, evaluate one doubled
    rank at that winner's *other unchanged parameters*. Clamp only new ranks
    to the fold dimensions and optional ``max_rank``. Strictly better complete
    CV scores replace the incumbent; ties retain the earlier candidate.
    ``scoring`` is ``"accuracy"`` (the default) or ``"balanced_accuracy"``,
    and the selected metric controls every initial and expanded stage.

    ``threads`` accepts positive Python/NumPy integer scalars within the native
    C-int range. ``seed`` follows shared-grid calibration's fixed integer domain
    0..2**32-1, even with explicit folds. Boolean and temporal scalars are not
    counts or fixed seeds. Both controls are normalized before fold generation,
    runtime initialization or checkpoint creation.

    ``max_expansions`` counts additional one-candidate stages, not the initial
    grid. Explicit initial candidates are never silently removed or clamped.
    Every stage uses the same validated folds, groups and sample identifiers.
    Class labels must not end in a NUL codepoint; groups and data IDs retain
    their separate metadata validation rules.
    Immutable identities bind checkpoints to all data, settings, package code
    and numerical runtime. Resuming requires the identical invocation.

    ``search_complete`` refers only to this bounded upper-rank policy. It is
    false after a user cap, budget exhaustion or an unqualified expansion, and is never
    a claim of global or joint optimality. The report retains stage candidate
    records, numerical failures and other parameter-boundary diagnostics. No
    full-training fit or held-out prediction occurs. Only the KNN estimators
    supported by the shared-grid evaluator are accepted.

    A pipeline containing FeatureSPA uses fixed-basis NNLS coordinates for
    both training and validation observations. Each training fold fits its
    own basis; neighbor choices reuse the coordinates actually supplied to
    that fold's classifier.
    """
    start = time.perf_counter()
    threads = _validate_threads(threads)
    seed = _validate_fixed_seed(seed)
    scoring = _validate_scoring(scoring)
    max_expansions = _positive_integer(max_expansions, "max_expansions", minimum=0)
    if max_rank is not None:
        max_rank = _positive_integer(max_rank, "max_rank")
    candidates = list(ParameterGrid(param_grid))
    raw_y = y
    feature_names = _input_feature_names(X)
    X, _, y = _check_labeled_data(X, y)
    _validate_class_labels(raw_y, y)
    # Validate before a custom clone can hide the original implementation.
    rng_limits = []
    implementation_sources(estimator, candidates, _split_limits=rng_limits)
    estimator = clone(estimator)
    parameter, stage_ranks = _rank_parameter(estimator, candidates, rank_parameter)
    initial_maximum = max(stage_ranks)
    if max_rank is not None and initial_maximum > max_rank:
        raise ValueError("max_rank must not exclude any explicit initial candidate.")
    X, y = np.array(X, copy=True), np.array(y, copy=True)
    X.setflags(write=False)
    y.setflags(write=False)
    groups = validate_metadata(groups, len(y), name="groups")
    data_ids = validate_metadata(data_ids, len(y), name="data_ids", unique=True)
    pairs = build_cv_folds(X, y, cv=cv, seed=seed, groups=groups, folds=folds)
    runtime = _runtime_identity(estimator, threads, candidates)
    identity = _jsonable({
        "schema_version": 2, "method": "bounded_coordinate_rank_expansion",
        "calibration": _search_identity(estimator, candidates, X, y, cv, seed, threads,
                                         refit=False, groups=groups, folds=pairs, data_ids=data_ids,
                                         scoring=scoring, _feature_names=feature_names),
        "scoring": scoring,
        "runtime": runtime, "rank_parameter": parameter,
        "max_expansions": max_expansions, "max_rank": max_rank,
        "proposal": "double_incumbent_rank_with_other_parameters_fixed",
        "tie_policy": "retain_incumbent_then_initial_parameter_grid_order",
    })
    fingerprint = _digest(identity)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stages, all_candidates = [], []
    # Keep numerical coordinates and the winner's original parameter objects.
    # The stage reports contain display values for custom estimators and ranks.
    attempted_ranks = []
    incumbent_parameters = None
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
            if "unverified scalar owner" in rng_limits:
                raise ValueError("Cannot verify the backing storage of a captured NumPy scalar in this saved rank search.")
            if "unverified fit RNG" in rng_limits or "ambient fit RNG" in rng_limits:
                raise ValueError("Cannot verify the fit generator in this saved rank search.")
            if "captured fit RNG" in rng_limits or "captured scalar storage" in rng_limits:
                # An expanded stage can be the first to change captured state.
                # Check all committed stages before refitting any missing fold.
                for stage in directory.glob("stage-*"):
                    stage_manifest = stage / "identity.json"
                    if stage_manifest.is_file():
                        _check_captured_folds(stage, _digest(_read_receipt(stage_manifest)))
                    elif any((stage / "folds").glob("*.json")):
                        raise ValueError("Saved rank-search folds require their original stage identity.")
        else:
            if _require_completed:
                return None
            if rng_limits and (any(directory.glob("stage-*/identity.json")) or any(directory.glob("stage-*/folds/*.json"))
                               or any(directory.glob("stage-*/report.json")) or (directory / "report.json").exists()):
                raise ValueError("Saved rank-search stages with unverified randomness or captured state require their original identity.")
            _write_receipt(manifest, identity)
        while True:
            if _runtime_identity(estimator, threads, candidates) != runtime:
                raise ValueError("Source or numerical runtime changed during rank search.")
            stage_parameters = list(ParameterGrid(stage_grid))
            stage_options = {"_reuse_only": True} if _require_completed else {}
            if feature_names is not None:
                stage_options["_feature_names"] = feature_names
            stage_search = _calibration_knn_grid if stage_options else calibration_knn_grid
            stage = stage_search(
                estimator, stage_grid, X, y, cv=cv, seed=seed, threads=threads,
                checkpoint_dir=directory / f"stage-{len(stages):03d}",
                groups=groups, folds=pairs, data_ids=data_ids, scoring=scoring,
                **stage_options,
            )
            if stage is None and _require_completed:
                return None
            stages.append(stage)
            new_valid = []
            for row in stage["candidates"]:
                row = deepcopy(row)
                attempted_ranks.append(stage_ranks[row["index"]])
                row.update(stage_candidate_index=row["index"],
                           index=len(all_candidates), stage=len(stages) - 1)
                all_candidates.append(row)
                if row["valid"]:
                    new_valid.append(row)
                    if incumbent is None or row["mean_score"] > incumbent["mean_score"]:
                        incumbent = row
                        incumbent_parameters = stage_parameters[row["stage_candidate_index"]]
            if incumbent is None:
                reason, complete = "no_qualified_initial_candidate", False
                break
            rank_limit, bounds = _physical_limit(estimator, incumbent_parameters, parameter,
                                                 incumbent, X, pairs)
            current_rank = attempted_ranks[incumbent["index"]]
            if not new_valid:
                reason, complete = "no_qualified_expansion_candidate", False
                break
            if current_rank < highest_attempted:
                higher_qualified = any(attempted_ranks[row["index"]] > current_rank
                                       for row in all_candidates if row["valid"])
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
            parameters = dict(incumbent_parameters, **{parameter: proposed})
            stage_grid = {key: [value] for key, value in parameters.items()}
            stage_ranks = [proposed]
            highest_attempted = proposed
            expansions += 1
        selected = incumbent["parameters"] if incumbent else None
        # An expansion changes only the named rank. The initial grid supplies
        # every other rank's full range, selected value and effective owners.
        fixed_rank_bounds = {
            key: stages[0]["refinement"]["parameters"][key]
            for key in selected or ()
            if key != parameter and _rank_axis_kind(estimator, candidates, key)
        }
        to_refine = ({key: value for key, value in selected.items() if key not in fixed_rank_bounds}
                     if fixed_rank_bounds else selected)
        refinement = refinement_diagnostics(
            [row["parameters"] for row in all_candidates], to_refine,
            _rank_coordinates={parameter: (attempted_ranks, current_rank)} if incumbent else None)
        if fixed_rank_bounds:
            bounds_by_key = refinement["parameters"] | fixed_rank_bounds
            refinement["parameters"] = {key: bounds_by_key[key] for key in selected}
        report = _jsonable({
            "schema_version": 2, "author": "Chang Cheng", "state": "cv_completed",
            "fingerprint": fingerprint, "identity": identity, "stages": stages,
            "scoring": scoring,
            "candidates": all_candidates,
            "selected_parameters": incumbent["parameters"] if incumbent else None,
            "selected_candidate_index": incumbent["index"] if incumbent else None,
            "selected_mean_score": incumbent["mean_score"] if incumbent else None,
            f"selected_mean_{scoring}": incumbent["mean_score"] if incumbent else None,
            "selection_role": "provisional_highest_complete_training_cv_score_among_evaluated_candidates",
            "rank_parameter": parameter, "expansions_performed": expansions,
            "largest_attempted_rank": highest_attempted, "physical_rank_limit": rank_limit,
            "physical_limit_evidence": bounds, "stop_reason": reason, "search_complete": complete,
            "unresolved_upper_rank_boundary": not complete,
            "refinement": mark_inactive_refinement(
                refinement,
                stages[incumbent["stage"]].get("screening_diagnostics") if incumbent else None,
                incumbent["stage_candidate_index"] if incumbent else None),
            "search_scope": "initial_explicit_grid_then_one_rank_coordinate_with_other_parameters_fixed",
            "global_optimum_claimed": False, "tuning_complete": False, "defaults_frozen": False,
            "refit_performed": False, "final_fit_seconds": 0.0,
            "tuning_seconds": sum(stage["tuning_seconds"] for stage in stages),
            "new_representation_fits": sum(stage["new_representation_fits"] for stage in stages),
            "reused_representation_folds": sum(stage["reused_representation_folds"] for stage in stages),
            "current_call_wall_seconds": time.perf_counter() - start,
        })
        _check_transform_output(runtime["sklearn_transform_output"])
        _write_receipt(directory / "report.json", report)
        return report
