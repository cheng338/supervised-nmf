"""Screening search diagnostics must describe candidates without changing them."""

import json
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import BinaryScreening, NMFClassifier
from supervised_nmf._search_diagnostics import (
    mark_inactive_refinement, screening_grid_diagnostics,
)


def quiet(estimator, candidates):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = screening_grid_diagnostics(estimator, candidates)
    assert not caught
    return result


def test_constructor_defaults_alone_do_not_warn_or_switch_modes():
    standalone = BinaryScreening()
    classifier = NMFClassifier()
    first = quiet(standalone, [{}])["candidates"][0]["screens"][0]
    second = quiet(classifier, [{}])["candidates"][0]["screens"][0]
    assert first["mode"] == "absolute"
    assert first["active_parameter"] == "score_threshold"
    assert second["mode"] == "absolute"
    assert second["active_parameter"] == "score_threshold"
    assert first["active_value"] == second["active_value"] == 2.0
    assert standalone.screening == classifier.screening == "absolute"


def test_explicit_inactive_grid_keys_warn_once_and_keep_all_candidates():
    candidates = list(ParameterGrid({"score_threshold": [1, 2, 4]}))
    with pytest.warns(UserWarning, match="Inactive screening grid parameters") as caught:
        report = screening_grid_diagnostics(BinaryScreening(screening="percentile"), candidates)
    assert len(caught) == 1
    assert len(report["candidates"]) == 3
    assert report["inactive_parameters"] == [
        {"parameter": "score_threshold", "mode": "percentile", "candidate_indices": [0, 1, 2]}]
    assert candidates == [{"score_threshold": 1}, {"score_threshold": 2}, {"score_threshold": 4}]


def test_conditional_grids_only_report_mode_appropriate_parameters():
    candidates = list(ParameterGrid([
        {"screening": ["absolute"], "score_threshold": [1, 3]},
        {"screening": ["percentile"], "percentile": [10, 30]},
    ]))
    report = quiet(BinaryScreening(), candidates)
    assert [row["screens"][0]["active_value"] for row in report["candidates"]] == [1, 3, 10, 30]
    assert report["inactive_parameters"] == []


def test_nested_prefixes_and_step_replacements_use_effective_candidate():
    original = Pipeline([("features", Pipeline([("screen", BinaryScreening(screening="percentile"))])),
                         ("knn", KNeighborsClassifier())])
    replacement = BinaryScreening(screening="absolute", score_threshold=7)
    candidates = [{"features__screen": replacement,
                   "features__screen__score_threshold": 3,
                   "features__screen__percentile": 40}]
    with pytest.warns(UserWarning) as caught:
        report = screening_grid_diagnostics(original, candidates)
    assert len(caught) == 1
    assert report["candidates"][0]["screens"] == [{
        "path": "features__screen", "mode": "absolute",
        "active_parameter": "features__screen__score_threshold", "active_value": 3,
        "inactive_grid_parameters": ["features__screen__percentile"],
    }]
    # Applying nested parameters to a replaced estimator must not mutate the
    # caller's grid value or original pipeline.
    assert replacement.score_threshold == 7 and replacement.percentile == 10
    assert original.named_steps["features"].named_steps["screen"].screening == "percentile"


def test_screening_none_marks_both_explicit_parameters_inactive():
    with pytest.warns(UserWarning) as caught:
        report = screening_grid_diagnostics(NMFClassifier(screening="none"), [
            {"percentile": 25, "score_threshold": 3}, {}])
    assert len(caught) == 1
    first, second = [row["screens"][0] for row in report["candidates"]]
    assert first["active_parameter"] is None and first["active_value"] is None
    assert first["inactive_grid_parameters"] == ["percentile", "score_threshold"]
    assert second["inactive_grid_parameters"] == []


class UnknownScreen(BaseEstimator):
    def __init__(self, screening="percentile", score_threshold=2):
        self.screening = screening
        self.score_threshold = score_threshold


def test_unknown_estimator_parameter_names_do_not_imply_known_semantics():
    report = quiet(UnknownScreen(), [{"score_threshold": 4}])
    assert report == {"candidates": [{"index": 0, "screens": []}], "inactive_parameters": []}


def test_invalid_configuration_is_left_for_ordinary_candidate_failure():
    report = quiet(BinaryScreening(), [{"not_a_parameter": 1}, {"screening": "unknown"}])
    assert [row["screens"] for row in report["candidates"]] == [[], []]


def test_known_subclasses_and_numpy_scalar_values_are_json_safe():
    class ScreenSubclass(BinaryScreening):
        pass
    report = quiet(ScreenSubclass(screening="absolute"), [{"score_threshold": np.float64(3)}])
    assert report["candidates"][0]["screens"][0]["active_value"] == 3
    json.dumps(report, allow_nan=False)


def test_refinement_marks_only_selected_candidates_inactive_keys():
    with pytest.warns(UserWarning):
        diagnostics = screening_grid_diagnostics(BinaryScreening(screening="percentile"), [
            {"score_threshold": 1}, {"screening": "absolute", "score_threshold": 2}])
    refinement = {"status": "pending_refinement_and_replication", "parameters": {
        "score_threshold": {"status": "lower_boundary_requires_expansion_or_constraint", "selected": 1},
        "n_components": {"status": "upper_boundary_requires_expansion_or_constraint", "selected": 2},
    }}
    first = mark_inactive_refinement(refinement, diagnostics, 0)
    second = mark_inactive_refinement(refinement, diagnostics, 1)
    assert first["parameters"]["score_threshold"]["status"] == "inactive_for_selected_screening_mode"
    assert first["parameters"]["score_threshold"]["screening_mode"] == "percentile"
    assert first["parameters"]["n_components"] == refinement["parameters"]["n_components"]
    assert second == refinement
    assert refinement["parameters"]["score_threshold"]["status"] == "lower_boundary_requires_expansion_or_constraint"
    assert mark_inactive_refinement(refinement, diagnostics, None) == refinement
