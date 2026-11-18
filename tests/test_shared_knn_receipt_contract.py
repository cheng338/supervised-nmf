"""Current shared-KNN report contracts used by the frozen BA readers."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _audit_identity import verify_shared_knn_cv_receipt


def receipt():
    """Only the metric/header portion; caller tests cover source/fold identities."""
    candidates = []
    for index, score in enumerate((.6, .8)):
        candidates.append({
            "index": index, "parameters": {"n_neighbors": index + 1},
            "valid": True, "invalid_fold_count": 0,
            "mean_score": score, "mean_balanced_accuracy": score,
            "folds": [{"fold": fold, "status": "valid", "score": score,
                       "balanced_accuracy": score, "warnings": []}
                      for fold in range(5)],
        })
    return {
        "schema_version": 4, "state": "cv_completed", "refit_performed": False,
        "scoring": "balanced_accuracy",
        "identity": {"schema_version": 4, "scoring": "balanced_accuracy",
                     "calibration": {"identity_schema": 2, "scoring": "balanced_accuracy"}},
        "candidates": candidates, "ranked_candidate_indices": [1, 0],
        "selected_parameters": {"n_neighbors": 2},
        "selected_mean_score": .8, "selected_mean_balanced_accuracy": .8,
    }


def failed_receipt(pattern):
    report = receipt()
    candidate = report["candidates"][0]
    candidate.update(valid=False, mean_score=None, mean_balanced_accuracy=None)
    if pattern in ("failed_absent", "failed_null"):
        candidate["folds"] = [
            {"fold": fold, "status": "failed", "error_type": "ValueError",
             "error": "screening selected no features", "warnings": []}
            for fold in range(5)]
        if pattern == "failed_null":
            for fold in candidate["folds"]:
                fold.update(score=None, balanced_accuracy=None)
        candidate["invalid_fold_count"] = 5
    elif pattern == "nonconverged_scored":
        # A scored nonconverged fold disqualifies the entire candidate; its
        # recorded score is still legitimate evidence and is not a winner.
        candidate["folds"][0].update(
            status="nonconverged", score=1., balanced_accuracy=1.,
            warnings=[{"category": "ConvergenceWarning", "message": "iteration limit"}])
        candidate["invalid_fold_count"] = 1
    elif pattern == "mixed_failed":
        candidate["folds"][0] = {
            "fold": 0, "status": "failed", "error_type": "ValueError",
            "error": "head could not be fitted", "warnings": []}
        candidate["invalid_fold_count"] = 1
    else:
        raise AssertionError(pattern)
    report.update(candidates=[candidate], ranked_candidate_indices=[],
                  selected_parameters=None, selected_mean_score=None,
                  selected_mean_balanced_accuracy=None)
    return report


def test_complete_qualified_report_accepts_integer_endpoints_without_mutation():
    report = receipt()
    for candidate, score in zip(report["candidates"], (0, 1)):
        candidate.update(mean_score=score, mean_balanced_accuracy=score)
        for fold in candidate["folds"]:
            fold.update(score=score, balanced_accuracy=score)
    report.update(selected_mean_score=1, selected_mean_balanced_accuracy=1)
    before = deepcopy(report)
    assert verify_shared_knn_cv_receipt(report)
    assert report == before


def test_scored_nonconverged_candidate_cannot_displace_qualified_winner():
    report = receipt()
    report["candidates"][0] = failed_receipt("nonconverged_scored")["candidates"][0]
    report["ranked_candidate_indices"] = [1]
    before = deepcopy(report)
    assert report["candidates"][0]["folds"][0]["score"] == 1.
    assert report["selected_mean_score"] == .8
    assert verify_shared_knn_cv_receipt(report)
    assert report == before


@pytest.mark.parametrize("pattern", [
    "failed_absent", "failed_null", "nonconverged_scored", "mixed_failed"])
def test_unqualified_producer_patterns_keep_scores_unavailable(pattern):
    report = failed_receipt(pattern)
    before = deepcopy(report)
    assert verify_shared_knn_cv_receipt(report)
    assert report == before


def score_record(report, level):
    if level == "fold":
        return report["candidates"][0]["folds"][0], "score", "balanced_accuracy"
    if level == "mean":
        return report["candidates"][0], "mean_score", "mean_balanced_accuracy"
    return report, "selected_mean_score", "selected_mean_balanced_accuracy"


@pytest.mark.parametrize("level", ["fold", "mean", "selected"])
@pytest.mark.parametrize("damage", ["missing_generic", "missing_named", "missing_both", "conflict"])
def test_current_score_aliases_are_complete_and_consistent(level, damage):
    report = receipt()
    record, generic, named = score_record(report, level)
    if damage in ("missing_generic", "missing_both"):
        record.pop(generic)
    if damage in ("missing_named", "missing_both"):
        record.pop(named)
    if damage == "conflict":
        record[generic] = .25
    with pytest.raises(ValueError):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("damage", ["generic_only", "named_only", "conflict"])
def test_failed_fold_cannot_retain_one_sided_or_conflicting_scores(damage):
    report = failed_receipt("failed_absent")
    fold = report["candidates"][0]["folds"][0]
    if damage == "generic_only":
        fold["score"] = None
    elif damage == "named_only":
        fold["balanced_accuracy"] = None
    else:
        fold.update(score=.2, balanced_accuracy=.3)
    with pytest.raises(ValueError):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("level", ["mean", "selected"])
def test_unqualified_candidate_cannot_acquire_an_aggregate_or_selection(level):
    report = failed_receipt("nonconverged_scored")
    record, generic, named = score_record(report, level)
    record.update({generic: .6, named: .6})
    with pytest.raises(ValueError):
        verify_shared_knn_cv_receipt(report)


def test_selected_aliases_must_name_the_qualified_local_best():
    report = receipt()
    # The pair agrees internally but incorrectly names the lower-scoring row.
    report.update(selected_mean_score=.6, selected_mean_balanced_accuracy=.6)
    with pytest.raises(ValueError, match="local best"):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("score", [True, ".6", None, -.1, 1.1, float("nan"), float("inf")])
def test_qualified_fold_score_is_a_finite_unit_interval_number_not_coercible_metadata(score):
    report = receipt()
    report["candidates"][0]["folds"][0].update(score=score, balanced_accuracy=score)
    with pytest.raises(ValueError):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("level", ["mean", "selected"])
def test_aggregate_score_rejects_boolean_even_when_numeric_comparison_would_match(level):
    report = receipt()
    for candidate in report["candidates"]:
        candidate.update(mean_score=1., mean_balanced_accuracy=1.)
        for fold in candidate["folds"]:
            fold.update(score=1., balanced_accuracy=1.)
    report.update(selected_mean_score=1., selected_mean_balanced_accuracy=1.)
    record, generic, named = score_record(report, level)
    record.update({generic: True, named: True})
    with pytest.raises(ValueError):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("level", ["report", "identity", "calibration"])
@pytest.mark.parametrize("damage", ["float", "boolean", "missing"])
def test_current_schema_markers_are_explicit_integer_versions(level, damage):
    report = receipt()
    record, key = ((report, "schema_version") if level == "report" else
                   (report["identity"], "schema_version") if level == "identity" else
                   (report["identity"]["calibration"], "identity_schema"))
    if damage == "missing":
        record.pop(key)
    else:
        record[key] = float(record[key]) if damage == "float" else True
    with pytest.raises(ValueError, match="schema"):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("value", [0, True, None, "missing"])
def test_current_cv_receipt_explicitly_excludes_a_final_refit(value):
    report = receipt()
    if value == "missing":
        report.pop("refit_performed")
    else:
        report["refit_performed"] = value
    with pytest.raises(ValueError, match="refit"):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("value", [1, 1.0, None, "missing"])
def test_candidate_qualification_is_explicit_boolean_metadata(value):
    report = receipt()
    if value == "missing":
        report["candidates"][0].pop("valid")
    else:
        report["candidates"][0]["valid"] = value
    with pytest.raises(ValueError, match="boolean"):
        verify_shared_knn_cv_receipt(report)


@pytest.mark.parametrize("level", ["report", "identity", "calibration"])
def test_all_current_identity_levels_retain_the_frozen_objective(level):
    report = receipt()
    record = (report if level == "report" else report["identity"] if level == "identity"
              else report["identity"]["calibration"])
    record["scoring"] = "accuracy"
    with pytest.raises(ValueError, match="objective"):
        verify_shared_knn_cv_receipt(report)


def test_unfinished_report_is_not_a_completed_cv_receipt():
    report = receipt()
    report["state"] = "cv_running"
    with pytest.raises(ValueError, match="complete"):
        verify_shared_knn_cv_receipt(report)
