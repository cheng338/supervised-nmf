"""Check score attainability and the evidence used to qualify archived folds."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from supervised_nmf import NMFClassifier, evaluation
from supervised_nmf._calibration_report import (
    _at_iteration_cap, _check_fold_convergence, _joint_support_possible, _legacy_extra_label_score,
    _possible_balanced_score, _validation_support_bounds,
)
from supervised_nmf._identity import configuration_identity


def test_binary_balanced_accuracy_attainability():
    for a in range(2, 10):
        for b in range(2, 10):
            for i in range(a + 1):
                for j in range(b + 1):
                    assert _possible_balanced_score((i / a + j / b) / 2, a, b)
    assert not _possible_balanced_score(.75, 1, 1)
    assert not _possible_balanced_score(.375, 1, 2)
    assert not _possible_balanced_score(float(np.nextafter(.5, 1)), 128, 128)
    assert not _possible_balanced_score(.25 + 2 * np.spacing(.25), 100, 100)
    # 1/12 belongs to the 2-by-3 denominator lattice, but bounded class
    # hit counts cannot attain it on that partition.
    assert not _possible_balanced_score(1 / 12, 2, 3)


def test_scores_of_competing_candidates_require_one_fold_composition():
    sizes = [6, 6]
    bounds = _validation_support_bounds(sizes, [6, 6])
    assert bounds == [(2, 4), (2, 4)]
    # 1/8 needs the 2+4 split and 1/6 needs 3+3. Either is possible alone.
    for score in (0.125, 1/6):
        assert _joint_support_possible([{score}, set()], sizes, bounds, 6)
    assert not _joint_support_possible([{0.125, 1/6}, set()], sizes, bounds, 6)
    assert not _joint_support_possible([{0.125}, {1/6}], sizes, bounds, 6)
    # Class 0 would have an even count in every fold, despite an odd total.
    sizes = [6, 6, 6]
    bounds = _validation_support_bounds(sizes, [9, 9])
    assert not _joint_support_possible([{0.125}] * 3, sizes, bounds, 9)
    assert _joint_support_possible([{0.125}, {0.125}, {1/6}], sizes, bounds, 9)


def test_new_supports_and_legacy_joint_attainability(tmp_path):
    path = tmp_path / "cross-fold.json"
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    a, b = np.arange(6), np.arange(6, 12)
    evaluation.calibration_search(
        DecisionTreeClassifier(random_state=3), {}, y[:, None], y, cv=2, folds=[(b, a), (a, b)],
        scoring="balanced_accuracy", refit=False, checkpoint_path=path,
    )
    genuine = evaluation.load_calibration_report(path)
    assert genuine["identity"]["validation_class_counts"] == [[2, 4], [4, 2]]
    for version, scores, possible in ((2, (1/8, 1/6), False), (1, (1/8, 1/6), False),
                                      (2, (1/6, 1.), False), (1, (1/6, 1.), True)):
        report = deepcopy(genuine)
        if version == 1:
            report["identity"]["fold_evidence_version"] = 1
            del report["identity"]["balanced_accuracy_support_method"]
            for fold in report["candidates"][0]["folds"]:
                del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
                del fold["balanced_accuracy_custom_scalar_inputs"]
            del report["identity"]["validation_class_counts"]
            report["fingerprint"] = evaluation._digest(report["identity"])
        row, = report["candidates"]
        for fold, score in zip(row["folds"], scores):
            fold.update(score=score, balanced_accuracy=score)
        mean = float(np.mean(scores))
        row.update(mean_score=mean, mean_balanced_accuracy=mean, fold_score_std=float(np.std(scores, ddof=1)))
        report["selected_mean_score"] = mean
        write(path, report)
        if possible:
            assert evaluation.load_calibration_report(path) == report
        else:
            with pytest.raises(ValueError, match="Balanced accuracy"):
                evaluation.load_calibration_report(path)
    del genuine["identity"]["validation_class_counts"]
    genuine["fingerprint"] = evaluation._digest(genuine["identity"])
    write(path, genuine)
    with pytest.raises(ValueError):
        evaluation.load_calibration_report(path)


def test_archived_balanced_score_uses_all_folds_and_exact_small_sample_rounding(tmp_path):
    path = tmp_path / "balanced.json"
    y = np.tile([0, 1], 5)
    a, b = np.arange(4), np.arange(4, 10)
    evaluation.calibration_search(
        DecisionTreeClassifier(random_state=3), {}, y[:, None], y, cv=2, folds=[(b, a), (a, b)],
        scoring="balanced_accuracy", refit=False, checkpoint_path=path,
    )
    genuine = evaluation.load_calibration_report(path)
    assert _validation_support_bounds([4, 6], [5, 5]) == [(2, 2), (3, 3)]
    for impossible in (1/6, float(np.nextafter(.25, 1))):
        report = deepcopy(genuine)
        row, = report["candidates"]
        row["folds"][0].update(score=impossible, balanced_accuracy=impossible)
        scores = [f["score"] for f in row["folds"]]
        mean = float(np.mean(scores))
        row.update(mean_score=mean, mean_balanced_accuracy=mean, fold_score_std=float(np.std(scores, ddof=1)))
        report["selected_mean_score"] = mean
        write(path, report)
        with pytest.raises(ValueError, match="Balanced accuracy"):
            evaluation.load_calibration_report(path)


class MixedLabelClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, container="array"):
        self.container = container

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        values = np.array([0, 0, 1, 1, 0, 2**63 + 1], dtype=np.uint64)[X[:, 0].astype(int)]
        if self.container == "column":
            return values[:, None]
        if self.container == "list":
            return list(values)
        return values


@pytest.mark.parametrize("container", ["array", "column", "list"])
def test_balanced_accuracy_with_rows_excluded_by_sklearn(tmp_path, container):
    path = tmp_path / "mixed-labels.json"
    y = np.tile(np.array([0, 0, 1, 1, 1, 1], dtype=np.int64), 2)
    X = np.tile(np.arange(6), 2)[:, None]
    a, b = np.arange(6), np.arange(6, 12)
    evaluation.calibration_search(MixedLabelClassifier(container), {}, X, y, cv=2, folds=[(b, a), (a, b)],
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    row, = report["candidates"]
    for fold in row["folds"]:
        assert fold["status"] == "valid" and fold["score"] == (1 + 2/3) / 2
        witness = fold["balanced_accuracy_support"]
        assert witness["class_counts"] == [2, 3]
        assert fold["balanced_accuracy_hits"] == [2, 2]
        assert fold["warnings"][witness["warning_index"]]["message"] == "y_pred contains classes not in y_true"

    changed = deepcopy(report)
    del changed["candidates"][0]["folds"][0]["balanced_accuracy_support"]
    write(path, changed)
    with pytest.raises(ValueError, match="Balanced accuracy"):
        evaluation.load_calibration_report(path)
    changed = deepcopy(report)
    changed["candidates"][0]["folds"][0]["convergence_check"]["warnings_checked"] = 0
    write(path, changed)
    with pytest.raises(ValueError, match="Balanced accuracy"):
        evaluation.load_calibration_report(path)
    report["identity"]["fold_evidence_version"] = 2
    del report["identity"]["balanced_accuracy_support_method"]
    report["fingerprint"] = evaluation._digest(report["identity"])
    for fold in row["folds"]:
        del fold["balanced_accuracy_support"], fold["balanced_accuracy_hits"], fold["balanced_accuracy_target_cache"]
        del fold["balanced_accuracy_prediction_kind"], fold["balanced_accuracy_custom_scalar_inputs"]
    write(path, report)
    assert evaluation.load_calibration_report(path) == report
    assert not _legacy_extra_label_score(.75, 1, 1)


def test_inconsistent_confusion_replay_preserves_live_score_and_refuses_archive(tmp_path, monkeypatch):
    ordinary_matrix = evaluation.confusion_matrix

    def replay_with_one_changed_hit(*args, **kwargs):
        matrix = ordinary_matrix(*args, **kwargs)
        matrix[0, 0] -= 1
        matrix[0, 1] += 1
        return matrix

    monkeypatch.setattr(evaluation, "confusion_matrix", replay_with_one_changed_hit)
    y = np.tile(np.array([0, 0, 1, 1, 1, 1], dtype=np.int64), 2)
    x = np.tile(np.arange(6), 2)[:, None]
    a, b = np.arange(6), np.arange(6, 12)
    path = tmp_path / "inconsistent-confusion.json"
    result = evaluation.calibration_search(MixedLabelClassifier("array"), {}, x, y, cv=2, folds=[(b, a), (a, b)],
                                           scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    for fold in result.report["candidates"][0]["folds"]:
        assert fold["status"] == "valid" and fold["score"] == (1 + 2 / 3) / 2
        assert fold["balanced_accuracy_support"]["class_counts"] == [2, 3]
        assert fold["balanced_accuracy_hits"] == [1, 2]
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="disagrees with its recorded hit counts"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def test_unrelated_warning_does_not_qualify_as_scorer_evidence():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("y_pred contains classes not in y_true", UserWarning)
        fold = {}
        assert evaluation._score_calibration_fold("balanced_accuracy", y, y.astype(np.float64),
                                                  np.unique(y), caught, fold) == 1
        warnings.warn("y_pred contains classes not in y_true", UserWarning)
    assert "balanced_accuracy_support" not in fold


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    path = tmp_path_factory.mktemp("convergence") / "cv.json"
    y = np.array([0] * 6 + [1] * 2 + [0] * 5 + [1] * 3)
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(
        LogisticRegression(random_state=11, solver="lbfgs", tol=1e-9),
        {"max_iter": [1, 100]}, np.ones((len(y), 1)), y,
        cv=2, folds=[(a, b), (b, a)], refit=False, checkpoint_path=path,
    )
    current = evaluation.load_calibration_report(path)
    return {"versioned": current, "legacy": _omit_later_fold_evidence(current)}


def _omit_later_fold_evidence(current):
    legacy = deepcopy(current)
    legacy["identity"].pop("fold_evidence_version")
    legacy["identity"].pop("validation_class_counts")
    legacy["identity"].pop("class_label_evidence")
    legacy["fingerprint"] = evaluation._digest(legacy["identity"])
    for row in legacy["candidates"]:
        for fold in row["folds"]:
            fold.pop("convergence_check")
            for item in fold["warnings"]:
                item.pop("is_convergence_warning")
    return legacy


def write(path, report):
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report), encoding="utf-8")


def test_archive_sha256_covers_exact_bytes_and_rejects_invalid_digests(reports, tmp_path):
    path = tmp_path / "archive.json"
    report = reports["versioned"]
    write(path, report)
    original = path.read_bytes()
    digest = sha256(original).hexdigest()
    assert evaluation.load_calibration_report(path, expected_sha256=digest.upper()) == report
    for malformed in (False, digest.encode(), "z" * 64, digest[:-1], digest + " "):
        with pytest.raises(ValueError, match="64-character hexadecimal"):
            evaluation.load_calibration_report(path, expected_sha256=malformed)
    for other in ("0" * 64, report["checkpoint_sha256"], report["fingerprint"]):
        with pytest.raises(ValueError, match="whole-file SHA-256"):
            evaluation.load_calibration_report(path, expected_sha256=other)
    path.write_bytes(b"\n" + original + b"\n")
    assert evaluation.load_calibration_report(path) == report
    with pytest.raises(ValueError, match="whole-file SHA-256"):
        evaluation.load_calibration_report(path, expected_sha256=digest)


def test_reader_parses_the_same_bytes_it_hashed(reports, tmp_path, monkeypatch):
    path = tmp_path / "replaced.json"
    report = reports["versioned"]
    write(path, report)
    digest = sha256(path.read_bytes()).hexdigest()
    read_bytes = Path.read_bytes
    reads = []

    def read_then_replace(file):
        data = read_bytes(file)
        if file == path:
            reads.append(file)
            path.write_bytes(b"{}")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
    assert reads == [path] and read_bytes(path) == b"{}"


@pytest.mark.parametrize("status", ["valid", "nonconverged"])
def test_legacy_warning_in_a_lower_scoring_candidate_keeps_winner(reports, tmp_path, status):
    report = deepcopy(reports["legacy"])
    row, winner = report["candidates"]
    for fold in row["folds"]:
        assert any(w["category"] == "ConvergenceWarning" for w in fold["warnings"])
        fold.update(score=0., accuracy=0., status=status)
    if status == "valid":
        _summarize_promoted_candidate(report, row)
    assert report["selected_candidate_index"] == winner["index"]
    path = tmp_path / "losing-row.json"
    write(path, report)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert evaluation.load_calibration_report(path) == report


def test_strict_legacy_warning_checks_scores_and_original_tie_order(reports, tmp_path):
    report = deepcopy(reports["legacy"])
    short, winner = report["candidates"]
    path = tmp_path / "tied-challenger.json"
    for fold, winning_fold in zip(short["folds"], winner["folds"]):
        fold["score"] = fold["accuracy"] = winning_fold["score"]
        # A convergence warning subclass need not be named ConvergenceWarning.
        for item in fold["warnings"]:
            item["category"] = "ModelNotice"
    write(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="challenger 0 could win"):
            evaluation.load_calibration_report(path, expected_sha256=digest)

    # Changing the grid order changes which candidate wins an equal-score tie.
    config = dict((k["value"], v) for k, v in report["identity"]["structural_configuration"]["items"])
    config["candidates"]["items"].reverse()
    report["candidates"] = [winner, short]
    winner["index"], short["index"] = 0, 1
    for i, row in enumerate(report["screening_diagnostics"]["candidates"]):
        row["index"] = i
    report["selected_candidate_index"] = 0
    report["fingerprint"] = evaluation._digest(report["identity"])
    write(path, report)
    assert evaluation.load_calibration_report(path) == report

    for fold in short["folds"]:
        fold["score"] = fold["accuracy"] = 1.
    write(path, report)
    with pytest.raises(ValueError, match="challenger 1 could win"):
        evaluation.load_calibration_report(path)

    # An exception disqualifies the challenger even when another fold's
    # warning could have been an unrelated class with the same name.
    short["folds"][0].update(status="failed", error_type="ValueError", error="fit failed")
    for field in ("accuracy", "score"):
        del short["folds"][0][field]
    write(path, report)
    assert evaluation.load_calibration_report(path) == report


class CustomTrackedHead(evaluation._TrackedLinearSVC):
    pass


@pytest.mark.parametrize("form", ["direct", "pipeline", "replaced", "custom"])
def test_legacy_tracked_fit_disqualifies_a_candidate_independently(tmp_path, form):
    path = tmp_path / "tracked.json"
    y = np.array([0] * 6 + [1] * 2 + [0] * 5 + [1] * 3)
    a, b = np.arange(8), np.arange(8, 16)
    head = evaluation._TrackedLinearSVC(dual=False, max_iter=1, tol=1e-12, random_state=7)
    grid = {}
    if form == "direct":
        estimator = evaluation._TrackedLogisticRegression(max_iter=1, random_state=7)
    elif form == "replaced":
        estimator = Pipeline([("head", LogisticRegression())])
        grid = {"head": [head]}
    else:
        if form == "custom":
            head = CustomTrackedHead(**head.get_params(deep=False))
        estimator = Pipeline([("head", head)])
    evaluation.calibration_search(estimator, grid, np.ones((len(y), 1)), y, cv=2,
                                  folds=[(a, b), (b, a)], refit=False, checkpoint_path=path)
    report = _omit_later_fold_evidence(json.loads(path.read_bytes()))
    row, = report["candidates"]
    assert report["selected_candidate_index"] is None
    prefix = "" if form == "direct" else "classifier_"
    for fold in row["folds"]:
        assert fold["warnings"] and fold["diagnostics"][prefix + "converged"] is False
    write(path, report)
    if form == "custom":
        with pytest.raises(ValueError, match="no-winner"):
            evaluation.load_calibration_report(path)
        return
    assert evaluation.load_calibration_report(path) == report

    altered = deepcopy(report)
    altered["candidates"][0]["folds"][0]["diagnostics"][prefix + "convergence_warnings"] = []
    write(path, altered)
    with pytest.raises(ValueError, match="convergence disagrees"):
        evaluation.load_calibration_report(path)
    for fold in row["folds"]:
        fold["status"] = "valid"
    _summarize_promoted_candidate(report, row)
    write(path, report)
    with pytest.raises(ValueError, match="tracked classifier convergence"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("generation", ["current", "legacy"])
def test_clean_direct_tracked_fit_cannot_lose_its_only_winner(tmp_path, generation):
    path = tmp_path / "tracked-winner.json"
    y = np.tile([0, 1], 12)
    model = evaluation._TrackedLogisticRegression(random_state=7)
    evaluation.calibration_search(model, {}, y[:, None], y, cv=2, seed=3, refit=False, checkpoint_path=path)
    report = json.loads(path.read_bytes())
    if generation == "legacy":
        report = _omit_later_fold_evidence(report)
    write(path, report)
    assert evaluation.load_calibration_report(path)["selected_candidate_index"] == 0
    row, = report["candidates"]
    for fold in row["folds"]:
        assert fold["warnings"] == [] and fold["diagnostics"]["converged"] is True
        fold["status"] = "nonconverged"
        if generation == "current":
            fold["convergence_check"]["reason"] = "estimator"
    row.update(valid=False, invalid_fold_count=2, mean_score=None, mean_accuracy=None, fold_score_std=None)
    report.update(selected_candidate_index=None, selected_mean_score=None, selected_parameters=None,
                  refinement=evaluation.refinement_diagnostics([{}], None))
    write(path, report)
    message = "convergence flag checks" if generation == "current" else "tracked classifier convergence"
    with pytest.raises(ValueError, match=message):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("pipeline", [False, True])
def test_failed_tracked_head_cannot_be_promoted_by_reclassifying_its_warning(tmp_path, pipeline):
    y = np.array([0] * 6 + [1] * 2 + [0] * 5 + [1] * 3)
    a, b = np.arange(8), np.arange(8, 16)
    model = evaluation._TrackedLinearSVC(dual=False, max_iter=1, tol=1e-12, random_state=7)
    if pipeline:
        model = Pipeline([("head", model)])
    path = tmp_path / "tracked.json"
    evaluation.calibration_search(model, {}, np.ones((len(y), 1)), y, cv=2, folds=[(a, b), (b, a)],
                                  refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_candidate_index"] is None
    row, = report["candidates"]
    for fold in row["folds"]:
        assert fold["diagnostics"]["classifier_converged" if pipeline else "converged"] is False
        fold["status"] = "valid"
        fold["convergence_check"]["reason"] = None
        for w in fold["warnings"]:
            w["is_convergence_warning"] = False
    _summarize_promoted_candidate(report, row)
    write(path, report)
    with pytest.raises(ValueError, match="tracked classifier convergence"):
        evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("form", ["tracked", "pipeline", "stock"])
@pytest.mark.parametrize("flag", [False, 0, None, np.nan], ids=["false", "zero", "none", "nan"])
def test_clean_tracked_or_stock_head_preserves_callback_decision(tmp_path, form, flag):
    if not hasattr(LogisticRegression, "set_callbacks"):
        pytest.skip("This control uses sklearn fit callbacks.")

    class MarkFit:
        def setup(self, estimator, context):
            estimator.classifier_converged_ = flag

        def teardown(self, estimator, context):
            pass

        def on_fit_task_begin(self, estimator, context):
            pass

        def on_fit_task_end(self, estimator, context):
            return False

    model = (LogisticRegression if form == "stock" else evaluation._TrackedLogisticRegression)(random_state=7)
    if form == "pipeline":
        model = Pipeline([("head", model)])
    model.set_callbacks(MarkFit())
    y = np.tile([0, 1], 12)
    path = tmp_path / "callback.json"
    evaluation.calibration_search(model, {}, y[:, None], y, cv=2, seed=3, refit=False, checkpoint_path=path)
    current = json.loads(path.read_bytes())
    assert current["selected_candidate_index"] == (0 if bool(flag) else None)
    for fold in current["candidates"][0]["folds"]:
        assert fold["convergence_check"]["reason"] == (None if bool(flag) else "estimator")
        assert fold["convergence_check"]["warnings_checked"] == 0
        expected_flag = True if form == "pipeline" else evaluation._jsonable(flag)
        assert json.dumps(fold["diagnostics"]["classifier_converged"]) == json.dumps(expected_flag)
    for report in (current, _omit_later_fold_evidence(current)):
        write(path, report)
        assert evaluation.load_calibration_report(path) == report


@pytest.mark.parametrize("display", [0, False])
def test_legacy_callback_flag_display_cannot_disqualify_tied_challenger(reports, tmp_path, display):
    report = deepcopy(reports["legacy"])
    challenger, winner = report["candidates"]
    for fold, winning_fold in zip(challenger["folds"], winner["folds"]):
        assert fold["warnings"]
        fold["score"] = fold["accuracy"] = winning_fold["score"]
        fold["diagnostics"]["classifier_converged"] = display
    path = tmp_path / "flag.json"
    write(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="challenger 0 could win"):
            evaluation.load_calibration_report(path, expected_sha256=digest)

    for fold in challenger["folds"]:
        fold["status"] = "valid"
    _summarize_promoted_candidate(report, challenger)
    write(path, report)
    with pytest.raises(ValueError, match="eligibility of selected candidate 0"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize(("generation", "changes"), [("versioned", "status"), ("legacy", "status"),
                                                    ("versioned", "boundary"), ("versioned", "membership")])
def test_promoted_iteration_limit_is_rejected(reports, tmp_path, generation, changes):
    report = deepcopy(reports[generation])
    path = tmp_path / "cv.json"
    write(path, report)
    if generation == "versioned":
        assert evaluation.load_calibration_report(path) == report
    else:
        with pytest.raises(ValueError, match="challenger 0 could win"):
            evaluation.load_calibration_report(path)
    row, qualified = report["candidates"]
    assert qualified["valid"] and not row["valid"]
    for fold in row["folds"]:
        assert any(w["category"] == "ConvergenceWarning" for w in fold["warnings"])
        fold["status"] = "valid"
        if changes != "status":
            fold["convergence_check"].update(warnings_checked=0, reason=None)
        if changes == "membership":
            for item in fold["warnings"]:
                item["is_convergence_warning"] = False
    scores = [f["score"] for f in row["folds"]]
    mean = float(np.mean(scores))
    row.update(valid=True, invalid_fold_count=0, mean_score=mean, mean_accuracy=mean,
               fold_score_std=float(np.std(scores, ddof=1)))
    report.update(selected_candidate_index=0, selected_parameters=row["parameters"], selected_mean_score=mean,
                  refinement=evaluation.refinement_diagnostics([r["parameters"] for r in report["candidates"]],
                                                               row["parameters"]))
    write(path, report)
    with pytest.raises(ValueError, match="convergence"):
        evaluation.load_calibration_report(path)


def test_writer_preserves_warning_identity_and_decision_boundary():
    class ConvergenceWarning(UserWarning):
        pass

    class Model:
        @property
        def converged_(self):
            warnings.warn("after warning inspection", SklearnConvergenceWarning)
            return True

    fold = {"status": "valid"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("unrelated warning with the same name", ConvergenceWarning)
        assert not evaluation._nonconverged(Model(), caught, fold_record=fold)
    fold["warnings"] = evaluation._warning_records(caught, classify_convergence=True)
    assert fold["convergence_check"]["warnings_checked"] == 1
    assert fold["convergence_check"]["reason"] is None
    assert [w["is_convergence_warning"] for w in fold["warnings"]] == [False, True]
    _check_fold_convergence(fold, 1, None)
    fold["convergence_check"]["warnings_checked"] = 2
    with pytest.raises(ValueError, match="convergence"):
        _check_fold_convergence(fold, 1, None)


def test_missing_versioned_convergence_evidence_is_rejected(reports, tmp_path):
    report = deepcopy(reports["versioned"])
    del report["candidates"][1]["folds"][0]["convergence_check"]
    path = tmp_path / "missing.json"
    write(path, report)
    with pytest.raises(ValueError):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("change", ["missing", "incomplete", "unvisited", "type", "decision", "display", "child"])
def test_live_flag_evidence_requires_the_actual_decision_path(reports, tmp_path, change):
    report = deepcopy(reports["versioned"])
    warned = report["candidates"][0]["folds"][0]
    clean = report["candidates"][1]["folds"][0]
    flags = clean["convergence_check"]["flag_checks"]
    if change == "missing":
        del clean["convergence_check"]["flag_checks"]
    elif change == "incomplete":
        del flags["classifier_converged"]
    elif change == "unvisited":
        warned["convergence_check"]["flag_checks"]["converged"] = flags["converged"]
    elif change == "type":
        flags["converged"]["passed"] = 1
    elif change == "decision":
        flags["last_transform_converged"]["passed"] = False
    elif change == "display":
        clean["diagnostics"]["converged"] = []
    else:
        flags["final_classifier_converged"]["exact_scalar"] = False
    path = tmp_path / "flag-evidence.json"
    write(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="convergence"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def test_version_five_stock_callback_can_emit_warnings_after_classification(reports, tmp_path):
    report = deepcopy(reports["versioned"])
    report["identity"]["fold_evidence_version"] = 5
    report["fingerprint"] = evaluation._digest(report["identity"])
    del report["refinement_axis_inputs"]
    for row in report["candidates"]:
        for fold in row["folds"]:
            del fold["convergence_check"]["flag_checks"]
    clean = report["candidates"][1]["folds"][0]
    clean["warnings"].append(dict(category="UserWarning", message="truth checked", is_convergence_warning=False))
    clean["diagnostics"]["classifier_converged"] = 0
    path = tmp_path / "before-observed-flags.json"
    write(path, report)
    assert evaluation.load_calibration_report(path) == report
    # Without a callback field, stock logistic's default flags cannot emit the warning.
    del clean["diagnostics"]["classifier_converged"]
    write(path, report)
    with pytest.raises(ValueError, match="warning boundary"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("generation", ["versioned", "legacy"])
@pytest.mark.parametrize("count", [0, 2, -1, True, None])
def test_stock_lbfgs_winner_requires_consistent_iteration_evidence(reports, tmp_path, generation, count):
    report = deepcopy(reports[generation])
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("These iteration diagnostics follow the scikit-learn 1.9.0 and 1.9.1 implementation.")
    row = report["candidates"][0]
    assert report["selected_candidate_index"] == 1 and row["parameters"]["max_iter"] == 1
    for fold in row["folds"]:
        assert fold["diagnostics"]["n_iter"] == [1]
        fold["status"] = "valid"
        if count is None:
            del fold["diagnostics"]["n_iter"]
        else:
            fold["diagnostics"]["n_iter"] = [count]
        if generation == "versioned":
            fold["convergence_check"]["reason"] = None
            # Reclassifying the warning as unrelated means the writer went
            # on to evaluate all remaining (default-true) fit predicates.
            fold["convergence_check"]["flag_checks"] = deepcopy(
                report["candidates"][1]["folds"][0]["convergence_check"]["flag_checks"])
            for warning in fold["warnings"]:
                warning["is_convergence_warning"] = False
    _summarize_promoted_candidate(report, row)
    assert report["selected_candidate_index"] == 0
    path = tmp_path / "impossible-lbfgs-winner.json"
    write(path, report)
    if generation == "versioned" and count == 0:
        # An in-range zero and explicit unrelated warning ancestry can be
        # genuine even when a callback copies the complete solver message.
        assert evaluation.load_calibration_report(path) == report
        return
    with pytest.raises(ValueError, match="warning ancestry" if generation == "legacy" and count == 0 else "L-BFGS"):
        evaluation.load_calibration_report(path)
    if count == 2:
        with pytest.raises(ValueError, match="L-BFGS iteration count exceeds"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("copy_warning", [False, True])
@pytest.mark.parametrize("outer_solver", ["lbfgs", "liblinear"])
@pytest.mark.parametrize("inner_cap", [1, False, True, 1.0, 1.5, -1.5])
def test_lbfgs_nested_fit_warning_ancestry_and_outer_count(tmp_path, copy_warning, outer_solver, inner_cap):
    if not hasattr(LogisticRegression, "set_callbacks"):
        pytest.skip("This control exercises sklearn's estimator callback API.")
    labels = np.tile([0, 1], 12)
    inner_features = (2 * labels - 1)[:, None]

    class ConvergenceWarning(UserWarning):
        pass

    class NestedFit:
        def setup(self, estimator, context):
            inner = LogisticRegression(max_iter=inner_cap, random_state=7)
            if copy_warning:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    inner.fit(inner_features, labels)
                message, = [str(w.message) for w in caught if w.category is SklearnConvergenceWarning]
                warnings.warn(message, ConvergenceWarning)
            else:
                inner.fit(inner_features, labels)
            assert inner.n_iter_.tolist() == [int(inner_cap)]

        def teardown(self, estimator, context):
            pass

        def on_fit_task_begin(self, estimator, context):
            pass

        def on_fit_task_end(self, estimator, context):
            return False

    model = LogisticRegression(max_iter=100, random_state=5).set_callbacks(NestedFit())
    path = tmp_path / "nested.json"
    with config_context(skip_parameter_validation=type(inner_cap) is float):
        evaluation.calibration_search(model, {"solver": [outer_solver]}, np.zeros((len(labels), 1)), labels, cv=2,
                                      refit=False, checkpoint_path=path)
    report = json.loads(path.read_bytes())
    assert evaluation.load_calibration_report(path) == report
    row, = report["candidates"]
    assert row["valid"] is copy_warning
    assert report["selected_candidate_index"] == (0 if copy_warning else None)
    if not copy_warning:
        changed = deepcopy(report)
        promoted, = changed["candidates"]
        for fold in promoted["folds"]:
            fold["status"] = "valid"
        _summarize_promoted_candidate(changed, promoted)
        write(path, changed)
        with pytest.raises(ValueError, match="convergence"):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())
    for fold in row["folds"]:
        warning, = [w for w in fold["warnings"] if w["category"] == "ConvergenceWarning"]
        assert warning["is_convergence_warning"] is not copy_warning
        if outer_solver == "lbfgs":
            assert fold["diagnostics"]["n_iter"] == [0]
        assert fold["status"] == ("valid" if copy_warning else "nonconverged")
        del fold["convergence_check"]
        for w in fold["warnings"]:
            del w["is_convergence_warning"]
    for key in ("fold_evidence_version", "validation_class_counts", "class_label_evidence"):
        del report["identity"][key]
    report["fingerprint"] = evaluation._digest(report["identity"])
    write(path, report)
    issue = "eligibility of selected candidate 0" if copy_warning else "no-winner result"
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match=issue):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def test_stock_lbfgs_zero_iterations_and_warning_free_cap(tmp_path):
    path = tmp_path / "zero-lbfgs.json"
    labels = np.tile([0, 1], 8)
    model = LogisticRegression(max_iter=np.int64(0), random_state=7)
    evaluation.calibration_search(model, {}, np.ones((len(labels), 1)), labels, cv=2,
                                  refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("These iteration diagnostics follow the scikit-learn 1.9.0 and 1.9.1 implementation.")
    row, = report["candidates"]
    assert row["valid"] and all(fold["diagnostics"]["n_iter"] == [0] for fold in row["folds"])
    fold = row["folds"][0]
    # An unqualified class name alone does not identify a stock solver warning.
    fold["warnings"].append({"category": "ConvergenceWarning",
                             "message": "lbfgs failed to converge after 1 iteration(s) (status=1):\nunrelated",
                             "is_convergence_warning": False})
    fold["convergence_check"]["warnings_checked"] += 1
    write(path, report)
    assert evaluation.load_calibration_report(path) == report
    fold["diagnostics"]["n_iter"] = [1]
    write(path, report)
    with pytest.raises(ValueError, match="exceeds its declared limit"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("cap,in_grid", [(1.0, False), (1.5, True), (-1.5, False), (-0.5, True),
                                        (np.float32(1.5), False), (np.float16(1.5), True)])
def test_unvalidated_lbfgs_float_cap_clips_then_truncates(tmp_path, cap, in_grid):
    path = tmp_path / "float-cap.json"
    labels = np.tile([0, 1], 12)
    features = (2 * labels - 1)[:, None]
    model = LogisticRegression(max_iter=100 if in_grid else cap, random_state=7)
    with config_context(skip_parameter_validation=True):
        evaluation.calibration_search(model, {"max_iter": [cap]} if in_grid else {}, features, labels,
                                      cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("The archived L-BFGS storage rules follow the scikit-learn 1.9.0 and 1.9.1 implementation.")
    row, = report["candidates"]
    assert all(f["status"] == "nonconverged" and f["diagnostics"]["n_iter"] == [int(cap)] for f in row["folds"])
    for impossible in (int(cap) + 1, min(-1, int(cap) - 1)):
        row["folds"][0]["diagnostics"]["n_iter"] = [impossible]
        write(path, report)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="L-BFGS iteration count"):
                evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("cap,impossible", [(100, 101), (2**31, 2**31), (2**31, -(2**31) - 1)])
def test_stock_lbfgs_bound_with_array_class_weights(tmp_path, cap, impossible):
    path = tmp_path / "array-weights.json"
    labels = np.tile([0, 1], 12)
    model = LogisticRegression(class_weight={0: np.array(1.), 1: np.array(1.)}, random_state=11)
    evaluation.calibration_search(model, {"max_iter": [cap]}, labels[:, None], labels, cv=2,
                                  refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("The archived solver iteration bound follows the scikit-learn 1.9.0 and 1.9.1 implementation.")
    assert report["selected_candidate_index"] == 0
    fold = report["candidates"][0]["folds"][0]
    assert 0 < fold["diagnostics"]["n_iter"][0] <= cap
    fold["diagnostics"]["n_iter"] = [impossible]
    write(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="L-BFGS iteration count exceeds"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def test_stationary_fit_with_uncertain_int64_constructor_cap(tmp_path):
    path = tmp_path / "stationary.json"
    y = np.tile([0, 1], 8)
    model = LogisticRegression(max_iter=np.int64(-(2**32) - 1), random_state=11)
    with config_context(skip_parameter_validation=True):
        evaluation.calibration_search(model, {}, np.zeros((len(y), 1)), y, cv=2, seed=3,
                                      refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    for fold in report["candidates"][0]["folds"]:
        if fold["status"] != "failed":
            assert fold["diagnostics"]["n_iter"] == [-1]


def test_unvalidated_lbfgs_integer_cap_below_int32(tmp_path):
    path = tmp_path / "integer-storage.json"
    labels = np.tile([0, 1], 12)
    cap = -(2**31) - 1
    with config_context(skip_parameter_validation=True):
        evaluation.calibration_search(LogisticRegression(random_state=7), {"max_iter": [cap, 100]},
                                      labels[:, None], labels, cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("This control follows the scikit-learn 1.9.0 and 1.9.1 integer storage path.")
    assert report["selected_candidate_index"] == 1
    for fold in report["candidates"][0]["folds"]:
        if fold["status"] == "failed":
            assert "score" not in fold and fold["error_type"] == "OverflowError"
            continue
        assert fold["status"] == "nonconverged" and fold["diagnostics"]["n_iter"] == [2**31 - 1]
        for impossible in (2**31 - 2, 0, -1):
            fold["diagnostics"]["n_iter"] = [impossible]
            write(path, report)
            for digest in (None, sha256(path.read_bytes()).hexdigest()):
                with pytest.raises(ValueError, match="L-BFGS iteration count"):
                    evaluation.load_calibration_report(path, expected_sha256=digest)
        fold["diagnostics"]["n_iter"] = [2**31 - 1]


def test_hashed_numpy_integer_cap_uses_only_a_matching_warning(tmp_path, reports):
    path = tmp_path / "hashed-integer.json"
    labels = np.tile([0, 1], 12)
    evaluation.calibration_search(LogisticRegression(max_iter=np.int64(700), random_state=7),
                                  {}, labels[:, None], labels, cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("This control follows the scikit-learn 1.9.0 and 1.9.1 optimizer warning format.")
    observed, sibling = report["candidates"][0]["folds"]
    source = reports["versioned"]["candidates"][0]["folds"][0]["warnings"]
    original = next(w["message"] for w in source if w["category"] == "ConvergenceWarning")
    warning = {"category": "ConvergenceWarning", "is_convergence_warning": False}
    observed["warnings"].append(warning)
    observed["convergence_check"]["warnings_checked"] += 1
    for printed, count in [(300, 400), (700, 701)]:
        warning["message"] = original.replace("after 1 iteration", f"after {printed} iteration").replace(
            "max_iter=1)", f"max_iter={printed})")
        sibling["diagnostics"]["n_iter"] = [count]
        write(path, report)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            if printed == 300:
                assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
            else:
                with pytest.raises(ValueError, match="L-BFGS iteration count exceeds"):
                    evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("solver,max_iter,reason", [("spa", 100, None), ("cd", 1, "warning")])
def test_completed_convergence_check_survives_a_failed_fold(tmp_path, solver, max_iter, reason):
    path = tmp_path / "postcheck.json"
    labels = np.tile([0, 1], 8)
    features = np.array([[1., .1], [.2, 1.]])[labels]
    model = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7,
                          solver=solver, max_iter=max_iter)
    delivered = []

    class Deadline(TimeoutError):
        def __str__(self):
            warnings.warn("while formatting the deadline", SklearnConvergenceWarning)
            return "fold deadline"

    def interrupt(frame, event, arg):
        if frame.f_code is evaluation._nonconverged.__code__ and not delivered:
            if event == "return":
                delivered.append(True)
                raise Deadline()
            return interrupt

    previous = sys.gettrace()
    try:
        sys.settrace(interrupt)
        evaluation.calibration_search(model, {}, features, labels, cv=2, refit=False, checkpoint_path=path)
    finally:
        sys.settrace(previous)
    original = json.loads(path.read_bytes())
    assert evaluation.load_calibration_report(path) == original
    row, = original["candidates"]
    fold = row["folds"][0]
    check = fold["convergence_check"]
    assert delivered and fold["status"] == "failed" and fold["score"] == fold["accuracy"]
    assert fold["error_type"] == "Deadline" and fold["error"] == "fold deadline"
    assert check["reason"] == reason and check["warnings_checked"] == len(fold["warnings"]) - 1
    assert fold["warnings"][-1]["is_convergence_warning"]
    assert not row["valid"] and original["selected_candidate_index"] is None
    for change in ("score", "diagnostics", "reason", "boundary"):
        altered = deepcopy(original)
        failed = altered["candidates"][0]["folds"][0]
        if change in ("score", "diagnostics"):
            del failed[change]
        elif change == "reason":
            failed["convergence_check"]["reason"] = "estimator"
        else:
            failed["convergence_check"]["warnings_checked"] = len(failed["warnings"]) + 1
        write(path, altered)
        with pytest.raises(ValueError, match="convergence"):
            evaluation.load_calibration_report(path)


@pytest.fixture(scope="module")
def nmf_reports(tmp_path_factory):
    path = tmp_path_factory.mktemp("nmf-convergence") / "cv.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    estimator = NMFClassifier(n_components=np.int64(2), n_neighbors=1, screening="none", random_state=7)
    grid = [{"solver": ["cd"], "max_iter": [1]}, {"solver": ["spa"], "max_iter": [100]}]
    evaluation.calibration_search(estimator, grid, X, y, cv=2, refit=False, checkpoint_path=path)
    current = evaluation.load_calibration_report(path)
    assert current["selected_candidate_index"] == 1
    assert all(f["diagnostics"]["factorization_converged"] is False for f in current["candidates"][0]["folds"])
    legacy = deepcopy(current)
    del legacy["identity"]["fold_evidence_version"], legacy["identity"]["validation_class_counts"]
    legacy["fingerprint"] = evaluation._digest(legacy["identity"])
    for row in legacy["candidates"]:
        for fold in row["folds"]:
            del fold["convergence_check"]
            for warning in fold["warnings"]:
                del warning["is_convergence_warning"]
    write(path, legacy)
    assert evaluation.load_calibration_report(path) == legacy
    return {"current": current, "legacy": legacy}


def _summarize_promoted_candidate(report, row):
    scores = [fold["score"] for fold in row["folds"]]
    row.update(valid=True, invalid_fold_count=0, mean_score=float(np.mean(scores)),
               mean_accuracy=float(np.mean(scores)), fold_score_std=float(np.std(scores, ddof=1)))
    winner = min((r for r in report["candidates"] if r["valid"]), key=lambda r: (-r["mean_score"], r["index"]))
    report.update(selected_candidate_index=winner["index"], selected_mean_score=winner["mean_score"],
                  selected_parameters=winner["parameters"],
                  refinement=evaluation.refinement_diagnostics([r["parameters"] for r in report["candidates"]],
                                                                winner["parameters"]))


@pytest.mark.parametrize("generation", ["current", "legacy"])
@pytest.mark.parametrize(("strip", "reported_count"), [
    ("none", 1), ("phases", 1), ("all", 1), ("all", 0), ("all", 2),
])
def test_promoted_nmf_fold_disagrees_with_actual_fit(nmf_reports, tmp_path, generation, strip, reported_count):
    report = deepcopy(nmf_reports[generation])
    if strip == "all" and report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("Iteration-limit evidence is specific to the archived scikit-learn 1.9.0 and 1.9.1 solver.")
    path = tmp_path / "promoted-nmf.json"
    row = report["candidates"][0]
    for fold in row["folds"]:
        fold["status"] = "valid"
        if strip != "none":
            diagnostics = fold["diagnostics"]
            assert any(w["message"] in diagnostics["factorization_convergence_warnings"] for w in fold["warnings"])
            diagnostics.update(factorization_convergence_warnings=[], factorization_converged=True,
                               last_transform_warnings=[], last_transform_converged=True,
                               convergence_warnings=[], converged=True)
            # CD/auto did not project the training coordinates.
            assert diagnostics["training_projection_performed"] is False
        if strip == "all":
            assert diagnostics["n_iter"] == diagnostics["decomposition"]["n_iter"] == row["parameters"]["max_iter"]
            diagnostics["n_iter"] = diagnostics["decomposition"]["n_iter"] = reported_count
            fold["warnings"].clear()
        if generation == "current":
            fold["convergence_check"].update(reason=None, warnings_checked=len(fold["warnings"]))
            for warning in fold["warnings"]:
                warning["is_convergence_warning"] = False
    _summarize_promoted_candidate(report, row)
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("generation", ["current", "legacy"])
@pytest.mark.parametrize(("status", "forwarded", "keep_score"), [
    ("nonconverged", False, True), ("nonconverged", True, True),
    ("failed", True, False), ("failed", True, True),
])
def test_fabricated_knn_head_failure_cannot_remove_only_winner(nmf_reports, tmp_path, generation,
                                                              status, forwarded, keep_score):
    report = deepcopy(nmf_reports[generation])
    winner = report["candidates"][1]
    assert winner["valid"] and not report["candidates"][0]["valid"]
    fold = winner["folds"][0]
    diagnostics = fold["diagnostics"]
    assert diagnostics["classifier_n_iter"] is None and not diagnostics["classifier_convergence_warnings"]
    message = "classifier reached its iteration limit"
    diagnostics.update(classifier_converged=False, classifier_convergence_warnings=[message],
                       converged=False, convergence_warnings=[message])
    fold["status"] = status
    if forwarded:
        warning = {"category": "ConvergenceWarning", "message": message}
        if generation == "current":
            warning["is_convergence_warning"] = True
        fold["warnings"].append(warning)
    if status == "failed":
        fold.update(error_type="RuntimeError", error="classification failed")
        fold.pop("convergence_check", None)
        if not keep_score:
            del fold["accuracy"], fold["score"]
    elif generation == "current":
        fold["convergence_check"].update(warnings_checked=len(fold["warnings"]),
                                         reason="warning" if forwarded else "estimator")
    winner.update(valid=False, invalid_fold_count=1, mean_score=None, mean_accuracy=None, fold_score_std=None)
    report.update(selected_candidate_index=None, selected_parameters=None, selected_mean_score=None,
                  refinement=evaluation.refinement_diagnostics([r["parameters"] for r in report["candidates"]], None))
    path = tmp_path / "lost-knn-winner.json"
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("solver", ["cd", "spa"])
def test_nmf_training_projection_must_agree_with_declared_solver(nmf_reports, tmp_path, solver):
    report = deepcopy(nmf_reports["current"])
    row = next(c for c in report["candidates"] if c["parameters"]["solver"] == solver)
    diagnostics = row["folds"][0]["diagnostics"]
    assert not diagnostics["training_projection_convergence_warnings"]
    performed = not diagnostics["training_projection_performed"]
    diagnostics.update(training_coordinates_mode="transform" if performed else "fit_transform",
                       training_projection_performed=performed,
                       training_projection_converged=True if performed else None)
    path = tmp_path / "inconsistent-nmf-training-mode.json"
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("change_both", [False, True])
def test_nmf_spa_iteration_count_matches_the_decomposition_and_rank(nmf_reports, tmp_path, change_both):
    report = deepcopy(nmf_reports["current"])
    diagnostics = report["candidates"][1]["folds"][0]["diagnostics"]
    assert diagnostics["n_iter"] == diagnostics["decomposition"]["n_iter"]
    diagnostics["n_iter"] += 1
    if change_both:
        diagnostics["decomposition"]["n_iter"] = diagnostics["n_iter"]
    path = tmp_path / "inconsistent-nmf-iterations.json"
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


def test_nmf_iteration_caps_for_cd_zero_tolerance_and_linear_svm(tmp_path):
    path = tmp_path / "nmf-caps.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    estimator = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7)
    grid = [{"classifier": ["linear_svm"], "svm_max_iter": [1]},
            {"solver": ["cd"], "max_iter": [1], "tol": [0.]}]
    evaluation.calibration_search(estimator, grid, X, y, cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("Iteration-limit evidence is specific to the archived scikit-learn 1.9.0 and 1.9.1 solver.")
    svm, cd = report["candidates"]
    assert not svm["valid"] and cd["valid"] and report["selected_candidate_index"] == cd["index"]
    for fold in cd["folds"]:
        assert fold["diagnostics"]["n_iter"] == 1 and fold["diagnostics"]["factorization_converged"]
    for fold in svm["folds"]:
        diagnostics = fold["diagnostics"]
        assert diagnostics["classifier_n_iter"] == 1 and diagnostics["classifier_convergence_warnings"]
        diagnostics.update(classifier_converged=True, classifier_convergence_warnings=[], converged=True,
                           convergence_warnings=[])
        fold.update(status="valid", warnings=[], convergence_check={"reason": None, "warnings_checked": 0})
    _summarize_promoted_candidate(report, svm)
    changed_path = tmp_path / "missing-svm-cap-warning.json"
    write(changed_path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(changed_path)


@pytest.mark.parametrize("integer", [int, np.int64])
def test_nmf_solver_iteration_ranges_with_declared_grid_limits(tmp_path, integer):
    path = tmp_path / "iteration-ranges.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    model = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7, max_iter=np.int64(1))
    grid = [
        {"solver": ["cd", "mu"], "max_iter": [integer(1)], "tol": [0.]},
        {"solver": ["mu"], "max_iter": [integer(21)], "tol": [0., 1e9]},
        {"solver": ["cd"], "tol": [0.]},
        {"solver": ["spa"], "max_iter": [100], "classifier": ["logistic"], "logistic_max_iter": [integer(1)]},
        {"solver": ["spa"], "max_iter": [100], "classifier": ["linear_svm"], "svm_max_iter": [integer(1)]},
    ]
    evaluation.calibration_search(model, grid, X, y, cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("Iteration ranges follow the archived scikit-learn 1.9.0 and 1.9.1 implementations.")
    for row in report["candidates"]:
        params = row["parameters"]
        diagnostics = row["folds"][0]["diagnostics"]
        if params["solver"] == "spa":
            changes = [2]
        elif params.get("max_iter", 1) == 1:
            # The constructor-only NumPy cap is also recoverable from its digest.
            assert diagnostics["n_iter"] == 1 and diagnostics["factorization_converged"]
            changes = [0, 2]
        elif params["tol"] == 0:
            assert diagnostics["n_iter"] == 21 and diagnostics["factorization_converged"]
            changes = [10]  # At zero tolerance even MU's tenth iteration cannot stop early.
        else:
            assert diagnostics["n_iter"] in (10, 20, 21)
            changes = [11]  # MU only checks convergence every ten iterations.
        for count in changes:
            changed = deepcopy(report)
            fold = changed["candidates"][row["index"]]["folds"][0]["diagnostics"]
            if params["solver"] != "spa":
                fold["n_iter"] = fold["decomposition"]["n_iter"] = count
            elif params["classifier"] == "logistic":
                fold["classifier_n_iter"] = [count]
            else:
                fold["classifier_n_iter"] = count
            changed_path = tmp_path / "inconsistent-limit.json"
            write(changed_path, changed)
            with pytest.raises(ValueError, match="NMF"):
                evaluation.load_calibration_report(changed_path)


def test_hashed_numpy_iteration_cap_retains_equality_and_dtype_bound():
    declaration = configuration_identity(np.int16(700))
    assert _at_iteration_cap(declaration, 700) is True
    # Beyond the bounded digest lookup, a different in-range count cannot
    # establish which side of the original cap it was on.
    assert _at_iteration_cap(declaration, 701) is False
    with pytest.raises(ValueError, match="dtype"):
        _at_iteration_cap(declaration, 32768)


@pytest.mark.parametrize("generation", ["current", "legacy"])
@pytest.mark.parametrize("change", ["order", "multiplicity"])
def test_forwarded_nmf_phase_warnings_preserve_order_and_multiplicity(nmf_reports, tmp_path, generation, change):
    report = deepcopy(nmf_reports[generation])
    fold = report["candidates"][0]["folds"][0]
    diagnostics = fold["diagnostics"]
    phase = diagnostics["factorization_convergence_warnings"]
    message = phase[0]
    outer = next(record for record in fold["warnings"] if record["message"] == message)
    if change == "order":
        # Each required message is present, but the second one appears only before the first.
        extra = "second factorization warning"
        phase.append(extra)
        fold["warnings"].insert(0, {**outer, "message": extra})
        if generation == "current":
            fold["convergence_check"]["warnings_checked"] = len(fold["warnings"])
    else:
        # A single forwarded occurrence cannot corroborate multiple diagnostic occurrences.
        occurrences = sum(record["message"] == message for record in fold["warnings"])
        phase.extend([message] * occurrences)
    diagnostics["convergence_warnings"] = (phase + diagnostics["training_projection_convergence_warnings"]
                                           + diagnostics["classifier_convergence_warnings"])
    path = tmp_path / "contradictory-warning-sequence.json"
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


@pytest.fixture(scope="module")
def nmf_head_report(tmp_path_factory):
    path = tmp_path_factory.mktemp("nmf-heads") / "cv.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    estimator = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7)
    grid = [{"classifier": ["knn"], "knn_normalization": ["sd"]},
            {"classifier": ["logistic"]}, {"classifier": ["linear_svm"]}]
    evaluation.calibration_search(estimator, grid, X, y, cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert all(candidate["valid"] for candidate in report["candidates"])
    return report


@pytest.mark.parametrize("head", ["knn", "logistic", "linear_svm"])
@pytest.mark.parametrize("change", ["head", "iterations"])
def test_nmf_head_and_iterations_agree_with_the_grid(nmf_head_report, tmp_path, head, change):
    report = deepcopy(nmf_head_report)
    rows = {row["parameters"]["classifier"]: row for row in report["candidates"]}
    diagnostics = rows[head]["folds"][0]["diagnostics"]
    if change == "head":
        # The replacement fields agree with one another and come from a real fitted head.
        other = rows["knn" if head == "logistic" else "logistic"]["folds"][0]["diagnostics"]
        for key in ("fitted_classifier_type", "classifier_n_iter", "knn_normalization"):
            diagnostics[key] = other[key]
    else:
        diagnostics["classifier_n_iter"] = {"knn": 0, "logistic": 1, "linear_svm": [1]}[head]
    path = tmp_path / "inconsistent-head.json"
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


def _unit_neighbor_weights(distances):
    return distances * 0 + 1


def _failed_neighbor_weights(distances):
    raise RuntimeError("neighbor weights failed after projection")


def test_nmf_failure_after_successful_query_projection(tmp_path):
    path = tmp_path / "knn-prediction-failed.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    estimator = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7)
    evaluation.calibration_search(estimator, [{"weights": [_failed_neighbor_weights]}, {}], X, y,
                                  cv=2, refit=False, checkpoint_path=path)
    original = json.loads(path.read_bytes())
    assert evaluation.load_calibration_report(path) == original
    assert original["selected_candidate_index"] == 1
    for fold in original["candidates"][0]["folds"]:
        assert fold["status"] == "failed" and "score" not in fold
        assert fold["error_type"] == "RuntimeError" and fold["error"] == "neighbor weights failed after projection"
        diagnostics = fold["diagnostics"]
        assert diagnostics["converged"] and diagnostics["last_transform_performed"]
        assert diagnostics["last_transform_converged"] and diagnostics["last_transform_error"] is None


def test_failed_nmf_fit_and_failed_query_preserve_distinct_evidence(tmp_path):
    path = tmp_path / "failed-nmf.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    X[0, 0] = -1.
    a, b = np.arange(8), np.arange(8, 16)
    estimator = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7)
    evaluation.calibration_search(estimator, {}, X, y, cv=2, folds=[(a, b), (b, a)],
                                  refit=False, checkpoint_path=path)
    original = json.loads(path.read_bytes())
    assert evaluation.load_calibration_report(path) == original
    during_fit, during_query = original["candidates"][0]["folds"]
    assert all(f["status"] == "failed" and "score" not in f for f in (during_fit, during_query))
    assert during_fit["diagnostics"]["last_transform_performed"] is False
    assert "fitted_classifier_type" not in during_fit["diagnostics"]
    diagnostics = during_query["diagnostics"]
    assert diagnostics["converged"] and diagnostics["last_transform_performed"]
    assert diagnostics["last_transform_converged"] is False
    assert diagnostics["last_transform_error"] == during_query["error_type"] + ": " + during_query["error"]

    for fields in ({"last_transform_performed": False}, {"last_transform_converged": True}):
        altered = deepcopy(original)
        altered["candidates"][0]["folds"][1]["diagnostics"].update(fields)
        changed_path = tmp_path / "changed-failed-query.json"
        write(changed_path, altered)
        with pytest.raises(ValueError, match="NMF"):
            evaluation.load_calibration_report(changed_path)


def test_nmf_knn_normalization_with_custom_neighbor_weights(tmp_path):
    path = tmp_path / "custom-weights.json"
    y = np.tile([0, 1], 8)
    X = np.array([[1., .1], [.2, 1.]])[y]
    estimator = NMFClassifier(n_components=2, n_neighbors=1, screening="none", random_state=7,
                              weights=_unit_neighbor_weights, knn_normalization="sd")
    evaluation.calibration_search(estimator, {"classifier": ["knn", "logistic"]}, X, y,
                                  cv=2, refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert all(row["valid"] for row in report["candidates"])
    for row in report["candidates"]:
        modified = deepcopy(report)
        diagnostics = modified["candidates"][row["index"]]["folds"][0]["diagnostics"]
        if row["parameters"]["classifier"] == "knn":
            assert diagnostics["knn_normalization"] == "sd"
            diagnostics["knn_normalization"] = None
        else:
            assert diagnostics["knn_normalization"] is None
            diagnostics["knn_normalization"] = "sd"
        write(path, modified)
        with pytest.raises(ValueError, match="NMF"):
            evaluation.load_calibration_report(path)


@pytest.mark.parametrize("change", ["phase", "summary", "projection", "status"])
def test_nmf_completed_fold_uses_its_fit_and_query_diagnostics(nmf_reports, tmp_path, change):
    report = deepcopy(nmf_reports["current"])
    fold = report["candidates"][1]["folds"][0]
    diagnostics = fold["diagnostics"]
    if change == "phase":
        diagnostics["classifier_converged"] = False
    elif change == "summary":
        diagnostics["converged"] = False
    elif change == "projection":
        diagnostics.update(last_transform_converged=False, last_transform_warnings=["iteration limit"])
    else:
        fold["status"] = "nonconverged"
        fold["convergence_check"]["reason"] = "estimator"
    path = tmp_path / "changed-nmf.json"
    write(path, report)
    with pytest.raises(ValueError, match="NMF"):
        evaluation.load_calibration_report(path)


def test_stock_logistic_cannot_be_disqualified_by_an_estimator_flag(reports, tmp_path):
    report = deepcopy(reports["versioned"])
    fold = report["candidates"][1]["folds"][0]
    fold["status"] = "nonconverged"
    fold["convergence_check"]["reason"] = "estimator"
    path = tmp_path / "demoted-logistic.json"
    write(path, report)
    with pytest.raises(ValueError, match="Stock logistic fold"):
        evaluation.load_calibration_report(path)


@pytest.mark.parametrize("short_cap", [np.int64(1), True, False])
def test_legacy_iteration_limit_with_class_weights_and_numpy_settings(tmp_path, short_cap):
    path = tmp_path / "weighted-lbfgs.json"
    y = np.array([0] * 6 + [1] * 2 + [0] * 5 + [1] * 3)
    a, b = np.arange(8), np.arange(8, 16)
    model = LogisticRegression(solver=np.str_("lbfgs"), random_state=np.random.RandomState(11), class_weight={0: 1, 1: 1},
                               fit_intercept=np.bool_(True), tol=np.float64(1e-9))
    evaluation.calibration_search(model, {"max_iter": [short_cap, np.int64(100)]}, np.ones((len(y), 1)), y,
                                  cv=2, folds=[(a, b), (b, a)], refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    for key in ("fold_evidence_version", "validation_class_counts", "class_label_evidence"):
        del report["identity"][key]
    report["fingerprint"] = evaluation._digest(report["identity"])
    for row in report["candidates"]:
        for fold in row["folds"]:
            del fold["convergence_check"]
            for w in fold["warnings"]:
                del w["is_convergence_warning"]
    write(path, report)
    with pytest.raises(ValueError, match="challenger 0 could win"):
        evaluation.load_calibration_report(path)
    row = report["candidates"][0]
    for fold in row["folds"]:
        assert fold["status"] == "nonconverged"
        fold["status"] = "valid"
    _summarize_promoted_candidate(report, row)
    write(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="eligibility of selected candidate 0"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        return
    row["folds"][0]["diagnostics"]["n_iter"] = [int(short_cap) + 1]
    write(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="L-BFGS iteration count exceeds"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
