"""Strict archive checks distinguish unattainable scores from unfinished searches."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import warnings

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from supervised_nmf import evaluation
from supervised_nmf._calibration_report import (
    _check_confusion_rows, _joint_support_possible, _legacy_extra_label_score, _plain_tree_setting, _possible_balanced_score,
    _retained_support_has_integer_target, _supports_can_sum_to,
)


def test_large_fold_needs_an_actual_rounded_score_witness():
    a, b = 100_000_000, 100_000_001
    assert _possible_balanced_score(math.nextafter(1., 0.), a, b) is False
    score = ((a - 123) / a + (b - 321) / b) / 2
    assert _possible_balanced_score(score, a, b) is True


def test_large_balanced_fold_records_and_checks_rounding_witness():
    count = 262_142
    truth = np.repeat(np.array([0, 1], dtype=np.uint8), count)
    predictions = 1 - truth
    predictions[:45_000] = 0
    predictions[count:count + 63_000] = 1
    record = {}
    with warnings.catch_warnings(record=True) as caught:
        score = evaluation._score_calibration_fold("balanced_accuracy", truth, predictions, np.unique(truth),
                                                  caught, record, [count, count])
    assert score == math.nextafter(108_000 / (2 * count), 1)
    assert record["balanced_accuracy_hits"] == [45_000, 63_000]
    assert _possible_balanced_score(score, count, count) is True


def _write_hit_checkpoint(path):
    count = 4200
    truth = np.repeat([0, 1], count)
    feature = 1 - truth
    feature[:2500] = 0
    feature[count:count + 3000] = 1
    a, b = np.arange(2 * count), np.arange(2 * count, 4 * count)
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {}, np.tile(feature, 2)[:, None],
                                  np.tile(truth, 2), cv=2, folds=[(a, b), (b, a)], seed=3,
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)


def test_recorded_hit_counts_are_validated_on_public_read(tmp_path):
    path = tmp_path / "fold-hits.json"
    _write_hit_checkpoint(path)
    report = evaluation.load_calibration_report(path)
    fold = report["candidates"][0]["folds"][0]
    assert fold["balanced_accuracy_hits"] == [2500, 3000]
    for hits, error in (([2500, 2999], "disagrees"), ([True, 3000], "Invalid"), ([4201, 3000], "Invalid")):
        fold["balanced_accuracy_hits"] = hits
        report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
        path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())


def test_failed_fold_before_score_publication_leaves_no_orphaned_hits(tmp_path, monkeypatch):
    original = evaluation._score_calibration_fold
    first_fold = True

    def score_until_deadline(*args, **kwargs):
        nonlocal first_fold
        score = original(*args, **kwargs)
        if first_fold:
            first_fold = False
            raise TimeoutError("fold deadline")
        return score

    monkeypatch.setattr(evaluation, "_score_calibration_fold", score_until_deadline)
    path = tmp_path / "fold-deadline.json"
    _write_hit_checkpoint(path)
    report = evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())
    failed, completed = report["candidates"][0]["folds"]
    assert failed["status"] == "failed" and failed["error_type"] == "TimeoutError"
    assert {"score", "balanced_accuracy", "balanced_accuracy_hits", "balanced_accuracy_target_cache",
            "balanced_accuracy_prediction_kind", "balanced_accuracy_custom_scalar_inputs"}.isdisjoint(failed)
    assert completed["balanced_accuracy_hits"] == [2500, 3000]
    assert report["selected_candidate_index"] is None


def test_large_mixed_integer_fold_binds_hits_to_scorer_retained_classes():
    repeats = 8192
    truth = np.tile(np.array([0, 0, 1, 1, 1, 1], dtype=np.int64), repeats)
    predictions = np.tile(np.array([0, 0, 1, 1, 0, 2**63 + 1], dtype=np.uint64), repeats)
    record = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = evaluation._score_calibration_fold("balanced_accuracy", truth, predictions, np.unique(truth),
                                                  caught, record, [2 * repeats, 4 * repeats])
    assert score == (1 + 2 / 3) / 2
    assert record["balanced_accuracy_support"]["class_counts"] == [2 * repeats, 3 * repeats]
    assert record["balanced_accuracy_hits"] == [2 * repeats, 2 * repeats]


def test_extra_predicted_label_can_keep_full_mixed_dtype_support_and_hits():
    count = 4200
    truth = np.repeat([False, True], count)
    predictions = (~truth).astype(float)
    predictions[:2500] = 0.
    predictions[count:count + 3000] = 1.
    predictions[-1] = 2.
    record = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = evaluation._score_calibration_fold("balanced_accuracy", truth, predictions, np.unique(truth),
                                                  caught, record, [count, count])
    assert [str(w.message) for w in caught] == ["y_pred contains classes not in y_true"]
    assert score == (2500 / count + 3000 / count) / 2
    assert record == {"balanced_accuracy_hits": [2500, 3000], "balanced_accuracy_target_cache": "none",
                      "balanced_accuracy_prediction_kind": "f", "balanced_accuracy_custom_scalar_inputs": "none"}


def _use_v3_evidence(report):
    identity = report["identity"]
    identity["fold_evidence_version"] = 3
    del identity["balanced_accuracy_support_method"]
    del identity["class_label_evidence"], identity["screening_diagnostics_sha256"]
    del report["refinement_axis_inputs"]
    for row in report["candidates"]:
        for fold in row["folds"]:
            del fold["convergence_check"]["flag_checks"]
            del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
            del fold["balanced_accuracy_custom_scalar_inputs"]
    report["fingerprint"] = evaluation._digest(identity)


def test_v3_numeric_witness_preserves_extended_and_nonfinite_float_displays():
    tagged = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "1", "exponent": 0}
    for labels in ([None, 2.], [1., None], [None, None], [None, tagged], [tagged, dict(tagged, exponent=1)]):
        report = {"identity": {"fold_evidence_version": 3}, "class_labels": labels}
        assert _retained_support_has_integer_target(report) is False


def test_only_complete_plain_object_constraint_vectors_establish_tree_behavior():
    vector = evaluation.configuration_identity(np.array([1], dtype=object))
    assert _plain_tree_setting(vector)
    assert not _plain_tree_setting(dict(vector, content=[]))
    assert not _plain_tree_setting(dict(vector, content=[{"kind": "estimator"}]))
    assert not _plain_tree_setting(dict(vector, dtype=dict(vector["dtype"], metadata={"kind": "dict", "items": []})))


@pytest.mark.parametrize("version", [3, 6])
def test_retained_numeric_support_cannot_justify_a_unicode_fold_score(tmp_path, version):
    path = tmp_path / "unicode.json"
    x = np.tile([0, 1], 8)
    y = np.array(["absent", "present"])[x]
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {}, x[:, None], y, seed=3, cv=2,
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["identity"]["class_label_evidence"]["dtype_kind"] == "U"
    assert report["identity"]["validation_class_counts"] == [[4, 4], [4, 4]]
    if version == 3:
        _use_v3_evidence(report)
    row, = report["candidates"]
    fold = row["folds"][0]
    fold["balanced_accuracy_support"] = {"warning_index": len(fold["warnings"]), "class_counts": [3, 4]}
    fold["warnings"].append(dict(category="UserWarning", message="y_pred contains classes not in y_true",
                                 is_convergence_warning=False))
    fold["convergence_check"]["warnings_checked"] += 1
    # One correct prediction among three retained samples would give 1/6.
    # With the observed four samples per class, all scores are multiples of 1/8.
    fold["score"] = fold["balanced_accuracy"] = 1 / 6
    scores = [f["score"] for f in row["folds"]]
    row["mean_score"] = row["mean_balanced_accuracy"] = report["selected_mean_score"] = float(np.mean(scores))
    row["fold_score_std"] = float(np.std(scores, ddof=1))
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Balanced accuracy"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize(("version", "tree", "dtype", "constraints"), [
    (3, DecisionTreeClassifier, np.bool_, None), (3, DecisionTreeClassifier, np.int64, None),
    (6, DecisionTreeClassifier, np.bool_, None), (6, DecisionTreeClassifier, np.int64, [1]),
    (6, DecisionTreeClassifier, np.uint64, np.array([1])), (6, ExtraTreeClassifier, np.bool_, None),
    (6, DecisionTreeClassifier, np.bool_, np.array([1], dtype=object)),
])
@pytest.mark.parametrize("pinned", [False, True])
def test_tree_cannot_win_by_claiming_reduced_integer_support(tmp_path, version, tree, dtype, constraints, pinned):
    _check_fabricated_tree_winner(tmp_path, version, tree, dtype, constraints, pinned)


@pytest.mark.parametrize("pinned", [False, True])
def test_plain_float_tree_cannot_win_by_claiming_reduced_support(tmp_path, pinned):
    _check_fabricated_tree_winner(tmp_path, 6, DecisionTreeClassifier, np.float64, None, pinned)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(("counts", "hits"), [([4, 3], [4, 2]), ([4, 0], [3, 0])])
def test_stock_tree_native_float_cache_limits_reduced_support(tmp_path, counts, hits, pinned):
    dtype = np.dtype("f8", metadata={"unique": np.array([0., 1.])})
    _check_fabricated_tree_winner(tmp_path, 6, DecisionTreeClassifier, dtype, None, pinned, counts, hits)


def _check_fabricated_tree_winner(tmp_path, version, tree, dtype, constraints, pinned, counts=(4, 3), hits=(4, 2)):
    path = tmp_path / "tree.json"
    y = np.tile(np.repeat(np.array([0, 1], dtype=dtype), 4), 2)
    x = np.tile([0, 0, 0, 1, 1, 1, 1, 0], 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(tree(random_state=3, monotonic_cst=constraints), {"min_samples_split": [2, 20]}, x, y,
                                  cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                  refit=False, checkpoint_path=path)
    digest = sha256(path.read_bytes()).hexdigest() if pinned else None
    report = evaluation.load_calibration_report(path, expected_sha256=digest)
    if report["identity"]["software"]["scikit-learn"] not in ("1.9.0", "1.9.1"):
        pytest.skip("This contradiction uses the archived scikit-learn tree implementation.")
    assert report["selected_candidate_index"] == 0
    if dtype is np.float64:
        assert report["identity"]["class_label_evidence"]["dtype_has_metadata"] is False
    winner, challenger = report["candidates"]
    assert (winner["mean_score"], challenger["mean_score"]) == (.75, .5)
    assert all("balanced_accuracy_support" not in fold for fold in challenger["folds"])
    if version == 3:
        _use_v3_evidence(report)

    # Four observations of each class give multiples of 1/8. Dropping one
    # observation would permit perfect recall and 2/3 recall, beating .75.
    score = float(np.mean([h / n for h, n in zip(hits, counts) if n]))
    for fold in challenger["folds"]:
        fold["balanced_accuracy_support"] = {"warning_index": len(fold["warnings"]), "class_counts": list(counts)}
        if version == 6:
            fold["balanced_accuracy_hits"] = list(hits)
        fold["warnings"].append(dict(category="UserWarning", message="y_pred contains classes not in y_true",
                                     is_convergence_warning=False))
        fold["convergence_check"]["warnings_checked"] += 1
        fold["score"] = fold["balanced_accuracy"] = score
    challenger.update(mean_score=score, mean_balanced_accuracy=score, fold_score_std=0.)
    selected = challenger if score > winner["mean_score"] else winner
    report.update(selected_candidate_index=selected["index"], selected_parameters=selected["parameters"],
                  selected_mean_score=selected["mean_score"],
                  refinement=evaluation.refinement_diagnostics([c["parameters"] for c in report["candidates"]],
                                                                selected["parameters"]))
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    digest = sha256(path.read_bytes()).hexdigest() if pinned else None
    with pytest.raises(ValueError, match="stock tree"):
        evaluation.load_calibration_report(path, expected_sha256=digest)


class _ChangingCachedOne:
    def __init__(self):
        self.comparisons = 0

    def __hash__(self):
        return hash(1.)

    def __eq__(self, other):
        if other is self:
            return True
        if other == 1.:
            self.comparisons += 1
            return self.comparisons % 4 != 0
        return False

    def __lt__(self, other):
        return 1. < other

    def __gt__(self, other):
        return 1. > other


@pytest.mark.parametrize("label_dtype", ["f8", "i8", "bool"])
def test_cached_tree_score_and_warning_follow_one_sklearn_call_per_fold(tmp_path, monkeypatch, label_dtype):
    x = np.tile([0, 0, 0, 1, 1, 1, 1, 0], 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)

    def run(path):
        key = _ChangingCachedOne()
        dtype = np.dtype(label_dtype, metadata={"unique": np.array([0., key, 2.], dtype=object)})
        y = np.tile(np.repeat(np.array([0., 1.], dtype=dtype), 4), 2)
        result = evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {"min_samples_split": [2, 20]},
                                               x, y, cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                               refit=False, checkpoint_path=path)
        return result.report, key.comparisons

    def original_score(scoring, truth, predictions, *args):
        return evaluation._score_predictions(scoring, truth, predictions)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "_score_calibration_fold", original_score)
        reference, expected_comparisons = run(None)
    path = tmp_path / "changing-cache.json"
    report, comparisons = run(path)
    assert comparisons == expected_comparisons
    assert report["selected_candidate_index"] == reference["selected_candidate_index"]
    assert report["identity"]["class_label_evidence"]["opaque_unique_cache"] is True
    for current, original in zip(report["candidates"], reference["candidates"], strict=True):
        for fold, expected in zip(current["folds"], original["folds"], strict=True):
            assert fold["status"] == expected["status"] == "valid"
            assert fold["warnings"] == expected["warnings"]
            assert fold["score"] == expected["score"]
            assert "balanced_accuracy_confusion_rows" in fold
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report

    for edit in ("missing", "extra", "hit", "size", "bool", "warning", "dimension", "stock-cache", "changed-prediction", "method"):
        altered = deepcopy(report)
        fold = altered["candidates"][0]["folds"][0]
        witness = fold["balanced_accuracy_confusion_rows"]
        if edit == "missing":
            del fold["balanced_accuracy_confusion_rows"]
        elif edit == "extra":
            fold["balanced_accuracy_hits"] = [2, 1]
        elif edit == "hit":
            witness["rows"][0][1] = witness["rows"][0][0] + 1
        elif edit == "size":
            witness["rows"][0][0] = 9
        elif edit == "bool":
            witness["rows"][0][0] = True
        elif edit == "warning":
            witness["warning_index"] = len(fold["warnings"])
        elif edit == "dimension":
            witness["matrix_size"] = 0
        elif edit == "stock-cache":
            altered["identity"]["class_label_evidence"]["opaque_unique_cache"] = False
            fold["balanced_accuracy_target_cache"] = "native"
        elif edit == "changed-prediction":
            witness["prediction"]["dtype_kind"] = "u"
            fold["balanced_accuracy_prediction_kind"] = "u"
        else:
            del altered["identity"]["balanced_accuracy_support_method"]
        altered["fingerprint"] = evaluation._digest(altered["identity"])
        altered["checkpoint_sha256"] = evaluation._digest({k: v for k, v in altered.items() if k != "checkpoint_sha256"})
        path.write_text(json.dumps(altered, allow_nan=False), encoding="utf-8")
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="(?i)confusion row|unique-label cache|target cache"):
                evaluation.load_calibration_report(path, expected_sha256=digest)


def test_reading_cache_kind_does_not_call_a_metadata_key():
    class Key:
        def __hash__(self):
            return hash("unique")

        def __eq__(self, other):
            raise AssertionError("metadata inspection invoked equality")

    dtype = np.dtype("f8", metadata={Key(): ()})
    assert evaluation._unique_cache_state(dtype) == "opaque"


def test_many_confusion_rows_use_sklearns_numpy_mean():
    rows = [[1, 1]] + [[5, 1]] * 7
    recalls = [h / n for n, h in rows]
    score = float(np.mean(recalls))
    pred = {"dtype_kind": "u", "opaque_unique_cache": True, "has_unique_cache": True}
    target = {"dtype_kind": "i", "opaque_unique_cache": False, "has_unique_cache": False}
    witness = {"rows": rows, "warning_index": None, "matrix_size": len(rows), "prediction": pred}
    fold = {"score": score, "balanced_accuracy_confusion_rows": witness}
    _check_confusion_rows(fold, [18, 18], 0, target, False)
    pred["opaque_unique_cache"] = False
    with pytest.raises(ValueError, match="declared cache"):
        _check_confusion_rows(fold, [18, 18], 0, target, False)
    pred["opaque_unique_cache"] = True
    fold["score"] = sum(recalls) / len(recalls)
    assert fold["score"] != score
    with pytest.raises(ValueError, match="disagrees"):
        _check_confusion_rows(fold, [18, 18], 0, target, False)


def test_matrix_dimension_binds_single_row_and_preserves_filtered_warning():
    cache = {"dtype_kind": "i", "opaque_unique_cache": True, "has_unique_cache": True}
    witness = {"rows": [[6, 5]], "warning_index": None, "matrix_size": 1, "prediction": cache}
    fold = {"score": 5 / 6, "balanced_accuracy_confusion_rows": witness}
    with pytest.raises(ValueError, match="one-class"):
        _check_confusion_rows(fold, [4, 4], 0, cache, True)
    witness["matrix_size"] = 2
    _check_confusion_rows(fold, [4, 4], 0, cache, True)

    dtype = np.dtype("i8", metadata={"unique": np.array([0, 1, 2], dtype=object)})
    truth = np.array([0, 1, 0, 1], dtype=dtype)
    record = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        score = evaluation._score_calibration_fold("balanced_accuracy", truth, truth, np.unique(truth), caught, record)
    assert record["balanced_accuracy_confusion_rows"]["warning_index"] is None
    assert score == 1.
    _check_confusion_rows(dict(record, score=score), [2, 2], 0, cache, True)


@pytest.mark.parametrize("kind", ["U", "S"])
def test_stock_native_text_rows_obey_the_original_two_classes(kind):
    dtype = np.dtype(kind + "1", metadata={"unique": np.array(["a", "b"], dtype=kind + "1")})
    assert evaluation._unique_cache_state(dtype) == "native"
    label = {"dtype_kind": kind, "native_text": True, "has_unique_cache": True, "opaque_unique_cache": False}
    pred = label.copy()
    witness = {"matrix_size": 3, "warning_index": None, "prediction": pred}
    fold = {"balanced_accuracy_confusion_rows": witness}
    for rows, valid in (([[3, 2], [5, 3]], True), ([[2, 2]], True), ([[5, 5]], True),
                        ([[5, 3], [3, 2]], False), ([[6, 6]], False), ([[2, 1]], False),
                        ([[1, 1], [2, 2], [5, 3]], False)):
        witness["rows"] = rows
        fold["score"] = float(np.mean([h / n for n, h in rows]))
        if valid:
            _check_confusion_rows(fold, [3, 5], 0, label, True)
        else:
            with pytest.raises(ValueError, match="stock tree"):
                _check_confusion_rows(fold, [3, 5], 0, label, True)
    # The fitted tree cannot manufacture an executable prediction cache
    # from this native target; a cache on object labels can execute equality.
    pred["opaque_unique_cache"] = True
    with pytest.raises(ValueError, match="stock tree"):
        _check_confusion_rows(fold, [3, 5], 0, label, True)
    label["opaque_unique_cache"] = True
    _check_confusion_rows(fold, [3, 5], 0, label, True)


class CachedTextTree(DecisionTreeClassifier):
    def __init__(self, label_dtype):
        super().__init__(random_state=3)
        self.label_dtype = label_dtype

    def predict(self, X):
        dtype = np.dtype(self.label_dtype, metadata={"unique": np.array(["a", "b"])})
        return np.array(["a", "a", "a", "b", "b", "b", "a", "z"], dtype=dtype)[X[:, 0].astype(int)]


def test_native_unicode_tree_cannot_win_by_claiming_two_partially_retained_classes(tmp_path):
    path = tmp_path / "unicode-tree.json"
    dtype = np.dtype("U1", metadata={"unique": np.array(["a", "b"])})
    y = np.tile(np.repeat(np.array(["a", "b"], dtype=dtype), 4), 2)
    x = np.tile([0, 0, 0, 1, 1, 1, 1, 0], 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {"min_samples_split": [2, 20]}, x, y,
                                  cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                  refit=False, checkpoint_path=path)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        report = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == .75
    challenger = report["candidates"][1]
    score = (1 + 2 / 3) / 2
    for fold in challenger["folds"]:
        fold["balanced_accuracy_confusion_rows"]["rows"] = [[4, 4], [3, 2]]
        fold["score"] = fold["balanced_accuracy"] = score
    challenger.update(mean_score=score, mean_balanced_accuracy=score, fold_score_std=0.)
    report.update(selected_candidate_index=1, selected_parameters=challenger["parameters"], selected_mean_score=score,
                  refinement=evaluation.refinement_diagnostics([c["parameters"] for c in report["candidates"]],
                                                                challenger["parameters"]))
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="stock tree"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def test_native_unicode_stock_tree_preserves_genuine_one_sided_cache(tmp_path):
    path = tmp_path / "one-class-cache.json"
    dtype = np.dtype("U1", metadata={"unique": np.array(["a", "c"])})
    y = np.tile(np.array(["a"] * 3 + ["b"] * 5, dtype=dtype), 2)
    x = np.tile([0, 0, 0, 1, 1, 1, 1, 1], 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {}, x, y, cv=2, folds=[(a, b), (b, a)],
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = confusion_matrix(y[a], DecisionTreeClassifier(random_state=3).fit(x[b], y[b]).predict(x[a]))
    rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        report = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert report["selected_mean_score"] == 1.
        for fold in report["candidates"][0]["folds"]:
            assert fold["balanced_accuracy_confusion_rows"]["rows"] == rows


def test_custom_tree_cannot_discard_required_rows_to_win(tmp_path):
    path = tmp_path / "missing-rows.json"
    dtype = np.dtype("U1", metadata={"unique": np.array(["a", "b"])})
    y = np.tile(np.repeat(np.array(["a", "b"], dtype=dtype), 4), 2)
    x = np.tile(np.arange(8), 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(CachedTextTree("U1"), {"label_dtype": ["U1", "object"]}, x, y,
                                  cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                  refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    winner = report["selected_candidate_index"]
    assert winner in (0, 1)
    challenger = 1 - winner
    candidates = report["candidates"]
    assert max(row["mean_score"] for row in candidates) < 1.
    row = candidates[challenger]
    for fold in row["folds"]:
        del fold["balanced_accuracy_confusion_rows"]
        fold["score"] = fold["balanced_accuracy"] = 1.
    row.update(mean_score=1., mean_balanced_accuracy=1., fold_score_std=0.)
    report.update(selected_candidate_index=challenger, selected_parameters=row["parameters"], selected_mean_score=1.,
                  refinement=evaluation.refinement_diagnostics([c["parameters"] for c in candidates], row["parameters"]))
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Missing balanced accuracy confusion rows"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
    for key in ("dtype_has_metadata", "has_unique_cache", "opaque_unique_cache"):
        del report["identity"]["class_label_evidence"][key]
    report["fingerprint"] = evaluation._digest(report["identity"])
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="missing its target label-cache observations"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("label_dtype", ["U1", "object"])
def test_text_cache_records_the_same_rows_sklearn_scored(tmp_path, label_dtype):
    path = tmp_path / "text-cache.json"
    dtype = np.dtype(label_dtype, metadata={"unique": np.array(["a", "b"])})
    y = np.tile(np.repeat(np.array(["a", "b"], dtype=dtype), 4), 2)
    x = np.tile(np.arange(8), 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    model = CachedTextTree(label_dtype)
    predicted = model.predict(x[a])
    expected = balanced_accuracy_score(y[a], predicted)
    matrix = confusion_matrix(y[a], predicted)
    rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    evaluation.calibration_search(model, {}, x, y, cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                  refit=False, checkpoint_path=path)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        report = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert report["selected_mean_score"] == expected
        assert report["identity"]["class_label_evidence"]["has_unique_cache"] is True
        for fold in report["candidates"][0]["folds"]:
            witness = fold["balanced_accuracy_confusion_rows"]
            assert fold["score"] == expected and witness["rows"] == rows
            assert witness["matrix_size"] == len(matrix)

    if label_dtype == "object":
        predicted[0] = "a\x00"
        with warnings.catch_warnings(record=True) as caught, pytest.raises(ValueError, match="NUL"):
            evaluation._score_calibration_fold("balanced_accuracy", y[a], predicted, np.unique(y[a]), caught, {})


class FloatPredictionTree(DecisionTreeClassifier):
    def predict(self, X):
        # sklearn accepts NumPy's optional unique-label cache on predictions.
        # A cached class list can outlive a change to the array's values.
        labels = np.dtype("float64", metadata={"unique": np.array([0., 1., 2.])})
        return np.array([0., 0., 0., 1., 1., 1., 0., 3.], dtype=labels)[X[:, 0]]


class FilteredWarningTree(DecisionTreeClassifier):
    def __init__(self, action="always", cached=True):
        super().__init__(random_state=3)
        self.action = action
        self.cached = cached

    def predict(self, X):
        warnings.filterwarnings(self.action, message="y_pred contains classes not in y_true", category=UserWarning,
                                module=r"^sklearn\.metrics\._classification$")
        if not self.cached:
            return np.zeros(len(X), dtype=int)
        dtype = np.dtype("float64", metadata={"unique": np.array([0., 1., 2.], dtype=object)})
        return np.array([0., 0., 0., 1., 1., 1., 0., 3.], dtype=dtype)[X[:, 0]]


@pytest.mark.parametrize("all_excluded", [False, True])
def test_cached_score_preserves_native_warning_location_and_rounding(all_excluded):
    if all_excluded:
        dtype = np.dtype("int64", metadata={"unique": np.array([2], dtype=object)})
        truth = np.array([0, 1], dtype=dtype)
        predicted = truth.copy()
    else:
        truth = np.repeat([0, 1], 10)
        predicted = np.array([0] + [1] * 8 + [2] + [0] * 8 + [1] * 2)

    def observe(operation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = operation(caught)
            except ValueError as exc:
                result = str(exc)
        return result, [(w.category, str(w.message), w.filename, w.lineno) for w in caught]

    original = observe(lambda caught: evaluation._score_predictions("balanced_accuracy", truth, predicted))
    actual = observe(lambda caught: evaluation._score_confusion_rows(truth, predicted, caught, {}))
    assert actual == original
    if not all_excluded:
        assert actual[0] == (.1 + .2) / 2 != .15
        warning, = actual[1]
        assert warning[1] == "y_pred contains classes not in y_true"
    elif np.__version__ == "1.26.3":
        assert actual[0] == "Nonfinite calibration score."
        assert [w[1] for w in actual[1]].count("y_pred contains classes not in y_true") == 1


@pytest.mark.parametrize("action", ["error", "ignore", "always"])
def test_module_filtered_cache_warning_preserves_sklearn_fold_and_winner(tmp_path, monkeypatch, action):
    y = np.tile(np.repeat([0, 1], 4), 2)
    x = np.tile(np.arange(8), 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)

    def search(path):
        return evaluation.calibration_search(FilteredWarningTree(action), {"cached": [True, False]}, x, y,
                                             scoring="balanced_accuracy", folds=[(a, b), (b, a)], cv=2, seed=3,
                                             refit=False, checkpoint_path=path).report

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "_score_calibration_fold",
                      lambda scoring, truth, pred, *args: evaluation._score_predictions(scoring, truth, pred))
        reference = search(None)
    path = tmp_path / "module-filter.json"
    report = search(path)
    assert report["selected_candidate_index"] == reference["selected_candidate_index"] == (1 if action == "error" else 0)
    for row, original in zip(report["candidates"], reference["candidates"], strict=True):
        assert row["valid"] == original["valid"]
        for fold, expected in zip(row["folds"], original["folds"], strict=True):
            for field in ("score", "status", "warnings", "error", "error_type"):
                assert fold.get(field) == expected.get(field)
            if row["parameters"]["cached"]:
                if action == "error":
                    assert fold["error_type"] == "UserWarning"
                    assert "balanced_accuracy_confusion_rows" not in fold
                else:
                    witness = fold["balanced_accuracy_confusion_rows"]
                    assert witness["warning_index"] == (0 if action == "always" else None)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report


class CachedIntegerPredictionTree(DecisionTreeClassifier):
    def predict(self, X):
        labels = np.dtype("int64", metadata={"unique": np.array([0, 1])})
        return np.array([0, 0, 0, 1, 1, 1, 0, 3], dtype=labels)[X[:, 0]]


class WideUnsignedPredictionTree(DecisionTreeClassifier):
    def predict(self, X):
        big = 2**62
        return np.array([0, 0, 0, big, big, big, 0, big + 1], dtype=np.uint64)[X[:, 0]]


def _check_tree_score_evidence(path, truth, predicted):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_score = balanced_accuracy_score(truth, predicted)
        warned = any(str(w.message) == "y_pred contains classes not in y_true" for w in caught)
        class_matrices = [confusion_matrix(truth, predicted, sample_weight=(truth == label).astype(int))
                          for label in np.unique(truth)]
        counts = [int(matrix.sum()) for matrix in class_matrices]
        hits = [int(matrix.trace()) for matrix in class_matrices]
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        report = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert report["selected_candidate_index"] == 0
        assert report["identity"]["balanced_accuracy_support_method"] == "confusion-matrix-v5"
        for fold in report["candidates"][0]["folds"]:
            assert fold["score"] == expected_score
            if sum(counts) < len(truth):
                witness = fold["balanced_accuracy_support"]
                assert witness["class_counts"] == counts
                assert fold["balanced_accuracy_hits"] == hits
                if warned:
                    assert fold["warnings"][witness["warning_index"]]["message"] == "y_pred contains classes not in y_true"
                else:
                    assert witness["warning_index"] is None
            else:
                assert "balanced_accuracy_support" not in fold
    return counts


@pytest.mark.parametrize("positive_count", [4, 5])
def test_custom_tree_can_emit_reduced_support_for_boolean_targets(tmp_path, positive_count):
    path = tmp_path / "custom-tree.json"
    truth = np.repeat([False, True], [4, positive_count])
    y = np.tile(truth, 2)
    x = np.tile(list(range(8)) + [0] * (positive_count - 4), 2)[:, None]
    a, b = np.arange(len(truth)), np.arange(len(truth), 2 * len(truth))
    evaluation.calibration_search(FloatPredictionTree(random_state=3), {}, x, y, cv=2, folds=[(a, b), (b, a)],
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    counts = _check_tree_score_evidence(path, y[b], FloatPredictionTree().predict(x[b]))
    if np.__version__ == "1.26.3":
        assert counts == [4, positive_count - 1]
    elif np.__version__ == "2.2.6":
        assert counts == [4, positive_count]


@pytest.mark.parametrize("float_or_int", [np.float64, np.int64])
@pytest.mark.parametrize("surviving_class", [0, 1])
def test_stock_tree_can_emit_reduced_support_for_float_targets(tmp_path, surviving_class, float_or_int):
    path = tmp_path / "cached-tree.json"
    dtype = np.dtype(float_or_int, metadata={"unique": np.array([surviving_class, 2], dtype=float_or_int)})
    y = np.tile(np.repeat(np.array([0., 1.], dtype=dtype), 4), 2)
    x = np.tile(np.repeat([0, 1], 4), 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {}, x, y, cv=2, folds=[(a, b), (b, a)],
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    predicted = DecisionTreeClassifier(random_state=3).fit(x[a], y[a]).predict(x[b])
    counts = _check_tree_score_evidence(path, y[b], predicted)
    if np.__version__ == "1.26.3":
        assert counts == ([4, 0] if surviving_class == 0 else [0, 4])
    elif np.__version__ == "2.2.6":
        assert counts == [4, 4]


@pytest.mark.parametrize("dtype", [bool, np.int64])
def test_native_prediction_cache_can_drop_rows_without_a_warning(tmp_path, dtype):
    y = np.tile(np.repeat(np.array([0, 1], dtype=dtype), 4), 2)
    x = np.tile(np.arange(8), 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    path = tmp_path / "cached-predictions.json"
    tree = CachedIntegerPredictionTree(random_state=3)
    evaluation.calibration_search(tree, {}, x, y, cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                  refit=False, checkpoint_path=path)
    counts = _check_tree_score_evidence(path, y[b], tree.predict(x[b]))
    if np.__version__ == "1.26.3":
        assert counts == [4, 3]
        fold = json.loads(path.read_bytes())["candidates"][0]["folds"][0]
        assert fold["warnings"] == [] and fold["balanced_accuracy_support"]["warning_index"] is None
    elif np.__version__ == "2.2.6":
        assert counts == [4, 4]


def test_native_mixed_integer_promotion_observes_dropped_rows_even_if_warning_filtered(tmp_path):
    y = np.tile(np.repeat(np.array([0, 2**62], dtype=np.int64), 4), 2)
    x = np.tile(np.arange(8), 2)[:, None]
    a, b = np.arange(8), np.arange(8, 16)
    path = tmp_path / "wide-predictions.json"
    tree = WideUnsignedPredictionTree(random_state=3)
    evaluation.calibration_search(tree, {}, x, y, cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                  refit=False, checkpoint_path=path)
    predicted = tree.predict(x[b])
    counts = _check_tree_score_evidence(path, y[b], predicted)
    assert counts == [4, 3]
    record = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("ignore", "y_pred contains classes not in y_true", UserWarning)
        score = evaluation._score_calibration_fold("balanced_accuracy", y[b], predicted, np.unique(y), caught,
                                                  record, [4, 4])
    assert score == (.75 + 2/3) / 2
    assert record["balanced_accuracy_support"] == {"warning_index": None, "class_counts": [4, 3]}
    assert record["balanced_accuracy_hits"] == [3, 2]


def test_legacy_retained_class_limits_and_alternative_partition_witness():
    assert _legacy_extra_label_score(math.nextafter(1., 0.), 64, 64) is False
    assert _legacy_extra_label_score((1 + 2 / 3) / 2, 64, 64) is True
    score = (1 / 63 + 2 / 65) / 2
    assert _legacy_extra_label_score(score, 64, 64) is None
    assert _joint_support_possible([set(), {1.}], [128, 128], [(64, 65), (63, 64)], 128,
                                   [{score}, set()]) is True


def test_sparse_legacy_folds_obey_shared_class_count_residue():
    # A six-row fold with this score needs a 2+4 or 4+2 split.
    scores = [{1 / 8}] * 2001
    sizes = [6] * len(scores)
    bounds = [(1, 5)] * len(scores)
    assert _joint_support_possible(scores, sizes, bounds, 6003) is False
    assert _joint_support_possible(scores, sizes, bounds, 6004) is True


def test_large_sparse_partition_search_requires_a_witness():
    options = [[0, 3, 1_000_000]] * 3
    assert _supports_can_sum_to(options, 1_000_003) is True
    assert _supports_can_sum_to(options, 999_998) is False
    many_possible_totals = [[0, 2 ** power, 10_000_001] for power in range(20)]
    assert _supports_can_sum_to(many_possible_totals, 750_000) is None


def test_unresolved_historical_balanced_score_refuses_even_with_exact_file_digest(tmp_path):
    path = tmp_path / "unsupported-score.json"
    y = np.tile([0, 1], 128)
    evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {}, y[:, None], y,
                                  cv=2, seed=3, scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    reported_supports = report["identity"]["validation_class_counts"]
    assert reported_supports == [[64, 64], [64, 64]]
    report["identity"]["fold_evidence_version"] = 2
    del report["identity"]["balanced_accuracy_support_method"]
    report["fingerprint"] = evaluation._digest(report["identity"])
    row, = report["candidates"]
    for fold in row["folds"]:
        del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
        del fold["balanced_accuracy_custom_scalar_inputs"]
    fold = row["folds"][0]
    assert "balanced_accuracy_hits" not in fold
    fold["warnings"].append(dict(category="UserWarning", message="y_pred contains classes not in y_true",
                                 is_convergence_warning=False))
    fold["convergence_check"]["warnings_checked"] += 1
    fold["score"] = fold["balanced_accuracy"] = math.pi / 10
    scores = [fold["score"] for fold in row["folds"]]
    row["mean_score"] = row["mean_balanced_accuracy"] = report["selected_mean_score"] = float(np.mean(scores))
    row["fold_score_std"] = float(np.std(scores, ddof=1))
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Balanced accuracy cannot be verified"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
