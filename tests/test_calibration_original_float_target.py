"""Qualification uses the two floating classes in the original calibration request."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from supervised_nmf import evaluation


pytestmark = pytest.mark.skipif(sklearn.__version__ not in ("1.9.0", "1.9.1"),
                                reason="Audited sklearn float-to-int64 target check.")


class FeaturePredictor(ClassifierMixin, BaseEstimator):
    external_target = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if FeaturePredictor.external_target is not None:
            target = FeaturePredictor.external_target
            np.floor(target, out=target)
        return self

    def predict(self, X):
        return X[:, 0]


def search(path, scoring, X, y, *, refit=False):
    a, b = np.arange(8), np.arange(8, 16)
    return evaluation.calibration_search(FeaturePredictor(), {}, X, y, cv=2, folds=[(a, b), (b, a)],
                                         threads=1, scoring=scoring, refit=refit, checkpoint_path=path)


def samples():
    return np.tile(np.repeat([0., 1.], [3, 5]), 2)[:, None]


def seal(path, report):
    report["fingerprint"] = evaluation._digest(report["identity"])
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf8")


def check_read(path, *, refused=None):
    raw = path.read_bytes()
    for options in ({}, {"expected_sha256": sha256(raw).hexdigest()}):
        if refused:
            with pytest.raises(ValueError, match=refused):
                evaluation.load_calibration_report(path, **options)
        else:
            assert evaluation.load_calibration_report(path, **options) == json.loads(raw)
        assert path.read_bytes() == raw


def use_v3_evidence(report):
    identity = report["identity"]
    identity["fold_evidence_version"] = 3
    for field in ("class_label_evidence", "balanced_accuracy_support_method", "screening_diagnostics_sha256",
                  "original_float_target_policy"):
        identity.pop(field, None)
    del report["refinement_axis_inputs"]
    for candidate in report["candidates"]:
        for fold in candidate["folds"]:
            fold.get("convergence_check", {}).pop("flag_checks", None)
            for field in ("balanced_accuracy_target_cache", "balanced_accuracy_prediction_kind",
                          "balanced_accuracy_custom_scalar_inputs"):
                fold.pop(field, None)


@pytest.mark.parametrize("scoring,metric", [("accuracy", accuracy_score),
                                            ("balanced_accuracy", balanced_accuracy_score)])
def test_changed_target_keeps_original_eligibility_and_scorer_errors(tmp_path, monkeypatch, scoring, metric):
    X = samples()
    y = X[:, 0] + .5
    fingerprint = evaluation.data_fingerprint(X, y)
    with pytest.raises(ValueError) as original_error:
        metric(y[:8], X[:8, 0])
    path = tmp_path / "calibration.json"
    stable = search(path, scoring, X, y).report
    check_read(path)
    for fold in stable["candidates"][0]["folds"]:
        assert fold["error"] == str(original_error.value)
        assert fold["status"] == "failed" and "score" not in fold
    monkeypatch.setattr(FeaturePredictor, "external_target", y)
    changed_path = tmp_path / "changed.json"
    changed = search(changed_path, scoring, X, y).report
    assert metric(y, X[:, 0]) == 1
    assert changed["identity"]["data_sha256"] == fingerprint
    assert changed["class_labels"] == [.5, 1.5] and changed["class_counts"] == [6, 10]
    assert changed["selected_candidate_index"] is None and changed["selected_mean_score"] is None
    for fold in changed["candidates"][0]["folds"]:
        assert fold["status"] == "failed" and "original floating class labels" in fold["error"]
        assert "score" not in fold and "accuracy" not in fold
        assert not any(key.startswith("balanced_accuracy") for key in fold)
    check_read(changed_path)

    y[:] = X[:, 0] + .5
    with pytest.warns(evaluation.CalibrationFallbackWarning):
        fallback = search(tmp_path / "refit.json", scoring, X, y, refit=True)
    assert fallback.report["is_fallback"] is True
    np.testing.assert_array_equal(fallback.estimator.classes_, [.5, 1.5])
    np.testing.assert_array_equal(fallback.estimator.predict(X), np.full(len(X), 1.5))


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_scored_failures_and_winners_cannot_claim_original_fractional_classes(tmp_path, scoring):
    X = samples()
    path = tmp_path / "fractional.json"
    failed = search(path, scoring, X, X[:, 0] + .5).report
    valid = search(tmp_path / "valid.json", scoring, X, X[:, 0]).report
    assert valid["selected_mean_score"] == 1
    for historical in (False, True):
        for scored_failure in (False, True):
            altered = deepcopy(failed if scored_failure else valid)
            if scored_failure:
                target = altered["candidates"][0]["folds"][0]
                target.update(score=1., **{scoring: 1.})
            else:
                altered["identity"] = deepcopy(failed["identity"])
                altered["class_labels"] = failed["class_labels"]
            if historical:
                del altered["identity"]["original_float_target_policy"]
            seal(path, altered)
            check_read(path, refused="original floating class labels")
    del failed["identity"]["original_float_target_policy"]
    seal(path, failed)
    check_read(path)


@pytest.mark.parametrize("state", ["cv_completed", "calibrating", "refitting"])
def test_computational_resume_checks_scores_before_reusing_or_rewriting_folds(tmp_path, state):
    X = samples()
    y = X[:, 0] + .5
    path = tmp_path / "saved.json"
    refit = state == "refitting"
    if refit:
        with pytest.warns(evaluation.CalibrationFallbackWarning):
            report = search(path, "accuracy", X, y, refit=True).report
    else:
        report = search(path, "accuracy", X, y).report
    report["state"] = state
    folds = report["candidates"][0]["folds"]
    folds[0].update(score=1., accuracy=1.)
    if state == "calibrating":
        folds.pop()
    seal(path, report)
    sealed = path.read_bytes()
    with pytest.raises(ValueError, match="original floating class labels"):
        search(path, "accuracy", X, y, refit=refit)
    assert sealed == path.read_bytes()
    del report["identity"]["original_float_target_policy"]
    seal(path, report)
    sealed = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        search(path, "accuracy", X, y, refit=refit)
    assert sealed == path.read_bytes()


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_native_float_int64_endpoints_and_fractional_neighbors(tmp_path, scoring):
    X = samples()
    limit = np.float64(2**63)
    for number, possible in ((-limit, True), (np.nextafter(limit, 0.), True),
                             (np.nextafter(-limit, -np.inf), False), (np.nextafter(limit, np.inf), False),
                             (np.nextafter(1., 2.), False), (limit, None)):
        y = np.array([0., number])[X[:, 0].astype(int)]
        path = tmp_path / f"{number.hex()}.json"
        # The features give the true label directly, including at the int64 boundary.
        report = search(path, scoring, y[:, None], y).report
        if possible is None:
            with np.errstate(invalid="ignore"):
                possible = bool(np.array_equal(y, y.astype(np.int64).astype(y.dtype)))
            expected = "positive-int64-boundary-passed-v1" if possible else "positive-int64-boundary-failed-v1"
            assert report["identity"]["original_float_target_policy"] == expected
        assert report["selected_mean_score"] == (1. if possible else None)
        check_read(path)
        if not possible:
            report["candidates"][0]["folds"][0].update(score=1., **{scoring: 1.})
            seal(path, report)
            check_read(path, refused="original floating class labels")
        if number == limit:
            del report["identity"]["original_float_target_policy"]
            seal(path, report)
            check_read(path, refused="round-trip observation")


def test_exact_extended_class_evidence_preserves_fraction_and_integer_precision(tmp_path):
    X = samples()
    path = tmp_path / "exact.json"
    valid = search(path, "accuracy", X, X[:, 0]).report
    def tag(n, e):
        return {"__supervised_nmf_scalar__": "longdouble-v1", "significand": str(n), "exponent": e}

    for labels in ([tag(1, 0), tag(2**63 + 1, -63)], [tag(2**63 - 1, 0), tag(1, 63)]):
        altered = deepcopy(valid)
        altered["class_labels"] = altered["identity"]["class_label_evidence"]["labels"] = labels
        seal(path, altered)
        check_read(path, refused="original floating class labels")


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= 52, reason="Requires a wider long double.")
def test_native_extended_neighbor_of_one_survives_writer_and_reader(tmp_path):
    X = samples()
    neighbor = np.nextafter(np.longdouble(1), np.longdouble(2))
    assert neighbor.as_integer_ratio()[1] > 1 and float(neighbor) == 1.0
    y = np.array([np.longdouble(0), neighbor])[X[:, 0].astype(int)]
    path = tmp_path / "longdouble.json"
    report = search(path, "accuracy", X, y).report
    assert report["selected_mean_score"] is None
    assert report["identity"]["original_float_target_policy"] == "outside-int64-roundtrip-v1"
    check_read(path)
    report["candidates"][0]["folds"][0].update(score=1., accuracy=1.)
    seal(path, report)
    check_read(path, refused="original floating class labels")


_TAGGED_HALF = {"__supervised_nmf_scalar__": "longdouble-v1", "significand": "1", "exponent": -1}


@pytest.mark.parametrize("labels,numeric_error", [
    ([.5, 1.5], "original floating class labels"), ([0., float(2**63)], "round-trip observation"),
    ([.5, None], "original floating class labels"), ([None, .5], "original floating class labels"),
    ([_TAGGED_HALF, None], "original floating class labels"), ([None, _TAGGED_HALF], "original floating class labels"),
    ([None, float(-(2**63))], "original floating class labels"), ([2049., None], "original floating class labels"),
    ([None, 1.], "displayed a nonfinite floating class as null"), ([None, None], "nonfinite floating class as null"),
])
def test_legacy_float_displays_with_and_without_numeric_witness(tmp_path, labels, numeric_error):
    X = samples()
    path = tmp_path / "legacy.json"
    report = search(path, "balanced_accuracy", X, X[:, 0]).report
    use_v3_evidence(report)
    seal(path, report)
    check_read(path)
    report["class_labels"] = labels
    seal(path, report)
    check_read(path, refused="omitted the target dtype")
    fold = report["candidates"][0]["folds"][0]
    fold["balanced_accuracy_support"] = {"warning_index": len(fold["warnings"]), "class_counts": [2, 5]}
    fold["warnings"].append({"category": "UserWarning", "message": "y_pred contains classes not in y_true",
                             "is_convergence_warning": False})
    fold["convergence_check"]["warnings_checked"] += 1
    seal(path, report)
    check_read(path, refused=numeric_error)


@pytest.mark.parametrize("first,second", [(.5, _TAGGED_HALF), ({"__supervised_nmf_scalar__": "custom"}, None),
                                          (None, {"__supervised_nmf_scalar__": "longdouble-v1",
                                                  "significand": "2", "exponent": -2})])
def test_legacy_object_displays_that_cannot_share_a_native_float_encoding(tmp_path, first, second):
    class Label(np.str_):
        def item(self):
            return first if self == "a" else second

    y = np.array([Label("a"), Label("b")] * 8, dtype=object)
    path = tmp_path / "objects.json"
    report = evaluation.calibration_search(DummyClassifier(), {}, samples(), y, cv=2, seed=3,
                                           scoring="accuracy", refit=False, checkpoint_path=path).report
    assert report["class_labels"] == [first, second] and report["selected_mean_score"] == .5
    check_read(path)
    use_v3_evidence(report)
    seal(path, report)
    check_read(path)


def test_legacy_unscored_partial_native_float_displays_remain_readable(tmp_path):
    X = samples()
    path = tmp_path / "failed.json"
    report = search(path, "balanced_accuracy", X, X[:, 0] + .5).report
    assert report["selected_mean_score"] is None
    del report["identity"]["original_float_target_policy"]
    for missing_dtype in (False, True):
        if missing_dtype:
            use_v3_evidence(report)
        for labels in ([.5, None], [None, _TAGGED_HALF], [None, 1.], [None, None]):
            report["class_labels"] = labels
            if not missing_dtype:
                report["identity"]["class_label_evidence"]["labels"] = labels
            seal(path, report)
            check_read(path)


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("labels", [[.5, np.inf], [-np.inf, .5], [1., np.nan], np.array([-np.inf, 1.], dtype="f2")])
def test_nonfinite_native_float_classes_rejected_before_fit_or_checkpoint(tmp_path, monkeypatch, scoring, mutate, labels):
    X = samples()
    y = np.asarray(labels)[X[:, 0].astype(int)]
    before = y.copy()
    calls = []

    def fit(self, X_train, y_train):
        calls.append("fit")
        if mutate:
            y[:] = X[:, 0]
        return self

    monkeypatch.setattr(FeaturePredictor, "fit", fit)
    with sklearn.config_context(assume_finite=True), pytest.raises(ValueError, match="finite original floating class labels"):
        search(tmp_path / "calibration.json", scoring, X, y)
    np.testing.assert_array_equal(y, before)
    assert not calls and not list(tmp_path.iterdir())


@pytest.mark.parametrize("scoring,metric", [("accuracy", accuracy_score), ("balanced_accuracy", balanced_accuracy_score)])
def test_bound_nonfinite_float_scores_require_the_original_class_and_dtype(tmp_path, scoring, metric):
    X = samples()
    y = np.array([-np.inf, 1.], dtype="f2")[X[:, 0].astype(int)]
    with np.errstate(invalid="ignore", over="ignore"), sklearn.config_context(assume_finite=True):
        # Some NumPy platforms cast -inf to INT64_MIN and then back to half
        # -inf. Such a genuine historical score is indistinguishable in JSON.
        if np.array_equal(y, y.astype(np.int64).astype(y.dtype)):
            assert metric(y, y) == 1.
    path = tmp_path / "historical.json"
    report = search(path, scoring, X, X[:, 0]).report
    for labels, message in (([None, 1.], "nonfinite floating class as null"), ([.5, None], "original floating class labels"),
                            ([float(2**63), None], "original floating class labels"),
                            ([None, dict(_TAGGED_HALF, exponent=0)], "original floating class labels")):
        report["class_labels"] = report["identity"]["class_label_evidence"]["labels"] = labels
        seal(path, report)
        check_read(path, refused=message)
