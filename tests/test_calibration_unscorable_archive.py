"""Archived dtype observations bound the outcomes of the recorded sklearn scorer."""

import json
from hashlib import sha256
import warnings

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score

from supervised_nmf import evaluation


pytestmark = pytest.mark.skipif(sklearn.__version__ not in ("1.9.0", "1.9.1"),
                                reason="These scorer paths were audited for sklearn 1.9.0 and 1.9.1.")


class TextKey:
    """One dictionary key can match either encoding of an ASCII label."""

    def __init__(self, text):
        self.text = text

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if isinstance(other, TextKey):
            other = other.text
        if isinstance(other, bytes):
            other = other.decode()
        return self.text == other

    def __lt__(self, other):
        return self.text < other.text


def _dtype(code, cache):
    if cache is None:
        return np.dtype(code)
    keys = (np.array([TextKey("a"), TextKey("b")], dtype=object) if cache == "opaque"
            else np.array(["a", "b"]))
    return np.dtype(code, metadata={"unique": keys})


class LabelPredictor(ClassifierMixin, BaseEstimator):
    def __init__(self, code="U1", cache=None, column=False):
        self.code = code
        self.cache = cache
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        dtype = _dtype(self.code, self.cache)
        labels = [0, 0, 0, 1, 1, 1, 0, 1] if dtype.kind == "c" else list("aaabbbab")
        values = np.array(labels, dtype=dtype)[X[:, 0].astype(int)]
        return values[:, None] if self.column else values


def _write_search(path, model, target_code="U1", target_cache=None, scoring="balanced_accuracy"):
    y = np.array(list("aaaabbbb" * 2), dtype=_dtype(target_code, target_cache))
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(model, {}, np.tile(a, 2)[:, None], y, folds=[(a, b), (b, a)],
                                  cv=2, scoring=scoring, threads=1, refit=False, checkpoint_path=path)
    report = json.loads(path.read_bytes())
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
    return report


def _refuses(path, report, match):
    report["fingerprint"] = evaluation._digest(report["identity"])
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match=match):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("cache", [None, "native"])
def test_native_byte_target_cannot_support_any_recorded_score(tmp_path, cache):
    model = LabelPredictor(cache=cache)
    failed = _write_search(tmp_path / "byte.json", model, "S1", cache)
    assert failed["selected_candidate_index"] is None
    for fold in failed["candidates"][0]["folds"]:
        assert fold["status"] == "failed" and "score" not in fold and "labels represented as bytes" in fold["error"]
    successful = _write_search(tmp_path / "unicode.json", model, "U1", cache)
    assert successful["selected_candidate_index"] == 0
    assert all(("balanced_accuracy_confusion_rows" in fold) == (cache is not None)
               for fold in successful["candidates"][0]["folds"])
    # Keep the actual byte request, including its data hash. Substitute the
    # otherwise consistent winning outcome produced with Unicode truth.
    successful.update(identity=failed["identity"], class_labels=failed["class_labels"])
    _refuses(tmp_path / "false-winner.json", successful, "target dtype")
    # A convergence failure can carry an already-published score and keep
    # the candidate ineligible. Its scoring inputs must still agree.
    failed["candidates"][0]["folds"][0].update(score=.75, balanced_accuracy=.75)
    failed["candidates"][0]["folds"][0]["balanced_accuracy_target_cache"] = cache or "none"
    failed["candidates"][0]["folds"][0]["balanced_accuracy_prediction_kind"] = "U"
    failed["candidates"][0]["folds"][0]["balanced_accuracy_custom_scalar_inputs"] = "none"
    _refuses(tmp_path / "failed-after-score.json", failed, "target dtype")


@pytest.mark.parametrize(("code", "expected_error", "match"), [
    ("S1", "labels represented as bytes", "byte predictions"),
    ("c16", "Complex data not supported", "complex predictions"),
])
def test_prediction_dtype_contradicts_recorded_balanced_accuracy(tmp_path, code, expected_error, match):
    failed = _write_search(tmp_path / "failed.json", LabelPredictor(code, "native"), target_cache="native")
    assert failed["selected_candidate_index"] is None
    assert all("score" not in fold and expected_error in fold["error"] for fold in failed["candidates"][0]["folds"])
    successful = _write_search(tmp_path / "unicode.json", LabelPredictor("U1", "native"), target_cache="native")
    assert successful["selected_candidate_index"] == 0
    prediction = successful["candidates"][0]["folds"][0]["balanced_accuracy_confusion_rows"]["prediction"]
    prediction["dtype_kind"] = np.dtype(code).kind
    successful["candidates"][0]["folds"][0]["balanced_accuracy_prediction_kind"] = np.dtype(code).kind
    if code == "c16":
        del prediction["native_text"]
    _refuses(tmp_path / "false-winner.json", successful, match)


def test_plain_object_string_target_excludes_native_byte_predictions(tmp_path):
    failed = _write_search(tmp_path / "actual-bytes.json", LabelPredictor("S1"), target_code="O")
    assert failed["selected_candidate_index"] is None
    assert all("score" not in fold for fold in failed["candidates"][0]["folds"])
    successful = _write_search(tmp_path / "actual-text.json", LabelPredictor("U1"), target_code="O")
    assert successful["selected_candidate_index"] == 0
    for fold in successful["candidates"][0]["folds"]:
        assert fold["balanced_accuracy_target_plain_strings"] is True
        fold["balanced_accuracy_prediction_kind"] = "S"
        fold["balanced_accuracy_confusion_rows"]["prediction"]["dtype_kind"] = "S"
    _refuses(tmp_path / "false-bytes.json", successful, "byte")


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_complex_target_contradicts_an_ordinary_fold_score(tmp_path, scoring):
    report = _write_search(tmp_path / "original.json", LabelPredictor(), scoring=scoring)
    assert report["selected_candidate_index"] == 0
    assert all("balanced_accuracy_confusion_rows" not in fold for fold in report["candidates"][0]["folds"])
    evidence = report["identity"]["class_label_evidence"]
    evidence.pop("native_text", None)
    evidence.update(dtype_kind="c", labels=["0j", "(1+0j)"])
    report["class_labels"] = evidence["labels"]
    _refuses(tmp_path / "complex-target.json", report, "target dtype")


@pytest.mark.skipif(np.__version__ != "1.26.3", reason="This NumPy version preserves the supplied label caches.")
@pytest.mark.parametrize("column", [False, True])
def test_opaque_keys_can_score_column_byte_predictions(tmp_path, column):
    report = _write_search(tmp_path / "opaque.json", LabelPredictor("S1", "opaque", column), target_cache="opaque")
    folds = report["candidates"][0]["folds"]
    if column:
        assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == .75
        for fold in folds:
            evidence = fold["balanced_accuracy_confusion_rows"]
            assert evidence["rows"] == [[4, 3], [4, 3]]
            assert evidence["prediction"] == {"dtype_kind": "S", "native_text": True,
                                              "has_unique_cache": True, "opaque_unique_cache": True}
    else:
        assert report["selected_candidate_index"] is None
        assert all("score" not in fold and "labels represented as bytes" in fold["error"] for fold in folds)


class ReinterpretingKey(str):
    def __hash__(self):
        original = getattr(self, "original", None)
        if original is not None:
            self.original = None
            original.dtype = np.complex128
        return str.__hash__(self)


_EXTRA_LABEL_WARNING = "y_pred contains classes not in y_true"


class ReinterpretingPredictor(LabelPredictor):
    def __init__(self, callback):
        self.callback = callback

    def predict(self, X):
        if self.callback == "cache":
            first = ReinterpretingKey("a")
            keys = np.array([first, ReinterpretingKey("b")], dtype=object)
            values = list("aaabbbab")
        else:
            keys = np.array(["a", "b", "z"])
            values = list("aaabbbaz")
        dtype = np.dtype("U4", metadata={"unique": keys})
        predictions = np.array(values, dtype=dtype)[X[:, 0].astype(int)]
        if self.callback == "cache":
            first.original = predictions
        else:
            showwarning = warnings.showwarning

            def show(message, category, *args, **kwargs):
                showwarning(message, category, *args, **kwargs)
                # Only the scorer warning triggers the mutation, which may itself warn.
                if category is UserWarning and str(message) == _EXTRA_LABEL_WARNING:
                    predictions.dtype = np.complex128

            warnings.showwarning = show
        return predictions


@pytest.mark.parametrize("callback", ["cache", "warning"])
def test_confusion_rows_record_dtype_before_user_code_runs_in_scorer(tmp_path, callback):
    model = ReinterpretingPredictor(callback)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = model.predict(np.arange(8)[:, None])
        expected = balanced_accuracy_score(np.array(list("aaaabbbb")), predictions)
    assert predictions.dtype.kind == "c"
    report = _write_search(tmp_path / "callback.json", model)
    assert report["identity"]["balanced_accuracy_support_method"] == "confusion-matrix-v5"
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == expected
    assert expected == (.75 if callback == "cache" else .625)
    for fold in report["candidates"][0]["folds"]:
        pred = fold["balanced_accuracy_confusion_rows"]["prediction"]
        assert pred == {"dtype_kind": "U", "native_text": True, "has_unique_cache": True,
                        "opaque_unique_cache": callback == "cache"}
        assert [w["message"] for w in fold["warnings"]] == [str(w.message) for w in caught]
        if callback == "warning":
            index = fold["balanced_accuracy_confusion_rows"]["warning_index"]
            assert index is not None
            assert caught[index].category is UserWarning and str(caught[index].message) == _EXTRA_LABEL_WARNING

    # Both scores are attainable with four observations per class. A v1
    # complex dtype might record a genuine callback, or an impossible claim
    # of scoring complex inputs. The archive alone cannot distinguish them.
    report["identity"]["balanced_accuracy_support_method"] = "confusion-matrix-v1"
    for fold in report["candidates"][0]["folds"]:
        del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
        del fold["balanced_accuracy_custom_scalar_inputs"]
        pred = fold["balanced_accuracy_confusion_rows"]["prediction"]
        pred["dtype_kind"] = "c"
        del pred["native_text"]
    _refuses(tmp_path / "historical.json", report, "prediction dtype after scoring")
