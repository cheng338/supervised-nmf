"""Non-object NumPy scalars can execute caller code inside sklearn's label validation."""

from hashlib import sha256
import json
import warnings

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from supervised_nmf import evaluation


class TextBuffer(np.void):
    original = None
    probes = 0

    @property
    def __array__(self):
        # type_of_target probes the first void scalar for an array conversion.
        # The same four bytes then become a Unicode code point for comparison.
        TextBuffer.probes += 1
        if TextBuffer.original is not None:
            TextBuffer.original.dtype = np.dtype("U1")
            TextBuffer.original = None
        raise AttributeError("__array__")


class VoidPredictions(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        text = np.array(list("aaabbbab"), dtype="U1")[X[:, 0].astype(int)]
        result = text.view(np.dtype((TextBuffer, "V4")))
        TextBuffer.original = result
        return result


def test_custom_void_predictions_use_the_scorers_observed_matrix(tmp_path):
    x = np.arange(8)[:, None]
    y = np.array(list("aaaabbbb"))
    model = VoidPredictions()
    TextBuffer.probes = 0
    predictions = model.predict(x)
    assert predictions.dtype.kind == "V" and predictions.dtype.type is TextBuffer
    expected = balanced_accuracy_score(y, predictions)
    direct_probes = TextBuffer.probes
    assert expected == .75 and predictions.dtype.kind == "U" and direct_probes > 0

    path = tmp_path / "void.json"
    a, b = np.arange(8), np.arange(8, 16)
    evaluation.calibration_search(model, {}, np.tile(x, (2, 1)), np.tile(y, 2), cv=2, folds=[(a, b), (b, a)],
                                  scoring="balanced_accuracy", refit=False, checkpoint_path=path)
    assert TextBuffer.probes == 3 * direct_probes
    report = json.loads(path.read_bytes())
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == expected
    for fold in report["candidates"][0]["folds"]:
        assert fold["balanced_accuracy_custom_scalar_inputs"] == "prediction"
        assert fold["balanced_accuracy_prediction_kind"] == "V"
        assert fold["balanced_accuracy_confusion_rows"]["rows"] == [[4, 3], [4, 3]]

    del report["candidates"][0]["folds"][0]["balanced_accuracy_confusion_rows"]
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="Missing balanced accuracy confusion rows"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def _labels(ids, kind, cache=None):
    values = ids.astype(kind) if kind in ("i", "m8[D]", "M8[D]") else np.array(["a", "b"], dtype=kind)[ids]
    if cache is not None:
        keys = np.array([0, 1]) if cache == "numeric" else np.unique(values)
        if cache == "opaque":
            keys = keys.astype(object)
        elif keys.dtype.kind == "O":
            keys = keys.astype("U")
        values = values.view(np.dtype(values.dtype, metadata={"unique": keys}))
    return values


class TypedPredictions(ClassifierMixin, BaseEstimator):
    def __init__(self, kind, cache=None, column=False):
        self.kind = kind
        self.cache = cache
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        values = _labels(X[:, 0].astype(int), self.kind, self.cache)
        return values[:, None] if self.column else values


@pytest.mark.skipif(sklearn.__version__ not in ("1.9.0", "1.9.1"), reason="Audited sklearn target-type check.")
@pytest.mark.parametrize("text_kind", ["U1", "O"])
@pytest.mark.parametrize("text_target", [False, True])
@pytest.mark.parametrize("cache", [None, "native"])
def test_plain_string_and_numeric_inputs_cannot_score_with_ordinary_keys(tmp_path, text_kind, text_target, cache):
    ids = np.tile([0, 1], 8)
    a, b = np.arange(8), np.arange(8, 16)
    x = ids[:, None]

    def write(path, target_kind, pred_kind):
        evaluation.calibration_search(TypedPredictions(pred_kind, cache), {}, x, _labels(ids, target_kind, cache),
                                      cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy",
                                      refit=False, checkpoint_path=path)
        return evaluation.load_calibration_report(path)

    target_kind, pred_kind = (text_kind, "i") if text_target else ("i", text_kind)
    with pytest.raises(ValueError):
        balanced_accuracy_score(_labels(ids, target_kind, cache), _labels(ids, pred_kind, cache))
    failed = write(tmp_path / "failed.json", target_kind, pred_kind)
    assert failed["selected_candidate_index"] is None
    assert all(f["status"] == "failed" and "score" not in f for f in failed["candidates"][0]["folds"])
    path = tmp_path / "scored.json"
    report = write(path, text_kind, text_kind)
    assert report["selected_mean_score"] == 1.
    report.update(identity=failed["identity"], fingerprint=failed["fingerprint"], class_labels=failed["class_labels"])
    for fold in report["candidates"][0]["folds"]:
        if text_target:
            fold["balanced_accuracy_prediction_kind"] = "i"
            if "balanced_accuracy_confusion_rows" in fold:
                pred = fold["balanced_accuracy_confusion_rows"]["prediction"]
                pred["dtype_kind"] = "i"
                pred.pop("plain_string_values", None)
                pred.pop("native_text", None)
        else:
            fold.pop("balanced_accuracy_target_plain_strings", None)
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="mixed string"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.skipif(sklearn.__version__ not in ("1.9.0", "1.9.1"), reason="Audited sklearn target-type check.")
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("target_kind", ["m8[D]", "M8[D]"])
def test_sklearn_rejects_native_byte_predictions_for_temporal_truth(target_kind, column):
    ids = np.tile([0, 1], 8)
    truth = _labels(ids, target_kind)
    predictions = TypedPredictions("S1", column=column).predict(ids[:, None])
    for metric in (confusion_matrix, balanced_accuracy_score):
        with pytest.raises((TypeError, ValueError)):
            metric(truth, predictions)


@pytest.mark.skipif(sklearn.__version__ not in ("1.9.0", "1.9.1"), reason="Audited sklearn target-type check.")
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("cache", [None, "native", "numeric"])
def test_integer_truth_cannot_score_native_byte_predictions(tmp_path, cache, column):
    ids = np.tile([0, 1], 8)
    x = ids[:, None]
    truth = _labels(ids, "i")
    model = TypedPredictions("S1", cache, column)
    for metric in (confusion_matrix, balanced_accuracy_score):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                value = metric(truth, model.predict(x))
            except (TypeError, ValueError):
                continue
        # A stale numeric key list can drop every byte prediction.
        if metric is confusion_matrix:
            assert np.sum(value) == 0
        else:
            assert not np.isfinite(value)

    a, b = np.arange(8), np.arange(8, 16)

    def write(path, estimator):
        evaluation.calibration_search(estimator, {}, x, truth, cv=2, folds=[(a, b), (b, a)],
                                      scoring="balanced_accuracy", refit=False, checkpoint_path=path)
        report = json.loads(path.read_bytes())
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
        return report

    failed = write(tmp_path / "failed.json", model)
    assert failed["selected_candidate_index"] is None
    assert all("score" not in fold and fold["status"] == "failed" for fold in failed["candidates"][0]["folds"])
    # Obtain real scores and, for the cache controls, real matrix counts.
    path = tmp_path / "winning.json"
    report = write(path, TypedPredictions("i", "opaque" if cache else None, column))
    assert report["selected_mean_score"] == 1.
    for fold in report["candidates"][0]["folds"]:
        fold["balanced_accuracy_prediction_kind"] = "S"
        if cache:
            pred = fold["balanced_accuracy_confusion_rows"]["prediction"]
            pred.update(dtype_kind="S", native_text=True, has_unique_cache=True, opaque_unique_cache=False)
        else:
            assert "balanced_accuracy_confusion_rows" not in fold
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="byte predictions"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.skipif(sklearn.__version__ not in ("1.9.0", "1.9.1"), reason="Audited sklearn target-type check.")
@pytest.mark.parametrize("kind", ["m8[D]", "M8[D]"])
def test_string_truth_and_temporal_predictions_in_calibration(tmp_path, kind):
    ids = np.tile([0, 1], 8)
    x, truth = ids[:, None], _labels(ids, "U1")
    estimator = TypedPredictions(kind)
    for metric in (confusion_matrix, balanced_accuracy_score):
        with pytest.raises((TypeError, ValueError)):
            metric(truth, estimator.predict(x))

    a, b = np.arange(8), np.arange(8, 16)
    path = tmp_path / "temporal-predictions.json"
    options = dict(cv=2, folds=[(a, b), (b, a)], scoring="balanced_accuracy", refit=False)
    evaluation.calibration_search(estimator, {}, x, truth, **options, checkpoint_path=path)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        failed = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert failed["selected_candidate_index"] is None
        assert all(f["status"] == "failed" and "score" not in f for f in failed["candidates"][0]["folds"])

    path = tmp_path / "altered.json"
    evaluation.calibration_search(TypedPredictions("U1"), {}, x, truth, **options, checkpoint_path=path)
    report = evaluation.load_calibration_report(path)
    assert report["selected_mean_score"] == 1.
    for fold in report["candidates"][0]["folds"]:
        fold["balanced_accuracy_prediction_kind"] = np.dtype(kind).kind
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="mixed string"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
