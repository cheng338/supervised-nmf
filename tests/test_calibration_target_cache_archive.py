"""Read the target cache used by each scored fold, including fit-time changes."""

from copy import deepcopy
from hashlib import sha256
import json
import sys
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from supervised_nmf import evaluation


class AlternatingKey(np.void):
    comparisons = 0
    previous = None
    match = False

    @property
    def rank(self):
        return int.from_bytes(self.tobytes(), sys.byteorder) - ord("a")

    def __lt__(self, other):
        return self.rank < other.rank

    def __hash__(self):
        return hash("b" if self.rank == 2 else "a")

    def __eq__(self, other):
        AlternatingKey.comparisons += 1
        if isinstance(other, AlternatingKey):
            return self is other
        if not isinstance(other, str):
            return False
        if self.rank == 2:
            return other == "b"
        if other != "a":
            return False
        if self.rank == 1:
            return True
        if other is not AlternatingKey.previous:
            AlternatingKey.previous = other
            AlternatingKey.match = not AlternatingKey.match
        return AlternatingKey.match


class CacheChangingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, mode):
        self.mode = mode

    def fit(self, X, y):
        keys = y.dtype.metadata["unique"]
        if self.mode == "opaque":
            AlternatingKey.comparisons, AlternatingKey.previous, AlternatingKey.match = 0, None, False
            keys.dtype = np.dtype((AlternatingKey, "V4"))
        elif self.mode == "integer":
            keys.dtype = np.int32
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if self.mode == "integer":
            return X[:, 0] // 4
        return np.array(list("aaaabbbb"), dtype=object)[X[:, 0]]


def _truth(mode, repeats):
    if mode == "integer":
        keys = np.array([0, 1], dtype=np.int32).view("V4")
        return np.array(([0] * 4 + [1] * 4) * repeats, dtype=np.dtype("i4", metadata={"unique": keys}))
    keys = np.array(list("abc"), dtype="U1")
    return np.array(list("aaaabbbb" * repeats), dtype=np.dtype("U1", metadata={"unique": keys}))


def _search(path, mode):
    x = np.arange(8)[:, None]
    left, right = np.arange(8), np.arange(8, 16)
    return evaluation.calibration_search(CacheChangingClassifier(mode), {}, np.tile(x, (2, 1)), _truth(mode, 2),
                                         cv=2, folds=[(left, right), (right, left)], scoring="balanced_accuracy",
                                         threads=1, refit=False, checkpoint_path=path).report


def _write(path, report):
    report["fingerprint"] = evaluation._digest(report["identity"])
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


def _check_read(path, report, error=None):
    raw = path.read_bytes()
    for digest in (None, sha256(raw).hexdigest()):
        if error:
            with pytest.raises(ValueError, match=error):
                evaluation.load_calibration_report(path, expected_sha256=digest)
        else:
            assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
    assert path.read_bytes() == raw


@pytest.mark.parametrize("mode", ["control", "opaque", "integer"])
def test_fold_cache_can_differ_from_search_cache(tmp_path, mode):
    path = tmp_path / "cache.json"
    x = np.arange(8)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y = _truth(mode, 1)
        model = CacheChangingClassifier(mode).fit(x, y)
        matrix = confusion_matrix(y, model.predict(x))
        comparisons = AlternatingKey.comparisons
        y = _truth(mode, 1)
        model.fit(x, y)
        expected = float(balanced_accuracy_score(y, model.predict(x)))
    report = _search(path, mode)
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == expected
    identity = report["identity"]
    assert identity["balanced_accuracy_support_method"] == "confusion-matrix-v5"
    assert identity["class_label_evidence"]["opaque_unique_cache"] is (mode == "integer")
    assert identity["validation_class_counts"] == [[4, 4], [4, 4]]
    rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    if mode == "opaque":
        assert [n for n, _ in rows] == [2, 2, 4]
        assert expected in (float(np.mean([1., 0., 1.])), float(np.mean([1., .5, 1.])))
        # The final fold executed exactly as many equality calls as one
        # direct matrix. Replaying the lookup can change both rows and score.
        assert AlternatingKey.comparisons == comparisons > 0
    else:
        assert rows == [[4, 4], [4, 4]]
    for fold in report["candidates"][0]["folds"]:
        assert fold["balanced_accuracy_target_cache"] == ("opaque" if mode == "opaque" else "native")
        if mode == "integer":
            assert "balanced_accuracy_confusion_rows" not in fold
        else:
            assert fold["balanced_accuracy_confusion_rows"]["rows"] == rows
            assert fold["balanced_accuracy_confusion_rows"]["prediction"]["plain_string_values"] is True
    _check_read(path, report)

    if mode == "opaque":
        false = deepcopy(report)
        false["candidates"][0]["folds"][0]["balanced_accuracy_target_cache"] = "native"
        _write(path, false)
        _check_read(path, false, "native text class counts")
    elif mode == "integer":
        false = deepcopy(report)
        false["candidates"][0]["folds"][0]["balanced_accuracy_target_cache"] = "opaque"
        _write(path, false)
        _check_read(path, false, "Missing balanced accuracy confusion rows")

    for fold in report["candidates"][0]["folds"]:
        del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
        del fold["balanced_accuracy_custom_scalar_inputs"]
    for method in ("confusion-matrix-v3", "confusion-matrix-v2", "confusion-matrix-v1"):
        identity["balanced_accuracy_support_method"] = method
        if mode != "integer" and method == "confusion-matrix-v2":
            for fold in report["candidates"][0]["folds"]:
                del fold["balanced_accuracy_confusion_rows"]["prediction"]["plain_string_values"]
        _write(path, report)
        _check_read(path, report, None if mode == "control" else "target cache|object scalar types")


def test_target_cache_observation_is_required_only_after_a_score(tmp_path, monkeypatch):
    path = tmp_path / "failure.json"
    original = evaluation._nonconverged
    first = True

    def fail_after_scoring(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("failed after scoring")
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_nonconverged", fail_after_scoring)
    report = _search(path, "control")
    failed, valid = report["candidates"][0]["folds"]
    assert failed["status"] == "failed" and failed["score"] == 1.
    assert valid["status"] == "valid" and valid["score"] == 1.
    _check_read(path, report)
    for index in range(2):
        for invalid in (None, "unknown", False, {}):
            edited = deepcopy(report)
            fold = edited["candidates"][0]["folds"][index]
            if invalid is None:
                del fold["balanced_accuracy_target_cache"]
            else:
                fold["balanced_accuracy_target_cache"] = invalid
            _write(path, edited)
            _check_read(path, edited, "target cache")

    del failed["score"], failed["balanced_accuracy"], failed["balanced_accuracy_confusion_rows"]
    del failed["balanced_accuracy_prediction_kind"], failed["balanced_accuracy_custom_scalar_inputs"]
    _write(path, report)
    _check_read(path, report, "target cache")
    del failed["balanced_accuracy_target_cache"]
    _write(path, report)
    _check_read(path, report)


class InstallsCache(CacheChangingClassifier):
    private_dtype = None

    def fit(self, X, y):
        # Only this test's newly allocated descriptor may be updated. Keep
        # the text dtype, byte order and itemsize identical for existing views.
        if y.dtype is not InstallsCache.private_dtype or y.dtype.isbuiltin or y.dtype.str != np.dtype("U1").str:
            raise RuntimeError("Training did not preserve the private fixed-width Unicode descriptor.")
        if y.dtype.metadata is None:
            keys = np.array(["a", "b"], dtype=object)
            updated = np.dtype("U1", metadata={"unique": keys})
            y.dtype.__setstate__(updated.__reduce__()[2])
        self.classes_ = np.unique(y)
        return self


def test_private_uncached_descriptor_can_acquire_a_target_cache_in_fit(tmp_path, monkeypatch):
    common_dtype = np.dtype("U1")
    descriptor = common_dtype.newbyteorder("=")
    assert descriptor is not common_dtype and not descriptor.isbuiltin and descriptor.metadata is None
    target = np.array(list("aaaabbbb" * 2), dtype=descriptor)
    assert target.dtype is descriptor
    monkeypatch.setattr(InstallsCache, "private_dtype", descriptor)
    x = np.tile(np.arange(8), 2)[:, None]
    left, right = np.arange(8), np.arange(8, 16)
    path = tmp_path / "installed-cache.json"
    report = evaluation.calibration_search(InstallsCache("control"), {}, x, target, cv=2,
                                          folds=[(left, right), (right, left)], scoring="balanced_accuracy",
                                          threads=1, refit=False, checkpoint_path=path).report
    assert common_dtype.metadata is None
    assert report["identity"]["class_label_evidence"]["dtype_has_metadata"] is False
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == 1.
    for fold in report["candidates"][0]["folds"]:
        assert fold["balanced_accuracy_target_cache"] == "opaque"
        assert fold["balanced_accuracy_confusion_rows"]["rows"] == [[4, 4], [4, 4]]
    _check_read(path, report)
