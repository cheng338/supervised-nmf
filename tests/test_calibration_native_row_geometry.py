"""Cached native labels still bound the true support of each confusion row."""

from copy import deepcopy
from hashlib import sha256
import json
import warnings

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from supervised_nmf import evaluation
from supervised_nmf._calibration_report import _check_confusion_rows


class UnsortedCacheTree(DecisionTreeClassifier):
    prediction_dtype = "U1"

    def predict(self, X):
        # An unordered cache omits the last predicted label.
        dtype = np.dtype(self.prediction_dtype, metadata={"unique": np.array(["b", "a"])})
        return np.array(["a", "b", "z"], dtype=dtype)[X[:, 0]]


class ObjectCacheTree(UnsortedCacheTree):
    prediction_dtype = object


class UncachedObjectTree(ObjectCacheTree):
    def predict(self, X):
        return np.array([np.str_("a"), "b", "z"], dtype=object)[X[:, 0]]


def _write_archive(path, report):
    report["fingerprint"] = evaluation._digest(report["identity"])
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


@pytest.mark.parametrize("first_count", [3, 4])
@pytest.mark.parametrize("tree", [UnsortedCacheTree, ObjectCacheTree, UncachedObjectTree])
def test_custom_tree_confusion_rows_cannot_change_the_winner(tmp_path, first_count, tree):
    path = tmp_path / "custom-rows.json"
    truth_dtype = np.dtype("U1", metadata={"unique": np.array(["b", "a"])})
    truth = np.repeat(np.array(["a", "b"], dtype=truth_dtype), [first_count, 8 - first_count])
    # Both candidates make the same predictions, including an out-of-cache
    # prediction for the last observation of the second class.
    x = np.array([0, 0, 0, 1, 1, 1, 0, 2])[:, None]
    left, right = np.arange(8), np.arange(8, 16)
    model = tree(random_state=3)
    evaluation.calibration_search(model, {"max_depth": [1, 2]}, np.tile(x, (2, 1)), np.tile(truth, 2),
                                  scoring="balanced_accuracy", folds=[(left, right), (right, left)], cv=2,
                                  refit=False, checkpoint_path=path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictions = model.predict(x)
        matrix = confusion_matrix(truth, predictions)
        expected = float(balanced_accuracy_score(truth, predictions))
    oracle_rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    if tree is UncachedObjectTree:
        assert [n for n, _ in oracle_rows] == [first_count, 8 - first_count]
    elif np.__version__ == "1.26.3":
        assert [n for n, _ in oracle_rows] == [first_count, 7 - first_count]
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        genuine = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert genuine["selected_candidate_index"] == 0 and genuine["selected_mean_score"] == expected
        for candidate in genuine["candidates"]:
            for fold in candidate["folds"]:
                witness = fold["balanced_accuracy_confusion_rows"]
                assert witness["rows"] == oracle_rows
                assert witness["prediction"]["dtype_kind"] == np.dtype(tree.prediction_dtype).kind
                pred = witness["prediction"]
                assert pred["has_unique_cache"] is (tree is not UncachedObjectTree) and not pred["opaque_unique_cache"]
                if tree.prediction_dtype is object:
                    assert pred["plain_string_values"] is True

    for rows in ([[5, 4], [3, 3]], [[1, 1], [3, 3], [4, 3]]):
        changed = deepcopy(genuine)
        challenger = changed["candidates"][1]
        score = float(np.mean([h / n for n, h in rows]))
        assert score > expected
        for fold in challenger["folds"]:
            fold["balanced_accuracy_confusion_rows"].update(rows=rows, matrix_size=len(rows), warning_index=None)
            fold["score"] = fold["balanced_accuracy"] = score
        challenger.update(mean_score=score, mean_balanced_accuracy=score, fold_score_std=0.)
        changed.update(selected_candidate_index=1, selected_parameters=challenger["parameters"],
                       selected_mean_score=score, refinement=evaluation.refinement_diagnostics(
                           [row["parameters"] for row in changed["candidates"]], challenger["parameters"]))
        _write_archive(path, changed)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="native text class counts"):
                evaluation.load_calibration_report(path, expected_sha256=digest)

        if tree is ObjectCacheTree:
            for candidate in changed["candidates"]:
                for fold in candidate["folds"]:
                    del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
                    del fold["balanced_accuracy_custom_scalar_inputs"]
                    del fold["balanced_accuracy_confusion_rows"]["prediction"]["plain_string_values"]
            for method in ("confusion-matrix-v1", "confusion-matrix-v2"):
                changed["identity"]["balanced_accuracy_support_method"] = method
                _write_archive(path, changed)
                for digest in (None, sha256(path.read_bytes()).hexdigest()):
                    with pytest.raises(ValueError, match="archive omitted the object scalar types"):
                        evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize(("target_cache", "prediction_cache", "counts", "dimension", "valid"), [
    (True, True, [2, 4], 3, True), (True, True, [5], 2, True), (True, True, [6], 2, False),
    (True, False, [3, 5], 3, True), (True, False, [5], 2, True), (True, False, [4], 2, False),
    (True, False, [2, 4], 3, False), (False, True, [3], 2, True), (False, True, [3], 1, False),
])
def test_custom_native_rows_respect_which_cache_can_omit_values(target_cache, prediction_cache, counts,
                                                                dimension, valid):
    target = dict(dtype_kind="U", native_text=True, has_unique_cache=target_cache, opaque_unique_cache=False)
    prediction = dict(target, has_unique_cache=prediction_cache)
    witness = dict(prediction=prediction, rows=[[n, n] for n in counts], matrix_size=dimension, warning_index=None)
    fold = dict(score=1., balanced_accuracy_confusion_rows=witness)
    if valid:
        _check_confusion_rows(fold, [3, 5], 0, target, False)
    else:
        with pytest.raises(ValueError, match="native text class counts"):
            _check_confusion_rows(fold, [3, 5], 0, target, False)


@pytest.mark.parametrize(("target_cache", "pred_cache", "rows", "dimension"), [
    (False, False, [[6, 6], [2, 2]], 2),
    (False, False, [[2, 2], [2, 2], [4, 4]], 3),
    (True, False, [[7, 7]], 4),
    (False, True, [[3, 3], [4, 4]], 5),
])
def test_plain_fold_with_unbound_request_classes(target_cache, pred_cache, rows, dimension):
    # Ordinary values in this fold alone do not identify its rows with the
    # two classes from the request. A cached list can also include unused keys.
    target = dict(dtype_kind="O", plain_string_values=False, has_unique_cache=target_cache, opaque_unique_cache=False)
    pred = dict(dtype_kind="U", native_text=True, has_unique_cache=pred_cache, opaque_unique_cache=False)
    witness = dict(rows=rows, matrix_size=dimension, warning_index=None, prediction=pred)
    fold = dict(score=1., balanced_accuracy_confusion_rows=witness, balanced_accuracy_target_plain_strings=True,
                balanced_accuracy_prediction_kind="U", balanced_accuracy_custom_scalar_inputs="none")
    _check_confusion_rows(fold, [3, 5], 0, target, False)


class _LookupKey:
    def __init__(self, rank):
        self.rank = rank
        self.previous_scalar = None
        self.match = False

    def __lt__(self, other):
        return self.rank < other.rank

    def __hash__(self):
        return hash("b" if self.rank == 2 else "a")

    def __eq__(self, other):
        if isinstance(other, _LookupKey):
            return self is other
        if not isinstance(other, str):
            return False
        if self.rank == 2:
            return other == "b"
        if other != "a":
            return False
        if self.rank == 1:
            return True
        # A dictionary can probe twice for one scalar. Alternate only
        # between distinct true 'a' samples, which arrive as fresh scalars.
        if other is not self.previous_scalar:
            self.previous_scalar = other
            self.match = not self.match
        return self.match


class _CacheChangingString(str):
    owner = None

    @property
    def __array__(self):
        if self.owner is not None:
            array, self.owner = self.owner, None
            keys = np.array([_LookupKey(i) for i in range(3)], dtype=object)
            array.dtype = np.dtype(object, metadata={"unique": keys})
        raise AttributeError("__array__")


class CacheChangingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, swap):
        self.swap = swap

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        dtype = np.dtype(object, metadata={"unique": np.array(["a", "b"])})
        values = np.array(list("aaaabbbb"), dtype=dtype)[X[:, 0]]
        if self.swap:
            scalar = _CacheChangingString(values[0])
            values[0] = scalar
            scalar.owner = values
        return values


@pytest.mark.parametrize("swap", [True, False])
def test_object_scalar_observation_preserves_executed_lookup(tmp_path, swap):
    if sklearn.__version__ not in ("1.9.0", "1.9.1") or np.__version__ not in ("1.26.3", "2.2.6"):
        pytest.skip("The scalar callback reproduces a specific sklearn lookup order.")
    path = tmp_path / "object-scalars.json"
    dtype = np.dtype("U1", metadata={"unique": np.array([], dtype="U1")})
    truth = np.array(list("aaaabbbb"), dtype=dtype)
    x = np.arange(8)[:, None]
    left, right = np.arange(8), np.arange(8, 16)
    model = CacheChangingClassifier(swap)
    matrix = confusion_matrix(truth, model.predict(x))
    expected = float(balanced_accuracy_score(truth, model.predict(x)))
    rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    assert [n for n, _ in rows] == ([2, 2, 4] if swap else [4, 4])
    result = evaluation.calibration_search(model, {}, np.tile(x, (2, 1)), np.tile(truth, 2), cv=2,
                                          folds=[(left, right), (right, left)], scoring="balanced_accuracy",
                                          threads=1, refit=False, checkpoint_path=path)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        report = evaluation.load_calibration_report(path, expected_sha256=digest)
        assert report == result.report
        assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == expected
        for fold in report["candidates"][0]["folds"]:
            witness = fold["balanced_accuracy_confusion_rows"]
            assert witness["rows"] == rows
            assert witness["prediction"] == dict(dtype_kind="O", has_unique_cache=True, opaque_unique_cache=False,
                                                 plain_string_values=not swap)

    for invalid in (None, 0, "False"):
        edited = deepcopy(report)
        prediction = edited["candidates"][0]["folds"][0]["balanced_accuracy_confusion_rows"]["prediction"]
        if invalid is None:
            del prediction["plain_string_values"]
        else:
            prediction["plain_string_values"] = invalid
        _write_archive(path, edited)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="object-string observation"):
                evaluation.load_calibration_report(path, expected_sha256=digest)

    for fold in report["candidates"][0]["folds"]:
        del fold["balanced_accuracy_target_cache"], fold["balanced_accuracy_prediction_kind"]
        del fold["balanced_accuracy_custom_scalar_inputs"]
        del fold["balanced_accuracy_confusion_rows"]["prediction"]["plain_string_values"]
    for method in ("confusion-matrix-v1", "confusion-matrix-v2"):
        report["identity"]["balanced_accuracy_support_method"] = method
        _write_archive(path, report)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            if swap:
                with pytest.raises(ValueError, match="archive omitted the object scalar types"):
                    evaluation.load_calibration_report(path, expected_sha256=digest)
            else:
                assert evaluation.load_calibration_report(path, expected_sha256=digest) == report


class PlainStringClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, prediction_dtype):
        self.prediction_dtype = prediction_dtype

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array(["a", "b", "z", "w"], dtype=self.prediction_dtype)[X[:, 0]]


@pytest.mark.parametrize(("truth_dtype", "prediction_dtype"), [
    ("O", "U1"), ("U1", "O"), ("O", "O"), ("T", "U1"), ("U1", "T"),
])
def test_uncached_exact_strings_require_full_support_and_distinct_errors(tmp_path, truth_dtype, prediction_dtype):
    if "T" in (truth_dtype, prediction_dtype) and not hasattr(np.dtypes, "StringDType"):
        pytest.skip("StringDType requires NumPy 2.")
    path = tmp_path / "plain-strings.json"
    x = np.array([0, 0, 2, 0, 1, 1, 1, 1])[:, None]
    truth = np.array(list("aaabbbbb"), dtype=truth_dtype)
    model = PlainStringClassifier(prediction_dtype)
    predictions = model.predict(x)
    normalized_truth = truth.astype(object) if truth_dtype == "T" else truth
    normalized_predictions = predictions.astype(object) if prediction_dtype == "T" else predictions
    matrix = confusion_matrix(normalized_truth, normalized_predictions)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected = balanced_accuracy_score(normalized_truth, normalized_predictions)
    left, right = np.arange(8), np.arange(8, 16)
    report = evaluation.calibration_search(model, {}, np.tile(x, (2, 1)), np.tile(truth, 2), cv=2,
                                          folds=[(left, right), (right, left)], scoring="balanced_accuracy",
                                          threads=1, refit=False, checkpoint_path=path).report
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == expected
    rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    assert rows == [[3, 2], [5, 4]]
    for fold in report["candidates"][0]["folds"]:
        assert fold["balanced_accuracy_prediction_kind"] == normalized_predictions.dtype.kind
        witness = fold["balanced_accuracy_confusion_rows"]
        assert witness["rows"] == rows and witness["matrix_size"] == len(matrix) == 3
        assert [w["message"] for w in fold["warnings"]] == [str(w.message) for w in caught]
        assert witness["warning_index"] == 0
        if normalized_truth.dtype.kind == "O":
            assert report["identity"]["class_label_evidence"]["plain_string_values"] is True
            assert fold["balanced_accuracy_target_plain_strings"] is True
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report

    for changed_rows, dimension in (([[3, 3]], 2), ([[2, 2], [5, 5]], 2),
                                    ([[3, 3], [5, 5]], 3), ([[3, 2], [5, 5]], 4)):
        altered = deepcopy(report)
        row = altered["candidates"][0]
        score = float(np.mean([h / n for n, h in changed_rows]))
        for fold in row["folds"]:
            fold["balanced_accuracy_confusion_rows"].update(rows=changed_rows, matrix_size=dimension, warning_index=None)
            fold["score"] = fold["balanced_accuracy"] = score
        row["mean_score"] = row["mean_balanced_accuracy"] = altered["selected_mean_score"] = score
        _write_archive(path, altered)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="class counts"):
                evaluation.load_calibration_report(path, expected_sha256=digest)

    for field in ("balanced_accuracy_confusion_rows", "balanced_accuracy_prediction_kind"):
        altered = deepcopy(report)
        del altered["candidates"][0]["folds"][0][field]
        _write_archive(path, altered)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="(?i)confusion row|prediction kind"):
                evaluation.load_calibration_report(path, expected_sha256=digest)
    if normalized_truth.dtype.kind == "O":
        for field in ("plain_string_values", "balanced_accuracy_target_plain_strings"):
            altered = deepcopy(report)
            observation = (altered["identity"]["class_label_evidence"] if field == "plain_string_values"
                           else altered["candidates"][0]["folds"][0])
            del observation[field]
            _write_archive(path, altered)
            for digest in (None, sha256(path.read_bytes()).hexdigest()):
                with pytest.raises(ValueError):
                    evaluation.load_calibration_report(path, expected_sha256=digest)


class _InstallingString(_CacheChangingString):
    @property
    def __array__(self):
        if self.owner is not None:
            empty = np.dtype("U1", metadata={"unique": np.array([], dtype="U1")})
            self.private_dtype.__setstate__(empty.__reduce__()[2])
        return super().__array__


class DisplayAsZero(np.str_):
    def item(self):
        return 0


@pytest.mark.parametrize("prediction_dtypes", [("U1", "U2"), ("O", "object")])
def test_object_target_observations_distinguish_request_and_scored_fold(tmp_path, prediction_dtypes):
    # The second validation fold has user-defined scalars whose serialized
    # display differs from the text used by the metric.
    left, right = np.arange(8), np.arange(8, 16)
    values = list("aaabbbbb")
    y = np.array(values + [DisplayAsZero(s) for s in values], dtype=object)
    # The object predictions use two distinct extra classes for the two
    # misses, attaining the largest possible matrix dimension of four.
    second_miss = 3 if prediction_dtypes[0] == "O" else 0
    x = np.tile(np.array([0, 0, 2, second_miss, 1, 1, 1, 1]), 2)[:, None]
    path = tmp_path / "mixed-object-target.json"
    model = PlainStringClassifier(prediction_dtypes[0])
    grid = {"prediction_dtype": list(prediction_dtypes)}
    report = evaluation.calibration_search(model, grid, x, y, cv=2, folds=[(right, left), (left, right)],
                                          scoring="balanced_accuracy", threads=1, refit=False,
                                          checkpoint_path=path).report
    assert report["identity"]["class_label_evidence"]["plain_string_values"] is False
    assert report["selected_candidate_index"] == 0
    for candidate in report["candidates"]:
        model = PlainStringClassifier(**candidate["parameters"])
        for indices, fold in zip((left, right), candidate["folds"]):
            assert fold["balanced_accuracy_target_plain_strings"] is (indices is left)
            assert fold["balanced_accuracy_target_cache"] == fold["balanced_accuracy_custom_scalar_inputs"] == "none"
            predicted = model.predict(x[indices])
            with pytest.warns(UserWarning, match="y_pred contains classes not in y_true"):
                score = balanced_accuracy_score(y[indices], predicted)
            assert fold["score"] == score
            witness = fold["balanced_accuracy_confusion_rows"]
            assert witness["prediction"]["has_unique_cache"] is False
            if predicted.dtype.kind == "O":
                assert witness["prediction"]["plain_string_values"] is True
            matrix = confusion_matrix(y[indices], predicted)
            assert witness["rows"] == [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
            assert witness["rows"] == [[3, 2], [5, 4]]
            assert witness["matrix_size"] == len(matrix) == (4 if second_miss == 3 else 3)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report

    for alteration in ("dimension", "lost sample"):
        changed = deepcopy(report)
        challenger = changed["candidates"][1]
        fold = challenger["folds"][0]
        witness = fold["balanced_accuracy_confusion_rows"]
        if alteration == "dimension":
            witness["matrix_size"] = 5
        else:
            # One miss can account for the third matrix column. It cannot
            # account for the disappearance of the eighth validation sample.
            witness.update(rows=[[3, 3], [4, 3]], matrix_size=3)
            fold["score"] = fold["balanced_accuracy"] = (1. + 3 / 4) / 2
            scores = [f["score"] for f in challenger["folds"]]
            mean = float(np.mean(scores))
            assert mean > report["selected_mean_score"]
            challenger.update(mean_score=mean, mean_balanced_accuracy=mean, fold_score_std=float(np.std(scores, ddof=1)))
            changed.update(selected_candidate_index=1, selected_parameters=challenger["parameters"],
                           selected_mean_score=mean, refinement=evaluation.refinement_diagnostics(
                               [row["parameters"] for row in changed["candidates"]], challenger["parameters"]))
        _write_archive(path, changed)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            with pytest.raises(ValueError, match="Uncached string confusion rows"):
                evaluation.load_calibration_report(path, expected_sha256=digest)


def test_uncached_plain_object_stock_tree_has_only_its_two_fitted_classes(tmp_path):
    if sklearn.__version__ not in ("1.9.0", "1.9.1"):
        pytest.skip("The archived stock tree implementation is verified on these releases.")
    path = tmp_path / "stock-object.json"
    y = np.array(list("aaabbbbb" * 2), dtype=object)
    left, right = np.arange(8), np.arange(8, 16)
    report = evaluation.calibration_search(DecisionTreeClassifier(random_state=7), {}, np.zeros((16, 1)), y,
                                          cv=2, folds=[(left, right), (right, left)], scoring="balanced_accuracy",
                                          threads=1, refit=False, checkpoint_path=path).report
    assert report["selected_mean_score"] == .5
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
    changed = deepcopy(report)
    changed["candidates"][0]["folds"][0]["balanced_accuracy_confusion_rows"]["prediction"]["plain_string_values"] = False
    _write_archive(path, changed)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="plain string classes of the stock tree"):
            evaluation.load_calibration_report(path, expected_sha256=digest)
    for fold in report["candidates"][0]["folds"]:
        witness = fold["balanced_accuracy_confusion_rows"]
        assert witness["rows"] == [[3, 0], [5, 5]] and witness["matrix_size"] == 2
        witness["matrix_size"] = 3
    _write_archive(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="class counts|stock tree"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


def _stock_string_report(path, tree, dtype):
    if sklearn.__version__ not in ("1.9.0", "1.9.1"):
        pytest.skip("The archived stock tree implementation is verified on these releases.")
    y = np.array(list("aaabbbbb" * 2), dtype=dtype)
    x = np.zeros((len(y), 1))
    a, b = np.arange(8), np.arange(8, 16)
    report = evaluation.calibration_search(tree(random_state=7), {"max_depth": [1, 2]}, x, y, cv=2,
                                           folds=[(a, b), (b, a)], scoring="balanced_accuracy", threads=1,
                                           refit=False, checkpoint_path=path).report
    fitted = tree(random_state=7, max_depth=2).fit(x[a], y[a])
    predictions = fitted.predict(x[b])
    assert fitted.classes_.tolist() == ["a", "b"] and predictions.tolist() == ["b"] * 8
    return report, y[b], predictions


@pytest.mark.parametrize("tree", [DecisionTreeClassifier, ExtraTreeClassifier])
def test_stock_tree_prediction_cache_cannot_remove_validation_rows(tmp_path, tree):
    path = tmp_path / "prediction-cache.json"
    report, truth, predictions = _stock_string_report(path, tree, object)
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == .5
    challenger = report["candidates"][1]
    # A key for an unused class can add an empty row. Even a cache missing
    # the only predicted value still leaves that key available from truth.
    for keys, dimension in ((["z", "b"], 3), ([], 2)):
        dtype = np.dtype(object, metadata={"unique": np.array(keys, dtype="U1")})
        cached = np.array(predictions.tolist(), dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            matrix = confusion_matrix(truth, cached)
            assert balanced_accuracy_score(truth, cached) == .5
        assert len(matrix) == dimension
        rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
        assert rows == [[3, 0], [5, 5]]
        for fold in challenger["folds"]:
            assert fold["balanced_accuracy_target_cache"] == "none"
            witness = fold["balanced_accuracy_confusion_rows"]
            assert witness["rows"] == rows
            witness["prediction"]["has_unique_cache"] = True
            witness["matrix_size"] = dimension
        _write_archive(path, report)
        for digest in (None, sha256(path.read_bytes()).hexdigest()):
            assert evaluation.load_calibration_report(path, expected_sha256=digest) == report

    # Drop the three misclassifications and recompute every affected summary.
    # The resulting winner is still impossible for these stock predictions.
    for fold in challenger["folds"]:
        fold["balanced_accuracy_confusion_rows"]["rows"] = [[5, 5]]
        fold["score"] = fold["balanced_accuracy"] = 1.
    challenger.update(mean_score=1., mean_balanced_accuracy=1.)
    report.update(selected_candidate_index=1, selected_parameters=challenger["parameters"], selected_mean_score=1.,
                  refinement=evaluation.refinement_diagnostics([c["parameters"] for c in report["candidates"]],
                                                              challenger["parameters"]))
    _write_archive(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="classes of the stock tree"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("tree", [DecisionTreeClassifier, ExtraTreeClassifier])
def test_stock_tree_target_cache_can_remove_a_validation_class(tmp_path, tree):
    path = tmp_path / "target-cache.json"
    dtype = np.dtype(object, metadata={"unique": np.array(["b"])})
    report, truth, predictions = _stock_string_report(path, tree, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert confusion_matrix(truth, predictions).tolist() == [[5]]
        assert balanced_accuracy_score(truth, predictions) == 1.
    for candidate in report["candidates"]:
        for fold in candidate["folds"]:
            assert fold["balanced_accuracy_target_cache"] == "native"
            assert fold["balanced_accuracy_confusion_rows"]["rows"] == [[5, 5]]
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == 1.
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report


class UncachedLookupClassifier(CacheChangingClassifier):
    private_dtype = None

    def fit(self, X, y):
        if y.dtype is not UncachedLookupClassifier.private_dtype or y.dtype.isbuiltin:
            raise RuntimeError("Expected the private Unicode descriptor.")
        if y.dtype.metadata and "unique" in y.dtype.metadata:
            clean = np.dtype("U1", metadata={})
            y.dtype.__setstate__(clean.__reduce__()[2])
        return super().fit(X, y)

    def predict(self, X):
        values = np.array(list("aaaabbbb"), dtype=object)[X[:, 0]]
        if self.swap:
            scalar = _InstallingString(values[0])
            values[0] = scalar
            scalar.private_dtype = UncachedLookupClassifier.private_dtype
            scalar.owner = values
        return values


@pytest.mark.parametrize("install", [True, False])
def test_uncached_object_lookup_records_the_scorer_matrix(tmp_path, monkeypatch, install):
    if sklearn.__version__ not in ("1.9.0", "1.9.1") or np.__version__ not in ("1.26.3", "2.2.6"):
        pytest.skip("The callback reproduces a specific scorer lookup order.")

    def private_truth(repeats):
        common = np.dtype("U1")
        descriptor = common.newbyteorder("=")
        assert descriptor is not common and not descriptor.isbuiltin and descriptor.metadata is None
        values = np.array(list("aaaabbbb" * repeats), dtype=descriptor)
        assert values.dtype is descriptor
        monkeypatch.setattr(UncachedLookupClassifier, "private_dtype", descriptor)
        return values

    path = tmp_path / "uncached-lookup.json"
    model = UncachedLookupClassifier(install)
    x = np.arange(8)[:, None]
    left, right = np.arange(8), np.arange(8, 16)
    truth = private_truth(1)
    matrix = confusion_matrix(truth, model.fit(x, truth).predict(x))
    truth = private_truth(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = float(balanced_accuracy_score(truth, model.fit(x, truth).predict(x)))
    rows = [[int(n), int(h)] for n, h in zip(matrix.sum(axis=1), matrix.diagonal()) if n]
    assert [n for n, _ in rows] == ([2, 2, 4] if install else [4, 4])
    assert score == (float(np.mean([.5, 1., 1.])) if install else 1.)
    report = evaluation.calibration_search(model, {}, np.tile(x, (2, 1)), private_truth(2), cv=2,
                                          folds=[(left, right), (right, left)], scoring="balanced_accuracy",
                                          threads=1, refit=False, checkpoint_path=path).report
    assert report["selected_candidate_index"] == 0 and report["selected_mean_score"] == score
    for fold in report["candidates"][0]["folds"]:
        assert fold["balanced_accuracy_target_cache"] == "none" and fold["balanced_accuracy_prediction_kind"] == "O"
        witness = fold["balanced_accuracy_confusion_rows"]
        assert witness["rows"] == rows and witness["matrix_size"] == len(matrix)
        assert witness["prediction"] == dict(dtype_kind="O", has_unique_cache=False, opaque_unique_cache=False,
                                             plain_string_values=not install)
        assert [w["message"] for w in fold["warnings"]] == [str(w.message) for w in caught]
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        assert evaluation.load_calibration_report(path, expected_sha256=digest) == report

    # V4 had no prediction kind or matrix on this uncached path.
    report["identity"]["balanced_accuracy_support_method"] = "confusion-matrix-v4"
    for fold in report["candidates"][0]["folds"]:
        del fold["balanced_accuracy_prediction_kind"], fold["balanced_accuracy_confusion_rows"]
        del fold["balanced_accuracy_custom_scalar_inputs"]
    _write_archive(path, report)
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        if install:
            with pytest.raises(ValueError, match="(?i)balanced accuracy|historical"):
                evaluation.load_calibration_report(path, expected_sha256=digest)
        else:
            assert evaluation.load_calibration_report(path, expected_sha256=digest) == report
