"""Identifier type boundaries must precede prediction and calibration fits."""

from decimal import Decimal
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from supervised_nmf.evaluation import calibration_search, evaluate_classifier
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.validation import build_cv_folds, validate_metadata


def state(values):
    """Capture caller values, scalar types, and ndarray storage metadata."""
    if isinstance(values, np.ndarray):
        return (values.shape, values.strides, values.dtype.str,
                values.flags.writeable, values.tobytes(),
                tuple((type(value), repr(value)) for value in values.flat))
    return type(values), tuple((type(value), repr(value)) for value in values)


def container(value, kind):
    if kind == "list":
        return [value]
    if kind == "tuple":
        return (value,)
    result = np.asarray([value], dtype=object if kind == "object" else None)
    result.setflags(write=False)
    return result


@pytest.mark.parametrize("kind", ["list", "tuple", "native", "object"])
@pytest.mark.parametrize("value", [
    pytest.param(True, id="python-bool"),
    pytest.param(np.bool_(False), id="numpy-bool"),
    pytest.param(3 + 0j, id="python-complex-zero-imaginary"),
    pytest.param(np.complex64(3), id="complex64-zero-imaginary"),
    pytest.param(np.complex128(3), id="complex128-zero-imaginary"),
    pytest.param(np.clongdouble(3), id="clongdouble-zero-imaginary"),
    pytest.param(np.datetime64("2026-11-02", "ns"), id="datetime-nanoseconds"),
    pytest.param(np.timedelta64(3, "ns"), id="timedelta-nanoseconds"),
])
def test_boolean_complex_and_temporal_ids_rejected_without_mutation(value, kind):
    values = container(value, kind)
    before = state(values)
    # unique=False prevents duplicate booleans from accidentally passing this
    # regression by failing the unrelated uniqueness rule.
    with pytest.raises(ValueError, match="ids"):
        validate_metadata(values, 1, name="ids")
    assert state(values) == before


@pytest.mark.parametrize("scalar_type", [complex, np.complex64, np.complex128, np.clongdouble])
def test_nonzero_imaginary_ids_are_also_rejected(scalar_type):
    with pytest.raises(ValueError, match="ids"):
        validate_metadata([scalar_type(2 + 3j)], 1, name="ids")


@pytest.mark.parametrize("value", [
    pytest.param(Fraction(3, 2), id="fraction"),
    pytest.param(Decimal("1.5"), id="decimal"),
    pytest.param(b"sample", id="bytes"),
    pytest.param(object(), id="arbitrary-object"),
])
def test_unsupported_objects_raise_valueerror_instead_of_numpy_typeerror(value):
    values = np.asarray([value], dtype=object)
    before = state(values)
    with pytest.raises(ValueError, match="ids"):
        validate_metadata(values, 1, name="ids")
    assert state(values) == before


@pytest.mark.parametrize("value", [None, "", "   ", float("nan"),
                                   np.float32("inf"), np.longdouble("-inf")])
def test_missing_and_nonfinite_ids_remain_invalid(value):
    with pytest.raises(ValueError, match="ids"):
        validate_metadata([value], 1, name="ids")


@pytest.mark.parametrize("values", [
    pytest.param([-2, 0, 2**80, 2**80 + 1], id="python-integers-without-float-coercion"),
    pytest.param((np.int8(-2), np.int64(3), np.uint64(2**64 - 1)), id="numpy-integers"),
    pytest.param([0.25, np.float16(0.5), np.float32(0.75), np.float64(1.25),
                  np.longdouble("1.5")], id="floating-scalar-widths"),
    pytest.param(np.asarray([np.longdouble("1.25"), np.finfo(np.longdouble).max],
                            dtype=np.longdouble), id="finite-longdouble-array"),
    pytest.param(np.asarray([1.5, 2.5], dtype=np.dtype("f8").newbyteorder("S")),
                 id="nonnative-float-array"),
    pytest.param(["01", "1", "sample", np.str_("subject")], id="nonempty-strings"),
])
def test_supported_ids_preserve_values_and_input(values):
    if isinstance(values, np.ndarray):
        values.setflags(write=False)
    before = state(values)
    result = validate_metadata(values, len(values), name="ids", unique=True)
    assert result.shape == (len(values),)
    assert all(actual == expected for actual, expected in zip(result, values))
    assert state(values) == before


@pytest.mark.parametrize("values", [
    [1, 1.0], [np.uint64(12), np.longdouble(12)], [np.int64(-2), np.float32(-2)],
])
def test_numeric_equality_still_defines_duplicate_ids(values):
    before = state(values)
    with pytest.raises(ValueError, match="unique"):
        validate_metadata(values, 2, name="ids", unique=True)
    assert state(values) == before


def test_large_integers_remain_distinct_beside_floating_ids():
    values = [2**53, 2**53 + 1, float(2**53 + 2), 2**100 + 1]
    result = validate_metadata(values, len(values), name="ids", unique=True)
    assert len(set(result.tolist())) == 4
    assert result[1] == 2**53 + 1 and result[3] == 2**100 + 1
    assert isinstance(result[1], (int, np.integer))
    assert isinstance(result[3], (int, np.integer))


def test_adjacent_longdouble_ids_are_not_rounded_together():
    first = np.longdouble(1)
    second = np.nextafter(first, np.longdouble(2))
    values = np.asarray([first, second], dtype=np.longdouble)
    values.setflags(write=False)
    before = state(values)
    result = validate_metadata(values, 2, name="ids", unique=True)
    assert result[0] != result[1] and result[1] == second
    assert state(values) == before


def test_optional_and_mixed_family_rules_are_preserved():
    assert validate_metadata(None, 2, name="ids") is None
    with pytest.raises(ValueError, match="mix numeric and string"):
        validate_metadata([1, "1"], 2, name="ids")


@pytest.fixture
def data():
    y = np.tile([0, 1], 6)
    X = np.column_stack([np.arange(len(y)), y]).astype(float)
    first, second = np.arange(6), np.arange(6, 12)
    return X, y, [(first, second), (second, first)]


@pytest.fixture
def model(data):
    X, y, _ = data
    return DummyClassifier(strategy="prior").fit(X, y)


def forbid_predictions(monkeypatch, model):
    calls = []
    for name in ("predict", "predict_proba", "decision_function",
                 "predict_with_probabilities", "predict_with_decision_function"):
        call = Mock(side_effect=AssertionError("Prediction preceded metadata validation"))
        monkeypatch.setattr(model, name, call, raising=False)
        calls.append(call)
    return calls


def evaluation_metadata():
    return {"sample_ids": list(range(10, 22)), "groups": list(range(20, 32)),
            "train_sample_ids": list(range(100, 112)), "train_groups": list(range(200, 212))}


@pytest.mark.parametrize("name", ["sample_ids", "groups", "train_sample_ids", "train_groups"])
@pytest.mark.parametrize("bad", [True, np.bool_(True), np.complex64(3)])
def test_every_evaluation_id_argument_rejected_before_any_prediction(
        monkeypatch, model, data, name, bad):
    X, y, _ = data
    metadata = evaluation_metadata()
    # The remaining numeric IDs are unique and disjoint across splits. Invalid
    # scalar types must fail even when there is no independent overlap error.
    metadata[name][0] = bad
    before = {key: state(value) for key, value in metadata.items()}
    predictions = forbid_predictions(monkeypatch, model)
    with pytest.raises(ValueError, match=name):
        evaluate_classifier(model, X, y, **metadata)
    assert {key: state(value) for key, value in metadata.items()} == before
    for prediction in predictions:
        prediction.assert_not_called()


@pytest.mark.parametrize("name", ["sample_ids", "groups"])
def test_integer_and_longdouble_overlap_is_rejected_before_prediction(
        monkeypatch, model, data, name):
    X, y, _ = data
    metadata = evaluation_metadata()
    metadata[f"train_{name}"][0] = np.longdouble(metadata[name][0])
    predictions = forbid_predictions(monkeypatch, model)
    with pytest.raises(ValueError, match="must not overlap"):
        evaluate_classifier(model, X, y, **metadata)
    for prediction in predictions:
        prediction.assert_not_called()


@pytest.mark.parametrize("strings", [False, True])
def test_ordinary_evaluation_metadata_preserves_predictions(model, data, strings):
    X, y, _ = data
    direct = evaluate_classifier(model, X, y)
    metadata = evaluation_metadata()
    if strings:
        metadata = {key: [f"id-{value:04d}" for value in values]
                    for key, values in metadata.items()}
    else:
        metadata["train_sample_ids"] = np.asarray(metadata["train_sample_ids"], dtype=float) + 0.5
        metadata["groups"] = np.asarray(metadata["groups"], dtype=np.int64)
    before = {key: state(value) for key, value in metadata.items()}
    report = evaluate_classifier(model, X, y, **metadata)
    assert report["status"] == "valid"
    assert report["metrics"] == direct["metrics"]
    for key in ("predicted_labels", "positive_class_probability"):
        assert report["predictions"][key] == direct["predictions"][key]
    assert report["predictions"]["sample_ids"] == list(metadata["sample_ids"])
    assert report["support_diagnostics"]["sample_overlap_checked"]
    assert report["support_diagnostics"]["group_overlap_checked"]
    assert {key: state(value) for key, value in metadata.items()} == before


def calibrate(api, X, y, folds, tmp_path, **metadata):
    estimator = KNeighborsClassifier(n_neighbors=1)
    common = dict(cv=2, threads=1, folds=folds, **metadata)
    if api == "search":
        return calibration_search(estimator, {}, X, y, refit=False, **common).report
    return calibration_knn_grid(estimator, {}, X, y,
                                checkpoint_dir=tmp_path / "grid", **common)


@pytest.mark.parametrize("api", ["search", "knn_grid"])
@pytest.mark.parametrize("name", ["groups", "data_ids"])
@pytest.mark.parametrize("bad", [True, np.complex64(3)])
def test_calibration_metadata_rejected_before_fit(monkeypatch, data, tmp_path, api, name, bad):
    X, y, folds = data
    values = np.asarray([bad, *range(11, 22)], dtype=object)
    values.setflags(write=False)
    before = state(values)
    fit = Mock(side_effect=AssertionError("Fit preceded metadata validation"))
    monkeypatch.setattr(KNeighborsClassifier, "fit", fit)
    with pytest.raises(ValueError, match=name):
        calibrate(api, X, y, folds, tmp_path, **{name: values})
    fit.assert_not_called()
    assert state(values) == before
    assert not (tmp_path / "grid").exists()


@pytest.mark.parametrize("api", ["search", "knn_grid"])
@pytest.mark.parametrize("strings", [False, True])
def test_ordinary_calibration_ids_still_fit(data, tmp_path, api, strings):
    X, y, folds = data
    groups = np.repeat(np.arange(6), 2)
    ids = np.arange(12, dtype=float) + 0.5
    if strings:
        groups = np.asarray([f"subject-{value}" for value in groups])
        ids = np.asarray([f"sample-{index:03d}" for index in range(12)])
    groups.setflags(write=False)
    ids.setflags(write=False)
    before = state(groups), state(ids)
    report = calibrate(api, X, y, folds, tmp_path, groups=groups, data_ids=ids)
    assert report["candidates"][0]["valid"]
    assert (state(groups), state(ids)) == before


@pytest.mark.parametrize("bad", [True, np.complex64(3)])
def test_build_cv_folds_rejects_boolean_and_complex_groups(data, bad):
    X, y, folds = data
    groups = [bad, *range(11, 22)]
    before = state(groups)
    with pytest.raises(ValueError, match="groups"):
        build_cv_folds(X, y, cv=2, groups=groups, folds=folds)
    assert state(groups) == before
