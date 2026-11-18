"""Owned native metadata, exact split identity and bounded group counting."""

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from supervised_nmf import evaluation, validation


def exact(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    numerator, denominator = value.as_integer_ratio()
    return Fraction(int(numerator), int(denominator))


def state(values):
    if values is None:
        return None
    return (values.shape, values.strides, values.dtype.str, values.dtype.type,
            dict(values.dtype.metadata or {}), values.flags.writeable,
            tuple((type(v), exact(v), bool(v == 0 and np.signbit(v)))
                  if not isinstance(v, str) else (type(v), v) for v in values.flat))


def metadata(groups, training=None, *, y=None, classes=None, **options):
    if y is None:
        y = np.arange(len(groups)) % 2
    if classes is None:
        classes = np.asarray([0, 1])
    defaults = dict(sample_ids=None, groups=groups, train_sample_ids=None,
                    train_groups=training, train_y=None,
                    min_class_support=10, min_groups=5)
    defaults.update(options)
    return evaluation._evaluation_metadata(y, classes, **defaults)


def assert_counts(support, groups, training, y, classes):
    query = [exact(v) for v in groups]
    train = None if training is None else [exact(v) for v in training]
    assert support["group_count"] == len(set(query))
    assert support["training_group_count"] == (None if train is None else len(set(train)))
    assert support["class_counts"] == [int(sum(v == label for v in y)) for label in classes]
    assert support["per_class_group_count"] == [
        len({query[i] for i in range(len(y)) if y[i] == label}) for label in classes]


@pytest.fixture
def model():
    return DummyClassifier(strategy="prior").fit(np.arange(8).reshape(4, 2), [0, 1, 0, 1])


def public_report(model, groups, training=None):
    y = np.arange(len(groups)) % 2
    X = np.column_stack([np.arange(len(y)), y]).astype(float)
    result = evaluation.evaluate_classifier(model, X, y, threads=1,
                                            groups=groups, train_groups=training)
    assert result["status"] == "valid"
    assert_counts(result["support_diagnostics"], groups, training, y, [0, 1])
    # Dense codes are an independent exact partition, not package key output.
    keys = [exact(v) for v in groups]
    training_keys = [] if training is None else [exact(v) for v in training]
    ordered = sorted(set(keys + training_keys))
    codes = {value: index for index, value in enumerate(ordered)}
    reference = evaluation.evaluate_classifier(
        model, X, y, threads=1,
        groups=np.asarray([codes[v] for v in keys]),
        train_groups=None if training is None else np.asarray([codes[v] for v in training_keys]))
    assert result["metrics"] == reference["metrics"]
    assert result["predictions"] == reference["predictions"]
    assert result["support_diagnostics"] == reference["support_diagnostics"]
    return result


@pytest.mark.parametrize("total", [7, 8, 9])
@pytest.mark.parametrize("formats", ["same_native", "native_float", "native_integer"])
@pytest.mark.parametrize("reverse", [False, True])
def test_tiny_combined_size_preserves_exact_public_partitions(model, total, formats, reverse):
    n, nt = 3, total - 3
    if formats == "same_native":
        query = np.arange(1, n + 1, dtype=np.longdouble)
        training = np.arange(100, 100 + nt, dtype=np.longdouble)
    elif formats == "native_float":
        base = np.arange(1, n + 1, dtype=np.longdouble)
        query = np.nextafter(base, base + np.longdouble(1))
        training = np.arange(1, nt + 1, dtype=np.float64)
    else:
        query = np.asarray([2**53 + 2 * i for i in range(n)], dtype=np.int64).astype(np.longdouble)
        training = np.asarray([2**53 + 2 * i + 1 for i in range(nt)], dtype=np.int64)
    if reverse:
        query, training = training, query
    query.flags.writeable = training.flags.writeable = False
    before = state(query), state(training)
    assert not {exact(v) for v in query} & {exact(v) for v in training}
    public_report(model, query, training)
    assert (state(query), state(training)) == before


@pytest.mark.parametrize("size", [8, 9])
def test_query_only_size_boundary_preserves_public_counts(model, size):
    groups = np.asarray([1 + i % 3 for i in range(size)], dtype=np.longdouble)
    groups.flags.writeable = False
    before = state(groups)
    public_report(model, groups)
    assert state(groups) == before


def precision_pair(family, touching):
    if family == "float":
        training = np.arange(1, 6, dtype=np.float64)
        values = training.astype(np.longdouble)
        query = np.nextafter(values, values + np.longdouble(1))
        if touching:
            query[0] = np.longdouble(1)
    elif family == "int64":
        query = np.asarray([-2**53, 0, 2**53, 2**53 + 2, 2**53 + 4], dtype=np.int64).astype(np.longdouble)
        training = np.asarray([-2**53 - 1, -1, 2**53 + 1, 2**53 + 3, 2**53 + 5], dtype=np.int64)
        if touching:
            training[1] = 0
    else:
        query = np.asarray([-4, -3, -2, 0, 2**64], dtype=np.longdouble)
        training = np.asarray([2**64 - 1, 1, 3, 5, 7], dtype=np.uint64)
        if touching:
            training[1] = 0
    return query, training


@pytest.mark.parametrize("family", ["float", "int64", "uint64"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("touching", [False, True])
@pytest.mark.parametrize("role", ["samples", "groups"])
def test_native_mixed_overlap_matches_exact_ratios(family, reverse, touching, role):
    query, training = precision_pair(family, touching)
    if reverse:
        query, training = training, query
    query.flags.writeable = training.flags.writeable = False
    before = state(query), state(training)
    overlap = bool({exact(v) for v in query} & {exact(v) for v in training})
    assert overlap is touching
    if role == "samples":
        left, left_identity = validation._validated_unique_metadata(query, len(query), name="sample_ids")
        right, right_identity = validation._validated_unique_metadata(training, len(training), name="train_sample_ids")
        assert validation._metadata_overlap(left, left_identity, right, right_identity) is overlap
        assert [exact(v) for v in left] == [exact(v) for v in query]
        assert [exact(v) for v in right] == [exact(v) for v in training]
        assert not np.shares_memory(left, query) and not np.shares_memory(right, training)
    elif overlap:
        with pytest.raises(ValueError, match="^Training and evaluation groups must not overlap\\.$"):
            metadata(query, training)
    else:
        result = metadata(query, training)
        assert_counts(result[2], query, training, np.arange(len(query)) % 2, [0, 1])
        assert [exact(v) for v in result[1]] == [exact(v) for v in query]
    assert (state(query), state(training)) == before


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.longdouble])
@pytest.mark.parametrize("layout", ["strided_readonly", "swapped", "metadata"])
def test_validated_sample_identity_owns_all_comparison_values(dtype, layout):
    original = np.asarray([1, 2, 3], dtype=dtype)
    if layout == "strided_readonly":
        storage = np.empty(6, dtype=dtype)
        storage[::2], storage[1::2] = original, -10
        original = storage[::2]
        original.flags.writeable = False
    elif layout == "swapped":
        original = original.astype(original.dtype.newbyteorder())
    else:
        tagged = np.empty(original.shape, dtype=np.dtype(dtype, metadata={"identifier": "owned"}))
        np.copyto(tagged, original)
        original = tagged
        assert original.dtype.metadata is not None
    before = state(original)
    array, identity = validation._validated_unique_metadata(original, len(original), name="ids")
    assert state(original) == before and not np.shares_memory(array, original)
    assert [exact(v) for v in array] == [Fraction(1), Fraction(2), Fraction(3)]
    original.flags.writeable = True
    original[:] = 100
    other, other_identity = validation._validated_unique_metadata(np.asarray([2], dtype=np.longdouble), 1, name="other")
    assert validation._metadata_overlap(array, identity, other, other_identity)
    assert [exact(v) for v in array] == [Fraction(1), Fraction(2), Fraction(3)]


@pytest.mark.parametrize("role", ["sample_ids", "train_sample_ids", "groups", "train_groups"])
def test_evaluation_uses_captured_snapshot_after_caller_mutation(monkeypatch, role):
    supplied = dict(sample_ids=np.asarray([11, 12, 13], dtype=np.longdouble),
                    train_sample_ids=np.asarray([111, 112, 113, 114], dtype=np.int64),
                    groups=np.asarray([1, 2, 1], dtype=np.longdouble),
                    train_groups=np.asarray([101, 102, 103, 104], dtype=np.int64))
    old = {name: values.copy() for name, values in supplied.items()}
    capture = validation._validate_metadata_snapshot
    captured = []

    def capture_then_mutate(values, count, *, name, **kwargs):
        result = capture(values, count, name=name, **kwargs)
        if name == role:
            assert values is supplied[role]
            assert not np.shares_memory(result[0], values)
            assert result[1] is not None and not np.shares_memory(result[1], values)
            captured.append(name)
            values[:] = (11 if "sample_ids" in role else (101 if role == "groups" else 1))
        return result

    monkeypatch.setattr(validation, "_validate_metadata_snapshot", capture_then_mutate)
    monkeypatch.setattr(evaluation, "_validate_metadata_snapshot", capture_then_mutate)
    result = metadata(supplied["groups"], supplied["train_groups"],
                      sample_ids=supplied["sample_ids"], train_sample_ids=supplied["train_sample_ids"])
    assert captured == [role]
    assert [exact(v) for v in result[0]] == [exact(v) for v in old["sample_ids"]]
    assert [exact(v) for v in result[1]] == [exact(v) for v in old["groups"]]
    assert_counts(result[2], old["groups"], old["train_groups"], [0, 1, 0], [0, 1])


@pytest.mark.parametrize("pattern", ["uniform", "different_ends", "same_ends"])
@pytest.mark.parametrize("profile", ["all_zero", "imbalanced"])
def test_uniform_and_endpoint_middle_groups_keep_exact_class_partitions(pattern, profile):
    groups = np.zeros(257, dtype=np.longdouble)
    groups[::2] = np.longdouble("-0")
    if pattern == "different_ends":
        groups[-1] = 2
    elif pattern == "same_ends":
        groups[len(groups) // 2] = 2
    y = np.zeros(len(groups), dtype=np.int64)
    if profile == "imbalanced":
        y[-1] = 1
    groups.flags.writeable = y.flags.writeable = False
    before = state(groups), state(y)
    result = metadata(groups, y=y)
    assert_counts(result[2], groups, None, y, [0, 1])
    assert result[2]["class_counts"][1] == (0 if profile == "all_zero" else 1)
    assert (state(groups), state(y)) == before


@pytest.mark.parametrize("size", [0, 1])
def test_uniform_helper_preserves_empty_and_singleton_inverse_contract(size):
    values = np.full(size, np.longdouble("-0"), dtype=np.longdouble)
    values.flags.writeable = False
    before = state(values)
    unique, inverse = evaluation._unique_native_groups(values, return_inverse=True)
    assert [exact(v) for v in unique] == ([] if size == 0 else [Fraction(0)])
    assert inverse.shape == values.shape and inverse.dtype == np.dtype(np.intp)
    assert [exact(unique[int(i)]) for i in inverse] == [exact(v) for v in values]
    assert not np.shares_memory(unique, values) and state(values) == before
    if size:
        assert np.signbit(unique[0])


@pytest.mark.parametrize("layout", ["native", "strided", "swapped", "empty"])
def test_bulk_float_keys_have_builtin_scalar_class_after_equal_storage_cast(layout):
    values = np.asarray([np.longdouble("-0"), np.longdouble("0.5"), np.longdouble(2**53)])
    if layout == "strided":
        values = np.repeat(values, 2)[::2]
    elif layout == "swapped":
        values = values.astype(values.dtype.newbyteorder())
    elif layout == "empty":
        values = values[:0]
    values.flags.writeable = False
    before = state(values)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        keys = validation._native_float_key_set(values)
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        # In particular, Windows can equate storage dtypes while keeping the
        # longdouble scalar class. Returning those keys was the actual defect.
        assert isinstance(keys, set)
        assert all(type(value) is float for value in keys)
        assert {exact(v) for v in keys} == {exact(v) for v in values}
        if not len(values):
            assert keys == set() and keys is not None
    else:
        assert keys is None
    assert state(values) == before


@pytest.mark.parametrize("warm", [False, True])
def test_native_boundaries_keep_strict_policy_and_scalar_values(monkeypatch, warm):
    monkeypatch.setattr(validation, "_NATIVE_KEY_FORMAT", None)
    with np.errstate(all="ignore"):
        zero, one = np.longdouble(0), np.longdouble(1)
        values = np.asarray([zero, np.nextafter(zero, one), one,
                             np.nextafter(one, np.longdouble(2)), np.finfo(np.longdouble).max])
    values.flags.writeable = False
    expected = [exact(v) for v in values]
    assert len(set(expected)) == len(values)
    if warm:
        validation._native_key_format()
    before, error_before, filters_before = state(values), np.geterr(), warnings.filters[:]
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        active = np.geterr()
        array, identity = validation._validated_unique_metadata(values, len(values), name="ids")
        assert [exact(v) for v in array] == expected
        assert all(type(v) is np.longdouble for v in array)
        assert validation._metadata_overlap(array, identity, array, identity)
        assert np.geterr() == active
    assert state(values) == before and np.geterr() == error_before and warnings.filters == filters_before


@pytest.mark.parametrize("problem", ["sample_duplicate", "sample_overlap"])
def test_sample_errors_precede_group_overlap_and_prediction(monkeypatch, model, problem):
    sample_ids = np.asarray([11, 12, 13, 14], dtype=np.longdouble)
    train_ids = np.asarray([101, 102, 103, 104], dtype=np.int64)
    if problem == "sample_duplicate":
        sample_ids[1] = sample_ids[0]
        message = "sample_ids must contain unique IDs."
    else:
        train_ids[0] = 11
        message = "Training and evaluation sample IDs must not overlap."
    groups = np.asarray([1, 2, 1, 2], dtype=np.longdouble)
    calls = []
    for name in ("predict", "predict_proba"):
        call = Mock(side_effect=AssertionError("Prediction preceded metadata validation"))
        monkeypatch.setattr(model, name, call)
        calls.append(call)
    before = state(sample_ids), state(train_ids), state(groups)
    with pytest.raises(ValueError) as caught:
        evaluation.evaluate_classifier(model, np.arange(8).reshape(4, 2), np.asarray([0, 1, 0, 1]),
                                       threads=1, sample_ids=sample_ids, train_sample_ids=train_ids,
                                       groups=groups, train_groups=groups.copy())
    assert str(caught.value) == message
    assert (state(sample_ids), state(train_ids), state(groups)) == before
    for call in calls:
        call.assert_not_called()


def test_object_labels_keep_their_original_group_partition():
    groups = np.asarray([1, 2, 1, 3], dtype=np.longdouble)
    y = np.asarray(["a", "b", "a", "b"], dtype=object)
    classes = np.asarray(["a", "b"], dtype=object)
    before = state(groups), state(y), state(classes)
    result = metadata(groups, y=y, classes=classes)
    assert_counts(result[2], groups, None, y, classes)
    assert (state(groups), state(y), state(classes)) == before


def test_private_nonarray_comparison_preserves_error_and_callback_order():
    events = []

    class ScalarMaskLabels:
        def __len__(self):
            return 4

        def __eq__(self, label):
            events.append(int(label))
            return True

    # The original native route rejects this two-dimensional indexed result.
    # The tiny canonical route would instead fail while hashing a nested list.
    with pytest.raises(ValueError):
        metadata(np.ones(4, dtype=np.longdouble), y=ScalarMaskLabels())
    assert events == [0, 1, 0]


def test_unknown_float_representation_keeps_exact_fallback(monkeypatch):
    class NumpyProxy:
        def __getattr__(self, name):
            return getattr(np, name)

        def finfo(self, dtype):
            return SimpleNamespace(nmant=105, minexp=-1022, maxexp=1024)

    monkeypatch.setattr(validation, "np", NumpyProxy())
    assert validation._exact_metadata_promotion(
        np.dtype(np.longdouble), np.dtype(np.float64),
        (Fraction(1), Fraction(2)), (Fraction(1), Fraction(2))) is None
