"""Class-label domain checks preserve exact values and screening behavior."""

import pickle

import numpy as np
import pytest

from supervised_nmf import BinaryScreening
from supervised_nmf import _labels


@pytest.fixture
def data():
    codes = np.arange(24) % 2
    X = np.random.default_rng(731).uniform(.1, 2., (24, 9))
    X[codes == 1, :4] += 1.5
    return X, codes


def target(pair, codes, container):
    values = [pair[int(code)] for code in codes]
    if container == "list":
        return values
    if container == "tuple":
        return tuple(values)
    if container == "column":
        return [[value] for value in values]
    if container == "object":
        return np.array(values, dtype=object)
    if container == "numpy_str_object":
        return np.array([np.str_(value) for value in values], dtype=object)
    if container == "string_dtype":
        if not hasattr(np.dtypes, "StringDType"):
            pytest.skip("StringDType is an optional NumPy 2 representation.")
        return np.array(values, dtype=np.dtypes.StringDType())
    raise AssertionError(container)


@pytest.mark.parametrize("pair", [
    ("a", "a\0"), ("a\0", "b"), ("a\0", "a\0\0"),
    ("\0", "\0\0"), ("a\0b", "a\0b\0"),
])
@pytest.mark.parametrize("container", [
    "list", "tuple", "column", "object", "numpy_str_object", "string_dtype",
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trailing_nul_is_rejected_before_screening(data, pair, container, dtype):
    X, codes = data
    y = target(pair, codes, container)
    before = pickle.dumps(y)
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        BinaryScreening(screening="percentile", percentile=100).fit(X.astype(dtype), y)
    assert pickle.dumps(y) == before


@pytest.mark.parametrize("pair", [
    ("a", "b"), ("", "b"), ("a\0x", "b\0y"), ("\0a", "\0b"),
    ("café", "猫"), ("a\\0", "b\\0"),
])
@pytest.mark.parametrize("container", ["list", "object", "numpy_str_object", "string_dtype"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_strings_match_integer_partition(data, pair, container, dtype):
    X, codes = data
    X = X.astype(dtype)
    reference = BinaryScreening(screening="percentile", percentile=100).fit(X, codes)
    y = target(pair, codes, container)
    fitted = BinaryScreening(screening="percentile", percentile=100).fit(X, y)
    assert fitted.classes_.tolist() == list(pair)
    for name in ("t_scores_", "scores_", "support_", "class_counts_",
                 "scaled_pooled_variances_", "scaled_standard_errors_"):
        np.testing.assert_array_equal(getattr(fitted, name), getattr(reference, name))
    np.testing.assert_array_equal(fitted.transform(X), reference.transform(X))


def test_fixed_width_padding_cannot_reveal_original_terminal_nul():
    ordinary = np.array(["a", "b"], dtype="<U2")
    lost = np.array(["a\0", "b"], dtype="<U2")
    assert ordinary.dtype == lost.dtype
    assert ordinary.shape == lost.shape
    assert ordinary.tobytes() == lost.tobytes()
    np.testing.assert_array_equal(_labels._validate_class_labels(lost), ordinary)
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        _labels._validate_class_labels(["a\0", "b"], lost)


def test_numpy_string_scalar_is_inspected_without_lossy_str_conversion():
    value = np.str_("a\0")
    assert list(value) == ["a", "\0"]
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        _labels._validate_class_label(value)
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        _labels._validate_class_labels(np.array([value, "b"], dtype=object))


class MethodTrap(str):
    def endswith(self, *args, **kwargs):
        raise AssertionError("overridden endswith must not be dispatched")

    def __hash__(self):
        raise AssertionError("label hashing must not be dispatched")

    def __eq__(self, other):
        raise AssertionError("label equality must not be dispatched")

    def __str__(self):
        raise AssertionError("str conversion must not be dispatched")


def test_guard_does_not_call_overridden_string_methods():
    values = [MethodTrap("a\0"), MethodTrap("b")]
    validated = np.array(["a", "b"], dtype="U1")
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        _labels._validate_class_labels(values, validated)
    ordinary = [MethodTrap("a"), MethodTrap("b")]
    assert _labels._validate_class_labels(ordinary, validated) is validated


@pytest.mark.parametrize("container", [list, tuple])
def test_numeric_sequences_need_no_raw_string_scan(monkeypatch, container):
    raw = container([0, 1, 0, 1])
    y = np.asarray(raw)

    def forbidden(values):
        raise AssertionError("numeric target triggered raw string scan")

    monkeypatch.setattr(_labels, "_raw_terminal_nul", forbidden)
    assert _labels._validate_class_labels(raw, y, classes=np.unique(y)) is y


def test_existing_structural_error_precedes_label_domain(data):
    X, codes = data
    y = target(("a\0", "b"), codes, "list")
    X[0, 0] = -1.
    with pytest.raises(ValueError, match="Negative values in data"):
        BinaryScreening().fit(X, y)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        BinaryScreening().fit(np.ones((3, 2)), y)


@pytest.mark.parametrize("method", ["transform", "get_support", "get_feature_names_out"])
def test_old_screen_with_hazardous_classes_cannot_be_reused(data, method):
    X, y = data
    model = BinaryScreening(screening="percentile", percentile=100).fit(X, y)
    model.classes_ = np.array(["a", "a\0"], dtype=object)
    before = pickle.dumps(model)
    args = (X,) if method == "transform" else ()
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        getattr(model, method)(*args)
    assert pickle.dumps(model) == before
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        BinaryScreening()._fit_from_statistics(model)


def test_failed_refit_clears_previous_screen(data):
    X, codes = data
    model = BinaryScreening(screening="percentile", percentile=100).fit(X, codes)
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        model.fit(X, target(("a", "a\0"), codes, "object"))
    assert not any(name.endswith("_") for name in vars(model))
