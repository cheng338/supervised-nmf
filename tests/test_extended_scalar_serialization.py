"""Exact extended scalars and unchanged ordinary JSON/checkpoint identities."""

from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from supervised_nmf.evaluation import (
    _digest, _jsonable, data_fingerprint, save_data_manifest, write_json,
)
from supervised_nmf.study import _read_receipt, _write_receipt


TAG = "__supervised_nmf_scalar__"


def assert_exact_real(encoded, value):
    """Decode the public representation independently using rational arithmetic."""
    if not np.isfinite(value):
        assert encoded is None
        return
    assert set(encoded) == {TAG, "significand", "exponent"}
    assert encoded[TAG] == "longdouble-v1"
    assert type(encoded["significand"]) is str
    assert type(encoded["exponent"]) is int
    coefficient = int(encoded["significand"])
    decoded = Fraction(coefficient) * Fraction(2) ** encoded["exponent"]
    assert decoded == Fraction(*value.as_integer_ratio())
    if value == 0:
        assert encoded["significand"] == ("-0" if np.signbit(value) else "0")
        assert encoded["exponent"] == 0
    else:
        assert abs(coefficient) % 2 == 1


@pytest.mark.parametrize("value", [
    pytest.param(np.longdouble("0.1"), id="fractional"),
    pytest.param(np.longdouble("-1.5"), id="negative"),
    pytest.param(np.longdouble(0), id="positive-zero"),
    pytest.param(np.longdouble("-0"), id="negative-zero"),
    pytest.param(np.nextafter(np.longdouble(1), np.longdouble(2)), id="adjacent-to-one"),
    pytest.param(np.longdouble(np.finfo(np.longdouble).max), id="maximum"),
    pytest.param(np.longdouble(-np.finfo(np.longdouble).max), id="negative-maximum"),
    pytest.param(np.longdouble(np.finfo(np.longdouble).tiny), id="minimum-normal"),
    pytest.param(np.nextafter(np.longdouble(0), np.longdouble(1)), id="minimum-subnormal"),
])
def test_finite_longdouble_is_exact_canonical_and_json_stable(value):
    encoded = _jsonable(value)
    assert_exact_real(encoded, value)
    assert _jsonable(encoded) == encoded
    assert json.loads(json.dumps(encoded, allow_nan=False)) == encoded
    assert _digest(value) == _digest(encoded)


@pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
def test_nonfinite_longdouble_uses_existing_null_policy(text):
    assert _jsonable(np.longdouble(text)) is None


def test_adjacent_values_zero_signs_and_string_ids_have_distinct_encodings():
    first = np.longdouble(1)
    second = np.nextafter(first, np.longdouble(2))
    assert _jsonable(first) != _jsonable(second)
    assert _digest(first) != _digest(second)
    assert _jsonable(np.longdouble(0)) != _jsonable(np.longdouble("-0"))
    assert _digest(first) != _digest("1")
    # A serialized tag is intentionally stable when passed back through JSON
    # normalization; dictionary-shaped IDs are rejected by metadata validation.
    encoded = _jsonable(first)
    assert _digest(encoded) != _digest(json.dumps(encoded, sort_keys=True))


@pytest.mark.parametrize("kind", ["native", "object", "list", "tuple"])
def test_array_and_sequence_elements_preserve_extended_values(kind):
    values = [np.longdouble("0.1"), np.nextafter(np.longdouble(1), np.longdouble(2))]
    if kind in ("native", "object"):
        source = np.asarray(values, dtype=np.longdouble if kind == "native" else object)
        source.setflags(write=False)
    else:
        source = values if kind == "list" else tuple(values)
    encoded = _jsonable({"values": source})
    assert type(encoded["values"]) is list
    for actual, expected in zip(encoded["values"], values):
        assert_exact_real(actual, expected)
    assert _jsonable(encoded) == encoded


def test_zero_dimensional_native_array_keeps_scalar_shape_and_input():
    value = np.longdouble("0.1")
    source = np.asarray(value)
    source.setflags(write=False)
    before = source.tobytes()
    assert_exact_real(_jsonable(source), value)
    assert source.shape == () and not source.flags.writeable
    assert source.tobytes() == before


def test_strided_multidimensional_readonly_array_is_not_mutated():
    backing = np.arange(24, dtype=np.longdouble).reshape(4, 6)
    backing[0, 5] = np.longdouble("0.1")
    backing[2, 3] = np.nextafter(np.longdouble(1), np.longdouble(2))
    backing.setflags(write=False)
    source = backing[::2, ::-2]
    before = (source.shape, source.strides, source.dtype.str,
              source.flags.writeable, source.tobytes(), backing.tobytes())
    encoded = _jsonable(source)
    assert len(encoded) == 2 and all(len(row) == 3 for row in encoded)
    for actual_row, expected_row in zip(encoded, source):
        for actual, expected in zip(actual_row, expected_row):
            assert_exact_real(actual, expected)
    assert (source.shape, source.strides, source.dtype.str,
            source.flags.writeable, source.tobytes(), backing.tobytes()) == before
    assert not source.flags.writeable and not backing.flags.writeable


def test_structured_void_preserves_extended_field_and_ordinary_fields():
    value = np.longdouble("0.1")
    source = np.asarray([(value, "row", 3)],
                        dtype=[("id", np.longdouble), ("label", "U4"), ("count", "i4")])
    source.setflags(write=False)
    before = source.tobytes()
    encoded = _jsonable(source[0])
    assert type(encoded) is list and encoded[1:] == ["row", 3]
    assert_exact_real(encoded[0], value)
    assert _jsonable(source) == [encoded]
    assert source.tobytes() == before and not source.flags.writeable


def test_unstructured_void_keeps_ordinary_bytes_repr():
    raw = b"\x00\x01\xfe"
    value = np.void(raw)
    assert _jsonable(value) == repr(raw)
    assert _jsonable(np.asarray([value])) == [repr(raw)]


@pytest.mark.parametrize("significand,exponent", [
    (2**112 - 1, 16000), (-(2**112 - 1), 16000),
    (2**112 - 3, -16000), (-(2**112 - 3), -16000),
])
def test_encoder_integer_arithmetic_on_synthetic_wide_dyadic_ratios(significand, exponent):
    from supervised_nmf.evaluation import _extended_float_json

    # This tests the helper's integer arithmetic, not native longdouble range.
    # The scalar stores 1; only its rational-value interface is substituted.
    expected = Fraction(significand) * Fraction(2) ** exponent

    class SyntheticRatio(np.longdouble):
        def as_integer_ratio(self):
            return expected.numerator, expected.denominator

    value = SyntheticRatio(1)
    encoded = _extended_float_json(value)
    assert_exact_real(encoded, value)
    assert len(json.dumps(encoded, allow_nan=False)) < 256


def extended_complex(real, imag):
    # Component assignment avoids first rounding through a Python complex.
    value = np.zeros((), dtype=np.clongdouble)
    value.real[()] = real
    value.imag[()] = imag
    return value[()]


@pytest.mark.parametrize("real,imag", [
    (np.longdouble("0.1"), np.longdouble(np.finfo(np.longdouble).max)),
    (np.longdouble(0), np.longdouble("-0")),
    (np.longdouble("nan"), np.longdouble("0.5")),
    (np.longdouble("0.5"), np.longdouble("inf")),
    (np.longdouble("-inf"), np.longdouble("nan")),
])
def test_extended_complex_components_terminate_and_preserve_finite_values(real, imag):
    value = extended_complex(real, imag)
    encoded = _jsonable(value)
    assert set(encoded) == {TAG, "real", "imag"}
    assert encoded[TAG] == "clongdouble-v1"
    assert_exact_real(encoded["real"], real)
    assert_exact_real(encoded["imag"], imag)
    assert _jsonable(encoded) == encoded
    assert json.loads(json.dumps(encoded, allow_nan=False)) == encoded
    assert _digest(value) == _digest(encoded)


def test_remaining_unboxable_numpy_scalar_has_terminating_fallback():
    class UnboxableString(np.str_):
        def item(self):
            return self

    value = UnboxableString("example")
    assert _jsonable(value) == repr(value)


def extended_payload():
    return {"ids": np.asarray([
        np.longdouble("0.1"), np.nextafter(np.longdouble(1), np.longdouble(2)),
    ], dtype=object), "nested": (np.longdouble("-0"), {
        "complex": extended_complex(np.longdouble("0.5"), np.longdouble("-0.25")),
    })}


def test_write_json_and_manifest_roundtrip_are_idempotent(tmp_path):
    payload = extended_payload()
    expected = _jsonable(payload)
    path = tmp_path / "report.json"
    write_json(path, payload)
    original = path.read_bytes()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == expected
    assert _digest(payload) == _digest(loaded)
    write_json(path, loaded)
    assert path.read_bytes() == original
    manifest = tmp_path / "manifest.json"
    save_data_manifest(manifest, payload)
    save_data_manifest(manifest, payload)
    save_data_manifest(manifest, expected)
    changed = extended_payload()
    changed["ids"][0] = np.nextafter(changed["ids"][0], np.longdouble(1))
    with pytest.raises(ValueError, match="incompatible"):
        save_data_manifest(manifest, changed)


def test_receipt_digest_survives_json_and_detects_changed_exact_value(tmp_path):
    path = tmp_path / "receipt.json"
    payload = extended_payload()
    _write_receipt(path, payload)
    expected = _jsonable(payload)
    assert _read_receipt(path) == expected
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["sha256"] == _digest(payload) == _digest(expected)
    envelope["payload"]["ids"][0]["exponent"] += 1
    write_json(path, envelope)
    with pytest.raises(ValueError, match="integrity"):
        _read_receipt(path)


def test_object_data_fingerprint_tracks_extended_values_and_id_family():
    first = np.longdouble(1)
    second = np.nextafter(first, np.longdouble(2))
    X = np.asarray([[first], [np.longdouble(2)]], dtype=object)
    y = np.asarray(["a", "b"], dtype=object)
    original = data_fingerprint(X, y)
    assert data_fingerprint(X.copy(), y.copy()) == original
    changed = X.copy()
    changed[0, 0] = second
    assert data_fingerprint(changed, y) != original
    strings = np.asarray([["1"], ["2"]], dtype=object)
    assert data_fingerprint(strings, y) != original


def test_ordinary_scalar_json_and_digest_golden_remain_unchanged():
    payload = {
        "bool": np.bool_(True), "complex": [np.complex64(1 + 2j), complex(1, 2)],
        "float": np.float32(0.5), "ints": np.asarray([1, 2], dtype=np.int16),
        "nested": (None, np.str_("abc")), "nonfinite": [np.float64("inf"), float("nan")],
        "zero": -0.0,
    }
    golden = (b'{"bool": true, "complex": ["(1+2j)", "(1+2j)"], "float": 0.5, '
              b'"ints": [1, 2], "nested": [null, "abc"], "nonfinite": [null, null], '
              b'"zero": -0.0}')
    assert json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False).encode() == golden
    assert _digest(payload) == hashlib.sha256(golden).hexdigest()
    for value in (np.float16(0.5), np.float32(0.5), np.float64(0.5)):
        assert type(_jsonable(value)) is float
        assert _jsonable(value) == 0.5


def test_ordinary_object_data_fingerprint_golden_remains_unchanged():
    X = np.asarray([[1, "row"], [np.float32(0.5), None]], dtype=object)
    y = np.asarray(["a", "b"], dtype=object)
    golden = b'(2, 2)|O[[1, "row"], [0.5, null]](2,)|O["a", "b"]'
    assert data_fingerprint(X, y) == hashlib.sha256(golden).hexdigest()
