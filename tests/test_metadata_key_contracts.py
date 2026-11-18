"""Exact identifier keys, owned validation and deferred native-format setup."""
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

from supervised_nmf import validation


@pytest.fixture(autouse=True)
def isolated_native_key_cache(monkeypatch):
    monkeypatch.setattr(validation, "_NATIVE_KEY_FORMAT", None)


@pytest.mark.parametrize("case", ["nonfinite", "duplicate"])
def test_returned_snapshot_is_checked_after_the_caller_changes_input(monkeypatch, case):
    original = np.asarray([np.nan, 1.5] if case == "nonfinite" else [1.5, 1.5])
    real_asarray = np.asarray
    captured = []

    def convert_then_change_input(value, *args, **kwargs):
        result = real_asarray(value, *args, **kwargs)
        if kwargs.get("dtype") is object and isinstance(value, np.ndarray):
            captured.append(result.copy())
            original[:] = [0.5, 1.5]
        return result

    monkeypatch.setattr(np, "asarray", convert_then_change_input)
    with pytest.raises(ValueError, match="nonfinite" if case == "nonfinite" else "unique"):
        validation.validate_metadata(original, 2, name="ids", unique=True)
    assert captured
    np.testing.assert_array_equal(original, [0.5, 1.5])
    if case == "nonfinite":
        assert np.isnan(captured[0][0])
    else:
        assert captured[0][0] == captured[0][1]


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_captured_temporal_dtype_is_rejected_before_object_conversion(monkeypatch, dtype):
    copies = []

    class ChangingCopy(np.ndarray):
        def copy(self, *args, **kwargs):
            copies.append(dtype)
            return np.asarray([1, 2], dtype=dtype)

    class NumpyProxy:
        ndarray = ChangingCopy

        def __getattr__(self, name):
            return getattr(np, name)

    # Model a dtype change at the copy boundary without scheduling a race.
    values = np.asarray([1.0, 2.0]).view(ChangingCopy)
    monkeypatch.setattr(validation, "np", NumpyProxy())
    with pytest.raises(ValueError, match="temporal IDs are not supported"):
        validation.validate_metadata(values, 2, name="ids", unique=True)
    assert copies == [dtype]


@pytest.mark.parametrize("kind", ["int", "uint", "float32", "float64", "unicode",
                                  "python_int", "python_float", "numpy_object"])
def test_ordinary_metadata_does_not_discover_native_format(monkeypatch, kind):
    values = {
        "int": np.asarray([1, 2], dtype=np.int64),
        "uint": np.asarray([1, 2], dtype=np.uint64),
        "float32": np.asarray([0.5, 1.5], dtype=np.float32),
        "float64": np.asarray([0.5, 1.5], dtype=np.float64),
        "unicode": np.asarray(["left", "right"]),
        "python_int": [1, 2], "python_float": [0.5, 1.5],
        "numpy_object": np.asarray([np.int64(1), np.float64(2)], dtype=object),
    }[kind]

    def forbidden():
        raise AssertionError("Ordinary metadata reached native-format discovery")

    monkeypatch.setattr(validation, "_native_key_format", forbidden)
    result = validation.validate_metadata(values, 2, name="ids", unique=True)
    assert len(set(validation._metadata_keys(result))) == 2
    assert validation._NATIVE_KEY_FORMAT is None


def test_import_does_not_discover_native_format(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import eagerly probed native floating format")

    with monkeypatch.context() as patch:
        patch.setattr(np, "finfo", forbidden)
        importlib.reload(validation)
    assert validation._NATIVE_KEY_FORMAT is None


@pytest.mark.parametrize("ratio", [(1, 3), (2, 6), (-1, 3), (1, -3), (4, 8), (12, 1)])
def test_scalar_subclass_preserves_rational_identity_without_discovery(monkeypatch, ratio):
    class RatioOverride(np.longdouble):
        def as_integer_ratio(self):
            return ratio

    def forbidden():
        raise AssertionError("Scalar subclass reached native-only discovery")

    monkeypatch.setattr(validation, "_native_key_format", forbidden)
    scalar = RatioOverride("7.25")
    expected = Fraction(*ratio)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        values = np.asarray([scalar], dtype=object)
        result = validation.validate_metadata(values, 1, name="ids", unique=True)
        key = validation._metadata_keys(result)[0]
    assert key == expected and hash(key) == hash(expected)
    assert result[0] is scalar and validation._NATIVE_KEY_FORMAT is None


def test_truthful_nonreduced_scalar_ratio_keeps_duplicates():
    class EquivalentRatio(np.longdouble):
        def as_integer_ratio(self):
            numerator, denominator = super().as_integer_ratio()
            return 3 * numerator, 3 * denominator

    scalar = EquivalentRatio("7.25")
    values = np.asarray([scalar, 7.25], dtype=object)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        keys = validation._metadata_keys(values)
        assert keys[0] == keys[1] == Fraction(29, 4)
        assert hash(keys[0]) == hash(keys[1])
        with pytest.raises(ValueError, match="unique"):
            validation.validate_metadata(values, 2, name="ids", unique=True)


def test_invalid_scalar_ratio_preserves_its_error():
    class InvalidRatio(np.longdouble):
        def as_integer_ratio(self):
            return 1, 0

    with pytest.raises(ZeroDivisionError):
        validation._metadata_keys(np.asarray([InvalidRatio("7.25")], dtype=object))


def test_successful_native_discovery_is_reused(monkeypatch):
    real_finfo = np.finfo
    calls = []

    def observed_finfo(dtype):
        calls.append(dtype)
        return real_finfo(dtype)

    monkeypatch.setattr(np, "finfo", observed_finfo)
    values = np.asarray([np.longdouble("0.5"), np.longdouble("7.25")] * 8, dtype=object)
    first = validation._metadata_keys(values)
    cache = validation._NATIVE_KEY_FORMAT
    second = validation._metadata_keys(values)
    assert list(first) == list(second)
    assert calls == [np.longdouble]
    assert cache is not None and validation._NATIVE_KEY_FORMAT is cache


@pytest.mark.parametrize("failure", ["call", "attribute"])
def test_failed_discovery_does_not_publish_partial_state(monkeypatch, failure):
    real_finfo = np.finfo

    class PartialInfo:
        nmant = 0

        @property
        def maxexp(self):
            raise RuntimeError("partial-format-probe")

    def broken(dtype):
        assert validation._NATIVE_KEY_FORMAT is None
        if failure == "call":
            raise RuntimeError("format-probe")
        return PartialInfo()

    values = np.asarray([np.longdouble("0.5")], dtype=object)
    before_error, before_filters = np.geterr(), list(warnings.filters)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        active = np.geterr()
        monkeypatch.setattr(np, "finfo", broken)
        with pytest.raises(RuntimeError, match="format-probe"):
            validation._metadata_keys(values)
        assert validation._NATIVE_KEY_FORMAT is None and np.geterr() == active
        monkeypatch.setattr(np, "finfo", real_finfo)
        assert validation._metadata_keys(values)[0] == Fraction(1, 2)
        assert validation._NATIVE_KEY_FORMAT is not None and np.geterr() == active
    assert np.geterr() == before_error and warnings.filters == before_filters


@pytest.mark.skipif(np.finfo(np.longdouble).nmant + 1 <= sys.float_info.mant_dig,
                    reason="Requires a native format wider than Python float")
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("offset", [-1, 1])
@pytest.mark.parametrize("extra_exponent", [None, 0, 1])
def test_exact_keys_across_precision_and_subnormal_boundaries(direction, offset, extra_exponent):
    precision = sys.float_info.mant_dig
    numerator = direction * ((1 << precision) + offset)
    exponent = 1 if extra_exponent is None else precision - sys.float_info.min_exp + extra_exponent
    expected = Fraction(numerator, 1 << exponent)
    value = np.ldexp(np.longdouble(numerator), -exponent)
    adjacent = np.nextafter(value, np.longdouble("inf"))
    exact_adjacent = Fraction(*adjacent.as_integer_ratio())
    assert Fraction(*value.as_integer_ratio()) == expected != exact_adjacent
    values = np.asarray([value, adjacent], dtype=object)
    values.flags.writeable = False
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        keys = validation._metadata_keys(values)
        assert list(keys) == [expected, exact_adjacent]
        assert [hash(key) for key in keys] == [hash(expected), hash(exact_adjacent)]
        assert len(set(keys)) == 2 and sorted(keys) == sorted([expected, exact_adjacent])
        with pytest.raises(ValueError, match="unique"):
            validation.validate_metadata(np.asarray([value, value], dtype=object), 2, name="ids", unique=True)


@pytest.mark.parametrize("warm", [False, True])
def test_exact_native_keys_preserve_readonly_inputs_and_error_state(warm):
    info = np.finfo(np.longdouble)
    values = np.asarray([np.longdouble("0.5"), np.longdouble("-0"), info.max, -info.max,
                         info.tiny, np.nextafter(np.longdouble(0), np.longdouble(1)),
                         np.nextafter(np.longdouble(1), np.longdouble(2)),
                         np.longdouble(np.finfo(float).max), np.longdouble(np.nextafter(0., 1.))], dtype=object)
    expected = [Fraction(*value.as_integer_ratio()) for value in values]
    originals = tuple(values)
    values.flags.writeable = False
    if warm:
        validation._native_key_format()
    before = np.geterr()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        active = np.geterr()
        keys = validation._metadata_keys(values)
        assert list(keys) == expected
        assert [hash(key) for key in keys] == [hash(key) for key in expected]
        assert sorted(keys) == sorted(expected) and np.geterr() == active
    assert np.geterr() == before
    assert not values.flags.writeable
    assert all(value is original for value, original in zip(values, originals))


@pytest.mark.parametrize("policy,warning_policy,concurrent", [
    ("raise", "error", False), ("raise", "always", False),
    ("warn", "error", False), ("warn", "always", False),
    ("ignore", "error", False), ("ignore", "always", False),
    ("raise", "error", True),
])
def test_fresh_process_native_setup_restores_state(tmp_path, policy, warning_policy, concurrent):
    output = tmp_path / "native.json"
    import_root = Path(validation.__file__).resolve().parents[1]
    command = [sys.executable, "-B", str(Path(__file__).with_name("_metadata_key_child.py")),
               "--import-root", str(import_root), "--output", str(output),
               "--policy", policy, "--warning-policy", warning_policy]
    if concurrent:
        command.append("--concurrent")
    environment = dict(os.environ)
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed" and receipt["concurrent"] is concurrent
    assert receipt["numpy_cache_identity_restored"] and receipt["error_and_warning_state_preserved"]
    assert receipt["finfo_calls"] == (2 if concurrent else 1)
