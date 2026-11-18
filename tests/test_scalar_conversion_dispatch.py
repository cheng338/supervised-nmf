"""Scalar dispatch keeps exact identities and observable conversion protocols."""

from contextlib import contextmanager
from fractions import Fraction
import math
import sys
import warnings

import numpy as np
import pytest

from supervised_nmf import evaluation, validation


NATIVE = np.dtype("longdouble").type
TAG = "__supervised_nmf_scalar__"


@pytest.fixture(autouse=True)
def isolated_native_format(monkeypatch):
    monkeypatch.setattr(validation, "_NATIVE_KEY_FORMAT", None)


@contextmanager
def stock_item_calls():
    """Observe actual C method calls, without replacing scalar protocols."""
    calls = []
    previous = sys.getprofile()

    def observe(frame, event, function):
        if (event == "c_call" and getattr(function, "__name__", None) == "item"
                and type(getattr(function, "__self__", None)) is NATIVE):
            calls.append(function.__self__)

    try:
        sys.setprofile(observe)
        control = NATIVE("0.25")
        control.item()
        # A positive control prevents a missing profiler event from making
        # every bypass assertion pass. This observer is CPython-specific.
        assert len(calls) == 1 and calls[0] is control
        calls.clear()
        yield calls
    finally:
        sys.setprofile(previous)


@pytest.mark.skipif(sys.implementation.name != "cpython", reason="Observes CPython C-method events")
def test_json_skips_only_stock_item_and_keeps_exact_types_and_zero_sign():
    events = []

    class Scalar(NATIVE):
        def item(self):
            events.append("item")
            return -0.0

    values = [NATIVE("0.5"), NATIVE("-0"), Scalar("7.25"), np.float64("-0")]
    with stock_item_calls() as calls:
        result = evaluation._jsonable(values)
    assert calls == [] and events == ["item"]
    assert result[:2] == [
        {TAG: "longdouble-v1", "significand": "1", "exponent": -1},
        {TAG: "longdouble-v1", "significand": "-0", "exponent": 0},
    ]
    assert all(type(value["significand"]) is str and type(value["exponent"]) is int
               for value in result[:2])
    assert all(type(value) is float and math.copysign(1, value) == -1 for value in result[2:])


@pytest.mark.skipif(sys.implementation.name != "cpython", reason="Observes CPython C-method events")
@pytest.mark.parametrize("container", [
    "stock-first", "python-first", "numpy-first", "subclass-first",
    "array-subclass", "list", "native-dtype",
])
def test_key_item_bypass_requires_first_stock_scalar_in_exact_object_array(container):
    events = []

    class Scalar(NATIVE):
        def item(self):
            events.append("item")
            return 0.75

    class Array(np.ndarray):
        pass

    first = {"python-first": 7, "numpy-first": np.int64(7),
             "subclass-first": Scalar("7.25")}.get(container, NATIVE("0.5"))
    members = [first, NATIVE("1.5"), NATIVE("2.5")]
    values = np.asarray(members, dtype=object)
    if container == "array-subclass":
        values = values.view(Array)
    elif container == "list":
        values = members
    elif container == "native-dtype":
        values = np.asarray(members, dtype=NATIVE)
    with stock_item_calls() as calls:
        keys = validation._metadata_keys(values)
    expected_calls = 0 if container == "stock-first" else (
        2 if container in ("python-first", "numpy-first", "subclass-first") else 3)
    assert len(calls) == expected_calls
    assert events == (["item"] if container == "subclass-first" else [])
    expected_first = 7 if container in ("python-first", "numpy-first") else (
        0.75 if container == "subclass-first" else 0.5)
    assert list(keys) == [expected_first, 1.5, 2.5]
    assert keys is not values


def convert_scalar(route, value):
    if route == "json":
        return evaluation._jsonable(value)
    first = NATIVE("0.5") if route == "native-first" else np.int64(7)
    return validation._metadata_keys(np.asarray([first, value], dtype=object))[1]


@pytest.mark.parametrize("route", ["json", "native-first", "ordinary-first"])
@pytest.mark.parametrize("returned", ["python", "native", "self"])
def test_subclass_item_result_and_original_ratio_keep_their_distinct_contracts(route, returned):
    events = []

    class Scalar(NATIVE):
        def item(self):
            events.append("item")
            if returned == "python":
                return 2.5
            if returned == "native":
                return NATIVE("1.25")
            return self

        def as_integer_ratio(self):
            events.append("ratio")
            return 6, 8

    result = convert_scalar(route, Scalar("7.25"))
    if returned == "python":
        assert type(result) is float and result == 2.5
    elif route == "json":
        assert result == {TAG: "longdouble-v1", "significand": "5" if returned == "native" else "3",
                          "exponent": -2}
    else:
        assert type(result) is Fraction and result == Fraction(3, 4)
    assert events == (["item"] if returned == "python" or (route == "json" and returned == "native")
                      else ["item", "ratio"])


@pytest.mark.parametrize("route", ["json", "native-first", "ordinary-first"])
@pytest.mark.parametrize("failure", ["item", "ratio", "warning"])
def test_subclass_errors_and_warning_as_error_preserve_protocol_order(route, failure):
    events = []

    class Scalar(NATIVE):
        def item(self):
            events.append("item")
            if failure == "item":
                raise LookupError("scalar item failed")
            if failure == "warning":
                warnings.warn("scalar item warning", RuntimeWarning)
            return self

        def as_integer_ratio(self):
            events.append("ratio")
            raise ValueError("scalar ratio failed")

    error = {"item": LookupError, "ratio": ValueError, "warning": RuntimeWarning}[failure]
    before_errors, before_filters = np.geterr(), list(warnings.filters)
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        active_errors, active_filters = np.geterr(), list(warnings.filters)
        with pytest.raises(error, match="scalar"):
            convert_scalar(route, Scalar("7.25"))
        assert np.geterr() == active_errors and warnings.filters == active_filters
    assert events == (["item", "ratio"] if failure == "ratio" else ["item"])
    assert np.geterr() == before_errors and warnings.filters == before_filters


@pytest.mark.parametrize("route", ["json", "native-first", "ordinary-first"])
def test_subclass_arithmetic_uses_the_callers_numpy_error_callback(route):
    events = []

    class Scalar(NATIVE):
        def item(self):
            events.append("item")
            np.divide(NATIVE(1), NATIVE(0))
            return 2.5

    def callback(message, flag):
        events.append((message, flag))

    previous_callback, previous_errors = np.geterrcall(), np.geterr()
    try:
        np.seterrcall(callback)
        with np.errstate(all="call"):
            assert convert_scalar(route, Scalar("7.25")) == 2.5
            assert np.geterrcall() is callback and set(np.geterr().values()) == {"call"}
        assert events == ["item", ("divide by zero", 1)]
    finally:
        np.seterrcall(previous_callback)
    assert np.geterr() == previous_errors and np.geterrcall() is previous_callback


@pytest.mark.parametrize("container", ["list", "array"])
@pytest.mark.parametrize("family", ["empty", "strings", "builtins", "numpy"])
def test_fallback_container_callbacks_and_no_copy_returns(container, family):
    events = []

    class List(list):
        def __len__(self):
            events.append("len")
            return super().__len__()

        def __iter__(self):
            events.append("iter")
            return super().__iter__()

        def copy(self):
            events.append("copy")
            return super().copy()

    class Array(np.ndarray):
        def __len__(self):
            events.append("len")
            return super().__len__()

        def __iter__(self):
            events.append("iter")
            return super().__iter__()

        def copy(self, *args, **kwargs):
            events.append("copy")
            return super().copy(*args, **kwargs)

    members = {"empty": [], "strings": ["a", "b"], "builtins": [1, 2],
               "numpy": [NATIVE("0.5"), np.int64(3)]}[family]
    values = List(members) if container == "list" else np.asarray(members, dtype=object).view(Array)
    keys = validation._metadata_keys(values)
    expected_events = ["len"]
    if family in ("builtins", "numpy"):
        expected_events.append("iter")
    if family == "numpy":
        expected_events.append("copy")
    assert events == expected_events
    assert (keys is values) is (family != "numpy")
    assert list(keys) == ([0.5, 3] if family == "numpy" else members)


@pytest.mark.parametrize("first", ["stock", "numpy"])
@pytest.mark.parametrize("layout", ["readonly", "reversed", "strided"])
def test_both_key_loops_own_output_and_preserve_readonly_storage(first, layout):
    events = []

    class Scalar(NATIVE):
        def item(self):
            events.append("item")
            return 0.75

    members = [NATIVE("0.5") if first == "stock" else np.uint64(2**64 - 1),
               NATIVE("1.25"), Scalar("7.25"), NATIVE("-0"), 2**64]
    if layout == "strided":
        owner = np.empty(2 * len(members), dtype=object)
        owner[::2], owner[1::2] = members, "untouched"
        values = owner[::2]
    elif layout == "reversed":
        owner = np.asarray(members[::-1], dtype=object)
        values = owner[::-1]
    else:
        owner = np.asarray(members, dtype=object)
        values = owner
    original_objects = tuple(owner)
    owner.flags.writeable = values.flags.writeable = False
    keys = validation._metadata_keys(values)
    assert list(keys) == [0.5 if first == "stock" else 2**64 - 1, 1.25, 0.75, 0, 2**64]
    assert events == ["item"]
    assert keys is not values and not np.shares_memory(keys, owner) and keys.flags.writeable
    keys[1] = 99
    assert all(current is original for current, original in zip(owner, original_objects))
    assert not owner.flags.writeable and not values.flags.writeable


class NumpyProxy:
    def __getattr__(self, name):
        return getattr(np, name)


@pytest.mark.parametrize("first", ["stock", "numpy"])
def test_format_probe_mutation_preserves_read_copy_and_later_callback_order(monkeypatch, first):
    events = []

    class Later(NATIVE):
        def item(self):
            events.append("later-item")
            return 9.25

    members = [NATIVE("0.5"), NATIVE("1.5"), NATIVE("2.5")]
    if first == "numpy":
        members.insert(0, np.int64(7))
    values = np.asarray(members, dtype=object)
    offset = 0 if first == "stock" else 1
    replacement = Later("17.5")

    class Probe(NumpyProxy):
        def finfo(self, dtype):
            events.append("probe")
            values[offset] = NATIVE("19.5")
            values[offset + 1] = replacement
            values[offset + 2] = 31
            return np.finfo(dtype)

    monkeypatch.setattr(validation, "np", Probe())
    keys = validation._metadata_keys(values)
    assert events == ["probe", "later-item"]
    assert values[offset] == NATIVE("19.5") and values[offset + 1] is replacement
    assert list(keys[offset:]) == [0.5, 9.25, 31 if first == "stock" else 2.5]
    assert keys is not values and not np.shares_memory(keys, values)


@pytest.mark.parametrize("first", ["stock", "numpy"])
@pytest.mark.parametrize("failure", ["call", "attribute"])
def test_mutating_failed_format_probe_retries_from_current_values(monkeypatch, first, failure):
    events = []

    class Later(NATIVE):
        def item(self):
            events.append("later-item")
            return 9.25

    class Partial:
        nmant = 0

        @property
        def maxexp(self):
            raise RuntimeError("incomplete format")

    members = [NATIVE("0.5"), NATIVE("1.5"), NATIVE("2.5")]
    if first == "numpy":
        members.insert(0, np.int64(7))
    values = np.asarray(members, dtype=object)
    offset = 0 if first == "stock" else 1
    replacement = Later("17.5")

    class BrokenProbe(NumpyProxy):
        def finfo(self, dtype):
            events.append("probe")
            values[offset:offset + 3] = [NATIVE("19.5"), replacement, 31]
            if failure == "call":
                raise RuntimeError("failed format")
            return Partial()

    with monkeypatch.context() as patch:
        patch.setattr(validation, "np", BrokenProbe())
        with pytest.raises(RuntimeError, match="format"):
            validation._metadata_keys(values)
    assert validation._NATIVE_KEY_FORMAT is None and events == ["probe"]
    after_failure = tuple(values)
    keys = validation._metadata_keys(values)
    assert list(keys[offset:]) == [19.5, 9.25, 31]
    assert events == ["probe", "later-item"] and validation._NATIVE_KEY_FORMAT is not None
    assert all(current is original for current, original in zip(values, after_failure))


@pytest.mark.parametrize("first", ["stock", "numpy"])
def test_subclass_item_can_replace_a_later_stock_scalar(first):
    events = []
    values = np.empty(4, dtype=object)

    class Later(NATIVE):
        def item(self):
            events.append("replacement-item")
            return 9.25

    replacement = Later("17.5")

    class Mutator(NATIVE):
        def item(self):
            events.append("mutator-item")
            values[2] = replacement
            return 7.75

    values[:] = [NATIVE("0.5") if first == "stock" else np.int64(7),
                 Mutator("1.5"), NATIVE("2.5"), NATIVE("3.5")]
    keys = validation._metadata_keys(values)
    assert events == ["mutator-item", "replacement-item"]
    assert values[2] is replacement and list(keys[1:]) == [7.75, 9.25, 3.5]
    assert keys is not values and not np.shares_memory(keys, values)


@pytest.mark.parametrize("first", ["stock", "numpy"])
def test_rebound_public_longdouble_alias_does_not_bypass_subclass_item(monkeypatch, first):
    events = []

    class Replacement(NATIVE):
        def item(self):
            events.append("item")
            return 2.5

        def as_integer_ratio(self):
            raise AssertionError("The subclass item result must be respected")

    def forbidden_probe():
        raise AssertionError("The original stock type no longer matches the public alias")

    values = np.asarray([NATIVE("0.5") if first == "stock" else np.int64(7),
                         NATIVE("1.25"), Replacement("0.75")], dtype=object)
    monkeypatch.setattr(np, "longdouble", Replacement)
    monkeypatch.setattr(validation, "_native_key_format", forbidden_probe)
    keys = validation._metadata_keys(values)
    assert list(keys) == [0.5 if first == "stock" else 7, 1.25, 2.5]
    assert type(keys[1]) is Fraction
    encoded = evaluation._jsonable([values[1], values[2]])
    assert encoded == [{TAG: "longdouble-v1", "significand": "5", "exponent": -2}, 2.5]
    assert events == ["item", "item"] and validation._NATIVE_KEY_FORMAT is None


@pytest.mark.parametrize("first", ["stock", "numpy"])
def test_both_key_loops_preserve_real_native_precision_and_range(first):
    info = np.finfo(NATIVE)
    if not (info.nmant > 52 and info.maxexp > 1024 and info.minexp < -1022):
        pytest.skip("Requires native precision and exponent range beyond binary64")
    native = [np.nextafter(NATIVE(1), NATIVE(2)),
              np.ldexp(NATIVE(1), 1024), np.ldexp(NATIVE(1), -1075)]
    expected = [Fraction(*value.as_integer_ratio()) for value in native]
    assert expected[0] > 1 and expected[1] > Fraction(sys.float_info.max)
    assert 0 < expected[2] < Fraction(2) ** -1074
    values = np.asarray([NATIVE("0.5") if first == "stock" else np.int64(7), *native], dtype=object)
    originals = tuple(values)
    values.flags.writeable = False
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        keys = validation._metadata_keys(values)
    assert list(keys[1:]) == expected
    assert [hash(key) for key in keys[1:]] == [hash(value) for value in expected]
    assert type(keys[1]) is Fraction and type(keys[2]) is int and type(keys[3]) is Fraction
    assert all(current is original for current, original in zip(values, originals))
    assert not values.flags.writeable
