"""Prepared QR transport admission and rebound ownership contracts.

Fixture contexts exercise storage boundaries only. They do not qualify a native
provider or run a solver; actual process and numerical tests cover those paths.
"""
from copy import copy, deepcopy
import os
import pickle

import numpy as np
import pytest

from supervised_nmf import _projection as projection


def transport(order="C"):
    design = np.arange(8, dtype=np.float64).reshape(4, 2)
    q = np.array(np.arange(8, dtype=np.float64).reshape(4, 2) + 10,
                 order=order, copy=True)
    r = np.array([[2., 3.], [0., 4.]], dtype=np.float64)
    native = (("storage-fixture-only", 1),)
    context = (os.getpid(), "local-controller-token", native)
    payload = (design.shape, design.tobytes(order="C"), q.tobytes(order=order),
               r.tobytes(order="C"), order, native)
    return payload, context, design, q, r


def assert_factors(entry, design, q, r, context):
    assert entry is not None and entry.matches(design, context)
    actual_q, actual_r = entry.factors()
    np.testing.assert_array_equal(actual_q, q)
    np.testing.assert_array_equal(actual_r, r)


@pytest.mark.parametrize("fault", [
    "tuple-subclass", "short", "shape-list", "shape-zero", "shape-bool",
    "order-invalid", "order-subclass",
])
def test_invalid_envelope_declines_before_callbacks(monkeypatch, fault):
    payload, _, _, _, _ = transport()
    calls = []

    class TupleSubclass(tuple):
        def __len__(self):
            calls.append("custom-length")
            raise AssertionError("custom tuple callback")

    class StringSubclass(str):
        def __eq__(self, other):
            calls.append("custom-equality")
            raise AssertionError("custom order callback")

    if fault == "tuple-subclass":
        payload = TupleSubclass(payload)
    elif fault == "short":
        payload = payload[:-1]
    else:
        values = list(payload)
        changed = {"shape-list": [4, 2], "shape-zero": (0, 2),
                   "shape-bool": (True, 2), "order-invalid": "A",
                   "order-subclass": StringSubclass("C")}
        values[4 if fault.startswith("order-") else 0] = changed[fault]
        payload = tuple(values)

    def forbidden(*args):
        calls.append("context-or-buffer")
        raise AssertionError("invalid envelope invoked a callback")

    monkeypatch.setattr(projection, "_qr_context", forbidden)
    monkeypatch.setattr(projection, "_qr_call_buffer", forbidden)
    assert projection._qr_call_bind(payload) is None
    # These observations stay outside the binder's optional-failure catch.
    assert calls == []


@pytest.mark.parametrize("fault", ["missing", "different", "allocation"])
def test_context_rejection_precedes_buffer_normalization(monkeypatch, fault):
    payload, context, _, _, _ = transport()
    calls = []

    def reader():
        calls.append("context")
        if fault == "allocation":
            raise MemoryError("optional context allocation")
        return None if fault == "missing" else context[:2] + (("different",),)

    def forbidden(*args):
        calls.append("buffer")
        raise AssertionError("rejected context reached buffers")

    monkeypatch.setattr(projection, "_qr_context", reader)
    monkeypatch.setattr(projection, "_qr_call_buffer", forbidden)
    assert projection._qr_call_bind(payload) is None
    assert calls == ["context"]


@pytest.mark.parametrize("failure", ["length", "allocation"])
@pytest.mark.parametrize("slot", [1, 2, 3], ids=["basis", "q", "r"])
def test_buffer_failure_stops_in_order_and_allows_retry(monkeypatch, slot, failure):
    payload, context, design, q, r = transport()
    attempted = list(payload)
    if failure == "length":
        attempted[slot] = attempted[slot][:-1]
    attempted = tuple(attempted)
    normalize = projection._qr_call_buffer
    calls, seen_values = [], []
    inject_failure = True

    def reader():
        calls.append("context")
        return context

    def observed(value, size):
        index = len(calls)
        calls.append((index, size))
        seen_values.append((index, value))
        if inject_failure and failure == "allocation" and index == slot:
            raise MemoryError("optional buffer allocation")
        return normalize(value, size)

    monkeypatch.setattr(projection, "_qr_context", reader)
    monkeypatch.setattr(projection, "_qr_call_buffer", observed)
    assert projection._qr_call_bind(attempted) is None
    expected = ["context", (1, 64), (2, 64), (3, 32)]
    assert calls == expected[:slot + 1]
    assert all(value is attempted[index] for index, value in seen_values)

    inject_failure = False
    calls.clear()
    seen_values.clear()
    rebound = projection._qr_call_bind(payload)
    assert calls == expected
    assert all(value is payload[index] for index, value in seen_values)
    assert_factors(rebound, design, q, r, context)


@pytest.mark.parametrize("slot", [1, 2, 3], ids=["basis", "q", "r"])
def test_rebound_preparation_does_not_borrow_mutable_owner(monkeypatch, slot):
    payload, context, design, q, r = transport()
    values = list(payload)
    owner = np.frombuffer(values[slot], dtype=np.uint8).copy()
    owner.flags.writeable = False
    values[slot] = owner
    monkeypatch.setattr(projection, "_qr_context", lambda: context)
    rebound = projection._qr_call_bind(tuple(values))
    assert_factors(rebound, design, q, r, context)
    owner.flags.writeable = True
    owner[:] = 0
    assert_factors(rebound, design, q, r, context)


@pytest.mark.parametrize("order", ["C", "F"])
def test_rebound_factors_preserve_order_bits_and_local_lifetime(monkeypatch, order):
    payload, context, design, q, r = transport(order)
    monkeypatch.setattr(projection, "_qr_context", lambda: context)
    # Call payload transport is separate from persistence of a prepared object.
    rebound = projection._qr_call_bind(pickle.loads(pickle.dumps(payload)))
    assert_factors(rebound, design, q, r, context)
    changed = design.copy()
    changed[0, 0] = -0.0
    assert not rebound.matches(changed, context)
    assert not rebound.matches(design, (context[0] + 1,) + context[1:])
    assert not rebound.matches(design, (context[0], "other-controller", context[2]))
    actual_q, actual_r = rebound.factors()
    assert actual_q.strides == q.strides
    for value in (actual_q, actual_r):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True
    assert copy(rebound) is None
    assert deepcopy(rebound) is None
    assert pickle.loads(pickle.dumps(rebound)) is None
