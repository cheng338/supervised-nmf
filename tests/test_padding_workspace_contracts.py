"""Shared protected-product contracts and scoped native output reuse."""

import gc
from pathlib import Path
import sys
import weakref

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supervised_nmf import _aocl, _padding_roles

assert Path(_aocl.__file__).resolve() == ROOT / "src/supervised_nmf/_aocl.py"
assert Path(_padding_roles.__file__).resolve() == ROOT / "src/supervised_nmf/_padding_roles.py"


@pytest.fixture(autouse=True)
def single_native_thread():
    with threadpool_limits(limits=1):
        yield


def operands(m, k, n):
    left = np.array((np.arange(m * k).reshape(m, k) % 13 - 6) / 8, order="C")
    right = np.array((np.arange(k * n).reshape(k, n) % 17 - 8) / 16, order="C")
    return left, right


def reference(left, right):
    return np.einsum("ik,kj->ij", left, right, optimize=False)


def state(value):
    return (type(value), value.dtype.str, value.shape, value.strides,
            value.flags.writeable, value.flags.aligned, value.tobytes())


class ArraySubclass(np.ndarray):
    pass


def layout(value, kind):
    if kind == "f":
        return np.array(value, order="F")
    if kind == "view":
        backing = np.empty((value.shape[0] + 2, value.shape[1]))
        result = backing[1:-1]
        result[:] = value
        return result
    if kind == "strided":
        backing = np.empty(tuple(2 * size for size in value.shape))
        result = backing[::2, ::2]
        result[:] = value
        return result
    if kind == "float32":
        return value.astype(np.float32)
    if kind == "subclass":
        return value.view(ArraySubclass)
    if kind == "unaligned":
        value.setflags(align=False)
    if kind == "singleton":
        return value.ravel()[:, None] if value.shape[1] == 1 else value.ravel()[None, :]
    return value


class NativeTrace:
    """Inspect real native calls; reject unprotected geometry before dispatch."""

    def __init__(self):
        self.calls = []
        self.failure_at = None

    def __getattr__(self, name):
        return getattr(np, name)

    def matmul(self, left, right, *, out=None):
        assert left.shape[0] % 8 == right.shape[1] % 8 == 0
        self.calls.append((left, right, out))
        if len(self.calls) == self.failure_at:
            if out is not None:
                out.fill(777.0)
            raise RuntimeError("injected native product failure")
        return np.matmul(left, right) if out is None else np.matmul(left, right, out=out)


@pytest.fixture
def trace(monkeypatch):
    value = NativeTrace()
    # The workspace imports np directly; patch both module bindings.
    monkeypatch.setattr(_aocl, "np", value)
    monkeypatch.setattr(_padding_roles, "np", value)
    return value


CASES = [
    (8, 7, 8, "c", "c", "aligned"),
    (8, 7, 9, "view", "c", "left"),
    (9, 7, 8, "c", "view", "right"),
    (9, 7, 17, "f", "view", "split"),
    (9, 13, 17, "strided", "c", "split"),
    (5, 7, 9, "c", "c", "fallback"),
    (9, 7, 5, "c", "c", "fallback"),
    (8, 7, 9, "f", "c", "fallback"),
    (9, 7, 8, "c", "strided", "fallback"),
    (8, 7, 9, "float32", "c", "fallback"),
    (9, 7, 8, "c", "subclass", "fallback"),
    (8, 7, 9, "unaligned", "c", "fallback"),
    (9, 7, 17, "c", "f", "fallback"),
    (9, 7, 17, "float32", "c", "fallback"),
    (9, 7, 17, "c", "unaligned", "fallback"),
    (8, 1, 9, "singleton", "c", "fallback"),
    (9, 1, 8, "c", "singleton", "fallback"),
    (9, 1, 17, "c", "singleton", "fallback"),
    (0, 7, 9, "c", "c", "fallback"),
    (9, 7, 0, "c", "c", "fallback"),
    (9, 0, 9, "c", "c", "fallback"),
]


@pytest.mark.parametrize("m,k,n,left_layout,right_layout,branch", CASES)
def test_shared_branch_contract(trace, m, k, n, left_layout, right_layout, branch):
    left, right = operands(m, k, n)
    left, right = layout(left, left_layout), layout(right, right_layout)
    left.setflags(write=False)
    right.setflags(write=False)
    before = state(left), state(right)
    mp, npad = ((m + 7) // 8) * 8, ((n + 7) // 8) * 8
    padded_left, padded_right = np.zeros((mp, k)), np.zeros((k, npad))
    padded_left[:m] = left
    padded_right[:, :n] = right
    core = (n // 8) * 8
    expected = [(padded_left, padded_right)]
    if branch == "split":
        fringe = np.zeros((k, 8))
        fringe[:, :n - core] = right[:, core:]
        expected = [(padded_left, right[:, :core]), (padded_left, fringe)]
    results, signatures = [], []
    workspace = _padding_roles.make_output_workspace(_aocl.padded_matmul)
    assert isinstance(workspace, _padding_roles._NativeOutputWorkspace)
    for operation in (_aocl.padded_matmul, workspace):
        trace.calls.clear()
        result = operation(left, right)
        assert len(trace.calls) == len(expected)
        signature = []
        for index, ((a, b, out), (expected_a, expected_b)) in enumerate(zip(trace.calls, expected)):
            assert a.shape[1] == b.shape[0] == k
            np.testing.assert_array_equal(a, expected_a)
            np.testing.assert_array_equal(b, expected_b)
            borrowed_a, borrowed_b = np.shares_memory(a, left), np.shares_memory(b, right)
            if left.size and right.size:
                assert borrowed_a == (branch in ("aligned", "left"))
                assert borrowed_b == (branch in ("aligned", "right") or (branch == "split" and index == 0))
            if branch == "split" and index == 0:
                assert b.strides == right.strides
            signature.append((a.shape, b.shape, a.strides, b.strides, borrowed_a, borrowed_b))
            assert (out is not None) == (operation is workspace and branch != "aligned")
            if out is not None:
                assert not np.shares_memory(result, out)
        assert result.flags.owndata and result.flags.c_contiguous and result.flags.writeable
        assert not np.shares_memory(result, left) and not np.shares_memory(result, right)
        np.testing.assert_array_equal(result, reference(left, right))
        assert before == (state(left), state(right))
        results.append(result)
        signatures.append(signature)
    assert signatures[0] == signatures[1]
    assert results[0].tobytes() == results[1].tobytes()
    workspace.clear()


def test_workspace_reuse_result_independence_and_slot_transitions(trace):
    workspace = _padding_roles.make_output_workspace(_aocl.padded_matmul)
    left, right = operands(9, 7, 17)
    first = workspace(left, right)
    first.fill(-321.0)  # Callers may consume the returned array in place.
    slots = tuple(workspace._buffers)
    second = workspace(left * 2, right * 3)
    assert all(actual is previous for actual, previous in zip(workspace._buffers, slots))
    np.testing.assert_array_equal(first, np.full(first.shape, -321.0))
    np.testing.assert_array_equal(second, reference(left * 2, right * 3))
    assert not np.shares_memory(first, second)
    assert all(not np.shares_memory(result, slot) for result in (first, second) for slot in slots)
    saved_second = second.copy()
    retired_fringe = weakref.ref(slots[1])
    core_slot = slots[0]
    del slots
    trace.calls.clear()
    # Two products -> one product of the same core shape: retain only slot 0.
    a, b = operands(9, 7, 16)
    np.testing.assert_array_equal(workspace(a, b), reference(a, b))
    assert workspace._buffers[0] is core_slot and workspace._buffers[1] is None
    gc.collect()
    assert retired_fringe() is None
    retired_core = weakref.ref(core_slot)
    del core_slot
    trace.calls.clear()
    # A changed native shape replaces the remaining slot.
    a, b = operands(25, 7, 16)
    np.testing.assert_array_equal(workspace(a, b), reference(a, b))
    assert workspace._buffers[0].shape == (32, 16)
    gc.collect()
    assert retired_core() is None
    trace.calls.clear()
    # Aligned direct multiplication discards all cached outputs.
    a, b = operands(8, 7, 8)
    np.testing.assert_array_equal(workspace(a, b), reference(a, b))
    assert workspace._buffers == [None, None] and workspace._used == 0
    np.testing.assert_array_equal(second, saved_second)
    workspace.clear()
    workspace.clear()


@pytest.mark.parametrize("failure_at", [1, 2])
def test_workspace_native_failure_clears_slots_and_recovers(trace, failure_at):
    workspace = _padding_roles.make_output_workspace(_aocl.padded_matmul)
    left, right = operands(9, 7, 17)
    retained = workspace(left, right)
    original = retained.copy()
    refs = [weakref.ref(slot) for slot in workspace._buffers]
    trace.calls.clear()
    trace.failure_at = failure_at
    with pytest.raises(RuntimeError, match="injected native product failure"):
        workspace(left * 2, right)
    assert len(trace.calls) == failure_at
    assert workspace._buffers == [None, None] and workspace._used == 0
    np.testing.assert_array_equal(retained, original)
    trace.calls.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
    trace.failure_at = None
    np.testing.assert_array_equal(workspace(left * 2, right), reference(left * 2, right))
    np.testing.assert_array_equal(retained, original)
    workspace.clear()
