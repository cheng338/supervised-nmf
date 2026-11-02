"""One-sided operand reuse preserves the existing padded-product contract.

Ineligible-layout/type branches are observed through reference arithmetic, not
unprotected native products. Ordinary native cases use only the helper's usual
protected dimensions. No guard pages or native-fault reproductions are used.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import _aocl

assert Path(_aocl.__file__).resolve() == ROOT / "src" / "supervised_nmf" / "_aocl.py"

# Archived release and qualification records retain exact historical source
# identities. These maintained tests check the current helper's behavior.


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def reference_product(left, right):
    return np.einsum("ik,kj->ij", np.asarray(left), np.asarray(right), optimize=False)


def operands(m, k, n):
    left = np.array((np.arange(m * k).reshape(m, k) % 19 - 9) / 8,
                    dtype=np.float64, order="C", copy=True)
    right = np.array((np.arange(k * n).reshape(k, n) % 23 - 11) / 16,
                     dtype=np.float64, order="C", copy=True)
    return left, right


def ordinary_layout(values, layout):
    if layout == "fortran":
        return np.array(values, dtype=np.float64, order="F", copy=True)
    if layout == "strided":
        storage = np.empty((values.shape[0] * 2, values.shape[1] * 2), dtype=np.float64)
        result = storage[::2, ::2]
        result[:] = values
        return result
    if layout == "c_view":
        storage = np.empty((values.shape[0] + 2, values.shape[1]), dtype=np.float64)
        result = storage[1:-1]
        result[:] = values
        assert result.flags.c_contiguous and not result.flags.owndata
        return result
    return np.array(values, dtype=np.float64, order="C", copy=True)


def observe(monkeypatch, left, right):
    """Record references before any copying, with no native matrix multiply."""
    allocations, calls = [], []
    original_zeros = np.zeros

    def zeros(shape, *args, **kwargs):
        result = original_zeros(shape, *args, **kwargs)
        allocations.append({"shape": tuple(shape), "array": result})
        return result

    def product(a, b):
        result = reference_product(a, b)
        calls.append({"left": a, "right": b, "result": result})
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_aocl.np, "zeros", zeros)
        patch.setattr(_aocl.np, "matmul", product)
        result = _aocl.padded_matmul(left, right)
    assert len(calls) in (1, 2)
    if len(calls) == 2:
        return result, {"split_calls": calls}, allocations
    return result, calls[0], allocations



def assert_two_product_contract(result, calls, allocations, left, right):
    m, k = left.shape
    n = right.shape[1]
    mp, npad = (m + 7) // 8 * 8, (n + 7) // 8 * 8
    mc, nc = m // 8 * 8, n // 8 * 8
    assert m >= 8 and n >= 8 and m % 8 and n % 8
    assert len(calls) == len(allocations) == 2
    if 'column' == 'column':
        assert [(c['left'].shape, c['right'].shape) for c in calls] == [((mp,k),(k,nc)),((mp,k),(k,8))]
        full, fringe, core = calls[0]['left'], calls[1]['right'], calls[0]['right']
        assert calls[1]['left'] is full
        assert allocations[0]['array'] is full and allocations[1]['array'] is fringe
        assert np.shares_memory(core, right)
        np.testing.assert_array_equal(core, right[:, :nc])
        np.testing.assert_array_equal(full[:m], left)
        np.testing.assert_array_equal(fringe[:, :n-nc], right[:, nc:])
        assert not np.count_nonzero(full[m:]) and not np.count_nonzero(fringe[:, n-nc:])
    else:
        assert [(c['left'].shape, c['right'].shape) for c in calls] == [((mc,k),(k,npad)),((8,k),(k,npad))]
        full, fringe, core = calls[0]['right'], calls[1]['left'], calls[0]['left']
        assert calls[1]['right'] is full
        assert allocations[0]['array'] is fringe and allocations[1]['array'] is full
        assert np.shares_memory(core, left)
        np.testing.assert_array_equal(core, left[:mc])
        np.testing.assert_array_equal(full[:, :n], right)
        np.testing.assert_array_equal(fringe[:m-mc], left[mc:])
        assert not np.count_nonzero(full[:, n:]) and not np.count_nonzero(fringe[m-mc:])
    for value in (full, fringe):
        assert value.dtype == np.float64 and value.flags.c_contiguous and value.flags.aligned
    assert result.shape == (m,n) and result.dtype == np.float64
    assert result.flags.owndata and result.flags.c_contiguous and result.base is None
    assert all(not np.shares_memory(result, value) for value in (left, right, *(c['result'] for c in calls)))
    np.testing.assert_array_equal(result, reference_product(left,right))


def assert_partial_contract(result, call, allocations, left, right, borrow_left, borrow_right):
    if "split_calls" in call:
        assert not borrow_left and not borrow_right
        assert_two_product_contract(result, call["split_calls"], allocations, left, right)
        return
    m, k = left.shape
    n = right.shape[1]
    mp, npad = (m + 7) // 8 * 8, (n + 7) // 8 * 8
    assert (mp, npad) != (m, n)
    a, b = call["left"], call["right"]
    assert (a is left) is borrow_left
    assert (b is right) is borrow_right
    assert len(allocations) == 2 - int(borrow_left) - int(borrow_right)
    assert a.shape == (mp, k) and b.shape == (k, npad)
    for array in (a, b):
        assert type(array) is np.ndarray
        assert array.dtype == np.dtype(np.float64) and array.dtype.isnative
        assert array.flags.c_contiguous and array.flags.aligned
        assert array.strides == (array.shape[1] * 8, 8)
    assert [entry["shape"] for entry in allocations] == (
        ([] if borrow_left else [(mp, k)]) + ([] if borrow_right else [(k, npad)]))
    if not borrow_left:
        assert any(entry["array"] is a for entry in allocations)
        assert not np.shares_memory(a, left)
    if not borrow_right:
        assert any(entry["array"] is b for entry in allocations)
        assert not np.shares_memory(b, right)
    np.testing.assert_array_equal(a[:m], np.asarray(left))
    np.testing.assert_array_equal(b[:, :n], np.asarray(right))
    assert not np.count_nonzero(a[m:])
    assert not np.count_nonzero(b[:, n:])
    assert result.shape == (m, n)
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.c_contiguous and result.flags.owndata
    assert result.base is None
    assert result is not call["result"]
    assert not np.shares_memory(result, call["result"])
    assert not np.shares_memory(result, left)
    assert not np.shares_memory(result, right)
    expected = reference_product(np.asarray(left, dtype=np.float64),
                                 np.asarray(right, dtype=np.float64))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("layout", ["c", "c_view"])
@pytest.mark.parametrize("readonly", [False, True])
def test_borrows_only_eligible_aligned_operand(monkeypatch, side, layout, readonly):
    left, right = operands(8, 7, 9) if side == "left" else operands(9, 7, 8)
    if side == "left":
        left = ordinary_layout(left, layout)
    else:
        right = ordinary_layout(right, layout)
    if readonly:
        left.setflags(write=False)
        right.setflags(write=False)
    before = left.copy(), right.copy()
    flags_before = left.flags.writeable, right.flags.writeable
    result, call, allocations = observe(monkeypatch, left, right)
    assert_partial_contract(result, call, allocations, left, right, side == "left", side == "right")
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])
    assert (left.flags.writeable, right.flags.writeable) == flags_before


def test_both_unaligned_operands_keep_two_real_padding_allocations(monkeypatch):
    left, right = operands(9, 7, 9)
    left.setflags(write=False)
    right.setflags(write=False)
    before = left.copy(), right.copy()
    result, call, allocations = observe(monkeypatch, left, right)
    assert_partial_contract(result, call, allocations, left, right, False, False)
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])


class ArraySubclass(np.ndarray):
    """A plain subclass detects accidental introduction of ufunc dispatch."""


def excluded_operand(values, kind):
    if kind in {"fortran", "strided"}:
        return ordinary_layout(values, kind)
    if kind == "negative_stride":
        return values[:, ::-1].copy()[:, ::-1]
    if kind == "float32":
        return values.astype(np.float32)
    if kind == "nonnative_float64":
        result = values.astype(np.dtype(np.float64).newbyteorder("S"))
        assert not result.dtype.isnative
        return result
    if kind == "subclass":
        return values.view(ArraySubclass)
    if kind == "unaligned_flag":
        # Normal allocated storage with the advisory flag cleared. No deliberately
        # misaligned pointer, guard page, or native boundary experiment is needed.
        result = values.copy()
        result.setflags(align=False)
        assert not result.flags.aligned
        return result
    raise AssertionError(kind)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("kind", ["fortran", "strided", "negative_stride", "float32",
                                  "nonnative_float64", "subclass", "unaligned_flag"])
def test_dimension_alignment_alone_does_not_enable_borrowing(monkeypatch, side, kind):
    left, right = operands(8, 7, 9) if side == "left" else operands(9, 7, 8)
    if side == "left":
        left = excluded_operand(left, kind)
    else:
        right = excluded_operand(right, kind)
    left.setflags(write=False)
    right.setflags(write=False)
    before = np.asarray(left).copy(), np.asarray(right).copy()
    metadata = [(array.dtype, array.strides, array.flags.aligned, array.flags.writeable)
                for array in (left, right)]
    result, call, allocations = observe(monkeypatch, left, right)
    assert_partial_contract(result, call, allocations, left, right, False, False)
    for array, values, original_metadata in zip((left, right), before, metadata):
        np.testing.assert_array_equal(np.asarray(array), values)
        assert (array.dtype, array.strides, array.flags.aligned, array.flags.writeable) == original_metadata


@pytest.mark.parametrize("side", ["left", "right"])
def test_contiguous_singletons_need_canonical_strides(monkeypatch, side):
    if side == "left":
        left = np.arange(8, dtype=np.float64)[:, None]
        right = np.arange(9, dtype=np.float64)[None, :].copy()
        exceptional = left
    else:
        left = np.arange(9, dtype=np.float64)[:, None].copy()
        right = np.arange(8, dtype=np.float64)[None, :]
        exceptional = right
    assert type(exceptional) is np.ndarray
    assert exceptional.dtype == np.dtype(np.float64)
    assert exceptional.flags.c_contiguous and exceptional.flags.aligned
    assert exceptional.strides != (exceptional.shape[1] * 8, 8)
    before = left.copy(), right.copy()
    result, call, allocations = observe(monkeypatch, left, right)
    assert_partial_contract(result, call, allocations, left, right, False, False)
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_both_aligned_fast_path_keeps_original_operands_and_native_result(monkeypatch, layout):
    left, right = operands(8, 7, 8)
    left, right = ordinary_layout(left, layout), ordinary_layout(right, layout)
    left.setflags(write=False)
    right.setflags(write=False)
    before = left.copy(), right.copy()
    result, call, allocations = observe(monkeypatch, left, right)
    assert not allocations
    assert call["left"] is left and call["right"] is right
    assert result is call["result"]
    np.testing.assert_allclose(result, reference_product(left, right), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])


def test_both_aligned_fast_path_does_not_add_float64_coercion(monkeypatch):
    # Observe the historical early return with a stub; float32 is outside the
    # helper's validated caller domain and need not reach native BLAS here.
    left, right = operands(8, 7, 8)
    left, right = left.astype(np.float32), right.astype(np.float32)
    result, call, allocations = observe(monkeypatch, left, right)
    assert not allocations
    assert call["left"] is left and call["right"] is right
    assert result is call["result"] and result.dtype == np.float32


@pytest.mark.parametrize("m,k,n", [(8, 7, 9), (9, 7, 8), (9, 7, 9), (8, 7, 8)])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_ordinary_signed_products_match_reference_without_mutation(m, k, n, layout):
    left, right = operands(m, k, n)
    left, right = ordinary_layout(left, layout), ordinary_layout(right, layout)
    left.setflags(write=False)
    right.setflags(write=False)
    before = left.copy(), right.copy()
    expected = reference_product(left, right)
    result = _aocl.padded_matmul(left, right)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.shape == (m, n) and result.dtype == np.float64
    assert result.flags.c_contiguous and result.flags.owndata
    assert not np.shares_memory(result, left) and not np.shares_memory(result, right)
    for row, column in ((0, 0), (m - 1, n - 1)):
        scalar = sum(float(left[row, index]) * float(right[index, column]) for index in range(k))
        assert abs(float(result[row, column]) - scalar) <= 1e-12
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])
    assert not left.flags.writeable and not right.flags.writeable
