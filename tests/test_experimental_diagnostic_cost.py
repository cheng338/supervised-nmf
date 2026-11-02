"""Diagnostic-private storage and isolation contracts with safe arithmetic.

Native route observations use an independent einsum reference, optimize=False;
no raw AOCL product is called by these focused storage tests.
"""
import gc
import inspect
from pathlib import Path
import sys
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pytest
import experimental_nmf as experimental
from supervised_nmf import _aocl

assert Path(experimental.__file__).resolve() == ROOT / 'scripts/experimental_nmf.py'
assert Path(_aocl.__file__).resolve() == ROOT / 'src/supervised_nmf/_aocl.py'


def bits(value):
    return value.tobytes(order='C')


def state(value):
    return (value.shape, value.strides, value.dtype.str, value.flags.writeable,
            value.flags.aligned, bits(value))


def reference(left, right):
    return np.array(np.einsum('ik,kj->ij', left, right, optimize=False),
                    dtype=np.float64, order='C', copy=True)


def inputs(m, k, n):
    left = (np.arange(m * k).reshape(m, k) % 7 - 3) / 8.
    right = (np.arange(k * n).reshape(k, n) % 9 - 4) / 16.
    X = (np.arange(m * n).reshape(m, n) % 5) / 32.
    for value in (left, right, X):
        value.setflags(write=False)
    return left, right, X


@pytest.mark.skipif(not hasattr(experimental, '_diagnostic_residual'), reason='No fused diagnostic role selected')
@pytest.mark.parametrize('m,k,n', [(16, 3, 17), (8, 1, 1), (9, 3, 17), (16, 3, 16)])
def test_private_residual_matches_current_native_operands_and_owned_result(monkeypatch, m, k, n):
    left, right, X = inputs(m, k, n)
    before = tuple(state(value) for value in (left, right, X))
    native = []

    class SafeNative:
        def __getattr__(self, name):
            return getattr(np, name)

        def matmul(self, a, b):
            assert a.shape[0] % 8 == b.shape[1] % 8 == 0
            assert a.shape[1] == b.shape[0] == k
            result = reference(a, b)
            native.append((a.shape, b.shape, a.strides, b.strides,
                           bits(a), bits(b), bits(result)))
            return result

        def empty(self, shape, **kwargs):
            result = np.empty(shape, **kwargs)
            result.fill(np.nan)
            return result

    with monkeypatch.context() as patch:
        proxy = SafeNative()
        patch.setattr(_aocl, 'np', proxy)
        patch.setattr(experimental, 'np', proxy)
        expected = _aocl.padded_matmul(left, right)
        expected -= X
        expected_native = list(native)
        native.clear()
        actual = experimental._diagnostic_residual(_aocl.padded_matmul, left, right, X)
    assert native == expected_native
    assert bits(actual) == bits(expected) == bits(reference(left, right) - X)
    assert actual.dtype == np.float64 and actual.flags.owndata and actual.base is None
    assert actual.flags.c_contiguous and actual.flags.writeable
    assert not any(np.shares_memory(actual, value) for value in (left, right, X))
    assert tuple(state(value) for value in (left, right, X)) == before


@pytest.mark.skipif(not hasattr(experimental, '_diagnostic_residual'), reason='No fused diagnostic role selected')
def test_private_residual_respects_non_native_selected_callable():
    left, right, X = inputs(9, 3, 7)
    calls = []

    def selected(a, b):
        calls.append((a is left, b is right))
        return reference(a, b)

    answer = experimental._diagnostic_residual(selected, left, right, X)
    assert calls == [(True, True)]
    assert bits(answer) == bits(reference(left, right) - X)
    assert answer.flags.owndata and answer.flags.c_contiguous


@pytest.mark.skipif(not hasattr(experimental, '_diagnostic_residual'), reason='No fused diagnostic role selected')
def test_private_residual_drops_helper_workspaces_with_retained_traceback(monkeypatch):
    left, right, X = inputs(16, 3, 17)
    refs = []

    class StopAllocation(RuntimeError):
        pass

    class SafeFailingNative:
        def __getattr__(self, name):
            return getattr(np, name)

        def zeros(self, shape, **kwargs):
            result = np.zeros(shape, **kwargs)
            refs.append(weakref.ref(result))
            return result

        def matmul(self, a, b):
            assert a.shape[0] % 8 == b.shape[1] % 8 == 0
            result = reference(a, b)
            refs.append(weakref.ref(result))
            return result

        def empty(self, shape, **kwargs):
            # This traceback holds shape/kwargs, not the earlier input arrays.
            raise StopAllocation('owned residual allocation')

    retained = None
    with monkeypatch.context() as patch:
        patch.setattr(experimental, 'np', SafeFailingNative())
        try:
            experimental._diagnostic_residual(_aocl.padded_matmul, left, right, X)
        except StopAllocation as error:
            retained = error
    assert retained is not None and retained.__traceback__ is not None
    assert len(refs) == 2
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.skipif(not hasattr(experimental, '_diagnostic_pg_values'), reason='No private PG guard selected')
def test_diagnostic_private_gradient_helper_does_not_replace_solver_helper(monkeypatch):
    original = experimental._pg_values
    private = experimental._diagnostic_pg_values
    assert private is not original
    assert '_diagnostic_pg_values' not in inspect.getsource(experimental.ANLSBPP)
    assert '_pg_values' in inspect.getsource(experimental.ANLSBPP)
    X = np.full((8, 8), 0.75)
    W = np.full((8, 3), 0.5)
    H = np.full((3, 8), 0.5)

    def solver_only(*args):
        raise AssertionError('descriptive diagnostic reached the solver helper')

    # Reference dispatch is explicit and uses no native fringe product.
    monkeypatch.setattr(experimental, 'nnls_matmul', lambda: reference)
    monkeypatch.setattr(experimental, '_pg_values', solver_only)
    answer = experimental.projected_gradient_kkt(X, W, H)
    assert answer['reconstruction_error'] == answer['objective'] == 0.
    assert answer['raw']['relative_projected_gradient'] == 0.
    assert answer['canonical']['relative_projected_gradient'] == 0.
    assert experimental._pg_values is solver_only
    assert inspect.unwrap(experimental.ANLSBPP.fit_transform).__globals__['_pg_values'] is solver_only
