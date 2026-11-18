"""Reuse, invalidation and failure behavior of fixed-basis QR preparation."""
from copy import copy, deepcopy
import gc
import os
import pickle
import warnings
import weakref

import numpy as np
import pytest
import scipy
import threadpoolctl

from supervised_nmf import _nnls, _projection
from tests._prepared_qr_support import (
    assert_coordinates, fixed_model, observed_calls, problem, qr_state, reference,
)


def test_repeated_projection_reuses_only_qualified_preparation(qr_state):
    owner, query, known = problem()
    model = fixed_model(owner)
    before = pickle.dumps(model)
    with observed_calls() as cold:
        first = model.transform(query)
    entry = _projection._qr_registry_get(owner)
    assert (entry is not None) is qr_state
    assert cold['qr'] == 1
    with observed_calls() as warm:
        second = model.transform(query)
    assert warm['qr'] == int(not qr_state)
    assert_coordinates(first, known)
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, owner) and not np.shares_memory(first, query)
    first[:] = -123
    assert_coordinates(model.transform(query), known)
    assert pickle.dumps(model) == before
    if entry is not None:
        assert _projection._qr_registry_get(owner) is entry
        for factor in entry.factors():
            assert not factor.flags.writeable
            with pytest.raises(ValueError):
                factor.flat[0] = 0


@pytest.mark.parametrize('mutation', ['nonzero_value', 'readonly_alias', 'signed_zero', 'replacement'])
def test_basis_mutations_cannot_reuse_stale_factors(qr_state, mutation):
    owner, query, _ = problem()
    model = fixed_model(owner)
    model.transform(query)
    previous = _projection._qr_registry_get(owner)
    original = owner.tobytes()
    if mutation == 'replacement':
        model.components_ = owner.copy()
        owner = model.components_
        owner.flat[np.flatnonzero(owner)[0]] *= 1.25
    elif mutation == 'signed_zero':
        owner.flat[np.flatnonzero(owner == 0)[0]] = -0.0
    else:
        alias = owner.view()
        if mutation == 'readonly_alias':
            owner.flags.writeable = False
        alias.flat[np.flatnonzero(alias)[0]] *= 1.25
    assert owner.tobytes() != original
    expected = reference(owner, query)
    with observed_calls() as calls:
        actual = model.transform(query)
    assert calls['qr'] == 1
    assert_coordinates(actual, expected)
    if qr_state:
        assert _projection._qr_registry_get(owner) is not previous


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['C', 'F'])
def test_precision_and_basis_layout_preserve_outputs_and_inputs(qr_state, dtype, layout):
    owner, query, known = problem(dtype=dtype)
    owner = np.array(owner, order=layout, copy=True)
    originals = owner.copy(), query.copy()
    model = fixed_model(owner)
    for _ in range(2):
        actual = model.transform(query)
        assert actual.dtype == np.float64
        assert_coordinates(actual, known)
    np.testing.assert_array_equal(owner, originals[0])
    np.testing.assert_array_equal(query, originals[1])


@pytest.mark.parametrize('policy', ['floating_errors', 'callback'])
def test_current_numerical_policy_is_rechecked_on_every_call(qr_state, policy):
    owner, query, known = problem()
    model = fixed_model(owner)
    model.transform(query)
    if policy == 'floating_errors':
        with np.errstate(all='raise'), observed_calls() as calls:
            actual = model.transform(query)
    else:
        previous = np.geterrcall()
        try:
            np.seterrcall(lambda *args: None)
            with observed_calls() as calls:
                actual = model.transform(query)
        finally:
            np.seterrcall(previous)
    assert calls['qr'] == 1
    assert_coordinates(actual, known)
    with observed_calls() as recovered:
        assert_coordinates(model.transform(query), known)
    assert recovered['qr'] == int(not qr_state)


@pytest.mark.parametrize('fault', ['context', 'storage', 'publication'])
def test_optional_failure_preserves_result_and_allows_fresh_retry(qr_state, monkeypatch, fault):
    owner, query, known = problem()
    model = fixed_model(owner)
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(True)
        raise MemoryError('optional preparation failure')
    with monkeypatch.context() as patch:
        if fault == 'context':
            patch.setattr(_projection, '_qr_native_context', fail)
        elif fault == 'storage':
            patch.setattr(_nnls._QRPrepared, '__new__', staticmethod(fail))
        else:
            patch.setattr(_projection, '_qr_registry_publish', fail)
        # A late publication failure must not rerun the already completed solve.
        with observed_calls() as calls:
            actual = model.transform(query)
        assert calls['qr'] == 1
    assert bool(attempts) is qr_state
    assert _projection._qr_registry_get(owner) is None
    assert_coordinates(actual, known)
    assert_coordinates(model.transform(query), known)
    assert (_projection._qr_registry_get(owner) is not None) is qr_state


@pytest.mark.parametrize('fault', ['warning', 'error'])
def test_replaced_required_qr_keeps_original_warning_and_fallback_semantics(qr_state, monkeypatch, fault):
    owner, query, _ = problem()
    model = fixed_model(owner)
    model.transform(query)
    original = _nnls.linalg.qr
    def required(*args, **kwargs):
        if fault == 'error':
            raise RuntimeError('required QR failure')
        warnings.warn('required QR warning', RuntimeWarning)
        return original(*args, **kwargs)
    monkeypatch.setattr(_nnls.linalg, 'qr', required)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always')
        actual = model.transform(query)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always')
        expected = reference(owner, query)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]
    assert len(actual_warnings) == int(fault == 'warning')
    # RuntimeError is an existing QR-to-auto fallback, not an uncaught error.
    assert_coordinates(actual, expected)


def test_required_solve_error_does_not_publish_partial_preparation(qr_state, monkeypatch):
    owner, query, known = problem()
    model = fixed_model(owner)
    def fail(*args, **kwargs):
        raise ValueError('required reduced solve failed')
    with monkeypatch.context() as patch:
        patch.setattr(_nnls, '_scipy_rows', fail)
        with pytest.raises(ValueError, match='required reduced solve failed'):
            model.transform(query)
    assert _projection._qr_registry_get(owner) is None
    assert_coordinates(model.transform(query), known)
    assert (_projection._qr_registry_get(owner) is not None) is qr_state


@pytest.mark.parametrize('control', ['below_geometry', 'short_query', 'small_cap', 'auto'])
def test_ineligible_controls_do_not_inspect_provider_or_cache(qr_state, control):
    owner, query, _ = problem(features=256 if control == 'below_geometry' else 1024)
    model = fixed_model(owner)
    if control == 'short_query':
        query = query[:1]
    elif control == 'small_cap':
        model.nnls_max_iter = 95
    elif control == 'auto':
        model.projection_solver = 'auto'
    expected = reference(owner, query, cap=model.nnls_max_iter, solver=model.projection_solver)
    with observed_calls() as calls:
        actual = model.transform(query)
    assert calls['provider'] == calls['context'] == calls['publish'] == 0
    assert _projection._qr_registry_get(owner) is None
    assert_coordinates(actual, expected)


def immutable_entry():
    # Storage/weak-ownership unit fixture only: never authorizes a numerical call.
    design = np.eye(2, dtype=np.float64)
    return _nnls._QRPrepared(design, design, design, ('test-storage-only',))


def test_prepared_storage_is_immutable_and_dropped_by_transport(qr_state):
    entry = immutable_entry()
    assert copy(entry) is None and deepcopy(entry) is None
    assert pickle.loads(pickle.dumps(entry)) is None
    for factor in entry.factors():
        assert not factor.flags.writeable
        with pytest.raises(ValueError):
            factor.flags.writeable = True


def test_live_owners_are_not_evicted_and_dead_owners_are_released(qr_state):
    entry = immutable_entry()
    owners = [np.ones((2, 2)) for _ in range(18)]
    for owner in owners:
        _projection._qr_registry_publish(owner, entry)
    assert len(_projection._qr_registry) == 18
    assert all(_projection._qr_registry_get(owner) is entry for owner in owners)
    reference_ = weakref.ref(owners[0])
    del owners[0]
    gc.collect()
    assert reference_() is None and len(_projection._qr_registry) == 17
    # A delayed old weakref callback cannot remove a replacement entry.
    owner = owners[0]
    old = _projection._qr_registry[id(owner)][0]
    _projection._qr_registry_publish(owner, entry)
    replacement = _projection._qr_registry[id(owner)][0]
    old.__callback__(old)
    assert _projection._qr_registry[id(owner)][0] is replacement


def test_pid_renewal_does_not_acquire_inherited_lock(qr_state, monkeypatch):
    owner = np.ones((2, 2))
    _projection._qr_registry_publish(owner, immutable_entry())
    old = _projection._qr_registry
    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError('inherited lock acquired')
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(_projection, '_qr_registry_pid', os.getpid() - 1)
    monkeypatch.setattr(_projection, '_qr_registry_lock', ForbiddenLock())
    assert _projection._qr_registry_get(owner) is None
    assert _projection._qr_registry is not old and not _projection._qr_registry
    assert _projection._qr_registry_pid == os.getpid()


@pytest.mark.parametrize('change', ['version', 'registry', 'controller_property'])
def test_unsupported_runtime_state_declines_without_optional_callbacks(qr_state, monkeypatch, change):
    invoked = []
    def forbidden(*args, **kwargs):
        invoked.append(True)
        raise AssertionError('optional callback invoked')
    with monkeypatch.context() as patch:
        if change == 'version':
            patch.setattr(scipy, '__version__', 'unqualified-version')
        elif change == 'registry':
            class UnknownController:
                def __init__(self, *args, **kwargs):
                    forbidden()
            patch.setattr(threadpoolctl, '_ALL_CONTROLLERS', [UnknownController])
        else:
            patch.setattr(threadpoolctl.ThreadpoolController, 'info', property(forbidden))
        assert _projection._qr_mkl_controller() is None
        assert not invoked
    owner, query, known = problem()
    assert_coordinates(fixed_model(owner).transform(query), known)
    assert (_projection._qr_registry_get(owner) is not None) is qr_state
