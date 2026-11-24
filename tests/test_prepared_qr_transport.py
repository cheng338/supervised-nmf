"""Immutable QR transport, ownership, and actual worker regressions.

Stub-context cases isolate representation and decline boundaries.
Public process cases independently qualify the installed provider.
"""
from contextlib import nullcontext
import json
import os
import tempfile

import numpy as np
import pytest
from joblib import parallel_config
from joblib.externals.loky import reusable_executor

from supervised_nmf import _nnls, _projection as projection
from tests._prepared_qr_support import (
    assert_coordinates, assert_worker_binding, fixed_model, parent_inventory_under_limit,
    fail_large_qr_payload_view, problem, qr_state,
)


LIMIT = 1048576


def test_exact_bytes_are_immutable_without_a_second_copy():
    value = bytes(range(16))
    assert projection._qr_call_buffer(value, len(value)) is value


@pytest.mark.parametrize('difference', [-1, 1])
def test_bytes_length_mismatch_is_rejected(difference):
    with pytest.raises(ValueError, match='transport buffer'):
        projection._qr_call_buffer(b'12345678', 8 + difference)


@pytest.mark.parametrize('kind', ['bytearray', 'memoryview', 'subclass'])
def test_bytes_shortcut_does_not_accept_mutable_or_custom_storage(kind):
    class BytesSubclass(bytes):
        def __len__(self):
            raise AssertionError('custom length callback')
    value = {'bytearray': bytearray(b'12345678'),
             'memoryview': memoryview(b'12345678'),
             'subclass': BytesSubclass(b'12345678')}[kind]
    with pytest.raises(ValueError, match='transport buffer'):
        projection._qr_call_buffer(value, 8)


@pytest.mark.parametrize('readonly', [False, True])
def test_owning_array_is_snapshotted_even_if_readonly(readonly):
    value = np.arange(16, dtype=np.uint8)
    value.flags.writeable = not readonly
    result = projection._qr_call_buffer(value, 16)
    assert type(result) is bytes and result == bytes(range(16))
    value.flags.writeable = True
    value[:] = 0
    assert result == bytes(range(16))


def test_mutable_base_cannot_gain_authority_by_a_readonly_view():
    owner = bytearray(range(16))
    value = np.frombuffer(owner, dtype=np.uint8)
    value.flags.writeable = False
    result = projection._qr_call_buffer(value, 16)
    assert type(result) is bytes and result == bytes(range(16))
    owner[:] = b'\x00' * 16
    assert result == bytes(range(16))


def test_original_bytes_backed_array_reuses_immutable_owner():
    owner = bytes(range(16))
    value = np.frombuffer(owner, dtype=np.uint8)
    assert projection._qr_call_buffer(value, 16) is owner


@pytest.mark.parametrize('kind', ['dtype', 'ndim', 'stride', 'size'])
def test_invalid_array_representation_is_rejected(kind):
    value = np.arange(16, dtype=np.uint8)
    if kind == 'dtype':
        value = value.astype(np.int8)
    elif kind == 'ndim':
        value = value.reshape(4, 4)
    elif kind == 'stride':
        value = np.arange(32, dtype=np.uint8)[::2]
    size = 15 if kind == 'size' else 16
    with pytest.raises(ValueError, match='transport buffer'):
        projection._qr_call_buffer(value, size)


@pytest.mark.parametrize('mode', ['r', 'r+'])
def test_actual_mapping_ownership_is_preserved(tmp_path, mode):
    path = tmp_path / 'buffer.bin'
    path.write_bytes(bytes(range(16)))
    value = np.memmap(path, dtype=np.uint8, mode=mode, shape=(16,))
    try:
        result = projection._qr_call_buffer(value, 16)
        if mode == 'r':
            assert result is value and not result.flags.writeable
        else:
            assert type(result) is bytes and result == bytes(range(16))
            value[:] = 0
            value.flush()
            assert result == bytes(range(16))
    finally:
        value._mmap.close()


@pytest.fixture
def structural_context(monkeypatch):
    # This token is deliberately not a native proof or numerical permission.
    native = (('fixture-only', 1),)
    context = (os.getpid(), 'structural-context', native)
    calls = []
    def native_context():
        calls.append('native-context')
        return context
    def context_reader():
        calls.append('child-context')
        return context
    class Controller:
        def limit(self, **kwargs):
            assert kwargs == {'limits': 1}
            calls.append('limit')
            return nullcontext()
    monkeypatch.setattr(projection, '_qr_native_context', native_context)
    monkeypatch.setattr(projection, '_native_controller', lambda: Controller())
    monkeypatch.setattr(projection, '_qr_context', context_reader)
    return context, calls


@pytest.mark.parametrize('features,rank', [(1024, 32), (4095, 32), (4096, 32),
                                          (4097, 32), (8192, 64)])
def test_parent_representation_keeps_exact_threshold_and_context_checks(
        structural_context, monkeypatch, features, rank):
    context, calls = structural_context
    design = np.ones((features, rank), dtype=np.float64)
    entry = _nnls._QRPrepared(design, design, np.eye(rank), context)
    def prepared(*args, **kwargs):
        assert len(args) == 1 and args[0] is design
        assert kwargs == {'_prepared': entry, '_owned': True, '_allow_prepare': False}
        return entry
    monkeypatch.setattr(_nnls, '_prepare_qr_entry', prepared)
    returned, payload = projection._qr_parent_payload(design, 1, entry, False)
    assert returned is entry and payload[0] == (features, rank)
    assert calls == ['native-context', 'limit']
    for value, original in zip(payload[1:4], entry[1:4]):
        if len(original) <= LIMIT:
            assert value is original and type(value) is bytes
        else:
            assert type(value) is np.ndarray and value.dtype == np.uint8
            assert value.base is original and value.nbytes == len(original)
    rebound = projection._qr_call_bind(payload)
    assert type(rebound) is _nnls._QRPrepared
    assert calls[-1] == 'child-context'
    assert rebound[0] == entry[0] and rebound[4:] == entry[4:]
    for actual, expected in zip(rebound[1:4], entry[1:4]):
        assert bytes(actual) == expected
    assert rebound.matches(design, context)
    changed = design.copy()
    changed[0, 0] = 2
    assert not rebound.matches(changed, context)
    for factor in rebound.factors():
        with pytest.raises(ValueError):
            factor.flags.writeable = True


@pytest.mark.parametrize('fault', ['native-state', 'missing-context', 'length', 'mutable'])
def test_representation_never_bypasses_fresh_context_or_shape_proof(
        structural_context, monkeypatch, fault):
    context, calls = structural_context
    design = np.ones((1024, 32), dtype=np.float64)
    entry = _nnls._QRPrepared(design, design, np.eye(32), context)
    values = list(entry[1:4])
    payload = [entry[0], *values, entry[4], context[2]]
    if fault == 'native-state':
        payload[-1] = (('different-native-state', 1),)
    elif fault == 'missing-context':
        monkeypatch.setattr(projection, '_qr_context', lambda: None)
    elif fault == 'length':
        payload[1] = payload[1][:-1]
    else:
        payload[1] = bytearray(entry[1])
    assert projection._qr_call_bind(tuple(payload)) is None
    if fault != 'missing-context':
        assert calls == ['child-context']


def test_large_array_payload_storage_failure_declines_without_retaining_entry(
        structural_context, monkeypatch):
    context, calls = structural_context
    design = np.ones((4097, 32), dtype=np.float64)
    entry = _nnls._QRPrepared(design, design, np.eye(32), context)
    monkeypatch.setattr(_nnls, '_prepare_qr_entry', lambda *args, **kwargs: entry)
    with fail_large_qr_payload_view(monkeypatch) as failures:
        assert projection._qr_parent_payload(design, 1, entry, False) == (None, None)
    assert failures == [design.nbytes]


@pytest.mark.parametrize('features', [4095, 4096, 4097])
def test_actual_public_process_transport_at_mapping_boundary(qr_state, tmp_path, monkeypatch, features):
    owner, rows, known = problem(features=features, rank=32, rows=512)
    model = fixed_model(owner, workers=2, batch=64)
    directory = tmp_path / 'receipts'
    directory.mkdir()
    receipt_directory = str(directory)
    parent_inventory = parent_inventory_under_limit(model)
    def observed_shared_block(*args, **kwargs):
        # A by-value observer wraps only the optional transport entry. It invokes
        # the untouched required numerical implementation in the actual child.
        import json
        import os
        import tempfile
        from supervised_nmf import _projection as implementation
        from tests._prepared_qr_support import observed_worker_binding
        payload = args[-1]
        records = []
        for value in payload[1:4]:
            records.append(dict(kind=type(value).__name__,
                size=len(value) if type(value) is bytes else value.nbytes,
                readonly=None if type(value) is bytes else not value.flags.writeable,
                mode=getattr(value, 'mode', None)))
        with observed_worker_binding(payload, parent_inventory) as record:
            answer = implementation._project_shared_block(*args, **kwargs)
        fd, path = tempfile.mkstemp(prefix='block-', suffix='.json', dir=receipt_directory)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), rows=len(args[1]), buffers=records,
                           **record), stream)
        return answer
    monkeypatch.setattr(projection, '_project_shared_block', observed_shared_block)
    before_owner, before_rows = owner.copy(), rows.copy()
    with tempfile.TemporaryDirectory(prefix='q', dir=tmp_path.parent) as folder, \
            parallel_config(temp_folder=folder):
        try:
            for _ in range(2):
                result = model.transform(rows)
                assert_coordinates(result, known)
                assert not np.shares_memory(result, owner)
                assert not np.shares_memory(result, rows)
        finally:
            executor = reusable_executor._executor
            if executor is not None:
                executor.shutdown(wait=True, kill_workers=True)
    np.testing.assert_array_equal(owner, before_owner)
    np.testing.assert_array_equal(rows, before_rows)
    assert (projection._qr_registry_get(owner) is not None) is qr_state
    records = [json.loads(path.read_text()) for path in directory.glob('*.json')]
    assert len(records) == 16 * int(qr_state)
    assert all(row['rows'] == 64 and row['pid'] != os.getpid() for row in records)
    assert len({row['pid'] for row in records}) <= 2
    sizes = [features * 32 * 8, features * 32 * 8, 32 * 32 * 8]
    expected = [('memmap' if size > LIMIT else 'bytes')
                for size in sizes]
    for row in records:
        assert_worker_binding(row)
        assert [value['kind'] for value in row['buffers']] == expected
        assert [value['size'] for value in row['buffers']] == sizes
        assert all(value['mode'] == 'r' and value['readonly']
                   for value in row['buffers'] if value['kind'] == 'memmap')
