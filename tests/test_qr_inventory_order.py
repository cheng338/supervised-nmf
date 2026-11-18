"""Canonical token contracts; these unit fixtures do not prove provider admission.

The real-process tests check admission and reuse on the actual runtime.
These scoped units isolate inventory representation after the positive gate.
"""
import copy
import itertools

import pytest

from supervised_nmf import _projection as p


RECORDS = [
    dict(user_api='openmp', internal_api='openmp', num_threads=1, prefix='libiomp',
         filepath='/runtime/libiomp5.so', version=None),
    dict(user_api='openmp', internal_api='openmp', num_threads=1, prefix='libgomp',
         filepath='/runtime/libgomp.so.1', version=None),
    dict(user_api='blas', internal_api='mkl', num_threads=1, prefix='libmkl_rt',
         filepath='/runtime/libmkl_rt.so', version='2026.2-Product', threading_layer='intel'),
]


class QualifiedInventoryStub:
    """Only models the result of an already successful gate for unit coverage."""
    def __init__(self, rows):
        self.rows = rows
        self.lib_controllers = [object() for _ in rows]
        self.info_calls = 0

    def info(self):
        self.info_calls += 1
        return self.rows

    def replace(self, rows):
        self.rows = rows
        self.lib_controllers = [object() for _ in rows]


@pytest.fixture
def inventory(monkeypatch):
    controller = QualifiedInventoryStub(copy.deepcopy(RECORDS))
    gate_calls = []
    def gate():
        gate_calls.append(True)
        return controller
    monkeypatch.setattr(p, '_qr_mkl_controller', gate)
    return controller, gate_calls


def test_all_provider_permutations_have_one_unchanged_complete_token(inventory):
    controller, gate_calls = inventory
    expected_records = tuple(sorted((tuple(sorted(row.items())) for row in RECORDS), key=repr))
    first = None
    for ordering in itertools.permutations(RECORDS):
        rows = copy.deepcopy(list(ordering))
        before = copy.deepcopy(rows)
        controller.replace(rows)
        value = p._qr_native_context()
        assert value[2] == expected_records
        assert rows == before
        assert controller.rows is rows
        if first is None:
            first = value
        else:
            assert value == first
    assert len(gate_calls) == controller.info_calls == 6


def test_same_filepath_distinct_records_and_duplicates_remain_exact(inventory):
    controller, _ = inventory
    altered = dict(RECORDS[0], num_threads=2)
    records = [copy.deepcopy(RECORDS[0]), copy.deepcopy(RECORDS[0]), altered, copy.deepcopy(RECORDS[2])]
    expected_records = tuple(sorted((tuple(sorted(row.items())) for row in records), key=repr))
    values = []
    for ordering in itertools.permutations(records):
        controller.replace(copy.deepcopy(list(ordering)))
        value = p._qr_native_context()
        assert len(value[2]) == 4
        assert value[2] == expected_records
        assert value[2].count(tuple(sorted(RECORDS[0].items()))) == 2
        values.append(value)
    assert all(value == values[0] for value in values)


@pytest.mark.parametrize('change', [
    'threads', 'filepath', 'version', 'prefix', 'user_api', 'internal_api', 'threading_layer',
    'new_field', 'removed_field', 'missing_record', 'extra_record', 'extra_duplicate', 'duplicate_replacement',
])
def test_any_inventory_content_or_multiplicity_change_changes_token(inventory, change):
    controller, _ = inventory
    original = p._qr_native_context()
    rows = copy.deepcopy(RECORDS)
    if change == 'threads':
        rows[0]['num_threads'] = 2
    elif change == 'filepath':
        rows[0]['filepath'] += '.changed'
    elif change == 'version':
        rows[0]['version'] = 'changed'
    elif change in ('prefix', 'user_api', 'internal_api'):
        rows[0][change] += '-changed'
    elif change == 'threading_layer':
        rows[2]['threading_layer'] = 'sequential'
    elif change == 'new_field':
        rows[0]['new_provider_state'] = True
    elif change == 'removed_field':
        del rows[0]['version']
    elif change == 'missing_record':
        rows.pop()
    elif change == 'extra_record':
        rows.append(dict(rows[0], filepath='/runtime/other.so'))
    elif change == 'extra_duplicate':
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[1] = copy.deepcopy(rows[0])
    controller.replace(rows)
    assert p._qr_native_context() != original


@pytest.mark.parametrize('bad', ['non_list', 'record_type', 'non_string_key', 'unsupported_value',
                                'missing_threads', 'bool_threads', 'float_threads', 'zero_threads',
                                'negative_threads', 'inventory_length'])
def test_existing_state_validation_still_declines(inventory, bad):
    controller, _ = inventory
    rows = copy.deepcopy(RECORDS)
    if bad == 'non_list':
        controller.replace(tuple(rows))
    elif bad == 'inventory_length':
        controller.replace(rows)
        controller.lib_controllers.pop()
    else:
        if bad == 'record_type':
            rows[0] = tuple(rows[0].items())
        elif bad == 'non_string_key':
            rows[0][3] = 'value'
        elif bad == 'unsupported_value':
            rows[0]['new_field'] = []
        elif bad == 'missing_threads':
            del rows[0]['num_threads']
        else:
            rows[0]['num_threads'] = {'bool_threads': True, 'float_threads': 1.0,
                'zero_threads': 0, 'negative_threads': -1}[bad]
        controller.replace(rows)
    assert p._qr_native_context() is None


def test_declined_positive_gate_is_never_turned_into_a_token(monkeypatch):
    calls = []
    def gate():
        calls.append(True)
        return None
    monkeypatch.setattr(p, '_qr_mkl_controller', gate)
    assert p._qr_native_context() is None
    assert calls == [True]


def test_inventory_read_error_keeps_existing_exception_semantics(monkeypatch):
    class Failing:
        def info(self):
            raise RuntimeError('fresh native state unavailable')
    monkeypatch.setattr(p, '_qr_mkl_controller', lambda: Failing())
    with pytest.raises(RuntimeError, match='fresh native state unavailable'):
        p._qr_native_context()


def test_pid_and_controller_identity_remain_part_of_context(inventory, monkeypatch):
    controller, _ = inventory
    original = p._qr_native_context()
    with monkeypatch.context() as patch:
        patch.setattr(p.os, 'getpid', lambda: original[0] + 10000)
        changed_pid = p._qr_native_context()
        assert changed_pid[0] != original[0]
        assert changed_pid[1:] == original[1:]
    other = QualifiedInventoryStub(copy.deepcopy(RECORDS))
    monkeypatch.setattr(p, '_qr_mkl_controller', lambda: other)
    changed_controller = p._qr_native_context()
    assert changed_controller[0] == original[0]
    assert changed_controller[1] != original[1]
    assert changed_controller[2] == original[2]
