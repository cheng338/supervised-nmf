"""Focused receipt/accounting validation; no model fit or learned payload is needed."""
from copy import deepcopy
from pathlib import Path
import zipfile

import pytest

from supervised_nmf import study
from supervised_nmf import _screening_recovery as recovery


EVENT = 'e' * 32
ORIGIN = 'a' * 64
FINGERPRINT = 'f' * 64
REPRESENTATIONS = ('1' * 64, '2' * 64)


def record(*, performed=True, key=ORIGIN, fold=0):
    return dict(key=key, fold=fold, screening_statistics_event_id=EVENT,
                screening_statistics_seconds=2.25 if performed else 0.0,
                screening_statistics_fit_performed=performed,
                screening_statistics_event_reused=not performed)


def store(*, fold=0, present=True):
    # These tests isolate consumer validation after the independently checked
    # metadata index. They do not stand in for real JSON/payload recovery tests.
    result = object.__new__(recovery.StatisticsEventStore)
    result.events = ({EVENT: dict(event_id=EVENT, fold=fold,
                                origin_record_key=ORIGIN, origin_seconds=2.25)} if present else {})
    result.layout_eligible = False
    result.current = {}
    return result


def key(representation, fold):
    return study._digest(dict(identity=FINGERPRINT, representation=representation, fold=fold))


def save_late_receipt(directory, *, fold=1):
    identifier = key(REPRESENTATIONS[-1], fold)
    value = record(performed=False, key=identifier, fold=fold)
    value.update(identity=FINGERPRINT, state='completed')
    path = directory/'folds'/f'{identifier}.json'
    study._write_receipt(path, value)
    return path, value


def no_payload(*args, **kwargs):
    raise AssertionError('consumer validation must not open learned payloads')


def test_original_and_reused_consumers_keep_exact_union(monkeypatch):
    monkeypatch.setattr(zipfile, 'ZipFile', no_payload)
    original = record()
    reused = record(performed=False, key='b'*64)
    owner = store()
    before = deepcopy(owner.events)
    events, consumers, backed, fits = owner.validate_consumers([original, reused])
    assert events == before == owner.events
    assert consumers == {EVENT: [ORIGIN, 'b'*64]} and backed == 2.25 and fits == 1
    costs = dict(screening_statistics_seconds=2.25, representation_fit_seconds=1.0)
    report = owner.summary([original, reused], costs)
    assert report['receipt_statistics_seconds'] == report['receipt_event_backed_seconds'] == 2.25
    assert report['durable_statistics_seconds'] == report['union_statistics_seconds'] == 2.25
    assert report['union_tuning_seconds'] == 3.25 and report['orphan_event_count'] == 0


@pytest.mark.parametrize('case', [
    'missing_event', 'wrong_fold', 'boolean_fold', 'neither_performed_nor_reused',
    'both_performed_and_reused', 'integer_performed', 'integer_reused',
    'origin_key', 'origin_seconds', 'restored_nonzero', 'boolean_seconds',
    'negative_seconds', 'nan_seconds', 'infinite_seconds'])
def test_invalid_event_consumer_is_fatal(case):
    value = record()
    if case == 'missing_event':
        value['screening_statistics_event_id'] = 'd'*32
    elif case == 'wrong_fold':
        value['fold'] = 1
    elif case == 'boolean_fold':
        value['fold'] = False
    elif case == 'neither_performed_nor_reused':
        value = record(performed=False)
        value['screening_statistics_event_reused'] = False
    elif case == 'both_performed_and_reused':
        value['screening_statistics_event_reused'] = True
    elif case == 'integer_performed':
        value['screening_statistics_fit_performed'] = 1
    elif case == 'integer_reused':
        value['screening_statistics_event_reused'] = 0
    elif case == 'origin_key':
        value['key'] = 'b'*64
    elif case == 'origin_seconds':
        value['screening_statistics_seconds'] = 2.5
    elif case == 'restored_nonzero':
        value = record(performed=False)
        value['screening_statistics_seconds'] = .25
    elif case == 'boolean_seconds':
        value = record(performed=False)
        value['screening_statistics_seconds'] = False
    elif case == 'negative_seconds':
        value['screening_statistics_seconds'] = -1.
    elif case == 'nan_seconds':
        value['screening_statistics_seconds'] = float('nan')
    elif case == 'infinite_seconds':
        value['screening_statistics_seconds'] = float('inf')
    with pytest.raises(recovery.StatisticsRecoveryError):
        store().validate_consumers([value])


def test_summary_uses_the_same_strict_flag_validation():
    value = record(performed=False)
    value['screening_statistics_event_reused'] = False
    with pytest.raises(recovery.StatisticsRecoveryError, match='consumer flags'):
        store().summary([value], dict(screening_statistics_seconds=0.0))


def test_fresh_no_folds_directory_skips_receipt_and_event_reads(tmp_path, monkeypatch):
    owner = store()
    def forbidden(*args, **kwargs):
        raise AssertionError('fresh consumer preflight performed a receipt/index read')
    monkeypatch.setattr(study, '_read_receipt', forbidden)
    monkeypatch.setattr(owner, 'validate_consumers', forbidden)
    original = Path.stat
    seen = []
    def stat(path, *args, **kwargs):
        if path == tmp_path/'folds':
            seen.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, owner)
    assert seen == [tmp_path/'folds'] and not (tmp_path/'attempts').exists()


def test_missing_earlier_work_does_not_hide_later_missing_authority(tmp_path, monkeypatch):
    path, value = save_late_receipt(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(zipfile, 'ZipFile', no_payload)
    with pytest.raises(recovery.StatisticsRecoveryError, match='missing or incompatible'):
        study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, store(fold=1, present=False))
    assert path.read_bytes() == before and not (tmp_path/'attempts').exists()


def test_valid_late_receipt_is_read_without_payload_or_rewrite(tmp_path, monkeypatch):
    path, value = save_late_receipt(tmp_path)
    before = path.read_bytes()
    original = study._read_receipt
    seen = []
    def read(candidate):
        seen.append(candidate)
        return original(candidate)
    monkeypatch.setattr(study, '_read_receipt', read)
    monkeypatch.setattr(zipfile, 'ZipFile', no_payload)
    owner = store(fold=1)
    study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, owner)
    assert seen == [tmp_path/'folds'/f'{key(rep,fold)}.json'
                    for rep in REPRESENTATIONS for fold in range(2)]
    assert path.read_bytes() == before and not (tmp_path/'attempts').exists()
    assert owner.events[EVENT]['origin_seconds'] == 2.25


@pytest.mark.parametrize('field', ['identity', 'key', 'state'])
def test_preflight_keeps_original_saved_receipt_identity_checks(tmp_path, field):
    path, value = save_late_receipt(tmp_path)
    value[field] = 'different'
    study._write_receipt(path, value)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='another run or is incomplete'):
        study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, store(fold=1))
    assert path.read_bytes() == before and not (tmp_path/'attempts').exists()


def test_preflight_preserves_original_receipt_checksum_failure(tmp_path):
    path, _ = save_late_receipt(tmp_path)
    path.write_bytes(b'{"payload":{},"sha256":"bad"}')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='integrity check'):
        study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, store(fold=1))
    assert path.read_bytes() == before and not (tmp_path/'attempts').exists()


def test_unreadable_existing_receipt_is_not_treated_as_absent(tmp_path, monkeypatch):
    path, _ = save_late_receipt(tmp_path)
    original = study._read_receipt
    def read(candidate):
        if candidate == path:
            raise PermissionError('controlled receipt denial')
        return original(candidate)
    monkeypatch.setattr(study, '_read_receipt', read)
    with pytest.raises(PermissionError, match='controlled receipt denial'):
        study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, store(fold=1))
    assert not (tmp_path/'attempts').exists()


def test_nondirectory_fold_authority_is_rejected(tmp_path):
    path = tmp_path/'folds'
    path.write_bytes(b'not a receipt directory')
    with pytest.raises(recovery.StatisticsRecoveryError, match='require a directory'):
        study._preflight_statistics_receipts(tmp_path, FINGERPRINT, REPRESENTATIONS, 2, store())
    assert path.read_bytes() == b'not a receipt directory'
