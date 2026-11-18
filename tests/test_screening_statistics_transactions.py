"""Durable JSON authority, immutable payload repair and accounting."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import zipfile

import numpy as np
import pytest

from tests._screening_recovery_support import (
    BinaryScreening, assert_accounting, assert_observed_equal, estimator, event_paths,
    metadata, observe_science, problem, receipts, recovery, run, science,
    screening_policy, stable_bytes, stop_at, study,
)

pytestmark = pytest.mark.usefixtures('screening_policy')


def payload_for(event):
    return event.with_name('payload-' + event.stem.removeprefix('event-') + '.npz')


def committed(path):
    with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
        run(path)
    events = event_paths(path)
    assert len(events) == 1 and payload_for(events[0]).is_file()
    return events[0]


def mutate_json(path, change):
    envelope = json.loads(path.read_bytes())
    change(envelope['payload'])
    envelope['sha256'] = recovery.sha(recovery._plain_json(envelope['payload']))
    path.write_bytes(recovery._plain_json(envelope))


@pytest.mark.parametrize('stage', [
    'before_payload_write', 'after_payload_write', 'before_metadata_write',
    'after_metadata_write', 'before_payload_commit', 'after_payload_commit',
    'before_event_commit', 'after_event_commit'])
def test_only_json_replacement_commits_a_recoverable_event(tmp_path, stage):
    with stop_at(stage), pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    is_committed = stage == 'after_event_commit'
    assert len(event_paths(tmp_path)) == int(is_committed)
    payloads = sorted((tmp_path/'statistics-events').glob('payload-*.npz'))
    assert len(payloads) == int(stage in ('after_payload_commit','before_event_commit','after_event_commit'))
    preserved = {p:p.read_bytes() for p in payloads + event_paths(tmp_path)}
    assert receipts(tmp_path) == []
    with observe_science() as observed:
        result = run(tmp_path)
    assert len(observed['fits']) == (1 if is_committed else 2)
    assert all(p.read_bytes() == value for p,value in preserved.items())
    assert len(event_paths(tmp_path)) == 2
    assert result['screening_statistics_recovery']['durable_event_count'] == 2
    assert result['screening_statistics_recovery']['current_call']['restored_events'] == int(is_committed)
    assert_accounting(result)


@pytest.mark.parametrize('damage', ['missing','truncated','directory','checksum','duplicate'])
def test_valid_json_survives_whole_payload_or_member_loss(tmp_path, damage):
    event = committed(tmp_path)
    before = event.read_bytes()
    old_seconds = metadata(event)['origin_seconds']
    payload = payload_for(event)
    if damage in ('missing','directory'):
        payload.unlink()
        if damage == 'directory':
            payload.mkdir()
    elif damage == 'truncated':
        payload.write_bytes(payload.read_bytes()[:41])
    else:
        with zipfile.ZipFile(payload) as archive:
            members = [(info.filename,archive.read(info.filename)) for info in archive.infolist()]
        if damage == 'checksum':
            members = [(name,raw[:-1]+bytes([raw[-1]^1]) if name=='scores_.npy' else raw) for name,raw in members]
        else:
            members.append(next(row for row in members if row[0]=='scores_.npy'))
        def write():
            with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_STORED) as archive:
                for name,raw in members:
                    archive.writestr(name,raw)
        if damage == 'duplicate':
            with pytest.warns(UserWarning,match='Duplicate name'):
                write()
        else:
            write()
    damaged = payload.read_bytes() if payload.is_file() else None
    with observe_science() as observed:
        report = run(tmp_path)
    assert len(observed['fits']) == 2 and event.read_bytes() == before
    if damaged is not None:
        assert payload.read_bytes() == damaged
    value = report['screening_statistics_recovery']
    assert value['durable_event_count'] == 3 and value['orphan_event_count'] == 1
    assert len(value['current_call']['unusable_payload_events']) == 1
    old = next(row for row in value['events'] if row['event_id']==metadata(event)['event_id'])
    assert old['origin_seconds'] == old_seconds and old['consumers'] == []
    assert_accounting(report)


@pytest.mark.parametrize('damage',['malformed','checksum','identity','shape'])
def test_json_authority_corruption_is_fatal_before_fit(tmp_path, damage):
    event = committed(tmp_path)
    if damage == 'malformed':
        event.write_bytes(b'{broken')
    elif damage == 'checksum':
        value = json.loads(event.read_bytes());value['payload']['origin_seconds'] += 1.
        event.write_text(json.dumps(value))
    elif damage == 'identity':
        mutate_json(event,lambda v:v.update(identity='0'*64))
    else:
        mutate_json(event,lambda v:v['state']['scalars'].update(n_samples_seen_=1))
    preserved = event.read_bytes()
    with observe_science() as observed, pytest.raises(recovery.StatisticsRecoveryError):
        run(tmp_path)
    assert not observed['fits'] and receipts(tmp_path) == [] and event.read_bytes() == preserved


def test_completed_resume_never_opens_even_missing_payloads(tmp_path, monkeypatch):
    original = run(tmp_path)
    events = event_paths(tmp_path)
    before = {p:p.read_bytes() for p in events}
    old_receipts = {p:p.read_bytes() for p in (tmp_path/'folds').glob('*.json')}
    for event in events:
        payload_for(event).unlink()
    def forbidden(*args,**kwargs):
        raise AssertionError('completed metadata accounting opened a learned payload')
    monkeypatch.setattr(zipfile,'ZipFile',forbidden)
    with observe_science() as observed:
        resumed = run(tmp_path)
    assert not any(observed.values()) and science(resumed) == science(original)
    assert all(p.read_bytes()==v for p,v in {**before,**old_receipts}.items())
    assert resumed['compute_seconds'] == original['compute_seconds']
    assert resumed['screening_statistics_recovery']['events'] == original['screening_statistics_recovery']['events']
    assert_accounting(resumed)


def test_missing_json_referenced_by_receipt_is_not_recreated(tmp_path):
    run(tmp_path)
    event = event_paths(tmp_path)[0]
    payload = payload_for(event);before = payload.read_bytes()
    event.unlink()
    with observe_science() as observed, pytest.raises(recovery.StatisticsRecoveryError):
        run(tmp_path)
    assert not observed['fits'] and not event.exists() and payload.read_bytes()==before


def test_json_change_after_index_is_not_a_payload_miss(tmp_path, monkeypatch):
    event = committed(tmp_path)
    original = recovery.StatisticsEventStore._index
    changed = []
    def index(store):
        value = original(store)
        if not changed:
            mutate_json(event,lambda row:row.update(origin_seconds=row['origin_seconds']+1.))
            changed.append(True)
        return value
    monkeypatch.setattr(recovery.StatisticsEventStore,'_index',index)
    with observe_science() as observed, pytest.raises(recovery.StatisticsRecoveryError,match='changed after indexing'):
        run(tmp_path)
    assert changed and not observed['fits'] and receipts(tmp_path)==[]


def test_json_rename_failure_leaves_payload_uncommitted(tmp_path, monkeypatch):
    original = Path.replace
    def failure(path,target):
        if path.name.startswith('.event-') and str(target).endswith('.json'):
            raise OSError('controlled JSON replacement failure')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',failure)
    result = run(tmp_path)
    assert not event_paths(tmp_path)
    assert len(list((tmp_path/'statistics-events').glob('payload-*.npz'))) == 2
    value = result['screening_statistics_recovery']
    assert value['durable_event_count']==0 and len(value['current_call']['publication_failures'])==2
    assert result['screening_statistics_fits']==2 and all(r['status']=='valid' for r in result['representation_folds'])
    assert_accounting(result)


def test_orphan_payload_does_not_recreate_a_historical_event(tmp_path):
    with stop_at('after_payload_commit'),pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    assert not event_paths(tmp_path)
    payload = next((tmp_path/'statistics-events').glob('payload-*.npz'))
    before = payload.read_bytes()
    (tmp_path/'identity.json').unlink()
    report = run(tmp_path)
    assert payload.read_bytes()==before and len(event_paths(tmp_path))==2
    assert report['screening_statistics_recovery']['durable_event_count']==2
    assert report['screening_statistics_fits']==2
    assert_accounting(report)


def test_payload_identifier_collision_preserves_existing_bytes(tmp_path, monkeypatch):
    created = []
    def hook(stage,details):
        if stage=='before_payload_commit':
            path = tmp_path/'statistics-events'/('payload-'+details['event_id']+'.npz')
            path.write_bytes(b'prior payload must survive')
            created.append(path)
    monkeypatch.setattr(recovery,'_event_hook',hook)
    with pytest.raises(recovery.StatisticsRecoveryError,match='collision'):
        run(tmp_path)
    assert len(created)==1 and created[0].read_bytes()==b'prior payload must survive'
    assert not event_paths(tmp_path) and receipts(tmp_path)==[]


@pytest.mark.parametrize('error_type', [ValueError, OSError])
def test_postcommit_observer_error_aborts_without_double_charged_receipt(tmp_path, error_type):
    with stop_at('after_event_commit', error_type=error_type), pytest.raises(recovery.StatisticsRecoveryError):
        run(tmp_path)
    assert len(event_paths(tmp_path)) == 1
    assert receipts(tmp_path) == []
    report = run(tmp_path)
    assert_accounting(report)
    value = report['screening_statistics_recovery']
    assert value['durable_event_count'] == 2
    assert value['receipt_event_backed_fits'] == 1
    assert report['screening_statistics_fits'] == 1


@pytest.mark.parametrize('missing_identity', [False, True])
def test_event_directory_enumeration_denial_is_not_an_empty_ledger(tmp_path, monkeypatch, missing_identity):
    with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    saved = {p: p.read_bytes() for p in event_paths(tmp_path)}
    if missing_identity:
        (tmp_path / 'identity.json').unlink()
    original = os.scandir
    def denied(path):
        if Path(path) == tmp_path / 'statistics-events':
            raise PermissionError('controlled existing-directory enumeration denial')
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'scandir', denied)
        with observe_science() as observed, pytest.raises(recovery.StatisticsRecoveryError):
            run(tmp_path)
    assert not observed['fits']
    assert receipts(tmp_path) == []
    assert all(p.read_bytes() == raw for p, raw in saved.items())
    assert (tmp_path / 'identity.json').exists() is (not missing_identity)


def test_partial_event_directory_enumeration_is_not_accepted(tmp_path, monkeypatch):
    run(tmp_path)
    saved = {p: p.read_bytes() for p in event_paths(tmp_path)}
    original = os.scandir
    class Partial:
        def __init__(self, path):
            self.stream = original(path)
            self.first = True
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def __iter__(self):
            return self
        def __next__(self):
            if self.first:
                self.first = False
                return next(self.stream)
            raise OSError('controlled partial directory enumeration')
    def partial(path):
        return Partial(path) if Path(path) == tmp_path / 'statistics-events' else original(path)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'scandir', partial)
        with observe_science() as observed, pytest.raises(recovery.StatisticsRecoveryError):
            run(tmp_path)
    assert not any(observed.values())
    assert all(p.read_bytes() == raw for p, raw in saved.items())


def test_reversed_event_creation_order_has_identical_completed_ledger(tmp_path, monkeypatch):
    sequence = iter([3, 2, 1])
    monkeypatch.setattr(recovery, 'uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(int=next(sequence))))
    X, y = problem()
    def call():
        return study.calibration_knn_grid(
            estimator(), {'screen__score_threshold': [0., 2.], 'knn__n_neighbors': [1]},
            X, y, cv=3, seed=29, threads=1, checkpoint_dir=tmp_path,
            screening_statistics_recovery='early')
    original = call()
    assert [row['event_id'] for row in original['screening_statistics_recovery']['events']] == [uuid.UUID(int=i).hex for i in (1, 2, 3)]
    assert [row['fold'] for row in original['screening_statistics_recovery']['events']] == [2, 1, 0]
    before = {p: p.read_bytes() for p in event_paths(tmp_path)}
    completed = call()
    assert science(original) == science(completed)
    for field in ('events', 'durable_statistics_seconds', 'receipt_statistics_seconds',
                  'receipt_event_backed_seconds', 'union_statistics_seconds', 'union_tuning_seconds'):
        assert original['screening_statistics_recovery'][field] == completed['screening_statistics_recovery'][field]
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert_accounting(original)
    assert_accounting(completed)
