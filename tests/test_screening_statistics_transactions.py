"""Durable JSON authority, immutable payload repair and accounting."""
from collections import UserDict
from contextlib import contextmanager
import inspect
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import zipfile

import numpy as np
import pytest
from supervised_nmf import evaluation

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


def fail_read_bookkeeping(patch, **faults):
    initialize = recovery.StatisticsEventStore.__init__

    class ReadCounters(dict):
        def __setitem__(self, key, value):
            if key in faults:
                raise faults[key]
            super().__setitem__(key, value)

    class UnusablePayloads(list):
        def append(self, item):
            raise faults['unusable_payload_events']

    def initialize_with_faults(store, *args, **kwargs):
        initialize(store, *args, **kwargs)
        if 'unusable_payload_events' in faults:
            store.current['unusable_payload_events'] = UnusablePayloads()
        store.current = ReadCounters(store.current)

    patch.setattr(recovery.StatisticsEventStore, '__init__', initialize_with_faults)


def assert_read_failure_recovers(directory, event, event_bytes, payload_bytes):
    payload = payload_for(event)
    assert event_paths(directory) == [event] and receipts(directory) == []
    assert event.read_bytes() == event_bytes
    assert (payload.read_bytes() if payload.exists() else None) == payload_bytes
    with observe_science() as observed:
        resumed = run(directory)
    ledger = resumed['screening_statistics_recovery']
    restored = int(payload_bytes is not None)
    assert ledger['current_call']['restored_events'] == restored
    assert ledger['current_call']['published_events'] == 2 - restored
    assert resumed['screening_statistics_fits'] == len(observed['fits']) == 2 - restored
    assert ledger['current_call']['actual_statistics_fits'] == 2 - restored
    assert ledger['durable_event_count'] == 3 - restored and ledger['orphan_event_count'] == 1 - restored
    unusable = ledger['current_call']['unusable_payload_events']
    event_id = metadata(event)['event_id']
    assert [item['event_id'] for item in unusable] == ([] if restored else [event_id])
    prior = next(row for row in ledger['events'] if row['event_id'] == event_id)
    assert bool(prior['consumers']) is bool(restored)
    assert event.read_bytes() == event_bytes
    assert (payload.read_bytes() if payload.exists() else None) == payload_bytes
    assert_accounting(resumed)
    return resumed, observed


@pytest.mark.parametrize(('field', 'missing'), [
    ('array_read_seconds', False), ('array_read_seconds', True),
    ('restored_events', False), ('unusable_payload_events', True),
])
def test_read_bookkeeping_failure_aborts_before_fold_receipt(tmp_path, monkeypatch, field, missing):
    event = committed(tmp_path)
    payload = payload_for(event)
    if missing:
        payload.unlink()
    event_bytes = event.read_bytes()
    payload_bytes = None if missing else payload.read_bytes()
    primary = MemoryError(f'cannot record {field}')
    faults = {'array_read_seconds': RuntimeError('secondary read timer failure'), field: primary}
    with monkeypatch.context() as patch:
        fail_read_bookkeeping(patch, **faults)
        # A caller may start a genuine restore while handling an unrelated error.
        try:
            raise LookupError('handled by caller')
        except LookupError:
            with observe_science() as attempt, pytest.raises(recovery.StatisticsRecoveryError) as caught:
                run(tmp_path)
    assert caught.value.__cause__ is primary and not any(attempt.values())
    resumed, observed = assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)
    if field == 'array_read_seconds' and not missing:
        with observe_science() as reference:
            ordinary = run(tmp_path / 'ordinary', early=False)
        assert science(resumed) == science(ordinary)
        assert_observed_equal(observed, reference, compare_fits=False)


@pytest.mark.parametrize(('stage', 'error_type'), [
    ('payload', MemoryError), ('observer', ValueError),
    ('observer', KeyboardInterrupt), ('observer', SystemExit),
])
def test_read_timer_preserves_primary_error_or_interrupt(tmp_path, monkeypatch, stage, error_type):
    event = committed(tmp_path)
    event_bytes, payload_bytes = event.read_bytes(), payload_for(event).read_bytes()
    primary = error_type('primary read failure')

    def fail_payload(*args, **kwargs):
        raise primary

    def observer(boundary, details):
        if boundary == 'after_statistics_loaded':
            raise primary

    with monkeypatch.context() as patch:
        fail_read_bookkeeping(patch, array_read_seconds=OSError('secondary timer failure'))
        if stage == 'payload':
            patch.setattr(zipfile.ZipExtFile, 'read', fail_payload)
        else:
            patch.setattr(recovery, '_event_hook', observer)
        expected = recovery.StatisticsRecoveryError if isinstance(primary, Exception) else error_type
        with observe_science() as attempt, pytest.raises(expected) as caught:
            run(tmp_path)
    assert (caught.value.__cause__ if isinstance(primary, Exception) else caught.value) is primary
    assert not any(attempt.values())
    assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)


def fault_payload_read(patch, payload, *, read_error=None, failing_closes=(), translate=False):
    """Inject faults only on this committed archive and its scores member."""
    open_member = zipfile.ZipFile.open
    close_archive = zipfile.ZipFile.close
    seen = []

    def target(archive):
        return archive.mode == 'r' and archive.filename == str(payload)

    def closing(boundary):
        seen.append(boundary)
        if boundary in failing_closes:
            raise OSError('failed to close ' + boundary)

    def opened(archive, name, *args, **kwargs):
        member = open_member(archive, name, *args, **kwargs)
        if target(archive) and member.name == 'scores_.npy':
            read = member.read
            close = member.close

            def read_scores(*args, **kwargs):
                if read_error is not None:
                    seen.append('read failure')
                    if translate:
                        try:
                            raise MemoryError('handled inside the decoder')
                        except MemoryError:
                            raise read_error from None
                    raise read_error
                data = read(*args, **kwargs)
                seen.append('read success')
                return data

            def close_scores():
                was_open = not member.closed
                close()
                if was_open:
                    closing('member')

            patch.setattr(member, 'read', read_scores)
            patch.setattr(member, 'close', close_scores)
        return member

    def closed(archive):
        was_target = archive.fp is not None and target(archive)
        close_archive(archive)
        if was_target:
            closing('archive')

    patch.setattr(zipfile.ZipFile, 'open', opened)
    patch.setattr(zipfile.ZipFile, 'close', closed)
    return seen


@pytest.mark.parametrize(('primary_type', 'failing_closes'), [
    (MemoryError, ('archive',)),
    (KeyboardInterrupt, ('member', 'archive')),
    (SystemExit, ('member',)),
])
def test_archive_cleanup_preserves_fatal_read(tmp_path, monkeypatch, primary_type, failing_closes):
    event = committed(tmp_path)
    payload = payload_for(event)
    event_bytes, payload_bytes = event.read_bytes(), payload.read_bytes()
    primary = primary_type('failed to read scores')
    expected = recovery.StatisticsRecoveryError if primary_type is MemoryError else primary_type
    with monkeypatch.context() as patch:
        seen = fault_payload_read(patch, payload, read_error=primary, failing_closes=failing_closes)
        with observe_science() as attempt, pytest.raises(expected) as caught:
            run(tmp_path)
    assert seen == ['read failure', 'member', 'archive']
    assert (caught.value.__cause__ if primary_type is MemoryError else caught.value) is primary
    assert not any(attempt.values())
    resumed, observed = assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)
    with observe_science() as reference:
        ordinary = run(tmp_path / 'ordinary', early=False)
    assert science(resumed) == science(ordinary)
    assert_observed_equal(observed, reference, compare_fits=False)


def test_directory_read_allocation_failure_survives_payload_file_close(tmp_path, monkeypatch):
    event = committed(tmp_path)
    payload = payload_for(event)
    event_bytes, payload_bytes = event.read_bytes(), payload.read_bytes()
    primary = MemoryError('cannot read ZIP directory')
    open_file = io.open
    steps = []

    with monkeypatch.context() as patch:
        def opened(file, *args, **kwargs):
            stream = open_file(file, *args, **kwargs)
            if isinstance(file, (str, Path)) and Path(file) == payload and stream.mode == 'rb':
                close = stream.close

                def fail_read(*args, **kwargs):
                    steps.append('read')
                    raise primary

                def fail_close():
                    if not stream.closed:
                        close()
                        steps.append('close')
                        raise OSError('closing the payload failed')

                patch.setattr(stream, 'read', fail_read)
                patch.setattr(stream, 'close', fail_close)
            return stream

        patch.setattr(io, 'open', opened)
        with observe_science() as attempt, pytest.raises(recovery.StatisticsRecoveryError) as caught:
            run(tmp_path)
    assert caught.value.__cause__ is primary and steps == ['read', 'close']
    assert not any(attempt.values())
    assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)


@pytest.mark.parametrize(('close_stage', 'corrupt', 'translated'), [
    ('member', False, False), ('archive', False, False), (None, True, False),
    (None, True, True), ('archive', True, True),
])
def test_recoverable_archive_error_charges_orphan_and_replacement(tmp_path, monkeypatch, close_stage,
                                                                 corrupt, translated):
    event = committed(tmp_path)
    payload = payload_for(event)
    before = stable_bytes(tmp_path)
    origin = metadata(event)
    read_error = zipfile.BadZipFile('corrupt scores') if corrupt else None
    with monkeypatch.context() as patch:
        seen = fault_payload_read(patch, payload, read_error=read_error,
                                  failing_closes=(close_stage,) if close_stage else (),
                                  translate=translated)
        # A handled allocation failure in the caller is unrelated to this restore.
        try:
            raise MemoryError('previously handled allocation error')
        except MemoryError:
            with observe_science() as observed:
                result = run(tmp_path)
    assert seen == ['read failure' if corrupt else 'read success', 'member', 'archive']
    assert before.items() <= stable_bytes(tmp_path).items()
    ledger = result['screening_statistics_recovery']
    call = ledger['current_call']
    unusable, = call['unusable_payload_events']
    assert unusable['event_id'] == origin['event_id']
    assert unusable['error_type'] == ('OSError' if close_stage else 'BadZipFile')
    assert unusable['error'] == ('failed to close ' + close_stage if close_stage else 'corrupt scores')
    assert call['restored_events'] == 0
    assert call['published_events'] == call['actual_statistics_fits'] == len(observed['fits']) == 2
    assert ledger['durable_event_count'] == 3 and ledger['orphan_event_count'] == 1
    orphan, = [row for row in ledger['events'] if not row['consumers']]
    assert orphan['event_id'] == origin['event_id'] and orphan['origin_seconds'] == origin['origin_seconds']
    assert_accounting(result)
    with observe_science() as reference:
        ordinary = run(tmp_path / 'ordinary', early=False)
    assert science(result) == science(ordinary)
    assert_observed_equal(observed, reference)


def test_array_read_timer_start_failure_aborts_before_fold_receipt(tmp_path, monkeypatch):
    event = committed(tmp_path)
    event_bytes, payload_bytes = event.read_bytes(), payload_for(event).read_bytes()
    primary = OSError('read clock unavailable')
    index = recovery.StatisticsEventStore._index

    def failed_clock():
        raise primary

    with monkeypatch.context() as patch:
        def index_then_lose_clock(store):
            events = index(store)
            patch.setattr(recovery, 'time', SimpleNamespace(perf_counter=failed_clock))
            return events

        patch.setattr(recovery.StatisticsEventStore, '_index', index_then_lose_clock)
        with observe_science() as attempt, pytest.raises(recovery.StatisticsRecoveryError) as caught:
            run(tmp_path)
    assert caught.value.__cause__ is primary and not any(attempt.values())
    assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)


@pytest.mark.parametrize(('allocation', 'preflight'), [
    ('metadata_read_seconds', False), ('metadata_read_seconds', True),
    ('array_read_seconds', False), ('payload_path', False),
])
def test_read_setup_allocation_failure_aborts_before_fold_receipt(tmp_path, monkeypatch, allocation, preflight):
    event = committed(tmp_path)
    payload = payload_for(event)
    event_bytes, payload_bytes = event.read_bytes(), payload.read_bytes()
    if preflight:
        # The receipt preflight also builds the metadata index outside load().
        (tmp_path / 'folds').mkdir(exist_ok=True)
    primary = MemoryError(f'cannot allocate {allocation}')
    store = recovery.StatisticsEventStore
    timed_read = store._timed_read
    join = Path.__truediv__

    with monkeypatch.context() as patch:
        def allocate_timer(self, field):
            if field == allocation:
                patch.setattr(store, '_timed_read', timed_read)
                raise primary
            return timed_read(self, field)

        def make_path(parent, name):
            if parent == event.parent and name == payload.name:
                patch.setattr(Path, '__truediv__', join)
                raise primary
            return join(parent, name)

        if allocation == 'payload_path':
            patch.setattr(Path, '__truediv__', make_path)
        else:
            patch.setattr(store, '_timed_read', allocate_timer)
        with observe_science() as attempt, pytest.raises(recovery.StatisticsRecoveryError) as caught:
            run(tmp_path)
    assert caught.value.__cause__ is primary and not any(attempt.values())
    assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)


@pytest.mark.parametrize('completed', [False, True], ids=['missing-fold', 'completed-study'])
def test_committed_metadata_status_denial_aborts_recovery(tmp_path, monkeypatch, completed):
    if completed:
        original = run(tmp_path)
        event = event_paths(tmp_path)[0]
        before = stable_bytes(tmp_path)
    else:
        event = committed(tmp_path)
        event_bytes, payload_bytes = event.read_bytes(), payload_for(event).read_bytes()
    primary = PermissionError('cannot inspect committed event file status')
    lstat = Path.lstat
    with monkeypatch.context() as patch:
        def deny_once(path):
            if path == event:
                patch.setattr(Path, 'lstat', lstat)
                raise primary
            return lstat(path)

        patch.setattr(Path, 'lstat', deny_once)
        with observe_science() as attempt, pytest.raises(recovery.StatisticsRecoveryError) as caught:
            run(tmp_path)
    assert caught.value.__cause__ is primary and not any(attempt.values())
    if completed:
        assert stable_bytes(tmp_path) == before
        with observe_science() as observed:
            resumed = run(tmp_path)
        assert not any(observed.values()) and science(resumed) == science(original)
        assert resumed['screening_statistics_recovery']['events'] == original['screening_statistics_recovery']['events']
        assert_accounting(resumed)
    else:
        assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)


def fault_metadata_stream(patch, event, read_error, close_error=None):
    open_file = Path.open
    steps = []

    def opened(path, mode='r', *args, **kwargs):
        stream = open_file(path, mode, *args, **kwargs)
        if path == event and mode == 'rb':
            read, close = stream.read, stream.close

            def read_metadata(*args, **kwargs):
                steps.append('read')
                if read_error is not None:
                    raise read_error
                return read(*args, **kwargs)

            def close_metadata():
                if not stream.closed:
                    close()
                    steps.append('close')
                    if close_error is not None:
                        raise close_error

            patch.setattr(stream, 'read', read_metadata)
            patch.setattr(stream, 'close', close_metadata)
        return stream

    patch.setattr(Path, 'open', opened)
    return steps


@pytest.mark.parametrize(('completed', 'error_type', 'close_type', 'keep_read'), [
    (False, KeyboardInterrupt, OSError, True), (True, SystemExit, OSError, True),
    (True, MemoryError, OSError, True), (False, None, OSError, False),
    (True, PermissionError, OSError, True), (True, MemoryError, KeyboardInterrupt, False),
    (True, PermissionError, MemoryError, False),
])
def test_metadata_file_close_preserves_read_abort(tmp_path, monkeypatch, completed, error_type,
                                                 close_type, keep_read):
    if completed:
        original = run(tmp_path)
        event = event_paths(tmp_path)[0]
    else:
        event = committed(tmp_path)
        event_bytes, payload_bytes = event.read_bytes(), payload_for(event).read_bytes()
    before = stable_bytes(tmp_path)
    read_error = error_type('authoritative JSON read interrupted') if error_type else None
    close_error = close_type('closing the authoritative JSON file failed')
    primary = read_error if keep_read else close_error
    expected = recovery.StatisticsRecoveryError if isinstance(primary, Exception) else type(primary)
    with monkeypatch.context() as patch:
        steps = fault_metadata_stream(patch, event, read_error, close_error)
        with observe_science() as attempt, pytest.raises(expected) as caught:
            run(tmp_path)
    assert (caught.value.__cause__ if expected is recovery.StatisticsRecoveryError else caught.value) is primary
    assert steps == ['read', 'close'] and not any(attempt.values())
    after = stable_bytes(tmp_path)
    assert before.items() <= after.items()
    if completed:
        assert before == after
        with observe_science() as observed:
            resumed = run(tmp_path)
        assert not any(observed.values()) and science(resumed) == science(original)
        assert resumed['screening_statistics_recovery']['events'] == original['screening_statistics_recovery']['events']
        assert_accounting(resumed)
    else:
        attempt_name, = after.keys() - before.keys()
        assert Path(attempt_name).parent == Path('attempts')
        attempt_record = study._read_receipt(tmp_path / attempt_name)
        assert attempt_record['attempt_error_type'] == expected.__name__
        assert attempt_record['state'] == ('interrupted' if expected is KeyboardInterrupt else 'failed')
        resumed, observed = assert_read_failure_recovers(tmp_path, event, event_bytes, payload_bytes)
        with observe_science() as reference:
            ordinary = run(tmp_path / 'ordinary', early=False)
        assert science(resumed) == science(ordinary)
        assert_observed_equal(observed, reference, compare_fits=False)


@pytest.mark.parametrize('read_error', [None, PermissionError, KeyboardInterrupt])
def test_metadata_read_timer_preserves_completed_study(tmp_path, monkeypatch, read_error):
    original = run(tmp_path)
    event = event_paths(tmp_path)[0]
    before = stable_bytes(tmp_path)
    timer = MemoryError('cannot record metadata read time')
    primary = read_error('metadata unavailable') if read_error else timer

    with monkeypatch.context() as patch:
        fail_read_bookkeeping(patch, metadata_read_seconds=timer)
        if read_error:
            fault_metadata_stream(patch, event, primary)
        expected = recovery.StatisticsRecoveryError if isinstance(primary, Exception) else read_error
        with observe_science() as attempt, pytest.raises(expected) as caught:
            run(tmp_path)
    assert (caught.value.__cause__ if isinstance(primary, Exception) else caught.value) is primary
    assert not any(attempt.values()) and stable_bytes(tmp_path) == before
    with observe_science() as completed:
        resumed = run(tmp_path)
    assert not any(completed.values()) and science(original) == science(resumed)
    assert original['screening_statistics_recovery']['events'] == resumed['screening_statistics_recovery']['events']
    assert_accounting(resumed)


def fault_staging_stream(patch, directory, kind, primary, cleanup, secondary):
    """Fail writes on the first real staging file and let ZIP/file cleanup run."""
    open_file = Path.open
    steps = []

    def opened(path, mode='r', *args, **kwargs):
        stream = open_file(path, mode, *args, **kwargs)
        if (path.parent != directory / 'statistics-events' or mode != 'xb'
                or not path.name.startswith(f'.{kind}-') or steps):
            return stream
        write, close = stream.write, stream.close

        def write_staged(data):
            # ZIP_STORED places the .npy bytes directly in the member.
            if primary is not None and not steps and (kind == 'event' or data[:6] == b'\x93NUMPY'):
                steps.append('write')
                raise primary
            if ((cleanup == 'member header' and steps == ['write'] and data[:4] == b'PK\x03\x04')
                    or (cleanup == 'ZIP directory' and data[:4] == b'PK\x01\x02')):
                steps.append(cleanup)
                raise secondary
            return write(data)

        def close_staged():
            if not stream.closed:
                close()
                if cleanup == 'file close':
                    steps.append(cleanup)
                    raise secondary

        patch.setattr(stream, 'write', write_staged)
        patch.setattr(stream, 'close', close_staged)
        return stream

    patch.setattr(Path, 'open', opened)
    return steps


@pytest.mark.parametrize(('kind', 'error_type', 'cleanup'), [
    ('payload', KeyboardInterrupt, 'member header'),
    ('payload', SystemExit, 'ZIP directory'),
    ('payload', KeyboardInterrupt, 'file close'),
    ('event', SystemExit, 'file close'),
])
def test_staging_write_interruption_survives_close(tmp_path, monkeypatch, kind, error_type, cleanup):
    primary = error_type('interrupted staged write')
    secondary = OSError('failed ' + cleanup)
    with monkeypatch.context() as patch:
        steps = fault_staging_stream(patch, tmp_path, kind, primary, cleanup, secondary)
        with pytest.raises(error_type) as caught:
            run(tmp_path)
    assert caught.value is primary and steps == ['write', cleanup]
    assert not receipts(tmp_path) and not event_paths(tmp_path)
    assert not list((tmp_path / 'statistics-events').glob('payload-*.npz'))
    with observe_science() as observed:
        resumed = run(tmp_path)
    ledger = resumed['screening_statistics_recovery']
    call = ledger['current_call']
    assert call['restored_events'] == 0 and call['publication_failures'] == []
    assert ledger['durable_event_count'] == call['published_events'] == call['actual_statistics_fits'] == 2
    assert_accounting(resumed)
    with observe_science() as reference:
        ordinary = run(tmp_path / 'ordinary', early=False)
    assert science(resumed) == science(ordinary)
    assert_observed_equal(observed, reference)
    before = stable_bytes(tmp_path)
    with observe_science() as completed:
        reused = run(tmp_path)
    after = stable_bytes(tmp_path)
    assert not any(completed.values()) and after.keys() == before.keys()
    assert all(after[name] == digest for name, digest in before.items() if name != 'report.json')
    assert science(reused) == science(resumed) and reused['screening_statistics_recovery']['events'] == ledger['events']
    assert_accounting(reused)


@pytest.mark.parametrize('error_type', [MemoryError, RuntimeError])
def test_member_write_error_keeps_failed_fold_when_directory_close_fails(tmp_path, monkeypatch, error_type):
    primary = error_type('cannot write staged array')
    secondary = OSError('cannot finish ZIP directory')
    reports = []
    for name, cleanup in (('write-only', None), ('write-and-close', 'ZIP directory')):
        directory = tmp_path / name
        with monkeypatch.context() as patch:
            steps = fault_staging_stream(patch, directory, 'payload', primary, cleanup, secondary)
            report = run(directory, grid={'knn__n_neighbors': [1]})
        assert steps == (['write', cleanup] if cleanup else ['write'])
        failed, valid = sorted(receipts(directory), key=lambda row: row['fold'])
        assert failed['status'] == 'failed' and failed['failure_stage'] == 'screening'
        assert failed['error_type'] == error_type.__name__ and failed['error'] == str(primary)
        assert failed['screening_statistics_fit_performed'] and failed.get('screening_statistics_event_id') is None
        assert valid['status'] == 'valid' and valid['screening_statistics_event_id'] is not None
        ledger = report['screening_statistics_recovery']
        call = ledger['current_call']
        assert call['publication_failures'] == [] and call['published_events'] == ledger['durable_event_count'] == 1
        assert call['actual_statistics_fits'] == report['screening_statistics_fits'] == 2
        assert_accounting(report)
        reports.append(science(report))
    assert reports[0] == reports[1]


def test_publication_close_interrupts_an_allocation_failure(tmp_path, monkeypatch):
    primary = MemoryError('cannot write staged array')
    cancel = KeyboardInterrupt('cancel during close')
    with monkeypatch.context() as patch:
        steps = fault_staging_stream(patch, tmp_path, 'payload', primary, 'member header', cancel)
        with pytest.raises(KeyboardInterrupt) as caught:
            run(tmp_path)
    assert caught.value is cancel and steps == ['write', 'member header']
    assert not receipts(tmp_path) and not event_paths(tmp_path)


@pytest.mark.parametrize('kind', ['event', 'payload'])
def test_buffered_staging_retry_interrupt_survives_raw_close(tmp_path, monkeypatch, kind):
    open_file = Path.open
    cancel = KeyboardInterrupt('cancel while retrying the buffered write')
    steps = []
    with monkeypatch.context() as patch:
        def opened(path, mode='r', *args, **kwargs):
            stream = open_file(path, mode, *args, **kwargs)
            if mode != 'xb' or not path.name.startswith(f'.{kind}-') or steps:
                return stream
            assert isinstance(stream, io.BufferedWriter)
            raw = stream.raw
            write, close = raw.write, raw.close

            def raw_write(data):
                # A ZIP without a comment ends with the 22-byte end-of-directory record.
                if not steps and (kind == 'event' or data[-22:-18] == b'PK\x05\x06'):
                    steps.append('flush')
                    raise OSError('cannot flush staged file')
                if steps == ['flush']:
                    steps.append('retry')
                    raise cancel
                return write(data)

            def raw_close():
                if not raw.closed:
                    close()
                    steps.append('raw close')
                    raise OSError('cannot close staged file')

            patch.setattr(raw, 'write', raw_write)
            patch.setattr(raw, 'close', raw_close)
            return stream

        patch.setattr(Path, 'open', opened)
        with pytest.raises(KeyboardInterrupt) as caught:
            run(tmp_path)
    assert caught.value is cancel and steps == ['flush', 'retry', 'raw close']
    assert not receipts(tmp_path) and not event_paths(tmp_path)
    assert not list((tmp_path / 'statistics-events').glob('payload-*.npz'))
    attempt, = (tmp_path / 'attempts').glob('*.json')
    record = study._read_receipt(attempt)
    assert record['state'] == 'interrupted' and record['attempt_error_type'] == 'KeyboardInterrupt'
    with observe_science() as observed:
        resumed = run(tmp_path)
    current = resumed['screening_statistics_recovery']['current_call']
    assert current['restored_events'] == 0 and current['publication_failures'] == []
    assert current['actual_statistics_fits'] == current['published_events'] == 2
    assert_accounting(resumed)
    with observe_science() as reference:
        ordinary = run(tmp_path / 'ordinary', early=False)
    assert science(resumed) == science(ordinary)
    assert_observed_equal(observed, reference)


def test_staged_json_flush_interruption_survives_raw_close(tmp_path, monkeypatch):
    open_file = Path.open
    cancel = KeyboardInterrupt('cancel during buffered flush')
    steps = []
    with monkeypatch.context() as patch:
        def opened(path, mode='r', *args, **kwargs):
            stream = open_file(path, mode, *args, **kwargs)
            if mode == 'xb' and path.name.startswith('.event-') and not steps:
                write, close = stream.raw.write, stream.raw.close

                def raw_write(data):
                    if not steps:
                        steps.append('flush')
                        raise cancel
                    return write(data)

                def raw_close():
                    if not stream.raw.closed:
                        close()
                        steps.append('close')
                        raise OSError('raw file close failed')

                patch.setattr(stream.raw, 'write', raw_write)
                patch.setattr(stream.raw, 'close', raw_close)
            return stream

        patch.setattr(Path, 'open', opened)
        with pytest.raises(KeyboardInterrupt) as caught:
            run(tmp_path)
    assert caught.value is cancel and steps == ['flush', 'close']
    assert not receipts(tmp_path) and not event_paths(tmp_path)


@pytest.mark.parametrize(('kind', 'cleanup'), [('payload', 'ZIP directory'), ('event', 'file close')])
def test_staging_close_oserror_keeps_valid_receipt(tmp_path, monkeypatch, kind, cleanup):
    secondary = OSError('failed ' + cleanup)
    with monkeypatch.context() as patch:
        steps = fault_staging_stream(patch, tmp_path, kind, None, cleanup, secondary)
        with observe_science() as observed:
            report = run(tmp_path)
    assert steps == [cleanup]
    ledger = report['screening_statistics_recovery']
    call = ledger['current_call']
    failure, = call['publication_failures']
    assert failure['error_type'] == 'OSError' and failure['error'] == str(secondary)
    rows = receipts(tmp_path)
    assert all(row['status'] == 'valid' for row in rows)
    origin, = [row for row in rows if row['fold'] == failure['fold'] and row['screening_statistics_fit_performed']]
    assert origin.get('screening_statistics_event_id') is None
    assert ledger['receipt_event_backed_fits'] == ledger['durable_event_count'] == call['published_events'] == 1
    assert call['actual_statistics_fits'] == len(observed['fits']) == 2
    assert_accounting(report)
    with observe_science() as reference:
        ordinary = run(tmp_path / 'ordinary', early=False)
    assert science(report) == science(ordinary)
    assert_observed_equal(observed, reference)


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


def assert_interrupted_publication_recovers(directory, event, *, moved=True):
    assert event is not None
    assert event_paths(directory) == ([event] if moved else [])
    assert receipts(directory) == []
    payload = payload_for(event)
    saved_payload = payload.read_bytes()
    saved_metadata = event.read_bytes() if moved else None

    resumed = run(directory)
    ledger = resumed['screening_statistics_recovery']
    assert ledger['durable_event_count'] == 2 and ledger['orphan_event_count'] == 0
    assert ledger['current_call']['restored_events'] == int(moved)
    assert ledger['receipt_event_backed_fits'] == resumed['screening_statistics_fits'] == (1 if moved else 2)
    assert ledger['union_statistics_seconds'] == ledger['durable_statistics_seconds']
    assert payload.read_bytes() == saved_payload
    if moved:
        assert event.read_bytes() == saved_metadata
        origin = metadata(event)
        receipt = next(row for row in receipts(directory) if row['key'] == origin['origin_record_key'])
        assert receipt['screening_statistics_event_id'] == origin['event_id']
        assert receipt['screening_statistics_fit_performed'] is False
        assert receipt['screening_statistics_seconds'] == 0.
    else:
        assert not event.exists()
    assert_accounting(resumed)
    return resumed


def fault_final_guard(patch, configure=None, *, fault_source=None):
    """Arm a one-shot fault after staging the first event and moving its payload."""
    initialize = recovery.StatisticsEventStore.__init__
    events = []

    def initialize_with_hook(store, *args, **kwargs):
        initialize(store, *args, **kwargs)

        def hook(stage, details):
            if stage == 'before_event_commit' and not events:
                events.append(store.root / f"event-{details['event_id']}.json")
                if configure is not None:
                    configure(store)
                if fault_source is not None:
                    guard = store.runtime_guard

                    def verify(*args):
                        store.runtime_guard = guard
                        with patch.context() as source_patch:
                            fault_source(source_patch)
                            return guard(*args)

                    store.runtime_guard = verify

        patch.setattr(recovery, '_event_hook', hook)

    patch.setattr(recovery.StatisticsEventStore, '__init__', initialize_with_hook)
    return events


def fail_guard_time_update(store, error):
    pending = [error]

    class Counters(dict):
        def __setitem__(self, key, value):
            if key == 'guard_seconds' and pending:
                raise pending.pop()
            super().__setitem__(key, value)

    store.current = Counters(store.current)


def assert_guard_abort_recovers(directory, events, error):
    event, = events
    attempts = [study._read_receipt(path) for path in (directory / 'attempts').glob('*.json')]
    state = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
    assert any(a['state'] == state and a.get('attempt_error_type') == type(error).__name__ for a in attempts)
    return assert_interrupted_publication_recovers(directory, event, moved=False)


@pytest.mark.parametrize('fault', ['interrupted', 'read-denied', 'source-changed'])
def test_final_guard_error_survives_timing_failure(tmp_path, monkeypatch, fault):
    primary = (KeyboardInterrupt('cancel source verification') if fault == 'interrupted'
               else PermissionError('source denied'))
    timer = MemoryError('cannot record guard time')
    source = Path(evaluation.__file__)
    if fault == 'source-changed':
        other = tmp_path / 'other'
        other.mkdir()
        (other / source.name).write_text('# changed source\n', encoding='utf8')

    def fault_source(patch):
        if fault == 'source-changed':
            # A different package directory has a different source manifest.
            patch.setattr(evaluation, '__file__', str(tmp_path / 'other' / source.name))
        else:
            fault_metadata_stream(patch, source, primary)

    with monkeypatch.context() as patch:
        events = fault_final_guard(patch, lambda store: fail_guard_time_update(store, timer),
                                   fault_source=fault_source)
        expected = KeyboardInterrupt if fault == 'interrupted' else recovery.StatisticsRecoveryError
        with pytest.raises(expected) as caught:
            run(tmp_path)
    if fault == 'source-changed':
        assert 'changed' in str(caught.value).lower() and 'source' in str(caught.value).lower()
    else:
        assert (caught.value if fault == 'interrupted' else caught.value.__cause__) is primary
    assert_guard_abort_recovers(tmp_path, events, caught.value)


@pytest.mark.parametrize('fail_eligibility', [True, False])
def test_final_policy_failure_aborts_before_receipt(tmp_path, monkeypatch, fail_eligibility):
    error = OSError('cannot read policy') if fail_eligibility else MemoryError('cannot record bypass')

    def configure(store):
        def eligibility():
            if fail_eligibility:
                raise error
            return 'changed'

        class Bypasses(list):
            def append(self, value):
                raise error

        store.eligibility_reason = eligibility
        store.current['bypasses'] = Bypasses(store.current['bypasses'])

    with monkeypatch.context() as patch:
        events = fault_final_guard(patch, configure)
        with pytest.raises(recovery.StatisticsRecoveryError) as caught:
            run(tmp_path)
    assert caught.value.__cause__ is error
    assert_guard_abort_recovers(tmp_path, events, caught.value)


@pytest.mark.parametrize(('phase', 'error_type'), [
    ('start', OSError), ('finish', MemoryError), ('record', OSError), ('record', SystemExit),
])
def test_final_guard_timing_failure_aborts_before_receipt(tmp_path, monkeypatch, phase, error_type):
    primary = error_type(f'guard timing failed at {phase}')
    with monkeypatch.context() as patch:
        def configure(store):
            if phase == 'record':
                fail_guard_time_update(store, primary)
                return
            original_time, guard = recovery.time, store.runtime_guard
            armed = phase == 'start'

            def verify(*args):
                nonlocal armed
                result = guard(*args)
                armed = True
                return result

            def clock():
                if armed:
                    patch.setattr(recovery, 'time', original_time)
                    raise primary
                return original_time.perf_counter()

            store.runtime_guard = verify
            patch.setattr(recovery, 'time', SimpleNamespace(perf_counter=clock, sleep=original_time.sleep))

        events = fault_final_guard(patch, configure)
        expected = recovery.StatisticsRecoveryError if isinstance(primary, Exception) else error_type
        with pytest.raises(expected) as caught:
            run(tmp_path)
    assert (caught.value.__cause__ if isinstance(primary, Exception) else caught.value) is primary
    with observe_science() as observed:
        resumed = assert_guard_abort_recovers(tmp_path, events, caught.value)
    if phase == 'start':
        with observe_science() as reference:
            ordinary = run(tmp_path / 'ordinary', early=False)
        assert science(resumed) == science(ordinary)
        assert_observed_equal(observed, reference)
        with observe_science() as completed:
            reused = run(tmp_path)
        assert not any(completed.values()) and science(reused) == science(resumed)


@pytest.mark.parametrize(('component', 'read_type', 'close_type', 'keep_read'), [
    ('package', KeyboardInterrupt, OSError, True), ('package', MemoryError, OSError, True),
    ('package', SystemExit, KeyboardInterrupt, True),
    ('package', PermissionError, SystemExit, False), ('package', MemoryError, KeyboardInterrupt, False),
    ('package', PermissionError, OSError, False), ('estimator', SystemExit, OSError, True),
])
def test_final_guard_source_read_and_close_abort_before_receipt(tmp_path, monkeypatch, component,
                                                              read_type, close_type, keep_read):
    source = (Path(evaluation.__file__) if component == 'package'
              else Path(inspect.getfile(type(estimator().named_steps['knn']))).resolve())
    read_error, close_error = read_type('source read failed'), close_type('source close failed')
    primary = read_error if keep_read else close_error
    expected = recovery.StatisticsRecoveryError if isinstance(primary, Exception) else type(primary)
    steps = []

    def fault_source(patch):
        steps.append(fault_metadata_stream(patch, source, read_error, close_error))

    with monkeypatch.context() as patch:
        events = fault_final_guard(patch, fault_source=fault_source)
        # A displaced interrupt should fail this case without stopping pytest.
        with pytest.raises((recovery.StatisticsRecoveryError, KeyboardInterrupt, SystemExit)) as caught:
            run(tmp_path)
    assert type(caught.value) is expected
    assert steps == [['read', 'close']]
    assert (caught.value.__cause__ if isinstance(primary, Exception) else caught.value) is primary
    if read_type is PermissionError and close_type is OSError:
        assert close_error.__context__ is read_error
    assert_guard_abort_recovers(tmp_path, events, caught.value)


@contextmanager
def expect_publication_abort(primary):
    with pytest.raises((recovery.StatisticsRecoveryError, MemoryError)) as caught:
        yield
    error = caught.value
    while isinstance(error, recovery.StatisticsRecoveryError):
        error = error.__cause__
    assert error is primary


@pytest.mark.parametrize('fault', ['serialization', 'hook'])
def test_precommit_publication_error_remains_a_failed_screening_fold(tmp_path, monkeypatch, fault):
    primary = MemoryError('cannot serialize the array') if fault == 'serialization' else RuntimeError('before commit')
    save = np.save
    injected = False
    boundaries = []

    def save_array(*args, **kwargs):
        nonlocal injected
        if fault == 'serialization' and not injected:
            injected = True
            raise primary
        return save(*args, **kwargs)

    def hook(stage, details):
        nonlocal injected
        if stage in ('before_event_commit', 'after_event_commit'):
            boundaries.append((stage, details['fold']))
        if fault == 'hook' and stage == 'before_event_commit' and not injected:
            injected = True
            raise primary

    monkeypatch.setattr(np, 'save', save_array)
    monkeypatch.setattr(recovery, '_event_hook', hook)
    grid = {'knn__n_neighbors': [1]}
    report = run(tmp_path, grid=grid)
    failed, valid = sorted(receipts(tmp_path), key=lambda row: row['fold'])
    assert injected and failed['state'] == 'completed' and failed['status'] == 'failed'
    assert failed['failure_stage'] == 'screening'
    assert failed['error_type'] == type(primary).__name__ and failed['error'] == str(primary)
    assert failed['screening_statistics_fit_performed'] and failed.get('screening_statistics_event_id') is None
    event, = event_paths(tmp_path)
    assert valid['status'] == 'valid' and valid['screening_statistics_event_id'] == metadata(event)['event_id']
    expected = [('before_event_commit', failed['fold'])] if fault == 'hook' else []
    expected += [('before_event_commit', valid['fold']), ('after_event_commit', valid['fold'])]
    assert boundaries == expected
    assert report['screening_statistics_recovery']['current_call']['publication_failures'] == []
    assert_accounting(report)

    saved = {p: p.read_bytes() for p in (tmp_path / 'folds').glob('*.json')}
    with observe_science() as observed:
        resumed = run(tmp_path, grid=grid)
    assert not any(observed.values()) and boundaries == expected
    assert science(resumed) == science(report)
    assert all(p.read_bytes() == raw for p, raw in saved.items())
    assert_accounting(resumed)


@pytest.mark.parametrize(('moved', 'rename_error', 'lookup_error'), [
    (True, PermissionError, None),
    (True, PermissionError, PermissionError),
    (False, PermissionError, PermissionError),
    (True, MemoryError, None),
    (True, OSError, MemoryError),
    (False, OSError, MemoryError),
])
def test_uncertain_json_commit_aborts_before_fold_receipt(tmp_path, monkeypatch, moved, rename_error, lookup_error):
    replace, lstat = Path.replace, Path.lstat
    primary = rename_error('lost JSON rename acknowledgement')
    event = None

    def fail_commit(source, target):
        nonlocal event
        if source.name.startswith('.event-'):
            if moved:
                replace(source, target)
            event = target
            raise primary
        return replace(source, target)

    def inspect(path):
        if lookup_error and path == event:
            raise lookup_error('cannot inspect the JSON destination')
        return lstat(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', fail_commit)
        patch.setattr(Path, 'lstat', inspect)
        with expect_publication_abort(primary):
            run(tmp_path)
    assert_interrupted_publication_recovers(tmp_path, event, moved=moved)


def test_committed_event_index_insertion_failure_aborts_before_receipt(tmp_path, monkeypatch):
    primary = MemoryError('cannot insert the committed event into the index')
    index = recovery.StatisticsEventStore._index

    class FullIndex(dict):
        def __setitem__(self, key, value):
            raise primary

    def full_index(store):
        events = index(store)
        if not isinstance(events, FullIndex):
            store.events = FullIndex(events)
        return store.events

    with monkeypatch.context() as patch:
        patch.setattr(recovery.StatisticsEventStore, '_index', full_index)
        with expect_publication_abort(primary):
            run(tmp_path)
    event, = event_paths(tmp_path)
    assert_interrupted_publication_recovers(tmp_path, event)


def test_write_time_update_failure_after_commit_aborts_before_receipt(tmp_path, monkeypatch):
    primary = MemoryError('cannot update statistics write time')
    initialize = recovery.StatisticsEventStore.__init__
    committed = False

    class WriteTimer(dict):
        def __setitem__(self, key, value):
            if committed and key == 'write_seconds':
                raise primary
            super().__setitem__(key, value)

    def initialize_with_timer(store, *args, **kwargs):
        initialize(store, *args, **kwargs)
        store.current = WriteTimer(store.current)

    def hook(stage, details):
        nonlocal committed
        if stage == 'after_event_commit':
            committed = True

    with monkeypatch.context() as patch:
        patch.setattr(recovery.StatisticsEventStore, '__init__', initialize_with_timer)
        patch.setattr(recovery, '_event_hook', hook)
        with expect_publication_abort(primary):
            run(tmp_path)
    event, = event_paths(tmp_path)
    assert_interrupted_publication_recovers(tmp_path, event)


@pytest.mark.parametrize('field', ['cache', 'screening_statistics_event_id', 'screening_statistics_event_reused'])
def test_committed_event_link_failure_aborts_before_first_receipt(tmp_path, monkeypatch, field):
    primary = MemoryError('cannot link the committed event to its first receipt')
    prepare = study._PersistentScreeningCache.prepare

    class FailingLink(UserDict):
        def __setitem__(self, key, value):
            if field == 'cache' or key == field:
                raise primary
            super().__setitem__(key, value)

    def prepare_with_fault(cache, template, fold, X, y, train, record, *, caught):
        link = FailingLink()
        if field == 'cache':
            link.data = cache.events
            cache.events = link
        else:
            link.data = record
            record = link
        return prepare(cache, template, fold, X, y, train, record, caught=caught)

    with monkeypatch.context() as patch:
        patch.setattr(study._PersistentScreeningCache, 'prepare', prepare_with_fault)
        with expect_publication_abort(primary):
            run(tmp_path)
    event, = event_paths(tmp_path)
    assert_interrupted_publication_recovers(tmp_path, event)


@pytest.mark.parametrize('lookup_denied', [False, True])
def test_payload_move_followed_by_permission_error_keeps_receipt_only_charge(tmp_path, monkeypatch, lookup_denied):
    replace, lstat = Path.replace, Path.lstat
    primary = PermissionError('lost payload rename acknowledgement')
    staged = payload = None

    def move_then_fail(source, target):
        nonlocal staged, payload
        value = replace(source, target)
        if payload is None and source.name.startswith('.payload-'):
            staged, payload = source, target
            raise primary
        return value

    def inspect(path):
        if lookup_denied and path == staged:
            raise PermissionError('cannot inspect the payload staging file')
        return lstat(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', move_then_fail)
        patch.setattr(Path, 'lstat', inspect)
        report = run(tmp_path)
    ledger = report['screening_statistics_recovery']
    failure, = ledger['current_call']['publication_failures']
    assert failure['error_type'] == 'PermissionError' and failure['error'] == str(primary)
    assert payload.is_file()
    event_id = payload.stem.removeprefix('payload-')
    assert not payload.with_name(f'event-{event_id}.json').exists()
    row, = [row for row in receipts(tmp_path)
            if row['fold'] == failure['fold'] and row['screening_statistics_fit_performed']]
    assert row.get('screening_statistics_event_id') is None
    assert ledger['durable_event_count'] == 1 and report['screening_statistics_fits'] == 2
    assert_accounting(report)


def test_committed_identifier_protects_a_missing_historical_payload(tmp_path, monkeypatch):
    event = committed(tmp_path)
    before = event.read_bytes()
    original_id = uuid.UUID(hex=metadata(event)['event_id'])
    payload = payload_for(event)
    payload.unlink()
    with monkeypatch.context() as patch:
        patch.setattr(recovery, 'uuid', SimpleNamespace(uuid4=lambda: original_id))
        with pytest.raises(recovery.StatisticsRecoveryError, match='collision'):
            run(tmp_path)
    assert event.read_bytes() == before and not payload.exists()
    assert receipts(tmp_path) == []

    report = run(tmp_path)
    ledger = report['screening_statistics_recovery']
    historical = next(row for row in ledger['events'] if row['event_id'] == original_id.hex)
    assert historical['consumers'] == []
    assert event.read_bytes() == before and not payload.exists()
    assert ledger['durable_event_count'] == 3 and report['screening_statistics_fits'] == 2
    assert_accounting(report)


def deny_first_rename(monkeypatch, stage):
    replace = Path.replace
    attempts = []

    def deny_once(source, target):
        if source.name.startswith(f'.{stage}-') and (not attempts or target == attempts[0]):
            attempts.append(target)
            if len(attempts) == 1:
                raise PermissionError('temporary sharing denial')
        return replace(source, target)

    monkeypatch.setattr(Path, 'replace', deny_once)
    return attempts


@pytest.mark.parametrize('stage', ['payload', 'event'])
def test_transient_publication_denial_retries_on_windows(tmp_path, monkeypatch, stage):
    attempts = deny_first_rename(monkeypatch, stage)
    report = run(tmp_path)
    ledger = report['screening_statistics_recovery']
    if os.name == 'nt':
        assert len(attempts) == 2
        assert ledger['durable_event_count'] == 2
        assert ledger['current_call']['publication_failures'] == []
    else:
        assert len(attempts) == 1
        assert ledger['durable_event_count'] == 1
        failure, = ledger['current_call']['publication_failures']
        assert failure['error'] == 'temporary sharing denial'
    assert report['screening_statistics_fits'] == 2
    assert_accounting(report)


@pytest.mark.skipif(os.name != 'nt', reason='only Windows retries publication')
def test_policy_change_during_retry_declines_event(tmp_path, monkeypatch):
    attempts = deny_first_rename(monkeypatch, 'event')
    monkeypatch.setattr(recovery.time, 'sleep', lambda seconds: np.seterr(under='raise'))
    report = run(tmp_path)
    assert len(attempts) == 1
    assert not event_paths(tmp_path)
    current = report['screening_statistics_recovery']['current_call']
    assert any(row['reason'] == 'policy_changed_before_commit' for row in current['bypasses'])
    assert current['publication_failures'] == []
    assert report['screening_statistics_fits'] == 2
    assert_accounting(report)


@pytest.mark.skipif(os.name != 'nt', reason='only Windows retries publication')
def test_source_change_during_retry_aborts_before_receipt(tmp_path, monkeypatch):
    source = tmp_path / 'evaluation.py'
    original = Path(evaluation.__file__).read_bytes()
    source.write_bytes(original)
    monkeypatch.setattr(evaluation, '__file__', str(source))
    attempts = deny_first_rename(monkeypatch, 'event')
    monkeypatch.setattr(recovery.time, 'sleep', lambda seconds: source.write_bytes(original + b'\n# changed\n'))
    directory = tmp_path / 'study'
    with pytest.raises(recovery.StatisticsRecoveryError, match='Source|source'):
        run(directory)
    assert len(attempts) == 1
    assert not event_paths(directory) and receipts(directory) == []


@pytest.mark.skipif(os.name != 'nt', reason='only Windows retries publication')
@pytest.mark.parametrize('stage', ['payload', 'event'])
def test_event_id_claimed_during_retry_is_preserved(tmp_path, monkeypatch, stage):
    attempts = deny_first_rename(monkeypatch, stage)

    def claim_id(seconds):
        target, = attempts
        identifier = target.stem.removeprefix(stage + '-')
        target.with_name(f'event-{identifier}.json').write_bytes(b'pre-existing authority')

    monkeypatch.setattr(recovery.time, 'sleep', claim_id)
    with pytest.raises(recovery.StatisticsRecoveryError, match='collision'):
        run(tmp_path)
    event, = event_paths(tmp_path)
    assert event.read_bytes() == b'pre-existing authority'
    assert payload_for(event).exists() is (stage == 'event')
    assert len(attempts) == 1 and receipts(tmp_path) == []


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



@pytest.mark.parametrize('failure', [None, PermissionError, MemoryError, KeyboardInterrupt])
def test_source_reader_observes_opener_and_suppression(monkeypatch, failure):
    import hashlib

    source = Path(evaluation.__file__)
    content = b'current source from the entered stream'
    error = failure('suppressed by opener') if failure is not None else None
    calls = []
    original_open = Path.open

    @contextmanager
    def managed():
        calls.append('enter')

        def read():
            calls.append('read')
            if error is not None:
                raise error
            return content

        try:
            yield SimpleNamespace(read=read)
        except BaseException as caught:
            assert caught is error
            calls.append('suppressed')
        finally:
            calls.append('exit')

    def opened(path, *args, **kwargs):
        if path == source:
            calls.append((args, kwargs))
            return managed()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    expected = content if error is None else None
    assert source.read_bytes() is expected
    ordinary_calls = list(calls)
    assert ordinary_calls[1:] == (['enter', 'read', 'exit'] if error is None else
                                  ['enter', 'read', 'suppressed', 'exit'])
    calls.clear()
    if error is None:
        assert evaluation.provenance()['source_sha256'][source.name] == hashlib.sha256(content).hexdigest()
    else:
        # The opener suppressed a read, leaving no bytes to hash.
        with pytest.raises(TypeError):
            evaluation.provenance()
    assert calls == ordinary_calls


def test_source_reader_preserves_rebound_method(tmp_path, monkeypatch):
    import hashlib

    alternate = tmp_path / 'alternate.py'
    alternate.write_bytes(b'alternate source')
    monkeypatch.setattr(Path, 'read_bytes', alternate.read_bytes)
    hashes = evaluation.provenance()['source_sha256']
    assert hashes and set(hashes.values()) == {hashlib.sha256(b'alternate source').hexdigest()}


def test_source_reader_preserves_exception_translated_by_override(monkeypatch):
    source = Path(evaluation.__file__)
    original_read = Path.read_bytes
    primary = KeyboardInterrupt('inside override')
    translated = OSError('override handles initial error')

    def read(path):
        if path != source:
            return original_read(path)
        try:
            raise primary
        except KeyboardInterrupt:
            raise translated

    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(OSError) as caught:
        evaluation.provenance()
    assert caught.value is translated and translated.__context__ is primary



def deny_package_listing(patch, error):
    package = Path(evaluation.__file__).parent
    for name in ('scandir', 'listdir'):
        original = getattr(os, name)

        def checked(path='.', *args, _original=original, **kwargs):
            if isinstance(path, (str, Path)) and Path(path) == package:
                raise error
            return _original(path, *args, **kwargs)

        patch.setattr(os, name, checked)


def test_inaccessible_package_directory_cannot_start_checkpoint(tmp_path, monkeypatch):
    error = PermissionError('package directory listing denied')
    with monkeypatch.context() as patch:
        deny_package_listing(patch, error)
        with pytest.raises(PermissionError) as provenance_error:
            evaluation.provenance()
        assert provenance_error.value is error
        with observe_science() as work, pytest.raises(PermissionError) as calibration_error:
            run(tmp_path)
        assert calibration_error.value is error
    assert not any(work.values()) and stable_bytes(tmp_path) == {}
    completed = run(tmp_path)
    assert all(candidate['valid'] for candidate in completed['candidates'])
    assert_accounting(completed)


def test_inaccessible_package_directory_at_final_guard_aborts_fold(tmp_path, monkeypatch):
    error = PermissionError('package directory listing denied')
    with monkeypatch.context() as patch:
        events = fault_final_guard(patch, fault_source=lambda scoped: deny_package_listing(scoped, error))
        with pytest.raises(recovery.StatisticsRecoveryError) as caught:
            run(tmp_path)
    assert caught.value.__cause__ is error
    assert_guard_abort_recovers(tmp_path, events, caught.value)


def test_empty_source_package_cannot_start_checkpoint(tmp_path, monkeypatch):
    package = tmp_path / 'package'
    package.mkdir()
    module_file = package / 'evaluation.py'
    checkpoint = tmp_path / 'checkpoint'
    monkeypatch.setattr(evaluation, '__file__', str(module_file))
    with observe_science() as work, pytest.raises(FileNotFoundError, match='package source manifest') as caught:
        run(checkpoint)
    assert Path(caught.value.filename) == module_file
    assert not any(work.values()) and stable_bytes(checkpoint) == {}

    # Another listed module cannot stand in for the missing running module.
    (package / 'other.py').write_bytes(b'')
    with pytest.raises(FileNotFoundError):
        evaluation.provenance()


def test_provenance_tracks_standalone_bytecode_and_ignores_regenerable_caches(tmp_path, monkeypatch):
    import hashlib
    from importlib.machinery import SOURCE_SUFFIXES

    module_file = tmp_path / 'evaluation.pyc'
    module_file.write_bytes(b'first implementation')
    monkeypatch.setattr(evaluation, '__file__', str(module_file))
    source = tmp_path / 'another.py'
    source.write_bytes(b'# source')
    (tmp_path / 'another.pyc').write_bytes(b'shadowed')
    upper_source = tmp_path / 'Different.py'
    upper_source.write_bytes(b'# different module stem')
    lower_bytecode = tmp_path / 'different.pyc'
    lower_bytecode.write_bytes(b'importable bytecode')
    upper_suffix = tmp_path / 'suffix.PY'
    upper_suffix.write_bytes(b'# noncanonical suffix')
    suffix_bytecode = tmp_path / 'suffix.pyc'
    suffix_bytecode.write_bytes(b'importable in a case-sensitive directory')
    gui_source = tmp_path / 'windowed.pyw'
    gui_source.write_bytes(b'# windowed module')
    gui_bytecode = tmp_path / 'windowed.pyc'
    gui_bytecode.write_bytes(b'standalone where .pyw cannot be imported')
    cache = tmp_path / '__pycache__'
    cache.mkdir()
    (cache / 'another.cpython-312.pyc').write_bytes(b'cached')

    def digests():
        gui = gui_source if '.pyw' in SOURCE_SUFFIXES else gui_bytecode
        paths = [module_file, source, upper_source, lower_bytecode, suffix_bytecode, gui]
        if os.path.normcase('.PY') == '.py':
            paths.append(upper_suffix)
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}

    assert evaluation.provenance()['source_sha256'] == digests()
    module_file.write_bytes(b'changed implementation')
    assert evaluation.provenance()['source_sha256'] == digests()
    companion = module_file.with_suffix('.py')
    companion.write_bytes(b'# source beside the running bytecode')
    assert evaluation.provenance()['source_sha256'] == {
        **digests(), companion.name: hashlib.sha256(companion.read_bytes()).hexdigest()}
    module_file.unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.provenance()


def test_provenance_checks_actual_running_file_for_alternate_case(tmp_path, monkeypatch):
    source = tmp_path / 'Evaluation.py'
    source.write_bytes(b'# running module')
    imported = tmp_path / 'evaluation.py'
    monkeypatch.setattr(evaluation, '__file__', str(imported))
    if imported.exists():
        assert set(evaluation.provenance()['source_sha256']) == {source.name}
    else:
        with pytest.raises(FileNotFoundError):
            evaluation.provenance()
