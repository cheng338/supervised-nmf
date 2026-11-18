"""Publication guards, operation lifetime and independent array ownership."""
from pathlib import Path
import weakref

import numpy as np
import pytest
from supervised_nmf import evaluation

from tests._screening_recovery_support import (
    BinaryScreening, assert_accounting, assert_observed_equal, estimator, event_paths,
    metadata, observe_science, problem, receipts, recovery, run, science,
    screening_policy, stable_bytes, stop_at, study,
)

pytestmark = pytest.mark.usefixtures('screening_policy')


@pytest.mark.parametrize('stage', ['after_payload_write', 'after_metadata_write', 'after_payload_commit', 'before_event_commit'])
def test_source_change_after_serialization_aborts_before_commit(tmp_path, monkeypatch, stage):
    # Mutate only a test-owned copy used by the actual package provenance reader.
    source = tmp_path / 'source-copy' / 'evaluation.py'
    source.parent.mkdir()
    source.write_bytes(Path(evaluation.__file__).read_bytes())
    monkeypatch.setattr(evaluation, '__file__', str(source))
    def hook(current, details):
        if current == stage:
            source.write_bytes(source.read_bytes() + b'\n# controlled source change\n')
    monkeypatch.setattr(recovery, '_event_hook', hook)
    with pytest.raises(recovery.StatisticsRecoveryError, match='Source|source'):
        run(tmp_path / 'study')
    assert not event_paths(tmp_path / 'study')
    assert receipts(tmp_path / 'study') == []
    attempts = [study._read_receipt(p) for p in (tmp_path / 'study' / 'attempts').glob('*.json')]
    assert len(attempts) == 1
    assert attempts[0]['state'] == 'failed'
    assert attempts[0]['attempt_error_type'] == 'StatisticsRecoveryError'


def test_policy_change_before_commit_declines_publication(tmp_path, monkeypatch):
    def hook(stage, details):
        if stage == 'before_event_commit':
            np.seterr(under='raise')
    monkeypatch.setattr(recovery, '_event_hook', hook)
    report = run(tmp_path)
    assert all(row['valid'] for row in report['candidates'])
    assert not event_paths(tmp_path)
    assert any(item['reason'] == 'policy_changed_before_commit'
               for item in report['screening_statistics_recovery']['current_call']['bypasses'])
    assert_accounting(report)


@pytest.mark.parametrize('change_data', [False, True])
def test_committed_events_cannot_recreate_missing_identity(tmp_path, change_data):
    with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    events = {p: p.read_bytes() for p in event_paths(tmp_path)}
    (tmp_path / 'identity.json').unlink()
    X, y = problem()
    if change_data:
        X = X.copy()
        X[0, 0] += .125
    with observe_science() as observed, pytest.raises((ValueError, recovery.StatisticsRecoveryError), match='identity|Identity'):
        run(tmp_path, data=(X, y))
    assert not observed['fits']
    assert not (tmp_path / 'identity.json').exists()
    assert all(p.read_bytes() == raw for p, raw in events.items())


def test_same_directory_writer_lock_precedes_event_or_fit_work(tmp_path, monkeypatch):
    blocked = []
    def hook(stage, details):
        if stage == 'before_payload_write' and not blocked:
            with pytest.raises(RuntimeError, match='Another process is writing') as error:
                run(tmp_path)
            blocked.append(str(error.value))
    monkeypatch.setattr(recovery, '_event_hook', hook)
    report = run(tmp_path)
    assert len(blocked) == 1
    assert len(event_paths(tmp_path)) == 2
    assert report['screening_statistics_recovery']['durable_event_count'] == 2
    assert_accounting(report)


def test_disjoint_nested_operation_owns_independent_event_store(tmp_path, monkeypatch):
    inner = []
    original_hook = recovery._event_hook
    def hook(stage, details):
        if stage == 'before_payload_write' and not inner:
            recovery._event_hook = original_hook
            try:
                inner.append(run(tmp_path / 'inner'))
            finally:
                recovery._event_hook = hook
    monkeypatch.setattr(recovery, '_event_hook', hook)
    outer = run(tmp_path / 'outer')
    assert len(inner) == 1
    assert science(outer) == science(inner[0])
    first_ids = {p.name for p in event_paths(tmp_path / 'outer')}
    second_ids = {p.name for p in event_paths(tmp_path / 'inner')}
    assert len(first_ids) == len(second_ids) == 2
    assert first_ids.isdisjoint(second_ids)
    assert_accounting(outer)
    assert_accounting(inner[0])


@pytest.mark.parametrize('interrupted', [False, True])
def test_store_cache_and_owned_statistics_release_after_operation(tmp_path, monkeypatch, interrupted):
    stores, caches, bases = [], [], []
    original_store = recovery.StatisticsEventStore.__init__
    original_cache = study._PersistentScreeningCache.__init__
    original_fit = BinaryScreening.fit
    def store_init(store, *args, **kwargs):
        original_store(store, *args, **kwargs)
        stores.append(weakref.ref(store))
    def cache_init(cache, *args, **kwargs):
        original_cache(cache, *args, **kwargs)
        caches.append(weakref.ref(cache))
    def fit(screen, X, y):
        result = original_fit(screen, X, y)
        bases.append(weakref.ref(result))
        return result
    monkeypatch.setattr(recovery.StatisticsEventStore, '__init__', store_init)
    monkeypatch.setattr(study._PersistentScreeningCache, '__init__', cache_init)
    monkeypatch.setattr(BinaryScreening, 'fit', fit)
    if interrupted:
        # Drop the exception/traceback before testing ordinary refcount lifetime.
        def invoke():
            try:
                with stop_at('after_event_commit'):
                    run(tmp_path)
            except KeyboardInterrupt:
                return
            raise AssertionError('interruption did not occur')
        invoke()
    else:
        run(tmp_path)
    assert len(stores) == len(caches) == 1 and bases
    assert all(ref() is None for ref in stores + caches + bases)


def test_loaded_arrays_remain_owned_after_archive_close(tmp_path, monkeypatch):
    with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    retained = []
    original = recovery.StatisticsEventStore.load
    def load(store, *args, **kwargs):
        result = original(store, *args, **kwargs)
        if result is not None:
            retained.append(result[0])
        return result
    monkeypatch.setattr(recovery.StatisticsEventStore, 'load', load)
    report = run(tmp_path)
    assert len(retained) == 1
    base = retained[0]
    selector = BinaryScreening(screening='percentile', percentile=50)._fit_from_statistics(base)
    before = {name: getattr(selector, name).copy() for name in recovery.ARRAYS}
    # Archive has closed; the cache and its store have already closed as well.
    for name in recovery.ARRAYS:
        first, second = getattr(base, name), getattr(selector, name)
        assert not np.shares_memory(first, second), name
        np.testing.assert_array_equal(second, before[name])
    base.scores_.fill(0.)
    np.testing.assert_array_equal(selector.scores_, before['scores_'])
    assert_accounting(report)
