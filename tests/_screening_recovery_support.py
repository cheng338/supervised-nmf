"""Small repository-owned fixtures for persistent screening statistics."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn import config_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf import study
from supervised_nmf import _screening_recovery as recovery
from supervised_nmf.screening import BinaryScreening
from supervised_nmf._neighbors import _KNNQueryReuse


@pytest.fixture
def screening_policy():
    """Each caller owns and restores its actual numerical policy."""
    prior = np.seterr(divide='warn', over='warn', under='ignore', invalid='warn')
    callback = np.geterrcall()
    np.seterrcall(None)
    try:
        with config_context(assume_finite=False), threadpool_limits(limits=1):
            yield
    finally:
        np.seterr(**prior)
        np.seterrcall(callback)


def problem():
    random = np.random.RandomState(981)
    y = np.tile([0, 1], 18)
    X = random.uniform(.1, 1., (len(y), 8))
    X[:, :3] += y[:, None] * .45
    X[:, -1] = 1.
    return X, y


def estimator():
    return Pipeline([('screen', BinaryScreening(screening='absolute', score_threshold=0.)),
                     ('knn', KNeighborsClassifier(algorithm='brute'))])


def run(path, *, data=None, model=None, grid=None, early=True, **kwargs):
    X, y = problem() if data is None else data
    return study.calibration_knn_grid(
        estimator() if model is None else model,
        {'screen__score_threshold': [0., 2.], 'knn__n_neighbors': [1, 3]} if grid is None else grid,
        X, y, cv=2, seed=29, threads=1, checkpoint_dir=path,
        screening_statistics_recovery='early' if early else 'off', **kwargs)


def event_paths(path):
    return sorted((Path(path) / 'statistics-events').glob('event-*.json'))


def receipts(path):
    return [study._read_receipt(p) for p in sorted((Path(path) / 'folds').glob('*.json'))]


def metadata(path):
    return json.loads(Path(path).read_bytes())['payload']


def stable_bytes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob('*')) if p.is_file() and p.name != '.study.lock'}


def science(report):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if not k.endswith('_seconds') and k not in ('fit_times', 'shared_representation_key')}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean({k: report[k] for k in ('candidates', 'ranked_candidate_indices',
                 'selected_parameters', 'selected_mean_score', 'scoring', 'refinement',
                 'screening_diagnostics', 'unique_representations',
                 'unique_representation_fits', 'preflight_rejected_representation_folds')})


def assert_accounting(report):
    rows = report['representation_folds']
    receipt_seconds = sum(r['screening_statistics_seconds'] for r in rows)
    assert report['compute_seconds']['screening_statistics_seconds'] == receipt_seconds
    assert report['screening_statistics_fits'] == sum(r['screening_statistics_fit_performed'] for r in rows)
    assert report['tuning_seconds'] == sum(report['compute_seconds'].values())
    value = report['screening_statistics_recovery']
    events = value['events']
    assert len({e['event_id'] for e in events}) == len(events) == value['durable_event_count']
    by_id = {e['event_id']: e for e in events}
    backed = sum(r['screening_statistics_seconds'] for r in rows
                 if r.get('screening_statistics_event_id') in by_id)
    durable = sum(e['origin_seconds'] for e in events)
    assert value['receipt_statistics_seconds'] == receipt_seconds
    assert value['receipt_event_backed_seconds'] == backed
    assert value['durable_statistics_seconds'] == durable
    assert value['union_statistics_seconds'] == receipt_seconds - backed + durable
    assert value['union_tuning_seconds'] == report['tuning_seconds'] - backed + durable
    assert value['orphan_event_count'] == sum(not e['consumers'] for e in events)
    for event in events:
        assert sorted(event['consumers']) == sorted(
            r['key'] for r in rows if r.get('screening_statistics_event_id') == event['event_id'])
    for row in rows:
        if row.get('screening_statistics_event_id') is not None:
            assert row['screening_statistics_event_id'] in by_id
            assert row['screening_statistics_event_reused'] is (not row['screening_statistics_fit_performed'])
            if not row['screening_statistics_fit_performed']:
                assert row['screening_statistics_seconds'] == 0.


@contextmanager
def stop_at(stage, *, error_type=KeyboardInterrupt):
    original = recovery._event_hook
    seen = []
    def hook(current, details):
        if current == stage:
            seen.append(dict(details))
            raise error_type('controlled statistics boundary: ' + stage)
    recovery._event_hook = hook
    try:
        yield seen
    finally:
        recovery._event_hook = original


@contextmanager
def observe_science():
    """Observe original code returns; do not replace admitted methods."""
    previous = sys.getprofile()
    assert previous is None, 'call observation needs an unprofiled process'
    codes = {BinaryScreening.fit.__code__: 'fits',
             BinaryScreening._fit_from_statistics.__code__: 'selectors',
             _KNNQueryReuse.predict.__code__: 'predictions'}
    observed = {'fits': [], 'selectors': [], 'predictions': []}
    fields = ('classes_', 'class_counts_', 'scores_', 't_scores_', 'statistic_offsets_',
              'statistic_scales_', 'scaled_pooled_variances_', 'scaled_standard_errors_',
              'support_', 'selected_features_', 'constant_features_')
    def profile(frame, event, result):
        label = codes.get(frame.f_code)
        if event != 'return' or label is None or result is None:
            return
        if label == 'predictions':
            observed[label].append(np.array(result[0], copy=True))
        else:
            observed[label].append({name: np.array(getattr(result, name), copy=True)
                                    for name in fields})
    sys.setprofile(profile)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def assert_observed_equal(left, right, *, compare_fits=True):
    for label in (('fits', 'selectors', 'predictions') if compare_fits else ('selectors', 'predictions')):
        assert len(left[label]) == len(right[label]), label
        for first, second in zip(left[label], right[label]):
            if isinstance(first, dict):
                assert first.keys() == second.keys()
                for name in first:
                    np.testing.assert_array_equal(first[name], second[name], err_msg=label + ':' + name)
            else:
                np.testing.assert_array_equal(first, second, err_msg=label)
