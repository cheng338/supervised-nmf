"""Public option, numerical policies and exact ordinary-result parity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import numpy as np
import pytest
from sklearn import config_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from tests._screening_recovery_support import (
    BinaryScreening, assert_accounting, assert_observed_equal, estimator, event_paths,
    metadata, observe_science, problem, receipts, recovery, run, science,
    screening_policy, stable_bytes, stop_at, study,
)

pytestmark = pytest.mark.usefixtures('screening_policy')


def tiny_variance_problem():
    # Scaling retains a normal-range class and a subnormal squared deviation in
    # the other class. This is an ordinary data counterexample, not injected FPE.
    block = np.array([0., 2.**-600, 2.**-599, .25, .5, 1.])[:, None]
    X = np.vstack([block, block])
    y = np.tile([0, 0, 0, 1, 1, 1], 2)
    folds = [(np.arange(6), np.arange(6, 12)), (np.arange(6, 12), np.arange(6))]
    return (X, y), folds


@pytest.mark.parametrize('policy', ['raise', 'warn', 'call'])
def test_nondefault_underflow_preserves_errors_warnings_and_callbacks(tmp_path, policy):
    data, folds = tiny_variance_problem()
    callbacks = []
    np.seterr(under=policy)
    if policy == 'call':
        np.seterrcall(lambda message, flag: callbacks.append((message, flag)))
    reference = run(tmp_path / 'ordinary', data=data, folds=folds,
                    grid={'knn__n_neighbors': [1]}, early=False)
    expected_callbacks = list(callbacks)
    callbacks.clear()
    actual = run(tmp_path / 'early', data=data, folds=folds,
                 grid={'knn__n_neighbors': [1]})
    assert science(actual) == science(reference)
    assert callbacks == expected_callbacks
    assert not event_paths(tmp_path / 'early')
    assert not actual['screening_statistics_recovery']['eligible']
    assert_accounting(actual)
    if policy == 'raise':
        assert all(row['status'] == 'failed' and row['failure_stage'] == 'screening'
                   for row in actual['representation_folds'])
    elif policy == 'warn':
        assert any(row['warnings'] for row in actual['representation_folds'])
    else:
        assert callbacks, 'fixture must actually invoke the underflow callback'


def test_default_event_does_not_hide_later_strict_failure(tmp_path):
    data, folds = tiny_variance_problem()
    options = dict(data=data, folds=folds, grid={'knn__n_neighbors': [1]})
    with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
        run(tmp_path / 'resume', **options)
    assert len(event_paths(tmp_path / 'resume')) == 1
    np.seterr(under='raise')
    reference = run(tmp_path / 'strict', early=False, **options)
    actual = run(tmp_path / 'resume', **options)
    assert science(actual) == science(reference)
    value = actual['screening_statistics_recovery']
    assert value['current_call']['restored_events'] == 0
    assert value['durable_event_count'] == value['orphan_event_count'] == 1
    assert value['receipt_event_backed_seconds'] == 0
    assert_accounting(actual)


@pytest.mark.parametrize('setting', ['callback', 'assume_finite'])
def test_other_numerical_validation_policies_bypass(tmp_path, setting):
    callback_events = []
    if setting == 'callback':
        np.seterrcall(lambda *args: callback_events.append(args))
    with config_context(assume_finite=setting == 'assume_finite'):
        reference = run(tmp_path / 'off', early=False)
        actual = run(tmp_path / 'early')
    assert science(actual) == science(reference)
    assert not event_paths(tmp_path / 'early')
    assert actual['screening_statistics_recovery']['eligible'] is False
    assert not callback_events
    assert_accounting(actual)


def test_warning_success_is_retained_but_not_published(tmp_path, monkeypatch):
    original = BinaryScreening.fit
    def warning_fit(screen, X, y):
        result = original(screen, X, y)
        warnings.warn('controlled successful canonical statistics warning', UserWarning)
        return result
    # Explicit controlled warning injection; the candidate has no callable gate.
    monkeypatch.setattr(BinaryScreening, 'fit', warning_fit)
    reference = run(tmp_path / 'off', early=False)
    actual = run(tmp_path / 'early')
    assert science(actual) == science(reference)
    assert all(row['warnings'] for row in actual['representation_folds'][:2])
    assert not event_paths(tmp_path / 'early')
    assert actual['screening_statistics_fits'] == 2
    reasons = actual['screening_statistics_recovery']['current_call']['bypasses']
    assert sum(row['reason'] == 'statistics_emitted_warning' for row in reasons) == 2
    assert_accounting(actual)


@pytest.mark.parametrize('labels', ['unicode', 'object_string', 'object_numpy_string', 'boolean'])
def test_supported_label_storage_preserves_current_success(tmp_path, labels):
    X, y = problem()
    if labels == 'unicode':
        y = np.where(y, 'present', 'absent')
    elif labels == 'object_string':
        y = np.array(['present' if item else 'absent' for item in y], dtype=object)
    elif labels == 'object_numpy_string':
        y = np.empty(len(y), dtype=object)
        y[:] = [np.str_('present' if i % 2 else 'absent') for i in range(len(y))]
    else:
        y = y.astype(bool)
    reference = run(tmp_path / 'off', data=(X, y), early=False)
    actual = run(tmp_path / 'early', data=(X, y))
    assert all(row['valid'] for row in reference['candidates'])
    assert science(actual) == science(reference)
    assert_accounting(actual)
    if labels == 'object_numpy_string':
        # These labels remain valid for ordinary fitting. Object storage with
        # NumPy string scalars deliberately bypasses the exact-Python-str codec.
        value = actual['screening_statistics_recovery']
        assert not event_paths(tmp_path / 'early')
        assert value['durable_event_count'] == 0
        assert value['current_call']['published_events'] == 0
        assert value['current_call']['restored_events'] == 0
        reasons = value['current_call']['bypasses']
        assert len(reasons) == 2 and all(row['reason'] == 'codec_unsupported' for row in reasons)
        before = stable_bytes(tmp_path / 'early')
        with observe_science() as observed:
            completed = run(tmp_path / 'early', data=(X, y))
        assert not any(observed.values()) and science(completed) == science(reference)
        assert completed['compute_seconds'] == actual['compute_seconds']
        after = stable_bytes(tmp_path / 'early')
        assert {name: pin for name, pin in before.items() if name != 'report.json'} == {
            name: pin for name, pin in after.items() if name != 'report.json'}
        completed_value = completed['screening_statistics_recovery']
        assert completed_value['durable_event_count'] == 0
        assert completed_value['current_call']['actual_statistics_fits'] == 0
        assert completed_value['current_call']['published_events'] == 0
        assert completed_value['current_call']['restored_events'] == 0
        assert_accounting(completed)
        return
    with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
        run(tmp_path / 'restore', data=(X, y))
    restored = run(tmp_path / 'restore', data=(X, y))
    assert science(restored) == science(reference)
    assert restored['screening_statistics_recovery']['current_call']['restored_events'] == 1
    assert_accounting(restored)


class CustomScreen(BinaryScreening):
    pass


def custom_weights(distances):
    return np.ones_like(distances)


@pytest.mark.parametrize('custom', ['screen', 'weights'])
def test_custom_or_unsupported_layout_remains_ordinary(tmp_path, custom):
    model = estimator()
    if custom == 'screen':
        model.steps[0] = ('screen', CustomScreen(score_threshold=0.))
    else:
        model.set_params(knn__weights=custom_weights)
    reference = run(tmp_path / 'off', model=model, early=False)
    actual = run(tmp_path / 'early', model=model)
    assert science(actual) == science(reference)
    assert not event_paths(tmp_path / 'early')
    assert actual['screening_statistics_recovery']['eligible'] is False
    assert_accounting(actual)


def test_cached_pipeline_remains_ordinary_on_cache_miss_and_hit(tmp_path):
    def cached_run(name, cache, *, early, expected_fits):
        directory = tmp_path / name
        model = estimator().set_params(memory=cache)
        with observe_science() as observed:
            report = run(directory, model=model, early=early)
        assert len(observed['fits']) == expected_fits
        assert len(observed['predictions']) == 8  # Four candidates in two folds.
        assert all(candidate['valid'] for candidate in report['candidates'])
        assert any(Path(cache).rglob('output.pkl')), 'ordinary Joblib cache was not published'
        assert not event_paths(directory)
        if early:
            assert report['screening_statistics_recovery']['eligible'] is False
            assert_accounting(report)
        return report, observed['predictions']

    def fold_warnings(report):
        return [(row['parameters'], row['fold'], row['warnings'])
                for row in report['representation_folds']]

    # Short cache paths leave room for Joblib's internal filenames on Windows.
    with TemporaryDirectory(prefix='i9m-') as off_cache, TemporaryDirectory(prefix='i9m-') as early_cache:
        cold_off, cold_off_predictions = cached_run('cold-off', off_cache, early=False, expected_fits=4)
        cold_early, cold_early_predictions = cached_run('cold-early', early_cache, early=True, expected_fits=4)
        warm_off, warm_off_predictions = cached_run('warm-off', off_cache, early=False, expected_fits=0)
        warm_early, warm_early_predictions = cached_run('warm-early', early_cache, early=True, expected_fits=0)

    for ordinary, early in ((cold_off, cold_early), (warm_off, warm_early)):
        assert science(early) == science(ordinary)
        assert fold_warnings(early) == fold_warnings(ordinary)
    assert all(not row['warnings'] for row in cold_off['representation_folds'])

    # Cache reads can emit dependency warnings. Compare the remaining scientific
    # fields on fresh projections so the original warning records stay intact.
    cold_science, warm_science = science(cold_off), science(warm_off)
    for projection in (cold_science, warm_science):
        for candidate in projection['candidates']:
            for fold in candidate['folds']:
                del fold['warnings']
    assert warm_science == cold_science
    for predictions in (cold_early_predictions, warm_off_predictions, warm_early_predictions):
        for actual, expected in zip(predictions, cold_off_predictions, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('failure', ['invalid_screen', 'zero_selected', 'neighbor_preflight', 'rank_after_stats'])
def test_original_validation_precedence_and_post_statistics_failure(tmp_path, failure):
    from sklearn.decomposition import PCA
    model = estimator()
    grid = {'knn__n_neighbors': [1]}
    if failure == 'invalid_screen':
        grid['screen__score_threshold'] = [-1.]
    elif failure == 'zero_selected':
        grid['screen__score_threshold'] = [1.e200]
    elif failure == 'neighbor_preflight':
        grid['knn__n_neighbors'] = [1000]
    else:
        model.steps.insert(1, ('pca', PCA(n_components=1000, svd_solver='full', random_state=7)))
    reference = run(tmp_path / 'off', model=model, grid=grid, early=False)
    actual = run(tmp_path / 'early', model=model, grid=grid)
    assert science(actual) == science(reference)
    assert all(not row['valid'] for row in actual['candidates'])
    assert_accounting(actual)
    if failure in ('invalid_screen', 'neighbor_preflight'):
        assert not event_paths(tmp_path / 'early')
        assert actual['screening_statistics_fits'] == 0
    else:
        assert len(event_paths(tmp_path / 'early')) == 2
        assert actual['screening_statistics_fits'] == 2
        assert actual['screening_statistics_recovery']['durable_event_count'] == 2


def test_default_off_has_no_event_or_added_accounting(tmp_path):
    report = run(tmp_path, early=False)
    assert not event_paths(tmp_path)
    assert 'screening_statistics_recovery' not in report
    assert 'screening_statistics_recovery' not in report['identity']
    assert report['screening_statistics_fits'] == 2


@pytest.mark.parametrize('layout', ['C', 'F'])
def test_fresh_recovery_preserves_arrays_predictions_and_actual_counts(tmp_path, layout):
    from tests._screening_recovery_support import problem
    X, y = problem()
    X = np.array(X, order=layout)
    with observe_science() as ordinary:
        reference = run(tmp_path / 'off', data=(X, y), early=False)
    with observe_science() as recovered:
        actual = run(tmp_path / 'early', data=(X, y))
    assert science(actual) == science(reference)
    assert_observed_equal(ordinary, recovered)
    assert len(recovered['fits']) == 2
    assert len(event_paths(tmp_path / 'early')) == 2
    assert_accounting(actual)
    assert actual['screening_statistics_recovery']['durable_event_count'] == 2
    assert actual['screening_statistics_recovery']['union_statistics_seconds'] == actual['compute_seconds']['screening_statistics_seconds']


def test_omitted_public_option_is_off_and_does_not_construct_store(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default off constructed the persistent event store')
    monkeypatch.setattr(recovery.StatisticsEventStore, '__init__', forbidden)
    X, y = problem()
    options = dict(cv=2, seed=29, threads=1)
    grid = {'screen__score_threshold': [0., 2.], 'knn__n_neighbors': [1, 3]}
    default = study.calibration_knn_grid(estimator(), grid, X, y,
        checkpoint_dir=tmp_path/'default', **options)
    explicit = study.calibration_knn_grid(estimator(), grid, X, y,
        checkpoint_dir=tmp_path/'off', screening_statistics_recovery='off', **options)
    assert science(default) == science(explicit)
    for report, directory in ((default, tmp_path/'default'), (explicit, tmp_path/'off')):
        assert 'screening_statistics_recovery' not in report
        assert 'screening_statistics_recovery' not in report['identity']
        assert not (directory/'statistics-events').exists()


@pytest.mark.parametrize('value', ['invalid', None, True])
def test_invalid_public_option_fails_without_checkpoint_or_fit(tmp_path, value):
    X, y = problem()
    with observe_science() as observed, pytest.raises(ValueError, match='screening_statistics_recovery'):
        study.calibration_knn_grid(estimator(), {'knn__n_neighbors': [1]}, X, y,
            checkpoint_dir=tmp_path/'study', screening_statistics_recovery=value,
            cv=2, threads=1)
    assert not any(observed.values()) and not (tmp_path/'study').exists()


@pytest.mark.parametrize('initial_early', [False, True])
def test_public_recovery_mode_is_part_of_checkpoint_identity(tmp_path, initial_early):
    run(tmp_path, early=initial_early)
    before = stable_bytes(tmp_path)
    with observe_science() as observed, pytest.raises(ValueError, match='incompatible'):
        run(tmp_path, early=not initial_early)
    assert not any(observed.values()) and stable_bytes(tmp_path) == before


@pytest.mark.parametrize('completed', [False, True])
def test_private_completed_probe_keeps_its_internal_option_without_new_work(tmp_path, completed):
    if completed:
        original = run(tmp_path)
    else:
        with stop_at('after_event_commit'), pytest.raises(KeyboardInterrupt):
            run(tmp_path)
    before = stable_bytes(tmp_path)
    X, y = problem()
    with observe_science() as observed:
        result = study._reuse_calibration_knn_grid(
            estimator(), {'screen__score_threshold': [0., 2.], 'knn__n_neighbors': [1, 3]},
            X, y, cv=2, seed=29, threads=1, checkpoint_dir=tmp_path,
            _screening_statistics_recovery='early')
    assert not any(observed.values())
    after = stable_bytes(tmp_path)
    assert {k:v for k,v in before.items() if k != 'report.json'} == {k:v for k,v in after.items() if k != 'report.json'}
    if completed:
        assert science(result) == science(original)
        assert_accounting(result)
    else:
        assert result is None and receipts(tmp_path) == []
