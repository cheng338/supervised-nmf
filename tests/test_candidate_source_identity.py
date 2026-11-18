"""Replacement estimator implementation changes must invalidate cached CV."""
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from supervised_nmf import evaluation
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid


def load_replacement(path, column, *, mutate_during_fit=False):
    """Same module/class/configuration, changed source; bypass bytecode caches."""
    path = Path(path)
    text = '''from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
FIT_COUNT = 0
class SelectColumn(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        global FIT_COUNT
        FIT_COUNT += 1
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        return np.asarray(X)[:, COLUMN:COLUMN + 1]
'''.replace('COLUMN', str(column))
    if mutate_during_fit:
        text = text.replace('        FIT_COUNT += 1\n',
                            '        FIT_COUNT += 1\n'
                            '        from pathlib import Path\n'
                            '        source = Path(__file__)\n'
                            '        source.write_text(source.read_text() + "\\n# changed during fit\\n")\n')
    path.write_text(text, encoding='utf-8')
    name = 'snmf_cache_identity_fixture'
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(text, str(path), 'exec'), module.__dict__)
    return module, module.SelectColumn()


def fixture():
    y = np.arange(36) % 2
    X = np.column_stack([y, np.zeros(36), np.arange(36)])
    groups = np.repeat(np.arange(18), 2)
    folds = [(np.flatnonzero(groups % 3 != fold), np.flatnonzero(groups % 3 == fold))
             for fold in range(3)]
    return X, y, groups, folds


def execute(kind, replacement, checkpoint):
    X, y, groups, folds = fixture()
    steps = [('feature', StandardScaler())]
    if kind == 'rank':
        steps.append(('pca', PCA(svd_solver='full', random_state=17)))
    steps.append(('knn', KNeighborsClassifier(n_neighbors=1)))
    estimator = Pipeline(steps)
    grid = {'feature': [replacement], 'knn__n_neighbors': [1]}
    options = dict(cv=3, groups=groups, folds=folds, data_ids=np.arange(len(y)))
    with threadpool_limits(limits=1):
        if kind == 'generic':
            result = calibration_search(estimator, grid, X, y, **options,
                                        checkpoint_path=checkpoint, refit=False)
            return result.report
        if kind == 'shared':
            return calibration_knn_grid(estimator, grid, X, y, **options,
                                        checkpoint_dir=checkpoint)
        grid['pca__n_components'] = [1]
        return calibration_rank_search(estimator, grid, X, y, **options,
                                       checkpoint_dir=checkpoint, max_expansions=0)


@pytest.mark.parametrize('kind', ['generic', 'shared', 'rank'])
def test_changed_replacement_source_rejects_stale_checkpoint(kind, tmp_path):
    module_path = tmp_path / 'replacement.py'
    module, first = load_replacement(module_path, 0)
    first_repr = repr(first)
    original = execute(kind, first, tmp_path / 'cached')
    assert original['selected_mean_score'] == 1.
    assert module.FIT_COUNT == 3
    module, second = load_replacement(module_path, 1)
    assert repr(second) == first_repr
    fresh = execute(kind, second, tmp_path / 'fresh')
    assert fresh['selected_mean_score'] == .5
    assert module.FIT_COUNT == 3
    with pytest.raises(ValueError, match='[Ii]ncompatible|[Ss]ource|[Ii]dentity'):
        execute(kind, second, tmp_path / 'cached')
    assert module.FIT_COUNT == 3, 'Cache rejection must occur before another fit'


@pytest.mark.parametrize('kind', ['generic', 'shared', 'rank'])
def test_unchanged_replacement_source_reuses_checkpoint(kind, tmp_path):
    module_path = tmp_path / 'replacement.py'
    module, replacement = load_replacement(module_path, 0)
    original = execute(kind, replacement, tmp_path / 'cached')
    assert module.FIT_COUNT == 3
    # Recreating the same class from unchanged source must not invalidate it.
    module, replacement = load_replacement(module_path, 0)
    resumed = execute(kind, replacement, tmp_path / 'cached')
    assert resumed['selected_mean_score'] == original['selected_mean_score']
    assert module.FIT_COUNT == 0


def test_generic_digest_reuse_is_local_to_each_search(tmp_path, monkeypatch):
    module, replacement = load_replacement(tmp_path / 'replacement.py', 0)
    original = evaluation.implementation_sources
    calls = []

    def observe(*values, _previous_files=None):
        if len(calls) == 1:  # First guard after the independent entry check.
            assert _previous_files == {}
        calls.append(_previous_files)
        return original(*values, _previous_files=_previous_files)

    monkeypatch.setattr(evaluation, 'implementation_sources', observe)
    stores = []
    first, second = tmp_path / 'first', tmp_path / 'second'
    for checkpoint, expected_fits in ((None, 3), (None, 3), (first, 3), (second, 3), (first, 0)):
        calls.clear()
        fits_before = module.FIT_COUNT
        report = execute('generic', replacement, checkpoint)
        assert report['selected_mean_score'] == 1.
        assert module.FIT_COUNT - fits_before == expected_fits
        assert calls[0] is None  # Initial identity is computed independently.
        guards = calls[1:]
        assert len(guards) == 2 * expected_fits
        if guards:
            store = guards[0]
            assert isinstance(store, dict) and store
            assert all(guard is store for guard in guards)
            assert all(store is not previous for previous in stores)
            stores.append(store)


@pytest.mark.parametrize('kind', ['generic', 'shared', 'rank'])
def test_replacement_source_changed_during_fit_stops_before_completed_receipt(kind, tmp_path):
    import json

    module, replacement = load_replacement(tmp_path / 'replacement.py', 0,
                                           mutate_during_fit=True)
    checkpoint = tmp_path / 'cached'
    with pytest.raises(ValueError, match='[Ss]ource|[Ii]mplementation|[Ii]dentity'):
        execute(kind, replacement, checkpoint)
    assert module.FIT_COUNT == 1, 'Changed implementation must stop before a second fit'
    if kind == 'generic':
        report = json.loads(checkpoint.read_text(encoding='utf-8'))
        assert all(not candidate['folds'] for candidate in report['candidates'])
    else:
        assert not list(checkpoint.glob('**/folds/*.json'))
