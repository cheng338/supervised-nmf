"""Complete current NMF and SPA recovery without refitting saved models."""
import hashlib
import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA, NMFClassifier
from supervised_nmf.evaluation import provenance


@pytest.mark.parametrize('kind', ['spa', 'classifier_spa', 'classifier_cd',
                                  'classifier_mu', 'classifier_rowwise'])
def test_current_model_recovery_preserves_complete_state_without_refitting(tmp_path, monkeypatch, kind):
    examples = Path(__file__).resolve().parents[1] / 'examples'
    monkeypatch.syspath_prepend(str(examples))
    recovery = importlib.import_module('recovery_store')
    codec = importlib.import_module('numeric_codec')
    rng = np.random.default_rng(20261105)
    weights = rng.uniform(.2, 2, (48, 3))
    basis = np.column_stack((np.eye(3), rng.uniform(.1, .7, (3, 9))))
    X = weights @ basis + rng.uniform(0, .001, (48, 12))
    y = (weights[:, 0] > np.median(weights[:, 0])).astype(int)
    query = np.vstack((X[:7], np.zeros((1, 12))))
    if kind == 'spa':
        model = FeatureSPA(n_components=3)
    else:
        policy = kind.removeprefix('classifier_')
        model = NMFClassifier(n_components=3, screening='none', n_neighbors=3,
                              solver='mu' if policy == 'rowwise' else policy,
                              mu_projection='rowwise' if policy == 'rowwise' else 'batch',
                              max_iter=2000, random_state=17)
    with threadpool_limits(limits=1):
        model.fit(X, y)
        expected = model.transform(query)
        labels = None if kind == 'spa' else model.predict(query)
        store = recovery.RecoveryStore(tmp_path / 'store',
                                       {'source': provenance(1)['source_sha256'], 'case': kind})
        store.commit({'fit_seconds': 1.25}, model=model)
        outcome = store.load_outcome()
        assert outcome['manifest']['codec']['codec'] == codec.CODEC
        assert outcome['manifest']['metadata']['fit_seconds'] == 1.25
        before = {p.relative_to(tmp_path): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in tmp_path.rglob('*') if p.is_file()}
        with patch.object(NMFClassifier, 'fit', side_effect=AssertionError('restore must not fit')), \
                patch.object(FeatureSPA, 'fit', side_effect=AssertionError('restore must not fit')):
            restored, _, stats = store.restore_model(outcome)
            actual = restored.transform(query)
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual, expected)
            if labels is not None:
                np.testing.assert_array_equal(restored.predict(query), labels)
        assert restored.get_params() == model.get_params()
        np.testing.assert_array_equal(restored.components_, model.components_)
        if kind != 'spa':
            assert restored.mu_projection_ == model.mu_projection_
            assert restored.training_coordinates_mode_ == model.training_coordinates_mode_
        assert stats['reconstructed_knn_heads'] == 0
        after = {p.relative_to(tmp_path): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in tmp_path.rglob('*') if p.is_file()}
        assert after == before
