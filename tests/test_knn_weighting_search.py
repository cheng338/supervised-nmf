"""Joint k/weight calibration agrees with independent sklearn CV."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.evaluation import calibration_search


@pytest.mark.parametrize("normalization", ["none", "sd"])
def test_joint_weight_and_k_search_matches_independent_fold_fits(normalization):
    rng = np.random.default_rng(20261103)
    X = rng.uniform(.05, 1.0, (96, 12))
    y = np.tile([0, 1], 48)
    X[:, :3] += .8 * y[:, None]
    groups = np.repeat(np.arange(48), 2)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=817)
                 .split(X, y, groups))
    model = NMFClassifier(n_components=2, score_threshold=1., random_state=817,
                          knn_normalization=normalization)
    grid = {"n_neighbors": [3, 7], "weights": ["uniform", "distance"]}
    with threadpool_limits(limits=1):
        independent = []
        for settings in ParameterGrid(grid):
            scores = []
            for train, query in folds:
                assert not set(groups[train]) & set(groups[query])
                fitted = clone(model).set_params(**settings).fit(X[train], y[train])
                scores.append(accuracy_score(y[query], fitted.predict(X[query])))
            independent.append(scores)
        result = calibration_search(model, grid, X, y, cv=3, groups=groups,
                                    folds=folds, refit=False, scoring="accuracy")
    assert result.report["scoring"] == "accuracy"
    for candidate, expected in zip(result.report["candidates"], independent):
        assert candidate["valid"]
        np.testing.assert_array_equal([f["score"] for f in candidate["folds"]], expected)
        assert candidate["mean_score"] == np.mean(expected)
    expected_winner = int(np.argmax(np.mean(independent, axis=1)))
    assert result.report["selected_candidate_index"] == expected_winner
    assert model.weights == "uniform"
    assert model.decision_threshold is None
