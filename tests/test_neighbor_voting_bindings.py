"""A shared neighbor query must preserve the selected classifier's vote."""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

if __name__ == "__main__":
    sys.path[:] = json.loads(sys.argv[1])

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.neighbors import _base, _classification
from sklearn.utils import extmath, fixes
from threadpoolctl import threadpool_limits

from tests.test_full_query_direct_methods import (
    assert_public_grid, assert_selected_probabilities, assert_stock_full_query, head, problem,
)


DEFINITIONS = {"_mode": fixes, "weighted_mode": extmath, "_get_weights": _base}


def compare_votes(name, events, selected_changed):
    from supervised_nmf._neighbors import _KNNQueryReuse

    X, y, folds = problem()
    train, valid = folds[0]
    training, query = X[train], X[valid]
    weights = "uniform" if name == "_mode" else "distance"
    for counts in ((1, 3), (1, 3, 12)):
        models = {k: head(k).set_params(weights=weights).fit(training, y[train]) for k in counts}
        for probabilities in (False, True):
            contract = tuple((k, probabilities, None, None) for k in counts) if 12 in counts else None
            adapter = _KNNQueryReuse(models, query, _study_call_contract=contract, _head_fit_input=training)
            for k, model in models.items():
                selected = model.predict_proba if probabilities else model.predict
                events.clear()
                if name == "_get_weights" and selected_changed and k == 12:
                    with pytest.raises(RuntimeError, match="selected full-width weights"):
                        selected(query)
                    assert events == [k]
                    events.clear()
                    with pytest.raises(RuntimeError, match="selected full-width weights"):
                        adapter.predict(k, probabilities=probabilities)
                    assert events == [k]
                else:
                    expected = selected(query)
                    expected_events = events.copy()
                    assert expected_events == ([k] if selected_changed and
                                               (not probabilities or name == "_get_weights") else [])
                    events.clear()
                    actual, metadata, _ = adapter.predict(k, probabilities=probabilities)
                    np.testing.assert_array_equal(actual, expected)
                    assert events == expected_events
                    assert not metadata["shared_query_attempted"]


def check_changes(name, directory):
    events = []

    def class_one(labels, *args, axis=0):
        events.append(labels.shape[1])
        mode = np.ones((len(labels), 1), dtype=np.intp)
        return mode, mode.copy()

    def farthest(distances, weights):
        events.append(distances.shape[1])
        if distances.shape[1] == 12:
            raise RuntimeError("selected full-width weights")
        return (distances == distances[:, -1:]).astype(float)

    replacement = farthest if name == "_get_weights" else class_one
    definition = DEFINITIONS[name]
    assert "supervised_nmf" not in sys.modules
    with threadpool_limits(1):
        # Replace both aliases before the first capture, then replace each alias
        # independently after a stock query has populated the method cache.
        for owners in ((definition, _classification), (_classification,), (definition,)):
            with ExitStack() as stack:
                for owner in owners:
                    stack.enter_context(patch.object(owner, name, replacement))
                compare_votes(name, events, _classification in owners)
                if name == "_mode" and len(owners) == 2:
                    scores = assert_public_grid("brute", (1, 3, 12), directory / "altered", events)
                    assert [row[0] for row in scores[12]] == [0.75, 0.5]
            assert_stock_full_query()
        if name == "_mode":
            scores = assert_public_grid("brute", (1, 3, 12), directory / "restored", events)
            assert [row[0] for row in scores[12]] == [0.25, 0.5]


@pytest.mark.skipif(
    (sklearn.__version__, joblib.__version__) not in {("1.9.1", "1.6.0"), ("1.9.0", "1.5.3")},
    reason="These voting bindings belong to the two qualified scikit-learn builds.",
)
@pytest.mark.parametrize("name", DEFINITIONS)
def test_selected_voting_bindings(name, tmp_path):
    paths = [str(Path(entry or ".").resolve()) for entry in sys.path]
    child = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                            json.dumps(paths), name, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr


def test_selected_full_head_retains_fitted_state_validation(tmp_path, monkeypatch):
    events = []
    original = _classification.check_is_fitted

    def check(model, *args, **kwargs):
        events.append(model.n_neighbors)
        if model.n_neighbors == 12:
            raise RuntimeError("selected full-width validation")
        return original(model, *args, **kwargs)

    with threadpool_limits(1):
        assert_stock_full_query()
        with monkeypatch.context() as changes:
            changes.setattr(_classification, "check_is_fitted", check)
            scores = assert_public_grid("brute", (1, 3, 12), tmp_path / "validation", events)
            assert all(row == (None, "RuntimeError", "selected full-width validation") for row in scores[12])
            assert_selected_probabilities("brute", (1, 3, 12), events)
        assert_stock_full_query()


if __name__ == "__main__":
    check_changes(sys.argv[2], Path(sys.argv[3]))
