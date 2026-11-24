"""Multiworker KDTree scheduling must observe each selected sklearn query."""
import warnings

import joblib
import pytest

from tests import test_full_query_direct_methods as fixtures
from tests.test_neighbor_tree_delayed import _observe


@pytest.fixture
def scheduling_calls(monkeypatch):
    if joblib.effective_n_jobs(2) != 2:
        pytest.skip("the installed backend disables parallel dispatch")
    calls = []

    def dispatch(self, iterator):
        # The stock threading backend creates its pool only after a batch
        # reaches submit. Intercept its first task on the calling thread.
        function, (_, k, distances), _ = next(iterator)
        assert function.__name__ == "query"
        calls.append((k, distances))
        warnings.warn(f"scheduling width {k}", UserWarning)
        raise RuntimeError(f"stopped before submitting width {k}")

    monkeypatch.setattr(joblib.Parallel, "dispatch_one_batch", dispatch)
    return calls


def test_multiworker_public_grid_uses_selected_widths(scheduling_calls, monkeypatch, tmp_path):
    original = fixtures.head
    monkeypatch.setattr(fixtures, "head", lambda k, algorithm: original(k, algorithm).set_params(n_jobs=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rows = fixtures.assert_public_grid("kd_tree", (1, 3), tmp_path, scheduling_calls)
    for k, folds in rows.items():
        assert folds == [(None, "RuntimeError", f"stopped before submitting width {k}")] * 2
    assert scheduling_calls == [(1, False), (3, False)] * 2


@pytest.mark.parametrize("jobs", [(1, 2), (2, None)])
def test_selected_and_shared_worker_counts_both_matter(jobs, scheduling_calls):
    from supervised_nmf._neighbors import _KNNQueryReuse

    X, y, folds = fixtures.problem()
    train, valid = folds[0]
    training, query = X[train], X[valid]
    models = {k: fixtures.head(k, "kd_tree").set_params(n_jobs=count, weights="distance").fit(training, y[train])
              for k, count in zip((1, 3), jobs)}
    native = [_observe(lambda: (model.predict_proba(query).tolist(), [])) for model in models.values()]
    expected_calls = scheduling_calls.copy()
    assert expected_calls == [(1 if jobs[0] == 2 else 3, True)]
    scheduling_calls.clear()
    reuse = _KNNQueryReuse(models, query, _head_fit_input=training)

    def selected(k):
        value, _, extra = reuse.predict(k, probabilities=True, decision_threshold=0.5, positive_index=1)
        return value.tolist(), extra

    assert [_observe(lambda: selected(k)) for k in models] == native
    assert scheduling_calls == expected_calls
