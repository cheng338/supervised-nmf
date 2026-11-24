"""Ordinary query sharing must preserve the selected estimator's callback sequence."""

from contextlib import nullcontext
from unittest.mock import patch
import warnings

import joblib
from joblib._parallel_backends import ThreadingBackend
import numpy as np
import pytest
from sklearn import config_context
from sklearn.neighbors import KDTree, KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf import _neighbors


def heads(backend, jobs):
    train = np.arange(24, dtype=float)[:, None]
    labels = np.repeat([0, 1], 12)
    models = {k: KNeighborsClassifier(n_neighbors=k, algorithm=backend, metric="euclidean",
                                     n_jobs=jobs).fit(train, labels) for k in (1, 3)}
    return train, np.array([[0.125], [18.25]]), models


def observe(train, query, models, *, share):
    reuse = _neighbors._KNNQueryReuse(models, query, _head_fit_input=train) if share else None
    results, attempted = [], []
    for k, model in models.items():
        with warnings.catch_warnings(record=True) as visible:
            warnings.simplefilter("always")
            try:
                if share:
                    value, meta, extra = reuse.predict(k)
                    attempted.append(meta["shared_query_attempted"])
                    visible.extend(extra)
                else:
                    value = model.predict(query)
                result = value.tolist()
            except RuntimeError as exc:
                result = (type(exc).__name__, str(exc))
        results.append((result, [(w.category, str(w.message)) for w in visible]))
    return results, attempted


@pytest.mark.parametrize("registration", ["context", "default_factory"])
@pytest.mark.parametrize("jobs", [None, 1])
@pytest.mark.parametrize("route", ["compiled_brute", "chunked_brute", "kd_tree"])
def test_public_joblib_controls_reach_each_selected_head(route, jobs, registration, monkeypatch):
    algorithm = "kd_tree" if route == "kd_tree" else "brute"
    train, query, models = heads(algorithm, jobs)
    events = []

    class Backend(ThreadingBackend):
        def effective_n_jobs(self, n_jobs):
            events.append(("effective", n_jobs))
            return 1

        def configure(self, *args, **kwargs):
            events.append(("configure",))
            return super().configure(*args, **kwargs)

    def factory(**kwargs):
        events.append(("factory",))
        return Backend(**kwargs)

    with threadpool_limits(1), config_context(enable_cython_pairwise_dist=route != "chunked_brute"):
        _, ordinary_attempted = observe(train, query, models, share=True)
    if registration == "default_factory":
        parallel = joblib.parallel
        monkeypatch.setattr(parallel, "BACKENDS", parallel.BACKENDS.copy())
        monkeypatch.setattr(parallel, "DEFAULT_BACKEND", parallel.DEFAULT_BACKEND)
        joblib.register_parallel_backend("selected-knn-test", factory, make_default=True)

    def run(share):
        scope = joblib.parallel_config(backend=Backend()) if registration == "context" else nullcontext()
        with scope:
            events.clear()
            outcomes, attempted = observe(train, query, models, share=share)
            return outcomes, list(events), attempted

    with threadpool_limits(1), config_context(enable_cython_pairwise_dist=route != "chunked_brute"):
        expected, native_events, _ = run(False)
        actual, shared_events, attempted = run(True)
    assert actual == expected and shared_events == native_events
    # For compiled brute with an explicit one, effective_n_jobs returns before
    # consulting either public Joblib control. Chunked brute still uses fallback.
    allowed = route == "compiled_brute" and jobs == 1
    assert attempted == (ordinary_attempted if allowed else [False, False])
    if allowed:
        assert not native_events
    elif jobs is None or route == "kd_tree":
        assert native_events


@pytest.mark.parametrize("slot", ["__sklearn_tags__", "__sklearn_is_fitted__"])
def test_selected_instance_validation_hooks_run_on_an_ordinary_grid(slot, monkeypatch):
    train, query, models = heads("brute", 1)
    model = models[1]
    original = getattr(model, slot, lambda: True)
    events = []

    def selected():
        events.append(model.n_neighbors)
        warnings.warn("selected KNN validation", UserWarning)
        return original()

    with threadpool_limits(1):
        _, ordinary_attempted = observe(train, query, models, share=True)
        monkeypatch.setattr(model, slot, selected, raising=False)
        expected, _ = observe(train, query, models, share=False)
        native_events = list(events)
        events.clear()
        actual, attempted = observe(train, query, models, share=True)
    assert native_events and events == native_events
    assert actual == expected and attempted == [False, ordinary_attempted[1]]


def test_changed_kd_tree_query_sees_selected_width_and_return_distance():
    train, query, models = heads("kd_tree", 1)
    original = KDTree.query
    events = []

    def selected(tree, X, k, return_distance):
        events.append((k, return_distance))
        if k == 3:
            raise RuntimeError("selected KDTree width")
        return original(tree, X, k, return_distance)

    with threadpool_limits(1), patch.object(KDTree, "query", selected):
        expected, _ = observe(train, query, models, share=False)
        native_events = list(events)
        events.clear()
        actual, _ = observe(train, query, models, share=True)
    assert events == native_events == [(1, False), (3, False)]
    assert actual == expected
