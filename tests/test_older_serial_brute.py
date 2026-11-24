"""The older qualified pair has a narrower stock-serial brute domain."""

from contextlib import ExitStack
import warnings

import joblib
from joblib._parallel_backends import ThreadingBackend
import numpy as np
import pytest
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from supervised_nmf import _neighbors


PAIR = sklearn.__version__, joblib.__version__
OLDER = PAIR == ("1.9.0", "1.5.3")
CURRENT = PAIR == ("1.9.1", "1.6.0")


def warning_records(caught):
    return [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize("mode", ["labels", "probabilities"])
@pytest.mark.parametrize("policy", [
    "ordinary", "explicit_one", "parallel_jobs", "assume_finite", "cython_disabled",
    "explicit_context", "custom_context", "registered_factory", "replaced_resolver", "tree",
])
def test_older_serial_brute_uses_only_ordinary_dispatch(mode, policy, monkeypatch):
    train, query = np.zeros((24, 16)), np.zeros((3, 16))
    n_jobs = {"explicit_one": 1, "parallel_jobs": 2}.get(policy)
    algorithm = "kd_tree" if policy == "tree" else "brute"
    model = KNeighborsClassifier(n_neighbors=24, n_jobs=n_jobs, algorithm=algorithm)
    model.fit(train, np.arange(24) % 2)
    probability = mode == "probabilities"
    events = []

    class ObservedBackend(ThreadingBackend):
        def effective_n_jobs(self, n_jobs):
            events.append("effective_n_jobs")
            return 1

    def factory(**kwargs):
        events.append("factory")
        return ObservedBackend(**kwargs)

    with ExitStack() as stack:
        stack.enter_context(np.errstate(divide="warn", over="warn", under="ignore", invalid="warn"))
        disable_cython = {"cython_disabled", "custom_context", "registered_factory", "replaced_resolver"}
        stack.enter_context(sklearn.config_context(
            enable_cython_pairwise_dist=policy not in disable_cython,
            assume_finite=policy == "assume_finite",
        ))
        if policy == "explicit_context":
            stack.enter_context(joblib.parallel_config(n_jobs=1))
        elif policy == "custom_context":
            stack.enter_context(joblib.parallel_config(backend=ObservedBackend(), n_jobs=1))
        elif policy == "registered_factory":
            parallel = joblib.parallel
            monkeypatch.setattr(parallel, "BACKENDS", parallel.BACKENDS.copy())
            monkeypatch.setattr(parallel, "DEFAULT_BACKEND", parallel.DEFAULT_BACKEND)
            joblib.register_parallel_backend("observed-test-backend", factory, make_default=True)
        elif policy == "replaced_resolver":
            original = _neighbors._callback_neighbors_base.effective_n_jobs
            def observe(*args, **kwargs):
                events.append("effective_n_jobs")
                return original(*args, **kwargs)
            monkeypatch.setattr(_neighbors._callback_neighbors_base, "effective_n_jobs", observe)
        with warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter("always")
            expected = (model.predict_proba if probability else model.predict)(query)
        native_events = list(events)
        if policy in {"custom_context", "registered_factory", "replaced_resolver"}:
            assert native_events
        events.clear()

        threshold, positive = (0.5, 1) if probability else (None, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reuse = _neighbors._KNNQueryReuse(
                {24: model}, query, _head_fit_input=train,
                _study_call_contract=((24, probability, threshold, positive),),
            )
            assert not events
            actual, _, shared = reuse.predict(
                24, probabilities=probability, decision_threshold=threshold, positive_index=positive,
            )
        np.testing.assert_array_equal(actual, expected)
        assert events == native_events
        assert warning_records([*caught, *shared]) == warning_records(native_warnings)
        short = bool(reuse.query_records()[0].get("study_short_query_active"))
        if not CURRENT:
            assert short == (OLDER and policy in {"ordinary", "explicit_one"})
        elif policy in {"ordinary", "explicit_one"}:
            assert short
