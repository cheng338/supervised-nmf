"""Native behavior and ownership contracts for uniform full-training KNN heads."""

from contextlib import contextmanager
import sys
import warnings

import joblib
from joblib import parallel_backend, parallel_config
from joblib._parallel_backends import ThreadingBackend
import numpy as np
import pytest
import sklearn
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits

import supervised_nmf._neighbors as neighbors


ORDINARY = dict(divide="warn", over="warn", under="ignore", invalid="warn")
QUALIFIED_RUNTIME = sklearn.__version__ == "1.9.1" and joblib.__version__ == "1.6.0"
SERIAL_BRUTE_RUNTIME = QUALIFIED_RUNTIME or (
    sklearn.__version__ == "1.9.0" and joblib.__version__ == "1.5.3")
requires_qualified_runtime = pytest.mark.skipif(
    not SERIAL_BRUTE_RUNTIME,
    reason="The stock serial brute full-membership proof requires a qualified pair.",
)


@pytest.fixture(autouse=True)
def ordinary_policy_and_native_budget():
    original_callback = np.geterrcall()
    try:
        np.seterrcall(None)
        with threadpool_limits(limits=1), np.errstate(**ORDINARY):
            yield
    finally:
        np.seterrcall(original_callback)


def problem(*, dtype=np.float64, algorithm="brute", classes=2,
            order=(1, 3, 24), zero=False, n_jobs=1):
    random = np.random.RandomState(8127)
    X = np.zeros((24, 4), dtype=dtype) if zero else random.normal(size=(24, 4)).astype(dtype)
    Q = np.zeros((3, 4), dtype=dtype) if zero else random.normal(size=(3, 4)).astype(dtype)
    y = np.asarray(["a", "m", "z"][:classes])[np.arange(len(X)) % classes]
    models = {k: KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                      algorithm=algorithm, n_jobs=n_jobs).fit(X, y)
              for k in order}
    return X, Q, y, models


def contract(models):
    return tuple((k, True, 0.5, 1) for k in models)


def predict(reuse, k):
    return reuse.predict(k, probabilities=True, decision_threshold=0.5, positive_index=1)


def assert_probability(actual, expected):
    # Uniform all-n counts have an exact native result, including class order.
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0, atol=1e-15)


@contextmanager
def native_queries(models=None, *, fault=None, on_shared=None):
    """Observe original method code without replacing a callable admission pin."""
    original_profile = sys.getprofile()
    code = KNeighborsClassifier.kneighbors.__code__
    model_ids = None if models is None else {id(model) for model in models.values()}
    calls = []
    injected = False

    def profile(frame, event, arg):
        nonlocal injected
        if frame.f_code is not code:
            return
        model = frame.f_locals["self"]
        if model_ids is not None and id(model) not in model_ids:
            return
        count = frame.f_locals.get("n_neighbors")
        count = model.n_neighbors if count is None else count
        if event == "call":
            calls.append((model.n_neighbors, count))
        elif (event == "return" and on_shared is not None and frame.f_back is not None
              and frame.f_back.f_code is neighbors._KNNQueryReuse._query.__code__):
            on_shared(count, arg)
        elif (event == "return" and fault is not None and not injected
              and frame.f_back is not None and frame.f_back.f_code.co_name == "_study_query"):
            injected = True
            if fault == "warning":
                warnings.warn("short-query observer warning", RuntimeWarning)
            else:
                raise RuntimeError("short-query observer failure")

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(original_profile)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
@pytest.mark.parametrize("classes", [2, 3])
def test_full_membership_preserves_native_ties_thresholds_and_diagnostics(dtype, algorithm, classes):
    X, Q, y, models = problem(dtype=dtype, algorithm=algorithm, classes=classes)
    n = len(X)
    expected_labels = models[n].predict(Q)
    expected_probability = models[n].predict_proba(Q)
    reuse = neighbors._KNNQueryReuse(models, Q, _head_fit_input=X)
    actual_labels, label_meta, caught = reuse.predict(n)
    actual, metadata, caught_probability = reuse.predict(
        n, probabilities=True, decision_threshold=float(1 / classes), positive_index=classes - 1)
    np.testing.assert_array_equal(actual_labels, expected_labels)
    assert_probability(actual, expected_probability)
    np.testing.assert_array_equal(actual[:, -1] > 1 / classes, expected_probability[:, -1] > 1 / classes)
    assert not caught and not caught_probability
    if algorithm == "ball_tree":
        for current in (label_meta, metadata):
            assert current["reason"] == "selected_width_ball_tree"
            assert not current["accelerated"] and not current["shared_query_attempted"]
            assert current["public_fallback_query_rows"] == len(Q)
        return
    assert metadata["near_vote_rows"] == len(Q)
    assert metadata["near_threshold_rows"] == len(Q)
    assert metadata["near_boundary_rows"] == metadata["invalid_vote_rows"] == 0
    assert metadata["uncertain_rows"] == len(Q)  # Certification must not erase diagnostics.
    for current in (label_meta, metadata):
        assert bool(current.get("full_membership_certified")) is (SERIAL_BRUTE_RUNTIME if algorithm == "brute" else QUALIFIED_RUNTIME)
        if (SERIAL_BRUTE_RUNTIME if algorithm == "brute" else QUALIFIED_RUNTIME):
            assert current["accelerated"] and current["public_fallback_query_rows"] == 0
        else:
            assert current["public_fallback_query_rows"] == len(Q)


@pytest.mark.parametrize("layout", ["fortran_query", "readonly"])
def test_supported_layouts_and_returned_arrays_preserve_input_ownership(layout):
    X, Q, y, models = problem(zero=True)
    if layout == "fortran_query":
        Q = np.asfortranarray(Q)
    else:
        X.flags.writeable = False
        Q.flags.writeable = False
    before = X.tobytes(), Q.tobytes(), y.tobytes(), models[24]._y.tobytes()
    expected = models[24].predict_proba(Q)
    reuse = neighbors._KNNQueryReuse(models, Q)
    first, metadata, _ = predict(reuse, 24)
    assert_probability(first, expected)
    assert bool(metadata.get("full_membership_certified")) is SERIAL_BRUTE_RUNTIME
    first[:] = 17
    second, _, _ = predict(reuse, 24)
    assert_probability(second, expected)
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(second, X) and not np.shares_memory(second, models[24]._y)
    assert (X.tobytes(), Q.tobytes(), y.tobytes(), models[24]._y.tobytes()) == before


@pytest.mark.parametrize("boundary", ["magnitude", "strict_policy", "error_callback",
                                     "array_subclass", "sklearn_version", "joblib_version",
                                     "selected_method", "invalid_threshold"])
def test_unknown_or_unsupported_full_context_retains_public_fallback(boundary, monkeypatch):
    X, Q, y, models = problem(dtype=np.float32 if boundary == "magnitude" else np.float64, zero=True)
    if boundary == "magnitude":
        X[:] = np.float32(4e18)
        Q[:] = np.float32(4e18)
    elif boundary == "array_subclass":
        class Query(np.ndarray):
            pass
        Q = Q.view(Query)
    elif boundary == "sklearn_version":
        monkeypatch.setattr(neighbors, "_SKLEARN_VERSION", "future-version")
    elif boundary == "joblib_version":
        monkeypatch.setattr(neighbors, "_JOBLIB_CALLBACK_VERSION", "future-version")
    elif boundary == "strict_policy":
        np.seterr(under="raise")
    elif boundary == "error_callback":
        np.seterrcall(lambda *args: None)
    selected_calls = []
    if boundary == "selected_method":
        original = models[24].predict_proba
        def selected(query):
            selected_calls.append(query)
            return original(query)
        monkeypatch.setattr(models[24], "predict_proba", selected)
    expected = models[24].predict_proba(Q)
    selected_calls.clear()
    threshold = -float(np.nextafter(0.0, 1.0)) if boundary == "invalid_threshold" else 0.5
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    assert getattr(reuse, "_study_plan", None) is None or boundary == "invalid_threshold"
    actual, metadata, _ = reuse.predict(24, probabilities=True,
                                       decision_threshold=threshold, positive_index=1)
    assert_probability(actual, expected)
    assert not metadata.get("full_membership_certified", False)
    assert not metadata["accelerated"]
    assert metadata["public_fallback_query_rows"] == len(Q)
    if boundary == "selected_method":
        assert len(selected_calls) == 1 and selected_calls[0] is Q


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_full_endpoint_keeps_native_query_validation_error(value):
    _, Q, _, models = problem()
    Q[0, 0] = value
    with pytest.raises(ValueError) as native:
        models[24].predict_proba(Q)
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    with pytest.raises(ValueError) as actual:
        predict(reuse, 24)
    assert str(actual.value) == str(native.value)
    assert not reuse.groups[0].get("study_short_query_active", False)


@pytest.mark.parametrize("order", [(1, 3, 24), (24, 1, 3), (24,)])
def test_owned_query_width_and_once_only_accounting_match_native(order):
    X, Q, y, models = problem(order=order, zero=True)
    expected = {k: model.predict_proba(Q) for k, model in models.items()}
    before = X.tobytes(), Q.tobytes(), y.tobytes()
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    with native_queries(models) as calls:
        for k in models:
            actual, metadata, caught = predict(reuse, k)
            assert_probability(actual, expected[k])
            assert not caught
            if k == 24 and SERIAL_BRUTE_RUNTIME:
                assert metadata["full_membership_vote_basis"] == "short_query_training_histogram"
                assert metadata["full_membership_certified"]
    record = reuse.query_records()[0]
    if SERIAL_BRUTE_RUNTIME:
        count = 1 if len(models) == 1 else 4
        assert record["shared_query_neighbor_counts"] == [count]
        assert record["study_short_query_active"]
        assert [width for k, width in calls if k == 24] == [count]
        assert 0 <= record["short_query_attempt_seconds"] <= record["query_seconds"]
    else:
        assert "shared_query_neighbor_counts" not in record
        assert getattr(reuse, "_study_plan", None) is None
    assert reuse.query_seconds == sum(row["query_seconds"] for row in reuse.query_records())
    assert (X.tobytes(), Q.tobytes(), y.tobytes()) == before


@requires_qualified_runtime
def test_active_owned_contract_rejects_a_later_changed_request():
    _, Q, _, models = problem(zero=True)
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    predict(reuse, 1)
    before = reuse.query_seconds
    with pytest.raises(RuntimeError, match="call contract changed"):
        reuse.predict(3, probabilities=True, decision_threshold=0.25, positive_index=1)
    assert reuse.query_seconds == before
    assert reuse.query_records()[0]["shared_query_neighbor_counts"] == [4]


@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree"])
def test_owned_tree_route_remains_a_full_native_query(algorithm):
    X, Q, _, models = problem(algorithm=algorithm, zero=True)
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models),
                                    _head_fit_input=X)
    assert getattr(reuse, "_study_plan", None) is None
    with native_queries(models) as calls:
        actual, _, _ = predict(reuse, 24)
    assert_probability(actual, models[24].predict_proba(Q))
    assert calls[0] == (24, 24)
    record = reuse.query_records()[0]
    assert "shared_query_neighbor_counts" not in record
    assert record["queried"] is (algorithm == "kd_tree")


def test_one_training_row_keeps_the_native_single_head_route():
    X, Q, y = np.zeros((1, 2)), np.zeros((2, 2)), np.asarray(["a"])
    model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", n_jobs=1).fit(X, y)
    reuse = neighbors._KNNQueryReuse({1: model}, Q, _study_call_contract=((1, True, None, None),))
    assert getattr(reuse, "_study_plan", None) is None
    actual, metadata, caught = reuse.predict(1, probabilities=True)
    assert_probability(actual, model.predict_proba(Q))
    assert metadata["reason"] == "single_head_backend" and not caught
    assert not reuse.query_records()[0]["queried"] and reuse.query_seconds == 0


@requires_qualified_runtime
@pytest.mark.parametrize("sole", [False, True])
@pytest.mark.parametrize("fault", ["warning", "error"])
def test_short_query_warning_or_error_uses_original_fallback_and_accounts_for_attempt(sole, fault):
    _, Q, _, models = problem(order=(24,) if sole else (24, 1, 3), zero=True)
    expected = models[24].predict_proba(Q)
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    with warnings.catch_warnings(record=True) as visible:
        warnings.simplefilter("always")
        with native_queries(models, fault=fault):
            actual, metadata, shared = predict(reuse, 24)
    assert_probability(actual, expected)
    assert not visible and not shared
    record = reuse.query_records()[0]
    assert record["shared_query_neighbor_counts"] == ([1] if sole else [4, 24])
    assert not record["study_short_query_active"]
    assert record["short_query_upgrade_reason"] == (
        "single_head_query_public_fallback" if sole else "short_query_warning_or_error")
    if fault == "warning":
        assert record["short_query_warning_records"] == [
            {"category": "RuntimeWarning", "message": "short-query observer warning"}]
    else:
        assert record["short_query_error_type"] == "RuntimeError"
        assert record["short_query_error"] == "short-query observer failure"
    assert reuse.query_seconds == record["query_seconds"]
    assert 0 <= record["short_query_attempt_seconds"] <= reuse.query_seconds
    if sole:
        assert metadata["reason"] == "single_head_backend"
        assert metadata["public_fallback_query_rows"] == len(Q)


@pytest.mark.parametrize("route", ["generic", "owned_multi", "owned_sole"])
def test_optional_histogram_allocation_failure_preserves_native_output_and_retry(route, monkeypatch):
    order = (24,) if route == "owned_sole" else (24, 1, 3)
    _, Q, _, models = problem(zero=True, order=order)
    expected = models[24].predict_proba(Q)
    reuse = neighbors._KNNQueryReuse(models, Q,
        _study_call_contract=contract(models) if route != "generic" else None)
    original = np.bincount
    failed = []
    def allocation_failure(values, *args, **kwargs):
        if values is models[24]._y:
            failed.append(True)
            raise MemoryError("optional histogram allocation")
        return original(values, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(np, "bincount", allocation_failure)
        actual, metadata, _ = predict(reuse, 24)
    assert_probability(actual, expected)
    assert not metadata.get("full_membership_certified", False)
    if SERIAL_BRUTE_RUNTIME:
        assert failed
        if route == "owned_multi":
            assert reuse.query_records()[0]["shared_query_neighbor_counts"] == [4, 24]
            assert reuse.query_records()[0]["short_query_upgrade_reason"] == "full_histogram_not_available"
        elif route == "owned_sole":
            assert reuse.query_records()[0]["shared_query_neighbor_counts"] == [1]
            assert reuse.query_records()[0]["short_query_upgrade_reason"] == "single_head_histogram_public_fallback"
    retry = neighbors._KNNQueryReuse(models, Q,
        _study_call_contract=contract(models) if route != "generic" else None)
    result, retried, _ = predict(retry, 24)
    assert_probability(result, expected)
    assert bool(retried.get("full_membership_certified")) is SERIAL_BRUTE_RUNTIME


class ReportingBackend(ThreadingBackend):
    """A supported public Joblib callback, not a replaced KNN method."""
    def __init__(self, behavior):
        super().__init__()
        self.behavior = behavior
        self.calls = []

    def effective_n_jobs(self, n_jobs):
        self.calls.append(n_jobs)
        if len(self.calls) == 2:
            if self.behavior == "warning":
                warnings.warn("public backend second call", RuntimeWarning)
            elif self.behavior == "error":
                raise RuntimeError("public backend second call")
        return 1


@pytest.mark.parametrize("leaf_size, signal_k", [(32, 3), (1, 24)])
@pytest.mark.parametrize("mode", ["predict", "predict_proba"])
@pytest.mark.parametrize("signal", ["warning", "error"])
@pytest.mark.parametrize("algorithm, dispatch", [
    ("ball_tree", "configured"), ("kd_tree", "configured"), ("kd_tree", "registered_default"),
    ("kd_tree", "registered_threading"), ("kd_tree", "wrapped_configure"),
])
def test_tree_preserves_selected_query_callbacks(leaf_size, signal_k, mode, signal,
                                                 algorithm, dispatch, monkeypatch):
    X = np.arange(24, dtype=float)[:, None]
    y = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    Q = np.array([[.125], [18.25]])
    heads = {k: KNeighborsClassifier(n_neighbors=k, algorithm=algorithm,
                                     leaf_size=leaf_size, n_jobs=1).fit(X, y)
             for k in (1, 3, 24)}

    class Backend(ThreadingBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def observe_query(self):
            self.calls += 1
            if self.calls == list(heads).index(signal_k) + 1:
                if signal == "error":
                    raise RuntimeError("selected query")
                warnings.warn("selected query", RuntimeWarning)

        def configure(self, *args, **kwargs):
            self.observe_query()
            return super().configure(*args, **kwargs)

    @contextmanager
    def backend_scope(backend):
        if dispatch == "configured":
            with parallel_backend(backend, n_jobs=1):
                yield
            return
        with monkeypatch.context() as local:
            if dispatch == "wrapped_configure":
                original = ThreadingBackend.configure
                def configure(self, *args, **kwargs):
                    backend.observe_query()
                    return original(self, *args, **kwargs)
                local.setattr(ThreadingBackend, "configure", configure)
            else:
                # Registration is process-wide. Keep both the registry and its
                # default scoped to this observation.
                parallel = joblib.parallel
                local.setattr(parallel, "BACKENDS", parallel.BACKENDS.copy())
                local.setattr(parallel, "DEFAULT_BACKEND", parallel.DEFAULT_BACKEND)
                use_default = dispatch == "registered_default"
                joblib.register_parallel_backend("n6-test" if use_default else "threading",
                                                 lambda **kwargs: backend, make_default=use_default)
            yield

    def observe(shared):
        backend = Backend()
        observations = []
        with backend_scope(backend):
            reuse = neighbors._KNNQueryReuse(heads, Q, _head_fit_input=X) if shared else None
            for k, model in heads.items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    shared_warnings = []
                    try:
                        if shared:
                            value, _, shared_warnings = reuse.predict(k, probabilities=mode == "predict_proba")
                        else:
                            value = getattr(model, mode)(Q)
                        outcome = value.tolist()
                    except RuntimeError as exc:
                        outcome = type(exc).__name__, str(exc)
                observations.append((outcome, [(w.category, str(w.message)) for w in caught + shared_warnings]))
        return observations, backend.calls

    expected, calls = observe(False)
    assert calls == len(heads)
    observation, recorded = expected[list(heads).index(signal_k)]
    if signal == "warning":
        assert (RuntimeWarning, "selected query") in recorded
    else:
        assert observation == ("RuntimeError", "selected query")
    assert observe(True) == (expected, calls)


def test_unrelated_joblib_registration_during_kd_tree_prediction(monkeypatch):
    if not neighbors._plain_tree_dispatch():
        pytest.skip("Requires ordinary Joblib tree dispatch.")
    X, query, _, heads = problem(algorithm="kd_tree", order=(1, 3))
    expected = heads[1].predict(query)
    reuse = neighbors._KNNQueryReuse(heads, query, _head_fit_input=X)
    registry = joblib.parallel.BACKENDS.copy()
    monkeypatch.setattr(joblib.parallel, "BACKENDS", registry)
    marker, added = "unrelated-existing-backend", "unrelated-new-backend"
    joblib.register_parallel_backend(marker, ThreadingBackend)

    # Schedule one real registration exactly as the guard reads this key.
    def type_at_registration(value):
        if value is marker and added not in registry:
            joblib.register_parallel_backend(added, ThreadingBackend)
        return type(value)

    monkeypatch.setattr(neighbors, "type", type_at_registration, raising=False)
    actual, _, _ = reuse.predict(1)
    assert added in registry
    np.testing.assert_array_equal(actual, expected)


def observe_full_native_route(models, Q, backend, *, adapter):
    shared, value, error = [], None, None
    reuse = None
    with parallel_backend(backend, n_jobs=2), warnings.catch_warnings(record=True) as visible:
        warnings.simplefilter("always")
        try:
            if adapter:
                reuse = neighbors._KNNQueryReuse(models, Q)
                predict(reuse, 1)
                value, metadata, shared = predict(reuse, 24)
                assert not metadata.get("full_membership_certified", False)
            else:
                models[1].predict_proba(Q)
                value = models[24].predict_proba(Q)
        except Exception as failure:
            error = type(failure), str(failure)
    if reuse is not None:
        shared = reuse.groups[0]["warnings"]
    describe = lambda caught: [(item.category, str(item.message)) for item in caught]
    return value, error, describe(shared), describe(visible), backend.calls


@pytest.mark.parametrize("n_jobs", [None, 2])
@pytest.mark.parametrize("behavior", ["silent", "warning", "error"])
def test_public_custom_backend_keeps_warning_error_and_callback_sequence(n_jobs, behavior):
    _, Q, _, models = problem(zero=True, n_jobs=n_jobs)
    reference = observe_full_native_route(models, Q, ReportingBackend(behavior), adapter=False)
    actual = observe_full_native_route(models, Q, ReportingBackend(behavior), adapter=True)
    assert actual[1:] == reference[1:]
    if QUALIFIED_RUNTIME:
        assert len(actual[-1]) == 2
    if reference[0] is None:
        assert actual[0] is None
    else:
        assert_probability(actual[0], reference[0])


def test_owned_admission_checks_smaller_heads_without_invoking_custom_backend():
    _, Q, _, models = problem(zero=True)
    models[1].n_jobs = None
    backend = ReportingBackend("error")
    with parallel_backend(backend, n_jobs=2):
        reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
        assert getattr(reuse, "_study_plan", None) is None
        assert backend.calls == []
    assert_probability(predict(reuse, 24)[0], models[24].predict_proba(Q))


@pytest.mark.parametrize("backend", ["default", "threading"])
def test_known_stock_backend_keeps_default_jobs_endpoint_eligible(backend):
    _, Q, _, models = problem(zero=True, n_jobs=None, order=(24, 1, 3))
    context = parallel_config() if backend == "default" else parallel_backend("threading", n_jobs=2)
    with context:
        expected = models[24].predict_proba(Q)
        reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
        actual, metadata, _ = predict(reuse, 24)
    assert_probability(actual, expected)
    assert bool(metadata.get("full_membership_certified")) is QUALIFIED_RUNTIME
    assert bool(getattr(reuse, "_study_plan", None)) is QUALIFIED_RUNTIME
    if QUALIFIED_RUNTIME:
        assert reuse.query_records()[0]["shared_query_neighbor_counts"] == [4]


@pytest.mark.parametrize("hook", ["tags", "fitted"])
def test_instance_validation_hooks_are_not_skipped(hook, monkeypatch):
    _, Q, _, models = problem(zero=True)
    expected = models[24].predict_proba(Q)
    calls = []
    if hook == "tags":
        original = models[24].__sklearn_tags__
        def callback():
            calls.append(True)
            return original()
        monkeypatch.setattr(models[24], "__sklearn_tags__", callback)
    else:
        def callback():
            calls.append(True)
            return True
        monkeypatch.setattr(models[24], "__sklearn_is_fitted__", callback, raising=False)
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    assert getattr(reuse, "_study_plan", None) is None
    assert not calls  # Admission must inspect ownership rather than invoking a hook.
    actual, metadata, _ = predict(reuse, 24)
    assert_probability(actual, expected)
    assert not metadata.get("full_membership_certified", False)
    assert calls


def test_unknown_default_backend_factory_is_not_called_by_admission(monkeypatch):
    _, Q, _, models = problem(zero=True, n_jobs=None)
    calls = []
    def factory(*args, **kwargs):
        calls.append(True)
        raise AssertionError("admission invoked an unknown backend factory")
    from joblib import parallel
    with parallel_config(), monkeypatch.context() as patch:
        patch.setitem(parallel.BACKENDS, parallel.DEFAULT_BACKEND, factory)
        reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
        assert getattr(reuse, "_study_plan", None) is None
        assert not neighbors._native_callback_context(models[24])
        assert calls == []


@pytest.mark.parametrize("hidden_metric", ["manhattan", "callable"])
def test_public_prefit_tree_with_hidden_metric_retains_native_callbacks(hidden_metric):
    X, Q, y, _ = problem(zero=True)
    calls = []
    def metric(left, right):
        calls.append(True)
        return float(np.abs(left - right).sum())
    prefit = NearestNeighbors(algorithm="ball_tree",
        metric=metric if hidden_metric == "callable" else "manhattan").fit(X)
    models = {k: KNeighborsClassifier(n_neighbors=k, algorithm="ball_tree",
                                     metric="euclidean", n_jobs=1).fit(prefit, y)
              for k in (1, 24)}
    expected = models[24].predict_proba(Q)
    calls.clear()
    reuse = neighbors._KNNQueryReuse(models, Q, _study_call_contract=contract(models))
    assert getattr(reuse, "_study_plan", None) is None
    actual, metadata, _ = predict(reuse, 24)
    assert_probability(actual, expected)
    assert not metadata.get("full_membership_certified", False)
    assert metadata["public_fallback_query_rows"] == len(Q)
    if hidden_metric == "callable":
        assert calls
