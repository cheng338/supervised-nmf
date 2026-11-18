"""Ordered MU recovery and call-local setup reuse.

The scalar solver is the numerical oracle. Read-only profiling observes the
actual optimized helpers without replacing functions used by admission gates.
The controlled fit supplies a dictionary; it does not claim to test native NMF
factorization. All fixtures are owned here and require no investigation files.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import inspect
import pickle
import sys
from threading import Barrier, Event, Lock, get_ident
from types import MethodType
from unittest.mock import patch
import warnings
import weakref

import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_info, threadpool_limits

from supervised_nmf import NMFClassifier
import supervised_nmf.classifier as classifier
import supervised_nmf._mu as mu
import supervised_nmf._mu_rowwise as rowwise


@pytest.fixture(scope="module", autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def fitted(basis, cap=21, tol=1e-8):
    model = NMF(n_components=len(basis), solver="mu", init="random",
                random_state=0, max_iter=cap, tol=tol)
    model.components_ = basis.copy(order="K")
    model.n_components_ = model._n_components = len(basis)
    model.n_features_in_ = basis.shape[1]
    model._beta_loss = 2
    model.n_iter_ = 7
    model.reconstruction_err_ = .25
    return model


def boundary_data(features=3):
    tiny = np.finfo(float).smallest_subnormal
    basis = np.zeros((2, features))
    basis[0, 0], basis[1, 1] = tiny, 1.
    fast = np.zeros(features)
    fast[:3] = [tiny, 1., 1e100]
    slow = fast.copy()
    slow[2] = 0.
    bad = fast.copy()
    bad[0] = 1.
    return basis, fast, slow, bad


def reusable_data():
    rng = np.random.default_rng(765312)
    basis = rng.uniform(.2, .8, (3, 31))
    basis[0, 0] = 2. ** -520
    query = np.ldexp(rng.uniform(.1, .9, (4, 31)), 600)
    return basis, query


def serial(model, query):
    return np.vstack([mu.transform_mu(model, row[None, :]) for row in query])


def outcome(operation, *, strict=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if strict:
            warnings.simplefilter("error", ConvergenceWarning)
        try:
            value, error = operation(), None
        except Exception as exc:
            value, error = None, (type(exc), str(exc))
    return value, [(w.category, str(w.message)) for w in caught], error


def same(actual, expected, *, error=None):
    assert actual[1:] == expected[1:]
    if error is None:
        assert actual[2] is None, actual[2]
        np.testing.assert_array_equal(actual[0], expected[0])
    else:
        assert actual[0] is expected[0] is None
        assert actual[2][0] is error


@contextmanager
def observe(*, hook=None):
    """Record identities and primitive values, never retain frames/cache arrays."""
    names = ("_project_log_mu_rows", "_project_log_mu", "_handoff_log_setup",
             "_row_log_basis", "_row_log_invariant", "_log_sum")
    codes = {getattr(mu, name).__code__: name for name in names}
    # Count the direct panel reduction entry on either original or owned routes.
    # Nested _log_sum fallback below an owned reducer must not count twice.
    owned_sum = getattr(mu, "_owned_log_sum", None)
    if owned_sum is not None:
        codes[owned_sum.__code__] = "_owned_log_sum"
    native = inspect.unwrap(NMF.transform).__code__
    panel = mu._log_cross_panel.__code__
    trace = dict(vectors=[], scalars=[], offers=[], reuse=[], panels=[],
                 warnings=[], native=0, handles=[])
    pending = {}
    missing = object()

    def profile(frame, event, value):
        if hook is not None:
            hook(frame, event, value)
        code, local = frame.f_code, frame.f_locals
        name = codes.get(code)
        if event == "call" and code is native:
            trace["native"] += 1
        if name in ("_log_sum", "_owned_log_sum") and event == "call" and frame.f_back.f_code is panel:
            parent = frame.f_back.f_locals
            trace["panels"].append((parent["left"].shape, parent["right"].shape,
                                    parent["start"], parent["stop"]))
        if name in ("_row_log_basis", "_row_log_invariant"):
            if event == "call":
                key = "basis" if name == "_row_log_basis" else local["key"]
                cache = local["cache"]
                old = cache.get(key, missing) if cache is not None else missing
                old = old[3] if key == "basis" and old is not missing else old
                pending[id(frame)] = key, old
            elif event == "return":
                key, old = pending.pop(id(frame))
                trace["reuse"].append((key, old is not missing and value is old))
        elif name == "_project_log_mu" and event == "call":
            cache = local.get("_basis_cache")
            trace["scalars"].append((None if cache is None else id(cache),
                                     tuple(sorted(cache or {}))))
        elif name == "_project_log_mu" and event == "return":
            trace.setdefault("iterations", []).append(local.get("iteration", 0))
        elif name == "_project_log_mu_rows" and event == "return":
            result = value.result if type(value) is mu._SuffixSetup else value
            count = result.count if type(result) is mu._CappedPrefix else (
                len(local["x"]) if isinstance(result, np.ndarray) else 0)
            trace["vectors"].append(dict(count=count, precap=local.get("precap", False),
                iterations=local.get("iterations", np.array([], dtype=int)).tolist(),
                active=local.get("active_rows", np.array([], dtype=bool)).tolist()))
        elif name == "_handoff_log_setup" and event == "return" and type(value) is mu._SuffixSetup:
            trace["offers"].append((0 if value.result is None else value.result.count, id(value.cache)))
            arrays = [value.cache["basis"][1], value.cache["basis"][3][0],
                      value.cache["gram"], value.cache["first_denominator"]]
            for array in arrays:
                while isinstance(array, np.ndarray):
                    trace["handles"].append(weakref.ref(array))
                    array = array.base
        if name == "_project_log_mu" and event == "c_call" and value is warnings.warn:
            trace["warnings"].append(tuple(sorted(local.get("_basis_cache") or {})))

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        yield trace
    finally:
        sys.setprofile(previous)
        assert not pending


@pytest.fixture(scope="module")
def stopped_rows():
    basis, fast, slow, bad = boundary_data()
    # Real stopping preconditions, not a fabricated return from the solver.
    for row, iterations in ((fast, 10), (slow, 20)):
        with observe() as trace:
            result = outcome(lambda: mu.transform_mu(fitted(basis), row[None, :]))
        assert result[2] is None and not result[1]
        assert trace["iterations"] == [iterations]
    return basis, fast, slow, bad


@pytest.mark.parametrize("shape", [(4, 8192, 5), (33, 2049, 2)])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_log_panel_budget_boundaries_match_scalar(shape, layout):
    rows, features, columns = shape
    rng = np.random.default_rng(712)
    left = rng.normal(size=(rows, features))
    right = rng.normal(size=(columns, features))
    if layout == "F":
        right = np.asfortranarray(right)
    elif layout == "strided":
        padded = np.empty((columns, features * 2))
        padded[:, ::2] = right
        right = padded[:, ::2]
    before = left.tobytes(), right.tobytes()
    expected = mu._log_cross_reference(left, right)
    with observe() as trace:
        actual = mu._log_cross(left, right)
    np.testing.assert_array_equal(actual, expected)
    width = max(1, (65536 if layout == "C" else 262144) // left.size)
    assert [(a, b) for _, _, a, b in trace["panels"]] == [
        (start, min(start + width, columns)) for start in range(0, columns, width)]
    assert (left.tobytes(), right.tobytes()) == before


@pytest.mark.parametrize("cap", [11, 19, 20, 21, 29, 30])
@pytest.mark.parametrize("first_slow", [0, 1, 16])
def test_ordered_prefix_and_precap_match_scalar(stopped_rows, cap, first_slow):
    basis, fast, slow, _ = stopped_rows
    query = np.repeat(fast[None, :], 17, axis=0)
    query[first_slow] = slow
    model = fitted(basis, cap)
    before = pickle.dumps(model), query.tobytes()
    expected = outcome(lambda: serial(fitted(basis, cap), query))
    with observe() as trace:
        actual = outcome(lambda: rowwise.transform_mu_rowwise(model, query))
    same(actual, expected)
    assert len(trace["vectors"]) == 1
    vector = trace["vectors"][0]
    if cap <= 20:
        assert vector["count"] == first_slow and vector["precap"]
        assert vector["iterations"][first_slow] == 10
        assert vector["active"][first_slow]
        assert len(trace["scalars"]) == len(query) - first_slow
        assert trace["offers"] == [(first_slow, trace["scalars"][0][0])]
        assert trace["scalars"][0][1] == ("basis", "first_denominator", "gram")
        assert trace["warnings"] == [()]
        assert all(handle() is None for handle in trace["handles"])
    else:
        assert vector["count"] == len(query)
        assert not trace["scalars"] and not trace["offers"]
    assert (pickle.dumps(model), query.tobytes()) == before


@pytest.mark.parametrize("kind", ["default", "false", "numpy_cap", "numpy_tol", "zero", "negative_zero"])
def test_private_defaults_and_nonbuiltin_controls_keep_original_route(stopped_rows, kind):
    basis, fast, slow, _ = stopped_rows
    query = np.vstack([fast, slow])
    cap, tol = (np.int64(20) if kind == "numpy_cap" else 20), 1e-8
    if kind == "numpy_tol":
        tol = np.float64(tol)
    elif kind in ("zero", "negative_zero"):
        tol = 0. if kind == "zero" else -0.
    kwargs = {} if kind == "default" else {"_retain_capped_prefix": kind != "false"}
    with observe() as trace:
        value = mu._project_log_mu_rows(query, basis, cap, tol, **kwargs)
    assert not trace["vectors"][0]["precap"] and not trace["offers"]
    assert trace["vectors"][0]["iterations"][1] == 20
    if kind in ("default", "false"):
        assert value is None
    elif tol > 0:
        assert type(value) is mu._CappedPrefix and value.count == 1
        assert value._fields == ("count", "coordinates")
    else:
        np.testing.assert_array_equal(value, serial(fitted(basis, cap, tol), query))


def replay(query, basis, mutation, *, optimized):
    model, output, events = fitted(basis, 20), np.full((len(query), len(basis)), -777.), []
    def show(message, category, *args, **kwargs):
        events.append((category, str(message), output.copy()))
        if len(events) != 1:
            return
        if mutation == "basis":
            model.components_[1, 1] *= 1.125
        elif mutation == "replace":
            model.components_ = np.asfortranarray(model.components_ * 2.)
        elif mutation == "query":
            query[-1, 1] *= 1.125
        elif mutation == "cap":
            model.max_iter = 1
        elif mutation == "tol":
            model.tol = 0.
        elif mutation == "native":
            model.transform = MethodType(lambda self, x: np.full((len(x), 2), .75), model)
        elif mutation == "helper":
            rowwise.transform_mu = lambda model, x: np.full((len(x), 2), .625)
        elif mutation == "raise":
            raise RuntimeError("warning callback")
    # Use the same mutable scalar callback binding as the recovery loop.
    with patch.object(rowwise, "transform_mu", rowwise.transform_mu), warnings.catch_warnings(), \
            patch.object(warnings, "showwarning", show), observe() as trace:
        warnings.simplefilter("error" if mutation == "strict" else "always", ConvergenceWarning)
        try:
            if optimized:
                rowwise._recover_rows(model, query, output, np.ones(len(query), dtype=bool))
            else:
                for index in range(len(query)):
                    output[index] = rowwise.transform_mu(model, query[index:index + 1])[0]
            error = None
        except Exception as exc:
            error = type(exc), str(exc)
    return output, events, error, model.components_.copy(), query, trace


@pytest.mark.parametrize("mutation", ["basis", "replace", "query", "cap", "tol", "native", "helper", "raise", "strict"])
def test_warning_replay_retains_prefix_and_fresh_scalar_state(stopped_rows, mutation):
    basis, fast, slow, _ = stopped_rows
    query = np.vstack([fast, slow, fast, fast])
    expected = replay(query.copy(), basis, mutation, optimized=False)
    actual = replay(query.copy(), basis, mutation, optimized=True)
    for index in (0, 3, 4):
        np.testing.assert_array_equal(actual[index], expected[index])
    assert actual[2] == expected[2]
    assert len(actual[1]) == len(expected[1])
    for got, wanted in zip(actual[1], expected[1]):
        assert got[:2] == wanted[:2]
        np.testing.assert_array_equal(got[2], wanted[2])
    if mutation in ("raise", "strict"):
        assert actual[2][0] is (RuntimeError if mutation == "raise" else ConvergenceWarning)
    else:
        assert actual[2] is None
    assert actual[5]["offers"][0][0] == 1
    assert actual[5]["warnings"] and all(keys == () for keys in actual[5]["warnings"])
    if mutation != "strict":
        assert np.all(actual[1][0][2][1:] == -777.)


@pytest.mark.parametrize("bad_first", [False, True])
def test_prefix_restoration_failure_preserves_first_public_error(stopped_rows, bad_first):
    basis, fast, slow, bad = stopped_rows
    query = np.vstack([bad, slow, fast] if bad_first else [fast, slow, bad])
    expected = outcome(lambda: serial(fitted(basis, 11), query))
    with observe() as trace:
        actual = outcome(lambda: rowwise.transform_mu_rowwise(fitted(basis, 11), query))
    same(actual, expected, error=FloatingPointError)
    assert len(actual[1]) == (0 if bad_first else 1)
    assert len(trace["vectors"]) == 1
    assert bool(trace["offers"]) is (not bad_first)


def test_indexing_replacement_preserves_original_callable_resolution(stopped_rows):
    basis, fast, slow, _ = stopped_rows
    original = rowwise._try_log_row_recovery
    calls = []
    def captured(x, h, cap, tol, *, _retain_capped_prefix=False):
        calls.append("captured")
        return original(x, h, cap, tol, _retain_capped_prefix=_retain_capped_prefix)
    def replacement(*args, **kwargs):
        raise AssertionError("Callable was resolved after argument indexing")
    class Query(np.ndarray):
        def __getitem__(self, key):
            if self is query and isinstance(key, np.ndarray):
                rowwise._try_log_row_recovery = replacement
                calls.append("index")
            return super().__getitem__(key)
    query = np.vstack([fast, slow]).view(Query)
    output = np.empty((2, 2))
    expected = outcome(lambda: serial(fitted(basis, 20), np.asarray(query)))
    with patch.object(rowwise, "_try_log_row_recovery", captured):
        actual = outcome(lambda: rowwise._recover_rows(fitted(basis, 20), query, output, np.ones(2, bool)))
    assert actual[2] is None and actual[1] == expected[1]
    np.testing.assert_array_equal(output, expected[0])
    assert calls == ["index", "captured"]


def test_sparse_recovery_mask_publishes_prefix_in_original_positions(stopped_rows):
    basis, fast, slow, _ = stopped_rows
    query = np.vstack([fast, fast, fast, slow, fast])
    recovery = np.asarray([False, True, False, True, True])
    expected, actual = np.full((5, 2), -777.), np.full((5, 2), -777.)
    model = fitted(basis, 20)
    def scalar():
        for index in np.flatnonzero(recovery):
            expected[index] = mu.transform_mu(model, query[index:index + 1])[0]
    oracle = outcome(scalar)
    with observe() as trace:
        result = outcome(lambda: rowwise._recover_rows(fitted(basis, 20), query, actual, recovery))
    assert result[2] is oracle[2] is None and result[1] == oracle[1]
    np.testing.assert_array_equal(actual, expected)
    assert trace["vectors"][0]["count"] == 1
    assert trace["offers"][0][0] == 1 and len(trace["scalars"]) == 2
    np.testing.assert_array_equal(actual[~recovery], np.full((2, 2), -777.))


def test_prefix_output_alias_cannot_bless_stale_handoff(stopped_rows):
    basis, fast, slow, _ = stopped_rows
    query = np.vstack([fast, slow])
    def solve(optimized):
        model = fitted(basis, 20)
        output = model.components_[:, :2]
        def operation():
            if optimized:
                rowwise._recover_rows(model, query, output, np.ones(2, bool))
            else:
                for index in range(2):
                    output[index] = mu.transform_mu(model, query[index:index + 1])[0]
            return output.copy()
        with observe() as trace:
            result = outcome(operation)
        return result, model.components_, trace
    expected, actual = solve(False), solve(True)
    same(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert not np.array_equal(actual[1], basis), "The prefix write must actually change H"
    assert actual[2]["offers"][0][0] == 1
    # The vector snapshot predates publication. The scalar revalidation must
    # discard it even though both H and output are ordinary float64 ndarrays.
    assert ("basis", True) not in actual[2]["reuse"]


@pytest.mark.parametrize("failure", ["comparison", "basis_publication", "invariant_publication"])
def test_optional_cache_memory_failures_recompute_without_losing_result(failure):
    basis = np.asarray([[.25, .5], [.75, 1.]])
    expected = mu._centered_logs(basis, per_row=False)
    class Cache(dict):
        def __setitem__(self, key, value):
            if failure == "basis_publication" and key == "basis":
                raise MemoryError("optional basis publication")
            if failure == "invariant_publication" and key == "gram":
                raise MemoryError("optional invariant publication")
            super().__setitem__(key, value)
    cache = Cache()
    value = mu._row_log_basis(basis, cache)
    if failure == "comparison":
        with patch.object(mu.np, "array_equal", side_effect=MemoryError("optional comparison")) as compare:
            value = mu._row_log_basis(basis, cache)
        compare.assert_called_once()
    elif failure == "invariant_publication":
        calls = []
        gram = np.eye(2)
        def calculate():
            calls.append(True)
            return gram
        assert mu._row_log_invariant(cache, "gram", calculate) is gram
        assert calls == [True]
    np.testing.assert_array_equal(value[0], expected[0])
    np.testing.assert_array_equal(value[1], expected[1])
    assert not cache
    # Mandatory mathematical errors are never treated as optional allocation failures.
    def required_failure():
        raise RuntimeError("required calculation")
    with pytest.raises(RuntimeError, match="required calculation"):
        mu._row_log_invariant({}, "gram", required_failure)
    def required_allocation():
        raise MemoryError("required calculation allocation")
    with pytest.raises(MemoryError, match="required calculation allocation"):
        mu._row_log_invariant({"basis": object()}, "gram", required_allocation)


def test_cache_revalidates_basis_identity_content_and_layout():
    basis, query = reusable_data()
    cache = {}
    model = fitted(basis, 9, 0.)
    for changed in (model.components_, model.components_.copy(),
                    np.asfortranarray(model.components_ * 1.125)):
        model.components_ = changed
        expected = mu.transform_mu(model, query[:1])
        actual = mu.transform_mu(model, query[:1], _basis_cache=cache)
        np.testing.assert_array_equal(actual, expected)
        assert cache["basis"][0] is changed
        assert not np.shares_memory(cache["basis"][1], changed)
        changed[1, 1] *= 1.125
        np.testing.assert_array_equal(mu.transform_mu(model, query[:1], _basis_cache=cache),
                                      mu.transform_mu(model, query[:1]))


def public_model(basis, cap=9, tol=0.):
    model = NMFClassifier(solver="mu", screening="none", n_components=len(basis),
                          mu_projection="rowwise", n_neighbors=1, max_iter=cap, tol=tol)
    model.decomposition_ = fitted(basis, cap, tol)
    model.mu_projection_ = "rowwise"
    model.selected_features_ = np.arange(basis.shape[1])
    model.n_features_in_ = model.n_features_selected_ = basis.shape[1]
    model.classes_ = np.asarray([0, 1])
    model.decision_threshold_ = None
    model.positive_class_index_ = model.positive_label_ = 1
    model.knn_normalization_ = None
    model.classifier_ = KNeighborsClassifier(n_neighbors=1).fit(
        np.vstack([np.zeros(len(basis)), np.ones(len(basis))]), np.asarray([0, 1]))
    return model


def test_native_observer_distinguishes_classifier_wrapper_from_nmf_body():
    basis, query = reusable_data()
    with observe() as actual:
        result = outcome(lambda: public_model(basis).transform(query))
    assert result[2] is None and not result[1] and actual["native"] == 0
    assert len(actual["scalars"]) == 4
    assert [reused for key, reused in actual["reuse"] if key == "basis"] == [False, True, True, True]
    ordinary = np.asarray([[1., .5], [.5, 1.]])
    with observe() as positive:
        value = fitted(ordinary, 2, 0.).transform(np.ones((1, 2)))
    assert np.isfinite(value).all() and positive["native"] == 1


@pytest.mark.parametrize("strict", [False, True])
def test_public_capped_projection_diagnostics_and_warning_forwarding(stopped_rows, strict):
    basis, fast, slow, _ = stopped_rows
    query = np.vstack([fast, slow, fast])
    def solve(reference):
        model = public_model(basis, 20, 1e-8)
        before = pickle.dumps(model), query.tobytes()
        scope = patch.object(classifier, "transform_mu_rowwise", serial) if reference else nullcontext()
        with scope, observe() as trace, classifier._capture_projection(model) as diagnostics:
            result = outcome(lambda: model.transform(query), strict=strict)
        diagnostics = {key: value for key, value in diagnostics.items() if key != "last_transform_seconds"}
        assert diagnostics["last_transform_performed"]
        assert diagnostics["last_transform_warnings"]
        assert (pickle.dumps(model), query.tobytes()) == before
        if not reference:
            assert trace["offers"][0][0] == 1
        return result, diagnostics
    expected, actual = solve(True), solve(False)
    same(actual[0], expected[0], error=ConvergenceWarning if strict else None)
    assert actual[1] == expected[1]


@pytest.mark.parametrize("first_cap", [None, 16])
def test_actual_fit_projection_and_persistence_use_recovery_reuse(first_cap):
    basis, fast, slow, _ = boundary_data(2049)
    training = np.repeat(fast[None, :], 33, axis=0)
    if first_cap is not None:
        training[first_cap] = slow
    labels, query = np.asarray([0] * 16 + [1] * 17), np.vstack([fast, slow])
    before = training.tobytes(), labels.tobytes(), basis.tobytes()
    native_transform = NMF.transform
    def exercise(reference):
        injected = []
        def controlled_fit(self, values, y=None, **kwargs):
            np.testing.assert_array_equal(values, training)
            self.components_ = basis.copy()
            self.n_components_ = self._n_components = 2
            self.n_features_in_ = 2049
            self._beta_loss = 2
            self.n_iter_ = 1
            self.reconstruction_err_ = 0.
            injected.append(True)
            return np.full((33, 2), .25)
        model = NMFClassifier(solver="mu", screening="none", n_components=2,
            training_coordinates="transform", mu_projection="rowwise", classifier="knn",
            n_neighbors=1, weights="uniform", knn_normalization="none", max_iter=20,
            tol=1e-8, random_state=0)
        route = patch.object(classifier, "transform_mu_rowwise", serial) if reference else nullcontext()
        with patch.object(NMF, "fit_transform", controlled_fit), route, observe() as trace:
            fitted_result = outcome(lambda: model.fit(training, labels))
        assert fitted_result[2] is None and injected == [True]
        assert NMF.transform is native_transform is mu._NMF_TRANSFORM
        assert model.training_coordinates_mode_ == "transform" and model.training_projection_performed_
        assert model.training_projection_converged_ is (first_cap is None)
        np.testing.assert_array_equal(model.classifier_._fit_X, model.training_coordinates_)
        np.testing.assert_array_equal(model.decomposition_.components_, basis)
        residual = np.linalg.norm(training - model.training_coordinates_ @ basis)
        np.testing.assert_allclose(model.reconstruction_err_, residual, rtol=2e-13)
        if not reference:
            assert trace["vectors"][0]["count"] == (33 if first_cap is None else 16)
            assert ((33, 2049), (2, 2049), 0, 1) in trace["panels"]
            assert ((33, 2049), (2, 2049), 1, 2) in trace["panels"]
            assert trace["native"] == 0
            if first_cap is None:
                assert not trace["offers"] and not trace["scalars"]
            else:
                assert trace["offers"] == [(16, trace["scalars"][0][0])]
                for key in ("basis", "gram", "first_denominator"):
                    flags = [hit for name, hit in trace["reuse"] if name == key]
                    assert flags == [True, False] + [True] * 15
        saved = pickle.dumps(model)
        with route:
            projected = outcome(lambda: model.transform(query))
            assert projected[2] is None and pickle.dumps(model) == saved
            joint = outcome(lambda: model.predict_with_probabilities(query))
        assert joint[2] is None
        np.testing.assert_array_equal(joint[0][0], model.classifier_.predict(projected[0]))
        np.testing.assert_array_equal(joint[0][1], model.classifier_.predict_proba(projected[0]))
        # Query counters inside native KDTree may change; learned arrays and
        # parameters must not. Pickle/copy preserve the fitted recovery policy.
        for restored in (pickle.loads(saved), deepcopy(model)):
            assert restored.get_params() == model.get_params()
            np.testing.assert_array_equal(restored.training_coordinates_, model.training_coordinates_)
            np.testing.assert_array_equal(restored.classifier_._fit_X, model.classifier_._fit_X)
            again = outcome(lambda: restored.predict_with_probabilities(query))
            assert again[1:] == joint[1:]
            for got, wanted in zip(again[0], joint[0]):
                np.testing.assert_array_equal(got, wanted)
        return model, fitted_result[1], projected, joint
    expected, actual = exercise(True), exercise(False)
    np.testing.assert_array_equal(actual[0].training_coordinates_, expected[0].training_coordinates_)
    assert actual[1] == expected[1]
    same(actual[2], expected[2])
    assert actual[3][1:] == expected[3][1:]
    for got, wanted in zip(actual[3][0], expected[3][0]):
        np.testing.assert_array_equal(got, wanted)
    assert (training.tobytes(), labels.tobytes(), basis.tobytes()) == before


def test_same_model_concurrent_public_calls_have_owned_cache():
    """Overlapping numerical work with LIFO exits of Python's warning contexts."""
    basis, query = reusable_data()
    model = public_model(basis)
    before = pickle.dumps(model), query.tobytes()
    expected = outcome(lambda: serial(fitted(basis, 9, 0.), query))
    assert expected[2] is None and not expected[1] and np.isfinite(expected[0]).all()
    enter, exit_ = warnings.catch_warnings.__enter__.__code__, warnings.catch_warnings.__exit__.__code__
    record = classifier._record_warnings.__code__
    first_basis = mu._row_log_basis.__code__
    lock, barrier, inner_done = Lock(), Barrier(2), Event()
    order, exits = [], []
    warning_state = (warnings.filters, list(warnings.filters), warnings.showwarning, warnings._showwarnmsg_impl)
    def solve():
        tid, met = get_ident(), False
        pools = threadpool_info()
        assert pools and all(pool["num_threads"] == 1 for pool in pools)
        errors, callback = np.geterr(), np.geterrcall()
        def hook(frame, event, value):
            nonlocal met
            if frame.f_back is not None and frame.f_back.f_code is record:
                if frame.f_code is enter and event == "call":
                    lock.acquire()
                elif frame.f_code is enter and event == "return":
                    order.append(tid)
                    lock.release()
                elif frame.f_code is exit_ and event == "call":
                    if tid == order[0]:
                        assert inner_done.wait(timeout=30)
                elif frame.f_code is exit_ and event == "return":
                    exits.append(tid)
            if frame.f_code is first_basis and event == "call" and not met:
                met = True
                barrier.wait(timeout=30)
        with observe(hook=hook) as trace:
            try:
                value = model.transform(query.copy())
            finally:
                if len(order) == 2 and tid == order[1]:
                    inner_done.set()
        assert met and trace["native"] == 0 and not trace["offers"]
        assert not trace["warnings"] and len(trace["scalars"]) == 4
        contexts = {context for context, _ in trace["scalars"]}
        assert len(contexts) == 1 and None not in contexts
        for key in ("basis", "gram", "first_denominator"):
            assert [hit for name, hit in trace["reuse"] if name == key] == [False, True, True, True]
        assert np.geterr() == errors and np.geterrcall() is callback
        assert sorted(threadpool_info(), key=lambda p: p["filepath"]) == sorted(pools, key=lambda p: p["filepath"])
        return value, contexts.pop()
    def worker():
        # Some native limits belong to the calling thread. Establish the test
        # budget there too, and restore that worker's original native state.
        before_pools = threadpool_info()
        with threadpool_limits(limits=1):
            result = solve()
        assert sorted(threadpool_info(), key=lambda p: p["filepath"]) == sorted(
            before_pools, key=lambda p: p["filepath"])
        return result
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        results = [future.result(timeout=60) for future in futures]
    assert exits == list(reversed(order)) and len(set(order)) == 2
    assert results[0][1] != results[1][1]
    for value, _ in results:
        np.testing.assert_array_equal(value, expected[0])
    assert warnings.filters is warning_state[0] and warnings.filters == warning_state[1]
    assert (warnings.showwarning, warnings._showwarnmsg_impl) == warning_state[2:]
    assert (pickle.dumps(model), query.tobytes()) == before
