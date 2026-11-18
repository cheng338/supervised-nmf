"""Public numerical and ownership contracts of the optional C-layout reuse.

The retained original route is the exact-parity oracle. Route observations
make sure an optimized/fallback comparison actually exercises its intended
domain; independent assertions cover the public behavior being preserved.
"""

from contextlib import contextmanager
import importlib
from types import FunctionType
from unittest.mock import patch

import numpy as np
import pytest
from sklearn import config_context, get_config
from threadpoolctl import threadpool_limits

from supervised_nmf import BinaryScreening, NMFClassifier
from supervised_nmf import screening as screening_module


_READY_NAMES = {
    "mean": "_SCREENING_MEAN_READY",
    "normalize": "_SCREENING_NORMALIZE_READY",
}
_SUPPORTED = {
    policy: getattr(screening_module, name).supports_policy
    for policy, name in _READY_NAMES.items()
}


@pytest.fixture(autouse=True)
def controlled_numerical_policy():
    with threadpool_limits(limits=1), np.errstate(under="ignore"):
        with config_context(assume_finite=False):
            yield


def data(shape=(1024, 128), dtype=np.float64):
    rng = np.random.default_rng(93471)
    y = np.arange(shape[0]) % 2
    X = rng.uniform(.5, 2., shape).astype(dtype)
    X[y == 1, :min(8, shape[1])] += .6
    return X, y


@contextmanager
def original_route():
    with patch.object(screening_module, _READY_NAMES["mean"], return_value=False):
        with patch.object(screening_module, _READY_NAMES["normalize"], return_value=False):
            yield


def observe_routes(monkeypatch):
    observed = {policy: [] for policy in _READY_NAMES}

    def wrap(policy, ready):
        def observed_ready(X, scratch):
            admitted = ready(X, scratch)
            observed[policy].append((admitted, X, scratch))
            return admitted
        return observed_ready

    for policy, name in _READY_NAMES.items():
        monkeypatch.setattr(screening_module, name,
                            wrap(policy, getattr(screening_module, name)))
    return observed


def assert_same_screen(actual, expected):
    actual_state = {k: v for k, v in vars(actual).items() if k.endswith("_")}
    expected_state = {k: v for k, v in vars(expected).items() if k.endswith("_")}
    assert actual_state.keys() == expected_state.keys()
    for name, value in actual_state.items():
        reference = expected_state[name]
        if isinstance(value, np.ndarray):
            assert value.dtype == reference.dtype
            np.testing.assert_array_equal(value, reference, err_msg=name)
            if np.issubdtype(value.dtype, np.floating):
                np.testing.assert_array_equal(np.signbit(value), np.signbit(reference),
                                              err_msg=name + " sign bits")
        else:
            assert value == reference, name


def fit_pair(X, y, **options):
    settings = dict(score_threshold=0.)
    settings.update(options)
    with original_route():
        expected = BinaryScreening(**settings).fit(X, y)
    actual = BinaryScreening(**settings).fit(X, y)
    assert_same_screen(actual, expected)
    return actual


@pytest.mark.parametrize("shape,mean_admitted,normalization_admitted", [
    ((131071, 1), False, False),
    ((131072, 1), True, True),
    ((131073, 1), True, True),
    ((1024, 128), True, True),
    ((8191, 2), False, False),
    ((8192, 2), True, False),
    ((8193, 2), True, False),
    ((8192, 3), False, False),
    ((65536, 2), True, True),
])
def test_size_boundaries_preserve_statistics_and_public_transform(
        monkeypatch, shape, mean_admitted, normalization_admitted):
    X, y = data(shape)
    before = X.copy()
    observed = observe_routes(monkeypatch)
    actual = fit_pair(X, y)
    assert [row[0] for row in observed["mean"]] == [mean_admitted and _SUPPORTED["mean"]]
    assert [row[0] for row in observed["normalize"]] == [
        normalization_admitted and _SUPPORTED["normalize"]]
    np.testing.assert_array_equal(X, before)
    selected = actual.transform(X[:5])
    np.testing.assert_array_equal(selected, before[:5, actual.get_support()])
    assert not np.shares_memory(selected, X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("readonly", [False, True])
def test_owned_normalization_preserves_caller_constants_and_positive_zero(
        monkeypatch, dtype, readonly):
    X, y = data(dtype=dtype)
    X[:, 0] = np.where(y, -0., 0.)
    X[:, 1] = 3.
    X[:, 2] = 1. + y
    before = X.copy()
    X.flags.writeable = not readonly
    observed = observe_routes(monkeypatch)
    actual = fit_pair(X, y, screening="percentile", percentile=75.)
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(np.signbit(X), np.signbit(before))
    assert X.flags.writeable is (not readonly)
    np.testing.assert_array_equal(actual.constant_features_, [0, 1])
    assert np.isposinf(actual.t_scores_[2])
    assert not actual.get_support()[0] and not actual.get_support()[1]
    assert actual.get_support()[2]
    scratch = observed["mean"][0][2]
    assert scratch.dtype == np.float64 and scratch.flags.owndata
    assert scratch.base is None and not np.shares_memory(scratch, X)
    np.testing.assert_array_equal(scratch[:, :2], 0.)
    assert not np.signbit(scratch[:, :2]).any()
    if _SUPPORTED["normalize"]:
        assert observed["normalize"][0][0] is True
        assert observed["normalize"][0][2] is scratch
    transformed = actual.transform(X[:9])
    assert transformed.dtype == dtype and not np.shares_memory(transformed, X)
    np.testing.assert_array_equal(transformed, before[:9, actual.get_support()])
    for value in vars(actual).values():
        if isinstance(value, np.ndarray):
            assert not np.shares_memory(value, X)


@pytest.mark.parametrize("layout", ["F", "strided"])
def test_non_c_input_retains_original_result_and_ownership(monkeypatch, layout):
    X, y = data()
    if layout == "F":
        X = np.asfortranarray(X)
    else:
        parent = np.empty((len(X), X.shape[1] * 2))
        parent[:, ::2] = X
        X = parent[:, ::2]
    before = X.copy()
    observed = observe_routes(monkeypatch)
    actual = fit_pair(X, y)
    assert not any(row[0] for rows in observed.values() for row in rows)
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(actual.transform(X), before[:, actual.get_support()])


@pytest.mark.parametrize("reason", ["numpy-before-2", "missing-optional-details"])
def test_unavailable_mean_support_uses_original_public_behavior(monkeypatch, reason):
    original_var = np.var
    with monkeypatch.context() as changes:
        if reason == "numpy-before-2":
            changes.setattr(np, "__version__", "1.26.4")
        else:
            # A valid public function need not expose NumPy's optional wrapper.
            def public_var(*args, **kwargs):
                return original_var(*args, **kwargs)
            changes.setattr(np, "var", public_var)
        unsupported = screening_module._screening_capture_reuse("mean")
    assert unsupported.supports_policy is False
    monkeypatch.setattr(screening_module, _READY_NAMES["mean"], unsupported)
    observed = observe_routes(monkeypatch)
    X, y = data()
    fit_pair(X, y)
    assert [row[0] for row in observed["mean"]] == [False]


@pytest.mark.parametrize("binding", ["zeros_like", "divide", "var"])
def test_changed_public_binding_retains_its_original_call_behavior(monkeypatch, binding):
    X, y = data()
    original = getattr(np, binding)
    calls = []

    def replacement(*args, **kwargs):
        calls.append(True)
        if binding == "var":
            raise AssertionError("The original ndarray.var route does not call np.var")
        return original(*args, **kwargs)

    monkeypatch.setattr(np, binding, replacement)
    observed = observe_routes(monkeypatch)
    fit_pair(X, y)
    assert [row[0] for row in observed["mean"]] == [False]
    if binding == "var":
        assert calls == []
    else:
        assert calls
        assert [row[0] for row in observed["normalize"]] == [False]


def numpy_methods():
    try:
        return importlib.import_module("numpy._core._methods")
    except ImportError:
        return importlib.import_module("numpy.core._methods")


@pytest.mark.parametrize("change", ["var-code", "mean-defaults", "zeros-code"])
def test_changed_function_code_or_defaults_preserves_fallback(monkeypatch, change):
    X, y = data()
    calls = []
    if change == "mean-defaults":
        function = numpy_methods()._mean
        original = function.__defaults__
        assert original
        monkeypatch.setattr(function, "__defaults__", tuple(list(original)))
    else:
        function = (numpy_methods()._var if change == "var-code"
                    else getattr(np.zeros_like, "__wrapped__", None))
        if function is None or function.__closure__:
            pytest.skip("This NumPy lacks the optional directly watched Python body")
        original = FunctionType(function.__code__, function.__globals__,
                                function.__name__, function.__defaults__)
        original.__kwdefaults__ = function.__kwdefaults__
        original_key = "_screening_reuse_test_original"
        calls_key = "_screening_reuse_test_calls"
        assert original_key not in function.__globals__ and calls_key not in function.__globals__
        monkeypatch.setitem(function.__globals__, original_key, original)
        monkeypatch.setitem(function.__globals__, calls_key, calls)
        namespace = {}
        exec("def replacement(*args, **kwargs):\n"
             "    _screening_reuse_test_calls.append(True)\n"
             "    return _screening_reuse_test_original(*args, **kwargs)\n", namespace)
        monkeypatch.setattr(function, "__code__", namespace["replacement"].__code__)
    observed = observe_routes(monkeypatch)
    fit_pair(X, y)
    policy = "normalize" if change == "zeros-code" else "mean"
    assert [row[0] for row in observed[policy]] == [False]
    if change != "mean-defaults":
        assert calls, "The replaced body must still be invoked by the original route"


def test_normalization_allocation_error_is_not_bypassed_and_clears_refit(monkeypatch):
    X, y = data()
    model = BinaryScreening(score_threshold=0.).fit(X, y)
    before = X.copy()
    injected = MemoryError("test normalization allocation")

    def fail(*args, **kwargs):
        raise injected

    with monkeypatch.context() as changes:
        changes.setattr(np, "zeros_like", fail)
        with pytest.raises(MemoryError) as raised:
            model.fit(X, y)
    assert raised.value is injected
    assert not any(name.endswith("_") for name in vars(model))
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("under,assume_finite", [("raise", False), ("warn", False),
                                               ("ignore", True)])
def test_runtime_error_and_validation_policies_keep_original_route(
        monkeypatch, under, assume_finite):
    X, y = data()
    observed = observe_routes(monkeypatch)
    with np.errstate(under=under), config_context(assume_finite=assume_finite):
        errors = np.geterr().copy()
        config = get_config().copy()
        fit_pair(X, y)
        assert np.geterr() == errors and get_config() == config
    assert [row[0] for row in observed["mean"]] == [False]
    assert [row[0] for row in observed["normalize"]] == [False]


def test_strict_underflow_remains_a_public_error_with_no_fitted_state():
    X = np.tile(np.array([0., 1e-200, 1., 1.])[:, None], (256, 128))
    y = np.tile([0, 0, 1, 1], 256)
    before = X.copy()
    model = BinaryScreening(score_threshold=0.)
    with np.errstate(under="raise"):
        with pytest.raises(ValueError, match="Screening statistics exceeded float64 range"):
            model.fit(X, y)
        assert np.geterr()["under"] == "raise"
    assert not any(name.endswith("_") for name in vars(model))
    np.testing.assert_array_equal(X, before)


def test_public_classifier_predictions_and_fitted_state_match_original(monkeypatch):
    X, y = data(dtype=np.float32)
    query = X[:13].copy()
    before = X.copy()
    settings = dict(solver="spa", screening="absolute", score_threshold=0.,
                    n_components=2, n_neighbors=3, random_state=43,
                    projection_workers=1, projection_native_threads=1)
    with original_route():
        expected = NMFClassifier(**settings).fit(X, y)
    observed = observe_routes(monkeypatch)
    actual = NMFClassifier(**settings).fit(X, y)
    assert_same_screen(actual.selector_, expected.selector_)
    for name in ("selected_features_", "selected_components_", "components_",
                 "training_coordinates_"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(actual.predict(query), expected.predict(query))
    np.testing.assert_array_equal(actual.predict_proba(query), expected.predict_proba(query))
    np.testing.assert_array_equal(actual.transform(query), expected.transform(query))
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(query, before[:13])
    assert [row[0] for row in observed["mean"]] == [_SUPPORTED["mean"]]
    assert [row[0] for row in observed["normalize"]] == [_SUPPORTED["normalize"]]
