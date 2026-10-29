"""Float64 MU range recovery preserves tiny components and fitted state."""

from fractions import Fraction
import math
import pickle
import warnings

import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
import supervised_nmf._mu as mu


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


def fitted_mu(basis, *, max_iter=100, tol=1e-4):
    """A controlled fixed dictionary, independent of joint factor estimation."""
    rank, features = basis.shape
    model = NMF(n_components=rank, solver="mu", init="random", random_state=0,
                max_iter=max_iter, tol=tol)
    model.components_ = basis.copy()
    model.n_components_ = model._n_components = rank
    model.n_features_in_ = features
    model._beta_loss = 2
    model.n_iter_ = 7
    model.reconstruction_err_ = .25
    return model


def layout_copy(array, layout):
    if layout == "F":
        return np.asfortranarray(array)
    if layout == "strided":
        backing = np.zeros((len(array), 3 * array.shape[1]), dtype=array.dtype)
        backing[:, ::3] = array
        return backing[:, ::3]
    result = array.copy()
    if layout == "readonly":
        result.flags.writeable = False
    return result


def forbid_log(*args, **kwargs):
    raise AssertionError("The admitted fixture should use scaled float64 arithmetic.")


def forbid_scaled(*args, **kwargs):
    raise AssertionError("This input must retain its existing projection route.")


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_harmless_basis_tails_use_scaling_without_mutation(layout, monkeypatch):
    basis = layout_copy(np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]]), layout)
    query = layout_copy(np.asarray([[1., 2., 0.], [2., 1., 0.], [0., 0., 0.]]), layout)
    model = fitted_mu(basis)
    model.components_ = basis
    state = pickle.dumps(model)
    before_basis, before_query = basis.copy(), query.copy()
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = mu.transform_mu(model, query)
    # The extra feature changes the minimizer only by O(1e-360), below an ULP
    # of the positive coefficients. Its presence must not change exact zeros.
    expected = np.asarray([[1., 2.], [2., 1.], [0., 0.]])
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, query)
    assert not np.shares_memory(actual, basis)
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(basis, before_basis)
    assert model.components_ is basis
    assert pickle.dumps(model) == state


def test_essential_tiny_component_retains_large_representable_coordinates(monkeypatch):
    small = math.ldexp(1., -600)
    basis = np.diag([1., small])
    query = np.asarray([[1., 1.], [2., .5]])
    # An orthogonal basis has the independently known solution X / diag(H).
    # Losing the tiny squared Gram entry would destroy this large coefficient.
    expected = np.asarray([[1., math.ldexp(1., 600)], [2., math.ldexp(1., 599)]])
    model = fitted_mu(basis)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = mu.transform_mu(model, query)
    assert np.isfinite(actual).all() and (actual > 0).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)


def test_rank_one_scaling_retains_exact_rational_restoration(monkeypatch):
    small = math.ldexp(1., -600)
    basis = np.asarray([[1., small]])
    query = np.asarray([[1., 0.], [0., 1.]])
    h = [Fraction.from_float(float(v)) for v in basis[0]]
    denominator = sum(v * v for v in h)
    expected = np.asarray([
        [float(sum(Fraction.from_float(float(x)) * b for x, b in zip(row, h)) / denominator)]
        for row in query
    ])
    original = mu._rank_one_output
    calls = []

    def observed(x, h):
        calls.append((x.copy(), h.copy()))
        return original(x, h)

    monkeypatch.setattr(mu, "_rank_one_output", observed)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = mu.transform_mu(fitted_mu(basis), query)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], query)
    np.testing.assert_array_equal(calls[0][1], basis)
    np.testing.assert_array_equal(actual, expected)
    assert actual[1, 0] > 0


@pytest.mark.parametrize("domain", ["ordinary_float64", "float32", "tiny_query", "zero_query"])
def test_unaffected_domains_do_not_attempt_scaled_float64(domain, monkeypatch):
    basis = np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]])
    query = np.asarray([[1., 2., 0.], [2., 1., 0.]])
    if domain == "ordinary_float64":
        basis[:, -1] = .1
    elif domain == "float32":
        basis = np.asarray([[1., 0., 1e-30], [0., 1., 1e-30]], dtype=np.float32)
        query = query.astype(np.float32)
    elif domain == "tiny_query":
        query = np.asarray([[math.ldexp(1., -600), 0., 0.]])
    else:
        query = np.zeros_like(query)
    model = fitted_mu(basis)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = mu.transform_mu(model, query)
    monkeypatch.setattr(mu, "_project_scaled_float64_mu", forbid_scaled)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = mu.transform_mu(model, query)
    np.testing.assert_array_equal(actual, expected)


def test_custom_transform_keeps_public_estimator_handling(monkeypatch):
    model = fitted_mu(np.asarray([[1., 1e-180]]))
    query = np.asarray([[1., 0.]])
    marker = np.asarray([[7.]])
    calls = []

    def custom(x):
        calls.append(x)
        return marker

    model.transform = custom
    monkeypatch.setattr(mu, "_project_scaled_float64_mu", forbid_scaled)
    assert mu.transform_mu(model, query) is marker
    assert len(calls) == 1 and calls[0] is query


@pytest.mark.parametrize("kind", ["very_small_tail", "mixed_large_small_components"])
def test_unadmitted_ranges_keep_logarithmic_recovery(kind, monkeypatch):
    if kind == "very_small_tail":
        basis = np.asarray([[1., 0., 1e-300], [0., 1., 1e-300]])
        query = np.asarray([[1., 2., 0.], [2., 1., 0.]])
        expected = query[:, :2]
    else:
        basis = np.diag([1e150, 1e-200])
        query = np.asarray([[1e150, 1.]])
        expected = np.asarray([[1., 1e200]])
    model = fitted_mu(basis)
    original = mu._project_log_mu
    calls = []

    def observed(x, h, max_iter, tol):
        calls.append((x.copy(), h.copy(), max_iter, tol))
        return original(x, h, max_iter, tol)

    monkeypatch.setattr(mu, "_project_log_mu", observed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = mu.transform_mu(model, query)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], query)
    np.testing.assert_array_equal(calls[0][1], basis)
    assert calls[0][2:] == (model.max_iter, model.tol)
    assert np.isfinite(actual).all() and (actual > 0).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


def test_successful_scaled_projection_preserves_iteration_warning(monkeypatch):
    basis = np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]])
    query = np.asarray([[1., 2., 0.]])
    model = fitted_mu(basis, max_iter=1, tol=1e-4)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = mu.transform_mu(model, query)
    assert len(caught) == 1 and issubclass(caught[0].category, ConvergenceWarning)
    assert "iterations 1 reached" in str(caught[0].message)
    np.testing.assert_allclose(actual, query[:, :2], rtol=2e-14, atol=0)


def test_numerical_retry_failure_preserves_original_cap_and_warning(monkeypatch):
    basis = np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]])
    query = np.asarray([[1., 2., 0.]])
    model = fitted_mu(basis, max_iter=1, tol=1e-4)
    original_log = mu._project_log_mu
    calls = []
    with warnings.catch_warnings(record=True) as baseline_warnings:
        warnings.simplefilter("always")
        expected = original_log(query, basis, model.max_iter, model.tol)

    def failed(*args):
        raise FloatingPointError("injected scaled arithmetic failure")

    def observed(x, h, max_iter, tol):
        calls.append((x.copy(), h.copy(), max_iter, tol))
        return original_log(x, h, max_iter, tol)

    monkeypatch.setattr(mu, "_project_scaled_float64_mu", failed)
    monkeypatch.setattr(mu, "_project_log_mu", observed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = mu.transform_mu(model, query)
    assert len(calls) == 1 and calls[0][2:] == (1, 1e-4)
    np.testing.assert_array_equal(calls[0][0], query)
    np.testing.assert_array_equal(calls[0][1], basis)
    np.testing.assert_array_equal(actual, expected)
    assert [(w.category, str(w.message)) for w in caught] == [
        (w.category, str(w.message)) for w in baseline_warnings
    ]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, MemoryError])
def test_scaled_retry_propagates_non_numerical_errors(error_type, monkeypatch):
    model = fitted_mu(np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]]))
    query = np.asarray([[1., 2., 0.]])
    error = error_type("preserve this failure")

    def failed(*args):
        raise error

    monkeypatch.setattr(mu, "_project_scaled_float64_mu", failed)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with pytest.raises(error_type) as raised:
        mu.transform_mu(model, query)
    assert raised.value is error


def test_unrepresentable_output_still_raises_without_state_changes():
    small = np.nextafter(np.float64(0), np.float64(1))
    model = fitted_mu(np.asarray([[small, 0.]]))
    query = np.asarray([[1., 0.]])
    state = pickle.dumps(model)
    with pytest.raises(FloatingPointError, match="output dtype"):
        mu.transform_mu(model, query)
    assert pickle.dumps(model) == state
    np.testing.assert_array_equal(query, np.asarray([[1., 0.]]))


def test_scaled_projection_survives_pickle_without_persistent_cache(monkeypatch):
    model = fitted_mu(np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]]))
    query = np.asarray([[1., 2., 0.], [2., 1., 0.]])
    state = pickle.dumps(model)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    expected = mu.transform_mu(model, query)
    assert pickle.dumps(model) == state
    restored = pickle.loads(state)
    restored_state = pickle.dumps(restored)
    actual = mu.transform_mu(restored, query)
    np.testing.assert_array_equal(actual, expected)
    assert pickle.dumps(restored) == restored_state
    assert set(vars(restored)) == set(vars(model))


def test_explicit_training_projection_uses_scaled_coordinates_before_head_fit(monkeypatch):
    basis = np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]])
    training = np.asarray([[1., 2., 0.], [1., 3., 0.], [2., 3., 0.],
                           [3., 1., 0.], [3., 2., 0.], [4., 1., 0.]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    query = np.asarray([[1.25, 2.25, 0.], [3.75, 1.25, 0.]])

    def controlled_fit(self, values, y=None, **kwargs):
        self.components_ = basis.copy()
        self.n_components_ = 2
        self.n_features_in_ = values.shape[1]
        self.n_iter_ = 1
        self.reconstruction_err_ = 0.
        return np.full((len(values), 2), .25, dtype=values.dtype)

    monkeypatch.setattr(NMF, "fit_transform", controlled_fit)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = NMFClassifier(solver="mu", screening="none", n_components=2,
                              n_neighbors=1, training_coordinates="transform",
                              projection_native_threads=1).fit(training, labels)
        actual = model.transform(query)
        predicted = model.predict(query)
    np.testing.assert_allclose(model.training_coordinates_, training[:, :2], rtol=2e-14, atol=0)
    np.testing.assert_allclose(actual, query[:, :2], rtol=2e-14, atol=0)
    np.testing.assert_array_equal(predicted, np.asarray([0, 1]))
    np.testing.assert_array_equal(model.decomposition_.components_, basis)


@pytest.mark.parametrize("kind", ["shape", "nan", "inf", "negative", "restoration_underflow"])
def test_invalid_scaled_outputs_fall_back_with_original_inputs(kind, monkeypatch):
    basis = np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]])
    query = np.asarray([[1., 2., 0.], [2., 1., 0.]])
    model = fitted_mu(basis, max_iter=1, tol=1e-4)
    state = pickle.dumps(model)
    before_query, before_basis = query.copy(), model.components_.copy()
    original_log, original_copy = mu._project_log_mu, mu.copy
    with warnings.catch_warnings(record=True) as baseline_warnings:
        warnings.simplefilter("always")
        expected = original_log(query, model.components_, model.max_iter, model.tol)
    copied, native_calls, fallback_calls = [], [], []

    def temporary_with_bad_output(value):
        copied.append(value)
        temporary = original_copy(value)

        def fake_transform(x):
            native_calls.append((x.copy(), temporary.components_.copy()))
            shape = (len(x), len(temporary.components_))
            if kind == "shape":
                return np.zeros((shape[0], shape[1] + 1))
            if kind == "restoration_underflow":
                # This is positive and normal before restoration. The required
                # inverse scale for the chosen basis pushes it below float64's
                # representable range; returning zeros would lose information.
                return np.full(shape, math.ldexp(1., -1000))
            return np.full(shape, {"nan": np.nan, "inf": np.inf, "negative": -1.}[kind])

        temporary.transform = fake_transform
        return temporary

    def observed_log(x, h, max_iter, tol):
        fallback_calls.append((x.copy(), h.copy(), max_iter, tol))
        assert x is query
        assert h is model.components_
        return original_log(x, h, max_iter, tol)

    # Keep the estimator class and original instance intact so qualification
    # succeeds; only the temporary retry has the deliberately faulty output.
    monkeypatch.setattr(mu, "copy", temporary_with_bad_output)
    monkeypatch.setattr(mu, "_project_log_mu", observed_log)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = mu.transform_mu(model, query)

    assert len(copied) == 1 and copied[0] is model
    assert len(native_calls) == len(fallback_calls) == 1
    assert not np.array_equal(native_calls[0][0], query)
    assert not np.array_equal(native_calls[0][1], basis)
    np.testing.assert_array_equal(fallback_calls[0][0], query)
    np.testing.assert_array_equal(fallback_calls[0][1], basis)
    assert fallback_calls[0][2:] == (model.max_iter, model.tol)
    np.testing.assert_array_equal(actual, expected)
    assert (actual > 0).all()
    assert [(w.category, str(w.message)) for w in caught] == [
        (w.category, str(w.message)) for w in baseline_warnings
    ]
    assert pickle.dumps(model) == state
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(model.components_, before_basis)


def test_failed_scaled_native_warning_remains_visible_during_recovery(monkeypatch):
    basis = np.asarray([[1., 0., 1e-180], [0., 1., 1e-180]])
    query = np.asarray([[1., 2., 0.]])
    model = fitted_mu(basis, max_iter=1, tol=1e-4)
    state = pickle.dumps(model)
    original_log, original_copy = mu._project_log_mu, mu.copy
    with warnings.catch_warnings(record=True) as baseline_warnings:
        warnings.simplefilter("always")
        expected = original_log(query, model.components_, model.max_iter, model.tol)
    native_calls, fallback_calls = [], []

    def temporary_with_failed_transform(value):
        temporary = original_copy(value)

        def fake_transform(x):
            native_calls.append(x.copy())
            warnings.warn("native retry diagnostic", RuntimeWarning, stacklevel=2)
            raise FloatingPointError("injected native range failure")

        temporary.transform = fake_transform
        return temporary

    def observed_log(x, h, max_iter, tol):
        fallback_calls.append((x.copy(), h.copy(), max_iter, tol))
        return original_log(x, h, max_iter, tol)

    monkeypatch.setattr(mu, "copy", temporary_with_failed_transform)
    monkeypatch.setattr(mu, "_project_log_mu", observed_log)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = mu.transform_mu(model, query)

    assert len(native_calls) == len(fallback_calls) == 1
    np.testing.assert_array_equal(fallback_calls[0][0], query)
    np.testing.assert_array_equal(fallback_calls[0][1], basis)
    assert fallback_calls[0][2:] == (model.max_iter, model.tol)
    np.testing.assert_array_equal(actual, expected)
    assert [(w.category, str(w.message)) for w in caught] == [
        (RuntimeWarning, "native retry diagnostic"),
        *[(w.category, str(w.message)) for w in baseline_warnings],
    ]
    assert pickle.dumps(model) == state
    np.testing.assert_array_equal(query, np.asarray([[1., 2., 0.]]))
    np.testing.assert_array_equal(model.components_, basis)
