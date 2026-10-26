"""Precision recovery keeps fixed-basis MU fast without losing tiny solutions."""

from copy import deepcopy
from fractions import Fraction
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
    """Supply a controlled fitted dictionary, independently of joint training."""
    rank, features = basis.shape
    model = NMF(n_components=rank, solver="mu", init="random", random_state=0,
                max_iter=max_iter, tol=tol)
    model.components_ = basis.copy()
    model.n_components_ = rank
    model.n_features_in_ = features
    model.n_iter_ = 7
    model.reconstruction_err_ = 0.25
    # These scalar caches are populated during a normal NMF fit. Prepopulate
    # them so input/model ownership checks concern projection, not first setup.
    model._n_components = rank
    model._beta_loss = 2
    return model


def forbid_log(*args, **kwargs):
    raise AssertionError("This fixture must recover through public float64 MU")


def rank_one_ratio(query, basis):
    h = [Fraction.from_float(float(value)) for value in basis[0]]
    x = [Fraction.from_float(float(value)) for value in query[0]]
    return sum(a * b for a, b in zip(x, h)) / sum(b * b for b in h)


@pytest.mark.parametrize("side", ["above_half_ulp", "below_three_half_ulps"])
def test_promoted_rank_one_retains_decisive_subnormal_terms(side, monkeypatch):
    small = np.float32(2.0 ** -149)
    unit = Fraction.from_float(float(small))
    if side == "above_half_ulp":
        # The query's small positive value equals the range-gate threshold.
        # The exact numerator correction lies above a subnormal midpoint but
        # disappears from a naive float64 dot-product ratio.
        basis = np.asarray([[1, 1, small, small]], dtype=np.float32)
        query = np.asarray([[0, 0, 1, 2.0 ** -63]], dtype=np.float32)
        exact = rank_one_ratio(query, basis)
        assert unit / 2 < exact < unit
    else:
        # The exact denominator correction puts the coefficient just below
        # 1.5 ULP. Rounding the double ratio first would incorrectly return 2 ULP.
        basis = np.asarray([[1, small]], dtype=np.float32)
        query = np.asarray([[0, 1.5]], dtype=np.float32)
        exact = rank_one_ratio(query, basis)
        assert unit < exact < 3 * unit / 2
    model = fitted_mu(basis)
    before = pickle.dumps(model)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = mu.transform_mu(model, query)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.asarray([[small]], dtype=np.float32))
    assert pickle.dumps(model) == before


def layout_copy(array, layout):
    if layout == "F":
        return np.asfortranarray(array)
    if layout == "strided":
        backing = np.zeros((len(array), 2 * array.shape[1]), dtype=array.dtype)
        backing[:, ::2] = array
        return backing[:, ::2]
    result = array.copy()
    if layout == "readonly":
        result.flags.writeable = False
    return result


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_tiny_basis_tails_use_fast_projection_without_mutation(layout, monkeypatch):
    small = np.float32(2.0 ** -149)
    basis = layout_copy(np.asarray([[1, 0, small], [0, 1, small]], dtype=np.float32), layout)
    query = layout_copy(np.asarray([[1, 2, 0], [2, 1, 0], [0, 0, 0]], dtype=np.float32), layout)
    model = fitted_mu(basis)
    model.components_ = basis
    before_basis, before_query = basis.copy(), query.copy()
    before_state = pickle.dumps(model)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = mu.transform_mu(model, query)
    # The tiny third coordinate changes the exact minimizer only by O(small**2),
    # far below a float32 ULP in the two positive coordinates.
    expected = np.asarray([[1, 2], [2, 1], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
    assert actual.dtype == query.dtype
    assert np.isfinite(actual).all() and (actual >= 0).all()
    assert not np.shares_memory(actual, query)
    assert not np.shares_memory(actual, basis)
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(basis, before_basis)
    assert model.components_ is basis
    assert pickle.dumps(model) == before_state


def test_actual_float32_update_underflow_recovers_without_log(monkeypatch):
    # The second coordinate decays toward an inactive NNLS component. Around
    # 100 iterations it reaches the float32 underflow range, while the same
    # 200-update recurrence remains well within float64's normal range.
    basis = np.asarray([[1, 1], [1, 4]], dtype=np.float32)
    query = np.asarray([[1, 0]], dtype=np.float32)
    model = fitted_mu(basis, max_iter=200, tol=0)
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError, match="underflow"):
            deepcopy(model).transform(query)
    reference = deepcopy(model)
    reference.components_ = basis.astype(np.float64)
    with np.errstate(all="raise"):
        expected64 = reference.transform(query.astype(np.float64))
    with np.errstate(under="ignore"):
        expected = expected64.astype(np.float32)
    before = pickle.dumps(model)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with np.errstate(all="raise"):
        actual = mu.transform_mu(model, query)
    np.testing.assert_array_equal(actual, expected)
    # Independently, the unique NNLS optimum for this dictionary is (1/2, 0).
    np.testing.assert_allclose(actual, np.asarray([[.5, 0]], dtype=np.float32), rtol=2e-7, atol=0)
    assert pickle.dumps(model) == before


def test_failed_widened_arithmetic_preserves_log_cap_and_warning(monkeypatch):
    small = np.float32(2.0 ** -149)
    model = fitted_mu(np.asarray([[1, small]], dtype=np.float32), max_iter=1, tol=1e-4)
    query = np.asarray([[1, 0]], dtype=np.float32)
    original_log = mu._project_log_mu
    calls = []

    def failed_widening(*args):
        raise FloatingPointError("injected widened arithmetic failure")

    def observed_log(x, h, max_iter, tol):
        calls.append((x.dtype, h.dtype, max_iter, tol))
        return original_log(x, h, max_iter, tol)

    monkeypatch.setattr(mu, "_project_float64_mu", failed_widening)
    monkeypatch.setattr(mu, "_project_log_mu", observed_log)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="raise"):
            actual = mu.transform_mu(model, query)
    assert calls == [(np.dtype("float32"), np.dtype("float32"), 1, 1e-4)]
    assert len(caught) == 1
    assert issubclass(caught[0].category, ConvergenceWarning)
    assert "iterations 1 reached" in str(caught[0].message)
    np.testing.assert_array_equal(actual, np.ones((1, 1), dtype=np.float32))


def test_successful_widened_solve_retains_iteration_limit_warning(monkeypatch):
    small = np.float32(2.0 ** -149)
    model = fitted_mu(np.asarray([[1, 0, small], [0, 1, small]], dtype=np.float32),
                      max_iter=1, tol=1e-4)
    query = np.asarray([[1, 2, 0]], dtype=np.float32)
    before = pickle.dumps(model)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="raise"):
            actual = mu.transform_mu(model, query)
    assert len(caught) == 1
    assert issubclass(caught[0].category, ConvergenceWarning)
    assert "iterations 1 reached" in str(caught[0].message)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all() and (actual >= 0).all()
    np.testing.assert_allclose(actual, np.asarray([[1, 2]], dtype=np.float32),
                               rtol=2e-7, atol=0)
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, MemoryError])
def test_recovery_does_not_swallow_non_numerical_errors(error_type, monkeypatch):
    small = np.float32(2.0 ** -149)
    model = fitted_mu(np.asarray([[1, small]], dtype=np.float32))
    error = error_type("preserve this upstream failure")

    def failed_widening(*args):
        raise error

    monkeypatch.setattr(mu, "_project_float64_mu", failed_widening)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    with pytest.raises(error_type) as raised:
        mu.transform_mu(model, np.asarray([[1, 0]], dtype=np.float32))
    assert raised.value is error


def test_widened_output_outside_float32_range_still_raises(monkeypatch):
    small = np.float32(2.0 ** -149)
    model = fitted_mu(np.asarray([[small, 0]], dtype=np.float32))
    query = np.asarray([[1, 0]], dtype=np.float32)
    original_log = mu._project_log_mu
    calls = []

    def observed_log(x, h, max_iter, tol):
        calls.append((x.dtype, h.dtype, max_iter, tol))
        return original_log(x, h, max_iter, tol)

    monkeypatch.setattr(mu, "_project_log_mu", observed_log)
    before = pickle.dumps(model)
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError, match="output dtype"):
            mu.transform_mu(model, query)
    assert calls == [(np.dtype("float32"), np.dtype("float32"), model.max_iter, model.tol)]
    assert pickle.dumps(model) == before
    np.testing.assert_array_equal(query, np.asarray([[1, 0]], dtype=np.float32))


def test_precision_recovery_adds_no_persistent_state_to_pickled_model(monkeypatch):
    small = np.float32(2.0 ** -149)
    model = fitted_mu(np.asarray([[1, 0, small], [0, 1, small]], dtype=np.float32))
    query = np.asarray([[1, 2, 0], [2, 1, 0]], dtype=np.float32)
    before = pickle.dumps(model)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    expected = mu.transform_mu(model, query)
    assert pickle.dumps(model) == before
    restored = pickle.loads(before)
    restored_before = pickle.dumps(restored)
    actual = mu.transform_mu(restored, query)
    np.testing.assert_array_equal(actual, expected)
    assert restored.components_.dtype == np.float32
    assert set(vars(restored)) == set(vars(model))
    assert pickle.dumps(restored) == restored_before


def test_explicit_training_projection_recovers_precision_before_fitting_head(monkeypatch):
    small = np.float32(2.0 ** -149)
    basis = np.asarray([[1, 0, small], [0, 1, small]], dtype=np.float32)
    training = np.asarray([[1, 2, 0], [1, 3, 0], [2, 3, 0],
                           [3, 1, 0], [3, 2, 0], [4, 1, 0]], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    query = np.asarray([[1.25, 2.25, 0], [3.75, 1.25, 0]], dtype=np.float32)
    before_training, before_query, before_basis = training.copy(), query.copy(), basis.copy()

    def controlled_fit(self, values, y=None, **kwargs):
        self.components_ = basis.copy()
        self.n_components_ = 2
        self.n_features_in_ = values.shape[1]
        self.n_iter_ = 1
        self.reconstruction_err_ = 0.0
        # Deliberately unsuitable for classification. The explicit training
        # projection must replace these joint-fit coordinates before head fit.
        return np.full((len(values), 2), .25, dtype=values.dtype)

    monkeypatch.setattr(NMF, "fit_transform", controlled_fit)
    monkeypatch.setattr(mu, "_project_log_mu", forbid_log)
    model = NMFClassifier(solver="mu", screening="none", n_components=2,
                          training_coordinates="transform", classifier="knn",
                          n_neighbors=1, weights="uniform", max_iter=100, tol=1e-4,
                          random_state=0, projection_native_threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(training, labels)
        actual_query = model.transform(query)
        predicted, probabilities = model.predict_with_probabilities(query)
    # This fixed dictionary's first two coordinates give its NNLS solution up
    # to O(small**2), below float32 resolution for every positive fixture value.
    oracle_train, oracle_query = training[:, :2], query[:, :2]
    np.testing.assert_allclose(model.training_coordinates_, oracle_train, rtol=2e-7, atol=0)
    np.testing.assert_allclose(actual_query, oracle_query, rtol=2e-7, atol=0)
    assert model.training_coordinates_.dtype == np.float32
    assert model.training_coordinates_mode_ == "transform"
    assert model.training_projection_performed_ is True
    assert model.training_projection_converged_ is True
    assert model.training_projection_convergence_warnings_ == ()
    np.testing.assert_array_equal(model.classifier_.predict(oracle_query), np.asarray([0, 1]))
    np.testing.assert_array_equal(predicted, np.asarray([0, 1]))
    np.testing.assert_array_equal(probabilities, np.eye(2))
    np.testing.assert_array_equal(probabilities, model.classifier_.predict_proba(oracle_query))
    np.testing.assert_array_equal(model.decomposition_.components_, before_basis)
    np.testing.assert_array_equal(model.selected_components_, before_basis)
    np.testing.assert_array_equal(training, before_training)
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(basis, before_basis)
