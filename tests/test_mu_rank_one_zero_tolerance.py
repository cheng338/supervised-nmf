"""Rank-one zero-tolerance recovery avoids unused logarithmic iteration work."""

from fractions import Fraction
import pickle
import warnings

import numpy as np
import pytest
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import supervised_nmf._mu as mu
from supervised_nmf._mu_rowwise import transform_mu_rowwise


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    return request.param


@pytest.fixture(params=[mu.transform_mu, transform_mu_rowwise], ids=["batch", "rowwise"])
def project(request):
    return request.param


def fixed_basis(basis, *, max_iter=37, tol=0):
    model = NMF(n_components=len(basis), solver="mu", init="random",
                random_state=0, max_iter=max_iter, tol=tol)
    model.components_ = basis
    model.n_components_ = len(basis)
    model.n_features_in_ = basis.shape[1]
    model.n_iter_ = 7
    model.reconstruction_err_ = .25
    model._n_components = len(basis)
    model._beta_loss = 2
    return model


def readonly_strided(values):
    backing = np.zeros((len(values), 2 * values.shape[1]), dtype=values.dtype)
    backing[:, ::2] = values
    result = backing[:, ::2]
    result.flags.writeable = False
    return result


def rational(value):
    return Fraction.from_float(float(value))


def rank_one_reference(query, basis):
    """Exact stored-input products, rounded by exact neighboring-float distances.

    Selecting among adjacent dtype values avoids double rounding for float32
    midpoint cases. This oracle does not use the production scaled-dot helper.
    """
    h = [rational(value) for value in basis[0]]
    denominator = sum(value * value for value in h)
    expected = np.zeros((len(query), 1), dtype=query.dtype)
    bits = np.uint32 if query.dtype == np.float32 else np.uint64
    for index, row in enumerate(query):
        exact = sum(rational(value) * component for value, component in zip(row, h)) / denominator
        assert exact <= rational(np.finfo(query.dtype).max)
        with np.errstate(under="ignore"):
            center = query.dtype.type(float(exact))
            neighbors = (center, np.nextafter(center, query.dtype.type(0)),
                         np.nextafter(center, query.dtype.type(np.inf)))
        expected[index, 0] = min(
            (value for value in neighbors if np.isfinite(value)),
            key=lambda value: (abs(rational(value) - exact),
                               int(np.asarray(value).view(bits)) & 1))
    return expected


@pytest.mark.parametrize("underflow", ["ignore", "warn", "raise"])
def test_mixed_subnormal_projection_preserves_values_and_ownership(dtype, project, underflow):
    small = np.finfo(dtype).smallest_subnormal
    values = np.zeros((4, 8), dtype=dtype)
    values[0, 0], values[1, 0], values[3, 1] = small, 1, small
    basis_values = np.zeros((1, 8), dtype=dtype)
    basis_values[0, 0] = .5
    query, basis = readonly_strided(values), readonly_strided(basis_values)
    model = fixed_basis(basis)
    before = pickle.dumps(model)
    expected = rank_one_reference(query, basis)

    # The mixed float64 batch previously failed in unused initial-residual
    # expm1 arithmetic under strict underflow or warning-as-error policies.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(over="raise", invalid="raise", divide="raise", under=underflow):
            actual = project(model, query)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == query.dtype
    assert actual.shape == (len(query), 1)
    assert actual[0, 0] > 0
    assert actual[2, 0] == actual[3, 0] == 0  # Zero row and disjoint support.
    assert not np.shares_memory(actual, query)
    assert not np.shares_memory(actual, basis)
    np.testing.assert_array_equal(query, values)
    np.testing.assert_array_equal(basis, basis_values)
    assert model.components_ is basis
    assert pickle.dumps(model) == before
    assert not query.flags.writeable and not basis.flags.writeable


def test_recovered_rank_one_skips_unused_logarithmic_work(dtype, project, monkeypatch):
    query = np.zeros((1, 8), dtype=dtype)
    query[0, 0] = np.finfo(dtype).smallest_subnormal
    basis = np.zeros((1, 8), dtype=dtype)
    basis[0, 0] = .5
    model = fixed_basis(basis, max_iter=20000)
    expected = rank_one_reference(query, basis)
    original = mu._rank_one_output
    calls = []

    def unused_work(*args, **kwargs):
        raise AssertionError("Rank-one tol=0 recovery must not prepare or iterate logarithmic coordinates")

    def restored(x, h):
        calls.append((x.dtype, h.shape))
        return original(x, h)

    monkeypatch.setattr(mu, "_centered_logs", unused_work)
    monkeypatch.setattr(mu, "_log_error", unused_work)
    monkeypatch.setattr(mu, "_log_cross", unused_work)
    monkeypatch.setattr(mu, "_rank_one_output", restored)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = project(model, query)
    assert calls == [(query.dtype, basis.shape)]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("decisive_term", [False, True], ids=["halfway_zero", "above_halfway"])
def test_zero_tolerance_recovery_preserves_subnormal_rounding(dtype, project, decisive_term):
    small = np.finfo(dtype).smallest_subnormal
    query, basis = np.zeros((1, 8), dtype=dtype), np.zeros((1, 8), dtype=dtype)
    query[0, 0], basis[0, 0] = small, 2
    if decisive_term:
        query[0, 1] = basis[0, 1] = small
    expected = rank_one_reference(query, basis)
    assert expected[0, 0] == (small if decisive_term else 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = project(fixed_basis(basis), query)
    np.testing.assert_array_equal(actual, expected)


def test_true_coordinate_overflow_keeps_the_range_error(dtype, project):
    query, basis = np.zeros((1, 8), dtype=dtype), np.zeros((1, 8), dtype=dtype)
    query[0, 0], basis[0, 0] = 1, np.finfo(dtype).smallest_subnormal
    model = fixed_basis(basis)
    before = pickle.dumps(model)
    query_before, basis_before = query.copy(), basis.copy()
    assert rational(query[0, 0]) / rational(basis[0, 0]) > rational(np.finfo(dtype).max)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            with pytest.raises(FloatingPointError, match="MU coordinates exceed the output dtype range"):
                project(model, query)
    assert pickle.dumps(model) == before
    assert model.components_ is basis
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_array_equal(basis, basis_before)


def test_positive_tolerance_still_computes_residual_and_warns_at_cap(dtype, project, monkeypatch):
    query, basis = np.zeros((1, 8), dtype=dtype), np.zeros((1, 8), dtype=dtype)
    query[0, 0], basis[0, 0] = np.finfo(dtype).tiny, .5
    expected = rank_one_reference(query, basis)
    original_error = mu._log_error
    residual_calls = []

    def observed_error(*args, **kwargs):
        residual_calls.append(True)
        return original_error(*args, **kwargs)

    monkeypatch.setattr(mu, "_log_error", observed_error)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            actual = project(fixed_basis(basis, max_iter=1, tol=1e-4), query)
    assert residual_calls == [True]
    assert len(caught) == 1 and issubclass(caught[0].category, ConvergenceWarning)
    assert "iterations 1 reached" in str(caught[0].message)
    np.testing.assert_array_equal(actual, expected)


def test_higher_rank_zero_tolerance_retains_the_finite_mu_recurrence(dtype, monkeypatch):
    query = np.asarray([[1, 2]], dtype=dtype)
    basis = np.asarray([[1, .5], [.25, 1]], dtype=dtype)
    h = [[rational(value) for value in row] for row in basis]
    x = [rational(value) for value in query[0]]
    gram = [[sum(a * b for a, b in zip(left, right)) for right in h] for left in h]
    numerator = [sum(a * b for a, b in zip(x, row)) for row in h]
    # Uniform initialization cancels from the first update; use exact scalar
    # rationals for all three finite updates, not an NNLS optimum as reference.
    w = [numerator[j] / sum(row[j] for row in gram) for j in range(len(h))]
    for _ in range(2):
        denominator = [sum(w[k] * gram[k][j] for k in range(len(h))) for j in range(len(h))]
        w = [w[j] * numerator[j] / denominator[j] for j in range(len(h))]
    expected = np.asarray([[float(value) for value in w]], dtype=dtype)

    def forbidden_rank_one(*args, **kwargs):
        raise AssertionError("Rank>1 must retain the logarithmic MU recurrence")

    monkeypatch.setattr(mu, "_rank_one_output", forbidden_rank_one)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = mu._project_log_mu(query, basis, max_iter=3, tol=0)
    np.testing.assert_allclose(actual, expected, rtol=2e-6 if dtype == np.float32 else 2e-12, atol=0)
