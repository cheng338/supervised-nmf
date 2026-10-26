"""Basis-fitting contracts for guarded SPA NNLS.

The disjoint-support fixtures have known factors independent of either solver.
The other cases retain the numerical-domain and failed-refit behavior of the
public estimator while exercising more than 32 feature columns.
"""

import pickle

import numpy as np
import pytest
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from supervised_nmf import FeatureSPA
from supervised_nmf import _nnls, _projection


@pytest.fixture(autouse=True)
def one_numerical_thread():
    with threadpool_limits(limits=1):
        yield


def disjoint_factors(rank):
    # Disjoint supports make the exact coefficients identifiable. Dyadic
    # values also keep this fixture exact when the input is cast to float32.
    scales = np.exp2(np.arange(rank) % 4)
    coordinates = np.vstack([np.diag(scales * value) for value in (1, 2, 4)])
    rng = np.random.default_rng(684)
    mixing = np.column_stack([
        np.eye(rank), rng.integers(1, 9, size=(rank, 40)) / 16,
        np.zeros((rank, 1)),
    ])
    return coordinates, mixing


def input_layout(values, dtype, layout):
    if layout == "strided":
        owner = np.full((len(values), 2 * values.shape[1]), 17, dtype=dtype)
        array = owner[:, ::2]
        array[:] = values
    else:
        owner = np.array(values, dtype=dtype, order="F" if layout == "F" else "C")
        array = owner
    if layout == "readonly":
        array.setflags(write=False)
    return array, owner


@pytest.mark.parametrize("rank", [4, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_known_basis_and_query_coordinates_preserve_inputs(rank, dtype, layout):
    coordinates, mixing = disjoint_factors(rank)
    X, owner = input_layout(coordinates @ mixing, dtype, layout)
    before = owner.copy()
    writeable = X.flags.writeable

    model = FeatureSPA(rank, nnls_max_iter=20000)
    fitted = model.fit_transform(X)
    assert set(model.anchor_indices_) == set(range(rank))
    expected_basis = mixing[model.anchor_indices_]
    expected_coordinates = coordinates[:, model.anchor_indices_]
    np.testing.assert_array_equal(fitted, expected_coordinates)
    np.testing.assert_allclose(model.components_, expected_basis, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(model.components_[:, -1], np.zeros(rank))
    np.testing.assert_allclose(fitted @ model.components_, X, rtol=1e-10, atol=1e-12)
    assert model.components_.dtype == np.float64
    assert model.components_.flags.c_contiguous
    assert not np.shares_memory(fitted, model.fit_coordinates_)

    rng = np.random.default_rng(962)
    incoming_coordinates = rng.integers(1, 17, size=(7, rank)) / 8
    query = incoming_coordinates @ mixing
    query_before, basis_before = query.copy(), model.components_.copy()
    actual = model.transform(query)
    np.testing.assert_allclose(
        actual, incoming_coordinates[:, model.anchor_indices_], rtol=1e-9, atol=1e-11
    )
    np.testing.assert_array_equal(query, query_before)
    np.testing.assert_array_equal(model.components_, basis_before)
    np.testing.assert_array_equal(owner, before)
    assert X.flags.writeable == writeable


def extreme_range_data():
    anchors = np.eye(4) + (1 - np.eye(4)) * 1e-300
    rng = np.random.default_rng(71025)
    return np.column_stack([anchors, anchors @ rng.uniform(.2, 1.2, size=(4, 64))])


def test_basis_fit_preserves_public_solver_domain_for_wide_exponent_span():
    X = extreme_range_data()
    original = X.copy()
    X.setflags(write=False)
    # This actual numerical-range rejection is why direct helper reuse alone
    # would break an input accepted by the original basis-fitting operation.
    with pytest.raises(FloatingPointError, match="exponent span"):
        _nnls.project_nnls(X[:, :4], X.T)

    model = FeatureSPA(4, nnls_max_iter=20000).fit(X)
    reference = np.column_stack([
        nnls(model.fit_coordinates_, X[:, column], maxiter=20000)[0]
        for column in range(X.shape[1])
    ])
    assert set(model.anchor_indices_) == set(range(4))
    np.testing.assert_array_equal(model.components_, reference)
    np.testing.assert_allclose(
        model.components_[:, model.anchor_indices_], np.eye(4), rtol=0, atol=1e-14
    )
    assert model.components_.flags.c_contiguous
    np.testing.assert_array_equal(X, original)
    assert not X.flags.writeable


def test_ill_conditioned_basis_uses_public_solver_without_normal_equations(monkeypatch):
    rng = np.random.default_rng(912)
    first = np.ones(80)
    anchors = np.column_stack([first, first + np.linspace(0, 1e-4, len(first))])
    mixing = np.column_stack([np.eye(2), rng.uniform(.1, .8, size=(2, 40))])
    X = anchors @ mixing
    before = X.copy()

    def unexpected_batch(*args, **kwargs):
        raise AssertionError("ill-conditioned basis must use public NNLS")

    monkeypatch.setattr(_nnls, "_batched_block", unexpected_batch)
    model = FeatureSPA(2, nnls_max_iter=20000).fit(X)
    assert model.original_anchor_condition_ > 1e4
    reference = np.column_stack([
        nnls(model.fit_coordinates_, column, maxiter=20000)[0] for column in X.T
    ])
    np.testing.assert_array_equal(model.components_, reference)
    np.testing.assert_array_equal(X, before)


def test_projection_execution_settings_do_not_change_basis_fit_or_create_workers(monkeypatch):
    rng = np.random.default_rng(1129)
    X = rng.gamma(2, 1, size=(80, 48))

    def unexpected_pool(*args, **kwargs):
        raise AssertionError("basis fitting must stay in the calling process")

    monkeypatch.setattr(_projection, "Parallel", unexpected_pool)
    reference = FeatureSPA(4).fit(X)
    for solver, workers, batch_size, native_threads in (
        ("scipy", 2, 7, 1), ("batched", 3, 1, 2), ("auto", 1, 512, 1)
    ):
        model = FeatureSPA(
            4, projection_solver=solver, projection_workers=workers,
            projection_batch_size=batch_size, projection_native_threads=native_threads
        ).fit(X)
        np.testing.assert_array_equal(model.anchor_indices_, reference.anchor_indices_)
        np.testing.assert_array_equal(model.fit_coordinates_, reference.fit_coordinates_)
        np.testing.assert_array_equal(model.components_, reference.components_)


@pytest.mark.parametrize("extreme_range", [False, True])
def test_iteration_failure_during_refit_clears_old_fitted_state(extreme_range):
    X = (extreme_range_data() if extreme_range else
         np.column_stack([np.eye(4), np.ones((4, 40))]))
    model = FeatureSPA(4, nnls_max_iter=20000).fit(X)
    before = X.copy()
    model.set_params(nnls_max_iter=1)
    with pytest.raises(RuntimeError, match="iterations"):
        model.fit(X)
    with pytest.raises(NotFittedError):
        model.transform(X)
    assert not hasattr(model, "fit_coordinates_")
    np.testing.assert_array_equal(X, before)

    model.set_params(nnls_max_iter=20000).fit(X)
    assert model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.components_).all()


def test_fitted_basis_condition_check_remains_after_accelerated_solve():
    model = FeatureSPA(2, condition_limit=100).fit(np.eye(2))
    # W remains well conditioned; the very large shared mixture makes H
    # ill conditioned. The check on H is separate from the batching guard.
    X = np.column_stack([np.eye(2), np.full((2, 40), 1e4)])
    with pytest.raises(ValueError, match="Fitted SPA basis"):
        model.fit(X)
    with pytest.raises(NotFittedError):
        model.transform(X)


def test_pickle_clone_and_refit_with_accelerated_basis():
    rng = np.random.default_rng(421)
    X = rng.gamma(2, 1, size=(72, 48))
    query = rng.gamma(2, 1, size=(11, 48))
    model = FeatureSPA(4, projection_solver="scipy", nnls_max_iter=20000).fit(X)
    expected = model.transform(query)
    basis_before = model.components_.copy()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.components_, basis_before)
    np.testing.assert_array_equal(restored.transform(query), expected)
    copied = clone(restored)
    assert copied.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        copied.transform(query)

    reversed_features = X[:, ::-1].copy()
    copied.fit(reversed_features)
    restored.fit(reversed_features)
    np.testing.assert_array_equal(restored.anchor_indices_, copied.anchor_indices_)
    np.testing.assert_array_equal(restored.components_, copied.components_)
    np.testing.assert_array_equal(model.components_, basis_before)
