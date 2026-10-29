"""Explicit MU projection policies preserve fitted behavior and numerical range."""

from copy import deepcopy
from fractions import Fraction
import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection
from supervised_nmf._mu import transform_mu
from supervised_nmf._mu_rowwise import transform_mu_rowwise
from supervised_nmf.screening import BinaryScreening


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=[np.float32, np.float64], ids=["float32", "float64"])
def data(request):
    rng = np.random.default_rng(831)
    basis = np.full((2, 12), .1)
    basis[0, ::2] += 2
    basis[1, 1::2] += 2
    latent = rng.uniform(.3, 2, (72, 2))
    x = np.column_stack((latent @ basis, np.zeros((72, 2)))).astype(request.param)
    y = np.where(latent[:, 0] > np.median(latent[:, 0]), "positive", "negative")
    query_latent = np.asarray([[.4, .7], [.6, 1.5], [1.8, .8], [1.9, 1.7],
                               [.3, 1.1], [1.7, .4], [.5, 1.8], [1.9, 1.]])
    q = np.column_stack((query_latent @ basis, np.zeros((8, 2)))).astype(request.param)
    return x, y, q


def estimator(**kwargs):
    parameters = dict(solver="mu", screening="none", n_components=2,
                      n_neighbors=3, weights="distance", max_iter=3000,
                      tol=1e-4, random_state=17)
    parameters.update(kwargs)
    return NMFClassifier(**parameters)


def fixed_basis(h, *, max_iter=1000, tol=1e-4):
    """Isolate projection from nonunique joint factorization coordinates."""
    rank, features = h.shape
    model = NMF(n_components=rank, solver="mu", init="random", random_state=0,
                beta_loss="frobenius", max_iter=max_iter, tol=tol)
    model.components_ = h.copy()
    model.n_components_ = model._n_components = rank
    model.n_features_in_ = features
    model._beta_loss = 2
    model.n_iter_ = 7
    model.reconstruction_err_ = .25
    return model


def fitted_with_basis(h):
    rng = np.random.default_rng(937)
    rank, features = h.shape
    scaffold = np.full((rank, features), .2)
    scaffold[np.arange(rank), np.arange(rank)] += 1
    latent = rng.uniform(.2, 2, (24, rank))
    x = (latent @ scaffold).astype(h.dtype)
    y = np.repeat([0, 1], 12)
    model = estimator(n_components=rank, mu_projection="rowwise", max_iter=1,
                      tol=0).fit(x, y)
    model.decomposition_ = fixed_basis(h)
    model.selected_components_ = h.copy()
    model.components_ = h.copy()
    return model


@pytest.mark.parametrize("bad", [None, True, 1, "auto", "single", [], np.asarray(["batch"])])
def test_invalid_policy_fails_before_factorization(data, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid policy reached factorization")

    monkeypatch.setattr(NMF, "fit_transform", forbidden)
    with pytest.raises(ValueError, match="mu_projection"):
        estimator(mu_projection=bad).fit(*data[:2])


@pytest.mark.parametrize("solver", ["cd", "spa"])
def test_rowwise_policy_rejects_non_mu_solver(data, solver):
    with pytest.raises(ValueError, match="mu_projection|rowwise"):
        estimator(solver=solver, mu_projection="rowwise").fit(*data[:2])
    model = estimator(solver=solver).fit(*data[:2])
    assert model.mu_projection_ is None


def test_default_keeps_joint_fit_and_batch_transform(data):
    x, y, q = data
    default = estimator().fit(x, y)
    explicit = estimator(mu_projection="batch").fit(x, y)
    assert default.mu_projection == default.mu_projection_ == "batch"
    assert default.training_coordinates_mode_ == "fit_transform"
    np.testing.assert_array_equal(default.training_coordinates_, explicit.training_coordinates_)
    np.testing.assert_array_equal(default.decomposition_.components_, explicit.decomposition_.components_)
    selected = q[:, default.selected_features_]
    expected = transform_mu(deepcopy(default.decomposition_), selected)
    np.testing.assert_array_equal(default.transform(q), expected)
    np.testing.assert_array_equal(default.predict(q), explicit.predict(q))


@pytest.mark.parametrize("policy", ["batch", "rowwise"])
def test_policy_clone_pickle_and_set_params_take_effect_on_refit(data, policy):
    x, y, q = data
    model = estimator(mu_projection=policy).fit(x, y)
    expected = model.transform(q)
    fitted_basis = model.decomposition_.components_.copy()
    opposite = "rowwise" if policy == "batch" else "batch"
    model.set_params(mu_projection=opposite)
    assert model.mu_projection == opposite and model.mu_projection_ == policy
    np.testing.assert_array_equal(model.transform(q), expected)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.mu_projection_ == policy
    assert restored.mu_projection == opposite
    np.testing.assert_array_equal(restored.transform(q), expected)
    fresh = clone(restored)
    assert fresh.mu_projection == opposite
    assert not hasattr(fresh, "mu_projection_")
    fresh.fit(x, y)
    assert fresh.mu_projection_ == opposite
    np.testing.assert_array_equal(fresh.decomposition_.components_, fitted_basis)
    model.fit(x, y)
    assert model.mu_projection_ == opposite


@pytest.mark.parametrize("keep_constructor", [False, True])
def test_legacy_fitted_state_retains_batch_policy(data, keep_constructor):
    x, y, q = data
    model = estimator().fit(x, y)
    expected = model.transform(q)
    del model.mu_projection_
    if keep_constructor:
        model.mu_projection = "rowwise"
    else:
        del model.mu_projection
    restored = pickle.loads(pickle.dumps(model))
    assert restored.mu_projection_ == "batch"
    assert restored.mu_projection == ("rowwise" if keep_constructor else "batch")
    np.testing.assert_array_equal(restored.transform(q), expected)
    assert clone(restored).get_params() == restored.get_params()


def test_failed_refit_clears_fitted_policy(data):
    x, y, q = data
    model = estimator(mu_projection="rowwise").fit(x, y)
    model.set_params(mu_projection="unsupported")
    with pytest.raises(ValueError, match="mu_projection"):
        model.fit(x, y)
    assert not hasattr(model, "mu_projection_")
    assert not hasattr(model, "classifier_")
    with pytest.raises(NotFittedError):
        model.transform(q)


@pytest.mark.parametrize("head", ["knn", "knn_sd", "logistic", "linear_svm"])
def test_rowwise_batch_consistency_and_head_interfaces(data, head):
    x, y, q = data
    model = estimator(mu_projection="rowwise", classifier="knn" if head == "knn_sd" else head,
                      knn_normalization="sd" if head == "knn_sd" else "none").fit(x, y)
    basis_before = model.decomposition_.components_.copy()
    training_before = model.training_coordinates_.copy()
    full = model.transform(q)
    singles = np.vstack([model.transform(row[None]) for row in q])
    chunks = np.vstack([model.transform(q[:3]), model.transform(q[3:])])
    reverse = model.transform(q[::-1])[::-1]
    assert full.dtype == x.dtype
    for other in (singles, chunks, reverse):
        np.testing.assert_allclose(other, full, rtol=5e-6, atol=1e-12)
        np.testing.assert_array_equal(model.classifier_.predict(other), model.classifier_.predict(full))
    np.testing.assert_array_equal(model.predict(q), model.classifier_.predict(full))
    if head == "linear_svm":
        labels, scores = model.predict_with_decision_function(q)
        np.testing.assert_allclose(scores, model.classifier_.decision_function(full), rtol=1e-6, atol=1e-12)
    else:
        labels, scores = model.predict_with_probabilities(q)
        np.testing.assert_allclose(scores, model.classifier_.predict_proba(full), rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(labels, model.predict(q))
    np.testing.assert_array_equal(model.decomposition_.components_, basis_before)
    np.testing.assert_array_equal(model.training_coordinates_, training_before)
    assert model.mu_projection_ == "rowwise"
    assert model.training_coordinates_mode_ == "fit_transform"


@pytest.mark.parametrize("training_coordinates", ["auto", "fit_transform", "transform"])
def test_explicit_training_projection_uses_same_fitted_policy(data, training_coordinates):
    x, y, _ = data
    model = estimator(mu_projection="rowwise", training_coordinates=training_coordinates).fit(x, y)
    resolved = "fit_transform" if training_coordinates == "auto" else training_coordinates
    assert model.training_coordinates_mode_ == resolved
    assert model.training_projection_performed_ == (resolved == "transform")
    if resolved == "transform":
        np.testing.assert_allclose(model.training_coordinates_, model.transform(x), rtol=5e-6, atol=1e-12)
        assert model.training_projection_converged_ is True
    else:
        baseline = estimator().fit(x, y)
        np.testing.assert_array_equal(model.training_coordinates_, baseline.training_coordinates_)
        assert model.training_projection_converged_ is None


def test_prepared_screening_preserves_policy_and_training_map(data):
    x, y, q = data
    parameters = dict(mu_projection="rowwise", training_coordinates="transform",
                      screening="absolute", score_threshold=0.)
    selector = BinaryScreening(screening="absolute", score_threshold=0.).fit(x, y)
    prepared = estimator(**parameters)._fit_with_screening(x, y, selector)
    ordinary = estimator(**parameters).fit(x, y)
    assert prepared.mu_projection_ == "rowwise"
    np.testing.assert_array_equal(prepared.selected_features_, ordinary.selected_features_)
    np.testing.assert_array_equal(prepared.training_coordinates_, ordinary.training_coordinates_)
    np.testing.assert_array_equal(prepared.transform(q), ordinary.transform(q))
    invalid = q.copy()
    invalid[0, -1] = -1  # A discarded column must still be validated.
    with pytest.raises(ValueError, match="nonnegative"), _capture_projection(prepared) as diagnostic:
        prepared.transform(invalid)
    assert diagnostic["last_transform_converged"] is False
    assert diagnostic["last_transform_error"]


@pytest.mark.parametrize("query_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("basis_dtype", [np.float32, np.float64])
def test_identity_projection_preserves_query_dtype_contract(basis_dtype, query_dtype):
    model = fitted_with_basis(np.eye(2, dtype=basis_dtype))
    small = np.finfo(query_dtype).smallest_subnormal
    query = np.asarray([[small, 1], [2, small], [0, 0], [1e-10, .5]], dtype=query_dtype)
    before_basis = model.decomposition_.components_.copy()
    with _capture_projection(model) as diagnostic:
        actual = model.transform(query)
    expected_dtype = basis_dtype if basis_dtype == query_dtype else np.float64
    assert actual.dtype == expected_dtype
    expected = query.astype(expected_dtype)
    np.testing.assert_allclose(actual, expected, rtol=2e-6 if expected_dtype == np.float32 else 2e-12, atol=0)
    tiny_entries = expected < np.finfo(expected_dtype).tiny
    np.testing.assert_array_equal(actual[tiny_entries], expected[tiny_entries])
    assert np.all(actual[expected > 0] > 0)
    np.testing.assert_array_equal(model.decomposition_.components_, before_basis)
    assert diagnostic["last_transform_converged"] is True


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_rank_one_projection_matches_closed_form_without_mutation(dtype, layout):
    h = np.asarray([[1, 2, 0, 4]], dtype=dtype)
    q = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 3, 0, 0],
                    [0, 0, 7, 0], [1, 2, 3, 4]], dtype=dtype)
    if layout == "F":
        q = np.asfortranarray(q)
    elif layout == "strided":
        backing = np.zeros((len(q), 8), dtype=dtype)
        backing[:, ::2] = q
        q = backing[:, ::2]
    elif layout == "readonly":
        q.flags.writeable = False
    d = fixed_basis(h)
    original = pickle.dumps(d)
    q_before = q.copy()
    expected = ((q.astype(float) @ h.astype(float).T) / 21).astype(dtype)
    actual = transform_mu_rowwise(d, q)
    np.testing.assert_allclose(actual, expected, rtol=2e-6 if dtype == np.float32 else 2e-12, atol=0)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(q, q_before)
    assert pickle.dumps(d) == original


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_rows_and_tiny_diagonal_coordinates(dtype, monkeypatch):
    small = np.finfo(dtype).smallest_subnormal
    d = fixed_basis(np.eye(2, dtype=dtype))

    def no_iteration(*args, **kwargs):
        raise AssertionError("Entirely zero input entered sklearn iterations")

    monkeypatch.setattr(d, "_fit_transform", no_iteration)
    zeros = np.zeros((5, 2), dtype=dtype)
    np.testing.assert_array_equal(transform_mu_rowwise(d, zeros), zeros)
    monkeypatch.undo()
    query = np.asarray([[small, small], [1, small], [0, 0]], dtype=dtype)
    actual = transform_mu_rowwise(d, query)
    singles = np.vstack([transform_mu_rowwise(d, row[None]) for row in query])
    for result in (actual, singles):
        np.testing.assert_allclose(result, query, rtol=2e-6 if dtype == np.float32 else 2e-12, atol=0)
        tiny_entries = query < np.finfo(dtype).tiny
        np.testing.assert_array_equal(result[tiny_entries], query[tiny_entries])
        assert np.all(result[query > 0] > 0)


@pytest.mark.parametrize("side", ["above_half_ulp", "below_three_half_ulps"])
def test_float32_rank_one_preserves_decisive_subnormal_rounding(side):
    small = np.float32(2.0 ** -149)
    if side == "above_half_ulp":
        h = np.asarray([[1, 1, small, small]], dtype=np.float32)
        x = np.asarray([[0, 0, 1, 2.0 ** -63]], dtype=np.float32)
    else:
        h = np.asarray([[1, small]], dtype=np.float32)
        x = np.asarray([[0, 1.5]], dtype=np.float32)
    hx = [Fraction.from_float(float(v)) for v in h[0]]
    xx = [Fraction.from_float(float(v)) for v in x[0]]
    exact = sum(a * b for a, b in zip(xx, hx)) / sum(v * v for v in hx)
    unit = Fraction.from_float(float(small))
    if side == "above_half_ulp":
        assert unit / 2 < exact < unit
    else:
        assert unit < exact < unit * 3 / 2
    actual = transform_mu_rowwise(fixed_basis(h), x)
    np.testing.assert_array_equal(actual, np.asarray([[small]], dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unrepresentable_coordinates_raise_clearly(dtype):
    small = np.finfo(dtype).smallest_subnormal
    d = fixed_basis(np.asarray([[small, 0]], dtype=dtype))
    with pytest.raises(FloatingPointError, match="range|represent"):
        transform_mu_rowwise(d, np.asarray([[1, 0]], dtype=dtype))


def test_iteration_cap_warning_and_zero_tolerance_semantics():
    h = np.asarray([[1, .2, .4], [.3, 1, .6]], dtype=float)
    q = np.asarray([[1, 2, 1], [2, .5, 1]], dtype=float)
    capped = fixed_basis(h, max_iter=1, tol=1e-4)
    with pytest.warns(ConvergenceWarning):
        values = transform_mu_rowwise(capped, q)
    assert np.isfinite(values).all() and (values >= 0).all()
    capped.tol = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fixed = transform_mu_rowwise(capped, q)
    np.testing.assert_array_equal(fixed, values)


@pytest.mark.parametrize("change", ["regularization", "custom_transform", "negative_basis", "dtype"])
def test_changed_projection_domain_is_not_silently_batch_projected(change):
    d = fixed_basis(np.eye(2, dtype=np.float64))
    x = np.ones((3, 2), dtype=np.float64)
    if change == "regularization":
        d.alpha_W = .1
    elif change == "custom_transform":
        d.transform = lambda query: np.ones((len(query), 2))
    elif change == "negative_basis":
        d.components_[0, 0] = -1
    else:
        x = x.astype(np.float32)
    with pytest.raises(ValueError):
        transform_mu_rowwise(d, x)


def test_non_numerical_recovery_failure_propagates(monkeypatch):
    from supervised_nmf import _mu_rowwise

    d = fixed_basis(np.eye(2, dtype=np.float64))
    x = np.asarray([[np.finfo(np.float64).smallest_subnormal, 1.]])

    def failed_recovery(*args, **kwargs):
        raise RuntimeError("recovery failure")

    monkeypatch.setattr(_mu_rowwise, "transform_mu", failed_recovery)
    with pytest.raises(RuntimeError, match="recovery failure"):
        transform_mu_rowwise(d, x)
