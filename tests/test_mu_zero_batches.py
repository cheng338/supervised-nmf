"""Exact-zero MU projection and preservation of the ordinary transform path."""

from copy import deepcopy
import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=[np.float32, np.float64], ids=["float32", "float64"])
def data(request):
    dtype = request.param
    rng = np.random.default_rng(2026102507 + 72 + 35 + 2)
    basis = rng.uniform(.05, .15, (2, 16))
    for component in range(2):
        basis[component, component::2] += 1
    latent = rng.uniform(.1, 2, (72, 2))
    x = (latent @ basis + rng.uniform(0, .01, (72, 16))).astype(dtype)
    y = (latent[:, 0] > np.median(latent[:, 0])).astype(int)
    query = (rng.uniform(.1, 2, (35, 2)) @ basis
             + rng.uniform(0, .01, (35, 16))).astype(dtype)
    # Constant training columns guarantee discarded features for absolute screening.
    return (np.column_stack((x, np.zeros((72, 2), dtype=dtype))), y,
            np.column_stack((query, np.zeros((35, 2), dtype=dtype))))


def fit_model(data, *, head="knn", screening="absolute"):
    x, y, _ = data
    return NMFClassifier(
        solver="mu", screening=screening, score_threshold=2,
        classifier="knn" if head == "knn_sd" else head,
        knn_normalization="sd" if head == "knn_sd" else "none",
        n_components=2, n_neighbors=3, weights="distance",
        max_iter=2000, tol=1e-4, random_state=10,
        projection_native_threads=1,
    ).fit(x, y)


@pytest.fixture(scope="module")
def fitted(data):
    return fit_model(data)


def forbid_iteration(monkeypatch, model):
    def unexpected(*args, **kwargs):
        raise AssertionError("An exact-zero query must not enter iterative factorization")

    monkeypatch.setattr(model.decomposition_, "_fit_transform", unexpected)


def capture_transform(operation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            values = operation()
        except Exception as error:
            return None, (type(error), str(error)), [(w.category, str(w.message)) for w in caught]
    return values, None, [(w.category, str(w.message)) for w in caught]


def assert_delegates_like_sklearn(monkeypatch, model, query):
    """Use the independently called fitted sklearn estimator as the fallback reference."""
    selected = query[:, model.selected_features_]
    reference = deepcopy(model.decomposition_)
    expected, expected_error, expected_warnings = capture_transform(
        lambda: reference.transform(selected))
    calls = []
    original = model.decomposition_._fit_transform

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.decomposition_, "_fit_transform", traced)
    with _capture_projection(model) as diagnostic:
        actual, actual_error, actual_warnings = capture_transform(lambda: model.transform(query))
    assert calls, "The fallback must invoke the fitted sklearn solver"
    assert actual_error == expected_error
    assert actual_warnings == expected_warnings
    if expected_error is None:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert diagnostic["last_transform_converged"] is False
        assert diagnostic["last_transform_error"] == f"{expected_error[0].__name__}: {expected_error[1]}"


@pytest.mark.parametrize("head", ["knn", "knn_sd", "logistic", "linear_svm"])
def test_zero_coordinates_and_fitted_head_oracle(data, head, monkeypatch):
    model = fit_model(data, head=head, screening="none")
    query = np.zeros_like(data[2])
    expected = np.zeros((len(query), 2), dtype=query.dtype)
    before_basis = model.decomposition_.components_.copy()
    before_training = model.training_coordinates_.copy()
    fit_state = (model.converged_, model.n_iter_, model.convergence_warnings_)
    forbid_iteration(monkeypatch, model)

    # Zero coordinates attain the global lower bound of the nonnegative objective.
    with warnings.catch_warnings(), _capture_projection(model) as diagnostic:
        warnings.simplefilter("error")
        with np.errstate(invalid="raise"):
            coordinates = model.transform(query)
            np.testing.assert_array_equal(coordinates, expected)
            assert coordinates.dtype == query.dtype
            np.testing.assert_array_equal(coordinates @ before_basis, np.zeros_like(query))
            if head == "linear_svm":
                labels, scores = model.predict_with_decision_function(query)
                expected_scores = model.classifier_.decision_function(expected)
            else:
                labels, scores = model.predict_with_probabilities(query)
                expected_scores = model.classifier_.predict_proba(expected)
    np.testing.assert_array_equal(labels, model.classifier_.predict(expected))
    np.testing.assert_array_equal(scores, expected_scores)
    np.testing.assert_array_equal(model.decomposition_.components_, before_basis)
    np.testing.assert_array_equal(model.training_coordinates_, before_training)
    np.testing.assert_array_equal(query, np.zeros_like(query))
    assert (model.converged_, model.n_iter_, model.convergence_warnings_) == fit_state
    assert diagnostic["last_transform_converged"] is True
    assert diagnostic["last_transform_warnings"] == ()
    assert diagnostic["last_transform_error"] is None


def test_selected_zero_layouts_and_discarded_positive_features(data, fitted, monkeypatch):
    model = deepcopy(fitted)
    forbid_iteration(monkeypatch, model)
    zero = np.zeros_like(data[2])
    backing = np.zeros((len(zero), zero.shape[1] * 2), dtype=zero.dtype)
    readonly = zero.copy()
    readonly.flags.writeable = False
    queries = [zero, np.asfortranarray(zero), backing[:, ::2], readonly,
               np.broadcast_to(zero[:1], zero.shape), zero[:1], np.full_like(zero, -0.0)]
    discarded = np.setdiff1d(np.arange(zero.shape[1]), model.selected_features_)
    assert discarded.size
    raw_nonzero = zero.copy()
    raw_nonzero[:, discarded] = 1
    queries.append(raw_nonzero)
    for query in queries:
        before = query.copy()
        actual = model.transform(query)
        np.testing.assert_array_equal(actual, np.zeros((len(query), 2), dtype=query.dtype))
        np.testing.assert_array_equal(query, before)


def test_zero_success_has_fresh_diagnostics_without_rewriting_old_model_fields(data, fitted, monkeypatch):
    model = deepcopy(fitted)
    fit_state = (model.converged_, model.n_iter_)
    with pytest.raises(ValueError), _capture_projection(model) as failed:
        model.transform(data[2][:, :-1])
    assert failed["last_transform_converged"] is False
    assert failed["last_transform_error"] is not None
    # Legacy fields may exist on an older saved model, but are not call diagnostics.
    model.last_transform_converged_ = False
    model.last_transform_error_ = failed["last_transform_error"]
    model.last_transform_warnings_ = ("previous projection warning",)
    model.last_transform_seconds_ = -1
    stale = {key: value for key, value in vars(model).items() if key.startswith("last_transform_")}
    forbid_iteration(monkeypatch, model)
    with _capture_projection(model) as diagnostic:
        model.transform(np.zeros_like(data[2]))
    assert diagnostic["last_transform_converged"] is True
    assert diagnostic["last_transform_warnings"] == ()
    assert diagnostic["last_transform_error"] is None
    assert diagnostic["last_transform_seconds"] >= 0
    assert {key: value for key, value in vars(model).items() if key.startswith("last_transform_")} == stale
    assert failed["last_transform_converged"] is False
    assert (model.converged_, model.n_iter_) == fit_state


@pytest.mark.parametrize("kind", ["positive", "mixed"])
def test_nonzero_selected_batches_keep_sklearn_path(data, fitted, kind, monkeypatch):
    model = deepcopy(fitted)
    query = data[2].copy()
    if kind == "mixed":
        query[::2] = 0
    assert np.any(query[:, model.selected_features_])
    model.decomposition_.max_iter = 11
    assert_delegates_like_sklearn(monkeypatch, model, query)


@pytest.mark.parametrize("kind", ["empty_rows", "empty_features", "wrong_width", "negative", "nan", "inf", "complex"])
def test_full_query_validation_precedes_zero_shortcut(data, fitted, kind):
    model = deepcopy(fitted)
    query = np.zeros_like(data[2])
    if kind == "empty_rows":
        query = query[:0]
    elif kind == "empty_features":
        query = query[:, :0]
    elif kind == "wrong_width":
        query = query[:, :-1]
    elif kind == "complex":
        query = query.astype(np.complex128)
    else:
        # The last training column is discarded, but invalid raw values still fail.
        assert query.shape[1] - 1 not in model.selected_features_
        query[0, -1] = {"negative": -1, "nan": np.nan, "inf": np.inf}[kind]
    with pytest.raises(ValueError), _capture_projection(model) as diagnostic:
        model.transform(query)
    assert diagnostic["last_transform_converged"] is False
    assert diagnostic["last_transform_error"] is not None
    assert model.converged_ == fitted.converged_


@pytest.mark.parametrize("kind", ["zero", "negative", "nan", "inf", "wrong_width", "matrix"])
def test_invalid_basis_keeps_upstream_behavior(data, fitted, kind, monkeypatch):
    model = deepcopy(fitted)
    basis = model.decomposition_.components_.copy()
    if kind == "zero":
        basis[:] = 0
    elif kind in ("negative", "nan", "inf"):
        basis[0, 0] = {"negative": -1, "nan": np.nan, "inf": np.inf}[kind]
    elif kind == "wrong_width":
        basis = basis[:, :-1]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            basis = np.asmatrix(basis)
    model.decomposition_.components_ = basis
    model.decomposition_.max_iter = 11
    # Identical nonfinite results from deliberately invalid H demonstrate fallback,
    # not numerical validity. Prediction or later validation may reject them.
    assert_delegates_like_sklearn(monkeypatch, model, np.zeros_like(data[2]))


@pytest.mark.parametrize("kind", ["zero_row", "dependent_rows"])
def test_zero_solution_does_not_require_full_rank(data, fitted, kind, monkeypatch):
    model = deepcopy(fitted)
    if kind == "zero_row":
        model.decomposition_.components_[0] = 0
    else:
        model.decomposition_.components_[0] = model.decomposition_.components_[1]
    forbid_iteration(monkeypatch, model)
    actual = model.transform(np.zeros_like(data[2]))
    np.testing.assert_array_equal(actual, np.zeros((len(data[2]), 2), dtype=data[2].dtype))


def test_fitted_configuration_survives_constructor_changes(data, fitted, monkeypatch):
    model = deepcopy(fitted)
    model.set_params(solver="cd", screening="none", n_components=1, max_iter=1, tol=-1)
    forbid_iteration(monkeypatch, model)
    actual = model.transform(np.zeros_like(data[2]))
    assert actual.shape == (len(data[2]), fitted.decomposition_.components_.shape[0])
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_projection_settings_still_validate_for_mu(data, fitted):
    model = deepcopy(fitted).set_params(projection_workers=0)
    with pytest.raises(ValueError), _capture_projection(model) as diagnostic:
        model.transform(np.zeros_like(data[2]))
    assert diagnostic["last_transform_converged"] is False


@pytest.mark.parametrize("kind", ["nndsvd", "feature_names"])
def test_unrelated_warnings_are_not_hidden(data, fitted, kind, monkeypatch):
    model = deepcopy(fitted)
    if kind == "nndsvd":
        model.decomposition_.init = "nndsvd"
    else:
        model.decomposition_.feature_names_in_ = np.array(
            [f"feature_{i}" for i in range(len(model.selected_features_))])
    model.decomposition_.max_iter = 11
    assert_delegates_like_sklearn(monkeypatch, model, np.zeros_like(data[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning), _capture_projection(model) as diagnostic:
            model.transform(np.zeros_like(data[2]))
    assert diagnostic["last_transform_converged"] is False


def test_mixed_dtype_zero_query_uses_common_dtype(data, fitted, monkeypatch):
    model = deepcopy(fitted)
    other_dtype = np.float64 if data[2].dtype == np.float32 else np.float32
    before_basis = model.decomposition_.components_.copy()
    forbid_iteration(monkeypatch, model)
    actual = model.transform(np.zeros_like(data[2], dtype=other_dtype))
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.zeros((len(data[2]), 2)))
    np.testing.assert_array_equal(model.decomposition_.components_, before_basis)
    assert model.decomposition_.components_.dtype == before_basis.dtype


def test_low_level_mu_still_delegates_unaligned_dtypes(data, fitted):
    # The public classifier aligns dtypes; the recovery helper keeps its
    # narrower contract and upstream validation for direct calls.
    from supervised_nmf._mu import transform_mu
    other_dtype = np.float64 if data[2].dtype == np.float32 else np.float32
    selected = np.zeros((len(data[2]), len(fitted.selected_features_)), dtype=other_dtype)
    with pytest.raises(TypeError, match="same dtype"):
        transform_mu(deepcopy(fitted.decomposition_), selected)


def test_valid_iteration_caps_do_not_force_zero_work(data, fitted, monkeypatch):
    for cap, tol in ((1, 0), (10, 1e-4), (37, 1e-4)):
        model = deepcopy(fitted)
        model.decomposition_.max_iter = cap
        model.decomposition_.tol = tol
        forbid_iteration(monkeypatch, model)
        with warnings.catch_warnings(), _capture_projection(model) as diagnostic:
            warnings.simplefilter("error")
            actual = model.transform(np.zeros_like(data[2]))
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
        assert diagnostic["last_transform_converged"] is True


def test_pickled_model_retains_exact_zero_projection(data, fitted, monkeypatch):
    model = pickle.loads(pickle.dumps(fitted))
    forbid_iteration(monkeypatch, model)
    query = np.zeros_like(data[2])
    coordinates = model.transform(query)
    expected = np.zeros((len(query), 2), dtype=query.dtype)
    np.testing.assert_array_equal(coordinates, expected)
    labels, probabilities = model.predict_with_probabilities(query)
    np.testing.assert_array_equal(labels, model.classifier_.predict(expected))
    np.testing.assert_array_equal(probabilities, model.classifier_.predict_proba(expected))


def test_zero_prediction_retains_reversed_positive_label_and_threshold(data, fitted, monkeypatch):
    x, y, query = data
    labels = np.where(y == 0, "negative", "positive")
    model = clone(fitted).set_params(decision_threshold=.5, positive_label="negative")
    model.fit(x, labels)
    assert model.positive_class_index_ == 0
    forbid_iteration(monkeypatch, model)
    zero_coordinates = np.zeros((len(query), 2), dtype=query.dtype)
    expected_scores = model.classifier_.predict_proba(zero_coordinates)
    expected_labels = np.where(expected_scores[:, 0] > .5, "negative", "positive")
    actual_labels, actual_scores = model.predict_with_probabilities(np.zeros_like(query))
    np.testing.assert_array_equal(actual_scores, expected_scores)
    np.testing.assert_array_equal(actual_labels, expected_labels)


def test_custom_instance_transform_is_preserved(data, fitted, monkeypatch):
    model = deepcopy(fitted)
    calls = []

    def custom_transform(selected):
        calls.append(selected.copy())
        return np.full((len(selected), 2), .25, dtype=selected.dtype)

    monkeypatch.setattr(model.decomposition_, "transform", custom_transform)
    query = np.zeros_like(data[2])
    actual = model.transform(query)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], query[:, model.selected_features_])
    np.testing.assert_array_equal(actual, np.full((len(query), 2), .25, dtype=query.dtype))


def test_unfitted_and_failed_refit_still_reject_zero_input(data, fitted):
    x, y, query = data
    model = clone(fitted)
    with pytest.raises(NotFittedError):
        model.transform(np.zeros_like(query))
    model = deepcopy(fitted)
    with pytest.raises(ValueError):
        model.fit(np.zeros_like(x), y)
    with pytest.raises(NotFittedError):
        model.transform(np.zeros_like(query))
