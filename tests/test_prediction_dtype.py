"""CD/MU query precision follows the stored basis without narrowing values."""

from copy import deepcopy
import pickle
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection
from supervised_nmf.evaluation import evaluate_classifier


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=["cd", "mu"])
def solver(request):
    return request.param


@pytest.fixture(scope="module", params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    return request.param


@pytest.fixture(scope="module", params=[
    ("knn", "none"), ("knn", "absolute"), ("knn", "percentile"),
    ("knn_sd", "absolute"), ("logistic", "absolute"), ("linear_svm", "none"),
], ids=lambda choice: "-".join(choice))
def fitted(solver, dtype, request):
    head, screening = request.param
    rng = np.random.default_rng(729)
    latent = rng.uniform(.3, 2, (80, 2))
    basis = rng.uniform(.1, .4, (2, 12))
    basis[0, ::2] += 2
    basis[1, 1::2] += 2
    x = np.column_stack((latent @ basis, np.zeros((80, 2)))).astype(dtype)
    y = (latent[:, 0] > np.median(latent[:, 0])).astype(int)
    query = np.column_stack((rng.uniform(.3, 2, (12, 2)) @ basis, np.zeros((12, 2))))
    model = NMFClassifier(
        solver=solver, screening=screening, percentile=75, score_threshold=2,
        classifier="knn" if head == "knn_sd" else head,
        knn_normalization="sd" if head == "knn_sd" else "none",
        n_components=2, n_neighbors=3, weights="distance", max_iter=3000,
        tol=1e-4, random_state=10,
    ).fit(x, y)
    return model, query


def reference_coordinates(model, query):
    # Public sklearn on the actual stored basis is independent of the package's
    # query adapter. Double precision exactly contains both supported dtypes.
    reference = deepcopy(model.decomposition_)
    reference.components_ = reference.components_.astype(np.float64)
    selected = np.asarray(query, dtype=np.float64)[:, model.selected_features_]
    return reference.transform(selected)


@pytest.mark.parametrize("kind", ["array", "list", "uint8"])
def test_mixed_query_types_match_public_projection_and_fitted_head(fitted, kind):
    model, values = fitted
    other = np.float64 if model.decomposition_.components_.dtype == np.float32 else np.float32
    query = values.astype(other)
    if kind == "list":
        query = query.tolist()
    elif kind == "uint8":
        query = query.astype(np.uint8)
    before_query = np.asarray(query).copy()
    before_basis = model.decomposition_.components_.copy()
    before_training = model.training_coordinates_.copy()
    before_head = pickle.dumps(model.classifier_)
    expected = reference_coordinates(model, query)

    actual = model.transform(query)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    # Projection must not alter the fitted head. Subsequent KNN queries can
    # legitimately update native tree counters, so compare before those calls.
    assert pickle.dumps(model.classifier_) == before_head
    expected_labels = model.classifier_.predict(expected)
    np.testing.assert_array_equal(model.predict(query), expected_labels)
    if hasattr(model, "predict_proba"):
        labels, scores = model.predict_with_probabilities(query)
        expected_scores = model.classifier_.predict_proba(expected)
        np.testing.assert_allclose(model.predict_proba(query), expected_scores, rtol=2e-12, atol=0)
    else:
        labels, scores = model.predict_with_decision_function(query)
        expected_scores = model.classifier_.decision_function(expected)
        np.testing.assert_allclose(model.decision_function(query), expected_scores, rtol=2e-12, atol=0)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_allclose(scores, expected_scores, rtol=2e-12, atol=0)
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(model.decomposition_.components_, before_basis)
    np.testing.assert_array_equal(model.training_coordinates_, before_training)
    assert model.decomposition_.components_.dtype == before_basis.dtype


def controlled_model(solver, basis):
    # Fit a real classifier first; replace only the dictionary whose projection
    # has a closed-form oracle, independently of nonunique joint NMF scaling.
    rank, features = basis.shape
    coordinates = np.asarray([[.5, 2], [1, 1.5], [1.5, 1], [2, .5],
                              [2.5, .25], [3, .1]], dtype=basis.dtype)[:, :rank]
    dictionary = np.full((rank, features), .25, dtype=basis.dtype)
    dictionary[np.arange(rank), np.arange(rank)] += 1
    training = coordinates @ dictionary
    training += np.random.default_rng(651).uniform(.01, .02, training.shape).astype(basis.dtype)
    # This fit only supplies a real fitted head and metadata. Use a fixed
    # iteration count so unrelated joint-fit stopping does not qualify the
    # independently specified projection problem below.
    model = NMFClassifier(solver=solver, screening="none", n_components=rank,
                          n_neighbors=1, max_iter=1, tol=0, random_state=10).fit(
                              training, np.array([0, 0, 0, 1, 1, 1]))
    model.decomposition_.components_ = basis.copy()
    model.decomposition_.max_iter = 3000
    model.decomposition_.tol = 1e-4
    model.selected_components_ = basis.copy()
    model.components_ = basis.copy()
    return model


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_rank_one_mixed_precision_preserves_values_outside_float32_range(solver, layout):
    model = controlled_model(solver, np.asarray([[.5, 0]], dtype=np.float32))
    values = np.asarray([[1e-100, 0], [1e100, 0], [0, 0], [1, 4]], dtype=np.float64)
    query = values.copy()
    if layout == "F":
        query = np.asfortranarray(query)
    elif layout == "strided":
        backing = np.zeros((len(query), 4))
        backing[:, ::2] = query
        query = backing[:, ::2]
    elif layout == "readonly":
        query.flags.writeable = False
    before = pickle.dumps(model.decomposition_)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            actual = model.transform(query)
    # For H=(1/2,0), the unique nonnegative minimizer is W=2*X[:,0].
    expected = 2 * values[:, :1]
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    np.testing.assert_array_equal(query, values)
    assert pickle.dumps(model.decomposition_) == before


def test_reverse_mismatch_does_not_narrow_fitted_basis(solver):
    model = controlled_model(solver, np.asarray([[1e-100, 0]], dtype=np.float64))
    query = np.asarray([[1, 0], [.5, 0], [0, 0]], dtype=np.float32)
    before = model.decomposition_.components_.copy()
    actual = model.transform(query)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, query[:, :1].astype(np.float64) / 1e-100,
                               rtol=2e-12, atol=0)
    np.testing.assert_array_equal(model.decomposition_.components_, before)


def test_mixed_mu_identity_basis_preserves_float64_subnormals():
    model = controlled_model("mu", np.eye(2, dtype=np.float32))
    small = np.nextafter(np.float64(0), np.float64(1))
    query = np.asarray([[small, 0], [0, 4 * small], [0, 0]])
    actual = model.transform(query)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, query)


@pytest.mark.parametrize("kind", ["negative", "nan", "inf", "width"])
def test_mixed_query_full_input_validation_precedes_projection(solver, kind):
    model = controlled_model(solver, np.asarray([[.5, 0]], dtype=np.float32))
    query = np.ones((2, 2), dtype=np.float64)
    if kind == "width":
        query = query[:, :1]
    else:
        query[0, 1] = {"negative": -1, "nan": np.nan, "inf": np.inf}[kind]
    with pytest.raises(ValueError), _capture_projection(model) as diagnostic:
        model.transform(query)
    assert diagnostic["last_transform_error"] is not None
    assert diagnostic["last_transform_converged"] is False


def test_mixed_query_keeps_convergence_warning_and_serialized_basis(solver):
    basis = np.asarray([[1, .3, .2], [.2, 1, .4]], dtype=np.float32)
    model = controlled_model(solver, basis)
    model.decomposition_.max_iter = 1
    model.decomposition_.tol = 1e-12
    query = np.asarray([[1.1, 2.2, .4], [3, .5, 1]], dtype=np.float64)
    original = pickle.dumps(model.decomposition_)
    with pytest.warns(ConvergenceWarning, match="iterations 1 reached"):
        expected = reference_coordinates(model, query)
    with pytest.warns(ConvergenceWarning, match="iterations 1 reached"), _capture_projection(model) as diagnostic:
        actual = model.transform(query)
    np.testing.assert_array_equal(actual, expected)
    assert diagnostic["last_transform_converged"] is False
    assert pickle.dumps(model.decomposition_) == original
    restored = pickle.loads(pickle.dumps(model))
    with pytest.warns(ConvergenceWarning, match="iterations 1 reached"):
        np.testing.assert_array_equal(restored.transform(query), actual)
    assert restored.decomposition_.components_.dtype == np.float32


def test_evaluation_accepts_mixed_precision_query(solver):
    model = controlled_model(solver, np.asarray([[.5, 0]], dtype=np.float32))
    query = np.asarray([[.25, 0], [.5, 0], [1, 0], [2, 0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    report = evaluate_classifier(model, query, labels, threads=1)
    assert report["status"] == "valid", report
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], model.predict(query))
