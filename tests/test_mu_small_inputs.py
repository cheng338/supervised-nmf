"""Fixed-basis MU preserves representable solutions for small positive inputs."""

from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf.classifier import _capture_projection


@pytest.fixture(scope="module", autouse=True)
def single_numerical_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    return request.param


def fixed_basis_model(basis, *, head="knn"):
    """Start with a fitted classifier, then set a controlled valid projection basis.

    The independent oracle concerns the actual stored fixed basis, not the
    nonunique scale or factors recovered by an NMF training solve.
    """
    rank, features = basis.shape
    rng = np.random.default_rng(548)
    coordinates = rng.uniform(.2, 2, (24, rank))
    training_basis = np.full((rank, features), .2)
    training_basis[np.arange(rank), np.arange(rank)] += 1
    training = (coordinates @ training_basis).astype(basis.dtype)
    labels = (coordinates[:, 0] > np.median(coordinates[:, 0])).astype(int)
    model = NMFClassifier(
        solver="mu", screening="none", classifier=head, n_components=rank,
        n_neighbors=1, weights="uniform", max_iter=2000, tol=1e-4,
        random_state=10, projection_native_threads=1,
    ).fit(training, labels)
    model.decomposition_.components_ = basis.copy()
    model.selected_components_ = basis.copy()
    model.components_ = basis.copy()
    return model


def decimal_rank_one(query, basis):
    with localcontext() as context:
        context.prec = 2500
        h = [Decimal.from_float(float(value)) for value in basis[0]]
        denominator = sum(value * value for value in h)
        values = [sum(Decimal.from_float(float(value)) * component
                      for value, component in zip(row, h)) / denominator
                  for row in query]
        quantum = Decimal.from_float(float(np.nextafter(query.dtype.type(0), query.dtype.type(1))))
        tiny = Decimal.from_float(float(np.finfo(query.dtype).tiny))
        rounded = [((value / quantum).to_integral_value(rounding=ROUND_HALF_EVEN) * quantum
                    if value < tiny else value) for value in values]
        return np.asarray([[float(value)] for value in rounded], dtype=query.dtype)


def assert_small_coordinates(actual, expected):
    """Use relative error or one output ULP, never a fixed absolute tolerance."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert (actual >= 0).all()
    rtol = Decimal("0.000002") if expected.dtype == np.float32 else Decimal("0.000000000002")
    with localcontext() as context:
        context.prec = 2500
        for got, target in zip(actual.flat, expected.flat):
            a, b = Decimal.from_float(float(got)), Decimal.from_float(float(target))
            if target == 0:
                assert got == 0, "An exact or correctly rounded zero must remain zero"
                continue
            assert got > 0, "A representable positive coordinate was lost"
            neighbor = np.nextafter(target, expected.dtype.type(0))
            ulp = abs(Decimal.from_float(float(neighbor)) - b)
            assert abs(a - b) <= max(rtol * abs(b), ulp), (a, b, ulp)


def projection_without_warnings(model, query):
    original_query = query.copy()
    original_basis = model.decomposition_.components_.copy()
    original_training = model.training_coordinates_.copy()
    fit_state = (model.n_iter_, model.converged_, model.convergence_warnings_)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(invalid="raise", divide="raise", over="raise"), _capture_projection(model) as diagnostic:
            result = model.transform(query)
    np.testing.assert_array_equal(query, original_query)
    np.testing.assert_array_equal(model.decomposition_.components_, original_basis)
    np.testing.assert_array_equal(model.training_coordinates_, original_training)
    assert (model.n_iter_, model.converged_, model.convergence_warnings_) == fit_state
    assert diagnostic["last_transform_converged"] is True
    assert diagnostic["last_transform_warnings"] == ()
    assert diagnostic["last_transform_error"] is None
    assert diagnostic["last_transform_seconds"] >= 0
    return result


RANK_ONE_CASES = (
    "smallest_sparse", "smallest_dense", "normal_minimum", "mixed_rows",
    "mixed_entries", "tiny_basis", "orthogonal", "rounded_zero", "exact_initial",
    "halfway_zero", "above_halfway", "below_halfway",
    "combined_underflow", "combined_underflow_above", "combined_underflow_below",
    "decisive_tiny_term",
)


def rank_one_case(name, dtype):
    small = np.nextafter(dtype(0), dtype(1))
    tiny = np.finfo(dtype).tiny
    combined = dtype(2 * np.sqrt(tiny))
    cases = {
        "smallest_sparse": ([[small, 0]], [[.5, 0]]),
        "smallest_dense": ([[small, small]], [[.5, .5]]),
        "normal_minimum": ([[tiny, tiny]], [[.5, .25]]),
        "mixed_rows": ([[small, 0], [1, 0], [0, 0]], [[.5, 0]]),
        "mixed_entries": ([[small, 1]], [[.5, .5]]),
        "tiny_basis": ([[small, 0]], [[tiny, 0]]),
        "orthogonal": ([[0, small]], [[.5, 0]]),
        "rounded_zero": ([[small, 0]], [[4, 0]]),
        "exact_initial": ([[1]], [[1]]),
        "halfway_zero": ([[small, 0]], [[2, 0]]),
        "above_halfway": ([[small, 0]], [[np.nextafter(dtype(2), dtype(0)), 0]]),
        "below_halfway": ([[small, 0]], [[np.nextafter(dtype(2), dtype(np.inf)), 0]]),
        "decisive_tiny_term": ([[small, small]], [[2, small]]),
        "combined_underflow": ([[combined, 0]], [[combined, 0]]),
        "combined_underflow_above": ([[np.nextafter(combined, dtype(np.inf)), 0]],
                                     [[combined, 0]]),
        "combined_underflow_below": ([[np.nextafter(combined, dtype(0)), 0]],
                                     [[combined, 0]]),
    }
    query, basis = cases[name]
    return np.asarray(query, dtype=dtype), np.asarray(basis, dtype=dtype)


@pytest.mark.parametrize("case", RANK_ONE_CASES)
def test_small_rank_one_projection_matches_decimal_oracle(dtype, case):
    query, basis = rank_one_case(case, dtype)
    model = fixed_basis_model(basis)
    expected = decimal_rank_one(query, basis)
    actual = projection_without_warnings(model, query)
    assert_small_coordinates(actual, expected)
    if case in ("smallest_sparse", "smallest_dense", "orthogonal", "rounded_zero", "exact_initial"):
        np.testing.assert_array_equal(actual, expected)


def test_explicit_training_projection_uses_the_corrected_map(dtype, monkeypatch):
    small = np.nextafter(dtype(0), dtype(1))
    x = np.asarray([[small, 0], [2 * small, 0], [1, 0], [2, 0]], dtype=dtype)
    basis = np.asarray([[.5, 0]], dtype=dtype)

    def controlled_basis(self, values, y=None, **kwargs):
        # Isolate training reprojection from the separate joint factorization.
        self.components_ = basis.copy()
        self.n_components_ = 1
        self.n_features_in_ = values.shape[1]
        self.n_iter_ = 1
        self.reconstruction_err_ = 0.0
        return np.ones((len(values), 1), dtype=values.dtype)

    monkeypatch.setattr(NMF, "fit_transform", controlled_basis)
    model = NMFClassifier(solver="mu", screening="none", n_components=1,
                          n_neighbors=1, training_coordinates="transform")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x, np.asarray([0, 0, 1, 1]))
    expected = decimal_rank_one(x, basis)
    assert_small_coordinates(model.training_coordinates_, expected)
    np.testing.assert_array_equal(model.training_coordinates_, model.transform(x))
    assert model.training_projection_converged_ is True


@pytest.mark.parametrize("kind", ["mixed_entries", "all_tiny"])
def test_identity_basis_preserves_tiny_coordinates_within_rows(dtype, kind):
    small = np.nextafter(dtype(0), dtype(1))
    query = np.asarray([[1, small], [small, 1]], dtype=dtype)
    if kind == "all_tiny":
        query = np.asarray([[small, small], [4 * small, 0]], dtype=dtype)
    model = fixed_basis_model(np.eye(2, dtype=dtype))
    actual = projection_without_warnings(model, query)
    # Identity projection has the unique solution X itself, including its zeros.
    assert_small_coordinates(actual, query)
    np.testing.assert_array_equal(actual[query < np.finfo(dtype).tiny],
                                  query[query < np.finfo(dtype).tiny])


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_small_positive_projection_accepts_supported_layouts(dtype, layout):
    small = np.nextafter(dtype(0), dtype(1))
    values = np.asarray([[small, 0], [2 * small, 0], [0, 0]], dtype=dtype)
    basis = np.asarray([[.5, 0]], dtype=dtype)
    if layout == "F":
        query = np.asfortranarray(values)
    elif layout == "strided":
        backing = np.zeros((len(values), values.shape[1] * 3), dtype=dtype)
        backing[:, ::3] = values
        query = backing[:, ::3]
    else:
        query = values.copy()
        if layout == "readonly":
            query.flags.writeable = False
    actual = projection_without_warnings(fixed_basis_model(basis), query)
    np.testing.assert_array_equal(actual, decimal_rank_one(values, basis))


@pytest.mark.parametrize("head", ["knn", "logistic", "linear_svm"])
def test_small_input_prediction_uses_recovered_coordinates(dtype, head):
    tiny = np.finfo(dtype).tiny
    basis = np.asarray([[tiny / 2, 0]], dtype=dtype)
    query = np.asarray([[tiny, 0], [tiny / 4, 0]], dtype=dtype)
    expected_coordinates = decimal_rank_one(query, basis)
    model = fixed_basis_model(basis, head=head)
    actual_coordinates = projection_without_warnings(model, query)
    assert_small_coordinates(actual_coordinates, expected_coordinates)
    if head == "linear_svm":
        expected_scores = model.classifier_.decision_function(expected_coordinates)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            predicted, scores = model.predict_with_decision_function(query)
    else:
        expected_scores = model.classifier_.predict_proba(expected_coordinates)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            predicted, scores = model.predict_with_probabilities(query)
    np.testing.assert_array_equal(predicted, model.classifier_.predict(expected_coordinates))
    np.testing.assert_allclose(scores, expected_scores, rtol=2e-6 if dtype == np.float32 else 2e-12, atol=0)
    assert np.any(predicted != model.classifier_.predict(np.zeros_like(expected_coordinates)))


def test_small_projection_retains_positive_tolerance_iteration_warning(dtype):
    tiny = np.finfo(dtype).tiny
    query = np.asarray([[tiny, tiny]], dtype=dtype)
    model = fixed_basis_model(np.asarray([[.5, .25]], dtype=dtype))
    model.decomposition_.max_iter = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(invalid="raise", divide="raise"), _capture_projection(model) as diagnostic:
            coordinates = model.transform(query)
    assert len(caught) == 1
    assert issubclass(caught[0].category, ConvergenceWarning)
    assert "iterations 1 reached" in str(caught[0].message)
    assert diagnostic["last_transform_converged"] is False
    assert len(diagnostic["last_transform_warnings"]) == 1
    assert diagnostic["last_transform_error"] is None
    assert_small_coordinates(coordinates, decimal_rank_one(query, model.decomposition_.components_))


def test_unrepresentable_small_basis_coordinates_raise_and_allow_recovery(dtype):
    small = np.nextafter(dtype(0), dtype(1))
    basis = np.asarray([[small, 0]], dtype=dtype)
    model = fixed_basis_model(basis)
    original_basis = model.decomposition_.components_.copy()
    query = np.asarray([[1, 0]], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            with pytest.raises(FloatingPointError, match="output dtype"), _capture_projection(model) as diagnostic:
                model.transform(query)
    assert diagnostic["last_transform_converged"] is False
    assert diagnostic["last_transform_error"].startswith("FloatingPointError:")
    np.testing.assert_array_equal(model.decomposition_.components_, original_basis)
    recovered = projection_without_warnings(model, np.asarray([[small, 0]], dtype=dtype))
    assert_small_coordinates(recovered, np.ones((1, 1), dtype=dtype))
