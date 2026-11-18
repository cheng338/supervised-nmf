"""Ordinary MU shortcuts retain the direct row-local recurrence and decisions."""

from copy import deepcopy
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf import _mu_rowwise as mu
from supervised_nmf._mu import transform_mu


@pytest.fixture(scope="module", autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


def operands(rows=128, features=512, rank=8, dtype=np.float64):
    rng = np.random.default_rng(731940 + features + 10000 * rank)
    h = rng.uniform(.05, .2, (rank, features))
    for component in range(rank):
        h[component, component::rank] += 1
    x = rng.uniform(.1, 2, (rows, rank)) @ h
    x += rng.uniform(0, .01, x.shape)
    return x.astype(dtype), h.astype(dtype)


def assert_pair_equal(actual, expected):
    for left, right in zip(actual, expected):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cap", [1, 9, 10, 11, 19, 20, 21, 200])
def test_singleton_matches_direct_counts_and_coordinates(dtype, cap):
    x, h = operands(1, 31, 3, dtype)
    before = x.copy(), h.copy()
    assert_pair_equal(mu._ordinary_rows_single(x, h, cap, 1e-4),
                      mu._ordinary_rows_cached(x, h, cap, 1e-4))
    for actual, expected in zip((x, h), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["strided", "readonly", "fortran"])
def test_singleton_layout_and_zero_tolerance(dtype, layout):
    x, h = operands(1, 31, 3, dtype)
    if layout == "strided":
        x = np.repeat(x, 2, axis=1)[:, ::2]
        h = np.repeat(h, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        x.flags.writeable = h.flags.writeable = False
    else:
        x, h = np.asfortranarray(x), np.asfortranarray(h)
    before = x.tobytes(), h.tobytes()
    actual = mu._ordinary_rows_single(x, h, 21, 0.)
    assert_pair_equal(actual, mu._ordinary_rows_cached(x, h, 21, 0.))
    assert actual[1].tolist() == [21]
    assert (x.tobytes(), h.tobytes()) == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_singleton_zero_initial_residual_and_zero_denominator(dtype):
    x = np.ones((1, 1), dtype=dtype)
    h = np.ones((1, 1), dtype=dtype)
    actual = mu._ordinary_rows_single(x, h, 21, 1e-4)
    assert_pair_equal(actual, mu._ordinary_rows_cached(x, h, 21, 1e-4))
    assert actual[1].tolist() == [0]
    h = np.array([[1., 0.], [0., 0.]], dtype=dtype)
    x = np.array([[2., 1.]], dtype=dtype)
    assert_pair_equal(mu._ordinary_rows_single(x, h, 21, 1e-4),
                      mu._ordinary_rows_cached(x, h, 21, 1e-4))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("side, expected_count", [("below", 11), ("equal", 11), ("above", 10)])
def test_singleton_strict_stopping_boundary(dtype, side, expected_count):
    x, h = operands(1, 31, 3, dtype)
    start = np.broadcast_to(np.sqrt(np.mean(x, axis=1) / len(h))[:, None], (1, len(h))).copy()
    initial = mu._direct_row_errors(x, start, h)
    at_ten, counts = mu._ordinary_rows_cached(x, h, 10, 0.)
    assert counts.tolist() == [10]
    ratio = ((initial - mu._direct_row_errors(x, at_ten, h)) / initial)[0]
    assert np.isfinite(ratio) and ratio > 0
    tolerance = {"below": np.nextafter(ratio, dtype(0)), "equal": ratio,
                 "above": np.nextafter(ratio, dtype(np.inf))}[side]
    expected = mu._ordinary_rows_cached(x, h, 11, float(tolerance))
    assert expected[1].tolist() == [expected_count]
    assert_pair_equal(mu._ordinary_rows_single(x, h, 11, float(tolerance)), expected)


@pytest.mark.parametrize("rows, features, rank, dtype, tol, expected", [
    (1, 512, 8, np.float64, 1e-4, "single"),
    (127, 512, 8, np.float64, 1e-4, "cached"),
    (128, 512, 8, np.float64, 1e-4, "gram"),
    (129, 512, 8, np.float64, 1e-4, "gram"),
    (128, 511, 8, np.float64, 1e-4, "cached"),
    (128, 513, 8, np.float64, 1e-4, "gram"),
    (128, 512, 64, np.float64, 1e-4, "gram"),
    (128, 512, 65, np.float64, 1e-4, "cached"),
    (128, 512, 8, np.float32, 1e-4, "cached"),
    (128, 512, 8, np.float64, 0., "cached"),
])
def test_dispatch_boundaries_execute_the_selected_recurrence(monkeypatch, rows, features,
                                                            rank, dtype, tol, expected):
    x, h = operands(rows, features, rank, dtype)
    reference = mu._ordinary_rows_cached(x, h, 21, tol)
    calls = []
    for name in ("single", "cached", "gram"):
        original = getattr(mu, "_ordinary_rows_" + name)

        def record(*args, _name=name, _original=original):
            calls.append(_name)
            return _original(*args)

        monkeypatch.setattr(mu, "_ordinary_rows_" + name, record)
    assert_pair_equal(mu._ordinary_rows(x, h, 21, tol), reference)
    assert calls == [expected]


@pytest.mark.parametrize("value", [np.nextafter(2. ** -200, 0.), np.nextafter(2. ** 200, np.inf)])
def test_outside_gram_range_keeps_direct_path(monkeypatch, value):
    x, h = operands()
    x[0, 0] = value
    expected = mu._ordinary_rows_cached(x, h, 11, 1e-4)

    def forbidden(*args):
        raise AssertionError("An out-of-range operand reached the Gram shortcut")

    monkeypatch.setattr(mu, "_ordinary_rows_gram", forbidden)
    assert_pair_equal(mu._ordinary_rows(x, h, 11, 1e-4), expected)


def test_gram_envelope_is_used_and_encloses_direct_residual(monkeypatch):
    x, h = operands()
    expected = mu._ordinary_rows_cached(x, h, 20, 1e-4)
    original_interval, original_direct = mu._gram_error_interval, mu._direct_row_errors
    envelopes, direct_calls = [], []

    def interval(current, numerator, gram, xnorm, features):
        low, high, valid = original_interval(current, numerator, gram, xnorm, features)
        # All rows remain active for this fixed low-tolerance two-block fixture.
        assert len(current) == len(x)
        direct = original_direct(x, current, h)
        assert np.all((low <= direct) & (direct <= high))
        envelopes.append(valid.copy())
        return low, high, valid

    def direct(*args):
        direct_calls.append(len(args[0]))
        return original_direct(*args)

    monkeypatch.setattr(mu, "_gram_error_interval", interval)
    monkeypatch.setattr(mu, "_direct_row_errors", direct)
    assert_pair_equal(mu._ordinary_rows(x, h, 20, 1e-4), expected)
    assert np.all(expected[1] == 20)
    assert len(envelopes) == 2 and all(values.all() for values in envelopes)
    assert direct_calls == [len(x)]  # Both stopping checks actually used the envelope.


@pytest.mark.parametrize("side", ["below", "equal", "above"])
def test_uncertain_second_check_restores_previous_full_block(monkeypatch, side):
    x, h = operands()
    start = np.broadcast_to(np.sqrt(np.mean(x, axis=1) / len(h))[:, None], (len(x), len(h))).copy()
    initial = mu._direct_row_errors(x, start, h)
    ten, _ = mu._ordinary_rows_cached(x, h, 10, 0.)
    twenty, _ = mu._ordinary_rows_cached(x, h, 20, 0.)
    error_ten, error_twenty = mu._direct_row_errors(x, ten, h), mu._direct_row_errors(x, twenty, h)
    ratios = (error_ten - error_twenty) / initial
    threshold = ratios[np.argsort(ratios)[len(x) // 2]]
    assert threshold > 0 and np.isfinite(threshold)
    tolerance = {"below": np.nextafter(threshold, 0.), "equal": threshold,
                 "above": np.nextafter(threshold, np.inf)}[side]
    assert np.all((initial - error_ten) / initial > tolerance)
    expected = mu._ordinary_rows_cached(x, h, 21, float(tolerance))
    original_direct = mu._direct_row_errors
    calls = []

    def direct(a, w, b):
        calls.append((a.copy(), w.copy()))
        return original_direct(a, w, b)

    monkeypatch.setattr(mu, "_direct_row_errors", direct)
    assert_pair_equal(mu._ordinary_rows(x, h, 21, float(tolerance)), expected)
    assert len(calls) == 3  # Initial, current at check20, previous at check10.
    np.testing.assert_array_equal(calls[-1][0], x)
    np.testing.assert_array_equal(calls[-1][1], ten)
    assert np.any(expected[1] == 20) and np.any(expected[1] == 21)


def test_shortcut_arithmetic_failure_uses_direct_residual(monkeypatch):
    x, h = operands()
    expected = mu._ordinary_rows_cached(x, h, 21, 1e-4)
    attempts = []

    def unavailable(*args):
        attempts.append(len(args[0]))
        raise FloatingPointError("Only the shortcut arithmetic failed")

    monkeypatch.setattr(mu, "_gram_error_interval", unavailable)
    assert_pair_equal(mu._ordinary_rows(x, h, 21, 1e-4), expected)
    assert attempts == [len(x), len(x)]


@pytest.fixture(scope="module", params=["auto", "rowwise", "batch"])
def fitted_policy(request):
    x, _ = operands(24, 512, 3)
    model = NMFClassifier(solver="mu", mu_projection=request.param, screening="none",
                          n_components=3, n_neighbors=3, random_state=17,
                          max_iter=21, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, np.tile([0, 1], 12))
    return model


@pytest.mark.parametrize("rows, expected_route", [(1, "single"), (16, "cached"), (128, "gram")])
def test_public_prediction_uses_fitted_policy_and_preserves_head(monkeypatch, fitted_policy,
                                                               rows, expected_route):
    model = deepcopy(fitted_policy)
    x, _ = operands(rows, 512, 3)
    selected = x[:, model.selected_features_]
    before = x.copy(), model.decomposition_.components_.copy(), model.training_coordinates_.copy()
    policy = model.mu_projection_
    model.set_params(mu_projection="batch" if policy == "rowwise" else "auto")
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        if policy == "batch":
            coordinates = transform_mu(deepcopy(model.decomposition_), selected)
        else:
            with monkeypatch.context() as direct:
                direct.setattr(mu, "_ordinary_rows", mu._ordinary_rows_cached)
                coordinates = mu.transform_mu_rowwise(deepcopy(model.decomposition_), selected)
    labels = model.classifier_.predict(coordinates)
    probabilities = model.classifier_.predict_proba(coordinates)
    calls = []
    for name in ("single", "cached", "gram"):
        original = getattr(mu, "_ordinary_rows_" + name)

        def record(*args, _name=name, _original=original):
            calls.append(_name)
            return _original(*args)

        monkeypatch.setattr(mu, "_ordinary_rows_" + name, record)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        actual_labels, actual_probabilities = model.predict_with_probabilities(x)
    np.testing.assert_array_equal(actual_labels, labels)
    np.testing.assert_array_equal(actual_probabilities, probabilities)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in reference_warnings]
    assert calls == ([expected_route] if policy == "rowwise" else [])
    assert model.mu_projection_ == policy
    for actual, expected in zip((x, model.decomposition_.components_, model.training_coordinates_), before):
        np.testing.assert_array_equal(actual, expected)
