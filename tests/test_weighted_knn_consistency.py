"""Weighted KNN convenience paths must preserve the native fitted decision."""
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from supervised_nmf import NMFClassifier
from supervised_nmf._neighbors import _KNNQueryReuse
from tests.test_neighbor_full_membership import native_queries


def near_tie_data():
    # Finite Euclidean data, found by a fixed bounded roundoff probe. The two
    # native sklearn reductions disagree at a rounding-level weighted tie.
    rng = np.random.default_rng(3881)
    y = np.arange(32) % 2
    for _ in range(2):
        weights = rng.uniform(.1, 1, 32)
        extra = weights[y == 0].sum() - weights[y == 1].sum()
        i = np.flatnonzero(y == (1 if extra > 0 else 0))[0]
        weights[i] += abs(extra)
    return (1. / weights)[:, None], y, np.zeros((1, 1))


def wrapper(head, threshold=None, positive_index=1):
    # Isolate prediction post-processing from the independently tested NNLS.
    model = NMFClassifier()
    model.classifier_ = head
    model.classes_ = head.classes_
    model.decision_threshold_ = threshold
    model.positive_class_index_ = positive_index
    model.transform = Mock(side_effect=lambda X: X)
    return model


@pytest.mark.parametrize("labels", [np.array([0, 1]), np.array(["absent", "present"])])
def test_roundoff_tie_combined_labels_match_native_predict(labels):
    X, y, query = near_tie_data()
    head = KNeighborsClassifier(n_neighbors=32, weights="distance", algorithm="kd_tree").fit(X, labels[y])
    model = wrapper(head)
    expected = head.predict(query)
    expected_proba = head.predict_proba(query)
    assert abs(expected_proba[0, 1] - .5) < 1e-14
    with patch.object(head, "kneighbors", wraps=head.kneighbors) as neighbors:
        predicted, probability = model.predict_with_probabilities(query)
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(probability, expected_proba)
    assert neighbors.call_count == 2
    model.transform.assert_called_once_with(query)


@pytest.mark.parametrize("positive_index", [0, 1])
def test_explicit_threshold_keeps_strict_probability_rule(positive_index):
    X, y, query = near_tie_data()
    head = KNeighborsClassifier(n_neighbors=32, weights="distance", algorithm="kd_tree").fit(X, y)
    model = wrapper(head, threshold=.5, positive_index=positive_index)
    with patch.object(head, "predict", side_effect=AssertionError("explicit rule must not use native decision")):
        predicted, probability = model.predict_with_probabilities(query)
    expected = np.where(probability[:, positive_index] > .5, positive_index, 1 - positive_index)
    np.testing.assert_array_equal(predicted, expected)
    model.transform.assert_called_once_with(query)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree", "brute"])
def test_unambiguous_votes_keep_one_neighbor_query(dtype, weights, algorithm):
    X = np.array([[0.], [1.], [2.], [8.], [9.], [10.]], dtype=dtype)
    y = np.array([0, 0, 0, 1, 1, 1])
    query = np.array([[.2], [9.8]], dtype=dtype)
    head = KNeighborsClassifier(n_neighbors=3, weights=weights, algorithm=algorithm).fit(X, y)
    expected, expected_proba = head.predict(query), head.predict_proba(query)
    model = wrapper(head)
    with patch.object(head, "kneighbors", wraps=head.kneighbors) as neighbors:
        predicted, probability = model.predict_with_probabilities(query)
    # sklearn's native uniform/brute path bypasses Python kneighbors.
    assert neighbors.call_count <= 1
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(probability, expected_proba)
    model.transform.assert_called_once_with(query)


def test_roundoff_fallback_preserves_whole_query_batch():
    X, y, first = near_tie_data()
    query = np.vstack((first, [[10.]]))
    head = KNeighborsClassifier(n_neighbors=32, weights="distance", algorithm="kd_tree").fit(X, y)
    model = wrapper(head)
    expected = head.predict(query)
    with patch.object(head, "predict", wraps=head.predict) as native:
        predicted, _ = model.predict_with_probabilities(query)
    native.assert_called_once_with(query)
    np.testing.assert_array_equal(predicted, expected)
    model.transform.assert_called_once_with(query)


def test_odd_k_does_not_remove_distance_weighted_vote_tie():
    X = np.array([[1.], [2.], [2.]])
    y = np.array([0, 1, 1])
    query = np.zeros((1, 1))
    head = KNeighborsClassifier(n_neighbors=3, weights="distance", algorithm="kd_tree").fit(X, y)
    model = wrapper(head)
    predicted, probability = model.predict_with_probabilities(query)
    np.testing.assert_array_equal(predicted, [0])
    np.testing.assert_array_equal(probability, [[.5, .5]])
    model.transform.assert_called_once_with(query)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scaled_distance_head_preserves_labels_and_scales_once(dtype):
    rng = np.random.default_rng(277)
    X = rng.uniform(.1, 3, (30, 3)).astype(dtype)
    y = np.arange(30) % 2
    query = rng.uniform(.1, 3, (7, 3)).astype(dtype)
    pipeline = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute")),
    ]).fit(X, y)
    model = wrapper(pipeline)
    expected, expected_proba = pipeline.predict(query), pipeline.predict_proba(query)
    scaler = pipeline.named_steps["scale"]
    with patch.object(scaler, "transform", wraps=scaler.transform) as scaling:
        predicted, probability = model.predict_with_probabilities(query)
    scaling.assert_called_once_with(query)
    model.transform.assert_called_once_with(query)
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(probability, expected_proba)


def test_callable_weights_delegate_to_native_decision():
    X = np.arange(6, dtype=float)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    head = KNeighborsClassifier(n_neighbors=3, weights=lambda d: 1 / (1 + d), algorithm="kd_tree").fit(X, y)
    query = np.array([[.2], [4.9]])
    model = wrapper(head)
    expected = head.predict(query)
    with patch.object(head, "predict", wraps=head.predict) as native:
        predicted, _ = model.predict_with_probabilities(query)
    native.assert_called_once_with(query)
    np.testing.assert_array_equal(predicted, expected)
    model.transform.assert_called_once_with(query)


def test_duplicate_zero_distances_and_exact_weighted_tie():
    X = np.array([[0.], [0.], [1.], [2.]])
    y = np.array([0, 1, 1, 1])
    query = np.zeros((2, 1))
    head = KNeighborsClassifier(n_neighbors=4, weights="distance", algorithm="kd_tree").fit(X, y)
    model = wrapper(head)
    predicted, probability = model.predict_with_probabilities(query)
    np.testing.assert_array_equal(predicted, head.predict(query))
    np.testing.assert_array_equal(probability, np.full((2, 2), .5))
    model.transform.assert_called_once_with(query)


@pytest.mark.parametrize("probabilities", [False, True])
def test_overflowed_neighbor_distances_force_public_batch_fallback(probabilities):
    X = (np.arange(1, 11, dtype=float) * 1e200)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    query = np.zeros((2, 1))
    models = {k: KNeighborsClassifier(n_neighbors=k, weights="uniform", algorithm="kd_tree").fit(X, y)
              for k in (3, 5)}
    shared = _KNNQueryReuse(models, query, _head_fit_input=X)
    with np.errstate(invalid="raise"):
        predicted, metadata, _ = shared.predict(3, probabilities=probabilities)
    expected = models[3].predict_proba(query) if probabilities else models[3].predict(query)
    np.testing.assert_array_equal(predicted, expected)
    assert not metadata["accelerated"]
    assert metadata["reason"] == "nonfinite_neighbor_distances"
    assert metadata["invalid_distance_rows"] == 2
    assert metadata["public_fallback_query_rows"] == len(query)


@pytest.mark.parametrize("bad_distance", [np.nan, np.inf])
def test_nonfinite_shared_query_never_qualifies_cached_votes(bad_distance):
    X = np.arange(10, dtype=float)[:, None]
    y = np.arange(10) % 2
    query = np.array([[.1], [8.9]])
    models = {k: KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="kd_tree").fit(X, y)
              for k in (3, 5)}
    shared = _KNNQueryReuse(models, query, _head_fit_input=X)
    def faulty_shared(count, result):
        assert count == 6
        distances, _ = result
        distances[0, -1] = bad_distance

    with native_queries(models, on_shared=faulty_shared):
        predicted, metadata, _ = shared.predict(3)
    np.testing.assert_array_equal(predicted, models[3].predict(query))
    assert not metadata["accelerated"]
    assert metadata["invalid_distance_rows"] == 1
    assert metadata["public_fallback_query_rows"] == 2
