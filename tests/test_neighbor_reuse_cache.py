"""Author: Chang Cheng. Contracts for qualified shared-neighbor vote caches.

Native selected-k estimators are the output oracle. Private-state observations
only verify the cache's safety, atomic publication and avoided repeated work.
"""

from unittest.mock import patch
import warnings

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

import supervised_nmf._neighbors as neighbors


ORDINARY = dict(divide="warn", over="warn", under="ignore", invalid="warn")


@pytest.fixture(autouse=True)
def native_budget_and_state():
    original_error = np.geterr().copy()
    original_filters = list(warnings.filters)
    with threadpool_limits(limits=1), np.errstate(**ORDINARY):
        yield
        assert np.geterr() == ORDINARY
        assert warnings.filters == original_filters
    assert np.geterr() == original_error


def problem(weights, *, ks=(1, 3, 5, 7), dtype=np.float64,
            coordinates=None, query=None, confident=False):
    X = np.asarray(np.arange(32) if coordinates is None else coordinates,
                   dtype=dtype)[:, None]
    Q = np.asarray([[.25], [.125]] if query is None else query, dtype=dtype)
    y = np.asarray(["z", "a", "m"])[np.arange(len(X)) % 3]
    if confident:
        y[:] = "a"
        y[-1] = "z"
    models = {k: KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree",
                                     weights=weights, n_jobs=1).fit(X, y)
              for k in ks}
    reference = {k: (model.predict(Q), model.predict_proba(Q))
                 for k, model in models.items()}
    reuse = neighbors._KNNQueryReuse(models, Q)
    assert len(reuse.groups) == 1
    return models, Q, reuse, reference


def compare(models, Q, reuse, reference, order=None):
    for k in models if order is None else order:
        labels, _, _ = reuse.predict(k)
        np.testing.assert_array_equal(labels, reference[k][0])
        for threshold in (0., .5, 1.):
            with patch.object(models[k], "predict_proba",
                              wraps=models[k].predict_proba) as public:
                actual, metadata, _ = reuse.predict(
                    k, probabilities=True, decision_threshold=threshold,
                    positive_index=1)
            if metadata["accelerated"]:
                public.assert_not_called()
                np.testing.assert_allclose(actual, reference[k][1], rtol=1e-12, atol=1e-14)
            else:
                public.assert_called_once()
                assert public.call_args.args[0] is Q
                np.testing.assert_array_equal(actual, reference[k][1])
            np.testing.assert_array_equal(actual[:, 1] > threshold,
                                          reference[k][1][:, 1] > threshold)


class AddProxy:
    """Intercept reduceat while forwarding all other operations of np.add."""

    def __init__(self, original, reduceat):
        self.original = original
        self.reduceat = reduceat

    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unsorted_clean_subset_and_training_endpoint_match_native(weights, dtype):
    models, Q, reuse, reference = problem(
        weights, ks=(8, 2, 5, 1, 3, 4, 7), dtype=dtype,
        coordinates=[0, 1, 1, 2, 3, 4, 5, 6])
    X = models[8]._fit_X
    before = X.tobytes(), Q.tobytes()
    compare(models, Q, reuse, reference)
    assert list(reuse.groups[0]["prefix_votes"]) == [1, 3, 4, 5, 7, 8]
    # k=2 cuts the duplicated coordinate; a cache for other heads cannot
    # certify it. The endpoint k=n needs no nonexistent next neighbor.
    with patch.object(models[2], "predict", wraps=models[2].predict) as public:
        actual, metadata, _ = reuse.predict(2)
    np.testing.assert_array_equal(actual, reference[2][0])
    public.assert_called_once()
    assert public.call_args.args[0] is Q
    assert metadata["near_boundary_rows"] == len(Q)
    assert metadata["vote_checks_performed"] is False
    assert (X.tobytes(), Q.tobytes()) == before


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_dense_requested_prefixes_match_native_without_interval_reduction(weights):
    models, Q, reuse, reference = problem(weights, ks=(7, 2, 6, 1, 3, 5, 4))

    def forbidden(*args, **kwargs):
        raise AssertionError("dense prefixes need no interval reduction")

    with patch.object(neighbors.np, "add", AddProxy(np.add, forbidden)):
        compare(models, Q, reuse, reference)
    group = reuse.groups[0]
    assert list(group["prefix_votes"]) == list(range(1, 8))
    assert "prefix_error_type" not in group


def test_sparse_requests_accumulate_only_selected_intervals():
    models, Q, reuse, reference = problem("distance", ks=(31, 1, 7, 19))
    original = np.cumsum
    shapes = []

    def observed(values, *args, **kwargs):
        shapes.append(values.shape)
        return original(values, *args, **kwargs)

    with patch.object(neighbors.np, "cumsum", observed):
        actual, _, _ = reuse.predict(31, probabilities=True)
    np.testing.assert_allclose(actual, reference[31][1], rtol=1e-12, atol=1e-14)
    assert shapes == [(len(Q), len(models))] * len(models[31].classes_)
    compare(models, Q, reuse, reference)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("endpoint_only", [False, True])
def test_one_clean_head_uses_selected_calculation_without_prefix_cache(weights, endpoint_only):
    if endpoint_only:
        settings = dict(ks=(8, 1, 3), coordinates=np.zeros(8))
    else:
        settings = dict(ks=(1, 2), coordinates=[0, 1, 1, 2, 3, 4, 5, 6])
    models, Q, reuse, reference = problem(weights, **settings)
    compare(models, Q, reuse, reference)
    group = reuse.groups[0]
    assert "prefix_votes" not in group
    assert group["prefix_unavailable"] is True
    assert group["prefix_skip_reason"] == "fewer_than_two_boundary_clean_heads"
    assert "prefix_error_type" not in group


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_zero_distance_prefixes_preserve_native_votes_and_thresholds(weights):
    models, Q, reuse, reference = problem(
        weights, ks=(2, 3, 5, 8), coordinates=[0, 0, 1, 2, 3, 4, 5, 6],
        query=[[0.], [.125]])
    compare(models, Q, reuse, reference)
    assert list(reuse.groups[0]["prefix_votes"]) == [2, 3, 5, 8]


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("probabilities", [False, True])
def test_boundary_fallback_skips_votes_and_reports_only_computed_diagnostics(weights, probabilities):
    models, Q, reuse, reference = problem(
        weights, ks=(1, 3, 5), coordinates=[0, 0, 1, 2, 3, 4],
        query=[[0.], [1.1]])
    method = "predict_proba" if probabilities else "predict"
    with patch.object(reuse, "_prefix_votes", side_effect=AssertionError("votes attempted")), \
            patch.object(models[1], method, wraps=getattr(models[1], method)) as public:
        actual, metadata, _ = reuse.predict(1, probabilities=probabilities)
    np.testing.assert_array_equal(actual, reference[1][int(probabilities)])
    public.assert_called_once()
    assert public.call_args.args[0] is Q
    assert metadata["reason"] == "numerical_or_boundary_uncertainty"
    assert metadata["near_boundary_rows"] == 1
    assert metadata["uncertainty_lower_bound_rows"] == 1
    assert metadata["vote_checks_performed"] is False
    assert metadata["public_fallback_query_rows"] == len(Q)
    assert metadata["shared_query_attempted"] is True
    assert not metadata["accelerated"]
    for key in ("near_vote_rows", "near_threshold_rows", "invalid_vote_rows",
                "vote_guard", "uncertain_rows"):
        assert metadata[key] is None


def test_boundary_fallback_preserves_selected_warning_as_error():
    models, Q, reuse, _ = problem(
        "distance", ks=(1, 3, 5), coordinates=[0, 0, 1, 2, 3, 4],
        query=[[0.], [1.1]])

    def selected(query):
        assert query is Q
        warnings.warn("selected warning", UserWarning)
        raise AssertionError("warning-as-error did not propagate")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        filters = list(warnings.filters)
        with patch.object(models[1], "predict", selected):
            with pytest.raises(UserWarning, match="selected warning"):
                reuse.predict(1)
        assert warnings.filters == filters


def test_complete_distance_matrix_is_qualified_once_across_reordered_heads():
    models, Q, reuse, reference = problem("distance", confident=True)
    group = reuse.groups[0]
    original = np.isfinite
    complete_checks = []

    def observed(values, *args, **kwargs):
        if values is group.get("distances"):
            complete_checks.append(values.shape)
        return original(values, *args, **kwargs)

    with patch.object(neighbors.np, "isfinite", observed):
        for k in (5, 1, 3, 7, 5):
            actual, metadata, _ = reuse.predict(k, probabilities=True)
            np.testing.assert_allclose(actual, reference[k][1], rtol=1e-12, atol=1e-14)
            assert metadata["accelerated"]
    assert complete_checks == [(len(Q), max(models) + 1)]
    assert group["invalid_distance"].shape == (len(Q),)
    assert not group["invalid_distance"].any()


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("operation", ["reduceat", "cumsum"])
@pytest.mark.parametrize("failure_call", [1, 2], ids=["first-class", "after-completed-class"])
def test_optional_prefix_failure_is_atomic_and_never_retried(weights, operation, failure_call):
    models, Q, reuse, reference = problem(weights)
    group = reuse.groups[0]
    original = np.add if operation == "reduceat" else np.cumsum
    calls = []

    def controlled(*args, **kwargs):
        calls.append(True)
        if len(calls) == failure_call:
            raise MemoryError("controlled optional prefix failure")
        function = original.reduceat if operation == "reduceat" else original
        return function(*args, **kwargs)

    target = "add" if operation == "reduceat" else "cumsum"
    replacement = AddProxy(original, controlled) if operation == "reduceat" else controlled
    with patch.object(neighbors.np, target, replacement):
        actual, metadata, _ = reuse.predict(1, probabilities=True)
    np.testing.assert_allclose(actual, reference[1][1], rtol=1e-12, atol=1e-14)
    assert metadata["accelerated"]
    assert len(calls) == failure_call
    assert "prefix_votes" not in group
    assert group["prefix_unavailable"] is True
    assert group["prefix_error_type"] == "MemoryError"

    def forbidden(*args, **kwargs):
        raise AssertionError("failed optional cache retried")

    replacement = AddProxy(original, forbidden) if operation == "reduceat" else forbidden
    with patch.object(neighbors.np, target, replacement):
        later, _, _ = reuse.predict(5, probabilities=True)
    np.testing.assert_allclose(later, reference[5][1], rtol=1e-12, atol=1e-14)
    assert "prefix_votes" not in group
    assert group["prefix_error"] == "controlled optional prefix failure"


class RejectCacheRead(dict):
    def __contains__(self, key):
        raise AssertionError("unqualified call inspected cached boundary proof")

    def __getitem__(self, key):
        raise AssertionError("unqualified call read cached votes")


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_strict_state_ignores_a_populated_cache_and_restores_state(weights):
    models, Q, reuse, _ = problem(weights, confident=True)
    reuse.predict(1, probabilities=True)
    group = reuse.groups[0]
    cached = group["prefix_votes"]
    saved = {k: value.tobytes() for k, value in cached.items()}
    group["prefix_votes"] = RejectCacheRead(cached)
    with np.errstate(all="raise"):
        state = np.geterr().copy()
        expected = models[3].predict_proba(Q)
        with patch.object(reuse, "_prefix_votes", side_effect=AssertionError("strict cache helper")):
            actual, _, _ = reuse.predict(3, probabilities=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        assert np.geterr() == state
    assert {k: value.tobytes() for k, value in cached.items()} == saved


@pytest.mark.parametrize("control", ["numpy-threshold", "bad-index", "bad-threshold"])
def test_unqualified_private_controls_do_not_consume_cached_proof(control):
    models, Q, reuse, reference = problem("distance", confident=True)
    reuse.predict(1, probabilities=True)
    group = reuse.groups[0]
    group["prefix_votes"] = RejectCacheRead(group["prefix_votes"])
    options = dict(probabilities=True, decision_threshold=np.float64(.5), positive_index=1)
    expected_error = None
    if control == "bad-index":
        options.update(decision_threshold=.5, positive_index=99)
        expected_error = IndexError
    elif control == "bad-threshold":
        options.update(decision_threshold="invalid")
        expected_error = TypeError
    with patch.object(reuse, "_prefix_votes", side_effect=AssertionError("unqualified cache helper")):
        if expected_error is None:
            actual, _, _ = reuse.predict(3, **options)
            np.testing.assert_allclose(actual, reference[3][1], rtol=1e-12, atol=1e-14)
        else:
            with pytest.raises(expected_error):
                reuse.predict(3, **options)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_published_clean_proof_reuses_readonly_mask_without_boundary_work(weights):
    models, Q, reuse, reference = problem(weights, confident=True)
    reuse.predict(1, probabilities=True)
    group = reuse.groups[0]
    assert set(group["prefix_votes"]) == set(models)
    mask = group["invalid_distance"]
    mask.flags.writeable = False
    saved_mask = mask.tobytes()
    saved_votes = {k: value.tobytes() for k, value in group["prefix_votes"].items()}
    results = []
    # These already qualified, confident heads need neither a new boundary
    # radius nor a public fallback. The final uncertainty union must not write
    # into the aliased finite-distance mask.
    with patch.object(neighbors.np, "maximum", side_effect=AssertionError("boundary recomputed")):
        for k in (7, 3, 5, 1, 3):
            with patch.object(models[k], "predict_proba", side_effect=AssertionError("public fallback")):
                results.append((k, reuse.predict(k, probabilities=True)))
    for k, (actual, metadata, _) in results:
        np.testing.assert_allclose(actual, reference[k][1], rtol=1e-12, atol=1e-14)
        assert metadata["accelerated"]
        assert metadata["near_boundary_rows"] == 0
    assert group["invalid_distance"] is mask
    assert not mask.flags.writeable
    assert mask.tobytes() == saved_mask
    assert {k: value.tobytes() for k, value in group["prefix_votes"].items()} == saved_votes
