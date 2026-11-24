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


def test_mismatched_head_uses_its_fitted_count_and_keeps_siblings_eligible():
    X = np.array([0., .35, .8, 1.5, 2.1, 3.05, 5., 6.2])[:, None]
    y = [1, 0, 0, 0, 1, 1, 0, 1]
    Q = np.array([[.01]])
    models = {key: KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X, y)
              for key, k in ((1, 3), (3, 3), (5, 5))}
    reuse = neighbors._KNNQueryReuse(models, Q)
    for key, model in models.items():
        actual, metadata, _ = reuse.predict(key, probabilities=True)
        np.testing.assert_array_equal(actual, model.predict_proba(Q))
        assert metadata["shared_query_attempted"] is (key != 1)
        if key == 1:
            assert metadata["reason"] == "neighbor_key_mismatch"


@pytest.mark.parametrize("unit", ["ns", "D"])
def test_timedelta_neighbor_key_delegates_to_selected_head(unit):
    X = np.arange(8, dtype=float)[:, None]
    Q = np.array([[.25], [2.25]])
    temporal = np.timedelta64(1, unit)
    models = {key: KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X, np.arange(8) % 2)
              for key, k in [(temporal, 1), (3, 3), (5, 5)]}
    reuse = neighbors._KNNQueryReuse(models, Q)
    actual, metadata, _ = reuse.predict(temporal, probabilities=True)
    np.testing.assert_array_equal(actual, models[temporal].predict_proba(Q))
    assert metadata["reason"] == "neighbor_key_mismatch"
    assert not metadata["shared_query_attempted"]
    assert reuse.predict(3, probabilities=True)[1]["shared_query_attempted"]


@pytest.mark.parametrize("alias", [True, 1.0])
def test_non_integer_lookup_of_an_integer_key_calls_its_own_estimator(alias):
    X = np.arange(8, dtype=float)[:, None]
    Q = np.array([[.25], [2.25]])
    heads = {k: KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X, np.arange(8) % 2)
             for k in (1, 3)}
    reuse = neighbors._KNNQueryReuse(heads, Q)
    assert heads[alias] is heads[1]
    actual, metadata, _ = reuse.predict(alias)
    np.testing.assert_array_equal(actual, heads[1].predict(Q))
    assert metadata["reason"] == "neighbor_key_mismatch"
    assert not reuse.query_records()[0]["queried"]


@pytest.mark.parametrize("unit", ["ns", "D"])
def test_timedelta_fitted_count_preserves_the_selected_call(unit):
    X = np.arange(8, dtype=float)[:, None]
    Q = np.array([[.25], [2.25]])
    models = {k: KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(X, np.arange(8) % 2)
              for k in (1, 3, 5)}
    models[1].set_params(n_neighbors=np.timedelta64(1, unit))
    reuse = neighbors._KNNQueryReuse(models, Q)
    assert reuse.by_k[1] == (None, "unsupported_neighbor_count")
    with patch.object(models[1], "predict_proba", side_effect=ValueError("selected count")) as selected:
        with pytest.raises(ValueError, match="selected count"):
            reuse.predict(1, probabilities=True)
    selected.assert_called_once_with(Q)
    assert not reuse.query_records()[0]["queried"]
    assert reuse.predict(3, probabilities=True)[1]["shared_query_attempted"]


@pytest.mark.parametrize("algorithm", ["brute", "ball_tree"])
@pytest.mark.parametrize("positive_index", [
    None, True, -1, 2, np.timedelta64(1, "ns"), np.timedelta64(1, "D"),
])
def test_grouped_threshold_requires_a_valid_class_column_before_query(algorithm, positive_index):
    X = np.arange(4, dtype=float)[:, None]
    Q = np.array([[.25]])
    models = {k: KNeighborsClassifier(n_neighbors=k, algorithm=algorithm).fit(X, [0, 1, 0, 1])
              for k in (1, 2)}
    reuse = neighbors._KNNQueryReuse(models, Q)
    with pytest.raises(IndexError, match="positive class index"):
        reuse.predict(2, probabilities=True, decision_threshold=.5, positive_index=positive_index)
    assert not reuse.query_records()[0]["queried"]
    actual, _, _ = reuse.predict(2, probabilities=True, decision_threshold=.5, positive_index=1)
    np.testing.assert_array_equal(actual, models[2].predict_proba(Q))


@pytest.mark.parametrize("algorithm", ["brute", "ball_tree"])
@pytest.mark.parametrize("threshold", [
    "invalid", np.full((2, 1), .5), np.timedelta64(1, "ns"), np.timedelta64(1, "D"),
])
def test_grouped_threshold_requires_a_real_scalar_before_query_or_fallback(algorithm, threshold):
    X = np.arange(8, dtype=float)[:, None]
    Q = np.array([[.25], [2.25]])
    y = np.arange(len(X)) % 2
    models = {k: KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, leaf_size=1).fit(X, y)
              for k in (1, 2)}
    reuse = neighbors._KNNQueryReuse(models, Q)
    with pytest.raises(TypeError, match="real scalar"):
        reuse.predict(2, probabilities=True, decision_threshold=threshold, positive_index=1)
    assert not reuse.query_records()[0]["queried"]
    actual, _, _ = reuse.predict(2, probabilities=True, decision_threshold=np.float32(.5), positive_index=1)
    np.testing.assert_array_equal(actual, models[2].predict_proba(Q))


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
    reuse = neighbors._KNNQueryReuse(models, Q, _head_fit_input=X)
    assert len(reuse.groups) == 1
    return models, Q, reuse, reference


def compare(models, Q, reuse, reference, order=None):
    for k in models if order is None else order:
        labels, _, _ = reuse.predict(k)
        np.testing.assert_array_equal(labels, reference[k][0])
        for threshold in (0., .5, 1.):
            with patch.object(reuse, "_public", wraps=reuse._public) as public:
                actual, metadata, _ = reuse.predict(
                    k, probabilities=True, decision_threshold=threshold,
                    positive_index=1)
            if metadata["accelerated"]:
                public.assert_not_called()
                np.testing.assert_allclose(actual, reference[k][1], rtol=1e-12, atol=1e-14)
            else:
                public.assert_called_once_with(models[k], True)
                assert reuse.query is Q
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
    with patch.object(reuse, "_public", wraps=reuse._public) as public:
        actual, metadata, _ = reuse.predict(2)
    np.testing.assert_array_equal(actual, reference[2][0])
    public.assert_called_once_with(models[2], False)
    assert reuse.query is Q
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
    with patch.object(reuse, "_prefix_votes", side_effect=AssertionError("votes attempted")), \
            patch.object(reuse, "_public", wraps=reuse._public) as public:
        actual, metadata, _ = reuse.predict(1, probabilities=probabilities)
    np.testing.assert_array_equal(actual, reference[1][int(probabilities)])
    public.assert_called_once_with(models[1], probabilities)
    assert reuse.query is Q
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

    # Only the selected head expects named columns. Its public validation
    # warns on the unnamed query after the shared boundary detects the tie.
    models[1].feature_names_in_ = np.array(["coordinate"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        filters = list(warnings.filters)
        with pytest.raises(UserWarning, match="feature names") as direct:
            models[1].predict(Q)
        with patch.object(reuse, "_public", wraps=reuse._public) as public:
            with pytest.raises(UserWarning) as actual:
                reuse.predict(1)
        public.assert_called_once_with(models[1], False)
        assert str(actual.value) == str(direct.value)
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
    with patch.object(neighbors.np, "maximum", side_effect=AssertionError("boundary recomputed")), \
            patch.object(reuse, "_public", side_effect=AssertionError("public fallback")):
        for k in (7, 3, 5, 1, 3):
            results.append((k, reuse.predict(k, probabilities=True)))
    for k, (actual, metadata, _) in results:
        np.testing.assert_allclose(actual, reference[k][1], rtol=1e-12, atol=1e-14)
        assert metadata["accelerated"]
        assert metadata["near_boundary_rows"] == 0
    assert group["invalid_distance"] is mask
    assert not mask.flags.writeable
    assert mask.tobytes() == saved_mask
    assert {k: value.tobytes() for k, value in group["prefix_votes"].items()} == saved_votes


def _kd_refit_head(k, *, weights="uniform", leaf_size=30):
    return KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", metric="euclidean",
                                weights=weights, leaf_size=leaf_size, n_jobs=1)


@pytest.mark.parametrize("donor_k", [1, 3])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_nominally_euclidean_heads_use_their_actual_fitted_metric(donor_k, weights):
    from sklearn.neighbors import NearestNeighbors

    X = np.array([[2., 2.], [2.05, 2.], [2.1, 2.], [3., 0.], [3.1, 0.], [3.2, 0.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    Q = np.zeros((1, 2))
    donor = NearestNeighbors(algorithm="kd_tree", metric="manhattan").fit(X)
    models = {k: _kd_refit_head(k, weights=weights).fit(donor if k == donor_k else X, y)
              for k in (1, 3)}
    assert models[1]._fit_X is models[3]._fit_X
    assert models[1].effective_metric_ == models[3].effective_metric_ == "euclidean"
    # At the origin the L1 and L2 nearest triples have opposite labels.
    for order in ((1, 3), (3, 1)):
        reuse = neighbors._KNNQueryReuse({k: models[k] for k in order}, Q)
        for k in order:
            expected = models[k].predict(Q)
            assert expected.tolist() == [int(k == donor_k)]
            np.testing.assert_array_equal(reuse.predict(k)[0], expected)
            actual = reuse.predict(k, probabilities=True, decision_threshold=.5, positive_index=1)[0]
            np.testing.assert_allclose(actual, models[k].predict_proba(Q), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("history", ["wrong_width", "feature_names"])
def test_public_euclidean_index_refit_retains_selected_head_validation(history):
    from sklearn.neighbors import NearestNeighbors

    X = np.arange(6, dtype=float)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    Q = np.array([[.1]])
    if history == "feature_names":
        # Only the test for a real named frame needs the optional dependency.
        pandas = pytest.importorskip("pandas")
        initial = pandas.DataFrame(X, columns=["coordinate"])
        expected_error = UserWarning
    else:
        initial = np.column_stack((X, X))
        expected_error = ValueError
    donor = NearestNeighbors(algorithm="kd_tree", metric="euclidean").fit(X)
    small = _kd_refit_head(1).fit(initial, y).fit(donor, y)
    reuse = neighbors._KNNQueryReuse({1: small, 3: _kd_refit_head(3).fit(X, y)}, Q)
    # The selected head validates Q against metadata retained from its first fit.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(expected_error) as reference:
            small.predict_proba(Q)
        with pytest.raises(type(reference.value)) as actual:
            reuse.predict(1, probabilities=True)
    assert str(actual.value) == str(reference.value)
    if history == "feature_names":
        with pytest.warns(UserWarning) as direct:
            expected = small.predict_proba(Q)
        with warnings.catch_warnings(record=True) as visible:
            warnings.simplefilter("always")
            actual, _, shared = reuse.predict(1, probabilities=True)
        assert [(w.category, str(w.message)) for w in [*visible, *shared]] == [
            (w.category, str(w.message)) for w in direct]
        np.testing.assert_array_equal(actual, expected)


def test_float32_trees_fit_before_and_after_input_edit_use_selected_snapshot():
    X = np.array([[1., 1.], [2., 2.], [3., 0.], [4., 0.]], dtype=np.float32)
    Q = np.zeros((1, 2), dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    small = _kd_refit_head(1).fit(X, y)
    X[[0, 2]] = X[[2, 0]]
    large = _kd_refit_head(3).fit(X, y)
    reuse = neighbors._KNNQueryReuse({1: small, 3: large}, Q)
    np.testing.assert_array_equal(reuse.predict(1)[0], small.predict(Q))


@pytest.mark.parametrize("warm", [False, True])
def test_kdtree_predictions_follow_selected_width_after_input_edit(warm):
    X = np.array([0., 1., 2., 4., 6., 7., 8., 9.])[:, None]
    y = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    Q = np.array([[4.125]])
    models = {k: _kd_refit_head(k, leaf_size=2).fit(X, y) for k in (1, 3)}
    reuse = neighbors._KNNQueryReuse(models, Q)
    if warm:
        np.testing.assert_array_equal(reuse.predict(1)[0], models[1].predict(Q))
        # With a prior result, moving a row in the left leaf changes native k=1.
        X[2, 0] = 4.12525
    else:
        # The right leaf retains its old lower bound at 6. Native k=1 may
        # prune it while a wider query visits it and includes the moved row.
        X[4, 0] = 4.13
    np.testing.assert_array_equal(reuse.predict(1)[0], models[1].predict(Q))


@pytest.mark.parametrize("threshold,index,error", [(.5, 2, IndexError), ("bad", 1, TypeError)])
def test_grouped_kdtree_threshold_errors_precede_selected_or_shared_query(threshold, index, error):
    X = np.arange(8, dtype=float)[:, None]
    y = np.arange(8) % 2
    Q = np.array([[.25]])
    models = {k: _kd_refit_head(k, leaf_size=1).fit(X, y) for k in (1, 3)}
    reuse = neighbors._KNNQueryReuse(models, Q)
    before = [model._tree.get_n_calls() for model in models.values()]
    with pytest.raises(error):
        reuse.predict(1, probabilities=True, decision_threshold=threshold, positive_index=index)
    assert [model._tree.get_n_calls() for model in models.values()] == before
    assert all(not record["queried"] for record in reuse.query_records())
