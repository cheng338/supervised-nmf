"""Execution order changes scheduling, while preserving shared-grid evidence."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier
from supervised_nmf import study
from supervised_nmf._neighbors import _KNNQueryReuse
from supervised_nmf.screening import BinaryScreening


class StringSubclass(str):
    pass


class CustomScreen(BinaryScreening):
    """An ordinary supported transformer outside the exact-screen shortcut."""


_weight_calls = []
_scalar_calls = []


class ObservableFloat(float, BaseEstimator):
    """A clonable constructor value whose validation has an observable effect."""

    def __new__(cls, value):
        return float.__new__(cls, value)

    def __init__(self, value):
        self.value = value

    def __le__(self, other):
        _scalar_calls.append((float(self), other))
        return float.__le__(self, other)


def stateful_weights(distances):
    """Make visiting a different fold first observable to a user callback."""
    _weight_calls.append((distances.shape, distances.tobytes()))
    result = np.ones_like(distances)
    if len(_weight_calls) % 2:
        result[:, 0] = 4
    else:
        result[:, -1] = 4
    return result


def custom_metric(left, right):
    return float(np.sqrt(np.sum((left - right) ** 2)))


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def problem():
    rng = np.random.RandomState(217)
    y = np.tile([0, 1], 24)
    coefficients = rng.uniform(.1, 1., (len(y), 2))
    coefficients[:, 0] += .5 * y
    basis = np.column_stack([np.eye(2), rng.uniform(.1, 1., (2, 6))])
    X = coefficients @ basis + rng.uniform(0, .025, (len(y), 8))
    # Constant features exercise exclusion without an artificial all-zero input.
    X[:, -1] = 1
    return X, y


def screened_pipeline(**head_options):
    return Pipeline([
        ("screen", BinaryScreening(screening="absolute", score_threshold=0)),
        ("knn", KNeighborsClassifier(algorithm="brute", **head_options)),
    ])


def run_grid(problem, path, estimator=None, grid=None, **kwargs):
    X, y = problem
    if estimator is None:
        estimator = screened_pipeline()
    if grid is None:
        grid = {"screen__score_threshold": [0, 2], "knn__n_neighbors": [1, 3]}
    return study.calibration_knn_grid(
        estimator, grid, X, y, cv=3, seed=17, threads=1,
        checkpoint_dir=path, **kwargs)


def observe_run(problem, path, estimator=None, grid=None, **kwargs):
    visits, predictions, statistics = [], {}, []
    original_run = study._run_representation_fold
    original_predict = _KNNQueryReuse.predict
    original_screen_fit = BinaryScreening.fit

    def run(*args, **options):
        record = args[9]
        visits.append((record["representation_key"], record["fold"]))
        return original_run(*args, **options)

    def predict(reuse, k, **options):
        result = original_predict(reuse, k, **options)
        predictions[(*visits[-1], k)] = np.array(result[0], copy=True)
        return result

    def fit(selector, X, y):
        statistics.append((np.array(X, copy=True), np.array(y, copy=True)))
        return original_screen_fit(selector, X, y)

    with patch.object(study, "_run_representation_fold", run), \
            patch.object(_KNNQueryReuse, "predict", predict), \
            patch.object(BinaryScreening, "fit", fit):
        report = run_grid(problem, path, estimator, grid, **kwargs)
    return report, visits, predictions, statistics


def representation_keys(report):
    return list(dict.fromkeys(row["representation_key"]
                              for row in report["representation_folds"]))


def canonical_schedule(report):
    return [(key, fold) for key in representation_keys(report) for fold in range(3)]


def fold_schedule(report):
    return [(key, fold) for fold in range(3) for key in representation_keys(report)]


def untimed(value):
    """Remove measured costs and checkpoint addresses, retaining numerical data."""
    if isinstance(value, dict):
        return {key: untimed(item) for key, item in value.items()
                if not key.endswith("_seconds")
                and key not in {"fit_times", "identity", "key", "shared_representation_key"}}
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def assert_same_results(expected, actual):
    for field in ("candidates", "ranked_candidate_indices", "selected_parameters",
                  "selected_mean_score", "scoring", "refinement", "screening_diagnostics",
                  "unique_representations", "unique_representation_fits",
                  "preflight_rejected_representation_folds"):
        assert untimed(actual[field]) == untimed(expected[field]), field
    assert [(row["representation_key"], row["fold"])
            for row in actual["representation_folds"]] == canonical_schedule(actual)
    for candidate in actual["candidates"]:
        assert [fold["fold"] for fold in candidate["folds"]] == [0, 1, 2]


def assert_accounting(report):
    rows = report["representation_folds"]
    mappings = {
        "screening_statistics_seconds": "screening_statistics_seconds",
        "screening_selection_seconds": "screening_selection_seconds",
        "representation_fit_seconds": "representation_fit_seconds",
        "validation_projection_seconds": "projection_seconds",
        "knn_reuse_setup_seconds": "neighbor_reuse_setup_seconds",
        "knn_shared_query_seconds": "neighbor_query_seconds",
    }
    for total, per_fold in mappings.items():
        assert report["compute_seconds"][total] == sum(row[per_fold] for row in rows)
    for total, per_head in (("knn_fit_seconds", "fit_seconds"),
                            ("knn_prediction_seconds", "prediction_seconds")):
        assert report["compute_seconds"][total] == sum(
            head[per_head] for row in rows for head in row["neighbors"].values())
    assert report["tuning_seconds"] == sum(report["compute_seconds"].values())
    assert report["screening_statistics_fits"] == sum(
        row["screening_statistics_fit_performed"] for row in rows)
    assert report["unique_representation_fits"] == sum(
        row["representation_fit_performed"] for row in rows)
    assert report["schema_version"] == report["identity"]["schema_version"] == 4


def assert_order_metadata(report, requested, resolved, reason=None):
    metadata = {"requested": requested, "resolved": resolved, "fallback_reason": reason}
    assert report["execution_order"] == metadata
    assert report["identity"]["execution_order"] == metadata


def receipt_bytes(directory, section="folds"):
    return {path: path.read_bytes() for path in (directory / section).glob("*.json")}


def assert_unchanged(snapshot):
    assert all(path.read_bytes() == content for path, content in snapshot.items())


@pytest.mark.parametrize("order", [None, True, 1, "folds", "FOLD", b"fold",
                                   ["fold"], np.array("fold"), StringSubclass("fold")])
def test_invalid_order_rejected_before_runtime_or_checkpoint_effects(problem, tmp_path, order):
    destination = tmp_path / "must-not-exist"
    with patch.object(study, "_runtime_identity", side_effect=AssertionError("runtime touched")), \
            patch.object(Path, "mkdir", side_effect=AssertionError("directory touched")), \
            pytest.raises((TypeError, ValueError), match="execution_order"):
        run_grid(problem, destination, execution_order=order)
    assert not destination.exists()


def test_default_preserves_schedule_and_skips_eligibility_check(problem, tmp_path):
    with patch.object(study, "_fold_order_eligible", side_effect=AssertionError("default inspected eligibility")):
        report, visits, _, _ = observe_run(problem, tmp_path)
        with patch.object(study, "_run_representation_fold", side_effect=AssertionError("complete resume fitted")):
            resumed = run_grid(problem, tmp_path, execution_order="representation")
    assert visits == canonical_schedule(report)
    assert_order_metadata(report, "representation", "representation")
    assert resumed["fingerprint"] == report["fingerprint"]
    assert resumed["new_representation_fits"] == 0


@pytest.mark.parametrize("order", [np.str_("representation"), np.str_("fold")])
def test_numpy_strings_normalize_and_resume_with_python_strings(problem, tmp_path, order):
    report = run_grid(problem, tmp_path, execution_order=order)
    resumed = run_grid(problem, tmp_path, execution_order=str(order))
    assert_order_metadata(report, str(order), str(order))
    assert type(report["identity"]["execution_order"]["requested"]) is str
    assert resumed["fingerprint"] == report["fingerprint"]
    assert resumed["new_representation_fits"] == 0


@pytest.mark.parametrize("layout,dtype,scoring", [
    ("screen", np.float64, "accuracy"),
    ("pca", np.float32, "balanced_accuracy"),
    ("spa", np.float32, "accuracy"),
    ("cd", np.float64, "balanced_accuracy"),
    ("mu", np.float32, "accuracy"),
])
def test_fold_order_preserves_predictions_masks_ranking_and_counts(problem, tmp_path, layout, dtype, scoring):
    X, y = problem
    problem = (X.astype(dtype), np.where(y, "present", "absent"))
    if layout == "screen":
        estimator = screened_pipeline(weights="distance")
        grid = {"screen__score_threshold": [0, 2], "knn__n_neighbors": [1, 3]}
    elif layout == "pca":
        estimator = Pipeline([
            ("screen", BinaryScreening(screening="percentile", percentile=100)),
            ("pca", PCA(svd_solver="full", whiten=True, random_state=7)),
            ("knn", KNeighborsClassifier(weights="distance", algorithm="brute")),
        ])
        grid = {"pca__n_components": [1, 2], "knn__n_neighbors": [1, 3]}
    else:
        estimator = NMFClassifier(
            solver=layout, screening="absolute", score_threshold=0,
            random_state=7, max_iter=5000, tol=1e-3, weights="distance",
            knn_normalization="sd", decision_threshold=.43, positive_label="absent")
        # On this positive fixture, rank-one NNDSVDa already starts near its
        # optimum; CD's relative initial-violation stopping test can then stay
        # at roundoff until the cap. Use two nontrivial CD ranks for the valid
        # prediction oracle, keeping the same tolerance and iteration limit.
        grid = {"n_components": [2, 3] if layout == "cd" else [1, 2],
                "n_neighbors": [1, 3]}
    original = observe_run(problem, tmp_path / "representation", estimator, grid, scoring=scoring)
    reordered = observe_run(problem, tmp_path / "fold", estimator, grid,
                            execution_order="fold", scoring=scoring)
    expected, expected_visits, expected_predictions, _ = original
    actual, visits, predictions, statistics = reordered
    assert expected_visits == canonical_schedule(expected)
    assert visits == fold_schedule(actual)
    assert visits != expected_visits
    assert_order_metadata(actual, "fold", "fold")
    assert_same_results(expected, actual)
    assert untimed(actual["representation_folds"]) == untimed(expected["representation_folds"])
    assert predictions.keys() == expected_predictions.keys()
    for key in predictions:
        np.testing.assert_array_equal(predictions[key], expected_predictions[key])
    assert all(candidate["valid"] for candidate in actual["candidates"])
    assert actual["new_representation_fits"] == actual["unique_representation_fits"] == 6
    assert actual["screening_statistics_fits"] == len(statistics) == 3
    for (seen_X, seen_y), fold in zip(statistics, actual["identity"]["folds"]):
        training = fold["training_indices"]
        np.testing.assert_array_equal(seen_X, problem[0][training])
        np.testing.assert_array_equal(seen_y, problem[1][training])
    for row in actual["representation_folds"]:
        train = actual["identity"]["folds"][row["fold"]]["training_indices"]
        params = row["parameters"]
        selector = BinaryScreening(
            screening="percentile" if layout == "pca" else "absolute",
            percentile=100, score_threshold=params.get("screen__score_threshold", 0))
        selector.fit(problem[0][train], problem[1][train])
        assert row["diagnostics"]["selected_feature_indices"] == np.flatnonzero(selector.support_).tolist()
    assert_accounting(expected)
    assert_accounting(actual)


@pytest.mark.parametrize("first,second", [("representation", "fold"), ("fold", "representation")])
def test_requested_order_is_checkpoint_identity_even_when_both_resolve_to_representation(
        problem, tmp_path, first, second):
    grid = {"knn__n_neighbors": [1, 3]}
    report = run_grid(problem, tmp_path, grid=grid, execution_order=first)
    assert_order_metadata(report, first, "representation",
                          "single_representation" if first == "fold" else None)
    snapshot = receipt_bytes(tmp_path)
    snapshot.update({tmp_path / name: (tmp_path / name).read_bytes()
                     for name in ("identity.json", "report.json")})
    with patch.object(study, "_run_representation_fold", side_effect=AssertionError("incompatible run fitted")), \
            pytest.raises(ValueError, match="incompatible"):
        run_grid(problem, tmp_path, grid=grid, execution_order=second)
    assert_unchanged(snapshot)


def test_single_unsupported_representation_has_single_representation_reason(problem, tmp_path):
    report = run_grid(problem, tmp_path, KNeighborsClassifier(), {"n_neighbors": [1, 3]},
                      execution_order="fold")
    assert_order_metadata(report, "fold", "representation", "single_representation")


@pytest.mark.parametrize("case", ["callable_weights", "callable_metric", "manhattan", "minkowski_p1",
                                  "metric_params", "custom_screen", "extra_preprocessing",
                                  "mixed_screen", "mixed_nmf", "invalid_screen", "observable_scalar"])
def test_unsupported_grids_preserve_original_order_and_failure_semantics(problem, tmp_path, case):
    estimator = screened_pipeline()
    grid = {"screen__score_threshold": [0, 2], "knn__n_neighbors": [1, 3]}
    if case == "callable_weights":
        estimator.set_params(knn__weights=stateful_weights)
    elif case == "callable_metric":
        estimator.set_params(knn__metric=custom_metric)
    elif case == "manhattan":
        estimator.set_params(knn__metric="manhattan")
    elif case == "minkowski_p1":
        estimator.set_params(knn__p=1)
    elif case == "metric_params":
        estimator.set_params(knn__metric_params={"p": 2})
    elif case == "custom_screen":
        estimator.set_params(screen=CustomScreen())
    elif case == "extra_preprocessing":
        estimator.steps.insert(0, ("scale", StandardScaler(with_mean=False)))
    elif case == "mixed_screen":
        grid = {"screen": [BinaryScreening(score_threshold=0), CustomScreen(score_threshold=0)],
                "knn__n_neighbors": [1, 3]}
    elif case == "mixed_nmf":
        estimator = NMFClassifier(n_components=2, random_state=7, score_threshold=0)
        grid = {"screening": ["absolute", "none"], "n_neighbors": [1, 3]}
    elif case == "invalid_screen":
        estimator.set_params(screen__screening="percentile", screen__percentile=-1)
    elif case == "observable_scalar":
        estimator.set_params(screen__percentile=ObservableFloat(100))
    _weight_calls.clear()
    _scalar_calls.clear()
    expected, expected_visits, expected_predictions, _ = observe_run(
        problem, tmp_path / "default", estimator, grid)
    expected_weight_calls = list(_weight_calls)
    expected_scalar_calls = list(_scalar_calls)
    _weight_calls.clear()
    _scalar_calls.clear()
    actual, visits, predictions, _ = observe_run(
        problem, tmp_path / "fold", estimator, grid, execution_order="fold")
    assert visits == expected_visits == canonical_schedule(actual)
    assert_order_metadata(actual, "fold", "representation", "unsupported_configuration")
    assert_same_results(expected, actual)
    assert untimed(actual["representation_folds"]) == untimed(expected["representation_folds"])
    for key in expected_predictions:
        np.testing.assert_array_equal(predictions[key], expected_predictions[key])
    if case == "callable_weights":
        assert expected_weight_calls
        assert _weight_calls == expected_weight_calls
    if case == "observable_scalar":
        # Establish that the original path accepts the value, then require that
        # eligibility checking adds no calls to its comparison hook.
        assert all(candidate["valid"] for candidate in expected["candidates"])
        assert expected_scalar_calls
        assert _scalar_calls == expected_scalar_calls
    if case.startswith("invalid_"):
        assert not any(candidate["valid"] for candidate in actual["candidates"])
    assert_accounting(actual)


def test_repeated_partial_resume_retains_receipts_and_recomputes_only_missing_folds(problem, tmp_path):
    destination = tmp_path / "resumed"
    original_run = study._run_representation_fold
    completed_visits, snapshots = [], []
    for allowed in (1, 2):
        visits = []

        def interrupt(*args, **kwargs):
            record = args[9]
            visits.append((record["representation_key"], record["fold"]))
            if len(visits) > allowed:
                raise KeyboardInterrupt("controlled interruption")
            return original_run(*args, **kwargs)

        with patch.object(study, "_run_representation_fold", interrupt), \
                pytest.raises(KeyboardInterrupt, match="controlled interruption"):
            run_grid(problem, destination, execution_order="fold")
        for snapshot in snapshots:
            assert_unchanged(snapshot)
        snapshots.append(receipt_bytes(destination))
        completed_visits.extend(visits[:-1])
    attempts_before = receipt_bytes(destination, "attempts")
    report, remaining, _, _ = observe_run(problem, destination, execution_order="fold")
    assert completed_visits + remaining == fold_schedule(report)
    assert report["new_representation_fits"] == report["reused_representation_folds"] == 3
    assert report["unique_representation_fits"] == 6
    # Interrupting within a fold loses the in-memory screen, so later receipts
    # retain the actual cost of recomputing it instead of claiming three fits.
    assert report["screening_statistics_fits"] == 5
    assert_unchanged(attempts_before)
    for snapshot in snapshots:
        assert_unchanged(snapshot)
    attempts = [json.loads(content)["payload"] for content in receipt_bytes(destination, "attempts").values()]
    assert sum(row["state"] == "interrupted" for row in attempts) == 2
    reference = run_grid(problem, tmp_path / "fresh", execution_order="fold")
    assert_same_results(reference, report)
    assert_accounting(report)
    completed = receipt_bytes(destination)
    with patch.object(study, "_run_representation_fold", side_effect=AssertionError("completed fold rerun")), \
            patch.object(BinaryScreening, "fit", side_effect=AssertionError("completed screen refitted")), \
            patch.object(_KNNQueryReuse, "predict", side_effect=AssertionError("completed neighbors queried")):
        resumed = run_grid(problem, destination, execution_order="fold")
    assert resumed["new_representation_fits"] == 0
    assert resumed["reused_representation_folds"] == 6
    assert resumed["candidates"] == report["candidates"]
    assert resumed["compute_seconds"] == report["compute_seconds"]
    assert_unchanged(completed)


def test_holey_resume_visits_only_missing_receipts_in_fold_order(problem, tmp_path):
    report = run_grid(problem, tmp_path, execution_order="fold")
    schedule = fold_schedule(report)
    missing = {schedule[0], schedule[-1]}
    for path in (tmp_path / "folds").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))["payload"]
        if (row["representation_key"], row["fold"]) in missing:
            path.unlink()
    preserved = receipt_bytes(tmp_path)
    resumed, visits, _, _ = observe_run(problem, tmp_path, execution_order="fold")
    assert visits == [visit for visit in schedule if visit in missing]
    assert resumed["new_representation_fits"] == 2
    assert resumed["reused_representation_folds"] == 4
    assert_same_results(report, resumed)
    assert_accounting(resumed)
    assert_unchanged(preserved)


@pytest.mark.parametrize("case", ["empty_screen", "impossible_rank", "impossible_k",
                                  "nonconvergence", "invalid_algorithm"])
def test_fold_order_preserves_completed_failure_receipts_and_reuses_them(problem, tmp_path, case):
    estimator = screened_pipeline()
    grid = {"screen__score_threshold": [0, 1e20], "knn__n_neighbors": [1, 3]}
    if case == "impossible_rank":
        estimator.steps.insert(1, ("pca", PCA(svd_solver="full", random_state=7)))
        grid = {"pca__n_components": [2, 99], "knn__n_neighbors": [1, 3]}
    elif case == "impossible_k":
        grid = {"screen__score_threshold": [0, 2], "knn__n_neighbors": [1, 100]}
    elif case == "nonconvergence":
        estimator = NMFClassifier(solver="cd", random_state=7, max_iter=1, tol=1e-12, score_threshold=0)
        grid = {"n_components": [2, 3], "n_neighbors": [1, 3]}
    elif case == "invalid_algorithm":
        estimator.set_params(knn__algorithm="unknown")
        grid = {"screen__score_threshold": [0, 2], "knn__n_neighbors": [1, 3]}
    expected = run_grid(problem, tmp_path / "default", estimator, grid)
    actual, visits, _, _ = observe_run(problem, tmp_path / "fold", estimator, grid, execution_order="fold")
    assert_order_metadata(actual, "fold", "fold")
    assert visits == fold_schedule(actual)
    assert_same_results(expected, actual)
    assert untimed(actual["representation_folds"]) == untimed(expected["representation_folds"])
    invalid = [candidate for candidate in actual["candidates"] if not candidate["valid"]]
    assert invalid
    assert all(candidate["mean_score"] is None and candidate["invalid_fold_count"] == 3
               for candidate in invalid)
    assert all(row["state"] == "completed" for row in actual["representation_folds"])
    if case in {"empty_screen", "impossible_rank"}:
        assert actual["preflight_rejected_representation_folds"] == 3
        assert actual["unique_representation_fits"] == 3
    snapshot = receipt_bytes(tmp_path / "fold")
    with patch.object(study, "_run_representation_fold", side_effect=AssertionError("failure receipt rerun")):
        resumed = run_grid(problem, tmp_path / "fold", estimator, grid, execution_order="fold")
    assert resumed["new_representation_fits"] == 0
    assert resumed["candidates"] == actual["candidates"]
    assert resumed["compute_seconds"] == actual["compute_seconds"]
    assert_accounting(actual)
    assert_unchanged(snapshot)


@pytest.mark.parametrize("when", ["during_fit", "before_reuse"])
def test_fold_order_retains_runtime_guards_around_new_and_cached_folds(problem, tmp_path, when):
    if when == "before_reuse":
        run_grid(problem, tmp_path, execution_order="fold")
    preserved = receipt_bytes(tmp_path)
    original_runtime = study._runtime_identity
    count = 0

    def change_runtime(*args, **kwargs):
        nonlocal count
        count += 1
        runtime = original_runtime(*args, **kwargs)
        if count >= (3 if when == "during_fit" else 2):
            runtime = {**runtime, "test_runtime_change": True}
        return runtime

    with patch.object(study, "_runtime_identity", change_runtime), \
            pytest.raises(ValueError, match="Source or numerical runtime changed"):
        run_grid(problem, tmp_path, execution_order="fold")
    assert_unchanged(preserved)
    if when == "during_fit":
        assert not receipt_bytes(tmp_path)
        attempts = [json.loads(content)["payload"] for content in receipt_bytes(tmp_path, "attempts").values()]
        assert len(attempts) == 1
        assert attempts[0]["state"] == "failed"
        assert attempts[0]["attempt_error_type"] == "ValueError"
