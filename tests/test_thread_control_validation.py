"""Thread scalar contracts at real execution and checkpoint boundaries.

Only limits 1 and 2 reach actual numerical libraries. C-int boundary tests stop
at the package's threadpoolctl entry point, before any native setter is called.
The small KNN/PCA examples test software behavior, not predictive performance.
"""

from enum import IntEnum
import json
import random
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf import evaluation, search, study, validation


class ThreadCount(IntEnum):
    ONE = 1
    TWO = 2


C_INT_MAX = int(np.iinfo(np.intc).max)
APIS = ("generic", "evaluation", "shared", "rank", "comparison")


def _legacy_unitless_duration():
    # Construct the legacy input deliberately; NumPy versions may deprecate it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return np.timedelta64(1)


INVALID_THREADS = [
    pytest.param(True, id="python-true"),
    pytest.param(False, id="python-false"),
    pytest.param(np.bool_(True), id="numpy-true"),
    pytest.param(np.bool_(False), id="numpy-false"),
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(np.int64(0), id="numpy-zero"),
    pytest.param(np.int64(-1), id="numpy-negative"),
    pytest.param(1.0, id="integral-float"),
    pytest.param(1.5, id="fractional-float"),
    pytest.param(np.float64(1), id="numpy-float"),
    pytest.param(np.nan, id="nan"),
    pytest.param(np.inf, id="infinity"),
    pytest.param(1 + 0j, id="complex"),
    pytest.param(np.array(1), id="scalar-array"),
    pytest.param(np.array([1]), id="vector-array"),
    pytest.param("1", id="string"),
    pytest.param(None, id="none"),
    pytest.param(np.timedelta64(1, "ns"), id="nanosecond-duration"),
    pytest.param(np.timedelta64(1, "D"), id="day-duration"),
    pytest.param(_legacy_unitless_duration(), id="unitless-duration"),
    pytest.param(np.timedelta64("NaT", "ns"), id="duration-nat"),
    pytest.param(np.datetime64("2000-01-01"), id="datetime"),
    pytest.param(np.datetime64("NaT", "ns"), id="datetime-nat"),
    pytest.param(C_INT_MAX + 1, id="above-c-int"),
    pytest.param(np.uint64(C_INT_MAX + 1), id="numpy-above-c-int"),
    pytest.param(np.uint64(2**64 - 1), id="numpy-uint64-max"),
    pytest.param(2**100, id="huge-python-int"),
]


@pytest.fixture
def tiny_data():
    # No RNG construction is needed to prepare invalid-input probes. Every
    # explicit training/validation fold contains both classes with support >=2.
    rows = np.arange(16, dtype=float)
    y = np.arange(16) % 2
    X = np.column_stack((3 * y + rows / 40, rows % 3, rows / 16))
    first, second = np.arange(8), np.arange(8, 16)
    folds = [(first, second), (second, first)]
    return X, y, folds


@pytest.fixture
def one_native_thread():
    # Numerical callers below may explicitly request 2, then restore this 1.
    with threadpool_limits(limits=1):
        yield


def _model_and_grid(api):
    if api == "rank":
        model = Pipeline([
            ("pca", PCA(n_components=1, svd_solver="full", random_state=17)),
            ("knn", KNeighborsClassifier(algorithm="brute")),
        ])
        return model, {"pca__n_components": [1], "knn__n_neighbors": [1, 3]}
    return KNeighborsClassifier(algorithm="brute"), {"n_neighbors": [1, 3]}


def _invoke(api, threads, data, path, *, fitted=None):
    X, y, folds = data
    model, grid = _model_and_grid(api)
    if api == "generic":
        return evaluation.calibration_search(
            model, grid, X, y, cv=2, folds=folds, seed=17,
            threads=threads, checkpoint_path=path / "generic.json",
        )
    if api == "shared":
        return study.calibration_knn_grid(
            model, grid, X, y, cv=2, folds=folds, seed=17,
            threads=threads, checkpoint_dir=path / "shared",
        )
    if api == "rank":
        return search.calibration_rank_search(
            model, grid, X, y, cv=2, folds=folds, seed=17,
            threads=threads, checkpoint_dir=path / "rank",
            max_expansions=0, max_rank=1,
        )
    if api == "evaluation":
        if fitted is None:
            # Invalid-input/boundary probes need class metadata but must not fit.
            model.classes_ = np.array([0, 1])
            fitted = model
        return evaluation.evaluate_classifier(fitted, X, y, threads=threads)
    return evaluation.run_comparison(
        {"knn": (model, grid)}, X, y, X + .01, y,
        output_dir=path / "comparison", seed=17, threads=threads,
    )


@pytest.mark.parametrize("value", [
    1, 2, np.int8(1), np.int16(1), np.int32(1), np.int64(1),
    np.uint8(1), np.uint16(1), np.uint32(1), np.uint64(1),
    ThreadCount.ONE, ThreadCount.TWO,
    C_INT_MAX, np.int64(C_INT_MAX), np.uint64(C_INT_MAX),
])
def test_thread_validator_returns_a_builtin_int(value):
    result = validation._validate_threads(value)
    assert type(result) is int
    assert result == int(value)


@pytest.mark.parametrize("value", INVALID_THREADS)
def test_thread_validator_rejects_noncounts_and_unrepresentable_values(value):
    with pytest.raises(ValueError, match="threads"):
        validation._validate_threads(value)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("value", INVALID_THREADS)
def test_invalid_threads_fail_before_rng_runtime_candidates_fit_or_files(
        tiny_data, tmp_path, monkeypatch, api, value):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"Invalid threads reached {name}.")
        return fail

    # These guards fail at the earliest effect boundary; they do not supply a
    # fake result which could make an invalid invocation appear to succeed.
    for module in (evaluation, study, search):
        for name in ("build_cv_folds", "ParameterGrid", "clone",
                     "_search_identity", "_initialized_provenance",
                     "_runtime_identity", "threadpool_limits", "write_json",
                     "_write_receipt", "_checkpoint_file", "_checkpoint_lock",
                     "_run_representation_fold"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden(f"{module.__name__}.{name}"))
    for name in ("fit", "predict", "predict_proba"):
        monkeypatch.setattr(KNeighborsClassifier, name, forbidden(f"knn.{name}"))
    for name in ("seed", "RandomState", "default_rng"):
        monkeypatch.setattr(np.random, name, forbidden(f"numpy.random.{name}"))
    monkeypatch.setattr(random, "seed", forbidden("random.seed"))

    destination = tmp_path / "untouched"
    with pytest.raises(ValueError, match="threads"):
        _invoke(api, value, tiny_data, destination)
    assert calls == []
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


class _NativeBoundaryReached(Exception):
    pass


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("value", [
    1, np.int64(2), ThreadCount.TWO,
    C_INT_MAX, np.int64(C_INT_MAX), np.uint64(C_INT_MAX),
])
def test_accepted_counts_are_canonical_before_native_boundary(
        tiny_data, tmp_path, monkeypatch, api, value):
    observed = []

    def intercept(*, limits, **kwargs):
        observed.append(limits)
        # Always stop, including small inputs. The upper bound must NEVER reach
        # threadpoolctl or any loaded native runtime through this regression.
        raise _NativeBoundaryReached

    monkeypatch.setattr(evaluation, "threadpool_limits", intercept)
    monkeypatch.setattr(study, "threadpool_limits", intercept)
    with pytest.raises(_NativeBoundaryReached):
        _invoke(api, value, tiny_data, tmp_path)
    assert observed == [int(value)]
    assert type(observed[0]) is int


def _independent_fold_scores(api, data):
    X, y, folds = data
    model, grid = _model_and_grid(api)
    return [
        [accuracy_score(y[valid], clone(model).set_params(**params)
                        .fit(X[train], y[train]).predict(X[valid]))
         for train, valid in folds]
        for params in ParameterGrid(grid)
    ]


@pytest.mark.usefixtures("one_native_thread")
@pytest.mark.parametrize("api", ["generic", "shared", "rank"])
@pytest.mark.parametrize("value", [np.int64(1), np.uint64(2), ThreadCount.TWO])
def test_real_integer_counts_match_independent_sklearn_folds(
        tiny_data, tmp_path, api, value):
    expected = _independent_fold_scores(api, tiny_data)
    result = _invoke(api, value, tiny_data, tmp_path)
    report = result.report if api == "generic" else result
    assert all(row["valid"] for row in report["candidates"])
    np.testing.assert_array_equal(
        [[fold["score"] for fold in row["folds"]] for row in report["candidates"]],
        expected,
    )
    expected_winner = int(np.argmax(np.mean(expected, axis=1)))
    _, grid = _model_and_grid(api)
    assert report["selected_parameters"] == list(ParameterGrid(grid))[expected_winner]
    assert report["selected_mean_score"] == np.mean(expected[expected_winner])
    identity = report["identity"] if api == "generic" else report["identity"]["calibration"]
    assert identity["threads"] == int(value)
    assert type(identity["threads"]) is int
    if api == "generic":
        X, y, _ = tiny_data
        model, grid = _model_and_grid(api)
        reference = clone(model).set_params(**list(ParameterGrid(grid))[expected_winner]).fit(X, y)
        np.testing.assert_array_equal(result.estimator.predict(X + .01), reference.predict(X + .01))
        assert report["provenance"]["native_threads_requested"] == int(value)
    else:
        assert report["new_representation_fits"] == 2


@pytest.mark.usefixtures("one_native_thread")
@pytest.mark.parametrize("api", ["generic", "shared", "rank"])
def test_equivalent_integer_types_resume_the_same_checkpoint(
        tiny_data, tmp_path, monkeypatch, api):
    observed_fits = []
    original_fit = KNeighborsClassifier.fit

    def observe_fit(self, X, y):
        observed_fits.append(len(y))
        return original_fit(self, X, y)

    monkeypatch.setattr(KNeighborsClassifier, "fit", observe_fit)
    first = _invoke(api, 1, tiny_data, tmp_path)
    first_report = first.report if api == "generic" else first
    # Shared paths fit one carrier plus both KNN heads in each fold.
    assert len(observed_fits) == (5 if api == "generic" else 6)
    original_folds = [row["folds"] for row in first_report["candidates"]]
    for value in (np.int64(1), np.uint64(1), ThreadCount.ONE):
        observed_fits.clear()
        resumed = _invoke(api, value, tiny_data, tmp_path)
        report = resumed.report if api == "generic" else resumed
        assert report["fingerprint"] == first_report["fingerprint"]
        assert report["selected_parameters"] == first_report["selected_parameters"]
        assert [row["folds"] for row in report["candidates"]] == original_folds
        if api == "generic":
            # Generic checkpoints save CV records, not the fitted final model.
            assert observed_fits == [len(tiny_data[1])]
            assert resumed.estimator is not None
        else:
            assert observed_fits == []
            assert report["new_representation_fits"] == 0
            assert report["reused_representation_folds"] == 2
    with pytest.raises(ValueError, match="incompatible"):
        _invoke(api, 2, tiny_data, tmp_path)


@pytest.mark.usefixtures("one_native_thread")
@pytest.mark.parametrize("value", [np.int64(1), np.uint64(2), ThreadCount.TWO])
def test_evaluation_integer_counts_preserve_real_predictions(tiny_data, tmp_path, value):
    X, y, _ = tiny_data
    fitted = KNeighborsClassifier(n_neighbors=3, algorithm="brute").fit(X, y)
    result = _invoke("evaluation", value, tiny_data, tmp_path, fitted=fitted)
    reference = evaluation.evaluate_classifier(fitted, X, y, threads=int(value))
    assert result["status"] == reference["status"] == "valid"
    assert result["metrics"] == reference["metrics"]
    assert result["predictions"] == reference["predictions"]
    np.testing.assert_array_equal(result["predictions"]["predicted_labels"], fitted.predict(X))
    np.testing.assert_array_equal(
        result["predictions"]["positive_class_probability"], fitted.predict_proba(X)[:, 1],
    )


@pytest.mark.usefixtures("one_native_thread")
@pytest.mark.parametrize("first_value", [np.int64(1), ThreadCount.ONE])
def test_comparison_normalizes_numpy_and_enum_and_reuses_completed_evaluation(
        tiny_data, tmp_path, monkeypatch, first_value):
    X, y, _ = tiny_data
    first = _invoke("comparison", first_value, tiny_data, tmp_path)
    record = first["models"]["knn"]
    assert record["evaluation"]["status"] == "valid"
    model, grid = _model_and_grid("comparison")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=17).split(X, y))
    scores = _independent_fold_scores("generic", (X, y, folds))
    winner = int(np.argmax(np.mean(scores, axis=1)))
    expected = clone(model).set_params(**list(ParameterGrid(grid))[winner]).fit(X, y)
    assert record["selected_parameters"] == list(ParameterGrid(grid))[winner]
    np.testing.assert_array_equal(
        record["evaluation"]["predictions"]["predicted_labels"], expected.predict(X + .01),
    )
    saved = json.loads((tmp_path / "comparison" / "knn-cv.json").read_text(encoding="utf-8"))
    assert saved["identity"]["threads"] == 1
    assert type(saved["identity"]["threads"]) is int

    def forbidden(*args, **kwargs):
        raise AssertionError("A completed comparison must not fit or predict on resume.")

    for method in ("fit", "predict", "predict_proba"):
        monkeypatch.setattr(KNeighborsClassifier, method, forbidden)
    for value in (1, np.uint64(1), ThreadCount.ONE):
        resumed = _invoke("comparison", value, tiny_data, tmp_path)
        assert resumed == first


@pytest.mark.parametrize("option", ["max_rank", "max_expansions", "explicit_rank"])
def test_native_thread_bound_does_not_cap_rank_policy_integers(
        tiny_data, tmp_path, monkeypatch, option):
    # Rank-policy scalars have no native setter transport requirement. Reach
    # the first runtime boundary safely without allocating or fitting big ranks.
    observed = []

    def intercept(*, limits, **kwargs):
        observed.append(limits)
        raise _NativeBoundaryReached

    monkeypatch.setattr(evaluation, "threadpool_limits", intercept)
    monkeypatch.setattr(study, "threadpool_limits", intercept)
    X, y, folds = tiny_data
    model, grid = _model_and_grid("rank")
    options = {"max_expansions": 0}
    if option == "explicit_rank":
        grid["pca__n_components"] = [C_INT_MAX + 1]
    else:
        options[option] = C_INT_MAX + 1
    with pytest.raises(_NativeBoundaryReached):
        search.calibration_rank_search(
            model, grid, X, y, cv=2, folds=folds, threads=1,
            checkpoint_dir=tmp_path / "not-created", **options,
        )
    assert observed == [1]
    assert list(tmp_path.iterdir()) == []
