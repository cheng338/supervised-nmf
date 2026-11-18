"""Scalar controls preserve seed domains, RNG state and checkpoint semantics."""

from contextlib import ExitStack
import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf import evaluation, search, study
from supervised_nmf.validation import (
    _validate_fixed_seed, _validate_refit, build_cv_folds,
)


@pytest.fixture
def data():
    y = np.arange(30) % 2
    X = np.random.RandomState(744).uniform(.1, 1., (30, 5))
    X[:, 0] += .1 * y
    return X, y


@pytest.fixture
def preserve_global_rng():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def raw_model():
    return KNeighborsClassifier(n_neighbors=1, algorithm="brute", n_jobs=1)


def model_and_grid(kind):
    if kind == "rank":
        return Pipeline([
            ("pca", PCA(n_components=2, svd_solver="full", random_state=5)),
            ("knn", raw_model()),
        ]), {"pca__n_components": [2], "knn__n_neighbors": [1]}
    return raw_model(), {"n_neighbors": [1]}


def execute(kind, data, destination, *, seed=17, folds=None, refit=False):
    X, y = data
    model, grid = model_and_grid(kind)
    options = dict(cv=3, seed=seed, threads=1, folds=folds)
    if kind == "generic":
        return evaluation.calibration_search(
            model, grid, X, y, checkpoint_path=destination, refit=refit, **options,
        ).report
    if kind == "shared":
        return study.calibration_knn_grid(
            model, grid, X, y, checkpoint_dir=destination, **options,
        )
    return search.calibration_rank_search(
        model, grid, X, y, checkpoint_dir=destination, max_expansions=0, **options,
    )


def assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def assert_folds_equal(actual, expected):
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected):
        if isinstance(observed, dict):
            observed = observed["training_indices"], observed["validation_indices"]
        np.testing.assert_array_equal(observed[0], reference[0])
        np.testing.assert_array_equal(observed[1], reference[1])


def forbid_calls(stack, targets):
    return [stack.enter_context(patch(target, side_effect=AssertionError(target)))
            for target in targets]


INVALID_REFIT = [
    pytest.param(0, id="zero"), pytest.param(1, id="one"),
    pytest.param(np.int64(1), id="numpy-int"), pytest.param(1., id="float"),
    pytest.param(None, id="none"), pytest.param("yes", id="truthy-string"),
    pytest.param("", id="empty-string"), pytest.param([], id="list"),
    pytest.param(np.array(True), id="zero-d-array"),
    pytest.param(np.array([True]), id="one-item-array"),
    pytest.param(np.array([True, False]), id="multi-item-array"),
]


@pytest.mark.parametrize("value", INVALID_REFIT)
@pytest.mark.parametrize("seed_kind", ["global", "randomstate"])
def test_invalid_refit_stops_before_rng_runtime_checkpoint_or_fit(
        value, seed_kind, data, tmp_path, preserve_global_rng):
    np.random.seed(91)
    seed = None if seed_kind == "global" else np.random.RandomState(91)
    rng = np.random if seed is None else seed
    before = rng.get_state()
    destination = tmp_path / "untouched" / "cv.json"
    with ExitStack() as stack:
        sentinels = forbid_calls(stack, [
            "supervised_nmf.evaluation.build_cv_folds",
            "supervised_nmf.evaluation._initialized_provenance",
            "supervised_nmf.evaluation.threadpool_limits",
            "supervised_nmf.evaluation._checkpoint_file",
            "sklearn.neighbors.KNeighborsClassifier.fit",
        ])
        with pytest.raises(ValueError, match="refit.*boolean"):
            execute("generic", data, destination, seed=seed, refit=value)
        for sentinel in sentinels:
            sentinel.assert_not_called()
    assert_rng_equal(rng.get_state(), before)
    assert not destination.parent.exists()


@pytest.mark.parametrize("value", [False, True, np.bool_(False), np.bool_(True)])
def test_refit_logical_scalars_normalize_to_python_bool(value):
    normalized = _validate_refit(value)
    assert type(normalized) is bool
    assert normalized is bool(value)


@pytest.mark.parametrize("refit", [False, True])
def test_numpy_refit_preserves_state_digest_and_resume_fit_counts(refit, data, tmp_path):
    X, y = data
    destination = tmp_path / "cv.json"
    original_fit = KNeighborsClassifier.fit
    fit_sizes = []

    def counted_fit(model, X_fit, y_fit):
        fit_sizes.append(len(X_fit))
        return original_fit(model, X_fit, y_fit)

    with patch.object(KNeighborsClassifier, "fit", counted_fit):
        first = evaluation.calibration_search(
            raw_model(), {}, X, y, cv=3, seed=17, threads=1,
            refit=refit, checkpoint_path=destination,
        )
        assert fit_sizes == [20] * 3 + ([30] if refit else [])
        first_saved = json.loads(destination.read_text(encoding="utf-8"))
        fit_sizes.clear()
        resumed = evaluation.calibration_search(
            raw_model(), {}, X, y, cv=3, seed=17, threads=1,
            refit=np.bool_(refit), checkpoint_path=destination,
        )
        # A completed generic checkpoint saves CV scores, not the fitted model.
        assert fit_sizes == ([30] if refit else [])
    resumed_saved = json.loads(destination.read_text(encoding="utf-8"))
    assert first.report["fingerprint"] == resumed.report["fingerprint"]
    assert first.report["candidates"] == resumed.report["candidates"]
    for saved in (first_saved, resumed_saved):
        assert saved["identity"]["refit"] is refit
        assert saved["state"] == ("completed" if refit else "cv_completed")
        assert saved["checkpoint_sha256"] == evaluation._digest({
            key: value for key, value in saved.items() if key != "checkpoint_sha256"
        })
    if refit:
        assert len(first_saved["final_fit_attempts"]) == 1
        assert len(resumed_saved["final_fit_attempts"]) == 2
        np.testing.assert_array_equal(first.estimator.predict(X), resumed.estimator.predict(X))
    else:
        assert first.estimator is resumed.estimator is None
        assert resumed.report["refit_performed"] is False
        assert resumed_saved["final_fit_attempts"] == []


INVALID_FIXED_SEEDS = [
    pytest.param(False, id="bool"), pytest.param(np.bool_(True), id="numpy-bool"),
    pytest.param(-1, id="negative"), pytest.param(2 ** 32, id="above-uint32"),
    pytest.param(2 ** 64, id="large-python-int"),
    pytest.param(np.uint64(2 ** 63), id="large-numpy-int"),
    pytest.param(0., id="integral-float"), pytest.param(.5, id="fractional-float"),
    pytest.param(None, id="none"), pytest.param("17", id="string"),
    pytest.param(np.random.RandomState(17), id="randomstate"),
    pytest.param(np.random.default_rng(17), id="generator"),
    pytest.param(np.array(17), id="zero-d-array"),
    pytest.param(np.timedelta64(17, "ns"), id="timedelta-ns"),
    pytest.param(np.timedelta64(17, "D"), id="timedelta-days"),
    pytest.param(np.timedelta64("NaT", "ns"), id="timedelta-nat"),
    pytest.param(np.datetime64("1970-01-01", "ns"), id="datetime"),
    pytest.param(np.datetime64("NaT", "ns"), id="datetime-nat"),
    pytest.param(float("nan"), id="nan"), pytest.param(float("inf"), id="infinity"),
]


@pytest.mark.parametrize("kind", ["shared", "rank"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("seed", INVALID_FIXED_SEEDS)
def test_fixed_seed_rejected_before_folds_source_runtime_or_files(
        kind, explicit, seed, data, tmp_path):
    folds = build_cv_folds(*data, cv=3, seed=17) if explicit else None
    destination = tmp_path / "untouched"
    module = "supervised_nmf." + ("study" if kind == "shared" else "search")
    with ExitStack() as stack:
        sentinels = forbid_calls(stack, [
            module + ".build_cv_folds", module + ".implementation_sources",
            module + "._runtime_identity", "supervised_nmf.study.threadpool_limits",
            "sklearn.neighbors.KNeighborsClassifier.fit",
        ])
        with pytest.raises(ValueError, match="seed.*fixed integer"):
            execute(kind, data, destination, seed=seed, folds=folds)
        for sentinel in sentinels:
            sentinel.assert_not_called()
    assert not destination.exists()


@pytest.mark.parametrize("value", [0, 2 ** 32 - 1, np.int32(0), np.int64(17),
                                  np.uint32(2 ** 32 - 1), np.uint64(2 ** 32 - 1)])
def test_fixed_seed_boundaries_and_numpy_integers_normalize_without_narrowing(value):
    normalized = _validate_fixed_seed(value)
    assert type(normalized) is int
    assert normalized == value


@pytest.mark.parametrize("kind", ["generic", "shared", "rank"])
def test_numpy_fixed_seed_resumes_equivalent_checkpoint_without_fits(kind, data, tmp_path):
    destination = tmp_path / "cached"
    original = execute(kind, data, destination, seed=17)
    assert original["selected_mean_score"] is not None
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("Unexpected fit")) as fit:
        for scalar_type in (np.int32, np.int64, np.uint32, np.uint64):
            resumed = execute(kind, data, destination, seed=scalar_type(17))
            assert resumed["fingerprint"] == original["fingerprint"]
            assert resumed["selected_mean_score"] == original["selected_mean_score"]
        fit.assert_not_called()


def fresh_generic_seed(kind):
    if kind == "none":
        np.random.seed(91)
        return None
    if kind == "module":
        np.random.seed(91)
        return np.random
    if kind == "randomstate":
        return np.random.RandomState(91)
    return {"bool": True, "numpy-int": np.int64(91), "upper-bound": 2 ** 32 - 1}[kind]


@pytest.mark.parametrize("seed_kind", ["none", "module", "randomstate", "bool", "numpy-int", "upper-bound"])
def test_generic_generated_seed_domain_matches_sklearn_folds_and_scores(
        seed_kind, data, preserve_global_rng):
    X, y = data
    seed = fresh_generic_seed(seed_kind)
    expected_folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(X, y))
    expected_scores = [accuracy_score(y[valid], clone(raw_model()).fit(X[train], y[train]).predict(X[valid]))
                       for train, valid in expected_folds]
    actual_folds = build_cv_folds(X, y, cv=3, seed=fresh_generic_seed(seed_kind))
    assert_folds_equal(actual_folds, expected_folds)
    report = execute("generic", data, None, seed=fresh_generic_seed(seed_kind))
    assert_folds_equal(report["folds"], expected_folds)
    assert [fold["score"] for fold in report["candidates"][0]["folds"]] == expected_scores
    assert report["selected_mean_score"] == np.mean(expected_scores)


@pytest.mark.parametrize("seed", [None, False, np.bool_(True), -1, 2 ** 64, 0., "unused",
                                  np.array([1, 2]), np.timedelta64(1, "ns")])
def test_explicit_generic_folds_keep_accepting_unused_seed_values(seed, data):
    folds = build_cv_folds(*data, cv=3, seed=17)
    assert_folds_equal(build_cv_folds(*data, cv=3, seed=seed, folds=folds), folds)
    report = execute("generic", data, None, seed=seed, folds=folds)
    assert_folds_equal(report["folds"], folds)
    assert report["selected_mean_score"] is not None


@pytest.mark.parametrize("kind", ["generic", "shared", "rank"])
def test_changed_unused_explicit_seed_still_rejects_checkpoint(kind, data, tmp_path):
    folds = build_cv_folds(*data, cv=3, seed=17)
    destination = tmp_path / "cached"
    execute(kind, data, destination, seed=17, folds=folds)
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("Unexpected fit")) as fit:
        with pytest.raises(ValueError, match="[Ii]ncompatible"):
            execute(kind, data, destination, seed=18, folds=folds)
        fit.assert_not_called()


@pytest.mark.parametrize("seed", [-1, 2 ** 32, 0., np.bool_(True), "17",
                                  np.random.default_rng(17), np.array(17),
                                  np.timedelta64(17, "ns"), np.datetime64("1970-01-01")])
def test_comparison_invalid_consumed_generic_seed_has_zero_effects(seed, data, tmp_path):
    X, y = data
    destination = tmp_path / "untouched"
    with ExitStack() as stack:
        sentinels = forbid_calls(stack, [
            "supervised_nmf.evaluation._search_identity",
            "supervised_nmf.evaluation._initialized_provenance",
            "supervised_nmf.evaluation.threadpool_limits",
            "supervised_nmf.evaluation.calibration_search",
            "sklearn.neighbors.KNeighborsClassifier.fit",
        ])
        # Preserve sklearn/NumPy's generic error type, including its unsafe
        # scalar-cast TypeError for durations; only fixed-seed APIs standardize it.
        expected_error = TypeError if isinstance(seed, np.timedelta64) else ValueError
        with pytest.raises(expected_error):
            evaluation.run_comparison({"knn": (raw_model(), {})}, X, y, X, y,
                                      output_dir=destination, seed=seed, threads=1)
        for sentinel in sentinels:
            sentinel.assert_not_called()
    assert not destination.exists()


@pytest.mark.parametrize("seed_kind", ["none", "module", "randomstate", "bool"])
def test_comparison_seed_preflight_does_not_draw_or_change_completed_resume(
        seed_kind, data, tmp_path, preserve_global_rng):
    X, y = data
    seed = fresh_generic_seed(seed_kind)
    rng = seed if seed_kind == "randomstate" else np.random
    destination = tmp_path / "comparison"
    before = rng.get_state()
    # The identity is the first operation after preflight and directory creation.
    with patch.object(evaluation, "_search_identity", side_effect=RuntimeError("stop after preflight")):
        with pytest.raises(RuntimeError, match="stop after preflight"):
            evaluation.run_comparison({"knn": (raw_model(), {})}, X, y, X, y,
                                      output_dir=destination, seed=seed, threads=1)
    assert_rng_equal(rng.get_state(), before)
    first = evaluation.run_comparison({"knn": (raw_model(), {})}, X, y, X, y,
                                     output_dir=destination, seed=seed, threads=1)
    assert first["models"]["knn"]["selected_mean_score"] is not None
    assert first["models"]["knn"]["evaluation"]["status"] == "valid"
    # Completed wrapper results retain their original cache semantics even if a
    # mutable/global RNG has since advanced; no new CV folds are requested.
    rng.random_sample(7)
    before_resume = rng.get_state()
    with ExitStack() as stack:
        sentinels = forbid_calls(stack, [
            "supervised_nmf.evaluation.calibration_search",
            "supervised_nmf.evaluation.evaluate_classifier",
            "sklearn.neighbors.KNeighborsClassifier.fit",
        ])
        resumed = evaluation.run_comparison({"knn": (raw_model(), {})}, X, y, X, y,
                                           output_dir=destination, seed=seed, threads=1)
        for sentinel in sentinels:
            sentinel.assert_not_called()
    assert first["models"] == resumed["models"]
    assert_rng_equal(rng.get_state(), before_resume)


@pytest.mark.parametrize("kind", ["generic", "shared", "rank"])
def test_self_consistent_old_source_checkpoint_remains_incompatible(kind, data, tmp_path):
    destination = tmp_path / "cached"
    execute(kind, data, destination)
    if kind == "generic":
        saved = json.loads(destination.read_text(encoding="utf-8"))
        saved["identity"]["source_sha256"] = {"old_source": "0" * 64}
        saved["fingerprint"] = evaluation._digest(saved["identity"])
        saved["checkpoint_sha256"] = evaluation._digest({
            key: value for key, value in saved.items() if key != "checkpoint_sha256"
        })
        destination.write_text(json.dumps(saved), encoding="utf-8")
    else:
        manifest = destination / "identity.json"
        saved = study._read_receipt(manifest)
        saved["calibration"]["source_sha256"] = {"old_source": "0" * 64}
        study._write_receipt(manifest, saved)
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("Unexpected fit")) as fit:
        with pytest.raises(ValueError, match="[Ii]ncompatible"):
            execute(kind, data, destination)
        fit.assert_not_called()
