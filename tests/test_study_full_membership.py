"""Public shared-grid and recovery contracts at the full-training KNN endpoint."""

from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import joblib
from joblib import parallel_backend
from joblib._parallel_backends import ThreadingBackend
import numpy as np
import pytest
import sklearn
from sklearn import config_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf import NMFClassifier, study
from supervised_nmf._neighbors import _KNNQueryReuse
from tests._screening_recovery_support import science
from tests.test_candidate_source_identity import load_replacement


@pytest.fixture(autouse=True)
def ordinary_numerical_policy():
    previous = np.seterr(divide="warn", over="warn", under="ignore", invalid="warn")
    callback = np.geterrcall()
    np.seterrcall(None)
    try:
        with config_context(assume_finite=False), threadpool_limits(limits=1):
            yield
    finally:
        np.seterr(**previous)
        np.seterrcall(callback)


@pytest.fixture
def problem():
    random = np.random.default_rng(811)
    y = np.arange(72) % 2
    coefficients = random.uniform(.1, 1., (72, 16))
    coefficients[:, 0] += y
    basis = np.column_stack([np.eye(16), random.uniform(.1, 1., (16, 16))])
    X = coefficients @ basis
    folds = [(np.r_[np.arange(i * 24), np.arange((i + 1) * 24, 72)],
              np.arange(i * 24, (i + 1) * 24)) for i in range(3)]
    return X, y, folds


def estimator(kind, *, jobs=1):
    if kind == "raw":
        return KNeighborsClassifier(algorithm="brute", weights="uniform", n_jobs=jobs)
    return NMFClassifier(
        solver="spa", screening="absolute", score_threshold=0, n_components=16,
        random_state=7, max_iter=500, weights="uniform",
        decision_threshold=.5, positive_label=1)


def run_grid(problem, directory, kind="raw", ks=(1, 3, 48), *, model=None, grid=None):
    X, y, folds = problem
    return study.calibration_knn_grid(
        estimator(kind) if model is None else model,
        {"n_neighbors": list(ks)} if grid is None else grid,
        X, y, folds=folds, cv=3, seed=7, threads=1,
        scoring="accuracy", checkpoint_dir=directory)


def native_proof_runtime():
    # Other supported dependency versions deliberately retain native fallback.
    return (sklearn.__version__, joblib.__version__) in {("1.9.1", "1.6.0"), ("1.9.0", "1.5.3")}


@pytest.mark.parametrize("unit", ["ns", "D"])
def test_temporal_neighbor_count_rejected_before_checkpoint_creation(problem, tmp_path, unit):
    directory = tmp_path / "grid"
    with pytest.raises(ValueError, match="n_neighbors.*positive integer"):
        run_grid(problem, directory, ks=(np.int64(3), np.timedelta64(1, unit)))
    assert not directory.exists()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ball_tree_calibration_matches_independent_selected_widths(dtype, tmp_path):
    # A distant cluster makes the rounded BallTree pruning bound sensitive
    # to how many neighbors the search requests, even with identical leaves.
    near = np.nextafter(dtype(1), dtype(2))
    query = np.nextafter(dtype(1), dtype(0))
    X = np.array([*range(-30, -15), .9, near, *([2**54] * 15),
                  query, .87, -24.13, -27.31], dtype=dtype)[:, None]
    y = np.r_[np.repeat([0, 1], 16), [1, 0, 0, 1]]
    train, valid = np.arange(32), np.arange(32, 36)
    folds = [(train, valid), (valid, train)]
    settings = dict(algorithm="ball_tree", leaf_size=1, metric="euclidean", n_jobs=1)
    report = study.calibration_knn_grid(
        KNeighborsClassifier(**settings), {"n_neighbors": [1, 3]}, X, y,
        folds=folds, cv=2, threads=1, seed=7, checkpoint_dir=tmp_path)
    means = {}
    for k, candidate in zip((1, 3), report["candidates"], strict=True):
        expected = []
        for train, valid in folds:
            fitted = KNeighborsClassifier(n_neighbors=k, **settings).fit(X[train], y[train])
            expected.append(np.mean(fitted.predict(X[valid]) == y[valid]))
        assert candidate["parameters"]["n_neighbors"] == k
        assert [fold["score"] for fold in candidate["folds"]] == expected
        means[k] = np.mean(expected)
        assert candidate["mean_score"] == means[k]
    assert report["selected_parameters"]["n_neighbors"] == max(means, key=means.get)


def saved_bytes(directory):
    return {path.relative_to(directory): path.read_bytes()
            for path in Path(directory).rglob("*") if path.is_file()}


def assert_accounting(report):
    rows = report["representation_folds"]
    expected = {
        "screening_statistics_seconds": sum(r["screening_statistics_seconds"] for r in rows),
        "screening_selection_seconds": sum(r["screening_selection_seconds"] for r in rows),
        "representation_fit_seconds": sum(r["representation_fit_seconds"] for r in rows),
        "validation_projection_seconds": sum(r["projection_seconds"] for r in rows),
        "knn_fit_seconds": sum(h["fit_seconds"] for r in rows for h in r["neighbors"].values()),
        "knn_reuse_setup_seconds": sum(r["neighbor_reuse_setup_seconds"] for r in rows),
        "knn_shared_query_seconds": sum(r["neighbor_query_seconds"] for r in rows),
        "knn_prediction_seconds": sum(h["prediction_seconds"] for r in rows
                                      for h in r["neighbors"].values()),
    }
    assert report["compute_seconds"] == expected
    assert report["tuning_seconds"] == sum(expected.values())
    for row in rows:
        assert row["neighbor_query_seconds"] == sum(q["query_seconds"] for q in row["neighbor_queries"])
        assert all(q.get("short_query_attempt_seconds", 0.) <= q["query_seconds"]
                   for q in row["neighbor_queries"])


@pytest.mark.parametrize("kind", ["raw", "nmf"])
@pytest.mark.parametrize("ks", [(1, 3, 48), (48,)], ids=["mixed", "sole"])
def test_full_training_heads_match_native_decisions(problem, tmp_path, kind, ks):
    X, y, _ = problem
    before = X.copy(), y.copy()
    original_factory, original_predict = study._KNNQueryReuse, _KNNQueryReuse.predict
    offered, compared = [], []

    def factory(models, query, **kwargs):
        offered.append(kwargs.get("_study_call_contract"))
        return original_factory(models, query, **kwargs)

    def compare(reuse, k, **kwargs):
        result = original_predict(reuse, k, **kwargs)
        model = reuse.models[k]
        expected = (model.predict_proba(reuse.query) if kwargs.get("probabilities")
                    else model.predict(reuse.query))
        np.testing.assert_array_equal(result[0], expected)
        if kwargs.get("decision_threshold") is not None:
            positive = kwargs["positive_index"]
            np.testing.assert_array_equal(result[0][:, positive] > kwargs["decision_threshold"],
                                          expected[:, positive] > kwargs["decision_threshold"])
        compared.append(k)
        return result

    with patch.object(study, "_KNNQueryReuse", factory), patch.object(_KNNQueryReuse, "predict", compare):
        report = run_grid(problem, tmp_path, kind, ks)
    assert compared == list(ks) * 3
    options = (True, .5, 1) if kind == "nmf" else (False, None, None)
    assert offered == [tuple((k, *options) for k in ks)] * 3
    assert all(candidate["valid"] for candidate in report["candidates"])
    for row in report["representation_folds"]:
        queries = row["neighbor_queries"]
        endpoint = row["neighbors"]["48"]["neighbor_query_reuse"]
        if native_proof_runtime():
            assert len(queries) == 1 and queries[0]["study_short_query_active"]
            assert queries[0]["shared_query_neighbor_counts"] == ([4] if len(ks) > 1 else [1])
            assert endpoint["full_membership_certified"]
            assert endpoint["full_membership_vote_basis"] == "short_query_training_histogram"
        else:
            assert not any(q.get("study_short_query_active", False) for q in queries)
            assert not endpoint.get("full_membership_certified", False)
    assert_accounting(report)
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(y, before[1])


@pytest.mark.parametrize("kind", ["raw", "nmf"])
def test_no_endpoint_does_not_offer_an_owned_call_contract(problem, tmp_path, kind):
    original = study._KNNQueryReuse
    offered = []

    def factory(models, query, **kwargs):
        offered.append((tuple(models), kwargs.get("_study_call_contract")))
        return original(models, query, **kwargs)

    with patch.object(study, "_KNNQueryReuse", factory):
        report = run_grid(problem, tmp_path, kind, (1, 3))
    assert offered == [((1, 3), None)] * 3
    assert all(candidate["valid"] for candidate in report["candidates"])
    assert_accounting(report)


def test_failed_endpoint_fit_does_not_offer_an_owned_call_contract(problem, tmp_path):
    original_factory, original_fit = study._KNNQueryReuse, KNeighborsClassifier.fit
    offered, failed = [], []

    def fit(model, X, y):
        result = original_fit(model, X, y)
        if model.n_neighbors == 48:
            failed.append(model)
            raise ValueError("controlled completed endpoint fit failure")
        return result

    def factory(models, query, **kwargs):
        offered.append((tuple(models), kwargs.get("_study_call_contract")))
        return original_factory(models, query, **kwargs)

    with patch.object(KNeighborsClassifier, "fit", fit), patch.object(study, "_KNNQueryReuse", factory):
        report = run_grid(problem, tmp_path)
    assert len(failed) == 3 and offered == [((1, 3), None)] * 3
    assert [c["valid"] for c in report["candidates"]] == [True, True, False]
    for row in report["representation_folds"]:
        endpoint = row["neighbors"]["48"]
        assert endpoint["status"] == "failed" and endpoint["error_type"] == "ValueError"
        assert endpoint["error"] == "controlled completed endpoint fit failure"
    assert_accounting(report)


@pytest.mark.parametrize("jobs", [None, 2])
@pytest.mark.parametrize("mode", ["silent", "warning", "error"])
def test_custom_joblib_backend_retains_native_sole_head_effects(problem, tmp_path, jobs, mode):
    X, y, folds = problem

    class Backend(ThreadingBackend):
        def __init__(self):
            super().__init__()
            self.events = []

        def effective_n_jobs(self, n_jobs):
            self.events.append(n_jobs)
            if len(self.events) == 2:
                if mode == "warning":
                    warnings.warn("native backend warning", RuntimeWarning)
                elif mode == "error":
                    raise ValueError("native backend error")
            return 1

    reference_backend = Backend()
    expected = []
    with parallel_backend(reference_backend, n_jobs=2):
        reference_backend.events.clear()
        for train, validation in folds:
            model = estimator("raw", jobs=jobs).set_params(n_neighbors=48).fit(X[train], y[train])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    prediction = model.predict(X[validation])
                    outcome = {"score": float(np.mean(prediction == y[validation]))}
                except ValueError as error:
                    outcome = {"error_type": type(error).__name__, "error": str(error)}
            outcome["warnings"] = [(w.category.__name__, str(w.message)) for w in caught]
            expected.append(outcome)

    actual_backend = Backend()
    with parallel_backend(actual_backend, n_jobs=2):
        actual_backend.events.clear()
        report = run_grid(problem, tmp_path, ks=(48,), model=estimator("raw", jobs=jobs))
    assert actual_backend.events == reference_backend.events
    assert actual_backend.events
    for row, native in zip(report["representation_folds"], expected):
        head = row["neighbors"]["48"]
        assert [(w["category"], w["message"]) for w in head["warnings"]] == native["warnings"]
        if "error" in native:
            assert head["status"] == "failed"
            assert (head["error_type"], head["error"]) == (native["error_type"], native["error"])
        else:
            assert head["status"] == "valid" and head["score"] == native["score"]
            assert not head["neighbor_query_reuse"].get("full_membership_certified", False)
        assert not any(q.get("study_short_query_active", False) for q in row["neighbor_queries"])
    assert_accounting(report)


@pytest.mark.parametrize("kind", ["raw", "nmf"])
def test_partial_endpoint_resume_preserves_receipts_and_charges_queries_once(problem, tmp_path, kind):
    directory = tmp_path / "partial"
    write = study._write_receipt
    writes = 0

    def interrupt_second_receipt(path, payload):
        nonlocal writes
        if Path(path).parent.name == "folds":
            writes += 1
            if writes == 2:
                raise KeyboardInterrupt("controlled endpoint receipt interruption")
        return write(path, payload)

    with patch.object(study, "_write_receipt", interrupt_second_receipt), \
            pytest.raises(KeyboardInterrupt, match="controlled endpoint receipt interruption"):
        run_grid(problem, directory, kind)
    assert writes == 2
    committed = list((directory / "folds").glob("*.json"))
    assert len(committed) == 1
    old_receipt = study._read_receipt(committed[0])
    before = saved_bytes(directory)
    attempts = [study._read_receipt(p) for p in (directory / "attempts").glob("*.json")]
    assert sorted(r["state"] for r in attempts) == ["completed", "interrupted"]
    assert all(r["representation_fit_performed"] for r in attempts)
    assert not (directory / "report.json").exists()

    resumed = run_grid(problem, directory, kind)
    assert resumed["new_representation_fits"] == 2
    assert resumed["reused_representation_folds"] == 1
    assert old_receipt in resumed["representation_folds"]
    after = saved_bytes(directory)
    assert all(after[path] == contents for path, contents in before.items())
    assert len(list((directory / "attempts").glob("*.json"))) == 4
    assert_accounting(resumed)
    reference = run_grid(problem, tmp_path / "uninterrupted", kind)
    assert science(resumed) == science(reference)

    before_completed = saved_bytes(directory)
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("completed fit")), \
            patch.object(study, "_KNNQueryReuse", side_effect=AssertionError("completed adapter")):
        completed = run_grid(problem, directory, kind)
    assert completed["new_representation_fits"] == 0 and completed["reused_representation_folds"] == 3
    for key in ("candidates", "representation_folds", "compute_seconds", "tuning_seconds"):
        assert completed[key] == resumed[key]
    assert_accounting(completed)
    assert study._read_receipt(directory / "report.json") == completed
    after_completed = saved_bytes(directory)
    assert {p: v for p, v in before_completed.items() if p.name != "report.json"} == {
        p: v for p, v in after_completed.items() if p.name != "report.json"}


def test_full_endpoint_checkpoint_rejects_changed_transformer_source(problem, tmp_path, monkeypatch):
    # Reuse the repository's test-owned source fixture; no package source is edited.
    name = "snmf_cache_identity_fixture"
    monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    path = tmp_path / "transformer.py"
    first_module, first = load_replacement(path, 0)
    make_pipeline = lambda feature: Pipeline([
        ("feature", feature), ("knn", estimator("raw"))])
    grid = {"knn__n_neighbors": [1, 48]}
    directory = tmp_path / "checkpoint"
    original = run_grid(problem, directory, model=make_pipeline(first), grid=grid)
    assert all(c["valid"] for c in original["candidates"])
    assert first_module.FIT_COUNT == 3
    before = saved_bytes(directory)
    second_module, second = load_replacement(path, 1)
    with pytest.raises(ValueError, match="[Ii]ncompatible|[Ss]ource|[Ii]dentity"):
        run_grid(problem, directory, model=make_pipeline(second), grid=grid)
    assert second_module.FIT_COUNT == 0
    assert saved_bytes(directory) == before
