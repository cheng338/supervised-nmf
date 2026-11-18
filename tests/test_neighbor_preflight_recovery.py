"""Recovery and saved-report contracts for impossible neighbor counts."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from supervised_nmf import study
from supervised_nmf.screening import BinaryScreening


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def problem():
    rng = np.random.default_rng(735)
    y = np.tile([0, 1], 12)
    X = rng.uniform(.1, 1., (24, 4)) + .2 * y[:, None]
    return X, y


def run_grid(problem, directory, order):
    X, y = problem
    estimator = Pipeline([
        ("screen", BinaryScreening(screening="absolute", score_threshold=0)),
        ("knn", KNeighborsClassifier()),
    ])
    # Both k values exceed every 16-row training fold. The second threshold
    # would select no features if executed, but neighbor feasibility is first.
    return study.calibration_knn_grid(
        estimator, {"screen__score_threshold": [0, 1e20], "knn__n_neighbors": [17, 19]},
        X, y, cv=3, seed=7, threads=1, scoring="balanced_accuracy",
        execution_order=order, checkpoint_dir=directory)


def saved_bytes(directory, relative="folds"):
    return {path: path.read_bytes() for path in (directory / relative).glob("*.json")}


def assert_preserved(snapshot):
    assert all(path.read_bytes() == contents for path, contents in snapshot.items())


def assert_rejected_report(report, order):
    assert report["execution_order"]["resolved"] == order
    assert report["state"] == "cv_completed"
    assert report["new_representation_fits"] == report["unique_representation_fits"] == 0
    assert report["screening_statistics_fits"] == 0
    assert report["preflight_rejected_representation_folds"] == 6
    assert report["selected_parameters"] is None
    assert report["selected_mean_score"] is None
    assert not report["ranked_candidate_indices"]
    assert report["compute_seconds"] and set(report["compute_seconds"].values()) == {0.0}
    assert report["tuning_seconds"] == 0.0
    assert len(report["candidates"]) == 4
    for candidate in report["candidates"]:
        assert not candidate["valid"] and candidate["mean_score"] is None
        assert candidate["invalid_fold_count"] == 3
        assert [row["fold"] for row in candidate["folds"]] == [0, 1, 2]
        for row in candidate["folds"]:
            assert row["status"] == "failed"
            assert row["failure_stage"] == "neighbor_feasibility"
            assert row["preflight_rejected"] is True
            assert row["error_type"] == "ValueError"
            assert not row["warnings"] and row["diagnostics"] == {}
    assert len(report["representation_folds"]) == 6
    for row in report["representation_folds"]:
        assert row["state"] == "completed" and row["status"] == "failed"
        assert row["neighbors"] == {} and row["neighbor_queries"] == []
        assert not row.get("diagnostics")


@pytest.mark.parametrize("order", ["representation", "fold"])
def test_completed_impossible_neighbor_receipts_reuse_without_execution(problem, tmp_path, order):
    with patch.object(BinaryScreening, "fit", side_effect=AssertionError("screen should not run")), \
            patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("KNN should not fit")):
        first = run_grid(problem, tmp_path, order)
    assert_rejected_report(first, order)
    committed = saved_bytes(tmp_path)
    attempts = saved_bytes(tmp_path, "attempts")
    identity = (tmp_path / "identity.json").read_bytes()
    with patch.object(study, "_run_representation_fold", side_effect=AssertionError("completed fold rerun")):
        resumed = run_grid(problem, tmp_path, order)
    assert_rejected_report(resumed, order)
    assert resumed["reused_representation_folds"] == 6
    assert resumed["candidates"] == first["candidates"]
    assert resumed["representation_folds"] == first["representation_folds"]
    assert_preserved(committed)
    assert_preserved(attempts)
    assert len(saved_bytes(tmp_path, "attempts")) == len(attempts)
    assert (tmp_path / "identity.json").read_bytes() == identity


@pytest.mark.parametrize("order", ["representation", "fold"])
@pytest.mark.parametrize("commit_second", [False, True], ids=["before-commit", "after-commit"])
def test_interrupted_neighbor_rejection_resumes_only_uncommitted_folds(
        problem, tmp_path, order, commit_second):
    directory = tmp_path / "resumed"
    write = study._write_receipt
    fold_writes = 0

    def interrupt_second_commit(path, payload):
        nonlocal fold_writes
        if Path(path).parent.name == "folds":
            fold_writes += 1
            if fold_writes == 2:
                if commit_second:
                    write(path, payload)
                raise KeyboardInterrupt("controlled neighbor receipt interruption")
        return write(path, payload)

    with patch.object(study, "_write_receipt", interrupt_second_commit), \
            pytest.raises(KeyboardInterrupt, match="controlled neighbor receipt interruption"):
        run_grid(problem, directory, order)
    committed = saved_bytes(directory)
    assert len(committed) == (2 if commit_second else 1)
    attempts = saved_bytes(directory, "attempts")
    states = [study._read_receipt(path)["state"] for path in attempts]
    assert states.count("interrupted") == 1
    assert not (directory / "report.json").exists()
    identity = (directory / "identity.json").read_bytes()
    execute = study._run_representation_fold
    visits = []

    def observe(*args, **kwargs):
        visits.append(args[9]["key"])
        return execute(*args, **kwargs)

    with patch.object(study, "_run_representation_fold", observe):
        resumed = run_grid(problem, directory, order)
    assert_rejected_report(resumed, order)
    assert resumed["reused_representation_folds"] == len(committed)
    assert len(visits) == 6 - len(committed)
    assert set(visits).isdisjoint(study._read_receipt(path)["key"] for path in committed)
    assert_preserved(committed)
    assert_preserved(attempts)
    assert (directory / "identity.json").read_bytes() == identity
    reference = run_grid(problem, tmp_path / "fresh", order)
    assert resumed["candidates"] == reference["candidates"]
    assert resumed["representation_folds"] == reference["representation_folds"]
    assert resumed["compute_seconds"] == reference["compute_seconds"]


def test_saved_fold_report_accepts_zero_fit_receipts_without_diagnostics(problem, tmp_path):
    report = run_grid(problem, tmp_path / "chunk-0000", "fold")
    assert_rejected_report(report, "fold")
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_celeba_study.py"
    spec = importlib.util.spec_from_file_location("_neighbor_preflight_reporter", path)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    inputs = reporter.Inputs(tmp_path)
    summary = reporter._fold_diagnostics(inputs, tmp_path)
    assert summary["unique_completed_representation_folds"] == 6
    assert summary["statuses"] == {"failed": 6}
    assert summary["error_types"] == {"ValueError": 6}
    assert summary["warning_categories_by_fold"] == {}
    assert set(summary["compute_seconds"].values()) == {0.0}
    assert summary["factorization_timed_folds"] == 0
    assert summary["maximum_recorded_iterations"] is None
    assert summary["nonadditive_breakdown_seconds"]["recorded_factorization_seconds"] is None
    assert len(inputs.hashes) == 6
