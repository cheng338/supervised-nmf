"""Extended metadata survives reporting and binds real calibration checkpoints."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf.evaluation import calibration_search, evaluate_classifier
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid


def half_integers(start, count):
    return np.arange(start, start + count, dtype=np.longdouble) + np.longdouble("0.5")


def half_tags(start, count):
    # These exact half-integers have an independently known odd significand.
    return [{"__supervised_nmf_scalar__": "longdouble-v1",
             "significand": str(2 * value + 1), "exponent": -1}
            for value in range(start, start + count)]


def array_state(value):
    return value.dtype.str, value.shape, value.strides, value.flags.writeable, value.tobytes()


def metadata_state(metadata):
    return {name: array_state(value) for name, value in metadata.items()}


def evaluation_metadata():
    return {"sample_ids": half_integers(100, 12),
            "train_sample_ids": half_integers(200, 12),
            "groups": np.repeat(half_integers(0, 6), 2),
            "train_groups": np.repeat(half_integers(1000, 6), 2)}


@pytest.fixture
def problem():
    index = np.arange(12, dtype=float)
    y = np.tile([0, 1], 6)
    X = np.column_stack([3 * y + index / 100, np.sin(index), np.cos(index)])
    first, second = np.arange(6), np.arange(6, 12)
    X.setflags(write=False)
    y.setflags(write=False)
    return X, y, [(first, second), (second, first)]


@pytest.fixture
def model(problem):
    X, y, _ = problem
    return DummyClassifier(strategy="prior").fit(X, y)


def forbid_predictions(monkeypatch, model):
    calls = []
    for name in ("predict", "predict_proba", "decision_function",
                 "predict_with_probabilities", "predict_with_decision_function"):
        call = Mock(side_effect=AssertionError("Invalid metadata reached prediction"))
        monkeypatch.setattr(model, name, call, raising=False)
        calls.append(call)
    return calls


def test_evaluation_reports_all_longdouble_metadata_without_changing_predictions(problem, model):
    X, y, _ = problem
    metadata = evaluation_metadata()
    metadata["sample_ids"][0] = np.longdouble("-0")
    for value in metadata.values():
        value.setflags(write=False)
    before = metadata_state(metadata), array_state(X), array_state(y)
    ordinary = {name: value.astype(float) for name, value in metadata.items()}
    expected = evaluate_classifier(model, X, y, threads=1, train_y=y, **ordinary)
    actual = evaluate_classifier(model, X, y, threads=1, train_y=y, **metadata)

    assert actual["status"] == expected["status"] == "valid"
    assert actual["metrics"] == expected["metrics"]
    assert actual["support_diagnostics"] == expected["support_diagnostics"]
    assert actual["training_majority_reference"] == expected["training_majority_reference"]
    for name, value in expected["predictions"].items():
        if name != "sample_ids":
            assert actual["predictions"][name] == value
    expected_ids = half_tags(100, 12)
    expected_ids[0] = {"__supervised_nmf_scalar__": "longdouble-v1",
                       "significand": "-0", "exponent": 0}
    assert actual["predictions"]["sample_ids"] == expected_ids
    assert actual["evaluation_groups"] == [tag for tag in half_tags(0, 6) for _ in range(2)]
    assert actual["support_diagnostics"]["sample_overlap_checked"]
    assert actual["support_diagnostics"]["group_overlap_checked"]
    assert actual["support_diagnostics"]["training_sample_count"] == 12
    assert actual["support_diagnostics"]["training_group_count"] == 6
    assert json.loads(json.dumps(actual, allow_nan=False))["predictions"]["sample_ids"] == expected_ids
    assert (metadata_state(metadata), array_state(X), array_state(y)) == before


@pytest.mark.parametrize("case,message", [
    ("duplicate_numeric", "unique"),
    ("duplicate_signed_zero", "unique"),
    ("sample_overlap", "sample IDs must not overlap"),
    ("group_signed_zero_overlap", "groups must not overlap"),
])
def test_reporting_does_not_relax_numeric_identity_checks(
        monkeypatch, problem, model, case, message):
    X, y, _ = problem
    metadata = evaluation_metadata()
    if case == "duplicate_numeric":
        metadata["sample_ids"] = metadata["sample_ids"].astype(object)
        metadata["sample_ids"][0] = np.longdouble(101.5)
        metadata["sample_ids"][1] = 101.5
    elif case == "duplicate_signed_zero":
        metadata["sample_ids"][:2] = [np.longdouble("-0"), np.longdouble("0")]
    elif case == "sample_overlap":
        metadata["train_sample_ids"] = metadata["train_sample_ids"].astype(object)
        metadata["train_sample_ids"][0] = 100.5
    else:
        metadata["groups"][:2] = np.longdouble("-0")
        metadata["train_groups"][:2] = np.longdouble("0")
    for value in metadata.values():
        value.setflags(write=False)
    before = metadata_state(metadata)
    calls = forbid_predictions(monkeypatch, model)
    with pytest.raises(ValueError, match=message):
        evaluate_classifier(model, X, y, threads=1, **metadata)
    for call in calls:
        call.assert_not_called()
    assert metadata_state(metadata) == before


@pytest.mark.parametrize("name", ["sample_ids", "train_sample_ids", "groups", "train_groups"])
def test_clongdouble_reporting_support_does_not_admit_complex_ids(
        monkeypatch, problem, model, name):
    X, y, _ = problem
    metadata = evaluation_metadata()
    metadata[name] = metadata[name].astype(object)
    metadata[name][0] = np.clongdouble(3)
    before = metadata_state(metadata)
    calls = forbid_predictions(monkeypatch, model)
    with pytest.raises(ValueError, match=name):
        evaluate_classifier(model, X, y, threads=1, **metadata)
    for call in calls:
        call.assert_not_called()
    assert metadata_state(metadata) == before


def calibrate(api, directory, problem, metadata):
    X, y, folds = problem
    estimator = Pipeline([("pca", PCA(n_components=1, svd_solver="full", random_state=7)),
                          ("knn", KNeighborsClassifier(n_neighbors=1))])
    grid = {"pca__n_components": [1], "knn__n_neighbors": [1]}
    common = dict(cv=2, seed=7, threads=1, folds=folds, **metadata)
    if api == "search":
        return calibration_search(estimator, grid, X, y, refit=False,
                                  checkpoint_path=directory / "cv.json", **common).report
    if api == "knn_grid":
        return calibration_knn_grid(estimator, grid, X, y,
                                    checkpoint_dir=directory, **common)
    return calibration_rank_search(estimator, grid, X, y,
                                   checkpoint_dir=directory, max_expansions=0, **common)


def checkpoint_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*.json"))}


def assert_identity(report, api):
    identity = report["identity"] if api == "search" else report["identity"]["calibration"]
    assert identity["data_ids"] == half_tags(100, 12)
    assert identity["groups"] == [tag for tag in half_tags(0, 6) for _ in range(2)]
    if api == "rank_search":
        assert len(report["stages"]) == 1
        assert report["expansions_performed"] == 0
        assert_identity(report["stages"][0], "knn_grid")


@pytest.mark.parametrize("api", ["search", "knn_grid", "rank_search"])
def test_longdouble_calibration_checkpoints_resume_and_reject_adjacent_changed_id(
        monkeypatch, tmp_path, problem, api):
    metadata = {"data_ids": half_integers(100, 12),
                "groups": np.repeat(half_integers(0, 6), 2)}
    for value in metadata.values():
        value.setflags(write=False)
    X, y, _ = problem
    before = metadata_state(metadata), array_state(X), array_state(y)
    directory = tmp_path / api
    first = calibrate(api, directory, problem, metadata)
    assert first["state"] == "cv_completed"
    assert len(first["candidates"]) == 1 and first["candidates"][0]["valid"]
    assert_identity(first, api)
    assert checkpoint_bytes(directory)
    if api != "search":
        assert first["new_representation_fits"] == 2

    calls = []
    for estimator, method in ((PCA, "fit"), (PCA, "fit_transform"), (KNeighborsClassifier, "fit")):
        call = Mock(side_effect=AssertionError("Completed or incompatible checkpoint reached fit"))
        monkeypatch.setattr(estimator, method, call)
        calls.append(call)
    resumed = calibrate(api, directory, problem, metadata)
    assert resumed["state"] == "cv_completed"
    assert resumed["candidates"] == first["candidates"]
    assert resumed["selected_parameters"] == first["selected_parameters"]
    assert resumed["fingerprint"] == first["fingerprint"]
    assert_identity(resumed, api)
    if api != "search":
        assert resumed["new_representation_fits"] == 0
        assert resumed["reused_representation_folds"] == 2

    # A valid resume can rewrite timing/report fields. Preserve the completed
    # post-resume bytes when testing rejection of a different metadata identity.
    saved = checkpoint_bytes(directory)
    changed_ids = metadata["data_ids"].copy()
    changed_ids[0] = np.nextafter(changed_ids[0], np.longdouble("inf"))
    changed_ids.setflags(write=False)
    assert changed_ids[0] != metadata["data_ids"][0]
    assert len(set(changed_ids.tolist())) == len(changed_ids)
    changed_before = array_state(changed_ids)
    with pytest.raises(ValueError, match="incompatible"):
        calibrate(api, directory, problem, dict(metadata, data_ids=changed_ids))
    for call in calls:
        call.assert_not_called()
    assert checkpoint_bytes(directory) == saved
    assert array_state(changed_ids) == changed_before
    assert (metadata_state(metadata), array_state(X), array_state(y)) == before
