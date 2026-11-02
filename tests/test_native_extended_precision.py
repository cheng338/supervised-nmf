"""Native long-double precision/range through reporting and checkpoint reuse."""

from fractions import Fraction
import json
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from supervised_nmf.evaluation import _digest, _jsonable, calibration_search, evaluate_classifier
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.validation import build_cv_folds, validate_metadata


_INFO = np.finfo(np.longdouble)
pytestmark = pytest.mark.skipif(
    not (_INFO.nmant > 52 and _INFO.maxexp > 1024 and _INFO.minexp < -1022),
    reason="Requires native long-double precision and exponent range beyond float64",
)


def exact_value(tag):
    assert set(tag) == {"__supervised_nmf_scalar__", "significand", "exponent"}
    assert tag["__supervised_nmf_scalar__"] == "longdouble-v1"
    assert type(tag["significand"]) is str and type(tag["exponent"]) is int
    return Fraction(int(tag["significand"])) * Fraction(2) ** tag["exponent"]


def assert_exact_tags(tags, values):
    assert len(tags) == len(values)
    for tag, value in zip(tags, values):
        assert exact_value(tag) == Fraction(*value.as_integer_ratio())


def native_pair(kind):
    if kind == "precision":
        first = np.longdouble(1)
        direction = np.longdouble("inf")
    elif kind == "huge":
        first = np.ldexp(np.longdouble(1), 1024)
        direction = np.longdouble("inf")
    else:
        first = np.ldexp(np.longdouble(1), -1075)
        # Both the exact halfway value and its predecessor round to float64 zero.
        direction = np.longdouble(0)
    return first, np.nextafter(first, direction)


@pytest.mark.parametrize("kind", ["precision", "huge", "tiny"])
@pytest.mark.parametrize("sign", [1, -1])
def test_native_values_lost_by_float64_keep_exact_distinct_json(kind, sign):
    first, second = (value * sign for value in native_pair(kind))
    assert np.isfinite(first) and np.isfinite(second)
    assert first != 0 and second != 0 and first != second
    expected_power = {"precision": 0, "huge": 1024, "tiny": -1075}[kind]
    assert Fraction(*first.as_integer_ratio()) == sign * Fraction(2) ** expected_power
    with np.errstate(over="ignore", under="ignore"):
        narrowed = np.asarray([first, second], dtype=np.longdouble).astype(np.float64)
    assert narrowed[0] == narrowed[1]
    if kind == "huge":
        assert np.isinf(narrowed).all()
    elif kind == "tiny":
        assert (narrowed == 0).all()
    encoded = _jsonable([first, second])
    assert_exact_tags(encoded, [first, second])
    assert encoded[0] != encoded[1] and _digest(first) != _digest(second)
    assert _jsonable(encoded) == encoded
    loaded = json.loads(json.dumps(encoded, allow_nan=False))
    assert loaded == encoded and _digest(loaded) == _digest([first, second])
    assert len(json.dumps(encoded)) < 512


def near_integer_pairs(integers):
    values = []
    for integer in integers:
        first = np.longdouble(integer)
        values.extend([first, np.nextafter(first, np.longdouble("inf"))])
    return np.asarray(values, dtype=np.longdouble)


def native_metadata():
    ids = np.asarray([*native_pair("huge"), *native_pair("tiny"),
                      *near_integer_pairs(range(10, 14))], dtype=np.longdouble)
    groups = np.repeat(near_integer_pairs(range(1, 4)), 2)
    return {"data_ids": ids, "groups": groups}


def array_state(values):
    # Compare the same original allocation before/after; padding is not identity.
    return values.shape, values.strides, values.dtype.str, values.flags.writeable, values.tobytes()


def metadata_state(metadata):
    return {name: array_state(values) for name, values in metadata.items()}


@pytest.fixture
def problem():
    rows = np.arange(12, dtype=float)
    y = np.tile([0, 1], 6)
    X = np.column_stack([3 * y + rows / 100, np.sin(rows), np.cos(rows)])
    first, second = np.arange(6), np.arange(6, 12)
    X.setflags(write=False)
    y.setflags(write=False)
    return X, y, [(first, second), (second, first)]


def evaluation_metadata():
    metadata = native_metadata()
    return {"sample_ids": metadata["data_ids"], "groups": metadata["groups"],
            "train_sample_ids": -metadata["data_ids"], "train_groups": -metadata["groups"]}


def test_native_group_and_sample_identities_survive_evaluation(problem):
    X, y, folds = problem
    metadata = evaluation_metadata()
    assert len(set(metadata["sample_ids"].tolist())) == 12
    assert len(set(metadata["groups"].tolist())) == 6
    assert len(set(metadata["groups"].astype(np.float64).tolist())) == 3
    for values in metadata.values():
        values.setflags(write=False)
    before = metadata_state(metadata), array_state(X), array_state(y)
    # The middle pair straddles the two folds but differs at native precision.
    build_cv_folds(X, y, cv=2, groups=metadata["groups"], folds=folds)
    model = DummyClassifier(strategy="prior").fit(X, y)
    control = evaluate_classifier(model, X, y, train_y=y, threads=1)
    report = evaluate_classifier(model, X, y, train_y=y, threads=1, **metadata)
    assert report["status"] == control["status"] == "valid"
    assert report["metrics"] == control["metrics"]
    for name, values in control["predictions"].items():
        if name != "sample_ids":
            assert report["predictions"][name] == values
    assert report["support_diagnostics"]["group_count"] == 6
    assert report["support_diagnostics"]["training_group_count"] == 6
    assert_exact_tags(report["predictions"]["sample_ids"], metadata["sample_ids"])
    assert_exact_tags(report["evaluation_groups"], metadata["groups"])
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert (metadata_state(metadata), array_state(X), array_state(y)) == before


@pytest.mark.parametrize("case,message", [
    ("duplicate", "unique"),
    ("sample_overlap", "sample IDs must not overlap"),
    ("group_overlap", "groups must not overlap"),
    ("integer_overlap", "sample IDs must not overlap"),
])
def test_native_true_duplicate_and_overlap_reject_before_prediction(problem, monkeypatch, case, message):
    X, y, _ = problem
    metadata = evaluation_metadata()
    if case == "duplicate":
        metadata["sample_ids"][1] = metadata["sample_ids"][0]
    elif case == "sample_overlap":
        metadata["train_sample_ids"][0] = metadata["sample_ids"][1]
    elif case == "group_overlap":
        metadata["train_groups"][:2] = metadata["groups"][0]
    else:
        integer = 2**53 + 1
        metadata["sample_ids"][0] = np.longdouble(str(integer))
        metadata["train_sample_ids"] = metadata["train_sample_ids"].astype(object)
        metadata["train_sample_ids"][0] = integer
    model = DummyClassifier(strategy="prior").fit(X, y)
    calls = []
    for name in ("predict", "predict_proba", "decision_function",
                 "predict_with_probabilities", "predict_with_decision_function"):
        call = Mock(side_effect=AssertionError("Invalid metadata reached prediction"))
        monkeypatch.setattr(model, name, call, raising=False)
        calls.append(call)
    before = metadata_state(metadata)
    with pytest.raises(ValueError, match=message):
        evaluate_classifier(model, X, y, train_y=y, threads=1, **metadata)
    for call in calls:
        call.assert_not_called()
    assert metadata_state(metadata) == before


def test_native_integer_identity_above_float64_precision():
    integer = 2**53 + 1
    exact = np.longdouble(str(integer))
    assert Fraction(*exact.as_integer_ratio()) == integer
    assert exact != np.longdouble(integer - 1)
    distinct = validate_metadata([np.longdouble(integer - 1), exact], 2, name="ids", unique=True)
    assert len(distinct) == 2
    with pytest.raises(ValueError, match="unique"):
        validate_metadata([exact, integer], 2, name="ids", unique=True)


def test_native_true_group_overlap_is_rejected_in_explicit_folds(problem):
    X, y, folds = problem
    groups = native_metadata()["groups"]
    groups[6:8] = groups[0]
    before = array_state(groups)
    with pytest.raises(ValueError, match="groups must not overlap"):
        build_cv_folds(X, y, cv=2, groups=groups, folds=folds)
    assert array_state(groups) == before


def calibrate(api, directory, problem, metadata):
    X, y, folds = problem
    model = Pipeline([("pca", PCA(n_components=1, svd_solver="full", random_state=7)),
                      ("knn", KNeighborsClassifier(n_neighbors=1))])
    grid = {"pca__n_components": [1], "knn__n_neighbors": [1]}
    options = dict(cv=2, seed=7, threads=1, folds=folds, **metadata)
    if api == "search":
        return calibration_search(model, grid, X, y, refit=False,
                                  checkpoint_path=directory / "cv.json", **options).report
    if api == "grid":
        return calibration_knn_grid(model, grid, X, y, checkpoint_dir=directory, **options)
    return calibration_rank_search(model, grid, X, y, checkpoint_dir=directory,
                                   max_expansions=0, **options)


def assert_metadata_identity(report, api, metadata):
    identity = report["identity"] if api == "search" else report["identity"]["calibration"]
    assert_exact_tags(identity["data_ids"], metadata["data_ids"])
    assert_exact_tags(identity["groups"], metadata["groups"])
    if api == "rank":
        assert len(report["stages"]) == 1
        assert_metadata_identity(report["stages"][0], "grid", metadata)


def checkpoint_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*.json"))}


@pytest.mark.parametrize("api", ["search", "grid", "rank"])
@pytest.mark.parametrize("changed_field", ["data_ids", "groups"])
def test_native_checkpoint_resume_and_adjacent_id_change(
        problem, tmp_path, monkeypatch, api, changed_field):
    metadata = native_metadata()
    for values in metadata.values():
        values.setflags(write=False)
    before = metadata_state(metadata)
    directory = tmp_path / api
    first = calibrate(api, directory, problem, metadata)
    assert first["state"] == "cv_completed" and first["candidates"][0]["valid"]
    assert_metadata_identity(first, api, metadata)
    calls = []
    for estimator, method in ((PCA, "fit"), (PCA, "fit_transform"), (KNeighborsClassifier, "fit")):
        call = Mock(side_effect=AssertionError("Completed or incompatible checkpoint reached fit"))
        monkeypatch.setattr(estimator, method, call)
        calls.append(call)
    resumed = calibrate(api, directory, problem, metadata)
    assert resumed["state"] == "cv_completed"
    assert resumed["candidates"] == first["candidates"]
    assert resumed["fingerprint"] == first["fingerprint"]
    assert_metadata_identity(resumed, api, metadata)
    if api != "search":
        assert first["new_representation_fits"] == 2
        assert resumed["new_representation_fits"] == 0
        assert resumed["reused_representation_folds"] == 2
    saved = checkpoint_bytes(directory)
    assert saved
    changed = dict(metadata)
    changed[changed_field] = metadata[changed_field].copy()
    if changed_field == "data_ids":
        # Advance the larger huge ID; the result stays unique and finite.
        changed[changed_field][1] = np.nextafter(changed[changed_field][1], np.longdouble("inf"))
        assert len(set(changed[changed_field].tolist())) == 12
    else:
        # Preserve both rows of this subject and keep the explicit folds valid.
        changed[changed_field][:2] = np.nextafter(changed[changed_field][0], np.longdouble(0))
        assert len(set(changed[changed_field].tolist())) == 6
    changed[changed_field].setflags(write=False)
    changed_before = array_state(changed[changed_field])
    with pytest.raises(ValueError, match="incompatible"):
        calibrate(api, directory, problem, changed)
    for call in calls:
        call.assert_not_called()
    assert checkpoint_bytes(directory) == saved
    assert array_state(changed[changed_field]) == changed_before
    assert metadata_state(metadata) == before

