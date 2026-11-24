"""Strict reusable identity must not depend on estimator display text."""
import functools
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from supervised_nmf._identity import (
    _share_calibration_observation, configuration_identity, identity_digest, implementation_sources,
)
from supervised_nmf.evaluation import calibration_search
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid


class SameReprEstimator(BaseEstimator):
    def __init__(self, value=1, array=None):
        self.value = value
        self.array = array

    def __repr__(self):
        raise AssertionError("Display repr must not be needed for identity")


def field_array(difference, changed):
    if difference == "title_lookup":
        dtype = np.dtype([(("alias", "word"), "u4")])
        if not changed:
            # NumPy retains the title in the field tuple but drops the alias lookup.
            dtype.names = dtype.names
    elif difference == "record":
        dtype = np.dtype([("word", "u4")])
        if changed:
            dtype = np.dtype((np.record, dtype))
    else:
        leaf = np.dtype("l")  # C long, the width of C int on Windows or C long long on LP64.
        if difference == "native_class":
            other = np.dtype("i") if leaf.itemsize == np.dtype("i").itemsize else np.dtype("q")
            assert leaf.type is not other.type and leaf.str == other.str
            if changed:
                leaf = other
        elif difference == "byteorder":
            native = "<" if sys.byteorder == "little" else ">"
            leaf = leaf.newbyteorder(native if changed else "=")
        elif difference == "builtin":
            leaf = np.dtype(leaf, copy=not changed)
        else:
            raise AssertionError(difference)
        dtype = np.dtype([("word", leaf)])
    return np.ones(1, dtype=dtype)


def field_choice(words, difference):
    if difference == "class_specifier":
        return np.ones(1, dtype=(words, [("word", "u4")]))[0].feature
    leaf = words.dtype.fields["word"][0]
    if difference == "native_class":
        return int(type(words["word"][0]) is not np.dtype("l").type)
    if difference == "byteorder":
        return int(leaf.byteorder != "=")
    if difference == "builtin":
        return int(leaf.isbuiltin)
    if difference == "title_lookup":
        return int("alias" in words.dtype.fields)
    return int(getattr(words[0], "word", 0))


class FieldClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, words, difference):
        self.words = words
        self.difference = difference

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.label_ = self.classes_[field_choice(self.words, self.difference)]
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)


class FoldPublished(KeyboardInterrupt):
    pass


def test_fold_observation_reuses_equal_records_and_preserves_different_reads():
    def field(encoded, name):
        return next(item for key, item in encoded["items"] if key["value"] == name)

    schema = {"columns": [0, 1, 3], "scale": (2, 5)}
    estimator = SameReprEstimator(value=schema)
    candidates = [{"schema": schema, "weights": np.arange(3), "width": 1}]
    full = configuration_identity({"estimator": estimator, "candidates": candidates})
    original_full = deepcopy(full)
    candidates[0]["width"] = 2  # Simulate a constructor change between the two reads.
    observation = []
    implementation_sources(estimator, candidates, _observation=observation)
    original_observation = deepcopy(observation)
    assert observation[0] is not field(full, "estimator")
    full_candidate = field(full, "candidates")["items"][0]
    assert field(full_candidate, "weights")["content"] is not None

    for _ in range(2):  # The second pass encounters records that already share identity.
        _share_calibration_observation(full, observation)
        assert full == original_full
        assert observation == original_observation
        assert observation[0] is field(full, "estimator")
        mixed = observation[1]["items"][0]
        assert field(mixed, "schema") is field(full_candidate, "schema")
        assert field(mixed, "weights")["content"] is None
        assert field(mixed, "width") != field(full_candidate, "width")


class CustomCloneEstimator(BaseEstimator):
    def __sklearn_clone__(self):
        return self


class PartialSubclass(functools.partial):
    def __call__(self, *args, **kwargs):
        return -super().__call__(*args, **kwargs)


class ScalarSubclass(np.float64):
    pass


class CloneBypassEstimator(BaseEstimator):
    clone_calls = []

    def __init__(self, role="pipeline", n_components=1):
        self.role = role
        self.n_components = n_components

    def __sklearn_clone__(self):
        self.clone_calls.append(self.role)
        if self.role == "pipeline":
            return Pipeline([("pca", PCA(n_components=self.n_components, random_state=0)),
                             ("knn", KNeighborsClassifier(n_neighbors=1))])
        return PCA(n_components=self.n_components, random_state=0)


def identity_function(values, offset=1., *, power=2):
    return (values + offset) ** power


def closure_function(offset):
    return lambda values: values + offset


def test_estimators_use_constructor_state_and_never_display_repr():
    first = SameReprEstimator(value=1)
    second = SameReprEstimator(value=2)
    assert identity_digest(first) != identity_digest(second)
    before = identity_digest(first)
    first.learned_ = np.arange(12)
    assert identity_digest(first) == before
    assert identity_digest(SameReprEstimator(value=1)) == before
    json.dumps(configuration_identity(first), allow_nan=False)


@pytest.mark.parametrize("left,right", [
    (True, 1), (1, 1.), (1., np.float64(1.)),
    ([1, 2], (1, 2)), ({1: "a"}, {"1": "a"}),
    (set([1, 2]), frozenset([1, 2])), (0., -0.),
    (b"abc", "abc"), (np.int32(1), np.int64(1)),
    (slice(1, 4, 1), (1, 4, 1)),
])
def test_types_and_exact_scalar_values_are_distinct(left, right):
    assert identity_digest(left) != identity_digest(right)


def test_dictionary_order_is_preserved_while_set_order_is_not_semantic():
    # A dict-valued parameter may intentionally implement ordered operations.
    assert identity_digest({"a": 1, "b": [2]}) != identity_digest({"b": [2], "a": 1})
    assert identity_digest({1, "a", 3.}) == identity_digest({3., "a", 1})


def test_full_array_content_and_dtype_shape_are_bound():
    first = np.zeros(2000)
    second = first.copy()
    second[1000] = 1
    assert repr(first) == repr(second)
    assert identity_digest(first) != identity_digest(second)
    assert identity_digest(first) != identity_digest(first.reshape(40, 50))
    assert identity_digest(first) != identity_digest(first.astype(np.float32))
    matrix = np.arange(12.).reshape(3, 4)
    assert identity_digest(matrix) == identity_digest(np.asfortranarray(matrix))
    first.setflags(write=False)
    assert identity_digest(first) == identity_digest(np.zeros(2000))


def test_structured_dtype_and_object_array_values_are_bound():
    left = np.zeros(3, dtype=[("left", "i4"), ("right", "f4")])
    right = np.zeros(3, dtype=[("other", "i4"), ("right", "f4")])
    assert left.dtype.str == right.dtype.str
    assert identity_digest(left) != identity_digest(right)
    values = np.array([SameReprEstimator(1), {"x": [2]}], dtype=object)
    changed = np.array([SameReprEstimator(2), {"x": [2]}], dtype=object)
    assert identity_digest(values) != identity_digest(changed)


@pytest.mark.parametrize("layout", ["direct", "field", "subarray_field", "standalone_subarray"])
@pytest.mark.parametrize("difference", ["scalar_class", "aligned"])
def test_structured_dtype_binds_nested_scalar_class_and_alignment(layout, difference):
    def make(changed):
        dtype = np.dtype([("word", "u4")], align=changed and difference == "aligned")
        if changed and difference == "scalar_class":
            dtype = np.dtype((np.record, dtype))
        if layout == "field":
            return np.dtype([("inner", dtype)])
        if layout == "subarray_field":
            return np.dtype([("inner", dtype, (2,))])
        if layout == "standalone_subarray":
            return np.dtype((dtype, (2,)))
        return dtype

    def records(value):
        observation = []
        implementation_sources(value, _observation=observation)
        return configuration_identity(value), observation[0]

    left, right = make(False), make(True)
    assert left.descr == right.descr and left.str == right.str
    assert left.itemsize == right.itemsize
    to_values = [lambda dtype: dtype]
    if layout != "standalone_subarray":  # Creating an array expands a top-level subarray into its shape.
        to_values += [lambda dtype: np.zeros(1, dtype=dtype), lambda dtype: np.zeros((), dtype=dtype)[()]]
    for convert in to_values:
        left_entry, left_fold = records(convert(left))
        right_entry, right_fold = records(convert(right))
        assert left_entry != right_entry
        assert left_fold != right_fold
        assert records(convert(make(True))) == (right_entry, right_fold)


@pytest.mark.parametrize("difference", ["native_class", "byteorder", "builtin", "title_lookup"])
def test_structured_field_properties_survive_array_copy_and_bind_fold_observations(difference):
    original, changed = (field_array(difference, state) for state in (False, True))
    assert original.dtype.descr == changed.dtype.descr and original.tobytes() == changed.tobytes()
    assert field_choice(deepcopy(original), difference) == 0
    assert field_choice(deepcopy(changed), difference) == 1
    for value in (lambda array: array.dtype, lambda array: array, lambda array: array[0]):
        first, second = value(original), value(changed)
        assert configuration_identity(first) != configuration_identity(second)
        first_observation, second_observation = [], []
        implementation_sources(first, _observation=first_observation)
        implementation_sources(second, _observation=second_observation)
        assert first_observation != second_observation


@pytest.mark.parametrize("difference", ["record", "native_class", "byteorder", "builtin", "title_lookup"])
@pytest.mark.parametrize("partial", [False, True])
def test_calibration_binds_field_behavior_before_reusing_folds(tmp_path, partial, difference):
    from supervised_nmf import evaluation

    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 14 + [1] * 6)
    path = tmp_path / "folds.json"

    def run(changed, checkpoint=None):
        estimator = FieldClassifier(field_array(difference, changed), difference)
        return calibration_search(estimator, {}, X, y, cv=2, seed=17, threads=1,
                                  refit=False, checkpoint_path=checkpoint).report

    def scores(report):
        candidate, = report["candidates"]
        assert candidate["valid"] and not report["is_fallback"]
        return [fold["score"] for fold in candidate["folds"]]

    writer, reader = run(True), run(False)
    assert writer["folds"] == reader["folds"]
    assert scores(writer) == [0.3, 0.3] and scores(reader) == [0.7, 0.7]
    if partial:
        publish = evaluation.write_json

        def stop_after_first_fold(destination, report):
            publish(destination, report)
            if report["candidates"][0]["folds"]:
                raise FoldPublished

        with patch.object(evaluation, "write_json", stop_after_first_fold), pytest.raises(FoldPublished):
            run(True, path)
    else:
        assert scores(run(True, path)) == scores(writer)

    before = path.read_bytes()
    saved = json.loads(before)["candidates"][0]["folds"]
    assert [fold["score"] for fold in saved] == scores(writer)[:1 if partial else 2]
    with (patch.object(FieldClassifier, "fit", autospec=True, side_effect=FieldClassifier.fit) as fit,
          patch.object(evaluation, "write_json", wraps=evaluation.write_json) as write):
        with pytest.raises(ValueError, match="incompatible"):
            run(False, path)
        fit.assert_not_called()
        write.assert_not_called()
    assert path.read_bytes() == before
    with patch.object(FieldClassifier, "fit", autospec=True, side_effect=FieldClassifier.fit) as fit:
        assert scores(run(True, path)) == scores(writer)
        assert fit.call_count == 2 - len(saved)


@pytest.mark.parametrize("difference", ["record", "native_class"])
@pytest.mark.parametrize("partial", [False, True])
def test_shared_grid_fits_each_structured_scalar_class(tmp_path, partial, difference):
    from supervised_nmf import study

    y = np.tile([0, 1], 20)
    X = np.column_stack([y, np.zeros(len(y))])
    halves = np.split(np.arange(len(y)), 2)
    folds = [halves, halves[::-1]]

    def column_selector(changed):
        words = field_array(difference, changed)

        def select(rows):
            return rows[:, [field_choice(words, difference)]]

        return FunctionTransformer(select)

    neighbors = [1, 3]
    expected = []
    for column in (1, 0):
        for count in neighbors:
            scores = []
            for train, valid in folds:
                model = KNeighborsClassifier(n_neighbors=count).fit(X[train, column:column+1], y[train])
                scores.append(model.score(X[valid, column:column+1], y[valid]))
            expected.append(scores)
    assert expected == [[.5, .5], [.5, .5], [1., 1.], [1., 1.]]

    def run(kinds):
        estimator = Pipeline([("features", FunctionTransformer()), ("knn", KNeighborsClassifier())])
        grid = {"features": [column_selector(kind) for kind in kinds], "knn__n_neighbors": neighbors}
        return calibration_knn_grid(estimator, grid, X, y, cv=2, folds=folds, threads=1,
                                    checkpoint_dir=tmp_path)

    if partial:
        publish = study._write_receipt

        def stop_after_first_fold(path, payload):
            publish(path, payload)
            if path.parent.name == "folds":
                raise FoldPublished

        with patch.object(study, "_write_receipt", stop_after_first_fold), pytest.raises(FoldPublished):
            run((True, False))
        with (patch.object(study, "_write_receipt", wraps=study._write_receipt) as write,
              patch.object(study, "_run_representation_fold", wraps=study._run_representation_fold) as fit,
              pytest.raises(ValueError, match="incompatible")):
            run((False, True))
        write.assert_not_called()
        fit.assert_not_called()
    report = run((True, False))
    assert report["unique_representations"] == 2
    assert report["new_representation_fits"] == (3 if partial else 4)
    assert report["reused_representation_folds"] == (1 if partial else 0)
    assert all(row["valid"] for row in report["candidates"])
    assert [[fold["score"] for fold in row["folds"]] for row in report["candidates"]] == expected
    assert report["ranked_candidate_indices"] == [2, 3, 0, 1]
    assert report["selected_mean_score"] == 1.0


@pytest.fixture
def scalar_flag_data():
    y = np.tile([0, 1], 20)
    X = np.column_stack([y, np.zeros(len(y))])
    halves = np.split(np.arange(len(y)), 2)
    return X, y, [halves, halves[::-1]]


def scalar_flag_selector(writeable):
    owner = np.zeros(1, dtype=[("word", "u1")])
    owner.setflags(write=writeable)
    scalar = owner[0]
    assert type(scalar) is np.void and scalar.base is not None
    assert scalar.tobytes() == b"\x00" and scalar.flags.writeable == writeable

    def select(rows):
        return rows[:, [int(scalar.flags.writeable)]]

    return select


def scalar_flag_pipeline(selector=None, neighbors=1):
    return Pipeline([("features", FunctionTransformer(selector)),
                     ("knn", KNeighborsClassifier(n_neighbors=neighbors))])


@pytest.mark.parametrize("first_writeable", [False, True], ids=["readonly-first", "writable-first"])
def test_shared_grid_matches_standalone_scalar_flags(tmp_path, scalar_flag_data, first_writeable):
    X, y, folds = scalar_flag_data
    neighbors = [1, 3]
    standalone = {}
    for writeable in (False, True):
        for count in neighbors:
            model = scalar_flag_pipeline(scalar_flag_selector(writeable), count)
            scores = [model.fit(X[train], y[train]).score(X[valid], y[valid]) for train, valid in folds]
            assert scores == ([.5, .5] if writeable else [1., 1.])
            standalone[writeable, count] = scores

    # Rebuild the first callback independently to exercise genuine representation sharing.
    declarations = [first_writeable, not first_writeable, first_writeable]
    grid = {"features__func": [scalar_flag_selector(flag) for flag in declarations],
            "knn__n_neighbors": neighbors}
    report = calibration_knn_grid(scalar_flag_pipeline(), grid, X, y, cv=2, folds=folds,
                                  threads=1, checkpoint_dir=tmp_path)
    assert all(row["valid"] for row in report["candidates"]) and not report["is_fallback"]
    scores = [[fold["score"] for fold in row["folds"]] for row in report["candidates"]]
    expected = [standalone[flag, count] for flag in declarations for count in neighbors]
    assert scores == expected, "shared scores must match standalone scalar flag selectors"
    assert report["unique_representations"] == 2 and report["new_representation_fits"] == 4
    assert report["ranked_candidate_indices"] == sorted(range(len(expected)), key=lambda i: -np.mean(expected[i]))
    assert report["selected_mean_score"] == 1.


@pytest.mark.parametrize("partial,writer_flag", [(False, True), (True, False)], ids=["complete", "partial"])
def test_calibration_resumes_only_matching_scalar_flags(tmp_path, scalar_flag_data, partial, writer_flag):
    from supervised_nmf import evaluation

    X, y, folds = scalar_flag_data
    path = tmp_path / "folds.json"

    def run(writeable, checkpoint=None):
        model = scalar_flag_pipeline(scalar_flag_selector(writeable))
        return calibration_search(model, {}, X, y, cv=2, folds=folds, threads=1,
                                  refit=False, checkpoint_path=checkpoint).report

    def scores(report):
        row, = report["candidates"]
        assert row["valid"] and not report["is_fallback"]
        return [fold["score"] for fold in row["folds"]]

    reference = {flag: scores(run(flag)) for flag in (False, True)}
    assert reference == {False: [1., 1.], True: [.5, .5]}
    if partial:
        publish = evaluation.write_json

        def stop_after_first_fold(destination, report):
            publish(destination, report)
            if report["candidates"][0]["folds"]:
                raise FoldPublished

        with patch.object(evaluation, "write_json", stop_after_first_fold), pytest.raises(FoldPublished):
            run(writer_flag, path)
    else:
        assert scores(run(writer_flag, path)) == reference[writer_flag]

    before = path.read_bytes()
    saved = json.loads(before)["candidates"][0]["folds"]
    assert [fold["score"] for fold in saved] == reference[writer_flag][:1 if partial else 2]
    with (patch.object(Pipeline, "fit", autospec=True, side_effect=Pipeline.fit) as fit,
          patch.object(evaluation, "write_json", wraps=evaluation.write_json) as write):
        with pytest.raises(ValueError, match="incompatible"):
            run(not writer_flag, path)
        fit.assert_not_called()
        write.assert_not_called()
    assert path.read_bytes() == before

    with patch.object(Pipeline, "fit", autospec=True, side_effect=Pipeline.fit) as fit:
        recovered = run(writer_flag, path)
    assert fit.call_count == len(folds) - len(saved)
    assert recovered["candidates"][0]["folds"][:len(saved)] == saved
    assert scores(recovered) == reference[writer_flag]


@pytest.mark.parametrize("opaque", [False, True], ids=["small", "external-buffer"])
def test_scalar_storage_shared_grid_and_checkpoint(tmp_path, scalar_flag_data, opaque):
    from supervised_nmf import study

    X, y, folds = scalar_flag_data

    def selector(column):
        dt = np.dtype([("word", "u4")])
        array = (np.ndarray(2, dtype=dt, buffer=bytearray(2 * dt.itemsize))
                 if opaque else np.zeros(2, dtype=dt))
        array["word"][1] = column
        item = array[0]

        def select(rows):
            return rows[:, [int(item.base["word"][1])]]

        return select

    def run(columns):
        return calibration_knn_grid(
            scalar_flag_pipeline(), {"features__func": [selector(c) for c in columns],
                                     "knn__n_neighbors": [1, 3]}, X, y,
            cv=2, folds=folds, threads=1, checkpoint_dir=tmp_path)

    report = run([0, 1])
    assert [[f["score"] for f in c["folds"]] for c in report["candidates"]] == [
        [1., 1.], [1., 1.], [.5, .5], [.5, .5]]
    assert report["unique_representations"] == 2 and report["new_representation_fits"] == 4
    if not opaque:
        assert run([0, 1])["new_representation_fits"] == 0
    archive = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with patch.object(study, "_write_receipt", wraps=study._write_receipt) as write:
        if opaque:
            with pytest.raises(ValueError, match="backing storage"):
                run([0, 1])
        with pytest.raises(ValueError, match="incompatible|backing storage"):
            run([1, 0])
        write.assert_not_called()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == archive


NESTED_RECORD_STORAGE = [pytest.param(np.void, id="void"), pytest.param(np.record, id="record"),
                         pytest.param(np.object_, id="named-object"), pytest.param(object, id="plain-object")]


def nested_record_selector(index, scalar_type):
    fields = [("saved", object)]
    dtype = object if scalar_type is object else np.dtype((scalar_type, fields))
    owner = np.zeros(3, dtype=dtype)
    if scalar_type in (np.void, np.record):
        # The outer record is synthesized on indexing. Its field holds a real scalar.
        saved = owner[index]
        owner["saved"][0] = saved
        assert type(saved) is scalar_type and saved.base is owner and owner["saved"][0] is saved
    else:
        # Object-kind arrays return the stored object itself, even with a named field.
        backing = np.ndarray(3, dtype=fields, buffer=owner)
        saved = backing[index]
        slots = owner["saved"] if owner.dtype.names else owner
        slots[0] = saved
        assert saved.base is backing and backing.base is owner and owner.flat[0] is saved

    def select(rows):
        value = owner["saved"][0] if owner.dtype.names else owner[0]
        return rows[:, [int(np.shares_memory(np.asarray(value), owner[1:2]))]]

    return select


@pytest.mark.parametrize("scalar_type", NESTED_RECORD_STORAGE)
@pytest.mark.parametrize("order", [(1, 2), (2, 1)], ids=["first-row-first", "second-row-first"])
def test_grid_matches_independent_nested_record_selectors(tmp_path, scalar_flag_data, scalar_type, order):
    X, y, folds = scalar_flag_data
    callbacks = [nested_record_selector(index, scalar_type) for index in order]
    expected = []
    for index, callback in zip(order, callbacks):
        for neighbors in (1, 3):
            scores = [scalar_flag_pipeline(callback, neighbors).fit(X[t], y[t]).score(X[v], y[v])
                      for t, v in folds]
            assert scores == ([.5, .5] if index == 1 else [1., 1.])
            expected.append(scores)

    report = calibration_knn_grid(
        scalar_flag_pipeline(), {"features__func": callbacks, "knn__n_neighbors": [1, 3]},
        X, y, cv=2, folds=folds, threads=1, checkpoint_dir=tmp_path)
    assert [[f["score"] for f in c["folds"]] for c in report["candidates"]] == expected
    assert report["unique_representations"] == 2 and report["new_representation_fits"] == 4
    assert report["ranked_candidate_indices"] == sorted(range(4), key=lambda i: -np.mean(expected[i]))


@pytest.mark.parametrize("scalar_type", NESTED_RECORD_STORAGE)
@pytest.mark.parametrize("reader", [1, 2], ids=["same", "changed"])
def test_grid_refuses_checkpoint_with_nested_record_scalar(tmp_path, scalar_flag_data, scalar_type, reader):
    from supervised_nmf import study

    X, y, folds = scalar_flag_data

    def run(index):
        return calibration_knn_grid(scalar_flag_pipeline(nested_record_selector(index, scalar_type)),
                                    {"knn__n_neighbors": [1]}, X, y, cv=2, folds=folds,
                                    threads=1, checkpoint_dir=tmp_path)

    report = run(1)
    assert [f["score"] for f in report["candidates"][0]["folds"]] == [.5, .5]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with (patch.object(study, "_write_receipt") as write, patch.object(study, "_run_representation_fold") as fit,
          pytest.raises(ValueError, match="backing storage|incompatible")):
        run(reader)
    fit.assert_not_called()
    write.assert_not_called()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_grid_reuses_synthesized_record_scalars(tmp_path, scalar_flag_data):
    X, y, folds = scalar_flag_data

    def selector():
        owner = np.zeros(2, dtype=[("column", object)])

        def select(rows):
            return rows[:, [owner[0]["column"]]]

        return select

    for expected_fits in (2, 0):
        grid = {"features__func": [selector(), selector()], "knn__n_neighbors": [1, 3]}
        report = calibration_knn_grid(scalar_flag_pipeline(), grid, X, y, cv=2, folds=folds,
                                      threads=1, checkpoint_dir=tmp_path)
        assert [[f["score"] for f in c["folds"]] for c in report["candidates"]] == [[1., 1.]] * 4
        assert report["unique_representations"] == 1 and report["new_representation_fits"] == expected_fits


def test_scalar_storage_offset_and_shared_view():
    def choose(shared, index):
        owner = np.zeros(2, dtype=[("word", "u4")])
        scalar = owner[0]
        view = (owner if shared else owner.copy())[index:index + 1]

        def matches():
            return np.shares_memory(np.asarray(scalar), view)

        return matches

    assert choose(True, 0)() and not choose(True, 1)() and not choose(False, 0)()
    assert len({identity_digest(choose(*case))
                for case in ((True, 0), (True, 1), (False, 0))}) == 3
    assert identity_digest(choose(True, 0)) == identity_digest(choose(True, 0))


@pytest.mark.parametrize("exporter", ["scalar", "memoryview"])
def test_indirect_array_view_keeps_scalar_callbacks_independent(tmp_path, scalar_flag_data, exporter):
    X, y, folds = scalar_flag_data

    def selector(shared):
        owner = np.zeros(2, dtype=[("word", "u4")])
        scalar = owner[0]
        other = owner if shared else owner.copy()
        view = (np.asarray(scalar if shared else other[0]) if exporter == "scalar"
                else np.frombuffer(memoryview(other), dtype=other.dtype, count=1))

        def select(rows):
            return rows[:, [int(np.shares_memory(np.asarray(scalar), view))]]

        return select

    callbacks = [selector(shared) for shared in (False, True)]
    params = {"features__func": callbacks, "knn__n_neighbors": [1, 3]}

    def run():
        return calibration_knn_grid(scalar_flag_pipeline(), params, X, y, cv=2, folds=folds,
                                    threads=1, checkpoint_dir=tmp_path)

    report = run()
    assert report["new_representation_fits"] == 4
    assert [[f["score"] for f in c["folds"]] for c in report["candidates"]] == [
        [1., 1.], [1., 1.], [.5, .5], [.5, .5]]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="backing storage"):
        run()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_grid_callback_can_share_scalar_storage_with_fixed_transformer(tmp_path, scalar_flag_data):
    from sklearn.base import clone
    from sklearn.model_selection import ParameterGrid

    X, y, folds = scalar_flag_data
    owner = np.zeros(2, dtype=[("word", "u4")])

    def expose(rows):
        return rows, owner

    def choose(scalar):
        def select(pair):
            rows, array = pair
            return rows[:, [int(np.shares_memory(scalar.base, array))]]

        return select

    model = Pipeline([("expose", FunctionTransformer(expose)), ("select", FunctionTransformer()),
                      ("knn", KNeighborsClassifier())])
    grid = {"select__func": [choose(owner.copy()[0]), choose(owner[0])], "knn__n_neighbors": [1, 3]}
    expected = []
    for params in ParameterGrid(grid):
        actual = clone(model).set_params(**params)
        expected.append([actual.fit(X[train], y[train]).score(X[valid], y[valid]) for train, valid in folds])
    assert sorted(expected) == [[.5, .5], [.5, .5], [1., 1.], [1., 1.]]
    for new_fits in (4, 0):
        report = calibration_knn_grid(model, grid, X, y, cv=2, folds=folds, threads=1, checkpoint_dir=tmp_path)
        assert report["new_representation_fits"] == new_fits and report["unique_representations"] == 2
        assert [[f["score"] for f in c["folds"]] for c in report["candidates"]] == expected


def test_rank_search_refuses_opaque_scalar_before_first_stage(tmp_path, scalar_flag_data):
    from sklearn.decomposition import PCA
    from supervised_nmf import search

    X, y, folds = scalar_flag_data
    dtype = np.dtype([("word", "u4")])
    scalar = np.ndarray(2, dtype=dtype, buffer=bytearray(8))[0]

    def select(rows):
        return rows[:, [int(scalar.base["word"][1])]]

    estimator = Pipeline([("select", FunctionTransformer(select)), ("reduce", PCA(random_state=1)),
                          ("knn", KNeighborsClassifier(n_neighbors=1))])

    def run():
        return search.calibration_rank_search(estimator, {"reduce__n_components": [1]}, X, y,
                                              cv=2, folds=folds, threads=1, checkpoint_dir=tmp_path)

    with patch.object(search, "calibration_knn_grid", side_effect=FoldPublished), pytest.raises(FoldPublished):
        run()
    before = (tmp_path / "identity.json").read_bytes()
    scalar.base["word"][1] = 1
    with patch.object(search, "calibration_knn_grid") as stage, pytest.raises(ValueError, match="backing storage"):
        run()
    stage.assert_not_called()
    assert (tmp_path / "identity.json").read_bytes() == before and not list(tmp_path.glob("stage-*"))


@pytest.mark.parametrize("update_scalar", [True, False], ids=["scalar-owner", "captured-array"])
def test_shared_grid_refuses_to_skip_a_scalar_update(tmp_path, scalar_flag_data, update_scalar):
    from supervised_nmf import study

    X, y, folds = scalar_flag_data

    def run(directory):
        scalar = np.zeros(2, dtype=[("used", "i4")])[0]
        counter = np.zeros(1, dtype=np.int64)

        def select(rows):
            if update_scalar:
                used = int(scalar.base["used"][1])
                scalar.base["used"][1] = used + 1
            else:
                used = int(counter[0])
                counter[0] = used + 1
            return rows[:, [(used // 2) % 2]]

        return calibration_knn_grid(scalar_flag_pipeline(select), {"knn__n_neighbors": [1]},
                                    X, y, cv=2, folds=folds, threads=1, checkpoint_dir=directory)

    full = run(tmp_path / "whole")
    assert [f["score"] for f in full["candidates"][0]["folds"]] == [1., .5]
    directory = tmp_path / "interrupted"
    publish = study._write_receipt

    def stop_after_fold(path, record):
        publish(path, record)
        if path.parent.name == "folds":
            raise FoldPublished

    with patch.object(study, "_write_receipt", stop_after_fold), pytest.raises(FoldPublished):
        run(directory)
    before = {p: p.read_bytes() for p in directory.rglob("*.json")}
    with (patch.object(study, "_run_representation_fold") as fit, patch.object(study, "_write_receipt") as write,
          pytest.raises(ValueError, match="changed during a saved fold")):
        run(directory)
    fit.assert_not_called()
    write.assert_not_called()
    assert {p: p.read_bytes() for p in directory.rglob("*.json")} == before


@pytest.mark.parametrize("fault,expected", [("unreadable", "implementation source cannot be read"),
                                           ("edited", "Source or numerical runtime changed")])
def test_scalar_grid_checks_callback_source_before_committing_fold(tmp_path, scalar_flag_data, fault, expected):
    from supervised_nmf import study

    X, y, folds = scalar_flag_data
    source = Path(__file__).resolve()
    read_bytes = Path.read_bytes
    run_fold = study._run_representation_fold
    finished = False

    def after_fit(*args, **kwargs):
        nonlocal finished
        run_fold(*args, **kwargs)
        finished = True

    def read(path):
        if finished and path == source:
            if fault == "unreadable":
                raise OSError("callback source unavailable")
            return read_bytes(path) + b"\n# edited\n"
        return read_bytes(path)

    with (patch.object(study, "_run_representation_fold", after_fit), patch.object(Path, "read_bytes", read),
          pytest.raises(ValueError, match=expected)):
        calibration_knn_grid(scalar_flag_pipeline(scalar_flag_selector(True)), {"knn__n_neighbors": [1]},
                             X, y, cv=2, folds=folds, threads=1, checkpoint_dir=tmp_path)
    assert finished and not list((tmp_path / "folds").glob("*.json"))


def word_type(column):
    class TaggedWord(np.void):
        @property
        def feature(self):
            return column

    return TaggedWord


@pytest.mark.parametrize("attachment", ["class", "class_specifier", "instance", "metadata"])
def test_opaque_structured_parameters_only_share_within_a_declaration(tmp_path, attachment):
    def selector(column):
        if attachment == "class":
            words = np.ones(1, dtype=(word_type(column), [("word", "u4")]))

            def choose(rows):
                return rows[:, [words[0].feature]]
        elif attachment == "class_specifier":
            cls = word_type(column)

            def choose(rows):
                return rows[:, [field_choice(cls, "class_specifier")]]
        else:
            scalar = np.ones(1, dtype=np.dtype((np.record, [("word", "u4")])))[0]
            # NumPy keeps this extra Python attribute outside the array bytes.
            object.__setattr__(scalar, "feature", column)
            if attachment == "instance":
                def choose(rows):
                    return rows[:, [scalar.feature]]
            else:
                dtype = np.dtype([("word", "u4")], metadata={"selector": scalar})

                def choose(rows):
                    return rows[:, [dtype.metadata["selector"].feature]]
        return FunctionTransformer(choose)

    y = np.tile([0, 1], 20)
    X = np.column_stack([y, np.zeros(len(y))])
    halves = np.split(np.arange(len(y)), 2)
    folds = [halves, halves[::-1]]
    estimator = Pipeline([("features", FunctionTransformer()), ("knn", KNeighborsClassifier())])
    grid = {"features": [selector(1), selector(0)], "knn__n_neighbors": [1, 3]}

    def run():
        return calibration_knn_grid(estimator, grid, X, y, cv=2, folds=folds, threads=1,
                                    checkpoint_dir=tmp_path)

    report = run()
    assert report["unique_representations"] == 2 and report["new_representation_fits"] == 4
    assert not report["is_fallback"] and all(row["valid"] for row in report["candidates"])
    assert [[f["score"] for f in row["folds"]] for row in report["candidates"]] == [
        [.5, .5], [.5, .5], [1., 1.], [1., 1.]]
    with pytest.raises(ValueError, match="unsupported fit randomness"):
        run()


@pytest.mark.parametrize("partial", [False, True])
def test_custom_void_class_specifier_requires_fresh_calibration(tmp_path, partial):
    from supervised_nmf import evaluation

    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 14 + [1] * 6)
    path = tmp_path / "cv.json"
    writer, reader = (FieldClassifier(word_type(label), "class_specifier") for label in (0, 1))

    def run(estimator, checkpoint=None):
        result = calibration_search(estimator, {}, X, y, cv=2, seed=17, threads=1,
                                    refit=False, checkpoint_path=checkpoint)
        row, = result.report["candidates"]
        assert row["valid"] and not result.report["is_fallback"]
        return [fold["score"] for fold in row["folds"]]

    assert run(writer) == [.7, .7] and run(reader) == [.3, .3]
    if partial:
        publish = evaluation.write_json

        def stop(destination, report):
            publish(destination, report)
            if report["candidates"][0]["folds"]:
                raise FoldPublished

        with patch.object(evaluation, "write_json", stop), pytest.raises(FoldPublished):
            run(writer, path)
    else:
        assert run(writer, path) == [.7, .7]
    before = path.read_bytes()
    with (patch.object(FieldClassifier, "fit", autospec=True) as fit,
          patch.object(evaluation, "write_json", wraps=evaluation.write_json) as write,
          pytest.raises(ValueError, match="verifiable")):
        run(reader, path)
    fit.assert_not_called()
    write.assert_not_called()
    assert path.read_bytes() == before


def test_source_opaque_custom_void_still_fits_and_declines_completed_cv(tmp_path):
    # An importable estimator can receive a dtype created in an interactive
    # session. Its scalar class need not have a source file.
    class Word(np.void):
        word = 1

    assert any(Word.__qualname__ in key for key in implementation_sources(np.dtype((Word, "u4"))))
    words = np.ones(1, dtype=np.dtype((Word, [("word", "u4")])))
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 14 + [1] * 6)
    path = tmp_path / "cv.json"
    Word.__module__ = "interactive_dtype_without_a_file"

    def run(checkpoint):
        return calibration_search(FieldClassifier(words, "record"), {}, X, y, cv=2, seed=17, threads=1,
                                  refit=False, checkpoint_path=checkpoint).report

    for checkpoint in (None, path):
        report = run(checkpoint)
        row, = report["candidates"]
        assert row["valid"] and not report["is_fallback"]
        assert [f["score"] for f in row["folds"]] == [.3, .3]
    before = path.read_bytes()
    with patch.object(FieldClassifier, "fit", autospec=True) as fit, pytest.raises(ValueError, match="verifiable"):
        run(path)
    fit.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("array", [
    np.ma.array([1., 2.], mask=[False, True]),
    np.arange(4.).view(type("ArraySubclass", (np.ndarray,), {})),
])
def test_array_subclass_state_is_not_silently_discarded(array):
    with pytest.raises(ValueError, match="ndarray subclass"):
        identity_digest(array)
    with pytest.raises(ValueError, match="ndarray subclass"):
        implementation_sources(array)


@pytest.mark.parametrize("value", [PartialSubclass(abs), ScalarSubclass(1.)])
def test_custom_partial_and_numpy_scalar_subclasses_are_rejected(value):
    with pytest.raises(ValueError, match="reusable search identity"):
        identity_digest(value)
    with pytest.raises(ValueError, match="reusable search identity"):
        implementation_sources(value)


def test_class_dtype_path_and_slice_values_are_supported():
    values = [np.float32, float, np.dtype("float32"), Path("relative/file"), slice(None, 4)]
    for value in values:
        record = configuration_identity(value)
        assert isinstance(record, dict)
        json.dumps(record, allow_nan=False)
    assert identity_digest(np.float32) != identity_digest(np.float64)


def test_random_state_and_generator_snapshots_do_not_advance_them():
    for make in (lambda: np.random.RandomState(17), lambda: np.random.default_rng(17)):
        first, reference = make(), make()
        original = identity_digest(first)
        assert identity_digest(first) == original
        np.testing.assert_array_equal(first.random(8), reference.random(8))
        assert identity_digest(first) != original


def test_functions_defaults_closures_and_partials_are_distinct():
    assert identity_digest(closure_function(1)) != identity_digest(closure_function(2))
    assert identity_digest(functools.partial(identity_function, offset=1)) != identity_digest(
        functools.partial(identity_function, offset=2))
    alternate = types.FunctionType(identity_function.__code__, identity_function.__globals__,
                                   identity_function.__name__, (3.,))
    alternate.__module__ = identity_function.__module__
    alternate.__qualname__ = identity_function.__qualname__
    alternate.__kwdefaults__ = dict(identity_function.__kwdefaults__)
    assert identity_digest(alternate) != identity_digest(identity_function)
    changed = types.FunctionType(identity_function.__code__, identity_function.__globals__,
                                 identity_function.__name__, identity_function.__defaults__)
    changed.__module__ = identity_function.__module__
    changed.__qualname__ = identity_function.__qualname__
    changed.__kwdefaults__ = {"power": 3}
    assert identity_digest(changed) != identity_digest(identity_function)
    assert identity_digest(np.add) != identity_digest(np.multiply)
    assert identity_digest(np.mean) != identity_digest(np.sum)
    json.dumps(configuration_identity(len), allow_nan=False)


@pytest.mark.parametrize("value", [object(), CustomCloneEstimator(), FrozenEstimator(StandardScaler())])
def test_opaque_or_state_preserving_clone_values_are_rejected(value):
    with pytest.raises(ValueError, match="reusable search identity"):
        configuration_identity(value)
    with pytest.raises(ValueError, match="reusable search identity"):
        implementation_sources(value)


@pytest.mark.parametrize("search", ["generic", "shared", "rank"])
@pytest.mark.parametrize("location", ["base", "replacement"])
def test_custom_clone_cannot_erase_its_identity_before_search_validation(
        search, location, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(CloneBypassEstimator, "clone_calls", calls)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Search must reject the custom clone before numerical fitting")

    monkeypatch.setattr(PCA, "fit_transform", unexpected_fit)
    estimator = Pipeline([("pca", PCA(n_components=1, random_state=0)),
                          ("knn", KNeighborsClassifier(n_neighbors=1))])
    grid = {"pca__n_components": [1]}
    if location == "base":
        estimator = CloneBypassEstimator()
    else:
        grid["pca"] = [CloneBypassEstimator(role="transformer")]
    X = np.arange(48., dtype=float).reshape(12, 4)
    y = np.tile([0, 1], 6)
    with pytest.raises(ValueError, match="__sklearn_clone__"):
        if search == "generic":
            calibration_search(estimator, grid, X, y, cv=2, refit=False)
        elif search == "shared":
            calibration_knn_grid(estimator, grid, X, y, cv=2, checkpoint_dir=tmp_path)
        else:
            calibration_rank_search(estimator, grid, X, y, cv=2,
                                    max_expansions=0, checkpoint_dir=tmp_path)
    assert calls == []


def test_cyclic_graph_is_rejected():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cyclic"):
        identity_digest(value)
    array = np.empty(1, dtype=object)
    array[0] = array
    with pytest.raises(ValueError, match="cyclic"):
        identity_digest(array)


def test_uninspectable_python_implementation_is_rejected():
    namespace = {"__name__": "unsaved_custom_module"}
    exec(compile("def weights(values):\n    return values\n", "<unsaved>", "exec"), namespace)
    with pytest.raises(ValueError, match="source is unavailable"):
        identity_digest(namespace["weights"])


def test_source_only_path_skips_numeric_array_content_hashes():
    array = np.arange(1000.)
    with patch("supervised_nmf._identity.np.ascontiguousarray",
               side_effect=AssertionError("source guard must not scan array contents")):
        result = implementation_sources(SameReprEstimator(array=array))
    assert result
    assert any("SameReprEstimator" in name for name in result)
    assert any("BaseEstimator" in name for name in result)


def test_source_file_changes_are_seen_on_next_call(tmp_path):
    path = tmp_path / "identity_custom_source.py"
    first = "from sklearn.base import BaseEstimator\nclass Custom(BaseEstimator):\n    def fit(self, X, y=None):\n        return self\n"
    second = first + "\n# implementation edit that must invalidate cached evidence\n"
    path.write_text(first, encoding="utf-8")
    name = "identity_custom_source"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        value = module.Custom()
        before = implementation_sources({"replacement": value})
        assert hashlib.sha256(path.read_bytes()).hexdigest() in before.values()
        path.write_text(second, encoding="utf-8")
        after = implementation_sources({"replacement": value})
        assert after != before
        assert hashlib.sha256(path.read_bytes()).hexdigest() in after.values()
    finally:
        sys.modules.pop(name, None)
