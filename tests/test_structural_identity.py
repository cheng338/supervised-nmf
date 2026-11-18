"""Strict reusable identity must not depend on estimator display text."""
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from supervised_nmf._identity import (
    configuration_identity, identity_digest, implementation_sources,
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
