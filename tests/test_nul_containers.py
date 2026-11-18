"""Read list/tuple label storage without invoking subclass protocols."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from supervised_nmf import BinaryScreening, NMFClassifier
from supervised_nmf import _labels
from supervised_nmf.evaluation import calibration_search, data_fingerprint, evaluate_classifier, run_comparison
from supervised_nmf.search import calibration_rank_search
from supervised_nmf.study import calibration_knn_grid
from supervised_nmf.validation import build_cv_folds


class LabelList(list):
    pass


class LabelTuple(tuple):
    pass


class OverrideTrap:
    def __iter__(self):
        raise AssertionError("guard dispatched custom iteration")

    def __getitem__(self, key):
        raise AssertionError("guard dispatched custom indexing")

    def __len__(self):
        raise AssertionError("guard dispatched custom length")

    @property
    def __class__(self):
        raise AssertionError("guard inspected overridden __class__")


class TrapList(OverrideTrap, list):
    pass


class TrapTuple(OverrideTrap, tuple):
    pass


class ClassSpoof:
    def __init__(self, claimed):
        self.claimed = claimed
        self.lookups = 0

    @property
    def __class__(self):
        self.lookups += 1
        return self.claimed


class StringClassTrap(str):
    @property
    def __class__(self):
        raise AssertionError("actual string subclass needs no __class__ lookup")

    def endswith(self, *args, **kwargs):
        raise AssertionError("guard must call the builtin string descriptor")


@pytest.fixture
def data():
    codes = np.arange(24) % 2
    X = np.random.default_rng(862).uniform(.1, 1., (len(codes), 6))
    X[codes == 1, :3] += 2
    ordinary = np.array(["a" if code == 0 else "b" for code in codes], dtype=object)
    return X, codes, ordinary


@pytest.mark.parametrize("outer", [LabelList, LabelTuple])
@pytest.mark.parametrize("inner", [None, LabelList, LabelTuple])
@pytest.mark.parametrize("scalar", [str, np.str_])
def test_plain_subclass_targets_rejected_before_screening(data, outer, inner, scalar):
    X, codes, _ = data
    values = [scalar("a\0" if code == 0 else "b") for code in codes]
    y = outer(values if inner is None else [inner([value]) for value in values])
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        BinaryScreening(screening="percentile", percentile=100).fit(X, y)


@pytest.mark.parametrize("family", [LabelList, LabelTuple])
@pytest.mark.parametrize("entrypoint", [
    "classifier", "score", "folds", "calibration", "knn_grid", "rank_search",
    "evaluation", "train_y", "fitted_classes", "fingerprint", "comparison_training",
    "comparison_evaluation",
])
def test_package_entrypoints_retain_subclass_raw_values(data, family, entrypoint, tmp_path):
    X, codes, ordinary = data
    labels = family(["a\0" if code == 0 else "b" for code in codes])
    classifier = NMFClassifier(screening="none", n_components=2, n_neighbors=1)
    destination = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        if entrypoint == "classifier":
            classifier.fit(X, labels)
        elif entrypoint == "score":
            classifier.fit(X, codes).score(X, labels)
        elif entrypoint == "folds":
            build_cv_folds(X, labels, cv=2)
        elif entrypoint == "calibration":
            calibration_search(DummyClassifier(), {}, X, labels, cv=2,
                               checkpoint_path=destination / "cv.json")
        elif entrypoint == "knn_grid":
            calibration_knn_grid(classifier, {"n_neighbors": [1]}, X, labels, cv=2,
                                 checkpoint_dir=destination)
        elif entrypoint == "rank_search":
            calibration_rank_search(classifier, {"n_components": [2]}, X, labels, cv=2,
                                    checkpoint_dir=destination)
        elif entrypoint == "evaluation":
            evaluate_classifier(DummyClassifier().fit(X, ordinary), X, labels)
        elif entrypoint == "train_y":
            evaluate_classifier(DummyClassifier().fit(X, ordinary), X, ordinary, train_y=labels)
        elif entrypoint == "fitted_classes":
            fitted = DummyClassifier().fit(X, ordinary)
            fitted.classes_ = family(["a\0", "b"])
            evaluate_classifier(fitted, X, ordinary)
        elif entrypoint == "fingerprint":
            data_fingerprint(X, labels)
        elif entrypoint == "comparison_training":
            run_comparison({"dummy": (DummyClassifier(), {})}, X, labels, X, ordinary,
                           output_dir=destination)
        else:
            run_comparison({"dummy": (DummyClassifier(), {})}, X, ordinary, X, labels,
                           output_dir=destination)
    assert not destination.exists()


@pytest.mark.parametrize("family", [LabelList, LabelTuple])
@pytest.mark.parametrize("pair", [("a", "b"), ("a\0x", "b\0y"), ("\0a", "\0b")])
def test_ordinary_subclass_targets_preserve_statistics_and_labels(data, family, pair):
    X, codes, _ = data
    labels = family([pair[int(code)] for code in codes])
    reference = BinaryScreening(screening="percentile", percentile=100).fit(X, codes)
    model = BinaryScreening(screening="percentile", percentile=100).fit(X, labels)
    assert model.classes_.tolist() == list(pair)
    np.testing.assert_array_equal(model.t_scores_, reference.t_scores_)
    np.testing.assert_array_equal(model.get_support(), reference.get_support())


@pytest.mark.parametrize("outer", [TrapList, TrapTuple])
@pytest.mark.parametrize("inner", [None, TrapList, TrapTuple])
@pytest.mark.parametrize("terminal", [False, True])
def test_guard_reads_base_storage_without_iteration_indexing_length_or_class_dispatch(outer, inner, terminal):
    values = [StringClassTrap("a\0" if terminal else "a"), "b"]
    raw = outer(values if inner is None else [inner([value]) for value in values])
    validated = np.array(["a", "b"])
    if terminal:
        with pytest.raises(ValueError, match="Class labels ending in a NUL"):
            _labels._validate_class_labels(raw, validated)
    else:
        assert _labels._validate_class_labels(raw, validated) is validated


@pytest.mark.parametrize("family", [TrapList, TrapTuple])
def test_fallback_restarts_base_iterator_for_first_column_and_later_mixed_value(family):
    validated = np.array(["a", "b"])
    for raw in (family([TrapList(["a\0"]), TrapTuple(["b"])]),
                family([7, "ordinary", "b\0"])):
        with pytest.raises(ValueError, match="Class labels ending in a NUL"):
            _labels._validate_class_labels(raw, validated)


@pytest.mark.parametrize("family", [TrapList, TrapTuple])
def test_numeric_validated_dtype_bypasses_subclass_storage(family, monkeypatch):
    raw = family([0, 1, 0, 1])
    validated = np.array([0, 1, 0, 1])
    def forbidden(values):
        raise AssertionError("numeric labels must not trigger raw storage scan")
    monkeypatch.setattr(_labels, "_raw_terminal_nul", forbidden)
    assert _labels._validate_class_labels(raw, validated, classes=np.array([0, 1])) is validated


@pytest.mark.parametrize("claimed", [str, list, tuple])
def test_unrelated_object_class_spoof_is_not_consulted(claimed):
    spoof = ClassSpoof(claimed)
    ordinary = np.array(["a", "b"])
    assert _labels._validate_class_labels(spoof, ordinary) is ordinary
    assert _labels._validate_class_label(spoof) is None
    objects = np.empty(2, dtype=object)
    objects[0], objects[1] = spoof, "ordinary"
    assert _labels._validate_class_labels(objects, objects) is objects
    with pytest.raises(ValueError, match="Class labels ending in a NUL"):
        _labels._validate_class_labels(LabelList([spoof, "a\0"]), ordinary)
    assert spoof.lookups == 0


def test_column_inspection_remains_bounded_for_cyclic_and_deep_subclasses():
    cyclic = LabelList()
    cyclic.append(cyclic)
    deep = LabelList([LabelTuple([LabelList(["a\0"])])])
    validated = np.array(["ordinary"])
    # These are invalid target shapes: this helper must not recursively search
    # arbitrary nesting or hang; public shape validation rejects them first.
    assert _labels._validate_class_labels(cyclic, validated) is validated
    assert _labels._validate_class_labels(deep, validated) is validated
