"""Estimator input and target metadata contracts. Author: Chang Cheng."""

from dataclasses import asdict
import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import ClassifierTags, get_tags
from sklearn.utils import estimator_checks

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier


class _BinaryCheckerScreen(BinaryScreening):
    """Supply binary fixtures to generic checks without changing the public API.

    The generic checker reads this classifier-specific metadata to prepare
    target values. This subclass is confined to tests; BinaryScreening remains
    a transformer with no classifier tags or prediction/scoring methods.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        return tags


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(820)
    y = np.arange(36) % 2
    X = rng.uniform(.1, 1.2, (len(y), 8))
    X[:, :3] += y[:, None]
    query = rng.uniform(.1, 1.7, (7, X.shape[1]))
    query[0] = 0.
    return X, y, query


def estimator(kind):
    if kind == "classifier":
        return NMFClassifier(n_components=2, screening="none", n_neighbors=3,
                             random_state=17, max_iter=1500, tol=1e-3)
    if kind == "spa":
        return FeatureSPA(n_components=2, nnls_max_iter=1500)
    return BinaryScreening(percentile=100.)


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
def test_constraint_tags_describe_the_existing_interface(kind):
    model = estimator(kind)
    tags = get_tags(model)
    assert tags.input_tags.positive_only
    assert tags.target_tags.required == (kind != "spa")
    assert not tags.target_tags.positive_only
    assert not tags.target_tags.one_d_labels
    assert not tags.target_tags.two_d_labels
    assert tags.transformer_tags is not None
    assert is_classifier(model) == (kind == "classifier")
    assert not is_regressor(model)
    if kind == "classifier":
        assert tags.estimator_type == "classifier"
        assert tags.classifier_tags.multi_class is False
    else:
        assert tags.estimator_type is None
        assert tags.classifier_tags is None
        assert not hasattr(model, "predict")
        assert not hasattr(model, "score")


@pytest.mark.parametrize("solver", ["spa", "cd", "mu"])
@pytest.mark.parametrize("head", ["knn", "logistic", "linear_svm"])
def test_classifier_constraints_apply_to_every_solver_and_head(solver, head):
    model = NMFClassifier(solver=solver, classifier=head)
    tags = get_tags(model)
    assert tags.input_tags.positive_only
    assert tags.target_tags.required
    assert tags.classifier_tags.multi_class is False
    assert tags.transformer_tags.preserves_dtype == ["float64"]


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
@pytest.mark.parametrize("fitted", [False, True])
def test_tag_objects_are_independent_and_do_not_change_estimator_state(binary_data, kind, fitted):
    X, y, _ = binary_data
    model = estimator(kind)
    if fitted:
        model.fit(X, y)
    before = pickle.dumps(model)
    first, second = get_tags(model), get_tags(model)
    expected = asdict(second)
    assert first is not second
    first.input_tags.positive_only = False
    first.target_tags.required = not first.target_tags.required
    first.transformer_tags.preserves_dtype.append("mutation_probe")
    if first.classifier_tags is not None:
        first.classifier_tags.multi_class = True
    assert asdict(second) == expected
    assert asdict(get_tags(model)) == expected
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
def test_clone_pickle_and_final_pipeline_preserve_identity(binary_data, kind):
    X, y, query = binary_data
    model = estimator(kind).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert asdict(get_tags(restored)) == asdict(get_tags(model))
    np.testing.assert_array_equal(restored.transform(query), model.transform(query))
    if kind == "classifier":
        np.testing.assert_array_equal(restored.predict(query), model.predict(query))
    unfitted = clone(restored)
    assert type(unfitted) is type(model)
    assert unfitted.get_params() == model.get_params()
    assert not any(name.endswith("_") for name in vars(unfitted))
    pipe = Pipeline([("model", unfitted)]).fit(X, y)
    assert is_classifier(pipe) == (kind == "classifier")
    assert (get_tags(pipe).classifier_tags is None) == (kind != "classifier")
    np.testing.assert_array_equal(pipe.transform(query), model.transform(query))
    splitter = check_cv(3, y, classifier=is_classifier(pipe))
    assert isinstance(splitter, StratifiedKFold if kind == "classifier" else KFold)


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
def test_negative_features_remain_rejected_and_zero_query_remains_valid(binary_data, kind):
    X, y, query = binary_data
    negative = X.copy()
    negative[0, 0] = -1.
    with pytest.raises(ValueError):
        estimator(kind).fit(negative, y)
    model = estimator(kind).fit(X, y)
    with pytest.raises(ValueError):
        model.transform(-np.ones_like(query))
    transformed = model.transform(np.zeros_like(query))
    np.testing.assert_array_equal(transformed, np.zeros_like(transformed))


@pytest.mark.parametrize("kind", ["classifier", "screen"])
@pytest.mark.parametrize("case", ["none", "single_class", "multiclass", "singleton_class"])
def test_supervised_estimators_keep_runtime_target_restrictions(binary_data, kind, case):
    X, y, _ = binary_data
    labels = {
        "none": None,
        "single_class": np.zeros_like(y),
        "multiclass": np.arange(len(y)) % 3,
        "singleton_class": np.r_[1, np.zeros(len(y)-1, dtype=int)],
    }[case]
    with pytest.raises(ValueError):
        estimator(kind).fit(X, labels)


@pytest.mark.parametrize("kind", ["classifier", "screen"])
def test_supervised_estimators_require_y_argument(binary_data, kind):
    X, _, _ = binary_data
    with pytest.raises(TypeError):
        estimator(kind).fit(X)


@pytest.mark.parametrize("kind", ["classifier", "screen"])
@pytest.mark.parametrize("label_kind", ["negative_numeric", "string"])
def test_nonnegative_input_tags_do_not_restrict_class_label_values(binary_data, kind, label_kind):
    X, y, _ = binary_data
    labels = np.where(y, -2, -7) if label_kind == "negative_numeric" else np.where(y, "case", "control")
    model = estimator(kind).fit(X, labels)
    np.testing.assert_array_equal(model.classes_, np.unique(labels))
    assert not get_tags(model).target_tags.positive_only


def test_spa_keeps_optional_ignored_labels(binary_data):
    X, _, query = binary_data
    unlabeled = estimator("spa").fit(X)
    multiclass_labels = np.arange(len(X)) % 3
    labeled = estimator("spa").fit(X, multiclass_labels)
    np.testing.assert_array_equal(unlabeled.components_, labeled.components_)
    np.testing.assert_array_equal(unlabeled.transform(query), labeled.transform(query))


@pytest.mark.parametrize("policy", ["absolute", "percentile"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_binary_checker_adapter_preserves_the_actual_screen(binary_data, policy, dtype):
    X, y, query = binary_data
    X, query = X.astype(dtype), query.astype(dtype)
    parameters = dict(screening=policy, percentile=75., score_threshold=2.)
    actual = BinaryScreening(**parameters).fit(X, y)
    adapted = _BinaryCheckerScreen(**parameters).fit(X, y)
    assert get_tags(actual).classifier_tags is None
    assert get_tags(adapted).classifier_tags.multi_class is False
    assert not is_classifier(actual) and not is_classifier(adapted)
    assert not hasattr(adapted, "predict") and not hasattr(adapted, "score")
    for attribute in ("support_", "selected_features_", "scores_", "t_scores_", "classes_"):
        np.testing.assert_array_equal(getattr(adapted, attribute), getattr(actual, attribute))
    np.testing.assert_array_equal(adapted.transform(query), actual.transform(query))
    assert adapted.transform(query).dtype == actual.transform(query).dtype == np.dtype(dtype)
    actual_pipe = Pipeline([("screen", clone(actual)), ("head", KNeighborsClassifier(n_neighbors=3))]).fit(X, y)
    adapted_pipe = Pipeline([("screen", clone(adapted)), ("head", KNeighborsClassifier(n_neighbors=3))]).fit(X, y)
    np.testing.assert_array_equal(adapted_pipe.predict(query), actual_pipe.predict(query))
    # Pipeline describes its final classifier; metadata does not aggregate
    # the binary restriction of an intermediate screening step.
    assert is_classifier(actual_pipe) and is_classifier(adapted_pipe)
    assert get_tags(actual_pipe).classifier_tags.multi_class
    assert get_tags(adapted_pipe).classifier_tags.multi_class


@pytest.mark.parametrize("check", [
    estimator_checks.check_dtype_object,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_methods_subset_invariance,
    estimator_checks.check_methods_sample_order_invariance,
    estimator_checks.check_readonly_memmap_input,
    estimator_checks.check_transformer_preserve_dtypes,
])
def test_applicable_generic_checks_with_binary_target_fixtures(check):
    # These checks exercise the inherited screening implementation using valid
    # binary targets. They do not establish full check_estimator conformance
    # for the public transformer or alter sklearn's checks.
    model = _BinaryCheckerScreen(percentile=100.)
    check(type(model).__name__, model)
    assert get_tags(BinaryScreening()).classifier_tags is None
