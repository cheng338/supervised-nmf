"""Validation messages and error precedence. Author: Chang Cheng."""

import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils import ClassifierTags, estimator_checks

from supervised_nmf import BinaryScreening, FeatureSPA, NMFClassifier


class NamedClassifier(NMFClassifier):
    pass


class NamedSPA(FeatureSPA):
    pass


class NamedScreening(BinaryScreening):
    pass


class _BinaryScreeningForChecks(BinaryScreening):
    """Provide binary checker fixtures without changing public metadata."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags(multi_class=False)
        return tags


@pytest.fixture
def data():
    rng = np.random.default_rng(712)
    y = np.arange(36) % 2
    X = rng.uniform(.2, 1.5, (len(y), 8))
    X[:, :3] += y[:, None]
    return X, y


def estimator(kind):
    if kind.startswith("classifier") or kind == "named_classifier":
        cls = NamedClassifier if kind == "named_classifier" else NMFClassifier
        head = kind.removeprefix("classifier_") if kind.startswith("classifier_") else "knn"
        return cls(n_components=2, screening="none", n_neighbors=3,
                   classifier=head, random_state=17, max_iter=1500)
    if kind in ("spa", "named_spa"):
        cls = NamedSPA if kind == "named_spa" else FeatureSPA
        return cls(n_components=2, nnls_max_iter=1500)
    cls = NamedScreening if kind == "named_screen" else BinaryScreening
    return cls(percentile=100.)


@pytest.mark.parametrize("kind", [
    "classifier_knn", "classifier_logistic", "classifier_linear_svm",
    "spa", "screen", "named_classifier", "named_spa", "named_screen",
])
@pytest.mark.parametrize("width", [1, 9])
def test_feature_count_message_names_the_estimator_on_every_query(data, kind, width):
    X, y = data
    model = estimator(kind).fit(X, y)
    query = np.ones((4, width))
    expected = (
        f"X has {width} features, but {type(model).__name__} is expecting "
        "8 features as input."
    )
    before = pickle.dumps(model)
    for method in ("transform", "predict", "predict_proba", "decision_function",
                   "predict_with_probabilities", "predict_with_decision_function", "score"):
        if not hasattr(model, method):
            continue
        arguments = (query, y[:4]) if method == "score" else (query,)
        with pytest.raises(ValueError) as error:
            getattr(model, method)(*arguments)
        assert str(error.value) == expected
        assert pickle.dumps(model) == before


@pytest.mark.parametrize("kind", ["classifier", "screen"])
@pytest.mark.parametrize("case", ["one_observation", "one_class", "three_classes"])
def test_binary_target_error_reports_the_observed_class_count(data, kind, case):
    X, y = data
    if case == "one_observation":
        X, y = X[:1], y[:1]
    elif case == "one_class":
        y = np.zeros_like(y)
    else:
        y = np.arange(len(y)) % 3
    observed = "3 classes" if case == "three_classes" else "1 class"
    with pytest.raises(ValueError) as error:
        estimator(kind).fit(X, y)
    assert str(error.value) == (
        "Only binary classification is supported. "
        "Binary classification requires exactly two classes; got " + observed + "."
    )


@pytest.mark.parametrize("kind", ["classifier", "screen"])
def test_two_classes_with_a_singleton_keep_the_sample_count_error(data, kind):
    X, y = data
    y = np.r_[1, np.zeros(len(y) - 1, dtype=int)]
    with pytest.raises(ValueError) as error:
        estimator(kind).fit(X, y)
    assert str(error.value) == "At least two training observations per class are required."


@pytest.mark.parametrize("kind", ["classifier", "screen"])
@pytest.mark.parametrize("labels", ["binary", "multiclass", "continuous"])
def test_negative_features_remain_the_first_domain_error(data, kind, labels):
    X, y = data
    negative = X.copy()
    negative[0, 0] = -1.
    if labels == "multiclass":
        y = np.arange(len(y)) % 3
    elif labels == "continuous":
        y = np.linspace(.1, 2.1, len(y))
    with pytest.raises(ValueError) as error:
        estimator(kind).fit(negative, y)
    assert str(error.value) == "Negative values in data. X must contain nonnegative values."


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
def test_negative_query_precedes_the_feature_count_error(data, kind):
    X, y = data
    model = estimator(kind).fit(X, y)
    before = pickle.dumps(model)
    with pytest.raises(ValueError) as error:
        model.transform(-np.ones((3, 7)))
    assert str(error.value) == (
        "Negative values in data. X must contain nonnegative values; "
        "preprocessing must be fitted on training data."
    )
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
def test_nonfinite_query_is_checked_before_feature_count(data, kind):
    X, y = data
    model = estimator(kind).fit(X, y)
    with pytest.raises(ValueError, match="NaN") as error:
        model.transform(np.full((3, 7), np.nan))
    assert "features as input" not in str(error.value)


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
def test_unfitted_state_is_checked_before_invalid_input(kind):
    with pytest.raises(NotFittedError):
        estimator(kind).transform(-np.ones((3, 1)))


@pytest.mark.parametrize("kind", ["classifier", "screen"])
def test_input_length_mismatch_precedes_domain_errors(data, kind):
    X, y = data
    with pytest.raises(ValueError, match="inconsistent numbers of samples") as error:
        estimator(kind).fit(-X, (np.arange(len(y)) % 3)[:-1])
    assert "Negative values in data" not in str(error.value)
    assert "Only binary classification" not in str(error.value)


def test_spa_inverse_transform_retains_nonnegative_input_explanation(data):
    X, _ = data
    model = estimator("spa").fit(X)
    with pytest.raises(ValueError) as error:
        model.inverse_transform(-np.ones((3, 3)))
    assert str(error.value) == (
        "Negative values in data. X must contain nonnegative values; "
        "preprocessing must be fitted on training data."
    )


@pytest.mark.parametrize("kind", ["classifier", "spa", "screen"])
@pytest.mark.parametrize("check", [
    estimator_checks.check_n_features_in_after_fitting,
    estimator_checks.check_positive_only_tag_during_fit,
    estimator_checks.check_fit2d_1sample,
])
def test_affected_generic_checks_with_applicable_target_fixtures(kind, check):
    # Binary screening requires binary fixtures. Multiclass rejection itself is
    # tested above using nonnegative X, independently of generic checker data.
    model = _BinaryScreeningForChecks(percentile=100.) if kind == "screen" else estimator(kind)
    check(type(model).__name__, model)
