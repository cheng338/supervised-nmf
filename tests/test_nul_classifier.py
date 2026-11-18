"""Terminal-NUL label boundaries and ordinary classifier compatibility."""

from copy import deepcopy
import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.decomposition import NMF
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from supervised_nmf import NMFClassifier
from supervised_nmf.decomposition import FeatureSPA


MESSAGE = "Class labels ending in a NUL codepoint"
HEADS = ("knn", "logistic", "linear_svm")
SOLVERS = ("spa", "cd", "mu")
SCREENS = ("absolute", "percentile", "none")


@pytest.fixture
def data():
    rng = np.random.default_rng(481)
    y = np.tile(np.array([0, 1]), 12)
    W = rng.uniform(.1, 1., (24, 2))
    W[y == 0, 0] += 2
    W[y == 1, 1] += 2
    H = np.column_stack((np.eye(2), rng.uniform(.2, .8, (2, 6))))
    return W @ H, y


def classifier(**options):
    defaults = dict(n_components=2, screening="none", n_neighbors=3,
                    score_threshold=0., percentile=100., random_state=17,
                    max_iter=1000, logistic_max_iter=2000, svm_max_iter=10000)
    return NMFClassifier(**(defaults | options))


def target(codes, representation):
    values = ["a\0" if code == 0 else "b" for code in codes]
    if representation == "list":
        return values
    if representation == "tuple":
        return tuple(values)
    if representation == "column":
        return [[value] for value in values]
    if representation == "numpy_scalar":
        return np.array([np.str_(value) for value in values], dtype=object)
    if representation == "string_dtype":
        if not hasattr(np.dtypes, "StringDType"):
            pytest.skip("StringDType is unavailable in this supported NumPy version")
        return np.array(values, dtype=np.dtypes.StringDType())
    return np.array(values, dtype=object)


def head(model):
    value = model.classifier_
    return value.steps[-1][1] if isinstance(value, Pipeline) else value


@pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="requires NumPy StringDType")
def test_variable_width_column_labels_keep_class_dtype_and_score_on_refit(data):
    X, codes = data
    labels = np.array(["", "prés\0ent"], dtype=object)[codes]
    strings = labels.astype(np.dtypes.StringDType())
    model = classifier(screening="percentile").fit(X, labels)
    expected_predictions = model.predict(X)
    weights = np.linspace(.2, 2, len(codes))
    expected_score = model.score(X, labels, sample_weight=weights)

    with pytest.warns(DataConversionWarning):
        model.fit(X, strings.reshape(-1, 1))
    assert model.classes_.dtype == strings.dtype
    assert model.classes_.tolist() == ["", "prés\0ent"]
    assert model.predict(X).tolist() == expected_predictions.tolist()
    assert model.score(X, strings, sample_weight=weights) == expected_score


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("screening", SCREENS)
@pytest.mark.parametrize("head_name", HEADS)
def test_rejects_before_factorization_for_all_solver_head_screen_combinations(
        data, monkeypatch, solver, screening, head_name):
    X, codes = data

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported labels reached factorization")

    monkeypatch.setattr(FeatureSPA, "fit", unexpected)
    monkeypatch.setattr(NMF, "fit_transform", unexpected)
    model = classifier(solver=solver, screening=screening, classifier=head_name)
    with pytest.raises(ValueError, match=MESSAGE):
        model.fit(X, target(codes, "object"))
    assert not any(name.endswith("_") for name in vars(model))


@pytest.mark.parametrize("representation", ("list", "tuple", "column", "object", "numpy_scalar", "string_dtype"))
@pytest.mark.parametrize("method", ("fit", "fit_transform"))
def test_raw_and_preserving_representations_are_rejected(data, representation, method):
    X, codes = data
    with pytest.raises(ValueError, match=MESSAGE):
        getattr(classifier(), method)(X, target(codes, representation))


@pytest.mark.parametrize("positive", ("a\0", np.str_("a\0")))
@pytest.mark.parametrize("head_name", HEADS)
def test_explicit_positive_label_checked_before_membership_or_factorization(data, monkeypatch, positive, head_name):
    X, codes = data
    labels = np.array(["a" if code == 0 else "b" for code in codes])

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported positive_label reached factorization")

    monkeypatch.setattr(FeatureSPA, "fit", unexpected)
    with pytest.raises(ValueError, match=MESSAGE):
        classifier(classifier=head_name, positive_label=positive).fit(X, labels)


@pytest.mark.parametrize("failure", ("target", "positive"))
def test_failed_refit_clears_state_and_allows_clean_refit(data, failure):
    X, codes = data
    model = classifier().fit(X, codes)
    y = codes
    if failure == "target":
        y = target(codes, "list")
    else:
        model.set_params(positive_label=np.str_("a\0"))
    with pytest.raises(ValueError, match=MESSAGE):
        model.fit(X, y)
    assert not any(name.endswith("_") for name in vars(model))
    with pytest.raises(NotFittedError):
        model.predict(X)
    model.set_params(positive_label=None).fit(X, codes)
    np.testing.assert_array_equal(model.predict(X), codes)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("head_name", HEADS)
@pytest.mark.parametrize("labels", (("a", "b"), ("a\0x", "b\0y"), ("\0a", "\0b")))
def test_ordinary_embedded_and_leading_labels_keep_numerics_and_head_types(data, solver, head_name, labels):
    X, codes = data
    y = np.array([labels[code] for code in codes], dtype=object)
    model = classifier(solver=solver, classifier=head_name, screening="percentile").fit(X, y)
    reference = classifier(solver=solver, classifier=head_name, screening="percentile").fit(X, codes)
    expected_type = {"knn": KNeighborsClassifier, "logistic": LogisticRegression, "linear_svm": LinearSVC}[head_name]
    assert type(head(model)) is expected_type
    assert model.classes_.dtype == y.dtype
    assert model.classes_.tolist() == list(labels)
    np.testing.assert_array_equal(model.components_, reference.components_)
    np.testing.assert_array_equal(model.training_coordinates_, reference.training_coordinates_)
    lookup = {value: code for code, value in enumerate(labels)}
    np.testing.assert_array_equal([lookup[value] for value in model.predict(X)], reference.predict(X))
    if hasattr(model, "predict_proba"):
        np.testing.assert_array_equal(model.predict_proba(X), reference.predict_proba(X))
    if hasattr(model, "decision_function"):
        np.testing.assert_array_equal(model.decision_function(X), reference.decision_function(X))


@pytest.mark.parametrize("head_name", HEADS)
@pytest.mark.parametrize("field", ("classes", "classes_list", "head_classes", "positive"))
@pytest.mark.parametrize("transport", ("memory_numpy_scalar", "pickle_python_string"))
def test_legacy_state_rejected_on_every_available_consumer_without_mutation(data, head_name, field, transport):
    X, codes = data
    model = classifier(classifier=head_name).fit(X, codes)
    hazardous = np.str_("a\0") if transport == "memory_numpy_scalar" else "a\0"
    if field == "classes":
        model.classes_ = np.array([hazardous, "b"], dtype=object)
    elif field == "classes_list":
        model.classes_ = [hazardous, "b"]
    elif field == "head_classes":
        head(model).classes_ = np.array([hazardous, "b"], dtype=object)
    else:
        model.positive_label_ = hazardous
    if transport == "pickle_python_string":
        # Builtin strings survive pickling exactly; NumPy scalar serialization
        # may already erase terminal NUL before the package receives the state.
        model = pickle.loads(pickle.dumps(model))
    before = pickle.dumps(model)
    methods = ("transform", "predict", "predict_proba", "decision_function",
               "predict_with_probabilities", "predict_with_decision_function")
    for name in methods:
        if hasattr(model, name):
            with pytest.raises(ValueError, match=MESSAGE):
                getattr(model, name)(X)
    for indices in (False, True):
        with pytest.raises(ValueError, match=MESSAGE):
            model.get_support(indices=indices)
    with pytest.raises(ValueError, match=MESSAGE):
        model.score(X, codes)
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("head_name", HEADS)
def test_custom_transform_still_delegates_once_and_cannot_bypass_prediction_guard(data, head_name):
    X, codes = data
    model = classifier(classifier=head_name).fit(X, codes)
    coordinates = model.transform(X)
    calls = []

    def custom(query):
        calls.append(query)
        return coordinates

    model.transform = custom
    model.positive_label_ = "bad\0"
    for name in ("predict", "predict_proba", "decision_function",
                 "predict_with_probabilities", "predict_with_decision_function"):
        if hasattr(model, name):
            before = len(calls)
            with pytest.raises(ValueError, match=MESSAGE):
                getattr(model, name)(X)
            assert len(calls) == before + 1
            assert calls[-1] is X


def test_queued_constructor_policies_do_not_affect_fitted_predictions(data):
    X, codes = data
    model = classifier().fit(X, codes)
    expected = model.predict(X)
    model.set_params(positive_label="queued\0", logistic_class_weight={"unused\0": -1},
                     svm_class_weight={"unused\0": -1})
    np.testing.assert_array_equal(model.predict(X), expected)
    np.testing.assert_array_equal(model.get_support(), np.ones(X.shape[1], dtype=bool))
    assert model.score(X, codes) == ClassifierMixin.score(model, X, codes)
    # Inactive head options remain ignored during an ordinary KNN refit as well.
    model.set_params(positive_label=None).fit(X, codes)


@pytest.mark.parametrize("head_name,prefix", (("logistic", "logistic"), ("linear_svm", "svm")))
def test_unused_weight_dictionary_keys_keep_native_semantics(data, head_name, prefix):
    X, codes = data
    ordinary = {0: .7, 1: 1.5}
    weights = ordinary | {np.str_("unused\0"): 3.}
    model = classifier(classifier=head_name, **{prefix + "_class_weight": weights}).fit(X, codes)
    reference = classifier(classifier=head_name, **{prefix + "_class_weight": ordinary}).fit(X, codes)
    assert head(model).class_weight is weights
    np.testing.assert_array_equal(head(model).coef_, head(reference).coef_)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))


@pytest.mark.parametrize("representation", ("list", "tuple", "column", "object", "numpy_scalar", "string_dtype"))
def test_score_rejects_separate_unsupported_targets(data, representation):
    X, codes = data
    model = classifier().fit(X, codes)
    with pytest.raises(ValueError, match=MESSAGE):
        model.score(X, target(codes, representation))


@pytest.mark.parametrize("head_name", HEADS)
@pytest.mark.parametrize("weighted", (False, True))
def test_score_keeps_accuracy_sample_weights_and_unseen_label_behavior(data, head_name, weighted):
    X, codes = data
    labels = np.array(["a\0x" if code == 0 else "b" for code in codes], dtype=object)
    model = classifier(classifier=head_name).fit(X, labels)
    truth = labels.copy()
    truth[:5] = "unseen"
    sample_weight = np.linspace(.2, 2, len(codes)) if weighted else None
    expected = ClassifierMixin.score(model, X, truth, sample_weight=sample_weight)
    assert model.score(X, truth, sample_weight=sample_weight) == expected
    before = deepcopy(vars(model))
    model.score(X, truth.reshape(-1, 1), sample_weight=sample_weight)
    assert vars(model).keys() == before.keys()


def test_score_and_transform_keep_structural_error_precedence(data):
    X, codes = data
    model = classifier().fit(X, codes)
    hazardous = target(codes, "object")
    with pytest.raises(ValueError, match="expecting"):
        model.score(X[:, :-1], hazardous)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        model.score(X, hazardous[:-1])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        model.score(X, hazardous, sample_weight=np.ones(len(codes) - 1))
    model.positive_label_ = "bad\0"
    with pytest.raises(ValueError, match="expecting"):
        model.transform(X[:, :-1])
    with pytest.raises(ValueError, match="Negative values"):
        model.transform(-X)


def test_missing_policy_fields_do_not_consult_pending_constructor_value(data):
    X, codes = data
    model = classifier().fit(X, codes)
    for name in ("decision_threshold_", "positive_label_", "positive_class_index_"):
        delattr(model, name)
    model.set_params(positive_label="queued\0")
    restored = pickle.loads(pickle.dumps(model))
    with pytest.raises(AttributeError, match="decision_threshold_"):
        restored.predict(X)
    assert restored.positive_label == "queued\0"
    assert all(name not in vars(restored) for name in (
        "decision_threshold_", "positive_label_", "positive_class_index_"))


@pytest.mark.parametrize("kind", ("object", "string", "category", "column"))
def test_score_pandas_target_containers_keep_sklearn_values_and_warnings(data, kind):
    pd = pytest.importorskip("pandas")
    X, codes = data
    labels = np.array(["a\0x" if code == 0 else "b" for code in codes], dtype=object)
    model = classifier().fit(X, labels)
    truth = pd.DataFrame({"label": labels}) if kind == "column" else pd.Series(labels, dtype=kind)
    weights = np.linspace(.2, 2, len(codes))
    with warnings.catch_warnings(record=True) as baseline_warnings:
        warnings.simplefilter("always")
        expected = ClassifierMixin.score(model, X, truth, sample_weight=weights)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = model.score(X, truth, sample_weight=weights)
    assert actual == expected
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in baseline_warnings]


def test_score_does_not_repeat_custom_target_array_conversion(data):
    X, codes = data
    model = classifier().fit(X, codes)
    calls = []

    class Target:
        def __len__(self):
            return len(codes)

        def __array__(self, dtype=None, copy=None):
            calls.append(dtype)
            array = np.asarray(codes, dtype=dtype)
            return array.copy() if copy else array

    assert model.score(X, Target()) == model.score(X, codes)
    assert len(calls) == 1
