"""Evaluator shortcuts preserve supplied public prediction policies."""

from contextlib import contextmanager
from types import MethodType
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from supervised_nmf.evaluation import (
    _TrackedLinearSVC, _TrackedLogisticRegression, evaluate_classifier,
)


class CountTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.calls_ = 0
        return self

    def transform(self, X):
        self.calls_ += 1
        return X


class FitOnlyPipeline(Pipeline):
    def fit(self, X, y=None, **params):
        result = super().fit(X, y, **params)
        self.fit_record_ = "fit-only subclass"
        return result


def problem():
    X = np.array([[-4.], [-3.], [-2.], [-1.], [1.], [2.], [3.], [4.]])
    y = np.array(["absent"] * 4 + ["present"] * 4)
    query = np.array([[-3.8], [-.2], [.2], [3.8]])
    return X, y, query, y[[0, 4, 0, 4]]


BASE_TYPES = {"knn": KNeighborsClassifier, "logistic": LogisticRegression, "linear_svc": LinearSVC}


def head_for(family, cls=None):
    cls = BASE_TYPES[family] if cls is None else cls
    parameters = {"knn": {"n_neighbors": 3, "algorithm": "kd_tree"},
                  "logistic": {"max_iter": 2000, "random_state": 7},
                  "linear_svc": {"dual": False, "max_iter": 2000, "random_state": 7}}[family]
    return cls(**parameters)


def wrap(head, in_pipeline=True, pipeline_type=Pipeline):
    return pipeline_type([("coordinates", CountTransform()), ("head", head)]) if in_pipeline else head


def fit_pair(factory, X, y):
    # Public oracle calls must not advance the tested estimator's callable state.
    return factory().fit(X, y), factory().fit(X, y)


def public_result(model, query):
    labels = model.predict(query)
    if hasattr(model, "predict_proba"):
        return labels, model.predict_proba(query)[:, 1], "probability"
    if hasattr(model, "decision_function"):
        return labels, model.decision_function(query), "decision_margin"
    return labels, None, None


def capture_public(model, query):
    result = error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = public_result(model, query)
        except Exception as caught_error:
            error = (type(caught_error).__name__, str(caught_error))
    recorded = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    return result, error, recorded


def assert_public_report(report, reference, truth, expected):
    result, error, recorded = expected
    assert report["warnings"] == recorded
    if error is not None:
        assert report["status"] == "failed", report
        assert (report["error_type"], report["error"]) == error
        assert "predictions" not in report and "metrics" not in report
        return
    labels, scores, score_type = result
    status = "provisional_nonconvergence" if any(w["category"] == "ConvergenceWarning" for w in recorded) else "valid"
    assert report["status"] == status, report
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], labels)
    assert report["ranking_score_type"] == score_type
    assert report["metrics"]["confusion_matrix"] == confusion_matrix(truth, labels, labels=reference.classes_).tolist()
    field = "positive_class_probability" if score_type == "probability" else "positive_class_decision_margin"
    other = "positive_class_decision_margin" if score_type == "probability" else "positive_class_probability"
    np.testing.assert_array_equal(report["predictions"][field], scores)
    assert report["predictions"][other] is None
    positive = truth == reference.classes_[1]
    assert report["metrics"]["roc_auc"] == roc_auc_score(positive, scores)
    assert report["metrics"]["average_precision"] == average_precision_score(positive, scores)


def evaluate_against_public(reference, model, query, truth):
    expected = capture_public(reference, query)
    report = evaluate_classifier(model, query, truth)
    assert_public_report(report, reference, truth, expected)
    return report


def near_tie_data():
    # Preserve the finite seed-3881 fixture from test_weighted_knn_consistency.py.
    rng = np.random.default_rng(3881)
    y = np.arange(32) % 2
    for _ in range(2):
        weights = rng.uniform(.1, 1, 32)
        extra = weights[y == 0].sum() - weights[y == 1].sum()
        i = np.flatnonzero(y == (1 if extra > 0 else 0))[0]
        weights[i] += abs(extra)
    return (1. / weights)[:, None], y, np.zeros((1, 1))


@pytest.mark.parametrize("in_pipeline", [False, True], ids=["direct", "pipeline"])
def test_weighted_near_tie_preserves_native_labels_for_whole_batch(in_pipeline):
    X, y, first = near_tie_data()
    labels = np.array(["absent", "present"])
    query, truth = np.vstack((first, [[10.], [3.]])), labels[[0, 1, 0]]
    reference, model = fit_pair(lambda: wrap(KNeighborsClassifier(
        n_neighbors=32, weights="distance", algorithm="kd_tree"), in_pipeline), X, labels[y])
    evaluate_against_public(reference, model, query, truth)


def changed_method(base, method):
    original = getattr(base, method)

    def changed(self, X, **params):
        result = original(self, X, **params)
        if method == "predict":
            return np.where(result == self.classes_[0], self.classes_[1], self.classes_[0])
        return result[:, ::-1] if method == "predict_proba" else -.625 - result

    return changed


@pytest.mark.parametrize("family,in_pipeline", [("knn", False), ("knn", True), ("linear_svc", True)])
@pytest.mark.parametrize("customization", ["subclass", "instance"])
def test_custom_head_predict_is_authoritative(family, in_pipeline, customization):
    X, y, query, truth = problem()
    base, changed = BASE_TYPES[family], changed_method(BASE_TYPES[family], "predict")
    cls = type("CustomHead", (base,), {"predict": changed}) if customization == "subclass" else base
    reference, model = fit_pair(lambda: wrap(head_for(family, cls), in_pipeline), X, y)
    if customization == "instance":
        for estimator in (reference, model):
            head = estimator.steps[-1][1] if in_pipeline else estimator
            head.predict = MethodType(changed, head)
    evaluate_against_public(reference, model, query, truth)


OUTER_POLICIES = [("knn", "predict"), ("knn", "predict_proba"),
                  ("logistic", "predict"), ("logistic", "predict_proba"),
                  ("linear_svc", "predict"), ("linear_svc", "decision_function")]


@pytest.mark.parametrize("family,method", OUTER_POLICIES)
@pytest.mark.parametrize("customization", ["subclass", "instance"])
def test_outer_pipeline_public_policy_is_authoritative(family, method, customization):
    X, y, query, truth = problem()
    changed = changed_method(Pipeline, method)
    cls = type("CustomPipeline", (Pipeline,), {method: changed}) if customization == "subclass" else Pipeline
    reference, model = fit_pair(lambda: wrap(head_for(family), pipeline_type=cls), X, y)
    if customization == "instance":
        for estimator in (reference, model):
            setattr(estimator, method, MethodType(changed, estimator))
    model.named_steps["coordinates"].calls_ = 0
    evaluate_against_public(reference, model, query, truth)
    assert model.named_steps["coordinates"].calls_ == 2


@pytest.mark.parametrize("family,head_type,pipeline_type", [
    ("logistic", _TrackedLogisticRegression, Pipeline),
    ("linear_svc", _TrackedLinearSVC, Pipeline),
    ("knn", KNeighborsClassifier, FitOnlyPipeline),
    ("logistic", LogisticRegression, FitOnlyPipeline),
    ("linear_svc", LinearSVC, FitOnlyPipeline),
], ids=["tracked_logistic", "tracked_svc", "inherited_knn", "inherited_logistic", "inherited_svc"])
def test_fit_only_subclasses_keep_one_preprocessing_call(family, head_type, pipeline_type):
    X, y, query, truth = problem()
    reference, model = fit_pair(lambda: wrap(head_for(family, head_type), pipeline_type=pipeline_type), X, y)
    model.named_steps["coordinates"].calls_ = 0
    evaluate_against_public(reference, model, query, truth)
    assert model.named_steps["coordinates"].calls_ == 1


class ProbabilityLinearSVC(LinearSVC):
    def predict_proba(self, X):
        positive = 1. / (1. + np.exp(-self.decision_function(X)))
        return np.column_stack((1. - positive, positive))


def test_probability_capable_svc_pipeline_prefers_probabilities():
    X, y, query, truth = problem()
    reference, model = fit_pair(lambda: wrap(head_for("linear_svc", ProbabilityLinearSVC)), X, y)
    assert evaluate_against_public(reference, model, query, truth)["ranking_score_type"] == "probability"


class AlternatingWeights:
    def __init__(self, fail_second):
        self.fail_second = fail_second
        self.shapes = []

    def __call__(self, distances):
        self.shapes.append(distances.shape)
        call = len(self.shapes)
        warnings.warn(f"weights invocation {call}", UserWarning)
        if self.fail_second and call == 2:
            raise RuntimeError("weights failed on second public operation")
        result = np.ones_like(distances)
        result[:, 0 if call % 2 else -1] = 4.
        return result


@pytest.mark.parametrize("in_pipeline,fail_second", [(False, True), (True, True), (True, False)],
                         ids=["direct_error", "pipeline_error", "pipeline_values"])
def test_stateful_weights_keep_native_operation_order(in_pipeline, fail_second):
    X, y, query, truth = np.array([[0.], [1.], [2.]]), np.array([0, 1, 1]), np.array([[.1], [.2], [.3]]), np.array([0, 1, 0])
    reference, model = fit_pair(lambda: wrap(KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree",
        weights=AlternatingWeights(fail_second)), in_pipeline), X, y)
    evaluate_against_public(reference, model, query, truth)
    actual_head = model.steps[-1][1] if in_pipeline else model
    reference_head = reference.steps[-1][1] if in_pipeline else reference
    assert actual_head.weights.shapes == reference_head.weights.shapes == [(len(query), 3)] * 2


class TraceMetric:
    def __init__(self):
        self.calls = []

    def __call__(self, first, second):
        self.calls.append((tuple(first.tolist()), tuple(second.tolist())))
        return float(np.sum(np.abs(first - second)))


@pytest.mark.parametrize("in_pipeline", [False, True], ids=["direct", "pipeline"])
def test_pyfunc_metric_keeps_public_callback_sequence(in_pipeline):
    X, y, query, truth = problem()
    reference, model = fit_pair(lambda: wrap(KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree",
        leaf_size=2, metric="pyfunc", metric_params={"func": TraceMetric()}), in_pipeline), X, y)
    heads = [estimator.steps[-1][1] if in_pipeline else estimator for estimator in (reference, model)]
    for head in heads:
        head.metric_params["func"].calls.clear()
    evaluate_against_public(reference, model, query, truth)
    assert heads[0].metric_params["func"].calls
    assert heads[1].metric_params["func"].calls == heads[0].metric_params["func"].calls


class PropertyProbabilityKNN(KNeighborsClassifier):
    @property
    def predict_proba(self):
        if "events_" in vars(self):
            self.events_.append("probability_property")
            warnings.warn("public probability property accessed", UserWarning)

        def probability(X):
            self.events_.append("probability_call")
            return KNeighborsClassifier.predict_proba(self, X)

        return probability


class AccessLinearSVC(LinearSVC):
    def __getattribute__(self, name):
        state = object.__getattribute__(self, "__dict__")
        if "events_" in state and name in ("predict", "predict_proba", "decision_function"):
            state["events_"].append(name)
        return object.__getattribute__(self, name)


@pytest.mark.parametrize("family,cls", [("knn", PropertyProbabilityKNN), ("linear_svc", AccessLinearSVC)])
def test_dynamic_public_access_does_not_gain_qualification_side_effects(family, cls):
    X, y, query, truth = problem()
    reference, model = fit_pair(lambda: wrap(head_for(family, cls)), X, y)
    for estimator in (reference, model):
        estimator.named_steps["head"].events_ = []
    evaluate_against_public(reference, model, query, truth)
    assert model.named_steps["head"].events_ == reference.named_steps["head"].events_


@contextmanager
def numerical_policy(mode="warn", handler=None):
    previous_handler = np.geterrcall()
    try:
        np.seterrcall(handler)
        with np.errstate(over=mode, divide="raise" if mode == "raise" else "warn",
                         invalid="raise" if mode == "raise" else "warn", under="ignore"):
            yield
    finally:
        np.seterrcall(previous_handler)


def overflow_knn_pair(in_pipeline):
    rows = np.arange(8, dtype=np.float64)
    X, y = np.abs(rows[:, None] - rows[None, :]), np.array([0] * 4 + [1] * 4)
    reference, model = fit_pair(lambda: wrap(KNeighborsClassifier(n_neighbors=8, weights="distance",
        metric="precomputed", algorithm="brute"), in_pipeline), X, y)
    return reference, model, np.full((2, 8), 1e-308), np.array([0, 1])


@pytest.mark.parametrize("in_pipeline", [False, True], ids=["direct", "pipeline"])
def test_strict_weight_overflow_preserves_first_native_error(in_pipeline):
    reference, model, query, truth = overflow_knn_pair(in_pipeline)
    with numerical_policy("raise"):
        report = evaluate_against_public(reference, model, query, truth)
    assert report["error_type"] == "FloatingPointError"


def overflow_svc_pair(cls=LinearSVC):
    X = np.array([[-.004], [-.003], [-.002], [-.001], [.001], [.002], [.003], [.004]])
    y = np.array([0] * 4 + [1] * 4)
    reference, model = fit_pair(lambda: wrap(cls(C=1e8, dual=False, tol=1e-12, max_iter=20000, random_state=7)), X, y)
    magnitude = np.finfo(np.float64).max / 4.
    return reference, model, np.array([[-magnitude], [magnitude]]), np.array([0, 1])


class OverflowHandler:
    def __init__(self, raise_second):
        self.raise_second = raise_second
        self.events = []

    def record(self, event):
        self.events.append(event)
        if self.raise_second and len(self.events) == 2:
            raise RuntimeError("public numerical handler failed on second invocation")

    def __call__(self, error, flag):
        self.record((str(error), int(flag)))

    def write(self, message):
        self.record(str(message))


def assert_invalid_score_report(report, score_type):
    message = ("Nonfinite or invalid predicted probabilities." if score_type == "probability"
               else "Decision margins must be a finite one-dimensional binary score.")
    assert report["status"] == "failed", report
    assert (report["error_type"], report["error"]) == ("ValueError", message)
    assert "predictions" not in report and "metrics" not in report


@pytest.mark.parametrize("cls,mode,raise_second", [
    (LinearSVC, "call", True), (_TrackedLinearSVC, "call", True),
    (LinearSVC, "log", True), (_TrackedLinearSVC, "log", True),
    (LinearSVC, "call", False),
], ids=["stock_call_error", "tracked_call_error", "stock_log_error", "tracked_log_error", "record_calls"])
def test_svc_numerical_handler_effects_and_errors_are_preserved(cls, mode, raise_second):
    reference, model, query, truth = overflow_svc_pair(cls)
    expected_handler, actual_handler = OverflowHandler(raise_second), OverflowHandler(raise_second)
    with numerical_policy(mode, expected_handler):
        expected = capture_public(reference, query)
    with numerical_policy(mode, actual_handler):
        report = evaluate_classifier(model, query, truth)
    if raise_second:
        assert expected[1] == ("RuntimeError", "public numerical handler failed on second invocation")
        assert_public_report(report, reference, truth, expected)
    else:
        assert expected[1] is None and not np.isfinite(expected[0][1]).all()
        assert_invalid_score_report(report, "decision_margin")
        assert report["warnings"] == expected[2]
    assert actual_handler.events == expected_handler.events
    assert len(expected_handler.events) == 2


@pytest.mark.parametrize("case", ["knn_direct", "knn_pipeline", "svc_pipeline"])
def test_default_numeric_warnings_still_produce_failed_invalid_score_report(case):
    reference, model, query, truth = overflow_svc_pair() if case == "svc_pipeline" else overflow_knn_pair(case == "knn_pipeline")
    with numerical_policy():
        expected = capture_public(reference, query)
        report = evaluate_classifier(model, query, truth)
    assert expected[1] is None and not np.isfinite(expected[0][1]).all()
    assert_invalid_score_report(report, expected[0][2])
    assert report["warnings"]
    assert all(w["category"] == "RuntimeWarning" for w in report["warnings"])
    assert any("overflow" in w["message"] for w in report["warnings"])
    # Default warning order/count can reflect intentionally eliminated work.
    # User numerical callbacks and first exceptions have separate strict tests.


def emitting_method(base, method, category, fail):
    original = getattr(base, method)

    def emit(self, X, **params):
        warnings.warn(f"public {method} warning", category)
        if fail:
            raise RuntimeError(f"public {method} failure")
        return original(self, X, **params)

    return emit


@pytest.mark.parametrize("family,method,scope,category,fail", [
    ("knn", "predict", "head", UserWarning, True),
    ("logistic", "predict_proba", "outer", UserWarning, False),
    ("linear_svc", "decision_function", "outer", ConvergenceWarning, False),
])
def test_custom_prediction_warnings_errors_and_convergence_status(family, method, scope, category, fail):
    X, y, query, truth = problem()
    base = BASE_TYPES[family] if scope == "head" else Pipeline
    cls = type("EmittingEstimator", (base,), {method: emitting_method(base, method, category, fail)})
    factory = (lambda: wrap(head_for(family, cls))) if scope == "head" else (lambda: wrap(head_for(family), pipeline_type=cls))
    reference, model = fit_pair(factory, X, y)
    evaluate_against_public(reference, model, query, truth)
