"""Named workflow inputs keep native column checks and positional callback contracts."""

import pickle
from types import MethodType

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from supervised_nmf.evaluation import calibration_search, data_fingerprint, evaluate_classifier, run_comparison

pd = pytest.importorskip("pandas")


@pytest.fixture
def binary_table():
    y = np.tile([0, 1], 16)
    frame = pd.DataFrame({"left": 1 - y, "right": y}, index=np.arange(101, 133))
    return frame, y


class ArrayClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        assert type(X) is np.ndarray
        self.classes_ = np.unique(y)
        self.right_wins_ = y[np.argmax(X[:, 1])] == 1
        return self

    def predict(self, X):
        assert type(X) is np.ndarray
        return ((X[:, 1] > X[:, 0]) == self.right_wins_).astype(int)


class InPlacePredictClassifier(ArrayClassifier):
    def predict(self, X):
        labels = super().predict(X)
        X[:, 1] = 0
        return labels

    def predict_proba(self, X):
        positive = super().predict(X)
        return np.column_stack((1 - positive, positive))


class ExternalTransform(TransformerMixin, BaseEstimator):
    def __init__(self, rename=False):
        self.rename = rename

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # The fitted input metadata belongs to the downstream estimator.
        assert X.index[0] == 101 and X.dtypes.iloc[0].kind in "iu"
        return X.rename(columns={"left": "a", "right": "b"}) if self.rename else X


class InPlaceTransform(TransformerMixin, BaseEstimator):
    def __init__(self, fail=False):
        self.fail = fail

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["signal"] = X.pop("right") * 2
        X.iloc[:, 0] *= 3
        X.index = X.index + 1
        if self.fail:
            raise ValueError("failed after changing the input")
        return X


class StableClassesClassifier(ArrayClassifier):
    def fit(self, X, y):
        previous = getattr(self, "classes_", None)
        super().fit(X, y)
        if previous is not None:
            previous[:] = self.classes_
            self.classes_ = previous
        return self


class GeneratedNamesClassifier(ArrayClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)
        return self


def test_positional_refit_uses_training_names_when_classifier_generates_names(binary_table, tmp_path):
    frame, y = binary_table
    result = calibration_search(GeneratedNamesClassifier(), {}, frame, y, cv=2)
    assert result.report["refit_input_container"] == "positional_array"
    np.testing.assert_array_equal(result.estimator.feature_names_in_, ["x0", "x1"])
    labels = evaluate_classifier(result, frame, y)["predictions"]["predicted_labels"]
    np.testing.assert_array_equal(labels, y)
    for wrong in (frame.iloc[:, ::-1], frame.set_axis(["x0", "x1"], axis="columns")):
        with pytest.raises(ValueError, match="[Ff]eature names"):
            evaluate_classifier(result, wrong, y)

    comparison = run_comparison({"generated": (GeneratedNamesClassifier(), {})},
                                frame, y, frame, y, output_dir=tmp_path)
    evaluation = comparison["models"]["generated"]["evaluation"]
    np.testing.assert_array_equal(evaluation["predictions"]["predicted_labels"], y)


@pytest.mark.parametrize("reverse_training", [False, True])
def test_positional_refit_guards_named_evaluation_and_pickle(binary_table, reverse_training):
    frame, y = binary_table
    if reverse_training:
        frame = frame.iloc[:, ::-1]
    wrong = frame.iloc[:, ::-1]
    result = calibration_search(StableClassesClassifier(), {}, frame, y, cv=2)
    assert result.report["refit_input_container"] == "positional_array"
    assert result.report["refit_evaluation_schema"] == "calibration"
    for fitted in (result, pickle.loads(pickle.dumps(result))):
        model = fitted.estimator
        assert type(model) is StableClassesClassifier and not hasattr(model, "feature_names_in_")
        np.testing.assert_array_equal(evaluate_classifier(fitted, frame, y)["predictions"]["predicted_labels"], y)
        with pytest.raises(ValueError, match="[Ff]eature names"):
            evaluate_classifier(fitted, wrong, y)
        with pytest.raises(ValueError, match="[Ff]eature names"):
            evaluate_classifier(model, wrong, y)
        assert not np.array_equal(model.predict(wrong.to_numpy()), y)
        classes = model.classes_
        model.fit(wrong.to_numpy(), y)
        assert model.classes_ is classes
        with pytest.raises(ValueError, match="CalibrationResult"):
            evaluate_classifier(model, frame, y)
        np.testing.assert_array_equal(evaluate_classifier(model, wrong.to_numpy(), y)["predictions"]["predicted_labels"], y)


def test_positional_calibration_preserves_trailing_nul_in_column_name(binary_table, tmp_path):
    frame, y = binary_table
    named = frame.set_axis(["left\0", "right"], axis="columns")
    result = calibration_search(ArrayClassifier(), {}, named, y, cv=2)
    assert result.report["refit_input_container"] == "positional_array"
    np.testing.assert_array_equal(evaluate_classifier(result, named, y)["predictions"]["predicted_labels"], y)
    with pytest.raises(ValueError, match="Feature names"):
        evaluate_classifier(result, frame, y)

    models = {"array": (ArrayClassifier(), {})}
    report = run_comparison(models, named, y, named, y, output_dir=tmp_path)
    record = report["models"]["array"]
    assert record["refit_evaluation_schema"] == "calibration"
    np.testing.assert_array_equal(record["evaluation"]["predictions"]["predicted_labels"], y)
    with pytest.raises(ValueError, match="Feature names"):
        run_comparison(models, named, y, frame, y, output_dir=tmp_path)


def test_array_calibration_result_evaluates_named_holdout_by_position(binary_table):
    frame, y = binary_table
    train = frame.iloc[:20].to_numpy()
    validation = frame.iloc[20:]
    model = RandomForestClassifier(n_estimators=5, random_state=11, n_jobs=1)
    result = calibration_search(model, {}, train, y[:20], cv=2)
    assert not result.report["is_fallback"]
    for fitted in (result, pickle.loads(pickle.dumps(result))):
        native = fitted.estimator.predict(validation.to_numpy())
        report = evaluate_classifier(fitted, validation, y[20:])
        assert report["status"] == "valid"
        np.testing.assert_array_equal(report["predictions"]["predicted_labels"], native)
    with pytest.raises(ValueError, match="CalibrationResult"):
        evaluate_classifier(result.estimator, validation, y[20:])


@pytest.mark.parametrize("named_training", [False, True])
def test_positional_result_isolates_named_prediction_calls(binary_table, named_training):
    frame, y = binary_table
    train = frame.iloc[:20]
    result = calibration_search(InPlacePredictClassifier(), {}, train if named_training else train.to_numpy(),
                                y[:20], cv=2)
    assert not result.report["is_fallback"]
    # Mixed numeric dtypes produce a writable array during validation even
    # when pandas protects a single-block frame with copy-on-write.
    validation = frame.iloc[20:].astype({"right": "float64"})
    original = validation.copy()
    digest = data_fingerprint(validation, y[20:])
    model = result.estimator
    labels = model.predict(validation.to_numpy(copy=True))
    probabilities = model.predict_proba(validation.to_numpy(copy=True))[:, 1]

    report = evaluate_classifier(result, validation, y[20:])
    assert report["status"] == "valid"
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], labels)
    np.testing.assert_array_equal(report["predictions"]["positive_class_probability"], probabilities)
    assert report["data_sha256"] == digest
    pd.testing.assert_frame_equal(validation, original)


@pytest.mark.parametrize("serial_final", [False, True])
def test_array_comparison_evaluates_named_holdout_by_position(binary_table, tmp_path, serial_final):
    frame, y = binary_table
    train = frame.iloc[:20].to_numpy()
    validation = frame.iloc[20:]
    native = ArrayClassifier().fit(train, y[:20]).predict(validation.to_numpy())
    result = run_comparison({"array": (ArrayClassifier(), {})}, train, y[:20], validation, y[20:],
                            output_dir=tmp_path, serial_final=serial_final)
    record = result["models"]["array"]
    assert not record["is_fallback"]
    assert record["evaluation"]["status"] == "valid"
    np.testing.assert_array_equal(record["evaluation"]["predictions"]["predicted_labels"], native)


@pytest.mark.parametrize("processor", [
    MinMaxScaler(),
    ColumnTransformer([("byname", "passthrough", ["left", "right"])]),
    ColumnTransformer([("impute", SimpleImputer(), ["left", "right"])]),
])
def test_stock_input_consumer_with_unlisted_head(binary_table, processor):
    frame, y = binary_table
    template = make_pipeline(processor, RandomForestClassifier(n_estimators=5, random_state=11, n_jobs=1))
    refit = calibration_search(template, {}, frame, y, cv=2)
    assert not refit.report["is_fallback"]
    assert refit.report["refit_input_container"] == "named_numeric_frame"
    assert refit.report["refit_evaluation_schema"] == "native"
    external = clone(template).fit(frame, y)
    for model in (external, refit.estimator):
        np.testing.assert_array_equal(evaluate_classifier(model, frame, y)["predictions"]["predicted_labels"], y)
        with pytest.raises(ValueError, match="[Ff]eature names"):
            evaluate_classifier(model, frame.iloc[:, ::-1], y)


@pytest.mark.parametrize("rename", [False, True])
def test_external_pipeline_preserves_table_until_named_stage(binary_table, rename):
    frame, y = binary_table
    model = make_pipeline(ExternalTransform(rename=rename), LogisticRegression()).fit(frame, y)
    np.testing.assert_array_equal(evaluate_classifier(model, frame, y)["predictions"]["predicted_labels"], y)
    wrong = frame.iloc[:, ::-1]
    with pytest.raises(ValueError, match="feature names"):
        model.predict(wrong)
    report = evaluate_classifier(model, wrong, y)
    assert report["status"] == "failed" and "feature names" in report["error"]


def test_external_column_selector_with_a_custom_branch(binary_table):
    frame, y = binary_table
    processor = ColumnTransformer([("byname", ExternalTransform(), ["left", "right"])])
    model = make_pipeline(processor, LogisticRegression()).fit(frame, y)
    assert not hasattr(model.steps[-1][1], "feature_names_in_")
    report = evaluate_classifier(model, frame, y)
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], model.predict(frame))
    with pytest.raises(ValueError, match="[Ff]eature names"):
        evaluate_classifier(model, frame.iloc[:, ::-1], y)


@pytest.mark.parametrize(("head", "operation"), [
    (LogisticRegression(), "joint_label_and_probability_prediction_one_projection"),
    (RandomForestClassifier(n_estimators=5, random_state=11, n_jobs=1), "predict_plus_predict_proba_calls"),
])
def test_external_inplace_transform_has_private_evaluation_inputs(binary_table, head, operation):
    frame, y = binary_table
    original = frame.copy()
    model = make_pipeline(InPlaceTransform(), head).fit(frame.copy(), y)
    labels = model.predict(frame.copy())
    probabilities = model.predict_proba(frame.copy())[:, 1]
    digest = data_fingerprint(frame, y)
    report = evaluate_classifier(model, frame, y)
    assert report["prediction_operation"] == operation and report["data_sha256"] == digest
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], labels)
    np.testing.assert_allclose(report["predictions"]["positive_class_probability"], probabilities)
    pd.testing.assert_frame_equal(frame, original)

    model.steps[0][1].set_params(fail=True)
    report = evaluate_classifier(model, frame, y)
    assert report["status"] == "failed" and "failed after changing" in report["error"]
    assert report["data_sha256"] == digest
    pd.testing.assert_frame_equal(frame, original)


def _predict_positional(self, X):
    assert type(X) is np.ndarray
    return (X[:, 1] > X[:, 0]).astype(int)


def _transform_positional(self, X):
    assert type(X) is np.ndarray
    return StandardScaler.transform(self, X)


@pytest.mark.parametrize("override", ["direct", "outer", "transform"])
def test_instance_overrides_keep_numeric_input(binary_table, override):
    frame, y = binary_table
    if override == "direct":
        model = LogisticRegression().fit(frame, y)
        model.predict = MethodType(_predict_positional, model)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(frame, y)
        if override == "outer":
            model.predict = MethodType(_predict_positional, model)
        else:
            scaler = model.steps[0][1]
            scaler.transform = MethodType(_transform_positional, scaler)
    np.testing.assert_array_equal(evaluate_classifier(model, frame, y)["predictions"]["predicted_labels"], y)
    with pytest.raises(ValueError, match="[Ff]eature names"):
        evaluate_classifier(model, frame.iloc[:, ::-1], y)


class IndexClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, record_names=False):
        self.record_names = record_names

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.record_names:
            self.feature_names_in_ = X.columns.to_numpy(copy=True)
        return self

    def predict(self, X):
        labels = X.index.to_numpy() % 2
        X.index = X.index + 1
        return labels


@pytest.mark.parametrize("scaled", [False, True])
def test_external_named_consumer_preserves_row_index(binary_table, scaled):
    frame, y = binary_table
    original = frame.copy()
    model = (make_pipeline(StandardScaler().set_output(transform="pandas"), IndexClassifier())
             if scaled else IndexClassifier(record_names=True))
    model.fit(frame, y)
    native = model.predict(frame.copy())
    report = evaluate_classifier(model, frame, y)
    assert report["status"] == "valid"
    np.testing.assert_array_equal(report["predictions"]["predicted_labels"], native)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="[Ff]eature names"):
        evaluate_classifier(model, frame.iloc[:, ::-1], y)
