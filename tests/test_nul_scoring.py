"""External prediction labels must qualify before calibration scoring."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from supervised_nmf import _calibration
from supervised_nmf.evaluation import calibration_search


OBJECTIVES = ("accuracy", "balanced_accuracy")
MESSAGE = "Class labels ending in a NUL codepoint"


def prediction_container(values, representation):
    if representation == "list":
        return list(values)
    if representation == "tuple":
        return tuple(values)
    if representation == "column":
        return [[value] for value in values]
    if representation == "numpy_scalar":
        return np.array([np.str_(value) for value in values], dtype=object)
    if representation == "string_dtype":
        if not hasattr(np, "dtypes") or not hasattr(np.dtypes, "StringDType"):
            pytest.skip("StringDType is unavailable in this supported NumPy version")
        return np.array(values, dtype=np.dtypes.StringDType())
    return np.array(values, dtype=object)


class ExternalPredictionClassifier(ClassifierMixin, BaseEstimator):
    """Fit ordinary classes, then return selected external label representations."""

    def __init__(self, output="ordinary"):
        self.output = output

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        labels = self.classes_[X[:, 0].astype(int) % 2]
        if self.output == "ordinary":
            return labels
        values = ["a\0" if label == "a" else label for label in labels.tolist()]
        return prediction_container(values, self.output)


@pytest.fixture
def data():
    X = np.column_stack((np.arange(24), np.ones(24)))
    return X, np.array(["a", "b"] * 12, dtype=object)


@pytest.mark.parametrize("scoring", OBJECTIVES)
@pytest.mark.parametrize("representation", ("list", "tuple", "column", "object", "numpy_scalar", "string_dtype"))
def test_direct_scorers_reject_preserved_prediction_values_before_metrics(scoring, representation, monkeypatch):
    truth = np.array(["a", "b"] * 4, dtype=object)
    predicted = prediction_container(["a\0", "b"] * 4, representation)

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported prediction labels reached the numerical metric")

    monkeypatch.setitem(_calibration._SCORERS, scoring, unexpected)
    with pytest.raises(ValueError, match=MESSAGE):
        _calibration._score_predictions(scoring, truth, predicted)


@pytest.mark.parametrize("scoring", OBJECTIVES)
@pytest.mark.parametrize("representation", ("list", "object", "numpy_scalar"))
@pytest.mark.parametrize("refit", (False, True))
def test_generic_cv_cannot_select_unsafe_predictions_or_present_fallback_as_valid(
        data, scoring, representation, refit, tmp_path):
    X, y = data
    result = calibration_search(ExternalPredictionClassifier(representation), {}, X, y,
                                cv=3, seed=5, threads=1, scoring=scoring, refit=refit,
                                checkpoint_path=tmp_path / "cv.json")
    report = result.report
    assert report["selected_candidate_index"] is None
    assert report["selected_parameters"] is None
    assert report["selected_mean_score"] is None
    candidate, = report["candidates"]
    assert candidate["valid"] is False
    assert candidate["mean_score"] is None
    assert candidate["mean_" + scoring] is None
    assert candidate["invalid_fold_count"] == 3
    for fold in candidate["folds"]:
        assert fold["status"] == "failed"
        assert fold["error_type"] == "ValueError"
        assert MESSAGE in fold["error"]
        assert "score" not in fold
        assert scoring not in fold
    if refit:
        assert isinstance(result.estimator, DummyClassifier)
        assert report["is_fallback"] is True
        assert [attempt["status"] for attempt in report["final_fit_attempts"]] == ["fallback"]
    else:
        assert result.estimator is None
        assert report["is_fallback"] is False
        assert report["final_fit_attempts"] == []


@pytest.mark.parametrize("scoring", OBJECTIVES)
def test_valid_candidate_remains_selectable_beside_invalid_external_predictions(data, scoring):
    X, y = data
    result = calibration_search(ExternalPredictionClassifier(),
                                {"output": ["list", "ordinary"]}, X, y,
                                cv=3, seed=5, scoring=scoring, threads=1)
    assert [candidate["valid"] for candidate in result.report["candidates"]] == [False, True]
    assert result.report["selected_parameters"] == {"output": "ordinary"}
    assert result.report["selected_mean_score"] == 1.
    assert result.report["is_fallback"] is False
    assert result.estimator.output == "ordinary"


@pytest.mark.parametrize("scoring", OBJECTIVES)
@pytest.mark.parametrize("pair", (("a", "b"), ("a\0x", "b\0y"), ("\0a", "\0b"), (-3, 7)))
def test_ordinary_and_embedded_prediction_scores_keep_integer_partition_values(scoring, pair):
    true_codes = np.array([0, 0, 0, 0, 1, 1])
    predicted_codes = np.array([0, 1, 0, 0, 1, 0])
    truth = np.array([pair[i] for i in true_codes])
    predicted = [pair[i] for i in predicted_codes]
    metric = accuracy_score if scoring == "accuracy" else balanced_accuracy_score
    assert _calibration._score_predictions(scoring, truth, predicted) == metric(true_codes, predicted_codes)


@pytest.mark.parametrize("scoring", OBJECTIVES)
def test_scorer_reuses_validated_prediction_array(scoring, monkeypatch):
    codes = np.array([0, 1, 0, 1])
    conversions = []
    received = []

    class Predictions:
        def __array__(self, dtype=None, copy=None):
            conversions.append(dtype)
            result = np.asarray(codes, dtype=dtype)
            return result.copy() if copy else result

    def metric(y_true, y_pred):
        received.append(y_pred)
        assert y_true is codes
        assert type(y_pred) is np.ndarray
        np.testing.assert_array_equal(y_pred, codes)
        return .75

    monkeypatch.setitem(_calibration._SCORERS, scoring, metric)
    assert _calibration._score_predictions(scoring, codes, Predictions()) == .75
    assert len(conversions) == len(received) == 1


@pytest.mark.parametrize("scoring", OBJECTIVES)
def test_existing_nonfinite_metric_rejection_remains(scoring, monkeypatch):
    monkeypatch.setitem(_calibration._SCORERS, scoring, lambda *args: np.nan)
    with pytest.raises(ValueError, match="Nonfinite calibration score"):
        _calibration._score_predictions(scoring, np.array([0, 1]), np.array([0, 1]))
