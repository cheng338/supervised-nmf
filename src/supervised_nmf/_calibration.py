"""Explicit prediction-based calibration objectives, authored by Chang Cheng."""

import math

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ._labels import _label_values, _validate_class_labels


_SCORERS = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}


def _validate_scoring(scoring):
    # Deliberately restrict this interface to named label-based objectives:
    # no extra prediction/projection is needed and checkpoint identity is exact.
    if not isinstance(scoring, str) or scoring not in _SCORERS:
        raise ValueError("scoring must be 'accuracy' or 'balanced_accuracy'.")
    return scoring


def _score_predictions(scoring, y_true, y_pred):
    # Public calibration callers already validate y_true before making folds.
    # An external estimator can still return unsupported prediction labels.
    y_pred = _validate_class_labels(y_pred)
    return _score_label_values(scoring, _label_values(y_true), _label_values(y_pred))


def _score_label_values(scoring, y_true, y_pred):
    score = float(_SCORERS[scoring](y_true, y_pred))
    if not math.isfinite(score):
        raise ValueError("Nonfinite calibration score.")
    return score


def _score_fields(scoring, value, *, mean=False):
    """Generic field plus an honestly named metric alias (including failures)."""
    prefix = "mean_" if mean else ""
    return {prefix + "score": value, prefix + scoring: value}
