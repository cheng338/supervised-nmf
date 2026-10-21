"""Binary supervised screening derived from Chang Cheng's original scripts.

Author: Chang Cheng. Absolute-score screening extends the original percentile
ANOVA screen. For two classes, the pooled two-sample t statistic squared is F.
"""

from numbers import Real
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _clear_fitted_attributes(estimator):
    """Do not retain an earlier model if a subsequent fit fails."""
    for name in list(vars(estimator)):
        if name.endswith("_") and not name.startswith("__"):
            delattr(estimator, name)


def _nonnegative_array(X):
    X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
    if np.any(X < 0):
        raise ValueError("X must contain nonnegative values; preprocessing must be fitted on training data.")
    return X


def _binary_data(X, y):
    X, y = check_X_y(X, y, accept_sparse=False, dtype=[np.float64, np.float32])
    if np.any(X < 0):
        raise ValueError("X must contain nonnegative values.")
    check_classification_targets(y)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        raise ValueError("Binary classification requires exactly two classes.")
    if np.any(counts < 2):
        raise ValueError("At least two training observations per class are required.")
    return X, y, classes, counts


def _percentile_support(scores, eligible, percentile):
    if percentile == 100:
        return eligible.copy()
    if percentile == 0:
        return np.zeros(scores.size, dtype=bool)
    # Any strictly increasing score mapping leaves the percentile mask unchanged.
    # Ranking distinct scores avoids undefined interpolation with infinite scores.
    ranking = scores.copy()
    ranking[~eligible] = -np.inf
    _, ordered = np.unique(ranking, return_inverse=True)
    threshold = np.percentile(ordered, 100 - percentile)
    mask = ordered > threshold
    ties = np.flatnonzero(ordered == threshold)
    if ties.size:
        n_keep = int(scores.size * percentile / 100)
        mask[ties[: max(0, n_keep - int(mask.sum()))]] = True
    return mask & eligible


class BinaryScreening(TransformerMixin, BaseEstimator):
    """Fit a binary ANOVA/pooled-t feature screen on training observations.

    ``percentile`` follows sklearn SelectPercentile rounding and deterministic
    index-order tie handling for finite nonconstant scores. Its denominator is
    the original feature count. Globally constant features are always excluded.
    ``absolute`` keeps ``abs(t) >= score_threshold`` (a t threshold, not an F
    threshold). Features constant within each class but different between
    classes have infinite scores and are eligible; no variance floor is added.
    The signed t statistic uses mean(classes_[1]) - mean(classes_[0]).
    Statistics use float64 after subtracting each training feature's minimum
    and rescaling its range. This preserves the exact pooled t/F statistic
    while avoiding avoidable cancellation and extreme measurement units.
    Transform preserves input float32/float64 values. Diagnostic variances and
    standard errors are in these centered, scaled units.
    F scores can overflow when finite t scores are squared; selection ranks |t|
    directly and ``n_f_overflow_`` records this distinct numerical event.
    Both modes raise when no features survive, making invalid search candidates
    visible rather than silently changing the requested screen.
    """

    def __init__(self, screening="percentile", percentile=10.0, score_threshold=2.0):
        self.screening = screening
        self.percentile = percentile
        self.score_threshold = score_threshold

    def fit(self, X, y):
        _clear_fitted_attributes(self)
        self._validate_settings()
        X, y, classes, counts = _binary_data(X, y)
        return self._fit_statistics(X, y, classes, counts)

    def _validate_settings(self):
        if self.screening not in {"percentile", "absolute"}:
            raise ValueError("screening must be 'percentile' or 'absolute'.")
        if not isinstance(self.percentile, Real) or not np.isfinite(self.percentile) or not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be finite and between 0 and 100.")
        if not isinstance(self.score_threshold, Real) or not np.isfinite(self.score_threshold) or self.score_threshold < 0:
            raise ValueError("score_threshold must be finite and nonnegative.")

    def _support_from_statistics(self, t_scores, constant):
        if self.screening == "percentile":
            support = _percentile_support(np.abs(t_scores), ~constant, self.percentile)
        else:
            support = (np.abs(t_scores) >= self.score_threshold) & ~constant
        if not support.any():
            raise ValueError("Screening selected zero features; choose a different threshold or percentile.")
        return support

    def _fit_from_statistics(self, reference):
        """Reuse trusted statistics from the identical training observations.

        This private helper is used only by the fold-local calibration cache.
        It recomputes the requested mask, including ties and infinite scores;
        it never borrows the reference screen's selected features.
        """
        _clear_fitted_attributes(self)
        self._validate_settings()
        if type(reference) is not BinaryScreening:
            raise TypeError("reference must be a fitted BinaryScreening instance.")
        check_is_fitted(reference, "support_")
        constant = np.zeros(reference.n_features_in_, dtype=bool)
        constant[reference.constant_features_] = True
        support = self._support_from_statistics(reference.t_scores_, constant)
        for name, value in vars(reference).items():
            if name.endswith("_") and not name.startswith("__"):
                setattr(self, name, deepcopy(value))
        self.support_ = support
        self.selected_features_ = np.flatnonzero(support)
        self.n_features_selected_ = int(support.sum())
        return self

    def _fit_statistics(self, X, y, classes, counts):
        # Translate and scale only statistics, never transformed features.
        # Subtract before dividing: scaling a large offset first can discard
        # a small spread that is still represented in the supplied input.
        # Nonnegative finite X makes this subtraction overflow-safe; a minimum
        # reference is independent of row order. Exact pooled t/F is invariant.
        X_statistics = X.astype(np.float64, copy=False)
        offsets = X_statistics.min(axis=0)
        X_statistics = X_statistics - offsets
        scales = X_statistics.max(axis=0)
        scaled = np.divide(X_statistics, scales, out=np.zeros_like(X_statistics), where=scales > 0)
        groups = [scaled[y == label] for label in classes]
        group_constant = [np.all(group == group[0], axis=0) for group in groups]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                means = [group.mean(axis=0) for group in groups]
                variances = [group.var(axis=0, ddof=1) for group in groups]
                for group, mean, variance, fixed in zip(groups, means, variances, group_constant):
                    mean[fixed] = group[0, fixed]
                    variance[fixed] = 0.0
                pooled = ((counts[0] - 1) * variances[0] + (counts[1] - 1) * variances[1]) / (counts.sum() - 2)
                difference = means[1] - means[0]
                standard_error = np.sqrt(pooled * (1 / counts[0] + 1 / counts[1]))
                t_scores = np.zeros(X.shape[1], dtype=np.float64)
                np.divide(difference, standard_error, out=t_scores, where=standard_error > 0)
        except FloatingPointError as exc:
            raise ValueError("Screening statistics exceeded float64 range; rescale X consistently before fitting.") from exc
        constant = np.all(X == X[0], axis=0)
        classwise_constant = group_constant[0] & group_constant[1] & ~constant
        if np.any((standard_error == 0) & ~constant & ~classwise_constant):
            raise ValueError("A nonconstant feature's within-class variance underflowed in float64.")
        t_scores[classwise_constant] = np.copysign(np.inf, difference[classwise_constant])
        t_scores[constant] = 0.0
        with np.errstate(over="ignore"):
            scores = np.square(t_scores)
        support = self._support_from_statistics(t_scores, constant)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.classes_ = classes
        self.class_counts_ = counts
        self.scores_ = scores
        self.t_scores_ = t_scores
        self.statistic_offsets_ = offsets
        self.statistic_scales_ = scales
        self.scaled_pooled_variances_ = pooled
        self.scaled_standard_errors_ = standard_error
        self.support_ = support
        self.selected_features_ = np.flatnonzero(support)
        self.n_features_selected_ = int(support.sum())
        self.constant_features_ = np.flatnonzero(constant)
        self.n_constant_features_ = int(constant.sum())
        self.n_infinite_scores_ = int(np.isinf(scores).sum())
        self.n_infinite_t_scores_ = int(np.isinf(t_scores).sum())
        self.n_f_overflow_ = int((np.isfinite(t_scores) & np.isinf(scores)).sum())
        self.n_classwise_constant_features_ = int(classwise_constant.sum())
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = _nonnegative_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}.")
        return X[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return self.selected_features_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or names.size != self.n_features_in_:
            raise ValueError("input_features must contain one name per original feature.")
        return names[self.support_]
