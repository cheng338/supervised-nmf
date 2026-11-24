"""Binary supervised screening derived from Chang Cheng's original scripts.

Author: Chang Cheng. Absolute-score screening extends the original percentile
ANOVA screen. For two classes, the pooled two-sample t statistic squared is F.
"""

from numbers import Real
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import get_config
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

from ._feature_names import _check_input_features, _input_feature_names
from ._labels import _check_labeled_data, _label_values, _validate_class_labels


def _clear_fitted_attributes(estimator):
    """Do not retain an earlier model if a subsequent fit fails."""
    for name in list(vars(estimator)):
        if name.endswith("_") and not name.startswith("__"):
            delattr(estimator, name)


def _nonnegative_array(X):
    X = check_array(X, accept_sparse=False, dtype=[np.float64, np.float32])
    if np.any(X < 0):
        raise ValueError("Negative values in data. X must contain nonnegative values; preprocessing must be fitted on training data.")
    return X


def _binary_data(X, y):
    raw_y = y
    X, y, original_y = _check_labeled_data(X, y, accept_sparse=False, dtype=[np.float64, np.float32])
    if np.any(X < 0):
        raise ValueError("Negative values in data. X must contain nonnegative values.")
    check_classification_targets(y)
    classes, counts = np.unique(y, return_counts=True)
    _validate_class_labels(raw_y, y, classes=classes)
    if classes.size != 2:
        raise ValueError(
            "Only binary classification is supported. Binary classification requires exactly two classes; "
            f"got {classes.size} {'class' if classes.size == 1 else 'classes'}."
        )
    if np.any(counts < 2):
        raise ValueError("At least two training observations per class are required.")
    if y is not original_y:
        classes = classes.astype(original_y.dtype)
    return X, original_y, classes, counts


def _percentile_support(scores, eligible, percentile):
    if percentile == 100:
        return eligible.copy()
    if percentile == 0:
        return np.zeros(scores.size, dtype=bool)
    # Interpolate between integer ranks so infinite or adjacent float64
    # magnitudes can be ordered without interpolating the magnitudes themselves.
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


def _screening_canonical_f(X):
    return (type(X) is np.ndarray and X.ndim == 2
            and X.shape[0] > 1 and X.shape[1] > 1
            and X.dtype.type in (np.float32, np.float64) and X.dtype.isnative
            and X.flags.aligned and X.flags.f_contiguous
            and not X.flags.c_contiguous
            and X.strides == (X.itemsize, X.shape[0] * X.itemsize))


def _screening_block_statistics(translated, y, classes, counts):
    # The caller has already completed the original full subtraction snapshot.
    # Capture class membership once; later blocks never reread caller labels.
    masks = [y == label for label in classes]
    p = translated.shape[1]
    scales = np.empty(p, dtype=np.float64)
    pooled = np.empty(p, dtype=np.float64)
    difference = np.empty(p, dtype=np.float64)
    standard_error = np.empty(p, dtype=np.float64)
    t_scores = np.zeros(p, dtype=np.float64)
    group_constant = [np.empty(p, dtype=bool) for _ in classes]
    begin = 0
    while begin < p:
        end = min(begin + 128, p)
        # Never create a one-column group: that changes NumPy's sum geometry.
        if p - end == 1:
            end -= 1
        columns = slice(begin, end)
        block = translated[:, columns]
        block_scales = block.max(axis=0)
        scaled = np.divide(block, block_scales, out=np.zeros_like(block),
                           where=block_scales > 0)
        groups = [scaled[mask] for mask in masks]
        fixed_groups = [np.all(group == group[0], axis=0) for group in groups]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                means = [group.mean(axis=0) for group in groups]
                variances = [group.var(axis=0, ddof=1) for group in groups]
                for group, mean, variance, fixed in zip(groups, means, variances, fixed_groups):
                    mean[fixed] = group[0, fixed]
                    variance[fixed] = 0.0
                block_pooled = ((counts[0] - 1) * variances[0] + (counts[1] - 1) * variances[1]) / (counts.sum() - 2)
                block_difference = means[1] - means[0]
                block_error = np.sqrt(block_pooled * (1 / counts[0] + 1 / counts[1]))
                np.divide(block_difference, block_error, out=t_scores[columns], where=block_error > 0)
        except FloatingPointError as exc:
            raise ValueError("Screening statistics exceeded float64 range; rescale X consistently before fitting.") from exc
        scales[columns] = block_scales
        pooled[columns] = block_pooled
        difference[columns] = block_difference
        standard_error[columns] = block_error
        for destination, fixed in zip(group_constant, fixed_groups):
            destination[columns] = fixed
        begin = end
    return scales, group_constant, pooled, difference, standard_error, t_scores


def _screening_capture_reuse(policy):
    """Capture direct numerical bindings for C-layout statistics reuse.

    Decline if captured operations are replaced or their code/defaults change.
    These checks cover direct dependencies, not arbitrary library changes.
    """
    numpy = np
    array_type = np.ndarray
    watched = []
    enabled = False

    def token(value):
        keywords = getattr(value, "__kwdefaults__", None)
        return (value, getattr(value, "__code__", None),
                getattr(value, "__defaults__", None), keywords,
                tuple(keywords.items()) if type(keywords) is dict else None)

    def watch(mapping, names):
        for name in names:
            watched.append((mapping, name, token(mapping[name])))

    def unchanged(value, saved):
        if (value is not saved[0] or getattr(value, "__code__", None) is not saved[1]
                or getattr(value, "__defaults__", None) is not saved[2]):
            return False
        keywords = getattr(value, "__kwdefaults__", None)
        if keywords is not saved[3]:
            return False
        return (saved[4] is None
                or (type(keywords) is dict and len(keywords) == len(saved[4])
                    and all(key in keywords and keywords[key] is item
                            for key, item in saved[4])))

    try:
        watch(globals(), ("np", "get_config"))
        watch(vars(np), ("ndarray", "float32", "float64", "geterr",
                         "zeros_like", "divide", "__version__"))
        if policy == "mean":
            # mean= is the public np.var API added in NumPy 2.0. It is not
            # assumed to exist on every earlier ndarray.var method.
            enabled = np.lib.NumpyVersion(np.__version__) >= "2.0.0"
            if enabled:
                import importlib
                methods = importlib.import_module("numpy._core._methods")
                watch(vars(np), ("var", "_NoValue"))
                watch(vars(np.ndarray), ("mean", "var"))
                watch(vars(np.var), ("__wrapped__",))
                watch(np.var.__wrapped__.__globals__, ("_methods", "mu", "np"))
                watch(vars(methods), ("_mean", "_var", "umr_sum",
                                     "_count_reduce_items", "asanyarray", "um"))
                watch(vars(methods.um), ("true_divide", "subtract", "square",
                                        "maximum"))
        elif policy == "normalize":
            watch(vars(np), ("any",))
            watch(vars(np.zeros_like), ("__wrapped__",))
            original = np.zeros_like.__wrapped__
            watch(original.__globals__, ("empty_like", "zeros", "multiarray"))
            watch(vars(original.__globals__["multiarray"]), ("copyto",))
            enabled = True
        else:
            raise ValueError("Unknown screening experiment")
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        # Missing optional implementation details preserve the original path.
        enabled = False

    def ready(X, scratch):
        # Leave existing F and noncanonical input routes alone. The inline
        # caller supplies the original subtraction/normalization result.
        if type(X) is not array_type or X.ndim != 2 or not X.flags.c_contiguous:
            return False
        # Skip small C-layout statistics, except tall two-feature means.
        # Keep the per-phase binding and numerical-policy checks below.
        if X.size < 131072 and not (
            policy == "mean" and X.shape[1] == 2 and X.shape[0] >= 8192
        ):
            return False
        if not enabled:
            return False
        try:
            if not all(unchanged(mapping[name], saved)
                       for mapping, name, saved in watched):
                return False
        except (KeyError, AttributeError):
            return False
        return (np is numpy and X.dtype.type in (np.float32, np.float64)
                and X.dtype.isnative
                and type(scratch) is array_type and scratch.ndim == 2
                and scratch.shape == X.shape and scratch.dtype.type is np.float64
                and scratch.dtype.isnative and scratch.flags.c_contiguous
                and scratch.flags.owndata and scratch.flags.writeable
                and scratch.base is None
                and np.geterr()["under"] == "ignore"
                and get_config()["assume_finite"] is False)

    # Descriptive only; changing these attributes cannot enable the closure.
    ready.policy = policy
    ready.supports_policy = enabled
    ready.binding_count = len(watched)
    return ready


_SCREENING_MEAN_READY = _screening_capture_reuse("mean")
_SCREENING_NORMALIZE_READY = _screening_capture_reuse("normalize")


class BinaryScreening(TransformerMixin, BaseEstimator):
    """Fit a binary ANOVA/pooled-t feature screen on training observations.

    This transformer and NMFClassifier default to ``screening="absolute"``
    with ``score_threshold=2.0``. Set ``screening="percentile"`` explicitly
    to use percentile selection. When choosing or tuning either policy,
    ``percentile`` controls percentile mode and ``score_threshold`` controls
    absolute mode. Changing either parameter does not switch the mode.

    Choose percentile mode for a relative feature-budget policy, subject to
    the documented rounding and eligibility rules. Choose absolute mode for
    an explicit pooled-t cutoff; its default ``score_threshold`` is 2.0 and
    the number retained depends on the data. More retained features increase
    downstream work; stricter selection can remove useful features or leave
    too few for the requested rank. This standalone screen accepts only these
    two modes; omit the step to perform no screening. Changing constructor
    settings, including through ``set_params``, affects the next fit.
    Transform keeps using the stored mask until refitting.

    Percentile screening applies the percentile cutoff to ranks of the
    computed float64 magnitudes of ``t_scores_``, using the original feature
    count. Features above the cutoff are retained. If a cutoff splits a group
    of equal magnitudes, lower input-column indices take the remaining places.
    Mathematically equal statistics can round to different magnitudes and then
    follow their computed order. Globally constant features are always excluded.
    ``absolute`` keeps ``abs(t) >= score_threshold`` (a t threshold, not an F
    threshold). Features constant within each class but different between
    classes have infinite scores and are eligible; no variance floor is added.
    The signed t statistic uses mean(classes_[1]) - mean(classes_[0]).
    Statistics use float64 after subtracting each training feature's minimum
    and rescaling its range. In exact arithmetic, centering and rescaling
    a nonconstant feature preserve its pooled t statistic.
    Transform preserves input float32/float64 values. Diagnostic variances and
    standard errors are in these centered, scaled units.
    F scores can overflow when finite t scores are squared; selection ranks |t|
    directly and ``n_f_overflow_`` records this distinct numerical event.
    Both modes raise when no features survive, making invalid search candidates
    visible rather than silently changing the requested screen.
    Class labels must not end in a NUL codepoint (``\\x00``). Encode such
    categories as distinct integers before converting them to NumPy strings.
    """

    def __init__(self, screening="absolute", percentile=10.0, score_threshold=2.0):
        self.screening = screening
        self.percentile = percentile
        self.score_threshold = score_threshold

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        tags.target_tags.required = True
        return tags

    def fit(self, X, y):
        _clear_fitted_attributes(self)
        self._validate_settings()
        feature_names = _input_feature_names(X)
        X, y, classes, counts = _binary_data(X, y)
        return self._fit_statistics(X, _label_values(y), classes, counts, feature_names)

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
        _validate_class_labels(reference.classes_)
        constant = np.zeros(reference.n_features_in_, dtype=bool)
        constant[reference.constant_features_] = True
        support = self._support_from_statistics(reference.t_scores_, constant)
        try:
            for name, value in vars(reference).items():
                if name.endswith("_") and not name.startswith("__"):
                    setattr(self, name, deepcopy(value))
            self.support_ = support
            self.selected_features_ = np.flatnonzero(support)
            self.n_features_selected_ = int(support.sum())
        except Exception:
            _clear_fitted_attributes(self)
            raise
        return self

    def _fit_statistics(self, X, y, classes, counts, feature_names=None):
        # Translate and scale only statistics, never transformed features.
        # Subtract before dividing: scaling a large offset first can discard
        # a small spread that is still represented in the supplied input.
        # Nonnegative finite X makes this subtraction overflow-safe; a minimum
        # reference is independent of row order. Exact pooled t/F is invariant.
        if (_screening_canonical_f(X) and X.dtype.type is np.float32
                and X.shape[1] > 128 and 8 * X.size <= 1_048_576):
            X_statistics = X.astype(np.float64, order="C", copy=False)
        else:
            X_statistics = X.astype(np.float64, copy=False)
        offsets = X_statistics.min(axis=0)
        X_statistics = X_statistics - offsets
        if (_screening_canonical_f(X) and X.shape[1] > 128
                and X_statistics.nbytes > 1_048_576
                and np.geterr()["under"] == "ignore"
                and get_config()["assume_finite"] is False):
            scales, group_constant, pooled, difference, standard_error, t_scores = _screening_block_statistics(X_statistics, y, classes, counts)
        else:
            scales = X_statistics.max(axis=0)
            reuse_normalization = _SCREENING_NORMALIZE_READY(X, X_statistics)
            if reuse_normalization:
                # X_statistics is this call's unchanged full subtraction
                # result. Caller X and the fitted state never alias it.
                zero_columns = scales == 0
                if np.any(zero_columns):
                    # Match zeros_like's positive zeros in masked columns.
                    X_statistics[:, zero_columns] = 0.0
                scaled = np.divide(X_statistics, scales, out=X_statistics,
                                   where=scales > 0)
            else:
                scaled = np.divide(X_statistics, scales, out=np.zeros_like(X_statistics), where=scales > 0)
            groups = [scaled[y == label] for label in classes]
            group_constant = [np.all(group == group[0], axis=0) for group in groups]
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    means = [group.mean(axis=0) for group in groups]
                    reuse_mean = (_SCREENING_MEAN_READY(X, scaled)
                                  and all(type(group) is np.ndarray
                                          and group.dtype.type is np.float64
                                          and group.dtype.isnative
                                          and group.flags.c_contiguous
                                          and group.shape[0] >= 2
                                          for group in groups))
                    if reuse_mean:
                        # Use the original means before constant-column repair.
                        variances = [np.var(group, axis=0, ddof=1, mean=mean[None, :])
                                     for group, mean in zip(groups, means)]
                    else:
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
        try:
            if feature_names is not None:
                self.feature_names_in_ = feature_names
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
        except Exception:
            _clear_fitted_attributes(self)
            raise
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        _check_input_features(self, X, reset=False)
        X = _nonnegative_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        _validate_class_labels(self.classes_)
        return X[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        _validate_class_labels(self.classes_)
        return self.selected_features_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        _validate_class_labels(self.classes_)
        fitted_names = getattr(self, "feature_names_in_", None)
        if input_features is None and fitted_names is not None:
            return fitted_names[self.support_]
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or names.size != self.n_features_in_:
            raise ValueError("input_features must contain one name per original feature.")
        if fitted_names is not None and not np.array_equal(names, fitted_names):
            raise ValueError("input_features is not equal to feature_names_in_")
        return names[self.support_]
