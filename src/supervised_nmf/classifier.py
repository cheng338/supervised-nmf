"""Screening and NMF classification, based on Chang Cheng's original scripts."""

from numbers import Integral, Real
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from .decomposition import FeatureSPA
from ._projection import validate_projection_parameters
from ._residual import reconstruction_error
from .screening import BinaryScreening, _binary_data, _clear_fitted_attributes, _nonnegative_array


def _record_warnings(operation):
    caught = []
    succeeded = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = operation()
        succeeded = True
    finally:
        # Failed fits can emit useful diagnostics before raising. Forward those
        # too, so the comparison runner retains every failed attempt's warning.
        for warning in caught:
            try:
                warnings.warn(warning.message, warning.category, stacklevel=3)
            except Warning:
                # A caller may promote warnings to exceptions. Do not replace
                # an already pending solver error with a forwarded warning.
                if succeeded:
                    raise
    messages = tuple(str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning))
    return result, messages


def _head_supports(method):
    """Use fitted capabilities until refit, including after set_params."""
    def check(estimator):
        if hasattr(estimator, "classifier_"):
            return hasattr(estimator.classifier_, method)
        classifier = estimator.classifier
        if not isinstance(classifier, str):
            return False
        if method == "predict_proba":
            return classifier in {"knn", "logistic"}
        return classifier in {"logistic", "linear_svm"}
    return check


class NMFClassifier(ClassifierMixin, BaseEstimator):
    """Binary supervised NMF classifier with an overridable practical default.

    ``solver='spa'`` and ``screening='absolute'`` are practical starting defaults
    supported by the method, numerical and workload reassessment. They are not
    universal accuracy or timing winners. Choose ``solver='cd'``
    or ``solver='mu'``, or screening ``'percentile'``/``'none'``, explicitly to use
    another method.
    Rank, screening level and final classifier parameters still require
    data-specific tuning.
    Coordinate descent uses sklearn NMF with NNDSVDa initialization and Frobenius
    loss, matching the original scripts. Multiplicative updates use the same
    initialization and loss with sklearn's MU solver. Raw nonnegative feature
    values and Euclidean KNN coordinates are not automatically scaled.

    ``classifier='knn'`` retains the original final classifier.
    ``knn_normalization='none'`` preserves its original metric. Optional ``'sd'``
    divides each coordinate by its training-fitted standard deviation without
    centering, using ``StandardScaler(with_mean=False)``. Constant/nearconstant
    coordinates follow sklearn's scale fallback. This changes the KNN metric;
    it does not change ``transform`` or ``training_coordinates_``. The fitted
    scaler and KNN are stored in a ``classifier_`` pipeline with steps ``'scale'``
    and ``'knn'``. This option is validated but unused by the other heads.

    The optional
    ``classifier='logistic'`` extension standardizes the NMF coordinates using
    training means and standard deviations, then fits L2-regularized logistic
    regression with sklearn's LBFGS solver. ``logistic_C`` is the inverse
    regularization strength; ``logistic_class_weight`` accepts ``None``,
    ``'balanced'``, or a dictionary of class weights. These parameters affect
    only logistic regression. ``n_neighbors`` and ``weights`` affect only KNN.
    The scaler and final estimator are stored in ``classifier_`` as a pipeline
    with steps ``'scale'`` and ``'logistic'`` when logistic regression is used.

    ``classifier='linear_svm'`` fits a linear SVM on the unscaled coordinates,
    using ``LinearSVC(penalty='l2', loss='squared_hinge', dual=False)``.
    ``svm_C``, ``svm_max_iter``, ``svm_tol`` and ``svm_class_weight`` affect only
    this head. C interacts with the coordinate units; no hidden scaler or
    probability calibration is fitted. ``decision_function`` returns real
    margins oriented toward ``classes_[1]``; an exact zero predicts
    ``classes_[0]``. ``predict_with_decision_function`` shares one projection
    between labels and margins. Probability methods are unavailable for SVM,
    and a non-None ``decision_threshold`` is rejected before factorization.

    ``decision_threshold=None`` preserves the final classifier's own decision
    rule. An explicit threshold predicts ``positive_label`` exactly when its
    probability is strictly greater than the threshold; equality is negative.
    ``positive_label=None`` uses the second class in ``classes_``. Thresholds
    lie in [0, 1], with the exact ``np.nextafter(0.0, -np.inf)`` sentinel also
    accepted for an all-positive rule. A threshold of 1 is all-negative.
    Probabilities are unchanged. No threshold is learned by this estimator:
    choose it on training/calibration data, never on test data. The fitted
    policy is stored in ``decision_threshold_`` and ``positive_label_``;
    changing constructor parameters takes effect on the next fit.
    ``positive_label`` affects probability-threshold decisions only. In
    particular, it does not reverse SVM margins or its zero-threshold rule.

    ``projection_solver``, ``projection_workers``, ``projection_batch_size``
    and ``projection_native_threads`` configure SPA's fixed-basis prediction
    projection (and training reprojection when requested). See ``FeatureSPA``
    for their numerical safeguards and process-pool behavior. They are
    validated but unused by CD and MU. Unlike the fitted decision policy,
    projection execution settings may be changed without refitting.

    ``training_coordinates='auto'`` uses fixed-basis ``'transform'`` coordinates
    for SPA and the historical ``'fit_transform'`` coordinates for CD and MU.
    Explicit ``'fit_transform'`` reproduces the scripts' convention;
    ``'transform'`` reprojects training data against the fitted basis using the
    same operation as prediction. ``training_coordinates_mode_`` records the
    resolved fitted choice; ``training_coordinates_`` holds the actual array.
    Changing this parameter takes effect on the next fit, not prediction.
    No operation on test data refits
    the screening or basis. ``components_`` is padded with zeros to the original
    feature count; ``selected_components_`` contains only retained columns.
    Fit times exclude tuning; callers must separately record search costs.
    The default iteration cap is 20,000 for CD/MU or each SPA NNLS solve;
    converged routines can terminate earlier. CD and MU use their respective
    sklearn stopping criteria, so identical tolerances do not imply identical
    numerical accuracy. SPA's NNLS routine does not use ``tol``. Increasing the
    cap can change a previously truncated fit without changing its objective
    or stopping tolerance.

    ``n_iter_`` records factorization iterations. ``classifier_n_iter_`` records
    logistic regression/SVM iterations, or is ``None`` for KNN. ``converged_`` and
    ``convergence_warnings_`` include factorization, training projection, and
    classifier fitting; classifier-specific diagnostics are also available as
    ``classifier_converged_`` and ``classifier_convergence_warnings_``.
    """

    def __init__(self, n_components=2, *, solver="spa", screening="absolute", percentile=10.0,
                 score_threshold=2.0, n_neighbors=5, weights="uniform", max_iter=20000,
                 tol=1e-4, random_state=None, training_coordinates="auto",
                 classifier="knn", knn_normalization="none",
                 logistic_C=1.0, logistic_max_iter=1000,
                 logistic_tol=1e-4, logistic_class_weight=None,
                 svm_C=1.0, svm_max_iter=20000, svm_tol=1e-5,
                 svm_class_weight=None,
                 decision_threshold=None, positive_label=None,
                 projection_solver="auto", projection_workers=1,
                 projection_batch_size=256, projection_native_threads=1):
        self.n_components = n_components
        self.solver = solver
        self.screening = screening
        self.percentile = percentile
        self.score_threshold = score_threshold
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.training_coordinates = training_coordinates
        self.classifier = classifier
        self.knn_normalization = knn_normalization
        self.logistic_C = logistic_C
        self.logistic_max_iter = logistic_max_iter
        self.logistic_tol = logistic_tol
        self.logistic_class_weight = logistic_class_weight
        self.svm_C = svm_C
        self.svm_max_iter = svm_max_iter
        self.svm_tol = svm_tol
        self.svm_class_weight = svm_class_weight
        self.decision_threshold = decision_threshold
        self.positive_label = positive_label
        self.projection_solver = projection_solver
        self.projection_workers = projection_workers
        self.projection_batch_size = projection_batch_size
        self.projection_native_threads = projection_native_threads

    def __setstate__(self, state):
        # Legacy fitted models retain their original serial projection policy;
        # restoring constructor defaults also keeps get_params/clone usable.
        state = state.copy()
        for name, value in (("decision_threshold", None), ("positive_label", None),
                            ("knn_normalization", "none"),
                            ("svm_C", 1.0), ("svm_max_iter", 20000),
                            ("svm_tol", 1e-5), ("svm_class_weight", None),
                            ("training_coordinates", "fit_transform"),
                            ("projection_solver", "scipy"), ("projection_workers", 1),
                            ("projection_batch_size", 256), ("projection_native_threads", 1)):
            state.setdefault(name, value)
        if "classifier_" in state and "knn_normalization_" not in state:
            head = state["classifier_"]
            if isinstance(head, KNeighborsClassifier):
                state["knn_normalization_"] = "none"
            elif isinstance(head, Pipeline) and isinstance(head.steps[-1][1], KNeighborsClassifier):
                state["knn_normalization_"] = "sd"
            else:
                state["knn_normalization_"] = None
        if "classifier_" in state and "training_coordinates_mode_" not in state:
            # This records the saved fit, even if set_params subsequently
            # changed its constructor policy. It never re-encodes fitted data.
            if "training_projection_performed_" in state:
                state["training_coordinates_mode_"] = (
                    "transform" if state["training_projection_performed_"] else "fit_transform")
            elif state["training_coordinates"] in {"fit_transform", "transform"}:
                state["training_coordinates_mode_"] = state["training_coordinates"]
        super().__setstate__(state)

    def fit(self, X, y):
        return self._fit(X, y)

    def _fit_with_screening(self, X, y, selector):
        """Reuse a selector fitted on this training fold by the study runner.

        This private entry point does not fit or select features globally. The
        caller owns the fold/data binding and counts screening computation once.
        """
        return self._fit(X, y, prepared_selector=selector)

    def _fit(self, X, y, prepared_selector=None):
        _clear_fitted_attributes(self)
        try:
            return self._fit_impl(X, y, prepared_selector)
        except Exception:
            _clear_fitted_attributes(self)
            raise

    def _fit_impl(self, X, y, prepared_selector=None):
        start = time.perf_counter()
        threshold = getattr(self, "decision_threshold", None)
        if threshold is not None and (
                isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
                or not np.isfinite(threshold)
                or (threshold < 0 and threshold != np.nextafter(0.0, -np.inf))
                or threshold > 1.0):
            raise ValueError("decision_threshold must be None, finite in [0, 1], or np.nextafter(0.0, -np.inf).")
        projection_settings = self._projection_settings()
        if self.solver not in {"cd", "spa", "mu"}:
            raise ValueError("solver must be 'cd', 'spa', or 'mu'.")
        if self.screening not in {"percentile", "absolute", "none"}:
            raise ValueError("screening must be 'percentile', 'absolute', or 'none'.")
        if (not isinstance(self.training_coordinates, str)
                or self.training_coordinates not in {"auto", "fit_transform", "transform"}):
            raise ValueError("training_coordinates must be 'auto', 'fit_transform', or 'transform'.")
        training_coordinates_mode = self.training_coordinates
        if training_coordinates_mode == "auto":
            training_coordinates_mode = "transform" if self.solver == "spa" else "fit_transform"
        if not isinstance(self.classifier, str) or self.classifier not in {"knn", "logistic", "linear_svm"}:
            raise ValueError("classifier must be 'knn', 'logistic', or 'linear_svm'.")
        if (not isinstance(self.knn_normalization, str)
                or self.knn_normalization not in {"none", "sd"}):
            raise ValueError("knn_normalization must be 'none' or 'sd'.")
        if self.classifier == "linear_svm" and threshold is not None:
            raise ValueError("decision_threshold requires probabilities and is unavailable for linear_svm.")
        if self.classifier in {"logistic", "linear_svm"}:
            prefix = "logistic" if self.classifier == "logistic" else "svm"
            for name in (f"{prefix}_C", f"{prefix}_tol"):
                value = getattr(self, name)
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive.")
            iterations = getattr(self, f"{prefix}_max_iter")
            if (isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, Integral)
                    or iterations < 1):
                raise ValueError(f"{prefix}_max_iter must be a positive integer.")
            class_weight = getattr(self, f"{prefix}_class_weight")
            if isinstance(class_weight, dict):
                for value in class_weight.values():
                    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                        raise ValueError(f"{prefix}_class_weight values must be finite and nonnegative.")
            elif class_weight is not None and not (
                    isinstance(class_weight, str) and class_weight == "balanced"):
                raise ValueError(f"{prefix}_class_weight must be None, 'balanced', or a dictionary.")
        X, y, classes, counts = _binary_data(X, y)
        positive_label = getattr(self, "positive_label", None)
        positive_index = 1
        if positive_label is not None:
            if not np.isscalar(positive_label):
                raise ValueError("positive_label must be one of the two training classes.")
            matches = [i for i, label in enumerate(classes) if label == positive_label]
            if len(matches) != 1:
                raise ValueError("positive_label must be one of the two training classes.")
            positive_index = matches[0]
        if self.classifier == "knn" and (isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, Integral) or not 1 <= self.n_neighbors <= X.shape[0]):
            raise ValueError("n_neighbors must be a positive integer no larger than the training sample count.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.tol, Real) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        stage = time.perf_counter()
        if self.screening == "none":
            if prepared_selector is not None:
                raise ValueError("A prepared selector requires screening to be enabled.")
            selector = None
            selected_features = np.arange(X.shape[1])
            X_selected = X
        else:
            if prepared_selector is None:
                selector = BinaryScreening(self.screening, self.percentile, self.score_threshold)
                X_selected = selector.fit_transform(X, y)
            else:
                selector = prepared_selector
                if type(selector) is not BinaryScreening:
                    raise TypeError("The prepared selector must be a fitted BinaryScreening instance.")
                check_is_fitted(selector, "support_")
                if (selector.screening != self.screening or selector.percentile != self.percentile
                        or selector.score_threshold != self.score_threshold
                        or selector.n_samples_seen_ != len(y)
                        or not np.array_equal(selector.classes_, classes)
                        or not np.array_equal(selector.class_counts_, counts)):
                    raise ValueError("Prepared screening does not match this training fold or estimator.")
                X_selected = selector.transform(X)
            selected_features = selector.get_support(indices=True)
        screen_seconds = time.perf_counter() - stage
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, Integral) or not 1 <= self.n_components <= min(X_selected.shape):
            raise ValueError("n_components must be between 1 and min(training samples, selected features).")
        if not np.any(X_selected):
            raise ValueError("Factorization requires at least one positive training value.")
        if self.solver in {"cd", "mu"}:
            decomposition = NMF(n_components=self.n_components, init="nndsvda", solver=self.solver,
                                beta_loss="frobenius",
                                max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
        else:
            decomposition = FeatureSPA(
                n_components=self.n_components, nnls_max_iter=self.max_iter,
                **projection_settings)
        stage = time.perf_counter()
        W, fitting_warnings = _record_warnings(lambda: decomposition.fit_transform(X_selected))
        factorization_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        projection_warnings = ()
        if training_coordinates_mode == "transform":
            W, projection_warnings = _record_warnings(lambda: decomposition.transform(X_selected))
        training_projection_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        if self.classifier == "knn":
            classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
            if self.knn_normalization == "sd":
                classifier = Pipeline([
                    ("scale", StandardScaler(with_mean=False)),
                    ("knn", classifier),
                ])
            classifier.fit(W, y)
            classifier_warnings = ()
            classifier_n_iter = None
        elif self.classifier == "logistic":
            classifier = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(
                    C=self.logistic_C, solver="lbfgs", max_iter=self.logistic_max_iter,
                    tol=self.logistic_tol, class_weight=self.logistic_class_weight,
                    random_state=self.random_state)),
            ])
            _, classifier_warnings = _record_warnings(lambda: classifier.fit(W, y))
            classifier_n_iter = classifier.named_steps["logistic"].n_iter_.copy()
        else:
            classifier = LinearSVC(
                C=self.svm_C, penalty="l2", loss="squared_hinge", dual=False,
                max_iter=self.svm_max_iter, tol=self.svm_tol,
                class_weight=self.svm_class_weight, random_state=self.random_state)
            _, classifier_warnings = _record_warnings(lambda: classifier.fit(W, y))
            classifier_n_iter = int(classifier.n_iter_)
        classifier_seconds = time.perf_counter() - stage
        components = np.zeros((self.n_components, X.shape[1]), dtype=np.float64)
        components[:, selected_features] = decomposition.components_
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.classes_ = classes
        self.class_counts_ = counts
        self.decision_threshold_ = None if threshold is None else float(threshold)
        self.positive_label_ = classes[positive_index]
        self.positive_class_index_ = positive_index
        self.selector_ = selector
        self.decomposition_ = decomposition
        self.classifier_ = classifier
        self.knn_normalization_ = self.knn_normalization if self.classifier == "knn" else None
        self.selected_features_ = selected_features
        self.n_features_selected_ = selected_features.size
        self.selected_components_ = decomposition.components_.copy()
        self.components_ = components
        self.n_components_ = int(self.n_components)
        self.n_iter_ = decomposition.n_iter_
        self.factorization_reconstruction_err_ = decomposition.reconstruction_err_
        self.reconstruction_err_ = (reconstruction_error(X_selected, W, decomposition.components_)
                                    if training_coordinates_mode == "transform"
                                    else decomposition.reconstruction_err_)
        self.classifier_convergence_warnings_ = classifier_warnings
        self.classifier_converged_ = not classifier_warnings
        self.classifier_n_iter_ = classifier_n_iter
        self.factorization_convergence_warnings_ = fitting_warnings
        self.factorization_converged_ = not fitting_warnings
        self.training_projection_convergence_warnings_ = projection_warnings
        self.training_coordinates_mode_ = training_coordinates_mode
        self.training_projection_performed_ = training_coordinates_mode == "transform"
        self.training_projection_converged_ = (not projection_warnings
                                              if self.training_projection_performed_ else None)
        self.convergence_warnings_ = fitting_warnings + projection_warnings + classifier_warnings
        self.converged_ = not self.convergence_warnings_
        self.training_coordinates_ = W.copy()
        self.fit_times_ = {"screening": screen_seconds, "factorization": factorization_seconds,
                           "training_projection": training_projection_seconds, "classifier": classifier_seconds,
                           "total": time.perf_counter() - start}
        return self

    def _projection_settings(self):
        settings = dict(
            projection_solver=getattr(self, "projection_solver", "scipy"),
            projection_workers=getattr(self, "projection_workers", 1),
            projection_batch_size=getattr(self, "projection_batch_size", 256),
            projection_native_threads=getattr(self, "projection_native_threads", 1))
        validate_projection_parameters(**settings)
        return settings

    def transform(self, X):
        check_is_fitted(self, "classifier_")
        start = time.perf_counter()
        self.last_transform_converged_ = False
        self.last_transform_warnings_ = ()
        self.last_transform_error_ = None
        try:
            projection_settings = self._projection_settings()
            if isinstance(self.decomposition_, FeatureSPA):
                self.decomposition_.set_params(**projection_settings)
            X = _nonnegative_array(X)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}.")
            X_selected = X[:, self.selected_features_]
            W, messages = _record_warnings(lambda: self.decomposition_.transform(X_selected))
            self.last_transform_warnings_ = messages
            self.last_transform_converged_ = not messages
            return W
        except Exception as error:
            self.last_transform_error_ = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.last_transform_seconds_ = time.perf_counter() - start

    def predict(self, X):
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self.transform(X))
        if getattr(self, "decision_threshold_", None) is None:
            return head.predict(coordinates)
        return self._labels_from_probabilities(head.predict_proba(coordinates))

    def _head_input(self, coordinates):
        """Apply a fitted head scaler once, without changing the NMF map."""
        if isinstance(self.classifier_, Pipeline):
            return self.classifier_.steps[-1][1], self.classifier_[:-1].transform(coordinates)
        return self.classifier_, coordinates

    def _labels_from_probabilities(self, probabilities):
        """Apply an explicit fitted threshold, including to shared KNN scores."""
        positive_index = self.positive_class_index_
        indices = np.where(probabilities[:, positive_index] > self.decision_threshold_,
                           positive_index, 1 - positive_index)
        return self.classes_[indices]

    @available_if(_head_supports("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(self.transform(X))

    @available_if(_head_supports("predict_proba"))
    def predict_with_probabilities(self, X):
        """Return labels and class probabilities with one fixed-basis projection.

        With no explicit threshold, KNN ties use the first class in ``classes_``
        and logistic labels use sklearn's decision rule, including when a small
        nonzero decision rounds to a probability of one half. An explicit
        fitted threshold uses the same strict probability rule as ``predict``.
        This avoids repeating expensive projection, and KNN neighbor searches, when
        benchmark metrics require both predicted labels and probabilities.
        """
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self.transform(X))
        probabilities = head.predict_proba(coordinates)
        if getattr(self, "decision_threshold_", None) is not None:
            labels = self._labels_from_probabilities(probabilities)
        elif isinstance(head, KNeighborsClassifier):
            labels = self.classes_[probabilities.argmax(axis=1)]
        else:
            labels = head.predict(coordinates)
        return labels, probabilities

    @available_if(_head_supports("decision_function"))
    def decision_function(self, X):
        """Return genuine margins toward classes_[1], with one projection."""
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self.transform(X))
        return head.decision_function(coordinates)

    @available_if(_head_supports("decision_function"))
    def predict_with_decision_function(self, X):
        """Return labels and real margins with one projection and head scaling.

        Logistic labels still obey an explicit fitted probability threshold;
        the margin retains sklearn's classes_[1] orientation independently.
        Linear SVM uses its strict positive-margin rule; exact zero selects
        ``classes_[0]``.
        """
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self.transform(X))
        margins = head.decision_function(coordinates)
        if getattr(self, "decision_threshold_", None) is not None:
            labels = self._labels_from_probabilities(head.predict_proba(coordinates))
        else:
            # Both supported binary margin heads use sklearn's strict > 0
            # rule. Reuse this score instead of multiplying by the head twice.
            labels = self.classes_[(margins > 0).astype(np.intp)]
        return labels, margins

    def get_support(self, indices=False):
        check_is_fitted(self, "classifier_")
        if indices:
            return self.selected_features_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_features_] = True
        return support
