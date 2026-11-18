"""Screening and NMF classification, based on Chang Cheng's original scripts."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from numbers import Integral, Real
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from .decomposition import FeatureSPA
from ._labels import _sklearn_labels, _validate_class_label, _validate_class_labels
from ._mu import transform_mu
from ._mu_rowwise import transform_mu_rowwise
from ._projection import validate_projection_parameters
from ._residual import reconstruction_error
from .screening import BinaryScreening, _binary_data, _clear_fitted_attributes, _nonnegative_array


_projection_record = ContextVar("supervised_nmf_projection_record", default=None)


def _uses_projection_diagnostics(estimator):
    """Leave custom transform implementations on their own diagnostic contract."""
    return (isinstance(estimator, NMFClassifier)
            and getattr(getattr(estimator, "transform", None), "__func__", None)
            is NMFClassifier.transform)


@contextmanager
def _capture_projection(estimator):
    """Collect a standard estimator's projection evidence for one invocation.

    The record belongs to the caller, never the fitted estimator. Context-local
    storage preserves ordinary method delegation and nested call isolation; it
    does not make Python warning capture or native thread controls thread-safe.
    """
    if not _uses_projection_diagnostics(estimator):
        yield None
        return
    record = dict(last_transform_performed=False, last_transform_converged=None,
                  last_transform_warnings=(), last_transform_error=None,
                  last_transform_seconds=0.0)
    token = _projection_record.set((estimator, record))
    try:
        yield record
    finally:
        _projection_record.reset(token)


def _record_warnings(operation, *, diagnostics=None):
    caught = []
    succeeded = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = operation()
        succeeded = True
    finally:
        if diagnostics is not None:
            diagnostics["last_transform_warnings"] = tuple(
                str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning))
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


def _knn_probabilities_preserve_labels(head, coordinates, probabilities):
    """Reuse stock KNN scores only away from uncertain weighted-vote ties.

    Callers own head-method qualification and their native fallback operation.
    Probability accumulation and native weighted voting can round differently.
    The conservative nonnegative-summation bound admits separated votes;
    uncertain votes require native prediction on the whole original batch.
    Float32 query or training geometry uses a float32 error bound.
    """
    if head.weights == "uniform":
        return True
    if head.weights != "distance":
        return False
    train_dtype = getattr(getattr(head, "_fit_X", None), "dtype", None)
    single = coordinates.dtype == np.float32 or train_dtype == np.float32
    epsilon = np.finfo(np.float32 if single else np.float64).eps
    product = (head.n_neighbors + len(head.classes_) + 4) * epsilon
    guard = 32.0 * product / (1.0 - product) if product < .25 else np.inf
    largest = np.sort(probabilities, axis=1)[:, -2:]
    return (np.isfinite(probabilities).all()
            and not (probabilities < 0).any() and not (probabilities > 1).any()
            and not (largest[:, -1] - largest[:, 0] <= guard).any())


class NMFClassifier(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Binary supervised NMF classifier with an overridable practical default.

    ``solver='spa'`` and ``screening='absolute'`` are practical starting defaults
    supported by the method, numerical and workload reassessment. They are not
    universal accuracy or timing winners. Choose ``solver='cd'``
    or ``solver='mu'``, or screening ``'percentile'``/``'none'``, explicitly to use
    another method.
    Rank, screening level and final classifier parameters still require
    data-specific tuning.
    Percentile screening expresses a relative feature budget; absolute
    screening uses a pooled-t cutoff with a data-dependent retained count.
    Use ``screening='none'`` for an unscreened comparison or to retain every
    supplied feature, including constants. Screening cutoffs are then unused;
    data, binary-label and factorization validation still apply. More features
    can increase downstream work and retain irrelevant information. Tune the
    unscreened representation independently using training data.
    Coordinate descent uses sklearn NMF with NNDSVDa initialization and Frobenius
    loss, matching the original scripts. Multiplicative updates use the same
    initialization and loss with sklearn's MU solver. Raw nonnegative feature
    values and Euclidean KNN coordinates are not automatically scaled.
    A fixed ``random_state`` does not guarantee identical CD/MU factors or
    predictions across numerical-library builds, backends or thread settings.
    For CD/MU, accepted float32 factorization inputs remain float32. Using float64
    for both training and prediction can improve agreement at additional cost, without
    guaranteeing identical results or better classification accuracy.
    When query and fitted-basis dtypes differ, projection uses float64 without
    narrowing either array. A temporary estimator holds any widened basis;
    the fitted basis and training coordinates retain their original values and
    dtypes. Matching dtypes retain their output dtype; CD and batch MU keep
    their existing calculation. Mixed-dtype
    coordinates can differ from those obtained by explicitly downcasting a query.

    ``classifier='knn'`` retains the original final classifier.
    ``knn_normalization='none'`` preserves its original metric. Optional ``'sd'``
    divides each coordinate by its training-fitted standard deviation without
    centering, using ``StandardScaler(with_mean=False)``. Constant/nearconstant
    coordinates follow sklearn's scale fallback. This changes the KNN metric;
    it does not change ``transform`` or ``training_coordinates_``. The fitted
    scaler and KNN are stored in a ``classifier_`` pipeline with steps ``'scale'``
    and ``'knn'``. This option is validated but unused by the other heads.

    KNN ``weights='uniform'`` gives selected neighbors equal votes; optional
    ``'distance'`` uses inverse distances when nearer observations should have
    more influence. If selected neighbors include zero-distance matches, those
    matches vote equally and other selected neighbors have zero weight.
    This can change probabilities and labels and emphasize very close noisy
    points; compare weighting and coordinate metrics on training-only CV.
    ``None`` also means uniform weighting. A callable receives distances and
    returns same-shaped weights; it can forgo shared-neighbor acceleration
    and must be serializable when saving the model. These choices affect KNN
    only and take effect on refit.

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
    projection (and training reprojection when requested). The default ``'auto'``
    uses guarded batching. Optional ``'qr'`` shares a QR reduction and solves
    the reduced NNLS problems with public SciPy, preserving the original
    objective and leaving SPA basis fitting unchanged. It retains range,
    conditioning and original-space qualification guards with public fallback;
    its cost depends on the workload. See ``FeatureSPA`` for the numerical
    limits, iteration-cap semantics and process-pool behavior. These controls are
    validated but unused by CD and MU. Unlike the fitted decision policy,
    projection execution settings may be changed without refitting.

    All other constructor parameter changes take effect on the next fit,
    including ``max_iter``, ``tol``, head settings and KNN normalization.
    Prediction uses the saved feature mask, decomposition and classifier;
    ``get_params()`` can describe settings queued for a future fit. Saving and
    loading with the same implementation and dependencies preserves both the
    constructor settings and fitted objects and
    does not perform a pending refit. This excludes direct mutation of fitted
    internals or stateful custom callables. Standalone ``FeatureSPA`` differs:
    its ``nnls_max_iter`` also affects subsequent transforms without refitting.

    ``mu_projection='auto'`` (the default) resolves to ``'rowwise'`` for MU
    and no MU projection policy for SPA/CD. Explicit ``'batch'`` retains MU's
    shared query-batch stopping rule. ``'rowwise'`` gives each observation its own initialization and
    stopping decision, with float64 working arithmetic for ordinary projection and guarded numerical
    recovery. It preserves the fixed-basis Frobenius objective and projection
    output dtype, but changes numerical coordinates and can cost more time.
    Choose it when observations may be predicted individually or in different
    batches. It removes shared stopping, not every possible floating-point
    difference across batch shapes. ``mu_projection_`` records the fitted
    policy for MU and is ``None`` for other solvers; ``'rowwise'`` requires MU.
    This policy changes only on refit. Current-format persistence retains the
    fitted policy. Joint factorization is unchanged. Explicitly combine rowwise
    projection with ``training_coordinates='transform'`` to use the same
    projection policy for classifier-training and query coordinates.

    ``training_coordinates='auto'`` uses fixed-basis ``'transform'`` coordinates
    for SPA and the historical ``'fit_transform'`` coordinates for CD and MU.
    Explicit ``'fit_transform'`` also uses fixed-basis coordinates for SPA;
    CD and MU retain their factorization coordinates in that mode.
    ``'transform'`` reprojects training data against the fitted basis using the
    same operation as prediction. ``training_coordinates_mode_`` records the
    resolved fitted choice; ``training_coordinates_`` holds the actual array.
    Changing this parameter takes effect on the next fit, not prediction.

    ``fit_transform(X, y)`` fits the complete classifier and returns projected
    training coordinates. For the standard SPA classifier it returns an
    independent copy of the coordinates already computed during fit, avoiding
    a second NNLS projection. Other solvers and custom transform implementations
    call ``transform(X)`` after fitting. Labels are required. These coordinates can
    differ from ``training_coordinates_`` when the head uses CD/MU factorization
    coordinates. The method does not change the training-coordinate
    policy. Only paths that call ``transform`` perform an additional projection.

    No operation on test data refits
    the screening or basis. ``components_`` is padded with zeros to the original
    feature count; ``selected_components_`` contains only retained columns.
    Fit times exclude tuning; callers must separately record search costs.
    The default iteration cap is 20,000 for CD/MU and SPA's public NNLS solves.
    SPA may first make a bounded batched attempt, then pass the original cap
    to SciPy fallback; this is not a shared total-work budget. Batched pivot
    rounds and SciPy iterations are different units. Converged routines can
    terminate earlier. CD uses sklearn's stopping criterion for fitting and
    projection. MU uses sklearn's shared stopping criterion for fitting and
    explicit batch projection; default rowwise MU projection checks residual
    improvement per observation.
    Identical tolerances do not imply identical
    numerical accuracy. SPA's NNLS routine does not use ``tol``. Increasing the
    cap can change a previously truncated fit without changing its objective
    or stopping tolerance.

    ``n_iter_`` records CD/MU factorization iterations or SPA's anchor count,
    not prediction or NNLS iterations. ``classifier_n_iter_`` records
    logistic regression/SVM iterations, or is ``None`` for KNN. ``converged_`` and
    ``convergence_warnings_`` include factorization, training projection, and
    classifier fitting; classifier-specific diagnostics are also available as
    ``classifier_converged_`` and ``classifier_convergence_warnings_``.
    For stages that return successfully, these flags indicate the absence of
    a captured sklearn ``ConvergenceWarning``; other warning categories do not
    determine them. They do not independently certify stationarity, KKT
    conditions, global optimality or agreement across environments. A skipped
    training projection has ``training_projection_converged_=None``. For an
    already fitted estimator, prediction leaves learned parameters and
    model-owned inference state unchanged. Eligible explicit QR calls may populate
    a transient external weak cache for the shared basis; its factors are not
    persisted with the estimator. Evaluation reports contain call-local
    projection diagnostics; ``last_transform_*`` attributes are no longer written
    on the model. A failed prediction does not change the fit's ``converged_``.
    Third-party query counters may still change. Use separate processes for
    concurrent evaluations requiring warning isolation; warning capture and
    native thread controls do not provide a general thread-safety guarantee.
    """

    def __init__(self, n_components=2, *, solver="spa", screening="absolute", percentile=10.0,
                 score_threshold=2.0, n_neighbors=5, weights="uniform", max_iter=20000,
                 tol=1e-4, random_state=None, training_coordinates="auto",
                 mu_projection="auto",
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
        self.mu_projection = mu_projection
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        return self._fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the classifier and return its fixed-basis training representation.

        Standard SPA fits already compute that representation. Return a copy
        without projecting again; preserve fit/transform delegation for other
        solvers, subclasses and overridden transform methods.
        """
        if (type(self) is not NMFClassifier or self.solver != "spa"
                or getattr(self.transform, "__func__", None) is not NMFClassifier.transform):
            return super().fit_transform(X, y, **fit_params)
        # Match TransformerMixin's label and fit-parameter delegation, including
        # the existing TypeError for missing labels or unsupported parameters.
        if y is None:
            self.fit(X, **fit_params)
        else:
            self.fit(X, y, **fit_params)
        if (type(self) is NMFClassifier
                and getattr(self.transform, "__func__", None) is NMFClassifier.transform
                and type(self.decomposition_) is FeatureSPA
                and self.training_coordinates_mode_ == "transform"):
            return self.training_coordinates_.copy()
        return self.transform(X)

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
        threshold = self.decision_threshold
        if threshold is not None and (
                isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
                or not np.isfinite(threshold)
                or (threshold < 0 and threshold != np.nextafter(0.0, -np.inf))
                or threshold > 1.0):
            raise ValueError("decision_threshold must be None, finite in [0, 1], or np.nextafter(0.0, -np.inf).")
        projection_settings = self._projection_settings()
        if self.solver not in {"cd", "spa", "mu"}:
            raise ValueError("solver must be 'cd', 'spa', or 'mu'.")
        if not isinstance(self.mu_projection, str) or self.mu_projection not in {"auto", "batch", "rowwise"}:
            raise ValueError("mu_projection must be 'auto', 'batch', or 'rowwise'.")
        if self.mu_projection == "rowwise" and self.solver != "mu":
            raise ValueError("mu_projection='rowwise' requires solver='mu'.")
        mu_projection_mode = None
        if self.solver == "mu":
            mu_projection_mode = "rowwise" if self.mu_projection == "auto" else self.mu_projection
        if self.screening not in {"percentile", "absolute", "none"}:
            raise ValueError("screening must be 'percentile', 'absolute', or 'none'.")
        if (not isinstance(self.training_coordinates, str)
                or self.training_coordinates not in {"auto", "fit_transform", "transform"}):
            raise ValueError("training_coordinates must be 'auto', 'fit_transform', or 'transform'.")
        training_coordinates_mode = self.training_coordinates
        if self.solver == "spa":
            # FeatureSPA.fit_transform uses the same fixed-basis NNLS map.
            # Resolve all public SPA modes to that map and account for its
            # one training projection separately from anchor/basis fitting.
            training_coordinates_mode = "transform"
        elif training_coordinates_mode == "auto":
            training_coordinates_mode = "fit_transform"
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
        positive_label = self.positive_label
        positive_index = 1
        if positive_label is not None:
            if not np.isscalar(positive_label):
                raise ValueError("positive_label must be one of the two training classes.")
            _validate_class_label(positive_label)
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
        if self.solver == "spa":
            # Calling fit_transform here would already project training data.
            # Fit only the anchors/basis, then project once in the stage below.
            _, fitting_warnings = _record_warnings(lambda: decomposition.fit(X_selected))
        else:
            W, fitting_warnings = _record_warnings(lambda: decomposition.fit_transform(X_selected))
        factorization_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        projection_warnings = ()
        if training_coordinates_mode == "transform":
            projection_transform = transform_mu_rowwise if mu_projection_mode == "rowwise" else transform_mu
            W, projection_warnings = _record_warnings(lambda: projection_transform(decomposition, X_selected))
        training_projection_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        y = _sklearn_labels(y)
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
        self.mu_projection_ = mu_projection_mode
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
            projection_solver=self.projection_solver,
            projection_workers=self.projection_workers,
            projection_batch_size=self.projection_batch_size,
            projection_native_threads=self.projection_native_threads)
        validate_projection_parameters(**settings)
        return settings

    def _validate_fitted_labels(self):
        """Check saved label values without consulting queued constructor policy."""
        for classes in (getattr(self, "classes_", None),
                        getattr(self.classifier_, "classes_", None)):
            if classes is not None:
                _validate_class_labels(classes)
        _validate_class_label(getattr(self, "positive_label_", None))

    def transform(self, X):
        check_is_fitted(self, "classifier_")
        active = _projection_record.get()
        diagnostics = active[1] if active is not None and active[0] is self else None
        if diagnostics is not None:
            start = time.perf_counter()
            diagnostics.update(last_transform_performed=True, last_transform_converged=False,
                               last_transform_warnings=(), last_transform_error=None)
        try:
            projection_settings = self._projection_settings()
            decomposition = self.decomposition_
            # SPA execution controls and sklearn NMF bookkeeping are call-local.
            # Fitted arrays stay shared unless the existing dtype path widens them.
            if isinstance(decomposition, (FeatureSPA, NMF)):
                decomposition = copy(decomposition)
            if isinstance(decomposition, FeatureSPA):
                decomposition.set_params(**projection_settings)
            X = _nonnegative_array(X)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                    f"{self.n_features_in_} features as input."
                )
            self._validate_fitted_labels()
            X_selected = X[:, self.selected_features_]
            if type(decomposition) is NMF and "transform" not in vars(decomposition):
                basis = decomposition.components_
                if (type(basis) in (np.ndarray, np.memmap)
                        and basis.dtype in (np.float32, np.float64)
                        and X_selected.dtype != basis.dtype):
                    # Widening preserves stored values, including positive queries
                    # outside float32's range. Never downcast or refit the basis.
                    X_selected = X_selected.astype(np.float64, copy=False)
                    if basis.dtype != np.float64:
                        decomposition.components_ = basis.astype(np.float64)
            projection_transform = (transform_mu_rowwise
                                    if self.mu_projection_ == "rowwise"
                                    else transform_mu)
            W, messages = _record_warnings(
                lambda: projection_transform(decomposition, X_selected), diagnostics=diagnostics)
            if diagnostics is not None:
                diagnostics["last_transform_converged"] = not messages
            return W
        except Exception as error:
            if diagnostics is not None:
                diagnostics["last_transform_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if diagnostics is not None:
                diagnostics["last_transform_seconds"] = time.perf_counter() - start

    def predict(self, X):
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self._prediction_coordinates(X))
        if self.decision_threshold_ is None:
            return head.predict(coordinates)
        return self._labels_from_probabilities(head.predict_proba(coordinates))

    def score(self, X, y, sample_weight=None):
        """Return sklearn accuracy while rejecting unsupported target labels."""
        from sklearn.metrics import accuracy_score

        predicted = self.predict(X)
        check_consistent_length(y, predicted, sample_weight)
        y = _validate_class_labels(y)
        return accuracy_score(_sklearn_labels(y), _sklearn_labels(predicted), sample_weight=sample_weight)

    def _prediction_coordinates(self, X):
        coordinates = self.transform(X)
        if getattr(self.transform, "__func__", None) is not NMFClassifier.transform:
            # Standard transform already checks saved labels after X validation.
            # Preserve custom transform delegation while guarding its consumers.
            self._validate_fitted_labels()
        return coordinates

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
        return self.classifier_.predict_proba(self._prediction_coordinates(X))

    @available_if(_head_supports("predict_proba"))
    def predict_with_probabilities(self, X):
        """Return labels and class probabilities with one fixed-basis projection.

        With no explicit threshold, KNN ties use the first class in ``classes_``
        and logistic labels use sklearn's decision rule, including when a small
        nonzero decision rounds to a probability of one half. An explicit
        fitted threshold uses the same strict probability rule as ``predict``.
        This avoids repeating expensive projection and normally reuses KNN
        probabilities for labels. Near weighted-vote ties use the native label
        calculation because its floating-point reduction can differ; callable
        weights also retain their native decision.
        """
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self._prediction_coordinates(X))
        probabilities = head.predict_proba(coordinates)
        if self.decision_threshold_ is not None:
            labels = self._labels_from_probabilities(probabilities)
        elif (type(head) is KNeighborsClassifier
              and _knn_probabilities_preserve_labels(head, coordinates, probabilities)):
            labels = self.classes_[probabilities.argmax(axis=1)]
        else:
            labels = head.predict(coordinates)
        return labels, probabilities

    @available_if(_head_supports("decision_function"))
    def decision_function(self, X):
        """Return genuine margins toward classes_[1], with one projection."""
        check_is_fitted(self, "classifier_")
        head, coordinates = self._head_input(self._prediction_coordinates(X))
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
        head, coordinates = self._head_input(self._prediction_coordinates(X))
        margins = head.decision_function(coordinates)
        if self.decision_threshold_ is not None:
            labels = self._labels_from_probabilities(head.predict_proba(coordinates))
        else:
            # Both supported binary margin heads use sklearn's strict > 0
            # rule. Reuse this score instead of multiplying by the head twice.
            labels = self.classes_[(margins > 0).astype(np.intp)]
        return labels, margins

    def get_support(self, indices=False):
        check_is_fitted(self, "classifier_")
        self._validate_fitted_labels()
        if indices:
            return self.selected_features_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_features_] = True
        return support
