"""Screening -> NMF -> KNN, based on Chang Cheng's original scripts."""

from numbers import Integral, Real
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from .decomposition import FeatureSPA
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


class NMFClassifier(ClassifierMixin, BaseEstimator):
    """Binary supervised NMF classifier with an overridable practical default.

    ``solver='spa'`` and ``screening='absolute'`` are the practical default from
    repeated CelebA accuracy/cost comparisons and the author's preference for
    absolute screening among comparably accurate methods. Choose ``solver='cd'``
    or ``solver='mu'``, or screening ``'percentile'``/``'none'``, explicitly to use
    another method.
    Rank, screening level and neighbor count still require data-specific tuning.
    Coordinate descent uses sklearn NMF with NNDSVDa initialization and Frobenius
    loss, matching the original scripts. Multiplicative updates use the same
    initialization and loss with sklearn's MU solver. Raw nonnegative feature
    values and Euclidean KNN coordinates are not automatically scaled.

    ``training_coordinates='fit_transform'`` reproduces the scripts' convention;
    ``'transform'`` instead reprojects training data against the fitted basis
    using the same operation as prediction. No operation on test data refits
    the screening or basis. ``components_`` is padded with zeros to the original
    feature count; ``selected_components_`` contains only retained columns.
    Fit times exclude tuning; callers must separately record search costs.
    The default iteration cap is 20,000 for CD/MU or each SPA NNLS solve;
    converged routines can terminate earlier. CD and MU use their respective
    sklearn stopping criteria, so identical tolerances do not imply identical
    numerical accuracy. SPA's NNLS routine does not use ``tol``. Increasing the
    cap can change a previously truncated fit without changing its objective
    or stopping tolerance.
    """

    def __init__(self, n_components=2, *, solver="spa", screening="absolute", percentile=10.0,
                 score_threshold=2.0, n_neighbors=5, weights="uniform", max_iter=20000,
                 tol=1e-4, random_state=None, training_coordinates="fit_transform"):
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

    def fit(self, X, y):
        _clear_fitted_attributes(self)
        start = time.perf_counter()
        if self.solver not in {"cd", "spa", "mu"}:
            raise ValueError("solver must be 'cd', 'spa', or 'mu'.")
        if self.screening not in {"percentile", "absolute", "none"}:
            raise ValueError("screening must be 'percentile', 'absolute', or 'none'.")
        if self.training_coordinates not in {"fit_transform", "transform"}:
            raise ValueError("training_coordinates must be 'fit_transform' or 'transform'.")
        X, y, classes, counts = _binary_data(X, y)
        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, Integral) or not 1 <= self.n_neighbors <= X.shape[0]:
            raise ValueError("n_neighbors must be a positive integer no larger than the training sample count.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.tol, Real) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        stage = time.perf_counter()
        if self.screening == "none":
            selector = None
            selected_features = np.arange(X.shape[1])
            X_selected = X
        else:
            selector = BinaryScreening(self.screening, self.percentile, self.score_threshold)
            X_selected = selector.fit_transform(X, y)
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
            decomposition = FeatureSPA(n_components=self.n_components, nnls_max_iter=self.max_iter)
        stage = time.perf_counter()
        W, fitting_warnings = _record_warnings(lambda: decomposition.fit_transform(X_selected))
        factorization_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        projection_warnings = ()
        if self.training_coordinates == "transform":
            W, projection_warnings = _record_warnings(lambda: decomposition.transform(X_selected))
        training_projection_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        knn.fit(W, y)
        classifier_seconds = time.perf_counter() - stage
        components = np.zeros((self.n_components, X.shape[1]), dtype=np.float64)
        components[:, selected_features] = decomposition.components_
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.classes_ = classes
        self.class_counts_ = counts
        self.selector_ = selector
        self.decomposition_ = decomposition
        self.classifier_ = knn
        self.selected_features_ = selected_features
        self.n_features_selected_ = selected_features.size
        self.selected_components_ = decomposition.components_.copy()
        self.components_ = components
        self.n_components_ = int(self.n_components)
        self.n_iter_ = decomposition.n_iter_
        self.factorization_reconstruction_err_ = decomposition.reconstruction_err_
        self.reconstruction_err_ = (float(np.linalg.norm(X_selected - W @ decomposition.components_))
                                    if self.training_coordinates == "transform"
                                    else decomposition.reconstruction_err_)
        self.convergence_warnings_ = fitting_warnings + projection_warnings
        self.converged_ = not self.convergence_warnings_
        self.training_coordinates_ = W.copy()
        self.fit_times_ = {"screening": screen_seconds, "factorization": factorization_seconds,
                           "training_projection": training_projection_seconds, "classifier": classifier_seconds,
                           "total": time.perf_counter() - start}
        return self

    def transform(self, X):
        check_is_fitted(self, "classifier_")
        X = _nonnegative_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}.")
        X_selected = X[:, self.selected_features_]
        start = time.perf_counter()
        W, messages = _record_warnings(lambda: self.decomposition_.transform(X_selected))
        self.last_transform_seconds_ = time.perf_counter() - start
        self.last_transform_warnings_ = messages
        self.last_transform_converged_ = not messages
        return W

    def predict(self, X):
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(self.transform(X))

    def predict_proba(self, X):
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(self.transform(X))

    def predict_with_probabilities(self, X):
        """Return labels and class probabilities with one fixed-basis projection.

        KNN ties use the first class in ``classes_``, as in sklearn's predict.
        This avoids repeating expensive projection and neighbor searches when
        benchmark metrics require both predicted labels and probabilities.
        """
        check_is_fitted(self, "classifier_")
        probabilities = self.classifier_.predict_proba(self.transform(X))
        labels = self.classes_[probabilities.argmax(axis=1)]
        return labels, probabilities

    def get_support(self, indices=False):
        check_is_fitted(self, "classifier_")
        if indices:
            return self.selected_features_.copy()
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.selected_features_] = True
        return support
