"""Feature-anchor SPA comparison for Chang Cheng's supervised NMF package."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg.blas import dgemv, dger
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._feature_names import _check_input_features, _input_feature_names
from .screening import _clear_fitted_attributes, _nonnegative_array
from ._nnls import project_nnls
from ._projection import project_rows, validate_projection_parameters
from ._residual import reconstruction_error


def _fit_basis(coordinates, X, maxiter):
    """Fit the fixed-anchor NNLS basis independently of projection settings."""
    try:
        rows = project_nnls(coordinates, X.T, solver="auto", maxiter=maxiter,
                            batch_size=256)
    except FloatingPointError:
        # The guarded helper has a bounded numerical domain. Preserve inputs
        # accepted by the original basis solver by retrying the public loop.
        # Validation and iteration-exhaustion errors must still propagate.
        return np.column_stack([
            nnls(coordinates, X[:, j], maxiter=maxiter)[0]
            for j in range(X.shape[1])
        ])
    return np.ascontiguousarray(rows.T)


class FeatureSPA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Successive Projection Algorithm with feature columns as anchors.

    Rows are observations and columns are features. With ``normalize=True``,
    training columns are L1-normalized *only during anchor selection*. For an
    exact feature-separable nonnegative matrix this makes each normalized
    column a convex combination of normalized anchor columns. Zero columns
    remain zero. There is no per-observation normalization or learned rescaling
    of incoming observations. With normalize=False the unnormalized SPA model
    needs the usual bound on the sums of mixing coefficients.

    ``n_components=2`` selects the number of anchors. Larger ranks can retain
    more structure but require more anchor-selection and NNLS work; the data
    must supply enough independent anchors. Choose ``normalize=False`` when
    selection by original feature magnitudes is intended: large-scale columns
    can then dominate. Neither setting rescales the saved anchor coordinates
    or incoming observations. The unnormalized recovery assumptions still apply.

    ``condition_limit=1e12`` rejects poorly conditioned selected anchors
    (both selection-scale and original-scale) and the fitted basis. A smaller
    finite limit greater than one imposes a stricter admission policy; raising
    it admits more ill-conditioned models without improving their accuracy.
    This is separate from the stricter batched and QR acceleration guards.

    ``nnls_max_iter=None`` uses the reference solver's default, currently three
    times the rank. A positive integer sets the cap for basis fitting and
    projection. Increase it when a solve exhausts its iteration allowance;
    larger caps allow more work, not a different objective or an accuracy target.
    A bounded batched attempt may precede SciPy fallback with the original cap,
    so this is not a shared total-work budget. Changing ``nnls_max_iter`` or
    projection execution controls affects subsequent transforms without refit.
    Changes to rank, normalization or condition admission require a new fit;
    they do not modify or requalify the existing basis.

    The selected original-scale columns form the anchor matrix A. We solve
    NNLS for every column to learn H such that X approximately equals A H.
    ``transform`` solves NNLS against this fixed H, without selecting anchors
    or refitting on new observations. ``fit_transform`` fits this basis and
    applies the same NNLS transformation to the training observations.

    A generic sklearn Pipeline therefore uses one fixed-basis coordinate map
    for training and prediction. ``fit_coordinates_`` retains the original
    anchor matrix A, which can differ from projected training coordinates on
    noisy data. ``reconstruction_err_`` measures the anchor-factorization
    residual ||X - A H||, not the residual after training projection. Refit all
    learned steps inside each training fold when performing cross-validation.

    This is a separate estimator from coordinate-descent NMF. Its recovery
    theory requires feature separability, a full-rank anchor matrix and noise
    conditions; success on a classification example does not establish them.
    References: Gillis and Vavasis (2014), doi:10.1109/TPAMI.2013.226.
    SPA/NNLS computations use float64, including when inputs are float32.

    Basis fitting uses guarded batched NNLS for suitable designs and at least
    32 feature columns, with public SciPy fallback. Its normal-equation path
    requires condition number at most 1e4 and qualified nonnegative solutions.
    A numerical-range rejection retries the original public column-wise solve.
    Basis fitting stays in the calling process, independently of projection
    controls. Batched pivot rounds and SciPy iterations are different units;
    fallback receives the original ``nnls_max_iter`` after the bounded attempt.

    ``projection_solver='auto'`` uses guarded batched NNLS for suitable query
    batches and public SciPy NNLS otherwise. ``'scipy'`` selects the legacy
    row-wise solver; ``'batched'`` attempts batching even for small inputs,
    with the same numerical guards and fallback. Optional ``'qr'`` shares an
    economic QR reduction across at least 32 right-hand sides when the design
    height is at least twice the rank and condition(R) <= 1e6. It solves the
    signed reduced problems with public SciPy NNLS, then qualifies coefficients
    against the original nonnegative design and targets. Orthogonal reduction
    preserves the fixed-basis NNLS minimizer in exact arithmetic; floating-point
    coefficients and execution time can differ. The current reversible scaling
    checks apply, and failed or ineligible QR attempts use the current solver
    with its public SciPy fallback. Small explicit caps keep the direct SciPy
    route; fallback receives the original cap and propagates reference errors.
    Q/R are local to a call, not cached in the estimator. These settings affect
    projection only; the default and basis-fitting policy are unchanged.
    ``projection_batch_size`` bounds rows per numerical block. Optional
    ``projection_workers>1`` uses reusable process workers for at least 512
    rows and two blocks; smaller queries run in the caller. Each projection
    worker uses ``projection_native_threads`` numerical threads. These controls
    are separate from outer cross-validation workers and classifier threads.
    QR admission applies per worker call and scaling group, so chunks with
    fewer than 32 rows can use the current solver despite a large total query.
    """

    def __init__(self, n_components=2, *, normalize=True, condition_limit=1e12, nnls_max_iter=None,
                 projection_solver="auto", projection_workers=1,
                 projection_batch_size=256, projection_native_threads=1):
        self.n_components = n_components
        self.normalize = normalize
        self.condition_limit = condition_limit
        self.nnls_max_iter = nnls_max_iter
        self.projection_solver = projection_solver
        self.projection_workers = projection_workers
        self.projection_batch_size = projection_batch_size
        self.projection_native_threads = projection_native_threads

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        return tags

    def _projection_options(self):
        options = dict(solver=self.projection_solver,
                       workers=self.projection_workers,
                       batch_size=self.projection_batch_size,
                       native_threads=self.projection_native_threads)
        validate_projection_parameters(*options.values())
        return options

    def fit(self, X, y=None):
        _clear_fitted_attributes(self)
        self._projection_options()
        feature_names = _input_feature_names(X)
        X = _nonnegative_array(X).astype(np.float64, copy=False)
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, Integral) or not 1 <= self.n_components <= min(X.shape):
            raise ValueError("n_components must be an integer between 1 and min(n_samples, n_features).")
        if not isinstance(self.normalize, (bool, np.bool_)):
            raise ValueError("normalize must be boolean.")
        if not isinstance(self.condition_limit, Real) or not np.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError("condition_limit must be finite and greater than 1.")
        if self.nnls_max_iter is not None and (isinstance(self.nnls_max_iter, bool) or not isinstance(self.nnls_max_iter, Integral) or self.nnls_max_iter < 1):
            raise ValueError("nnls_max_iter must be a positive integer or None.")
        with np.errstate(over="ignore"):
            scales = X.sum(axis=0) if self.normalize else np.ones(X.shape[1])
        if not np.isfinite(scales).all():
            raise ValueError("Training column sums overflowed; rescale X before fitting SPA.")
        # An underflow handler or a custom rank can edit X during selection.
        # Keep normalized columns when revisiting all of them, or when
        # retaining at most 32 KiB avoids another indexed division.
        ordinary_selection = (type(self) is FeatureSPA and type(self.n_components) is int
                              and np.geterr()["under"] == "ignore")
        if not ordinary_selection or (
            self.normalize and (X.size <= 4096 or self.n_components == X.shape[1])
        ):
            selection = np.divide(X, scales, out=np.zeros_like(X), where=scales > 0)
            residual = selection.copy()
        elif self.normalize:
            selection = None
            residual = np.zeros_like(X, order="C")
            np.divide(X, scales, out=residual, where=scales > 0)
        else:
            selection = X
            residual = X.copy(order="C")
        norms = np.linalg.norm(residual, axis=0)
        initial_norm = norms.max()
        if not np.isfinite(initial_norm) or initial_norm == 0:
            del norms
            raise ValueError("SPA requires nonzero data with finite column norms.")
        if not ordinary_selection:
            del norms
        tolerance = np.finfo(np.float64).eps * max(X.shape) * initial_norm
        anchors = []
        directions = []
        for pivot in range(self.n_components):
            if pivot or not ordinary_selection:
                norms = np.linalg.norm(residual, axis=0)
            norms[anchors] = -1
            index = int(np.argmax(norms))
            if norms[index] <= tolerance:
                raise ValueError("SPA anchor matrix is numerically rank deficient at the requested n_components.")
            direction = residual[:, index] / norms[index]
            # Reorthogonalization keeps finite-precision projection from
            # reselecting an already removed direction in ill-conditioned data.
            for old_direction in directions:
                direction -= old_direction * np.dot(old_direction, direction)
            norm = np.linalg.norm(direction)
            if norm <= np.finfo(float).eps:
                raise ValueError("SPA could not obtain an independent anchor direction.")
            direction /= norm
            anchors.append(index)
            directions.append(direction)
            # residual is an owned C-order workspace. Its transpose is an
            # F-order view, allowing BLAS to update it in place without the
            # full-size outer-product temporary. Keep the original residual
            # layout for column norms and the same reorthogonalized SPA step.
            projection = dgemv(1.0, residual.T, direction)
            residual = dger(-1.0, projection, direction,
                            a=residual.T, overwrite_a=1).T
        anchors = np.asarray(anchors, dtype=int)
        del residual, directions
        if anchors.size:
            del direction, projection, norms
        if len(anchors) > 1:
            del old_direction
        if selection is None:
            selected = X[:, anchors] / scales[anchors]
        else:
            selected = selection[:, anchors]
        anchor_condition = float(np.linalg.cond(selected))
        del selected
        original_anchor_condition = (anchor_condition if selection is X
                                     else float(np.linalg.cond(X[:, anchors])))
        del selection
        if max(anchor_condition, original_anchor_condition) > self.condition_limit:
            raise ValueError("SPA anchors exceed condition_limit; reduce rank or examine feature scaling.")
        coordinates = X[:, anchors].copy()
        components = _fit_basis(coordinates, X, self.nnls_max_iter)
        basis_condition = float(np.linalg.cond(components))
        if not np.isfinite(basis_condition) or basis_condition > self.condition_limit:
            raise ValueError("Fitted SPA basis is rank deficient or exceeds condition_limit.")
        try:
            if feature_names is not None:
                self.feature_names_in_ = feature_names
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = X.shape[0]
            self.n_components_ = int(self.n_components)
            self.anchor_indices_ = anchors
            self.column_scales_ = scales
            self.anchor_condition_ = anchor_condition
            self.original_anchor_condition_ = original_anchor_condition
            self.basis_condition_ = basis_condition
            self.components_ = components
            self.fit_coordinates_ = coordinates
            self.reconstruction_err_ = reconstruction_error(X, coordinates, components)
            self.n_iter_ = self.n_components_
        except Exception:
            _clear_fitted_attributes(self)
            raise
        return self

    @property
    def _n_features_out(self):
        return self.n_components_

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the basis, then project training observations by fixed-basis NNLS.

        This is the same operation as ``fit(X, y).transform(X)``. The anchor
        coordinates remain separately available as ``fit_coordinates_``.
        """
        if fit_params:
            raise TypeError(f"Unsupported fit parameters: {', '.join(fit_params)}")
        return self.fit(X, y).transform(X)

    def transform(self, X):
        """Project observations by NNLS against the fitted basis.

        Training inputs use this projection too; their returned coordinates
        can differ from ``fit_coordinates_``. Projection does not update the
        stored anchor coordinates or their ``reconstruction_err_`` diagnostic.
        """
        check_is_fitted(self, "components_")
        _check_input_features(self, X, reset=False)
        X = _nonnegative_array(X).astype(np.float64, copy=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        # scipy.optimize.nnls requests a C-order design matrix. Prepare the
        # shared fixed basis once, rather than copying its transpose per row.
        # Never overwrite the fitted basis. The guarded batched path falls
        # back to the public solver when numerical qualification fails.
        design = np.ascontiguousarray(self.components_.T)
        if type(self) is FeatureSPA and type(self.projection_solver) is str and self.projection_solver == "qr" and type(self.components_) is np.ndarray:
            return project_rows(design, X, maxiter=self.nnls_max_iter,
                _qr_owner=self.components_, _qr_owned=design.flags.owndata,
                **self._projection_options())
        return project_rows(design, X, maxiter=self.nnls_max_iter,
                            **self._projection_options())

    def inverse_transform(self, W):
        check_is_fitted(self, "components_")
        W = _nonnegative_array(W)
        if W.shape[1] != self.n_components_:
            raise ValueError(f"W must have {self.n_components_} columns.")
        return W @ self.components_
