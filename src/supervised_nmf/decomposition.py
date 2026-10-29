"""Feature-anchor SPA comparison for Chang Cheng's supervised NMF package."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg.blas import dgemv, dger
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

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


class FeatureSPA(TransformerMixin, BaseEstimator):
    """Successive Projection Algorithm with feature columns as anchors.

    Rows are observations and columns are features. With ``normalize=True``,
    training columns are L1-normalized *only during anchor selection*. For an
    exact feature-separable nonnegative matrix this makes each normalized
    column a convex combination of normalized anchor columns. Zero columns
    remain zero. There is no per-observation normalization or learned rescaling
    of incoming observations. With normalize=False the unnormalized SPA model
    needs the usual bound on the sums of mixing coefficients.

    The selected original-scale columns form training coordinates W. We solve
    NNLS for every column to learn H such that X approximately equals W H.
    ``transform`` solves NNLS against this fixed H, without selecting anchors
    or refitting on new observations. ``fit_transform`` returns the anchor
    coordinates W; ``transform(X_train)`` may differ on noisy data.

    A generic sklearn Pipeline therefore trains its next step on anchor
    coordinates and predicts using projected coordinates. For a common SPA
    classification map, use ``NMFClassifier(solver='spa')`` with
    ``training_coordinates='auto'`` or ``'transform'``, or explicitly call
    ``fit`` then ``transform`` on the training data before fitting a separate
    classifier. Refit all learned
    steps inside each training fold when performing cross-validation.

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

    def __setstate__(self, state):
        # Make get_params/clone work for models serialized before these
        # execution controls existed, retaining their original projection solver.
        state = dict(state)
        state.setdefault("projection_solver", "scipy")
        state.setdefault("projection_workers", 1)
        state.setdefault("projection_batch_size", 256)
        state.setdefault("projection_native_threads", 1)
        super().__setstate__(state)

    def _projection_options(self):
        # Older serialized estimators retain their historical projection path.
        options = dict(solver=getattr(self, "projection_solver", "scipy"),
                       workers=getattr(self, "projection_workers", 1),
                       batch_size=getattr(self, "projection_batch_size", 256),
                       native_threads=getattr(self, "projection_native_threads", 1))
        validate_projection_parameters(*options.values())
        return options

    def fit(self, X, y=None):
        _clear_fitted_attributes(self)
        self._projection_options()
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
        scaled = np.divide(X, scales, out=np.zeros_like(X), where=scales > 0)
        residual = scaled.copy()
        initial_norm = np.linalg.norm(residual, axis=0).max()
        if not np.isfinite(initial_norm) or initial_norm == 0:
            raise ValueError("SPA requires nonzero data with finite column norms.")
        tolerance = np.finfo(np.float64).eps * max(X.shape) * initial_norm
        anchors = []
        directions = []
        for _ in range(self.n_components):
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
        anchor_condition = float(np.linalg.cond(scaled[:, anchors]))
        original_anchor_condition = float(np.linalg.cond(X[:, anchors]))
        if max(anchor_condition, original_anchor_condition) > self.condition_limit:
            raise ValueError("SPA anchors exceed condition_limit; reduce rank or examine feature scaling.")
        # Anchor selection and its condition checks are finished. Release the
        # full-size workspaces before allocating coordinates and fitting H;
        # loop variables otherwise keep the final directions alive as well.
        del scaled, residual, directions, direction, projection, norms
        if len(anchors) > 1:
            del old_direction
        coordinates = X[:, anchors].copy()
        components = _fit_basis(coordinates, X, self.nnls_max_iter)
        basis_condition = float(np.linalg.cond(components))
        if not np.isfinite(basis_condition) or basis_condition > self.condition_limit:
            raise ValueError("Fitted SPA basis is rank deficient or exceeds condition_limit.")
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
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the anchor factorization and return a copy of its anchor columns.

        This returns ``fit_coordinates_``, not a fixed-basis NNLS projection.
        It can differ from ``fit(X, y).transform(X)`` on noisy data. A generic
        pipeline passes these anchor coordinates to its next step during fit.
        """
        if fit_params:
            raise TypeError(f"Unsupported fit parameters: {', '.join(fit_params)}")
        return self.fit(X, y).fit_coordinates_.copy()

    def transform(self, X):
        """Project observations by NNLS against the fitted basis.

        Training inputs use this projection too; their returned coordinates
        can differ from ``fit_coordinates_``. Projection does not update the
        stored anchor coordinates or their ``reconstruction_err_`` diagnostic.
        """
        check_is_fitted(self, "components_")
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
        return project_rows(design, X, maxiter=self.nnls_max_iter,
                            **self._projection_options())

    def inverse_transform(self, W):
        check_is_fitted(self, "components_")
        W = _nonnegative_array(W)
        if W.shape[1] != self.n_components_:
            raise ValueError(f"W must have {self.n_components_} columns.")
        return W @ self.components_
