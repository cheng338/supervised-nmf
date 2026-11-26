# API reference

This page describes version 0.6.0, including its solver-aware MU projection default. Per-observation MU
projection was added as an option in 0.4.0; newly fitted MU models now select it
through `mu_projection="auto"`. Version 0.3.3 introduced optional QR
projection and MU numerical recovery. Optional KNN SD
scaling and the linear-SVM head were introduced in 0.3.0. Historical release
and validation records retain their original scope.

## Package imports

`import supervised_nmf` loads the feature-decomposition and screening API and
defers the classifier module until it is requested. Both
`from supervised_nmf import NMFClassifier` and
`from supervised_nmf.classifier import NMFClassifier` return the same class;
`supervised_nmf.NMFClassifier` and `supervised_nmf.classifier` also load it on
first access. This avoids unused classifier setup in projection-only workers.
Numerical capability captures remain eager.

Errors from classifier-only dependencies and classifier definition now occur
on first access rather than on the initial package import. Import
`NMFClassifier` explicitly during application startup if that is where you want
those errors to surface. `dir(supervised_nmf)` advertises the public names;
before first access, the deferred class and module need not be present in
`vars(supervised_nmf)`.

The class's module identity and current-model persistence contract are
unchanged. Pickle/joblib loading can trigger the same deferred import, without
refitting or adding compatibility for older model formats; see
[model persistence](#model-persistence).

## Classifier

### Input columns

`NMFClassifier`, `BinaryScreening` and `FeatureSPA` record
`feature_names_in_` when fitted on a table with unique string column names.
Supply the same columns in the same order for later predictions and
transformations. A renamed, missing, added or reordered feature raises an
error. Duplicate string labels are ambiguous. Rename the columns before
fitting, or pass a numerical array when feature positions are intentional.
Plain arrays and tables with integer-only column labels use positional order.
Switching between named and unnamed input on a fitted estimator follows
scikit-learn's feature-name warning convention.

`BinaryScreening.get_feature_names_out()` returns the selected input names.
A standalone `FeatureSPA().set_output(transform="pandas")` returns coordinates
with the original row index and component names such as `featurespa0`.
The classifier's own coordinate transformations return arrays.

Calibration and comparison checkpoints include the ordered names when
present. Resume with the same schema. On a named search, the report's
`refit_input_container` records how the final estimator was fitted.
For a positional refit, supply features in training order when calling the
returned third-party estimator directly. To check named inputs against the
original positional refit, pass the result itself to
`evaluate_classifier(result, X_test, y_test)`. The result keeps that
calibration's column order. If the training data had no column names,
evaluation with the result reads validation columns by position. Supply
a validation table in training order even when its columns have names.
After independently refitting the estimator,
use its new calibration result or supply an array in the new training order.
The numerical search rebuilds named inputs using an available table library
(pandas, Polars or PyArrow). Positional searches take arrays directly.

```python
from supervised_nmf import NMFClassifier

classifier = NMFClassifier(
    n_components=2,
    solver="spa",                # default; also "mu" or "cd"
    screening="absolute",        # default; also "percentile" or "none"
    score_threshold=2.0,         # absolute pooled-t units, inclusive comparison
    percentile=10.0,             # used only by percentile screening
    classifier="knn",            # also "logistic" or "linear_svm"
    n_neighbors=5,
    weights="uniform",           # equal votes; "distance" or a callable also supported
    knn_normalization="none",    # optional "sd", fitted on training coordinates
    logistic_C=1.0,              # inverse L2 regularization strength
    logistic_max_iter=1000,
    logistic_tol=1e-4,
    logistic_class_weight=None,  # also "balanced" or a class-weight dictionary
    svm_C=1.0,                  # linear SVM only; coordinates remain unscaled
    svm_max_iter=20000,
    svm_tol=1e-5,
    svm_class_weight=None,
    max_iter=20000,
    tol=1e-4,
    random_state=42,
    training_coordinates="auto",  # SPA: transform; CD/MU: fit_transform
    mu_projection="auto",       # MU: "rowwise"; SPA/CD: None; explicit "batch" available
    projection_solver="auto",   # SPA only: "auto", "scipy", "batched", or "qr"
    projection_workers=1,        # maximum process count for large queries
    projection_batch_size=256,
    projection_native_threads=1,
    decision_threshold=None,    # probability cutoff: KNN/logistic only
    positive_label=None,        # thresholded class; does not reverse margins
)
classifier.fit(X_train, y_train)
W_new = classifier.transform(X_new)
y_pred = classifier.predict(X_new)
probabilities = classifier.predict_proba(X_new)
```

The default is SPA with absolute-score screening. Available solvers are
`"spa"`, `"mu"`, and `"cd"`; CD reproduces the original solver choice. The
[four-method development comparison](../research/nmf-solver-comparison.md) supplies empirical
context, not a universal solver ranking. Rank, cutoff and classifier parameters remain
tunable. The current `max_iter=20000` default provides a numerical allowance for
CD/MU iterations and SPA's public NNLS solves; it does not request 20,000 steps
when a routine's stopping condition is reached earlier. The tolerance remains
`1e-4` for CD/MU and is not used by SPA's NNLS routine. CD and MU have different
stopping criteria; equal tolerance values do not establish equal numerical
accuracy. See [iteration limits and qualification](#iteration-limits-and-qualification).
Explicit overrides are preserved; `None` is invalid for `solver` or `screening`.

Use `screening="percentile"` when a relative feature budget is appropriate,
or `"absolute"` when selection should follow a pooled-t cutoff. The latter's
retained count varies with the training data; neither policy guarantees enough
features survive for the requested rank. Choose `screening="none"` for an
unscreened NMF comparison or when all supplied features should reach the
factorization. It retains every column, including constants, and ignores the
screening cutoffs. More columns can increase fitting and projection work and
retain irrelevant features. It still requires dense finite nonnegative data,
binary labels with at least two training observations per class, and valid
factorization inputs. Tune rank and classifier settings for the unscreened
representation independently; a screened model's selected settings need not
transfer. See [screening details](#screening-and-anchor-details).

Choose multiplicative updates with `NMFClassifier(solver="mu", random_state=42)`.
Both MU and CD use scikit-learn NMF with NNDSVDa initialization, unregularized
Frobenius loss, and the existing `max_iter`, `tol`, and `random_state` parameters.
NNDSVDa fills initialization zeros with the data mean, avoiding the zero-locking
initialization of plain NNDSVD under MU. Features and factors retain their
original scales during factorization. The fitted screening mask and basis are
reused for prediction.

`classifier="knn"` with `knn_normalization="none"` preserves Euclidean KNN on
unscaled NMF coordinates. Optional `knn_normalization="sd"` fits
`StandardScaler(with_mean=False)` on the training coefficients and divides
query coefficients by those same scales. It changes the head's distance
metric, not the factorization, `transform` output or `training_coordinates_`.
Constant and nearconstant coordinates use scikit-learn's scale fallback;
there is no row normalization or added variance floor. The option is validated
but unused by logistic regression and linear SVM.

For KNN, `weights="uniform"` gives the selected neighbors equal votes.
Choose `weights="distance"` to give closer neighbors more influence through
inverse-distance weights. Among the selected neighbors, zero-distance matches
vote equally and other neighbors receive zero weight whenever such matches
exist. Distance weighting changes class probabilities and can change labels,
including decisions under a saved threshold. It can make very close or noisy
observations more influential; compare it with uniform voting on training-only
CV together with the coordinate metric and neighbor count. It is not a
guaranteed accuracy or speed improvement. `weights=None` also means uniform
voting. A callable may implement a custom distance-to-weight rule: it receives
the distance array and must return weights of the same shape. Custom weighting
can forgo shared-neighbor acceleration; saving the model also requires a
serializable callable. These choices affect KNN only and take effect on refit.

`classifier="logistic"` uses a pipeline containing `StandardScaler` and
L2-regularized scikit-learn `LogisticRegression` with the `lbfgs` solver.
Means and scales are learned only from the training coordinates. The same
scaler is reused for prediction; `transform` and reconstruction diagnostics
still use the unscaled NMF coordinates. `logistic_C` must be finite and positive;
larger values mean weaker regularization. `logistic_class_weight="balanced"`
uses training class frequencies, without resampling or changing screening.
A dictionary maps class labels to finite nonnegative weights, for example
`{0: 1.0, 1: 2.0}`. The default `None` leaves observations equally weighted.
Class weighting applies to the logistic loss; the coordinate scaler remains
unweighted.

The logistic iteration limit and tolerance are separate from the NMF
`max_iter` and `tol`. Tune `logistic_C` for predictive performance, and set the
numerical controls to obtain qualified fits. KNN's `n_neighbors` and `weights`
do not tune logistic regression.

`classifier="linear_svm"` fits scikit-learn `LinearSVC` directly on the unscaled
training coefficients, with `penalty="l2"`, `loss="squared_hinge"` and
`dual=False`. There is no hidden standardization or probability calibration.
`svm_C=1.0` is the positive inverse regularization strength and interacts with
the coordinate units. `svm_max_iter=20000` and `svm_tol=1e-5` control the head's
primal solve independently of `max_iter` and `tol` for factorization.
`svm_class_weight` accepts `None`, `"balanced"` or finite nonnegative weights
keyed by training class labels. Weighting changes the SVM loss, not screening
or the representation.

```python
svm = NMFClassifier(
    classifier="linear_svm", svm_C=0.1, random_state=42,
).fit(X_train, y_train)
labels, margins = svm.predict_with_decision_function(X_new)
```

The binary margin is oriented toward `classes_[1]`: a margin strictly greater
than zero selects that class, and equality selects `classes_[0]`. Supported
numeric or string labels are preserved. `positive_label` does not change
margin orientation or the SVM decision rule. A non-`None` `decision_threshold`
with SVM is rejected before factorization.

| Method | KNN | Logistic | Linear SVM |
| --- | --- | --- | --- |
| `predict`, `transform` | Available | Available | Available |
| `predict_proba`, `predict_with_probabilities` | Available | Available | Unavailable |
| `decision_function`, `predict_with_decision_function` | Unavailable | Available | Available |

The combined methods return `(labels, scores)` with one NMF projection and
one application of any fitted head scaler. `predict_with_probabilities` returns
class-probability columns; `predict_with_decision_function` returns genuine
one-dimensional margins. Logistic labels still obey an explicit fitted
probability threshold when requesting margins. Margins are not converted to
probabilities. Method availability follows the fitted head until refit, even
if `set_params(classifier=...)` has changed the constructor setting.

Inputs are dense finite nonnegative matrices. Binary labels can be strings or
numbers; `classes_[1]` is the positive class in evaluation reports. Each training
class needs at least two observations for screening, and each class needs at
least five observations for five-fold calibration. Sparse inputs and multiclass
classification are not supported.

`selected_features_` maps back to the original columns. `selected_components_`
is the learned basis in retained-feature coordinates; `components_` pads that
basis with zeros in discarded columns. `training_coordinates_` contains the
unscaled coordinates supplied to the final classifier; `training_coordinates_mode_`
records the fitted policy as `"transform"` or `"fit_transform"`. `get_support()` returns
the original feature mask. `classifier_` is a bare KNN for unscaled KNN, a
pipeline named `scale`/`knn` for SD KNN, a pipeline named `scale`/`logistic` for
logistic regression, or a bare `LinearSVC` for SVM. `knn_normalization_` records
the fitted KNN metric and is `None` for other heads.

### Class-label domain

String class labels must not end in the NUL codepoint U+0000 (`\x00`). Such
labels can silently change class membership or metrics in string conversions
and comparisons. A NUL followed by other text, such as `"a\x00b"`, remains
accepted; the literal two characters `"\\0"` are also ordinary text. Numeric
and Boolean class-label behavior is unchanged.

The restriction covers screening/classifier fitting, classifier `score` targets,
fold construction, calibration, evaluation targets and fitted classes, and data
fingerprints. Comparison and calibration helpers check labels before creating or
reusing checkpoints/results. Unsupported input labels raise `ValueError`. If an
external estimator returns unsupported prediction labels, evaluation records a
failed prediction instead of computing metrics. These checks do not change the
separate sample/group identifier policy or intercept direct calls to external
scikit-learn estimators and scorers.

Pass original labels to the package while they still retain their values. To
use categories whose names end in NUL, map the original strings to distinct
integer codes before converting them to fixed-width NumPy strings. Converting
an already truncated array to `object` cannot recover the original values:
fixed-width `U2` storage cannot distinguish `"a"` plus padding from `"a\x00"`.
Raw lists/tuples and object/StringDType arrays retaining the original strings
are checked without silently trimming or merging labels. For variable-width
string arrays, class comparisons and scikit-learn routines use the original
Python strings. Calibration checkpoints still distinguish variable-width
string inputs from object inputs.

Older fitted screens/classifiers still containing terminal-NUL class values
are rejected when used for transformation, prediction or support queries.
Encode the original targets and refit; saved model state and historical results
are not rewritten. Information already lost through earlier conversion or
serialization cannot be reconstructed by validation.

### Image values and saved-model reconstruction

Screening retains the supplied nonnegative feature values: it selects columns
without multiplying them by screening scores or standardizing the returned
pixels. "Supplied" means after the caller's explicit preprocessing. Floating-point
conversion can occur. For decoded 8-bit images, an explicit preprocessing choice
is to divide intensities by 255 before screening and factorization:

```python
import numpy as np

# All images have the same dimensions, color convention and flattening order.
X_train = train_images.reshape(len(train_images), -1).astype(np.float32) / 255.0
X_query = query_images.reshape(len(query_images), -1).astype(np.float32) / 255.0
```

The estimator does not infer bit depth or apply `/255` automatically. Other
nonnegative data need their own explicitly chosen units. A fixed division by
255 learns nothing from the data; any learned preprocessing must instead be
fitted within each CV training fold and reused unchanged on its validation fold.

Temporary rescaling for stable pooled-t statistics does not change returned
pixels. SPA also normalizes feature columns temporarily for anchor selection,
then fits its basis against the supplied-scale data. Mean-centering the pixels
can introduce negative values and is not performed before NMF. Optional scaling
for the final classifier acts on learned coordinates, after factorization.

Classification does not erase the saved NMF basis. A basis visualization uses
`selected_components_` and the saved feature locations; a query reconstruction
uses the unscaled nonnegative coordinates returned by `transform`. Assuming
`classifier` was fitted with the `/255` preprocessing above, reconstruction in
the original intensity units is approximately `255 * (W_query @ H)`:

```python
# Reuse the fitted mask and basis; no fit or anchor selection occurs here.
W_query = classifier.transform(X_query)
H = classifier.selected_components_
reconstructed_retained = 255.0 * (W_query @ H)

# Separate plotting/export arrays distinguish excluded features from true zeros.
reconstruction_canvas = np.full(
    (len(X_query), classifier.n_features_in_), np.nan, dtype=np.float64
)
reconstruction_canvas[:, classifier.selected_features_] = reconstructed_retained
basis_canvas = classifier.components_.copy()
basis_canvas[:, ~classifier.get_support()] = np.nan
```

Reshape canvas rows to the original image dimensions and channel order. For RGB,
selection operates on channel features, which need not retain all three channels
of a spatial pixel. NaNs above mark excluded features for display/export only;
never supply these canvases to `fit` or `transform`. Excluded features were not
modeled and cannot be recovered from this factorization. The zero padding in
`components_` is a storage convention, not evidence that excluded pixels are black.

Individual basis amplitudes are not literal 0-255 intensities: positive reciprocal
rescaling of the two factors leaves their product unchanged. Use a stated display
scale for components; multiply the reconstruction product by 255, not both factors.
Approximate reconstructed intensities can exceed 255; retain the numerical result
and clip only a separate display copy if needed. Nonnegativity gives an additive
representation but does not guarantee anatomically localized facial components.

### Failed fits

`FeatureSPA.fit` and `BinaryScreening.fit` discard earlier learned state
when a new fit starts. If fitting raises an ordinary exception, including
during final diagnostic or fitted-attribute construction, they clear partial
learned state and propagate the original error. Fit successfully again before
calling methods that require a fitted model; a failed refit does not restore
the previous model. Internal screening-statistics reuse follows this policy.

If fitting succeeds and a later `transform` call fails, the successful fitted
state remains available. This also applies to the transform stage of
`fit_transform`.

### Changing settings on a fitted model

`set_params` changes constructor settings; it does not refit learned objects.
For `NMFClassifier`, changing the solver, screen, rank, classifier head, KNN
weights/normalization, class weights, thresholds, training-coordinate policy
or MU projection policy takes effect on the next fit. This also applies to
`max_iter`, `tol`, `random_state` and the head's numerical settings: prediction
uses the settings already copied into the fitted decomposition and classifier.
Changing an inactive head- or solver-specific parameter does not activate it.

The exceptions are the four SPA execution controls: `projection_solver`,
`projection_workers`, `projection_batch_size` and `projection_native_threads`.
They affect the next SPA projection without refitting; valid settings remain
unused by CD/MU. Changing a projection solver can change numerical coordinates
and predictions, so record execution settings when reproducing results.

| Public estimator | Changes requiring a new fit | Changes used by the next transform |
| --- | --- | --- |
| `NMFClassifier` | All constructor parameters except the four SPA execution controls | Those four controls, for a fitted SPA model |
| `BinaryScreening` | Screening mode and its selection parameters | None; transformation uses the fitted feature mask |
| `FeatureSPA` | Rank, `normalize`, `condition_limit` | `nnls_max_iter` and the four projection controls |

In particular, changing standalone `FeatureSPA.nnls_max_iter` can change the
next projection's iteration allowance. Changing `NMFClassifier.max_iter`
requires refitting, even for SPA. Neither operation requalifies or modifies
an existing fitted basis by itself.

For resolved decisions, inspect `training_coordinates_mode_`, `mu_projection_`,
`knn_normalization_`, `decision_threshold_` and `positive_label_`, together with
the fitted decomposition, feature mask and `classifier_`. `get_params()` may
describe settings queued for a future fit. Saving/loading with the same
implementation and dependency versions preserves both those constructor settings
and the fitted objects; it does not apply a pending refit.
Direct edits to fitted internals or a stateful custom callable are outside this
parameter-change contract. Do not change settings during an active prediction.

### Model persistence

Save and load models with the same supervised-nmf implementation and dependency
versions. Pickle and joblib round trips preserve constructor settings and
fitted objects without refitting, including settings queued for a future fit.
Ordinary copying and cloning remain supported; cloning creates an unfitted
estimator with the saved constructor settings.

`NMFClassifier` and `FeatureSPA` no longer fill missing fields from older
releases during loading. Cross-release serialized-model compatibility is
unsupported. An older artifact may still load, but successful loading does
not establish that prediction, parameter inspection, cloning or refitting
will work. Use its originating implementation to load an older model, or
refit from the original training data after upgrading. This is not an
automatic conversion or retraining API.

### Prediction dtype

CD and MU preserve float32 training inputs and fitted bases. At prediction,
matching query and basis dtypes retain that output dtype. CD and explicit MU
batch projection retain their respective calculations. MU's default rowwise
policy uses float64 working arithmetic with guarded numerical recovery, then
returns the projection dtype.
If one is float32 and the other float64, projection uses float64 and returns
float64 coordinates. Python lists and integer arrays are validated as float64.
Widening preserves the stored values of both arrays; it avoids rounding tiny
positive queries to zero or overflowing large queries by converting to float32.
A widened basis belongs to a temporary estimator, so the fitted basis, training
coordinates and learned classifier parameters are unchanged. Input validation
still covers every feature before screening.

Mixed-dtype projection can cost more than float32 projection and can produce
different coordinates or predictions from an explicitly downcast query.
It retains the fitted solver, objective and stopping settings; it does not
guarantee numerical success for every extreme input. SPA retains its existing
float64 projection behavior.

### Prediction batches

The default `mu_projection="auto"` resolves at fit time to `"rowwise"` for
`solver="mu"` and `None` for SPA/CD. Explicit `"rowwise"` also requires MU;
explicit `"batch"` selects MU's shared stopping rule and is inactive for SPA/CD.
The default factorization solver remains SPA.

The fixed-basis Frobenius objective separates by observation, but CD and explicit
MU batch projection use stopping criteria shared across the query batch. Batch
MU checks relative improvement in the whole batch's reconstruction error every
ten iterations.
Changing the other observations in a call can change the stopping iteration
and therefore an observation's coordinates and, near a classifier boundary,
its predicted label. This is dependence on batch composition, not random
variation between otherwise identical calls. In batch mode, MU's numerical
recovery paths retain the shared stopping rule.

Rowwise MU supports observations predicted individually or in changing batches.
Each observation receives its own initializer and
relative-error stopping decision, checked every ten iterations. This retains
the fixed-basis unregularized Frobenius objective and the fitted basis while
changing the numerical projection policy. Float64 working arithmetic reduces
the sensitivity of stopping decisions to float32 rounding. Range guards retain
the existing handling of exact zeros, tiny inputs and extreme values, including
per-observation recovery when needed. The output dtype follows the rules above.

During fixed-basis MU projection, a positive denominator can be too small to
represent even when the updated coordinate is representable. The projection
checks for this loss with scikit-learn 1.9.0 and 1.9.1 and recovers the
coordinate when needed. Versions 1.6.1, 1.7.0–1.7.2 and 1.8.0 were also
qualified for this check before the package minimum rose to 1.9.0.
On versions outside this list, ordinary batch projection delegates to the
fitted estimator's transform.
Rowwise projection still solves each observation independently, with the
additional denominator check disabled. If you replace `numpy.dot`
to instrument the solve,
the package preserves the numerical calls, order and exceptions of the original
projection path. With extreme input and basis scales, that path can underflow
silently and give a different prediction even if your wrapper simply forwards
every call. Restore the original `numpy.dot` to use the checked path again.

`"auto"` selects a policy from the solver at fit time, never from incoming query
composition. `get_params()` retains the requested setting; `mu_projection_`
records the resolved fitted policy. Current-format persistence preserves both.
Changing `mu_projection` with `set_params` takes effect on the next fit, so an
already fitted batch model continues to use batch projection until refitted.
Checkpoint identities retain requested and resolved settings; auto and explicit
rowwise requests are not interchangeable checkpoint identities.

To use the same fixed-basis projection for classifier training and prediction,
select both settings explicitly:

```python
model = NMFClassifier(
    solver="mu",
    mu_projection="rowwise",
    training_coordinates="transform",
    random_state=42,
).fit(X_train, y_train)

print(model.mu_projection_)             # "rowwise"
print(model.training_coordinates_mode_)  # "transform"
predictions = model.predict(X_new)
```

`training_coordinates="auto"` still retains MU's joint-fit training coordinates,
including with the new prediction default. Reprojection adds fitting work and
can change classifier predictions; the projection-default change does not select
it automatically.

Use `mu_projection="batch"` when its measured throughput benefit matters for
your workload and batch-composition sensitivity is acceptable. Costs depend on
query size, dtype, numerical range and convergence. In the earlier
default-selection investigation on local Windows and WSL, rowwise singleton
predictions cost about 0.26–0.47 ms more per request than batch, averaged over
sequences of 32 requests. In a difficult
2,000-row float64 Fashion cohort, plain rowwise prediction took 19.03 s versus
18.02 s for batch on Windows and 14.48 s versus 13.66 s on WSL. Those full calls
were single observations, not confidence bounds. Rowwise accuracy was 83.5%
versus batch's 83.7% there; other investigated datasets showed differences in
both directions. A fixed four-row capped recovery case added about 2.4–2.8 ms
(27–29%) through speculative work followed by ordered replay. These historical
observations retain their original scope. Current guarded prefix and setup reuse
reduce repeated work in eligible capped recovery without a new public option.
A later fixed comparison found substantial capped-recovery gains, while ordinary
and full-query costs remained mixed; it does not promise zero overhead or faster
prediction for every workload. See the [original policy evidence](../research/engineering/development-protocol.md#mu-projection-default-and-recovery)
and the [recovery refinement and costs](../research/engineering/development-protocol.md#interrupted-mu-recovery-and-panel-refinement).

Rowwise projection removes the shared stopping decision, but does not guarantee
bitwise agreement across batch shapes, convergence at the iteration cap, or
better classification accuracy. Matrix-kernel rounding can still differ.

A smaller `tol` may reduce these differences but does not guarantee identical
results across batch sizes. `tol=0` disables tolerance-based early stopping and
ordinarily uses the iteration cap. Exact-zero shortcuts and the rank-one
fixed-basis recovery exception below can return without iterations. Disabling
early stopping does not certify convergence or bitwise batch invariance.
The tolerance and cap also apply to factorization; the recovery exception does
not change joint fitting. Keep prediction batch construction in reproducibility
records. See the
[theory and implementation discussion](theory-implementation.md) for the
distinction between the mathematical projection and its numerical execution.

### Iteration limits and qualification

`NMFClassifier.max_iter` is a positive integer shared by factorization and any
training or prediction projection. Its units depend on the numerical routine:

| Routine | Meaning of the allowance |
| --- | --- |
| CD | Coordinate-descent iterations for a fit or fixed-basis projection, with a relative projected-gradient stopping test. |
| MU fitting or explicit batch projection | MU iterations, with a shared relative reconstruction-error improvement test every ten iterations. |
| Default rowwise MU projection | A separate iteration allowance and residual-improvement stopping decision for each observation, checked every ten iterations. |
| SPA | A limit for each public SciPy NNLS solve in basis fitting or projection. Anchor selection is controlled by rank, not this limit. |

For SPA, an explicit cap below three times the rank bypasses batched/QR
acceleration and uses the public reference solve. At larger caps an accelerated
attempt can precede fallback with the original cap. Batched pivot rounds and
SciPy iterations are different units. Standalone `FeatureSPA(nnls_max_iter=None)`
uses SciPy's default allowance, currently three times rank; `None` is not a valid
`NMFClassifier.max_iter` value.

A sufficiently large allowance permits early stopping; reducing it can instead
truncate a fit or projection. Increasing a truncated allowance can change the
learned basis, coordinates and predictions without changing the objective or
stopping tolerance. Crossing a SPA routing boundary can also change the
calculation used. An iteration cap is neither an accuracy tolerance nor a time
deadline. Numerical retries and fallback solves may each use the configured
allowance, so it is not a total-work budget. More iterations cannot repair
invalid input or make unrepresentable output finite.

Choose an allowance that numerically qualifies the intended method on training
and validation data. Inspect fit warnings and each prediction call's diagnostics,
and retain failed or nonconverged candidate/fold records when comparing caps.
The calibration ranking requires valid folds; a shorter unqualified run is not
a faster qualified substitute. Positive-tolerance CD/MU calls can warn on
reaching the cap even when the last iteration satisfies a stopping test. Equal
outputs therefore do not by themselves establish equal qualification. Warning
absence is also not an independent stationarity or optimality certificate.

Changing `NMFClassifier.max_iter` after fitting is pending configuration for the
next fit; it does not alter the saved decomposition's prediction allowance.
Standalone `FeatureSPA.nnls_max_iter` does affect its next transform. See
[settings on a fitted model](#changing-settings-on-a-fitted-model) and the
[zero-tolerance and recovery behavior](#prediction-batches).

### Fit diagnostics

`fit_times_`, `n_iter_`, `converged_`, and `convergence_warnings_` expose fit
diagnostics. `n_iter_` records CD/MU factorization iterations or SPA's selected
anchor count; it is not a prediction-iteration or NNLS-iteration counter.
`classifier_n_iter_`, `classifier_converged_`, and
`classifier_convergence_warnings_` report the final classifier separately;
`classifier_n_iter_` is `None` for KNN. The overall convergence flag and warning
list include factorization, any training reprojection and classifier fitting.
Classifier fit timing includes scaler fitting for SD KNN or logistic regression.
Prediction diagnostics are recorded per call in evaluation reports, as described
under [prediction state and diagnostics](#prediction-state-and-diagnostics).

For MU, an entirely zero matrix after fitted feature selection has the exact
fixed-basis solution of zero coordinates. The package returns that solution
without MU iterations for its standard unregularized Frobenius configuration,
and reports a successful transform. Full input validation still applies,
including to discarded columns. This shortcut applies only to exact zeros.
The fitted classifier then predicts from those coordinates using its existing
scaler and decision rule.

With `mu_projection="batch"`, ordinary inputs retain the scikit-learn calculation.
For float32 queries, tiny fitted basis entries or a floating-point failure first trigger a retry
with float64 copies of the query and basis on a temporary estimator. Internal
underflow remains an error; only final output rounding allows underflow.
The fitted estimator and its arrays are not changed by this retry.

For float64 queries with tiny positive basis entries, a bounded retry scales
the basis by a power of two and the query by its square, then restores the
coordinates. This coupled scaling preserves the uniform initializer and the
relative-error stopping rule in exact arithmetic. It requires reversible
scaling, conservative exponent bounds and valid positive products. These checks
do not guarantee every subsequent floating-point operation; a numerical failure
returns to logarithmic recovery. The retry uses a temporary estimator without
changing the fitted basis or adding a public setting.

Positive query entries below the square root of the input dtype's smallest
normal value enter numerical recovery directly. Float64 problems outside the
scaling bounds, ordinary float64 arithmetic failures and failed retries also
enter this recovery. Its general path uses logarithmic arithmetic to preserve
tiny input contributions, with the rank-one exception below. The iterative
recovery paths retain the MU update equations, uniform initialization and
relative-error stopping rule for the observations supplied to the recovery call.
Rowwise recovery retains a separate initializer and stopping state for every
affected observation. Eligible float64 rows can execute together without reducing
across rows; other cases use the serial route. Logarithmic panels reduce repeated
dispatch while retaining guarded layout-sensitive reductions. A qualified capped
vector attempt can retain completed rows before the first capped row. That row
and the remaining suffix are replayed serially in input order; other declined
attempts retain full serial replay. Under admitted positive-tolerance controls,
final speculative updates that cannot produce an accepted result are skipped.
Eligible recovery can reuse guarded basis setup within the call, including
setup from the vector attempt for its first scalar row. Reuse is cleared before
warnings or unknown callbacks and is not retained on the model or across calls.
These paths preserve warning/error behavior but can still add work for capped
or failing calls. These execution choices do not switch the fitted projection
policy or introduce a public setting. The paths
preserve the fitted basis and return the projection dtype described above;
the float32 recovery path converts back only after solving. Nonzero inputs are
not replaced with zeros; a coordinate below the output dtype's representable
range can still round to zero. Unrepresentably large coordinates raise an error.
In the recovery paths, the rank-one closed-form first MU update supplies the
final coordinates; subnormal outputs use exact input ratios to resolve
round-to-nearest ties.

When a rank-one fixed-basis call reaches logarithmic recovery with `tol=0`, it
returns that same closed-form solution directly. It does not construct the
logarithmic initializer or initial residual, or perform discarded MU updates.
Warnings and exceptions from that removed setup no longer occur: a valid
projection can succeed under strict underflow handling where the earlier setup
failed. Output dtype, subnormal rounding and errors for unrepresentably large
coordinates are retained. Earlier native, promoted and scaled attempts keep
their existing order and warning/error handling. Ordinary successful projection,
positive-tolerance recovery and higher-rank recovery keep their existing paths.

Recovery can cost more time and can change rounding and stopping iterations
relative to the ordinary calculation. An exhausted positive-tolerance iteration
cap still emits a convergence warning. Each iterative retry keeps the configured
cap and tolerance; a failed attempt followed by recovery can do more total work
than a single attempt.
An explicit training reprojection uses the same correction as prediction;
joint factorization and the training-coordinate policy remain unchanged.

For a stage that returns successfully, its convergence flag records the absence
of a captured scikit-learn `ConvergenceWarning`. Other warning categories do not
determine that flag. A skipped training projection has
`training_projection_performed_=False` and `training_projection_converged_=None`.
A failed transform raises its error without changing the preceding fit's
`converged_`. Warnings emitted before a solver failure are forwarded;
`evaluate_classifier` retains them together with the failure in its call record.

These flags do not independently verify stationarity, KKT conditions, global
optimality, statistical consistency or agreement across environments. CD and MU
use different stopping criteria: CD measures relative projected-gradient
violation, while MU measures relative reconstruction-error improvement.
Identical `tol` values or iteration counts therefore do not imply equal numerical
accuracy. See [numerical reproducibility](compatibility.md#cdmu-numerical-reproducibility)
for input precision and runtime considerations.

### Guarded logarithmic MU reductions

Eligible logarithmic MU projection and recovery reuse a newly allocated product
workspace to reduce temporary allocation and reduction work. Caller inputs and
the fitted basis remain unchanged. Projection policy, stopping rules and public
defaults are unchanged; no new setting is required.

Two NumPy/SciPy pairs qualify for this acceleration: 2.5.4/1.18.1 and
2.5.2/1.18.0. Each also requires the expected SciPy source and numerical
bindings. When the source is unavailable or a required binding is missing,
the package uses the standard reduction. The owned float64 route
preserves the qualified stable-reduction order; `FloatingPointError` retains scalar recovery
on the original operands. Library upgrades may disable acceleration until
requalified.

Benefits depend on the workflow, and some small or capped requests can be
slower. The [development decision](../research/engineering/development-protocol.md#guarded-logarithmic-mu-reductions)
records the measured gains, adverse cases, setup cost and maintenance scope.

### Prediction state and diagnostics

In current source, standard `NMFClassifier` prediction preserves learned arrays
and package-owned fitted state. Projection uses a shallow local decomposition
object for execution settings and temporary bookkeeping; it shares the learned
basis rather than copying it. Existing dtype recovery still makes numerical
copies when needed. Changes to projection execution controls between calls take
effect without updating the stored decomposition during prediction. Eligible
explicit QR calls may populate a transient external weak cache for that shared
basis; this is separate from model-owned state and is not serialized with the
estimator. See [QR preparation reuse](#reusing-prepared-qr-factors).

The `last_transform_converged_`, `last_transform_warnings_`,
`last_transform_error_` and `last_transform_seconds_` estimator attributes used
by version 0.3.3 are no longer created or updated. For labeled evaluation, read
the corresponding fields from that call's report instead:

```python
from supervised_nmf.evaluation import evaluate_classifier

report = evaluate_classifier(model, X_validation, y_validation)
diagnostics = report["diagnostics"]
print(diagnostics["last_transform_converged"])
print(diagnostics["last_transform_error"])
print(diagnostics["last_transform_seconds"])
```

The report retains `last_transform_warnings` and adds
`last_transform_performed`. When projection was not attempted, the latter is
`False` and `last_transform_converged` is `None`; a fit-only report must not
reuse an earlier prediction's status. Projection failure is distinct from a
classifier failure after successful projection. Calibration and shared
representation study reports use the same call-local convention.

Older serialized models may still contain `last_transform_*` attributes.
Prediction leaves those historical values untouched, and standard evaluation
ignores them when reporting a new call. Do not use them to infer that call's
convergence. Custom estimators with their own diagnostic contract retain their
existing handling. Direct prediction methods keep their usual return values,
warning forwarding and exception behavior.

This state contract does not make simultaneous calls in one process generally
thread-safe. Warning capture and some native thread-limit controls can interfere
across threads. Use process workers with a per-worker numerical-thread budget
for concurrent evaluations that require reliable warning and convergence
records. Do not refit or change parameters while another call is running.
Third-party KNN query counters may change during prediction even though learned
parameters do not; byte-identical serialization after prediction is not promised.

### Training coordinates

Since version 0.3.0, the default is `training_coordinates="auto"`.
The archived 0.2.0 wheel defaulted to `"fit_transform"`.
For SPA, this resolves to `"transform"`: after fitting the nonnegative basis,
the classifier re-encodes its training observations by fixed-basis NNLS.
Training and prediction therefore use the same fitted basis and projection map.
For CD and MU, `"auto"` resolves to `"fit_transform"`, retaining the coordinates
returned by factorization. The default rowwise MU prediction policy does not
alter this resolution. For a common MU training/query projection policy, also select
`training_coordinates="transform"`.

An explicit `"transform"` re-encodes training observations for any solver.
In version 0.5.0, standalone `FeatureSPA.fit_transform`
also returns fixed-basis NNLS coordinates. Consequently, all accepted
`training_coordinates` settings for a newly fitted SPA classifier resolve to
`"transform"`, including explicit `"fit_transform"`. The fitted
`training_coordinates_mode_` records this resolution. The classifier fits the
SPA basis directly and projects the training observations once during `fit`;
it does not call the standalone projection and then repeat it. For CD and MU,
explicit `"fit_transform"` continues to select the factorization coordinates.

Training projection contributes to `fit_times_["training_projection"]` and
total fit time. A change from anchor to projected coordinates can change a
retrained classifier's predictions and accuracy; an accuracy improvement is
not guaranteed. Report the coordinate policy alongside the solver when
comparing fits. Historical source snapshots and results retain the convention
actually used. Loading does not retrain or re-encode a model; current-format
persistence preserves its saved fit, and the current training policy takes effect
on refit. Older formats are subject to the [persistence support boundary](#model-persistence).

`reconstruction_err_` refers to the classifier's actual training coordinates;
`factorization_reconstruction_err_` preserves the initial factorization diagnostic.

### Fitting and transforming together

`NMFClassifier.fit_transform(X, y)` fits the complete supervised classifier,
including screening, factorization and the classification head, then returns
projected training coordinates. Standard SPA models reuse an independent copy
of the NNLS coordinates already computed during fit, avoiding a second projection.
CD/MU and custom subclasses or transform overrides retain the fit-then-transform
path. Labels are required. The returned array contains projected,
unscaled NMF coordinates; it can differ from `training_coordinates_` when a
CD/MU head was trained on factorization coordinates.

The `fit_transform` method and the `training_coordinates="fit_transform"`
parameter have different roles: the method returns projections after fitting,
while the parameter selects the coordinates used to train the head. Calling
the method preserves that policy. For standard SPA, `fit_times_` already includes
the single training projection; the final array copy is outside that fit time.
For other paths, the additional projection remains outside the recorded fit time.
The operation-scoped projection diagnostics record calls to public `transform`;
returning stored SPA coordinates does not report another transform. Its training
projection remains recorded by `training_projection_performed_` and the associated
convergence/timing attributes. Use `fit(X, y)` when only a fitted classifier is needed.

## Scikit-learn metadata

The estimators declare their input requirements through scikit-learn's public
tag interface:

| Estimator | Input requirement | Target requirement |
| --- | --- | --- |
| `NMFClassifier` | Nonnegative features | Required, binary labels |
| `FeatureSPA` | Nonnegative features | Optional; labels are ignored |
| `BinaryScreening` | Nonnegative features | Required; binary labels are validated during fitting |

`NMFClassifier` retains both classifier and transformer capabilities.
`BinaryScreening` remains a supervised feature transformer, without classifier
tags or prediction methods. Scikit-learn has no general binary-only target tag
for transformers, so callers must still supply valid binary labels. A pipeline's
final classifier tags do not summarize restrictions imposed by earlier steps.

These declarations describe existing requirements. Runtime validation also
checks finiteness, dimensions, class sample counts and usable training features.
Tags do not replace these checks or change fitting, projection or prediction.

`NMFClassifier` and `BinaryScreening` require exactly two classes, with at least
two training observations per class. After array and shape validation, negative
feature values are rejected before target-type and class-count checks. When
negative features and multiclass labels occur together, the negative-input
error takes precedence. Generic scikit-learn checks can supply this combination
while expecting a target error; multiclass rejection is tested separately with
nonnegative features.

The tags do not imply that every generic estimator check passes; the standalone
SPA coordinate convention below still applies.

## Screening and SPA separately

### Standalone SPA options

```python
from supervised_nmf import FeatureSPA

spa = FeatureSPA(
    n_components=2,
    normalize=True,
    condition_limit=1e12,
    nnls_max_iter=None,
    projection_solver="auto",
    projection_workers=1,
    projection_batch_size=256,
    projection_native_threads=1,
)
```

| Setting and default | When to change it and tradeoff |
| --- | --- |
| `n_components=2` | Choose rank using training data. Higher ranks add anchor-selection and NNLS work and require enough independent anchors. |
| `normalize=True` | L1-normalizes columns only for anchor selection. Choose `False` when original feature magnitudes should influence selection; large-scale columns can dominate, and the unnormalized SPA recovery assumptions apply. Returned anchor coordinates remain on their original scale either way. |
| `condition_limit=1e12` | Limits the condition numbers of the selection-scale anchors, original-scale anchors and fitted basis. A smaller finite value greater than one rejects more ill-conditioned fits. Raising it relaxes rejection without improving numerical accuracy. It does not change the separate batched/QR acceleration guards. |
| `nnls_max_iter=None` | Uses SciPy's default, currently three times the rank. Set a positive integer for a different basis/projection iteration allowance, for example after iteration exhaustion. A larger cap permits more work; it is not an accuracy tolerance. A bounded batched attempt may precede fallback with the original cap, so this is not a shared total-work budget. |

The four [projection controls](#spa-projection-execution) have the defaults
shown above. They and `nnls_max_iter` affect subsequent standalone transforms
without refitting. Rank, normalization and condition admission affect the next
fit; changing them does not alter or requalify the existing basis.
`nnls_max_iter` also applies to basis fitting on the next fit. The standalone
[training-coordinate convention](#standalone-spa-coordinates) now uses the
same fixed-basis NNLS map as the SPA classifier.

### SPA basis fitting

Version 0.3.3 fits the basis by guarded batched NNLS against the selected
original-scale anchor columns. It uses the same fixed-anchor nonnegative
least-squares objective as the previous public SciPy column loop. At least
32 features and a nonnegative full-rank design with condition number at most
1e4 are required for batching. Numerical blocks contain at most 256 target
columns. Feasibility and original-design KKT checks qualify each result;
ineligible designs and failed numerical checks use public SciPy NNLS.

The helper uses normal equations, which square conditioning. Coefficients can
lose precision near the acceleration limit even when residual and optimality
checks pass. The acceleration guard is separate from `condition_limit`, and
the fitted basis must still pass the existing rank and condition checks.
If the helper raises `FloatingPointError`, basis fitting retries the original
public column loop on the original inputs, preserving its accepted numerical
range. Validation and iteration-exhaustion errors propagate. Projection retains
its own numerical-range behavior described below.

Basis fitting runs in the calling process and does not use the projection
execution controls. Explicit iteration caps below three times the rank use
SciPy directly for ordinary-range inputs. Otherwise, a batched attempt is
bounded by three times the rank before fallback receives the original cap.
This preserves the objective, not bitwise results or universal stopping parity.
Loading a current-format model preserves its stored basis; refitting uses
this implementation. No constructor parameters or fitted-state fields were added.

### SPA projection execution

AOCL environments automatically use a package-side dimension-padding workaround
for explicit matrix products in SPA's NNLS basis and projection solves.
MKL-only environments retain direct multiplication. This preserves the NNLS
objective and its numerical checks without changing the installed AOCL library.
This is a temporary compatibility measure. Padding will be bypassed for verified
AMD-fixed runtimes and removed when the supported AOCL builds no longer require
it; unqualified versions retain the safeguard. See the
[implementation, evidence scope and retirement policy](../research/engineering/aocl-nnls-workaround.md).

These controls apply to both `FeatureSPA` and `NMFClassifier(solver="spa")`.
They affect fixed-basis projection, including `FeatureSPA.fit_transform` and
the training projection of newly fitted SPA classifiers. They do not control
anchor selection or basis fitting. CD/MU ignore valid SPA projection controls.

`projection_solver="auto"` attempts batched NNLS for at least 32 query rows.
`"batched"` also attempts it for smaller queries. Both require a nonnegative,
full-column-rank design with condition number at most 1e4, a conservative
acceleration guard independent of SPA's existing model condition limit.
Solutions undergo finite, nonnegative and original-design KKT checks. Failed
checks, ill-conditioned bases and small explicit iteration caps use public
SciPy NNLS with the original cap; its errors propagate. Internal batched
pivot attempts are bounded by the smaller of the supplied cap and three times
the rank, then fall back. This does not change the NNLS objective. Floating-point
coefficients need not be bitwise identical. `projection_solver="scipy"` selects
the public row-wise solver for every query.

`projection_solver="qr"` requests a shared economic QR factorization followed
by public SciPy NNLS solves on the reduced matrix. The original nonnegative
design must have full column rank and at least twice as many rows as columns;
each group submitted to the solver must have at least 32 right-hand sides.
The estimated condition number of the reduced matrix must be finite and at
most 1e6. This bound is separate from both the batched solver's 1e4 limit and
SPA's model condition limit. It is an admission safeguard, not a guarantee of
coefficient accuracy or identical classifier decisions.

For a fixed design `A = Q @ R` with orthonormal columns in `Q`,

```text
||A @ z - b||² = ||R @ z - Q.T @ b||² + ||(I - Q @ Q.T) @ b||².
```

The last term is independent of `z`, so reduced NNLS retains the original
objective minimizer and the constraint `z >= 0` in exact arithmetic. The
reduced matrix and targets may be signed. They go to the public SciPy solver;
finite, nonnegative coefficients must pass KKT checks against the original
design and targets before the QR result is accepted. Ill-conditioned or
ineligible designs, failed numerical checks and reduced-solver numerical
failures use the current guarded solver and its public SciPy fallback. Explicit iteration
caps below three times the rank retain the current direct SciPy route. Each
fallback receives the original cap, and reference-solver errors propagate;
the cap is not a combined work budget across an attempt and its fallback.

#### Reusing prepared QR factors

Eligible explicit `projection_solver="qr"` calls can reuse an immutable QR
preparation for the same live fitted basis, including across predictions and
eligible process workers. The default remains `"auto"`, and SPA basis fitting
is unchanged. Current positive qualification requires NumPy 2.5.4, SciPy 1.18.1
and threadpoolctl 3.7.0 with the accepted stock MKL/OpenMP composition. These
checks do not pin or change your dependencies. Other supported runtimes,
unqualified implementations and changed numerical policies retain the ordinary
QR/fallback route. A dependency upgrade may disable this extra acceleration.
Only Linux attempts optional discovery before the process has initialized its
thread controller. On Windows with qualifying MKL libraries, an ordinary
serial projection can initialize that controller. A later eligible call can
then save factors for subsequent calls. Each process worker checks its own
controller before using factors prepared by the parent.
Parent and worker inventories compare complete native-library records in a
canonical order. Different enumeration order alone no longer rejects matching
state; library identities, duplicate records, process identity, thread settings
and the remaining capability checks still matter.

Reuse is additional to the numerical guards above. It requires rank at least
32 and `n_features * rank**2 >= 2**20`, a qualifying query and iteration cap,
and a freshly verified basis and numerical/native context. Changed basis bits
or context cannot reuse stale factors. These conditions are conservative
implementation limits, not a promise that every admitted call will be faster.

The preparation lives in a transient external weak cache, not in fitted
estimator attributes. Prediction preserves model-owned state while this cache
may populate. Each live basis owner can retain one immutable preparation;
collection releases its entry, and a new process starts with an empty registry.
Pickle/joblib model saves do not store the factors. Loaded models and deep
copies begin with new owners; shallow copies sharing a basis may reuse its
entry. Optional preparation failures use the ordinary route. Required solver,
process-transport and assembly errors retain their existing behavior.

Each already immutable factor buffer of at most 1 MiB stays as bytes during
worker transport. Larger buffers retain the existing read-only mapping route.
Workers validate and bind the payload against their own current context before
reuse; transport does not persist a preparation in an estimator or checkpoint.

Retained factors and snapshots use extra memory. Sharing a tall reduction can
help repeated large queries, but eligibility, copying, transport and reduced
per-row solves can also cost more than the current batched path. Small changes
in coefficients from floating-point arithmetic are possible. Select the solver
using complete fit/prediction workloads: QR is not a general speed or accuracy
improvement. See the [measured benefits, costs and evidence limits](
../research/engineering/development-protocol.md#reusing-prepared-fixed-basis-qr-factors), including the
[native inventory, transport and import refinement](
../research/engineering/development-protocol.md#native-inventory-transport-and-classifier-import-refinement).

Input magnitude is checked before every projection solver.
Ordinary-range inputs retain their solver path. Extreme inputs use exactly
reversible common binary scaling of the design and each independent RHS;
rows with the same scale share execution. The scale preserves the NNLS
minimizers and does not change conditioning. Nonzero scaled values must have
binary exponents within [-450, 450]; an input span that cannot meet this
conservative bound raises `FloatingPointError` before solving any scale group.
Unsupported range does not mean the mathematical problem has no solution.
Original iteration caps and reference-solver errors still propagate.

Keep the default `projection_workers=1` for small workloads. More processes
can cost more in startup, transfer and ordered result collection than they save
in computation. A measured rank-16 workflow with 256 features was faster overall
with one worker on both Windows AOCL and Ubuntu MKL, although individual
prediction calls were not uniformly faster. That result is guidance for the
tested workload, not an automatic size threshold. Measure complete fitting and
prediction, including first-use startup, before opting into more workers.

Blocks contain at most `projection_batch_size=256` rows. With
`projection_workers>1`, queries of at least 512 rows and two full blocks use
joblib's reusable loky process pool. Smaller queries execute in the caller.
Compatible calls reuse workers automatically, including across models; no
pool or cached fitted basis is serialized with an estimator. First use pays
startup cost, and joblib retires idle workers. `projection_native_threads=1`
limits numerical threads during projection. On exit, threadpoolctl restores
the previously observed effective thread counts. With MKL, threadpoolctl 3.7.0
restores the calling thread's count as a local setting. If the application
later changes MKL's global budget, it can resume inheriting that budget on
this thread by calling `mkl_set_num_threads_local(0)` there.
[Intel MKL thread-local settings](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-set-num-threads-local.html)

The controller discovers NumPy/SciPy libraries at first NNLS or
projection use, which can occur during basis fitting.
Some native thread-limit controls affect shared process state; use process
workers for concurrent work.
Eligible process calls can prepare a shared reduction in the parent and bind
it against each worker's current context. Calls with more chunks than requested
workers may create a preparation; a single wave uses this path only when a
matching preparation already exists. This single-wave distinction applies to
built-in integer worker controls; other accepted integer types can use the
conservative cold-parent path. With built-in integer controls, a cold single-wave
call keeps the ordinary worker route. Joblib's original `"1M"` mapping threshold remains constant across
models and reuse decisions. Without an admitted shared preparation, each
eligible worker call performs its own QR setup. Numerical admission still
applies separately inside each worker and scaling group; a final chunk with
fewer than 32 rows can use the current solver even for a large full query.

Execution controls may change after fitting, for example
`model.set_params(projection_workers=8)`, without refitting. Outer CV scheduling
remains the caller's responsibility: budget approximately
`CV workers * projection workers * native threads` simultaneous numerical
threads. For many concurrent CV fits, keep projection workers at one.
All counts must be positive integers; `-1` is not supported.
Current constructors set all four execution controls, and current-format
persistence retains them. Fit and transform use those attributes directly;
accessing a missing control raises `AttributeError` instead of silently
selecting SciPy or default resource limits. Existing input and value validation
still applies. The `"auto"` constructor default, explicit
`projection_solver="scipy"` option and numerical fallback behavior are unchanged.
Missing fields from older releases are not filled during loading; see the
[persistence support boundary](#model-persistence).

### Saved decisions

For binary KNN, the native rule is majority voting: more than half of the
neighbor votes (or total distance-based voting weight) selects `classes_[1]`.
Exact ties select `classes_[0]`. Both `weights="uniform"` and
`weights="distance"` are supported; uniform remains the package default.
Neither weighting choice changes the nonnegative NMF representation.
Odd `n_neighbors` avoids binary count ties with uniform votes, but does not
guarantee the absence of weighted ties or equal-distance neighbor boundaries.

To compare the weighting choices, tune `weights` jointly with `n_neighbors`
using training-only CV, keeping screening and factorization inside each fold.
Keep the majority boundary fixed unless the application explicitly requires a
different decision objective. The general `calibration_search` helper supports
this joint grid; `calibration_knn_grid` requires a fixed weighting choice and
only varies `n_neighbors` among classifier parameters. Use identical folds for
separate fixed-weight searches if comparing their scores.

`decision_threshold=None` preserves the final estimator's original prediction
rule, including KNN ties and logistic decisions whose probabilities round to
0.5. For KNN and logistic regression, an explicit threshold uses the strict rule
`P(positive_label) > threshold`.
The default positive label is `classes_[1]`; probabilities always retain the
order in `classes_`. The cutoff must lie in [0, 1], or be exactly
`np.nextafter(0.0, -np.inf)` to represent the study's all-positive sentinel.
Threshold one yields all-negative predictions. Probabilities are unchanged.

The threshold and positive label freeze during fit as `decision_threshold_`
and `positive_label_` and survive serialization. Set them from training-only
calibration before fitting the final estimator; the package does not select a
cutoff from new labels. Changing their constructor settings requires refitting
to change the fitted decision policy. `predict_with_probabilities` returns
labels and probabilities with one projection. The shared KNN calibration
helper honors fixed thresholds, but threshold grids require the general
`calibration_search` helper.

These execution and decision-policy additions were released in version 0.2.0.
They do not revise saved study results or establish an accuracy improvement.

`NMFClassifier` sets `decision_threshold` and `positive_label` during
construction; fitting reads these attributes directly. Successful fits record `decision_threshold_` and
`mu_projection_`. `predict`, `predict_with_probabilities` and
`predict_with_decision_function` require the fitted threshold field;
`predict_proba` and `decision_function` do not use it.
Standard `transform` requires the fitted `mu_projection_` field, including
its explicit `None` value for SPA and CD. Applying an explicit
threshold also requires `positive_class_index_`, including in shared NMF KNN
calibration. Accessing a missing required field raises `AttributeError` instead
of supplying a policy; existing input, fitted-state and label checks retain
their order and may fail first. Explicit `decision_threshold=None` and
`mu_projection="batch"` remain valid settings; the current requested projection
default is `"auto"`. Changes to these constructor
policy settings apply to the next fit, while predictions use fitted policies.
Diagnostic reports continue to omit unavailable optional metadata.

### Screening and anchor details

Both estimators use the same screening default:

| Estimator | Default screening | Active selection parameter |
| --- | --- | --- |
| `NMFClassifier()` | `"absolute"` | `score_threshold=2.0`, an inclusive cutoff on the absolute pooled-t score |
| `BinaryScreening()` | `"absolute"` | `score_threshold=2.0`, the same absolute pooled-t rule |

Use percentile screening for a relative feature budget and absolute screening
for a pooled-t cutoff whose retained count varies with the training data.
Retaining more features increases downstream work; tighter selection can
discard useful features or leave too few for the requested rank. Standalone
`BinaryScreening` accepts only `"percentile"` and `"absolute"`; omit that step
to skip screening. `NMFClassifier` separately offers `screening="none"` to
retain every input feature. A fitted standalone mask survives parameter
changes and serialization: `set_params` affects the next fit, while `transform`
uses the saved mask.

Absolute screening needs no mode override; percentile screening is explicit:

```python
from supervised_nmf import BinaryScreening

absolute_screen = BinaryScreening(score_threshold=2.0)
percentile_screen = BinaryScreening(screening="percentile", percentile=10.0)
```

Changing `percentile` or `score_threshold`, including through `set_params` or
a parameter grid, does not switch `screening`. Both values must be valid, but
only the parameter for the selected mode affects feature selection. For example,
`BinaryScreening(percentile=20.0)` still uses absolute screening. To retain a
percentile, set `screening="percentile"`.

For a classifier pipeline with a screening step named `screen`, compare modes
with separate grid dictionaries so that each parameter is active:

```python
screening_grid = [
    {"screen__screening": ["percentile"], "screen__percentile": [5.0, 10.0, 25.0]},
    {"screen__screening": ["absolute"], "screen__score_threshold": [1.0, 2.0, 3.0]},
]
```

Pass the complete pipeline and grid to the search so screening is fitted inside
each training fold. Neither mode guarantees that any features survive or that
enough remain for a requested factorization rank.

The calibration helpers record `screening_diagnostics`, including the active
mode and selection parameter for recognized `NMFClassifier` and
`BinaryScreening` configurations. They emit one aggregate warning per calibration
grid call if a grid explicitly supplies a known screening parameter that is
inactive in one or more candidates. For example, a percentile grid under
`screening="absolute"` does not tune the screen; under `screening="none"`, both
selection parameters are inactive. For mixed-mode grids, the diagnostics retain
which configurations are affected. Use separate grid dictionaries as above to
avoid this ambiguity. The check does not switch modes, prune candidates, or
warn merely because an estimator retains its unused constructor defaults.
Each rank-search stage is a separate calibration grid call.

`BinaryScreening` is a sklearn transformer with `fit`, `transform`, and
`get_support`. It exposes signed `t_scores_`, squared `scores_`, selected indices,
class counts and constant/infinite-score counts. In version 0.3.0,
statistics subtract each training feature's minimum before dividing by the
remaining spread. This preserves represented small variation behind a large
offset. `statistic_offsets_` and `statistic_scales_` record these operations;
scaled variance/standard-error diagnostics are provided. Older fitted objects
can lack `statistic_offsets_`; transforming them requires only their saved mask.
The returned data are never multiplied by the screening scores.

Absolute thresholds must be finite and nonnegative. Globally constant columns
are excluded even at threshold zero. Different classwise constants have infinite
absolute scores. Percentile screening calculates its cutoff on ranks of the
computed `abs(t_scores_)`, using the original feature count. Eligible features
above the cutoff are retained. At the cutoff, lower input-column indices fill
any remaining places up to `int(n_features_in_ * percentile / 100)`.
Mathematically equal statistics can round to different magnitudes, in which
case the larger computed magnitude ranks first. Screening raises if no feature
survives. Near-zero variances are used as computed.

#### Screening execution layouts

Screening uses an internal layout specialization without changing the pooled-t
formula, selection rules or returned feature values. After validation, aligned,
native-endian float32/float64 NumPy arrays with canonical Fortran strides, more
than 128 columns and no simultaneous C-contiguity can use it. For float32 inputs
whose float64 working array is at most 1 MiB, the already-required widening
allocation uses C order. Larger working arrays retain the full translated
snapshot and compute statistics in bounded column blocks. Blocking requires
NumPy's underflow policy to be `"ignore"` and sklearn's `assume_finite=False`;
other policies retain the original calculation route. The small widening branch
is independent of those blocking guards. Widths at most 128, C arrays and other
layouts retain their original arithmetic. These internal boundaries are fixed
engineering choices, not a universal performance crossover or new public setting.

This optimization serves inputs that already have the qualifying layout.
Ordinary CV row gathers such as `X[train]` produce C arrays, and compatible
calibration searches reuse statistics within a fold. Do not convert inputs to
Fortran order solely on the assumption that screening will become faster: the
conversion has a cost, and small or bypassed cases can retain overhead. The
[development decision](../research/engineering/development-protocol.md#screening-statistics-for-fortran-order-inputs)
records measured gains, costs and the scope of the numerical checks.

### Standalone SPA coordinates

`FeatureSPA` is the separately defined feature-anchor transformer, with
`anchor_indices_`, conditioning diagnostics, and a fixed `components_` basis.
Its `normalize=True` option normalizes training feature columns only during
anchor selection. See [the mathematical specification](methodology.md).

**Anchor-selection limitation:** after supervised screening, SPA chooses anchors
without using the labels. With L1 column normalization, its first anchor maximizes
empirical coefficient of variation; later choices maximize residual norms after
projection. Surviving background features with high CV can therefore dominate
the selected anchors. Retaining all signal features does not guarantee that the
low-rank representation preserves their classification information. This also
applies when `NMFClassifier` projects both training and query observations.
See the [mathematical qualifications](theory-implementation.md#spa-anchor-selection-and-screening-false-positives)
and [retrospective diagnostic](../research/wang-simulation-validation.md#october-20-anchor-and-information-diagnostic).

Version 0.5.0 makes `FeatureSPA.fit_transform(X)` use the
same fixed-basis NNLS map as `fit(X).transform(X)`. A generic sklearn pipeline
therefore trains its next step on the same representation used for prediction.
The original anchor columns remain available separately:

| Operation | Output |
| --- | --- |
| `spa.fit_transform(X_train)` | Training observations projected by NNLS against the newly fitted `components_` |
| `spa.fit(X_train).transform(X_train)` | Training observations projected by NNLS against the fitted `components_` |
| `spa.transform(X_query)` | Query observations projected against that same fitted basis |
| `spa.fit_coordinates_` | Original-scale anchor columns used to fit the basis |

The basis is fitted with the anchor coordinates held fixed. Projection then
holds the basis fixed and solves for coordinates. Projected training coordinates
can differ materially from the anchors on noisy data. Exact projection cannot
increase reconstruction error against that fixed basis, but does not guarantee
better classification. Numerical tolerances still apply. Standalone
`reconstruction_err_` continues to describe the anchor-coordinate factorization;
neither `fit_transform` nor `transform` replaces it or `fit_coordinates_` with
projected results. To measure the returned representation's reconstruction,
evaluate its product with `components_` against the supplied training data.

Newly fitted `NMFClassifier(solver="spa")` models use fixed-basis projection for
classifier training and prediction for all accepted `training_coordinates`
settings. The standalone correction adds one training projection to generic
pipelines that previously used anchors; prediction work is unchanged. The
default SPA classifier already used projected training coordinates and does not
gain a second projection during `fit`. Archived releases and historical studies
retain their recorded coordinate convention. Loading does not retrain or
re-encode a saved classifier; see the [persistence support boundary](#model-persistence).

When composing screening, SPA and a classifier manually, project both sets of
observations. For example, with nonnegative training and validation arrays:

```python
from sklearn.neighbors import KNeighborsClassifier
from supervised_nmf import BinaryScreening, FeatureSPA

screen = BinaryScreening(screening="absolute", score_threshold=2.0)
X_train_selected = screen.fit_transform(X_train, y_train)
spa = FeatureSPA(n_components=2)
Z_train = spa.fit_transform(X_train_selected)
head = KNeighborsClassifier(n_neighbors=3).fit(Z_train, y_train)

Z_validation = spa.transform(screen.transform(X_validation))
predictions = head.predict(Z_validation)
```

The cutoff, rank and neighbor count above are illustrative; enough features and
independent anchors must survive screening for the requested rank. For CV,
refit the screen, basis and classifier inside every training fold, then transform
that fold's validation observations with those fitted objects. Do not precompute
screening or SPA coordinates on the full dataset before splitting into folds.

## Calibration and reporting

`calibration_search`, `calibration_knn_grid` and `calibration_rank_search`
default to `cv=5`, `seed=0` and `threads=1`. The examples below explicitly
override the seed. Use supplied `groups` for subject-disjoint folds or `folds`
for an exact declared split; the seed governs generated folds, not every
estimator's random state. Fix the estimator's `random_state` separately when
reproducible initialization is needed.

`threads` limits supported native numerical thread pools; it does not launch
parallel CV candidates or folds. Those loops execute serially in these helpers.
SPA's projection workers and per-worker native-thread limits are separate and
can override the surrounding limit, so `threads` is not a global concurrency
ceiling. Budget both levels explicitly; see
[SPA projection execution](#spa-projection-execution).

For a staged training-only workflow, start with the
[quickstart tuning guide](../README.md#training-only-tuning) and pass
`refit=False` to `calibration_search`. That mode returns `estimator=None` plus
the CV report; it does not fit a final model or access final evaluation data.
Review numerical qualification and boundary/refinement diagnostics, complete
the declared training-only search, and freeze settings before final evaluation.
Constructor defaults and the small grids below are illustrative starting values.

Each CV fold fits its own screen, representation and classifier on that fold's
training rows. Validation rows use the fitted mask, basis and any fitted scaler.
Candidate values are also cloned before configuration, so an ordinary estimator
supplied as a replacement pipeline step does not carry fitted state between
folds or mutate the caller's grid objects. The final full-training refit starts
fresh. Shared-KNN calibration reuses a representation only within its training
fold and compatible representation settings.

Keep `y` fixed throughout `calibration_search`, including in callbacks or
other code that shares the target array. For object labels, equality matters
as well as the displayed value. Replacing a string with a subclass that displays
the same text but compares differently can change which observations count
toward balanced accuracy and which candidate wins.

`calibration_search` requires finite class labels for native floating-point
targets, including when scikit-learn's `assume_finite=True` skips its own
input checks. The search rejects NaN and infinity before fitting or writing a
checkpoint. With scikit-learn 1.9.0 or 1.9.1, both accuracy
objectives qualify fold scores against the original native floating-point class
labels. Both classes occur in every validation fold. Scikit-learn requires each
value to survive conversion to a signed 64-bit integer and back to the original
float dtype. A fractional value, even one representable step from an integer,
fails this check. The search records the local conversion result at the
platform-sensitive upper boundary `+2**63`. If estimator code changes the
caller's labels and the metric subsequently returns a score for a disqualified
request, the fold records a failure and the candidate has no mean score.
The training-majority fallback uses the original class counts and labels.

These guarantees follow sklearn's ordinary cloning contract. The calibration
helpers reject frozen estimators and custom `__sklearn_clone__` overrides: their
retained state is not fully described by constructor parameters, so it cannot
be safely identified for these searches. They are not silently unfrozen.
Learned preprocessing performed before calling CV is not undone. Keep it
inside the estimator pipeline, and supply groups when observations belong to
the same person or other dependent unit.

Rank expansion retains actual candidate objects internally. JSON report entries
for estimator-valued parameters are display descriptions, not executable
estimators to reconstruct from text. Resume uses the supplied original grid.

### Reading a completed calibration result

| Call | Checkpoint argument | How to identify the selection |
| --- | --- | --- |
| `calibration_search(..., refit=False)` | Optional `checkpoint_path` | `selected_candidate_index` identifies the CV winner, or is `None` if no candidate qualified. `result.estimator` is `None` and `is_fallback` is `False`. |
| `calibration_search(..., refit=True)` | Optional `checkpoint_path` | `selected_candidate_index` identifies the first candidate in CV order that passes the full-training refit checks. If none passes, the search emits `CalibrationFallbackWarning`, the index is `None`, `is_fallback` is `True`, and `result.estimator` is a fitted training-majority classifier. |
| `calibration_knn_grid` | Required `checkpoint_dir` | The first entry in `ranked_candidate_indices` is the CV winner. An empty list means no candidate qualified. |
| `calibration_rank_search` | Required `checkpoint_dir` | `selected_candidate_index` identifies the CV winner across the evaluated stages, or is `None` if no candidate qualified. The top-level indices span all stages. Each stage also has its own local indices. |

For `calibration_search`, set `report = result.report`. For either specialized
search, use the returned dictionary as `report`. This gives the qualified CV
order, with earlier candidates first when their scores tie.

```python
qualified = sorted(
    (row for row in report["candidates"] if row["valid"]),
    key=lambda row: (-row["mean_score"], row["index"]),
)
cv_order = [row["index"] for row in qualified]
```

When a general search writes a checkpoint, the file contains the report
with its `checkpoint_sha256` field. The shared-grid and bounded-rank
`report.json` files store the report under `payload` with the checksum
in the sibling `sha256` field.

### Reading a completed CV report

`supervised_nmf.evaluation.load_calibration_report(path)` returns the saved
report dictionary from a completed schema-2 `calibration_search` checkpoint
created with `refit=False`. This example looks up the winning parameters
and their recorded mean validation score.

```python
from supervised_nmf.evaluation import load_calibration_report

report = load_calibration_report("results/calibration-cv.json")
winner = report["selected_candidate_index"]
if winner is None:
    print("No candidate qualified.")
else:
    row = report["candidates"][winner]
    print(row["parameters"], row["mean_score"])
```

The file path is the only required input. You can read an existing completed CV-only
checkpoint even if it records `resume_eligible=False` or the callback's
original random generator is unavailable. The reader checks the checksum,
the stored request fingerprint, and whether the recorded fold results,
candidate scores and selected candidate agree. These checks establish
internal consistency. Establishing who produced the file requires a trusted
source. Reproducing the scores requires the original data, configuration
and computational environment.

Each archived fold score describes the labels the scorer received for that
candidate. If those labels changed during calibration, the archive alone
cannot establish what the candidate would have scored on the original labels.

If the recorded evidence leaves a decisive claim unresolved, the reader raises
`ValueError` identifying the claim. For example, early writers saved a warning's
class name and message. A solver and a callback could produce identical text
under different warning classes. If that distinction could change which
candidate won, or whether anyone qualified, the archive cannot establish
the result. A candidate already ruled out by an independent failed fold
does not create that ambiguity. Current writers record the warning
classification and the point in the fold when eligibility was decided.
They also save the truth values actually tested on the estimator and its
classifier. This matters for callbacks, because a value that appears as
zero in JSON could have evaluated to true during the fit. Reading the
report uses the recorded decision without invoking the callback.

To check the exact file against an independently retained SHA-256, use
`load_calibration_report(path, expected_sha256=original_file_digest)`.
The digest applies to the bytes read from disk. A mismatch raises
`ValueError`. A match still requires the same checks on the saved scores,
candidate ranking, class labels and refinement advice. The embedded
`checkpoint_sha256` covers a different representation.

The reader checks fold coverage, recomputes means and sample standard
deviations, and applies the original lowest-index tie rule. When the archived
class dtype and scikit-learn version establish the floating-point check above,
the reader applies it to every saved fold score, including scores retained by
failed folds. Older reports with a score that depends on the unrecorded local
conversion at `+2**63` raise `ValueError`. If an old report omits the target
dtype and its displayed values could represent either unscorable native floats
or object labels, the reader cannot verify a saved score. A version-3 retained
class-count record provides independent evidence of a native numeric target.
Older writers displayed every nonfinite floating label as `null`. A finite
class beside that display can still disprove a score. If the original value
and float format leave the score uncertain, the reader raises `ValueError`.
This includes a historical float16 target whose infinity actually survived
the scorer's conversion. Archives containing only unscored failures remain
readable when their remaining evidence checks pass.

For balanced accuracy, the reader checks that the recorded scores are attainable with a shared
validation-fold composition. Current reports identify how the writer
collected the evidence. When a numeric label conversion drops rows, the
writer saves the retained counts and correct predictions used to check the
score. If either set of scorer inputs is an object array, uses a custom scalar
type, or has a label cache that can run caller-defined comparisons, the writer scores one confusion
matrix and saves each nonempty row's count and number of correct predictions.
An object label can affect lookup even when both inputs start without a cache.
If an older report omitted the matrix and its score does not fit the recorded
validation counts, the reader cannot verify that score.
The writer also records correct counts for large folds when reversing the
scorer's rounding can be inconclusive. If the reader exhausts its numerical
search without verifying that the score is attainable, it raises an error.
Earlier reports can also lack the class dtype needed to interpret duplicate
or reversed numeric class displays. For fixed-width byte labels, the reader
checks the byte values behind the displayed strings and their ordering.

When the recorded labels establish how observations map to native Unicode
classes, a custom predictor can lose observations during scoring. Each
retained class count must still fit within that class's validation count.
For object-array predictions, current reports also record whether the values
are ordinary strings. A caller-defined string can change how scikit-learn
looks up a label while it builds the confusion matrix. In that case a single
true class can genuinely contribute to more than one row. If an older report
needs that explanation and lacks the observation that distinguishes ordinary
strings from caller-defined values, the reader reports the uncertainty.
When both scorer inputs contain ordinary strings and neither has a label
cache, the confusion matrix must account for every validation observation.
Each class that occurs only in the predictions requires at least one incorrect
prediction. These checks also apply to an ordinary-string fold when
caller-defined labels elsewhere in the request obscure its class counts.
An ordinary decision tree fitted on the fold's two string classes can predict
only those classes. If its true validation labels have no cached label list,
the matrix must include every observation even when the predictions have a
cache. A cache on the true labels can produce a genuine single-row matrix.
For scikit-learn 1.9.0 and 1.9.1 the reader also checks whether a reported
score is possible for the recorded dtypes. A native byte target can explain
a failed fold. A byte prediction can sometimes score when it is a column and
caller-defined label keys match its values to the target.

The current writer records the prediction dtype before calling the scorer.
For each balanced-accuracy score it also records the state of the target's
label cache at that fold. A custom estimator can change a shared cache during
fitting even when the target values and validation split stay the same.
The earlier `confusion-matrix-v1` writer recorded the prediction dtype after scoring. A custom
label comparison could change that dtype even when the score was valid. If
such a report records a dtype that conflicts with its score, the reader raises
`ValueError` because the file does not establish which dtype the scorer saw.
This check applies to every score the report returns, including a score saved
before a later step in the same fold failed.

Earlier writers could consult a cached label list after scikit-learn had
discarded that cache during scoring. On NumPy 2.2.6 with scikit-learn 1.9.0
or 1.9.1, an old reduced count alone cannot verify a score that is impossible
on the full validation fold. A stock tree with floating labels could also
encounter a caller-defined comparison while looking up cached labels. The
reader requires independent support for the score if that ambiguity matters.
It raises an error when the score contradicts the recorded counts.

Current checkpoints bind the observed screening diagnostics to the request
fingerprint. The reader checks the winning candidate's inactive parameters
and advice against that observation. Fold evidence version 6 also records
the ordered numeric coordinates used to produce each winning parameter's
refinement advice, including NaN, positive and negative infinity, and whether
the grid offered `None`. These observations are saved with the completed
CV result. They preserve distinctions that the public JSON display renders
as `null`. The reader checks the advice against the observations and the
recoverable grid declarations. Older files remain readable when their saved
evidence settles the advice. For example, the recorded scikit-learn version
can establish how a stock pipeline applied its screening parameters.
If a hidden screening override or an unknown coordinate order could change
the recommendation, the reader reports the unresolved parameter.

Reading a report leaves the checkpoint untouched and performs no fitting or
resumption. A malformed, detectably inconsistent or unsupported report raises
`ValueError`. File-access errors such as `FileNotFoundError` propagate to
the caller. Use the originating workflow for partial checkpoints, refit
checkpoints, shared-grid and bounded-rank reports, and comparison manifests.
When `calibration_search` reopens a checkpoint, it checks the saved scoring
objective, class labels and counts, and fold indices against the live request
before reusing a fold or updating the file. For example, a checkpoint
displaying `[0.5, 1.5]` for a request with classes `[0.0, 1.0]` raises
`ValueError`. A completed CV-only checkpoint also goes through the archive
reader's checks. Its recorded tuning time must be finite and nonnegative, and
its thread-pool inventory may be empty. A valid restart returns the saved
report without fitting or rewriting the checkpoint.

### Input preparation and CV usage conditions

For decoded 8-bit face images, pass floating-point pixel values divided by
255 to CV when using the recommended image convention: one image per row,
one pixel/channel value per column. Apply the same conversion to later query
images. The package does not divide by 255 automatically. Here, input before
learned preprocessing does not mean that values must remain in the 0–255 range.

- **Fixed input conversion is safe before CV.** Division by the known constant
  255 learns nothing from any observations or labels. Put learned operations,
  including screening, basis fitting and any learned scaling, inside the
  estimator pipeline so that each fold fits them using its training rows only.
  Screening all observations before calling CV leaks information that the
  package cannot undo.
- **Provide groups for dependent observations.** For subject-disjoint face
  evaluation, supply person IDs and keep each person in one partition of every
  training/validation or final-test split. The helpers enforce separation for
  supplied groups; they cannot infer identities from image pixels or detect
  incorrect identity metadata.
- **Separate selection from performance estimation.** Reserve final test data
  before exploring settings and leave it untouched until the procedure is
  frozen. The winning inner-CV score can be optimistic because it was used to
  select settings. Alternatively, use outer CV that repeats the entire tuning
  procedure within each outer training split; evaluate only on that split's
  untouched outer validation observations. Preserve subject separation at both
  levels where applicable.
- **Keep custom components within the same training-only contract.** Their
  `fit` methods may learn from the supplied training rows, but must not read
  validation data/labels or externally fitted state through globals, closures
  or files. Custom callable parameters must behave as fixed functions. The
  implementation/source checks below do not inspect all external dependencies.

### Scalar controls for calibration and evaluation

`calibration_search`, `evaluate_classifier`, `calibration_knn_grid`,
`calibration_rank_search` and `run_comparison` accept ordinary Python and NumPy
integer scalars for `threads`. Accepted values become Python integers before
native execution and checkpoint identity construction. Counts must be positive
and fit the native signed C-int argument (`np.iinfo(np.intc).max`, normally
2,147,483,647). This is an argument-representation bound, not a recommendation
to request large pools or a bound based on the machine's CPU count. Python and
NumPy booleans, datetime/timedelta values, floats, arrays, strings and `None`
are rejected. Separate projection-worker controls still apply; `threads` does
not impose a global process/thread budget.

General calibration accepts Python `bool` and NumPy `bool_` for `refit`,
normalizing to Python `bool`. Numeric `0`/`1`, strings, arrays, `None` and other
truthy objects are not Boolean flags. Invalid thread counts and refit flags
are rejected before fold generation can consume RNG state, before native
runtime setup and before checkpoint output.

The seed contracts remain deliberately different:

| API | Split-seed contract |
| --- | --- |
| General calibration and `build_cv_folds`, generated folds | The existing sklearn random-state contract, including integer seeds, `None` and `numpy.random.RandomState`. |
| General calibration and `build_cv_folds`, explicit folds | The supplied folds determine the partition; seed is unused for splitting. Its supplied value remains in general calibration's checkpoint identity. |
| Shared-grid and rank search | A fixed, nonboolean Python/NumPy integer from `0` through `2**32-1`, including with explicit folds. Temporal values are rejected. |
| `run_comparison` | The generated-fold generic contract, checked without drawing random numbers before output creation or native-runtime setup. |

For reproducible restarts, give the splitter and each estimator independent
integer seeds. General calibration can also recognize a supported private
`RandomState` reconstructed at the same starting state. An advanced split
generator can select different folds. The supplied split seed is also part
of the request when folds are explicit.

An estimator can use NumPy's global random state through `random_state=None`
or by holding the global generator itself. Skipping a completed fit can skip
draws that a later fit needs. A restart can refuse to reuse results when it
cannot establish the starting state for the requested work. Start a fresh
search in this case. Recovery does not rewind the calling process. A custom
fit that calls `default_rng(None)` draws fresh system entropy on each fit.
Pass an explicit integer seed if that fit must be reproducible.

A callback can declare its dependency on NumPy's global random state through
a default, bound argument or captured variable named `random_state`, `rng` or
`seed` with the value `None`. Recovery may decline a wrapped callback when it
cannot establish which arguments the fit used.

For supported estimator-owned NumPy generators, recovery checks the draw state,
the seed sequence used to spawn children, and shared references that affect
the fit. A fresh search can use configurations whose generator state cannot
be verified for recovery. For example, a generator initialized from a mutable
seed array can create child streams during a fresh run. On NumPy 2.x, if a
supported estimator takes a generator and the comparison will be extended
later, initialize the generator from an integer, such as
`np.random.default_rng(123)`.

On NumPy 1.26, copying a generator can preserve its next ordinary draws
while losing the seed sequence used for child streams. A fresh calibration
can fit and save its results with a live NumPy `RandomState` or `Generator`.
Reopening that CV checkpoint requires verifiable RNG state for the estimator
and its callbacks, including when a callback captures an RNG that the
executed branch never uses. An integer seed on the classifier controls only
the classifier. For restartable fits, remove unused captures and pass each
randomized estimator its own integer seed. For example, a custom transformer
can construct a local RNG from its `random_state` integer in `fit`, store the
projection it draws, and apply that same projection in `transform`.

A function used as an estimator parameter can keep its closure and default
arguments across fits. When these hold a generator, sharing a container or a
seed-pool view can change what the fit draws. The reader distinguishes
supported sharing patterns. If a captured view exposes other parent storage,
or a retained object has state the reader cannot verify, plan to repeat the
fit in a fresh location. When a custom callback reads a fitted attribute or
storage outside its declared array values, the caller must reconstruct that
state too. A successful read alone cannot establish that it was reconstructed.

If a callback advances a private generator during a saved representation fold,
restart that shared-grid or rank study in a fresh directory. In a model
comparison, loading an earlier model skips its draws during fitting and
validation. A later model that depends on the skipped draws can require a
fresh comparison, even if it already has its own saved evaluation. Independent
integer seeds avoid this dependency. A callback's array view or a generator
stored in a NumPy dtype's metadata or field title can also link two models.
When the reader cannot verify their independence, rerun the comparison from
scratch. This applies both to reading the later model on its own and to
appending a model whose constructor or grid may use a buffer changed by the
skipped work. The same restart restriction applies to a fit with a custom
Python object as a NumPy string dtype's missing-value sentinel.
For ordinary string dtypes, the checkpoint distinguishes a missing element
from a literal string with the same display value and records the exact
scalar sentinel, including floating-point NaN bits.
Use a stock NumPy splitter or an integer seed for restartable comparisons.
A `RandomState` subclass can change estimator inputs during `shuffle`.
Restoring its saved partition skips those changes.

`run_comparison` saves one common
partition for its first model and uses it for every subsequent model, including
models appended in later calls. Earlier releases drew folds for each model.
With `seed=None` or a mutable `RandomState` split seed, later models can
therefore receive different folds and produce different results after upgrading.
Reading a compatible completed evaluation also reuses the saved partition.
Integer seed normalization
does not alter estimator constructor parameters, Boolean class labels, or
`positive_label` semantics.

These changes reject general calibration's former accidental acceptance of
Boolean thread counts and shared/rank search's accidental acceptance of certain
timedeltas. Valid ordinary integer controls keep their numerical behavior.
Older-source checkpoints remain historical evidence; source guards are not
bypassed or migrated.

### Reusable candidate and checkpoint identity

Calibration helpers identify configurations structurally, independently of
their display text. Identity binds nested estimator classes and constructor
parameters, exact ordinary NumPy array contents/dtypes/shapes, scalar types,
containers, supported RNG states, and inspectable callable configuration.
Dictionary order is preserved because a dict-valued custom parameter may use
ordered operations. Equivalent fresh estimator objects can resume without
depending on memory addresses printed by `repr`.

The base estimator and replacements supplied in the grid both contribute
implementation-source hashes, including inherited class implementations.
`calibration_search` checks these implementation files before and after each
new fold fit and each final candidate refit. If a check detects a source
change, the search raises an error before saving the affected fit result.
Changed source or configuration requires a new checkpoint location.
Identity schema 2 removes the old
display-only `parameters` and `candidates` entries from the calibration identity;
candidate report entries still contain readable parameter descriptions. Old
checkpoints remain historical evidence and are not migrated to the new identity.

Unsupported opaque or cyclic values and array subclasses carrying extra state
raise an error before fitting. Save custom Python estimators and user-defined
functions in an importable source file so the search can inspect their
implementations. These restrictions do not change ordinary estimator
`fit`/`predict` calls. Source-file tracking is not a dependency sandbox: mutable
globals, external data/services, imported helper implementations not traversed
through the configuration, runtime monkeypatches and old class objects retained
across module reloads are outside this guarantee. Keep custom implementations
fixed and self-contained, and start a fresh search when such dependencies change.
Custom callable parameters must behave as fixed functions; cloning does not
isolate stateful closures or mutable globals, and constructor configuration must
remain unchanged while a search is running.

#### Recovering a search with custom transformations

With a checkpoint, `calibration_search` compares the estimator and grid at
calibration entry with their state just before publishing each fold. This also
applies to the calibrations started by `run_comparison`. When the check first
observes a change, it saves `resume_eligible=False` together with the affected
fold. The report gives its zero-based `resume_ineligibility_candidate_index`
and `resume_ineligibility_fold`, and explains the decision in
`resume_ineligibility_reason`. The current call can still finish.

- To obtain a fitted estimator after a marked fold, rerun the **entire search**
  with a new checkpoint path or comparison output directory. Even an
  output-neutral change, such as incrementing a captured list counter, can
  require this rerun.
- A compatible, completed `refit=False` checkpoint still returns its
  checksummed CV report. A compatible saved model evaluation still supplies
  its archived predictions and carries the same marker fields with the
  `calibration_` prefix. When reopening through `calibration_search` or
  `run_comparison`, reconstruct the estimator and grid as they were at
  calibration entry. Reusing a callback instance
  mutated by an earlier call can fail the entry fingerprint check.
  `calibration_search` checks the fit RNG before returning a CV archive,
  even with `refit=False`. On NumPy 1.26 a callback capturing a live
  NumPy RNG can therefore require a fresh checkpoint path even for a
  completed CV report. To inspect the saved CV results directly, see
  [Reading a completed CV report](#reading-a-completed-cv-report).
- When the folds were clean and a change occurred only during the final refit,
  restart from the original configuration. The search reuses the folds and
  repeats the refit.

The entry fingerprint includes the contents of ordinary NumPy arrays and
scalars. For structured parameters it records how NumPy exposes each field,
including its scalar type, byte order, offset, alignment and lookup names.
This applies to nested fields and fixed-size subarrays. For example, a
transformer can read a named field as an attribute on an `np.record` scalar.
For a function that captures a plain `np.void` or `np.record` scalar, the
fingerprint also records its NumPy flags. Two callbacks that select different
features based on `scalar.flags.writeable` need distinct representation fits.
For supported NumPy buffers, the identity also records the scalar's position,
the layout and contents of the backing allocation, and sharing with other
captured views. This matters when a callback reads another record through
`scalar.base`. Changing any of these properties can require a fresh checkpoint.

The scalar's chain of ordinary NumPy arrays can contain up to four arrays,
including a contiguous allocation of at most 4 KiB, using supported
non-object dtypes. A directly linked array view can share that allocation.
An indirect view, for example one created through a memoryview, can obscure
whether the callback sees the same storage. A callback with larger or
unverifiable backing storage can run in a fresh search. This includes a
callback that retrieves a record scalar stored in an object field of its own
backing array. An object array with named fields can also return the stored
scalar directly when indexed. Iterating over ordinary records with object-valued fields
remains supported when the stored values themselves can be identified.
The shared grid fits distinct opaque declarations independently
and can still share a fit across neighbor counts for the same declaration.
To make such a search restartable, capture the small value the callback
actually needs, such as its feature index or a Boolean writeability flag.
If a shared-grid callback changes a supported scalar or its backing data
during a committed fold, restarting in a fresh directory ensures that the
callback runs again. Rank search checks committed folds across all stages
before it resumes.

A fresh calibration or shared KNN grid can use a custom `np.void` subclass
even when the class was defined interactively and its source is unavailable.
A shared grid accepts the subclass as a dtype or as a class.
It can also use a record scalar with extra Python
attributes, including one stored in dtype metadata. In these cases each
separately supplied representation receives its own fit. Neighbor counts
can share a fit when they use the same parameter objects. Recovery requires
a fresh search because the scalar's Python state can affect the result.

At fold publication the check tracks Python configuration structure,
Python scalar values and the contents of supported NumPy RNG states. Outside
those RNG states, it checks the dtype, array shape and supported captured
layout or flags. It also checks the bounded backing storage of captured record
scalars. Other ordinary non-object buffer contents are skipped. Replacing a
number, a fixed-width string, a date or a field in a non-object structured array can
escape detection if shape and dtype stay the same. Equal-length byte
replacements, mutable globals and state shared across models can also escape
detection. A checkpoint without a marker can therefore yield different results
if a callback consumes external state. Skipping completed folds or a cached
model also skips the random draws that happened in that work.

For a random feature map, give each estimator an independent integer seed.
Learn the map in `fit` and apply that fitted map to each validation row in
`transform`. For example, scikit-learn's `GaussianRandomProjection` stores
its sampled matrix as `components_`. Both pipelines below can reconstruct
their own maps when a later call adds a model or resumes an interrupted fit.

```python
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeClassifier
from supervised_nmf.evaluation import run_comparison

models = {}
for name, seed in [("map_a", 101), ("map_b", 202)]:
    model = Pipeline([
        ("map", GaussianRandomProjection(n_components=4, random_state=seed)),
        ("tree", DecisionTreeClassifier(random_state=seed)),
    ])
    models[name] = model, {"tree__max_depth": [2, 4]}

results = run_comparison(models, X_train, y_train, X_validation, y_validation,
                         output_dir="runs/random-maps", seed=42)
```

Keep `comparison-folds.json`, `comparison-folds-receipt.json` and the
per-model records together. The receipt identifies the saved fold plan. Each
model evaluation identifies the estimator and grid configuration recorded
when its calibration began. The directory reserves `comparison.json` and
`<model>-cv.json` or `<model>-evaluation.json` for names made of ASCII
letters, digits, underscores and hyphens. Start a new comparison in a fresh
directory if an older comparison lacks its plan or receipt. To finish a job
started before upgrading, use its original source and environment. With the
upgraded package, use a fresh checkpoint path or comparison directory.

Each call writes `comparison.json` for the models supplied to that call.
If A already has results, calling with B alone keeps A's per-model files and
writes a summary listing only B. Pass both A and B to produce a combined
summary using their compatible saved evaluations.

### Selection objective and report fields

In version 0.5.0, `calibration_search`, `calibration_knn_grid`,
`calibration_rank_search`, and `run_comparison` accept `scoring="accuracy"`
by default. This selects by ordinary accuracy: the fraction of correctly
classified validation observations. Set `scoring="balanced_accuracy"`
explicitly to select by mean class recall. These are the two supported strings;
other names and callable scorers are rejected. Class prevalence does not change
the objective automatically.

Each candidate must qualify in every required fold. Its selection score is the
equal-weight arithmetic mean of the fold scores, so it can differ from a pooled
out-of-fold score when fold sizes differ. Candidates are ranked by higher mean
score, preserving their declared order on ties. Bounded rank exploration uses
the same objective throughout and keeps the incumbent on equal scores.

Reports name the selected objective as `scoring`. Fold records contain `score`
and the corresponding metric field (`accuracy` or `balanced_accuracy`). Candidate
records contain `mean_score` and `mean_accuracy` or `mean_balanced_accuracy`.
The selected candidate's CV mean is `selected_mean_score`. Unqualified candidates
have no usable mean; a search without a qualified selection reports a null
selected mean. Metric-specific aliases name only the metric actually used:
accuracy scores are never stored under `balanced_accuracy` keys.

In the original research workspace, three private helpers requested balanced
accuracy for both initial calibration and refinement to match the criterion
used to select winners. The study scripts were `scripts/dataset_search_support.py`,
`scripts/validate_celeba_unscreened.py` and `scripts/validate_celeba_screened_cd.py`.
Their protocol captures sorted complete threadpool records, including duplicates,
so enumeration order alone did not prevent reuse. Reuse still required matching
library paths, versions, thread counts and source. A source change required a new
search directory. The frozen protocols retained their original runtime scope.

In the public source, the [CelebA grid reviewer](../scripts/review_celeba_cv.py),
[CelebA default selector](../scripts/select_celeba_default.py) and
[four-solver reporter](../scripts/report_four_solver_benchmarks.py) audit saved
shared-KNN CV results under their declared balanced-accuracy objectives.
Current receipts require integer schema tags, Boolean qualification flags,
an explicit no-refit record, and matching generic and metric-specific fold,
candidate and selected scores. Numeric scores must be finite and within [0, 1];
Boolean values are not scores. Invalid candidate means and an unavailable
selected mean remain null. Failed or nonconverged folds may omit scores or
retain a matching numeric/null pair.

Historical receipts retain their existing read-only checks and recorded metric
names. Receipt auditing does not convert artifacts, refit models or relax
source/runtime requirements for checkpoint reuse. The public calibration
functions above continue to support both declared scoring choices.

Selection scores describe training CV, not independent held-out performance.
`evaluate_classifier` still reports both ordinary accuracy and balanced accuracy,
plus class-specific measures, from the supplied evaluation predictions.
`run_comparison` forwards `scoring` to each method's calibration and records that
selection objective separately from its development-validation metrics.

Calibration defaults to ordinary accuracy. Pass `scoring="balanced_accuracy"`
to weight the two classes equally. Changing the scorer or package source
requires a fresh checkpoint path or comparison directory.

The following compact API example uses the default `refit=True`, which fits
a selected configuration on all supplied training data after CV. It illustrates
the call sequence, not a completed tuning protocol. Do not run its evaluation
step on a final reserved cohort until the training-only review and freeze above.

```python
from supervised_nmf.evaluation import calibration_search, evaluate_classifier

calibrated = calibration_search(
    classifier,
    {"score_threshold": [1.0, 2.0, 3.0], "n_components": [2, 4],
     "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1, scoring="accuracy",
    checkpoint_path="runs/my-development-search.json",
)
if calibrated.report["is_fallback"]:
    raise RuntimeError("Calibration returned a majority fallback, not a qualified NMF model.")
print(calibrated.report["scoring"], calibrated.report["selected_mean_score"])
validation = evaluate_classifier(calibrated, X_validation, y_validation)
```

The cutoff grid above uses the absolute-screening `classifier` configured at
the start of this page. Suitable ranges depend on the dataset. Any
failed or nonconverged required fold disqualifies its candidate. The report
retains each attempt, parameters, warnings, folds, selected dimensions, and
timings. If no candidate qualifies in CV or all qualifying candidates fail
the full-training refit, standalone `calibration_search(..., refit=True)`
returns a classifier that predicts the training majority and emits
`supervised_nmf.evaluation.CalibrationFallbackWarning`. Use this category
to filter the warning or turn it into an exception. With checkpointing enabled,
the search saves the completed report before issuing the warning. A retry
can reuse those CV folds and runs the final refit stage again. With `refit=False`,
`selected_parameters=None` identifies the absence of a qualified candidate.

`CalibrationFallbackWarning` applies only to standalone refitting.
`run_comparison` returns a report with an `is_fallback` field for each model,
so callers can identify the substitution on fresh and resumed comparisons.
Pass the model record to `format_evaluation_summary` to display the fallback
alongside its metrics.

For logistic regression, pass `NMFClassifier(classifier="logistic", screening="absolute")` and replace
`n_neighbors` with `logistic_C`, for example `[0.01, 0.1, 1.0, 10.0, 100.0]`.
Each fold fits its own screening, NMF, scaler and classifier. A logistic
convergence warning disqualifies a candidate just as a factorization warning
does; raising the logistic cap requires a recorded numerical configuration.
For linear SVM, use `NMFClassifier(classifier="linear_svm", screening="absolute")` and the grid key
`svm_C`. Its cap and tolerance qualify the primal solve; they are not substitutes
for tuning C. SVM warnings are retained and disqualify candidates in the same
way. Optional KNN SD scaling is fitted separately within each training fold.

### Resuming a calibration search

Checkpoints can resume only when data, parameters, scoring, versions, folds and package
source hashes match. Changing code or settings requires a new run. The source
examples create unique run directories and expose `--resume` for compatible runs.

Source hashes describe the files on disk. Python can keep executing code it
imported before those files changed. Start each study process from a fixed
installation or a verified source snapshot. Keep that copy and the custom
estimators, callbacks, execution backends and helpers it uses fixed for the
life of the process. You can edit a separate development checkout while the
study runs. To reuse saved model results, keep the recorded implementation
files at their original paths. The paths reported for loaded numerical
libraries must also match. Moving a recorded implementation or library can
require a fresh run even when the copied files have identical contents.

After all writers have stopped, you can move an otherwise compatible run's
saved outputs. For `calibration_search`, move the checkpoint file and pass
its new location as `checkpoint_path`. For `calibration_knn_grid` and
`calibration_rank_search`, move the entire checkpoint directory and pass its
new location as `checkpoint_dir`. For `run_comparison`, move the entire
comparison directory and pass its new location as `output_dir`. Keep the
directory contents and their relative paths intact.

If code or a tracked dependency changes in the active environment, restart
the study processes from a clean installation or snapshot of the intended
version. Use a new checkpoint path or run directory for the changed study.
Recompute affected results produced or reused after the change and regenerate
reports derived from them.

Even for numeric input arrays, checkpoint compatibility includes the installed
Pillow distribution version or its recorded absence. A custom transformer or
execution backend can use Pillow during fitting or prediction. If the recorded
version or presence changes, start a new checkpointed run.

`examples/celeba_parameter_study.py` additionally offers example-specific
`--final-fit-recovery off|all` (default `off`). `all` commits completed selected
fits before evaluation and terminal held/failed outcomes; a compatible completed
final result still takes precedence. It restores only your own trusted local
joblib/pickle checkpoints, with source, data and active-runtime checks, and adds
ordinary-run verification and storage costs. The choice is bound to the run and
must match on resume. It cannot continue an interrupted solver. This does not
change general calibration's checkpoint format or guarantees. See the
[full recovery contract](celeba-parameter-study-recovery.md) for option guidance,
commit precedence, attempt accounting and the exact finalization-lock scope.
Eligible raw KNN retains complete native state through public protocol-5
serialization with one verified training-cache buffer, without another fit.
Loading copies that matrix into an owned read-only snapshot. Keep the cache
verified and unchanged through this copy; the restored model has no later
training-cache dependency. Unsupported layouts use complete joblib before
commit; an incompatible committed reference is an error. Original fit and
prediction timers retain their numerical boundaries. Allocation and copying
are included in model loading; total workflow time also includes checkpoint
work, and cache effects can change measured numerical durations. In the latest
focused raw-KNN comparison, fresh `all` aggregation averaged 81.68 ms longer
in MKL and 150.79 ms longer in AOCL than the frozen pre-change default.
AOCL prediction means were about 1.1% higher, with substantial uncertainty.
Recovery comparisons used complete-joblib restoration as their reference,
not refitting. An earlier small 26-result workload roughly doubled its final
aggregation time when recovery was enabled. See the
[recovery guide](celeba-parameter-study-recovery.md) for component differences,
uncertainty bounds and workload limits. Choose `all` for the value of preserving
completed fits, with its ordinary overhead disclosed.

General calibration owns its checkpoint file from the first read through the
final write and return, including final refitting and fallback. A competing
process or thread using the same destination receives `RuntimeError` promptly;
retry after the active call finishes. Different checkpoint files remain
independent, including files in the same directory. `checkpoint_path=None`
creates no checkpoint, temporary file or lock marker. `save_data_manifest`
likewise protects its complete create-or-verify operation, so a competing
creator cannot silently replace a different manifest.

The adjacent `.<filename>.lock` marker is reserved and intentionally remains
after completion or interruption. Its existence does not indicate a busy
checkpoint: the operating system owns and releases the lock. Do not delete or
replace the marker. Shared-grid and rank-search directory locks retain their
existing `.study.lock` convention. Coordination covers cooperating callers
using the same destination, including ordinary parent-directory aliases.
Final symlinks and hard links are not treated as interchangeable replacement
destinations; network and cloud-synchronized filesystems require their own
locking qualification. This does not make multi-file comparisons transactional.

`write_json` stages each complete report in its own exclusively created sibling
temporary file before atomic replacement. Direct concurrent calls still use
last-replacement-wins semantics; they do not merge report state or acquire a
calibration lock. Ordinary failures attempt cleanup of only that call's temporary
file. A temporary file left by abrupt termination does not block subsequent
writes. Atomic replacement preserves complete-file visibility; it does not
promise durability through power loss. Checkpoints from earlier package source
remain incompatible and are not migrated by this change.

### Groups, folds and sample identifiers

Both `calibration_search` and `calibration_knn_grid` accept `groups`, `folds`,
and `data_ids`. `groups` keeps rows from a subject together. `folds` supplies
exact training/validation index pairs and must contain `cv` exhaustive splits;
each validation row appears once and training indices are its complement.
Every validation fold must contain both classes, and every training fold needs
at least two observations per class. Supplied IDs and folds are validated and
bound to checkpoints. See [tuning improvements](../research/engineering/tuning-improvements.md).

Evaluation metadata is optional, but independence is checked only when the
corresponding metadata is supplied:

```python
from supervised_nmf.evaluation import format_evaluation_summary

validation = evaluate_classifier(
    calibrated, X_validation, y_validation,
    sample_ids=validation_ids, train_sample_ids=training_ids,
    groups=validation_subjects, train_groups=training_subjects,
    train_y=y_train,
)
print(format_evaluation_summary(validation))
```

Passing the full calibration result lets the evaluation report and its
formatted summary identify a substituted classifier. The fitted estimator
alone carries no record of why calibration selected it.

Training/evaluation IDs and subjects must not overlap. Each corresponding pair
(`sample_ids`/`train_sample_ids` and `groups`/`train_groups`) must use the same
identifier type: both numeric or both string. Python/NumPy integers and finite
floating-point IDs can be mixed; nonblank strings retain their exact values.
Boolean values (including NumPy booleans and Boolean arrays), complex values
(including zero imaginary parts), temporal values, and other objects such as
`Decimal` or `Fraction` are rejected with `ValueError`. The same policy applies
to calibration `data_ids`/`groups` and cross-validation subject IDs. Validation
does not round numeric identities: large integers stay distinct, while
`1` and `1.0` remain the same identifier for duplicate/overlap checks.
Numeric identity uses exact represented values, including mixed NumPy
scalars and Python integers. Internal comparison keys prevent NumPy hashing
or mixed-type rounding from missing an overlap or merging distinct groups;
the original identifiers remain unchanged in returned metadata and reports.
This policy also governs automatically generated group-disjoint folds.
A string/numeric mismatch raises an error before prediction. A mask already
converted to an integer array by the caller has lost its Boolean origin and
is treated as the supplied integer IDs.

Sample IDs and subject IDs may use different types from each other. Prepare
each pair with a consistent encoding before evaluation.
These checks compare the supplied identifiers; they cannot verify real-world
identity or reconcile different spellings within the same identifier type.

Extended-precision `np.longdouble` IDs retain their precision in validation,
evaluation reports and calibration checkpoints. Reports encode these scalars as
tagged JSON objects rather than converting them to Python floats. For example,
`np.longdouble("-1.5")` becomes:

```json
{"__supervised_nmf_scalar__": "longdouble-v1", "significand": "-3", "exponent": -1}
```

The exact value is `int(significand) * 2**exponent`, interpreted with exact
arithmetic. Nonzero significands are odd decimal strings; signed zeros use
`"0"` or `"-0"` with exponent zero. This representation distinguishes adjacent
values without depending on storage padding or narrowing to float64. Actual
`np.longdouble` precision remains platform-dependent. Ordinary integer, string,
and float16/32/64 JSON representations are unchanged.

The `__supervised_nmf_scalar__` key is reserved for scalar output tags. These
objects are report/checkpoint representations, not accepted input identifiers;
pass original numeric IDs to evaluation and calibration APIs. Reusing the same
IDs produces the same checkpoint identity, and changing an adjacent value is
detected. Existing source/runtime compatibility checks still apply; this change
does not migrate checkpoints created with different package source.

For general report values, extended complex scalars use `clongdouble-v1` with
`real` and `imag` fields encoded as above. Nonfinite real values/components
become JSON `null`, matching the ordinary nonfinite-float policy. Complex and
nonfinite identifiers remain invalid.

### Identifier validation costs

Numeric identifier validation uses an owned snapshot so validation, comparisons
and returned values refer to the same captured input. Eligible evaluation
checks reuse Python numeric sets or sorted native values. Mixed native values
may use a common floating-point comparison format only when precision,
normal/subnormal range and integer-format or actual integer bounds prove that
conversion exact. Otherwise the existing exact-key fallback remains. These
internal representations do not narrow the original IDs in reports or change
the accepted identifier types, numeric equality, or validation/error order.
Array and scalar subclasses retain their existing fallback behavior.

Exact stock `np.longdouble` scalars avoid redundant intermediate scalar copies
in report serialization and eligible identifier-key conversion. Key conversion
chooses its ordinary or native loop from the first value of a validated
one-dimensional NumPy object array, without scanning the full array first. Later
values still receive individual type checks on the native path, so scalar
subclasses retain their conversion, warning and error behavior. Inputs beginning
with an ordinary scalar keep the ordinary conversion loop. Exact numeric keys,
original report scalars and the tagged JSON representation remain unchanged.

This removes repeated work for homogeneous long-double identifiers, but retains
tradeoffs: short ordinary calls can pay dispatch overhead, and a native first
value followed mostly by ordinary values can add per-value checks. Complete
evaluation and calibration calls may also be faster or slower. Choose identifier
types for their meaning and required precision; no public option or new dtype
requirement is introduced by this optimization.

Group bookkeeping also uses a bounded exact-key route for eligible requests
with at most eight combined evaluation/training rows, and avoids sorting when
an owned native group array is proved uniform. Different endpoints reject
uniformity cheaply; equal endpoints still require checking the whole array.
These internal choices add no public option and do not imply an optimal cutoff
or a speedup for every input. Nonuniform scans, small snapshots and fallback
conversion retain costs. Identifier handling does not change model fitting,
predictions, support thresholds or the meaning of reported metrics.

The earlier isolated WSL investigation is historical evidence for its own
source, not a measurement of these current routes. It found substantial gains
in long-double evaluation workloads, while repeated conversion of 10,000
wider-precision keys alone cost roughly 1.3–1.6 ms more and first use added a few
microseconds in a two-ID request. The later Windows AOCL/physical Ubuntu MKL
investigation measured the native comparison and group routes separately,
including adverse ordinary and complete-transaction costs. See the
[current design and evidence](../research/engineering/development-protocol.md#exact-identifier-evaluation-and-group-bookkeeping).
Exact identity and source/runtime resume requirements take precedence over
narrowing values for speed.

### Evaluation outputs

Training labels determine the majority-class reference; evaluation labels
never choose that class.
`min_class_support=10` and `min_groups=5` flag small evaluation samples without
altering predictions or excluding rows. The summary includes accuracy and balanced
accuracy, per-class precision and recall, evaluation and supplied training class
counts, available subject counts, timing, and numerical-stage diagnostics.
Missing training counts and unavailable metrics are reported explicitly.
Tuning and final fitting costs remain separate from prediction. Final fitting
can include unsuccessful refit attempts; the fitted estimator's total already
includes its screening, factorization, training projection and classifier stages.
These totals overlap and must not be added. Individual-operation latency and
batch throughput are distinct; sums of stage medians or concurrent worker times
do not establish total elapsed time.

For a CPU timing comparison, finish CV before timing the selected model. Run
each final fit and held-out prediction with one application worker and one
native BLAS/OpenMP thread. Use the same prediction batch size for every method.
Schedule one timed job at a time on each physical host, including across seeds
and Python environments. Report CV wall time and its worker/thread settings
separately. CV can use the available parallel budget.

The simulation and CelebA comparison examples use
`run_comparison(..., threads=cv_threads, serial_final=True)`. With this policy
the comparison finishes CV for every declared method before fitting any final
model. The simulation example also finishes CV for all declared replications
before final fitting begins. Start a fresh serial comparison to change the
method list or CV worker settings. The final clone
uses serial settings for Joblib, KNN and SPA projection, including when the CV
grid selected a parallel setting. The report records timing intervals and the
native pools observed. Use a fresh output directory when switching timing
policies. For a restart from backup, restore the study directory, its CV
timing journal (where provided), and its model records from the same snapshot.
Independent runs of the same protocol have different elapsed times.
In a serial comparison, `final_fit_seconds` gives the duration of
the fit that produced the evaluated estimator. `cumulative_final_fit_seconds`
also includes failed attempts and refits following an interruption. A caught
interrupt saves its measured duration. If the thread-pool check fails after
a fit, its attempt records the unverified interval and the run stops. A
subsequent successful run includes that cost in the cumulative duration.
If a process ends before it can save
the duration, `tuning_seconds_complete` or
`cumulative_final_fit_seconds_complete` is false and the corresponding
number is the measured subtotal. `first_final_fit_started_utc` is null when
an earlier fit may have started before its timestamp was saved. Each CV interval records its Joblib
settings and native thread limit. Custom
estimators with their own worker pools need their own serial
configuration. Serial comparisons and their example launchers reserve the host for the current user
while they run. On a host with an existing study scheduler, set
`SUPERVISED_NMF_TIMING_LOCK` to its absolute lock-file path for all timing
launchers in every environment. The parent directory must exist. On Unix
the drivers acquire an exclusive `flock` on this file. Launch them with the
lock available to acquire. After forcibly terminating a coordinator, wait
for its CV worker processes to exit before starting another timed run.
Coordinate other accounts and unrelated jobs when timing on a shared machine.

Numerical-phase diagnostics on `NMFClassifier` now include
`factorization_converged_` and `training_projection_converged_`, with their
corresponding `*_convergence_warnings_` attributes.

For a direct `NMFClassifier`, the report's `diagnostics` also records
`decision_threshold`, `positive_label`, `positive_class_index`, and
`mu_projection` from the corresponding fitted attributes. These describe the
saved fit even if `set_params` has changed constructor settings for a future
refit. The threshold label and class index identify the probability column used
by an explicit threshold; they are inactive when `decision_threshold` is `None`,
which selects the fitted head's ordinary decision rule. Evaluation metrics and
ranking scores continue to use `classes_[1]` regardless of these settings.

An explicit `mu_projection=None` means that the MU policy is inapplicable.
`"batch"` and `"rowwise"` describe the fitted MU projection policy, not the
particular numerical kernel or recovery path executed by a prediction.
Missing fitted attributes, including unavailable legacy fields, are omitted
rather than inferred from constructor settings; explicit `None` is retained
as JSON `null`. Loading does not synthesize missing legacy fitted fields. These fields
are available in diagnostic dictionaries, including failed evaluation and
calibration records; the text summary does not display them. Policy extraction
from arbitrary outer pipelines is outside this direct-estimator reporting scope.

### Prediction policies during evaluation

`evaluate_classifier` preserves the supplied estimator's label and score
policies. `predict_with_probabilities` is preferred when available.
`predict_with_decision_function` is used only when `predict_proba` is unavailable;
otherwise, available probabilities take precedence over decision margins for
ranking metrics. Explicit combined methods keep their own label contract.
The package does not replace a custom decision rule with probability argmax.

Eligible pipelines with inherited prediction methods share preprocessing
once, including subclasses that customize only fitting. Custom outer
prediction or score methods use the supplied pipeline's public APIs.
Within eligible pipelines, custom head prediction methods are called
natively. Sharing preprocessing does not reproduce the side effects of
two separate transformations.

For generic adapters, stock KNN may reuse probability votes when its native
decision is safely preserved. Uncertain distance-weighted votes use native
prediction on the whole coordinate batch. Callable weights/metrics and explicit NumPy
error or callback policies retain native head operations. The same
numerical-policy restriction applies to eligible linear-SVM margin reuse.
These checks add no public option and do not change fitted thresholds.

Warnings describe the work actually performed. Invalid scores still
produce a failed report without predictions or metrics. Under the default
numerical policy, warning order or multiplicity for invalid stock scores
can differ from separate public prediction and scoring calls. Reuse can
avoid duplicate warnings from calculations performed only once. Native
fallback for uncertain votes and custom policies can cost more than an
unqualified shortcut; ordinary qualified inputs retain shared work.

For SVM, `evaluate_classifier` uses real decision margins for AUROC and average
precision. The report sets `ranking_score_type="decision_margin"`, stores the
scores in `predictions["positive_class_decision_margin"]` and leaves
`predictions["positive_class_probability"]` as `None`. With a probability head,
the score type is `"probability"` and the margin field is `None`. Ranking
scores always refer to `classes_[1]`; class weights do not make SVM margins
calibrated probabilities. The summary states when ranking metrics use margins
rather than probabilities.

The commands below exercise a paired development comparison:

```powershell
python examples/compare_simulation.py --quick --output runs
python examples/compare_simulation.py --scenario xor --seeds 1 2 3 --output runs
python examples/compare_simulation.py --quick --classifier linear_svm --output runs
```

`--quick` is a correctness smoke test. The ordinary grids are starting points
for tuning. Use `--grid-json` for expanded sklearn-style parameter grids keyed
by model name, and record the reason for each expansion.
Refinement diagnostics flag boundaries and interior intervals; they do not
automatically certify that tuning is complete. Neither command freezes defaults.

For numeric selections, Python and NumPy booleans are excluded from the
reported numeric range. If the declared candidates for that parameter
include an explicit `None`, its numeric diagnostic entry also records
`includes_none: true`, including when there is only one numeric value.
This optional field records a declared alternative, even if it failed
qualification; it does not mean that `None` qualified or won. The field is
omitted otherwise. A selected `None`, boolean or string retains its
categorical diagnostic. Full candidate records remain the source for
qualification and selection evidence.

Set rank, the applicable screening parameter, and classifier parameter in each method's
explicit grid. Tune unscreened NMF independently, and use the same saved folds
for paired method comparisons. `comparison_models()` includes independently
tuned `screened_mu_absolute` and `screened_mu_percentile` alongside the existing
CD, SPA, PCA and screening-only comparisons. The raw KNN, majority-class and
historical unscreened-CD baselines remain available. CD, SPA and MU, and
percentile and absolute screening, remain separate alternatives.
`comparison_models(classifier="logistic")` substitutes scaled logistic
regression for KNN in each representation path, including raw features, and
tunes its inverse regularization strength independently. The ordinary C grid
is `[0.01, 0.1, 1.0, 10.0, 100.0]`; the quick grid is `[1.0]` and is only a
smoke check. Logistic model names have a `_logistic` suffix; the raw-feature
model is `raw_logistic`. NMF estimators use the grid key `logistic_C`, while
the standard sklearn baseline pipelines use `logistic__C`.
`comparison_models(classifier="linear_svm")` provides the same representation
paths with unscaled primal squared-hinge SVMs and a separate C grid.
Names use `_linear_svm`, including `raw_linear_svm`. The grid keys are
`svm_C` for `NMFClassifier` and `linear_svm__C` for baseline pipelines.
The ordinary and quick C grids match those listed above; `svm_max_iter`,
`svm_tol` and `svm_class_weight` configure the SVM numerical/risk policy.
`comparison_models()` retains the original KNN choices.

The simulation and CelebA comparison CLIs expose this head through
`--classifier linear_svm`. Both accept `--svm-max-iter` (default `20000`),
`--svm-tol` (default `1e-5`), and `--svm-class-weight` (`none` or `balanced`,
default `none`, converted to Python `None`). These options apply only when
`linear_svm` is selected and supply constructor defaults; a per-model
`--grid-json` override retains precedence for each candidate. There is no
separate CLI C flag: use the head-specific grid keys above. No scaler or
probability conversion is added.

SVM selections save these three settings alongside the full grid overrides in
`configuration.json`; changing them rejects an incompatible resume. KNN and
logistic selections retain their historical configuration fields, including
when inactive SVM options are supplied. Existing estimator, candidate, data,
source and runtime fingerprint checks also remain in force. A changed grid
does not silently reuse previously completed results.

Set `max_iter` and `tol` to numerically qualify
the intended method; do not select them to exploit unconverged fits.

A completed declared grid is evidence for that grid, not proof that tuning is
finished. Inspect boundary winners and nearby intervals on training CV, record
revised grids where needed, and keep official test evaluation behind the protocol
freeze. The [CelebA study plan](../research/celeba-study.md) describes the four main
parameters and the attribute-specific evaluation requirements.

## Explicit grids with shared representations

```python
from supervised_nmf.study import calibration_knn_grid

report = calibration_knn_grid(
    classifier,
    {"score_threshold": [1.0, 2.0, 3.0], "n_components": [2, 4],
     "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1, scoring="accuracy",
    checkpoint_dir="runs/my-explicit-grid",
)
```

This example also uses the absolute-screening `classifier` configured above.
The function evaluates the declared grid and returns a JSON-compatible report,
with provisional CV ranking and grid-refinement diagnostics. It performs no
full-training refit, validation evaluation, majority fallback, or default
selection. All required folds must qualify for a candidate to be ranked.
The `scoring` choice and generic/metric-specific score fields follow
[the calibration report contract](#selection-objective-and-report-fields).

For fixed representation parameters and a fixed training fold, it fits the
representation once and reuses the exact training coordinates and validation-fold
projection across neighbor counts. A shared representation failure invalidates
every associated neighbor count; a KNN-specific failure invalidates that
candidate. Computational accounting counts the shared fit once. Representation
random states must be fixed integers. Supported estimators are `NMFClassifier`,
raw `KNeighborsClassifier`, and standard sklearn pipelines ending in KNN;
other KNN settings stay fixed within this grid.
An `NMFClassifier` using logistic regression or linear SVM is not supported by
this helper; use `calibration_search` instead. For SD KNN, the fitted training-fold
scaler is reused across K values and applied once to each validation representation.
`knn_normalization` stays fixed within one shared grid; changing the metric
requires a separate declared comparison.

Version 0.3.0 also shares Euclidean neighbor queries among
compatible fitted KNN heads. Callable weights, other metrics, sparse inputs,
unknown backends and uncertain distance/vote boundaries use each selected-k
public estimator on the whole original query batch. This internal adapter
does not add a user setting or change tie rules. Its sklearn fitted-state
compatibility is checked conservatively.

Current source checks the complete shared distance matrix for finite values
once per immutable group. Under the ordinary NumPy error policy, it checks
neighbor boundaries before vote arithmetic, reuses votes for requested counts
with clean boundaries, and avoids repeating completed boundary qualification.
Nondefault NumPy error policies retain the original vote/guard order. Optional
vote-cache construction failures use the original per-count calculation.
Weighted accumulation can differ in its final floating-point bits; the existing
vote and threshold guards still select native execution on uncertainty.

When an early boundary check selects public fallback, per-head
`neighbor_query_reuse` diagnostics set `vote_checks_performed=False` and leave
`near_vote_rows`, `near_threshold_rows`, `invalid_vote_rows`, `vote_guard` and
`uncertain_rows` as JSON `null`. These fields were not computed.
`near_boundary_rows` records the observed boundary count, and
`uncertainty_lower_bound_rows` is a lower bound on the total uncertain rows.
Report consumers must not interpret an uncomputed count as zero.

This reduces repeated qualification and vote work; a duplicate-heavy batch
can still pay for a shared query followed by native per-count calls on the
whole original query batch. It does not guarantee faster complete calibration:
small or sparse grids and full workflows retain measured costs and variability.
The [development record](../research/engineering/development-protocol.md#guarded-neighbor-reuse-in-explicit-grids)
describes the decision and its limits. Use a new checkpoint directory after a
source change; historical checkpoints are not migrated to the new identity.

Report schema 3 adds `knn_reuse_setup_seconds` and `knn_shared_query_seconds`
to `compute_seconds`. Shared queries are charged once per fold/backend group;
per-head `prediction_seconds` records voting, guards and any public fallback.
A completed resume retains historical compute totals and performs no new
representation fits or neighbor queries. Those totals are not resume runtime.
Source and schema identities prevent reuse of old checkpoints under this change.

Shared KNN calibration uses the training coordinates from the original fit. If
scikit-learn does not expose usable fitted coordinates, calibration raises
`RuntimeError` and leaves the affected fold available to retry. Completed folds
can be reused once the coordinates are available under the same source and
runtime identity. The shared grid fits the pipeline once per representation
and fold using `n_neighbors=1`. It requires one final KNN fit on the fold
labels after preprocessing. On validation rows it calls each active
intermediate step's `transform` in prediction order. An intermediate step
can supply `fit_transform` and `transform` without a separate `fit` method.

A scikit-learn fit callback can inspect the neighbor count and change only
one candidate's fit. The shared grid checks for `set_callbacks` registrations
after applying each candidate's parameters, including step replacements and
setters. If it finds a registration, it raises `RuntimeError` before fitting
or accessing its checkpoint directory.

For configuration instances, the check follows the values of the dictionary
returned by public `get_params(deep=False)`. This also applies to containers
and NumPy subclasses with additional parameters, and to objects without a
`fit` method. If the method rejects `deep=False` and its signature confirms
that the keyword is unsupported, the check skips it. If the signature is
unavailable or a compatible method raises an error, the error propagates.
This prevents an inspection failure from silently allowing checkpoint reuse.
The check also follows the references below.

- Members of lists, tuples (including named tuples), sets, frozensets, and
  deques. For dictionaries it follows keys, values, and a `defaultdict`'s
  factory. It also reads mapping proxies over finite, side-effect-free mappings,
  the backing mappings of `ChainMap`, and the `data` of `UserList` and `UserDict`.
- Object references in NumPy arrays, scalars, and dtypes, and the components of slices.
- Defaults and closures of Python functions, owners of bound methods, and the
  callable and arguments bound by `functools.partial`.

Use `calibration_search(..., refit=False, checkpoint_path=None)` with the same
folds to run the complete pipeline independently for each neighbor count. Use
it when a callback is reachable only through another container, opaque callable,
global, class or module namespace, or undeclared object attribute, or when user code installs callbacks
during fitting, transformation, or prediction.

Checksummed fold records can be reused only with matching data, folds, grid, scoring,
estimator, source and runtime identity. A report describes the declared grid and
does not certify completed tuning. Record subsequent grid decisions and numerical
qualification before freezing the final study.

### Full-training neighbor counts

For `weights="uniform"`, choosing `n_neighbors` equal to the number of training
rows in a fold makes every training row contribute one vote. The resulting
class probabilities are that fold's class frequencies. This is a legal grid
endpoint, not a recommendation to choose it: select neighbor counts through
the declared training-only comparison. The full-training count can differ
between folds; a count exceeding a fold's training size remains invalid.

Eligible dense Euclidean grids can reuse this result without a redundant
full-size prediction query. In the fixed shared-grid call sequence, a brute
query can request only enough neighbors to serve the smaller counts and their
boundary checks. A sole full-size head still performs a real one-neighbor
query before producing the full-training votes. Native input validation,
probabilities, class ordering, tie decisions and explicit thresholds are
preserved. The implementation does not replace query validation with an
unconditional class-frequency result.

This internal optimization has no public switch. It requires qualified native
methods, numeric state, error policy and callback behavior. Among supported
version pairs, only scikit-learn 1.9.1 with Joblib 1.6.0 qualifies for the
full-membership shortcut. The checks compare the live classifier and its
direct operations with the source in the installed distribution. Other pairs,
including scikit-learn 1.9.0 with Joblib 1.6.0, can still share an ordinary
full-width neighbor query when eligible. Otherwise, each classifier makes
its own prediction at its selected neighbor count.
Custom backends or methods, unsafe coordinate magnitudes and distance-weighted
heads do not acquire this shortcut merely because their neighbor count equals
the training size. With BallTree, each fitted head makes its own prediction at
the selected neighbor count. Widening that search can change which branches
the tree visits. In the ordinary serial Joblib configuration (`n_jobs=None`
or `1`), KDTree heads fitted by the grid on the same numeric array can
share a full-width query. A head configured for multiple workers queries its
selected width so its scheduler receives the expected calls. Pipeline
calibration fits each neighbor classifier before preprocessing the validation
fold, as an independent pipeline fit would. If validation changes the stored
coordinates used by the fitted heads, each KDTree queries its selected neighbor
count using its fitted index. If a pipeline provides a prebuilt neighbor
index during training, calibration fits
each candidate from that index. This preserves the distance metric stored in
a KDTree or BallTree. Each KDTree head fitted from an index queries its
selected neighbor count. In an explicit Joblib context, with a custom dispatch
backend, or when calibration cannot verify the functions that schedule a
KDTree query, each classifier predicts at its selected neighbor count. This
preserves warnings and errors specific to that count. When instrumentation
needs to observe every candidate's native prediction, use `calibration_search`
with the same folds to fit and query the candidates independently. Grids without a
successfully fitted full-size head skip this extra planning setup.

Per-head `neighbor_query_reuse` records can identify the vote basis as
`native_query_training_histogram` or `short_query_training_histogram`.
When full-membership proof resolves an otherwise uncertain vote, the record
sets `full_membership_certified=True` and
`reason="uniform_full_membership_certified"`; the original uncertainty counts
remain available. Those fields do not report a new fit or a changed tie rule.

Fold `neighbor_queries` records describe attempted shared-query widths in
`shared_query_neighbor_counts`. A failed optional short attempt can be followed
by the original full shared query or the selected head's public prediction.
The records retain `short_query_attempt_seconds`, attempt warnings/errors and
`short_query_upgrade_reason` when applicable. Shared `query_seconds` already
includes each shared attempt: do not add `short_query_attempt_seconds` again.
Native per-head fallback remains charged inside that head's prediction time.

Complete-call benefits depend on whether this endpoint occurs and how much
neighbor work it saves. Ordinary grids, guarded fallback and completed resumes
can retain overhead; this is not a speedup for every input. Completed resumes
perform no new queries, retain historical compute totals and still validate
source/runtime identity. Use a new checkpoint directory after a source change.
The [development decision](../research/engineering/development-protocol.md#uniform-knn-full-training-membership)
records the measured benefit, retained costs and compatibility tradeoff.

Recognized NMFClassifier and screening/PCA/KNN layouts also share training-fold
screening statistics across cutoff and rank settings. The report counts
`screening_statistics_seconds` and `screening_selection_seconds` separately.
Empty screens and impossible ranks remain explicit failed candidates with a
`failure_stage`; no threshold or rank is silently substituted. Generic pipeline
layouts retain their ordinary fit path when at least one requested neighbor
count is feasible for the fold.

For each representation and fold, if every requested `n_neighbors` exceeds the
actual training-fold size, the helper records a failed fold with
`failure_stage="neighbor_feasibility"` and `preflight_rejected=True` before
screening, fitting or validation projection. Skipped operations have zero
recorded work and no fitted diagnostics or warnings. This neighbor-count failure
takes precedence over errors or warnings that those unexecuted operations
might have produced. A count equal to the training-fold size remains allowed.

The declared candidates and all fold records remain in the report. A group with
at least one feasible neighbor count still fits its shared representation. If
the representation is valid, oversized counts fail individually. If it fails
or does not converge, its outcome takes precedence for every count in that
fold. Any head fits already performed retain their recorded time. With unequal
fold sizes, feasible folds
still execute and retain their diagnostics even if a candidate fails elsewhere.
A candidate must be valid on every fold to qualify; successful folds are never
averaged into a valid score for an otherwise failed candidate. Completed failure
receipts follow the ordinary source-bound resume rules. This check is specific
to `calibration_knn_grid`; generic search does not infer a training-size rule
from an arbitrary estimator parameter named `n_neighbors`.

### Execution order for restartable grids

`calibration_knn_grid(..., execution_order="representation")` is the default:
it evaluates all folds for one representation group before advancing to the
next group. Set `execution_order="fold"` explicitly to evaluate all representation
groups within a fold before advancing to the next fold, when the grid qualifies:

```python
fold_report = calibration_knn_grid(
    classifier,
    {"n_components": [2, 4, 8], "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1,
    execution_order="fold",
    checkpoint_dir="runs/my-fold-first-grid",
)
print(fold_report["execution_order"])
# {"requested": "fold", "resolved": "fold", "fallback_reason": None}
```

This example uses the screened `classifier` with fixed integer randomness from
above. Fold-first execution is intended for long, restartable grids with
multiple ranks or other representation settings. Completing those settings
within each fold can reduce repeated screening after partial resumes. It
changes intermediate progress: individual candidates can take longer to have
all their CV folds complete. Candidate selection, tie-breaking and final report
records retain canonical representation-then-fold order.

The fold-first option applies only when every representation group qualifies:
an exact built-in `NMFClassifier` with screening enabled, or a recognized exact
screen/KNN or screen/PCA/KNN pipeline, with plain constructor values, fixed
integer randomness and supported built-in KNN settings. Subclasses, custom
transforms, callable weights or metrics, estimator-valued hooks, and pipeline
memory or metadata hooks are outside this gate. Otherwise accepted
configurations keep the original order when they do not qualify; this option
does not bypass ordinary estimator or search validation.

The report's `execution_order` dictionary records `requested`, `resolved` and
`fallback_reason`. The same dictionary is saved under `execution_order` in
the payload of the checkpoint's `identity.json`. A successful opt-in has `resolved="fold"`
and `fallback_reason=None`. For a fold-first request, a single representation
group resolves to `"representation"` with
`fallback_reason="single_representation"`; an unsupported configuration resolves
to `"representation"` with
`fallback_reason="unsupported_configuration"`. The default request resolves
to `"representation"` with no fallback reason. Changing the requested order
requires a new checkpoint directory, even when both requests would resolve to
the same order. Only `"representation"` and `"fold"` are accepted values.

This option changes scheduling only; it does not itself persist screening
statistics or change the fold-receipt commit boundary. Compute totals add
historical committed work
to newly committed work. Screening actually repeated after a restart is new
computation and is counted once when its fold receipt commits; interrupted
attempts without committed receipts remain outside those totals. A complete
resume already performs zero screening or representation fits. A single
representation offers no scheduling advantage. Benefits depend on the grid and
interruption history; short workflows can be slower, and no universal runtime
improvement is promised.

### Early recovery of screening statistics

`calibration_knn_grid(..., screening_statistics_recovery="off")` keeps the
default receipt-based restart behavior. Set `screening_statistics_recovery="early"`
to save eligible training-fold screening statistics before the corresponding
representation fit and fold receipt complete:

```python
report = calibration_knn_grid(
    classifier,
    {"n_components": [2, 4], "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1, scoring="accuracy",
    checkpoint_dir="runs/my-recoverable-grid",
    screening_statistics_recovery="early",
)
recovery = report["screening_statistics_recovery"]
```

Use this opt-in for restartable screened grids when preserving statistics from
an interruption before a fold receipt is useful. It can help even with one
representation; it is independent of `execution_order`. It does not save a
fitted NMF/PCA representation or avoid its unfinished fit. Ordinary successful
runs pay extra serialization, metadata and source/runtime-guard costs; completed
resumes still check metadata. Neither setting promises faster total execution.
Only `"off"` and `"early"` are accepted. The requested policy is part of checkpoint
identity, so changing it requires a new checkpoint directory.

Persistence is limited to recognized exact screened `NMFClassifier`, screen/KNN
and screen/PCA/KNN layouts with ordinary parameters, fixed integer randomness
and supported built-in KNN settings. Custom subclasses/transforms, pipeline
memory or metadata hooks, callable metrics/weights and other unsupported
configurations retain their ordinary computation. Early reuse also requires
default NumPy error handling without a callback and standard finite checks;
statistics that emitted warnings are not published. Unsupported fitted-state
encodings also retain ordinary computation. In particular, object-dtype
class arrays containing NumPy string scalars bypass persistence with
`codec_unsupported`; object arrays whose elements are exact Python strings
can be persisted and restored when the other eligibility conditions hold.
The report records `eligible`, `bypass_reason` and current-call
bypass/publication information.
Existing checkpoint and metadata validation still apply when reuse is bypassed.

Each committed event has immutable JSON accounting metadata and a separate
learned-array payload. A successful statistics event can survive an interruption
before its representation receipt, and a later call can restore it. With valid
metadata, an unavailable or invalid learned payload can be recomputed as a new
event; the old event, consumers and recorded cost remain unchanged. Allocation
failures and invalid accounting authority are not silently treated as cache misses.
Missing or invalid metadata is an error. In particular, all expected existing
receipt references are checked under the checkpoint lock before new fold attempts
or fitting, even when an earlier receipt is missing. Input validation and runtime
initialization still precede this preflight. This option does not repair lost accounting
authority. A completed-only call validates metadata without opening the learned
payloads; that is not a claim that unused payload bytes were verified.

The final JSON rename commits the event. Recovery relies on atomic renames
within the checkpoint directory and on subsequent file lookups seeing their
outcome. On Windows, a permission error during a rename can trigger up to
five retries, with up to 1.55 seconds of scheduled waiting per file. If a
failed JSON rename leaves its destination present or its status uncertain,
calibration stops before saving the affected fold receipt. It also stops if
it cannot record the link between a committed event and its fold. Restart
with the same checkpoint directory so calibration can verify the saved event
and account for the fit once. Temporary files and payloads without committed
JSON are retained as interruption evidence and ignored on restart.

If calibration cannot record the time spent reading a checkpoint or complete
the bookkeeping for a restored screening event, it raises
`StatisticsRecoveryError`. Restart with the same directory to retry the read.
If the read is already aborting, its original error or interruption propagates.

Original `compute_seconds`, `tuning_seconds`, `screening_statistics_fits` and
fold work flags retain their committed-receipt meaning. The additional
`screening_statistics_recovery` report records each unique event and its
consumers. Its `union_statistics_seconds` and `union_tuning_seconds` remove
receipt-backed event costs once before adding all durable events, including
events without a completed consumer. They are alternative historical accounting
totals, not extra costs to add to the original totals. `current_call` separately
records actual statistics fits, restored/published events and recovery timers.
Interrupted attempt records remain separate evidence of uncommitted work.
These overlapping/historical clocks are not a wall-time partition.

Data, grid, fold, estimator, policy, source and runtime identity must still match.
No cross-source checkpoint migration, generic estimator serialization or
power-loss durability is promised. The [development decision](../research/engineering/development-protocol.md#early-recovery-of-screening-statistics)
records the tested benefits, costs and recovery boundaries. This option belongs
to `calibration_knn_grid`; `calibration_rank_search` retains its existing default
`off` calls and does not expose it.

## Reconstruction diagnostics

For large, low-rank float64 fits, report-only reconstruction norms can use
128-row blocks. The input shape must be 8,192–32,768 rows by 1,024–4,096
features, with rank 1–32 and matching coordinate/component shapes. The runtime
and all three arrays must satisfy one of these configurations:

- All detected BLAS pools are one-thread MKL, and the arrays are C- or
  F-contiguous.
- On Windows, exactly one BLAS pool is detected as AOCL/BLIS with prefix
  `aocl-libblis` and version
  `AOCL-BLAS 5.3.0 Build 20260511`, using one thread, and all arrays are
  C-contiguous.
- On Windows, exactly two BLAS pools are detected: one MKL and one AOCL/BLIS
  with prefix `aocl-libblis`, version `AOCL-BLAS 5.3.0 Build 20260511`, and
  architecture `zen3`. Both use one thread, all arrays are C-contiguous, and
  both the observation count and feature count are divisible by eight.

Larger fits can also use blocks. The extended range covers 8,192–162,112 rows
and 1,024–4,608 features, up to 162,112 × 4,096 matrix elements and rank 1–32.
It applies when either dimension exceeds the original range. All three operands
must be plain, contiguous float64 NumPy arrays, and NumPy must be using its
default floating-point error policy.

- On Linux with one single-threaded MKL pool, a C-contiguous input uses
  128-row blocks. A column-major input uses blocks of features.
- On Windows, the two-pool MKL and AOCL/BLIS configuration described above
  uses 32-row blocks. The layout and alignment requirements still apply.

The check uses the thread counts in effect when the fit reports its error,
after projection thread controls have ended. On Windows the mixed-pool rule
reflects the tested NumPy AOCL configuration. The inventory alone cannot
identify which library NumPy uses. Screened arrays can be column-major and
then use the full product on this mixed runtime.

Blocking uses the supplied layout directly. Other shapes, layouts and runtimes
use the full product. In the extended range, extreme or nonfinite factor
values, an unsafe block calculation, or a large combined norm also trigger
the full expression. This fallback can require full-size temporary storage.

The result is the Frobenius norm of the reconstruction residual. Changing
the block shape can change floating-point rounding, especially for a nearly
exact fit. The estimator computes this diagnostic after fitting. Optimizer
steps, stopping decisions and predictions use the fitted factors directly.

## Bounded rank exploration

`supervised_nmf.search.calibration_rank_search` explores a rank parameter
in a supported KNN estimator. Like `calibration_knn_grid`, it requires
`checkpoint_dir` and accepts `scoring="accuracy"` or `"balanced_accuracy"`.
The shared grid also accepts `execution_order` and
`screening_statistics_recovery`. Bounded rank search uses their defaults
internally. Its stages, winner comparison, and refinement decisions
use that same selected objective. It adds `rank_parameter=None`, `max_expansions=2`, and
`max_rank=None`. The rank parameter is inferred only when exactly one
`n_components` grid parameter is declared. An upper-boundary winner may be
tested at doubled rank with its other parameters fixed. New ranks are bounded
by known fold-local dimensions and an optional user cap. Original grid entries
are preserved, including infeasible ones.

Bounded search accepts positive integer ranks, including custom integral
values that can be cloned, identified for checkpointing, and fitted by the
estimator. The initial estimator must expose the parameter name. Each grid
entry must assign it to the package's NMF classifier or to a PCA, NMF, or
FeatureSPA pipeline step, including when the grid replaces the step.
Boolean values and NumPy durations are invalid for ranks and expansion limits.
For a custom integral value, Python indexing and integer conversion must
agree when both are available. PCA uses indexing to slice components, while
other supported estimators can use integer conversion to store their rank.

Candidate records retain the declared parameter display. Rank comparisons and
refinement bounds use the numerical values, including declared ranks that
exceeded a fold's dimensions. For example, if rank 1 qualifies and rank 9
does not, the report retains both bounds and marks the upper rank unresolved.
When the grid includes two PCA steps, for example, `rank_parameter` selects
which step to expand. The final refinement report keeps the other step's
range from the full initial grid, including when a later stage wins.
In a standalone shared grid, a parameter name used by both a factorizer and
an unrelated transformer is reported as categorical. Its settings do not
share a numerical rank scale.
When reporting a supported representation, a setting with no usable numeric
coordinate retains its fit result but does not supply a numeric bound. A
selected setting of this kind is reported as categorical. Finite extended-
precision PCA fractions retain their precision in the report. Checkpoint
reuse requires the same scikit-learn `skip_parameter_validation` setting:
changing it can make a previously failed PCA setting eligible.

The returned report contains each stage, its candidates, the selected settings,
`scoring`, `selected_mean_score`, `stop_reason`, and
`unresolved_upper_rank_boundary`. Exhausted budgets, user
caps, and unresolved failed upper candidates do not certify completed search.
Even when `search_complete` is true for this upper-rank policy,
`tuning_complete` remains false: screening and neighbor choices have not been
jointly reoptimized and no global optimum is claimed. This helper supports the
shared evaluator's KNN layouts and performs no refit or outer evaluation.

An accepted pipeline containing `FeatureSPA` uses fixed-basis NNLS coordinates
for both its training and validation observations. Rank search refits that map
inside each training fold and preserves the fitted coordinates supplied to the
classifier when reusing the representation across neighbor choices. See
[standalone SPA coordinates](#standalone-spa-coordinates).

Run `python examples/grouped_tuning.py --output runs/grouped-example` for a
small synthetic subject-separated example. Calibration, final fitting and
evaluation each use one native numerical thread.

## Local CelebA data

```powershell
python examples/celeba_classification.py --dir PATH/img_align_celeba `
  --attributes PATH/list_attr_celeba.txt `
  --partitions PATH/list_eval_partition.txt `
  --attribute Smiling --quick
```

Replace these example paths with the supplied data location. Attributes may also
be a CSV with an image ID column. Empty CSV data records (zero fields) after a
valid header are ignored. Whitespace-only CSV records, quoted empty/whitespace
fields and delimiter-only empty fields retain their existing validation; they
are not treated as empty records. Quoted multiline fields are parsed intact.
Official whitespace text also ignores whitespace-only data lines. A valid
header with no labels still fails, and leading empty records do not replace
the required first header. CSV malformed-row numbers count logical data records
after the header, including ignored empty records, rather than physical lines.

The loader checks IDs, labels, required images and official partition codes.
It samples uniformly within each partition. This development example uses only
training and validation images. It checks that their image IDs are disjoint.
Checking whether the same person appears in both partitions requires a separate
identity audit. The final-test attempt in the original
[default-selection study](../research/celeba-default-results.md) ended before it produced
test predictions. A separate [October 12 official-partition study](../research/validation.md#october-12-official-partition-evaluation)
evaluated held-out test images with identity and exact-image-duplicate exclusions.
