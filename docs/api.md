# API reference

This page describes version 0.4.1, including its temporary AOCL NNLS workaround.
The optional MU per-observation projection policy was added in 0.4.0. Version
0.3.3 introduced optional QR projection and MU numerical recovery. Optional KNN SD
scaling and the linear-SVM head were introduced in 0.3.0. Historical release
and validation records retain their original scope.

## Classifier

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
    weights="uniform",           # sklearn KNN weighting choice
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
    mu_projection="batch",      # MU only: "batch" or "rowwise"
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
[four-method development comparison](nmf-solver-comparison.md) supplies empirical
context, not a universal solver ranking. Rank, cutoff and classifier parameters remain
tunable. `max_iter=20000` preserves that study's numerical allowance for
CD/MU iterations and SPA's public NNLS solves. SPA may first make a bounded
batched attempt, then pass the original cap to SciPy fallback; the cap is not
a shared total-work budget. Batched pivot rounds and SciPy iterations are
different units. Converged routines stop earlier. The tolerance remains `1e-4` for
CD/MU and is not used by SPA's NNLS routine. CD and MU have different stopping
criteria; equal tolerance values do not establish equal numerical accuracy.
Explicit overrides are preserved; `None` is invalid for `solver` or `screening`.

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
than zero selects that class, and equality selects `classes_[0]`. Arbitrary
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

### Prediction dtype

CD and MU preserve float32 training inputs and fitted bases. At prediction,
matching query and basis dtypes retain that output dtype. CD and the default MU
batch policy keep their existing projection calculation; MU's optional rowwise
policy uses float64 working arithmetic with guarded numerical recovery.
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

The fixed-basis Frobenius objective separates by observation, but CD and the
default MU policy, `mu_projection="batch"`, use stopping criteria shared across
the query batch. Batch MU checks the relative improvement in the whole batch's
reconstruction error every ten iterations.
Changing the other observations in a call can change the stopping iteration
and therefore an observation's coordinates and, near a classifier boundary,
its predicted label. This is dependence on batch composition, not random
variation between otherwise identical calls. In batch mode, MU's numerical
recovery paths retain the shared stopping rule.

For MU predictions that may arrive individually or in changing batches, choose
`mu_projection="rowwise"`. Each observation receives its own initializer and
relative-error stopping decision, checked every ten iterations. This retains
the fixed-basis unregularized Frobenius objective and the fitted basis while
changing the numerical projection policy. Float64 working arithmetic reduces
the sensitivity of stopping decisions to float32 rounding. Range guards retain
the existing handling of exact zeros, tiny inputs and extreme values, including
per-observation recovery when needed. The output dtype follows the rules above.

The policy is explicit, never selected automatically from a query batch. It is
recorded at fit time as `mu_projection_` and preserved by saving/loading the
model. Changing `mu_projection` with `set_params` takes effect on the next fit.
Legacy fitted MU models retain `"batch"`. For CD and SPA, `mu_projection_` is
`None`; requesting `"rowwise"` with either solver raises an error at fit time.

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
including when rowwise prediction is selected. Reprojection adds fitting work
and can change classifier predictions. The rowwise policy can also cost several
times more projection time on ordinary workloads; the cost depends on the data
and recovery path. Use batch mode to reproduce the existing convention or when
its throughput benefit is appropriate and batch sensitivity is acceptable.

Rowwise projection removes the shared stopping decision, but does not guarantee
bitwise agreement across batch shapes, convergence at the iteration cap, or
better classification accuracy. Matrix-kernel rounding can still differ.

A smaller `tol` may reduce these differences but does not guarantee identical
results across batch sizes. `tol=0` disables early stopping and uses the
iteration cap; it does not certify convergence or bitwise batch invariance.
Both settings also apply to factorization. Keep prediction batch construction
in reproducibility records. See the
[theory and implementation discussion](theory-implementation.md) for the
distinction between the mathematical projection and its numerical execution.

### Fit diagnostics

`fit_times_`, `n_iter_`, `converged_`, and `convergence_warnings_` expose fit
diagnostics. `n_iter_` remains the factorization iteration count.
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
normal value use logarithmic arithmetic directly. Float64 problems outside the
scaling bounds, ordinary float64 arithmetic failures and failed retries also
use the logarithmic calculation. This preserves tiny input contributions while
avoiding the more expensive path for qualified problems with tiny basis tails.
The recovery paths retain the MU update equations, uniform initialization and
relative-error stopping rule for the observations supplied to the recovery call.
Rowwise mode invokes this recovery separately for affected observations. The
paths preserve the fitted basis and return the projection dtype described above;
the float32 recovery path converts back only after solving. Nonzero inputs are
not replaced with zeros; a coordinate below the output dtype's representable
range can still round to zero. Unrepresentably large coordinates raise an error.
In the recovery paths, the rank-one closed-form first MU update supplies the
final coordinates; subnormal outputs use exact input ratios to resolve
round-to-nearest ties.
Recovery can cost more time and can change rounding and stopping iterations
relative to the ordinary calculation. An exhausted positive-tolerance iteration
cap still emits a convergence warning. Each retry keeps the configured cap and
tolerance; a failed attempt followed by recovery can do more total work than a
single attempt.
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

### Prediction state and diagnostics

In current source, standard `NMFClassifier` prediction preserves learned arrays
and package-owned fitted state. Projection uses a shallow local decomposition
object for execution settings and temporary bookkeeping; it shares the learned
basis rather than copying it. Existing dtype recovery still makes numerical
copies when needed. Changes to projection execution controls between calls take
effect without updating the stored decomposition during prediction.

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

Version 0.3.0 defaults to `training_coordinates="auto"`.
The 0.2.0 wheel defaults to `"fit_transform"`.
For SPA, this resolves to `"transform"`: after fitting the nonnegative basis,
the classifier re-encodes its training observations by fixed-basis NNLS.
Training and prediction therefore use the same fitted basis and projection map.
For CD and MU, `"auto"` resolves to `"fit_transform"`, retaining the coordinates
returned by factorization. Selecting `mu_projection="rowwise"` does not alter
this resolution. For a common MU training/query projection policy, also select
`training_coordinates="transform"`.

An explicit `"transform"` re-encodes training observations for any solver.
An explicit `"fit_transform"` retains the previous convention, including SPA's
anchor-derived training coordinates. Training reprojection adds computation to
`fit_times_["training_projection"]` and total fit time. It can change a retrained
classifier's predictions and accuracy; this correction does not guarantee an
accuracy improvement. Report the coordinate policy alongside the solver when
comparing fits or reproducing historical results.

`reconstruction_err_` refers to the classifier's actual training coordinates;
`factorization_reconstruction_err_` preserves the initial factorization diagnostic.

### Fitting and transforming together

`NMFClassifier.fit_transform(X, y)` fits the complete supervised classifier,
including screening, factorization and the classification head, then returns
`transform(X)`. Labels are required. The returned array contains projected,
unscaled NMF coordinates; it can differ from `training_coordinates_` when the
head was trained on factorization or anchor coordinates.

The `fit_transform` method and the `training_coordinates="fit_transform"`
parameter have different roles: the method returns projections after fitting,
while the parameter selects the coordinates used to train the head. Calling
the method preserves that policy and adds a projection after `fit`; this
additional projection is outside the recorded fit time. Use `fit(X, y)` when
only a fitted classifier is needed.

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
Existing serialized models keep their stored basis; refitting uses this
implementation. No constructor parameters or fitted-state fields were added.

### SPA projection execution

AOCL environments automatically use a package-side dimension-padding workaround
for explicit matrix products in SPA's NNLS basis and projection solves.
MKL-only environments retain direct multiplication. This preserves the NNLS
objective and its numerical checks without changing the installed AOCL library.
This is a temporary compatibility measure. Padding will be bypassed for verified
AMD-fixed runtimes and removed when the supported AOCL builds no longer require
it; unqualified versions retain the safeguard. See the
[implementation, evidence scope and retirement policy](aocl-nnls-workaround.md).

These controls apply to both `FeatureSPA` and `NMFClassifier(solver="spa")`.
They affect fixed-basis projection, including training reprojection when
`training_coordinates="transform"` or when `"auto"` resolves to it. They do not
control anchor selection or basis fitting. CD/MU ignore valid SPA projection
controls.

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

QR factors are local to one scaling group within a call. They are not cached
in the fitted estimator or reused across prediction calls. Sharing a tall
reduction can help a large query batch, but reduced per-row solves and
qualification can also cost more than the current batched path. Small changes
in coefficients from floating-point arithmetic are possible. Select the solver
using the complete fit/prediction workload; QR is not a general speed or
accuracy improvement. The default remains `"auto"`, and this choice does not
change SPA basis fitting.

Input magnitude is checked before every projection solver.
Ordinary-range inputs retain their solver path. Extreme inputs use exactly
reversible common binary scaling of the design and each independent RHS;
rows with the same scale share execution. The scale preserves the NNLS
minimizers and does not change conditioning. Nonzero scaled values must have
binary exponents within [-450, 450]; an input span that cannot meet this
conservative bound raises `FloatingPointError` before solving any scale group.
Unsupported range does not mean the mathematical problem has no solution.
Original iteration caps and reference-solver errors still propagate.

Blocks contain at most `projection_batch_size=256` rows. With
`projection_workers>1`, queries of at least 512 rows and two full blocks use
joblib's reusable loky process pool. Smaller queries execute in the caller.
Compatible calls reuse workers automatically, including across models; no
pool or cached fitted basis is serialized with an estimator. First use pays
startup cost, and joblib retires idle workers. `projection_native_threads=1`
limits numerical threads per projection process and restores previous limits
afterward. The controller discovers NumPy/SciPy libraries at first NNLS or
projection use, which can occur during basis fitting.
Some native thread-limit controls affect shared process state; use process
workers for concurrent work.
QR admission applies separately inside each worker and scaling group. A worker
chunk or final chunk with fewer than 32 rows can therefore use the current
solver even when the full query batch is large. Splitting a query also repeats
QR setup in each eligible worker call.

Execution controls may change after fitting, for example
`model.set_params(projection_workers=8)`, without refitting. Outer CV scheduling
remains the caller's responsibility: budget approximately
`CV workers * projection workers * native threads` simultaneous numerical
threads. For many concurrent CV fits, keep projection workers at one.
All counts must be positive integers; `-1` is not supported.
Older serialized models restore the historical SciPy projection setting and
one-worker defaults.

### Saved decisions

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

### Screening and anchor details

The two estimators retain their existing screening defaults:

| Estimator | Default screening | Active selection parameter |
| --- | --- | --- |
| `NMFClassifier()` | `"absolute"` | `score_threshold=2.0`, an inclusive cutoff on the absolute pooled-t score |
| `BinaryScreening()` | `"percentile"` | `percentile=10.0`, using the percentile and tie rules below |

Set the mode explicitly when composing a standalone screen:

```python
from supervised_nmf import BinaryScreening

percentile_screen = BinaryScreening(screening="percentile", percentile=10.0)
absolute_screen = BinaryScreening(screening="absolute", score_threshold=2.0)
```

Changing `percentile` or `score_threshold`, including through `set_params` or
a parameter grid, does not switch `screening`. Both values must be valid, but
only the parameter for the selected mode affects feature selection. For example,
`BinaryScreening(score_threshold=3.0)` still uses percentile screening.

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
absolute scores. Percentile ranking uses the original feature count as its
denominator and follows deterministic feature-index tie breaking. Degenerate
features and empty selection are explicit; near-zero variance is not silently
regularized.

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
and [retrospective diagnostic](wang-simulation-validation.md#october-20-anchor-and-information-diagnostic).

**Pipeline caveat:** a generic sklearn pipeline containing `FeatureSPA` trains
its next step on anchor columns but predicts using NNLS coordinates. These can
differ materially on noisy data, including when predicting the training rows.
The standalone API intentionally retains the factorization's anchor coordinates:

| Operation | Output |
| --- | --- |
| `spa.fit_transform(X_train)` | Original-scale selected columns, also stored in `fit_coordinates_` |
| `spa.fit(X_train).transform(X_train)` | Training observations projected by NNLS against the fitted `components_` |
| `spa.transform(X_query)` | Query observations projected against that same fitted basis |

The basis is fitted with the anchor coordinates held fixed. Projection instead
holds the basis fixed and solves for coordinates. The two operations need not
return the same training coefficients. Exact minimization of the latter problem
cannot increase reconstruction error against that basis, but does not guarantee
better classification. Numerical tolerances still apply. Standalone
`reconstruction_err_` describes the anchor-coordinate factorization; calling
`transform` does not replace it or `fit_coordinates_` with projected results.

For the supported classification heads, use `NMFClassifier(solver="spa")` with
the default `training_coordinates="auto"` or explicit `"transform"`. Both use
fixed-basis projection for classifier training and prediction. Explicit
`training_coordinates="fit_transform"` retains the historical anchor convention.
These classifier choices do not change generic pipelines automatically.

When composing screening, SPA and a classifier manually, project both sets of
observations. For example, with nonnegative training and validation arrays:

```python
from sklearn.neighbors import KNeighborsClassifier
from supervised_nmf import BinaryScreening, FeatureSPA

screen = BinaryScreening(screening="absolute", score_threshold=2.0)
X_train_selected = screen.fit_transform(X_train, y_train)
spa = FeatureSPA(n_components=2).fit(X_train_selected)
Z_train = spa.transform(X_train_selected)
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

For a staged training-only workflow, start with the
[quickstart tuning guide](../README.md#training-only-tuning) and pass
`refit=False` to `calibration_search`. That mode returns `estimator=None` plus
the CV report; it does not fit a final model or access final evaluation data.
Review numerical qualification and boundary/refinement diagnostics, complete
the declared training-only search, and freeze settings before final evaluation.
Constructor defaults and the small grids below are illustrative starting values.

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
    X_train, y_train, cv=5, seed=42, threads=1,
    checkpoint_path="runs/my-development-search.json",
)
if calibrated.report["is_fallback"]:
    raise RuntimeError("Calibration returned a majority fallback, not a qualified NMF model.")
validation = evaluate_classifier(calibrated.estimator, X_validation, y_validation)
```

The cutoff grid above uses the absolute-screening `classifier` configured at
the start of this page. Suitable ranges depend on the dataset. Any
failed or nonconverged required fold disqualifies its candidate. The report
retains each attempt, parameters, warnings, folds, selected dimensions, and
timings. If every candidate fails, a clearly labeled training-majority fallback
is returned when `refit=True`. It must not be reported as the requested method's
successful fit. With `refit=False`, no fallback is fitted and
`selected_parameters=None` identifies the absence of a qualified candidate.

For logistic regression, pass `NMFClassifier(classifier="logistic", screening="absolute")` and replace
`n_neighbors` with `logistic_C`, for example `[0.01, 0.1, 1.0, 10.0, 100.0]`.
Each fold fits its own screening, NMF, scaler and classifier. A logistic
convergence warning disqualifies a candidate just as a factorization warning
does; raising the logistic cap requires a recorded numerical configuration.
For linear SVM, use `NMFClassifier(classifier="linear_svm", screening="absolute")` and the grid key
`svm_C`. Its cap and tolerance qualify the primal solve; they are not substitutes
for tuning C. SVM warnings are retained and disqualify candidates in the same
way. Optional KNN SD scaling is fitted separately within each training fold.

Checkpoints can resume only when data, parameters, versions, folds and package
source hashes match. Changing code or settings requires a new run. The source
examples create unique run directories and expose `--resume` for compatible runs.

Both `calibration_search` and `calibration_knn_grid` accept `groups`, `folds`,
and `data_ids`. `groups` keeps rows from a subject together. `folds` supplies
exact training/validation index pairs and must contain `cv` exhaustive splits;
each validation row appears once and training indices are its complement.
Every validation fold must contain both classes, and every training fold needs
at least two observations per class. Supplied IDs and folds are validated and
bound to checkpoints. See [tuning improvements](tuning-improvements.md).

Evaluation metadata is optional, but independence is checked only when the
corresponding metadata is supplied:

```python
from supervised_nmf.evaluation import format_evaluation_summary

validation = evaluate_classifier(
    calibrated.estimator, X_validation, y_validation,
    sample_ids=validation_ids, train_sample_ids=training_ids,
    groups=validation_subjects, train_groups=training_subjects,
    train_y=y_train,
)
print(format_evaluation_summary(validation))
```

Training/evaluation IDs and subjects must not overlap. Training labels determine
the majority-class reference; evaluation labels never choose that class.
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
Numerical-phase diagnostics on `NMFClassifier` now include
`factorization_converged_` and `training_projection_converged_`, with their
corresponding `*_convergence_warnings_` attributes.

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
```

`--quick` is a correctness smoke test. The ordinary grids are starting points
for tuning. Use `--grid-json` for expanded sklearn-style parameter grids keyed
by model name, and record the reason for each expansion.
Refinement diagnostics flag boundaries and interior intervals; they do not
automatically certify that tuning is complete. Neither command freezes defaults.

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
Set `max_iter` and `tol` to numerically qualify
the intended method; do not select them to exploit unconverged fits.

A completed declared grid is evidence for that grid, not proof that tuning is
finished. Inspect boundary winners and nearby intervals on training CV, record
revised grids where needed, and keep official test evaluation behind the protocol
freeze. The [CelebA study plan](celeba-study.md) describes the four main
parameters and the attribute-specific evaluation requirements.

## Explicit grids with shared representations

```python
from supervised_nmf.study import calibration_knn_grid

report = calibration_knn_grid(
    classifier,
    {"score_threshold": [1.0, 2.0, 3.0], "n_components": [2, 4],
     "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1,
    checkpoint_dir="runs/my-explicit-grid",
)
```

This example also uses the absolute-screening `classifier` configured above.
The function evaluates the declared grid and returns a JSON-compatible report,
with provisional CV ranking and grid-refinement diagnostics. It performs no
full-training refit, validation evaluation, majority fallback, or default
selection. All required folds must qualify for a candidate to be ranked.

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

Report schema 3 adds `knn_reuse_setup_seconds` and `knn_shared_query_seconds`
to `compute_seconds`. Shared queries are charged once per fold/backend group;
per-head `prediction_seconds` records voting, guards and any public fallback.
A completed resume retains historical compute totals and performs no new
representation fits or neighbor queries. Those totals are not resume runtime.
Source and schema identities prevent reuse of old checkpoints under this change.

Checksummed fold records can be reused only with matching data, folds, grid,
estimator, source and runtime identity. A report describes the declared grid and
does not certify completed tuning. Record subsequent grid decisions and numerical
qualification before freezing the final study.

Recognized NMFClassifier and screening/PCA/KNN layouts also share training-fold
screening statistics across cutoff and rank settings. The report counts
`screening_statistics_seconds` and `screening_selection_seconds` separately.
Empty screens and impossible ranks remain explicit failed candidates with a
`failure_stage`; no threshold or rank is silently substituted. Generic pipeline
layouts retain their ordinary fit path.

## Reconstruction diagnostics

For large, low-rank float64 fits, report-only reconstruction norms can use
128-row blocks. The input shape must be 8,192–32,768 rows by 1,024–4,096
features, with rank 1–32 and matching coordinate/component shapes. The runtime
and all three arrays must satisfy one of these configurations:

- All detected BLAS pools are one-thread MKL, and the arrays are C- or
  F-contiguous.
- On Windows, exactly one BLAS pool is detected as AOCL/BLIS with version
  `AOCL-BLAS 5.3.0 Build 20260511`, using one thread, and all arrays are
  C-contiguous.

Other configurations, including unrecognized AOCL builds, mixed BLAS backends,
multithreaded settings and unsupported shapes or layouts, retain the
full-product expression. No array is copied merely to enter the blocked path.
Block norms are combined to report the Frobenius norm; matrix-product rounding
can differ from the full expression.
This affects reconstruction diagnostics only. It does not change optimizers,
stopping rules, KKT qualification, fitted coordinates or predictions. The
runtime restrictions reflect measured timing results, not a universal speed
advantage.

## Bounded rank exploration

`supervised_nmf.search.calibration_rank_search` accepts the same estimator,
explicit grid, training data, fold metadata, and checkpoint controls as
`calibration_knn_grid`. It adds `rank_parameter=None`, `max_expansions=2`, and
`max_rank=None`. The rank parameter is inferred only when exactly one
`n_components` grid parameter is declared. An upper-boundary winner may be
tested at doubled rank with its other parameters fixed. New ranks are bounded
by known fold-local dimensions and an optional user cap. Original grid entries
are preserved, including infeasible ones.

The returned report contains each stage, its candidates, the selected settings,
`stop_reason`, and `unresolved_upper_rank_boundary`. Exhausted budgets, user
caps, and unresolved failed upper candidates do not certify completed search.
Even when `search_complete` is true for this upper-rank policy,
`tuning_complete` remains false: screening and neighbor choices have not been
jointly reoptimized and no global optimum is claimed. This helper supports the
shared evaluator's KNN layouts and performs no refit or outer evaluation.

An accepted pipeline containing `FeatureSPA` keeps its anchor-derived training
coordinates and fixed-basis query projection. Rank search evaluates that declared
procedure; it does not convert the pipeline to the SPA classifier's common map.
Use `NMFClassifier(solver="spa")` with `training_coordinates="auto"` or
`"transform"` when that map is intended. See
[standalone SPA coordinates](#standalone-spa-coordinates).

Run `python examples/grouped_tuning.py --output runs/grouped-example` for a
small synthetic subject-separated example.

## Local CelebA data

```powershell
python examples/celeba_classification.py --dir PATH/img_align_celeba `
  --attributes PATH/list_attr_celeba.txt `
  --partitions PATH/list_eval_partition.txt `
  --attribute Smiling --quick
```

Replace these example paths with the supplied data location. Attributes may also
be a CSV with an image ID column. The loader checks IDs, labels, required images,
and official partition codes. Sampling is uniform within a partition; it does
not artificially balance classes. This development example loads training and
validation partitions only. Official test evaluation and identity-overlap audit
remain separate work after the protocol and defaults are frozen.
