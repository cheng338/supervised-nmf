# API reference

## Classifier

```python
from supervised_nmf import NMFClassifier

classifier = NMFClassifier(
    n_components=2,
    solver="spa",                # default; also "mu" or "cd"
    screening="absolute",        # default; also "percentile" or "none"
    score_threshold=2.0,         # absolute pooled-t units, inclusive comparison
    percentile=10.0,             # used only by percentile screening
    classifier="knn",            # also "logistic"
    n_neighbors=5,
    weights="uniform",           # sklearn KNN weighting choice
    logistic_C=1.0,              # inverse L2 regularization strength
    logistic_max_iter=1000,
    logistic_tol=1e-4,
    logistic_class_weight=None,  # also "balanced" or a class-weight dictionary
    max_iter=20000,
    tol=1e-4,
    random_state=42,
    training_coordinates="fit_transform",
    projection_solver="auto",   # SPA only: "auto", "scipy", or "batched"
    projection_workers=1,        # maximum process count for large queries
    projection_batch_size=256,
    projection_native_threads=1,
    decision_threshold=None,    # optional fixed, training-selected cutoff
    positive_label=None,        # defaults to classes_[1]
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
tunable. `max_iter=20000` preserves that study's numerical allowance: it caps
CD/MU iterations or each SPA NNLS solve, depending on
the solver. Converged routines stop earlier. The tolerance remains `1e-4` for
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

`classifier="knn"` preserves Euclidean KNN on unscaled NMF coordinates.
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

Inputs are dense finite nonnegative matrices. Binary labels can be strings or
numbers; `classes_[1]` is the positive class in evaluation reports. Each training
class needs at least two observations for screening, and each class needs at
least five observations for five-fold calibration. Sparse inputs and multiclass
classification are not supported.

`selected_features_` maps back to the original columns. `selected_components_`
is the learned basis in retained-feature coordinates; `components_` pads that
basis with zeros in discarded columns. `training_coordinates_` contains the
unscaled coordinates supplied to the final classifier. `get_support()` returns
the original feature mask. `classifier_` is the fitted KNN estimator or the
pipeline with steps named `scale` and `logistic`.
`fit_times_`, `n_iter_`, `converged_`, and `convergence_warnings_` expose fit
diagnostics. `n_iter_` remains the factorization iteration count.
`classifier_n_iter_`, `classifier_converged_`, and
`classifier_convergence_warnings_` report the final classifier separately;
`classifier_n_iter_` is `None` for KNN. The overall convergence flag and warning
list include both stages. Classifier
fit timing includes scaler fitting when logistic regression is selected.
`last_transform_converged_` and `last_transform_warnings_` concern
the most recent transform. Convergence here means the numerical routine did not
report nonconvergence; it is not a certificate of global optimality or consistency.

`training_coordinates="transform"` explicitly re-encodes training observations
against the fitted basis. The default `"fit_transform"` preserves the scripts'
coefficient convention. Report this choice when comparing CD, MU and SPA fits.
`reconstruction_err_` refers to the classifier's actual training coordinates;
`factorization_reconstruction_err_` preserves the initial factorization diagnostic.

## Screening and SPA separately

### SPA projection execution

These controls apply to both `FeatureSPA` and `NMFClassifier(solver="spa")`.
They affect fixed-basis projection, including training reprojection when
`training_coordinates="transform"`; SPA anchor selection and basis fitting
still use the original algorithms. CD/MU ignore valid SPA projection controls.

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
the historical row-wise solver for every query.

Blocks contain at most `projection_batch_size=256` rows. With
`projection_workers>1`, queries of at least 512 rows and two full blocks use
joblib's reusable loky process pool. Smaller queries execute in the caller.
Compatible calls reuse workers automatically, including across models; no
pool or cached fitted basis is serialized with an estimator. First use pays
startup cost, and joblib retires idle workers. `projection_native_threads=1`
limits numerical threads per projection process and restores previous limits
afterward. The controller discovers NumPy/SciPy libraries at first projection.
Thread limits are process-wide; use process workers for concurrent work.

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
0.5. An explicit threshold uses the strict rule `P(positive_label) > threshold`.
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

These execution and decision-policy additions are included in version 0.2.0.
They do not revise saved study results or establish an accuracy improvement.

### Screening and anchor details

`BinaryScreening` is a sklearn transformer with `fit`, `transform`, and
`get_support`. It exposes signed `t_scores_`, squared `scores_`, selected indices,
class counts and constant/infinite-score counts. Statistical rescaling is recorded
by `statistic_scales_`; scaled variance/standard-error diagnostics are provided.
The returned data are never multiplied by the screening scores.

Absolute thresholds must be finite and nonnegative. Globally constant columns
are excluded even at threshold zero. Different classwise constants have infinite
absolute scores. Percentile ranking uses the original feature count as its
denominator and follows deterministic feature-index tie breaking. Degenerate
features and empty selection are explicit; near-zero variance is not silently
regularized.

`FeatureSPA` is the separately defined feature-anchor transformer, with
`anchor_indices_`, conditioning diagnostics, and a fixed `components_` basis.
Its `normalize=True` option normalizes training feature columns only during
anchor selection. See [the mathematical specification](methodology.md).

## Calibration and reporting

```python
from supervised_nmf.evaluation import calibration_search, evaluate_classifier

calibrated = calibration_search(
    classifier,
    {"score_threshold": [1.0, 2.0, 3.0], "n_components": [2, 4],
     "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1,
    checkpoint_path="runs/my-development-search.json",
)
validation = evaluate_classifier(calibrated.estimator, X_validation, y_validation)
```

The grid above is an example; suitable ranges depend on the dataset. Any
failed or nonconverged required fold disqualifies its candidate. The report
retains each attempt, parameters, warnings, folds, selected dimensions, and
timings. If every candidate fails, a clearly labeled training-majority fallback
is returned. It must not be reported as the requested method's successful fit.

For logistic regression, pass `NMFClassifier(classifier="logistic")` and replace
`n_neighbors` with `logistic_C`, for example `[0.01, 0.1, 1.0, 10.0, 100.0]`.
Each fold fits its own screening, NMF, scaler and classifier. A logistic
convergence warning disqualifies a candidate just as a factorization warning
does; raising the logistic cap requires a recorded numerical configuration.

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
altering predictions or excluding rows. The summary includes both class recalls,
class counts, available subject counts, timing, and numerical-stage diagnostics.
Numerical-phase diagnostics on `NMFClassifier` now include
`factorization_converged_` and `training_projection_converged_`, with their
corresponding `*_convergence_warnings_` attributes.

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

This function evaluates the declared grid and returns a JSON-compatible report,
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
An `NMFClassifier` using logistic regression is not supported by this helper;
use `calibration_search` instead.

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

Run `python examples/grouped_tuning.py --output runs/grouped-example` for a
small synthetic subject-separated example.

## Local CelebA data

```powershell
python examples/celeba_classification.py --dir D:/datasets/img_align_celeba `
  --attributes D:/datasets/list_attr_celeba.txt `
  --partitions D:/datasets/list_eval_partition.txt `
  --attribute Smiling --quick
```

Replace these example paths with the supplied data location. Attributes may also
be a CSV with an image ID column. The loader checks IDs, labels, required images,
and official partition codes. Sampling is uniform within a partition; it does
not artificially balance classes. This development example loads training and
validation partitions only. Official test evaluation and identity-overlap audit
remain separate work after the protocol and defaults are frozen.
