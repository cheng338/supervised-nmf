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
    n_neighbors=5,
    weights="uniform",           # sklearn KNN weighting choice
    max_iter=20000,
    tol=1e-4,
    random_state=42,
    training_coordinates="fit_transform",
)
classifier.fit(X_train, y_train)
W_new = classifier.transform(X_new)
y_pred = classifier.predict(X_new)
probabilities = classifier.predict_proba(X_new)
```

The default is SPA with absolute-score screening. Available solvers are
`"spa"`, `"mu"`, and `"cd"`; CD reproduces the original solver choice. The
[four-method development comparison](nmf-solver-comparison.md) supplies empirical
context, not a universal solver ranking. Rank, cutoff and neighbor count remain
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
original scales. The fitted screening mask and basis are reused for prediction.

Inputs are dense finite nonnegative matrices. Binary labels can be strings or
numbers; `classes_[1]` is the positive class in evaluation reports. Each training
class needs at least two observations for screening, and each class needs at
least five observations for five-fold calibration. Sparse inputs and multiclass
classification are not supported.

`selected_features_` maps back to the original columns. `selected_components_`
is the learned basis in retained-feature coordinates; `components_` pads that
basis with zeros in discarded columns. `training_coordinates_` contains the
coordinates used to train KNN. `get_support()` returns the original feature mask.
`fit_times_`, `n_iter_`, `converged_`, and `convergence_warnings_` expose fit
diagnostics. `last_transform_converged_` and `last_transform_warnings_` concern
the most recent transform. Convergence here means the numerical routine did not
report nonconvergence; it is not a certificate of global optimality or consistency.

`training_coordinates="transform"` explicitly re-encodes training observations
against the fitted basis. The default `"fit_transform"` preserves the scripts'
coefficient convention. Report this choice when comparing CD, MU and SPA fits.
`reconstruction_err_` refers to the classifier's actual training coordinates;
`factorization_reconstruction_err_` preserves the initial factorization diagnostic.

## Screening and SPA separately

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

Checkpoints can resume only when data, parameters, versions, folds and package
source hashes match. Changing code or settings requires a new run. The source
examples create unique run directories and expose `--resume` for compatible runs.

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

Set rank, the applicable screening parameter, and neighbor count in each method's
explicit grid. Tune unscreened NMF independently, and use the same saved folds
for paired method comparisons. `comparison_models()` includes independently
tuned `screened_mu_absolute` and `screened_mu_percentile` alongside the existing
CD, SPA, PCA and screening-only comparisons. The raw KNN, majority-class and
historical unscreened-CD baselines remain available. CD, SPA and MU, and
percentile and absolute screening, remain separate alternatives.
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

Checksummed fold records can be reused only with matching data, folds, grid,
estimator, source and runtime identity. A report describes the declared grid and
does not certify completed tuning. Record subsequent grid decisions and numerical
qualification before freezing the final study.

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
