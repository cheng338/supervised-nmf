# Overview

`supervised-nmf` implements supervised nonnegative matrix factorization for
binary classification with a scikit-learn-style interface. It screens features
using the training labels, fits a nonnegative representation, and classifies
new observations with k-nearest neighbors (KNN), logistic regression or a
linear support vector machine (SVM).

The package provides:

- **Supervised NMF classification** (`NMFClassifier`), with feature-anchor SPA,
  multiplicative updates (MU), and coordinate descent (CD).
- **Feature screening** (`BinaryScreening`), using an absolute pooled-t cutoff
  or a retained percentile.
- **Feature-anchor factorization** (`FeatureSPA`), available as a separate
  transformer. Its `fit_transform` and `transform` use the same fixed-basis
  NNLS map, including in generic classification pipelines. Selected anchor
  columns remain available in `fit_coordinates_`; see
  [standalone SPA coordinates](docs/api.md#standalone-spa-coordinates).
- **Classification comparisons** with screened PCA, screening without
  factorization, raw features, and unscreened CD-NMF, using the selected classifier.

Chang Cheng developed the method and original research scripts. This package
implements Chapter 7 of:

> Cheng, Chang (2026). *Ensemble Kernel Distance-Weighted Discrimination and
> Supervised Nonnegative Matrix Factorization for Classification*. Doctoral dissertation
> manuscript, Purdue University, Department of Statistics.

See [CITATION.cff](CITATION.cff) for citation details and
[source provenance](provenance/README.md) for the original scripts.
Version **0.3.0** adds a common SPA training/prediction coordinate map,
optional SD-scaled KNN and linear SVM, and numerical and execution safeguards.
See the [0.3.0 release notes](docs/release-0.3.0.md) for migration and validation limits.
Version **0.3.3** adds numerical recovery for MU projection, guarded SPA basis
fitting, optional QR projection and clearer evaluation reports. Classifier
choices and defaults are unchanged. See the
[0.3.3 release notes](docs/release-0.3.3.md) for details and verification scope.
Version **0.4.0** fixes mixed-dtype prediction, improves qualified float64 MU
recovery, adds optional rowwise MU projection, and moves prediction diagnostics
into evaluation results. See the [0.4.0 release notes](docs/release-0.4.0.md)
for migration details and the comparison with 0.3.3.

Version **0.4.2** fixes evaluation and reporting edge cases, reduces repeated
numerical work, and adds exact-repeat stopping to experimental ANLS-BPP.
Public solver defaults are unchanged. See the [0.4.2 release notes](docs/release-0.4.2.md)
for changes, compatibility details and validation limits.

Version **0.4.3** corrects sample/group identifier handling, preserves NumPy
extended-precision scalars in reports and checkpoints, and stabilizes
calibration checkpoint identities when native libraries load lazily. See
the [0.4.3 release notes](docs/release-0.4.3.md) for compatibility and
validation scope.

Version **0.5.0** aligns standalone SPA training and query coordinates, ranks
calibration candidates by ordinary accuracy by default, and uses per-observation
projection for newly fitted MU classifiers. Standalone screening now defaults
to the absolute pooled-t cutoff. See the [upgrade guide](docs/release-0.5.0.md)
for explicit alternatives and the [changelog](CHANGELOG.md) for the release history.

Version **0.5.1** strengthens checkpoint recovery tests and documents the Linux
startup settings for reusing completed study results in the same process.
See the [0.5.1 release notes](docs/release-0.5.1.md).

**Copyright (c) 2026 Chang Cheng. All Rights Reserved.** A distribution license
has not yet been selected. Public availability of this repository does not
grant an open-source license. See [LICENSE](LICENSE) and the separately
applicable [third-party notices](THIRD_PARTY_NOTICES).

# Installation

Python 3.11 or later is required for version 0.5.1. Python 3.10 support
ended on October 20, 2026. Dependencies are NumPy, SciPy, scikit-learn
1.6 or later (below 2.0), threadpoolctl, and joblib 1.3 or later.
Download the wheel from the [GitHub releases](https://github.com/cheng338/supervised-nmf/releases)
and install it:

```shell
python -m pip install --upgrade supervised_nmf-0.5.1-py3-none-any.whl
```

To install the release from source:

```shell
git clone --branch v0.5.1 https://github.com/cheng338/supervised-nmf.git
cd supervised-nmf
python -m pip install .
```

Image-loading examples require Pillow. To install the optional example or
study dependencies from the source directory:

```shell
python -m pip install ".[examples]"
python -m pip install ".[study]"
```

For a downloaded wheel, append the extra to the local filename, for example
`python -m pip install "supervised_nmf-0.5.1-py3-none-any.whl[examples]"`.
The study extra also includes Matplotlib. Use `python -m pip install -e .`
for an editable development installation. Tested environments and remaining
compatibility checks are listed in [validation](docs/validation.md).
Python 3.12 with AOCL remains the primary development environment.
See [runtime compatibility](docs/compatibility.md) for the current setup policy
and historical numerical profiles. Earlier Python 3.10 test results describe
their recorded revisions, not support for current development.

# Basic use

**For your own data, start with [training-only tuning](#training-only-tuning).**
The constructor's rank (`2`), absolute screening cutoff (`2.0`) and neighbor
count (`5`) are starting values, not calibrated settings. Calling `fit` fits
the specified configuration; it does not search for better parameters.

This example generates a small dataset from the original simulation model and
fits both SPA and MU. It requires no download and demonstrates the API with
illustrative parameters; its printed scores are not solver-selection evidence.

```python
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import make_original_simulation

data = make_original_simulation(n_samples=200, n_features=100, signal_ratio=0.1)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25,
    stratify=data.target, random_state=42,
)

models = {}
for solver in ("spa", "mu"):
    model = NMFClassifier(
        solver=solver, screening="absolute", score_threshold=2.0,
        n_components=2, n_neighbors=5, random_state=42,
    ).fit(X_train, y_train)
    models[solver] = model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    print(solver, balanced_accuracy_score(y_test, predictions))
```

Rows are observations and columns are features. Inputs must be dense, finite
and nonnegative. The classifier preserves the two supported input class labels;
string labels must not end in a NUL codepoint (`\x00`). See the
[class-label domain](docs/api.md#class-label-domain) for safe encoding and input
conversion limits. Probability columns follow `model.classes_`. Screening and factorization are
fitted on training data. Prediction reuses the fitted feature mask and basis.
Version 0.4.0 records prediction diagnostics in evaluation results instead of
updating the fitted model's `last_transform_*` attributes. See
[prediction state and diagnostics](docs/api.md#prediction-state-and-diagnostics)
for migration from version 0.3.3 and concurrency limits.

This example demonstrates the API with illustrative parameters. SPA selects
anchors geometrically after screening. Surviving background features with high
variability relative to their means can occupy a small representation even when
signal features are retained. Training-only tuning and comparisons with screening-only,
PCA and other solvers remain necessary; a common training/prediction map does
not by itself resolve this limitation. See the
[anchor-selection explanation](docs/theory-implementation.md#spa-anchor-selection-and-screening-false-positives)
and [completed simulation diagnostic](docs/wang-simulation-validation.md#october-20-anchor-and-information-diagnostic).

## Training-only tuning

Reserve the final evaluation data before exploring settings. Fit screening,
the representation and the classifier inside each training CV fold; do not
screen the full dataset before CV. For repeated subjects or duplicate groups,
keep groups separate in the outer split and supply training `groups` to the
calibration helper. The synthetic rows above illustrate the independent-row case.
For an application, begin this workflow before viewing any final-cohort scores;
the synthetic scores printed above are demonstration output.

For 8-bit face images, pass floating-point pixels divided by 255; this fixed
conversion is safe before CV. Keep learned preprocessing inside the pipeline.
See [input preparation and CV usage conditions](docs/api.md#input-preparation-and-cv-usage-conditions)
for subject groups, independent evaluation and custom-component requirements.

The helper can evaluate an explicit grid without a full-training refit or
final-test prediction. In 0.5.0, it ranks candidates by
mean ordinary accuracy by default. Every required fold must qualify numerically:

```python
from supervised_nmf.evaluation import calibration_search

calibrated = calibration_search(
    NMFClassifier(solver="spa", screening="absolute", random_state=42),
    {"n_components": [2, 4], "score_threshold": [0.0, 2.0],
     "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1, refit=False, scoring="accuracy",
    checkpoint_path="runs/quickstart-cv.json",
)
if calibrated.report["selected_parameters"] is None:
    raise RuntimeError("No candidate qualified in every training CV fold.")
print(calibrated.report["selected_parameters"])
print(calibrated.report["scoring"], calibrated.report["selected_mean_score"])
print(calibrated.report["refinement"])
```

This small grid illustrates the workflow, not recommended ranges for other
datasets. With `refit=False`, `calibrated.estimator` is `None`. A completed grid
and a selected candidate do not establish completed tuning. Review candidate
failures, warnings and refinement diagnostics; investigate feasible boundary
winners and nearby settings using training data. Preserve the same folds and
use a new checkpoint path when changing the grid. Compare with screening-only,
PCA and raw-feature baselines using the same training folds.

Choose `scoring="balanced_accuracy"` explicitly when mean class recall is the
selection objective. These are the two supported scorers; the helper does not
switch objectives automatically when it detects imbalance. CV scores are the
equal-weight mean of the fold scores, rather than a pooled out-of-fold score.
The selected CV score is separate from accuracy measured on reserved data.

**November 3 update:** calibration now defaults to ordinary accuracy; choose
`scoring="balanced_accuracy"` explicitly when intended. Use a new checkpoint
after changing the objective or source. Historical study results remain unchanged.

Once that training-only review is complete, freeze the chosen configuration,
fit it on all training data, check numerical qualification, and evaluate it on
the reserved data. Do not change settings in response to the final scores.
Report balanced accuracy and class-specific recall alongside ordinary accuracy,
especially for imbalanced classes.

See [calibration and reporting](docs/api.md#calibration-and-reporting) for
metadata, failures and final evaluation, and
[shared KNN grids](docs/api.md#explicit-grids-with-shared-representations) to
reuse fold-local representations across neighbor counts. The
[grouped example](examples/grouped_tuning.py) demonstrates subject metadata and
bounded rank exploration; its caps and subsequent demonstration fit do not
certify completed tuning. The [October 29 CelebA comparison](docs/current-package-evidence-20261029.md)
records how strongly untuned and tuned results can differ on the evaluated tasks.

## Other classifier and scaling options

KNN remains the default. To use L2-regularized logistic regression on the NMF
coordinates:

```python
logistic_model = NMFClassifier(
    classifier="logistic", logistic_C=1.0, random_state=42,
).fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
```

This optional extension standardizes the fitted coordinates inside the
classifier, using training means and scales. It does not change screening,
factorization, or the coordinates returned by `transform`.

Optional KNN scaling divides each coordinate by its
training-fitted standard deviation without centering:

```python
scaled_knn = NMFClassifier(
    knn_normalization="sd", n_neighbors=5, random_state=42,
).fit(X_train, y_train)
```

The default `knn_normalization="none"` preserves unscaled KNN. SD changes the
distance metric used by the head; it does not alter screening, factorization,
`training_coordinates_` or `transform`. The fitted scaler is reused at
prediction time and follows scikit-learn's constant/nearconstant scale handling.
SD can amplify weak or nuisance coordinates, so it is an explicit choice rather
than an automatic accuracy improvement. Logistic regression retains its existing
scaler; the KNN option does not add another one.

The linear-SVM head uses unscaled NMF coordinates and a primal
L2-regularized squared-hinge objective:

```python
svm_model = NMFClassifier(
    classifier="linear_svm", svm_C=1.0,
    svm_max_iter=20000, svm_tol=1e-5, random_state=42,
).fit(X_train, y_train)
svm_predictions, margins = svm_model.predict_with_decision_function(X_test)
```

Margins are real decision scores toward `classes_[1]`, not probabilities.
Positive margins select that class; exact zero selects `classes_[0]`.
SVM has no `predict_proba` or `predict_with_probabilities` method, and fitting
with a non-`None` `decision_threshold` raises an error. There is no implicit
standardization or probability calibration. Tune `svm_C` with the coordinate
units in mind; its numerical cap and tolerance are separate from NMF's.

# Solvers and defaults

The default is **SPA with absolute screening**. Use `solver="mu"` for
multiplicative updates or `solver="cd"` for the dissertation's original
coordinate-descent implementation.

The default `mu_projection="auto"` resolves to `"rowwise"`
for MU and `None` for SPA/CD at fit time. Rowwise MU gives each observation its
own initialization and stopping decision, supporting predictions made singly or
in changing batches. It uses guarded float64 working arithmetic while preserving
the projection output dtype. The fitted choice is recorded in `mu_projection_`;
changing constructor settings or loading a current-format model does not replace
that choice before refitting. The default factorization solver remains SPA.

Select `mu_projection="batch"` explicitly when its measured throughput benefit
is useful and request-composition sensitivity is acceptable: shared stopping
can change coordinates or labels when unrelated query rows change. Rowwise
projection does not guarantee bitwise invariance, convergence or better accuracy,
and can cost more for small requests or failed recovery attempts. MU/CD still use
joint-fit training coordinates by default; `training_coordinates="transform"`
separately requests a common training/query projection policy. See
[MU prediction policies and measured costs](docs/api.md#prediction-batches).

Since version 0.3.0, the default `training_coordinates="auto"` trains the
SPA classifier on fixed-basis NNLS coordinates, using the same fitted basis
and projection map for training and prediction. Version 0.5.0 also makes standalone `FeatureSPA.fit_transform` use that map. For newly
fitted SPA classifiers, all accepted `training_coordinates` settings resolve
to `"transform"`, including explicit `"fit_transform"`. The classifier projects
training observations once during `fit`, without duplicating the standalone
transformer's projection. CD and MU retain their existing coordinate policies.
Standard SPA `NMFClassifier.fit_transform` reuses a copy of those saved NNLS
coordinates, avoiding another projection after `fit`. Custom transform
implementations retain their own fit-then-transform behavior.
See [standalone SPA coordinates](docs/api.md#standalone-spa-coordinates).

The standalone correction adds training projection work and may change a
retrained pipeline's predictions; improved accuracy is not guaranteed. The
default SPA classifier already performed this projection. Version 0.2.0
defaulted to anchor-derived SPA training coordinates. Archived artifacts and
historical results retain their recorded behavior; loading an older fitted
model does not retrain its classifier or rewrite its coordinate metadata.

SPA uses direct BLAS updates and reuses the fixed NNLS matrix layout during
projection. Version 0.3.3 also uses guarded batched NNLS to fit the basis
from the selected anchors, with public SciPy fallback. This preserves the
fixed-anchor objective; see [basis fitting](docs/api.md#spa-basis-fitting)
for conditioning, numerical-range and iteration-limit behavior.
The default `projection_solver="auto"` uses guarded batched NNLS for suitable
query batches. Use `"scipy"` for the original row-wise projection or `"qr"` to
request guarded QR reduction with public SciPy solves. QR shares the reduction
across a query batch while preserving the fixed-basis NNLS objective. Its speed
depends on the workload; it can also add overhead or fall back to the current
solver. Selecting it does not change anchor selection or SPA basis fitting.
Optional `projection_workers=8` enables reusable process workers for large
batches; the default is one worker. Keep that default for small workloads:
process startup and result collection can outweigh parallel execution. Compare
complete fit/prediction workloads before increasing the worker count. See
[projection controls](docs/api.md#spa-projection-execution),
the [0.2.0 implementation record](docs/implementation-improvements.md), and the
earlier [BLAS checks](docs/blas-optimization.md).

| Solver | Factorization |
| --- | --- |
| `"spa"` | Selects feature anchors, then solves nonnegative least-squares problems to fit the basis and encode new observations. |
| `"mu"` | Uses scikit-learn's multiplicative updates with Frobenius loss and NNDSVDa initialization. |
| `"cd"` | Uses scikit-learn's coordinate descent with Frobenius loss and NNDSVDa initialization. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_components` | `2` | Number of NMF components. |
| `solver` | `"spa"` | Factorization method. |
| `screening` | `"absolute"` | `"percentile"` selects by relative rank; `"none"` retains all input columns and ignores screening cutoffs. |
| `score_threshold` | `2.0` | Inclusive cutoff on the absolute pooled-t score. |
| `percentile` | `10.0` | Retained percentage when percentile screening is selected. |
| `classifier` | `"knn"` | Final classifier; also accepts `"logistic"` and `"linear_svm"`. |
| `n_neighbors` | `5` | Number of neighbors; used only by KNN. |
| `weights` | `"uniform"` | Equal KNN votes; `"distance"` uses inverse distances. A callable can supply a custom rule; `None` also means uniform. |
| `knn_normalization` | `"none"` | `"sd"` uses training-fitted SD without centering. |
| `logistic_C` | `1.0` | Inverse L2 regularization strength for logistic regression. |
| `logistic_max_iter` | `1000` | Logistic-regression iteration limit, independent of the NMF cap. |
| `logistic_tol` | `1e-4` | Logistic-regression stopping tolerance. |
| `logistic_class_weight` | `None` | Also accepts `"balanced"` from training labels, or a dictionary of class weights. |
| `svm_C` | `1.0` | Linear-SVM inverse L2 regularization strength; depends on coordinate units. |
| `svm_max_iter` | `20000` | Linear-SVM iteration cap, independent of the NMF cap. |
| `svm_tol` | `1e-5` | Linear-SVM stopping tolerance. |
| `svm_class_weight` | `None` | Linear-SVM class weights: `None`, `"balanced"` or a dictionary. |
| `max_iter` | `20000` | CD/MU iteration limit or the limit for each SPA NNLS solve. |
| `tol` | `1e-4` | CD/MU stopping tolerance; SPA's NNLS routine does not use this parameter. |
| `random_state` | `None` | Set an integer for reproducible CD/MU initialization. |
| `training_coordinates` | `"auto"` | Newly fitted SPA models always resolve to `"transform"`; CD/MU default to `"fit_transform"` and allow explicit `"transform"`. |
| `mu_projection` | `"auto"` | Resolves to `"rowwise"` for MU and `None` for SPA/CD; explicit `"batch"` retains shared stopping. Recorded at fit time. |
| `projection_solver` | `"auto"` | SPA fixed-basis projection: guarded batching; also `"scipy"`, `"batched"` or guarded `"qr"`. |
| `projection_workers` | `1` | Maximum SPA projection processes; independent of outer CV workers. |
| `projection_batch_size` | `256` | Maximum query rows per NNLS block. |
| `projection_native_threads` | `1` | Numerical threads per projection process. |
| `decision_threshold` | `None` | Optional frozen probability cutoff for KNN/logistic; SVM requires `None`. |
| `positive_label` | `None` | Class for probability thresholding; defaults to `classes_[1]`. Does not reverse SVM margins. |

For when to choose these options, their costs and restrictions, see the
[API reference](docs/api.md#classifier). The
[parameter-change guide](docs/api.md#changing-settings-on-a-fitted-model)
distinguishes settings requiring refit from execution controls that affect the
next prediction. Standalone SPA has additional
[normalization, conditioning and iteration controls](docs/api.md#standalone-spa-options).

Both `NMFClassifier()` and standalone `BinaryScreening()` default to absolute
pooled-t screening with `score_threshold=2.0`. Choose
`screening="percentile"` explicitly to use `percentile`; changing a selection
parameter does not switch the mode. The November 3 update changes standalone
`BinaryScreening` from its previous percentile default.
Calibration reports identify the active screening mode and parameter, and warn
once per calibration grid call when an explicitly gridded, known screening
parameter is inactive for a candidate. Rank-search stages are separate grid
calls. These diagnostics do not change modes, remove grid entries, or add warnings
to ordinary estimator construction or fitting.
See [screening details](docs/api.md#screening-and-anchor-details).

Rank, screening cutoff and the selected classifier's predictive parameters
require tuning for the data. CD and MU use different stopping criteria, so
equal tolerances do not imply equal
numerical accuracy. Features and factors are not automatically rescaled for
NMF. KNN is unscaled by default and offers explicit SD scaling.
Logistic regression uses a separate fitted coordinate scaler; linear
SVM uses the unscaled coordinates.

MU projection returns exact zero coordinates for an entirely zero selected
query batch. For ordinary float32 queries whose fitted basis has tiny tails,
or whose arithmetic raises a floating-point error, batch mode first retries in float64.
Tiny positive queries and failed retries use logarithmic arithmetic. These
corrections preserve the fitted basis and returned dtype; they do not change
joint factorization. See [MU projection](docs/api.md#fit-diagnostics) for
numerical limits, warnings and possible additional cost.

Version 0.3.3 also supports a bounded, one-thread AOCL path for reporting
reconstruction norms, alongside the existing MKL path. It affects diagnostics
only, not optimization, stopping, or predictions. See the
[runtime and array requirements](docs/api.md#reconstruction-diagnostics).

The SPA/absolute-screening choice follows the recorded accuracy-time comparison
on CelebA and Fashion-MNIST. The disclosed CelebA screening-preference amendment
depends on its accuracy allowances, assumed workload and development
configurations; see the [amended sensitivity results](docs/celeba-default-results.md#amended-screening-preference-rule).
Those historical comparisons used anchor-derived
SPA training coordinates (`training_coordinates="fit_transform"`), before the
v0.3.0 switch to fixed-basis training projection. Their accuracy and timing
results describe the procedures evaluated then. The solver-selection evidence
remains under final audit and does not establish a universal best solver.
SPA's factor-recovery results require additional separability and noise
assumptions; classification accuracy does not establish
those assumptions. See [solver comparisons](docs/nmf-solver-comparison.md),
[methodology](docs/methodology.md), and [theoretical scope](docs/theory-implementation.md).

# Release changes

Version 0.4.0 supports mixed float32/float64 queries for CD and MU, adds guarded
float64 MU scaling recovery and the explicit `mu_projection="rowwise"` option,
and keeps prediction diagnostics separate from fitted estimator state.
It also adds the standard transformer interface and corrects estimator tags,
validation messages and historical-study reporting. Existing defaults remain
unchanged. See the [0.4.0 release notes](docs/release-0.4.0.md).

Version 0.3.3 adds MU projection recovery, guarded batched SPA basis fitting and
the explicit `projection_solver="qr"` option. Evaluation summaries show class
balance and distinguish overlapping timing totals. Source tests run under
pytest. The default remains SPA with absolute screening and unscaled KNN;
the new projection option does not change factorization or classifier defaults.
See the [0.3.3 release notes](docs/release-0.3.3.md).

Version 0.3.0 changes SPA classifier training to the common fixed-basis map
described above, adds optional SD KNN and linear SVM, stabilizes screening for
large offsets, and guards extreme-range NNLS and factor-recovery calculations.
Compatible KNN grids reuse neighbor queries; a narrowly guarded one-thread MKL
path accelerates report-only reconstruction norms. These changes do not imply
a uniform speed or memory improvement. Controlled memory probes found peaks
about 27–72% higher in particular screening and projection cases, relative to
the earliest recoverable source. See the [release notes](docs/release-0.3.0.md)
for the measured scope, mixed timing results and unresolved tuning boundaries.

Version 0.1.0a2 added subject-aware training folds, bounded rank exploration,
reused fold-local screening statistics, and evaluation summaries with class
support and numerical-stage diagnostics. Solver, screening, and rank defaults
were unchanged in that release. Version 0.2.0 additionally provided guarded
batched projection and saved decision thresholds. In that release, older
serialized models retained their SciPy projection path. Current loading no longer
fills missing legacy fields; see the [model-persistence contract](docs/api.md#model-persistence).
See the [0.2.0 release record](docs/release-0.2.0.md),
[tuning improvements](docs/tuning-improvements.md), and the [API guide](docs/api.md).

From the source checkout or unpacked source archive, run
`python examples/grouped_tuning.py --output runs/grouped-example` for a
small subject-separated example. Search budgets and failed upper-rank
candidates remain explicit; a completed grid does not establish a global
optimum or complete statistical validation.
Calibration, final fitting and evaluation each use one native numerical thread.
The final-fit context restores the caller's previous thread limits on return or
error.

# Parameter tuning

Tune the complete classifier inside training folds so that each fold fits its
own feature screening and factorization. The package's calibration helper
records failed and unconverged fits, parameters, selected features and timings.

```python
from supervised_nmf.evaluation import calibration_search

search = calibration_search(
    NMFClassifier(solver="mu", screening="absolute", random_state=42),
    {"score_threshold": [1.0, 2.0, 3.0],
     "n_components": [2, 4], "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1, scoring="accuracy",
)
selected_model = search.estimator
print(search.report["selected_parameters"])
print(search.report["is_fallback"])
```

These are starting ranges. Check boundary winners and nearby parameter values
using training data, while keeping the previously reserved evaluation
observations untouched.
A candidate with a failed or unconverged required fold cannot win. If all
candidates fail, the helper returns a flagged majority-class fallback; inspect
`is_fallback` before treating the result as a fitted NMF classifier.

For larger grids, `calibration_knn_grid` reuses each training-fold representation
across neighbor counts. Both helpers support resumable records with matching
data, scoring objective, settings, source and runtime identities. The default
selection objective is ordinary accuracy; use `scoring="balanced_accuracy"`
explicitly to select by mean class recall. See the [API guide](docs/api.md).

`calibration_knn_grid` keeps `execution_order="representation"` by default.
For long, restartable grids with multiple representation settings, explicitly
choosing `execution_order="fold"` can reduce screening repeated after partial
resumes. It changes which candidates complete early and can be slower on short
workflows. Reports record the requested and resolved order, including any
eligibility fallback; changing the request requires a new checkpoint directory.
See [execution-order options and an example](docs/api.md#execution-order-for-restartable-grids).

For logistic regression, use `calibration_search` with
`NMFClassifier(classifier="logistic", screening="absolute")` and tune `logistic_C`, for example over
`[0.01, 0.1, 1.0, 10.0, 100.0]`, alongside rank and screening. Do not include
KNN parameters in that grid. The scaler is fitted within each training fold.
`calibration_knn_grid` is restricted to KNN.

For linear SVM, use `calibration_search` with
`NMFClassifier(classifier="linear_svm", screening="absolute")` and tune `svm_C` alongside rank and
screening. `comparison_models(classifier="linear_svm")` supplies independent
SVM grids for the NMF, PCA, screening-only and raw-feature comparisons, with
no added scaler. Evaluation uses genuine margins for AUROC and average
precision and records them separately from probabilities. The completed
integration checks do not establish a universal accuracy advantage.

# Examples and diagnostics

Run the example scripts from the source checkout or source archive:

```shell
python examples/quickstart.py
python examples/compare_simulation.py --quick --output runs
python examples/compare_simulation.py --quick --classifier linear_svm --output runs
python examples/compare_classifiers.py --quick --output runs
python examples/celeba_classification.py --help
python examples/celeba_parameter_study.py --help
```

`compare_classifiers.py --threads` accepts positive integers within the native
C-int range, including counts above eight; its default remains one. Choose a
count within the machine's numerical-thread budget. See the
[thread-count contract](docs/api.md#scalar-controls-for-calibration-and-evaluation).

Both CelebA study CV launchers use explicit process contexts, with `spawn` for
new output directories. The default-study CLI can automatically use an
empty-preload `forkserver` for eligible Linux multiworker checkpoint runs. See
the [startup policy](docs/celeba-study.md#process-startup-for-current-runs) for
eligibility, worker budgets, measured costs and source-bound resume compatibility.

The simulation and CelebA examples compare independently tuned SPA, MU, CD,
screened PCA and screening-only KNN. CelebA requires local images, attributes
and official partition files. Images use the original 64 by 64 grayscale
preprocessing; the examples use the training and validation partitions.
Datasets are not included in the package.

The parameter-study example reuses compatible completed results by default.
Use `--final-fit-recovery all` when repeating a completed final fit would be
costly and you want to preserve it before evaluation. The default is `off`:
checkpointing adds work even to successful ordinary runs. The recovery
guide reports that overhead, observed prediction-time increases and uncertainty.
This option loads joblib/pickle state only from your own trusted local study,
and the same choice must be supplied with `--resume`. It does not continue an
interrupted solver. See the [recovery options and restart contract](docs/celeba-parameter-study-recovery.md)
for costs, compatibility, strict chunk checks and finalization ownership.
Eligible raw KNN preserves its complete native state through public serialization
buffers, without fitting again. Recovery copies the verified training matrix
into an owned read-only snapshot inside model loading; the restored model
does not depend on that cache afterward. Other layouts and declared models
retain complete joblib state.

Both `compare_simulation.py` and `celeba_classification.py` accept
`--classifier linear_svm`, alongside `knn` (the default) and `logistic`.
The SVM options are `--svm-max-iter` (20000), `--svm-tol` (1e-5), and
`--svm-class-weight` (`none` or `balanced`, default `none`). They set the SVM
constructor defaults; `--grid-json` can override them per candidate. SVM model
names end in `_linear_svm`; use `svm_C` for NMF grids and `linear_svm__C` for
baseline pipeline grids. SVM scores are decision margins, not probabilities.
Selected SVM settings and grid overrides are saved for strict resume checks;
KNN/logistic configuration formats are unchanged. A quick run remains a
correctness check, not completed tuning or evidence for new package defaults.

`compare_classifiers.py` compares KNN and logistic regression across matched
NMF, PCA, screening-only and raw-feature paths. The completed
[October 6 Wang validation](docs/wang-validation-report.md) adds 900 independently
retuned simulation draws, ten fully retuned CelebA resamples and twenty fixed-setting
resamples. It finds classifier-dependent benefits and retains unfavorable
comparisons with simpler baselines. Its SPA classifiers used anchor-derived
`fit_transform` training coordinates. No classifier default was changed by
that study.

The October 6, 2026 [CelebA Smiling identity validation](docs/celeba-identity-validation.md)
uses 2,000 training identities and one face from each of 1,000 separate official-test
identities. All four KNN procedures receive 104 initial neighbor choices in
training-only CV. SPA sNMF+KNN with `training_coordinates="fit_transform"`
achieved 81.40% accuracy versus 78.50% for sPCA+KNN; the balanced-accuracy
difference is +2.86 percentage points
(paired 95% interval +0.68 to +5.05). Both independent audits pass. This supports
a dataset-specific KNN comparison; the task was selected from prior development
evidence, and the difference from screening-only KNN remains inconclusive.
These results precede the v0.3.0 training-coordinate change and do not measure
the current default's accuracy.

Later [projected-coordinate studies](docs/validation.md#evidence-for-projected-spa-training-coordinates)
used fixed-basis NNLS for both SPA training and query observations. An October 12
CelebA study completed 300 audited official-partition outcomes; it used a frozen
study adapter, shared screening and tuned coordinate scaling. sPCA KNN had slightly
higher mean balanced accuracy on all three attributes. An October 20
public-estimator comparison with unscaled KNN reported higher mean balanced
accuracy for sNMF than sPCA on two attributes and lower on the third, with
unresolved rank boundaries. These are descriptive results
on previously explored data, not exact current-default accuracy estimates or
independent confirmation. The linked record includes all ten October 12
comparators, imbalance limitations and subsequent implementation checks.

The October 6 [unscreened NMF supplement](docs/celeba-unscreened-supplement.md) restores
the dissertation's standard CD-NMF+KNN baseline and adds unscreened SPA-NMF+KNN,
bringing this face comparison to six KNN procedures. The supplement was designed
after the original test outcomes were known; the original four results remain
unchanged. Standard CD-NMF+KNN achieved 73.00% accuracy and unscreened SPA-NMF+KNN
61.90%, using `fit_transform` training coordinates; both evaluations and
independent audits passed. Both added methods select rank 80 at the explored upper boundary,
so their results do not establish the best attainable unscreened-NMF accuracy.

The October 6 [screened CD supplement](docs/celeba-screened-cd-supplement.md)
completes the seven-method comparison. Screened CD-NMF+KNN obtains 79.00%
accuracy after selecting cutoff 8, rank 10 and K 39; screened SPA remains at
81.40% from the original `fit_transform` study. Both retain the same 124 pixels,
but their ranks and K differ. SPA's balanced-accuracy advantage is 2.40 points
(descriptive, unadjusted 95% interval
0.18 to 4.62). All 2,084 new training candidates and the final CD fit qualify;
both independent audits pass. This additional known-test comparison preserves
all six earlier results and does not establish universal solver superiority.

The October 6 [additional-dataset study](docs/multi-dataset-results.md) adds
FEI, CMU Faces, Sonar, Spambase and WDBC, with all 35 final evaluations qualified
and both independent audits passed. Its SPA results also use anchor-derived
`fit_transform` training coordinates. Face tuning and testing separate subjects.
SPA sNMF has higher accuracy than screened PCA on FEI (94.17% versus 91.67%),
ties on WDBC, and is lower on the other three tasks. The FEI difference remains
inconclusive under its subject-bootstrap interval. Every dataset outcome is
retained; one reserved split per dataset does not constitute repeated nested-CV
replication. CMU has only six test subjects; its
[uncertainty sensitivity analysis](docs/multi-dataset-results.md#cmu-uncertainty-sensitivity)
qualifies the nominal intervals without establishing population superiority.
See the [fixed protocol and sources](docs/multi-dataset-validation.md).

After fitting, inspect `selected_features_`, `selected_components_`,
`training_coordinates_`, `n_iter_`, `converged_`, and `convergence_warnings_`.
`components_` maps the basis back to the original feature columns.

For 8-bit images, explicitly divide pixel intensities by 255 before fitting if
that is the chosen preprocessing; the package does not rescale pixels automatically.
Screening retains supplied values. Saved factors can still be visualized after
classification, and `255 * (W @ selected_components_)` reconstructs retained
features approximately in the original intensity units when inputs used `/255`.
Mark excluded features separately from zero-valued retained features, and do not
interpret individual basis amplitudes as literal pixel intensities. See
[image values and saved-model reconstruction](docs/api.md#image-values-and-saved-model-reconstruction).

`classifier_n_iter_`, `classifier_converged_`, and
`classifier_convergence_warnings_` distinguish classifier diagnostics from
factorization diagnostics. `fit_times_` separates screening, factorization,
training projection and classifier fitting, including fitted head scaling.
Evaluation reports record prediction time; search reports record
tuning and final-fit time separately.

Numerical convergence does not certify a global optimum or classification
consistency. Balanced accuracy, class counts, precision and recall accompany
ordinary accuracy in the comparison reports. See the
[CelebA study](docs/celeba-study.md) and [validation record](docs/validation.md)
for completed work and remaining checks. The current API supports binary
classification with dense inputs; sparse inputs and multiclass classification
are outside this release.

# Tests and local builds

From the source checkout, install the test and study dependencies:

```shell
python -m pip install -e ".[study,test]"
```

Run the pytest suite through the launcher, which also configures native-failure
tracebacks and Windows error handling:

```shell
python scripts/check.py
```

On Linux the launcher sets the OpenMP, MKL and OpenBLAS startup limits to one
thread before loading pytest. Tests of in-process checkpoint reuse need a fresh
worker thread to observe those limits. Use the same settings when running
those tests directly. This command runs both recovery modules.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/test_completed_reuse.py tests/test_completed_reuse_core.py
```

For an IDE test run, set these three environment variables to `1` in the
test-run configuration before starting its interpreter.

The public suite requires pytest 9 or later. It covers current package behavior
and rejection of incompatible study requests. Qualifying an original frozen
study requires its archived sources and runtime. See
[test scope and archived qualification](docs/validation.md#current-regressions-and-archived-qualification).

To check an installed copy on synthetic data, run the following command from
the source checkout or unpacked source archive. Choose a new output filename
in an existing directory. The report records six fit, prediction and model
round-trip checks, plus the installed environment's dependency status.

```shell
python -I scripts/verify_snmf_environment.py --output environment-check.json
```

Build release distributions from a fresh checkout of the intended public
revision. The source archive includes the examples, public study tools,
tests, protocols and small replay fixtures. Local research launchers, raw
datasets and machine-specific run records belong outside the build checkout.
The build includes files by directory and file type.

With the build tools installed, run:

```shell
python scripts/build_release.py
python -I scripts/check_wheel.py
```

The build produces `dist/supervised_nmf-0.5.1-py3-none-any.whl` and a source
archive. The wheel check verifies
imports, authorship metadata, SPA/MU/CD fitting and prediction, and archive
contents. It does not publish a release.
