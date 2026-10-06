# CelebA development study

Author: Chang Cheng. The first explicit starting-grid comparison and a subsequent
four-combination default study are complete. This document preserves the first
study's protocol and records the later scoped decision separately.

The subsequent [four-solver comparison](nmf-solver-comparison.md#development-results)
on Fashion-MNIST and CelebA also selected SPA with absolute screening. That report
remains `selected_for_audit`. The CelebA evidence and both screening decisions are
preserved separately. Official-test evaluation has not been completed.

See [the first development results](celeba-initial-results.md) for observed
metrics, timing examples and limitations.

## Repeated default comparison and preference amendment

After training-only refinement, the four CD/SPA and percentile/absolute
combinations were tested on 2,000, 4,000 and 8,000 training images at seeds 43,
44 and 45, for both attributes. All methods used the same original 4,000-image
official validation sample. All 72 selected evaluations and 216 additional serial
timing repeats qualified, with exact prediction/probability agreement. All 2,050
feasible CV candidates qualified; 110 of 2,160 declared candidates were excluded
by the preserved dimensional-feasibility preflight. CV used 1,025 shared
representation-fold fits, reusing each representation across neighbor counts.

The original frozen rule selected SPA with percentile screening. After the
development results, the screening preference was amended to favor supervised
PCA's absolute-score screening when differences were narrow. The separate amendment
retains the global 1-percentage-point macro balanced-accuracy and 2-point
per-attribute tolerances, prefers qualifying absolute methods, then uses the
unchanged cost/tie rule. That CelebA amendment selects **SPA with absolute screening**,
the installed combination. Its macro
balanced accuracy is 71.7846%, 0.5587 points below the best combination; its
Smiling and Necktie deficits are 0.4479 and 0.9222 points. CD-absolute remains a
more accurate but costlier alternative. Rank, cutoff and neighbors remain tunable.

The original decision is preserved under `runs/default-selection/decision/`;
the amendment is under `runs/default-selection/decision-spca-preference/`.
This preference was not preregistered. Both revised theoretical proof routes
remain conditional and do not compel a solver; absolute screening has a simpler
connection to the threshold theorem, but CV does not verify its assumptions.
See [the default results](celeba-default-results.md),
[the original protocol](default-selection-protocol.md) and
[theory/implementation scope](theory-implementation.md).

The repeated CV command took 319.674 seconds with eight workers and one native
thread each. The subsequent selected evaluations and serial profiling command
took 530.846 seconds. These command wall times include overhead and are distinct
from per-method CV worker-wall sums and per-case median fit/predict costs. They
exclude earlier refinement and data preparation. Whole-process peak memory was
not measured. These overlapping configurations are not independent datasets.

The first final-test attempt was interrupted after development revalidation and
after entering test loading, before any recorded test prediction or evaluation.
No test outcomes entered either decision; partial in-memory decoding cannot be
excluded. The interrupted attempt is preserved. Final evaluation requires an
evidence audit and a new protocol freeze. Identity-disjointness, broad simulation
and generalization claims remain outside the completed default study.

## Questions and comparisons

The first priority is whether factorization improves on screening alone and
whether screened NMF improves on screened PCA (Wang's R7). Compare the two NMF
estimators, CD and feature-anchor SPA, and the two screening rules, percentile
and absolute pooled-t cutoff. Include screening-only KNN, raw-feature KNN,
independently tuned unscreened NMF, and majority prediction. These completed
comparisons used KNN throughout. Optional logistic regression is now available
for a separate comparison across the same representation paths; its matched
development results and theoretical assessment remain pending.

Each method receives its own explicit parameter grid and is tuned separately
for Smiling and Wearing_Necktie. Within each attribute, shared samples, folds
and numerical settings make paired comparisons interpretable; stratification
may produce different fold assignments for the two attributes. A rank chosen
for supervised NMF must not be imposed on the unscreened baseline.

## Four main parameters

| Parameter | Applies to | Meaning and feasibility |
| --- | --- | --- |
| `n_components` | NMF and PCA | Number of components. Positive integer; check the actual retained-feature count and training-fold dimensions. Numerical rank may impose further restrictions. |
| `percentile` | Percentile screening | Requested fraction of original columns, in percentage points from 0 to 100. Degenerate columns are excluded; a setting that selects no columns is a failed fit. |
| `score_threshold` | Absolute screening | Nonnegative inclusive cutoff on `abs(t)` in pooled-t units. This is not a significance level. Selected feature counts vary across folds and attributes. |
| `n_neighbors` | Every KNN method | Positive integer neighbor count, no larger than the available training observations in each fit. |

Percentile and cutoff are alternatives for separate methods. `solver` and
screening rule are method comparisons. Initialization, coefficient convention,
`max_iter`, `tol`, seed and KNN weighting must also be recorded, but iteration
limits and tolerance are numerical controls rather than substitutes for tuning
the four parameters above.

Declare grid values before a run. Review boundary winners and promising intervals
using training CV, then record any additional grid values. A finite grid or an
exhausted compute allocation does not establish that tuning is complete. Final
ranges and numerical settings must be frozen for each subsequent run. The repeated
default study used its separately saved finite shortlists; it does not establish
a global tuning optimum.

The completed starting grid was:

| Parameter | Starting values |
| --- | --- |
| Rank | 2, 5, 10, 20, 40, 80 |
| Retained percentile | 1, 2, 5, 10, 20, 40, 80, 100 |
| Absolute pooled-t cutoff | 0, 1, 2, 4, 8, 16, 32 |
| KNN neighbors | 1, 3, 5, 9, 15, 25, 41 |

These are starting grids, not selected settings or package defaults. Infeasible
combinations must remain visible in the report. For example, 1% of 4,096 pixels
retains about 40 features, which cannot support rank 80. The exact feature count
and numerical rank must be checked in each fold. The first starting-grid cap was
`max_iter=10000`, with `tol=1e-4` unchanged. In a training-only pilot at rank 80
and the 25th percentile, CD reached the 2,000-iteration cap; the larger cap
allowed that fit to converge in 2,814 iterations. This changes the available
numerical work, not the NMF equations or tolerance. Every candidate and required
fold must still pass its own fit and transform checks; that pilot does not
qualify other ranks, screening settings or samples.

The later refinement and repeated default study used `max_iter=20000` and the
same `tol=1e-4`. This cap increase preserves the objective and stopping tolerance;
qualification remains specific to the tested fits and transformations.

## Data separation and attribute-specific evidence

Use the official training and validation partitions, with the same seeded image
IDs across methods and attributes. Preserve natural class proportions; sample
sizes and seeds must be declared in the run configuration. Fit screening and
factorization inside each of five stratified training folds. Use validation for
the declared development assessment after training-only selection; save and
disclose all development evaluations. Do not use official test data until the
protocol is frozen. Identity overlap remains a separate audit because identity
annotations have not been supplied.

The first substantive development allocation is 4,000 official training images
and 4,000 official validation images, sampled with seed 42. Both attributes use
the same image IDs and the original 64 by 64 grayscale preprocessing. The saved
allocation has the following class counts:

| Attribute | Training negative / positive | Validation negative / positive |
| --- | --- | --- |
| Smiling | 2,036 / 1,964 | 2,061 / 1,939 |
| Wearing_Necktie | 3,706 / 294 | 3,708 / 292 |

These counts describe the chosen development sample, not model performance.
No official test images were loaded for this allocation. This first comparison
does not replace the repeated study or freeze solver/screening defaults.

Smiling and Wearing_Necktie need separate reports. Balanced accuracy is the CV
selection metric for both. Report accuracy, ROC AUC, average precision, positive
precision/recall/F1, prevalence and confusion counts; distinguish average
precision from trapezoidal PR area. For Wearing_Necktie, positive sample counts
and positive recall are essential because majority prediction can have high
accuracy. Do not balance the validation sample or choose a decision threshold
using its labels to improve the reported result.

Wang's R9 concerns misleading accuracy on a rare positive class; it does not
require a balanced resample. In this validation allocation, Wearing_Necktie has
292 positives out of 4,000 observations (7.3%). An all-negative predictor obtains
92.7% accuracy, 50% balanced accuracy and zero positive recall. Report this
baseline alongside every method. Use stratified training folds and the balanced
accuracy selection criterion; do not silently oversample, alter the classifier,
or adjust its decision threshold in response to validation outcomes.

## Numerical and computational checks

A candidate must pass every required fold's fit and transform checks. Preserve
failed/nonconverged attempts; they cannot win using only their successful folds.
Qualify the iteration cap at relevant sample sizes and ranks on training data.
The earlier 500-image rank-2 checks support only those tested configurations.
Increasing an iteration cap preserves the intended method but must be recorded;
relaxing a tolerance changes its numerical qualification.

An implementation may share an identical fold's fitted representation across
KNN neighbor counts. It must preserve the original training-coordinate convention
and produce equivalent cached and uncached predictions and diagnostics. Different
ranks, masks, estimators, numerical settings or input folds require their own
representation fits. Record unique representation fits and classifier fits as
well as total tuning time, final-fit time and peak memory; a cache hit does not
constitute a new independent factorization.

The shared grid retains the initial pipeline's KNN fit so its original training
coordinates can be reused exactly. That fit's cost is included once in the shared
representation-fit timer; per-neighbor KNN fits are timed separately. Thus shared
fit time is not an MF-only measurement. The separate profiler measures the
decomposition and classifier stages at the selected settings.

## Resource plan and execution history

The study uses up to eight worker processes, each limited to one native
numerical thread, on the current machine with 39.23 GiB total physical RAM.
Available RAM is shared with the operating system and other applications; total
physical capacity is not a reserved memory allocation for this study. Monitor
usage while running and keep the worker limit explicit.

The completed study began with four workers and continued with eight. The original
run, `runs/celeba-parameter-study-20261003T085938Z-638645d8/`, is preserved. Its
continuation is `runs/celeba-parameter-study-20261003T091802Z-68d2aa40/`, with a
recorded execution-transition manifest. All 371 completed fold-fit receipts were
preserved byte-for-byte; four uncommitted attempts were retried. Only the worker
count changed: data, sample IDs, folds, source, parameter grids and the one-thread
numerical setting remained identical.

Timing reports must describe this mixed four-to-eight-worker execution. Count
committed fit costs once, retain the transition history, and distinguish elapsed
wall time from the sum of worker compute times. Time spent in interrupted,
uncommitted fits is not recoverable as a complete per-stage timing record.
Memory observations are sampled worker working sets, not a measured peak for
the whole process tree. They also exclude memory used by other applications and
must not be presented as the study's whole-run peak memory.

All 124 grid chunks completed, covering 4,088 declared candidates across the two
attributes and 2,920 committed shared representation-fold attempts. All 22
selected method/attribute models produced valid development-validation results.
Some grid candidates were infeasible or failed numerical qualification and were
excluded from selection; their failure records are retained.

The completed starting grid by itself does not establish convergence for untested
settings, computational speedup, completed tuning or package defaults. Its
training-only review identified the boundaries and numerical qualifications used
for the later refinement/default study above. Wider replication, uncertainty
analysis and final-test evaluation remain open.

## Running the study, profiling and reporting

Use conda environment `py12_df_aocl` from the repository root. The source checkout
or source archive includes these scripts; the library wheel does not. The optional
`study` dependencies can be installed with `python -m pip install -e ".[study]"`
when setting up a new environment. Only the recorded local dependency versions
have been verified. With the existing
fingerprinted local cache, a new explicit-grid run can be started with:

```powershell
conda run --no-capture-output -n py12_df_aocl python examples/celeba_parameter_study.py `
  --data-cache data/celeba-development-4000-seed42 `
  --grid-json examples/celeba_starting_grid.json `
  --max-train 4000 --max-validation 4000 --seed 42 `
  --workers 8 --threads 1 --output runs
```

An existing cache takes precedence over image paths and is verified by its data
fingerprints. Creating a cache also requires `--dir`, `--attributes` and
`--partitions`. To resume an unchanged run, add `--resume RUN_DIRECTORY` with the
same settings. The four-to-eight-worker continuation above was a separately
recorded execution transition, rather than an in-place configuration change.
Keep output paths short enough for the host's filesystem limits.

The specialized `celeba_parameter_study.py` runner remains KNN-only. For an
optional logistic development comparison, use the general example with local
data paths:

```powershell
conda run --no-capture-output -n py12_df_aocl python examples/celeba_classification.py `
  --dir D:/datasets/img_align_celeba `
  --attributes D:/datasets/list_attr_celeba.txt `
  --partitions D:/datasets/list_eval_partition.txt `
  --attribute Smiling --classifier logistic --quick --output runs
```

Replace these paths with the local files. `--quick` verifies execution only;
it does not tune sufficiently or revise the historical default decision.
For a substantive comparison, independently tune `logistic_C` for NMF and
`logistic__C` for the sklearn baseline pipelines, alongside the applicable
representation parameters. Standardization is fitted within training folds.
Record logistic iteration limits, tolerances and class weights separately from
factorization settings, and compare against KNN on matching data and folds.

After the main study has completed, replace `RUN_DIRECTORY` below with its
printed run directory. Review training-only evidence before inspecting validation
performance or choosing a follow-up grid:

```powershell
conda run -n py12_df_aocl python scripts/review_celeba_cv.py RUN_DIRECTORY
```

This review requires the complete declared CV grid. It checks tied winners,
conditional neighboring settings, feasibility limits and matching screening masks.
It reads training CV records and configuration only; it does not select a new grid,
freeze defaults or read validation outcomes.

The separate stage profiler performs additional fits at
the saved selected settings, with one worker and one native thread. Run it alone
to avoid overlap with the main worker pool:

```powershell
conda run --no-capture-output -n py12_df_aocl python scripts/profile_celeba_stages.py `
  --run RUN_DIRECTORY --data-cache data/celeba-development-4000-seed42 --repeats 1
```

The profiler creates a unique directory under `RUN_DIRECTORY/profiles` and
prints its path; `--output` can choose another parent directory. It records
screening, decomposition, classifier and prediction stages separately, retaining
the original study timings alongside them. It checks exact agreement with saved
validation predictions and reports numerical failures or mismatches. Repeats are
timing checks at the same seed, not independent scientific replications. These
additional profiling costs must not replace the original tuning or final-fit
costs.

The reporter reads saved results and draws curves without fitting or selecting
models. It additionally requires Matplotlib. Replace `PROFILE_DIRECTORY` with
the profiler's printed directory, or omit `--profile` to report the main study
without the additional timing fits:

```powershell
conda run -n py12_df_aocl python scripts/report_celeba_study.py RUN_DIRECTORY `
  --scan-folds --profile PROFILE_DIRECTORY/profiling.json
```

The default report destination is `RUN_DIRECTORY/report`; `--output-dir` selects
another location. `--scan-folds` counts numerical outcomes once per shared fold
representation. Generating a report does not mark tuning complete or freeze
solver/screening defaults.
