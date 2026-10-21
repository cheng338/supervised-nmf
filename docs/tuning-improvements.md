# Tuning and reporting improvements

Author: Chang Cheng. Version 0.1.0a2.

This update adds subject-aware calibration, bounded rank exploration, reused
training-fold screening statistics, and evaluation summaries. It preserves the
SPA/absolute-screening/KNN defaults and the dissertation's screening statistic,
factorization objectives, training-coordinate convention, and numerical caps.

## Keep subjects together

Both calibration helpers accept `groups` with one subject identifier per row.
They construct stratified group folds, then reject any split with subject
overlap or insufficient class support. A supplied `folds` sequence instead
fixes the exact `(training_indices, validation_indices)` pairs. Each validation
row must appear exactly once, and each training fold must be its complement.
Both classes must be present in validation and have at least two training
observations. Arbitrary binary class labels are preserved.

`data_ids` provides optional unique sample identifiers. Data values, row order,
subject and sample identifiers, actual folds, settings, source, and numerical
runtime are bound to checkpoints. Changing any binding requires a new search
directory. No automatic reseeding or replacement of infeasible folds occurs.

## Expand rank with a declared budget

```python
from supervised_nmf import NMFClassifier
from supervised_nmf.search import calibration_rank_search

report = calibration_rank_search(
    NMFClassifier(random_state=42),
    {"n_components": [2, 5, 10, 20],
     "score_threshold": [0.0, 2.0, 4.0],
     "n_neighbors": [3, 5, 9]},
    X_train, y_train, groups=training_subjects,
    cv=5, seed=42, threads=1,
    max_expansions=2, max_rank=80,
    checkpoint_dir="runs/grouped-rank-search",
)
print(report["selected_parameters"])
print(report["stop_reason"])
print(report["unresolved_upper_rank_boundary"])
```

The initial grid is explicit. A qualified upper-rank winner may receive one
doubled-rank candidate per expansion, keeping its other parameters fixed.
Only new proposals are bounded by training-fold dimensions and the optional
rank cap. Initial candidates are never silently clamped or discarded.
Ties preserve the earlier candidate. This is coordinate exploration for KNN
pipelines, not joint optimization of rank, screening, and neighbors.

Reaching the expansion budget or a user rank cap leaves the boundary unresolved.
`search_complete` describes only the declared upper-rank policy;
`tuning_complete` remains false and no global optimum is claimed. All failed
and unconverged candidates remain recorded. The function neither refits on
all training data nor evaluates an outer validation or test set. Freeze and
numerically qualify the selected final fit separately.

In a training-only diagnostic of 29 earlier boundary selections, one additional
rank improved mean CV balanced accuracy in 10 cases and reduced it in 19. The
largest improvement was 2.35 percentage points. Nine improving cases remained
at the expanded budget boundary. Higher rank also substantially increased CD
cost. These local comparisons motivate explicit budgets and boundary reporting;
they do not justify a higher default rank or establish held-out improvement.

## Reuse screening without changing it

`calibration_knn_grid` computes pooled-t statistics within each training fold
and reuses them across compatible cutoffs and ranks. Each requested mask keeps
the same handling of ties, constant columns, and infinite scores. Empty masks
and impossible ranks become explicit failed candidate records before
factorization. They do not trigger a lower threshold or an automatically
reduced rank.

This optimization applies to recognized NMFClassifier and screening/PCA/KNN
layouts. Other sklearn pipelines retain their ordinary fit path. Shared
screening, representation fitting, projection, and classifier costs are counted
separately; a cached representation is not charged once per neighbor count.
Checkpoint reuse retains numerical failures as well as successful fits.

The independent diagnostic matched 372 dimensionally impossible representation
folds and 528 valid folds in saved receipts. Thirty direct valid before/after
model pairs produced identical predictions. This establishes checked
equivalence, not a general end-to-end tuning speed guarantee.

## Display imbalance and convergence

`evaluate_classifier` can receive training labels, subject identifiers, and
sample identifiers along with the evaluation data. It rejects supplied
training/evaluation overlap before prediction. The majority-class reference
is determined from training labels only. A missing training reference remains
unavailable rather than being inferred from evaluation labels.

`format_evaluation_summary` presents accuracy, balanced accuracy, both class
recalls and support counts, the available training-majority reference, subject
counts, and numerical qualification. Support thresholds flag limitations;
they do not rebalance, omit, or replace observations. Subject counts alone do
not provide confidence intervals or establish independence beyond the supplied
identifiers.

The classifier separately records factorization, optional training projection,
final-classifier fitting, and latest prediction-projection convergence. The
iteration cap and tolerance are unchanged. Numerical convergence is not a
global-optimality certificate.

## Projection decision

SPA retains public SciPy NNLS for fixed-basis projection. The direct BLAS anchor
updates and once-per-transform matrix-layout preparation preserve the numerical
method. Paired training-only benchmarks measured 11.5–17.0% lower fit-plus-
projection time on four workloads, with identical tested outputs.

An unconditional QR-based NNLS replacement was rejected. The real-model checks
passed for KNN and logistic regression in both numerical environments, including
both training-coordinate conventions. Synthetic nonnegative full-rank stress
systems nevertheless exceeded the predefined coefficient-agreement tolerance.
Those systems met the condition-number bound but were not bases learned from
the face datasets; no classification loss was demonstrated. QR was also slower
for one single-image workload. Four-worker NNLS was slower in the original
three-workload comparison. Neither alternative is included in this update.

## Compatibility and verification

This section records version 0.1.0a2 validation. Python 3.10 support ended for
current development on October 20, 2026; see the
[current compatibility policy](compatibility.md).

Python 3.10 support in 0.1.0a2 did not introduce a slower shared solver,
version-dependent fallback, or dependency downgrade in the primary environment.
That release used one pure-Python wheel. The BLAS optimization above was timed in the primary
Python 3.12/AOCL environment; compatibility checks do not replace those
performance measurements.

The exact upload source passed the following checks on Windows 11:

| Environment | NumPy / SciPy / scikit-learn | Source tests | Installed-wheel checks |
| --- | --- | --- | --- |
| Python 3.12.14, primary AOCL environment | 2.5.2 / 1.18.0 / 1.9.0 | 327 passed | SPA, MU, CD with KNN and logistic regression |
| Python 3.10.12, pinned compatibility environment | 2.2.6 / 1.15.3 / 1.7.0 | 327 passed | SPA, MU, CD with KNN and logistic regression |

Both installed-wheel checks also exercised grouped rank exploration and the
evaluation formatter. The source tests include checkpoint reuse, invalid-rank
preflight, screening equivalence, subject separation, and numerical failure
reporting. These checks establish compatibility for the tested profiles; their
wall times are not comparative performance benchmarks. Ubuntu was not tested
for this update.
