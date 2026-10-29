# Evidence for Wang's comments on supervised NMF

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. Development validation, October 2026.

The October 5–6, 2026 SPA classification results use their frozen study sources:
anchor-derived training coordinates (`training_coordinates="fit_transform"`)
and fixed-basis NNLS projections for queries. Since version 0.3.0, the default
`NMFClassifier` projects SPA training observations as well. The accuracy and
timing results here describe the recorded procedures, rather than a rerun of
that later default. See the
[coordinate migration](release-0.3.0.md#spa-training-coordinates-migration).

**Completed and independently audited: 900 simulation draws and 780 CelebA
evaluations, with zero final-fit failures or audit discrepancies.** The
evidence supports a classifier-dependent contribution, not a general NMF
advantage. The BLAS implementation reduces measured fit-plus-projection time
by 12.6-16.6% without changing the paired benchmark outputs.

This study evaluates the package's absolute-screening sNMF procedure against
screened PCA, screening without factorization, raw features, and unscreened
NMF, with both KNN and logistic regression where specified. It preserves
unfavorable comparisons. The objective is a reproducible response to Wang's
empirical questions, not a predetermined claim that NMF must win.

## Studies and inferential scope

| Evidence | Design | What its uncertainty describes |
| --- | --- | --- |
| Independent simulations | 27 signal/support/correlation cells plus cross-block correlation, null and XOR; 30 independent draws per regime; 160 training rows, 480 test rows, 320 features; complete five-fold tuning inside each draw | Monte Carlo uncertainty for these specified generators and finite parameter grids |
| CelebA, independently retuned | Ten overlapping 2,000-image training resamples, each independently tuned with five-fold CV and one local refinement step; two attributes and 13 methods | Variation of the tuning-and-fitting procedure conditional on a fixed training reservoir and the same 4,000 reused validation images |
| CelebA, fixed settings | Twenty overlapping training resamples after a separate 2,000-image tuning pilot; same two attributes, methods and validation images | Variation from training resampling, conditional on the fixed pilot-selected settings |
| Fashion-MNIST reanalysis | Existing seven-method, three-task KNN results; 21 audited evaluation receipts; no new fitting | Paired validation-image bootstrap uncertainty conditional on the fixed models and three tasks |
| Historical numerical diagnosis | Original generator, sample size, dimensions, split and reported rank; 11 controlled diagnostic arms | A one-dataset numerical diagnosis, not a repeated predictive comparison |
| BLAS benchmark | Four official-training-only workloads, seven randomized paired repeats after warm-up, one native thread | Local implementation timing and numerical equivalence on the measured workloads |

The CelebA resamples exclude their common tuning pilot but overlap one another.
No official image-test outcomes were accessed. Validation labels were already
known from package development when this follow-up was designed. Identity
annotations were unavailable, so there is no claim about unseen-person
performance. The ten retuned samples are also the first ten samples of the
fixed-settings arm; these are related analyses, not independent confirmations.

The complete CV procedure refits screening, representation and classifier
inside each training fold. Candidates must qualify in every fold; a failed
method is recorded without a replacement prediction. Finite grids and one
local refinement do not establish globally optimal tuning. Numerical
qualification is necessary, but does not prove a global NMF optimum.

Logistic regression uses training-only standardization and, in CelebA, independently
tuned class weights. KNN uses unscaled coordinates and uniform votes. Their
comparison concerns these complete classifier procedures, not an isolated effect
of a parametric link function. The matched PCA ablation holds SPA's chosen rank
and screening cutoff fixed and tunes PCA's downstream classifier independently
over the starting classifier grid, without a further local refinement. It
controls mask and dimension, not identical classifier settings or search
resolution. The primary independently tuned PCA and SPA paths both receive
the full starting-grid plus local-refinement procedure.

## Does the factorization improve prediction?

All **900 independently retuned simulation draws** and **9,930 final method
evaluations** are complete and numerically qualified, with no final-fit or
prediction warnings. The finite search attempted 99,960 parameter candidates
and 499,800 folds; 23,015 infeasible fold fits were rejected and preserved
(12,215 empty screens, 7,200 excessive NMF ranks and 3,600 excessive PCA ranks).
The simulation ran for 1,665.20 seconds (27.75 minutes) using six workers with
one numerical thread each on a shared host. Its independent complete audit
passed with zero discrepancies across 4,766,400 saved predictions and 11,757
manifest files. Both CelebA studies are also complete and passed the final
combined audit.

The six display regimes were fixed before generating fresh outcomes. Values
below are balanced accuracy in percent, mean +/- Monte Carlo SE across 30
independent draws. The full 27-cell factorial and three stress regimes are in
the simulation tables (retained locally: `../runs/wang-simulations-blas-20261006/balanced_accuracy_summary.csv`).

| Prespecified display regime | SPA + KNN | PCA + KNN | SPA + logistic | PCA + logistic |
| --- | ---: | ---: | ---: | ---: |
| Weak signal: SNR 0.5, support 16, correlation 0 | 61.49 +/- 1.52 | 67.65 +/- 1.32 | 69.98 +/- 1.29 | 74.55 +/- 0.71 |
| Strong signal: SNR 2.5, support 16, correlation 0 | 93.35 +/- 2.46 | 97.26 +/- 1.65 | 96.36 +/- 2.45 | 97.24 +/- 1.92 |
| Correlated: SNR 1.25, support 16, correlation 0.9 | 71.25 +/- 0.68 | 71.03 +/- 0.67 | 73.74 +/- 0.47 | 74.24 +/- 0.37 |
| Sparse: SNR 1.25, support 4, correlation 0 | 77.72 +/- 2.12 | 80.62 +/- 1.72 | 85.93 +/- 1.38 | 87.16 +/- 1.31 |
| Null | 49.55 +/- 0.31 | 49.84 +/- 0.44 | 49.83 +/- 0.40 | 49.71 +/- 0.49 |
| XOR | 49.75 +/- 0.39 | 49.84 +/- 0.45 | 50.15 +/- 0.38 | 49.74 +/- 0.42 |

Screened SPA improves substantially over raw KNN in the displayed additive
settings, but that does not isolate the value of factorization: screened PCA
and screening alone are essential comparators. In the weak-signal display
case, SPA minus PCA is **-6.16 BA points** with paired pointwise 95% t interval
**[-7.84, -4.48]** for KNN, and **-4.57 [-6.75, -2.39]** for logistic regression.
All prespecified paired contrasts, including CD and screening-only comparisons,
remain in the saved tables.

The XOR result exposes a limitation of marginal screening: SPA + KNN averages
49.75%, while a separately labelled KNN oracle supplied the generating support
achieves **99.87%** (MC SE 0.03 points). The paired oracle advantage is
50.12 points [49.32, 50.93]. That oracle has information unavailable to the
practical methods and is not a deployable competitor.

Strong additive signal does not remove instability in this representation.
The SPA + KNN results range from 47.29% to 100%, with 3/30 below 75%; SPA +
logistic ranges from 40.63% to 100%, with 2/30 below 75%. Those low-score cases
retain all 16 generating signal features, plus 2-6 background features. They
are qualified fits, not convergence failures. These observations distinguish
support inclusion from predictive success without establishing a causal
explanation for the instability. Approximate t intervals are not truncated
and can extend outside the metric's [0,1] range.

A separately labelled numerical check refitted seven already-selected
low-score SPA configurations with their exact frozen parameters using both
the pre-BLAS reference and the study's recorded BLAS implementation. All
anchors, factors, coordinates, labels and probabilities were bitwise identical
and matched the saved study predictions. This rules out the BLAS change as the
explanation for those cases; it is not additional tuning or a replacement result.

The later [October 20 anchor and information diagnostic](wang-simulation-validation.md#october-20-anchor-and-information-diagnostic)
examined all seven original low-score triggers and three already inspected
follow-up cohorts. Its controls implicate low-rank representation loss and
coordinate scaling, while retaining cases where rank expansion does not repair
prediction. This retrospective evidence supplements the explanation; it does
not replace the October 6 results or establish a universal remedy.

### CelebA: ten independently retuned training samples

All **260/260 evaluations** qualify, with no final convergence warnings or
fallbacks. Each value below is balanced accuracy in percent, mean +/-
conditional Monte Carlo SE across ten training samples. These are complete
tuning-and-fitting repetitions, conditional on the finite reservoir and reused
validation set, rather than ten new datasets or unseen-person evaluations.

| Procedure | Smiling | Wearing Necktie |
| --- | ---: | ---: |
| Screened SPA + KNN | 82.49 +/- 0.39 | 56.43 +/- 0.42 |
| Screened SPA + logistic | 85.89 +/- 0.26 | 71.65 +/- 0.69 |
| Screened PCA + KNN | 80.60 +/- 0.43 | 55.51 +/- 0.24 |
| Screened PCA + logistic | 87.35 +/- 0.22 | 73.77 +/- 0.49 |
| Screening only + KNN | 80.30 +/- 0.40 | 56.31 +/- 0.75 |
| Screening only + logistic | 88.70 +/- 0.13 | 74.60 +/- 0.32 |
| Raw KNN | 60.98 +/- 0.28 | 52.30 +/- 0.31 |
| Raw logistic | 87.64 +/- 0.09 | 73.94 +/- 0.34 |
| Unscreened SPA + KNN | 57.86 +/- 0.50 | 53.36 +/- 0.30 |
| Unscreened SPA + logistic | 61.81 +/- 1.33 | 69.20 +/- 0.41 |
| Matched PCA + KNN | 80.20 +/- 0.50 | 56.15 +/- 0.47 |
| Matched PCA + logistic | 86.89 +/- 0.19 | 73.22 +/- 0.63 |
| Majority prediction | 50.00 +/- 0.00 | 50.00 +/- 0.00 |

The contribution is **classifier-dependent**. SPA helps the KNN procedure on
Smiling, including against the secondary matched-mask/rank PCA procedure.
For logistic regression, PCA and simpler pixel baselines perform better than
SPA on both attributes. For Necktie KNN, the interval for SPA versus PCA includes
zero. Paired differences across the ten training samples are:

| Contrast | Smiling: BA points [pointwise 95% t interval] | Necktie: BA points [pointwise 95% t interval] |
| --- | ---: | ---: |
| SPA KNN minus independently tuned PCA KNN | 1.90 [1.318, 2.476] | 0.92 [-0.249, 2.090] |
| SPA logistic minus independently tuned PCA logistic | -1.46 [-1.747, -1.180] | -2.13 [-3.685, -0.565] |
| SPA KNN minus matched-mask/rank PCA KNN | 2.30 [1.665, 2.929] | 0.28 [-0.682, 1.249] |
| SPA logistic minus matched-mask/rank PCA logistic | -1.01 [-1.473, -0.540] | -1.57 [-2.837, -0.304] |
| SPA logistic minus SPA KNN | 3.39 [2.947, 3.836] | 15.21 [13.396, 17.034] |

These intervals describe conditional training-resample variation, including
tuning; they are unadjusted for multiple comparisons and are not population
confidence intervals. The paired image-bootstrap intervals in the raw reports
instead condition on the seed61010 fitted case. Repeated predictions of a
validation image are never pooled as independent observations.

The related fixed-setting study completed **520/520 qualified evaluations**.
For Smiling, SPA/PCA KNN means are 82.30%/78.88%, and SPA/PCA logistic means are
84.84%/87.23%. For Necktie, the respective means are 55.87%/55.48% and
68.78%/72.58%. Its 20-resample tables measure fitting variability after a
disjoint pilot chooses the settings; they are not 20 full tuning replications.
See [the complete CelebA methods and results](wang-real-data-validation.md).

### Necktie: accuracy alone conceals poor minority prediction

There are 292 positives among 4,000 validation images (7.3%). Always predicting
negative gives 92.7% accuracy, zero positive recall and 50% balanced accuracy.
Means over the ten fully retuned fits are:

| Procedure | ROC AUC | AP | Positive precision | Positive recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened SPA + KNN | 0.647 | 0.154 | 0.349 | 0.152 | 0.210 |
| Screened SPA + logistic | 0.776 | 0.259 | 0.167 | 0.721 | 0.270 |
| Screened PCA + logistic | 0.817 | 0.316 | 0.200 | 0.697 | 0.310 |
| Screening only + logistic | 0.842 | 0.349 | 0.235 | 0.664 | 0.346 |
| Raw logistic | 0.849 | 0.367 | 0.253 | 0.624 | 0.360 |
| Majority prediction | 0.500 | 0.073 | 0.000 | 0.000 | 0.000 |

Logistic SPA greatly improves minority recall relative to KNN SPA, but at low
precision. This does not make it the best procedure: raw or screened-pixel
logistic regression and PCA logistic provide stronger overall comparisons.
Exact confusion counts and PR curves are preserved, with curve figures using
the prespecified seed61010 case rather than a selected best fit.

### Fashion-MNIST: retain the unfavorable comparison

The already completed Fashion-MNIST reanalysis does **not** support a general
NMF advantage. Across the three fixed tasks, SPA sNMF with KNN averaged 88.57%
balanced accuracy, compared with 90.12% for screened PCA, 89.95% for screening
alone and 89.90% for raw KNN. All procedures use their historical training-CV
choices. Paired differences are:

| SPA sNMF minus comparator | BA difference, percentage points | Pointwise 95% paired bootstrap interval |
| --- | ---: | ---: |
| Screened PCA | -1.550 | [-2.300, -0.817] |
| Screening alone | -1.383 | [-2.083, -0.717] |
| Raw KNN | -1.333 | [-2.067, -0.633] |

The 5,000 bootstrap draws resample images within class and retain pairing
across methods. They do not include training, tuning or task-population
uncertainty, and are unadjusted for multiple comparisons. See the
[Fashion-MNIST report](wang-fashion-reanalysis.md) for every task and its
qualification/provenance checks.

## Why did historical standard NMF give 32.42%?

The original-size reconstruction gave **32.75%**, close to the dissertation's
rounded 32.42%. The rank-2 coordinate-descent model reached the old iteration
cap of 400 during factorization and projection. Increasing only that cap to
2,000, at the same tolerance, gave **92.92%**; factorization stopped at
iteration 1,226 without recorded convergence warnings. Recomputing the
training coordinates against the final converged basis gave 94.58%.

The same observations, split, rank and classifier were used. Specified row or
basis normalization changes and conversion to float64 did not remove the
cap-400 failure. These controls support insufficient optimization as a cause
in the reconstruction. They do not establish the original runtime behavior,
identify separate factorization/projection contributions, or validate the
proposed clipping/topological-inversion explanation. Original historical
predictions and exact library versions remain unavailable. The
[historical diagnostic report](wang-historical-diagnosis.md) records all 11
arms, confusion matrices and convergence evidence.

## BLAS implementation and measured speed

The requested optimization uses SciPy's public BLAS `dgemv` and `dger` calls
for SPA's residual updates and prepares the fixed NNLS design matrix once per
transform. It avoids the old full outer-product temporary and repeated layout
conversions. The model, classifier, public NNLS solver and numerical failure
policy stay the same.

Across four training-only workloads, median fitting-plus-projection time fell
by **12.6–16.6%**. All 28 measured before/after pairs produced identical anchor
indices, factors, coordinates, classifier labels and probabilities. Seven
numerical regression tests cover additional layouts, near ties, ill-conditioning
and failure behavior. The final 303-test suite passed
in both the primary Python 3.12 environment and the pinned Python 3.10
compatibility environment. See the [BLAS report](blas-optimization.md) for
separate timings, versions and limits of the equivalence claim.

The earlier scientific runs were interrupted and preserved when this change
was requested. The successor studies use fresh source freezes and tuning
records; results from the old partial runs are not pooled with them. Study
jobs share this host, so their observed wall times are not isolated algorithm
speed comparisons. The separate serial BLAS benchmark provides that comparison.

A two-repeat, fresh-process memory supplement recorded native Windows lifetime
peak working sets of about 265-535 MiB across the same four workloads. Peaks
were essentially unchanged before/after optimization. They include imports,
data and screening; they are not factorization-only or whole-CV memory. Other
CelebA jobs remained active, and these cold-run timings are not used to claim
speed improvements. Full scientific-study peak RSS remains unmeasured.

## What remains outside these results

- Image results are development evidence. A further untouched, frozen
  official-test or external-cohort evaluation is required for a confirmatory
  claim, and person-level separation remains unverified.
- Simulations use reduced dimensions relative to the earlier study proposal.
  They do not establish large-scale behavior, NMF separability, exact low rank,
  a population support-recovery guarantee, posterior sufficiency, or any
  theorem's assumptions. Empirical generating-support inclusion is measured.
- The empirical classifiers are finite tuned procedures. Their results do not
  prove statistical consistency or supply the missing classifier-specific
  theoretical arguments.
- Remaining tuning boundaries, numerical failures and weak/negative comparisons
  must be reported. The study does not change package defaults automatically.
  After the specified local refinement, 9/20 primary fixed-pilot settings and
  81/200 primary retuned settings retain at least one unresolved search edge.
  Final fits all qualify; infeasible CV candidates remain explicitly rejected.
- Dissertation proof edits, literature attribution and figure permissions are
  separate from package validation.

## Reproducibility and supporting records

The final combined audit (retained locally: `../runs/wang-validation-audit-20261006/all-audit.json`)
passed with zero discrepancies. It checked all 10,710 new final evaluation
records, 113,156 CV parameter candidates and 565,780 candidate-fold records, and
recomputed metrics for **7,886,400 individual predictions**. All 120
SPA/matched-PCA mask/rank pairs agree exactly, and all 48 fixed-case
image-bootstrap intervals were reproduced. Historical and Fashion audits
remain separately bound by their recorded hashes.

The audit reconstructs simulation data fingerprints, seeds, folds and screening
masks; verifies qualified winner selection and source/protocol/data bindings;
and recomputes final prediction metrics and aggregate/paired uncertainty.
CV fold predictions were not retained, so their scores are checked from
receipts rather than independently refitting every candidate. Local timestamps
and checksums establish internal consistency, not external preregistration.

Observed CelebA tuning command times were 347.67 seconds for the fixed pilot
and 2,110.46 seconds summed across the ten independent retunings. Final
evaluation command times were 1,712.01 and 583.96 seconds, respectively. These
are shared-host wall times for different workloads, not a direct speed
comparison. The real-data stage allocation is coarse: screen timing includes
column-selection copies, and majority fit/predict stage zeros are analytical
placeholders. Actual total wall times and shared representation costs are
preserved in the run records.

The frozen protocols retain source snapshots, exact training/validation row IDs,
data and source hashes, seeds, folds, parameter candidates, failures, selected
settings and individual predictions. Machine-readable records are stored under
the following local run directories; raw image data are not included in the
package distribution.

| Study | Run directory | Design and reproduction instructions |
| --- | --- | --- |
| Fresh simulations | `runs/wang-simulations-blas-20261006` | [Simulation protocol](wang-simulation-validation.md) |
| CelebA, ten retuned samples | `runs/wang-real-retuned-blas-20261006` | [CelebA protocol](wang-real-data-validation.md) |
| CelebA, twenty fixed-setting samples | `runs/wang-real-blas-20261006` | [CelebA protocol](wang-real-data-validation.md) |
| Fashion-MNIST reanalysis | `runs/wang-fashion-reanalysis-20261006` | [Paired reanalysis](wang-fashion-reanalysis.md) |
| Historical diagnosis | `runs/wang-historical-20261006` | [Numerical diagnosis](wang-historical-diagnosis.md) |
| BLAS performance | `runs/blas-spa-20261006/aocl` | [Benchmark](blas-optimization.md) |
| Independent result audit | `runs/wang-validation-audit-20261006` | All new scientific runs passed; see `all-audit.json` |

The [concern tracker](wang-concerns.md) maps evidence and remaining work to
Wang's individual comments. Interpretation should distinguish code validation,
predictive comparisons, diagnostic evidence and conditional theoretical claims.
