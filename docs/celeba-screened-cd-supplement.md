# Screened CD-NMF+KNN on the existing CelebA identity cohort

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. Supplement designed 6 October 2026 (UTC).

The SPA comparisons below retain the October 6 frozen predictions, obtained
with anchor-derived training coordinates (`training_coordinates="fit_transform"`)
and fixed-basis NNLS query projections. Since [version 0.3.0](release-0.3.0.md#spa-training-coordinates-migration),
the default SPA classifier also projects training observations. These historical
comparisons do not measure that later default. CD retains its fitted-coordinate
training convention.

**Complete: both independent audits passed with zero errors.** Screened CD sNMF+KNN achieved **79.0% accuracy and 78.9756% balanced accuracy** on the existing 1,000 test identities. Screened SPA achieved 81.4% and 81.3725%, respectively: its paired balanced-accuracy advantage over screened CD was **2.3970 percentage points**, with a descriptive 95% interval of **[0.1841, 4.6178]**. Screened CD improved on unscreened CD, but its differences from screened PCA and screening-only KNN were inconclusive.

This seventh method is supervised screening followed by coordinate-descent (CD) NMF and KNN. It restores the supervised solver family used in the dissertation and compares it with both the frozen screened SPA pipeline and the separately tuned unscreened CD baseline.

**All six earlier methods' test outcomes were known when this supplement was designed.** It reuses the same 1,000 test identities and is descriptive supplementary evidence, not another independent holdout confirmation. The [original four-method validation](celeba-identity-validation.md) and [two-method unscreened supplement](celeba-unscreened-supplement.md) remain unchanged. Their frozen predictions, rather than new fits, supply the six comparators.

| Method | Absolute screening cutoff | Retained pixels | Rank | K | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Screened SPA sNMF+KNN, original | 8 | 124 | 20 | 11 | 81.4% | 81.3725% |
| Screened PCA+KNN, original | 8 | 124 | 40 | 10 | 78.5% | 78.5094% |
| Screening-only KNN, original | 6 | 218 | — | 10 | 79.6% | 79.5758% |
| Raw KNN, original | None | 4,096 | — | 28 | 62.2% | 61.9848% |
| Unscreened SPA NMF+KNN, earlier supplement | None | 4,096 | 80 | 18 | 61.9% | 61.8227% |
| Unscreened CD NMF+KNN, earlier supplement | None | 4,096 | 80 | 101 | 73.0% | 72.9532% |
| **Screened CD sNMF+KNN, this supplement** | 8 | 124 | 10 | 39 | **79.0%** | **78.9756%** |

The six existing rows come from their original results (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/results.json`) and unscreened results (retained locally: `../runs/celeba-unscreened-supplement-20261006/results.json`). The new row comes from the completed screened-CD results (retained locally: `../runs/celeba-screened-cd-supplement-20261006/results.json`). All seven evaluations qualified. Both unscreened methods selected the upper explored rank of 80; those unresolved limits continue to apply to their new comparisons.

The complete comparison figure is available as PNG (retained locally: `../runs/celeba-screened-cd-supplement-20261006/seven-method-comparison.png`) and SVG (retained locally: `../runs/celeba-screened-cd-supplement-20261006/seven-method-comparison.svg`), with its figure manifest (retained locally: `../runs/celeba-screened-cd-supplement-20261006/figure-manifest.json`).

## Why screened CD belongs in the comparison

Chang Cheng's dissertation, Chapter 7's methodology and Section 7.3.2, describes CD with NNDSVDa initialization for the supervised NMF experiments, followed by Euclidean KNN on the fitted nonnegative coefficients. It also says that both the supervised and standard unsupervised NMF face pipelines used CD. Thus screened CD is directly relevant to the dissertation's method, not just another arbitrary benchmark.

The [preserved CelebA implementation](../provenance/original/snmf_am.py), lines 193–206 and 255–275, confirms this: its training pipeline uses `SelectPercentile(score_func=f_classif)`, CD NMF with NNDSVDa initialization, and KNN with three neighbors. The final supervised fit uses the selected percentile and rank, `max_iter=400`, and seed 42. The October 6 identity study's primary face result instead uses FeatureSPA and an absolute feature-score cutoff. The earlier unscreened CD supplement restored the dissertation's standard unsupervised baseline; the present addition separately restores its supervised factorization family.

This is not an exact historical reproduction. It retains the current absolute screening rule, identity-based cohort, dense neighbor grid, independent hyperparameter selection, and a fixed CD iteration budget of 20,000. Those choices differ from the historical percentile screen, fixed three-neighbor classifier, cap of 400, seed, and data split. Historical dissertation accuracies must therefore not be treated as targets for this evaluation.

This addition speaks most directly to the [Wang concern tracker's](wang-concerns.md) R15 solver comparison, R7 question about what NMF contributes beyond screening, and R21 reproducibility. Final convergence evidence also matters for the numerical concerns behind R8. It does not by itself resolve R6's request for broader replication, prove feature recovery or consistency, or identify the cause of a historical result. The [historical numerical reconstruction](wang-historical-diagnosis.md) remains a separate simulation diagnostic.

## What the new comparisons can identify

The label **screened CD sNMF** is intentional: using labels for screening makes this a supervised pipeline even though the subsequent CD factorization minimizes a reconstruction objective without directly receiving labels.

| Comparison | Scientific interpretation | Remaining qualification |
|---|---|---|
| Screened CD versus screened SPA | Compare independently tuned supervised NMF pipelines using different factorization solvers. | These selected pipelines use the same cutoff and 124-pixel mask, but different ranks and K; the result is not an isolated causal solver effect at one fixed configuration. |
| Screened CD versus unscreened CD | Compare independently tuned CD pipelines with and without supervised screening. | Rank and K may differ; the unscreened rank remains at its explored upper boundary. This is not a fixed-configuration causal screening effect. |
| Screened CD versus screening-only KNN | Assess whether the selected CD representation improves on a separately tuned pixel-screening pipeline. | Each procedure chooses its own cutoff and K; this is a whole-pipeline comparison. |
| Screened CD versus screened PCA | Compare independently tuned screened representations with the same classifier family. | This concerns the selected procedures on this cohort, not universal NMF-versus-PCA superiority. |
| Screened CD versus raw KNN or unscreened SPA | Supply additional context using the preserved reference predictions. | More than one pipeline component can differ. |

No additional fixed-setting ablation is implied by these comparisons. Every comparator is reported below, regardless of ranking. An interval crossing zero does not establish equivalence.

## Preserved data and frozen calibration

The supplement uses the same immutable cache (retained locally: `../data/celeba-identity-validation-20261006/manifest.json`) and sampling plan (retained locally: `../data/celeba-identity-validation-20261006/plan.json`): 2,000 official-training images, one per identity, and the existing saved test inputs from 1,000 separate official-test identities. There are 1,029 non-Smiling and 971 Smiling training labels; the known test counts are 490 and 510. No rows, labels, images, task, or seed were selected using the new test result. Training and test identities are disjoint. Exact image-duplicate checks passed in the earlier study; they do not establish the absence of near duplicates or annotation errors.

Inputs remain 64-by-64 grayscale images, flattened to 4,096 nonnegative `float32` values divided by 255. No learned input rescaling or coefficient normalization is added. Screening and CD factorization are fitted only on each training fold. Final training coordinates use `fit_transform`; test coordinates use the frozen selected features and fitted NMF basis.

The frozen starting grid matches the earlier screened methods: absolute cutoffs `{0, 2, 4, 8}`, ranks `{2, 5, 10, 20, 40}`, and every integer K from 1 through 101 plus 125, 151, and 201. This is **2,080 initial candidates**: four cutoffs, five ranks, and 104 neighbor values. The same five stratified folds, seed 62106, select by mean training balanced accuracy. Because each training identity occurs once, these folds cannot place multiple images of the same annotated person on both sides.

CD uses NNDSVDa initialization, Frobenius loss, tolerance `1e-4`, fixed seed 62006, and `max_iter=20000`, with no regularization. KNN uses uniform Euclidean votes and the existing deterministic decision rule. The 20,000 cap is a budget, not proof of convergence or global optimality; fit and out-of-sample projection must separately qualify.

One deterministic coordinate-wise refinement followed the initial training-CV winner, using the same rule as the previous methods. It changed one coordinate at a time; candidates were derived from this method's own winner. Selection pools only candidates that qualify on all five folds. Failures and omitted infeasible proposals would remain in the receipts, and successful subsets of failed candidates would not be averaged into eligibility. No rank was silently reduced and no fallback was used. Representation fits were shared across K values, so candidate-fold scores and decomposition fits are counted separately below.

The immutable run is `runs/celeba-screened-cd-supplement-20261006`, produced by the screened-CD driver (retained locally: `../scripts/validate_celeba_screened_cd.py`), with sole method key `screened_cd`. Its protocol (retained locally: `../runs/celeba-screened-cd-supplement-20261006/protocol.json`), saved at **05:30:12 UTC** on 6 October, binds scientific sources, environment, cache, and all prior evidence by checksum. Final evaluation used the public `NMFClassifier` and a separately recorded equivalence check against the matching calibration pipeline. The audit before this evaluation was not an audit before first test exposure: all earlier test outcomes were already known.

## Training calibration and final numerical checks

The selected-settings receipt (retained locally: `../runs/celeba-screened-cd-supplement-20261006/tuning/screened_cd/selection.json`) completed at **05:36:44 UTC** and the final freeze (retained locally: `../runs/celeba-screened-cd-supplement-20261006/final-freeze.json`) at **05:36:45 UTC**, before supplemental evaluation. The selected setting remained the initial winner: **absolute cutoff 8, rank 10, K 39**, with **81.1046% selected five-fold training balanced accuracy**. This CV score is an optimistic selection statistic, not independent validation or a test result.

| Calibration quantity | Completed value |
|---|---:|
| Initial candidates | 2,080 |
| Added one-coordinate candidates | 4 |
| Qualified / evaluated candidates | 2,084 / 2,084 |
| Qualified / evaluated candidate-fold scores | 10,420 / 10,420 |
| Failed candidates / candidate folds | 0 / 0 |
| Shared representation-fold fits, initial + refinement | 100 + 20 |
| Warning records in shared CV fits/projections | 0 |
| Omitted physical-bound proposals | 0 |
| Recorded tuning wall time | 381.4425 s |
| Sum of recorded CV compute times | 334.8114 s |

The four actual refinement proposals were `(cutoff 16, rank 10, K 39)`, `(cutoff 6, rank 10, K 39)`, `(cutoff 8, rank 15, K 39)`, and `(cutoff 8, rank 8, K 39)`. These are separate one-coordinate candidates, not a Cartesian grid. No new K value was needed because the proposed adjacent values were already in the dense initial range. All proposals qualified; the higher cutoff remained feasible for rank 10. The initial and refinement reports preserve all candidate scores and shared fits, and the original setting won their pooled comparison.

The selection receipt has **no unresolved coordinate edges**. In particular, cutoff 8 was compared with 16 and rank 10 with 8 and 15, while K 39 lies inside the initial neighbor range. This does not establish a global optimum or cover every untested combination of refined coordinates. The earlier unscreened rank-80 boundaries remain unresolved. Timings include study computation and record writing; they are not an isolated speed benchmark.

Final fit, test projection, and public-pipeline equivalence qualified independently of CV. The evaluation receipt (retained locally: `../runs/celeba-screened-cd-supplement-20261006/evaluations/screened_cd.json`) records **705 CD iterations**, successful fit and last-transform convergence, and **zero fit, prediction, or equivalence warnings**. It retains all 1,000 predictions, with no failed final evaluation or fallback. Screening selected 124 of 4,096 pixels, with zero constant features and zero infinite feature scores. These are exactly the same 124 pixels selected by the original screened SPA fit; they are not a known ground-truth facial-feature support.

The public `NMFClassifier(solver='cd', screening='absolute')` and separately fitted calibration pipeline had identical predicted labels, identical selected-feature masks, and maximum probability difference **0.0**. Training-coordinate comparisons passed at `rtol=1e-6, atol=1e-7`; the declared probability tolerance was `rtol=atol=1e-12`. This check verifies implementation agreement, not another independent evaluation sample.

| Recorded operation | Time |
|---|---:|
| Full training fit | 0.3314 s |
| Prediction on 1,000 test rows | 0.0462 s |
| Separate equivalence fit/check | 0.4022 s |

The fit's internal timers record 0.1066 s for screening and 0.2100 s for factorization. These workload timings are not a controlled solver-speed comparison and exclude the separate tuning effort. The equivalence fit/check is separate from the primary full-training fit.

## Screened-CD test metrics

All 1,000 planned test rows were evaluated. Class 0 is non-Smiling (490 rows); class 1 is Smiling (510 rows). The matrix below uses true labels as rows and predicted labels as columns, ordered `[0, 1]`.

| Metric | Screened CD sNMF+KNN |
|---|---:|
| Accuracy | 79.0000% |
| Balanced accuracy | 78.9756% |
| Class-0 precision | 79.0456% |
| Class-0 recall | 77.7551% |
| Class-0 F1 | 78.3951% |
| Class-1 precision | 78.9575% |
| Class-1 recall | 80.1961% |
| Class-1 F1 | 79.5720% |
| ROC AUC | 0.875686 |
| Average precision | 0.886578 |
| Confusion matrix | `[[381, 109], [101, 409]]` |

This table includes every saved scalar classification metric. ROC AUC and average precision use the frozen KNN probability for class 1; no test-dependent decision threshold was fitted.

## Paired results and uncertainty

The direction below is **screened CD minus each prior method**, in percentage points, with all six comparators retained. The 95% intervals use 10,000 paired class-stratified bootstrap resamples with seed 62107. The discordance column counts screened-CD-only correct rows, then comparator-only correct rows. Exact McNemar p-values are two-sided and concern accuracy, not balanced accuracy.

| Comparator | Balanced-accuracy difference [95% interval] | Accuracy difference [95% interval] | CD-only / comparator-only correct | McNemar p, accuracy |
|---|---:|---:|---:|---:|
| Screened SPA sNMF | −2.3970 [−4.6178, −0.1841] | −2.4 [−4.6, −0.2] | 52 / 76 | 0.0416383 |
| Screened PCA | +0.4662 [−1.8328, 2.7112] | +0.5 [−1.8, 2.7] | 71 / 66 | 0.7326841 |
| Screening-only KNN | −0.6002 [−3.0452, 1.8948] | −0.6 [−3.0, 1.9] | 75 / 81 | 0.6890543 |
| Raw KNN | +16.9908 [13.6755, 20.2981] | +16.8 [13.5, 20.1] | 250 / 82 | 6.8310 × 10⁻²¹ |
| Unscreened SPA NMF | +17.1529 [13.4334, 20.7784] | +17.1 [13.4, 20.7] | 273 / 102 | 3.9109 × 10⁻¹⁹ |
| Unscreened CD NMF | +6.0224 [3.1031, 8.9097] | +6.0 [3.1, 8.9] | 142 / 82 | 0.0000737103 |

The saved keys are `primary_snmf`, `primary_spca`, `primary_screen`, `primary_raw`, `unscreened_spa`, and `unscreened_cd`. A seventh direction, `screened_spa_minus_screened_cd`, reverses the first row: **SPA minus CD balanced accuracy +2.3970 [0.1841, 4.6178] points; accuracy +2.4 [0.2, 4.6] points**. Its 76/52 discordance and p-value 0.0416383 are the same paired evidence with the labels reversed, not an independent result.

All new contrasts, intervals, and p-values are **descriptive and unadjusted**. They do not form a fresh confirmatory family merely because settings were frozen before these additional fits. In particular, the narrow exclusion of zero for the SPA–CD comparison must not be interpreted as multiplicity-adjusted evidence or universal SPA solver superiority. The selected screened CD pipeline improved over the selected unscreened CD pipeline, while its differences from screening-only and screened PCA remain inconclusive.

The sampling unit is one annotated test identity. Fixed-model resampling describes uncertainty conditional on the selected training cohort, fitted models, and observed class counts. It does not include model-selection uncertainty, the choice to add screened CD after observing the six earlier results, variation across new training samples or datasets, or inference to all face-recognition tasks. The Smiling task itself was selected using earlier development evidence, as disclosed in the original report.

The analytical chance references remain separate from the seven fitted methods. Label-independent guessing has expected balanced accuracy 50%. Predicting the training-majority label 0 for every row gives 49% accuracy and 50% balanced accuracy on the known 490/510 test split; choosing its test-majority label 1 would give 51% accuracy but uses test-label knowledge. No random-classifier fit or favorable random seed is added.

## Audit and saved evidence

All **six focused synthetic tests** passed in both numerical environments before protocol freeze. The primary-runtime receipt (retained locally: `../tmp/celeba-screened-cd-tests-primary-20261006.json`) and pinned-runtime receipt (retained locally: `../tmp/celeba-screened-cd-tests-pinned-20261006.json`) bind the test commands, scientific-source hashes, log hashes, runtime versions, and successful exit codes. This is six focused tests per environment, not a claim of a newly run full-package suite.

The pre-evaluation audit (retained locally: `../runs/celeba-screened-cd-supplement-20261006/pretest-audit.json`) passed at **05:37:14 UTC**, after the final settings freeze and before the additional test evaluation. It verified the exact protocol, source/data bindings, full candidate grid, folds, qualification, winner/refinement logic, and final-setting bindings: **2,084 candidates, 10,420 candidate-fold scores, 120 shared representation-fold receipts, and 120 independently reconstructed screening masks**, with zero errors. All shared representations happened to qualify in this run; in general, the mask check applies to qualified receipts and does not count missing failed-fit masks as independently verified.

Evaluation completed at **05:37:37 UTC**. The post-evaluation audit (retained locally: `../runs/celeba-screened-cd-supplement-20261006/posttest-audit.json`) passed at **05:38:04 UTC**, with zero errors. It independently reconstructed the final full-training mask, recomputed metrics from **1,000 new predictions**, rechecked **6,000 prior predictions**, verified all **seven saved contrast directions**, and recomputed all **21 distinct pairs** among the seven qualified methods. It verified that the earlier six methods' artifacts remained unchanged.

The auditor did not refit the models or regenerate CV predictions. It checked source-bound CV receipts and implementation-equivalence receipts; screening arithmetic, final saved prediction metrics, and paired statistics were independently reconstructed. No parameters, source, or test rows were revised after this supplemental result. The completed comparison extends the empirical solver evidence on this one task while retaining the known-test design and independently tuned pipeline limitations.
