# Seven pipelines on five additional datasets

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. First expanded benchmark, 6 October 2026 (UTC).

Both independent audit gates passed with zero errors. All **35 planned method/dataset outcomes** are retained: **35 qualified**, **0 unqualified or not evaluated**. The five dataset/tasks and seven pipelines were fixed before new model results; earlier CelebA outcomes were already known. This is one reserved split per dataset, not repeated nested cross-validation or a complete resolution of Wang's replication concern R6.

**Accuracy (%)**

| Method | CMU | FEI | Sonar | Spambase | WDBC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened SPA sNMF + KNN | 73.16 | 94.17 | 88.89 | 85.82 | 95.32 |
| Screened CD sNMF + KNN | 72.63 | 94.17 | 90.48 | 89.06 | 94.15 |
| Screened PCA + KNN | 75.26 | 91.67 | 92.06 | 88.75 | 95.32 |
| Screening-only KNN | 73.16 | 92.50 | 90.48 | 89.38 | 95.32 |
| Raw-feature KNN | 69.47 | 69.17 | 84.13 | 89.38 | 94.74 |
| Unscreened SPA NMF + KNN | 76.84 | 57.50 | 88.89 | 86.53 | 94.74 |
| Unscreened CD NMF + KNN | 65.79 | 88.33 | 87.30 | 89.06 | 90.64 |
| Training-majority constant reference | 49.47 | 50.00 | 53.97 | 60.14 | 62.57 |
| Uniform random guess (expected) | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |

**Total training-fit time (seconds)**

| Method | CMU | FEI | Sonar | Spambase | WDBC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened SPA sNMF + KNN | 0.0672887 | 0.024288 | 0.0134989 | 0.0760748 | 0.0039365 |
| Screened CD sNMF + KNN | 0.0196254 | 0.0816066 | 0.0193511 | 0.122292 | 0.0057306 |
| Screened PCA + KNN | 0.008621 | 0.0107967 | 0.0026068 | 0.0111968 | 0.002705 |
| Screening-only KNN | 0.0060684 | 0.0057191 | 0.0019277 | 0.0051949 | 0.0021595 |
| Raw-feature KNN | 0.0010428 | 0.0009914 | 0.0008615 | 0.0021283 | 0.0012172 |
| Unscreened SPA NMF + KNN | 0.74395 | 0.297157 | 0.013012 | 0.0777953 | 0.0067949 |
| Unscreened CD NMF + KNN | 0.385168 | 3.19985 | 0.129398 | 0.126058 | 0.0219384 |

**Total reserved-set prediction time (seconds)**

| Method | CMU | FEI | Sonar | Spambase | WDBC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened SPA sNMF + KNN | 0.0896358 | 0.0292722 | 0.0192427 | 0.0792473 | 0.0176739 |
| Screened CD sNMF + KNN | 0.0170891 | 0.0169949 | 0.01624 | 0.0306363 | 0.0166849 |
| Screened PCA + KNN | 0.0164032 | 0.0158902 | 0.0155318 | 0.0293179 | 0.0166731 |
| Screening-only KNN | 0.0169961 | 0.0159574 | 0.0161151 | 0.0304382 | 0.016242 |
| Raw-feature KNN | 0.0194216 | 0.0190011 | 0.0158248 | 0.0302918 | 0.0157508 |
| Unscreened SPA NMF + KNN | 0.461409 | 0.112637 | 0.0198128 | 0.0828702 | 0.0227907 |
| Unscreened CD NMF + KNN | 0.027754 | 0.0446531 | 0.0169975 | 0.0314848 | 0.0172361 |

Accuracy is the fraction of correct reserved-set predictions, in percent. The last two accuracy rows are analytical references, separate from the seven fitted methods: the constant rule chooses its label from training counts only, while uniform independent binary guessing has expected accuracy 50%. No random classifier was fitted and no favorable seed selected. `NA (status)` means an unqualified outcome, never zero accuracy. Timings cover the entire final training set or reserved set, include the common input preprocessing, and exclude tuning and the separate implementation-equivalence fits. One numerical thread was used per process; recorded shared-host workload timings are not controlled speed benchmarks. Different cohort sizes also preclude interpreting a cross-dataset time difference as an algorithm effect.

## Cohorts and endpoints

| Dataset | Train rows | Test rows | Train class 0/1 | Test class 0/1 | Train/test subjects | Input features | Input scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMU | 434 | 190 | [219, 215] | [94, 96] | 14 / 6 | 960 | none |
| FEI | 280 | 120 | [140, 140] | [60, 60] | 140 / 60 | 1408 | none |
| Sonar | 145 | 63 | [68, 77] | [29, 34] | Individual cases | 60 | none |
| Spambase | 2942 | 1262 | [1769, 1173] | [759, 503] | Individual cases | 57 | maxabs |
| WDBC | 398 | 171 | [250, 148] | [107, 64] | Individual cases | 30 | maxabs |

CMU predicts documented open-eyes versus sunglasses filename labels. Its reserved cohort contains only **six subjects**, so grouped uncertainty can be wide and must not be replaced by image-level independence. FEI predicts provided frontal filename condition **a=0 versus b=1**; neutral/smile semantic orientation remains unverified and is not inferred from the outcomes. The nonface caches retain their original development/reserved splits and label mappings.

Face splits hold out whole subjects using the frozen seed; the same person never crosses a training/test or inner-CV boundary. Face cache preparation decoded all source images, but reserved model evaluation was gated until all 35 training searches completed and the global freeze passed audit. The three nonface reserved archives were not decoded by the runner before that gate. Exact duplicate checks do not establish the absence of near duplicates or dataset-specific sampling bias.

## Supporting balanced accuracy

| Method | CMU | FEI | Sonar | Spambase | WDBC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened SPA sNMF + KNN | 73.33 | 94.17 | 88.44 | 86.60 | 94.38 |
| Screened CD sNMF + KNN | 72.86 | 94.17 | 89.91 | 88.73 | 92.82 |
| Screened PCA + KNN | 75.45 | 91.67 | 91.63 | 88.13 | 94.38 |
| Screening-only KNN | 73.32 | 92.50 | 90.16 | 88.86 | 94.38 |
| Raw-feature KNN | 69.65 | 69.17 | 83.77 | 88.86 | 93.91 |
| Unscreened SPA NMF + KNN | 77.07 | 57.50 | 88.44 | 86.66 | 93.91 |
| Unscreened CD NMF + KNN | 65.98 | 88.33 | 87.22 | 88.73 | 88.76 |
| Training-majority constant reference | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| Uniform random guess (expected) | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |

Balanced accuracy is the mean of class-specific recalls. Label-independent guessing has expected balanced accuracy 50%; the training-majority constant reference also has 50% balanced accuracy when both test classes are present. Its ordinary accuracy depends on the reserved class counts. There is no test-majority oracle or fitted random classifier in the comparison.

## Selected settings and finite-search limits

Every pipeline was independently selected using the same five inner folds for its dataset: GroupKFold by subject for faces, StratifiedKFold otherwise. Initial ranks were 2, 5, 10, 20, 40; absolute cutoffs 0, 2, 4, 8; K covered every integer 1–101 plus 125, 151, 201. Neighbor values above the minimum inner-training size were explicitly omitted. One deterministic coordinate refinement followed the initial winner. Initial rank infeasibility and screen-dependent failures were retained. No silent rank reduction, fallback method, dataset replacement, or post-test parameter revision was allowed.

Spambase and WDBC use training-fold-fitted MaxAbs scaling for all seven methods; 'raw-feature KNN' means no screening or dimension reduction after that common scaling. CMU, FEI and Sonar use their fixed nonnegative input scales. CD uses NNDSVDa, tolerance 1e-4, and a fixed 20,000 iteration cap; both fit and prediction must qualify. Selected settings can differ across the pipelines, so contrasts do not isolate a single causal solver or screening effect.

| Dataset | Method | Rank | Cutoff | K | Retained features | Evaluation | Unresolved edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CMU | Screened SPA sNMF + KNN | 80 | 2 | 7 | 119 | valid | rank upper=80 |
| CMU | Screened CD sNMF + KNN | 5 | 2 | 12 | 119 | valid | None recorded |
| CMU | Screened PCA + KNN | 40 | 2 | 13 | 119 | valid | None recorded |
| CMU | Screening-only KNN | — | 2 | 7 | 119 | valid | None recorded |
| CMU | Raw-feature KNN | — | — | 5 | 960 | valid | None recorded |
| CMU | Unscreened SPA NMF + KNN | 80 | — | 21 | 960 | valid | rank upper=80 |
| CMU | Unscreened CD NMF + KNN | 20 | — | 6 | 960 | valid | None recorded |
| FEI | Screened SPA sNMF + KNN | 20 | 2 | 11 | 206 | valid | None recorded |
| FEI | Screened CD sNMF + KNN | 4 | 2 | 5 | 206 | valid | None recorded |
| FEI | Screened PCA + KNN | 4 | 2 | 3 | 206 | valid | None recorded |
| FEI | Screening-only KNN | — | 2 | 1 | 206 | valid | None recorded |
| FEI | Raw-feature KNN | — | — | 25 | 1408 | valid | None recorded |
| FEI | Unscreened SPA NMF + KNN | 40 | — | 7 | 1408 | valid | None recorded |
| FEI | Unscreened CD NMF + KNN | 40 | — | 45 | 1408 | valid | None recorded |
| Sonar | Screened SPA sNMF + KNN | 40 | 0 | 1 | 60 | valid | rank upper=40 |
| Sonar | Screened CD sNMF + KNN | 10 | 2 | 1 | 26 | valid | None recorded |
| Sonar | Screened PCA + KNN | 20 | 1 | 1 | 44 | valid | None recorded |
| Sonar | Screening-only KNN | — | 1 | 1 | 44 | valid | None recorded |
| Sonar | Raw-feature KNN | — | — | 1 | 60 | valid | None recorded |
| Sonar | Unscreened SPA NMF + KNN | 40 | — | 1 | 60 | valid | rank upper=40 |
| Sonar | Unscreened CD NMF + KNN | 20 | — | 1 | 60 | valid | None recorded |
| Spambase | Screened SPA sNMF + KNN | 40 | 2 | 5 | 56 | valid | rank upper=40 |
| Spambase | Screened CD sNMF + KNN | 30 | 0 | 1 | 57 | valid | None recorded |
| Spambase | Screened PCA + KNN | 40 | 1 | 3 | 57 | valid | rank upper=40 |
| Spambase | Screening-only KNN | — | 0 | 1 | 57 | valid | None recorded |
| Spambase | Raw-feature KNN | — | — | 1 | 57 | valid | None recorded |
| Spambase | Unscreened SPA NMF + KNN | 40 | — | 12 | 57 | valid | rank upper=40 |
| Spambase | Unscreened CD NMF + KNN | 30 | — | 1 | 57 | valid | None recorded |
| WDBC | Screened SPA sNMF + KNN | 10 | 8 | 1 | 20 | valid | None recorded |
| WDBC | Screened CD sNMF + KNN | 2 | 8 | 4 | 20 | valid | None recorded |
| WDBC | Screened PCA + KNN | 20 | 4 | 5 | 25 | valid | None recorded |
| WDBC | Screening-only KNN | — | 4 | 5 | 25 | valid | None recorded |
| WDBC | Raw-feature KNN | — | — | 3 | 30 | valid | None recorded |
| WDBC | Unscreened SPA NMF + KNN | 30 | — | 3 | 30 | valid | None recorded |
| WDBC | Unscreened CD NMF + KNN | 5 | — | 7 | 30 | valid | None recorded |

A recorded edge marks an unresolved explored-range boundary. No recorded edge does not establish global optimization or cover untested combinations of refined coordinates. Retained real-data pixels/features are not a known generating support.

The frozen searches retain **38,854 candidates** and **37,411 failed candidate-fold scores**. Scores and failures are not omitted from the selection denominator. Candidate-fold scores share representation fits across K values; they are not counts of separately fitted decompositions. Selected CV scores below are optimistic tuning statistics, not independent validation.

| Dataset | Method | Qualified / total candidates | Failed candidate-folds | Initially omitted K | Selected CV BA (%) | Tuning wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CMU | Screened SPA sNMF + KNN | 1460 / 2084 | 3120 | 0 | 81.75 | 41.6278 |
| CMU | Screened CD sNMF + KNN | 1460 / 2084 | 3120 | 0 | 76.44 | 63.6239 |
| CMU | Screened PCA + KNN | 1460 / 2084 | 3120 | 0 | 81.42 | 38.5977 |
| CMU | Screening-only KNN | 314 / 418 | 520 | 0 | 81.21 | 9.67201 |
| CMU | Raw-feature KNN | 104 / 104 | 0 | 0 | 67.32 | 3.16826 |
| CMU | Unscreened SPA NMF + KNN | 522 / 522 | 0 | 0 | 75.28 | 19.3845 |
| CMU | Unscreened CD NMF + KNN | 522 / 522 | 0 | 0 | 61.51 | 34.0484 |
| FEI | Screened SPA sNMF + KNN | 1980 / 2084 | 416 | 0 | 93.57 | 46.791 |
| FEI | Screened CD sNMF + KNN | 1980 / 2084 | 416 | 0 | 95.00 | 94.3652 |
| FEI | Screened PCA + KNN | 1980 / 2084 | 416 | 0 | 94.29 | 43.3043 |
| FEI | Screening-only KNN | 418 / 418 | 0 | 0 | 93.21 | 10.8146 |
| FEI | Raw-feature KNN | 104 / 104 | 0 | 0 | 69.64 | 2.95066 |
| FEI | Unscreened SPA NMF + KNN | 522 / 522 | 0 | 0 | 59.64 | 18.0429 |
| FEI | Unscreened CD NMF + KNN | 521 / 522 | 1 | 0 | 75.36 | 383.513 |
| Sonar | Screened SPA sNMF + KNN | 1012 / 2022 | 5050 | 3 | 83.23 | 21.6576 |
| Sonar | Screened CD sNMF + KNN | 912 / 2024 | 5152 | 3 | 86.65 | 41.29 |
| Sonar | Screened PCA + KNN | 1012 / 2024 | 5060 | 3 | 82.97 | 23.6822 |
| Sonar | Screening-only KNN | 305 / 406 | 505 | 3 | 82.97 | 6.52254 |
| Sonar | Raw-feature KNN | 101 / 101 | 0 | 3 | 81.17 | 1.46157 |
| Sonar | Unscreened SPA NMF + KNN | 506 / 506 | 0 | 3 | 83.23 | 8.55669 |
| Sonar | Unscreened CD NMF + KNN | 406 / 507 | 101 | 3 | 81.07 | 29.1356 |
| Spambase | Screened SPA sNMF + KNN | 1979 / 2083 | 520 | 0 | 88.64 | 170.677 |
| Spambase | Screened CD sNMF + KNN | 1978 / 2082 | 520 | 0 | 89.07 | 147.56 |
| Spambase | Screened PCA + KNN | 1978 / 2082 | 520 | 0 | 88.50 | 130.618 |
| Spambase | Screening-only KNN | 417 / 417 | 0 | 0 | 88.26 | 31.1109 |
| Spambase | Raw-feature KNN | 104 / 104 | 0 | 0 | 88.26 | 7.66355 |
| Spambase | Unscreened SPA NMF + KNN | 521 / 521 | 0 | 0 | 88.45 | 46.1632 |
| Spambase | Unscreened CD NMF + KNN | 521 / 521 | 0 | 0 | 89.07 | 39.2018 |
| WDBC | Screened SPA sNMF + KNN | 1563 / 2084 | 2604 | 0 | 96.77 | 33.779 |
| WDBC | Screened CD sNMF + KNN | 1563 / 2084 | 2605 | 0 | 95.32 | 38.2254 |
| WDBC | Screened PCA + KNN | 1563 / 2084 | 2605 | 0 | 97.55 | 33.0689 |
| WDBC | Screening-only KNN | 418 / 418 | 0 | 0 | 97.55 | 8.40272 |
| WDBC | Raw-feature KNN | 104 / 104 | 0 | 0 | 96.88 | 1.72768 |
| WDBC | Unscreened SPA NMF + KNN | 418 / 522 | 520 | 0 | 96.88 | 9.01405 |
| WDBC | Unscreened CD NMF + KNN | 418 / 522 | 520 | 0 | 93.30 | 9.96582 |

## Paired differences: screened SPA minus each comparator

All differences below are percentage points; signs always mean screened SPA minus the named comparator. Intervals are descriptive, unadjusted 95% percentile intervals from 10,000 paired bootstrap draws with the declared seed. Nonface draws resample cases within class; face draws resample entire subjects with every image and its multiplicity retained. Any single-class group draw is counted and omitted as specified in its receipt, not replaced by an adaptive extra draw. Face-image McNemar tests are not used because images within a subject are dependent.

| Dataset | Comparator | Accuracy difference [95% interval] | BA difference [95% interval] | SPA-only / comparator-only correct |
| --- | --- | ---: | ---: | ---: |
| CMU | Screened CD sNMF + KNN | +0.53 [-14.14, +13.90] | +0.47 [-14.10, +13.54] | 23 / 22 |
| CMU | Screened PCA + KNN | -2.11 [-6.38, +2.11] | -2.13 [-6.43, +2.08] | 6 / 10 |
| CMU | Screening-only KNN | +0.00 [-5.73, +4.79] | +0.01 [-5.72, +4.64] | 7 / 7 |
| CMU | Raw-feature KNN | +3.68 [-3.68, +10.42] | +3.68 [-3.65, +10.41] | 23 / 16 |
| CMU | Unscreened SPA NMF + KNN | -3.68 [-12.17, +3.14] | -3.75 [-12.35, +3.10] | 6 / 13 |
| CMU | Unscreened CD NMF + KNN | +7.37 [+1.05, +14.58] | +7.35 [+0.81, +14.55] | 32 / 18 |
| FEI | Screened CD sNMF + KNN | +0.00 [-2.50, +2.50] | +0.00 [-2.50, +2.50] | 2 / 2 |
| FEI | Screened PCA + KNN | +2.50 [-0.83, +6.67] | +2.50 [-0.83, +6.67] | 5 / 2 |
| FEI | Screening-only KNN | +1.67 [-1.67, +5.00] | +1.67 [-1.67, +5.00] | 3 / 1 |
| FEI | Raw-feature KNN | +25.00 [+18.33, +31.67] | +25.00 [+18.33, +31.67] | 32 / 2 |
| FEI | Unscreened SPA NMF + KNN | +36.67 [+30.00, +43.33] | +36.67 [+30.00, +43.33] | 46 / 2 |
| FEI | Unscreened CD NMF + KNN | +5.83 [+0.00, +11.67] | +5.83 [+0.00, +11.67] | 11 / 4 |
| Sonar | Screened CD sNMF + KNN | -1.59 [-7.98, +4.76] | -1.47 [-8.62, +5.17] | 2 / 3 |
| Sonar | Screened PCA + KNN | -3.17 [-12.70, +6.35] | -3.19 [-13.03, +6.90] | 3 / 5 |
| Sonar | Screening-only KNN | -1.59 [-11.11, +7.94] | -1.72 [-11.82, +8.37] | 4 / 5 |
| Sonar | Raw-feature KNN | +4.76 [-3.17, +12.70] | +4.67 [-3.70, +13.29] | 5 / 2 |
| Sonar | Unscreened SPA NMF + KNN | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0 / 0 |
| Sonar | Unscreened CD NMF + KNN | +1.59 [-7.94, +11.11] | +1.22 [-8.62, +10.80] | 5 / 4 |
| Spambase | Screened CD sNMF + KNN | -3.25 [-5.23, -1.27] | -2.13 [-4.08, -0.15] | 64 / 105 |
| Spambase | Screened PCA + KNN | -2.93 [-4.99, -0.95] | -1.53 [-3.55, +0.49] | 68 / 105 |
| Spambase | Screening-only KNN | -3.57 [-5.63, -1.51] | -2.26 [-4.37, -0.11] | 75 / 120 |
| Spambase | Raw-feature KNN | -3.57 [-5.63, -1.51] | -2.26 [-4.37, -0.11] | 75 / 120 |
| Spambase | Unscreened SPA NMF + KNN | -0.71 [-2.06, +0.63] | -0.06 [-1.41, +1.33] | 34 / 43 |
| Spambase | Unscreened CD NMF + KNN | -3.25 [-5.23, -1.27] | -2.13 [-4.08, -0.15] | 64 / 105 |
| WDBC | Screened CD sNMF + KNN | +1.17 [-1.75, +4.09] | +1.56 [-1.72, +5.00] | 4 / 2 |
| WDBC | Screened PCA + KNN | +0.00 [-1.75, +1.75] | +0.00 [-2.34, +2.34] | 1 / 1 |
| WDBC | Screening-only KNN | +0.00 [-1.75, +1.75] | +0.00 [-2.34, +2.34] | 1 / 1 |
| WDBC | Raw-feature KNN | +0.58 [-1.17, +2.92] | +0.47 [-1.88, +2.81] | 2 / 1 |
| WDBC | Unscreened SPA NMF + KNN | +0.58 [-1.17, +2.92] | +0.47 [-1.88, +2.81] | 2 / 1 |
| WDBC | Unscreened CD NMF + KNN | +4.68 [+0.58, +8.77] | +5.62 [+0.62, +10.78] | 11 / 3 |

These intervals condition on the selected training cohort, fitted models and observed dataset. They do not measure full retraining uncertainty, constitute multiplicity-adjusted confirmation, or establish general solver superiority. An interval crossing zero is inconclusive, not evidence of equivalence. The complete dataset-by-method table retains negative results and numerical failures; this study did not continue searching until a favorable dataset appeared.

## Audit and reproducibility

Pretest audit: **passed**, zero errors. Posttest audit: **passed**, zero errors. The posttest receipt explicitly binds this exact aggregate result file. The audits check frozen sources/data, group/fold plans, full candidate receipts, selection arithmetic, predictions and paired comparisons; they do not independently refit the scientific models or regenerate CV predictions.

Saved evidence: protocol (retained locally: `../runs/multi-dataset-validation-20261006/protocol.json`), preparation (retained locally: `../runs/multi-dataset-validation-20261006/preparation.json`), final settings freeze (retained locally: `../runs/multi-dataset-validation-20261006/final-freeze.json`), results (retained locally: `../runs/multi-dataset-validation-20261006/results.json`), pretest audit (retained locally: `../runs/multi-dataset-validation-20261006/pretest-audit.json`), posttest audit (retained locally: `../runs/multi-dataset-validation-20261006/posttest-audit.json`).

Result SHA256: `8703d9556bc544703b54f21d71d2d944b8f3b3e676059c1ce7954c3f9d4a1112`. Posttest-audit SHA256: `d3e5806c2ddad35e2dbbb2e979875bb19b871ca69ffb5a54e678eab1e1b23533`. Report-generator SHA256: `f29af7ed42e981eedfb73648e86fa8c74e21e87ee0383190df265e4b8edab5ee`.
