# Restoring unscreened NMF+KNN to the CelebA comparison

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. Supplement designed 6 October 2026 (UTC).

The subsequent [screened CD supplement](celeba-screened-cd-supplement.md) adds
screening followed by CD-NMF and KNN as a seventh method, retaining all six
results below. That report compares the two independently tuned screened
NMF pipelines and separately discloses its known-test design.

**Completed; both independent audit gates passed with zero errors.** On these 1,000 CelebA test identities, screened sNMF+KNN obtained **81.37% balanced accuracy**, versus **72.95% for standard CD NMF+KNN** and **61.82% for unscreened SPA NMF+KNN**. Its paired advantage over standard CD was **8.42 percentage points**, with a descriptive 95% interval of **[5.53, 11.30]**. Standard NMF+KNN was in the dissertation; omitting it from the four-method comparison was incorrect. The baseline is now restored.

**This supplement was designed after the original official-test results were known.** It reuses the same test cohort; it is not a new independent holdout confirmation. Both unscreened methods selected rank 80, the upper explored boundary, so these results do not establish their best possible performance. The original four methods, their selected settings, predictions, and audited results remain unchanged in the [identity-based validation report](celeba-identity-validation.md).

| Method | Screening cutoff | Retained pixels | Rank | K | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Screened sNMF (SPA)+KNN | 8 | 124 | 20 | 11 | 81.4% | 81.3725% |
| Screened PCA+KNN | 8 | 124 | 40 | 10 | 78.5% | 78.5094% |
| Screening-only KNN | 6 | 218 | — | 10 | 79.6% | 79.5758% |
| Raw KNN | None | 4,096 | — | 28 | 62.2% | 61.9848% |
| Unscreened SPA NMF+KNN, supplement | None | 4,096 | 80 | 18 | 61.9% | 61.8227% |
| Standard CD NMF+KNN, supplement | None | 4,096 | 80 | 101 | 73.0% | 72.9532% |

All six evaluations are qualified. Screening uses an absolute feature-score cutoff. The comparison figure is available as PNG (retained locally: `../runs/celeba-unscreened-supplement-20261006/six-method-comparison.png`) and SVG (retained locally: `../runs/celeba-unscreened-supplement-20261006/six-method-comparison.svg`), with its figure manifest (retained locally: `../runs/celeba-unscreened-supplement-20261006/figure-manifest.json`). Exact new metrics, predictions, settings, and contrasts are in results.json (retained locally: `../runs/celeba-unscreened-supplement-20261006/results.json`).

## What the dissertation and preserved scripts contain

Chang Cheng's dissertation, Section 7.2.2, explicitly includes full-feature unsupervised NMF fitted by coordinate descent, followed by downstream KNN, as one of the simulation comparisons. Section 7.3.2 and Table 7.2 include standard unsupervised NMF in the CelebA comparison alongside raw KNN and supervised NMF. The baseline is therefore part of the dissertation's comparison design.

The preserved [CelebA script](../provenance/original/snmf_am.py), in its standard-NMF block at lines 289–313, implements:

- All input pixels, without supervised feature screening.
- Scikit-learn NMF with `solver='cd'`, `init='nndsvda'`, `max_iter=400`, and `random_state=42`.
- A rank inherited as `optimal_K` from the supervised search, followed by three-neighbor KNN.
- `fit_transform` coordinates for training and fixed-basis `transform` coordinates for the test set.

The preserved [simulation script](../provenance/original/sim.py) likewise includes the standard-NMF comparison; the simulation uses five-neighbor classification. The different K values in the face and simulation sources should not be conflated. The *Supervised NMF Experiment Study Draft* also expressly calls for independently tuning standard NMF rather than borrowing the supervised rank.

The previous face validation focused on screened SPA, screened PCA, screening-only KNN, and raw KNN. Those controls addressed useful comparisons but left the dissertation's unscreened NMF baseline missing. Adding only an unscreened SPA row would still not restore the dissertation's coordinate-descent baseline; the two supplemental rows address different questions.

## Two supplemental methods

| Supplemental method | Features and representation | Purpose |
|---|---|---|
| Unscreened SPA NMF+KNN | All 4,096 grayscale pixels; FeatureSPA; KNN | Compare screening choices while retaining the current SPA solver family. |
| Standard CD NMF+KNN | All 4,096 grayscale pixels; coordinate-descent NMF; KNN | Restore the dissertation's standard unsupervised NMF baseline family. |

Both use `screening='none'`, uniform KNN votes, and the existing `fit_transform` training-coordinate convention. The final estimator is the public `NMFClassifier`; its predictions are checked against the matching calibration pipeline. SPA uses an NNLS iteration cap of 20,000. CD uses NNDSVDa initialization, Frobenius loss, tolerance `1e-4`, iteration cap 20,000, and fixed seed 62006. No learned pixel rescaling or coefficient normalization is introduced.

The higher CD cap is deliberate. The [earlier historical numerical diagnostic](wang-historical-diagnosis.md) found that the original cap of 400 could materially distort a synthetic standard-NMF comparison when fit and projection had not converged. A cap of 20,000 does not itself establish convergence or a global optimum: warnings and convergence status must still be checked, and unqualified candidates cannot win. That earlier diagnosis does not establish the cause of the dissertation's CelebA result.

This restores a **baseline family**, not an exact historical reproduction. The cohort, rank/neighbor search, iteration cap, seed, software environment, and model-selection treatment differ from the original script. In particular, each supplemental method selects its own rank and K; it does not inherit settings from sNMF or the dissertation.

Comparing screened and unscreened SPA holds the solver family fixed, but independently tuned ranks and K differ. The resulting difference compares the complete tuned pipelines; it is not the causal effect of removing screening at one identical fixed configuration. Comparing standard CD with screened SPA additionally changes the factorization solver.

## Preserved data and training-only tuning

The supplement uses the existing prepared training cache (retained locally: `../data/celeba-identity-validation-20261006/manifest.json`): 2,000 official-training images, one per annotated identity, with 1,029 class-0 and 971 class-1 labels. Inputs remain grayscale 64 by 64, flattened to 4,096 `float32` values divided by 255. The evaluation reuses the **exact saved test inputs** from the dense-neighbor study: 1,000 images from 1,000 separate official-test identities, with class counts 490 and 510. It does not draw another test sample or decode replacement images to obtain a different result.

Each supplemental method starts from ranks `{2, 5, 10, 20, 40}` and the same 104 K values used in the original dense search: every integer 1–101 plus 125, 151, and 201. That is 520 initial rank/K candidates per method, before refinement. Both use the same five stratified training folds with seed 62106 and independent selection by mean training balanced accuracy. Representation fitting can be shared across K values, while each candidate's fold scores remain recorded.

One deterministic coordinate-wise refinement follows each initial winner, using the existing refinement rule and physical bounds. Its actual proposals depend on the supplemental training-CV winner; they must not be borrowed from the screened methods' results. Every candidate must qualify on all five folds. Failures, warnings, nonconvergence, omitted infeasible proposals, ties, and unresolved boundaries remain in the receipts. No fallback or silent rank reduction is permitted. A complete finite search and one local refinement do not claim global optimization.

The separate run is `runs/celeba-unscreened-supplement-20261006`, with method keys `spa` and `cd`. The runner (retained locally: `../scripts/validate_celeba_unscreened.py`) and package/helper sources are bound by the saved protocol (retained locally: `../runs/celeba-unscreened-supplement-20261006/protocol.json`), which also records environment, seeds, data hashes, and prior-result hashes. Saved records include `tuning/{spa,cd}/selection.json`, `final-freeze.json`, both audit receipts, `evaluations/{spa,cd}.json`, and `results.json`. The original four-method run was not refitted, relabeled, or overwritten.

The supplemental `pretest-audit.json` name means an audit **before the new supplemental evaluations**, not an audit before any exposure to these test outcomes. The original outcomes were already known when this supplement was requested. Freezing the supplemental settings and checking all training-only choices still protects the computation from further unrecorded adaptation, but it cannot make this reused test cohort pristine again.

## Completed training calibration

The final settings freeze (retained locally: `../runs/celeba-unscreened-supplement-20261006/final-freeze.json`) was saved at **05:09:12 UTC** on 6 October 2026, before either supplemental test evaluation. Both methods have qualified selections. The scores below are **optimistically selected training-CV scores**, not supplemental test results.

| Method | Selected rank | Selected K | Five-fold training balanced accuracy | Initial + added candidates | Qualified / evaluated | Recorded tuning wall time |
|---|---:|---:|---:|---:|---:|---:|
| Unscreened SPA NMF+KNN | 80 | 18 | 61.1025% | 520 + 2 | 522 / 522 | 151.5818 s |
| Standard CD NMF+KNN | 80 | 101 | 67.9070% | 520 + 3 | 523 / 523 | 1,137.0123 s |

The SPA selection (retained locally: `../runs/celeba-unscreened-supplement-20261006/tuning/spa/selection.json`) refined its initial `(rank 40, K 18)` winner by proposing rank 30 and rank 80 at K=18. The CD selection (retained locally: `../runs/celeba-unscreened-supplement-20261006/tuning/cd/selection.json`) refined `(rank 40, K 101)` by proposing `(rank 30, K 101)`, `(rank 80, K 101)`, and `(rank 40, K 113)`. These are separate one-coordinate candidates, not a Cartesian grid of refined values. Neither method omitted a proposal for physical infeasibility.

All **1,045 candidates** and **5,225 candidate-fold evaluations** qualified, with zero failed folds and zero warning records in the initial and refinement reports. SPA used 35 shared representation-fold fits (25 initial, ten refined); CD used 40 (25 initial, 15 refined). Candidate-fold scores therefore must not be mistaken for separate decomposition fits. Recorded tuning wall times include the study's computation and receipt-writing workload and are not a controlled isolated speed benchmark.

**Both selected ranks of 80 remain at the upper explored boundary.** Both source-bound receipts explicitly retain that unresolved edge. One predeclared expansion was completed, without a further adaptive amendment. Consequently the study does not establish the best possible unscreened SPA or CD configuration or a global optimum. The original four-method study used the same finite initial rank range and the same single-refinement rule, but this fairness of procedure does not remove the remaining rank-range limitation for these two baselines. The separate final-fit, projection, and pipeline-equivalence checks also passed, as recorded below.

## Supplemental test metrics and numerical qualification

Both methods evaluated all 1,000 test rows. Class 0 is not smiling (490 rows); class 1 is smiling (510 rows). The following table reports every saved scalar classification metric, with precision, recall, and F1 also shown separately for both classes. AUC and average precision use the class-1 KNN probability. Confusion matrices have true labels in rows and predicted labels in columns, ordered `[0, 1]`.

| Metric | Unscreened SPA NMF+KNN | Standard CD NMF+KNN |
|---|---:|---:|
| Accuracy | 61.9000% | 73.0000% |
| Balanced accuracy | 61.8227% | 72.9532% |
| Class-0 precision | 61.8736% | 73.3051% |
| Class-0 recall | 57.9592% | 70.6122% |
| Class-0 F1 | 59.8525% | 71.9335% |
| Class-1 precision | 61.9224% | 72.7273% |
| Class-1 recall | 65.6863% | 75.2941% |
| Class-1 F1 | 63.7488% | 73.9884% |
| ROC AUC | 0.659346 | 0.807073 |
| Average precision | 0.654369 | 0.823059 |
| Confusion matrix | `[[284, 206], [175, 335]]` | `[[346, 144], [126, 384]]` |

The SPA evaluation (retained locally: `../runs/celeba-unscreened-supplement-20261006/evaluations/spa.json`) and CD evaluation (retained locally: `../runs/celeba-unscreened-supplement-20261006/evaluations/cd.json`) retain the full diagnostics and each prediction. Both final fits and test projections converged, with **zero fit, prediction, or equivalence warnings**, no failed final evaluations, and no fallback. CD stopped after **2,199 iterations**, below its fixed 20,000 cap. SPA records **80 anchor-selection steps**; that count has a different meaning from CD iterations.

| Recorded final operation | Unscreened SPA | Standard CD |
|---|---:|---:|
| Fit on all 2,000 training rows | 8.0445 s | 137.2111 s |
| Predict the 1,000 test rows | 8.0201 s | 0.3013 s |
| Separate pipeline-equivalence fit/check | 16.1246 s | 142.9625 s |

The public `NMFClassifier(screening='none')` and its independently fitted matching pipeline gave identical test labels and **maximum probability difference 0.0** for each method. Training-coordinate checks also passed. Equivalence is a computation check, not another test sample. Timings are recorded workloads, not a controlled performance benchmark; the separate equivalence fits are not included in the final-fit times above.

## Paired differences on the reused test cohort

All differences below are **first named method minus second named method**, in percentage points. The 95% intervals use 10,000 paired, class-stratified bootstrap resamples with seed 62107. They are unadjusted descriptive intervals conditional on the selected training data, fitted models, and observed test class counts. They do not account for designing this supplement after seeing the earlier test outcomes, uncertainty from choosing the model or training sample, or variability across new datasets.

| Comparison | Balanced-accuracy difference [95% interval] | Accuracy difference [95% interval] |
|---|---:|---:|
| Screened sNMF − standard CD | +8.4194 [5.5282, 11.2966] | +8.4 [5.5, 11.3] |
| Screened sNMF − unscreened SPA | +19.5498 [16.0304, 23.1394] | +19.5 [16.0, 23.1] |

The complete saved supplemental contrast family is retained below, including comparisons with every original method and the comparison between the two new methods. The two reversed sNMF rows repeat the preceding differences with the required supplemental-minus-original direction; they are not additional independent evidence.

| Comparison | Balanced-accuracy difference [95% interval] | Accuracy difference [95% interval] |
|---|---:|---:|
| Unscreened SPA − screened sNMF | −19.5498 [−23.1394, −16.0304] | −19.5 [−23.1, −16.0] |
| Unscreened SPA − screened PCA | −16.6867 [−20.3863, −13.0831] | −16.6 [−20.3, −13.0] |
| Unscreened SPA − screening-only | −17.7531 [−21.2946, −14.2416] | −17.7 [−21.2, −14.2] |
| Unscreened SPA − raw KNN | −0.1621 [−3.3553, 3.0352] | −0.3 [−3.5, 2.9] |
| Standard CD − screened sNMF | −8.4194 [−11.2966, −5.5282] | −8.4 [−11.3, −5.5] |
| Standard CD − screened PCA | −5.5562 [−8.2394, −2.8091] | −5.5 [−8.2, −2.8] |
| Standard CD − screening-only | −6.6226 [−9.4278, −3.8053] | −6.6 [−9.4, −3.8] |
| Standard CD − raw KNN | +10.9684 [7.5629, 14.3418] | +10.8 [7.4, 14.2] |
| Unscreened SPA − standard CD | −11.1305 [−14.9020, −7.4889] | −11.1 [−14.9, −7.5] |

Exact two-sided McNemar tests and discordant-prediction counts are preserved for all 11 saved directions in `results.json`. These tests concern **accuracy**, not balanced accuracy, and their p-values are also unadjusted and descriptive. For screened sNMF versus standard CD, sNMF alone was correct on 156 rows and CD alone on 72 (`p=2.7207e-8`); versus unscreened SPA those counts were 280 and 85 (`p=2.0214e-25`). This supplements, rather than replaces, the paired effect sizes and intervals.

Standard CD was appreciably better than raw KNN and unscreened SPA on this cohort, but remained below the three screened methods. Unscreened SPA was close to raw KNN; its interval for that difference includes zero. The completed comparison supports a study-specific advantage for the selected screened sNMF pipeline over both selected unscreened NMF pipelines. It does not show that SPA universally improves on CD, that screening has a fixed-configuration causal effect, or that increasing unscreened rank beyond 80 could not change the comparison.

## Audit, reproducibility, and interpretation

The pre-evaluation audit (retained locally: `../runs/celeba-unscreened-supplement-20261006/pretest-audit.json`) passed at **05:09:42 UTC** on 6 October, after the settings freeze and before the supplemental evaluation. It independently checked the source/data bindings, complete grids, folds, qualification, selection, refinement, tie and boundary logic: **1,045 candidates, 5,225 candidate folds, and 75 shared representation-fold receipts**, with zero errors. Missing per-pipeline selected-width diagnostics were not treated as observed measurements: all 75 unscreened input widths were verified structurally from the actual 4,096-column data and no-screen pipeline.

Evaluation completed at **05:15:25 UTC**. The post-evaluation audit (retained locally: `../runs/celeba-unscreened-supplement-20261006/posttest-audit.json`) passed at **05:17:06 UTC**, with zero errors. It independently recomputed metrics from **2,000 new predictions**, rechecked **4,000 preserved original predictions**, verified all **11 saved paired contrasts**, and recomputed all **15 distinct pairs** among the six qualified methods in its details. It verified that the original four-method files were unchanged. The auditor did not refit the models or regenerate CV predictions; numerical-equivalence claims come from the source-bound fit/check receipts, while saved prediction metrics and paired arithmetic were independently recomputed.

All six focused synthetic tests passed in both the primary Python 3.12 environment and the pinned Python 3.10 environment. The primary test receipt (retained locally: `../tmp/celeba-unscreened-tests-primary-20261006.json`) and pinned test receipt (retained locally: `../tmp/celeba-unscreened-tests-pinned-20261006.json`) bind the executed commands, test-source hashes, logs, environment versions, and successful exit codes. These are six focused tests per environment, not a claim that a new full-package test suite was run for this supplement. No scientific source or parameter was revised after supplemental test results.

An analytical reference helps interpret low scores without adding a seventh fitted method. Label-independent guessing has expected balanced accuracy **50%**. The training-majority label is class 0 (1,029 of 2,000 training rows), so a constant class-0 rule would obtain exactly **49% accuracy and 50% balanced accuracy** on the existing 490/510 test split. Always choosing the test-majority class 1 would obtain 51% accuracy, but that is a test-informed majority oracle rather than the training-majority reference. These values are arithmetic references; no random classifier is fitted or favorable random seed selected.

The face task itself was selected using a known earlier Smiling result. The original identity-based report explains that selection and the official-test separation; this supplementary comparison inherits that scope and adds known-test exposure at design time. It restores the omitted dissertation baseline with complete saved outcomes, while leaving historical-number reproduction, global rank optimization, and confirmation on a new independent cohort unclaimed.
