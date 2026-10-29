# CelebA Smiling validation on separate identities

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded. The
[paired-statistics example](#reproducing-the-paired-statistics) includes the
saved labels and predictions needed to reproduce the comparisons below;
full study reproduction requires the retained local workspace.

Author: Chang Cheng. Study date: 6 October 2026 (UTC).

The October 6 frozen source trained SPA's classifier on anchor-derived
coordinates (`training_coordinates="fit_transform"`) and projected queries
against the fitted basis with NNLS. Since [version 0.3.0](release-0.3.0.md#spa-training-coordinates-migration),
the default SPA classifier also projects its training observations. The results
below describe the frozen study procedure; they do not measure that later default.

The later [unscreened NMF supplement](celeba-unscreened-supplement.md) adds the
dissertation's standard CD-NMF+KNN baseline and unscreened SPA-NMF+KNN. It brings
the comparison to six methods, using these same saved test inputs and original
four predictions. That addition was designed after these test outcomes were
known and is reported separately; it does not alter this study's frozen results.

The subsequent [screened CD supplement](celeba-screened-cd-supplement.md) adds
screening followed by CD-NMF and KNN as a seventh method. It retains the same
cohort and independently tunes the same screened grids; all six earlier fits
and predictions remain frozen. Its design also follows known test outcomes.

**Complete: both independent audits passed.** On the frozen 1,000-identity official-test cohort, the October 6 study's sNMF+KNN achieved **81.3725% balanced accuracy**, versus **78.5094% for sPCA+KNN**. The paired advantage was **2.8631 percentage points**, with a prespecified bootstrap 95% interval of **[0.6822, 5.0481]**. Accuracy improved by 2.9 points. This supports a positive, study-specific comparison with sPCA. The advantage over screening-only KNN was smaller and its interval crossed zero.

Only the dense-neighbor successor's frozen configurations and test results support that conclusion. The original coarse-neighbor calibration is preserved below as superseded exploratory tuning evidence. KNN tuning was broadened after that calibration and before new official-test access. The successor retained the same training cache, planned test identities, task, rank/cutoff grids, and folds. No parameters, cohort, seed, or task were changed using the test outcomes.

The study asked whether its frozen absolute-screening SPA sNMF followed by KNN improves on absolute-screening PCA followed by KNN for CelebA's binary **Smiling** annotation. Smiling was selected because an earlier, disclosed development comparison favored SPA+KNN by approximately 1.90 balanced-accuracy percentage points. The present task choice therefore used earlier outcomes. The new evaluation tested that promising finding on separate official-test identities; it was not a blind search across datasets.

A nonface shortlist and data preparation preceded the selection of face data for this study. No nonface classifiers were fitted in that search. The prepared nonface records remain separate and provide no supporting performance evidence for this study. Earlier simulation and real-data results, including unfavorable comparisons, remain in their original reports; this study does not replace them.

## Official-test results

The saved results (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/results.json`) contain all four valid evaluations on the same 1,000 planned images from 1,000 distinct identities: 490 non-Smiling labels (class 0) and 510 Smiling labels (class 1). Every planned row was evaluated. All four final fits and predictions qualified, with zero warnings and no fallback. Percentages below are rounded; the JSON and integer confusion counts retain the exact results.

| Method | Balanced accuracy | Accuracy | Class 0 recall | Class 1 recall | Correct / evaluated |
|---|---:|---:|---:|---:|---:|
| sNMF+KNN | 81.3725% | 81.4% | 80.0000% | 82.7451% | 814 / 1,000 |
| sPCA+KNN | 78.5094% | 78.5% | 78.9796% | 78.0392% | 785 / 1,000 |
| Screening-only KNN | 79.5758% | 79.6% | 78.3673% | 80.7843% | 796 / 1,000 |
| Raw KNN | 61.9848% | 62.2% | 51.2245% | 72.7451% | 622 / 1,000 |

Confusion matrices use true labels as rows and predicted labels as columns, ordered `[0, 1]`; equivalently, each row below gives `[[TN, FP], [FN, TP]]`.

| Method | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| sNMF+KNN | 392 | 98 | 88 | 422 |
| sPCA+KNN | 387 | 103 | 112 | 398 |
| Screening-only KNN | 384 | 106 | 98 | 412 |
| Raw KNN | 251 | 239 | 139 | 371 |

The following additional scores are descriptive, on a 0–1 scale. ROC AUC and average precision use Smiling as the positive class and the frozen KNN vote probabilities; they were not used to select a test-time decision threshold.

| Method | Class 0 precision | Class 0 F1 | Class 1 precision | Class 1 F1 | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| sNMF+KNN | 0.8167 | 0.8082 | 0.8115 | 0.8194 | 0.8961 | 0.9017 |
| sPCA+KNN | 0.7756 | 0.7826 | 0.7944 | 0.7873 | 0.8766 | 0.8823 |
| Screening-only KNN | 0.7967 | 0.7901 | 0.7954 | 0.8016 | 0.8733 | 0.8751 |
| Raw KNN | 0.6436 | 0.5705 | 0.6082 | 0.6625 | 0.6744 | 0.6622 |

Paired differences are **sNMF minus the comparator**, in percentage points. The sPCA row is the prespecified primary contrast; the remaining rows are descriptive controls with unadjusted intervals.

| Comparator | Balanced-accuracy difference [95% CI] | Accuracy difference [95% CI] | sNMF-only correct / comparator-only correct | Exact two-sided McNemar p, accuracy |
|---|---:|---:|---:|---:|
| sPCA+KNN, primary | +2.8631 [0.6822, 5.0481] | +2.9 [0.7, 5.1] | 77 / 48 | 0.011959 |
| Screening-only KNN | +1.7967 [−0.4362, 4.1156] | +1.8 [−0.4, 4.1] | 76 / 58 | 0.141662 |
| Raw KNN | +19.3878 [16.0442, 22.7531] | +19.2 [15.9, 22.6] | 263 / 71 | 4.5950 × 10⁻²⁷ |

The primary result meets the frozen rule: both balanced-accuracy and accuracy point differences are positive, and the primary balanced-accuracy interval excludes zero. This is evidence for sNMF over the independently tuned sPCA comparator on this identity-based CelebA Smiling design. It does not establish a reliable incremental advantage over screening-only KNN, whose interval includes zero, or universal superiority across datasets and tasks.

Final training-fit diagnostics confirm the requested configurations were used:

| Method | Absolute cutoff | Retained pixels / 4,096 | Requested rank | K |
|---|---:|---:|---:|---:|
| sNMF+KNN | 8 | 124 | 20 | 11 |
| sPCA+KNN | 8 | 124 | 40 | 10 |
| Screening-only KNN | 6 | 218 | — | 10 |
| Raw KNN | — | 4,096, unscreened | — | 28 |

The screened methods reported zero constant features and zero infinite screening scores. sNMF reported successful fit and test-coordinate convergence. Its public `NMFClassifier` output agreed with the equivalent search pipeline: identical predicted labels and selected-feature masks, with maximum absolute probability difference **0.0**. Selected pixel lists and all diagnostics remain in the per-method evaluation receipts (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/evaluations`). Pixel retention is not evidence of recovering a known ground-truth biological or facial-feature support.

The result figure is available as PNG (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/identity-validation.png`) and SVG (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/identity-validation.svg`), with a figure provenance receipt (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/figure-manifest.json`).

## Reproducing the paired statistics

Run the [reproduction example](../examples/celeba_identity_reproduction/reproduce.py)
from the repository root:

```bash
python examples/celeba_identity_reproduction/reproduce.py
```

The example needs NumPy and SciPy, both package dependencies. It can also run
independently with its three files in one directory. The
[prediction table](../examples/celeba_identity_reproduction/predictions.csv)
contains the true label and four predicted labels for each of the 1,000 rows,
in the original evaluation order. Class 0 means not Smiling and class 1 means
Smiling. Columns `snmf`, `spca`, `screen` and `raw` correspond to the four
methods above. The table omits image and identity identifiers, pixels and
probabilities. Repeated label/prediction rows are expected; they do not indicate
repeated identities and must not be removed.

The script reproduces the confusion matrices, class recalls, accuracy, balanced
accuracy and all three paired comparisons. It preserves the original seed
62107, resets it for each comparison, draws 10,000 bootstrap replicates within
class 0 and then class 1, and uses linear percentile interpolation. Row order
and pairing matter for replaying these particular seeded intervals. Replacing
rows with aggregate counts can preserve the theoretical bootstrap distribution
while changing its Monte Carlo endpoints.

[Expected results](../examples/celeba_identity_reproduction/expected.json)
retain full precision, extraction details, historical source hashes and library
versions. Effects and intervals in the JSON use proportions, so multiply by
100 for the percentage points reported here. The script checks the fixture's
checksum of the ordered label table, serialized with LF line endings, and prints the recalculated
statistics with an `exact_historical_match` flag. A mismatch returns a nonzero
exit status while retaining the calculated output; library roundoff or sampling
changes may cause a mismatch. This is a historical replay check, not a package
compatibility test. No new constraints are placed on package dependencies.

This example uses fixed historical predictions. It does not reproduce fitting,
calibration, probability-based scores, or verification of identity separation.
Its intervals remain conditional on the fitted models and the study design;
replay does not establish population coverage. The screening-only and raw
comparisons remain descriptive with unadjusted intervals, and McNemar applies
to accuracy. The repeated-subject CMU data require their separate grouped
procedure.

## Cohort and provenance

The authoritative preparation is the immutable cache manifest (retained locally: `../data/celeba-identity-validation-20261006/manifest.json`), with the complete sampling rule and image/identity lists in the data plan (retained locally: `../data/celeba-identity-validation-20261006/plan.json`). Identity annotations are opaque integer groups, not inferred personal names. The identity-source receipt (retained locally: `../data/celeba-metadata/identity-source.json`) records an official download, the published Torchvision MD5 match, and SHA256 `c6143857c3e2630ac2da9f782e9c1232e5e59be993a9d44e8a7916c78a6158c0`.

| Official partition | Images | Distinct annotated identities |
|---|---:|---:|
| Training | 162,770 | 8,192 |
| Validation | 19,867 | 985 |
| Test | 19,962 | 1,000 |

The full original partitions are pairwise disjoint by annotated identity. Preparation recursively scanned only existing `data/**/*manifest.json` records, excluding the new cache, and saved their paths and hashes. The resulting exclusion set contained 30,358 previously cached image IDs, including 26,358 official-training IDs and 4,000 official-validation IDs. After those image exclusions, 136,412 training images from 8,176 identities were eligible.

Using NumPy `RandomState(62012)`, preparation sampled 2,000 eligible training identities uniformly without replacement, then selected one eligible image uniformly within each selected identity. The resulting training cohort has 2,000 images from 2,000 identities: 1,029 labels in class 0 and 971 in class 1. Sampling did not balance classes or inspect labels. All selected image IDs are fresh relative to the scanned caches. Some selected identities may have appeared in earlier training caches; excluding earlier **images** does not mean all training **people** are new.

Using a separate `RandomState(62013)`, the test plan selected one official-test image uniformly within each of all 1,000 official-test identities. The evaluated test cohort contains exactly those 1,000 images from 1,000 identities and shares no identity or image ID with the new training cohort. It is not a random 1,000-image sample from the 19,962-image test partition: each identity contributes equally. Test class counts were unavailable during preparation and calibration; all planned rows were used without outcome-based replacement.

Training preparation decoded only official-training pixels and extracted Smiling labels only for the selected training IDs. The test plan contains filenames and opaque identity numbers only. The training cache includes raw-JPEG, resized grayscale-pixel, and model-input row hashes; no exact duplicates were detected within the selected training cohort. At the gated test evaluation, the driver checked test-to-training and within-test duplicate preprocessed images and found none. The frozen rule would stop on a duplicate and retain the access record rather than silently replace a row. These checks address exact duplicates; they do not prove the absence of near duplicates or annotation errors.

There is one historical qualification to test provenance: an earlier abandoned final-test attempt began loading test data, with no persisted test cache, prediction attempts, evaluations, or comparison recorded. Partial in-memory image decoding in that attempt is unknown. The [historical chronology](validation.md) and current protocol disclose this. The present study does not claim that no official-test pixel was ever decoded; no prior test outcomes were recorded or used to choose this protocol.

## Frozen comparisons and calibration

The current dense-neighbor study protocol (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/protocol.json`) binds the driver, tuning helper, preparation script, package sources, helper tests, metadata plan, and training arrays by checksum. It uses the separate immutable run directory `runs/celeba-identity-validation-dense-knn-20261006`; the original coarse protocol (retained locally: `../runs/celeba-identity-validation-20261006/protocol.json`) and its receipts remain unchanged. Package defaults and model code remain unchanged for this study. The training cache preserves the existing loader expression: Pillow grayscale `L`, resize to 64 by 64 with the unchanged default interpolation, flatten, and convert to `float32 / 255`. There are 4,096 nonnegative input features. No learned per-pixel rescaling or centering precedes screening; PCA performs its usual centering inside each training fold.

| Method | Training pipeline | Tuned quantities |
|---|---|---|
| sNMF+KNN | Absolute t screening, FeatureSPA, KNN | Cutoff, rank, neighbors |
| sPCA+KNN | The same screening rule, PCA, KNN | Cutoff, rank, neighbors |
| Screening-only KNN | The same screening rule, KNN | Cutoff, neighbors |
| Raw KNN | Original grayscale pixel inputs, KNN | Neighbors |

All KNN models use uniform voting and the fixed default decision rule. The initial rank grid is `{2, 5, 10, 20, 40}` and the initial absolute t-cutoff grid is `{0, 2, 4, 8}`. The current initial neighbor grid contains **every integer from 1 through 101, plus 125, 151, and 201: 104 values**, identically available to all four methods. This gives 2,080 initial candidates for each of sNMF and sPCA, 416 for screening-only, and 104 for raw KNN, before any local refinement. The SPA NNLS iteration budget is 20,000. These are finite, prespecified initial grids, not claims of fully optimized methods.

Each method is tuned independently on the same five stratified folds, with split seed 62106 and training balanced accuracy as the criterion. Every learned operation, including screening and the representation, is fitted within its training fold. Because every training identity appears once, ordinary stratified folds cannot split repeated measurements of the same annotated identity. Four worker processes each use one numerical thread.

After each method's initial grid, one deterministic local refinement proposes changes to one coordinate at a time around that method's new initial winner. It uses the same folds, arithmetic intermediate values, integer rounding for rank and neighbors, and explicit physical bounds. At upper boundaries, this rule can propose rank 80 or cutoff 16; if a method's initial winner selects the largest K of 201, the rule proposes an upper expansion to K 402, subject to the training-size bound. The actual new refinement candidates depend on the dense search's winners. They must not be copied from the superseded coarse run. Many integer-K neighbors within 1–101 are already covered and therefore add no distinct candidate. The rule does not repeatedly expand until a preferred result appears or jointly explore every combination of refined coordinates. Final selection pools the initial and refined candidates and uses highest complete mean balanced accuracy, then canonical parameter order for ties. Boundary qualifications and omitted physically impossible proposals remain recorded.

A candidate must qualify on all five folds. Failed or nonconverged folds are retained and make the candidate ineligible; their successful folds are not averaged as if the candidate completed. The search returns no substitute method if none qualify. All four methods must have qualified settings before the final configuration freeze. The public `NMFClassifier` supplies the final sNMF result, with a separately recorded comparison against the equivalent search pipeline. A discrepancy invalidates that evaluation instead of replacing its predictions.

The new design uses training CV followed by one independent test. It does **not** run the previously proposed nested cross-validation search to select among datasets: the face task was already selected from earlier evidence. Training CV scores are used for tuning and are not reported as independent validation results. A single local refinement does not establish global optimality or fulfill a claim that all potentially useful hyperparameters have been exhausted.

## Current dense-neighbor calibration

The dense final settings freeze (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/final-freeze.json`) was written at **04:25:35 UTC**, before new test pixels or outcomes were accessed. All four methods have qualified selected settings. The scores below are **selected five-fold training-CV scores**, optimistic as performance estimates because the same folds selected the configurations. They are not independent validation or heldout-test results.

| Method | Selected rank | Selected absolute cutoff | Selected neighbors | Selected CV balanced accuracy |
|---|---:|---:|---:|---:|
| sNMF+KNN | 20 | 8 | 11 | 82.0025% |
| sPCA+KNN | 40 | 8 | 10 | 78.9607% |
| Screening-only KNN | — | 6 | 10 | 78.9542% |
| Raw KNN | — | — | 28 | 61.2364% |

| Method | Initial candidates | Added refinement candidates | Qualified / total candidates | Valid / total candidate-fold evaluations |
|---|---:|---:|---:|---:|
| sNMF+KNN | 2,080 | 4 | 2,083 / 2,084 | 10,415 / 10,420 |
| sPCA+KNN | 2,080 | 4 | 2,083 / 2,084 | 10,415 / 10,420 |
| Screening-only KNN | 416 | 2 | 418 / 418 | 2,090 / 2,090 |
| Raw KNN | 104 | 0 | 104 / 104 | 520 / 520 |
| Total | 4,680 | 10 | 4,688 / 4,690 | 23,440 / 23,450 |

The actual refinement proposals changed one coordinate of each new initial winner. For sNMF, `(rank 20, cutoff 8, K 11)` proposed rank 15 or 30 or cutoff 6 or 16. For sPCA, `(rank 40, cutoff 8, K 10)` proposed rank 30 or 80 or cutoff 6 or 16. Screening-only's `(cutoff 8, K 10)` proposed cutoff 6 or 16. There were no new K proposals outside the initial dense grid for these winners, and raw KNN added no distinct refinement candidate. No proposal was omitted for exceeding a physical bound. These were separate one-coordinate candidates, not a Cartesian product of refined values.

Two refinement candidates failed all five folds and were retained as ineligible: sNMF `(rank 20, cutoff 16, K 11)` and sPCA `(rank 40, cutoff 16, K 10)`. At cutoff 16, screening retained 12, 13, or 14 pixels across folds, making the requested ranks infeasible. These ten failures account for the complete dense-search candidate-fold failure count. All initial-grid candidates qualified, and the initial/refinement CV reports contain zero warning records. No rank was silently reduced and no fallback method was used. Candidate-fold counts count each neighbor candidate's score; the search implementation shares representation fitting across K values, so these are not counts of distinct decomposition fits.

All selected configurations have empty unresolved-coordinate-edge lists in the receipts. In particular, sPCA's selected rank 40 was compared with the proposed rank 80; the selected K values 11, 10, 10, and 28 lie within the explored neighbor range. Empty edge lists do not establish global optimality or cover untested combinations of refined ranks, cutoffs, and neighbors. The independent pretest audit confirmed complete fold receipts, warnings, matching folds, and exact selection bindings before test access.

## Superseded coarse-neighbor tuning evidence

The original run used initial neighbors `{3, 9, 25}` and finished with its coarse settings freeze (retained locally: `../runs/celeba-identity-validation-20261006/final-freeze.json`) at 04:18:40 UTC. That configuration freeze is **superseded and was not used for test evaluation**. The following historical values are **selected five-fold training-CV scores**, which are optimistic as performance estimates because the same folds selected the settings. They are not the current dense-neighbor selections, heldout-test results, or evidence that the new validation succeeded.

| Method | Selected rank | Selected absolute cutoff | Selected neighbors | Selected CV balanced accuracy | Qualified candidates / evaluated |
|---|---:|---:|---:|---:|---:|
| sNMF+KNN | 20 | 8 | 17 | 81.3983% | 65 / 66 |
| sPCA+KNN | 20 | 6 | 9 | 78.7706% | 65 / 66 |
| Screening-only KNN | — | 8 | 6 | 78.6847% | 16 / 16 |
| Raw KNN | — | — | 17 | 59.7815% | 5 / 5 |

The initial candidate counts were 60, 60, 12, and 3 respectively. Refinement added 6, 6, 4, and 2 candidates. For sNMF it changed one coordinate of the initial winner `(rank 20, cutoff 8, K 25)` to rank 15 or 30, cutoff 6 or 16, or K 17 or 50. For sPCA it changed one coordinate of `(rank 20, cutoff 8, K 9)` to rank 15 or 30, cutoff 6 or 16, or K 6 or 17. Screening-only refined `(cutoff 8, K 9)` using cutoffs 6 or 16 or K 6 or 17. Raw KNN refined its initial K 25 to 17 or 50. These added settings form separate one-coordinate candidates, not a full Cartesian grid.

Across the four superseded coarse searches, 151 of 153 candidates qualified. There were 765 candidate-fold evaluations, including ten feasibility failures: the rank-20/cutoff-16 candidate failed all five folds for each of sNMF and sPCA because screening retained only 12–14 pixels. Both candidates were excluded without lowering their requested ranks or substituting another method. No selected coarse setting was flagged at the edge of its explored coordinate range; this diagnostic did not establish adequate neighbor exploration, motivating the broader successor search. These counts remain historical and must not be combined with or substituted for the successor's tuning counts.

## Test gate and uncertainty

Before new test access, the dense successor's driver required its own complete `final-freeze.json` and an independent passed pretest audit bound to that exact file. The original coarse freeze could not substitute for these receipts. The current final freeze binds every method's selected-configuration receipt and the current study protocol. The driver then recorded a test-access receipt before loading planned test pixels and labels. It refuses an automatic rerun when a prior test-access attempt exists and refuses test-based retuning, changes to rows or seeds, or selection of a different task after the result. No such test-dependent changes were made.

The primary contrast is **sNMF+KNN minus sPCA+KNN balanced accuracy**. Accuracy and both class recalls are also relevant to interpretation. Screening-only and raw KNN remain visible regardless of their rankings: an advantage over sPCA alone does not establish an advantage over simpler alternatives.

The frozen uncertainty procedure uses 10,000 paired, class-stratified bootstrap replicates of the unique-identity test rows, with seed 62107 and percentile 95% intervals. Each resampled row retains both methods' predictions together. The accuracy interval uses the observed test class proportions; the balanced-accuracy interval weights the two class recalls equally. An exact two-sided McNemar test of the discordant predictions is reported for **accuracy only**, not relabeled as a test of balanced accuracy. Additional comparator intervals are descriptive and unadjusted for multiple comparisons.

These intervals condition on the chosen training cohort, selected settings, and fitted models. They do not measure variability from retraining on another cohort, rerunning calibration, selecting a different task, or choosing a different image of each person. Interpreting them as generalization intervals also treats the observed identities as informative about a wider target population; the study itself uses all 1,000 official-test identities with one sampled image each. Identity separation limits leakage and repeated-person pseudoreplication, but does not make CelebA representative of all people, cameras, facial expressions, or deployment settings.

The success rule was frozen before new test access: positive balanced-accuracy and accuracy differences constitute an observed advantage; a balanced-accuracy 95% lower endpoint above zero provides stronger evidence for this dataset and cohort design. A zero-crossing interval remains inconclusive, and an unfavorable estimate remains unfavorable. Neither outcome permits another seed, another image selection, another task, or further tuning against these test outcomes.

## Independent audits and chronology

Both audits passed with **zero errors**. The pretest audit (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/pretest-audit.json`) verified 4,690 candidates, 23,450 candidate-fold evaluations, 275 shared representation-fold receipts, and 260 independently reconstructed screening masks. It redecoded all 2,000 training images, checked the frozen metadata/exclusions and source bindings, and accessed zero test pixels or test labels. Its exact final-freeze hash is recorded in the subsequent access receipt.

The posttest audit (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/posttest-audit.json`) redecoded all 1,000 test images and independently checked all four saved prediction vectors, 4,000 individual predictions, final screening masks, reported metrics, and all three paired bootstrap contrasts. The preserved test-data receipt (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/test-data.json`), test-access receipt (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/test-access.json`), and results (retained locally: `../runs/celeba-identity-validation-dense-knn-20261006/results.json`) bind the single test evaluation to the frozen configurations. The audit validates computational integrity and reporting; it does not remove the scientific limitations described above.

| Event | 6 October 2026, UTC |
|---|---|
| Dense protocol frozen | 04:22:05 |
| Dense final configurations frozen | 04:25:35 |
| Independent pretest audit passed | 04:29:20 |
| New test access recorded | 04:29:50 |
| Four-method test results completed | 04:29:55 |
| Independent posttest audit passed | 04:30:10 |

This completed validation provides a documented real-data case where the frozen October 6 sNMF procedure outperformed the independently tuned sPCA pipeline under the declared finite tuning procedure. Its scope is the selected Smiling task, the preserved image preprocessing, one image per heldout identity, and these fitted models. It does not certify all package settings, general superiority, the formal screening/separability assumptions, or that every concern in Wang's theoretical review has been resolved.
