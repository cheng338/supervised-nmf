# Validation status and historical checks

Author: Chang Cheng.

Historical study results and subsequent implementation checks are distinguished
below. The projected-coordinate evidence summary was added on October 28, 2026
from existing records; no study was rerun for that summary.

The October 3–6 SPA classification studies below used anchor-derived training
coordinates (`training_coordinates="fit_transform"`) and fixed-basis NNLS for
queries. Their accuracy and timing belong to those frozen study sources.
The default SPA classifier has projected training observations as well since
[v0.3.0](release-0.3.0.md#spa-training-coordinates-migration). Later implementation
checks and projected-coordinate studies are recorded separately and do
not reattribute earlier study results to that default.

October 3–6 environment: Python 3.12.14, NumPy 2.5.2,
SciPy 1.18.0, scikit-learn 1.9.0, threadpoolctl 3.5.0, AOCL-BLAS 5.2.0,
Windows. Comparison execution explicitly limited numerical libraries to one
thread. These results do not establish compatibility with every allowed version.

## Evidence for projected SPA training coordinates

Completed studies evaluate the same coordinate-policy concept as the current
SPA classifier: fit a basis on training data, then use fixed-basis NNLS for both
training and query observations. They support this procedure as a usable option,
with mixed classification outcomes. They do not establish that SPA is generally
the best solver, or that the untuned current constructor reproduces their scores.
Rank, screening cutoff and classifier settings still require training-only tuning.

### October 12 official-partition evaluation

This study completed all **300 planned and audited final outcomes**: ten methods,
three CelebA attributes and ten 20,000-image training draws per attribute. Each
draw used the same **19,962 official-test images**. Configuration selection used
grouped five-fold CV on a separate 19,838-image development reserve. Identity and
exact-image-duplicate exclusions separated the pools. Natural class proportions
were preserved; no class-rebalancing or decision-threshold tuning was used.

The frozen source was version **0.2.0 plus a study adapter** that projected SPA
training observations. It was not the later public estimator implementation.
The screening-only, sPCA and sNMF comparisons shared a training-fitted absolute
t-score screen with a 35% retained-feature ceiling. Its cutoff maximized the equal
mean of those three KNN procedures' best CV selection scores. Selection used
accuracy for Smiling and Wearing_Lipstick, and balanced accuracy for
Wearing_Necktie. KNN scaling was tuned over unscaled and training-fitted
standard-deviation scaling; it was not fixed to the current unscaled default.

The following are saved mean **balanced accuracies (%)** across the ten training
draws. Unscreened NMF in this study also used SPA. Logistic regression denotes
the separately tuned classifier extension.

| Procedure | Smiling | Wearing_Necktie | Wearing_Lipstick |
| --- | ---: | ---: | ---: |
| Raw features + KNN | 71.197 | 58.127 | 77.406 |
| Screening + KNN | 83.743 | 61.565 | 83.465 |
| sPCA + KNN | 85.713 | 62.158 | 86.091 |
| Unscreened NMF + KNN | 72.631 | 60.683 | 74.843 |
| sNMF + KNN | 85.400 | 62.131 | 85.931 |
| Raw features + logistic regression | 89.442 | 82.021 | 88.808 |
| Screening + logistic regression | 86.018 | 81.898 | 86.862 |
| sPCA + logistic regression | 85.825 | 79.977 | 86.603 |
| Unscreened NMF + logistic regression | 78.968 | 74.818 | 77.564 |
| sNMF + logistic regression | 85.757 | 80.651 | 86.346 |

The selected sNMF KNN configurations were:

| Attribute | Absolute cutoff | Rank | Neighbors | Coordinate scaling |
| --- | ---: | ---: | ---: | --- |
| Smiling | 32 | 20 | 21 | Standard deviation |
| Wearing_Necktie | 8 | 30 | 1 | None |
| Wearing_Lipstick | 32 | 20 | 12 | Standard deviation |

sPCA KNN had slightly higher mean balanced accuracy and ordinary accuracy than
sNMF KNN on all three attributes; these small descriptive differences do not
establish statistical superiority or equivalence. Raw-feature logistic regression
had the highest mean balanced accuracy among the ten procedures on all three
attributes. For Wearing_Necktie, sNMF KNN's **91.117% ordinary accuracy** accompanied
only **28.420% positive-class recall**. The imbalance problem is not resolved by
the high ordinary accuracy.

Selection was frozen before evaluation access within this run, but CelebA had
already been explored in earlier work. The protocol explicitly describes the
study as a descriptive reassessment, not fresh independent confirmation. Training
draws can overlap and share one development selection and one test cohort.
Variation across draws is conditional on those fixed data and settings; it does
not measure uncertainty from new test subjects or repeated tuning. The finite
search also does not establish globally optimal configurations.

### October 20 public-estimator comparison

A separate study used the frozen October 20 integration that preceded v0.3.0.
Its common-coordinate arm exercised the public SPA estimator with absolute
screening and unscaled KNN, more directly matching the current method choices.
It used **5,002 previously exposed development images**, 253 identity/duplicate
groups and three overlapping grouped outer splits, with five-fold inner tuning.
It did not access the official test partition. The saved mean **balanced
accuracies (%)** for its unscaled-KNN arm were:

| Procedure | Smiling | Wearing_Necktie | Wearing_Lipstick |
| --- | ---: | ---: | ---: |
| Raw features + KNN | 62.56 | 59.48 | 70.08 |
| Screening + KNN | 80.45 | 57.75 | 74.45 |
| sPCA + KNN | 81.63 | 59.68 | 74.39 |
| sNMF + KNN | 82.31 | 58.44 | 76.48 |

sNMF led these KNN comparators on Smiling and Wearing_Lipstick, but trailed raw
KNN and sPCA on Wearing_Necktie. Its positive-class recall on Wearing_Necktie
was **21.42%**. The three splits and three attributes are not independent datasets.
These scores cannot be compared directly with October 12 as a package improvement
or regression: the cohorts, source, tuning and scaling conditions differ.

Across all declared source arms and classifier choices, **135 final outcomes
qualified**. This does not mean every attempted configuration succeeded: the
initial search retained **198 failed representations affecting 1,112 fold-candidate
evaluations**, alongside 14,818 successful initial and 1,155 successful refinement
fold-candidate evaluations. The bounded rank grid was `{2, 10, 30}`, with cutoffs
`{0, 2, 8}` and one classifier-only refinement. All selected sNMF ranks reached 30; higher
feasible ranks remain unresolved. Completion of the bounded validation is not
completion of tuning. Scientific fit/predict times came from concurrent
scheduling and should not be read as isolated speed comparisons.

### Relationship to the present implementation

Subsequent engineering comparisons cover changes after those studies. The
October 26 comparison against the released v0.3.2 wheel recorded **1,176 qualified
saved-array source pairs** (888 measured and 288 warmup pairs) across the covered
methods and configurations. Five additional pairs from an incomplete original
attempt remain recorded separately. These are repeated execution comparisons,
not independent datasets. Predicted labels, screening masks, SPA anchors and class
ordering were unchanged on the qualified matched cases. It included ordinary
inputs and numerical stress cases in four environments. Its CelebA workload used
previously exposed Smiling data, with 1,600 training and 400 query observations
and fixed settings.

The frozen candidate's SPA basis-fitting, NNLS, projection, residual and screening
modules match v0.3.3 by SHA-256. The October 28 source retains those numerical
operations; later docstring clarifications in `decomposition.py` change its file
hash without changing executable statements. The surrounding classifier has
subsequent changes, including mixed-dtype handling and the optional MU projection
policy. These engineering checks support the numerical
implementation on the tested cases; they do not rerun the full October 12 or
October 20 scientific comparisons or establish new classification superiority.

Reproducing an entire historical table with current source would require freezing
that source and replaying the saved data splits and selected settings under a
separate protocol. Such a replay would measure reproducibility on already exposed
data. Independent statistical confirmation would additionally
require an appropriately reserved evaluation cohort. Neither is claimed here,
and no default is changed by this record review.

### Record provenance

The summary above transcribes saved aggregate records, without recomputing
predictions or altering the frozen studies. The archived records are separate
from the package distributions:

- October 12: `celeba-official-20261012/final-evidence/`, including `protocol.json`,
  `source-context.json`, `source-binding.json`, `report/summary.json`,
  `report/rows.json`, `report/results.json` and `prediction-audit.json`.
  The aggregate `report/summary.json` has SHA-256
  `1efdda358ec83c1ecee6f1456279ccf98b1e7010cafa06dabc1d6afe577b0a5f`.
- October 20: `latest-vs-first-20261020/`, including `PROTOCOL.md`,
  `frozen-sources-v1-manifest.json`, `final-report-v2/validation-report.txt`,
  `celeba-completeness-notes.md` and the final `celeba-summary.md`.
  The [v0.3.0 notes](release-0.3.0.md) also describe the implementation changes
  and validation limits.
- October 26: `final-release-comparison-20261026/REPORT.txt` and
  `source-manifest.json`; the latter has SHA-256
  `88bd04b5cbcda9f328ca1178449514377bd802e6e74b307c42732386703d8a2f`.

## October 6 implementation checks

- The [five-dataset expansion](multi-dataset-validation.md) adds **24 focused
  checks** in each numerical environment: six for the seven-method helper,
  nine for the grouped-split runner, six for source-image preparation, and
  three for independent receipt/metric auditing. All pass. The independent
  face-data audit reconstructs 1,024 images and 1,162,240 feature values exactly,
  with zero discrepancies. All 35 final evaluations qualify. The pretest audit
  checks 38,854 candidates and 2,270 representation-fold records; the posttest
  audit reproduces metrics from 12,642 predictions and all 30 paired contrasts,
  with zero errors. [Complete tables](multi-dataset-results.md) retain the mixed
  outcomes. These checks do not change the frozen core implementation or
  constitute repeated nested-CV replication.
- The screened-CD supplement passed **six focused tests** in both numerical
  environments (4.321 and 4.282 seconds). They check nontrivial fold-local
  screening, public/reference feature masks and predictions, training-only
  fitting, held-out-label invariance, the original refinement rule, failure
  retention and frozen-access integrity. Exact logs/source/runtime receipts are
  `tmp/celeba-screened-cd-tests-primary-20261006.*` and
  `tmp/celeba-screened-cd-tests-pinned-20261006.*`. Both study audits passed;
  these study-helper checks did not modify or rerun the frozen core test suite.
- The unscreened-NMF supplement passed **six dedicated tests** in both numerical
  environments (5.238 and 5.124 seconds), checking both public-estimator/pipeline
  equivalence paths, training-only fits, failure qualification, frozen receipt
  integrity, contrast direction and repeated-access refusal. Logs and exact
  source/version receipts are `tmp/celeba-unscreened-tests-primary-20261006.*`
  and `tmp/celeba-unscreened-tests-pinned-20261006.*`. The frozen core package is
  unchanged; these are additional study-helper checks, not a claimed full-suite
  rerun. Both independent supplemental audits passed with zero errors.
- The later identity-validation helper passed all **10 dedicated tests** in both
  numerical environments (5.417 and 6.124 seconds). These cover fold qualification,
  frozen-input integrity, scaling choices and equivalence with public
  `NMFClassifier`, including the unscaled 4,096-feature float32 face path. The
  core package sources still match the 303-test revision below. The new study's
  independent pretest and posttest audits pass with zero errors; no full-suite
  rerun is claimed for these study-only additions.
- The final Wang-study revision passed **303 tests** in the primary environment
  (121.042 seconds) and the exact Python 3.10.12 numerical profile (65.999
  seconds), with no skips. The first compatibility attempt passed 302 tests
  and encountered one missing optional Matplotlib dependency in a new report
  test. After installing Matplotlib 3.10.9 under the unchanged numerical
  constraints, the complete suite passed; `pip check` was clean. Logs are
  `tmp/wang-final-check-aocl.log`, `tmp/wang-final-check-robust.log` (the retained
  initial failure), and `tmp/wang-final-check-robust-with-study.log`.
  `runs/wang-validation-audit-20261006/implementation-checks.json` binds the
  logs, package sources and tests by hash. Observed times reflect concurrent
  host work and do not compare environment speeds.
- The final local wheel and source archive were rebuilt with the BLAS change
  and completed study documentation. Isolated wheel checks passed in both
  environments; packaged implementation, study sources and reports match the
  files at that revision byte for byte. The later identity-validation study and
  its documentation are not included in those archived builds. Prior archives
  are preserved separately.
  `runs/wang-validation-audit-20261006/local-build.json` records archive hashes
  and checks; build and wheel logs use the `tmp/wang-final-` prefix. These are
  local artifacts, with no package publication.
- At the initial [SPA BLAS optimization](blas-optimization.md) revision, all **300 tests**
  passed in the primary Python 3.12/AOCL environment (61.257 seconds) and the exact Python 3.10.12
  compatibility environment (58.437 seconds). The seven dedicated numerical
  regression tests cover the former NumPy/NNLS algorithm, layouts, ties,
  ill-conditioning, mutation, iteration failures and both final classifiers.
  Seven paired timing repeats on each of four training-only workloads produced
  identical anchors, factors, coordinates and classifier outputs, with
  12.6–16.6% lower fit-plus-projection time. These checks do not complete the
  separate Wang scientific validation. Logs are `tmp/blas-full-check-aocl.log`
  and `tmp/blas-full-check-robust.log`.
- Version **0.1.0a1** adds optional logistic regression while retaining
  SPA, absolute screening and KNN as defaults. The complete suite passed
  **281 tests in 59.991 seconds** in the primary Python 3.12/AOCL environment and **281 tests in
  57.313 seconds** under Windows 11 with Python 3.10.12, NumPy 2.2.6,
  SciPy 1.15.3, scikit-learn 1.7.0, joblib 1.5.1 and threadpoolctl 3.6.0.
  Pillow 12.3.0 enabled the image-loader fixtures in the latter environment.
  These checks support the updated Python >=3.10 requirement. Ubuntu execution
  remains untested; see [runtime compatibility](compatibility.md).
- Logistic checks cover explicit sklearn reference equivalence for SPA/CD/MU,
  both training-coordinate conventions, fold-local screening and scaling,
  fixed-basis prediction, string labels, exact decision ties, cloning,
  serialization, classifier switching, numerical warnings and rejection of
  nonconverged candidates. Generated raw-feature logistic baselines retain fit
  convergence evidence for later evaluation. The KNN representation-sharing
  helper rejects logistic configurations and directs them to the general search.
- The paired classifier example completed all **25 methods** on the exact
  Python 3.10 profile with no fallbacks or nonconverged evaluations: 125 training
  folds followed by final refits and development-validation predictions. It used
  one candidate per method and one numerical thread. This is an implementation
  check, not evidence for changing the classifier default or resolving the
  requested CelebA comparisons. Logs and generated reports remain local.
- The 0.1.0a1 wheel and source archive built successfully. Isolated wheel-import
  checks passed in both environments. The wheel also installed with `--no-deps`
  into the exact Python 3.10.12 profile, preserving all five numerical pins.
  All six SPA/CD/MU by KNN/logistic combinations passed installed-package
  fit/predict checks; `pip check` reported no broken requirements. Archive
  checks verified the author metadata, Python requirement, original-source
  hashes, compatibility pins, documentation links and exclusion of local data.
- After the public MU addition, `python scripts/check.py` passed **269 tests in
  58.743 seconds**. The log is `tmp/mu-public-full-check.log`. The wheel and source
  archive built successfully. An isolated wheel check verified SPA, CD and MU
  fit/predict, constructor defaults, explicit overrides, author metadata and
  archive contents; see `tmp/mu-public-wheel-check.log` and
  `tmp/mu-public-build.log`.
- A preceding full `python scripts/check.py` run passed **179 tests in 53.660
  seconds**. The log is `tmp/four-solver-full-check-final.log`. A subsequent
  matched-baseline reporting test brought the permanent test count to **180**;
  all **17 reporter tests passed in 0.623 seconds** after that addition. The
  180-test suite was not rerun at that point; later full-suite runs are recorded
  separately.
- The preceding full `python scripts/check.py` run passed **128 tests in 28.104
  seconds**, after the absolute-screening amendment and its 10 new pure decision
  tests. This record verifies implementation behavior at that revision.
- Historical `python scripts/check.py`: **65 tests passed** after withdrawal of the extra
  adaptive workflow and addition of explicit shared-representation grids.
  Coverage includes original
  float32 CD fidelity, pooled-t/F agreement, percentile ties, inclusive thresholds,
  exact/near-degenerate features, fold-local fitting, fixed-basis prediction,
  feature-anchor SPA on separable fixtures, serialization, warning/error records,
  label mapping, imbalanced metrics, official image partitions, and resume integrity.
  The 13 shared-grid tests also compare actual per-fold prediction arrays and
  numerical/support diagnostics against ordinary calibration, including tied
  distances; they test projection warnings and interrupted-fold reuse.
- Earlier `python examples/quickstart.py` check: executed successfully with explicit historical-
  style settings and a converged CD fit on a small generated dataset.
- Earlier `python examples/compare_simulation.py --quick --output runs --threads 1`:
  all **11 comparison methods** completed five-fold calibration and development
  validation without a fallback in this smoke run. Reports are under
  `runs/simulation-smoke-20261003T052810Z-47a76e20/` (ignored by Git).
- Original snapshots match both recorded SHA-256 hashes in `provenance/sources.json`.
- Local wheel and source archives build successfully. An isolated Python process
  imports directly from the wheel, verifies Chang Cheng's author metadata, and
  executes fit/predict without importing the checkout or installing into conda.
  Source-archive checks verify inclusion of original scripts, tests and examples.
  The current archives include the explicit-grid study module and exclude the
  withdrawn workflow modules, documentation and reference provenance.
- The CelebA CLI and loader pass synthetic-fixture tests and the real-data smoke
  checks described below.

The current suite and distribution checks apply to the explicit-grid
implementation. The additional adaptive-search workflow's historical test totals
are not part of the current count. Earlier scientific smoke runs below remain
separate records and are not substantive tuning results.

The paired smoke uses a single dataset, one candidate per method, and a small
development split. It tests functionality and audit records. Its accuracy and
timings cannot determine defaults, establish scientific superiority, reproduce
the historical experiment, or close Wang's empirical concerns. In particular,
the historical smoke runs did not complete the repeated simulation study or
numerical tuning/refinement. The later completed study is recorded separately
below; its finite-grid and local-refinement limitations remain explicit.

## Completed Wang validation after the BLAS change

The [integrated report](wang-validation-report.md) records 900 independently
retuned simulation draws (9,930 valid evaluations), ten fully retuned CelebA
resamples (260 valid evaluations) and twenty fixed-setting resamples (520 valid
evaluations). The combined independent audit passed with zero discrepancies:
113,156 CV candidates, 565,780 candidate-fold records and 7,886,400 saved predictions were
checked, including all matched masks and representative-case bootstrap intervals.
All final fits qualified; infeasible CV candidates were retained and rejected.
The 11-arm historical diagnostic and 21-receipt Fashion reanalysis remain
separate supporting records. That study accessed no official image-test outcomes.

SPA improves the KNN procedure on CelebA Smiling but trails logistic PCA/pixel
baselines. Simulations and Fashion-MNIST do not support a general NMF advantage.
The original-size historical diagnostic identifies an insufficient iteration
cap as a sufficient numerical cause in the reconstruction, not an exact
historical-environment reproduction. Finite tuning grids, reused image
validation, unverified identity separation, reduced simulation dimensions,
whole-study memory and theoretical assumptions remain qualified in the report.

## Completed identity-disjoint face validation

The [CelebA Smiling report](celeba-identity-validation.md) records a separate
October 6 validation chosen from the earlier positive development finding.
Official identity metadata, verified against the published checksum, confirms
that all original CelebA partitions are pairwise disjoint by annotated identity.
The new study uses 2,000 training identities and one preselected image from each
of the 1,000 official-test identities. Training image IDs are fresh relative to
prior caches; training identities may recur. The historical uncertainty about
partial in-memory test decoding in an abandoned attempt remains disclosed.

Broader KNN tuning superseded the first coarse calibration
before any new test access. The dense successor offers every integer K from 1 to
101, plus 125, 151 and 201, to all four methods. Five-fold training-only tuning
and one local refinement produced 4,690 candidates and 23,450 candidate-fold
records. Two infeasible candidates account for all ten invalid fold records;
their failures are retained. The independent pretest audit passed before the
frozen configurations were evaluated on all planned test rows.

The historical SPA `fit_transform` procedure obtained **81.40% accuracy and
81.37% balanced accuracy**; sPCA obtained **78.50% and 78.51%**, respectively.
The primary paired balanced-
accuracy difference is **+2.86 percentage points**, with a 95% class-stratified
bootstrap interval **[+0.68, +5.05]**. Screening-only KNN obtained 79.60% accuracy
and 79.58% balanced accuracy; the sNMF contrast with that control remains
inconclusive. Raw-pixel KNN obtained 62.20% and 61.98%. All four final evaluations
qualified without warnings. The public sNMF estimator and reference pipeline
had identical labels, screening masks and probabilities.

The posttest audit independently reconstructed all 1,000 test images and labels,
checked 4,000 predictions, and reproduced every metric and all three paired
contrasts with zero discrepancies. The result supports this selected task and
KNN comparison, conditional on a fixed training cohort, finite search and fitted
models. It does not establish a universal NMF advantage, a retraining uncertainty
interval, or superiority over parametric classifiers. This study evaluated
official-test outcomes once; earlier studies above retain their own test-access
scope. Frozen receipts and the rendered figure are under
`runs/celeba-identity-validation-dense-knn-20261006/`.

## Completed unscreened-NMF supplement

The [supplement](celeba-unscreened-supplement.md) restores the dissertation's
standard CD-NMF+KNN baseline and adds unscreened SPA-NMF+KNN. It was designed
after the original face-test results were known and therefore provides
descriptive additional comparisons on the same people. The original four
models, saved predictions and results remain unchanged.

Each new method independently used the same five training folds, initial five
ranks and 104 K values, followed by the same single local refinement rule.
All **1,045 candidates and 5,225 candidate-fold results** qualified without
warnings. Standard CD selected rank 80 and K 101; unscreened SPA selected rank
80 and K 18. Both ranks remain at the explored upper boundary. These are
finite-search results and do not establish globally optimized NMF baselines.

Standard CD-NMF+KNN achieved **73.00% accuracy / 72.95% balanced accuracy**;
unscreened SPA-NMF+KNN achieved **61.90% / 61.82%**. Both used all 4,096 pixels,
qualified without fit/projection warnings, and matched their reference pipeline
labels and probabilities exactly. The CD final fit converged at 2,199 iterations
under the declared 20,000 cap. The original screened SPA result remains
81.40% accuracy / 81.37% balanced accuracy.

Descriptive paired balanced-accuracy differences favor screened SPA by
8.42 points over standard CD (95% interval [5.53, 11.30]) and 19.55 points over
unscreened SPA ([16.03, 23.14]). Independently selected ranks/K differ, so these
compare complete procedures; they are not fixed-setting causal effects of
screening. The post-evaluation audit checked all 2,000 new predictions and
4,000 original predictions, every reported metric and all 11 saved directional
contrasts, with zero discrepancies. It also records all 15 distinct six-method
pairs. Supplemental intervals are unadjusted and conditional on the fitted
models; they do not provide a new unseen-test confirmation.

The full six-method table and figure, both audit receipts, and frozen source/data
bindings are linked from the supplement, with raw receipts under
`runs/celeba-unscreened-supplement-20261006/`. The historical below-chance
simulation diagnostic remains a separate experiment.

## Completed screened-CD supplement

The [seventh-method supplement](celeba-screened-cd-supplement.md) adds absolute
screening followed by CD-NMF and KNN. It uses the same 2,000 training and 1,000
test identities, the original screened cutoff/rank/dense-neighbor grid, and
one local refinement. All **2,084 candidates and 10,420 candidate-fold scores**
qualify without warnings; 120 shared representation-fold fits underlie those
scores. The selected cutoff 8, rank 10 and K 39 have no flagged unresolved
grid boundary, although this finite search is not global optimization.

Screened CD obtains **79.00% accuracy / 78.98% balanced accuracy**. Its 124-pixel
mask exactly matches the original screened SPA mask. Rank and K differ from
SPA's rank 20 and K 11, so the comparison concerns independently tuned complete
pipelines. CD converges in 705 iterations; final fit, prediction and separate
public-estimator equivalence checks produce no warnings. Public/reference
labels and masks are exact, with zero maximum probability difference.

Screened SPA's advantage is **2.40 balanced-accuracy points**, with a
descriptive, unadjusted 95% paired interval of **[0.18, 4.62]**. Screened CD's
6.02-point advantage over unscreened CD has interval [3.10, 8.91]; its differences
from screened PCA (+0.47 points) and screening-only KNN (-0.60 points) remain
inconclusive. These additional results were designed after the prior six test
outcomes were known, and do not provide new independent confirmation.

The pre-evaluation audit verifies 120 training-fold screening masks independently
and all grid, fold, selection, numerical and source bindings. The post-evaluation
audit recomputes 1,000 new predictions, rechecks 6,000 existing predictions,
verifies seven reported directional contrasts and records all 21 distinct
seven-method pairs. Both audits pass with zero errors. All earlier scientific
sources and results remain unchanged. Full metrics, inference limits, the
seven-method figure and raw receipts are linked from the new report; its run is
`runs/celeba-screened-cd-supplement-20261006/`.

## Four-solver benchmark

Eight real Fashion-MNIST training-only smoke fits covered CD, SPA, MU and
ANLS-BPP at ranks 5 and 20. All eight passed without warnings; the record is
`runs/four-solvers-audit/training-smoke.json`. These checks measured no validation
accuracy and do not establish a predictive ranking.

The four-solver benchmark began under `runs/four-solvers-20261003/` and completed
under `runs/four-solvers-20261004-repair/` after the documented ANLS-BPP repair.
It compared the four solvers with absolute screening fixed on Fashion-MNIST and
a matched CelebA extension. The frozen rule selects SPA; the report remains
`selected_for_audit`. That comparison did not evaluate official-test outcomes. The reporter includes
a separate Fashion-MNIST baseline table matched
at 8,000 training observations and seed 43, so its three-task baselines are not
mistaken for comparisons against NMF averages across all 18 configurations.

## Local CelebA readiness and execution checks

The local dataset contains 202,599 aligned JPG images, a CSV with 202,599 unique
attribute rows, and `list_eval_partition.txt`. Image filenames, attribute IDs,
and partition IDs match exactly. No missing/extra IDs, duplicate IDs, malformed
rows, or invalid Smiling/Wearing_Necktie labels were found. The partition file's
MD5 is `d32c9cbf5e040fd4025c592c306e6668`, matching the
[torchvision dataset manifest](https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py).
It defines 162,770 training, 19,867 validation, and 19,962 test images.

Twelve sampled images decoded successfully (178 by 218 RGB) and passed the
original grayscale/resize preprocessing. This is not a full corruption scan.
The preliminary decode audit sampled filenames before partitions were available;
it performed no fitting or evaluation. Detailed integrity reports and local
paths are saved under ignored `runs/celeba-data-audit/` and
`data/celeba-metadata/source.json`. Dataset files remain outside distribution
archives. Identity annotations were not supplied; the identity audit remains open.

For each attribute, the real-data smoke uses seed 42, a uniform sample of 500
official training images and 200 official validation images, 64 by 64 grayscale,
one candidate per method, and five-fold calibration within the training sample.
No official test images are loaded by these model comparisons. Both attributes
use the same sampled image IDs. No class balancing or 20% allocation was applied.

The initial `max_iter=400`, `tol=1e-4` runs rejected all three CD candidates
(absolute screening, percentile screening, and unscreened NMF) because at least
one calibration fold hit the iteration cap. Their flagged majority fallbacks
are not CD performance results. Those diagnostics are retained in:

- `runs/celeba-smiling-development-20261003T073943Z-2a17afd4/`
- `runs/celeba-wearing_necktie-development-20261003T073943Z-01c0edca/`

Increasing only the iteration cap to 2,000, while retaining tolerance, data,
seeds, folds and method, allowed all 11 methods for each attribute to finish with
valid validation results and no fallback. CD training fits used at most 834
iterations across calibration and refitting. Successful runs:

- `runs/celeba-smiling-development-20261003T074137Z-4b06209a/`
- `runs/celeba-wearing_necktie-development-20261003T074137Z-16d94dee/`

This is a recorded numerical run setting, not a change to package defaults or
the dissertation's factorization method. The two attribute processes ran
concurrently with one numerical thread each; these timings are not isolated
performance benchmarks. These small, single-seed checks establish ingestion and
execution only. They do not complete tuning, select defaults, establish global
optimality, or resolve Wang's empirical concerns.

## Explicit-grid CLI fixture and resume

The `examples/celeba_parameter_study.py` CLI completed a synthetic fixture with
50 training rows, 30 validation rows and 4,096 nonnegative float32 features.
Both attribute names used generated binary labels and distinct fake image IDs;
no CelebA images were read. The declared grid was rank 2, percentile 5, absolute
cutoff 1 and KNN neighbors 1 and 3, with `max_iter=2000`, `tol=1e-4`, two workers
and one numerical thread per worker.

Each of the two attributes completed all 11 comparison methods. The 40 tuned
candidates produced 200 valid fold records from 100 unique fold representation
fits, reusing each representation across the two neighbor counts. All 22
development evaluations completed. An audit confirmed exact saved five-fold
assignments, disjoint fake training/validation IDs, valid checksums and bindings
to each method, attribute, run and selected candidate.

A real CLI resume reused all 100 representation folds with zero new
representation fits. All 100 fold receipts, 100 attempt receipts and 22 final
development-evaluation receipts remained byte-identical; no new files appeared.
Only the 20 CV reports and two run summaries changed to record reuse and current
call timing. These checks covered 289 JSON files.

The ignored local run is
`tmp/cli/celeba-parameter-study-20261003T085644Z-6d6343cf/`; audit scripts and
records are under `tmp/celeba-parameter-study-fixture-20261003/`. A preceding
attempt with a deeply nested output directory exceeded this Windows host's path
limit at a 283-character checkpoint filename. The successful check used the
shorter output root above. Fixture results establish execution and reuse only;
their scores have no interpretation as CelebA predictive performance.

## Separate profiler and reporter fixture checks

These execution checks were separate from the historical 65-test suite above. The stage
profiler was rerun against the completed synthetic CLI fixture and its exact
cached arrays, using one process and one native numerical thread. All 22 method
and attribute combinations reproduced the saved validation labels and class
probabilities exactly. A regression check that performs no fitting verified
estimator-clone isolation across attributes; the completed matched-prediction
run additionally confirmed the behavior during actual fits. The profiling record
is retained at
`tmp/profile-fixture-checks/celeba-stage-profile-20261003T103852Z-f2fea472/profiling.json`.

Reporter fixture checks passed for recomputed metric integrity, generated tables,
PNG visual inspection, and attachment of execution-transition and separate
profiling evidence. Reporting uses saved predictions and does not tune or fit
models. The profiler's additional fit costs remain separate from the original
study costs.

These synthetic checks ran while other work was active. Their timings establish
instrumentation and execution correctness, not isolated performance benchmarks.
They provide no real CelebA performance evidence, scientific replication,
completed-tuning claim or package-default decision.

## Withdrawn workflow records

The additional adaptive-search workflow was withdrawn. Its simulation run
`runs/simulation-smoke-20261003T082622Z-15fdf082/` and small CelebA run
`runs/celeba-smiling-development-20261003T082658Z-5e0a0b2c/` remain ignored local
historical records. Their bounded execution and resume checks neither validate
the current workflow nor provide a substantive method comparison.

The complete preceding validation record, workflow documentation and reference
metadata were copied with matching SHA-256 hashes into ignored
`tmp/withdrawn-adaptive-workflow/` before the current documentation was revised.
They are not included in package distributions or required for current studies.

## First development study

The first explicit-grid CelebA comparison completed on 4,000 official training
and 4,000 official validation images at seed 42. Five-fold training-only selection
covered 4,088 declared candidates, with 2,920 shared representation-fold attempts
across both attributes. All 124 chunks completed and all 22 selected
method/attribute models produced valid validation evaluations. Infeasible and
nonconverged candidates remain recorded and were excluded from selection.

The completed local run is
`runs/celeba-parameter-study-20261003T091802Z-68d2aa40/`. Its mixed four-to-eight
worker execution, data allocation and numerical settings are documented in
[the CelebA study](celeba-study.md). Its training-only review verified the complete
grid and identified outstanding boundary and numerical qualification questions.

Additional serial timing fits, using one process and one native numerical thread,
reproduced all 22 saved validation-label and probability arrays exactly. These
are extra fits at the selected settings, not scientific replications or replacements
for original tuning/final-fit costs. Their record is
`profiles/celeba-stage-profile-20261003T114136Z-3c25da37/profiling.json` within that run.

This is one development comparison. It does not complete tuning, select package
defaults, establish a statistically reliable method ranking or evaluate the
official test set.

Observed metrics and timing examples are summarized in
[the first development results](celeba-initial-results.md). The completed-run
integrity audit passed: exact folds, all candidates, receipt/configuration
bindings, official training/validation membership and all saved validation
metrics were checked independently.

## Repeated default study and disclosed amendment

The training-only refinement and repeated four-combination comparison completed
for three training sizes (2,000, 4,000 and 8,000), seeds 43, 44 and 45, and both
attributes. All 72 selected development evaluations were valid. All 216 serial
profiling repeats reproduced their original predictions and probabilities
exactly, without qualifying warnings. All 2,050 feasible five-fold candidates
were valid; 110 candidates were excluded before fitting by documented dimensional
feasibility checks. The CV computation used 1,025 shared representation-fold fits.

The completed repeated run is `runs/default-selection/repeated/`. Its original
decision, `runs/default-selection/decision/decision.json`, selected SPA with
percentile screening. After the development results, the screening preference was
amended to favor absolute screening for supervised-PCA alignment when it passes
the original accuracy gate.
The separate amendment at
`runs/default-selection/decision-spca-preference/decision.json` selects SPA with
absolute screening. The subsequent four-solver comparison also selected SPA with
absolute screening; its status remains `selected_for_audit`. Original evidence
and both CelebA decisions are preserved.
The amendment helper revalidates development receipts, cases and original decision
arithmetic; it does not fit models, change source defaults or read test outcomes.

The repeated CV execution took 319.674 seconds with eight workers, followed by
530.846 seconds for selected evaluations and serial profiling with one worker.
Each worker used one native numerical thread. Per-method CV sums are worker-wall
compute; serial case costs use medians of per-repeat fit-plus-prediction sums.
Nested component timings must not be added to total fit/prediction. Peak memory
for the full process tree was not measured.

All methods shared the same 4,000 validation images and overlapping training
allocations. These are descriptive development comparisons, not independent
datasets or proof of predictive equivalence. The theoretical review distinguishes
solver-neutral conditional representation bounds from the stronger assumptions
needed for screening retention, anchor recovery and sample-split KNN rates.

The original final-test attempt was stopped after eight development revalidations
and after test loading began. The chronology audit recorded no persisted test
cache, test prediction attempts, evaluations or comparison. Partial in-memory
decoding is unknown. The original attempt remains preserved, and no test outcomes
were inspected or used for the amendment. No new final-test run has started;
final evaluation requires a completed evidence audit and a new protocol freeze.
See [the results](celeba-default-results.md) and
[theory/implementation limits](theory-implementation.md).

## Four-solver repair and completion

The October 4 four-solver run encountered a reproducible native access violation
in the high-level SciPy least-squares dispatcher used by experimental ANLS-BPP.
The [numerical-repair record](numerical-repair-20261004.md) documents the retained
original run, captured failing matrices, direct-DGELSD workaround, unchanged
NMF objective/tolerances, and versioned checkpoint recovery. All 17 numerical
tests and 12 recovery tests passed. The complete suite at that revision, including the
existing portable-workflow tests, passed **266 tests in 61.417 seconds**;
see `tmp/four-solver-repair-full-check.log`. The repaired path also passed the
captured matrices and a two-sweep training-only integration with 5,035 direct
least-squares calls and zero inner failures. That deliberately short integration
is not a production-convergence claim. Crash dialogs are disabled only for test
and benchmark processes; crash logging remains enabled.

The successor is `runs/four-solvers-20261004-repair`. Verified unchanged-method
receipts retain their original bytes and timing. Every ANLS-BPP chunk was rerun;
old ANLS-BPP results remain historical evidence.

Recovery completed with 2,033 complete reports and two committed partial folds
reused; no ANLS-BPP receipts were imported. Production restarted at
2026-10-04 10:19:34 UTC with eight single-thread numerical workers and 926
Fashion-MNIST chunks pending. The first repaired ANLS-BPP chunks completed
successfully; the subsequent completion is recorded below.

The repaired comparison completed at 2026-10-05 10:37:02 UTC (6:37 a.m.
Eastern). All 2,959 Fashion-MNIST and 292 CelebA CV chunks were processed.
All 89 selected models qualified on development validation, each with three
valid timing repetitions. The generated [report](nmf-solver-comparison.md#development-results)
selects SPA with absolute screening under the frozen rule. SPA alone passes all accuracy gates;
its combined balanced accuracy is 80.7146%. Its weighted geometric deployment
cost is essentially tied with MU, so a speed advantage over MU does not explain
the choice. Matched Fashion-MNIST PCA, screening-only and raw-KNN baselines
outperform the NMF methods in mean balanced accuracy. Dataset/task omission
and workload sensitivities limit the scope of this recommendation. The report
remains `selected_for_audit`; official test data were not evaluated.

## October 20 approved source integration

The integration batch ran on the validation host with a total budget of 20 compute
threads and Python 3.12. Numerical stages run sequentially; one-thread and
20-thread measurements are separate. No local numerical checks were run.
Historical study and release results above retain their original source.

The frozen v3 source passed 260 regression tests, including classifier
capabilities, fitted SD scaling, margins, serialization, warning handling,
checkpoint identity and accounting, sparse/uncertain KNN fallback, numerical
range repairs, and recovery reporting. The original-trigger replay passed
114 NNLS checks and 204 screening checks. NNLS returned 102 qualified solutions
and preserved six expected iteration-cap errors and six explicit range errors.
Screening used 17 saved fixtures, four layouts and three cutoffs against the
represented-input oracle, including empty-selection and underflow failures.

All eight integrated KNN grid comparisons retained exact fold-level predictions,
candidate rankings and resume accounting. These fixed-fixture checks and
profiling repetitions do not add independent dataset replications or establish
universal predictive gains. Detailed source-bound receipts, including the
initial sparse-metadata error and corrected rerun, are retained in
`output/implementation-20261020/`.

The residual comparisons passed all 17 diagnostic cases and six actual public-fit
cases at each of one and twenty native threads. At one thread, all ten accelerated
diagnostic cases were faster in every paired repetition; complete large fits
improved by about 5–8% in median time, with unchanged fitted state and predictions.
Twenty-thread execution retained the original expression.

Final source v4 narrows this branch to recognized MKL pools at one thread.
Its three focused routing/arithmetic tests passed, along with actual 1/20-thread
routing checks and paired diagnostic/common-map fits against v3. All other
package source files were verified unchanged. AOCL/BLIS, OpenBLAS and mixed or
unknown backends retain the existing path. The source changes remain unreleased;
version 0.2.0 artifacts do not include them. Decisions, measured costs and
preserved receipts are recorded in the locally retained implementation report,
`output/implementation-20261020/IMPLEMENTATION-REPORT.md`, which is not bundled
with the public source.

Release annotation, October 20, 2026: version 0.3.0 includes the approved
integration described above. The unreleased status in that record describes
the source before this release; the 0.2.0 artifacts and historical validation
receipts remain unchanged. See the [0.3.0 release notes](release-0.3.0.md) for
the release scope and additional validation evidence.

## Pending scientific and release work

- Final evidence audit and release documentation for the completed
  CD/SPA/MU/ANLS-BPP comparison with absolute screening fixed, before freezing
  the general solver default for official-test evaluation.
- Exact historical-environment reproduction remains unavailable; the completed
  original-size controlled reconstruction identifies an insufficient iteration
  cap as a sufficient cause of below-chance behavior in that reconstruction.
- Broader independent real-data replication beyond the completed 900-draw
  simulation study, conditional CelebA resamples and identity-disjoint Smiling
  test. The latter completes an identity audit and frozen final-test comparison
  for that task; other datasets, retraining uncertainty and population scope
  remain open. Independently tuned PCA/screening-only and logistic development
  comparisons are complete within the reported finite grids.
- Full-study memory and broad scaling measurements. The completed BLAS benchmark
  includes native process peak-memory measurements; study figures and tables
  appear in the [Wang validation report](wang-validation-report.md).
- Broader compatibility verification and any package-index
  publication. Chang Cheng's original work is All Rights Reserved while the
  distribution license remains undecided; see [LICENSE](../LICENSE).
