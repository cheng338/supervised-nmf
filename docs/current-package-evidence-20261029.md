# Current-package CelebA comparison — October 29, 2026

Author: Chang Cheng.

All **34 prespecified final evaluations qualified**: 17 task/configuration arms
fitted separately in AOCL and MKL. The primary Smiling comparison gave tuned
current SPA **81.35% balanced accuracy**, versus **80.10%** for independently
tuned screened PCA. The difference was **+1.24 percentage points**, with a paired
95% interval of **−1.00 to +3.47 points**. This interval includes zero and does
not establish superiority or equivalence.

These results address review finding 4c's scoped gap in current-package
classification evidence. A favorable result was not a completion criterion.
The scientific outcomes are complete; the execution history retains an unknown
duration and missing final controller observations for an earlier Smiling/PCA
interruption. Accordingly, the report records `scientific_results_complete=true`
and `complete=false`. The study was conducted during a release pause. Chang Cheng
subsequently authorized the 0.4.1 GitHub release on October 29, 2026; that decision
does not change this evidence qualification or the package defaults.

## Data and scope

The study used 1,999 training images and 1,000 query images from the previously
exposed CelebA cohort. A metadata audit excluded training image `124398.jpg`
before fitting because its global identity/exact-JPEG component touched the
query cohort. The training images comprise 1,999 annotated identities and
1,998 components; the query images comprise 1,000 identities and singleton
components. The two retained training rows sharing a component stayed together
in cross-validation.

Both partitions use the exact archived 64×64 grayscale float32/255 features
produced with Pillow 11.1. Current Pillow 13 decoding did not reproduce those
features; that unsuccessful check is preserved separately. This comparison
qualifies current estimation on the archived features, not the Pillow 13 image
preprocessing path.

Smiling was the primary task; Wearing_Necktie was a prespecified secondary task
for imbalance. Query class counts were 490/510 and 924/76, respectively, in
class order 0/1. Natural prevalence, uniform KNN votes, unscaled coordinates and
the fixed decision rule were preserved. No class balancing or threshold tuning
was performed. Identity separation prevents within-study person leakage but
does not undo previous exposure to these observations and outcomes.

## Calibration and source transition

Training-only selection used the same five stratified component-group folds
within each task, balanced accuracy, and seed 62106. Screening and representation
fitting were fold-local. A frozen policy expanded boundaries and refined crossed
rank/cutoff neighborhoods, retaining exact-score ties and numerical failures.
All eight selections qualified within this finite search; this does not establish
global optimality over every possible interaction or configuration.

The original study completed seven selections before native AOCL failures
interrupted Smiling/SPA. Following the instruction to rerun only crashed work,
the continuation preserved those seven selections, Smiling/SPA stages 0000–0007,
and all **97 completed representation/fold cells** in stage 0008. It computed
the **133 missing cells** and subsequent required stages with the
[package-side AOCL workaround](aocl-nnls-workaround.md). Original receipt bytes,
identities and source attribution remain unchanged. No AOCL library was modified.

Thus calibration combines pre-fix evidence with post-fix recovery. It is not an
entirely current-source calibration, and a full retuning with the new source is
not proved to select the same parameters. **Every final fit and prediction used
the frozen current package**, including the AOCL-specific NNLS workaround.
Final configurations and their selection origins were frozen and independently
audited before new query evaluation. MKL refitted the identical frozen settings;
it supplied a portability check, not another independent statistical replicate.

The selected configurations are below. Cutoffs are rounded here; exact values,
ties, boundary certificates and source origins are in the frozen records.

| Task | Method | Absolute cutoff | Rank | K |
| --- | --- | ---: | ---: | ---: |
| Smiling | Raw KNN | — | — | 12 |
| Smiling | Screening + KNN | 6.121094 | — | 12 |
| Smiling | Screened PCA + KNN | 6.246094 | 32 | 14 |
| Smiling | Screened SPA + KNN | 8.392388 | 17 | 16 |
| Wearing_Necktie | Raw KNN | — | — | 1 |
| Wearing_Necktie | Screening + KNN | 5.356351 | — | 1 |
| Wearing_Necktie | Screened PCA + KNN | 4.509079 | 41 | 1 |
| Wearing_Necktie | Screened SPA + KNN | 4.517136 | 21 | 1 |

Tuned SPA uses `training_coordinates="auto"`, resolved to projected training
coordinates, with `max_iter=20000` and `tol=1e-4`. The literal `NMFClassifier()`
arm preserves all constructor settings, including `random_state=None`; its
results must not be described as those of the tuned method.

## Final classification results

These are AOCL results, in percent. All corresponding MKL predicted labels
were identical. The full report also retains both class recalls, precision,
F1, AUROC, average precision, confusion counts and unavailable quantities.

| Task | Method | Accuracy | Balanced accuracy | Positive recall |
| --- | --- | ---: | ---: | ---: |
| Smiling | Raw KNN, tuned | 60.60 | 60.51 | 65.10 |
| Smiling | Screening + KNN, tuned | 79.40 | 79.38 | 80.59 |
| Smiling | Screened PCA + KNN, tuned | 80.10 | 80.10 | 80.00 |
| Smiling | Screened SPA + KNN, tuned | 81.30 | 81.35 | 79.02 |
| Smiling | Literal `NMFClassifier()` | 54.30 | 54.41 | 48.82 |
| Smiling | Training-majority reference | 49.00 | 50.00 | 0.00 |
| Wearing_Necktie | Raw KNN, tuned | 89.70 | 58.20 | 21.05 |
| Wearing_Necktie | Screening + KNN, tuned | 90.10 | 62.64 | 30.26 |
| Wearing_Necktie | Screened PCA + KNN, tuned | 90.80 | 65.44 | 35.53 |
| Wearing_Necktie | Screened SPA + KNN, tuned | 89.60 | 62.98 | 31.58 |
| Wearing_Necktie | Literal `NMFClassifier()` | 92.10 | 50.44 | 1.32 |
| Wearing_Necktie | Training-majority reference | 92.40 | 50.00 | 0.00 |

For the primary Smiling contrast, the ordinary-accuracy difference was
**+1.20 points** (95% interval **−1.00 to +3.40**), accompanying the balanced-
accuracy contrast reported above. Wearing_Necktie favored PCA descriptively:
SPA minus PCA balanced accuracy was **−2.46 points** (interval **−6.45 to +1.33**).
SPA detected 24/76 positives and PCA 27/76. The literal constructor detected
only **1/76 positives**, despite 92.1% ordinary accuracy. These outcomes do not
support treating high overall accuracy as successful rare-class detection or
the untuned constructor as a substitute for training-only calibration.

Intervals use 10,000 paired class-stratified bootstrap samples of the singleton
query components, seed 62107. They are conditional on the fixed fitted models
and this cohort. They omit calibration uncertainty, repeated-training variation,
earlier data exposure and transport to another population. Only the AOCL Smiling
SPA-minus-PCA balanced-accuracy contrast was primary. Other contrasts are
descriptive and are not multiplicity-adjusted confirmations.

## Fixed historical configurations

The prespecified Smiling bridge refitted historical settings with current source
on the cleaned 1,999-row training cohort. It is not an exact reproduction of the
earlier 2,000-row experiment, and its settings were not chosen from these results.

| Configuration | Accuracy % | Balanced accuracy % |
| --- | ---: | ---: |
| Raw KNN, K=28 | 62.20 | 61.99 |
| Screening cutoff=6, K=10 | 79.70 | 79.68 |
| PCA cutoff=8, rank=40, K=10 | 79.40 | 79.43 |
| SPA cutoff=8, rank=20, K=11, projected training coordinates | 81.20 | 81.18 |
| SPA cutoff=8, rank=20, K=11, legacy anchor training coordinates | 81.70 | 81.69 |

Within each environment, the two SPA bridge fits had identical stored screening
masks, full bases and selected bases, while their training coordinates differed.
This qualified the intended coordinate-policy comparison. Projected minus legacy
balanced accuracy was **−0.51 points** (95% interval **−2.22 to +1.20**). It does
not establish a general preference between these coordinate policies.

Across all 17 arms, AOCL and MKL had zero changed predicted labels. Several
stored PCA/SPA bases and coordinate arrays were not bitwise identical across
the environments. Matching predictions therefore do not imply identical
floating-point execution or universal backend equivalence.

## Costs and retained failures

Execution used the local PC, an eight-thread total budget, one native thread per
worker, and sequential final fits. Completed Smiling/SPA calibration component
costs comprise **1,409.34 seconds of preserved historical work** and **405.76
seconds of new work**. The new controller process took **663.01 seconds**,
including startup, source/receipt verification, checkpoint work and calculations.
The other seven selections required no new calibration fits. Component compute
and inclusive process durations are different accounting views and must not be
added together; nested stage durations are already included in their parent.

Representative isolated final operation times were:

| AOCL task/method | Fit seconds | Predict 1,000 rows, seconds |
| --- | ---: | ---: |
| Smiling, tuned SPA | 0.234 | 0.042 |
| Smiling, tuned PCA | 0.163 | 0.027 |
| Wearing_Necktie, tuned SPA | 0.349 | 0.066 |
| Wearing_Necktie, tuned PCA | 0.176 | 0.026 |

Each is one fitted operation, not a repeated benchmark or an estimate of AOCL
padding overhead. Per-stage costs, both environments and all arms are retained
in the report. Approximately 100ms process-tree RSS sampling includes startup,
data loading and artifact work; it is not an exact peak or stage-specific memory
measurement. One MKL sample series has a recorded gap. No memory-saving slowdown
is proposed by this study.

All four recorded historical access violations remain in the execution record,
including the failed nested stage. Successful recovery does not turn them into
successful observations or erase their costs. The earlier native investigation
reproduced an AOCL fringe overread; not every historical fault had its own
captured instruction. The resumed selection and all final evaluations qualified,
with no unresolved native selection failure and no final fit/prediction warnings.
The earlier Smiling/PCA controller interruption still lacks its full process
observations and duration, so the complete historical process-time total remains
unavailable. Diagnostic runs and helper tests are not charged as tuning attempts.

## Evidence and interpretation

This document is the portable public summary. The separately retained
`current-package-evidence-aocl-padding-20261029` archive contains the full
`report/report.txt` and machine-readable `report/report.json`; the original
`current-package-evidence-20261029` archive remains unchanged. Those records bind
the exact metrics, selections, failures, costs and input hashes. Raw study
archives are not included in package distributions; their identifying hashes
are recorded below.

Before the final configuration freeze, ten study-helper suites passed **263
tests in each environment**, AOCL and MKL. The pre-query audit passed, and the
inference pass independently recomputed
metrics and paired contrasts from the recorded predictions with no scientific
qualification issues. Source bindings remained unchanged through the new final
evaluations and inference execution.

Key SHA-256 bindings:

| Record | SHA-256 |
| --- | --- |
| Continuation protocol | `659342a704817785e0fe7ba1310ec7cc590b5f3e217325d7efdb942ed00171b5` |
| Final configuration freeze | `847eaf853f8d2013bab1c4a3b91dae624eed9cb4845d7f457aef921bd3521bac` |
| Pre-query audit | `9cd390c75e7051d18fa84fe05d7eed8836ab6c74c7119418f445ca22bdcbc1dd` |
| Statistical results | `a8809388eec014de3d7867d357d1b9803ccfe42dd6bcacef14e91358a0e857b8` |
| Final report JSON | `2f3da74c3e45402c16191e7588aafae21fb86dca2ee96aff4d174b907676f7e8` |
| Preserved historical manifest | `23250adb2f84e547e633a4cb5fdf0a3673e191e42fe1fa6bf55f2eaa56a77625` |
| Current `_nnls.py` | `823655ea1be23565e0e8f185a1d151a5adbee068587ed92b5102340fde590a4b` |
| Added `_aocl.py` | `40cdffdad759adbff6e75fda18cbf085148233b84196a6b152284bf4ce999294` |

The finding-4c evidence gap is addressed within this declared retrospective
comparison, with mixed calibration provenance and the remaining execution-history
qualification explicit. These observations do not prove SPA's theoretical
assumptions, establish general superiority over other solvers/classifiers,
constitute fresh independent confirmation, or authorize a release or default
change. Unfavorable and inconclusive results are retained without retuning on
the query outcomes.
