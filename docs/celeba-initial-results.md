# First CelebA development comparison

Author: **Chang Cheng**. Recorded October 3, 2026.

The first explicit starting-grid comparison is complete. These are development
results from one paired sample, not final-test results or selected package
defaults. The [study protocol](celeba-study.md) defines the data, grids and
numerical settings.

Historical scope: the later [repeated default study](celeba-default-results.md)
and disclosed author-preference amendment selected SPA with absolute pooled-t
screening for a scoped practical default. The later
[four-solver comparison](nmf-solver-comparison.md#development-results)
also selects SPA with absolute screening and remains `selected_for_audit`.
The one-sample results and limitations below are preserved separately.

## Design and completed work

Both attributes used the same 4,000 official training and 4,000 official
validation image IDs, sampled with seed 42. Images retained the original
64 by 64 grayscale preprocessing. Each tunable method selected its own parameters by
balanced accuracy on five stratified training folds, with screening and
factorization fitted inside each fold. Class proportions were preserved.

All 124 grid chunks completed: 4,088 declared candidates and 2,920 shared
representation-fold attempts. Of those attempts, 2,572 were valid, 342 failed
and six did not converge. These are fold-level attempt counts, not candidate
counts. A candidate required all five folds to qualify. All 22 selected
method/attribute models subsequently produced valid validation evaluations.

## Observed validation results

Values below are percentages, rounded to one decimal place. Each row uses the
parameters selected separately for that method and attribute. Differences are
descriptive; no confidence intervals or statistically reliable ranking have
been established.

| Method | Smiling balanced accuracy | Necktie balanced accuracy | Necktie positive recall |
| --- | ---: | ---: | ---: |
| CD NMF, percentile | 84.4 | 60.7 | 27.4 |
| CD NMF, absolute pooled-t cutoff | 85.2 | 61.2 | 27.4 |
| SPA NMF, percentile | 83.4 | 63.5 | 32.9 |
| SPA NMF, absolute pooled-t cutoff | 79.4 | 60.1 | 24.7 |
| PCA, percentile | 82.2 | 61.5 | 27.1 |
| PCA, absolute pooled-t cutoff | 80.0 | 62.1 | 28.4 |
| Screening only, percentile | 81.5 | 61.5 | 27.1 |
| Screening only, absolute pooled-t cutoff | 79.5 | 62.0 | 28.1 |
| Independently tuned unscreened CD NMF | 74.0 | 53.1 | 12.3 |
| Raw KNN | 61.4 | 58.0 | 20.2 |
| Majority class | 50.0 | 50.0 | 0.0 |

CD with absolute screening had the highest observed Smiling validation balanced
accuracy, using rank 20, cutoff 8 and 25 neighbors. SPA with percentile screening
had the highest observed Necktie validation balanced accuracy, using rank 10,
2% retention and one neighbor. The different outcomes do not justify a universal
solver or screening default from this single sample. They also do not establish
factor recovery or resolve the theoretical differences between CD and SPA.

Wearing_Necktie had 292 positive validation observations out of 4,000. Majority
prediction achieved 92.7% accuracy while detecting none of the positives. The
SPA-percentile model detected 96 of 292 positives, with 30.8% precision and
32.9% recall. Its 89.7% accuracy was lower than the majority baseline despite
better balanced accuracy and positive recall. Minority-class detection remains
limited and requires further evidence.

Every tuned Necktie method selected one neighbor, yielding probabilities of
either zero or one. Consequently its ROC AUC equals balanced accuracy and its
PR/ROC curve has few operating points. Average precision, full confusion counts
and all other metrics are retained in the local report.

## Timing and resource accounting

The study began with four worker processes and continued with eight, with one
native numerical thread per worker. The continuation reused 371 committed fold
receipts exactly; four interrupted uncommitted attempts were retried. Its
calibration call took 8,053.94 seconds (about 2 hours 14 minutes), separately from
the preceding observed process span of 1,008.25 seconds and the handoff. These
figures are not a reconstructed end-to-end elapsed time. Per-method worker
compute totals are distinct from elapsed wall time, and reused committed costs
are counted once. Lost stage time from the four uncommitted attempts is unknown.

Original tuning, final fit, prediction and available decomposition times were
saved. An additional serial profiling pass measured screening, decomposition,
classifier fitting and prediction stages for every selected model. All 22
profiles reproduced the original validation labels and probabilities exactly.
These extra fits do not replace original costs or count as scientific
replications. Components are included in their respective fit/prediction totals.

For example, the separate selected-setting profile for Smiling CD-absolute took
1.846 seconds for total fit, including 1.590 seconds for decomposition, and
0.145 seconds for prediction on 4,000 validation images. Necktie SPA-percentile
took 0.284 seconds for total fit, including 0.034 seconds for decomposition, and
0.150 seconds for prediction. Each is one instrumented timing repeat; different
selected ranks and masks prevent interpreting these as an isolated solver
speed comparison. All methods' stage times are in the local report.

The host has 39.23 GiB total physical RAM, shared with other applications.
Sampled worker memory records are retained; a whole-process-tree peak was not
measured.

## Training review and audit trail

Training-only review identified unresolved boundaries, including Smiling
CD-percentile at the largest tested neighbor count, Smiling SPA-absolute at
the largest tested rank, and Necktie PCA-percentile near its retained-feature
dimension. Some CD fits reached the 10,000-iteration cap. The saved follow-up
plan proposes separately checking a larger cap at the same tolerance, refining
screening/rank intervals and extending appropriate boundaries. The NMF subset
was subsequently completed in the [repeated default study](celeba-default-results.md);
the broader PCA/screening-only follow-up remains open. The natural endpoint of
one neighbor cannot be expanded downward.

The follow-up plan was constructed from training CV records without access to
validation performance. One validation row had already been printed during study
execution, so overall blinding is not claimed. The draft plan and its hash were
saved before inspection of the full validation tables. The final version retained
the same parameter grids and clarified attempt budgets; both hashes are recorded.
Follow-up grids were fixed independently of the reported validation outcomes.

The repeated paired default comparison has since completed. Uncertainty analysis,
the identity-overlap audit and final-test evaluation remain open. The official
test partition was not loaded by this initial study. Original numerical failures
are preserved; no fallback result is presented as a successful factorization.

The ignored local run directory is
`runs/celeba-parameter-study-20261003T091802Z-68d2aa40/`. It contains:

- `report/report.md`, `report/summary.json` and the two PR/ROC figures;
- `cv-review/training-grid-review.json` and the proposed follow-up plan;
- `profiles/celeba-stage-profile-20261003T114136Z-3c25da37/profiling.json`;
- `audit/`, configuration, predictions, checksummed fold receipts and transition records.

Local datasets, image IDs and per-observation predictions are excluded from
package distributions. The source archive includes the scripts needed to
regenerate the study and report from locally supplied data. Chang Cheng's
authorship and original-script provenance are preserved.
