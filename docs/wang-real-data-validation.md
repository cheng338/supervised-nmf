# Matched CelebA validation for Wang's comments

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. This study uses the package's feature-anchor SPA with an
absolute pooled-t screen, compares its contribution with matched alternatives,
and tests the choice between KNN and a parametric logistic classifier. It is a
development validation study; it cannot establish person-independent accuracy or
the conditional theoretical assumptions of the dissertation.

The initial run was paused at the user's request before validation predictions
to investigate BLAS acceleration. Its 109 completed CV checkpoints and source
snapshot remain under `runs/wang-real-20261006/`; it produced no validation
outcomes. After the numerical implementation passed equivalence tests, fresh
successor studies restarted all calibration. Old-source checkpoints are not
reused under the changed implementation.

The frozen local protocol is
`runs/wang-real-blas-20261006/protocol.json` (SHA-256
`59c194541a232ad822e28d2639aaae0096e1bb661c4b989956fa68ee0fb980f6`).
Its source snapshot, runtime versions, exact source-image filenames, cache hashes,
pilot folds and resampling indices were saved before the new validation outcomes.
The already published development-validation results were known when designing
this follow-up. No official test outcomes enter it.

## Completed fixed-setting results

All 520 evaluations in the twenty-resample fixed-setting study qualified, with
no evaluation failure or convergence warning. The independently retuned extension
is reported separately below. The fixed-setting pilot evaluated
1,200 candidate attempts: 1,079 qualified on every fold and 121 were invalid.
Representation sharing required 109 groups and 545 representation-fold attempts.

| Pipeline | Smiling BA ± conditional SE | Wearing Necktie BA ± conditional SE |
|---|---:|---:|
| Screened SPA + KNN | 0.8230 ± 0.0034 | 0.5587 ± 0.0025 |
| Screened PCA + KNN | 0.7888 ± 0.0049 | 0.5548 ± 0.0022 |
| Screening-only + KNN | 0.7806 ± 0.0052 | 0.5470 ± 0.0023 |
| Raw KNN | 0.6130 ± 0.0022 | 0.5233 ± 0.0017 |
| Unscreened SPA + KNN | 0.5852 ± 0.0037 | 0.5334 ± 0.0025 |
| Screened SPA + logistic | 0.8484 ± 0.0027 | 0.6878 ± 0.0021 |
| Screened PCA + logistic | 0.8723 ± 0.0017 | 0.7258 ± 0.0024 |
| Screening-only + logistic | 0.8860 ± 0.0011 | 0.7572 ± 0.0021 |
| Raw logistic | 0.8749 ± 0.0008 | 0.7408 ± 0.0022 |
| Unscreened SPA + logistic | 0.6252 ± 0.0081 | 0.6822 ± 0.0040 |
| Majority | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |

In this fixed-setting design, SPA helps the KNN path on Smiling relative to all
four primary KNN alternatives. Its matched-mask/rank PCA comparator attains
0.7981 ± 0.0033, versus SPA's 0.8230. On Wearing Necktie the matched PCA KNN
result is 0.5603 ± 0.0023, close to SPA's 0.5587. Logistic variants tell a different
story: PCA, screening-only and raw logistic all outperform screened SPA logistic
in mean balanced accuracy on both tasks. These findings support a specific KNN
use case, not universal superiority of NMF.

Wearing Necktie has 292 positive images among 4,000 validation images (7.3%).
Screened SPA KNN has mean precision 0.4121, recall 0.1325, F1 0.1993 and AP 0.1650.
Screened SPA logistic raises recall to 0.7336 but precision is only 0.1394, with
F1 0.2342 and AP 0.2301. Screening-only logistic attains precision 0.2414, recall
0.6842, F1 0.3567 and AP 0.3738. The higher minority recall therefore carries a
substantial false-positive burden; ordinary accuracy alone would obscure it.
All exact per-fit confusion counts and probabilities are retained.

Nine of twenty primary selected method settings remain on an unconstrained search
edge after the declared one-step refinement. Five Smiling NMF/PCA settings select
rank 40, and raw KNN selects K=50. Wearing Necktie SPA KNN also selects rank 40;
raw and screening-only logistic choose C=0.001. These are finite-search results,
not claims of globally optimal tuning. The observed tuning command took 347.674 s;
worker wall times sum to 1,411.960 s, and serial evaluation took 1,712.011 s under
concurrent system load. These are distinct measurements and must not be added as
if they were interchangeable compute-time estimates.

The full fixed-setting table (retained locally: `../runs/wang-real-blas-20261006/results.md`),
machine-readable summary (retained locally: `../runs/wang-real-blas-20261006/results.json`),
balanced-accuracy figure (retained locally: `../runs/wang-real-blas-20261006/balanced-accuracy.png`),
empirical precision-recall figure (retained locally: `../runs/wang-real-blas-20261006/necktie-precision-recall.png`)
and screening-mask figure (retained locally: `../runs/wang-real-blas-20261006/screening-masks-seed61010.png`)
are local development artifacts. The PR figure uses empirical steps consistent
with the AP calculation. The mask figure contains no face images or anatomical
ground-truth annotations.

## Completed independently retuned results

All 260 evaluations in the ten independently retuned resamples qualified. Each
resample repeated the full primary grid, five-fold CV and local refinement on its
own 2,000 training images before predicting the unchanged validation sample.
No original pilot configurations were transferred into this extension.

| Pipeline | Smiling BA ± conditional SE | Wearing Necktie BA ± conditional SE |
|---|---:|---:|
| Screened SPA + KNN | 0.8249 ± 0.0039 | 0.5643 ± 0.0042 |
| Screened PCA + KNN | 0.8060 ± 0.0043 | 0.5551 ± 0.0024 |
| Screening-only + KNN | 0.8030 ± 0.0040 | 0.5631 ± 0.0075 |
| Raw KNN | 0.6098 ± 0.0028 | 0.5230 ± 0.0031 |
| Unscreened SPA + KNN | 0.5786 ± 0.0050 | 0.5336 ± 0.0030 |
| Screened SPA + logistic | 0.8589 ± 0.0026 | 0.7165 ± 0.0069 |
| Screened PCA + logistic | 0.8735 ± 0.0022 | 0.7377 ± 0.0049 |
| Screening-only + logistic | 0.8870 ± 0.0013 | 0.7460 ± 0.0032 |
| Raw logistic | 0.8764 ± 0.0009 | 0.7394 ± 0.0034 |
| Unscreened SPA + logistic | 0.6181 ± 0.0133 | 0.6920 ± 0.0041 |
| Majority | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |

The main finding survives independent retuning: screened SPA helps KNN on Smiling,
but it is not the most accurate pipeline when logistic alternatives are included.
On Wearing Necktie its KNN mean is close to screening-only KNN and recall remains
low. The matched PCA KNN means are 0.8020 ± 0.0050 for Smiling and 0.5615 ± 0.0047
for Wearing Necktie. The corresponding matched PCA logistic means are
0.8689 ± 0.0019 and 0.7322 ± 0.0063. These secondary comparisons control the mask
and rank, with the classifier-search qualification described in the design below.

Retuned Wearing Necktie SPA KNN attains precision 0.3492, recall 0.1517, F1 0.2099,
AUC 0.6472 and AP 0.1539. SPA logistic attains precision 0.1667, recall 0.7205,
F1 0.2703, AUC 0.7763 and AP 0.2589. Screening-only logistic has the highest mean
BA among these tested pipelines (0.7460), with precision 0.2350, recall 0.6640 and
AP 0.3492. Raw logistic has higher AP (0.3665) and precision (0.2532), with recall
0.6240. Therefore the leading method depends on the stated metric; balancing
class recall does not remove the false-positive tradeoff.

The extension evaluated 11,996 candidate attempts, of which 10,776 qualified and
1,220 were invalid. It used 1,016 representation groups and 5,080 representation
fold attempts. No CV convergence warnings were captured; invalid candidates
reflect empty screens or infeasible ranks. Eighty-one of 200 primary selected
method settings retain an unresolved search edge. Aggregate tuning-command wall
time was 2,110.455 s; summed CV worker wall time was 8,177.649 s and summed serial
evaluation-command time was 583.962 s. Host load varied, so comparing these with
the pilot study's times is not an isolated speed benchmark.

The retuned table (retained locally: `../runs/wang-real-retuned-blas-20261006/results.md`),
full summary (retained locally: `../runs/wang-real-retuned-blas-20261006/results.json`),
balanced-accuracy plot (retained locally: `../runs/wang-real-retuned-blas-20261006/balanced-accuracy.png`),
precision-recall plot (retained locally: `../runs/wang-real-retuned-blas-20261006/necktie-precision-recall.png`)
and mask comparison (retained locally: `../runs/wang-real-retuned-blas-20261006/screening-masks-seed61010.png`)
retain the same reporting conventions as the fixed-setting study.

## Independent audit and paired comparisons

The independent audit (retained locally: `../runs/wang-validation-audit-20261006/all-audit.json`)
passed with zero errors. It verified all 780 real-data evaluations, source and
protocol bindings, CV qualifications and selected settings; independently
recomputed metrics from every saved prediction; checked all 120 matched SPA/PCA
mask-and-rank pairs; and reproduced all 48 fixed-case paired image-bootstrap
intervals. Across the two designs, 6,314 candidate-fold records were rejected:
5,390 empty screens, 482 infeasible SPA ranks and 442 infeasible PCA ranks. None
was excluded for a numerical convergence warning. Repeated predictions remain
predictions on the same 4,000 validation images, not extra independent images.

The auditor also calculated paired differences across training resamples. In the
fully retuned design, SPA KNN exceeds independently tuned PCA KNN on Smiling by
1.90 percentage points (conditional 95% t interval 1.32 to 2.48), and exceeds
the matched-mask/rank PCA KNN ablation by 2.30 points (1.67 to 2.93). On Wearing
Necktie, the SPA-minus-PCA KNN difference is 0.92 points (-0.25 to 2.09), leaving
its direction uncertain. Screened SPA logistic falls below screened PCA logistic
by 1.46 points on Smiling (-1.75 to -1.18) and 2.13 points on Wearing Necktie
(-3.69 to -0.57). Switching the screened SPA path from KNN to logistic raises BA
by 3.39 points on Smiling (2.95 to 3.84) and 15.22 points on Wearing Necktie
(13.40 to 17.03), including the path's scaling and class-weight choices.

These intervals measure Monte Carlo uncertainty in the mean paired difference
under the declared training-sampling procedure, conditional on the finite
reservoir and fixed development-validation images. They are unadjusted for
multiple comparisons and are distinct from the seed61010 image-bootstrap
intervals. Full paired records are available for the
twenty fixed-setting resamples (retained locally: `../runs/wang-validation-audit-20261006/real-fixed-paired.json`)
and ten independently retuned resamples (retained locally: `../runs/wang-validation-audit-20261006/real-retuned-paired.json`).

## Design

The existing CelebA seed43 cache contains 8,000 official-training images with
unchanged 64 × 64 grayscale preprocessing and natural attribute prevalence. Seed
61006 chooses an unstratified 2,000-image tuning pilot. The other 6,000 rows form a
finite reservoir. Twenty prespecified seeds, 61010 through 61029, each sample
2,000 rows without replacement from that reservoir. Samples are matched across
methods, overlap each other, and exclude every pilot image. Each fit predicts the
same previously used 4,000 official-validation images (seed42). Training and
validation filenames are disjoint; person identity separation is unverified.

The two tasks are Smiling and Wearing_Necktie. Each task has independently tuned
screened SPA, screened PCA, screening-only, raw-feature and unscreened SPA paths
with both KNN and logistic classifiers. A majority baseline preserves the meaning
of accuracy under imbalance. Unscreened SPA is explicitly this package's
feature-anchor factorization, not a reproduction of historical coordinate-descent
NMF or the original unexplained 32.42% result.

Five-fold stratified CV on the pilot selects each method/classifier combination
by mean balanced accuracy. The starting grids are ranks {2, 5, 10, 20}, absolute
|t| cutoffs {0, 2, 4, 8}, KNN neighbors {3, 9, 25}, logistic C {0.01, 0.1, 1, 10,
100}, and logistic class weights {None, balanced}. Logistic standardization and
balanced class weights use the training fold alone. Every screen and decomposition
is refitted inside each training fold. SPA uses the package's fit-transform
training coordinates and fixed-basis NNLS for incoming images. PCA uses the full
SVD. KNN uses unscaled coordinates and uniform Euclidean votes.

One prespecified local, coordinate-wise refinement step expands boundary choices
and adds interior points around each initial winner. It does not establish global
optimality, exhaustively search interactions, or guarantee that the final winner
is interior. All five folds must succeed without convergence warnings. Invalid
candidates retain their errors; no fallback is substituted. Logistic fits use a
1,000-iteration cap and tolerance 1e-4; each SPA NNLS solve permits 20,000 iterations.

For the matched PCA ablation, each classifier's screened SPA pilot winner supplies
the rank and cutoff. Those identical screening settings yield the same mask on
the same training data. PCA's final classifier is independently tuned on the
pilot at that fixed representation specification. This ablation helps separate
representation choice from independently selected ranks or mask sizes.
The ablation uses the declared starting classifier grid; it does not repeat the
primary paths' local classifier refinement. It therefore controls mask and rank,
while classifier settings and final search resolution can still differ.

## Uncertainty and computation

Settings are frozen before the new validation predictions. Twenty subsequent
fits measure **fixed-setting training-resample variability**. They are not twenty
complete retuned experiments. The reported SD and SD/√20 describe variability and
Monte Carlo error conditional on this reservoir, fixed pilot-selected settings,
and reused validation sample. They are not independent-dataset or population
standard errors.

A separately frozen extension independently repeats the entire starting grid,
five-fold CV, refinement, and matched PCA calibration inside each of ten training
resamples (seeds 61010–61019). Those settings are frozen before the corresponding
validation evaluation. Its top-level protocol is
`runs/wang-real-retuned-blas-20261006/protocol.json`, SHA-256
`462749735d68da37ec5ea375f722be19d4f918e5da3b64651b030cca98046e12`.
Each seed has a separate source-bound protocol, folds, candidates and selection
receipt. This extension measures variability of the full tuning procedure,
conditional on the same reservoir and reused validation images. It remains
distinct from independent-dataset or person-independent replication.

Paired stratified bootstrap intervals compare balanced accuracy using seed61010
only, resampling validation images within their true class. Both methods receive
the same bootstrap rows; recurring validation images from the twenty fits are
never pooled as additional observations. The 2,000-replicate percentile intervals
are unadjusted for multiple comparisons and do not account for unknown person
clusters. Bootstrap uncertainty and training-resample variability are distinct.

Each successor CV runner uses six workers and one native numerical thread per
worker. Representations
are shared across classifier settings, and the leading full-SVD PCA coordinates
are shared across ranks. Eight focused tests verify equivalence to actual package
NMF fits and explicit sklearn baseline pipelines, rank sharing, pilot exclusion,
paired bootstrap behavior, invalid-candidate rejection, feasible lower PCA ranks
when a larger grouped rank is invalid, and exclusion of unqualified cases from
bootstrap/PR reporting. The final NMF fits use
the public `NMFClassifier` directly. Timings record tuning, representation fitting,
classification and prediction separately. Evaluation is serial within the runner;
concurrent system workloads are not controlled, so these are observed costs, not
isolated solver speed claims. The fixed-setting runner, retuning runner and
simulation study initially ran concurrently; their aggregate worker count can
exceed the host's 16 logical processors. Peak RSS is unavailable in this
environment because psutil is absent; it must not be reported as zero.
The majority baseline's fit/prediction stage fields are zero placeholders; use
its measured `wall_seconds` if reporting that baseline's construction cost.
For the manually assembled baselines, `screen_seconds` includes column-selection
copies for training and incoming data. Separate stage attribution is therefore
coarse; the retained per-case total wall times are the appropriate end-to-end
cost record.

The successor sNMF implementation uses SciPy's public `dgemv` and
`dger` wrappers for SPA updates and prepares the fixed NNLS basis layout once;
it retains the same estimator and prediction conventions at that revision.
The [BLAS implementation report](blas-optimization.md) records the numerical
equivalence checks, thread limits and public SciPy references.

## Reproduction and artifacts

The historical workflow used the following local test target, study scripts
and arguments, with `PYTHONPATH=src`. These study tools are retained locally
and are not shipped in the public distribution. The original machine-specific
invocations are preserved with the local study record; this list documents
their configuration rather than providing runnable checkout examples.

```text
unittest discover -s tests -p test_wang_real_validation.py -v
scripts/validate_wang_real_data.py --output runs/wang-real-blas-20261006 --workers 6 --phase all
scripts/validate_wang_real_data.py --output runs/wang-real-retuned-blas-20261006 --workers 6 --design fully_retuned --repetitions 10 --phase all
scripts/report_wang_real_validation.py runs/wang-real-blas-20261006 runs/wang-real-retuned-blas-20261006
```

Existing compatible checkpoints are reused; changed frozen source is rejected.
Do not delete or rewrite earlier evidence to resume a changed design. The
`runtime.json` file records the actual Python/package versions and numerical
libraries. `selected-settings.json` records every chosen configuration and its
pilot score, with a timestamp preceding new validation prediction.

Local artifacts include `summary.csv`, `results.json`, `results.md`, individual
evaluation JSON/NPZ files, CV candidate/fold records, and PNG/SVG plots of balanced
accuracy and Wearing Necktie precision-recall. Seed61010 bases and selected masks
are retained for reproducibility. Local image filenames, predictions and data
remain excluded from package distribution. The numerical summaries and plots do
not redistribute face images.
