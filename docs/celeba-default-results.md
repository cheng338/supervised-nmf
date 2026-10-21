# CelebA default selection: SPA with absolute-score screening

Author: Chang Cheng. October 3, 2026.

The CelebA amendment selected **feature-anchor SPA with absolute-score screening**.
The amendment prefers supervised PCA's screening design when accuracy is close.
Both absolute-screening methods met the existing accuracy tolerances; SPA was
faster. Rank, score cutoff and KNN neighbors still require training-only tuning
for each task. The [theory review](theory-implementation.md) supports a simpler
connection between absolute screening and the retention theorem; its conditional
classification bounds do not favor one solver.

The subsequent [four-solver comparison](nmf-solver-comparison.md#development-results)
on Fashion-MNIST and CelebA also selected SPA with absolute screening and remains
`selected_for_audit`. Both CelebA decisions and their evidence are preserved
separately. No new CelebA final-test run has started; the interrupted prior attempt
has no recorded test predictions or outcomes.

The original time-first rule selected SPA with percentile screening. The screening
preference was amended after that development comparison and the theory review.
The amended decision is explicitly subsequent to those results;
it is not presented as the original preregistered rule. Original data, protocol,
decision and source snapshots are preserved. No official-test predictions or
metrics were available when the preference was applied.

## Evidence and decision

The [frozen protocol](default-selection-protocol.md) compared all four combinations
on 2,000, 4,000 and 8,000 official training images, with seeds 43, 44 and 45, for
Smiling and Wearing_Necktie. Every method used the same 4,000 seed-42 official
validation images. The primary metric was balanced accuracy (BA), averaged equally
over the nine configurations and the two attributes. Natural class proportions
were preserved. No validation score selected rank, screening amount or neighbors.

Training-only refinement evaluated 1,295 feasible candidates using 1,630 shared
representation-fold fits. Each method then transferred three qualified rank/screen
pairs, including its original qualified incumbent, to the nine configurations.
The second stage declared 2,160 candidates; 110 were excluded with explicit
dimensional proofs, leaving 2,050 candidates and 1,025 representation-fold fits.
Every fitted candidate qualified on all five folds. All 72 selected evaluations
were valid, and all 216 additional profiling repeats reproduced labels and
probabilities exactly. Every combination qualified in all 18 cases.

| Combination | Mean BA | Smiling BA | Necktie BA | Mean fit + 4,000-image prediction (s) | Geometric mean cost (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CD + percentile | 0.7176 | 0.8341 | 0.6011 | 2.734 | 1.581 |
| CD + absolute cutoff | 0.7233 | 0.8422 | 0.6044 | 1.769 | 1.507 |
| SPA + percentile | 0.7234 | 0.8396 | 0.6072 | 0.549 | 0.508 |
| **SPA + absolute cutoff (CelebA amendment)** | **0.7178** | **0.8377** | **0.5980** | **1.217** | **0.790** |

The unchanged accuracy gate accepts methods within one BA percentage point of
the best overall mean and two points of the best mean on each attribute. All four
passed. The author's preference then retains the absolute-screening methods;
within that group, SPA alone is within 5% of the lowest aggregate serial cost.
Its mean BA is 0.559 percentage points below the best overall method, and 0.544
points below CD with absolute screening. Its Smiling and Necktie regrets relative
to the best respective methods are 0.448 and 0.922 points. The accuracy reference
is still all four methods; it was not redefined to make the preference pass.

The tradeoff is explicit: SPA with absolute screening averages 1.217 seconds per
fit-plus-prediction workload, versus 1.769 for CD with absolute screening and 0.549
for SPA with percentile screening. These are observed practical differences,
not equivalence or significance claims. The original near tie between CD with
absolute screening and SPA with percentile screening was 0.015 BA percentage
points; the selected SPA-absolute alternative has the larger 0.559-point gap
reported above, still within the previously specified one-point tolerance.

## Time, numerical allowance and imbalance

Runs used Python 3.12 with AOCL and one native numerical thread per fit. CV used up
to eight workers. The serial pass ran after CV and audit processes finished, with
one worker and a recorded randomized method order within each case. Each case's
cost is the median of three **fit + prediction** sums. The table above then takes
arithmetic and geometric means across the 18 cases.

| Combination | MF fit (s) | KNN fit (s) | Total fit (s) | Prediction (s) | Recorded CV compute (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CD + percentile | 2.284 | 0.003 | 2.566 | 0.168 | 650.85 |
| CD + absolute cutoff | 1.313 | 0.003 | 1.601 | 0.168 | 614.48 |
| SPA + percentile | 0.069 | 0.002 | 0.343 | 0.205 | 252.58 |
| SPA + absolute cutoff | 0.230 | 0.001 | 0.509 | 0.705 | 372.33 |

Stage columns are means of case medians. Factorization and KNN fitting are already
inside total fit; do not add them again. Separate stage medians need not sum to the
median of paired sums. CV compute sums measured worker wall time with shared work
counted once; it is not isolated serial runtime, CPU time or elapsed tuning time.
The different tuned ranks and masks also prevent treating these as pure solver
speed comparisons.

All new fits used `max_iter=20000`, `tol=1e-4`. Six folds previously limited by a
10,000-iteration cap converged between 10,334 and 12,892 iterations. The constructor
and ordinary comparison defaults now use 20,000, without changing equations,
tolerance, coefficient conventions or feature scaling. For SPA this is each NNLS
solve's cap. Converged routines stop earlier. The old source and results remain
archived; truncated old fits are not relabeled as qualified results.

Wearing_Necktie has 292 positives among 4,000 validation images (7.3%). Always
predicting negative yields 92.7% ordinary accuracy but BA 0.5 and positive recall
zero. The selected default's mean positive recall was **0.2485**, with average
precision **0.1234**. For Smiling, mean recall was 0.8302 and average precision
0.9140. Thus the default decision does not imply strong minority-class detection,
or the best average precision: other methods can lead on secondary metrics.
Per-case precision, recall, F1, average precision, ROC AUC, ordinary accuracy,
prevalence and confusion counts are retained in the detailed decision artifact.

## Sensitivity and scope

Under the original time-first rule, SPA with percentile screening remains selected
when any one training size or seed is omitted. With its accuracy tolerances, it
also wins when prediction
cost is multiplied by 0.1, 1 or 10. At **100 times the prediction workload**, CD
with absolute screening wins when tuning cost is excluded. Adding the recorded
parallel tuning compute returns SPA with percentile screening in all declared
sensitivity cases; that mixed measurement is descriptive only. The primary
time-first result remains archived; the subsequent preference amendment has its
own recalculated sensitivity results.

The training sizes are nested within seed, different seeds overlap, and all cases
reuse the same validation images. These are repeated development configurations,
not independent datasets or independent validation replications. Person-level
separation is unverified. The finite grids and sample-size scaling of absolute
cutoffs are declared search choices, not guarantees of global tuning optimality
or theoretical threshold schedules. The earlier study's validation results were
known before this follow-up protocol was designed.

The [first CelebA comparison](celeba-initial-results.md) supplies the independently
tuned PCA, screening-only, raw KNN, unscreened NMF and majority context. Those
baselines were not rerun in all nine configurations of this four-way default
decision. Original and harder simulation studies remain separate research work;
they do not enter the CelebA decision. The later Fashion-MNIST and matched CelebA
comparison fixed absolute screening and added MU and ANLS-BPP. Its separate
decision does not change the scope or results of this study.

## Reproduction and records

The public machine-readable rule is
[`examples/celeba_default_decision_protocol.json`](../examples/celeba_default_decision_protocol.json).
Its SHA-256 is
`ac24396d7ddddef4f80adc875bfbba3456edffd929bc1e766993f10a6ac64d6d`.
The Stage 2 declared-grid hash is
`d6ee839c1a1c8f4a957a998444d0aab53d785a621236b5e28f9ea1a7cc37783e`;
the feasible-grid hash is
`85f9035bac247cec916ac799830c364f2570dc97288cffd5c4644beba1b53ecb`.

Local evidence is under `runs/default-selection/`: `refine/`, `repeated/`,
`decision/decision.json`, `decision/decision.md`, `decision/cases.csv`,
`decision/accuracy-cost.png`, `audit/`, and both `source-before-default*` snapshots.
The subsequent author-preference decision is in `decision-spca-preference/`.
The decision JSON binds its source run, selector, input receipts and frozen rule
with hashes. Local data, predictions and checkpoints are excluded from package
archives. These records can be inspected without rewriting them to match the new
constructor defaults; old runs intentionally refuse resume under changed source.

The helpers are `plan_celeba_default_study.py`,
`prepare_celeba_default_caches.py`, `plan_celeba_default_repetitions.py`,
`preflight_celeba_default_grid.py`, `celeba_default_study.py`,
`select_celeba_default.py` and `evaluate_celeba_frozen_default.py` under `scripts/`.
They preserve declared grids and compatible checkpoints. There is no automatic
adaptive search policy inside estimator fitting.
