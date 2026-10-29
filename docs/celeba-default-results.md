# CelebA default selection: SPA with absolute-score screening

Author: Chang Cheng. October 3, 2026.

The saved source for this study used `training_coordinates="fit_transform"`:
SPA classifiers trained on the original-scale anchor columns and predicted
using fixed-basis NNLS. The accuracy and timing below apply to that convention.
Since [v0.3.0](release-0.3.0.md#spa-training-coordinates-migration), the default
SPA classifier also projects its training observations against the fitted basis.
The historical selection does not by itself establish the performance of that
later training procedure.

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
separately. The final-test phase of this original default-selection protocol
remains unfinished; its interrupted attempt has no recorded test predictions or
outcomes. Later, separate studies did evaluate projected SPA training coordinates,
including an October 12 official-partition study. Their settings, results and
limits are summarized in [the validation record](validation.md#evidence-for-projected-spa-training-coordinates);
they do not retroactively complete this frozen protocol.

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

### Original time-first rule

Under the original time-first rule, SPA with percentile screening remains selected
when any one training size or seed is omitted. With its accuracy tolerances, it
also wins when prediction
cost is multiplied by 0.1, 1 or 10. At **100 times the prediction workload**, CD
with absolute screening wins when tuning cost is excluded. Adding the recorded
parallel tuning compute returns SPA with percentile screening in all declared
sensitivity cases; that mixed measurement is descriptive only. This original
decision remains archived separately from the preference amendment below.

### Amended screening-preference rule

The following tables report all 32 sensitivity settings saved with the
October 3 amendment. Each calculation applies the global accuracy gate before
preferring eligible absolute-screening methods. The accuracy reference and
cost normalization retain all operationally qualified methods. The preference
cannot admit a method that fails that gate or numerical qualification.

Margins are allowed balanced-accuracy shortfalls in **percentage points**;
the per-attribute margin is twice the macro margin. Prediction weight multiplies
the saved prediction time in each paired timing repeat. A weight of one is the
primary workload: one fit plus prediction on the fixed 4,000-image validation
set. These weights are cost-model sensitivities, not measured runs on larger
prediction batches.

**Serial fit and prediction cost, excluding tuning**

| Macro / attribute margin (pp) | Prediction weight 0.1 | 1 | 10 | 100 |
| --- | --- | --- | --- | --- |
| 0 / 0 | SPA + percentile (fallback) | SPA + percentile (fallback) | SPA + percentile (fallback) | SPA + percentile (fallback) |
| 0.5 / 1 | CD + absolute | CD + absolute | CD + absolute | CD + absolute |
| 1 / 2 | SPA + absolute | SPA + absolute | CD + absolute | CD + absolute |
| 2 / 4 | SPA + absolute | SPA + absolute | CD + absolute | CD + absolute |

**Serial cost plus recorded parallel tuning compute**

| Macro / attribute margin (pp) | Prediction weight 0.1 | 1 | 10 | 100 |
| --- | --- | --- | --- | --- |
| 0 / 0 | SPA + percentile (fallback) | SPA + percentile (fallback) | SPA + percentile (fallback) | SPA + percentile (fallback) |
| 0.5 / 1 | CD + absolute | CD + absolute | CD + absolute | CD + absolute |
| 1 / 2 | SPA + absolute | SPA + absolute | SPA + absolute | CD + absolute |
| 2 / 4 | SPA + absolute | SPA + absolute | SPA + absolute | CD + absolute |

At zero margins, no method meets both attribute optima. The original
minimum-worst-regret fallback selects SPA with percentile screening; it does
not satisfy a zero-loss accuracy gate, and the absolute preference is not
applied. At every nonzero margin shown, the preference is applied only to
absolute-screening methods that pass the gate.

The second table adds accumulated parallel worker-wall tuning compute to serial
fit/prediction cost. It is a mixed-measurement sensitivity, not elapsed serial
or end-to-end runtime. Neither table replaces the primary amended choice at
the 1/2-point margins and prediction weight one.

The saved omission calculations use the primary margins and workload:

| Omitted development configurations | Amended selection |
| --- | --- |
| Training size 2,000 | SPA + absolute |
| Training size 4,000 | SPA + absolute |
| Training size 8,000 | SPA + absolute |
| Seed 43 | SPA + absolute |
| Seed 44 | **CD + absolute** |
| Seed 45 | SPA + absolute |

Without seed 44, SPA with absolute screening has a macro BA shortfall of
**1.019951 points**, slightly above the one-point allowance. Its per-attribute
shortfalls remain within two points. This reversal is caused by the macro
accuracy gate. The overlapping configurations and shared validation images
make these descriptive sensitivities, not independent votes for a solver or
estimates of a population selection probability.

### Retrospective boundary investigation

An October 28 investigation reconstructed all 72 evaluations and 216 timing
profiles from the saved receipts and reproduced both rules' decisions. An
independent implementation agreed across 5,321 checks in each of two numerical
environments, including 20 hand-derived edge cases and 400 synthetic panels.
These checks establish reproduction and agreement on the tested inputs; they
add no new fitted models or predictive validation data. The boundaries below
were investigated retrospectively and were not part of the original protocol.

With macro allowance `delta` and per-attribute allowance `2 * delta`, each
method enters the accuracy gate at the larger of its macro regret and half
its largest attribute regret, apart from the declared numerical allowance:

| Method | Macro allowance at gate entry (pp, approximately) | Controlling restriction |
| --- | ---: | --- |
| CD + percentile | 0.584725 | Macro accuracy |
| CD + absolute | 0.141175 | Attribute accuracy |
| SPA + percentile | 0.126365 | Attribute accuracy |
| SPA + absolute | 0.558723 | Macro accuracy |

SPA with absolute screening therefore fails the 0.5-point gate on the
full-precision evidence. Rounding the published means does not explain the
different selections at 0.5 and 1 point. These are practical policy thresholds,
not tests of statistical equivalence or guarantees supplied by the theory.

Costs use the median of paired `fit + weight * prediction` sums, not the sum
of separate fit and prediction medians. The analysis accounted for changes in
which repeat supplies that median. It also checked the common geometric
normalization: relative method costs equal the geometric mean of their paired
raw cost ratios, so retaining percentile methods in the normalization does not
create a relative advantage between the two absolute-screening methods.

At the adopted 1/2-point accuracy margins, the approximate cost boundaries are:

| Cost model | CD enters SPA's 5% cost band | Equal aggregate cost | SPA leaves CD's 5% band; selection changes to CD |
| --- | ---: | ---: | ---: |
| Fit + weighted prediction | 3.804× | 4.230× | **4.711×** |
| Same cost plus recorded parallel tuning compute | 39.379× | 45.833× | **53.213×** |

CD becomes cheaper near 4.230×, but SPA retains the tuning-cost tie-break
while it remains within CD's 5% band. The selected method changes near 4.711×.
Numerical comparison slack affects microscopic boundary details; the displayed
values are approximate. They extrapolate historical timings and do not measure
hardware scaling or the later projected-training procedure.

At the primary workload, each of the three individual timing repeats preserved
SPA with absolute screening. Cost-band ratios from 1 to 3 also preserved it
at the tested values. All nine individual training-size/seed block deletions
preserved it, although deleting the whole seed-44 panel did not. The block
results do not negate the seed sensitivity: deleting a seed removes its three
nested-size configurations together.

The amendment remains a preference-based historical choice under stated
accuracy allowances, workload assumptions and development-panel weighting.
The investigation preserves its primary selection and does not change the
current estimator default or add an automatic solver-switching policy. A small
crossing of a historical threshold is not, by itself, grounds for a new adoption
decision. Such decisions also require methodological, theoretical, numerical
and practical justification.

### Scope of the evidence

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

The original frozen machine-readable rule is
[`examples/celeba_default_decision_protocol.json`](../examples/celeba_default_decision_protocol.json).
Its SHA-256 is
`ac24396d7ddddef4f80adc875bfbba3456edffd929bc1e766993f10a6ac64d6d`.

The file retains the historical environment label `py12_df_aocl` because the
saved records identify its exact bytes. Version 0.3.3 replaced that label with
`Python 3.12 with AOCL`, changing the hash without changing the decision rules.
The original bytes are now restored at the same path. The label records the
historical configuration; it does not select an environment for current work.
Historical receipts and strict verification checks remain unchanged. Complete
replay also requires the archived sources, data and compatible recorded paths.

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
