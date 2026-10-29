# Selecting one solver and screening default

Author: Chang Cheng. Protocol version 1, October 3, 2026.

This follow-up study will select **one package-wide combination** from coordinate
descent (CD) or feature-anchor SPA, each with percentile or absolute pooled-t
screening. Rank, screening amount, and KNN neighbor count remain tunable. The
selected combination will be a practical default for the evaluated workload,
not a guarantee of optimality on every classification problem.

The decision rule below was fixed before this follow-up study ran. Earlier CelebA
development results were already available, so the protocol was not preregistered
before all development evidence. Grid design used the prior training CV records
without access to validation performance values, but the overall study was not
blinded to those earlier results. The completed results and subsequent preference
amendment are recorded in [CelebA default results](celeba-default-results.md).

Retrospective note: the [amended sensitivity results](celeba-default-results.md#amended-screening-preference-rule)
report the later preference rule separately. The October 28
[boundary investigation](celeba-default-results.md#retrospective-boundary-investigation)
adds calculations from saved evidence; those boundaries were not prespecified
in this protocol and do not change its historical instructions.

## Data and comparisons

Use training sizes **2,000, 4,000, and 8,000**, each with seeds **43, 44, and 45**.
For each seed, draw one deterministic ordering of official training image IDs
and use its prefixes for the three sizes. Preserve natural class proportions,
the existing 64 by 64 grayscale preprocessing, and identical images across both
attributes and all four methods. Record IDs, class counts, fingerprints and
the exact sampling implementation. The two attributes are `Smiling` and
`Wearing_Necktie`.

Use the **same original 4,000 official validation images sampled with seed 42**
for every assessment. Do not replace this set after inspecting new results or
balance its labels. Keeping this set fixed supports paired comparisons and avoids
adding a new evaluation allocation during this default-selection study. A full
official-validation assessment is not required for this phase. Official test
images remain unopened until the decision and final evaluation protocol are fixed.

The nine training configurations overlap, the sizes are deliberately nested,
and all assessments share validation images. Treat these as repeated development
configurations, not nine independent datasets or independent validation samples.
Use descriptive paired differences, ranges and rank frequencies; do not calculate
an independence-based standard error from the nine rows. Identity-level separation
has not been established, so image-partition results do not establish generalization
to unseen people.

## Stage 1: finish the training-only refinement

Use the original 4,000 training images, seed 42 and the original five folds. The
source is the sealed `training-follow-up-plan.json` under
`runs/celeba-parameter-study-20261003T091802Z-68d2aa40/cv-review/`, SHA-256
`95092c9959760c69ec2b27e723e74c7ef1326068d1c68643f41ed7588079411b`.

Execute its essential checks and optional fine grids relevant to the four screened
NMF combinations. Retain original incumbents explicitly. PCA, screening-only and
raw comparisons remain part of Wang's broader study; they are outside this four-way
solver/screening default decision. Materialize and hash the deduplicated grids
before fitting. Do not change them using validation results.

All new numerical phases use `max_iter=20000` and `tol=1e-4`. This increases the
iteration allowance from 10,000; it does not change the equations, initialization,
float32 CD coordinates, SPA conventions or convergence tolerance. Preserve earlier
failures. Only candidates with every required fit and projection qualified can
win. Historical 10,000-cap results guide the grid but are not relabeled as new
20,000-cap results; explicitly refit the retained incumbents in the new phase.

For each attribute and method, rank **distinct representation pairs** (rank plus
screening value) by their best qualified five-fold mean balanced accuracy across
KNN values. Use descending unrounded score, then the frozen canonical candidate
order for exact ties. Report near ties within `1e-12` without changing selection.
Choose exactly three pairs: the original incumbent representation, if it remains
qualified, plus the two highest-ranked other pairs. If the incumbent is unqualified,
retain its failure record and take the three highest-ranked qualified pairs.
If fewer than three qualify, record the shortfall and resolve it using training-only
evidence in a versioned amendment before Stage 2; do not inspect new validation scores.

The numerical and boundary checks prevent an unresolved first-grid endpoint from
being mistaken for completed tuning. The three-pair shortlist is a declared finite
search budget; it is not evidence that the global optimum has been found.

## Stage 2: paired tuning on the nine training configurations

Transfer those same three pairs to each training size. Percentiles and ranks remain
unchanged. For absolute screening, multiply each Stage 1 cutoff by
`sqrt(n_train / 4000)` and write the resulting literal floating-point values into
the grid manifest. The factors are approximately 0.707107, 1 and 1.414214. Use only
these size-adjusted values, retaining the same three-representation budget for all
four methods.

This is an explicit search-grid heuristic for the sample-size sensitivity of a
t-statistic, not a change to the pooled-t statistic, a significance threshold,
a theoretical guarantee, or a new estimator default. Do not add the unscaled
cutoff as an extra candidate for only the absolute methods. Record retained-feature
counts and boundaries so inadequate transfer remains visible. A later expansion
requires a new training-only protocol version; this phase reports only the
declared search resolution.

Each pair receives the identical KNN grid:
`[1, 2, 3, 5, 9, 15, 25, 41, 65, 81]`.
Use five stratified folds, with the configuration seed, shared across methods within
each attribute. Fit screening, factorization and projection inside each training
fold. Reuse a representation only across KNN values for the identical fold and
parameters. No automatic search expansion or stopping rule is introduced.

Materialize all 72 method/attribute/configuration job grids and hash the complete
manifest before fitting any Stage 2 candidate. Keep declared infeasible tuples as
failed or explicitly preflight-excluded records with reasons; never silently omit
them. Do not replace them with validation-informed alternatives. Every winning
candidate must have five qualified folds. Select its unrounded mean CV balanced
accuracy, with canonical order resolving exact ties, and perform the qualified
full-training fit before assessing the fixed validation sample. Validation must not
choose rank, cutoff/percentile, KNN count or a prediction threshold.

Complete every declared Stage 2 CV job before opening the new validation comparison.
The resulting 18 attribute/configuration cases per method are the inputs to the
single default decision. Original seed-42 validation performance is contextual
history and does not enter that decision's averages.

## Measuring time

Use Python 3.12 with AOCL, one native numerical thread per fit, the same machine and package
versions. The CV pool may use up to eight workers. Preserve unique representation
attempts, successful and failed committed costs, KNN costs, and interruption history.
Summed CV worker wall time is **recorded tuning compute**, not isolated elapsed time
or CPU time. Count shared work once and report unrecoverable interrupted work separately.

After selection, profile each of the 72 fitted configurations **three additional
times**, with one worker, one native thread and no concurrent study pool. Verify
each repeat's predictions against the saved selected model; a warning, numerical
failure or mismatch cannot supply a valid timing. Use a seed-42043 randomized,
recorded order within timing blocks to reduce systematic method-order effects.
Record all three repeats. The primary time for a case is the median of its three
`total_fit_seconds + prediction_seconds` sums, rather than the sum of separate
stage medians. The original fitting timer remains separate.

The workload is **one fit plus one prediction batch of 4,000 images**. Total fit
already includes screening, factorization, training projection and classifier
fitting; do not add its component timers again. Prediction includes the full
screening/transform/classifier path. Report these stage breakdowns and timing spread
alongside the total. Tuning is a secondary selection criterion because parallel
worker wall times are not interchangeable with these serial measurements.

## Deterministic decision rule

The four candidates, in the final deterministic tie order, are:

1. CD with percentile screening.
2. CD with absolute screening.
3. SPA with percentile screening.
4. SPA with absolute screening.

First require a candidate to produce a qualified selected model and evaluation in
**all nine configurations for both attributes**, with three valid profiling repeats
per case. Report any exclusion and all underlying failed attempts. If no candidate
is operationally qualified, this is a numerical/execution defect to resolve under
a recorded amendment; an invalid estimator must not be installed as the default.
This condition is not a statistical-significance requirement.

For each qualified method `m`, let `A[m,a]` be its mean validation balanced accuracy
across the nine configurations for attribute `a`. Let `M[m]` be the equal mean of its
two attribute means. Thus attributes, training sizes and seeds receive equal weight;
neither the common class nor the larger training allocation dominates the average.

Let `M*` be the largest qualified macro mean and `A*[a]` the largest qualified mean
for each attribute. The accuracy-acceptable set consists of methods satisfying
both `M* - M[m] <= 0.01` and `A*[a] - A[m,a] <= 0.02` for each attribute. These are
one- and two-percentage-point practical tolerances, not equivalence tests or
confidence statements. Apply `1e-12` only as numerical comparison slack.

If that set is empty, calculate
`R[m] = max((M* - M[m])/0.01, max_a((A*[a] - A[m,a])/0.02))`.
Retain the method(s) with the smallest `R[m]`, allowing `1e-12` numerical ties.
This is a declared minimum-worst-regret compromise; report every tolerance it
violates rather than describing it as accuracy-equivalent.

For each retained method, normalize its serial cost in each of the 18 cases by the
lowest serial cost among all operationally qualified methods for that case. Let
`G[m]` be the geometric mean of those 18 ratios. Find the smallest `G` in the
accuracy-acceptable set (or the fallback set) and retain methods within **5%** of it.
This band treats small measured timing differences as operationally close; it is
not a statistical test.

If multiple methods remain, choose the one with the lowest geometric mean of
case-normalized recorded CV tuning compute. This secondary quantity includes
committed failed attempts, counts shared fits once, and excludes explicitly
unmeasured interrupted attempts. Compare it as the measured study workload, not
an isolated algorithmic speed claim. Resolve a numerical tie within `1e-12` by
higher `M[m]`, then by the fixed method order above. With at least one qualified
method, this rule returns exactly one solver/screening combination.

Report positive precision, recall, F1, average precision, ROC AUC, ordinary accuracy,
prevalence and confusion counts for every case. These explain consequences, especially
for Wearing_Necktie; they do not silently replace the frozen balanced-accuracy rule.

## Sensitivity, budget and final holdout

Repeat the descriptive selection calculation for macro tolerances
`[0, 0.005, 0.01, 0.02]`, with attribute tolerances twice each macro tolerance, and
prediction-workload multipliers `[0.1, 1, 10, 100]`. For a zero tolerance with an
empty gate, use the minimum maximum **unscaled** macro/attribute regret. Compute
each workload's median directly from `fit + multiplier * prediction` per repeat.
Also show a sensitivity using the recorded parallel tuning compute as an added
workload cost, labeling its different measurement conditions. Sensitivity winners
do not replace the primary winner. Show leave-one-size and leave-one-seed descriptive
checks, per-attribute results, and any instability without significance claims.

Stage 1's four-method subset has a conservative upper bound of 2,080 representation-
fold attempts: 350 relevant essential checks, 1,690 optional fine-grid attempts and
up to 40 explicit incumbent attempts, before deduplication and feasibility accounting.
These overlap; the actual fixed manifest supplies the exact count. This is not a
wall-time estimate and excludes PCA/screening-only follow-ups.

Stage 2 has 2,160 predictive candidates, **1,080 shared representation-fold attempts**
and up to 10,800 candidate-fold KNN evaluations. There are 72 selected final fits and
216 additional serial profiling fits, for **1,368 fit attempts** beyond Stage 1,
before any separately recorded technical retry. Screening checks and bookkeeping
are additional work. Increasing the candidate count, changing the cap/tolerance or
adding samples requires a versioned amendment; a finite budget does not certify
global tuning completion.

Once the rule selects the general default, freeze the choice, source, numerical
settings and evaluation plan before the final official-test assessment. A practical
fixed final assessment is the already declared 8,000-image, seed-43 training
configuration, with all four independently tuned combinations reported once on the
official test partition. Preserve the chosen default even if the final ranking
differs; any later revision starts a new disclosed development cycle. PCA,
screening-only, raw KNN and majority remain necessary contextual comparisons for
the broader classification claims, with their own training-only tuning.

Original and harder simulations provide complementary failure/recovery diagnostics;
they do not become unannounced extra votes in this CelebA decision. Another external
dataset is not required to choose this scoped initial default. Broader predictive
claims need broader evidence, and consistency or latent-factor recovery claims need
their own assumptions and theoretical justification. Neither repeat sampling nor
a favorable default-selection score establishes those claims.

Machine-readable metadata and the freeze receipt are in `tmp/default-selection/`.
