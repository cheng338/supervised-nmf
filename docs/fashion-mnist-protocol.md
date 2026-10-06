# External solver study with absolute screening fixed

Author: Chang Cheng. October 3, 2026.
Fashion-MNIST: Han Xiao, Kashif Rasul and Roland Vollgraf.

October 4 numerical amendment: a reproducible native least-squares crash in
ANLS-BPP requires the [documented repair](numerical-repair-20261004.md),
[Fashion-MNIST v3](../examples/fashion_mnist_solver_protocol_v3.json) and
[CelebA v2](../examples/celeba_four_solver_protocol_v2.json). The grids and
decision rule below are unchanged. The original source freeze and results
remain preserved; ANLS-BPP is rerun under the amended implementation.

MU and ANLS-BPP were added to the protocol before any Fashion-MNIST fits.
This study compares CD/HALS, feature-anchor SPA, multiplicative updates
(MU), and alternating nonnegative least squares with block principal pivoting
(ANLS-BPP), with **absolute pooled-t screening fixed**. The cutoff value, rank and KNN
neighbors are tuned independently for each method using training folds only.
Existing CD/SPA equations, initialization, dtype and coefficient convention stay
as previously tested. MU and ANLS-BPP are explicitly new experimental algorithms,
using the Frobenius objective and NNDSVDa initialization. Each classifier uses
its original fitted training coordinates and fixed-basis out-of-sample projection.
The comparison completed on October 5, 2026. Its
[report](nmf-solver-comparison.md#development-results) selects SPA under the
frozen rule and remains `selected_for_audit`. The protocol below records the
study design; official-test evaluation is separate.

## Dataset and tasks

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) provides 60,000
training and 10,000 test images, each 28 by 28 pixels, with ten balanced classes.
Its [original paper](https://arxiv.org/abs/1708.07747) presents it as a machine
learning benchmark. It is an independent clothing-image domain and directly
nonnegative; it is still image data and does not establish performance on text,
omics or all classification problems.

Three binary subsets are declared before our Fashion-MNIST modeling outcomes:

| Task | Negative class | Positive class |
| --- | --- | --- |
| T-shirt versus shirt | 0, T-shirt/top | 6, shirt |
| Pullover versus coat | 2, pullover | 4, coat |
| Sneaker versus ankle boot | 7, sneaker | 9, ankle boot |

These are chosen to cover related upper-body clothing and footwear without
selecting pairs from our results. They are binary subsets created for this
package, not the standard ten-class benchmark. All six labels are distinct.
Each pair has 12,000 official training rows. A fixed stratified 2,000-row
development validation set uses seed 104729. From the remaining 10,000 rows,
seeds 43, 44 and 45 each produce nested balanced samples of 2,000 and 8,000.
The same rows and five stratified folds are used for all four methods. Repeated
samples overlap; the resulting cases are not independent datasets.

Only official training archives are downloaded. Published MD5 checksums are
verified; SHA-256 hashes bind archives, decoded arrays, manifests and selected
row indices. Native pixels become float32 values divided by 255, flattened in
row-major order. There is no centering, learned scaling, resizing or augmentation.
The official test files are neither downloaded nor evaluated by this runner.
Class balance is preserved at 50/50; CelebA retains its natural imbalance.

## Frozen computation

The executable specification is
[fashion_mnist_solver_protocol_v2.json](../examples/fashion_mnist_solver_protocol_v2.json).
The earlier two-solver v1 is preserved but was superseded before any fits.
Version 2 declares 72 NMF jobs: three tasks, two sizes, three seeds and four solvers.
Each starts with the same Cartesian grid:

- Rank: 2, 5, 10, 20, 40, 80.
- Absolute pooled-t cutoff: 0, 1, 2, 4, 8, 16, 32.
- KNN neighbors: 1, 3, 5, 9, 15, 25, 41.

This is 294 candidates and 210 representation-fold fits per job, or 21,168
candidates and 15,120 representation-fold fits over all NMF jobs, before
dimensional exclusions. A training-fold-only preflight records selected-feature
counts and excludes a rank/cutoff pair only when it cannot be fitted in every
fold. Exclusions and numerical failures remain in the audit. Each representation
is fitted once per fold and shared across neighbor counts. Selection requires
five valid folds; exact mean-balanced-accuracy ties use declared rank, threshold,
then neighbor order. Convergence warnings disqualify candidates. There is no
substitution after a selected fit fails.

Screened PCA, screening-only KNN and raw KNN are independently tuned on all three
8,000-row seed-43 tasks as contextual baselines. Their grids use the corresponding
rank, cutoff and neighbor values above. They do not enter the NMF solver decision.
No linear classifier is introduced.

The run uses conda `py12_df_aocl`, up to eight worker processes, and one native
numerical thread per fit. CD and SPA retain the existing 20,000-iteration
allowance; CD retains tolerance 1e-4. MU and ANLS-BPP configuration and their
inner/outer stopping rules are bound in the executable protocol and source freeze.
MU uses the same 20,000-iteration cap and nominal tolerance 1e-4. ANLS-BPP uses
float64 arithmetic, up to 1,000 outer NNLS sweeps with relative projected-gradient
tolerance 1e-4, and at most `max(100, 5 * rank)` inner block-pivot iterations with
NNLS relative KKT tolerance 1e-10 plus an explicit roundoff allowance of
`256 * float64_epsilon * max(NNLS design shape)`. A full ANLS sweep solves both NNLS subproblems;
it is not the same amount of work as one CD or MU iteration. Its factors are not
rescaled for classification. These are explicit experimental-method choices.
These are method-specific stopping rules,
not a claim of equal optimization error. All CV jobs finish before validation
evaluation. Boundary selections are reported. This fixed-budget comparison does
not establish globally optimal tuning; any later grid change requires a new
versioned protocol and disclosure of outcomes already seen.

After both datasets' CV phases, the serial phase fits each selected setting and
records validation balanced accuracy, ordinary accuracy, AUC, average precision,
class recalls and confusion counts. Three additional serial profiles in seeded
random order must reproduce labels and probabilities exactly. Tuning wall time
and aggregate compute, MF/PCA time, KNN fit time, total fit time, projection and
prediction time are retained separately. Nested component times are not added
again to totals. The main cost is the median of three fit-plus-prediction sums.
No numerical workload is deliberately run alongside these serial profiles.
Common reconstruction and projected-gradient diagnostics are descriptive: SPA
does not optimize unrestricted NMF factors, and a small reconstruction residual
does not by itself establish superior classification or latent-factor recovery.

## Matched CelebA extension

The [matched CelebA protocol](../examples/celeba_four_solver_protocol.json) adds
eight jobs: all four solvers on Smiling and Wearing_Necktie, using the existing
8,000-row seed-43 training sample and 4,000-row development validation set. All
four receive the same six-rank/seven-cutoff grid, augmented with the same two
previously CV-selected representation pairs per attribute. Neighbors are
1, 2, 3, 5, 9, 15, 25, 41, 65 and 81, retaining previous settings. This declares
3,520 candidates and 1,760 representation-fold fits before exclusions. Natural
imbalance is preserved. There are no new official-test evaluations.

The existing nine-configuration CD/SPA study is retained as historical context
and sensitivity evidence. The matched eight-job extension supplies the primary
CelebA contribution for the four-method decision; missing MU/BPP cases are never
imputed into the old study. Only one CelebA training configuration is repeated
here, which limits the four-method comparison's stability evidence.

## Prospective decision rule

CelebA outcomes were known before this protocol; Fashion-MNIST outcomes were not.
The old time-first decision and later author-preference amendment remain intact.
This is a prospective external-data check, not a claim that the entire project
was preregistered before all evidence.

All four methods with absolute screening are candidates. Every selected NMF case must
qualify numerically and reproduce all three profiles. Aggregate balanced accuracy
gives each dataset half the weight, then weights tasks and size/seed cases equally
inside that dataset. Thus extra repeated cases cannot make one dataset dominate.
An eligible method passes if its regret is at most one percentage point globally,
one point within each dataset, and two points within each task, relative to the
best eligible absolute-screening solver at the corresponding level. Comparisons
use absolute numerical slack 1e-12. A method with missing or failed selected
cases is ineligible; complete valid cases may still be reported without fallback.

Among methods passing those gates, compare the weighted geometric mean of
per-case cost divided by the fastest eligible solver's cost on that case. Workload
is one fit and prediction on the declared batch: 4,000 for CelebA and 2,000 for
Fashion-MNIST. Among methods within 5% of the minimum cost, prefer higher weighted
balanced accuracy, then lower cost, then the fixed order CD, SPA, MU, ANLS-BPP
for an exact tie. If no method passes
the accuracy gates, minimize the maximum normalized regret, then cost, and report
the unresolved conflict. If no solver qualifies numerically, do not finalize.

Sensitivity analyses vary accuracy margins, prediction workload, inclusion of
tuning compute and omission of a dataset, task or seed as specified in JSON.
They are descriptive, not a basis for silently replacing the primary rule. The
[theory review](theory-implementation.md) continues to apply conditionally to
the fitted representation maps; observed classification accuracy or convergence does not prove
latent-factor recovery or consistency.

MU and ANLS-BPP were experimental implementations in the frozen comparison.
MU entered the public API on October 5, 2026, after separate numerical and
distribution checks; ANLS-BPP remains experimental. That API extension does not
alter the frozen comparison or the earlier CD/SPA experiments.
