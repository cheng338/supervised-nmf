# Wang simulation validation: independent repeated training and test draws

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. **All 900 independent draws and 9,930 final evaluations
completed successfully**, with five-fold training-only tuning repeated within
every draw. The study provides reproducible classification and support evidence;
it does **not** establish a general SPA-sNMF advantage over screened PCA or
screening alone. In the prespecified weak and sparse additive examples, SPA was
less accurate than screened PCA. The XOR oracle nearly solved a problem on
which all tested marginal-screening pipelines remained near chance.

The authoritative run is `runs/wang-simulations-blas-20261006`, using the BLAS
implementation verified before this study. The protocol and source snapshot
were frozen before its data were generated. The independent record audit
**passed with zero errors**: all 4,766,400 saved predictions, CV selection
records, source/configuration receipts, aggregate and paired statistics, and
11,757 main/supplemental manifest files were verified. See the
audit receipt (retained locally: `../runs/wang-validation-audit-20261006/simulation-audit.json`).
Earlier partial studies remain preserved and superseded; none of their outcomes
is pooled with this completed run.

## Completed results

There are 30 valid independent draws for every regime and method, including the
separate XOR oracle. No final fit or prediction failed or emitted a warning.
The fixed search attempted 99,960 candidates and 499,800 CV fold fits: 476,785
were valid and 23,015 failed. The failures were entirely feasibility errors:
12,215 empty screens, 7,200 NMF rank/selected-width violations, and 3,600 PCA
rank/selected-width violations. All failed candidates were ineligible; no method
was replaced with a fallback. There were zero CV convergence warnings. Wall time
for the scientific run, aggregation and plots was **1,665.20 seconds (27.75
minutes)** on six workers, with one numerical thread each. Summed calibration
worker wall time was 7,868.14 seconds; this is not the elapsed study time or an
isolated hardware benchmark.

The following six display cases were specified before outcomes. Entries are
**balanced accuracy (%) ± Monte Carlo SE in percentage points**, from 30 draws.
The full 30-regime table (retained locally: `../runs/wang-simulations-blas-20261006/balanced_accuracy_summary.csv`)
also includes CD, raw classifiers, majority prediction and the XOR oracle.

| Prespecified case | SPA KNN | PCA KNN | Screen KNN | SPA logistic | PCA logistic | Screen logistic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak additive | 61.49 ± 1.52 | 67.65 ± 1.32 | 67.24 ± 1.22 | 69.98 ± 1.29 | 74.55 ± 0.71 | 75.25 ± 0.66 |
| Strong additive | 93.35 ± 2.46 | 97.26 ± 1.65 | 99.02 ± 0.52 | 96.36 ± 2.45 | 97.24 ± 1.92 | 100.00 ± 0.00 |
| Correlated additive | 71.25 ± 0.68 | 71.03 ± 0.67 | 70.90 ± 0.60 | 73.74 ± 0.47 | 74.24 ± 0.37 | 73.76 ± 0.42 |
| Sparse additive | 77.72 ± 2.12 | 80.62 ± 1.72 | 80.19 ± 1.59 | 85.93 ± 1.38 | 87.16 ± 1.31 | 88.54 ± 0.23 |
| Null | 49.55 ± 0.31 | 49.84 ± 0.44 | 49.58 ± 0.37 | 49.83 ± 0.40 | 49.71 ± 0.49 | 50.08 ± 0.44 |
| XOR | 49.75 ± 0.39 | 49.84 ± 0.45 | 50.15 ± 0.40 | 50.15 ± 0.38 | 49.74 ± 0.42 | 50.43 ± 0.39 |

Weak: nominal SNR 0.5, 16 signal features, correlation 0. Strong: SNR 2.5,
16 signals, correlation 0. Correlated: SNR 1.25, 16 signals, correlation 0.9.
Sparse: SNR 1.25, four signals, correlation 0. The null and XOR are separate
stress regimes. All have 160 training rows, 480 test rows and 320 features.

Figure: Prespecified display cases (retained locally: `../runs/wang-simulations-blas-20261006/balanced_accuracy.png`)

The paired estimates below are **percentage-point differences [pointwise 95%
t interval]**, each using all 30 common draws. They compare independently tuned
pipelines with the same screening rule and candidate ranges, not a common-mask
ablation. The weak-case SPA KNN deficit against PCA is 6.160 points, with an
interval from −7.839 to −4.481; its corresponding logistic deficit is 4.569
points, from −6.749 to −2.390. The correlated display case gives no clear
SPA-versus-PCA difference; this does not prove equivalence.

| Prespecified case | SPA KNN minus PCA KNN | SPA KNN minus screening KNN |
| --- | ---: | ---: |
| Weak additive | -6.160 [-7.839, -4.481] | -5.757 [-7.678, -3.836] |
| Strong additive | -3.917 [-7.829, -0.004] | -5.674 [-9.947, -1.400] |
| Correlated additive | +0.215 [-0.898, 1.329] | +0.354 [-0.998, 1.707] |
| Sparse additive | -2.896 [-4.516, -1.275] | -2.472 [-3.921, -1.024] |
| Null | -0.292 [-1.329, 0.746] | -0.035 [-1.015, 0.945] |
| XOR | -0.089 [-1.136, 0.959] | -0.399 [-1.591, 0.793] |

| Prespecified case | SPA logistic minus PCA logistic | SPA logistic minus screening logistic |
| --- | ---: | ---: |
| Weak additive | -4.569 [-6.749, -2.390] | -5.271 [-7.615, -2.927] |
| Strong additive | -0.875 [-2.167, 0.417] | -3.639 [-8.656, 1.378] |
| Correlated additive | -0.507 [-1.188, 0.174] | -0.021 [-0.797, 0.755] |
| Sparse additive | -1.229 [-2.294, -0.164] | -2.611 [-5.393, 0.171] |
| Null | +0.118 [-0.743, 0.979] | -0.250 [-1.421, 0.921] |
| XOR | +0.414 [-0.481, 1.309] | -0.280 [-1.302, 0.742] |

Figure: Paired SPA comparisons (retained locally: `../runs/wang-simulations-blas-20261006/paired_spa_contrasts.png`)

All these intervals are **approximate, untruncated Student t intervals**.
Some individual-method intervals extend above one at the strong-signal ceiling;
that is an approximation artifact, not a feasible accuracy. Zero empirical MC
SE at a ceiling does not establish zero population error. Across many tested
cells, pointwise intervals have no simultaneous error control, and the strong
KNN/PCA interval's upper endpoint is particularly close to zero. No universal
superiority or equivalence conclusion is warranted.

## Instability, null and XOR diagnostics

The strong display case has substantial repeat-to-repeat variation despite its
high mean. SPA KNN ranged from **0.4729 to 1.0000**, with 3/30 draws below 0.75
and 5/30 below 0.90. SPA logistic ranged from **0.40625 to 1.0000**, with 2/30
below 0.75. These were valid converged fits, not solver failures. Their screening
masks retained all 16 signal features plus two to six background features, at
cutoff 3 and rank 2 or 4. Replication 8 retained six backgrounds and had low
test accuracy across SPA, PCA and CD; replication 24 gave SPA logistic test BA
0.40625 after selected training CV BA 1.0, while retaining two backgrounds.
This demonstrates that full screening retention and favorable CV scores do not
ensure a stable small-rank representation on a fresh draw. The observed
co-occurrence is not a controlled causal diagnosis of which stage caused the
loss. No scientific result was replaced or retuned to repair these outcomes;
all are included in the reported means and intervals. The saved
diagnostic records (retained locally: `../runs/wang-simulations-blas-20261006/strong_display_diagnostics.json`)
make the ranges and low-score cases reviewable.

A separately requested **exploratory numerical equivalence check** subsequently
refitted the seven already-selected low-score SPA configurations (six draws)
once with the current implementation and once with the saved pre-BLAS
implementation. All 14 fits used the exact original parameters and seeds; there
was no CV or new parameter search. Anchors, basis, training coordinates, fixed-
basis training/test projections, labels and probabilities matched **bitwise**
between implementations, and the predictions matched the original scientific
records. No warnings occurred. This rules out a BLAS-induced difference for
these checked fits; it does not establish a universal numerical guarantee or
identify the cause of representation loss. The
separate check (retained locally: `../runs/blas-spa-20261006/strong-case-equivalence/result.json`) and
hashed protocol (retained locally: `../runs/blas-spa-20261006/strong-case-equivalence/protocol.json`)
are outside the scientific run and replace none of its results.

Every non-oracle null and XOR method had mean balanced accuracy near 0.5. The
XOR **known-support KNN oracle reached 0.998735 (MC SE 0.000258)**, with a paired
advantage over SPA KNN of **0.501232 [0.493212, 0.509251]**. Predictive information
is present in the interaction case, but the tested marginal-screening pipelines
do not recover useful classification. The oracle uses information unavailable
to a practical feature-selection method and is a diagnostic, not a deployable
competitor. No theorem of universal consistency follows from these data.

## Generating-support results

For the weak display case, SPA KNN retained a mean 11.1 features, with generating-
support precision **0.881** and recall **0.567**; full support was retained in
**0/30** draws. SPA logistic precision/recall were **0.910/0.538**, also 0/30.
For the strong, correlated and sparse display cases, SPA retained the full
generating support in 30/30 draws for both classifiers; this did not prevent the
strong-case predictive failures above. A Wilson 95% interval is **[0, 0.114]**
for 0/30 and **[0.886, 1]** for 30/30, rather than certainty at either boundary.

In XOR, SPA KNN recall was **0.173** and precision **0.053**, with 0/30 full-support
inclusions; the logistic values were **0.167/0.039**, also 0/30. The null has no
predictive shift support, so its recall and full-support inclusion are undefined.
The complete support table (retained locally: `../runs/wang-simulations-blas-20261006/support_summary.csv`)
and Wilson inclusion table (retained locally: `../runs/wang-simulations-blas-20261006/full_support_wilson.csv`)
retain all methods, denominators and conditions.

Figure: Generating support across all conditions (retained locally: `../runs/wang-simulations-blas-20261006/generating_support.png`)

Figure: All 27 factorial cells (retained locally: `../runs/wang-simulations-blas-20261006/factorial_balanced_accuracy.png`)

## Frozen design and scope

The frozen protocol crosses all 27 combinations of nominal signal-to-noise
ratio 0.5, 1.25 and 2.5; generating support 4, 16 and 40; and within-block
Gaussian correlation 0, 0.5 and 0.9. Cross-block correlation, a null, and an XOR
interaction case add three stress regimes. Each specifies 30 independent
training/test draws, with 160 training rows, 480 test rows and 320 features.
These dimensions and support counts differ explicitly from the earlier proposed
study and historical reproduction. They preserve a p > n setting, not its scale.

Eleven methods compare absolute screening with SPA, CD and PCA, screening alone,
and raw features, each followed by KNN or logistic regression, plus a majority
classifier. XOR adds a separately labelled known-support KNN oracle. The modest
grids include absolute pooled-t cutoffs 1, 2 and 3, ranks 2 and 4, KNN neighbor
counts 3 and 9, and logistic inverse penalties 0.1, 1 and 10. They require further
range/refinement sensitivity and do not establish globally optimal tuning.

- The original sensitivity protocol calibrates each regime on a separate pilot
  with five-fold training-only CV, then freezes settings. **674 of 900 draws**
  and all 331 pilot model searches were saved when paused.
- The separate main-study extension uses fresh seed domains and repeats all
  five-fold training-only model selection within each independent draw. **34
  of 900 draws**, with 400 completed model CV searches, were saved when paused.
- Both processes have stopped. Every completed record retains parameters,
  seeds, source hashes, warnings, failure status, feature support diagnostics
  and individual predictions. Atomic writes preserve completed records.
- No hidden fallback replaces a failed method. Planned mean, Monte Carlo SE
  and paired t intervals count independent draws; invalid-result exclusions
  and denominators are explicit. A null has no true predictive support; its
  support recall and full-support inclusion rate are undefined.

The superseded protocols, code snapshots and records are in
`runs/wang-simulations-20261006`, with the repeated-CV extension in its `retuned`
subdirectory. `pause-status.json` records the stopped sessions and counts.
`scripts/validate_wang_simulations.py` contains the reproducible runners; the
initial sensitivity study retains its exact earlier source snapshot because
the repeated-CV option was subsequently added to the runner.

No settings were changed in response to replication outcomes. The successor
uses a new seed offset of 40,000,000, disjoint from the stopped protocols. All
900 draws repeated five-fold training-only tuning independently; six workers
used one numerical thread each. Completed totals are 9,930 model evaluations,
99,960 parameter candidates, and 499,800 CV fold fits. The package uses public SciPy
NNLS with a C-contiguous design and BLAS residual updates; the optimization
audit did not compute classification outcomes.

## Prespecified analysis

The primary measure is test balanced accuracy. Each cell reports its mean,
Monte Carlo standard error (sample SD divided by the square root of independent
completed draws), and a two-sided pointwise 95% Student t interval. SPA-minus-
baseline differences are computed within the same draw before constructing
their intervals. The prespecified comparisons are screened PCA, screening
alone, raw features, CD, and majority prediction, separately for KNN and
logistic regression; SPA logistic versus SPA KNN is also reported. XOR adds
oracle KNN versus SPA KNN. The intervals have no multiple-comparison correction
and do not establish universal superiority or equivalence.

All transformations, screening and classifier parameters are fitted within
each training CV fold. The selected full-training fit precedes test generation
and evaluation. Logistic regression standardizes its representation using
training-only moments; KNN retains the package's unscaled Euclidean geometry.
SPA and CD retain the package's `fit_transform` training-coordinate convention.
Each full method tunes its own screening cutoff/rank/classifier parameters;
comparisons assess the pipelines and are not a common-mask causal ablation of
factorization alone. Candidates failing any fold or emitting convergence
warnings cannot win. Full-fit or prediction failures are saved without a
majority fallback. Valid-only summaries disclose the number excluded.

Additive generating signal features are `max(0, 1 + delta*y + 0.8*noise)`;
background features are `max(0, 3 + 9*noise)`. Gaussian noise correlation is
specified before clipping, separately within the signal and background blocks
in the factorial design. The cross-block stress case uses one shared Gaussian
factor instead. Nominal SNR is `delta/0.8`; it is not an observed standardized
effect. Every draw saves empirical post-clipping class-specific block means,
average feature variances, and zero fractions. XOR uses two independent fair
bits whose exclusive-or is the label, with separate half-support blocks shifted
by each bit; individual features have no population marginal label association.

Support summaries concern the **designated generating/shift support**. In the
cross-block case, background features can be jointly useful for denoising even
though they receive no label shift. Counting them as outside generating support
does not establish that they contain no conditional predictive information.
The saved field `true_predictive_support_indices` is a legacy name for this
generating-support mask; the null mask is empty. Full-support inclusion is an
empirical frequency, not a support-recovery guarantee. These statistics measure
the screening mask, not SPA anchor identification or latent-factor recovery.
Raw-feature methods retain every feature, so their complete retention is
trivial and is not successful feature identification.

These generators do not impose exact low-rank NMF, feature separability, or the
full assumptions of the representation/support theorems. They test the fitted
classification procedures under specified stress conditions. A reporting-only
supplement adds Wilson 95% intervals and exact inclusion counts for the binary
full-support event, because an empirical MC SE of zero at 0/30 or 30/30 does not
mean the underlying inclusion probability is known. This supplement reads saved
draws, preserves the original t-interval tables, and changes no fitted models.

## Reproduction

The historical local workflow used the script and arguments below. The script
and original machine-specific invocation are retained locally, outside the
public distribution. This is a configuration record, not a runnable example
provided by this checkout.

```text
scripts/validate_wang_simulations.py --retune --seed-offset 40000000
  --study-id wang-simulations-blas-20261006
  --output runs/wang-simulations-blas-20261006
```

The runner refuses incompatible source/software/configuration resumes.
`protocol.json`, `source_snapshot/`, `provenance.json`, per-draw selected-setting
receipts, complete CV logs and prediction records allow independent verification.
This reduced-dimensional factorial study addresses replication and simulation
stress testing, while historical reproduction, much larger dimensions, grid
refinement, and universal theoretical guarantees remain distinct work.
