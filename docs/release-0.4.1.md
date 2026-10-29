# supervised-nmf 0.4.1

Version 0.4.1 adds a temporary AOCL-specific workaround for a native matrix
multiplication defect encountered during SPA NNLS fitting. It also documents the
completed current-package classification comparisons and makes training-only
tuning guidance more prominent. SPA, absolute screening and unscaled KNN remain
the defaults. The published [0.4.0 release](release-0.4.0.md) and its validation
records retain their original scope.

## AOCL-specific correction

When the loaded numerical runtime is identified as AOCL, the package pads the
outer dimensions of its eight explicit NNLS matrix products to multiples of
eight with zeros, leaves the inner dimension unchanged, and crops the result
back to its original shape. This avoids the observed double-precision 5×3 fringe
overread in the inspected AOCL path. MKL-only environments retain direct NumPy
multiplication; other loaded libraries do not override an identified AOCL runtime.
No AOCL source, installed binary, dependency or estimator option is changed.

The original product and NNLS objective are unchanged in exact arithmetic.
Floating-point rounding can differ, so numerical qualification and the existing
SciPy fallback remain in place. The fix covers explicit products in the package
NNLS helper, including QR and fallback paths; it does not replace all internal
NumPy/SciPy operations. Recorded tests do not guarantee every native kernel,
architecture, build or thread schedule.

This is a correctness repair, not a memory optimization. In the bounded local
integration comparison, ratios of summed case medians were 1.057 and 1.049 for
AOCL 5.2 and 5.3, respectively, and 0.992 for MKL. These aggregates are not an
overhead bound for every shape. The earlier captured-workload comparison found
about 11–12% overhead. MKL used no padding-helper calls and retained bitwise
identical results in all nine recorded numerical comparisons.

Padding is temporary compatibility code, not part of the NMF methodology.
The [retirement policy](aocl-nnls-workaround.md#retirement-policy) requires
identifying and testing AMD's corrected distributed builds before exempting
them, then removing the helper when supported runtimes no longer need it.
No fixed AOCL build has yet been qualified in this package's recorded evidence.

## Training guidance and current-package evidence

Calling `fit` does not tune rank, screening cutoff or the classifier. Constructor
values are starting settings. Use [training-only tuning](../README.md#training-only-tuning),
fit preprocessing inside each fold, retain a separate final evaluation set,
and investigate feasible boundary winners and nearby settings. The new study
does not introduce an automatic search or select new defaults.

The [October 29 comparison](current-package-evidence-20261029.md) qualified all
34 final evaluations: 17 frozen task/configuration arms fitted in AOCL and MKL.
On Smiling, tuned SPA reached 81.35% balanced accuracy versus screened PCA's
80.10%, a difference of +1.24 percentage points with a paired 95% interval of
−1.00 to +3.47. The interval includes zero. On Wearing_Necktie, the corresponding
balanced accuracies were 62.98% and 65.44%. The literal constructor reached
54.41% and 50.44%, respectively, and detected only 1/76 Necktie positives despite
92.1% ordinary accuracy. These outcomes support explicit calibration guidance,
not a claim that SPA generally outperforms PCA or that the untuned defaults
are suitable for every dataset.

Only crashed work was continued: seven completed selections and completed
Smiling/SPA fold cells retain their pre-fix identities. Calibration therefore
mixes pre-fix evidence and post-fix recovery; all final fits used the frozen
post-fix implementation. Every arm had identical predicted labels across AOCL
and MKL, although some internal numerical arrays differed.

The cohort was previously exposed. Inference is conditional on the fitted
models, and does not account for calibration uncertainty or fresh populations.
Four historical native failures remain recorded; one earlier PCA controller
interruption still lacks complete process observations and duration. The
scientific outcomes are available, while the complete execution-history flag
remains false. The study is not fresh independent confirmation.

## Verification scope

Before release preparation, the integrated workaround passed **155 focused tests
and 113 subtests in each of AOCL 5.2, AOCL 5.3 and MKL 2026.2**, seven NNLS cases
and two complete SPA fit/query comparisons per environment, and 30 guarded
product checks across native thread budgets of one and twenty. Separately, ten
study-helper suites passed **263 tests in each AOCL/MKL environment** before the
classification study's final freeze. Its pre-query audit and independent
prediction-statistics pass qualified all planned scientific outcomes.

These are completed local checks with their recorded sources and workloads.
They are separate from the earlier 0.4.0 comparison and from the GitHub-hosted
release build and artifact-verification batch. No GitHub CI outcome is claimed
by these counts.

Python 3.11 or later is required. Chang Cheng's original work remains All Rights
Reserved, with the separate third-party notices preserved.
