# Projection and numerical implementation

Author: Chang Cheng.

Version 0.2.0 adds guarded batched NNLS projection, optional reusable
process workers, separate projection/native-thread limits, and a fitted
binary decision threshold. The default projection setting is `auto` with one
worker; an explicit `scipy` setting and legacy serialized models retain the
original solver. See the [API](api.md#spa-projection-execution) for guards,
fallback, serialization and threshold semantics. Joblib is now a direct
runtime dependency.

In version 0.2.0, SPA anchor selection and basis NNLS retained their numerical operations.
After the anchor condition checks, `FeatureSPA.fit` releases the scaled
matrix, residual, directions and loop-array references before allocating
training coordinates. An anchor-count guard avoids a `locals()` snapshot
retaining the previous direction on Python 3.12.

## Validation

The full source suite in a clean checkout passed all 370 tests on both
Windows Python 3.12 and Python 3.10. This includes 155 package regression
tests covering guarded fallback, coefficient and
prediction agreement, errors/refits, input immutability, legacy serialization,
process reuse, thresholds and workspace lifetime. The two lifetime tests
cover 16 rank/layout/precision/normalization combinations, and fail against
the old implementation as expected.

The source wheel also passed an isolated installation and import check. It
contains both new projection modules, their third-party notice and the direct
joblib dependency. Batched projection and stored-threshold smoke checks passed.

Before the final workspace change, the projection and threshold source also
passed 153 regression tests on Linux Python 3.12/OpenBLAS. Saved-model replay
tested serial SciPy and auto with 1, 8 and 40 workers on six SPA models.
All 24 configurations retained the original decisions, with coefficients and
probabilities inside the declared tolerances. Separate untimed worker traces
verified the requested process counts. Forty workers were not consistently
faster than eight; pool startup and repeated-call timings must be distinguished.

An isolated serial memory experiment used three fresh processes per variant
on each of two workloads, with one numerical thread. Whole-process median
peak memory fell from 2.772 to 2.243 GB on 16000 by 4096 synthetic data at rank
4, and from 0.807 to 0.671 GB on 4000 by 4096 training-image features at rank
20. Numerical outputs matched exactly and fit times changed little. The
17–19% memory reductions apply to those workloads; they are not whole-CV
speed or memory guarantees. No classification-accuracy gain is attributed
to these implementation changes.

The completed study evidence keeps its original source fingerprints. KNN
normalization and the training-coordinate convention retain their defaults;
the exploratory normalization results did not support a universal change.

The projection and threshold changes above are included in formal release
0.2.0; the historical 0.1.0 release tag remains unchanged.

Run the current source tests with `python scripts/check.py` after installing
the test and study dependencies with `python -m pip install -e ".[study,test]"`.
The launcher uses pytest.

## October 20 source changes

The October 20 integration, subsequently released in version 0.3.0, repairs
offset-sensitive screening, extreme-range fixed-basis NNLS and factor-recovery
arithmetic. It adds guarded KNN query reuse, optional fitted SD KNN scaling and
an optional linear SVM head, together with bounded reconstruction reporting.

The integration checks used a budget of 20 total compute threads. Their scope
and results are summarized in the [0.3.0 release notes](release-0.3.0.md) and
[validation record](validation.md). They do not replace earlier accuracy,
timing or compatibility records.
Python 3.10 results above remain historical; current source requires Python 3.11
or later. See [the protocol](development-protocol.md#october-20-implementation)
for the mathematical scope and adoption requirements.

## SPA basis fitting in version 0.3.3

Basis fitting now reuses the guarded multiple-right-hand-side NNLS kernel
after SPA selects its anchors. A small internal wrapper preserves the C-order
basis layout and retries the original public SciPy loop when the helper rejects
an unsupported numerical range. Existing condition and optimality guards remain
in place. The fixed-anchor objective, anchor selection, classifier defaults and
projection controls are unchanged.

Normal equations can lose coefficient precision near the acceleration condition
limit, and batched pivot rounds are not equivalent to SciPy iteration counts.
Fallback retains the original public-solver cap and errors. The
[API description](api.md#spa-basis-fitting) states these limits and distinguishes
basis fitting from query projection. Earlier release and study records retain
their original implementations and results.

## Optional QR projection

`projection_solver="qr"` adds a guarded alternative for fixed-basis SPA
projection, including training reprojection when requested. The default
`"auto"` solver, anchor selection and SPA basis fitting remain unchanged.
The same projection selector is available in `FeatureSPA` and `NMFClassifier`;
it does not add another factorization method or classifier.

For a full-column-rank design `A = Q @ R`, orthogonal reduction separates the
least-squares objective into the reduced problem and a residual independent
of the coefficients. Solving NNLS with `R` and `Q.T @ b` therefore preserves
the original nonnegative coefficient constraint and minimizer in exact
arithmetic. Public SciPy NNLS handles the signed reduced data; the shared
reduction avoids repeating that part of the computation for every right-hand
side. Solver internals and roundoff behavior depend on the SciPy version.

The QR path requires nonnegative original inputs, at least 32 right-hand sides
per submitted scaling group, design height at least twice the rank, and a
finite estimated condition number of `R` no greater than 1e6. It retains the
common reversible scaling wrapper and qualifies results against the original
design and targets. Ineligible inputs or failed QR checks use the current
solver and public SciPy fallback. The original iteration cap and fallback
errors are preserved; a fallback is not charged against a combined iteration
budget. Small explicit caps keep the existing direct SciPy route.

Q/R factors live only within a call, with no persistent estimator cache.
Process-worker chunks are admitted independently; small chunks can use the
current path even when the overall query is large. These choices preserve
the existing cloning, serialization and worker-ownership model.

QR has workload-dependent costs. Sharing a tall reduction can substantially
reduce projection time, while per-row reduced solves, qualification and
fallback can make other workloads slower. Floating-point coefficients can
differ without changing the objective. Neither a condition bound nor a small
residual guarantees unchanged decisions near classifier boundaries. This
explicit algorithm choice does not replace the automatic policy.

## Reconstruction diagnostics

Reported reconstruction norms use 128-row blocks only within the existing
float64 bounds: 8,192–32,768 observations, 1,024–4,096 features and rank 1–32.
The existing one-thread MKL path accepts C/F-contiguous arrays. A Windows
AOCL/BLIS path also accepts C-contiguous arrays when exactly one BLAS pool is
detected, its version is `AOCL-BLAS 5.3.0 Build 20260511`, and it uses one thread.
The input, coordinates and components must all satisfy the layout and dtype
requirements; no conversion copy is made to enter this path.

Other configurations, including unrecognized AOCL builds and mixed BLAS
backends, keep the full-product expression. The block calculation and norm
combination are unchanged. This extension only broadens the qualified runtime
dispatch for diagnostics; it does not alter optimizers, stopping rules, numerical
qualification, fitted factors or predictions. Its bounded timing evidence
does not imply that blocking is faster on other runtimes.

## MU projection of zero batches

For an entirely zero selected query matrix, zero coordinates attain the global
minimum of the unregularized Frobenius projection objective. The package returns
that exact solution directly for the standard fitted MU configuration. This
avoids an undefined relative-error stopping calculation that can otherwise run
to the iteration limit, report nonconvergence or raise under strict floating-point
settings. The basis and preceding fit diagnostics are unchanged.

Full input validation precedes the shortcut. Unsupported or altered decomposition
states retain the original transform call, including its errors and warnings.
No tolerance is used to classify a batch as zero. This implementation correction
preserves the projection objective, final classifier and package defaults.

## MU projection near underflow

Positive values can disappear during ordinary MU initialization or matrix
products, even in a batch containing ordinary-scale observations. At the same
time, a fitted float32 basis can have tiny positive tails on otherwise ordinary
data. Sending every such basis or arithmetic failure directly to logarithmic
MU caused substantial prediction overhead during development.

Version 0.3.3 first retries eligible float32 projections with float64 copies of
the query and basis on a temporary estimator. The original estimator and its
arrays remain unchanged. Internal underflow, overflow, division by zero and
invalid arithmetic remain errors during this retry. Finite nonnegative
coordinates within the output range are converted back to the query dtype only
after solving; final rounding can produce zero below that dtype's range.

Tiny positive query entries use logarithmic arithmetic directly. Tiny float64
basis entries, float64 arithmetic failures and failed precision retries also
use the logarithmic calculation. In recovery, rank-one projections use the
closed-form first MU update, with exact stored-input ratios resolving subnormal
rounding ties. Unrepresentably large coordinates raise an error.

Both recovery paths preserve the Frobenius objective, fitted basis, uniform
initialization formula and relative-improvement stopping rule. They do not
change joint factorization or introduce a new public solver. An explicit
training reprojection uses the same correction as prediction. Ordinary supported
inputs retain the original calculation. Recovery can still cost more time and
change rounding or stopping iterations; positive-tolerance cap exhaustion
retains its convergence warning. These corrections address numerical range,
and their adoption does not rely on a memory saving.

## Evaluation summaries and tests

Version 0.3.3 reports per-class precision, recall and evaluation counts alongside
training counts when supplied. Missing observations remain explicit, including
when prediction fails. Tuning, final fitting and prediction retain their own
timing fields; the fitted estimator's total already includes its component
stages and must not be added to them.

The source launcher runs the suite under pytest 9 or later, including existing
test cases and subtests. New regression tests cover MU range recovery, guarded
SPA basis fitting, optional QR projection and reporting behavior. See the
[0.3.3 release notes](release-0.3.3.md) for release verification.
