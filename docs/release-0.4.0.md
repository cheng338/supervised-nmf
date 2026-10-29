# supervised-nmf 0.4.0

supervised-nmf 0.4.0 improves prediction with mixed dtypes and extreme numerical
ranges, adds optional rowwise MU projection, and separates prediction diagnostics
from fitted estimator state. SPA with absolute screening and unscaled KNN remains
the default. CD, MU, logistic regression and linear SVM remain available.

## Prediction and numerical changes

- CD and MU accept a float32 query with a float64 fitted basis, or the reverse,
  by projecting in float64. The fitted basis is preserved, and float64 query
  values are not silently narrowed. Matching-dtype inputs keep their existing
  arithmetic. Widening can add query time and temporary storage.
- Qualified float64 MU projections with tiny positive basis tails try reversible
  binary scaling before the existing logarithmic fallback. This preserves the
  fixed-basis Frobenius objective up to a common scale in exact arithmetic.
  Numerical guards retain fallback or explicit range rejection when required.
  Joint factorization and solver tolerances are unchanged.
- `mu_projection="rowwise"` gives each query observation its own MU initializer
  and stopping decision. Use it when predictions need to be insensitive to
  unrelated observations in the query batch. It can cost more than batch MU and
  does not guarantee bitwise invariance across numerical kernels or higher
  accuracy. `mu_projection="batch"` remains the default, with no automatic switch.
  To use the same projection policy for training coordinates, explicitly set
  `training_coordinates="transform"`.

See [MU prediction policies](api.md#prediction-batches) for numerical limits and
selection guidance. Old serialized models retain the batch policy when loaded.

## Prediction diagnostics and migration

Standard predictions no longer create or update the fitted model's
`last_transform_*` attributes. Read per-call diagnostics from
`evaluate_classifier(...)` results instead. Existing legacy fields in saved
models may remain present but do not describe a subsequent standard prediction.

Evaluation, calibration and study reports now associate diagnostics with the
projection being evaluated. They distinguish an unperformed projection from a
failed one, preserve warnings before a later error, and separate projection
success from classifier failure. Stored fit-convergence status remains separate.
Prediction preserves package-owned fitted state; this is not a guarantee that
all third-party counters, warning mechanisms or native thread controls are
safe under arbitrary concurrent use.

See [prediction state and diagnostics](api.md#prediction-state-and-diagnostics)
for migration examples and concurrency limits.

## Transformer interface and reports

`NMFClassifier` now provides the standard `TransformerMixin` interface.
`fit_transform(X, y)` fits the full classifier and then projects `X`; its output
can differ from `training_coordinates_` when the head uses factorization or
anchor coordinates. Classifier identity is preserved. Estimator tags describe
the existing binary-classification, nonnegative-input and required-label
constraints, and invalid-input messages follow scikit-learn conventions.

The historical study documentation clarifies coordinate policies, the amended
screening-preference rule, subject-level uncertainty and the limits of solver
recovery arguments. The standalone
[CelebA reproduction example](../examples/celeba_identity_reproduction/reproduce.py)
replays the stored paired prediction statistics without fitting models or
requiring the image dataset. These reporting changes do not create new held-out
evidence or change the installed defaults.

## Verification

The pre-release source was compared with v0.3.3 on Windows using Python 3.12
with AOCL 5.3 and MKL 2026.2, with at most eight total compute threads. Each
environment passed 1,090 tests and 443 subtests. The standalone reproduction
example passed 38 additional checks in each environment.

The comparison covered ordinary synthetic, CelebA and Fashion-MNIST workloads,
all three solvers and classifier heads, imbalance, high rank, wide inputs,
near-collinearity, different layouts and dtypes, and targeted zero/tiny/extreme
numerical cases. All 3,396 saved array comparisons in the common workload
matrix matched exactly. Independent MU reference checks improved from 73/80
to 80/80 in each environment. Eighteen baseline serialized models per environment
loaded with unchanged predictions.

Across 48 fixed workloads per environment, median paired fit-plus-prediction
ratios were approximately 1.000 in both runtimes. Single-row prediction latency
increased about 1.8-1.9%, or 18-20 microseconds. A qualified tiny-basis MU fixture
ran 69-75 times faster; this is not a general MU speedup. Individual slower
microbenchmarks remain part of the evidence. Repeated, finer memory observations
did not demonstrate persistent large growth in the investigated outliers;
sampled RSS is not an exact allocation bound. Mixed-dtype recovery can require
more memory than the earlier failing operation.

These checks reuse development cohorts and do not constitute a new held-out
study. Python 3.11 or later is required. Chang Cheng's original work remains
All Rights Reserved, with the separate third-party notices preserved.

```shell
python -m pip install --upgrade supervised_nmf-0.4.0-py3-none-any.whl
```
