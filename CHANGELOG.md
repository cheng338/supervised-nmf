# Changelog

This package originated in Chang Cheng's supervised NMF research scripts
accompanying his dissertation. This file summarizes published releases and
subsequent development changes; detailed release notes and historical study
records retain their original scope.

Release dates are GitHub publication dates in America/Indianapolis time.

## 0.4.2 — 2026-11-02

See the [release notes](docs/release-0.4.2.md) for compatibility and validation details.

- Reject mismatched numeric/string identifier types between training and
  evaluation sample IDs or subject IDs before prediction, so overlap checks
  cannot silently miss differently represented identifiers. Integer and
  floating-point identifiers remain compatible.
- Include the fitted decision threshold, positive label, positive-class index
  and MU projection policy in evaluation diagnostics when available. Reports
  use fitted choices rather than later constructor-parameter changes.
- Correct tuning-refinement reports to exclude Python and NumPy booleans from
  numeric ranges and record a declared `None` alternative with `includes_none`.
- Correct the CelebA stage profiler to expect training reprojection for SPA
  with `training_coordinates="auto"`, including when that stage was not reached.
- Extend the selected backend's matrix-product policy to explicit products in
  the experimental ANLS-BPP solver and its diagnostics. AOCL uses protected
  products; MKL retains direct multiplication. This does not repair or replace
  installed native libraries.
- Reduce redundant AOCL operand copying, use protected column-core/fringe
  products for eligible operands, and reuse scoped native output workspaces.
  Retain the full reduction dimension, protected outer dimensions and required
  output ownership. Experimental ANLS fitting also reuses its fixed training
  transpose and prepared immutable right-hand operand.
- Reuse the existing cross product in admitted nonnegative batched-NNLS
  qualification, and reuse owned residual and roundoff workspaces in qualified
  paths. Original-design numerical checks and SciPy fallback remain in place.
- Disclose changed AOCL overflow behavior in experimental NNLS: rejected
  overflowing products can produce different warning counts and, with NumPy's
  `over="raise"`, a `FloatingPointError` instead of the later `ValueError`.
  Overflow is not accepted as a successful solve.
- Extend blocked reconstruction-error reporting to the qualified mixed Windows
  AOCL 5.3/MKL configuration with Zen3 dispatch, one native thread and eligible
  aligned dimensions. This changes report computation, not fitting or stopping.
- Stop experimental rank-one ANLS-BPP sweeps when an accepted factor pair repeats
  exactly and `tol > 0`. The fit remains unconverged, emits `ConvergenceWarning`,
  and records `convergence_reason_="repeated_iterate"` and scalar `cycle_event_`
  metadata. The original projected-gradient criterion, `tol=0` behavior and
  higher-rank stopping rules are unchanged. See the
  [repeat policy and measured costs](docs/nmf-solver-comparison.md#experimental-anls-stopping-and-repeated-iterates).
- Reduce redundant finite checks and combine protected reconstruction with
  residual subtraction in the experimental projected-gradient diagnostic,
  preserving its nonfinite checks and the solver's stopping calculation. See
  the [diagnostic measurements](docs/nmf-solver-comparison.md#experimental-diagnostic-cost--november-2-2026).
- Add linear-SVM controls to the simulation and CelebA example CLIs, with
  per-model grid overrides and settings recorded for resume. SVM decision
  margins remain distinct from probabilities.
- Ignore genuinely empty CSV data records in CelebA attribute files while
  retaining validation of whitespace-only records, explicit empty fields and
  quoted multiline records.
- Expand the [API guide](docs/api.md) and docstrings with defaults, reasons to
  choose each opt-in, tradeoffs, refit requirements and separate concurrency
  controls. Add this changelog, a README link and source-distribution inclusion.

The AOCL workaround remains limited to covered products; native operations
inside other libraries, including MU, are not comprehensively protected.
Performance evidence is workload- and environment-specific, with measured
costs as well as gains; these changes do not establish a universal speedup.
The default solver, screening and classifier remain SPA, absolute screening
and KNN. Experimental ANLS-BPP remains outside the public solver choices.

## 0.4.1 — 2026-10-29

- Add a temporary AOCL-specific workaround for a reproduced native
  matrix-multiplication crash in explicit SPA NNLS products. Pad outer
  dimensions while preserving the reduction dimension, numerical qualification
  and SciPy fallback; MKL retains direct multiplication. The workaround does
  not repair the installed AOCL library or cover every internal native operation.
- Clarify training-only tuning and current-package comparison guidance without
  reinterpreting frozen historical studies.

See the [release notes](docs/release-0.4.1.md) and
[workaround retirement policy](docs/aocl-nnls-workaround.md#retirement-policy).

## 0.4.0 — 2026-10-29

- Add explicit `mu_projection="rowwise"` for per-observation MU initialization
  and stopping; `"batch"` remains the default and the compatibility choice for older serialized
  models. Record the resolved policy in `mu_projection_`. Apply the same policy
  to training reprojection with `training_coordinates="transform"`.
- Add the standard scikit-learn transformer interface and `fit_transform` to
  `NMFClassifier`.
- Fix mixed-dtype CD/MU prediction and add guarded float64 MU range recovery.
- Move prediction diagnostics into per-call evaluation results. Prediction no
  longer updates estimator `last_transform_*` attributes; callers that inspect
  those attributes must use the evaluation report instead.

See the [release notes and migration guidance](docs/release-0.4.0.md).

## 0.3.3 — 2026-10-26

- Add guarded batched SPA basis fitting and optional `projection_solver="qr"`,
  with original-design qualification and reference fallback.
- Handle exact-zero MU queries and improve float32 and logarithmic MU projection
  recovery for qualified numerical cases.
- Correct resumable-study fingerprints when native libraries load lazily.
- Expand class-support and timing reports; require pytest 9 or later for the
  source test launcher. Classifier choices and defaults remain unchanged.

See the [release notes](docs/release-0.3.3.md).

## 0.3.2 — 2026-10-20

- Simplify source-distribution inclusion to directory patterns and remove the
  additional archive-rebuild and unlisted-file checks introduced in 0.3.1.
- Clarify research documentation. Numerical behavior, public APIs, defaults
  and dependency requirements are unchanged.

See the [release notes](docs/release-0.3.2.md).

## 0.3.1 — 2026-10-20

- Remove the portable PC-throughput benchmark and its dedicated tests from
  public source and source distributions; installed wheels never included it.
- Add an explicit source-file manifest and distribution checks, and clean up
  documentation. Numerical behavior and public APIs are unchanged. The
  additional packaging checks are subsequently simplified in 0.3.2.

See the [release notes](docs/release-0.3.1.md).

## 0.3.0 — 2026-10-20

- Make `training_coordinates="auto"` the default. SPA classification now uses
  fixed-basis training reprojection, matching the prediction coordinate map.
  Explicit `"fit_transform"` retains the historical convention; retraining with
  the new default can change predictions.
- Raise the minimum Python version from 3.10 to 3.11.
- Add optional SD-scaled KNN coordinates, a linear-SVM classifier head and a
  shared prediction/decision-margin method.
- Improve screening stability at large offsets, add reversible NNLS range
  scaling and expand factor-recovery reporting.
- Reuse neighbors across KNN-grid candidates and add bounded reconstruction-error
  computation for qualified reporting workloads.

See the [release notes and migration guidance](docs/release-0.3.0.md).

## 0.2.0 — 2026-10-09

- Add guarded batched SPA projection, reusable projection processes and native
  thread controls. Explicit `projection_solver="scipy"` selects historical
  rowwise projection; older serialized models retain that behavior.
- Add fitted probability thresholds and positive labels, plus
  `predict_with_probabilities` to share projection work.
- Release SPA anchor workspaces earlier and make `joblib>=1.3` a direct dependency.

See the [release notes](docs/release-0.2.0.md).

## 0.1.0 — 2026-10-07

- Publish the first formal release of the 0.1.0a2 implementation.
- Update citation, README and version metadata; numerical behavior is unchanged
  from 0.1.0a2.

See the [GitHub release](https://github.com/cheng338/supervised-nmf/releases/tag/v0.1.0).

## 0.1.0a2 — 2026-10-06

- Publish the early prerelease of the supervised-NMF package, including
  `NMFClassifier`, `BinaryScreening` and `FeatureSPA`, SPA/MU/CD factorization,
  and KNN or logistic-regression classification.
- Add subject-aware/custom folds, checkpoint-bound sample identities, bounded
  rank exploration, reused fold-local screening statistics, imbalance summaries
  and numerical-stage diagnostics.
- Improve SPA anchor updates through BLAS and prepare NNLS layouts once per
  transform. Retain SPA, absolute screening and KNN defaults with the historical
  `training_coordinates="fit_transform"` convention.

See the [GitHub prerelease](https://github.com/cheng338/supervised-nmf/releases/tag/v0.1.0a2)
and [tuning improvements](docs/tuning-improvements.md).
