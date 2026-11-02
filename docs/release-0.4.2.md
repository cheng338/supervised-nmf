# supervised-nmf 0.4.2

Version 0.4.2 corrects evaluation and reporting edge cases, reduces repeated
numerical work, and adds exact-repeat stopping to the experimental ANLS-BPP
solver. SPA, absolute screening and unscaled KNN remain the public defaults.
There are no new public solver choices or installed-runtime replacements.

## Evaluation, examples and documentation

- Training and evaluation sample or subject identifiers must use compatible
  types. A numeric/string mismatch now raises before prediction, preventing
  overlap checks from silently missing differently represented identifiers.
  Integer and floating-point identifiers remain compatible.
- Evaluation diagnostics report the fitted threshold, positive label, class
  index and MU projection policy rather than later constructor changes.
  Refinement reports exclude booleans from numeric ranges and record declared
  `None` alternatives. The CelebA stage profiler correctly accounts for SPA's
  default training reprojection, including stages not reached after a failure.
- Simulation and CelebA example CLIs expose linear-SVM controls and preserve
  per-model grid overrides in resume settings. CelebA CSV loading skips genuinely
  empty records while retaining validation of whitespace-only records, explicit
  empty fields and quoted multiline records.
- The [API guide](api.md) and docstrings explain opt-in behavior, tradeoffs,
  refit requirements and separate concurrency controls. The new
  [changelog](../CHANGELOG.md) summarizes current and earlier releases.

## Numerical work and experimental stopping

Eligible AOCL products avoid redundant operand copies and use protected
column-core/fringe calculations. Batched-NNLS qualification reuses an existing
cross product and owned temporary storage while retaining original-design
checks and SciPy fallback. Experimental ANLS fitting reuses its training
transpose, prepared immutable right-hand operands and scoped output storage.
The full reduction dimension, protected outer dimensions and required output
ownership remain intact. MKL retains its direct multiplication policy.

The experimental solver and its diagnostics now follow the selected backend's
product policy. For rank one with positive `tol`, a detected exact repeat of
both accepted factors stops further sweeps. This is an effort stop, not a
convergence certificate: the fit emits `ConvergenceWarning`, retains
`converged_=False`, and reports `convergence_reason_="repeated_iterate"` with
scalar `cycle_event_` metadata. Original convergence and failure checks take
priority. Higher ranks and `tol=0` bypass repeat detection. Retained history and
factor comparisons add memory and runtime costs when no repeat is found. See
the [stopping policy and measurements](nmf-solver-comparison.md#experimental-anls-stopping-and-repeated-iterates).

The experimental projected-gradient diagnostic removes redundant finite scans
and combines protected reconstruction cropping with residual subtraction.
Nonfinite checks, including the behavior under `assume_finite=True`, are
preserved; the solver's stopping calculation is unchanged. Separately, blocked
reconstruction-error reporting admits the qualified mixed Windows AOCL 5.3/MKL
configuration with Zen3 dispatch, one native thread and eligible aligned
dimensions. These reporting paths do not change factorization objectives.

## Compatibility and remaining limits

On the recorded AOCL experimental-NNLS overflow fixtures, warning counts differ
from 0.4.1. With NumPy `over="raise"`, an overflowing product can raise
`FloatingPointError` instead of the later `ValueError`. All these outcomes
reject the overflowing computation; callers requiring exact historical
warning or exception behavior should account for this difference.

The AOCL workaround protects covered package products, not every operation
inside NumPy, SciPy or scikit-learn. Native AOCL exposure, including MU paths,
remains unresolved. This release does not certify those native paths or modify
installed libraries. The [workaround and retirement policy](aocl-nnls-workaround.md)
retain that scope.

Local validation used the existing Windows Python 3.12 MKL and AOCL environments.
The rank-one integration received broad regression and live-fit checks; the
subsequent diagnostic patch received separate focused tests, edge-case
qualification and four saved real-workflow replays. These are distinct evidence
sets, not a claim that the earlier full suite was rerun after the last patch.
Release-build and artifact checks are recorded separately with the release.

Measured benefits and costs depend on workload, backend and thread settings.
The [diagnostic measurements](nmf-solver-comparison.md#experimental-diagnostic-cost--november-2-2026)
compare diagnostic calls with the preceding local implementation, not complete
fits or a new 0.4.1 benchmark. Generic padding costs and noncycling repeat-check
overhead remain. No universal speedup, convergence or native-safety guarantee
is implied. Experimental ANLS-BPP remains outside `NMFClassifier`'s public
solver choices.

Python 3.11 or later is required. Historical release notes and study results
retain their original scope. Chang Cheng's original work remains All Rights
Reserved, with separate third-party notices preserved.
