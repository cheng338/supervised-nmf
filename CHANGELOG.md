# Changelog

This package originated in Chang Cheng's supervised NMF research scripts
accompanying his dissertation. This file summarizes published releases and
subsequent development changes; detailed release notes and historical study
records retain their original scope.

Release dates are GitHub publication dates in America/Indianapolis time.

## Unreleased

## 0.5.0 — 2026-11-18

See the [release notes](docs/release-0.5.0.md) for changes to defaults, migration guidance and measured costs.

- Speed up SPA basis fitting and projection by applying the first batched
  NNLS pivot to all pending right-hand sides together.

- Reduce repeated source hashing in `calibration_search`, including checkpointed searches.
  Each check reads the source files and reuses a SHA-256 digest only when
  the freshly read bytes match exactly. Checkpoint reports also serialize
  long lists of integer fold indices with fewer conversions.

- Share CelebA study runtime identity observation between full initialization
  and later checks, preserving numerical behavior. Recovery still verifies
  the recorded source and runtime before reusing a fit. See the
  [maintenance decision and timing tradeoffs](docs/development-protocol.md#shared-runtime-identity-observation).

- Keep prepared-QR construction and transport export in their owning class,
  preserving validation order, immutable buffer ownership, process reuse
  and numerical behavior.

- Bind current CelebA and Fashion-MNIST study helpers to their sibling source
  files. Reject helpers from another checkout and loaded helpers whose source
  has changed, with guidance to restart the interpreter. Preserve shared
  recovery ownership, lazy helper loading, numerical calculations and saved
  schemas; see [study helper imports](docs/development-protocol.md#study-helper-imports-and-source-changes).

- Clear partial learned state after ordinary fit failures in `FeatureSPA`
  and `BinaryScreening`, including screening-statistics reuse. Preserve
  successful calculations and the original exception; see
  [failed fits](docs/api.md#failed-fits).

- Preserve native weighted-KNN decisions and custom head/pipeline prediction
  policies during generic evaluation. Share eligible preprocessing and score
  calculations, retaining native calls for uncertain weighted votes and
  explicit numerical error handlers. Document the
  [evaluation prediction contract](docs/api.md#prediction-policies-during-evaluation),
  including warnings from reused calculations.

- Correct prepared-QR reuse when qualified parent and worker native libraries
  are enumerated in different orders. Preserve all state and ownership checks;
  keep immutable factor buffers up to 1 MiB as bytes during transport and retain
  read-only mapping for larger buffers. Model persistence is unchanged. See
  [QR reuse](docs/api.md#reusing-prepared-qr-factors).
- Defer classifier initialization until first access so projection-only workers
  avoid unused classifier setup. Public imports return the same class; errors
  from classifier-only imports now arise on first access. Document this
  [import behavior](docs/api.md#package-imports) and when to keep the existing
  one-worker projection default. No solver or resource default changes.

- Reduce redundant scalar copying in exact long-double identifier conversion
  and report serialization. Select the ordinary or native key loop once while
  preserving precision, scalar-subclass behavior and saved representations.
  Benefits and costs depend on identifier mixtures and the complete workflow;
  see the [identifier contract](docs/api.md#identifier-validation-costs).

- Reduce repeated mean and normalization work in eligible C-order screening
  statistics, preserving screening rules, selected values, fitted diagnostics,
  and warning/error behavior. Unsupported contexts retain the existing path;
  no public option or dependency minimum changes. Benefits and costs depend on
  shape, layout and the complete workflow; see the
  [screening reuse decision](docs/development-protocol.md#guarded-c-layout-screening-reuse).

- Reduce temporary allocation and reduction work in eligible logarithmic MU
  projection and recovery using an owned workspace. Preserve numerical order,
  warning/error behavior and scalar fallback; unsupported implementations keep
  the existing calculation. No option or default changes. Benefits and costs
  depend on the workload; see [MU reductions](docs/api.md#guarded-logarithmic-mu-reductions).

- Retry transient Windows replacement denials when saving shared study receipts
  and progress files. Preserve existing progress retry limits, file contents,
  and failure reporting.

- Retry transient Windows file-replacement failures when saving three-label
  study receipts, preserving receipt checksums and durable staging. Persistent
  failures still raise after the bounded retry limit.

- Restore NMF projection timing in Fashion-MNIST pipeline profiles, including
  failed projection attempts, while preserving classifier projection behavior.

- Prevent concurrent projection diagnostic receipts from sharing a filename
  by creating each file atomically. Numerical calculations are unchanged.

- Reuse verified fixed-basis QR preparations across eligible SPA predictions
  and process workers on qualified MKL runtimes. Keep QR opt-in and the default
  `"auto"` unchanged; unsupported contexts retain ordinary execution. Factors
  live in a transient external weak cache, with unchanged model-owned state and
  no persisted preparation. Benefits, startup and small-workload costs remain
  workload-dependent; see [QR reuse](docs/api.md#reusing-prepared-qr-factors).

- Reduce repeated full-neighborhood work in eligible uniform-weight KNN
  calibration grids. Preserve native validation, predictions, tie rules and
  warning/error behavior; record every attempted shared query once. Grids
  without a successfully fitted full-size head avoid unused planning setup.
  No option or default changes. Gains and costs depend on the workload; see
  [full-training neighbor counts](docs/api.md#full-training-neighbor-counts).

- Add `calibration_knn_grid(..., screening_statistics_recovery="early")` to
  preserve eligible screening statistics before a fold receipt completes;
  the default remains `"off"`. Validate saved event consumers before extending
  partial checkpoints, retain immutable historical costs, and recompute
  unavailable learned payloads only when their metadata remains valid.
  Ordinary runs and metadata checks retain added costs; see the
  [recovery contract](docs/api.md#early-recovery-of-screening-statistics).

- Reduce repeated runtime-initialization and library-discovery work in the
  CelebA parameter study while retaining fresh recovery identity checks.
  Serialize disjoint same-process finalizations through reentrant ownership,
  keep completed-result reuse outside that ownership, and preserve retained
  controller discovery after registration changes. Numerical algorithms and
  the default recovery policy remain unchanged. Gains and costs depend on the
  workflow; see the
  [finalization decision](docs/development-protocol.md#celeba-finalization-initialization-and-ownership).

- Reduce repeated sample/group identifier work during evaluation using owned
  native comparisons, proved exact mixed-format promotion and guarded small or
  uniform-group routes. Preserve identifier precision, original report scalars
  and validation/error behavior; no public option or solver change is added.
  Benefits and costs remain workload-dependent; see the
  [identifier evaluation decision](docs/development-protocol.md#exact-identifier-evaluation-and-group-bookkeeping).

- Reduce screening work for eligible Fortran-order inputs using bounded
  statistics blocks and the existing float32 widening allocation. Preserve
  screening rules, selected values and fitted diagnostics; no public option is
  added. Gains and costs depend on shape and layout; see the
  [screening layout decision](docs/development-protocol.md#screening-statistics-for-fortran-order-inputs).

- Reduce copying work for eligible experimental ANLS-BPP fitted/returned factors
  and final reconstruction-report workspaces. Preserve independent C-order
  factor arrays, solver arithmetic and stopping rules; other layouts retain
  their existing copies. No public option is added. Whole-workflow effects
  remain workload-dependent; see the
  [copying decision](docs/development-protocol.md#experimental-anls-factor-and-report-copying).

- Reuse eligible completed H-solve residuals and gradients within experimental
  ANLS-BPP training. Preserve solver results, stopping and warning/error behavior;
  bypass reuse for rank-one inputs with at most 16,384 elements. No public option
  is added. Benefits and costs remain workload-dependent; see the
  [training-state reuse decision](docs/development-protocol.md#experimental-anls-training-state-reuse).

- Reduce fingerprinting work in eligible rank-one experimental ANLS-BPP repeat
  detection using a CRC32 index with unchanged full-factor bit verification.
  Preserve stopping rules and retained arrays; no option is added. Whole-fit
  effects remain workload-dependent; see the
  [indexing decision](docs/development-protocol.md#exact-rank-one-repeat-indexing).

- Reduce repeated work in eligible MU recovery by retaining completed row
  prefixes, reusing guarded setup within a call and using smaller logarithmic
  panels for C-contiguous right operands. Preserve ordered scalar recovery for
  the remaining rows and the existing warning/error contract. Public defaults
  are unchanged and no option is added. Gains and costs depend on the workload;
  see the [recovery decision](docs/development-protocol.md#interrupted-mu-recovery-and-panel-refinement).

- Reduce Linux default-study CLI startup for eligible multiworker checkpoint
  runs using an empty-preload `forkserver`. New output directories retain
  `spawn`; original workers still validate every checkpoint. Imported defaults,
  calls requesting one worker and the parameter-study launcher are unchanged. No new
  public option is added. See the [startup policy](docs/celeba-study.md#process-startup-for-current-runs).

- Reduce redundant implementation-source discovery in reusable search checks.
  Preserve fresh per-check source hashes, configuration validation and existing
  checkpoint guards; custom source layouts retain their original discovery path.
  See the [development decision](docs/development-protocol.md#implementation-source-discovery-for-reusable-searches).

- Correct supporting study utilities to request balanced accuracy explicitly,
  capture projection diagnostics per call, and profile MU prediction without
  changing its dispatch. Close arrays opened by screening/worker utilities on
  success and failure while preserving caller-owned arrays and mapped views
  until their last use. Core estimator implementations and public defaults are
  unchanged.
- Update current-consumer test fixtures to use the installed package identity;
  retain historical qualification checks as explicitly marked, opt-in tests.
  Correct test import isolation, owned-process cleanup and portable fixture
  paths. See the [test scope](docs/validation.md#current-regressions-and-archived-qualification)
  for the qualification boundaries and measured utility cleanup costs.

- Reduce repeated distance, boundary and vote work in compatible shared-KNN
  calibration grids. Preserve the native selected-k fallback for uncertain
  batches and the original calculation order under nondefault NumPy error
  policies. Uncomputed vote diagnostics are explicitly null. Complete-workflow
  timings remain mixed; see the [reuse guidance](docs/api.md#explicit-grids-with-shared-representations).

- Reduce completed CelebA resume startup on Linux when only one worker is
  effective. Reuse still validates checkpoint contents and source/runtime
  identity; incomplete or ineligible work uses the existing process route.
  Multiworker runs retain their process parallelism. See the
  [startup policy](docs/celeba-study.md#process-startup-for-current-runs).
- Accelerate exact sample/group identifier checks and report serialization.
  Preserve extended precision, scalar-subclass behavior and saved representations;
  discover the native floating-point format only when needed. Small requests
  and wider-precision key conversion can retain overhead; see the
  [identifier contract](docs/api.md#identifier-validation-costs).

- Change `NMFClassifier.mu_projection` to `"auto"`: newly fitted MU models use
  per-observation initialization and stopping; SPA/CD resolve to no MU policy.
  Retain explicit `"batch"` for callers accepting request-composition sensitivity.
  Existing fitted policies remain unchanged until refit. The default SPA solver
  and MU/CD training-coordinate policy are unchanged.
- Accelerate rowwise MU with guarded ordinary-row and singleton execution,
  layout-safe logarithmic panels, and independent vector recovery with ordered
  serial replay on failure. Preserve output dtype and numerical diagnostics;
  document small-request and failure-path costs, without promising bitwise
  batch invariance or higher accuracy. See [MU projection policies](docs/api.md#prediction-batches).

- Make retained dataset and CelebA study helpers request balanced accuracy
  explicitly for initial calibration and refinement. Canonicalize threadpool
  record order in new protocols while retaining exact source/runtime checks.
  The public calibration default remains accuracy. See the
  [selection objective](docs/api.md#selection-objective-and-report-fields).
- Validate three-label report declarations and use explicit amended schedules
  for scope and repeat counts. Label reconstructed scope as provisional and
  withhold affirmative planned-study inference. Preserve saved metadata and
  disclose scope provenance; see the [reporting contract](docs/three-label-refinement-reuse.md#declared-reporting-scope).

- Remove the combined four-solver runner's environment-directory-name restriction
  while preserving source, executable and runtime identity requirements for
  resume. Mark frozen three-label and shared-screening launch guides as historical
  tooling outside current runtime qualification; direct current users to the
  grouped example and public API.

- Publish four-solver orchestration JSON through private temporary files and
  bounded retries when replacement is denied. Preserve the previous complete
  file on publication denial, retain interrupted evidence, and keep exact-source
  resume requirements. Document retry costs, reader availability and durability
  limits in the [status-file publication guide](docs/nmf-solver-comparison.md#status-file-publication-for-new-runs).

- Retire the legacy schema-1 four-solver numerical-repair migration command.
  Preserve historical source and receipts, read-only provenance checks, and
  ordinary study resume with identical data, settings, source and runtime.
  See the [four-solver workflow](docs/nmf-solver-comparison.md).
- Verify the original three-label refinement ledger before activating reused
  artifacts. Reject a missing or changed recorded ledger, or a newly appearing
  unrecorded ledger, before writing activation outputs. Preserve legitimate
  no-ledger searches, historical charges and failed-operation evidence. See the
  [refinement reuse contract](docs/three-label-refinement-reuse.md).

- Reject inconsistent current shared-KNN CV receipts in the CelebA and
  four-solver audit tools. Check schema types, qualification/refit flags and
  agreement between generic and balanced-accuracy scores, retaining legitimate
  failed/nonconverged records and historical read-only checks. See
  [report fields](docs/api.md#selection-objective-and-report-fields).

- Require NMF constructor and fitted prediction-policy fields where they are
  used, instead of supplying defaults for missing attributes. Apply the same
  fitted-policy requirement in shared NMF KNN calibration. Preserve explicit
  `None` and explicit batch settings, fitted-versus-pending settings, current model
  persistence and optional diagnostic metadata. See
  [saved decisions](docs/api.md#saved-decisions).

- Require the four projection execution attributes in `NMFClassifier` and
  `FeatureSPA` instead of supplying legacy values when fields are missing.
  Preserve constructor defaults, explicit SciPy selection, numerical fallbacks
  and current model persistence. See the
  [projection controls](docs/api.md#spa-projection-execution).

- Remove automatic legacy-state field filling when `NMFClassifier` and
  `FeatureSPA` models are loaded. Preserve current-format pickle/joblib
  round trips, copying, cloning and refitting. Cross-release artifacts are
  unsupported; successful loading does not guarantee they remain usable.
  See the [model-persistence contract](docs/api.md#model-persistence).

- Put release building, wheel checking and environment reporting behind explicit
  `main()` entry points, so importing the three utilities does not execute them.
- Let `compare_classifiers.py --threads` accept positive native C-int counts above
  eight. Keep the default of one and reject invalid counts before data or output
  work.

- Require the CelebA parameter study's declared chunk-report inventory and one
  finalization owner. Add `--final-fit-recovery all` for completed fits and
  terminal outcomes from a trusted local study. Preserve complete native state
  without refitting, using a public verified-cache buffer for eligible raw
  KNN and complete joblib state for other layouts. Restore the raw matrix as
  an owned read-only snapshot, count copying inside model loading, and reject
  incompatible committed references.
  Keep recovery `off` by default to avoid its verification and storage costs.
  Document ordinary-run overhead, measured restore costs, prediction-time
  increases and uncertainty; use the option when preserving costly completed
  fits is worthwhile.
  Preserve completed-result precedence, original fit accounting and interrupted
  evidence, with strict errors for incompatible committed recovery state.
  Existing runs are not migrated. See the [restart contract](docs/celeba-parameter-study-recovery.md).

- Apply the grouped-tuning example's budget of one native thread to its selected
  model's final fit, matching calibration and evaluation. Restore the caller's
  previous thread limits when final fitting returns or raises an error.
- Prepare CelebA default-study caches privately and verify each cache before
  publication. Prevent competing writers and recover verified completed work
  after interruption, including a completed largest training array. Add
  `--recover-incomplete` to preserve and rebuild legacy final directories with
  missing or malformed manifests; keep this option off by default and retain
  hard errors for committed integrity failures. Compatible completed caches
  keep their existing arrays and manifests. See the
  [restart policy](docs/celeba-default-results.md#preparing-and-restarting-caches).
- Skip shared-grid screening, representation fitting and validation projection
  when every requested neighbor count exceeds a fold's training size. Record
  the neighbor-feasibility failure and zero skipped work, retaining all declared
  candidates, feasible-fold diagnostics and checkpoint accounting. This early
  failure takes precedence over errors or warnings from unexecuted fitting.
- Reject class labels ending in NUL before screening, fitting, scoring,
  evaluation and checkpoint reuse can silently misinterpret them. Preserve
  ordinary and embedded-NUL labels, validate original sequences before their
  information is lost, and document safe encoding and prior-conversion limits.
  Existing fitted models retaining unsupported labels require refitting.
- Add opt-in fold-first execution to shared KNN grids for eligible restartable
  multi-representation studies. Preserve representation-first execution by
  default, record requested/resolved order and fallback reasons, and bind the
  choice to checkpoint identity. Keep canonical final reports and committed-work
  accounting without adding persistent screening state.
- Return the existing closed-form coordinates directly for rank-one MU
  logarithmic recovery with `tol=0`. Skip unused setup and iterations, including
  their incidental underflow warnings/errors, while retaining output-range
  checks and the existing behavior of other projection paths.
- Describe the current AOCL workaround's product routing and scoped storage
  reuse separately from its original validation evidence. Identify the retained
  local scripts and study inputs needed by historical reproduction commands.
- Validate four-solver reporting margins, numerical slack and derived regret
  calculations before ranking. Reject unsupported zero/invalid margins and
  nonfinite calculations with clear errors. Resolve explicit sensitivity margins
  independently of the primary limits, preserving the frozen positive-margin
  protocols and their ordinary decisions.
- Keep experimental ANLS-BPP transforms from changing fitted state. Provide
  caller-owned query diagnostics and use them in experimental study reporting,
  preventing stale failure records and cross-caller diagnostic attribution.
  Preserve solver behavior, warnings, returned coordinates and historical reports;
  ignore legacy model-side query fields when qualifying current study calls.
- Normalize Python/NumPy integer thread counts consistently across calibration,
  evaluation and comparison helpers, rejecting Boolean/temporal values and counts
  outside the native integer range before runtime or file activity. Accept NumPy
  Boolean refit flags and validate them before generating folds. Validate shared/
  rank fixed seeds before side effects and reject temporal seed values. Preserve
  and document general calibration's existing random-state compatibility.
- Prevent competing calibration calls from overwriting checkpoint progress by
  holding per-file ownership through the full call, including refit and fallback.
  Protect manifest creation/verification and isolate atomic JSON writes with
  exclusive temporary files. Competing checkpoint writers fail promptly;
  independent destinations remain independent. Retain source compatibility
  checks and persistent lock markers.
- Use absolute pooled-t screening by default in both screening APIs, and ordinary
  accuracy by default in calibration. Balanced accuracy remains explicit. Explain
  inactive screening controls and omit them from refinement recommendations.
- Apply the same fixed-basis NNLS map in SPA training and prediction, including
  standalone `FeatureSPA.fit_transform`. Reuse already computed training
  projections in the standard SPA classifier's `fit_transform`.
- Match native KNN decisions at weighted rounding ties and fall back to public
  neighbor prediction when shared-distance qualification is unsafe.
- Clone estimator-valued search parameters for every fit, retain live candidate
  objects during rank expansion, and bind reusable computations to typed
  configurations and implementation sources. Unsupported opaque or custom-cloning
  states fail explicitly. Old checkpoints remain historical evidence.
- Adapt maintained study runners and readers to the new scoring and identity
  contracts, retaining explicitly declared historical objectives. Relocation of
  identical implementation files is allowed only for read-only receipt audits;
  checkpoint-resume identity remains exact.
- Update the theory guide to the final dissertation and document training-only
  transformations, training-sample reuse, and the limits of the theoretical claims.

## 0.4.3 — 2026-11-02

See the [release notes](docs/release-0.4.3.md) for compatibility and validation details.

- Consistently reject Boolean, complex and temporal sample/group identifiers
  across scalar and array representations before fitting or prediction.
  Unsupported identifier objects raise `ValueError`. Python/NumPy integer,
  finite floating-point and nonblank string identities are preserved; large
  Python integers no longer fail a NumPy finiteness check. Document the
  identifier policy.
- Use exact internal numeric keys for identifier uniqueness, overlap checks,
  subject counts and grouped CV. Mixed NumPy scalars and Python integers
  no longer miss equal IDs through inconsistent hashes or merge distinct
  groups through rounded comparisons. Original IDs and report encodings
  remain unchanged.
- Fix recursive serialization of NumPy extended-precision scalars during report
  construction and checkpoint writing. Tagged JSON preserves finite long-double
  values exactly, including adjacent values, extreme magnitudes and signed zero;
  extended complex report values also serialize without recursion. Ordinary
  scalar encodings and identifier validation rules are unchanged.
- Initialize the numerical-library inventory before freezing general
  calibration checkpoint identities, matching shared-grid calibration.
  Lazy BLAS/OpenMP initialization no longer makes an otherwise identical
  resume incompatible; actual source/runtime changes still reject
  reuse. This also applies to comparison identities; initialization occurs
  before their calibration fit/tuning timers.

- Expand padding-workspace ownership, reuse and failure-recovery tests.
  Keep maintained AOCL operand tests focused on behavior without pinning
  one historical implementation hash. Clarify the historical benchmark
  wrappers' environment restrictions and the scope of portable runners.

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
