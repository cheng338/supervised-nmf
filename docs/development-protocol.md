# Development decisions and experiment protocol

Author: Chang Cheng.

This record follows development through version 0.5.0. Timing and validation
results apply to the source, hosts and workloads described in each study. The
study workspace retains the raw observations, source snapshots and test logs.
See the [0.5.0 release notes](release-0.5.0.md) for the published behavior.

## Shared runtime identity observation

CelebA study recovery shares the ordered lazy imports, native-library observation
and identity assembly used by full initialization and later checks. The
initializing path retains its native probes, and the warm path still observes
fresh guarded inventory. Custom helpers, failure behavior and cache ownership
retain their contracts. Fitted-outcome recovery still verifies the recorded
source and runtime before reusing a fit. Source changes therefore require a new
run under the [existing recovery policy](celeba-parameter-study-recovery.md).

The shared implementation is adopted for simpler maintenance despite measured
timing costs. Eight balanced source pairs per host compared fixed raw/PCA
fixtures with one native thread in unchanged Windows AOCL and physical Ubuntu
MKL environments. Contract and recovery checks passed with identical numerical
outputs. Selected mean elapsed changes follow, with positive values indicating
slower execution. These observations do not promise a general speed improvement.

| Measured operation | Windows AOCL | Ubuntu MKL |
| --- | ---: | ---: |
| Tiny raw, new fit with recovery `all` | -0.194 ms (-3.06%) | +0.033 ms (+0.57%) |
| Wide raw, restored prediction | -0.301 ms (-3.14%) | +0.099 ms (+0.72%) |
| Wide raw, fit with recovery `off` | +0.520 ms (+7.49%) | -0.076 ms (-1.26%) |
| Tiny raw, restored prediction | +0.361 ms (+6.20%) | -0.053 ms (-0.91%) |
| PCA, fresh prediction with recovery `all` | +0.304 ms (+4.82%) | -0.005 ms (-0.09%) |
| Wide raw, fresh checkpoint | +2.041 ms (+12.12%) | -0.018 ms (-1.58%) |
| Wide raw, whole fresh call with recovery `all` | +2.804 ms (+1.87%) | -0.181 ms (-0.27%) |
| Tiny raw, whole fresh call with recovery `all` | +0.302 ms (+0.22%) | +0.332 ms (+0.57%) |

## Study helper imports and source changes

The current CelebA parameter/default studies and Fashion-MNIST study require
their recovery, codec, profiling and experimental helpers to come from the same
source checkout. They check the relevant helper's origin and compare its source
recorded at import with freshly observed file contents before accepting study
identity or using the helper. A conflicting checkout, a changed loaded helper,
or a helper loaded before this source-recording policy raises an error at the
relevant import or study boundary.

Start a fresh interpreter after editing these helper files or changing
checkouts. Keep the imported checkout in place for that process; live helper
replacement or moving an imported checkout is unsupported. Restarting does not
make an old checkpoint compatible: data, settings, source and runtime must still
match its recorded identity. Preserve previous results and use a new output
directory for changed computations.

Supported callers retain one shared recovery module and its reentrant ownership
guard. The codec and experimental helper remain lazy at their existing use
boundaries. Read-only Fashion diagnostics reuse the surrounding study's source
checks before publication instead of rereading helper contents on every call.
Numerical algorithms, fit/prediction timing boundaries and saved schemas are
unchanged. Source validation adds work; complete-workflow timings remain
workload-dependent.

## Guarded C-layout screening reuse

Eligible C-layout screening reuses two intermediate results within a fit.
Normalization writes into that call's owned float64 minimum-subtraction
workspace instead of allocating a second full zero array. Masked constant
columns are explicitly set to positive zero. Variance calculation uses the
already computed class means through public `np.var(..., ddof=1, mean=...)`,
before the existing constant-class correction. Input and label validation,
class grouping, pooled statistics, feature selection and transformed values
remain unchanged. No public option, default or persistent fitted field is added.

Both operations require at least 131,072 validated input elements. Mean reuse
also permits exactly two features with at least 8,192 rows, subject to all its
other checks. Normalization has no corresponding exception. These are fixed
internal execution policies, not universal crossover estimates or per-host
tuning. The common minimum gives up measured Windows medium-size gains to
avoid repeatedly slower normalization on the same admitted Ubuntu workloads.

Admission examines the array returned by ordinary validation: an exact native
float32 or float64 C-contiguous ndarray, together with a same-shaped, native
float64 C-contiguous scratch array owned by this call, writable and without a
base object. An original subclass can be normalized by validation before these
checks. Mean reuse additionally requires canonical float64 class groups with
at least two observations. Caller arrays are not used as mutable workspaces.
The existing Fortran-order statistics branch is unchanged.

Each operation checks its direct callable bindings, available code/default
identities and selected NumPy implementation bindings at its own use point.
Changed watched bindings, unavailable optional implementation details,
unsupported ownership/layout, nonignored underflow or
`assume_finite=True` retain the original calculation. The checks cover direct
operations, not arbitrary dependency modifications. Mean reuse requires
the public variance `mean` argument introduced in NumPy 2.0; this is an
acceleration condition, not a dependency pin or an increase in the package's
NumPy minimum. Supported older implementations retain the original variance
path. The two operations can qualify independently, and the original error
translation and fallback operations remain in place.

The isolated comparison used unchanged Windows AOCL and physical Ubuntu MKL
environments with one native thread. Each host ran 18 fresh processes in all
six orders of current, the preceding candidate and the selected source. All
22 public cases retained one warmup and three measured fit-plus-query calls
per process; query means transform for screening and predict for classifiers.
The repetitions were averaged within each process before the six paired
comparisons. Mean elapsed changes versus current were:

| Workflow | Windows AOCL | Ubuntu MKL |
| --- | ---: | ---: |
| Large C float64 screening, 8,192 by 1,024 | -15.922% | -12.086% |
| Large C float32 screening, 8,192 by 1,024 | -16.314% | -12.403% |
| Fashion screening | -17.115% | -10.945% |
| Fashion fold screening | -15.611% | -19.084% |
| Two-feature screening, 8,192 rows | +5.662% | -3.887% |
| Full Fashion C classifier | -1.657% | -0.124% |
| Full Fashion F classifier | -1.162% | +1.607% |

All six pairs improved for each large and Fashion screening row on each host.
The Windows two-feature case instead cost 0.092 ms more, with four of six pairs
slower; its Ubuntu counterpart saved 0.072 ms in all six pairs. Small declined
cases are not universally faster: the 600-by-130 constant-feature case cost
0.079 ms more on Windows and 0.032 ms more on Ubuntu. The Ubuntu full C
classifier's small mean improvement included four slower pairs. Screening
gains are not presented as equivalent full-classifier gains, and measured
small guard costs do not explain every difference in complete calls.

The initial Ubuntu full F classifier cost 9.042 ms more, with five of six pairs
slower. This prompted a separate fixed follow-up retaining the complete
22-case prefix and the same public clocks. Two cycles of all six orders used
36 fresh Ubuntu processes: two current controls shared the identical source
path, and the third arm used the selected source. Existing fit-stage timers
were copied after the clocks. All 36 separate post-panel route witnesses
reached the original F block-statistics path and called neither new readiness
function, with matching scientific signatures and restored observation state.

Across the 12 paired blocks, the selected source's full F workflow averaged
0.086 ms slower than current A (+0.015%), 3.549 ms faster than current B
(-0.621%), and 1.732 ms faster than their combined mean (-0.304%). The two
identical-code controls differed by 3.635 ms (+0.640%). Their differences were
not subtracted from candidate clocks; the two candidate contrasts share
observations and are not 24 independent replications. Screening and
factorization stage differences changed sign between the two cycles. No stable
adverse F-workflow mechanism attributable to this change emerged in that
follow-up. This does not identify the unrecorded cause of the earlier loss,
prove zero cost, or transfer an allocator diagnosis from a different workload.

All other follow-up cases were retained. Large and Fashion C screening saved
about 8–11 ms, with all 12 blocks faster in each case; the full C classifier
saved 8.478 ms (1.456%), also in all 12 blocks. Small positive means remained,
including 0.017 ms for the 600-by-128 float64 case and 0.013 ms for the tiny
case. The earlier Windows costs and all original measurements remain part of
the decision. Adoption rests on repeated execution-speed benefits with these
tradeoffs, not memory savings or a claim of universal improvement.

The isolated candidate passed 134 scientific cases and 12 active-path guard
controls per host, with saved arrays independently checked, plus 257 focused
tests and 68 subtests per host. These cover preserved statistics and outputs,
selection boundaries, input ownership, failed refits, warnings/callbacks,
error policy and watched-binding changes. The performance and correctness
evidence belongs to the tested isolated source; it does not substitute for
integration, accumulated-suite or release validation.

## Guarded logarithmic MU reductions

Eligible logarithmic MU projection and recovery reuse an owned product
workspace during stable reduction. This preserves the existing panel geometry,
last-axis reduction order, tied-maximum handling and float64 arithmetic
association. No public option, fitted projection policy, iteration allowance,
training-coordinate default or model-owned state changes.

The acceleration currently qualifies NumPy 2.5.4 and SciPy 1.18.1 with the
expected normalized SciPy implementation source and callable bindings. These
are acceleration checks, not dependency pins. Other supported versions,
unavailable source and incompatible bindings retain the existing calculation.
Private SciPy helper and compatibility-namespace usage requires requalification
when that implementation changes; the public logsumexp API alone does not
guarantee identical rounding or warning behavior across implementations.

Only freshly owned native-float64 C-contiguous product arrays are admitted.
Each row needs a finite maximum of absolute value at most `2**60`; the feature
count is bounded by `2**53`. Mixed negative infinity is accepted. Unsupported
inputs decline before workspace mutation. A `FloatingPointError` after
mutation replays the original scalar calculation on untouched operands.
Per-call binding checks and recovery handoff checks remain active. Caller
arrays are never used as the mutable workspace, and no persistent cache is added.

The isolated source comparison used unchanged Windows AOCL and physical Ubuntu
MKL environments. Four balanced process blocks covered nine saved Fashion
projection/prediction cases per implementation. The same previously fitted
donor was used within each host; these are prediction workflows, not new NMF
fits. Only the public joint prediction/probability call was timed. Mean elapsed
changes versus the then-current source were:

| Workflow | Windows AOCL | Ubuntu MKL |
| --- | ---: | ---: |
| 64 ordinary queries | -10.607% | -10.827% |
| Full 2,000-query workflow | -14.801% | -12.098% |
| Recovery cap 180, 61 queries | +1.036% | -8.187% |
| Float32, 32 queries | -0.476% | +0.294% |

All four pairs improved for both the 64-query and full-query cases on each
host. The Windows cap-180 and Ubuntu float32 cases each had two slower pairs;
Windows float32 had three slower pairs despite its lower mean. These results
do not establish universal improvement or statistical significance. Tiny
Windows reductions can also cost several microseconds more. A common small-work
cutoff would sacrifice measured Ubuntu gains, so no shape or platform whitelist
is used. Smaller panels, private-body shortcuts and a public-NumPy spelling
were assessed without establishing a better overall choice.

The one-time implementation check occurred outside those prediction clocks.
Its first observed constructor cost was 0.549 ms on Windows and 0.162 ms on
Ubuntu. Dependencies were already loaded by preceding normal imports; these
are not total startup or cold-filesystem bounds. Per-call checks were included
in the public timings. A historical CPU-active Ubuntu timing excursion did not
recur in fixed follow-up contexts; its unrecorded historical cause remains
unknown. No measurements were discarded or runtime allocator settings adopted.

Direct numerical/guard checks, public outputs and warning/recovery behavior,
persistence, concurrent calls and focused regressions supported the isolated
candidate. The retained performance evidence belongs to that source comparison;
integration formatting and diagnostic-label cleanup are not new timing results.

## Reusing prepared fixed-basis QR factors

Explicit `projection_solver="qr"` can reuse a verified QR preparation across
calls for a live fitted SPA basis and share it with eligible process workers.
The default remains `"auto"`; no public option, dependency requirement, solver
objective, basis-fitting rule or persistent estimator field changes. The
[API contract](api.md#reusing-prepared-qr-factors) describes the supported
lifecycle and fallback.

Positive reuse is currently qualified for NumPy 2.5.4, SciPy 1.18.1 and
threadpoolctl 3.7.0 with the accepted stock MKL/OpenMP composition. These are
acceleration checks, not dependency pins. Other supported versions, providers,
replaced callables and unsupported numerical policies retain ordinary execution.
Cold controller discovery uses a guarded, warning-free stock Linux path;
unqualified discovery declines. A conservative negative provider memo selects
only the ordinary route and does not bypass fresh positive-context checks.
Library upgrades may disable reuse until its implementation is requalified.

The preparation contains an immutable basis snapshot and factors. Each use
checks exact basis bits and the current numerical/native context. The external
process-local registry has one preparation per live weak basis owner; it does
not keep that owner alive. Collection removes its entry, a new process clears
inherited registry state, and a changed basis or context prevents stale reuse.
Model-owned fitted state and serialization remain unchanged. Shallow estimator
copies can share the same basis owner; deep copies and loaded models have new
owners. No preparation is persisted in a model or checkpoint. Retained snapshots
and factors consume memory while their owners live; this is not a memory-saving
change or a fixed global byte-budget cache.

The additional geometry gate requires rank at least 32 and
`n_features * rank**2 >= 2**20`, along with the existing QR query, cap, range,
conditioning and output checks. These fixed gates were selected from a bounded
comparison, not a universal crossover claim. Eligible serial calls capture a
successful required reduction. For process calls, the parent may prepare once
when chunks outnumber requested workers, or reuse a matching preparation for a
single wave. For built-in integer worker controls, a cold single-wave call keeps
the existing worker route; other accepted integer types can conservatively
attempt cold parent preparation. Successful
assembly precedes publication; failed optional context/storage work declines,
while required solver, transport and assembly behavior remains observable.
Workers rebind the immutable payload against their own current context. The
original constant Joblib `max_nbytes="1M"` is retained for all process calls,
including mixed models and declined reuse; no adaptive mapping threshold is used.

The final isolated comparison used unchanged Windows AOCL and physical Ubuntu
MKL runtimes. Each host first passed a separate, independently audited complete
two-source preflight, then four fixed opposite-order pairs. Each fresh process
ran six actual fitted synthetic histories: a fit or `fit_transform` followed by
three transforms. The main fit-transform recipes used 1,024 training rows,
512 query rows, two projection workers, one native thread and batch size 256
unless stated otherwise. Rank 128 is prospective, not historical incidence.
Mean complete-history differences below are candidate minus the previous source;
negative values are faster.

| Recipe | Windows AOCL | Ubuntu MKL |
| --- | ---: | ---: |
| 2,048 features, rank 32 | +26.185 ms (+0.734%) | -3.361 ms (-0.180%) |
| 8,192 features, rank 64 | -9.218 ms (-0.125%) | -341.371 ms (-6.630%) |
| 8,192 features, prospective rank 128 | +68.454 ms (+0.660%) | -1,082.867 ms (-12.061%) |
| Cold fit-only, 2,048 features, rank 32 | -10.265 ms (-0.309%) | +14.529 ms (+0.895%) |
| 4,096 features, rank 32, batch size 64 | -39.302 ms (-0.874%) | -83.288 ms (-3.198%) |
| Below the reuse gate: 256 features, rank 16 | -16.514 ms (-0.618%) | +7.474 ms (+0.698%) |

All four Ubuntu pairs improved for the rank-64, rank-128 and batch-64 histories.
Their three disjoint query calls together improved by 322.891 ms (19.318%),
819.273 ms (26.870%) and 56.632 ms (7.623%), respectively. The cold Ubuntu
history was slower in three pairs; the below-gate history was slower in all
four. Windows declined reuse, and both signs of its elapsed differences remain
reported. Unsupported and below-gate differences are not wholly attributed to
optional preparation. Enclosing-history, fit and query clocks overlap and are
not added. All phase rows, raw pairs and adverse results remain retained;
four pairs support descriptive evidence, not significance, provider causation,
a guaranteed speedup or pooling with earlier panels and controls.

Separate fresh-process measurements found package-import mean increases of
7.865 ms on Windows and 2.634 ms on Ubuntu, after numerical-runtime bootstrap.
The larger guarded child-wall changes also include bootstrap, verification and
I/O and are not pure import costs. Neither interval is subtracted from the
workflow results. This startup cost applies even when a user does not select QR.

Three saved real-data cohorts were separately refitted with the earlier serial
reuse candidate. Fashion's selected basis was rank 32 with 700 features, and
faces was rank 16 with 871 features; both declined the geometry gate. A high-rank
cohort selected rank 64 with 557 features and positively reused its preparation.
Its original QR plus condition work was less than about 1 ms per query, and the
complete serial query effects were small and mixed. These source-bound results
are not relabeled as final-candidate timings. Default `"auto"` does not activate
QR reuse. One eligible cohort among three is not a usage-frequency estimate.

Alternatives included eager fit preparation, capture during training projection,
serial weak-owner reuse, per-call process preparation, factor transport,
geometry/worker-wave gates, and strict or warm-hit parent sharing. A changing
process mapping threshold forced executor replacement in an earlier prototype;
the final constant threshold removes that specific configuration change.
Broad-provider variants had adverse Windows costs and required a private AOCL
compatibility recognizer. The selected provider restriction removes that
recognizer, but exact private controller qualification, immutable ownership,
lifetime and transport still require maintenance. The isolated three-file
implementation added 600 lines and removed 21 before integration cleanup.
The implementation was adopted for the measured gains on eligible workloads.
A dissenting assessment favored keeping the previous implementation because of
the long-term maintenance cost. It raised no unresolved correctness issue.

Final isolated checks passed 45 focused cases per host, comprising 31 structural
guards and 27 actual operations across 14 caller cases, plus 83 existing parent
tests and 45 subtests. The final comparison verified 328 actual arrays per host
after the separate 82-array preflight. An additional actual classifier witness
verified fitted shallow-copy reuse, two combined predictions, 12 arrays and
eight child receipts per host. Earlier source-bound tests cover unchanged
precision, lifecycle and failure mechanisms; they are not counted as freshly
executed final-candidate cases. Actual saved records were independently audited.
These isolated results do not substitute for validation of the integrated source.

Earlier observer/readback corrections and real transport failures are retained,
including a Windows long-path fixture failure. The earlier four-arm bridge panel
lacked its specified standalone complete preflight; that sequencing deviation
remains disclosed and is not repaired retroactively by the correctly ordered
final panel. A diagnostic correction replaced collision-prone clock filenames
with atomic unique-file creation, without changing numerical calculations.
This panel assessed the isolated implementation.

### Native inventory, transport and classifier import refinement

This later refinement corrects prepared-QR rejection when parent and worker
processes enumerate identical native libraries in different orders. Canonical
ordering retains every field and duplicate record, with fresh process,
controller, basis and numerical-context checks. Each immutable factor buffer
up to 1 MiB stays as bytes during transport; larger buffers retain read-only
mapping and the existing constant Joblib threshold. Solver arithmetic,
weak-owner lifetimes, model-owned state and current-model persistence remain
unchanged.

Classifier initialization now occurs on first public access, allowing internal
projection workers to avoid unused classifier setup. The actual class and its
module identity are unchanged; numerical MU/rowwise captures remain eager.
Classifier-only dependency failures, class definition and decorators occur
later. Private aliases can observe replacements made before that first import;
arbitrary private monkeypatch timing is outside the estimator contract. Public
imports, reload, concurrent access, failure/retry, copy/clone, persistence and
fresh process transport were checked. See [package imports](api.md#package-imports).

The isolated comparison used the then-current package after scalar identifier
refinement, not released 0.4.3. Each host ran four qualification workers and
36 clean workers covering the original six synthetic histories and faces, with
all six orders of two identical-current controls and the selected source.
Cold first-query means improved by 38.779–48.154 ms on Windows AOCL and
10.123–18.090 ms on Ubuntu MKL. These are workload-specific results, not a
promise of faster calls. Ubuntu's rank-64 whole result lay between its two
current controls; the Windows faces whole increase of 2.784–4.481% was retained.
Module and exact-source-location follow-ups did not reproduce a stable faces
regression of that size. They did not identify its native cause or establish
timing equivalence.

Ubuntu's below-gate first query cost 4.720–6.610 ms more despite faster complete
histories. A passive diagnostic observed an extra original 10 ms Joblib
ordered-result polling sleep while its second task was pending. It identifies
that call's waiting mechanism, not every clean sample's cause or event frequency.
No observation time was subtracted from clean measurements.

The existing one-worker default was then compared with two workers on the
rank-16, 256-feature fixture: 1,024 training rows and 512 query rows. Cold fit
plus three predictions averaged 121.912 versus 2,522.572 ms on Windows and
79.165 versus 1,079.266 ms on Ubuntu. Most measured savings appeared in fit,
which includes process startup in the two-worker arm. These are configuration
effects, not patch gains. Windows' second prediction averaged 0.032 ms slower
serial; individual calls were not uniformly faster. Keep the default for this
small workload without inferring a general automatic worker-count threshold.

All-bytes transport and intermediate cutoffs above 1 MiB lacked a mean
whole-workflow advantage in the measured reuse cases. The cold-owner shortcut
and fuller MU-body deferral lacked consistent benefit; required positive
proofs, existing geometry limits and eager numerical captures remain.
The isolated composite passed 173 native/transport cases with 45 subtests and
203 public regression cases per host. Arrays, numerical oracles and diagnostics
passed. The study workspace retains the source and runtime versions, timing
samples and failed prototypes. The [prepared QR contract](api.md#reusing-prepared-qr-factors)
describes eligibility and costs in the released implementation.

## Uniform KNN full-training membership

Shared calibration can avoid repeated full-neighborhood work when an eligible
uniform-weight KNN head requests every training row in its fold. Its vote
histogram then equals the training-label histogram, including ties. The
implementation retains a real native query and qualifies the native numeric,
method and callback context before using that equivalence. Class order,
probabilities, decisions, warnings/errors and validation remain native-compatible.
No user option, estimator default, solver objective or feature-selection rule
changes. The [API contract](api.md#full-training-neighbor-counts) describes
eligibility and report interpretation.

Within the fixed calibration loop, eligible brute queries request the largest
smaller neighbor count plus its boundary neighbor. A sole full-size head still
queries one neighbor. Tree queries keep their original width because full-tree
membership does not establish equivalent rounded pruning for a shorter query.
Unknown versions, custom callbacks/methods, unsupported numerical policies and
unsafe magnitudes retain the existing route. The proof is currently qualified
for scikit-learn 1.9.1 and Joblib 1.6.0; dependency changes can disable this
extra acceleration while preserving ordinary execution.

The caller constructs its operation-owned sequence only when the fitted-head
dictionary contains the full-fold count. This also avoids unnecessary work if
that endpoint's fit failed. There is no persistent histogram or policy cache.
Optional failures preserve original fallback behavior. Query records retain
attempt widths, warnings/errors and elapsed attempts, with shared attempts
charged once; per-head native fallback stays inside prediction timing.
Completed resumes keep historical compute values and still check the source,
runtime and checkpoint contents. Existing checkpoints are not migrated across
this source change.

The isolated core comparison used unchanged Windows AOCL and physical Ubuntu
MKL environments. Its fixed panel contained nine implementations in 18 balanced
process-order blocks per host, plus two opposite identical-source control pairs.
All 332 workers passed scientific-output checks. Each ran 23 adapter cases and
eight actual public recipes, fresh and completed, with one native thread.
For 768 observations, 32 features and three folds with 512 training rows, the
selected core's complete fresh-call mean savings versus the then-current source
were:

| Public recipe | Windows AOCL | Ubuntu MKL |
| --- | ---: | ---: |
| Mixed grid containing the full-fold count | 88.782 ms (21.008%) | 80.089 ms (35.334%) |
| Duplicate-heavy grid containing the full-fold count | 92.042 ms (22.108%) | 75.874 ms (33.612%) |
| Sole full-fold head | 58.350 ms (15.747%) | 47.008 ms (25.135%) |

These are targeted workload measurements, not an overall package speedup.
The indexed historical study grids did not contain this endpoint, so no speedup
is claimed for those records. The core panel retained adverse results: ordinary
raw-grid means rose 1.152 ms on Windows and 0.286 ms on Ubuntu; the Windows
extreme-magnitude fallback rose 0.554 ms (8.055%). Completed-resume increases
included 1.397 ms on Windows and up to 0.913 ms on Ubuntu. Additional proof,
source hashing and truthful query accounting are real work; the elapsed
differences are not wholly attributed to those operations.

The final caller prerequisite was qualified separately with the core helper
unchanged. Its two-host refinement panel and fixed Windows confirmation showed
smaller ordinary setup costs, without establishing an additional general API
speedup. One original Windows process had large fresh and completed delays;
they did not recur in the fixed reversed-initial-order confirmation. All original
observations remain retained, and the historical cause is not established.
Separate panels are not pooled, corrected using controls or relabeled as
measurements of the integrated source.

Alternatives included sorted membership checks, existing-vote certificates,
row and joint histograms, full-query histogram reuse, shorter shared queries,
and validation-only sole-head paths. Public prefit-tree and custom-Joblib cases
exposed real prototype boundary errors; actual tree-metric and callback-context
checks corrected them before selection. A local cache of failed magnitude proof
added state and exclusions without consistent realized benefit across hosts,
so it was not adopted. Validation-only and narrower alternatives offered less
capability or insufficient incremental benefit for their maintenance cost.

The public endpoint benefit justified maintaining compatibility with native
neighbor behavior and operation order. A dissenting assessment favored the
simpler previous implementation on maintenance grounds. Among alternatives
that implemented the endpoint, the reviewers preferred the selected design.
The isolated qualification and timing covered this workload only. After
integration, an exact copy of the live source passed all
319 focused regression cases and 95 additional subtests on each of Windows
AOCL and physical Ubuntu MKL, with no skips or exclusions. The total includes
68 new permanent cases for full-membership decisions, native callbacks and
fallback, ownership, query accounting and checkpoint recovery. Eight public
recipes also passed fresh and completed calls on each host (16 calls each).
Independent saved-record checks verified source/runtime identity, numerical
state restoration, diagnostic accounting and checkpoint contents. These
checks used unchanged installed runtimes and one numerical worker per host;
they do not add a new performance comparison or release-validation claim.

## Early recovery of screening statistics

The public calibration grid provides the option
`calibration_knn_grid(..., screening_statistics_recovery="off")`, accepting
`"off"` or `"early"`. Eligible `early` runs preserve canonical training-fold
screening statistics before the representation and its fold receipt finish.
This extends restart capability to that earlier boundary, including grids with
a single representation. It does not change screening mathematics, the NMF/PCA
objective, candidate selection or the existing representation-fit recovery
boundary. The [API contract](api.md#early-recovery-of-screening-statistics)
defines eligibility, identity requirements and report fields.

The selected design separates JSON accounting authority from its learned-array
payload. Missing or invalid metadata is fatal. With valid metadata, unavailable
learned arrays permit a replacement computation and a separate new event;
original events, consumers and costs are never rewritten. An interrupted
replacement can itself be restored before its receipt exists. Completed-only
reuse checks metadata without reading learned payloads. Original receipt-based
totals remain unchanged in meaning; the separate union removes event costs
already represented by receipts before adding each durable event once, including
events with no completed consumer. Actual uncommitted attempts remain separate.

The focused follow-up identified a real late-error boundary in the isolated T
prototype: a later saved consumer's missing event metadata was rejected only
after earlier missing work had run. The corrected T2 design preflights all
expected existing receipts under the existing lock before new attempts. The
same validator serves final accounting and requires consistent event references,
opposite Boolean performed/reused flags, and finite nonnegative non-Boolean
durations. Original per-fold receipt reads after runtime/source guards remain.
Per-reuse checks alone cannot meet this boundary when an earlier receipt is
missing; reusing preflight-decoded records would introduce a different snapshot
and invalidation contract. Keeping the fresh reads is a correctness and
maintenance decision, not a selected timing cutoff.

The completed isolated investigation used unchanged Windows AOCL and physical
Ubuntu MKL runtimes, with one native thread per worker. Its fixed main panel
included current source, the single-archive candidate, the two-file T candidate
and both candidates' `off` controls. Four source-order blocks balanced pair
direction and mean ordinal; separate identical-source controls retained their
observed variation. The real workload used 8,000 observations and 4,096 features,
one representation and five folds. Ordinary histories made two API calls;
two-interruption histories made four, with each interruption after real fitting
but before its first receipt. T reduced actual statistics fits across the latter
history from seven to five. Summed API wall-time mean differences were:

| T history contrast | Windows AOCL | Ubuntu MKL |
| --- | --- | --- |
| Ordinary versus current source | +308.194 ms | +133.784 ms |
| Two interruptions versus current source | -430.470 ms | -416.593 ms |
| Ordinary versus T's own `off` control | +122.273 ms | +92.470 ms |
| Two interruptions versus T's own `off` control | -494.258 ms | -407.998 ms |

Positive differences are slower. All four T/current-source ordinary pairs were
slower and all four interrupted pairs faster on each host. Controls and all
adverse observations remain retained without subtraction or an assumed noise
explanation. These are original T source-bound measurements, not integrated
public-source or T2 timing. The capability can save statistics work that grows
with observations and features while persisting state proportional to features;
this scaling argument is not an unmeasured large-data speed claim. The option
is selected for that explicit recovery capability and its bounded ownership
contract, not for memory savings or a universal runtime benefit.

The T2 follow-up passed 84 regression cases on Windows and 85 on Ubuntu,
including 25 new consumer-preflight cases per host. Each host also completed 40
fresh-process partial-history qualification phases under both execution orders,
plus two preflight component workers. Healthy resumes, payload loss, metadata
loss with an earlier missing receipt, interrupted replacement publication and
completed-only calls preserved their defined scientific and accounting results.
The corrected authority failures performed zero new attempts, fits or payload
opens. Independent saved-data audits reread full checkpoint evidence and verified
source/runtime/state guards. Qualification observers were separate from clean
performance evidence; these counts are isolated investigation results.

For six receipts, the added preflight's 21 retained observations per mode/order
had median costs of 0.0586--0.0588 ms on Windows and 0.007714--0.008035 ms on
Ubuntu when no fold directory existed. With metadata already loaded, receipt
scan/validation medians were 1.8777--1.9054 ms and 0.742496--0.753065 ms.
Full preflight medians were 3.2652--3.3758 ms and 1.150863--1.159679 ms, including
metadata work shifted from later accounting. That full clock is not wholly
incremental; filesystem caches may be warm. Do not add or subtract these
component values from the earlier whole-history measurements.

The study workspace retains the failed Windows collection check, its unchanged
continuation and the handling of internal pytest links during archiving.
The [recovery contract](api.md#early-recovery-of-screening-statistics) covers
compatible checkpoints under a cooperating writer lock. Migration of old
checkpoints, generic model persistence and power-loss durability are outside
that contract.

Integration validation passed on Windows AOCL and physical Ubuntu MKL. On
each host, all 11 selected modules passed: 195 tests (98 new permanent cases
and 97 existing caller cases), plus 10 subtests, with no skips, expected
failures or deselections. The permanent cases include 15 fresh-process
checks per host for partial recovery, repair, completed reuse, missing
metadata and process locking. Source, runtime and process-state checks
passed. These are behavior-validation results, not a new performance
comparison, accumulated-suite result or release-validation claim.

The initial full runs each recorded 194 passing tests, one failing test
assertion and 10 passing subtests. That assertion incorrectly expected
object-dtype NumPy string scalars to be persisted; the codec deliberately
keeps ordinary computation for that encoding. Only the test was corrected,
production code remained unchanged, and all 11 modules were then rerun on
both hosts. The initial results and the correction are retained.

## CelebA finalization initialization and ownership

The implementation reduces repeated work in the CelebA parameter study's
finalization and optional fitted-outcome recovery. The first full numerical
initialization remains inside an aggregation when it first needs final-fit
recovery identity checks. The `off` and completed-only paths do not perform
that initialization. Later guards observe current runtime, source and
native-file identity. No identity is persisted across operations and checkpoint
verification is not weakened. Unknown private
discovery implementations and changes to guarded state retain the qualified
fallback/reset behavior. The default final-fit recovery policy remains `off`;
core fitting, projection, prediction and scoring algorithms are unchanged.

The selected discovery path skips already-proved negative library-prefix
routes while checking the current registry and keeping fresh recognized
controller construction and metadata. Linux also removes repeated paths within
the current module scan. Finished private controllers release their extra
routing-state reference but retain original native parent links and discovery
behavior. Retained-registration controls showed that a cached negative route
could miss a later custom registration. Original discovery and the selected
forwarding behavior found the new adapter; full parent-rescan failure in the
cached variant depended on discovery order. Restoring this original behavior
is not a memory-only justification. An actual FlexiBLAS backend switch was not
part of that control. No cross-operation code-proof memo is
included: the qualified memo alternative had mixed whole-workflow effects
without a sufficient practical benefit for its additional state and machinery.

An aggregate acquires a reentrant process-local guard immediately before its
first required fit or recovery operation and holds it through aggregate exit.
This owns final fitting, evaluation and their process-global warning/native-limit
scope. Ordinary disjoint callers wait rather than receiving a new rejection. Calls
that only reuse completed results do not acquire numerical ownership. The
existing per-output filesystem lock still rejects competing finalizers for the
same output. After actual contention, source is checked again before an
`off`-mode fit; `all` retains its identity checks. Same-thread nested calls are
reentrant, but callbacks must not synchronously wait for another thread that
needs this ownership. There is no FIFO/fairness promise; process parallelism
remains available. The public usage contract is in the
[recovery guide](celeba-parameter-study-recovery.md).

The isolated investigation used unchanged Windows AOCL and physical Ubuntu MKL
runtimes, with one native thread per worker. The final fixture had 40 training
and 12 validation observations, 4,096 features and 13 models across two
attributes. Its 24 actual CV donor calls were outside aggregate clocks. The
selected source, a qualified memo alternative and the then-current package
were compared in four fixed source-order blocks: current/selected/memo,
memo/selected/current, current/memo/selected and selected/memo/current. These
balance pair direction and mean ordinal, not every exact position. Each host
ran 12 ten-phase workflow workers and 12 independent first-use raw-restoration
workers. All three raw/PCA/terminal first-use recipes were also qualified.
First-use restoration here excludes runtime/bootstrap and CV preparation; it
is not process-startup or cold-native-runtime timing.

The following whole-call mean differences compare the selected source directly
with that current-source reference. Positive values are slower. Percentages
are ratios of arithmetic means; each comparison retains all four pairs.

| Phase | Windows AOCL | Ubuntu MKL |
| --- | --- | --- |
| Fresh, recovery `off` | +30.278 ms (+2.25%); 3/4 slower | -17.060 ms (-1.99%); 4/4 faster |
| Completed, recovery `off` | +8.888 ms (+1.35%); 1/4 slower | -1.079 ms (-0.41%); 2/4 faster |
| Fresh, recovery `all` | -421.731 ms (-11.80%); 4/4 faster | -600.498 ms (-30.54%); 4/4 faster |
| Completed, recovery `all` | -23.350 ms (-3.69%); 2/4 faster | -1.471 ms (-0.55%); 2/4 faster |
| Raw KNN restoration | -6.987 ms (-5.35%); 3/4 faster | -14.672 ms (-18.76%); 4/4 faster |
| Raw KNN completed | -0.109 ms (-0.95%); 3/4 faster | -0.526 ms (-18.17%); 4/4 faster |
| PCA restoration | -9.724 ms (-6.67%); 4/4 faster | -14.161 ms (-17.65%); 4/4 faster |
| PCA completed | +0.391 ms (+1.54%); 4/4 slower | +0.829 ms (+8.31%); 4/4 slower |
| Terminal restoration | -4.509 ms (-4.83%); 4/4 faster | -7.061 ms (-16.34%); 4/4 faster |
| Terminal completed | +0.280 ms (+1.25%); 3/4 slower | +0.098 ms (+1.20%); 2/4 slower |
| Independent first raw restoration | -6.425 ms (-5.31%); 4/4 faster | -14.257 ms (-17.97%); 4/4 faster |
| Independent first completed result | +0.101 ms (+0.99%); 2/4 slower | -0.024 ms (-0.96%); 3/4 faster |

Recorded fit and prediction times are not promised unchanged. Windows fresh
`off` added 5.760 ms in the fit clock and 3.068 ms in prediction; fresh `all`
added 8.639 ms and 0.724 ms respectively. Ubuntu fresh `all` prediction added
1.514 ms, with mixed signs. Windows first raw restoration added 1.053 ms to
prediction and 0.643 ms to verification, each slower in all four pairs, despite
the shorter whole restoration. These nested clocks are not an exclusive
partition of wall time. Restored fit and completed nested clocks are historical
and are not counted as newly performed work. All phase/clock comparisons remain
in the source-bound report rather than being pooled with earlier panels.

Twenty bounded diagnostic workers subsequently passed: eight natural GC
audits, eight corrected first-use timer audits and four Windows off-path scope
audits. Ubuntu completed-PCA showed a collection-generation placement mechanism;
the Windows diagnostic did not reproduce the same adverse mechanism. No GC
collection overlapped the observed first-use prediction or verification timer
windows. The unchanged timer bodies and before-optimization controls did not
identify a promising added prediction/verification operation. The Windows
scope observer measured direct added ownership at about 0.093 ms per fresh call
and 0.010 ms per completed call, including callback overhead. A common-ancestor
partition retained existing NumPy context-manager work and localized larger
observed variation to existing fitting, evaluation, JSON writing, report reads,
digests and hashes. Closure/branch work remained in the aggregate residuals.
These findings do not reconstruct the exact external cause of every earlier
clean interval, establish globally optimal I/O, or remove the measured costs.
No forced collection, GC-policy change, warm-up relocation or observer-time
subtraction is included. Two failed observer dispatches and their narrow
line-event correction remain preserved, not counted as passing observations.

Here, clean timing means uninstrumented execution, not an otherwise idle
machine. The original Windows natural-workflow saved-data audit, about 10--20
seconds, overlapped some independent first-restoration workers. Natural saved
audits, about three seconds, overlapped the earlier failed observer version,
not its corrected workers. Every original measurement is retained without
exclusion, correction or an assumed explanation from that overlap.

A narrower per-method ownership alternative allowed a mixed partial-resume
caller to finish about 350 ms sooner on Windows and 159 ms sooner on Ubuntu,
while extending the other caller and retaining ordinary/equal-work latency
costs. Whole-operation ownership was selected for its simpler successful-call
contract and complete tradeoff evidence, without assuming workload frequencies.
Similarly, a completed-path speed difference where a memo is inactive is not
evidence of a direct memo saving. The finite candidate-specific alternatives
are closed; this does not claim every possible algorithm has been exhausted.

The selected isolated source passed 174 focused regression cases on Windows
and 184 on Ubuntu, including the unchanged 73-case final-recovery caller module.
The memo comparison passed 178/188. Exact scientific outputs, genuine committed
outcome reuse, original fit metadata, completed records and source/runtime/state
boundaries were checked. These are investigation results, not a claim that
integration validation, the accumulated suite or release checks have passed.
The [recovery guide](celeba-parameter-study-recovery.md#measured-costs-and-limits)
reports the measured costs and limits for users choosing a recovery mode.
The detailed comparison of alternatives is retained locally.

After integration, the focused permanent regressions passed 152 cases on
Windows AOCL and 162 on physical Ubuntu MKL, including 79 and 89 new cases
respectively and the unchanged 73-case final-recovery caller module. Four
actual workflow/recovery workers per host also passed: the ten-phase workflow
and independent first-use raw, PCA and terminal recovery. Source, runtime and
process-state preservation checks passed, and all 23 core package source files
remained byte-for-byte unchanged. This was focused behavior validation, not a
new performance comparison, an accumulated test-suite run or release validation.

## Exact identifier evaluation and group bookkeeping

The implementation reduces identifier work around evaluation, without changing
model fitting, the NMF objective, projection, prediction algorithms or public
options. Validation and returned scalar IDs use the same owned numeric capture.
Sample uniqueness data and group partitions are reused for overlap and support
counts. Sorted native comparisons avoid rebuilding exact keys where possible;
mixed formats use a common floating-point comparison representation only after
precision and exponent/subnormal-range proofs, plus integer-format or actual
integer-range bounds, establish exactness. Unknown or unsafe promotion keeps
the exact canonical fallback. Original report scalars, numeric equality,
validation order and subclass callbacks retain their contracts.

The selected group refinements combine a bounded canonical-key route for
eligible totals of at most eight rows with a uniform-native shortcut. The
uniform check first compares endpoints; different endpoints avoid a full
comparison allocation, while matching endpoints still require checking all
values. Uniform inverse codes preserve the existing per-class counting rules.
The small route admits ordinary array labels only, so unusual private masks
retain their prior behavior. The cutoff is a fixed engineering choice, not an
estimated universal crossover. The bulk native-key fallback on platforms where
long double fits binary64 uses a real float64 copy: equivalent storage dtypes
can retain different scalar classes when a conversion is elided.

The isolated investigation used unchanged Windows AOCL and physical Ubuntu MKL
runtimes, with one native thread per worker. It compared exact native/group
counting methods, mixed-format comparisons, proven lossless promotion, narrow
bulk key conversion, ownership/lifetime alternatives and the small/uniform
refinements. The last fixed panel used nine source labels, 28 cases and four
source-order blocks, including two independently seeded orders and their
reverses: 36 workers and 2,016 public-plus-serialization calls plus 2,016 later
metadata calls per host. The identical-source label remained separate. These
orders balance pair direction and mean ordinal, not every exact position.
Earlier panels, current-relative costs and failed harness attempts remain
separate records; their deltas are not pooled or added to the final panel.

The following final-panel means compare the selected small/uniform combination
with its qualified exact-comparison parent, which lacks those two group
refinements. They are not fresh measurements against an earlier released
package. On Ubuntu, 2,000 evaluation/8,000 training rows with uniform groups
saved 0.200/0.202 ms in first/repeated public calls and 0.198/0.239 ms in complete
transactions; all four pairs were faster in each comparison. At 10,000/40,000
rows, metadata saved 1.459/1.379 ms, all four pairs; public calls saved
0.844/1.877 ms, with mixed first-call signs and all repeated calls faster.
Mixed-eight inputs with native training group IDs saved 0.175/0.203 ms in public
calls and 0.195/0.201 ms in transactions, all four pairs. The opposite
orientation had consistent metadata savings but mixed public effects.

Costs remain. Ubuntu nonuniform-2,000 metadata added 0.032/0.026 ms, and a
large array with equal endpoints but a different middle added 0.040/0.019 ms.
Ubuntu no-ID public calls added 0.191/0.065 ms. Windows string public means
added 1.660/0.446 ms, with mixed signs. Windows Fashion first public evaluation
added 9.245 ms, with block deltas -0.611, +2.470, -1.851 and +36.974 ms; 7.427 ms
of the mean difference was inside the recorded prediction interval. Windows
Fashion repeated transactions added 32.395 ms while the separately recorded
write interval added 33.821 ms, including one +135.411 ms write delta. CelebA
first/repeated transactions added 24.455/14.666 ms, with corresponding write
interval differences of 24.305/24.958 ms. Those clocks localize observed work;
they do not establish the underlying external cause or justify excluding it.
An interval crossing zero, an inactive branch or the identical-source control
does not establish zero overhead. No control subtraction or assumed workload
frequency is used to select the implementation.

A bounded same-native scalar-set alternative was correct and saved roughly
10–44 microseconds in some direct metadata comparisons. It did not provide a
consistent whole-public advantage over the simpler selected combination and
retained broader costs, so its extra route was not selected. Earlier native
identity release likewise lacked consistent metadata/transaction benefit.
These decisions weigh exactness, practical benefit, maintenance and the full
cost evidence; they do not trade execution speed for memory savings.

Separate causal controls kept the public code and fitted models fixed while
changing only bound metadata engines. Natural first-visit costs chiefly
followed the unchanged transform/context in those controls; prewarming would
relocate work without a demonstrated complete-workflow saving. An intervention
using the original evaluator reproduced particular long delays associated
with garbage-collection placement and accounted for the deferred collection
cost. It did not identify the cause of every older uninstrumented event.
No garbage-collection policy, global warm-up, measurement adjustment or native
runtime change is included in the patch. Some historical delays remain
unattributed, and new write-window delays have localization rather than a
proved external cause.

The isolated source checks cover exact mixed signed/unsigned/float identity,
large integers, true-wide values, signed zero, empty and one-class masks,
ownership, dtype metadata, strict error/warning policies and callback/error
order. They support the selected implementation on the declared runtimes;
they do not promise all-workload acceleration or a globally optimal threshold.
Production integration checks, permanent regressions, accumulated validation
and release checks remain distinct; this section does not report their results.
See the [identifier contract](api.md#identifier-validation-costs) for public
behavior and the historical section below for the earlier WSL work.

## Screening statistics for Fortran-order inputs

The screening implementation combines bounded statistics blocks for large
canonical Fortran-order inputs with C-order output from the already-required
float32 widening allocation for smaller eligible inputs. Both require more than
128 columns; the translated float64 size separates the paths at 1 MiB, with
blocking strictly above that boundary. The full minimum-subtraction snapshot,
class row order and reduction geometry are retained, including a 127-plus-2
column tail instead of a one-column block. Strict-policy and layout fallbacks
are described in the [API contract](api.md#screening-execution-layouts). No public
option, selection policy or transformed-value scaling is added. The gates are
fixed engineering policies rather than an estimated universal crossover.

The final isolated comparison used Windows AOCL and physical Ubuntu MKL with
one native thread. Each host completed 18 source processes in all six orders of
current, gated-blocking and the selected hybrid source, covering 16 cases and
288 timed fit-plus-transform sequences. Wide 16,000-by-4,096 F32/F64 inputs
averaged 16.43%/17.19% faster on Windows and 22.83%/22.56% faster on Ubuntu than
current, with all six pairs faster in each comparison. Percentages are ratios
of arithmetic mean times. The 512-by-255 F32 case saved 0.527 ms on Windows and
0.408 ms on Ubuntu against the simpler gated-blocking parent; the 512-by-256
case saved 0.262 ms and 0.440 ms, with the Windows interval crossing zero.
Ubuntu's complete Fashion F fit-plus-transform workflow saved 19.100 ms (3.36%),
with all six pairs faster. Windows Fashion F averaged 85.744 ms faster, but its
interval crossed zero and included one 502.314 ms saving; that mean is not a
stable expected workflow gain. Consumer predictions were checked separately
and excluded from these clocks.

Costs remain part of the decision. The 2,048-by-128 F32 Ubuntu bypass averaged
0.407 ms slower (9.82%) than current, with five of six pairs slower and an
unadjusted 95% paired interval of [-0.360, +1.174] ms. Both chronological halves
and source-order strata were adverse. Windows 64-by-17 and 512-by-129 F32 cases
averaged 0.101 ms and 0.077 ms slower, respectively, with intervals crossing zero.
Windows 512-by-257 F32 transform alone cost 0.034 ms more, with all six pairs
slower, although its complete call averaged 0.100 ms faster. Windows Fashion C
was 10.690 ms slower than the gated-blocking parent, all six pairs slower, with
interval [-7.785, +29.165] ms; its comparison with current was instead favorable.
These comparisons are retained separately. Neither fallback arithmetic nor an
interval crossing zero establishes zero overhead. Earlier adverse and
identical-source observations remain without noise exclusions or subtraction.

The decision weighs preserved public numerical/error contracts, the substantial
existing-F benefit, the small widening branch's direct benefit and maintenance
cost. It does not rely on memory savings or assumed workload frequencies.
Typical CV row gathers are C, and screening statistics can be reused across
candidates within a fold. Preparing wide F arrays cost about 0.99–1.07 seconds
for F32 and 3.23–3.59 seconds for F64 in the final panel, outside public clocks;
these results do not recommend conversion merely for speed. Alternative
subtraction/normalization layouts and the per-block C composition were measured
and not selected. The narrow two-feature grouping alternative had genuine
41–68% selector gains, but changed reductions could alter support and
zero-feature failure at legal cutoffs, without uniform numerical-accuracy
improvement. Retaining the existing public selection/error policy motivated
its nonadoption; historical-record reproduction was not the criterion.

The selected isolated source passed 176 focused tests and 39 subtests per host.
Qualification covered 53 candidate fits, 12 fresh current references, a cutoff
probe and five cache calls per host, with exact required comparisons. Independent
review of all final timing outputs found no changes in statistics, support,
selected values, warnings/errors, predictions, probabilities or convergence in
the declared semantic comparison. Fashion remained 1,567/2,000 correct with five
SPA iterations. Earlier full raw CelebA checks qualify the unchanged large-block
mechanism as inherited evidence, not a newly executed final-source workflow.
The study workspace retains the raw comparison against the isolated source.
The [screening layout guide](api.md#screening-execution-layouts) describes
which input layouts use each calculation. Integrated-source and release
validation are separate from these measurements.

## Experimental ANLS factor and report copying

The change reduces buffer-copying work in the experimental ANLS-BPP
implementation. One bounded helper serves both publication of W and the final
returned coordinates, preserving independent C-order arrays. On eligible
Fortran-order float64 factors, an integer-view traversal copies the exact bits
into the required C array. The existing final reconstruction report has a
separate bounded four-column copy into its required padded left workspace.
The final native matrix product and norm remain unchanged. Solver iterations,
stopping, fitted-factor semantics and public defaults are unchanged; no public
option is added. The [copying contract](nmf-solver-comparison.md#experimental-anls-factor-and-report-copying)
describes the eligibility and allocation-failure boundaries.

The decision favors these bounded copies over broader traversal, F/K-order
publication, raw-private-factor return and borrowing fitted public storage.
Those alternatives incurred consumer conversions, lacked a useful complete
workflow advantage, or changed ownership and mutation behavior. The selected
composition preserves the existing public layout while sharing one copy helper
between publication and return. The decision is based on reduced copying work,
preserved contracts and maintenance cost; it is not justified by memory saving.

The final isolated workflow comparison used Windows AOCL and physical Ubuntu
MKL with one native thread. Each host completed 76 workflows: 12 qualification
calls and 64 timed calls, with current, selected C-publication, K-publication
and identical-current labels. The four cases covered ordinary inputs, the
8,192-row/rank-20 trigger, and fixed three-iteration Fashion/CelebA recipes.
Current-to-selected mean complete-workflow changes at the trigger were
-10.018 ms on Windows and -31.396 ms on Ubuntu. Both pointwise paired intervals
crossed zero, as did all eight selected-versus-current total-time intervals.
These measurements do not establish universal whole-workflow acceleration or
timing neutrality.

The adverse observations remain part of the decision. Windows CelebA fitting
averaged 33.021 ms slower, with all four pairs slower; its complete workflow
averaged 47.651 ms slower, with three of four pairs slower. The intervals were
[-14.643, +80.685] ms for fitting and [-30.688, +125.990] ms for the complete
workflow. The earlier direct-report-only Windows trigger averaged 49.436 ms
slower (+2.12%). Identical-source controls are retained separately, without
subtraction, noise exclusions or claims that they explain away candidate costs.
The 8,000-row natural cases bypass the bounded factor-copy traversal, so these
costs cannot be assigned to its active branch from the timing data alone.

Eight additional full-natural correctness workflows compared current and
selected implementations on Fashion and CelebA on both hosts. Within each
host, fitted factors, histories, final reports, query outputs, classifications,
probabilities, neighbors and consumer state matched exactly. Fashion converged
after 223 iterations and classified 1,638 of 2,000 query rows correctly;
CelebA converged after 70 iterations and classified 3,254 of 4,000 correctly.
These correctness runs complement the admitted-copy and failure-boundary checks;
their elapsed times are not extra benchmark samples.

These results qualify the isolated source. The study workspace retains the
source identities, adverse observations and failed harness attempts.
Integration validation is tied to the production files and permanent regressions.
The [ANLS copying contract](nmf-solver-comparison.md#experimental-anls-factor-and-report-copying)
describes which copies are eligible.

## Experimental ANLS training-state reuse

The solver reuses a successful H solve's completed residual and, when
its layout permits, H gradient in the same outer training step. It preserves
native operation/layout guards, factor consistency checks, complete quiet
finalization with caller-policy replay, and the original outer W-gradient
calculation. Rank-one inputs with at most 16,384 elements bypass this handoff.
The fixed gate avoids optional work on inexpensive products; it does not alter
stopping criteria or add a public setting. The
[solver contract](nmf-solver-comparison.md#experimental-anls-training-state-reuse)
sets out the scope and fallback behavior.

The final isolated follow-up used the existing local Windows AOCL environment
and physical Ubuntu MKL environment, with one native thread. Each fresh process
loaded one source and performed one fit, query projection and external KKT
calculation. Per host, 15 qualification calls preceded 180 clean calls: twelve
rounds of three diagnostic recipes with baseline, ungated, gated and identical
ungated-source labels, plus six rounds of the two original natural recipes with
baseline, ungated and gated sources. Natural training used 8,000 rows at ranks
5 and 20, with complete query sets and `max_iter=1000`; Fashion and CelebA
actually converged after 223 and 70 iterations on both hosts.

Compared with the pre-change baseline, the selected gated source reduced mean
fit time on the 65,536-by-16 rank-one case by 525.5 ms on Windows and 205.3 ms
on Ubuntu (geometric paired changes of -10.49% and -11.02%), with all twelve pairs faster
per host. Windows full-CelebA fitting fell by 1.342 s (-3.27%), with five of six
pairs faster and a wholly negative pointwise mean interval. On Ubuntu small64
(64-by-64, rank one), the gate saved 4.579 ms (-4.05%) versus the ungated proposal,
with all twelve
pairs faster; it also beat the independently scheduled identical-source label
in all twelve pairs. These are solver measurements, not classifier accuracy
or full study wall time. Both rank-one recipes completed 128 outer iterations;
short CelebA below used rank 20 and three outer iterations.

Costs remain. Windows small64 was 3.897 ms slower than baseline on average
(+2.33%), with median +4.456 ms and eight of twelve slower pairs; its interval
crossed zero. Relative to the ungated proposal, Windows tall fitting averaged
+12.854 ms and short CelebA fitting +79.420 ms, both with intervals crossing
zero. Short CelebA's summed fit/query/diagnostic cost was +109.410 ms; its query
projection alone cost +29.372 ms, with pointwise mean interval
[+2.720, +56.024] ms. The query comparison against baseline was instead
+8.150 ms with an interval crossing zero. Ubuntu short CelebA had near-zero
mean fit change versus the ungated proposal but an adverse median +19.078 ms.
The four full-natural gated-versus-ungated fit means and medians favored the
gate, but every interval crossed zero and several chronological halves reversed.
No zero-cost, equivalence or universal speedup claim follows.

The identical-source controls were retained separately, without subtraction or
use as a tolerance. For example, Windows short-CelebA query time differed by
+11.571 ms even between identical ungated sources, with a positive pointwise
interval; this does not identify a cause or erase the gated comparison. Orders
and positions were balanced overall, but chronological halves were not
independently balanced replicates. The decision considers the bounded work
reduction, preserved behavior, maintenance cost and uncertain gains/costs; it
does not assume workload frequencies or justify slower work by memory savings.

Independent saved-result reconciliation passed all 390 calls, 240 five-clock
comparisons and 1,170 archived array payloads with no findings. Within each host,
the tested factors, fitted state, histories, stopping events, warnings and query
projections matched the baseline. Constructor-plus-fit overlaps fit; total is
fit plus query plus external diagnostic, excluding startup and bookkeeping.
These results qualify the isolated investigated source, not a release or the
subsequently integrated checkout. Integration validation is a separate batch
bound to the actual source, permanent tests and documentation.

The study workspace retains the reconciled arrays and timings, earlier
co-resident comparisons, failed attempts and source identities. The follow-up
ran on physical Ubuntu. Its timings do not explain the differences observed
in the earlier WSL study. See the [ANLS training-state reuse contract](nmf-solver-comparison.md#experimental-anls-training-state-reuse).

## Exact rank-one repeat indexing

The experimental ANLS-BPP implementation replaces the full-factor
BLAKE2b index with CRC32 for the existing rank-one, positive-tolerance repeat
policy. Contiguous accepted W and H factors are scanned in order. Shape/dtype
keys and complete bit equality still determine a repeat and its first matching
iteration; CRC equality alone cannot stop a fit. Other layouts retain the Brent
checkpoint route. The convergence criterion, warning behavior and existing
retention of accepted factor arrays are unchanged. No public solver choice,
parameter or opt-in is added. CRC collisions can increase comparison work and
cost; the index is not collision-free.

The isolated comparison favored this small index change over sampled/promotion
and copied-byte alternatives: useful gains in some geometries did not establish
a common whole-fit advantage for their additional state and allocation work.
In eight balanced process blocks, mean continuous constructor-plus-fit time on
the fixed tall-progression case fell by 73.4 ms on Windows AOCL and 107.2 ms on
WSL MKL (geometric paired changes of -1.43% and -4.50%). Ordinary cases were
mixed. In the separate six-block full-Fashion comparison, the Windows mean fell
by 260.1 ms (-0.272%), while the WSL mean increased by 590.4 ms (+0.704%). The
WSL difference interval crossed zero; that does not establish zero cost. These
cohorts are not pooled and the results do not imply a universal speedup.

The isolated qualification and scientific comparisons preserve the tested
factor bits, stop events, histories, warnings and query projections. They do
not validate the integrated source or a release: integration validation is a
separate source-bound batch. Historical studies and their original source
identities remain unchanged. The current contract and retained-memory costs
are described in the [experimental repeat policy](nmf-solver-comparison.md#experimental-anls-stopping-and-repeated-iterates).

## Linux multiworker CelebA checkpoint startup

This extends the earlier one-effective-worker startup decision below; that
investigation's measurements and historical records remain unchanged.
The change is confined to `scripts/celeba_default_study.py`. Its
Linux command-line entry point may select `forkserver` when more than one worker
is requested and every submitted chunk has identity and report files. The check
runs after the existing completed-work shortcut and selects a process context
only. It neither reads receipt contents nor authorizes reuse. Original workers,
source/runtime/data identities, checkpoint validation and numerical algorithms
remain in place; a missing fitted fold is recovered normally, and a corrupt
checksum remains an error.

The owned CLI pool uses an explicitly empty forkserver preload list. No global
start method or new public option is introduced. New output directories and
missing identity/report files retain `spawn`; ordinary hint failures also choose
that route. Imported defaults, non-Linux calls, all requested-one-worker paths
and the parameter-study launcher keep their existing routing. This does not
provide an atomic filesystem snapshot or qualify arbitrary startup hooks and
preexisting custom forkservers.

The decision follows fixed local WSL comparisons and boundary checks. The final
source reduced mean parent elapsed time by about 198 ms for completed real-image
runs, 371 ms for matched real-image recovery and 217 ms for matched synthetic
recovery. The completed-run adverse pair and all other observations are retained.
Earlier recovery comparisons that selected different logical folds are excluded
from causal speed claims. The final-source fresh panel initially averaged 217 ms
slower; a separate fixed eight-pair follow-up averaged 23 ms faster, with both
order means negative and three adverse pairs. Actual diagnostic worker entries
used `spawn`. Neither these results nor unchanged numerical code establish zero
overhead or identify the exact cause of the earlier difference. A common-bootstrap
intervention had its own measurable cost and was not substituted for ordinary
CLI evidence or subtracted as a calibration.

The timings above apply to the isolated source. Checkpoint reuse requires
matching source identities. The [startup guide](celeba-study.md#process-startup-for-current-runs)
describes the released routing behavior and the workloads covered by the timings.

## Implementation-source discovery for reusable searches

The implementation reduces redundant source-file discovery in the
private identity encoder used by calibration and checkpoint reuse. Within one
eligible ordinary class-hierarchy traversal, it reuses resolution of the same
literal filename. An existing plain Python source file can be located directly
through its ordinary module namespace, avoiding repeated inspection before the
required file check. This retains one encoder and adds no public option,
persistent cache, metadata parser or worker pool. The decision rests on the
bounded work reduction, preserved contracts and maintenance cost, not memory
savings or an assumed workload frequency.

Each later source check still creates fresh encoder state and hashes current
file bytes. Resolution reuse cannot cross the class traversal or a subsequent
parameter callback. Full configuration encoding, clone and parameter validation,
source/runtime guard locations, checkpoint integrity, locks and publication are
unchanged. Custom namespaces, unusual or missing paths, functions and replaced
inspection predicates retain the original discovery path. The optimization
preserves the documented [source-identity contract](api.md#reusable-candidate-and-checkpoint-identity);
it does not provide an atomic filesystem snapshot or guarantee identical
low-level stat histories under transient failures or asynchronous file changes.

The isolated investigation compared a separate source-only value visitor,
metadata-version parsing, parallel source reads and narrower discovery reuse.
The visitor duplicated validation logic without a convincing broad benefit;
the metadata shortcut added ongoing compatibility policy for modest gains;
parallel reads substantially increased the measured WSL workflow cost. The
selected change retains fresh validation and removes only the qualified
redundant discovery. These alternatives and their favorable and adverse
observations remain part of the evidence.

In the fixed small-workflow and checkpoint panel, the sum of thirteen timed
public calls improved by about 12.45% on Windows and 2.49% on WSL. The WSL sum's
interval included zero. All three source-only microbenchmarks improved in every
paired observation on both systems. These measurements do not establish that
the selected implementation universally outperforms the simpler resolution-
only alternative, which was measured in a separate comparison.

The larger natural-workflow panel was mixed: eight-call sums changed by about
-0.18% on Windows and -0.62% on WSL, with both intervals spanning zero. Windows
Necktie uniform and distance fresh calls had adverse means of about 84 ms and
44 ms; Necktie uniform resume added about 25 ms on Windows and 15 ms on WSL.
An existing Windows Necktie uniform screening clock was about 20 ms slower in
all six pairs. No mechanism was assigned to that component difference. All
pairs, order effects and separate same-source controls were retained; controls
were not subtracted and component clocks were not combined into a residual.
The adoption rationale is a targeted reduction of guard work, not a universal
whole-workflow speedup or a claim that every remaining cost is unavoidable.

The isolated source passed 88 external differential tests with two symbolic-
link privilege skips on Windows, and all 90 on WSL. The eight existing targeted
identity, provenance, checkpoint, study and rank-search modules passed 134 tests
with two platform skips and 12 subtests on Windows, and 136 tests with 12
subtests on WSL. The Windows regression skips covered POSIX creation permissions
and unavailable directory symlinks. Fresh, completed and recovery qualification
preserved the declared scientific outputs, warnings and work. The runs used
the installed Windows and WSL runtimes sequentially, with a total budget of
eight compute threads. The final investigation audit found all 1,089 included
live files and their inventory unchanged; historical experiments were not
rerun or requalified by this work.

The counts above cover the isolated source. The study workspace retains
separate integration results for the production code and permanent tests.
The [0.5.0 release notes](release-0.5.0.md#performance-and-validation)
summarize release-level validation.

## Guarded neighbor reuse in explicit grids

The implementation reduces repeated finite-distance qualification,
boundary checks and vote work in `calibration_knn_grid`'s private Euclidean
neighbor-query adapter. It qualifies the complete shared matrix once, constructs
reusable votes only at requested clean-boundary endpoints, and reuses a boundary
fact already established by a completed vote cache. This decision rests on the
existing immutable-array lifecycle, numerical guards, measured work reduction
and limited maintenance cost, not memory savings alone. There is no new public
option or change to ordinary classifier prediction outside this adapter.

Nondefault NumPy error policies retain the original calculation order. Optional
cache failure publishes no partial result. Zero-distance weighting, vote and
threshold uncertainty checks, and the whole-original-batch native selected-k
fallback remain. Early boundary fallback explicitly marks vote diagnostics as
uncomputed. The summation argument assumes fixed nonnegative weights, ordinary
rounding and gradual underflow; it does not prove bitwise weighted-probability
identity or identical native neighbor membership. See the
[API reporting contract](api.md#explicit-grids-with-shared-representations).

The isolated investigation qualified numerical/error-state boundaries and
compared native public outputs, fresh calibration, own-source resume and
recovery in the existing local Windows and WSL runtimes. It examined simpler
per-count calculation, alternative vote reuse, duplicate-run routing, sampled
queries and exact-coordinate lookup. Some routing alternatives had substantial
niche gains but also discarded useful sharing on clean later neighbor counts;
they were not adopted as general defaults. The retained native fallback can
still require both an initial shared query and subsequent per-count queries.

Final component comparisons reduced duplicate-CV work by roughly 7–10%, LFWA
work by 4–12%, and a many-head case by 23–25%, but complete-calibration timings
were mixed. An original WSL CelebA call was 6.51% slower, about 243 ms. A fixed
comparison with the same package source and all adapter implementations loaded
did not reproduce that late penalty; it neither identified its cause nor
established universal timing neutrality. Faster prediction work did not always
reduce combined KNN or whole-call time. All costs, outliers and order effects
remain in the evidence; no controls were subtracted or samples trimmed.

These are bounded implementation observations, not new classification-accuracy
evidence or a claim of the fastest possible KNN engine. The investigation
qualified the isolated implementation. For the scope of the integrated
checks, see [0.5.0 performance and validation](release-0.5.0.md#performance-and-validation).

## Completed-resume startup and exact identifiers

This section records the earlier combined implementation and its Windows/WSL
evidence. Its wide-key conversion policy is historical; the later native
comparison and group refinements are described in
[exact identifier evaluation](#exact-identifier-evaluation-and-group-bookkeeping).
The completed-resume policy below is a separate change.

The implementation preserves exact identifier comparisons while using
owned numeric snapshots, reusing validated sets and deferring native-format
discovery. Only proven exactly representable native values use Python-float
keys; wider values retain exact rational keys, with original report values
preserved. The complementary serializer shortcut handles exact builtin types
and retains the existing subclass and extended-scalar paths.

The two CelebA launchers may validate completed CV work in one fresh empty-context
thread on Linux when only one worker is effective. Actual numerical-runtime and
execution-state qualification, source/data identities, checkpoint locks and
authoritative validation remain required. A report-presence hint can only avoid
an unnecessary attempt; it cannot accept a checkpoint. Missing work stops reuse
in order, restores state and falls back to the original full process schedule.
Multiworker parallelism and non-Linux routing remain unchanged. No public option
or relaxation of historical source-bound resume is introduced.

The isolated investigation compared credible alternatives, including broad
serial/threaded reuse, forkserver, eager initialization and a simpler lazy
conversion policy. The selected identifier guard retains a measured wide-key
conversion cost in exchange for a larger ordinary metadata-workflow benefit;
the simpler policy was tested and not selected. Startup fresh/recovery timing
was variable. Neither unchanged numerical code nor small aggregate differences
establish universal timing neutrality. Public cost scope is recorded in the
[startup guide](celeba-study.md#process-startup-for-current-runs) and
[identifier guide](api.md#identifier-validation-costs).

Integration combined the independently qualified edits, removed experimental
activation and tracing, and limited the new parameter-worker mapping cleanup
to Linux. The focused checks were designed to cover ordinary fallback, context
restoration, exact serialization, concurrency, resume with matching source and
rejection of incompatible source. The timings above apply to the investigation.

## MU projection default and recovery

The 0.5.0 default is `mu_projection="auto"`, resolved at fit
time to rowwise for MU and no MU policy for SPA/CD. The fixed-basis unregularized
Frobenius objective separates by observation; deliberately shared stopping can
change a prediction when unrelated query observations change. The default is
chosen for that predictor interpretation, not an assumed public workload mix or
historical reproduction. Explicit batch projection remains available. The SPA
solver default and MU/CD joint-fit training-coordinate convention are unchanged.
Fitted policies survive constructor changes and current-format persistence until
refit; requested and resolved policies remain distinct checkpoint identity fields.

The selected implementation combines guarded ordinary-row execution, a singleton
shortcut, layout-safe logarithmic panels and independent vector recovery for
eligible float64 rows. Each row retains its initializer and stopping state.
A qualified capped attempt retains completed rows before the first capped row;
that row and the remaining suffix use ordered scalar replay. Other declined
attempts retain full serial replay, and short positive-tolerance caps bypass
speculation. Guarded setup can be reused within the call, including from the
vector attempt to its first eligible scalar row; it is cleared before warnings
or unknown callbacks. This preserves observable warnings, errors and basis-mutating
callbacks within the qualified contract, without adding a public option.
It does not establish universal bitwise batch invariance, certified NNLS accuracy
or better classification accuracy. Analytical shortcut arguments remain subject
to their stated floating-point and reduction assumptions and empirical checks.

The earlier default-selection investigation used existing local Windows and WSL
runtimes sequentially, normally at one native thread; concurrency controls used two
one-thread workers. Windows contains NumPy MKL and SciPy AOCL/BLIS, not an entirely
AOCL-linked stack. Both environments passed the combined focused 540-test suite
plus 23 subtests. Separate recovery boundaries covered per-row stops, layouts,
range failures, strict warnings, callback mutation, publication and concurrency.
Actual Fashion calibration, selected fitting, evaluation and completed-CV reuse
confirmed auto/explicit-rowwise scientific parity and distinct requested-policy
identities. Resume forbade new fitting/projection/query work. Existing source and
runtime files and original experimental records remained unchanged.

In the difficult 2,000-query true-float64 Fashion cohort, selected rowwise
transform took 19.06 s on Windows and 14.97 s on WSL, versus 88.05 s and 72.86 s
for the optimized serial-panel reference. One immutable reference transform per
backend supplied coordinates and head-only label/probability oracles; no full
serial public-predict timing is inferred. Actual selected plain prediction took
19.03/14.48 s, versus optimized explicit batch's 18.02/13.66 s. Rowwise coordinates,
outputs and diagnostics excluding timing fields matched the serial oracle exactly.
Rowwise accuracy/BA was 83.5%, versus batch's 83.7%; this is a policy difference,
not an optimization regression. These full calls were single observations on
development training-pool data, not an untouched-test evaluation.

Measured ordinary singleton requests still cost about 0.26–0.47 ms more than
batch; a four-row cap11 replay case added 2.4–2.8 ms (27–29%). No universal timing
bound is inferred. That investigation examined larger panels and a separate
shared-basis cache but did not select them; measured speed and maintenance, not
memory savings, drove that choice. The later scalar-suffix decision below
addresses a different reuse opportunity. Preserved failures include outdated
test oracles, a layout-sensitive rounding difference, a preflight basis-hash
error, and an observer comparison of
discarded speculative traces. Additive corrections retain their originals and
do not turn failed raw receipts into successful runs.

The study workspace retains the source maps, failures, source-transfer records
and integration checks. The [MU prediction policies](api.md#prediction-batches)
describe public usage and costs. The investigation measured the selected
workloads.

### Interrupted MU recovery and panel refinement

The refinement retains a qualified completed prefix, skips the final
speculative updates that cannot produce an accepted result under admitted
positive-tolerance controls, and reuses guarded basis setup inside the remaining
scalar recovery. Reuse ends before warnings or unknown callbacks, even inside a
standard warning recorder. The cache belongs to one call; it does not persist on
a fitted model. Logarithmic panels use a
65,536-product-element budget for C-contiguous right operands instead of 262,144;
these are element counts, not byte budgets. Public defaults and options are
unchanged. Default MU fitting still uses native `fit_transform`; the affected
projection also applies when `training_coordinates="transform"` is requested.

The isolated four-source comparison used 672 clean joint-prediction calls in
12 balanced blocks per panel and runtime on local Windows and WSL. On the saved
Fashion fixture, three controlled iteration caps improved mean combined time
by 13.7–78.3% versus the then-current baseline, with every pair favorable. These
caps exercise recovery and do not estimate its ordinary workload frequency.
Ordinary cases remained mixed. Full 2,000-query means were 46.0 ms slower on
Windows and 12.3 ms slower on WSL, with descriptive 95% difference intervals
[-9.9, 102.0] ms and [-195.8, 220.3] ms. Crossing zero does not establish zero
cost. Adding panels to recovery alone saved Windows cap130/181 means of
93.7/13.7 ms but cost 11.5 ms on WSL cap181; the combined decision retains that
tradeoff. Factor effects were measured together rather than assumed additive.

Each runtime passed 25 composition cases plus 167 existing regressions and
56 small plus four full scientific calls. Exactness claims remain limited to
the qualified outputs and contracts; this is not a universal private-helper
float32 or batch-invariance guarantee. The investigation preserves all adverse
observations, original failures and test-only corrections. Integration validation
is a separate source-bound batch; these measurements do not qualify a release
or establish native factorization or whole-training speedups. Historical studies
and their original source identities remain unchanged.

## Grouped example final-fit thread budget — November 4, 2026

The grouped-tuning example now limits its selected model's final fit to one
native numerical thread, matching its existing calibration and evaluation
policy. The context restores the caller's previous limits after success or
failure. This correction is confined to the example; estimator APIs, search
settings, projection policies and historical study results retain their scope.
The correction enforces the example's declared thread budget.

The local investigation covered 70 example invocations across the existing MKL
and AOCL environments: 46 with phase observations and 24 without function
instrumentation. With caller limits of one or two, the candidate's fitted
arrays, selected parameters, CV results, predictions, metrics, groups and IDs
matched each backend's one-thread reference exactly. Fresh and completed-reuse
calls, copied baseline checkpoints and 12 controlled error and qualification
calls preserved their intended behavior and restored the caller's limits.
Immutable checkpoint receipts and supplied originals stayed unchanged. Existing
rank-search, grouping and thread-control regressions passed 272 tests and 64
subtests per environment.

Three alternating pairs per backend timed the unchanged 160-row, 16-feature
example without function instrumentation. Paired median total time changed by
-2.91% in MKL and -3.53% in AOCL for fresh calls, and by -1.90% and +6.62% for
completed reuse. AOCL reuse ranged from -6.16% to +12.49%; its unchanged search
and checkpoint stage had a similar +6.48% paired median. Printed final fitting
took about 3.7 milliseconds with mixed small differences. These descriptive
measurements do not isolate context-entry cost or establish a general speed
claim. Whole-call times include the context overhead; printed fit times do not.
The slower observations remain part of the decision record.

The integrated example matched the candidate examined on the Windows host.
Its final-check configuration used the installed runtimes and a budget of
eight total compute threads. The study workspace retains the initial harness
failures alongside the source snapshots and integration record.

## Interrupted CelebA cache preparation — November 4, 2026

The default-study cache preparer now holds one output-scoped writer lock,
writes into private directories on the same filesystem, verifies each completed
cache, and publishes it at the existing per-cache boundary. Compatible committed
arrays and manifests remain unchanged. Sampling, preprocessing, labels, numerical
methods and preparation identities are preserved. An atomic receipt for the
completed largest training array allows restart after a later write failure
without decoding those images again; a verified complete private cache can also
be published on retry. Recovery verifies identity, metadata, selected IDs, shape,
dtype and checksums. It never accepts an unfinished training-array write.

Legacy final directories with absent or malformed manifests still fail by
default. The explicit `--recover-incomplete` option preserves such a directory
under a unique sibling wrapper and recreates the cache at the original name,
reusing separately verified private work when available. It does not
bypass committed identity or integrity failures. Failed artifacts are retained,
so recovery requires additional storage. Owned memory maps are closed on success
and error, and compact temporary names preserve the tested Windows path budget.
The user-facing option policy is in [the cache guide](celeba-default-results.md#preparing-and-restarting-caches).

The investigation's final candidate passed 56 interruption/restart cases and
22 focused payload cases in each existing local MKL/AOCL environment. It covered
concurrent writers, ordinary and abrupt owner exits, legacy recovery, corrupted
receipts and arrays, actual consumers, Unicode paths and baseline-valid
250-character array paths. Tests and full workflows preserved original inputs.

Four counterbalanced process rounds per environment measured the full nine-cache
synthetic workflow. Fresh creation's paired median cost was +3.68% in MKL and
+4.41% in AOCL (about 0.6 seconds); completed reuse was +0.68% and +0.88%.
The training receipt's incremental creation cost over staging alone was +2.15%
and +0.00%, with ranges including both directions. After an injected later write
failure, verified payload reuse reduced interrupted-plus-restart time by about
33% versus rebuilding. Actual CelebA recovery avoided a second 8000-image decode
and reduced that combined time by about 41–42% in single observations. Real-data
ordinary runs were not counterbalanced and filesystem caches were uncontrolled;
their observed differences do not establish an algorithmic speedup. These costs
are accepted for reliable restart, not for memory savings.

A full six-array recovery journal was considered but not adopted: the focused
receipt retains the expensive completed decode, while the remaining work was
small and a broader journal would add recovery states. All nine inspected
historical caches had completed successfully; this is not an interruption-rate
estimate. The validation establishes local Windows process-interruption and
ordinary I/O behavior, not power-loss durability or new Linux evidence.

The integrated implementation matched the tested candidate byte for byte.
Its final-check configuration on the Windows host used the installed runtimes
and a budget of eight total compute threads. The study workspace retains the
source snapshots and integration record.

## Impossible neighbor counts in shared grids — November 3, 2026

Shared KNN calibration now checks whether every requested neighbor count for
a representation exceeds the current fold's training size before screening,
fitting or projecting that representation. Such folds record
`failure_stage="neighbor_feasibility"`, zero executed work and no fitted
diagnostics or warnings. This failure precedes errors or warnings that the
skipped operations might have produced. Equality with the training size is
allowed; mixed grids still share a representation for their feasible counts.
All declared candidate/fold records remain, including diagnostics from feasible
folds when a candidate fails on a smaller fold. Qualification still requires
every fold to be valid. Generic calibration keeps its estimator-specific rules.

The isolated investigation compared 50 main scenarios and five corrected
custom-transformer controls in each local MKL/AOCL environment. Ordinary
training coordinates, validation projections, call counts and retained scores
matched exactly. Zero-work counters, earlier failure diagnostics, completed
reuse and interrupted commits were checked. The corrected controls retain
the original fixture limitation and its evidence rather than overwriting it.

Four alternating process rounds per environment covered ten workflow cases.
Fresh all-infeasible NMF grids were about 17% faster in both environments;
raw-KNN and ordinary timings were mixed, with no universal speed claim.
None of 275 inspected historical executed receipts had impossible neighbor
counts. Broader planner pruning was evaluated but not adopted because it
changes partial-fold evidence and requires coordinated plan/report policies.
The integration preserves the measured executable logic; only its public
docstring differs from the investigated candidate, verified by an AST comparison.

The integrated 18-file regression batch passed 243 tests and 252 subtests in
each existing local environment, including 39 new cases. There were no skips;
two warnings per environment were expected inactive-grid diagnostics. Tests
cover both resolved execution schedules, checkpoint recovery and saved-report
consumers. The study workspace retains the before-and-after source, validation
logs and raw observations.

## Terminal-NUL class labels — November 3, 2026

Current source rejects class labels ending in U+0000 (NUL). The investigation
reproduced silent errors in screening, logistic fitting and evaluation, as well
as crashes and invalid fold checks. A screening-only change did not resolve the
public workflow problem. The adopted policy preserves ordinary and embedded-NUL
labels, numerical formulas, fitted head types and the separate identifier policy.

A shared guard retains original sequences across validation, checks exact
object/StringDType values, and inspects list/tuple storage without invoking
subclass iteration overrides. Numeric targets avoid string scans. Fitting,
scoring, evaluation, folds and calibration validate labels before consuming them;
comparison/checkpoint paths cannot bypass checks through completed-result reuse.
External estimator predictions are checked before calibration metrics. Older
models retaining unsupported labels require safe original-target encoding and
refitting; saved state and historical results are not rewritten.

The final combined batch passed 1,721 tests and 436 subtests in each existing
local MKL and AOCL environment, with eight pre-existing platform/precision skips.
New cases cover failed refits, legacy models, separate score targets, public
cache paths, custom prediction outputs, sequence subclasses and ordinary-output
parity. A byte-level regression also demonstrates why fixed-width Unicode
padding cannot recover an original terminal NUL already lost before entry.

Four alternating before/after process pairs per environment covered 19 natural
workflow cases. Predictions, selected features, NMF bases/training coordinates
and warning sets matched exactly. A 20,000-string screening fixture added about
7–9% (roughly one millisecond); the one-column string fixture added about 14%
(roughly 0.10 milliseconds). Numeric columns avoided that string-scan cost.
NMF fit/score medians ranged from about -4.2% to +1.8%, evaluation remained within
about 1%, and fresh/cached calibration and rank workflows ranged from about -1.1%
to +4.5%, with retained timing variability. These bounded measurements do not
claim universal zero overhead. The added work prevents demonstrated incorrect
results; it is not a memory-saving tradeoff or a changed statistical objective.

The study workspace retains the test failures, timing observations and source
snapshots. The [class-label domain](api.md#class-label-domain) describes input
and serialization limits.

## Final-dissertation theory documentation — November 3, 2026

The [theory guide](theory-implementation.md) was checked against the October 16
final dissertation PDF and matching Chapter 7 source, superseding the October 1
draft mapping. Final Theorem 7.4.4 explicitly allows training-sample
reuse across screening/selection, representation fitting and uniform KNN.
The same observations can train the representation and classifier without an
extra split between those stages; CV validation and final test observations
still remain outside the corresponding fitting/selection steps.

The corrected guide includes final theorem numbering, the matching
constant-feature convention, selected-cutoff/retained-count discussion,
the reused-sample risk bound, its reference corollary and the exactly separable
consistency result. It distinguishes these conditional results from general
package options and from unconditional guarantees for fitted face models.
The review compared the chapter source with the rendered pages. The study
workspace retains the source hashes and earlier mapping. This correction
concerned the documentation only.

## CV candidate isolation — November 3, 2026

Scalar-parameter sNMF CV already fitted screening, basis, optional scaling and
classifier inside each training fold. Tests in two Python 3.12 environments
identified a separate general-search defect. Estimator objects supplied in the
parameter grid were inserted by reference after cloning the base estimator,
allowing state to carry between folds.

The defect was reproduced using sklearn's warm-start random forest and an
instrumented warm-start transformer. On deliberately small diagnostic data,
the forest's apparent mean CV accuracy was 83.33% instead of the independent
sklearn reference's 48.33%; the shared-transformer example gave 83.33% instead
of 50%. The caller's replacement objects were fitted or changed. These are
software-contract probes, not predictive-performance study results, and do not
establish that previous scalar-grid sNMF experiments were affected.

All six candidate configuration sites now clone parameter values using
`clone(parameters, safe=False)` before applying them, matching sklearn's
ordinary cloning semantics. This covers generic fold fits and final refits,
shared-grid preflight/fitting/report configuration and rank-bound inspection.
Rank search separately retains actual candidate configurations by stage/index;
it no longer attempts to use JSON display strings as live replacement
estimators during expansion. The old source hashes remain in old checkpoints;
the existing source guards reject their reuse under changed package code.

Both environments now reproduce the independent reference scores, preserve
caller objects and pass 144 tests plus 93 subtests. Checks include replacement
heads and transformers, previously fitted replacements, nested parameters,
fresh state per fold/final refit, shared-KNN reuse, real PCA rank expansion,
resume, groups, scalar grids and scoring. The study workspace retains the
initial failures, before-and-after fixtures, source hashes and logs.
No screening formula, NMF objective, default or historical study was changed.

At the end of that batch, cloning still respected custom sklearn clone behavior;
frozen estimators and global state had no new isolation guarantee. Two separate
source-review concerns remained: text-based candidate identity and missing
source binding for replacement classes. The follow-up investigation below
addresses those concerns; the earlier batch records remain unchanged.

## Candidate and cached-source identity — November 3, 2026

Tests in two Python 3.12 environments reproduced both defects. Different
transformers with the same display text, including truncated NumPy-array
parameters, were grouped together. A diagnostic candidate whose independent
CV accuracy was 50% inherited another
candidate's 100% score. Generic, shared-KNN and rank-search checkpoints also
accepted changed replacement settings or source and reused stale scores.
These are software probes, not accuracy evidence for a dataset or classifier.

Search identity now uses typed constructor configurations, exact ordinary
array contents and implementation-source hashes through both the base
estimator and candidate replacements. Readable display descriptions remain
in candidate reports but do not determine sharing or checkpoint identity.
Source checks around folds and generic final refits prevent changed-source
work from being committed as completed evidence. Equivalent fresh objects
resume without depending on their display text or process addresses.

The search helpers reject configurations whose state cannot be safely described:
opaque/cyclic values, unsupported array/scalar/callable subclasses, unavailable
custom source, and frozen/custom cloning hooks. Originals are checked before
any clone hook can substitute another object and hide that state. Ordinary
estimator fit/predict APIs are unaffected. Global/external dependencies,
arbitrary imported helper implementations, runtime monkeypatches and old class
objects retained after reload are outside the source-file contract; they are
not claimed to be captured automatically.

The adoption rationale is computational correctness: unequal computations
must not share results, and changed code must not silently inherit old results.
This changes neither the screening statistic nor a factorization/classification
objective. Constructor arrays are hashed when building identity; source-only
guards skip numerical-array contents and hash each source file once per check.
These checks add bookkeeping work for correctness, not for memory savings;
no general runtime improvement or universal dependency-capture claim is made.
Old checkpoints retain their historical identities. The changed source and
identity schema require a fresh checkpoint location. The study workspace
retains the reproductions of the defects and final-source regression results.

Final validation passed 198 tests plus 93 subtests in each environment, including
54 new identity tests and all earlier CV-isolation regressions in the selected
batch. Targeted after-fix counts are included in that total, not added to it.
Both representation-collision fixtures now match independent fold scores;
all three search APIs reject changed configurations/source, accept unchanged
fresh objects, reject custom clone bypasses, and stop mid-fit source changes
without committing a completed fold. Final hashes and a 36-file source/test
snapshot were audited. Historical dataset results were not rerun or certified
by these software tests.

## KNN weighting and decision policy — November 3, 2026

The dissertation analyzes uniform KNN. The November 3 development study
compared uniform and distance weighting jointly with `n_neighbors` using
training-only cross-validation on face data. It selected by ordinary accuracy
and fixed the majority-vote boundary at 0.5, with equality assigned to
`classes_[0]` when the positive class was `classes_[1]`. Either voting rule
uses the same nonnegative NMF representation. Chang Cheng retained uniform
weighting as the package default and supported joint tuning of neighbor count
and weighting on training data.

A 0.5 boundary is standard binary majority voting, including normalized
nonnegative weighted votes; it is not an assurance of optimal performance for
every cost-sensitive or imbalanced-data objective. The current constructor's
`decision_threshold=None` delegates to the head's native rule. Explicit
thresholds use strict probability comparison. Preserve that distinction,
especially for floating-point weighted ties and reversed positive labels.

Both Python 3.12 environments reproduced disagreement between probability-argmax
and native distance-weighted voting at a floating-point tie. The
combined prediction API now delegates ambiguous weighted votes to native
prediction on the whole original coordinate batch; separated votes keep the
shared probability calculation. Callable weights also use their native decision.
There is still only one NMF projection. This can require a second neighbor query
for an ambiguous batch; the reason is preserving the fitted decision rule, not
memory reduction or an accuracy-based change to the cutoff.

Finite extreme inputs also produced infinite neighbor distances that evaded
the shared-query boundary guard. The adapter now detects nonfinite distances
and uses the public selected-k path before cached voting. The concrete fixture
demonstrated invalid qualification and a warning, not a changed class label.
This fallback preserves underlying estimator behavior; it does not repair
distance overflow inside sklearn. The study workspace retains the initial
failures and numerical fixtures.

The completed batch passed 291 tests and 155 subtests per environment, including
the affected-file follow-up after correcting an existing test-helper collection
error. A finite five-fold face comparison used 4,000 previously explored
official-training images, 4,096 grayscale pixels, fixed SPA rank 10, and a
training-only coarse density rule for the screen. Both environments produced
identical predictions for all 28 label/weight/k configurations. Selected mean
CV accuracy was 65.802% versus 65.852% for Smiling (uniform versus distance,
both k=25), and 91.675% versus 91.825% for Necktie (both k=15). Necktie positive
recall was only 8.50% versus 10.26%; its majority reference accuracy was 91.475%.
These are selected training-CV scores, not held-out confirmation or a fully
tuned representation comparison. No winner reached the declared k boundary.
The small gains on these two attributes leave the general choice of voting
policy open. The study workspace retains the predictions used for this
comparison. The [prediction policies](api.md#prediction-policies-during-evaluation)
describe how evaluation handles weighted ties.

## SPA training-coordinate consistency — November 3, 2026

Standard SPA `NMFClassifier.fit_transform` now returns an independent
copy of the projected training coordinates already stored by `fit`, eliminating
its previously inherited second projection. CD/MU, custom subclasses and
overridden transform methods retain fit-then-transform delegation. The returned
SPA representation is mathematically the same; rerunning NNLS can differ at
floating-point roundoff because fit/query data layouts can differ. Returning
saved coordinates does not create a new public-transform diagnostic receipt;
the training projection is recorded in the fitted training-phase attributes.
The follow-up passed 277 tests and 161 subtests in each of the two Python
environments. On the fixed generated data, public `fit_transform` took about
30–36% less time across the three classifier heads, with identical predictions
and fitted coordinates. The timing applies to that probe. See the
[fit and transform contract](api.md#fitting-and-transforming-together).

The development source makes standalone `FeatureSPA.fit_transform` fit the
anchor-based basis and return the same fixed-basis NNLS representation as
`transform`. This gives generic classifier pipelines a common training and query
map. Anchor selection, column normalization and the basis-fitting objective are
unchanged. `fit_coordinates_` retains the original anchor columns, and standalone
`reconstruction_err_` retains their factorization diagnostic.

For newly fitted SPA `NMFClassifier` models, all accepted
`training_coordinates` settings resolve to `"transform"`, including explicit
`"fit_transform"`. The classifier fits the basis directly and performs one
training projection during `fit`, avoiding duplicate work. CD and MU coordinate
policies are unchanged. The November 3 implementation applied its policy on
refit and kept the learned state and coordinate metadata when loading a model.
For version 0.5.0, load an older model in its originating environment or refit
from the original training data. [Model persistence](api.md#model-persistence)

The correction can change accuracy and adds projection work to standalone
pipelines previously trained on anchors. The default SPA classifier already
used projected coordinates. Exact NNLS projection cannot worsen reconstruction
against the fixed basis, but does not guarantee better classification. Adoption
rests on consistent methodology and public API behavior as well as numerical,
accuracy and runtime evidence; record validation separately from these
analytical statements.

The Python 3.12 tests covered the AOCL and MKL environments used on November 3.
Each passed 258 unique tests and 161 subtests, including the affected-file rerun
after correcting an overly strict floating-point assertion. A three-seed generated-data
probe preserved default-classifier coordinates and predictions exactly for KNN,
logistic regression and linear SVM. Standalone SPA+KNN mean query accuracy changed
from 95.78% to 94.44%; its median fit time increased from about 4 ms to about
10 ms due to the added training projection. These small fixed-parameter probes
establish neither general accuracy improvement nor scalability. The common
training and query map was adopted on methodological and API grounds despite
the added fitting cost. The study workspace retains the failures, timings
and source hashes.

## Public screening and calibration API — November 3, 2026

The public interface now uses ordinary accuracy as its default calibration
objective. `calibration_search`, `calibration_knn_grid`, `calibration_rank_search`
and `run_comparison` accept explicit `scoring="balanced_accuracy"` when the
prediction objective calls for it, particularly with extreme class imbalance.
No class-proportion rule silently switches objectives. CV ranking and bounded
rank expansion use the same chosen metric; independent evaluation metrics
remain separate. Scoring is part of saved checkpoint identity.

Both `BinaryScreening` and `NMFClassifier` default to absolute pooled-t
screening at `score_threshold=2.0`. Percentile screening remains an explicit
mode. Public API consistency takes priority over retaining the standalone
transformer's earlier percentile default. Historical study records are unchanged.

Calibration diagnostics record active screening controls and warn about
explicitly gridded inactive parameters. Refinement reports mark those parameters
inactive rather than suggesting an expansion. Neither diagnostics nor metric
selection change the pooled-t formula, constant-feature handling, NMF objective,
training/query coordinate maps, or prediction rules. Every learned stage remains
inside the training fold. Classification score selection is distinct from
defining the feature score and selecting its cutoff.

The scoring interface supports accuracy and balanced accuracy. Other objectives
can use an external sklearn search. The shared KNN implementation keeps its
existing prediction path and records the chosen metric with the checkpoint.
The study workspace retains validation logs from both environments.
[Selection objective and report fields](api.md#selection-objective-and-report-fields)

## Retained image values — November 3, 2026

Retain the current numerical behavior: screening selects supplied feature
columns without standardizing their values or multiplying them by screening
scores. For decoded 8-bit images, divide by 255 explicitly before modeling;
the estimator does not infer bit depth or rescale pixels automatically.
Temporary normalization for screening statistics or SPA anchor selection does
not change the values supplied to the factorization objective.

Saved nonnegative coordinates and basis factors support visualization and
approximate reconstruction after classification. With inputs divided by 255,
multiply the reconstruction product by 255 to return to intensity units.
Individual basis amplitudes have arbitrary factor scale. Excluded features
must be distinguished from zero-valued retained features in displays; they
cannot be reconstructed from the screened model. Nonnegativity alone does not
guarantee anatomically localized components.

This decision documents existing behavior and adds a saved-model reconstruction
recipe in `docs/api.md`; it introduces no numerical code change or validation run.

## Research policy — October 20, 2026

Small differences in accuracy or execution speed do not determine adoption by
themselves. Decisions also consider mathematical fidelity, theoretical support,
numerical reliability, practical benefit, software design, maintenance cost,
and uncertainty in the evidence. Analytical guarantees remain distinct from
empirical observations: benchmark performance alone does not establish
correctness or theoretical validity. Memory savings do not justify slower
execution, including through an optional implementation.

The host used for the October 20 integration had 20 physical cores and 40
logical threads. That run used a total budget of 20 compute threads across
workers and native numerical libraries.

Python 3.10 support ended on October 20, 2026. The development environment
used at the time was Python 3.12 with AOCL. Version 0.5.0 requires Python
3.11 or later. [Runtime compatibility](compatibility.md#current-support-policy)

## Public source, rights and citation — October 20, 2026

The package is publicly available on GitHub, with All Rights Reserved for
Chang Cheng's original work pending a distribution-license decision. Source and
wheel archives include both the project LICENSE and THIRD_PARTY_NOTICES,
preserving the separate third-party copyright and BSD notices.
Public source availability does not supply an open-source license for the
author's work, and historical release artifacts remain unchanged.

The dissertation title is *Ensemble Kernel Distance-Weighted Discrimination and
Supervised Nonnegative Matrix Factorization for Classification*. The package
implements the supervised NMF methodology in Chapter 7.

The public repository uses standard GitHub-hosted runners for CI.

## Classifier scope decision — October 20, 2026

After reviewing the five-classifier face-image pilot, Chang Cheng selected
three classifiers for continued package development:

- K-nearest neighbors (KNN).
- L2-regularized logistic regression.
- Linear support vector machine (linear SVM).

Histogram gradient boosting and shrinkage LDA are outside the adopted
classifier scope. Preserve their completed pilot results, study adapters,
failures and provenance as historical evidence. This scope decision supersedes
the pilot's recommendation to retain boosting as a candidate for the proposed
classifier interface; it is not a claim that the excluded methods performed
worse.

KNN remains the package default. KNN and logistic regression were already public
classifier options at the time of this scope decision; linear SVM existed in
the study adapter. The integration below subsequently completed the public SVM
API and its integration checks; it is included in version 0.3.0. SVM decision margins
are reported as scores, not probabilities. DWD remains outside the current project.

The face-image validation plan covers these three classifiers, with training-only
transformations and explicit imbalance handling.

## Scope and method

1. Follow Chang Cheng's dissertation and original scripts; preserve his authorship.
   Report every bug fix, performance change, or methodological departure.
2. Track Wang's NMF concerns against the available evidence and remaining work.
3. Determine coordinate descent versus SPA and percentile versus absolute
   screening through development comparisons that consider time and accuracy,
   without unsupported transfer of theoretical guarantees. The October 3 CelebA
   author-preference amendment selected
   [SPA with absolute-score screening](celeba-default-results.md). The subsequent
   comparison added Fashion-MNIST, MU and ANLS-BPP. The completed
   [four-solver report](nmf-solver-comparison.md#development-results)
   selects SPA and remains `selected_for_audit`. SPA remains installed; absolute
   screening is fixed in that comparison. Both original CelebA decisions are preserved.
4. Include screened PCA and screening without factorization as classification
   comparisons, together with raw-feature baselines. KNN remains the default;
   optional logistic regression is evaluated independently for the NMF study.
5. Complete tuning and numerical confirmation: expand boundary winners and refine
   interior intervals. An exhausted time/search budget is not completion.
6. Fit all learned transformations inside training folds. Use five-fold calibration;
   freeze qualified settings before access to the final test data. Report tuning
   and final-fitting cost separately. Save failures, seeds, source IDs, checkpoints,
   and compatible completed results.
7. Training allocations are specified by each study; the earlier 20% rule was withdrawn.
8. The October 3, 2026 CelebA study used Python 3.12 with AOCL and local copies
   of the images, attribute annotations and official partition metadata.
   Training and validation followed the published partitions.
9. The additional adaptive workflow was withdrawn. Use the package's
   own explicit grids for NMF/PCA rank, screening and the final classifier. Record grid
   choices, boundary checks, and any expansions or refinements as study decisions.
   No external search policy is required.
10. Freeze external development protocols before evaluation. Official-test
    evaluation requires a completed evidence audit and a frozen evaluation plan;
    preserve the interrupted prior CelebA attempt.
11. On October 5, 2026, the public API added `solver="mu"` alongside `"spa"`
    and historical `"cd"`, preserving the
    SPA/absolute defaults and the screening-to-factorization-to-KNN pipeline.
    MU retains the experimental comparison's unregularized Frobenius objective
    and NNDSVDa initialization. ANLS-BPP remains experimental. Record this API
    extension separately from the completed, frozen benchmark evidence.
12. Logistic regression extends the dissertation's KNN classification stage.
    The October 6 [matched CelebA comparison](wang-real-data-validation.md)
    fitted each coordinate scaler within the training fold and tuned inverse L2
    regularization for each representation. Those development results are
    available. A logistic-specific theoretical assessment remains open.
13. On October 5, direct BLAS calls were investigated before
    the Wang validation resumed. SPA now uses public SciPy GEMV/GER updates
    and one fixed-basis layout conversion per transform. The
    [performance and equivalence record](blas-optimization.md) distinguishes
    this implementation change from the statistical method. Preserve the
    interrupted study snapshots and use a new source freeze for subsequent
    runs; do not silently combine timing records from different implementations.
14. On October 6, a review of the four-method face validation identified the
    omission of the dissertation's standard NMF+KNN baseline. A separate supplement
    restores unscreened CD-NMF+KNN and adds unscreened SPA-NMF+KNN, using the same
    cohorts, independent rank/K tuning and public-estimator qualification.
    The original four results remain frozen. Since the original test outcomes
    were already known, the additions are descriptive supplementary comparisons,
    not new unseen-test confirmation. Preserve numerical failures and any
    unresolved rank boundaries; do not claim globally optimized baselines.
15. On October 6, screened CD-NMF+KNN was added to the comparison plan as a seventh
    method after observing the unscreened CD/SPA comparison. Use the same
    absolute-screening, rank and dense-K starting grids, five training folds,
    single local refinement and numerical qualification rules as the frozen
    face comparison. Independently tune the screened CD pipeline; preserve all
    six earlier fits and results. This is another post-test-designed descriptive
    supplement, not new independent confirmation. Comparing the tuned screened
    SPA and CD pipelines may change cutoff, rank and K as well as solver;
    do not interpret it as a fixed-configuration causal solver effect or use
    its test outcome to revise package defaults.
16. On October 6, the validation scope expanded beyond a single
    face task. The first expansion fixes two new
    face datasets (FEI smile/neutral and CMU sunglasses/open) and three naturally
    nonnegative nonface datasets (Sonar, Spambase and WDBC) before fitting.
    Evaluate all seven KNN pipelines on every included task, retaining losses,
    inconclusive comparisons and failures. Face identities must remain disjoint
    across both the reserved split and every inner training/validation fold.
    Retain the existing nonface preparation's reserved partitions. Use matched
    five-fold tuning and the declared dense neighbor grid, with physical
    small-sample omissions recorded before fitting. This first expansion is a
    multiple-dataset benchmark, not a completed repeated nested-CV study or a
    claim that R6's entire replication request has been resolved. Reports include
    accuracy, fit-time and prediction-time tables.
17. On October 6, additional labels in the existing datasets, including FEI,
    were added to the exploratory study plan. Explore observable CelebA attributes,
    CMU source pose/expression conditions, and FEI source acquisition conditions
    using all seven pipelines. Freeze the task list, development cohorts,
    label coding, reduced image representation, grids and support rules before
    fitting. Keep old official-test and reserved subjects outside this
    exploratory study. Several labels from the same people are correlated
    tasks, not independent datasets; previously used training images are not
    fresh confirmatory evidence. Retain every task and failed method. Use
    balanced accuracy to assess imbalanced attributes and report majority and
    random-guess references with ordinary accuracy. Do not assign semantic
    pose/expression names where the source only verifies condition identifiers.

## Stages and what constitutes evidence

On October 9, training-only prototype tests supported guarded batched SPA
projection, reusable process workers, separate projection and native-thread
controls, and a saved decision threshold. The new models then used
`projection_solver="auto"`. The loader at the time retained SciPy for older
models. In 0.5.0, load a saved model with its original implementation and
dependency versions, or refit from the original training data after upgrading.
[Model persistence](api.md#model-persistence)

These changes affected query execution and preserved basis fitting, the NNLS
objective, numerical checks and fallback. The optional decision threshold
used the tested strict probability rule. Benchmarks retain the source and
execution controls used for their measurements.
[SPA projection execution](api.md#spa-projection-execution)

The remaining diagnostics supported early release of SPA anchor-selection
workspaces. This memory optimization occurs after the
anchor condition checks and before basis NNLS. It preserves the tested
numerical path and estimator defaults. The normalization results do not justify
a universal KNN normalization default. The completed study and diagnostic
receipts retain their frozen source hashes; applying this package fix does not
rerun or replace those studies.

- **Implementation checks:** unit tests and a small development run establish
  execution, isolation, deterministic behavior, and numerical edge-case handling.
  They cannot choose defaults or resolve Wang's scientific concerns.
- **Historical diagnosis:** reproduce the old simulation configuration and examine
  the 32.42% standard-NMF result. Preserve historical and revised configurations as
  separate rows. Change one diagnostic factor at a time. Do not choose label
  flipping using test labels or call a changed study design a bug fix.
- **Development comparison:** the completed CelebA study compared both screening
  rules and CD/SPA. The completed external follow-up fixes absolute screening and
  compares CD, SPA, MU and ANLS-BPP on matched datasets and folds, with independent
  tuning and relevant PCA/screening-only context. Record support recovery where known, predictive
  performance, convergence, conditioning, time and failures.
- **Protocol freeze:** document selected ranges, convergence settings, stopping
  criteria, refinement history, replications, and default decisions. A quick CLI
  run is explicitly provisional and leaves this stage open.
- **Final study:** execute the frozen protocol on untouched tests, then report
  paired differences and uncertainty without tuning on final outcomes.

The original proposal called for 27 simulation conditions, harder
correlation and interaction cases, and 20 paired CelebA training runs per
attribute. The October 6 simulation study completed all 27 factorial
conditions plus three stress cases (cross-block correlation, null signal
and XOR), with 30 independent training/test draws each.
[Simulation results](wang-simulation-validation.md)

For Smiling and Wearing Necktie, the October 6 development study completed
20 paired training resamples with fixed settings and 10 independently retuned
resamples on shared validation images. The final official-test phase of the
original default-selection protocol remains unfinished. Reusing the same
evaluation images does not yield independent test samples.
[Matched CelebA results](wang-real-data-validation.md) · [Original default-selection results](celeba-default-results.md)

The October 6 CelebA development study selected candidates by balanced accuracy.
The original study draft also called for reporting ordinary and balanced
accuracy, ROC AUC, average precision, precision, recall, F1, prevalence,
confusion counts and a majority-prediction reference. Average precision and
trapezoidal PR area are distinct measures.
Version 0.5.0 generic calibration defaults to ordinary accuracy. Choose
`scoring="balanced_accuracy"` to select by mean class recall. Specify the
objective on development data before evaluating on the test set.
[Calibration objective](api.md#selection-objective-and-report-fields)

The feature-anchor SPA model and its normalization must be stated alongside
results. General clipped simulations and CelebA are not assumed separable.
Numerical convergence and empirical success do not prove the theory's assumptions.

## Parameters and numerical qualification

The main predictive parameters are rank, retained percentile or absolute pooled-t
cutoff, and KNN neighbor count or logistic inverse regularization strength.
Rank applies to NMF/PCA; the screening parameters
apply to their corresponding rule. Tune each method independently on the same
five training folds. The external follow-up compares CD, SPA, MU and ANLS-BPP with
absolute screening fixed; only its cutoff is tuned. Percentile comparisons remain available
for historical reproduction. Record choices explicitly rather than presenting
them as an unchanged historical dissertation method.

Iteration cap and tolerance control numerical qualification. Establish and record
these settings on training data before comparing predictive candidates. Reject
candidates with failed or nonconverged required folds and retain the diagnostics.
Any numerical change requires a recorded new configuration; a successful rank-2
smoke fit does not qualify larger ranks or samples.

Logistic regression has its own iteration cap and tolerance. Qualify both
factorization and classifier convergence; a converged factorization does not
excuse an unconverged logistic fit. The starting C grid is
`[0.01, 0.1, 1.0, 10.0, 100.0]`. Apply the same boundary and refinement checks
as for representation parameters, using training folds only. Use matched
partitions and metrics for KNN/logistic comparisons on NMF, PCA, screening-only
and raw-feature paths. Record class weighting as part of the configuration;
balanced accuracy does not require fitting with balanced class weights.

Screened PCA is a priority comparison for Wang's R7. Smiling and Wearing_Necktie
require separate tuning and reports. For the latter, report positive counts and
precision/recall measures prominently so majority accuracy cannot conceal a
failure to recognize positives. See [the CelebA study plan](celeba-study.md).

## Numerical test interpretation

Composition tests compare the package with the corresponding estimator pipeline
in the same environment, using matched data, dtype, layout, initialization and
training-coordinate policy. Preserve their strict assertions. Independent
numerical checks, such as the rank-one projection oracle, serve a different
purpose and should continue to cover supported dtypes, layouts and input
immutability.

Environment comparisons use the same source, data, parameters and seeds, with
controlled thread settings and recorded numerical-library builds. Compare
selected features, finite/nonnegative factors, the original reconstruction
objective, appropriate projected-gradient diagnostics, warnings, iterations,
predictions and measured costs. For fixed-basis projection, only W is optimized;
an H gradient is not a query-solve optimality criterion. State diagnostic
normalizations before comparing their values.

Raw factor equality is not a portability requirement: NMF admits component
permutations, positive rescaling and different local solutions. Nearby objective
values or unchanged accuracy do not establish equivalent predictions. Investigate
material differences with common initialization, basis and classifier-coordinate
replays. A large numerical discrepancy does not identify its causal importance;
use controlled interventions to distinguish mechanisms.

Do not relax suite-wide tolerances or change solver stopping settings merely to
make environments agree. Retain failed cases and justify any test-specific
tolerance or design change mathematically. Warning-free completion does not
independently establish stationarity or global optimality. Use float64 as an
explicit study choice where warranted, recording its cost and the changed
configuration rather than altering the package's dtype behavior.

## October 20 implementation

Following the retrospective review, the package adopted minimum-reference
screening statistics, conditional per-RHS NNLS scaling, stable factor-recovery
reporting, guarded neighbor-query reuse in explicit KNN grids, optional fitted
SD scaling for KNN, and a public linear SVM head. Report-only residual blocking
was conditional on integrated cost checks supporting its dispatch domain.
The completed checks support a bounded dense float64, single-thread MKL path;
other execution settings retain the original residual expression. See the
[validation record](validation.md) for the completed integration evidence.
These changes were released in version 0.3.0; earlier releases and study
receipts retain their original implementations and configurations.

The screening and NNLS repairs preserve their mathematical objectives.
Minimum-reference subtraction precedes scaling when computing pooled statistics;
the selected data passed to factorization remain uncentered. NNLS uses a common,
exactly reversible binary scale for a basis and each RHS, grouping equal scales
for execution. It does not improve conditioning or certify every extreme input.
Unrepresentable ranges fail explicitly. Factor-recovery reports normalize before
forming cosines while retaining full-truth support and unmatched-rank penalties.

KNN SD scaling and linear SVM are optional statistical choices, rather than
numerically equivalent replacements for the existing classifier. SD scaling is
fitted within each training fold, changes the KNN metric, and leaves factorization
coordinates unchanged. The SVM head fits an unscaled, L2-regularized squared-hinge
primal problem. Tune its C independently of representation parameters and qualify
its own iteration cap and tolerance. Its decision margins support AUC/AP ranking;
they are not probabilities. A probability threshold is inapplicable.

Neighbor reuse is internal to compatible fold-local KNN grids. Unsupported or
numerically uncertain cases use the selected-k public estimator on the original
query batch. Reports charge actual shared query costs once and retain separate
head costs. Old checkpoint identities cannot resume under changed package source.

The integration batch ran on the validation host with at most 20 total compute
threads. It covered correctness, API/serialization contracts, original numerical
triggers, and targeted paired timing. It did not rerun the historical dataset
studies or retune defaults, and it does not establish a new theorem.
Adoption weighs mathematical fidelity, numerical reliability, software design,
maintenance and measured cost together. A memory saving cannot justify a
slowdown, including through an optional implementation.

## October 25 SPA basis integration

The high-rank scaling investigation supported reusing the guarded batched NNLS
kernel for SPA basis fitting. The fixed anchor coordinates,
screening rules, NNLS objective, classifier defaults and projection controls are
preserved. A basis-specific internal wrapper retains the original public SciPy
column loop as a compatibility fallback for numerical-range rejection. It adds
no public execution option or nested basis workers.

The implementation retains the 1e4 normal-equation condition guard, numerical
qualification and fitted-basis condition check. Near the guard, small residuals do not imply
the same coefficient precision as public SciPy. Batched pivot rounds and SciPy
iterations are different units; fallback receives the original iteration cap.
These limits accompany the runtime evidence and shared-kernel maintenance
benefit in the adoption decision. The October 25 implementation preserved saved
bases and used the revised kernel on refit. For loading saved models in version
0.5.0, see [model persistence](api.md#model-persistence).

## October 26 MU precision recovery

MU range correction applies to fixed-basis projection, including an explicit
training reprojection. Joint factorization and the default training-coordinate
policy remain unchanged. Exact zero batches return zero coordinates after full
input validation. Positive values are not classified as zero by a tolerance.

Ordinary float32 observations can encounter tiny fitted basis tails or an
underflowing update without requiring a full logarithmic solve. The projection
therefore retries eligible float32 calculations in float64 before logarithmic
recovery. It uses a temporary estimator and leaves the fitted basis unchanged.
Tiny positive inputs and unresolved range failures retain logarithmic recovery.
Both paths preserve the MU recurrence and stopping rule in exact arithmetic;
finite precision can change rounding and stopping iterations. Output dtype,
iteration-cap warnings and errors for unrepresentable coordinates are retained.

The evaluation separates three implementations: the published 0.3.2 package,
the earlier range correction and the revised precision recovery. It considers
ordinary data, the failures that motivated each correction and difficult
numerical cases. A large improvement over the earlier correction does not
erase remaining overhead relative to the published package. Correctness,
methodological fidelity, numerical reliability and measured cost all enter the
decision; memory reduction is not a justification for slower execution.

## October 29 third-party review workflow

The October 29 protocol called for assessing a reported behavior against
reproduction evidence, original source and the documented API contract. It
distinguished demonstrated failures from plausible risks and considered the
option of keeping existing behavior. Decision criteria included correctness,
methodological fidelity, numerical behavior, compatibility, runtime and
maintenance costs.

The protocol called for checks appropriate to the change on both AOCL and
MKL, with the actual loaded numerical libraries recorded alongside each
runtime's results. The [numerical test interpretation](#numerical-test-interpretation)
describes which solver diagnostics and outcomes to compare.

