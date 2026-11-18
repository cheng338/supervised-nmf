# Four NMF methods in the development comparison

Author: Chang Cheng. The comparison extends the dissertation's supervised NMF
pipeline to four factorization methods. The package supports `solver="spa"`,
`"mu"`, and `"cd"`, with SPA as the default and CD as the original solver.
ANLS-BPP is an experimental study implementation. The frozen benchmark records
are preserved separately from later API changes.

The October 4 [numerical repair](numerical-repair-20261004.md) records a native
least-squares crash during the first run, the direct LAPACK workaround, and
the requirement to rerun ANLS-BPP under a new source freeze. Unchanged methods'
verified receipts retain their original provenance.

| Method | Role in this study | Reconstruction approach |
| --- | --- | --- |
| CD/HALS | Package solver; historical reference | scikit-learn coordinate descent, using Fast HALS |
| SPA | Default package solver | Feature-anchor selection followed by nonnegative least squares |
| MU | Package solver | scikit-learn multiplicative updates with Frobenius loss |
| ANLS-BPP | Experimental comparison | Alternating nonnegative least-squares subproblems solved using block principal pivoting |

[scikit-learn documents CD and MU](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html),
with CD as its default. [Lee and Seung](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
provide the classic multiplicative-update reference.
[Kim and Park's software and publications](https://faculty.cc.gatech.edu/~hpark/nmfsoftware.html)
describe ANLS with block principal pivoting and active-set approaches. This is
a list of established algorithm families, not a quantified field-wide ranking
of usage or a claim that one is universally fastest.

All four use the same absolute pooled-t screening rule and KNN classifier in
the [frozen comparison](fashion-mnist-protocol.md). Rank, score cutoff and KNN
neighbors are selected independently within matched training folds. CD, MU
and ANLS-BPP use NNDSVDa initialization; SPA retains its anchor construction.
Classification in the frozen study uses each method's `fit_transform` training
coordinates, retaining their original scales and classifier geometry. For SPA,
these are original-scale anchor columns; query coordinates use fixed-basis NNLS.

The public MU path preserves that study's unregularized Frobenius objective and
NNDSVDa initialization. It exposes the existing `max_iter`, `tol`, `random_state`,
and `training_coordinates` parameters. For MU, the current `"auto"` policy
resolves to `"fit_transform"`, preserving the study's fitted-coordinate convention;
`"transform"` explicitly recomputes them against the fitted basis. The comparison-model
factory includes `screened_mu_absolute` and `screened_mu_percentile` alongside
the existing SPA/CD models. These API options do not extend the absolute-only
four-solver study into evidence about percentile MU.

The algorithms have different stopping criteria and per-iteration costs. CD
uses a relative coordinate optimality measure; scikit-learn MU uses relative
reconstruction improvement. ANLS-BPP has outer factorization convergence and
inner NNLS qualification. SPA's anchor selection is not unrestricted NMF
optimization. Iteration counts and equal tolerance numbers are consequently
not interchangeable measures of numerical accuracy.

The run retains convergence warnings, failures, factor dtypes, reconstruction
residuals and common optimality diagnostics. Numerical qualification prevents
warning-producing selected fits from being treated as complete successful
evidence. It does not certify a global NMF optimum. Factor rescaling also
changes gradient norms; any diagnostic normalization must be reported and
must not change the factors used for classification.

The dissertation's reconstruction-based argument can apply conditionally to
any fitted basis and coefficient map, including these additional algorithms.
It still requires the stated population reconstruction and smoothness terms;
faster convergence or lower training reconstruction error does not establish
those assumptions. The reference-basis argument additionally needs alignment,
conditioning and controlled projection error. The independent-sample KNN
corollary is not automatically established by any of the four implementations.

## Experimental ANLS query diagnostics

`ANLSBPP.transform` returns query coordinates without changing fitted estimator
state. To inspect a query's NNLS qualification, use the optional
`capture_transform_diagnostics` context manager from the experimental script.
This is useful for numerical audits and study reporting; ordinary callers can
continue using `transform` directly with its existing warnings and exceptions.
These helpers live in `scripts/experimental_nmf.py`, not the installed package's
public classifier API. With that script directory on the Python import path:

```python
from experimental_nmf import ANLSBPP, capture_transform_diagnostics

model = ANLSBPP(n_components=2).fit(X_train)
with capture_transform_diagnostics(model) as query:
    coordinates = model.transform(X_query)

qualified = query["converged"]
solver_details = query["diagnostics"]
```

Each context yields a new record owned by its caller. Use one scope per transform
and retain that record for the corresponding output. No second solve is needed.
The record has these fields:

| Field | Meaning |
| --- | --- |
| `performed` | Whether this estimator attempted a transform in the scope, including a call rejected by fitted-state or input validation. |
| `converged` | `None` before an attempt, otherwise the current solve's qualification or `False` if the call raises. |
| `diagnostics` | The current returned NNLS information dictionary; `None` when no solve result was returned. |
| `error` | `None` unless the call raises, then its exception type and message. The exception is still propagated. |

A normally returned capped or unqualified solve retains its existing feasible
coordinates and `ConvergenceWarning`, with `converged=False` and the current
solver details. If the caller promotes that warning to an exception, no solver
result is returned: the record contains the error and no solver dictionary.
Input errors and interrupted calls likewise cannot retain an earlier query's
successful diagnostics. Capturing a record does not suppress or change warnings.
A scope that never calls its matching estimator remains unperformed; it is not
evidence of convergence. Non-ANLS estimators yield `None` for study adapters.

Separate captures in different threads keep their records separate; nested
captures restore their outer context. This is diagnostic isolation, not a
general thread-safety guarantee for warning filters, native runtime controls,
or external instrumentation. Do not share one active capture across concurrent
operations, refit an estimator during inference, or change its parameters while
a query is running.

The experimental Fashion/CelebA study paths pass the query record explicitly to
reporting and qualification. Fit-only records contain training diagnostics, not
evidence from an earlier query. Existing study query-field names remain in JSON
reports, with additional performed/error information. Shared CV retains its
existing warning/exception qualification; it does not gain detailed per-query
ANLS records through this change.

Older pickles may contain `last_transform_converged_` and
`last_transform_diagnostics_`. These fields are left untouched during inference
and are ignored by the revised experimental study reporting. Use a new capture
for current evidence. Refitting retains the existing learned-state reset.
Historical JSON reports and source identities are preserved; changed-source
study runs still require fresh output locations.

## Experimental ANLS factor and report copying

`ANLSBPP.fit_transform` continues to publish independent C-order fitted factors
and return independent C-order training coordinates. For compatible float64
Fortran-order W factors, a bounded copy routine can populate the required C
array through an integer view, preserving every bit without floating-point
arithmetic. This route requires at least 8,192 rows and 20 columns, a row count
divisible by 4,096, and exact array, dtype and layout conditions. Other inputs
retain their existing copy operation. Fitted and returned arrays do not borrow
the private solver factors or each other's storage.

On the existing padded backend, the final reconstruction report also uses a
bounded copy route for compatible W/H layouts. It populates the original
required left workspace in four-column panels when the row count is divisible
by 4,096, rank is at least 12 and the left factor has at least 65,536 elements.
The route requires an already aligned row dimension and a genuinely padded
output-column dimension. It retains the full reduction dimension, protected
product sizes, final matrix product and norm. This private report-only choice
does not affect training-loop products or query projection.

Allocation failure in optional copy preparation returns to the original copy
path. Required storage allocation and required fallback failures still
propagate. An incomplete optional factor copy releases its scratch array before
fallback; an incomplete report-workspace copy is overwritten in full before
the matrix product. These rules preserve ownership and failure boundaries, not
identical allocator or garbage-collection scheduling under memory pressure.

No solver objective, convergence rule, public parameter or user opt-in changes.
The fixed eligibility rules limit extra work on small or incompatible arrays;
they are not universal performance crossovers. See the
[decision and measured costs](development-protocol.md#experimental-anls-factor-and-report-copying)
for the supported scope and the limits of the performance evidence.

## Experimental ANLS training-state reuse

During `ANLSBPP.fit_transform`, a successful H solve has already computed a
reconstruction residual and H gradient. Eligible outer iterations can reuse
those arrays instead of repeating the same products. Reuse requires compatible
float64 operands, native operation and layout, the expected factors and input,
and the caller's numerical policy. Some eligible layouts reuse only the residual;
the outer H gradient is then computed normally. This does not reuse the W
solve's earlier residual or replace the outer W-gradient calculation.

Reuse is bypassed when `n_components == 1` and `X.size <= 16384`, where `X.size`
counts training-matrix elements, not bytes. All other inputs still undergo the
normal eligibility checks. This fixed cost policy does not change tolerances,
iteration limits or repeat detection, and is not a universal performance
crossover. It adds no public option or package solver choice.

The private handoff lasts only for the current training step; it is not stored
on the fitted estimator or shared with query transforms. The complete H-solve
finalization is checked with floating-point exceptions enabled, retaining
ignored underflow. If this raises a floating-point error, the original
calculation runs under the caller's policy. Before reusing an H gradient, the
W-gradient calculation has its own quiet check; a floating-point signal discards
the saved H gradient before replay so ordinary warning/error callbacks retain
their effect on the subsequent calculation. Ineligible calls retain the original
arithmetic route. Allocation failure in optional handoff metadata falls back to
required arithmetic; required numerical failures still propagate.

Public return signatures, fitted factors, stopping rules and caller-owned query
diagnostics retain their contracts. This optimization assumes the installed
numerical helpers; it does not promise compatibility with arbitrary runtime
replacement of private functions or mutation of private handoff state. Ordinary
NumPy error policies and warning/error callbacks remain part of the supported
behavior. The [decision and measurements](development-protocol.md#experimental-anls-training-state-reuse)
describe the tested scope, retained costs and separate integration-validation
requirement.

## Experimental ANLS stopping and repeated iterates

The experimental `ANLSBPP` in `scripts/experimental_nmf.py` retains its outer
criterion: projected-gradient norm no greater than `tol` times its initial
norm. A nearly stationary rank-one initialization can make that denominator
extremely small, so accurate factors can still fail the relative test. The
repeat policy does not change that criterion or declare such fits converged.

For rank one with positive `tol`, a detected exact repeat of an earlier accepted
factor pair ends further sweeps. Original inner-solve failure,
objective/nonfinite checks and ordinary convergence take priority. Initialization
is not indexed as an accepted sweep. `tol=0` and higher ranks bypass repeat
detection and retain their original stopping behavior.

A repeat leaves `converged_=False`, sets
`convergence_reason_="repeated_iterate"`, and emits `ConvergenceWarning`.
`n_iter_`, objective/gradient histories and `relative_projected_gradient_`
describe the actual completed prefix; the terminal ratio can differ by cycle
phase. `cycle_event_` is `None` without a detected repeat, otherwise a dictionary
of scalar metadata: `method`, `detector`, `detected_iteration`,
`matched_iteration` and `period`. The period is the distance between the
matched states, not a proof of a minimal mathematical cycle. No retained factor
arrays are exposed through this diagnostic.

CRC32 fingerprints index the complete bytes of contiguous accepted factors.
Shape/dtype keys and complete bit equality for both W and H, including signed
zero, verify every repeat and retain its first matching iteration. CRC
collisions can increase comparisons and cost, but cannot alone stop a fit;
neither can near-equal factors, equivalent products or factor rescaling. Other
layouts use the existing continuously advanced Brent checkpoint. No option is
added. See the [indexing decision and measured tradeoffs](development-protocol.md#exact-rank-one-repeat-indexing).

The fit-local history still retains references to immutable accepted factors
until exit; fingerprinting does not copy those arrays. For rank one, its factor
payload grows as `O(max_iter * (n_samples + n_features))`, in addition to hash
and Python-object overhead. CRC scans both contiguous factors for each accepted
pair passed to repeat detection. This is an effort limit based on represented
states: it does not prove optimality or universal recurrence of hidden
native-library state.

The following measurements describe the original stopping-policy investigation,
before this CRC indexing change.

The supporting isolated study used the existing Windows AOCL and MKL environments
with one native thread. Cycling anchors showed about 96–99% lower measured
`fit_transform + transform + external diagnostics` time than fresh unchanged
current-source runs. Measured noncycling controls also showed real costs of
roughly 1–3%; no universal speedup or nonregression claim follows. On large-factor
controls, 128 retained states represented about 67 MB of logical factor payload.
Matching stop/observe runs on both real datasets and both backends continued
observation to 1,000 sweeps without later ordinary convergence or material
factor/query changes, using the original `rtol=1e-9`, `atol=1e-10` comparisons.
These are finite checks, not guarantees for every input, backend or thread count.

The local investigation record is
`tmp/rank1-stopping-exploration/SUMMARY.txt`, with source snapshots, plans and
checked output archives under the same study directory; those raw records are
not included in the distribution. This evidence concerns isolated prototypes
and does not replace validation of the exact integrated source. It does not
revise the frozen October comparison or historical release measurements below,
and adds no public solver option.

## Experimental diagnostic cost — November 2, 2026

The descriptive `projected_gradient_kkt` diagnostic now avoids redundant finite
scans and combines the protected reconstruction crop with subtraction into one
owned C-order residual. Private diagnostic helpers preserve the original solver
helper, ANLS iterations, stopping rules and fitted factors. Native selection,
protected product dimensions, full reduction dimension and float64 arithmetic
remain unchanged. When sklearn bypasses input validation with `assume_finite`,
the original X/H norm finite checks remain enabled; nonfinite gradients retain
the original checked norm path and operation order.

Fixed measurements in the existing local environments used eight independent
processes per environment and one native thread. The selected prototype reduced
the diagnostic call time relative to the then-current source by:

| Environment | CelebA | Fashion-MNIST |
| --- | ---: | ---: |
| AOCL | 6.59% | 6.41% |
| MKL | 4.02% | 5.72% |

These are diagnostic-call measurements, not whole-fit speedups or a new comparison
with release 0.4.1. The concrete patch matches the prototype's qualified AST and
passed 429 full diagnostic cases, 28 direct helper comparisons and 91 focused
tests in each environment. On four saved live workflow states, its diagnostic
matched the current implementation and selected prototype exactly within each
replay process. The concrete source was not separately retimed. Coverage remains
limited to the recorded existing environments and thread setting; historical
study results and the original failed live-checker attempt remain preserved.

Retained development evidence is under `tmp/diagnostic-cost-investigation/`,
including `COMPLETION.json`, `PROPOSAL.json`, `CONFIRMATION-ANALYSIS.json` and
`REPLAY-PROPOSAL-{aocl,mkl}.json`. These local records and their source, runtime,
archive and result hashes are not included in package distributions.

## Development results

The comparison completed on October 5, 2026, with 89 qualified selected models
and three valid timing repetitions per model. It selects SPA under the declared
accuracy/time rule and remains `selected_for_audit`.

The October 5 accuracy and timing results describe the study's frozen source
and its `fit_transform` training-coordinate convention. Since
[v0.3.0](release-0.3.0.md#spa-training-coordinates-migration), the default SPA
classifier uses fixed-basis projections for training as well as prediction.
That changes the classifier's training coordinates and adds projection cost;
the historical measurements are not measurements of the later default.

| Dataset | Method | Balanced accuracy | Fit + predict (seconds) |
| --- | --- | ---: | ---: |
| CelebA | CD | 73.51% | 3.736 |
| CelebA | SPA | 73.10% | 1.423 |
| CelebA | MU | 72.23% | 2.451 |
| CelebA | ANLS-BPP | 71.88% | 22.421 |
| Fashion-MNIST | CD | 86.58% | 7.443 |
| Fashion-MNIST | SPA | 88.33% | 3.357 |
| Fashion-MNIST | MU | 88.22% | 2.439 |
| Fashion-MNIST | ANLS-BPP | 87.78% | 95.538 |

CelebA rows average two attribute cases; Fashion-MNIST rows average 18
task/size/seed cases. Timing is the mean of case-level medians of three paired
fit-plus-prediction measurements, excluding tuning. Prediction batches contain
4,000 CelebA or 2,000 Fashion-MNIST observations. These are measurements of the
declared workloads on the development machine, not general speed ratios.

The selection rule weights the two datasets equally and permits deficits of
one balanced-accuracy percentage point overall and per dataset, and two points
per task. A five-percent cost band applies among accuracy-qualified methods.
Changing task coverage or the prediction workload can change the selected
method. On the matched 8,000-row Fashion-MNIST baseline comparison, screened
PCA, screening-only KNN and raw KNN each had higher mean balanced accuracy than
the four NMF methods. This study does not establish an NMF advantage over those
baselines or evaluate official test partitions.

The full local record is
`runs/four-solvers-20261004-repair/report/report.md`, with machine-readable
results and checksummed experiment records under the same run directory. Raw
run records and datasets are not included in the package distribution.

The combined [four-solver orchestration](../scripts/run_four_solver_benchmarks.py)
accepts an environment independently of its directory name. The former
`py12_df_aocl` basename restriction is removed; this does not qualify every
numerical runtime or change the scientific protocol. Actual source, executable
and runtime identities remain recorded and must match for resume.

The historical schema-1 numerical-repair migration writer has been retired
from current tooling. Its original source, migration manifests and results
remain preserved with the historical run records; their provenance can still
be checked by the reporting tools. There is no current four-solver checkpoint-migration
command. Ordinary study resume remains available when data, settings, source
and runtime match exactly. Keep historical records unchanged and use a separate
output directory for changed computations.

The [Fashion-MNIST](../scripts/fashion_mnist_study.py) and
[CelebA](../scripts/celeba_four_solver_study.py) study runners also accept explicit
protocols and output directories. To preserve this comparison's protocol,
complete both datasets' CV before either evaluation. These runners do not migrate
historical checkpoints. Existing combined runs remain tied to their recorded
sources, executable and numerical runtime; changed computations require a new run.

The benchmark scripts do not change public defaults. Interpret the result with
accuracy, fit/prediction workload, numerical reliability, two-dataset scope and
the implementation/theory qualifications together. Neither the comparison nor
the API extension establishes separability for CelebA or evaluates official tests.

### Status-file publication for new runs

The combined runner publishes `status.json` and `source-freeze.json` from
private temporary files beside their destinations. If replacement is denied,
it tries publication at most six times, with up to 1.55 seconds of deliberate
waiting in total. File I/O and scheduling can take longer; a successful first
attempt does not wait. If publication remains denied, the error is reported
and the previous complete destination file is preserved.

Ordinary failure attempts to remove only the writer's own temporary file;
cleanup can also be denied. Abrupt process exit can leave an unpublished file,
which later writes leave untouched. Preserve that evidence when investigating
a failed run. Private staging does not merge concurrent updates or replace the
runner's existing output ownership lock. A completed phase is skipped on resume
only after its completion was recorded; an unrecorded phase can be invoked again.

These writes protect complete JSON publication. They do not guarantee that a
concurrent reader can always open the file: Windows can temporarily deny an
open during replacement. They also provide no power-loss durability guarantee.
Persistent access restrictions still require attention after the bounded
attempts fail; the runner does not delete the old destination to force progress.

The runner and this guide are part of the recorded source identity. This source
change requires a new output directory for a new combined run. Resume existing
runs only with their exact recorded source, settings, executable and numerical
runtime; do not rewrite their source freezes or historical results. Environment
directory names do not qualify runtimes or permit reuse across changed identities.

Bounded measurements used 12 balanced blocks of fresh and completed-resume
orchestration, with real source freezing and JSON writes but simulated dataset
phases. Mean added time was about 2.28 ms fresh and 0.30 ms on completed resume
on local Windows, and 0.69 ms and 1.21 ms respectively under WSL. Unchanged-code
controls also varied, especially under WSL. These are observations of
administrative file work, not estimates of fitting, prediction, full historical
study time or the frequency of naturally occurring write failures. Denied
publication can incur the separate retry waits described above.
