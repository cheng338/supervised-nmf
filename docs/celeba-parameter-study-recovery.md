# Resuming the CelebA parameter study

`examples/celeba_parameter_study.py` reuses compatible completed development
results. Its optional final-fit recovery also saves a selected, completed fit
before evaluating it. This option belongs to this example; it does not add model
recovery to other examples or to the general calibration APIs.

## Choosing final-fit recovery

`--final-fit-recovery` accepts `off` or `all`; its default is `off`.

| Value | Work performed | When to use it |
| --- | --- | --- |
| `off` | Keeps existing CV checkpoints and completed final JSON results. Does not save or load fitted-model checkpoints or write recovery attempts. | Ordinary runs where avoiding checkpoint overhead matters more than preserving a completed fit across an interruption. |
| `all` | Writes a verified fitted outcome before evaluation, or a terminal JSON outcome when no usable fit exists. Checks compatibility and records recovery attempts. | Your own trusted local study when repeating a completed final fit would be costly and you want recovery after its checkpoint commits. |

The choice is saved in the run configuration and must match on resume. There is
no duration threshold or interruption-only mode. `all` covers every declared
method, including cheap methods for which saving can cost more than fitting.
It adds identity checks, disk writes, retained model files and attempt records
to an ordinary successful run. It is not a claim that every run or resume is
faster. `off` still uses the strict report checks and exclusive finalization
ownership described below.

The default remains `off` because completed-result reuse already avoids repeated
finished work, while fitted-state recovery protects only the narrower period
between a completed fit checkpoint and the final result. Historical studies do
not establish how often that interruption occurs. `all` adds work to successful
ordinary runs and an exact compatibility/trust contract, so choose it deliberately
for a study where preserving that completed fit matters. This default is not
based only on a small timing percentage.

**Use `all` only with checkpoints from your own trusted study.** The fitted-model
formats are uncompressed joblib and protocol-5 pickle with an owned cache
buffer. Both use pickle. A checksum detects changed
bytes; it does not establish who created them or make an untrusted model safe to
load. `--resume` selects your existing run directory, not a downloaded model or
a general model-import file. Keep that directory under your control.

For example, create a new run with an existing compatible data cache:

```text
python examples/celeba_parameter_study.py --data-cache data/celeba-development-4000-seed42 --workers 4 --threads 1 --final-fit-recovery all
```

Use the run directory printed by that invocation when resuming:

```text
python examples/celeba_parameter_study.py --data-cache data/celeba-development-4000-seed42 --workers 4 --threads 1 --final-fit-recovery all --resume runs/celeba-parameter-study-RUN_ID
```

Replace `RUN_ID` with the actual directory name. The directory must contain
the original `configuration.json`, `data.json`, and `provenance.json`. Keep
the original grid, seed, worker count, thread count, and recovery policy.
Grid values must match their saved JSON types, including the distinction
between `0` and `0.0`. Use `--max-train` and `--max-validation` values that
match the number of saved image IDs in each split. Omit
`--final-fit-recovery all`, or specify `off`, for the default policy on a new run;
do not change policies within an existing run.

`--data-cache` remains fingerprint-verified and takes precedence over image
paths. `--grid-json` defines the complete declared candidate inventory. Changing
it requires a new run. During CV, `--workers` controls the number of concurrent
workers and `--threads` sets the native numerical threads per worker. Multiply
the two when budgeting CV threads. After CV, the example times final fitting
and prediction one model at a time with one worker and one native thread.

If the cache is missing, supply the original raw inputs to rebuild it. The
runner verifies the rebuilt manifest before starting CV. A cache originally
created with a limit larger than the available split can have a different
creation limit and saved row count. In that case, restore the original or an
identically rebuilt cache and resume with the saved row count.

## What a resume reuses

After validating the declared CV reports and selecting the same candidate, the
runner checks each attribute/method in this order:

| Available state | Action |
| --- | --- |
| Valid, compatible `development-evaluation.json` | Reuse that complete result, including a `held` or `failed` result. No model load, final fit or evaluation. |
| Committed terminal fit outcome, with `all` | Recreate the recorded held/failed result from metadata. Do not load a model or retry the fit. |
| Committed usable fitted outcome, with `all` | Verify and restore the fitted model, evaluate it, then publish the final result. No new fit or KNN-head reconstruction. |
| No committed fit outcome | Fit again when a valid candidate exists. Temporary payloads and attempt records are evidence, not reusable checkpoints. |
| Corrupt or incompatible committed state | Stop and preserve the existing files. Do not silently refit or choose a different generation. |

A valid completed final result takes precedence over unneeded fitted state. A
missing model payload therefore does not invalidate that final result. A corrupt
final result is an error; it does not permit falling back to a model checkpoint.
Completed-result reuse still performs ordinary run, source and report checks;
it is not a bypass around compatibility checking.

Model recovery protects the interval **after the fit-outcome commit and before
the final-result commit**. It cannot continue an interrupted NMF, PCA, screening
or classifier fit. Termination before the outcome commit requires fitting again.
Evaluation may also run again if its results had not reached the authoritative
final commit. An `evaluation-complete` or `result-ready` attempt record does not
replace that commit.

## Compatibility and model scope

The example's 13 declared methods are covered: percentile and absolute screening
with CD, SPA or MU NMF; screened PCA; screening-only KNN; raw KNN; the majority
baseline; and unscreened CD NMF. The saved model retains its fitted representation,
training coordinates, labels and classifier state. Restoration does not repeat
factorization, anchor selection, preprocessing or neighbor-tree fitting.
This is not a general interchange format for custom estimators or arbitrary
modified pipelines.

Screening-only, majority, NMF, SPA and PCA models retain complete joblib state.
Eligible raw KNN instead serializes the complete native model through public
pickle protocol-5 buffers. One buffer references the verified original training
cache; the rest of the fitted state stays in the payload. Loading supplies
an owned read-only copy of that verified matrix to public pickle restoration.
This choice is automatic within `all`, with no additional option and no
second head fit. The normal save representation does not copy the matrix.

The eligible construction uses the exact declared raw KNN settings: uniform
Euclidean neighbors, default `auto` algorithm, one job, and a read-only native
C-contiguous `float32` matrix with 4096 columns and native `int64` labels `0` and
`1`. Only a read-only serialized buffer with the same starting address and byte
length as the whole verified training matrix can be referenced. Equal-valued
copies, partial overlaps and noncontiguous views do not establish that match.
No named private fitted field is inspected, removed, replaced or inferred.

Saving first discovers the representation in memory. No matching buffer,
multiple matches, or an unsupported layout selects complete joblib before any
buffer payload is written. Discovery stops before inlining a nonmatching buffer
at least as large as the training matrix, avoiding a redundant matrix copy before
fallback. Genuine serialization errors propagate; they are not format choices.
Ordinary additional fitted attributes and small arrays remain part of the saved
native state. The original estimator is not changed.

After a buffer reference commits, missing or incompatible cache context is an
error. Recovery does not replace that choice with a refit or another payload.
The loader checks the declaration and payload and requires exactly one consumed
external buffer with no trailing payload data. The loaded model preserves
complete saved native state and its array aliases, using an independent
read-only snapshot of the training matrix. It is not refitted or reconstructed
from estimator parameters.

Keep the verified original training cache present and unchanged until loading
and its copy finish. Read-only views alone cannot prevent another process or
writable alias from changing the source while it is copied. The study owns
and verifies its cache before use. After a successful load, the restored
model retains its owned snapshot and no longer depends on the training-cache
mapping or backing file. Releasing the original context or later changing
that file cannot change the restored model. The cache is still required for
a future restore, and every study invocation retains its normal data checks.

The cache context is local to the current call, with method, attribute, run,
evaluation, manifest and training-data identities. The CLI verifies actual cache
fingerprints before aggregation; this branch does not rehash the entire matrix
for each model. An independent aggregation/helper caller must provide the same
verified training data and keep it immutable through restoration and copying.
Validation inputs must remain unchanged during evaluation as usual. Merely
supplying matching identity strings is not a substitute for that input check.

Complete joblib loading uses its default native-byte-order behavior, supported
since the declared joblib minimum. The public-buffer branch requires the
standard native cache layout and preserves the serialized native model state.
Exact non-native byte-order preservation is not a cross-format guarantee.

Fitted outcomes use the `celeba-final-fit-outcome-v1` receipt schema, with
`celeba-final-fit-joblib-v1` for complete state or
`celeba-final-raw-knn-public-buffer-v1` for the public cache-buffer reference.
Compatibility includes the run/grid and
selected candidate, attribute/method, verified data manifest and sample order,
numerical controls, runner/package/helper sources, Python executable and version,
loaded dependency versions, and initialized native-library inventory and hashes.
The native thread setting is part of that identity. Values, shapes, dtypes and
sample IDs are covered by the existing verified data manifest.

The runner refreshes source and active-runtime identity before reading a method's
fit outcome, immediately before restoring its model, before publishing a new
outcome, and after evaluation. Native-library hashes may be reused within one
aggregation only while their observed file metadata is unchanged; a new
aggregation starts a new hash cache. This assumes the installed runtime remains
unchanged. It does not inspect already mapped machine-code bytes or promise to
detect deliberate changes that preserve all observed file metadata.

Within one aggregation, unchanged initialization hooks and thread requests can
reuse the initial native-library warm-up. Each identity check still refreshes
current library membership and metadata; the requested native thread setting
remains part of its identity. A new aggregation,
a changed initialization hook or thread request, or a failed initialization
requires fresh initialization. There is no persistent cached runtime identity.

The optimized library-discovery adapter is used only when its private dependency
implementation is recognized; otherwise the public discovery path is used.
Positive library discovery and native getters remain current. Its routing state
belongs to the ongoing aggregation. A controller retained beyond an observation
resumes original discovery behavior for later rescans, so it does not retain
that observation's negative-path decisions.

Use a new run after changing bound configuration, data, code or runtime. There
is no automatic migration, fallback from incompatible committed state, or
conversion of older runs. Preserve historical runs with their original compatible
workflow. Do not edit identities or checksums to force reuse. Completed final JSON results retain
their existing compatibility rules; the stronger fitted-state identity applies
when a fitted outcome is used.

## Measured costs and limits

Local checks used the existing MKL and AOCL environments, with one native thread
per process. Complete CLI runs and resumed runs covered all 13 methods on two
attributes. A small synthetic fixture had 40 training observations, 12 validation
observations and 4096 features; its 26 method/attribute results were 20 valid and
six held. Numerical outputs and qualification matched between recovery policies.
This is a workflow check, not evidence about real-data predictive accuracy.

Five paired, unmodified aggregation calls per environment measured final fitting,
evaluation, report checks and checkpoint work after CV reports were available:

| Aggregation phase and policy | MKL median | AOCL median |
| --- | ---: | ---: |
| Fresh final work, `off` | 1.745 s | 1.727 s |
| Fresh final work, `all` | 3.550 s | 3.657 s |
| Already completed results, `off` | 1.037 s | 1.031 s |
| Already completed results, `all` | 1.017 s | 1.013 s |

Thus `all` roughly doubled this small final-work workload. Its identity checks,
serialization, integrity checks and attempt publication are real ordinary-run
costs, even when no interruption occurs. The table excludes CV setup and process
startup; it is not a whole-CLI benchmark. Separate fresh/resume CLI runs confirmed
workflow behavior, with one observation per policy rather than a repeated CLI
timing claim. Completed-result component fields remain historical; only the
current aggregation wall time is comparable in the completed rows.

### Direct raw KNN comparison with the pre-change default

A later fixed comparison used the existing 4000-by-4096 real-data training
cache and `k=3`, with 128 measured calls in fresh child processes and one native
thread. Each of four environment/workflow sequences had eight balanced blocks,
with duplicate calls to each implementation and two complete ordering periods.
Four recovery-preparation calls were excluded. The original fit and prediction
code ran without phase observers or extra predictions; byte audits followed the
timed sequences. Numerical results, fit/load counts and preserved state passed
the comparison checks.

The AOCL environment also loads MKL; these environment labels do not identify
the backend of every numerical operation.

Fresh calls compared `all` with the frozen pre-change runner's default behavior,
not with the new `off` path. This measures the whole enabled-policy difference.
Recovered calls compared the owned-copy implementation with complete-joblib
fitted-state recovery, **not** with refitting or the default `off` policy.
The table gives mean differences in milliseconds, proposed minus reference;
positive values mean more time. Intervals use eight paired block means and a
small-sample t calculation, adjusted across the six primary comparisons for
approximately 95% family coverage. Serial ordering and the small number of
blocks limit their interpretation.

| Workflow and numerical operation | MKL mean difference [interval] | AOCL mean difference [interval] |
| --- | ---: | ---: |
| Fresh fit | -0.41 [-2.53, +1.70] ms | +0.68 [-0.75, +2.11] ms |
| Fresh prediction | -23.40 [-70.17, +23.37] ms | +28.74 [-26.73, +84.21] ms |
| Recovered prediction | -20.47 [-84.65, +43.71] ms | +29.66 [-55.41, +114.73] ms |

AOCL prediction was about 1.1% slower on average in both workflows, with
positive differences in seven of eight fresh blocks and six of eight recovery
blocks. The upper interval endpoints correspond to about 3.3% and 4.3%; these
are uncertainty bounds, not observed worst cases. MKL prediction means were
about 0.9% and 0.8% faster. None of the adjusted intervals excludes zero, which
does **not** establish timing equivalence or eliminate the observed AOCL cost.
Recovered fit-time fields describe the original fit; no new fit was performed.

Mean whole-call time increased by 81.68 ms in MKL and 150.79 ms in AOCL for
fresh `all` versus the previous default. Recovery was 93.87 ms faster in MKL and
35.98 ms faster in AOCL than complete-joblib recovery. These secondary recovery
gains do not cancel the measured prediction slowdown, and do not measure the
benefit of avoiding a repeated fit. The observations describe this local raw-KNN
workload; they are not a general NMF timing claim or a promise of unchanged
numerical duration.

### Earlier checkpoint and recovery measurements

The earlier raw KNN comparison used the existing 4000-by-4096 real-data training cache,
`k=3` and the same completed-fit recovery contract. Its baseline saved and loaded
the complete model with uncompressed joblib. Five paired ordinary runs measured
the public-buffer save path, which is unchanged in the owned-copy implementation;
the new copy loader does not execute during a fresh fit. These are paired median
differences, with a negative value meaning less time than complete-state recovery:

| Ordinary raw KNN component | MKL difference | AOCL difference |
| --- | ---: | ---: |
| Checkpoint publication | -98.67 ms | -94.80 ms |
| Final fit | +0.11 ms | -1.27 ms |
| Prediction | +3.82 ms | +26.26 ms |
| Complete aggregation | -93.96 ms | -70.65 ms |

These earlier ordinary runs retained model references during their audit;
the later recovery pairs used normal object lifetimes. These protocols differ
from the latest eight-block comparison and are not pooled with it.

This shows a benefit over the complete-state checkpoint implementation, not
that checkpointing is free relative to `off`. The ordinary prediction changes
remain part of the result: paired ranges were -29.98 to +25.67 ms on MKL and
-7.63 to +35.29 ms on AOCL. The implementation does not claim identical fit or
prediction duration merely because checkpoint work is outside their timers.

Four additional balanced pairs restored the owned-copy implementation in fresh
processes, without prediction observers or retained model references that could
extend fitted-object lifetime. The complete model was recovered without fitting;
original fit time, diagnostics, warnings and final-result semantics were retained.

| Recovery measurement | MKL | AOCL |
| --- | ---: | ---: |
| Owned snapshot preparation, median | 21.75 ms | 22.36 ms |
| Total owned-copy model load, median | 23.14 ms | 23.82 ms |
| Complete-joblib model load, median | 58.59 ms | 58.80 ms |
| Model-load paired difference | -35.46 ms | -34.95 ms |
| Prediction paired median difference | -11.28 ms | +14.41 ms |
| Prediction paired range | -26.23 to +13.68 ms | -54.61 to +100.68 ms |
| Aggregation paired median difference | -44.36 ms | -55.85 ms |
| Aggregation paired range | -143.02 to +57.67 ms | -113.55 to +49.59 ms |

The snapshot allocation and copy are included in model loading. They are not
charged to fitting or prediction, and `cache_binding_seconds` must not be added
to `model_load_seconds` again. The observed load reduction was consistent in
these pairs; prediction and whole-call effects were mixed, including a +100.68 ms
AOCL prediction observation and slower individual aggregation calls. An earlier
instrumented first-prediction diagnostic also had mixed signs: paired medians
were +14.44 ms on MKL and -34.94 ms on AOCL. That diagnostic and the later
no-retention measurements are distinct evidence, not interchangeable repeats.

The small natural-workflow audit also inspected every method/attribute slot's
fit and prediction fields. Twenty slots performed a valid final fit; six CD
slots remained held and did not fit or predict. Across the five fresh `all`/`off`
pairs, cumulative fit-time median differences were -4.04 ms on MKL and -0.50 ms
on AOCL; cumulative prediction differences were +0.29 and +3.49 ms. The largest
positive per-slot fit median was +1.98 ms for one AOCL SPA case, with a range of
-1.54 to +4.48 ms. One AOCL SPA prediction observation was +46.55 ms, despite a
paired median of +0.23 ms and a range beginning at -0.56 ms. Those outliers are
retained rather than treated as zero cost. These small-case observations do not
replace the real-data prediction results above.

In that same small workload, named checkpoint-time medians totaled about 455 ms
on MKL and 470 ms on AOCL, while explicit identity-check medians were about
1058 and 1104 ms. They identify substantial recovery overhead; sums of component
medians are not an exact decomposition of paired wall-time differences. The
default `off` path was also checked against the previous runner: fresh aggregation
paired medians were 299 ms faster on MKL and 351 ms faster on AOCL in this fixture.

The owned snapshot preserves complete native state, avoids private fitted-field
manipulation and another head fit, and removes the training-cache dependency
after successful loading. Its roughly 22 ms preparation in the earlier four-pair
recovery measurement is an explicit restore
cost. It was chosen for that ownership contract and measured practical behavior,
not to trade execution speed for memory savings. Complete joblib remains the
fallback for unsupported valid layouts. Borrowed mappings and prefaulting retain
a backing-cache lifetime requirement; reconstruction from public parameters adds
a new fit and a different preservation contract.

Choose `all` when preserving a completed, expensive fit across the protected
interruption interval is valuable enough to pay its ordinary-run overhead.
`off` remains the default. The study does not estimate interruption probability
or claim a universal break-even point. The earlier tables use four or five paired
observations; the later direct comparison above uses eight blocks. These local
measurements do not establish timing equivalence, cold disks,
an otherwise idle machine, or performance on every dataset and runtime.

The numerical timer boundaries are unchanged: serialization is outside final
fitting and prediction; snapshot preparation is inside model loading. Allocation,
cache state and system variation can still influence later numerical work.
Recovery preparation occurs only when loading a committed model; ordinary fresh
fitting and completed-result reuse bypass it.


## Reports and ownership

Finalization requires exactly the declared `chunk-*/report.json` inventory for
each declared attribute/method. Missing reports and unexpected reports, including
unexpected empty reports, cause an explicit error before final-fitting work.
The runner preserves those files. Candidate coverage must remain complete and
unique, and ties retain `ParameterGrid` order. An extra empty report must not
inflate reported tuning time.

This preflight validates the declared path inventory, not every report's complete
provenance in advance. Direct aggregation assumes reports produced by compatible
study workers. The CLI runs identity-validated workers for every expected chunk
before aggregation; direct aggregation does not replace those worker checks.

The CLI may regenerate a missing derived chunk report from valid committed CV
fold receipts before finalization. That existing repair is distinct from allowing
an incomplete grid during direct aggregation. Previously committed fold results
and attempt history remain authoritative.

Only one process or thread may finalize a given output directory at a time. A
competing finalizer fails promptly and can retry after the owner exits. The OS
releases ownership on normal exit, exceptions or process termination. The
`.final-evaluation.lock` marker can remain after release; its presence alone
does not indicate an active owner, and it should not be deleted to force access.
Separate processes may finalize different output directories concurrently. Within
one process, a finalization that needs fitting or recovery waits for another
thread's active finalization to finish. It acquires numerical ownership at its
first unfinished result and retains it through the rest of that aggregation,
including when fitted-state recovery is off. This prevents overlapping warning
capture and native thread-limit contexts from restoring process-wide state out
of order. The per-output lock is checked first, so a competing finalizer for the
same output directory still fails promptly.

Completed-only calls do not acquire this additional numerical ownership. A short
resume with unfinished work can wait behind a longer finalization; there is no
priority or fairness guarantee. Use separate processes for parallel numerical
finalization of independent outputs. Same-thread nested calls to different
outputs remain reentrant. A callback inside finalization must not synchronously
wait for another thread to finish a finalization that needs this ownership.

The per-output lock covers report preflight, final-result checks, fitted-outcome access,
fitting, evaluation and final-result publication. It does not make simultaneous
whole-CLI invocations a supported workflow: earlier CV/progress writes and the
later `comparison.json` write are outside finalization ownership.

## Outcomes, timing and attempt records

The existing numerical qualification rules remain in force. A missing valid CV
candidate produces a held result. A nonconverged final fit stays held. A fit or
evaluation exception retains its failed outcome. Recovery does not promote these
results to valid or mark tuning complete. Development-validation results remain
development evidence, not official-test results or frozen defaults.

Original fit diagnostics, fit-warning category/message/order and qualification
are retained. A recovered evaluation obtains fresh evaluation diagnostics and
warnings. `final_fit_seconds` and diagnostic fit-time fields still describe the
original fit; they are not replaced by zero or by loading time. Tuning compute
time retains the committed CV accounting and is not added again on resume.

With `all`, the final result includes a `recovery` record:

| Field | Meaning |
| --- | --- |
| `policy`, `scope` | Chosen policy and the record's scope: the invocation that committed this final result. |
| `fit_performed`, `new_fit_seconds` | Whether that invocation called final `.fit`, and its measured fit time in seconds. |
| `original_fit_performed` | Whether the recorded outcome involved an original selected fit. False for a no-valid-candidate outcome. |
| `outcome_reused` | Whether a committed fitted or terminal outcome was reused. |
| `fit_reused` | Whether a usable fitted model was restored; terminal replay is false. |
| `identity_seconds` | Time spent in explicit source/runtime checks inside final-fit recovery; excludes the outer per-method source guard. |
| `checkpoint_seconds` | Time spent publishing a new outcome in that invocation; zero for outcome reuse. |
| `verification_seconds` | Time spent validating an existing committed outcome and its payload inventory. |
| `model_load_seconds` | Time spent loading the model, including the codec's own validation. |
| `cache_reference_restored` | False: raw-buffer restoration uses an owned snapshot rather than a retained reference to the training cache. Also false for complete-state loading and new fits. |
| `cache_binding_seconds` | Raw-buffer snapshot allocation, copy and read-only buffer preparation. A subset of `model_load_seconds`, not additional time to add again. Zero when this preparation did not occur. The historical field name does not imply private-field attachment. |
| `head_reconstruction_seconds`, `reconstructed_knn_heads` | Both zero: neither complete-state nor public-buffer restoration performs a new head fit. |
| `model_bytes` | Bytes in the committed model payload; zero for terminal metadata-only outcomes. Excludes receipts, attempt files and the existing referenced training cache. |
| `generation` | The committed outcome generation used or created. |
| `kind` | Present on a newly published outcome: `fitted` or `terminal`. |

These component timers are not an exhaustive whole-call wall-time accounting.
Attempt publication and other orchestration can add work outside them. Payload
verification and model loading intentionally perform separate integrity checks;
both costs are real. Do not sum historical fit time into current resume work.
Completed-result reuse returns its historical `recovery` record unchanged; it
does not report a new fit, recovery attempt or measured current reading time.

Under each method's `.fit-recovery` directory, immutable attempt records identify
started operations, fit/evaluation milestones and recorded interruptions. They
are retained separately from the final result's timing fields. A process killed
without cleanup may leave an attempt with no terminal record. A fully written
generation is reusable only after the authoritative `current.json` commit;
uncommitted generations and temporary files are preserved and ignored, not
selected by age or apparent completeness. Repeated interruptions can therefore
increase retained disk use. This workflow does not automatically delete that
evidence.

Persistence errors such as a failed write, invalid checksum or incompatible
identity are reported as recovery errors, outside numerical failure handling.
They are not relabeled as nonconvergence or silently ignored. If recording an
interruption also fails, the original error is preserved and the secondary
failure is attached. Preserve the directory and resolve the reported problem
before retrying; start a new run for incompatible configuration or state.

The commit protocol handles process interruptions and cooperating local writers.
It does not promise power-loss durability or coordination across hosts sharing
a filesystem. A fresh-process recovery check is also not a cold-disk benchmark.
