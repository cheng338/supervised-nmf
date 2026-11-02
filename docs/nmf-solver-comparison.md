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

Full-history fingerprints index contiguous factors; a match must also pass
complete bit equality for both W and H, including signed zero. Hash collisions,
near-equal factors, equivalent products or factor rescaling alone cannot stop a
fit. Other layouts use a continuously advanced Brent checkpoint. The fit-local
history retains immutable accepted factors until exit. For rank one, its factor
payload grows as `O(max_iter * (n_samples + n_features))`, in addition to hash
and Python-object overhead; hashing scans both contiguous factors each accepted sweep.
It is an effort limit based on represented states: it does not prove optimality
or universal recurrence of hidden native-library state.

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
and [numerical-repair recovery](../scripts/prepare_four_solver_recovery.py)
scripts preserve the recorded historical workflow and require an environment
whose basename is `py12_df_aocl`. This name check is an administrative
restriction, not verification of the AOCL binaries.

For fresh studies in other environments, the
[Fashion-MNIST](../scripts/fashion_mnist_study.py) and
[CelebA](../scripts/celeba_four_solver_study.py) study runners accept explicit
protocols and output directories without that restriction. To preserve this
comparison's protocol, complete both datasets' CV before either evaluation.
These runners do not replace historical checkpoint migration. Existing combined
runs remain tied to their recorded sources, executable and numerical runtime,
and a different environment requires a new run or a separately qualified migration.

The benchmark scripts do not change public defaults. Interpret the result with
accuracy, fit/prediction workload, numerical reliability, two-dataset scope and
the implementation/theory qualifications together. Neither the comparison nor
the API extension establishes separability for CelebA or evaluates official tests.
