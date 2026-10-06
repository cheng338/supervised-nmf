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
Classification uses each method's fitted training coordinates, retaining their
original scales and the resulting classifier geometry.

The public MU path preserves that study's unregularized Frobenius objective and
NNDSVDa initialization. It exposes the existing `max_iter`, `tol`, `random_state`,
and `training_coordinates` parameters. The default `"fit_transform"` training
coordinates preserve the study's fitted-coordinate convention; `"transform"`
explicitly recomputes them against the fitted basis. The comparison-model
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

## Development results

The comparison completed on October 5, 2026, with 89 qualified selected models
and three valid timing repetitions per model. It selects SPA under the declared
accuracy/time rule and remains `selected_for_audit`.

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

The benchmark scripts do not change public defaults. Interpret the result with
accuracy, fit/prediction workload, numerical reliability, two-dataset scope and
the implementation/theory qualifications together. Neither the comparison nor
the API extension establishes separability for CelebA or evaluates official tests.
