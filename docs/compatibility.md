# Runtime compatibility

## Current support policy

Current source requires NumPy 1.26.3 or later and SciPy 1.15.0 or later.
SciPy 1.16.0 and 1.16.1 are excluded because repeated calls to
`scipy.optimize.nnls` leak memory. SciPy fixed the leak in 1.16.2.
Release 0.6.0 declared `numpy>=1.26` and `scipy>=1.11`.
[SciPy 1.16.2 release notes](https://docs.scipy.org/doc/scipy/release/1.16.2-notes.html#issues-closed-for-1-16-2)

On November 24, 2026, we compared SciPy 1.15.0 and 1.16.2 with NumPy 1.26.3
on CPython 3.11. On each host, Windows (AOCL) and Ubuntu (MKL), both SciPy
versions produced identical public test outcomes, including the failures
and errors observed in that run.

On November 26, 2026, we tested the package with all declared runtime and
extra dependencies at their minimum versions, including NumPy 1.26.3 and
SciPy 1.15.0. We ran the full public suite on CPython 3.11 and 3.12 on
both native Windows (AOCL) and physical Ubuntu (MKL). Each of the four
runs collected 6,301 tests and finished with zero failures or errors.
Between 253 and 435 tests were skipped per run.

Current source requires scikit-learn 1.9.0 or later. With 1.8.0, a
caller-supplied, uniform-weight Manhattan KNN using brute-force search failed
to predict string class labels. In our three-fold check, calibration warned
and returned a training-majority classifier. With 1.9.0,
calibration returned the requested KNN and evaluation classified both held-out
observations correctly on Windows (AOCL) and Ubuntu (MKL).
[Upstream KNN fix](https://github.com/scikit-learn/scikit-learn/pull/33048)

Release 0.6.0 declared `scikit-learn>=1.6.1`. If an earlier study used
scikit-learn 1.6.x or 1.7.x, upgrading can change automatically generated
grouped CV folds even with the same seed. Scikit-learn 1.8.0 corrected how
the splitter associates class counts with subjects after shuffling. To reuse
a valid prior partition, pass its saved fold indices with the original row
order and group labels.
[Upstream grouped-fold fix](https://github.com/scikit-learn/scikit-learn/pull/32540)

Current source requires threadpoolctl 3.7.0 or later. On Ubuntu MKL, a limit
set in one Python thread now applies to that thread alone. If you start your
own worker threads, set the MKL budget inside each worker that needs a limit.
The package sets limits inside its own projection workers. On Windows AOCL,
3.7.0 also removes a spurious OpenMP warning that could raise an exception
during fitting when applications treated runtime warnings as errors.
[threadpoolctl 3.7.0 release notes](https://github.com/joblib/threadpoolctl/blob/3.7.0/CHANGES.md)

Current source requires Joblib 1.6.0 or later on all supported Python versions.
On CPython 3.15.2, two-worker `FeatureSPA.transform` calls with Joblib 1.5.2
and 1.5.3 returned the expected coordinates on Windows AOCL and Ubuntu MKL,
but their loky resource tracker rejected messages used to track temporary
resources for cleanup. The temporary directories were empty after normal exit.
Joblib 1.6.0 bundles loky 3.6.0 and completed the same calls with clean stderr.
Release 0.6.0 declared `joblib>=1.3`.
[Upstream Python 3.15 issue](https://github.com/joblib/loky/issues/624) ·
[Joblib 1.6.0 release notes](https://joblib.readthedocs.io/en/stable/CHANGES.html#release-1-6-0-2026-08-31)

Python 3.10 support ended on October 20, 2026. Current source requires Python
3.11 or later; Python 3.12 with AOCL remains the primary development
environment. This removes Python 3.10 from future compatibility work and
validation plans without changing the numerical algorithms or reducing
performance to accommodate an older runtime.

The declared minimum is an installation requirement, not a claim that a new
validation run has been completed on every eligible Python version.

Existing releases, frozen Python 3.10 bundles, requirements, and receipts remain
historical records. Their original support declarations and results are not
changed retroactively. Do not use them as installation guidance for current
source.

The three-label and shared-screening setup/launch workflows are frozen study
tooling, outside current runtime support and qualification. Their retained
scripts, package locks and guides describe the original kits; they are not
installation instructions for the current package. Preserve their source and
runtime checks, and do not resume a frozen run with changed source. For current
subject-separated tuning, final fitting and held-out evaluation, use the
[grouped example](../examples/grouped_tuning.py) and [public API guide](api.md).
Retaining historical tools does not qualify new runs or require maintaining
Python 3.10 compatibility.

## Test runner

The test extra requires `pytest>=9.0.3,!=9.1.0` on all supported Python
versions. Pytest 9.0.3 includes the Unix temporary-directory mitigation for
CVE-2025-71176. Version 9.1.0 rejects an indirect override of a parametrized
fixture during collection, which blocks 32 solver tests. Version 9.1.1
restores collection. On Ubuntu MKL with Python 3.15.2, pytest 9.0.3 and
9.1.1 produced the same outcomes for all 5,697 distributed tests, including
the same 44 failures.
[pytest 9.0.3 release notes](https://github.com/pytest-dev/pytest/releases/tag/9.0.3) ·
[pytest 9.1.1 release notes](https://github.com/pytest-dev/pytest/releases/tag/9.1.1)

## pandas in the test extra

Pandas is optional for ordinary installations. Installing the `test` extra
adds pandas with a minimum that depends on the Python version.

| Python | Minimum pandas |
| --- | --- |
| 3.11–3.12 | 2.1.2 |
| 3.13 | 2.2.3 |
| 3.14 | 2.3.3 |
| 3.15 and later | 3.0.6 |

On Python 3.11–3.12, pandas 2.1.2 through 2.2.1 require NumPy below 2.
If an environment requires NumPy 2, pip therefore selects pandas 2.2.2 or
later. The newer Python bounds follow pandas' interpreter support. The
pandas-focused tests passed at each minimum on Windows AOCL and Ubuntu MKL
through Python 3.15.
[pandas NumPy 2 support](https://pandas.pydata.org/docs/whatsnew/v2.2.2.html)

## Matplotlib in the study extra

Matplotlib is optional for ordinary installations. The `study` extra
installs it for plotting. Its minimum depends on the Python version.

| Python | Minimum Matplotlib |
| --- | --- |
| 3.11–3.12 | 3.7.3 |
| 3.13 | 3.9.2 |
| 3.14 | 3.10.5 |
| 3.15 and later | 3.11.2 |

On Python 3.11–3.12, Matplotlib 3.7.3 through 3.8.3 require NumPy below 2.
If the environment requires NumPy 2, pip selects Matplotlib 3.8.4 or later.
The study reporters rendered PNGs at each minimum on Windows x64 with AOCL
and Ubuntu x86-64 with MKL through Python 3.15.
[Matplotlib 3.8.4 release notes](https://github.com/matplotlib/matplotlib/releases/tag/v3.8.4)

## Pillow in the examples and study extras

Pillow is optional for ordinary installations. Both the `examples` and
`study` extras require Pillow 13.0.0 or later. The CelebA loader calls
`Image.open()` on local files. Pillow 13.0.0 fixes excessive memory use
that can occur while reading malformed JPEG EXIF metadata, before the
loader resizes the image. On Windows AOCL and physical Ubuntu MKL,
Pillow 13.0.0 produced the expected grayscale pixel values and all three
study PNGs on Python 3.11–3.15.
[Pillow JPEG metadata advisory](https://github.com/python-pillow/Pillow/security/advisories/GHSA-wqgw-9xfh-qf2g)

## Class-label strings

Version 0.5.0 rejects class labels ending in U+0000 (NUL). Earlier source
could accept these values and silently compute incorrect class statistics,
predictions or evaluation metrics. Ordinary labels and NUL followed by other
text retain their behavior. This is an explicit input restriction; the package
does not trim or merge categories.

Original labels must remain available for detection. Fixed-width Unicode arrays
can erase a terminal-NUL distinction before package entry, and casting such an
array to `object` does not restore it. Older fitted models still containing
unsupported labels require safely encoded original targets and refitting.
See the [class-label domain](api.md#class-label-domain) for API scope and encoding.

## CD/MU numerical reproducibility

A fixed `random_state` controls initialization randomness; it does not promise
identical CD/MU factors or predictions across numerical environments. Record
dependency versions and binary builds, loaded BLAS/LAPACK libraries, thread
settings, input dtype and layout, source revision, data and model parameters
when reproducing a study. Matching version numbers alone is insufficient.

CD and MU use scikit-learn's NNDSVDa initialization and Frobenius objective.
Small floating-point differences during initialization or iterative updates can
lead to different fitted coordinates. NNDSVDa's truncation and mean filling of
zeros can amplify some initialization differences, but the largest initial
factor discrepancy need not determine the final model. Similar reconstruction
errors, identical iteration counts or warning-free fits do not establish
identical classification behavior.

CD/MU retain accepted float32 factorization inputs; the package does not silently
promote them to float64. Supplying float64 consistently for training and
prediction can improve agreement across environments, with workload-dependent
execution and memory costs. It does not guarantee identical results or higher
classification accuracy. Solver defaults, initialization, tolerances and input
precision remain explicit parts of the study configuration.

Mixed float32/float64 queries and fitted bases use float64 projection without
narrowing either array. Matching dtypes keep their original calculation. See
[prediction dtype](api.md#prediction-dtype) for output precision and costs.

The fixed-basis MU numerical fallback uses float64 logarithms for extreme-range
queries and returns coordinates in that projection dtype. It does not promote
training factorization inputs or change the fitted basis. See the
[projection diagnostics](api.md#fit-diagnostics) for its scope and cost.

See [fit diagnostics](api.md#fit-diagnostics) for the meaning of convergence
flags and [numerical test interpretation](../research/engineering/development-protocol.md#numerical-test-interpretation)
for the distinction between composition checks and environment comparisons.

## Historical Python 3.10 profile

The original supplied compatibility profile described an x86-64 Ubuntu 22.04.5 LTS
system running Python 3.10.12 with the following numerical stack:

| Package | Version |
| --- | --- |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.0 |
| joblib | 1.5.1 |
| threadpoolctl | 3.6.0 |

These five versions are recorded in
[requirements/robust.txt](../requirements/robust.txt). This file preserves the
historical target runtime for reproducibility. It is not the current recommended
environment and should not be applied to an existing development environment.

## Historical validation scope

On October 5, 2026, the complete source test suite passed in an isolated
Windows 11 environment using Python 3.10.12 and these exact package versions:
303 tests passed in 65.999 seconds, with no skipped tests. Pillow 12.3.0 enabled
the image-loader fixtures; the new report test also needed the optional
Matplotlib dependency (3.10.9). Installation preserved all five numerical pins,
and `pip check` reported no broken requirements. The runtime reported NumPy
OpenBLAS 0.3.29 and SciPy OpenBLAS 0.3.28. The primary Python 3.12 environment
passed all 303 tests in 121.042 seconds while other study jobs were active;
these suite times are not cross-environment speed comparisons.

At that revision the package declared Python 3.10 or later. These results
validate that tested source revision on Windows; they did not establish Ubuntu
execution. Matching Python, package and OpenBLAS version numbers does not establish identical
binary builds, Linux behavior or timing. Each runtime's actual libraries and
thread counts must be recorded with its results.

The 0.1.0a1 wheel installed in the exact Python 3.10.12 profile without changing
the five pinned dependencies. Installed-package fit/predict checks passed for
all six SPA/CD/MU by KNN/logistic combinations, and `pip check` found no broken
requirements. See [validation](../research/validation.md) for completed checks and their limits.

## SPA training-coordinate default in version 0.3.0

Version 0.3.0 changes `NMFClassifier`'s default to
`training_coordinates="auto"`. The 0.2.0 wheel defaults to `"fit_transform"`.
For SPA, this uses fixed-basis NNLS for both the classifier's training
coordinates and later predictions. For CD and MU, it retains the coordinates
returned by `fit_transform`. The standalone `FeatureSPA` API and its behavior
inside generic sklearn pipelines are unchanged; see
[training coordinates](api.md#training-coordinates).

SPA training now includes an additional projection, recorded in fit time.
Retrained models can produce different predictions and accuracy, with no
guarantee of improvement. Use explicit `training_coordinates="fit_transform"`
to reproduce the prior coordinate convention, together with the original
recorded runtime and other settings. Existing saved study results remain
historical evidence and are not revised by this source change. The corrected
simulation portable retains its pinned released 0.2.0 wheel and explicitly
projects training observations; it does not depend on this new default.

The coordinate behavior in the version 0.3.0 section above describes that
historical release. Since version 0.5.0, newly fitted
`FeatureSPA` pipelines and every accepted SPA `NMFClassifier` coordinate mode
use fixed-basis NNLS for training. Explicit `training_coordinates="fit_transform"`
no longer selects anchor coordinates on refit with current source. Reproducing
an older anchor-trained study requires its archived implementation and settings;
loading an existing fitted model does not retrain it or rewrite its metadata.

## Use an existing target environment

Use Python 3.11 or later with the dependencies in the
[current support policy](#current-support-policy). Python 3.12 is the
primary development target. In the preconfigured environment, install only
the project from the source checkout, then check that the installed
dependencies satisfy the declared requirements.

```shell
python -m pip install --no-deps .
python -m pip check
```

Image-loading examples additionally require Pillow, and plotting requires
Matplotlib.

## Optional Python 3.12 environment for the October studies

For the published 0.6.0 release, follow the [installation guide](../README.md#installation).
To install current development source, run `python -m pip install .` from its checkout.
To reproduce the October studies’ five Python 3.12 numerical dependency versions,
install the [study requirements](../requirements/robust-py312.txt) in a separate
environment and run the following commands from the archived study checkout.

```shell
python -m pip install -r requirements/robust-py312.txt
python -m pip install --no-deps .
```

Screening, fitting and prediction also run with the stock OpenBLAS wheels. Some optimizations depend on the loaded libraries as well as their
versions. For example, repeated explicit QR projections recompute the factors
on OpenBLAS, while a qualified MKL configuration can reuse them. Record the
loaded BLAS libraries and thread counts when comparing runs.

## Historical version 0.2.0 validation

The [October 9 release record](releases/release-0.2.0.md) and
[implementation record](../research/engineering/implementation-improvements.md) preserve the evidence
for the released 0.2.0 source: 370 tests passed on Windows Python 3.12 and
Python 3.10. Earlier Linux Python 3.12 checks covered projection and thresholds;
the final workspace-lifetime change was checked on Windows. These are historical
results, not validation of the current modified source. Current source requires
Python 3.11 or later.
