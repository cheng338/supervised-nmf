# Runtime compatibility

## Current support policy

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
requirements. See [validation](validation.md) for completed checks and their limits.

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

## Use an existing target environment

Use an environment with Python 3.11 or later; Python 3.12 is the primary
development target. Activate the intended environment and install from the
source checkout without changing its already-provisioned numerical dependencies:

```shell
python -m pip install --no-deps .
```

This command does not install missing dependencies or validate the environment.
Image-loading examples additionally require Pillow, and plotting requires
Matplotlib.

## Provision a separate Python 3.12 environment

For a new, isolated Python 3.12 environment, the numerical version profile is
recorded in [requirements/robust-py312.txt](../requirements/robust-py312.txt).
Install those versions before installing the package:

```shell
python -m pip install -r requirements/robust-py312.txt
python -m pip install --no-deps .
```

These pins do not reproduce the Windows AOCL binaries on Linux. Record the
actual BLAS libraries and thread configuration, and assess performance in the
chosen environment. The same version numbers do not establish equivalent
numerical performance across machines.

## Historical version 0.2.0 validation

The [October 9 release record](release-0.2.0.md) and
[implementation record](implementation-improvements.md) preserve the evidence
for the released 0.2.0 source: 370 tests passed on Windows Python 3.12 and
Python 3.10. Earlier Linux Python 3.12 checks covered projection and thresholds;
the final workspace-lifetime change was checked on Windows. These are historical
results, not validation of the current modified source. Current source requires
Python 3.11 or later.
