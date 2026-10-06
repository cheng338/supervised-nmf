# Runtime compatibility

The supplied `robust` profile describes an x86-64 Ubuntu 22.04.5 LTS system
running Python 3.10.12 with the following numerical stack:

| Package | Version |
| --- | --- |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.0 |
| joblib | 1.5.1 |
| threadpoolctl | 3.6.0 |

These five versions are recorded in
[requirements/robust.txt](../requirements/robust.txt). This file pins the target
runtime for reproducibility; it is not a list of new dependencies to add to an
environment that already supplies those versions. joblib is a scikit-learn
dependency.

## Validation scope

On October 5, 2026, the complete source test suite passed in an isolated
Windows 11 environment using Python 3.10.12 and these exact package versions:
281 tests passed in 57.313 seconds, with no skipped tests. Pillow 12.3.0 was
installed for the image-loader fixtures. The runtime reported NumPy OpenBLAS
0.3.29 and SciPy OpenBLAS 0.3.28. The primary Python 3.12 environment also passed
all 281 tests in 59.991 seconds.

The package now declares Python 3.10 or later. These results validate the tested
source revision on Windows; Ubuntu execution has not been tested. Matching
Python, package and OpenBLAS version numbers does not establish identical
binary builds, Linux behavior or timing. Each runtime's actual libraries and
thread counts must be recorded with its results.

The 0.1.0a1 wheel installed in the exact Python 3.10.12 profile without changing
the five pinned dependencies. Installed-package fit/predict checks passed for
all six SPA/CD/MU by KNN/logistic combinations, and `pip check` found no broken
requirements. See [validation](validation.md) for completed checks and their limits.

## Use an existing target environment

Activate the intended environment, then run these commands from the source
checkout. First check that its Python and numerical packages match the profile:

```shell
python --version
python -m pip show numpy scipy scikit-learn joblib threadpoolctl
python -c "import sys; from importlib.metadata import version; expected = {'numpy': '2.2.6', 'scipy': '1.15.3', 'scikit-learn': '1.7.0', 'joblib': '1.5.1', 'threadpoolctl': '3.6.0'}; assert sys.version_info[:3] == (3, 10, 12), sys.version; assert all(version(name) == wanted for name, wanted in expected.items()), 'Numerical package versions do not match requirements/robust.txt'; print('Target runtime versions match.')"
```

After that check succeeds, install the package without changing runtime
dependencies:

```shell
python -m pip install --no-deps .
python -c "import supervised_nmf; print(supervised_nmf.__version__)"
```

The profile also lists cvxpy, dwd, ipykernel and a missing debugpy dependency.
These are not requirements of `supervised-nmf`; this installation does not add,
remove or repair them. A general `pip check` may still report unrelated
environment issues. Image-loading examples additionally require Pillow, and
plotting requires Matplotlib; those optional dependencies are outside the five
runtime pins above.

## Reproduce the numerical stack separately

For a new, isolated environment with Python 3.10.12, install the pinned numerical
stack before installing the package:

```shell
python -m pip install -r requirements/robust.txt
python -m pip install --no-deps .
```

Use a separate environment when the active one has different dependency
versions. The requirements file should not be applied to an unrelated working
environment merely to run the package.
