# Validation

The initial validation used small correctness checks. These checks were not
CPU/RAM capacity measurements and do not provide a PC speed comparison.

A full PC1 run completed on October 4, 2026: all six compatibility checks,
11 configurations and 33 measurement
windows passed without warnings or numerical failures. It used Python 3.12.14
and AOCL-BLAS 5.2.0, with 10,222 training rows and 2,048 features per fit. Its
original manifest SHA-256 is
`3ec06219a25ebe8c971ffa7bf80b987c5b8fd7b578031c70e540ce78603dbeec`.
These measurements belong to that frozen bundle. They can support PC1 and the
hypothetical PC3 projection when the saved manifests verify compatibility.
Changes confined to the permitted reporting files do not require a new timed
run. Changes to bundled package files, benchmark code, protocol or data require
separate qualification; the original measurements remain unchanged.

An isolated workspace virtual environment used Windows 11, Python 3.12.14, and
the supplied target's exact numerical-package versions:

| Package | Tested version |
|---|---|
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.0 |
| threadpoolctl | 3.6.0 |
| joblib | 1.5.1 |

The Windows wheels reported NumPy OpenBLAS 0.3.29 and SciPy OpenBLAS 0.3.28.
Those version numbers resemble the supplied target report, but the operating
systems, interpreter versions, binary builds, and OpenMP runtimes differ.

- `check_compatibility.py --expect-target-versions`: all six checks passed.
  It checked the frozen-source manifest, pinned versions, direct bundled-source
  import, absolute screening against SciPy's pooled t statistic (including the
  inclusive cutoff boundary), and converged/reproducible CD fit/prediction for
  float32 and float64 with held-out data leaving the fitted basis/screen unchanged.
- The t-score reference's maximum absolute difference was about `2.49e-14`.
  The small float32 and float64 models converged in 294 and 293 iterations,
  respectively, with no warnings. Different dtypes need not take identical
  iteration counts.
- `tests/test_core.py`: 24 tests passed against these exact numerical versions.
- `tests/test_datasets.py`: 6 tests passed; 8 image-loader tests were skipped
  because optional Pillow was absent. The tested numerical simulation paths
  do not require Pillow.
- The frozen-data one-worker smoke test passed with valid numerical output and
  completed fits. Its timings are validation-only and are not a PC comparison.

The archived checker output is retained locally and is excluded from the Git
repository because it contains machine-specific paths. It records the development
snapshot's manifest identity. The final distribution has
additional documentation and runner changes, so it has a different overall
manifest hash. These archived results validate the recorded development
snapshot; they do not by themselves validate later package, sweep or launcher
changes. The compatibility checker also passed all six checks in PC1's existing
environment at validation time.

These frozen-snapshot checks do **not** establish Ubuntu 22.04 / Python 3.10.12
compatibility. The original protocol called for target-version and smoke checks
on that machine. The frozen snapshot declared Python >=3.11. A later package
revision declared Python >=3.10 and has separate
[runtime compatibility results](../docs/compatibility.md), including a full
source-suite check on Windows with Python 3.10.12. Those newer checks do not
retroactively qualify this benchmark bundle or its recorded timings. Benchmark
validation used the selected environment without changing its installed packages.
Current development ended Python 3.10 support on October 20, 2026 and requires
Python >=3.11. See the [current policy](../docs/compatibility.md) before planning
new runs; the historical target instructions are not current support guidance.
