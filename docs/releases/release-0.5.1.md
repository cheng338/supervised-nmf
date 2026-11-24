# supervised-nmf 0.5.1

This patch updates the source test suite, study documentation and package
description. It uses the same numerical implementation, public API, defaults
and runtime requirements as 0.5.0.

## Running the Linux tests

`python scripts/check.py` now sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to `1` before importing pytest on Linux. This lets a
fresh worker thread exercise completed-checkpoint reuse. Set the same variables
before starting Python to run those tests directly or from an IDE.

The one-thread setting applies to the test launcher. When resuming a CelebA
study, the scheduler still checks the native pools visible to a fresh worker
thread. If it sees a wider pool, a process worker validates the saved folds.
A new test starts native pools with two threads and checks the scheduler's
decision against the limits the worker actually sees. Other tests verify that
checkpoint validation reaches saved scores that are nonfinite or too small
for in-process reuse.

## Calibration with a pipeline cache

A regression test compares `screening_statistics_recovery="off"` and
`"early"` with separate, initially empty joblib caches. It checks that both
settings refit the screening step on cache misses, reuse it on cache hits and
issue matching warnings under the same cache conditions. It also compares every held-out
prediction across both settings and both cache states.

## Documentation and installation

The study guides identify when historical results and timing measurements
were collected, explain what within-study resampling measures, and identify
which study tools ship in the public source. The package description now
includes wheel installation instructions, a short example and links to the
tagged documentation. The source archive includes the full README and API
guide for offline use.
