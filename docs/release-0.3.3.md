# supervised-nmf 0.3.3

supervised-nmf 0.3.3 improves fixed-basis MU projection, adds guarded SPA basis
fitting and optional QR projection, and clarifies evaluation reports. The
default remains SPA with absolute screening and unscaled KNN. CD, MU, logistic
regression and linear SVM remain available.

## Numerical changes

- MU returns exact zero coordinates for an entirely zero selected query batch
  after validating the full input. Tiny positive inputs retain a separate
  recovery path and are not treated as zero by a tolerance.
- Eligible float32 MU projections retry in float64 when tiny fitted basis tails
  or arithmetic failures make ordinary calculation unreliable. Logarithmic
  arithmetic handles tiny positive inputs and unresolved range failures.
  Recovery preserves the fitted basis, MU recurrence and returned dtype;
  joint factorization is unchanged. Rounding and stopping iterations can differ,
  and recovery can still cost more than ordinary arithmetic.
- SPA basis fitting reuses the guarded batched NNLS kernel, with the original
  public SciPy column loop as a numerical-range fallback. The fixed-anchor
  objective and anchor selection are unchanged. Existing conditioning and
  optimality guards remain in place.
- `projection_solver="qr"` provides guarded QR reduction for fixed-basis SPA
  projection. It preserves the NNLS objective in exact arithmetic and falls
  back when admission or numerical checks fail. Its cost depends on the
  workload; `"auto"` remains the default.
- Reconstruction reporting extends the existing bounded single-thread MKL
  path to a specifically qualified Windows AOCL build. The dispatch changes
  diagnostics only, with no effect on optimization or stopping rules.

See the [API reference](api.md) for admission conditions, numerical limits,
iteration-cap behavior and fitted-state semantics. These implementation
changes do not establish new statistical assumptions or a universal speed,
accuracy or memory improvement.

## Reports and tests

Evaluation summaries include per-class precision, recall, evaluation counts
and available training counts. Missing metrics and observations remain
explicit. Timing reports distinguish tuning, final fitting, prediction and
the fitted estimator's component stages; overlapping totals must not be added.

Calibration and the Fashion-MNIST study runner initialize BLAS before recording
their runtime fingerprints. This prevents a lazily loaded OpenMP library from
appearing as a runtime change during fitting or between parent and worker
processes. Source and runtime checks for resume remain strict.

The source test launcher now uses pytest 9 or later. Install the test and study
dependencies with `python -m pip install -e ".[study,test]"`, then run
`python scripts/check.py`. Regression coverage includes MU range recovery,
SPA basis fitting, QR projection and evaluation summaries.

Python 3.11 or later is required. Chang Cheng's original work remains All Rights
Reserved, and the separate third-party notices are preserved. Packaging retains
the directory patterns introduced in 0.3.2.

```shell
python -m pip install --upgrade supervised_nmf-0.3.3-py3-none-any.whl
```

## Verification

Local release checks passed on Windows with Python 3.12 in four numerical
environments: MKL 2026.1 and 2026.2, and AOCL 5.2 and 5.3. Each passed
667 tests and 443 subtests, plus a separate fresh-process worker/resume
check. The wheel and source archive were built locally and checked in all
four runtimes.

MU validation replayed 93 numerical edge fixtures per environment and compared
16 workloads against 0.3.2 in each environment: 512 runs, including 128 warmups
and 384 measured runs. All 256 saved label comparisons matched; basis and
training coordinates also matched. Query coordinates and scores can differ
through finite-precision recovery. The large development-time MU slowdown
is resolved, although recovery retains workload-dependent overhead relative
to 0.3.2. Timing ranges and comparison details accompany the release. These
checks reuse exposed development cohorts and are not a new held-out study.
