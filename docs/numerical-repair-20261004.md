# ANLS-BPP numerical repair, October 4, 2026

Author: Chang Cheng. This record describes an implementation repair to the
experimental ANLS-BPP comparison.

## Observed interruption

The original four-solver run is preserved at `runs/four-solvers-20261003`.
After a pause and resume, a worker terminated abruptly and
Python raised `BrokenProcessPool` at 09:46:59 UTC on October 4. A diagnostic
retry with Python's fault handler enabled captured a Windows access violation
inside `scipy.linalg.lstsq`, at its `_batched_linalg._lstsq` call. The caller was
the original-design least-squares fallback in ANLS-BPP's `_passive_solve`.

The stack locates the failing call; it does not establish which vendor's code
caused it or prove memory exhaustion. This native process failure is separate
from a recorded, ordinary NNLS/KKT nonconvergence warning. All benchmark workers
were stopped before diagnosis or source changes.

The diagnostic crash tests produced Windows application-error dialogs. Further
checks and benchmark children suppress those dialogs using per-process
`SetErrorMode`; Python's fault handler and nonzero failure exits remain enabled.
This does not change the Windows registry or system-wide error-reporting policy.

Evidence is retained in the original run's `fashion-cv-attempt2.log`,
`fashion-cv-attempt3.log`, and `native-fault-recovery-20261004.json`.
There are 2,646 completed calibration reports: 2,033 from unchanged methods and
613 ANLS-BPP reports. A completed report can contain rejected candidates;
these counts do not imply all fits converged. Partial completed fold receipts
are also retained.

## Numerical change

The amended implementation calls SciPy's low-level LAPACK **DGELSD** wrapper
directly for ill-conditioned passive sets, avoiding the high-level batched
`lstsq` dispatcher. It retains the original-design minimum-norm SVD
least-squares problem and the explicit float64 machine-epsilon rank cutoff.
The wrapper checks array dimensions, pads the right-hand side as LAPACK
requires, queries workspaces, and checks the returned status.

SciPy's [1.18 release notes](https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html)
describe moving `lstsq` batching into C. The direct-wrapper approach follows the
dispatch structure in SciPy's [1.17 implementation](https://raw.githubusercontent.com/scipy/scipy/v1.17.0/scipy/linalg/_basic.py).
This evidence supports the dispatch workaround; it is not an upstream diagnosis
of the particular native fault.

The NMF objective, BPP pivoting, NNDSVDa initialization, regularization policy,
outer and inner tolerances, iteration caps, and classifier factor geometry
remain unchanged. This does not relax the numerical qualification rule or
turn earlier nonconverged candidates into successful fits. Finite-precision
and timing differences are possible and will be measured under the amended
implementation.

## Evidence lineage and comparison

The successor run is `runs/four-solvers-20261004-repair`, with its own source
freeze and the [Fashion-MNIST v3 protocol](../examples/fashion_mnist_solver_protocol_v3.json)
and [CelebA v2 protocol](../examples/celeba_four_solver_protocol_v2.json).
All data, folds, predictive grids, decision gates, and sensitivity analyses
are inherited without changes. Partial training-CV evidence existed before
this amendment. No Fashion-MNIST development-validation outcomes or new matched
CelebA validation outcomes had been evaluated. Historical CelebA results were
already known, as previously disclosed.

Unchanged CD, SPA, MU, PCA, screening-only and raw-KNN chunk directories may
be copied byte-for-byte only after verifying their identities and integrity
against the new configuration. Their original source/runtime identities and
timings are retained; imported receipts remain distinguishable from new fits.
An import manifest records the original run and hashes. New aggregate reports
are generated against the successor configuration.

**Every ANLS-BPP chunk is rerun.** The 613 original ANLS-BPP reports and partial
folds remain historical diagnostic evidence and do not enter the repaired
four-solver decision. The inherited CV costs and previous elapsed/interrupted
execution time are identified separately from new-run wall time. Serial
deployment timing is performed after both datasets' CV completes.

## Qualification status

The original call was reproduced on a captured training-only design with shape
6,400 by 50 and three right-hand sides. Its high-level GELSD replay exited with
Windows access-violation code 3221225477. The direct DGELSD route solved those
same matrices three times; the integrated repaired wrapper reproduced those
solutions exactly and agreed with a separately tested GELSS route. Its relative
normal-equation residual was approximately 7.81e-14. This is a least-squares
diagnostic, not a classification result or an NMF convergence certificate.

The 17 experimental-NMF regression tests pass, covering random and degenerate
NNLS cases, NNLS/KKT qualification, minimum-norm tall/wide systems, immutable
inputs, workspace/solver errors and prevention of high-level dispatcher reuse.
The captured fixture and diagnostic logs remain under
`tmp/bpp-native-diagnostic`; its archive SHA-256 is
`e6ccc619e9628fdf8cd9ceaec0244de2ec7d2f45a1a1aeb99d10e78b9ecd332e`.

An integration check on the same 6,400-row training fold completed two diagnostic
outer sweeps in 24.297 seconds, including 5,035 direct DGELSD passive-group
solves and the formerly failing shape. There were zero failed inner solves;
the maximum recorded inner relative KKT residual was approximately 8.09e-14.
The objective decreased on both sweeps. The deliberately shortened two-sweep
cap produced its expected outer-cap warning, so this check is evidence of safe
execution through the repaired path, not convergence at the production cap.
No held-out projection, prediction or classification score was computed.

The 12 receipt-recovery and launch-policy tests pass, as do the 17 reporter
tests. A read-only check of one actual completed chunk from each of CD, SPA,
MU, PCA, screening-only and raw KNN confirmed compatibility with the amended
configuration without changing receipt bytes.

The regression suite at the time of repair passed **266 tests in 61.417 seconds**,
including the existing portable-workflow tests. Its log is
`tmp/four-solver-repair-full-check.log`. All seven package source files are
unchanged from the original run's freeze. Eligible receipts were subsequently
verified and copied before production resumed. The repaired comparison completed
on October 5, 2026; see [validation status](validation.md). No public solver default
or dissertation proof changed as part of this repair.
