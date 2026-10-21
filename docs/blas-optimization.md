# BLAS optimization of feature-anchor SPA

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. October 5, 2026 (Eastern; local run names use UTC October 6).

SPA now uses direct BLAS matrix-vector and rank-one updates during anchor
selection, and prepares the fixed NNLS design matrix once per transform call.
On four training-only CelebA workloads, the measured median time for fitting
plus projecting 500 further training rows fell by **12.6–16.6%**. This is a
numerical implementation change, not a change to the NMF model or classifier.

## What changed

The former anchor update constructed an entire outer-product array and then
subtracted it from the residual. The new update uses SciPy's public `dgemv`
and `dger` wrappers to perform the same operation in place on an owned workspace.
The residual stays in its original C layout; its transpose supplies the
Fortran layout expected by BLAS. This avoids one samples-by-features temporary
at each anchor step without modifying the caller's observations.

Fixed-basis projection still calls public `scipy.optimize.nnls` for each row.
Its design matrix is prepared in C order once per transform. Previously,
passing the transposed basis on every call caused SciPy to repeat the same
layout conversion. Finite checks, the NNLS algorithm, iteration cap, failure
behavior and fitted basis remain unchanged. No private SciPy solver is called.

The package already used the installed BLAS through NumPy/SciPy. CD and MU
delegate their matrix products to scikit-learn's BLAS-backed implementation;
they have not been replaced. No DWD classifier or dissertation theory is
incorporated by this optimization. SciPy is an existing dependency, so the
change adds neither a compiler requirement nor a hard-coded vendor library.

## Measured performance and equivalence

Environment: Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0,
AOCL-BLAS 5.2.0; one native thread. Each method received one warm-up, then seven
paired measurements with randomized method order. Data were selected only from
the official-training cache; no validation/test labels or scores were used.
Screening and downstream classification are excluded from these timings.

| Training rows | Retained columns | Rank | Original fit + projection (s) | Optimized (s) | Time reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1,310 | 2 | 0.05716 | 0.04996 | 12.60% |
| 2,000 | 1,629 | 10 | 0.38326 | 0.32386 | 15.50% |
| 2,000 | 1,629 | 20 | 0.82736 | 0.70002 | 15.39% |
| 2,000 | 4,096 | 20 | 2.03959 | 1.70146 | 16.58% |

The total column is the median of each paired fit-plus-projection sum, rather
than a sum of separate medians. Fit alone took 11.0–17.2% less time and
projection took 5.5–21.4% less time in this final comparison. These are observed
workloads on this machine, not a general speed guarantee or a claim of the same
reduction in full tuning time. The majority of projection work remains NNLS.

All 28 measured before/after pairs, and the warm-ups, produced **bitwise equal
anchor indices, basis matrices and projected coordinates**. KNN/logistic labels
and probabilities also agreed. Broader tests compare against an independent
implementation of the former NumPy algorithm, including separate classifier
fits, C/F/read-only inputs, near ties, zero columns, ill-conditioning, both
training-coordinate conventions, immutability and NNLS iteration failures.
Floating-point equivalence is checked with suitable tolerances outside the
benchmark cases; bitwise equality is not promised for every matrix/backend.

The final source suite passed all 303 tests in both the primary Python 3.12
environment and the pinned Python 3.10 numerical profile with the optional
plotting dependencies installed. In a separately labelled post-hoc check,
seven already-selected low-score simulation configurations produced bitwise
identical outputs before and after BLAS optimization. Thus their poor
predictive results are not explained by this implementation change. The check
used exact frozen parameters and did not replace or retune any study result.

The completed records are `runs/blas-spa-20261006/aocl/benchmark.json` and
the preserved reference source under `runs/blas-spa-20261006/reference/`.
The extra simulation equivalence receipt is
`runs/blas-spa-20261006/strong-case-equivalence/result.json`.
The reference is from commit `832734c6149644492fea98e38fc89e4e0d552d31`.
The reproducible benchmark is benchmark_spa_blas.py (retained locally: `../scripts/benchmark_spa_blas.py`):

```powershell
python scripts/benchmark_spa_blas.py `
  --reference-source runs/blas-spa-20261006/reference/decomposition.py `
  --output runs/blas-spa-20261006/aocl-recheck
```

## Study provenance

This investigation began while the Wang validation was running.
The old source snapshots, partial runs and protocols are preserved. Their
timings are measurements of the old implementation. Source-sensitive study
checkpoints must not be silently resumed or relabeled under the optimized
implementation; subsequent validation needs a new source freeze and run record.
No package-default decision or dissertation proof follows from this speedup.

## Process memory supplement

The separate Windows-native resource check ran each workload/method in a fresh
Python process, serially, twice, with one native thread. All output hashes
matched before/after. Median lifetime peak working sets were:

| Training rows / retained columns / rank | Original peak (MiB) | BLAS peak (MiB) |
| --- | ---: | ---: |
| 1,000 / 1,310 / 2 | 265.38 | 265.36 |
| 2,000 / 1,629 / 10 | 390.25 | 390.42 |
| 2,000 / 1,629 / 20 | 390.39 | 390.21 |
| 2,000 / 4,096 / 20 | 534.96 | 534.56 |

Whole-process peaks were essentially unchanged; no measured peak-memory
reduction is claimed. These OS high-water marks include imports, input arrays,
screening and all fitting/projection work. They are not stage-exclusive
allocation peaks or whole-CV memory. The last CelebA CV jobs were active on the
host; the resource check's cold timings are not used for speed claims.
Records and host context are in `runs/blas-spa-20261006/resources/`;
profile_spa_memory.py (retained locally: `../scripts/profile_spa_memory.py`) reproduces the check.

Sources: [NumPy dot and BLAS](https://numpy.org/doc/stable/reference/generated/numpy.dot.html),
[SciPy dger interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.blas.dger.html).
The installed SciPy NNLS and scikit-learn CD/MU source paths were also inspected
and saved by profile_snmf_blas.py (retained locally: `../scripts/profile_snmf_blas.py`).

## Early release of SPA workspaces (October 9)

`FeatureSPA.fit` now releases the scaled matrix, projection residual and
anchor-selection vectors after the anchor condition checks, before allocating
training coordinates and solving the basis NNLS problems. The scaling factors,
anchor indices and condition diagnostics remain available as fitted attributes.
This changes array lifetimes without changing numerical operations or defaults.

The isolated before/after experiment used three fresh-process repetitions per
variant with one numerical thread. Whole-process median peak memory fell from
2.772 to 2.243 GB on 16,000 by 4,096 synthetic data (rank 4), and from 0.807 to
0.671 GB on 4,000 by 4,096 saved training-image features (rank 20). These are
19.1% and 16.9% reductions on those workloads; fit times changed little.
Anchors, basis matrices, coordinates, transforms and residual diagnostics
matched exactly. The records remain frozen in
`output/remaining-improvement-tests-20261009/memory/`.

The integrated regression checks verify that the temporary arrays are no
longer referenced when basis NNLS begins, including the single-anchor case.
The final implementation uses the anchor count to decide whether the previous
direction variable exists. A `locals()` probe retained that vector through a
frame-locals snapshot on Python 3.12; the new weak-reference check exposed it
without inspecting the frame, and the anchor-count guard fixes it.
Existing numerical tests cover layout, precision, input immutability, downstream
predictions and failed refits. Saved study timings keep their original source
provenance; this implementation change does not supply new accuracy evidence.
