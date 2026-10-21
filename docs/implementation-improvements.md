# Projection and SPA memory improvements

Author: Chang Cheng.

Version 0.2.0 adds guarded batched NNLS projection, optional reusable
process workers, separate projection/native-thread limits, and a fitted
binary decision threshold. The default projection setting is `auto` with one
worker; an explicit `scipy` setting and legacy serialized models retain the
original solver. See the [API](api.md#spa-projection-execution) for guards,
fallback, serialization and threshold semantics. Joblib is now a direct
runtime dependency.

SPA anchor selection and basis NNLS retain their numerical operations.
After the anchor condition checks, `FeatureSPA.fit` releases the scaled
matrix, residual, directions and loop-array references before allocating
training coordinates. An anchor-count guard avoids a `locals()` snapshot
retaining the previous direction on Python 3.12.

## Validation

The full source suite in a clean checkout passed all 370 tests on both
Windows Python 3.12 and Python 3.10. This includes 155 package regression
tests covering guarded fallback, coefficient and
prediction agreement, errors/refits, input immutability, legacy serialization,
process reuse, thresholds and workspace lifetime. The two lifetime tests
cover 16 rank/layout/precision/normalization combinations, and fail against
the old implementation as expected.

The source wheel also passed an isolated installation and import check. It
contains both new projection modules, their third-party notice and the direct
joblib dependency. Batched projection and stored-threshold smoke checks passed.

Before the final workspace change, the projection and threshold source also
passed 153 regression tests on Linux Python 3.12/OpenBLAS. Saved-model replay
tested serial SciPy and auto with 1, 8 and 40 workers on six SPA models.
All 24 configurations retained the original decisions, with coefficients and
probabilities inside the declared tolerances. Separate untimed worker traces
verified the requested process counts. Forty workers were not consistently
faster than eight; pool startup and repeated-call timings must be distinguished.

An isolated serial memory experiment used three fresh processes per variant
on each of two workloads, with one numerical thread. Whole-process median
peak memory fell from 2.772 to 2.243 GB on 16000 by 4096 synthetic data at rank
4, and from 0.807 to 0.671 GB on 4000 by 4096 training-image features at rank
20. Numerical outputs matched exactly and fit times changed little. The
17–19% memory reductions apply to those workloads; they are not whole-CV
speed or memory guarantees. No classification-accuracy gain is attributed
to these implementation changes.

The completed study evidence keeps its original source fingerprints. KNN
normalization and the training-coordinate convention retain their defaults;
the exploratory normalization results did not support a universal change.

Run the source tests with `python -m unittest discover -s tests -q` after
installing the optional study dependencies. These changes are included in
formal release 0.2.0; the historical 0.1.0 release tag remains unchanged.

## October 20 source changes

The retrospective review led to a further approved integration, separate from
the released implementation above. It repairs offset-sensitive screening,
extreme-range fixed-basis NNLS and factor-recovery arithmetic; adds guarded
KNN query reuse, optional fitted SD KNN scaling and an optional linear SVM head;
and evaluates a bounded report-only residual optimization.

The original and revised source snapshots, failure records and validation
receipts are preserved in `output/implementation-20261020/`. The current
integration is validated on the validation host under a 20-total-thread budget.
Its results do not replace historical accuracy, timing or compatibility records.
Python 3.10 results above remain historical; current source requires Python 3.11
or later. See [the protocol](development-protocol.md#approved-october-20-implementation)
for the mathematical scope and adoption requirements.
