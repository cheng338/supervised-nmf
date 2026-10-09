# supervised-nmf 0.2.0

Formal release, October 9, 2026. Author: Chang Cheng.

SPA now uses guarded batched NNLS for suitable fixed-basis query batches,
with public SciPy NNLS fallback. Optional reusable process workers and
separate native-thread controls allow callers to budget projection resources.
The default remains one projection worker. Use `projection_solver="scipy"`
to request the original row-wise solver; older serialized models retain it.
Coefficient roundoff can differ under batching, while numerical qualification
and fallback protect the original NNLS problem.

`NMFClassifier` can store a training-selected binary `decision_threshold`
and `positive_label`. An explicit threshold uses strict probability greater
than cutoff; equality is negative. With no threshold, the underlying
classifier retains its original decision rule. Shared KNN calibration honors
fixed thresholds, and `predict_with_probabilities` uses one projection.

SPA releases anchor-selection workspaces before basis NNLS. Earlier measured
workloads showed 17–19% lower whole-process peak memory with identical outputs
and little change in fit time. This is not an accuracy improvement or a
whole-study performance guarantee. KNN normalization and classifier defaults
remain unchanged; completed study results retain their original provenance.

## Compatibility and validation

- Python 3.10 or later; joblib>=1.3 is now a direct dependency.
- Full source suite: 370 tests passed on Windows Python 3.12 and Python 3.10.
- Guarded fallback, old-model serialization, actual worker execution,
  threshold policies, numerical equivalence and array lifetimes are covered.
- Earlier Linux Python 3.12 checks covered projection and thresholds; the
  final workspace-lifetime change has been verified on Windows.
- The wheel is pure Python (`py3-none-any`) and supports Python 3.12; native
  BLAS libraries come from the installed NumPy/SciPy distributions.

See [API details](api.md#spa-projection-execution) and the
[implementation record](implementation-improvements.md).

## Install

Download the wheel from the GitHub release and run:

```shell
python3.12 -m pip install --upgrade supervised_nmf-0.2.0-py3-none-any.whl
```

The source archive supports building from source. `SHA256SUMS.txt` records
checksums for the wheel and source archive. Previous release assets and tags
remain available. This release is published on GitHub; no PyPI publication is
implied.
