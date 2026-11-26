# supervised-nmf 0.6.1

Version 0.6.1 adds a reader for completed calibration reports, corrects a numerical
edge case in batched nonnegative least squares, and updates the package's
dependency minimums.

## Upgrading from 0.6.0

- Ordinary installations require Python 3.11 or later, NumPy 1.26.3 or later,
  SciPy 1.15.0 or later, scikit-learn 1.9.0 or later, threadpoolctl 3.7.0 or
  later, and Joblib 1.6.0 or later. SciPy 1.16.0 and 1.16.1 are excluded
  because repeated calls to the NNLS solver can leak memory.
- If you need to reproduce grouped cross-validation folds generated with
  scikit-learn 1.6.x or 1.7.x, supply the saved fold indices and keep the
  original row order and group labels. Upgrading can change the folds even
  with the same seed.
- Finish in-progress studies in their original environment and source checkout.
  To repeat a study after upgrading, use a fresh checkpoint path or comparison
  directory. If your own worker threads set MKL thread limits, set the limit
  inside each worker that needs it.
- On CPython 3.11 and 3.12 with NumPy 1.26.3, the test extra requires pandas
  2.1.2 or later and the study extra requires Matplotlib 3.7.3 or later.
  With NumPy 2 on those interpreters, use pandas 2.2.2 or later and
  Matplotlib 3.8.4 or later. The test extra requires pytest 9.0.3 or later
  and excludes 9.1.0. The examples and study extras require Pillow 13.0.0
  or later. For the minimums on other Python versions, see the
  [compatibility guide](https://github.com/cheng338/supervised-nmf/blob/v0.6.1/docs/compatibility.md).

## Reading a completed calibration report

Use `load_calibration_report` to inspect a completed schema-2 checkpoint
written by `calibration_search(..., refit=False)`. The reader returns the
saved report dictionary after checking that the fold results, candidate
scores, selection, and refinement are internally consistent. Reproducing
the scores requires the original data and computational setup.
A completed search marked ineligible for
restart can still be read if its evidence is sufficient. An older report can
raise `ValueError` when a decisive claim cannot be verified.

```python
from supervised_nmf.evaluation import load_calibration_report

report = load_calibration_report("calibration.json")
```

Supply `expected_sha256=` with a trusted digest to verify the original file
bytes. For the checkpoint format and restart requirements, see the
[API guide](https://github.com/cheng338/supervised-nmf/blob/v0.6.1/docs/api.md).

## Numerical correction

For strided or unaligned target arrays, batched NNLS now recomputes the scale
for its optimality check when taking absolute values changes the array layout
or alignment. This preserves the reference decision at floating-point boundaries.
