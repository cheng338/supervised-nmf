# supervised-nmf 0.6.0

Version 0.6.0 gives all models in a development comparison the same
cross-validation folds, makes calibration fallbacks visible, and checks the
feature names supplied to fitted estimators.

## Upgrading from 0.5.1

- Use Python 3.11 or later. The scikit-learn requirement changes from
  `>=1.6,<2` to `>=1.6.1`. Version 1.6.1 fixed cross-validation with
  metadata routing enabled when callers omit `params` or pass `None`.
- Finish an in-progress study with the source and dependencies that created
  it. To repeat it with 0.6.0, use a fresh checkpoint path or comparison
  directory. Load older fitted models in their original environment, or
  refit from the training data after upgrading.

## Calibration and comparison

- `run_comparison` saves one CV partition and reuses it for every model,
  including models appended on restart. With `seed=None` or a mutable
  `RandomState`, later models can receive different folds than in 0.5.1.
  Use an integer split seed to reproduce a partition in a fresh run.
- If every candidate fails CV qualification or every qualifying candidate
  fails the full-data refit, standalone `calibration_search(..., refit=True)`
  returns a training-majority classifier and emits `CalibrationFallbackWarning`.
  Pass the calibration result to
  `evaluate_classifier` to retain the fallback in the report. In a
  comparison, check each model's `is_fallback` field.
- Shared KNN calibration preserves the fitted neighbor index and the
  selected classifier's prediction behavior. This can change fold scores
  and the selected neighbor count for affected tree searches and custom
  pipelines. Bounded rank searches now compare integer-like ranks by their
  numerical values and check which pipeline step each grid entry assigns
  the rank to.
- A checkpointed search records the first candidate and fold where it sees
  the estimator or grid change during fitting. The current call can finish.
  To fit again from a marked search, repeat it in a fresh location.
- The comparison examples complete CV before timing final fits and
  predictions with one worker and one native numerical thread. They report
  CV and final-fit time separately.

## Input columns and MU projection

- `NMFClassifier`, `BinaryScreening` and `FeatureSPA` record unique string
  column names at fit time. When supplying named tables again, keep the
  original names and order. Calibration and comparison checkpoints record
  the same schema. `BinaryScreening.get_feature_names_out()` returns
  the selected names, and standalone `FeatureSPA` supports named component
  columns with pandas output.
- Fixed-basis multiplicative updates (MU) recover coordinates when a positive
  update denominator rounds to zero. This can change predictions on
  extreme-range inputs. Wide float64 rowwise projection can also resolve a
  stopping decision from error bounds and uses a direct residual calculation
  when those bounds are inconclusive.

## Finding the documentation

The usage and API guides are in `docs/`. Study protocols and results moved
to `research/`, engineering records to `research/engineering/`, and release
notes to `docs/releases/`. See the [documentation index](https://github.com/cheng338/supervised-nmf/blob/v0.6.0/docs/README.md)
and the [changelog](https://github.com/cheng338/supervised-nmf/blob/v0.6.0/CHANGELOG.md).
