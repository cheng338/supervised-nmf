# supervised-nmf 0.5.0

Version 0.5.0 changes how standalone SPA produces training coordinates, how
new MU classifiers project observations, and how calibration ranks candidates
by default. Review these settings before repeating a 0.4.3 analysis. The
default classifier continues to use feature-anchor SPA and unscaled KNN.

## Changes affecting results

| When you use | Behavior in 0.5.0 | How to choose explicitly |
|---|---|---|
| Standalone `BinaryScreening()` | Keeps features with an absolute pooled-t score of at least 2, matching the classifier default. | Use `screening="percentile"` for percentile selection. |
| Calibration without a scoring argument | Ranks candidates by ordinary accuracy. The choice can change the selected configuration on imbalanced data. | Pass `scoring="balanced_accuracy"` to weight the two classes equally. |
| `FeatureSPA.fit_transform(X)` | Projects training rows against the fitted basis using the same nonnegative least-squares map as `transform`. This applies inside a scikit-learn pipeline. | Read `fit_coordinates_` after fitting if you need the original anchor columns. |
| A new `NMFClassifier(solver="mu")` | The default `mu_projection="auto"` initializes and stops each observation independently. Query coordinates and predictions near a decision boundary can change. | Set `mu_projection="batch"` before fitting to use the shared batch stopping rule. |

Newly fitted SPA classifiers use projected training coordinates for every
accepted `training_coordinates` setting. Reproduce a historical classifier
trained on anchor columns in its original environment.

Evaluation follows native weighted-KNN decisions at rounding ties and honors
custom prediction methods. Class labels ending in NUL now raise an error before
screening, fitting or scoring. [Prediction policies](api.md#prediction-policies-during-evaluation) · [Projection in batches](api.md#prediction-batches)

## Upgrading models and checkpoints

Load saved estimators with the package and dependency versions used to fit them.
To use an older model with 0.5.0, refit it from the original training data.
`FeatureSPA` and `NMFClassifier` now require the stored projection controls
and fitted policies when those fields are used. [Model persistence](api.md#model-persistence)

Start a new checkpoint when upgrading. Calibration checks the implementation
source around each reusable fit and gives one writer ownership of a checkpoint
through refitting. With shared KNN grids, `execution_order="fold"` advances
all eligible representations together within each fold, and
`screening_statistics_recovery="early"` saves eligible screening work before
the fold completes. The defaults remain `"representation"` and `"off"`.
The CelebA parameter study also offers `--final-fit-recovery all` for saving
completed fits in trusted local studies. This adds verification, storage and
loading costs. [Restartable grids](api.md#execution-order-for-restartable-grids) · [CelebA fit recovery](celeba-parameter-study-recovery.md)

## Performance and validation

The comparison with 0.4.3 covered ordinary fits and predictions, extreme
numerical inputs, ties, grouped calibration, and interrupted or completed
checkpoints on Windows AOCL and Ubuntu MKL. Batching the first NNLS pivot
reduced the time for a three-fold Digits SPA pipeline fit by 21% on Windows
and 15% on Ubuntu in a focused comparison with the preceding implementation.
The new standalone SPA mapping adds a fixed-basis projection during pipeline
training.

Repeated source checks have a measurable cost on short searches. On 267 Digits
training images and 90 held-out images, PCA/KNN calibration and evaluation
without a checkpoint took about 155 ms longer on Windows and 25 ms longer on
Ubuntu than in 0.4.3. The selected settings and predictions matched. On a
240-row grouped search, a fresh checkpointed call took about 124 ms longer
on Windows and 20 ms longer on Ubuntu. Completed Linux study resumes can
also spend time starting worker processes. These timings depend on the
workload, runtime and filesystem. [Source checks](development-protocol.md#implementation-source-discovery-for-reusable-searches) · [Study worker startup](celeba-study.md#process-startup-for-current-runs)

The pre-release public suite collected 4,753 cases on each platform. Windows
AOCL passed 4,592 with 161 skips, and Ubuntu MKL passed 4,752 with one skip.
Both passed all 1,229 subtests. Wheel and source-archive installations on
each host passed six solver/classifier checks and reproduced fresh calibration
results on resume. The source archive includes the public workflows, their
tests and replay fixtures. To check an installed copy on synthetic data, run
`python -I scripts/verify_snmf_environment.py --output environment-check.json`
from the source archive with a new output filename. [Full changelog](../CHANGELOG.md)
