# supervised-nmf

`supervised-nmf` is a Python package for binary classification of dense,
nonnegative data. It uses the training labels to select features, represents
each observation by a small set of nonnegative coordinates, and fits a
classifier on those coordinates. New observations use the feature mask and
basis learned during training.

The main estimator, `NMFClassifier`, follows scikit-learn’s `fit`, `transform`
and `predict` interface. Feature screening (`BinaryScreening`) and feature-anchor
factorization (`FeatureSPA`) are also available as standalone transformers.

## Installation

Use Python 3.11 or later. Download the wheel from the
[GitHub release](https://github.com/cheng338/supervised-nmf/releases/tag/v0.5.1),
then run this command in the download directory.

```sh
python -m pip install ./supervised_nmf-0.5.1-py3-none-any.whl
```

For the examples, tests and study scripts, use the matching source archive or
clone the release.

```sh
git clone --branch v0.5.1 https://github.com/cheng338/supervised-nmf.git
cd supervised-nmf
python -m pip install .
```

Use `python -m pip install ".[examples]"` from the source directory for image
examples. The `study` extra also installs Matplotlib.

## Quick start

Each row is an observation and each column a feature. Supply a finite,
nonnegative matrix and two classes of training labels. This example generates
its data locally, holds out a stratified test set, and predicts five labels.

```python
from sklearn.model_selection import train_test_split

from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import make_original_simulation

data = make_original_simulation(n_samples=200, n_features=100, signal_ratio=0.1)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
)

model = NMFClassifier(random_state=42).fit(X_train, y_train)
print(model.predict(X_test[:5]))
```

To inspect the fit, use `model.selected_features_` for the retained feature
indices, `model.components_` for the basis in the original feature space, and
`model.transform(X_test)` for the test observations’ coordinates.
`model.predict_proba(X_test)` returns probabilities in `model.classes_` order
for KNN and logistic regression. For the linear SVM, use
`model.decision_function(X_test)` for decision margins.
See the [classifier reference](docs/api.md#classifier) for fitted attributes
and [fit diagnostics](docs/api.md#fit-diagnostics) for convergence and warnings.

## Training-only tuning

The defaults use an absolute pooled-t cutoff of 2.0, two components and five
neighbors. Tune all three on the training set for a new dataset. Each
cross-validation fold must learn its own feature mask, basis and classifier.
If several rows belong to one subject, keep that subject together when
reserving the test set and pass the training subject IDs through `groups` for
cross-validation.

Using the training split above, this example searches a small grid. By default,
`calibration_search` ranks candidates by mean fold accuracy. Set
`scoring="balanced_accuracy"` to give each class equal weight in the score.

```python
from supervised_nmf.evaluation import calibration_search

result = calibration_search(
    NMFClassifier(random_state=42),
    {
        "score_threshold": [0.0, 2.0],
        "n_components": [2, 4],
        "n_neighbors": [3, 5, 9],
    },
    X_train, y_train, cv=5, seed=42, refit=False,
)
params = result.report["selected_parameters"]
if params is None:
    raise RuntimeError("No candidate qualified in every training fold.")
print(params)
```

With `refit=False`, the result contains the CV report only. Review warnings,
failures and winners at the edge of the grid before settling on the settings.
Then fit the chosen model on all training observations, check its convergence,
and evaluate it on the reserved test set. Compare with screening alone, PCA
or raw features using the same splits to see what factorization contributes.
The [calibration guide](docs/api.md#calibration-and-reporting) covers automatic
refitting, fallback models and checkpoint reuse. The
[grouped example](examples/grouped_tuning.py) works through subject-separated
tuning and final evaluation.

## Solvers and classifiers

| Solver | How it represents an observation |
| --- | --- |
| `"spa"` (default) | Selects anchor features and uses nonnegative least squares to encode training and new observations against the fitted basis. |
| `"mu"` | Fits NMF by multiplicative updates. By default, it projects each new observation with its own initialization and stopping decision. |
| `"cd"` | Fits NMF by coordinate descent. Like MU, it uses the coordinates from the joint fit for training by default. |

The default classifier is KNN with uniform votes and unscaled coordinates.
Choose `classifier="logistic"` for logistic regression with training-fitted
standardization, or `classifier="linear_svm"` for a linear SVM on unscaled
coordinates. Tune the neighbor count or regularization strength alongside the
rank and screening settings. The [classifier options](docs/api.md#classifier)
explain percentile screening, KNN scaling, decision thresholds and other
controls. See [training coordinates](docs/api.md#training-coordinates) and
[prediction batches](docs/api.md#prediction-batches) when choosing a projection
policy for repeated or batched predictions.

## Examples and further reading

From a source checkout or unpacked source archive, run a small classification
example or a grouped tuning study.

```sh
python examples/quickstart.py
python examples/grouped_tuning.py --output runs/grouped-example
```

- [API reference](docs/api.md) — estimator settings, fitted models, calibration and evaluation.
- [Methodology](docs/methodology.md) and [theory](docs/theory-implementation.md) — how screening and factorization work, and the assumptions behind factor recovery.
- [Validation studies](docs/validation.md) — comparisons with simpler baselines, including the solver settings and data splits used in each study. Accuracy from an earlier study applies to its recorded configuration.
- [Runtime compatibility](docs/compatibility.md) and [model persistence](docs/api.md#model-persistence) — numerical backends and loading fitted models.
- [Study recovery](docs/celeba-parameter-study-recovery.md) — checkpoint costs and how to resume a CelebA parameter study. Use only trusted local model files for restoration.

## Tests and local builds

From the source directory, install the study and test extras and run the
project’s test launcher.

```sh
python -m pip install -e ".[study,test]"
python scripts/check.py
```

On Linux, the launcher sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to `1` before pytest starts. Set the same variables
before starting Python when running checkpoint-reuse tests directly or in an
IDE. To check the installed package on synthetic data, run
`python -I scripts/verify_snmf_environment.py --output environment-check.json`
from the source directory, using a new output filename.

To build and inspect both distributions, start with a clean checkout of the
public release and install its build tools in the active environment.

```sh
python -m pip install "setuptools>=77" wheel
python scripts/build_release.py
python -I scripts/check_wheel.py
```

## Citation and license

Chang Cheng developed the method and original research scripts for Chapter 7
of *Ensemble Kernel Distance-Weighted Discrimination and Supervised Nonnegative
Matrix Factorization for Classification* (Purdue University, 2026). The
[citation file](CITATION.cff) gives the software citation, and
[source provenance](provenance/README.md) identifies the original scripts.

Copyright © 2026 Chang Cheng. All Rights Reserved. A distribution license for
the original work has yet to be selected. See the [license notice](LICENSE)
and [third-party notices](THIRD_PARTY_NOTICES) for the applicable terms.

See the [changelog](CHANGELOG.md) for release history.
