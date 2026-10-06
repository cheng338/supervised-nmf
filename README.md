# Overview

`supervised-nmf` implements supervised nonnegative matrix factorization for
binary classification with a scikit-learn-style interface. It screens features
using the training labels, fits a nonnegative representation, and classifies
new observations with k-nearest neighbors (KNN) or logistic regression.

The package provides:

- **Supervised NMF classification** (`NMFClassifier`), with feature-anchor SPA,
  multiplicative updates (MU), and coordinate descent (CD).
- **Feature screening** (`BinaryScreening`), using an absolute pooled-t cutoff
  or a retained percentile.
- **Feature-anchor factorization** (`FeatureSPA`), available as a separate
  transformer with fixed-basis projection for new observations.
- **Classification comparisons** with screened PCA, screening without
  factorization, raw features, and unscreened CD-NMF, using either classifier.

Chang Cheng developed the method and original research scripts. This package
implements Chapter 7 of:

> Cheng, Chang (2026). *Overcoming the Curses of Dimensionality and
> Kernelization: Scalable DWD and Supervised NMF*. Doctoral dissertation
> manuscript, Purdue University, Department of Statistics.

See [CITATION.cff](CITATION.cff) for citation details and
[source provenance](provenance/README.md) for the original scripts. This is a
development release, version 0.1.0a1. A distribution license has not yet been
selected.

# Installation

Python 3.10 or later is required. Dependencies are NumPy, SciPy, scikit-learn
1.6 or later (below 2.0), and threadpoolctl. With access to the repository:

```shell
git clone https://github.com/cheng338/supervised-nmf.git
cd supervised-nmf
python -m pip install .
```

Image-loading examples require Pillow. Install the optional example or study
dependencies as needed:

```shell
python -m pip install ".[examples]"
python -m pip install ".[study]"
```

The study extra also includes Matplotlib. Use `python -m pip install -e .`
for an editable development installation. Tested environments and remaining
compatibility checks are listed in [validation](docs/validation.md).
The supplied Python 3.10.12 numerical profile is recorded in
[runtime compatibility](docs/compatibility.md), including exact dependency pins
and instructions for preserving an existing environment. All 281 source tests
passed with that exact numerical stack on Windows 11; Ubuntu execution has not
been tested.

# Basic use

This example generates a small dataset from the original simulation model and
fits both SPA and MU. It requires no download.

```python
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from supervised_nmf import NMFClassifier
from supervised_nmf.datasets import make_original_simulation

data = make_original_simulation(n_samples=200, n_features=100, signal_ratio=0.1)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25,
    stratify=data.target, random_state=42,
)

models = {}
for solver in ("spa", "mu"):
    model = NMFClassifier(
        solver=solver, screening="absolute", score_threshold=2.0,
        n_components=2, n_neighbors=5, random_state=42,
    ).fit(X_train, y_train)
    models[solver] = model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    print(solver, balanced_accuracy_score(y_test, predictions))
```

Rows are observations and columns are features. Inputs must be dense, finite
and nonnegative. The classifier preserves the two input class labels;
probability columns follow `model.classes_`. Screening and factorization are
fitted on training data. Prediction reuses the fitted feature mask and basis.

KNN remains the default. To use L2-regularized logistic regression on the NMF
coordinates:

```python
logistic_model = NMFClassifier(
    classifier="logistic", logistic_C=1.0, random_state=42,
).fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
```

This optional extension standardizes the fitted coordinates inside the
classifier, using training means and scales. It does not change screening,
factorization, or the coordinates returned by `transform`.

# Solvers and defaults

The default is **SPA with absolute screening**. Use `solver="mu"` for
multiplicative updates or `solver="cd"` for the dissertation's original
coordinate-descent implementation.

| Solver | Factorization |
| --- | --- |
| `"spa"` | Selects feature anchors, then solves nonnegative least-squares problems to fit the basis and encode new observations. |
| `"mu"` | Uses scikit-learn's multiplicative updates with Frobenius loss and NNDSVDa initialization. |
| `"cd"` | Uses scikit-learn's coordinate descent with Frobenius loss and NNDSVDa initialization. |

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_components` | `2` | Number of NMF components. |
| `solver` | `"spa"` | Factorization method. |
| `screening` | `"absolute"` | Also accepts `"percentile"` or `"none"`. |
| `score_threshold` | `2.0` | Inclusive cutoff on the absolute pooled-t score. |
| `percentile` | `10.0` | Retained percentage when percentile screening is selected. |
| `classifier` | `"knn"` | Final classifier; also accepts `"logistic"`. |
| `n_neighbors` | `5` | Number of neighbors; used only by KNN. |
| `weights` | `"uniform"` | KNN-only weighting; `"distance"` is also available. |
| `logistic_C` | `1.0` | Inverse L2 regularization strength for logistic regression. |
| `logistic_max_iter` | `1000` | Logistic-regression iteration limit, independent of the NMF cap. |
| `logistic_tol` | `1e-4` | Logistic-regression stopping tolerance. |
| `logistic_class_weight` | `None` | Also accepts `"balanced"` from training labels, or a dictionary of class weights. |
| `max_iter` | `20000` | CD/MU iteration limit or the limit for each SPA NNLS solve. |
| `tol` | `1e-4` | CD/MU stopping tolerance; SPA's NNLS routine does not use this parameter. |
| `random_state` | `None` | Set an integer for reproducible CD/MU initialization. |
| `training_coordinates` | `"fit_transform"` | Uses the factorization's returned training coordinates; `"transform"` re-encodes them against the fitted basis. |

Rank, screening cutoff and the selected classifier's predictive parameters
require tuning for the data. CD and MU use different stopping criteria, so
equal tolerances do not imply equal
numerical accuracy. Features and factors are not automatically rescaled for
NMF or KNN. Logistic regression uses a separate fitted coordinate scaler.

The default follows the recorded accuracy-time comparison on CelebA and
Fashion-MNIST. Its evidence remains under final audit and does not establish
a universal best solver. SPA's factor-recovery results require additional
separability and noise assumptions; classification accuracy does not establish
those assumptions. See [solver comparisons](docs/nmf-solver-comparison.md),
[methodology](docs/methodology.md), and [theoretical scope](docs/theory-implementation.md).

# Parameter tuning

Tune the complete classifier inside training folds so that each fold fits its
own feature screening and factorization. The package's calibration helper
records failed and unconverged fits, parameters, selected features and timings.

```python
from supervised_nmf.evaluation import calibration_search

search = calibration_search(
    NMFClassifier(solver="mu", random_state=42),
    {"score_threshold": [1.0, 2.0, 3.0],
     "n_components": [2, 4], "n_neighbors": [3, 5, 9]},
    X_train, y_train, cv=5, seed=42, threads=1,
)
selected_model = search.estimator
print(search.report["selected_parameters"])
print(search.report["is_fallback"])
```

These are starting ranges. Check boundary winners and nearby parameter values
using training data, then reserve independent observations for final evaluation.
A candidate with a failed or unconverged required fold cannot win. If all
candidates fail, the helper returns a flagged majority-class fallback; inspect
`is_fallback` before treating the result as a fitted NMF classifier.

For larger grids, `calibration_knn_grid` reuses each training-fold representation
across neighbor counts. Both helpers support resumable records with matching
data, settings, source and runtime identities. See the [API guide](docs/api.md).

For logistic regression, use `calibration_search` with
`NMFClassifier(classifier="logistic")` and tune `logistic_C`, for example over
`[0.01, 0.1, 1.0, 10.0, 100.0]`, alongside rank and screening. Do not include
KNN parameters in that grid. The scaler is fitted within each training fold.
`calibration_knn_grid` is restricted to KNN.

# Examples and diagnostics

Run the example scripts from the source checkout or source archive:

```shell
python examples/quickstart.py
python examples/compare_simulation.py --quick --output runs
python examples/compare_classifiers.py --quick --output runs
python examples/celeba_classification.py --help
python examples/celeba_parameter_study.py --help
```

The simulation and CelebA examples compare independently tuned SPA, MU, CD,
screened PCA and screening-only KNN. CelebA requires local images, attributes
and official partition files. Images use the original 64 by 64 grayscale
preprocessing; the examples use the training and validation partitions.
Datasets are not included in the package.

`compare_classifiers.py` compares KNN and logistic regression across matched
NMF, PCA, screening-only and raw-feature paths. Completed CelebA and
Fashion-MNIST solver studies used KNN; logistic regression has not been selected
as a new default or established as superior by those results.

After fitting, inspect `selected_features_`, `selected_components_`,
`training_coordinates_`, `n_iter_`, `converged_`, and `convergence_warnings_`.
`components_` maps the basis back to the original feature columns.
`classifier_n_iter_`, `classifier_converged_`, and
`classifier_convergence_warnings_` distinguish classifier diagnostics from
factorization diagnostics. `fit_times_` separates screening, factorization,
training projection and classifier fitting, including logistic scaling.
Evaluation reports record prediction time; search reports record
tuning and final-fit time separately.

Numerical convergence does not certify a global optimum or classification
consistency. Balanced accuracy, class counts, precision and recall accompany
ordinary accuracy in the comparison reports. See the
[CelebA study](docs/celeba-study.md) and [validation record](docs/validation.md)
for completed work and remaining checks. The current API supports binary
classification with dense inputs; sparse inputs and multiclass classification
are outside this release.

# Tests and local builds

From a source checkout with the dependencies installed:

```shell
python scripts/check.py
python scripts/build_release.py
python -I scripts/check_wheel.py
```

The build produces a local wheel and source archive. The wheel check verifies
imports, authorship metadata, SPA/MU/CD fitting and prediction, and archive
contents. It does not publish a release.
