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
[source provenance](provenance/README.md) for the original scripts. The first
formal release is 0.1.0. It includes the algorithms and BLAS improvements from
0.1.0a2; this version change does not change their numerical behavior.
A distribution license has not yet been selected.

# Installation

Python 3.10 or later is required. Dependencies are NumPy, SciPy, scikit-learn
1.6 or later (below 2.0), threadpoolctl, and joblib 1.3 or later. With access to the repository:

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

SPA uses direct BLAS anchor updates and releases anchor-selection workspaces
before fitting its basis. The current source uses guarded batched NNLS for
suitable query batches, with public SciPy NNLS as its numerical fallback.
Use `projection_solver="scipy"` for the original row-wise projection.
Optional `projection_workers=8` enables reusable processes for large batches;
the default is one worker. See [projection controls](docs/api.md#spa-projection-execution)
and the [implementation checks](docs/implementation-improvements.md).

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
| `projection_solver` | `"auto"` | SPA query projection: guarded batching; also `"scipy"` or `"batched"`. |
| `projection_workers` | `1` | Maximum SPA projection processes; independent of outer CV workers. |
| `projection_batch_size` | `256` | Maximum query rows per NNLS block. |
| `projection_native_threads` | `1` | Numerical threads per projection process. |
| `decision_threshold` | `None` | Optional frozen probability cutoff; `None` preserves the classifier's decision rule. |
| `positive_label` | `None` | Class to threshold; defaults to `classes_[1]`. |

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

# Tuning and reporting update

Version 0.1.0a2 adds subject-aware training folds, bounded rank exploration,
reused fold-local screening statistics, and evaluation summaries with class
support and numerical-stage diagnostics. Solver, screening, and rank defaults
are unchanged in that release. The current unreleased source additionally
provides guarded batched projection, saved decision thresholds and lower SPA
workspace memory. Existing serialized models retain their SciPy projection path. See [tuning improvements](docs/tuning-improvements.md) and the
[API guide](docs/api.md).

Run `python examples/grouped_tuning.py --output runs/grouped-example` for a
small subject-separated example. Search budgets and failed upper-rank
candidates remain explicit; a completed grid does not establish a global
optimum or complete statistical validation.

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
