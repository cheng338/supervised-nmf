# supervised-nmf 0.3.0

October 20, 2026 (America/Indianapolis). Author: Chang Cheng.

Version 0.3.0 changes the default SPA classifier training coordinates, adds
optional SD-scaled KNN and a linear SVM head, and incorporates numerical
safeguards and guarded execution improvements. SPA with absolute screening
and unscaled KNN remain the solver, screening and classifier defaults.

## SPA training coordinates: migration

`NMFClassifier` now defaults to `training_coordinates="auto"`. For SPA, this
resolves to `"transform"`: after fitting the feature mask and basis, the
classifier is trained on fixed-basis NNLS coordinates. Training and prediction
therefore use the same fitted representation map. The additional training
projection is included in fit time and reported separately in `fit_times_`.
CD and MU continue to use their factorization's `fit_transform` coordinates.

This changes the learning procedure when an SPA model is retrained with the
default settings. Predictions and accuracy can change; improvement is not
guaranteed. To retain the coordinate convention used by version 0.2.0, set:

```python
from supervised_nmf import NMFClassifier

model = NMFClassifier(training_coordinates="fit_transform")
```

That setting retains the former convention, not every historical source or
runtime detail. Record the solver, complete parameters, source version and
numerical environment when reproducing an earlier result. The standalone
`FeatureSPA` transformer and its behavior in generic sklearn pipelines are
unchanged by this classifier-default change. Existing study results and
released 0.2.0 artifacts are preserved rather than relabeled as 0.3.0 results.
See the [training-coordinate API](api.md#training-coordinates).

## Optional classifiers

- `knn_normalization="sd"` divides coordinates by training-fitted standard
  deviations without centering. The same scaler is used for prediction. It
  changes KNN distances but leaves screening, factorization, `transform` and
  `training_coordinates_` unchanged. Unscaled KNN remains the default; SD
  scaling can amplify weak or nuisance coordinates.
- `classifier="linear_svm"` fits a primal L2-regularized squared-hinge linear
  SVM on unscaled coordinates. Its independent controls are `svm_C`,
  `svm_max_iter`, `svm_tol` and `svm_class_weight`. Real decision margins point
  toward `classes_[1]`; exact zero selects `classes_[0]`. There is no implicit
  scaling, probability calibration or SVM `predict_proba`. A non-`None`
  `decision_threshold` is rejected. Evaluation uses margins for AUROC and
  average precision, and `predict_with_decision_function` avoids projecting
  the same query twice.
- Logistic regression remains available with its existing training-fitted
  coordinate standardization. Classifier diagnostics and timing remain
  separate from NMF diagnostics.

Tune the chosen head within training folds together with screening and rank.
None of these options establishes a universally best classifier. See the
[classifier API](api.md#classifier).

## Numerical and execution changes

Screening statistics now subtract each feature's training minimum before
range scaling. Translation and positive scaling preserve the pooled t/F
statistic in exact arithmetic; this order protects represented small spreads
around large offsets. The public transform still returns the original
selected feature values. Explicit failures remain for unsupported statistics
or nonconstant features whose within-class variance underflows.

Fixed-basis NNLS checks magnitude before either projection solver. Supported
extreme inputs use exactly reversible common power-of-two scaling of the
design and each independent right-hand side. This preserves exact NNLS
minimizers and does not improve conditioning. Unsupported exponent spans fail
explicitly; they are not treated as proof that the mathematical problem has
no solution. Factor-recovery reporting also uses stable row normalization,
with zero rows receiving no matching credit.

Compatible Euclidean KNN heads share neighbor queries during grid evaluation.
Unsupported configurations and uncertain distance or vote boundaries fall
back to the public selected-K estimators. Reports charge shared setup/query
work once and retain per-head costs; changed source/schema identities prevent
silently reusing incompatible checkpoints.

Report-only reconstruction norms have a bounded block path for contiguous
float64 inputs with 8,192–32,768 rows, 1,024–4,096 features and rank 1–32, when
all detected BLAS pools are one-thread MKL. Other shapes, backends and thread
settings retain the full-product expression. This affects diagnostic cost,
not model fitting, stopping or prediction. See the [implementation record](implementation-improvements.md)
and [calibration API](api.md).

## Validation and limitations

The October 20 integration was validated on Robust under a budget of 20 total
compute threads. The recorded integration includes 260 regression checks,
original-trigger numerical replays, classifier/serialization checks and
matched KNN and residual comparisons. A subsequent comparison against the
earliest recoverable complete October 3 source used the previously explored
CelebA development data and explicit difficult-case fixtures. It is not an
exact reconstruction of the missing earlier build or a direct measurement
of every difference from the released 0.2.0 wheel.

All 42 isolated, fixed-recipe CelebA source pairs had identical predicted
labels when their coordinate policy was held fixed. All 135 selected outcomes
in the separate 5,002-image, three-label, three-seed comparison qualified.
Those repeated outer splits overlap, and the official test partition was not
used. The study compared a finite grid with one head-only refinement; every
selected sNMF rank was at the upper explored boundary. It does not establish
globally optimal rank, general defaults across populations, universal accuracy
gains or the dissertation's conditional theoretical assumptions. Difficult
numerical cases that failed their strict certificate remain recorded as
unqualified; there is no blanket guarantee for all finite inputs.

Timing and memory depend on the operation and workload. Matched CelebA
prediction calls were faster in the isolated repeated comparison, while some
fit observations and the separate single-call screening diagnostics were
slower. The latter are not repeated timing estimates. Whole-process memory
can increase substantially despite faster prediction:

| Controlled public call | Input | Baseline peak | 0.3.0-source peak | Increase |
| --- | --- | ---: | ---: | ---: |
| Feature screening fit | 16,384 × 4,096, float64 | 1,924.9 MiB | 2,437.0 MiB | 26.6% |
| Fixed-basis SPA projection | 16,384 × 4,096, float64 | 709.6 MiB | 1,222.3 MiB | 72.3% |

These fresh-process, one-thread probes compare the earliest recovered source
with the validated 0.3.0 implementation. Peaks include inputs, imports and
allocator behavior through the measured call; they are not permanent model
storage or full-study memory. Float32 screening peaks were essentially
unchanged in the corresponding probe. An extreme-scale preparation fixture
also retained 512 MiB of scaled design copies across 128 distinct groups;
that preparation-only measurement is not a CelebA prevalence estimate.

Numerical safeguards have correctness reasons independent of small performance
differences. The increased peaks remain disclosed limitations; no slower
memory-saving replacement was adopted. Detailed receipts and figures are
retained locally under `output/latest-vs-first-20261020/` and
`output/implementation-20261020/`, outside the public distribution. The
[validation record](validation.md) distinguishes historical checks and their
scope. Release-build results must be read from the corresponding GitHub
Actions run and release assets; these research receipts do not certify a
future build or every supported dependency combination.

## Compatibility and installation

Python **3.11 or later** is required. Python 3.12 in `py12_df_aocl` remains the
primary development environment. Python 3.10 support ended on October 20;
its previous release assets and test results remain historical records.
The package wheel is pure Python; the installed NumPy/SciPy distributions
supply the native numerical libraries. Dense nonnegative binary-classification
inputs remain the supported scope; sparse inputs and multiclass classification
are outside this release. See [runtime compatibility](compatibility.md).

Download the 0.3.0 wheel from [GitHub releases](https://github.com/cheng338/supervised-nmf/releases),
then install it in the intended environment:

```shell
python -m pip install --upgrade supervised_nmf-0.3.0-py3-none-any.whl
```

For a source installation:

```shell
git clone --branch v0.3.0 https://github.com/cheng338/supervised-nmf.git
cd supervised-nmf
python -m pip install .
```

To build locally from that source checkout, install the build tools and run:

```shell
python -m pip install "setuptools>=77" wheel
python scripts/build_release.py
```

This writes the wheel and source archive to `dist/`; it does not publish them.
The `[examples]` extra adds Pillow; `[study]` also adds Matplotlib. GitHub
distribution does not imply a PyPI publication.

## Authorship and distribution

Chang Cheng developed the methodology and original scripts. The package
implements Chapter 7 of *Ensemble Kernel Distance-Weighted Discrimination and
Supervised Nonnegative Matrix Factorization for Classification* (2026),
Purdue University, Department of Statistics. The dissertation's other part
is not implemented or combined with supervised NMF here.

The repository is public. Chang Cheng's original work remains **All Rights
Reserved**, with no distribution license yet selected; public access is not
an open-source license grant. The separately identified third-party material
retains its BSD-3-Clause notices and terms. See [LICENSE](../LICENSE),
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES), [CITATION.cff](../CITATION.cff)
and [source provenance](../provenance/README.md).
