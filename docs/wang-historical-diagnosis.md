# Historical standard-NMF diagnostic for Wang's R8

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. Executed October 5, 2026 (Eastern; the run directory uses
the UTC date `20261006`).

The original-size, reported-rank reconstruction reproduced the **below-chance
failure closely: 32.75% accuracy**, compared with the dissertation's rounded
32.42%. Increasing only the NMF iteration cap from 400 to 2,000, at the same
`tol=1e-4`, produced **92.92%**, with the factorization stopping at 1,226
iterations and no factorization or projection convergence warnings. This is a
controlled numerical diagnostic on one synthetic dataset. It supports an
insufficient iteration cap as a cause of the failure in the reconstructed
procedure. It does not verify the original environment or establish the
dissertation's proposed clipping/topological-inversion explanation.

## Historical configuration and provenance

The preserved [original script](../provenance/original/sim.py) generates
6,000 observations, 25,000 columns and 250 signal columns, using NumPy's legacy
seed 42. Signal means are 1 and 3 with Gaussian noise standard deviation 0.8;
the remaining columns have Gaussian mean 3 and standard deviation 9. All
entries are clipped below at zero and converted to float32. The original
stratified 80/20 split uses seed 42. This reconstruction has 2,400 observations
per class in training and 600 per class in the held-out sample.

Dissertation Table 7.1 and Section 7.2.4 explicitly identify the selected rank
as 2 and the supervised percentile as 0.1%. The standard-NMF arm used the
full 25,000 columns, CD, NNDSVDa, `random_state=42`, `max_iter=400`, and the
default tolerance of 1e-4, followed by Euclidean five-neighbor classification
on unnormalized coefficients. The standard-NMF rank was inherited from the
supervised search. This investigation conditions on that reported rank; it
does not rerun the historical supervised CV search or present a newly tuned
standard-NMF benchmark.

The diagnostic script (retained locally: `../scripts/diagnose_historical_nmf.py`) extracts only
the original generator's syntax tree and executes it unchanged, avoiding the
original top-level warning suppression and plotting/tuning entry point. Array
hashes, source hashes, split IDs, labels, current versions, thread-pool settings,
predictions and factors are preserved in
`runs/wang-historical-20261006/`. The initial source snapshot is
`diagnostic-source.py` within that directory.

The current environment is `py12_df_aocl`: NumPy 2.5.2, SciPy 1.18.0 and
scikit-learn 1.9.0. One process and one numerical thread were used. The original
script allowed all KNN threads and did not pin its historical libraries or
BLAS. Historical raw predictions, arrays, CV receipts and exact versions are
unavailable. Consequently **32.42% is not an exact historical reproduction**;
the new 32.75% corresponds to four more correct classifications than a 32.42%
rounded result on 1,200 observations, without identifying which historical
predictions differed.

## Controlled results

All initial arms were declared in `protocol.json` before their outcomes were
computed. They use the same generated observations, split, label convention,
NNDSVDa seed, KNN settings and tolerance. Accuracy and balanced accuracy agree
because the held-out sample is balanced. A recorded numerical warning is
preserved as a diagnostic observation; that row is not a qualified model for
the new validation study.

| Change from the historical configuration | Accuracy | Convergence status |
| --- | ---: | --- |
| None: rank 2, float32, cap 400; alternating-fit training coefficients | 32.75% | Fit and both projections reach cap 400 |
| Recompute training coefficients against the same final basis, cap 400 | 38.58% | Same unqualified factorization/projections |
| Normalize every training/test coefficient row to unit L2 | 33.67% | Same unqualified factorization/projections |
| Unit-L2 basis rows with compensating training/test coefficient-column scaling | 33.17% | Same unqualified factorization/projections |
| Classify using coefficient-row L2 norm alone | 34.42% | Same unqualified factorization/projections |
| Increase cap only to 2,000 | 92.92% | Fit stops at 1,226; no recorded fit/projection warnings |
| Cap 2,000 and recompute training coefficients against its final basis | 94.58% | No recorded fit/projection warnings |
| Convert the same float32-generated data to float64 before fitting, cap 400 | 32.58% | Fit and projections reach cap 400 |
| Float64 cap 400 with recomputed training coefficients | 38.58% | Same unqualified factorization/projections |
| Increase rank only to 10, float32, cap 400 | 68.33% | Fit stops at 168; no recorded fit/projection warnings |
| Rank 10 with recomputed training coefficients | 69.17% | No recorded fit/projection warnings |

All 11 predeclared classifier arms completed. Rank 10 was a prespecified
diagnostic change, not an independently tuned winner. Its convergence within
400 iterations does not qualify the historical rank-2 fit at the same cap.
No new experimental arms were launched following the request to investigate
direct BLAS performance.

The 400-iteration confusion matrix, with rows true classes 0/1 and columns
predicted classes 0/1, is `[[277, 323], [484, 116]]`. The cap-2,000 matrix is
`[[553, 47], [38, 562]]`. Positive-class probabilities use column 1 of the
verified class ordering `[0, 1]`. The corresponding original-coordinate ROC
AUCs, AP, per-class recalls and prediction arrays are in the individual JSON
and NPZ records. Label inversion was not applied or selected. Complementing
binary predictions would arithmetically change 32.75% to 67.25%; that identity
does not establish a label-encoding error or a valid correction.

## What the checks establish

- **Labels and split:** training/held-out IDs are unique, disjoint and complete;
  features and labels use exactly the same IDs. All features are finite and
  nonnegative. There is no observed feature/label alignment error in this
  reconstruction.
- **Held-out transformation:** the original source already uses `transform`
  against the fitted training basis. Basis checksums remain identical before
  and after both held-out and training transformations. Refitting the basis on
  held-out observations is not the explanation supported by this source.
- **Training coordinates:** at cap 400, recomputing training coefficients changes
  their Frobenius norm by 25.66% relative to the returned alternating-fit
  coefficients. At cap 2,000 the relative change is 4.47%. The cap-400
  recomputation itself remains unconverged. These values describe a numerical
  difference; they do not identify a universal preferred coordinate convention.
- **Normalization and dtype:** neither of the specified normalization changes
  nor conversion to float64 removes the cap-400 failure. The float64 arm uses
  the same already-rounded generated observations; it changes computational
  dtype rather than the data-generating process.
- **Numerical qualification:** extending the cap, with the objective, tolerance,
  data, rank and classifier fixed, removes the below-chance result in this
  reconstruction. The cap applies to factorization and fixed-basis projection;
  this intervention alone does not separate their contributions. A solver's
  stopping criterion does not certify a global NMF optimum.
- **Reconstruction versus prediction:** training relative reconstruction error
  changes only from 0.762662 to 0.762630 between the two caps, despite the large
  classification difference. Reconstruction error alone is therefore an
  insufficient diagnostic of predictive behavior in this example.

## Scope and remaining work

R8 is **partially addressed with a direct numerical explanation in the
reconstructed procedure**. The historical below-chance phenomenon is recovered
closely, its rank and coefficient convention are documented, warning suppression
is exposed, and a prespecified numerical intervention removes the failure.
The unverified topological-inversion account should not be presented as an
established causal finding. Exact numerical reproduction of the old 32.42%
value and attribution to the unknown historical environment remain unresolved.

This exercise is one dataset and one split, does not independently tune standard
NMF, and does not establish general superiority or complete Wang's replication,
harder-simulation, ablation or theory requirements. All image datasets and their
official test partitions are outside this diagnostic.

A disclosed exact rank-two NNLS follow-up is implemented behind
`--exact-fixed-basis`. Its active-set solver passed 75 synthetic comparisons
against `scipy.optimize.nnls`, including zero and boundary solutions, with
maximum per-row relative projected-gradient residual below 1.3e-16. **The
historical-data follow-up has not been executed** following the user's steering
to investigate direct BLAS performance. It must retain its separate protocol
and results if subsequently run; it is not part of the results above.

The historical local workflow used `scripts/diagnose_historical_nmf.py` with
the output argument below. That study script and its original machine-specific
invocation are retained locally, outside the public distribution. This records
the workflow configuration; it is not an executable example for this checkout.

```text
scripts/diagnose_historical_nmf.py --output runs/NEW-HISTORICAL-DIAGNOSTIC
```

Existing output directories are protected from replacement. Measured stage
times are retained for bookkeeping; concurrent work and hardware/library
differences preclude interpreting them as an isolated speed comparison.

## Verification and output references

The execution manifest passed an independent read-only receipt audit: source
and protocol hashes, split membership, class alignment, every prediction hash,
and recomputed accuracy, balanced accuracy, AUC, AP and confusion counts agreed
for all 11 classifier arms. `receipt-audit.json` binds the execution manifest
and records the separate 75-case exact-NNLS implementation check. The small
100-by-100 fixture established script execution before the full diagnostic;
its predictive results are not scientific evidence.

The local result directory contains `protocol.json`, `data.json`, `split.npz`,
`threadpools.json`, `results.json`, individual factorization/metric records,
predictions and coefficients, `manifest.json`, `receipt-audit.json`, and PNG/PDF
versions of the coordinate diagnostic. The PNG was visually checked for
readability. These ignored run outputs remain outside package distributions.

Figure: Historical rank-two coordinate diagnostic (retained locally: `../runs/wang-historical-20261006/historical-coordinate-diagnostic.png`)

The displayed factorization and projections use the historical cap 400 and
remain numerically unqualified. The plot illustrates training/held-out
coordinate differences; it is not evidence of a proved geometric inversion.
