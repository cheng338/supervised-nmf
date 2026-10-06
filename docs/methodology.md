# Methodology and changes from the research scripts

Author: Chang Cheng. This package is based on his dissertation and original
research scripts, preserved in `provenance/original/`.

## Retained behavior

Rows are observations; columns are nonnegative features. Binary labels supervise
univariate pooled-variance screening. Retained columns keep their original
nonnegative values. The historical estimator uses Frobenius NMF, coordinate
descent, NNDSVDa initialization, and Euclidean KNN on the returned coefficients.
The historical path leaves coefficients and basis vectors unnormalized. The fitted
feature mask and basis are reused for new observations. The historical simulation
generator preserves the original random draw order with a local RandomState.

## Explicit changes and reasons

| Change | Reason and effect |
| --- | --- |
| Reusable sklearn-style estimators | Extract the script pipeline without coupling training to plotting, console output, or dataset paths. |
| SPA with absolute-score screening as the default | The revised CelebA selection policy prioritizes absolute screening among accuracy-qualified methods, then measured cost. The completed [four-solver comparison](nmf-solver-comparison.md#development-results) also selects SPA under its declared rule; its status remains `selected_for_audit`. The original time-first decision and revised policy are recorded in the [results](celeba-default-results.md). CD and historical percentile screening remain available. |
| MU solver alongside SPA and CD | `NMFClassifier(solver="mu")` uses the same unregularized Frobenius objective and NNDSVDa initialization as the MU comparison. Screening, KNN, and the SPA default are unchanged. ANLS-BPP remains experimental. Adding MU to the API does not provide additional benchmark evidence or establish theoretical superiority. |
| Numerical cap increased to 20,000, tolerance unchanged at 1e-4 | Six formerly limited CD folds converged above 10,000 iterations. The classifier and comparison convenience defaults now match the qualified study allowance. This changes permitted work, not equations or stopping tolerance; previously truncated outputs need not be identical. |
| Absolute pooled-t cutoff alongside percentile selection | Supports threshold-based screening with an inclusive boundary. Historical percentile screening remains a separate option. |
| Stable float64 screening statistics | Per-feature rescaling prevents overflow/underflow without changing the pooled-t statistic mathematically. The returned feature values and float32 CD input retain their original scale and dtype. Finite-precision differences from f_classif remain possible. |
| Explicit globally constant and classwise-constant feature handling | Globally constant features provide no class information; classwise constants at different values have infinite separation scores. No variance floor silently changes the statistic. Near-zero denominators remain visible through score diagnostics. |
| Empty selections and infeasible ranks are reported as failed fits | Prevent invalid NNDSVDa configurations or silent feature additions. A CV candidate with a failed required fold cannot be ranked using only its successes. |
| Consistent NMF settings within tuning and final fit | Removes the original simulation's 300-versus-400 iteration mismatch. Iteration caps do not certify convergence. |
| Warnings and numerical diagnostics retained | Makes failed and nonconverged attempts auditable instead of suppressing them. |
| Independent baseline tuning and PCA/screening-only comparisons | Prevents unscreened NMF from inheriting a rank optimized for the screened pipeline; tests the contribution of factorization. |
| Broader metrics, timing, and run records | Separates classification under imbalance, model-selection expense, fit expense, and numerical reliability. |
| Official CelebA partition loader | Replaces selecting the first sorted files then randomly splitting. This is a new evaluation protocol, not historical reproduction. Grayscale resizing and intensity scale are retained. |
| Optional fixed-basis training-coordinate recomputation | A diagnostic alternative to the original returned training coefficients. It is explicit and cannot be relabeled as a confirmed correction before controlled diagnosis. |
| Controlled simulation extensions | Independent within-block noise factors vary signal/background correlations; an explicit all-features mode tests cross-block correlation. The exact original generator remains separate. XOR is an additional marginal-screening failure case. |
| Explicit per-method parameter grids | Tune NMF/PCA rank, the applicable screening parameter, and KNN neighbors on the same training folds. Record grid revisions and numerical qualification before final evaluation. |

## SPA procedure and scope

`FeatureSPA` uses features as candidate anchor columns, preserving the orientation
in the revised dissertation. Training column L1 normalization is used for anchor
selection. Successive projections select linearly independent residual columns.
The original-scale selected columns supply training coefficients; nonnegative
least squares estimates their basis across retained features. New observations
are encoded by nonnegative least squares against that fixed basis.

Selection normalization does not normalize observations or CD coefficients.
Comparisons report each solver's normalization convention.
Selected-column conditioning and rank deficiency are checked. Exact separable
fixtures can test mathematical orientation and reconstruction; arbitrary noisy
data do not thereby inherit a separable-recovery guarantee.

Reference: Gillis and Vavasis, *Fast and Robust Recursive Algorithms for Separable
Nonnegative Matrix Factorization*, https://arxiv.org/abs/1208.1237.

## MU procedure and scope

The public `solver="mu"` path uses scikit-learn multiplicative updates with
Frobenius loss, NNDSVDa initialization, and no factor regularization. NNDSVDa
fills zeros in the initial factors with the input mean; it avoids using the
zero-filled NNDSVD initialization with multiplicative updates. The existing
`max_iter`, `tol`, and `random_state` parameters apply.

By default, KNN fits the training coefficients returned by `fit_transform`.
The explicit `training_coordinates="transform"` option instead recomputes
training coordinates against the fitted basis. New observations always use the
fitted feature mask and fixed basis. Both options retain the coefficient scales
and the screening-to-factorization-to-KNN pipeline.

MU uses a different convergence criterion from CD. Objective decrease and a
successful numerical fit do not establish global factor recovery or statistical
consistency. The reconstruction-based classification argument remains conditional
on its stated assumptions for MU, SPA, and CD alike.

## Limits of claims

Prediction accuracy, complete support inclusion, reconstruction error, factor
identification and runtime are different outcomes. Retaining 25 of 250 true
features cannot establish full-support recovery. CD and SPA cannot share recovery
claims solely because both return nonnegative factors. The reconstruction-based
theory remains conditional on its stated assumptions and out-of-sample errors.
Neither choosing a cutoff by CV nor switching to a linear classifier proves
statistical consistency.

