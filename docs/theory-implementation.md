# Theory and implementation

Author: Chang Cheng.

Both proof routes in dissertation Section 7.4 can accommodate CD, feature-anchor
SPA, MU and ANLS-BPP, together with either screening rule, under the corresponding
assumptions. Absolute screening has the simpler connection to the retention
theorem. The preferred reconstruction route does not establish a theoretical
ranking of the solvers or certify consistency or factor recovery for the fitted
CelebA models.

The completed [four-solver comparison](nmf-solver-comparison.md#development-results)
selects SPA under its declared accuracy/time rule with absolute screening fixed;
its status remains `selected_for_audit`. This is empirical support for the package
default. It does not establish separability or supply a theoretical preference for
SPA. Official test evaluation remains outstanding.

The source is Chang Cheng's dissertation,
`Chang-Cheng-dissertation-main-2026-10-01.pdf`, Chapter 7, pages 124–138. The four
results below depend on the assumptions stated in that chapter.

| Result | What it establishes | Application to the package |
| --- | --- | --- |
| 7.4.1, Retention of Marginally Separated Features, pp. 125–127 | A finite-sample retention bound under bounded observations, class-size control, positive population pooled variance, marginal signal and a prescribed cutoff schedule. | Independent of the factorization solver. Absolute screening matches the form directly; percentile retention additionally needs a population rank gap, variance control across all coordinates and sufficient retained count. |
| 7.4.2, Reconstruction Bound for Representation Information Loss, pp. 129–131 | Representation information loss is at most the selected-posterior Lipschitz constant times fresh-observation reconstruction RMSE. | Applies to any fitted basis and coefficient map. Global NMF optimization, SPA anchors, factor identifiability and full rank are not required. |
| 7.4.3, Excess Risk of the Fitted Classification Pipeline, p. 132 | Decomposes excess risk into classifier estimation error plus screening and representation information loss. | Allows the stages to reuse training observations. It does not prove that any of the retained terms vanishes. All four solvers and either screening rule may be used. |
| 7.4.4, Reference-Basis Stability and Classification Information, pp. 133–135 | Bounds NNLS coefficient perturbation and resulting information loss relative to a reference representation. | Requires a full-row-rank, well-conditioned reference basis, aligned component order and scale, small basis error, controlled coefficient objective gap, and reference-posterior approximation/smoothness. These conditions must be established for the fitted representation; selecting a solver does not establish them. |

## Why absolute screening is easier to connect

The threshold theorem uses

\[
\gamma_N=\frac{c_0}{2}\sqrt{\frac{n_0n_1}{N}}N^{-\kappa},
\qquad 0\le\kappa<\tfrac12.
\]

The implemented positive-variance statistic is the same pooled two-sample
absolute t statistic. Numerical rescaling in the calculation leaves that
statistic mathematically unchanged. However, the package's cross-validated cutoff
is a random choice, not the prescribed schedule. Applying the retention result
to a selected cutoff needs additional control: for example, the theorem's
failure bound plus the probability that the selected cutoff exceeds `gamma_N`,
or an appropriately justified restriction on the candidate grid. Cross-validation
alone supplies no such bound. The study's square-root sample-size transfer is a
search heuristic; it is not a proof of that requirement.

Percentile selection needs the additional population ordering/rank-gap condition
and control of the event that cross-validation retains fewer than the required
number of features. Thus it is accommodated by the dissertation, but with more
conditions to establish. Feature retention also requires a target sufficient for
the class posterior before it implies absence of screening information loss.
Marginal screening does not cover an interaction-only signal with no marginal
mean difference.

The implementation excludes globally constant columns and gives infinite scores
to columns that are constant within each class but differ between classes. The
theorem instead assigns zero to an undefined sample statistic and includes such
sample outcomes in its failure event. On the theorem's positive-variance
concentration event these conventions agree on the target's retention; degenerate
population coordinates remain outside its positive-variance assumption.

Original supervised PCA selects features using a cross-validated absolute
univariate-score threshold, as described by
[Bair, Hastie, Paul and Tibshirani (2006), Section 2.1](https://hastie.su.domains/Papers/spca_JASA.pdf).
The package adapts this design to binary pooled-t scores, an inclusive boundary
and a classification criterion. It does not reproduce that paper's quantitative
response score or claim its results automatically transfer.

## Comparing the solver-specific proof burden

CD matches the dissertation's historical joint Frobenius reconstruction
optimization. CD, MU and ANLS-BPP address a nonconvex objective; the preferred
reconstruction bound concerns the actual fitted representation and does not
require a global optimum. SPA can use the same bound without invoking anchor
recovery. All four still need a useful fresh-observation reconstruction bound
and posterior smoothness; a small training objective does not establish either.

SPA has a direct recovery argument in the restricted noiseless,
normalized, feature-separable model discussed on page 136. The package normalizes
training feature columns for anchor selection and retains original-scale anchor
coordinates for classification. Extending recovery claims to noisy data requires
the relevant geometry, rank, conditioning and noise bounds, as well as consistent
component scaling. Recovery alone also does not establish posterior sufficiency.
These conditions have not been established on CelebA. This specialized result
therefore provides no additional theoretical justification for choosing SPA in
the current classification study. Selecting SPA does not remove the main
assumptions of Theorem 7.4.4.

The SPA prediction implementation solves NNLS separately for each observation,
giving a direct pointwise coefficient map. The current sklearn CD transform uses
a shared stopping criterion across a query batch. A deterministic synthetic probe
in the tested environment found a maximum batch-versus-single-row CD coefficient
difference of `2.2843480110168457e-05`, with **no changed labels or probabilities**;
SPA coefficients were identical in that probe. All fits and transforms qualified.
This is a finite-tolerance implementation detail, not a refutation of the four
proofs or evidence of an observed classification error. A strict pointwise CD
theory-to-code statement should account for its numerical/batch context, or
specify pointwise encoding. The package retains the original CD implementation.
The same batch coupling needs attention before using an independent-residual
Hoeffding bound unchanged. SPA's separate per-observation solves avoid this
particular implementation qualification.

## What empirical checks do and do not establish

The theorem's reconstruction quantity uses fresh observations, the actual
prediction transform, the selected basis, and a decoder clipped to the bounded
feature domain. `reconstruction_err_` is an un-clipped **training** diagnostic,
and may use the original fit coordinates; it is not that population quantity.
The independent-observation concentration bound in Equation 7.22 requires data
independent of model selection. The reused development-validation sample does
not meet that requirement after selecting the default. A warning-free stop or an
NNLS return likewise does not alone certify a numerical objective gap. Direct
fresh-observation reconstruction includes the actual solver error and avoids
needing a separate numerical-gap certificate for Version 1, but leaves posterior
smoothness and the other risk terms unresolved.

The separate KNN corollary on pages 137–138 analyzes an explicit sample-splitting
variant: representation fitting and classifier fitting use independent samples,
all classifier observations use the same fixed-basis map, and smoothness/local-mass
conditions hold. Its consistency conclusion also requires an appropriate growing
neighbor count. The current pipeline reuses observations, preserves original
`fit_transform` training coordinates, and tunes finite neighbor counts. Changing
the factorization solver does not make that corollary apply. The general pipeline
decomposition still applies conditionally, with classifier estimation error left
explicit.

The four results concern ordinary 0–1/Bayes risk. The study's balanced accuracy,
average precision and AUC are empirical metrics; those risk bounds are not proofs
about these metrics. A balanced-risk extension must specify a class-reweighted
distribution and its posterior, reconstruction and estimation assumptions.
CelebA intensities satisfy the bounded-data setup. The original Gaussian simulation
clipped only below zero is not bounded above and is outside those assumptions as
written; a separate bounded simulation or concentration argument would be needed.

## Wang's comments and the decision

Wang's R10 requires reconciling percentile practice with threshold theory; absolute
screening is one permitted remedy. R15 strongly recommends an SPA comparison and
also permits comparing both methods or distinguishing the theorem's estimator.
The reconstruction argument applies to the fitted representation without
attributing SPA recovery guarantees to another solver. CD and SPA have both
been compared, together with MU and ANLS-BPP. R7's screened-PCA and
screening-only comparisons remain separate evidence about NMF's contribution.

Absolute screening is therefore preferred for methodological/theoretical alignment
when accuracy is close. All four solvers remain eligible under the common proof
route. SPA with absolute screening was selected by the revised CelebA policy
using the stated accuracy tolerance and measured time, and by the subsequent
four-solver comparison. Its specialized recovery assumptions have not been
validated on CelebA; the general reconstruction-based risk bound also retains
unresolved population reconstruction, smoothness and classifier-error terms.
