# Theory and implementation

Author: Chang Cheng.

The theorem numbers and printed page references below follow Chapter 7,
Section 7.4 of Chang Cheng's *Ensemble Kernel Distance-Weighted Discrimination
and Supervised Nonnegative Matrix Factorization for Classification* (Purdue
University, 2026). The November 3 review used the October 16 final PDF and
matching chapter source. Their hashes and the review record are retained
in the original study workspace. [Citation metadata](../CITATION.cff)

**The final theory explicitly permits training-sample reuse.** The same
training observations may select features, fit a representation and train KNN,
with settings selected using training labels and cross-validation. Theorem
7.4.4 accounts for that selection under its stated conditions; it does not
require an extra split between representation fitting and classifier fitting.
Validation/test observations remain separate from the corresponding fit.

This is also the ordinary statistical-pipeline workflow: the transformer and
predictor are fitted on the same training samples. See the
[scikit-learn pipeline guide](https://scikit-learn.org/stable/modules/compose#pipeline-chaining-estimators)
and [leakage guidance](https://scikit-learn.org/stable/common_pitfalls#data-leakage).
Within each CV fold, all learned steps are fitted on that fold's training rows.
Shared training observations across stages do not mean carrying fitted objects
between folds or fitting on validation observations.

Public NMF solvers are SPA, CD and MU. ANLS-BPP appears in experimental study
comparisons, not as a public `NMFClassifier` solver. The general risk
decomposition and coefficient-stability statements do not select a solver.
The exactly separable consistency result specifies SPA/NNLS and a more
restricted procedure.

The four-solver development comparison selected SPA under its accuracy/time rule
with absolute screening fixed. Its recorded status is `selected_for_audit`.
This empirical selection leaves separability unverified. [Four-solver results](nmf-solver-comparison.md#development-results)

The official-test phase of the original CelebA default-selection protocol remains
unfinished. A separate October 12 study evaluated projected SPA coordinates
on an official test partition. [Original protocol status](celeba-default-results.md)
and [October 12 results](validation.md#october-12-official-partition-evaluation)

| Result | What it establishes | Application to the package |
| --- | --- | --- |
| Theorem 7.4.1, Retention of Marginally Separated Features, p. 133 | A finite-sample target-retention bound under bounded observations, admissible class sizes, positive pooled population variance, marginal signal and a specified threshold. | Independent of the factorization solver. The subsequent discussion explicitly covers selected cutoffs and feature counts, with additional failure terms. |
| Theorem 7.4.2, Excess Risk of the Fitted Classification Pipeline, p. 138 | Bounds expected ordinary excess risk by classifier probability-estimation error plus screening and representation information loss. | Explicitly allows the same training data at all stages. The remaining error terms need control; the decomposition alone does not make them vanish. |
| Theorem 7.4.3, Reference-Basis Stability and Classification Information, pp. 140–141 | Bounds pointwise coefficient error and representation information loss relative to an aligned reference basis. | Requires reference conditioning, small basis perturbation, controlled numerical objective gaps and informative/smooth reference posteriors. Global joint NMF optimization is not required. |
| Theorem 7.4.4, KNN Risk with Training-Sample Reuse, p. 147 | Bounds expected ordinary excess risk by screening loss, local neighbor-posterior bias and a penalty for label-dependent configuration selection. | Covers uniform KNN with the common fitted map and fixed 0.5 rule under the conditions below, including training-based screening, CV and final refitting. |
| Corollary 7.4.1, Reference Bound for the Reused-Sample Classifier, p. 149 | Bounds local bias using reference-posterior and coefficient errors at both selected training neighbors and the fresh query, plus scaled neighbor distances. | Connects the common NNLS map to classification; fresh-query reconstruction measurements alone do not control these terms. |
| Corollary 7.4.2, Consistency in an Exactly Separable Model, p. 152 | Establishes Bayes consistency and a rate for a specified screening–SPA/NNLS–KNN procedure in a separable population model. | A genuine conditional consistency result. Finite-tolerance CV defaults do not automatically instantiate its population model or full sequence of procedures. |

The former October 1 draft's standalone reconstruction-information theorem,
theorem numbering and sample-split KNN description are not the final results.
Final Equation 7.22 is a pointwise coefficient-error bound.

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

The final discussion also treats a retained count \(q\). If the \(q\)-th and
\((q+1)\)-th largest eligible scores are strictly separated, top-\(q\) selection
is exactly an absolute screen with cutoff equal to the \(q\)-th score.
Boundary ties cannot be split by a cutoff on the unmodified scores alone; the
count rule uses its index convention, and this equivalence route includes a
boundary-tie failure term. A separate population rank-gap route gives the
bound \(8d\exp\{-CN^{1-2\kappa}\}+P(\widehat q_N<q_0)\), with positive variance
control across all coordinates. These are explicit final-dissertation routes,
not an assertion that every percentile grid satisfies the required conditions.
Unconditional screening bounds also account for inadmissible class counts.
Feature retention requires a target sufficient for the class posterior before
it implies absence of screening information loss.
Marginal screening does not cover an interaction-only signal with no marginal
mean difference.

The implementation and final dissertation agree: globally constant columns are
excluded, and columns constant within each class but different between classes
receive infinite absolute scores. The retention theorem's covered target
features have positive population pooled variance; its concentration event
gives positive empirical denominators. Degenerate population coordinates are
handled by the methodology's separate rule, outside that variance assumption.

Original supervised PCA selects features using a cross-validated absolute
univariate-score threshold, as described by
[Bair, Hastie, Paul and Tibshirani (2006), Section 2.1](https://hastie.su.domains/Papers/spca_JASA.pdf).
The package adapts this design to binary pooled-t scores, an inclusive boundary
and a classification criterion. It does not reproduce that paper's quantitative
response score or claim its results automatically transfer.

## What training-sample reuse means

Within a training sample, fit screening and the basis, transform those same
observations and fit KNN on their coordinates and labels. Within CV, do this
anew using only the training portion of each fold. After selection, the final
fit may use all available training observations. Reuse concerns observations
across these modeling stages, not validation leakage or cached fitted state
across different folds.

Final Section 7.4.5, pp. 145–150, proves a result for this workflow. Its
configuration collection \(\Theta_N\) is finite and deterministic and covers
every possible final feature set, rank, scaling rule, neighbor count and other
settings. Label-selected random seeds are included as settings. For each
**fixed candidate**, the representation, common training/query map, scaling
and distance-tie rule use selected feature values and independent algorithmic
randomness only. They do not use labels or unselected feature values.

The selection of a candidate may nevertheless depend arbitrarily on all
training features and labels. This includes supervised screening, a shared
cutoff, CV and final refitting. CV need not find an optimal candidate. The
proof obtains a simultaneous bound over possible candidates before applying
that arbitrary selection rule. It does not assert independence after
conditioning on the selected mask.

For \(0<\alpha<1\) and positive deterministic proof weights \(\omega_\theta\) summing to one,
the selection term is

\[
u_N(\theta,\alpha)
=\sqrt{\frac{\log(2/(\alpha\omega_\theta))}{2k_\theta}}.
\]

These are weights over configurations in the proof, not distance weights in
KNN. The classifier uses equal neighbor votes and
\(f_D(x)=1\{q_{\widehat\theta}(x)>1/2\}\). Writing \(\mathcal B_D(x)\) for the
absolute difference between the mean selected-neighbor posterior and the query's
selected-feature posterior, Theorem 7.4.4 gives

\[
E[R(f_D)]-R_X^*
\le E[\Delta_{\mathrm{screen}}(D)]+\alpha
+E\!\left[\min\{1,2(\mathcal B_D(X)+u_N(\widehat\theta,\alpha))\}\right].
\]

This concerns expected ordinary risk over iid training samples and a fresh
test observation. It is not a confidence statement for every fitted model.
Uniform configuration weights give
\(u_N=\sqrt{\log(2|\Theta_N|/\alpha)/(2k_\theta)}\). All possible selected
masks count, not merely the candidates appearing in one realized CV log.
Continuous basis entries need not be discretized, and intermediate fold fits
need not be counted as final configurations.

Approximate coefficient solvers are allowed when their computation respects
the fixed-candidate dependence restrictions. Numerical failures require a
specified completion rule or a fitting-failure probability term; a flagged
majority fallback is not silently a qualified sNMF fit. Group-disjoint face
splits prevent subject overlap but do not themselves establish iid image
sampling.

## SPA anchor selection and screening false positives

Supervised screening uses training labels; the subsequent feature-anchor SPA
step uses only the retained matrix. Its geometric anchors need not be the
features most informative about the label.

For a nonzero nonnegative training column \(x\) with \(n\) observations, define
\(\bar{x}=n^{-1}\sum_i x_i>0\),
\(s_0^2=n^{-1}\sum_i(x_i-\bar{x})^2\), and
\(\mathrm{CV}(x)=s_0/\bar{x}\). These are empirical moments with variance divisor
\(n\). With `normalize=True`, the squared norm used for the first SPA selection is

\[
\left\|\frac{x}{\|x\|_1}\right\|_2^2
=\frac{\sum_i x_i^2}{(n\bar{x})^2}
=\frac{1+\mathrm{CV}(x)^2}{n}.
\]

Thus the first anchor maximizes empirical CV, subject to ties and floating-point
effects. Zero columns remain zero and cannot supply an independent anchor.
This identity does not rank all later anchors. In exact arithmetic, writing
\(z_j=x_j/\|x_j\|_1\) and letting \(Q\) contain the previously selected orthonormal
directions, the next squared residual norm is

\[
\|(I-QQ^\top)z_j\|_2^2
=\frac{1+\mathrm{CV}(x_j)^2}{n}-\|Q^\top z_j\|_2^2.
\]

A high-CV column nearly in the selected span can lose to a lower-CV signal
column. This uses uncentered orthogonal geometry, not Pearson correlation.
Consequently, it is incorrect to infer that every background column must be
selected before every signal column merely from their initial CV ordering.

The original simulation combines clipped high-variance background columns with
lower-variance class-shifted signal columns. Background features that survive
screening can occupy the available anchors at small rank. Empirical CV ranges
from particular draws are observations, not distribution-wide bounds on every
finite sample. The [October 20 diagnostic](wang-simulation-validation.md#october-20-anchor-and-information-diagnostic)
documents representation loss despite full signal retention, together with
cases where rank expansion or coordinate scaling is insufficient.

The retention theorem does not promise to exclude every irrelevant feature.
Likewise, SPA's recovery guarantees concern geometric anchors under separability,
rank and noise assumptions, not supervised feature importance. Retaining the
generating support and recovering an anchor representation are distinct from
preserving the class posterior. Zero anchors from the generating support also
does not prove that projected coordinates carry no signal: the fitted basis
uses all retained columns. In correlated settings, features outside the
designated generating support can be useful for conditional prediction.

The present implementation and defaults are unchanged. A different anchor rule
would require separate methodological and empirical justification. Multiplying
a column by a positive screening weight before L1 normalization cancels exactly:
\(w_jx_j/\|w_jx_j\|_1=x_j/\|x_j\|_1\). Zero weights remove candidates instead.
Restricting candidates, weighting after normalization or disabling normalization
changes the selection geometry; the existing recovery guarantees do not transfer
automatically. Stronger screening, higher rank or another solver likewise needs
training-only evaluation and cost assessment rather than an assumed remedy.

## Solver and numerical conditions

CD and MU address the joint nonconvex Frobenius objective. Experimental
ANLS-BPP is another study implementation of alternating optimization. SPA
instead selects anchors and solves the resulting NNLS problems. Final Theorem
7.4.2 leaves classifier and representation information losses explicit;
Theorem 7.4.3 controls the latter through a suitable reference and coefficient
stability. Neither requires a global joint NMF optimum. A small training
reconstruction objective alone establishes neither an informative reference
nor the required coefficient-error and posterior bounds.

SPA has a direct recovery argument in the restricted noiseless,
normalized, feature-separable model discussed on page 144. The package normalizes
training feature columns for anchor selection and fits the basis using the
original-scale anchor columns. Those anchors remain available as factorization
diagnostics. The current SPA classifier and, since version 0.5.0,
standalone `FeatureSPA.fit_transform` project training observations
against the fitted basis. Historical studies using anchor-derived training
coordinates retain that recorded convention. Extending recovery claims to noisy data
requires the relevant geometry, rank, conditioning and noise bounds, as well as consistent
component scaling. Recovery alone also does not establish posterior sufficiency.
These conditions have not been established on CelebA. This specialized result
therefore does not certify the fitted CelebA representation. Selecting SPA
does not establish the reference assumptions of Theorem 7.4.3. Final
Corollary 7.4.2 gives a separate consistency result in its specified separable
population model, described below.

The mathematical SPA prediction map comprises independent fixed-basis NNLS
problems. Its original SciPy execution solved one observation at a time.
The sklearn CD transform and explicit MU `mu_projection="batch"` use shared
stopping criteria across a query batch, although their fixed-basis Frobenius
objectives separate by observation. Batch MU checks relative improvement in the
whole batch's reconstruction error every ten iterations. Its stopping iteration,
coordinates and potentially predicted
labels can therefore depend on which observations share the call. The package's
iterative MU numerical recovery paths preserve this stopping rule in batch mode. A tighter
tolerance does not guarantee batch invariance. Disabling early stopping removes this
particular dependence but does not guarantee convergence or bitwise agreement
across batch shapes.

The current default `mu_projection="auto"` resolves to `"rowwise"` for MU and
no MU policy for SPA/CD. Rowwise MU assigns each observation its own initializer
and stopping decision. This fits the per-observation predictor interpretation
of the separable fixed-basis objective, without relying on a particular query
batch. It retains the fixed-basis unregularized Frobenius
objective, uses float64 working arithmetic for ordinary projection, and retains
guarded per-observation recovery for numerical-range failures. This removes the
shared stopping mechanism; finite precision, matrix-kernel shapes and iteration
limits still require an approximation argument. It does not establish bitwise
pointwise execution, an independently certified objective gap or convergence.
The policy is resolved from the solver at fit time and preserved with the model,
rather than chosen from incoming batch composition. Guarded execution can batch
independent rows or use a singleton path without changing that policy. Explicit
batch MU remains available when its throughput benefit warrants accepting
composition-sensitive stopping; approximate batch transforms are not inherently
invalid, and rowwise projection does not promise higher classification accuracy.

A deterministic synthetic probe
in the tested environment found a maximum batch-versus-single-row CD coefficient
difference of `2.2843480110168457e-05`, with **no changed labels or probabilities**;
SPA coefficients were identical in that probe. All fits and transforms qualified.
Those observations concern CD and SPA in that saved probe; they do not establish
MU's sensitivity or exclude label changes in other settings. Batch coupling is
a finite-tolerance implementation issue, not by itself a refutation of the
final results. A strict pointwise CD or MU theory-to-code statement should account for
its numerical/batch context, or specify pointwise encoding. CD and explicit batch
MU retain shared stopping criteria; the default rowwise MU map also needs its
numerical approximation specified.
The current guarded batched SPA execution preserves
independent NNLS objectives, but its kernel shape and acceleration/fallback
choices can depend on the query batch. That code should not be described as
bitwise pointwise merely because the mathematical objective separates. The
historical probe describes its saved implementation, not every current execution
configuration. A proof-to-code statement must specify the numerical map or
account for its approximation.

## Mapping the final results to current package policies

The current SPA classifier and standalone `FeatureSPA.fit_transform` use a
common fixed-basis NNLS policy for training and new observations. Uniform
Euclidean KNN, unscaled coordinates and native binary voting are defaults.
Recoding `classes_[1]` as 1, native uniform voting gives the theorem's strict
`q > 0.5` rule, with a tied vote assigned to `classes_[0]`. After fixing the
feature set/settings, basis fitting and the optional KNN scaler use selected
feature values; labels enter the KNN votes. These choices align with the
structural form of Theorem 7.4.4. The population, selection-complexity, local-bias
and numerical conditions still need to be established in a particular use.

| Package option | Scope of the final theory |
| --- | --- |
| Uniform KNN, common coefficient map, native binary decision | Matches the classifier form in Theorem 7.4.4 after binary recoding, subject to its other assumptions. |
| Positive diagonal KNN coordinate scaling learned from selected training features | Allowed by the general reused-sample framework; Corollary 7.4.1 includes the inverse scale in its neighbor-distance term. |
| Distance-weighted or callable KNN weights | Supported software options, but the displayed equal-vote bound is not a weighted-vote theorem. An extension needs its own concentration/local-bias analysis. |
| A changed decision threshold | Supported software option; the stated theorem uses 0.5 and does not supply the same bound verbatim for another decision rule. |
| CD/MU with automatic training-coordinate policy | Retains joint-fit training coefficients. `training_coordinates="transform"` requests the common fixed-basis policy for training and prediction; the numerical qualification above still applies. |
| Logistic regression or linear SVM | Separate classifier extensions. The general risk decomposition can retain an appropriate classifier-error term, but their estimation/approximation guarantees do not follow from the uniform-KNN result. |

For the rowwise MU policy at both stages, use `solver="mu"` with
`training_coordinates="transform"`; `mu_projection="auto"` resolves to rowwise,
or it can be requested explicitly. The default training-coordinate policy still
uses joint-fit coefficients for MU/CD. The 0.2.0 SPA classifier defaulted to
anchor-derived training coordinates. Version 0.3.0 changed the classifier
default to projected coordinates. Standalone SPA and the classifier's explicit
`training_coordinates="fit_transform"` option retained anchor-derived training
coordinates until 0.5.0. Use the originating package and dependency versions
to load those fitted models. [Model persistence](api.md#model-persistence)

Current calibration defaults to ordinary accuracy, with balanced accuracy an
explicit option. Theorem 7.4.4 allows arbitrary training-based selection, so
choosing settings by balanced accuracy does not by itself violate that
selection condition. Its conclusion remains about ordinary 0–1/Bayes risk,
not balanced accuracy, AUC or average precision.

## What the reference and consistency results require

Corollary 7.4.1 controls local bias using posterior approximation and coefficient
errors at **both selected training neighbors and the fresh query**. Its
distance term includes the inverse fitted positive diagonal scale. Applying
Theorem 7.4.3 additionally needs an aligned informative reference basis with the
same selected features, rank, component order and scale, suitable conditioning,
small basis perturbation and controlled objective gaps.

In final Equation 7.22, the pointwise gap is the achieved squared reconstruction
objective minus its exact fixed-basis NNLS minimum. `reconstruction_err_` is a
training residual diagnostic, sometimes using original fit coordinates; it is
not that gap. Nor does a warning-free solver stop certify a population
coefficient-error bound. A small fresh-query reconstruction residual alone
does not bound posterior residuals or errors at the selected training neighbors.

For consistency through the general reference corollary, screening loss,
reference-event failure, local bias and the selection term must vanish along
a sequence of procedures. In particular,
\(\log(2/(\alpha_N\omega_{\widehat\theta_N}))/k_{\widehat\theta_N}\to0\)
is required together with \(\alpha_N\to0\) and the stated local-error controls.
A fixed constructor neighbor count or one finite CV result is not evidence
of that asymptotic conclusion.

Final Corollary 7.4.2 supplies a more explicit result in an exactly separable
population model. It assumes fixed signal rank and support size, a fixed
nonnegative loading matrix whose only pure columns are the unit anchors, a
latent density supported on \([1,2]^K\) and bounded below there, a Lipschitz
posterior, positive probabilities for both classes, and marginal separation
of every signal feature. The bounded background vector has coordinate
variances uniformly bounded below and is independent of the latent
coefficients and class label; its coordinates may be mutually dependent.
The ambient dimension obeys \(\log d_N=o(N)\).

The specified procedure uses the cutoff

\[
\gamma_N=\sqrt{\frac{n_0n_1}{N}}
\left(\frac{\log\{N(d_N+1)\}}{N}\right)^{1/4},
\]

the correct rank, exact SPA/NNLS with the common map, no additional scaling,
uniform Euclidean KNN, fixed 0.5 decisions, deterministic label-independent
distance ties, and class-0 completion on failed fits. For deterministic
\(k_N\to\infty\), \(k_N/N\to0\), it attains the full-feature Bayes risk;
\(k_N\asymp N^{2/(K+2)}\) gives the stated \(N^{-1/(K+2)}\) excess-risk rate.
This is an established result under its model and procedure. The package's
CV cutoff, finite NNLS tolerances, optional scaling and ordinary error handling
are not automatically that complete sequence.

Decoded, appropriately scaled CelebA pixels are bounded, but this alone does
not establish iid sampling, exact separability, informative reference
coefficients or the other population assumptions. A Gaussian simulation
clipped only below zero is not bounded above and does not satisfy the bounded
feature assumption as written.

The optional logistic classifier needs approximation/estimation control for
its regularized linear log-odds model, including its fitted coordinate scaler.
The linear SVM needs an appropriate margin/loss argument. Optimization
convergence does not supply those statistical guarantees. These remain
classifier-specific questions, rather than an objection to ordinary
same-sample pipeline training.

## Wang's comments and the decision

Wang's R10 requires reconciling percentile practice with threshold theory; absolute
screening is one permitted remedy. R15 strongly recommends an SPA comparison and
also permits comparing both methods or distinguishing the theorem's estimator.
The final decomposition and coefficient-stability arguments concern the fitted
representation without attributing SPA recovery guarantees to another solver.
CD and SPA have both
been compared, together with MU and ANLS-BPP. R7's screened-PCA and
screening-only comparisons remain separate evidence about NMF's contribution.
The October 6 CelebA study compared KNN with L2-logistic regression on raw
features and fitted representations, including screened SPA. In that study
SPA used anchor-derived training coordinates and NNLS projections for
validation. The study recorded training and prediction costs. Over ten
independently retuned training samples,
screened SPA with logistic regression improved mean balanced accuracy over
screened SPA with KNN on both attributes. Screening-only logistic regression
scored higher still. These are comparisons of the complete tuned procedures.
A classifier-specific bound for the logistic error term remains open.
[Matched CelebA results](wang-real-data-validation.md#completed-independently-retuned-results)

Absolute screening is therefore preferred for methodological/theoretical alignment
when accuracy is close. General bounds can assess different solvers under
their stated conditions; the separable consistency corollary specifies SPA.
SPA with absolute screening was selected by the revised CelebA policy
using the stated accuracy tolerance and measured time, and by the subsequent
four-solver comparison. Its specialized recovery assumptions have not been
validated on CelebA. Final same-sample KNN theory removes the old need to appeal
only to a separately split classifier sample; it still retains screening,
local-bias, configuration-selection and applicable reference/numerical terms.

## October 20 numerical and classifier extensions

Subtracting a training feature's minimum before computing pooled t statistics
preserves the exact statistic under translation. Scaling the remaining spread
avoids losing represented variation behind a large common offset. This repair
does not center the feature matrix supplied to NMF or change the screening
theorem's retention assumptions.

For fixed-basis NNLS, multiplying both the design and one RHS by the same positive
power of two multiplies its squared objective by a positive constant, preserving
its minimizers. The implementation checks that scaling is exactly reversible and
uses independent RHS scale groups; a row's admissible scaling interval depends
on that row and the shared design, not on unrelated query magnitudes. This does
not imply batch-independent acceleration or identical floating-point output.
Scaling does not reduce the condition number, establish
uniqueness, or repair the separate experimental alternating ANLS-BPP solver.

The optional noncentering SD map for KNN is fitted on training coordinates.
When each fitted scale is its strictly positive empirical standard deviation,
with no constant or near-constant fallback, it removes positive coordinate-scale
dependence from Euclidean neighbor distances in exact arithmetic. Constant and
near-constant columns follow the fitted scaler's numerical policy and do not
have that general invariance guarantee. This is a different metric from
raw-coordinate KNN, so its statistical assumptions and empirical comparisons
must be assessed separately.

The optional linear SVM introduces a linear-margin approximation and a
regularized squared-hinge estimation problem in the learned coordinates.
The general pipeline risk decomposition can retain its classifier-error term,
but the KNN corollary is not an SVM theorem. Margin calibration arguments also
require their own assumptions. Convergence of the primal optimization does not
prove ordinary-risk consistency or a balanced-accuracy guarantee. This extension
remains within the NMF classification part of the dissertation.

Stable factor-recovery cosines and blocked training residuals are reporting
calculations. They supply neither a population reconstruction bound nor a
numerical optimality certificate. Their implementation changes do not strengthen
the proof assumptions described above.
