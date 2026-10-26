# Development decisions and experiment protocol

Author: Chang Cheng.

## Research policy — October 20, 2026

Small differences in accuracy or execution speed do not determine adoption by
themselves. Decisions also consider mathematical fidelity, theoretical support,
numerical reliability, practical benefit, software design, maintenance cost,
and uncertainty in the evidence. Analytical guarantees remain distinct from
empirical observations: benchmark performance alone does not establish
correctness or theoretical validity. Memory savings do not justify slower
execution, including through an optional implementation.

The validation host has 20 physical cores and 40 logical threads. The October 20 integration
used a budget of 20 total compute threads across workers and native numerical
threads. Historical configurations and results retain their recorded settings.

Python 3.10 support ended on October 20, 2026. Current development requires
Python 3.11 or later, with Python 3.12 and AOCL as the primary environment.
Previous requirements, results, and released artifacts remain historical records.

## Public source, rights and citation — October 20, 2026

The package is publicly available on GitHub, with All Rights Reserved for
Chang Cheng's original work pending a distribution-license decision. Source and
wheel archives include both the project LICENSE and THIRD_PARTY_NOTICES,
preserving the separate third-party copyright and BSD notices.
Public source availability does not supply an open-source license for the
author's work, and historical release artifacts remain unchanged.

The dissertation title is *Ensemble Kernel Distance-Weighted Discrimination and
Supervised Nonnegative Matrix Factorization for Classification*. The package
implements the supervised NMF methodology in Chapter 7.

The public repository uses standard GitHub-hosted runners for CI.

## Classifier scope decision — October 20, 2026

After reviewing the five-classifier face-image pilot, Chang Cheng selected
three classifiers for continued package development:

- K-nearest neighbors (KNN).
- L2-regularized logistic regression.
- Linear support vector machine (linear SVM).

Histogram gradient boosting and shrinkage LDA are outside the adopted
classifier scope. Preserve their completed pilot results, study adapters,
failures and provenance as historical evidence. This scope decision supersedes
the pilot's recommendation to retain boosting as a candidate for the proposed
classifier interface; it is not a claim that the excluded methods performed
worse.

KNN remains the package default. KNN and logistic regression were already public
classifier options at the time of this scope decision; linear SVM existed in
the study adapter. The integration below subsequently completed the public SVM
API and its integration checks; it is included in version 0.3.0. SVM decision margins
are reported as scores, not probabilities. DWD remains outside the current project.

The face-image validation plan covers these three classifiers, with training-only
transformations and explicit imbalance handling.

## Scope and method

1. Follow Chang Cheng's dissertation and original scripts; preserve his authorship.
   Report every bug fix, performance change, or methodological departure.
2. Track Wang's NMF concerns against the available evidence and remaining work.
3. Determine coordinate descent versus SPA and percentile versus absolute
   screening through development comparisons that consider time and accuracy,
   without unsupported transfer of theoretical guarantees. The October 3 CelebA
   author-preference amendment selected
   [SPA with absolute-score screening](celeba-default-results.md). The subsequent
   comparison added Fashion-MNIST, MU and ANLS-BPP. The completed
   [four-solver report](nmf-solver-comparison.md#development-results)
   selects SPA and remains `selected_for_audit`. SPA remains installed; absolute
   screening is fixed in that comparison. Both original CelebA decisions are preserved.
4. Include screened PCA and screening without factorization as classification
   comparisons, together with raw-feature baselines. KNN remains the default;
   optional logistic regression is evaluated independently for the NMF study.
5. Complete tuning and numerical confirmation: expand boundary winners and refine
   interior intervals. An exhausted time/search budget is not completion.
6. Fit all learned transformations inside training folds. Use five-fold calibration;
   freeze qualified settings before access to the final test data. Report tuning
   and final-fitting cost separately. Save failures, seeds, source IDs, checkpoints,
   and compatible completed results.
7. Training allocations are specified by each study; the earlier 20% rule was withdrawn.
8. The primary study environment uses Python 3.12 with AOCL. Local CelebA images,
   attribute annotations, and official partition metadata were available on October 3, 2026.
   Preserve the published partitions in the training/validation examples.
9. The additional adaptive workflow was withdrawn. Use the package's
   own explicit grids for NMF/PCA rank, screening and the final classifier. Record grid
   choices, boundary checks, and any expansions or refinements as study decisions.
   No external search policy is required.
10. Freeze external development protocols before evaluation. Official-test
    evaluation requires a completed evidence audit and a frozen evaluation plan;
    preserve the interrupted prior CelebA attempt.
11. On October 5, 2026, the public API added `solver="mu"` alongside `"spa"`
    and historical `"cd"`, preserving the
    SPA/absolute defaults and the screening-to-factorization-to-KNN pipeline.
    MU retains the experimental comparison's unregularized Frobenius objective
    and NNDSVDa initialization. ANLS-BPP remains experimental. Record this API
    extension separately from the completed, frozen benchmark evidence.
12. Logistic regression is an explicit extension of the dissertation's KNN
    classification stage. Fit its coordinate scaler within training folds;
    tune inverse L2 regularization strength independently for each representation.
    Preserve the historical KNN comparisons and solver/screening decision.
    Matched logistic comparisons and their theoretical assessment remain separate
    work before reconsidering the final classifier default.
13. On October 5, direct BLAS calls were investigated before
    the Wang validation resumed. SPA now uses public SciPy GEMV/GER updates
    and one fixed-basis layout conversion per transform. The
    [performance and equivalence record](blas-optimization.md) distinguishes
    this implementation change from the statistical method. Preserve the
    interrupted study snapshots and use a new source freeze for subsequent
    runs; do not silently combine timing records from different implementations.
14. On October 6, a review of the four-method face validation identified the
    omission of the dissertation's standard NMF+KNN baseline. A separate supplement
    restores unscreened CD-NMF+KNN and adds unscreened SPA-NMF+KNN, using the same
    cohorts, independent rank/K tuning and public-estimator qualification.
    The original four results remain frozen. Since the original test outcomes
    were already known, the additions are descriptive supplementary comparisons,
    not new unseen-test confirmation. Preserve numerical failures and any
    unresolved rank boundaries; do not claim globally optimized baselines.
15. On October 6, screened CD-NMF+KNN was added to the comparison plan as a seventh
    method after observing the unscreened CD/SPA comparison. Use the same
    absolute-screening, rank and dense-K starting grids, five training folds,
    single local refinement and numerical qualification rules as the frozen
    face comparison. Independently tune the screened CD pipeline; preserve all
    six earlier fits and results. This is another post-test-designed descriptive
    supplement, not new independent confirmation. Comparing the tuned screened
    SPA and CD pipelines may change cutoff, rank and K as well as solver;
    do not interpret it as a fixed-configuration causal solver effect or use
    its test outcome to revise package defaults.
16. On October 6, the validation scope expanded beyond a single
    face task. The first expansion fixes two new
    face datasets (FEI smile/neutral and CMU sunglasses/open) and three naturally
    nonnegative nonface datasets (Sonar, Spambase and WDBC) before fitting.
    Evaluate all seven KNN pipelines on every included task, retaining losses,
    inconclusive comparisons and failures. Face identities must remain disjoint
    across both the reserved split and every inner training/validation fold.
    Retain the existing nonface preparation's reserved partitions. Use matched
    five-fold tuning and the declared dense neighbor grid, with physical
    small-sample omissions recorded before fitting. This first expansion is a
    multiple-dataset benchmark, not a completed repeated nested-CV study or a
    claim that R6's entire replication request has been resolved. Reports include
    accuracy, fit-time and prediction-time tables.
17. On October 6, additional labels in the existing datasets, including FEI,
    were added to the exploratory study plan. Explore observable CelebA attributes,
    CMU source pose/expression conditions, and FEI source acquisition conditions
    using all seven pipelines. Freeze the task list, development cohorts,
    label coding, reduced image representation, grids and support rules before
    fitting. Keep old official-test and reserved subjects outside this
    exploratory study. Several labels from the same people are correlated
    tasks, not independent datasets; previously used training images are not
    fresh confirmatory evidence. Retain every task and failed method. Use
    balanced accuracy to assess imbalanced attributes and report majority and
    random-guess references with ordinary accuracy. Do not assign semantic
    pose/expression names where the source only verifies condition identifiers.

## Stages and what constitutes evidence

On October 9, training-only prototype tests supported integration of
guarded batched SPA projection, optional reusable process workers,
separate projection/native-thread controls, and a saved decision threshold.
The implementation changes query execution while preserving basis fitting
and the NNLS objective. New source models use `projection_solver="auto"`;
old serialized models retain SciPy. Numerical checks and fallback are required.
Saved benchmarks keep their original source fingerprints; subsequent benchmarks
must freeze this new source and record all execution controls. Optional
decision thresholds preserve the already-tested strict probability rule;
they do not supply new accuracy evidence. See the [API details](api.md#spa-projection-execution).

The remaining diagnostics supported early release of SPA anchor-selection
workspaces. This memory optimization occurs after the
anchor condition checks and before basis NNLS. It preserves the tested
numerical path and estimator defaults. The normalization results do not justify
a universal KNN normalization default. The completed study and diagnostic
receipts retain their frozen source hashes; applying this package fix does not
rerun or replace those studies.

- **Implementation checks:** unit tests and a small development run establish
  execution, isolation, deterministic behavior, and numerical edge-case handling.
  They cannot choose defaults or resolve Wang's scientific concerns.
- **Historical diagnosis:** reproduce the old simulation configuration and examine
  the 32.42% standard-NMF result. Preserve historical and revised configurations as
  separate rows. Change one diagnostic factor at a time. Do not choose label
  flipping using test labels or call a changed study design a bug fix.
- **Development comparison:** the completed CelebA study compared both screening
  rules and CD/SPA. The completed external follow-up fixes absolute screening and
  compares CD, SPA, MU and ANLS-BPP on matched datasets and folds, with independent
  tuning and relevant PCA/screening-only context. Record support recovery where known, predictive
  performance, convergence, conditioning, time and failures.
- **Protocol freeze:** document selected ranges, convergence settings, stopping
  criteria, refinement history, replications, and default decisions. A quick CLI
  run is explicitly provisional and leaves this stage open.
- **Final study:** execute the frozen protocol on untouched tests, then report
  paired differences and uncertainty without tuning on final outcomes.

The reference study proposes 27 simulation conditions with 30 independent data
replications, harder correlation/interaction cases, and 20 paired CelebA training
runs per attribute. These are a research protocol proposal, not completed work.
The initial implementation provides generators and auditable comparisons; it
does not imply this full study has been run. Repeated evaluations of the same
CelebA test images are not independent test samples.

Balanced accuracy is the development tuning metric, following the study draft.
Report accuracy, balanced accuracy, ROC AUC, average precision, precision, recall,
F1, prevalence and confusion counts. Include majority prediction. Distinguish
average precision from trapezoidal PR area. Revisit metric choices only using
development data, and record any change.

The feature-anchor SPA model and its normalization must be stated alongside
results. General clipped simulations and CelebA are not assumed separable.
Numerical convergence and empirical success do not prove the theory's assumptions.

## Parameters and numerical qualification

The main predictive parameters are rank, retained percentile or absolute pooled-t
cutoff, and KNN neighbor count or logistic inverse regularization strength.
Rank applies to NMF/PCA; the screening parameters
apply to their corresponding rule. Tune each method independently on the same
five training folds. The external follow-up compares CD, SPA, MU and ANLS-BPP with
absolute screening fixed; only its cutoff is tuned. Percentile comparisons remain available
for historical reproduction. Record choices explicitly rather than presenting
them as an unchanged historical dissertation method.

Iteration cap and tolerance control numerical qualification. Establish and record
these settings on training data before comparing predictive candidates. Reject
candidates with failed or nonconverged required folds and retain the diagnostics.
Any numerical change requires a recorded new configuration; a successful rank-2
smoke fit does not qualify larger ranks or samples.

Logistic regression has its own iteration cap and tolerance. Qualify both
factorization and classifier convergence; a converged factorization does not
excuse an unconverged logistic fit. The starting C grid is
`[0.01, 0.1, 1.0, 10.0, 100.0]`. Apply the same boundary and refinement checks
as for representation parameters, using training folds only. Use matched
partitions and metrics for KNN/logistic comparisons on NMF, PCA, screening-only
and raw-feature paths. Record class weighting as part of the configuration;
balanced accuracy does not require fitting with balanced class weights.

Screened PCA is a priority comparison for Wang's R7. Smiling and Wearing_Necktie
require separate tuning and reports. For the latter, report positive counts and
precision/recall measures prominently so majority accuracy cannot conceal a
failure to recognize positives. See [the CelebA study plan](celeba-study.md).

## Numerical test interpretation

Composition tests compare the package with the corresponding estimator pipeline
in the same environment, using matched data, dtype, layout, initialization and
training-coordinate policy. Preserve their strict assertions. Independent
numerical checks, such as the rank-one projection oracle, serve a different
purpose and should continue to cover supported dtypes, layouts and input
immutability.

Environment comparisons use the same source, data, parameters and seeds, with
controlled thread settings and recorded numerical-library builds. Compare
selected features, finite/nonnegative factors, the original reconstruction
objective, appropriate projected-gradient diagnostics, warnings, iterations,
predictions and measured costs. For fixed-basis projection, only W is optimized;
an H gradient is not a query-solve optimality criterion. State diagnostic
normalizations before comparing their values.

Raw factor equality is not a portability requirement: NMF admits component
permutations, positive rescaling and different local solutions. Nearby objective
values or unchanged accuracy do not establish equivalent predictions. Investigate
material differences with common initialization, basis and classifier-coordinate
replays. A large numerical discrepancy does not identify its causal importance;
use controlled interventions to distinguish mechanisms.

Do not relax suite-wide tolerances or change solver stopping settings merely to
make environments agree. Retain failed cases and justify any test-specific
tolerance or design change mathematically. Warning-free completion does not
independently establish stationarity or global optimality. Use float64 as an
explicit study choice where warranted, recording its cost and the changed
configuration rather than altering the package's dtype behavior.

## October 20 implementation

Following the retrospective review, the package adopted minimum-reference
screening statistics, conditional per-RHS NNLS scaling, stable factor-recovery
reporting, guarded neighbor-query reuse in explicit KNN grids, optional fitted
SD scaling for KNN, and a public linear SVM head. Report-only residual blocking
was conditional on integrated cost checks supporting its dispatch domain.
The completed checks support a bounded dense float64, single-thread MKL path;
other execution settings retain the original residual expression. See the
[validation record](validation.md) for the completed integration evidence.
These changes were released in version 0.3.0; earlier releases and study
receipts retain their original implementations and configurations.

The screening and NNLS repairs preserve their mathematical objectives.
Minimum-reference subtraction precedes scaling when computing pooled statistics;
the selected data passed to factorization remain uncentered. NNLS uses a common,
exactly reversible binary scale for a basis and each RHS, grouping equal scales
for execution. It does not improve conditioning or certify every extreme input.
Unrepresentable ranges fail explicitly. Factor-recovery reports normalize before
forming cosines while retaining full-truth support and unmatched-rank penalties.

KNN SD scaling and linear SVM are optional statistical choices, rather than
numerically equivalent replacements for the existing classifier. SD scaling is
fitted within each training fold, changes the KNN metric, and leaves factorization
coordinates unchanged. The SVM head fits an unscaled, L2-regularized squared-hinge
primal problem. Tune its C independently of representation parameters and qualify
its own iteration cap and tolerance. Its decision margins support AUC/AP ranking;
they are not probabilities. A probability threshold is inapplicable.

Neighbor reuse is internal to compatible fold-local KNN grids. Unsupported or
numerically uncertain cases use the selected-k public estimator on the original
query batch. Reports charge actual shared query costs once and retain separate
head costs. Old checkpoint identities cannot resume under changed package source.

The integration batch ran on the validation host with at most 20 total compute
threads. It covered correctness, API/serialization contracts, original numerical
triggers, and targeted paired timing. It did not rerun the historical dataset
studies or retune defaults, and it does not establish a new theorem.
Adoption weighs mathematical fidelity, numerical reliability, software design,
maintenance and measured cost together. A memory saving cannot justify a
slowdown, including through an optional implementation.

## October 25 SPA basis integration

The high-rank scaling investigation supported reusing the guarded batched NNLS
kernel for SPA basis fitting. The fixed anchor coordinates,
screening rules, NNLS objective, classifier defaults and projection controls are
preserved. A basis-specific internal wrapper retains the original public SciPy
column loop as a compatibility fallback for numerical-range rejection. It adds
no public execution option or nested basis workers.

The implementation retains the 1e4 normal-equation condition guard, numerical
qualification and fitted-basis condition check. Near the guard, small residuals do not imply
the same coefficient precision as public SciPy. Batched pivot rounds and SciPy
iterations are different units; fallback receives the original iteration cap.
These limits accompany the runtime evidence and shared-kernel maintenance
benefit in the adoption decision. Historical source snapshots and study results
remain unchanged. Existing serialized bases are preserved; refitting uses the
current implementation.

## October 26 MU precision recovery

MU range correction applies to fixed-basis projection, including an explicit
training reprojection. Joint factorization and the default training-coordinate
policy remain unchanged. Exact zero batches return zero coordinates after full
input validation. Positive values are not classified as zero by a tolerance.

Ordinary float32 observations can encounter tiny fitted basis tails or an
underflowing update without requiring a full logarithmic solve. The projection
therefore retries eligible float32 calculations in float64 before logarithmic
recovery. It uses a temporary estimator and leaves the fitted basis unchanged.
Tiny positive inputs and unresolved range failures retain logarithmic recovery.
Both paths preserve the MU recurrence and stopping rule in exact arithmetic;
finite precision can change rounding and stopping iterations. Output dtype,
iteration-cap warnings and errors for unrepresentable coordinates are retained.

The evaluation separates three implementations: the published 0.3.2 package,
the earlier range correction and the revised precision recovery. It considers
ordinary data, the failures that motivated each correction and difficult
numerical cases. A large improvement over the earlier correction does not
erase remaining overhead relative to the published package. Correctness,
methodological fidelity, numerical reliability and measured cost all enter the
decision; memory reduction is not a justification for slower execution.

