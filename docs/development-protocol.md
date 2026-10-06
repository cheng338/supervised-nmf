# Development decisions and experiment protocol

Author: Chang Cheng.

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
8. Use conda environment `py12_df_aocl`. Local CelebA images,
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

## Stages and what constitutes evidence

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

