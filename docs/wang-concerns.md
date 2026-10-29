# Wang concern tracker

These comments concern an earlier 104-page dissertation draft. This tracker records
package-development evidence and remaining work. Some comments have already been
addressed in the October 1 revision; the status below is limited to package work.

The October 3–6 SPA classification evidence summarized here used the saved
study sources and anchor-derived `fit_transform` training coordinates, with
fixed-basis NNLS for prediction. Since
[v0.3.0](release-0.3.0.md#spa-training-coordinates-migration), the default SPA
classifier also projects its training observations. The historical results
support the procedures evaluated then; they do not measure the later default's
accuracy or timing.

Author: Chang Cheng. The [completed validation report](wang-validation-report.md)
adds **900 independently retuned simulation draws and 780 CelebA fits**, all
numerically qualified and independently audited with zero discrepancies.
The results show classifier-dependent benefits and retain mixed or negative
comparisons. They provide empirical evidence under the stated designs, not
proof of recovery, sufficiency or consistency. That study accessed no official
image-test outcomes.

The later [identity-disjoint CelebA Smiling validation](celeba-identity-validation.md)
adds a frozen official-test comparison on 1,000 separate identities after
five-fold tuning on 2,000 training identities. It addresses the limited KNN grid
with 104 initial neighbor values for every method. sNMF+KNN reaches 81.40%
accuracy versus 78.50% for sPCA+KNN; its balanced-accuracy advantage is +2.86
points (paired 95% interval +0.68 to +5.05). Independent audits verify all 4,690
candidates, 23,450 candidate-fold records and 4,000 test predictions. This extends
R7/R21 evidence and resolves identity separation for this cohort. The task was
selected from earlier development results; one fixed training cohort does not
resolve R6's broader replication concern. The screening-only contrast remains
inconclusive, and no universal or parametric-classifier superiority is claimed.
The historical analyses below retain their original scope and limitations.

The [unscreened NMF supplement](celeba-unscreened-supplement.md) also restores
the standard CD-NMF+KNN baseline already present in the dissertation, which had
been incorrectly omitted from the four-method identity comparison. It adds
unscreened SPA-NMF+KNN for a comparison within the current solver family.
These independently tuned procedures use the same data and saved original
predictions, with the later, post-test design disclosed. Both added methods
select the upper explored rank of 80; their tuning does not establish a global
optimum. Standard CD achieves 73.00% accuracy and unscreened SPA 61.90%, with
both numerical qualification and independent audits passed. The CD final fit
converges at 2,199 iterations under its declared 20,000 cap. The historical
32.42% investigation remains a separate simulation
diagnostic, not the result of these face-data baselines.

The [screened CD supplement](celeba-screened-cd-supplement.md) completes seven
methods and restores the dissertation's supervised CD solver family as well.
It independently selects cutoff 8, rank 10 and K 39, achieving 79.00% accuracy
and 78.98% balanced accuracy; its final fit converges in 705 iterations without
warnings. Screened SPA reaches 81.40% accuracy, a 2.40-point balanced-accuracy
advantage with a descriptive, unadjusted 95% interval of [0.18, 4.62]. Both
use the same 124-pixel mask, while ranks and K differ. The screened CD contrasts
with screening-only KNN and screened PCA remain inconclusive. Both audits pass,
all six earlier results are preserved, and known-test reuse is explicit. These
results address a tuned-pipeline solver comparison, not universal superiority
or an isolated causal solver effect; package defaults remain unchanged.

The [five additional datasets](multi-dataset-results.md) extend the comparison
beyond CelebA: CMU Faces, FEI, Sonar, Spambase and WDBC. All 35 final evaluations
qualify; both independent audits pass, including 38,854 tuning candidates and
12,642 saved test predictions. Face splits and inner folds separate subjects,
and intervals resample whole subjects. Dataset inclusion was fixed before new
model outcomes. Screened SPA exceeds screened PCA on FEI (94.17% versus 91.67%),
ties on WDBC, and loses on the other three tasks. FEI's +2.50-point difference
has an interval of [-0.83, +6.67], so it is inconclusive. Chance and training-majority
references accompany the complete tables. This broadens R7/R21 evidence and
partly addresses R6 through additional datasets; one holdout per dataset still
does not measure repeated-training uncertainty. CMU has only six test subjects;
its [uncertainty sensitivity analysis](multi-dataset-results.md#cmu-uncertainty-sensitivity)
preserves the historical intervals and adds subject influence and alternative
intervals for every comparison. This improves reporting but does not establish
population superiority or resolve R6. FEI's filename a/b expression orientation
remains unverified.

The earlier repeated development comparison completed 72 valid evaluations,
216 exact-match serial profiling repeats and 2,050 qualified CV candidates.
The original automatic rule selected SPA with percentile screening.
The author's later supervised-PCA alignment preference selects **SPA with an
absolute pooled-t cutoff**, within the same global accuracy tolerance. The
original decision is preserved; the amendment is disclosed after development
results and before official-test predictions/outcomes. Its selection is
conditional on the accuracy allowances, assumed workload and development
configurations: the [amended sensitivities](celeba-default-results.md#amended-screening-preference-rule)
include a reversal when seed 44 is omitted and when prediction cost receives
greater weight. These results do not establish a universally preferred solver.
See the [default results](celeba-default-results.md) and
[theory/implementation scope](theory-implementation.md).

The earlier [four-solver comparison](nmf-solver-comparison.md#development-results)
adds Fashion-MNIST and a matched CelebA extension, with absolute screening fixed.
SPA alone passes its accuracy gates; that separate historical report remains
`selected_for_audit`. The new validation audit does not change that recorded status.
SPA, CD and MU are public options; ANLS-BPP remains experimental. That comparison
did not evaluate official-test outcomes.

| Concern | Development response | Status |
| --- | --- | --- |
| CD versus SPA, R15 | Both estimators were compared after training-CV refinement across nine overlapping CelebA training configurations and two attributes. A subsequent Fashion-MNIST/CelebA comparison added MU and ANLS-BPP. The new audited simulations include both SPA and CD. | SPA selected by both the earlier CelebA amendment and the four-solver rule; the latter report's separate `selected_for_audit` status is unchanged. The conditional representation bounds do not favor SPA. Its separate anchor-recovery guarantees require geometry/noise assumptions not established for these datasets. |
| Percentile versus absolute threshold, R10 | Earlier CelebA studies compared both rules; the new studies retain the author's absolute-screening choice and tune its cutoff inside training folds. Simulations now report generating-support precision, recall, false selections and full-support inclusion, with denominators and supplementary Wilson intervals. | Empirical support recovery is measured, not guaranteed. A CV cutoff need not equal the theorem's threshold. Even full generating-support inclusion can coexist with poor prediction; marginal screening fails in the XOR stress case. Population support retention and posterior sufficiency remain unproved. |
| Contribution of NMF, R7 | Completed independently tuned SPA, PCA, screening-only and raw-feature paths with KNN and logistic regression; CelebA also includes unscreened SPA and secondary PCA ablations matching SPA's mask/rank. All 120 fitted SPA/matched-PCA mask/rank pairs agree exactly. Primary PCA and SPA paths both receive the starting grid and local refinement; the secondary matched PCA classifier uses the starting grid only. | Benefit is classifier-dependent. In ten retuned CelebA samples, SPA helps KNN on Smiling; its Necktie KNN advantage over PCA has an interval including zero. PCA and simpler pixel logistic procedures outperform SPA logistic on both attributes. Fashion-MNIST and several simulations also favor simpler baselines. The completed comparisons do not support a general NMF advantage. |
| Standard NMF 32.42%, R8 | [Original-size controlled reconstruction](wang-historical-diagnosis.md): 32.75% at cap 400; changing only the cap to 2,000 gives 92.92%, converging at iteration 1,226 without warnings. | Below-chance behavior closely reproduced and a numerical cause demonstrated in the current environment. Exact historical versions/predictions remain unavailable; the clipping/topological-inversion explanation is not established. |
| Wearing Necktie imbalance, R9 | Both completed CelebA designs tune with balanced accuracy and preserve natural class proportions. Report ROC AUC, AP/PR curves, positive precision/recall/F1, exact confusion counts and the majority baseline on the same 4,000 validation images, including 292 positives. | Across ten fully retuned fits, SPA KNN recall is 0.152; SPA logistic recall rises to 0.721 at precision 0.167. This tradeoff does not establish the best procedure: PCA and raw/screened-pixel logistic baselines have stronger overall comparisons. Always-negative accuracy is 92.7% but recall is zero. Validation reuse and unknown person overlap limit generalization claims. |
| Constant/near-constant features | Explicit score policy; no hidden variance floor; invalid fits recorded. | Numerical unit checks pass; theoretical denominator assumptions remain separate. |
| Harder simulations, R16 | Completed 27 signal-strength/support/correlation cells plus cross-block correlation, null and XOR; 30 independent training/test draws per regime, each with full five-fold tuning. Each draw uses 160 training rows, 480 test rows and 320 features. All 9,930 final method evaluations qualify. A separate [October 20 mechanism diagnostic](wang-simulation-validation.md#october-20-anchor-and-information-diagnostic) covers all seven original low-score triggers and three previously observed follow-up cohorts. | Null and XOR screened methods remain near chance; the separately labelled generating-support XOR KNN oracle reaches 99.87% BA. The later controls implicate low-rank/anchor representation loss and KNN scaling despite full signal retention; rank expansion and common coordinates do not repair every case. That retrospective diagnostic has 426/432 qualified final classifier configurations and does not establish a universal remedy. These reduced-dimensional generators test specific regimes, not large-scale behavior or the theorem's assumptions. |
| Replication, R6 | Completed 30 independent draws per simulation regime, 20 fixed-setting CelebA training resamples after a disjoint tuning pilot, and ten fully retuned resamples. Saved paired predictions support mean/Monte Carlo SE and pointwise paired t intervals; fixed-case image bootstraps are reported separately. The later [five-dataset expansion](multi-dataset-results.md) adds 35 qualified method/dataset evaluations with whole-subject face splits. [CMU sensitivity results](multi-dataset-results.md#cmu-uncertainty-sensitivity) add subject-deletion and interval comparisons using saved predictions. | Earlier CelebA resamples overlap and reuse the same validation reservoir; their uncertainty is conditional. The five additional datasets broaden empirical coverage, with all outcomes retained, but each uses only one reserved split. Repeated-training uncertainty and broad population generalization remain unresolved. CMU has only six held-out subjects; nominal 95% coverage is not assured, and the sensitivity analysis is not a new independent evaluation. |
| Tuning cost, R18 | Retain 113,156 CV candidates and 565,780 candidate-fold receipts, with shared real-data representations counted once for stage costs. Separate tuning, screening, representation, classifier and prediction timings. The [BLAS benchmark](blas-optimization.md) compares four training-only workloads in 28 paired repeats. | Direct public BLAS calls plus NNLS layout reuse reduce median fit-plus-projection time by 12.6–16.6%, with identical paired outputs. A separate native memory supplement records whole-worker lifetime peaks of about 265–535 MiB, essentially unchanged; these include imports/data/screening. Shared-host study wall times are not isolated speed comparisons. Whole-CV peak memory and broad scaling remain unmeasured. |
| Reproducibility, R21 | Frozen source snapshots, data hashes, IDs, seeds, folds, complete grids, selected settings, failure records and predictions are preserved. The combined audit (`runs/wang-validation-audit-20261006/all-audit.json`, retained locally) checks 10,710 final evaluation records, recomputes metrics from 7,886,400 saved predictions and reproduces all 48 fixed-case bootstrap intervals. | Complete audit passed with zero discrepancies; final fits all qualify, with infeasible CV attempts retained. CV fold scores are checked from receipts, not independently refitted; checksums/timestamps establish internal consistency, not external preregistration. The final 303 tests pass in both numerical environments. Source is available in the [cheng338/supervised-nmf repository](https://github.com/cheng338/supervised-nmf). Chang Cheng's original work is All Rights Reserved while the distribution license remains undecided; separate third-party notices are preserved. |
| KNN versus parametric classifier, R14 | Completed KNN versus L2-logistic comparisons with training-fold scaling, independent C tuning and, for CelebA, tuned class weights. In the ten retuned CelebA samples, SPA logistic exceeds SPA KNN by 3.39 BA points on Smiling and 15.21 on Necktie. | The evidence supports including parametric baselines, while other logistic representations outperform SPA logistic. Contrasts concern independently selected complete procedures, not an isolated link-function effect. Classifier-specific theoretical analysis remains unresolved; the study does not automatically change the package's KNN default. |
| Theory/overclaiming, R4/R11/R13 | Reviewed the four revised Chapter 7 results and separate sample-split KNN corollary against the implementation and Wang's original requests. | The representation bounds accommodate the fitted solvers conditionally; threshold/percentile retention have different assumptions. No recovery, global CD convergence, consistency, or balanced-accuracy guarantee follows from the development results. No dissertation proof changes made. |
| Related work/attribution, R20/R22 | Author attribution, citation metadata and original-source provenance preserved. | Dissertation bibliography and external figure permissions remain separate. |

The approved October 20 source integration adds an optional linear SVM alongside
KNN and logistic regression, including matched representation comparators and
genuine margin-based ranking reports. Optional training-fitted SD scaling makes
the KNN coordinate metric an explicit choice. These additions support further
R7/R14 comparisons; their API and numerical checks are not new comparative
accuracy evidence and do not change the default classifier.

The same integration repairs screening sensitivity to a large common offset,
guards extreme-range fixed-basis NNLS, and stabilizes factor-recovery cosines.
It retains absolute screening, the fitted factorization objectives and the
full-domain truth penalties in recovery reports. These numerical repairs do not
establish screening support retention, separability, population reconstruction
or classifier consistency. Experimental alternating ANLS-BPP remains distinct
from the repaired fixed-basis projection solver.

