# Wang concern tracker

These comments concern an earlier 104-page dissertation draft. This tracker records
package-development evidence and remaining work. Some comments have already been
addressed in the October 1 revision; the status below is limited to package work.

Author: Chang Cheng. The repeated development comparison completed 72 valid
evaluations, 216 exact-match serial profiling repeats and 2,050 qualified CV
candidates. The original automatic rule selected SPA with percentile screening.
The author's later supervised-PCA alignment preference selects **SPA with an
absolute pooled-t cutoff**, within the same global accuracy tolerance. The
original decision is preserved; the amendment is disclosed after development
results and before official-test predictions/outcomes. See the
[default results](celeba-default-results.md) and
[theory/implementation scope](theory-implementation.md).

The completed [four-solver comparison](nmf-solver-comparison.md#development-results)
adds Fashion-MNIST and a matched CelebA extension, with absolute screening fixed.
SPA alone passes its accuracy gates; the report remains `selected_for_audit`.
SPA, CD and MU are public options; ANLS-BPP remains experimental. No new
official-test run has started.

| Concern | Development response | Status |
| --- | --- | --- |
| CD versus SPA, R15 | Both estimators were compared after training-CV refinement across nine overlapping CelebA training configurations and two attributes. A subsequent Fashion-MNIST/CelebA comparison added MU and ANLS-BPP. | SPA selected by both the CelebA amendment and the four-solver rule; final evidence audit remains open. The conditional representation bounds do not favor SPA. Its separate anchor-recovery guarantees require geometry/noise assumptions not established for these datasets. |
| Percentile versus absolute threshold, R10 | Both rules were tuned and compared on CelebA. The author subsequently fixed absolute screening for the external follow-up. | Absolute screening retained; the cutoff remains tunable. A CV cutoff does not automatically equal the theorem's prescribed threshold; full-support retention and posterior sufficiency are not established. |
| Contribution of NMF, R7 | Independently tuned screened PCA, screening-only KNN, raw KNN and unscreened NMF in the first CelebA study; matched PCA/screening-only/raw-KNN baselines in the external study. Logistic versions of these paths, including raw-feature logistic regression, are now available. | Matched Fashion-MNIST KNN baselines outperform the NMF methods in mean balanced accuracy. CelebA baseline refinement and replication, and matched logistic development runs, remain pending. R7 remains partially addressed. |
| Standard NMF 32.42%, R8 | Exact original source retained; fixed-basis transformations and coefficient diagnostics available. | Historical numerical reproduction and causal diagnosis unresolved. |
| Wearing Necktie imbalance, R9 | Tune with balanced accuracy; report ROC AUC, average precision/PR evidence, positive precision/recall/F1, confusion counts and prevalence separately from Smiling. Preserve natural class proportions and include majority prediction. | Imbalance-aware reporting covers the first study and all 72 repeated evaluations. The same validation sample has 292 positives; minority recall remains limited. Reused validation images do not provide independent replication. |
| Constant/near-constant features | Explicit score policy; no hidden variance floor; invalid fits recorded. | Numerical unit checks pass; theoretical denominator assumptions remain separate. |
| Harder simulations, R16 | Configurable signal strength, support, correlation and XOR case. | Generators implemented; full factorial/replication study pending. |
| Replication, R6 | Nine training configurations (three sizes by three seeds), paired across four combinations and two attributes; saved predictions. | Completed scoped default comparison. Samples overlap and share validation images; this does not fulfill the full requested independent-replication/standard-error study. |
| Tuning cost, R18 | Separate CV/calibration, screening, factorization, classifier fitting and prediction; count shared fold costs once. | Repeated CV wall time 319.674 s with eight workers; selected evaluation plus 216 serial profiles took 530.846 s. Per-method worker-wall sums and serial stage timings are separate. Whole-process peak memory and broad scaling remain unmeasured. |
| Reproducibility, R21 | Original hashes, version metadata, seeds/folds/configs, warnings and resumable records. | Source is available in the private [cheng338/supervised-nmf repository](https://github.com/cheng338/supervised-nmf); a distribution license remains undecided. |
| KNN versus parametric classifier, R14 | Optional L2-regularized logistic regression with training-fold scaling, independent C tuning and classifier convergence reporting. KNN remains the default. | Implementation available. Matched development comparisons and classifier-specific theoretical analysis remain pending; the recommendation is not yet resolved by empirical evidence. |
| Theory/overclaiming, R4/R11/R13 | Reviewed the four revised Chapter 7 results and separate sample-split KNN corollary against the implementation and Wang's original requests. | The representation bounds accommodate the fitted solvers conditionally; threshold/percentile retention have different assumptions. No recovery, global CD convergence, consistency, or balanced-accuracy guarantee follows from the development results. No dissertation proof changes made. |
| Related work/attribution, R20/R22 | Author attribution, citation metadata and original-source provenance preserved. | Dissertation bibliography and external figure permissions remain separate. |

