# Additional datasets for seven-method validation

This report preserves a historical study. Detailed records, data caches and
figures are retained locally and are not included in the public repository.
Study scripts marked 'retained locally' are also excluded; reproduction
commands refer to the retained local study workspace.

Author: Chang Cheng. Study preparation date: October 6, 2026.

Status: **completed and independently audited**, with all 35 final evaluations
qualified. See the [accuracy and timing tables](multi-dataset-results.md).
No new model accuracy was used to select these datasets. The expansion fixes five datasets
for evaluation with all seven KNN pipelines. This increases the number of
independent dataset tasks; it does not complete a repeated nested-CV study.
The existing CelebA results remain separate and unchanged.

Screened SPA sNMF reaches 94.17% accuracy on the additional FEI face task,
versus 91.67% for screened PCA. The paired subject-bootstrap difference is
+2.50 percentage points with a descriptive 95% interval of [-0.83, +6.67],
so that comparison is inconclusive. Across the five new tasks, SPA has higher
ordinary accuracy than screened PCA on FEI, ties on WDBC, and is lower on CMU,
Sonar and Spambase. These mixed results support reporting dataset-specific
behavior rather than a universal superiority claim.

| Dataset and source | Fixed binary task | Source size | Why include it |
|---|---|---:|---|
| [FEI](https://fei.edu.br/~cet/facedatabase.html) | Frontal expression condition a versus b | 400 images, 200 people | The source describes one neutral/smiling pair per person. Filename conditions are coded a=0, b=1; the accessible primary text does not verify which suffix denotes smiling. Accuracy is invariant to reversing both labels. |
| [CMU Faces](https://archive.ics.uci.edu/dataset/124/cmu%2Bface%2Bimages) | Sunglasses versus open eyes | 640 planned images, 624 usable, 20 people | Changes the binary target, pose distribution and acquisition source; use one quarter-resolution copy per exposure. |
| [Sonar](https://archive.ics.uci.edu/dataset/151/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks) | Metal cylinder versus rock | 208 records, 60 features | Nonnegative frequency-band energy measurements, outside the image domain. |
| [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) | Spam versus nonspam | 4,601 original records, 57 features | Word/character frequencies and capital-run measurements; the existing deduplicated preparation retains 4,204 records. |
| [WDBC](https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic) | Malignant versus benign source diagnosis | 569 records, 30 features | Nonnegative cell-nucleus measurements, with a different scale and dimension; used only as a benchmark. |

Selection depends on existing binary labels, nonnegative measurements, source
availability and variation in application domains. Every selected dataset will
be reported regardless of which method wins. ARCENE and DEXTER remain available
for a later high-dimensional stress test; their artificial probe features and
unverified source/batch grouping make them less straightforward first additions.
[JAFFE](https://zenodo.org/records/14974867) was considered as another face
source but has only ten expressers; FEI supplies substantially more subjects.

The seven methods are screened SPA-NMF, screened CD-NMF, screened PCA,
screening-only KNN, raw KNN, unscreened SPA-NMF and unscreened CD-NMF. Each
factorized representation is followed by KNN. The package defaults are unchanged.

## Splits and tuning

For each face dataset, fix a 70/30 split of whole subjects before fitting. Keep
subjects disjoint again in all five inner training/validation folds. Multiple
images and different resolutions of the same exposure are not independent
subjects. Use only the CMU quarter-resolution archive and the FEI manually
aligned frontal subset. FEI images are converted to grayscale and resized to
32 by 44 pixels; CMU images retain their original 32 by 30 quarter resolution.
Pixel values are divided by 255. No learned image rescaling is added.
CMU has 434 training images from 14 people and 190 reserved images from six
people; FEI has 280 training images from 140 people and 120 reserved images
from 60 people. The two source-native FEI size exceptions, `5a.jpg` and
`158a.jpg`, receive the same fixed 32 by 44 resize; no images are dropped.

The independent data audit reconstructs all 1,024 source images and 1,162,240
prepared feature values exactly, with 11,557 assertions and zero discrepancies.
Neither face cache contains an exact duplicate feature row. This verifies
exact equality, not near-duplicate detection or cross-dataset identity exclusion.

Retain the earlier prepared nonface development/reserved partitions: Sonar
145/63 rows, Spambase 2,942/1,262, and WDBC 398/171. The earlier search was
superseded before fitting; it produced no model or reserved-set comparisons.
Exact duplicate-feature handling and conflicting-label exclusions from that
preparation remain disclosed rather than silently changing this benchmark's
population. Original subject/document acquisition groups are unavailable for
these three caches; their row splits cannot establish source-group independence.

Use five-fold training-only calibration, the same initial ranks
`{2, 5, 10, 20, 40}`, screening cutoffs `{0, 2, 4, 8}`, and K values `1..101,
125, 151, 201`. Remove K values only when they exceed the smallest inner
training size and record those physical omissions before fitting. Retain
screen-dependent infeasible rank candidates as explicit failures. Apply the
same single coordinate-wise refinement and report unresolved boundaries.

Sonar already has features on the physical `[0,1]` scale. Spambase and WDBC
receive MaxAbs scaling fitted separately within each training fold and applied
equally to all seven pipelines. Test values are not used to choose scales,
cutoffs, ranks, K, seeds, solver caps or dataset inclusion.

CD uses NNDSVDa initialization, Frobenius loss, tolerance `1e-4`, iteration cap
20,000 and fixed seed 62006. SPA keeps its declared NNLS cap. Fit and projection
must qualify separately; failures remain visible and cannot be replaced by an
outcome-selected fallback. Freeze all completed selections before any reserved
model evaluation; an unqualified method remains an explicit missing result.

## Reporting scope

The report presents accuracy, total final-fit time and total prediction time
in three separate tables. Timings include preprocessing, exclude tuning and
equivalence refits, and use one native numerical thread. Tuning cost is retained
separately. Different selected dimensions and ranks make these pipeline workload
measurements, not an isolated solver-speed benchmark.

Save balanced accuracy, class recalls, confusion counts, AUC, average precision,
warnings, settings and predictions as supporting evidence. Label-independent
chance has expected balanced accuracy 50%; the constant majority reference is
chosen from training labels. For face uncertainty, resample whole test subjects,
not individual correlated images. Report conditional, descriptive intervals and
avoid treating CV folds or multiple methods as independent dataset replications.

FEI is research-only and its images must not be redistributed. The University
of Stirling provides an [authorized mirror](https://pics.stir.ac.uk/Other_face_databases.htm).
Source images stay in the ignored local data cache; neither report panels nor
face-image galleries are opened automatically.

## Implementation evidence

The new helper passes six focused tests, the runner nine, the image preparer
six, and the independent receipt auditor three, each in the primary Python
3.12 environment and the pinned Python 3.10 environment. These check explicit
group folds, pipeline/public-estimator equivalence, failure retention, frozen
input bindings, blocked repeat test access, grouped uncertainty, and rejection
of tampered receipts. They are additional study checks; the frozen package
implementation and the earlier full-suite evidence remain unchanged.

The protocol and machine-readable evidence are saved in
`runs/multi-dataset-validation-20261006/`. Source caches are under the ignored
`data/` directory. Reproduction uses `scripts/validate_multi_dataset.py` for
`prepare`, `freeze`, `tune`, `finalize`, and `validate`, with the independent
`scripts/audit_multi_dataset.py pretest` gate before `validate` and `posttest`
afterwards. Existing completed test access cannot be repeated in that run.
