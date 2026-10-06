# Validation status and historical checks

Records through October 5, 2026. Author: Chang Cheng.

Environment: conda `py12_df_aocl`, Python 3.12.14, NumPy 2.5.2,
SciPy 1.18.0, scikit-learn 1.9.0, threadpoolctl 3.5.0, AOCL-BLAS 5.2.0,
Windows. Comparison execution explicitly limited numerical libraries to one
thread. These results do not establish compatibility with every allowed version.

## Current implementation checks

- After the public MU addition, `python scripts/check.py` passed **269 tests in
  58.743 seconds**. The log is `tmp/mu-public-full-check.log`. The wheel and source
  archive built successfully. An isolated wheel check verified SPA, CD and MU
  fit/predict, constructor defaults, explicit overrides, author metadata and
  archive contents; see `tmp/mu-public-wheel-check.log` and
  `tmp/mu-public-build.log`.
- A preceding full `python scripts/check.py` run passed **179 tests in 53.660
  seconds**. The log is `tmp/four-solver-full-check-final.log`. A subsequent
  matched-baseline reporting test brought the permanent test count to **180**;
  all **17 reporter tests passed in 0.623 seconds** after that addition. The
  180-test suite was not rerun at that point; later full-suite runs are recorded
  separately.
- The preceding full `python scripts/check.py` run passed **128 tests in 28.104
  seconds**, after the absolute-screening amendment and its 10 new pure decision
  tests. This record verifies implementation behavior at that revision.
- Historical `python scripts/check.py`: **65 tests passed** after withdrawal of the extra
  adaptive workflow and addition of explicit shared-representation grids.
  Coverage includes original
  float32 CD fidelity, pooled-t/F agreement, percentile ties, inclusive thresholds,
  exact/near-degenerate features, fold-local fitting, fixed-basis prediction,
  feature-anchor SPA on separable fixtures, serialization, warning/error records,
  label mapping, imbalanced metrics, official image partitions, and resume integrity.
  The 13 shared-grid tests also compare actual per-fold prediction arrays and
  numerical/support diagnostics against ordinary calibration, including tied
  distances; they test projection warnings and interrupted-fold reuse.
- Earlier `python examples/quickstart.py` check: executed successfully with explicit historical-
  style settings and a converged CD fit on a small generated dataset.
- Earlier `python examples/compare_simulation.py --quick --output runs --threads 1`:
  all **11 comparison methods** completed five-fold calibration and development
  validation without a fallback in this smoke run. Reports are under
  `runs/simulation-smoke-20261003T052810Z-47a76e20/` (ignored by Git).
- Original snapshots match both recorded SHA-256 hashes in `provenance/sources.json`.
- Local wheel and source archives build successfully. An isolated Python process
  imports directly from the wheel, verifies Chang Cheng's author metadata, and
  executes fit/predict without importing the checkout or installing into conda.
  Source-archive checks verify inclusion of original scripts, tests and examples.
  The current archives include the explicit-grid study module and exclude the
  withdrawn workflow modules, documentation and reference provenance.
- The CelebA CLI and loader pass synthetic-fixture tests and the real-data smoke
  checks described below.

The current suite and distribution checks apply to the explicit-grid
implementation. The additional adaptive-search workflow's historical test totals
are not part of the current count. Earlier scientific smoke runs below remain
separate records and are not substantive tuning results.

The paired smoke uses a single dataset, one candidate per method, and a small
development split. It tests functionality and audit records. Its accuracy and
timings cannot determine defaults, establish scientific superiority, reproduce
the historical experiment, or close Wang's empirical concerns. In particular,
the full repeated simulation study and numerical tuning/refinement remain open.

## Four-solver benchmark

Eight real Fashion-MNIST training-only smoke fits covered CD, SPA, MU and
ANLS-BPP at ranks 5 and 20. All eight passed without warnings; the record is
`runs/four-solvers-audit/training-smoke.json`. These checks measured no validation
accuracy and do not establish a predictive ranking.

The four-solver benchmark began under `runs/four-solvers-20261003/` and completed
under `runs/four-solvers-20261004-repair/` after the documented ANLS-BPP repair.
It compared the four solvers with absolute screening fixed on Fashion-MNIST and
a matched CelebA extension. The frozen rule selects SPA; the report remains
`selected_for_audit`. No new official-test run has started. The reporter includes
a separate Fashion-MNIST baseline table matched
at 8,000 training observations and seed 43, so its three-task baselines are not
mistaken for comparisons against NMF averages across all 18 configurations.

## Local CelebA readiness and execution checks

The local dataset contains 202,599 aligned JPG images, a CSV with 202,599 unique
attribute rows, and `list_eval_partition.txt`. Image filenames, attribute IDs,
and partition IDs match exactly. No missing/extra IDs, duplicate IDs, malformed
rows, or invalid Smiling/Wearing_Necktie labels were found. The partition file's
MD5 is `d32c9cbf5e040fd4025c592c306e6668`, matching the
[torchvision dataset manifest](https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py).
It defines 162,770 training, 19,867 validation, and 19,962 test images.

Twelve sampled images decoded successfully (178 by 218 RGB) and passed the
original grayscale/resize preprocessing. This is not a full corruption scan.
The preliminary decode audit sampled filenames before partitions were available;
it performed no fitting or evaluation. Detailed integrity reports and local
paths are saved under ignored `runs/celeba-data-audit/` and
`data/celeba-metadata/source.json`. Dataset files remain outside distribution
archives. Identity annotations were not supplied; the identity audit remains open.

For each attribute, the real-data smoke uses seed 42, a uniform sample of 500
official training images and 200 official validation images, 64 by 64 grayscale,
one candidate per method, and five-fold calibration within the training sample.
No official test images are loaded by these model comparisons. Both attributes
use the same sampled image IDs. No class balancing or 20% allocation was applied.

The initial `max_iter=400`, `tol=1e-4` runs rejected all three CD candidates
(absolute screening, percentile screening, and unscreened NMF) because at least
one calibration fold hit the iteration cap. Their flagged majority fallbacks
are not CD performance results. Those diagnostics are retained in:

- `runs/celeba-smiling-development-20261003T073943Z-2a17afd4/`
- `runs/celeba-wearing_necktie-development-20261003T073943Z-01c0edca/`

Increasing only the iteration cap to 2,000, while retaining tolerance, data,
seeds, folds and method, allowed all 11 methods for each attribute to finish with
valid validation results and no fallback. CD training fits used at most 834
iterations across calibration and refitting. Successful runs:

- `runs/celeba-smiling-development-20261003T074137Z-4b06209a/`
- `runs/celeba-wearing_necktie-development-20261003T074137Z-16d94dee/`

This is a recorded numerical run setting, not a change to package defaults or
the dissertation's factorization method. The two attribute processes ran
concurrently with one numerical thread each; these timings are not isolated
performance benchmarks. These small, single-seed checks establish ingestion and
execution only. They do not complete tuning, select defaults, establish global
optimality, or resolve Wang's empirical concerns.

## Explicit-grid CLI fixture and resume

The `examples/celeba_parameter_study.py` CLI completed a synthetic fixture with
50 training rows, 30 validation rows and 4,096 nonnegative float32 features.
Both attribute names used generated binary labels and distinct fake image IDs;
no CelebA images were read. The declared grid was rank 2, percentile 5, absolute
cutoff 1 and KNN neighbors 1 and 3, with `max_iter=2000`, `tol=1e-4`, two workers
and one numerical thread per worker.

Each of the two attributes completed all 11 comparison methods. The 40 tuned
candidates produced 200 valid fold records from 100 unique fold representation
fits, reusing each representation across the two neighbor counts. All 22
development evaluations completed. An audit confirmed exact saved five-fold
assignments, disjoint fake training/validation IDs, valid checksums and bindings
to each method, attribute, run and selected candidate.

A real CLI resume reused all 100 representation folds with zero new
representation fits. All 100 fold receipts, 100 attempt receipts and 22 final
development-evaluation receipts remained byte-identical; no new files appeared.
Only the 20 CV reports and two run summaries changed to record reuse and current
call timing. These checks covered 289 JSON files.

The ignored local run is
`tmp/cli/celeba-parameter-study-20261003T085644Z-6d6343cf/`; audit scripts and
records are under `tmp/celeba-parameter-study-fixture-20261003/`. A preceding
attempt with a deeply nested output directory exceeded this Windows host's path
limit at a 283-character checkpoint filename. The successful check used the
shorter output root above. Fixture results establish execution and reuse only;
their scores have no interpretation as CelebA predictive performance.

## Separate profiler and reporter fixture checks

These execution checks were separate from the historical 65-test suite above. The stage
profiler was rerun against the completed synthetic CLI fixture and its exact
cached arrays, using one process and one native numerical thread. All 22 method
and attribute combinations reproduced the saved validation labels and class
probabilities exactly. A regression check that performs no fitting verified
estimator-clone isolation across attributes; the completed matched-prediction
run additionally confirmed the behavior during actual fits. The profiling record
is retained at
`tmp/profile-fixture-checks/celeba-stage-profile-20261003T103852Z-f2fea472/profiling.json`.

Reporter fixture checks passed for recomputed metric integrity, generated tables,
PNG visual inspection, and attachment of execution-transition and separate
profiling evidence. Reporting uses saved predictions and does not tune or fit
models. The profiler's additional fit costs remain separate from the original
study costs.

These synthetic checks ran while other work was active. Their timings establish
instrumentation and execution correctness, not isolated performance benchmarks.
They provide no real CelebA performance evidence, scientific replication,
completed-tuning claim or package-default decision.

## Withdrawn workflow records

The additional adaptive-search workflow was withdrawn. Its simulation run
`runs/simulation-smoke-20261003T082622Z-15fdf082/` and small CelebA run
`runs/celeba-smiling-development-20261003T082658Z-5e0a0b2c/` remain ignored local
historical records. Their bounded execution and resume checks neither validate
the current workflow nor provide a substantive method comparison.

The complete preceding validation record, workflow documentation and reference
metadata were copied with matching SHA-256 hashes into ignored
`tmp/withdrawn-adaptive-workflow/` before the current documentation was revised.
They are not included in package distributions or required for current studies.

## First development study

The first explicit-grid CelebA comparison completed on 4,000 official training
and 4,000 official validation images at seed 42. Five-fold training-only selection
covered 4,088 declared candidates, with 2,920 shared representation-fold attempts
across both attributes. All 124 chunks completed and all 22 selected
method/attribute models produced valid validation evaluations. Infeasible and
nonconverged candidates remain recorded and were excluded from selection.

The completed local run is
`runs/celeba-parameter-study-20261003T091802Z-68d2aa40/`. Its mixed four-to-eight
worker execution, data allocation and numerical settings are documented in
[the CelebA study](celeba-study.md). Its training-only review verified the complete
grid and identified outstanding boundary and numerical qualification questions.

Additional serial timing fits, using one process and one native numerical thread,
reproduced all 22 saved validation-label and probability arrays exactly. These
are extra fits at the selected settings, not scientific replications or replacements
for original tuning/final-fit costs. Their record is
`profiles/celeba-stage-profile-20261003T114136Z-3c25da37/profiling.json` within that run.

This is one development comparison. It does not complete tuning, select package
defaults, establish a statistically reliable method ranking or evaluate the
official test set.

Observed metrics and timing examples are summarized in
[the first development results](celeba-initial-results.md). The completed-run
integrity audit passed: exact folds, all candidates, receipt/configuration
bindings, official training/validation membership and all saved validation
metrics were checked independently.

## Repeated default study and disclosed amendment

The training-only refinement and repeated four-combination comparison completed
for three training sizes (2,000, 4,000 and 8,000), seeds 43, 44 and 45, and both
attributes. All 72 selected development evaluations were valid. All 216 serial
profiling repeats reproduced their original predictions and probabilities
exactly, without qualifying warnings. All 2,050 feasible five-fold candidates
were valid; 110 candidates were excluded before fitting by documented dimensional
feasibility checks. The CV computation used 1,025 shared representation-fold fits.

The completed repeated run is `runs/default-selection/repeated/`. Its original
decision, `runs/default-selection/decision/decision.json`, selected SPA with
percentile screening. After the development results, the screening preference was
amended to favor absolute screening for supervised-PCA alignment when it passes
the original accuracy gate.
The separate amendment at
`runs/default-selection/decision-spca-preference/decision.json` selects SPA with
absolute screening. The subsequent four-solver comparison also selected SPA with
absolute screening; its status remains `selected_for_audit`. Original evidence
and both CelebA decisions are preserved.
The amendment helper revalidates development receipts, cases and original decision
arithmetic; it does not fit models, change source defaults or read test outcomes.

The repeated CV execution took 319.674 seconds with eight workers, followed by
530.846 seconds for selected evaluations and serial profiling with one worker.
Each worker used one native numerical thread. Per-method CV sums are worker-wall
compute; serial case costs use medians of per-repeat fit-plus-prediction sums.
Nested component timings must not be added to total fit/prediction. Peak memory
for the full process tree was not measured.

All methods shared the same 4,000 validation images and overlapping training
allocations. These are descriptive development comparisons, not independent
datasets or proof of predictive equivalence. The theoretical review distinguishes
solver-neutral conditional representation bounds from the stronger assumptions
needed for screening retention, anchor recovery and sample-split KNN rates.

The original final-test attempt was stopped after eight development revalidations
and after test loading began. The chronology audit recorded no persisted test
cache, test prediction attempts, evaluations or comparison. Partial in-memory
decoding is unknown. The original attempt remains preserved, and no test outcomes
were inspected or used for the amendment. No new final-test run has started;
final evaluation requires a completed evidence audit and a new protocol freeze.
See [the results](celeba-default-results.md) and
[theory/implementation limits](theory-implementation.md).

## Four-solver repair and completion

The October 4 four-solver run encountered a reproducible native access violation
in the high-level SciPy least-squares dispatcher used by experimental ANLS-BPP.
The [numerical-repair record](numerical-repair-20261004.md) documents the retained
original run, captured failing matrices, direct-DGELSD workaround, unchanged
NMF objective/tolerances, and versioned checkpoint recovery. All 17 numerical
tests and 12 recovery tests passed. The complete suite at that revision, including the
existing portable-workflow tests, passed **266 tests in 61.417 seconds**;
see `tmp/four-solver-repair-full-check.log`. The repaired path also passed the
captured matrices and a two-sweep training-only integration with 5,035 direct
least-squares calls and zero inner failures. That deliberately short integration
is not a production-convergence claim. Crash dialogs are disabled only for test
and benchmark processes; crash logging remains enabled.

The successor is `runs/four-solvers-20261004-repair`. Verified unchanged-method
receipts retain their original bytes and timing. Every ANLS-BPP chunk was rerun;
old ANLS-BPP results remain historical evidence.

Recovery completed with 2,033 complete reports and two committed partial folds
reused; no ANLS-BPP receipts were imported. Production restarted at
2026-10-04 10:19:34 UTC with eight single-thread numerical workers and 926
Fashion-MNIST chunks pending. The first repaired ANLS-BPP chunks completed
successfully; the subsequent completion is recorded below.

The repaired comparison completed at 2026-10-05 10:37:02 UTC (6:37 a.m.
Eastern). All 2,959 Fashion-MNIST and 292 CelebA CV chunks were processed.
All 89 selected models qualified on development validation, each with three
valid timing repetitions. The generated [report](nmf-solver-comparison.md#development-results)
selects SPA with absolute screening under the frozen rule. SPA alone passes all accuracy gates;
its combined balanced accuracy is 80.7146%. Its weighted geometric deployment
cost is essentially tied with MU, so a speed advantage over MU does not explain
the choice. Matched Fashion-MNIST PCA, screening-only and raw-KNN baselines
outperform the NMF methods in mean balanced accuracy. Dataset/task omission
and workload sensitivities limit the scope of this recommendation. The report
remains `selected_for_audit`; official test data were not evaluated.

## Pending scientific and release work

- Final evidence audit and release documentation for the completed
  CD/SPA/MU/ANLS-BPP comparison with absolute screening fixed, before freezing
  the general solver default for official-test evaluation.
- Historical 32.42% baseline reproduction and controlled causal diagnosis.
- Full harder-simulation study and broader independent replication; the scoped
  CelebA refinement/default comparison is complete.
- Wider independently tuned PCA/screening-only comparisons beyond the first
  starting grid, identity audit, and frozen final-test evaluation.
- Full memory/scaling measurements and dissertation-ready figures/tables.
- Broader compatibility verification, author-selected distribution license, and
  any publication to GitHub or a package index.
