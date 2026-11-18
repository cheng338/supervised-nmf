# Three-label refinement reuse contract

The original study workspace retains the refinement bridge
(`scripts/three_label/refinement.py`). It reuses
recorded artifacts for the explicitly approved local-refinement amendment.
It retains the study's exact data, fold, configuration, source and runtime
checks. This contract does not qualify a historical run for a different
environment or change its recorded search policy.

## Immutable original evidence

Preparation records the original JSON evidence in a checksummed manifest.
Before activating reused artifacts, the bridge verifies both the recorded
presence and the bytes of `resource-ledger.json`:

- A recorded ledger must still exist and match its original hash.
- If no ledger was recorded, it must remain absent. Legitimate no-ledger
  searches continue to use an empty original operation map.

A missing, changed or newly appearing ledger stops activation before it writes
additional receipts or payloads. Previously prepared or completed artifacts
remain available for inspection; rejection does not roll them back. Preserve
the original evidence and the failed attempt rather than modifying its manifest
or discarding recorded failures to force reuse.

The amended search retains historical compute charges and failed-operation
evidence. Reusing an artifact does not erase its original charge or count it as
new fitting. The final audit continues to check the original evidence and the
search's qualification rules independently.

Treat original and reused numerical payloads as immutable: some payloads share
storage through hard links, so modifying either path can modify both. This
bridge is not a transaction over the entire study and does not protect against
arbitrary concurrent changes to historical files. Keep historical records
unchanged and use a separate output directory for changed computations.

## Declared reporting scope

The current three-label reporter validates the declarations it consumes before
writing report outputs. An explicit `planned_records` schedule determines the
report's procedure scope, labels and planned repeat counts, including a smaller
amended schedule than the original design. It does not infer a smaller plan
from successful results. An explicitly empty schedule remains empty. Malformed
consumed declarations, unsupported procedure identities and duplicate normalized
plan entries are rejected; duplicate observations remain invalid observations.

If the schedule is absent, the reporter can reconstruct descriptive scope.
Reconstructed scope or missing bootstrap-count, bootstrap-seed or
primary-contrast count declarations make the report provisional, even if the input supplies
`audited: true`. Saved-prediction metrics remain available, but affirmative
planned-study inference is withheld. Existing study/record audit requirements
and the gate on a bootstrap count differing from the design still apply.

JSON, CSV and text reports disclose scope provenance and inferred fields.
Machine-readable metadata also records derived labels and the supplied audit
flag, while preserving the original design. Reporting recomputes metrics from
saved predictions; it does not refit models, independently re-audit model or
cohort artifacts, or qualify historical executions for a different runtime.
