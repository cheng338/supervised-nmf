# sNMF benchmark protocol

## Work and thread counts

The sweep begins with one worker using one native numerical thread. At each
larger thread budget N it runs N independent workers with one thread each, followed
by one worker with N native numerical threads. This separates aggregate fitting
throughput from the benefit, if any, of multithreading a single fit. The OS
schedules workers and threads without fixed-core pinning. Requested thread budgets
cannot exceed the available logical CPUs.

Each fit uses the same matrix size throughout a PC's sweep. Planning uses that
PC's maximum available worker count, even if `--counts` requests fewer cases. The
maximum-worker case is the RAM capacity measurement; cases with fewer simultaneous
workers naturally use less total RAM. Different PCs may use different matrix
sizes because their available RAM differs.

## Scientific settings

`protocol.json` explicitly specifies coordinate descent (`solver="cd"`), absolute
screening (`abs(t) >= 2.0`), NMF rank 12, Euclidean 5-neighbor classification,
`max_iter=20000`, `tol=1e-4`, `random_state=43`, and
`training_coordinates="fit_transform"`. CD uses NNDSVDa initialization and
Frobenius loss. These are fixed benchmark parameters, not a claim of optimal
predictive accuracy. The source package's default solver is irrelevant because
the benchmark explicitly selects CD.

Capacity data follow the original simulation distribution with 2,048 features,
signal ratio 0.25, and 256 prediction rows. Generation is deterministic and
chunked; its random draw order is not identical to calling the original generator
at the full large shape. Training observations alone determine feature screening
and the fitted basis. Test data are projected through that fitted representation.

The separate frozen validation fixture has 768 training rows, 256 held-out rows,
and 512 features, generated with the package's original simulation function and a
balanced deterministic split. The optional `--mode matched` uses this fixed data
instead of RAM-scaled data. It is a diagnostic equal-work baseline.

## RAM and timing

Capacity planning reserves `max(4 GiB, 10% of total physical RAM)` for the OS and
other processes, then budgets 90% of the remaining available physical RAM by
default. The estimate accounts for worker overhead and temporary arrays. It grows
useful training data rather than allocating unused padding. Swap is not treated
as extra benchmark RAM.

The budget is an estimate, not a promise of exact resident memory occupancy.
Screening, numerical-library builds, temporary arrays, and allocator behavior
change actual memory use. Reports retain estimates and sampled process-tree and
system memory/CPU usage; sampling can miss brief peaks. Concurrent jobs alter both
the available-memory plan and measured performance.

Each case defaults to three 60-second measurement windows. Data generation,
source loading and warm-up are outside the timed window. Tasks finishing after
the common deadline are excluded from its completion rate. A separate batch rate
including the completion period is explicitly labeled; it does not replace the
window rate. A window with no completed fits is unqualified. Use a longer common
`--seconds` setting on both PCs if needed.

Startup and completion timeouts default to 1,800 seconds each. Very large problems
may need longer explicit timeouts. `--train-rows` and `--memory-fraction` are
recorded overrides for smaller validation workloads; these must not be described
as the default full-capacity result. Use `--help` to inspect available arguments.

## Compatibility and provenance

The reference Ubuntu environment is Ubuntu 22.04.5, 40 online logical CPUs,
503 GiB total / 496 GiB available RAM, Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.0, threadpoolctl 3.6.0 and joblib 1.5.1. Its CPU model and physical
core count were not recorded. PC1 has a Ryzen 9 6900HS,
8 physical cores / 16 logical processors and approximately 39.23 GiB usable RAM.
Execution-time environment and resource records govern result interpretation.

The package metadata still declares Python >=3.11. Direct use of bundled source
under the target's Python 3.10 is provisional until its own checks pass. Matching
target numerical-package versions were tested in an isolated Windows/Python 3.12
environment; this does not validate Ubuntu's interpreter or binary builds.
`VALIDATION.md` gives exact coverage and exclusions.

The sweep runs `check_compatibility.py` automatically. It verifies the bundle,
imports only bundled sNMF source, compares absolute screening against SciPy's
pooled t statistic including the inclusive cutoff, and checks converged,
reproducible float32/float64 CD fit and held-out prediction without refitting.
For a separate check requiring the exact supplied target versions, run
`python3 check_compatibility.py --expect-target-versions` on Ubuntu.

No Python runtime or dependency binaries are bundled. Use the existing intended
environment; the launchers invoke `python` on Windows and `python3` on Linux.
`requirements-reference.txt` records the builder's versions, not a cross-platform
installation lockfile. It cannot reproduce AOCL/OpenBLAS binary builds. Optional
`psutil` is unnecessary when native Windows/Linux resource probes are available.

The current `supervised_nmf` Python source, package metadata, documentation,
citation and original research provenance are included under `package/`.
New snapshots also include `package/LICENSE` and `package/THIRD_PARTY_NOTICES`,
preserving the project's rights notice and separate third-party terms.
The runner imports that source directly instead of an installed sNMF distribution.
Chang Cheng's authorship is preserved in `package/CITATION.cff` and provenance.

`manifest.json` hashes every distributed file except itself. The runner verifies
it before fitting. The builder snapshots input hashes, verifies copies and the
complete ZIP, and refuses to overwrite an existing destination. Results are
outside the frozen manifest. These hashes detect changes/corruption; they are
not digital signatures. To rebuild from settled repository files:

```text
python portable_speedtest/build_bundle.py
```

## Interpretation

The participants are registered in `pc_profiles.json`: PC1 is the physical
8-core/16-thread Ryzen PC; PC2 is the physical 40-thread Ubuntu PC; PC3 is a
**hypothetical** 16-core/32-thread PC with 96 GB RAM. Nominal PC3 RAM is represented
as 96 GiB; usable memory has not been measured. PC3 retains PC1's assumed CPU
generation, per-core speed, operating system and numerical environment.

The comparison adds PC3 automatically from the first (PC1) report. Its ideal
32-worker/one-thread throughput is 2 times a qualified, full 16-worker/one-thread
PC1 result, keeping per-fit dimensions fixed. The estimated 32-worker footprint
must fit PC3's memory budget after the OS reserve and PC1's observed non-benchmark
memory use. Its extra RAM is capacity headroom, not a speed multiplier. No larger
dataset timing, single-fit/32-thread speed, CPU utilization or actual RAM usage
is inferred. Missing or invalid baseline evidence leaves projected speed pending.
Projection metadata and metrics are saved separately from measured results.

Within a PC, fixed per-fit data permit comparison to its one-worker/one-thread
baseline. Inspect convergence, iterations, selected features, output checks,
thread pools, CPU use and memory alongside throughput. More numerical threads
inside one CD fit may improve or reduce performance.

Across PCs, fits/second on different matrix sizes represent different work.
Training input values/second also depends on dimensions, screening and iteration
counts; its ratio is workload-dependent system throughput, not pure CPU speedup.
OS, Python versions and BLAS/library differences further qualify the comparison.

To generate the three-PC comparison from the two complete real sweep folders:

```text
python compare.py results/PC1-sweep-folder results/PC2-sweep-folder --output comparisons/three-pcs
```

Use actual generated folder names, a new output directory, and `python3` on Linux
when that is the active environment command. Preserve all case reports, including
unqualified results, in the complete result folders.
The first input must be the 8-core/16-thread PC. PC3 is added as a separate
projection; no third input folder or third physical run is needed.

The completed PC1 `py12_df_aocl` sweep from October 4, 2026 remains valid for its
frozen bundle. Its original manifest is preserved byte-for-byte under
`compatible_manifests/`. If report manifest hashes differ, `compare.py` verifies
both saved manifests by their SHA-256 hashes and requires every non-reporting file
hash to agree, including runners, launchers, data, environment references and the
entire bundled package. Only comparison/projection files, PC profiles, root
documentation, build provenance and archived manifest receipts may differ.
Unknown manifests or benchmark changes block comparison. Edits to bundled
package documentation, metadata or source also change the required package
identity, even when benchmark settings remain fixed. Preserve the original
result files and their recorded identities; a new build does not retroactively
validate earlier results.
