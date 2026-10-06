# Overview

This benchmark measures sNMF fitting throughput under different worker and
numerical-thread counts. It includes Chang Cheng's sNMF source and uses each PC's
existing Python environment. Every test uses **CD with absolute cutoff 2.0**.
Extract the same ZIP on both PCs; sNMF installation and internet access are not
required.

The comparison includes three PCs:

| PC | Cores / threads | RAM | Result type |
|---|---|---|---|
| PC1: Ryzen 9 6900HS | 8 / 16 | About 39 GiB usable | Run on the PC |
| PC2: Ubuntu | Physical cores not supplied / 40 threads | About 503 GiB usable | Run on the PC |
| PC3: scaled PC1 | 16 / 32 | 96 GB nominal | **Projection only** |

PC3 assumes PC1's per-core performance and software. After the two real runs,
`compare.py` adds its projected column automatically. Ideal independent-fit
throughput is twice PC1's full 16-worker rate at the same per-fit size, provided
the RAM estimate fits. This does not predict the speed of one fit using 32 threads.
PC3 requires no separate benchmark run.

The completed `py12_df_aocl` PC1 sweep qualified all 11 configurations and 33
measurement windows. Those results describe its frozen bundle. Reuse with another
bundle requires the saved manifests to verify identical benchmark code, bundled
package, protocol and data. Reporting-only updates can qualify; a rebuild with
changed package files is not automatically compatible.

## Run

Open a terminal in the extracted folder with the intended environment activated.

**Windows PC:**

```text
.\RUN_SWEEP.cmd --label PC1
```

**Ubuntu PC:**

```sh
sh RUN_SWEEP.sh --label PC2
```

The runner checks compatibility automatically, then tests different thread counts.
It stops and saves the report if a check fails. Run the benchmark without
substantial competing jobs.

## What it tests

It starts with one worker and one thread. For each larger thread count, it compares
more independent workers with more threads inside one worker:

| Case | Workers | Threads per worker |
|---|---:|---:|
| Baseline | 1 | 1 |
| Two workers | 2 | 1 |
| Two threads in one worker | 1 | 2 |

It repeats that pair for larger counts up to the available CPU threads. The OS
schedules the work automatically. To run just the three cases above:

```text
python sweep.py --label PC1-basic --counts 1,2
```

Use `python3` and a `PC2-basic` label on Ubuntu.

The training-matrix size is chosen automatically from available RAM at the
maximum worker count, leaving room for the OS and temporary arrays. Each fit keeps
that same size throughout a PC's sweep. Cases with fewer workers use less total
RAM. The PC with more RAM can receive a larger workload.

Each case uses three 60-second measurement windows, plus setup and completion of
in-progress fits. Large workloads can take longer. If no fit completes inside a
window, retain the report and rerun both PCs with a longer common setting such as
`--seconds 300`.

## Results

Retain the **entire new folder under `results/`**, including `sweep-summary.md`,
`sweep.json`, `compatibility.json`, and all case subfolders. The summary shows
throughput and improvement over the one-worker, one-thread baseline.

The target's numerical-library versions passed local compatibility tests on
Windows/Python 3.12; Ubuntu/Python 3.10 is checked when it runs there. Different
workload sizes, operating systems, and library builds make this a comparison of
the complete PCs as configured, rather than a pure CPU speed ratio.

See [PROTOCOL.md](PROTOCOL.md) for technical details,
[VALIDATION.md](VALIDATION.md) for completed checks, and
[PROMPT_FOR_OTHER_PC.txt](PROMPT_FOR_OTHER_PC.txt) for Ubuntu run instructions.
