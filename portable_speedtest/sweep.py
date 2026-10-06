"""Compare N workers with one thread against one worker with N threads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import benchmark


ROOT = Path(__file__).resolve().parent


def thread_counts(maximum, physical=None, requested=None):
    if requested:
        values = {int(part) for part in requested.split(",")}
        if not values or min(values) < 1 or max(values) > maximum:
            raise ValueError("Counts must lie between one and the available logical CPU count.")
    else:
        values = {value for value in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128)
                  if value <= maximum}
        values.add(maximum)
    values.add(1)
    return sorted(values)


def stages(counts):
    result = [{"id": "workers-001", "kind": "throughput", "workers": 1,
               "native_threads": 1, "count": 1}]
    for count in sorted(set(counts) - {1}):
        result.extend([
            {"id": f"workers-{count:03d}", "kind": "throughput", "workers": count,
             "native_threads": 1, "count": count},
            {"id": f"threads-{count:03d}", "kind": "single-fit", "workers": 1,
             "native_threads": count, "count": count},
        ])
    return result


def write_summary(output, report):
    baseline = {}
    for item in report["runs"]:
        if item["status"] == "qualified" and item["count"] == 1:
            baseline[item["kind"]] = item
    lines = ["sNMF CPU-count sweep", "", f"PC: {report['label']}",
             f"Training shape per fit: {report['n_train']} x {report['n_features']}",
             "Per-fit size is fixed across this PC's sweep; maximum-worker RAM sizing sets it.", "",
             "| Test | Workers | Native threads/worker | Qualified fits/s | Median fit+predict (s) | Gain vs 1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in report["runs"]:
        rate, latency = item.get("fits_per_second"), item.get("median_fit_prediction_seconds")
        base = baseline.get("throughput")
        gain = None
        if base and item["status"] == "qualified":
            if rate and base.get("fits_per_second"):
                gain = rate / base["fits_per_second"]
        item["within_pc_gain_vs_one"] = gain
        def fmt(value):
            return "unqualified" if value is None else f"{value:.4g}"
        lines.append(f"| {item['kind']} | {item['workers']} | {item['native_threads']} | "
                     f"{fmt(rate)} | {fmt(latency)} | {fmt(gain)} |")
    lines += ["", "Gains use total fitting throughput relative to one worker with one numerical thread. "
              "CPU scheduling is automatic, with no fixed-core affinity. Each thread count compares "
              "N workers x 1 thread with 1 worker x N threads, after the 1 x 1 baseline.",
              "Lower worker counts use less total RAM. The maximum-worker stage is the RAM-scaled "
              "capacity measurement. Across PCs, different matrix sizes or software require the "
              "capacity qualifications in README.md.", ""]
    benchmark.json_write(output / "sweep.json", report)
    (output / "sweep-summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--counts", help="Optional comma-separated counts, e.g. 1,4,8,16")
    parser.add_argument("--mode", choices=("capacity", "matched"), default="capacity")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--train-rows", type=int, help="Explicit reduced-size validation run; recorded as an override")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--drain-timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    protocol, manifest_hash, protocol_hash = benchmark.verify_bundle()
    machine = benchmark.hardware()
    maximum = machine["logical_cpus"] or 1
    counts = thread_counts(maximum, machine.get("physical_cores"), args.counts)
    # Plan once at the true machine maximum, even for a reduced-count sweep.
    plan_args = argparse.Namespace(smoke=False, workers="auto", native_threads=1, mode=args.mode,
                                  seconds=args.seconds, repeats=args.repeats, memory_fraction=args.memory_fraction,
                                  train_rows=args.train_rows)
    plan = benchmark.describe_configuration(plan_args, protocol, machine)
    if not plan["auto_memory_guard_passed"]:
        raise ValueError("The maximum-worker workload does not fit the current RAM budget.")
    work = stages(counts)
    description = {"hardware": machine, "per_fit_configuration": plan, "counts": counts,
                   "execution_order": work, "memory_policy": "Per-fit size frozen from maximum-worker plan; lower counts use less RAM."}
    if args.describe:
        print(json.dumps(description, indent=2))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", args.label).strip("-")[:60] or "pc"
    output = args.output.resolve() if args.output else ROOT / "results" / f"{timestamp}-{slug}-sweep"
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "label": args.label, "status": "running", "manifest_sha256": manifest_hash,
              "protocol_sha256": protocol_hash, "n_train": plan["n_train"], "n_features": plan["n_features"],
              "description": description, "runs": [], "configuration_overrides": vars(args).copy()}
    report["configuration_overrides"]["output"] = str(output)
    benchmark.json_write(output / "sweep.json", report)
    print(f"Sweep output: {output}", flush=True)
    print(f"Counts: {counts}; {len(work)} settings; {plan['duration_seconds']:g}s x {plan['repeats']} windows each, plus warm-up/drain.", flush=True)
    # Compatibility is deliberately done once, before spending time or memory on the sweep.
    compatibility = subprocess.run([sys.executable, str(ROOT / "check_compatibility.py"),
                                    "--output", str(output / "compatibility.json")])
    if compatibility.returncode:
        report["status"] = "compatibility_failed"
        write_summary(output, report)
        return compatibility.returncode
    for stage in work:
        destination = output / stage["id"]
        command = [sys.executable, str(ROOT / "benchmark.py"), "--label", args.label + "-" + stage["id"],
                   "--mode", args.mode, "--workers", str(stage["workers"]), "--seconds", str(plan["duration_seconds"]),
                   "--repeats", str(plan["repeats"]), "--memory-fraction", str(args.memory_fraction),
                   "--startup-timeout", str(args.startup_timeout), "--drain-timeout", str(args.drain_timeout),
                   "--output", str(destination)]
        if args.mode == "capacity":
            command += ["--train-rows", str(plan["n_train"])]
        if stage["native_threads"] != 1:
            command += ["--native-threads", str(stage["native_threads"])]
        print(f"Starting {stage['id']}", flush=True)
        log_path = output / (stage["id"] + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        stage_result = {**stage, "returncode": process.returncode, "result_directory": str(destination),
                        "status": "failed", "command": command}
        child_path = destination / "report.json"
        if child_path.is_file():
            child = json.loads(child_path.read_text(encoding="utf-8"))
            stage_result["status"] = child.get("status", "failed")
            stage_result["fits_per_second"] = child.get("aggregate", {}).get("qualified_fits_per_second_median")
            latencies = [r["timing"]["fit_plus_prediction_seconds"] for rep in child.get("repeats", [])
                         for r in rep["records"] if r["status"] == "qualified" and r["completed_within_window"]]
            stage_result["median_fit_prediction_seconds"] = statistics.median(latencies) if latencies else None
        report["runs"].append(stage_result)
        if process.returncode or stage_result["status"] != "qualified":
            report["status"] = "failed"
            write_summary(output, report)
            print(f"Test failed. Details: {log_path}", flush=True)
            return process.returncode or 2
        print(f"  passed: {stage['workers']} worker(s) x {stage['native_threads']} thread(s)", flush=True)
        write_summary(output, report)
    report["status"] = "qualified"
    write_summary(output, report)
    print((output / "sweep-summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
