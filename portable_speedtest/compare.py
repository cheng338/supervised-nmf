"""Compare two measured PCs and a projected 16-core/32-thread, 96 GB third PC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def load_report(path):
    path = Path(path)
    sweep = None
    if path.is_dir():
        if (path / "sweep.json").is_file():
            sweep = json.loads((path / "sweep.json").read_text(encoding="utf-8"))
            runs = sweep.get("runs", [])
            if not runs:
                raise ValueError("Sweep has no completed stages: " + str(path))
            selected = max(runs, key=lambda item: item["workers"])
            if Path(selected["id"]).name != selected["id"]:
                raise ValueError("Invalid sweep stage path")
            path = path / selected["id"]
        path = path / "report.json"
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported report schema: " + str(path))
    for key in ("configuration", "aggregate", "provenance", "label"):
        if key not in result:
            raise ValueError("Missing report field " + key + ": " + str(path))
    if sweep is not None:
        result["sweep_context"] = sweep
        result["label"] = sweep["label"]
        if sweep.get("status") != "qualified":
            result["status"] = "incomplete-sweep"
        if (sweep.get("manifest_sha256") != result.get("manifest_sha256")
                or sweep.get("protocol_sha256") != result.get("protocol_sha256")):
            raise ValueError("Sweep identity does not match its maximum-worker report.")
    return result


def ratio(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
            return b / a
    return None


def compatible_benchmark_manifests(first_hash, second_hash, bundle_root=None):
    """Allow reporting-only updates when both complete manifests are verifiable."""
    root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parent
    reporting_only = {
        "compare.py", "projection.py", "pc_profiles.json", "builder-provenance.json",
        "README.md", "PROTOCOL.md", "PROMPT_FOR_OTHER_PC.txt", "VALIDATION.md",
    }
    required = {
        "benchmark.py", "sweep.py", "check_compatibility.py", "protocol.json",
        "data/workload.npz", "package/supervised_nmf/__init__.py",
    }

    def benchmark_files(digest):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            return None
        candidates = (root / "manifest.json", root / "compatible_manifests" / (digest + ".json"))
        for path in candidates:
            try:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest:
                    continue
                manifest = json.loads(raw)
            except (OSError, ValueError):
                continue
            if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                    or manifest.get("hash_algorithm") != "sha256"):
                continue
            files = manifest.get("files")
            if not isinstance(files, dict) or not required.issubset(files):
                continue
            if any(not isinstance(name, str) or not isinstance(value, str)
                   or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                   for name, value in files.items()):
                continue
            return {name: value for name, value in files.items()
                    if name not in reporting_only and not name.startswith("compatible_manifests/")}
        return None

    left, right = benchmark_files(first_hash), benchmark_files(second_hash)
    return left is not None and right is not None and left == right


def assess(first, second, *, bundle_root=None):
    """Keep comparability decisions separate from presentation for auditability."""
    blockers, qualifications = [], []
    left, right = first["configuration"], second["configuration"]
    first_hash, second_hash = first.get("manifest_sha256"), second.get("manifest_sha256")
    if not first_hash or first_hash != second_hash:
        if compatible_benchmark_manifests(first_hash, second_hash, bundle_root):
            qualifications.append(
                "Bundle identities differ only in reporting, profiles, documentation or build provenance; "
                "the saved manifests verify identical benchmark code, sNMF source, protocol and data."
            )
        else:
            blockers.append("The bundle/protocol identity differs or is missing: manifest_sha256")
    if not first.get("protocol_sha256") or first.get("protocol_sha256") != second.get("protocol_sha256"):
        blockers.append("The bundle/protocol identity differs or is missing: protocol_sha256")
    for key in ("mode", "duration_seconds", "repeats", "model"):
        if key not in left or left.get(key) != right.get(key):
            blockers.append("Run settings differ or are missing: " + key)
    shape_keys = ("n_train", "n_test", "n_features")
    same_shape = all(k in left and left.get(k) == right.get(k) for k in shape_keys)
    if left.get("mode") == "matched" and not same_shape:
        blockers.append("Matched-work mode requires identical dataset dimensions.")
    if left.get("mode") == "capacity":
        qualifications.append(
            "Capacity mode sizes training data to each system. Input values/s is a "
            "descriptive throughput measure; NMF iteration counts, selected features, "
            "and matrix dimensions affect the amount of arithmetic per input value."
        )
    if left.get("native_threads_per_worker", 1) != right.get("native_threads_per_worker", 1):
        qualifications.append("Numerical threads per worker differ; this compares different parallel allocations.")
    if not same_shape:
        qualifications.append(
            "Dataset dimensions differ. Fits/s and per-fit times do not establish "
            "an equal-work speedup between these systems."
        )
    for report in (first, second):
        if report.get("status") != "qualified":
            blockers.append(str(report["label"]) + " did not finish successfully.")
        if report.get("smoke") or report["configuration"].get("smoke"):
            blockers.append(str(report["label"]) + " is only a smoke/compatibility check.")
        rate = report["aggregate"].get("qualified_fits_per_second_median")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            blockers.append(str(report["label"]) + " has no positive qualified in-window throughput.")
        repeats = report.get("repeats", [])
        if not repeats or len(repeats) != report["configuration"].get("repeats"):
            blockers.append(str(report["label"]) + " has missing measurement windows.")
        for repeat in repeats:
            if repeat.get("status") != "qualified":
                blockers.append(str(report["label"]) + " contains an unqualified measurement window.")
            records = repeat.get("records", [])
            rejected = [r for r in records if not r.get("qualified", r.get("status") == "qualified")]
            if rejected:
                qualifications.append(
                    str(report["label"]) + " has unqualified records; inspect convergence, "
                    "numerical agreement and warnings before interpreting its throughput."
                )
    p1, p2 = first["provenance"], second["provenance"]
    if left.get("mode") == "matched":
        def reference(report):
            windows = report.get("repeats", [])
            ready = windows[0].get("worker_readiness", []) if windows else []
            return ready[0].get("warmup", {}).get("quality", {}) if ready else {}
        ref1, ref2 = reference(first), reference(second)
        for key in ("selected_mask_sha256", "labels_sha256", "probabilities_sha256", "n_iter"):
            if ref1.get(key) != ref2.get(key):
                qualifications.append("Matched-data numerical results differ: " + key
                                      + ". Review the saved reference arrays and solver diagnostics.")
    differences = []
    for key in ("python", "versions", "platform"):
        if p1.get(key) != p2.get(key):
            differences.append(key)
    # Paths and selected CPU architecture may differ even with the same build.
    def libraries(provenance):
        pools = provenance.get("blas", provenance.get("thread_pools", []))
        return sorted((str(p.get("internal_api")), str(p.get("prefix")),
                       str(p.get("version")), str(p.get("threading_layer"))) for p in pools)
    if libraries(p1) != libraries(p2):
        differences.append("numerical libraries")
    if differences:
        qualifications.append(
            "Software differs (" + ", ".join(differences) + "). This compares the "
            "complete systems and environments; it does not isolate CPU hardware."
        )
    checks = {}
    if not blockers:
        checks["input_values_rate_ratio_second_over_first"] = ratio(
            first["aggregate"].get("qualified_input_values_per_second_median"),
            second["aggregate"].get("qualified_input_values_per_second_median"),
        )
        # Capacity-generated data can differ with worker count even at equal size.
        if same_shape and left.get("mode") == "matched":
            checks["matched_fit_rate_ratio_second_over_first"] = ratio(
                first["aggregate"].get("qualified_fits_per_second_median"),
                second["aggregate"].get("qualified_fits_per_second_median"),
            )
    return {"comparable_protocol": not blockers, "same_shape": same_shape,
            "blockers": blockers, "qualifications": list(dict.fromkeys(qualifications)),
            "ratios": checks}


def fmt(value):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return format(value, ".4g")
    return str(value)


def render(first, second, decision, projected=None):
    def escape(value):
        return fmt(value).replace("|", "\\|").replace("\n", " ")
    header = "| Metric | " + escape(first["label"]) + " | " + escape(second["label"]) + " |"
    if projected is not None:
        header += " " + escape(projected["label"]) + " (estimate) |"
    lines = ["sNMF PC comparison", "", header,
             "|---|---:|---:|---:|" if projected is not None else "|---|---:|---:|"]
    def initial_row(label, first_value, second_value, third_value):
        line = "| " + label + " | " + escape(first_value) + " | " + escape(second_value) + " |"
        if projected is not None:
            line += " " + escape(third_value) + " |"
        lines.append(line)
    initial_row("Result type", "Measured run", "Measured run", "Projection only")
    initial_row("Physical cores", first["provenance"].get("physical_cores"),
                second["provenance"].get("physical_cores"), 16)
    initial_row("Logical CPU threads", first["provenance"].get("logical_cpus"),
                second["provenance"].get("logical_cpus"), 32)
    def ram(provenance):
        value = provenance.get("memory", {}).get("total_bytes")
        return round(value / 1024 ** 3, 2) if value else None
    initial_row("RAM (GiB)", ram(first["provenance"]), ram(second["provenance"]), "96 nominal")
    rows = [("Mode", "configuration", "mode"),
            ("Parallel workers", "configuration", "workers"),
            ("Training rows per worker", "configuration", "n_train"),
            ("Input features", "configuration", "n_features"),
            ("Prediction rows per fit", "configuration", "n_test"),
            ("Seconds per measured window", "configuration", "duration_seconds"),
            ("Measured windows", "configuration", "repeats"),
            ("Fits/s (measured median or ideal estimate)", "aggregate", "qualified_fits_per_second_median"),
            ("Training input values/s (measured median or ideal estimate)", "aggregate", "qualified_input_values_per_second_median"),
            ("Mean total CPU utilization (%)", "aggregate", "cpu_utilization_percent_mean"),
            ("Peak benchmark resident RAM (bytes)", "aggregate", "benchmark_peak_rss_bytes"),
            ("Peak system used RAM (bytes)", "aggregate", "system_peak_used_ram_bytes")]
    for label, group, key in rows:
        line = "| " + label + " | " + escape(first[group].get(key)) + " | " + escape(second[group].get(key)) + " |"
        if projected is not None:
            if key == "mode":
                value = "Ideal CPU scaling"
            elif key == "workers":
                value = "32 assumed"
            elif key in ("n_train", "n_features", "n_test"):
                value = projected.get("configuration", {}).get(key)
            elif key == "qualified_fits_per_second_median":
                value = projected.get("metrics", {}).get("ideal_fits_per_second")
            elif key == "qualified_input_values_per_second_median":
                value = projected.get("metrics", {}).get("ideal_input_values_per_second")
            else:
                value = "not measured"
            line += " " + escape(value) + " |"
        lines.append(line)
    if projected is not None:
        lines += ["", "PC3 is an estimate, not a third physical benchmark. It assumes the same per-core "
                  "performance and per-fit data as PC1, with twice the independent-fit parallel capacity "
                  "and 96 GB RAM. It does not predict one fit using 32 numerical threads."]
        if projected.get("blockers"):
            lines.append("Projected speed is pending:")
            lines.extend("- " + item for item in projected["blockers"])
    lines.extend(["", "Ratios use the second report divided by the first."])
    for key, value in decision["ratios"].items():
        if value is not None:
            label = ("Matched-work training throughput" if key.startswith("matched_")
                     else "Training input-value throughput (workload dependent)")
            lines.append(label + ": " + fmt(value) + "x.")
    if decision["blockers"]:
        lines.extend(["", "No comparison ratio was qualified:"])
        lines.extend("- " + item for item in decision["blockers"])
    if decision["qualifications"]:
        lines.append("")
        lines.extend("- " + item for item in decision["qualifications"])
    if first.get("sweep_context") and second.get("sweep_context"):
        lines += ["", "Workers versus threads (fits/second; fixed per-fit size within each PC):", "",
                  "| Thread budget N | " + escape(first["label"]) + " N workers x 1 | "
                  + escape(first["label"]) + " 1 worker x N | " + escape(second["label"])
                  + " N workers x 1 | " + escape(second["label"]) + " 1 worker x N |",
                  "|---:|---:|---:|---:|---:|"]
        stages = [{(r["workers"], r["native_threads"]): r for r in report["sweep_context"]["runs"]}
                  for report in (first, second)]
        counts = {w * t for side in stages for w, t in side}
        for count in sorted(counts):
            cells = [str(count)]
            for side in stages:
                cells += [escape(side.get(allocation, {}).get("fits_per_second"))
                          for allocation in ((count, 1), (1, count))]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Inspect each report's iterations, selected-feature counts, reconstruction "
                  "errors, warnings and CPU/RAM telemetry alongside throughput. Warm-up, "
                  "data generation and post-window draining are outside the timed window.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="PC1 (8-core/16-thread) sweep/results directory or report.json")
    parser.add_argument("second", type=Path, help="PC2 (Ubuntu) sweep/results directory or report.json")
    parser.add_argument("--output", type=Path, help="New comparison directory; never overwrites")
    args = parser.parse_args(argv)
    first, second = load_report(args.first), load_report(args.second)
    decision = assess(first, second)
    from projection import make_projection
    profiles = json.loads((Path(__file__).resolve().parent / "pc_profiles.json").read_text(encoding="utf-8"))
    profile = next(pc for pc in profiles["pcs"] if pc["id"] == "PC3")
    projected = make_projection(first, profile)
    rendered = render(first, second, decision, projected)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "comparison.md").write_text(rendered, encoding="utf-8")
        (args.output / "comparison.json").write_text(
            json.dumps({"schema_version": 1, "first": str(args.first.resolve()),
                        "second": str(args.second.resolve()), "assessment": decision,
                        "projected_pc": projected}, indent=2, allow_nan=False)
            + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision["comparable_protocol"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
