"""Ideal CPU scaling for the hypothetical third PC; never runs model fits."""

from __future__ import annotations

import math


GIB = 1024 ** 3
DEFAULT_PROFILE = {
    "label": "PC3 - projected 16-core / 32-thread",
    "physical_cores": 16,
    "logical_cpus": 32,
    "memory_total_bytes": 96 * GIB,
}


def positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def make_projection(report, profile=None):
    """Return a separately labelled ideal estimate, or reasons it is unavailable.

    RAM is interpreted in GiB, as used for installed hardware. The target keeps
    the measured per-fit shape; the extra RAM is a feasibility check, not a
    multiplier for fitting speed. No measured telemetry is synthesized.
    """
    profile = dict(DEFAULT_PROFILE if profile is None else profile)
    if profile.get("physical_cores") != 16 or profile.get("logical_cpus") != 32:
        raise ValueError("The third-PC profile must specify 16 cores and 32 threads.")
    target_ram = profile.get("memory_total_bytes")
    if not positive(target_ram):
        raise ValueError("The third-PC profile needs a positive RAM capacity in bytes.")
    config = report.get("configuration", {})
    provenance = report.get("provenance", {})
    aggregate = report.get("aggregate", {})
    sweep = report.get("sweep_context")
    blockers = []
    if (report.get("schema_version") != 1 or report.get("status") != "qualified"
            or report.get("kind") == "projection"):
        blockers.append("A qualified measured source report is required.")
    if report.get("smoke") or config.get("smoke"):
        blockers.append("Smoke checks cannot supply the projection baseline.")
    if any(not report.get(key) for key in ("manifest_sha256", "protocol_sha256")):
        blockers.append("Source bundle and protocol identities are required.")
    if provenance.get("physical_cores") != 8 or provenance.get("logical_cpus") != 16:
        blockers.append("The source must be the measured 8-core / 16-thread PC.")
    if config.get("workers") != 16 or config.get("native_threads_per_worker") != 1:
        blockers.append("The source must use all 16 workers with one thread each.")
    model = config.get("model", {})
    if (model.get("solver") != "cd" or model.get("screening") != "absolute"
            or model.get("score_threshold") != 2.0):
        blockers.append("The source must use CD with absolute cutoff 2.0.")
    if config.get("mode") not in ("capacity", "matched"):
        blockers.append("The source workload mode is missing or unsupported.")
    if any(not isinstance(config.get(key), int) or isinstance(config.get(key), bool)
           or config[key] <= 0 for key in ("n_train", "n_test", "n_features")):
        blockers.append("The source workload dimensions are missing or invalid.")
    windows = report.get("repeats", [])
    if not windows or len(windows) != config.get("repeats"):
        blockers.append("All requested measurement windows must be present.")
    if any(window.get("status") != "qualified" for window in windows):
        blockers.append("Every source measurement window must be qualified.")
    if any(not record.get("qualified", record.get("status") == "qualified")
           for window in windows for record in window.get("records", [])):
        blockers.append("Unqualified source fit records prevent extrapolation.")
    if not positive(config.get("duration_seconds")):
        blockers.append("The source measurement-window duration is invalid.")
    rate = aggregate.get("qualified_fits_per_second_median")
    input_rate = aggregate.get("qualified_input_values_per_second_median")
    if not positive(rate) or not positive(input_rate):
        blockers.append("Positive finite qualified measured throughput is required.")
    elif not positive(rate * 2) or not positive(input_rate * 2):
        blockers.append("The ideal throughput estimate must remain finite.")
    if sweep is not None:
        if (sweep.get("status") != "qualified"
                or any(sweep.get(key) != report.get(key)
                       for key in ("manifest_sha256", "protocol_sha256"))):
            blockers.append("The source sweep must be qualified and match its measured report.")
        overrides = sweep.get("configuration_overrides", {})
        plan = sweep.get("description", {}).get("per_fit_configuration", {})
        if overrides.get("train_rows") is not None:
            blockers.append("A reduced-size validation sweep cannot supply this projection.")
        if config.get("mode") == "capacity" and (
                plan.get("training_rows_selection") != "memory_capacity"
                or plan.get("workers") != 16
                or any(plan.get(key) != config.get(key)
                       for key in ("n_train", "n_test", "n_features"))):
            blockers.append("The source sweep needs its full 16-worker RAM-sized workload.")
    elif config.get("mode") == "capacity" and config.get("training_rows_selection") != "memory_capacity":
        blockers.append("An explicit-size validation run cannot supply this projection.")

    memory = provenance.get("memory", {})
    total, available = memory.get("total_bytes"), memory.get("available_bytes")
    fraction, per_worker = config.get("memory_fraction"), config.get("estimated_peak_bytes_per_worker")
    memory_plan = {"estimated_available_bytes": None, "memory_budget_bytes": None,
                   "estimated_peak_total_bytes": None, "auto_memory_guard_passed": False}
    valid_memory = (positive(total) and positive(available) and available <= total
                    and positive(fraction) and fraction <= 1 and positive(per_worker))
    if not valid_memory:
        blockers.append("Source RAM telemetry and the per-worker memory estimate are required.")
    else:
        estimated_available = max(0, target_ram - (total - available))
        reserve = max(4 * GIB, int(0.1 * target_ram))
        budget = int(max(0, estimated_available - reserve) * fraction)
        peak = 32 * per_worker
        memory_plan.update(estimated_available_bytes=estimated_available,
                           memory_budget_bytes=budget, memory_reserve_bytes=reserve,
                           estimated_peak_total_bytes=peak, auto_memory_guard_passed=peak <= budget)
        if peak > budget:
            blockers.append("The original per-fit workload does not fit 32 workers in the target RAM budget.")
    estimated = not blockers
    return {
        "schema_version": 1, "kind": "projection",
        "status": "estimated" if estimated else "unavailable",
        "label": profile.get("label", DEFAULT_PROFILE["label"]),
        "hardware": {"physical_cores": 16, "logical_cpus": 32,
                     "memory": {"total_bytes": target_ram, "measurement": "hypothetical",
                                "note": "User-specified RAM; GB interpreted as GiB for installed hardware."}},
        "source": {"label": report.get("label"),
                   "manifest_sha256": report.get("manifest_sha256"),
                   "protocol_sha256": report.get("protocol_sha256"),
                   "physical_cores": provenance.get("physical_cores"),
                   "logical_cpus": provenance.get("logical_cpus")},
        "configuration": {"workers": 32, "native_threads_per_worker": 1,
                          "mode": config.get("mode"),
                          **{key: config.get(key) for key in ("n_train", "n_test", "n_features")}},
        "metrics": {"ideal_fits_per_second": rate * 2 if estimated else None,
                    "ideal_input_values_per_second": input_rate * 2 if estimated else None,
                    "ideal_throughput_multiplier": 2.0 if estimated else None},
        "memory_plan": memory_plan,
        "assumptions": [
            "Hypothetical estimate, not a measured third PC or a guaranteed speedup.",
            "Same CPU architecture, per-core speed, software, and measured per-fit workload as the source PC.",
            "Ideal linear scaling from 16 to 32 independent workers with one thread each; contention and bandwidth limits are ignored.",
            "Same background RAM usage as the source snapshot; preserve the source memory fraction and target OS reserve.",
            "Extra RAM only checks whether 32 copies fit; timing a larger RAM-sized dataset requires a real run.",
            "Single-worker multithread speed, per-fit latency, CPU utilization, and actual peak RAM are not projected.",
        ],
        "blockers": list(dict.fromkeys(blockers)),
    }
