"""No-fit checks for the explicitly hypothetical third PC."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portable_projection_test_module", ROOT / "portable_speedtest" / "projection.py"
)
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)
GIB = 1024 ** 3


def source_report():
    """An invented qualified receipt, never a measurement of a real PC."""
    rows, features, test_rows = 8192, 2048, 256
    per_worker = 40 * rows * features + 16 * test_rows * features + 256 * 1024 ** 2
    return {
        "schema_version": 1,
        "label": "Fixture only: current 8-core PC",
        "status": "qualified",
        "smoke": False,
        "manifest_sha256": "fixture-source-identity",
        "protocol_sha256": "fixture-protocol-identity",
        "configuration": {
            "mode": "capacity", "smoke": False,
            "workers": 16, "native_threads_per_worker": 1,
            "total_requested_native_threads": 16,
            "duration_seconds": 60, "repeats": 3,
            "n_train": rows, "n_test": test_rows, "n_features": features,
            "training_rows_selection": "memory_capacity",
            "memory_fraction": 0.9,
            "estimated_peak_bytes_per_worker": per_worker,
            "estimated_peak_total_bytes": 16 * per_worker,
            "auto_memory_guard_passed": True, "memory_guard_passed": True,
            "model": {"solver": "cd", "screening": "absolute", "score_threshold": 2.0},
        },
        "aggregate": {
            "qualified_fits_per_second_median": 2.0,
            "qualified_input_values_per_second_median": 2.0 * rows * features,
            "fit_seconds_median": 8.0,
            "process_cpu_seconds_median": 7.0,
        },
        "provenance": {
            "physical_cores": 8, "logical_cpus": 16,
            "processor": "AMD Ryzen 9 6900HS",
            "memory": {"total_bytes": 40 * GIB, "available_bytes": 30 * GIB},
        },
        "repeats": [
            {"repeat": i, "status": "qualified", "records": [
                {"status": "qualified", "qualified": True,
                 "timing": {"fit_seconds": 8.0, "process_cpu_seconds": 7.0},
                 "quality": {"n_iter": 27}}
            ]}
            for i in (1, 2, 3)
        ],
    }


class PortableProjectionTests(unittest.TestCase):
    def assert_unavailable(self, report):
        result = projection.make_projection(report)
        self.assertEqual(result["kind"], "projection")
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(result["blockers"])
        self.assertTrue(all(value is None for value in result["metrics"].values()))
        return result

    def test_ideal_projection_is_separate_from_measured_results(self):
        report = source_report()
        original = deepcopy(report)
        result = projection.make_projection(report)
        self.assertEqual(report, original)
        self.assertEqual(result["kind"], "projection")
        self.assertEqual(result["status"], "estimated")
        self.assertIn("project", result["label"].lower())
        self.assertTrue(result["assumptions"])
        self.assertFalse(result["blockers"])
        self.assertEqual(result["hardware"]["physical_cores"], 16)
        self.assertEqual(result["hardware"]["logical_cpus"], 32)
        self.assertEqual(result["hardware"]["memory"]["total_bytes"], 96 * GIB)
        self.assertEqual(result["configuration"]["workers"], 32)
        self.assertEqual(result["configuration"]["native_threads_per_worker"], 1)
        for key in ("n_train", "n_test", "n_features"):
            self.assertEqual(result["configuration"][key], report["configuration"][key])
        self.assertEqual(result["metrics"], {
            "ideal_fits_per_second": 4.0,
            "ideal_input_values_per_second": 4.0 * 8192 * 2048,
            "ideal_throughput_multiplier": 2.0,
        })
        for key in ("aggregate", "repeats", "records", "timing", "n_iter"):
            self.assertNotIn(key, result)
        self.assertEqual(result["source"]["label"], report["label"])
        self.assertEqual(result["source"]["manifest_sha256"], report["manifest_sha256"])

    def test_ram_plan_carries_background_usage_and_reserves_headroom(self):
        report = source_report()
        result = projection.make_projection(report)
        memory = result["memory_plan"]
        target_available = 96 * GIB - (40 * GIB - 30 * GIB)
        target_reserve = int(0.1 * 96 * GIB)
        self.assertEqual(memory["estimated_available_bytes"], target_available)
        self.assertEqual(memory["memory_budget_bytes"], int((target_available - target_reserve) * 0.9))
        self.assertEqual(memory["estimated_peak_total_bytes"],
                         32 * report["configuration"]["estimated_peak_bytes_per_worker"])
        self.assertTrue(memory["auto_memory_guard_passed"])

    def test_ram_shortfall_blocks_throughput_estimate(self):
        report = source_report()
        report["configuration"]["estimated_peak_bytes_per_worker"] = 3 * GIB
        result = self.assert_unavailable(report)
        self.assertFalse(result["memory_plan"]["auto_memory_guard_passed"])

    def test_missing_identity_hardware_and_memory_metadata_block_projection(self):
        for group, keys in (
            (None, ("schema_version", "manifest_sha256", "protocol_sha256", "repeats")),
            ("provenance", ("physical_cores", "logical_cpus", "memory")),
            ("configuration", ("workers", "native_threads_per_worker", "n_train",
                               "estimated_peak_bytes_per_worker", "memory_fraction")),
        ):
            for key in keys:
                with self.subTest(group=group, missing=key):
                    report = source_report()
                    del (report if group is None else report[group])[key]
                    self.assert_unavailable(report)

    def test_only_full_source_cpu_baseline_can_support_extrapolation(self):
        cases = [
            ("configuration", "workers", 8),
            ("configuration", "workers", 32),
            ("configuration", "native_threads_per_worker", 2),
            ("provenance", "physical_cores", 16),
            ("provenance", "logical_cpus", 32),
            ("configuration", "training_rows_selection", "explicit"),
        ]
        for group, key, value in cases:
            with self.subTest(group=group, key=key, value=value):
                report = source_report()
                report[group][key] = value
                self.assert_unavailable(report)

    def test_smoke_partial_failed_or_invalid_measurements_are_unavailable(self):
        for status in ("invalid", "running", "failed", "incomplete-sweep"):
            with self.subTest(status=status):
                report = source_report()
                report["status"] = status
                self.assert_unavailable(report)
        for group in (None, "configuration"):
            with self.subTest(smoke_group=group):
                report = source_report()
                (report if group is None else report[group])["smoke"] = True
                self.assert_unavailable(report)
        for change in ("missing-window", "invalid-window", "invalid-record"):
            with self.subTest(change=change):
                report = source_report()
                if change == "missing-window":
                    report["repeats"].pop()
                elif change == "invalid-window":
                    report["repeats"][1]["status"] = "invalid"
                else:
                    report["repeats"][1]["records"][0]["qualified"] = False
                self.assert_unavailable(report)

    def test_invalid_rates_and_memory_cannot_yield_numeric_estimates(self):
        for key in ("qualified_fits_per_second_median", "qualified_input_values_per_second_median"):
            for value in (None, 0, -1, float("nan"), float("inf"), True):
                with self.subTest(rate=key, value=value):
                    report = source_report()
                    report["aggregate"][key] = value
                    self.assert_unavailable(report)
        for memory in ({"total_bytes": 40 * GIB, "available_bytes": None},
                       {"total_bytes": 40 * GIB, "available_bytes": 41 * GIB},
                       {"total_bytes": float("inf"), "available_bytes": 30 * GIB}):
            with self.subTest(memory=memory):
                report = source_report()
                report["provenance"]["memory"] = memory
                self.assert_unavailable(report)

    def test_projection_requires_cd_and_absolute_cutoff(self):
        for key, value in (("solver", "mu"), ("screening", "relative"), ("score_threshold", None)):
            with self.subTest(key=key, value=value):
                report = source_report()
                report["configuration"]["model"][key] = value
                self.assert_unavailable(report)

    def test_no_baseline_still_describes_third_pc_without_speed_numbers(self):
        result = self.assert_unavailable({})
        self.assertEqual(result["hardware"]["physical_cores"], 16)
        self.assertEqual(result["hardware"]["logical_cpus"], 32)
        self.assertEqual(result["hardware"]["memory"]["total_bytes"], 96 * GIB)

    def test_full_sweep_can_supply_its_frozen_maximum_worker_workload(self):
        report = source_report()
        plan = deepcopy(report["configuration"])
        report["configuration"]["training_rows_selection"] = "explicit"
        report["sweep_context"] = {
            "status": "qualified",
            "manifest_sha256": report["manifest_sha256"],
            "protocol_sha256": report["protocol_sha256"],
            "configuration_overrides": {"train_rows": None},
            "description": {"per_fit_configuration": plan},
        }
        self.assertEqual(projection.make_projection(report)["status"], "estimated")
        for change in ("partial", "different-source", "reduced-size", "wrong-worker-plan", "different-shape"):
            with self.subTest(change=change):
                invalid = deepcopy(report)
                sweep = invalid["sweep_context"]
                if change == "partial":
                    sweep["status"] = "running"
                elif change == "different-source":
                    sweep["manifest_sha256"] = "different-source"
                elif change == "reduced-size":
                    sweep["configuration_overrides"]["train_rows"] = 8192
                elif change == "wrong-worker-plan":
                    sweep["description"]["per_fit_configuration"]["workers"] = 8
                else:
                    sweep["description"]["per_fit_configuration"]["n_train"] += 2
                self.assert_unavailable(invalid)


if __name__ == "__main__":
    unittest.main()
