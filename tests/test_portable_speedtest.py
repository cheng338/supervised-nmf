"""No-fit checks for portable benchmark accounting and comparison integrity."""

import ast
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "portable_speedtest"


def load_script(name):
    spec = importlib.util.spec_from_file_location("portable_test_" + name, PORTABLE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_script("compare")
builder = load_script("build_bundle")
benchmark = load_script("benchmark")
with patch.dict(sys.modules, {"benchmark": benchmark}):
    sweep = load_script("sweep")


def report_fixture(mode="capacity"):
    return {
        "schema_version": 1, "label": "first", "status": "qualified",
        "manifest_sha256": "source-and-data-identity", "protocol_sha256": "protocol-identity",
        "configuration": {
            "mode": mode, "duration_seconds": 60, "repeats": 3,
            "workers": 8, "native_threads_per_worker": 1,
            "n_train": 2048, "n_features": 512, "n_test": 256,
            "model": {"solver": "cd", "screening": "absolute", "random_state": 43},
        },
        "aggregate": {
            "qualified_fits_per_second_median": 2.0,
            "qualified_input_values_per_second_median": 2.0 * 2048 * 512,
        },
        "provenance": {"python": "3.10.12", "versions": {"numpy": "1.26.4"}, "platform": "test-os"},
        "repeats": [{"repeat": repeat, "status": "qualified", "records": []} for repeat in (1, 2, 3)],
    }


class PortableComparisonTests(unittest.TestCase):
    def setUp(self):
        self.first = report_fixture()
        self.second = deepcopy(self.first)
        self.second["label"] = "second"
        self.second["configuration"]["workers"] = 40
        self.second["aggregate"]["qualified_fits_per_second_median"] *= 2
        self.second["aggregate"]["qualified_input_values_per_second_median"] *= 2

    def test_capacity_never_claims_equal_work_speedup_even_at_equal_shape(self):
        decision = compare.assess(self.first, self.second)
        self.assertTrue(decision["comparable_protocol"])
        self.assertEqual(decision["ratios"], {"input_values_rate_ratio_second_over_first": 2.0})
        self.assertIn("descriptive throughput", " ".join(decision["qualifications"]))

    def test_more_ram_may_increase_workload_without_claiming_matched_speedup(self):
        self.second["configuration"]["n_train"] *= 10
        decision = compare.assess(self.first, self.second)
        self.assertTrue(decision["comparable_protocol"])
        self.assertFalse(decision["same_shape"])
        self.assertNotIn("matched_fit_rate_ratio_second_over_first", decision["ratios"])
        self.assertIn("do not establish an equal-work speedup", " ".join(decision["qualifications"]))

    def test_matched_mode_allows_more_workers_and_uses_second_over_first(self):
        for item in (self.first, self.second):
            item["configuration"]["mode"] = "matched"
        decision = compare.assess(self.first, self.second)
        self.assertTrue(decision["comparable_protocol"])
        self.assertEqual(decision["ratios"]["matched_fit_rate_ratio_second_over_first"], 2.0)

    def test_matched_mode_requires_all_dimensions_to_match(self):
        for item in (self.first, self.second):
            item["configuration"]["mode"] = "matched"
        for key in ("n_train", "n_test", "n_features"):
            with self.subTest(dimension=key):
                different = deepcopy(self.second)
                different["configuration"][key] += 1
                decision = compare.assess(self.first, different)
                self.assertFalse(decision["comparable_protocol"])
                self.assertFalse(decision["ratios"])

    def test_missing_or_different_bundle_protocol_and_settings_block_ratios(self):
        changes = [(None, "manifest_sha256"), (None, "protocol_sha256")]
        changes += [("configuration", key) for key in ("mode", "duration_seconds", "repeats", "model")]
        for group, key in changes:
            for missing in (True, False):
                with self.subTest(group=group, key=key, missing=missing):
                    different = deepcopy(self.second)
                    target = different if group is None else different[group]
                    if missing:
                        del target[key]
                    else:
                        target[key] = "different"
                    decision = compare.assess(self.first, different)
                    self.assertFalse(decision["comparable_protocol"])
                    self.assertFalse(decision["ratios"])

    def test_invalid_smoke_and_nonpositive_or_nonfinite_rates_block_ratios(self):
        changes = [("status", "invalid"), ("smoke", True)]
        changes += [("rate", value) for value in (None, 0, -1, float("nan"), float("inf"))]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                different = deepcopy(self.second)
                if key == "status":
                    different[key] = value
                elif key == "smoke":
                    different["configuration"][key] = value
                else:
                    different["aggregate"]["qualified_fits_per_second_median"] = value
                decision = compare.assess(self.first, different)
                self.assertFalse(decision["comparable_protocol"])
                self.assertFalse(decision["ratios"])

    def test_software_difference_qualifies_complete_system_comparison(self):
        self.second["provenance"]["versions"]["numpy"] = "2.0.0"
        self.second["provenance"]["thread_pools"] = [{"internal_api": "blis", "prefix": "libblis",
                                                        "version": "1.0", "threading_layer": "openmp"}]
        decision = compare.assess(self.first, self.second)
        self.assertTrue(decision["comparable_protocol"])
        self.assertIn("does not isolate CPU hardware", " ".join(decision["qualifications"]))
        self.assertIn("numerical libraries", " ".join(decision["qualifications"]))

    def test_unqualified_window_blocks_ratio_despite_positive_aggregate(self):
        self.second["repeats"][1]["status"] = "invalid"
        decision = compare.assess(self.first, self.second)
        self.assertFalse(decision["comparable_protocol"])
        self.assertFalse(decision["ratios"])
        self.assertIn("unqualified measurement window", " ".join(decision["blockers"]))

    def test_runner_top_level_smoke_marker_blocks_ratios(self):
        self.second["smoke"] = True
        decision = compare.assess(self.first, self.second)
        self.assertFalse(decision["comparable_protocol"])
        self.assertFalse(decision["ratios"])

    def test_missing_repeat_receipts_block_positive_aggregate_claim(self):
        for repeats in ([], self.second["repeats"][:2]):
            with self.subTest(count=len(repeats)):
                different = deepcopy(self.second)
                different["repeats"] = repeats
                decision = compare.assess(self.first, different)
                self.assertFalse(decision["comparable_protocol"])
                self.assertFalse(decision["ratios"])

    def test_record_warning_does_not_hide_a_later_invalid_window(self):
        self.second["repeats"][0]["records"] = [{"status": "invalid"}]
        self.second["repeats"][2]["status"] = "invalid"
        decision = compare.assess(self.first, self.second)
        self.assertFalse(decision["comparable_protocol"])
        self.assertFalse(decision["ratios"])
        self.assertIn("unqualified records", " ".join(decision["qualifications"]))
        self.assertIn("unqualified measurement window", " ".join(decision["blockers"]))

    def test_library_paths_and_report_order_do_not_create_false_backend_difference(self):
        pools = [{"internal_api": "openblas", "prefix": "libopenblas", "version": "1", "filepath": "/pc1/a"},
                 {"internal_api": "openmp", "prefix": "libomp", "version": "2", "filepath": "/pc1/b"}]
        self.first["provenance"]["blas"] = pools
        self.second["provenance"]["blas"] = [{**pool, "filepath": "/pc2/other"} for pool in reversed(pools)]
        self.assertNotIn("Software differs", " ".join(compare.assess(self.first, self.second)["qualifications"]))

    def test_report_directory_loading_checks_schema_and_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.first), encoding="utf-8")
            self.assertEqual(compare.load_report(directory), self.first)
            for key in ("schema_version", "configuration", "aggregate", "provenance", "label"):
                invalid = deepcopy(self.first)
                del invalid[key]
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    compare.load_report(path)

    def test_sweep_loading_selects_maximum_workers_and_rejects_partial_sweep_comparison(self):
        with tempfile.TemporaryDirectory(prefix="snmf-sweep-load-") as directory:
            path = Path(directory)
            runs = []
            for workers in (1, 40, 8):
                identifier = f"workers-{workers:03d}"
                stage_path = path / identifier
                stage_path.mkdir()
                child = deepcopy(self.first)
                child["configuration"]["workers"] = workers
                (stage_path / "report.json").write_text(json.dumps(child), encoding="utf-8")
                runs.append({"id": identifier, "workers": workers})
            sweep_report = {"label": "Whole PC", "status": "qualified", "runs": runs,
                            "manifest_sha256": self.first["manifest_sha256"],
                            "protocol_sha256": self.first["protocol_sha256"]}
            sweep_path = path / "sweep.json"
            sweep_path.write_text(json.dumps(sweep_report), encoding="utf-8")
            loaded = compare.load_report(path)
            self.assertEqual(loaded["configuration"]["workers"], 40)
            self.assertEqual(loaded["label"], "Whole PC")
            self.assertTrue(compare.assess(self.first, loaded)["comparable_protocol"])
            sweep_report["status"] = "running"
            sweep_path.write_text(json.dumps(sweep_report), encoding="utf-8")
            decision = compare.assess(self.first, compare.load_report(path))
            self.assertFalse(decision["comparable_protocol"])
            self.assertFalse(decision["ratios"])


class PortableZipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="snmf-portable-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bundle = self.directory / "bundle"
        self.bundle.mkdir()
        (self.bundle / "model.py").write_bytes(b"# unpublished sNMF test snapshot\n")
        self.manifest = {"files": {"model.py": builder.sha256(self.bundle / "model.py")}}
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.archive = self.directory / "bundle.zip"

    def make_zip(self, omit=None, replacements=None, extra=None):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name in ("model.py", "manifest.json"):
                if name != omit:
                    archive.writestr(name, (replacements or {}).get(name, (self.bundle / name).read_bytes()))
            for name, content in extra or []:
                archive.writestr(name, content)

    def test_valid_zip_is_identical_to_manifest_and_frozen_directory(self):
        self.make_zip()
        builder.verify_zip(self.archive, self.bundle, self.manifest)

    def test_missing_extra_and_duplicate_members_rejected(self):
        cases = [{"omit": "model.py"}, {"extra": [("foreign.py", b"injected")]},
                 {"extra": [("model.py", b"duplicate")]}]
        for arguments in cases:
            with self.subTest(arguments=arguments), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.make_zip(**arguments)
                with self.assertRaises(RuntimeError):
                    builder.verify_zip(self.archive, self.bundle, self.manifest)

    def test_modified_archive_or_directory_rejected(self):
        self.make_zip(replacements={"model.py": b"changed archive"})
        with self.assertRaises(RuntimeError):
            builder.verify_zip(self.archive, self.bundle, self.manifest)
        self.make_zip()
        (self.bundle / "model.py").write_bytes(b"changed disk")
        with self.assertRaises(RuntimeError):
            builder.verify_zip(self.archive, self.bundle, self.manifest)

    def test_matching_modified_archive_and_directory_cannot_bypass_manifest(self):
        (self.bundle / "model.py").write_bytes(b"same injected source in both locations")
        self.make_zip()
        with self.assertRaises(RuntimeError):
            builder.verify_zip(self.archive, self.bundle, self.manifest)

    def test_builder_includes_all_entrypoints_and_unpublished_python_source(self):
        inputs = builder.inputs()
        for name in ("benchmark.py", "compare.py", "check_compatibility.py", "sweep.py"):
            with self.subTest(entrypoint=name):
                self.assertEqual(inputs[Path(name)], PORTABLE / name)
        source = ROOT / "src" / "supervised_nmf"
        expected = {path for path in source.rglob("*.py")}
        bundled = {path for relative, path in inputs.items()
                   if relative.parts[:2] == ("package", "supervised_nmf")}
        self.assertTrue(expected)
        self.assertEqual(bundled, expected)


class PortableManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="snmf-manifest-test-")
        self.addCleanup(self.temporary.cleanup)
        # Match benchmark.ROOT, which is resolved when the module is loaded.
        self.bundle = (Path(self.temporary.name) / "bundle").resolve()
        self.bundle.mkdir()
        self.payloads = {"benchmark.py": b"# runner\n", "compare.py": b"# compare\n",
                         "check_compatibility.py": b"# compatibility\n", "sweep.py": b"# sweep\n",
                         "data/workload.npz": b"fixture bytes",
                         "package/supervised_nmf/__init__.py": b"# unpublished source\n",
                         "protocol.json": json.dumps(builder.protocol()).encode("utf-8")}
        for name, content in self.payloads.items():
            path = self.bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.files = {name: benchmark.digest(self.bundle / name) for name in self.payloads}
        self.save_manifest()
        self.root_patch = patch.object(benchmark, "ROOT", self.bundle)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def save_manifest(self):
        (self.bundle / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": self.files}), encoding="utf-8")

    def test_checked_bundle_returns_manifest_and_protocol_identity(self):
        protocol, manifest_hash, protocol_hash = benchmark.verify_bundle()
        self.assertEqual(protocol["model"]["solver"], "cd")
        self.assertEqual(protocol["model"]["screening"], "absolute")
        self.assertEqual(manifest_hash, benchmark.digest(self.bundle / "manifest.json"))
        self.assertEqual(protocol_hash, benchmark.digest(self.bundle / "protocol.json"))

    def test_changed_source_missing_file_and_untracked_module_are_rejected(self):
        path = self.bundle / "package/supervised_nmf/__init__.py"
        path.write_bytes(b"tampered source")
        with self.assertRaisesRegex(ValueError, "checksum"):
            benchmark.verify_bundle()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "manifest path"):
            benchmark.verify_bundle()
        path.write_bytes(self.payloads["package/supervised_nmf/__init__.py"])
        (path.parent / "foreign.py").write_bytes(b"untracked module")
        with self.assertRaisesRegex(ValueError, "Unmanifested"):
            benchmark.verify_bundle()

    def test_paths_outside_bundle_and_missing_required_binding_rejected(self):
        outside = self.bundle.parent / "outside.py"
        outside.write_bytes(b"outside source")
        self.files["../outside.py"] = benchmark.digest(outside)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "manifest path"):
            benchmark.verify_bundle()
        del self.files["../outside.py"]
        del self.files["data/workload.npz"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "required"):
            benchmark.verify_bundle()

    def test_protocol_without_cd_absolute_is_rejected_even_with_updated_hash(self):
        for key, value in (("solver", "mu"), ("screening", "percentile")):
            with self.subTest(key=key):
                protocol = builder.protocol()
                protocol["model"][key] = value
                path = self.bundle / "protocol.json"
                path.write_text(json.dumps(protocol), encoding="utf-8")
                self.files["protocol.json"] = benchmark.digest(path)
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "CD with absolute"):
                    benchmark.verify_bundle()

    def test_all_portable_entrypoints_require_manifest_bindings(self):
        original = self.files.copy()
        for name in ("benchmark.py", "compare.py", "sweep.py", "check_compatibility.py"):
            with self.subTest(entrypoint=name):
                self.files = {key: value for key, value in original.items() if key != name}
                self.save_manifest()
                with self.assertRaises(ValueError):
                    benchmark.verify_bundle()

    def test_unmanifested_root_module_cannot_shadow_numerical_dependency(self):
        (self.bundle / "numpy.py").write_bytes(b"# accidental or injected shadow module\n")
        with self.assertRaisesRegex(ValueError, "Unmanifested"):
            benchmark.verify_bundle()


class PortableCompatibilityTests(unittest.TestCase):
    def test_entrypoints_and_current_source_parse_under_python_311_grammar(self):
        paths = list(PORTABLE.glob("*.py")) + list((ROOT / "src/supervised_nmf").rglob("*.py"))
        self.assertGreater(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 11))

    def test_native_pool_check_requires_one_thread_in_every_pool(self):
        self.assertTrue(benchmark.native_pools_valid([{"num_threads": 1}, {"num_threads": 1}]))
        for pools in ([], [{"num_threads": 2}], [{}], [{"num_threads": 1}, {"num_threads": 8}]):
            with self.subTest(pools=pools):
                self.assertFalse(benchmark.native_pools_valid(pools))

    def test_native_pool_check_validates_every_pool_against_explicit_hybrid_count(self):
        pools = [{"num_threads": 2}, {"num_threads": 2}]
        self.assertTrue(benchmark.native_pools_valid(pools, expected=2))
        self.assertFalse(benchmark.native_pools_valid(pools))
        self.assertFalse(benchmark.native_pools_valid([{"num_threads": 2}, {"num_threads": 4}], expected=2))
        environment = benchmark.thread_environment(2)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
            self.assertEqual(environment[name], "2")
        self.assertEqual(environment["OMP_DYNAMIC"], "FALSE")
        self.assertEqual(environment["MKL_DYNAMIC"], "FALSE")


class PortableResourceTests(unittest.TestCase):
    HEADERS = {"X_train": {"shape": [768, 512], "dtype": "<f4"},
               "X_test": {"shape": [256, 512], "dtype": "<f4"},
               "y_train": {"shape": [768], "dtype": "<i8"},
               "y_test": {"shape": [256], "dtype": "<i8"}}

    def configuration(self, total_gib=64, available_gib=56, cpus=16, **changes):
        arguments = dict(smoke=False, workers="auto", seconds=None, repeats=None,
                         mode="capacity", train_rows=None, memory_fraction=0.9)
        arguments.update(changes)
        provenance = {"logical_cpus": cpus, "memory": {
            "total_bytes": int(total_gib * benchmark.GIB),
            "available_bytes": int(available_gib * benchmark.GIB)}}
        with patch.object(benchmark, "array_headers", return_value=self.HEADERS):
            return benchmark.describe_configuration(SimpleNamespace(**arguments), builder.protocol(), provenance)

    def test_capacity_uses_available_cpu_count_and_nearly_all_reserved_ram_budget(self):
        configuration = self.configuration(cpus=40, total_gib=503, available_gib=496)
        self.assertEqual(configuration["workers"], 40)
        self.assertEqual(configuration["native_threads_per_worker"], 1)
        self.assertEqual(configuration["n_train"] % 2, 0)
        self.assertGreaterEqual(configuration["n_train"], 32)
        self.assertGreaterEqual(configuration["memory_reserve_bytes"], 4 * benchmark.GIB)
        self.assertGreaterEqual(configuration["memory_reserve_bytes"], int(503 * benchmark.GIB * 0.1))
        self.assertTrue(configuration["auto_memory_guard_passed"])
        estimate = configuration["estimated_peak_total_bytes"]
        budget = configuration["memory_budget_bytes"]
        self.assertLessEqual(estimate, budget)
        self.assertLess(budget, 496 * benchmark.GIB - configuration["memory_reserve_bytes"])
        self.assertGreater(estimate, 0.99 * budget)

    def test_larger_ram_increases_real_training_rows_and_more_workers_share_budget(self):
        small = self.configuration(total_gib=40, available_gib=30, cpus=16)
        large = self.configuration(total_gib=503, available_gib=496, cpus=16)
        parallel = self.configuration(total_gib=503, available_gib=496, cpus=40)
        self.assertGreater(large["n_train"], small["n_train"])
        self.assertLess(parallel["n_train"], large["n_train"])
        self.assertEqual(parallel["workers"], 40)

    def test_small_memory_cannot_silently_reduce_worker_count_or_exceed_reserve(self):
        with self.assertRaises(ValueError):
            self.configuration(total_gib=8, available_gib=5, cpus=40)

    def test_explicit_oversized_workload_is_visible_as_failed_memory_guard(self):
        configuration = self.configuration(train_rows=1_000_000, total_gib=40, available_gib=30)
        self.assertFalse(configuration["auto_memory_guard_passed"])
        self.assertGreater(configuration["estimated_peak_total_bytes"], configuration["memory_budget_bytes"])

    def test_smoke_is_small_matched_work_even_when_large_resources_are_available(self):
        configuration = self.configuration(smoke=True, total_gib=503, available_gib=496, cpus=40,
                                           workers="40", seconds=120, repeats=4)
        self.assertEqual(configuration["mode"], "matched")
        self.assertEqual(configuration["workers"], 1)
        self.assertEqual(configuration["duration_seconds"], 3)
        self.assertEqual(configuration["repeats"], 1)
        self.assertEqual(configuration["n_train"], 768)
        self.assertEqual(configuration["n_features"], 512)

    def test_invalid_worker_window_and_memory_inputs_are_rejected(self):
        changes = [{"workers": "0"}, {"seconds": 0}, {"seconds": float("inf")}, {"repeats": 0},
                   {"memory_fraction": 0}, {"memory_fraction": 1.1}, {"memory_fraction": float("nan")},
                   {"train_rows": 31}, {"train_rows": 33}, {"mode": "matched", "train_rows": 64}]
        for arguments in changes:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.configuration(**arguments)

    def test_hybrid_allocation_is_bounded_by_available_cpu_threads(self):
        configuration = self.configuration(cpus=16, workers="8", native_threads=2)
        self.assertEqual(configuration["workers"], 8)
        self.assertEqual(configuration["native_threads_per_worker"], 2)
        self.assertEqual(configuration["total_requested_native_threads"], 16)
        for arguments in ({"workers": "9", "native_threads": 2}, {"workers": "17"},
                          {"native_threads": 0}, {"native_threads": -1}, {"native_threads": True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.configuration(cpus=16, **arguments)
        smoke = self.configuration(smoke=True, workers="8", native_threads=16)
        self.assertEqual(smoke["workers"], 1)
        self.assertEqual(smoke["native_threads_per_worker"], 1)

    def test_explicit_worker_count_cannot_bypass_run_memory_guard(self):
        hardware = {"logical_cpus": 16, "memory": {"total_bytes": 40 * benchmark.GIB,
                                                    "available_bytes": 30 * benchmark.GIB}}
        with tempfile.TemporaryDirectory(prefix="snmf-memory-test-") as directory:
            output = Path(directory) / "must-not-run"
            with patch.object(benchmark, "verify_bundle", return_value=(builder.protocol(), "manifest", "protocol")), \
                    patch.object(benchmark, "hardware", return_value=hardware), \
                    patch.object(benchmark, "array_headers", return_value=self.HEADERS), \
                    patch.object(benchmark, "run_repeat", side_effect=AssertionError("Must not launch workers")) as run:
                with self.assertRaises(ValueError):
                    benchmark.main(["--label", "too-large", "--workers", "1", "--train-rows", "1000000",
                                    "--output", str(output)])
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_psutil_cpu_affinity_limits_automatic_workers(self):
        fake_psutil = SimpleNamespace(cpu_count=lambda logical=False: 8,
                                      Process=lambda: SimpleNamespace(cpu_affinity=lambda: [2, 4, 6]))
        with patch.dict(sys.modules, {"psutil": fake_psutil}), \
                patch.object(benchmark.os, "cpu_count", return_value=40), \
                patch.object(benchmark.os, "sched_getaffinity", return_value=set(range(40)), create=True), \
                patch.object(benchmark, "memory_info", return_value={"total_bytes": 64 * benchmark.GIB,
                                                                     "available_bytes": 56 * benchmark.GIB}):
            hardware = benchmark.hardware()
        self.assertEqual(hardware["host_logical_cpus"], 40)
        self.assertEqual(hardware["affinity"], [2, 4, 6])
        self.assertEqual(hardware["logical_cpus"], 3)

    def test_linux_without_psutil_uses_available_memory_not_free_memory(self):
        meminfo = "MemTotal: 527433728 kB\nMemFree: 1000 kB\nMemAvailable: 520093696 kB\n"
        with patch.dict(sys.modules, {"psutil": None}), patch.object(benchmark.sys, "platform", "linux"), \
                patch.object(benchmark.Path, "read_text", return_value=meminfo):
            memory = benchmark.memory_info()
        self.assertEqual(memory["total_bytes"], 527433728 * 1024)
        self.assertEqual(memory["available_bytes"], 520093696 * 1024)
        self.assertEqual(memory["source"], "/proc/meminfo")


class PortableAccountingTests(unittest.TestCase):
    @staticmethod
    def record(status="qualified", completed=10.0, started_before=True):
        return {"status": status, "completed_within_window": completed <= 60,
                "started_before_deadline": started_before, "completed_offset_seconds": completed,
                "timing": {}, "quality": None}

    def test_nonconverged_failed_and_post_window_work_do_not_inflate_window_rate(self):
        records = [self.record(completed=2), self.record(completed=60),
                   self.record(status="invalid", completed=3), self.record(status="failed", completed=4),
                   self.record(completed=90), self.record(status="invalid", completed=120)]
        result = benchmark.summarize_task_counts(records, 60, 200, 20)
        self.assertEqual(result["completed_within_window"], 4)
        self.assertEqual(result["qualified_completed"], 2)
        self.assertEqual(result["qualified_fits_per_second"], 2 / 60)
        self.assertEqual(result["qualified_training_input_values_per_second"], 2 * 200 * 20 / 60)
        self.assertEqual(result["drain_completed"], 2)
        self.assertEqual(result["drain_qualified_completed"], 1)
        self.assertEqual(result["batch_qualified_completed_including_drain"], 3)
        self.assertEqual(result["batch_elapsed_seconds_including_drain"], 120)
        self.assertEqual(result["batch_qualified_fits_per_second_including_drain"], 3 / 120)

    def test_work_not_started_before_deadline_cannot_count_in_window(self):
        # An inconsistent receipt must never inflate the primary timed count.
        result = benchmark.summarize_task_counts([self.record(started_before=False)], 60, 200, 20)
        self.assertEqual(result["qualified_completed"], 0)
        self.assertEqual(result["qualified_fits_per_second"], 0)

    def test_empty_or_only_draining_window_has_zero_primary_throughput(self):
        for records in ([], [self.record(completed=120)]):
            with self.subTest(records=records):
                result = benchmark.summarize_task_counts(records, 60, 200, 20)
                self.assertTrue(result["window_had_zero_completions"])
                self.assertEqual(result["qualified_fits_per_second"], 0)
                self.assertEqual(result["qualified_training_input_values_per_second"], 0)
                if records:
                    self.assertEqual(result["batch_qualified_fits_per_second_including_drain"], 1 / 120)
                else:
                    self.assertIsNone(result["batch_qualified_fits_per_second_including_drain"])

    def repeat(self, identifier, count, status="qualified"):
        records = [self.record(completed=10) for _ in range(count)]
        summary = benchmark.summarize_task_counts(records, 60, 2048, 512)
        return {**summary, "repeat": identifier, "status": status, "records": records,
                "system_peak_used_ram_bytes": None, "sampled_peak_process_tree_rss_bytes": None,
                "cpu_utilization_percent_mean": None}

    def write(self, repeats):
        report = report_fixture()
        report["configuration"]["metric_note"] = "Only qualified in-window fits count."
        report["repeats"] = repeats
        with tempfile.TemporaryDirectory(prefix="snmf-report-test-") as directory:
            benchmark.write_reports(Path(directory), report)
            saved = compare.load_report(directory)
        self.assertEqual(saved, report)
        return report

    def test_invalid_repeat_is_excluded_from_median_and_entire_report_is_unqualified(self):
        report = self.write([self.repeat(1, 1), self.repeat(2, 100, "invalid"), self.repeat(3, 3)])
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["aggregate"]["qualified_repeats"], 2)
        self.assertEqual(report["aggregate"]["qualified_fits_per_second_median"], 2 / 60)
        decision = compare.assess(report_fixture(), report)
        self.assertFalse(decision["comparable_protocol"])
        self.assertFalse(decision["ratios"])

    def test_missing_or_all_invalid_repeats_cannot_produce_a_qualified_report(self):
        for repeats in ([], [self.repeat(i, 20, "invalid") for i in (1, 2, 3)]):
            with self.subTest(count=len(repeats)):
                report = self.write(repeats)
                self.assertEqual(report["status"], "invalid")
                self.assertEqual(report["aggregate"]["qualified_repeats"], 0)
                self.assertIsNone(report["aggregate"]["qualified_fits_per_second_median"])

    def test_zero_completion_windows_are_recorded_without_undefined_batch_rate_crash(self):
        report = self.write([self.repeat(i, 0) for i in (1, 2, 3)])
        self.assertEqual(report["aggregate"]["qualified_fits_per_second_median"], 0)
        self.assertIsNone(report["aggregate"]["batch_qualified_fits_per_second_median_including_drain"])
        self.assertFalse(compare.assess(report_fixture(), report)["comparable_protocol"])


class PortableSweepTests(unittest.TestCase):
    def test_default_counts_start_at_baseline_and_reach_each_pc_maximum(self):
        self.assertEqual(sweep.thread_counts(16, physical=8), [1, 2, 4, 8, 12, 16])
        self.assertEqual(sweep.thread_counts(40, physical=20), [1, 2, 4, 8, 12, 16, 24, 32, 40])
        self.assertEqual(sweep.thread_counts(6), [1, 2, 4, 6])

    def test_custom_counts_are_bounded_deduplicated_and_always_include_baseline(self):
        self.assertEqual(sweep.thread_counts(16, requested="1,8,4,8,16"), [1, 4, 8, 16])
        self.assertEqual(sweep.thread_counts(40, requested="2,1"), [1, 2])
        self.assertEqual(sweep.thread_counts(40, requested="8"), [1, 8])
        for requested in ("0,1", "-1,4", "1,17", "x,4"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                sweep.thread_counts(16, requested=requested)

    def test_first_three_stages_are_baseline_two_workers_and_two_threads(self):
        stages = sweep.stages([2, 1])
        self.assertEqual([(stage["workers"], stage["native_threads"]) for stage in stages],
                         [(1, 1), (2, 1), (1, 2)])
        self.assertEqual(len({stage["id"] for stage in stages}), 3)
        self.assertEqual([stage["kind"] for stage in stages], ["throughput", "throughput", "single-fit"])

    def test_every_larger_count_has_worker_and_single_fit_pairs_with_equal_cpu_budget(self):
        for maximum in (16, 40):
            with self.subTest(maximum=maximum):
                counts = sweep.thread_counts(maximum)
                stages = sweep.stages(counts)
                self.assertEqual(len(stages), 1 + 2 * (len(counts) - 1))
                self.assertEqual(len({stage["id"] for stage in stages}), len(stages))
                for count in counts[1:]:
                    pair = [stage for stage in stages if stage["count"] == count]
                    self.assertEqual([(stage["workers"], stage["native_threads"]) for stage in pair],
                                     [(count, 1), (1, count)])
                    self.assertTrue(all(stage["workers"] * stage["native_threads"] == count for stage in pair))

    def run_fake_sweep(self, destination, compatibility_status=0, invalid_stage=None):
        hardware = {"logical_cpus": 40, "physical_cores": 20, "memory": {
            "total_bytes": 503 * benchmark.GIB, "available_bytes": 496 * benchmark.GIB}}
        commands = []

        def fake_subprocess(command, **kwargs):
            commands.append(command)
            if Path(command[1]).name == "check_compatibility.py":
                return SimpleNamespace(returncode=compatibility_status)
            self.assertEqual(kwargs["stderr"], sweep.subprocess.STDOUT)
            kwargs["stdout"].write("simulated worker log\n")
            stage_output = Path(command[command.index("--output") + 1])
            stage_output.mkdir()
            workers = int(command[command.index("--workers") + 1])
            child = report_fixture()
            child["status"] = "invalid" if workers == invalid_stage else "qualified"
            child["aggregate"]["qualified_fits_per_second_median"] = float(workers)
            (stage_output / "report.json").write_text(json.dumps(child), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with patch.object(benchmark, "verify_bundle", return_value=(builder.protocol(), "manifest", "protocol")), \
                patch.object(benchmark, "hardware", return_value=hardware), \
                patch.object(benchmark, "array_headers", return_value=PortableResourceTests.HEADERS), \
                patch.object(benchmark, "describe_configuration", wraps=benchmark.describe_configuration) as planner, \
                patch.object(sweep.subprocess, "run", side_effect=fake_subprocess), \
                patch("sys.stdout", new_callable=io.StringIO):
            arguments = ["--label", "fake-PC", "--counts", "4,2,1", "--seconds", "10", "--repeats", "1",
                         "--output", str(destination)]
            result = sweep.main(arguments)
        planner.assert_called_once()
        self.assertEqual(planner.call_args.args[0].workers, "auto")
        self.assertEqual(planner.call_args.args[2]["logical_cpus"], 40)
        saved = json.loads((destination / "sweep.json").read_text(encoding="utf-8"))
        return result, commands, saved

    def test_reduced_count_sweep_freezes_every_fit_size_from_true_maximum_cpu_plan(self):
        with tempfile.TemporaryDirectory(prefix="snmf-sweep-test-") as directory:
            result, commands, report = self.run_fake_sweep(Path(directory) / "sweep")
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(Path(commands[0][1]).name, "check_compatibility.py")
        self.assertEqual(len(commands), 6)
        plan = report["description"]["per_fit_configuration"]
        self.assertEqual(plan["workers"], 40)
        self.assertEqual(report["n_train"], plan["n_train"])
        self.assertEqual([int(command[command.index("--workers") + 1]) for command in commands[1:]], [1, 2, 1, 4, 1])
        self.assertEqual({int(command[command.index("--train-rows") + 1]) for command in commands[1:]},
                         {plan["n_train"]})
        self.assertEqual({command[command.index("--seconds") + 1] for command in commands[1:]}, {"10.0"})
        self.assertEqual([run["within_pc_gain_vs_one"] for run in report["runs"]], [1.0, 2.0, 1.0, 4.0, 1.0])

    def test_failed_compatibility_check_prevents_all_benchmark_stages(self):
        with tempfile.TemporaryDirectory(prefix="snmf-sweep-test-") as directory:
            result, commands, report = self.run_fake_sweep(Path(directory) / "sweep", compatibility_status=2)
        self.assertEqual(result, 2)
        self.assertEqual(len(commands), 1)
        self.assertEqual(report["status"], "compatibility_failed")
        self.assertEqual(report["runs"], [])

    def test_paired_commands_keep_fixed_fit_shape_and_pass_native_threads_only_for_single_fit(self):
        with tempfile.TemporaryDirectory(prefix="snmf-paired-test-") as directory:
            result, commands, report = self.run_fake_sweep(Path(directory) / "sweep")
            for stage in report["runs"]:
                log = Path(directory) / "sweep" / (stage["id"] + ".log")
                self.assertEqual(log.read_text(encoding="utf-8"), "simulated worker log\n")
        self.assertEqual(result, 0)
        self.assertEqual(len(commands), 6)
        self.assertTrue(all("--native-threads" not in commands[index] for index in (1, 2, 4)))
        self.assertEqual([int(commands[index][commands[index].index("--native-threads") + 1]) for index in (3, 5)], [2, 4])
        self.assertEqual({int(command[command.index("--train-rows") + 1]) for command in commands[1:]},
                         {report["n_train"]})

    def test_unqualified_child_stops_sweep_even_with_zero_subprocess_exit_code(self):
        with tempfile.TemporaryDirectory(prefix="snmf-sweep-test-") as directory:
            result, commands, report = self.run_fake_sweep(Path(directory) / "sweep", invalid_stage=2)
        self.assertEqual(result, 2)
        self.assertEqual(len(commands), 3)
        self.assertEqual(report["status"], "failed")
        self.assertEqual([run["workers"] for run in report["runs"]], [1, 2])
        self.assertEqual(report["runs"][0]["within_pc_gain_vs_one"], 1)
        self.assertIsNone(report["runs"][1]["within_pc_gain_vs_one"])


if __name__ == "__main__":
    unittest.main()
