"""No-fit checks for reusing benchmark results after reporting-only updates."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "portable_manifest_reuse_test_module", ROOT / "portable_speedtest" / "compare.py"
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest_fixture():
    paths = (
        "benchmark.py", "sweep.py", "check_compatibility.py", "protocol.json",
        "data/workload.npz", "data/workload-metadata.json",
        "package/supervised_nmf/__init__.py", "package/supervised_nmf/classifier.py",
        "package/pyproject.toml", "package/README.md", "RUN_SWEEP.sh",
        "requirements-reference.txt", "README.md", "compare.py",
    )
    return {"schema_version": 1, "hash_algorithm": "sha256",
            "files": {path: digest("fixture content: " + path) for path in paths}}


def write_receipts(root, previous, current):
    """Store hash-addressed historical metadata plus the current manifest."""
    previous_bytes = json.dumps(previous, sort_keys=True).encode("utf-8")
    current_bytes = json.dumps(current, sort_keys=True).encode("utf-8")
    previous_hash = hashlib.sha256(previous_bytes).hexdigest()
    current_hash = hashlib.sha256(current_bytes).hexdigest()
    receipts = root / "compatible_manifests"
    receipts.mkdir(exist_ok=True)
    (receipts / (previous_hash + ".json")).write_bytes(previous_bytes)
    (root / "manifest.json").write_bytes(current_bytes)
    return previous_hash, current_hash


def report_fixture(manifest_hash):
    return {
        "schema_version": 1, "label": "Fixture PC", "status": "qualified",
        "manifest_sha256": manifest_hash,
        "protocol_sha256": manifest_fixture()["files"]["protocol.json"],
        "configuration": {
            "mode": "capacity", "duration_seconds": 60, "repeats": 3,
            "workers": 16, "native_threads_per_worker": 1,
            "n_train": 2048, "n_features": 512, "n_test": 256,
            "model": {"solver": "cd", "screening": "absolute", "score_threshold": 2.0},
        },
        "aggregate": {"qualified_fits_per_second_median": 2.0,
                      "qualified_input_values_per_second_median": 2.0 * 2048 * 512},
        "provenance": {"python": "fixture", "versions": {}, "platform": "fixture"},
        "repeats": [{"status": "qualified", "records": []} for _ in range(3)],
    }


class PortableManifestReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="snmf-manifest-reuse-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previous = manifest_fixture()
        self.current = deepcopy(self.previous)
        self.current["files"]["README.md"] = digest("new projection instructions")

    def compare_manifests(self, previous=None, current=None):
        first_hash, second_hash = write_receipts(
            self.root, self.previous if previous is None else previous,
            self.current if current is None else current,
        )
        return compare.compatible_benchmark_manifests(first_hash, second_hash,
                                                     bundle_root=self.root)

    def test_reporting_changes_allow_reuse_in_both_directions(self):
        reporting_paths = (
            "compare.py", "projection.py", "pc_profiles.json", "builder-provenance.json",
            "README.md", "PROTOCOL.md", "PROMPT_FOR_OTHER_PC.txt", "VALIDATION.md",
            "compatible_manifests/older-receipt.json",
        )
        for operation in ("add", "change", "remove"):
            with self.subTest(operation=operation):
                previous, current = deepcopy(self.previous), deepcopy(self.current)
                for path in reporting_paths:
                    if operation == "add":
                        previous["files"].pop(path, None)
                        current["files"][path] = digest("new " + path)
                    else:
                        previous["files"][path] = digest("old " + path)
                        if operation == "change":
                            current["files"][path] = digest("new " + path)
                        else:
                            current["files"].pop(path, None)
                old_hash, new_hash = write_receipts(self.root, previous, current)
                self.assertTrue(compare.compatible_benchmark_manifests(
                    old_hash, new_hash, bundle_root=self.root))
                self.assertTrue(compare.compatible_benchmark_manifests(
                    new_hash, old_hash, bundle_root=self.root))

    def test_changes_to_workload_package_environment_or_launchers_block_reuse(self):
        for path in self.previous["files"]:
            if path in ("README.md", "compare.py"):
                continue
            for operation in ("change", "remove"):
                with self.subTest(path=path, operation=operation):
                    changed = deepcopy(self.current)
                    if operation == "change":
                        changed["files"][path] = digest("modified workload")
                    else:
                        del changed["files"][path]
                    self.assertFalse(self.compare_manifests(current=changed))

    def test_new_unrecognized_files_are_not_silently_treated_as_reporting(self):
        for path in ("new_runner.py", "data/new-workload.npz", "package/projection.py",
                     "package/README.md", "compatible_manifests_extra/receipt.json"):
            with self.subTest(path=path):
                changed = deepcopy(self.current)
                changed["files"][path] = digest("new unreviewed content")
                self.assertFalse(self.compare_manifests(current=changed))

    def test_unknown_and_corrupt_receipts_fail_closed(self):
        old_hash, new_hash = write_receipts(self.root, self.previous, self.current)
        unknown = digest("unavailable manifest")
        self.assertFalse(compare.compatible_benchmark_manifests(
            unknown, new_hash, bundle_root=self.root))
        old_path = self.root / "compatible_manifests" / (old_hash + ".json")
        for content in (b"invalid json", json.dumps(self.current).encode("utf-8")):
            with self.subTest(content=content[:20]):
                old_path.write_bytes(content)
                self.assertFalse(compare.compatible_benchmark_manifests(
                    old_hash, new_hash, bundle_root=self.root))
        write_receipts(self.root, self.previous, self.current)
        (self.root / "manifest.json").write_text("{}", encoding="utf-8")
        self.assertFalse(compare.compatible_benchmark_manifests(
            old_hash, new_hash, bundle_root=self.root))

    def test_invalid_manifest_schema_or_missing_core_files_cannot_authorize_reuse(self):
        invalid_manifests = []
        for key, value in (("schema_version", 2), ("hash_algorithm", "sha1"),
                           ("files", []), ("files", {})):
            invalid = deepcopy(self.previous)
            invalid[key] = value
            invalid_manifests.append((key + "=" + str(value), invalid))
        for path in ("benchmark.py", "sweep.py", "check_compatibility.py", "protocol.json",
                     "data/workload.npz", "package/supervised_nmf/__init__.py"):
            invalid = deepcopy(self.previous)
            del invalid["files"][path]
            invalid_manifests.append(("missing " + path, invalid))
        for label, invalid in invalid_manifests:
            with self.subTest(invalid=label):
                updated = deepcopy(invalid)
                if isinstance(updated.get("files"), dict):
                    updated["files"]["README.md"] = digest("report update")
                else:
                    updated["note"] = "report update"
                self.assertFalse(self.compare_manifests(previous=invalid, current=updated))

    def test_invalid_hashes_are_rejected_even_when_both_receipts_agree(self):
        for invalid_hash in (None, 123, "", "0" * 63, "g" * 64, "A" * 64):
            with self.subTest(hash=invalid_hash):
                previous, current = deepcopy(self.previous), deepcopy(self.current)
                previous["files"]["benchmark.py"] = invalid_hash
                current["files"]["benchmark.py"] = invalid_hash
                self.assertFalse(self.compare_manifests(previous=previous, current=current))
                self.assertFalse(compare.compatible_benchmark_manifests(
                    invalid_hash, invalid_hash, bundle_root=self.root))

    def test_assessment_allows_verified_reuse_with_a_visible_qualification(self):
        old_hash, new_hash = write_receipts(self.root, self.previous, self.current)
        first, second = report_fixture(old_hash), report_fixture(new_hash)
        baseline = compare.assess(first, first, bundle_root=self.root)
        decision = compare.assess(first, second, bundle_root=self.root)
        self.assertTrue(decision["comparable_protocol"])
        self.assertFalse(decision["blockers"])
        self.assertEqual(decision["ratios"], baseline["ratios"])
        self.assertGreater(len(decision["qualifications"]), len(baseline["qualifications"]))

    def test_metadata_reuse_does_not_override_protocol_or_measurement_checks(self):
        old_hash, new_hash = write_receipts(self.root, self.previous, self.current)
        first = report_fixture(old_hash)
        for mutation in ("protocol", "missing-protocol", "smoke", "incomplete-windows"):
            with self.subTest(mutation=mutation):
                second = report_fixture(new_hash)
                if mutation == "protocol":
                    second["protocol_sha256"] = digest("different protocol")
                elif mutation == "missing-protocol":
                    del second["protocol_sha256"]
                elif mutation == "smoke":
                    second["smoke"] = True
                else:
                    second["repeats"].pop()
                decision = compare.assess(first, second, bundle_root=self.root)
                self.assertFalse(decision["comparable_protocol"])
                self.assertTrue(decision["blockers"])
                self.assertFalse(decision["ratios"])

    def test_unknown_differing_identity_blocks_assessment_without_receipts(self):
        first, second = report_fixture(digest("old")), report_fixture(digest("new"))
        decision = compare.assess(first, second, bundle_root=self.root)
        self.assertFalse(decision["comparable_protocol"])
        self.assertTrue(any("manifest_sha256" in item for item in decision["blockers"]))
        self.assertFalse(decision["ratios"])

    def test_identical_report_identities_do_not_require_historical_receipts(self):
        report = report_fixture(digest("same frozen bundle"))
        with patch.object(compare, "compatible_benchmark_manifests") as checker:
            decision = compare.assess(report, deepcopy(report), bundle_root=self.root)
        self.assertTrue(decision["comparable_protocol"])
        checker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
