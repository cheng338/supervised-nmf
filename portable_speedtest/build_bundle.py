"""Freeze the unpublished sNMF source and benchmark into a verified ZIP.

Run this from a suitable development Python environment. Nothing is installed,
downloaded, or changed in src/. The finished bundle uses its own source snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEPENDENCIES = ("numpy", "scipy", "scikit-learn", "threadpoolctl", "joblib", "psutil")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inputs():
    result = {}
    for name in ("benchmark.py", "compare.py", "projection.py", "pc_profiles.json", "check_compatibility.py", "sweep.py", "README.md", "PROTOCOL.md", "VALIDATION.md",
                 "RUN_BENCHMARK.cmd", "RUN_BENCHMARK.sh", "RUN_SWEEP.cmd", "RUN_SWEEP.sh", "PROMPT_FOR_OTHER_PC.txt"):
        result[Path(name)] = HERE / name
    for name in ("README.md", "CITATION.cff", "pyproject.toml"):
        result[Path("package") / name] = REPO / name
    for path in sorted((REPO / "docs").rglob("*.md")):
        result[Path("package") / "docs" / path.relative_to(REPO / "docs")] = path
    for path in sorted((REPO / "src" / "supervised_nmf").rglob("*.py")):
        result[Path("package") / "supervised_nmf" / path.relative_to(REPO / "src" / "supervised_nmf")] = path
    for path in sorted((REPO / "provenance").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            result[Path("package") / "provenance" / path.relative_to(REPO / "provenance")] = path
    for name in ("environment-target.txt", "environment-current.txt"):
        if (HERE / name).is_file():
            result[Path(name)] = HERE / name
    for path in sorted((HERE / "validation").glob("*.json")):
        result[Path("validation") / path.name] = path
    for path in sorted((HERE / "compatible_manifests").glob("*.json")):
        if sha256(path) != path.stem:
            raise RuntimeError("Archived manifest hash does not match its filename: " + str(path))
        result[Path("compatible_manifests") / path.name] = path
    if not any(relative.parts[:2] == ("package", "supervised_nmf") for relative in result):
        raise RuntimeError("No supervised_nmf Python source files found.")
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required bundle input is not ready: {path}")
    return result


def snapshot(paths):
    return {relative.as_posix(): sha256(path) for relative, path in paths.items()}


def git_provenance():
    def run(*args):
        try:
            completed = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True)
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def environment():
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import numpy  # Load numerical libraries so threadpool_info reports the actual backend.
    import scipy.linalg
    import sklearn
    from threadpoolctl import threadpool_info
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "versions": versions,
        "thread_pools": threadpool_info(),
    }


def protocol():
    return {
        "schema_version": 1,
        "benchmark_id": "snmf-cd-absolute-capacity-v1",
        "default_mode": "capacity",
        "duration_seconds": 60,
        "repeats": 3,
        "model": {
            "solver": "cd",
            "screening": "absolute",
            "n_components": 12,
            "score_threshold": 2.0,
            "n_neighbors": 5,
            "max_iter": 20000,
            "tol": 1e-4,
            "random_state": 43,
            "training_coordinates": "fit_transform",
        },
        "capacity": {
            "generator": "original_simulation_v1",
            "input_features": 2048,
            "signal_ratio": 0.25,
            "test_rows": 256,
            "bytes_per_train_feature": 40,
            "worker_overhead_bytes": 268435456,
            "data_seed": 43,
        },
        "fixture": {
            "path": "data/workload.npz",
            "generator": "supervised_nmf.datasets.make_original_simulation",
            "train_rows": 768,
            "test_rows": 256,
            "input_features": 512,
            "signal_ratio": 0.25,
            "random_state": 42,
            "split_random_state": 43,
        },
    }


def write_fixture(destination, configuration):
    import numpy as np
    # Import the frozen source to generate the frozen data, never an installed package.
    sys.path.insert(0, str(destination / "package"))
    sys.dont_write_bytecode = True
    from supervised_nmf.datasets import make_original_simulation
    module_file = Path(sys.modules[make_original_simulation.__module__].__file__).resolve()
    if destination.resolve() not in module_file.parents:
        raise RuntimeError("The workload generator did not come from the bundle's source snapshot.")
    settings = configuration["fixture"]
    n_train, n_test = settings["train_rows"], settings["test_rows"]
    generated = make_original_simulation(
        n_samples=n_train + n_test, n_features=settings["input_features"],
        signal_ratio=settings["signal_ratio"], random_state=settings["random_state"],
    )
    rng = np.random.RandomState(settings["split_random_state"])
    halves = [rng.permutation(np.flatnonzero(generated.target == label)) for label in (0, 1)]
    train = rng.permutation(np.concatenate([part[:n_train // 2] for part in halves]))
    test = rng.permutation(np.concatenate([part[n_train // 2:] for part in halves]))
    data_dir = destination / "data"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "workload.npz", X_train=generated.data[train], y_train=generated.target[train],
        X_test=generated.data[test], y_test=generated.target[test],
    )
    write_json(data_dir / "workload-metadata.json", {
        "generator_metadata": generated.metadata,
        "split": "Seeded balanced stratified split, with shuffled train/test row order.",
        "settings": settings,
        "purpose": "Identical validation and optional fixed-workload baseline on both PCs; capacity mode generates larger real workloads.",
    })


def verify_zip(archive, destination, manifest):
    expected = set(manifest["files"]) | {"manifest.json"}
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise RuntimeError("ZIP file list does not exactly match the bundle manifest.")
        if zipped.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed.")
        for name in names:
            digest = hashlib.sha256()
            with zipped.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            expected_hash = sha256(destination / name) if name == "manifest.json" else manifest["files"][name]
            if digest.hexdigest() != expected_hash or sha256(destination / name) != expected_hash:
                raise RuntimeError(f"ZIP differs from the verified directory: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "dist")
    parser.add_argument("--name", default=None, help="New destination directory name; existing paths are never overwritten.")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or f"snmf-pc-speedtest-{timestamp}"
    if not name or Path(name).name != name or name in (".", ".."):
        parser.error("--name must be a single new directory name.")
    destination = args.output_root.resolve() / name
    archive = destination.parent / (destination.name + ".zip")
    if destination.exists() or archive.exists():
        parser.error(f"Refusing to overwrite an existing destination: {destination} or {archive}")
    paths = inputs()
    source_before = snapshot(paths)
    builder_hash = sha256(Path(__file__))
    provenance = environment()
    destination.mkdir(parents=True, exist_ok=False)
    for relative, path in paths.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if snapshot(inputs()) != source_before:
        raise RuntimeError("Source inputs changed during copying. Incomplete output preserved for inspection; rebuild under a new name.")
    for relative, expected in source_before.items():
        if sha256(destination / relative) != expected:
            raise RuntimeError(f"Copied source hash mismatch: {relative}")
    settings = protocol()
    write_json(destination / "protocol.json", settings)
    write_fixture(destination, settings)
    write_json(destination / "builder-provenance.json", {
        "built_at_utc": timestamp,
        "builder_sha256": builder_hash,
        "source_inputs_sha256": source_before,
        "git": git_provenance(),
        "environment": provenance,
        "python_runtime_included": False,
        "dependency_binaries_included": False,
        "authorship": "Chang Cheng; see package/CITATION.cff and package/provenance/README.md.",
    })
    references = [
        "# Builder environment reference ONLY; not a cross-platform installation lockfile.",
        "# Use each PC's existing suitable environment. No pip installation of sNMF is needed.",
        "# Numerical/BLAS builds (including AOCL/OpenBLAS) are not reproduced by these version pins.",
        "# psutil is optional when the platform has a supported native resource probe.",
    ]
    references.extend(f"{name}=={version}" if version is not None else f"# {name}: not installed in builder environment"
                      for name, version in provenance["versions"].items())
    (destination / "requirements-reference.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
    source_after = snapshot(inputs())
    if source_after != source_before or sha256(Path(__file__)) != builder_hash:
        raise RuntimeError("Source inputs changed during the build. Rebuild under a new name after edits finish.")
    members = {path.relative_to(destination).as_posix(): sha256(path)
               for path in sorted(destination.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}
    manifest = {"schema_version": 1, "hash_algorithm": "sha256", "files": members,
                "note": "manifest.json excludes its own hash; all other distributed files are listed. Runtime results are not part of the frozen bundle."}
    write_json(destination / "manifest.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for relative in sorted(set(members) | {"manifest.json"}):
            zipped.write(destination / relative, arcname=relative)
    verify_zip(archive, destination, manifest)
    print(json.dumps({"directory": str(destination), "zip": str(archive), "zip_sha256": sha256(archive),
                      "verified_files": len(members), "source_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
