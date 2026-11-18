"""Current checkpoint consumers retain their exact-source reuse boundary."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock, patch


def _file_hashes(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*.json")}


def _exercise_current_consumers(source, output):
    """Run only in a child process: imports and source edits stay isolated."""
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    import numpy as np
    import pytest
    from threadpoolctl import threadpool_limits

    import fashion_mnist_study as fashion
    import supervised_nmf.study as study

    assert Path(fashion.__file__).is_relative_to(source)
    assert Path(study.__file__).is_relative_to(source)
    experimental = source / "scripts/experimental_nmf.py"
    package_file = source / "src/supervised_nmf/_recovery.py"
    originals = {path: path.read_bytes() for path in (experimental, package_file)}
    parent = output / "parent"
    parent_hashes = None

    def chunk_path(root, plan):
        return root / plan["job"]["id"] / plan["chunks"][0]["id"]

    def worker(root, plan, configuration):
        return fashion._worker((str(root), plan, plan["chunks"][0], configuration["sources"]))

    try:
        with threadpool_limits(limits=1):
            rng = np.random.RandomState(101)
            y = np.arange(60) % 2
            latent = rng.uniform(.25, 1., size=(60, 2))
            latent[:, 0] += 1.5 * y
            X = latent @ rng.uniform(.2, 1., size=(2, 8)) + rng.uniform(.001, .01, size=(60, 8))
            arrays = fashion._save_arrays(output / "data", X, y)
            grid = {"n_components": [2], "score_threshold": [0.], "n_neighbors": [1, 3]}
            preflight = fashion._preflight(X, y, 17, grid["score_threshold"])
            plans = []
            for name, method in (("raw", "raw_knn"), ("bpp", "screened_anls_bpp_absolute")):
                job = {"id": name, "method": method, "seed": 17, "n_train": len(y),
                       "max_iter": 3000, "tol": 1e-3, "anls_max_iter": 100,
                       "anls_nnls_max_iter": 100, "anls_nnls_tol": 1e-10,
                       "cache": arrays["path"], "training_array_record": arrays,
                       "training_data_sha256": arrays["data_sha256"]}
                plans.append(fashion._plan(job, grid, preflight))
                # Load both implementations before recording the native inventory.
                fashion._estimator(job)
            raw, bpp = plans
            configuration = {"sources": fashion._sources(), "plans": plans}
            for plan in plans:
                produced = worker(parent, plan, configuration)
                assert produced["new_representation_fits"] == 5
            original_summary = fashion.aggregate_cv(parent, configuration, require_complete=True)
            assert original_summary["state"] == "cv_completed"
            original_raw = study._read_receipt(chunk_path(parent, raw) / "report.json")
            parent_hashes = _file_hashes(parent)

            # Editing an already-loaded helper must not mint a new identity.
            experimental.write_bytes(originals[experimental] + b"\n# isolated source-identity test\n")
            with pytest.raises(ValueError, match=r"experimental_nmf\.py.*(?:changed|restart)"):
                fashion._sources()
            with pytest.raises(ValueError, match=r"experimental_nmf\.py.*(?:changed|restart)"):
                fashion._check_source(configuration["sources"])

            # The pytest parent starts the changed-source continuation in a new
            # interpreter, so it owns both subprocess lifetimes directly.
            transfer = output / "updated-source-state.json"
            transfer.write_text(json.dumps({"configuration": configuration,
                "original_raw": original_raw, "original_summary": original_summary,
                "parent_hashes": parent_hashes,
                "updated_experimental_sha256": hashlib.sha256(experimental.read_bytes()).hexdigest()}),
                encoding="utf-8")
    finally:
        for path, content in originals.items():
            path.write_bytes(content)
        assert all(path.read_bytes() == content for path, content in originals.items())
        if parent_hashes is not None:
            assert _file_hashes(parent) == parent_hashes



def _continue_current_consumers(source, output, transfer):
    """Import the changed helper freshly, then exercise every original reuse case."""
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    import pytest
    from threadpoolctl import threadpool_limits

    import fashion_mnist_study as fashion
    import supervised_nmf.study as study
    from supervised_nmf._neighbors import _KNNQueryReuse

    assert Path(fashion.__file__).is_relative_to(source)
    assert Path(study.__file__).is_relative_to(source)
    state = json.loads(transfer.read_text(encoding="utf-8"))
    configuration = state["configuration"]
    plans = configuration["plans"]
    raw, bpp = plans
    original_raw, original_summary = state["original_raw"], state["original_summary"]
    parent, parent_hashes = output / "parent", state["parent_hashes"]
    package_file = source / "src/supervised_nmf/_recovery.py"
    package_original = package_file.read_bytes()

    def chunk_path(root, plan):
        return root / plan["job"]["id"] / plan["chunks"][0]["id"]

    def worker(root, plan, configuration):
        return fashion._worker((str(root), plan, plan["chunks"][0], configuration["sources"]))

    def copy_chunk(root, plan):
        old, new = chunk_path(parent, plan), chunk_path(root, plan)
        for path in old.rglob("*.json"):
            target = new / path.relative_to(old)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        assert _file_hashes(old) == _file_hashes(new)
        return _file_hashes(new)

    try:
        with threadpool_limits(limits=1):
            # Load both implementations before recording the native inventory.
            for plan in plans:
                fashion._estimator(plan["job"])
            child_configuration = deepcopy(configuration)
            child_configuration["sources"] = fashion._sources()
            assert child_configuration["sources"]["package_source_sha256"] == configuration["sources"]["package_source_sha256"]
            assert child_configuration["sources"]["runner_source_sha256"]["experimental_nmf.py"] != configuration["sources"]["runner_source_sha256"]["experimental_nmf.py"]

            resumed_root = output / "resume"
            copied = copy_chunk(resumed_root, raw)
            nofit = Mock(side_effect=AssertionError("Complete resume attempted another fold fit"))
            noquery = Mock(side_effect=AssertionError("Complete resume attempted another query"))
            with patch.object(study, "_run_representation_fold", nofit), \
                    patch.object(_KNNQueryReuse, "predict", noquery):
                resumed = worker(resumed_root, raw, child_configuration)
            nofit.assert_not_called()
            noquery.assert_not_called()
            assert resumed["new_representation_fits"] == 0
            assert resumed["reused_representation_folds"] == 5
            resumed_report = study._read_receipt(chunk_path(resumed_root, raw) / "report.json")
            assert resumed_report["identity"] == original_raw["identity"]
            assert resumed_report["compute_seconds"] == original_raw["compute_seconds"]
            # Ordinary resume regenerates report.json; committed evidence stays intact.
            assert {key: value for key, value in _file_hashes(chunk_path(resumed_root, raw)).items()
                    if key != "report.json"} == {key: value for key, value in copied.items() if key != "report.json"}

            stale = output / "stale"
            for plan in plans:
                copy_chunk(stale, plan)
            with pytest.raises(ValueError, match="CV checkpoint identity or grid coverage differs"):
                fashion.aggregate_cv(stale, child_configuration, require_complete=True)
            with patch.object(study, "_run_representation_fold", nofit):
                with pytest.raises(ValueError, match="checkpoint is incompatible"):
                    worker(stale, bpp, child_configuration)
            nofit.assert_not_called()

            child = output / "child"
            raw_bytes = copy_chunk(child, raw)
            incomplete = fashion.aggregate_cv(child, child_configuration)
            assert incomplete["state"] == "cv_incomplete"
            assert len(incomplete["missing_chunks"]) == 1
            with pytest.raises(ValueError, match="chunks are missing"):
                fashion.aggregate_cv(child, child_configuration, require_complete=True)
            fresh_bpp = worker(child, bpp, child_configuration)
            assert fresh_bpp["new_representation_fits"] == 5
            assert fresh_bpp["reused_representation_folds"] == 0
            completed = fashion.aggregate_cv(child, child_configuration, require_complete=True)
            assert completed["state"] == "cv_completed"
            assert set(completed["jobs"]) == {"raw", "bpp"}
            assert _file_hashes(chunk_path(child, raw)) == raw_bytes
            assert completed["jobs"]["raw"]["compute_seconds"] == original_summary["jobs"]["raw"]["compute_seconds"]

            # Even an unrelated package-file edit invalidates old raw-KNN evidence.
            package_file.write_bytes(package_original + b"\n# isolated package-identity test\n")
            changed_package = deepcopy(child_configuration)
            changed_package["sources"] = fashion._sources()
            with patch.object(study, "_run_representation_fold", nofit):
                with pytest.raises(ValueError, match="checkpoint is incompatible"):
                    worker(resumed_root, raw, changed_package)
            nofit.assert_not_called()
            with pytest.raises(ValueError, match="CV checkpoint identity or grid coverage differs"):
                fashion.aggregate_cv(child, changed_package, require_complete=True)
    finally:
        package_file.write_bytes(package_original)
        assert package_file.read_bytes() == package_original
        assert _file_hashes(parent) == parent_hashes

def test_current_consumers_preserve_compatible_receipts_and_reject_changed_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    private = tmp_path / "source"
    files = list((root / "src/supervised_nmf").rglob("*.py"))
    files += [root / "scripts" / name for name in
              ("fashion_mnist_study.py", "profile_celeba_stages.py", "experimental_nmf.py")]
    original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for path in files:
        copied = private / path.relative_to(root)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    output = tmp_path / "runs"
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(private), str(output)]
    private_originals = {private / path.relative_to(root): path.read_bytes() for path in files}
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout) == {"current_consumers_completed": True}
        transfer = output / "updated-source-state.json"
        state = json.loads(transfer.read_text(encoding="utf-8"))
        experimental = private / "scripts/experimental_nmf.py"
        experimental.write_bytes(private_originals[experimental] + b"\n# isolated source-identity test\n")
        assert hashlib.sha256(experimental.read_bytes()).hexdigest() == state["updated_experimental_sha256"]
        continuation = subprocess.run(command + ["--continue", str(transfer)],
                                      env=environment, capture_output=True, text=True, timeout=120)
        assert continuation.returncode == 0, continuation.stdout + continuation.stderr
        assert json.loads(continuation.stdout) == {"updated_consumers_completed": True}
    finally:
        for path, content in private_originals.items():
            path.write_bytes(content)
        assert all(path.read_bytes() == content for path, content in private_originals.items())
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in original_hashes.items())


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "--continue":
        _continue_current_consumers(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4]))
        print(json.dumps({"updated_consumers_completed": True}))
    else:
        _exercise_current_consumers(Path(sys.argv[1]), Path(sys.argv[2]))
        print(json.dumps({"current_consumers_completed": True}))
