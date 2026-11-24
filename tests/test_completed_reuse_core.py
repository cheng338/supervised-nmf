"""Authoritative checkpoint reuse, error precedence and mapping ownership."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from supervised_nmf import study

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "examples"))
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux completed-work reuse")


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT = load("completed_reuse_core_default", "scripts/celeba_default_study.py")
PARAMETER = load("completed_reuse_core_parameter", "examples/celeba_parameter_study.py")


def hashes(path):
    return {file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file() and file.name != ".study.lock"}



@pytest.fixture
def data():
    rng = np.random.default_rng(745)
    return rng.uniform(.1, 1, (30, 8)), np.arange(30) % 2



def call(directory, data, *, reuse=False, grid=None, **kwargs):
    fn = study._calibration_knn_grid if reuse else study.calibration_knn_grid
    if reuse:
        kwargs['_reuse_only'] = True
    return fn(KNeighborsClassifier(), grid or {"n_neighbors": [1, 3]}, *data,
              cv=3, seed=17, threads=1, checkpoint_dir=directory, **kwargs)



def no_fit(*args, **kwargs):
    raise AssertionError("reuse-only path attempted representation work")



def scientific(report):
    return {key: value for key, value in report.items()
            if key not in {"new_representation_fits", "reused_representation_folds", "current_call_wall_seconds"}}



def test_completed_reuse_keeps_science_and_committed_bytes(tmp_path, data):
    first = call(tmp_path, data)
    before = hashes(tmp_path)
    original_data = [value.copy() for value in data]
    with patch.object(study, "_run_representation_fold", no_fit):
        reused = call(tmp_path, data, reuse=True)
    assert scientific(first) == scientific(reused)
    assert reused["new_representation_fits"] == 0
    assert reused["reused_representation_folds"] == 3
    after = hashes(tmp_path)
    assert {name: value for name, value in before.items() if name != "report.json"} == {
        name: value for name, value in after.items() if name != "report.json"}
    for actual, expected in zip(data, original_data):
        np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize("report_state", ["absent", "corrupt"])
def test_derived_report_is_rebuilt_from_valid_folds(tmp_path, data, report_state):
    first = call(tmp_path, data)
    path = tmp_path / "report.json"
    if report_state == "absent":
        path.unlink()
    else:
        path.write_text("not a receipt", encoding="utf-8")
    with patch.object(study, "_run_representation_fold", no_fit):
        reused = call(tmp_path, data, reuse=True)
    assert scientific(reused) == scientific(first)
    assert study._read_receipt(path) == reused



def test_missing_identity_never_publishes_or_fits(tmp_path, data):
    with patch.object(study, "_run_representation_fold", no_fit):
        assert call(tmp_path, data, reuse=True) is None
    assert hashes(tmp_path) == {}



def test_missing_fold_never_publishes_then_ordinary_path_repairs(tmp_path, data):
    call(tmp_path, data)
    target = sorted((tmp_path / "folds").glob("*.json"))[1]
    target.unlink()
    before = hashes(tmp_path)
    with patch.object(study, "_run_representation_fold", no_fit):
        assert call(tmp_path, data, reuse=True) is None
    assert hashes(tmp_path) == before
    repaired = call(tmp_path, data)
    assert repaired["new_representation_fits"] == 1
    assert repaired["reused_representation_folds"] == 2



@pytest.mark.parametrize("part", ["identity", "fold"])
def test_corrupt_authoritative_receipt_is_error_without_fallback(tmp_path, data, part):
    call(tmp_path, data)
    path = tmp_path / "identity.json" if part == "identity" else sorted((tmp_path / "folds").glob("*.json"))[0]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["sha256"] = "0" * 64
    path.write_text(json.dumps(envelope), encoding="utf-8")
    before = hashes(tmp_path)
    with patch.object(study, "_run_representation_fold", no_fit), pytest.raises(ValueError, match="integrity"):
        call(tmp_path, data, reuse=True)
    assert hashes(tmp_path) == before



@pytest.mark.parametrize("payload", [[], None, "invalid"])
def test_checksummed_non_mapping_identity_retains_value_error(tmp_path, data, payload):
    call(tmp_path, data)
    study._write_receipt(tmp_path / "identity.json", payload)
    for reuse in (False, True):
        with pytest.raises(ValueError, match="incompatible"):
            call(tmp_path, data, reuse=reuse)



@pytest.mark.parametrize("change", ["data", "seed", "scoring", "terminal_state"])
def test_real_incompatibility_is_not_missing_work(tmp_path, data, change):
    call(tmp_path, data)
    altered = data
    kwargs = {}
    if change == "data":
        altered = (data[0] + .01, data[1])
    elif change == "seed":
        identity = study._read_receipt(tmp_path / "identity.json")
        identity["calibration"]["seed"] += 1
        study._write_receipt(tmp_path / "identity.json", identity)
    elif change == "scoring":
        kwargs["scoring"] = "balanced_accuracy"
    else:
        path = sorted((tmp_path / "folds").glob("*.json"))[0]
        record = study._read_receipt(path)
        record["state"] = "running"
        study._write_receipt(path, record)
    before = hashes(tmp_path)
    for reuse in (False, True):
        with patch.object(study, "_run_representation_fold", no_fit), pytest.raises(ValueError):
            call(tmp_path, altered, reuse=reuse, **kwargs)
    assert hashes(tmp_path) == before



def test_parent_runtime_difference_only_defers_to_clean_worker(tmp_path, data):
    call(tmp_path, data)
    original = study._runtime_identity
    def extra(*args, **kwargs):
        runtime = deepcopy(original(*args, **kwargs))
        runtime["numerical_libraries"].append({"filepath": "/parent-only-library.so"})
        return runtime
    before = hashes(tmp_path)
    with patch.object(study, "_runtime_identity", extra), patch.object(study, "_run_representation_fold", no_fit):
        assert call(tmp_path, data, reuse=True) is None
        with pytest.raises(ValueError, match="incompatible"):
            call(tmp_path, data)
    assert hashes(tmp_path) == before
    assert call(tmp_path, data, reuse=True)["new_representation_fits"] == 0



def test_runtime_change_during_probe_does_not_publish(tmp_path, data):
    call(tmp_path, data)
    original = study._runtime_identity
    invocations = []
    def changing(*args, **kwargs):
        runtime = deepcopy(original(*args, **kwargs))
        invocations.append(True)
        if len(invocations) >= 2:
            runtime["numerical_libraries"].append({"filepath": "/parent-only-library.so"})
        return runtime
    before = hashes(tmp_path)
    with patch.object(study, "_runtime_identity", changing), patch.object(study, "_run_representation_fold", no_fit):
        assert call(tmp_path, data, reuse=True) is None
    assert len(invocations) == 2
    assert hashes(tmp_path) == before



def test_default_source_context_mismatch_defers_before_loading_data():
    work = ("output", {"job": {}}, {"id": "chunk"}, {"runtime": "worker"})
    with patch.object(DEFAULT, "_source_identity", return_value={"runtime": "parent"}), \
            patch.object(DEFAULT.np, "load", side_effect=AssertionError("must not load")):
        assert DEFAULT._worker(work, reuse_only=True) is None
        with pytest.raises(ValueError, match="Source or numerical runtime changed"):
            DEFAULT._worker(work)



def _worker_files(directory, attribute):
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.save(directory / "train-X.npy", X)
    np.save(directory / f"train-{attribute}-y.npy", [0, 1, 0, 1])
    return X


def _assert_detached_snapshot(directory, array, expected):
    # A caller may retain its read-only array while the original file is replaced.
    owner = array
    while isinstance(owner, np.ndarray):
        assert getattr(owner, "_mmap", None) is None
        owner = owner.base
    assert not array.flags.writeable
    original = directory / "train-X.npy"
    original.rename(directory / "original.npy")
    np.save(original, np.zeros_like(expected))
    np.testing.assert_array_equal(array, expected)


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_default_missing_probe_detaches_input_and_defers_tail_check(monkeypatch, platform, tmp_path):
    monkeypatch.setattr(DEFAULT.sys, "platform", platform)
    expected_X = _worker_files(tmp_path, "a")
    expected = {"runtime": "worker"}
    work = ("output", {"job": {"id": "job", "cache": str(tmp_path), "attribute": "a", "seed": 17}},
            {"id": "chunk", "grid": {}}, expected)
    with patch.object(DEFAULT, "_source_identity", return_value=expected), \
            patch.object(DEFAULT, "_check_source", side_effect=AssertionError("incomplete probe has no tail check")), \
            patch.object(DEFAULT, "_estimator", return_value=object()), \
            patch.object(study, "_calibration_knn_grid", return_value=None) as calibrate:
        assert DEFAULT._worker(work, reuse_only=True) is None
    _assert_detached_snapshot(tmp_path, calibrate.call_args.args[2], expected_X)



def test_lock_conflict_propagates_and_preserves_checkpoint(tmp_path, data):
    call(tmp_path, data)
    before = hashes(tmp_path)
    with study._checkpoint_lock(tmp_path), pytest.raises(RuntimeError, match="Another process"):
        call(tmp_path, data, reuse=True)
    assert hashes(tmp_path) == before



def test_invalid_terminal_folds_are_reused_without_retry(tmp_path, data):
    grid = {"n_neighbors": [100]}
    first = call(tmp_path, data, grid=grid)
    assert not first["candidates"][0]["valid"]
    with patch.object(study, "_run_representation_fold", no_fit):
        reused = call(tmp_path, data, reuse=True, grid=grid)
    assert scientific(reused) == scientific(first)
    assert reused["new_representation_fits"] == 0



@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("reuse_only", [False, True])
def test_parameter_worker_keeps_detached_input_valid(monkeypatch, platform, error, reuse_only, tmp_path):
    monkeypatch.setattr(PARAMETER.sys, "platform", platform)
    expected_X = _worker_files(tmp_path, "attribute")
    job = (str(tmp_path), "output", "attribute", "method", "chunk", {}, {}, 17, 1, "source")
    with patch.object(PARAMETER, "_source_hash", return_value="source"), \
            patch.object(PARAMETER, "study_models", return_value={"method": (object(), {})}), \
            patch.object(study, "_calibration_knn_grid",
                         return_value=None if reuse_only else {"candidates": [], "unique_representation_fits": 0,
                             "tuning_seconds": 0.0, "new_representation_fits": 0, "reused_representation_folds": 3},
                         side_effect=ValueError("validation") if error else None) as calibrate:
        if error:
            with pytest.raises(ValueError, match="validation"):
                PARAMETER._worker(job, reuse_only=reuse_only)
        else:
            result = PARAMETER._worker(job, reuse_only=reuse_only)
            assert (result is None) is reuse_only
    _assert_detached_snapshot(tmp_path, calibrate.call_args.args[2], expected_X)

