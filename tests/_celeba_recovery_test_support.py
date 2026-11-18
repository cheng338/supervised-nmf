"""Shared builders and fixtures, independent of collected recovery test modules."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf.evaluation import data_fingerprint, provenance


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "celeba_final_recovery_consumer", ROOT / "examples/celeba_parameter_study.py")
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)
recovery = study.recovery_store
codec = importlib.import_module("numeric_codec")


def receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))["payload"]


def hashes(directory):
    return {p.relative_to(directory): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def make_cv(root, neighbors=1, raw_cache=False):
    cache, output = root / "c", root / "s"
    cache.mkdir()
    y = np.tile(np.array([0, 1], dtype=np.int64), 30)
    rows = np.arange(len(y), dtype=float)
    X = np.column_stack((3 * y + rows / 1000, 2 + np.sin(rows)))
    if raw_cache:
        X = np.tile(X, (1, 2048)).astype("float32")
    Xv, yv = X[:12].copy(), y[:12].copy()
    for name, array in (("train-X", X), ("train-Smiling-y", y),
                        ("validation-X", Xv), ("validation-Smiling-y", yv)):
        np.save(cache / (name + ".npy"), array, allow_pickle=False)
    models = {"raw_knn": (KNeighborsClassifier(n_jobs=1), {"n_neighbors": [neighbors]})}
    manifest = {"splits": {
        "train": {"attributes": {"Smiling": {"data_sha256": data_fingerprint(X, y)}}},
        "validation": {"image_ids": [f"validation-{i}" for i in range(len(yv))]},
    }}
    config = {"script_sha256": study._source_hash(), "source_sha256": provenance(1)["source_sha256"],
              "selection_metric": "balanced_accuracy", "seed": 17, "cv": 5, "threads": 1,
              "data_manifest_sha256": study._digest(manifest)}
    with patch.object(study, "study_models", return_value=models), threadpool_limits(limits=1):
        study._worker((cache, output, "Smiling", "raw_knn", "chunk-000", {"n_neighbors": [neighbors]},
                       {}, 17, 1, config["script_sha256"]))
    return SimpleNamespace(cache=cache, output=output, models=models, manifest=manifest,
                           config=config, X=X, y=y, yv=yv)


def copy_case(completed_cv, tmp_path_factory, monkeypatch):
    # Short roots avoid coupling Windows payload paths to descriptive test names.
    value = deepcopy(completed_cv)
    value.output = tmp_path_factory.mktemp("fr") / "s"
    shutil.copytree(completed_cv.output, value.output)
    value.directory = value.output / "Smiling" / "raw_knn"
    value.final = value.directory / "development-evaluation.json"
    value.fit = value.directory / ".fit-recovery"
    value.report = value.directory / "chunk-000" / "report.json"
    monkeypatch.setattr(study, "ATTRIBUTES", ("Smiling",))
    return value


def aggregate(case):
    with threadpool_limits(limits=1):
        summary = study.aggregate_and_evaluate(case.output, case.cache, case.models,
                                              case.manifest, 1, case.config)
    return summary["Smiling"]["models"]["raw_knn"]


def committed(case):
    pointer = receipt(case.fit / "current.json")
    directory = case.fit / "generations" / pointer["generation"]
    return directory, receipt(directory / "outcome.json")


def interrupt_after_commit(case):
    def stop(event, details=None):
        if event == "manifest-commit":
            raise KeyboardInterrupt("stop after the completed outcome commits")
    with patch.object(recovery, "_hook", side_effect=stop), pytest.raises(KeyboardInterrupt):
        aggregate(case)
    assert not case.final.exists()
    return committed(case)


@pytest.fixture
def recovery_module():
    return recovery

@pytest.fixture
def inventory_runtime():
    # A different interpreter/threadpoolctl implementation must take the
    # supported public fallback. Mechanism-specific cases apply only when the
    # running source actually qualifies the optional discovery optimization.
    recovery_module = recovery
    state = {}
    direct = recovery_module.active_runtime_identity(1)
    assert recovery_module.active_runtime_identity(1, _initialization_cache=state) == direct
    assert recovery_module.active_runtime_identity(1, _initialization_cache=state) == direct
    if not state.get('_inventory_state', {}).get('qualified', False):
        pytest.skip('This runtime uses the public inventory fallback')
    return recovery_module

@pytest.fixture(scope='module')
def guard_support():
    return sys.modules[__name__]

@pytest.fixture(scope='module')
def guard_cv(guard_support, tmp_path_factory):
    return guard_support.make_cv(tmp_path_factory.mktemp('gcv'))

@pytest.fixture
def guard_cases(guard_support, guard_cv, tmp_path_factory, monkeypatch):
    return [guard_support.copy_case(guard_cv, tmp_path_factory, monkeypatch) for _ in range(2)]
