"""Released CelebA worker/aggregation must retain its declared CV objective."""

import importlib.util
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from supervised_nmf.evaluation import data_fingerprint, provenance


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "released_celeba_parameter_consumer", ROOT / "examples/celeba_parameter_study.py")
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)
REVIEW_SPEC = importlib.util.spec_from_file_location(
    "released_celeba_cv_reviewer", ROOT / "scripts/review_celeba_cv.py")
reviewer = importlib.util.module_from_spec(REVIEW_SPEC)
sys.modules[REVIEW_SPEC.name] = reviewer
REVIEW_SPEC.loader.exec_module(reviewer)


def test_real_worker_aggregation_and_resume_keep_frozen_balanced_accuracy(tmp_path, monkeypatch):
    # Exercise the actual cached-array worker, five-fold search, receipt reader,
    # aggregation and final prediction on a bounded synthetic fixture. No image
    # download or historical study is involved.
    cache, output = tmp_path / "cache", tmp_path / "study"
    cache.mkdir()
    rows = np.arange(60, dtype=float)
    y = np.tile([0, 0, 0, 1], 15)
    X = np.column_stack([3 * y + rows / 1000, 2 + np.sin(rows)])
    Xv, yv = X[:16].copy(), y[:16].copy()
    np.save(cache / "train-X.npy", X)
    np.save(cache / "train-Smiling-y.npy", y)
    np.save(cache / "validation-X.npy", Xv)
    np.save(cache / "validation-Smiling-y.npy", yv)
    models = {"raw_knn": (KNeighborsClassifier(), {"n_neighbors": [1, 3]})}
    monkeypatch.setattr(study, "ATTRIBUTES", ("Smiling",))
    monkeypatch.setattr(reviewer, "ATTRIBUTES", ("Smiling",))
    monkeypatch.setattr(study, "study_models", lambda settings, seed: models)
    manifest = {"splits": {
        "train": {"attributes": {"Smiling": {"data_sha256": data_fingerprint(X, y)}}},
        "validation": {"image_ids": [f"fixture-{index}" for index in range(len(Xv))]},
    }}
    configuration = {"script_sha256": study._source_hash(),
                     "source_sha256": provenance(1)["source_sha256"],
                     "selection_metric": "balanced_accuracy", "seed": 17, "cv": 5, "threads": 1}
    job = (cache, output, "Smiling", "raw_knn", "chunk-000", {"n_neighbors": [1, 3]},
           {}, 17, 1, configuration["script_sha256"])
    with threadpool_limits(limits=1):
        worker = study._worker(job)
        assert worker["valid_candidates"] == worker["candidates"] == 2
        report_path = output / "Smiling/raw_knn/chunk-000/report.json"
        report = study._read_report(report_path)
        assert report["scoring"] == "balanced_accuracy"
        assert report["identity"]["calibration"]["scoring"] == "balanced_accuracy"
        assert all(row["mean_score"] == row["mean_balanced_accuracy"] for row in report["candidates"])
        summary = study.aggregate_and_evaluate(output, cache, models, manifest, 1, configuration)
        result = summary["Smiling"]["models"]["raw_knn"]
        assert result["evaluation"]["status"] == "valid"
        assert result["cv_balanced_accuracy"] == max(row["mean_balanced_accuracy"] for row in report["candidates"])
        assert result["selected_parameters"] == report["selected_parameters"]
        with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("Completed resume reached fit")) as fit:
            study._worker(job)
            resumed = study.aggregate_and_evaluate(output, cache, models, manifest, 1, configuration)
            fit.assert_not_called()
        assert resumed == summary

    chunks = {"raw_knn": [("chunk-000", {"n_neighbors": [1, 3]})]}

    def collect():
        return reviewer._collect(output, configuration, models, chunks, {})

    def save(payload):
        payload["fingerprint"] = study._digest(payload["identity"])
        study.write_json(report_path, {"payload": payload, "sha256": study._digest(payload)})

    current_candidates, _, _, count = collect()
    assert count == 1
    assert len(current_candidates["Smiling"]["raw_knn"]) == 2

    def relocate(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                return {"path": "/home/robust/install/" + value["path"].replace("\\", "/").rsplit("/", 1)[-1],
                        "sha256": value["sha256"]}
            return {key: relocate(item) for key, item in value.items()}
        return [relocate(item) for item in value] if isinstance(value, list) else value

    remote = relocate(deepcopy(report))
    for inventory in (remote["identity"]["calibration"]["implementation_source"],
                      remote["identity"]["runtime"]["estimator_source"]):
        replaced = {name.partition("@")[0] + "@/home/robust/install/" + name.rsplit("/", 1)[-1]: digest
                    for name, digest in inventory.items()}
        inventory.clear()
        inventory.update(replaced)
    save(remote)
    assert collect()[0] == current_candidates
    # Recompute receipt checksums after each alteration: the reader must verify
    # semantic identities and score aliases, not merely JSON integrity.
    alterations = [
        lambda value: value.update(scoring="accuracy"),
        lambda value: value["identity"]["calibration"].update(structural_configuration={}),
        lambda value: value["identity"]["calibration"].update(implementation_source={}),
        lambda value: value["identity"]["runtime"].update(estimator_source={}),
        lambda value: value["identity"]["calibration"].update(identity_schema=99),
        lambda value: value["identity"]["calibration"].pop("identity_schema"),
        lambda value: value["candidates"][0].update(mean_score=-1.),
        lambda value: value["candidates"][0]["folds"][0].update(score=-1.),
        lambda value: value.update(selected_mean_score=-1.),
    ]
    for alter in alterations:
        changed = deepcopy(report)
        alter(changed)
        save(changed)
        with pytest.raises(ValueError):
            collect()

    # A local fixture of the released identity shape exercises the preserved
    # historical branch; no actual historical receipt is converted or replaced.
    legacy = deepcopy(report)
    legacy["schema_version"] = 3
    legacy["identity"]["schema_version"] = 3
    calibration = legacy["identity"]["calibration"]
    for key in ("identity_schema", "structural_configuration", "implementation_source", "scoring"):
        calibration.pop(key)
    estimator = models["raw_knn"][0]
    calibration["parameters"] = reviewer._jsonable(estimator.get_params(deep=True))
    calibration["candidates"] = [value["parameters"] for value in report["candidates"]]
    legacy["identity"].pop("scoring")
    legacy.pop("scoring")
    legacy.pop("selected_mean_score")
    legacy.pop("selected_mean_balanced_accuracy")
    sources = {}
    for value in (estimator, *estimator.get_params(deep=True).values()):
        if hasattr(value, "get_params"):
            path = inspect.getsourcefile(type(value))
            if path and Path(path).is_file():
                sources[f"{type(value).__module__}.{type(value).__qualname__}"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    legacy["identity"]["runtime"]["estimator_source"] = sources
    for candidate in legacy["candidates"]:
        candidate.pop("mean_score")
        for fold in candidate["folds"]:
            fold.pop("score")
    for version in (1, 3):
        legacy["schema_version"] = legacy["identity"]["schema_version"] = version
        save(legacy)
        assert collect()[0] == current_candidates
        # A schema-4 outer marker must not enter the legacy interpretation
        # when its structural identity and metric fields are absent.
        mixed = deepcopy(legacy)
        mixed["schema_version"] = 4
        save(mixed)
        with pytest.raises(ValueError, match="schema"):
            collect()
    legacy["identity"]["calibration"]["parameters"]["n_neighbors"] = 99
    save(legacy)
    with pytest.raises(ValueError, match="grid identity"):
        collect()
    save(report)
