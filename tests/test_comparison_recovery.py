"""Recover an interrupted comparison and append a model in a fresh interpreter."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import ExtraTreeClassifier

from supervised_nmf import evaluation


class FittedProjection(TransformerMixin, BaseEstimator):
    fits = []

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y=None):
        self.fits.append((self.seed, len(X)))
        self.projection_ = np.random.RandomState(self.seed).normal(size=(X.shape[1], 4))
        return self

    def transform(self, X):
        return X @ self.projection_


def compare(folder, names=("a", "b"), *, split_seed=None):
    rng = np.random.RandomState(894)
    X = rng.normal(size=(103, 7))
    y = (1.4 * X[:, 0] + X[:, 2] * X[:, 3] + rng.normal(size=len(X)) > .1).astype(int)
    models = {}
    for name in names:
        seed = {"a": 1234, "b": 2345}[name]
        model = Pipeline([("map", FittedProjection(seed)),
                          ("tree", ExtraTreeClassifier(random_state=seed + 1))])
        models[name] = (model, {"tree__max_depth": [1, 3]})
    return evaluation.run_comparison(models, X[:73], y[:73], X[73:], y[73:],
                                     seed=split_seed, threads=1, output_dir=folder)


class Interrupted(BaseException):
    pass


def read(path):
    return json.loads(path.read_bytes())


def initial_runs(root):
    np.random.seed(71)
    compare(root / "joint")
    full_fits = FittedProjection.fits.copy()
    assert len(full_fits) == 22
    assert [seed for seed, rows in full_fits if rows == 73] == [1234, 2345]
    np.random.seed(71)
    FittedProjection.fits.clear()
    compare(root / "append", names=("a",))
    assert FittedProjection.fits == full_fits[:11]
    np.random.seed(71)
    FittedProjection.fits.clear()
    write = evaluation.write_json

    def stop_at_second_fold(path, report):
        write(path, report)
        if Path(path).name == "a-cv.json" and len(report["candidates"][0]["folds"]) == 2:
            raise Interrupted

    with patch.object(evaluation, "write_json", stop_at_second_fold), pytest.raises(Interrupted):
        compare(root / "partial")
    assert FittedProjection.fits == full_fits[:2]
    (root / "fits.json").write_text(json.dumps(full_fits), encoding="utf-8")


def resumed_runs(root):
    np.random.seed(9001)
    expected = [tuple(row) for row in read(root / "fits.json")]
    with patch.object(StratifiedKFold, "split", side_effect=AssertionError("saved split required")):
        compare(root / "partial")
        assert FittedProjection.fits == expected[2:]
        FittedProjection.fits.clear()
        compare(root / "append")
        assert FittedProjection.fits == expected[11:]


def scientific_results(folder, name):
    cv = read(folder / f"{name}-cv.json")
    record = read(folder / f"{name}-evaluation.json")
    assert record == read(folder / "comparison.json")["models"][name]
    assert not record["is_fallback"] and record["evaluation"]["status"] == "valid"
    assert record["selected_parameters"] == cv["selected_parameters"]
    predictions = record["evaluation"]["predictions"]
    assert len(predictions["predicted_labels"]) == len(predictions["positive_class_probability"]) == 30
    scores = [[fold["score"] for fold in candidate["folds"]] for candidate in cv["candidates"]]
    assert [len(row) for row in scores] == [5, 5]
    return (cv["folds"], scores, cv["selected_candidate_index"], cv["selected_parameters"],
            predictions["predicted_labels"], predictions["positive_class_probability"])


def test_saved_comparison_recovers_fitted_projections_and_appends_model(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(evaluation.__file__).resolve().parents[1])
    command = [sys.executable, str(Path(__file__).resolve()), str(tmp_path)]
    subprocess.run([*command, "initial"], env=env, check=True, timeout=120)
    partial, append, joint = (tmp_path / name for name in ("partial", "append", "joint"))
    prefix = read(partial / "a-cv.json")["candidates"][0]["folds"]
    archived = {name: (append / name).read_bytes() for name in ("a-cv.json", "a-evaluation.json")}
    plan_name = "comparison-folds.json"
    receipt_name = "comparison-folds-receipt.json"
    plan = read(joint / plan_name)
    assert read(joint / receipt_name)["fold_plan_sha256"] == plan["sha256"]
    subprocess.run([*command, "resume"], env=env, check=True, timeout=120)
    assert read(partial / "a-cv.json")["candidates"][0]["folds"][:2] == prefix
    assert all((append / name).read_bytes() == contents for name, contents in archived.items())
    for folder in (partial, append):
        assert read(folder / plan_name) == plan
        assert read(folder / receipt_name) == read(joint / receipt_name)
        for name in ("a", "b"):
            observed = scientific_results(folder, name)
            assert observed == scientific_results(joint, name)
            assert [fold["validation_indices"] for fold in observed[0]] == plan["validation_indices"]


@pytest.mark.parametrize("damage", ["missing-plan", "missing-receipt", "replaced-partition"])
def test_disjoint_model_cannot_start_from_an_unbound_partition(tmp_path, damage, monkeypatch):
    monkeypatch.setattr(FittedProjection, "fits", [])
    compare(tmp_path, names=("a",), split_seed=71)
    plan = tmp_path / "comparison-folds.json"
    if damage == "missing-plan":
        plan.unlink()
    elif damage == "missing-receipt":
        (tmp_path / "comparison-folds-receipt.json").unlink()
    else:
        changed = read(plan)
        changed.pop("sha256")
        changed["validation_indices"].reverse()
        evaluation.write_json(plan, dict(changed, sha256=evaluation._digest(changed)))
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    with patch.object(FittedProjection, "fit") as fit, patch.object(evaluation, "write_json") as write:
        with pytest.raises(ValueError, match="fold plan|fold receipt"):
            compare(tmp_path, names=("b",), split_seed=71)
    fit.assert_not_called()
    write.assert_not_called()
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.json")} == before


if __name__ == "__main__":
    {"initial": initial_runs, "resume": resumed_runs}[sys.argv[2]](Path(sys.argv[1]))
