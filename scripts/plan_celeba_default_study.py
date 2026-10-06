"""Materialize explicit CelebA comparison grids. Author: Chang Cheng.

Only training-CV evidence determines candidate grids. This is not an adaptive
search service; each emitted protocol is immutable once execution starts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sklearn.model_selection import ParameterGrid

METHODS = ("screened_cd_percentile", "screened_cd_absolute",
           "screened_spa_percentile", "screened_spa_absolute")
ATTRIBUTES = ("Smiling", "Wearing_Necktie")
NEIGHBORS = [1, 2, 3, 5, 9, 15, 25, 41, 65, 81]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def box(point):
    return {name: [value] for name, value in point.items()}


def canonical_boxes(boxes):
    points = {key(point): point for point in ParameterGrid(boxes)}
    return [box(points[name]) for name in sorted(points)]


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Refusing to overwrite a different protocol: {path}")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Protocol: {path.resolve()}\nSHA256: {sha(path)}", flush=True)


def stage_one(plan_path, cache):
    plan = read(plan_path)
    grids = {}
    for attribute in ATTRIBUTES:
        for method in METHODS:
            proposal = plan["proposals"][attribute][method]
            boxes = deepcopy(proposal["declared_parameter_grid_boxes"])
            boxes.append(box(proposal["incumbent_parameters"]))
            boxes.extend(box(row["parameters"]) for row in proposal["retain_prior_distinct_representation_guards"])
            grids[attribute, method] = boxes
    for check in plan["essential_follow_up_checks"]:
        if "grids" in check:
            grids.setdefault((check["attribute"], check["method"]), []).extend(deepcopy(check["grids"]))
        elif "methods_and_ranks" in check:
            for method, ranks in check["methods_and_ranks"].items():
                if method.startswith("screened_cd_") or method.startswith("screened_spa_"):
                    point = {"n_neighbors": check["n_neighbors"], "score_threshold": check["score_threshold"], "n_components": ranks}
                else:
                    point = {"knn__n_neighbors": check["n_neighbors"], "screen__score_threshold": check["score_threshold"]}
                    if ranks is not None:
                        point["pca__n_components"] = ranks
                grids.setdefault((check["attribute"], method), []).append(point)
    for row in plan["numerical_qualification_proposal"]["affected_representation_settings"]:
        point = box(row["parameters"])
        point["n_neighbors"] = [1, 3, 5, 9, 15, 25, 41]
        grids.setdefault((row["attribute"], row["method"]), []).append(point)
    jobs = []
    for (attribute, method), boxes in sorted(grids.items()):
        grid = canonical_boxes(boxes)
        representations = {key({k: v for k, v in p.items() if not k.endswith("n_neighbors")})
                           for p in ParameterGrid(grid)}
        jobs.append({"id": f"{attribute.lower()}-{method}", "cache": str(Path(cache).resolve()),
                     "seed": 42, "attribute": attribute, "method": method, "grid": grid,
                     "max_iter": 20000, "tol": 0.0001,
                     "role": "refinement" if method in METHODS else "baseline_qualification",
                     "declared_candidates": len(grid), "nominal_representation_folds": 5 * len(representations)})
    return {"schema_version": 1, "author": "Chang Cheng", "cv": 5, "native_threads": 1,
            "profile_repeats": 3, "purpose": "Training-only refinement and numerical qualification before general-default comparison",
            "stage": 1, "official_test_loaded": False, "default_selection": False,
            "source_plan": str(Path(plan_path).resolve()), "source_plan_sha256": sha(plan_path),
            "planner_sha256": sha(__file__), "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = stage_one(args.plan, args.cache)
    write_new(args.output, protocol)
    print(f"{len(protocol['jobs'])} jobs, {sum(j['declared_candidates'] for j in protocol['jobs'])} candidates, "
          f"{sum(j['nominal_representation_folds'] for j in protocol['jobs'])} nominal representation-fold attempts")


if __name__ == "__main__":
    main()
