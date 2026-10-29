"""Freeze repeated-comparison grids using completed training CV only.

Author: Chang Cheng. This script never reads development-validation outcomes.
The decision protocol fixes the shortlist and grid-transfer rules in advance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from supervised_nmf.study import _read_receipt
from supervised_nmf.evaluation import _digest
from plan_celeba_default_study import METHODS, read, sha, key, box, write_new


def representation(parameters):
    return {name: value for name, value in parameters.items() if name != "n_neighbors"}


def shortlist(summary, incumbent):
    best = {}
    for row in summary["candidates"]:
        if row["valid"]:
            signature = key(representation(row["parameters"]))
            old = best.get(signature)
            if old is None or (-row["mean_balanced_accuracy"], row["index"]) < (-old["mean_balanced_accuracy"], old["index"]):
                best[signature] = row
    ranking = sorted(best.values(), key=lambda row: (-row["mean_balanced_accuracy"], row["index"]))
    if len(ranking) < 3:
        raise ValueError("Fewer than three qualified representation pairs: needs a versioned training-only amendment")
    incumbent_key = key(representation(incumbent))
    chosen = [best[incumbent_key]] if incumbent_key in best else []
    chosen += [row for row in ranking if key(representation(row["parameters"])) != incumbent_key][:3 - len(chosen)]

    def evidence(row):
        return {"parameters": representation(row["parameters"]),
                "best_training_cv_ba": row["mean_balanced_accuracy"],
                "best_neighbors": row["parameters"]["n_neighbors"], "candidate_index": row["index"]}

    # Near ties describe the ranking of distinct representations. They never
    # round scores, merge pairs, or replace the exact-score/canonical ordering.
    tolerance = 1e-12
    near_ties = []
    for first, left in enumerate(ranking):
        for second in range(first + 1, len(ranking)):
            right = ranking[second]
            gap = left["mean_balanced_accuracy"] - right["mean_balanced_accuracy"]
            if gap > tolerance:
                break
            near_ties.append({"first_rank": first + 1, "second_rank": second + 1,
                              "first_candidate_index": left["index"], "second_candidate_index": right["index"],
                              "score_gap": gap, "exact_score_tie": gap == 0})
    return {"incumbent_representation": representation(incumbent), "incumbent_qualified": incumbent_key in best,
            "pairs": [evidence(row) for row in chosen],
            "ranked_representations": [{"rank": rank + 1, **evidence(row)} for rank, row in enumerate(ranking)],
            "near_tie_diagnostics": {"tolerance": tolerance, "diagnostic_only": True,
                                     "selection_uses_exact_scores_and_canonical_order": True,
                                     "pairs": near_ties},
            "qualified_distinct_representations": len(ranking)}


def build(refinement, decision_path, source_plan_path):
    refinement = Path(refinement)
    decision, source_plan = read(decision_path), read(source_plan_path)
    if sha(source_plan_path) != decision["source_refinement_plan"]["sha256"]:
        raise ValueError("Original follow-up plan does not match frozen decision rule")
    cv = _read_receipt(refinement / "cv-summary.json")
    config = _read_receipt(refinement / "configuration.json")
    if cv["state"] != "cv_completed" or cv["configuration_sha256"] != _digest(config):
        raise ValueError("All refinement CV must complete before shortlisting")
    choices, evidence = {}, {}
    for attribute in decision["attributes"]:
        choices[attribute] = {}
        for method in METHODS:
            matches = [entry for entry in cv["jobs"].values()
                       if entry["job"]["attribute"] == attribute and entry["job"]["method"] == method]
            if len(matches) != 1:
                raise ValueError("Expected exactly one refinement job per method/attribute")
            summary = matches[0]
            saved_path = refinement / summary["job"]["id"] / "calibration-summary.json"
            if _read_receipt(saved_path) != summary:
                raise ValueError("Refinement summary changed")
            if summary["job"]["max_iter"] != decision["numerical_configuration"]["max_iter"] or summary["job"]["tol"] != decision["numerical_configuration"]["tol"]:
                raise ValueError("Numerical settings differ from frozen decision protocol")
            evidence[str(saved_path.resolve())] = sha(saved_path)
            choices[attribute][method] = shortlist(summary, source_plan["proposals"][attribute][method]["incumbent_parameters"])
    jobs = []
    rng = random.Random(decision["timing"]["order_seed"])
    blocks = []
    for n_train in decision["training_sizes"]:
        for seed in decision["training_seeds"]:
            for attribute in decision["attributes"]:
                order = list(METHODS)
                rng.shuffle(order)
                blocks.append({"n_train": n_train, "seed": seed, "attribute": attribute, "method_order": order})
                for method in order:
                    grid = []
                    for pair in choices[attribute][method]["pairs"]:
                        point = deepcopy(pair["parameters"])
                        if "score_threshold" in point:
                            point["score_threshold"] *= math.sqrt(n_train / 4000)
                        declared = box(point)
                        declared["n_neighbors"] = decision["stage_2"]["n_neighbors"]
                        grid.append(declared)
                    jobs.append({"id": f"n{n_train}-s{seed}-{attribute.lower()}-{method}",
                                 "cache": f"data/default-selection/n{n_train}-seed{seed}",
                                 "seed": seed, "attribute": attribute, "method": method, "n_train": n_train,
                                 "grid": grid, "max_iter": decision["numerical_configuration"]["max_iter"],
                                 "tol": decision["numerical_configuration"]["tol"]})
    return {"schema_version": 1, "author": "Chang Cheng", "cv": 5, "native_threads": 1,
            "profile_repeats": decision["timing"]["additional_repeats_per_selected_model"],
            "stage": 2, "purpose": decision["purpose"], "official_test_loaded": False,
            "decision_protocol_sha256": sha(decision_path), "planner_sha256": sha(__file__),
            "refinement_configuration_sha256": sha(refinement / "configuration.json"),
            "refinement_cv_summary_sha256": sha(refinement / "cv-summary.json"),
            "refinement_inputs_sha256": evidence, "training_only_shortlists": choices,
            "timing_order": {"seed": decision["timing"]["order_seed"], "blocks": blocks,
                             "three_profiles_are_consecutive_within_each_method": True},
            "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--decision-protocol", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = build(args.refinement, args.decision_protocol, args.source_plan)
    write_new(args.output, protocol)
    print(f"{len(protocol['jobs'])} repeated-comparison jobs; three representations and ten k values each")


if __name__ == "__main__":
    main()
