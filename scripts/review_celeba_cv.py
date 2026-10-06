"""Review a complete declared CelebA grid using training-CV records only.

Author: Chang Cheng. This diagnostic does not fit models, propose a search
policy, select package defaults, or read development/final-test evaluations.
Only configuration.json and chunk-*/report.json are study inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supervised_nmf.evaluation import _digest, _jsonable, comparison_models

ATTRIBUTES = ("Smiling", "Wearing_Necktie")
NEAR_TIE_TOLERANCE = 1e-12


def _read(path, root, hashes, receipt=False):
    raw = path.read_bytes()
    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    result = json.loads(raw)
    if receipt:
        if result.get("sha256") != _digest(result.get("payload")):
            raise ValueError(f"CV report failed its checksum: {path}")
        return result["payload"]
    return result


def _models_and_chunks(configuration):
    settings = configuration["grid"]
    models = comparison_models(random_state=configuration["seed"],
                               max_iter=settings["max_iter"], tol=settings["tol"])
    chunks = {}
    for name, (estimator, initial) in models.items():
        grid = {key: settings[key.rsplit("__", 1)[-1]] for key in initial}
        models[name] = estimator, grid
        if name == "majority_class":
            chunks[name] = []
            continue
        screen = next((key for key in grid if key.rsplit("__", 1)[-1] in {"percentile", "score_threshold"}), None)
        parts = []
        for index, value in enumerate(grid[screen] if screen else [None]):
            part = dict(grid)
            if screen:
                part[screen] = [value]
            parts.append((f"chunk-{index:03d}", part))
        chunks[name] = parts
    return models, chunks


def _validate_folds(folds, cv):
    if len(folds) != cv:
        raise ValueError("The training-CV identity does not contain the declared fold count.")
    validation_sets, universe, minimum_training = [], None, math.inf
    for row in folds:
        train, validation = row["training_indices"], row["validation_indices"]
        a, b = set(train), set(validation)
        if (len(a) != len(train) or len(b) != len(validation) or not a or not b or a & b
                or any(not isinstance(value, int) or value < 0 for value in a | b)):
            raise ValueError("Invalid or overlapping within-training CV indices.")
        if universe is None:
            universe = a | b
        if a | b != universe:
            raise ValueError("Training-CV folds do not cover a common sample universe.")
        minimum_training = min(minimum_training, len(a))
        validation_sets.append(b)
    if (universe != set(range(len(universe))) or set.union(*validation_sets) != universe
            or sum(map(len, validation_sets)) != len(universe)):
        raise ValueError("Each training observation must appear in exactly one CV held-out fold.")
    return len(universe), int(minimum_training)


def _condense_candidate(candidate, expected, cv):
    if candidate["parameters"] != expected:
        raise ValueError("Candidate parameters differ from their declared grid position.")
    folds = candidate["folds"]
    if len(folds) != cv or sorted(row["fold"] for row in folds) != list(range(cv)):
        raise ValueError("Candidate is missing or duplicating a training-CV fold.")
    statuses = Counter(row["status"] for row in folds)
    valid = statuses.get("valid", 0) == cv
    if valid != candidate["valid"] or candidate["invalid_fold_count"] != cv - statuses.get("valid", 0):
        raise ValueError("Candidate qualification disagrees with its fold statuses.")
    scores = [row.get("balanced_accuracy") for row in sorted(folds, key=lambda item: item["fold"])]
    if valid:
        if any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError("A qualified fold has an invalid balanced accuracy.")
        if not math.isclose(candidate["mean_balanced_accuracy"], math.fsum(scores) / cv,
                            rel_tol=0, abs_tol=1e-14):
            raise ValueError("Candidate mean disagrees with the complete training fold scores.")
    elif candidate["mean_balanced_accuracy"] is not None:
        raise ValueError("An invalid candidate must not have an aggregate score.")
    counts, supports = [], []
    for row in sorted(folds, key=lambda item: item["fold"]):
        diagnostic = row.get("diagnostics", {})
        count = diagnostic.get("n_selected_features")
        if count is None and "Screening selected zero features" in row.get("error", ""):
            count = 0
        counts.append(count)
        support = diagnostic.get("selected_feature_indices")
        supports.append(_digest(support) if support is not None else None)
    return {"parameters": expected, "valid": valid,
            "mean_balanced_accuracy": candidate["mean_balanced_accuracy"],
            "fold_scores": scores, "fold_status_counts": dict(sorted(statuses.items())),
            "errors": sorted({f"{row.get('error_type', '')}: {row['error']}" for row in folds if row.get("error")}),
            "selected_features_per_fold": counts, "support_hashes_per_fold": supports}


def _collect(root, configuration, models, chunks, hashes):
    expected_paths = {f"{attribute}/{name}/{chunk}/report.json"
                      for attribute in ATTRIBUTES for name, entries in chunks.items() for chunk, _ in entries}
    existing_paths = {path.relative_to(root).as_posix() for attribute in ATTRIBUTES
                      for path in (root / attribute).glob("*/chunk-*/report.json")}
    if existing_paths != expected_paths:
        missing, extra = sorted(expected_paths - existing_paths), sorted(existing_paths - expected_paths)
        raise ValueError(f"Declared CV grid is incomplete or has unexpected chunks: missing={missing[:8]}, extra={extra[:8]}.")
    all_candidates, attribute_identities, runtime = {}, {}, None
    for attribute in ATTRIBUTES:
        all_candidates[attribute] = {}
        for name, (estimator, grid) in models.items():
            estimator_sources = {}
            for value in (estimator, *estimator.get_params(deep=True).values()):
                if hasattr(value, "get_params"):
                    filename = inspect.getsourcefile(type(value))
                    if filename and Path(filename).is_file():
                        estimator_sources[f"{type(value).__module__}.{type(value).__qualname__}"] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            global_grid = list(ParameterGrid(grid))
            canonical = {_digest(params): index for index, params in enumerate(global_grid)}
            if len(canonical) != len(global_grid):
                raise ValueError("Duplicate parameter tuples obscure canonical grid coverage.")
            collected = []
            for chunk_name, chunk_grid in chunks[name]:
                path = root / attribute / name / chunk_name / "report.json"
                report = _read(path, root, hashes, receipt=True)
                identity = report["identity"]
                calibration = identity["calibration"]
                expected = list(ParameterGrid(chunk_grid))
                if (report["state"] != "cv_completed" or report["refit_performed"] is not False
                        or report["fingerprint"] != _digest(identity)):
                    raise ValueError(f"Incomplete, refitted, or inconsistent training-CV report: {path}")
                if (calibration["candidates"] != expected or calibration["parameters"] != _jsonable(estimator.get_params(deep=True))
                        or calibration["estimator"] != f"{type(estimator).__module__}.{type(estimator).__qualname__}"
                        or calibration["seed"] != configuration["seed"] or calibration["cv"] != configuration["cv"]
                        or calibration["threads"] != configuration["threads"] or calibration["refit"] is not False
                        or calibration["source_sha256"] != configuration["source_sha256"]):
                    raise ValueError(f"CV data-generation settings, source, or grid identity differ: {path}")
                n_train, minimum_fold = _validate_folds(identity["folds"], configuration["cv"])
                shared_identity = {"training_data_sha256": calibration["data_sha256"],
                                   "folds_sha256": _digest(identity["folds"]), "n_train": n_train,
                                   "minimum_training_fold_size": minimum_fold}
                if attribute in attribute_identities and attribute_identities[attribute] != shared_identity:
                    raise ValueError(f"Methods/chunks use different training data or folds for {attribute}.")
                attribute_identities[attribute] = shared_identity
                if (identity["runtime"]["estimator_source"] != estimator_sources
                        or identity["runtime"]["package_source"] != configuration["source_sha256"]):
                    raise ValueError("Estimator source identity differs from the frozen training implementation.")
                common_runtime = {key: value for key, value in identity["runtime"].items() if key != "estimator_source"}
                if runtime is not None and runtime != common_runtime:
                    raise ValueError("CV chunks were generated under different software/numerical runtime identities.")
                runtime = common_runtime
                if (len(report["candidates"]) != len(expected)
                        or [row["index"] for row in report["candidates"]] != list(range(len(expected)))):
                    raise ValueError(f"Chunk candidate coverage/order differs from its declared grid: {path}")
                chunk_rows = []
                for candidate, params in zip(report["candidates"], expected):
                    row = _condense_candidate(candidate, params, configuration["cv"])
                    row.update(index=canonical[_digest(params)], chunk=chunk_name)
                    collected.append(row)
                    chunk_rows.append(row)
                local_ranked = sorted((i for i, row in enumerate(chunk_rows) if row["valid"]),
                                      key=lambda index: (-chunk_rows[index]["mean_balanced_accuracy"], index))
                local_selected = chunk_rows[local_ranked[0]]["parameters"] if local_ranked else None
                if (report["ranked_candidate_indices"] != local_ranked or report["selected_parameters"] != local_selected):
                    raise ValueError("Saved chunk ranking differs from the declared complete-fold rule.")
            collected.sort(key=lambda row: row["index"])
            if name != "majority_class" and [row["index"] for row in collected] != list(range(len(global_grid))):
                raise ValueError("Global declared grid is not covered exactly once.")
            all_candidates[attribute][name] = collected
    return all_candidates, attribute_identities, runtime, len(expected_paths)


def _screen_parameter(parameters):
    return next((key for key in parameters if key.rsplit("__", 1)[-1] in {"percentile", "score_threshold"}), None)


def _feature_counts(candidate, methods):
    counts = candidate["selected_features_per_fold"]
    if counts and all(count is not None for count in counts):
        return counts, "candidate fold diagnostics"
    parameter = _screen_parameter(candidate["parameters"])
    if parameter:
        base = parameter.rsplit("__", 1)[-1]
        method = "screening_only_percentile" if base == "percentile" else "screening_only_absolute"
        matches = [row for row in methods[method]
                   if row["parameters"][f"screen__{base}"] == candidate["parameters"][parameter]
                   and all(value is not None for value in row["selected_features_per_fold"])]
    else:
        matches = [row for row in methods["raw_knn"]
                   if all(value is not None for value in row["selected_features_per_fold"])]
    return ((matches[0]["selected_features_per_fold"], "matching screening-only/raw training-fold diagnostics")
            if matches else (None, "unavailable: inspect recorded failures"))


def _brief(candidate):
    if candidate is None:
        return None
    return {key: candidate[key] for key in ("index", "parameters", "valid", "mean_balanced_accuracy",
                                           "fold_scores", "fold_status_counts", "errors")}


def _neighborhood(candidate, candidates, methods, minimum_fold):
    result = {}
    counts, evidence = _feature_counts(candidate, methods)
    for parameter, selected in candidate["parameters"].items():
        others = {key: value for key, value in candidate["parameters"].items() if key != parameter}
        conditional = sorted([row for row in candidates if all(row["parameters"][key] == value for key, value in others.items())],
                             key=lambda row: row["parameters"][parameter])
        values = [row["parameters"][parameter] for row in conditional]
        location = values.index(selected)
        below, above = conditional[:location], conditional[location + 1:]
        valid_below = [row for row in below if row["valid"]]
        valid_above = [row for row in above if row["valid"]]
        valid_values = [row["parameters"][parameter] for row in conditional if row["valid"]]
        base = parameter.rsplit("__", 1)[-1]
        lower, upper, lower_inclusive = None, None, True
        if base == "n_components":
            lower, upper = 1, min([minimum_fold] + counts) if counts is not None else None
        elif base == "n_neighbors":
            lower, upper = 1, minimum_fold
        elif base == "percentile":
            lower, upper, lower_inclusive = 0, 100, False
        elif base == "score_threshold":
            lower = 0
        result[parameter] = {
            "selected": selected, "other_parameters_fixed": others,
            "declared_conditional_bounds": [min(values), max(values)],
            "valid_tested_conditional_bounds": [min(valid_values), max(valid_values)] if valid_values else None,
            "dimensional_or_natural_bounds": {"lower": lower, "upper": upper, "lower_inclusive": lower_inclusive,
                "note": "Necessary bounds only; empty screens, SPA conditioning and convergence can impose stricter limits."},
            "selected_features_per_fold": counts, "selected_feature_count_evidence": evidence,
            "at_declared_lower_bound": location == 0, "at_declared_upper_bound": location == len(values) - 1,
            "immediate_declared_below": _brief(below[-1]) if below else None,
            "immediate_declared_above": _brief(above[0]) if above else None,
            "nearest_valid_below": _brief(valid_below[-1]) if valid_below else None,
            "nearest_valid_above": _brief(valid_above[0]) if valid_above else None,
            "has_valid_tested_neighbors_on_both_sides": bool(valid_below and valid_above),
        }
    return result


def review(root):
    hashes = {}
    configuration = _read(root / "configuration.json", root, hashes)
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT / "src" / "supervised_nmf").glob("*.py"))}
    if sources != configuration["source_sha256"]:
        raise ValueError("Current package source differs from the frozen training study.")
    if hashlib.sha256((ROOT / "examples" / "celeba_parameter_study.py").read_bytes()).hexdigest() != configuration["script_sha256"]:
        raise ValueError("The original training-study script differs from its recorded version.")
    models, chunks = _models_and_chunks(configuration)
    candidates, identities, runtime, chunk_count = _collect(root, configuration, models, chunks, hashes)
    result = {"author": "Chang Cheng", "role": "training_cv_grid_diagnostics_only", "run_name": root.name,
              "complete_declared_chunk_count": chunk_count, "cv": configuration["cv"], "seed": configuration["seed"],
              "near_tie_tolerance": NEAR_TIE_TOLERANCE, "near_ties_include_exact_ties": True,
              "validation_outcomes_read": False, "automatic_grid_proposal": False, "defaults_frozen": False,
              "tuning_complete": False, "training_identities": identities, "runtime_identity": runtime,
              "source_sha256": sources, "reviewer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": dict(sorted(hashes.items())), "attributes": {},
              "screening_aliases_across_modes": {},
              "interpretation": [
                  "Conditional neighbors hold every other parameter fixed; failed settings do not certify a performance bracket.",
                  "Valid tested bounds are empirical grid bounds, not theoretical feasibility limits or convergence guarantees.",
                  "Dimensional rank caps use the smallest retained training-fold feature count and training-fold size.",
                  "Exact scores determine the existing canonical winner. The 1e-12 near-tie tolerance is reporting only.",
                  "All near-tied winners contribute to boundary summaries even when parameter examples are truncated.",
                  "A bracket, interior winner, or single paired CV run does not establish completed tuning or statistical superiority.",
              ]}
    for attribute, methods in candidates.items():
        output = {}
        cross_mode_masks = {}
        for mode, method, parameter in (("percentile", "screening_only_percentile", "screen__percentile"),
                                        ("absolute", "screening_only_absolute", "screen__score_threshold")):
            for row in methods[method]:
                if all(value is not None for value in row["support_hashes_per_fold"]):
                    key = _digest(row["support_hashes_per_fold"])
                    cross_mode_masks.setdefault(key, set()).add((mode, row["parameters"][parameter]))
        result["screening_aliases_across_modes"][attribute] = [
            [{"screening": mode, "value": value} for mode, value in sorted(values)]
            for values in cross_mode_masks.values() if len({mode for mode, _ in values}) > 1]
        for method, rows in methods.items():
            valid = sorted((row for row in rows if row["valid"]), key=lambda row: (-row["mean_balanced_accuracy"], row["index"]))
            winner = valid[0] if valid else None
            exact = [row for row in valid if row["mean_balanced_accuracy"] == winner["mean_balanced_accuracy"]] if winner else []
            near = [row for row in valid if winner["mean_balanced_accuracy"] - row["mean_balanced_accuracy"] <= NEAR_TIE_TOLERANCE] if winner else []
            neighborhoods = [_neighborhood(row, rows, methods, identities[attribute]["minimum_training_fold_size"]) for row in near]
            boundary_summary = {}
            for parameter in winner["parameters"] if winner else []:
                checks = [neighborhood[parameter] for neighborhood in neighborhoods]
                boundary_summary[parameter] = {
                    "near_tie_values": sorted({row["parameters"][parameter] for row in near}),
                    "near_ties_at_declared_lower": sum(check["at_declared_lower_bound"] for check in checks),
                    "near_ties_at_declared_upper": sum(check["at_declared_upper_bound"] for check in checks),
                    "near_ties_without_valid_tested_neighbors_on_both_sides": sum(not check["has_valid_tested_neighbors_on_both_sides"] for check in checks),
                    "near_ties_with_invalid_immediate_neighbor": sum(any(check[key] is not None and not check[key]["valid"]
                                                                       for key in ("immediate_declared_below", "immediate_declared_above")) for check in checks),
                }
            screen_masks = {}
            for row in rows:
                parameter = _screen_parameter(row["parameters"])
                if parameter and all(value is not None for value in row["support_hashes_per_fold"]):
                    mask_key = _digest(row["support_hashes_per_fold"])
                    screen_masks.setdefault(mask_key, set()).add(row["parameters"][parameter])
            output[method] = {
                "state": "untuned_baseline" if method == "majority_class" else "complete_cv_grid",
                "declared_candidates": len(rows), "valid_candidates": len(valid), "invalid_candidates": len(rows) - len(valid),
                "canonical_winner": _brief(winner), "exact_tie_count": len(exact), "near_tie_count": len(near),
                "exact_tie_parameter_examples": [row["parameters"] for row in exact[:12]],
                "near_tie_parameter_examples": [row["parameters"] for row in near[:12]],
                "tie_examples_truncated": len(near) > 12,
                "canonical_conditional_neighborhoods": neighborhoods[0] if neighborhoods else {},
                "all_near_ties_boundary_summary": boundary_summary,
                "screening_value_aliases_identical_masks_in_all_folds": [sorted(values) for values in screen_masks.values() if len(values) > 1],
                "failure_types": dict(Counter(error.split(":", 1)[0] for row in rows for error in row["errors"])),
                "failure_counting_note": "Failure types here count candidate/error combinations, not unique factorization attempts.",
            }
        result["attributes"][attribute] = output
    return result


def _fmt(value):
    return "—" if value is None else f"{value:.8f}"


def markdown(result):
    lines = ["# Training-CV grid review", "", "Author: Chang Cheng", "",
             f"Run `{result['run_name']}`: all {result['complete_declared_chunk_count']} declared CV chunks verified. "
             "No development-validation outcomes or final-test data were read. These diagnostics neither propose "
             "an automatic search nor finalize tuning or package defaults.", ""]
    for attribute, methods in result["attributes"].items():
        lines += [f"## {attribute}", "", "| Method | Valid / declared | Best training CV BA | Exact ties | Near ties (1e-12) | Canonical parameters |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, item in methods.items():
            winner = item["canonical_winner"]
            parameters = json.dumps(winner["parameters"], sort_keys=True) if winner else "Untuned baseline" if name == "majority_class" else "No qualified candidate"
            lines.append(f"| {name} | {item['valid_candidates']} / {item['declared_candidates']} | "
                         f"{_fmt(winner['mean_balanced_accuracy'] if winner else None)} | {item['exact_tie_count']} | "
                         f"{item['near_tie_count']} | `{parameters}` |")
        lines += ["", "### Conditional checks for each canonical winner", ""]
        for name, item in methods.items():
            if not item["canonical_winner"]:
                continue
            lines += [f"**{name}**", ""]
            for parameter, check in item["canonical_conditional_neighborhoods"].items():
                neighbors = []
                for direction, key in (("below", "immediate_declared_below"), ("above", "immediate_declared_above")):
                    row = check[key]
                    neighbors.append(f"{direction}: absent" if row is None else
                                     f"{direction}: {row['parameters'][parameter]} ({'valid' if row['valid'] else row['fold_status_counts']}; CV BA {_fmt(row['mean_balanced_accuracy'])})")
                bounds = check["dimensional_or_natural_bounds"]
                lines.append(f"- `{parameter}={check['selected']}`; declared conditional bounds "
                             f"{check['declared_conditional_bounds']}; valid tested bounds {check['valid_tested_conditional_bounds']}; "
                             f"necessary natural/dimensional limits {bounds['lower']} to {bounds['upper']} "
                             f"(lower {'inclusive' if bounds['lower_inclusive'] else 'exclusive'}). " + "; ".join(neighbors) + ".")
            ties = item["all_near_ties_boundary_summary"]
            lines += ["", "Near-tie boundary audit: " + "; ".join(
                f"`{key}` values {value['near_tie_values']}, lower/upper endpoints "
                f"{value['near_ties_at_declared_lower']}/{value['near_ties_at_declared_upper']}, "
                f"invalid immediate neighbors {value['near_ties_with_invalid_immediate_neighbor']}"
                for key, value in ties.items()) + ".", ""]
    lines += ["## Interpretation", ""] + [f"- {note}" for note in result["interpretation"]]
    lines += ["", "The JSON retains fold scores, failure messages for neighboring settings, feature-count evidence, "
              "nearest valid conditional neighbors, mask aliases, all-tie summaries, input hashes, and source versions. "
              "Preserve numerical controls and earlier incumbents when declaring any follow-up grid. "
              "Screening values with identical masks are aliases, not independent confirmation. "
              "A natural endpoint (such as k=1 or retained percentile=100) needs a constraint interpretation, "
              "not an impossible outward expansion.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN_DIRECTORY/cv-review")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    result = review(root)
    output = (args.output_dir or root / "cv-review").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-grid-review.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "training-grid-review.md").write_text(markdown(result), encoding="utf-8")
    print(f"Verified {result['complete_declared_chunk_count']} complete training-CV chunks; diagnostic review written to {output}")


if __name__ == "__main__":
    main()
