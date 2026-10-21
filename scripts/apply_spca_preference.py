"""Apply Chang Cheng's disclosed screening preference to saved development evidence.

Author: Chang Cheng. No fitting, image access or official-test outcomes. The
original decision is immutable; this separate amendment reuses its validated
cases and original accuracy, timing, normalization and tie-breaking rules.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_snmf_original_default_selector", ROOT / "scripts/select_celeba_default.py")
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def amended_choose(rows, protocol, **kwargs):
    """Preserve the global gate and normalization before preferring absolute."""
    result = deepcopy(selector.choose(rows, protocol, **kwargs))
    result["methods_before_screening_preference"] = result.get("methods_before_cost", [])[:]
    result["screening_preference_applied"] = False
    if result["winner"] is None:
        return result
    # Preference cannot rescue an inaccurate method or change the original
    # minimum-regret fallback when no method passes the accuracy gate.
    preferred = [method for method in result["accuracy_acceptable_methods"] if method.endswith("_absolute")]
    if not preferred:
        return result
    result["screening_preference_applied"] = True
    result["methods_before_cost"] = preferred
    scores = result["scores"]
    slack = protocol["decision_rule"]["numerical_comparison_slack"]
    cheapest = min(scores[method]["normalized_serial_cost"] for method in preferred)
    close = [method for method in preferred if scores[method]["normalized_serial_cost"] <=
             cheapest * protocol["timing"]["primary_cost_closeness_ratio"] + slack]
    result["methods_within_cost_band"] = close[:]
    best_tuning = min(scores[method]["normalized_tuning_compute"] for method in close)
    close = [method for method in close if scores[method]["normalized_tuning_compute"] <= best_tuning + slack]
    result["methods_after_tuning_tie_break"] = close[:]
    best_macro = max(scores[method]["macro"] for method in close)
    close = [method for method in close if scores[method]["macro"] >= best_macro - slack]
    winner = next(method for method in protocol["methods_in_final_tie_order"] if method in close)
    result.update(winner=winner, winning_solver="cd" if "_cd_" in winner else "spa", winning_screening="absolute")
    return result


def amendment_rule():
    return {
        "schema_version": 1, "author": "Chang Cheng", "amendment_id": "supervised-pca-screening-preference-v1",
        "timing_of_preference": "Disclosed after development results; not preregistered. Before official-test predictions or outcomes, per chronology audit.",
        "decision_context": [
            "Narrow predictive differences favor alignment with supervised PCA's absolute-score screening, subject to the unchanged accuracy gate.",
            "The theoretical review considers the dissertation's four proofs, Wang's comments and the difficulty of establishing the required assumptions for each solver and screening choice.",
            "The solver comparison is coordinate descent (CD) versus successive projection (SPA).",
        ],
        "rule": [
            "Retain original all-case numerical qualification and compute original global accuracy gate against all qualified methods.",
            "If any absolute-screening method passes that gate, retain only the absolute methods that pass it before comparing cost.",
            "If no absolute method passes the gate, retain the original gate result; if the gate is empty, retain the original minimum-regret fallback.",
            "Keep per-case cost and tuning normalization across all originally qualified methods, including percentile methods.",
            "Within the retained set apply original 5% serial-cost band, normalized recorded tuning, macro balanced accuracy and stable method order.",
        ],
        "unchanged": {"macro_margin": .01, "attribute_margin": .02, "serial_cost_ratio": 1.05,
                      "workload": "one selected-model fit plus prediction on the fixed 4000 validation images",
                      "case_cost": "median of three per-repeat fit-plus-prediction sums"},
        "theory_scope": [
            "Both reconstruction and reference-basis proof routes are conditional and do not force CD or SPA.",
            "Absolute screening has the simpler connection to the threshold theorem; a CV cutoff is not automatically its prescribed threshold.",
            "Percentile retention needs rank separation and enough retained features; CV does not prove those conditions.",
            "SPA's additional anchor-recovery route needs stronger geometry/noise assumptions; observed classification does not establish recovery.",
            "The separate KNN rate needs sample splitting, fixed transformation, smoothness/local mass and appropriate growing neighbor counts.",
            "Ordinary-risk bounds are not proofs of the balanced-accuracy decision criterion or of empirical superiority.",
        ],
        "test_policy": "No official-test outcomes enter this amendment; preserve interrupted original attempt and refreeze before a new test evaluation.",
    }


def summarize_amendment(original, protocol):
    rows = original["cases"]
    result = deepcopy(original)
    result["original_primary"] = deepcopy(original["primary"])
    result["primary"] = amended_choose(rows, protocol)
    result["winner_method"] = result["primary"]["winner"]
    result["sensitivity"] = [amended_choose(rows, protocol, macro_margin=margin,
                                           prediction_multiplier=multiplier, include_tuning=include)
                             for include in (False, True) for margin in protocol["sensitivity"]["macro_margins"]
                             for multiplier in protocol["sensitivity"]["prediction_multipliers"]]
    result["leave_one_out"] = [{"omitted_field": field, "omitted_value": value,
                               "decision": amended_choose([row for row in rows if row[field] != value], protocol,
                                                          allowed_methods=result["primary"]["qualified_methods"])}
                              for field, values in (("n_train", protocol["training_sizes"]), ("seed", protocol["training_seeds"]))
                              for value in values]
    return result


def markdown(report):
    decision = report["primary"]
    winner = decision["winner"]
    score = decision["scores"].get(winner, {})
    lines = ["# CelebA default: disclosed screening-preference amendment", "", "Author: Chang Cheng.", "",
             f"**Selected combination: {selector.LABELS.get(winner, 'No qualified method')}.**", "",
             "The original automatic decision selected SPA + percentile. It remains unchanged in its original directory. "
             "After development results were reviewed, the screening preference was amended to favor supervised PCA's absolute-score approach "
             "when predictive differences are narrow. The accompanying theoretical review considers the dissertation's four proofs and Wang's "
             "comments for the CD-versus-SPA comparison. This preference was disclosed before official-test predictions or outcomes; "
             "it was not preregistered and introduces no new experiment.", "",
             "The unchanged global gate permits at most 1 percentage point of macro balanced-accuracy regret and 2 points for either "
             "attribute, relative to the best of all operationally qualified methods. Among methods passing that gate, absolute screening "
             "is preferred before the unchanged cost rule. All four methods remain in cost normalization. No accuracy equivalence is claimed.", "",
             "| Method | Qualified cases | Macro BA (%) | Smiling BA (%) | Necktie BA (%) | Macro regret (pp) | Normalized serial cost | Geometric fit + predict (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["methods"]:
        value = decision["scores"].get(row["method"])
        if value is None:
            lines.append(f"| {selector.table_cell(row['label'])} | {row['qualified_cases']} / {row['expected_cases']} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(f"| {selector.table_cell(row['label'])} | {row['qualified_cases']} / {row['expected_cases']} | "
                     f"{100*value['macro']:.4f} | {100*value['attributes']['Smiling']:.4f} | "
                     f"{100*value['attributes']['Wearing_Necktie']:.4f} | {100*value['macro_regret']:.4f} | "
                     f"{value['normalized_serial_cost']:.4f} | {value['geometric_workload_seconds']:.4f} |")
    if winner:
        lines += ["", f"The selected method's macro deficit is {100*score['macro_regret']:.4f} percentage points; "
                  f"its attribute deficits are {100*score['attribute_regrets']['Smiling']:.4f} (Smiling) and "
                  f"{100*score['attribute_regrets']['Wearing_Necktie']:.4f} (Wearing Necktie). "
                  "Both absolute methods pass the unchanged gate. SPA-absolute alone is inside the preferred set's 5% cost band; "
                  "no secondary tie-break is needed. CD-absolute remains an alternative with macro BA only 0.0148 points below "
                  "the overall best, but higher measured serial cost. Rank, cutoff and KNN neighbors remain tunable."]
    lines += ["", "## Sensitivity to the declared accuracy margin and workload", "",
              "The preference is reapplied after the global gate in each descriptive sensitivity; these do not change the primary choice. "
              "Added CV compute is a parallel worker-wall sum and is not an isolated serial timing measurement.", "",
              "| Macro / attribute margin (pp) | Prediction multiplier | Add CV compute | Winner | Absolute preference applied |",
              "| --- | ---: | --- | --- | --- |"]
    for value in report["sensitivity"]:
        lines.append(f"| {100*value['macro_margin']:g} / {100*value['attribute_margin']:g} | {value['prediction_multiplier']:g} | "
                     f"{'yes' if value['include_parallel_tuning_in_workload'] else 'no'} | "
                     f"{selector.table_cell(selector.LABELS.get(value['winner'], 'None qualified'))} | "
                     f"{'yes' if value['screening_preference_applied'] else 'no'} |")
    lines += ["", "Leave-one-size/seed checks:", ""]
    lines += [f"- Omit {row['omitted_field']}={row['omitted_value']}: {selector.LABELS.get(row['decision']['winner'], 'None qualified')}."
              for row in report["leave_one_out"]]
    lines += ["", "## Theory and scope", ""]
    lines += [f"- {item}" for item in report["amendment_rule"]["theory_scope"]]
    lines += ["", "The 72 evaluations and 216 timing repeats reuse one 4,000-image validation sample. Training samples overlap; "
              "these are not independent datasets. No universal optimum, statistical superiority, unseen-person guarantee or completed "
              "full simulation study is claimed. The original report retains all metrics, selected parameters, failure records and nested "
              "stage timings; this JSON preserves those cases and recomputes the preference decisions.", "",
              "## Test chronology and provenance", "",
              "The interrupted original final-test attempt completed eight exact development revalidations and entered test loading. "
              "The chronology audit found no persisted test cache, prediction attempts, evaluations or final comparison. Partial in-memory "
              "decoding cannot be ruled out. Interruption was confirmed with exit code 1. No official-test outcomes were inspected "
              "or used for this amendment. The original attempt remains preserved and a new freeze is required.", "",
              f"Original decision SHA-256: `{report['original_decision_sha256']}`.", "",
              f"Amendment rule SHA-256: `{report['amendment_rule_sha256']}`.", "",
              f"Chronology audit SHA-256: `{report['chronology_sha256']}`.", "",
              "`input_sha256` binds the original evidence, original decision, unchanged decision protocol, amendment rule, "
              "chronology audit and decision-helper source. No package source/default is changed by this helper.", ""]
    return "\n".join(lines)


def write_amendment(original_path, protocol_path, chronology_path, output):
    original_path, protocol_path, chronology_path, output = map(lambda p: Path(p).resolve(),
                                                               (original_path, protocol_path, chronology_path, output))
    selector.require(output != original_path.parent, "Amendment must not replace the original decision directory.")
    selector.require(not any((output / name).exists() for name in ("decision.json", "decision.md", "amendment-rule.json")),
                     "Amendment outputs already exist; use a new directory.")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    chronology = json.loads(chronology_path.read_text(encoding="utf-8"))
    hashes = dict(original["input_sha256"])
    hashes.update({str(path): sha(path) for path in (original_path, protocol_path, chronology_path, Path(__file__),
                                                   ROOT / "scripts/select_celeba_default.py")})
    for path, expected in hashes.items():
        selector.require(sha(path) == expected, f"Original evidence changed: {path}")
    selector.require(original["decision_protocol_sha256"] == sha(protocol_path), "Original protocol hash mismatch.")
    selector.require(original["decision_protocol"] == protocol, "Original protocol content differs.")
    # Reload only the development study to verify receipts, case coverage and
    # metrics; the chronology file is the sole final-attempt input.
    rows, validated_protocol, inputs, configuration = selector.load_cases(original["source_run"], protocol_path)
    selector.require(rows == original["cases"], "Original decision cases differ from the development receipts.")
    selector.require(selector.choose(rows, validated_protocol) == original["primary"], "Original primary decision does not reproduce.")
    selector.require(selector.digest(configuration) == original["configuration_sha256"], "Development configuration mismatch.")
    counts = chronology["observed_counts"]
    selector.require(counts["test_prediction_attempts"] == 0 and counts["test_evaluation_files"] == 0
                     and counts["final_comparison_exists"] is False, "Chronology does not support pre-outcome amendment.")
    rule = amendment_rule()
    result = summarize_amendment(original, protocol)
    result.update(author="Chang Cheng", created_utc=datetime.now(timezone.utc).isoformat(),
                  original_decision=str(original_path), original_decision_sha256=sha(original_path),
                  amendment_rule=rule, chronology_audit=str(chronology_path), chronology_sha256=sha(chronology_path),
                  amendment_source_sha256=sha(__file__), original_decision_preserved=True,
                  interpretation="Disclosed post-development preference amendment before official-test outcomes; theoretical routes remain conditional.",
                  test_outcomes_used_for_selection=False, final_test_evaluated=False, package_defaults_modified=False)
    output.mkdir(parents=True, exist_ok=True)
    rule_path = output / "amendment-rule.json"
    rule_path.write_text(json.dumps(rule, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes[str(rule_path)] = sha(rule_path)
    result.update(amendment_rule_sha256=sha(rule_path), input_sha256=hashes)
    inputs.verify_unchanged()
    for path, expected in hashes.items():
        selector.require(sha(path) == expected, f"Evidence changed while amending: {path}")
    (output / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "decision.md").write_text(markdown(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-decision", type=Path, required=True)
    parser.add_argument("--decision-protocol", type=Path, required=True)
    parser.add_argument("--chronology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_amendment(args.original_decision, args.decision_protocol, args.chronology, args.output)
    print(json.dumps({"winner_method": report["winner_method"], "output": str(args.output.resolve()),
                      "decision_sha256": sha(args.output / "decision.json")}, indent=2))
    return 0 if report["winner_method"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
