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


def amendment_rule(protocol):
    decision, timing = protocol["decision_rule"], protocol["timing"]
    return {
        "schema_version": 1, "author": "Chang Cheng", "amendment_id": "supervised-pca-screening-preference-v1",
        "timing_of_preference": "Disclosed after development results; not preregistered. See the supplied chronology audit for recorded observations and limitations.",
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
            "Within the retained set apply the protocol's declared serial-cost band, normalized recorded tuning, macro balanced accuracy and stable method order.",
        ],
        "unchanged": {"macro_margin": decision["macro_regret_tolerance"],
                      "attribute_margin": decision["attribute_regret_tolerance"],
                      "serial_cost_ratio": timing["primary_cost_closeness_ratio"],
                      "workload": f"one selected-model fit plus prediction on the fixed {protocol['sampling']['validation_sample_count']} validation images",
                      "case_cost": f"median of {timing['additional_repeats_per_selected_model']} per-repeat fit-plus-prediction sums"},
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


def _method_names(methods):
    return ", ".join(selector.table_cell(selector.LABELS[method]) for method in methods) or "None"


def _decision_stages(decision, protocol):
    """Describe recorded stages without recomputing gates or tie-breaks."""
    winner, qualified, scores = decision["winner"], decision["qualified_methods"], decision["scores"]
    names = ("accuracy_acceptable_methods", "methods_before_cost",
             "methods_within_cost_band", "methods_after_tuning_tie_break")
    stages = [decision.get(name, []) if winner is None else decision[name] for name in names]
    for methods in (qualified, *stages):
        selector.require(isinstance(methods, list) and all(isinstance(m, str) for m in methods)
                         and len(methods) == len(set(methods)) and set(methods) <= selector.LABELS.keys(),
                         "Invalid recorded method set.")
    selector.require(set(scores) == set(qualified), "Scores differ from the qualified methods.")
    preference, fallback = decision["screening_preference_applied"], decision["fallback_used"]
    selector.require(type(preference) is bool and type(fallback) is bool, "Invalid preference/fallback flags.")
    acceptable, retained, band, tuning = stages
    if winner is None:
        selector.require(not qualified and not any(stages) and not preference and not fallback,
                         "A blocked decision contains selection results.")
    else:
        selector.require(set(acceptable) <= set(qualified) and set(retained) <= set(qualified)
                         and set(band) <= set(retained) and set(tuning) <= set(band) and winner in tuning,
                         "Winner and recorded selection stages disagree.")
        selector.require(fallback == (not acceptable), "Fallback flag differs from the accuracy gate.")
        selector.require(not preference or (not fallback and set(retained) <= set(acceptable)
                         and all(m.endswith("_absolute") for m in retained)),
                         "Absolute preference differs from the recorded methods.")
    for key in ("macro_margin", "attribute_margin"):
        selector.finite(100 * selector.finite(decision[key], key), key + " (pp)")
    ratio = selector.finite(protocol["timing"]["primary_cost_closeness_ratio"], "cost ratio")
    selector.require(ratio >= 1, "Cost ratio must be at least one.")
    percentage = selector.finite(100 * (ratio - 1), "cost band (%)")
    for score in scores.values():
        for key in ("macro", "macro_regret", "normalized_serial_cost", "normalized_tuning_compute",
                    "geometric_workload_seconds"):
            selector.finite(score[key], key, unit_interval=key in ("macro", "macro_regret"))
        for key in ("attributes", "attribute_regrets"):
            selector.require(set(score[key]) == set(protocol["attributes"]), "Score attributes differ from the protocol.")
            for value in score[key].values():
                selector.finite(value, key, unit_interval=True)
    return [
        ("Operationally qualified", _method_names(qualified)),
        ("Passed the accuracy gate", _method_names(acceptable)),
        ("Absolute preference applied", "Yes" if preference else "No"),
        ("Minimum-regret fallback used", "Yes" if fallback else "No"),
        ("Retained before cost comparison", _method_names(retained)),
        (f"Within the {percentage:g}% cost band", _method_names(band)),
        ("Retained after tuning-cost comparison", _method_names(tuning)),
        ("Selected combination", _method_names([winner]) if winner is not None else "No qualified method"),
    ]


def _markdown(report):
    decision = report["primary"]
    protocol = report["decision_protocol"]
    winner = decision["winner"]
    selector.require(report.get("winner_method", winner) == winner, "Recorded winners disagree.")
    stages = _decision_stages(decision, protocol)
    score = decision["scores"].get(winner, {})
    original = report.get("original_primary")
    original_choice = ("Unavailable in this report" if original is None else
                       _method_names([original["winner"]]) if original["winner"] is not None else "No qualified method")
    rows = report["cases"]
    selector.require(isinstance(rows, list), "Recorded cases must be a list.")
    for row in rows:
        selector.require(type(row["qualified"]) is bool and isinstance(row["timing_pairs"], list)
                         and (row["metrics"] is None or isinstance(row["metrics"], dict)), "Invalid recorded case evidence.")
    sample_count = protocol.get("sampling", {}).get("validation_sample_count")
    selector.require(sample_count is None or (type(sample_count) is int and sample_count > 0),
                     "Invalid validation sample count.")
    lines = ["# CelebA default: disclosed screening-preference amendment", "", "Author: Chang Cheng.", "",
             f"**Selected combination: {selector.table_cell(selector.LABELS.get(winner, 'No qualified method'))}.**", "",
             f"Original recorded choice: {original_choice}.", "",
             "After development results were reviewed, the screening preference was amended to favor supervised PCA's absolute-score approach "
             "when predictive differences are narrow. The accompanying theoretical review considers the dissertation's four proofs and Wang's "
             "comments for the CD-versus-SPA comparison. This is a disclosed post-development preference; "
             "it was not preregistered and introduces no new experiment. The original decision is preserved separately.", "",
             f"The primary accuracy margins are {100*decision['macro_margin']:g} pp for macro balanced accuracy and "
             f"{100*decision['attribute_margin']:g} pp for each attribute, relative to the best qualified score. "
             "Absolute screening is preferred among methods passing that gate before applying the cost rule. "
             "Cost normalization uses all operationally qualified methods. No accuracy equivalence is claimed.", "",
             "| Recorded decision stage | Result |", "| --- | --- |"]
    lines += [f"| {label} | {value} |" for label, value in stages]
    if winner is None:
        lines += ["", "Selection is blocked because no method is operationally qualified."]
    elif decision["fallback_used"]:
        lines += ["", "No method passed the accuracy gate; the recorded choice uses the minimum-regret fallback."]
    lines += ["", "The stage sets are recorded decision outputs. Displayed numeric summaries are rounded; selection uses stored precision.", "",
             "| Method | Qualified cases | Macro BA (%) | Smiling BA (%) | Necktie BA (%) | Macro regret (pp) | Normalized serial cost | Geometric fit + predict (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["methods"]:
        expected, qualified = row["expected_cases"], row["qualified_cases"]
        selector.require(type(expected) is int and type(qualified) is int
                         and 0 <= qualified <= expected, "Invalid recorded method counts.")
        label = selector.table_cell(str(row["label"]).replace("\r", " "))
        value = decision["scores"].get(row["method"])
        if value is None:
            lines.append(f"| {label} | {row['qualified_cases']} / {row['expected_cases']} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(f"| {label} | {row['qualified_cases']} / {row['expected_cases']} | "
                     f"{100*value['macro']:.4f} | {100*value['attributes']['Smiling']:.4f} | "
                     f"{100*value['attributes']['Wearing_Necktie']:.4f} | {100*value['macro_regret']:.4f} | "
                     f"{value['normalized_serial_cost']:.4f} | {value['geometric_workload_seconds']:.4f} |")
    if winner:
        lines += ["", f"The selected method's macro deficit is {100*score['macro_regret']:.4f} percentage points; "
                  f"its attribute deficits are {100*score['attribute_regrets']['Smiling']:.4f} (Smiling) and "
                  f"{100*score['attribute_regrets']['Wearing_Necktie']:.4f} (Wearing Necktie). "
                  "Rank, cutoff and KNN neighbors remain tunable."]
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
    lines += [f"- Omit {selector.table_cell(row['omitted_field'])}={selector.table_cell(row['omitted_value'])}: "
              f"{selector.table_cell(selector.LABELS.get(row['decision']['winner'], 'None qualified'))}."
              for row in report["leave_one_out"]]
    lines += ["", "## Theory and scope", ""]
    lines += [f"- {item}" for item in report["amendment_rule"]["theory_scope"]]
    lines += ["", f"Recorded evidence contains {len(rows)} case rows, "
              f"{sum(row['metrics'] is not None for row in rows)} cases with evaluation metrics, "
              f"{sum(row['qualified'] for row in rows)} qualified cases, and "
              f"{sum(len(row['timing_pairs']) for row in rows)} recorded timing repeats.",
              f"The protocol records a validation sample of {sample_count} images." if sample_count is not None
              else "Validation sample size is unavailable in this report.",
              "The frozen study reuses one validation sample and overlapping training samples; "
              "these are not independent datasets. No universal optimum, statistical superiority, unseen-person guarantee or completed "
              "full simulation study is claimed. The original report retains all metrics, selected parameters, failure records and nested "
              "stage timings; this JSON preserves those cases and recomputes the preference decisions.", "",
              "## Test chronology and provenance", "",
              "Consult the referenced chronology audit for observed files, events and limitations. "
              "Absence of persisted test outcomes does not establish absence of in-memory access or verify process termination. "
              "No official-test outcomes enter the decision calculation. Historical attempts remain preserved; "
              "a new freeze is required before a new final-test evaluation.", "",
              f"Original decision SHA-256: `{report.get('original_decision_sha256', 'Unavailable')}`.", "",
              f"Amendment rule SHA-256: `{report.get('amendment_rule_sha256', 'Unavailable')}`.", "",
              f"Chronology audit SHA-256: `{report.get('chronology_sha256', 'Unavailable')}`.", ""]
    if report.get("input_sha256"):
        lines += ["`input_sha256` binds the original evidence, original decision, unchanged decision protocol, amendment rule, "
                  "chronology audit and decision-helper source.", ""]
    else:
        lines += ["Input checksums are unavailable in this report.", ""]
    lines += ["No package source/default is changed by this helper.", ""]
    return "\n".join(lines)


def markdown(report):
    try:
        return _markdown(report)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Missing or malformed amendment report evidence: {error}") from error


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
    rule = amendment_rule(protocol)
    result = summarize_amendment(original, protocol)
    result.update(author="Chang Cheng", created_utc=datetime.now(timezone.utc).isoformat(),
                  original_decision=str(original_path), original_decision_sha256=sha(original_path),
                  amendment_rule=rule, chronology_audit=str(chronology_path), chronology_sha256=sha(chronology_path),
                  amendment_source_sha256=sha(__file__), original_decision_preserved=True,
                  interpretation="Disclosed post-development preference amendment; consult the supplied chronology audit for observations and limitations. Theoretical routes remain conditional.",
                  test_outcomes_used_for_selection=False, final_test_evaluated=False, package_defaults_modified=False)
    rule_path = output / "amendment-rule.json"
    rule_bytes = (json.dumps(rule, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    rule_hash = hashlib.sha256(rule_bytes).hexdigest()
    result.update(amendment_rule_sha256=rule_hash, input_sha256={**hashes, str(rule_path): rule_hash})
    # Prepare all content before creating files. In particular, malformed
    # reporting evidence must not leave outputs that prevent a corrected retry.
    markdown_bytes = markdown(result).encode("utf-8")
    decision_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    inputs.verify_unchanged()
    for path, expected in hashes.items():
        selector.require(sha(path) == expected, f"Evidence changed while amending: {path}")
    output.mkdir(parents=True, exist_ok=True)
    rule_path.write_bytes(rule_bytes)
    selector.require(sha(rule_path) == rule_hash, "Amendment rule changed while writing.")
    (output / "decision.json").write_bytes(decision_bytes)
    (output / "decision.md").write_bytes(markdown_bytes)
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
