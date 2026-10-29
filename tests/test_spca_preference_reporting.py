"""Preference-report facts and provenance. Author: Chang Cheng."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_snmf_test_preference_reporting", ROOT / "scripts/apply_spca_preference.py")
amendment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(amendment)
METHODS = list(amendment.selector.LABELS)
CP, CA, SP, SA = METHODS


def protocol():
    return {
        "methods_in_final_tie_order": METHODS[:],
        "attributes": ["Smiling", "Wearing_Necktie"],
        "training_sizes": [100, 200], "training_seeds": [7, 11],
        "sampling": {"validation_sample_count": 64},
        "timing": {
            "primary_cost_closeness_ratio": 1.05,
            "additional_repeats_per_selected_model": 3,
            "primary_workload": "one selected-model fit plus validation prediction",
            "case_cost": "median of paired fit-plus-prediction sums",
        },
        "decision_rule": {
            "macro_regret_tolerance": .01, "attribute_regret_tolerance": .02,
            "numerical_comparison_slack": 1e-12,
        },
        "sensitivity": {
            "attribute_margin_multiplier": 2,
            "macro_margins": [.005, .01], "prediction_multipliers": [1, 10],
        },
    }


def rows_for(config, *, scores=None, costs=None, tuning=None):
    scores = scores or {method: (.8, .7) for method in METHODS}
    costs = costs or {CP: 4., CA: 3., SP: 1., SA: 2.}
    tuning = tuning or {method: 10. for method in METHODS}
    return [
        {"method": method, "attribute": attribute, "n_train": size, "seed": seed,
         "qualified": True, "metrics": {"balanced_accuracy": scores[method][index]},
         "tuning_compute_seconds": tuning[method],
         "timing_pairs": [{"fit": .8 * costs[method], "prediction": .2 * costs[method]}
                          for _ in range(3)]}
        for method in METHODS for size in config["training_sizes"]
        for seed in config["training_seeds"]
        for index, attribute in enumerate(config["attributes"])
    ]


def report_for(rows=None, config=None, **kwargs):
    config = deepcopy(config) if config is not None else protocol()
    rows = deepcopy(rows) if rows is not None else rows_for(config)
    decision = amendment.amended_choose(rows, config, **kwargs)
    methods = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        methods.append({
            "method": method, "label": amendment.selector.LABELS[method],
            "expected_cases": len(subset),
            "qualified_cases": sum(row["qualified"] for row in subset),
            "evaluated_cases": sum(row["metrics"] is not None for row in subset),
        })
    return {
        "primary": decision, "winner_method": decision["winner"],
        "original_primary": amendment.selector.choose(rows, config, **kwargs),
        "decision_protocol": config, "cases": rows, "methods": methods,
        "sensitivity": [], "leave_one_out": [],
        "amendment_rule": amendment.amendment_rule(config),
        "original_decision_sha256": "0" * 64,
        "amendment_rule_sha256": "1" * 64,
        "chronology_sha256": "2" * 64,
        "input_sha256": {}, "original_decision_preserved": True,
    }


def stage(text, name):
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip().replace(r"\|", "|")
                 for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) == 2 and cells[0] == name:
            return cells[1]
    raise AssertionError(f"Missing recorded decision stage: {name}")


def method_names(methods):
    return ", ".join(amendment.selector.LABELS[method] for method in methods) if methods else "None"


@pytest.mark.parametrize("scenario", ["spa_cost", "cd_tuning", "percentile", "fallback", "all_tied", "blocked"])
def test_report_matches_recorded_gate_preference_and_tie_stages(scenario):
    config = protocol()
    kwargs = {}
    rows = rows_for(config)
    expected = dict(winner=SA, gate=METHODS, retained=[CA, SA], band=[SA],
                    after_tuning=[SA], preferred=True, fallback=False, qualified=METHODS)
    if scenario == "cd_tuning":
        rows = rows_for(config, costs={CP: 4., CA: 2.08, SP: 1., SA: 2.},
                        tuning={CP: 10., CA: 5., SP: 10., SA: 10.})
        expected.update(winner=CA, band=[CA, SA], after_tuning=[CA])
    elif scenario == "percentile":
        rows = rows_for(config, scores={CP: (.8, .7), CA: (.75, .65), SP: (.8, .7), SA: (.75, .65)})
        expected.update(winner=SP, gate=[CP, SP], retained=[CP, SP], band=[SP],
                        after_tuning=[SP], preferred=False)
    elif scenario == "fallback":
        rows = rows_for(config, scores={CP: (.81, .69), CA: (.79, .71), SP: (.805, .705), SA: (.8, .70)})
        kwargs["macro_margin"] = 0.
        expected.update(winner=SP, gate=[], retained=[SP], band=[SP],
                        after_tuning=[SP], preferred=False, fallback=True)
    elif scenario == "all_tied":
        rows = rows_for(config, costs={method: 1. for method in METHODS})
        expected.update(winner=CA, band=[CA, SA], after_tuning=[CA, SA])
    elif scenario == "blocked":
        for row in rows:
            row["qualified"] = False
        expected.update(winner=None, gate=[], retained=[], band=[], after_tuning=[],
                        preferred=False, qualified=[])
    report = report_for(rows, config, **kwargs)
    assert report["primary"]["winner"] == expected["winner"]
    text = amendment.markdown(report)
    for name, key in (
        ("Operationally qualified", "qualified"), ("Passed the accuracy gate", "gate"),
        ("Retained before cost comparison", "retained"), ("Within the 5% cost band", "band"),
        ("Retained after tuning-cost comparison", "after_tuning"),
    ):
        assert stage(text, name) == method_names(expected[key])
    assert stage(text, "Absolute preference applied") == ("Yes" if expected["preferred"] else "No")
    assert stage(text, "Minimum-regret fallback used") == ("Yes" if expected["fallback"] else "No")
    selected = amendment.selector.LABELS.get(expected["winner"], "No qualified method")
    assert stage(text, "Selected combination") == selected
    assert "SPA-absolute alone" not in text
    assert "0.0148 points" not in text
    assert "no secondary tie-break is needed" not in text
    if scenario == "blocked":
        assert "blocked" in text.lower()
    if scenario == "fallback":
        assert "No method passed the accuracy gate" in text


def test_renderer_preserves_per_attribute_table_sensitivity_and_disclosure():
    report = report_for()
    rows, config = report["cases"], report["decision_protocol"]
    report["sensitivity"] = [amendment.amended_choose(rows, config, macro_margin=.005, prediction_multiplier=10)]
    report["leave_one_out"] = [{"omitted_field": "seed", "omitted_value": 7,
                               "decision": amendment.amended_choose([r for r in rows if r["seed"] != 7], config)}]
    text = amendment.markdown(report)
    assert "Smiling BA (%)" in text
    assert "Necktie BA (%)" in text or "Wearing_Necktie BA (%)" in text
    assert "Macro regret (pp)" in text
    assert "| 0.5 / 1 | 10 | no |" in text
    assert "Omit seed=7:" in text
    assert "after development" in text.lower()
    assert "not preregistered" in text.lower()
    assert "No accuracy equivalence" in text or "not an accuracy-equivalence" in text


def test_numeric_rule_metadata_and_report_follow_supplied_protocol():
    config = protocol()
    config["decision_rule"].update(macro_regret_tolerance=.005, attribute_regret_tolerance=.0075)
    config["timing"].update(primary_cost_closeness_ratio=1.2, additional_repeats_per_selected_model=2)
    config["sampling"]["validation_sample_count"] = 137
    report = report_for(config=config)
    rule = report["amendment_rule"]
    assert rule["unchanged"]["macro_margin"] == .005
    assert rule["unchanged"]["attribute_margin"] == .0075
    assert rule["unchanged"]["serial_cost_ratio"] == 1.2
    assert "137" in rule["unchanged"]["workload"]
    assert "2" in rule["unchanged"]["case_cost"] or "two" in rule["unchanged"]["case_cost"]
    assert "5%" not in " ".join(rule["rule"])
    text = amendment.markdown(report)
    assert "0.5 pp" in text and "0.75 pp" in text
    assert stage(text, "Within the 20% cost band")
    assert "137 images" in text


def test_failed_case_counts_are_derived_from_available_evidence():
    config = protocol()
    rows = rows_for(config)
    failed = next(row for row in rows if row["method"] == CA)
    failed.update(qualified=False, metrics=None, timing_pairs=[])
    report = report_for(rows, config)
    text = amendment.markdown(report)
    assert "32 case rows" in text
    assert "31 cases with evaluation metrics" in text
    assert "31 qualified cases" in text
    assert "93 recorded timing repeats" in text
    assert "72 evaluations" not in text and "216 timing repeats" not in text
    assert stage(text, "Operationally qualified") == method_names([CP, SP, SA])


def test_original_choice_is_reported_from_its_record():
    config = protocol()
    report = report_for(rows_for(config, costs={CP: .5, CA: 3., SP: 1., SA: 2.}), config)
    assert report["original_primary"]["winner"] == CP
    text = amendment.markdown(report)
    assert "Original recorded choice: CD + percentile." in text
    assert "The original automatic decision selected SPA + percentile" not in text


def test_missing_optional_context_is_explicitly_unavailable():
    report = report_for()
    for key in ("original_primary", "original_decision_sha256", "amendment_rule_sha256", "chronology_sha256", "input_sha256"):
        report.pop(key)
    report["decision_protocol"]["sampling"].pop("validation_sample_count")
    text = amendment.markdown(report)
    assert "Original recorded choice: Unavailable" in text
    assert "Validation sample size is unavailable" in text
    assert "Chronology audit SHA-256: `Unavailable`" in text
    assert "eight exact development" not in text
    assert "exit code 1" not in text


def test_rendering_does_not_run_selection_or_mutate_its_evidence(monkeypatch):
    report = report_for()
    before = deepcopy(report)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must use the recorded selection.")

    monkeypatch.setattr(amendment, "amended_choose", forbidden)
    monkeypatch.setattr(amendment.selector, "choose", forbidden)
    text = amendment.markdown(report)
    assert stage(text, "Selected combination") == amendment.selector.LABELS[SA]
    assert report == before


@pytest.mark.parametrize("kind", [
    "primary", "protocol", "cases", "scores", "selected_score", "cost_stage", "tuning_stage",
    "winner_outside_stages", "top_level_winner", "nonfinite_macro", "nonfinite_cost",
    "nonfinite_tuning", "nonfinite_margin", "overflowing_percentage", "impossible_method_count",
])
def test_missing_or_contradictory_computational_evidence_is_rejected(kind):
    report = report_for()
    decision = report["primary"]
    if kind in ("primary", "protocol", "cases"):
        report.pop({"primary": "primary", "protocol": "decision_protocol", "cases": "cases"}[kind])
    elif kind == "scores":
        decision.pop("scores")
    elif kind == "selected_score":
        decision["scores"].pop(SA)
    elif kind in ("cost_stage", "tuning_stage"):
        decision.pop("methods_within_cost_band" if kind == "cost_stage" else "methods_after_tuning_tie_break")
    elif kind == "winner_outside_stages":
        decision["winner"] = CA
        report["winner_method"] = CA
    elif kind == "top_level_winner":
        report["winner_method"] = CA
    elif kind in ("nonfinite_macro", "nonfinite_cost", "nonfinite_tuning", "overflowing_percentage"):
        field = {"nonfinite_macro": "macro", "nonfinite_cost": "normalized_serial_cost",
                 "nonfinite_tuning": "normalized_tuning_compute", "overflowing_percentage": "macro_regret"}[kind]
        decision["scores"][SA][field] = 1e308 if kind == "overflowing_percentage" else float("nan")
    elif kind == "nonfinite_margin":
        decision["macro_margin"] = float("inf")
    else:
        report["methods"][0]["qualified_cases"] = report["methods"][0]["expected_cases"] + 1
    with pytest.raises(ValueError):
        amendment.markdown(report)


def test_method_labels_escape_pipes_and_line_breaks():
    report = report_for()
    report["methods"][0]["label"] = "CD | alternate\nlabel\rline"
    text = amendment.markdown(report)
    assert "\r" not in text
    line = next(line for line in text.splitlines() if line.startswith(r"| CD \| alternate"))
    assert "label line" in line
    assert len(re.split(r"(?<!\\)\|", line)[1:-1]) == 8


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def writer_fixture(tmp_path, monkeypatch):
    config = protocol()
    report = report_for(config=config)
    original = deepcopy(report)
    original["primary"] = amendment.selector.choose(report["cases"], config)
    original["winner_method"] = original["primary"]["winner"]
    original.pop("original_primary")
    original["source_run"] = str(tmp_path / "source-study")
    configuration = {"fixture": "paired saved development rows"}
    original["configuration_sha256"] = amendment.selector.digest(configuration)
    original_path = tmp_path / "original.json"
    protocol_path = tmp_path / "protocol.json"
    chronology_path = tmp_path / "chronology.json"
    evidence_path = tmp_path / "evidence.json"
    write_json(evidence_path, {"fixture": True})
    write_json(protocol_path, config)
    original["decision_protocol_sha256"] = sha(protocol_path)
    original["input_sha256"] = {str(evidence_path): sha(evidence_path)}
    write_json(original_path, original)
    chronology = {"observed_counts": {
        "test_prediction_attempts": 0, "test_evaluation_files": 0,
        "final_comparison_exists": False, "development_revalidations": 3,
    }}
    write_json(chronology_path, chronology)
    calls = {"load": 0, "verify": 0}

    def verify_unchanged():
        calls["verify"] += 1
        assert sha(evidence_path) == original["input_sha256"][str(evidence_path)]

    def load_cases(source_run, requested_protocol):
        calls["load"] += 1
        assert str(source_run) == original["source_run"]
        assert Path(requested_protocol) == protocol_path
        return (deepcopy(report["cases"]), deepcopy(config),
                SimpleNamespace(verify_unchanged=verify_unchanged), deepcopy(configuration))

    # Only receipt loading is replaced. Selection, hashing, metadata creation,
    # report rendering and file writes all exercise the real writer.
    monkeypatch.setattr(amendment.selector, "load_cases", load_cases)
    return SimpleNamespace(original=original, original_path=original_path, protocol_path=protocol_path,
                           chronology=chronology, chronology_path=chronology_path,
                           evidence_path=evidence_path, output=tmp_path / "amended", calls=calls)


def test_writer_keeps_evidence_and_binds_generated_rule_report_and_hashes(tmp_path, monkeypatch):
    fixture = writer_fixture(tmp_path, monkeypatch)
    protected = {path: path.read_bytes() for path in (
        fixture.original_path, fixture.protocol_path, fixture.chronology_path, fixture.evidence_path)}
    result = amendment.write_amendment(fixture.original_path, fixture.protocol_path,
                                      fixture.chronology_path, fixture.output)
    decision_path, rule_path = fixture.output / "decision.json", fixture.output / "amendment-rule.json"
    saved = json.loads(decision_path.read_text(encoding="utf-8"))
    assert saved == result
    assert result["winner_method"] == SA
    assert result["original_primary"]["winner"] == SP
    assert result["original_decision_sha256"] == sha(fixture.original_path)
    assert result["amendment_rule_sha256"] == sha(rule_path)
    assert result["chronology_sha256"] == sha(fixture.chronology_path)
    assert result["input_sha256"][str(rule_path)] == sha(rule_path)
    assert json.loads(rule_path.read_text(encoding="utf-8")) == result["amendment_rule"]
    text = (fixture.output / "decision.md").read_text(encoding="utf-8")
    assert text == amendment.markdown(result)
    assert stage(text, "Selected combination") == amendment.selector.LABELS[SA]
    assert result["package_defaults_modified"] is False
    assert result["test_outcomes_used_for_selection"] is False
    assert result["original_decision_preserved"] is True
    assert fixture.calls == {"load": 1, "verify": 1}
    assert all(path.read_bytes() == before for path, before in protected.items())
    written = {path: path.read_bytes() for path in (decision_path, rule_path, fixture.output / "decision.md")}
    with pytest.raises(ValueError, match="already exist"):
        amendment.write_amendment(fixture.original_path, fixture.protocol_path,
                                  fixture.chronology_path, fixture.output)
    assert all(path.read_bytes() == before for path, before in written.items())


@pytest.mark.parametrize("field,value", [
    ("test_prediction_attempts", 1), ("test_evaluation_files", 1), ("final_comparison_exists", True),
])
def test_writer_rejects_official_test_outcomes_before_writing(tmp_path, monkeypatch, field, value):
    fixture = writer_fixture(tmp_path, monkeypatch)
    fixture.chronology["observed_counts"][field] = value
    write_json(fixture.chronology_path, fixture.chronology)
    with pytest.raises(ValueError, match="Chronology"):
        amendment.write_amendment(fixture.original_path, fixture.protocol_path,
                                  fixture.chronology_path, fixture.output)
    assert not fixture.output.exists()


def test_writer_does_not_replace_original_directory(tmp_path, monkeypatch):
    fixture = writer_fixture(tmp_path, monkeypatch)
    before = fixture.original_path.read_bytes()
    with pytest.raises(ValueError, match="original decision directory"):
        amendment.write_amendment(fixture.original_path, fixture.protocol_path,
                                  fixture.chronology_path, tmp_path)
    assert fixture.original_path.read_bytes() == before
    assert fixture.calls["load"] == 0


def test_writer_does_not_create_outputs_for_malformed_report_metadata(tmp_path, monkeypatch):
    fixture = writer_fixture(tmp_path, monkeypatch)
    fixture.original["methods"][0]["expected_cases"] = -1
    write_json(fixture.original_path, fixture.original)
    with pytest.raises(ValueError):
        amendment.write_amendment(fixture.original_path, fixture.protocol_path,
                                  fixture.chronology_path, fixture.output)
    assert not fixture.output.exists()
