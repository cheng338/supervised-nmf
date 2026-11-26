"""Evidence used to check a completed writer's numeric refinement advice."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from supervised_nmf import evaluation


@pytest.fixture(scope="module")
def observed_report(tmp_path_factory):
    path = tmp_path_factory.mktemp("refinement-axes") / "cv.json"
    y = np.tile([0, 1], 12)
    evaluation.calibration_search(
        DecisionTreeClassifier(random_state=3), {"max_depth": [1, np.nan, np.inf, -np.inf, None]},
        y[:, None], y, cv=2, seed=3, threads=1, refit=False, checkpoint_path=path,
    )
    report = evaluation.load_calibration_report(path, expected_sha256=sha256(path.read_bytes()).hexdigest())
    assert report == json.loads(path.read_bytes())
    return report


def test_writer_retains_coordinates_before_json_loss(observed_report):
    report = observed_report
    assert report["identity"]["fold_evidence_version"] == 6
    assert report["selected_parameters"] == {"max_depth": 1}
    observation = report["refinement_axis_inputs"]["max_depth"]
    assert observation["includes_none"] is True
    coordinates = observation["coordinates"]
    assert len(coordinates) == 4 and set(coordinates) == {1.0, "nan", "+inf", "-inf"}
    entry = report["refinement"]["parameters"]["max_depth"]
    assert entry["values"] == [None if isinstance(x, str) else x for x in coordinates]
    position = coordinates.index(1)
    expected = ("lower_boundary_requires_expansion_or_constraint" if position == 0
                else "upper_boundary_requires_expansion_or_constraint" if position == 3
                else "interior_interval_requires_refinement")
    assert entry["status"] == expected


@pytest.mark.parametrize("change", ["missing", "extra", "repeated_infinity", "wrong_finite", "none_flag", "status"])
def test_changed_observation_or_advice_fails_even_with_matching_digest(tmp_path, observed_report, change):
    report = deepcopy(observed_report)
    axes = report["refinement_axis_inputs"]
    coordinates = axes["max_depth"]["coordinates"]
    entry = report["refinement"]["parameters"]["max_depth"]
    if change == "missing":
        axes.clear()
    elif change == "extra":
        axes["unknown"] = axes["max_depth"]
    elif change == "repeated_infinity":
        coordinates[coordinates.index("nan")] = "+inf"
    elif change == "wrong_finite":
        coordinates[coordinates.index(1)] = 23.0
        entry["values"][entry["values"].index(1)] = 23.0
    elif change == "none_flag":
        axes["max_depth"]["includes_none"] = False
    else:
        entry["status"] = "single_value_requires_exploration"
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    for digest in (None, sha256(path.read_bytes()).hexdigest()):
        with pytest.raises(ValueError, match="(?i)refinement"):
            evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("damage, error", [("axis", "Refinement axis"), ("tuning_complete", "CV-only"),
                                            ("defaults_frozen", "CV-only"), ("schema_version", "schema-2"),
                                            ("duplicate", "Duplicate JSON field")])
def test_completed_checkpoint_return_verifies_report_before_returning(tmp_path, damage, error):
    y = np.tile([0, 1], 12)
    path = tmp_path / "completed.json"

    def search():
        return evaluation.calibration_search(DecisionTreeClassifier(random_state=3), {"max_depth": [1, 2]},
                                             y[:, None], y, cv=2, seed=3, threads=1, refit=False,
                                             checkpoint_path=path).report

    search()
    saved = json.loads(path.read_bytes())
    saved["resume_eligible"] = False
    for damaged in (False, True):
        if damaged and damage == "axis":
            del saved["refinement_axis_inputs"]["max_depth"]
        elif damaged and damage != "duplicate":
            saved[damage] = 3 if damage == "schema_version" else True
        saved["checkpoint_sha256"] = evaluation._digest({k: v for k, v in saved.items() if k != "checkpoint_sha256"})
        encoded = json.dumps(saved)
        if damaged and damage == "duplicate":
            encoded = '{"tuning_complete": true,' + encoded[1:]
        path.write_text(encoded, encoding="utf-8")
        if damaged:
            with pytest.raises(ValueError, match=error):
                search()
        else:
            assert search() == saved
