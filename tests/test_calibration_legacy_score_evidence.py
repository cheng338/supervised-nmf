"""Path-normalized fixtures derived from genuine NumPy 2.2.6 / sklearn 1.9.0 archives.

The historical writer scored both candidates and saved the auxiliary class counts.
The public fixtures replace local paths and host descriptions, then reseal the hashes.
"""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from supervised_nmf import evaluation


FIXTURES = Path(__file__).with_name("fixtures") / "calibration_legacy_score"


@pytest.mark.parametrize("pinned", [False, True])
def test_legacy_score_attainable_on_both_supports_remains_readable(pinned):
    path = FIXTURES / "genuine-4x4.json"
    digest = sha256(path.read_bytes()).hexdigest() if pinned else None
    report = evaluation.load_calibration_report(path, expected_sha256=digest)
    first, second = report["candidates"]
    assert report["identity"]["validation_class_counts"] == [[4, 4], [4, 4]]
    assert first["mean_score"] == second["mean_score"] == .625
    assert report["selected_candidate_index"] == 0
    assert all(fold["balanced_accuracy_support"]["class_counts"] == [4, 3] for fold in second["folds"])


@pytest.mark.parametrize("pinned", [False, True])
def test_legacy_auxiliary_counts_alone_cannot_reverse_winner(tmp_path, pinned):
    report = json.loads((FIXTURES / "genuine-4x4.json").read_bytes())
    first, second = report["candidates"]
    assert first["mean_score"] == second["mean_score"] == .625
    assert report["selected_candidate_index"] == 0
    # sklearn counted 3/4 and 2/4 correct. Only the writer's auxiliary
    # lookup discarded the final wrong prediction by the second candidate.
    score = (3 / 4 + 2 / 3) / 2
    for fold in second["folds"]:
        assert fold["score"] == .625
        assert fold["balanced_accuracy_support"]["class_counts"] == [4, 3]
        fold["score"] = fold["balanced_accuracy"] = score
    second["mean_score"] = second["mean_balanced_accuracy"] = score
    report.update(selected_candidate_index=1, selected_parameters=second["parameters"],
                  selected_mean_score=score,
                  refinement=evaluation.refinement_diagnostics([first["parameters"], second["parameters"]],
                                                               second["parameters"]))
    report["checkpoint_sha256"] = evaluation._digest({k: v for k, v in report.items() if k != "checkpoint_sha256"})
    path = tmp_path / "altered-winner.json"
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    digest = sha256(path.read_bytes()).hexdigest() if pinned else None
    with pytest.raises(ValueError, match="(?i)balanced accuracy"):
        evaluation.load_calibration_report(path, expected_sha256=digest)


@pytest.mark.parametrize("pinned", [False, True])
def test_legacy_genuine_score_with_contradictory_auxiliary_counts_is_refused(pinned):
    path = FIXTURES / "genuine-4x5.json"
    raw = path.read_bytes()
    report = json.loads(raw)
    assert report["identity"]["validation_class_counts"] == [[4, 5], [4, 5]]
    for fold in report["candidates"][1]["folds"]:
        assert fold["score"] == (3 / 4 + 2 / 5) / 2 == .575
        assert fold["balanced_accuracy_support"]["class_counts"] == [4, 4]
    digest = sha256(raw).hexdigest() if pinned else None
    with pytest.raises(ValueError, match="(?i)balanced accuracy"):
        evaluation.load_calibration_report(path, expected_sha256=digest)
