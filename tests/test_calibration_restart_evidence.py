"""Computational restart checks the saved report against the live calibration."""

from copy import deepcopy
from hashlib import sha256
import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from supervised_nmf import evaluation


def search(path, scoring, *, refit=False):
    y = np.tile(np.repeat([0., 1.], [3, 5]), 2)
    first, second = np.arange(8), np.arange(8, 16)
    return evaluation.calibration_search(
        DecisionTreeClassifier(max_depth=1, random_state=7), {}, y[:, None], y,
        cv=2, folds=[(first, second), (second, first)], seed=7, threads=1,
        scoring=scoring, refit=refit, checkpoint_path=path,
    )


def reseal(path, report):
    report["checkpoint_sha256"] = evaluation._digest(
        {key: value for key, value in report.items() if key != "checkpoint_sha256"}
    )
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    return path.read_bytes()


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_ordinary_completed_cv_verifies_then_returns_saved_report(tmp_path, scoring):
    path = tmp_path / "cv.json"
    original = search(path, scoring).report
    before = path.read_bytes()
    assert original["selected_mean_score"] == 1.
    with patch.object(DecisionTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as write:
        result = search(path, scoring)
        assert result.estimator is None and result.report == evaluation.load_calibration_report(path)
        assert path.read_bytes() == before
        for field, value in (("class_labels", [.5, 1.5]), ("class_labels", [False, True]),
                             ("class_counts", [10, 6]), ("scoring", "unknown"),
                             ("selected_mean_score", .5), ("defaults_frozen", True)):
            altered = deepcopy(original)
            altered[field] = value
            before = reseal(path, altered)
            with pytest.raises(ValueError):
                search(path, scoring)
            assert path.read_bytes() == before
        fit.assert_not_called()
        write.assert_not_called()


def test_completed_cv_preserves_timing_and_pool_observations(tmp_path):
    path = tmp_path / "cv.json"
    # A coarse clock and an empty inventory are both legitimate observations.
    with patch.object(evaluation, "time", wraps=evaluation.time) as clock, patch.object(
        evaluation, "threadpool_info", return_value=[]
    ):
        clock.perf_counter.return_value = 12.
        search(path, "accuracy")
        before = path.read_bytes()
        original = json.loads(before)
        assert original["tuning_seconds"] == 0. and original["active_threadpools"] == []
        with patch.object(DecisionTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as write:
            assert evaluation.load_calibration_report(path) == original
            assert evaluation.load_calibration_report(path, expected_sha256=sha256(before).hexdigest()) == original
            assert search(path, "accuracy").report == original
            assert path.read_bytes() == before
            fit.assert_not_called()
            write.assert_not_called()


def test_completed_cv_requires_a_duration_and_list_of_pool_records(tmp_path):
    path = tmp_path / "cv.json"
    search(path, "accuracy")
    before = path.read_bytes()
    original = json.loads(before)
    assert original["tuning_seconds"] > 0.
    missing = object()
    with patch.object(DecisionTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as write:
        assert evaluation.load_calibration_report(path) == original
        assert evaluation.load_calibration_report(path, expected_sha256=sha256(before).hexdigest()) == original
        assert search(path, "accuracy").report == original
        assert path.read_bytes() == before
        for field, values in (("tuning_seconds", [missing, "1.0", -1., True, None]),
                              ("active_threadpools", [missing, {}, ["not a record"], None])):
            for value in values:
                altered = original.copy()
                if value is missing:
                    del altered[field]
                else:
                    altered[field] = value
                before = reseal(path, altered)
                for options in ({}, {"expected_sha256": sha256(before).hexdigest()}):
                    with pytest.raises(ValueError, match=field):
                        evaluation.load_calibration_report(path, **options)
                with pytest.raises(ValueError, match=field):
                    search(path, "accuracy")
                assert path.read_bytes() == before
        fit.assert_not_called()
        write.assert_not_called()


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
@pytest.mark.parametrize("state", ["calibrating", "refitting", "completed"])
def test_restart_binds_display_before_reusing_folds_or_refitting(tmp_path, scoring, state):
    path = tmp_path / "cv.json"
    refit = state != "calibrating"
    publish = evaluation.write_json

    class Interrupted(BaseException):
        pass

    def stop_at_checkpoint(path, report):
        publish(path, report)
        if report["state"] == state and report["candidates"][0]["folds"]:
            raise Interrupted

    with patch.object(evaluation, "write_json", stop_at_checkpoint), pytest.raises(Interrupted):
        search(path, scoring, refit=refit)
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    assert len(original["candidates"][0]["folds"]) == (2 if refit else 1)

    with patch.object(DecisionTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as write:
        for field in ("class_labels", "class_counts", "scoring", "folds"):
            altered = deepcopy(original)
            if field == "class_labels":
                altered[field] = [.5, 1.5]
            elif field == "class_counts":
                altered[field] = [6., 10.]
            elif field == "scoring":
                altered[field] = "balanced_accuracy" if scoring == "accuracy" else "accuracy"
            else:
                # Bool and int compare equal in Python, but only int identifies a row.
                altered[field][0]["training_indices"][0] = False
            before = reseal(path, altered)
            with pytest.raises(ValueError, match=field):
                search(path, scoring, refit=refit)
            assert path.read_bytes() == before
        fit.assert_not_called()
        write.assert_not_called()

    path.write_bytes(original_bytes)
    with patch.object(DecisionTreeClassifier, "fit", autospec=True, side_effect=DecisionTreeClassifier.fit) as fit:
        result = search(path, scoring, refit=refit)
        assert [len(call.args[1]) for call in fit.call_args_list] == ([16] if refit else [8])
    assert result.report["selected_mean_score"] == 1.
    assert (result.estimator is not None) == refit
