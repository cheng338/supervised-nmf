"""Observed configuration changes affect computational recovery at fold publication."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import ExtraTreeClassifier

from supervised_nmf import evaluation


class Interrupted(BaseException):
    pass


def inputs():
    rng = np.random.RandomState(892)
    X = rng.normal(size=(45, 5))
    y = (X[:, 0] + X[:, 1] * .5 > 0).astype(int)
    return X, y


def configuration(kind):
    if kind == "counter":
        calls = [0]

        def transform(X):
            calls[0] += 1
            return X
    else:
        # Keep the projection's ordinary draw state across calls. A live RNG
        # has a separate checkpoint contract, tested below.
        state = [np.random.RandomState(13).get_state()]

        def transform(X):
            stream = np.random.RandomState(0)
            stream.set_state(state[0])
            projection = stream.normal(size=(X.shape[1], 3))
            if kind == "random" or len(X) == 30:
                state[0] = stream.get_state()
            return X @ projection

    model = Pipeline([("map", FunctionTransformer()), ("tree", ExtraTreeClassifier(random_state=17))])
    return model, {"map__func": [transform] if kind == "refit-only" else [None, transform]}


def run(path, kind, *, refit=True, comparison=False):
    X, y = inputs()
    model, grid = configuration(kind)
    if comparison:
        return evaluation.run_comparison({"a": (model, grid)}, X[:30], y[:30], X[30:], y[30:],
                                         output_dir=path, seed=5)["models"]["a"]
    return evaluation.calibration_search(model, grid, X[:30], y[:30], cv=3, seed=5,
                                         checkpoint_path=path, refit=refit)


def saved(path):
    raw = json.loads(path.read_bytes())
    assert raw["checkpoint_sha256"] == evaluation._digest(
        {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    )
    return raw


@pytest.mark.parametrize("kind", ["counter", "random"])
def test_first_changed_fold_is_committed_and_computational_retry_is_read_only(tmp_path, kind):
    file = tmp_path / "cv.json"
    write = evaluation.write_json

    def stop_at_mark(path, report):
        marked = report.get("resume_eligible") is False
        if not marked:
            assert report["candidates"][1]["folds"] == []
        write(path, report)
        if marked:
            raise Interrupted

    with patch.object(evaluation, "write_json", stop_at_mark), pytest.raises(Interrupted):
        run(file, kind)
    before = file.read_bytes()
    report = saved(file)
    assert [len(row["folds"]) for row in report["candidates"]] == [3, 1]
    assert report["candidates"][1]["folds"][0]["status"] == "valid"
    assert report["resume_eligible"] is False
    assert report["resume_ineligibility_candidate_index"] == 1
    assert report["resume_ineligibility_fold"] == 0
    assert "candidate 1" in report["resume_ineligibility_reason"].lower()
    assert "fold 0" in report["resume_ineligibility_reason"].lower()
    with patch.object(ExtraTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as writes:
        with pytest.raises(ValueError, match="ineligible"):
            run(file, kind)
    fit.assert_not_called()
    writes.assert_not_called()
    assert file.read_bytes() == before


def test_marked_completed_reports_are_available_as_archives(tmp_path):
    cv = tmp_path / "cv.json"
    first = run(cv, "counter", refit=False)
    assert first.estimator is None and first.report["resume_eligible"] is False
    before = cv.read_bytes()
    with patch.object(ExtraTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as writes:
        archived = run(cv, "counter", refit=False)
    assert archived.estimator is None and archived.report == first.report
    fit.assert_not_called()
    writes.assert_not_called()
    assert cv.read_bytes() == before

    comparison = tmp_path / "comparison"
    record = run(comparison, "counter", comparison=True)
    journal = comparison / "a-cv.json"
    cv_report = saved(journal)
    assert cv_report["resume_eligible"] is False
    for key in ("resume_eligible", "resume_ineligibility_reason", "resume_ineligibility_candidate_index",
                "resume_ineligibility_fold"):
        assert record["calibration_" + key] == cv_report[key]
    assert record["evaluation"]["status"] == "valid"
    before = {path: path.read_bytes() for path in comparison.glob("*.json")}
    with patch.object(ExtraTreeClassifier, "fit") as fit:
        assert run(comparison, "counter", comparison=True) == record
    fit.assert_not_called()
    assert all(path.read_bytes() == contents for path, contents in before.items())
    (comparison / "a-evaluation.json").unlink()
    with patch.object(ExtraTreeClassifier, "fit") as fit, patch.object(evaluation, "write_json") as writes:
        with pytest.raises(ValueError, match="ineligible"):
            run(comparison, "counter", comparison=True)
    fit.assert_not_called()
    writes.assert_not_called()


@pytest.mark.parametrize("draw", [False, True], ids=["unused", "advancing"])
def test_completed_cv_checks_rng_captured_by_callback(tmp_path, draw):
    X, y = inputs()
    path = tmp_path / "cv.json"

    def search():
        rng = np.random.RandomState(13)

        def transform(X):
            if draw:
                return X @ rng.normal(size=(X.shape[1], 3))
            return X

        model = Pipeline([("map", FunctionTransformer(transform)), ("tree", ExtraTreeClassifier(random_state=17))])
        return evaluation.calibration_search(model, {}, X, y, cv=3, seed=5, refit=False, checkpoint_path=path)

    first = search().report
    assert first["state"] == "cv_completed"
    assert [fold["status"] for fold in first["candidates"][0]["folds"]] == ["valid"] * 3
    if draw:
        assert first["resume_eligible"] is False
        assert (first["resume_ineligibility_candidate_index"], first["resume_ineligibility_fold"]) == (0, 0)
    else:
        assert first.get("resume_eligible", True)
    before = path.read_bytes()

    with patch.object(ExtraTreeClassifier, "fit") as fit, patch.object(
        evaluation, "write_json", wraps=evaluation.write_json
    ) as writes:
        if np.__version__.split(".", 1)[0] == "2":
            archived = search()
            assert archived.estimator is None
            assert archived.report["candidates"] == first["candidates"]
            if draw:
                writes.assert_not_called()
                assert path.read_bytes() == before
        else:
            with pytest.raises(ValueError, match="verifiable clone behavior"):
                search()
            writes.assert_not_called()
            assert path.read_bytes() == before
        fit.assert_not_called()


def test_refit_only_mutation_can_repeat_full_training_fit(tmp_path):
    X, _ = inputs()
    file = tmp_path / "refit.json"
    reference = run(file, "refit-only")
    assert reference.report.get("resume_eligible", True)
    retained = saved(file)["candidates"]
    fits = []
    original_fit = ExtraTreeClassifier.fit

    def count(tree, X, y):
        fits.append(len(X))
        return original_fit(tree, X, y)

    with patch.object(ExtraTreeClassifier, "fit", count):
        recovered = run(file, "refit-only")
    assert fits == [30] and saved(file)["candidates"] == retained
    np.testing.assert_array_equal(reference.estimator.predict(X[30:]), recovered.estimator.predict(X[30:]))
    np.testing.assert_array_equal(reference.estimator.predict_proba(X[30:]), recovered.estimator.predict_proba(X[30:]))


def crash_at_mark(root, when):
    file = root / f"{when}.json"
    replace = Path.replace

    def publish(temporary, destination):
        if destination == file and saved(temporary).get("resume_eligible") is False:
            # Both payloads pass their own checksum before killing the writer.
            assert [len(c["folds"]) for c in saved(file)["candidates"]] == [3, 0]
            if when == "after":
                replace(temporary, destination)
            os._exit(73)
        return replace(temporary, destination)

    with patch.object(Path, "replace", publish):
        run(file, "random")
    raise AssertionError("No marked fold reached atomic replacement")


def resume_after_crashes(root):
    before, after = root / "before.json", root / "after.json"
    prefix = saved(before)["candidates"][0]["folds"]
    fit = ExtraTreeClassifier.fit
    with patch.object(ExtraTreeClassifier, "fit", autospec=True, side_effect=fit) as fits:
        recovered = run(before, "random")
    assert [len(call.args[1]) for call in fits.call_args_list] == [20, 20, 20, 30]
    assert recovered.report["candidates"][0]["folds"] == prefix
    assert recovered.report["resume_eligible"] is False
    original = after.read_bytes()
    with patch.object(ExtraTreeClassifier, "fit") as fits, patch.object(evaluation, "write_json") as writes:
        with pytest.raises(ValueError, match="ineligible"):
            run(after, "random")
    fits.assert_not_called()
    writes.assert_not_called()
    assert after.read_bytes() == original


def test_abrupt_exit_on_either_side_of_atomic_mark_publication(tmp_path):
    command = [sys.executable, str(Path(__file__).resolve()), str(tmp_path)]
    env = dict(os.environ, PYTHONPATH=str(Path(evaluation.__file__).resolve().parents[1]))
    for when, marked in [("before", False), ("after", True)]:
        child = subprocess.run([*command, when], env=env, timeout=120)
        assert child.returncode == 73
        report = saved(tmp_path / f"{when}.json")
        assert report.get("resume_eligible", True) is (not marked)
        assert [len(c["folds"]) for c in report["candidates"]] == [3, int(marked)]
    subprocess.run([*command, "resume"], env=env, check=True, timeout=120)


def test_numeric_array_value_changes_are_outside_fold_observation(tmp_path):
    # Calibration sees the array's shape and dtype. Its eight mutable data bytes
    # can still change predictions without producing an ineligibility marker.
    state = np.array([0], dtype=np.int64)

    def changing_projection(X):
        state[0] += 1
        rng = np.random.RandomState(int(state[0]))
        return X @ rng.normal(size=(X.shape[1], 3))

    X, y = inputs()
    model = Pipeline([("map", FunctionTransformer(changing_projection)),
                      ("tree", ExtraTreeClassifier(random_state=17))])
    result = evaluation.calibration_search(model, {}, X, y, cv=3, seed=5, refit=False,
                                           checkpoint_path=tmp_path / "cv.json")
    assert state.tolist() == [6]
    assert result.report.get("resume_eligible", True)
    assert all(fold["status"] == "valid" for fold in result.report["candidates"][0]["folds"])


@pytest.mark.parametrize("checkpointed", [False, True])
def test_source_checks_keep_their_order_around_fold_diagnostics(tmp_path, monkeypatch, checkpointed):
    events = []
    tree = ExtraTreeClassifier(random_state=17, max_depth=1)
    source = evaluation.implementation_sources
    diagnostics = evaluation._diagnostics

    def record_source(*args, **kwargs):
        if kwargs.get("_previous_files") is not None:
            events.append("observe" if kwargs.get("_observation") is not None else "source")
        return source(*args, **kwargs)

    def change_during_diagnostics(*args):
        events.append("diagnostics")
        tree.max_depth = 3
        return diagnostics(*args)

    monkeypatch.setattr(evaluation, "implementation_sources", record_source)
    monkeypatch.setattr(evaluation, "_diagnostics", change_during_diagnostics)
    X, y = inputs()
    result = evaluation.calibration_search(tree, {}, X, y, cv=2, refit=False,
                                           checkpoint_path=tmp_path / "cv.json" if checkpointed else None)
    ordinary = ["source", "source", "diagnostics"]
    if checkpointed:
        assert result.report["resume_ineligibility_fold"] == 0
        assert events == ["source", "diagnostics", "observe", *ordinary]
    else:
        assert events == ordinary * 2


if __name__ == "__main__":
    root, phase = Path(sys.argv[1]), sys.argv[2]
    if phase == "resume":
        resume_after_crashes(root)
    else:
        crash_at_mark(root, phase)
