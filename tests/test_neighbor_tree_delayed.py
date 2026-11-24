"""Selected tree predictions must preserve sklearn's dispatch, labels and warnings."""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import warnings


def _observe(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, extra = call()
        except RuntimeError as error:
            result = {"error_type": type(error).__name__, "error": str(error)}
        else:
            result = {"value": value}
            caught.extend(extra)
    result["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    return result


def _measure(directory, events):
    import numpy as np
    from supervised_nmf import study
    from supervised_nmf._neighbors import _KNNQueryReuse
    from tests.test_full_query_direct_methods import head, problem

    X, y, folds = problem()
    counts = (1, 3)
    events.clear()
    expected, expected_labels = [], []
    for train, valid in folds:
        for k in counts:
            model = head(k, "kd_tree").fit(X[train], y[train])

            def predict():
                labels = model.predict(X[valid])
                expected_labels.append(labels.tolist())
                return float(np.mean(labels == y[valid])), []

            expected.append(_observe(predict))
    native = {"scores": expected, "labels": expected_labels, "events": events.copy()}

    events.clear()
    labels = []
    score = study._score_predictions

    def score_predictions(scoring, y_true, y_pred):
        labels.append(y_pred.tolist())
        return score(scoring, y_true, y_pred)

    with patch.object(study, "_score_predictions", score_predictions):
        report = study.calibration_knn_grid(head(1, "kd_tree"), {"n_neighbors": list(counts)},
                                            X, y, folds=folds, cv=2, threads=1, checkpoint_dir=directory)
    rows = []
    for index in range(len(folds)):
        for candidate in report["candidates"]:
            row = candidate["folds"][index]
            outcome = {"value": row["score"]} if row["status"] == "valid" else {
                "error_type": row["error_type"], "error": row["error"]}
            rows.append({**outcome, "warnings": row["warnings"]})
    actual = {"scores": rows, "labels": labels, "events": events.copy()}

    train, valid = folds[0]
    training, query = X[train], X[valid[:2]]
    models = {k: head(k, "auto").set_params(weights="distance").fit(training, y[train]) for k in counts}
    assert all(model._fit_method == "kd_tree" for model in models.values())
    events.clear()
    native["probabilities"] = [_observe(lambda: (model.predict_proba(query).tolist(), []))
                               for model in models.values()]
    native["probability_events"] = events.copy()
    events.clear()
    reuse = _KNNQueryReuse(models, query, _head_fit_input=training)

    def probabilities(k):
        values, _, shared = reuse.predict(k, probabilities=True, decision_threshold=0.5, positive_index=1)
        return values.tolist(), shared

    actual["probabilities"] = [_observe(lambda: probabilities(k)) for k in counts]
    actual["probability_events"] = events.copy()
    queried = [q["queried"] for fold in report["representation_folds"] for q in fold["neighbor_queries"]]
    return {"native": native, "actual": actual, "queried": queried}


def _phases(directory, events, name, owners, replacement, *, caller_alias=True):
    from threadpoolctl import threadpool_limits

    assert "sklearn.neighbors._base" not in sys.modules and "supervised_nmf" not in sys.modules
    original = getattr(owners[0], name)
    with threadpool_limits(1):
        with ExitStack() as changes:
            for owner in owners:
                changes.enter_context(patch.object(owner, name, replacement))
            from sklearn.neighbors import _base
            if caller_alias:
                assert getattr(_base, name) is replacement
            before = _measure(directory / "before", events)
        if caller_alias:
            setattr(_base, name, original)
        stock = _measure(directory / "stock", events)
        with ExitStack() as changes:
            for owner in ((*owners, _base) if caller_alias else owners):
                changes.enter_context(patch.object(owner, name, replacement))
            after = _measure(directory / "after", events)
        restored = _measure(directory / "restored", events)
    return {"before_import": before, "stock": stock, "after_capture": after, "restored": restored}


def _delayed(directory):
    from sklearn.utils import parallel

    original, events = parallel.delayed, []

    def delayed(function):
        scheduled = original(function)
        if getattr(function, "__name__", None) != "query":
            return scheduled

        def schedule(X, k, return_distance):
            events.append([k, return_distance])
            warnings.warn(f"selected KDTree width {k}", UserWarning)
            if k == 3:
                raise RuntimeError("selected KDTree width 3 failed")
            return scheduled(X[::-1], k, return_distance)

        return schedule

    return _phases(directory, events, "delayed", (parallel,), delayed)


def _collect(script, name, directory, paths=None):
    if paths is None:
        paths = [str(Path(entry or ".").resolve()) for entry in sys.path]
    child = subprocess.run([sys.executable, "-I", "-B", str(script), json.dumps(paths), name, str(directory)],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads(child.stdout)


def _check_selected(native, name):
    if name == "gen_even_slices":
        calls, probability_calls = [[12, 1]] * 4, [[2, 1]] * 2
    elif name in ("redecorated_slicer", "logger_init"):
        marker = "validate n" if name == "redecorated_slicer" else "construct"
        calls, probability_calls = [marker] * 4, [marker] * 2
    else:
        calls = [[1, False], [3, False]] * 2
        probability_calls = [[1, True], [3, True]]
    message = {"delayed": "selected KDTree width 3 failed", "Parallel": "selected Parallel width 3",
               "joblib_call": "selected Parallel width 3", "gen_even_slices": "second selected slice request",
               "redecorated_slicer": "second selected slice request", "executed_task": "selected task width 3",
               "logger_init": "second Parallel construction"}[name]
    assert native["events"] == calls
    assert native["probability_events"] == probability_calls
    assert all("value" in row for row in [*native["scores"][::2], native["probabilities"][0]])
    for row in [*native["scores"][1::2], native["probabilities"][1]]:
        assert (row["error_type"], row["error"]) == ("RuntimeError", message)


def _check(evidence, name):
    for phase, record in evidence.items():
        native = record["native"]
        assert record["actual"] == native, phase
        if phase in ("before_import", "after_capture"):
            assert record["queried"] == [False, False]
            _check_selected(native, name)
            if name in ("delayed", "executed_task"):
                stock = evidence["stock"]["native"]
                assert native["labels"][0] != stock["labels"][0]
                assert native["probabilities"][0]["value"] != stock["probabilities"][0]["value"]
        else:
            assert native["events"] == native["probability_events"] == []
            assert all("value" in row for row in native["scores"] + native["probabilities"])
    assert evidence["restored"] == evidence["stock"]


def test_selected_tree_delayed(tmp_path):
    evidence = _collect(Path(__file__).resolve(), "delayed", tmp_path)
    _check(evidence, "delayed")
    for row in evidence["before_import"]["native"]["scores"][1::2]:
        assert row == {"error_type": "RuntimeError", "error": "selected KDTree width 3 failed",
                       "warnings": [{"category": "UserWarning", "message": "selected KDTree width 3"}]}


if __name__ == "__main__":
    sys.path[:] = json.loads(sys.argv[1])
    print(json.dumps(_delayed(Path(sys.argv[3]))))
