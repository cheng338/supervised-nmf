"""The grouped example caps final fitting and restores its caller's capacity."""

import functools
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg.blas import dgemv
from threadpoolctl import threadpool_info, threadpool_limits

from supervised_nmf import NMFClassifier


ROOT = Path(__file__).resolve().parents[1]


def widths():
    pools = threadpool_info()
    assert pools, "The numerical test must observe actual native pools"
    return {row["filepath"]: row["num_threads"] for row in pools}


@pytest.fixture
def example():
    original_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("grouped_example_threads_test", ROOT / "examples/grouped_tuning.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        # Some MKL runtimes discover an additional native pool on their first
        # real BLAS call. Initialize it before testing restoration, outside the
        # example invocation and before establishing the tested ambient limit.
        with threadpool_limits(limits=1):
            values = np.ones((32, 32))
            np.dot(values, values)
            dgemv(1.0, values, np.ones(32))
        yield module
    finally:
        sys.path[:] = original_path


@pytest.mark.parametrize("ambient", [1, 2])
@pytest.mark.parametrize("outcome", ["success", "fit-error", "interrupted", "no-selection", "nonconverged"])
def test_final_fit_budget_and_branch_restoration(example, monkeypatch, tmp_path, ambient, outcome):
    events = []
    selected = {"n_components": 4, "score_threshold": 0.0, "n_neighbors": 3}
    expected_error = (ValueError("final fit failed") if outcome == "fit-error" else
                      KeyboardInterrupt("final fit interrupted") if outcome == "interrupted" else None)

    class Classifier:
        def __init__(self, **kwargs):
            assert kwargs == {"screening": "absolute", "random_state": 42}

        def set_params(self, **kwargs):
            assert kwargs == selected
            events.append("set-parameters")
            return self

        def fit(self, X, y):
            events.append("fit")
            assert set(widths().values()) == {1}
            assert X.shape == (120, 16)
            np.testing.assert_array_equal(y, np.tile([0, 0, 1, 1], 30))
            if expected_error is not None:
                raise expected_error
            self.converged_ = outcome != "nonconverged"
            self.fit_times_ = {"total": 0.0}
            return self

    def search(estimator, grid, X, y, **kwargs):
        events.append("search")
        assert set(widths().values()) == {ambient}
        assert grid == {"n_components": [2, 4], "score_threshold": [0.0, 2.0], "n_neighbors": [3, 5]}
        assert kwargs["threads"] == 1 and kwargs["cv"] == 5 and kwargs["seed"] == 42
        assert kwargs["max_expansions"] == 1 and kwargs["max_rank"] == 8
        np.testing.assert_array_equal(kwargs["groups"], np.repeat(np.arange(30), 4))
        np.testing.assert_array_equal(kwargs["data_ids"], np.arange(120))
        return {"stop_reason": "fixture", "selected_parameters": None if outcome == "no-selection" else selected,
                "tuning_seconds": 0.0}

    def evaluate(estimator, X, y, **kwargs):
        events.append("evaluation")
        # The final-fit scope must have ended; evaluation owns its own explicit
        # threads=1 policy, rather than inheriting a leaked fitting scope.
        assert set(widths().values()) == {ambient}
        assert kwargs["threads"] == 1 and X.shape == (40, 16)
        np.testing.assert_array_equal(kwargs["sample_ids"], np.arange(120, 160))
        np.testing.assert_array_equal(kwargs["train_sample_ids"], np.arange(120))
        np.testing.assert_array_equal(kwargs["groups"], np.repeat(np.arange(30, 40), 4))
        np.testing.assert_array_equal(kwargs["train_groups"], np.repeat(np.arange(30), 4))
        np.testing.assert_array_equal(kwargs["train_y"], np.tile([0, 0, 1, 1], 30))
        return {"status": "valid"}

    monkeypatch.setattr(example, "NMFClassifier", Classifier)
    monkeypatch.setattr(example, "calibration_rank_search", search)
    monkeypatch.setattr(example, "evaluate_classifier", evaluate)
    monkeypatch.setattr(example, "format_evaluation_summary", lambda report: report["evaluation"]["status"])
    monkeypatch.setattr(sys, "argv", [str(example.__file__), "--output", str(tmp_path / "cp")])
    original = widths()
    with threadpool_limits(limits=ambient):
        before = widths()
        assert set(before.values()) == {ambient}
        if expected_error is not None:
            with pytest.raises(type(expected_error)) as caught:
                example.main()
            assert caught.value is expected_error
        elif outcome == "no-selection":
            with pytest.raises(RuntimeError, match="No numerically qualified training candidate"):
                example.main()
        elif outcome == "nonconverged":
            with pytest.raises(RuntimeError, match="selected final fit did not converge"):
                example.main()
        else:
            example.main()
        assert widths() == before
    assert widths() == original
    expected = ["search"] if outcome == "no-selection" else ["search", "set-parameters", "fit"]
    if outcome == "success":
        expected.append("evaluation")
    assert events == expected


def test_real_grouped_workflow_and_completed_resume_keep_one_thread(example, monkeypatch, tmp_path_factory):
    # A short checkpoint root avoids making hashed fold filenames depend on
    # the length of the test's descriptive name on Windows.
    checkpoint = tmp_path_factory.mktemp("gt") / "cp"
    monkeypatch.setattr(sys, "argv", [str(example.__file__), "--output", str(checkpoint)])
    phase = "final"
    events, searches, evaluations, final_models = [], [], [], []
    original_fit = NMFClassifier._fit_impl
    original_prediction = NMFClassifier.predict_with_probabilities
    original_search, original_evaluate = example.calibration_rank_search, example.evaluate_classifier

    @functools.wraps(original_fit)
    def fit(self, *args, **kwargs):
        # Shared-grid CV may use _fit_with_screening, so observe its common
        # fitting operation as well as the example's ordinary direct .fit.
        events.append((phase, widths()))
        result = original_fit(self, *args, **kwargs)
        if phase == "final":
            final_models.append({"components": self.components_.copy(),
                                 "coordinates": self.training_coordinates_.copy(),
                                 "converged": self.converged_})
        return result

    @functools.wraps(original_prediction)
    def predict(self, *args, **kwargs):
        events.append(("prediction", widths()))
        return original_prediction(self, *args, **kwargs)

    @functools.wraps(original_search)
    def search(*args, **kwargs):
        nonlocal phase
        phase = "cv"
        try:
            report = original_search(*args, **kwargs)
            searches.append(report)
            return report
        finally:
            phase = "final"

    @functools.wraps(original_evaluate)
    def evaluate(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        evaluations.append(report)
        return report

    def immutable_receipts():
        return {path.relative_to(checkpoint): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in checkpoint.rglob("*.json") if path.name != "report.json"}

    monkeypatch.setattr(NMFClassifier, "_fit_impl", fit)
    monkeypatch.setattr(NMFClassifier, "predict_with_probabilities", predict)
    monkeypatch.setattr(example, "calibration_rank_search", search)
    monkeypatch.setattr(example, "evaluate_classifier", evaluate)
    original = widths()
    with threadpool_limits(limits=2):
        before = widths()
        assert set(before.values()) == {2}
        example.main()
        assert widths() == before
        fresh_events = list(events)
        saved = immutable_receipts()
        assert saved
        events.clear()
        example.main()
        assert widths() == before
        assert immutable_receipts() == saved
    assert widths() == original
    assert any(kind == "cv" for kind, _ in fresh_events)
    assert [kind for kind, _ in events] == ["final", "prediction"]
    assert all(set(pools.values()) == {1} for _, pools in fresh_events + events)
    assert searches[0]["new_representation_fits"] > 0
    assert searches[1]["new_representation_fits"] == 0 and searches[1]["reused_representation_folds"] > 0
    assert searches[0]["selected_parameters"] == searches[1]["selected_parameters"]
    assert len(final_models) == len(evaluations) == 2
    assert all(model["converged"] for model in final_models)
    for key in ("components", "coordinates"):
        np.testing.assert_array_equal(final_models[0][key], final_models[1][key])
    assert evaluations[0]["status"] == evaluations[1]["status"] == "valid"
    assert evaluations[0]["predictions"] == evaluations[1]["predictions"]
    assert evaluations[0]["metrics"] == evaluations[1]["metrics"]
    assert evaluations[0]["predictions"]["sample_ids"] == list(range(120, 160))
    assert set(evaluations[0]["evaluation_groups"]) == set(range(30, 40))
