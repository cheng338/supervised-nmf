"""A completed CV report retains the evaluator's live callback flag decisions."""

import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from supervised_nmf import evaluation


class _TruthyZero(int):
    def __bool__(self):
        if hasattr(self, "tested"):
            raise AssertionError("truth checked twice")
        self.tested = True
        return True


class _FalsyOne(float):
    def __bool__(self):
        return False


class _DeadlineFlag(int):
    def __bool__(self):
        raise TimeoutError("during convergence check")


class _SelfRemovingFlag:
    def __init__(self, estimator):
        self.estimator = estimator

    def __bool__(self):
        del self.estimator.classifier_converged_
        return False


class _FlagCallback:
    def __init__(self, kind):
        self.kind = kind

    def setup(self, estimator, context):
        factories = {
            "truthy zero": lambda: _TruthyZero(0),
            "falsy one": lambda: _FalsyOne(1.0),
            "python int": lambda: 0,
            "numpy int": lambda: np.int64(0),
            "array": lambda: np.array([0]),
            "list": lambda: [0],
            "deadline": lambda: _DeadlineFlag(0),
            "self-removing": lambda: _SelfRemovingFlag(estimator),
        }
        estimator.classifier_converged_ = factories[self.kind]()

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context):
        pass

    def on_fit_task_end(self, estimator, context):
        return False


def _callback_report(path, callback, model_type=LogisticRegression):
    model = model_type(random_state=7)
    if not hasattr(model, "set_callbacks"):
        pytest.skip("This control uses sklearn fit callbacks.")
    model.set_callbacks(callback)
    y = np.tile([0, 1], 12)
    evaluation.calibration_search(model, {}, y[:, None], y, cv=2, seed=3, refit=False, checkpoint_path=path)
    saved = json.loads(path.read_bytes())
    assert evaluation.load_calibration_report(path) == saved
    return saved


@pytest.mark.parametrize("kind, display, passed, exact", [
    ("truthy zero", 0, True, False),
    ("falsy one", 1.0, False, False),
    ("python int", 0, False, True),
    ("numpy int", 0, False, False),
    ("array", [0], False, False),
    ("list", [0], True, False),
])
def test_public_callback_truth_is_independent_of_its_json_display(tmp_path, kind, display, passed, exact):
    report = _callback_report(tmp_path / "callback.json", _FlagCallback(kind))
    assert report["selected_candidate_index"] == (0 if passed else None)
    for fold in report["candidates"][0]["folds"]:
        assert fold["diagnostics"]["classifier_converged"] == display
        checks = fold["convergence_check"]["flag_checks"]
        assert checks["classifier_converged"] == {"passed": passed, "exact_scalar": exact}
        assert ("final_classifier_converged" in checks) is passed


def test_direct_tracked_callback_can_remove_its_flag_during_the_check(tmp_path):
    report = _callback_report(tmp_path / "removed.json", _FlagCallback("self-removing"),
                              evaluation._TrackedLogisticRegression)
    assert report["selected_candidate_index"] is None
    for fold in report["candidates"][0]["folds"]:
        assert fold["status"] == "nonconverged" and fold["warnings"] == []
        assert fold["diagnostics"]["converged"] is True
        assert "classifier_converged" not in fold["diagnostics"]
        check = fold["convergence_check"]
        assert check["reason"] == "estimator"
        assert check["flag_checks"]["classifier_converged"] == {"passed": False, "exact_scalar": False}


def test_interrupted_callback_flag_keeps_score_and_no_partial_convergence_check(tmp_path):
    report = _callback_report(tmp_path / "deadline.json", _FlagCallback("deadline"))
    for fold in report["candidates"][0]["folds"]:
        assert fold["status"] == "failed" and fold["error_type"] == "TimeoutError"
        assert fold["error"] == "during convergence check"
        assert fold["score"] == fold["accuracy"]
        assert "convergence_check" not in fold


class _WarningFlag(int):
    def __new__(cls, name, category, passed):
        flag = super().__new__(cls, int(passed))
        flag.name, flag.category, flag.passed = name, category, passed
        return flag

    def __bool__(self):
        if hasattr(self, "tested"):
            raise AssertionError("truth checked twice")
        self.tested = True
        warnings.warn(self.name, self.category)
        return self.passed


class _OrderedCallback(_FlagCallback):
    def setup(self, estimator, context):
        category = ConvergenceWarning if self.kind == "warning" else UserWarning
        estimator.last_transform_converged_ = _WarningFlag("projection", category, True)
        estimator.converged_ = _WarningFlag("outer", UserWarning, True)
        estimator.classifier_converged_ = _WarningFlag("classifier", UserWarning, False)


@pytest.mark.parametrize("reason", ["estimator", "warning"])
def test_public_callback_preserves_warning_boundary_and_short_circuit(tmp_path, reason):
    report = _callback_report(tmp_path / "ordered.json", _OrderedCallback(reason))
    for fold in report["candidates"][0]["folds"]:
        check = fold["convergence_check"]
        assert check["reason"] == reason and check["warnings_checked"] == 1
        expected = ["projection"] if reason == "warning" else ["projection", "outer", "classifier"]
        assert [w["message"] for w in fold["warnings"]] == expected
        evaluated = {"last_transform_converged"}
        if reason == "estimator":
            evaluated.update(("converged", "classifier_converged"))
        assert check["flag_checks"].keys() == evaluated
