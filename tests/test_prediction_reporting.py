"""Prediction reports must describe their own call, including failed stages."""

from copy import deepcopy
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from supervised_nmf import FeatureSPA, NMFClassifier
from supervised_nmf.evaluation import (
    _diagnostics, _nonconverged, calibration_search, evaluate_classifier,
    format_evaluation_summary,
)
from supervised_nmf.study import calibration_knn_grid


@pytest.fixture
def data():
    rng = np.random.default_rng(927)
    y = np.arange(60) % 2
    coefficients = rng.uniform(.1, 1, (60, 2))
    coefficients[:, 0] += y
    basis = np.column_stack((np.eye(2), rng.uniform(.1, 1, (2, 6))))
    return coefficients @ basis + rng.uniform(0, .0001, (60, 8)), y


@pytest.fixture
def fitted(data):
    X, y = data
    return NMFClassifier(screening="none", n_components=2, n_neighbors=3,
                         random_state=11, max_iter=500).fit(X, y)


def stale_diagnostics(model):
    model.last_transform_converged_ = False
    model.last_transform_warnings_ = ("stale warning",)
    model.last_transform_error_ = "stale error"
    model.last_transform_seconds_ = -1.0
    return {k: deepcopy(v) for k, v in vars(model).items() if k.startswith("last_transform_")}


def test_success_report_ignores_but_does_not_mutate_old_saved_fields(fitted, data):
    X, y = data
    stale = stale_diagnostics(fitted)
    report = evaluate_classifier(fitted, X, y)
    diagnostic = report["diagnostics"]
    assert report["status"] == "valid"
    assert diagnostic["last_transform_performed"] is True
    assert diagnostic["last_transform_converged"] is True
    assert diagnostic["last_transform_warnings"] == []
    assert diagnostic["last_transform_error"] is None
    assert diagnostic["last_transform_seconds"] >= 0
    assert {k: vars(fitted)[k] for k in stale} == stale


@pytest.mark.parametrize("outcome", ["warning", "warning_error", "invalid_width", "head_error"])
def test_failed_stage_and_warning_reports_are_call_local(fitted, data, monkeypatch, outcome):
    X, y = data
    stale = stale_diagnostics(fitted)
    if outcome.startswith("warning"):
        def projection(rows):
            warnings.warn("current projection warning", ConvergenceWarning)
            if outcome == "warning_error":
                raise ValueError("current projection failure")
            return np.zeros((len(rows), 2))
        monkeypatch.setattr(fitted.decomposition_, "transform", projection)
    elif outcome == "invalid_width":
        X = X[:, :-1]
    else:
        def head_error(rows):
            raise RuntimeError("current classifier failure")
        monkeypatch.setattr(fitted.classifier_, "predict_proba", head_error)
    report = evaluate_classifier(fitted, X, y)
    diagnostic = report["diagnostics"]
    assert report["status"] == ("provisional_nonconvergence" if outcome == "warning" else "failed")
    assert diagnostic["last_transform_performed"] is True
    assert diagnostic["last_transform_converged"] is (outcome == "head_error")
    assert diagnostic["converged"] is True
    if outcome.startswith("warning"):
        assert diagnostic["last_transform_warnings"] == ["current projection warning"]
        assert report["warnings"] == [{"category": "ConvergenceWarning", "message": "current projection warning"}]
    else:
        assert diagnostic["last_transform_warnings"] == []
    if outcome in ("warning", "head_error"):
        assert diagnostic["last_transform_error"] is None
    else:
        assert diagnostic["last_transform_error"].startswith("ValueError:")
    assert {k: vars(fitted)[k] for k in stale} == stale


def test_projection_success_does_not_erase_fit_nonconvergence(fitted, data):
    fitted.converged_ = False
    fitted.convergence_warnings_ = ("earlier fit warning",)
    report = evaluate_classifier(fitted, *data)
    assert report["status"] == "provisional_nonconvergence"
    assert report["diagnostics"]["last_transform_converged"] is True
    assert report["diagnostics"]["convergence_warnings"] == ["earlier fit warning"]


def test_fit_only_diagnostics_have_no_prediction_evidence(fitted):
    stale_diagnostics(fitted)
    diagnostic = _diagnostics(fitted)
    assert diagnostic["last_transform_performed"] is False
    assert diagnostic["last_transform_converged"] is None
    assert diagnostic["last_transform_error"] is None
    assert diagnostic["last_transform_warnings"] == []
    assert not _nonconverged(fitted, [])
    assert "| Latest projection | not performed |" in format_evaluation_summary({"diagnostics": diagnostic})


class StaleAfterFitClassifier(NMFClassifier):
    """Emulate old saved diagnostic fields without overriding prediction."""

    def fit(self, X, y):
        super().fit(X, y)
        stale_diagnostics(self)
        return self


def test_calibration_prediction_and_final_fit_use_separate_records(data):
    model = StaleAfterFitClassifier(screening="none", n_components=2,
                                   n_neighbors=3, random_state=11, max_iter=500)
    result = calibration_search(model, {}, *data, cv=3, threads=1)
    assert result.report["is_fallback"] is False
    for fold in result.report["candidates"][0]["folds"]:
        assert fold["status"] == "valid"
        assert fold["diagnostics"]["last_transform_performed"] is True
        assert fold["diagnostics"]["last_transform_converged"] is True
        assert fold["diagnostics"]["last_transform_error"] is None
    final = result.report["final_fit_attempts"][0]
    assert final["status"] == "valid"
    assert final["diagnostics"]["last_transform_performed"] is False
    assert final["diagnostics"]["last_transform_converged"] is None
    assert result.estimator.last_transform_error_ == "stale error"


class LegacyTransformClassifier(NMFClassifier):
    """A custom transform retains its separately managed diagnostic contract."""

    def transform(self, X):
        result = super().transform(X)
        self.last_transform_converged_ = False
        self.last_transform_warnings_ = ("custom projection qualification",)
        self.last_transform_error_ = None
        return result


def test_custom_transform_legacy_diagnostics_remain_effective(data):
    X, y = data
    model = LegacyTransformClassifier(screening="none", n_components=2,
                                      random_state=11, max_iter=500).fit(X, y)
    report = evaluate_classifier(model, X, y)
    assert report["status"] == "provisional_nonconvergence"
    assert report["diagnostics"]["last_transform_converged"] is False
    assert report["diagnostics"]["last_transform_warnings"] == ["custom projection qualification"]


@pytest.mark.parametrize("outcome", ["success", "warning", "error"])
def test_shared_representation_projection_keeps_its_diagnostics(data, monkeypatch, tmp_path, outcome):
    original = FeatureSPA.transform

    def projection(self, rows):
        # Three-fold training has 40 rows; only the 20 held-out rows trigger this.
        if len(rows) == 20:
            if outcome == "warning":
                warnings.warn("held-out projection warning", ConvergenceWarning)
            elif outcome == "error":
                raise ValueError("held-out projection failure")
        return original(self, rows)

    monkeypatch.setattr(FeatureSPA, "transform", projection)
    model = NMFClassifier(screening="none", n_components=2, random_state=11, max_iter=500)
    report = calibration_knn_grid(model, {"n_neighbors": [1, 3]}, *data,
                                  cv=3, threads=1, checkpoint_dir=tmp_path)
    for fold in report["representation_folds"]:
        diagnostic = fold["diagnostics"]
        assert diagnostic["last_transform_performed"] is True
        assert diagnostic["converged"] is True
        assert diagnostic["last_transform_converged"] is (outcome == "success")
        assert fold["status"] == {"success": "valid", "warning": "nonconverged", "error": "failed"}[outcome]
        if outcome == "warning":
            assert diagnostic["last_transform_warnings"] == ["held-out projection warning"]
        if outcome == "error":
            assert diagnostic["last_transform_error"] == "ValueError: held-out projection failure"
