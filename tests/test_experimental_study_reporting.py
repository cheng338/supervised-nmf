"""Experimental query receipts reach real study gates without stale model state."""
from copy import deepcopy
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pytest
from scipy.optimize import nnls
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental
import fashion_mnist_study as runner
from supervised_nmf.evaluation import evaluate_classifier
from supervised_nmf.screening import BinaryScreening
from supervised_nmf.study import _read_receipt, calibration_knn_grid


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def problem():
    X = np.tile(np.array([[1., 3.], [2., 6.], [4., 1.], [8., 2.]]), (4, 1))
    y = np.tile([0, 0, 1, 1], 4)
    return X, y, X[:4].copy(), y[:4].copy()


def template():
    return Pipeline([
        ("screen", BinaryScreening(screening="absolute", score_threshold=0)),
        ("nmf", experimental.ANLSBPP(n_components=2, random_state=0)),
        ("knn", KNeighborsClassifier(n_neighbors=1, n_jobs=1)),
    ])


def controlled_fit(monkeypatch, *, mode="success"):
    """Fix the training fixture; retain real transform/NNLS and study machinery."""
    fitted = []

    def fit_transform(self, X, y=None, **kwargs):
        self.components_ = np.array([[1., 0.], [1., 1.]])
        self.n_features_in_, self.n_components_ = X.shape[1], 2
        self.training_coordinates_ = np.array([nnls(self.components_.T, row)[0] for row in X])
        self.converged_, self.n_iter_ = mode != "held", 1
        self.convergence_reason_ = "controlled_training_fixture"
        self.reconstruction_err_ = float(np.linalg.norm(X - self.training_coordinates_ @ self.components_))
        self.nnls_diagnostics_ = {"training_fixture": True}
        self.computation_dtype_ = "float64"
        # Simulate legacy serialized state. Revised consumers must ignore it.
        self.last_transform_converged_ = False
        self.last_transform_diagnostics_ = {"success": True, "legacy": True}
        if mode == "cap":
            self.nnls_max_iter = 1
        fitted.append(self)
        return self.training_coordinates_.copy()

    monkeypatch.setattr(experimental.ANLSBPP, "fit_transform", fit_transform)
    if mode == "solver_error":
        def fail(*args, **kwargs):
            raise RuntimeError("controlled query failure")
        monkeypatch.setattr(experimental, "nnls_block_pivot", fail)
    return fitted


def current_receipt(*, performed=True, converged=True, error=None):
    return dict(performed=performed, converged=converged,
                diagnostics={"success": converged} if performed and error is None else None,
                error=error)


@pytest.mark.parametrize("receipt,qualified", [
    (None, False),
    (current_receipt(performed=False, converged=None), False),
    (current_receipt(converged=False), False),
    (current_receipt(converged=False, error="ValueError: current failure"), False),
    (current_receipt(), True),
])
def test_query_qualification_uses_current_receipt_and_requires_performed(
        monkeypatch, problem, receipt, qualified):
    controlled_fit(monkeypatch)
    X, y, _, _ = problem
    model = template().fit(X, y)
    assert runner._model_nonconverged(model, []) is False  # fit-only
    assert runner._model_nonconverged(model, [], receipt, query=True) is (not qualified)
    report = runner._model_diagnostics(model, receipt)["decomposition"]
    if receipt is None:
        assert not any(key.startswith("last_transform_") for key in report)
    else:
        assert report["last_transform_performed_"] == receipt["performed"]
        assert report["last_transform_converged_"] == receipt["converged"]
        assert report["last_transform_diagnostics_"] == receipt["diagnostics"]
        assert report["last_transform_error_"] == receipt["error"]
        assert "legacy" not in str(report["last_transform_diagnostics_"])


def test_old_pickle_attributes_remain_inert_and_reports_are_detached(monkeypatch, problem):
    controlled_fit(monkeypatch)
    X, y, query, _ = problem
    model = pickle.loads(pickle.dumps(template().fit(X, y)))
    decomposition = model.named_steps["nmf"]
    legacy = deepcopy({key: value for key, value in vars(decomposition).items()
                       if key.startswith("last_transform_")})
    with experimental.capture_transform_diagnostics(decomposition) as projection:
        decomposition.transform(query)
    report = runner._model_diagnostics(model, projection)
    saved = deepcopy(report)
    projection["diagnostics"]["success"] = False
    assert report == saved
    assert runner._model_nonconverged(model, []) is False
    assert {key: value for key, value in vars(decomposition).items()
            if key.startswith("last_transform_")} == legacy


@pytest.mark.parametrize("mode", ["success", "cap", "solver_error", "held", "missing_projection"])
def test_profile_captures_fitted_clone_and_retains_real_query_outcome(monkeypatch, problem, mode):
    X, y, query, query_y = problem
    controlled_fit(monkeypatch)
    reference = evaluate_classifier(template().fit(X, y), query, query_y, sample_ids=list(range(4)))
    fitted = controlled_fit(monkeypatch, mode=mode)
    if mode == "missing_projection":
        monkeypatch.setattr(runner, "evaluate_classifier", lambda *a, **k: deepcopy(reference))
    profile = runner._profile(template(), X, y, query, query_y,
                              reference=reference, sample_ids=list(range(4)))
    assert len(fitted) == 1
    decomposition = fitted[0]
    assert decomposition.last_transform_diagnostics_ == {"success": True, "legacy": True}
    detail = profile["diagnostics"]["decomposition"]
    if mode == "held":
        assert profile["status"] == "nonconverged" and profile["evaluation"] is None
        assert not any(key.startswith("last_transform_") for key in detail)
    else:
        assert detail["last_transform_performed_"] is (mode != "missing_projection")
        assert detail["last_transform_converged_"] is (True if mode == "success" else None if mode == "missing_projection" else False)
        if mode == "success":
            assert profile["status"] == "valid" and profile["exact_prediction_match"]
            assert profile["evaluation"]["predictions"] == reference["predictions"]
            assert profile["stage_calls"]["decomposition_projection_seconds"] == 1
            assert profile["stage_calls"]["decomposition_fit_seconds"] == 1
            assert detail["last_transform_diagnostics_"]["success"] is True
        elif mode == "solver_error":
            assert profile["evaluation"]["status"] == "failed"
            assert detail["last_transform_error_"] == "RuntimeError: controlled query failure"
            assert detail["last_transform_diagnostics_"] is None
        elif mode == "cap":
            assert profile["evaluation"]["status"] == "provisional_nonconvergence"
            assert detail["last_transform_diagnostics_"]["success"] is False
            assert profile["evaluation"]["warnings"]
        if mode != "success":
            assert profile["status"] != "valid"


def configure_evaluation(monkeypatch, problem):
    """Supply an already-frozen tiny CV selection to the actual evaluate path."""
    X, y, query, query_y = problem
    job = dict(id="tiny-anls", pair={"id": "pair"}, method="screened_anls_bpp_absolute",
               seed=0, n_train=len(y), array_key="train")
    selected = dict(selected_parameters={}, cv_balanced_accuracy=1., tuning_compute_seconds=0.,
                    selected_candidate={"folds": [{"warnings": []}]})
    configuration = dict(evaluation_order=[job["id"]],
                         arrays={"train": {"kind": "train", "data_sha256": "train"},
                                 "pair/validation": {"kind": "validation", "data_sha256": "validation"}},
                         plans=[{"job": job}], sources={},
                         splits={"pair": {"validation_indices": list(range(len(query_y)))}},
                         protocol={"profile_repeats": 1})
    monkeypatch.setattr(runner, "aggregate_cv", lambda *a, **k: {"jobs": {job["id"]: selected}})
    monkeypatch.setattr(runner, "_verify_arrays", lambda record: None)
    monkeypatch.setattr(runner, "_check_source", lambda source: None)
    monkeypatch.setattr(runner, "_load_arrays", lambda record: (X, y) if record["kind"] == "train" else (query, query_y))
    monkeypatch.setattr(runner, "_estimator", lambda job: template())
    return configuration


@pytest.mark.parametrize("mode", ["success", "cap", "solver_error", "held", "missing_projection", "outer_error"])
def test_selected_evaluation_writes_current_receipt_and_resumes_without_new_query(
        monkeypatch, tmp_path, problem, mode):
    configuration = configure_evaluation(monkeypatch, problem)
    X, y, query, query_y = problem
    controlled_fit(monkeypatch)
    reference = evaluate_classifier(template().fit(X, y), query, query_y, sample_ids=list(range(4)))
    fitted = controlled_fit(monkeypatch, mode=mode)
    if mode == "missing_projection":
        monkeypatch.setattr(runner, "evaluate_classifier", lambda *a, **k: deepcopy(reference))
    elif mode == "outer_error":
        def fail_after_transform(model, X, y, **kwargs):
            model.named_steps["nmf"].transform(X)
            raise RuntimeError("controlled evaluator failure after projection")
        monkeypatch.setattr(runner, "evaluate_classifier", fail_after_transform)
    result = runner.evaluate(tmp_path, configuration)
    report = _read_receipt(tmp_path / "tiny-anls" / "evaluation.json")
    detail = report["diagnostics"]["decomposition"]
    if mode == "held":
        assert report["evaluation"]["status"] == "held"
        assert not any(key.startswith("last_transform_") for key in detail)
    else:
        assert detail["last_transform_performed_"] is (mode != "missing_projection")
        if mode == "success":
            assert result["valid_evaluations"] == 1 and report["profiling_valid"]
            assert report["evaluation"]["predictions"] == reference["predictions"]
            assert report["profiling"][0]["diagnostics"]["decomposition"]["last_transform_diagnostics_"]["success"]
        elif mode == "solver_error":
            assert report["evaluation"]["status"] == "failed"
            assert detail["last_transform_error_"] == "RuntimeError: controlled query failure"
            assert detail["last_transform_diagnostics_"] is None
        elif mode in ("cap", "missing_projection"):
            assert report["evaluation"]["status"] == "provisional_nonconvergence"
            assert detail["last_transform_converged_"] is (None if mode == "missing_projection" else False)
        else:
            assert report["evaluation"]["status"] == "failed"
            assert detail["last_transform_converged_"] is True
            assert detail["last_transform_error_"] is None
    count = len(fitted)
    before = (tmp_path / "tiny-anls" / "evaluation.json").read_bytes()
    assert runner.evaluate(tmp_path, configuration) == result
    assert len(fitted) == count
    assert (tmp_path / "tiny-anls" / "evaluation.json").read_bytes() == before


@pytest.mark.parametrize("mode", ["cap", "solver_error"])
def test_shared_cv_still_rejects_query_warning_and_exception(monkeypatch, tmp_path, problem, mode):
    controlled_fit(monkeypatch, mode=mode)
    X, y, _, _ = problem
    folds = [(np.arange(8, 16), np.arange(8)), (np.arange(8), np.arange(8, 16))]
    report = calibration_knn_grid(template(), {"knn__n_neighbors": [1]}, X, y,
                                  folds=folds, cv=2, checkpoint_dir=tmp_path, threads=1)
    candidate = report["candidates"][0]
    assert candidate["valid"] is False
    for fold in candidate["folds"]:
        if mode == "cap":
            assert fold["status"] == "nonconverged" and fold["warnings"]
        else:
            assert fold["status"] == "failed"
            assert "controlled query failure" in fold["error"]


def test_natural_anls_profile_preserves_predictions_and_one_projection():
    # This established small study fixture executes the original training solver.
    random = np.random.RandomState(71)
    y = np.repeat(np.array([0, 1], dtype=np.uint8), 60)
    X = random.uniform(.1, .3, size=(120, 16)).astype(np.float32)
    X[:, :5] += y[:, None] * .5
    X[:, -1] = 0
    job = dict(method="screened_anls_bpp_absolute", seed=43, max_iter=20000,
               tol=1e-4, anls_max_iter=1000, anls_nnls_max_iter=None, anls_nnls_tol=1e-10)
    model = runner._estimator(job).set_params(nmf__n_components=2, screen__score_threshold=0,
                                             knn__n_neighbors=3)
    model.fit(X, y)
    before = pickle.dumps(model.named_steps["nmf"])
    reference = evaluate_classifier(model, X[:12], y[:12], sample_ids=list(range(12)))
    assert pickle.dumps(model.named_steps["nmf"]) == before
    report = runner._profile(model, X, y, X[:12], y[:12], reference=reference,
                             sample_ids=list(range(12)))
    assert report["status"] == "valid" and report["exact_prediction_match"]
    assert report["stage_calls"]["decomposition_projection_seconds"] == 1
    detail = report["diagnostics"]["decomposition"]
    assert detail["last_transform_performed_"] and detail["last_transform_converged_"]
    assert detail["last_transform_diagnostics_"]["success"] is True
    assert detail["last_transform_error_"] is None
