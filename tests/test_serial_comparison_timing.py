"""Experimental final fits and predictions must run serially after calibration."""

from contextlib import contextmanager
from datetime import datetime
from fractions import Fraction
import json
from numbers import Integral
import os

from joblib import effective_n_jobs, parallel_config
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info

from supervised_nmf import evaluation
from supervised_nmf.evaluation import run_comparison


class ObservedClassifier(ClassifierMixin, BaseEstimator):
    observations = []

    def __init__(self, *, n_jobs=None, projection_workers=1, projection_native_threads=1,
                 reject_final=False):
        self.n_jobs = n_jobs
        self.projection_workers = projection_workers
        self.projection_native_threads = projection_native_threads
        self.reject_final = reject_final

    def _observe(self, phase, X):
        self.observations.append((phase, len(X), self.n_jobs, self.projection_workers,
                                  self.projection_native_threads, effective_n_jobs(None),
                                  {pool["num_threads"] for pool in threadpool_info()}))

    def fit(self, X, y):
        self._observe("fit", X)
        if self.reject_final and len(X) == 40:
            raise ValueError("failed whole-data fit")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        self._observe("predict", X)
        return (X[:, 0] > .5).astype(int)


class CallbackClassifier(ObservedClassifier):
    def __init__(self, name, callback):
        super().__init__()
        self.name = name
        self.callback = callback

    def fit(self, X, y):
        self.callback(self.name, len(X))
        return super().fit(X, y)


class InPlaceLogistic(LogisticRegression):
    def predict(self, X):
        predicted = super().predict(X)
        X.fill(0)
        return predicted


class InPlaceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, reject_final=False):
        self.reject_final = reject_final

    def fit(self, X, y):
        if not np.array_equal(X[:, 0], 2.0 + 4.0 * y):
            raise ValueError("a previous method changed the input")
        self.classes_ = np.unique(y)
        X.fill(-10)
        y.fill(0)
        if self.reject_final and len(X) == 40:
            raise ValueError("reject the first full fit")
        return self

    def predict(self, X):
        labels = (X[:, 0] > 4).astype(int)
        X.fill(-10)
        return labels

    def predict_proba(self, X):
        positive = (X[:, 0] > 4).astype(float)
        X.fill(8)
        return np.column_stack([1 - positive, positive])


def compare(path, *, serial_final, reject=(True, False), name="model"):
    X = np.arange(53)[:, None] % 2
    y = X[:, 0]
    estimator = Pipeline([("head", ObservedClassifier(projection_workers=3))])
    grid = {"head__n_jobs": [2], "head__projection_native_threads": [2],
            "head__reject_final": list(reject)}
    return run_comparison({name: (estimator, grid)}, X[:40], y[:40], X[40:], y[40:],
                          seed=9, threads=2, output_dir=path, serial_final=serial_final)["models"][name]


def check_evidence(receipt):
    assert receipt["verified"] is True
    assert receipt["pid"] == os.getpid()
    assert receipt["host"]
    assert receipt["application_workers"] == receipt["native_threads"] == 1
    assert {pool["num_threads"] for pool in receipt["threadpools"]} == {1}
    assert datetime.fromisoformat(receipt["started_utc"]) <= datetime.fromisoformat(receipt["finished_utc"])
    assert receipt["seconds"] >= 0


def test_explicit_parallel_cv_then_serial_retries_prediction_and_resume(tmp_path):
    ObservedClassifier.observations.clear()
    with parallel_config(backend="threading", n_jobs=2):
        result = compare(tmp_path, serial_final=True)
        assert effective_n_jobs(None) == 2
    cv_rows = [row for row in ObservedClassifier.observations if row[1] in (32, 8)]
    assert len(cv_rows) == 20
    assert all(row[2:6] == (2, 3, 2, 2) for row in cv_rows)
    assert all(row[6] and max(row[6]) <= 2 for row in cv_rows)
    final_rows = [row for row in ObservedClassifier.observations if row[1] in (40, 13)]
    assert [row[:2] for row in final_rows] == [("fit", 40), ("fit", 40), ("predict", 13)]
    assert all(row[2:] == (1, 1, 1, 1, {1}) for row in final_rows)
    assert result["selected_parameters"]["head__n_jobs"] == 2
    assert result["evaluation"]["metrics"]["accuracy"] == 1

    cv = json.loads((tmp_path / "model-cv.json").read_text())
    intervals = result["cv_timing"]
    assert intervals == cv["cv_timing"] and len(intervals) == 1
    interval, = intervals
    assert interval["host"] and interval["pid"] == os.getpid()
    assert interval["native_threads"] == 2
    assert interval["joblib"] == cv["identity"]["cv_joblib"]
    assert interval["joblib"]["effective_n_jobs"] == 2
    assert interval["joblib"]["backend"].endswith(".ThreadingBackend")
    assert interval["started_utc"] <= interval["finished_utc"]
    assert interval["seconds"] == result["tuning_seconds"]
    attempts = cv["final_fit_attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "valid"]
    for attempt in attempts:
        check_evidence(attempt["fit_timing"])
        assert attempt["seconds"] == attempt["fit_timing"]["seconds"]
        assert attempt["fit_timing"]["worker_parameter_overrides"] == {
            "head__n_jobs": 1, "head__projection_workers": 1, "head__projection_native_threads": 1}
    check_evidence(result["prediction_timing"])
    assert attempts[-1]["fit_timing"]["finished_utc"] <= result["prediction_timing"]["started_utc"]
    assert interval["finished_utc"] <= attempts[0]["fit_timing"]["started_utc"]
    assert result["final_fit_seconds"] == attempts[-1]["seconds"]
    assert result["cumulative_final_fit_seconds"] == sum(attempt["seconds"] for attempt in attempts)
    assert result["evaluation"]["prediction_seconds"] == result["prediction_timing"]["seconds"]
    ObservedClassifier.observations.clear()
    with pytest.raises(ValueError, match="comparison folds are incompatible"):
        compare(tmp_path, serial_final=True)
    with parallel_config(backend="threading", n_jobs=2):
        assert compare(tmp_path, serial_final=True) == result
    assert not ObservedClassifier.observations
    with pytest.raises(ValueError, match="comparison folds are incompatible"):
        compare(tmp_path, serial_final=False, name="other")
    assert not ObservedClassifier.observations


def test_serial_final_clone_overrides_registered_integral_worker_counts():
    @Integral.register
    class Workers(Fraction):
        pass

    value = Workers(2)
    model = Pipeline([("head", ObservedClassifier(n_jobs=value, projection_workers=value,
                                                  projection_native_threads=value))])
    X = np.array([[0], [1]])
    attempt = {}
    fitted = evaluation._serial_comparison_fit(clone(model), X, X[:, 0], attempt)
    for name in ("head__n_jobs", "head__projection_workers", "head__projection_native_threads"):
        assert model.get_params()[name] == 2
        assert fitted.get_params()[name] == attempt["fit_timing"]["worker_parameter_overrides"][name] == 1


def test_serial_majority_fallback_and_unrestricted_default(tmp_path):
    result = compare(tmp_path / "serial", serial_final=True, reject=(True,))
    assert result["is_fallback"]
    attempts = json.loads((tmp_path / "serial" / "model-cv.json").read_text())["final_fit_attempts"]
    assert [a["status"] for a in attempts] == ["failed", "fallback"]
    check_evidence(attempts[-1]["fit_timing"])
    check_evidence(result["prediction_timing"])
    ObservedClassifier.observations.clear()
    ordinary = compare(tmp_path / "ordinary", serial_final=False, reject=(False,))
    assert "prediction_timing" not in ordinary
    assert all(row[2:5] == (2, 3, 2) for row in ObservedClassifier.observations)
    with pytest.raises(ValueError, match="comparison folds are incompatible"):
        compare(tmp_path / "ordinary", serial_final=True)


@pytest.mark.parametrize("failed_phase", ["fit", "prediction"])
def test_unconfirmed_resource_limit_stops_the_experiment(monkeypatch, tmp_path, failed_phase):
    # A scientific model failure may select the next best candidate. A failure
    # of the timing protocol must abort without publishing a substitute result.
    effective = iter([2] if failed_phase == "fit" else [1, 2])
    original = evaluation._serial_timing_phase
    @contextmanager
    def break_limit(receipt):
        with monkeypatch.context() as patcher:
            patcher.setattr(evaluation, "effective_n_jobs", lambda jobs: next(effective))
            with original(receipt) as finish:
                yield finish

    monkeypatch.setattr(evaluation, "_serial_timing_phase", break_limit)
    with pytest.raises(RuntimeError, match="Joblib did not accept"):
        compare(tmp_path, serial_final=True, reject=(False,))
    assert not (tmp_path / "model-evaluation.json").exists()
    attempts = json.loads((tmp_path / "model-cv.json").read_text())["final_fit_attempts"]
    assert [a["status"] for a in attempts] == ([] if failed_phase == "fit" else ["valid"])


def test_missing_and_late_native_pools_cannot_certify_a_timing(monkeypatch):
    observed = iter([[], [{"num_threads": 1}], [{"num_threads": 1}, {"num_threads": 4}]])
    monkeypatch.setattr(evaluation, "threadpool_info", lambda: next(observed))
    with pytest.raises(RuntimeError, match="No native thread pool"):
        with evaluation._serial_timing_phase({}):
            pytest.fail("The unverified timing should never start")
    with pytest.raises(RuntimeError, match="final native thread limits"):
        with evaluation._serial_timing_phase({}):
            pass


def test_failed_fit_with_unverified_exit_inventory_aborts_and_preserves_cost(monkeypatch, tmp_path):
    real_info = evaluation.threadpool_info
    interrupt_run = True
    failed = False
    fit = ObservedClassifier.fit

    def corrupt_final_fit(self, X, y):
        nonlocal failed
        if interrupt_run and len(X) == 40:
            failed = True
            raise ValueError("fit registered a new multi-threaded library")
        return fit(self, X, y)

    def inventory():
        return [*real_info(), {"num_threads": 8}] if failed else real_info()

    monkeypatch.setattr(ObservedClassifier, "fit", corrupt_final_fit)
    monkeypatch.setattr(evaluation, "threadpool_info", inventory)
    with pytest.raises(RuntimeError, match="final native thread limits") as failure:
        compare(tmp_path, serial_final=True, reject=(False,))
    assert isinstance(failure.value.__context__, ValueError)
    assert not (tmp_path / "model-evaluation.json").exists()
    journal = json.loads((tmp_path / "model-cv.json").read_text())
    attempt, = journal["final_fit_attempts"]
    receipt = attempt["fit_timing"]
    assert attempt["status"] == "timing_verification_failed" and receipt["verified"] is False
    assert receipt["final_threadpools"][-1]["num_threads"] == 8
    assert 0 < receipt["seconds"] == attempt["seconds"] == journal["cumulative_final_fit_seconds"]
    interrupt_run = failed = False
    result = compare(tmp_path, serial_final=True, reject=(False,))
    assert result["cumulative_final_fit_seconds_complete"] and result["fit_timing"]["verified"]
    assert result["cumulative_final_fit_seconds"] == attempt["seconds"] + result["final_fit_seconds"]
    assert result["first_final_fit_started_utc"] == receipt["started_utc"]


def test_interrupt_during_final_fit_setup_can_resume(monkeypatch, tmp_path):
    original_phase = evaluation._serial_timing_phase

    def interrupted_inventory():
        raise KeyboardInterrupt("interrupted before the measured fit")

    @contextmanager
    def stop_during_setup(receipt):
        with monkeypatch.context() as patcher:
            patcher.setattr(evaluation, "threadpool_info", interrupted_inventory)
            with original_phase(receipt) as finish:
                yield finish

    with monkeypatch.context() as patcher:
        patcher.setattr(evaluation, "_serial_timing_phase", stop_during_setup)
        with pytest.raises(KeyboardInterrupt, match="before the measured fit"):
            compare(tmp_path, serial_final=True, reject=(False,))
    journal = json.loads((tmp_path / "model-cv.json").read_text())
    attempt, = journal["final_fit_attempts"]
    assert attempt["status"] == "interrupted_before_fit" and attempt["seconds"] is None
    result = compare(tmp_path, serial_final=True, reject=(False,))
    assert result["cumulative_final_fit_seconds_complete"]
    assert result["cumulative_final_fit_seconds"] == result["final_fit_seconds"]
    assert result["first_final_fit_started_utc"] == result["fit_timing"]["started_utc"]


def test_resume_preserves_prior_fit_and_cv_costs(tmp_path):
    first = compare(tmp_path, serial_final=True)
    original = json.loads((tmp_path / "model-cv.json").read_text())
    # An interruption after publishing a full-data fit requires a fresh fitted
    # object for prediction. The resumed comparison retains both sets of costs.
    (tmp_path / "model-evaluation.json").unlink()
    second = compare(tmp_path, serial_final=True)
    after = json.loads((tmp_path / "model-cv.json").read_text())
    assert second["cv_timing"] == first["cv_timing"]
    assert second["tuning_seconds"] == first["tuning_seconds"]
    assert after["final_fit_attempts"][:2] == original["final_fit_attempts"]
    assert len(after["final_fit_attempts"]) == 4
    assert second["final_fit_seconds"] == after["final_fit_attempts"][-1]["seconds"]
    assert second["cumulative_final_fit_seconds"] == sum(a["seconds"] for a in after["final_fit_attempts"])
    assert second["cv_checkpoint_sha256"] == after["checkpoint_sha256"]
    # Restoring an intact earlier evaluation would discard the costs of this refit.
    evaluation.write_json(tmp_path / "model-evaluation.json", first)
    with pytest.raises(ValueError, match="evaluation failed its checkpoint or integrity check"):
        compare(tmp_path, serial_final=True)


def test_interrupted_cv_accumulates_only_observed_intervals(monkeypatch, tmp_path):
    write = evaluation.write_json

    class StopAfterFold(BaseException):
        pass

    def interrupt(path, report):
        write(path, report)
        if path.name == "model-cv.json" and len(report["candidates"][0]["folds"]) == 2:
            raise StopAfterFold

    with monkeypatch.context() as patcher:
        patcher.setattr(evaluation, "write_json", interrupt)
        with pytest.raises(StopAfterFold):
            compare(tmp_path, serial_final=True, reject=(False,))
    previous = json.loads((tmp_path / "model-cv.json").read_text())
    result = compare(tmp_path, serial_final=True, reject=(False,))
    first, second = result["cv_timing"]
    assert [first] == previous["cv_timing"]
    assert first["finished_utc"] <= second["started_utc"] <= second["finished_utc"]
    assert result["tuning_seconds"] == first["seconds"] + second["seconds"]
    assert second["finished_utc"] <= result["fit_timing"]["started_utc"]


def test_all_methods_and_replications_finish_cv_before_final(tmp_path):
    X = np.arange(53)[:, None] % 2
    y = X[:, 0]
    models = {name: (ObservedClassifier(), {}) for name in ("a", "b")}
    results = []
    directories = [tmp_path / str(seed) for seed in (3, 4)]
    with evaluation._serial_comparison_batch(root=tmp_path, directories=directories) as batch:
        for seed in (3, 4):
            results.append(run_comparison(models, X[:40], y[:40], X[40:], y[40:], seed=seed,
                                          output_dir=tmp_path / str(seed), serial_final=True, _batch=batch))
            assert not list(tmp_path.rglob("*-evaluation.json"))
    cv_ends, fit_starts = [], []
    for seed, result in zip((3, 4), results):
        for name, record in result["models"].items():
            cv_ends.append(record["cv_timing"][-1]["finished_utc"])
            report = json.loads((tmp_path / str(seed) / f"{name}-cv.json").read_text())
            fit_starts.append(report["final_fit_attempts"][0]["fit_timing"]["started_utc"])
    assert max(cv_ends) <= min(fit_starts)
    # Losing a checkpoint must not combine new CV with an earlier final.
    for suffix in ("-cv.json", "-evaluation.json"):
        (tmp_path / "3" / ("b" + suffix)).unlink()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    ObservedClassifier.observations.clear()
    with pytest.raises(ValueError, match="Missing or changed CV"):
        with evaluation._serial_comparison_batch(root=tmp_path, directories=directories):
            pytest.fail("The damaged batch was admitted")
    assert not ObservedClassifier.observations
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not (tmp_path / "3/b-evaluation.json").exists()

    # A surviving later replication closes the whole batch even when the seal
    # and every record from the first replication disappear.
    (tmp_path / "comparison-cv-closed.json").unlink()
    for path in directories[0].iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="lack their closed CV"):
        with evaluation._serial_comparison_batch(root=tmp_path, directories=directories):
            pytest.fail("Later replication evidence should prevent earlier CV")
    assert not list(directories[0].iterdir())


def test_completed_single_model_cannot_replace_missing_owned_files(tmp_path, monkeypatch):
    compare(tmp_path, serial_final=True, reject=(False,))
    (tmp_path / "model-cv.json").unlink()
    (tmp_path / "model-evaluation.json").unlink()
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(evaluation, "write_json", lambda *a: pytest.fail("Closed preflight wrote a file"))
    ObservedClassifier.observations.clear()
    with pytest.raises(ValueError, match="Missing or changed CV"):
        compare(tmp_path, serial_final=True, reject=(False,))
    assert not ObservedClassifier.observations
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_serial_batch_verifies_later_evaluation_before_reusing_any_results(tmp_path, monkeypatch):
    X = np.arange(53)[:, None] % 2
    y = X[:, 0]
    models = {"model": (ObservedClassifier(), {})}
    directories = [tmp_path / label for label in ("first", "last")]
    with evaluation._serial_comparison_batch(root=tmp_path, directories=directories) as batch:
        for directory in directories:
            run_comparison(models, X[:40], y[:40], X[40:], y[40:], seed=9,
                           output_dir=directory, serial_final=True, _batch=batch)
    first_path, last_path = [d / "model-evaluation.json" for d in directories]
    first, last = [json.loads(path.read_bytes()) for path in (first_path, last_path)]
    assert first["comparison_fingerprint"] == last["comparison_fingerprint"]
    assert first["fold_plan_sha256"] == last["fold_plan_sha256"]
    assert first["cv_checkpoint_sha256"] != last["cv_checkpoint_sha256"]
    del last["prediction_timing"]
    # Try both a complete record from another fit and a damaged local record.
    ObservedClassifier.observations.clear()
    monkeypatch.setattr(evaluation, "write_json", lambda *a: pytest.fail("Damaged batch wrote a record"))
    for record in (first, last):
        last_path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="evaluation failed its checkpoint or integrity check"):
            with evaluation._serial_comparison_batch(root=tmp_path, directories=directories):
                pytest.fail("Damaged batch was admitted")
    assert not ObservedClassifier.observations


@pytest.mark.parametrize("phase", ["cv", "fit"])
@pytest.mark.parametrize("hard_exit", [False, True])
def test_timing_recovery_after_interrupted_call(monkeypatch, tmp_path, phase, hard_exit):
    original = ObservedClassifier.fit
    image = None
    calls = 0
    cv_path = tmp_path / "model-cv.json"

    def stop(self, X, y):
        nonlocal calls, image
        if len(X) == (32 if phase == "cv" else 40):
            calls += 1
            if calls == (2 if phase == "cv" else 1):
                # This is the exact durable image a killed interpreter leaves
                # while its numerical call is still executing.
                image = cv_path.read_bytes()
                raise KeyboardInterrupt("interrupt the numerical call")
        return original(self, X, y)

    monkeypatch.setattr(ObservedClassifier, "fit", stop)
    with pytest.raises(KeyboardInterrupt):
        compare(tmp_path, serial_final=True, reject=(False,))
    caught = json.loads(cv_path.read_bytes())
    if phase == "cv":
        interval, = caught["cv_timing"]
        assert interval["state"] == "interrupted" and interval["seconds"] > 0
        assert interval["seconds"] == caught["tuning_seconds"]
    else:
        attempt, = caught["final_fit_attempts"]
        assert attempt["status"] == "interrupted" and attempt["fit_timing"]["verified"]
        assert caught["cumulative_final_fit_seconds"] == attempt["seconds"]
    if hard_exit:
        cv_path.write_bytes(image)
    result = compare(tmp_path, serial_final=True, reject=(False,))
    complete = "tuning_seconds_complete" if phase == "cv" else "cumulative_final_fit_seconds_complete"
    assert result[complete] is (not hard_exit)
    if phase == "cv":
        prior, current = result["cv_timing"]
        assert prior["seconds"] > 0
        assert result["tuning_seconds"] == prior["seconds"] + current["seconds"]
    else:
        prior, current = json.loads(cv_path.read_bytes())["final_fit_attempts"]
        assert result["cumulative_final_fit_seconds"] == (prior["seconds"] or 0) + current["seconds"]
        expected_start = None if hard_exit else prior["fit_timing"]["started_utc"]
        assert result["first_final_fit_started_utc"] == expected_start


def test_live_callbacks_cross_the_cv_barrier_without_replay(tmp_path):
    events = []

    def record(name, rows):
        events.append((name, rows))

    X = np.arange(53)[:, None] % 2
    y = X[:, 0]
    models = {name: (CallbackClassifier(name, record), {}) for name in ("a", "b")}
    summary = run_comparison(models, X[:40], y[:40], X[40:], y[40:],
                             seed=11, output_dir=tmp_path, serial_final=True)
    assert events == [("a", 32)] * 5 + [("b", 32)] * 5 + [("a", 40), ("b", 40)]
    assert all(model["calibration_resume_eligible"] is False for model in summary["models"].values())


def test_serial_methods_retries_and_score_calls_receive_private_inputs(tmp_path):
    y = np.arange(54) % 2
    X = np.asfortranarray(np.column_stack([2.0 + 4.0 * y, np.arange(len(y)) % 3]))
    original_X, original_y = X.copy(), y.copy()
    digest = evaluation.data_fingerprint(X[40:], y[40:])
    models = {"scale": (Pipeline([("scale", StandardScaler(copy=False)), ("head", InPlaceLogistic(random_state=7))]), {}),
              "retry": (InPlaceClassifier(), {"reject_final": [True, False]})}
    summary = run_comparison(models, X[:40], y[:40], X[40:], y[40:],
                             seed=10, output_dir=tmp_path, serial_final=True)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    for result in summary["models"].values():
        assert not result["is_fallback"]
        report = result["evaluation"]
        assert report["status"] == "valid" and report["data_sha256"] == digest
        assert report["metrics"]["accuracy"] == report["metrics"]["roc_auc"] == 1.0
    attempts = json.loads((tmp_path / "retry-cv.json").read_text())["final_fit_attempts"]
    assert [a["status"] for a in attempts] == ["failed", "valid"]
    resumed = run_comparison(models, X[:40], y[:40], X[40:], y[40:],
                             seed=10, output_dir=tmp_path, serial_final=True)
    assert resumed == summary


def test_wrong_length_after_prediction_preserves_verified_seconds(monkeypatch, tmp_path):
    predict = ObservedClassifier.predict

    def short_prediction(self, X):
        values = predict(self, X)
        return values[:-1] if len(values) == 13 else values

    monkeypatch.setattr(ObservedClassifier, "predict", short_prediction)
    result = compare(tmp_path, serial_final=True, reject=(False,))
    assert result["evaluation"]["status"] == "failed"
    check_evidence(result["prediction_timing"])
    assert result["evaluation"]["prediction_seconds"] == result["prediction_timing"]["seconds"]


def test_failed_prediction_retains_its_single_measured_interval(monkeypatch, tmp_path):
    predict = ObservedClassifier.predict

    def fail_on_held_out(self, X):
        if len(X) == 13:
            raise ValueError("held-out prediction failed")
        return predict(self, X)

    monkeypatch.setattr(ObservedClassifier, "predict", fail_on_held_out)
    result = compare(tmp_path, serial_final=True, reject=(False,))
    assert result["evaluation"]["status"] == "failed"
    assert result["evaluation"]["error"] == "held-out prediction failed"
    check_evidence(result["prediction_timing"])
    assert result["evaluation"]["prediction_seconds"] == result["prediction_timing"]["seconds"]
