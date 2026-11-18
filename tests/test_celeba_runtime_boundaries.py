"""Every final-fit identity guard remains fresh across initialization reuse."""
from contextlib import contextmanager
import sys
from unittest.mock import patch
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from tests._celeba_recovery_test_support import guard_support, guard_cv

@pytest.fixture(scope='module')
def boundary_support(guard_support):
    return guard_support

@pytest.fixture
def boundary_case(boundary_support, guard_cv, tmp_path_factory, monkeypatch):
    case = boundary_support.copy_case(guard_cv, tmp_path_factory, monkeypatch)
    case.config['final_fit_recovery'] = 'all'
    return case


@contextmanager
def identity_witness(support):
    recovery = support.recovery
    original_helper, original_hook = recovery.active_runtime_identity, recovery._hook
    helper_code = original_helper.__code__
    observation_code = recovery._runtime_identity_observation.__code__
    previous = sys.getprofile()
    assert previous is None, "Use a separate qualification process without another profiler"
    counts = {"full": 0, "opted": 0, "observation": 0, "caches": []}

    def profile(frame, event, arg):
        if event != "call":
            return
        if frame.f_code is helper_code:
            cache = frame.f_locals.get("_initialization_cache")
            if cache is None:
                counts["full"] += 1
            else:
                assert type(cache) is dict
                counts["opted"] += 1
                counts["caches"].append(cache)
        elif frame.f_code is observation_code:
            counts["observation"] += 1

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)
        assert recovery.active_runtime_identity is original_helper
        assert recovery._hook is original_hook


def assert_counts(counts, *, opted, full, observation):
    assert {name: counts[name] for name in ("opted", "full", "observation")} == {
        "opted": opted, "full": full, "observation": observation}
    assert len(counts["caches"]) == opted
    assert all(cache is counts["caches"][0] for cache in counts["caches"])


def seed_fitted_outcome(support, case):
    result = support.aggregate(case)
    assert result["evaluation"]["status"] == "valid"
    assert result["recovery"]["fit_performed"]
    directory, outcome = support.committed(case)
    assert outcome["kind"] == "fitted"
    # Removing only this test-owned final JSON exercises the genuine committed
    # fitted outcome; its manifest, payload and hashes remain untouched.
    case.final.unlink()
    return directory, outcome


def test_g1_changed_current_version_rejects_committed_outcome_before_loading(
        boundary_support, boundary_case, monkeypatch):
    support, case = boundary_support, boundary_case
    generation, _ = seed_fitted_outcome(support, case)
    before = support.hashes(generation)
    with monkeypatch.context() as change:
        change.setattr(np, "__version__", np.__version__ + "+recovery-test-g1")
        with patch.object(support.recovery.RecoveryStore, "restore_model") as restore, \
                patch.object(KNeighborsClassifier, "fit") as fit, \
                identity_witness(support) as counts, \
                pytest.raises(support.recovery.RecoveryError, match="incompatible identity"):
            support.aggregate(case)
        restore.assert_not_called()
        fit.assert_not_called()
    assert_counts(counts, opted=1, full=1, observation=0)
    assert not case.final.exists()
    assert support.hashes(generation) == before


def test_g2_version_change_after_payload_verification_rejects_at_fast_preload_guard(
        boundary_support, boundary_case, monkeypatch):
    support, case = boundary_support, boundary_case
    generation, _ = seed_fitted_outcome(support, case)
    before = support.hashes(generation)
    original_load = support.recovery.RecoveryStore.load_outcome
    calls = []
    with monkeypatch.context() as change:
        def verified_then_changed(store):
            outcome = original_load(store)
            assert outcome is not None and outcome["manifest"]["kind"] == "fitted"
            calls.append(outcome["generation"])
            change.setattr(np, "__version__", np.__version__ + "+recovery-test-g2")
            return outcome

        change.setattr(support.recovery.RecoveryStore, "load_outcome", verified_then_changed)
        with patch.object(support.recovery.RecoveryStore, "restore_model") as restore, \
                patch.object(KNeighborsClassifier, "fit") as fit, \
                identity_witness(support) as counts, \
                pytest.raises(support.recovery.RecoveryError, match="Source or active runtime changed"):
            support.aggregate(case)
        restore.assert_not_called()
        fit.assert_not_called()
    assert calls == [generation]
    assert_counts(counts, opted=2, full=1, observation=1)
    assert counts["caches"][0] == {}
    assert not case.final.exists()
    assert support.hashes(generation) == before


def test_g3_version_change_after_successful_fit_rejects_at_fast_precommit_guard(
        boundary_support, boundary_case, monkeypatch):
    support, case = boundary_support, boundary_case
    original_fit = KNeighborsClassifier.fit
    calls = []
    with monkeypatch.context() as change:
        def fitted_then_changed(model, *args, **kwargs):
            fitted = original_fit(model, *args, **kwargs)
            calls.append(fitted)
            change.setattr(np, "__version__", np.__version__ + "+recovery-test-g3")
            return fitted

        change.setattr(KNeighborsClassifier, "fit", fitted_then_changed)
        with patch.object(support.recovery.RecoveryStore, "commit") as commit, \
                identity_witness(support) as counts, \
                pytest.raises(support.recovery.RecoveryError, match="Source or active runtime changed"):
            support.aggregate(case)
        commit.assert_not_called()
    assert len(calls) == 1
    assert_counts(counts, opted=2, full=1, observation=1)
    assert counts["caches"][0] == {}
    assert not (case.fit / "current.json").exists()
    assert not case.final.exists()


def test_g4_version_change_after_evaluation_preserves_fit_and_retry_reuses_it(
        boundary_support, boundary_case, monkeypatch):
    support, case = boundary_support, boundary_case
    original_evaluate = support.study.evaluate_classifier
    reports = []
    with monkeypatch.context() as change:
        def evaluated_then_changed(*args, **kwargs):
            report = original_evaluate(*args, **kwargs)
            assert report["status"] == "valid"
            reports.append(report)
            change.setattr(np, "__version__", np.__version__ + "+recovery-test-g4")
            return report

        change.setattr(support.study, "evaluate_classifier", evaluated_then_changed)
        with patch.object(support.study, "_publish_final_result") as publish, \
                identity_witness(support) as failed_counts, \
                pytest.raises(support.recovery.RecoveryError, match="Source or active runtime changed"):
            support.aggregate(case)
        publish.assert_not_called()
    assert len(reports) == 1
    assert_counts(failed_counts, opted=3, full=1, observation=2)
    assert failed_counts["caches"][0] == {}
    assert not case.final.exists()
    generation, outcome = support.committed(case)
    assert outcome["kind"] == "fitted"
    before = support.hashes(generation)

    # Actual version and ordinary evaluator have been restored. A fresh public
    # aggregate must create a fresh initialization token and load the old fit.
    with patch.object(KNeighborsClassifier, "fit", side_effect=AssertionError("Unexpected refit")) as fit, \
            identity_witness(support) as retry_counts:
        result = support.aggregate(case)
    fit.assert_not_called()
    assert_counts(retry_counts, opted=3, full=1, observation=2)
    assert retry_counts["caches"][0] is not failed_counts["caches"][0]
    assert result["evaluation"]["status"] == "valid"
    assert result["evaluation"]["predictions"]["predicted_labels"] == case.yv.tolist()
    assert result["recovery"]["fit_reused"] and not result["recovery"]["fit_performed"]
    assert result["recovery"]["new_fit_seconds"] == 0.0
    for key in ("final_fit_seconds", "diagnostics", "final_fit_warnings"):
        assert result[key] == outcome["metadata"][key]
    assert support.hashes(generation) == before
    assert case.final.exists()
