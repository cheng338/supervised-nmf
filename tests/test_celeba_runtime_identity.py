"""Initialization reuse always retains fresh identity and error boundaries."""
import builtins
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.linalg import blas
import threadpoolctl
from tests._celeba_recovery_test_support import recovery_module


class InjectedFailure(RuntimeError):
    pass


@pytest.fixture
def recovery(recovery_module):
    return recovery_module


@contextmanager
def observed_calls(recovery):
    """Count original helper/probe execution without replacing its callables."""
    previous = sys.getprofile()
    helper_code = recovery.active_runtime_identity.__code__
    observation_code = recovery._runtime_identity_observation.__code__
    ones_code = np.ones.__code__
    counts = {"full": 0, "opted": 0, "observation": 0, "ones": 0}

    def profile(frame, event, arg):
        if event == "call":
            if frame.f_code is helper_code:
                kind = "full" if frame.f_locals.get("_initialization_cache") is None else "opted"
                counts[kind] += 1
            elif frame.f_code is observation_code:
                counts["observation"] += 1
            elif frame.f_code is ones_code:
                parent = frame.f_back
                while parent is not None:
                    if parent.f_code is helper_code or parent.f_code is observation_code:
                        counts["ones"] += 1
                        break
                    parent = parent.f_back
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def identity(recovery, state, files, threads=1):
    return recovery.active_runtime_identity(threads, file_cache=files,
                                             _initialization_cache=state)


def assert_not_ready(state):
    assert "ready" not in state
    assert "ready_threads" not in state


def test_first_success_reuses_only_initialization(recovery):
    state, files = {}, {}
    with observed_calls(recovery) as first_count:
        first = identity(recovery, state, files)
    assert first_count["full"] == 1 and first_count["ones"] == 4
    assert state == {"ready": True, "ready_threads": 1}
    with observed_calls(recovery) as next_count:
        second = identity(recovery, state, files)
    assert next_count["full"] == 0 and next_count["ones"] == 0
    assert next_count["observation"] == 1
    assert first == second
    assert state is not first and state is not second


def test_default_direct_helper_always_runs_original_probes(recovery):
    files = {}
    with observed_calls(recovery) as counts:
        values = [recovery.active_runtime_identity(1, file_cache=files) for _ in range(3)]
    assert counts["full"] == 3 and counts["ones"] == 12
    assert counts["opted"] == counts["observation"] == 0
    assert values[0] == values[1] == values[2]


@pytest.mark.parametrize("requested", [(1, 2, 1), (2, 1, 2)])
def test_changed_requested_threads_reinitialize_and_restore_ambient(recovery, requested):
    state, files = {}, {}
    with threadpoolctl.threadpool_limits(limits=2):
        before = {row["filepath"]: row["num_threads"] for row in threadpoolctl.threadpool_info()}
        assert before and all(count == 2 for count in before.values())
        with observed_calls(recovery) as counts:
            for threads in requested:
                result = identity(recovery, state, files, threads)
                assert result["threads"] == threads
                assert {row["filepath"]: row["num_threads"] for row in threadpoolctl.threadpool_info()} == before
        assert counts["full"] == 3 and counts["ones"] == 12
        with observed_calls(recovery) as reused:
            identity(recovery, state, files, requested[-1])
        assert reused["full"] == reused["ones"] == 0
        assert {row["filepath"]: row["num_threads"] for row in threadpoolctl.threadpool_info()} == before


def test_modified_recovery_hook_uses_original_path_and_rebuilds(recovery, monkeypatch):
    state, files = {}, {}
    first = identity(recovery, state, files)
    with monkeypatch.context() as change:
        change.setattr(recovery, "_hook", lambda *args, **kwargs: None)
        with observed_calls(recovery) as counts:
            assert identity(recovery, state, files) == first
            assert identity(recovery, state, files) == first
        assert counts["full"] == 2 and counts["ones"] == 8
        assert_not_ready(state)
    with observed_calls(recovery) as retry:
        assert identity(recovery, state, files) == first
    assert retry["full"] == 1 and retry["ones"] == 4


@pytest.mark.parametrize("error_policy", ["raise", "warn", "ignore"])
def test_strict_numpy_policies_do_not_require_repeated_positive_probes(recovery, error_policy):
    state, files = {}, {}
    with np.errstate(all=error_policy), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        before = np.geterr().copy()
        first = identity(recovery, state, files)
        with observed_calls(recovery) as counts:
            repeated = identity(recovery, state, files)
        direct = recovery.active_runtime_identity(1, file_cache=files)
        assert np.geterr() == before
    assert first == repeated == direct
    assert not caught
    assert counts["ones"] == counts["full"] == 0


@pytest.mark.parametrize("action", ["error", "always"])
def test_warning_policy_parity_without_canonical_filter_gate(recovery, action):
    state, files = {}, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        before = list(warnings.filters)
        first = identity(recovery, state, files)
        with observed_calls(recovery) as counts:
            repeated = identity(recovery, state, files)
        direct = recovery.active_runtime_identity(1, file_cache=files)
        assert warnings.filters == before
    assert first == repeated == direct
    assert not caught
    assert counts["ones"] == counts["full"] == 0


def inject_failure(recovery, change, phase):
    def fail(*args, **kwargs):
        raise InjectedFailure(phase)

    if phase.startswith("context_"):
        original = threadpoolctl.threadpool_limits

        class BrokenContext:
            def __init__(self, *args, **kwargs):
                if phase == "context_construct":
                    fail()
                self.inner = original(*args, **kwargs)

            def __enter__(self):
                if phase == "context_enter":
                    # The real constructor applies limits. This artificial
                    # enter failure cleans its own injected state for the test.
                    self.inner.restore_original_limits()
                    fail()
                return self.inner.__enter__()

            def __exit__(self, kind, error, traceback):
                result = self.inner.__exit__(kind, error, traceback)
                if phase == "context_exit":
                    fail()
                return result

        change.setattr(threadpoolctl, "threadpool_limits", BrokenContext)
    elif phase == "numpy_probe":
        change.setattr(np, "dot", fail)
    elif phase == "scipy_probe":
        change.setattr(blas, "dgemv", fail)
    elif phase == "inventory":
        change.setattr(threadpoolctl, "threadpool_info", fail)
    elif phase == "native_hash":
        change.setattr(recovery, "_runtime_file_hash", fail)
    elif phase == "helper_hash":
        original = recovery._sha

        def source_hash(path):
            if Path(path).name == "numeric_codec.py":
                fail()
            return original(path)

        change.setattr(recovery, "_sha", source_hash)
    else:
        raise AssertionError(phase)


@pytest.mark.parametrize("phase", ["context_construct", "context_enter", "numpy_probe", "scipy_probe",
                                   "context_exit", "inventory", "native_hash", "helper_hash"])
def test_failed_first_initialization_never_marks_ready_and_retry_is_full(recovery, monkeypatch, phase):
    state, files = {}, {}
    with monkeypatch.context() as change:
        inject_failure(recovery, change, phase)
        with pytest.raises(InjectedFailure, match=phase):
            identity(recovery, state, files)
    assert_not_ready(state)
    with observed_calls(recovery) as counts:
        retried = identity(recovery, state, files)
    assert counts["full"] == 1 and counts["ones"] == 4
    assert state["ready"] is True
    assert retried == recovery.active_runtime_identity(1, file_cache=files)


@pytest.mark.parametrize("phase", ["inventory", "native_hash", "helper_hash"])
def test_failed_fresh_observation_invalidates_before_full_retry(recovery, monkeypatch, phase):
    state, files = {}, {}
    first = identity(recovery, state, files)
    with monkeypatch.context() as change:
        inject_failure(recovery, change, phase)
        with pytest.raises(InjectedFailure, match=phase):
            identity(recovery, state, files)
    assert_not_ready(state)
    with observed_calls(recovery) as counts:
        retried = identity(recovery, state, files)
    assert retried == first
    assert counts["full"] == 1 and counts["ones"] == 4


@pytest.mark.parametrize("field", ["version", "source", "library_inventory", "native_file"])
def test_fresh_observation_never_returns_cached_identity(recovery, monkeypatch, tmp_path, field):
    state, files = {}, {}
    harmless = tmp_path / "owned-library-identity.bin"
    harmless.write_bytes(b"first")
    real_inventory = threadpoolctl.threadpool_info
    extra = {"user_api": "blas", "internal_api": "owned-test", "prefix": "owned", "version": "1",
             "threading_layer": "test", "architecture": "test", "num_threads": 1,
             "filepath": str(harmless.resolve())}

    def inventory():
        return real_inventory() + [dict(extra)]

    if field == "native_file":
        monkeypatch.setattr(threadpoolctl, "threadpool_info", inventory)
    first = identity(recovery, state, files)
    if field == "version":
        monkeypatch.setattr(np, "__version__", np.__version__ + ".changed")
    elif field == "source":
        original = recovery._sha

        def source_hash(path):
            return "0" * 64 if Path(path).name == "numeric_codec.py" else original(path)

        monkeypatch.setattr(recovery, "_sha", source_hash)
    elif field == "library_inventory":
        monkeypatch.setattr(threadpoolctl, "threadpool_info", inventory)
    elif field == "native_file":
        harmless.write_bytes(b"second-content-with-a-different-size")
    with observed_calls(recovery) as counts:
        changed = identity(recovery, state, files)
    direct = recovery.active_runtime_identity(1, file_cache=files)
    assert changed != first
    assert changed == direct
    assert counts["full"] == counts["ones"] == 0
    assert counts["observation"] == 1
    # The helper reads facts; the original runner compares/rejects them.
    assert state["ready"] is True


def test_separate_aggregation_caches_do_not_share_readiness(recovery):
    first_state, second_state, files = {}, {}, {}
    with observed_calls(recovery) as counts:
        first = identity(recovery, first_state, files)
        assert identity(recovery, first_state, files) == first
        second = identity(recovery, second_state, files)
    assert counts["full"] == 2 and counts["ones"] == 8
    assert counts["observation"] == 1
    assert first == second
    assert first_state is not second_state
    assert first_state["ready"] and second_state["ready"]


@pytest.mark.parametrize("cached", [False, True])
def test_initialization_retains_bindings_captured_before_probes(recovery, monkeypatch, cached):
    numpy_version = np.__version__
    rebound = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Initialization must retain its captured native binding")

    with monkeypatch.context() as change:
        change.setattr(blas, "dgemv", lambda *args: None)
        change.setattr(threadpoolctl, "threadpool_info", lambda: [])

        def rebind(*args):
            rebound.append(True)
            change.setattr(blas, "dgemv", forbidden)
            change.setattr(threadpoolctl, "threadpool_info", forbidden)
            change.setitem(sys.modules, "numpy", SimpleNamespace(__version__="replacement"))

        change.setattr(np, "dot", rebind)
        result = identity(recovery, {}, {}) if cached else recovery.active_runtime_identity(1)

    assert sys.modules["numpy"] is np
    assert rebound == [True]
    assert result["versions"]["numpy"] == numpy_version
    assert result["native_libraries"] == []


@pytest.mark.parametrize("ready", [False, True])
def test_dependency_import_failure_clears_readiness_before_full_retry(recovery, monkeypatch, ready):
    retained = object()
    state, files = {"caller": retained}, {"previous": retained}
    if ready:
        state.update(ready=True, ready_threads=1, _inventory_state=object())
    failure = KeyboardInterrupt("owned dependency import failure")
    original_import = builtins.__import__

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "joblib":
            raise failure
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as change:
        change.setattr(builtins, "__import__", unavailable)
        with observed_calls(recovery) as failed, pytest.raises(KeyboardInterrupt) as caught:
            identity(recovery, state, files)
    assert caught.value is failure and failed["ones"] == 0
    assert state == {"caller": retained} and files == {"previous": retained}

    with observed_calls(recovery) as retried:
        identity(recovery, state, files)
    assert retried["full"] == 1 and retried["ones"] == 4
    assert state["ready"] is True and state["ready_threads"] == 1
    assert state["caller"] is files["previous"] is retained


def test_observation_without_inventory_cache_does_not_initialize(recovery, monkeypatch):
    inventories = []
    monkeypatch.setattr(recovery, "_inventory_fresh_info", lambda cache: inventories.append(cache) or [])
    with observed_calls(recovery) as counts:
        result = recovery._runtime_identity_observation(1)
    assert inventories == [None] and result["native_libraries"] == []
    assert counts["full"] == counts["ones"] == 0
