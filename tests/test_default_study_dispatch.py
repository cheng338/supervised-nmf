"""Resume CLI routing contracts; fake pools do not validate child startup."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import warnings

import pytest
SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "scripts/celeba_default_study.py"
SPEC = importlib.util.spec_from_file_location("_default_study_dispatch_tests", SCRIPT)
DEFAULT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEFAULT
SPEC.loader.exec_module(DEFAULT)
assert Path(DEFAULT.__file__).resolve() == SCRIPT


@pytest.mark.parametrize("platform,flag,method", [
    ("linux", False, "spawn"), ("linux", True, "forkserver"),
    ("win32", False, "spawn"), ("win32", True, "spawn"),
    ("darwin", False, "spawn"), ("darwin", True, "spawn"),
])
def test_selected_context_keeps_all_eight_original_workers(monkeypatch, platform, flag, method, present_checkpoints):
    work = [object() for _ in range(8)]
    calls, contexts, completed = [], [], []
    class Context:
        def set_forkserver_preload(self, names):
            assert names == []
            calls.append(("preload", names))
    context = Context()

    class Future:
        def __init__(self, item):
            self.item = item
        def result(self):
            return self.item

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 8 and mp_context is context
        def __enter__(self):
            return self
        def __exit__(self, *error):
            calls.append(("exit", error[0]))
            return False
        def submit(self, function, item):
            assert function is DEFAULT._worker
            calls.append(("submit", item))
            return Future(item)

    def get_context(name):
        contexts.append(name)
        return context

    def try_completed(items, workers, **kwargs):
        completed.append((items, workers, set(kwargs)))
        return None

    monkeypatch.setattr(DEFAULT.sys, "platform", platform)
    monkeypatch.setattr(DEFAULT, "_try_completed", try_completed)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", get_context)
    monkeypatch.setattr(DEFAULT, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(DEFAULT, "as_completed", lambda futures: reversed(futures))
    assert list(DEFAULT._completed_or_spawned(work, 8, _cli_forkserver=flag)) == list(reversed(work))
    assert contexts == [method]
    assert completed == [(work, 8, {"describe", "invoke"})]
    assert calls == ([("preload", [])] if method == "forkserver" else []) + [("submit", item) for item in work] + [("exit", None)]


@pytest.mark.parametrize("flag", [1, "enabled", None])
def test_nonboolean_private_values_do_not_enable_forkserver(monkeypatch, flag, present_checkpoints):
    selected = []
    monkeypatch.setattr(DEFAULT.sys, "platform", "linux")
    monkeypatch.setattr(DEFAULT, "_try_completed", lambda *args, **kwargs: None)

    def context(name):
        selected.append(name)
        raise OSError("context sentinel")

    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", context)
    with pytest.raises(OSError, match="context sentinel"):
        list(DEFAULT._completed_or_spawned([object()], 8, _cli_forkserver=flag))
    assert selected == ["spawn"]


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_one_worker_completed_reuse_precedes_any_context(monkeypatch, flag, frozen, present_checkpoints):
    item, environment, token = object(), object(), object()
    pin = {"sha256": "a" * 64}
    calls = []

    def worker(value, *, reuse_only, _reuse_environment, **kwargs):
        assert value is item and reuse_only is True and _reuse_environment is environment
        assert kwargs == ({"_frozen_report": pin} if frozen else {})
        calls.append(value)
        return token

    def try_completed(work, workers, *, describe, invoke):
        assert work == [item] and workers == 1
        assert callable(describe)
        return [invoke(item, environment)]

    monkeypatch.setattr(DEFAULT, "_worker", worker)
    monkeypatch.setattr(DEFAULT, "_try_completed", try_completed)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context",
                        lambda *args: pytest.fail("completed reuse constructed a context"))
    monkeypatch.setattr(DEFAULT, "ProcessPoolExecutor",
                        lambda **kwargs: pytest.fail("completed reuse created a pool"))
    pins = [pin] if frozen else None
    assert list(DEFAULT._completed_or_spawned([item], 1, _cli_forkserver=flag, _pins=pins)) == [token]
    assert calls == [item]


def test_empty_work_does_not_create_process_context(monkeypatch, present_checkpoints):
    monkeypatch.setattr(DEFAULT, "_try_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context",
                        lambda *args: pytest.fail("empty work created a context"))
    assert list(DEFAULT._completed_or_spawned([], 8, _cli_forkserver=True)) == []


@pytest.mark.parametrize("kind", [OSError, ValueError, KeyboardInterrupt])
def test_completed_guard_errors_propagate_without_pool_retry(monkeypatch, kind, present_checkpoints):
    error = kind("authoritative completed error")
    calls = []

    def completed(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(DEFAULT, "_try_completed", completed)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context",
                        lambda *args: pytest.fail("guard error created a context"))
    with pytest.raises(kind) as caught:
        list(DEFAULT._completed_or_spawned([object()], 8, _cli_forkserver=True))
    assert caught.value is error and calls == [True]


@pytest.mark.parametrize("stage", ["context", "preload", "constructor", "submit", "result"])
def test_executor_failures_are_not_retried_or_serialized_as_success(monkeypatch, stage, present_checkpoints):
    error = RuntimeError("owned executor failure")
    events = []
    class Context:
        def set_forkserver_preload(self, names):
            assert names == []
            events.append("preload")
            if stage == "preload":
                raise error
    context = Context()

    def selected(name):
        events.append("context")
        assert name == "forkserver"
        if stage == "context":
            raise error
        return context

    class Future:
        def result(self):
            events.append("result")
            raise error

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            events.append("constructor")
            assert max_workers == 8 and mp_context is context
            if stage == "constructor":
                raise error
        def __enter__(self):
            events.append("enter")
            return self
        def __exit__(self, kind, value, trace):
            events.append("exit")
            assert value is error
            return False
        def submit(self, function, item):
            assert function is DEFAULT._worker
            events.append("submit")
            if stage == "submit":
                raise error
            return Future()

    monkeypatch.setattr(DEFAULT.sys, "platform", "linux")
    monkeypatch.setattr(DEFAULT, "_try_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", selected)
    monkeypatch.setattr(DEFAULT, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(DEFAULT, "as_completed", lambda values: values)
    with pytest.raises(RuntimeError) as caught:
        list(DEFAULT._completed_or_spawned([object()], 8, _cli_forkserver=True))
    assert caught.value is error
    expected = {
        "context": ["context"],
        "preload": ["context", "preload"],
        "constructor": ["context", "preload", "constructor"],
        "submit": ["context", "preload", "constructor", "enter", "submit", "exit"],
        "result": ["context", "preload", "constructor", "enter", "submit", "result", "exit"],
    }
    assert events == expected[stage]


@pytest.mark.parametrize("policy", ["always", "error"])
def test_context_warning_follows_existing_filter(monkeypatch, policy, present_checkpoints):
    events = []
    sentinel = RuntimeError("context completed")

    def context(name):
        events.append(name)
        warnings.warn("context warning", RuntimeWarning)
        raise sentinel

    monkeypatch.setattr(DEFAULT.sys, "platform", "linux")
    monkeypatch.setattr(DEFAULT, "_try_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", context)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        with pytest.raises(RuntimeWarning if policy == "error" else RuntimeError) as error:
            list(DEFAULT._completed_or_spawned([object()], 8, _cli_forkserver=True))
        if policy == "always":
            assert error.value is sentinel
            assert [(row.category, str(row.message)) for row in caught] == [(RuntimeWarning, "context warning")]
        else:
            assert not caught
    assert events == ["forkserver"]


@pytest.mark.parametrize("flag", [False, True])
def test_run_cv_default_call_signature_and_explicit_private_forwarding(monkeypatch, tmp_path, flag, present_checkpoints):
    configuration = {"sources": {"sentinel": 1}, "plans": [
        {"job": {"id": "j"}, "chunks": [{"id": "c0000"}]}]}
    calls = []
    row = {"job": "j", "chunk": "c0000", "valid_candidates": 1, "candidates": 1,
           "new_representation_fits": 0, "reused_representation_folds": 5}

    def default_scheduler(work, workers):
        calls.append((work, workers, {}))
        yield row

    def explicit_scheduler(work, workers, *, _cli_forkserver):
        assert _cli_forkserver is True
        calls.append((work, workers, {"_cli_forkserver": True}))
        yield row

    monkeypatch.setattr(DEFAULT, "_completed_or_spawned", explicit_scheduler if flag else default_scheduler)
    monkeypatch.setattr(DEFAULT, "aggregate_cv",
                        lambda output, configuration: {"state": "cv_completed", "missing_chunks": []})
    result = DEFAULT.run_cv(tmp_path, configuration, workers=8,
                            **({"_cli_forkserver": True} if flag else {}))
    assert result == {"state": "cv_completed", "completed_chunks": [row], "missing_chunks": []}
    assert calls == [([(str(tmp_path), configuration["plans"][0],
                       configuration["plans"][0]["chunks"][0], configuration["sources"])],
                     8, {"_cli_forkserver": True} if flag else {})]


@pytest.mark.parametrize("flag,mode", [(False, "cv"), (True, "cv"), (False, "evaluate"), (True, "evaluate")])
def test_imported_main_default_and_explicit_flag_keep_public_flow(monkeypatch, tmp_path, flag, mode, present_checkpoints):
    output = tmp_path / "output"
    configuration = {"plans": [], "sources": {}}
    calls, writes = [], []
    monkeypatch.setattr(DEFAULT.sys, "argv", [str(SCRIPT), "--protocol", str(tmp_path / "protocol.json"),
                                           "--output", str(output), "--mode", mode, "--workers", "8"])
    monkeypatch.setattr(DEFAULT, "_checkpoint_lock", lambda path: nullcontext())
    monkeypatch.setattr(DEFAULT, "initialize", lambda protocol, root: configuration)
    monkeypatch.setattr(DEFAULT, "_write_receipt", lambda path, value: writes.append((path, dict(value))))

    def run_cv(root, saved, *, workers, job_ids, **kwargs):
        assert root == output and saved is configuration
        calls.append(("cv", workers, job_ids, kwargs))
        return {"state": "cv_completed"}

    def evaluate(root, saved):
        assert root == output and saved is configuration
        calls.append(("evaluate",))
        return {"state": "development_evaluation_completed"}

    monkeypatch.setattr(DEFAULT, "run_cv", run_cv)
    monkeypatch.setattr(DEFAULT, "evaluate", evaluate)
    DEFAULT.main(**({"_cli_forkserver": True} if flag else {}))
    expected = [("evaluate",)] if mode == "evaluate" else [
        ("cv", 8, None, {"_cli_forkserver": True} if flag else {})]
    assert calls == expected
    assert len(writes) == 2 and writes[0][1]["state"] == "running"
    assert writes[-1][1]["state"] == "completed"
    assert writes[-1][1]["workers"] == (8 if mode == "cv" else 1)


@pytest.mark.parametrize("flag", [False, True])
def test_imported_main_preserves_authoritative_error_receipt(monkeypatch, tmp_path, flag, present_checkpoints):
    error = ValueError("authoritative worker error")
    calls, writes = [], []
    monkeypatch.setattr(DEFAULT.sys, "argv", [str(SCRIPT), "--protocol", str(tmp_path/"protocol.json"),
                                           "--output", str(tmp_path/"output"), "--mode", "cv"])
    monkeypatch.setattr(DEFAULT, "_checkpoint_lock", lambda path: nullcontext())
    monkeypatch.setattr(DEFAULT, "initialize", lambda *args: {"plans": [], "sources": {}})
    monkeypatch.setattr(DEFAULT, "_write_receipt", lambda path, value: writes.append(dict(value)))

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(DEFAULT, "run_cv", fail)
    with pytest.raises(ValueError) as caught:
        DEFAULT.main(**({"_cli_forkserver": True} if flag else {}))
    assert caught.value is error and len(calls) == 1
    assert len(writes) == 2
    assert writes[-1]["state"] == "failed" and writes[-1]["error_type"] == "ValueError"
    assert writes[-1]["error"] == str(error)


@pytest.mark.parametrize("policy", ["always", "error"])
def test_preload_warning_or_error_precedes_pool_and_is_not_retried(monkeypatch, policy, present_checkpoints):
    events = []
    sentinel = RuntimeError("preload completed")

    class Context:
        def set_forkserver_preload(self, names):
            events.append(("preload", names))
            warnings.warn("preload warning", RuntimeWarning)
            raise sentinel

    def context(method):
        assert method == "forkserver"
        events.append(("context", method))
        return Context()

    monkeypatch.setattr(DEFAULT.sys, "platform", "linux")
    monkeypatch.setattr(DEFAULT, "_try_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", context)
    monkeypatch.setattr(DEFAULT, "ProcessPoolExecutor",
                        lambda **kwargs: pytest.fail("failed preload constructed pool"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        with pytest.raises(RuntimeWarning if policy == "error" else RuntimeError) as error:
            list(DEFAULT._completed_or_spawned([object()], 8, _cli_forkserver=True))
        if policy == "always":
            assert error.value is sentinel
            assert [(row.category, str(row.message)) for row in caught] == [(RuntimeWarning, "preload warning")]
        else:
            assert not caught
    assert events == [("context", "forkserver"), ("preload", [])]


@pytest.fixture
def present_checkpoints(monkeypatch):
    """Provide present checkpoints for controls that isolate dispatcher behavior."""
    monkeypatch.setattr(DEFAULT, "_checkpoint_files_present", lambda work: True)


def _work(tmp_path, count=2):
    result = []
    for index in range(count):
        chunk = {"id": f"c{index:04d}"}
        plan = {"job": {"id": "j"}}
        directory = tmp_path / "j" / chunk["id"]
        directory.mkdir(parents=True)
        # Presence is not validity: these bytes are intentionally not receipts.
        (directory / "identity.json").write_bytes(b"not validated")
        (directory / "report.json").write_bytes(b"not parsed")
        result.append((str(tmp_path), plan, chunk, {"unchanged": True}))
    return result


@pytest.mark.parametrize("state", ["present", "missing_identity", "missing_report",
                                  "identity_directory", "report_directory", "invalid_bytes"])
def test_real_hint_requires_files_but_never_parses_or_builds_models(tmp_path, monkeypatch, state):
    work = _work(tmp_path)
    directory = tmp_path / "j/c0000"
    if state in ("missing_identity", "identity_directory"):
        (directory / "identity.json").unlink()
        if state == "identity_directory":
            (directory / "identity.json").mkdir()
    elif state in ("missing_report", "report_directory"):
        (directory / "report.json").unlink()
        if state == "report_directory":
            (directory / "report.json").mkdir()
    elif state == "invalid_bytes":
        (directory / "report.json").write_bytes(b"\xff\x00malformed")
    monkeypatch.setattr(DEFAULT, "_read_receipt", lambda *a, **k: pytest.fail("hint parsed receipt"))
    monkeypatch.setattr(DEFAULT, "_estimator", lambda *a, **k: pytest.fail("hint constructed estimator"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("hint read payload"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("hint read payload"))
    assert DEFAULT._checkpoint_files_present(work) is (state in ("present", "invalid_bytes"))


@pytest.mark.parametrize("missing", ["identity.json", "report.json"])
def test_first_missing_file_shortcuts_remaining_stats(tmp_path, monkeypatch, missing):
    work = _work(tmp_path, 8)
    (tmp_path / "j/c0000" / missing).unlink()
    original = Path.is_file
    calls = []

    def observed(path):
        calls.append(path.relative_to(tmp_path).as_posix())
        return original(path)

    monkeypatch.setattr(Path, "is_file", observed)
    assert DEFAULT._checkpoint_files_present(work) is False
    assert calls == (["j/c0000/identity.json"] if missing == "identity.json" else
                     ["j/c0000/identity.json", "j/c0000/report.json"])


@pytest.mark.parametrize("kind", [PermissionError, OSError, ValueError, TypeError])
def test_ordinary_hint_path_errors_decline_without_acceptance(tmp_path, monkeypatch, kind):
    work = _work(tmp_path)
    error = kind("optional file hint failed")
    calls = []

    def fail(path):
        calls.append(path)
        raise error

    monkeypatch.setattr(Path, "is_file", fail)
    assert DEFAULT._checkpoint_files_present(work) is False
    assert len(calls) == 1


@pytest.mark.parametrize("issue", ["invalid_path", "missing_job", "missing_chunk", "wrong_tuple"])
def test_malformed_optional_path_shape_defers_to_worker(tmp_path, issue):
    work = _work(tmp_path)
    if issue == "invalid_path":
        work[0] = (object(), *work[0][1:])
    elif issue == "missing_job":
        work[0][1]["job"].clear()
    elif issue == "missing_chunk":
        work[0][2].clear()
    else:
        work[0] = (str(tmp_path),)
    assert DEFAULT._checkpoint_files_present(work) is False


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_non_hint_errors_are_not_swallowed(tmp_path, monkeypatch, kind):
    work = _work(tmp_path)
    error = kind("stop hint")

    def fail(path):
        raise error

    monkeypatch.setattr(Path, "is_file", fail)
    with pytest.raises(kind) as caught:
        DEFAULT._checkpoint_files_present(work)
    assert caught.value is error


@pytest.mark.parametrize("case", ["imported_default", "nonlinux", "empty", "completed_one", "single_incomplete"])
def test_hint_is_unreachable_outside_eligible_cli_pool(tmp_path, monkeypatch, case):
    work = _work(tmp_path) if case == "single_incomplete" else ([] if case == "empty" else [object()])
    if case == "single_incomplete":
        assert DEFAULT._checkpoint_files_present(work) is True
    token = object()
    events = []

    def completed(items, workers, **kwargs):
        events.append(("completed", workers))
        return [token] if case == "completed_one" else None

    def context(method):
        events.append(("context", method))
        assert method == "spawn"
        raise OSError("original spawn selected")

    monkeypatch.setattr(DEFAULT.sys, "platform", "win32" if case == "nonlinux" else "linux")
    monkeypatch.setattr(DEFAULT, "_try_completed", completed)
    monkeypatch.setattr(DEFAULT, "_checkpoint_files_present",
                        lambda *a: pytest.fail("ineligible call inspected checkpoint paths"))
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", context)
    kwargs = {} if case == "imported_default" else {"_cli_forkserver": True}
    workers = 1 if case in ("completed_one", "single_incomplete") else 8
    if case == "completed_one":
        assert list(DEFAULT._completed_or_spawned(work, workers, **kwargs)) == [token]
    elif case == "empty":
        assert list(DEFAULT._completed_or_spawned(work, workers, **kwargs)) == []
    else:
        with pytest.raises(OSError, match="original spawn selected"):
            list(DEFAULT._completed_or_spawned(work, workers, **kwargs))
    assert events[0] == ("completed", workers)
    assert len(events) == (1 if case in ("empty", "completed_one") else 2)


@pytest.mark.parametrize("state", ["present", "missing", "hint_error"])
@pytest.mark.parametrize("frozen", [False, True])
def test_real_hint_selects_only_backend_and_preserves_worker_dispatch(tmp_path, monkeypatch, state, frozen):
    work = _work(tmp_path)
    pins = [{"sha256": str(i)} for i in range(len(work))] if frozen else None
    expected = "forkserver" if state == "present" else "spawn"
    events, submitted = [], []
    original_is_file = Path.is_file

    def is_file(path):
        events.append(("hint", path.name))
        if state == "hint_error":
            raise PermissionError("stat denied")
        return original_is_file(path)

    if state == "missing":
        (tmp_path / "j/c0000/report.json").unlink()

    class Context:
        def set_forkserver_preload(self, modules):
            assert expected == "forkserver" and modules == []
            events.append(("preload", []))

    class Future:
        def __init__(self, item):
            self.item = item
        def result(self):
            return self.item

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 8 and isinstance(mp_context, Context)
            events.append(("pool", max_workers))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append(("exit", None))
        def submit(self, function, item, **kwargs):
            assert function is DEFAULT._worker
            assert kwargs == ({"reuse_only": True, "_frozen_report": pins[len(submitted)]} if frozen else {})
            submitted.append(item)
            return Future(item)

    def context(method):
        assert method == expected
        events.append(("context", method))
        return Context()

    monkeypatch.setattr(DEFAULT.sys, "platform", "linux")
    monkeypatch.setattr(DEFAULT, "_try_completed",
                        lambda *a, **k: events.append(("completed", 8)) or None)
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(DEFAULT.multiprocessing, "get_context", context)
    monkeypatch.setattr(DEFAULT, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(DEFAULT, "as_completed", lambda futures: futures)
    assert list(DEFAULT._completed_or_spawned(work, 8, _cli_forkserver=True, _pins=pins)) == work
    assert submitted == work
    assert events[0] == ("completed", 8)
    assert next(event for event in events if event[0] == "context") == ("context", expected)
    assert sum(event[0] == "preload" for event in events) == (expected == "forkserver")
    assert events[-1] == ("exit", None)
