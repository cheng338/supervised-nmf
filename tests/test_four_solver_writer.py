"""Four-solver JSON publication, staging ownership and bounded failure contracts."""
from concurrent.futures import ThreadPoolExecutor
import errno
import importlib.util
import json
import os
from pathlib import Path
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_four_solver_writer_under_test", ROOT / "scripts/run_four_solver_benchmarks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def existing(tmp_path):
    target = tmp_path / "status.json"
    files = {target: b'{"generation":"old"}\n',
             tmp_path / "status.json.tmp": b"another writer's staging",
             tmp_path / ".fdeadbeef0000": b"an interrupted writer's staging"}
    for path, content in files.items():
        path.write_bytes(content)
    return target, files


def unchanged(target, files):
    assert set(target.parent.iterdir()) == set(files)
    assert {path: path.read_bytes() for path in files} == files


def test_preserves_serializer_order_nonfinite_values_and_text_bytes(existing):
    target, files = existing
    value = {"z": [float("nan"), float("inf"), -0.0, None, True],
             "unicode": "caf\u00e9", 7: ("first", "second"), "a": {"z": 1, "a": 2}}
    runner.write_json(target, value)
    expected = (json.dumps(value, indent=2) + "\n").replace("\n", os.linesep).encode("utf-8")
    unchanged(target, files | {target: expected})


def test_serialization_failure_does_not_touch_files(existing):
    target, files = existing
    with pytest.raises(TypeError):
        runner.write_json(target, {"unsupported": object()})
    unchanged(target, files)


def test_missing_parent_is_not_created(tmp_path):
    target = tmp_path / "missing" / "status.json"
    with pytest.raises(FileNotFoundError):
        runner.write_json(target, {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("collisions", [1, 10])
def test_exclusive_collisions_never_truncate_or_remove_foreign_files(existing, monkeypatch, collisions):
    target, files = existing
    real_open = Path.open
    occupied = {}
    attempts = []

    def collide(path, mode="r", *args, **kwargs):
        if mode == "x":
            attempts.append(path)
            if len(occupied) < collisions:
                with real_open(path, "xb") as stream:
                    stream.write(b"foreign writer")
                occupied[path] = b"foreign writer"
                raise FileExistsError(errno.EEXIST, "injected exclusive collision", str(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", collide)
    if collisions == 10:
        with pytest.raises(FileExistsError):
            runner.write_json(target, {"new": True})
        assert len(attempts) == 10
    else:
        runner.write_json(target, {"new": True})
        assert len(attempts) == 2 and attempts[0] != attempts[1]
        files[target] = target.read_bytes()
        assert json.loads(files[target]) == {"new": True}
    unchanged(target, files | occupied)


@pytest.mark.parametrize("stage", ["partial_write", "write_interrupt", "close"])
def test_staging_failure_closes_handle_and_cleans_only_owned_file(existing, monkeypatch, stage):
    target, files = existing
    real_open, handles = Path.open, []
    error = KeyboardInterrupt("interrupted write") if stage == "write_interrupt" else OSError(errno.EIO, stage)

    class Stream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, value):
            if stage != "close":
                self.stream.write(value[:3])
                raise error
            return self.stream.write(value)

        def __exit__(self, *args):
            self.stream.close()
            if stage == "close":
                raise error

    def faulty_open(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if mode in ("x", "w"):
            handles.append(stream)
            return Stream(stream)
        return stream

    monkeypatch.setattr(Path, "open", faulty_open)
    with pytest.raises(type(error)) as caught:
        runner.write_json(target, {"new": "payload"})
    assert caught.value is error
    assert len(handles) == 1 and handles[0].closed
    unchanged(target, files)


@pytest.mark.parametrize("interrupt", [False, True], ids=["io_error", "interrupt"])
def test_nonpermission_replace_failure_is_not_retried(existing, monkeypatch, interrupt):
    target, files = existing
    error = KeyboardInterrupt("interrupted replace") if interrupt else OSError(errno.EIO, "replace failed")
    calls, sleeps = [], []

    def fail(source, destination):
        calls.append(source)
        assert json.loads(source.read_text()) == {"new": True}
        assert target.read_bytes() == files[target]
        raise error

    monkeypatch.setattr(Path, "replace", fail)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(type(error)) as caught:
        runner.write_json(target, {"new": True})
    assert caught.value is error and len(calls) == 1 and sleeps == []
    unchanged(target, files)


@pytest.mark.parametrize("denials", [1, 5, 6])
def test_permission_retries_preserve_old_target_until_bounded_publication(existing, monkeypatch, denials):
    target, files = existing
    real_replace, calls, sleeps = Path.replace, [], []
    error = PermissionError(errno.EACCES, "controlled replacement denial")

    def replace(source, destination):
        assert target.read_bytes() == files[target]
        assert json.loads(source.read_text()) == {"new": True}
        calls.append(source)
        if len(calls) <= denials:
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    if denials == 6:
        with pytest.raises(PermissionError) as caught:
            runner.write_json(target, {"new": True})
        assert caught.value is error
    else:
        runner.write_json(target, {"new": True})
        files[target] = target.read_bytes()
        assert json.loads(files[target]) == {"new": True}
    assert len(calls) == min(denials + 1, 6) and len(set(calls)) == 1
    assert sleeps == [.05 * 2**index for index in range(min(denials, 5))]
    unchanged(target, files)


def test_cleanup_denial_preserves_primary_error_and_orphan(existing, monkeypatch):
    target, files = existing
    error = OSError(errno.EIO, "primary replacement failure")
    staged = []

    def fail(source, destination):
        staged.append(source)
        raise error

    def deny_cleanup(path, *args, **kwargs):
        assert path == staged[0]
        raise PermissionError(errno.EACCES, "cleanup denied")

    monkeypatch.setattr(Path, "replace", fail)
    monkeypatch.setattr(Path, "unlink", deny_cleanup)
    with pytest.raises(OSError) as caught:
        runner.write_json(target, {"new": True})
    assert caught.value is error and len(staged) == 1
    assert json.loads(staged[0].read_text()) == {"new": True}
    unchanged(target, files | {staged[0]: staged[0].read_bytes()})


def test_publication_closes_short_sibling_staging_before_replace(existing, monkeypatch):
    target, files = existing
    real_open, real_replace, handles = Path.open, Path.replace, []

    def track(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if mode in ("x", "w"):
            handles.append(stream)
        return stream

    def replace(source, destination):
        assert len(handles) == 1 and handles[0].closed
        assert source.parent == target.parent and source != target
        assert len(source.name) <= len(target.name + ".tmp")
        assert json.loads(source.read_text()) == {"new": True}
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "open", track)
    monkeypatch.setattr(Path, "replace", replace)
    runner.write_json(target, {"new": True})
    unchanged(target, files | {target: target.read_bytes()})


def test_successful_replace_does_not_clean_a_reused_staging_name(existing, monkeypatch):
    target, files = existing
    real_replace, reused = Path.replace, {}

    def replace(source, destination):
        result = real_replace(source, destination)
        source.write_bytes(b"a new owner reused this name")
        reused[source] = source.read_bytes()
        return result

    monkeypatch.setattr(Path, "replace", replace)
    runner.write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    unchanged(target, files | {target: target.read_bytes()} | reused)


def test_two_direct_writers_publish_their_own_complete_payloads(existing, monkeypatch):
    target, files = existing
    ready, installed_first = threading.Barrier(2), threading.Event()
    owner = threading.local()
    real_replace, staging, installed = Path.replace, {}, {}
    values = {"first": {"owner": "first", "data": "A" * 4096},
              "last": {"owner": "last", "data": "B" * 8192}}

    def replace(source, destination):
        label = owner.label
        if label not in staging:
            staging[label] = source
            ready.wait(timeout=10)
        else:
            assert source == staging[label]
        assert json.loads(source.read_text()) == values[label]
        if label == "last":
            assert installed_first.wait(timeout=10)
        result = real_replace(source, destination)
        installed[label] = json.loads(target.read_text())
        if label == "first":
            installed_first.set()
        return result

    def write(label):
        owner.label = label
        runner.write_json(target, values[label])

    monkeypatch.setattr(Path, "replace", replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, label) for label in values]
        for future in futures:
            future.result(timeout=25)
    assert staging["first"] != staging["last"] and installed == values
    assert json.loads(target.read_text()) == values["last"]
    unchanged(target, files | {target: target.read_bytes()})


def test_interruption_after_publication_can_leave_the_new_complete_target(existing, monkeypatch):
    target, files = existing
    real_replace = Path.replace

    def interrupt(source, destination):
        real_replace(source, destination)
        raise KeyboardInterrupt("publication completed before interruption")

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    unchanged(target, files | {target: target.read_bytes()})


def test_interrupted_backoff_preserves_old_target_and_cleans_owned_staging(existing, monkeypatch):
    target, files = existing

    def denied(*args):
        raise PermissionError(errno.EACCES, "replacement denied")

    def interrupt(seconds):
        raise KeyboardInterrupt("interrupted backoff")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(runner.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.write_json(target, {"new": True})
    unchanged(target, files)
