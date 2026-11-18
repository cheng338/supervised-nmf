"""Atomic JSON staging and OS-backed checkpoint ownership contracts."""

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import stat
import threading

import numpy as np
import pytest

from supervised_nmf import evaluation, study


def _existing_report(directory):
    """Keep unrelated staging files as sentinels against broad cleanup."""
    target = directory / "report.json"
    target.write_bytes(b'{"generation": "previous"}\n')
    unrelated = directory / "report.json.tmp"
    unrelated.write_bytes(b"another writer owns this legacy staging file")
    orphan = directory / ".snmf-abandoned.tmp"
    orphan.write_bytes(b"an interrupted writer's staging file")
    return target, {path: path.read_bytes() for path in directory.iterdir()}


def _assert_files_unchanged(directory, original):
    assert set(directory.iterdir()) == set(original)
    assert {path: path.read_bytes() for path in directory.iterdir()} == original


def test_write_json_preserves_normalization_and_human_readable_format(tmp_path):
    target = tmp_path / "nested" / "report.json"
    evaluation.write_json(target, {
        "z": np.asarray([1.0, np.nan, np.inf]),
        "a": {"value": np.int64(2), "path": Path("example")},
    })
    assert target.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "path": "example",\n    "value": 2\n  },\n'
        '  "z": [\n    1.0,\n    null,\n    null\n  ]\n}\n'
    )
    assert set(target.parent.iterdir()) == {target}


def test_concurrent_direct_writers_install_their_own_complete_payloads(tmp_path, monkeypatch):
    target, original = _existing_report(tmp_path)
    reports = {
        "first": {"writer": "first", "values": ["A"] * 1024},
        "last": {"writer": "last", "values": ["B"] * 1536},
    }
    ready = threading.Barrier(2)
    first_installed = threading.Event()
    writer = threading.local()
    staged = {}
    installed = {}
    sources = {}
    real_replace = Path.replace

    def ordered_replace(source, destination):
        if Path(destination) != target:
            return real_replace(source, destination)
        label = writer.label
        ready.wait(timeout=10)
        sources[label] = source
        staged[label] = json.loads(source.read_text(encoding="utf-8"))
        if label == "last":
            assert first_installed.wait(timeout=10), "First writer failed to replace"
        try:
            result = real_replace(source, destination)
            installed[label] = json.loads(target.read_text(encoding="utf-8"))
            return result
        finally:
            if label == "first":
                first_installed.set()

    def write(label):
        writer.label = label
        evaluation.write_json(target, reports[label])

    monkeypatch.setattr(Path, "replace", ordered_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(write, label) for label in reports]
        for result in pending:
            result.result(timeout=20)

    assert staged == reports
    assert installed == reports
    assert sources["first"] != sources["last"]
    assert all(source.parent == target.parent for source in sources.values())
    assert json.loads(target.read_text(encoding="utf-8")) == reports["last"]
    assert set(tmp_path.iterdir()) == set(original)
    for path, contents in original.items():
        if path != target:
            assert path.read_bytes() == contents


def test_serialization_failure_preserves_target_and_other_temporary_files(tmp_path):
    target, original = _existing_report(tmp_path)

    class Unserializable:
        def __repr__(self):
            raise ValueError("deliberate serialization failure")

    with pytest.raises(ValueError, match="deliberate serialization failure"):
        evaluation.write_json(target, {"bad": Unserializable()})
    _assert_files_unchanged(tmp_path, original)


def test_temporary_creation_failure_preserves_target_and_unrelated_files(tmp_path, monkeypatch):
    target, original = _existing_report(tmp_path)
    real_open = Path.open
    failure = OSError(errno.ENOSPC, "deliberate temporary creation failure")

    def failing_open(path, mode="r", *args, **kwargs):
        if "x" in mode:
            raise failure
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError) as caught:
        evaluation.write_json(target, {"generation": "new"})
    assert caught.value is failure
    _assert_files_unchanged(tmp_path, original)


@pytest.mark.parametrize("failure", [
    OSError(errno.ENOSPC, "deliberate partial write failure"),
    KeyboardInterrupt("deliberate interrupted write"),
], ids=["io-error", "keyboard-interrupt"])
def test_partial_write_closes_and_removes_only_owned_temporary(tmp_path, monkeypatch, failure):
    target, original = _existing_report(tmp_path)
    real_open = Path.open
    opened = []

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *exception):
            return self.stream.__exit__(*exception)

        def write(self, text):
            self.stream.write(text[:5])
            self.stream.flush()
            raise failure

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def failing_open(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if "x" in mode:
            opened.append(stream)
            return FailingStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(type(failure)) as caught:
        evaluation.write_json(target, {"generation": "new"})
    assert caught.value is failure
    assert opened and all(stream.closed for stream in opened)
    _assert_files_unchanged(tmp_path, original)


def test_exclusive_creation_collision_never_truncates_or_cleans_other_writer(tmp_path, monkeypatch):
    target, original = _existing_report(tmp_path)
    real_open = Path.open
    collided = []
    sentinel = b"a different writer exclusively created this candidate"

    def collision_once(path, mode="r", *args, **kwargs):
        if "x" in mode and not collided:
            with real_open(path, "xb") as other_writer:
                other_writer.write(sentinel)
            collided.append(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", collision_once)
    evaluation.write_json(target, {"generation": "new"})
    assert len(collided) == 1
    assert collided[0].read_bytes() == sentinel
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "new"}
    assert set(tmp_path.iterdir()) == set(original) | set(collided)
    for path, contents in original.items():
        if path != target:
            assert path.read_bytes() == contents


def test_repeated_creation_collisions_fail_without_removing_unowned_files(tmp_path, monkeypatch):
    target, original = _existing_report(tmp_path)
    real_open = Path.open
    collided = {}

    def always_collide(path, mode="r", *args, **kwargs):
        if "x" in mode and path not in collided:
            sentinel = f"unrelated writer {len(collided)}".encode()
            with real_open(path, "xb") as other_writer:
                other_writer.write(sentinel)
            collided[path] = sentinel
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", always_collide)
    with pytest.raises(FileExistsError):
        evaluation.write_json(target, {"generation": "new"})
    assert collided
    _assert_files_unchanged(tmp_path, original | collided)


@pytest.mark.parametrize("failure", [
    OSError(errno.EIO, "deliberate replacement failure"),
    KeyboardInterrupt("deliberate interrupted replacement"),
], ids=["io-error", "keyboard-interrupt"])
def test_replace_failure_preserves_prior_target_and_cleans_owned_temp(tmp_path, monkeypatch, failure):
    target, original = _existing_report(tmp_path)
    sources = []

    def failing_replace(source, destination):
        assert Path(destination) == target
        assert json.loads(source.read_text(encoding="utf-8")) == {"generation": "new"}
        assert target.read_bytes() == original[target]
        sources.append(source)
        raise failure

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(type(failure)) as caught:
        evaluation.write_json(target, {"generation": "new"})
    assert caught.value is failure
    assert len(sources) == 1
    _assert_files_unchanged(tmp_path, original)


@pytest.mark.parametrize("failures", [1, 5])
def test_permission_error_retries_keep_old_report_until_success(tmp_path, monkeypatch, failures):
    target, original = _existing_report(tmp_path)
    real_replace = Path.replace
    attempts = []
    sleeps = []

    def transient_replace(source, destination):
        assert target.read_bytes() == original[target]
        assert json.loads(source.read_text(encoding="utf-8")) == {"generation": "new"}
        attempts.append(source)
        if len(attempts) <= failures:
            raise PermissionError(errno.EACCES, "temporary sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", transient_replace)
    monkeypatch.setattr(evaluation.time, "sleep", sleeps.append)
    evaluation.write_json(target, {"generation": "new"})
    assert len(attempts) == failures + 1
    assert len(set(attempts)) == 1
    assert sleeps == [0.05 * 2**attempt for attempt in range(failures)]
    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "new"}
    assert set(tmp_path.iterdir()) == set(original)
    for path, contents in original.items():
        if path != target:
            assert path.read_bytes() == contents


def test_permission_error_exhaustion_preserves_target_and_removes_owned_temp(tmp_path, monkeypatch):
    target, original = _existing_report(tmp_path)
    attempts = []
    sleeps = []
    failure = PermissionError(errno.EACCES, "persistent sharing violation")

    def denied_replace(source, destination):
        assert target.read_bytes() == original[target]
        attempts.append(source)
        raise failure

    monkeypatch.setattr(Path, "replace", denied_replace)
    monkeypatch.setattr(evaluation.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        evaluation.write_json(target, {"generation": "new"})
    assert caught.value is failure
    assert len(attempts) == 6
    assert sleeps == [0.05, 0.1, 0.2, 0.4, 0.8]
    _assert_files_unchanged(tmp_path, original)


def test_cleanup_failure_does_not_mask_original_replacement_error(tmp_path, monkeypatch):
    target, original = _existing_report(tmp_path)
    real_unlink = Path.unlink
    sources = []
    failure = OSError(errno.EIO, "original replacement failure")

    def failing_replace(source, destination):
        sources.append(source)
        raise failure

    def denied_cleanup(path, *args, **kwargs):
        if path in sources:
            raise PermissionError(errno.EACCES, "temporary cleanup denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", failing_replace)
    monkeypatch.setattr(Path, "unlink", denied_cleanup)
    with pytest.raises(OSError) as caught:
        evaluation.write_json(target, {"generation": "new"})
    assert caught.value is failure
    assert len(sources) == 1
    assert json.loads(sources[0].read_text(encoding="utf-8")) == {"generation": "new"}
    assert set(tmp_path.iterdir()) == set(original) | set(sources)
    for path, contents in original.items():
        assert path.read_bytes() == contents


def test_staging_handle_is_closed_before_replace(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    real_open = Path.open
    real_replace = Path.replace
    streams = []

    def track_open(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if "x" in mode:
            streams.append(stream)
        return stream

    def check_replace(source, destination):
        assert streams and all(stream.closed for stream in streams)
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(Path, "replace", check_replace)
    evaluation.write_json(target, {"complete": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}


@pytest.mark.skipif(os.name == "nt", reason="POSIX creation-mode/umask contract")
def test_json_replacement_uses_ordinary_file_creation_permissions(tmp_path):
    reference = tmp_path / "ordinary.txt"
    reference.write_text("ordinary creation", encoding="utf-8")
    expected = stat.S_IMODE(reference.stat().st_mode)
    target = tmp_path / "report.json"
    evaluation.write_json(target, {"generation": 1})
    assert stat.S_IMODE(target.stat().st_mode) == expected
    evaluation.write_json(target, {"generation": 2})
    assert stat.S_IMODE(target.stat().st_mode) == expected


def test_preexisting_lock_marker_is_reusable_and_never_deleted(tmp_path):
    marker = tmp_path / ".existing.lock"
    marker.write_bytes(b"persistent marker")
    identity = marker.stat().st_ino
    for _ in range(2):
        with evaluation._file_lock(marker, "this checkpoint is busy"):
            with pytest.raises(RuntimeError, match="this checkpoint is busy"):
                with evaluation._file_lock(marker, "this checkpoint is busy"):
                    pytest.fail("A separate handle acquired an owned marker")
        assert marker.read_bytes() == b"persistent marker"
        assert marker.stat().st_ino == identity


@pytest.mark.parametrize("failure", [
    ValueError("owner failed"), KeyboardInterrupt("owner interrupted"),
], ids=["exception", "keyboard-interrupt"])
def test_checkpoint_ownership_released_after_baseexception(tmp_path, failure):
    target = tmp_path / "nested" / "report.json"
    with pytest.raises(type(failure)) as caught:
        with evaluation._checkpoint_file(target) as owned:
            assert owned == target
            raise failure
    assert caught.value is failure
    marker = target.parent / ".report.json.lock"
    assert marker.exists()
    with evaluation._checkpoint_file(target) as owned:
        assert owned == target
    assert marker.exists()
    assert not target.exists()


def test_same_path_thread_contends_while_other_checkpoint_is_independent(tmp_path):
    target = tmp_path / "report.json"
    independent = tmp_path / "other.json"
    acquired = threading.Event()
    release = threading.Event()

    def own_checkpoint():
        with evaluation._checkpoint_file(target) as owned:
            acquired.set()
            assert release.wait(timeout=10), "Main thread did not release the owner"
            return owned

    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(own_checkpoint)
        try:
            assert acquired.wait(timeout=10), "Owner did not acquire checkpoint"
            with pytest.raises(RuntimeError) as caught:
                with evaluation._checkpoint_file(target):
                    pytest.fail("Competing thread acquired the same checkpoint")
            assert str(target) in str(caught.value)
            with evaluation._checkpoint_file(independent) as owned:
                assert owned == independent
            assert not target.exists() and not independent.exists()
        finally:
            release.set()
        assert owner.result(timeout=10) == target
    with evaluation._checkpoint_file(target):
        pass
    assert {path.name for path in tmp_path.iterdir()} == {
        ".report.json.lock", ".other.json.lock",
    }


def test_relative_and_parent_dot_aliases_share_one_destination_lock(tmp_path, monkeypatch):
    parent = tmp_path / "checkpoints"
    (parent / "branch").mkdir(parents=True)
    target = parent / "report.json"
    monkeypatch.chdir(tmp_path)
    relative = Path("checkpoints") / "branch" / ".." / "report.json"
    with evaluation._checkpoint_file(relative) as owned:
        assert owned == target
        with pytest.raises(RuntimeError):
            with evaluation._checkpoint_file(target):
                pytest.fail("Parent dot alias acquired independent ownership")
    assert (parent / ".report.json.lock").exists()
    assert not target.exists()


def _directory_alias(alias, parent):
    try:
        alias.symlink_to(parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable on this validation host: {error}")


def test_parent_directory_aliases_share_one_destination_lock(tmp_path):
    parent = tmp_path / "real"
    parent.mkdir()
    alias = tmp_path / "alias"
    _directory_alias(alias, parent)
    target = parent / "report.json"
    with evaluation._checkpoint_file(alias / target.name) as owned:
        assert owned == target
        with pytest.raises(RuntimeError):
            with evaluation._checkpoint_file(target):
                pytest.fail("Parent directory alias acquired independent ownership")
    assert set(parent.iterdir()) == {parent / ".report.json.lock"}


def test_checkpoint_path_is_stable_after_working_directory_changes(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    with evaluation._checkpoint_file(Path("report.json")) as owned:
        monkeypatch.chdir(second)
        evaluation.write_json(owned, {"location": "first"})
    assert json.loads((first / "report.json").read_text()) == {"location": "first"}
    assert list(second.iterdir()) == []


def test_no_checkpoint_yields_none_without_filesystem_or_lock_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No-checkpoint route attempted filesystem or lock work")

    with monkeypatch.context() as guard:
        guard.setattr(evaluation, "_file_lock", forbidden)
        for method in ("mkdir", "open", "resolve", "absolute", "exists"):
            guard.setattr(Path, method, forbidden)
        with evaluation._checkpoint_file(None) as owned:
            assert owned is None


def test_unexpected_os_lock_failure_is_not_reported_as_competing_owner(tmp_path, monkeypatch):
    failure = OSError(errno.EBADF, "deliberate invalid lock handle")

    def fail_lock(*args):
        raise failure

    if os.name == "nt":
        import msvcrt
        monkeypatch.setattr(msvcrt, "locking", fail_lock)
    else:
        import fcntl
        monkeypatch.setattr(fcntl, "flock", fail_lock)
    with pytest.raises(OSError) as caught:
        with evaluation._file_lock(tmp_path / ".report.lock", "busy"):
            pytest.fail("Invalid lock operation entered its protected block")
    assert caught.value is failure


def test_lock_handle_is_not_inheritable_by_spawned_workers(tmp_path, monkeypatch):
    marker = tmp_path / ".report.lock"
    real_open = Path.open
    streams = []

    def track_open(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if path == marker:
            streams.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", track_open)
    with evaluation._file_lock(marker, "busy"):
        assert len(streams) == 1
        assert not os.get_inheritable(streams[0].fileno())
    assert streams[0].closed


def test_study_wrapper_preserves_existing_marker_and_shared_os_lock(tmp_path):
    marker = tmp_path / ".study.lock"
    marker.write_bytes(b"existing study marker")
    with study._checkpoint_lock(tmp_path):
        with pytest.raises(RuntimeError, match="study checkpoint directory"):
            with study._checkpoint_lock(tmp_path):
                pytest.fail("Study wrapper admitted a competing owner")
        with pytest.raises(RuntimeError, match="generic owner conflict"):
            with evaluation._file_lock(marker, "generic owner conflict"):
                pytest.fail("Study wrapper did not use the shared marker lock")
    with study._checkpoint_lock(tmp_path):
        pass
    assert marker.read_bytes() == b"existing study marker"
    assert set(tmp_path.iterdir()) == {marker}


def test_successful_replace_does_not_clean_a_reused_temporary_name(tmp_path, monkeypatch):
    """A completed replacement no longer owns its former staging pathname."""
    destination = tmp_path / "report.json"
    real_replace = Path.replace
    reused = []

    def replace_and_reuse(path, target):
        result = real_replace(path, target)
        path.write_text("another writer's temporary", encoding="utf-8")
        reused.append(path)
        return result

    monkeypatch.setattr(Path, "replace", replace_and_reuse)
    evaluation.write_json(destination, {"writer": "completed"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"writer": "completed"}
    assert len(reused) == 1
    assert reused[0].read_text(encoding="utf-8") == "another writer's temporary"

