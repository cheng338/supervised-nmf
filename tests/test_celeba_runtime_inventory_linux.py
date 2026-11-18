"""Linux maps deduplication preserves current path membership and errors."""
import builtins
import io
import os
import sys
import pytest
import threadpoolctl as tpc
from tests._celeba_recovery_test_support import inventory_runtime
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux maps discovery')

@pytest.fixture
def recovery(inventory_runtime):
    return inventory_runtime


def maps_text(paths):
    return "\n".join(f"00001000-00002000 r-xp 00000000 08:02 7 {path}" for path in paths)


def run_finder(function, text, exists, monkeypatch):
    observed = []
    checks = []
    opens = []

    class Collector:
        def _make_controller_from_path(self, path):
            observed.append(path)

    original_open = builtins.open

    def maps_open(path, *args, **kwargs):
        if path == "/proc/self/maps":
            opens.append((args, kwargs))
            return io.StringIO(text)
        return original_open(path, *args, **kwargs)

    def checked_exists(path):
        checks.append(path)
        return exists(path)

    with monkeypatch.context() as change:
        change.setattr(builtins, "open", maps_open)
        change.setattr(os.path, "exists", checked_exists)
        function(Collector())
    assert opens == [((), {})]
    return observed, checks


def test_stable_duplicate_missing_and_nonlibrary_records(recovery, monkeypatch):
    paths = ["/a/libone.so", "/a/libone.so", "/a/libtwo.so.2", "/a/libone.so",
             "/gone/libmissing.so", "/gone/libmissing.so", "/a/plain.txt",
             "[heap]", "/old/libremoved.so (deleted)", "/a/libtwo.so.2"]
    existing = {"/a/libone.so", "/a/libtwo.so.2", "/a/plain.txt"}
    original, full_checks = run_finder(tpc.ThreadpoolController._find_libraries_with_linux,
                                     maps_text(paths), existing.__contains__, monkeypatch)
    optimized, short_checks = run_finder(recovery._inventory_linux_dedup,
                                        maps_text(paths), existing.__contains__, monkeypatch)
    assert optimized == original
    assert set(optimized) == {"/a/libone.so", "/a/libtwo.so.2"}
    assert len(full_checks) == 8 and len(short_checks) == 5
    assert short_checks == ["/a/libone.so", "/a/libtwo.so.2", "/gone/libmissing.so",
                            "/gone/libmissing.so", "/old/libremoved.so (deleted)"]


def test_new_and_removed_paths_are_reconsidered_each_scan(recovery, monkeypatch):
    first_paths = ["/a/libone.so", "/a/libone.so", "/a/libtwo.so"]
    next_paths = ["/a/libtwo.so", "/a/libnew.so", "/a/libnew.so"]
    for paths in (first_paths, next_paths, first_paths):
        original, _ = run_finder(tpc.ThreadpoolController._find_libraries_with_linux,
                                maps_text(paths), lambda path: True, monkeypatch)
        current, _ = run_finder(recovery._inventory_linux_dedup,
                               maps_text(paths), lambda path: True, monkeypatch)
        assert current == original and set(current) == set(paths)


@pytest.mark.parametrize("error_type", (PermissionError, OSError, UnicodeDecodeError))
def test_read_failures_propagate_without_partial_inventory(recovery, monkeypatch, error_type):
    error = (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
             if error_type is UnicodeDecodeError else error_type("maps unavailable"))
    original_open = builtins.open

    def fail(path, *args, **kwargs):
        if path == "/proc/self/maps":
            raise error
        return original_open(path, *args, **kwargs)

    class Collector:
        def _make_controller_from_path(self, path):
            raise AssertionError("No path may be published after failed maps read")

    with monkeypatch.context() as change:
        change.setattr(builtins, "open", fail)
        for function in (tpc.ThreadpoolController._find_libraries_with_linux,
                         recovery._inventory_linux_dedup):
            with pytest.raises(error_type) as caught:
                function(Collector())
            assert caught.value is error


def test_path_check_error_is_not_converted_to_missing(recovery, monkeypatch):
    def fails(path):
        raise RuntimeError("path inspection failed")

    for function in (tpc.ThreadpoolController._find_libraries_with_linux,
                     recovery._inventory_linux_dedup):
        with pytest.raises(RuntimeError, match="path inspection failed"):
            run_finder(function, maps_text(["/a/libone.so"]), fails, monkeypatch)


@pytest.mark.parametrize("responses", ((False, True), (True, False), (False, False)))
def test_midscan_file_changes_preserve_original_membership(recovery, monkeypatch, responses):
    path = "/a/libone.so"

    def one_run(function):
        sequence = iter(responses)
        return run_finder(function, maps_text([path, path]),
                          lambda value: next(sequence), monkeypatch)

    original, full_checks = one_run(tpc.ThreadpoolController._find_libraries_with_linux)
    dedup, short_checks = one_run(recovery._inventory_linux_dedup)
    assert original == dedup == ([path] if any(responses) else [])
    assert full_checks == [path, path]
    assert short_checks == ([path] if responses[0] else [path, path])
    # The next complete inventory samples the path afresh and observes it.
    next_scan, _ = run_finder(recovery._inventory_linux_dedup,
                              maps_text([path, path]), lambda value: True, monkeypatch)
    assert next_scan == [path]
