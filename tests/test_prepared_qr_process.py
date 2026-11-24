"""Prepared QR process reuse and failure boundaries through real loky workers.

Controlled fitted bases isolate projection from training; separate public-fit
tests cover estimator construction. Every test runs on unsupported providers too,
where ordinary numerical results are required and optional publication is absent.
"""
from contextlib import contextmanager
import json
import os
import pickle
import sys
import tempfile
import weakref

import numpy as np
import pytest
from joblib import parallel_config
from joblib.executor import MemmappingExecutor
from joblib.externals.loky import get_reusable_executor

from supervised_nmf import _projection
from tests._prepared_qr_support import (
    assert_coordinates, assert_worker_binding, fixed_model, observed_calls,
    fail_large_qr_payload_view, parent_inventory_under_limit, problem, qr_state, reference,
)


@pytest.fixture
def process_resources(qr_state, tmp_path):
    """Keep real two-worker pools and temporary mappings local to each test."""
    # Keep Joblib's own long mapping names away from pytest's test-name path.
    with tempfile.TemporaryDirectory(prefix="q", dir=tmp_path.parent) as folder, \
            parallel_config(temp_folder=folder):
        try:
            yield qr_state
        finally:
            get_reusable_executor(max_workers=2, reuse=True).shutdown(
                wait=True, kill_workers=True)


def entry(owner):
    return _projection._qr_registry_get(owner)


def checked(model, rows, known):
    owner_before = model.components_.copy()
    rows_before = rows.copy()
    actual = model.transform(rows)
    assert_coordinates(actual, known)
    np.testing.assert_array_equal(model.components_, owner_before)
    np.testing.assert_array_equal(rows, rows_before)
    assert not np.shares_memory(actual, rows)
    assert not np.shares_memory(actual, model.components_)
    return actual


@contextmanager
def executor_calls(monkeypatch):
    """Record whether each factory call returns the same live executor."""
    factory = MemmappingExecutor.get_memmapping_executor.__func__
    first = None
    records = []

    def observe(cls, *args, **kwargs):
        nonlocal first
        executor = factory(cls, *args, **kwargs)
        if first is None:
            first = weakref.ref(executor)
        records.append(first() is executor)
        return executor

    with monkeypatch.context() as patch:
        patch.setattr(MemmappingExecutor, "get_memmapping_executor", classmethod(observe))
        yield records


def test_cold_one_wave_then_multiwave_and_warm_one_wave(
        process_resources, monkeypatch, tmp_path):
    owner, rows, known = problem(rows=768)
    model = fixed_model(owner, workers=2)
    receipt_directory = str(tmp_path / "children")
    (tmp_path / "children").mkdir()
    # Exercise a different ambient limit while taking the parent snapshot.
    # The payload below must still match the snapshot at the model's limit.
    native = _projection._controller
    with native.limit(limits=2):
        ambient = native.info()
        parent_inventory = parent_inventory_under_limit(model)
        assert native.info() == ambient
    def observed_shared_block(*args, **kwargs):
        # This local by-value wrapper imports the untouched function in the
        # actual child. It does not replace the required _project_block anchor.
        import json
        import os
        import tempfile
        from supervised_nmf import _projection as p
        from tests._prepared_qr_support import observed_worker_binding
        with observed_worker_binding(args[-1], parent_inventory) as record:
            answer = p._project_shared_block(*args, **kwargs)
        fd, path = tempfile.mkstemp(prefix="block-", suffix=".json",
                                    dir=receipt_directory)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(rows=len(args[1]), pid=os.getpid(), **record), stream)
        return answer
    monkeypatch.setattr(_projection, "_project_shared_block", observed_shared_block)
    assert_coordinates(reference(owner, rows[:512]), known[:512])
    with observed_calls() as cold:
        checked(model, rows[:512], known[:512])
    assert cold["qr"] == cold["publish"] == 0
    assert entry(owner) is None
    assert list((tmp_path / "children").glob("*.json")) == []
    with observed_calls() as multi:
        checked(model, rows, known)
    assert multi["qr"] == multi["publish"] == int(process_resources)
    prepared = entry(owner)
    assert (prepared is not None) is process_resources
    assert len(list((tmp_path / "children").glob("*.json"))) == 3 * int(process_resources)
    with observed_calls() as warm:
        checked(model, rows[:512], known[:512])
    assert warm["qr"] == warm["publish"] == 0
    assert entry(owner) is prepared
    receipts = [json.loads(path.read_text())
                for path in (tmp_path / "children").glob("*.json")]
    assert len(receipts) == 5 * int(process_resources)
    for row in receipts:
        assert row["rows"] == 256 and row["pid"] != os.getpid()
        assert_worker_binding(row)


@pytest.mark.parametrize("qr_state", ["cold"], indirect=True)
def test_process_only_calls_do_not_warm_a_declined_parent(process_resources):
    owner, rows, known = problem(rows=768)
    model = fixed_model(owner, workers=2)
    first_publish = process_resources and sys.platform == "linux"
    for step in range(2):
        with observed_calls() as calls:
            checked(model, rows, known)
        assert calls["qr"] == calls["publish"] == int(step == 0 and first_publish)
        assert (entry(owner) is not None) is first_publish
        assert (_projection._controller is not None) is first_publish
    # A small ordinary serial call initializes the parent on every platform.
    checked(model, rows[:1], known[:1])
    with observed_calls() as calls:
        checked(model, rows, known)
    assert calls["qr"] == calls["publish"] == int(process_resources and not first_publish)
    assert (entry(owner) is not None) is process_resources


def test_mixed_models_keep_original_mapping_threshold_and_executor(
        process_resources, monkeypatch):
    first = problem(features=1024, rows=768)
    second = problem(features=2048, rows=768)
    small = problem(features=256, rank=16, rows=768)
    models = [fixed_model(item[0], workers=2) for item in (first, second, small)]
    original = _projection.Parallel
    dispatches = []
    def parallel(*args, **kwargs):
        dispatches.append(dict(kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(_projection, "Parallel", parallel)
    with executor_calls(monkeypatch) as executions:
        for index in (0, 1, 2, 0):
            owner, rows, known = (first, second, small)[index]
            checked(models[index], rows, known)
    assert len(dispatches) == len(executions) == 4
    assert all(row["n_jobs"] == 2 and row["max_nbytes"] == "1M"
               and row["mmap_mode"] == "r" for row in dispatches)
    assert all(executions)
    assert (entry(first[0]) is not None) is process_resources
    assert (entry(second[0]) is not None) is process_resources
    assert entry(small[0]) is None


def test_optional_payload_storage_failure_falls_back_and_can_retry(
        process_resources, monkeypatch):
    # At rank 32 this exceeds 1 MiB and exercises the retained array-view path.
    owner, rows, known = problem(features=4097, rows=768)
    model = fixed_model(owner, workers=2)
    with fail_large_qr_payload_view(monkeypatch) as failures:
        checked(model, rows, known)
    assert failures == ([owner.nbytes] if process_resources else [])
    assert entry(owner) is None
    checked(model, rows, known)
    assert (entry(owner) is not None) is process_resources


def test_optional_publication_failure_does_not_fail_projection(
        process_resources, monkeypatch):
    owner, rows, known = problem(rows=768)
    model = fixed_model(owner, workers=2)
    failures = []
    def publish(*args):
        failures.append(True)
        raise MemoryError("optional prepared publication")
    with monkeypatch.context() as patch:
        patch.setattr(_projection, "_qr_registry_publish", publish)
        checked(model, rows, known)
    assert len(failures) == int(process_resources)
    assert entry(owner) is None
    checked(model, rows, known)
    assert (entry(owner) is not None) is process_resources


def test_real_worker_failure_propagates_without_publishing(
        process_resources, monkeypatch):
    owner, rows, known = problem(rows=768)
    model = fixed_model(owner, workers=2)
    original = _projection.Parallel
    def worker_failure():
        raise RuntimeError("required worker failure")
    def parallel(*args, **kwargs):
        backend = original(*args, **kwargs)
        def submit(tasks):
            tasks = list(tasks)
            tasks[0] = (worker_failure, (), {})
            return backend(tasks)
        return submit
    with monkeypatch.context() as patch:
        patch.setattr(_projection, "Parallel", parallel)
        with pytest.raises(RuntimeError, match="required worker failure"):
            model.transform(rows)
    assert entry(owner) is None
    checked(model, rows, known)
    assert (entry(owner) is not None) is process_resources


def test_assembly_failure_propagates_without_publishing(
        process_resources, monkeypatch):
    owner, rows, known = problem(rows=768)
    model = fixed_model(owner, workers=2)
    original = np.vstack
    projection_code = _projection.project_rows.__code__
    failures = []
    def vstack(*args, **kwargs):
        if sys._getframe(1).f_code is projection_code:
            failures.append(True)
            raise RuntimeError("required projection assembly")
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(np, "vstack", vstack)
        with pytest.raises(RuntimeError, match="required projection assembly"):
            model.transform(rows)
    assert failures == [True]
    assert entry(owner) is None
    checked(model, rows, known)
    assert (entry(owner) is not None) is process_resources


def test_real_mapping_failure_is_not_silently_retried(
        process_resources, tmp_path):
    owner, rows, known = problem(rows=768)
    model = fixed_model(owner, workers=2)
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"not a directory")
    # Each 256-row target block exceeds1MiB. The real Joblib mapping path
    # therefore needs a directory, on admitted and ordinary provider routes.
    with parallel_config(temp_folder=str(blocked)):
        with pytest.raises((OSError, RuntimeError, pickle.PicklingError)):
            model.transform(rows)
    assert blocked.read_bytes() == b"not a directory"
    assert entry(owner) is None
    checked(model, rows, known)
    assert (entry(owner) is not None) is process_resources
