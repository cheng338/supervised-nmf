"""Small deterministic fixtures for prepared-QR behavioral regressions."""
from collections import Counter
from contextlib import contextmanager
import inspect
import os
import sys
import threading

import numpy as np
import pytest
import scipy
import threadpoolctl
from threadpoolctl import ThreadpoolController, threadpool_limits

from supervised_nmf import FeatureSPA, _nnls, _projection


ORDINARY = dict(divide="warn", over="warn", under="ignore", invalid="warn")


@contextmanager
def fail_large_qr_payload_view(monkeypatch):
    """Fail a zero-copy view of a large factor during the real payload export."""
    original_export = _nnls._QRPrepared._transport_payload
    original_view = np.frombuffer
    failures = []

    def export(entry):
        large = [buffer for buffer in entry[1:4] if len(buffer) > 1024 ** 2]
        def view(buffer, *args, **kwargs):
            if any(buffer is original for original in large):
                failures.append(len(buffer))
                raise MemoryError("optional large payload view")
            return original_view(buffer, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(np, "frombuffer", view)
            return original_export(entry)

    with monkeypatch.context() as patch:
        patch.setattr(_nnls._QRPrepared, "_transport_payload", export)
        yield failures


def expected_inventory(native=None):
    """Read the installed build and stock controller independently of QR admission."""
    if (np.__version__, scipy.__version__, threadpoolctl.__version__) != (
            "2.5.4", "1.18.1", "3.7.0"):
        return None
    for module in (np, scipy):
        config = getattr(module.__config__, "CONFIG", {})
        deps = config.get("Build Dependencies", {})
        for kind in ("blas", "lapack"):
            name = deps.get(kind, {}).get("name")
            if not isinstance(name, str) or not name or any(
                    other in name.lower() for other in ("openblas", "blis", "aocl", "accelerate")):
                return None
    names = ("OpenBLASController", "BLISController", "MKLController",
             "OpenMPController", "FlexiBLASController")
    classes = [getattr(threadpoolctl, name, None) for name in names]
    registry = getattr(threadpoolctl, "_ALL_CONTROLLERS", None)
    if not isinstance(registry, list) or len(registry) != len(classes):
        return None
    if any(actual is not required for actual, required in zip(registry, classes)):
        return None
    if native is None:
        native = ThreadpoolController()
    if type(native) is not ThreadpoolController:
        return None
    libraries = native.lib_controllers
    if (not any(type(lib) is classes[2] for lib in libraries)
            or any(type(lib) not in (classes[2], classes[3]) for lib in libraries)
            or {"libomp", "libiomp"}.issubset({lib.prefix for lib in libraries})):
        return None
    rows = native.info()
    for row in rows:
        if (type(row.get("num_threads")) is not int or row["num_threads"] < 1
                or any(type(key) is not str for key in row)
                or any(type(v) not in (str, int, bool, type(None)) for v in row.values())):
            return None
    # Multiset equality preserves duplicates without copying the production ordering.
    return Counter(tuple(sorted(row.items())) for row in rows)


def expected_reuse():
    return expected_inventory() is not None


def parent_inventory_under_limit(model):
    """Observe the controller that prepares the payload at the model's thread limit."""
    native = _projection._controller
    assert type(native) is ThreadpoolController and _projection._controller_pid == os.getpid()
    with native.limit(limits=model.projection_native_threads):
        return expected_inventory(native)


@pytest.fixture
def qr_state(request):
    """Isolate transient caches without changing any numerical callable."""
    names = ("_controller", "_controller_pid", "_qr_registry", "_qr_registry_pid",
             "_qr_registry_lock", "_qr_mkl_negative")
    saved = {name: getattr(_projection, name) for name in names}
    callback = np.geterrcall()
    try:
        _projection._controller = None
        _projection._controller_pid = None
        _projection._qr_registry = {}
        _projection._qr_registry_pid = os.getpid()
        _projection._qr_registry_lock = threading.RLock()
        _projection._qr_mkl_negative = None
        np.seterrcall(None)
        with threadpool_limits(limits=1), np.errstate(**ORDINARY):
            mode = getattr(request, "param", "warm")
            assert mode in ("cold", "warm")
            if mode == "warm":
                # Use the required stock controller. Registry and basis stay cold.
                _projection._native_controller()
            yield expected_inventory(_projection._controller) is not None
    finally:
        np.seterrcall(callback)
        for name, value in saved.items():
            setattr(_projection, name, value)


def problem(*, features=1024, rank=32, rows=32, dtype=np.float64):
    """Disjoint-support columns give an exact, nonnegative coefficient oracle."""
    design = np.zeros((features, rank), dtype=dtype)
    design[np.arange(features), np.arange(features) % rank] = 1
    known = ((np.arange(rows * rank).reshape(rows, rank) % 17) + 1) / 16
    query = np.ascontiguousarray(known @ design.T, dtype=dtype)
    owner = np.array(design.T, order="C", copy=True)
    return owner, query, known


def fixed_model(owner, *, workers=1, batch=256):
    """A controlled fitted basis for projection tests; public fit has own tests."""
    model = FeatureSPA(owner.shape[0], projection_solver="qr", projection_workers=workers,
                       projection_batch_size=batch, projection_native_threads=1,
                       nnls_max_iter=10000)
    model.components_ = owner
    model.n_features_in_ = owner.shape[1]
    model.n_components_ = owner.shape[0]
    return model


def reference(owner, query, *, batch=256, cap=10000, solver="qr"):
    return _nnls.project_nnls(np.ascontiguousarray(owner.T), query, solver=solver,
                              batch_size=batch, maxiter=cap)


@contextmanager
def observed_worker_binding(payload, parent_inventory):
    """Compare the actual bind and QR calls against each worker's live inventory."""
    assert parent_inventory is not None and Counter(payload[-1]) == parent_inventory
    qr = inspect.unwrap(_nnls._QR_ORIGINAL_QR).__code__
    bind = _projection._qr_call_bind.__code__
    record = dict(qr_calls=0, bindings=[], expected=[],
                  parent_pools=[dict(row) for row in parent_inventory.elements()])
    def observe(frame, event, value):
        if frame.f_code is qr and event == "call":
            record["qr_calls"] += 1
        elif frame.f_code is bind:
            if event == "call":
                # A DLL loaded after controller construction need not appear
                # in its inventory. Read that actual stock controller under its
                # active limit without calling a candidate gate or binding.
                assert _projection._controller_pid == os.getpid()
                assert type(_projection._controller) is ThreadpoolController
                worker = expected_inventory(_projection._controller)
                record["worker_pools"] = None if worker is None else [dict(row) for row in worker.elements()]
                record["expected"].append(worker == parent_inventory
                                          and np.geterr() == ORDINARY and np.geterrcall() is None)
            elif event == "return":
                record["bindings"].append(type(value) is _nnls._QRPrepared)
    previous = sys.getprofile()
    assert previous is None
    sys.setprofile(observe)
    try:
        yield record
    finally:
        sys.setprofile(previous)


def assert_worker_binding(record):
    assert len(record["expected"]) == 1
    assert record["bindings"] == record["expected"]
    assert record["qr_calls"] == int(not record["expected"][0])


@contextmanager
def observed_calls():
    """Count original code entries without replacing the admission callables."""
    targets = {
        "qr": inspect.unwrap(_nnls._QR_ORIGINAL_QR).__code__,
        "condition": inspect.unwrap(_nnls._QR_ORIGINAL_COND).__code__,
        "provider": _projection._qr_mkl_controller.__code__,
        "context": _projection._qr_native_context.__code__,
        "publish": _projection._qr_registry_publish.__code__,
    }
    counts = dict.fromkeys(targets, 0)
    previous = sys.getprofile()
    def observe(frame, event, argument):
        if event == "call":
            for name, code in targets.items():
                if frame.f_code is code:
                    counts[name] += 1
    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def assert_coordinates(actual, expected):
    assert np.isfinite(actual).all() and (actual >= 0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
