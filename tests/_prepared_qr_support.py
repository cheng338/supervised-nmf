"""Small deterministic fixtures for prepared-QR behavioral regressions."""
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


def expected_reuse():
    """Independent runtime expectation, never inferred from candidate success.

    The two approved validation hosts exercise actual MKL admission and actual
    AOCL decline. Other supported installations still run all semantic checks.
    This predicate does not manufacture a provider or skip a test.
    """
    if (np.__version__, scipy.__version__, threadpoolctl.__version__) != (
            "2.5.4", "1.18.1", "3.7.0"):
        return False
    names = ("OpenBLASController", "BLISController", "MKLController",
             "OpenMPController", "FlexiBLASController")
    classes = [getattr(threadpoolctl, name, None) for name in names]
    registry = getattr(threadpoolctl, "_ALL_CONTROLLERS", None)
    if not isinstance(registry, list) or len(registry) != len(classes):
        return False
    if any(actual is not required for actual, required in zip(registry, classes)):
        return False
    libraries = ThreadpoolController().lib_controllers
    return (sys.platform == "linux" and any(type(lib) is classes[2] for lib in libraries)
            and all(type(lib) in (classes[2], classes[3]) for lib in libraries)
            and not {"libomp", "libiomp"}.issubset({lib.prefix for lib in libraries}))


@pytest.fixture
def qr_state():
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
            yield expected_reuse()
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
