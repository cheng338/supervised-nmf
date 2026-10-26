"""Execution controls for fixed-basis SPA projection.

Process pools belong to joblib's reusable loky backend, never to a fitted
estimator. Models therefore remain cloneable/pickleable and refits do not
retain a pool or stale numerical state. No threading backend is used.
"""

from numbers import Integral
import os

from joblib import Parallel, delayed, parallel_config
import numpy as np
from threadpoolctl import ThreadpoolController

from ._nnls import project_nnls


_controller = None
_controller_pid = None


def _native_controller():
    # NumPy/SciPy are imported above before this lazy controller is created.
    # Reuse discovery of their numerical libraries: scanning loaded Windows
    # DLLs on every single-image query costs more than its NNLS solve. Forked
    # children get their own controller; no controller is stored in a model.
    global _controller, _controller_pid
    if _controller is None or _controller_pid != os.getpid():
        _controller = ThreadpoolController()
        _controller_pid = os.getpid()
    return _controller


def validate_projection_parameters(projection_solver, projection_workers,
                                   projection_batch_size, projection_native_threads):
    if (not isinstance(projection_solver, str)
            or projection_solver not in {"auto", "scipy", "batched", "qr"}):
        raise ValueError("projection_solver must be 'auto', 'scipy', 'batched', or 'qr'.")
    for name, value in (("projection_workers", projection_workers),
                        ("projection_batch_size", projection_batch_size),
                        ("projection_native_threads", projection_native_threads)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _project_block(design, rows, solver, maxiter, batch_size, native_threads):
    # Each process owns its thread controller; callers must not wrap this in
    # a threading backend with overlapping process-wide controllers.
    with _native_controller().limit(limits=native_threads):
        return project_nnls(design, rows, solver=solver, maxiter=maxiter,
                            batch_size=batch_size)


def project_rows(design, rows, *, solver, workers, batch_size, native_threads, maxiter):
    """Return ordered coordinates, using processes only for large batches.

Repeated calls with compatible controls reuse loky workers automatically.
Keeping pool objects outside model state also permits ordinary joblib/pickle
serialization. A first call still pays process startup cost.
"""
    validate_projection_parameters(solver, workers, batch_size, native_threads)
    if workers == 1 or len(rows) < max(512, 2 * batch_size):
        return _project_block(design, rows, solver, maxiter, batch_size, native_threads)
    spans = range(0, len(rows), batch_size)
    count = min(workers, (len(rows) + batch_size - 1) // batch_size)
    with parallel_config(backend="loky", inner_max_num_threads=native_threads):
        chunks = Parallel(n_jobs=count, batch_size=1, max_nbytes="1M",
                          mmap_mode="r", pre_dispatch="2*n_jobs")(
            delayed(_project_block)(design, rows[start:start + batch_size],
                                    solver, maxiter, batch_size, native_threads)
            for start in spans
        )
    return np.vstack(chunks)
