"""Execution controls for fixed-basis SPA projection.

Process pools belong to joblib's reusable loky backend, never to a fitted
estimator. Models therefore remain cloneable/pickleable and refits do not
retain a pool or stale numerical state. No threading backend is used.
"""

from numbers import Integral
import os
import threading
import weakref
import sys as _sys
from types import ModuleType as _ModuleType

from joblib import Parallel, delayed, parallel_config
import numpy as np
import scipy as _scipy
import threadpoolctl as _threadpoolctl
from threadpoolctl import ThreadpoolController

from ._nnls import project_nnls, _QRPrepared, _QRFitCapture, _qr_context


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


_QR_CONTROLLER = ThreadpoolController
_QR_NATIVE_CONTROLLER = _native_controller


def _qr_runtime_versions():
    for module, expected in ((np, "2.5.4"), (_scipy, "1.18.1"), (_threadpoolctl, "3.7.0")):
        if type(module) is not _ModuleType:
            return False
        version = vars(module).get("__version__")
        if type(version) is not str or version != expected:
            return False
    return True


def _qr_build_allows_mkl():
    # Generic conda "blas"/"lapack" names are not positive provider evidence.
    # Refuse explicitly named incompatible builds before optional discovery;
    # the actual loaded-controller proof remains mandatory for every success.
    for module in (np, _scipy):
        config_module = vars(module).get("__config__")
        if type(config_module) is not _ModuleType:
            return False
        config = vars(config_module).get("CONFIG")
        if type(config) is not dict or any(type(key) is not str for key in config):
            return False
        dependencies = config.get("Build Dependencies")
        if type(dependencies) is not dict or any(type(key) is not str for key in dependencies):
            return False
        for kind in ("blas", "lapack"):
            dependency = dependencies.get(kind)
            if type(dependency) is not dict or any(type(key) is not str for key in dependency):
                return False
            name = dependency.get("name")
            if type(name) is not str or not name:
                return False
            if any(token in name.lower() for token in ("openblas", "blis", "aocl", "accelerate")):
                return False
    return True


def _qr_raw_class_attribute(cls, name):
    """Read a known class descriptor without calling a replacement property."""
    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return None


def _qr_bind_mkl_classes():
    # Optional acceleration must not make an older supported runtime fail import.
    if not _qr_runtime_versions():
        return None
    try:
        namespace = vars(_threadpoolctl)
        classes = tuple(namespace.get(name) for name in
                        ("OpenBLASController", "BLISController", "MKLController",
                         "OpenMPController", "FlexiBLASController"))
        if any(not isinstance(cls, type) for cls in classes):
            return None
        controller_methods = tuple(_qr_raw_class_attribute(_QR_CONTROLLER, name)
                                   for name in ("__init__", "info", "limit"))
        methods = tuple((cls,) + tuple(_qr_raw_class_attribute(cls, name) for name in
                        ("info", "get_num_threads", "set_num_threads", "num_threads"))
                        for cls in (classes[2], classes[3]))
        if (any(value is None for value in controller_methods)
                or any(value is None for entry in methods for value in entry)):
            return None
        cold_methods = tuple((name, _qr_raw_class_attribute(_QR_CONTROLLER, name)) for name in
                             ("__new__", "_from_controllers", "_find_libraries_with_linux",
                              "_make_controller_from_path", "_check_prefix",
                              "__getattribute__", "__getattr__", "__setattr__"))
        if (any(value is None for name, value in cold_methods if name != "__getattr__")
                or type(cold_methods[1][1]) is not classmethod):
            return None
        # Snapshot raw class namespaces, including inherited constructors and
        # attribute hooks. This also detects a newly installed property setter
        # on a constructor-assigned field; no descriptor is invoked by the check.
        cold_classes = []
        for cls in (_QR_CONTROLLER,) + classes:
            for base in type.__getattribute__(cls, "__mro__"):
                if base is not object and not any(base is old for old in cold_classes):
                    cold_classes.append(base)
        cold_libraries = tuple((cls, tuple(type.__getattribute__(cls, "__dict__").items()))
                               for cls in cold_classes)
        return controller_methods, methods, classes, cold_methods, cold_libraries
    except Exception:
        return None


_QR_MKL_BINDING = _qr_bind_mkl_classes()
if _QR_MKL_BINDING is None:
    _QR_CONTROLLER_INIT = _QR_CONTROLLER_INFO = _QR_CONTROLLER_LIMIT = None
    _QR_LIBRARY_METHODS = _QR_DISCOVERY_CLASSES = _QR_COLD_METHODS = _QR_COLD_LIBRARIES = ()
else:
    (_QR_CONTROLLER_INIT, _QR_CONTROLLER_INFO, _QR_CONTROLLER_LIMIT), _QR_LIBRARY_METHODS, _QR_DISCOVERY_CLASSES, _QR_COLD_METHODS, _QR_COLD_LIBRARIES = _QR_MKL_BINDING

_qr_mkl_negative = None


def _qr_mkl_controller():
    """Fail closed before optional work on unsupported provider compositions."""
    global _controller, _controller_pid, _qr_mkl_negative
    previous_negative = _qr_mkl_negative
    _qr_mkl_negative = None
    try:
        if _QR_MKL_BINDING is None or not _qr_runtime_versions() or not _qr_build_allows_mkl():
            return None
        registry = vars(_threadpoolctl).get("_ALL_CONTROLLERS")
        if (type(registry) is not list or len(registry) != len(_QR_DISCOVERY_CLASSES)
                or any(actual is not expected for actual, expected in zip(registry, _QR_DISCOVERY_CLASSES))):
            return None
        if (_native_controller is not _QR_NATIVE_CONTROLLER
                or ThreadpoolController is not _QR_CONTROLLER
                or _qr_raw_class_attribute(_QR_CONTROLLER, "__init__") is not _QR_CONTROLLER_INIT
                or _qr_raw_class_attribute(_QR_CONTROLLER, "info") is not _QR_CONTROLLER_INFO
                or _qr_raw_class_attribute(_QR_CONTROLLER, "limit") is not _QR_CONTROLLER_LIMIT
                or any(_qr_raw_class_attribute(_QR_CONTROLLER, name) is not value
                       for name, value in _QR_COLD_METHODS)):
            return None
        pid = os.getpid()
        fresh = _controller is None or _controller_pid != pid
        current = None if fresh else _controller
        if current is not None and type(current) is not _QR_CONTROLLER:
            return None
        current_libraries = None if current is None else vars(current).get("lib_controllers")
        if current is not None and type(current_libraries) is not list:
            return None
        fingerprint = (() if current_libraries is None else
                       (id(current_libraries), tuple((id(lib), id(type(lib))) for lib in current_libraries)))
        key = (pid, id(_controller), _controller_pid, id(_QR_MKL_BINDING), id(registry),
               tuple(id(cls) for cls in registry), fingerprint,
               tuple(vars(module).get("__version__") for module in (np, _scipy, _threadpoolctl)))
        # A memo can only select the original route. It never authorizes factors
        # or skips any fresh positive context/basis proof. No owner is retained.
        if previous_negative == key:
            _qr_mkl_negative = previous_negative
            return None
        if fresh:
            if _sys.platform != "linux":
                return None
            # Discovery may construct a stock non-MKL controller before the
            # composition is rejected. Guard every directly reached constructor,
            # helper and prefix/symbol descriptor before invoking that path.
            for cls, descriptors in _QR_COLD_LIBRARIES:
                namespace = type.__getattribute__(cls, "__dict__")
                if (len(namespace) != len(descriptors)
                        or any(namespace.get(name) is not expected for name, expected in descriptors)):
                    return None
            # The qualified Linux /proc discovery method and stock library
            # constructors do not emit Python warnings. Build locally, then
            # decline the separate incompatible-OpenMP warning condition before
            # publishing. No warning filters or global showwarning are changed.
            creator = _QR_COLD_METHODS[1][1]
            controller = creator.__func__(_QR_CONTROLLER, [])
            _QR_COLD_METHODS[2][1](controller)
        else:
            controller = current
        if type(controller) is not _QR_CONTROLLER:
            return None
        state = vars(controller)
        libraries = state.get("lib_controllers")
        if "info" in state or "limit" in state or type(libraries) is not list:
            return None
        has_mkl = False
        unsupported = False
        prefixes = []
        for library in libraries:
            if not any(type(library) is cls for cls in _QR_DISCOVERY_CLASSES):
                return None
            library_state = vars(library)
            if any(type(name) is not str for name in library_state):
                return None
            prefix = library_state.get("prefix")
            if type(prefix) is not str:
                return None
            prefixes.append(prefix)
            entry = next((row for row in _QR_LIBRARY_METHODS if type(library) is row[0]), None)
            if entry is None:
                unsupported = True
                continue
            cls, info, threads, setter, property_ = entry
            if (any(_qr_raw_class_attribute(cls, name) is not expected for name, expected in
                    (("info", info), ("get_num_threads", threads),
                     ("set_num_threads", setter), ("num_threads", property_)))
                    or any(name in library_state for name in
                           ("info", "get_num_threads", "set_num_threads", "num_threads"))):
                return None
            has_mkl = has_mkl or cls is _QR_LIBRARY_METHODS[0][0]
        if "libomp" in prefixes and "libiomp" in prefixes:
            return None
        if unsupported or not has_mkl:
            # Only complete silent unsupported composition is memoized.
            # Exceptions, incompatible-warning state and failed proof are not.
            _qr_mkl_negative = key
            return None
        if fresh:
            _controller, _controller_pid = controller, pid
        return controller
    except Exception:
        return None


def _qr_native_context():
    """Read fresh qualified MKL/OpenMP state inside the numerical limit."""
    controller = _qr_mkl_controller()
    if controller is None:
        return None
    rows = controller.info()
    if type(rows) is not list or len(rows) != len(controller.lib_controllers):
        return None
    state = []
    for row in rows:
        if (type(row) is not dict or any(type(key) is not str for key in row)
                or any(type(value) not in (str, int, bool, type(None))
                       for value in row.values())
                or type(row.get("num_threads")) is not int
                or row["num_threads"] < 1):
            return None
        state.append(tuple(sorted(row.items())))
    # This live token is never serialized: _QRPrepared reduces to None.
    # The cached controller renews after fork; preparation stores no controller.
    # Discovery order is not native state. Preserve every field and
    # duplicate record while giving identical inventories one order.
    # Values were restricted to exact primitive types above.
    state.sort(key=repr)
    return os.getpid(), id(controller), tuple(state)


def _qr_call_buffer(value, size):
    if type(value) is bytes:
        if len(value) != size:
            raise ValueError("Unsupported prepared QR transport buffer.")
        return value
    if ((type(value) is not np.ndarray and type(value) is not np.memmap) or value.dtype != np.uint8
            or value.ndim != 1 or not value.flags.c_contiguous or value.nbytes != size):
        raise ValueError("Unsupported prepared QR transport buffer.")
    if (type(value) is np.memmap and type(value.mode) is str
            and value.mode == "r" and not value.flags.writeable):
        # This is an operation-owned Joblib temporary mapping, not persistent
        # authority or protection against an unrelated external file writer.
        return value
    if (type(value) is np.ndarray and type(value.base) is bytes
            and len(value.base) == size and not value.flags.writeable):
        return value.base
    # Pickled sub-threshold arrays can have writable owning storage. Rebind
    # those to immutable bytes; never trust flags alone on a mutable owner.
    return value.tobytes(order="C")


def _qr_call_bind(payload):
    try:
        if type(payload) is not tuple or len(payload) != 6:
            return None
        shape, basis, q, r, order, native_state = payload
        if (type(shape) is not tuple or len(shape) != 2
                or any(type(n) is not int or n < 1 for n in shape)
                or type(order) is not str or order not in ("C", "F")):
            return None
        context = _qr_context()
        if context is None or context[2] != native_state:
            return None
        basis = _qr_call_buffer(basis, 8 * shape[0] * shape[1])
        q = _qr_call_buffer(q, 8 * shape[0] * shape[1])
        r = _qr_call_buffer(r, 8 * shape[1] ** 2)
        # Existing matches/factors consume the read-only buffer protocol;
        # basis bit comparison and all required solver validation stay intact.
        return _QRPrepared._from_checked_transport(shape, basis, q, r, order, context)
    except Exception:
        return None


def _project_shared_block(design, rows, solver, maxiter, batch_size,
                          native_threads, payload):
    with _native_controller().limit(limits=native_threads):
        prepared = _qr_call_bind(payload)
        return project_nnls(design, rows, solver=solver, maxiter=maxiter,
                            batch_size=batch_size, _qr_prepared=prepared)


def _qr_parent_payload(design, native_threads, prepared, allow_prepare):
    """Use only a fresh matching parent entry, or optional admitted cold work."""
    try:
        if _qr_native_context() is None:
            return None, None
        with _native_controller().limit(limits=native_threads):
            from ._nnls import _prepare_qr_entry
            entry = _prepare_qr_entry(design, _prepared=prepared, _owned=True,
                                      _allow_prepare=allow_prepare)
        if type(entry) is not _QRPrepared:
            return None, None
        return entry, entry._transport_payload()
    except Exception:
        # Only optional parent context/matching/preparation and payload work.
        return None, None


# Process-local preparations follow weak basis ownership, outside model state.
_qr_registry = {}
_qr_registry_pid = os.getpid()
_qr_registry_lock = threading.RLock()


def _qr_registry_current():
    global _qr_registry, _qr_registry_pid, _qr_registry_lock
    if _qr_registry_pid != os.getpid():
        # Do not acquire an inherited lock in a child process.
        _qr_registry = {}
        _qr_registry_lock = threading.RLock()
        _qr_registry_pid = os.getpid()
    return _qr_registry, _qr_registry_lock


def _qr_registry_get(owner):
    if type(owner) is not np.ndarray:
        return None
    table, lock = _qr_registry_current()
    with lock:
        found = table.get(id(owner))
        return found[1] if found is not None and found[0]() is owner else None


def _qr_registry_publish(owner, entry):
    if type(owner) is not np.ndarray or type(entry) is not _QRPrepared:
        return
    key = id(owner)
    table, lock = _qr_registry_current()

    def remove(reference):
        current, current_lock = _qr_registry_current()
        with current_lock:
            found = current.get(key)
            if found is not None and found[0] is reference:
                del current[key]

    reference = weakref.ref(owner, remove)
    with lock:
        # One preparation per live weak owner; a cache hit introduces no write.
        table[key] = reference, entry


def _project_weak_block(design, rows, solver, maxiter, batch_size,
                        native_threads, owner, owned=False):
    with _native_controller().limit(limits=native_threads):
        prepared = capture = None
        # Current public validation still runs inside project_nnls. These are
        # only cheap declines for new optional registry work.
        if (type(solver) is str and solver == "qr" and type(owner) is np.ndarray
                and type(design) is np.ndarray and design.ndim == 2
                and len(rows) >= 32 and design.shape[0] >= 2 * design.shape[1]
                and design.shape[1] >= 32
                and design.shape[0] * design.shape[1] ** 2 >= 1048576
                and (maxiter is None or (type(maxiter) is int
                                        and maxiter >= 3 * design.shape[1]))):
            try:
                prepared = _qr_registry_get(owner)
                capture = _QRFitCapture()
            except Exception:
                prepared = capture = None
        answer = project_nnls(design, rows, solver=solver, maxiter=maxiter,
                              batch_size=batch_size, _qr_prepared=prepared,
                              _qr_capture=capture, _qr_owned=owned)
        if (capture is not None and type(capture.entry) is _QRPrepared
                and capture.entry is not prepared):
            try:
                _qr_registry_publish(owner, capture.entry)
            except Exception:
                pass
        return answer


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


_QR_STANDARD_BLOCK = _project_block


def project_rows(design, rows, *, solver, workers, batch_size, native_threads, maxiter,
                 _qr_owner=None, _qr_owned=False):
    """Return ordered coordinates, using processes only for large batches.

Repeated calls with compatible controls reuse loky workers automatically.
Keeping pool objects outside model state also permits ordinary joblib/pickle
serialization. A first call still pays process startup cost.
"""
    validate_projection_parameters(solver, workers, batch_size, native_threads)
    if workers == 1 or len(rows) < max(512, 2 * batch_size):
        if (_project_block is _QR_STANDARD_BLOCK and type(solver) is str and solver == "qr"
                and type(_qr_owner) is np.ndarray
                and type(design) is np.ndarray and design.ndim == 2
                and type(rows) is np.ndarray and rows.ndim == 2 and len(rows) >= 32
                and design.shape[0] >= 2 * design.shape[1] and design.shape[1] >= 32
                and design.shape[0] * design.shape[1] ** 2 >= 1048576
                and (maxiter is None or (type(maxiter) is int and maxiter >= 3 * design.shape[1]))
                and _qr_mkl_controller() is not None):
            return _project_weak_block(design, rows, solver, maxiter, batch_size,
                                       native_threads, _qr_owner, _qr_owned)
        return _project_block(design, rows, solver, maxiter, batch_size, native_threads)
    spans = range(0, len(rows), batch_size)
    count = min(workers, (len(rows) + batch_size - 1) // batch_size)
    payload = None
    parent_entry = previous_entry = parent_owner = None
    if (_project_block is _QR_STANDARD_BLOCK and type(solver) is str and solver == "qr"
            and type(batch_size) is int and batch_size >= 32
            and type(native_threads) is int
            and type(design) is np.ndarray and design.dtype == np.float64 and design.ndim == 2
            and type(rows) is np.ndarray and rows.dtype == np.float64 and rows.ndim == 2
            and design.shape[0] >= 2 * design.shape[1]
            and (maxiter is None or (type(maxiter) is int and maxiter >= 3 * design.shape[1]))
            and type(_qr_owner) is np.ndarray and _qr_owned is True
            and design.flags.owndata and design.flags.c_contiguous
            and design.shape[1] >= 32
            and design.shape[0] * design.shape[1] ** 2 >= 1048576
            and _qr_mkl_controller() is not None):
        one_wave = type(workers) is int and len(spans) <= workers
        parent_owner = _qr_owner
        try:
            previous_entry = _qr_registry_get(parent_owner)
        except Exception:
            previous_entry = None
        # A one-wave miss adds no parent context scan, copy or QR.
        if not one_wave or type(previous_entry) is _QRPrepared:
            parent_entry, payload = _qr_parent_payload(
                design, native_threads, previous_entry, not one_wave)
    with parallel_config(backend="loky", inner_max_num_threads=native_threads):
        chunks = Parallel(n_jobs=count, batch_size=1, max_nbytes="1M",
                          mmap_mode="r", pre_dispatch="2*n_jobs")(
            (delayed(_project_shared_block)(design, rows[start:start + batch_size],
                 solver, maxiter, batch_size, native_threads, payload)
             if payload is not None else
             delayed(_project_block)(design, rows[start:start + batch_size],
                 solver, maxiter, batch_size, native_threads))
            for start in spans
        )
    answer = np.vstack(chunks)
    if (parent_owner is not None and type(parent_entry) is _QRPrepared
            and parent_entry is not previous_entry):
        try:
            _qr_registry_publish(parent_owner, parent_entry)
        except Exception:
            pass
    return answer
