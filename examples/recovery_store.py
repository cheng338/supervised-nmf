"""Final-fit outcome transactions for the CelebA parameter study.

Only current.json commits a generation. Uncommitted files remain as evidence.
Caller owns the existing whole-output OS lock throughout every operation.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
import uuid


_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# The imported checkout location is fixed; content hashes remain freshly read.
_NUMERIC_CODEC_PATH = Path(__file__).resolve().with_name("numeric_codec.py")
_NUMERIC_CODEC_ORIGIN = str(_NUMERIC_CODEC_PATH)


_FINAL_AGGREGATION_LOCKS_BY_PID = {os.getpid(): threading.RLock()}


def _reset_final_aggregation_guards_after_fork():
    # The child must not inherit an active owner's lock (or an old PID entry).
    # This callback runs before the child starts any additional Python threads.
    global _FINAL_AGGREGATION_LOCKS_BY_PID
    _FINAL_AGGREGATION_LOCKS_BY_PID = {os.getpid(): threading.RLock()}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_final_aggregation_guards_after_fork)


@contextmanager
def _final_aggregation_guard():
    """Wait for another thread; same-thread nesting remains reentrant."""
    pid = os.getpid()
    lock = _FINAL_AGGREGATION_LOCKS_BY_PID.get(pid)
    if lock is None:
        # PID ownership also avoids ordinary inherited-parent-lock reuse. The
        # supported CPython dict operation publishes one lock if callers race.
        lock = _FINAL_AGGREGATION_LOCKS_BY_PID.setdefault(pid, threading.RLock())
    waited = not lock.acquire(blocking=False)
    if waited:
        lock.acquire(blocking=True)
    try:
        yield waited
    finally:
        lock.release()


SCHEMA = "celeba-final-fit-outcome-v1"
POLICIES = ("off", "all")


class RecoveryError(RuntimeError):
    """Persistence failure distinct from numerical failure."""


def _hook(event, details=None):
    """No-op boundary replaced only by isolated subprocess probes."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RecoveryError(f"Refusing to overwrite recovery evidence: {path}")
    temporary = path.with_name(".j-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
    if path.exists() or path.is_symlink():
        raise RecoveryError(f"Recovery destination appeared: {path}")
    temporary.rename(path)


def _envelope(value):
    return {"payload": value, "sha256": _digest(value)}


def _read(path):
    path = Path(path)
    try:
        if path.is_symlink() or not path.is_file():
            raise RecoveryError(f"Recovery receipt is not a regular file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or set(value) != {"payload", "sha256"}
                or value["sha256"] != _digest(value["payload"])):
            raise RecoveryError(f"Invalid recovery checksum: {path}")
        return value["payload"]
    except RecoveryError:
        raise
    except Exception as error:
        raise RecoveryError(f"Cannot read recovery receipt {path}: {error}") from error


def validate_policy(policy):
    if policy not in POLICIES:
        raise ValueError(f"final_fit_recovery must be one of {POLICIES}")
    return policy


def _check_codec_source(codec, source_hash):
    origin = getattr(codec, "__file__", None)
    if (origin is None or (not (type(origin) is str and origin == _NUMERIC_CODEC_ORIGIN)
            and Path(origin).resolve()
            != Path(__file__).resolve().with_name("numeric_codec.py").resolve())):
        raise ImportError("Recovery requires its sibling numeric_codec.py; "
                          "restart with the intended checkout on the import path.")
    if getattr(codec, "_SOURCE_SHA256_AT_IMPORT", None) != source_hash:
        raise ImportError("numeric_codec.py changed after import or lacks an import source pin; "
                          "restart the interpreter before continuing this study.")


def _numeric_codec_module():
    """Validate the lazy codec before using its retained implementation."""
    import numeric_codec
    _check_codec_source(numeric_codec, hashlib.sha256(_NUMERIC_CODEC_PATH.read_bytes()).hexdigest())
    return numeric_codec


def validate_loaded_sources(sources):
    """Validate imported helpers against freshly observed source facts.

    Runtime identity observation stays fact-only. Its owning study calls this
    at admission, before accepting an identity or using a saved outcome.
    """
    if sources.get("recovery_store.py") != _SOURCE_SHA256_AT_IMPORT:
        raise RecoveryError("recovery_store.py changed after import; "
                            "restart the interpreter before continuing this study.")
    codec = sys.modules.get("numeric_codec")
    if codec is not None:
        _check_codec_source(codec, sources.get("numeric_codec.py"))


def _runtime_file_signature(path):
    """Observation key for an already resolved active-library path."""
    observed = path.stat()
    return (str(path), observed.st_size, observed.st_mtime_ns,
            observed.st_ctime_ns, observed.st_ino)


def _runtime_file_hash(path, file_cache):
    """Reuse a hash only within the caller's aggregation and unchanged metadata.

    This implements the runner's unchanged-runtime contract, not protection
    against deliberate byte changes that preserve all observed metadata. It also
    does not inspect bytes already mapped into process memory. No global cache is
    used: a new aggregation/process starts empty and hashes every active file.
    """
    before = _runtime_file_signature(path)
    cached = None if file_cache is None else file_cache.get(str(path))
    if cached is not None and cached["signature"] == before:
        return cached["sha256"]
    digest = _sha(path)
    if _runtime_file_signature(path) != before:
        raise RecoveryError("Native runtime file changed while its identity was read")
    if file_cache is not None:
        file_cache[str(path)] = {"signature": before, "sha256": digest}
    return digest


def active_runtime_identity(threads, file_cache=None, *, _initialization_cache=None):
    """Rescan active libraries, versions and source; optionally reuse DLL hashes.

    With no cache every call hashes every active library. A caller-provided cache
    is limited to one aggregation; source hashes are always read in full.
    """
    if _initialization_cache is not None:
        return _call_local_runtime_identity(threads, file_cache, _initialization_cache)
    return _runtime_identity_observe(threads, file_cache, initialize=True)


_INITIALIZATION_DEFAULT_HELPER = active_runtime_identity
_INITIALIZATION_DEFAULT_HOOK = _hook


def _call_local_runtime_identity(threads, file_cache, initialization):
    """Initialize once per aggregate; every identity remains a fresh read.

    This internal path belongs to the predefined example workflow. The unused
    positive finite probes initialize loaded numerical libraries, rather than
    validate fitted models or promise repeated arbitrary callback execution.
    """
    if type(initialization) is not dict:
        return _INITIALIZATION_DEFAULT_HELPER(threads, file_cache)
    try:
        if (_hook is not _INITIALIZATION_DEFAULT_HOOK
                or type(threads) is not int or threads < 1):
            initialization.pop("ready", None)
            initialization.pop("ready_threads", None)
            initialization.pop("_inventory_state", None)
            return _INITIALIZATION_DEFAULT_HELPER(threads, file_cache)
        if (initialization.get("ready") is True
                and initialization.get("ready_threads") == threads):
            return _runtime_identity_observation(threads, file_cache, _inventory_cache=initialization)
        initialization.pop("ready", None)
        initialization.pop("ready_threads", None)
        initialization.pop("_inventory_state", None)
        observed = _INITIALIZATION_DEFAULT_HELPER(threads, file_cache)
        initialization["ready_threads"] = threads
        initialization["ready"] = True
        return observed
    except BaseException:
        initialization.pop("ready", None)
        initialization.pop("ready_threads", None)
        initialization.pop("_inventory_state", None)
        raise


def _runtime_identity_observation(threads, file_cache=None, *, _inventory_cache=None):
    """Observe fresh identity without repeating native initialization."""
    return _runtime_identity_observe(
        threads, file_cache, initialize=False, inventory_cache=_inventory_cache)


def _runtime_identity_observe(threads, file_cache, *, initialize, inventory_cache=None):
    """Share ordered identity observations across full and warm routes."""
    import numpy as np
    import scipy
    import sklearn
    import threadpoolctl
    import joblib
    from scipy.linalg.blas import dgemv
    from threadpoolctl import threadpool_info, threadpool_limits
    if initialize:
        with threadpool_limits(limits=threads):
            np.dot(np.ones((32, 32)), np.ones((32, 32)))
            dgemv(1.0, np.ones((32, 32)), np.ones(32))
    pools = []
    for pool in (threadpool_info() if initialize else _inventory_fresh_info(inventory_cache)):
        item = {key: pool.get(key) for key in (
            "user_api", "internal_api", "prefix", "version", "threading_layer", "architecture")}
        path = Path(pool["filepath"]).resolve()
        item.update(filepath=str(path), sha256=_runtime_file_hash(path, file_cache))
        pools.append(item)
    pools.sort(key=lambda item: item["filepath"])
    return {
        "python": sys.version, "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(), "threads": int(threads), "native_libraries": pools,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__,
                     "threadpoolctl": threadpoolctl.__version__, "joblib": joblib.__version__},
        "recovery_sources": {name: _sha(Path(__file__).with_name(name))
                             for name in ("recovery_store.py", "numeric_codec.py")},
    }


import types as _inventory_types

_INVENTORY_RELEASED_ROUTES = _inventory_types.MappingProxyType({'routes_enabled': False})

_INVENTORY_METHOD_CODE_PINS = {'__init__': '3808bd5d9ec1b64ad26e43b441c7d43dec448d1f6c8e04b08f2eb8c50636c244', '_load_libraries': 'a284699a9160dcbccb0e59f20aed94726e992b1b5374cb24bce5e4446233dab3', '_find_libraries_with_linux': '6dc651f0b2b18d957850cbbe91ff646758954af7bfa287d977d21223c12fb610', '_make_controller_from_path': '05403a85901a29b28181ecab24e0457ddd243e05b1ff503f0f6e750cef4b9b15', '_check_prefix': 'a4ca831a21ed6484bf2a5ff91ac146d7370c5ac718c5ddd69947ceaca8d8212e', 'threadpool_info': '6b7fe054a71e45485f5bacdf639898d8cb12688d9f73f69351c60b63ca508467'}


def _inventory_encode_code(value):
    if isinstance(value, _inventory_types.CodeType):
        fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount",
                  "co_nlocals", "co_stacksize", "co_flags", "co_code",
                  "co_consts", "co_names", "co_varnames", "co_name",
                  "co_freevars", "co_cellvars", "co_exceptiontable")
        return {name: _inventory_encode_code(getattr(value, name, None)) for name in fields}
    if isinstance(value, tuple):
        return [_inventory_encode_code(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, frozenset):
        return {"frozenset": sorted((_inventory_encode_code(item) for item in value), key=repr)}
    return {"type": type(value).__name__, "value": repr(value)}


def _inventory_method_shape(function):
    code = getattr(function, "__code__", None)
    if code is None:
        return None
    payload = json.dumps(_inventory_encode_code(code), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _inventory_registry_snapshot(module):
    # The negative predicate depends only on ordered classes/current prefixes.
    # Positive paths always run the original constructor and symbol admission.
    classes = tuple(module._ALL_CONTROLLERS)
    rows = []
    for cls in classes:
        prefixes = cls.filename_prefixes
        if type(prefixes) not in (tuple, list) or any(type(value) is not str for value in prefixes):
            return None
        rows.append((id(cls), tuple(prefixes)))
    return tuple(rows), classes


def _inventory_update_routes(state, module):
    observed = _inventory_registry_snapshot(module)
    if observed is None:
        state["routes_enabled"] = False
        state["negative_paths"] = set()
        return
    signature, classes = observed
    if signature != state.get("registry_signature"):
        state["negative_paths"] = set()
    state["registry_signature"] = signature
    state["registry_classes"] = classes  # Keep class identities alive with key.
    state["routes_enabled"] = True


def _inventory_negative_make(self, filepath):
    state = self._recovery_inventory_state
    module = self._recovery_inventory_module
    original = self._recovery_original_make
    if not state["routes_enabled"]:
        return original(self, filepath)
    canonical = module._realpath(filepath)
    if canonical in state["negative_paths"]:
        return
    filename = os.path.basename(canonical).lower()
    # Read the live registry on a cache miss, as the original implementation
    # does. A positive constructor may synchronously register another class.
    if any(self._check_prefix(filename, cls.filename_prefixes) is not None
           for cls in module._ALL_CONTROLLERS):
        result = original(self, filepath)
        _inventory_update_routes(state, module)
        return result
    state["negative_paths"].add(canonical)


def _inventory_linux_dedup(self):
    with open("/proc/self/maps") as f:
        maps = f.read()
    filepaths = set()
    for line in maps.splitlines():
        start_index = line.find("/")
        if start_index == -1 or ".so" not in line:
            continue
        filepath = line[start_index:]
        if filepath in filepaths:
            continue
        if os.path.exists(filepath):
            filepaths.add(filepath)
    for filepath in filepaths:
        self._make_controller_from_path(filepath)


def _inventory_fresh_info(initialization):
    import threadpoolctl as module
    if type(initialization) is not dict:
        return module.threadpool_info()
    use_g = True
    use_h = sys.platform == "linux"
    if not (use_g or use_h) or sys.platform not in ("win32", "linux"):
        return module.threadpool_info()
    defaults = getattr(module.threadpool_info, "__defaults__", None)
    keyword_defaults = getattr(module.threadpool_info, "__kwdefaults__", None)
    if not (type(defaults) is tuple and len(defaults) == 1
            and defaults[0] is False and keyword_defaults is None):
        # The replaced public entrypoint normally passes False explicitly to
        # controller.info. Preserve any changed default via the public call.
        initialization.pop("_inventory_state", None)
        return module.threadpool_info()
    names = ("__init__", "_load_libraries")
    if use_g:
        names += ("_make_controller_from_path", "_check_prefix")
    if use_h:
        names += ("_find_libraries_with_linux",)
    try:
        base = module.ThreadpoolController
        functions = (("threadpool_info", module.threadpool_info),) + tuple(
            (name, getattr(base, name)) for name in names)
        key = (module.__version__, base, module.threadpool_info, module._realpath,
               tuple((name, function, getattr(function, "__code__", None))
                     for name, function in functions))
    except AttributeError:
        initialization.pop("_inventory_state", None)
        return module.threadpool_info()
    state = initialization.get("_inventory_state")
    if type(state) is not dict or state.get("qualification_key") != key:
        state = {"qualification_key": key, "qualified": False}
        initialization["_inventory_state"] = state
        try:
            state["qualified"] = (module.__version__ == "3.7.0" and all(
                _inventory_method_shape(function) == _INVENTORY_METHOD_CODE_PINS[name]
                for name, function in functions))
        except (OSError, TypeError, ValueError, AttributeError, SyntaxError):
            # An unavailable code shape declines optimization. Native getter,
            # controller and discovery errors below are never swallowed.
            state["qualified"] = False
        if state["qualified"]:
            attributes = {}
            if use_g:
                attributes["_make_controller_from_path"] = _inventory_negative_make
            if use_h:
                attributes["_find_libraries_with_linux"] = _inventory_linux_dedup
            state["controller_type"] = type("_RecoveryInventoryController", (base,), attributes)
    if not state["qualified"]:
        return module.threadpool_info()
    controller_type = state["controller_type"]
    # Allocate the private subclass, install operation-local routing state, then
    # run the exact original initializer/discovery/warning sequence.
    controller = controller_type.__new__(controller_type)
    if use_g:
        _inventory_update_routes(state, module)
        controller._recovery_inventory_state = state
        controller._recovery_inventory_module = module
        controller._recovery_original_make = base._make_controller_from_path
    try:
        base.__init__(controller)
        return controller.info(debugging_info=False)
    finally:
        # A retained native/custom parent can rescan through the original route.
        # It no longer owns this operation's negative paths or code/type state.
        controller.__dict__["_recovery_inventory_state"] = _INVENTORY_RELEASED_ROUTES


def _inventory(directory):
    directory = Path(directory)
    values = {}
    if directory.is_symlink() or not directory.is_dir():
        raise RecoveryError(f"Invalid model payload directory: {directory}")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RecoveryError(f"Unexpected link in owned model payload: {path}")
        if path.is_file():
            values[path.relative_to(directory).as_posix()] = {
                "sha256": _sha(path), "bytes": path.stat().st_size}
    return values


class RecoveryStore:
    def __init__(self, directory, identity, *, policy="all",
                 attempt_identity=None):
        self.directory = Path(directory)
        self.policy = validate_policy(policy)
        self._identity_factory = identity if callable(identity) else lambda: identity
        self._identity_value = None
        self.attempt_identity = attempt_identity if attempt_identity is not None else (
            identity if not callable(identity) else {})
        self.pointer = self.directory / "current.json"

    @property
    def identity(self):
        if self._identity_value is None:
            self._identity_value = json.loads(_canonical(self._identity_factory()))
        return self._identity_value

    @property
    def identity_digest(self):
        return _digest(self.identity)

    def start_attempt(self, operation):
        if self.policy == "off":
            return None
        attempt = self.directory / "attempts" / ("a-" + uuid.uuid4().hex)
        _write_new(attempt / "start.json", _envelope({
            "schema": SCHEMA, "identity": self.attempt_identity, "operation": operation,
            "pid": os.getpid(), "time_ns": time.time_ns()}))
        return attempt

    def record(self, attempt, stage, details=None):
        if attempt is None:
            return
        if not re.fullmatch(r"[a-z][a-z0-9-]*", stage):
            raise ValueError("Invalid attempt stage")
        _write_new(Path(attempt) / (stage + ".json"), _envelope({
            "stage": stage, "time_ns": time.time_ns(), "details": details or {}}))

    def load_outcome(self):
        if self.policy == "off" or (not self.pointer.exists() and not self.pointer.is_symlink()):
            return None
        start = time.perf_counter()
        pointer = _read(self.pointer)
        if (not isinstance(pointer, dict) or pointer.get("schema") != SCHEMA
                or pointer.get("identity_sha256") != self.identity_digest
                or not isinstance(pointer.get("generation"), str)
                or not re.fullmatch(r"g-[0-9a-f]{32}", pointer["generation"])):
            raise RecoveryError("Committed fit outcome has incompatible identity or schema")
        generation = self.directory / "generations" / pointer["generation"]
        manifest_path = generation / "outcome.json"
        if (generation.is_symlink() or not generation.is_dir()
                or not manifest_path.is_file() or manifest_path.is_symlink()
                or _sha(manifest_path) != pointer.get("manifest_sha256")):
            raise RecoveryError("Committed fit outcome manifest is missing or changed")
        manifest = _read(manifest_path)
        if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
                or manifest.get("identity") != self.identity
                or manifest.get("kind") not in {"fitted", "terminal"}
                or not isinstance(manifest.get("metadata"), dict)):
            raise RecoveryError("Committed fit outcome metadata is incompatible")
        if manifest["kind"] == "fitted":
            if _inventory(generation / "payload") != manifest.get("files"):
                raise RecoveryError("Committed model payload is missing or changed")
        elif manifest.get("files") != {} or "codec" in manifest:
            raise RecoveryError("Terminal outcome unexpectedly references model state")
        return {"manifest": manifest, "generation": generation,
                "verify_seconds": time.perf_counter() - start}

    def restore_model(self, outcome, *, codec_context=None):
        if outcome["manifest"]["kind"] != "fitted":
            raise RecoveryError("A terminal fit outcome has no fitted model")
        numeric_codec = _numeric_codec_module()
        start = time.perf_counter()
        stats = {}
        codec_arguments = {} if codec_context is None else {"context": codec_context}
        try:
            model = numeric_codec.load_model(
                outcome["generation"] / "payload", outcome["manifest"]["codec"], stats=stats,
                **codec_arguments)
        except Exception as error:
            raise RecoveryError(f"Model reconstruction failed: {error}") from error
        return model, time.perf_counter() - start, stats

    def commit(self, metadata, *, model=None, details=None, codec_context=None):
        if self.policy == "off":
            return None
        if self.pointer.exists() or self.pointer.is_symlink():
            raise RecoveryError("A fit outcome is already committed; recover it instead")
        start = time.perf_counter()
        generation = self.directory / "generations" / ("g-" + uuid.uuid4().hex)
        generation.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": SCHEMA, "identity": self.identity,
                    "kind": "terminal" if model is None else "fitted",
                    "metadata": metadata, "files": {}}
        event = dict(details or {}, generation=generation.name)
        try:
            if model is not None:
                numeric_codec = _numeric_codec_module()
                payload = generation / "payload"
                payload.mkdir()
                _hook("payload-start", event)
                codec_arguments = {} if codec_context is None else {"context": codec_context}
                manifest["codec"] = numeric_codec.dump_model(model, payload, **codec_arguments)
                manifest["files"] = _inventory(payload)
                _hook("payload-complete", event)
            _write_new(generation / "outcome.json", _envelope(manifest))
            _hook("before-manifest-commit", event)
            _write_new(self.pointer, _envelope({
                "schema": SCHEMA, "identity_sha256": self.identity_digest,
                "generation": generation.name,
                "manifest_sha256": _sha(generation / "outcome.json")}))
        except RecoveryError:
            raise
        except Exception as error:
            raise RecoveryError(f"Could not publish fit outcome: {error}") from error
        _hook("manifest-commit", event)
        return {"generation": generation.name, "checkpoint_seconds": time.perf_counter() - start,
                "model_bytes": sum(item["bytes"] for item in manifest["files"].values()),
                "kind": manifest["kind"]}
