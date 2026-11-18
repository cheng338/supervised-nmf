"""Strict identities for reusable estimator configurations.

Report text is deliberately not an identity. This module describes ordinary
sklearn constructor parameters, exact array values and inspectable Python
implementations without fitting or mutating the supplied objects. It does not
infer dependencies on mutable globals, external files/services, monkeypatched
methods, or stale class objects retained across module reloads.
"""

from __future__ import annotations

from abc import ABCMeta
import functools
import hashlib
import inspect
import json
import marshal
from pathlib import Path, PurePath
import struct
import sys
import types

import numpy as np
from sklearn.base import BaseEstimator


def _qualified(value):
    return f"{value.__module__}.{value.__qualname__}"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _native_module(name):
    return (name in {"builtins", "math", "operator", "_operator", "_functools"}
            or name == "numpy" or name.startswith("numpy.")
            or name == "scipy" or name.startswith("scipy.")
            or name == "sklearn" or name.startswith("sklearn."))


def _stdlib_source_function(function, prefix):
    """Decline ordinary wrappers, including those installed before import."""
    if type(function) is not types.FunctionType:
        return False
    module = sys.modules.get(function.__module__)
    if type(module) is not types.ModuleType:
        return False
    namespace = vars(module)
    name = namespace.get("__name__", "")
    filename = namespace.get("__file__")
    return (type(name) is str and (name == prefix or name.startswith(prefix + "."))
            and function.__globals__ is namespace and type(filename) is str
            and Path(function.__code__.co_filename) == Path(filename))


_SOURCE_LOOKUP_FUNCTIONS = (inspect.getsourcefile, inspect.getfile, Path.is_file)
_SOURCE_LOOKUP_HAS_STDLIB_ORIGIN = all(
    _stdlib_source_function(function, prefix)
    for function, prefix in zip(_SOURCE_LOOKUP_FUNCTIONS, ("inspect", "inspect", "pathlib"))
)


def _ordinary_source_mro(cls):
    """Use the short resolution scope only for ordinary source namespaces."""
    if type(cls) is not type and type(cls) is not ABCMeta:
        return False
    for parent in cls.__mro__:
        if parent is object:
            continue
        namespace = vars(parent)
        module_name = namespace.get("__module__")
        if (type(module_name) is not str
                or type(type.__getattribute__(parent, "__qualname__")) is not str
                or type(type.__getattribute__(parent, "__name__")) is not str):
            return False
        module = sys.modules.get(module_name)
        if type(module) is not types.ModuleType:
            return False
        namespace = vars(module)
        if type(namespace.get("__file__")) is not str or "__getattr__" in namespace:
            return False
    return True


class _Encoder:
    def __init__(self, *, sources_only=False, previous_files=None):
        self.sources_only = sources_only
        self.previous_files = previous_files
        self.active = set()
        self.sources = {}
        self.file_hashes = {}
        self.class_records = {}
        self._mro_resolutions = None

    def _error(self, path, reason):
        raise ValueError(f"Cannot establish reusable search identity for {path}: {reason}.")

    def _file_digest(self, source):
        content = source.read_bytes()
        slot = len(self.file_hashes)
        if slot >= 16 or type(content) is not bytes or len(content) > 512 * 1024:
            return hashlib.sha256(content).hexdigest()
        # Read order can change. Only exact byte equality permits digest reuse.
        saved = self.previous_files.get(slot)
        if saved is not None and content == saved[0]:
            return saved[1]
        digest = hashlib.sha256(content).hexdigest()
        if saved is None:
            self.previous_files[slot] = (content, digest)
        return digest

    def _source(self, value, path):
        direct = None
        if (self._mro_resolutions is not None and _SOURCE_LOOKUP_HAS_STDLIB_ORIGIN
                and inspect.getsourcefile is _SOURCE_LOOKUP_FUNCTIONS[0]
                and inspect.getfile is _SOURCE_LOOKUP_FUNCTIONS[1]
                and Path.is_file is _SOURCE_LOOKUP_FUNCTIONS[2]):
            module = sys.modules.get(vars(value).get("__module__"))
            namespace = vars(module) if type(module) is types.ModuleType else {}
            candidate = namespace.get("__file__")
            if (type(candidate) is str and candidate.endswith(".py")
                    and "__getattr__" not in namespace):
                try:
                    if Path(candidate).is_file():
                        direct = candidate
                except (OSError, ValueError):
                    pass  # Preserve the original discovery/error path below.
        try:
            filename = direct if direct is not None else inspect.getsourcefile(value)
        except (TypeError, OSError):
            filename = None
        module_name = getattr(value, "__module__", "") or ""
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        if filename is not None and (direct is not None or Path(filename).is_file()):
            cache = self._mro_resolutions
            if cache is None or type(filename) is not str:
                source = Path(filename).resolve()
            elif filename in cache:
                source = cache[filename]
            else:
                source = Path(filename).resolve()
                cache[filename] = source
            try:
                if source not in self.file_hashes:
                    if self.previous_files is None:
                        digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    else:
                        digest = self._file_digest(source)
                    self.file_hashes[source] = digest
            except OSError as error:
                self._error(path, f"implementation source cannot be read ({type(error).__name__})")
            key = f"{module_name}.{name}@{source.as_posix()}"
            digest = self.file_hashes[source]
            self.sources[key] = digest
            return {"path": source.as_posix(), "sha256": digest}
        if _native_module(module_name):
            # Native implementations are additionally bound by the search's
            # Python and numerical-package version/runtime identities.
            return {"native_module": module_name}
        self._error(path, "implementation source is unavailable; supply an inspectable estimator or callable")

    def _class(self, cls, path):
        if cls not in self.class_records:
            # Source-only scans may resolve the same file for several ordinary
            # MRO entries. Keep every is_file check and discard resolutions
            # before graph traversal can invoke another user callback.
            previous = self._mro_resolutions
            self._mro_resolutions = {} if self.sources_only and _ordinary_source_mro(cls) else None
            try:
                sources = []
                for parent in cls.__mro__:
                    sources.append({"type": _qualified(parent),
                                    "source": self._source(parent, path)})
                self.class_records[cls] = {"type": _qualified(cls), "implementations": sources}
            finally:
                self._mro_resolutions = previous
        return self.class_records[cls]

    def _dtype(self, dtype, path):
        dtype = np.dtype(dtype)
        return {"str": dtype.str, "descr": self.encode(dtype.descr, path + ".descr"),
                "metadata": self.encode(dict(dtype.metadata), path + ".metadata")
                if dtype.metadata else None}

    def _native_callable(self, value, path):
        name = getattr(value, "__name__", None)
        module = getattr(value, "__module__", None)
        if name and module and _native_module(module):
            return {"kind": "native_callable", "name": f"{module}.{name}",
                    "source": self._source(value, path)}
        # NumPy ufuncs have no __module__. Accept only actual exported library
        # objects, not arbitrary frompyfunc functions sharing a display name.
        if name and isinstance(value, np.ufunc):
            matches = []
            for module_name, module_object in tuple(sys.modules.items()):
                if not _native_module(module_name) or module_object is None:
                    continue
                if vars(module_object).get(name) is value:
                    matches.append(module_name)
            if matches:
                module = min(matches, key=lambda candidate: (len(candidate), candidate))
                return {"kind": "native_callable", "name": f"{module}.{name}",
                        "source": {"native_module": module}}
        self._error(path, "opaque native callable has no stable supported library identity")

    def encode(self, value, path="configuration"):
        if value is None:
            return {"kind": "none"}
        if value is np._NoValue:
            # NumPy's exported singleton appears in the defaults of ordinary
            # array-function dispatchers (for example np.mean). Its identity
            # is bound by the NumPy runtime, not arbitrary object state.
            return {"kind": "numpy_singleton", "name": "numpy._NoValue"}
        if type(value) is bool:
            return {"kind": "bool", "value": value}
        if type(value) is str:
            return {"kind": "str", "value": value}
        if type(value) is bytes:
            return {"kind": "bytes", "length": len(value),
                    "sha256": None if self.sources_only else hashlib.sha256(value).hexdigest()}
        if type(value) is int:
            return {"kind": "int", "value": str(value)}
        if type(value) is float:
            return {"kind": "float", "bits": struct.pack("!d", value).hex()}
        if type(value) is complex:
            return {"kind": "complex", "real": self.encode(value.real, path + ".real"),
                    "imag": self.encode(value.imag, path + ".imag")}
        if isinstance(value, PurePath):
            return {"kind": "path", "type": _qualified(type(value)), "value": str(value)}
        if isinstance(value, np.dtype):
            return {"kind": "dtype", **self._dtype(value, path)}
        if isinstance(value, type):
            # Classes such as np.float32 must not be mistaken for instances
            # exposing an unbound get_params method.
            return {"kind": "class", **self._class(value, path)}

        marker = id(value)
        if marker in self.active:
            self._error(path, "cyclic parameter graphs are unsupported")
        self.active.add(marker)
        try:
            return self._compound(value, path)
        finally:
            self.active.remove(marker)

    def _compound(self, value, path):
        if isinstance(value, np.generic):
            if type(value) is not value.dtype.type:
                self._error(path, "NumPy scalar subclass state is not represented by ordinary scalar values")
            if value.dtype.hasobject:
                content = self.encode(value.item(), path + ".item")
            else:
                content = None if self.sources_only else hashlib.sha256(value.tobytes()).hexdigest()
            return {"kind": "numpy_scalar", "dtype": self._dtype(value.dtype, path + ".dtype"),
                    "content": content}
        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray:
                self._error(path, "ndarray subclass state is not represented by ordinary array values")
            if value.dtype.hasobject:
                content = [self.encode(item, f"{path}[{index}]")
                           for index, item in enumerate(value.flat)]
            elif self.sources_only:
                content = None
            else:
                content = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            return {"kind": "ndarray", "dtype": self._dtype(value.dtype, path + ".dtype"),
                    "shape": list(value.shape), "content": content}
        if isinstance(value, slice):
            return {"kind": "slice", "start": self.encode(value.start, path + ".start"),
                    "stop": self.encode(value.stop, path + ".stop"),
                    "step": self.encode(value.step, path + ".step")}
        if type(value) is dict:
            entries = [[self.encode(key, f"{path}.items[{index}].key"),
                        self.encode(item, f"{path}.items[{index}].value")]
                       for index, (key, item) in enumerate(value.items())]
            # Insertion order is part of Python dict semantics. A dict-valued
            # transformer parameter may intentionally specify ordered steps.
            return {"kind": "dict", "items": entries}
        if type(value) in (list, tuple):
            return {"kind": type(value).__name__,
                    "items": [self.encode(item, f"{path}[{index}]") for index, item in enumerate(value)]}
        if type(value) in (set, frozenset):
            items = [self.encode(item, path + ".item") for item in value]
            return {"kind": type(value).__name__, "items": sorted(items, key=_canonical)}
        if type(value) is np.random.RandomState:
            return {"kind": "random_state", "state": self.encode(value.get_state(), path + ".state")}
        if type(value) is np.random.Generator:
            return {"kind": "generator", "bit_generator": self._class(type(value.bit_generator), path),
                    "state": self.encode(value.bit_generator.state, path + ".state")}
        if type(value) is functools.partial:
            return {"kind": "partial", "function": self.encode(value.func, path + ".func"),
                    "args": self.encode(value.args, path + ".args"),
                    "keywords": self.encode(value.keywords, path + ".keywords")}
        if isinstance(value, types.FunctionType):
            closure = []
            for index, cell in enumerate(value.__closure__ or ()):
                try:
                    content = cell.cell_contents
                except ValueError:
                    self._error(path, "callable contains an empty closure cell")
                closure.append(self.encode(content, f"{path}.closure[{index}]"))
            return {"kind": "function", "name": _qualified(value),
                    "source": self._source(value, path),
                    "code_sha256": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
                    "defaults": self.encode(value.__defaults__, path + ".defaults"),
                    "kwdefaults": self.encode(value.__kwdefaults__, path + ".kwdefaults"),
                    "closure": closure}
        if isinstance(value, (types.BuiltinFunctionType, np.ufunc)):
            return self._native_callable(value, path)
        if _qualified(type(value)) == "numpy._ArrayFunctionDispatcher":
            native = self._native_callable(value, path)
            wrapped = getattr(value, "__wrapped__", None)
            if not isinstance(wrapped, types.FunctionType):
                self._error(path, "NumPy dispatcher does not expose an inspectable implementation")
            return {"kind": "numpy_array_function", "native": native,
                    "implementation": self.encode(wrapped, path + ".implementation")}
        if hasattr(value, "get_params") and callable(value.get_params):
            clone_hook = getattr(type(value), "__sklearn_clone__", None)
            if clone_hook is not None and clone_hook is not BaseEstimator.__sklearn_clone__:
                self._error(path, "custom or frozen __sklearn_clone__ behavior can preserve state not described by constructor parameters")
            try:
                parameters = value.get_params(deep=False)
            except Exception as error:
                self._error(path, f"constructor parameters cannot be inspected ({type(error).__name__})")
            if type(parameters) is not dict:
                self._error(path, "get_params(deep=False) must return a dictionary")
            return {"kind": "estimator", "implementation": self._class(type(value), path),
                    "parameters": self.encode(parameters, path + ".parameters")}
        self._error(path, f"unsupported opaque value of type {_qualified(type(value))}")


def configuration_identity(value):
    """Return a structural JSON-safe identity, rejecting opaque/cyclic values.

    Ordinary fitted estimator attributes are intentionally omitted because
    sklearn clones the constructor configuration before each CV fit. Custom
    cloning that can preserve fitted state is rejected instead of falsely
    treating its constructor parameters as a complete reusable identity.
    """
    return _Encoder().encode(value)


def identity_digest(value):
    """Hash :func:`configuration_identity` with a canonical JSON encoding."""
    return hashlib.sha256(_canonical(configuration_identity(value)).encode("utf-8")).hexdigest()


def implementation_sources(*values, _previous_files=None):
    """Collect implementation file hashes through nested configurations.

    Numeric array contents are not hashed on this source-only path. Every
    call reads the source files discovered by its own configuration traversal.
    Runtime monkeypatching and mutable external/global dependencies are outside
    this contract; their behavior is not inferred from source file text.
    """
    encoder = _Encoder(sources_only=True, previous_files=_previous_files)
    for index, value in enumerate(values):
        encoder.encode(value, f"configuration[{index}]")
    return dict(sorted(encoder.sources.items()))
