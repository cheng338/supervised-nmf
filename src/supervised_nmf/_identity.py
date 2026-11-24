"""Strict identities for reusable estimator configurations.

Report text is deliberately not an identity. This module describes ordinary
sklearn constructor parameters and output settings, exact array values and inspectable Python
implementations without fitting or mutating the supplied objects. It does not
infer dependencies on mutable globals, external files/services, monkeypatched
methods, or stale class objects retained across module reloads. Primitive
values are compared by value. Allocation addresses and scalar interning are
outside the structural identity.
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
import weakref

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
_PATH_READ_BYTES = Path.read_bytes
_SOURCE_READ_HAS_STDLIB_ORIGIN = (_stdlib_source_function(_PATH_READ_BYTES, "pathlib")
                                  and _PATH_READ_BYTES.__name__ == "read_bytes")


def _source_bytes(path):
    """Read the current source, preserving interruptions during file cleanup."""
    reader = path.read_bytes
    if (not _SOURCE_READ_HAS_STDLIB_ORIGIN or type(reader) is not types.MethodType
            or reader.__func__ is not _PATH_READ_BYTES or reader.__self__ is not path):
        return reader()
    # Match pathlib's observable opener arguments. Python 3.14 switched
    # whole-file reads to unbuffered I/O.
    manager = (path.open(mode="rb", buffering=0) if sys.version_info >= (3, 14)
               else path.open(mode="rb"))
    read_abort = None
    try:
        with manager as stream:
            try:
                return stream.read()
            except BaseException as error:
                if isinstance(error, MemoryError) or not isinstance(error, Exception):
                    read_abort = error
                raise
        return None  # The opener explicitly suppressed the read error.
    except BaseException as exit_error:
        if (read_abort is None or exit_error is read_abort or
                (isinstance(read_abort, MemoryError) and not isinstance(exit_error, Exception))):
            raise
    # Cleanup replaced an interruption, or replaced an allocation failure
    # with an ordinary error. Restore the read error outside the exit handler.
    raise read_abort


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


_STOCK_ENGINES = (np.random.MT19937, np.random.PCG64, np.random.PCG64DXSM,
                  np.random.Philox, np.random.SFC64)
_COPIES_SEED_SEQUENCE = np.__version__.split(".", 1)[0] == "2"
_UNKNOWN_LINEAGE = object()
_RNG_NAMES = ("random_state", "rng", "seed")
_STRING_DTYPE = getattr(getattr(np, "dtypes", None), "StringDType", None)
_OWNER_NUMBER_DTYPES = tuple(np.dtype(code) for code in "?bBhHiIlLqQefdFDgG")
_SCALAR_OWNER_MAX_BYTES = 4096
_SCALAR_OWNER_MAX_ARRAYS = 4


def _integer_text(value):
    # 2048 bits fit within even Python's minimum nonzero decimal digit limit.
    # Hexadecimal conversion is exact and independent of that process setting.
    return str(value) if value.bit_length() <= 2048 else hex(value)


def _plain_dtype_value(value):
    return value is None or any(type(value) is cls for cls in (str, bytes, bool, int, float, complex))


def _child_stream_state(engine):
    """Qualify the ordinary immutable seed lineage copied by NumPy 2.x."""
    if not _COPIES_SEED_SEQUENCE or not any(type(engine) is cls for cls in _STOCK_ENGINES):
        return _UNKNOWN_LINEAGE
    seq = engine.seed_seq
    if seq is None:
        return None
    if (type(seq) is not np.random.SeedSequence or type(seq.entropy) is not int
            or type(seq.spawn_key) is not tuple or any(type(k) is not int for k in seq.spawn_key)):
        return _UNKNOWN_LINEAGE
    pool = seq.pool
    if (type(pool) is not np.ndarray or pool.dtype != np.uint32 or pool.ndim != 1
            or pool.dtype.metadata is not None or pool.dtype.names is not None or pool.dtype.subdtype is not None
            or pool.strides != (pool.itemsize,)
            or not pool.flags.writeable or pool.size != seq.pool_size):
        return _UNKNOWN_LINEAGE
    state = seq.state
    if (tuple(state) != ("entropy", "spawn_key", "pool_size", "n_children_spawned")
            or type(state["pool_size"]) is not int or type(state["n_children_spawned"]) is not int):
        return _UNKNOWN_LINEAGE
    return {"format": 1, "entropy": _integer_text(state["entropy"]),
            "spawn_key": [_integer_text(word) for word in state["spawn_key"]],
            "pool_size": state["pool_size"], "n_children_spawned": _integer_text(state["n_children_spawned"]),
            "pool_dtype": pool.dtype.str, "pool_length": pool.size,
            "pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest()}


class _SourceUnavailable(ValueError):
    pass


class _Encoder:
    def __init__(self, *, sources_only=False, collect_sources=False, previous_files=None, observe_rng=False,
                 nonstandard_text=None, split_limits=None, split_seed=None, fit_rng_dependencies=None,
                 output_libraries=None):
        self.output_libraries = output_libraries
        self.split_limits = split_limits
        self.split_seed = split_seed
        self.fit_rng_dependencies = fit_rng_dependencies
        self.sources_only = sources_only
        self.collect_sources = collect_sources
        self.observe_rng = observe_rng
        self.nonstandard_text = nonstandard_text
        self.previous_files = previous_files
        self.active = set()
        self.fit_rng_count = 0
        self.fit_rng_objects = None
        self.rng_pool_owners = None
        self.external_seed_buffers = None
        self.scalar_storage_ids = None
        self.in_rng_state = False
        self.in_function = False
        self.in_copy_graph = False
        self.preserved_objects = {}
        self.unverified_owner = False
        self.sources = {}
        self.file_hashes = {}
        self.class_records = {}
        self._mro_resolutions = None

    def _error(self, path, reason, error_type=ValueError):
        raise error_type(f"Cannot establish reusable search identity for {path}: {reason}.")

    def _limit(self, reason):
        if self.split_limits is not None and reason not in self.split_limits:
            self.split_limits.append(reason)

    def _record_alias(self, value, record, field):
        # Strong references keep object IDs unique until this traversal ends.
        key = id(value)
        if key in self.preserved_objects:
            record[field] = self.preserved_objects[key][1]
        else:
            self.preserved_objects[key] = value, len(self.preserved_objects)

    def _text(self, value):
        # A path or library can supply a str subclass with user-defined equality.
        # Preserve the value and let calibration keep its two snapshots separate.
        if type(value) is not str and self.nonstandard_text is not None and not self.nonstandard_text:
            self.nonstandard_text.append(True)
        return value

    def _file_digest(self, source):
        content = _source_bytes(source)
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
                        digest = hashlib.sha256(_source_bytes(source)).hexdigest()
                    else:
                        digest = self._file_digest(source)
                    self.file_hashes[source] = digest
            except OSError as error:
                self._error(path, f"implementation source cannot be read ({type(error).__name__})")
            key = f"{module_name}.{name}@{source.as_posix()}"
            digest = self.file_hashes[source]
            self.sources[key] = digest
            return {"path": self._text(source.as_posix()), "sha256": digest}
        if _native_module(module_name):
            # Native implementations are additionally bound by the search's
            # Python and numerical-package version/runtime identities.
            return {"native_module": self._text(module_name)}
        self._error(path, "implementation source is unavailable; supply an inspectable estimator or callable",
                    _SourceUnavailable)

    def _class(self, cls, path):
        if cls not in self.class_records:
            # Source-only scans may resolve the same file for several ordinary
            # MRO entries. Keep every is_file check and discard resolutions
            # before graph traversal can invoke another user callback.
            previous = self._mro_resolutions
            self._mro_resolutions = {} if (self.sources_only or self.collect_sources) and _ordinary_source_mro(cls) else None
            try:
                sources = []
                for parent in cls.__mro__:
                    sources.append({"type": _qualified(parent),
                                    "source": self._source(parent, path)})
                self.class_records[cls] = {"type": _qualified(cls), "implementations": sources}
            finally:
                self._mro_resolutions = previous
        return self.class_records[cls]

    def _custom_void_class(self, cls, path):
        # A factory can create classes with identical source and names but
        # different field accessors. NumPy also accepts the class as a dtype
        # specifier before any array exists.
        self.unverified_owner = True
        self._limit("unverified dtype")
        self._limit("unverified fit RNG")
        try:
            return self._class(cls, path)
        except _SourceUnavailable:
            # A fresh fit can use an interactive class without a source file.
            return {"type": _qualified(cls)}

    def _check_dtype_owners(self, dtype):
        # NumPy copies can retain these Python objects. A descriptor need not
        # expose their state (StringDType, for example, uses a sentinel's repr).
        opaque = type(dtype) is _STRING_DTYPE and not _plain_dtype_value(getattr(dtype, "na_object", None))
        if dtype.metadata:
            self.unverified_owner = True
            opaque |= any(not _plain_dtype_value(key) or not _plain_dtype_value(value)
                          for key, value in dtype.metadata.items())
        if opaque:
            self.unverified_owner = True
            self._limit("unverified dtype")
            self._limit("unverified fit RNG")

    def _string_dtype(self, dtype, path):
        record = {"str": "StringDType", "coerce": dtype.coerce}
        try:
            sentinel = dtype.na_object
        except AttributeError:
            return record
        if not _plain_dtype_value(sentinel):
            return None
        # NumPy's descriptor uses repr, which loses NaN bits and can fail
        # for large integers. Preserve byte sentinels in fold observations too.
        setting = self.encode(sentinel, path + ".na_object")
        if self.sources_only and type(sentinel) is bytes:
            setting["sha256"] = hashlib.sha256(sentinel).hexdigest()
        record["na_object"] = setting
        return record

    def _dtype(self, dtype, path):
        dtype = np.dtype(dtype)
        if self.split_limits is not None:
            self._check_dtype_owners(dtype)
        record = self._string_dtype(dtype, path) if type(dtype) is _STRING_DTYPE else None
        if dtype.names is not None:
            # descr flattens field dtypes, losing their scalar types, alignment
            # and metadata. Encode titles and field dtypes once in field order.
            fields = []
            record = {"str": self._text(dtype.str), "itemsize": dtype.itemsize,
                      "aligned": dtype.isalignedstruct, "alignment": dtype.alignment, "fields": fields,
                      "field_keys": [self._text(key) for key in dtype.fields]}
            end = 0
            for index, name in enumerate(dtype.names):
                field = dtype.fields[name]
                field_dtype, offset = field[:2]
                if offset < end:
                    self._error(path, "structured dtypes with overlapping or out-of-order fields are unsupported")
                end = offset + field_dtype.itemsize
                if (self.split_limits is not None and
                        (type(name) is not str or len(field) > 2 and not _plain_dtype_value(field[2]))):
                    self.unverified_owner = True
                    self._limit("unverified dtype")
                    self._limit("unverified fit RNG")
                field_path = f"{path}.fields[{index}]"
                entry = {"name": name if type(name) is str else self.encode(name, field_path + ".name"),
                         "offset": offset}
                if len(field) > 2:
                    entry["title"] = self.encode(field[2], field_path + ".title")
                entry["dtype"] = self._dtype(field_dtype, field_path + ".dtype")
                fields.append(entry)
        elif record is None:
            record = {"str": self._text(dtype.str)}
            # For an ordinary scalar or subarray, descr just wraps dtype.str.
            # Keep extension descriptors that can carry additional settings.
            if (type(dtype) is _STRING_DTYPE or dtype.isbuiltin == 2
                    or type(dtype).descr is not np.dtype.descr):
                record["descr"] = self.encode(dtype.descr, path + ".descr")
        # The normalized byte-format string can hide the native scalar class
        # and explicit byte order. Array fields retain their descriptor flags.
        record["format"] = [self._text(dtype.char), self._text(dtype.byteorder), dtype.isbuiltin, dtype.flags]
        if dtype.type is np.record:
            # Even a record dtype without fields yields a different scalar type.
            record["scalar_type"] = "numpy.record"
        elif dtype.kind == "V" and dtype.type is not np.void:
            record["scalar_type"] = self._custom_void_class(dtype.type, path + ".type")
        elif dtype.type is not np.void and dtype.names is not None:
            record["scalar_type"] = self._class(dtype.type, path + ".type")
        if dtype.metadata is not None:
            record["metadata"] = self.encode(dict(dtype.metadata), path + ".metadata")
        if dtype.subdtype:
            # For a standalone subarray dtype, descr reports only its width.
            base, shape = dtype.subdtype
            record["subarray"] = {"base": self._dtype(base, path + ".base"), "shape": list(shape)}
        return record

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
                        "source": {"native_module": self._text(module)}}
        self._error(path, "opaque native callable has no stable supported library identity")

    def encode(self, value, path="configuration", *, array_record=False):
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
            return {"kind": "int", "value": _integer_text(value)}
        if type(value) is float:
            return {"kind": "float", "bits": struct.pack("!d", value).hex()}
        if type(value) is complex:
            return {"kind": "complex", "real": self.encode(value.real, path + ".real"),
                    "imag": self.encode(value.imag, path + ".imag")}
        if isinstance(value, PurePath):
            return {"kind": "path", "type": _qualified(type(value)), "value": self._text(str(value))}
        if isinstance(value, np.dtype):
            return {"kind": "dtype", **self._dtype(value, path)}
        if isinstance(value, type):
            # Classes such as np.float32 must not be mistaken for instances
            # exposing an unbound get_params method.
            if issubclass(value, np.void) and value is not np.void and value is not np.record:
                return {"kind": "class", **self._custom_void_class(value, path)}
            return {"kind": "class", **self._class(value, path)}

        marker = id(value)
        if marker in self.active:
            self._error(path, "cyclic parameter graphs are unsupported")
        self.active.add(marker)
        previous = self.in_copy_graph
        if (type(value) in (functools.partial, slice) or issubclass(type(value), np.void)
                or type(value) is np.ndarray and value.dtype.hasobject):
            # sklearn deep-copies each such parameter in one operation. Its
            # mutable children can keep aliases across all of its fields.
            self.in_copy_graph = True
        try:
            record = self._compound(value, path, array_record)
            if not self.in_rng_state:
                if self.fit_rng_dependencies is not None and type(value) in (list, dict, set):
                    self.fit_rng_dependencies.append((value, self.in_function))
                if ((self.in_function or self.in_copy_graph)
                        and type(value) in (list, tuple, dict, set, frozenset, np.ndarray,
                                            np.void, np.record, slice, functools.partial, types.FunctionType)):
                    self._record_alias(value, record, "shared_container")
            return record
        finally:
            self.in_copy_graph = previous
            self.active.remove(marker)

    def _fit_rng(self, value, path):
        legacy = type(value) is np.random.RandomState
        engine = value._bit_generator if legacy else value.bit_generator
        if legacy:
            record = {"kind": "random_state", "state": self._rng_state(value.get_state(), path + ".state")}
        else:
            record = {"kind": "generator", "bit_generator": self._class(type(engine), path),
                      "state": self._rng_state(engine.state, path + ".state")}
        if self.sources_only and not self.observe_rng and self.split_limits is None:
            return record
        if self.in_function:
            self._limit("captured fit RNG")
        if engine is np.random.get_bit_generator():
            record["ambient_fit_rng"] = "singleton" if value is np.random.mtrand._rand else "engine"
            self._limit("ambient fit RNG")
        if (issubclass(type(self.split_seed), np.random.RandomState)
                and engine is np.random.RandomState._bit_generator.__get__(self.split_seed)):
            record["split_rng"] = "object" if value is self.split_seed else "engine"
            self._limit("shared split RNG")
        self.fit_rng_count += 1
        # NumPy deepcopy can preserve RNG, engine and SeedSequence sharing.
        # Bind references by traversal order so fresh processes can compare them.
        if self.fit_rng_objects is None:
            self.fit_rng_objects = {}
        shared = []
        sequence = (engine.seed_seq if _COPIES_SEED_SEQUENCE
                    and any(type(engine) is kind for kind in _STOCK_ENGINES) else None)
        if self.fit_rng_dependencies is not None:
            # Even a lineage ineligible for recovery can be used by a fresh
            # model after an archived callback. Its pool is still a dependency.
            dependency = sequence
            if not _COPIES_SEED_SEQUENCE and any(type(engine) is kind for kind in _STOCK_ENGINES):
                dependency = getattr(engine, "_seed_seq", None)
            if issubclass(type(dependency), np.random.SeedSequence):
                pool = np.random.SeedSequence.pool.__get__(dependency)
                self.fit_rng_dependencies.append((pool, self.in_function))
        for part in (value, engine, sequence):
            if part is None:
                continue
            if self.fit_rng_dependencies is not None:
                self.fit_rng_dependencies.append((part, self.in_function))
            key = id(part)
            if key in self.fit_rng_objects:
                shared.append(self.fit_rng_objects[key][1])
            else:
                self.fit_rng_objects[key] = part, len(self.fit_rng_objects)
                shared.append(None)
        if any(k is not None for k in shared):
            record["shared_fit_rng"] = shared
        lineage = _child_stream_state(engine)
        if lineage is _UNKNOWN_LINEAGE:
            self._limit("unverified fit RNG")
            record["child_stream"] = "unverified"
        elif lineage is not None:
            record["child_stream"] = lineage
            if self.rng_pool_owners is None:
                self.rng_pool_owners = {}
            pool = sequence.pool
            self.rng_pool_owners.setdefault(id(pool), (pool, self.fit_rng_count))
        return record

    def _finish_rng_pools(self):
        pools = self.rng_pool_owners or {}
        for buffer, record, captured in self.external_seed_buffers or ():
            value = buffer if captured else buffer()
            if value is None:
                continue
            exact = pools.get(id(value))
            if exact is not None:
                record["fit_rng_pool_owner"] = exact[1]
                if captured:
                    self._limit("captured fit RNG")
            elif captured:
                scalar_views = self.scalar_storage_ids
                linked = scalar_views is not None and id(value.base) in scalar_views
                if linked:
                    self._record_alias(value.base, record, "shared_buffer_owner")
                    record["shared_buffer_offset"] = value.ctypes.data - value.base.ctypes.data
                record["captured_layout"] = {
                    "strides": list(value.strides), "writeable": value.flags.writeable,
                    "aligned": value.flags.aligned,
                }
                if value.base is None and value.flags.owndata:
                    continue
                direct = pools.get(id(value.base))
                if direct is not None:
                    pool, owner = direct
                    self._limit("captured fit RNG")
                    record["fit_rng_pool_view"] = {"owner": owner, "offset": value.ctypes.data - pool.ctypes.data}
                    if self.fit_rng_dependencies is not None:
                        self.fit_rng_dependencies.append((pool, True))
                else:
                    # The function can inspect or change any part of its base,
                    # including elements outside the recorded array's bounds.
                    record["captured_buffer_owner"] = "external"
                    self.unverified_owner = True
                    if scalar_views is not None and not linked:
                        # A memoryview or another exporter may hide sharing
                        # with a captured scalar, even for identical bytes.
                        self._limit("unverified scalar owner")
                    if self.fit_rng_dependencies is not None and type(value.base) is np.ndarray:
                        self.fit_rng_dependencies.append((value.base, True))
        if self.unverified_owner:
            self._limit("unverified captured state")
            if self.fit_rng_count or self.split_limits and "implicit RNG" in self.split_limits:
                self._limit("unverified fit RNG")

    def _rng_state(self, state, path):
        # A saved RNG state includes numeric buffers (e.g. MT19937's keys).
        # Compare those bytes even when the fold check skips ordinary buffers.
        sources_only, in_state = self.sources_only, self.in_rng_state
        if self.observe_rng:
            self.sources_only = False
        self.in_rng_state = True
        try:
            return self.encode(state, path)
        finally:
            self.sources_only, self.in_rng_state = sources_only, in_state

    def _scalar_storage(self, scalar):
        # Bind only ordinary views into a small, contiguous NumPy allocation.
        # A scalar may expose every sibling in that allocation via its base.
        if self.sources_only and not self.observe_rng and self.split_limits is None:
            return None
        chain = []
        array = scalar.base
        while (type(array) is np.ndarray and len(chain) < _SCALAR_OWNER_MAX_ARRAYS and not scalar.dtype.hasobject
               and (array.dtype is scalar.dtype or any(array.dtype is dt for dt in _OWNER_NUMBER_DTYPES))):
            chain.append(array)
            if array.base is None:
                break
            array = array.base
        if (not chain or chain[-1].base is not None or not chain[-1].flags.owndata
                or not (chain[-1].flags.c_contiguous or chain[-1].flags.f_contiguous)
                or not 0 < chain[-1].nbytes <= _SCALAR_OWNER_MAX_BYTES):
            self._limit("unverified scalar owner")
            return "unverified"
        root = chain[-1]
        start = root.ctypes.data
        offset = np.asarray(scalar).ctypes.data - start
        if not 0 <= offset <= root.nbytes - scalar.itemsize:
            self._limit("unverified scalar owner")
            return "unverified"
        views = []
        for array in chain:
            origin = array.ctypes.data - start
            steps = [(n - 1) * stride for n, stride in zip(array.shape, array.strides)]
            if (array.size == 0 or origin + sum(min(0, step) for step in steps) < 0
                    or origin + sum(max(0, step) for step in steps) + array.itemsize > root.nbytes):
                self._limit("unverified scalar owner")
                return "unverified"
            view = {"offset": origin, "shape": list(array.shape),
                    "strides": list(array.strides), "flags": array.flags.num,
                    "dtype": "scalar" if array.dtype is scalar.dtype else self._dtype(array.dtype, "scalar.base.dtype")}
            self._record_alias(array, view, "shared_container")
            self.scalar_storage_ids.add(id(array))
            # Link the allocation to any separately declared seed pool.
            if self.external_seed_buffers is None:
                self.external_seed_buffers = []
            self.external_seed_buffers.append((array, view, True))
            if self.fit_rng_dependencies is not None:
                self.fit_rng_dependencies.append((array, True))
            views.append(view)
        content = None
        if not self.sources_only or self.observe_rng:
            content = hashlib.sha256(root.tobytes(order="A")).hexdigest()
        return {"offset": offset, "views": views, "sha256": content}

    def _compound(self, value, path, array_record):
        if isinstance(value, np.generic):
            if type(value) is not value.dtype.type:
                self._error(path, "NumPy scalar subclass state is not represented by ordinary scalar values")
            if value.dtype.hasobject:
                content = self.encode(value.item(), path + ".item")
            else:
                content = None if self.sources_only else hashlib.sha256(value.tobytes()).hexdigest()
            record = {"kind": "numpy_scalar", "dtype": self._dtype(value.dtype, path + ".dtype"),
                      "content": content}
            if self.in_function and type(value) in (np.void, np.record):
                # A cloned function retains the scalar and can read its NumPy flags.
                record["scalar_flags"] = value.flags.num
                if not array_record:
                    self._limit("captured scalar storage")
                    if self.scalar_storage_ids is None:
                        self.scalar_storage_ids = set()
            if (self.fit_rng_dependencies is not None and not self.in_rng_state
                    and type(value) in (np.void, np.record) and value.dtype.names is not None):
                # A structured scalar's bytes can still alias a different
                # model's pool. Cloning copies their value at fit entry.
                self.fit_rng_dependencies.append((np.asarray(value), self.in_function))
            if (issubclass(type(value), np.void) and type(value) is not np.void
                    and (type(value) is not np.record or vars(value))):
                record["scalar_attributes"] = "unverified"
                self._limit("unverified dtype")
                self._limit("unverified captured state")
                self._limit("unverified fit RNG")
            if self.in_function and value.base is not None:
                record["captured_scalar_owner"] = "external"
                self.unverified_owner = True
                if type(value) in (np.void, np.record) and not array_record:
                    record["scalar_storage"] = self._scalar_storage(value)
                if self.fit_rng_dependencies is not None and type(value.base) is np.ndarray:
                    self.fit_rng_dependencies.append((value.base, True))
            return record
        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray:
                self._error(path, "ndarray subclass state is not represented by ordinary array values")
            if self.fit_rng_dependencies is not None and not self.in_rng_state:
                # A different model can own the pool backing this array.
                self.fit_rng_dependencies.append((value, self.in_function))
            if value.dtype.hasobject:
                if ((not self.sources_only or self.observe_rng) and type(value.dtype) is _STRING_DTYPE
                        and type(getattr(value.dtype, "na_object", None)) is str):
                    # Scalar access returns the same text for a missing value
                    # and an equal literal. NumPy preserves the missing tag
                    # when converting between StringDTypes with sentinels.
                    # Bound temporary string storage for large parameters.
                    blocks = np.nditer(value, flags=("refs_ok", "external_loop", "buffered", "zerosize_ok"),
                                       order="C", buffersize=8192)
                    nullable = _STRING_DTYPE(na_object=None)
                    content = [self.encode(word) for block in blocks for word in block.astype(nullable)]
                else:
                    # A record array creates a scalar on indexing. An object
                    # array can have named fields and still return the stored
                    # object itself. Only skip storage for a generated record.
                    scalar_type = value.dtype.type
                    array_records = scalar_type is np.void or scalar_type is np.record
                    content = [self.encode(item, f"{path}[{index}]", array_record=array_records)
                               for index, item in enumerate(value.flat)]
            elif self.sources_only:
                content = None
            else:
                content = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            record = {"kind": "ndarray", "dtype": self._dtype(value.dtype, path + ".dtype"),
                      "shape": list(value.shape), "content": content}
            if (not self.in_rng_state and (self.in_function or value.ndim == 1 and value.dtype == np.uint32)
                    and (not self.sources_only or self.observe_rng or self.split_limits is not None)):
                if self.external_seed_buffers is None:
                    self.external_seed_buffers = []
                # A captured get_params can return a transient view that must
                # survive until the remaining closure cells have been visited.
                buffer = value if self.in_function else weakref.ref(value)
                self.external_seed_buffers.append((buffer, record, self.in_function))
            return record
        if isinstance(value, slice):
            return {"kind": "slice", "start": self.encode(value.start, path + ".start"),
                    "stop": self.encode(value.stop, path + ".stop"),
                    "step": self.encode(value.step, path + ".step")}
        if type(value) is dict:
            if self.split_limits is not None and "implicit RNG" not in self.split_limits:
                for key, setting in value.items():
                    if setting is not None or not issubclass(type(key), str):
                        continue
                    # A str subclass can change how set_params interprets its name.
                    if type(key) is not str or key.rsplit("__", 1)[-1] in _RNG_NAMES:
                        self.split_limits.append("implicit RNG")
                        break
            entries = [[self.encode(key, f"{path}.items[{index}].key"),
                        self.encode(item, f"{path}.items[{index}].value")]
                       for index, (key, item) in enumerate(value.items())]
            # Insertion order is part of Python dict semantics. A dict-valued
            # transformer parameter may intentionally specify ordered steps.
            return {"kind": "dict", "items": entries}
        if type(value) in (list, tuple):
            return {"kind": self._text(type(value).__name__),
                    "items": [self.encode(item, f"{path}[{index}]") for index, item in enumerate(value)]}
        if type(value) in (set, frozenset):
            before = self.fit_rng_count
            items = [self.encode(item, path + ".item") for item in value]
            if (self.fit_rng_count != before and self.split_limits is not None
                    and "unverified fit RNG" not in self.split_limits):
                self.split_limits.append("unverified fit RNG")
            record = {"kind": self._text(type(value).__name__), "items": sorted(items, key=_canonical)}
            if self.in_function or self.in_copy_graph:
                # Reference numbers assigned during unordered traversal cannot
                # bind captured objects across independent processes.
                record["retained_set_order"] = "unverified"
                self.unverified_owner = True
            return record
        if type(value) is np.random.RandomState or type(value) is np.random.Generator:
            return self._fit_rng(value, path)
        # Some RNG subclasses expose get_params and would otherwise be encoded
        # as ordinary estimators, omitting their mutable stream or spawn cursor.
        if (self.split_limits is not None and "unverified fit RNG" not in self.split_limits
                and issubclass(type(value), (np.random.RandomState, np.random.Generator,
                                             np.random.BitGenerator, np.random.SeedSequence))):
            self.split_limits.append("unverified fit RNG")
        if type(value) is functools.partial:
            if self.split_limits is not None and any(arg is None for arg in value.args):
                function = value.func
                names = (function.__code__.co_varnames[:function.__code__.co_argcount]
                         if type(function) is types.FunctionType else ())
                if any(arg is None and (i >= len(names) or names[i] in _RNG_NAMES)
                       for i, arg in enumerate(value.args)):
                    self._limit("implicit RNG")
            record = {"kind": "partial", "function": self.encode(value.func, path + ".func"),
                      "args": self.encode(value.args, path + ".args"),
                      "keywords": self.encode(value.keywords, path + ".keywords")}
            # The native reducer exposes the optional instance namespace
            # without creating one on the caller's partial.
            if functools.partial.__reduce__(value)[2][3] is not None:
                record["callable_attributes"] = "unverified"
                self._limit("unverified captured state")
                self._limit("unverified fit RNG")
            return record
        if isinstance(value, types.FunctionType):
            previous = self.in_function
            self.in_function = True
            try:
                closure = []
                for index, (name, cell) in enumerate(zip(value.__code__.co_freevars, value.__closure__ or ())):
                    try:
                        content = cell.cell_contents
                    except ValueError:
                        self._error(path, "callable contains an empty closure cell")
                    if content is None and name in _RNG_NAMES:
                        self._limit("implicit RNG")
                    item = self.encode(content, f"{path}.closure[{index}]")
                    self._record_alias(cell, item, "shared_closure_cell")
                    if self.fit_rng_dependencies is not None:
                        self.fit_rng_dependencies.append((cell, True))
                    closure.append(item)
                defaults = value.__defaults__
                if self.split_limits is not None and type(defaults) is tuple and defaults:
                    end = value.__code__.co_argcount
                    names = value.__code__.co_varnames[max(0, end - len(defaults)):end]
                    if any(setting is None and name in _RNG_NAMES
                           for name, setting in zip(reversed(names), reversed(defaults))):
                        self._limit("implicit RNG")
                record = {"kind": "function", "name": _qualified(value),
                        "source": self._source(value, path),
                        "code_sha256": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
                        "defaults": self.encode(defaults, path + ".defaults"),
                        "kwdefaults": self.encode(value.__kwdefaults__, path + ".kwdefaults"),
                        "closure": closure}
                namespace = value.__dict__
                if type(namespace) is not dict or namespace:
                    record["callable_attributes"] = "unverified"
                    self._limit("unverified captured state")
                    self._limit("unverified fit RNG")
                return record
            finally:
                self.in_function = previous
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
            record = {"kind": "estimator", "implementation": self._class(type(value), path),
                      "parameters": self.encode(parameters, path + ".parameters")}
            # sklearn.clone also copies set_output settings, including on nested
            # steps and grid replacements. They can change what the next step fits.
            output_config = getattr(value, "_sklearn_output_config", None)
            if output_config is not None:
                record["output_config"] = self.encode(output_config, path + ".output_config")
                if self.output_libraries is not None and type(output_config) is dict:
                    mode = output_config.get("transform")
                    if type(mode) in (str, np.str_) and mode in ("pandas", "polars"):
                        self.output_libraries.add(str(mode))
            if self.in_function or self.in_copy_graph:
                # A retained or deep-copied estimator can expose fitted state
                # that an ordinary top-level sklearn clone would discard.
                record["retained_estimator_state"] = "unverified"
                self.unverified_owner = True
            return record
        self._error(path, f"unsupported opaque value of type {_qualified(type(value))}")


def configuration_identity(value, *, _nonstandard_text=None, _split_limits=None, _split_seed=None,
                           _fit_rng_dependencies=None, _source_records=None, _output_libraries=None):
    """Return a structural JSON-safe identity, rejecting opaque/cyclic values.

    Ordinary fitted estimator attributes are intentionally omitted because
    sklearn clones the constructor configuration and output settings before each CV fit. Custom
    cloning that can preserve fitted state is rejected instead of falsely
    treating its constructor parameters as a complete reusable identity.

    If requested, collect (object, retained_by_function) pairs for comparison
    readers checking the effects of an earlier skipped fit in this process.
    """
    encoder = _Encoder(nonstandard_text=_nonstandard_text, split_limits=_split_limits, split_seed=_split_seed,
                       fit_rng_dependencies=_fit_rng_dependencies, collect_sources=_source_records is not None,
                       output_libraries=_output_libraries)
    record = encoder.encode(value)
    encoder._finish_rng_pools()
    if _source_records is not None:
        _source_records.update(encoder.sources)
    return record


def identity_digest(value):
    """Hash :func:`configuration_identity` with a canonical JSON encoding."""
    return hashlib.sha256(_canonical(configuration_identity(value)).encode("utf-8")).hexdigest()


def _share_observed_record(observed, full):
    """Reuse equal encoded subtrees without mutating either input."""
    if observed == full:
        return full
    if type(observed) is dict and type(full) is dict and observed.keys() == full.keys():
        return {key: _share_observed_record(value, full[key]) for key, value in observed.items()}
    if type(observed) is list and type(full) is list and len(observed) == len(full):
        return [_share_observed_record(value, other) for value, other in zip(observed, full)]
    return observed


def _share_calibration_observation(full_configuration, observation):
    # _search_identity encodes the ordered mapping {"estimator": ..., "candidates": ...}.
    # Both encoders have finished and checked for nonstandard text equality.
    # Their records are private snapshots: keep the
    # source scan's values wherever the full scan differed, including array hashes.
    for index, (_, full) in enumerate(full_configuration["items"]):
        observation[index] = _share_observed_record(observation[index], full)


def implementation_sources(*values, _previous_files=None, _observation=None, _nonstandard_text=None,
                           _split_limits=None, _split_seed=None):
    """Collect implementation file hashes through nested configurations.

    With an observation list, also save the structures produced by this scan,
    including exact NumPy RNG state. Ordinary numeric buffer contents are
    skipped. Every call reads the source files found by its own traversal.
    Runtime monkeypatching and mutable external/global dependencies are outside
    this contract; their behavior is not inferred from source file text.
    """
    encoder = _Encoder(sources_only=True, previous_files=_previous_files,
                       observe_rng=_observation is not None, nonstandard_text=_nonstandard_text,
                       split_limits=_split_limits, split_seed=_split_seed)
    for index, value in enumerate(values):
        record = encoder.encode(value, f"configuration[{index}]")
        if _observation is not None:
            _observation.append(record)
        del record  # A source-only caller can discard this value before scanning the next.
    encoder._finish_rng_pools()
    return dict(sorted(encoder.sources.items()))
