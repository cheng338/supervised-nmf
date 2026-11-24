"""Guarded reuse of Euclidean KNN queries in explicit calibration grids.

Author: Chang Cheng. This private adapter preserves the public estimator's
decision on uncertain batches; it never moves a voting threshold to break a tie.
"""

import sys
import time
import warnings
from abc import ABCMeta
from functools import lru_cache
from importlib.machinery import ModuleSpec, SourceFileLoader
from numbers import Integral, Real
from pathlib import Path
from types import CodeType, FunctionType, GetSetDescriptorType

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KDTree, BallTree
from sklearn import __version__ as _SKLEARN_VERSION, get_config

from ._labels import _label_values

try:
    from sklearn.metrics._dist_metrics import EuclideanDistance64 as _FULL_MEMBERSHIP_EUCLIDEAN
except ImportError:
    # Unsupported sklearn layouts retain their ordinary native route.
    _FULL_MEMBERSHIP_EUCLIDEAN = None


_QUERY_METHODS = frozenset(("kneighbors", "predict", "predict_proba", "_kneighbors_reduce_func"))
_VOTE_FUNCTIONS = frozenset(("_mode", "weighted_mode", "_get_weights", "_adjusted_metric",
                             "check_is_fitted", "validate_data", "_num_samples", "_check_precomputed",
                             "get_tags"))
_QUERY_DEFAULTS = {
    "KNeighborsMixin.kneighbors": (None, None, True),
    "ArgKmin.compute": ("euclidean", None, None, None, False),
    "ArgKminClassMode.compute": ("euclidean", None, None, None),
    "_mode": (0,),
    "_adjusted_metric": (None,),
    "check_is_fitted": (None,),
    "validate_data": ("no_validation", "no_validation", True, False, False),
    "Logger.__init__": (3, None),
    "ParallelBackendBase.__init__": (None, None),
    "ParallelBackendBase.configure": (1, None, None, None),
    "ThreadingBackend.configure": (1, None),
}
_QUERY_KEYWORDS = {"weighted_mode": {"axis": 0}, "check_is_fitted": {"msg": None, "all_or_any": all},
                   "gen_even_slices": {"n_samples": None}}
_SLICE_WRAPPER = "validate_params.<locals>.decorator.<locals>.wrapper"
_TREE_JOBLIB_METHODS = ("__init__", "__call__", "_initialize_backend", "_reset_run_tracking",
                        "_get_sequential_output", "_get_batch_size", "print_progress")
_TREE_CLASS_METADATA = {"__module__", "__doc__", "__firstlineno__", "__static_attributes__"}
_SKLEARN_PARALLEL_FIELDS = _TREE_CLASS_METADATA | {"__call__"}
_FULL_MEMBERSHIP_TREES = {
    "kd_tree": (KDTree, KDTree.query, KDTree.__getstate__),
    "ball_tree": (BallTree, BallTree.query, BallTree.__getstate__),
}


# Public Joblib contexts can execute user callbacks even with stock KNN.
# Inspect only native configuration and class dictionaries: do not resolve a
# backend by calling get_active_backend(), a factory, or a user descriptor.
from joblib import __version__ as _JOBLIB_CALLBACK_VERSION
from joblib import parallel as _callback_parallel
from sklearn.neighbors import _base as _callback_neighbors_base
from sklearn import utils as _sklearn_utils
from sklearn.utils import parallel as _tree_parallel

_CALLBACK_MISSING = object()
# The classifier can retain the generator after the optional utils re-export
# disappears. Its native prediction remains available in that configuration.
_tree_slices = vars(_sklearn_utils).get(
    "gen_even_slices", vars(_callback_neighbors_base).get("gen_even_slices", _CALLBACK_MISSING))


def _callback_static(owner, name):
    for base in owner.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return _CALLBACK_MISSING


_CALLBACK_JOBLIB_FUNCTIONS = ("effective_n_jobs", "get_active_backend", "_get_active_backend", "_get_config_param")
_CALLBACK_LOCAL = getattr(_callback_parallel, "_backend", None)
_CALLBACK_DEFAULTS = getattr(_callback_parallel, "default_parallel_config", None)
_CALLBACK_DEFAULT_KEYS = ("backend", "n_jobs", "prefer", "require", "verbose")
_CALLBACK_SENTINELS = ({key: _CALLBACK_DEFAULTS.get(key) for key in _CALLBACK_DEFAULT_KEYS}
                       if type(_CALLBACK_DEFAULTS) is dict else {})
_CALLBACK_CLASS_FIELDS = ("__init__", "__new__", "__getattribute__", "effective_n_jobs",
                          "in_main_thread", "uses_threads", "supports_sharedmem",
                          "__dict__", "nesting_level")
_CALLBACK_STOCK = {}
for _name in ("SequentialBackend", "ThreadingBackend", "LokyBackend", "MultiprocessingBackend"):
    _owner = getattr(_callback_parallel, _name, None)
    if isinstance(_owner, type):
        _CALLBACK_STOCK[_owner] = tuple(_callback_static(_owner, name) for name in _CALLBACK_CLASS_FIELDS)
_TREE_BACKEND_CONFIGURES = tuple((owner, _callback_static(owner, "configure")) for owner in _CALLBACK_STOCK)


def _plain_tree_dispatch():
    """Recognize stock Joblib tree dispatch without invoking backend factories."""
    if (vars(_callback_neighbors_base).get("Parallel") is not vars(_tree_parallel).get("Parallel", _CALLBACK_MISSING)
            or vars(_callback_neighbors_base).get("delayed") is not vars(_tree_parallel).get("delayed", _CALLBACK_MISSING)
            or _CALLBACK_LOCAL is None or type(_CALLBACK_DEFAULTS) is not dict
            or getattr(_callback_parallel, "_backend", None) is not _CALLBACK_LOCAL
            or getattr(_callback_parallel, "default_parallel_config", None) is not _CALLBACK_DEFAULTS
            or getattr(_CALLBACK_LOCAL, "config", _CALLBACK_DEFAULTS) is not _CALLBACK_DEFAULTS):
        return False
    # Parallel(prefer="threads") can construct both the default and threading
    # backends. A registered factory may have observable effects at construction.
    registry = getattr(_callback_parallel, "BACKENDS", None)
    default = getattr(_callback_parallel, "DEFAULT_BACKEND", None)
    threaded = getattr(_callback_parallel, "DEFAULT_THREAD_BACKEND", None)
    if (type(registry) is not dict or type(default) is not str or default not in {"loky", "threading"}
            or type(threaded) is not str or threaded != "threading"):
        return False
    # Another thread can register a backend while predictions are running.
    registry = registry.copy()
    if any(type(name) is not str for name in registry):
        return False
    providers = vars(_callback_parallel)
    threading_backend, loky_backend = registry.get("threading"), registry.get("loky")
    return (threading_backend is not None and threading_backend is providers.get("ThreadingBackend")
            and (default == "threading" or
                 (loky_backend is not None and loky_backend is providers.get("LokyBackend"))))


def _older_serial_brute(model):
    """Admit the separately qualified older pair only in its stock serial brute domain."""
    state = object.__getattribute__(model, "__dict__")
    jobs, method = state.get("n_jobs"), state.get("_fit_method")
    if (not (jobs is None or (type(jobs) is int and jobs == 1))
            or type(method) is not str or method != "brute"
            or _tree_slices is _CALLBACK_MISSING
            or vars(_sklearn_utils).get("gen_even_slices", _CALLBACK_MISSING) is not _tree_slices
            or vars(_callback_neighbors_base).get("gen_even_slices") is not _tree_slices
            or not _plain_tree_dispatch()
            or not all(_callback_static(owner, "configure") is configure
                       for owner, configure in _TREE_BACKEND_CONFIGURES)):
        return False
    # The existing native callback guard checks stock Joblib resolvers,
    # backend methods and default sentinels for n_jobs=None. Under ordinary
    # defaults, stock effective_n_jobs(None) resolves to one.
    config = get_config()
    return (config.get("enable_cython_pairwise_dist") is True
            and config.get("array_api_dispatch") is False
            and config.get("assume_finite") is False)


@lru_cache(maxsize=12)
def _installed_query_code(module):
    """Read installed query definitions without executing a module or a custom loader."""
    namespace = vars(module)
    spec = namespace.get("__spec__")
    filename = namespace.get("__file__")
    if type(spec) is not ModuleSpec or type(filename) is not str:
        return {}
    origin = vars(spec).get("origin")
    if type(origin) is not str or origin != filename:
        return {}
    try:
        path = Path(filename)
        if not path.is_absolute() or path.suffix != ".py" or not path.is_file():
            return {}
        loader = namespace.get("__loader__")
        loader_path = vars(loader).get("path") if type(loader) is SourceFileLoader else None
        if type(loader_path) is str and loader_path == filename:
            # A standard loader instance can also have per-instance hooks.
            # Reuse the normal bytecode cache without invoking those hooks.
            code = SourceFileLoader(module.__name__, filename).get_code(module.__name__)
        else:
            # Profilers and import hooks can wrap an ordinary source loader.
            # Read its declared file directly without calling the wrapper.
            code = compile(path.read_bytes(), filename, "exec", dont_inherit=True)
        definitions = [item for item in code.co_consts if type(item) is CodeType]
        names = _QUERY_METHODS | {"compute", "valid_metrics", "is_usable_for", "__sklearn_tags__",
                                  "_get_tags", "_more_tags", *_TREE_JOBLIB_METHODS,
                                  "with_config", "with_config_and_warning_filters",
                                  "configure", "effective_n_jobs", "compute_batch_size"}
        methods = {method.co_qualname: method for owner in definitions for method in owner.co_consts
                   if type(method) is CodeType and method.co_name in names}
        functions = _VOTE_FUNCTIONS | {"effective_n_jobs", "get_active_backend", "_get_active_backend",
                                       "_get_config_param", "get_config", "delayed", "gen_even_slices",
                                       "validate_params", "_with_config", "_with_config_and_warning_filters",
                                       "_get_threadlocal_config", "config_context", "set_config", "memstr_to_bytes"}
        methods.update((item.co_name, item) for item in definitions if item.co_name in functions)
        # The slice dispatcher is decorated. Its public callable and its
        # underlying generator have separate installed definitions.
        nested = methods.get("validate_params")
        for name in ("decorator", "wrapper"):
            if nested is None:
                break
            nested = next((item for item in nested.co_consts
                           if type(item) is CodeType and item.co_name == name), None)
        if nested is not None:
            methods[nested.co_qualname] = nested
        return methods
    except (ImportError, OSError, EOFError, AttributeError, TypeError, ValueError, SyntaxError):
        return {}


_QUERY_BINDINGS = {}


def _capture_query_function(function, module, name, slots, functions):
    if (type(function) is not FunctionType or function.__globals__ is not vars(module)
            or function.__code__ != _installed_query_code(module).get(name)):
        return False
    defaults, expected = function.__defaults__, _QUERY_DEFAULTS.get(name)
    if name == "set_config":
        expected = (None,) * function.__code__.co_argcount
    if expected is None:
        if defaults is not None:
            return False
    elif (type(defaults) is not tuple or len(defaults) != len(expected)
          or any(type(value) is not type(want) or value != want for value, want in zip(defaults, expected))):
        return False
    keywords, expected = function.__kwdefaults__, _QUERY_KEYWORDS.get(name)
    if name == "config_context":
        code = function.__code__
        expected = dict.fromkeys(code.co_varnames[code.co_argcount:code.co_argcount + code.co_kwonlyargcount])
    if expected is None:
        if keywords is not None:
            return False
    else:
        if (type(keywords) is not dict or len(keywords) != len(expected)
                or any(keywords.get(key, _CALLBACK_MISSING) is not value for key, value in expected.items())):
            return False
        slots.extend((keywords, key, value) for key, value in expected.items())
    functions.append((function, function.__code__, defaults, keywords))
    return True


def _stock_tree_state(state):
    classes, constraint_cell, wrapper, raw, interval_type = state
    wrapper_fields, raw_fields = vars(wrapper), vars(raw)
    for namespace, allowed in classes:
        if any(type(key) is not str for key in namespace) or not namespace.keys() <= allowed:
            return False
    if (type(wrapper_fields) is not dict or type(raw_fields) is not dict
            or any(type(key) is not str for ns in (wrapper_fields, raw_fields) for key in ns)
            or wrapper_fields.get("__wrapped__") is not raw
            or any(name in raw_fields for name in ("__signature__", "__wrapped__", "__text_signature__"))):
        return False
    constraints = constraint_cell.cell_contents
    if (type(constraints) is not dict or any(type(key) is not str for key in constraints)
            or constraints.keys() != {"n", "n_packs", "n_samples"}):
        return False
    for name, choices in constraints.items():
        optional = name == "n_samples"
        if (type(choices) is not list or len(choices) != 1 + optional
                or (optional and choices[1] is not None) or type(choices[0]) is not interval_type):
            return False
        fields = object.__getattribute__(choices[0], "__dict__")
        if (type(fields) is not dict or any(type(key) is not str for key in fields)
                or fields.keys() != {"type", "left", "right", "closed", "hidden"}
                or fields["type"] is not Integral or fields["hidden"] is not False
                or type(fields["left"]) is not int or fields["left"] != 1 or fields["right"] is not None
                or type(fields["closed"]) is not str or fields["closed"] != "left"):
            return False
    return True


def _capture_dispatch_methods(module, class_name, owner, names, slots, functions):
    """Capture a dispatcher class and its installed method definitions."""
    if type(owner) is not type or vars(module).get(class_name) is not owner:
        return False
    slots.append((vars(module), class_name, owner))
    namespace = vars(owner)
    for name in names:
        method = namespace.get(name)
        if not _capture_query_function(method, module, class_name + "." + name, slots, functions):
            return False
        slots.append((namespace, name, method))
    return True


def _capture_serial_backends(slots, functions, hierarchies, cells):
    """Recognize the backend methods called by an ordinary one-worker tree query."""
    from joblib import _parallel_backends as backends

    module = vars(backends)
    installed = _installed_query_code(backends)
    base, pool, auto = (module[name] for name in ("ParallelBackendBase", "PoolManagerMixin", "AutoBatchingMixin"))
    # Lookup on the concrete class and each mixin matters: a subclass can
    # override an inherited method before the adapter first sees the backend.
    route = (
        ("ParallelBackendBase", (object,), ("__init__", "configure", "compute_batch_size")),
        ("PoolManagerMixin", (object,), ("__init__", "effective_n_jobs")),
        ("AutoBatchingMixin", (object,), ("__init__",)),
        ("LokyBackend", (auto, base), ("__init__", "effective_n_jobs")),
        ("ThreadingBackend", (pool, base), ("__init__", "configure", "effective_n_jobs")),
        ("SequentialBackend", (base,), ("__init__", "configure", "effective_n_jobs", "compute_batch_size")),
        ("FallbackToBackend", (Exception,), ("__init__",)),
    )
    for class_name, bases, methods in route:
        owner = module[class_name]
        metaclass = ABCMeta if class_name == "ParallelBackendBase" or any(parent is base for parent in bases) else type
        if type(owner) is not metaclass or owner.__bases__ != bases:
            return False
        slots.append((module, class_name, owner))
        if class_name in ("LokyBackend", "ThreadingBackend", "SequentialBackend", "FallbackToBackend"):
            if vars(_callback_parallel).get(class_name) is not owner:
                return False
            slots.append((vars(_callback_parallel), class_name, owner))
        hierarchies.append((owner, owner.__mro__))
        namespace = vars(owner)
        for name in methods:
            descriptor = namespace.get(name, _CALLBACK_MISSING)
            key = class_name + "." + name
            if key in installed:
                if not _capture_query_function(descriptor, backends, key, slots, functions):
                    return False
                if descriptor.__code__.co_freevars == ("__class__",):
                    cells.append((descriptor.__closure__[0], owner))
            elif descriptor is not _CALLBACK_MISSING:
                return False
            slots.append((namespace, name, descriptor))
        # Construction, thread preference, and the fallback exception use
        # ordinary attribute access. A descriptor here could select a pool.
        for name in ("__new__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__",
                     "uses_threads", "supports_sharedmem", "nesting_level"):
            expected = _CALLBACK_MISSING
            if (name in ("uses_threads", "supports_sharedmem")
                    and class_name in ("ThreadingBackend", "SequentialBackend")):
                expected = True
            elif name == "nesting_level" and owner is base:
                expected = None
            if namespace.get(name, _CALLBACK_MISSING) is not expected:
                return False
            slots.append((namespace, name, expected))
    return True


def _capture_parallel_delegates(owner, slots, functions, hierarchies, cells):
    # These are the inherited constructor, the super().__call__ target and its
    # ordinary serial dispatch methods. Recognize their installed code before caching.
    from joblib import logger as joblib_logger

    config = vars(_callback_parallel).get("default_parallel_config")
    logger = vars(_callback_parallel).get("Logger")
    if (type(owner) is not type or type(config) is not dict or type(logger) is not type
            or len(owner.__mro__) != 3 or owner.__mro__[1] is not logger or owner.__mro__[2] is not object
            or not _capture_dispatch_methods(joblib_logger, "Logger", logger, ("__init__",), slots, functions)):
        return False
    hierarchies.append((owner, owner.__mro__))
    # Construction and assignment rely on object's native special methods.
    for ancestor in (owner, logger):
        for name in ("__new__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__del__"):
            if name in vars(ancestor):
                return False
            slots.append((vars(ancestor), name, _CALLBACK_MISSING))
    ns = vars(owner)
    for name in _TREE_JOBLIB_METHODS:
        fn = ns.get(name)
        key = "Parallel." + name
        if name != "__init__":
            if not _capture_query_function(fn, _callback_parallel, key, slots, functions):
                return False
        else:
            if (type(fn) is not FunctionType or fn.__globals__ is not vars(_callback_parallel)
                    or fn.__code__ != _installed_query_code(_callback_parallel).get(key)
                    or fn.__kwdefaults__ is not None or fn.__code__.co_freevars != ("__class__",)):
                return False
            parameters = fn.__code__.co_varnames[1:fn.__code__.co_argcount]
            literals = {"return_as": "list", "timeout": None, "pre_dispatch": "2 * n_jobs", "batch_size": "auto"}
            defaults = fn.__defaults__
            if type(defaults) is not tuple or len(defaults) != len(parameters):
                return False
            for field, value in zip(parameters, defaults):
                if field in literals:
                    if type(value) is not type(literals[field]) or value != literals[field]:
                        return False
                elif config.get(field, _CALLBACK_MISSING) is value:
                    slots.append((config, field, value))
                else:
                    return False
            cells.append((fn.__closure__[0], owner))
            functions.append((fn, fn.__code__, defaults, None))
        slots.append((ns, name, fn))
    return True


def _capture_tree_bindings(slots, functions, hierarchies, cells):
    """Recognize the three direct callables in the installed tree query expression."""
    import contextlib
    import joblib
    from joblib import disk
    from sklearn import _config, utils
    from sklearn.utils import _chunking, _param_validation

    caller = vars(_callback_neighbors_base)
    for provider, name, aliases in (
        (_tree_parallel, "delayed", (caller,)),
        (_tree_parallel, "Parallel", (caller,)),
        (_chunking, "gen_even_slices", (vars(utils), caller)),
    ):
        namespace = vars(provider)
        target = namespace.get(name)
        if target is None or any(alias.get(name) is not target for alias in aliases):
            return False
        slots.extend((owner, name, target) for owner in (namespace, *aliases))

    for source, name, aliases in (
        (_config, "get_config", (_tree_parallel, _param_validation)),
        (_config, "_get_threadlocal_config", ()), (_config, "set_config", ()),
        (disk, "memstr_to_bytes", (_callback_parallel,)),
    ):
        target = vars(source).get(name)
        if (not _capture_query_function(target, source, name, slots, functions)
                or any(vars(module).get(name) is not target for module in aliases)):
            return False
        slots.extend((vars(module), name, target) for module in (source, *aliases))
    # The emitted task and the slice validator enter the same sklearn context
    # through contextlib's ordinary wrapper, which calls its captured generator.
    context = vars(_config).get("config_context")
    helper = next((code for code in contextlib.contextmanager.__code__.co_consts
                   if type(code) is CodeType and code.co_name == "helper"), None)
    if (type(context) is not FunctionType or context.__globals__ is not vars(contextlib)
            or context.__code__ is not helper or context.__code__.co_freevars != ("func",)
            or context.__defaults__ is not None or context.__kwdefaults__ is not None):
        return False
    raw = context.__closure__[0].cell_contents
    if not _capture_query_function(raw, _config, "config_context", slots, functions):
        return False
    for module in (_config, _tree_parallel, _param_validation):
        if vars(module).get("config_context") is not context:
            return False
        slots.append((vars(module), "config_context", context))
    functions.append((context, helper, None, None))
    cells.append((context.__closure__[0], raw))
    if (not _capture_serial_backends(slots, functions, hierarchies, cells)
            or not _capture_query_function(caller["delayed"], _tree_parallel, "delayed", slots, functions)):
        return False
    # delayed emits a callable task. Parallel attaches configuration to it and
    # Joblib invokes it with the selected width and return_distance flag.
    provider = vars(_tree_parallel)
    task = provider.get("_FuncWrapper")
    # Earlier sklearn releases only propagated the configuration to the task.
    config_method = ("with_config_and_warning_filters"
                     if "_with_config_and_warning_filters" in _installed_query_code(_tree_parallel) else "with_config")
    task_methods = ("__init__", config_method, "__call__")
    helper_name = "_" + config_method
    attach = provider.get(helper_name)
    if (type(task) is not type or task.__base__ is not object
            or not _capture_dispatch_methods(_tree_parallel, "_FuncWrapper", task, task_methods, slots, functions)
            or not _capture_query_function(attach, _tree_parallel, helper_name, slots, functions)):
        return False
    slots.append((provider, helper_name, attach))
    hierarchies.append((task, task.__mro__))
    task_namespace = vars(task)
    for name in ("__dict__", "__weakref__"):
        descriptor = task_namespace.get(name)
        if (type(descriptor) is not GetSetDescriptorType or descriptor.__objclass__ is not task
                or descriptor.__name__ != name):
            return False
        slots.append((task_namespace, name, descriptor))
    parallel = caller["Parallel"]
    if (type(parallel) is not type or len(parallel.__bases__) != 1
            or parallel.__base__ is not vars(_callback_parallel).get("Parallel")
            or parallel.__base__ is not vars(joblib).get("Parallel")
            or not _capture_parallel_delegates(parallel.__base__, slots, functions, hierarchies, cells)):
        return False
    slots.extend((vars(module), "Parallel", parallel.__base__) for module in (joblib, _callback_parallel))
    namespace = vars(parallel)
    call = namespace.get("__call__")
    if not _capture_query_function(call, _tree_parallel, "Parallel.__call__", slots, functions):
        return False
    hierarchies.append((parallel, parallel.__mro__))
    slots.append((namespace, "__call__", call))
    cells.append((call.__closure__[call.__code__.co_freevars.index("__class__")], parallel))

    slices = caller["gen_even_slices"]
    if not _capture_query_function(slices, _param_validation, _SLICE_WRAPPER, slots, functions):
        return False
    wrapper_fields = vars(slices)
    if type(wrapper_fields) is not dict or any(type(key) is not str for key in wrapper_fields):
        return False
    raw = wrapper_fields.get("__wrapped__")
    if not _capture_query_function(raw, _chunking, "gen_even_slices", slots, functions):
        return False
    # functools.wraps exposes metadata; the actual call uses the closure.
    if slices.__code__.co_freevars != ("func", "parameter_constraints", "prefer_skip_nested_validation"):
        return False
    generator, constraints, prefer_skip = slices.__closure__
    cells.extend(((generator, raw), (prefer_skip, True)))
    task_fields = _TREE_CLASS_METADATA | {"__dict__", "__weakref__", *task_methods}
    classes = ((namespace, _SKLEARN_PARALLEL_FIELDS), (task_namespace, task_fields))
    state = classes, constraints, slices, raw, vars(_param_validation)["Interval"]
    if all(cell.cell_contents is value for cell, value in cells) and _stock_tree_state(state):
        return state
    return False


def _capture_callback_bindings(slots, functions):
    # Validate hooks against the installed definitions. An import-time snapshot
    # alone can capture a callback a profiler or application already installed.
    for owner in KNeighborsClassifier.__mro__:
        if not owner.__module__.startswith("sklearn."):
            continue
        module = sys.modules[owner.__module__]
        code = _installed_query_code(module)
        names = ("__sklearn_tags__",)
        if _SKLEARN_VERSION.startswith("1.6."):
            names += ("_get_tags", "_more_tags")
        for name in names:
            namespace = vars(owner)
            function = namespace.get(name, _CALLBACK_MISSING)
            key = owner.__name__ + "." + name
            if function is not _CALLBACK_MISSING or key in code:
                if not _capture_query_function(function, module, key, slots, functions):
                    return False
            slots.append((namespace, name, function))
        if "__sklearn_is_fitted__" in vars(owner):
            return False
        slots.append((vars(owner), "__sklearn_is_fitted__", _CALLBACK_MISSING))

    for name in _CALLBACK_JOBLIB_FUNCTIONS:
        namespace = vars(_callback_parallel)
        function = namespace.get(name)
        if (type(function) is not FunctionType or function.__globals__ is not namespace
                or function.__code__ != _installed_query_code(_callback_parallel).get(name)
                or function.__kwdefaults__ is not None):
            return False
        defaults = function.__defaults__
        if name == "effective_n_jobs":
            if (type(defaults) is not tuple or len(defaults) != 1
                    or type(defaults[0]) is not int or defaults[0] != -1
                    or vars(_callback_neighbors_base).get(name) is not function):
                return False
            slots.append((vars(_callback_neighbors_base), name, function))
        elif name == "_get_config_param":
            if defaults is not None:
                return False
        elif (type(defaults) is not tuple or len(defaults) != 3 or any(
                value is not _CALLBACK_SENTINELS.get(key)
                for value, key in zip(defaults, ("prefer", "require", "verbose")))):
            return False
        slots.append((namespace, name, function))
        functions.append((function, function.__code__, defaults, None))
    return True


def _capture_query_bindings(brute):
    """Capture raw lookup slots only after comparing them with the installed definitions."""
    from sklearn.neighbors import _classification
    from sklearn.utils import _tags, arrayfuncs, extmath, fixes, validation

    if vars(_classification).get("KNeighborsClassifier") is not KNeighborsClassifier:
        return None
    methods = [(KNeighborsClassifier, _callback_neighbors_base, "KNeighborsMixin", "kneighbors"),
               (KNeighborsClassifier, _callback_neighbors_base, "KNeighborsMixin", "_kneighbors_reduce_func"),
               (KNeighborsClassifier, _classification, "KNeighborsClassifier", "predict"),
               (KNeighborsClassifier, _classification, "KNeighborsClassifier", "predict_proba")]
    slots, functions = [], []
    if not _capture_callback_bindings(slots, functions):
        return None
    hierarchies = [(KNeighborsClassifier, KNeighborsClassifier.__mro__)]
    cells = []
    tree_state = None if brute else _capture_tree_bindings(slots, functions, hierarchies, cells)
    if not brute and not tree_state:
        return None
    # These are the validation and voting functions that the public classifier
    # calls directly. Reuse can omit them even when its three methods are stock.
    neighbors_base = _callback_neighbors_base
    for module, names, aliases in (
        (fixes, ("_mode",), (_classification,)),
        (extmath, ("weighted_mode",), (_classification,)),
        (neighbors_base, ("_get_weights", "_check_precomputed"), (_classification,)),
        (_classification, ("_adjusted_metric",), ()),
        (validation, ("_num_samples",), (_classification,)),
        (validation, ("check_is_fitted", "validate_data"), (_classification, neighbors_base)),
        (_tags, ("get_tags",), (neighbors_base,)),
        (arrayfuncs, ("_all_with_any_reduction_axis_1",), (_classification,)),
    ):
        for name in names:
            function = vars(module).get(name)
            if module is arrayfuncs:
                if (type(function) is not type(vars(arrayfuncs).get("min_pos"))
                        or type(function).__name__ != "fused_cython_function"
                        or function.__name__ != name
                        or function.__module__ not in (module.__name__, type(function).__module__)):
                    return None
            elif not _capture_query_function(function, module, name, slots, functions):
                return None
            for owner in (module, *aliases):
                namespace = vars(owner)
                if namespace.get(name) is not function:
                    return None
                slots.append((namespace, name, function))
    if brute:
        from sklearn.metrics._pairwise_distances_reduction import _argkmin, _argkmin_classmode, _dispatcher
        from sklearn import _config

        if (vars(_dispatcher).get("get_config") is not get_config
                or not _capture_query_function(get_config, _config, "get_config", slots, functions)):
            return None
        slots.append((vars(_dispatcher), "get_config", get_config))

        base = vars(_dispatcher).get("BaseDistancesReductionDispatcher")
        if type(base) is not type:
            return None
        slots.append((vars(_dispatcher), "BaseDistancesReductionDispatcher", base))
        for host, name, extension in ((_callback_neighbors_base, "ArgKmin", _argkmin),
                                      (_classification, "ArgKminClassMode", _argkmin_classmode)):
            dispatch = vars(_dispatcher).get(name)
            if (type(dispatch) is not type or dispatch.__bases__ != (base,)
                    or vars(host).get(name) is not dispatch):
                return None
            hierarchies.append((dispatch, dispatch.__mro__))
            slots.extend(((vars(host), name, dispatch), (vars(_dispatcher), name, dispatch)))
            # The Python dispatcher calls these dtype-specific extension types
            # directly. Py_TPFLAGS_IMMUTABLETYPE guarantees their compute
            # descriptors cannot be rebound on the class.
            for suffix in ("32", "64"):
                kernel_name = name + suffix
                kernel = vars(extension).get(kernel_name)
                if (type(kernel) is not type or not (kernel.__flags__ & (1 << 8))
                        or kernel.__name__ != kernel_name or kernel.__module__ != extension.__name__
                        or type(vars(kernel).get("compute")) is not classmethod
                        or vars(_dispatcher).get(kernel_name) is not kernel):
                    return None
                slots.append((vars(_dispatcher), kernel_name, kernel))
            metrics_owner = name if name == "ArgKminClassMode" else "BaseDistancesReductionDispatcher"
            methods.extend(((dispatch, _dispatcher, name, "compute"),
                            (dispatch, _dispatcher, "BaseDistancesReductionDispatcher", "is_usable_for"),
                            (dispatch, _dispatcher, metrics_owner, "valid_metrics")))

    for receiver, module, owner_name, name in methods:
        owner = vars(module).get(owner_name)
        for ancestor in receiver.__mro__:
            namespace = vars(ancestor)
            descriptor = namespace.get(name, _CALLBACK_MISSING)
            slots.append((namespace, name, descriptor))
            if ancestor is owner:
                break
            if descriptor is not _CALLBACK_MISSING:
                return None
        else:
            return None
        function = descriptor
        if receiver is not KNeighborsClassifier:
            if type(descriptor) is not classmethod:
                return None
            function = descriptor.__func__
        qualified_name = owner_name + "." + name
        if not _capture_query_function(function, module, qualified_name, slots, functions):
            return None
    return slots, functions, hierarchies, cells, tree_state


def _native_query_methods(model, shared_model):
    """Recheck live public and dispatcher bindings before sharing any selected-k query."""
    if type(model) is not KNeighborsClassifier or type(shared_model) is not KNeighborsClassifier:
        return False
    state, shared_state = vars(model), vars(shared_model)
    backend, shared_backend = state.get("_fit_method"), shared_state.get("_fit_method")
    if (type(backend) is not str or type(shared_backend) is not str or backend != shared_backend
            or any(name in state or name in shared_state for name in _QUERY_METHODS)):
        return False
    brute = backend == "brute"
    bindings = _QUERY_BINDINGS.get(brute)
    if bindings is None:
        try:
            bindings = _capture_query_bindings(brute)
        except (ImportError, AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            return False
        if bindings is None:
            # A pre-import override may be restored between predictions.
            return False
        _QUERY_BINDINGS[brute] = bindings
    slots, functions, hierarchies, cells, tree_state = bindings
    try:
        if (any(cell.cell_contents is not value for cell, value in cells)
                or (tree_state is not None and not _stock_tree_state(tree_state))):
            return False
    except (IndexError, KeyError, RuntimeError, ValueError):  # emptied cell or concurrently changed state
        return False
    return (all(owner.__mro__ is mro for owner, mro in hierarchies)
            and all(namespace.get(name, _CALLBACK_MISSING) is value for namespace, name, value in slots)
            and all(function.__code__ is code and function.__defaults__ is defaults
                    and function.__kwdefaults__ is keywords
                    for function, code, defaults, keywords in functions))


def _native_callback_context(model, *, full_membership=True, verified_serial_tree=False):
    """Permit proof only when omitted native calls contain no user callbacks."""
    try:
        if (type(model) is not KNeighborsClassifier
                or vars(_callback_neighbors_base).get("effective_n_jobs") is not _callback_parallel.effective_n_jobs
                or "__sklearn_tags__" in model.__dict__
                or "__sklearn_is_fitted__" in model.__dict__):
            return False
        if full_membership:
            pair = (_SKLEARN_VERSION, _JOBLIB_CALLBACK_VERSION)
            if pair != ("1.9.1", "1.6.0") and (pair != ("1.9.0", "1.5.3") or not _older_serial_brute(model)):
                return False
        jobs = model.n_jobs
        if type(jobs) is int and jobs == 1:
            # Original effective_n_jobs returns here without backend lookup.
            return True
        if not (jobs is None or (type(jobs) is int and jobs != 0)):
            return False
        if (_callback_parallel._backend is not _CALLBACK_LOCAL
                or _callback_parallel.default_parallel_config is not _CALLBACK_DEFAULTS
                or type(_CALLBACK_DEFAULTS) is not dict):
            return False
        config = getattr(_CALLBACK_LOCAL, "config", _CALLBACK_DEFAULTS)
        if type(config) is not dict:
            return False
        current = {}
        for key in _CALLBACK_DEFAULT_KEYS:
            sentinel = _CALLBACK_SENTINELS[key]
            if _CALLBACK_DEFAULTS.get(key, _CALLBACK_MISSING) is not sentinel:
                return False
            value = config.get(key, _CALLBACK_MISSING)
            if value is _CALLBACK_MISSING:
                return False
            current[key] = (0 if key == "verbose" else None) if value is sentinel else value
        # Non-default backend hints can choose another factory or inspect its
        # flags. Retain the original route rather than emulate that resolver.
        if current["prefer"] is not None or current["require"] is not None:
            return False
        if type(current["verbose"]) is not int:
            return False
        if jobs is None:
            jobs = current["n_jobs"]
            if not (jobs is None or (type(jobs) is int and jobs != 0)):
                return False
        backend = current["backend"]
        if verified_serial_tree and backend is None:
            # The tree binding check has positively identified the current
            # backend methods, including ones restored after package import.
            return True
        if backend is None:
            registry = _callback_parallel.BACKENDS
            name = _callback_parallel.DEFAULT_BACKEND
            if (type(registry) is not dict or type(name) is not str
                    or any(type(key) is not str for key in registry)):
                return False
            factory = registry.get(name)
            if not any(factory is owner for owner in _CALLBACK_STOCK):
                return False
            owner = factory
            state = None
        else:
            owner = type(backend)
            if not any(owner is known for known in _CALLBACK_STOCK):
                return False
        originals = _CALLBACK_STOCK[owner]
        if any(_callback_static(owner, name) is not original
               for name, original in zip(_CALLBACK_CLASS_FIELDS, originals)):
            return False
        if backend is not None:
            # Class descriptors are still the captured stock objects before
            # reading the plain instance dictionary. Unknown class hashing,
            # properties and custom factories are never invoked for admission.
            state = object.__getattribute__(backend, "__dict__")
            if type(state) is not dict:
                return False
            if any(name in state for name in ("effective_n_jobs", "in_main_thread", "__getattribute__")):
                return False
            nesting = state.get("nesting_level", None)
            if not (nesting is None or (type(nesting) is int and nesting >= 0)):
                return False
            for name in ("uses_threads", "supports_sharedmem"):
                if name in state and type(state[name]) is not bool:
                    return False
        return True
    except Exception:
        return False


def _is_integer_index(value):
    # NumPy durations register as Integral and can compare equal to integers.
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_, np.timedelta64))


def _has_euclidean_metric(model):
    metric = getattr(model, "effective_metric_", None)
    params = getattr(model, "effective_metric_params_", None)
    return type(metric) is str and metric == "euclidean" and type(params) is dict and not params


class _KNNQueryReuse:
    """Share neighbor order only among fitted heads with identical training data.

    ``models`` contains successful, independently fitted heads indexed by k.
    Unknown sklearn state, sparse inputs, callable weights and other metrics use
    the public selected-k estimator. A query error or any uncertain row also
    falls back for the whole original query batch, preserving its kernel shape.
    Private fitted-state use is deliberately confined to this adapter.
    """

    def __init__(self, models, query, *, _study_call_contract=None, _head_fit_input=None,
                 _query_values=None, _native_input_validation=False):
        self.models = models
        self.query = query
        # Only the study can identify the array it just passed to every head's fit.
        self._fresh_kd_coordinates = _head_fit_input if type(_head_fit_input) is np.ndarray else None
        # Sparse matrices deliberately reject len(); their unsupported path
        # must still reach the ordinary public estimator.
        values = query if _query_values is None else _query_values
        self.query_rows = values.shape[0] if hasattr(values, "shape") else len(values)
        self.groups = []
        self.by_k = {}
        self.query_seconds = 0.0
        for k, model in models.items():
            reason = ("native_pipeline_input_validation" if _native_input_validation
                      else self._unsupported(k, model, query))
            if reason is not None:
                self.by_k[k] = (None, reason)
                continue
            group = next((group for group in self.groups
                          if self._same_geometry(group["model"], model)), None)
            if group is None:
                group = {"model": model, "ks": [], "queried": False,
                         "key": len(self.groups), "warnings": [],
                         "query_seconds": 0.0}
                self.groups.append(group)
            group["ks"].append(k)
            if k > group["model"].n_neighbors:
                group["model"] = model
            self.by_k[k] = (group, None)
        if _study_call_contract is not None:
            self._install_study_contract(_study_call_contract)

    @staticmethod
    def _unsupported(k, model, query):
        if type(model) is not KNeighborsClassifier:
            return "unsupported_estimator"
        weights = getattr(model, "weights", None)
        if type(weights) is not str or weights not in {"uniform", "distance"}:
            return "callable_or_unsupported_weights"
        method = getattr(model, "_fit_method", None)
        if type(method) is not str or method not in {"brute", "kd_tree", "ball_tree"}:
            return "unknown_neighbor_backend"
        if not _has_euclidean_metric(model):
            return "non_euclidean_metric"
        train = getattr(model, "_fit_X", None)
        classes = getattr(model, "classes_", None)
        encoded = getattr(model, "_y", None)
        if (not isinstance(train, np.ndarray) or not isinstance(query, np.ndarray)
                or train.ndim != 2 or query.ndim != 2
                or train.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or query.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not isinstance(classes, np.ndarray) or classes.ndim != 1
                or not isinstance(encoded, np.ndarray) or encoded.ndim != 1
                or len(encoded) != len(train) or query.shape[1] != train.shape[1]
                or not len(query)):
            return "unsupported_fitted_state_or_array"
        if not _is_integer_index(model.n_neighbors) or not 1 <= model.n_neighbors <= len(train):
            return "unsupported_neighbor_count"
        if not _is_integer_index(k) or k != model.n_neighbors:
            return "neighbor_key_mismatch"
        return None

    @staticmethod
    def _same_geometry(left, right):
        return (left._fit_X is right._fit_X
                and left._fit_method == right._fit_method
                and left.weights == right.weights
                and np.array_equal(_label_values(left.classes_), _label_values(right.classes_))
                and np.array_equal(left._y, right._y))

    def _query(self, group):
        if group["queried"]:
            return
        group["queried"] = True
        model = group["model"]
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                count = min(len(model._fit_X), max(group["ks"]) + 1)
                group["distances"], group["indices"] = model.kneighbors(
                    self.query, n_neighbors=count)
            except Exception as error:
                # An optimization error must not suppress a valid smaller-k
                # candidate. Each public selected-k call is still attempted.
                group["error_type"] = type(error).__name__
                group["error"] = str(error)
            finally:
                group["query_seconds"] = time.perf_counter() - start
                self.query_seconds += group["query_seconds"]
        group["warnings"] = list(caught)

    def _prefix_votes(self, group):
        # The caller admits only ordinary policy and finite native distances.
        # Keep all per-head guards; this only avoids building unused prefixes.
        if "prefix_votes" not in group and not group.get("prefix_unavailable", False):
            try:
                model = group["model"]
                distances = group["distances"]
                epsilon = np.finfo(distances.dtype).eps
                ks = []
                selected_ks = group["ks"]
                if group.get("study_short_query_active", False):
                    selected_ks = [value for value in selected_ks
                                   if value < len(model._fit_X)]
                for selected_k in sorted(selected_ks):
                    if selected_k == len(model._fit_X):
                        ks.append(selected_k)
                    else:
                        left, right = distances[:, selected_k - 1], distances[:, selected_k]
                        radius = np.maximum(np.abs(left), np.abs(right))
                        near_boundary = np.abs(right - left) <= 64.0 * epsilon * radius
                        if not near_boundary.any():
                            ks.append(selected_k)
                if len(ks) < 2:
                    group["prefix_unavailable"] = True
                    group["prefix_skip_reason"] = "fewer_than_two_boundary_clean_heads"
                    return None
                maximum = ks[-1]
                distance = distances[:, :maximum]
                # Sorted native distances put zero neighbors first; this
                # max-prefix mask equals the original mask at every positive k.
                if model.weights == "uniform":
                    weights = np.ones_like(distance)
                else:
                    zero = distance == 0
                    with np.errstate(divide="ignore", over="ignore"):
                        weights = 1.0 / distance
                    zero_rows = zero.any(axis=1)
                    weights[zero_rows] = zero[zero_rows]
                encoded = model._y[group["indices"][:, :maximum]]
                all_votes = np.empty((len(ks), self.query_rows, len(model.classes_)),
                                     dtype=np.float64)
                starts = np.asarray([0] + ks[:-1], dtype=np.intp)
                # Sorted unique positive ks fill every prefix exactly when H=K.
                complete_prefix = len(ks) == maximum
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for j in range(len(model.classes_)):
                        masked = weights * (encoded == j)
                        if complete_prefix:
                            prefix = np.cumsum(masked, axis=1, dtype=np.float64)
                        else:
                            intervals = np.add.reduceat(masked, starts, axis=1,
                                                        dtype=np.float64)
                            prefix = np.cumsum(intervals, axis=1, dtype=np.float64)
                        for position, selected_k in enumerate(ks):
                            all_votes[position, :, j] = prefix[:, position]
                # Publish only complete selected-k rows, after all classes.
                group["prefix_votes"] = {selected_k: all_votes[position]
                                         for position, selected_k in enumerate(ks)}
            except Exception as error:
                # Internal diagnostics only; optional failure uses the original
                # selected-k calculation and never exposes partial votes.
                group["prefix_unavailable"] = True
                group["prefix_error_type"] = type(error).__name__
                group["prefix_error"] = str(error)
        return group.get("prefix_votes")

    def predict(self, k, *, probabilities=False, decision_threshold=None, positive_index=None):
        """Return predictions (or probabilities), diagnostics and shared warnings."""
        model = self.models[k]
        group, reason = self.by_k[k]
        # Dictionary lookup can match integer keys to booleans or floats.
        # Such a lookup value cannot safely index a neighbor array.
        if type(k) is not int and not _is_integer_index(k):
            group, reason = None, "neighbor_key_mismatch"
        study_plan = self._study_plan_for_call(
            group, k, probabilities, decision_threshold, positive_index)
        metadata = {"accelerated": False, "shared_query_attempted": False,
                    "public_fallback_query_rows": self.query_rows}
        if group is None or (len(group["ks"]) == 1 and study_plan is None):
            metadata["reason"] = reason or "single_head_backend"
            return self._public(model, probabilities), metadata, []
        if decision_threshold is not None:
            if not _is_integer_index(positive_index) or not 0 <= positive_index < len(model.classes_):
                raise IndexError("decision_threshold requires a valid positive class index.")
            if not isinstance(decision_threshold, Real) or isinstance(decision_threshold, np.timedelta64):
                raise TypeError("decision_threshold must be a real scalar.")
        if model._fit_method == "ball_tree":
            # Rounded pruning can depend on query width. Even a one-leaf or
            # full-width search runs the selected head's Joblib callbacks.
            metadata["reason"] = "selected_width_ball_tree"
            return self._public(model, probabilities), metadata, []
        shared = group["model"]
        if model._fit_method == "kd_tree":
            # A worker pool can schedule batches and run completion callbacks
            # for each selected query. Only the serial path is qualified.
            jobs = (vars(model).get("n_jobs"), vars(shared).get("n_jobs"))
            if (any(value is not None and (type(value) is not int or value != 1) for value in jobs)
                    or not _plain_tree_dispatch() or "query" in vars(KDTree)
                    or any(type(getattr(head, "_tree", None)) is not KDTree or "query" in getattr(head._tree, "__dict__", {})
                           for head in (model, shared))):
                metadata["reason"] = "selected_width_kd_tree_dispatch"
                return self._public(model, probabilities), metadata, []
            # Group members share _fit_X by identity. A prebuilt index can use
            # a different metric or different pruning bounds for those coordinates.
            if self._fresh_kd_coordinates is not model._fit_X:
                metadata["reason"] = "unverified_kd_tree_geometry"
                return self._public(model, probabilities), metadata, []
        if not _native_query_methods(model, shared):
            metadata["reason"] = "changed_neighbor_method"
            return self._public(model, probabilities), metadata, []
        serial_tree = model._fit_method == "kd_tree"
        if (not _native_callback_context(model, full_membership=study_plan is not None, verified_serial_tree=serial_tree)
                or (shared is not model and not _native_callback_context(
                    shared, full_membership=study_plan is not None, verified_serial_tree=serial_tree))):
            metadata["reason"] = "changed_neighbor_callback_context"
            return self._public(model, probabilities), metadata, []
        if model._fit_method == "brute" and (
                get_config().get("enable_cython_pairwise_dist") is not True
                or self.query.dtype != model._fit_X.dtype or not model._fit_X.flags.c_contiguous):
            # Chunked brute can execute Joblib tasks and the selected head's
            # reduction callback at a different width, even with n_jobs=1.
            metadata["reason"] = "selected_width_chunked_brute"
            return self._public(model, probabilities), metadata, []
        planned_full_votes = None
        if study_plan is None:
            self._query(group)
        else:
            planned_full_votes = self._study_query(group, study_plan, k, probabilities)
        metadata.update(shared_query_attempted=True, shared_query_key=group["key"],
                        backend=model._fit_method)
        if (study_plan is not None and study_plan["sole_full_head"]
                and planned_full_votes is None):
            # Baseline sole-head behavior is one public prediction call. Do
            # not add a full shared retry or duplicate its visible warnings.
            metadata["reason"] = "single_head_backend"
            return self._public(model, probabilities), metadata, []
        if "error" in group:
            metadata["reason"] = "shared_query_failed"
            return self._public(model, probabilities), metadata, group["warnings"]
        distances, indices = group["distances"], group["indices"]
        # The shared query arrays are privately owned and immutable for this
        # adapter's lifetime. Qualify the complete max-k-plus-one matrix once.
        if "invalid_distance" not in group:
            group["invalid_distance"] = ~np.isfinite(distances).all(axis=1)
        invalid_distance = group["invalid_distance"]
        if invalid_distance.any():
            # Even finite input can overflow distance arithmetic. In particular
            # inf - inf is NaN and would evade the ordinary boundary-tie guard.
            # Do not qualify cached neighbor order in this state.
            metadata.update(reason="nonfinite_neighbor_distances",
                            invalid_distance_rows=int(invalid_distance.sum()))
            return self._public(model, probabilities), metadata, group["warnings"]
        # Reordering qualification is permitted only under the ordinary
        # NumPy policy: strict underflow/error paths retain their original order.
        # The public selected-k call still receives the entire original batch.
        early_boundary = None
        threshold_is_valid = (
            decision_threshold is None
            or (type(decision_threshold) is float
                and 0.0 <= decision_threshold <= 1.0))
        if (k < len(model._fit_X) and threshold_is_valid
                and np.geterr() == {"divide": "warn", "over": "warn",
                                    "under": "ignore", "invalid": "warn"}):
            cached_prefix = group.get("prefix_votes")
            if cached_prefix is not None and k in cached_prefix:
                # Successful publication already proved this exact k clean on
                # the same immutable distances. The earlier finite-distance
                # guard established this private mask is all false; the tail
                # only reads it, and creates a fresh uncertainty union.
                early_boundary = invalid_distance
            else:
                boundary_epsilon = np.finfo(distances.dtype).eps
                left, right = distances[:, k - 1], distances[:, k]
                radius = np.maximum(np.abs(left), np.abs(right))
                early_boundary = np.abs(right - left) <= 64.0 * boundary_epsilon * radius
            if early_boundary.any():
                boundary_rows = int(early_boundary.sum())
                metadata.update(reason="numerical_or_boundary_uncertainty",
                                near_boundary_rows=boundary_rows,
                                near_vote_rows=None, near_threshold_rows=None,
                                invalid_vote_rows=None, vote_guard=None,
                                uncertain_rows=None, vote_checks_performed=False,
                                uncertainty_lower_bound_rows=boundary_rows)
                return self._public(model, probabilities), metadata, group["warnings"]
        prefix_votes = None
        if (threshold_is_valid
                and np.geterr() == {"divide": "warn", "over": "warn",
                                    "under": "ignore", "invalid": "warn"}):
            prefix_votes = self._prefix_votes(group)
        early_votes = planned_full_votes
        if (early_votes is None and prefix_votes is None and "prefix_votes" not in group
                and group.get("prefix_skip_reason") == "fewer_than_two_boundary_clean_heads"
                and "prefix_error_type" not in group
                and type(k) is int and k == len(model._fit_X)
                and type(model.weights) is str and model.weights == "uniform"):
            early_votes = self._full_membership_row_votes(
                group, model, k, probabilities=probabilities,
                threshold_is_valid=threshold_is_valid)
        if early_votes is not None:
            votes = early_votes
            metadata["full_membership_vote_basis"] = (
                "short_query_training_histogram" if planned_full_votes is not None
                else "native_query_training_histogram")
        elif prefix_votes is None:
            distance = distances[:, :k]
            if model.weights == "uniform":
                vote_weights = np.ones_like(distance)
            else:
                zero = distance == 0
                with np.errstate(divide="ignore", over="ignore"):
                    vote_weights = 1.0 / distance
                zero_rows = zero.any(axis=1)
                vote_weights[zero_rows] = zero[zero_rows]
            encoded = model._y[indices[:, :k]]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                votes = np.column_stack([
                    np.sum(vote_weights * (encoded == j), axis=1, dtype=np.float64)
                    for j in range(len(model.classes_))
                ])
        else:
            votes = prefix_votes[k]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            totals = votes.sum(axis=1, keepdims=True)
            probability = votes / totals
        epsilon = np.finfo(distances.dtype).eps
        product = (k + len(model.classes_) + 4) * epsilon
        gamma = product / (1.0 - product) if product < .25 else np.inf
        # A conservative forward-error guard covers differing nonnegative vote
        # reductions. It selects the public calculation, never a new decision.
        guard = 32.0 * gamma
        invalid = (~np.isfinite(probability).all(axis=1)
                   | ~np.isfinite(votes).all(axis=1)
                   | ~np.isfinite(totals[:, 0]) | (totals[:, 0] <= 0))
        if len(model.classes_) > 1:
            largest = np.sort(probability, axis=1)[:, -2:]
            margin = largest[:, 1] - largest[:, 0]
            near_vote = ~np.isfinite(margin) | (margin <= guard)
        else:
            margin = np.ones(self.query_rows)
            near_vote = np.zeros(self.query_rows, dtype=bool)
        near_threshold = np.zeros(self.query_rows, dtype=bool)
        if decision_threshold is not None:
            near_threshold = np.abs(probability[:, positive_index] - decision_threshold) <= guard
        if early_boundary is not None:
            near_boundary = early_boundary
        elif k == len(model._fit_X):
            near_boundary = np.zeros(self.query_rows, dtype=bool)
        else:
            left, right = distances[:, k - 1], distances[:, k]
            radius = np.maximum(np.abs(left), np.abs(right))
            near_boundary = np.abs(right - left) <= 64.0 * epsilon * radius
        uncertain = invalid | near_vote | near_threshold | near_boundary
        metadata.update(near_boundary_rows=int(near_boundary.sum()),
                        near_vote_rows=int(near_vote.sum()),
                        near_threshold_rows=int(near_threshold.sum()),
                        invalid_vote_rows=int(invalid.sum()),
                        vote_guard=float(guard), uncertain_rows=int(uncertain.sum()))
        if uncertain.any():
            if (type(k) is int and k == len(model._fit_X)
                    and type(model.weights) is str and model.weights == "uniform"
                    and not invalid.any()
                    and (early_votes is not None or self._full_membership_certified(
                        group, model, k, votes=votes, probabilities=probabilities,
                        threshold_is_valid=threshold_is_valid))):
                metadata.update(full_membership_certified=True,
                                full_membership_basis="native_training_histogram",
                                reason="uniform_full_membership_certified")
            else:
                metadata["reason"] = "numerical_or_boundary_uncertainty"
                return self._public(model, probabilities), metadata, group["warnings"]
        metadata.update(accelerated=True, public_fallback_query_rows=0)
        result = probability if probabilities else model.classes_[probability.argmax(axis=1)]
        return result, metadata, group["warnings"]

    def _full_membership_row_votes(self, group, model, k, *, probabilities,
                                   threshold_is_valid):
        """Use the proved stock full-query histogram when prefixes are unavailable."""
        # The original shared query and finite-distance validation still run.
        # Under the common stock-method/magnitude gates, full-k native queries
        # contain every training row. Private query arrays remain immutable.
        # Unknown contexts and optional allocation failures retain native fallback.
        if (type(probabilities) is not bool or not threshold_is_valid
                or group["warnings"]
                or np.geterr() != {"divide": "warn", "over": "warn",
                                  "under": "ignore", "invalid": "warn"}
                or np.geterrcall() is not None):
            return None
        try:
            indices, encoded, classes = group["indices"], model._y, model.classes_
            if (type(indices) is not np.ndarray or indices.dtype != np.dtype(np.intp)
                    or indices.shape != (self.query_rows, k)
                    or type(encoded) is not np.ndarray or encoded.ndim != 1
                    or encoded.dtype != np.dtype(np.intp) or len(encoded) != k
                    or type(classes) is not np.ndarray or classes.ndim != 1
                    or not 1 <= len(classes) <= k <= 2**53
                    or (encoded < 0).any() or (encoded >= len(classes)).any()):
                return None
            context = self._full_membership_context(
                group, model, k, probabilities=probabilities,
                threshold_is_valid=threshold_is_valid)
            if context is None:
                return None
            encoded, classes = context
            expected = np.bincount(encoded, minlength=len(classes))
            cell_count = self.query_rows * len(classes)
            if cell_count > np.iinfo(np.intp).max:
                return None
            # Stock native full-k membership proves these counts for each
            # query row; n<=2**53 makes their float64 conversion exact. Keep a
            # fresh owned matrix so no fitted state or result aliases a cache.
            return np.broadcast_to(
                expected, (self.query_rows, len(classes))).astype(np.float64)
        except Exception:
            # No warning-policy change or partial publication accompanies a
            # failed optional histogram broadcast.
            pass
        return None

    def _full_membership_certified(self, group, model, k, *, votes, probabilities,
                                   threshold_is_valid):
        """Certify existing votes without changing their validation order."""
        context = self._full_membership_context(
            group, model, k, probabilities=probabilities,
            threshold_is_valid=threshold_is_valid)
        if context is None:
            return False
        encoded, classes = context
        try:
            if (type(votes) is not np.ndarray or votes.dtype != np.dtype(np.float64)
                    or votes.shape != (self.query_rows, len(classes))):
                return False
            # The bounded native all-n query visits every training row. Its
            # uniform float64 histogram is exact for n<=2**53. Certify the
            # already computed votes, not the identities of shared indices.
            expected = np.bincount(encoded, minlength=len(classes))
            return bool((votes == expected).all())
        except Exception:
            # This optional proof must not prevent the original selected-k call.
            return False

    def _full_membership_context(self, group, model, k, *, probabilities,
                                 threshold_is_valid):
        """Validate the unchanged native all-training-row context."""
        if not _native_callback_context(model):
            return None
        if (type(k) is not int or type(probabilities) is not bool
                or type(model.weights) is not str or model.weights != "uniform"
                or not threshold_is_valid
                or type(model.metric) is not str
                or model.metric not in {"euclidean", "l2", "minkowski"}
                or not (type(model.p) is int or type(model.p) is float)
                or model.p != 2
                or not (model.metric_params is None
                        or (type(model.metric_params) is dict and not model.metric_params))):
            return None
        try:
            train, query = model._fit_X, self.query
            if (type(train) is not np.ndarray or type(query) is not np.ndarray
                    or train.dtype != query.dtype or train.ndim != 2 or query.ndim != 2
                    or train.shape[1] < 1 or query.shape[1] != train.shape[1]
                    or model.outputs_2d_ is not False
                    or train.shape[1] * np.finfo(np.float64).eps >= 0.25
                    or group["model"] is not model or group["warnings"]
                    or type(model.n_neighbors) is not int
                    or type(model.n_samples_fit_) is not int
                    or not k == model.n_neighbors == model.n_samples_fit_ == len(train)
                    or not 1 <= k <= 2**53):
                return None
            if (np.geterr() != {"divide": "warn", "over": "warn",
                              "under": "ignore", "invalid": "warn"}
                    or np.geterrcall() is not None):
                return None
            if not _native_query_methods(model, model):
                return None
            backend = model._fit_method
            if type(backend) is not str:
                return None
            if backend == "brute":
                if model._tree is not None:
                    return None
            else:
                tree_spec = _FULL_MEMBERSHIP_TREES.get(backend)
                tree = model._tree
                if (tree_spec is None or type(tree) is not tree_spec[0]
                        or type(tree).query is not tree_spec[1]
                        or "query" in getattr(tree, "__dict__", {})
                        or _FULL_MEMBERSHIP_EUCLIDEAN is None
                        or type(tree).__getstate__ is not tree_spec[2]):
                    return None
                # A public fit from a prefit neighbor estimator can copy a
                # tree with a different hidden metric. Inspect the original
                # state descriptor without invoking a custom metric callback.
                tree_state = tree_spec[2](tree)
                if (type(tree_state) is not tuple or len(tree_state) != 13
                        or type(tree_state[11]) is not _FULL_MEMBERSHIP_EUCLIDEAN):
                    return None
            encoded, classes = model._y, model.classes_
            if (type(encoded) is not np.ndarray or encoded.ndim != 1
                    or encoded.dtype != np.dtype(np.intp) or len(encoded) != k
                    or type(classes) is not np.ndarray or classes.ndim != 1
                    or not 1 <= len(classes) <= k
                    or (encoded < 0).any() or (encoded >= len(classes)).any()):
                return None
            # Native ClassMode uses direct differences, unlike the shared
            # Euclidean GEMM route. Keep their squared sums well below the
            # input-dtype limit, including the float32 metric return type.
            bound = float(np.sqrt(float(np.finfo(train.dtype).max)
                                  / (16.0 * train.shape[1])))
            if not (np.max(np.abs(train)) <= bound and np.max(np.abs(query)) <= bound):
                return None
            return encoded, classes
        except Exception:
            # This optional proof must not prevent the original selected-k call.
            return None

    def _install_study_contract(self, contract):
        """Admit only the operation-owned study's fixed, callback-free call loop."""
        try:
            if (type(contract) is not tuple or len(contract) != len(self.models)
                    or len(contract) < 1 or len(self.groups) != 1
                    or len(self.groups[0]["ks"]) != len(self.models)
                    or np.geterr() != {"divide": "warn", "over": "warn",
                                      "under": "ignore", "invalid": "warn"}
                    or np.geterrcall() is not None):
                return
            for row in contract:
                if (type(row) is not tuple or len(row) != 4 or type(row[0]) is not int
                        or type(row[1]) is not bool
                        or not (row[2] is None or (type(row[2]) is float
                                and 0.0 <= row[2] <= 1.0))
                        or not (row[3] is None or type(row[3]) is int)):
                    return
            if (tuple(row[0] for row in contract) != tuple(self.models)
                    or any(row[1:] != contract[0][1:] for row in contract)):
                return
            group = self.groups[0]
            full = group["model"]
            # Brute query width changes selection, not distance computation.
            # Full-n tree membership alone does not prove rounded pruning
            # equivalence for a shorter query; retain full queries for both tree routes.
            if type(full._fit_method) is not str or full._fit_method != "brute":
                return
            n = len(full._fit_X)
            if n not in self.models or self.models[n] is not full:
                return
            smaller = [row[0] for row in contract if row[0] < n]
            count = max(smaller) + 1 if smaller else 1
            if count >= n:
                return
            if not _native_query_methods(full, full):
                return
            for key, model in self.models.items():
                if (type(model) is not KNeighborsClassifier
                        or type(model.n_neighbors) is not int or model.n_neighbors != key
                        or type(model.weights) is not str or model.weights != "uniform"
                        or type(model.metric) is not str
                        or model.metric not in {"euclidean", "l2", "minkowski"}
                        or not (type(model.p) is int or type(model.p) is float) or model.p != 2
                        or not (model.metric_params is None or
                                (type(model.metric_params) is dict and not model.metric_params))
                        or not (model.n_jobs is None or (type(model.n_jobs) is int and model.n_jobs != 0))
                        or type(model._fit_method) is not str
                        or type(model.classes_) is not np.ndarray
                        or model.classes_.dtype.kind not in "biufUS"
                        or not _QUERY_METHODS.isdisjoint(vars(model))
                        or not _native_callback_context(model)):
                    return
                if model._fit_method != "brute":
                    tree_spec = _FULL_MEMBERSHIP_TREES.get(model._fit_method)
                    tree = model._tree
                    if (tree_spec is None or type(tree) is not tree_spec[0]
                            or type(tree).query is not tree_spec[1]
                            or "query" in getattr(tree, "__dict__", {})
                            or _FULL_MEMBERSHIP_EUCLIDEAN is None
                            or type(tree).__getstate__ is not tree_spec[2]):
                        return
                    state = tree_spec[2](tree)
                    if (type(state) is not tuple or len(state) != 13
                            or type(state[11]) is not _FULL_MEMBERSHIP_EUCLIDEAN):
                        return
            _, probabilities, threshold, positive_index = contract[0]
            if threshold is not None and not (
                    type(positive_index) is int and 0 <= positive_index < len(full.classes_)):
                return
            context = self._full_membership_context(
                group, full, n, probabilities=probabilities, threshold_is_valid=True)
            if context is None:
                return
            self._study_contract = contract
            self._study_position = 0
            self._study_plan = {"group": group, "full_k": n, "count": count,
                                "sole_full_head": not smaller}
        except Exception:
            # Optional planning never changes ordinary constructor errors.
            return

    def _study_plan_for_call(self, group, k, probabilities, threshold, positive_index):
        plan = getattr(self, "_study_plan", None)
        if plan is None:
            return None
        position = self._study_position
        expected = self._study_contract[position] if position < len(self._study_contract) else None
        actual = (k, probabilities, threshold, positive_index)
        same = (expected is not None
                and all(type(a) is type(b) for a, b in zip(actual, expected))
                and actual == expected)
        ordinary = (np.geterr() == {"divide": "warn", "over": "warn",
                                   "under": "ignore", "invalid": "warn"}
                    and np.geterrcall() is None)
        if not same or not ordinary:
            if plan["group"].get("study_short_query_active", False):
                # This private mode is supplied only by the fixed study loop.
                # Generic varying-policy/request adapters never enable it.
                raise RuntimeError("The operation-owned KNN call contract changed after query planning.")
            self._study_plan = None
            return None
        self._study_position += 1
        return plan if group is plan["group"] else None

    def _study_query(self, group, plan, k, probabilities):
        """Retain a real native query and charge every attempted query once."""
        if not group["queried"]:
            model = group["model"]
            error = None
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    distances, indices = model.kneighbors(self.query, n_neighbors=plan["count"])
                except Exception as failure:
                    error = failure
                finally:
                    elapsed = time.perf_counter() - start
                    self.query_seconds += elapsed
            group["short_query_attempt_seconds"] = elapsed
            group["shared_query_neighbor_counts"] = [plan["count"]]
            if error is not None or caught:
                if error is not None:
                    group["short_query_error_type"] = type(error).__name__
                    group["short_query_error"] = str(error)
                group["short_query_warning_records"] = [
                    {"category": item.category.__name__, "message": str(item.message)}
                    for item in caught]
                if plan["sole_full_head"]:
                    group.update(queried=True, query_seconds=elapsed, warnings=[],
                                 study_short_query_active=False,
                                 short_query_upgrade_reason="single_head_query_public_fallback")
                else:
                    self._query(group)
                    group["query_seconds"] += elapsed
                    group["shared_query_neighbor_counts"].append(plan["full_k"])
                    group["study_short_query_active"] = False
                    group["short_query_upgrade_reason"] = "short_query_warning_or_error"
            else:
                group.update(queried=True, distances=distances, indices=indices,
                             query_seconds=elapsed, warnings=[], study_short_query_active=True)
        if not group.get("study_short_query_active", False) or k != plan["full_k"]:
            return None
        # Preserve the original complete matrix check before producing votes.
        if "invalid_distance" not in group:
            group["invalid_distance"] = ~np.isfinite(group["distances"]).all(axis=1)
        if group["invalid_distance"].any():
            if plan["sole_full_head"]:
                group["study_short_query_active"] = False
                group["short_query_upgrade_reason"] = "single_head_nonfinite_public_fallback"
            return None
        votes = self._study_full_votes(group, plan, probabilities)
        if votes is not None:
            return votes
        if plan["sole_full_head"]:
            group["study_short_query_active"] = False
            group["short_query_upgrade_reason"] = "single_head_histogram_public_fallback"
            return None
        previous = group["query_seconds"]
        for key in ("distances", "indices", "invalid_distance", "prefix_votes",
                    "prefix_unavailable", "prefix_skip_reason", "prefix_error_type", "prefix_error",
                    "error_type", "error"):
            group.pop(key, None)
        group["queried"] = False
        group["study_short_query_active"] = False
        self._query(group)
        group["query_seconds"] += previous
        group["shared_query_neighbor_counts"].append(plan["full_k"])
        group["short_query_upgrade_reason"] = "full_histogram_not_available"
        return None

    def _study_full_votes(self, group, plan, probabilities):
        try:
            context = self._full_membership_context(
                group, group["model"], plan["full_k"],
                probabilities=probabilities, threshold_is_valid=True)
            if context is None:
                return None
            encoded, classes = context
            if self.query_rows * len(classes) > np.iinfo(np.intp).max:
                return None
            expected = np.bincount(encoded, minlength=len(classes))
            return np.broadcast_to(expected, (self.query_rows, len(classes))).astype(np.float64)
        except Exception:
            return None

    def _public(self, model, probabilities):
        return (model.predict_proba(self.query) if probabilities
                else model.predict(self.query))

    def query_records(self):
        """Return JSON-compatible shared-cost records, excluding query arrays."""
        records = []
        for group in self.groups:
            record = {key: value for key, value in group.items()
                      if key in {"key", "ks", "queried", "query_seconds", "error_type", "error",
                                 "study_short_query_active", "shared_query_neighbor_counts",
                                 "short_query_attempt_seconds", "short_query_warning_records",
                                 "short_query_error_type", "short_query_error",
                                 "short_query_upgrade_reason"}}
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in group["warnings"]
            ]
            records.append(record)
        return records
