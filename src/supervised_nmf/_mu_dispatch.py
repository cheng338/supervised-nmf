"""Select the fixed-basis MU solver at the actual scikit-learn transform boundary."""

from math import sqrt
import sys
from types import FunctionType, MethodType

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF


_QUALIFIED = {"1.6.1", "1.7.0", "1.7.1", "1.7.2", "1.8.0", "1.9.0", "1.9.1"}
QUALIFIED_VERSION = sklearn.__version__ in _QUALIFIED
_LEGACY_TAGS = sklearn.__version__ == "1.6.1"
if QUALIFIED_VERSION:
    from sklearn.decomposition import _nmf
    from sklearn.utils import _set_output, extmath, validation

    from ._mu_certificate import capture, qualification
    from ._mu_updates import joint


# NumPy moved multiarray in 2.0. Its original export can identify a caller's
# pre-import override without binding or executing the replacement.
_multiarray = sys.modules.get("numpy._core.multiarray") or sys.modules.get("numpy.core.multiarray")
_STOCK_DOT = vars(_multiarray).get("dot") if _multiarray is not None else None
_MISSING = object()
_METHODS = ("transform", "_fit_transform", "_check_params", "_check_w_h",
            "_compute_regularization", "__sklearn_is_fitted__")
_NUMERICS = ("_fit_multiplicative_update", "_multiplicative_update_w",
             "_beta_divergence", "_check_init", "_beta_loss_to_float")
_MODEL_KEYS = frozenset((
    "alpha_H", "alpha_W", "beta_loss", "components_", "init", "l1_ratio",
    "max_iter", "n_components", "n_components_", "n_features_in_", "n_iter_",
    "random_state", "reconstruction_err_", "shuffle", "solver", "tol", "verbose",
    "_n_components", "_beta_loss", "_sklearn_output_config",
))
_MODERN_TAGS = "__sklearn_tags__"


def stock_dot():
    """Check the live public binding after each caller-controlled boundary."""
    return _STOCK_DOT is not None and vars(np).get("dot") is _STOCK_DOT


def _defined_function(function, module, qualified_name):
    # functools.wraps copies __module__, __qualname__ and __wrapped__.
    return (type(function) is FunctionType and function.__globals__ is vars(module)
            and function.__code__.co_qualname == qualified_name)


def _defined_method(owner, name, method):
    if owner is NMF and name in ("_check_w_h", "_compute_regularization"):
        base = NMF.__mro__[1]
        if base.__qualname__ == "_BaseNMF" and vars(base).get(name) is method:
            owner = base
    if name == "transform":
        if not _defined_function(method, _set_output, "_wrap_method_output.<locals>.wrapped"):
            return False
        wrapped = getattr(method, "__wrapped__", None)
        closed = dict(zip(method.__code__.co_freevars, method.__closure__ or ()))
        if "f" not in closed or closed["f"].cell_contents is not wrapped:
            return False
        method = wrapped
    module = sys.modules.get(owner.__module__)
    return (module is not None and owner.__module__.startswith("sklearn.")
            and _defined_function(method, module, f"{owner.__qualname__}.{name}"))


def _original_bindings():
    """Capture raw dictionary slots, including absent and inherited methods."""
    bindings = []
    for name in _METHODS:
        found = False
        for owner in NMF.__mro__:
            namespace = vars(owner)
            method = namespace.get(name, _MISSING)
            bindings.append((namespace, name, method))
            if method is not _MISSING:
                found = True
                if not _defined_method(owner, name, method):
                    return None
        if not found and name != "__sklearn_is_fitted__":
            return None
    for name in _NUMERICS:
        method = vars(_nmf).get(name)
        if not _defined_function(method, _nmf, name):
            return None
        bindings.append((vars(_nmf), name, method))
    return tuple(bindings)


_INITIAL_TRANSFORM = vars(NMF).get("transform")
_TRUSTED_TRANSFORM = (_INITIAL_TRANSFORM if not QUALIFIED_VERSION
                      or _defined_method(NMF, "transform", _INITIAL_TRANSFORM) else _MISSING)
_BINDINGS = _original_bindings() if QUALIFIED_VERSION else None


def initial_or_restored_transform(method):
    """Recognize stock transform even after a temporary import-time override."""
    return (method is _TRUSTED_TRANSFORM
            or QUALIFIED_VERSION and _defined_method(NMF, "transform", method))


def _custom_solver(model):
    global _BINDINGS
    if not vars(model).keys().isdisjoint(_METHODS):
        return True
    if _BINDINGS is None:
        # An import-time descriptor may have since been removed.
        bindings = _original_bindings()
        if bindings is None:
            return True
        _BINDINGS = bindings
    return any(namespace.get(name, _MISSING) is not method for namespace, name, method in _BINDINGS)


def _custom_tags(model):
    if _MODERN_TAGS in vars(model):
        return True
    if _LEGACY_TAGS:
        for owner in NMF.__mro__:
            namespace = vars(owner)
            if _MODERN_TAGS not in namespace and ("_more_tags" in namespace or "_get_tags" in namespace):
                return True
    # These providers cooperate through super(). Check raw definitions to
    # leave descriptor binding and legacy warning order to scikit-learn.
    providers = iter((_nmf._BaseNMF, TransformerMixin, BaseEstimator))
    expected = next(providers)
    for owner in NMF.__mro__:
        method = vars(owner).get(_MODERN_TAGS, _MISSING)
        if owner is expected:
            if not _defined_method(owner, _MODERN_TAGS, method):
                return True
            if owner is BaseEstimator:
                return False
            expected = next(providers)
        elif method is not _MISSING and method is not vars(expected).get(_MODERN_TAGS):
            return True
    return True


_INTS = (int,) + tuple(np.dtype(code).type for code in np.typecodes["AllInteger"])
_REALS = _INTS + (float,) + tuple(np.dtype(code).type for code in np.typecodes["Float"])
_CONTROLS = {
    "solver": (str,), "init": (str, type(None)), "beta_loss": (str,) + _REALS,
    "alpha_W": _REALS, "alpha_H": (str,) + _REALS, "l1_ratio": _REALS,
    "tol": _REALS, "verbose": _INTS, "max_iter": _INTS, "n_components": _INTS,
    "_n_components": _INTS, "_beta_loss": _REALS,
}
_STATE_LOOKUPS = _CONTROLS.keys() | {"components_", "__getattribute__", "__setattr__"}
_INITIALIZERS = (None, "random", "custom", "nndsvda", "nndsvdar")


def _stock_inner(model, method):
    # Validation can have changed both receiver state and class methods.
    if (type(model) is not NMF or type(method) is not MethodType or method.__self__ is not model
            or not _defined_method(NMF, "_fit_transform", method.__func__)):
        return False
    if any(not _STATE_LOOKUPS.isdisjoint(vars(owner)) for owner in NMF.__mro__ if owner is not object):
        return False
    base = _nmf._BaseNMF
    for name in ("_check_params", "_check_w_h", "_compute_regularization"):
        if name in vars(model):
            return False
        original = vars(base).get(name)
        if not _defined_method(base, name, original):
            return False
        local = vars(NMF).get(name, original)
        if name == "_check_params":
            if not _defined_method(NMF, name, local):
                return False
        elif local is not original:
            return False
    if any(not _defined_function(vars(_nmf).get(name), _nmf, name) for name in _NUMERICS):
        return False
    helpers = (("safe_sparse_dot", extmath), ("squared_norm", extmath),
               ("check_array", validation), ("check_non_negative", validation))
    return all(_defined_function(vars(_nmf).get(name), owner, name) for name, owner in helpers)


def _admitted(model, x, method, h):
    if (not _stock_inner(model, method) or x.ndim != 2 or h.ndim != 2 or not x.size or not h.size
            or x.shape[1] != h.shape[1] or x.dtype != h.dtype or x.dtype not in (np.float32, np.float64)
            or sklearn.get_config().get("array_api_dispatch")):
        return False
    state = vars(model)
    if any(key not in state or type(state[key]) not in kinds for key, kinds in _CONTROLS.items()):
        return False
    if (state["solver"] != "mu" or state["init"] not in _INITIALIZERS
            or state["beta_loss"] not in (2, "frobenius") or state["alpha_W"] != 0
            or state["alpha_H"] not in (0, "same") or state["l1_ratio"] != 0
            or state["verbose"] != 0 or state["max_iter"] <= 0 or not 0 <= state["tol"] < float("inf")
            or state["n_components"] != len(h) or state["_n_components"] != len(h) or state["_beta_loss"] != 2):
        return False
    threshold = sqrt(float(np.finfo(x.dtype).tiny))
    for operand in (x, h):
        if (not np.isfinite(operand).all() or np.any(operand < 0) or not np.any(operand)
                or np.any((operand > 0) & (operand < threshold))):
            return False
    return True


def _corrected_weights(x, h, rank, cap, tol):
    from ._mu import _project_log_mu

    try:
        return joint(x, h, cap, tol, rank=rank, emit_warning=False)
    except FloatingPointError:
        return _project_log_mu(x, h, cap, tol, emit_warning=False)


def _native_weights(method, values, basis):
    coordinates, *_ = method(values, H=basis, update_H=False)
    return coordinates


def _validated_weights(model, values, method, basis):
    if (not stock_dot() or type(values) not in (np.ndarray, np.memmap)
            or type(basis) not in (np.ndarray, np.memmap)):
        return _native_weights(method, values, basis)
    # Use base arrays for private reductions, but pass the validated operands
    # and the original basis to the genuine bound solver.
    x, h = np.asarray(values), np.asarray(basis)
    if not _admitted(model, x, method, h):
        return _native_weights(method, values, basis)
    rank, cap, tol = model.n_components, model.max_iter, model.tol
    cutoff = qualification(x, h, rank)
    if cutoff is True or (cutoff is not None and int(cap) <= cutoff):
        return _native_weights(method, values, basis)

    # A warning can mutate inputs, solver controls, and even the public dot
    # binding. The real return supplies the only authoritative update count.
    frozen_x, frozen_h = capture(x), capture(h)
    coordinates, _, count = method(values, H=basis, update_H=False)
    if cutoff is not None and type(count) is int and 0 < count <= cap and count <= cutoff:
        return coordinates
    if not stock_dot():
        return coordinates
    # Deliver all genuine warnings and callback exceptions once. During
    # recovery, keep ignored signals ignored and handle other modes locally.
    with np.errstate(**{key: "ignore" if value == "ignore" else "raise"
                        for key, value in np.geterr().items()}):
        corrected = _corrected_weights(frozen_x, frozen_h, rank, cap, tol)
    return coordinates if np.array_equal(np.asarray(coordinates), corrected) else corrected


def _validated_transform(model, x):
    # This is the native transform prefix on the qualified wheels. Let the
    # wheel own tag dispatch, fitted validation and its output container.
    _nmf.check_is_fitted(model)
    values = _nmf.validate_data(model, x, accept_sparse=("csr", "csc"),
                                dtype=[np.float64, np.float32], reset=False, ensure_non_negative=True)
    with _nmf.config_context(assume_finite=True):
        method = model._fit_transform
        basis = model.components_
        return _validated_weights(model, values, method, basis)


if QUALIFIED_VERSION:
    _validated_transform = _set_output._wrap_method_output(_validated_transform, "transform")


def _owns_native(model, x):
    from ._mu import _standard_mu_basis

    state = vars(model)
    if (type(x) is not np.ndarray or type(model.components_) is not np.ndarray
            or not state.keys() <= _MODEL_KEYS or _standard_mu_basis(model, x) is None):
        return False
    seed = state.get("random_state")
    return (type(model.init) in (str, type(None)) and model.init in _INITIALIZERS
            and type(state.get("shuffle")) is bool and type(model.max_iter) is int
            and type(model.tol) in (int, float) and type(model.n_components) is int
            and "random_state" in state and (seed is None or type(seed) is int and 0 <= seed < 2**32)
            and state.get("_n_components") == model.n_components and state.get("_beta_loss") == 2
            and not sklearn.get_config().get("array_api_dispatch"))


def _public_params(model):
    state = vars(model)
    seed = state.get("random_state")
    return ((state.get("init") is None or type(state["init"]) is str and state["init"] in _INITIALIZERS)
            and type(state.get("shuffle")) in (bool, np.bool_) and "random_state" in state
            and (seed is None or type(seed) is np.random.RandomState
                 or type(seed) in _INTS and 0 <= seed < 2**32))


class _CorrectedOutput(TransformerMixin, BaseEstimator):
    """Wrap recovered coordinates using the real result's index and column labels."""

    def __init__(self, coordinates, names):
        self.coordinates = coordinates
        self.names = names

    def get_feature_names_out(self, input_features=None):
        return self.names

    def transform(self, X):
        return self.coordinates


def _restore(result, corrected):
    if type(result) is np.ndarray:
        return corrected
    for kind in ("pandas", "polars"):
        if type(result) is getattr(sys.modules.get(kind), "DataFrame", None):
            output = _CorrectedOutput(corrected, np.asarray(result.columns, dtype=object))
            return output.set_output(transform=kind).transform(result)
    raise TypeError(f"Cannot restore checked coordinates for {type(result).__name__}")


def _guarded_transform(model, x):
    from ._mu import _standard_mu_basis

    if (type(x) not in (np.ndarray, np.memmap) or _standard_mu_basis(model, x) is None
            or not _public_params(model) or not (x.min() >= 0 and np.isfinite(x.max()))):
        return model.transform(x)
    h = model.components_
    rank, cap, tol = model.n_components, model.max_iter, model.tol
    if qualification(x, h, rank, bounded_prefix=False) is True:
        return model.transform(x)
    frozen_x = np.array(x, copy=True, order="K")
    frozen_h = np.array(h, copy=True, order="K")
    result = model.transform(x)
    if not stock_dot():
        return result
    corrected = _corrected_weights(frozen_x, frozen_h, rank, cap, tol)
    return result if np.array_equal(np.asarray(result), corrected) else _restore(result, corrected)


def _custom_storage(value):
    # Basic memmaps carry their OS mapping and file metadata in these
    # instance dictionaries. A per-instance reduction is caller code, and
    # must be reached only by the established package and scikit-learn calls.
    return (type(value) is np.memmap
            and any(name in vars(np.ndarray) or name in vars(np.memmap) for name in vars(value)))


def native_projection(model, x):
    """Dispatch after the package has selected ordinary, widened or scaled operands."""
    if not QUALIFIED_VERSION or type(model) is not NMF:
        return model.transform(x)
    if "transform" in vars(model) or not initial_or_restored_transform(vars(NMF).get("transform")):
        return model.transform(x)
    if _custom_tags(model):
        return _validated_transform(model, x)
    if _custom_solver(model):
        if ("__sklearn_is_fitted__" in vars(model)
                or any("__sklearn_is_fitted__" in vars(owner) for owner in NMF.__mro__)):
            return _validated_transform(model, x)
        return model.transform(x)
    if not stock_dot() or _custom_storage(x) or _custom_storage(vars(model).get("components_")):
        return model.transform(x)
    if not _owns_native(model, x) or not (x.min() >= 0 and np.isfinite(x.max())):
        return _guarded_transform(model, x)
    local_config = vars(model).get("_sklearn_output_config", {})
    if type(local_config) is not dict:
        return _guarded_transform(model, x)
    output = local_config.get("transform", sklearn.get_config().get("transform_output"))
    if type(output) is not str or output != "default":
        return _guarded_transform(model, x)
    coordinates = joint(x, model.components_, model.max_iter, model.tol)
    # A cap warning can change set_output before sklearn wraps the result.
    return _set_output._wrap_data_with_container("transform", coordinates, x, model)
