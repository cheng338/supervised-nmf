"""Successful H solves may supply the outer residual and H gradient.

Natural fits use the original selector. A separate fit counts logical products
at the selected backend and its output workspaces while running their arithmetic.
Terminal replay tests run a bound NNLS solve with controlled passive-set
proposals. The reconstruction-overflow case also changes its basis at that point.
No historical solver copy, benchmark timer, or installed backend mock is used.
"""

from collections import Counter
from contextlib import contextmanager
import gc
import inspect
import math
from pathlib import Path
import struct
import sys
from traceback import walk_tb
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental


assert Path(experimental.__file__).resolve() == ROOT / "scripts" / "experimental_nmf.py"
FIT = inspect.unwrap(experimental.ANLSBPP.fit_transform).__code__
CORE = experimental._nnls_block_pivot.__code__
DEFAULT_POLICY = dict(divide="warn", over="warn", under="ignore", invalid="warn")


@pytest.fixture(autouse=True)
def isolated_observation_state():
    policy, callback = np.geterr(), np.geterrcall()
    profile, trace = sys.getprofile(), sys.gettrace()
    with threadpool_limits(limits=1):
        try:
            yield
            assert np.geterr() == policy and np.geterrcall() is callback
            assert sys.getprofile() is profile and sys.gettrace() is trace
        finally:
            np.seterr(**policy)
            np.seterrcall(callback)
            if sys.getprofile() is not profile:
                sys.setprofile(profile)
            if sys.gettrace() is not trace:
                sys.settrace(trace)


@pytest.fixture
def idle_profiler():
    assert sys.getprofile() is None, "The handoff observer requires an idle profile hook."


def _snapshot(value):
    """Represent every fitted value, including signed zero/NaN bits, exactly."""
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes(order="C"))
    if isinstance(value, (float, np.floating)):
        return struct.pack("=d", float(value))
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


def _fit_result(data, rank, observer=None):
    before = _snapshot(data)
    model = experimental.ANLSBPP(rank, max_iter=1, tol=0., random_state=43)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if observer is None:
            coordinates = model.fit_transform(data)
        else:
            with observer.active():
                coordinates = model.fit_transform(data)
    assert _snapshot(data) == before
    assert model.n_iter_ == 1 and model.convergence_reason_ == "outer_iteration_limit"
    assert not any("handoff" in name for name in vars(model))
    return (_snapshot(coordinates), _snapshot(vars(model)),
            [(item.category, str(item.message)) for item in caught])


def _positional_arg(frame, index):
    return frame.f_locals[frame.f_code.co_varnames[index]]


class _FitObserver:
    def __init__(self, data, rank, fault=None, *, count_products=False):
        rows, columns = data.shape
        self.target = weakref.ref(data)
        self.product_shapes = {
            ((rows, rank), (rank, columns)): "reconstruction",
            ((rows, columns), (columns, rank)): "W gradient",
            ((rank, rows), (rows, columns)): "H gradient",
        }
        self.count_products = count_products
        self.products = Counter()
        self.after_pg = Counter()
        self.active_counts = None
        self.takes = self.gradient_reuses = 0
        self.objective_inputs = []
        self.usability_calls = self.fired = 0
        self.roles, self.arrays = [], []
        self.offer_writes = 0
        self.last_residual = self.last_gradient = None
        self.check_objective_inputs = False
        self.producer_role = None
        self.tail_calls = 0
        self.fault = fault
        self.error = (RuntimeError("required projected-gradient sentinel")
                      if fault == "required-pg" else MemoryError("handoff allocation sentinel"))

    def _raise(self):
        assert self.fired == 0
        self.fired = 1
        raise self.error

    def holds_taken_products(self, value):
        """Recognize the full offer by the arrays actually returned to this fit."""
        return (type(value) is tuple and len(value) == 5
                and value[2] is not None and value[3] is not None
                and self.last_residual is not None and self.last_gradient is not None
                and value[2] is self.last_residual() and value[3] is self.last_gradient())

    def _observe_products(self, patch):
        """Count logical products, including direct and workspace dispatch."""
        selector = experimental.nnls_matmul
        backend = selector()
        make_workspace = experimental.make_output_workspace

        def counted(operation):
            def product(left, right, *args, **kwargs):
                if self.active_counts is not None:
                    kind = self.product_shapes.get((left.shape, right.shape))
                    if kind is not None:
                        self.active_counts[kind] += 1
                return operation(left, right, *args, **kwargs)
            return product

        observed = counted(backend)

        def select():
            assert selector() is backend
            return observed

        def workspace(operation):
            assert operation is observed
            original = make_workspace(backend)
            if original is backend:
                return observed
            product = counted(original)
            product.clear = original.clear
            return product

        # Keep this instrumented fit on the same identity-gated route. The
        # independent natural fit exercises the unmodified identities as well.
        if backend is experimental.padded_matmul:
            patch.setattr(experimental, "padded_matmul", observed)
            patch.setattr(experimental, "_HANDOFF_PADDED", observed)
        else:
            assert backend is np.matmul
            patch.setattr(experimental, "_HANDOFF_MATMUL", observed)
        patch.setattr(experimental, "nnls_matmul", select)
        patch.setattr(experimental, "make_output_workspace", workspace)

    def _inject_optional_fault(self, patch):
        if self.fault == "offer-preparation":
            geterr = np.geterr

            def policy_snapshot():
                # Fail the producer's own request after the final NNLS state.
                # NumPy 1.26 also calls geterr while restoring errstate.
                if self.tail_calls and sys._getframe(1).f_code is CORE:
                    assert self.tail_calls == 1 and self.active_counts is None
                    assert self.offer_writes == 0
                    self._raise()
                return geterr()

            patch.setattr(np, "geterr", policy_snapshot)
        if self.fault in ("offer-preparation", "optional-publication"):
            role_type = experimental._FixedTrainingRHS
            store = role_type.__setattr__

            def publish(role, name, value):
                if (name == "_handoff" and value is not None and self.producer_role is not None
                        and role is self.producer_role()):
                    self.offer_writes += 1
                    if self.fault == "optional-publication" and not self.fired:
                        assert self.tail_calls == 1
                        assert value[2] is self.arrays[0]() and value[3] is self.arrays[1]()
                        self._raise()
                store(role, name, value)

            patch.setattr(role_type, "__setattr__", publish)
        elif self.fault in ("optional-take", "optional-consumer"):
            usable = experimental._handoff_usable
            signature = inspect.signature(usable)
            after_take = int(self.fault == "optional-consumer")

            def check(*args, **kwargs):
                if self.takes == after_take and not self.fired:
                    payload, role, *_ = signature.bind(*args, **kwargs).arguments.values()
                    assert role is self.producer_role() and role._handoff is None
                    assert self.tail_calls == 1 and payload is not None
                    assert payload[2] is self.arrays[0]() and payload[3] is self.arrays[1]()
                    assert usable(*args, **kwargs)
                    if after_take:
                        assert self.products == {"W gradient": 1}
                        assert self.objective_inputs == [True]
                        assert self.holds_taken_products(payload) and np.geterr() == payload[4]
                    self._raise()
                return usable(*args, **kwargs)

            patch.setattr(experimental, "_handoff_usable", check)

    def profile(self, frame, event, arg):
        if event == "return" and frame.f_code is experimental._FixedTrainingRHS.__init__.__code__:
            self.roles.append(weakref.ref(_positional_arg(frame, 0)))
        if event == "call" and frame.f_code is experimental._FixedTrainingRHS.solve.__code__:
            role = _positional_arg(frame, 0)
            if role._targets is self.target():
                self.producer_role = weakref.ref(role)
        if event == "return" and frame.f_code is experimental._nnls_final_state.__code__ and type(arg) is tuple:
            assert _positional_arg(frame, 2) is self.producer_role()._targets
            self.tail_calls += 1
            self.arrays.extend(weakref.ref(value) for value in arg[2:])
        if (event == "return" and frame.f_code is CORE and arg is not None
                and _positional_arg(frame, 1) is self.target() and arg[1]["success"]):
            self.active_counts = self.products
        if event == "return" and frame.f_code is experimental._take_handoff.__code__ and arg is not None:
            self.takes += 1
            self.check_objective_inputs = True
            assert self.tail_calls == 1 and arg[2] is self.arrays[0]()
            assert arg[3] is None or arg[3] is self.arrays[1]()
            self.last_residual = weakref.ref(arg[2])
            self.last_gradient = None if arg[3] is None else weakref.ref(arg[3])
        if event != "call":
            return
        if frame.f_code is experimental._handoff_usable.__code__:
            self.usability_calls += 1
        if self.check_objective_inputs and frame.f_code is experimental.linalg.norm.__code__:
            used_residual = _positional_arg(frame, 0) is self.last_residual()
            self.objective_inputs.append(used_residual)
            if self.fault == "required-objective-norm" and used_residual:
                self._raise()
        if frame.f_code is experimental._pg_values.__code__:
            if self.active_counts is self.products:
                self.active_counts = self.after_pg
            self.check_objective_inputs = False
            if self.last_gradient is not None and _positional_arg(frame, 3) is self.last_gradient():
                self.gradient_reuses += 1
            if self.fault == "required-pg" and self.takes:
                self._raise()

    @contextmanager
    def active(self):
        with pytest.MonkeyPatch.context() as patch:
            if self.count_products:
                self._observe_products(patch)
            self._inject_optional_fault(patch)
            try:
                sys.setprofile(self.profile)
                yield
            finally:
                sys.setprofile(None)

    def assert_released(self):
        assert all(ref() is None or ref()._handoff is None for ref in self.roles)
        gc.collect()
        assert all(ref() is None for ref in self.arrays)


@pytest.mark.usefixtures("idle_profiler")
@pytest.mark.parametrize("shape,rank,bypass,residual_only", [
    ((127, 129), 1, True, False),   # 16383 elements
    ((128, 128), 1, True, False),   # inclusive 16384 boundary
    ((5, 3277), 1, False, False),  # 16385, unchanged eligible rank-one route
    ((127, 129), 3, False, False), # gate is rank-one only
    ((65, 32), 8, False, True),    # AOCL residual-only geometry
    ((64, 32), 8, False, False),   # aligned higher-rank rejection
])
def test_fit_reuses_only_qualified_products_and_matches_ordinary_route(
        monkeypatch, shape, rank, bypass, residual_only):
    data = np.random.default_rng(4308).uniform(.05, 2., size=shape)
    data.setflags(write=False)
    operation = experimental.nnls_matmul()
    admitted = not bypass and (rank == 1 or (
        operation is experimental.padded_matmul and (shape[0] % 8 != 0 or shape[1] % 8 != 0)))
    with monkeypatch.context() as context:
        context.setattr(experimental, "_handoff_producer", lambda *args, **kwargs: False)
        ordinary = _fit_result(data, rank)
    joint = admitted and not residual_only
    for count_products in (False, True):
        observer = _FitObserver(data, rank, count_products=count_products)
        assert _fit_result(data, rank, observer) == ordinary
        observer.assert_released()
        assert observer.tail_calls == observer.takes == int(admitted)
        assert observer.objective_inputs == ([True] if admitted else [])
        assert observer.gradient_reuses == int(joint)
        if bypass:
            assert observer.usability_calls == 0
        elif joint:
            assert observer.usability_calls == 2
        if count_products:
            expected_products = Counter({"W gradient": 1})
            if not admitted:
                expected_products["reconstruction"] = 1
            if not joint:
                expected_products["H gradient"] = 1
            assert observer.products == expected_products
            # A completed fit may reconstruct once to report its final error.
            assert observer.after_pg <= Counter({"reconstruction": 1})


def _payload():
    a = np.arange(9., dtype=np.float64).reshape(9, 1)
    w = a.copy()
    h = np.ones((1, 9), dtype=np.float64)
    x = np.ones((9, 9), dtype=np.float64)
    role = experimental._FixedTrainingRHS(x)
    role._handoff_enabled = True
    payload = (a, h, np.zeros_like(x), np.zeros_like(h), np.geterr())
    return payload, role, w, h, x


@pytest.mark.parametrize("reason", [
    "role-disabled", "other-target", "unowned-role", "subclass", "float32",
    "fortran-target", "strided-design", "unaligned", "other-operation",
    "raw-higher-rank", "aligned-higher-rank",
])
def test_producer_declines_unqualified_roles_layouts_and_operations(reason):
    payload, role, a, _, x = _payload()
    operation = experimental.nnls_matmul()
    assert experimental._handoff_producer(role, a, x, operation)
    if reason == "role-disabled":
        role._handoff_enabled = False
    elif reason == "other-target":
        x = x.copy()
    elif reason == "unowned-role":
        role = object()
    elif reason == "subclass":
        class Subclass(np.ndarray):
            pass
        a = a.view(Subclass)
    elif reason == "float32":
        a = a.astype(np.float32)
    elif reason == "fortran-target":
        x = np.array(x, order="F")
        role._targets = x
    elif reason == "strided-design":
        a = np.zeros((18, 1))[::2]
    elif reason == "unaligned":
        a = np.ndarray((9, 1), dtype=np.float64, buffer=bytearray(73), offset=1)
    elif reason == "other-operation":
        operation = lambda left, right: None
    elif reason == "raw-higher-rank":
        a = np.zeros((9, 3))
        operation = experimental._HANDOFF_MATMUL
    else:
        a, x = np.zeros((8, 3)), np.zeros((8, 8))
        role._targets = x
        operation = experimental._HANDOFF_PADDED
    assert experimental._handoff_producer(role, a, x, operation) is False


@pytest.mark.parametrize("reason", ["H-identity", "X-identity", "policy", "basis-bit", "zero-sign", "layout"])
def test_take_rejects_stale_or_nonidentical_payload_once(reason):
    payload, role, w, h, x = _payload()
    operation = experimental._HANDOFF_MATMUL  # metadata only; no fake native call
    assert experimental._handoff_usable(payload, role, w, h, x, operation)
    if reason == "H-identity":
        h = h.copy()
    elif reason == "X-identity":
        x = x.copy()
    elif reason == "basis-bit":
        w[-1, 0] = np.nextafter(w[-1, 0], np.inf)
    elif reason == "zero-sign":
        w[0, 0] = -0.0
        assert np.array_equal(payload[0], w)  # numeric equality would be unsafe
    elif reason == "layout":
        storage = np.zeros((18, 1)); storage[::2] = w
        w = storage[::2]
    with np.errstate(over="ignore" if payload[4]["over"] != "ignore" else "warn") if reason == "policy" else np.errstate():
        role._handoff = payload
        assert experimental._take_handoff(role, w, h, x, operation) is None
    assert role._handoff is None
    assert experimental._take_handoff(role, w, h, x, operation) is None


def test_take_accepts_exact_offset_alias_and_equal_copy_but_not_other_offsets():
    payload, role, _, h, x = _payload()
    storage = np.arange(11., dtype=np.float64).reshape(11, 1)
    a, w = storage[1:10], storage[1:10].view()
    payload = (a, h, payload[2], payload[3], np.geterr())
    operation = experimental._HANDOFF_MATMUL
    for basis in (w, w.copy()):
        role._handoff = payload
        assert experimental._take_handoff(role, basis, h, x, operation) is payload
        assert role._handoff is None
    role._handoff = payload
    assert experimental._take_handoff(role, storage[:9], h, x, operation) is None
    assert role._handoff is None


def test_public_nnls_is_two_results_and_owned_role_clears_offer_on_next_solve():
    a, b = np.ones((9, 1)), np.ones((9, 9))
    direct = experimental.nnls_block_pivot(a, b)
    assert type(direct) is tuple and len(direct) == 2 and direct[1]["success"]
    role = experimental._FixedTrainingRHS(b)
    disabled = role.solve(a)
    assert _snapshot(disabled) == _snapshot(direct) and role._handoff is None
    role._handoff_enabled = True
    enabled = role.solve(a)
    assert _snapshot(enabled) == _snapshot(direct) and role._handoff is not None
    with pytest.raises(ValueError, match="equal row counts"):
        role.solve(a[:-1])
    assert role._handoff is None


def _terminal_problem(kind, patch):
    """Give the real pivot solver a finite basis, target and passive-set proposal."""
    columns = 9
    a, b = np.zeros((9, 1)), np.zeros((9, columns))
    coefficients = np.zeros((1, columns))
    ap, cp, bp, tol = {
        "quiet": (0, 0, -1, 1.),
        "reconstruction-overflow": (0, 600, None, 1.),
        "late-overflow": (-450, 1000, 549, 2.),
        "underflow": (0, -600, -599, 1.),
        "unsuccessful": (0, 0, -1, np.nextafter(1., 0.)),
    }[kind]
    a[0, 0], coefficients[0, 0] = math.ldexp(1., ap), math.ldexp(1., cp)
    if bp is not None:
        b[0, 0] = math.ldexp(1., bp)
    if kind == "underflow":
        # Exactly solved companion columns keep the pivot tolerances normal.
        # The first inexact tiny product occurs in the terminal diagnostic.
        coefficients[0, 1:] = b[0, 1:] = 1.
    initial = np.ones_like(coefficients)
    if kind == "unsuccessful":
        # Start at zero so the solver pivots the first column into its passive
        # set. A negative proposal at the cap leaves a real primal violation.
        initial = None
        coefficients[0, 0] = -1.

    def propose(A, B, gram, cross, passive):
        if kind == "reconstruction-overflow":
            # Its initial Gram product must stay finite. Change the controlled
            # basis after that product, so the real reconstruction overflows.
            A[0, 0] = math.ldexp(1., 600)
        return coefficients[:, :passive.shape[1]].copy(), 1, 0

    patch.setattr(experimental, "_passive_solve", propose)
    return a, b, dict(init=initial, max_iter=1, tol=tol)


def _terminal_run(kind, mode, enabled, repeats=1):
    policy = dict(DEFAULT_POLICY)
    if mode in ("raise", "callback", "callback-error"):
        policy["over"] = "raise" if mode == "raise" else "call"
    if mode in ("under-warn", "under-raise"):
        policy["under"] = mode.removeprefix("under-")
    saved_callback = np.geterrcall()
    callbacks, values, errors, publications = [], [], [], []
    helper_returns = []
    sentinel = RuntimeError("owned floating-point callback sentinel")

    def observe(frame, event, value):
        if event == "return" and frame.f_code is experimental._nnls_final_state.__code__:
            helper_returns.append(value is not None)

    # NumPy and scipy may report at their own Python reduction sites. Fresh
    # registries make default-filter multiplicity comparable, then restore them.
    namespaces = [experimental.__dict__]
    for function in (experimental.linalg.norm, np.linalg.norm):
        namespace = getattr(inspect.unwrap(function), "__globals__", None)
        if namespace is not None and all(namespace is not entry for entry in namespaces):
            namespaces.append(namespace)
    missing = object()
    registries = [(namespace, namespace.get("__warningregistry__", missing)) for namespace in namespaces]
    for namespace, _ in registries:
        namespace["__warningregistry__"] = {}
    try:
        with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings(record=True) as caught, np.errstate(**policy):
            basis, target, options = _terminal_problem(kind, patch)
            if mode == "default":
                # Input validation changes Python's warning-filter version.
                # Validate these fixed inputs once so successive tails share a registry.
                for array in (basis, target, options["init"]):
                    assert experimental._array(array, nonnegative=array is options["init"]) is array

                def validated(value, *, nonnegative=False):
                    assert any(value is array for array in (basis, target, options["init"]))
                    assert nonnegative == (value is options["init"])
                    return value

                patch.setattr(experimental, "_array", validated)
            warnings.simplefilter("error" if mode == "warning-error" else
                                  "default" if mode == "default" else "always")
            for iteration in range(repeats):
                target_before = target.copy()
                role = experimental._FixedTrainingRHS(target)
                # W and H share the same ordinary warning site. Alternation
                # explicitly checks default suppression after a replaying H tail.
                role._handoff_enabled = enabled and (repeats == 1 or iteration % 2 == 1)
                mutated = False

                def callback(message, flag):
                    nonlocal mutated
                    callbacks.append((message, int(flag)))
                    if mode == "callback-error":
                        raise sentinel
                    if not mutated:
                        target[0, 0] = 1.
                        mutated = True

                caller_callback = callback if mode.startswith("callback") else saved_callback
                np.seterrcall(caller_callback)
                try:
                    sys.setprofile(observe)
                    result = role.solve(basis, **options)
                    if kind == "unsuccessful":
                        info = result[1]
                        assert not info["success"] and info["n_iter"] == info["max_iter"]
                        assert info["projected_gradient_relative"] <= info["kkt_tolerance"]
                    values.append(_snapshot(result))
                    errors.append(None)
                except (FloatingPointError, RuntimeWarning, RuntimeError, ValueError) as error:
                    errors.append((type(error), str(error), error is sentinel,
                                   None if error.__context__ is None else
                                   (type(error.__context__), str(error.__context__))))
                    values.append(None)
                    error.__traceback__ = None
                finally:
                    sys.setprofile(None)
                assert np.geterrcall() is caller_callback
                publications.append(role._handoff is not None)
                if mode == "callback":
                    assert mutated and target_before[0, 0] != 1.
                    target_before[0, 0] = 1.
                assert _snapshot(target) == _snapshot(target_before)
                values.append(_snapshot(target))
                role._handoff = None
            final_policy = np.geterr()
        warning_records = [(item.category, str(item.message), item.filename, item.lineno) for item in caught]
        return values, errors, warning_records, callbacks, final_policy, publications, helper_returns
    finally:
        sys.setprofile(None)
        np.seterrcall(saved_callback)
        for namespace, registry in registries:
            if registry is missing:
                namespace.pop("__warningregistry__", None)
            else:
                namespace["__warningregistry__"] = registry
        sentinel.__traceback__ = sentinel.__context__ = sentinel.__cause__ = None


@pytest.mark.usefixtures("idle_profiler")
@pytest.mark.parametrize("kind,mode,repeats", [
    ("quiet", "always", 1),
    ("unsuccessful", "always", 1),
    ("reconstruction-overflow", "always", 1),
    ("late-overflow", "default", 4),
    ("late-overflow", "warning-error", 1),
    ("late-overflow", "raise", 1),
    ("late-overflow", "callback", 1),
    ("late-overflow", "callback-error", 1),
    ("underflow", "always", 1),
    ("underflow", "under-warn", 1),
    ("underflow", "under-raise", 1),
])
def test_terminal_replay_matches_caller_policy_warnings_errors_and_mutations(kind, mode, repeats):
    ordinary = _terminal_run(kind, mode, False, repeats)
    optimized = _terminal_run(kind, mode, True, repeats)
    assert optimized[:5] == ordinary[:5]
    assert not any(ordinary[5]) and ordinary[6] == []
    if kind in ("quiet", "underflow") and mode == "always":
        assert optimized[5] == [True] and optimized[6] == [True]
    elif kind == "unsuccessful":
        assert optimized[5] == [False] and optimized[6] == [True]
        assert optimized[2]
    else:
        assert not any(optimized[5])
        assert optimized[6] == [False] * (repeats // 2 if repeats > 1 else 1)
    expected_error = {"raise": FloatingPointError, "under-raise": FloatingPointError,
                      "warning-error": RuntimeWarning, "callback-error": RuntimeError}.get(mode)
    if kind == "reconstruction-overflow":
        expected_error = ValueError  # the nonfinite reconstruction reaches the norm
    if expected_error is None:
        assert optimized[1] == [None] * repeats
    else:
        error, = optimized[1]
        assert error[0] is expected_error and error[2] == (mode == "callback-error")
        assert error[3] is None
    if mode.startswith("callback"):
        assert any("overflow" in message for message, flag in optimized[3])
    if mode == "default":
        once = _terminal_run(kind, mode, False)[2]
        assert once and optimized[2] == once


def _consumer_run(monkeypatch, mode, enabled):
    """Give the real fit controlled passive-set solutions and let it qualify both."""
    data = np.zeros((129, 128))
    w = np.zeros((data.shape[0], 1))
    h = np.zeros((1, data.shape[1]))
    data[0, 0] = math.ldexp(1., 499)
    w[0, 0] = math.ldexp(1., -100)
    h[0, 0] = math.ldexp(1., 600)
    data.setflags(write=False)
    before = _snapshot(data)
    solutions, roles, takes, w_policies = [], [], [], []
    events, policy_at_pg = [], []
    h_products, h_at_pg = [], []
    accepted_w = None

    def initialize(X, n_components, init=None, eps=1e-6, random_state=None):
        assert X is data and n_components == 1
        return np.ones_like(w), np.ones_like(h)

    def passive_solve(A, B, gram, cross, passive):
        # The two transposed target shapes identify the normal W and H solves.
        # Their qualification, publication and all matrix products run in production.
        if B.shape == data.T.shape:
            solution = w.T
            solutions.append("W")
        else:
            assert B is data and np.array_equal(A, w)
            solution = h
            solutions.append("H")
        assert solution.shape == passive.shape and passive.all()
        return solution.copy(), 1, 0

    def callback(message, flag):
        events.append((message, int(flag)))
        np.seterr(over="ignore")
        # A callback can change a factor before the next gradient is computed.
        accepted_w[0, 0] = math.ldexp(1., -99)

    make_workspace = experimental.make_output_workspace

    def observed_workspace(operation):
        workspace = make_workspace(operation)

        def product(left, right, *args, **kwargs):
            if left.shape == data.shape and right.shape == (data.shape[1], 1):
                w_policies.append(np.geterr())
            result = workspace(left, right, *args, **kwargs)
            if left.shape == (1, data.shape[0]) and right.shape == data.shape:
                h_products.append(float(result[0, 0]))
            return result

        # The native operation itself has no workspace to release.
        product.clear = workspace.clear if workspace is not operation else lambda: None
        return product

    def observe(frame, event, value):
        nonlocal accepted_w
        if frame.f_code is experimental._take_handoff.__code__:
            if event == "call":
                role = _positional_arg(frame, 0)
                accepted_w = _positional_arg(frame, 1)
                assert role._targets is data
                roles.append(role)
            if event == "return":
                takes.append(value is not None)
                if value is not None:
                    assert value[3] is not None
        if frame.f_code is experimental._pg_values.__code__ and event == "call" and takes:
            policy_at_pg.append(np.geterr())
            h_at_pg.append(float(_positional_arg(frame, 3)[0, 0]))
        return None

    policy = dict(DEFAULT_POLICY, over="call" if mode == "callback" else
                  "raise" if mode == "raise" else "warn")
    probe_policy = dict.fromkeys(DEFAULT_POLICY, "raise")
    probe_policy["under"] = "ignore"
    saved_callback = np.geterrcall()
    with monkeypatch.context() as patch, warnings.catch_warnings(record=True) as caught, np.errstate(**policy):
        patch.setattr(experimental, "_initialize_nmf", initialize)
        patch.setattr(experimental, "_passive_solve", passive_solve)
        patch.setattr(experimental, "make_output_workspace", observed_workspace)
        if not enabled:
            patch.setattr(experimental, "_handoff_producer", lambda *args, **kwargs: False)
        warnings.simplefilter("error" if mode == "warning-error" else "always")
        np.seterrcall(callback if mode == "callback" else saved_callback)
        try:
            # At nnls_tol=2 both real NNLS tails qualify. The proposed residual
            # has the same norm as the initial one to float64 precision. Only
            # the outer W gradient overflows, from 2**499 times 2**600.
            model = experimental.ANLSBPP(1, max_iter=1, tol=0., nnls_tol=2.)
            with pytest.raises((FloatingPointError, RuntimeWarning, ValueError)) as caught_error:
                sys.setprofile(observe)
                try:
                    model.fit_transform(data)
                finally:
                    sys.setprofile(None)
            error = caught_error.value
            assert error.__context__ is None
            result = (type(error), str(error))
            error.__traceback__ = None
            final_policy = np.geterr()
        finally:
            np.seterrcall(saved_callback)
    assert solutions == ["W", "H"] and takes == [enabled]
    assert len(roles) == 1 and roles[0]._handoff is None and _snapshot(data) == before
    # The strict W product raises before replay under the caller's policy.
    assert w_policies == ([probe_policy] if enabled else []) + [policy]
    # R[0, 0] is 2**499. In callback mode W[0, 0] doubles before W.T @ R.
    expected_h = math.ldexp(1., 400 if mode == "callback" else 399)
    assert h_products == h_at_pg == ([expected_h] if mode in ("warn", "callback") else [])
    return result, [(item.category, str(item.message)) for item in caught], events, policy_at_pg, final_policy


@pytest.mark.usefixtures("idle_profiler")
@pytest.mark.parametrize("mode", ["warn", "raise", "warning-error", "callback"])
def test_outer_gradient_replay_preserves_callback_policy_and_error_context(monkeypatch, mode):
    ordinary = _consumer_run(monkeypatch, mode, False)
    optimized = _consumer_run(monkeypatch, mode, True)
    assert optimized == ordinary
    (error_type, _), caught, events, policy_at_pg, final_policy = optimized
    assert error_type is {"warn": ValueError, "raise": FloatingPointError,
                          "warning-error": RuntimeWarning, "callback": ValueError}[mode]
    assert len(policy_at_pg) == int(mode in ("warn", "callback"))
    if mode == "warn":
        assert any(category is RuntimeWarning and "overflow" in message for category, message in caught)
    if mode == "callback":
        assert events and policy_at_pg[0]["over"] == final_policy["over"] == "ignore"


@pytest.mark.usefixtures("idle_profiler")
@pytest.mark.parametrize("fault", [
    "optional-publication", "optional-take", "optional-consumer",
    "required-objective-norm", "required-pg",
])
def test_optional_failures_fall_back_and_required_errors_release_handoff(monkeypatch, fault):
    data = np.random.default_rng(4308).uniform(.05, 2., size=(513, 33))
    data.setflags(write=False)
    observer = _FitObserver(data, 1, fault, count_products=fault == "optional-consumer")
    if fault.startswith("optional"):
        with monkeypatch.context() as context:
            context.setattr(experimental, "_handoff_producer", lambda *args, **kwargs: False)
            expected = _fit_result(data, 1)
        if fault == "optional-publication":
            # Allocation of the offer's policy snapshot can also fail before
            # the store. Both failures must preserve the ordinary result.
            preparation = _FitObserver(data, 1, "offer-preparation")
            assert _fit_result(data, 1, preparation) == expected
            assert preparation.fired == 1 and preparation.takes == 0
            assert preparation.offer_writes == 0
            preparation.error.__traceback__ = None
            preparation.assert_released()
        actual = _fit_result(data, 1, observer)
        assert actual == expected
        assert observer.takes == int(fault == "optional-consumer")
        # The owned injected exception may retain the swallowed fallback's
        # traceback. Pending ownership must already be gone; array release is
        # required after the test releases that deliberately retained traceback.
        assert all(ref() is None or ref()._handoff is None for ref in observer.roles)
        observer.error.__traceback__ = observer.error.__context__ = observer.error.__cause__ = None
    else:
        caught = None
        try:
            _fit_result(data, 1, observer)
        except (MemoryError, RuntimeError) as error:
            caught = error
        assert caught is observer.error
        assert all(ref() is None or ref()._handoff is None for ref in observer.roles)
        fit_frame, = (frame for frame, _ in walk_tb(caught.__traceback__)
                      if frame.f_code is FIT)
        assert observer.takes == 1 and observer.last_gradient is not None
        if fault == "required-objective-norm":
            gc.collect()
            assert observer.last_gradient() is None
        else:
            assert observer.gradient_reuses == 1
        # The failing projected-gradient call owns an argument. Release its
        # traceback while the fit frame remains alive to check the fit's cleanup.
        caught.__traceback__ = caught.__context__ = caught.__cause__ = None
        caught = None
        gc.collect()
        assert observer.last_gradient() is None
        del fit_frame
    assert observer.fired == 1
    if fault == "optional-publication":
        assert observer.offer_writes == 1
    if fault == "optional-consumer":
        assert observer.products == {"W gradient": 1, "H gradient": 1}
        assert observer.after_pg <= Counter({"reconstruction": 1})
    observer.assert_released()
