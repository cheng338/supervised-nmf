"""Successful H solves may supply the outer residual and H gradient.

Natural-fit checks observe original arithmetic without replacing its selector.
The terminal-state tests deliberately enter the final NNLS qualification at a
controlled seam: their extreme values are not claims of full-solver reachability.
No historical solver copy, benchmark timer, or installed backend mock is used.
"""

import ast
from collections import Counter
from contextlib import contextmanager
import copy
import gc
import inspect
import math
from pathlib import Path
import struct
import sys
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import experimental_nmf as experimental


assert Path(experimental.__file__).resolve() == ROOT / "scripts" / "experimental_nmf.py"
TREE = ast.parse(Path(experimental.__file__).read_text(encoding="utf-8"))
FIT = inspect.unwrap(experimental.ANLSBPP.fit_transform).__code__
CORE = experimental._nnls_block_pivot.__code__
DEFAULT_POLICY = dict(divide="warn", over="warn", under="ignore", invalid="warn")


@pytest.fixture(autouse=True)
def isolated_observation_state():
    policy, callback = np.geterr(), np.geterrcall()
    profile, trace = sys.getprofile(), sys.gettrace()
    assert profile is None and trace is None
    with threadpool_limits(limits=1):
        try:
            yield
            assert np.geterr() == policy and np.geterrcall() is callback
            assert sys.getprofile() is profile and sys.gettrace() is trace
        finally:
            np.seterr(**policy)
            np.seterrcall(callback)
            sys.setprofile(profile)
            sys.settrace(trace)


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


def _fit_product_lines():
    """Executed source routes, not claimed native-BLAS event instrumentation."""
    fit = next(node for node in ast.walk(TREE)
               if isinstance(node, ast.FunctionDef) and node.name == "fit_transform")
    names = {"outer_reconstruction", "outer_gradient_w", "outer_gradient_h"}
    lines = {}
    for node in ast.walk(fit):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            lines.setdefault(node.lineno, []).append(node.func.id)
    return lines


class _FitObserver:
    def __init__(self, fault=None):
        self.lines = _fit_product_lines()
        self.routes = Counter()
        self.takes = self.objective_reuses = self.gradient_reuses = 0
        self.usability_calls = self.fired = self.post_gradient_checkpoint = 0
        self.roles, self.arrays = [], []
        self.last_residual = self.last_gradient = None
        self.target = None
        self.tail_calls = 0
        self.fault = fault
        self.error = (RuntimeError("required projected-gradient sentinel")
                      if fault == "required-pg" else MemoryError("handoff allocation sentinel"))
        core = next(node for node in TREE.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_nnls_block_pivot")
        publications = [node for node in ast.walk(core) if isinstance(node, ast.Assign)
                        and isinstance(node.value, ast.Tuple)
                        and any(isinstance(target, ast.Attribute) and target.attr == "_handoff"
                                for target in node.targets)]
        assert len(publications) == 1
        self.publication_line = publications[0].lineno
        fit = next(node for node in ast.walk(TREE)
                   if isinstance(node, ast.FunctionDef) and node.name == "fit_transform")
        self.objective_line = next(node.lineno for node in ast.walk(fit)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "objective" for target in node.targets))

    def _raise(self):
        assert self.fired == 0
        self.fired = 1
        raise self.error

    def trace(self, frame, event, arg):
        if frame.f_code is FIT:
            if event == "line":
                self.routes.update(self.lines.get(frame.f_lineno, ()))
            return self.trace
        if self.fault == "optional-publication" and frame.f_code is CORE:
            if event == "line" and frame.f_lineno == self.publication_line and not self.fired:
                self._raise()
            return self.trace
        return None

    def profile(self, frame, event, arg):
        if event == "return" and frame.f_code is experimental._FixedTrainingRHS.__init__.__code__:
            self.roles.append(weakref.ref(frame.f_locals["self"]))
        if event == "return" and frame.f_code is experimental._nnls_final_state.__code__ and type(arg) is tuple:
            role = frame.f_back.f_locals["cross_rhs"]
            assert role._handoff_enabled and role._targets is self.target()
            self.tail_calls += 1
            self.arrays.extend(weakref.ref(value) for value in arg[2:])
        if event == "return" and frame.f_code is experimental._take_handoff.__code__ and arg is not None:
            self.takes += 1
            self.last_residual = weakref.ref(arg[2])
            self.last_gradient = None if arg[3] is None else weakref.ref(arg[3])
        if event != "call":
            return
        if frame.f_code is experimental._handoff_usable.__code__:
            self.usability_calls += 1
            if self.fault == "optional-take" and self.usability_calls == 1:
                self._raise()
            if self.fault == "optional-consumer" and self.usability_calls == 2:
                caller = frame.f_back
                assert caller.f_code is FIT and caller.f_locals["_w_quiet"] is True
                assert caller.f_locals["_handoff"] is frame.f_locals["payload"]
                self.post_gradient_checkpoint += 1
                self._raise()
        caller = frame.f_back
        if caller is None or caller.f_code is not FIT:
            return
        if frame.f_code is experimental.linalg.norm.__code__:
            if (caller.f_lineno == self.objective_line and self.last_residual is not None
                    and caller.f_locals.get("residual") is self.last_residual()):
                self.objective_reuses += 1
                if self.fault == "required-objective-norm":
                    self._raise()
        if frame.f_code is experimental._pg_values.__code__ and "iteration" in caller.f_locals:
            if self.last_gradient is not None and frame.f_locals["grad_H"] is self.last_gradient():
                self.gradient_reuses += 1
            if self.fault == "required-pg":
                self._raise()

    @contextmanager
    def active(self):
        assert sys.getprofile() is None and sys.gettrace() is None
        try:
            sys.setprofile(self.profile)
            sys.settrace(self.trace)
            yield
        finally:
            sys.setprofile(None)
            sys.settrace(None)

    def assert_released(self):
        assert all(ref() is None or ref()._handoff is None for ref in self.roles)
        gc.collect()
        assert all(ref() is None for ref in self.arrays)


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
    observer = _FitObserver()
    observer.target = weakref.ref(data)
    optimized = _fit_result(data, rank, observer)
    observer.assert_released()
    # This is a declared ordinary-arithmetic reference, not a supported public
    # runtime-replacement contract. The natural route above used original math.
    with monkeypatch.context() as context:
        context.setattr(experimental, "_handoff_producer", lambda *args: False)
        ordinary = _fit_result(data, rank)
    assert optimized == ordinary
    assert observer.tail_calls == observer.takes == observer.objective_reuses == int(admitted)
    joint = admitted and not residual_only
    assert observer.gradient_reuses == int(joint)
    assert observer.routes["outer_reconstruction"] == int(not admitted)
    assert observer.routes["outer_gradient_w"] == 1
    assert observer.routes["outer_gradient_h"] == int(not joint)
    if bypass:
        assert observer.usability_calls == 0
    elif joint:
        assert observer.usability_calls == 2


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


TAIL_ARGUMENTS = ("matmul", "A", "B", "coefficients", "pending", "gradient_scale", "tol",
                  "iterations", "cap", "solves", "direct_solves", "backups", "cross_rhs")


def _terminal_seam():
    """Execute the actual core's final qualification/replay, with explicit locals."""
    core = next(node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == "_nnls_block_pivot")
    starts = [i for i, node in enumerate(core.body) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "_handoff_quiet"
                      for target in node.targets)]
    assert len(starts) == 1
    body = copy.deepcopy(core.body[starts[0]:])
    assert isinstance(body[-1], ast.Return)
    function = ast.FunctionDef(name="_controlled_terminal",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in TAIL_ARGUMENTS],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=body, decorator_list=[])
    ast.copy_location(function, body[0])
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(tree, experimental.__file__, "exec"), experimental.__dict__, namespace)
    return namespace["_controlled_terminal"]


def _terminal_arguments(kind):
    a, coefficients, b = np.zeros((9, 1)), np.zeros((1, 9)), np.zeros((9, 9))
    scale = np.ones(9)
    # Exact powers of two from the qualified finite-fault recipes. Later
    # norm reductions may also signal; compare every actual emitted event.
    powers = {
        "quiet": (0, 0, -1, 1., 0),
        "reconstruction-overflow": (600, 600, None, 1., 0),
        "late-overflow": (-450, 1000, 549, 2., 99),
        "underflow": (0, -600, -599, 1., -599),
        "gradient-w-overflow": (-100, 600, 499, 2., 399),
        "unsuccessful": (0, 0, -1, 1., 0),
    }
    ap, cp, bp, tol, sp = powers[kind]
    a[0, 0], coefficients[0, 0] = math.ldexp(1., ap), math.ldexp(1., cp)
    if bp is not None:
        b[0, 0] = math.ldexp(1., bp)
    scale[0] = math.ldexp(1., sp)
    return dict(matmul=experimental.nnls_matmul(), A=a, B=b, coefficients=coefficients,
                pending=np.array([0], dtype=int) if kind == "unsuccessful" else np.empty(0, dtype=int), gradient_scale=scale, tol=tol,
                iterations=1, cap=3, solves=1, direct_solves=0, backups=0)


def _terminal_run(kind, mode, enabled, repeats=1):
    terminal = _terminal_seam()
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
        with warnings.catch_warnings(record=True) as caught, np.errstate(**policy):
            warnings.simplefilter("error" if mode == "warning-error" else
                                  "default" if mode == "default" else "always")
            for iteration in range(repeats):
                arguments = _terminal_arguments(kind)
                role = experimental._FixedTrainingRHS(arguments["B"])
                # W and H share the same ordinary warning site. Alternation
                # explicitly checks default suppression after a replaying H tail.
                role._handoff_enabled = enabled and (repeats == 1 or iteration % 2 == 1)
                arguments["cross_rhs"] = role
                mutated = False

                def callback(message, flag):
                    nonlocal mutated
                    callbacks.append((message, int(flag)))
                    if mode == "callback-error":
                        raise sentinel
                    if not mutated:
                        arguments["B"][0, 0] = 1.
                        mutated = True

                np.seterrcall(callback if mode.startswith("callback") else saved_callback)
                try:
                    sys.setprofile(observe)
                    result = terminal(**arguments)
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
                publications.append(role._handoff is not None)
                values.append(_snapshot(arguments["B"]))
                role._handoff = None
            final_policy = np.geterr()
        warning_records = [(item.category, str(item.message)) for item in caught]
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
        assert optimized[6] and not any(optimized[6])
    if mode.startswith("callback"):
        assert optimized[3]  # prove a genuine arithmetic callback, not a fake FP exception
    if mode == "default":
        assert optimized[2]  # repeated W/H tails preserve shared-site suppression


def _consumer_seam():
    """Use the actual take/objective/PG block with explicit controlled fit locals."""
    fit = next(node for node in ast.walk(TREE)
               if isinstance(node, ast.FunctionDef) and node.name == "fit_transform")

    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == name for target in node.targets)

    take = next(node for node in ast.walk(fit) if assigned(node, "_handoff")
                and isinstance(node.value, ast.IfExp))
    residual = next(node for node in ast.walk(fit) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "_handoff is None"
                    and any(assigned(item, "residual") for item in node.body))
    objective = next(node for node in ast.walk(fit) if assigned(node, "objective"))
    pg = next(node for node in ast.walk(fit) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "_handoff is None or _handoff[3] is None")
    finish = ast.copy_location(ast.Return(value=ast.Name(id="current_pg", ctx=ast.Load())), pg)
    arguments = ("W", "H", "X", "matmul", "training_H_rhs", "outer_reconstruction",
                 "outer_gradient_w", "outer_gradient_h", "_fit_handoff_enabled")
    # The copied pre-acceptance take/reconstruction names the proposed factors
    # new_W/new_H; this controlled seam supplies those same factors as W/H.
    aliases = ast.copy_location(ast.parse("new_W, new_H = W, H").body[0], take)
    function = ast.FunctionDef(name="_controlled_consumer",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in arguments],
                           vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=copy.deepcopy([aliases, take, residual, objective, pg, finish]), decorator_list=[])
    ast.copy_location(function, take)
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(tree, experimental.__file__, "exec"), experimental.__dict__, namespace)
    return namespace["_controlled_consumer"]


def _consumer_run(mode, enabled):
    terminal, consume = _terminal_seam(), _consumer_seam()
    arguments = _terminal_arguments("gradient-w-overflow")
    role = experimental._FixedTrainingRHS(arguments["B"])
    role._handoff_enabled = enabled
    arguments["cross_rhs"] = role
    operation = arguments["matmul"]
    workspaces = [experimental.make_output_workspace(operation) for _ in range(3)]
    saved_callback = np.geterrcall()
    events, policy_at_pg = [], []
    sentinel = RuntimeError("unexpected consumer result")

    def callback(message, flag):
        events.append((message, int(flag)))
        np.seterr(over="ignore")

    def observe(frame, event, value):
        if event == "call" and frame.f_code is experimental._pg_values.__code__:
            policy_at_pg.append(np.geterr())

    try:
        policy = dict(DEFAULT_POLICY, over="call" if mode == "callback" else
                      "raise" if mode == "raise" else "warn")
        with warnings.catch_warnings(record=True) as caught, np.errstate(**policy):
            warnings.simplefilter("error" if mode == "warning-error" else "always")
            np.seterrcall(callback if mode == "callback" else saved_callback)
            h, info = terminal(**arguments)
            assert info["success"] and (role._handoff is not None) == enabled
            # These terminal values finish quietly; only actual outer GW
            # overflows. In enabled mode its strict attempt must replay.
            before = _snapshot(arguments["B"])
            result = None
            try:
                sys.setprofile(observe)
                consume(arguments["A"], h, arguments["B"], operation, role,
                        *workspaces, enabled)
                raise sentinel
            except (FloatingPointError, RuntimeWarning, ValueError) as error:
                result = (type(error), str(error), None if error.__context__ is None else
                          (type(error.__context__), str(error.__context__)))
                error.__traceback__ = None
            finally:
                sys.setprofile(None)
            assert result is not None and result[2] is None
            assert role._handoff is None and _snapshot(arguments["B"]) == before
            final_policy = np.geterr()
        return result, [(item.category, str(item.message)) for item in caught], events, policy_at_pg, final_policy
    finally:
        sys.setprofile(None)
        np.seterrcall(saved_callback)
        role._handoff = None
        for workspace in workspaces:
            if workspace is not operation:
                workspace.clear()


@pytest.mark.parametrize("mode", ["warn", "raise", "warning-error", "callback"])
def test_outer_gradient_replay_preserves_callback_policy_and_error_context(mode):
    ordinary = _consumer_run(mode, False)
    optimized = _consumer_run(mode, True)
    assert optimized == ordinary
    assert len(optimized[3]) == int(mode in ("warn", "callback"))
    if mode == "callback":
        assert optimized[2] and optimized[3][0]["over"] == "ignore"


@pytest.mark.parametrize("fault", [
    "optional-publication", "optional-take", "optional-consumer",
    "required-objective-norm", "required-pg",
])
def test_optional_failures_fall_back_and_required_errors_release_handoff(monkeypatch, fault):
    data = np.random.default_rng(4308).uniform(.05, 2., size=(513, 33))
    data.setflags(write=False)
    observer = _FitObserver(fault)
    observer.target = weakref.ref(data)
    if fault.startswith("optional"):
        with monkeypatch.context() as context:
            context.setattr(experimental, "_handoff_producer", lambda *args: False)
            expected = _fit_result(data, 1)
        actual = _fit_result(data, 1, observer)
        assert actual == expected
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
        traceback = caught.__traceback__
        fit_frames = 0
        while traceback is not None:
            if traceback.tb_frame.f_code is FIT:
                fit_frames += 1
                assert traceback.tb_frame.f_locals.get("_handoff") is None
            traceback = traceback.tb_next
        assert fit_frames == 1
        caught.__traceback__ = caught.__context__ = caught.__cause__ = None
        caught = None
    assert observer.fired == 1
    assert observer.post_gradient_checkpoint == int(fault == "optional-consumer")
    observer.assert_released()
