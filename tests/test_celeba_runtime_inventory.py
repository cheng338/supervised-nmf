"""Fresh native inventory, public fallbacks and retained-controller behavior."""
from contextlib import contextmanager
import _ctypes
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
import weakref
import numpy as np
import pytest
import threadpoolctl as tpc
from tests._celeba_recovery_test_support import inventory_runtime

@pytest.fixture
def recovery(inventory_runtime):
    return inventory_runtime

def assert_admitted(recovery, state):
    observed = state['_inventory_state']
    assert observed['qualified'] is True and observed['routes_enabled'] is True
    assert issubclass(observed['controller_type'], tpc.ThreadpoolController)


class InventoryFailure(RuntimeError):
    pass


def identity(recovery, state, files=None, threads=1):
    return recovery.active_runtime_identity(threads, file_cache={} if files is None else files,
                                            _initialization_cache=state)


@contextmanager
def original_calls(recovery):
    """Observe original code objects without replacing runtime functions."""
    previous = sys.getprofile()
    active = recovery.active_runtime_identity.__code__
    ones = np.ones.__code__
    prefix = tpc.ThreadpoolController._check_prefix.__code__
    exists_function = os.path.exists
    exists = getattr(exists_function, "__code__", None)
    counts = {"full": 0, "opted": 0, "ones": 0, "prefix": 0, "exists": 0}
    active_frames = set()

    def profile(frame, event, arg):
        if event == "call":
            if frame.f_code is active:
                active_frames.add(id(frame))
                counts["full" if frame.f_locals.get("_initialization_cache") is None else "opted"] += 1
            if active_frames:
                for key, code in (("ones", ones), ("prefix", prefix), ("exists", exists)):
                    if frame.f_code is code:
                        counts[key] += 1
        elif event == "return" and frame.f_code is active:
            active_frames.discard(id(frame))
        elif event == "c_call" and active_frames and arg is exists_function:
            counts["exists"] += 1
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)
        assert not active_frames


@pytest.fixture
def custom_registry():
    names = ("_ALL_CONTROLLERS", "_ALL_USER_APIS", "_ALL_INTERNAL_APIS", "_ALL_PREFIXES")
    originals = {name: (getattr(tpc, name), list(getattr(tpc, name))) for name in names}
    control = {"version": "one", "threads": 1, "extra": True, "fail": None,
               "warn": False, "versions": 0, "attributes": 0, "get_threads": 0}
    basename = Path(_ctypes.__file__).name.lower()

    class ProbeController(tpc.LibController):
        user_api = "blas"
        internal_api = "recovery_test_probe"
        filename_prefixes = (basename,)

        def get_version(self):
            control["versions"] += 1
            if control["fail"] == "version":
                raise InventoryFailure("custom version failure")
            if control["warn"]:
                warnings.warn("recovery_test custom inventory warning", UserWarning)
            return control["version"]

        def set_additional_attributes(self):
            control["attributes"] += 1
            if control["fail"] == "attributes":
                raise InventoryFailure("custom attributes failure")
            if control["extra"]:
                self.conditional_metadata = "present"

        def get_num_threads(self):
            control["get_threads"] += 1
            if control["fail"] == "threads":
                raise InventoryFailure("custom thread getter failure")
            return control["threads"]

        def set_num_threads(self, value):
            control["threads"] = value

    try:
        yield ProbeController, control
    finally:
        for name, (original_object, contents) in originals.items():
            assert getattr(tpc, name) is original_object
            original_object[:] = contents


def test_negative_routes_are_actually_reused_when_enabled(recovery):
    state = {}
    oracle = identity(recovery, state)
    with original_calls(recovery) as first:
        assert identity(recovery, state) == oracle
    with original_calls(recovery) as next_call:
        assert identity(recovery, state) == oracle
    assert_admitted(recovery, state)
    assert state["_inventory_state"]["negative_paths"]
    assert 0 < next_call["prefix"] < first["prefix"]
    assert first["full"] == next_call["full"] == 0


def test_register_prefix_change_and_remove_real_loaded_adapter(recovery, custom_registry):
    cls, control = custom_registry
    state = {}
    baseline = identity(recovery, state)
    assert identity(recovery, state) == baseline
    tpc.register(cls)
    with original_calls(recovery) as observed:
        added = identity(recovery, state)
    assert observed["full"] == 0 and control["versions"] > 0
    assert added == recovery.active_runtime_identity(1)
    assert added != baseline
    assert_admitted(recovery, state)
    original_prefixes = cls.filename_prefixes
    cls.filename_prefixes = ("recovery_test_never_loaded_",)
    assert identity(recovery, state) == baseline
    cls.filename_prefixes = original_prefixes
    assert identity(recovery, state) == added
    tpc._ALL_CONTROLLERS.remove(cls)
    assert identity(recovery, state) == baseline


def test_registered_metadata_is_fresh_on_each_warm_guard(recovery, custom_registry):
    cls, control = custom_registry
    tpc.register(cls)
    state = {}
    first = identity(recovery, state)
    identity(recovery, state)
    before = {name: control[name] for name in ("versions", "attributes", "get_threads")}
    control["version"] = "two"
    control["extra"] = False
    control["threads"] = 3
    with original_calls(recovery) as observed:
        changed = identity(recovery, state)
    assert observed["full"] == observed["ones"] == 0
    assert_admitted(recovery, state)
    assert changed != first
    assert all(control[name] > before[name] for name in before)
    assert changed == recovery.active_runtime_identity(1)
    current = [row for row in tpc.threadpool_info() if row["internal_api"] == cls.internal_api]
    assert len(current) == 1 and current[0]["version"] == "two"
    assert current[0]["num_threads"] == 3 and "conditional_metadata" not in current[0]


@pytest.mark.parametrize("policy", ("always", "error"))
def test_custom_warning_is_not_suppressed_by_cached_routing(recovery, custom_registry, policy):
    cls, control = custom_registry
    tpc.register(cls)
    state = {}
    expected = identity(recovery, state)
    identity(recovery, state)
    control["warn"] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy)
        if policy == "error":
            with pytest.raises(UserWarning, match="recovery_test custom inventory warning"):
                identity(recovery, state)
            assert "ready" not in state
        else:
            assert identity(recovery, state) == expected
            assert identity(recovery, state) == expected
            assert len(caught) == 2
            assert all(str(row.message) == "recovery_test custom inventory warning" for row in caught)


def test_changed_threadpool_info_is_honored(recovery, monkeypatch):
    state = {}
    expected = identity(recovery, state)
    identity(recovery, state)
    original = tpc.threadpool_info
    called = []

    def custom_info():
        called.append(True)
        return original()

    with monkeypatch.context() as change:
        change.setattr(tpc, "threadpool_info", custom_info)
        assert identity(recovery, state) == expected
    assert called == [True]
    assert identity(recovery, state) == expected


def test_unreviewed_version_uses_public_inventory_then_readmits(recovery, monkeypatch):
    state = {}
    identity(recovery, state)
    identity(recovery, state)
    with monkeypatch.context() as change:
        change.setattr(tpc, "__version__", "3.7.0-unreviewed")
        actual = identity(recovery, state)
        assert actual == recovery.active_runtime_identity(1)
        assert state["_inventory_state"]["qualified"] is False
    identity(recovery, state)
    assert_admitted(recovery, state)


def test_changed_public_default_is_observed_by_fallback(recovery, monkeypatch):
    state = {}
    expected = identity(recovery, state)
    identity(recovery, state)
    original_info = tpc.ThreadpoolController.info
    calls = []

    def observe_info(self, debugging_info=False):
        calls.append(debugging_info)
        # This test verifies dispatch/default forwarding, not threadpoolctl's
        # intrusive native thread-scope experiment. Fresh ordinary info remains.
        return original_info(self, debugging_info=False)

    with monkeypatch.context() as change:
        change.setattr(tpc.threadpool_info, "__defaults__", (True,))
        change.setattr(tpc.ThreadpoolController, "info", observe_info)
        assert identity(recovery, state) == expected
    assert calls == [True]
    identity(recovery, state)
    assert_admitted(recovery, state)


@pytest.mark.parametrize("mutation", ("register", "prefix"))
def test_positive_constructor_refreshes_negative_routes_within_same_scan(
        recovery, custom_registry, tmp_path, mutation):
    """Controlled routing order; real custom/native construction is tested above."""
    class Trigger:
        user_api = "blas"
        internal_api = "recovery_test_route_trigger"
        filename_prefixes = ("recovery_test-trigger",)

    class Added:
        user_api = "blas"
        internal_api = "recovery_test_route_added"
        filename_prefixes = ("recovery_test-new" if mutation == "register" else "recovery_test-unmatched",)

    tpc.register(Trigger)
    if mutation == "prefix":
        tpc.register(Added)
    state = {}
    recovery._inventory_update_routes(state, tpc)
    constructed = []
    negative = str(tmp_path / "recovery_test-new.so")
    trigger = str(tmp_path / "recovery_test-trigger.so")

    def original_construction(self, path):
        assert self is controller
        constructed.append(path)
        if path == trigger:
            if mutation == "register":
                tpc.register(Added)
            else:
                Added.filename_prefixes = ("recovery_test-new",)

    controller = SimpleNamespace(_recovery_inventory_state=state, _recovery_inventory_module=tpc,
                                 _recovery_original_make=original_construction)
    controller._check_prefix = tpc.ThreadpoolController._check_prefix.__get__(controller, type(controller))
    recovery._inventory_negative_make(controller, negative)
    assert not constructed and tpc._realpath(negative) in state["negative_paths"]
    recovery._inventory_negative_make(controller, trigger)
    recovery._inventory_negative_make(controller, negative)
    assert constructed == [trigger, negative]


@contextmanager
def observe(module):
    assert sys.getprofile() is None
    controllers = []
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is module._inventory_fresh_info.__code__:
            value = frame.f_locals.get('controller')
            if value is not None:
                controllers.append(value)
    sys.setprofile(profile)
    try:
        yield controllers
    finally:
        sys.setprofile(None)


def released(recovery, controller):
    assert controller._recovery_inventory_state is recovery._INVENTORY_RELEASED_ROUTES
    assert dict(controller._recovery_inventory_state) == {'routes_enabled': False}
    assert controller._recovery_inventory_module is tpc
    assert controller._recovery_original_make is tpc.ThreadpoolController._make_controller_from_path


def prime(recovery):
    state = {}
    expected = identity(recovery, state)
    with observe(recovery) as controllers:
        assert identity(recovery, state) == expected
    assert len(controllers) == 1
    return state, expected, controllers[0]


def test_fresh_controllers_native_children_and_immutable_disabled_state(recovery):
    state, expected, first = prime(recovery)
    with observe(recovery) as controllers:
        assert identity(recovery, state) == expected
    second = controllers[0]
    assert type(first) is type(second) is state["_inventory_state"]["controller_type"]
    assert first is not second and first.lib_controllers is not second.lib_controllers
    for controller in (first, second):
        released(recovery, controller)
        assert controller.lib_controllers
        assert all(child.parent is controller for child in controller.lib_controllers)
        assert sorted(controller.info(debugging_info=False), key=lambda row: row['filepath']) == sorted(tpc.threadpool_info(), key=lambda row: row['filepath'])
    with pytest.raises(TypeError):
        first._recovery_inventory_state['routes_enabled'] = True
    assert first._recovery_inventory_state is second._recovery_inventory_state


def test_operation_state_releases_while_retained_parent_and_bound_route_work(recovery):
    state, expected, controller = prime(recovery)
    saved_route = controller._make_controller_from_path
    reference = weakref.ref(state['_inventory_state']['negative_paths'])
    state.clear()
    assert reference() is None
    saved_route(_ctypes.__file__)
    controller._load_libraries()
    assert all(child.parent is controller for child in controller.lib_controllers)
    assert identity(recovery, state) == expected


def test_separate_operations_keep_live_routing_caches_independent(recovery):
    a, expected, old = prime(recovery)
    b, value, new = prime(recovery)
    assert value == expected
    left, right = a['_inventory_state'], b['_inventory_state']
    assert left is not right and left['negative_paths'] is not right['negative_paths']
    assert left['controller_type'] is not right['controller_type']
    assert left['routes_enabled'] is right['routes_enabled'] is True
    released(recovery, old)
    released(recovery, new)
    a.clear()
    old._load_libraries()
    assert identity(recovery, b) == expected
    assert b['_inventory_state'] is right and right['routes_enabled'] is True


@pytest.mark.parametrize('route', ['saved_method', 'full_rescan'])
def test_late_custom_registration_is_seen_after_previous_negative(recovery, custom_registry, route):
    cls, control = custom_registry
    state, expected, old = prime(recovery)
    canonical = tpc._realpath(_ctypes.__file__)
    assert canonical in state['_inventory_state']['negative_paths']
    saved_route = old._make_controller_from_path
    tpc.register(cls)
    if route == 'saved_method':
        saved_route(_ctypes.__file__)
    else:
        old._load_libraries()
    rows = [row for row in old.info() if row['internal_api'] == 'recovery_test_probe']
    assert len(rows) == 1 and rows[0]['version'] == control['version']
    fresh = identity(recovery, state)
    assert any(row['internal_api'] == 'recovery_test_probe' for row in fresh['native_libraries'])


def test_retained_custom_parent_rescans_live_prefixes_and_getters(recovery, custom_registry):
    cls, control = custom_registry
    prefixes = cls.filename_prefixes
    cls.filename_prefixes = ('recovery_test_no_loaded_library_matches',)
    state, expected, old = prime(recovery)
    saved = old._make_controller_from_path
    tpc.register(cls)
    saved(_ctypes.__file__)
    assert not any(row['internal_api'] == 'recovery_test_probe' for row in old.info())
    cls.filename_prefixes = prefixes
    saved(_ctypes.__file__)
    versions = control['versions']
    control['threads'] = 2
    old._load_libraries()
    assert control['versions'] > versions
    rows = [row for row in old.info() if row['internal_api'] == 'recovery_test_probe']
    assert len(rows) == 1 and rows[0]['num_threads'] == 2
    control['warn'] = True
    with pytest.warns(UserWarning, match='recovery_test custom inventory warning'):
        old._load_libraries()


@pytest.mark.parametrize('failure', ['version', 'attributes', 'threads'])
def test_error_releases_state_and_retained_parent_can_retry(recovery, custom_registry, failure):
    cls, control = custom_registry
    tpc.register(cls)
    state, expected, _ = prime(recovery)
    control['fail'] = failure
    with observe(recovery) as controllers:
        with pytest.raises(InventoryFailure):
            identity(recovery, state)
    assert controllers
    for controller in controllers:
        released(recovery, controller)
    assert 'ready' not in state and '_inventory_state' not in state
    control['fail'] = None
    for controller in controllers:
        controller._load_libraries()
        assert any(row['internal_api'] == 'recovery_test_probe' for row in controller.info())
    assert identity(recovery, state) == expected
    assert identity(recovery, state) == expected


def test_changed_executing_discovery_method_uses_public_fallback(recovery, monkeypatch):
    state, expected, _ = prime(recovery)
    original = tpc.ThreadpoolController._load_libraries
    calls = []
    def wrapped(controller):
        calls.append(controller)
        return original(controller)
    with monkeypatch.context() as change:
        change.setattr(tpc.ThreadpoolController, '_load_libraries', wrapped)
        assert identity(recovery, state) == expected
        assert calls and state['_inventory_state']['qualified'] is False
    assert identity(recovery, state) == expected
    assert_admitted(recovery, state)


@pytest.mark.parametrize('reset', ['hook', 'threads'])
def test_initialized_inventory_is_discarded_before_reinitialization(recovery, monkeypatch, reset):
    state, expected, _ = prime(recovery)
    old = state['_inventory_state']
    if reset == 'hook':
        with monkeypatch.context() as change:
            change.setattr(recovery, '_hook', lambda *args, **kwargs: None)
            with original_calls(recovery) as observed:
                assert identity(recovery, state) == expected
            assert observed['full'] == 1 and observed['ones'] == 4
            assert 'ready' not in state and '_inventory_state' not in state
        threads = 1
    else:
        with original_calls(recovery) as observed:
            changed = identity(recovery, state, threads=2)
        assert changed['threads'] == 2
        assert observed['full'] == 1 and observed['ones'] == 4
        assert '_inventory_state' not in state
        threads = 2
    direct = recovery.active_runtime_identity(threads)
    assert identity(recovery, state, threads=threads) == direct
    assert identity(recovery, state, threads=threads) == direct
    assert_admitted(recovery, state)
    assert state['_inventory_state'] is not old
    assert state['_inventory_state']['controller_type'] is not old['controller_type']
