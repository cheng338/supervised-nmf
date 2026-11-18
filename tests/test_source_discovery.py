"""Fresh implementation discovery and its conservative inspection fallbacks."""
from contextlib import contextmanager
import errno
from functools import wraps
import hashlib
import importlib
import importlib.machinery
import inspect
import linecache
import os
from pathlib import Path
import sys
import types
from unittest.mock import Mock
import warnings

import pytest

from supervised_nmf import _identity as identity


SOURCE = """from abc import ABCMeta
class Base: pass
class Left(Base): pass
class Right(Base): pass
class Leaf(Left, Right): pass
class Other(Base): pass
class ABCLeaf(Left, Right, metaclass=ABCMeta): pass
class CustomMeta(type): pass
class CustomLeaf(Left, Right, metaclass=CustomMeta): pass
def callback(value):
    return value
# revision A
"""


@pytest.fixture
def external(tmp_path, monkeypatch):
    path = tmp_path / "implementation.py"
    path.write_text(SOURCE, encoding="utf-8")
    name = "_test_source_discovery_external"
    assert name not in sys.modules
    module = types.ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(SOURCE, str(path), "exec"), vars(module))
    return module, path


def expected_sources(module, path, names):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {f"{module.__name__}.{name}@{path.resolve().as_posix()}": digest
            for name in names}


@contextmanager
def filesystem_calls(path, monkeypatch):
    """Count only this fixture's I/O; the is_file spy selects inspect fallback."""
    calls = dict(resolve=0, is_file=0, read_bytes=0)
    with monkeypatch.context() as scoped:
        for name in calls:
            original = getattr(Path, name)

            def observe(self, *args, _name=name, _original=original, **kwargs):
                if self == path:
                    calls[_name] += 1
                return _original(self, *args, **kwargs)

            scoped.setattr(Path, name, observe)
        yield calls


@pytest.mark.parametrize("name", ["Leaf", "ABCLeaf"])
def test_ordinary_mro_keeps_each_source_guard_and_one_resolution(external, monkeypatch, name):
    module, path = external
    expected = expected_sources(module, path, [name, "Left", "Right", "Base"])
    with filesystem_calls(path, monkeypatch) as calls:
        assert identity.implementation_sources(getattr(module, name)) == expected
    assert calls == dict(resolve=1, is_file=4, read_bytes=1)


@pytest.mark.parametrize("name", ["Leaf", "ABCLeaf"])
def test_existing_python_class_source_avoids_redundant_exists(external, monkeypatch, name):
    module, path = external
    expected = expected_sources(module, path, [name, "Left", "Right", "Base"])
    seen = []
    original = inspect.os.path.exists

    def exists(filename):
        if str(filename) == str(path):
            seen.append(filename)
        return original(filename)

    with monkeypatch.context() as scoped:
        scoped.setattr(inspect.os.path, "exists", exists)
        assert identity.implementation_sources(getattr(module, name)) == expected
    # Unknown stdlib shapes may conservatively retain ordinary inspection.
    expected_visits = 0 if identity._SOURCE_LOOKUP_HAS_STDLIB_ORIGIN else 4
    assert len(seen) == expected_visits


def test_resolution_is_scoped_to_one_class_walk_and_one_scan(external, monkeypatch):
    module, path = external
    expected = expected_sources(module, path, ["Leaf", "Left", "Right", "Base", "Other"])
    for _ in range(2):
        with filesystem_calls(path, monkeypatch) as calls:
            assert identity.implementation_sources([module.Leaf, module.Other, module.Leaf]) == expected
        assert calls == dict(resolve=2, is_file=6, read_bytes=1)


def test_digest_reuse_reads_every_source_in_current_order(external, monkeypatch):
    module, first = external

    class Second:
        pass

    second = Path(__file__).resolve()
    expected = expected_sources(module, first, ["Base"])
    expected.update(expected_sources(sys.modules[__name__], second, [Second.__qualname__]))
    reads = []
    digest = Mock(wraps=hashlib.sha256)
    original_read = Path.read_bytes

    def read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(identity, "hashlib", types.SimpleNamespace(sha256=digest))
    previous = {}
    for graph, ordered_paths, new_hashes in [
        ([module.Base, Second], [first, second], 2),
        ([module.Base, Second], [first, second], 0),
        ([Second, module.Base], [second, first], 2),
    ]:
        reads.clear()
        digest.reset_mock()
        assert identity.implementation_sources(*graph, _previous_files=previous) == expected
        assert reads == ordered_paths
        assert digest.call_count == new_hashes


@pytest.mark.parametrize("kind", ["mutable", "bytes_subclass", "oversize"])
def test_ineligible_source_bytes_are_hashed_without_retention(external, monkeypatch, kind):
    module, path = external

    class CustomBytes(bytes):
        def __eq__(self, other):
            raise AssertionError("A bytes subclass must not participate in digest reuse")

    if kind == "oversize":
        path.write_bytes(b"#" * (512 * 1024 + 1))
    expected = expected_sources(module, path, ["Base"])
    previous = {}
    assert identity.implementation_sources(module.Base, _previous_files=previous) == expected

    original_read = Path.read_bytes
    wrap = {"mutable": bytearray, "bytes_subclass": CustomBytes, "oversize": bytes}[kind]
    monkeypatch.setattr(Path, "read_bytes", lambda source: wrap(original_read(source)))
    digest = Mock(wraps=hashlib.sha256)
    monkeypatch.setattr(identity, "hashlib", types.SimpleNamespace(sha256=digest))
    empty = {}
    assert identity.implementation_sources(module.Base, _previous_files=previous) == expected
    assert identity.implementation_sources(module.Base, _previous_files=empty) == expected
    assert digest.call_count == 2
    assert not empty


@pytest.mark.parametrize("mode", ["full_class", "function"])
def test_full_identity_and_function_keep_inspect_discovery(external, monkeypatch, mode):
    module, path = external
    seen = []
    original = inspect.getsourcefile

    def discover(value):
        seen.append(value)
        return original(value)

    with monkeypatch.context() as scoped:
        scoped.setattr(inspect, "getsourcefile", discover)
        with filesystem_calls(path, monkeypatch) as calls:
            if mode == "full_class":
                result = identity.configuration_identity(module.Leaf)
            else:
                result = identity.implementation_sources(module.callback)
    if mode == "full_class":
        assert seen == list(module.Leaf.__mro__)
        assert calls == dict(resolve=4, is_file=4, read_bytes=1)
        assert result["implementations"][-1]["source"] == {"native_module": "builtins"}
    else:
        assert seen == [module.callback]
        assert result == expected_sources(module, path, ["callback"])
        assert calls == dict(resolve=1, is_file=1, read_bytes=1)


@pytest.mark.parametrize("mode", ["metaclass", "module_subclass", "module_getattr",
                                  "filename_subclass", "module_name_subclass", "qualname_subclass"])
def test_custom_metadata_retains_per_parent_discovery(external, monkeypatch, mode):
    module, path = external

    class Text(str):
        pass

    class Module(types.ModuleType):
        pass

    cls = module.CustomLeaf if mode == "metaclass" else module.Leaf
    if mode == "module_subclass":
        module.__class__ = Module
    elif mode == "module_getattr":
        module.__getattr__ = lambda name: None
    elif mode == "filename_subclass":
        module.__file__ = Text(str(path))
    elif mode == "module_name_subclass":
        cls.__module__ = Text(cls.__module__)
    elif mode == "qualname_subclass":
        cls.__qualname__ = Text(cls.__qualname__)
    expected = expected_sources(module, path, [cls.__name__, "Left", "Right", "Base"])
    with filesystem_calls(path, monkeypatch) as calls:
        assert identity.implementation_sources(cls) == expected
    assert calls == dict(resolve=4, is_file=4, read_bytes=1)


@pytest.mark.parametrize("mode", ["edit", "replace"])
@pytest.mark.parametrize("reuse", [False, True])
def test_same_size_restored_mtime_changes_are_fresh(external, mode, reuse):
    module, path = external
    previous = {} if reuse else None
    before = identity.implementation_sources(module.Leaf, _previous_files=previous)
    saved = path.stat()
    replacement = path.read_bytes().replace(b"revision A", b"revision B")
    if mode == "edit":
        path.write_bytes(replacement)
    else:
        other = path.with_name("replacement.py")
        other.write_bytes(replacement)
        os.replace(other, path)
    os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
    assert path.stat().st_size == saved.st_size
    assert path.stat().st_mtime_ns == saved.st_mtime_ns
    after = identity.implementation_sources(module.Leaf, _previous_files=previous)
    assert after == expected_sources(module, path, ["Leaf", "Left", "Right", "Base"])
    assert after != before
    path.unlink()
    with pytest.raises(ValueError, match="implementation source is unavailable"):
        identity.implementation_sources(module.Leaf, _previous_files=previous)


@pytest.mark.parametrize("action", ["retarget", "delete"])
def test_parameter_callback_cannot_reuse_previous_walk_resolution(external, action):
    module, path = external
    alternate = path.with_name("alternate.py")
    alternate.write_text(SOURCE + "\n# alternate\n", encoding="utf-8")
    first = expected_sources(module, path, ["Leaf", "Left", "Right", "Base"])

    class ChangeSource:
        def get_params(self, deep=False):
            assert deep is False
            if action == "retarget":
                module.__file__ = str(alternate)
            else:
                path.unlink()
            return {}

    if action == "delete":
        with pytest.raises(ValueError, match=r"configuration\[2\].*source is unavailable"):
            identity.implementation_sources(module.Leaf, ChangeSource(), module.Other)
    else:
        result = identity.implementation_sources(module.Leaf, ChangeSource(), module.Other)
        assert all(result[key] == value for key, value in first.items())
        second = expected_sources(module, alternate, ["Other", "Base"])
        assert all(result[key] == value for key, value in second.items())


@pytest.mark.parametrize("boundary", ["parameter_callback", "module_callback"])
def test_same_filename_symlink_retarget_is_observed(external, boundary):
    module, first = external
    second = first.with_name("second.py")
    second.write_text(SOURCE + "\n# different implementation\n", encoding="utf-8")
    alias = first.with_name("alias.py")
    try:
        alias.symlink_to(first)
    except OSError as error:
        if (getattr(error, "winerror", None) == 1314
                or error.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EPERM}):
            pytest.skip(f"Test-owned symbolic link unavailable: {error}")
        raise
    module.__file__ = str(alias)

    def retarget():
        alias.unlink()
        alias.symlink_to(second)

    if boundary == "parameter_callback":
        class Retarget:
            def get_params(self, deep=False):
                assert deep is False
                retarget()
                return {}

        result = identity.implementation_sources(module.Leaf, Retarget(), module.Other)
        expected = expected_sources(module, first, ["Leaf", "Left", "Right", "Base"])
        expected.update(expected_sources(module, second, ["Other", "Base"]))
        assert all(result[key] == value for key, value in expected.items())
    else:
        visits = 0

        class DynamicModule(types.ModuleType):
            def __getattribute__(self, name):
                nonlocal visits
                if name == "__file__":
                    visits += 1
                    if visits == 3:
                        retarget()
                return super().__getattribute__(name)

        module.__class__ = DynamicModule
        result = identity.implementation_sources(module.Leaf)
        assert visits >= 3
        assert len(result) == 4
        expected_hashes = {path.resolve().as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (first, second)}
        assert {key.rsplit("@", 1)[1] for key in result} == set(expected_hashes)
        assert all(value == expected_hashes[key.rsplit("@", 1)[1]]
                   for key, value in result.items())
        # Full identity retains ordinary inspection and supplies an independent
        # discovery oracle; reset the actual link and callback before calling it.
        alias.unlink()
        alias.symlink_to(first)
        visits = 0
        full = identity.configuration_identity(module.Leaf)
        oracle = {f"{item['type']}@{item['source']['path']}": item["source"]["sha256"]
                  for item in full["implementations"] if "path" in item["source"]}
        assert visits >= 3
        assert result == oracle
    assert vars(module)["__file__"] == str(alias)
    assert alias.resolve() == second.resolve()


def test_each_mro_entry_checks_file_presence(external, monkeypatch):
    module, path = external
    original = Path.is_file
    calls = 0

    def is_file(self):
        nonlocal calls
        if self == path:
            calls += 1
            if calls == 2:
                path.unlink()
        return original(self)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "is_file", is_file)
        with pytest.raises(ValueError, match="source is unavailable"):
            identity.implementation_sources(module.Leaf)
    assert calls == 2


@pytest.mark.parametrize("operation,error_type", [("resolve", OSError),
                                                  ("read_bytes", PermissionError),
                                                  ("read_bytes", RuntimeError)])
def test_failed_walk_clears_scope_and_preserves_error(external, monkeypatch, operation, error_type):
    module, path = external
    encoder = identity._Encoder(sources_only=True)
    marker = error_type("controlled source failure")
    original = getattr(Path, operation)

    def fail(self, *args, **kwargs):
        if self == path:
            raise marker
        return original(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, operation, fail)
        if operation == "read_bytes" and issubclass(error_type, OSError):
            with pytest.raises(ValueError, match="source cannot be read \\(PermissionError\\)"):
                encoder.encode(module.Leaf)
        else:
            with pytest.raises(error_type) as raised:
                encoder.encode(module.Leaf)
            assert raised.value is marker
    assert encoder._mro_resolutions is None
    assert module.Leaf not in encoder.class_records
    assert not encoder.active
    encoder.encode(module.Leaf)
    assert encoder.sources == expected_sources(module, path, ["Leaf", "Left", "Right", "Base"])
    assert encoder._mro_resolutions is None


@pytest.mark.parametrize("kind", ["noncanonical", "bytecode", "extension"])
def test_noncanonical_suffix_uses_inspect_rules(external, kind):
    module, path = external
    suffix = {"noncanonical": ".txt", "bytecode": ".pyc",
              "extension": importlib.machinery.EXTENSION_SUFFIXES[0]}[kind]
    other = path.with_suffix(suffix)
    other.write_bytes(b"different source bytes\n")
    module.__file__ = str(other)
    if kind == "extension":
        with pytest.raises(ValueError, match="source is unavailable"):
            identity.implementation_sources(module.Leaf)
    else:
        selected = path if kind == "bytecode" else other
        assert identity.implementation_sources(module.Leaf) == expected_sources(
            module, selected, ["Leaf", "Left", "Right", "Base"])


@pytest.mark.parametrize("kind", ["no_loader", "loader", "linecache"])
def test_missing_python_file_cannot_be_replaced_by_loader_or_linecache(external, kind):
    module, path = external
    path.unlink()
    if kind == "loader":
        module.__loader__ = object()
    try:
        if kind == "linecache":
            linecache.cache[str(path)] = (0, None, [], str(path))
        with pytest.raises(ValueError, match="source is unavailable"):
            identity.implementation_sources(module.Leaf)
    finally:
        linecache.cache.pop(str(path), None)


def test_native_module_missing_source_retains_native_identity(external, monkeypatch):
    module, path = external
    path.unlink()
    native_name = "sklearn._source_discovery_native_fixture"
    monkeypatch.setitem(sys.modules, native_name, module)
    for cls in (module.Leaf, module.Left, module.Right, module.Base):
        cls.__module__ = native_name
    assert identity.implementation_sources(module.Leaf) == {}
    record = identity.configuration_identity(module.Leaf)
    assert [item["source"] for item in record["implementations"]] == [
        {"native_module": native_name}] * 4 + [{"native_module": "builtins"}]


@pytest.mark.parametrize("kind", ["absent", "none"])
def test_namespace_without_file_is_rejected(external, kind):
    module, _ = external
    if kind == "absent":
        del module.__file__
    else:
        module.__file__ = None
    with pytest.raises(ValueError, match="source is unavailable"):
        identity.implementation_sources(module.Leaf)


@pytest.mark.parametrize("when", ["before_import", "after_import"])
@pytest.mark.parametrize("name", ["getsourcefile", "getfile", "is_file"])
def test_wrapped_discovery_entrypoints_remain_authoritative(external, monkeypatch, when, name):
    module, path = external
    alternate = path.with_name("observed.py")
    alternate.write_text("# observed source\n", encoding="utf-8")
    owner = Path if name == "is_file" else inspect
    original = getattr(owner, name)
    calls = []

    @wraps(original)
    def changed(value):
        if (value == path if name == "is_file" else value in module.Leaf.__mro__[:-1]):
            calls.append(value)
            return False if name == "is_file" else str(alternate)
        return original(value)

    # Reload the actual production module to exercise capture before import.
    # Restore its same dictionary and original object identities, not just semantics.
    namespace = vars(identity)
    saved = namespace.copy()
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(owner, name, changed)
            if when == "before_import":
                assert importlib.reload(identity) is identity
            if name == "is_file":
                with pytest.raises(ValueError, match="source is unavailable"):
                    identity.implementation_sources(module.Leaf)
            else:
                assert identity.implementation_sources(module.Leaf) == expected_sources(
                    module, alternate, ["Leaf", "Left", "Right", "Base"])
            assert calls
    finally:
        if when == "before_import":
            namespace.clear()
            namespace.update(saved)
    assert vars(identity) is namespace
    assert identity._Encoder is saved["_Encoder"]
    assert identity.implementation_sources is saved["implementation_sources"]
    assert getattr(owner, name) is original


@pytest.mark.parametrize("error_type", [PermissionError, RuntimeError])
def test_is_file_hook_errors_are_not_swallowed(external, monkeypatch, error_type):
    module, path = external
    marker = error_type("controlled predicate failure")
    original = Path.is_file

    def fail(self):
        if self == path:
            raise marker
        return original(self)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "is_file", fail)
        with pytest.raises(error_type) as raised:
            identity.implementation_sources(module.Leaf)
    assert raised.value is marker


@pytest.mark.parametrize("name", ["getsourcefile", "getfile"])
@pytest.mark.parametrize("error_type", [TypeError, OSError, RuntimeError])
def test_inspect_hook_error_boundary_is_preserved(external, monkeypatch, name, error_type):
    module, _ = external
    marker = error_type("controlled inspect failure")

    def fail(value):
        raise marker

    with monkeypatch.context() as scoped:
        scoped.setattr(inspect, name, fail)
        if error_type in (TypeError, OSError):
            with pytest.raises(ValueError, match="source is unavailable"):
                identity.implementation_sources(module.Leaf)
        else:
            with pytest.raises(RuntimeError) as raised:
                identity.implementation_sources(module.Leaf)
            assert raised.value is marker


@pytest.mark.parametrize("action", ["always", "error"])
def test_callback_warning_policy_is_preserved(action):
    class WarningParameters:
        def get_params(self, deep=False):
            warnings.warn("source inspection warning", UserWarning)
            return {}

    original_filters = list(warnings.filters)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter(action)
        if action == "error":
            with pytest.raises(ValueError, match="parameters cannot be inspected \\(UserWarning\\)"):
                identity.implementation_sources(WarningParameters())
            assert not seen
        else:
            assert identity.implementation_sources(WarningParameters())
            assert [(item.category, str(item.message)) for item in seen] == [
                (UserWarning, "source inspection warning")]
    assert warnings.filters == original_filters
