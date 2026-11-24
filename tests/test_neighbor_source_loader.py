"""Optional neighbor qualification reads only regular installed sources."""

from importlib.machinery import ModuleSpec, SourceFileLoader
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from supervised_nmf._neighbors import _installed_query_code


def source_module(path, wrapped):
    module = ModuleType("test_neighbor_definitions")
    module.__file__ = str(path)
    loader = object() if wrapped else SourceFileLoader(module.__name__, str(path))
    module.__loader__ = loader
    module.__spec__ = ModuleSpec(module.__name__, loader, origin=str(path))
    return module


@pytest.mark.parametrize("wrapped", [False, True])
def test_nonregular_neighbor_source_declines_before_any_read(tmp_path, wrapped):
    path = tmp_path / "_classification.py"
    if os.name == "nt":
        path.mkdir()
    else:
        os.mkfifo(path)
    module = source_module(path, wrapped)
    error = AssertionError("optional qualification tried to read nonregular source")
    with (
        patch.object(SourceFileLoader, "get_code", side_effect=error),
        patch.object(Path, "read_bytes", side_effect=error),
    ):
        assert _installed_query_code(module) == {}


def test_regular_source_ignores_standard_loader_instance_hooks(tmp_path):
    path = tmp_path / "_classification.py"
    path.write_text("class KNeighborsClassifier:\n    def predict(self, X): return X\n", encoding="utf-8")
    module = source_module(path, wrapped=False)
    with patch.object(module.__loader__, "get_code", side_effect=AssertionError("installed loader hook ran")):
        definitions = _installed_query_code(module)
    assert definitions["KNeighborsClassifier.predict"].co_filename == str(path)
