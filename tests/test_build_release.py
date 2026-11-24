"""A failed source build preserves the contents of the release directory."""
from contextlib import chdir
from pathlib import Path
import runpy
from shutil import copyfile
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("prior_pair", [False, True], ids=["first-build", "rebuild"])
def test_sdist_failure_preserves_dist(tmp_path, monkeypatch, prior_pair):
    project = tmp_path / "project"
    script = project / "scripts" / "build_release.py"
    script.parent.mkdir(parents=True)
    copyfile(Path(__file__).resolve().parents[1] / "scripts" / script.name, script)
    dist = project / "dist"
    dist.mkdir()
    wheel = "example-1.0-py3-none-any.whl"
    sdist = "example-1.0.tar.gz"
    original = {"notes.txt": b"keep these notes"}
    if prior_pair:
        original.update({wheel: b"previous wheel", sdist: b"previous sdist"})
    for name, contents in original.items():
        (dist / name).write_bytes(contents)

    wheel_built = False

    def build_wheel(directory):
        nonlocal wheel_built
        (Path(directory) / wheel).write_bytes(b"new wheel")
        wheel_built = True
        return wheel

    def build_sdist(directory):
        assert wheel_built
        (Path(directory) / sdist).write_bytes(b"partial sdist")
        raise OSError("injected sdist failure")

    backend = ModuleType("setuptools.build_meta")
    backend.build_wheel = build_wheel
    backend.build_sdist = build_sdist
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    with chdir(tmp_path), pytest.raises(OSError, match="injected sdist failure"):
        runpy.run_path(str(script), run_name="__main__")

    assert {path.name for path in dist.iterdir()} == set(original)
    for name, contents in original.items():
        assert (dist / name).read_bytes() == contents
