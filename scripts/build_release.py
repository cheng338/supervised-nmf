"""Build a local wheel and source archive with the current environment.

This does not install packages or publish anything. Build dependencies must
already be available, or use `python -m pip wheel .` in an authorized environment.
"""

import os
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel

root = Path(__file__).resolve().parents[1]
os.chdir(root)
build_wheel(str(root / "dist"))
build_sdist(str(root / "dist"))
