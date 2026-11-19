"""Supervised NMF classification, based on Chang Cheng's research scripts."""

import sys as _sys

from .decomposition import FeatureSPA
from .screening import BinaryScreening
# Preserve numerical capability captures when the package is first imported.
from . import _mu, _mu_rowwise

__author__ = "Chang Cheng"
__version__ = "0.5.1"
__all__ = ["NMFClassifier", "FeatureSPA", "BinaryScreening"]


def __getattr__(name):
    if name == "NMFClassifier":
        from .classifier import NMFClassifier
        globals()[name] = NMFClassifier
        return NMFClassifier
    if name == "classifier":
        from importlib import import_module
        module = import_module(".classifier", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"classifier"})


# Reload rebinds an already available classifier, just as the eager import did.
# Before first use, reloading retains the deliberately deferred public export.
if ("NMFClassifier" in globals() or "classifier" in globals()
        or __name__ + ".classifier" in _sys.modules):
    from .classifier import NMFClassifier
