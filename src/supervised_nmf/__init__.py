"""Supervised NMF classification, based on Chang Cheng's research scripts."""

from .classifier import NMFClassifier
from .decomposition import FeatureSPA
from .screening import BinaryScreening

__author__ = "Chang Cheng"
__version__ = "0.1.0a0"
__all__ = ["NMFClassifier", "FeatureSPA", "BinaryScreening"]

