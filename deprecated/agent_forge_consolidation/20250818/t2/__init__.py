"""
Transformer² (T²) - Dynamic expert specialization and low-rank mixing.
Implements two-pass dispatch with singular component mixing for efficient inference.
"""

from .features import FeatureExtractor
from .mixer import T2Mixer

__all__ = ["T2Mixer", "FeatureExtractor"]
