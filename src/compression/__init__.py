"""Simple compression module for mobile optimization.

This module provides real, working compression for mobile devices.
Target: 4x compression ratio for 2GB phones.
"""

"""Convenience imports for the consolidated compression pipeline."""

from src.core.compression.unified_compressor import UnifiedCompressor

from .pipeline import compress, decompress
from .simple_quantizer import CompressionError, SimpleQuantizer
from .test_model_generator import create_test_model

__all__ = [
    "CompressionError",
    "SimpleQuantizer",
    "UnifiedCompressor",
    "compress",
    "create_test_model",
    "decompress",
]
