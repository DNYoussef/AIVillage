"""Compatibility wrapper for :mod:`core.compression.simple_quantizer`.

This module previously contained a standalone implementation of the
``SimpleQuantizer``.  The codebase now uses the implementation located in
``core.compression`` so we re-export that class here to provide a single
source of truth while maintaining backwards compatibility.
"""

from src.core.compression.simple_quantizer import CompressionError, SimpleQuantizer

__all__ = ["CompressionError", "SimpleQuantizer"]
