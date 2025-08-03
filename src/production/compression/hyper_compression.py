"""Lightweight hyper-compression encoder.

This module provides a minimal implementation of the HyperCompression stage
from the research pipeline.  It simply applies ``zlib`` compression to the
byte representation produced by earlier stages.  The interface mirrors the
expected ``encode``/``decode`` API so the pipeline can experiment with more
sophisticated approaches in the future.
"""
from __future__ import annotations

import zlib
from typing import ByteString


class HyperCompressionEncoder:
    """Wrapper around :mod:`zlib` providing encode/decode helpers."""

    def encode(self, data: ByteString) -> bytes:  # pragma: no cover - thin wrapper
        return zlib.compress(data, level=9)

    def decode(self, data: ByteString) -> bytes:  # pragma: no cover - thin wrapper
        return zlib.decompress(data)


__all__ = ["HyperCompressionEncoder"]
