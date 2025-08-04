"""Unified compression pipeline entry points.

This module consolidates all compression interfaces in the repository into a
single set of helper functions.  Internally it relies on
:class:`core.compression.unified_compressor.UnifiedCompressor` which selects
between a simple quantizer and the multi-stage advanced compressor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.compression.unified_compressor import UnifiedCompressor

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def compress(
    model: torch.nn.Module | str | Path,
    *,
    target_device: str = "mobile",
    memory_limit_mb: int = 2048,
    target_compression: float | None = None,
) -> dict[str, object]:
    """Compress ``model`` using the :class:`UnifiedCompressor`.

    Parameters mirror :class:`UnifiedCompressor` and are forwarded directly to
    it.  A new compressor instance is created for each call which keeps the
    interface stateless and easy to use in tests.
    """
    compressor = UnifiedCompressor(
        target_device=target_device,
        memory_limit_mb=memory_limit_mb,
        target_compression=target_compression,
    )
    return compressor.compress(model)


def decompress(
    compressed: dict[str, object],
) -> torch.nn.Module | dict[str, torch.Tensor]:
    """Decompress data produced by :func:`compress`.

    The compressor type is inferred from the metadata contained in ``compressed``.
    """
    compressor = UnifiedCompressor()
    return compressor.decompress(compressed)
