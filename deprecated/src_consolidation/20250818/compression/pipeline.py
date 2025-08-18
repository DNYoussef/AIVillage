"""Unified compression pipeline entry points.

This module consolidates all compression interfaces in the repository into a
single canonical interface.  It exposes a :class:`UnifiedCompressor` class
that wraps :class:`core.compression.unified_compressor.UnifiedCompressor` and
adds light‑weight configuration handling and optional evaluation hooks.  The
module also re‑exports the individual compression operators used by the
advanced pipeline so that callers only need to import from this file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.agent_forge.compression.bitnet import BITNETCompressor
from src.agent_forge.compression.bitnet import compress as bitnet_compress
from src.agent_forge.compression.seedlm import LinearFeedbackShiftRegister, SEEDLMCompressor
from src.agent_forge.compression.seedlm import compress as seedlm_compress
from src.agent_forge.compression.vptq import VPTQCompressor
from src.agent_forge.compression.vptq import compress as vptq_compress
from src.core.compression.unified_compressor import UnifiedCompressor as _CoreUnifiedCompressor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - used for type checkers only
    from pathlib import Path

    import torch


class CompressionConfig(BaseModel):
    """Minimal configuration model for :class:`UnifiedCompressor`.

    The production pipeline exposes a rich Pydantic configuration model.  The
    version here provides the commonly used parameters and can be expanded as
    needed without breaking the public API.
    """

    target_device: str = Field(default="mobile")
    memory_limit_mb: int = Field(default=2048, ge=1)
    target_compression: float | None = Field(default=None, ge=0.0)
    eval_before_after: bool = Field(default=False, description="Call evaluation hook before and after compression")


class UnifiedCompressor(_CoreUnifiedCompressor):
    """Canonical compressor with configuration and evaluation hooks."""

    def __init__(
        self,
        *,
        config: CompressionConfig | None = None,
        eval_hook: Callable[[object, str], dict[str, float]] | None = None,
        **kwargs: object,
    ) -> None:
        # Allow passing of raw kwargs for backwards compatibility
        if config is None:
            config = CompressionConfig(**kwargs)
        self.config = config
        self.eval_hook = eval_hook
        super().__init__(
            target_device=config.target_device,
            memory_limit_mb=config.memory_limit_mb,
            target_compression=config.target_compression,
        )

    # ------------------------------------------------------------------
    def compress(self, model: torch.nn.Module | str | Path) -> dict[str, object]:
        """Compress ``model`` with optional evaluation callbacks."""

        metrics_before = None
        if self.config.eval_before_after and self.eval_hook is not None:
            try:  # pragma: no cover - defensive
                metrics_before = self.eval_hook(model, "before")
            except Exception:  # pragma: no cover - evaluation failures shouldn't abort
                logger.warning("Pre-compression evaluation hook failed", exc_info=True)
        result = super().compress(model)
        if self.config.eval_before_after and self.eval_hook is not None:
            try:  # pragma: no cover - defensive
                metrics_after = self.eval_hook(result, "after")
                result["evaluation"] = {
                    "before": metrics_before,
                    "after": metrics_after,
                }
            except Exception:  # pragma: no cover - evaluation failures shouldn't abort
                logger.warning("Post-compression evaluation hook failed", exc_info=True)
        return result


def compress(
    model: torch.nn.Module | str | Path,
    *,
    target_device: str = "mobile",
    memory_limit_mb: int = 2048,
    target_compression: float | None = None,
    eval_hook: Callable[[object, str], dict[str, float]] | None = None,
) -> dict[str, object]:
    """Compress ``model`` using the :class:`UnifiedCompressor`.

    A new compressor instance is created for each call which keeps the
    interface stateless and easy to use in tests.
    """

    compressor = UnifiedCompressor(
        target_device=target_device,
        memory_limit_mb=memory_limit_mb,
        target_compression=target_compression,
        eval_hook=eval_hook,
    )
    return compressor.compress(model)


def decompress(
    compressed: dict[str, object],
) -> torch.nn.Module | dict[str, torch.Tensor]:
    """Decompress data produced by :func:`compress`."""

    compressor = UnifiedCompressor()
    return compressor.decompress(compressed)


__all__ = [
    "CompressionConfig",
    "UnifiedCompressor",
    "compress",
    "decompress",
    "BITNETCompressor",
    "VPTQCompressor",
    "SEEDLMCompressor",
    "LinearFeedbackShiftRegister",
    "bitnet_compress",
    "vptq_compress",
    "seedlm_compress",
]
