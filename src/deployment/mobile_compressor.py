"""Mobile-optimized compression with automatic method selection."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.core.compression.unified_compressor import UnifiedCompressor

logger = logging.getLogger(__name__)


class MobileCompressor:
    """Compression tuned for different mobile device profiles."""

    DEVICE_PROFILES = {
        "low_end": {
            "memory_mb": 1500,
            "target_compression": 50.0,
            "prefer_simple": True,
        },
        "mid_range": {
            "memory_mb": 3000,
            "target_compression": 20.0,
            "prefer_simple": False,
        },
        "high_end": {
            "memory_mb": 4000,
            "target_compression": 10.0,
            "prefer_simple": False,
        },
    }

    def __init__(self, device_profile: str = "low_end") -> None:
        self.profile = self.DEVICE_PROFILES[device_profile]
        self.compressor = UnifiedCompressor(
            target_device="mobile",
            memory_limit_mb=self.profile["memory_mb"],
            target_compression=self.profile["target_compression"],
        )
        logger.info("MobileCompressor initialized for %s devices", device_profile)

    def prepare_model_for_device(self, model_path: str | Path) -> dict[str, object]:
        model = torch.load(Path(model_path), map_location="cpu")
        param_count = sum(p.numel() for p in model.parameters())
        original_mb = param_count * 4 / 1024 / 1024
        logger.info("Preparing %.1fMB model for mobile", original_mb)

        compressed = self.compressor.compress(model)
        package = {
            "model_data": compressed["data"],
            "compression_method": compressed["method"],
            "device_profile": self.profile,
            "original_size_mb": original_mb,
            "compressed_size_mb": len(compressed["data"]) / 1024 / 1024,
            "compression_ratio": original_mb * 1024 * 1024 / len(compressed["data"]),
            "decompressor_required": compressed["method"],
            "sprint9_compatible": compressed["method"] == "simple",
        }
        if package["compressed_size_mb"] > self.profile["memory_mb"] * 0.5:
            logger.warning("Model may not fit in %dMB device memory", self.profile["memory_mb"])
        return package
