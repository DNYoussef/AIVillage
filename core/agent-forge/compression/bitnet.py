#!/usr/bin/env python3
"""
BitNet 1.58-bit Compression Module
Extracted from the main bitnet_compression.py phase for modular usage.

This module provides the core BitNet quantization classes that can be imported
and used independently from the main compression phase.
"""

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BitNetConfig:
    """Configuration for BitNet compression."""

    quantization_bits: float = 1.58  # BitNet 1.58-bit quantization
    preserve_embedding_precision: bool = True  # Keep embeddings in higher precision
    preserve_output_precision: bool = True  # Keep final layers in higher precision
    sparsity_threshold: float = 0.1  # Threshold for setting weights to 0
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42


class BitNetQuantizer:
    """
    Core BitNet 1.58-bit quantization implementation.

    Quantizes weights to {-1, 0, +1} with dynamic scaling and sparsity.
    """

    def __init__(self, config: BitNetConfig = None):
        self.config = config or BitNetConfig()
        self.quantization_stats = {
            "layers_quantized": 0,
            "total_parameters": 0,
            "quantized_parameters": 0,
            "sparsity_ratio": 0.0,
        }

    def quantize_tensor(self, tensor: torch.Tensor, preserve_precision: bool = False) -> dict[str, Any]:
        """
        Apply BitNet 1.58-bit quantization to a tensor.

        Args:
            tensor: Input tensor to quantize
            preserve_precision: If True, skip quantization for this tensor

        Returns:
            Dictionary containing quantized data and metadata
        """
        if preserve_precision or tensor.numel() < 1024:  # Keep small tensors unquantized
            return {
                "weights": tensor.cpu().numpy(),
                "scale": 1.0,
                "quantization_type": "none",
                "is_quantized": False,
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
            }

        # Calculate dynamic scale per channel/row for better precision
        if len(tensor.shape) >= 2:
            # Per-output-channel scaling for Linear layers
            scale = tensor.abs().mean(dim=list(range(1, len(tensor.shape))), keepdim=True)
        else:
            # Global scaling for 1D tensors
            scale = tensor.abs().mean()

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        # Normalize by scale
        normalized = tensor / scale

        # Apply sparsity threshold - set small values to 0
        sparsity_mask = tensor.abs() < (scale * self.config.sparsity_threshold)

        # Quantize to {-1, 0, +1}
        quantized = torch.sign(normalized)
        quantized[sparsity_mask] = 0

        # Calculate sparsity for statistics
        sparsity = (quantized == 0).float().mean().item()

        # Convert to int8 for storage efficiency
        quantized_int8 = quantized.to(torch.int8)

        # Update statistics
        self.quantization_stats["total_parameters"] += tensor.numel()
        self.quantization_stats["quantized_parameters"] += tensor.numel()
        self.quantization_stats["sparsity_ratio"] = (
            self.quantization_stats["sparsity_ratio"] * self.quantization_stats["layers_quantized"] + sparsity
        ) / (self.quantization_stats["layers_quantized"] + 1)
        self.quantization_stats["layers_quantized"] += 1

        return {
            "weights": quantized_int8.cpu().numpy(),
            "scale": scale.cpu().numpy(),
            "quantization_type": "bitnet_1.58",
            "is_quantized": True,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "sparsity": sparsity,
        }

    def dequantize_tensor(self, quantized_data: dict[str, Any]) -> torch.Tensor:
        """Dequantize a tensor from BitNet format."""
        if not quantized_data.get("is_quantized", False):
            # Return unquantized tensor as-is
            weights = np.array(quantized_data["weights"])
            return torch.from_numpy(weights).reshape(quantized_data["shape"])

        # Reconstruct quantized tensor
        weights = torch.from_numpy(quantized_data["weights"]).float()
        scale = torch.from_numpy(quantized_data["scale"]).float()

        # Restore original scale
        dequantized = weights * scale

        return dequantized.reshape(quantized_data["shape"])


class BITNETCompressor:
    """
    BitNet compression wrapper compatible with existing test infrastructure.

    This provides a compatible interface for tests expecting this class.
    """

    def __init__(self, config: dict = None):
        """Initialize BitNet compressor with config."""
        if config is None:
            config = {}

        # Convert dict config to BitNetConfig if needed
        if isinstance(config, dict):
            bitnet_config = BitNetConfig(**{k: v for k, v in config.items() if hasattr(BitNetConfig, k)})
        else:
            bitnet_config = config

        self.quantizer = BitNetQuantizer(bitnet_config)
        self.config = bitnet_config

    def compress(self, model: nn.Module, **kwargs) -> dict:
        """Compress a model using BitNet quantization."""
        try:
            logger.info("Starting BitNet compression...")

            # Basic compression logic
            compressed_layers = {}
            original_size = 0
            compressed_size = 0

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                    if hasattr(module, "weight") and module.weight is not None:
                        # Quantize weight
                        preserve = self._should_preserve_precision(name, module)
                        compressed_weight = self.quantizer.quantize_tensor(
                            module.weight.data, preserve_precision=preserve
                        )
                        compressed_layers[f"{name}.weight"] = compressed_weight

                        # Calculate sizes
                        original_size += module.weight.numel() * module.weight.element_size()
                        if compressed_weight.get("is_quantized", False):
                            # Rough estimate: 1.58 bits per weight + scale overhead
                            compressed_size += np.array(compressed_weight["weights"]).nbytes
                            compressed_size += np.array(compressed_weight["scale"]).nbytes
                        else:
                            compressed_size += np.array(compressed_weight["weights"]).nbytes

            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            result = {
                "success": True,
                "compression_ratio": compression_ratio,
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024),
                "quantization_stats": self.quantizer.quantization_stats,
                "compressed_layers": len(compressed_layers),
                "method": "BitNet-1.58",
            }

            logger.info(f"BitNet compression completed: {compression_ratio:.2f}x ratio")
            return result

        except Exception as e:
            logger.exception(f"BitNet compression failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "compression_ratio": 1.0,
                "original_size_mb": 0,
                "compressed_size_mb": 0,
            }

    def _should_preserve_precision(self, layer_name: str, module: nn.Module) -> bool:
        """Determine if a layer should preserve full precision."""
        # Preserve embedding layers
        if self.config.preserve_embedding_precision:
            if "embed" in layer_name.lower():
                return True

        # Preserve output layers
        if self.config.preserve_output_precision:
            if any(keyword in layer_name.lower() for keyword in ["output", "classifier", "head", "lm_head"]):
                return True

        # Preserve very small layers
        if hasattr(module, "weight") and module.weight.numel() < 1024:
            return True

        return False


# For backwards compatibility
def create_bitnet_compressor(config: dict = None) -> BITNETCompressor:
    """Factory function to create BitNet compressor."""
    return BITNETCompressor(config)
