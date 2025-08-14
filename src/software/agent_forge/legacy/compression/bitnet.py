"""BitNet: ternary 1.58-bit quantization utilities."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BITNETCompressor:
    """Ternary quantization compressor.

    This implementation follows the 1.58-bit BitNet approach where each
    weight is mapped to ``{-1, 0, 1}`` and stored using two bits.  A single
    scale value per tensor is retained to reconstruct approximate floating
    point weights.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self.bits_per_weight = 2  # packed 4 values per byte
        logger.info("BitNet 1.58-bit quantization initialized")

    # ------------------------------------------------------------------
    def compress(self, weights: torch.Tensor) -> dict[str, object]:
        """Quantize ``weights`` to ternary representation.

        Args:
            weights: tensor of float32 values

        Returns:
            dict containing packed bytes and scale information
        """
        original_shape = tuple(weights.shape)
        flat = weights.flatten()
        scale = flat.abs().mean()
        if scale == 0:
            scale = torch.tensor(1.0, dtype=flat.dtype)
        norm = flat / scale
        ternary = torch.zeros_like(norm, dtype=torch.int8)
        ternary[norm > self.threshold] = 1
        ternary[norm < -self.threshold] = -1
        packed = _pack_ternary(ternary)
        return {
            "packed_weights": packed,
            "scale": float(scale),
            "original_shape": original_shape,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    def decompress(self, compressed: dict[str, object]) -> torch.Tensor:
        """Reconstruct tensor from compressed representation."""
        packed = compressed["packed_weights"]
        scale = torch.tensor(compressed["scale"], dtype=torch.float32)
        shape = compressed["original_shape"]
        num_weights = int(np.prod(shape))
        ternary = _unpack_ternary(packed, num_weights)
        weights = ternary * scale
        return weights.reshape(shape)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _pack_ternary(ternary: torch.Tensor) -> bytes:
    """Pack ternary values into bytes (2 bits per value)."""
    vals = (ternary + 1).to(torch.uint8).cpu().numpy()
    packed = bytearray()
    for i in range(0, len(vals), 4):
        chunk = vals[i : i + 4]
        if len(chunk) < 4:
            chunk = np.pad(chunk, (0, 4 - len(chunk)), constant_values=1)
        byte = (chunk[0] << 6) | (chunk[1] << 4) | (chunk[2] << 2) | chunk[3]
        packed.append(int(byte))
    return bytes(packed)


def _unpack_ternary(packed: bytes, n: int) -> torch.Tensor:
    """Unpack bytes produced by :func:`_pack_ternary`."""
    vals = []
    for byte in packed:
        vals.extend(
            [
                (byte >> 6) & 0b11,
                (byte >> 4) & 0b11,
                (byte >> 2) & 0b11,
                byte & 0b11,
            ]
        )
    vals = vals[:n]
    ternary = torch.tensor(vals, dtype=torch.float32) - 1
    return ternary


# Convenience wrapper ---------------------------------------------------


def compress(weights: torch.Tensor) -> dict[str, object]:
    """Compress ``weights`` using :class:`BITNETCompressor`."""
    return BITNETCompressor().compress(weights)
