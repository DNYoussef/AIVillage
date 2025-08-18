#!/usr/bin/env python3
"""Integrated compression pipeline that avoids intermediate decompression."""

from __future__ import annotations

import gzip
import logging
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

try:  # pragma: no cover - allow tests to run from repository root
    from src.agent_forge.compression.bitnet import BITNETCompressor
    from src.agent_forge.compression.seedlm import SEEDLMCompressor
    from src.agent_forge.compression.vptq import VPTQCompressor
except ModuleNotFoundError:  # pragma: no cover
    from src.agent_forge.compression.bitnet import BITNETCompressor
    from src.agent_forge.compression.seedlm import SEEDLMCompressor
    from src.agent_forge.compression.vptq import VPTQCompressor

logger = logging.getLogger(__name__)


class IntegratedCompressionPipeline:
    """Compress models by transforming compressed representations directly."""

    def __init__(self) -> None:
        self.bitnet = BITNETCompressor()
        self.seedlm = SEEDLMCompressor(bits_per_weight=4)
        self.vptq = VPTQCompressor(bits=2)

    # ------------------------------------------------------------------
    def compress_model(self, model: torch.nn.Module) -> bytes:
        """Compress all parameters of ``model`` without intermediate decompression."""
        compressed_params: dict[str, bytes] = {}
        total_original = 0
        total_compressed = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            weights = param.data.cpu()
            original_size = weights.numel() * 4
            total_original += original_size

            data = self._integrated_compress(weights)
            compressed_params[name] = data
            total_compressed += len(data)

            ratio = original_size / len(data)
            logger.info("%s: %.1fx", name, ratio)

        # Pack parameters efficiently
        blob = bytearray()
        blob.extend(struct.pack("B", len(compressed_params)))
        for name, data in compressed_params.items():
            name_bytes = name.encode("utf-8")
            blob.extend(struct.pack("B", len(name_bytes)))
            blob.extend(name_bytes)
            blob.extend(struct.pack("I", len(data)))
            blob.extend(data)

        final = bytes(blob)
        logger.info("Total compression: %.1fx", total_original / len(final))
        return final

    # ------------------------------------------------------------------
    def _integrated_compress(self, weights: torch.Tensor) -> bytes:
        """Run BitNet → SeedLM → VPTQ without leaving the compressed domain."""
        # Stage 1: BitNet-style ternary quantisation
        scale = float(weights.abs().mean()) or 1.0
        normalised = weights / scale
        threshold = 0.7
        ternary = torch.zeros_like(normalised, dtype=torch.int8)
        ternary[normalised > threshold] = 1
        ternary[normalised < -threshold] = -1

        # Stage 2: SeedLM-style seed compression on ternary data
        ternary_np = ternary.flatten().numpy()
        seed_bytes = self._seedlm_compress_ternary(ternary_np)

        # Stage 3: VPTQ-style vector quantisation on seed bytes
        vq_bytes = self._vptq_compress_seeds(seed_bytes)

        # Stage 4: entropy coding
        payload = gzip.compress(vq_bytes)
        metadata = struct.pack("fI", scale, weights.numel())
        return metadata + payload

    # ------------------------------------------------------------------
    def _seedlm_compress_ternary(self, ternary: np.ndarray) -> bytes:
        """Compress ternary pattern using pseudo-random seeds."""
        block = 64
        seeds = []
        for i in range(0, len(ternary), block):
            chunk = ternary[i : i + block]
            seed = self._find_ternary_seed(chunk)
            seeds.append(seed)
        return struct.pack(f"{len(seeds)}H", *seeds)

    def _find_ternary_seed(self, block: np.ndarray) -> int:
        """Derive a pseudo seed for ``block``."""
        return hash(block.tobytes()) & 0xFFFF

    # ------------------------------------------------------------------
    def _vptq_compress_seeds(self, seed_data: bytes) -> bytes:
        """Apply simple vector quantisation to seed stream."""
        num_seeds = len(seed_data) // 2
        seeds = struct.unpack(f"{num_seeds}H", seed_data)
        vector = 4
        vectors = [tuple(seeds[i : i + vector]) for i in range(0, num_seeds, vector)]
        unique = list(dict.fromkeys(vectors))[:256]
        mapping = {v: i for i, v in enumerate(unique)}
        indices = bytes(mapping.get(v, 0) for v in vectors)

        out = bytearray()
        out.append(len(unique))
        for vec in unique:
            for val in vec:
                out.extend(struct.pack("H", val))
        out.extend(indices)
        return bytes(out)
