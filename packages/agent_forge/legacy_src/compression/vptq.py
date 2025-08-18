"""VPTQ: Vector post-training quantisation."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class VPTQCompressor:
    """Vector quantisation using a small learned codebook."""

    def __init__(self, bits: int = 2, vector_dim: int = 4, iterations: int = 5) -> None:
        self.bits = bits
        self.codebook_size = 2**bits
        self.vector_dim = vector_dim
        self.iterations = iterations
        logger.info(
            "VPTQ initialised: %s bits, codebook=%s, vecdim=%s",
            bits,
            self.codebook_size,
            vector_dim,
        )

    # ------------------------------------------------------------------
    def compress(self, weights: torch.Tensor) -> dict[str, object]:
        original_shape = tuple(weights.shape)
        flat = weights.flatten()
        pad = (-len(flat)) % self.vector_dim
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
        vectors = flat.view(-1, self.vector_dim)

        codebook = self._init_codebook(vectors)
        codebook, indices = self._optimize(vectors, codebook)
        scale = torch.std(flat)
        offset = torch.mean(flat)
        return {
            "codebook": codebook,
            "indices": indices,
            "scale": scale,
            "offset": offset,
            "original_shape": original_shape,
            "pad_length": pad,
            "vector_dim": self.vector_dim,
        }

    # ------------------------------------------------------------------
    def _init_codebook(self, vectors: torch.Tensor) -> torch.Tensor:
        n = vectors.size(0)
        codebook = torch.empty(self.codebook_size, self.vector_dim, device=vectors.device)
        idx = torch.randint(0, n, (1,))
        codebook[0] = vectors[idx]
        for i in range(1, self.codebook_size):
            dist = torch.cdist(vectors, codebook[:i])
            mins, _ = dist.min(dim=1)
            probs = mins**2
            if probs.sum() == 0:
                probs = torch.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1)
            codebook[i] = vectors[idx]
        return codebook

    def _optimize(self, vectors: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.iterations):
            dist = torch.cdist(vectors, codebook)
            indices = dist.argmin(dim=1)
            for i in range(self.codebook_size):
                mask = indices == i
                if mask.any():
                    codebook[i] = vectors[mask].mean(dim=0)
        dist = torch.cdist(vectors, codebook)
        indices = dist.argmin(dim=1)
        return codebook, indices

    # ------------------------------------------------------------------
    def decompress(self, compressed: dict[str, object]) -> torch.Tensor:
        codebook = compressed["codebook"]
        indices = compressed["indices"]
        vectors = codebook[indices]
        flat = vectors.flatten()
        if compressed["pad_length"]:
            flat = flat[: -compressed["pad_length"]]
        flat = flat * compressed["scale"] + compressed["offset"]
        return flat.view(compressed["original_shape"])


def compress(weights: torch.Tensor) -> dict[str, object]:
    """Convenience wrapper."""
    return VPTQCompressor().compress(weights)
