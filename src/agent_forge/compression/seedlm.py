"""SeedLM: lightweight seed-based weight compression."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LinearFeedbackShiftRegister:
    """Simple LFSR used for reproducible pseudo-random matrices."""

    def __init__(self, seed_length: int = 16) -> None:
        self.seed_length = seed_length

    def generate_matrix(self, seed: int, rows: int, cols: int) -> np.ndarray:
        rng = np.random.default_rng(seed % (2**self.seed_length))
        return rng.standard_normal((rows, cols), dtype=np.float32)


class SEEDLMCompressor:
    """Compress tensors using pseudo-random projections.

    This is a simplified implementation of the SeedLM approach.  Each block of
    ``C`` weights is approximated using a small latent dimension ``P`` and a
    pseudo-random projection generated from a seed.  Only the seed, quantised
    coefficients and a shared exponent are stored.
    """

    def __init__(self, bits_per_weight: int = 4, max_candidates: int = 16) -> None:
        self.bits_per_weight = bits_per_weight
        if bits_per_weight == 3:
            self.C, self.P = 12, 4
        elif bits_per_weight == 4:
            self.C, self.P = 8, 3
        else:
            msg = "Unsupported bit width"
            raise ValueError(msg)
        self.lfsr = LinearFeedbackShiftRegister(16)
        self.max_candidates = max_candidates
        self.Q = np.array(range(-8, 8), dtype=np.int8)  # 4-bit quantisation
        logger.info(
            "SeedLM initialised: %s bits, block=%s latent=%s",
            bits_per_weight,
            self.C,
            self.P,
        )

    # ------------------------------------------------------------------
    def compress(self, weights: torch.Tensor) -> dict[str, object]:
        original_shape = tuple(weights.shape)
        flat = weights.flatten().cpu().numpy()
        pad = (-len(flat)) % self.C
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
        blocks = flat.reshape(-1, self.C)

        seeds: list[int] = []
        coeffs: list[np.ndarray] = []
        exps: list[int] = []
        for block in blocks:
            max_val = np.max(np.abs(block))
            exp = int(np.floor(np.log2(max_val))) if max_val > 0 else 0
            scaled = block / (2**exp) if max_val > 0 else block
            seed, c = self._find_best_seed(scaled)
            seeds.append(seed)
            coeffs.append(c)
            exps.append(exp)

        coeff_arr = np.stack(coeffs).astype(np.int8)
        compressed = {
            "seeds": np.array(seeds, dtype=np.uint16),
            "coefficients": coeff_arr,
            "shared_exponents": np.array(exps, dtype=np.int8),
            "original_shape": original_shape,
            "block_size": self.C,
            "latent_dim": self.P,
            "pad_length": pad,
        }
        return compressed

    # ------------------------------------------------------------------
    def _find_best_seed(self, block: np.ndarray) -> tuple[int, np.ndarray]:
        best_seed, best_c, best_err = 0, None, float("inf")
        for seed in range(1, self.max_candidates + 1):
            U = self.lfsr.generate_matrix(seed, self.C, self.P)
            c, *_ = np.linalg.lstsq(U, block, rcond=None)
            q = self._quantise(c)
            err = np.linalg.norm(block - U @ q)
            if err < best_err:
                best_seed, best_c, best_err = seed, q, err
        assert best_c is not None
        return best_seed, best_c

    def _quantise(self, coeffs: np.ndarray) -> np.ndarray:
        idx = np.abs(coeffs[:, None] - self.Q[None, :]).argmin(axis=1)
        return self.Q[idx]

    # ------------------------------------------------------------------
    def decompress(self, compressed: dict[str, object]) -> torch.Tensor:
        seeds = compressed["seeds"]
        coeffs = compressed["coefficients"]
        exps = compressed["shared_exponents"]
        blocks = []
        for seed, c, exp in zip(seeds, coeffs, exps, strict=False):
            U = self.lfsr.generate_matrix(int(seed), self.C, self.P)
            block = U @ c
            block = block * (2 ** int(exp))
            blocks.append(block)
        flat = np.concatenate(blocks)
        if compressed["pad_length"]:
            flat = flat[: -compressed["pad_length"]]
        return torch.tensor(flat, dtype=torch.float32).reshape(compressed["original_shape"])


def compress(weights: torch.Tensor) -> dict[str, object]:
    """Convenience wrapper."""
    return SEEDLMCompressor().compress(weights)
