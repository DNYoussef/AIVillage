import hashlib
from typing import List
import numpy as np


class SimHashEmbedder:
    """Deterministic offline embedder using character n-gram simhash."""

    def __init__(self, ngram_size: int = 3, dim: int = 1024) -> None:
        self.ngram_size = ngram_size
        self.dim = dim
        self._bytes = dim // 8  # number of bytes for bit representation

    def _hash_ngram(self, ngram: str) -> bytes:
        """Return 1024-bit hash for the ngram using sha256 blocks."""
        data = ngram.encode("utf-8")
        # concatenate four sha256 digests to reach 1024 bits
        return b"".join(
            hashlib.sha256(data + i.to_bytes(1, "little")).digest()
            for i in range(4)
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a normalized float32 vector."""
        if not text:
            # return zero vector (normalized) if text empty
            return np.zeros(self.dim, dtype=np.float32)

        n = self.ngram_size
        ngrams: List[str] = [
            text[i : i + n] for i in range(max(len(text) - n + 1, 1))
        ]
        accum = np.zeros(self.dim, dtype=np.int32)

        for ngram in ngrams:
            h = self._hash_ngram(ngram)
            for byte_index, byte in enumerate(h):
                for bit in range(8):
                    idx = byte_index * 8 + bit
                    if byte & (1 << bit):
                        accum[idx] += 1
                    else:
                        accum[idx] -= 1

        bits = (accum >= 0).astype(np.float32)
        norm = np.linalg.norm(bits)
        if norm > 0:
            bits /= norm
        return bits
