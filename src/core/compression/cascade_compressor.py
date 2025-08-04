#!/usr/bin/env python3
"""Experimental cascade compressor for multiplicative gains."""

from __future__ import annotations

from collections import Counter
import lzma
from typing import Any

import torch


class CascadeCompressor:
    """Compress tensors through quantisation, pattern finding and entropy coding."""

    # ------------------------------------------------------------------
    def compress(self, weights: torch.Tensor) -> bytes:
        quantised, meta = self.quantize_cascade(weights)
        pattern, pmeta = self.pattern_compress(quantised)
        if isinstance(pattern, dict):
            entropy_input = self._pack_patterns(pattern)
        else:
            entropy_input = bytes(pattern)
        entropy_data = self.entropy_compress(entropy_input)
        metadata = self.pack_metadata(meta, pmeta)
        return metadata + entropy_data

    # ------------------------------------------------------------------
    def quantize_cascade(
        self, weights: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        levels = 8
        scale = torch.quantile(weights.abs(), 0.99)
        normalised = weights / (scale or 1.0)
        quant = torch.clamp(normalised * (levels // 2), -(levels // 2), levels // 2 - 1)
        quant = quant.round().to(torch.int8)
        return quant, {"scale": float(scale), "levels": levels}

    # ------------------------------------------------------------------
    def pattern_compress(self, data: torch.Tensor) -> tuple[Any, Any]:
        arr = data.flatten().tolist()
        best = None
        best_size = len(arr)
        for length in [2, 4, 8, 16]:
            patterns = [tuple(arr[i : i + length]) for i in range(0, len(arr), length)]
            counts = Counter(patterns)
            unique = list(counts)
            if len(unique) < 256:
                size = len(unique) * length + len(patterns)
                if size < best_size:
                    best = {"len": length, "unique": unique, "patterns": patterns}
                    best_size = size
        if best is None:
            return arr, None
        return best, {"method": "patterns"}

    # ------------------------------------------------------------------
    def entropy_compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=9)

    # ------------------------------------------------------------------
    def pack_metadata(self, *metas: dict[str, Any]) -> bytes:
        out = bytearray()
        for meta in metas:
            if not meta:
                continue
            for key, value in meta.items():
                key_b = key.encode("utf-8")
                out.append(len(key_b))
                out.extend(key_b)
                if isinstance(value, str):
                    val_b = value.encode("utf-8")
                else:
                    val_b = str(value).encode("utf-8")
                out.extend(val_b)
                out.append(0)
        out.append(0)
        return bytes(out)

    # ------------------------------------------------------------------
    def _pack_patterns(self, data: dict[str, Any]) -> bytes:
        uniq = data["unique"]
        patterns = data["patterns"]
        length = data["len"]
        mapping = {u: i for i, u in enumerate(uniq)}
        indices = bytes(mapping[p] for p in patterns)
        out = bytearray()
        out.append(len(uniq))
        out.append(length)
        for u in uniq:
            out.extend(bytes(int(x) & 0xFF for x in u))
        out.extend(indices)
        return bytes(out)
