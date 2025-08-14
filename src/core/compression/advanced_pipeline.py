"""Advanced four-stage compression pipeline."""

from __future__ import annotations

import json
import logging
import lzma
import struct
import zlib
from pathlib import Path
from typing import Any

import torch

try:  # pragma: no cover - allow running without installation
    from src.agent_forge.compression.bitnet import BITNETCompressor
    from src.agent_forge.compression.seedlm import SEEDLMCompressor
    from src.agent_forge.compression.vptq import VPTQCompressor
except ModuleNotFoundError:  # pragma: no cover - tests run from repo root
    from src.agent_forge.compression.bitnet import BITNETCompressor
    from src.agent_forge.compression.seedlm import SEEDLMCompressor
    from src.agent_forge.compression.vptq import VPTQCompressor

try:  # pragma: no cover - optional dependency
    from src.production.compression.hyper_compression import HyperCompressionEncoder
except ModuleNotFoundError:  # pragma: no cover - fallback

    class HyperCompressionEncoder:  # type: ignore
        """Lightweight fallback encoder when production module unavailable."""

        def encode(self, data: bytes) -> bytes:
            return data

        def decode(self, data: bytes) -> bytes:  # type: ignore[override]
            return data


logger = logging.getLogger(__name__)


class AdvancedCompressionPipeline:
    """Compose BitNet, SeedLM, VPTQ and HyperCompression."""

    def __init__(self, target_compression: float = 100.0) -> None:
        self.target_compression = target_compression
        self.stage1_bitnet = BITNETCompressor()
        self.stage2_seedlm = SEEDLMCompressor(bits_per_weight=4)
        self.stage3_vptq = VPTQCompressor(bits=2)
        self.stage4_hyper = HyperCompressionEncoder()

    # ------------------------------------------------------------------
    def compress_model(self, model: torch.nn.Module | str | Path) -> bytes:
        model = self._load_model(model)
        original = self._get_model_size(model)
        params: dict[str, tuple[tuple[int, ...], bytes]] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            data = param.data.cpu()
            # Stage 1
            s1 = self.stage1_bitnet.compress(data)
            s1_w = self.stage1_bitnet.decompress(s1)
            # Stage 2
            s2 = self.stage2_seedlm.compress(s1_w)
            s2_w = self.stage2_seedlm.decompress(s2)
            # Stage 3
            s3 = self.stage3_vptq.compress(s2_w)
            s3_bytes = self._serialize_stage3(s3)
            # Stage 4 with effectiveness check
            s4_bytes = self.stage4_hyper.encode(s3_bytes)
            if len(s4_bytes) >= len(s3_bytes) * 0.9:
                logger.warning("HyperCompression ineffective for %s, skipping", name)
                s4_bytes = s3_bytes

            params[name] = (tuple(param.shape), s4_bytes)

        blob = self.pack_compressed_data(params)
        ratio = original / len(blob)
        logger.info("Compression ratio: %.1fx", ratio)
        return blob

    # ------------------------------------------------------------------
    def decompress_model(self, data: bytes) -> dict[str, torch.Tensor]:
        payload = lzma.decompress(data)
        if len(payload) < 8:
            raise ValueError("Corrupted payload: header missing")
        (length, checksum) = struct.unpack_from("II", payload, 0)
        raw = payload[8:]
        if len(raw) != length:
            raise ValueError("Length mismatch in compressed data")
        if zlib.adler32(raw) & 0xFFFFFFFF != checksum:
            raise ValueError("Checksum mismatch in compressed data")
        view = memoryview(raw)
        offset = 0
        params: dict[str, tuple[tuple[int, ...], bytes]] = {}
        count = view[offset]
        offset += 1
        for _ in range(count):
            name_len = view[offset]
            offset += 1
            name = bytes(view[offset : offset + name_len]).decode("utf-8")
            offset += name_len
            dims = view[offset]
            offset += 1
            shape = []
            for _ in range(dims):
                (dim,) = struct.unpack_from("I", view, offset)
                offset += 4
                shape.append(dim)
            (length,) = struct.unpack_from("I", view, offset)
            offset += 4
            s4_bytes = bytes(view[offset : offset + length])
            offset += length
            params[name] = (tuple(shape), s4_bytes)

        if offset != len(view):
            raise ValueError("Trailing data after parameter payload")

        result: dict[str, torch.Tensor] = {}
        for name, (shape, s4_bytes) in params.items():
            try:
                s3_bytes = self.stage4_hyper.decode(s4_bytes)
            except Exception:
                s3_bytes = s4_bytes
            s3 = self._deserialize_stage3(s3_bytes)
            w = self.stage3_vptq.decompress(s3).reshape(shape)
            result[name] = w
        return result

    # ------------------------------------------------------------------
    def pack_compressed_data(self, params: dict[str, tuple[tuple[int, ...], bytes]]) -> bytes:
        """Pack parameter data with minimal overhead and lzma compression."""
        blob = bytearray()
        blob.append(len(params))
        for name, (shape, data) in params.items():
            name_b = name.encode("utf-8")
            blob.append(len(name_b))
            blob.extend(name_b)
            blob.append(len(shape))
            for dim in shape:
                blob.extend(struct.pack("I", dim))
            blob.extend(struct.pack("I", len(data)))
            blob.extend(data)
        raw = bytes(blob)
        checksum = zlib.adler32(raw) & 0xFFFFFFFF
        header = struct.pack("II", len(raw), checksum)
        return lzma.compress(header + raw, preset=9)

    # ------------------------------------------------------------------
    def _serialize_stage3(self, data: dict[str, object]) -> bytes:
        """Safely serialise stage3 output using JSON."""

        def _tensor_to_dict(t: torch.Tensor) -> dict[str, object]:
            return {"dtype": str(t.dtype).split(".")[-1], "data": t.tolist()}

        safe: dict[str, object] = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                safe[k] = _tensor_to_dict(v)
            elif isinstance(v, tuple):
                safe[k] = list(v)
            else:
                safe[k] = v
        return json.dumps(safe).encode("utf-8")

    def _deserialize_stage3(self, data: bytes) -> dict[str, object]:
        """Reverse of :meth:`_serialize_stage3`."""

        def _tensor_from_dict(d: dict[str, Any]) -> torch.Tensor:
            dtype = getattr(torch, str(d["dtype"]))
            return torch.tensor(d["data"], dtype=dtype)

        raw = json.loads(data.decode("utf-8"))
        result: dict[str, object] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "dtype" in v:
                result[k] = _tensor_from_dict(v)
            elif k == "original_shape":
                result[k] = tuple(v)
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------
    def _load_model(self, model: torch.nn.Module | str | Path) -> torch.nn.Module:
        if isinstance(model, torch.nn.Module):
            return model
        return torch.load(Path(model), map_location="cpu")

    def _get_model_size(self, model: torch.nn.Module) -> int:
        return sum(p.numel() * 4 for p in model.parameters())
