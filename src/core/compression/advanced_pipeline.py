"""Advanced four-stage compression pipeline."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Union

import torch

from agent_forge.compression.bitnet import BITNETCompressor
from agent_forge.compression.seedlm import SEEDLMCompressor
from agent_forge.compression.vptq import VPTQCompressor

try:  # pragma: no cover - optional dependency
    from production.compression.hyper_compression import HyperCompressionEncoder
except ModuleNotFoundError:  # pragma: no cover - fallback
    class HyperCompressionEncoder:  # type: ignore
        """Lightweight fallback encoder when production module unavailable."""

        def encode(self, data: bytes) -> bytes:  # noqa: D401 - simple passthrough
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
    def compress_model(self, model: Union[torch.nn.Module, str, Path]) -> bytes:
        model = self._load_model(model)
        original = self._get_model_size(model)
        total_compressed = 0
        params: Dict[str, Any] = {}

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
            import pickle as _p
            s3_bytes = _p.dumps(s3)
            # Stage 4
            s4_bytes = self.stage4_hyper.encode(s3_bytes)
            params[name] = {
                "data": s4_bytes,
                "shape": tuple(param.shape),
            }
            total_compressed += len(s4_bytes)

        blob = pickle.dumps({"params": params})
        ratio = original / len(blob)
        logger.info("Compression ratio: %.1fx", ratio)
        return blob

    # ------------------------------------------------------------------
    def decompress_model(self, data: bytes) -> Dict[str, torch.Tensor]:
        payload = pickle.loads(data)
        result: Dict[str, torch.Tensor] = {}
        for name, entry in payload["params"].items():
            s3_bytes = self.stage4_hyper.decode(entry["data"])
            s3 = pickle.loads(s3_bytes)
            w = self.stage3_vptq.decompress(s3)
            result[name] = w.reshape(entry["shape"])
        return result

    # ------------------------------------------------------------------
    def _load_model(self, model: Union[torch.nn.Module, str, Path]) -> torch.nn.Module:
        if isinstance(model, torch.nn.Module):
            return model
        return torch.load(Path(model), map_location="cpu")

    def _get_model_size(self, model: torch.nn.Module) -> int:
        return sum(p.numel() * 4 for p in model.parameters())
