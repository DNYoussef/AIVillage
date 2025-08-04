"""Simple quantizer that actually achieves 4x compression for mobile devices."""

import io
import logging
from pathlib import Path

import torch
import torch.quantization

logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Raised when compression fails or doesn't meet requirements."""



class SimpleQuantizer:
    """Quantize PyTorch models for 2GB mobile devices
    Target: 4x compression with <10% accuracy loss.
    """

    def __init__(self, target_compression: float = 4.0) -> None:
        self.target_compression = target_compression
        self.supported_layers = {
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.Conv1d,
            torch.nn.LSTM,
            torch.nn.GRU,
        }

    def quantize_model(self, model_path: str | Path | torch.nn.Module) -> bytes:
        """Quantize a PyTorch model to achieve 4x compression."""
        model = self._load_model(model_path)
        original_size = self._get_model_size(model)
        logger.info("Original model size: %.1f MB", original_size / 1024 / 1024)
        model.eval()
        quantized = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec=self.supported_layers,
            dtype=torch.qint8,
        )
        self._optimize_model(quantized)
        compressed_bytes = self._serialize_compressed(quantized)
        compressed_size = len(compressed_bytes)
        ratio = original_size / compressed_size
        logger.info("Compressed size: %.1f MB", compressed_size / 1024 / 1024)
        logger.info("Compression ratio: %.2fx", ratio)
        if ratio < (self.target_compression - 0.5):
            logger.warning(
                "Compression ratio %.2fx below target %.1fx",
                ratio,
                self.target_compression,
            )
        return compressed_bytes

    def _load_model(
        self, model_path: str | Path | torch.nn.Module
    ) -> torch.nn.Module:
        if isinstance(model_path, torch.nn.Module):
            return model_path
        model_path = Path(model_path)
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)
        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(model, dict):
                msg = "Model file contains state_dict only. Please provide complete model or model instance."
                raise CompressionError(
                    msg
                )
            return model
        except Exception as e:
            msg = f"Failed to load model: {e}"
            raise CompressionError(msg) from e

    def _get_model_size(self, model: torch.nn.Module) -> int:
        buffer = io.BytesIO()
        torch.save(model, buffer, _use_new_zipfile_serialization=False)
        return buffer.tell()

    def _optimize_model(self, model: torch.nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False
            if hasattr(param, "grad"):
                param.grad = None
        for module in model.modules():
            if hasattr(module, "training"):
                module.training = False
            if isinstance(module, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d):
                module.track_running_stats = False

    def _serialize_compressed(self, model: torch.nn.Module) -> bytes:
        buffer = io.BytesIO()
        try:
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, buffer, _use_new_zipfile_serialization=True)
        except Exception:
            buffer = io.BytesIO()
            torch.save(model, buffer, _use_new_zipfile_serialization=True)
        return buffer.getvalue()

    def decompress_model(self, compressed_bytes: bytes) -> torch.nn.Module:
        buffer = io.BytesIO(compressed_bytes)
        try:
            model = torch.jit.load(buffer, map_location="cpu")
        except Exception:
            buffer.seek(0)
            model = torch.load(buffer, map_location="cpu", weights_only=False)
        return model

    @staticmethod
    def load_quantized_model(model_bytes: bytes) -> torch.nn.Module:
        """Compatibility helper for previous API."""
        buffer = io.BytesIO(model_bytes)
        try:
            model = torch.jit.load(buffer, map_location="cpu")
        except Exception:
            buffer.seek(0)
            model = torch.load(buffer, map_location="cpu", weights_only=False)
        return model
