"""Utility wrappers for BitNet-style linear layers and quantization helpers."""

from .model_compression import BitNetModel, BitLinear, convert_to_bitnet

# quantization helpers located in training modules
try:
    from agent_forge.training.training import quantize_weights
    from agent_forge.training.sleep_and_dream import quantize_activations
except Exception:  # pragma: no cover - optional if training module unavailable
    def quantize_weights(x):
        raise ImportError("quantize_weights not available")
    def quantize_activations(x):
        raise ImportError("quantize_activations not available")

__all__ = [
    "BitNetModel",
    "BitLinear",
    "convert_to_bitnet",
    "quantize_weights",
    "quantize_activations",
]
