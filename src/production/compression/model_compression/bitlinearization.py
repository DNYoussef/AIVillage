"""Utility wrappers for BitNet-style linear layers and quantization helpers."""

from typing import NoReturn

from agent_forge.compression.stage1_bitnet import convert_to_bitnet

# quantization helpers located in foundation and training modules
try:
    from agent_forge.foundation.bitnet import q_bitnet as quantize_weights
    from agent_forge.training.sleep_and_dream import quantize_activations
except Exception:  # pragma: no cover - optional if training module unavailable

    def quantize_weights(x) -> NoReturn:
        msg = "quantize_weights not available"
        raise ImportError(msg)

    def quantize_activations(x) -> NoReturn:
        msg = "quantize_activations not available"
        raise ImportError(msg)


__all__ = [
    "convert_to_bitnet",
    "quantize_activations",
    "quantize_weights",
]
