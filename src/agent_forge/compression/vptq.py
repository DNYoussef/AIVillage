"""Stub implementation for vptq compression.
This is a placeholder to fix test infrastructure.
"""

import warnings
from typing import Any

import torch

warnings.warn(
    "vptq is a stub implementation. "
    "Replace with actual implementation before production use.",
    UserWarning,
    stacklevel=2,
)


class VPTQCompressor:
    """Placeholder compressor for testing."""

    def __init__(self):
        self.compression_ratio = 4.0  # Mock compression ratio

    def compress(self, model: Any) -> dict[str, Any]:
        """Stub compression method."""
        return {"compressed": True, "method": "vptq", "ratio": self.compression_ratio}

    def decompress(self, compressed_data: dict[str, Any]) -> Any:
        """Stub decompression method."""
        return torch.nn.Linear(10, 10)  # Return dummy model


# Convenience function for tests
def compress(model: Any) -> dict[str, Any]:
    """Convenience function for compression."""
    compressor = VPTQCompressor()
    return compressor.compress(model)
