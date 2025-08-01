"""Stub implementation for bitnet compression.
This is a placeholder to fix test infrastructure.
"""

from typing import Any
import warnings

import torch

warnings.warn(
    "bitnet is a stub implementation. "
    "Replace with actual implementation before production use.",
    UserWarning,
    stacklevel=2,
)


class BITNETCompressor:
    """Placeholder compressor for testing."""

    def __init__(self):
        self.compression_ratio = 4.0  # Mock compression ratio

    def compress(self, model: Any) -> dict[str, Any]:
        """Stub compression method."""
        return {"compressed": True, "method": "bitnet", "ratio": self.compression_ratio}

    def decompress(self, compressed_data: dict[str, Any]) -> Any:
        """Stub decompression method."""
        return torch.nn.Linear(10, 10)  # Return dummy model


# Convenience function for tests
def compress(model: Any) -> dict[str, Any]:
    """Convenience function for compression."""
    compressor = BITNETCompressor()
    return compressor.compress(model)
