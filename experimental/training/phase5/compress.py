"""Wrapper to create final compressed package."""

from torch import nn

from ..compression import CompressionConfig, stream_compress_model


def final_package(model: nn.Module, out_path: str) -> dict:
    data = stream_compress_model(model, CompressionConfig())
    # Saving handled externally; just return compression stats
    return data
