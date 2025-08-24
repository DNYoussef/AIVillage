"""Wrapper to create final compressed package."""

from AIVillage.experimental.training.compression import CompressionConfig, stream_compress_model
from torch import nn


def final_package(model: nn.Module, out_path: str) -> dict:
    data = stream_compress_model(model, CompressionConfig())
    # Saving handled externally; just return compression stats
    return data
