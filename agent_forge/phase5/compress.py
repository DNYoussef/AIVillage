"""Wrapper to create final compressed package."""
from ..compression import stream_compress_model, CompressionConfig
import torch.nn as nn


def final_package(model: nn.Module, out_path: str) -> dict:
    data = stream_compress_model(model, CompressionConfig())
    # Saving handled externally; just return compression stats
    return data
