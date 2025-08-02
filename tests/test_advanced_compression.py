"""Tests for the AdvancedCompressionPipeline."""
import os
import sys
from pathlib import Path

import psutil
import torch
from torch import nn

# Ensure the ``src`` directory is on the Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.compression.advanced_pipeline import AdvancedCompressionPipeline


class TestAdvancedCompression:
    def test_pipeline_initialization(self) -> None:
        pipeline = AdvancedCompressionPipeline()
        assert pipeline.stage1_quantizer is not None
        assert pipeline.stage2_seedlm is not None
        assert pipeline.stage3_vptq is not None
        assert pipeline.stage4_hyper is not None

    def test_extreme_compression(self) -> None:
        model = nn.Linear(1000, 1000, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        pipeline = AdvancedCompressionPipeline()
        compressed = pipeline.compress_model(model)
        original_size = pipeline._get_model_size(model)
        ratio = original_size / len(compressed)
        print(f"Achieved compression: {ratio:.1f}x")
        assert ratio > 50

    def test_mobile_memory_constraint(self) -> None:
        model = nn.Linear(128, 128)
        pipeline = AdvancedCompressionPipeline()
        proc = psutil.Process(os.getpid())
        start = proc.memory_info().rss / (1024 * 1024)
        pipeline.compress_model(model)
        end = proc.memory_info().rss / (1024 * 1024)
        assert (end - start) < 2048
