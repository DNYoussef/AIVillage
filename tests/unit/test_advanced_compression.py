import os
import sys
from pathlib import Path

import psutil
import pytest
import torch
from torch import nn

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_forge.compression.bitnet import BITNETCompressor
from agent_forge.compression.seedlm import SEEDLMCompressor
from agent_forge.compression.vptq import VPTQCompressor

from core.compression.advanced_pipeline import AdvancedCompressionPipeline


class TestIndividualStages:
    def test_bitnet_compression(self) -> None:
        comp = BITNETCompressor()
        weights = torch.randn(256, 256)
        compressed = comp.compress(weights)
        original_size = weights.numel() * 4
        compressed_size = len(compressed["packed_weights"]) + 4
        ratio = original_size / compressed_size
        assert ratio > 10
        decomp = comp.decompress(compressed)
        assert decomp.shape == weights.shape
        unique_vals = torch.unique(decomp / compressed["scale"])
        assert len(unique_vals) <= 3

    def test_seedlm_compression(self) -> None:
        comp = SEEDLMCompressor(bits_per_weight=4, max_candidates=8)
        weights = torch.randn(64, 64)
        compressed = comp.compress(weights)
        total_bits = (
            len(compressed["seeds"]) * 16
            + compressed["coefficients"].size * 4
            + len(compressed["shared_exponents"]) * 4
        )
        bits_per_weight = total_bits / weights.numel()
        assert bits_per_weight < 5
        decomp = comp.decompress(compressed)
        mse = torch.mean((weights - decomp) ** 2)
        assert mse < 1.0

    def test_vptq_compression(self) -> None:
        comp = VPTQCompressor(bits=2)
        weights = torch.randn(64, 64)
        compressed = comp.compress(weights)
        assert compressed["codebook"].shape[0] == 4
        decomp = comp.decompress(compressed)
        assert decomp.shape == weights.shape


class TestAdvancedPipeline:
    def test_pipeline_initialization(self) -> None:
        pipe = AdvancedCompressionPipeline()
        assert pipe.stage1_bitnet is not None
        assert pipe.stage2_seedlm is not None
        assert pipe.stage3_vptq is not None
        assert pipe.stage4_hyper is not None

    def test_small_model_compression(self) -> None:
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        pipe = AdvancedCompressionPipeline(target_compression=20.0)
        blob = pipe.compress_model(model)
        original = sum(p.numel() * 4 for p in model.parameters())
        ratio = original / len(blob)
        assert ratio > 1

    def test_memory_efficiency(self) -> None:
        proc = psutil.Process(os.getpid())
        start = proc.memory_info().rss / (1024 * 1024)
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        pipe = AdvancedCompressionPipeline()
        pipe.compress_model(model)
        end = proc.memory_info().rss / (1024 * 1024)
        assert (end - start) < 1024

    @pytest.mark.slow
    def test_large_model_compression(self) -> None:
        model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        params = sum(p.numel() for p in model.parameters())
        pipe = AdvancedCompressionPipeline(target_compression=50.0)
        blob = pipe.compress_model(model)
        ratio = params * 4 / len(blob)
        assert ratio > 50

    def test_round_trip_decompression(self) -> None:
        model = nn.Linear(16, 8)
        pipe = AdvancedCompressionPipeline()
        blob = pipe.compress_model(model)
        params = pipe.decompress_model(blob)
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        assert set(params) == names
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert params[name].shape == p.shape

    def test_checksum_validation(self) -> None:
        model = nn.Linear(16, 8)
        pipe = AdvancedCompressionPipeline()
        blob = pipe.compress_model(model)
        tampered = bytearray(blob)
        tampered[-1] ^= 0xFF
        with pytest.raises(ValueError):
            pipe.decompress_model(bytes(tampered))
