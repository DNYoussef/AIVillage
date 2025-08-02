"""Test that compression works on mobile device constraints"""
import pytest
import torch
import gc
import os
import psutil
from core.compression import SimpleQuantizer


class TestMobileConstraints:
    """Test compression works within mobile device limits"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean up before each test"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_2gb_memory_limit(self):
        """Compression stays under 2GB RAM usage"""
        process = psutil.Process(os.getpid())
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )
        initial_mb = process.memory_info().rss / 1024 / 1024
        quantizer = SimpleQuantizer()
        _ = quantizer.quantize_model(model)
        peak_mb = process.memory_info().rss / 1024 / 1024
        used_mb = peak_mb - initial_mb
        assert used_mb < 2048, f"Used {used_mb:.1f}MB, exceeds 2GB limit"

    def test_batch_compression(self):
        """Can compress multiple models without memory explosion"""
        quantizer = SimpleQuantizer(target_compression=1.0)
        compressed_models = []
        for _ in range(5):
            model = torch.nn.Linear(256, 128)
            compressed_models.append(quantizer.quantize_model(model))
            del model
            gc.collect()
        assert len(compressed_models) == 5
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 2048
