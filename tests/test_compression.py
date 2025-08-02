"""Test compression actually works with real models"""
import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from core.compression import SimpleQuantizer, CompressionError


class SimpleModel(nn.Module):
    """Test model that's realistic but small"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LargerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class TestSimpleQuantizer:
    """Test real compression functionality"""

    def test_quantizer_creation(self):
        """Can create quantizer with custom target"""
        quantizer = SimpleQuantizer(target_compression=3.5)
        assert quantizer.target_compression == 3.5

    def test_compress_simple_model(self):
        """Test compression of a simple model"""
        model = SimpleModel()
        quantizer = SimpleQuantizer()
        compressed = quantizer.quantize_model(model)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        original_size = self._get_model_size(model)
        compressed_size = len(compressed)
        ratio = original_size / compressed_size
        assert ratio >= 3.5, f"Compression ratio {ratio:.2f}x too low"

    def test_compress_from_file(self):
        """Test compression from saved model file"""
        model = SimpleModel()
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(model, tmp.name)
            model_path = tmp.name
        try:
            quantizer = SimpleQuantizer()
            compressed = quantizer.quantize_model(model_path)
            original_size = os.path.getsize(model_path)
            ratio = original_size / len(compressed)
            assert ratio >= 3.5
        finally:
            os.unlink(model_path)

    def test_decompress_model(self):
        """Test model works after compression/decompression"""
        model = SimpleModel()
        test_input = torch.randn(1, 784)
        with torch.no_grad():
            original_output = model(test_input)
        quantizer = SimpleQuantizer()
        compressed = quantizer.quantize_model(model)
        decompressed = quantizer.decompress_model(compressed)
        with torch.no_grad():
            new_output = decompressed(test_input)
        assert new_output.shape == original_output.shape
        diff = torch.abs(original_output - new_output).mean().item()
        relative_diff = diff / torch.abs(original_output).mean().item()
        assert relative_diff < 0.1, f"Accuracy loss {relative_diff:.1%} too high"

    def test_memory_constraint(self):
        """Test compression works within 2GB memory limit"""
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        model = SimpleModel()
        quantizer = SimpleQuantizer()
        _ = quantizer.quantize_model(model)
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_used = peak_memory - initial_memory
        assert memory_used < 2048, f"Used {memory_used:.1f}MB, exceeds 2GB limit"

    def test_large_model_compression(self):
        """Test compression of larger model"""
        model = LargerModel()
        quantizer = SimpleQuantizer()
        compressed = quantizer.quantize_model(model)
        original_size = self._get_model_size(model)
        ratio = original_size / len(compressed)
        assert ratio >= 3.5, f"Large model compression {ratio:.2f}x too low"

    def test_compression_failure(self):
        """Test handling of models that can't compress enough"""
        model = TinyModel()
        quantizer = SimpleQuantizer(target_compression=10.0)
        with pytest.raises(CompressionError):
            quantizer.quantize_model(model)

    def _get_model_size(self, model):
        import io
        buffer = io.BytesIO()
        torch.save(model, buffer)
        return buffer.tell()
