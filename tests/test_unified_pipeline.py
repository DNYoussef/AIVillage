import torch

from src.compression import pipeline


def test_simple_roundtrip():
    model = torch.nn.Linear(4, 4)
    result = pipeline.compress(model)
    assert result["method"] == "simple"
    restored = pipeline.decompress(result)
    assert isinstance(restored, torch.nn.Module)
    assert sum(p.numel() for p in restored.parameters()) == sum(p.numel() for p in model.parameters())


def test_advanced_roundtrip():
    model = torch.nn.Linear(4, 4)
    result = pipeline.compress(model, target_compression=100.0)
    assert result["method"] == "advanced"
    restored = pipeline.decompress(result)
    assert isinstance(restored, dict)
    assert restored["weight"].shape == model.weight.shape
