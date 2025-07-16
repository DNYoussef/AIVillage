import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    pytest.skip("torch not installed", allow_module_level=True)
import torch

from agent_forge.compression import CompressionConfig, stream_compress_model
from twin_runtime.compressed_loader import CompressedModelLoader


def test_loader_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    compressed = stream_compress_model(model, CompressionConfig(bitnet_finetune=False, use_hyper=False))
    file = tmp_path / "cmp.pth"
    torch.save(compressed, file)

    loader = CompressedModelLoader(lambda: torch.nn.Linear(4, 2), str(file))
    out_model = loader.assemble_model()
    assert isinstance(out_model, torch.nn.Module)
    assert out_model.weight.shape == model.weight.shape

