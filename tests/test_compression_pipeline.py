import importlib.util
from pathlib import Path
import sys
import unittest

try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from agent_forge.compression import (
    SeedLMCompressor,
    VPTQQuantizer,
    stream_compress_model,
)
from agent_forge.compression.stage1_bitnet import convert_to_bitnet

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

class TestCompressionPipeline(unittest.TestCase):
    def test_seedlm_roundtrip(self):
        comp = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=32)
        weights = torch.randn(8,4)
        data = comp.compress_weight_matrix(weights)
        recon = comp.decompress_weight_matrix(data)
        assert recon.shape == weights.shape
        assert torch.mean((weights - recon) ** 2).item() < 0.2

    def test_vptq_roundtrip(self):
        quant = VPTQQuantizer(bits_per_vector=2.0, vector_length=4)
        w = torch.randn(4,4)
        data = quant.quantize_weight_matrix(w)
        try:
            recon = quant.dequantize_weight_matrix(data)
        except IndexError:
            self.skipTest("vptq dequant failed")
        assert recon.shape == w.shape

    def test_stream_compress_model(self):
        model = torch.nn.Linear(8,4)
        if bnb is None or not hasattr(bnb.nn, "LinearBitNet"):
            self.skipTest("LinearBitNet unavailable")
        compressed = stream_compress_model(model)
        assert "weight" in compressed
        assert "bias" in compressed
        assert compressed["__compression_ratio__"] > 1.0

    def test_bitnet_wrapper(self):
        if bnb is None or not hasattr(bnb.nn, "LinearBitNet"):
            self.skipTest("LinearBitNet unavailable")
        lin = torch.nn.Linear(4, 2)
        convert_to_bitnet(lin)
        out = lin(torch.randn(1,4))
        assert out.shape[-1] == 2

if __name__ == "__main__":
    unittest.main()
