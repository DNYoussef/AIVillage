import unittest
import importlib.util
import sys
from pathlib import Path

try:
    torch_spec = importlib.util.find_spec('torch')
except ValueError:
    torch_spec = None
if torch_spec is None:
    raise unittest.SkipTest('PyTorch not installed')

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from agent_forge.compression import SeedLMCompressor, VPTQQuantizer, stream_compress_model
from agent_forge.model_compression.bitlinearization import BitNetModel

class TestCompressionPipeline(unittest.TestCase):
    def test_seedlm_roundtrip(self):
        comp = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=32)
        weights = torch.randn(8,4)
        data = comp.compress_weight_matrix(weights)
        recon = comp.decompress_weight_matrix(data)
        self.assertEqual(recon.shape, weights.shape)
        self.assertLess(torch.mean((weights - recon)**2).item(), 1e-1)

    def test_vptq_roundtrip(self):
        quant = VPTQQuantizer(bits_per_vector=2.0, vector_length=4)
        w = torch.randn(4,4)
        data = quant.quantize_weight_matrix(w)
        recon = quant.dequantize_weight_matrix(data)
        self.assertEqual(recon.shape, w.shape)

    def test_stream_compress_model(self):
        model = torch.nn.Linear(8,4)
        compressed = stream_compress_model(model)
        self.assertIn('weight', compressed)
        self.assertIn('bias', compressed)
        self.assertGreater(compressed['__compression_ratio__'], 1.0)

    def test_bitnet_wrapper(self):
        lin = torch.nn.Linear(4,2)
        bit = BitNetModel(lin)
        out = bit(torch.randn(1,4))
        self.assertEqual(out.shape[-1], 2)

if __name__ == '__main__':
    unittest.main()
