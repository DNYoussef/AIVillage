from .seedlm import SeedLMCompressor
from .vptq import VPTQQuantizer
from .hyperfn import HyperCompressionEncoder
from .stage1_bitnet import convert_to_bitnet

from dataclasses import dataclass
import torch
import torch.nn as nn
import bitsandbytes as bnb
from typing import Optional, Dict

@dataclass
class CompressionConfig:
    bitnet_finetune: bool = True
    bitnet_zero_threshold: float = 0.02
    bitnet_batch_size: int = 2
    bitnet_finetuning_epochs: int = 1
    bitnet_learning_rate: float = 1e-4
    seed_block_size: int = 8
    seed_latent_dim: int = 4
    seed_num_candidates: int = 256
    vptq_bits: float = 2.0
    vptq_vector_length: int = 32
    use_hyper: bool = True
    hyper_clusters: int = 16

class TwoStageCompressor:
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.seedlm = SeedLMCompressor(config.seed_block_size, config.seed_latent_dim, config.seed_num_candidates)
        self.vptq = VPTQQuantizer(config.vptq_bits, config.vptq_vector_length)
        self.hyper = HyperCompressionEncoder(config.hyper_clusters) if config.use_hyper else None

    def compress_layer(self, weight: torch.Tensor) -> Dict:
        seed_data = self.seedlm.compress_weight_matrix(weight)
        decompressed = self.seedlm.decompress_weight_matrix(seed_data)
        vptq_data = self.vptq.quantize_weight_matrix(decompressed)
        result={'seedlm':seed_data,'vptq':vptq_data}
        if self.hyper:
            hyper_data = self.hyper.compress_weight_matrix(decompressed)
            result['hyper']=hyper_data
        return result

    def decompress_layer(self, data: Dict) -> torch.Tensor:
        if 'hyper' in data:
            return self.hyper.decompress_weight_matrix(data['hyper'])
        if 'vptq' in data:
            return self.vptq.dequantize_weight_matrix(data['vptq'])
        return self.seedlm.decompress_weight_matrix(data['seedlm'])

import torch

def stream_compress_model(model: nn.Module, config: Optional[CompressionConfig]=None) -> Dict[str, Dict]:
    cfg = config or CompressionConfig()
    compressor = TwoStageCompressor(cfg)
    handled = set()
    if cfg.bitnet_finetune:
        try:
            model = convert_to_bitnet(model, threshold=cfg.bitnet_zero_threshold)
        except ImportError as exc:
            print(f"BitNet conversion unavailable: {exc}")

    compressed = {}
    have_bitnet = hasattr(bnb.nn, 'LinearBitNet')
    for mod_name, module in model.named_modules():
        if have_bitnet and isinstance(module, bnb.nn.LinearBitNet):
            w_name = f"{mod_name}.weight" if mod_name else "weight"
            compressed[w_name] = compressor.compress_layer(module.to_float())
            handled.add(w_name)
            if module.bias is not None:
                b_name = f"{mod_name}.bias" if mod_name else "bias"
                compressed[b_name] = module.bias.data
                handled.add(b_name)

    for name, param in model.named_parameters():
        if name in handled:
            continue
        if param.dim() >= 2:
            compressed[name] = compressor.compress_layer(param.data)
        else:
            compressed[name] = param.data
    # estimate compression ratio using serialization sizes
    import io
    buf_o = io.BytesIO()
    torch.save(model.state_dict(), buf_o)
    original_size = buf_o.tell()

    buf_c = io.BytesIO()
    torch.save(compressed, buf_c)
    compressed_size = buf_c.tell() or 1
    compressed['__compression_ratio__'] = original_size / compressed_size
    return compressed
