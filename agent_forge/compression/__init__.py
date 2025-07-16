from dataclasses import dataclass
from typing import Dict, Optional

import bitsandbytes as bnb
import torch
from torch import nn

from .hyperfn import HyperCompressionEncoder
from .seedlm import SeedLMCompressor
from .stage1_bitnet import convert_to_bitnet
from .vptq import VPTQQuantizer


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
        self.seedlm = SeedLMCompressor(
            config.seed_block_size, config.seed_latent_dim, config.seed_num_candidates
        )
        self.vptq = VPTQQuantizer(config.vptq_bits, config.vptq_vector_length)
        self.hyper = (
            HyperCompressionEncoder(config.hyper_clusters) if config.use_hyper else None
        )

    def compress_layer(self, weight: torch.Tensor) -> dict:
        seed_data = self.seedlm.compress_weight_matrix(weight)
        decompressed = self.seedlm.decompress_weight_matrix(seed_data)
        vptq_data = self.vptq.quantize_weight_matrix(decompressed)
        result = {"seedlm": seed_data, "vptq": vptq_data}
        if self.hyper:
            hyper_data = self.hyper.compress_weight_matrix(decompressed)
            result["hyper"] = hyper_data
        return result

    def decompress_layer(self, data: dict) -> torch.Tensor:
        if "hyper" in data:
            return self.hyper.decompress_weight_matrix(data["hyper"])
        if "vptq" in data:
            return self.vptq.dequantize_weight_matrix(data["vptq"])
        return self.seedlm.decompress_weight_matrix(data["seedlm"])


import torch


def run_stage1(input_path: str, output_path: str, config_path: str = None) -> dict:
    """Wrapper function for Stage-1 compression pipeline

    Args:
        input_path: Path to input model checkpoint
        output_path: Path to save compressed model (.stage1.pt)
        config_path: Optional path to configuration file

    Returns:
        Dictionary with compression results and metrics
    """
    import json

    from .stage1 import run_stage1_compression
    from .stage1_config import DEFAULT_STAGE1_CONFIG, Stage1Config

    # Load configuration
    if config_path:
        with open(config_path) as f:
            config_dict = json.load(f)
        config = Stage1Config(**config_dict)
    else:
        config = DEFAULT_STAGE1_CONFIG

    # Run the compression pipeline
    return run_stage1_compression(input_path, output_path, config)


def stream_compress_model(
    model: nn.Module, config: CompressionConfig | None = None
) -> dict[str, dict]:
    cfg = config or CompressionConfig()
    compressor = TwoStageCompressor(cfg)
    handled = set()
    if cfg.bitnet_finetune:
        try:
            model = convert_to_bitnet(model, threshold=cfg.bitnet_zero_threshold)
        except ImportError as exc:
            print(f"BitNet conversion unavailable: {exc}")

    compressed = {}
    have_bitnet = hasattr(bnb.nn, "LinearBitNet")
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
    compressed["__compression_ratio__"] = original_size / compressed_size
    return compressed
