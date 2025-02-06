# model_compression.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

########################################
# 1. Ternary Quantizer & Activation Quantization
########################################

class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scale = torch.mean(torch.abs(input)).clamp(min=1e-8)
        quantized = torch.round(torch.clamp(input / scale, -1, 1)).to(torch.int8)
        return quantized, scale

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale):
        input, = ctx.saved_tensors
        grad_input = grad_quantized.clone().float()
        grad_input[input.abs() > 1] = 0
        return grad_input, None

def quantize_activations(x: torch.Tensor) -> torch.Tensor:
    scale = 127.0 / torch.clamp(torch.max(torch.abs(x), dim=-1, keepdim=True)[0], min=1e-5)
    return torch.round(torch.clamp(x * scale, -127, 127)) / scale

########################################
# 2. CompressionConfig
########################################

@dataclass
class CompressionConfig:
    vector_size: int = 8
    codebook_size: int = 256
    group_size: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 5
    device: str = 'cpu'  # Changed from 'cuda'
    mixed_precision: bool = True
    num_workers: int = 4

    @staticmethod
    def from_model(model: nn.Module) -> 'CompressionConfig':
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 1e9:
            return CompressionConfig(vector_size=32, codebook_size=1024, group_size=512, batch_size=1, num_workers=1)
        elif total_params > 1e8:
            return CompressionConfig(vector_size=16, codebook_size=512, group_size=256, batch_size=2, num_workers=2)
        else:
            return CompressionConfig(vector_size=8, codebook_size=256, group_size=128, batch_size=4, num_workers=4)

########################################
# 3. VPTQLinear: Quantized Linear Layer with Vector Partitioning & k-Means++ Initialization
########################################

class VPTQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, config: Optional[CompressionConfig] = None):
        super(VPTQLinear, self).__init__(in_features, out_features, bias)
        self.config = config or CompressionConfig()
        self.register_buffer('centroids', None)
        self.register_buffer('assignments', None)
        self.register_buffer('scales', None)
        self.quantized_cache = None
        self._prev_weight = self.weight.clone()
        total_elements = in_features * out_features
        self.vector_size = 1
        while self.vector_size * 2 <= min(total_elements // 100, self.config.vector_size):
            self.vector_size *= 2

    def _reshape_weights(self):
        weight_flat = self.weight.view(-1)
        total_elements = weight_flat.numel()
        num_complete = (total_elements // self.vector_size) * self.vector_size
        vectors = weight_flat[:num_complete].view(-1, self.vector_size)
        remaining = weight_flat[num_complete:]
        return vectors, remaining

    def _init_centroids(self, vectors):
        norms = torch.norm(vectors, dim=1, keepdim=True) + 1e-8
        normalized = vectors / norms
        num_centroids = min(self.config.codebook_size, normalized.size(0))
        indices = torch.randint(0, normalized.size(0), (1,))
        centroids = normalized[indices].unsqueeze(0)
        for _ in range(num_centroids - 1):
            distances = torch.cdist(normalized, centroids, p=2)
            min_dists, _ = distances.min(dim=1)
            probs = min_dists ** 2
            total = probs.sum()
            if total.item() == 0:
                probs = torch.ones_like(probs) / len(probs)
            else:
                probs /= total
            next_idx = torch.multinomial(probs, 1)
            new_centroid = normalized[next_idx].unsqueeze(0)
            centroids = torch.cat([centroids, new_centroid], dim=0)
        return centroids

    def quantize(self):
        if self.quantized_cache is not None and torch.equal(self.weight, self._prev_weight):
            return self.quantized_cache
        vectors, remaining = self._reshape_weights()
        if vectors.size(0) == 0:
            quantized_full, _ = TernaryQuantizer.apply(self.weight)
            self.quantized_cache = quantized_full.float()
            self._prev_weight = self.weight.clone()
            return self.quantized_cache
        if self.centroids is None or self.centroids.size(0) != min(self.config.codebook_size, vectors.size(0)):
            self.centroids = self._init_centroids(vectors)
        norms = torch.norm(vectors, dim=1, keepdim=True) + 1e-8
        normalized = vectors / norms
        distances = torch.cdist(normalized, self.centroids, p=2)
        self.assignments = torch.argmin(distances, dim=1)
        assigned = self.centroids[self.assignments]
        self.scales = (vectors * assigned).sum(dim=1) / (((assigned) ** 2).sum(dim=1) + 1e-8)
        quantized_vectors = (assigned * self.scales.unsqueeze(1))
        if remaining.numel() > 0:
            remaining_quant, _ = TernaryQuantizer.apply(remaining)
            quantized = torch.cat([quantized_vectors.view(-1), remaining_quant.float()], dim=0)
        else:
            quantized = quantized_vectors.view(-1)
        quantized = quantized.view_as(self.weight)
        self.quantized_cache = quantized
        self._prev_weight = self.weight.clone()
        return quantized

    def forward(self, x):
        q_weight = self.quantize()
        return F.linear(x, q_weight, self.bias)

########################################
# 4. BitLinear: Linear Layer with Integrated Requantization
########################################

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('quantized_weight', torch.zeros_like(self.weight, dtype=torch.int8))
    
    def quantize_weight(self):
        self.quantized_weight, self.weight_scale = TernaryQuantizer.apply(self.weight)
    
    def forward(self, x):
        self.quantize_weight()
        x_norm = F.layer_norm(x, x.shape[-1:])
        x_quant = x_norm + (quantize_activations(x_norm) - x_norm).detach()
        wq = self.quantized_weight.float() * self.weight_scale
        return F.linear(x_quant, wq, self.bias)

########################################
# 5. CompressedModel: Recursive Model Conversion
########################################

class CompressedModel(nn.Module):
    def __init__(self, model: nn.Module, config: Optional[CompressionConfig] = None):
        super().__init__()
        self.config = config or CompressionConfig.from_model(model)
        self.model = self._convert_to_vptq(model)
    
    def _convert_to_vptq(self, module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                vptq_layer = VPTQLinear(child.in_features, child.out_features, bias=(child.bias is not None), config=self.config)
                vptq_layer.weight.data.copy_(child.weight.data.clone())
                if child.bias is not None:
                    vptq_layer.bias.data.copy_(child.bias.data.clone())
                setattr(module, name, vptq_layer)
            else:
                self._convert_to_vptq(child)
        return module

    def convert_to_bitnet(self):
        for name, module in self.model.named_modules():
            if isinstance(module, VPTQLinear):
                q_weight = module.quantize()
                bit_layer = BitLinear(module.in_features, module.out_features, bias=(module.bias is not None))
                bit_layer.weight.data.copy_(q_weight.clone())
                if module.bias is not None:
                    bit_layer.bias.data.copy_(module.bias.data.clone())
                bit_layer.quantize_weight()
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split('.')[-1]
                    setattr(parent, child_name, bit_layer)

    def _get_parent_module(self, module_name: str) -> Optional[nn.Module]:
        parts = module_name.split('.')
        if len(parts) == 1:
            return self.model
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
