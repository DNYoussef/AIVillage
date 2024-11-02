"""
initial_compression.py - First stage compression implementing VPTQ followed by BitNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for VPTQ and BitNet compression"""
    # VPTQ settings - will be adjusted based on model size
    vector_size: int = 8
    codebook_size: int = 256
    group_size: int = 128
    
    # BitNet settings
    lambda_warmup: int = 1000  # Steps for gradual quantization
    lambda_schedule: str = 'linear'  # or 'exponential'
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 5
    
    # Hardware settings
    device: str = 'cuda'
    mixed_precision: bool = True
    num_workers: int = 4
    
    @staticmethod
    def from_model(model: nn.Module) -> 'CompressionConfig':
        """Create config with settings optimized for model size."""
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Adjust settings based on model size
        if total_params > 1e9:  # >1B params
            return CompressionConfig(
                vector_size=32,
                codebook_size=1024,
                group_size=512,
                batch_size=1,
                num_workers=1
            )
        elif total_params > 1e8:  # >100M params
            return CompressionConfig(
                vector_size=16,
                codebook_size=512,
                group_size=256,
                batch_size=2,
                num_workers=2
            )
        else:  # Smaller models
            return CompressionConfig(
                vector_size=8,
                codebook_size=256,
                group_size=128,
                batch_size=4,
                num_workers=4
            )

class TernaryQuantizer(torch.autograd.Function):
    """Convert weights to ternary values (-1, 0, 1) with straight-through estimator"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scale = torch.mean(torch.abs(input)).clamp(min=1e-8)
        return torch.round(torch.clamp(input / scale, -1, 1)).to(torch.int8), scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

def quantize_activations(x):
    """INT8 quantization for activations"""
    scale = 127.0 / torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return torch.round(torch.clamp(x * scale, -127, 127)) / scale

class VPTQLinear(nn.Linear):
    """Linear layer with VPTQ quantization"""
    def __init__(self, in_features, out_features, bias=True, config: Optional[CompressionConfig] = None):
        super().__init__(in_features, out_features, bias)
        self.config = config or CompressionConfig()
        self.register_buffer('centroids', None)
        self.register_buffer('assignments', None)
        self.register_buffer('scales', None)
        
        # Calculate safe vector size
        total_elements = in_features * out_features
        self.vector_size = 1
        while self.vector_size * 2 <= min(total_elements // 100, self.config.vector_size):
            self.vector_size *= 2
        
        logger.info(f"Layer size: {in_features}x{out_features}, Vector size: {self.vector_size}")

    def _reshape_weights(self):
        """Safely reshape weights into vectors."""
        weight_flat = self.weight.view(-1)
        total_elements = weight_flat.size(0)
        
        # Calculate number of complete vectors
        num_complete_vectors = (total_elements // self.vector_size) * self.vector_size
        vectors = weight_flat[:num_complete_vectors].view(-1, self.vector_size)
        remaining = weight_flat[num_complete_vectors:]
        
        return vectors, remaining

    def _init_centroids(self, vectors):
        """Initialize centroids using k-means++."""
        norms = torch.norm(vectors, dim=1, keepdim=True)
        normalized = vectors / (norms + 1e-8)
        
        # Choose initial centroids
        num_centroids = min(self.config.codebook_size, len(vectors))
        centroids = [normalized[torch.randint(0, normalized.size(0), (1,))]]
        
        for _ in range(num_centroids - 1):
            # Calculate distances to existing centroids
            dists = torch.cat([torch.norm(normalized.unsqueeze(1) - c, dim=2).min(1)[0]
                             for c in centroids], dim=0)
            # Choose next centroid with probability proportional to distance squared
            probs = dists ** 2
            probs /= probs.sum()
            next_idx = torch.multinomial(probs, 1)
            centroids.append(normalized[next_idx])
        
        return torch.stack(centroids)

    def quantize(self):
        """Apply VPTQ quantization"""
        try:
            # Reshape weights into vectors
            vectors, remaining = self._reshape_weights()
            
            if len(vectors) == 0:
                # If no complete vectors, just do ternary quantization
                return TernaryQuantizer.apply(self.weight)[0]
            
            if self.centroids is None:
                self.centroids = self._init_centroids(vectors)
            
            # Normalize vectors
            norms = torch.norm(vectors, dim=1, keepdim=True)
            normalized = vectors / (norms + 1e-8)
            
            # Find nearest centroids
            distances = torch.cdist(normalized, self.centroids)
            self.assignments = torch.argmin(distances, dim=1)
            
            # Calculate optimal scales
            assigned = self.centroids[self.assignments]
            self.scales = (vectors * assigned).sum(1) / ((assigned ** 2).sum(1) + 1e-8)
            
            # Reconstruct vectors
            quantized = (self.centroids[self.assignments] * self.scales.unsqueeze(1))
            
            # Handle remaining elements with ternary quantization
            if len(remaining) > 0:
                remaining_quant = TernaryQuantizer.apply(remaining)[0]
                quantized = torch.cat([quantized.view(-1), remaining_quant])
            
            return quantized.view_as(self.weight)
            
        except RuntimeError as e:
            if "out of bounds" in str(e) or "dimension" in str(e):
                # Reduce vector size and try again
                self.vector_size = max(1, self.vector_size // 2)
                logger.warning(f"Reducing vector size to {self.vector_size} and retrying...")
                return self.quantize()
            raise

    def forward(self, x):
        # Use quantized weights for forward pass
        quantized_weight = self.quantize()
        return F.linear(x, quantized_weight, self.bias)

class BitLinear(nn.Linear):
    """Linear layer with ternary weight quantization"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('quantized_weight', torch.zeros_like(self.weight, dtype=torch.int8))
        self.steps = 0

    def quantize_weight(self):
        """Convert weights to ternary values"""
        self.quantized_weight, self.weight_scale = TernaryQuantizer.apply(self.weight)

    def forward(self, x):
        self.steps += 1
        # Layer normalize input
        x_norm = F.layer_norm(x, x.shape[-1:])
        # Quantize activations
        x_quant = x_norm + (quantize_activations(x_norm) - x_norm).detach()
        # Use quantized weights
        w_quant = self.quantized_weight.float() * self.weight_scale
        return F.linear(x_quant, w_quant, self.bias)

class CompressedModel(nn.Module):
    """Model wrapper for compression pipeline"""
    def __init__(self, model: nn.Module, config: Optional[CompressionConfig] = None):
        super().__init__()
        self.config = config or CompressionConfig()
        # First convert to VPTQ
        self.model = self._convert_to_vptq(model)
        self.steps = 0

    def _convert_to_vptq(self, model: nn.Module) -> nn.Module:
        """Convert Linear layers to VPTQLinear"""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                vptq_linear = VPTQLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    self.config
                )
                vptq_linear.weight.data = module.weight.data
                if module.bias is not None:
                    vptq_linear.bias.data = module.bias.data
                setattr(model, name, vptq_linear)
            else:
                self._convert_to_vptq(module)
        return model

    def convert_to_bitnet(self):
        """Convert VPTQLinear layers to BitLinear"""
        for name, module in self.model.named_modules():
            if isinstance(module, VPTQLinear):
                # Get quantized weights from VPTQ
                quantized_weight = module.quantize()
                
                # Create BitLinear layer
                bit_linear = BitLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None
                )
                
                # Initialize with quantized weights
                bit_linear.weight.data = quantized_weight
                if module.bias is not None:
                    bit_linear.bias.data = module.bias.data
                    
                # Initial ternary quantization
                bit_linear.quantize_weight()
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = self.model if parent_name == '' else getattr(self.model, parent_name)
                setattr(parent_module, child_name, bit_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

async def compress_and_train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Optional[CompressionConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Complete compression pipeline"""
    config = config or CompressionConfig()
    
    # Step 1: VPTQ compression
    compressed_model = CompressedModel(model, config)
    
    # Step 2: Train with VPTQ
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting VPTQ training...")
    for epoch in range(config.epochs):
        compressed_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = compressed_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Step 3: Convert to BitNet
    logger.info("Converting to BitNet...")
    compressed_model.convert_to_bitnet()
    
    # Step 4: Fine-tune BitNet
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=config.learning_rate)
    
    logger.info("Starting BitNet fine-tuning...")
    for epoch in range(config.epochs):
        compressed_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = compressed_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Re-quantize weights after each update
            for module in compressed_model.modules():
                if isinstance(module, BitLinear):
                    module.quantize_weight()
    
    return compressed_model, {
        'config': config,
        'final_loss': loss.item()
    }

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create dummy model and data
        model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        train_data = torch.randn(1000, 1024)
        train_labels = torch.randint(0, 256, (1000,))
        train_loader = torch.utils.data.DataLoader(
            list(zip(train_data, train_labels)),
            batch_size=32,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            list(zip(train_data[:100], train_labels[:100])),
            batch_size=32
        )
        
        # Run compression pipeline
        compressed_model, stats = await compress_and_train(
            model, train_loader, val_loader
        )
        
        print("Compression complete!")
        print(f"Final loss: {stats['final_loss']:.4f}")
    
    asyncio.run(main())
