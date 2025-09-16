"""
Agent Forge Phase 5: BitNet Training Optimization
=================================================

Specialized training optimization for BitNet models with quantization-aware training,
gradient processing, and memory efficiency optimizations.

Key Features:
- Quantization-aware training (QAT)
- Straight-through estimator optimization
- 1-bit weight gradient computation
- Mixed precision training
- Memory-efficient gradient processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BitNetConfig:
    """Configuration for BitNet training optimization."""
    use_straight_through: bool = True
    gradient_clipping: float = 1.0
    weight_decay_1bit: float = 0.0  # No weight decay on 1-bit weights
    weight_decay_other: float = 0.01
    temperature: float = 1.0
    noise_scale: float = 0.01
    use_gradient_centralization: bool = True
    use_sign_momentum: bool = True
    momentum_beta: float = 0.9


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for BitNet training.

    Allows gradients to flow through non-differentiable quantization operations.
    """

    @staticmethod
    def forward(ctx, input, quantize_func):
        """Forward pass with quantization."""
        return quantize_func(input)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with straight-through gradient."""
        # Gradient flows straight through
        return grad_output, None


def sign_with_straight_through(x: torch.Tensor) -> torch.Tensor:
    """Sign function with straight-through estimator."""
    return StraightThroughEstimator.apply(x, torch.sign)


def quantize_weights_1bit(weights: torch.Tensor, noise_scale: float = 0.0) -> torch.Tensor:
    """
    Quantize weights to 1-bit (-1, +1) with optional noise injection.

    Args:
        weights: Input weight tensor
        noise_scale: Scale for noise injection during training

    Returns:
        Quantized weights
    """
    if noise_scale > 0 and weights.training:
        # Add noise during training for better convergence
        noise = torch.randn_like(weights) * noise_scale
        weights = weights + noise

    # Quantize to {-1, +1}
    return sign_with_straight_through(weights)


def quantize_activations_1bit(activations: torch.Tensor) -> torch.Tensor:
    """Quantize activations to 1-bit."""
    return sign_with_straight_through(activations)


class BitNetLinear(nn.Module):
    """
    BitNet Linear layer with 1-bit weights and optimized training.

    Implements quantization-aware training with straight-through estimators
    and specialized gradient processing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        noise_scale: float = 0.01
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noise_scale = noise_scale

        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize weights
        self._initialize_weights()

        # Track quantized weights for analysis
        self.register_buffer('quantized_weight', torch.zeros_like(self.weight))

    def _initialize_weights(self):
        """Initialize weights with appropriate scaling."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with BitNet quantization."""
        # Quantize weights to 1-bit
        quantized_weight = quantize_weights_1bit(self.weight, self.noise_scale)

        # Store quantized weights for analysis
        if not self.training:
            self.quantized_weight.data.copy_(quantized_weight.data)

        # Linear operation with quantized weights
        output = F.linear(x, quantized_weight, self.bias)

        return output


class BitNetTrainingOptimizer:
    """
    Specialized optimizer for BitNet model training.

    Handles quantization-aware training, gradient processing, and
    memory optimization for 1-bit neural networks.
    """

    def __init__(self, config):
        self.config = config
        self.bitnet_config = BitNetConfig()

        # Setup logging
        self.logger = logging.getLogger('bitnet_optimizer')

        # Gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=config.use_fp16)

        # Track gradient statistics
        self.grad_stats = {
            'norms': [],
            'scales': [],
            'clipping_events': 0,
            'nan_gradients': 0
        }

        # Sign momentum tracking for BitNet layers
        self.sign_momentum = {}

        self.logger.info("BitNet training optimizer initialized")

    def get_parameter_groups(self, model: nn.Module) -> List[Dict]:
        """
        Create parameter groups with different settings for BitNet and other layers.

        Args:
            model: Model to create parameter groups for

        Returns:
            List of parameter groups for optimizer
        """
        bitnet_params = []
        other_params = []
        bias_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Identify BitNet layers
            is_bitnet = any(
                isinstance(module, BitNetLinear)
                for module_name, module in model.named_modules()
                if name.startswith(module_name.replace('.', '_'))
            )

            if 'bias' in name:
                bias_params.append(param)
            elif is_bitnet:
                bitnet_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with different settings
        param_groups = []

        if bitnet_params:
            param_groups.append({
                'params': bitnet_params,
                'weight_decay': self.bitnet_config.weight_decay_1bit,
                'lr': self.config.learning_rate,
                'group_name': 'bitnet_weights'
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'weight_decay': self.bitnet_config.weight_decay_other,
                'lr': self.config.learning_rate,
                'group_name': 'full_precision_weights'
            })

        if bias_params:
            param_groups.append({
                'params': bias_params,
                'weight_decay': 0.0,  # No weight decay on biases
                'lr': self.config.learning_rate * 2,  # Higher learning rate for biases
                'group_name': 'biases'
            })

        self.logger.info(
            f"Parameter groups created: BitNet({len(bitnet_params)}), "
            f"Other({len(other_params)}), Bias({len(bias_params)})"
        )

        return param_groups

    def process_gradients(self, model: nn.Module) -> Dict[str, float]:
        """
        Process gradients for BitNet training with specialized handling.

        Args:
            model: Model with gradients to process

        Returns:
            Dictionary of gradient statistics
        """
        total_norm = 0.0
        num_params = 0
        nan_count = 0
        bitnet_grad_norm = 0.0
        other_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            # Check for NaN gradients
            if torch.isnan(param.grad).any():
                nan_count += 1
                self.logger.warning(f"NaN gradient detected in {name}")
                param.grad.data.zero_()
                continue

            # Calculate parameter-wise gradient norm
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1

            # Track BitNet vs other layer gradients
            is_bitnet = any(
                isinstance(module, BitNetLinear)
                for module_name, module in model.named_modules()
                if name.startswith(module_name.replace('.', '_'))
            )

            if is_bitnet:
                bitnet_grad_norm += param_norm.item() ** 2

                # Apply sign momentum for BitNet weights
                if self.bitnet_config.use_sign_momentum and 'weight' in name:
                    self._apply_sign_momentum(name, param)

            else:
                other_grad_norm += param_norm.item() ** 2

            # Apply gradient centralization
            if self.bitnet_config.use_gradient_centralization and param.grad.dim() > 1:
                param.grad.data = self._centralize_gradient(param.grad.data)

        # Calculate total gradient norm
        total_norm = math.sqrt(total_norm)
        bitnet_grad_norm = math.sqrt(bitnet_grad_norm)
        other_grad_norm = math.sqrt(other_grad_norm)

        # Apply gradient clipping
        clipped = False
        if self.bitnet_config.gradient_clipping > 0:
            if total_norm > self.bitnet_config.gradient_clipping:
                clip_coef = self.bitnet_config.gradient_clipping / (total_norm + 1e-6)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)
                clipped = True
                self.grad_stats['clipping_events'] += 1

        # Update statistics
        self.grad_stats['norms'].append(total_norm)
        self.grad_stats['nan_gradients'] += nan_count

        gradient_stats = {
            'total_norm': total_norm,
            'bitnet_norm': bitnet_grad_norm,
            'other_norm': other_grad_norm,
            'clipped': clipped,
            'nan_count': nan_count,
            'num_params': num_params
        }

        return gradient_stats

    def _apply_sign_momentum(self, param_name: str, param: nn.Parameter) -> None:
        """Apply sign-based momentum for BitNet weights."""
        if param_name not in self.sign_momentum:
            self.sign_momentum[param_name] = torch.zeros_like(param.grad.data)

        # Update sign momentum
        sign_grad = torch.sign(param.grad.data)
        self.sign_momentum[param_name] = (
            self.bitnet_config.momentum_beta * self.sign_momentum[param_name] +
            (1 - self.bitnet_config.momentum_beta) * sign_grad
        )

        # Modify gradient based on sign momentum
        param.grad.data = param.grad.data + 0.1 * self.sign_momentum[param_name]

    def _centralize_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply gradient centralization to improve training stability."""
        if gradient.dim() > 1:
            # Calculate mean over all dimensions except the first (output dimension)
            dims = list(range(1, gradient.dim()))
            mean = gradient.mean(dim=dims, keepdim=True)
            return gradient - mean
        return gradient

    def initialize(self, model: nn.Module) -> None:
        """Initialize BitNet-specific training components."""
        # Count BitNet layers
        bitnet_layers = 0
        total_layers = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                total_layers += 1
                if isinstance(module, BitNetLinear):
                    bitnet_layers += 1

        self.logger.info(f"Model analysis: {bitnet_layers}/{total_layers} layers are BitNet")

        # Initialize sign momentum buffers
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                is_bitnet = any(
                    isinstance(module, BitNetLinear)
                    for module_name, module in model.named_modules()
                    if name.startswith(module_name.replace('.', '_'))
                )
                if is_bitnet:
                    self.sign_momentum[name] = torch.zeros_like(param.data)

    def get_quantization_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get quantization statistics for BitNet layers."""
        stats = {
            'bitnet_layers': 0,
            'total_parameters': 0,
            'bitnet_parameters': 0,
            'compression_ratio': 0.0,
            'weight_distributions': {}
        }

        for name, module in model.named_modules():
            if isinstance(module, BitNetLinear):
                stats['bitnet_layers'] += 1

                # Parameter counts
                layer_params = module.weight.numel()
                if module.bias is not None:
                    layer_params += module.bias.numel()

                stats['bitnet_parameters'] += layer_params

                # Weight distribution analysis
                with torch.no_grad():
                    weights = module.weight.data
                    stats['weight_distributions'][name] = {
                        'mean': weights.mean().item(),
                        'std': weights.std().item(),
                        'min': weights.min().item(),
                        'max': weights.max().item(),
                        'sparsity': (weights.abs() < 1e-6).float().mean().item()
                    }

            elif isinstance(module, nn.Linear):
                # Count other linear layer parameters
                layer_params = module.weight.numel()
                if module.bias is not None:
                    layer_params += module.bias.numel()

        # Calculate compression ratio
        total_params = sum(p.numel() for p in model.parameters())
        stats['total_parameters'] = total_params

        if stats['bitnet_parameters'] > 0:
            # Theoretical compression: 32-bit -> 1-bit for BitNet weights
            theoretical_bitnet_size = stats['bitnet_parameters'] / 32
            other_params_size = (total_params - stats['bitnet_parameters'])
            compressed_size = theoretical_bitnet_size + other_params_size
            stats['compression_ratio'] = total_params / compressed_size

        return stats

    def save_training_state(self, checkpoint_dir: str, epoch: int) -> None:
        """Save BitNet training state."""
        state = {
            'grad_stats': self.grad_stats,
            'sign_momentum': self.sign_momentum,
            'bitnet_config': self.bitnet_config,
            'epoch': epoch
        }

        checkpoint_path = Path(checkpoint_dir) / f'bitnet_state_epoch_{epoch}.pt'
        torch.save(state, checkpoint_path)

        self.logger.info(f"BitNet training state saved: {checkpoint_path}")

    def load_training_state(self, checkpoint_path: str) -> bool:
        """Load BitNet training state."""
        try:
            state = torch.load(checkpoint_path)

            self.grad_stats = state.get('grad_stats', self.grad_stats)
            self.sign_momentum = state.get('sign_momentum', {})
            self.bitnet_config = state.get('bitnet_config', self.bitnet_config)

            self.logger.info(f"BitNet training state loaded from {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load BitNet training state: {e}")
            return False

    def get_training_summary(self) -> Dict[str, Any]:
        """Generate training summary for BitNet optimization."""
        avg_grad_norm = sum(self.grad_stats['norms']) / len(self.grad_stats['norms']) if self.grad_stats['norms'] else 0

        return {
            'optimizer_type': 'BitNet Training Optimizer',
            'gradient_statistics': {
                'average_norm': avg_grad_norm,
                'clipping_events': self.grad_stats['clipping_events'],
                'nan_gradients': self.grad_stats['nan_gradients'],
                'total_steps': len(self.grad_stats['norms'])
            },
            'bitnet_config': {
                'use_straight_through': self.bitnet_config.use_straight_through,
                'gradient_clipping': self.bitnet_config.gradient_clipping,
                'use_sign_momentum': self.bitnet_config.use_sign_momentum,
                'gradient_centralization': self.bitnet_config.use_gradient_centralization
            },
            'sign_momentum_layers': len(self.sign_momentum),
            'mixed_precision': self.config.use_fp16
        }


def convert_model_to_bitnet(model: nn.Module, layers_to_convert: Optional[List[str]] = None) -> nn.Module:
    """
    Convert specified layers in a model to BitNet layers.

    Args:
        model: Model to convert
        layers_to_convert: List of layer names to convert (None for all Linear layers)

    Returns:
        Model with BitNet layers
    """
    if layers_to_convert is None:
        # Convert all Linear layers
        layers_to_convert = [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

    converted_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in layers_to_convert:
            # Create BitNet replacement
            bitnet_layer = BitNetLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            )

            # Copy weights and biases
            with torch.no_grad():
                bitnet_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None and bitnet_layer.bias is not None:
                    bitnet_layer.bias.data.copy_(module.bias.data)

            # Replace the layer
            parent_name = '.'.join(name.split('.')[:-1])
            layer_name = name.split('.')[-1]

            parent_module = model
            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)

            setattr(parent_module, layer_name, bitnet_layer)
            converted_count += 1

    print(f"Converted {converted_count} layers to BitNet")
    return model


if __name__ == "__main__":
    # Example usage and testing
    def test_bitnet_training():
        """Test BitNet training components."""
        from training_config import TrainingConfig

        config = TrainingConfig()

        # Create test model with BitNet layers
        model = nn.Sequential(
            BitNetLinear(128, 64),
            nn.ReLU(),
            BitNetLinear(64, 32),
            nn.ReLU(),
            BitNetLinear(32, 10)
        )

        # Create BitNet optimizer
        optimizer = BitNetTrainingOptimizer(config)
        optimizer.initialize(model)

        # Test parameter groups
        param_groups = optimizer.get_parameter_groups(model)
        print(f"✓ Parameter groups created: {len(param_groups)}")

        # Test quantization stats
        stats = optimizer.get_quantization_stats(model)
        print(f"✓ Quantization stats: {stats['compression_ratio']:.2f}x compression")

        # Test gradient processing
        dummy_loss = torch.tensor(1.0, requires_grad=True)
        dummy_loss.backward()

        grad_stats = optimizer.process_gradients(model)
        print(f"✓ Gradient processing: norm = {grad_stats['total_norm']:.4f}")

        print("BitNet training optimization test completed successfully")

    # Run test
    test_bitnet_training()