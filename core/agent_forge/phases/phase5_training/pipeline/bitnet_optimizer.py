"""
Agent Forge Phase 5 - BitNet-Specific Optimizer
1-bit weight optimization with straight-through estimator
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import math
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

class QuantizationMode(Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    LEARNED_THRESHOLD = "learned_threshold"

@dataclass
class BitNetConfig:
    """BitNet optimization configuration"""
    # Quantization parameters
    quantization_mode: QuantizationMode = QuantizationMode.DETERMINISTIC
    weight_bits: int = 1
    activation_bits: int = 8
    gradient_bits: int = 8

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    eps: float = 1e-8

    # BitNet specific
    straight_through_estimator: bool = True
    quantization_warmup_steps: int = 1000
    full_precision_warmup: bool = True

    # Learned quantization
    learnable_threshold: bool = False
    threshold_lr_multiplier: float = 10.0

    # Gradient scaling
    gradient_scale_factor: float = 1.0
    adaptive_gradient_scaling: bool = True

    # Sparsity
    target_sparsity: float = 0.0
    sparsity_warmup_steps: int = 5000

class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for binary quantization"""

    @staticmethod
    def forward(ctx, input_tensor, threshold=0.0, stochastic=False):
        """Forward pass with quantization"""
        ctx.save_for_backward(input_tensor)

        if stochastic:
            # Stochastic quantization
            prob = torch.sigmoid((input_tensor - threshold) * 10.0)
            noise = torch.rand_like(input_tensor)
            quantized = torch.where(noise < prob,
                                  torch.ones_like(input_tensor),
                                  -torch.ones_like(input_tensor))
        else:
            # Deterministic quantization
            quantized = torch.where(input_tensor > threshold,
                                  torch.ones_like(input_tensor),
                                  -torch.ones_like(input_tensor))

        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with straight-through gradient"""
        input_tensor, = ctx.saved_tensors

        # Straight-through: gradient passes through unchanged
        # with optional clipping for stability
        grad_input = grad_output.clone()

        # Clip gradients for weights outside [-1, 1]
        grad_input = torch.where(torch.abs(input_tensor) <= 1.0,
                               grad_input,
                               grad_input * 0.1)  # Reduced gradient for outliers

        return grad_input, None, None

class BitNetQuantizer(nn.Module):
    """BitNet quantization module with learnable parameters"""

    def __init__(self, config: BitNetConfig, shape: torch.Size):
        super().__init__()
        self.config = config
        self.shape = shape

        # Learnable threshold
        if config.learnable_threshold:
            self.threshold = nn.Parameter(torch.zeros(shape))
        else:
            self.register_buffer('threshold', torch.zeros(shape))

        # Scaling factors
        self.register_buffer('scale_factor', torch.ones(shape))

        # Statistics tracking
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Quantize weights/activations"""
        if training:
            # Update running statistics
            with torch.no_grad():
                batch_mean = x.mean(dim=0, keepdim=True)
                batch_var = x.var(dim=0, keepdim=True, unbiased=False)

                momentum = 0.1
                self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
                self.num_batches_tracked += 1

        # Normalize
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.config.eps)

        # Apply quantization
        if self.config.quantization_mode == QuantizationMode.STOCHASTIC:
            quantized = StraightThroughEstimator.apply(normalized, self.threshold, True)
        else:
            quantized = StraightThroughEstimator.apply(normalized, self.threshold, False)

        # Scale back
        return quantized * self.scale_factor

    def extra_repr(self) -> str:
        return f'bits={self.config.weight_bits}, mode={self.config.quantization_mode.value}'

class BitNetLayer(nn.Module):
    """BitNet quantized linear layer"""

    def __init__(self, in_features: int, out_features: int, config: BitNetConfig, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Full precision weights for training
        self.weight_fp = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Quantizers
        self.weight_quantizer = BitNetQuantizer(config, self.weight_fp.shape)

        # Training state
        self.register_buffer('quantization_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('full_precision_mode', torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights"""
        # Check if we should use full precision during warmup
        if (self.config.full_precision_warmup and
            self.quantization_step < self.config.quantization_warmup_steps):
            weight = self.weight_fp
        else:
            # Use quantized weights
            weight = self.weight_quantizer(self.weight_fp, self.training)

        # Standard linear operation
        output = F.linear(x, weight, self.bias)

        # Update step counter during training
        if self.training:
            self.quantization_step += 1

        return output

    def get_quantization_loss(self) -> torch.Tensor:
        """Additional loss term for quantization regularization"""
        # Encourage weights to be close to quantization boundaries
        quantized_weight = self.weight_quantizer(self.weight_fp, False)
        quantization_loss = F.mse_loss(self.weight_fp, quantized_weight.detach())
        return quantization_loss

class BitNetOptimizer(Optimizer):
    """Specialized optimizer for BitNet training"""

    def __init__(
        self,
        params,
        config: BitNetConfig,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.config = config
        self.global_step = 0

        # Gradient scaling
        self.gradient_scaler = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _get_gradient_scale(self, param_name: str, grad: torch.Tensor) -> float:
        """Compute adaptive gradient scaling"""
        if not self.config.adaptive_gradient_scaling:
            return self.config.gradient_scale_factor

        # Track gradient statistics
        if param_name not in self.gradient_scaler:
            self.gradient_scaler[param_name] = {
                'sum_sq': 0.0,
                'count': 0
            }

        stats = self.gradient_scaler[param_name]
        stats['sum_sq'] += grad.norm().item() ** 2
        stats['count'] += 1

        # Compute RMS gradient
        rms_grad = math.sqrt(stats['sum_sq'] / stats['count'])

        # Scale factor based on gradient magnitude
        if rms_grad > 1.0:
            scale_factor = 1.0 / rms_grad
        else:
            scale_factor = 1.0

        return scale_factor * self.config.gradient_scale_factor

    def _apply_sparsity(self, param: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """Apply magnitude-based sparsity"""
        if sparsity_ratio <= 0.0:
            return param

        # Calculate threshold for sparsity
        param_flat = param.view(-1)
        num_params = param_flat.numel()
        num_sparse = int(num_params * sparsity_ratio)

        if num_sparse > 0:
            # Find threshold
            sorted_params, _ = torch.sort(torch.abs(param_flat))
            threshold = sorted_params[num_sparse - 1]

            # Apply sparsity mask
            sparse_mask = torch.abs(param) > threshold
            return param * sparse_mask.float()

        return param

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.dtype in {torch.float16, torch.bfloat16}:
                        grads.append(p.grad.float())
                    else:
                        grads.append(p.grad)

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state['step'] += 1
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']

            # Perform BitNet-specific updates
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                # Apply gradient scaling for BitNet
                param_name = f"param_{id(param)}"
                grad_scale = self._get_gradient_scale(param_name, grad)
                grad = grad * grad_scale

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # BitNet-specific: Adjust learning rate for quantized layers
                if hasattr(param, '_bitnet_quantized'):
                    if self.global_step < self.config.quantization_warmup_steps:
                        # Reduced learning rate during warmup
                        step_size *= 0.1

                # Apply update
                param.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply sparsity if configured
                current_sparsity = self._get_current_sparsity()
                if current_sparsity > 0:
                    param.copy_(self._apply_sparsity(param, current_sparsity))

                # Clamp weights for BitNet layers
                if hasattr(param, '_bitnet_quantized'):
                    # Soft clamp to encourage convergence to quantized values
                    param.data = torch.tanh(param.data)

        return loss

    def _get_current_sparsity(self) -> float:
        """Get current sparsity ratio based on training schedule"""
        if self.global_step < self.config.sparsity_warmup_steps:
            progress = self.global_step / self.config.sparsity_warmup_steps
            return self.config.target_sparsity * progress
        else:
            return self.config.target_sparsity

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about quantization state"""
        info = {
            'global_step': self.global_step,
            'quantization_warmup_complete': self.global_step >= self.config.quantization_warmup_steps,
            'current_sparsity': self._get_current_sparsity(),
            'gradient_scalers': {}
        }

        # Add gradient scaler info
        for param_name, stats in self.gradient_scaler.items():
            if stats['count'] > 0:
                info['gradient_scalers'][param_name] = {
                    'rms_gradient': math.sqrt(stats['sum_sq'] / stats['count']),
                    'update_count': stats['count']
                }

        return info

class BitNetLossFunction:
    """Custom loss function with BitNet-specific regularization"""

    def __init__(self, config: BitNetConfig, base_loss_fn=None):
        self.config = config
        self.base_loss_fn = base_loss_fn or nn.CrossEntropyLoss()

    def __call__(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss with BitNet regularization"""
        # Base loss
        if 'target' in batch:
            predictions = model(batch['input'])
            base_loss = self.base_loss_fn(predictions, batch['target'])
        else:
            # For unsupervised tasks
            predictions = model(batch['input'])
            base_loss = torch.tensor(0.0, device=predictions.device)

        total_loss = base_loss

        # Add quantization regularization
        quantization_loss = 0.0
        quantization_count = 0

        for module in model.modules():
            if isinstance(module, BitNetLayer):
                q_loss = module.get_quantization_loss()
                quantization_loss += q_loss
                quantization_count += 1

        if quantization_count > 0:
            # Scale quantization loss
            quantization_weight = 0.01  # Adjust based on needs
            total_loss = total_loss + quantization_weight * (quantization_loss / quantization_count)

        # Sparsity regularization
        if self.config.target_sparsity > 0:
            sparsity_loss = 0.0
            param_count = 0

            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # L1 regularization to encourage sparsity
                    sparsity_loss += torch.sum(torch.abs(param))
                    param_count += 1

            if param_count > 0:
                sparsity_weight = 1e-5  # Adjust based on needs
                total_loss = total_loss + sparsity_weight * (sparsity_loss / param_count)

        return total_loss

def convert_model_to_bitnet(model: nn.Module, config: BitNetConfig) -> nn.Module:
    """Convert regular model to BitNet by replacing Linear layers"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with BitNet layer
            bitnet_layer = BitNetLayer(
                module.in_features,
                module.out_features,
                config,
                module.bias is not None
            )

            # Copy weights
            with torch.no_grad():
                bitnet_layer.weight_fp.copy_(module.weight)
                if module.bias is not None:
                    bitnet_layer.bias.copy_(module.bias)

            # Mark as quantized for optimizer
            bitnet_layer.weight_fp._bitnet_quantized = True

            setattr(model, name, bitnet_layer)
        else:
            # Recursively convert child modules
            convert_model_to_bitnet(module, config)

    return model

if __name__ == "__main__":
    # Example usage and testing
    import torch.nn as nn

    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )

        def forward(self, x):
            return self.layers(x)

    # Test BitNet conversion
    model = TestModel()
    config = BitNetConfig(
        learning_rate=1e-3,
        quantization_warmup_steps=100,
        target_sparsity=0.1
    )

    # Convert to BitNet
    bitnet_model = convert_model_to_bitnet(model, config)

    # Create optimizer
    optimizer = BitNetOptimizer(bitnet_model.parameters(), config)

    # Create loss function
    loss_fn = BitNetLossFunction(config)

    # Test forward pass
    x = torch.randn(32, 128)
    target = torch.randint(0, 10, (32,))

    # Training step
    optimizer.zero_grad()
    batch = {'input': x, 'target': target}
    loss = loss_fn(bitnet_model, batch)
    loss.backward()
    optimizer.step()

    print(f"Test completed successfully!")
    print(f"Loss: {loss.item():.6f}")
    print(f"Quantization info: {optimizer.get_quantization_info()}")

    # Test quantized layer
    for name, module in bitnet_model.named_modules():
        if isinstance(module, BitNetLayer):
            print(f"BitNet layer {name}: quantization_step = {module.quantization_step}")
            break