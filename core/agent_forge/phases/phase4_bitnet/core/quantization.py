"""
BitNet Quantization Engine for Agent Forge Phase 4
==================================================

Advanced quantization algorithms and utilities for BitNet implementation.
Provides comprehensive weight quantization, activation scaling, and
optimization techniques for neural network compression.

Key Features:
- Multiple quantization strategies (binary, ternary, adaptive)
- Gradient estimation for quantized weights
- Dynamic scaling and calibration
- Memory-efficient quantization operations

Author: BitNet Core Implementation Specialist - Agent Forge Phase 4
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from ..config.bitnet_config import BitNetConfig, QuantizationMode


class QuantizationStrategy(Enum):
    """Different quantization strategies."""
    DETERMINISTIC = "deterministic"  # Standard sign-based quantization
    STOCHASTIC = "stochastic"       # Stochastic rounding
    LEARNED = "learned"             # Learned quantization thresholds
    ADAPTIVE = "adaptive"           # Adaptive based on weight distribution


class BitNetQuantizationEngine:
    """
    Core quantization engine for BitNet layers.

    Provides various quantization algorithms, calibration methods,
    and optimization techniques for neural network compression.
    """

    def __init__(self, config: BitNetConfig):
        self.config = config
        self.quantization_mode = config.layer_config.quantization_mode
        self.device = config.device
        self.dtype = config.dtype

        # Calibration statistics
        self.calibration_stats = {}
        self.is_calibrated = False

        # Quantization thresholds and scales
        self.global_scale = 1.0
        self.layer_scales = {}

    def calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader,
                 num_batches: int = 100) -> Dict[str, Any]:
        """
        Calibrate quantization parameters using representative data.

        Args:
            model: Model to calibrate
            calibration_data: DataLoader with calibration data
            num_batches: Number of batches to use for calibration

        Returns:
            Dictionary with calibration statistics
        """
        model.eval()
        self.calibration_stats = {}

        # Hook to collect activation statistics
        def get_stats_hook(name):
            def hook(module, input, output):
                if name not in self.calibration_stats:
                    self.calibration_stats[name] = {
                        'inputs': [],
                        'outputs': [],
                        'weights': []
                    }

                # Collect input/output statistics
                if isinstance(input, (tuple, list)):
                    input_tensor = input[0]
                else:
                    input_tensor = input

                if isinstance(output, (tuple, list)):
                    output_tensor = output[0]
                else:
                    output_tensor = output

                self.calibration_stats[name]['inputs'].append(
                    input_tensor.detach().cpu()
                )
                self.calibration_stats[name]['outputs'].append(
                    output_tensor.detach().cpu()
                )

                # Collect weight statistics for linear layers
                if hasattr(module, 'weight'):
                    self.calibration_stats[name]['weights'].append(
                        module.weight.detach().cpu()
                    )

            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(get_stats_hook(name))
                hooks.append(hook)

        # Collect statistics
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

                model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute statistics
        self._compute_calibration_statistics()
        self.is_calibrated = True

        return self.calibration_stats

    def _compute_calibration_statistics(self):
        """Compute final calibration statistics from collected data."""
        for layer_name, stats in self.calibration_stats.items():
            # Compute weight statistics
            if stats['weights']:
                weights = torch.cat(stats['weights'], dim=0)
                stats['weight_mean'] = weights.mean().item()
                stats['weight_std'] = weights.std().item()
                stats['weight_abs_mean'] = weights.abs().mean().item()
                stats['weight_abs_max'] = weights.abs().max().item()

            # Compute activation statistics
            if stats['inputs']:
                inputs = torch.cat(stats['inputs'], dim=0)
                stats['input_mean'] = inputs.mean().item()
                stats['input_std'] = inputs.std().item()
                stats['input_abs_mean'] = inputs.abs().mean().item()
                stats['input_abs_max'] = inputs.abs().max().item()

            if stats['outputs']:
                outputs = torch.cat(stats['outputs'], dim=0)
                stats['output_mean'] = outputs.mean().item()
                stats['output_std'] = outputs.std().item()
                stats['output_abs_mean'] = outputs.abs().mean().item()
                stats['output_abs_max'] = outputs.abs().max().item()

            # Clear raw data to save memory
            stats['inputs'] = []
            stats['outputs'] = []
            stats['weights'] = []

    def quantize_tensor(self, tensor: Tensor, mode: Optional[QuantizationMode] = None,
                       scale: Optional[Tensor] = None,
                       strategy: QuantizationStrategy = QuantizationStrategy.DETERMINISTIC) -> Tuple[Tensor, Tensor]:
        """
        Quantize a tensor using specified mode and strategy.

        Args:
            tensor: Input tensor to quantize
            mode: Quantization mode (defaults to config mode)
            scale: Optional pre-computed scale
            strategy: Quantization strategy

        Returns:
            Tuple of (quantized_tensor, scale_factor)
        """
        if mode is None:
            mode = self.quantization_mode

        if strategy == QuantizationStrategy.DETERMINISTIC:
            return self._deterministic_quantize(tensor, mode, scale)
        elif strategy == QuantizationStrategy.STOCHASTIC:
            return self._stochastic_quantize(tensor, mode, scale)
        elif strategy == QuantizationStrategy.LEARNED:
            return self._learned_quantize(tensor, mode, scale)
        elif strategy == QuantizationStrategy.ADAPTIVE:
            return self._adaptive_quantize(tensor, mode, scale)
        else:
            raise ValueError(f"Unsupported quantization strategy: {strategy}")

    def _deterministic_quantize(self, tensor: Tensor, mode: QuantizationMode,
                              scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Standard deterministic quantization."""
        if mode == QuantizationMode.BINARY:
            return self._binary_quantize(tensor, scale)
        elif mode == QuantizationMode.TERNARY:
            return self._ternary_quantize(tensor, scale)
        elif mode == QuantizationMode.ABSMEAN:
            return self._absmean_scale(tensor, scale)
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")

    def _binary_quantize(self, tensor: Tensor, scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Binary quantization to {-1, +1}."""
        if scale is None:
            # Use absolute mean as scale
            scale = tensor.abs().mean().clamp(min=1e-8)

        # Normalize and quantize
        normalized = tensor / scale
        quantized = torch.sign(normalized)

        # Handle zeros (map to +1)
        quantized = torch.where(quantized == 0, torch.ones_like(quantized), quantized)

        return quantized, scale

    def _ternary_quantize(self, tensor: Tensor, scale: Optional[Tensor] = None,
                         threshold_factor: float = 0.7) -> Tuple[Tensor, Tensor]:
        """Ternary quantization to {-1, 0, +1}."""
        if scale is None:
            # Use absolute mean as scale
            scale = tensor.abs().mean().clamp(min=1e-8)

        # Normalize
        normalized = tensor / scale

        # Adaptive threshold based on weight distribution
        if self.is_calibrated:
            # Use calibrated threshold
            threshold = threshold_factor * normalized.abs().mean()
        else:
            # Default threshold
            threshold = threshold_factor

        # Quantize with threshold
        quantized = torch.zeros_like(normalized)
        quantized[normalized > threshold] = 1.0
        quantized[normalized < -threshold] = -1.0

        return quantized, scale

    def _absmean_scale(self, tensor: Tensor, scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Absolute mean scaling without quantization."""
        if scale is None:
            scale = tensor.abs().mean().clamp(min=1e-8)

        scaled = tensor / scale
        return scaled, scale

    def _stochastic_quantize(self, tensor: Tensor, mode: QuantizationMode,
                           scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Stochastic quantization with probability-based rounding."""
        if scale is None:
            scale = tensor.abs().mean().clamp(min=1e-8)

        normalized = tensor / scale

        if mode == QuantizationMode.BINARY:
            # Stochastic rounding for binary
            prob = (normalized + 1) / 2  # Map [-1, 1] to [0, 1]
            prob = torch.clamp(prob, 0, 1)
            random_tensor = torch.rand_like(prob)
            quantized = torch.where(random_tensor < prob,
                                  torch.ones_like(normalized),
                                  -torch.ones_like(normalized))
        else:
            # Stochastic ternary quantization
            abs_normalized = normalized.abs()
            sign = torch.sign(normalized)

            # Probability of being quantized to non-zero
            prob_nonzero = torch.clamp(abs_normalized, 0, 1)
            random_tensor = torch.rand_like(prob_nonzero)

            quantized = torch.where(random_tensor < prob_nonzero, sign, torch.zeros_like(sign))

        return quantized, scale

    def _learned_quantize(self, tensor: Tensor, mode: QuantizationMode,
                         scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Learned quantization with trainable thresholds."""
        # This would require learned parameters - simplified version
        # In practice, this would use learned thresholds stored in the layer
        return self._deterministic_quantize(tensor, mode, scale)

    def _adaptive_quantize(self, tensor: Tensor, mode: QuantizationMode,
                          scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Adaptive quantization based on weight distribution."""
        if scale is None:
            # Adaptive scaling based on weight distribution
            if tensor.numel() > 1000:  # For larger tensors, use percentile-based scaling
                scale = torch.quantile(tensor.abs(), 0.75).clamp(min=1e-8)
            else:
                scale = tensor.abs().mean().clamp(min=1e-8)

        # Adaptive threshold for ternary quantization
        if mode == QuantizationMode.TERNARY:
            normalized = tensor / scale
            # Use standard deviation to determine threshold
            std = normalized.std()
            threshold = 0.5 * std

            quantized = torch.zeros_like(normalized)
            quantized[normalized > threshold] = 1.0
            quantized[normalized < -threshold] = -1.0

            return quantized, scale
        else:
            return self._deterministic_quantize(tensor, mode, scale)

    def compute_quantization_error(self, original: Tensor, quantized: Tensor, scale: Tensor) -> Dict[str, float]:
        """
        Compute quantization error metrics.

        Args:
            original: Original tensor
            quantized: Quantized tensor
            scale: Scale factor used

        Returns:
            Dictionary with error metrics
        """
        # Reconstruct quantized tensor
        reconstructed = quantized * scale

        # Compute various error metrics
        mse = F.mse_loss(reconstructed, original).item()
        mae = F.l1_loss(reconstructed, original).item()

        # Signal-to-noise ratio
        signal_power = (original ** 2).mean().item()
        noise_power = ((original - reconstructed) ** 2).mean().item()
        snr = 10 * math.log10(signal_power / (noise_power + 1e-8))

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            original.flatten(), reconstructed.flatten(), dim=0
        ).item()

        return {
            'mse': mse,
            'mae': mae,
            'snr_db': snr,
            'cosine_similarity': cos_sim,
            'scale_factor': scale.item() if scale.numel() == 1 else scale.mean().item()
        }

    def quantize_model_weights(self, model: nn.Module,
                             layer_filter: Optional[Callable[[str, nn.Module], bool]] = None) -> Dict[str, Any]:
        """
        Quantize weights of all compatible layers in a model.

        Args:
            model: Model to quantize
            layer_filter: Optional function to filter which layers to quantize

        Returns:
            Dictionary with quantization statistics
        """
        quantization_stats = {}

        for name, module in model.named_modules():
            # Default filter: quantize Linear and Conv2d layers
            if layer_filter is None:
                should_quantize = isinstance(module, (nn.Linear, nn.Conv2d))
            else:
                should_quantize = layer_filter(name, module)

            if should_quantize and hasattr(module, 'weight'):
                # Quantize weights
                original_weight = module.weight.data.clone()
                quantized_weight, scale = self.quantize_tensor(module.weight.data)

                # Update module weights
                module.weight.data = quantized_weight * scale

                # Compute statistics
                error_stats = self.compute_quantization_error(
                    original_weight, quantized_weight, scale
                )

                quantization_stats[name] = {
                    'layer_type': type(module).__name__,
                    'weight_shape': list(module.weight.shape),
                    'quantization_mode': self.quantization_mode.value,
                    'scale_factor': scale.item() if scale.numel() == 1 else scale.mean().item(),
                    **error_stats
                }

        return quantization_stats

    def estimate_compression_ratio(self, model: nn.Module) -> Dict[str, float]:
        """
        Estimate compression ratio for a model.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with compression statistics
        """
        total_params = 0
        quantizable_params = 0

        for module in model.modules():
            if hasattr(module, 'weight'):
                param_count = module.weight.numel()
                total_params += param_count

                # Count quantizable parameters
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    quantizable_params += param_count

        # Estimate compression based on quantization mode
        if self.quantization_mode == QuantizationMode.BINARY:
            bits_per_weight = 1
        elif self.quantization_mode == QuantizationMode.TERNARY:
            bits_per_weight = math.log2(3)  # ~1.58 bits
        else:
            bits_per_weight = 32  # Full precision

        # Original: 32 bits per parameter
        original_bits = total_params * 32
        quantized_bits = (total_params - quantizable_params) * 32 + quantizable_params * bits_per_weight

        compression_ratio = original_bits / quantized_bits

        return {
            'total_parameters': total_params,
            'quantizable_parameters': quantizable_params,
            'quantization_coverage': quantizable_params / total_params,
            'bits_per_quantized_weight': bits_per_weight,
            'theoretical_compression_ratio': compression_ratio,
            'memory_reduction_percent': (1 - 1/compression_ratio) * 100
        }

    def validate_quantization_quality(self, original_model: nn.Module, quantized_model: nn.Module,
                                    validation_data: torch.utils.data.DataLoader,
                                    num_batches: int = 50) -> Dict[str, float]:
        """
        Validate quantization quality by comparing model outputs.

        Args:
            original_model: Original full-precision model
            quantized_model: Quantized model
            validation_data: Validation data loader
            num_batches: Number of batches to validate

        Returns:
            Dictionary with quality metrics
        """
        original_model.eval()
        quantized_model.eval()

        total_mse = 0.0
        total_cosine_sim = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_data):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

                # Get outputs from both models
                original_output = original_model(inputs)
                quantized_output = quantized_model(inputs)

                # Ensure outputs have the same shape
                if original_output.shape != quantized_output.shape:
                    warnings.warn(f"Output shape mismatch: {original_output.shape} vs {quantized_output.shape}")
                    continue

                # Compute metrics
                batch_mse = F.mse_loss(quantized_output, original_output).item()

                # Flatten for cosine similarity
                original_flat = original_output.flatten()
                quantized_flat = quantized_output.flatten()
                batch_cosine_sim = F.cosine_similarity(
                    original_flat, quantized_flat, dim=0
                ).item()

                total_mse += batch_mse * inputs.size(0)
                total_cosine_sim += batch_cosine_sim * inputs.size(0)
                total_samples += inputs.size(0)

        # Compute average metrics
        avg_mse = total_mse / total_samples
        avg_cosine_sim = total_cosine_sim / total_samples

        # Compute PSNR (Peak Signal-to-Noise Ratio)
        psnr = 20 * math.log10(1.0 / (math.sqrt(avg_mse) + 1e-8))

        return {
            'average_mse': avg_mse,
            'average_cosine_similarity': avg_cosine_sim,
            'psnr_db': psnr,
            'samples_evaluated': total_samples
        }

    def save_quantization_config(self, path: str):
        """Save quantization configuration and statistics."""
        config_data = {
            'quantization_mode': self.quantization_mode.value,
            'calibration_stats': self.calibration_stats,
            'is_calibrated': self.is_calibrated,
            'global_scale': self.global_scale,
            'layer_scales': self.layer_scales
        }

        torch.save(config_data, path)

    def load_quantization_config(self, path: str):
        """Load quantization configuration and statistics."""
        config_data = torch.load(path, map_location=self.device)

        self.quantization_mode = QuantizationMode(config_data['quantization_mode'])
        self.calibration_stats = config_data['calibration_stats']
        self.is_calibrated = config_data['is_calibrated']
        self.global_scale = config_data['global_scale']
        self.layer_scales = config_data['layer_scales']


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for gradient computation through quantization.

    Forward pass uses quantized values, backward pass uses the gradient
    of the original values for better training dynamics.
    """

    @staticmethod
    def forward(ctx, input_tensor: Tensor, quantize_fn: Callable[[Tensor], Tuple[Tensor, Tensor]]):
        """Forward pass with quantization."""
        quantized, scale = quantize_fn(input_tensor)
        ctx.save_for_backward(input_tensor, scale)
        return quantized * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Backward pass with straight-through gradients."""
        input_tensor, scale = ctx.saved_tensors

        # Straight-through: gradient flows through as if no quantization
        # Scale gradient appropriately
        grad_input = grad_output.clone()

        # Optional: clip gradients for stability
        grad_input = torch.clamp(grad_input, -1.0, 1.0)

        return grad_input, None


def apply_straight_through_quantization(tensor: Tensor, quantization_engine: BitNetQuantizationEngine) -> Tensor:
    """Apply quantization with straight-through estimator for gradients."""
    def quantize_fn(x):
        return quantization_engine.quantize_tensor(x)

    return StraightThroughEstimator.apply(tensor, quantize_fn)