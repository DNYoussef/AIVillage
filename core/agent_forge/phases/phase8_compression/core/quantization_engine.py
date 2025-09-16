"""
Model Quantization Engine
Implements state-of-the-art quantization techniques for model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as torch_quant
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class QuantizationMode(Enum):
    """Quantization modes."""
    PTQ = "post_training_quantization"
    QAT = "quantization_aware_training"
    DYNAMIC = "dynamic_quantization"
    STATIC = "static_quantization"

@dataclass
class QuantizationConfig:
    """Configuration for quantization algorithms."""
    mode: QuantizationMode = QuantizationMode.PTQ
    weight_bits: int = 8
    activation_bits: int = 8
    backend: str = "fbgemm"  # fbgemm, qnnpack, x86
    calibration_batches: int = 100
    symmetric: bool = False
    per_channel: bool = True
    reduce_range: bool = False

class BaseQuantizer(ABC):
    """Base class for all quantization algorithms."""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = []
        self.observers = {}

    @abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization."""
        pass

    @abstractmethod
    def calibrate(self, model: nn.Module,
                 dataloader: torch.utils.data.DataLoader) -> None:
        """Calibrate quantization parameters."""
        pass

    @abstractmethod
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model."""
        pass

    def calculate_compression_stats(self, original_model: nn.Module,
                                  quantized_model: nn.Module) -> Dict[str, float]:
        """Calculate compression statistics."""
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

        compression_ratio = original_size / quantized_size
        size_reduction = (original_size - quantized_size) / original_size

        return {
            'original_size_mb': original_size / 1024 / 1024,
            'quantized_size_mb': quantized_size / 1024 / 1024,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction * 100
        }

class PostTrainingQuantizer(BaseQuantizer):
    """Post-Training Quantization (PTQ) implementation."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for post-training quantization."""
        model.eval()

        # Set quantization backend
        torch.backends.quantized.engine = self.config.backend

        # Configure quantization
        if self.config.backend == 'fbgemm':
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        elif self.config.backend == 'qnnpack':
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        else:
            # Custom qconfig
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric if self.config.per_channel
                    else torch.per_tensor_symmetric
                )
            )

        # Prepare model
        prepared_model = torch.quantization.prepare(model, inplace=False)
        return prepared_model

    def calibrate(self, model: nn.Module,
                 dataloader: torch.utils.data.DataLoader) -> None:
        """Calibrate model with representative data."""
        model.eval()

        batch_count = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                if batch_count >= self.config.calibration_batches:
                    break

                _ = model(inputs)
                batch_count += 1

        logger.info(f"Calibration completed with {batch_count} batches")

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Convert calibrated model to quantized model."""
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model

class QuantizationAwareTrainer(BaseQuantizer):
    """Quantization-Aware Training (QAT) implementation."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization-aware training."""
        model.train()

        # Set quantization backend
        torch.backends.quantized.engine = self.config.backend

        # Configure for QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.config.backend)

        # Prepare model for QAT
        prepared_model = torch.quantization.prepare_qat(model, inplace=False)
        return prepared_model

    def calibrate(self, model: nn.Module,
                 dataloader: torch.utils.data.DataLoader) -> None:
        """QAT doesn't need separate calibration."""
        logger.info("QAT mode: No separate calibration needed")

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model."""
        model.eval()
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model

    def train_qat(self, model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  criterion: nn.Module,
                  epochs: int = 5) -> nn.Module:
        """Train model with quantization awareness."""
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

            avg_loss = running_loss / len(dataloader)
            logger.info(f'Epoch {epoch} completed, Average Loss: {avg_loss:.6f}')

        return model

class DynamicQuantizer(BaseQuantizer):
    """Dynamic quantization for runtime quantization."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Dynamic quantization doesn't need preparation."""
        return model

    def calibrate(self, model: nn.Module,
                 dataloader: torch.utils.data.DataLoader) -> None:
        """Dynamic quantization doesn't need calibration."""
        logger.info("Dynamic quantization: No calibration needed")

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        # Specify layers to quantize
        layers_to_quantize = [nn.Linear, nn.LSTM, nn.GRU]

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=torch.qint8
        )

        return quantized_model

class MixedPrecisionOptimizer:
    """Mixed precision optimization using different bit-widths."""

    def __init__(self, sensitivity_threshold: float = 0.1):
        self.sensitivity_threshold = sensitivity_threshold
        self.layer_sensitivities = {}
        self.bit_allocation = {}

    def analyze_sensitivity(self, model: nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          criterion: nn.Module) -> Dict[str, float]:
        """Analyze layer sensitivity to quantization."""
        model.eval()
        sensitivities = {}

        # Get baseline accuracy
        baseline_acc = self._evaluate_model(model, dataloader, criterion)

        # Test each layer individually
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Simulate quantization noise
                original_weight = module.weight.data.clone()

                # Add quantization noise (simplified)
                noise_scale = original_weight.std() * 0.1
                module.weight.data.add_(torch.randn_like(original_weight) * noise_scale)

                # Evaluate with noise
                noisy_acc = self._evaluate_model(model, dataloader, criterion)
                sensitivity = baseline_acc - noisy_acc

                sensitivities[name] = sensitivity

                # Restore original weights
                module.weight.data.copy_(original_weight)

        self.layer_sensitivities = sensitivities
        return sensitivities

    def allocate_bits(self, sensitivity_analysis: Dict[str, float],
                     target_bits: float = 6.0) -> Dict[str, int]:
        """Allocate bit-widths based on sensitivity analysis."""
        sorted_layers = sorted(sensitivity_analysis.items(),
                             key=lambda x: x[1], reverse=True)

        allocation = {}
        for name, sensitivity in sorted_layers:
            if sensitivity > self.sensitivity_threshold:
                # High sensitivity - use more bits
                allocation[name] = 8
            elif sensitivity > self.sensitivity_threshold / 2:
                # Medium sensitivity
                allocation[name] = 6
            else:
                # Low sensitivity - use fewer bits
                allocation[name] = 4

        self.bit_allocation = allocation
        return allocation

    def _evaluate_model(self, model: nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       criterion: nn.Module) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total

class CustomQuantizationScheme:
    """Custom quantization schemes for specialized use cases."""

    @staticmethod
    def int4_quantization(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """4-bit integer quantization."""
        # Scale and zero-point calculation
        min_val = tensor.min()
        max_val = tensor.max()

        # 4-bit range: -8 to 7
        scale = (max_val - min_val) / 15.0
        zero_point = torch.round(-min_val / scale) - 8
        zero_point = torch.clamp(zero_point, -8, 7)

        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, -8, 7)

        return quantized.to(torch.int8), scale

    @staticmethod
    def binary_quantization(tensor: torch.Tensor) -> torch.Tensor:
        """Binary quantization (+1/-1)."""
        return torch.sign(tensor)

    @staticmethod
    def ternary_quantization(tensor: torch.Tensor,
                           threshold_ratio: float = 0.7) -> torch.Tensor:
        """Ternary quantization (+1/0/-1)."""
        threshold = threshold_ratio * tensor.abs().max()

        quantized = torch.zeros_like(tensor)
        quantized[tensor > threshold] = 1
        quantized[tensor < -threshold] = -1

        return quantized

class QuantizationOrchestrator:
    """Orchestrates different quantization techniques."""

    def __init__(self):
        self.quantizers = {
            QuantizationMode.PTQ: PostTrainingQuantizer,
            QuantizationMode.QAT: QuantizationAwareTrainer,
            QuantizationMode.DYNAMIC: DynamicQuantizer
        }

    def create_quantizer(self, config: QuantizationConfig) -> BaseQuantizer:
        """Create quantizer instance."""
        if config.mode not in self.quantizers:
            raise ValueError(f"Unsupported quantization mode: {config.mode}")

        return self.quantizers[config.mode](config)

    def quantize_model(self, model: nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      config: QuantizationConfig) -> Tuple[nn.Module, Dict[str, float]]:
        """Complete quantization pipeline."""
        logger.info(f"Starting quantization with mode: {config.mode.value}")

        # Create quantizer
        quantizer = self.create_quantizer(config)

        # Prepare model
        prepared_model = quantizer.prepare_model(model)

        # Calibrate if needed
        quantizer.calibrate(prepared_model, dataloader)

        # Quantize
        quantized_model = quantizer.quantize_model(prepared_model)

        # Calculate statistics
        stats = quantizer.calculate_compression_stats(model, quantized_model)

        logger.info(f"Quantization completed. Compression ratio: {stats['compression_ratio']:.2f}x")

        return quantized_model, stats

    def compare_quantization_methods(self, model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   modes: List[QuantizationMode]) -> Dict[str, Dict[str, float]]:
        """Compare different quantization methods."""
        results = {}

        for mode in modes:
            logger.info(f"Testing {mode.value}...")

            config = QuantizationConfig(mode=mode)

            try:
                # Create copy of model for testing
                model_copy = torch.nn.utils.prune.identity(model)

                quantized_model, stats = self.quantize_model(model_copy, dataloader, config)
                results[mode.value] = stats

            except Exception as e:
                logger.error(f"Failed to quantize with {mode.value}: {e}")
                results[mode.value] = {'error': str(e)}

        return results

    def progressive_quantization(self, model: nn.Module,
                               dataloader: torch.utils.data.DataLoader,
                               trainer_fn: Optional[Callable] = None,
                               start_bits: int = 8,
                               end_bits: int = 4,
                               steps: int = 3) -> Dict[str, float]:
        """Progressive quantization with gradual bit reduction."""
        current_model = model
        results = []

        bit_schedule = np.linspace(start_bits, end_bits, steps, dtype=int)

        for step, bits in enumerate(bit_schedule):
            logger.info(f"Progressive quantization step {step + 1}/{steps}, bits: {bits}")

            config = QuantizationConfig(
                mode=QuantizationMode.QAT if trainer_fn else QuantizationMode.PTQ,
                weight_bits=bits,
                activation_bits=bits
            )

            quantized_model, stats = self.quantize_model(current_model, dataloader, config)

            # Retrain if trainer function provided
            if trainer_fn and bits < 8:  # Skip retraining for 8-bit
                trainer_fn(quantized_model, epochs=2)

            results.append(stats)
            current_model = quantized_model

        return {
            'final_stats': results[-1],
            'progression': results
        }

# Utility functions
def benchmark_quantized_model(original_model: nn.Module,
                            quantized_model: nn.Module,
                            dataloader: torch.utils.data.DataLoader,
                            device: str = 'cpu') -> Dict[str, float]:
    """Benchmark quantized model performance."""
    import time

    # Move models to device
    original_model = original_model.to(device)
    quantized_model = quantized_model.to(device)

    # Warmup
    dummy_input = next(iter(dataloader))[0].to(device)
    for _ in range(10):
        _ = original_model(dummy_input)
        _ = quantized_model(dummy_input)

    # Benchmark original model
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = original_model(inputs)
    original_time = time.time() - start_time

    # Benchmark quantized model
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = quantized_model(inputs)
    quantized_time = time.time() - start_time

    speedup = original_time / quantized_time

    return {
        'original_time': original_time,
        'quantized_time': quantized_time,
        'speedup': speedup
    }

# Factory function
def create_quantizer(mode: str = "ptq",
                    weight_bits: int = 8,
                    activation_bits: int = 8,
                    backend: str = "fbgemm") -> BaseQuantizer:
    """Factory function to create quantizers."""
    mode_map = {
        'ptq': QuantizationMode.PTQ,
        'qat': QuantizationMode.QAT,
        'dynamic': QuantizationMode.DYNAMIC
    }

    config = QuantizationConfig(
        mode=mode_map[mode],
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        backend=backend
    )

    orchestrator = QuantizationOrchestrator()
    return orchestrator.create_quantizer(config)

if __name__ == "__main__":
    # Example usage
    import torchvision.models as models

    # Create a sample model
    model = models.resnet18(pretrained=True)

    # Create sample data loader
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 1000, (100,))
    )
    dataloader = DataLoader(dummy_data, batch_size=32)

    # Test different quantization methods
    orchestrator = QuantizationOrchestrator()

    modes_to_test = [
        QuantizationMode.PTQ,
        QuantizationMode.DYNAMIC
    ]

    results = orchestrator.compare_quantization_methods(model, dataloader, modes_to_test)

    for method, stats in results.items():
        if 'error' not in stats:
            print(f"{method}: {stats['compression_ratio']:.2f}x compression, "
                  f"{stats['size_reduction_percent']:.1f}% size reduction")
        else:
            print(f"{method}: Error - {stats['error']}")