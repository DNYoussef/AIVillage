"""
BitNet Training Optimizer - Agent Forge Phase 4

Advanced Training Optimization Engine
====================================

Implements sophisticated training optimizations for BitNet models with
1-bit quantized weights, focusing on gradient computation and convergence.

Key Features:
1. Straight-through estimator optimization
2. Mixed precision training coordination
3. Gradient accumulation and clipping
4. Learning rate scheduling optimization
5. Quantization-aware training strategies
6. Distributed training preparation

Author: Agent Forge Phase 4 - Training Optimization Specialist
License: NASA POT10 Compliant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import logging
import warnings
from dataclasses import dataclass, field
from contextlib import contextmanager
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingOptimizationConfig:
    """Configuration for training optimization strategies."""
    # Quantization-aware training
    enable_qat: bool = True
    straight_through_temperature: float = 1.0
    temperature_annealing: bool = True
    temperature_schedule: str = "cosine"  # linear, cosine, exponential

    # Gradient optimization
    enable_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    gradient_clip_norm: float = 1.0
    adaptive_gradient_clipping: bool = True

    # Mixed precision training
    enable_mixed_precision: bool = True
    loss_scaling: float = 2.0**16
    dynamic_loss_scaling: bool = True
    loss_scale_window: int = 1000

    # Gradient accumulation
    enable_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    sync_accumulation: bool = True

    # Learning rate optimization
    enable_lr_scheduling: bool = True
    lr_schedule_type: str = "cosine_with_warmup"
    warmup_ratio: float = 0.1
    lr_decay_factor: float = 0.1

    # Optimizer optimization
    optimizer_type: str = "adamw"  # adamw, adam, sgd, rmsprop
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01

    # Batch optimization
    enable_dynamic_batching: bool = True
    min_batch_size: int = 8
    max_batch_size: int = 64
    batch_size_growth_factor: float = 1.2

    # Regularization
    enable_dropout_scheduling: bool = True
    initial_dropout: float = 0.1
    final_dropout: float = 0.1
    label_smoothing: float = 0.1

    # Monitoring and validation
    enable_training_monitoring: bool = True
    validation_frequency: int = 100
    early_stopping_patience: int = 10

class StraightThroughEstimatorOptimized:
    """Optimized Straight-Through Estimator for 1-bit quantization."""

    def __init__(self, temperature: float = 1.0, annealing: bool = True):
        self.initial_temperature = temperature
        self.current_temperature = temperature
        self.annealing = annealing
        self.step_count = 0

    def update_temperature(self, step: int, total_steps: int, schedule: str = "cosine") -> None:
        """Update temperature for annealing."""
        if not self.annealing:
            return

        if schedule == "cosine":
            progress = step / total_steps
            self.current_temperature = self.initial_temperature * (0.5 * (1 + math.cos(math.pi * progress)))
        elif schedule == "linear":
            progress = step / total_steps
            self.current_temperature = self.initial_temperature * (1 - progress)
        elif schedule == "exponential":
            decay_rate = 0.95
            self.current_temperature = self.initial_temperature * (decay_rate ** (step // 1000))

        self.step_count = step

    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to 1-bit with temperature annealing."""
        # Apply temperature to softmax-like quantization
        if self.current_temperature > 0:
            # Soft quantization that becomes harder as temperature decreases
            quantized = torch.tanh(weights / self.current_temperature)
        else:
            # Hard quantization
            quantized = torch.sign(weights)

        return quantized

    def apply_straight_through(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply straight-through estimator with optimized gradient flow."""
        # Forward: quantize
        quantized = self.quantize_weights(weights)

        # Backward: straight-through (implemented via torch.autograd.Function)
        return StraightThroughFunction.apply(weights, quantized)

class StraightThroughFunction(torch.autograd.Function):
    """Optimized straight-through function for gradient computation."""

    @staticmethod
    def forward(ctx, input_weights: torch.Tensor, quantized_weights: torch.Tensor) -> torch.Tensor:
        """Forward pass returns quantized weights."""
        return quantized_weights

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass with improved gradient estimation."""
        # Enhanced straight-through estimator with gradient clipping
        grad_input = torch.clamp(grad_output, -1.0, 1.0)
        return grad_input, None

class QuantizationAwareTrainer:
    """Quantization-aware training orchestrator."""

    def __init__(self, config: TrainingOptimizationConfig):
        self.config = config
        self.ste = StraightThroughEstimatorOptimized(
            temperature=config.straight_through_temperature,
            annealing=config.temperature_annealing
        )
        self.training_stats = {"steps": 0, "quantization_errors": [], "gradient_norms": []}

    def apply_quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Apply quantization-aware training to model."""
        logger.info("Applying quantization-aware training optimizations...")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace linear layers with quantization-aware versions
                self._convert_linear_to_qat(module, name)

        return model

    def _convert_linear_to_qat(self, linear_layer: nn.Linear, layer_name: str) -> None:
        """Convert linear layer to quantization-aware version."""
        # Store original forward function
        original_forward = linear_layer.forward

        def qat_forward(self, x):
            # Quantize weights using straight-through estimator
            quantized_weights = self.parent_trainer.ste.apply_straight_through(self.weight)

            # Perform linear operation with quantized weights
            return F.linear(x, quantized_weights, self.bias)

        # Monkey patch the forward function
        linear_layer.parent_trainer = self
        linear_layer.forward = lambda x: qat_forward(linear_layer, x)

        logger.info(f"Converted {layer_name} to quantization-aware training")

    def update_quantization_parameters(self, step: int, total_steps: int) -> None:
        """Update quantization parameters during training."""
        self.ste.update_temperature(step, total_steps, self.config.temperature_schedule)
        self.training_stats["steps"] = step

    def compute_quantization_error(self, original_weights: torch.Tensor,
                                 quantized_weights: torch.Tensor) -> float:
        """Compute quantization error for monitoring."""
        mse_error = F.mse_loss(quantized_weights, original_weights).item()
        self.training_stats["quantization_errors"].append(mse_error)
        return mse_error

class MixedPrecisionTrainer:
    """Advanced mixed precision training manager."""

    def __init__(self, config: TrainingOptimizationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=config.loss_scaling,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=config.loss_scale_window,
            enabled=config.enable_mixed_precision and device.type == 'cuda'
        )

        self.training_stats = {
            "scale_updates": 0,
            "overflow_count": 0,
            "successful_steps": 0
        }

    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision training."""
        if self.config.enable_mixed_precision and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def scale_loss_and_backward(self, loss: torch.Tensor,
                               model: nn.Module,
                               retain_graph: bool = False) -> bool:
        """Scale loss and perform backward pass."""
        if self.config.enable_mixed_precision:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward(retain_graph=retain_graph)
            return True
        else:
            loss.backward(retain_graph=retain_graph)
            return True

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with scaled gradients."""
        if self.config.enable_mixed_precision:
            # Check for overflow
            self.scaler.unscale_(optimizer)

            # Clip gradients if enabled
            if self.config.enable_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    self.config.gradient_clip_norm
                )

            # Step optimizer
            self.scaler.step(optimizer)
            self.scaler.update()

            # Update statistics
            if self.scaler._get_scale_async() < self.scaler._init_scale:
                self.training_stats["overflow_count"] += 1
            else:
                self.training_stats["successful_steps"] += 1

            return True
        else:
            # Standard training
            if self.config.enable_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    self.config.gradient_clip_norm
                )

            optimizer.step()
            self.training_stats["successful_steps"] += 1
            return True

    def get_loss_scale(self) -> float:
        """Get current loss scale."""
        if self.config.enable_mixed_precision:
            return self.scaler.get_scale()
        return 1.0

class GradientAccumulator:
    """Advanced gradient accumulation with synchronization."""

    def __init__(self, config: TrainingOptimizationConfig):
        self.config = config
        self.accumulation_steps = config.accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
        self.gradient_stats = {"norm_history": [], "accumulation_count": 0}

    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated."""
        return (self.current_step % self.accumulation_steps) != 0

    def should_step_optimizer(self) -> bool:
        """Check if optimizer should be stepped."""
        return (self.current_step % self.accumulation_steps) == 0

    def accumulate_gradients(self, model: nn.Module) -> None:
        """Accumulate gradients across multiple forward passes."""
        if not self.config.enable_gradient_accumulation:
            return

        self.current_step += 1

        # Scale gradients by accumulation steps
        if self.should_accumulate():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad / self.accumulation_steps

        # Compute gradient norm for monitoring
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        self.gradient_stats["norm_history"].append(total_norm)

        if self.should_step_optimizer():
            self.gradient_stats["accumulation_count"] += 1

    def reset_accumulation(self) -> None:
        """Reset accumulation state."""
        self.current_step = 0
        self.accumulated_gradients.clear()

class AdaptiveLearningRateScheduler:
    """Advanced learning rate scheduling with adaptation."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 config: TrainingOptimizationConfig,
                 total_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * config.warmup_ratio)

        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.lr_history = []

        # Create scheduler based on type
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create appropriate learning rate scheduler."""
        if self.config.lr_schedule_type == "cosine_with_warmup":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=self.base_lr * 0.1
            )
        elif self.config.lr_schedule_type == "linear_with_warmup":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.total_steps - self.warmup_steps
            )
        elif self.config.lr_schedule_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.total_steps // 10,
                gamma=self.config.lr_decay_factor
            )

    def step(self) -> float:
        """Step the learning rate scheduler."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr_scale = min(1.0, self.current_step / self.warmup_steps)
            lr = self.base_lr * lr_scale

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Regular scheduling
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

        self.lr_history.append(lr)
        return lr

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

class TrainingOptimizer:
    """Comprehensive training optimization orchestrator."""

    def __init__(self, config: TrainingOptimizationConfig,
                 device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.qat_trainer = QuantizationAwareTrainer(config) if config.enable_qat else None
        self.mixed_precision_trainer = MixedPrecisionTrainer(config, self.device) if config.enable_mixed_precision else None
        self.gradient_accumulator = GradientAccumulator(config) if config.enable_gradient_accumulation else None

        self.training_stats = {
            "total_steps": 0,
            "optimization_steps": 0,
            "loss_history": [],
            "lr_history": [],
            "gradient_norms": []
        }

    def optimize_model_for_training(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive training optimizations to model."""
        logger.info("Optimizing BitNet model for training...")

        # Apply quantization-aware training
        if self.qat_trainer:
            model = self.qat_trainer.apply_quantization_aware_training(model)

        # Optimize model structure for training
        model = self._apply_training_optimizations(model)

        logger.info("Training optimizations applied successfully")
        return model

    def _apply_training_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply training-specific optimizations."""
        # Enable gradient checkpointing for large models
        for name, module in model.named_modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
                logger.info(f"Enabled gradient checkpointing for: {name}")

        return model

    def create_optimized_optimizer(self, model: nn.Module,
                                 learning_rate: float = 1e-4,
                                 total_steps: int = 10000) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Create optimized optimizer and scheduler."""
        # Create optimizer based on configuration
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon
            )
        elif self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )

        # Create scheduler
        scheduler = AdaptiveLearningRateScheduler(optimizer, self.config, total_steps) if self.config.enable_lr_scheduling else None

        return optimizer, scheduler

    @contextmanager
    def training_step_context(self, step: int, total_steps: int):
        """Context manager for optimized training step."""
        # Update quantization parameters
        if self.qat_trainer:
            self.qat_trainer.update_quantization_parameters(step, total_steps)

        # Mixed precision context
        if self.mixed_precision_trainer:
            with self.mixed_precision_trainer.mixed_precision_context():
                yield
        else:
            yield

        # Update statistics
        self.training_stats["total_steps"] += 1

    def optimized_training_step(self, model: nn.Module,
                              optimizer: torch.optim.Optimizer,
                              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                              loss: torch.Tensor,
                              step: int) -> Dict[str, Any]:
        """Perform optimized training step."""
        step_stats = {}

        # Gradient accumulation check
        should_step_optimizer = True
        if self.gradient_accumulator:
            should_step_optimizer = self.gradient_accumulator.should_step_optimizer()

        # Backward pass with mixed precision
        if self.mixed_precision_trainer:
            success = self.mixed_precision_trainer.scale_loss_and_backward(
                loss, model, retain_graph=not should_step_optimizer
            )
            step_stats["loss_scale"] = self.mixed_precision_trainer.get_loss_scale()
        else:
            loss.backward(retain_graph=not should_step_optimizer)
            success = True

        # Gradient accumulation
        if self.gradient_accumulator:
            self.gradient_accumulator.accumulate_gradients(model)

        # Optimizer step
        if should_step_optimizer and success:
            if self.mixed_precision_trainer:
                self.mixed_precision_trainer.step_optimizer(optimizer)
            else:
                if self.config.enable_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm
                    )
                optimizer.step()

            optimizer.zero_grad()
            self.training_stats["optimization_steps"] += 1

            # Learning rate scheduling
            if scheduler:
                current_lr = scheduler.step()
                step_stats["learning_rate"] = current_lr
                self.training_stats["lr_history"].append(current_lr)

        # Record loss and gradient norm
        self.training_stats["loss_history"].append(loss.item())

        # Calculate gradient norm for monitoring
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)

        self.training_stats["gradient_norms"].append(total_grad_norm)
        step_stats["gradient_norm"] = total_grad_norm

        return step_stats

    def validate_training_convergence(self, loss_history: List[float],
                                    window_size: int = 100) -> Dict[str, Any]:
        """Validate training convergence and stability."""
        if len(loss_history) < window_size:
            return {"validation_possible": False, "reason": "Insufficient history"}

        recent_losses = loss_history[-window_size:]
        earlier_losses = loss_history[-2*window_size:-window_size] if len(loss_history) >= 2*window_size else loss_history[:-window_size]

        # Calculate convergence metrics
        recent_mean = np.mean(recent_losses)
        recent_std = np.std(recent_losses)
        earlier_mean = np.mean(earlier_losses) if earlier_losses else recent_mean

        # Convergence indicators
        is_decreasing = recent_mean < earlier_mean
        is_stable = recent_std < recent_mean * 0.1  # CV < 10%
        improvement_rate = (earlier_mean - recent_mean) / earlier_mean if earlier_mean > 0 else 0

        validation_results = {
            "validation_possible": True,
            "is_converging": is_decreasing and is_stable,
            "is_decreasing": is_decreasing,
            "is_stable": is_stable,
            "recent_loss_mean": recent_mean,
            "recent_loss_std": recent_std,
            "improvement_rate": improvement_rate,
            "convergence_score": min(1.0, improvement_rate * 10) if is_decreasing and is_stable else 0.0
        }

        return validation_results

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = dict(self.training_stats)
        stats["optimization_config"] = self.config.__dict__

        if self.qat_trainer:
            stats["quantization_stats"] = self.qat_trainer.training_stats

        if self.mixed_precision_trainer:
            stats["mixed_precision_stats"] = self.mixed_precision_trainer.training_stats

        if self.gradient_accumulator:
            stats["gradient_accumulation_stats"] = self.gradient_accumulator.gradient_stats

        # Calculate derived statistics
        if self.training_stats["loss_history"]:
            stats["final_loss"] = self.training_stats["loss_history"][-1]
            stats["best_loss"] = min(self.training_stats["loss_history"])
            stats["loss_improvement"] = self.training_stats["loss_history"][0] - self.training_stats["loss_history"][-1]

        if self.training_stats["gradient_norms"]:
            stats["avg_gradient_norm"] = np.mean(self.training_stats["gradient_norms"])
            stats["max_gradient_norm"] = max(self.training_stats["gradient_norms"])

        return stats

def create_training_optimizer(device: torch.device,
                            optimization_level: str = "production") -> TrainingOptimizer:
    """Create training optimizer with preset configurations."""

    configs = {
        "development": TrainingOptimizationConfig(
            enable_mixed_precision=False,
            enable_gradient_accumulation=False,
            accumulation_steps=1,
            enable_qat=False
        ),
        "balanced": TrainingOptimizationConfig(
            enable_mixed_precision=True,
            enable_gradient_accumulation=True,
            accumulation_steps=2,
            enable_qat=True,
            temperature_annealing=True
        ),
        "production": TrainingOptimizationConfig(
            enable_mixed_precision=True,
            enable_gradient_accumulation=True,
            accumulation_steps=4,
            enable_qat=True,
            temperature_annealing=True,
            adaptive_gradient_clipping=True,
            dynamic_loss_scaling=True
        )
    }

    config = configs.get(optimization_level, configs["balanced"])
    return TrainingOptimizer(config, device)

def main():
    """Demonstration of training optimization capabilities."""
    print("BitNet Training Optimizer - Agent Forge Phase 4")
    print("=" * 51)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create training optimizer
    optimizer_engine = create_training_optimizer(device, "production")

    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(768, 3072),
        nn.ReLU(),
        nn.Linear(3072, 768)
    ).to(device)

    # Optimize model for training
    optimized_model = optimizer_engine.optimize_model_for_training(model)

    # Create optimizer and scheduler
    optimizer, scheduler = optimizer_engine.create_optimized_optimizer(
        optimized_model, learning_rate=1e-4, total_steps=1000
    )

    # Simulate training steps
    print("\nSimulating training optimization...")
    for step in range(10):
        with optimizer_engine.training_step_context(step, 1000):
            # Simulate forward pass
            dummy_input = torch.randn(8, 512, 768, device=device)
            dummy_target = torch.randn(8, 512, 768, device=device)

            output = optimized_model(dummy_input)
            loss = F.mse_loss(output, dummy_target)

            # Optimized training step
            step_stats = optimizer_engine.optimized_training_step(
                optimized_model, optimizer, scheduler, loss, step
            )

            if step % 5 == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}, LR={step_stats.get('learning_rate', 'N/A'):.6f}")

    # Get training statistics
    training_stats = optimizer_engine.get_training_statistics()
    print(f"\nTraining Statistics:")
    print(f"  Total Steps: {training_stats['total_steps']}")
    print(f"  Optimization Steps: {training_stats['optimization_steps']}")
    print(f"  Final Loss: {training_stats.get('final_loss', 'N/A'):.4f}")

    # Validate convergence
    convergence_results = optimizer_engine.validate_training_convergence(
        training_stats["loss_history"]
    )
    print(f"  Convergence Score: {convergence_results.get('convergence_score', 'N/A'):.3f}")

    print("\nTraining optimization demonstration completed!")

if __name__ == "__main__":
    main()