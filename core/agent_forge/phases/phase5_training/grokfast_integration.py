"""
Agent Forge Phase 5: Grokfast Integration
==========================================

Grokfast rapid capability acquisition system for accelerated learning
and knowledge consolidation in BitNet training.

Key Features:
- Rapid capability acquisition algorithms
- Knowledge consolidation strategies
- Transfer learning enhancement
- Learning efficiency optimization
- Adaptive learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
from pathlib import Path


@dataclass
class GrokfastConfig:
    """Configuration for Grokfast acceleration."""

    # Core Grokfast parameters
    alpha: float = 0.98          # EMA coefficient for gradient tracking
    lambda_reg: float = 2.0      # Regularization strength
    use_layer_adaptation: bool = True
    adaptation_threshold: float = 0.1

    # Learning acceleration
    rapid_learning_phases: int = 3
    phase_duration: int = 1000    # Steps per phase
    capability_threshold: float = 0.85

    # Knowledge consolidation
    consolidation_interval: int = 500
    knowledge_retention: float = 0.9
    transfer_strength: float = 0.3

    # Adaptive scheduling
    use_adaptive_lr: bool = True
    lr_acceleration_factor: float = 2.0
    lr_deceleration_factor: float = 0.5
    plateau_patience: int = 100


class GrokfastLayer(nn.Module):
    """
    Enhanced layer with Grokfast capability acquisition.

    Implements rapid learning mechanisms and knowledge consolidation
    for efficient capability acquisition.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        config: GrokfastConfig
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config

        # Grokfast state tracking
        self.register_buffer('ema_gradients', None)
        self.register_buffer('capability_scores', torch.zeros(1))
        self.register_buffer('learning_phase', torch.zeros(1, dtype=torch.long))

        # Knowledge consolidation buffers
        self.knowledge_bank = {}
        self.capability_history = deque(maxlen=1000)

        # Adaptation tracking
        self.layer_stats = {
            'activations': deque(maxlen=100),
            'gradients': deque(maxlen=100),
            'learning_rates': deque(maxlen=100)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with capability tracking."""

        # Base layer computation
        output = self.base_layer(x)

        # Track activation statistics during training
        if self.training:
            with torch.no_grad():
                activation_norm = output.norm(dim=-1).mean().item()
                self.layer_stats['activations'].append(activation_norm)

                # Update capability score based on activation patterns
                self._update_capability_score(output)

        return output

    def _update_capability_score(self, output: torch.Tensor) -> None:
        """Update layer capability score based on output patterns."""

        with torch.no_grad():
            # Measure output diversity and stability
            output_std = output.std(dim=0).mean().item()
            output_mean = output.mean().item()

            # Calculate capability score (higher = more capable)
            capability = min(1.0, output_std / (abs(output_mean) + 1e-8))

            # Update EMA capability score
            self.capability_scores = (
                0.9 * self.capability_scores + 0.1 * capability
            )

            self.capability_history.append(capability)

    def consolidate_knowledge(self) -> Dict[str, torch.Tensor]:
        """Consolidate learned knowledge for transfer."""

        consolidated = {}

        # Store important weight patterns
        if hasattr(self.base_layer, 'weight'):
            consolidated['weights'] = self.base_layer.weight.data.clone()

        if hasattr(self.base_layer, 'bias') and self.base_layer.bias is not None:
            consolidated['bias'] = self.base_layer.bias.data.clone()

        # Store capability patterns
        consolidated['capability_score'] = self.capability_scores.clone()

        return consolidated

    def transfer_knowledge(self, knowledge: Dict[str, torch.Tensor]) -> None:
        """Transfer consolidated knowledge to accelerate learning."""

        if 'weights' in knowledge and hasattr(self.base_layer, 'weight'):
            # Weighted transfer based on transfer strength
            current_weights = self.base_layer.weight.data
            transferred_weights = knowledge['weights']

            # Blend current and transferred weights
            blend_factor = self.config.transfer_strength
            self.base_layer.weight.data = (
                (1 - blend_factor) * current_weights +
                blend_factor * transferred_weights
            )

        if 'bias' in knowledge and hasattr(self.base_layer, 'bias'):
            if self.base_layer.bias is not None:
                current_bias = self.base_layer.bias.data
                transferred_bias = knowledge['bias']

                blend_factor = self.config.transfer_strength
                self.base_layer.bias.data = (
                    (1 - blend_factor) * current_bias +
                    blend_factor * transferred_bias
                )


class GrokfastAccelerator:
    """
    Grokfast acceleration system for rapid capability acquisition.

    Implements gradient-based acceleration, knowledge consolidation,
    and adaptive learning strategies for efficient training.
    """

    def __init__(self, config):
        self.config = config
        self.grok_config = GrokfastConfig()

        # Setup logging
        self.logger = logging.getLogger('grokfast_accelerator')

        # State tracking
        self.global_step = 0
        self.current_phase = 0
        self.acceleration_active = False

        # Gradient tracking for Grokfast algorithm
        self.gradient_ema = {}
        self.capability_scores = {}

        # Learning efficiency metrics
        self.learning_metrics = {
            'loss_improvements': deque(maxlen=1000),
            'gradient_norms': deque(maxlen=1000),
            'learning_speeds': deque(maxlen=1000),
            'capability_growth': deque(maxlen=1000)
        }

        # Knowledge consolidation
        self.knowledge_bank = {}
        self.consolidation_schedule = []

        # Adaptive learning rate tracking
        self.lr_history = deque(maxlen=1000)
        self.plateau_counter = 0
        self.best_loss = float('inf')

        self.logger.info("Grokfast accelerator initialized")

    def initialize(self, model: nn.Module) -> None:
        """Initialize Grokfast acceleration for model."""

        # Wrap compatible layers with GrokfastLayer
        self._wrap_model_layers(model)

        # Initialize gradient EMA buffers
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.gradient_ema[name] = torch.zeros_like(param.data)
                self.capability_scores[name] = 0.0

        # Setup consolidation schedule
        self._setup_consolidation_schedule()

        self.logger.info(f"Grokfast initialized for {len(self.gradient_ema)} parameters")

    def _wrap_model_layers(self, model: nn.Module) -> None:
        """Wrap model layers with Grokfast enhancement."""

        wrapped_count = 0

        for name, module in model.named_modules():
            # Wrap Linear and Conv layers
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                # Create GrokfastLayer wrapper
                grokfast_layer = GrokfastLayer(module, self.grok_config)

                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]

                parent_module = model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)

                setattr(parent_module, layer_name, grokfast_layer)
                wrapped_count += 1

        self.logger.info(f"Wrapped {wrapped_count} layers with Grokfast enhancement")

    def _setup_consolidation_schedule(self) -> None:
        """Setup knowledge consolidation schedule."""

        total_phases = self.grok_config.rapid_learning_phases
        phase_duration = self.grok_config.phase_duration
        consolidation_interval = self.grok_config.consolidation_interval

        for phase in range(total_phases):
            phase_start = phase * phase_duration
            phase_end = (phase + 1) * phase_duration

            # Schedule consolidations within each phase
            for step in range(phase_start, phase_end, consolidation_interval):
                self.consolidation_schedule.append({
                    'step': step,
                    'phase': phase,
                    'type': 'regular_consolidation'
                })

            # End-of-phase major consolidation
            self.consolidation_schedule.append({
                'step': phase_end,
                'phase': phase,
                'type': 'phase_consolidation'
            })

    def accelerate_if_needed(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        current_loss: float
    ) -> bool:
        """Check if acceleration is needed and apply Grokfast."""

        self.global_step += 1

        # Update learning metrics
        if self.learning_metrics['loss_improvements']:
            loss_improvement = self.learning_metrics['loss_improvements'][-1] - current_loss
            self.learning_metrics['loss_improvements'].append(current_loss)
        else:
            self.learning_metrics['loss_improvements'].append(current_loss)
            loss_improvement = 0.0

        # Calculate learning speed
        learning_speed = abs(loss_improvement) if loss_improvement != 0 else 0.0
        self.learning_metrics['learning_speeds'].append(learning_speed)

        # Apply Grokfast gradient acceleration
        acceleration_applied = self._apply_grokfast_gradients(model, optimizer)

        # Check for consolidation
        consolidation_applied = self._check_consolidation(model)

        # Adaptive learning rate adjustment
        lr_adjusted = self._adjust_learning_rate(optimizer, current_loss)

        # Phase progression
        self._update_learning_phase()

        return acceleration_applied or consolidation_applied or lr_adjusted

    def _apply_grokfast_gradients(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> bool:
        """Apply Grokfast gradient acceleration."""

        acceleration_applied = False
        total_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is None or name not in self.gradient_ema:
                continue

            # Update gradient EMA
            current_grad = param.grad.data
            self.gradient_ema[name] = (
                self.grok_config.alpha * self.gradient_ema[name] +
                (1 - self.grok_config.alpha) * current_grad
            )

            # Calculate gradient norm for monitoring
            grad_norm = current_grad.norm().item()
            total_grad_norm += grad_norm ** 2

            # Apply Grokfast acceleration when beneficial
            if self._should_accelerate(name, current_grad):
                # Grokfast acceleration: blend current and EMA gradients
                acceleration_factor = self._calculate_acceleration_factor(name)

                accelerated_grad = (
                    current_grad +
                    acceleration_factor * self.gradient_ema[name]
                )

                # Apply regularization to prevent instability
                reg_term = self.grok_config.lambda_reg * (
                    accelerated_grad - current_grad
                )

                param.grad.data = accelerated_grad - reg_term
                acceleration_applied = True

                # Update capability score
                self._update_parameter_capability(name, accelerated_grad)

        # Track overall gradient norm
        total_grad_norm = math.sqrt(total_grad_norm)
        self.learning_metrics['gradient_norms'].append(total_grad_norm)

        if acceleration_applied:
            self.logger.debug(f"Grokfast acceleration applied at step {self.global_step}")

        return acceleration_applied

    def _should_accelerate(self, param_name: str, current_grad: torch.Tensor) -> bool:
        """Determine if parameter should be accelerated."""

        if param_name not in self.gradient_ema:
            return False

        ema_grad = self.gradient_ema[param_name]

        # Check gradient consistency (same direction)
        cosine_sim = F.cosine_similarity(
            current_grad.flatten(),
            ema_grad.flatten(),
            dim=0
        ).item()

        # Accelerate if gradients are consistent and significant
        return (
            cosine_sim > 0.5 and  # Same direction
            ema_grad.norm().item() > 1e-6  # Significant magnitude
        )

    def _calculate_acceleration_factor(self, param_name: str) -> float:
        """Calculate dynamic acceleration factor for parameter."""

        base_factor = 0.1  # Conservative base acceleration

        # Increase acceleration for high-capability parameters
        capability_bonus = self.capability_scores.get(param_name, 0.0) * 0.2

        # Adjust based on learning phase
        phase_factor = 1.0 + (self.current_phase * 0.1)

        return base_factor * (1.0 + capability_bonus) * phase_factor

    def _update_parameter_capability(
        self,
        param_name: str,
        accelerated_grad: torch.Tensor
    ) -> None:
        """Update capability score for parameter."""

        # Measure gradient stability and magnitude
        grad_magnitude = accelerated_grad.norm().item()
        grad_stability = 1.0 / (1.0 + accelerated_grad.var().item())

        # Combine metrics for capability score
        new_capability = min(1.0, grad_magnitude * grad_stability * 0.1)

        # Update with EMA
        if param_name in self.capability_scores:
            self.capability_scores[param_name] = (
                0.9 * self.capability_scores[param_name] +
                0.1 * new_capability
            )
        else:
            self.capability_scores[param_name] = new_capability

    def _check_consolidation(self, model: nn.Module) -> bool:
        """Check if knowledge consolidation should be performed."""

        # Check consolidation schedule
        for scheduled in self.consolidation_schedule:
            if scheduled['step'] == self.global_step:
                return self._perform_consolidation(model, scheduled['type'])

        # Adaptive consolidation based on capability growth
        if len(self.learning_metrics['capability_growth']) >= 10:
            recent_growth = np.mean(list(self.learning_metrics['capability_growth'])[-10:])
            if recent_growth < 0.01:  # Slow capability growth
                return self._perform_consolidation(model, 'adaptive_consolidation')

        return False

    def _perform_consolidation(
        self,
        model: nn.Module,
        consolidation_type: str
    ) -> bool:
        """Perform knowledge consolidation."""

        consolidated_layers = 0

        for name, module in model.named_modules():
            if isinstance(module, GrokfastLayer):
                # Consolidate knowledge from this layer
                knowledge = module.consolidate_knowledge()

                # Store in knowledge bank
                layer_key = f"{name}_{consolidation_type}"
                self.knowledge_bank[layer_key] = knowledge

                consolidated_layers += 1

        if consolidated_layers > 0:
            self.logger.info(
                f"Knowledge consolidation completed: {consolidated_layers} layers, "
                f"type: {consolidation_type}"
            )
            return True

        return False

    def _adjust_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        current_loss: float
    ) -> bool:
        """Adaptive learning rate adjustment based on Grokfast principles."""

        if not self.grok_config.use_adaptive_lr:
            return False

        # Track loss improvement
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)

        lr_adjusted = False

        # Accelerate learning rate during rapid improvement
        if self.plateau_counter == 0 and len(self.lr_history) >= 10:
            recent_improvements = sum(
                1 for i in range(-10, -1)
                if i < len(self.learning_metrics['loss_improvements']) - 1 and
                self.learning_metrics['loss_improvements'][i] >
                self.learning_metrics['loss_improvements'][i + 1]
            )

            if recent_improvements >= 7:  # Consistent improvement
                new_lr = current_lr * self.grok_config.lr_acceleration_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min(new_lr, self.config.learning_rate * 10)
                lr_adjusted = True

                self.logger.debug(f"Learning rate accelerated: {current_lr:.6f} -> {new_lr:.6f}")

        # Decelerate learning rate during plateau
        elif self.plateau_counter >= self.grok_config.plateau_patience:
            new_lr = current_lr * self.grok_config.lr_deceleration_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(new_lr, self.config.learning_rate * 0.01)
            lr_adjusted = True

            self.plateau_counter = 0  # Reset counter
            self.logger.debug(f"Learning rate decelerated: {current_lr:.6f} -> {new_lr:.6f}")

        return lr_adjusted

    def _update_learning_phase(self) -> None:
        """Update current learning phase."""

        phase_duration = self.grok_config.phase_duration
        new_phase = min(
            self.global_step // phase_duration,
            self.grok_config.rapid_learning_phases - 1
        )

        if new_phase > self.current_phase:
            self.current_phase = new_phase
            self.logger.info(f"Entering learning phase {self.current_phase}")

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create Grokfast-optimized learning rate scheduler."""

        class GrokfastScheduler(_LRScheduler):
            """Custom scheduler with Grokfast acceleration."""

            def __init__(self, optimizer, grokfast_accelerator, num_training_steps, num_warmup_steps):
                self.grokfast = grokfast_accelerator
                self.num_training_steps = num_training_steps
                self.num_warmup_steps = num_warmup_steps
                super().__init__(optimizer)

            def get_lr(self):
                current_step = self._step_count

                if current_step < self.num_warmup_steps:
                    # Warmup phase
                    warmup_factor = current_step / max(1, self.num_warmup_steps)
                    return [base_lr * warmup_factor for base_lr in self.base_lrs]

                else:
                    # Grokfast-enhanced cosine annealing
                    progress = (current_step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)

                    # Base cosine decay
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

                    # Grokfast phase modulation
                    phase_factor = 1.0
                    if self.grokfast.current_phase < self.grokfast.grok_config.rapid_learning_phases:
                        phase_factor = 1.0 + (0.2 * self.grokfast.current_phase)

                    return [base_lr * cosine_factor * phase_factor for base_lr in self.base_lrs]

        return GrokfastScheduler(optimizer, self, num_training_steps, num_warmup_steps)

    def transfer_capabilities(
        self,
        source_model: nn.Module,
        target_model: nn.Module
    ) -> int:
        """Transfer learned capabilities between models."""

        transfers_performed = 0

        # Match layers by name and type
        source_layers = {name: module for name, module in source_model.named_modules()}
        target_layers = {name: module for name, module in target_model.named_modules()}

        for name in source_layers:
            if (name in target_layers and
                isinstance(source_layers[name], GrokfastLayer) and
                isinstance(target_layers[name], GrokfastLayer)):

                # Transfer knowledge from source to target
                source_knowledge = source_layers[name].consolidate_knowledge()
                target_layers[name].transfer_knowledge(source_knowledge)

                transfers_performed += 1

        self.logger.info(f"Capability transfer completed: {transfers_performed} layers")
        return transfers_performed

    def get_acceleration_stats(self) -> Dict[str, Any]:
        """Get comprehensive Grokfast acceleration statistics."""

        # Calculate average capability score
        avg_capability = (
            sum(self.capability_scores.values()) / len(self.capability_scores)
            if self.capability_scores else 0.0
        )

        # Calculate learning speed trends
        recent_speeds = list(self.learning_metrics['learning_speeds'])[-100:]
        avg_learning_speed = np.mean(recent_speeds) if recent_speeds else 0.0

        return {
            'global_step': self.global_step,
            'current_phase': self.current_phase,
            'acceleration_active': self.acceleration_active,
            'average_capability_score': avg_capability,
            'total_parameters_tracked': len(self.gradient_ema),
            'knowledge_bank_size': len(self.knowledge_bank),
            'consolidations_completed': len([k for k in self.knowledge_bank.keys() if 'consolidation' in k]),
            'learning_metrics': {
                'average_learning_speed': avg_learning_speed,
                'gradient_norm_trend': np.mean(list(self.learning_metrics['gradient_norms'])[-100:]) if self.learning_metrics['gradient_norms'] else 0.0,
                'loss_improvement_rate': np.mean(np.diff(list(self.learning_metrics['loss_improvements'])[-50:])) if len(self.learning_metrics['loss_improvements']) > 50 else 0.0
            },
            'adaptive_lr_stats': {
                'current_lr': self.lr_history[-1] if self.lr_history else 0.0,
                'plateau_counter': self.plateau_counter,
                'best_loss': self.best_loss
            }
        }


if __name__ == "__main__":
    # Example usage and testing
    def test_grokfast_integration():
        """Test Grokfast integration components."""
        from training_config import TrainingConfig

        config = TrainingConfig()

        # Create test model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        # Create Grokfast accelerator
        accelerator = GrokfastAccelerator(config)
        accelerator.initialize(model)

        # Test acceleration detection
        optimizer = torch.optim.Adam(model.parameters())
        dummy_loss = 1.0

        acceleration_applied = accelerator.accelerate_if_needed(model, optimizer, dummy_loss)
        print(f"✓ Acceleration system: {acceleration_applied}")

        # Test scheduler creation
        scheduler = accelerator.create_scheduler(
            optimizer,
            num_training_steps=1000,
            num_warmup_steps=100
        )
        print(f"✓ Grokfast scheduler created: {type(scheduler)}")

        # Test statistics
        stats = accelerator.get_acceleration_stats()
        print(f"✓ Acceleration stats: Phase {stats['current_phase']}, Capability {stats['average_capability_score']:.4f}")

        print("Grokfast integration test completed successfully")

    # Run test
    test_grokfast_integration()