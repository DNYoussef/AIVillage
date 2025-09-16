"""
Agent Forge Phase 5 - Advanced Learning Rate Scheduler
Adaptive learning rate scheduling with BitNet and Grokfast integration
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import time

class SchedulerType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"
    ADAPTIVE = "adaptive"
    GROKFAST = "grokfast"
    BITNET = "bitnet"

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    # Basic parameters
    scheduler_type: SchedulerType = SchedulerType.COSINE
    base_lr: float = 1e-3
    max_lr: float = 1e-2
    min_lr: float = 1e-6
    final_lr: float = 1e-5

    # Warmup settings
    warmup_steps: int = 1000
    warmup_type: str = "linear"  # linear, cosine, constant

    # Schedule parameters
    total_steps: int = 100000
    decay_steps: int = 50000
    decay_rate: float = 0.95
    power: float = 1.0

    # Cyclic parameters
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.95
    cycle_length: int = 2000

    # Adaptive parameters
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr_threshold: float = 1e-7

    # Performance tracking
    monitor_metric: str = "loss"
    mode: str = "min"  # min or max
    performance_window: int = 100

    # BitNet specific
    bitnet_warmup_multiplier: float = 0.1
    quantization_phase_lr: float = 5e-4

    # Grokfast specific
    acceleration_multiplier: float = 5.0
    consolidation_multiplier: float = 0.2
    phase_transition_smoothing: int = 100

class BaseScheduler(_LRScheduler):
    """Base scheduler with enhanced features"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
        last_epoch: int = -1
    ):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_window)
        self.best_metric = float('inf') if config.mode == 'min' else float('-inf')
        self.plateau_count = 0
        self.cooldown_count = 0
        self.step_count = 0

        super().__init__(optimizer, last_epoch)

    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics for adaptive scheduling"""
        if self.config.monitor_metric in metrics:
            current_metric = metrics[self.config.monitor_metric]
            self.performance_history.append(current_metric)

            # Check for improvement
            is_better = (
                current_metric < self.best_metric if self.config.mode == 'min'
                else current_metric > self.best_metric
            )

            if is_better and abs(current_metric - self.best_metric) > self.config.threshold:
                self.best_metric = current_metric
                self.plateau_count = 0
                self.cooldown_count = 0
            else:
                self.plateau_count += 1

    def get_performance_trend(self) -> float:
        """Calculate performance trend over recent history"""
        if len(self.performance_history) < 2:
            return 0.0

        history = list(self.performance_history)
        x = np.arange(len(history))
        trend = np.polyfit(x, history, 1)[0]

        return trend

class LinearScheduler(BaseScheduler):
    """Linear learning rate scheduler"""

    def get_lr(self):
        if self.last_epoch < self.config.warmup_steps:
            # Warmup phase
            if self.config.warmup_type == "linear":
                lr_scale = self.last_epoch / self.config.warmup_steps
            elif self.config.warmup_type == "cosine":
                lr_scale = 0.5 * (1 + math.cos(math.pi * (1 - self.last_epoch / self.config.warmup_steps)))
            else:  # constant
                lr_scale = self.config.base_lr / self.config.max_lr

            return [self.config.max_lr * lr_scale for _ in self.base_lrs]
        else:
            # Linear decay
            progress = min(1.0, (self.last_epoch - self.config.warmup_steps) /
                          (self.config.total_steps - self.config.warmup_steps))

            lr = self.config.max_lr + progress * (self.config.final_lr - self.config.max_lr)
            return [max(lr, self.config.min_lr) for _ in self.base_lrs]

class CosineScheduler(BaseScheduler):
    """Cosine annealing scheduler with restarts"""

    def __init__(self, optimizer, config: SchedulerConfig, last_epoch=-1):
        super().__init__(optimizer, config, last_epoch)
        self.restart_count = 0

    def get_lr(self):
        if self.last_epoch < self.config.warmup_steps:
            # Warmup phase
            lr_scale = self.last_epoch / self.config.warmup_steps
            return [self.config.base_lr * lr_scale for _ in self.base_lrs]
        else:
            # Cosine annealing
            effective_step = self.last_epoch - self.config.warmup_steps
            effective_total = self.config.total_steps - self.config.warmup_steps

            progress = effective_step / effective_total
            lr = (
                self.config.min_lr +
                0.5 * (self.config.max_lr - self.config.min_lr) *
                (1 + math.cos(math.pi * progress))
            )

            return [max(lr, self.config.min_lr) for _ in self.base_lrs]

class OneCycleScheduler(BaseScheduler):
    """One cycle learning rate policy"""

    def get_lr(self):
        if self.last_epoch < self.config.warmup_steps:
            # Initial ramp up
            progress = self.last_epoch / self.config.warmup_steps
            lr = self.config.base_lr + progress * (self.config.max_lr - self.config.base_lr)
        elif self.last_epoch < self.config.total_steps // 2:
            # Ramp down to base
            phase_steps = self.config.total_steps // 2 - self.config.warmup_steps
            progress = (self.last_epoch - self.config.warmup_steps) / phase_steps
            lr = self.config.max_lr - progress * (self.config.max_lr - self.config.base_lr)
        else:
            # Final ramp down
            phase_start = self.config.total_steps // 2
            phase_steps = self.config.total_steps - phase_start
            progress = (self.last_epoch - phase_start) / phase_steps
            lr = self.config.base_lr - progress * (self.config.base_lr - self.config.final_lr)

        return [max(lr, self.config.min_lr) for _ in self.base_lrs]

class CyclicScheduler(BaseScheduler):
    """Cyclic learning rate scheduler"""

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.config.cycle_length))
        x = abs(self.last_epoch / self.config.cycle_length - 2 * cycle + 1)

        lr = self.config.base_lr + (self.config.max_lr - self.config.base_lr) * max(0, 1 - x)

        return [max(lr, self.config.min_lr) for _ in self.base_lrs]

    def get_momentum(self):
        """Get momentum for cyclic momentum"""
        if not self.config.cycle_momentum:
            return None

        cycle = math.floor(1 + self.last_epoch / (2 * self.config.cycle_length))
        x = abs(self.last_epoch / self.config.cycle_length - 2 * cycle + 1)

        momentum = self.config.max_momentum - (self.config.max_momentum - self.config.base_momentum) * max(0, 1 - x)

        return momentum

class AdaptiveScheduler(BaseScheduler):
    """Adaptive learning rate scheduler based on performance"""

    def __init__(self, optimizer, config: SchedulerConfig, last_epoch=-1):
        super().__init__(optimizer, config, last_epoch)
        self.lr_reductions = 0

    def get_lr(self):
        # Base learning rate with reductions
        current_lr = self.config.base_lr * (self.config.factor ** self.lr_reductions)
        return [max(current_lr, self.config.min_lr_threshold) for _ in self.base_lrs]

    def step(self, metrics=None):
        if metrics is not None:
            self.update_metrics(metrics)

        # Check for plateau
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
        elif self.plateau_count >= self.config.patience:
            if self.get_last_lr()[0] > self.config.min_lr_threshold:
                self.lr_reductions += 1
                self.plateau_count = 0
                self.cooldown_count = self.config.cooldown
                logging.info(f"Reducing learning rate to {self.get_lr()[0]:.2e}")

        super().step()

class GrokfastScheduler(BaseScheduler):
    """Specialized scheduler for Grokfast training phases"""

    def __init__(self, optimizer, config: SchedulerConfig, last_epoch=-1):
        super().__init__(optimizer, config, last_epoch)
        self.phase_transitions = {
            'warmup': config.warmup_steps,
            'acceleration': config.warmup_steps + 5000,
            'consolidation': config.warmup_steps + 7000,
            'refinement': config.warmup_steps + 9000
        }

    def get_current_phase(self) -> str:
        """Determine current Grokfast phase"""
        step = self.last_epoch

        if step < self.phase_transitions['warmup']:
            return 'warmup'
        elif step < self.phase_transitions['acceleration']:
            return 'acceleration'
        elif step < self.phase_transitions['consolidation']:
            return 'consolidation'
        elif step < self.phase_transitions['refinement']:
            return 'refinement'
        else:
            return 'completed'

    def get_lr(self):
        phase = self.get_current_phase()

        if phase == 'warmup':
            progress = self.last_epoch / self.config.warmup_steps
            lr = self.config.base_lr * progress
        elif phase == 'acceleration':
            lr = self.config.base_lr * self.config.acceleration_multiplier
        elif phase == 'consolidation':
            lr = self.config.base_lr * self.config.consolidation_multiplier
        elif phase == 'refinement':
            lr = self.config.base_lr * self.config.consolidation_multiplier * 0.5
        else:
            lr = self.config.min_lr

        # Apply phase transition smoothing
        if hasattr(self, '_last_phase') and self._last_phase != phase:
            smoothing_steps = self.config.phase_transition_smoothing
            transition_progress = min(1.0, (self.last_epoch - self._transition_start) / smoothing_steps)
            lr = self._smooth_transition(self._last_lr, lr, transition_progress)

        self._last_phase = phase
        self._last_lr = lr
        if not hasattr(self, '_transition_start'):
            self._transition_start = self.last_epoch

        return [max(lr, self.config.min_lr) for _ in self.base_lrs]

    def _smooth_transition(self, old_lr: float, new_lr: float, progress: float) -> float:
        """Smooth transition between learning rates"""
        # Cosine interpolation
        factor = 0.5 * (1 - math.cos(math.pi * progress))
        return old_lr + factor * (new_lr - old_lr)

class BitNetScheduler(BaseScheduler):
    """Specialized scheduler for BitNet training"""

    def __init__(self, optimizer, config: SchedulerConfig, last_epoch=-1):
        super().__init__(optimizer, config, last_epoch)
        self.quantization_start = config.warmup_steps

    def get_lr(self):
        if self.last_epoch < self.config.warmup_steps:
            # Reduced learning rate during warmup for stability
            progress = self.last_epoch / self.config.warmup_steps
            lr = self.config.base_lr * self.config.bitnet_warmup_multiplier * progress
        elif self.last_epoch < self.quantization_start + 1000:
            # Quantization phase with special learning rate
            lr = self.config.quantization_phase_lr
        else:
            # Standard cosine schedule
            effective_step = self.last_epoch - self.quantization_start - 1000
            effective_total = self.config.total_steps - self.quantization_start - 1000

            if effective_total > 0:
                progress = min(1.0, effective_step / effective_total)
                lr = (
                    self.config.min_lr +
                    0.5 * (self.config.base_lr - self.config.min_lr) *
                    (1 + math.cos(math.pi * progress))
                )
            else:
                lr = self.config.base_lr

        return [max(lr, self.config.min_lr) for _ in self.base_lrs]

class CompositeScheduler:
    """Composite scheduler that can switch between different strategies"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: Dict[str, BaseScheduler],
        strategy: str = "sequential"
    ):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.strategy = strategy
        self.current_scheduler = None
        self.scheduler_history = []

    def set_scheduler(self, name: str):
        """Switch to a specific scheduler"""
        if name in self.schedulers:
            self.current_scheduler = self.schedulers[name]
            self.scheduler_history.append((name, time.time()))
            logging.info(f"Switched to scheduler: {name}")
        else:
            raise ValueError(f"Scheduler '{name}' not found")

    def step(self, metrics=None):
        """Step the current scheduler"""
        if self.current_scheduler is not None:
            if hasattr(self.current_scheduler, 'step'):
                if metrics is not None and hasattr(self.current_scheduler, 'update_metrics'):
                    self.current_scheduler.update_metrics(metrics)
                self.current_scheduler.step()

    def get_last_lr(self):
        """Get last learning rate"""
        if self.current_scheduler is not None:
            return self.current_scheduler.get_last_lr()
        return [0.0]

    def state_dict(self):
        """Get state dict of all schedulers"""
        return {
            name: scheduler.state_dict()
            for name, scheduler in self.schedulers.items()
        }

    def load_state_dict(self, state_dict):
        """Load state dict for all schedulers"""
        for name, scheduler_state in state_dict.items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(scheduler_state)

class SchedulerFactory:
    """Factory for creating learning rate schedulers"""

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig
    ) -> BaseScheduler:
        """Create scheduler based on configuration"""

        if config.scheduler_type == SchedulerType.LINEAR:
            return LinearScheduler(optimizer, config)
        elif config.scheduler_type == SchedulerType.COSINE:
            return CosineScheduler(optimizer, config)
        elif config.scheduler_type == SchedulerType.ONE_CYCLE:
            return OneCycleScheduler(optimizer, config)
        elif config.scheduler_type == SchedulerType.CYCLIC:
            return CyclicScheduler(optimizer, config)
        elif config.scheduler_type == SchedulerType.ADAPTIVE:
            return AdaptiveScheduler(optimizer, config)
        elif config.scheduler_type == SchedulerType.GROKFAST:
            return GrokfastScheduler(optimizer, config)
        elif config.scheduler_type == SchedulerType.BITNET:
            return BitNetScheduler(optimizer, config)
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    @staticmethod
    def create_composite_scheduler(
        optimizer: torch.optim.Optimizer,
        configs: Dict[str, SchedulerConfig]
    ) -> CompositeScheduler:
        """Create composite scheduler with multiple strategies"""
        schedulers = {}
        for name, config in configs.items():
            schedulers[name] = SchedulerFactory.create_scheduler(optimizer, config)

        return CompositeScheduler(optimizer, schedulers)

class SchedulerMonitor:
    """Monitor and analyze scheduler performance"""

    def __init__(self):
        self.lr_history = []
        self.metric_history = []
        self.phase_history = []
        self.step_times = []

    def record_step(
        self,
        step: int,
        lr: float,
        metrics: Dict[str, float],
        phase: Optional[str] = None
    ):
        """Record a training step"""
        self.lr_history.append((step, lr))
        self.metric_history.append((step, metrics.copy()))
        if phase:
            self.phase_history.append((step, phase))
        self.step_times.append(time.time())

    def get_lr_effectiveness(self, metric_name: str = "loss") -> Dict[str, float]:
        """Analyze learning rate effectiveness"""
        if len(self.metric_history) < 10:
            return {}

        # Extract data
        steps = [item[0] for item in self.lr_history]
        lrs = [item[1] for item in self.lr_history]
        metrics = [item[1].get(metric_name, 0) for item in self.metric_history]

        # Calculate correlations
        lr_metric_corr = np.corrcoef(lrs[-100:], metrics[-100:])[0, 1] if len(lrs) >= 100 else 0

        # Calculate learning rate stability
        lr_stability = 1.0 / (1.0 + np.std(lrs[-100:]))

        # Calculate convergence rate
        if len(metrics) >= 50:
            recent_trend = np.polyfit(range(50), metrics[-50:], 1)[0]
            convergence_rate = -recent_trend if recent_trend < 0 else 0
        else:
            convergence_rate = 0

        return {
            'lr_metric_correlation': lr_metric_corr,
            'lr_stability': lr_stability,
            'convergence_rate': convergence_rate,
            'final_metric': metrics[-1] if metrics else 0
        }

    def plot_history(self, save_path: Optional[str] = None):
        """Plot scheduler history (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot learning rate
            steps = [item[0] for item in self.lr_history]
            lrs = [item[1] for item in self.lr_history]
            ax1.plot(steps, lrs, 'b-', linewidth=2)
            ax1.set_ylabel('Learning Rate')
            ax1.set_title('Learning Rate Schedule')
            ax1.grid(True)

            # Plot metrics
            metric_steps = [item[0] for item in self.metric_history]
            for metric_name in ['loss', 'accuracy', 'val_loss']:
                values = [item[1].get(metric_name) for item in self.metric_history]
                if any(v is not None for v in values):
                    values = [v for v in values if v is not None]
                    ax2.plot(metric_steps[:len(values)], values, label=metric_name)

            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Metric Value')
            ax2.set_title('Training Metrics')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            logging.warning("Matplotlib not available for plotting")

if __name__ == "__main__":
    # Example usage and testing
    import torch.nn as nn

    # Create test model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test different schedulers
    configs = {
        'cosine': SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            base_lr=1e-3,
            max_lr=1e-2,
            total_steps=1000,
            warmup_steps=100
        ),
        'grokfast': SchedulerConfig(
            scheduler_type=SchedulerType.GROKFAST,
            base_lr=1e-3,
            acceleration_multiplier=5.0,
            consolidation_multiplier=0.2
        ),
        'bitnet': SchedulerConfig(
            scheduler_type=SchedulerType.BITNET,
            base_lr=1e-3,
            bitnet_warmup_multiplier=0.1,
            quantization_phase_lr=5e-4
        )
    }

    # Test individual schedulers
    for name, config in configs.items():
        scheduler = SchedulerFactory.create_scheduler(optimizer, config)
        print(f"\n{name.upper()} Scheduler Test:")

        for step in range(10):
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step}: LR = {lr:.6f}")
            scheduler.step()

    # Test composite scheduler
    print("\nComposite Scheduler Test:")
    composite = SchedulerFactory.create_composite_scheduler(optimizer, configs)

    composite.set_scheduler('cosine')
    for step in range(5):
        lr = composite.get_last_lr()[0]
        print(f"  Cosine Step {step}: LR = {lr:.6f}")
        composite.step()

    composite.set_scheduler('grokfast')
    for step in range(5):
        lr = composite.get_last_lr()[0]
        print(f"  Grokfast Step {step}: LR = {lr:.6f}")
        composite.step()

    # Test monitoring
    monitor = SchedulerMonitor()
    for step in range(50):
        lr = 1e-3 * (0.95 ** step)
        metrics = {'loss': 1.0 * (0.99 ** step)}
        monitor.record_step(step, lr, metrics)

    effectiveness = monitor.get_lr_effectiveness()
    print(f"\nScheduler Effectiveness: {effectiveness}")

    print("Scheduler test completed successfully!")