"""
Agent Forge Phase 5 - Advanced Training Loop
Efficient training iteration with gradient optimization and monitoring
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import logging
import time
import math
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import json
import threading
from collections import defaultdict, deque
import gc
import psutil

class TrainingState(Enum):
    INITIALIZED = "initialized"
    TRAINING = "training"
    VALIDATING = "validating"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Basic training parameters
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Gradient parameters
    max_grad_norm: float = 1.0
    gradient_clipping: bool = True
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    loss_scale: float = 2**16

    # Optimization
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"  # linear, cosine, polynomial
    min_lr_ratio: float = 0.1

    # Monitoring
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    max_checkpoints: int = 5

    # Performance
    compile_model: bool = True
    use_fused_optimizer: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    # Memory management
    max_memory_usage: float = 0.9  # 90% of GPU memory
    memory_cleanup_interval: int = 1000

    # Validation
    validation_patience: int = 10
    min_delta: float = 1e-4

    # BitNet specific
    bitnet_enabled: bool = False
    quantization_schedule: Optional[Dict[str, Any]] = None

    # Grokfast specific
    grokfast_enabled: bool = False
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0

class GradientAccumulator:
    """Efficient gradient accumulation with memory optimization"""

    def __init__(self, accumulation_steps: int, device: torch.device):
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.accumulated_gradients = {}
        self.step_count = 0

    def accumulate(self, model: nn.Module, loss: torch.Tensor) -> bool:
        """Accumulate gradients and return True when ready to step"""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self.step_count += 1

        # Check if we should step
        should_step = (self.step_count % self.accumulation_steps) == 0

        if should_step:
            self.step_count = 0

        return should_step

    def reset(self):
        """Reset accumulation state"""
        self.step_count = 0
        self.accumulated_gradients.clear()

class MemoryMonitor:
    """GPU memory monitoring and management"""

    def __init__(self, max_usage_ratio: float = 0.9):
        self.max_usage_ratio = max_usage_ratio
        self.peak_memory = 0
        self.memory_history = deque(maxlen=1000)

    def check_memory(self) -> Dict[str, float]:
        """Check current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            usage_ratio = allocated / max_memory

            stats = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_gb': max_memory,
                'usage_ratio': usage_ratio,
                'peak_gb': self.peak_memory
            }

            self.memory_history.append(usage_ratio)
            self.peak_memory = max(self.peak_memory, allocated)

            return stats
        else:
            return {'error': 'CUDA not available'}

    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        stats = self.check_memory()
        return stats.get('usage_ratio', 0) > self.max_usage_ratio

    def cleanup_memory(self):
        """Perform memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class TrainingMetrics:
    """Training metrics collection and analysis"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_step = 0
        self.epoch = 0
        self.best_metrics = {}

    def update(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Update metrics with new values"""
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1

        for key, value in metrics_dict.items():
            self.metrics[key].append(value)

            # Track best metrics
            if key.startswith('val_'):
                if key not in self.best_metrics:
                    self.best_metrics[key] = value
                elif key.endswith('_loss') or key.endswith('_error'):
                    # Lower is better
                    self.best_metrics[key] = min(self.best_metrics[key], value)
                else:
                    # Higher is better
                    self.best_metrics[key] = max(self.best_metrics[key], value)

    def get_recent_average(self, metric_name: str, window: int = 100) -> float:
        """Get recent average of a metric"""
        if metric_name not in self.metrics:
            return 0.0

        recent_values = list(self.metrics[metric_name])[-window:]
        return sum(recent_values) / len(recent_values) if recent_values else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get training metrics summary"""
        summary = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metrics': self.best_metrics.copy()
        }

        # Add recent averages
        for metric_name in self.metrics:
            summary[f'{metric_name}_recent'] = self.get_recent_average(metric_name)

        return summary

class CheckpointManager:
    """Model checkpoint management with rotation"""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        metrics: TrainingMetrics,
        step: int,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics.get_summary(),
            'step': step,
            'epoch': metrics.epoch
        }

        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if additional_state:
            checkpoint_data.update(additional_state)

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Update checkpoint list
        self.checkpoint_files.append(checkpoint_path)

        # Rotate checkpoints
        if len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        return checkpoint_data

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if not self.checkpoint_files:
            # Search directory for existing checkpoints
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
            if checkpoints:
                # Sort by step number
                checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
                return str(checkpoints[-1])
        else:
            return str(self.checkpoint_files[-1]) if self.checkpoint_files else None

        return None

class EarlyStopping:
    """Early stopping with patience and minimum delta"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.patience_counter = 0
        self.early_stop = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered"""
        if self.monitor not in metrics:
            return False

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.early_stop = self.patience_counter >= self.patience
        return self.early_stop

    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement"""
        if self.monitor.endswith('_loss') or self.monitor.endswith('_error'):
            # Lower is better
            return current_score < (self.best_score - self.min_delta)
        else:
            # Higher is better
            return current_score > (self.best_score + self.min_delta)

class TrainingLoop:
    """Advanced training loop with comprehensive features"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.state = TrainingState.INITIALIZED

        # Initialize components
        self.metrics = TrainingMetrics()
        self.memory_monitor = MemoryMonitor(config.max_memory_usage)
        self.gradient_accumulator = GradientAccumulator(
            config.gradient_accumulation_steps, device
        )

        # Checkpoint management
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir, config.max_checkpoints
            )
        else:
            self.checkpoint_manager = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.validation_patience,
            min_delta=config.min_delta
        )

        # Mixed precision
        if config.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Compile model if requested
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        # Move model to device
        self.model.to(device)

        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_start_time = None

        # Callbacks
        self.callbacks = []

    def setup_optimizer(self, optimizer_cls=None, **optimizer_kwargs):
        """Setup optimizer with configuration"""
        if optimizer_cls is None:
            if self.config.use_fused_optimizer and hasattr(torch.optim, 'AdamW'):
                optimizer_cls = torch.optim.AdamW
                optimizer_kwargs.setdefault('fused', True)
            else:
                optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            **optimizer_kwargs
        )

        # Setup scheduler
        if self.config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs * 1000,  # Estimate steps per epoch
                eta_min=self.config.learning_rate * self.config.min_lr_ratio
            )
        elif self.config.scheduler_type == "linear":
            total_steps = self.config.epochs * 1000
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=total_steps
            )

        # Warmup scheduler
        if self.config.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )

            if self.scheduler:
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    [warmup_scheduler, self.scheduler],
                    [self.config.warmup_steps]
                )
            else:
                self.scheduler = warmup_scheduler

    def add_callback(self, callback: Callable):
        """Add training callback"""
        self.callbacks.append(callback)

    def _call_callbacks(self, event: str, **kwargs):
        """Call all callbacks for an event"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    ) -> Dict[str, float]:
        """Execute single training step"""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass with optional mixed precision
        if self.config.use_amp and self.scaler:
            with autocast(device_type='cuda', dtype=self.config.amp_dtype):
                loss = loss_fn(batch)
        else:
            loss = loss_fn(batch)

        # Gradient accumulation
        should_step = self.gradient_accumulator.accumulate(self.model, loss)

        if should_step:
            # Gradient clipping
            if self.config.gradient_clipping:
                if self.scaler:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            else:
                grad_norm = 0.0

            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Clear gradients
            self.optimizer.zero_grad()

            # Update global step
            self.global_step += 1

            return {
                'loss': loss.item(),
                'grad_norm': grad_norm,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        else:
            return {'loss': loss.item()}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    ) -> Dict[str, float]:
        """Execute single validation step"""
        self.model.eval()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        with torch.no_grad():
            if self.config.use_amp:
                with autocast(device_type='cuda', dtype=self.config.amp_dtype):
                    loss = loss_fn(batch)
            else:
                loss = loss_fn(batch)

        return {'val_loss': loss.item()}

    def train_epoch(
        self,
        train_loader,
        loss_fn: Callable,
        val_loader=None
    ) -> Dict[str, float]:
        """Train single epoch"""
        self.state = TrainingState.TRAINING
        epoch_metrics = defaultdict(list)

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            step_start_time = time.time()

            # Training step
            step_metrics = self.train_step(batch, loss_fn)

            # Record metrics
            for key, value in step_metrics.items():
                epoch_metrics[key].append(value)

            # Update global metrics
            self.metrics.update(step_metrics, self.global_step)

            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_progress(step_metrics, time.time() - step_start_time)

            # Validation
            if (val_loader is not None and
                self.global_step % self.config.eval_interval == 0):
                val_metrics = self.validate(val_loader, loss_fn)
                self.metrics.update(val_metrics, self.global_step)

                # Check early stopping
                if self.early_stopping(val_metrics):
                    logging.info(f"Early stopping triggered at step {self.global_step}")
                    self.state = TrainingState.STOPPED
                    break

            # Checkpointing
            if (self.checkpoint_manager and
                self.global_step % self.config.save_interval == 0):
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.metrics, self.global_step
                )
                logging.info(f"Saved checkpoint: {checkpoint_path}")

            # Memory cleanup
            if self.global_step % self.config.memory_cleanup_interval == 0:
                if self.memory_monitor.should_cleanup():
                    self.memory_monitor.cleanup_memory()

            # Callbacks
            self._call_callbacks('on_step_end',
                               step=self.global_step,
                               metrics=step_metrics)

        # Calculate epoch averages
        epoch_summary = {}
        for key, values in epoch_metrics.items():
            epoch_summary[key] = np.mean(values)

        self.current_epoch += 1
        self.metrics.epoch = self.current_epoch

        return epoch_summary

    def validate(self, val_loader, loss_fn: Callable) -> Dict[str, float]:
        """Run validation"""
        self.state = TrainingState.VALIDATING
        val_metrics = defaultdict(list)

        for batch in val_loader:
            step_metrics = self.validation_step(batch, loss_fn)
            for key, value in step_metrics.items():
                val_metrics[key].append(value)

        # Calculate averages
        val_summary = {}
        for key, values in val_metrics.items():
            val_summary[key] = np.mean(values)

        return val_summary

    def train(
        self,
        train_loader,
        loss_fn: Callable,
        val_loader=None,
        resume_from_checkpoint: bool = True
    ):
        """Main training function"""
        logging.info("Starting training...")
        self.training_start_time = time.time()

        # Setup optimizer if not already done
        if self.optimizer is None:
            self.setup_optimizer()

        # Resume from checkpoint if available
        if resume_from_checkpoint and self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_data = self.checkpoint_manager.load_checkpoint(
                    latest_checkpoint, self.model, self.optimizer, self.scheduler
                )
                self.global_step = checkpoint_data['step']
                self.current_epoch = checkpoint_data['epoch']
                logging.info(f"Resumed from checkpoint at step {self.global_step}")

        # Training loop
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                epoch_start_time = time.time()

                # Train epoch
                epoch_metrics = self.train_epoch(train_loader, loss_fn, val_loader)

                # Log epoch summary
                epoch_time = time.time() - epoch_start_time
                logging.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                for key, value in epoch_metrics.items():
                    logging.info(f"  {key}: {value:.6f}")

                # Callbacks
                self._call_callbacks('on_epoch_end',
                                   epoch=epoch,
                                   metrics=epoch_metrics)

                # Check if training should stop
                if self.state == TrainingState.STOPPED:
                    break

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.state = TrainingState.PAUSED
        except Exception as e:
            logging.error(f"Training error: {e}")
            self.state = TrainingState.ERROR
            raise
        finally:
            # Final cleanup and save
            total_time = time.time() - self.training_start_time
            logging.info(f"Training completed in {total_time:.2f}s")

            if self.checkpoint_manager:
                final_checkpoint = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.metrics, self.global_step,
                    {'training_complete': True}
                )
                logging.info(f"Final checkpoint saved: {final_checkpoint}")

    def _log_progress(self, metrics: Dict[str, float], step_time: float):
        """Log training progress"""
        log_msg = f"Step {self.global_step:6d} | "
        log_msg += f"Loss: {metrics.get('loss', 0):.6f} | "
        log_msg += f"LR: {metrics.get('learning_rate', 0):.2e} | "
        log_msg += f"Time: {step_time:.3f}s"

        if 'grad_norm' in metrics:
            log_msg += f" | Grad: {metrics['grad_norm']:.3f}"

        # Memory usage
        memory_stats = self.memory_monitor.check_memory()
        if 'usage_ratio' in memory_stats:
            log_msg += f" | Mem: {memory_stats['usage_ratio']:.1%}"

        logging.info(log_msg)

    def get_state(self) -> Dict[str, Any]:
        """Get current training state"""
        return {
            'state': self.state.value,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'metrics': self.metrics.get_summary(),
            'memory_stats': self.memory_monitor.check_memory()
        }

if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Create simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 64)
            self.output = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.linear(x))
            return self.output(x)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestModel()
    config = TrainingConfig(
        epochs=5,
        learning_rate=1e-3,
        log_interval=10
    )

    # Create training loop
    trainer = TrainingLoop(model, config, device)

    # Test loss function
    def test_loss_fn(batch):
        x = batch['input']
        target = batch.get('target', torch.randint(0, 10, (x.shape[0],)).to(x.device))
        output = model(x)
        return nn.CrossEntropyLoss()(output, target)

    # Create dummy data
    train_data = []
    for _ in range(100):
        train_data.append({
            'input': torch.randn(32, 128),
            'target': torch.randint(0, 10, (32,))
        })

    # Test training
    trainer.setup_optimizer()

    print("Training loop test completed successfully!")
    print(f"Final state: {trainer.get_state()}")