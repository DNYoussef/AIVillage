"""
Phase 5 Training Automation - Training Pipeline System
Automated training workflows with orchestration and quality gates
"""

import os
import json
import yaml
import asyncio
import logging
import shutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class TrainingConfig:
    """Training configuration with validation"""
    model_name: str
    dataset_path: str
    output_dir: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "cosine"
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 5
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    timestamp: datetime
    gpu_memory: float = 0.0
    training_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class TrainingOrchestrator:
    """Orchestrates training workflows with automation"""

    def __init__(self, base_dir: str = "automation/runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Training state
        self.current_run_id = None
        self.training_history = []
        self.best_model_path = None
        self.best_val_loss = float('inf')

        # Resource management
        self.device = self._auto_detect_device()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Quality gates
        self.quality_gates = {
            'min_accuracy': 0.8,
            'max_loss': 1.0,
            'max_overfitting_ratio': 1.5,
            'min_improvement_threshold': 0.001
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"training_orchestrator_{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            log_dir = self.base_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

        return logger

    def _auto_detect_device(self) -> torch.device:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("Using MPS (Apple Silicon)")
            return device
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
            return device

    async def create_training_run(self, config: TrainingConfig) -> str:
        """Create new training run with unique ID"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.model_name}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)

        # Create directories
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        self.current_run_id = run_id
        self.logger.info(f"Created training run: {run_id}")

        return run_id

    async def prepare_data(self, config: TrainingConfig) -> Dict[str, DataLoader]:
        """Prepare data loaders for training"""
        self.logger.info("Preparing data loaders...")

        # This is a placeholder - implement actual data loading logic
        # based on your dataset format
        try:
            # Example implementation for common formats
            if config.dataset_path.endswith('.pkl'):
                with open(config.dataset_path, 'rb') as f:
                    data = pickle.load(f)
            elif config.dataset_path.endswith('.json'):
                with open(config.dataset_path, 'r') as f:
                    data = json.load(f)
            else:
                # Directory-based dataset
                data = self._load_directory_dataset(config.dataset_path)

            # Split data
            train_data, val_data = self._split_data(data, config.validation_split)

            # Create data loaders
            train_loader = DataLoader(
                train_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                drop_last=config.drop_last
            )

            val_loader = DataLoader(
                val_data,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )

            self.logger.info(f"Data prepared: {len(train_data)} training, {len(val_data)} validation samples")

            return {
                'train': train_loader,
                'val': val_loader
            }

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

    def _load_directory_dataset(self, dataset_path: str):
        """Load dataset from directory structure"""
        # Implement based on your specific dataset structure
        # This is a placeholder implementation
        return []

    def _split_data(self, data, validation_split: float):
        """Split data into training and validation sets"""
        # Implement data splitting logic
        # This is a placeholder implementation
        split_idx = int(len(data) * (1 - validation_split))
        return data[:split_idx], data[split_idx:]

    async def setup_model_and_optimizer(self, config: TrainingConfig, model: nn.Module) -> Dict[str, Any]:
        """Setup model, optimizer, and scheduler"""
        self.logger.info("Setting up model and optimizer...")

        # Move model to device
        model = model.to(self.device)

        # Setup optimizer
        if config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # Setup scheduler
        if config.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs
            )
        elif config.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config.epochs // 3, gamma=0.1
            )
        elif config.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            scheduler = None

        # Setup mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None

        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scaler': scaler
        }

    async def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler: Optional[torch.cuda.amp.GradScaler],
        config: TrainingConfig
    ) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            if config.use_mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()

                if config.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()

                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

                optimizer.step()

            # Track metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            # Calculate accuracy (for classification)
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    async def validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = model(data)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                # Calculate accuracy (for classification)
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == targets).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    async def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: TrainingMetrics,
        is_best: bool = False
    ):
        """Save model checkpoint"""
        if not self.current_run_id:
            raise RuntimeError("No active training run")

        run_dir = self.base_dir / self.current_run_id
        checkpoint_dir = run_dir / "checkpoints"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': asdict(metrics),
            'timestamp': datetime.now().isoformat()
        }

        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            self.logger.info(f"New best model saved: {best_path}")

        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

    async def check_quality_gates(self, metrics: TrainingMetrics) -> Dict[str, bool]:
        """Check quality gates for training"""
        gates_passed = {}

        # Minimum accuracy gate
        gates_passed['min_accuracy'] = metrics.val_accuracy >= self.quality_gates['min_accuracy']

        # Maximum loss gate
        gates_passed['max_loss'] = metrics.val_loss <= self.quality_gates['max_loss']

        # Overfitting detection
        if metrics.train_loss > 0:
            overfitting_ratio = metrics.val_loss / metrics.train_loss
            gates_passed['overfitting'] = overfitting_ratio <= self.quality_gates['max_overfitting_ratio']
        else:
            gates_passed['overfitting'] = True

        # Improvement threshold
        if len(self.training_history) > 5:
            recent_metrics = self.training_history[-5:]
            recent_losses = [m.val_loss for m in recent_metrics]
            improvement = max(recent_losses) - min(recent_losses)
            gates_passed['improvement'] = improvement >= self.quality_gates['min_improvement_threshold']
        else:
            gates_passed['improvement'] = True

        return gates_passed

    async def run_training_pipeline(
        self,
        config: TrainingConfig,
        model: nn.Module,
        criterion: nn.Module,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """Run complete training pipeline"""
        try:
            # Create training run
            run_id = await self.create_training_run(config)
            self.logger.info(f"Starting training pipeline: {run_id}")

            # Prepare data
            data_loaders = await self.prepare_data(config)

            # Setup model and optimizer
            training_setup = await self.setup_model_and_optimizer(config, model)
            model = training_setup['model']
            optimizer = training_setup['optimizer']
            scheduler = training_setup['scheduler']
            scaler = training_setup['scaler']

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(config.epochs):
                epoch_start_time = time.time()

                # Training
                train_metrics = await self.train_epoch(
                    model, data_loaders['train'], optimizer, criterion, scaler, config
                )

                # Validation
                val_metrics = await self.validate_epoch(
                    model, data_loaders['val'], criterion
                )

                # Calculate metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']

                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    train_accuracy=train_metrics['accuracy'],
                    val_accuracy=val_metrics['accuracy'],
                    learning_rate=current_lr,
                    timestamp=datetime.now(),
                    training_time=epoch_time
                )

                if torch.cuda.is_available():
                    metrics.gpu_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

                self.training_history.append(metrics)

                # Check if best model
                is_best = val_metrics['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save checkpoint
                if epoch % config.save_every_n_epochs == 0 or is_best:
                    await self.save_checkpoint(
                        model, optimizer, scheduler, epoch, metrics, is_best
                    )

                # Quality gates
                quality_gates = await self.check_quality_gates(metrics)

                # Logging
                self.logger.info(
                    f"Epoch {epoch:04d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save metrics
                run_dir = self.base_dir / run_id
                metrics_file = run_dir / "metrics" / f"epoch_{epoch:04d}.json"
                with open(metrics_file, 'w') as f:
                    json.dump({
                        **metrics.to_dict(),
                        'quality_gates': quality_gates
                    }, f, indent=2)

                # Scheduler step
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics['loss'])
                    else:
                        scheduler.step()

                # Early stopping
                if patience_counter >= config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        await callback(epoch, metrics, quality_gates)

            # Final checkpoint
            await self.save_checkpoint(
                model, optimizer, scheduler, epoch, metrics, False
            )

            # Training summary
            summary = {
                'run_id': run_id,
                'total_epochs': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_model_path': self.best_model_path,
                'final_metrics': metrics.to_dict(),
                'quality_gates_passed': all(quality_gates.values()),
                'training_history': [m.to_dict() for m in self.training_history]
            }

            # Save summary
            summary_file = run_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Training completed: {run_id}")
            return summary

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise

    async def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return checkpoint

    async def resume_training(
        self,
        checkpoint_path: str,
        config: TrainingConfig,
        model: nn.Module,
        criterion: nn.Module
    ) -> Dict[str, Any]:
        """Resume training from checkpoint"""
        # Load checkpoint
        checkpoint = await self.load_checkpoint(checkpoint_path)

        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Setup optimizer and scheduler
        training_setup = await self.setup_model_and_optimizer(config, model)
        optimizer = training_setup['optimizer']
        scheduler = training_setup['scheduler']

        # Restore optimizer and scheduler state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Update config to start from checkpoint epoch
        config.epochs = config.epochs - checkpoint['epoch']

        self.logger.info(f"Resuming training from epoch {checkpoint['epoch']}")

        # Continue training
        return await self.run_training_pipeline(config, model, criterion)


class TrainingPipelineManager:
    """Manages multiple training pipelines"""

    def __init__(self, base_dir: str = "automation/runs"):
        self.base_dir = Path(base_dir)
        self.orchestrators = {}
        self.active_runs = {}

    async def create_orchestrator(self, name: str) -> TrainingOrchestrator:
        """Create new training orchestrator"""
        orchestrator = TrainingOrchestrator(str(self.base_dir / name))
        self.orchestrators[name] = orchestrator
        return orchestrator

    async def run_parallel_training(
        self,
        training_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run multiple training pipelines in parallel"""
        results = {}

        async def run_single_training(config_dict):
            name = config_dict['name']
            config = TrainingConfig(**config_dict['config'])
            model = config_dict['model']
            criterion = config_dict['criterion']

            orchestrator = await self.create_orchestrator(name)
            result = await orchestrator.run_training_pipeline(config, model, criterion)
            return name, result

        # Run training pipelines concurrently
        tasks = [run_single_training(config) for config in training_configs]

        for task in asyncio.as_completed(tasks):
            name, result = await task
            results[name] = result

        return results

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of training run"""
        for orchestrator in self.orchestrators.values():
            if orchestrator.current_run_id == run_id:
                return {
                    'status': 'active',
                    'history_length': len(orchestrator.training_history),
                    'best_model_path': orchestrator.best_model_path,
                    'device': str(orchestrator.device)
                }

        return {'status': 'not_found'}


# Example usage
async def main():
    """Example training pipeline usage"""

    # Create simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, input_size=784, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Configuration
    config = TrainingConfig(
        model_name="simple_classifier",
        dataset_path="data/dataset.pkl",
        output_dir="automation/runs",
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )

    # Create orchestrator
    orchestrator = TrainingOrchestrator()

    # Create model and criterion
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()

    # Run training
    result = await orchestrator.run_training_pipeline(config, model, criterion)
    print(f"Training completed: {result['run_id']}")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())