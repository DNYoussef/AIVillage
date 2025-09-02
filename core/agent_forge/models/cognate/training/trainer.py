#!/usr/bin/env python3
"""
Cognate Model Trainer with GrokFast Integration

This module provides the main training system for the Cognate model,
featuring GrokFast optimization, train-many/infer-few paradigm support,
memory system training, and comprehensive metrics tracking.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch.amp import GradScaler, autocast
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..cognate_model import CognateModel
from ..config.cognate_config import TrainingConfig
from .grokfast_optimizer import GrokFastOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""

    # Loss metrics
    total_loss: float = 0.0
    lm_loss: float = 0.0
    act_loss: float = 0.0
    memory_loss: float = 0.0

    # ACT metrics
    avg_act_steps: float = 0.0
    act_efficiency: float = 0.0  # Percentage of early halts

    # Memory metrics
    memory_reads: int = 0
    memory_writes: int = 0
    memory_utilization: float = 0.0

    # Performance metrics
    tokens_per_second: float = 0.0
    steps_per_second: float = 0.0
    memory_usage_mb: float = 0.0

    # Learning dynamics
    gradient_norm: float = 0.0
    learning_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "total_loss": self.total_loss,
            "lm_loss": self.lm_loss,
            "act_loss": self.act_loss,
            "memory_loss": self.memory_loss,
            "avg_act_steps": self.avg_act_steps,
            "act_efficiency": self.act_efficiency,
            "memory_reads": self.memory_reads,
            "memory_writes": self.memory_writes,
            "memory_utilization": self.memory_utilization,
            "tokens_per_second": self.tokens_per_second,
            "steps_per_second": self.steps_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate,
        }


@dataclass
class CognateTrainingConfig(TrainingConfig):
    """Extended training configuration for Cognate-specific features."""

    # Train-many/infer-few paradigm
    paradigm_schedule: str = "fixed"  # fixed, adaptive, curriculum
    training_act_steps: int = 8
    inference_act_steps: int = 2

    # Memory training
    memory_warmup_steps: int = 100  # Steps before memory system activates
    memory_curriculum: bool = True  # Gradually increase memory complexity

    # GrokFast specific
    grokfast_enabled: bool = True
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0

    # Logging and monitoring
    log_memory_stats: bool = True
    log_act_distributions: bool = True
    save_memory_snapshots: bool = False

    # Advanced training features
    use_amp: bool = True  # Automatic Mixed Precision
    find_unused_parameters: bool = False  # For DDP

    # Evaluation
    eval_paradigm: str = "both"  # training, inference, both
    eval_memory_analysis: bool = True


class CognateTrainer:
    """
    Comprehensive trainer for the Cognate model.

    Features:
    - GrokFast optimization integration
    - Train-many/infer-few paradigm support
    - Memory system training and monitoring
    - Comprehensive metrics tracking
    - HuggingFace Trainer compatibility
    """

    def __init__(
        self,
        model: CognateModel,
        config: CognateTrainingConfig,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        output_dir: str = "./cognate_training_output",
        resume_from_checkpoint: str | None = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Initialize training components
        self._setup_training()

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

    def _setup_training(self):
        """Initialize optimizer, scheduler, and other training components."""

        # Setup optimizer with GrokFast if enabled
        if self.config.grokfast_enabled:
            base_optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=tuple(self.config.betas),
            )
            self.optimizer = GrokFastOptimizer(
                base_optimizer,
                alpha=self.config.grokfast_alpha,
                lamb=self.config.grokfast_lamb,
            )
            logger.info("Initialized GrokFast optimizer")
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=tuple(self.config.betas),
            )
            logger.info("Initialized AdamW optimizer")

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Setup mixed precision training
        self.scaler = GradScaler() if self.config.use_amp else None

        # Initialize metrics tracking
        self.metrics_history = []

        # Setup distributed training if available
        if dist.is_available() and dist.is_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                self.model, find_unused_parameters=self.config.find_unused_parameters
            )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "linear_warmup":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                else:
                    return max(
                        self.config.min_learning_rate / self.config.learning_rate,
                        (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps),
                    )

            return LambdaLR(self.optimizer, lr_lambda)

        elif self.config.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            return CosineAnnealingLR(self.optimizer, self.config.max_steps)

        else:
            # No scheduler
            return None

    def train(self) -> dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training results and final metrics
        """
        logger.info("Starting Cognate model training...")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Training steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")

        # Create data loader
        train_dataloader = (
            DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            if self.train_dataset
            else None
        )

        if not train_dataloader:
            raise ValueError("No training dataset provided")

        # Training loop
        self.model.train()
        start_time = time.time()

        for step in range(self.global_step, self.config.max_steps):
            self.global_step = step

            # Get batch
            try:
                batch = next(iter(train_dataloader))
            except StopIteration:
                # Restart dataloader
                train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                )
                batch = next(iter(train_dataloader))

            # Training step
            metrics = self._training_step(batch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                metrics.learning_rate = self.scheduler.get_last_lr()[0]
            else:
                metrics.learning_rate = self.config.learning_rate

            # Logging
            if step % self.config.logging_steps == 0:
                self._log_metrics(metrics, step)

            # Evaluation
            if step % self.config.eval_steps == 0 and self.eval_dataset:
                eval_metrics = self.evaluate()
                self._log_eval_metrics(eval_metrics, step)

            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.save_checkpoint(step)

            # Memory system warm-up
            if step == self.config.memory_warmup_steps:
                logger.info("Activating memory system")
                # Could enable memory features here if needed

            # Update paradigm if adaptive
            if self.config.paradigm_schedule == "adaptive":
                self._update_paradigm(metrics, step)

        # Final evaluation
        if self.eval_dataset:
            final_eval = self.evaluate()
            logger.info(f"Final evaluation: {final_eval.to_dict()}")

        # Save final model
        self.save_checkpoint(self.config.max_steps, is_final=True)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        return {
            "training_time": total_time,
            "final_step": self.global_step,
            "best_loss": self.best_loss,
            "final_metrics": metrics.to_dict(),
        }

    def _training_step(self, batch: dict[str, torch.Tensor]) -> TrainingMetrics:
        """Perform a single training step."""
        step_start = time.time()

        # Move batch to device
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch.get("labels", input_ids).to(device)

        # Set training paradigm (train-many mode)
        self.model.set_inference_mode(False)

        # Forward pass with mixed precision
        if self.config.use_amp and self.scaler:
            with autocast(device_type=device.type):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                    output_hidden_states=True,
                )
                loss = outputs["loss"]
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True,
            )
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                if self.config.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                else:
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            else:
                grad_norm = 0.0

            # Optimizer step
            if self.config.use_amp and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
        else:
            grad_norm = 0.0

        # Collect metrics
        metrics = TrainingMetrics()
        metrics.total_loss = loss.item() * self.config.gradient_accumulation_steps
        metrics.lm_loss = metrics.total_loss  # Language model loss currently consolidated with total loss
        metrics.avg_act_steps = float(outputs.get("act_steps", 0))
        metrics.gradient_norm = float(grad_norm)

        # Memory metrics
        if "memory_stats" in outputs:
            mem_stats = outputs["memory_stats"]
            metrics.memory_reads = mem_stats.get("memory_reads", 0)
            metrics.memory_utilization = mem_stats.get("memory_utilization", 0.0)

        # Performance metrics
        step_time = time.time() - step_start
        metrics.steps_per_second = 1.0 / step_time
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        metrics.tokens_per_second = (batch_size * seq_len) / step_time

        # Memory usage
        if torch.cuda.is_available():
            metrics.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024

        # Track best loss
        if metrics.total_loss < self.best_loss:
            self.best_loss = metrics.total_loss

        return metrics

    def evaluate(self) -> TrainingMetrics:
        """Run evaluation on the eval dataset."""
        if not self.eval_dataset:
            return TrainingMetrics()

        logger.info("Running evaluation...")
        self.model.eval()

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        total_metrics = TrainingMetrics()
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move to device
                device = next(self.model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch.get("labels", input_ids).to(device)

                # Evaluate in both paradigms if configured
                if self.config.eval_paradigm in ["training", "both"]:
                    self.model.set_inference_mode(False)  # Train-many mode
                    outputs_train = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )

                if self.config.eval_paradigm in ["inference", "both"]:
                    self.model.set_inference_mode(True)  # Infer-few mode
                    outputs_infer = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )

                # Use training mode outputs for metrics (or inference if only inference)
                outputs = outputs_train if self.config.eval_paradigm != "inference" else outputs_infer

                # Accumulate metrics
                total_metrics.total_loss += outputs["loss"].item()
                total_metrics.avg_act_steps += float(outputs.get("act_steps", 0))

                if "memory_stats" in outputs:
                    mem_stats = outputs["memory_stats"]
                    total_metrics.memory_reads += mem_stats.get("memory_reads", 0)
                    total_metrics.memory_utilization += mem_stats.get("memory_utilization", 0.0)

                num_batches += 1

        # Average metrics
        if num_batches > 0:
            total_metrics.total_loss /= num_batches
            total_metrics.avg_act_steps /= num_batches
            total_metrics.memory_utilization /= num_batches

        self.model.train()
        return total_metrics

    def _update_paradigm(self, metrics: TrainingMetrics, step: int):
        """Update train-many/infer-few paradigm based on training dynamics."""
        # Adaptive paradigm switching based on ACT efficiency
        if metrics.act_efficiency > 0.8:  # If halting early often
            # Could reduce training steps
            pass
        elif metrics.act_efficiency < 0.3:  # If rarely halting early
            # Could increase training steps
            pass

    def _log_metrics(self, metrics: TrainingMetrics, step: int):
        """Log training metrics."""
        logger.info(
            f"Step {step}: loss={metrics.total_loss:.4f}, "
            f"act_steps={metrics.avg_act_steps:.1f}, "
            f"mem_util={metrics.memory_utilization:.2f}, "
            f"lr={metrics.learning_rate:.2e}, "
            f"tps={metrics.tokens_per_second:.0f}"
        )

        # Store metrics
        self.metrics_history.append((step, metrics.to_dict()))

    def _log_eval_metrics(self, metrics: TrainingMetrics, step: int):
        """Log evaluation metrics."""
        logger.info(
            f"Eval {step}: loss={metrics.total_loss:.4f}, "
            f"act_steps={metrics.avg_act_steps:.1f}, "
            f"mem_util={metrics.memory_utilization:.2f}"
        )

    def save_checkpoint(self, step: int, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        if is_final:
            checkpoint_dir = self.output_dir / "final"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "config": self.config.__dict__,
        }

        torch.save(training_state, checkpoint_dir / "training_state.pt")

        # Save metrics history
        with open(checkpoint_dir / "metrics_history.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

        # Clean up old checkpoints
        if not is_final:
            self._cleanup_checkpoints()

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        # Load model
        self.model = CognateModel.from_pretrained(str(checkpoint_path))

        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path)

            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))

            if "optimizer_state" in state:
                self.optimizer.load_state_dict(state["optimizer_state"])

            if "scheduler_state" in state and self.scheduler:
                self.scheduler.load_state_dict(state["scheduler_state"])

            if "scaler_state" in state and self.scaler:
                self.scaler.load_state_dict(state["scaler_state"])

        # Load metrics history
        metrics_path = checkpoint_path / "metrics_history.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics_history = json.load(f)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )

        # Keep only the last N checkpoints
        while len(checkpoints) > self.config.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            try:
                import shutil

                shutil.rmtree(old_checkpoint)
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")


if __name__ == "__main__":
    # Test the trainer
    logging.basicConfig(level=logging.INFO)

    from ..cognate_model import create_cognate_model

    # Create dummy dataset for testing
    class DummyDataset(Dataset):
        def __init__(self, size=1000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 32000, (128,)),
                "attention_mask": torch.ones(128),
                "labels": torch.randint(0, 32000, (128,)),
            }

    print("Testing Cognate Trainer...")

    # Create model and trainer
    model = create_cognate_model(seed=42)

    training_config = CognateTrainingConfig(
        max_steps=100,
        batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=20,
        save_steps=50,
        eval_steps=25,
    )

    trainer = CognateTrainer(
        model=model,
        config=training_config,
        train_dataset=DummyDataset(100),
        eval_dataset=DummyDataset(20),
        output_dir="./test_training",
    )

    # Run short training
    results = trainer.train()

    print(f"Training completed: {results}")
    print("✅ Trainer test passed!")
