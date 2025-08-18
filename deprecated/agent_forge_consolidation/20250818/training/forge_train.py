# isort: skip_file
"""
Main Forge Training Loop that integrates all components.
This is the entry point for the complete training pipeline.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dream import DreamAugmenter, DreamBuffer, DreamCycleManager, DreamExample
from .edge import EdgeController
from .geometry import GeometryProbe
from .grok import GrokController, GrokfastScheduler
from .grokfast_optimizer import GrokfastAdamW
from .self_model import SelfModelHead, StageClassifier, TempCurriculum
from .telemetry import TelemetryFrame, TelemetryLogger

logger = logging.getLogger(__name__)


@dataclass
class ForgeTrainConfig:
    """Configuration for Forge training loop."""

    # Model configuration
    model_name: str = "gpt2"
    tap_layers: list[int] = field(default_factory=lambda: [4, 8, 12])
    hidden_dim: int = 768

    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_steps: int = 100000
    warmup_steps: int = 1000

    # Grokfast configuration
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda_init: float = 0.05
    grokfast_lambda_max: float = 0.25

    # Edge-of-chaos configuration
    enable_edge_control: bool = True
    target_success_range: tuple[float, float] = (0.55, 0.75)

    # Self-modeling configuration
    enable_self_model: bool = True
    self_model_weight: float = 0.1
    self_model_warmup: int = 5000

    # Temperature curriculum
    enable_temp_curriculum: bool = True
    temp_curriculum_interval: int = 2000

    # Dream/sleep configuration
    enable_dream_cycles: bool = True
    dream_cycle_interval: int = 1000
    dream_duration: int = 50
    dream_buffer_capacity: int = 10000

    # Geometry probing
    geometry_probe_interval: int = 100
    geometry_layers: list[int] = field(default_factory=lambda: [4, 8, 12])

    # Task/loss configuration
    task_type: str = "classification"  # classification, regression, seq2seq
    custom_loss_fn: Callable[[Any, dict[str, torch.Tensor]], torch.Tensor] | None = None

    # Logging configuration
    log_interval: int = 10
    checkpoint_interval: int = 1000
    wandb_project: str | None = "forge-train"

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("./forge_output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./forge_checkpoints"))

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class ForgeTrainer:
    """
    Main trainer that orchestrates the complete Forge training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset,
        config: ForgeTrainConfig,
        tokenizer=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize components
        self._init_components()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_performance = 0.0

    def _init_components(self):
        """Initialize all training components."""

        # Telemetry
        self.telemetry_logger = TelemetryLogger(
            wandb_project=self.config.wandb_project,
            log_interval=self.config.log_interval,
        )

        # Edge controller
        if self.config.enable_edge_control:
            self.edge_controller = EdgeController(
                target_range=self.config.target_success_range, window_size=100
            )
        else:
            self.edge_controller = None

        # Geometry probe
        self.geometry_probe = GeometryProbe(
            layer_ids=self.config.geometry_layers, sample_size=1024
        )

        # Grokfast controller
        if self.config.enable_grokfast:
            self.grok_controller = GrokController(
                ema_alpha=self.config.grokfast_ema_alpha,
                lam_init=self.config.grokfast_lambda_init,
                lam_max=self.config.grokfast_lambda_max,
            )
            self.grok_scheduler = GrokfastScheduler(
                controller=self.grok_controller, warmup_steps=self.config.warmup_steps
            )
        else:
            self.grok_controller = None
            self.grok_scheduler = None

        # Optimizer with Grokfast
        self.optimizer = GrokfastAdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            ema_alpha=self.config.grokfast_ema_alpha,
            grokfast_lambda=self.config.grokfast_lambda_init,
            grokfast_enabled=self.config.enable_grokfast,
        )

        # Self-modeling head
        if self.config.enable_self_model:
            self.self_model_head = SelfModelHead(
                tap_layers=self.config.tap_layers, hidden_dim=self.config.hidden_dim
            ).to(self.device)

            # Separate optimizer for self-model head
            self.self_model_optimizer = torch.optim.Adam(
                self.self_model_head.parameters(), lr=self.config.learning_rate
            )
        else:
            self.self_model_head = None
            self.self_model_optimizer = None

        # Temperature curriculum
        if self.config.enable_temp_curriculum:
            self.temp_curriculum = TempCurriculum()
        else:
            self.temp_curriculum = None

        # Dream components
        if self.config.enable_dream_cycles:
            self.dream_buffer = DreamBuffer(
                capacity=self.config.dream_buffer_capacity,
                save_path=self.config.output_dir / "dream_buffer.json",
            )
            self.dream_augmenter = DreamAugmenter()
            self.dream_manager = DreamCycleManager(
                dream_buffer=self.dream_buffer,
                augmenter=self.dream_augmenter,
                cycle_interval=self.config.dream_cycle_interval,
                dream_duration=self.config.dream_duration,
            )
        else:
            self.dream_buffer = None
            self.dream_manager = None

        # Stage classifier
        self.stage_classifier = StageClassifier()

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def train(self):
        """Main training loop."""
        logger.info("Starting Forge training loop")

        for epoch in range(100):  # Max epochs
            self.epoch = epoch
            self.train_epoch()

            # Evaluation
            if self.global_step % self.config.checkpoint_interval == 0:
                metrics = self.evaluate()
                self.save_checkpoint(metrics)

            # Check for convergence
            if self.global_step >= self.config.max_steps:
                break

        logger.info("Training complete")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        for _batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Check for dream cycle
            if self.dream_manager and self.dream_manager.should_dream(self.global_step):
                self.run_dream_cycle()

            # Regular training step
            metrics = self.training_step(batch)

            # Update difficulty if using edge control
            if self.edge_controller and "accuracy" in metrics:
                self.edge_controller.update([metrics["accuracy"]])
                # Apply difficulty adjustments to next batch
                # This would modify task sampling or generation

            # Temperature curriculum check
            if (
                self.temp_curriculum
                and self.global_step % self.config.temp_curriculum_interval == 0
            ):
                self.run_temperature_curriculum()

            self.global_step += 1

            # Break if max steps reached
            if self.global_step >= self.config.max_steps:
                break

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        start_time = time.time()

        # Forward pass with activation collection
        outputs, activations = self.forward_with_activations(batch)

        # Compute main task loss
        task_loss = self.compute_task_loss(outputs, batch)

        # Self-modeling loss
        self_model_loss = torch.tensor(0.0).to(self.device)
        if self.self_model_head and self.global_step > self.config.self_model_warmup:
            _, self_model_loss = self.self_model_head(activations)
            self_model_loss = self_model_loss * self.config.self_model_weight

        # Total loss
        total_loss = task_loss + self_model_loss

        # Compute gradients
        total_loss.backward()

        # Get gradient statistics
        grad_norm = self.compute_grad_norm()

        # Compute telemetry
        telemetry = self.compute_telemetry(
            loss=task_loss.item(),
            activations=activations,
            grad_norm=grad_norm,
            batch=batch,
        )

        # Grokfast control
        grokfast_lambda = 0.0
        if self.grok_scheduler:
            # Get aggregated gradient for Grokfast
            grad = self.get_aggregated_gradient()
            grok_result = self.grok_scheduler.step(grad, telemetry)
            grokfast_lambda = grok_result["lambda"]

            # Update optimizer lambda
            self.optimizer.set_grokfast_lambda(grokfast_lambda)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Self-model optimizer step
        if self.self_model_optimizer and self_model_loss > 0:
            self.self_model_optimizer.step()
            self.self_model_optimizer.zero_grad()

        # Update telemetry with final values
        telemetry.grokfast_lambda = grokfast_lambda
        telemetry.latency_ms = (time.time() - start_time) * 1000

        # Log telemetry
        self.telemetry_logger.log(telemetry)

        # Store example in dream buffer
        if self.dream_buffer and telemetry.pass_at_1 > 0:
            self.store_dream_example(batch, outputs, telemetry)

        return {
            "loss": task_loss.item(),
            "accuracy": telemetry.pass_at_1,
            "grad_norm": grad_norm,
            "grokfast_lambda": grokfast_lambda,
        }

    def forward_with_activations(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[Any, dict[int, torch.Tensor]]:
        """Forward pass with activation collection."""
        activations = {}

        # Register hooks to collect activations
        handles = []
        for layer_idx in self.config.tap_layers:

            def hook(module, input, output, layer_id=layer_idx):
                activations[layer_id] = output.detach()

            # This assumes transformer architecture
            # Adjust based on actual model structure
            if hasattr(self.model, "transformer"):
                layer = self.model.transformer.h[layer_idx]
            elif hasattr(self.model, "layers"):
                layer = self.model.layers[layer_idx]
            else:
                # Fallback - try to find layers
                layer = list(self.model.modules())[layer_idx]

            handle = layer.register_forward_hook(hook)
            handles.append(handle)

        # Forward pass
        outputs = self.model(**batch)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return outputs, activations

    def compute_task_loss(self, outputs, batch) -> torch.Tensor:
        """Compute the main task loss."""
        # Custom loss override
        if self.config.custom_loss_fn is not None:
            return self.config.custom_loss_fn(outputs, batch)

        if hasattr(outputs, "loss"):
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        task_type = getattr(self.config, "task_type", "classification")

        if task_type == "regression":
            if "labels" not in batch:
                raise ValueError("Regression task requires 'labels' in batch")
            preds = logits.squeeze(-1)
            targets = batch["labels"].float()
            return F.mse_loss(preds.view_as(targets), targets)

        if task_type in {"seq2seq", "generation"}:
            if "labels" not in batch:
                raise ValueError("Seq2seq task requires 'labels' in batch")
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch["labels"].reshape(-1),
                ignore_index=-100,
            )

        if "labels" in batch:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1)
            )

        raise NotImplementedError("Task loss computation not implemented")

    def compute_grad_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def get_aggregated_gradient(self) -> torch.Tensor:
        """Get aggregated gradient tensor for Grokfast."""
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())

        if grads:
            return torch.cat(grads)
        else:
            return torch.zeros(1).to(self.device)

    def compute_telemetry(
        self,
        loss: float,
        activations: dict[int, torch.Tensor],
        grad_norm: float,
        batch: dict[str, torch.Tensor],
    ) -> TelemetryFrame:
        """Compute comprehensive telemetry."""

        # Compute geometry metrics
        id_by_layer, d_by_layer = {}, {}
        if self.global_step % self.config.geometry_probe_interval == 0:
            id_by_layer, d_by_layer = self.geometry_probe.compute(activations)

        # Compute accuracy (task-specific)
        pass_at_1 = self.compute_accuracy(batch)

        # Get EMA statistics from telemetry logger
        ema_grad_norm = self.telemetry_logger.ema_grad_norm or grad_norm
        ema_cos = 0.0
        if self.grok_controller:
            stats = self.grok_controller.get_statistics()
            ema_cos = stats.get("avg_cos_sim", 0.0)

        # Classify stage
        stage = self.stage_classifier.classify(
            loss=loss,
            accuracy=pass_at_1,
            grad_norm=grad_norm,
            id_value=np.mean(list(id_by_layer.values())) if id_by_layer else 0.0,
            ema_cos=ema_cos,
            step=self.global_step,
        )

        return TelemetryFrame(
            step=self.global_step,
            loss=loss,
            pass_at_1=pass_at_1,
            grad_norm=grad_norm,
            ema_grad_norm=ema_grad_norm,
            ema_cos=ema_cos,
            id_by_layer=id_by_layer,
            d_by_layer=d_by_layer,
            temperature=1.0,  # Default, updated during temp curriculum
            stage=stage,
        )

    def compute_accuracy(self, batch: dict[str, torch.Tensor]) -> float:
        """Compute accuracy for the batch."""
        # This is task-specific - implement based on your task
        # For now, return a placeholder
        return np.random.random() * 0.5 + 0.25  # Placeholder

    def store_dream_example(
        self, batch: dict[str, torch.Tensor], outputs, telemetry: TelemetryFrame
    ):
        """Store example in dream buffer."""
        # Extract first example from batch
        # This is task-specific - adjust based on your data format

        example = DreamExample(
            prompt="",  # Extract from batch
            target="",  # Extract from batch
            model_output="",  # Extract from outputs
            loss=telemetry.loss,
            accuracy=telemetry.pass_at_1,
            confidence=0.5,  # Compute from outputs
            temperature=telemetry.temperature,
            step=telemetry.step,
            stage=telemetry.stage,
            grad_norm=telemetry.grad_norm,
            ema_cos=telemetry.ema_cos,
            id_value=(
                np.mean(list(telemetry.id_by_layer.values()))
                if telemetry.id_by_layer
                else 0.0
            ),
        )

        self.dream_buffer.push(example)

    def run_dream_cycle(self):
        """Run a complete dream cycle."""
        logger.info(f"Starting dream cycle at step {self.global_step}")
        self.dream_manager.start_dream_cycle()

        while self.dream_manager.is_dreaming:
            # Get dream batch
            dream_examples = self.dream_manager.get_dream_batch(
                stage=(
                    self.stage_classifier.stage_history[-1]
                    if self.stage_classifier.stage_history
                    else None
                )
            )

            if not dream_examples:
                break

            # Convert to batch format
            # This is task-specific - implement based on your data format
            dream_batch = self.examples_to_batch(dream_examples)

            # Train on dream batch
            self.training_step(dream_batch)

        logger.info("Dream cycle complete")

    def examples_to_batch(
        self, examples: list[DreamExample]
    ) -> dict[str, torch.Tensor]:
        """Convert dream examples to batch format."""
        # This is task-specific - implement based on your data format
        # For now, return empty batch
        return {}

    def run_temperature_curriculum(self):
        """Run temperature curriculum training."""
        if not self.temp_curriculum or not self.tokenizer:
            return

        logger.info(f"Running temperature curriculum at step {self.global_step}")

        # Generate samples across temperature bins
        prompts = [
            "Write a function to",
            "Implement",
            "Create a class",
        ]  # Task-specific
        samples = self.temp_curriculum.generate_samples(
            self.model, self.tokenizer, prompts, self.device
        )

        # Train on temperature samples
        for _sample in samples:
            # Convert to batch format and train
            # This is task-specific
            pass

        # Check if should advance round
        if self.temp_curriculum.should_advance_round():
            self.temp_curriculum.advance_round()

    def evaluate(self) -> dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs, _ = self.forward_with_activations(batch)
                loss = self.compute_task_loss(outputs, batch)
                accuracy = self.compute_accuracy(batch)

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

        self.model.train()

        metrics = {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches,
        }

        logger.info(f"Evaluation at step {self.global_step}: {metrics}")

        return metrics

    def save_checkpoint(self, metrics: dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        if self.self_model_head:
            checkpoint["self_model_state_dict"] = self.self_model_head.state_dict()
            checkpoint["self_model_optimizer_state_dict"] = (
                self.self_model_optimizer.state_dict()
            )

        checkpoint_path = (
            self.config.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if metrics.get("eval_accuracy", 0) > self.best_performance:
            self.best_performance = metrics["eval_accuracy"]
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved to {best_path}")


def create_forge_trainer(
    model: nn.Module,
    train_dataset,
    eval_dataset,
    config: ForgeTrainConfig | None = None,
    **kwargs,
) -> ForgeTrainer:
    """Factory function to create a Forge trainer."""
    if config is None:
        config = ForgeTrainConfig(**kwargs)

    return ForgeTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
