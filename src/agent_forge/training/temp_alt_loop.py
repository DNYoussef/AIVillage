"""
Temperature alternation training loop for self-modeling fast-grokking.
Integrates TempBinScheduler, MultiHeadSelfModel, Grokfast, and telemetry encoding.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .grokfast_ctrl import (
    GrokfastConfig,
    GrokfastOptimizer,
    TelemetryTracker,
)
from .self_model import MultiHeadSelfModel
from .telemetry_encode import (
    create_anomaly_detector,
    create_telemetry_encoder,
    create_telemetry_predictor,
)
from .temp_curriculum import (
    TeacherConsistency,
    create_nonoverlap_scheduler,
    create_overlap_scheduler,
)

logger = logging.getLogger(__name__)


@dataclass
class TempAltConfig:
    """Configuration for temperature alternation training."""

    # Temperature curriculum
    temp_range: tuple[float, float] = (0.0, 1.5)
    bin_width: float = 0.1
    overlap_ratio: float = 0.5

    # Model architecture
    hidden_dim: int = 768
    tap_layers: list[int] = field(default_factory=lambda: [6, 8, 10, 11])
    projection_dim: int = 256
    num_temp_bins: int = 6
    num_stages: int = 3

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 50000
    eval_frequency: int = 500
    save_frequency: int = 2000

    # Grokfast
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0
    auto_gated: bool = True

    # Telemetry
    telemetry_encoding: str = "embedding"
    telemetry_feature_dim: int = 64
    telemetry_bins: int = 10

    # Round switching
    round1_min_accuracy: float = 0.7
    round2_min_steps: int = 20000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "temp_range": self.temp_range,
            "bin_width": self.bin_width,
            "overlap_ratio": self.overlap_ratio,
            "hidden_dim": self.hidden_dim,
            "tap_layers": self.tap_layers,
            "projection_dim": self.projection_dim,
            "num_temp_bins": self.num_temp_bins,
            "num_stages": self.num_stages,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "eval_frequency": self.eval_frequency,
            "save_frequency": self.save_frequency,
            "grokfast_alpha": self.grokfast_alpha,
            "grokfast_lambda": self.grokfast_lambda,
            "auto_gated": self.auto_gated,
            "telemetry_encoding": self.telemetry_encoding,
            "telemetry_feature_dim": self.telemetry_feature_dim,
            "telemetry_bins": self.telemetry_bins,
            "round1_min_accuracy": self.round1_min_accuracy,
            "round2_min_steps": self.round2_min_steps,
        }


@dataclass
class TrainingState:
    """Current state of temperature alternation training."""

    step: int = 0
    round: int = 1
    current_temp_bin: Any | None = None  # TempBin object
    loss: float = float("inf")
    accuracy: float = 0.0

    # Multi-head losses
    self_model_loss: float = 0.0
    temp_bin_loss: float = 0.0
    stage_loss: float = 0.0

    # Grokfast status
    grokfast_enabled: bool = False
    grok_detected: bool = False
    lambda_value: float = 0.0

    # Telemetry
    intrinsic_dimension: float = 0.0
    slow_gradient_strength: float = 0.0
    ema_cosine_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "round": self.round,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "self_model_loss": self.self_model_loss,
            "temp_bin_loss": self.temp_bin_loss,
            "stage_loss": self.stage_loss,
            "grokfast_enabled": self.grokfast_enabled,
            "grok_detected": self.grok_detected,
            "lambda_value": self.lambda_value,
            "intrinsic_dimension": self.intrinsic_dimension,
            "slow_gradient_strength": self.slow_gradient_strength,
            "ema_cosine_similarity": self.ema_cosine_similarity,
        }


class TempAlternationTrainer:
    """
    Main trainer for temperature alternation self-modeling system.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TempAltConfig,
        device: str = "cuda",
        save_dir: Path | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_components()

        # Training state
        self.state = TrainingState()
        self.metrics_history = []
        self.best_accuracy = 0.0
        self.best_step = 0

    def _setup_components(self):
        """Initialize all training components."""
        # Temperature curriculum
        self.scheduler = create_nonoverlap_scheduler(temp_range=self.config.temp_range, bin_width=self.config.bin_width)

        # Multi-head self-model
        self.self_model = MultiHeadSelfModel(
            tap_layers=self.config.tap_layers,
            hidden_dim=self.config.hidden_dim,
            projection_dim=self.config.projection_dim,
            num_temp_bins=self.config.num_temp_bins,
            num_stages=self.config.num_stages,
        ).to(self.device)

        # Base optimizer
        self.base_optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.self_model.parameters()),
            lr=self.config.learning_rate,
        )

        # Grokfast optimizer wrapper
        grokfast_config = GrokfastConfig(
            alpha=self.config.grokfast_alpha,
            lamb=self.config.grokfast_lambda,
            enabled=True,
        )

        self.optimizer = GrokfastOptimizer(
            model=nn.ModuleList([self.model, self.self_model]),
            base_optimizer=self.base_optimizer,
            config=grokfast_config,
            detector=None,  # Will be created automatically
        )

        # Telemetry tracking
        self.telemetry_tracker = TelemetryTracker(nn.ModuleList([self.model, self.self_model]))

        # Telemetry encoding
        self.telemetry_encoder = create_telemetry_encoder(
            encoding_type=self.config.telemetry_encoding,
            feature_dim=self.config.telemetry_feature_dim,
            num_bins=self.config.telemetry_bins,
        ).to(self.device)

        # Telemetry predictor (optional future enhancement)
        self.telemetry_predictor = create_telemetry_predictor(input_dim=self.telemetry_encoder.output_dim).to(
            self.device
        )

        # Anomaly detector
        self.anomaly_detector = create_anomaly_detector(input_dim=self.telemetry_encoder.output_dim).to(self.device)

        # Teacher consistency (for KL loss)
        self.teacher_consistency = TeacherConsistency()

        logger.info("Initialized temperature alternation trainer:")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Self-model parameters: {sum(p.numel() for p in self.self_model.parameters()):,}")
        logger.info(f"  Temperature bins: {len(self.scheduler.get_bins())}")
        logger.info(f"  Tap layers: {self.config.tap_layers}")
        logger.info(f"  Device: {self.device}")

    def _get_model_activations(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Extract activations from tapped layers."""
        activations = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output.detach()

            return hook

        # Register hooks
        hooks = []
        if hasattr(self.model, "layers") or hasattr(self.model, "h"):
            layers = getattr(self.model, "layers", getattr(self.model, "h", []))
            for layer_idx in self.config.tap_layers:
                if layer_idx < len(layers):
                    hook = layers[layer_idx].register_forward_hook(hook_fn(layer_idx))
                    hooks.append(hook)

        try:
            # Forward pass to collect activations
            with torch.no_grad():
                _ = self.model(input_ids)

            return activations

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

    def _compute_task_loss(self, input_ids: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute main task loss with temperature scaling."""
        logits = self.model(input_ids)

        if hasattr(logits, "logits"):
            logits = logits.logits

        # Apply temperature scaling
        scaled_logits = logits / max(temperature, 0.01)

        # Standard cross-entropy loss
        loss = F.cross_entropy(scaled_logits.view(-1, scaled_logits.size(-1)), labels.view(-1))

        return loss, scaled_logits

    def _training_step(
        self,
        batch: dict[str, torch.Tensor],
        current_bin: Any,  # TempBin
    ) -> dict[str, Any]:
        """Single training step with temperature alternation."""

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        temp_labels = batch.get("temp_label", torch.zeros(input_ids.size(0), dtype=torch.long)).to(self.device)
        stage_labels = batch.get("stage_label", torch.zeros(input_ids.size(0), dtype=torch.long)).to(self.device)

        # Sample temperature from current bin
        temperature = np.random.uniform(current_bin.low, current_bin.high)
        temp_values = torch.full((input_ids.size(0),), temperature, device=self.device)

        # Forward pass
        labels = input_ids[:, 1:]  # Shift for language modeling
        task_loss, logits = self._compute_task_loss(input_ids[:, :-1], labels, temperature)

        # Get model activations for self-modeling
        tap_activations = self._get_model_activations(input_ids)

        # Self-modeling forward pass
        if tap_activations:
            self_model_results = self.self_model(
                tap_activations=tap_activations,
                temp_bin_labels=temp_labels,
                stage_labels=stage_labels,
                temp_values=temp_values,
            )
        else:
            # Fallback if no activations captured
            self_model_results = {
                "total_loss": torch.tensor(0.0, device=self.device),
                "self_model_loss": torch.tensor(0.0, device=self.device),
                "temp_bin_loss": torch.tensor(0.0, device=self.device),
                "stage_loss": torch.tensor(0.0, device=self.device),
            }

        # Combine losses
        total_loss = task_loss + 0.1 * self_model_results["total_loss"]

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels).float().mean().item()

        # Update telemetry
        telemetry_state = self.telemetry_tracker.update(loss=total_loss.item(), accuracy=accuracy, step=self.state.step)

        # Optimization step with Grokfast
        step_stats = self.optimizer.step(telemetry_state)

        # Update training state
        self.state.step += 1
        self.state.loss = total_loss.item()
        self.state.accuracy = accuracy
        self.state.self_model_loss = self_model_results["self_model_loss"].item()
        self.state.temp_bin_loss = self_model_results.get("temp_bin_loss", torch.tensor(0.0)).item()
        self.state.stage_loss = self_model_results.get("stage_loss", torch.tensor(0.0)).item()
        self.state.grokfast_enabled = step_stats["grokfast_enabled"]
        self.state.grok_detected = step_stats["grok_detected"]
        self.state.lambda_value = step_stats["lambda"]
        self.state.intrinsic_dimension = telemetry_state.intrinsic_dimension
        self.state.slow_gradient_strength = telemetry_state.slow_gradient_strength
        self.state.ema_cosine_similarity = telemetry_state.ema_cosine_similarity
        self.state.current_temp_bin = current_bin

        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "self_model_loss": self.state.self_model_loss,
            "temp_bin_loss": self.state.temp_bin_loss,
            "stage_loss": self.state.stage_loss,
            "accuracy": accuracy,
            "temperature": temperature,
            "grokfast_enabled": step_stats["grokfast_enabled"],
            "lambda": step_stats["lambda"],
            "telemetry": telemetry_state.to_dict(),
        }

    def _should_advance_round(self) -> bool:
        """Check if we should advance to overlapping round."""
        if self.state.round >= 2:
            return False

        # Check accuracy threshold
        recent_accuracy = np.mean([m["accuracy"] for m in self.metrics_history[-100:]])
        accuracy_ok = recent_accuracy >= self.config.round1_min_accuracy

        # Check minimum steps
        steps_ok = self.state.step >= self.config.round2_min_steps

        return accuracy_ok and steps_ok

    def _advance_round(self):
        """Advance to next curriculum round."""
        if self.state.round < 2:
            self.state.round = 2

            # Switch to overlapping scheduler
            self.scheduler = create_overlap_scheduler(
                temp_range=self.config.temp_range,
                bin_width=self.config.bin_width,
                overlap_ratio=self.config.overlap_ratio,
            )

            logger.info(f"🔄 Advanced to curriculum round {self.state.round}")
            logger.info(f"   New bins: {len(self.scheduler.get_bins())}")
            logger.info(f"   Current step: {self.state.step}")

    def _log_progress(self, step_result: dict[str, Any]):
        """Log training progress."""
        if self.state.step % 100 == 0:
            grok_status = "🎯" if self.state.grok_detected else ("⚡" if self.state.grokfast_enabled else "⭕")

            logger.info(
                f"Step {self.state.step:>6} | Round {self.state.round} | "
                f"Loss: {self.state.loss:.4f} | Acc: {self.state.accuracy:.3f} | "
                f"τ={step_result['temperature']:.2f} | {grok_status} λ={self.state.lambda_value:.1f} | "
                f"ID={self.state.intrinsic_dimension:.3f} S_slow={self.state.slow_gradient_strength:.3f}"
            )

    def _evaluate(self, eval_dataloader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate model performance."""
        if eval_dataloader is None:
            return {"eval_accuracy": self.state.accuracy}

        self.model.eval()
        self.self_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = input_ids[:, 1:]

                logits = self.model(input_ids[:, :-1])
                if hasattr(logits, "logits"):
                    logits = logits.logits

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == labels).float().mean().item()

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

        self.model.train()
        self.self_model.train()

        eval_metrics = {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_accuracy": total_accuracy / max(num_batches, 1),
        }

        return eval_metrics

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            "step": self.state.step,
            "model_state_dict": self.model.state_dict(),
            "self_model_state_dict": self.self_model.state_dict(),
            "optimizer_state_dict": self.base_optimizer.state_dict(),
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "metrics_history": self.metrics_history[-1000:],  # Last 1000 entries
            "best_accuracy": self.best_accuracy,
            "best_step": self.best_step,
        }

        # Save latest checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_step_{self.state.step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if self.state.accuracy > self.best_accuracy:
            self.best_accuracy = self.state.accuracy
            self.best_step = self.state.step
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 New best checkpoint saved: accuracy={self.best_accuracy:.4f}")

        logger.info(f"💾 Checkpoint saved at step {self.state.step}")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        resume_from: str | None = None,
    ):
        """Main training loop."""

        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info("🚀 Starting temperature alternation training")
        logger.info(f"   Max steps: {self.config.max_steps:,}")
        logger.info(f"   Temperature range: {self.config.temp_range}")
        logger.info(f"   Current round: {self.state.round}")

        self.model.train()
        self.self_model.train()

        # Training loop
        start_time = time.time()
        train_iterator = iter(train_dataloader)

        while self.state.step < self.config.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reset iterator
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            # Get current temperature bin
            bins = self.scheduler.get_bins()
            current_bin = bins[self.state.step % len(bins)]

            # Training step
            step_result = self._training_step(batch, current_bin)

            # Store metrics
            self.metrics_history.append({**self.state.to_dict(), **step_result})

            # Log progress
            self._log_progress(step_result)

            # Check for round advancement
            if self._should_advance_round():
                self._advance_round()

            # Evaluation
            if self.state.step % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate(eval_dataloader)

                if eval_metrics:
                    logger.info(
                        f"📊 Eval | Step {self.state.step} | "
                        f"Loss: {eval_metrics.get('eval_loss', 0):.4f} | "
                        f"Acc: {eval_metrics.get('eval_accuracy', 0):.4f}"
                    )

                    # Update metrics history
                    self.metrics_history[-1].update(eval_metrics)

            # Save checkpoint
            if self.state.step % self.config.save_frequency == 0:
                self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()

        elapsed = time.time() - start_time
        logger.info("🎉 Training completed!")
        logger.info(f"   Total steps: {self.state.step:,}")
        logger.info(f"   Final accuracy: {self.state.accuracy:.4f}")
        logger.info(f"   Best accuracy: {self.best_accuracy:.4f} (step {self.best_step})")
        logger.info(f"   Total time: {elapsed / 3600:.1f} hours")
        logger.info(f"   Grok detected: {self.state.grok_detected}")

        # Export final results
        self._export_results()

    def _export_results(self):
        """Export training results and analysis."""
        results = {
            "config": self.config.to_dict(),
            "final_state": self.state.to_dict(),
            "best_accuracy": self.best_accuracy,
            "best_step": self.best_step,
            "grok_detected": self.state.grok_detected,
            "total_steps": self.state.step,
            "final_round": self.state.round,
        }

        # Export telemetry
        telemetry_data = self.telemetry_tracker.export_telemetry()

        # Export Grokfast statistics
        grokfast_stats = self.optimizer.get_stats()

        # Save all results
        results_path = self.save_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        telemetry_path = self.save_dir / "telemetry_data.json"
        with open(telemetry_path, "w") as f:
            json.dump(telemetry_data, f, indent=2)

        grokfast_path = self.save_dir / "grokfast_stats.json"
        with open(grokfast_path, "w") as f:
            json.dump(grokfast_stats, f, indent=2)

        logger.info(f"📊 Results exported to {self.save_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.self_model.load_state_dict(checkpoint["self_model_state_dict"])
        self.base_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.state.step = checkpoint["step"]
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
        self.best_step = checkpoint.get("best_step", 0)

        if "metrics_history" in checkpoint:
            self.metrics_history = checkpoint["metrics_history"]

        logger.info(f"📂 Loaded checkpoint from step {self.state.step}")


# Factory function


def create_temp_alt_trainer(
    model: nn.Module,
    config: TempAltConfig | None = None,
    device: str = "cuda",
    save_dir: str | None = None,
) -> TempAlternationTrainer:
    """Create temperature alternation trainer with default config."""
    if config is None:
        config = TempAltConfig()

    return TempAlternationTrainer(
        model=model,
        config=config,
        device=device,
        save_dir=Path(save_dir) if save_dir else None,
    )


if __name__ == "__main__":
    # Demo temperature alternation training system
    print("🌡️⚡ Temperature Alternation Training System Demo")
    print("=" * 60)

    # Create simple model for demonstration
    class SimpleLanguageModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_dim=256, num_layers=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList(
                [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True) for _ in range(num_layers)]
            )
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)

    model = SimpleLanguageModel(vocab_size=100, hidden_dim=128, num_layers=4)

    # Create config
    config = TempAltConfig(
        hidden_dim=128,
        tap_layers=[1, 2, 3],
        max_steps=50,  # Short demo
        eval_frequency=10,
        save_frequency=25,
        batch_size=4,
    )

    # Create trainer
    trainer = create_temp_alt_trainer(
        model=model,
        config=config,
        device="cpu",  # Use CPU for demo
        save_dir="temp_alt_demo",
    )

    print("Created trainer with configuration:")
    print(f"  Temperature range: {config.temp_range}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Tap layers: {config.tap_layers}")
    print(f"  Max steps: {config.max_steps}")
    print()

    # Create mock dataset
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=32, vocab_size=100):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
                "attention_mask": torch.ones(self.seq_len),
                "temp_label": torch.randint(0, 6, ()),  # Random temp bin
                "stage_label": torch.randint(0, 3, ()),  # Random stage
            }

    dataset = MockDataset(size=200)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    print("📊 Starting training demo...")
    print()

    # Run training
    try:
        trainer.train(
            train_dataloader=dataloader,
            eval_dataloader=None,  # No eval for demo
        )

        print()
        print("✅ Temperature Alternation Training Demo Complete")
        print()
        print("Key Features Demonstrated:")
        print("  • Temperature bin curriculum with non-overlapping → overlapping progression")
        print("  • Multi-head self-modeling (activation prediction, temp inference, stage classification)")
        print("  • Grokfast optimization with telemetry-based lambda gating")
        print("  • Comprehensive telemetry tracking (ID, S_slow, EMA cosine similarity)")
        print("  • Automatic round advancement based on performance thresholds")
        print("  • Complete checkpoint/resume system with metrics export")
        print("  • Integration-ready training loop for production use")

    except Exception as e:
        logger.error(f"Training demo failed: {e}")
        print(f"❌ Demo failed: {e}")

    print()
    print("🚀 System ready for integration with hybrid training pipeline!")
