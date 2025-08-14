"""
Telemetry frame for tracking training metrics and states.
Part of the Forge Train Loop MVP.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class TelemetryFrame:
    """Comprehensive telemetry snapshot for training monitoring."""

    step: int
    loss: float
    pass_at_1: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)

    # Gradient statistics
    grad_norm: float = 0.0
    ema_grad_norm: float = 0.0
    ema_cos: float = 0.0  # Cosine similarity between grad and EMA grad

    # Geometry metrics (ID/d per layer)
    id_by_layer: dict[int, float] = field(default_factory=dict)
    d_by_layer: dict[int, float] = field(default_factory=dict)

    # Training dynamics
    temperature: float = 1.0
    learning_rate: float = 0.0
    grokfast_lambda: float = 0.0

    # Performance metrics
    latency_ms: float = 0.0
    vram_gb: float = 0.0
    tokens_per_second: float = 0.0

    # Stage classification
    stage: str = "pre_grok"  # pre_grok, onset, consolidation, post_grok

    # Edge-of-chaos metrics
    success_rate: float = 0.0
    complexity_score: float = 0.0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryLogger:
    """Centralized telemetry logging with W&B integration."""

    def __init__(self, wandb_project: str | None = None, log_interval: int = 10):
        self.wandb_project = wandb_project
        self.log_interval = log_interval
        self.frames = []
        self.ema_alpha = 0.98
        self.ema_grad_norm = None
        self.ema_grad = None

        if wandb_project:
            try:
                import wandb

                self.wandb = wandb
                if not wandb.run:
                    wandb.init(project=wandb_project, name="forge_train")
            except ImportError:
                print("W&B not installed, logging to memory only")
                self.wandb = None
        else:
            self.wandb = None

    def update_ema(self, grad_norm: float, grad: torch.Tensor | None = None):
        """Update exponential moving averages for gradient statistics."""
        if self.ema_grad_norm is None:
            self.ema_grad_norm = grad_norm
        else:
            self.ema_grad_norm = (
                self.ema_alpha * self.ema_grad_norm + (1 - self.ema_alpha) * grad_norm
            )

        if grad is not None:
            if self.ema_grad is None:
                self.ema_grad = grad.clone()
            else:
                self.ema_grad = (
                    self.ema_alpha * self.ema_grad + (1 - self.ema_alpha) * grad
                )

        return self.ema_grad_norm

    def compute_ema_cos(self, grad: torch.Tensor) -> float:
        """Compute cosine similarity between gradient and EMA gradient."""
        if self.ema_grad is None:
            return 0.0

        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(
                grad.flatten().unsqueeze(0), self.ema_grad.flatten().unsqueeze(0)
            ).item()

        return cos_sim

    def classify_stage(self, frame: TelemetryFrame) -> str:
        """Classify training stage based on telemetry signals."""
        # Simple heuristic - can be made more sophisticated
        if frame.ema_cos < 0.3 and frame.loss > 1.0:
            return "pre_grok"
        elif 0.3 <= frame.ema_cos < 0.7 and frame.grad_norm > frame.ema_grad_norm * 1.5:
            return "onset"
        elif frame.ema_cos >= 0.7 and abs(frame.grad_norm - frame.ema_grad_norm) < 0.1:
            return "consolidation"
        else:
            return "post_grok"

    def log(self, frame: TelemetryFrame):
        """Log telemetry frame to memory and optionally W&B."""
        frame.stage = self.classify_stage(frame)
        self.frames.append(frame)

        if len(self.frames) % self.log_interval == 0:
            if self.wandb and self.wandb.run:
                metrics = {
                    "loss": frame.loss,
                    "pass_at_1": frame.pass_at_1,
                    "grad_norm": frame.grad_norm,
                    "ema_grad_norm": frame.ema_grad_norm,
                    "ema_cos": frame.ema_cos,
                    "temperature": frame.temperature,
                    "learning_rate": frame.learning_rate,
                    "grokfast_lambda": frame.grokfast_lambda,
                    "latency_ms": frame.latency_ms,
                    "vram_gb": frame.vram_gb,
                    "stage": frame.stage,
                    "success_rate": frame.success_rate,
                    "complexity_score": frame.complexity_score,
                }

                # Add geometry metrics
                for layer, id_val in frame.id_by_layer.items():
                    metrics[f"id_layer_{layer}"] = id_val
                for layer, d_val in frame.d_by_layer.items():
                    metrics[f"d_layer_{layer}"] = d_val

                self.wandb.log(metrics, step=frame.step)

    def get_recent_frames(self, n: int = 10) -> list:
        """Get the most recent n telemetry frames."""
        return self.frames[-n:] if len(self.frames) >= n else self.frames

    def compute_statistics(self) -> dict:
        """Compute aggregate statistics over all frames."""
        if not self.frames:
            return {}

        stats = {
            "total_steps": len(self.frames),
            "avg_loss": np.mean([f.loss for f in self.frames]),
            "avg_pass_at_1": np.mean([f.pass_at_1 for f in self.frames]),
            "avg_success_rate": np.mean([f.success_rate for f in self.frames]),
            "stage_distribution": {},
        }

        # Count stage distribution
        for frame in self.frames:
            stats["stage_distribution"][frame.stage] = (
                stats["stage_distribution"].get(frame.stage, 0) + 1
            )

        return stats
