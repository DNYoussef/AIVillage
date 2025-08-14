"""
Grokfast controller with telemetry-based gating.
Implements lambda gating based on geometry (ID↓) + slow gradient (S_slow↑) signals.
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GrokfastMode(Enum):
    """Grokfast operation modes."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO_GATED = "auto_gated"  # Gate based on telemetry


@dataclass
class TelemetryState:
    """Current telemetry measurements."""

    intrinsic_dimension: float = 0.0
    slow_gradient_strength: float = 0.0
    ema_cosine_similarity: float = 0.0
    loss: float = float("inf")
    accuracy: float = 0.0
    grad_norm: float = 0.0
    step: int = 0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "id": self.intrinsic_dimension,
            "s_slow": self.slow_gradient_strength,
            "ema_cos": self.ema_cosine_similarity,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "grad_norm": self.grad_norm,
            "step": self.step,
        }


@dataclass
class GrokfastConfig:
    """Configuration for Grokfast optimizer."""

    alpha: float = 0.98  # EMA decay for slow gradient
    lamb: float = 2.0  # Grokfast amplification factor
    enabled: bool = True

    # Gating thresholds
    id_threshold: float = 0.95  # ID must be < this to enable
    s_slow_threshold: float = 0.1  # S_slow must be > this to enable
    min_steps: int = 1000  # Minimum steps before gating

    # Lambda scheduling
    lambda_schedule: str = "constant"  # "constant", "linear", "exponential"
    lambda_min: float = 1.0
    lambda_max: float = 5.0
    lambda_warmup_steps: int = 5000


class GrokSignalDetector:
    """
    Detects grok onset signals: ID↓ + S_slow↑
    """

    def __init__(
        self,
        id_window: int = 100,
        s_slow_window: int = 50,
        detection_threshold: float = 0.7,
    ):
        self.id_window = id_window
        self.s_slow_window = s_slow_window
        self.detection_threshold = detection_threshold

        # History buffers
        self.id_history = deque(maxlen=id_window)
        self.s_slow_history = deque(maxlen=s_slow_window)

        # Detection state
        self.grok_detected = False
        self.detection_step = -1
        self.confidence_score = 0.0

    def update(self, telemetry: TelemetryState) -> bool:
        """
        Update with new telemetry and detect grok onset.

        Returns:
            True if grok onset detected
        """
        self.id_history.append(telemetry.intrinsic_dimension)
        self.s_slow_history.append(telemetry.slow_gradient_strength)

        # Need sufficient history for detection
        if len(self.id_history) < 20 or len(self.s_slow_history) < 20:
            return False

        # Check ID trend (should be decreasing)
        recent_id = list(self.id_history)[-20:]
        id_trend = np.polyfit(range(20), recent_id, 1)[0]
        id_decreasing = id_trend < -0.001

        # Check S_slow trend (should be increasing)
        recent_s_slow = list(self.s_slow_history)[-20:]
        s_slow_trend = np.polyfit(range(20), recent_s_slow, 1)[0]
        s_slow_increasing = s_slow_trend > 0.0001

        # Current values check
        current_id = telemetry.intrinsic_dimension
        current_s_slow = telemetry.slow_gradient_strength

        # Compute confidence
        id_confidence = 1.0 if (id_decreasing and current_id < 0.95) else 0.0
        s_slow_confidence = 1.0 if (s_slow_increasing and current_s_slow > 0.1) else 0.0

        # Combined confidence
        self.confidence_score = (id_confidence + s_slow_confidence) / 2.0

        # Detection logic
        if not self.grok_detected and self.confidence_score >= self.detection_threshold:
            self.grok_detected = True
            self.detection_step = telemetry.step
            logger.info(
                f"🎯 Grok onset detected at step {telemetry.step}! "
                f"ID={current_id:.3f}, S_slow={current_s_slow:.3f}, "
                f"confidence={self.confidence_score:.2f}"
            )
            return True

        return self.grok_detected

    def get_gating_decision(
        self, telemetry: TelemetryState, config: GrokfastConfig
    ) -> bool:
        """
        Make gating decision based on current telemetry.

        Returns:
            True if Grokfast should be enabled
        """
        # Must have minimum steps
        if telemetry.step < config.min_steps:
            return False

        # Check thresholds
        id_condition = telemetry.intrinsic_dimension < config.id_threshold
        s_slow_condition = telemetry.slow_gradient_strength > config.s_slow_threshold

        return id_condition and s_slow_condition

    def reset(self):
        """Reset detection state."""
        self.grok_detected = False
        self.detection_step = -1
        self.confidence_score = 0.0
        self.id_history.clear()
        self.s_slow_history.clear()


class GrokfastOptimizer:
    """
    Grokfast-enhanced optimizer that amplifies slow gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        config: GrokfastConfig,
        detector: GrokSignalDetector | None = None,
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.config = config
        self.detector = detector or GrokSignalDetector()

        # Slow gradient EMA
        self.slow_grads = {}
        self.initialized = False

        # Statistics
        self.stats = {
            "total_steps": 0,
            "grokfast_steps": 0,
            "lambda_values": [],
            "telemetry_history": [],
            "gating_decisions": [],
        }

    def _initialize_slow_grads(self):
        """Initialize slow gradient buffers."""
        if self.initialized:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.slow_grads[name] = torch.zeros_like(param.grad)

        self.initialized = True
        logger.info(
            f"Initialized Grokfast slow gradients for {len(self.slow_grads)} parameters"
        )

    def _update_slow_grads(self):
        """Update slow gradient EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.slow_grads:
                    self.slow_grads[name] = torch.zeros_like(param.grad)

                # EMA update: slow_grad = α * slow_grad + (1-α) * grad
                self.slow_grads[name] = (
                    self.config.alpha * self.slow_grads[name]
                    + (1 - self.config.alpha) * param.grad
                )

    def _compute_lambda(self, step: int) -> float:
        """Compute lambda value based on schedule."""
        if self.config.lambda_schedule == "constant":
            return self.config.lamb

        elif self.config.lambda_schedule == "linear":
            if step < self.config.lambda_warmup_steps:
                progress = step / self.config.lambda_warmup_steps
                return self.config.lambda_min + progress * (
                    self.config.lambda_max - self.config.lambda_min
                )
            return self.config.lambda_max

        elif self.config.lambda_schedule == "exponential":
            if step < self.config.lambda_warmup_steps:
                progress = step / self.config.lambda_warmup_steps
                return self.config.lambda_min * (
                    (self.config.lambda_max / self.config.lambda_min) ** progress
                )
            return self.config.lambda_max

        return self.config.lamb

    def step(self, telemetry: TelemetryState) -> dict[str, Any]:
        """
        Perform optimization step with optional Grokfast enhancement.

        Args:
            telemetry: Current telemetry measurements

        Returns:
            Step statistics
        """
        self._initialize_slow_grads()
        self.stats["total_steps"] += 1

        # Update detector
        grok_detected = self.detector.update(telemetry)

        # Make gating decision
        if self.config.enabled:
            if self.detector:
                grokfast_enabled = self.detector.get_gating_decision(
                    telemetry, self.config
                )
            else:
                grokfast_enabled = True
        else:
            grokfast_enabled = False

        step_stats = {
            "step": telemetry.step,
            "grokfast_enabled": grokfast_enabled,
            "grok_detected": grok_detected,
            "confidence": self.detector.confidence_score,
            "lambda": 0.0,
            "slow_grad_norm": 0.0,
            "fast_grad_norm": 0.0,
        }

        # Update slow gradients
        self._update_slow_grads()

        if grokfast_enabled:
            # Apply Grokfast: grad = grad + λ * slow_grad
            lambda_val = self._compute_lambda(telemetry.step)
            slow_grad_norm = 0.0
            fast_grad_norm = 0.0

            for name, param in self.model.named_parameters():
                if (
                    param.requires_grad
                    and param.grad is not None
                    and name in self.slow_grads
                ):
                    slow_grad = self.slow_grads[name]
                    fast_grad = param.grad - slow_grad  # Fast component

                    # Track norms
                    slow_grad_norm += slow_grad.norm().item() ** 2
                    fast_grad_norm += fast_grad.norm().item() ** 2

                    # Apply Grokfast update
                    param.grad = param.grad + lambda_val * slow_grad

            slow_grad_norm = np.sqrt(slow_grad_norm)
            fast_grad_norm = np.sqrt(fast_grad_norm)

            step_stats.update(
                {
                    "lambda": lambda_val,
                    "slow_grad_norm": slow_grad_norm,
                    "fast_grad_norm": fast_grad_norm,
                }
            )

            self.stats["grokfast_steps"] += 1
            self.stats["lambda_values"].append(lambda_val)

        # Perform base optimization step
        self.base_optimizer.step()

        # Store statistics
        self.stats["telemetry_history"].append(telemetry.to_dict())
        self.stats["gating_decisions"].append(grokfast_enabled)

        return step_stats

    def zero_grad(self):
        """Zero gradients."""
        self.base_optimizer.zero_grad()

    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        total_steps = self.stats["total_steps"]
        grokfast_steps = self.stats["grokfast_steps"]

        stats = {
            "total_steps": total_steps,
            "grokfast_steps": grokfast_steps,
            "grokfast_ratio": grokfast_steps / total_steps if total_steps > 0 else 0,
            "avg_lambda": np.mean(self.stats["lambda_values"])
            if self.stats["lambda_values"]
            else 0,
            "grok_detected": self.detector.grok_detected,
            "detection_step": self.detector.detection_step,
            "current_confidence": self.detector.confidence_score,
        }

        return stats


class TelemetryTracker:
    """
    Tracks training telemetry for grok detection.
    """

    def __init__(self, model: nn.Module, history_size: int = 1000):
        self.model = model
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

        # Running statistics
        self.grad_ema = {}
        self.param_ema = {}
        self.ema_decay = 0.99

    def compute_intrinsic_dimension(self, gradients: dict[str, torch.Tensor]) -> float:
        """
        Compute intrinsic dimension using gradient covariance.
        Approximation: ID ≈ tr(Cov) / ||Cov||_F
        """
        if not gradients:
            return 1.0

        # Flatten all gradients
        all_grads = []
        for grad in gradients.values():
            if grad is not None:
                all_grads.append(grad.flatten())

        if not all_grads:
            return 1.0

        grad_vec = torch.cat(all_grads)

        # Simple approximation using gradient norm ratio
        grad_norm = grad_vec.norm().item()
        grad_std = grad_vec.std().item()

        if grad_norm < 1e-8:
            return 1.0

        # Normalized complexity measure
        id_estimate = min(grad_std / (grad_norm / grad_vec.numel()), 1.0)
        return id_estimate

    def compute_slow_gradient_strength(self) -> float:
        """
        Compute slow gradient strength as EMA of gradient norms.
        """
        total_slow_norm = 0.0
        total_fast_norm = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad

                # Update EMA
                if name not in self.grad_ema:
                    self.grad_ema[name] = torch.zeros_like(grad)

                self.grad_ema[name] = (
                    self.ema_decay * self.grad_ema[name] + (1 - self.ema_decay) * grad
                )

                slow_norm = self.grad_ema[name].norm().item()
                fast_norm = (grad - self.grad_ema[name]).norm().item()

                total_slow_norm += slow_norm**2
                total_fast_norm += fast_norm**2

        total_slow_norm = np.sqrt(total_slow_norm)
        total_fast_norm = np.sqrt(total_fast_norm)

        # Ratio of slow to total gradient
        total_norm = np.sqrt(total_slow_norm**2 + total_fast_norm**2)
        if total_norm < 1e-8:
            return 0.0

        return total_slow_norm / total_norm

    def compute_ema_cosine_similarity(self) -> float:
        """
        Compute EMA cosine similarity between consecutive gradient steps.
        """
        if len(self.history) < 2:
            return 0.0

        current_grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                current_grads[name] = param.grad.clone()

        if not current_grads or not hasattr(self, "_prev_grads"):
            self._prev_grads = current_grads
            return 0.0

        # Compute cosine similarity
        dot_product = 0.0
        norm_current = 0.0
        norm_prev = 0.0

        for name in current_grads:
            if name in self._prev_grads:
                current = current_grads[name].flatten()
                prev = self._prev_grads[name].flatten()

                dot_product += torch.dot(current, prev).item()
                norm_current += current.norm().item() ** 2
                norm_prev += prev.norm().item() ** 2

        norm_current = np.sqrt(norm_current)
        norm_prev = np.sqrt(norm_prev)

        if norm_current < 1e-8 or norm_prev < 1e-8:
            cos_sim = 0.0
        else:
            cos_sim = dot_product / (norm_current * norm_prev)

        self._prev_grads = current_grads
        return cos_sim

    def update(self, loss: float, accuracy: float, step: int) -> TelemetryState:
        """
        Update telemetry with current training state.

        Returns:
            Current telemetry state
        """
        # Collect gradients
        gradients = {}
        grad_norm = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.clone()
                grad_norm += param.grad.norm().item() ** 2

        grad_norm = np.sqrt(grad_norm)

        # Compute telemetry metrics
        id_value = self.compute_intrinsic_dimension(gradients)
        s_slow = self.compute_slow_gradient_strength()
        ema_cos = self.compute_ema_cosine_similarity()

        # Create telemetry state
        telemetry = TelemetryState(
            intrinsic_dimension=id_value,
            slow_gradient_strength=s_slow,
            ema_cosine_similarity=ema_cos,
            loss=loss,
            accuracy=accuracy,
            grad_norm=grad_norm,
            step=step,
        )

        # Store in history
        self.history.append(telemetry)

        return telemetry

    def get_recent_history(self, n: int = 100) -> list[TelemetryState]:
        """Get recent telemetry history."""
        return list(self.history)[-n:]

    def export_telemetry(self) -> dict[str, list[Any]]:
        """Export telemetry history for analysis."""
        if not self.history:
            return {}

        export = {
            "id": [],
            "s_slow": [],
            "ema_cos": [],
            "loss": [],
            "accuracy": [],
            "grad_norm": [],
            "step": [],
        }

        for t in self.history:
            export["id"].append(t.intrinsic_dimension)
            export["s_slow"].append(t.slow_gradient_strength)
            export["ema_cos"].append(t.ema_cosine_similarity)
            export["loss"].append(t.loss)
            export["accuracy"].append(t.accuracy)
            export["grad_norm"].append(t.grad_norm)
            export["step"].append(t.step)

        return export


# Factory functions


def create_grokfast_optimizer(
    model: nn.Module,
    base_optimizer: torch.optim.Optimizer,
    alpha: float = 0.98,
    lamb: float = 2.0,
    auto_gated: bool = True,
) -> GrokfastOptimizer:
    """Create Grokfast optimizer with default settings."""
    config = GrokfastConfig(alpha=alpha, lamb=lamb, enabled=True)
    detector = GrokSignalDetector() if auto_gated else None

    return GrokfastOptimizer(
        model=model, base_optimizer=base_optimizer, config=config, detector=detector
    )


def create_telemetry_tracker(model: nn.Module) -> TelemetryTracker:
    """Create telemetry tracker for grok detection."""
    return TelemetryTracker(model)


if __name__ == "__main__":
    # Demo Grokfast controller system
    print("⚡ Grokfast Controller + Telemetry System Demo")
    print("=" * 50)

    # Create simple model for demonstration
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))

    # Create base optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create Grokfast optimizer
    grokfast_opt = create_grokfast_optimizer(model, base_optimizer)
    telemetry = create_telemetry_tracker(model)

    print("1. Created Grokfast optimizer with config:")
    print(f"   Alpha: {grokfast_opt.config.alpha}")
    print(f"   Lambda: {grokfast_opt.config.lamb}")
    print(f"   Auto-gated: {grokfast_opt.detector is not None}")
    print()

    # Simulate training steps
    print("2. Simulating training steps with telemetry:")

    for step in range(10):
        # Simulate forward pass
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        # Backward pass
        grokfast_opt.zero_grad()
        loss.backward()

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean().item()

        # Update telemetry
        telem_state = telemetry.update(loss.item(), accuracy, step)

        # Optimization step
        step_stats = grokfast_opt.step(telem_state)

        print(
            f"   Step {step}: loss={loss.item():.3f}, acc={accuracy:.2f}, "
            f"ID={telem_state.intrinsic_dimension:.3f}, "
            f"S_slow={telem_state.slow_gradient_strength:.3f}, "
            f"Grokfast={'✓' if step_stats['grokfast_enabled'] else '✗'}"
        )

    print()

    # Show final statistics
    print("3. Final Statistics:")
    stats = grokfast_opt.get_stats()

    print(f"   Total steps: {stats['total_steps']}")
    print(f"   Grokfast steps: {stats['grokfast_steps']}")
    print(f"   Grokfast ratio: {stats['grokfast_ratio']:.2f}")
    print(f"   Average lambda: {stats['avg_lambda']:.2f}")
    print(f"   Grok detected: {stats['grok_detected']}")
    print(f"   Current confidence: {stats['current_confidence']:.2f}")

    print()

    # Test telemetry export
    print("4. Telemetry Export:")
    export = telemetry.export_telemetry()
    print(f"   Exported {len(export['step'])} telemetry records")
    print(f"   ID range: [{min(export['id']):.3f}, {max(export['id']):.3f}]")
    print(
        f"   S_slow range: [{min(export['s_slow']):.3f}, {max(export['s_slow']):.3f}]"
    )

    print()
    print("✅ Grokfast Controller Demo Complete")
    print()
    print("Key Features Demonstrated:")
    print("  • Telemetry-based grok detection (ID↓ + S_slow↑)")
    print("  • Automatic lambda gating based on geometry signals")
    print("  • EMA slow gradient tracking and amplification")
    print("  • Comprehensive training statistics and export")
    print("  • Integration-ready optimization wrapper")
