"""
GrokFast Integration for Cogment Training.

Provides selective GrokFast application with different parameters for each component:
- RefinementCore + ACT: Aggressive GrokFast for rapid grokking
- GatedLTM Memory: Gentler GrokFast to preserve memory dynamics  
- ACT Halting: NO GrokFast to preserve halting dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

# Import existing GrokFast implementation
from experiments.training.grokfast import GrokFastOptimizer

logger = logging.getLogger(__name__)


@dataclass
class GrokFastConfig:
    """Configuration for GrokFast optimization."""

    # Core refinement parameters (aggressive for grokking)
    core_alpha: float = 0.98  # Strong EMA smoothing
    core_lamb: float = 2.0  # 2x slow gradient amplification
    core_window_size: int = 100

    # Memory parameters (gentler to preserve dynamics)
    memory_alpha: float = 0.95  # Lighter smoothing
    memory_lamb: float = 1.5  # Gentler amplification
    memory_window_size: int = 50

    # ACT halting parameters (disabled)
    halting_enabled: bool = False  # NO GrokFast for halting

    # Stage-specific scheduling
    stage_1_2_enabled: bool = True  # Aggressive in ARC + Algorithmic
    stage_3_4_reduction: float = 0.6  # Reduce lamb by 40% in later stages

    # Monitoring
    grokking_detection_threshold: float = 0.7  # Slow gradient ratio threshold
    monitoring_interval: int = 50  # Steps between grokking checks


class CogmentGrokFastOptimizer:
    """
    GrokFast optimizer wrapper specifically for Cogment components.

    Provides selective application with different parameters for each model component.
    """

    def __init__(
        self,
        model_components: dict[str, nn.Module],
        base_optimizers: dict[str, torch.optim.Optimizer],
        config: GrokFastConfig,
    ):
        self.config = config
        self.model_components = model_components
        self.base_optimizers = base_optimizers
        self.grokfast_optimizers: dict[str, GrokFastOptimizer] = {}
        self.statistics: dict[str, dict] = {}

        # Initialize GrokFast optimizers for each component
        self._initialize_grokfast_optimizers()

        logger.info(f"Initialized CogmentGrokFast with {len(self.grokfast_optimizers)} components")

    def _initialize_grokfast_optimizers(self):
        """Initialize GrokFast optimizers for each model component."""

        for component_name, model in self.model_components.items():
            base_optimizer = self.base_optimizers[component_name]

            if component_name == "refinement_core":
                # Aggressive GrokFast for core refinement
                grokfast_opt = GrokFastOptimizer(
                    base_optimizer=base_optimizer,
                    model=model,
                    alpha=self.config.core_alpha,
                    lamb=self.config.core_lamb,
                    window_size=self.config.core_window_size,
                )
                self.grokfast_optimizers[component_name] = grokfast_opt

            elif component_name == "memory" or component_name == "gated_ltm":
                # Gentler GrokFast for memory to preserve dynamics
                grokfast_opt = GrokFastOptimizer(
                    base_optimizer=base_optimizer,
                    model=model,
                    alpha=self.config.memory_alpha,
                    lamb=self.config.memory_lamb,
                    window_size=self.config.memory_window_size,
                )
                self.grokfast_optimizers[component_name] = grokfast_opt

            elif component_name == "act_halting":
                # NO GrokFast for ACT halting to preserve dynamics
                if self.config.halting_enabled:
                    logger.warning("GrokFast enabled for ACT halting - may interfere with halting dynamics")
                    grokfast_opt = GrokFastOptimizer(
                        base_optimizer=base_optimizer,
                        model=model,
                        alpha=0.9,  # Very conservative
                        lamb=1.1,  # Minimal amplification
                        window_size=20,
                    )
                    self.grokfast_optimizers[component_name] = grokfast_opt
                else:
                    # Use base optimizer directly
                    self.grokfast_optimizers[component_name] = base_optimizer

            else:
                # Default GrokFast for other components
                grokfast_opt = GrokFastOptimizer(
                    base_optimizer=base_optimizer,
                    model=model,
                    alpha=self.config.core_alpha,
                    lamb=self.config.core_lamb,
                    window_size=self.config.core_window_size,
                )
                self.grokfast_optimizers[component_name] = grokfast_opt

            logger.info(f"Initialized GrokFast for {component_name}")

    def step(self, stage: int = 1, apply_grokfast: bool = True):
        """
        Perform optimization step with stage-specific GrokFast application.

        Args:
            stage: Current training stage (1-4)
            apply_grokfast: Whether to apply GrokFast amplification
        """
        # Determine if GrokFast should be applied based on stage
        if not apply_grokfast:
            # Use base optimizers only
            for component_name, optimizer in self.base_optimizers.items():
                optimizer.step()
            return

        # Stage-specific GrokFast application
        stage_reduction = 1.0
        if stage >= 3 and self.config.stage_3_4_reduction > 0:
            stage_reduction = self.config.stage_3_4_reduction

        for component_name, grokfast_opt in self.grokfast_optimizers.items():
            if isinstance(grokfast_opt, GrokFastOptimizer):
                # Apply stage reduction to lambda
                original_lamb = grokfast_opt.lamb
                if stage >= 3:
                    grokfast_opt.lamb = original_lamb * stage_reduction

                # Determine if this component should use GrokFast in this stage
                use_amplification = True
                if stage <= 2 and not self.config.stage_1_2_enabled:
                    use_amplification = False
                elif component_name == "act_halting" and not self.config.halting_enabled:
                    use_amplification = False

                # Perform step
                grokfast_opt.step(amplify=use_amplification)

                # Restore original lambda
                if stage >= 3:
                    grokfast_opt.lamb = original_lamb

            else:
                # Base optimizer (e.g., for ACT halting when GrokFast disabled)
                grokfast_opt.step()

    def zero_grad(self):
        """Zero gradients for all optimizers."""
        for optimizer in self.grokfast_optimizers.values():
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()
            else:
                # Base optimizer
                optimizer.zero_grad()

    def get_grokking_statistics(self) -> dict[str, dict]:
        """Get grokking statistics for all components."""
        stats = {}

        for component_name, grokfast_opt in self.grokfast_optimizers.items():
            if isinstance(grokfast_opt, GrokFastOptimizer):
                component_stats = grokfast_opt.get_statistics()
                stats[component_name] = component_stats
            else:
                stats[component_name] = {"status": "base_optimizer", "grokking_detected": False, "step_count": 0}

        return stats

    def detect_grokking_onset(self) -> dict[str, bool]:
        """Detect grokking onset for each component."""
        grokking_status = {}

        for component_name, grokfast_opt in self.grokfast_optimizers.items():
            if isinstance(grokfast_opt, GrokFastOptimizer):
                stats = grokfast_opt.get_statistics()
                grokking_status[component_name] = stats["grokking_detected"]
            else:
                grokking_status[component_name] = False

        return grokking_status

    def adjust_for_stage(self, stage: int):
        """
        Adjust GrokFast parameters for the current training stage.

        Args:
            stage: Current training stage (1-4)
        """
        logger.info(f"Adjusting GrokFast parameters for stage {stage}")

        for component_name, grokfast_opt in self.grokfast_optimizers.items():
            if isinstance(grokfast_opt, GrokFastOptimizer):
                if stage >= 3:
                    # Reduce amplification in later stages
                    if component_name == "refinement_core":
                        grokfast_opt.lamb = self.config.core_lamb * self.config.stage_3_4_reduction
                    elif component_name in ["memory", "gated_ltm"]:
                        grokfast_opt.lamb = self.config.memory_lamb * self.config.stage_3_4_reduction
                else:
                    # Restore original amplification for early stages
                    if component_name == "refinement_core":
                        grokfast_opt.lamb = self.config.core_lamb
                    elif component_name in ["memory", "gated_ltm"]:
                        grokfast_opt.lamb = self.config.memory_lamb


class SelectiveGrokFastManager:
    """
    High-level manager for selective GrokFast application across training stages.

    Orchestrates GrokFast optimization with stage-specific scheduling and monitoring.
    """

    def __init__(self, config: GrokFastConfig):
        self.config = config
        self.current_stage = 1
        self.grokking_history: list[dict] = []
        self.optimizer: CogmentGrokFastOptimizer | None = None

        logger.info("Initialized SelectiveGrokFastManager")

    def setup_optimizers(
        self, model_components: dict[str, nn.Module], base_optimizers: dict[str, torch.optim.Optimizer]
    ):
        """Setup GrokFast optimizers for model components."""
        self.optimizer = CogmentGrokFastOptimizer(
            model_components=model_components, base_optimizers=base_optimizers, config=self.config
        )

        logger.info("GrokFast optimizers configured for all components")

    def training_step(self, step: int, apply_grokfast: bool = True):
        """Perform a training step with GrokFast."""
        if self.optimizer is None:
            raise RuntimeError("Optimizers not setup. Call setup_optimizers first.")

        # Apply GrokFast optimization
        self.optimizer.step(stage=self.current_stage, apply_grokfast=apply_grokfast)

        # Monitor grokking progress
        if step % self.config.monitoring_interval == 0:
            self._monitor_grokking_progress(step)

    def zero_grad(self):
        """Zero gradients for all optimizers."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def transition_to_stage(self, new_stage: int):
        """Transition to a new training stage."""
        if new_stage != self.current_stage:
            logger.info(f"Transitioning from stage {self.current_stage} to stage {new_stage}")
            self.current_stage = new_stage

            if self.optimizer is not None:
                self.optimizer.adjust_for_stage(new_stage)

    def _monitor_grokking_progress(self, step: int):
        """Monitor and log grokking progress."""
        if self.optimizer is None:
            return

        # Get current statistics
        stats = self.optimizer.get_grokking_statistics()
        grokking_status = self.optimizer.detect_grokking_onset()

        # Log progress
        progress_summary = {
            "step": step,
            "stage": self.current_stage,
            "grokking_detected": grokking_status,
            "statistics": stats,
        }

        self.grokking_history.append(progress_summary)

        # Log significant grokking events
        for component, is_grokking in grokking_status.items():
            if is_grokking:
                component_stats = stats.get(component, {})
                slow_ratio = component_stats.get("gradient_stats", {}).get("slow_count", 0) / max(
                    component_stats.get("gradient_stats", {}).get("slow_count", 0)
                    + component_stats.get("gradient_stats", {}).get("fast_count", 0),
                    1,
                )
                logger.info(f"🚀 Grokking detected in {component} at step {step} (slow ratio: {slow_ratio:.3f})")

    def get_grokking_summary(self) -> dict:
        """Get comprehensive grokking summary."""
        if not self.grokking_history:
            return {"status": "no_data", "history_length": 0}

        latest = self.grokking_history[-1]

        # Count grokking components
        grokking_components = sum(1 for status in latest["grokking_detected"].values() if status)
        total_components = len(latest["grokking_detected"])

        return {
            "current_stage": self.current_stage,
            "total_steps_monitored": len(self.grokking_history),
            "grokking_components": grokking_components,
            "total_components": total_components,
            "grokking_ratio": grokking_components / max(total_components, 1),
            "latest_statistics": latest["statistics"],
            "grokking_timeline": [
                (h["step"], sum(h["grokking_detected"].values())) for h in self.grokking_history[-10:]
            ],
        }
