"""
Cogment Training Engine.

Main training engine with multi-optimizer setup, GrokFast integration, and 4-stage curriculum.
Replaces HRRM's training approach with enhanced loss functions and stage-specific optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..core.config import CogmentConfig

# Import Cogment components
from ..core.model import Cogment, CogmentOutput
from .curriculum import FourStageCurriculum, StageConfig
from .evaluator import EvaluationMetrics, StageEvaluator
from .grokfast_integration import GrokFastConfig, SelectiveGrokFastManager

# Import training components
from .losses import CogmentLoss

logger = logging.getLogger(__name__)


@dataclass
class MultiOptimizerConfig:
    """Configuration for multi-component optimization."""

    # Core refinement optimizer
    core_lr: float = 3e-4
    core_weight_decay: float = 0.01
    core_betas: tuple[float, float] = (0.9, 0.95)

    # Memory optimizer
    memory_lr: float = 1e-4
    memory_weight_decay: float = 0.001
    memory_betas: tuple[float, float] = (0.9, 0.999)

    # ACT halting optimizer
    halting_lr: float = 5e-4
    halting_weight_decay: float = 0.01
    halting_betas: tuple[float, float] = (0.9, 0.95)

    # Other components optimizer
    other_lr: float = 2e-4
    other_weight_decay: float = 0.01
    other_betas: tuple[float, float] = (0.9, 0.95)

    # Scheduler parameters
    scheduler_type: str = "cosine"  # 'cosine', 'linear', 'constant'
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""

    # Model configuration
    model_config: CogmentConfig

    # Optimizer configuration
    optimizer_config: MultiOptimizerConfig

    # GrokFast configuration
    grokfast_config: GrokFastConfig

    # Training parameters
    max_epochs: int = 10
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1

    # Evaluation parameters
    eval_interval: int = 500
    save_interval: int = 2000
    log_interval: int = 100

    # Early stopping
    early_stopping_patience: int = 5000
    min_delta: float = 1e-4

    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16

    # Checkpointing
    save_dir: str = "./checkpoints"
    save_best_only: bool = False

    # Curriculum
    use_curriculum: bool = True
    auto_advance_stages: bool = True

    # Memory management
    ltm_decay_interval: int = 100
    ltm_consolidation_interval: int = 5000


class CogmentTrainer:
    """
    Main training engine for Cogment with multi-optimizer setup and GrokFast integration.

    Orchestrates training across 4 curriculum stages with specialized optimization
    for each model component and automatic grokking acceleration.
    """

    def __init__(self, model: Cogment, config: TrainingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Initialize curriculum
        self.curriculum = FourStageCurriculum() if config.use_curriculum else None

        # Initialize loss function
        self.loss_fn = CogmentLoss()

        # Initialize evaluator
        self.evaluator = StageEvaluator()

        # Setup optimizers and GrokFast
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()
        self.grokfast_manager = self._setup_grokfast()

        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_metrics: dict[str, float] = {}
        self.training_history: list[dict[str, Any]] = []

        # AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CogmentTrainer with {sum(p.numel() for p in model.parameters()):,} parameters")

    def _setup_optimizers(self) -> dict[str, optim.Optimizer]:
        """Setup separate optimizers for different model components."""
        opt_config = self.config.optimizer_config
        optimizers = {}

        # Separate parameters by component
        core_params = []
        memory_params = []
        halting_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "refinement_core" in name:
                    core_params.append(param)
                elif "gated_ltm" in name or "memory" in name:
                    memory_params.append(param)
                elif "act_halting" in name or "halting" in name:
                    halting_params.append(param)
                else:
                    other_params.append(param)

        # Create optimizers for each component group
        if core_params:
            optimizers["core"] = optim.AdamW(
                core_params,
                lr=opt_config.core_lr,
                weight_decay=opt_config.core_weight_decay,
                betas=opt_config.core_betas,
            )
            logger.info(f"Core optimizer: {len(core_params)} parameters")

        if memory_params:
            optimizers["memory"] = optim.AdamW(
                memory_params,
                lr=opt_config.memory_lr,
                weight_decay=opt_config.memory_weight_decay,
                betas=opt_config.memory_betas,
            )
            logger.info(f"Memory optimizer: {len(memory_params)} parameters")

        if halting_params:
            optimizers["halting"] = optim.AdamW(
                halting_params,
                lr=opt_config.halting_lr,
                weight_decay=opt_config.halting_weight_decay,
                betas=opt_config.halting_betas,
            )
            logger.info(f"Halting optimizer: {len(halting_params)} parameters")

        if other_params:
            optimizers["other"] = optim.AdamW(
                other_params,
                lr=opt_config.other_lr,
                weight_decay=opt_config.other_weight_decay,
                betas=opt_config.other_betas,
            )
            logger.info(f"Other optimizer: {len(other_params)} parameters")

        return optimizers

    def _setup_schedulers(self) -> dict[str, Any]:
        """Setup learning rate schedulers for each optimizer."""
        schedulers = {}
        opt_config = self.config.optimizer_config

        for name, optimizer in self.optimizers.items():
            if opt_config.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_epochs * 1000,  # Approximate steps
                    eta_min=optimizer.param_groups[0]["lr"] * opt_config.min_lr_ratio,
                )
            elif opt_config.scheduler_type == "linear":
                scheduler = optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=opt_config.min_lr_ratio,
                    total_iters=self.config.max_epochs * 1000,
                )
            else:
                scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

            schedulers[name] = scheduler

        return schedulers

    def _setup_grokfast(self) -> SelectiveGrokFastManager:
        """Setup GrokFast manager with model components."""
        # Extract model components for GrokFast
        model_components = {"refinement_core": self.model.refinement_core, "backbone": self.model.backbone}

        # Add memory if available
        if hasattr(self.model, "gated_ltm"):
            model_components["memory"] = self.model.gated_ltm
        elif hasattr(self.model.refinement_core, "memory"):
            model_components["memory"] = self.model.refinement_core.memory

        # Add ACT halting
        if hasattr(self.model, "act_halting"):
            model_components["act_halting"] = self.model.act_halting

        # Create GrokFast manager
        manager = SelectiveGrokFastManager(self.config.grokfast_config)
        manager.setup_optimizers(model_components, self.optimizers)

        return manager

    def training_step(self, batch: dict[str, torch.Tensor], stage_config: StageConfig) -> dict[str, float]:
        """Perform a single training step."""
        self.model.train()

        # Extract batch data
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        augmented_ids = batch.get("augmented_input_ids", None)

        # Zero gradients
        self.grokfast_manager.zero_grad()

        step_metrics = {}

        # Forward pass with AMP if enabled
        if self.scaler is not None:
            with torch.cuda.amp.autocast(dtype=self.config.amp_dtype):
                total_loss, loss_info = self._forward_pass(input_ids, labels, augmented_ids, stage_config)
        else:
            total_loss, loss_info = self._forward_pass(input_ids, labels, augmented_ids, stage_config)

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(list(self.optimizers.values())[0])  # Unscale for clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            # GrokFast optimization step
            self.grokfast_manager.training_step(self.current_step, apply_grokfast=True)

            # Update schedulers
            for scheduler in self.schedulers.values():
                scheduler.step()

            self.scaler.update()
        else:
            total_loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            # GrokFast optimization step
            self.grokfast_manager.training_step(self.current_step, apply_grokfast=True)

            # Update schedulers
            for scheduler in self.schedulers.values():
                scheduler.step()

        # Memory management
        if self.current_step % self.config.ltm_decay_interval == 0:
            self._apply_memory_decay()

        if self.current_step % self.config.ltm_consolidation_interval == 0:
            self._consolidate_memory()

        # Collect metrics
        step_metrics.update(
            {
                "total_loss": total_loss.item(),
                "step": self.current_step,
                "epoch": self.current_epoch,
                "stage": self.curriculum.current_stage.value if self.curriculum else 0,
            }
        )
        step_metrics.update({f"loss_{k}": v for k, v in loss_info["component_losses"].items()})

        return step_metrics

    def _forward_pass(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        augmented_ids: torch.Tensor | None,
        stage_config: StageConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass with stage-specific configuration."""
        # Set stage-specific model parameters
        max_steps = stage_config.max_refinement_steps

        # Forward pass for original input
        output: CogmentOutput = self.model(
            input_ids=input_ids, labels=labels, max_refinement_steps=max_steps, return_refinement_details=True
        )

        # Get step logits for deep supervision
        step_logits = []
        if output.refinement_outputs:
            step_logits = [ref_out.combined_logits for ref_out in output.refinement_outputs]
        else:
            step_logits = [output.logits]

        # Forward pass for augmented input (if available)
        augmented_logits = None
        if augmented_ids is not None and stage_config.consistency_weight > 0:
            aug_output: CogmentOutput = self.model(
                input_ids=augmented_ids.to(self.device), max_refinement_steps=max_steps
            )
            augmented_logits = aug_output.logits

        # Compute combined loss
        total_loss, loss_info = self.loss_fn(
            step_logits=step_logits,
            targets=labels,
            ponder_costs=output.ponder_cost if output.ponder_cost is not None else torch.zeros(1, device=self.device),
            augmented_logits=augmented_logits,
            stage=self.curriculum.current_stage.value if self.curriculum else 1,
            step=self.current_step,
            return_components=True,
        )

        return total_loss, loss_info

    def _apply_memory_decay(self):
        """Apply memory decay if model has memory component."""
        if hasattr(self.model, "gated_ltm"):
            self.model.gated_ltm.decay_step()
        elif hasattr(self.model.refinement_core, "memory"):
            if hasattr(self.model.refinement_core.memory, "decay_step"):
                self.model.refinement_core.memory.decay_step()

    def _consolidate_memory(self):
        """Consolidate memory if model has memory component."""
        if hasattr(self.model, "gated_ltm"):
            reset_count = self.model.gated_ltm.consolidate_memory()
            if reset_count > 0:
                logger.info(f"Consolidated {reset_count} memory slots")

    def evaluate(self, eval_dataloader: DataLoader, stage_config: StageConfig | None = None) -> EvaluationMetrics:
        """Evaluate model on validation data."""
        self.model.eval()

        if stage_config is None and self.curriculum is not None:
            stage_config = self.curriculum.get_current_config()

        eval_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        total_ponder_cost = 0.0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                # Forward pass
                output: CogmentOutput = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    max_refinement_steps=stage_config.max_refinement_steps if stage_config else 4,
                )

                # Accumulate metrics
                if output.loss is not None:
                    eval_loss += output.loss.item() * input_ids.size(0)

                # Accuracy calculation
                predictions = output.logits.argmax(dim=-1)
                correct = (predictions == labels).float()
                mask = (labels != -100).float()
                correct_predictions += (correct * mask).sum().item()
                total_samples += mask.sum().item()

                # Ponder cost
                if output.ponder_cost is not None:
                    total_ponder_cost += output.ponder_cost.sum().item()

        # Calculate final metrics
        avg_loss = eval_loss / max(len(eval_dataloader), 1)
        accuracy = correct_predictions / max(total_samples, 1)
        avg_ponder_cost = total_ponder_cost / max(len(eval_dataloader), 1)

        metrics = EvaluationMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            ponder_cost=avg_ponder_cost,
            step=self.current_step,
            stage=self.curriculum.current_stage.value if self.curriculum else 0,
        )

        return metrics

    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader | None = None) -> dict[str, Any]:
        """
        Main training loop with curriculum and stage management.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results and statistics
        """
        logger.info("Starting Cogment training")
        start_time = time.time()

        best_eval_loss = float("inf")
        patience_counter = 0

        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                epoch_metrics = []

                # Get current stage configuration
                if self.curriculum is not None:
                    stage_config = self.curriculum.get_current_config()
                    logger.info(f"Epoch {epoch}: Training stage {stage_config.name}")
                else:
                    stage_config = None

                # Training loop
                for batch_idx, batch in enumerate(train_dataloader):
                    step_metrics = self.training_step(batch, stage_config)
                    epoch_metrics.append(step_metrics)
                    self.current_step += 1

                    # Logging
                    if self.current_step % self.config.log_interval == 0:
                        self._log_training_progress(step_metrics)

                    # Evaluation
                    if eval_dataloader is not None and self.current_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate(eval_dataloader, stage_config)
                        self._log_evaluation_metrics(eval_metrics)

                        # Check for improvement
                        if eval_metrics.loss < best_eval_loss - self.config.min_delta:
                            best_eval_loss = eval_metrics.loss
                            patience_counter = 0
                            self.best_metrics = eval_metrics.__dict__.copy()

                            # Save best model
                            if self.config.save_best_only:
                                self.save_checkpoint("best")
                        else:
                            patience_counter += 1

                        # Early stopping
                        if patience_counter >= self.config.early_stopping_patience:
                            logger.info(f"Early stopping triggered at step {self.current_step}")
                            break

                        # Stage advancement check
                        if self.curriculum is not None and self.config.auto_advance_stages and eval_metrics is not None:

                            stage_metrics = {
                                "accuracy": eval_metrics.accuracy,
                                "ponder_cost": eval_metrics.ponder_cost,
                                "loss": eval_metrics.loss,
                            }

                            if self.curriculum.advance_stage(stage_metrics):
                                logger.info(f"Advanced to stage {self.curriculum.current_stage.name}")

                                # Update GrokFast for new stage
                                self.grokfast_manager.transition_to_stage(self.curriculum.current_stage.value)

                    # Checkpointing
                    if self.current_step % self.config.save_interval == 0:
                        self.save_checkpoint(f"step_{self.current_step}")

                    # Break if stage/epoch limits reached
                    if stage_config and self.current_step >= stage_config.max_steps:
                        logger.info(f"Reached stage step limit: {stage_config.max_steps}")
                        break

                # End of epoch processing
                if epoch_metrics:
                    avg_epoch_loss = sum(m["total_loss"] for m in epoch_metrics) / len(epoch_metrics)
                    logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

                    self.training_history.append(
                        {
                            "epoch": epoch,
                            "avg_loss": avg_epoch_loss,
                            "step_count": len(epoch_metrics),
                            "final_step": self.current_step,
                        }
                    )

                # Early stopping check
                if patience_counter >= self.config.early_stopping_patience:
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Final evaluation and summary
        training_time = time.time() - start_time

        final_results = {
            "training_completed": True,
            "training_time": training_time,
            "total_steps": self.current_step,
            "total_epochs": self.current_epoch + 1,
            "best_metrics": self.best_metrics,
            "training_history": self.training_history,
            "grokfast_summary": self.grokfast_manager.get_grokking_summary(),
            "final_stage": self.curriculum.current_stage.name if self.curriculum else None,
            "curriculum_summary": self.curriculum.get_curriculum_summary() if self.curriculum else None,
        }

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best evaluation loss: {self.best_metrics.get('loss', 'N/A')}")

        return final_results

    def _log_training_progress(self, metrics: dict[str, float]):
        """Log training progress."""
        stage_name = self.curriculum.current_stage.name if self.curriculum else "N/A"

        logger.info(
            f"Step {metrics['step']:6d} | Stage: {stage_name} | "
            f"Loss: {metrics['total_loss']:.4f} | "
            f"Epoch: {metrics['epoch']}"
        )

    def _log_evaluation_metrics(self, metrics: EvaluationMetrics):
        """Log evaluation metrics."""
        logger.info(
            f"Eval Step {metrics.step} | "
            f"Loss: {metrics.loss:.4f} | "
            f"Accuracy: {metrics.accuracy:.3f} | "
            f"Ponder: {metrics.ponder_cost:.2f}"
        )

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.save_dir) / f"{name}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_states": {k: opt.state_dict() for k, opt in self.optimizers.items()},
            "scheduler_states": {k: sch.state_dict() for k, sch in self.schedulers.items()},
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "best_metrics": self.best_metrics,
            "curriculum_state": {
                "current_stage": self.curriculum.current_stage if self.curriculum else None,
                "stage_history": self.curriculum.stage_history if self.curriculum else [],
            },
            "grokfast_summary": self.grokfast_manager.get_grokking_summary(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer states
        for name, optimizer in self.optimizers.items():
            if name in checkpoint["optimizer_states"]:
                optimizer.load_state_dict(checkpoint["optimizer_states"][name])

        # Load scheduler states
        for name, scheduler in self.schedulers.items():
            if name in checkpoint["scheduler_states"]:
                scheduler.load_state_dict(checkpoint["scheduler_states"][name])

        # Load training state
        self.current_step = checkpoint.get("current_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.best_metrics = checkpoint.get("best_metrics", {})

        # Load curriculum state
        if self.curriculum and "curriculum_state" in checkpoint:
            curriculum_state = checkpoint["curriculum_state"]
            if curriculum_state["current_stage"] is not None:
                self.curriculum.current_stage = curriculum_state["current_stage"]
            if curriculum_state["stage_history"]:
                self.curriculum.stage_history = curriculum_state["stage_history"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from step {self.current_step}, epoch {self.current_epoch}")
