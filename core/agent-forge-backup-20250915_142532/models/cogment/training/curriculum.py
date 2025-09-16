"""
4-Stage Curriculum for Cogment Training.

Replaces HRRM's 3-phase approach with enhanced curriculum designed for accelerated grokking:
- Stage 0: Sanity checks (synthetic linear maps, toy mazes)
- Stage 1: ARC-like visual reasoning (~300 augmentations per task)
- Stage 2: Algorithmic puzzles (Sudoku, Mazes, ListOps)
- Stage 3: Math & multi-hop text (GSM8K, HotpotQA)
- Stage 4: Long-context tasks (LongBench, SCROLLS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Training curriculum stages."""

    SANITY = 0  # Sanity checks and basic functionality
    ARC_VISUAL = 1  # ARC-like visual reasoning with augmentations
    ALGORITHMIC = 2  # Algorithmic puzzles and structured reasoning
    MATH_TEXT = 3  # Mathematical reasoning and multi-hop text
    LONG_CONTEXT = 4  # Long-context understanding and generation


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""

    # Stage identification
    stage: CurriculumStage
    name: str
    description: str

    # Training parameters
    max_steps: int = 2000
    batch_size: int = 8
    sequence_length: int = 512
    learning_rate: float = 1e-4

    # ACT and refinement parameters
    max_refinement_steps: int = 8
    act_threshold: float = 0.99
    ponder_cost_initial: float = 0.005
    ponder_cost_final: float = 0.02

    # GrokFast parameters
    grokfast_enabled: bool = True
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0

    # LTM memory parameters
    ltm_read_only: bool = True
    ltm_write_alpha: float = 0.0
    ltm_decay_rate: float = 1e-3

    # Loss weights
    deep_supervision_weight: float = 1.0
    improvement_weight: float = 0.1
    consistency_weight: float = 0.1
    ponder_weight: float = 1.0

    # Data augmentation
    augmentation_enabled: bool = False
    augmentation_rate: float = 0.0
    augmentation_types: list[str] = field(default_factory=list)

    # Evaluation criteria
    convergence_patience: int = 500
    min_accuracy: float = 0.7
    max_ponder_cost: float = 4.0

    # Stage-specific settings
    extra_config: dict[str, Any] = field(default_factory=dict)


class FourStageCurriculum:
    """
    4-Stage curriculum for progressive Cogment training.

    Implements a structured learning progression from basic sanity checks
    to complex long-context reasoning, with stage-specific optimization
    and GrokFast integration for accelerated grokking.
    """

    def __init__(self):
        self.stages = self._create_default_stages()
        self.current_stage = CurriculumStage.SANITY
        self.stage_history: list[dict[str, Any]] = []

        logger.info(f"Initialized 4-stage curriculum with {len(self.stages)} stages")

    def _create_default_stages(self) -> dict[CurriculumStage, StageConfig]:
        """Create default configuration for all curriculum stages."""

        stages = {}

        # Stage 0: Sanity Checks
        stages[CurriculumStage.SANITY] = StageConfig(
            stage=CurriculumStage.SANITY,
            name="Sanity Checks",
            description="Basic functionality validation with synthetic tasks",
            # Conservative training parameters
            max_steps=500,
            batch_size=16,
            sequence_length=128,
            learning_rate=5e-4,
            # Minimal ACT complexity
            max_refinement_steps=2,
            act_threshold=0.95,
            ponder_cost_initial=0.001,
            ponder_cost_final=0.005,
            # No GrokFast for sanity checks
            grokfast_enabled=False,
            # Read-only memory
            ltm_read_only=True,
            ltm_write_alpha=0.0,
            # Basic loss configuration
            deep_supervision_weight=1.0,
            improvement_weight=0.0,
            consistency_weight=0.0,
            ponder_weight=0.1,
            # No augmentation
            augmentation_enabled=False,
            # Lenient convergence
            convergence_patience=100,
            min_accuracy=0.8,
            max_ponder_cost=2.0,
            extra_config={
                "task_types": ["linear_maps", "toy_mazes", "simple_sequences"],
                "synthetic_data_only": True,
                "validation_strict": False,
            },
        )

        # Stage 1: ARC-like Visual Reasoning
        stages[CurriculumStage.ARC_VISUAL] = StageConfig(
            stage=CurriculumStage.ARC_VISUAL,
            name="ARC Visual Reasoning",
            description="Visual pattern recognition with heavy augmentation",
            # Moderate training parameters
            max_steps=4000,
            batch_size=8,
            sequence_length=256,
            learning_rate=3e-4,
            # Increased ACT complexity
            max_refinement_steps=4,
            act_threshold=0.98,
            ponder_cost_initial=0.005,
            ponder_cost_final=0.015,
            # Aggressive GrokFast for visual grokking
            grokfast_enabled=True,
            grokfast_alpha=0.98,
            grokfast_lamb=2.0,
            # Still read-only memory
            ltm_read_only=True,
            ltm_write_alpha=0.0,
            # Enhanced loss with improvement
            deep_supervision_weight=1.0,
            improvement_weight=0.5,
            consistency_weight=0.1,
            ponder_weight=0.5,
            # Heavy augmentation for visual reasoning
            augmentation_enabled=True,
            augmentation_rate=0.8,
            augmentation_types=["rotate_90", "rotate_180", "rotate_270", "flip_h", "flip_v"],
            # Stricter convergence for grokking
            convergence_patience=800,
            min_accuracy=0.75,
            max_ponder_cost=3.0,
            extra_config={
                "task_types": ["arc_visual", "pattern_completion", "visual_analogies"],
                "augmentations_per_task": 300,
                "grokking_target": True,
                "visual_reasoning_focus": True,
            },
        )

        # Stage 2: Algorithmic Puzzles
        stages[CurriculumStage.ALGORITHMIC] = StageConfig(
            stage=CurriculumStage.ALGORITHMIC,
            name="Algorithmic Reasoning",
            description="Structured algorithmic puzzles and logical reasoning",
            # Standard training parameters
            max_steps=8000,
            batch_size=6,
            sequence_length=512,
            learning_rate=2e-4,
            # Full ACT complexity
            max_refinement_steps=8,
            act_threshold=0.99,
            ponder_cost_initial=0.005,
            ponder_cost_final=0.02,
            # Strong GrokFast for algorithmic grokking
            grokfast_enabled=True,
            grokfast_alpha=0.98,
            grokfast_lamb=2.0,
            # Begin memory writes
            ltm_read_only=False,
            ltm_write_alpha=0.05,
            ltm_decay_rate=5e-4,
            # Full loss configuration
            deep_supervision_weight=1.0,
            improvement_weight=1.0,
            consistency_weight=0.5,
            ponder_weight=1.0,
            # Moderate augmentation
            augmentation_enabled=True,
            augmentation_rate=0.3,
            augmentation_types=["shuffle_ops", "add_noise", "permute_vars"],
            # Performance-focused convergence
            convergence_patience=1500,
            min_accuracy=0.7,
            max_ponder_cost=4.0,
            extra_config={
                "task_types": ["sudoku", "mazes", "listops", "graph_algorithms"],
                "structured_reasoning": True,
                "algorithm_templates": True,
                "step_by_step_supervision": True,
            },
        )

        # Stage 3: Math & Multi-hop Text
        stages[CurriculumStage.MATH_TEXT] = StageConfig(
            stage=CurriculumStage.MATH_TEXT,
            name="Math & Multi-hop Reasoning",
            description="Mathematical reasoning and complex text understanding",
            # Increased complexity parameters
            max_steps=16000,
            batch_size=4,
            sequence_length=1024,
            learning_rate=1e-4,
            # Full ACT capability
            max_refinement_steps=8,
            act_threshold=0.99,
            ponder_cost_initial=0.01,
            ponder_cost_final=0.02,
            # Reduced GrokFast to preserve convergence
            grokfast_enabled=True,
            grokfast_alpha=0.95,
            grokfast_lamb=1.2,
            # Active memory usage
            ltm_read_only=False,
            ltm_write_alpha=0.1,
            ltm_decay_rate=1e-3,
            # Balanced loss with consistency
            deep_supervision_weight=0.8,
            improvement_weight=1.0,
            consistency_weight=1.0,
            ponder_weight=1.0,
            # Minimal augmentation
            augmentation_enabled=True,
            augmentation_rate=0.1,
            augmentation_types=["paraphrase", "reorder_steps"],
            # Quality-focused convergence
            convergence_patience=3000,
            min_accuracy=0.65,
            max_ponder_cost=5.0,
            extra_config={
                "task_types": ["gsm8k", "hotpotqa", "math_word_problems", "multi_hop_qa"],
                "reasoning_chains": True,
                "mathematical_notation": True,
                "context_integration": True,
            },
        )

        # Stage 4: Long-context Tasks
        stages[CurriculumStage.LONG_CONTEXT] = StageConfig(
            stage=CurriculumStage.LONG_CONTEXT,
            name="Long-context Understanding",
            description="Extended context processing and generation",
            # Maximum complexity parameters
            max_steps=32000,
            batch_size=2,
            sequence_length=2048,
            learning_rate=5e-5,
            # Conservative ACT for stability
            max_refinement_steps=6,
            act_threshold=0.99,
            ponder_cost_initial=0.015,
            ponder_cost_final=0.025,
            # Minimal GrokFast for stability
            grokfast_enabled=True,
            grokfast_alpha=0.9,
            grokfast_lamb=1.1,
            # Full memory utilization
            ltm_read_only=False,
            ltm_write_alpha=0.15,
            ltm_decay_rate=1e-3,
            # Efficiency-focused loss
            deep_supervision_weight=0.6,
            improvement_weight=1.0,
            consistency_weight=1.0,
            ponder_weight=1.2,
            # No augmentation for long sequences
            augmentation_enabled=False,
            # Stability-focused convergence
            convergence_patience=5000,
            min_accuracy=0.6,
            max_ponder_cost=6.0,
            extra_config={
                "task_types": ["longbench", "scrolls", "long_summarization", "document_qa"],
                "context_windows": [1024, 2048, 4096],
                "memory_critical": True,
                "efficiency_priority": True,
            },
        )

        return stages

    def get_stage_config(self, stage: CurriculumStage) -> StageConfig:
        """Get configuration for a specific stage."""
        if stage not in self.stages:
            raise ValueError(f"Unknown curriculum stage: {stage}")
        return self.stages[stage]

    def get_current_config(self) -> StageConfig:
        """Get configuration for the current stage."""
        return self.get_stage_config(self.current_stage)

    def advance_stage(self, metrics: dict[str, float]) -> bool:
        """
        Check if conditions are met to advance to the next stage.

        Args:
            metrics: Current training metrics

        Returns:
            bool: Whether stage advancement occurred
        """
        current_config = self.get_current_config()

        # Check advancement criteria
        accuracy = metrics.get("accuracy", 0.0)
        ponder_cost = metrics.get("ponder_cost", float("inf"))
        loss = metrics.get("loss", float("inf"))

        advancement_criteria = {
            "accuracy_met": accuracy >= current_config.min_accuracy,
            "ponder_cost_met": ponder_cost <= current_config.max_ponder_cost,
            "loss_stable": loss < 2.0,  # Basic stability check
        }

        # All criteria must be met
        can_advance = all(advancement_criteria.values())

        if can_advance:
            # Record stage completion
            stage_record = {
                "stage": self.current_stage,
                "final_metrics": metrics.copy(),
                "advancement_criteria": advancement_criteria,
                "completed": True,
            }
            self.stage_history.append(stage_record)

            # Advance to next stage
            stage_values = list(CurriculumStage)
            current_idx = stage_values.index(self.current_stage)

            if current_idx < len(stage_values) - 1:
                self.current_stage = stage_values[current_idx + 1]
                logger.info(f"Advanced to stage {self.current_stage.name}")
                return True
            else:
                logger.info("Curriculum completed - reached final stage")
                return False

        return False

    def set_stage(self, stage: CurriculumStage):
        """Manually set the current stage."""
        if stage not in self.stages:
            raise ValueError(f"Unknown curriculum stage: {stage}")

        logger.info(f"Manually setting stage to {stage.name}")
        self.current_stage = stage

    def get_stage_progression(self) -> list[tuple[CurriculumStage, str, bool]]:
        """Get progression status for all stages."""
        progression = []
        completed_stages = {record["stage"] for record in self.stage_history if record["completed"]}

        for stage in CurriculumStage:
            self.get_stage_config(stage)
            is_completed = stage in completed_stages
            is_current = stage == self.current_stage

            if is_current:
                status = "CURRENT"
            elif is_completed:
                status = "COMPLETED"
            else:
                status = "PENDING"

            progression.append((stage, status, is_completed))

        return progression

    def get_training_schedule(self, stage: CurriculumStage | None = None) -> dict[str, Any]:
        """
        Get comprehensive training schedule for a stage.

        Args:
            stage: Stage to get schedule for (default: current stage)

        Returns:
            Training schedule with all parameters
        """
        target_stage = stage or self.current_stage
        config = self.get_stage_config(target_stage)

        schedule = {
            "stage_info": {"stage": target_stage, "name": config.name, "description": config.description},
            "training_params": {
                "max_steps": config.max_steps,
                "batch_size": config.batch_size,
                "sequence_length": config.sequence_length,
                "learning_rate": config.learning_rate,
            },
            "model_params": {
                "max_refinement_steps": config.max_refinement_steps,
                "act_threshold": config.act_threshold,
                "ponder_cost_range": (config.ponder_cost_initial, config.ponder_cost_final),
            },
            "optimization": {
                "grokfast_enabled": config.grokfast_enabled,
                "grokfast_params": (
                    {"alpha": config.grokfast_alpha, "lamb": config.grokfast_lamb} if config.grokfast_enabled else None
                ),
            },
            "memory_config": {
                "read_only": config.ltm_read_only,
                "write_alpha": config.ltm_write_alpha,
                "decay_rate": config.ltm_decay_rate,
            },
            "loss_weights": {
                "deep_supervision": config.deep_supervision_weight,
                "improvement": config.improvement_weight,
                "consistency": config.consistency_weight,
                "ponder": config.ponder_weight,
            },
            "augmentation": {
                "enabled": config.augmentation_enabled,
                "rate": config.augmentation_rate,
                "types": config.augmentation_types.copy(),
            },
            "convergence_criteria": {
                "patience": config.convergence_patience,
                "min_accuracy": config.min_accuracy,
                "max_ponder_cost": config.max_ponder_cost,
            },
            "stage_specific": config.extra_config.copy(),
        }

        return schedule

    def is_complete(self) -> bool:
        """Check if the entire curriculum has been completed."""
        return self.current_stage == CurriculumStage.LONG_CONTEXT and len(self.stage_history) == len(CurriculumStage)

    def get_curriculum_summary(self) -> dict[str, Any]:
        """Get comprehensive curriculum status summary."""
        progression = self.get_stage_progression()
        completed_count = sum(1 for _, _, is_completed in progression if is_completed)

        return {
            "current_stage": {
                "stage": self.current_stage,
                "name": self.get_current_config().name,
                "description": self.get_current_config().description,
            },
            "progression": {
                "completed_stages": completed_count,
                "total_stages": len(CurriculumStage),
                "completion_percentage": (completed_count / len(CurriculumStage)) * 100,
                "is_complete": self.is_complete(),
            },
            "stage_details": progression,
            "history": self.stage_history.copy(),
            "next_stage": {
                "available": completed_count < len(CurriculumStage) - 1,
                "stage": (
                    list(CurriculumStage)[list(CurriculumStage).index(self.current_stage) + 1]
                    if list(CurriculumStage).index(self.current_stage) < len(CurriculumStage) - 1
                    else None
                ),
            },
        }
