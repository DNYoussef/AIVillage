"""
Agent Forge Unified Pipeline

Complete end-to-end orchestration of all Agent Forge phases:
1. Cognate: Model creation and initialization (NEWLY IMPLEMENTED)
2. EvoMerge: Evolutionary model optimization
3. Quiet-STaR: Reasoning enhancement baking
4. BitNet 1.58: Initial compression
5. Forge Training: Main training loop with Grokfast
6. Tool & Persona Baking: Identity and capability baking with Grokfast
7. ADAS: Architecture search with vector composition (Transformers Squared)
8. Final Compression: SeedLM + VPTQ + Hypercompression

NOW COMPLETE 8-PHASE PIPELINE WITH MODEL CREATION!
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn

# Import phase controller infrastructure
from agent_forge.core.phase_controller import (
    PhaseController,
    PhaseOrchestrator,
    PhaseResult,
)

logger = logging.getLogger(__name__)


@dataclass
class UnifiedConfig:
    """Configuration for the complete Agent Forge pipeline."""

    # Base configuration
    base_models: list[str] = field(
        default_factory=lambda: [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
            "Qwen/Qwen2-1.5B-Instruct",
        ]
    )
    output_dir: Path = Path("./agent_forge_output")
    checkpoint_dir: Path = Path("./agent_forge_checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase control flags (8-phase pipeline)
    enable_cognate: bool = True  # NEW - Phase 1: Model creation
    enable_evomerge: bool = True
    enable_quietstar: bool = True
    enable_initial_compression: bool = True
    enable_training: bool = True
    enable_tool_baking: bool = True
    enable_adas: bool = True
    enable_final_compression: bool = True

    # Cognate configuration (Phase 1 - NEW)
    cognate_init_strategy: str = "xavier_uniform"  # xavier_uniform, kaiming_normal, custom
    cognate_merge_strategy: str = "average"  # average, weighted, evolutionary
    cognate_target_architecture: str = "auto"  # auto, custom, or specific model
    cognate_validate_compatibility: bool = True

    # EvoMerge configuration
    evomerge_generations: int = 50
    evomerge_population_size: int = 8
    evomerge_techniques: list[str] = field(
        default_factory=lambda: ["linear", "slerp", "ties", "dare", "frankenmerge", "dfs"]
    )

    # Quiet-STaR configuration
    quietstar_thought_length: int = 32
    quietstar_num_thoughts: int = 4
    quietstar_training_steps: int = 1000

    # Initial BitNet compression
    bitnet_bits: float = 1.58
    bitnet_group_size: int = 128

    # Forge training configuration
    training_steps: int = 100000
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Grokfast settings (used across all training phases)
    grokfast_enabled: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda_init: float = 0.05
    grokfast_lambda_max: float = 0.25

    # DSPy prompt optimization
    enable_dspy_optimization: bool = False

    # Edge-of-chaos settings
    edge_control_enabled: bool = True
    target_success_range: tuple[float, float] = (0.55, 0.75)

    # Self-modeling settings
    self_model_enabled: bool = True
    self_model_weight: float = 0.1
    tap_layers: list[int] = field(default_factory=lambda: [4, 8, 12])

    # Dream cycle settings
    dream_enabled: bool = True
    dream_interval: int = 1000
    dream_duration: int = 50

    # Tool & Persona baking
    tools_to_bake: list[str] = field(default_factory=lambda: ["rag_query", "code_execution", "web_search"])
    persona_traits: dict[str, float] = field(
        default_factory=lambda: {
            "helpfulness": 0.9,
            "creativity": 0.7,
            "precision": 0.8,
        }
    )

    # ADAS configuration
    adas_iterations: int = 10
    adas_technique_pool_size: int = 100
    adas_vector_composition_scale: float = 0.1  # For Transformers Squared

    # Final compression
    seedlm_seed_ratio: float = 0.05
    vptq_codebook_size: int = 256
    hypercompression_ratio: float = 0.5

    # Monitoring
    wandb_project: str | None = "agent_forge"
    wandb_entity: str | None = None
    log_interval: int = 10
    checkpoint_interval: int = 1000

    # P2P Federated training
    enable_federated: bool = False
    federated_peers: list[str] = field(default_factory=list)
    federated_aggregation: str = "fedavg"

    # Fog compute settings
    enable_fog_compute: bool = False
    fog_nodes: list[str] = field(default_factory=list)
    fog_scheduling: str = "round_robin"

    def __post_init__(self):
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class UnifiedPipeline:
    """
    Main orchestrator for the complete Agent Forge pipeline.
    Manages all phases and ensures smooth model passing between stages.
    """

    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.orchestrator = PhaseOrchestrator()
        self.phase_results = []
        self.checkpoints = {}

        # Initialize phase controllers from consolidated modules
        self.phases = self._initialize_phases()

    def _initialize_phases(self):
        """Initialize all phase controllers and return enabled phases."""
        phases = []

        try:
            # Import consolidated phase modules

            # Phase 1: Cognate (Model Creation) - NEWLY IMPLEMENTED
            if self.config.enable_cognate:
                from .phases.cognate import CognateConfig, CognatePhase

                cognate_config = CognateConfig(
                    base_models=self.config.base_models,
                    target_architecture=self.config.cognate_target_architecture,
                    init_strategy=self.config.cognate_init_strategy,
                    merge_strategy=self.config.cognate_merge_strategy,
                    validate_compatibility=self.config.cognate_validate_compatibility,
                    device=self.config.device,
                )
                phases.append(("CognatePhase", CognatePhase(cognate_config)))

            if self.config.enable_evomerge:
                from .phases.evomerge import EvoMergeConfig, EvoMergePhase

                evomerge_config = EvoMergeConfig(
                    base_models=self.config.base_models,
                    population_size=self.config.evomerge_population_size,
                    generations=self.config.evomerge_generations,
                    merge_techniques=self.config.evomerge_techniques,
                    output_dir=str(self.config.output_dir / "evomerge"),
                    device=self.config.device,
                )
                phases.append(("EvoMergePhase", EvoMergePhase(evomerge_config)))

            if self.config.enable_quietstar:
                from .phases.quietstar import QuietSTaRConfig, QuietSTaRPhase

                quietstar_config = QuietSTaRConfig(
                    model_path="",  # Will be set from previous phase output
                    output_path=str(self.config.output_dir / "quietstar"),
                    max_thought_length=self.config.quietstar_thought_length,
                    convergence_threshold=0.95,
                    thought_probability=0.5,
                    learning_rate=1e-5,
                    num_epochs=3,
                )
                phases.append(("QuietSTaRPhase", QuietSTaRPhase(quietstar_config)))

            if self.config.enable_initial_compression:
                from .phases.bitnet_compression import BitNetCompressionPhase, BitNetCompressionConfig

                bitnet_config = BitNetCompressionConfig(
                    model_path="",  # Will be set from previous phase output
                    output_path=str(self.config.output_dir / "bitnet"),
                    quantization_bits=self.config.bitnet_bits,
                    enable_grokfast=self.config.grokfast_enabled,
                    device=self.config.device,
                    calibration_samples=100,
                    enable_fine_tuning=True,
                )
                phases.append(("BitNetCompressionPhase", BitNetCompressionPhase(bitnet_config)))

            if self.config.enable_training:
                from .phases.forge_training import ForgeTrainingConfig, ForgeTrainingPhase

                training_config = ForgeTrainingConfig(
                    model_path="",  # Will be set from previous phase output
                    output_path=str(self.config.output_dir / "forge_training"),
                    max_steps=self.config.training_steps,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    enable_grokfast=self.config.grokfast_enabled,
                    grokfast_ema_alpha=self.config.grokfast_ema_alpha,
                    grokfast_lambda_init=self.config.grokfast_lambda_init,
                )
                phases.append(("ForgeTrainingPhase", ForgeTrainingPhase(training_config)))

            if self.config.enable_tool_baking:
                from .phases.tool_persona_baking import ToolPersonaBakingConfig, ToolPersonaBakingPhase

                toolbaking_config = ToolPersonaBakingConfig(
                    model_path="",  # Will be set from previous phase output
                    output_path=str(self.config.output_dir / "tool_persona_baking"),
                    enable_grokfast=self.config.grokfast_enabled,
                    grokfast_ema_alpha=self.config.grokfast_ema_alpha,
                    grokfast_lambda_init=self.config.grokfast_lambda_init,
                    baking_iterations=5,
                    convergence_threshold=0.90,
                )
                phases.append(("ToolPersonaBakingPhase", ToolPersonaBakingPhase(toolbaking_config)))

            if self.config.enable_adas:
                from .phases.adas import ADASConfig, ADASPhase

                adas_config = ADASConfig(
                    population_size=20,
                    num_generations=self.config.adas_iterations,
                    composition_scale=self.config.adas_vector_composition_scale,
                    enable_grokfast_training=self.config.grokfast_enabled,
                    grokfast_ema_alpha=self.config.grokfast_ema_alpha,
                    grokfast_lambda=self.config.grokfast_lambda_init,
                )
                phases.append(("ADASPhase", ADASPhase(adas_config)))

            if self.config.enable_final_compression:
                from .phases.final_compression import FinalCompressionConfig, FinalCompressionPhase

                compression_config = FinalCompressionConfig(
                    enable_seedlm=True,
                    enable_vptq=True,
                    enable_hypercompression=True,
                    seedlm_seed_ratio=self.config.seedlm_seed_ratio,
                    vptq_codebook_size=self.config.vptq_codebook_size,
                    hyper_compression_ratio=self.config.hypercompression_ratio,
                    enable_grokfast_optimization=self.config.grokfast_enabled,
                )
                phases.append(("FinalCompressionPhase", FinalCompressionPhase(compression_config)))

        except ImportError as e:
            self.logger.error(f"Failed to import phase module: {e}")
            # Continue with available phases

        # Validate phase compatibility
        if not self.orchestrator.validate_phase_compatibility(phases):
            raise ValueError("Phase sequence is not compatible")

        self.logger.info(f"Initialized {len(phases)} phases: {[name for name, _ in phases]}")
        return phases

    async def run_pipeline(self, resume_from: str | None = None) -> PhaseResult:
        """
        Run the complete Agent Forge pipeline using phase orchestrator.

        Args:
            resume_from: Optional phase name to resume from

        Returns:
            Final PhaseResult with the completed model
        """
        start_time = time.time()
        self.logger.info("Starting Agent Forge Unified Pipeline")

        # Initialize tracking
        if self.config.wandb_project:
            self._init_wandb()

        try:
            # Determine starting model and phases
            if resume_from:
                initial_model = self._load_checkpoint(resume_from)
                # Filter phases to start from resume point
                phases_to_run = self._get_phases_from_resume_point(resume_from)
            else:
                # Create initial model (from base models or scratch)
                initial_model = self._create_initial_model()
                phases_to_run = self.phases

            # Run phase sequence using orchestrator
            self.logger.info(f"Running {len(phases_to_run)} phases")
            self.phase_results = await self.orchestrator.run_phase_sequence(phases_to_run, initial_model)

            # Check if pipeline completed successfully
            if not self.phase_results:
                raise RuntimeError("No phases were executed")

            final_result = self.phase_results[-1]
            if not final_result.success:
                raise RuntimeError(f"Pipeline failed at {final_result.phase_name}: {final_result.error}")

            # Save final checkpoint
            self._save_checkpoint("final_pipeline", final_result.model)

            # Generate comprehensive report
            self._generate_final_report()

            duration = time.time() - start_time

            # Create final pipeline result
            pipeline_result = PhaseResult(
                success=True,
                model=final_result.model,
                phase_name="UnifiedPipeline",
                metrics={
                    "total_duration_seconds": duration,
                    "phases_completed": len([r for r in self.phase_results if r.success]),
                    "total_phases": len(self.phase_results),
                    "phase_metrics": {r.phase_name: r.metrics for r in self.phase_results if r.success},
                    "pipeline_success_rate": len([r for r in self.phase_results if r.success])
                    / len(self.phase_results),
                },
                artifacts={
                    "phase_results": [
                        {
                            "phase_name": r.phase_name,
                            "success": r.success,
                            "duration": r.duration_seconds,
                            "metrics": r.metrics,
                            "error": r.error,
                        }
                        for r in self.phase_results
                    ],
                    "config": self.config.__dict__,
                },
                duration_seconds=duration,
            )

            self.logger.info(
                f"Agent Forge pipeline completed successfully in {duration:.1f}s. "
                f"Completed {pipeline_result.metrics['phases_completed']}/{pipeline_result.metrics['total_phases']} phases"
            )

            return pipeline_result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Agent Forge pipeline failed: {str(e)}"
            self.logger.error(error_msg)

            # Return failure result with any completed phases
            return PhaseResult(
                success=False,
                model=self.phase_results[-1].model if self.phase_results else None,
                phase_name="UnifiedPipeline",
                error=error_msg,
                metrics={
                    "total_duration_seconds": duration,
                    "phases_completed": len([r for r in self.phase_results if r.success]),
                    "total_phases": len(self.phase_results) if self.phase_results else 0,
                },
                artifacts={
                    "partial_phase_results": (
                        [
                            {"phase_name": r.phase_name, "success": r.success, "error": r.error}
                            for r in self.phase_results
                        ]
                        if self.phase_results
                        else []
                    )
                },
                duration_seconds=duration,
            )

    def _create_initial_model(self) -> nn.Module | None:
        """
        Create initial model for pipeline start.

        With Cognate phase enabled, returns None to let Cognate create the model.
        Otherwise creates a dummy model for backward compatibility.
        """
        if self.config.enable_cognate:
            self.logger.info("Cognate phase enabled - will create model from base models")
            return None  # Cognate phase will handle model creation

        # Fallback: create dummy model if Cognate is disabled
        self.logger.info("Creating fallback initial model for pipeline")

        # Create a simple transformer-like model as starting point
        model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))

        # Add some basic config attributes expected by phases
        model.config = type(
            "Config",
            (),
            {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 50257,
                "max_position_embeddings": 2048,
            },
        )()

        self.logger.info("Initial model created")
        return model

    def _get_phases_from_resume_point(self, resume_from: str) -> list[tuple[str, PhaseController]]:
        """Get phases to run starting from resume point."""
        phase_names = [name for name, _ in self.phases]

        if resume_from not in phase_names:
            self.logger.warning(f"Resume point '{resume_from}' not found, starting from beginning")
            return self.phases

        # Find index and return phases from that point onward
        resume_index = phase_names.index(resume_from)
        phases_to_run = self.phases[resume_index:]

        self.logger.info(f"Resuming from {resume_from}, running {len(phases_to_run)} remaining phases")
        return phases_to_run

    def _save_checkpoint(self, phase_name: str, model: nn.Module):
        """Save checkpoint for a phase."""
        checkpoint_path = self.config.checkpoint_dir / f"{phase_name}_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "phase": phase_name,
                "config": self.config,
                "timestamp": datetime.now().isoformat(),
            },
            checkpoint_path,
        )
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _load_checkpoint(self, phase_name: str) -> nn.Module:
        """Load checkpoint from a phase."""
        checkpoint_path = self.config.checkpoint_dir / f"{phase_name}_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path)
        # Model reconstruction would happen here
        self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return checkpoint["model"]

    def _get_phase_index(self, phase_name: str) -> int:
        """Get the index of a phase."""
        phases = ["evomerge", "quietstar", "bitnet", "training", "tool_baking", "adas", "final"]
        return phases.index(phase_name)

    def _aggregate_metrics(self) -> dict[str, Any]:
        """Aggregate metrics from all phases."""
        metrics = {}
        for result in self.phase_results:
            metrics[result.phase_name] = result.metrics
        return metrics

    def _generate_final_report(self):
        """Generate a comprehensive report of the pipeline run."""
        report = {
            "pipeline_config": self.config.__dict__,
            "phase_results": [
                {"phase": r.phase_name, "metrics": r.metrics, "timestamp": r.timestamp.isoformat()}
                for r in self.phase_results
            ],
            "final_metrics": self._aggregate_metrics(),
        }

        report_path = self.config.output_dir / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Pipeline report saved to: {report_path}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__,
                tags=["agent_forge", "unified-pipeline"],
            )
        except ImportError:
            self.logger.warning("W&B not installed, skipping tracking")


# CLI integration
def create_pipeline(config_path: str | None = None, **kwargs) -> UnifiedPipeline:
    """
    Create a unified pipeline instance.

    Args:
        config_path: Optional path to configuration file
        **kwargs: Override configuration parameters

    Returns:
        Configured UnifiedPipeline instance
    """
    if config_path:
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = UnifiedConfig(**config_dict)
    else:
        config = UnifiedConfig(**kwargs)

    return UnifiedPipeline(config)
