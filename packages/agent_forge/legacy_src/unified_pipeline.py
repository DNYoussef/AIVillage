#!/usr/bin/env python3
"""Agent Forge Unified Pipeline.

Complete end-to-end workflow that integrates all phases:
1. EvoMerge: Evolutionary model optimization (50 generations)
2. Quiet-STaR: Reasoning enhancement with thought injection
3. BitNet Compression: Production-ready model compression

This creates a fully optimized, reasoning-enhanced, and compressed model
ready for deployment.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import wandb
from pydantic import BaseModel, Field

from production.compression.compression_pipeline import CompressionConfig, CompressionPipeline

# Import pipeline components
from production.evolution.evomerge_pipeline import EvolutionConfig, EvoMergePipeline

from .quietstar_baker import QuietSTaRBaker, QuietSTaRConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Unified Configuration
# ============================================================================


class UnifiedPipelineConfig(BaseModel):
    """Configuration for the complete Agent Forge pipeline."""

    # Pipeline control
    enable_evomerge: bool = Field(default=True, description="Run evolutionary merging")
    enable_quietstar: bool = Field(default=True, description="Run Quiet-STaR reasoning enhancement")
    enable_compression: bool = Field(default=True, description="Run BitNet compression")

    # Output configuration
    base_output_dir: Path = Field(default=Path("./unified_pipeline_output"))
    final_model_name: str = Field(default="agent_forge_final")

    # EvoMerge configuration
    evomerge_config: dict | None = Field(default=None)
    evomerge_base_models: list[str] = Field(
        default_factory=lambda: [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
            "Qwen/Qwen2-1.5B-Instruct",
        ]
    )
    evomerge_generations: int = Field(default=50, ge=1, le=200)

    # Quiet-STaR configuration
    quietstar_config: dict | None = Field(default=None)
    quietstar_eval_samples: int = Field(default=100, ge=10, le=500)
    quietstar_ab_rounds: int = Field(default=3, ge=1, le=10)

    # Compression configuration
    compression_config: dict | None = Field(default=None)
    compression_calibration_samples: int = Field(default=1000, ge=100, le=10000)

    # System configuration
    device: str = Field(default="auto")
    max_memory_gb: float = Field(default=8.0, gt=0.0)

    # W&B configuration
    wandb_project: str = Field(default="agent-forge")
    wandb_entity: str | None = None
    wandb_tags: list[str] = Field(default_factory=lambda: ["unified", "end-to-end"])

    # Resume configuration
    resume_from_phase: str | None = Field(default=None, description="Resume from specific phase")
    checkpoint_dir: Path = Field(default=Path("./unified_checkpoints"))

    def __post_init__(self):
        """Create output directories."""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pipeline State Management
# ============================================================================


class PipelineState(BaseModel):
    """Tracks the current state of the unified pipeline."""

    run_id: str
    start_time: datetime
    current_phase: str = "not_started"
    completed_phases: list[str] = Field(default_factory=list)

    # Model paths
    evomerge_model_path: str | None = None
    quietstar_model_path: str | None = None
    final_model_path: str | None = None

    # Phase results
    evomerge_results: dict | None = None
    quietstar_results: dict | None = None
    compression_results: dict | None = None

    # Performance metrics
    final_performance: dict[str, float] = Field(default_factory=dict)
    total_improvement: float = 0.0
    compression_ratio: float = 1.0

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """Save pipeline state checkpoint."""
        checkpoint_path = checkpoint_dir / f"unified_pipeline_{self.run_id}.json"

        with open(checkpoint_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

        logger.info(f"Pipeline checkpoint saved: {checkpoint_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "PipelineState":
        """Load pipeline state from checkpoint."""
        with open(checkpoint_path) as f:
            data = json.load(f)

        # Convert string dates back to datetime
        if isinstance(data.get("start_time"), str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])

        return cls(**data)


# ============================================================================
# Unified Pipeline
# ============================================================================


class UnifiedPipeline:
    """Main unified pipeline orchestrator."""

    def __init__(self, config: UnifiedPipelineConfig) -> None:
        self.config = config
        self.state = PipelineState(
            run_id=f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
        )
        self.wandb_run = None

        logger.info(f"Unified pipeline initialized - Run ID: {self.state.run_id}")

    def initialize_wandb(self) -> None:
        """Initialize W&B tracking for unified pipeline."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                job_type="unified_pipeline",
                tags=[*self.config.wandb_tags, f"run-{self.state.run_id}"],
                config=self.config.model_dump(),
                name=f"unified_pipeline_{self.state.run_id}",
            )

            logger.info(f"W&B initialized: {self.wandb_run.url}")

        except Exception as e:
            logger.exception(f"W&B initialization failed: {e}")
            self.wandb_run = None

    async def run_complete_pipeline(self) -> dict[str, Any]:
        """Run the complete end-to-end pipeline."""
        try:
            # Initialize W&B
            self.initialize_wandb()

            # Phase 1: EvoMerge (if enabled)
            if self.config.enable_evomerge and "evomerge" not in self.state.completed_phases:
                await self.run_evomerge_phase()
                self.state.completed_phases.append("evomerge")
                self.state.save_checkpoint(self.config.checkpoint_dir)

            # Phase 2: Quiet-STaR (if enabled)
            if self.config.enable_quietstar and "quietstar" not in self.state.completed_phases:
                if not self.state.evomerge_model_path:
                    msg = "EvoMerge model path required for Quiet-STaR phase"
                    raise ValueError(msg)

                await self.run_quietstar_phase()
                self.state.completed_phases.append("quietstar")
                self.state.save_checkpoint(self.config.checkpoint_dir)

            # Phase 3: Compression (if enabled)
            if self.config.enable_compression and "compression" not in self.state.completed_phases:
                source_model = self.state.quietstar_model_path or self.state.evomerge_model_path
                if not source_model:
                    msg = "Source model path required for compression phase"
                    raise ValueError(msg)

                await self.run_compression_phase(source_model)
                self.state.completed_phases.append("compression")
                self.state.save_checkpoint(self.config.checkpoint_dir)

            # Calculate final metrics
            await self.calculate_final_metrics()

            # Generate final report
            return await self.generate_final_report()

        except Exception as e:
            logger.exception(f"Unified pipeline failed: {e}")
            raise

        finally:
            if self.wandb_run:
                self.wandb_run.finish()

    async def run_evomerge_phase(self) -> None:
        """Run EvoMerge evolutionary optimization."""
        logger.info("🧬 Starting EvoMerge Phase")
        self.state.current_phase = "evomerge"

        # Configure EvoMerge
        if self.config.evomerge_config:
            evomerge_config = EvolutionConfig(**self.config.evomerge_config)
        else:
            from production.evolution.evomerge_pipeline import BaseModelConfig

            evomerge_config = EvolutionConfig(
                base_models=[
                    BaseModelConfig(name="deepseek", path=self.config.evomerge_base_models[0]),
                    BaseModelConfig(name="nemotron", path=self.config.evomerge_base_models[1]),
                    BaseModelConfig(name="qwen2", path=self.config.evomerge_base_models[2]),
                ],
                max_generations=self.config.evomerge_generations,
                device=self.config.device,
                output_dir=self.config.base_output_dir / "evomerge",
                wandb_project=self.config.wandb_project,
            )

        # Run EvoMerge
        pipeline = EvoMergePipeline(evomerge_config)
        best_candidate = await pipeline.run_evolution()

        if best_candidate and best_candidate.model_path:
            self.state.evomerge_model_path = best_candidate.model_path
            self.state.evomerge_results = {
                "best_fitness": best_candidate.overall_fitness,
                "generation": best_candidate.generation,
                "fitness_scores": best_candidate.fitness_scores,
                "model_path": best_candidate.model_path,
            }

            # Log to unified W&B
            if self.wandb_run:
                self.wandb_run.log(
                    {
                        "evomerge_best_fitness": best_candidate.overall_fitness,
                        "evomerge_generations": best_candidate.generation,
                        "evomerge_code_score": best_candidate.fitness_scores.get("code", 0),
                        "evomerge_math_score": best_candidate.fitness_scores.get("math", 0),
                    }
                )

            logger.info(f"✅ EvoMerge completed - Best fitness: {best_candidate.overall_fitness:.3f}")
        else:
            msg = "EvoMerge failed to produce a best candidate"
            raise RuntimeError(msg)

    async def run_quietstar_phase(self) -> None:
        """Run Quiet-STaR reasoning enhancement."""
        logger.info("🤔 Starting Quiet-STaR Phase")
        self.state.current_phase = "quietstar"

        # Configure Quiet-STaR
        if self.config.quietstar_config:
            quietstar_config = QuietSTaRConfig(**self.config.quietstar_config)
        else:
            quietstar_config = QuietSTaRConfig(
                model_path=self.state.evomerge_model_path,
                output_path=str(self.config.base_output_dir / "quietstar"),
                eval_samples=self.config.quietstar_eval_samples,
                ab_test_rounds=self.config.quietstar_ab_rounds,
                device=self.config.device,
                wandb_project=self.config.wandb_project,
            )

        # Run Quiet-STaR
        baker = QuietSTaRBaker(quietstar_config)
        results = await baker.run_baking_pipeline()

        self.state.quietstar_model_path = quietstar_config.output_path
        self.state.quietstar_results = results

        # Log to unified W&B
        if self.wandb_run:
            self.wandb_run.log(
                {
                    "quietstar_winner": results["winner"],
                    "quietstar_improvement": results["improvement"],
                    "quietstar_baseline_accuracy": results["ab_test_results"]["baseline_accuracy"],
                    "quietstar_thoughts_accuracy": results["ab_test_results"]["thoughts_accuracy"],
                }
            )

        logger.info(
            f"✅ Quiet-STaR completed - Winner: {results['winner']}, Improvement: {results['improvement']:.1f}%"
        )

    async def run_compression_phase(self, source_model_path: str) -> None:
        """Run BitNet compression."""
        logger.info("🗜️ Starting Compression Phase")
        self.state.current_phase = "compression"

        # Configure compression
        if self.config.compression_config:
            compression_config = CompressionConfig(**self.config.compression_config)
        else:
            compression_config = CompressionConfig(
                input_model_path=source_model_path,
                output_model_path=str(self.config.base_output_dir / "compressed"),
                calibration_samples=self.config.compression_calibration_samples,
                device=self.config.device,
                wandb_project=self.config.wandb_project,
            )

        # Run compression
        pipeline = CompressionPipeline(compression_config)
        results = await pipeline.run_compression_pipeline()

        self.state.final_model_path = compression_config.output_model_path
        self.state.compression_results = results
        self.state.compression_ratio = results["compression_ratio"]

        # Log to unified W&B
        if self.wandb_run:
            self.wandb_run.log(
                {
                    "compression_ratio": results["compression_ratio"],
                    "compression_savings_mb": results["memory_savings_mb"],
                    "compression_success": results["success"],
                }
            )

        logger.info(f"✅ Compression completed - Ratio: {results['compression_ratio']:.1f}x")

    async def calculate_final_metrics(self) -> None:
        """Calculate final performance metrics."""
        logger.info("📊 Calculating final metrics")

        # Calculate total improvement
        base_performance = 0.5  # Assume baseline
        final_performance = base_performance

        # Apply EvoMerge improvement
        if self.state.evomerge_results:
            evomerge_improvement = self.state.evomerge_results["best_fitness"] - base_performance
            final_performance += evomerge_improvement

        # Apply Quiet-STaR improvement
        if self.state.quietstar_results:
            quietstar_improvement = self.state.quietstar_results["improvement"] / 100
            final_performance += quietstar_improvement

        self.state.total_improvement = ((final_performance - base_performance) / base_performance) * 100

        # Final performance breakdown
        self.state.final_performance = {
            "base_performance": base_performance,
            "final_performance": final_performance,
            "total_improvement_percent": self.state.total_improvement,
            "evomerge_contribution": (
                self.state.evomerge_results["best_fitness"] if self.state.evomerge_results else 0
            ),
            "quietstar_contribution": (
                self.state.quietstar_results["improvement"] if self.state.quietstar_results else 0
            ),
            "compression_ratio": self.state.compression_ratio,
        }

        # Log final metrics
        if self.wandb_run:
            self.wandb_run.log(
                {
                    "final_total_improvement": self.state.total_improvement,
                    "final_compression_ratio": self.state.compression_ratio,
                    "final_model_ready": True,
                }
            )

    async def generate_final_report(self) -> dict[str, Any]:
        """Generate comprehensive final report."""
        end_time = datetime.now()
        total_duration = (end_time - self.state.start_time).total_seconds()

        report = {
            "pipeline_summary": {
                "run_id": self.state.run_id,
                "start_time": self.state.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_hours": total_duration / 3600,
                "completed_phases": self.state.completed_phases,
                "final_model_path": self.state.final_model_path,
            },
            "performance_summary": {
                "total_improvement": f"{self.state.total_improvement:.1f}%",
                "compression_ratio": f"{self.state.compression_ratio:.1f}x",
                "evomerge_fitness": (self.state.evomerge_results["best_fitness"] if self.state.evomerge_results else 0),
                "quietstar_improvement": (
                    f"{self.state.quietstar_results['improvement']:.1f}%" if self.state.quietstar_results else "0%"
                ),
                "final_model_size": ("Compressed" if self.state.compression_ratio > 1 else "Original"),
            },
            "phase_details": {
                "evomerge": self.state.evomerge_results,
                "quietstar": self.state.quietstar_results,
                "compression": self.state.compression_results,
            },
            "final_performance": self.state.final_performance,
            "deployment_ready": bool(self.state.final_model_path),
        }

        # Save report
        report_path = self.config.base_output_dir / f"unified_pipeline_report_{self.state.run_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Log to W&B as artifact
        if self.wandb_run:
            artifact = wandb.Artifact(f"unified_pipeline_report_{self.state.run_id}", type="report")
            artifact.add_file(str(report_path))
            self.wandb_run.log_artifact(artifact)

            # Also log final model if available
            if self.state.final_model_path:
                model_artifact = wandb.Artifact(
                    f"agent_forge_final_model_{self.state.run_id}",
                    type="model",
                    description=(
                        f"Complete Agent Forge model with {self.state.total_improvement:.1f}% improvement and "
                        f"{self.state.compression_ratio:.1f}x compression"
                    ),
                )
                model_artifact.add_dir(self.state.final_model_path)
                self.wandb_run.log_artifact(model_artifact)

        # Print summary
        self.print_final_summary(report)

        return report

    def print_final_summary(self, report: dict) -> None:
        """Print comprehensive final summary."""
        print("\n" + "=" * 80)
        print("🎉 AGENT FORGE UNIFIED PIPELINE COMPLETE")
        print("=" * 80)

        summary = report["pipeline_summary"]
        performance = report["performance_summary"]

        print(f"📋 Run ID: {summary['run_id']}")
        print(f"⏱️  Total Time: {summary['total_duration_hours']:.1f} hours")
        print(f"✅ Completed Phases: {', '.join(summary['completed_phases'])}")

        print("\n🚀 PERFORMANCE RESULTS:")
        print(f"  📈 Total Improvement: {performance['total_improvement']}")
        print(f"  🗜️  Compression Ratio: {performance['compression_ratio']}")
        print(f"  🧬 EvoMerge Fitness: {performance['evomerge_fitness']:.3f}")
        print(f"  🤔 Quiet-STaR Boost: {performance['quietstar_improvement']}")

        print(f"\n💾 Final Model: {summary['final_model_path']}")
        print(f"🚢 Deployment Ready: {'Yes' if report['deployment_ready'] else 'No'}")

        print("\n🎯 NEXT STEPS:")
        if report["deployment_ready"]:
            print(f"  1. Load model from: {summary['final_model_path']}")
            print(f"  2. Deploy with {performance['compression_ratio']} memory efficiency")
            print(f"  3. Expect {performance['total_improvement']} better performance")
        else:
            print("  ❌ Pipeline incomplete - check logs for issues")

        print("=" * 80)


# ============================================================================
# CLI Interface
# ============================================================================


@click.group()
def forge() -> None:
    """Agent Forge CLI."""


@forge.command()
@click.option("--config", help="Configuration JSON file")
@click.option("--evomerge/--no-evomerge", default=True, help="Enable/disable EvoMerge phase")
@click.option("--quietstar/--no-quietstar", default=True, help="Enable/disable Quiet-STaR phase")
@click.option(
    "--compression/--no-compression",
    default=True,
    help="Enable/disable compression phase",
)
@click.option("--generations", default=50, help="EvoMerge generations")
@click.option("--output-dir", default="./unified_output", help="Base output directory")
@click.option("--device", default="auto", help="Device to use")
@click.option("--resume", help="Resume from checkpoint file")
def run_pipeline(config, evomerge, quietstar, compression, generations, output_dir, device, resume) -> None:
    """Run complete Agent Forge pipeline: EvoMerge → Quiet-STaR → Compression."""
    try:
        # Load configuration
        if config and Path(config).exists():
            with open(config) as f:
                config_data = json.load(f)
            pipeline_config = UnifiedPipelineConfig(**config_data)
        else:
            pipeline_config = UnifiedPipelineConfig(
                enable_evomerge=evomerge,
                enable_quietstar=quietstar,
                enable_compression=compression,
                evomerge_generations=generations,
                base_output_dir=Path(output_dir),
                device=device,
            )

        # Handle resume
        resume_state = None
        if resume and Path(resume).exists():
            logger.info(f"Resuming from checkpoint: {resume}")
            resume_path = Path(resume)
            resume_state = PipelineState.load_checkpoint(resume_path)
            completed = set(resume_state.completed_phases)

            if "evomerge" in completed:
                pipeline_config.enable_evomerge = False
            if "quietstar" in completed:
                pipeline_config.enable_quietstar = False
            if "compression" in completed:
                pipeline_config.enable_compression = False

            pipeline_config.checkpoint_dir = resume_path.parent

        # Run unified pipeline
        pipeline = UnifiedPipeline(pipeline_config)

        if resume_state:
            pipeline.state = resume_state

        logger.info("🚀 Starting Agent Forge Unified Pipeline")
        logger.info(
            f"Phases enabled: EvoMerge={pipeline.config.enable_evomerge}, "
            f"Quiet-STaR={pipeline.config.enable_quietstar}, "
            f"Compression={pipeline.config.enable_compression}"
        )

        results = asyncio.run(pipeline.run_complete_pipeline())

        print("\n🎉 Pipeline completed successfully!")
        print(f"📊 Check full report at: {results['pipeline_summary']['run_id']}")

    except Exception as e:
        logger.exception(f"Unified pipeline failed: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    forge()
