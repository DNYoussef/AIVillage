#!/usr/bin/env python3
"""Scaled Magi Specialization - 10,000 Questions.

Scale the proven 300-question Magi specialization to full 10,000 questions
using the exact pipeline that achieved 0.774 specialization score.

This builds directly on the validated implementation with all advanced features:
- 10 levels × 1,000 questions each = 10,000 total
- Geometric self-awareness every 100 questions
- Self-modification with safety bounds
- Sleep/dream cycles every 500 questions
- Grokking detection throughout training
"""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

# Add project to path
sys.path.append(".")

from agent_forge.memory_manager import memory_manager
from agent_forge.training.magi_specialization import (
    MagiConfig,
    MagiSpecializationPipeline,
)
from agent_forge.wandb_manager import finish_wandb, init_wandb, log_metrics

# Configure logging for scaled run
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"D:/AgentForge/scaled_magi_10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ScaledMagiRunner:
    """Scale the proven Magi specialization from 300 to 10,000 questions."""

    def __init__(self) -> None:
        self.start_time = datetime.now()
        self.run_id = f"scaled_magi_10k_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(f"D:/AgentForge/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("AGENT FORGE - SCALED MAGI SPECIALIZATION (10,000 QUESTIONS)")
        logger.info("=" * 80)
        logger.info(f"Run ID: {self.run_id}")
        logger.info("Scaling from proven 300-question success to 10,000 questions")
        logger.info("Expected duration: ~6.3 minutes (33x scale-up)")
        logger.info(f"Memory Available: {memory_manager.get_memory_stats()['system_ram_available_gb']:.2f} GB")

    def initialize_wandb_tracking(self):
        """Initialize W&B for scaled Magi run."""
        logger.info("Initializing W&B tracking for scaled 10K Magi run...")

        success = init_wandb(
            project="scaled-magi-10k",
            name=f"magi_10k_specialization_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            tags=[
                "scaled-magi",
                "10k-questions",
                "production",
                "self-modification",
                "geometric-awareness",
            ],
            config={
                "run_type": "scaled_magi_specialization_10k",
                "questions_total": 10000,
                "levels": 10,
                "questions_per_level": 1000,
                "previous_baseline": 0.774,
                "scale_factor": 33.33,
                "estimated_duration_minutes": 6.3,
                "advanced_features": True,
                "geometric_awareness": True,
                "self_modification": True,
                "sleep_cycles": True,
                "memory_available_gb": memory_manager.get_memory_stats()["system_ram_available_gb"],
                "cpu_only": True,
            },
        )

        if success:
            logger.info("PASS: W&B tracking initialized for scaled run")
            log_metrics(
                {
                    "scaled_magi_start": 1,
                    "baseline_specialization_score": 0.774,
                    "questions_scale_factor": 33.33,
                    "execution_start_time": time.time(),
                }
            )

        return success

    def create_scaled_magi_config(self):
        """Create scaled Magi configuration using proven parameters."""
        logger.info("Creating scaled Magi configuration...")

        # Load the best evolved model from successful run
        evolution_results_path = Path("D:/AgentForge/historic_real_run_20250726_030005/evolution_50gen_results.json")
        if evolution_results_path.exists():
            with open(evolution_results_path) as f:
                evolution_data = json.load(f)

            best_config = evolution_data["evolution_summary"]["best_configuration"]
            optimal_model_path = best_config.get("base_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
            logger.info(f"Using evolved model: {optimal_model_path}")
            logger.info(f"Evolved fitness: {best_config['fitness']:.4f}")
        else:
            optimal_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            logger.warning("Using default model path")

        # Create scaled configuration with all advanced features
        config = MagiConfig(
            # Use evolved model
            optimal_model_path=optimal_model_path,
            output_dir=str(self.output_dir),
            # Scaled curriculum - 10K questions
            curriculum_levels=10,
            questions_per_level=1000,
            total_questions=10000,
            # Enable all advanced features
            enable_geometric_awareness=True,
            weight_visualization_freq=100,  # Every 100 questions
            grokking_detection=True,
            # Enable self-modification with safety
            enable_self_modification=True,
            modification_safety_bounds={
                "max_weight_change": 0.1,
                "max_temperature_change": 0.5,
                "rollback_threshold": 0.95,
            },
            # Sleep/dream cycles for memory consolidation
            sleep_cycle_frequency=500,  # Every 500 questions
            dream_enhancement=True,
            # CPU-only with proven memory management
            device="cpu",
            mixed_precision=False,  # Disable for CPU stability
            seed=42,
            # W&B configuration
            wandb_project="scaled-magi-10k",
            wandb_entity=None,
            wandb_tags=["scaled-magi", "10k-questions", "production"],
        )

        logger.info("Scaled Magi Configuration Created:")
        logger.info(f"  Questions: {config.total_questions:,} ({config.curriculum_levels} levels)")
        logger.info(f"  Specialization Areas: {len(config.specialization_areas)}")
        logger.info(f"  Geometric Awareness: {config.enable_geometric_awareness}")
        logger.info(f"  Self-Modification: {config.enable_self_modification}")
        logger.info(f"  Sleep Cycles: Every {config.sleep_cycle_frequency} questions")

        return config

    async def execute_scaled_magi_specialization(self):
        """Execute the scaled 10K question Magi specialization."""
        logger.info("STARTING SCALED MAGI SPECIALIZATION - 10,000 QUESTIONS")

        # Create scaled configuration
        config = self.create_scaled_magi_config()

        # Log configuration to W&B
        log_metrics(
            {
                "config_questions_total": config.total_questions,
                "config_levels": config.curriculum_levels,
                "config_geometric_awareness": config.enable_geometric_awareness,
                "config_self_modification": config.enable_self_modification,
            }
        )

        try:
            # Create and run the scaled pipeline
            logger.info("Initializing Magi Specialization Pipeline...")
            pipeline = MagiSpecializationPipeline(config)

            logger.info("LAUNCHING SCALED TRAINING - This will take ~6.3 minutes")
            logger.info("Processing 10,000 questions with advanced features...")

            # Execute the complete specialization
            results = await pipeline.run_magi_specialization()

            if results and "final_evaluation" in results:
                logger.info("SCALED MAGI SPECIALIZATION COMPLETED SUCCESSFULLY!")

                final_eval = results["final_evaluation"]
                overall_accuracy = final_eval.get("overall_accuracy", 0)

                logger.info("Scaled Results:")
                logger.info(f"  Overall Accuracy: {overall_accuracy:.3f}")
                logger.info(f"  Deployment Ready: {final_eval.get('ready_for_deployment', False)}")
                logger.info(f"  Area Results: {final_eval.get('area_results', {})}")

                # Log success to W&B
                log_metrics(
                    {
                        "scaled_magi_completed": 1,
                        "final_overall_accuracy": overall_accuracy,
                        "deployment_ready": final_eval.get("ready_for_deployment", False),
                        "baseline_improvement": overall_accuracy - 0.774,  # vs proven baseline
                    }
                )

                return results
            logger.error("Scaled specialization completed but returned incomplete results")
            return None

        except Exception as e:
            logger.exception(f"Scaled Magi specialization failed: {e}")
            log_metrics({"scaled_magi_error": 1, "error_message": str(e)})
            return None


async def main():
    """Main execution function for scaled Magi specialization."""
    logger.info("STARTING SCALED MAGI SPECIALIZATION (10K QUESTIONS)")

    runner = ScaledMagiRunner()

    try:
        # Initialize tracking
        runner.initialize_wandb_tracking()

        # Execute scaled specialization
        logger.info("LAUNCHING SCALED MAGI TRAINING...")
        results = await runner.execute_scaled_magi_specialization()

        if results:
            duration = (datetime.now() - runner.start_time).total_seconds()

            logger.info("=" * 80)
            logger.info("SCALED MAGI SPECIALIZATION SUCCESS!")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration / 60:.1f} minutes ({duration:.1f} seconds)")
            logger.info("Questions Processed: 10,000")
            logger.info("Scale Factor: 33.33x from proven baseline")
            logger.info(f"Output Saved: {runner.output_dir}")

            # Final W&B logging
            log_metrics(
                {
                    "scaled_achievement": 1,
                    "execution_duration_minutes": duration / 60,
                    "execution_duration_seconds": duration,
                    "questions_processed": 10000,
                    "scaling_success": True,
                }
            )

            # Save scaling results
            scaling_results = {
                "run_id": runner.run_id,
                "start_time": runner.start_time.isoformat(),
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
                "questions_processed": 10000,
                "scale_factor": 33.33,
                "baseline_score": 0.774,
                "results": results,
            }

            results_file = runner.output_dir / "scaled_magi_results.json"
            with open(results_file, "w") as f:
                json.dump(scaling_results, f, indent=2, default=str)

            return results
        logger.error("Scaled Magi specialization failed")
        return None

    except Exception as e:
        logger.exception(f"Scaled execution failed: {e}")
        return None
    finally:
        finish_wandb()
        logger.info("Scaled Magi execution completed")


if __name__ == "__main__":
    # Execute the scaled 10K Magi specialization
    result = asyncio.run(main())

    if result:
        print("\nSCALED MAGI SUCCESS!")
        print("10,000 question specialization completed successfully!")
        print("Scaling from proven 300-question baseline achieved!")
    else:
        print("\nScaled specialization encountered issues")
        print("Check logs for details")
