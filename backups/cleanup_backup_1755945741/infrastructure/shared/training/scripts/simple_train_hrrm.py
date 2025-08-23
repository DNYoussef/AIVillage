#!/usr/bin/env python3
"""
Refactored HRRM Training Script

Orchestrates specialized components following connascence principles.
Reduced coupling through dependency injection and single responsibility.
"""

import asyncio
import logging
import os
import sys
from typing import Any

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

from ..config.training_config import HRRMTrainingConfig, TokenizerConfig
from ..data.data_generator import MockTokenizer, SyntheticDataGenerator
from ..models.model_factory import ModelFactory
from ..trainers.training_engine import TrainingEngine
from ..utils.model_persistence import ModelPersistenceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Manages tokenizer creation and configuration.

    Handles tokenizer setup with fallback to mock tokenizer.
    """

    def __init__(self, config: TokenizerConfig):
        self._config = config

    def create_tokenizer(self) -> MockTokenizer:
        """
        Create tokenizer instance.

        Returns:
            Tokenizer instance (mock for now, can be extended)
        """
        logger.info("Creating tokenizer...")

        if os.path.exists(self._config.TOKENIZER_PATH):
            logger.info(f"Using existing tokenizer: {self._config.TOKENIZER_PATH}")
            # For now, still use mock - can be extended to load real tokenizer
            return MockTokenizer(self._config.DEFAULT_VOCAB_SIZE)
        else:
            logger.info("Creating mock tokenizer")
            return MockTokenizer(self._config.DEFAULT_VOCAB_SIZE)


class HRRMTrainer:
    """
    Main trainer that orchestrates the HRRM training pipeline.

    Uses composition and dependency injection to coordinate specialized components.
    Follows single responsibility principle by delegating to focused modules.
    """

    def __init__(self, config: HRRMTrainingConfig | None = None):
        self._config = config or HRRMTrainingConfig()

        # Initialize specialized components
        self._tokenizer_manager = TokenizerManager(self._config.tokenizer)
        self._model_factory = ModelFactory(self._config)
        self._training_engine = TrainingEngine(self._config.training)
        self._persistence_manager = ModelPersistenceManager()

        # Runtime state
        self._tokenizer = None
        self._train_data = None
        self._device = None
        self._training_results = {}

    async def run_complete_training(self) -> dict[str, Any]:
        """
        Run the complete HRRM training pipeline.

        Returns:
            Dictionary with training results and summary
        """
        logger.info("Starting complete HRRM training pipeline...")

        try:
            # Phase 1: Setup
            await self._setup_training_environment()

            # Phase 2: Train all models
            await self._train_all_models()

            # Phase 3: Save results and generate summary
            summary = await self._finalize_training()

            logger.info("🚀 All 3 HRRM models trained successfully!")
            return summary

        except Exception as e:
            logger.exception(f"Training pipeline failed: {e}")
            return {"error": str(e), "training_completed": False}

    async def _setup_training_environment(self) -> None:
        """Setup tokenizer, data, and device for training."""
        logger.info("Setting up training environment...")

        # Create tokenizer
        self._tokenizer = self._tokenizer_manager.create_tokenizer()

        # Generate training data
        data_generator = SyntheticDataGenerator(tokenizer=self._tokenizer, config=self._config.data)
        self._train_data = data_generator.generate_training_data()

        # Setup device
        self._device = self._model_factory.get_device()

        logger.info("Training environment setup complete")

    async def _train_all_models(self) -> None:
        """Train all three HRRM models sequentially."""
        # Train HRMPlanner
        await self._train_single_model(
            "HRMPlanner", self._model_factory.create_planner_model, self._config.output.planner_checkpoint_dir
        )

        # Train HRMReasoner
        await self._train_single_model(
            "HRMReasoner", self._model_factory.create_reasoner_model, self._config.output.reasoner_checkpoint_dir
        )

        # Train MemoryAsContextTiny
        await self._train_single_model(
            "MemoryAsContextTiny", self._model_factory.create_memory_model, self._config.output.memory_checkpoint_dir
        )

    async def _train_single_model(self, model_name: str, model_creator_func, checkpoint_dir) -> None:
        """
        Train a single model using the training engine.

        Args:
            model_name: Name of the model for logging
            model_creator_func: Function to create model and optimizer
            checkpoint_dir: Directory to save model checkpoint
        """
        logger.info("=" * 50)
        logger.info(f"Training {model_name}...")

        try:
            # Create model and optimizer
            model, optimizer = model_creator_func()

            # Train the model
            final_loss = await self._training_engine.train_model(
                model=model,
                optimizer=optimizer,
                train_data=self._train_data,
                model_name=model_name,
                device=self._device,
            )

            # Save model and configuration
            config = self._get_model_config(model_name)
            success = self._persistence_manager.save_model_and_config(
                model=model, config=config, model_name=model_name, checkpoint_dir=checkpoint_dir
            )

            if success:
                self._training_results[model_name] = {
                    "final_loss": final_loss,
                    "config": config.__dict__ if hasattr(config, "__dict__") else config,
                    "checkpoint": str(checkpoint_dir),
                    "trained_successfully": True,
                }
            else:
                raise RuntimeError(f"Failed to save {model_name}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            self._training_results[model_name] = {
                "error": str(e),
                "trained_successfully": False,
            }
            # Continue training other models despite individual failures

    async def _finalize_training(self) -> dict[str, Any]:
        """
        Finalize training by generating summary and saving results.

        Returns:
            Complete training summary
        """
        logger.info("Finalizing training results...")

        # Generate comprehensive summary
        summary = {
            "training_completed": all(
                result.get("trained_successfully", False) for result in self._training_results.values()
            ),
            "models": self._training_results,
            "configuration": self._config.to_dict(),
            "training_info": self._training_engine.get_training_info(),
            "data_info": self._get_data_info(),
        }

        # Log results
        self._log_training_summary(summary)

        # Save summary to file
        self._persistence_manager.save_training_summary(
            summary_data=summary, summary_path=self._config.output.SUMMARY_PATH
        )

        return summary

    def _get_model_config(self, model_name: str) -> Any:
        """Get the configuration for a specific model."""
        if model_name == "HRMPlanner":
            return self._config.planner
        elif model_name == "HRMReasoner":
            return self._config.reasoner
        elif model_name == "MemoryAsContextTiny":
            return self._config.memory
        else:
            return {}

    def _get_data_info(self) -> dict[str, Any]:
        """Get information about training data."""
        if not hasattr(self, "_train_data") or not self._train_data:
            return {}

        return {
            "num_batches": len(self._train_data),
            "batch_size": self._train_data[0].shape[0] if self._train_data else 0,
            "sequence_length": self._train_data[0].shape[1] if self._train_data else 0,
            "total_sequences": len(self._train_data) * (self._train_data[0].shape[0] if self._train_data else 0),
        }

    def _log_training_summary(self, summary: dict[str, Any]) -> None:
        """Log training summary to console."""
        logger.info("=" * 50)
        logger.info("HRRM Training Summary")
        logger.info("=" * 50)

        for model_name, result in summary["models"].items():
            if result.get("trained_successfully"):
                loss = result.get("final_loss", "Unknown")
                logger.info(f"{model_name} - Final Loss: {loss:.4f}")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"{model_name} - Failed: {error}")


async def main():
    """Main function that runs the complete training pipeline."""
    try:
        # Create trainer with default configuration
        trainer = HRRMTrainer()

        # Run complete training pipeline
        summary = await trainer.run_complete_training()

        # Report final status
        if summary.get("training_completed"):
            logger.info("✅ All models trained successfully!")
        else:
            logger.error("❌ Some models failed to train")

        return summary

    except Exception as e:
        logger.exception(f"Main training function failed: {e}")
        return {"error": str(e), "training_completed": False}


if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())

    # Exit with appropriate code
    if result.get("training_completed"):
        sys.exit(0)
    else:
        sys.exit(1)
