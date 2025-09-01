"""
Training Service Usage Example

Demonstrates how to use the extracted TrainingService with mock implementations
for testing and development purposes.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the infrastructure path for imports
sys.path.append(str(Path(__file__).parent.parent / "infrastructure" / "gateway" / "services"))

from training_service import TrainingService, TrainingConfig, MockProgressEmitter, MockDatasetLoader, MockModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def example_basic_training():
    """Example of basic training service usage with mock implementations."""
    print("=== Basic Training Service Example ===")

    # Create configuration
    config = TrainingConfig(
        max_steps=1000,  # Reduced for quick example
        batch_size=4,
        learning_rate=1e-4,
        output_dir="./example_models",
        grokfast_enabled=True,
        act_enabled=True,
        ltm_enabled=True,
    )

    # Create service with mock implementations
    service = TrainingService(
        progress_emitter=MockProgressEmitter(),
        dataset_loader=MockDatasetLoader(),
        model_trainer=MockModelTrainer(),
        config=config,
    )

    # Start training session
    task_id = "example_training_001"
    session_info = await service.start_training_session(
        task_id=task_id, training_parameters={"example_mode": True}, model_names=["example_model_1", "example_model_2"]
    )

    print(f"Started training session: {session_info['task_id']}")

    # Execute training pipeline
    trained_models = await service.execute_training_pipeline(task_id)

    # Display results
    print(f"\nTraining completed! {len(trained_models)} models trained:")
    for model in trained_models:
        print(f"  - {model.model_name} ({model.model_id})")
        print(f"    Parameters: {model.parameter_count:,}")
        print(f"    Focus: {model.focus}")
        print(f"    Final loss: {model.training_stats.get('final_loss', 'N/A')}")
        print(f"    Capabilities: {len(model.capabilities)} features")

    return trained_models


async def example_custom_configuration():
    """Example with custom training configuration."""
    print("\n=== Custom Configuration Example ===")

    # Custom configuration for larger models
    config = TrainingConfig(
        # Model architecture
        d_model=512,  # Larger model
        n_layers=24,  # More layers
        n_heads=8,  # More attention heads
        vocab_size=50000,  # Larger vocabulary
        # Training parameters
        max_steps=5000,
        batch_size=8,
        learning_rate=5e-5,
        gradient_accumulation_steps=8,
        # GrokFast optimization
        grokfast_alpha=0.95,
        grokfast_lamb=1.5,
        # Dataset configuration
        dataset_sources=["GSM8K", "SVAMP", "HotpotQA", "StrategyQA"],
        max_train_samples=10000,
        max_eval_samples=1000,
        # Output
        output_dir="./custom_models",
    )

    service = TrainingService(
        progress_emitter=MockProgressEmitter(),
        dataset_loader=MockDatasetLoader(),
        model_trainer=MockModelTrainer(),
        config=config,
    )

    task_id = "custom_training_001"

    # Start with custom model names
    session_info = await service.start_training_session(
        task_id=task_id,
        training_parameters={"custom_config": True},
        model_names=["large_reasoning_model", "large_memory_model", "large_adaptive_model"],
    )

    print(f"Started custom training: {session_info['task_id']}")

    # Monitor progress during training
    async def monitor_progress():
        """Monitor training progress in parallel."""
        while True:
            status = await service.get_training_status(task_id)
            if status and status["status"] == "running":
                print(f"Training progress: {status.get('progress', 0):.1%}")
                await asyncio.sleep(5)  # Check every 5 seconds
            else:
                break

    # Run training and monitoring in parallel
    monitoring_task = asyncio.create_task(monitor_progress())
    trained_models = await service.execute_training_pipeline(task_id)
    monitoring_task.cancel()

    print(f"\nCustom training completed! Parameter count per model: {trained_models[0].parameter_count:,}")

    return trained_models


async def example_progress_tracking():
    """Example demonstrating progress tracking capabilities."""
    print("\n=== Progress Tracking Example ===")

    # Create custom progress emitter that logs detailed information
    class DetailedProgressEmitter(MockProgressEmitter):
        def __init__(self):
            super().__init__()
            self.start_time = None

        async def emit_progress(self, progress):
            if self.start_time is None:
                import time

                self.start_time = time.time()

            await super().emit_progress(progress)

            # Calculate elapsed time
            import time

            elapsed = time.time() - self.start_time

            print(f"[{elapsed:.1f}s] {progress.progress:.1%} - {progress.message}")

            if progress.step is not None and progress.total_steps is not None:
                print(f"  └─ Step {progress.step}/{progress.total_steps}")
            if progress.loss is not None:
                print(f"  └─ Loss: {progress.loss:.4f}")
            if progress.learning_rate is not None:
                print(f"  └─ Learning Rate: {progress.learning_rate:.2e}")

    config = TrainingConfig(max_steps=500)  # Quick example

    service = TrainingService(
        progress_emitter=DetailedProgressEmitter(),
        dataset_loader=MockDatasetLoader(),
        model_trainer=MockModelTrainer(),
        config=config,
    )

    task_id = "progress_tracking_001"

    await service.start_training_session(
        task_id=task_id, training_parameters={"detailed_progress": True}, model_names=["progress_demo_model"]
    )

    trained_models = await service.execute_training_pipeline(task_id)

    # Show final statistics
    emitter = service.progress_emitter
    print(f"\nProgress Events: {len(emitter.progress_events)}")
    print(f"Model Events: {len(emitter.model_events)}")

    return trained_models


async def main():
    """Run all examples."""
    print("Training Service Examples\n")

    try:
        # Run examples
        await example_basic_training()
        await example_custom_configuration()
        await example_progress_tracking()

        print("\nAll examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
