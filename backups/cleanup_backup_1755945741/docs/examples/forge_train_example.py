"""
Example script demonstrating the complete Forge training loop.

This example shows how to use all the components:
- Grokfast for accelerated grokking
- Edge-of-chaos curriculum
- Geometry probing for phase detection
- Self-modeling heads
- Dream/sleep cycles
- Temperature curriculum
"""

import logging
from pathlib import Path

from datasets import Dataset
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import Forge components
from agent_forge.training.forge_train import ForgeTrainConfig, ForgeTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_dataset(num_samples: int = 1000) -> Dataset:
    """Create a synthetic coding task dataset for demonstration."""

    # Simple arithmetic coding tasks
    tasks = []
    for i in range(num_samples):
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        operation = np.random.choice(["+", "-", "*"])

        if operation == "+":
            result = a + b
            prompt = f"def add_numbers():\n    return {a} + {b}"
            expected = f"def add_numbers():\n    return {result}"
        elif operation == "-":
            result = a - b
            prompt = f"def subtract_numbers():\n    return {a} - {b}"
            expected = f"def subtract_numbers():\n    return {result}"
        else:  # multiplication
            result = a * b
            prompt = f"def multiply_numbers():\n    return {a} * {b}"
            expected = f"def multiply_numbers():\n    return {result}"

        tasks.append(
            {
                "input": prompt,
                "target": expected,
                "difficulty": min(max(a, b), 100) / 100,  # Normalized difficulty
            }
        )

    return Dataset.from_list(tasks)


def preprocess_function(examples, tokenizer):
    """Preprocess examples for training."""
    inputs = tokenizer(
        examples["input"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    targets = tokenizer(
        examples["target"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    inputs["labels"] = targets["input_ids"]
    return inputs


def main():
    """Run the example training."""

    logger.info("🚀 Starting Forge Training Example")

    # Configuration
    config = ForgeTrainConfig(
        model_name="gpt2",
        tap_layers=[4, 8, 11],  # For GPT2-small (12 layers)
        hidden_dim=768,
        # Training settings
        learning_rate=5e-5,
        batch_size=8,
        max_steps=2000,
        warmup_steps=200,
        # Enable all features for demonstration
        enable_grokfast=True,
        enable_edge_control=True,
        enable_self_model=True,
        enable_temp_curriculum=True,
        enable_dream_cycles=True,
        # Grokfast settings
        grokfast_lambda_init=0.05,
        grokfast_lambda_max=0.2,
        # Edge control settings
        target_success_range=(0.4, 0.8),  # Easier range for synthetic tasks
        # Dream settings
        dream_cycle_interval=500,
        dream_duration=20,
        # Output
        output_dir=Path("./forge_example_output"),
        wandb_project=None,  # Disable W&B for example
        log_interval=50,
        checkpoint_interval=500,
    )

    # Load model and tokenizer
    logger.info("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    logger.info("Creating synthetic dataset...")
    train_data = create_synthetic_dataset(800)
    eval_data = create_synthetic_dataset(200)

    # Preprocess
    train_data = train_data.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    eval_data = eval_data.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Create trainer
    logger.info("Creating Forge trainer...")
    trainer = ForgeTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        config=config,
        tokenizer=tokenizer,
    )

    # Start training
    logger.info("Starting training with Forge enhancements:")
    logger.info(f"  • Grokfast: {config.enable_grokfast}")
    logger.info(f"  • Edge Control: {config.enable_edge_control}")
    logger.info(f"  • Self-Modeling: {config.enable_self_model}")
    logger.info(f"  • Dream Cycles: {config.enable_dream_cycles}")
    logger.info(f"  • Temperature Curriculum: {config.enable_temp_curriculum}")

    try:
        trainer.train()
        logger.info("✅ Training complete!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Print final statistics
    if trainer.telemetry_logger.frames:
        final_frame = trainer.telemetry_logger.frames[-1]
        stats = trainer.telemetry_logger.compute_statistics()

        logger.info("🎯 Final Training Statistics:")
        logger.info(f"  • Steps: {final_frame.step}")
        logger.info(f"  • Final Loss: {final_frame.loss:.4f}")
        logger.info(f"  • Final Accuracy: {final_frame.pass_at_1:.2%}")
        logger.info(f"  • Final Stage: {final_frame.stage}")
        logger.info(f"  • Avg Success Rate: {stats['avg_success_rate']:.2%}")

        if trainer.grok_controller:
            grok_stats = trainer.grok_controller.get_statistics()
            logger.info(f"  • Final Grokfast λ: {grok_stats['current_lambda']:.4f}")
            logger.info(f"  • Final Phase: {grok_stats['current_phase']}")

        if trainer.dream_manager:
            dream_stats = trainer.dream_manager.get_metrics()
            logger.info(f"  • Total Dreams: {dream_stats['total_dreams']}")
            logger.info(f"  • Buffer Size: {dream_stats['buffer_stats']['total_examples']}")


if __name__ == "__main__":
    main()
