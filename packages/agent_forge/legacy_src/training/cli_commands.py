"""
CLI commands for the Forge training system.
Integrates with the main Agent Forge CLI.
"""

import json
import logging
from pathlib import Path

import click
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .forge_train import ForgeTrainConfig, ForgeTrainer

logger = logging.getLogger(__name__)


@click.group()
def forge():
    """Agent Forge training commands."""
    pass


@forge.command()
@click.option("--model-name", default="gpt2", help="Base model to train")
@click.option("--dataset", default="openai_humaneval", help="Dataset to use")
@click.option("--output-dir", default="./forge_output", help="Output directory")
@click.option("--batch-size", default=8, type=int, help="Training batch size")
@click.option("--learning-rate", default=1e-4, type=float, help="Learning rate")
@click.option("--max-steps", default=10000, type=int, help="Maximum training steps")
@click.option("--enable-grokfast/--no-grokfast", default=True, help="Enable Grokfast")
@click.option("--enable-edge/--no-edge", default=True, help="Enable edge-of-chaos control")
@click.option("--enable-self-model/--no-self-model", default=True, help="Enable self-modeling")
@click.option("--enable-dreams/--no-dreams", default=True, help="Enable dream cycles")
@click.option("--wandb-project", default="forge-train", help="W&B project name")
@click.option("--resume", type=str, help="Resume from checkpoint")
def train(
    model_name: str,
    dataset: str,
    output_dir: str,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    enable_grokfast: bool,
    enable_edge: bool,
    enable_self_model: bool,
    enable_dreams: bool,
    wandb_project: str,
    resume: str | None,
):
    """
    Run the complete Forge training loop with all enhancements.

    This implements:
    - FastGROK (Grokfast) gradient amplification
    - Edge-of-chaos curriculum control
    - Geometry probing for phase detection
    - Self-modeling heads
    - Dream/sleep consolidation cycles
    - Temperature curriculum
    """
    click.echo(f"🚀 Starting Forge training with model: {model_name}")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load model and tokenizer
    click.echo(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    click.echo(f"Loading dataset: {dataset}")
    if dataset == "openai_humaneval":
        # Load HumanEval for code generation
        train_dataset = load_dataset("openai_humaneval", split="test[:80%]")
        eval_dataset = load_dataset("openai_humaneval", split="test[80%:]")
    elif dataset == "mbpp":
        # Load MBPP for code generation
        train_dataset = load_dataset("mbpp", split="train")
        eval_dataset = load_dataset("mbpp", split="test")
    else:
        # Custom dataset loading
        train_dataset = load_dataset(dataset, split="train")
        eval_dataset = load_dataset(dataset, split="validation")

    # Create configuration
    config = ForgeTrainConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_steps=max_steps,
        enable_grokfast=enable_grokfast,
        enable_edge_control=enable_edge,
        enable_self_model=enable_self_model,
        enable_dream_cycles=enable_dreams,
        wandb_project=wandb_project if wandb_project != "none" else None,
        output_dir=Path(output_dir),
    )

    # Create trainer
    trainer = ForgeTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    if resume:
        click.echo(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        trainer.epoch = checkpoint["epoch"]

    # Start training
    click.echo("Starting training loop...")
    click.echo(f"  • Grokfast: {'✅' if enable_grokfast else '❌'}")
    click.echo(f"  • Edge Control: {'✅' if enable_edge else '❌'}")
    click.echo(f"  • Self-Modeling: {'✅' if enable_self_model else '❌'}")
    click.echo(f"  • Dream Cycles: {'✅' if enable_dreams else '❌'}")

    try:
        trainer.train()
        click.echo("✅ Training complete!")

        # Save final model
        final_model_path = config.output_dir / "final_model"
        trainer.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        click.echo(f"Model saved to: {final_model_path}")

    except KeyboardInterrupt:
        click.echo("\n⚠️ Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = config.checkpoint_dir / "emergency_checkpoint.pt"
        torch.save(
            {
                "global_step": trainer.global_step,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            },
            emergency_path,
        )
        click.echo(f"Emergency checkpoint saved to: {emergency_path}")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise


@forge.command()
@click.option("--checkpoint", required=True, help="Path to checkpoint")
@click.option("--output", default="./analysis", help="Output directory for analysis")
def analyze(checkpoint: str, output: str):
    """
    Analyze a training checkpoint to understand grokking dynamics.

    Generates reports on:
    - Intrinsic dimension evolution
    - Grokfast lambda history
    - Stage transitions
    - Dream buffer statistics
    """
    click.echo(f"📊 Analyzing checkpoint: {checkpoint}")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu")

    # Extract metrics
    metrics = ckpt.get("metrics", {})
    config = ckpt.get("config", {})

    # Create analysis report
    report = {
        "checkpoint": checkpoint,
        "global_step": ckpt.get("global_step", 0),
        "metrics": metrics,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
    }

    # Save report
    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    click.echo(f"Analysis saved to: {report_path}")

    # Print summary
    click.echo("\n📈 Training Summary:")
    click.echo(f"  • Steps: {report['global_step']}")
    click.echo(f"  • Eval Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    click.echo(f"  • Eval Accuracy: {metrics.get('eval_accuracy', 'N/A'):.2%}")


@forge.command()
@click.option("--model-path", required=True, help="Path to trained model")
@click.option("--prompt", required=True, help="Prompt to test")
@click.option("--temperature", default=0.7, type=float, help="Generation temperature")
@click.option("--max-tokens", default=100, type=int, help="Maximum tokens to generate")
def test(model_path: str, prompt: str, temperature: float, max_tokens: int):
    """
    Test a trained Forge model with a prompt.
    """
    click.echo(f"🧪 Testing model: {model_path}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
        )

    # Decode and display
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    click.echo("\n📝 Prompt:")
    click.echo(prompt)
    click.echo("\n🤖 Generated:")
    click.echo(generated_text[len(prompt) :])


@forge.command()
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.option("--dry-run", is_flag=True, help="Validate config without training")
def validate(config: str | None, dry_run: bool):
    """
    Validate a training configuration.
    """
    click.echo("🔍 Validating configuration...")

    if config:
        # Load config from file
        with open(config) as f:
            config_data = json.load(f)

        try:
            train_config = ForgeTrainConfig(**config_data)
            click.echo("✅ Configuration is valid!")

            if dry_run:
                click.echo("\nConfiguration details:")
                for key, value in train_config.__dict__.items():
                    click.echo(f"  • {key}: {value}")

        except Exception as e:
            click.echo(f"❌ Invalid configuration: {e}")
            return

    else:
        # Show default configuration
        default_config = ForgeTrainConfig()
        click.echo("Default configuration:")
        for key, value in default_config.__dict__.items():
            click.echo(f"  • {key}: {value}")


# Export commands for CLI integration
commands = {
    "train": train,
    "analyze": analyze,
    "test": test,
    "validate": validate,
}
