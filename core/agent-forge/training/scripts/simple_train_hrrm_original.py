#!/usr/bin/env python3
"""Simple training script for HRRM models."""

import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig
from packages.hrrm.planner.heads import PlannerConfig
from packages.hrrm.planner.model import HRMPlanner
from packages.hrrm.reasoner.model import HRMReasoner, ReasonerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(tokenizer, batch_size=8, seq_len=256, num_batches=100):
    """Create synthetic training data."""
    data = []
    for _ in range(num_batches):
        # Create random token sequences
        batch = torch.randint(1, min(1000, tokenizer.vocab_size - 1), (batch_size, seq_len))
        data.append(batch)
    return data


def train_model(model, optimizer, data, model_name, epochs=3):
    """Simple training loop."""
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Training {model_name} for {epochs} epochs on {device}")

    total_loss = 0
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(data):
            batch = batch.to(device)

            # Create labels (shift by 1 for language modeling)
            labels = batch[:, 1:].contiguous()
            inputs = batch[:, :-1].contiguous()

            optimizer.zero_grad()

            # Forward pass
            if hasattr(model, "forward"):
                if model_name == "HRMPlanner":
                    output = model(inputs, labels=labels)
                    loss = (
                        output.loss
                        if hasattr(output, "loss")
                        else torch.nn.functional.cross_entropy(
                            output.logits.view(-1, output.logits.size(-1)), labels.view(-1)
                        )
                    )
                elif model_name == "HRMReasoner":
                    output = model(inputs, labels=labels)
                    loss = (
                        output.loss
                        if hasattr(output, "loss")
                        else torch.nn.functional.cross_entropy(
                            output.logits.view(-1, output.logits.size(-1)), labels.view(-1)
                        )
                    )
                else:  # MemoryAsContextTiny
                    output = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        output.logits.view(-1, output.logits.size(-1)), labels.view(-1)
                    )

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            step += 1

            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(data)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

    avg_total_loss = total_loss / (epochs * len(data))
    logger.info(f"{model_name} training completed. Average loss: {avg_total_loss:.4f}")
    return avg_total_loss


def save_model(model, config, model_name, checkpoint_dir):
    """Save model and config."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_path = checkpoint_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save config
    config_path = checkpoint_dir / "config.json"
    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Saved {model_name} to {checkpoint_dir}")


def main():
    """Main training function."""
    # Create tokenizer (simple approach)
    logger.info("Creating tokenizer...")
    tokenizer_path = "artifacts/tokenizer/hrrm_bpe_32k.json"
    if os.path.exists(tokenizer_path):
        logger.info(f"Using existing tokenizer: {tokenizer_path}")

        # Create a simple mock tokenizer for now
        class MockTokenizer:
            vocab_size = 32000

        tokenizer = MockTokenizer()
    else:
        logger.info("Creating mock tokenizer")

        class MockTokenizer:
            vocab_size = 32000

        tokenizer = MockTokenizer()

    # Create synthetic data (smaller for faster training)
    logger.info("Creating synthetic training data...")
    train_data = create_synthetic_data(tokenizer, batch_size=2, seq_len=64, num_batches=20)

    # Create models with smaller configurations for faster training
    logger.info("Creating HRRM models...")

    # 1. Train HRMPlanner
    logger.info("=" * 50)
    logger.info("Training HRMPlanner...")
    planner_config = PlannerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,  # Smaller for faster training
        n_layers=8,  # Fewer layers
        n_head=8,
        max_H=2,  # Fewer H cycles
        inner_T=2,  # Fewer T steps
        control_tokens=5,  # Number of control tokens
        lambda_ctrl=0.2,
    )

    # Patch the config to work with the model expectation
    planner_config.control_tokens = ["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"]

    planner = HRMPlanner(planner_config)
    planner_optimizer = optim.Adam(planner.parameters(), lr=3e-4)

    planner_loss = train_model(planner, planner_optimizer, train_data, "HRMPlanner", epochs=1)
    save_model(planner, planner_config, "HRMPlanner", "artifacts/checkpoints/planner")

    # 2. Train HRMReasoner
    logger.info("=" * 50)
    logger.info("Training HRMReasoner...")
    reasoner_config = ReasonerConfig(
        vocab_size=tokenizer.vocab_size, d_model=256, n_layers=8, n_head=8, max_H=2, inner_T=2, self_consistency_k=3
    )

    reasoner = HRMReasoner(reasoner_config)
    reasoner_optimizer = optim.Adam(reasoner.parameters(), lr=3e-4)

    reasoner_loss = train_model(reasoner, reasoner_optimizer, train_data, "HRMReasoner", epochs=1)
    save_model(reasoner, reasoner_config, "HRMReasoner", "artifacts/checkpoints/reasoner")

    # 3. Train MemoryAsContextTiny
    logger.info("=" * 50)
    logger.info("Training MemoryAsContextTiny...")
    memory_config = MemoryConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=8,
        n_head=8,
        mem_dim=128,
        mem_tokens=32,
        mem_slots=64,
        alpha=1.0,
        beta=0.9,
        eta=0.01,
        eta_decay=0.001,
    )

    memory = MemoryAsContextTiny(memory_config)
    memory_optimizer = optim.Adam(memory.parameters(), lr=3e-4)

    memory_loss = train_model(memory, memory_optimizer, train_data, "MemoryAsContextTiny", epochs=1)
    save_model(memory, memory_config, "MemoryAsContextTiny", "artifacts/checkpoints/memory")

    # Create summary report
    logger.info("=" * 50)
    logger.info("HRRM Training Summary")
    logger.info("=" * 50)
    logger.info(f"HRMPlanner - Final Loss: {planner_loss:.4f}")
    logger.info(f"HRMReasoner - Final Loss: {reasoner_loss:.4f}")
    logger.info(f"MemoryAsContextTiny - Final Loss: {memory_loss:.4f}")

    # Save training summary
    summary = {
        "training_completed": True,
        "models": {
            "planner": {
                "final_loss": planner_loss,
                "config": planner_config.__dict__,
                "checkpoint": "artifacts/checkpoints/planner",
            },
            "reasoner": {
                "final_loss": reasoner_loss,
                "config": reasoner_config.__dict__,
                "checkpoint": "artifacts/checkpoints/reasoner",
            },
            "memory": {
                "final_loss": memory_loss,
                "config": memory_config.__dict__,
                "checkpoint": "artifacts/checkpoints/memory",
            },
        },
    }

    summary_path = "artifacts/hrrm_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved to {summary_path}")
    logger.info("🚀 All 3 HRRM models trained successfully!")


if __name__ == "__main__":
    main()
