#!/usr/bin/env python3
"""Training script for HRM Reasoner model."""

import argparse
import logging
from pathlib import Path

from accelerate import Accelerator
import torch
import yaml

from packages.hrrm.common.data_mixture import text_stream
from packages.hrrm.common.param_math import assert_tiny_params
from packages.hrrm.common.training_utils import set_seed
from packages.hrrm.reasoner.model import HRMReasoner
from packages.hrrm.reasoner.scratchpad import ReasonerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasonerDataset:
    """Dataset wrapper for reasoner training with thought supervision."""

    def __init__(self, batch_size=8, seq_len=256, limit_steps=1000):
        self.data_iter = text_stream(batch_size, seq_len, limit_steps)
        self.sot_token_id = 32006  # <SoT>
        self.eot_token_id = 32007  # <EoT>

    def __iter__(self):
        for batch in self.data_iter:
            x_ids = batch["x_ids"]
            labels = batch["labels"]

            # Create thought mask for reasoning spans
            thought_mask = torch.zeros_like(x_ids, dtype=torch.float)

            # Mark positions between <SoT> and <EoT> as thought spans
            for i in range(x_ids.shape[0]):
                in_thought = False
                for j in range(x_ids.shape[1]):
                    if x_ids[i, j] == self.sot_token_id:
                        in_thought = True
                    elif x_ids[i, j] == self.eot_token_id:
                        in_thought = False
                    elif in_thought:
                        thought_mask[i, j] = 1.0

            yield {"x_ids": x_ids, "labels": labels, "thought_mask": thought_mask, "attn_mask": None}


def load_config(config_path: str) -> ReasonerConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return ReasonerConfig(**config_dict)


def train_reasoner(config: ReasonerConfig, max_steps: int = 10000, lr: float = 3e-4):
    """Train HRM Reasoner model."""
    set_seed(1337)

    # Initialize model
    model = HRMReasoner(
        vocab=config.vocab_size,
        d=config.d_model,
        L_layers=config.n_layers,
        n_head=config.n_head,
        max_H=config.max_H,
        inner_T=config.inner_T,
    )

    # Verify parameter count
    assert_tiny_params(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Reasoner model parameters: {param_count:,}")

    # Setup training
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # Training loop
    dataset = ReasonerDataset(batch_size=8, seq_len=256, limit_steps=max_steps)
    model.train()

    step = 0
    loss_ema = None

    for batch in dataset:
        if step >= max_steps:
            break

        # Prepare batch
        batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        loss, logits = model(
            x_ids=batch["x_ids"],
            labels=batch["labels"],
            attn_mask=batch["attn_mask"],
            thought_mask=batch["thought_mask"],
            lambda_thought=0.1,
        )

        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        step += 1
        loss_ema = loss.item() if loss_ema is None else 0.9 * loss_ema + 0.1 * loss.item()

        if step % 50 == 0 and accelerator.is_local_main_process:
            logger.info(f"Step {step}/{max_steps}, Loss: {loss_ema:.4f}")

    if accelerator.is_local_main_process:
        logger.info("Training completed")

        # Save checkpoint
        checkpoint_dir = Path("artifacts/checkpoints/reasoner")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            {
                "model_state_dict": unwrapped_model.state_dict(),
                "config": config,
                "steps": step,
            },
            checkpoint_dir / "latest.pt",
        )

        logger.info(f"Saved checkpoint to {checkpoint_dir / 'latest.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train HRM Reasoner")
    parser.add_argument("--cfg", required=True, help="Config file path")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    config = load_config(args.cfg)
    train_reasoner(config, args.steps, args.lr)


if __name__ == "__main__":
    main()
