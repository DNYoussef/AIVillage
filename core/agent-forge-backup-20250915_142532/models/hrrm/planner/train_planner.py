#!/usr/bin/env python3
"""Training script for HRM Planner model."""

import argparse
import logging
from pathlib import Path

from accelerate import Accelerator
import torch
import yaml

from packages.hrrm.common.data_mixture import text_stream
from packages.hrrm.common.param_math import assert_tiny_params
from packages.hrrm.common.training_utils import set_seed
from packages.hrrm.planner.heads import PlannerConfig
from packages.hrrm.planner.model import HRMPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlannerDataset:
    """Dataset wrapper for planner training with control token supervision."""

    def __init__(self, batch_size=8, seq_len=256, limit_steps=1000):
        self.data_iter = text_stream(batch_size, seq_len, limit_steps)
        self.control_token_ids = [32001, 32002, 32003, 32004, 32005]  # <PLAN>, <SUBGOAL>, etc.

    def __iter__(self):
        for batch in self.data_iter:
            x_ids = batch["x_ids"]
            labels = batch["labels"]

            # Create control mask for auxiliary supervision
            control_mask = torch.zeros(x_ids.shape[0], 5)  # [B, 5] for 5 control tokens
            for i, token_id in enumerate(self.control_token_ids):
                control_mask[:, i] = (x_ids == token_id).any(dim=1).float()

            yield {"x_ids": x_ids, "labels": labels, "control_mask": control_mask, "attn_mask": None}


def load_config(config_path: str) -> PlannerConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return PlannerConfig(**config_dict)


def train_planner(config: PlannerConfig, max_steps: int = 10000, lr: float = 3e-4):
    """Train HRM Planner model."""
    set_seed(1337)

    # Initialize model
    model = HRMPlanner(
        vocab=config.vocab_size,
        d=config.d_model,
        L_layers=config.n_layers,
        n_head=config.n_head,
        control_tokens=config.control_tokens,
        max_H=config.max_H,
        inner_T=config.inner_T,
    )

    # Verify parameter count
    assert_tiny_params(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Planner model parameters: {param_count:,}")

    # Setup training
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # Training loop
    dataset = PlannerDataset(batch_size=8, seq_len=256, limit_steps=max_steps)
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
            control_mask=batch["control_mask"],
            lambda_ctrl=config.lambda_ctrl,
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
        checkpoint_dir = Path("artifacts/checkpoints/planner")
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
    parser = argparse.ArgumentParser(description="Train HRM Planner")
    parser.add_argument("--cfg", required=True, help="Config file path")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    config = load_config(args.cfg)
    train_planner(config, args.steps, args.lr)


if __name__ == "__main__":
    main()
