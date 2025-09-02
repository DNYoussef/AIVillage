#!/usr/bin/env python3
"""Training script for Memory model with Titans integration."""

import argparse
import logging
from pathlib import Path

from accelerate import Accelerator
import torch
import torch.nn.functional as F
import yaml

from packages.hrrm.common.data_mixture import text_stream
from packages.hrrm.common.param_math import assert_tiny_params
from packages.hrrm.common.training_utils import set_seed
from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryDataset:
    """Dataset wrapper for memory training with surprise-based updates."""

    def __init__(self, batch_size=8, seq_len=256, limit_steps=1000):
        self.data_iter = text_stream(batch_size, seq_len, limit_steps)

    def __iter__(self):
        for batch in self.data_iter:
            x_ids = batch["x_ids"]
            labels = batch["labels"]

            yield {
                "x_ids": x_ids,
                "targets": labels,  # For memory updates
                "attn_mask": None,
                "loss_like": None,  # Will be computed during forward pass
            }


def load_config(config_path: str) -> MemoryConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return MemoryConfig(**config_dict)


def train_memory(config: MemoryConfig, max_steps: int = 10000, lr: float = 3e-4):
    """Train Memory model with Titans online updates."""
    set_seed(1337)

    # Initialize model
    model = MemoryAsContextTiny(
        vocab=config.vocab_size,
        d=config.d_model,
        L=config.n_layers,
        h=config.n_head,
        mem_dim=config.mem_dim,
        mem_tokens=config.mem_tokens,
    )

    # Verify parameter count
    assert_tiny_params(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Memory model parameters: {param_count:,}")

    # Setup training
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # Training loop
    dataset = MemoryDataset(batch_size=8, seq_len=256, limit_steps=max_steps)
    model.train()

    step = 0
    loss_ema = None

    for batch in dataset:
        if step >= max_steps:
            break

        # Prepare batch
        batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        logits = model(
            x_ids=batch["x_ids"],
            targets=batch["targets"],
            attn_mask=batch["attn_mask"],
            loss_like=None,  # Will be computed inside model
        )

        # Compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["targets"].view(-1), ignore_index=-100)

        # Update memory with surprise signal (loss as surprise measure)
        with torch.no_grad():
            if hasattr(model, "mem") or hasattr(accelerator.unwrap_model(model), "mem"):
                unwrapped_model = accelerator.unwrap_model(model)
                # Use loss as surprise signal for memory update
                q = unwrapped_model.q_proj(unwrapped_model.tok(batch["x_ids"])[:, -1, :])
                v = unwrapped_model.v_proj(q)  # Simplified target
                unwrapped_model.mem.update(q, v, loss)

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
        checkpoint_dir = Path("artifacts/checkpoints/memory")
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
    parser = argparse.ArgumentParser(description="Train Memory model")
    parser.add_argument("--cfg", required=True, help="Config file path")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    config = load_config(args.cfg)
    train_memory(config, args.steps, args.lr)


if __name__ == "__main__":
    main()
