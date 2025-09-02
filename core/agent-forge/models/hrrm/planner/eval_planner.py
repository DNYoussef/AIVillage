#!/usr/bin/env python3
"""Evaluation script for HRM Planner model."""

import argparse
import json
import logging
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from packages.hrrm.common.data_mixture import text_stream
from packages.hrrm.common.param_math import count_params
from packages.hrrm.planner.model import HRMPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_perplexity(model, num_samples=100):
    """Evaluate perplexity on held-out data."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        data_iter = text_stream(batch_size=4, seq_len=256, limit_steps=num_samples // 4)

        for batch in data_iter:
            x_ids = batch["x_ids"]
            labels = batch["labels"]

            if torch.cuda.is_available():
                x_ids = x_ids.cuda()
                labels = labels.cuda()

            # Forward pass
            result = model(x_ids=x_ids, labels=labels)
            if isinstance(result, tuple):
                loss, logits = result
            else:
                logits = result
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}


def evaluate_control_tokens(model, num_samples=50):
    """Evaluate control token detection accuracy."""
    model.eval()
    correct = 0
    total = 0

    control_token_ids = [32001, 32002, 32003, 32004, 32005]  # <PLAN>, <SUBGOAL>, etc.

    with torch.no_grad():
        data_iter = text_stream(batch_size=4, seq_len=256, limit_steps=num_samples // 4)

        for batch in data_iter:
            x_ids = batch["x_ids"]

            if torch.cuda.is_available():
                x_ids = x_ids.cuda()

            # Create control mask ground truth
            control_mask = torch.zeros(x_ids.shape[0], 5)
            for i, token_id in enumerate(control_token_ids):
                control_mask[:, i] = (x_ids == token_id).any(dim=1).float()

            if torch.cuda.is_available():
                control_mask = control_mask.cuda()

            # Forward pass
            result = model(x_ids=x_ids, control_mask=control_mask)
            if isinstance(result, tuple):
                loss, logits = result
            else:
                logits = result

            # Extract control predictions from ControllerHead
            # Simplified: assume model outputs control logits
            if hasattr(model, "ctrl"):
                h_last = logits[:, -1, :]
                ctrl_logits = model.ctrl(h_last)
                ctrl_probs = torch.sigmoid(ctrl_logits)
                ctrl_preds = (ctrl_probs > 0.5).float()

                correct += (ctrl_preds == control_mask).sum().item()
                total += control_mask.numel()

    accuracy = correct / total if total > 0 else 0.0

    return {"control_accuracy": accuracy, "correct": correct, "total": total}


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    model = HRMPlanner(
        vocab=config.vocab_size,
        d=config.d_model,
        L_layers=config.n_layers,
        n_head=config.n_head,
        control_tokens=config.control_tokens,
        max_H=config.max_H,
        inner_T=config.inner_T,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if torch.cuda.is_available():
        model = model.cuda()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRM Planner")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples for evaluation")

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.ckpt}")
    model, config = load_model(args.ckpt)

    param_count = count_params(model)
    logger.info(f"Model parameters: {param_count:,}")

    # Run evaluations
    start_time = time.time()

    logger.info("Evaluating perplexity...")
    ppl_results = evaluate_perplexity(model, args.samples)

    logger.info("Evaluating control token detection...")
    ctrl_results = evaluate_control_tokens(model, args.samples)

    eval_time = time.time() - start_time

    # Compile results
    results = {
        "model_type": "planner",
        "checkpoint": args.ckpt,
        "param_count": param_count,
        "eval_samples": args.samples,
        "eval_time_seconds": eval_time,
        "perplexity": ppl_results,
        "control_tokens": ctrl_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    results_dir = Path("artifacts/eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "planner_eval.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 50)
    logger.info("PLANNER EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Parameters: {param_count:,}")
    logger.info(f"Perplexity: {ppl_results['perplexity']:.2f}")
    logger.info(f"Control Token Accuracy: {ctrl_results['control_accuracy']:.3f}")
    logger.info(f"Evaluation Time: {eval_time:.1f}s")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
