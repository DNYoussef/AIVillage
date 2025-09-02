#!/usr/bin/env python3
"""Evaluation script for Memory model."""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from packages.hrrm.common.data_mixture import text_stream
from packages.hrrm.common.param_math import count_params
from packages.hrrm.memory.model import MemoryAsContextTiny

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
            logits = model(x_ids=x_ids, targets=labels)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}


def evaluate_memory_retrieval(model, num_samples=50):
    """Evaluate memory retrieval performance."""
    model.eval()
    retrieval_scores = []

    with torch.no_grad():
        data_iter = text_stream(batch_size=4, seq_len=256, limit_steps=num_samples // 4)

        for batch in data_iter:
            x_ids = batch["x_ids"]

            if torch.cuda.is_available():
                x_ids = x_ids.cuda()

            # Test memory retrieval
            if hasattr(model, "mem"):
                # Query memory with last token representation
                x_emb = model.tok(x_ids)
                q = model.q_proj(x_emb[:, -1, :])

                # Retrieve from memory
                v, idx, att = model.mem.read(q, topk=8)

                # Compute retrieval metrics
                # Attention entropy as diversity measure
                att_entropy = -(att * torch.log(att + 1e-8)).sum(-1).mean()
                retrieval_scores.append(att_entropy.item())

    avg_retrieval_score = np.mean(retrieval_scores) if retrieval_scores else 0.0

    return {"retrieval_score": avg_retrieval_score, "num_retrievals": len(retrieval_scores)}


def evaluate_memory_adaptation(model, num_samples=20):
    """Evaluate online memory adaptation (surprise-based updates)."""
    model.eval()

    # Track memory changes over time
    initial_keys = model.mem.keys.clone() if hasattr(model, "mem") else None
    adaptation_scores = []

    with torch.no_grad():
        data_iter = text_stream(batch_size=2, seq_len=128, limit_steps=num_samples // 2)

        for i, batch in enumerate(data_iter):
            x_ids = batch["x_ids"]
            labels = batch["labels"]

            if torch.cuda.is_available():
                x_ids = x_ids.cuda()
                labels = labels.cuda()

            # Forward pass to get loss (surprise signal)
            logits = model(x_ids=x_ids, targets=labels)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            # Simulate memory update with surprise
            if hasattr(model, "mem") and initial_keys is not None:
                q = model.q_proj(model.tok(x_ids)[:, -1, :])
                v = model.v_proj(q)

                # Apply memory update
                model.mem.update(q, v, loss)

                # Measure adaptation (change in memory keys)
                key_change = torch.norm(model.mem.keys - initial_keys).item()
                adaptation_scores.append(key_change)

                # Update baseline
                initial_keys = model.mem.keys.clone()

    avg_adaptation = np.mean(adaptation_scores) if adaptation_scores else 0.0

    return {"adaptation_score": avg_adaptation, "num_updates": len(adaptation_scores)}


def evaluate_memory_capacity(model):
    """Evaluate memory storage capacity."""
    if not hasattr(model, "mem"):
        return {"capacity": 0, "slots": 0, "dimensions": 0}

    memory = model.mem

    # Memory statistics
    total_slots = memory.keys.size(0)
    key_dim = memory.keys.size(1)
    val_dim = memory.vals.size(1)

    # Compute utilization (non-zero entries)
    key_utilization = (memory.keys.abs() > 1e-6).float().mean().item()
    val_utilization = (memory.vals.abs() > 1e-6).float().mean().item()

    return {
        "total_slots": total_slots,
        "key_dim": key_dim,
        "val_dim": val_dim,
        "key_utilization": key_utilization,
        "val_utilization": val_utilization,
    }


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    model = MemoryAsContextTiny(
        vocab=config.vocab_size,
        d=config.d_model,
        L=config.n_layers,
        h=config.n_head,
        mem_dim=config.mem_dim,
        mem_tokens=config.mem_tokens,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if torch.cuda.is_available():
        model = model.cuda()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memory model")
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

    logger.info("Evaluating memory retrieval...")
    retrieval_results = evaluate_memory_retrieval(model, args.samples // 2)

    logger.info("Evaluating memory adaptation...")
    adaptation_results = evaluate_memory_adaptation(model, args.samples // 5)

    logger.info("Evaluating memory capacity...")
    capacity_results = evaluate_memory_capacity(model)

    eval_time = time.time() - start_time

    # Compile results
    results = {
        "model_type": "memory",
        "checkpoint": args.ckpt,
        "param_count": param_count,
        "eval_samples": args.samples,
        "eval_time_seconds": eval_time,
        "perplexity": ppl_results,
        "memory_retrieval": retrieval_results,
        "memory_adaptation": adaptation_results,
        "memory_capacity": capacity_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    results_dir = Path("artifacts/eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "memory_eval.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 50)
    logger.info("MEMORY EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Parameters: {param_count:,}")
    logger.info(f"Perplexity: {ppl_results['perplexity']:.2f}")
    logger.info(f"Retrieval Score: {retrieval_results['retrieval_score']:.3f}")
    logger.info(f"Adaptation Score: {adaptation_results['adaptation_score']:.3f}")
    logger.info(f"Memory Slots: {capacity_results['total_slots']}")
    logger.info(f"Key Utilization: {capacity_results['key_utilization']:.3f}")
    logger.info(f"Evaluation Time: {eval_time:.1f}s")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
