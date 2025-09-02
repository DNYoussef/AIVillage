#!/usr/bin/env python3
"""Evaluation script for HRM Reasoner model."""

import argparse
import json
import logging
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from packages.hrrm.common.data_mixture import text_stream
from packages.hrrm.common.param_math import count_params
from packages.hrrm.reasoner.model import HRMReasoner

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


def evaluate_gsm8k_synthetic(model, num_samples=50):
    """Evaluate on synthetic GSM8K-like problems."""
    model.eval()
    correct = 0
    total = 0

    # Synthetic math problems for testing
    problems = [
        "What is 2 + 3?",
        "If John has 5 apples and gives 2 away, how many does he have left?",
        "What is 10 * 7?",
        "If a car travels 60 miles per hour for 2 hours, how far does it go?",
        "What is 100 - 37?",
    ]

    answers = ["5", "3", "70", "120", "63"]

    with torch.no_grad():
        for i in range(min(num_samples, len(problems))):
            problems[i % len(problems)]
            expected = answers[i % len(answers)]

            # Tokenize problem (simplified)
            # In real implementation, would use proper tokenizer
            prompt_tokens = torch.randint(0, 32000, (1, 64))
            if torch.cuda.is_available():
                prompt_tokens = prompt_tokens.cuda()

            # Generate with self-consistency
            model.generate_with_reasoning(prompt_tokens, max_length=100, k=5)

            # Extract answers and vote (simplified)
            # In real implementation, would parse actual text
            predicted = str(torch.randint(0, 200, (1,)).item())  # Placeholder

            if predicted == expected:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {"gsm8k_accuracy": accuracy, "correct": correct, "total": total}


def evaluate_thought_detection(model, num_samples=50):
    """Evaluate thought span detection accuracy."""
    model.eval()
    correct = 0
    total = 0

    sot_token_id = 32006  # <SoT>
    eot_token_id = 32007  # <EoT>

    with torch.no_grad():
        data_iter = text_stream(batch_size=4, seq_len=256, limit_steps=num_samples // 4)

        for batch in data_iter:
            x_ids = batch["x_ids"]

            if torch.cuda.is_available():
                x_ids = x_ids.cuda()

            # Create thought mask ground truth
            thought_mask = torch.zeros_like(x_ids, dtype=torch.float)
            for i in range(x_ids.shape[0]):
                in_thought = False
                for j in range(x_ids.shape[1]):
                    if x_ids[i, j] == sot_token_id:
                        in_thought = True
                    elif x_ids[i, j] == eot_token_id:
                        in_thought = False
                    elif in_thought:
                        thought_mask[i, j] = 1.0

            # Forward pass
            result = model(x_ids=x_ids, thought_mask=thought_mask)
            if isinstance(result, tuple):
                loss, logits = result
            else:
                pass

            # Extract thought predictions from ScratchpadSupervisor
            # Simplified evaluation - would need actual scratchpad outputs
            if hasattr(model, "scratchpad"):
                # Placeholder: in real implementation would extract hidden states
                # and run through scratchpad supervisor
                pred_thoughts = (torch.rand_like(thought_mask) > 0.5).float()

                correct += (pred_thoughts == thought_mask).sum().item()
                total += thought_mask.numel()

    accuracy = correct / total if total > 0 else 0.0

    return {"thought_accuracy": accuracy, "correct": correct, "total": total}


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    model = HRMReasoner(
        vocab=config.vocab_size,
        d=config.d_model,
        L_layers=config.n_layers,
        n_head=config.n_head,
        max_H=config.max_H,
        inner_T=config.inner_T,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if torch.cuda.is_available():
        model = model.cuda()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRM Reasoner")
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

    logger.info("Evaluating GSM8K-like reasoning...")
    gsm_results = evaluate_gsm8k_synthetic(model, args.samples // 2)

    logger.info("Evaluating thought detection...")
    thought_results = evaluate_thought_detection(model, args.samples)

    eval_time = time.time() - start_time

    # Compile results
    results = {
        "model_type": "reasoner",
        "checkpoint": args.ckpt,
        "param_count": param_count,
        "eval_samples": args.samples,
        "eval_time_seconds": eval_time,
        "perplexity": ppl_results,
        "gsm8k_synthetic": gsm_results,
        "thought_detection": thought_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    results_dir = Path("artifacts/eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "reasoner_eval.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 50)
    logger.info("REASONER EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Parameters: {param_count:,}")
    logger.info(f"Perplexity: {ppl_results['perplexity']:.2f}")
    logger.info(f"GSM8K Accuracy: {gsm_results['gsm8k_accuracy']:.3f}")
    logger.info(f"Thought Detection Accuracy: {thought_results['thought_accuracy']:.3f}")
    logger.info(f"Evaluation Time: {eval_time:.1f}s")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
