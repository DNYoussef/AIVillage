#!/usr/bin/env python3
"""HRRM metrics reporting CLI."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_results(results_dir: Path) -> dict[str, Any]:
    """Load evaluation results from all models."""
    results = {}

    for model_type in ["planner", "reasoner", "memory"]:
        result_file = results_dir / f"{model_type}_eval.json"
        if result_file.exists():
            with open(result_file) as f:
                results[model_type] = json.load(f)
        else:
            logger.warning(f"No results found for {model_type}")
            results[model_type] = None

    return results


def format_metrics_table(results: dict[str, Any]) -> str:
    """Format metrics as a table."""

    table = []
    table.append("=" * 80)
    table.append("HRRRM BOOTSTRAP METRICS REPORT")
    table.append("=" * 80)
    table.append("")

    # Header
    table.append(f"{'Model':<12} {'Params':<12} {'Perplexity':<12} {'Task Metric':<15} {'Eval Time':<10}")
    table.append("-" * 80)

    # Model rows
    for model_type in ["planner", "reasoner", "memory"]:
        result = results.get(model_type)
        if result is None:
            table.append(f"{model_type:<12} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<10}")
            continue

        # Extract metrics
        params = f"{result['param_count']:,}" if "param_count" in result else "N/A"
        ppl = f"{result['perplexity']['perplexity']:.2f}" if "perplexity" in result else "N/A"
        eval_time = f"{result['eval_time_seconds']:.1f}s" if "eval_time_seconds" in result else "N/A"

        # Model-specific task metric
        if model_type == "planner":
            task_metric = f"{result['control_tokens']['control_accuracy']:.3f}" if "control_tokens" in result else "N/A"
        elif model_type == "reasoner":
            task_metric = f"{result['gsm8k_synthetic']['gsm8k_accuracy']:.3f}" if "gsm8k_synthetic" in result else "N/A"
        elif model_type == "memory":
            task_metric = (
                f"{result['memory_retrieval']['retrieval_score']:.3f}" if "memory_retrieval" in result else "N/A"
            )

        table.append(f"{model_type:<12} {params:<12} {ppl:<12} {task_metric:<15} {eval_time:<10}")

    table.append("")
    table.append("Metrics:")
    table.append("- Params: Parameter count (target: 48M-55M)")
    table.append("- Perplexity: Language modeling perplexity (lower is better)")
    table.append("- Planner Ctrl Acc: Control token detection accuracy")
    table.append("- Reasoner GSM8K Acc: Math reasoning accuracy")
    table.append("- Memory Retr Score: Memory retrieval quality score")
    table.append("")

    return "\n".join(table)


def generate_summary_stats(results: dict[str, Any]) -> dict[str, Any]:
    """Generate summary statistics."""

    summary = {
        "total_models": len([r for r in results.values() if r is not None]),
        "total_params": 0,
        "avg_perplexity": 0,
        "models_in_param_range": 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    valid_results = [r for r in results.values() if r is not None]

    if valid_results:
        # Calculate totals
        summary["total_params"] = sum(r.get("param_count", 0) for r in valid_results)

        # Average perplexity
        ppls = [r["perplexity"]["perplexity"] for r in valid_results if "perplexity" in r]
        summary["avg_perplexity"] = sum(ppls) / len(ppls) if ppls else 0

        # Models in target parameter range (48M-55M)
        summary["models_in_param_range"] = sum(
            1 for r in valid_results if "param_count" in r and 48_000_000 <= r["param_count"] <= 55_000_000
        )

    return summary


def check_acceptance_criteria(results: dict[str, Any], summary: dict[str, Any]) -> dict[str, bool]:
    """Check if acceptance criteria are met."""

    criteria = {}

    # All 3 models have results
    criteria["three_models_trained"] = summary["total_models"] == 3

    # All models in parameter range
    criteria["param_counts_valid"] = summary["models_in_param_range"] == 3

    # All models have reasonable perplexity (< 100)
    valid_ppls = [
        r["perplexity"]["perplexity"]
        for r in results.values()
        if r is not None and "perplexity" in r and r["perplexity"]["perplexity"] < 100
    ]
    criteria["perplexity_reasonable"] = len(valid_ppls) == 3

    # HF exports exist
    hf_dir = Path("artifacts/hf_exports")
    criteria["hf_exports_exist"] = all((hf_dir / model).exists() for model in ["planner", "reasoner", "memory"])

    # Checkpoints exist
    checkpoint_dir = Path("artifacts/checkpoints")
    criteria["checkpoints_exist"] = all(
        (checkpoint_dir / model / "latest.pt").exists() for model in ["planner", "reasoner", "memory"]
    )

    return criteria


def main():
    parser = argparse.ArgumentParser(description="Generate HRRM metrics report")
    parser.add_argument("--results-dir", default="artifacts/eval_results", help="Evaluation results directory")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress table output")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load results
    results = load_eval_results(results_dir)

    # Generate summary
    summary = generate_summary_stats(results)

    # Check acceptance criteria
    criteria = check_acceptance_criteria(results, summary)

    # Create full report
    report = {
        "timestamp": summary["timestamp"],
        "summary": summary,
        "acceptance_criteria": criteria,
        "detailed_results": results,
        "criteria_met": all(criteria.values()),
    }

    # Save JSON report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

    # Print table unless quiet
    if not args.quiet:
        table = format_metrics_table(results)
        print(table)

        # Print acceptance criteria
        print("ACCEPTANCE CRITERIA:")
        print("-" * 40)
        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{criterion.replace('_', ' ').title():<25} {status}")

        print(f"\nOverall Status: {'✓ READY' if report['criteria_met'] else '✗ NOT READY'}")

    # Return exit code based on criteria
    return 0 if report["criteria_met"] else 1


if __name__ == "__main__":
    exit(main())
