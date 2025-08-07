"""Agent Forge Complete Benchmark Integration

Orchestrates the complete benchmarking workflow for Agent Forge models:
1. Validates all pipeline outputs
2. Runs comprehensive benchmarks (MMLU, GSM8K, HumanEval, etc.)
3. Compares against baseline 1.5B and frontier models
4. Generates W&B reports and publication-ready summaries
5. Creates cross-stage performance analysis

This script integrates all Agent Forge components for end-to-end evaluation.
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import wandb

from agent_forge.benchmark_runner import BenchmarkRunner

# Agent Forge imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_forge_benchmark.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class AgentForgeBenchmarkOrchestrator:
    """Orchestrates complete Agent Forge benchmarking workflow."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "benchmark_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Agent Forge pipeline model paths
        self.model_paths = {
            "original_compressed": "./final_compressed_model",
            "mastery_trained": "./mastery_output/final_model",
            "evomerge_best": "./evomerge_output/best_model",
            "quietstar_enhanced": "./quietstar_enhanced",
            "unified_pipeline": "./unified_checkpoints/final_model",
        }

        # Baseline models for comparison (1.5B parameter range)
        self.baseline_models = [
            "microsoft/DialoGPT-large",  # 762M - closest to 1.5B available
            "facebook/opt-1.3b",  # 1.3B
            "EleutherAI/gpt-neo-1.3B",  # 1.3B
        ]

        # Frontier models for comparison
        self.frontier_models = [
            "microsoft/phi-2",  # 2.7B - high performance
            "mistralai/Mistral-7B-Instruct-v0.1",  # 7B - instruction tuned
        ]

        self.benchmark_runner = BenchmarkRunner(str(self.results_dir))

    async def validate_models(self) -> dict[str, bool]:
        """Validate that all expected models exist."""
        logger.info("Validating Agent Forge model availability")

        validation_results = {}

        for model_name, model_path in self.model_paths.items():
            path = Path(model_path)
            exists = path.exists()
            validation_results[model_name] = exists

            if exists:
                logger.info(f"✅ {model_name}: Found at {model_path}")

                # Check if it's a proper model directory
                if path.is_dir():
                    has_config = (path / "config.json").exists()
                    has_model = any(path.glob("*.bin")) or any(
                        path.glob("*.safetensors")
                    )

                    if not (has_config and has_model):
                        logger.warning(
                            f"⚠️  {model_name}: Directory exists but missing model files"
                        )
                        validation_results[model_name] = False

            else:
                logger.warning(f"❌ {model_name}: Not found at {model_path}")

        return validation_results

    async def run_comprehensive_evaluation(
        self,
        quick_mode: bool = False,
        skip_baselines: bool = False,
        skip_frontier: bool = False,
    ) -> dict[str, Any]:
        """Run comprehensive evaluation of all Agent Forge models."""
        logger.info("Starting comprehensive Agent Forge evaluation")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Include baselines: {not skip_baselines}")
        logger.info(f"Include frontier: {not skip_frontier}")

        # Validate models first
        validation_results = await self.validate_models()
        available_models = {
            name: path
            for name, path in self.model_paths.items()
            if validation_results[name]
        }

        if not available_models:
            logger.error("No valid models found for evaluation")
            return {"status": "failed", "reason": "no_valid_models"}

        logger.info(f"Found {len(available_models)} valid models for evaluation")

        # Initialize W&B for overall tracking
        wandb.init(
            project="agent-forge-comprehensive-benchmark",
            name=f"agent_forge_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "models_evaluated": list(available_models.keys()),
                "quick_mode": quick_mode,
                "include_baselines": not skip_baselines,
                "include_frontier": not skip_frontier,
                "baseline_models": self.baseline_models if not skip_baselines else [],
                "frontier_models": self.frontier_models if not skip_frontier else [],
            },
        )

        # Run benchmarks for each model
        evaluation_results = {}

        for model_name, model_path in available_models.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"EVALUATING: {model_name}")
            logger.info(f"{'=' * 60}")

            try:
                result = await self.benchmark_runner.run_complete_benchmark_suite(
                    model_path,
                    f"agent-forge-{model_name}",
                    include_baselines=not skip_baselines,
                    include_frontier=not skip_frontier,
                    quick_mode=quick_mode,
                )

                evaluation_results[model_name] = result

                if result["status"] == "success":
                    logger.info(f"✅ {model_name} evaluation completed successfully")

                    # Log key metrics to W&B
                    performance = result["comparison_report"].performance_summary
                    wandb.log(
                        {
                            f"{model_name}/average_score": performance["average_score"],
                            f"{model_name}/total_benchmarks": performance[
                                "total_benchmarks"
                            ],
                            f"{model_name}/wins_vs_baseline": performance.get(
                                "wins_vs_baseline", 0
                            ),
                            f"{model_name}/wins_vs_frontier": performance.get(
                                "wins_vs_frontier", 0
                            ),
                        }
                    )

                    # Log individual benchmark scores
                    for benchmark, score in performance["benchmark_scores"].items():
                        wandb.log({f"{model_name}/{benchmark}_score": score})

                else:
                    logger.error(
                        f"❌ {model_name} evaluation failed: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                logger.error(f"❌ Exception during {model_name} evaluation: {e}")
                evaluation_results[model_name] = {"status": "failed", "error": str(e)}

        # Generate comprehensive comparison
        await self._generate_comprehensive_comparison(evaluation_results)

        # Create executive summary
        executive_summary = await self._create_executive_summary(evaluation_results)

        return {
            "status": "completed",
            "models_evaluated": len(available_models),
            "successful_evaluations": len(
                [r for r in evaluation_results.values() if r.get("status") == "success"]
            ),
            "evaluation_results": evaluation_results,
            "executive_summary": executive_summary,
            "results_directory": str(self.results_dir),
        }

    async def _generate_comprehensive_comparison(
        self, evaluation_results: dict[str, Any]
    ):
        """Generate comprehensive comparison across all Agent Forge models."""
        logger.info("Generating comprehensive model comparison")

        # Extract successful results
        successful_results = {
            name: result
            for name, result in evaluation_results.items()
            if result.get("status") == "success"
        }

        if len(successful_results) < 2:
            logger.warning("Insufficient successful results for comparison")
            return

        # Create comparison matrix
        comparison_data = []
        benchmarks = set()

        # Collect all benchmarks
        for result in successful_results.values():
            if "comparison_report" in result:
                benchmarks.update(result["comparison_report"].target_results.keys())

        # Build comparison table
        for benchmark in sorted(benchmarks):
            row = {"Benchmark": benchmark}

            for model_name, result in successful_results.items():
                if "comparison_report" in result:
                    target_results = result["comparison_report"].target_results
                    if benchmark in target_results:
                        score = target_results[benchmark].overall_score
                        row[model_name] = score
                    else:
                        row[model_name] = None

            comparison_data.append(row)

        # Create DataFrame for analysis
        df = pd.DataFrame(comparison_data)
        df = df.set_index("Benchmark")

        # Calculate statistics
        model_averages = df.mean(axis=0, skipna=True)
        model_rankings = model_averages.rank(ascending=False)

        # Save comparison data
        comparison_file = self.results_dir / "agent_forge_model_comparison.json"
        comparison_summary = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": list(successful_results.keys()),
            "benchmark_comparison": comparison_data,
            "model_averages": model_averages.to_dict(),
            "model_rankings": model_rankings.to_dict(),
            "best_model": model_averages.idxmax(),
            "best_average_score": model_averages.max(),
        }

        with open(comparison_file, "w") as f:
            json.dump(comparison_summary, f, indent=2, default=str)

        # Log to W&B
        wandb.log(
            {
                "model_comparison/best_model": comparison_summary["best_model"],
                "model_comparison/best_score": comparison_summary["best_average_score"],
                "model_comparison/models_compared": len(successful_results),
            }
        )

        # Create comparison visualization data for W&B
        comparison_table_data = []
        for model_name in successful_results:
            row = [
                model_name,
                model_averages[model_name],
                int(model_rankings[model_name]),
            ]
            comparison_table_data.append(row)

        wandb.log(
            {
                "model_comparison_table": wandb.Table(
                    data=comparison_table_data,
                    columns=["Model", "Average Score", "Rank"],
                )
            }
        )

        # Generate detailed comparison report
        await self._create_detailed_comparison_report(comparison_summary, df)

        logger.info(f"Comprehensive comparison saved: {comparison_file}")

    async def _create_detailed_comparison_report(
        self, comparison_summary: dict[str, Any], comparison_df: pd.DataFrame
    ):
        """Create detailed comparison report."""
        best_model = comparison_summary["best_model"]
        best_score = comparison_summary["best_average_score"]

        report_content = f"""# Agent Forge Model Comparison Report

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Executive Summary

This report compares the performance of all Agent Forge pipeline models across standardized benchmarks.

### Key Findings

🏆 **Best Performing Model**: **{best_model}** (Average Score: {best_score:.3f})

### Model Rankings

"""

        rankings = comparison_summary["model_rankings"]
        averages = comparison_summary["model_averages"]

        for model, rank in sorted(rankings.items(), key=lambda x: x[1]):
            avg_score = averages[model]
            report_content += f"{int(rank)}. **{model}**: {avg_score:.3f}\n"

        report_content += """

## Detailed Performance Analysis

### Benchmark Comparison Matrix

| Benchmark |"""

        models = list(comparison_summary["models_compared"])
        for model in models:
            report_content += f" {model} |"

        report_content += "\n|-----------|"
        for _ in models:
            report_content += "----------|"
        report_content += "\n"

        for _, row in comparison_df.iterrows():
            report_content += f"| {row.name} |"
            for model in models:
                score = row[model]
                if pd.isna(score):
                    report_content += " N/A |"
                else:
                    report_content += f" {score:.3f} |"
            report_content += "\n"

        # Add insights section
        report_content += """

## Performance Insights

### Strengths and Weaknesses

"""

        for model in models:
            model_scores = comparison_df[model].dropna()
            if len(model_scores) > 0:
                best_benchmark = model_scores.idxmax()
                worst_benchmark = model_scores.idxmin()

                report_content += f"""
#### {model}
- **Strongest Performance**: {best_benchmark} ({model_scores[best_benchmark]:.3f})
- **Weakest Performance**: {worst_benchmark} ({model_scores[worst_benchmark]:.3f})
- **Average Score**: {model_scores.mean():.3f}
- **Consistency**: {1 - model_scores.std():.3f} (higher is more consistent)
"""

        # Add recommendations
        report_content += f"""

## Recommendations

### Production Deployment
- **Recommended Model**: {best_model}
- **Deployment Justification**: Highest average performance across benchmarks

### Model Selection Guidelines
1. **General Purpose**: Use {best_model} for best overall performance
2. **Specific Domains**: Consider individual benchmark strengths for domain-specific applications
3. **Resource Constraints**: Evaluate model size vs. performance trade-offs

### Future Improvements
- Focus improvement efforts on weakest performing benchmarks
- Consider ensemble approaches combining strengths of different models
- Investigate domain-specific fine-tuning for specialized applications

---

*This report was automatically generated by the Agent Forge benchmarking system.*
"""

        # Save detailed report
        report_file = self.results_dir / "agent_forge_detailed_comparison.md"
        with open(report_file, "w") as f:
            f.write(report_content)

        logger.info(f"Detailed comparison report saved: {report_file}")

    async def _create_executive_summary(
        self, evaluation_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Create executive summary of all evaluations."""
        successful_results = {
            name: result
            for name, result in evaluation_results.items()
            if result.get("status") == "success"
        }

        if not successful_results:
            return {"status": "no_successful_evaluations"}

        # Find best performing model
        best_model = None
        best_score = 0.0

        model_performances = {}

        for model_name, result in successful_results.items():
            if "comparison_report" in result:
                avg_score = result["comparison_report"].performance_summary[
                    "average_score"
                ]
                model_performances[model_name] = avg_score

                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model_name

        # Calculate overall statistics
        all_scores = list(model_performances.values())

        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "models_evaluated": len(successful_results),
            "best_model": best_model,
            "best_score": best_score,
            "average_score_across_models": (
                sum(all_scores) / len(all_scores) if all_scores else 0.0
            ),
            "score_range": {
                "min": min(all_scores) if all_scores else 0.0,
                "max": max(all_scores) if all_scores else 0.0,
            },
            "model_performances": model_performances,
            "recommendations": [
                f"Deploy {best_model} for production use (score: {best_score:.3f})",
                (
                    f"Average performance across pipeline: {sum(all_scores) / len(all_scores):.3f}"
                    if all_scores
                    else "No valid scores"
                ),
                "Review detailed reports for specific benchmark analysis",
            ],
        }

        # Save executive summary
        summary_file = self.results_dir / "executive_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Create markdown summary
        md_content = f"""# Agent Forge Evaluation - Executive Summary

**Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Key Results

🎯 **Best Model**: {best_model} (Score: {best_score:.3f})
📊 **Models Evaluated**: {len(successful_results)}
📈 **Average Performance**: {summary["average_score_across_models"]:.3f}

## Model Performance Summary

"""

        for model, score in sorted(
            model_performances.items(), key=lambda x: x[1], reverse=True
        ):
            md_content += f"- **{model}**: {score:.3f}\n"

        md_content += f"""

## Recommendations

{chr(10).join(f"- {rec}" for rec in summary["recommendations"])}

---

*View detailed results in the benchmark_results directory*
"""

        md_summary_file = self.results_dir / "executive_summary.md"
        with open(md_summary_file, "w") as f:
            f.write(md_content)

        logger.info(f"Executive summary saved: {summary_file}")

        return summary


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Agent Forge Comprehensive Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with all comparisons
  python run_agent_forge_benchmark.py --full

  # Quick evaluation for testing
  python run_agent_forge_benchmark.py --quick

  # Evaluation without frontier model comparisons
  python run_agent_forge_benchmark.py --skip-frontier
        """,
    )

    # Run modes
    parser.add_argument(
        "--full", action="store_true", help="Run full comprehensive evaluation"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick evaluation with limited samples"
    )

    # Comparison options
    parser.add_argument(
        "--skip-baselines", action="store_true", help="Skip baseline model comparisons"
    )
    parser.add_argument(
        "--skip-frontier", action="store_true", help="Skip frontier model comparisons"
    )

    # Output options
    parser.add_argument(
        "--results-dir", default="./benchmark_results", help="Results output directory"
    )

    # Model validation
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate models, don't run benchmarks",
    )

    args = parser.parse_args()

    # Set default mode if none specified
    if not (args.full or args.quick or args.validate_only):
        args.full = True
        logger.info("No mode specified, defaulting to --full")

    # Initialize orchestrator
    orchestrator = AgentForgeBenchmarkOrchestrator(args.results_dir)

    if args.validate_only:
        logger.info("Running model validation only")
        validation_results = await orchestrator.validate_models()

        print(f"\n{'=' * 60}")
        print("AGENT FORGE MODEL VALIDATION")
        print(f"{'=' * 60}")

        for model_name, is_valid in validation_results.items():
            status = "✅ Valid" if is_valid else "❌ Missing"
            print(f"{model_name:25}: {status}")

        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        print(
            f"\nSummary: {valid_count}/{total_count} models available for benchmarking"
        )

        return 0 if valid_count > 0 else 1

    # Run comprehensive evaluation
    logger.info("Starting Agent Forge comprehensive benchmark evaluation")

    start_time = time.time()

    try:
        results = await orchestrator.run_comprehensive_evaluation(
            quick_mode=args.quick,
            skip_baselines=args.skip_baselines,
            skip_frontier=args.skip_frontier,
        )

        execution_time = time.time() - start_time

        print(f"\n{'=' * 80}")
        print("AGENT FORGE COMPREHENSIVE BENCHMARK RESULTS")
        print(f"{'=' * 80}")

        if results["status"] == "completed":
            print(
                f"✅ Evaluation completed successfully in {execution_time:.1f} seconds"
            )
            print(f"📊 Models evaluated: {results['models_evaluated']}")
            print(f"✅ Successful evaluations: {results['successful_evaluations']}")

            if results.get("executive_summary"):
                summary = results["executive_summary"]
                if "best_model" in summary:
                    print(
                        f"🏆 Best performing model: {summary['best_model']} ({summary['best_score']:.3f})"
                    )
                    print(
                        f"📈 Average performance: {summary['average_score_across_models']:.3f}"
                    )

            print(f"\n📁 Detailed results: {results['results_directory']}")
            print(
                "📊 W&B Dashboard: https://wandb.ai/agent-forge/agent-forge-comprehensive-benchmark"
            )

        else:
            print(f"❌ Evaluation failed: {results.get('reason', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Evaluation failed with exception: {e}")
        print(f"❌ Evaluation failed: {e}")
        return 1

    print("\n🎉 Agent Forge benchmark evaluation completed!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
