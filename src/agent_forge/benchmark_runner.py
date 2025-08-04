"""Automated Benchmark Runner.

Orchestrates the complete benchmarking pipeline for Agent Forge models:
- Automatically runs trained models through all benchmark suites
- Compares against baseline 1.5B and frontier models
- Generates comprehensive W&B reports with statistical analysis
- Creates publication-ready performance summaries
- Integrates with existing training pipeline outputs
"""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch

from agent_forge.benchmark_suite import (
    BenchmarkConfig,
    ComparisonReport,
    ComprehensiveBenchmark,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Automated benchmark runner for Agent Forge models."""

    def __init__(self, base_output_dir: str = "./benchmark_results") -> None:
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Model paths from Agent Forge pipeline
        self.pipeline_outputs = {
            "evomerge": "./evomerge_output/best_model",
            "quietstar": "./quietstar_enhanced",
            "compressed": "./final_compressed_model",
            "mastery": "./mastery_output/final_model",
        }

        # Baseline models for comparison
        self.baseline_models = [
            "microsoft/DialoGPT-small",  # 117M
            "microsoft/DialoGPT-medium",  # 345M
            "microsoft/DialoGPT-large",  # 762M
            "facebook/opt-1.3b",  # 1.3B
        ]

        # Frontier models for comparison
        self.frontier_models = [
            "microsoft/phi-2",  # 2.7B
            "mistralai/Mistral-7B-v0.1",  # 7B
            "meta-llama/Llama-2-7b-hf",  # 7B
        ]

        self.results_history = []

    async def run_complete_benchmark_suite(
        self,
        target_model_path: str,
        model_name: str,
        include_baselines: bool = True,
        include_frontier: bool = True,
        quick_mode: bool = False,
    ) -> dict[str, Any]:
        """Run complete benchmark suite for a model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = self.base_output_dir / f"{model_name}_{timestamp}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting comprehensive benchmark for {model_name}")
        logger.info(f"Results will be saved to: {run_output_dir}")

        # Create benchmark configuration
        config = BenchmarkConfig(
            model_path=target_model_path,
            model_name=model_name,
            output_dir=str(run_output_dir),
            run_mmlu=True,
            run_gsm8k=True,
            run_humaneval=True,
            run_hellaswag=True,
            run_arc=True,
            batch_size=4 if not quick_mode else 8,
            max_samples=100 if quick_mode else None,  # Quick mode for testing
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16" if torch.cuda.is_available() else "fp32",
            baseline_models=self.baseline_models if include_baselines else [],
            frontier_models=self.frontier_models if include_frontier else [],
            wandb_project="agent-forge-benchmark",
            wandb_entity=None,
        )

        # Initialize comprehensive benchmark
        benchmark = ComprehensiveBenchmark(config)

        try:
            # Run benchmark
            comparison_report = await benchmark.run_comprehensive_benchmark()

            # Save comprehensive results
            await self._save_comprehensive_results(comparison_report, run_output_dir)

            # Generate publication report
            await self._generate_publication_report(comparison_report, run_output_dir)

            # Log to history
            self.results_history.append(
                {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "output_dir": str(run_output_dir),
                    "comparison_report": comparison_report,
                }
            )

            logger.info(f"Benchmark completed successfully for {model_name}")
            return {
                "status": "success",
                "model_name": model_name,
                "output_dir": str(run_output_dir),
                "comparison_report": comparison_report,
                "wandb_url": f"https://wandb.ai/agent-forge/{config.wandb_project}",
            }

        except Exception as e:
            logger.exception(f"Benchmark failed for {model_name}: {e}")
            return {"status": "failed", "model_name": model_name, "error": str(e)}

    async def benchmark_pipeline_outputs(
        self, quick_mode: bool = False
    ) -> dict[str, Any]:
        """Benchmark all Agent Forge pipeline outputs."""
        logger.info("Benchmarking all Agent Forge pipeline outputs")

        results = {}

        for stage_name, model_path in self.pipeline_outputs.items():
            if Path(model_path).exists():
                logger.info(f"Benchmarking {stage_name} stage output")

                result = await self.run_complete_benchmark_suite(
                    model_path,
                    f"agent-forge-{stage_name}",
                    include_baselines=True,
                    include_frontier=True,
                    quick_mode=quick_mode,
                )

                results[stage_name] = result

            else:
                logger.warning(f"Model not found for {stage_name}: {model_path}")
                results[stage_name] = {
                    "status": "skipped",
                    "reason": "model_not_found",
                    "path": model_path,
                }

        # Generate cross-stage comparison
        await self._generate_cross_stage_comparison(results)

        return results

    async def _save_comprehensive_results(
        self, comparison_report: ComparisonReport, output_dir: Path
    ) -> None:
        """Save comprehensive benchmark results."""
        # Save detailed JSON report
        report_file = output_dir / "comprehensive_report.json"
        with open(report_file, "w") as f:
            # Convert dataclasses to dict for JSON serialization
            report_dict = {
                "target_model": comparison_report.target_model,
                "baseline_results": {
                    model: {
                        bench: result.__dict__ for bench, result in benchmarks.items()
                    }
                    for model, benchmarks in comparison_report.baseline_results.items()
                },
                "frontier_results": {
                    model: {
                        bench: result.__dict__ for bench, result in benchmarks.items()
                    }
                    for model, benchmarks in comparison_report.frontier_results.items()
                },
                "target_results": {
                    bench: result.__dict__
                    for bench, result in comparison_report.target_results.items()
                },
                "statistical_analysis": comparison_report.statistical_analysis,
                "performance_summary": comparison_report.performance_summary,
                "recommendations": comparison_report.recommendations,
            }
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Comprehensive report saved: {report_file}")

    async def _generate_publication_report(
        self, comparison_report: ComparisonReport, output_dir: Path
    ) -> None:
        """Generate publication-ready performance report."""
        model_name = comparison_report.target_model
        performance = comparison_report.performance_summary

        # Create markdown report
        report_content = f"""# {model_name} - Benchmark Performance Report

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Executive Summary

The **{model_name}** model has been evaluated across multiple standardized benchmarks and compared against baseline 1.5B parameter models and frontier models.

### Key Results
- **Overall Average Score**: {performance["average_score"]:.3f}
- **Benchmarks Completed**: {performance["total_benchmarks"]}
- **Baseline Wins**: {performance["wins_vs_baseline"]:.1f}/{performance["total_benchmarks"]}
- **Frontier Comparisons**: {performance["wins_vs_frontier"]:.1f}/{performance["total_benchmarks"]}

## Detailed Performance

### Benchmark Scores

| Benchmark | Score | Category | Performance |
|-----------|--------|----------|-------------|
"""

        # Add benchmark scores
        for benchmark_name, score in performance["benchmark_scores"].items():
            performance_level = (
                "🟢 Excellent"
                if score > 0.8
                else "🟡 Good"
                if score > 0.6
                else "🔴 Needs Improvement"
            )
            category = self._get_benchmark_category(benchmark_name)
            report_content += f"| {benchmark_name} | {score:.3f} | {category} | {performance_level} |\n"

        # Add statistical analysis
        report_content += """

## Statistical Analysis

### Comparison with Baseline Models
"""

        for benchmark_name, analysis in comparison_report.statistical_analysis.items():
            report_content += f"""
#### {benchmark_name}
- **Target Score**: {analysis["target_score"]:.3f}
- **Baseline Mean**: {analysis["baseline_mean"]:.3f} (±{analysis["baseline_std"]:.3f})
- **Percentile vs Baselines**: {analysis["baseline_percentile"]:.1f}%
"""

        # Add performance recommendations
        report_content += """

## Recommendations

"""
        for i, recommendation in enumerate(comparison_report.recommendations, 1):
            report_content += f"{i}. {recommendation}\n"

        # Add technical details
        report_content += f"""

## Technical Details

### Model Specifications
- **Model Name**: {model_name}
- **Evaluation Device**: {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}
- **Precision**: Mixed precision (FP16/FP32)
- **Batch Size**: 4

### Benchmark Details
- **MMLU**: Massive Multitask Language Understanding (57 subjects)
- **GSM8K**: Grade School Math 8K problems
- **HumanEval**: Python code generation tasks
- **HellaSwag**: Commonsense reasoning
- **ARC**: AI2 Reasoning Challenge

### Resource Usage
"""

        for benchmark_name in performance["execution_times"]:
            exec_time = performance["execution_times"][benchmark_name]
            memory_usage = performance["memory_usage"][benchmark_name]
            report_content += f"- **{benchmark_name}**: {exec_time:.1f}s execution, {memory_usage:.1f}GB memory\n"

        # Add methodology
        report_content += f"""

## Methodology

This evaluation follows standardized benchmarking protocols:

1. **Few-shot Learning**: 5-shot prompting for consistency
2. **Greedy Decoding**: Temperature=0.0 for reproducible results
3. **Standardized Prompts**: Using established benchmark formats
4. **Statistical Testing**: T-tests and percentile analysis for significance
5. **Fair Comparison**: Same hardware and settings across all models

## Reproducibility

All evaluation code and detailed results are available in the output directory:
```
{output_dir}
```

The evaluation can be reproduced using:
```bash
python agent_forge/benchmark_runner.py --model-path {model_name} --output-dir {output_dir}
```

---

*This report was automatically generated by the Agent Forge benchmarking system.*
"""

        # Save markdown report
        report_file = output_dir / "performance_report.md"
        with open(report_file, "w") as f:
            f.write(report_content)

        # Also create a brief summary
        summary_content = f"""# {model_name} - Performance Summary

**Overall Score**: {performance["average_score"]:.3f}

**Key Strengths**:
{self._extract_strengths(comparison_report)}

**Areas for Improvement**:
{self._extract_weaknesses(comparison_report)}

**Recommendation**: {comparison_report.recommendations[0] if comparison_report.recommendations else "No specific recommendations"}
"""

        summary_file = output_dir / "executive_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary_content)

        logger.info(f"Publication report saved: {report_file}")
        logger.info(f"Executive summary saved: {summary_file}")

    def _get_benchmark_category(self, benchmark_name: str) -> str:
        """Get category for benchmark."""
        categories = {
            "MMLU": "Knowledge & Reasoning",
            "GSM8K": "Mathematical Reasoning",
            "HumanEval": "Code Generation",
            "HellaSwag": "Commonsense Reasoning",
            "ARC": "Scientific Reasoning",
        }
        return categories.get(benchmark_name, "General")

    def _extract_strengths(self, comparison_report: ComparisonReport) -> str:
        """Extract key strengths from report."""
        strengths = []

        for benchmark_name, analysis in comparison_report.statistical_analysis.items():
            if analysis["baseline_percentile"] > 75:
                strengths.append(
                    f"- Strong {benchmark_name} performance ({analysis['target_score']:.3f})"
                )

        return (
            "\n".join(strengths)
            if strengths
            else "- Consistent performance across benchmarks"
        )

    def _extract_weaknesses(self, comparison_report: ComparisonReport) -> str:
        """Extract areas for improvement from report."""
        weaknesses = []

        for benchmark_name, analysis in comparison_report.statistical_analysis.items():
            if analysis["target_score"] < 0.5:
                weaknesses.append(
                    f"- {benchmark_name} below average ({analysis['target_score']:.3f})"
                )

        return (
            "\n".join(weaknesses) if weaknesses else "- No major weaknesses identified"
        )

    async def _generate_cross_stage_comparison(self, stage_results: dict[str, Any]) -> None:
        """Generate comparison across pipeline stages."""
        logger.info("Generating cross-stage comparison")

        # Extract successful results
        successful_results = {
            stage: result
            for stage, result in stage_results.items()
            if result.get("status") == "success"
        }

        if len(successful_results) < 2:
            logger.warning("Insufficient successful results for cross-stage comparison")
            return

        # Create comparison table
        comparison_data = []

        benchmarks = set()
        for stage_result in successful_results.values():
            if "comparison_report" in stage_result:
                benchmarks.update(
                    stage_result["comparison_report"].target_results.keys()
                )

        # Build comparison table
        for benchmark in benchmarks:
            row = {"Benchmark": benchmark}

            for stage_name, stage_result in successful_results.items():
                if "comparison_report" in stage_result:
                    target_results = stage_result["comparison_report"].target_results
                    if benchmark in target_results:
                        score = target_results[benchmark].overall_score
                        row[f"{stage_name.title()}"] = f"{score:.3f}"
                    else:
                        row[f"{stage_name.title()}"] = "N/A"

            comparison_data.append(row)

        # Save cross-stage comparison
        comparison_file = self.base_output_dir / "cross_stage_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "stages_compared": list(successful_results.keys()),
                    "comparison_data": comparison_data,
                    "summary": {
                        "total_stages": len(successful_results),
                        "benchmarks_evaluated": list(benchmarks),
                    },
                },
                f,
                indent=2,
            )

        # Create cross-stage report
        report_content = f"""# Agent Forge Pipeline - Cross-Stage Performance Comparison

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Pipeline Performance Evolution

This report compares performance across different stages of the Agent Forge pipeline:

"""

        for stage_name in successful_results:
            report_content += f"- **{stage_name.title()}**: {self.pipeline_outputs.get(stage_name, 'Unknown path')}\n"

        report_content += "\n## Performance Comparison\n\n"
        report_content += "| Benchmark |"

        for stage_name in successful_results:
            report_content += f" {stage_name.title()} |"
        report_content += "\n|-----------|"

        for _ in successful_results:
            report_content += "----------|"
        report_content += "\n"

        for row in comparison_data:
            report_content += f"| {row['Benchmark']} |"
            for stage_name in successful_results:
                score = row.get(f"{stage_name.title()}", "N/A")
                report_content += f" {score} |"
            report_content += "\n"

        # Add insights
        report_content += """

## Pipeline Insights

### Performance Progression
The Agent Forge pipeline shows the following performance evolution across stages:

"""

        # Calculate improvements
        if len(successful_results) >= 2:
            stages = list(successful_results.keys())
            first_stage = stages[0]
            last_stage = stages[-1]

            improvements = []
            for benchmark in benchmarks:
                first_score = None
                last_score = None

                if (
                    benchmark
                    in successful_results[first_stage][
                        "comparison_report"
                    ].target_results
                ):
                    first_score = (
                        successful_results[first_stage]["comparison_report"]
                        .target_results[benchmark]
                        .overall_score
                    )

                if (
                    benchmark
                    in successful_results[last_stage][
                        "comparison_report"
                    ].target_results
                ):
                    last_score = (
                        successful_results[last_stage]["comparison_report"]
                        .target_results[benchmark]
                        .overall_score
                    )

                if first_score is not None and last_score is not None:
                    improvement = ((last_score - first_score) / first_score) * 100
                    improvements.append(
                        f"- **{benchmark}**: {improvement:+.1f}% ({first_score:.3f} → {last_score:.3f})"
                    )

            report_content += "\n".join(improvements)

        report_content += """

### Recommendations for Pipeline Optimization

Based on the cross-stage analysis:

1. **Best Performing Stage**: Identify the stage with highest overall performance
2. **Bottleneck Analysis**: Focus improvement efforts on stages with minimal gains
3. **Resource Allocation**: Balance training time vs. performance improvements
4. **Production Deployment**: Consider deploying the best-performing stage

---

*This cross-stage comparison was automatically generated by the Agent Forge benchmarking system.*
"""

        cross_stage_report = self.base_output_dir / "cross_stage_report.md"
        with open(cross_stage_report, "w") as f:
            f.write(report_content)

        logger.info(f"Cross-stage comparison saved: {cross_stage_report}")

    async def quick_benchmark(self, model_path: str, model_name: str) -> dict[str, Any]:
        """Run quick benchmark for development/testing."""
        logger.info(f"Running quick benchmark for {model_name}")

        return await self.run_complete_benchmark_suite(
            model_path,
            model_name,
            include_baselines=False,  # Skip baselines for speed
            include_frontier=False,  # Skip frontier for speed
            quick_mode=True,  # Use limited samples
        )


# CLI interface
async def main() -> int:
    """Main CLI for benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge Benchmark Runner")

    # Model specification
    parser.add_argument("--model-path", help="Path to specific model to benchmark")
    parser.add_argument("--model-name", help="Name for the model")

    # Run modes
    parser.add_argument(
        "--pipeline", action="store_true", help="Benchmark all pipeline outputs"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick benchmark mode (limited samples)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir", default="./benchmark_results", help="Output directory"
    )

    # Comparison settings
    parser.add_argument(
        "--skip-baselines", action="store_true", help="Skip baseline model comparisons"
    )
    parser.add_argument(
        "--skip-frontier", action="store_true", help="Skip frontier model comparisons"
    )

    args = parser.parse_args()

    # Initialize benchmark runner
    runner = BenchmarkRunner(args.output_dir)

    if args.pipeline:
        # Benchmark entire pipeline
        logger.info("Benchmarking entire Agent Forge pipeline")
        results = await runner.benchmark_pipeline_outputs(quick_mode=args.quick)

        print(f"\n{'=' * 60}")
        print("AGENT FORGE PIPELINE BENCHMARK RESULTS")
        print(f"{'=' * 60}")

        for stage_name, result in results.items():
            print(f"\n{stage_name.title()} Stage:")
            if result["status"] == "success":
                report = result["comparison_report"]
                avg_score = report.performance_summary["average_score"]
                print(f"  ✅ Average Score: {avg_score:.3f}")
                print(
                    f"  📊 Benchmarks: {report.performance_summary['total_benchmarks']}"
                )
            else:
                print(f"  ❌ Status: {result['status']}")
                if "error" in result:
                    print(f"  💭 Error: {result['error']}")

    elif args.model_path and args.model_name:
        # Benchmark specific model
        logger.info(f"Benchmarking specific model: {args.model_name}")

        result = await runner.run_complete_benchmark_suite(
            args.model_path,
            args.model_name,
            include_baselines=not args.skip_baselines,
            include_frontier=not args.skip_frontier,
            quick_mode=args.quick,
        )

        print(f"\n{'=' * 60}")
        print(f"BENCHMARK RESULTS - {args.model_name}")
        print(f"{'=' * 60}")

        if result["status"] == "success":
            report = result["comparison_report"]
            performance = report.performance_summary

            print("\nOverall Performance:")
            print(f"  Average Score: {performance['average_score']:.3f}")
            print(f"  Benchmarks: {performance['total_benchmarks']}")

            print("\nBenchmark Scores:")
            for benchmark, score in performance["benchmark_scores"].items():
                print(f"  {benchmark}: {score:.3f}")

            print("\nTop Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {rec}")

            print(f"\nDetailed results: {result['output_dir']}")
            print(f"W&B Dashboard: {result['wandb_url']}")

        else:
            print(f"❌ Benchmark failed: {result.get('error', 'Unknown error')}")

    else:
        print(
            "Error: Must specify either --pipeline or both --model-path and --model-name"
        )
        parser.print_help()
        return 1

    print(f"\nAll results saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
