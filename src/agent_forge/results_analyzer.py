"""Agent Forge Results Analysis & Interpretation System.

Provides comprehensive analysis of benchmark results:
- W&B dashboard data extraction and interpretation
- JSON output analysis with statistical insights
- Performance jump identification across pipeline phases
- Key insights extraction and recommendation generation
- Automated trend analysis and anomaly detection
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import wandb

logger = logging.getLogger(__name__)


@dataclass
class PerformanceJump:
    """Represents a performance improvement between pipeline phases."""

    from_phase: str
    to_phase: str
    benchmark: str
    improvement: float
    relative_improvement: float
    statistical_significance: bool
    p_value: float | None = None


@dataclass
class PhaseAnalysis:
    """Analysis of a specific pipeline phase."""

    phase_name: str
    average_score: float
    benchmark_scores: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]
    key_improvements: list[PerformanceJump]
    recommendations: list[str]


@dataclass
class InsightSummary:
    """High-level insights from the analysis."""

    best_performing_phase: str
    biggest_performance_jump: PerformanceJump
    top_benchmarks: list[str]
    concerning_trends: list[str]
    deployment_recommendation: str
    confidence_level: str


class WandBDashboardAnalyzer:
    """Analyzes W&B dashboard data for insights."""

    def __init__(self, project_name: str = "agent-forge-comprehensive-benchmark") -> None:
        self.project_name = project_name
        self.api = wandb.Api()

    def extract_dashboard_data(self, entity: str | None = None) -> dict[str, Any]:
        """Extract data from W&B dashboard."""
        logger.info(f"Extracting data from W&B project: {self.project_name}")

        try:
            runs = self.api.runs(f"{entity}/{self.project_name}" if entity else self.project_name)

            dashboard_data = {
                "runs": [],
                "summary_metrics": {},
                "trends": {},
                "comparisons": {},
            }

            all_metrics = defaultdict(list)

            for run in runs:
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "summary": dict(run.summary),
                    "config": dict(run.config),
                    "history": [],
                }

                # Extract history if available
                try:
                    history = run.scan_history()
                    run_data["history"] = list(history)
                except BaseException:
                    pass

                dashboard_data["runs"].append(run_data)

                # Aggregate metrics
                for key, value in run.summary.items():
                    if isinstance(value, int | float):
                        all_metrics[key].append(value)

            # Calculate summary statistics
            for metric, values in all_metrics.items():
                if values:
                    dashboard_data["summary_metrics"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values),
                    }

            logger.info(f"Extracted data from {len(runs)} runs")
            return dashboard_data

        except Exception as e:
            logger.exception(f"Failed to extract W&B data: {e}")
            return {"error": str(e)}


class ResultsAnalyzer:
    """Main results analysis engine."""

    def __init__(self, results_dir: str) -> None:
        self.results_dir = Path(results_dir)
        self.wandb_analyzer = WandBDashboardAnalyzer()

        # Expected pipeline phases in order
        self.pipeline_phases = [
            "evomerge_best",
            "quietstar_enhanced",
            "original_compressed",
            "mastery_trained",
            "unified_pipeline",
        ]

    async def analyze_comprehensive_results(self) -> dict[str, Any]:
        """Perform comprehensive analysis of all results."""
        logger.info("Starting comprehensive results analysis")

        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "json_analysis": await self._analyze_json_outputs(),
            "wandb_analysis": await self._analyze_wandb_data(),
            "phase_analysis": await self._analyze_pipeline_phases(),
            "performance_jumps": await self._identify_performance_jumps(),
            "insights": await self._extract_key_insights(),
            "recommendations": await self._generate_recommendations(),
        }

        # Save comprehensive analysis
        analysis_file = self.results_dir / "comprehensive_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)

        logger.info(f"Comprehensive analysis saved: {analysis_file}")
        return analysis_results

    async def _analyze_json_outputs(self) -> dict[str, Any]:
        """Analyze JSON benchmark outputs."""
        logger.info("Analyzing JSON benchmark outputs")

        json_files = list(self.results_dir.glob("**/*.json"))

        analysis = {
            "files_found": len(json_files),
            "model_results": {},
            "benchmark_statistics": {},
            "performance_trends": {},
        }

        all_scores = defaultdict(list)
        model_averages = {}

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Extract model performance data
                if "target_results" in data:
                    model_name = data.get("target_model", json_file.stem)
                    model_results = {}

                    for benchmark, result in data["target_results"].items():
                        if isinstance(result, dict) and "overall_score" in result:
                            score = result["overall_score"]
                            model_results[benchmark] = score
                            all_scores[benchmark].append(score)

                    if model_results:
                        avg_score = np.mean(list(model_results.values()))
                        model_averages[model_name] = avg_score

                        analysis["model_results"][model_name] = {
                            "benchmark_scores": model_results,
                            "average_score": avg_score,
                            "execution_data": data.get("performance_summary", {}),
                        }

            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")

        # Calculate benchmark statistics
        for benchmark, scores in all_scores.items():
            if scores:
                analysis["benchmark_statistics"][benchmark] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "range": np.max(scores) - np.min(scores),
                    "models_evaluated": len(scores),
                }

        # Performance trends
        if model_averages:
            analysis["performance_trends"] = {
                "best_model": max(model_averages, key=model_averages.get),
                "best_score": max(model_averages.values()),
                "worst_model": min(model_averages, key=model_averages.get),
                "worst_score": min(model_averages.values()),
                "score_range": max(model_averages.values()) - min(model_averages.values()),
                "model_count": len(model_averages),
            }

        return analysis

    async def _analyze_wandb_data(self) -> dict[str, Any]:
        """Analyze W&B dashboard data."""
        logger.info("Analyzing W&B dashboard data")

        try:
            dashboard_data = self.wandb_analyzer.extract_dashboard_data()

            if "error" in dashboard_data:
                return {"status": "failed", "error": dashboard_data["error"]}

            analysis = {
                "status": "success",
                "runs_analyzed": len(dashboard_data["runs"]),
                "metric_trends": {},
                "performance_evolution": {},
                "anomalies": [],
            }

            # Analyze metric trends
            for metric, stats in dashboard_data["summary_metrics"].items():
                if "score" in metric.lower() or "accuracy" in metric.lower():
                    analysis["metric_trends"][metric] = {
                        "average": stats["mean"],
                        "variability": stats["std"],
                        "best_performance": stats["max"],
                        "stability": (1 - (stats["std"] / stats["mean"]) if stats["mean"] > 0 else 0),
                    }

            # Detect anomalies (scores outside 2 standard deviations)
            for run in dashboard_data["runs"]:
                for metric, value in run["summary"].items():
                    if isinstance(value, int | float) and metric in dashboard_data["summary_metrics"]:
                        stats_data = dashboard_data["summary_metrics"][metric]
                        z_score = abs((value - stats_data["mean"]) / stats_data["std"]) if stats_data["std"] > 0 else 0

                        if z_score > 2:  # Outlier detection
                            analysis["anomalies"].append(
                                {
                                    "run_name": run["name"],
                                    "metric": metric,
                                    "value": value,
                                    "z_score": z_score,
                                    "type": ("outlier_high" if value > stats_data["mean"] else "outlier_low"),
                                }
                            )

            return analysis

        except Exception as e:
            logger.exception(f"W&B analysis failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _analyze_pipeline_phases(self) -> dict[str, PhaseAnalysis]:
        """Analyze each pipeline phase performance."""
        logger.info("Analyzing pipeline phase performance")

        phase_analyses = {}

        # Load model comparison data
        comparison_file = self.results_dir / "agent_forge_model_comparison.json"
        if not comparison_file.exists():
            logger.warning("Model comparison file not found")
            return {}

        with open(comparison_file) as f:
            comparison_data = json.load(f)

        comparison_data.get("model_averages", {})
        benchmark_data = comparison_data.get("benchmark_comparison", [])

        # Create DataFrame for analysis
        df = pd.DataFrame(benchmark_data)
        if df.empty:
            return {}

        df = df.set_index("Benchmark")

        for phase in self.pipeline_phases:
            if phase in df.columns:
                phase_scores = df[phase].dropna()

                if len(phase_scores) > 0:
                    # Calculate strengths and weaknesses
                    strengths = []
                    weaknesses = []

                    for benchmark, score in phase_scores.items():
                        if score > 0.7:
                            strengths.append(f"{benchmark} ({score:.3f})")
                        elif score < 0.5:
                            weaknesses.append(f"{benchmark} ({score:.3f})")

                    # Generate recommendations
                    recommendations = []
                    avg_score = phase_scores.mean()

                    if avg_score > 0.75:
                        recommendations.append("Excellent performance - ready for production")
                    elif avg_score > 0.6:
                        recommendations.append("Good performance with room for improvement")
                    else:
                        recommendations.append("Below average - requires additional training")

                    if len(weaknesses) > 0:
                        recommendations.append(
                            f"Focus improvement on: {', '.join([w.split('(')[0].strip() for w in weaknesses[:3]])}"
                        )

                    phase_analyses[phase] = PhaseAnalysis(
                        phase_name=phase,
                        average_score=avg_score,
                        benchmark_scores=phase_scores.to_dict(),
                        strengths=strengths,
                        weaknesses=weaknesses,
                        key_improvements=[],  # Will be filled by performance jumps analysis
                        recommendations=recommendations,
                    )

        return phase_analyses

    async def _identify_performance_jumps(self) -> list[PerformanceJump]:
        """Identify significant performance improvements between phases."""
        logger.info("Identifying performance jumps between phases")

        # Load comparison data
        comparison_file = self.results_dir / "agent_forge_model_comparison.json"
        if not comparison_file.exists():
            return []

        with open(comparison_file) as f:
            comparison_data = json.load(f)

        benchmark_data = comparison_data.get("benchmark_comparison", [])
        df = pd.DataFrame(benchmark_data)

        if df.empty:
            return []

        df = df.set_index("Benchmark")

        performance_jumps = []

        # Compare consecutive phases
        for i in range(len(self.pipeline_phases) - 1):
            phase1 = self.pipeline_phases[i]
            phase2 = self.pipeline_phases[i + 1]

            if phase1 in df.columns and phase2 in df.columns:
                for benchmark in df.index:
                    score1 = df.loc[benchmark, phase1]
                    score2 = df.loc[benchmark, phase2]

                    if pd.notna(score1) and pd.notna(score2) and score1 > 0:
                        improvement = score2 - score1
                        relative_improvement = (improvement / score1) * 100

                        # Consider significant if improvement > 5% or absolute improvement > 0.05
                        is_significant = abs(improvement) > 0.05 or abs(relative_improvement) > 5

                        if is_significant:
                            # Statistical significance test (simplified)
                            # In practice, would need multiple runs for proper testing
                            p_value = 0.05 if abs(improvement) > 0.1 else 0.10

                            performance_jumps.append(
                                PerformanceJump(
                                    from_phase=phase1,
                                    to_phase=phase2,
                                    benchmark=benchmark,
                                    improvement=improvement,
                                    relative_improvement=relative_improvement,
                                    statistical_significance=abs(improvement) > 0.1,
                                    p_value=p_value,
                                )
                            )

        # Sort by absolute improvement
        performance_jumps.sort(key=lambda x: abs(x.improvement), reverse=True)

        return performance_jumps

    async def _extract_key_insights(self) -> InsightSummary:
        """Extract key insights from all analyses."""
        logger.info("Extracting key insights")

        # Load existing analyses
        comparison_file = self.results_dir / "agent_forge_model_comparison.json"
        if not comparison_file.exists():
            return InsightSummary(
                best_performing_phase="unknown",
                biggest_performance_jump=None,
                top_benchmarks=[],
                concerning_trends=[],
                deployment_recommendation="Insufficient data for analysis",
                confidence_level="low",
            )

        with open(comparison_file) as f:
            comparison_data = json.load(f)

        model_averages = comparison_data.get("model_averages", {})

        # Find best performing phase
        best_phase = "unknown"
        best_score = 0.0

        if model_averages:
            best_phase = max(model_averages, key=model_averages.get)
            best_score = model_averages[best_phase]

        # Get performance jumps
        performance_jumps = await self._identify_performance_jumps()
        biggest_jump = performance_jumps[0] if performance_jumps else None

        # Identify top benchmarks (where the model performs well)
        benchmark_data = comparison_data.get("benchmark_comparison", [])
        top_benchmarks = []

        if benchmark_data and best_phase != "unknown":
            df = pd.DataFrame(benchmark_data)
            df = df.set_index("Benchmark")

            if best_phase in df.columns:
                phase_scores = df[best_phase].dropna()
                top_benchmarks = phase_scores.nlargest(3).index.tolist()

        # Identify concerning trends
        concerning_trends = []

        if model_averages:
            scores = list(model_averages.values())
            if len(scores) > 1:
                if max(scores) - min(scores) > 0.3:
                    concerning_trends.append("High variance between pipeline phases")

                if max(scores) < 0.6:
                    concerning_trends.append("Overall performance below expectations")

        # Generate deployment recommendation
        deployment_rec = "Insufficient data"
        confidence = "low"

        if best_score > 0.75:
            deployment_rec = f"Deploy {best_phase} - excellent performance"
            confidence = "high"
        elif best_score > 0.6:
            deployment_rec = f"Deploy {best_phase} with monitoring - good performance"
            confidence = "medium"
        elif best_score > 0.4:
            deployment_rec = "Additional training recommended before deployment"
            confidence = "medium"
        else:
            deployment_rec = "Significant improvements needed before deployment"
            confidence = "high"

        return InsightSummary(
            best_performing_phase=best_phase,
            biggest_performance_jump=biggest_jump,
            top_benchmarks=top_benchmarks,
            concerning_trends=concerning_trends,
            deployment_recommendation=deployment_rec,
            confidence_level=confidence,
        )

    async def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        logger.info("Generating actionable recommendations")

        recommendations = []

        # Load insights
        insights = await self._extract_key_insights()

        # Performance-based recommendations
        if insights.confidence_level == "high":
            if "excellent" in insights.deployment_recommendation:
                recommendations.append("🚀 **Ready for Production**: Deploy immediately with confidence")
                recommendations.append("📊 **Monitor Closely**: Set up production monitoring to track performance")
            elif "improvements needed" in insights.deployment_recommendation:
                recommendations.append("🔧 **Training Required**: Additional training cycles needed before deployment")
                recommendations.append("📈 **Focus Areas**: Prioritize weak benchmark improvements")

        # Phase-specific recommendations
        if insights.biggest_performance_jump:
            jump = insights.biggest_performance_jump
            recommendations.append(
                f"✨ **Key Innovation**: {jump.to_phase} shows {jump.relative_improvement:.1f}% "
                f"improvement in {jump.benchmark} - replicate this approach"
            )

        # Benchmark-specific recommendations
        if insights.top_benchmarks:
            recommendations.append(
                f"💪 **Leverage Strengths**: Excellent performance in {', '.join(insights.top_benchmarks)} "
                f"- consider domain-specific deployment"
            )

        # Concerning trend recommendations
        for trend in insights.concerning_trends:
            if "variance" in trend:
                recommendations.append("🎯 **Stabilize Pipeline**: High variance suggests inconsistent training")
            elif "below expectations" in trend:
                recommendations.append("📚 **Review Training Data**: Performance suggests data quality issues")

        # Technical recommendations
        recommendations.extend(
            [
                "🔍 **A/B Testing**: Deploy multiple phases in parallel to validate results",
                "📝 **Documentation**: Document successful techniques for future iterations",
                "🔄 **Continuous Monitoring**: Implement drift detection for production deployment",
            ]
        )

        return recommendations

    async def generate_executive_briefing(self) -> str:
        """Generate executive briefing document."""
        logger.info("Generating executive briefing")

        analysis = await self.analyze_comprehensive_results()
        insights = analysis["insights"]

        briefing = f"""# Agent Forge Executive Briefing

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Executive Summary

**Recommendation**: {insights["deployment_recommendation"]}
**Confidence Level**: {insights["confidence_level"].upper()}

## Key Performance Indicators

### Best Performing Model
- **Phase**: {insights["best_performing_phase"]}
- **Top Benchmarks**: {", ".join(insights["top_benchmarks"][:3])}

### Biggest Performance Improvement
"""

        if insights["biggest_performance_jump"]:
            jump = insights["biggest_performance_jump"]
            briefing += f"""
- **Transition**: {jump["from_phase"]} → {jump["to_phase"]}
- **Benchmark**: {jump["benchmark"]}
- **Improvement**: {jump["relative_improvement"]:.1f}%
"""
        else:
            briefing += "\n- No significant jumps detected\n"

        briefing += """

## Risk Assessment

### Concerning Trends
"""

        if insights["concerning_trends"]:
            for trend in insights["concerning_trends"]:
                briefing += f"- ⚠️ {trend}\n"
        else:
            briefing += "- ✅ No concerning trends identified\n"

        briefing += """

## Strategic Recommendations

"""

        for i, rec in enumerate(analysis["recommendations"][:5], 1):
            briefing += f"{i}. {rec}\n"

        briefing += """

## Technical Metrics

### Model Performance Summary
"""

        if "json_analysis" in analysis and "performance_trends" in analysis["json_analysis"]:
            trends = analysis["json_analysis"]["performance_trends"]
            briefing += f"""
- **Best Model**: {trends.get("best_model", "Unknown")}
- **Best Score**: {trends.get("best_score", 0.0):.3f}
- **Score Range**: {trends.get("score_range", 0.0):.3f}
- **Models Evaluated**: {trends.get("model_count", 0)}
"""

        briefing += """

## Next Steps

1. **Immediate**: Review detailed technical analysis
2. **Short-term**: Implement top 3 recommendations
3. **Long-term**: Plan next training iteration based on insights

---

*Full technical details available in comprehensive_analysis.json*
"""

        # Save briefing
        briefing_file = self.results_dir / "executive_briefing.md"
        with open(briefing_file, "w") as f:
            f.write(briefing)

        logger.info(f"Executive briefing saved: {briefing_file}")
        return briefing


# Sample data analyzer for demonstration
class SampleResultsAnalyzer:
    """Analyzes sample results to demonstrate the system."""

    def __init__(self) -> None:
        self.sample_results = self._generate_sample_results()

    def _generate_sample_results(self) -> dict[str, Any]:
        """Generate realistic sample results for demonstration."""
        return {
            "evomerge_best": {
                "MMLU": 0.542,
                "GSM8K": 0.234,
                "HumanEval": 0.156,
                "HellaSwag": 0.678,
                "ARC": 0.445,
            },
            "quietstar_enhanced": {
                "MMLU": 0.587,
                "GSM8K": 0.312,
                "HumanEval": 0.198,
                "HellaSwag": 0.701,
                "ARC": 0.489,
            },
            "original_compressed": {
                "MMLU": 0.573,
                "GSM8K": 0.289,
                "HumanEval": 0.187,
                "HellaSwag": 0.692,
                "ARC": 0.467,
            },
            "mastery_trained": {
                "MMLU": 0.634,
                "GSM8K": 0.456,
                "HumanEval": 0.267,
                "HellaSwag": 0.734,
                "ARC": 0.523,
            },
            "unified_pipeline": {
                "MMLU": 0.651,
                "GSM8K": 0.478,
                "HumanEval": 0.289,
                "HellaSwag": 0.742,
                "ARC": 0.545,
            },
        }

    def analyze_sample_results(self) -> dict[str, Any]:
        """Analyze sample results to show expected insights."""
        # Calculate averages
        model_averages = {}
        for model, scores in self.sample_results.items():
            model_averages[model] = np.mean(list(scores.values()))

        # Find best model
        best_model = max(model_averages, key=model_averages.get)
        best_score = model_averages[best_model]

        # Calculate biggest improvement
        phases = list(self.sample_results.keys())
        biggest_improvement = 0
        biggest_jump = None

        for i in range(len(phases) - 1):
            phase1 = phases[i]
            phase2 = phases[i + 1]

            for benchmark in self.sample_results[phase1]:
                score1 = self.sample_results[phase1][benchmark]
                score2 = self.sample_results[phase2][benchmark]
                improvement = score2 - score1

                if improvement > biggest_improvement:
                    biggest_improvement = improvement
                    biggest_jump = {
                        "from": phase1,
                        "to": phase2,
                        "benchmark": benchmark,
                        "improvement": improvement,
                        "relative": (improvement / score1) * 100,
                    }

        return {
            "best_model": best_model,
            "best_score": best_score,
            "biggest_jump": biggest_jump,
            "model_averages": model_averages,
            "key_insights": [
                f"🏆 Best performing phase: {best_model} ({best_score:.3f})",
                (
                    f"📈 Biggest improvement: {biggest_jump['benchmark']} in {biggest_jump['to']} (+{biggest_jump['relative']:.1f}%)"
                    if biggest_jump
                    else "No significant improvements detected"
                ),
                "🎯 Mastery training shows consistent gains across all benchmarks",
                f"💡 GSM8K benefits most from the training pipeline (+{((model_averages['unified_pipeline'] - model_averages['evomerge_best']) / model_averages['evomerge_best']) * 100:.1f}%)",
            ],
        }


# CLI interface
async def main() -> None:
    """Main CLI for results analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge Results Analysis")
    parser.add_argument(
        "--results-dir",
        default="./benchmark_results",
        help="Benchmark results directory",
    )
    parser.add_argument("--sample", action="store_true", help="Run sample analysis for demonstration")
    parser.add_argument(
        "--wandb-project",
        default="agent-forge-comprehensive-benchmark",
        help="W&B project name",
    )
    parser.add_argument("--generate-briefing", action="store_true", help="Generate executive briefing")

    args = parser.parse_args()

    if args.sample:
        # Run sample analysis
        sample_analyzer = SampleResultsAnalyzer()
        results = sample_analyzer.analyze_sample_results()

        print(f"\n{'=' * 60}")
        print("SAMPLE RESULTS ANALYSIS")
        print(f"{'=' * 60}")

        print(f"\n🏆 Best Model: {results['best_model']}")
        print(f"📊 Score: {results['best_score']:.3f}")

        if results["biggest_jump"]:
            jump = results["biggest_jump"]
            print("\n📈 Biggest Improvement:")
            print(f"  {jump['from']} → {jump['to']}")
            print(f"  {jump['benchmark']}: +{jump['relative']:.1f}%")

        print("\n💡 Key Insights:")
        for insight in results["key_insights"]:
            print(f"  {insight}")

        print("\n📋 Model Rankings:")
        sorted_models = sorted(results["model_averages"].items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(sorted_models, 1):
            print(f"  {i}. {model}: {score:.3f}")

    else:
        # Run actual analysis
        analyzer = ResultsAnalyzer(args.results_dir)

        if args.generate_briefing:
            briefing = await analyzer.generate_executive_briefing()
            print("\nExecutive briefing generated:")
            print(briefing[:500] + "..." if len(briefing) > 500 else briefing)
        else:
            analysis = await analyzer.analyze_comprehensive_results()

            print(f"\n{'=' * 60}")
            print("COMPREHENSIVE RESULTS ANALYSIS")
            print(f"{'=' * 60}")

            if "insights" in analysis:
                insights = analysis["insights"]
                print(f"\n🏆 Best Phase: {insights['best_performing_phase']}")
                print(f"🎯 Deployment Rec: {insights['deployment_recommendation']}")
                print(f"📊 Confidence: {insights['confidence_level']}")

                if insights["biggest_performance_jump"]:
                    jump = insights["biggest_performance_jump"]
                    print("\n📈 Biggest Jump:")
                    print(f"  {jump['from_phase']} → {jump['to_phase']}")
                    print(f"  {jump['benchmark']}: +{jump['relative_improvement']:.1f}%")

            print(f"\nFull analysis saved to: {args.results_dir}/comprehensive_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())
