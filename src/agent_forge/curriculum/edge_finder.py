"""Edge Finder - Identify model's edge-of-chaos band for productive struggle.

Analyzes telemetry data to find the optimal difficulty window where the model
achieves 55-75% accuracy, representing the sweet spot for learning.
"""

import logging
import statistics
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterLLM
from .schemas import (
    DifficultyScale,
    EdgeAssessmentRequest,
    EdgeAssessmentResponse,
    EdgeConstraints,
    TelemetryEntry,
)

logger = logging.getLogger(__name__)


class EdgeFinder:
    """Finds optimal edge-of-chaos band from model telemetry data."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        temperature: float = 0.3,
    ):
        """Initialize EdgeFinder.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for edge analysis
            temperature: Sampling temperature
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

        # Load template
        template_path = Path(__file__).parent / "templates" / "edge_finder.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        logger.info(f"EdgeFinder initialized with model {model}")

    def _analyze_telemetry_locally(
        self, telemetry: list[TelemetryEntry], constraints: EdgeConstraints
    ) -> dict[str, Any]:
        """Perform local analysis of telemetry data for additional insights."""

        # Group by difficulty bins
        bins = {}
        for entry in telemetry:
            bin_key = round(entry.difficulty * 10) / 10  # 0.1 bins
            if bin_key not in bins:
                bins[bin_key] = {"correct": 0, "total": 0, "entries": []}

            bins[bin_key]["total"] += 1
            if entry.correct:
                bins[bin_key]["correct"] += 1
            bins[bin_key]["entries"].append(entry)

        # Calculate accuracy per bin
        bin_accuracies = {}
        for bin_key, data in bins.items():
            bin_accuracies[bin_key] = data["correct"] / data["total"]

        # Find potential edge bands
        edge_candidates = []
        for bin_key, accuracy in bin_accuracies.items():
            if constraints.target_low <= accuracy <= constraints.target_high:
                edge_candidates.append(
                    {
                        "difficulty": bin_key,
                        "accuracy": accuracy,
                        "sample_size": bins[bin_key]["total"],
                    }
                )

        # Overall statistics
        total_correct = sum(1 for entry in telemetry if entry.correct)
        overall_accuracy = total_correct / len(telemetry) if telemetry else 0

        # Difficulty distribution
        difficulties = [entry.difficulty for entry in telemetry]

        analysis = {
            "total_entries": len(telemetry),
            "overall_accuracy": overall_accuracy,
            "difficulty_range": {
                "min": min(difficulties) if difficulties else 0,
                "max": max(difficulties) if difficulties else 1,
                "mean": statistics.mean(difficulties) if difficulties else 0.5,
                "median": statistics.median(difficulties) if difficulties else 0.5,
            },
            "bins": bins,
            "bin_accuracies": bin_accuracies,
            "edge_candidates": sorted(edge_candidates, key=lambda x: x["sample_size"], reverse=True),
            "in_target_range": len(edge_candidates),
            "recommendations": self._generate_recommendations(bin_accuracies, constraints),
        }

        return analysis

    def _generate_recommendations(self, bin_accuracies: dict[float, float], constraints: EdgeConstraints) -> list[str]:
        """Generate recommendations based on local analysis."""
        recommendations = []

        # Check if we have good coverage in target range
        target_bins = [
            bin_key
            for bin_key, acc in bin_accuracies.items()
            if constraints.target_low <= acc <= constraints.target_high
        ]

        if len(target_bins) == 0:
            recommendations.append("No difficulty bins in target accuracy range - need broader sampling")
        elif len(target_bins) < 3:
            recommendations.append("Limited coverage in target range - consider more diverse difficulty sampling")

        # Check for edge-of-chaos indicators
        sorted_bins = sorted(bin_accuracies.items())
        if len(sorted_bins) >= 3:
            # Look for steep accuracy drops (indicators of chaos edge)
            for i in range(len(sorted_bins) - 1):
                curr_diff, curr_acc = sorted_bins[i]
                next_diff, next_acc = sorted_bins[i + 1]

                if curr_acc - next_acc > 0.3:  # Steep drop
                    recommendations.append(f"Potential chaos edge detected around difficulty {curr_diff:.1f}")

        # Sample size recommendations
        low_sample_bins = [
            bin_key for bin_key, data in bin_accuracies.items() if bin_key in target_bins and len(target_bins) < 5
        ]

        if low_sample_bins:
            recommendations.append("Some target bins have low sample sizes - consider more data collection")

        return recommendations[:3]  # Limit to top 3 recommendations

    async def find_edge(
        self,
        domain: str,
        telemetry: list[TelemetryEntry],
        difficulty_scale: DifficultyScale | None = None,
        constraints: EdgeConstraints | None = None,
    ) -> EdgeAssessmentResponse:
        """Find optimal edge-of-chaos band from telemetry data.

        Args:
            domain: Problem domain (e.g., "coding-python")
            telemetry: List of telemetry entries with difficulty and correctness
            difficulty_scale: Scale bounds (defaults to 0.0-1.0)
            constraints: Edge finding constraints (defaults to 55-75% target)

        Returns:
            EdgeAssessmentResponse with optimal edge band and generation plan

        Raises:
            ValueError: If telemetry data is insufficient or invalid
        """
        # Validate inputs
        if not telemetry:
            raise ValueError("Telemetry data cannot be empty")

        if len(telemetry) < 5:
            raise ValueError("Need at least 5 telemetry entries for reliable edge detection")

        # Set defaults
        if difficulty_scale is None:
            difficulty_scale = DifficultyScale(min=0.0, max=1.0)

        if constraints is None:
            constraints = EdgeConstraints(target_low=0.55, target_high=0.75, problem_budget=1000)

        # Create request
        request = EdgeAssessmentRequest(
            domain=domain,
            telemetry=telemetry,
            difficulty_scale=difficulty_scale,
            constraints=constraints,
        )

        # Perform local analysis
        local_analysis = self._analyze_telemetry_locally(telemetry, constraints)
        logger.info(f"Local analysis: {local_analysis['in_target_range']} bins in target range")

        # Render prompt with telemetry data
        prompt = self.llm_client.render_template(
            self.template,
            domain=request.domain,
            telemetry=request.telemetry,
            difficulty_scale=request.difficulty_scale,
            constraints=request.constraints,
        )

        logger.info(f"Finding edge for {domain} with {len(telemetry)} telemetry entries")

        # Get LLM assessment
        response = await self.llm_client.invoke_with_schema(
            prompt=prompt,
            schema_class=EdgeAssessmentResponse,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            max_schema_retries=3,
        )

        # Validate response against local analysis
        self._validate_response(response, local_analysis, constraints)

        logger.info(f"Edge found: {response.edge.low:.1%} - {response.edge.high:.1%}")
        return response

    def _validate_response(
        self,
        response: EdgeAssessmentResponse,
        local_analysis: dict[str, Any],
        constraints: EdgeConstraints,
    ) -> None:
        """Validate LLM response against local analysis and constraints."""

        # Check edge window is reasonable
        edge_width = response.edge.high - response.edge.low
        if edge_width < 0.1:
            logger.warning(f"Edge window very narrow: {edge_width:.2f}")
        elif edge_width > 0.4:
            logger.warning(f"Edge window very wide: {edge_width:.2f}")

        # Check edge overlaps with target range
        target_center = (constraints.target_low + constraints.target_high) / 2
        edge_center = (response.edge.low + response.edge.high) / 2

        center_distance = abs(edge_center - target_center)
        if center_distance > 0.2:
            logger.warning(f"Edge center far from target center: {center_distance:.2f}")

        # Check topic mix sums to reasonable total
        total_weight = sum(topic.weight for topic in response.topic_mix)
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(f"Topic weights don't sum to 1.0: {total_weight:.2f}")

        # Check generation plan is reasonable
        if response.generation_plan.n_total != constraints.problem_budget:
            logger.warning(
                f"Generation plan total ({response.generation_plan.n_total}) "
                f"doesn't match budget ({constraints.problem_budget})"
            )

        # Check distribution sums to budget
        total_distribution = sum(point.count for point in response.distribution)
        if abs(total_distribution - response.generation_plan.n_total) > 50:
            logger.warning(
                f"Distribution total ({total_distribution}) doesn't match "
                f"generation plan ({response.generation_plan.n_total})"
            )

        logger.debug("Response validation completed")

    async def analyze_edge_stability(
        self,
        domain: str,
        telemetry_batches: list[list[TelemetryEntry]],
        constraints: EdgeConstraints | None = None,
    ) -> dict[str, Any]:
        """Analyze edge stability across multiple telemetry batches.

        Args:
            domain: Problem domain
            telemetry_batches: List of telemetry batches to compare
            constraints: Edge finding constraints

        Returns:
            Dictionary with stability analysis results
        """
        if constraints is None:
            constraints = EdgeConstraints()

        edges = []
        for i, batch in enumerate(telemetry_batches):
            try:
                response = await self.find_edge(domain, batch, constraints=constraints)
                edges.append(
                    {
                        "batch": i,
                        "edge": response.edge,
                        "topic_mix": response.topic_mix,
                        "generation_plan": response.generation_plan,
                    }
                )
                logger.info(f"Batch {i}: Edge {response.edge.low:.2f}-{response.edge.high:.2f}")
            except Exception as e:
                logger.error(f"Failed to find edge for batch {i}: {e}")
                continue

        if len(edges) < 2:
            return {"error": "Need at least 2 successful edge detections for stability analysis"}

        # Calculate stability metrics
        low_bounds = [edge["edge"].low for edge in edges]
        high_bounds = [edge["edge"].high for edge in edges]
        widths = [edge["edge"].high - edge["edge"].low for edge in edges]

        stability_analysis = {
            "num_batches": len(edges),
            "edge_bounds": {
                "low": {
                    "mean": statistics.mean(low_bounds),
                    "stdev": statistics.stdev(low_bounds) if len(low_bounds) > 1 else 0,
                    "range": max(low_bounds) - min(low_bounds),
                },
                "high": {
                    "mean": statistics.mean(high_bounds),
                    "stdev": statistics.stdev(high_bounds) if len(high_bounds) > 1 else 0,
                    "range": max(high_bounds) - min(high_bounds),
                },
            },
            "width_stability": {
                "mean": statistics.mean(widths),
                "stdev": statistics.stdev(widths) if len(widths) > 1 else 0,
                "coefficient_of_variation": statistics.stdev(widths) / statistics.mean(widths)
                if len(widths) > 1 and statistics.mean(widths) > 0
                else 0,
            },
            "stability_score": self._calculate_stability_score(low_bounds, high_bounds, widths),
            "edges": edges,
        }

        return stability_analysis

    def _calculate_stability_score(
        self, low_bounds: list[float], high_bounds: list[float], widths: list[float]
    ) -> float:
        """Calculate a stability score (0-1, higher is more stable)."""

        # Penalize high variance in bounds
        low_cv = statistics.stdev(low_bounds) / statistics.mean(low_bounds) if statistics.mean(low_bounds) > 0 else 1
        high_cv = (
            statistics.stdev(high_bounds) / statistics.mean(high_bounds) if statistics.mean(high_bounds) > 0 else 1
        )
        width_cv = statistics.stdev(widths) / statistics.mean(widths) if statistics.mean(widths) > 0 else 1

        # Average coefficient of variation (lower is more stable)
        avg_cv = (low_cv + high_cv + width_cv) / 3

        # Convert to stability score (invert and normalize)
        stability_score = max(0, 1 - min(1, avg_cv / 0.5))  # CV > 0.5 = very unstable

        return stability_score


async def find_model_edge(
    api_key: str,
    domain: str,
    telemetry: list[TelemetryEntry],
    model: str = "anthropic/claude-3-5-sonnet-20241022",
    **kwargs,
) -> EdgeAssessmentResponse:
    """Convenience function to find edge with minimal setup.

    Args:
        api_key: OpenRouter API key
        domain: Problem domain
        telemetry: Telemetry data
        model: Model to use for analysis
        **kwargs: Additional arguments for EdgeFinder

    Returns:
        EdgeAssessmentResponse with edge analysis
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        finder = EdgeFinder(client, model=model)
        return await finder.find_edge(domain, telemetry, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import random

    async def demo():
        # Generate sample telemetry
        telemetry = []
        for i in range(50):
            difficulty = random.uniform(0.2, 0.9)
            # Simulate edge-of-chaos at 0.55-0.75
            if 0.55 <= difficulty <= 0.75:
                correct_prob = 0.65  # Target range
            elif difficulty < 0.4 or difficulty > 0.8:
                correct_prob = 0.25  # Too easy/hard
            else:
                correct_prob = 0.45  # Transition zones

            correct = random.random() < correct_prob
            telemetry.append(TelemetryEntry(task_id=f"demo_task_{i:03d}", difficulty=difficulty, correct=correct))

        # Mock API key for demo
        try:
            import os

            api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

            if api_key == "demo-key":
                print("🔧 Demo mode: Set OPENROUTER_API_KEY for live API testing")
                return

            result = await find_model_edge(api_key=api_key, domain="coding-python", telemetry=telemetry)

            print(f"✅ Edge found: {result.edge.low:.1%} - {result.edge.high:.1%}")
            print(f"📊 Topics: {len(result.topic_mix)}")
            print(f"🎯 Generation plan: {result.generation_plan.n_total} problems")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    asyncio.run(demo())
