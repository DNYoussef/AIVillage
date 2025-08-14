"""Edge Controller - Maintain optimal difficulty via edge nudging.

Monitors performance and adjusts the edge-of-chaos window to keep
accuracy in the productive struggle range (55-75%).
"""

import logging
import statistics
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterLLM
from .schemas import (
    ControllerRequest,
    ControllerResponse,
    EdgeConstraints,
    EdgeDelta,
    EdgeWindow,
)

logger = logging.getLogger(__name__)


class EdgeController:
    """Controls difficulty edges to maintain target accuracy range."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.1,
    ):
        """Initialize EdgeController.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for edge control decisions
            temperature: Very low temperature for consistent control
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

        # Load template
        template_path = Path(__file__).parent / "templates" / "edge_controller.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        # Control parameters
        self.min_window_width = 0.15  # Minimum edge width
        self.max_window_width = 0.35  # Maximum edge width
        self.min_difficulty = 0.10  # Absolute minimum difficulty
        self.max_difficulty = 0.90  # Absolute maximum difficulty
        self.nudge_factor = 0.5  # How aggressively to nudge (0-1)

        logger.info("EdgeController initialized")

    def _calculate_local_nudge(
        self,
        window_accuracy: float,
        current_edge: EdgeWindow,
        constraints: EdgeConstraints,
    ) -> EdgeWindow:
        """Calculate edge nudge using local control theory.

        Args:
            window_accuracy: Current accuracy in the edge window
            current_edge: Current edge window
            constraints: Edge constraints and targets

        Returns:
            New edge window after nudging
        """
        target_center = (constraints.target_low + constraints.target_high) / 2
        accuracy_error = window_accuracy - target_center

        # Calculate nudge magnitude based on error size
        error_magnitude = abs(accuracy_error)

        if error_magnitude < 0.05:  # Very close to target
            nudge_magnitude = 0.02
        elif error_magnitude < 0.15:  # Moderate error
            nudge_magnitude = error_magnitude * self.nudge_factor
        else:  # Large error
            nudge_magnitude = min(0.15, error_magnitude * self.nudge_factor)

        # Direction: if accuracy too high, increase difficulty (shift right)
        # If accuracy too low, decrease difficulty (shift left)
        if accuracy_error > 0:  # Too easy
            shift = nudge_magnitude
        else:  # Too hard
            shift = -nudge_magnitude

        # Calculate new bounds
        new_low = current_edge.low + shift
        new_high = current_edge.high + shift

        # Ensure minimum width is preserved
        current_width = current_edge.high - current_edge.low
        if current_width < self.min_window_width:
            # Expand window symmetrically
            expansion = (self.min_window_width - current_width) / 2
            new_low -= expansion
            new_high += expansion

        # Clamp to absolute bounds
        if new_low < self.min_difficulty:
            new_low = self.min_difficulty
            new_high = new_low + current_width

        if new_high > self.max_difficulty:
            new_high = self.max_difficulty
            new_low = new_high - current_width

        # Final bounds check
        new_low = max(self.min_difficulty, new_low)
        new_high = min(self.max_difficulty, new_high)

        # Ensure low < high
        if new_low >= new_high:
            center = (new_low + new_high) / 2
            new_low = center - self.min_window_width / 2
            new_high = center + self.min_window_width / 2

        return EdgeWindow(low=new_low, high=new_high)

    def _validate_edge_window(self, edge: EdgeWindow) -> bool:
        """Validate that edge window meets basic constraints."""

        if edge.low >= edge.high:
            return False

        width = edge.high - edge.low
        if width < self.min_window_width or width > self.max_window_width:
            return False

        if edge.low < self.min_difficulty or edge.high > self.max_difficulty:
            return False

        return True

    async def nudge_edge(
        self,
        window_accuracy: float,
        current_edge: EdgeWindow,
        constraints: EdgeConstraints,
        use_llm_control: bool = True,
        use_local_fallback: bool = True,
    ) -> ControllerResponse:
        """Nudge edge window to maintain target accuracy.

        Args:
            window_accuracy: Current accuracy in the edge window (0.0-1.0)
            current_edge: Current edge window
            constraints: Edge control constraints
            use_llm_control: Whether to use LLM for control decisions
            use_local_fallback: Use local control if LLM fails

        Returns:
            ControllerResponse with new edge window and delta

        Raises:
            ValueError: If parameters are invalid
        """
        if not (0.0 <= window_accuracy <= 1.0):
            raise ValueError(f"Window accuracy must be 0-1: {window_accuracy}")

        if not self._validate_edge_window(current_edge):
            raise ValueError(f"Invalid current edge: {current_edge.low}-{current_edge.high}")

        logger.info(
            f"Nudging edge: accuracy {window_accuracy:.1%} in window {current_edge.low:.2f}-{current_edge.high:.2f}"
        )

        # Check if nudge is needed
        in_target_range = constraints.target_low <= window_accuracy <= constraints.target_high

        if in_target_range:
            # Small fine-tuning nudge toward center
            target_center = (constraints.target_low + constraints.target_high) / 2
            center_distance = abs(window_accuracy - target_center)

            if center_distance < 0.03:  # Very close to center
                logger.info("Edge is optimal, no nudge needed")
                return ControllerResponse(
                    ok=True,
                    msg="optimal - no nudge needed",
                    new_edge=current_edge,
                    delta=EdgeDelta(low=0.0, high=0.0),
                )

        # Try LLM control
        if use_llm_control:
            try:
                request = ControllerRequest(
                    window_accuracy=window_accuracy,
                    current_edge=current_edge,
                    constraints=constraints,
                )

                # Render prompt
                prompt = self.llm_client.render_template(
                    self.template,
                    window_accuracy=request.window_accuracy,
                    current_edge=request.current_edge,
                    constraints=request.constraints,
                )

                response = await self.llm_client.invoke_with_schema(
                    prompt=prompt,
                    schema_class=ControllerResponse,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=1024,
                    max_schema_retries=2,
                )

                # Validate LLM response
                if self._validate_edge_window(response.new_edge):
                    logger.info(
                        f"LLM control: {current_edge.low:.2f}-{current_edge.high:.2f} → {response.new_edge.low:.2f}-{response.new_edge.high:.2f}"
                    )
                    return response
                else:
                    logger.warning("LLM produced invalid edge window")

            except Exception as e:
                logger.error(f"LLM control failed: {e}")

        # Fallback to local control
        if use_local_fallback:
            logger.info("Using local edge control")

            new_edge = self._calculate_local_nudge(window_accuracy, current_edge, constraints)

            # Calculate delta
            delta = EdgeDelta(
                low=new_edge.low - current_edge.low,
                high=new_edge.high - current_edge.high,
            )

            logger.info(
                f"Local control: {current_edge.low:.2f}-{current_edge.high:.2f} → {new_edge.low:.2f}-{new_edge.high:.2f}"
            )

            return ControllerResponse(ok=True, msg="nudged locally", new_edge=new_edge, delta=delta)

        # No control available - return unchanged
        return ControllerResponse(
            ok=False,
            msg="no control available",
            new_edge=current_edge,
            delta=EdgeDelta(low=0.0, high=0.0),
        )

    def analyze_edge_stability(
        self,
        accuracy_history: list[float],
        edge_history: list[EdgeWindow],
        target_range: tuple = (0.55, 0.75),
    ) -> dict[str, Any]:
        """Analyze edge control stability over time.

        Args:
            accuracy_history: List of accuracy values over time
            edge_history: List of edge windows over time
            target_range: Target accuracy range (low, high)

        Returns:
            Dictionary with stability analysis
        """
        if len(accuracy_history) != len(edge_history):
            raise ValueError("History lengths must match")

        if not accuracy_history:
            return {"error": "No history data provided"}

        # Time in target range
        in_target = sum(1 for acc in accuracy_history if target_range[0] <= acc <= target_range[1])
        target_percentage = in_target / len(accuracy_history) * 100

        # Edge window stability
        edge_centers = [(edge.low + edge.high) / 2 for edge in edge_history]
        edge_widths = [edge.high - edge.low for edge in edge_history]

        center_stability = statistics.stdev(edge_centers) if len(edge_centers) > 1 else 0
        width_stability = statistics.stdev(edge_widths) if len(edge_widths) > 1 else 0

        # Control effectiveness
        large_adjustments = sum(
            1 for i in range(1, len(edge_history)) if abs(edge_centers[i] - edge_centers[i - 1]) > 0.1
        )

        # Convergence analysis
        if len(accuracy_history) >= 5:
            recent_accuracy = accuracy_history[-5:]
            recent_variance = statistics.variance(recent_accuracy)
            converging = recent_variance < 0.01  # Low recent variance
        else:
            recent_variance = None
            converging = None

        return {
            "total_periods": len(accuracy_history),
            "time_in_target_percent": target_percentage,
            "average_accuracy": statistics.mean(accuracy_history),
            "accuracy_stability": statistics.stdev(accuracy_history) if len(accuracy_history) > 1 else 0,
            "edge_center_stability": center_stability,
            "edge_width_stability": width_stability,
            "large_adjustments": large_adjustments,
            "adjustment_rate": large_adjustments / len(edge_history) * 100,
            "recent_variance": recent_variance,
            "appears_converging": converging,
            "control_quality": self._assess_control_quality(target_percentage, center_stability, large_adjustments),
        }

    def _assess_control_quality(self, target_percentage: float, center_stability: float, large_adjustments: int) -> str:
        """Assess overall control quality based on metrics."""

        if target_percentage >= 80 and center_stability < 0.05 and large_adjustments <= 2:
            return "excellent"
        elif target_percentage >= 65 and center_stability < 0.10 and large_adjustments <= 4:
            return "good"
        elif target_percentage >= 50 and center_stability < 0.15:
            return "fair"
        else:
            return "poor"

    async def adaptive_control_loop(
        self,
        accuracy_measurements: list[float],
        initial_edge: EdgeWindow,
        constraints: EdgeConstraints,
        max_iterations: int = 10,
    ) -> list[ControllerResponse]:
        """Run adaptive control loop to find stable edge.

        Args:
            accuracy_measurements: Sequence of accuracy measurements
            initial_edge: Starting edge window
            constraints: Control constraints
            max_iterations: Maximum control iterations

        Returns:
            List of ControllerResponse objects showing control progression
        """
        responses = []
        current_edge = initial_edge

        for i, accuracy in enumerate(accuracy_measurements):
            if i >= max_iterations:
                break

            logger.info(f"Control iteration {i + 1}: accuracy {accuracy:.1%}")

            response = await self.nudge_edge(
                window_accuracy=accuracy,
                current_edge=current_edge,
                constraints=constraints,
            )

            responses.append(response)

            # Update for next iteration
            current_edge = response.new_edge

            # Check for convergence
            if i > 0 and abs(response.delta.low) < 0.01 and abs(response.delta.high) < 0.01:
                logger.info("Control loop converged")
                break

        return responses

    def recommend_control_parameters(
        self,
        historical_accuracy: list[float],
        problem_difficulty_distribution: list[float],
    ) -> dict[str, float]:
        """Recommend control parameters based on historical data.

        Args:
            historical_accuracy: Historical accuracy measurements
            problem_difficulty_distribution: Distribution of problem difficulties

        Returns:
            Dictionary with recommended control parameters
        """
        if not historical_accuracy or not problem_difficulty_distribution:
            return {
                "nudge_factor": self.nudge_factor,
                "min_window_width": self.min_window_width,
                "max_window_width": self.max_window_width,
                "recommended_center": 0.65,
            }

        # Analyze historical performance
        avg_accuracy = statistics.mean(historical_accuracy)
        accuracy_stability = statistics.stdev(historical_accuracy) if len(historical_accuracy) > 1 else 0

        # Analyze problem distribution
        problem_center = statistics.mean(problem_difficulty_distribution)
        problem_spread = (
            statistics.stdev(problem_difficulty_distribution) if len(problem_difficulty_distribution) > 1 else 0
        )

        # Adjust nudge factor based on stability
        if accuracy_stability < 0.05:  # Very stable
            recommended_nudge_factor = 0.3  # Gentle nudges
        elif accuracy_stability < 0.15:  # Moderate stability
            recommended_nudge_factor = 0.5  # Standard nudges
        else:  # Unstable
            recommended_nudge_factor = 0.7  # Stronger nudges

        # Adjust window width based on problem spread
        if problem_spread < 0.1:  # Narrow problem range
            recommended_window_width = 0.15  # Narrow window
        elif problem_spread < 0.25:  # Moderate spread
            recommended_window_width = 0.20  # Standard window
        else:  # Wide spread
            recommended_window_width = 0.25  # Wider window

        # Recommend center based on historical performance
        if avg_accuracy < 0.45:  # Too hard
            recommended_center = problem_center - 0.1
        elif avg_accuracy > 0.85:  # Too easy
            recommended_center = problem_center + 0.1
        else:  # Good range
            recommended_center = problem_center

        return {
            "nudge_factor": recommended_nudge_factor,
            "min_window_width": self.min_window_width,
            "max_window_width": self.max_window_width,
            "recommended_window_width": recommended_window_width,
            "recommended_center": recommended_center,
            "analysis": {
                "historical_avg_accuracy": avg_accuracy,
                "accuracy_stability": accuracy_stability,
                "problem_center": problem_center,
                "problem_spread": problem_spread,
            },
        }


async def control_curriculum_edge(
    api_key: str,
    window_accuracy: float,
    current_edge: EdgeWindow,
    constraints: EdgeConstraints,
    model: str = "openai/gpt-4o-mini",
    **kwargs,
) -> ControllerResponse:
    """Convenience function for edge control with minimal setup.

    Args:
        api_key: OpenRouter API key
        window_accuracy: Current accuracy in edge window
        current_edge: Current edge window
        constraints: Edge control constraints
        model: Model to use for control decisions
        **kwargs: Additional arguments for EdgeController

    Returns:
        ControllerResponse with edge adjustment
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        controller = EdgeController(client, model=model)
        return await controller.nudge_edge(window_accuracy, current_edge, constraints, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os

    async def demo():
        # Create test scenarios
        scenarios = [
            {"accuracy": 0.45, "description": "Too hard - needs easier problems"},
            {"accuracy": 0.85, "description": "Too easy - needs harder problems"},
            {"accuracy": 0.65, "description": "Perfect range - minimal adjustment"},
            {"accuracy": 0.30, "description": "Very hard - major adjustment needed"},
        ]

        current_edge = EdgeWindow(low=0.55, high=0.75)
        constraints = EdgeConstraints(target_low=0.55, target_high=0.75, problem_budget=1000)

        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        if api_key == "demo-key":
            print("🔧 Demo mode: Testing local edge control")

            dummy_client = OpenRouterLLM(api_key="dummy")
            controller = EdgeController(dummy_client)

            for scenario in scenarios:
                accuracy = scenario["accuracy"]
                description = scenario["description"]

                print(f"\n📊 Scenario: {description}")
                print(f"   Current accuracy: {accuracy:.1%}")
                print(f"   Current edge: {current_edge.low:.2f} - {current_edge.high:.2f}")

                # Use local control
                new_edge = controller._calculate_local_nudge(accuracy, current_edge, constraints)
                delta = EdgeDelta(
                    low=new_edge.low - current_edge.low,
                    high=new_edge.high - current_edge.high,
                )

                print(f"   New edge: {new_edge.low:.2f} - {new_edge.high:.2f}")
                print(f"   Delta: {delta.low:+.3f} / {delta.high:+.3f}")

                direction = "harder" if delta.low > 0 else "easier" if delta.low < 0 else "unchanged"
                print(f"   Direction: {direction}")

            # Test stability analysis
            print("\n📈 Testing stability analysis...")
            accuracy_history = [0.45, 0.52, 0.61, 0.58, 0.63, 0.67, 0.65, 0.66]
            edge_history = [
                EdgeWindow(low=0.55, high=0.75),
                EdgeWindow(low=0.52, high=0.72),
                EdgeWindow(low=0.53, high=0.73),
                EdgeWindow(low=0.54, high=0.74),
                EdgeWindow(low=0.55, high=0.75),
                EdgeWindow(low=0.56, high=0.76),
                EdgeWindow(low=0.57, high=0.77),
                EdgeWindow(low=0.57, high=0.77),
            ]

            analysis = controller.analyze_edge_stability(accuracy_history, edge_history)
            print(f"   Time in target: {analysis['time_in_target_percent']:.1f}%")
            print(f"   Control quality: {analysis['control_quality']}")
            print(f"   Appears converging: {analysis['appears_converging']}")

            return

        # Live API test
        print("🎯 Testing live edge control...")

        try:
            result = await control_curriculum_edge(
                api_key=api_key,
                window_accuracy=0.45,  # Too hard
                current_edge=current_edge,
                constraints=constraints,
            )

            print("✅ Edge control complete")
            print(f"   Original: {current_edge.low:.2f} - {current_edge.high:.2f}")
            print(f"   New edge: {result.new_edge.low:.2f} - {result.new_edge.high:.2f}")
            print(f"   Delta: {result.delta.low:+.3f} / {result.delta.high:+.3f}")
            print(f"   Message: {result.msg}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    asyncio.run(demo())
