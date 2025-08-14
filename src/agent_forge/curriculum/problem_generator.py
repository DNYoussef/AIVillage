"""Problem Generator - Create targeted coding problems within difficulty bands.

Generates coding problems at specified difficulty levels with proper test cases,
canonical solutions, and grading rubrics.
"""

import logging
import random
import uuid
from pathlib import Path

from .openrouter import OpenRouterLLM
from .schemas import (
    EdgeWindow,
    Problem,
    ProblemGenerationRequest,
    ProblemGenerationResponse,
    TopicMix,
)

logger = logging.getLogger(__name__)


class ProblemGenerator:
    """Generates coding problems within specified difficulty bands."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
    ):
        """Initialize ProblemGenerator.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for problem generation
            temperature: Sampling temperature (higher for creativity)
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

        # Load template
        template_path = Path(__file__).parent / "templates" / "problem_generator.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        logger.info(f"ProblemGenerator initialized with model {model}")

    def _validate_generation_request(self, domain: str, edge: EdgeWindow, topic_mix: list[TopicMix], n: int) -> None:
        """Validate problem generation request parameters."""

        if not domain:
            raise ValueError("Domain cannot be empty")

        if edge.low >= edge.high:
            raise ValueError(f"Invalid edge window: {edge.low} >= {edge.high}")

        if not topic_mix:
            raise ValueError("Topic mix cannot be empty")

        # Check topic weights sum approximately to 1.0
        total_weight = sum(topic.weight for topic in topic_mix)
        if abs(total_weight - 1.0) > 0.15:  # Allow some tolerance
            logger.warning(f"Topic weights sum to {total_weight:.2f}, not 1.0")

        if n <= 0:
            raise ValueError(f"Number of problems must be positive: {n}")

        if n > 200:  # Practical limit for single batch
            raise ValueError(f"Too many problems requested in single batch: {n}")

    def _allocate_problems_by_topic(self, topic_mix: list[TopicMix], n_total: int) -> dict[str, int]:
        """Allocate problem counts by topic based on weights."""

        allocations = {}
        remaining = n_total

        # Allocate by weight, ensuring at least 1 per topic
        for i, topic in enumerate(topic_mix):
            if i == len(topic_mix) - 1:  # Last topic gets remainder
                allocations[topic.topic] = remaining
            else:
                allocated = max(1, round(topic.weight * n_total))
                allocations[topic.topic] = min(allocated, remaining - (len(topic_mix) - i - 1))
                remaining -= allocations[topic.topic]

        logger.debug(f"Topic allocation: {allocations}")
        return allocations

    def _distribute_difficulties(self, edge: EdgeWindow, n_problems: int) -> list[float]:
        """Generate difficulty values distributed across edge window."""

        difficulties = []
        edge_width = edge.high - edge.low

        # Create a mix of difficulties favoring the center
        for i in range(n_problems):
            if i % 3 == 0:
                # Center-focused (normal distribution)
                center = (edge.low + edge.high) / 2
                std = edge_width / 6  # 3 standard deviations span the window
                difficulty = random.gauss(center, std)
            elif i % 3 == 1:
                # Uniform distribution
                difficulty = random.uniform(edge.low, edge.high)
            else:
                # Slightly outside edge for boundary testing
                jitter = edge_width * 0.05  # 5% jitter outside
                if random.choice([True, False]):
                    difficulty = edge.low - random.uniform(0, jitter)
                else:
                    difficulty = edge.high + random.uniform(0, jitter)

            # Clamp to reasonable bounds
            difficulty = max(0.1, min(0.9, difficulty))
            difficulties.append(round(difficulty, 2))

        return difficulties

    async def generate_problems(
        self,
        domain: str,
        edge: EdgeWindow,
        topic_mix: list[TopicMix],
        n: int = 10,
        style: str = "default",
        batch_size: int = 5,
    ) -> ProblemGenerationResponse:
        """Generate problems within specified difficulty and topic constraints.

        Args:
            domain: Problem domain (e.g., "coding-python")
            edge: Difficulty window for problems
            topic_mix: Topic distribution with weights
            n: Number of problems to generate
            style: Problem style (default, concise, verbose)
            batch_size: Problems per LLM call (for rate limiting)

        Returns:
            ProblemGenerationResponse with generated problems

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_generation_request(domain, edge, topic_mix, n)

        logger.info(f"Generating {n} problems for {domain} in difficulty {edge.low:.2f}-{edge.high:.2f}")

        # Allocate problems by topic
        topic_allocations = self._allocate_problems_by_topic(topic_mix, n)

        all_problems = []

        # Generate problems in batches by topic
        for topic, count in topic_allocations.items():
            if count <= 0:
                continue

            logger.debug(f"Generating {count} problems for topic: {topic}")

            # Generate difficulties for this topic
            topic_difficulties = self._distribute_difficulties(edge, count)

            # Split into batches
            for i in range(0, count, batch_size):
                batch_end = min(i + batch_size, count)
                batch_count = batch_end - i
                batch_difficulties = topic_difficulties[i:batch_end]

                # Create single-topic mix for this batch
                single_topic_mix = [TopicMix(topic=topic, weight=1.0)]

                # Create request
                request = ProblemGenerationRequest(
                    domain=domain,
                    edge=edge,
                    topic_mix=single_topic_mix,
                    n=batch_count,
                    style=style,
                )

                # Render prompt
                prompt = self.llm_client.render_template(
                    self.template,
                    domain=request.domain,
                    edge=request.edge,
                    topic_mix=request.topic_mix,
                    n=request.n,
                    style=request.style,
                )

                logger.debug(f"Generating batch {i // batch_size + 1} for {topic}: {batch_count} problems")

                # Get LLM response
                try:
                    response = await self.llm_client.invoke_with_schema(
                        prompt=prompt,
                        schema_class=ProblemGenerationResponse,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=4096,
                        max_schema_retries=3,
                    )

                    # Post-process problems
                    processed_problems = self._post_process_problems(response.problems, batch_difficulties, topic)

                    all_problems.extend(processed_problems)

                    logger.debug(f"Generated {len(processed_problems)} problems for {topic}")

                except Exception as e:
                    logger.error(f"Failed to generate problems for {topic} batch {i // batch_size + 1}: {e}")
                    # Continue with other batches
                    continue

        if not all_problems:
            raise RuntimeError("Failed to generate any problems")

        # Create final response
        final_response = ProblemGenerationResponse(
            ok=True,
            msg=f"generated {len(all_problems)} problems",
            problems=all_problems,
        )

        logger.info(f"Successfully generated {len(all_problems)} problems")
        return final_response

    def _post_process_problems(
        self,
        problems: list[Problem],
        target_difficulties: list[float],
        expected_topic: str,
    ) -> list[Problem]:
        """Post-process generated problems for quality and consistency."""

        processed = []

        for i, problem in enumerate(problems):
            # Ensure unique ID
            if not problem.id or any(p.id == problem.id for p in processed):
                problem.id = f"{expected_topic}_{uuid.uuid4().hex[:8]}"

            # Adjust difficulty if provided
            if i < len(target_difficulties):
                problem.difficulty = target_difficulties[i]

            # Ensure topic matches
            if problem.topic != expected_topic:
                logger.debug(f"Correcting topic from {problem.topic} to {expected_topic}")
                problem.topic = expected_topic

            # Validate unit tests format
            validated_tests = self._validate_unit_tests(problem.unit_tests)
            problem.unit_tests = validated_tests

            # Basic quality checks
            if self._validate_problem_quality(problem):
                processed.append(problem)
            else:
                logger.warning(f"Skipping low-quality problem: {problem.id}")

        return processed

    def _validate_unit_tests(self, unit_tests: list[str]) -> list[str]:
        """Validate and clean unit test formats."""

        validated = []

        for test in unit_tests:
            # Clean up common formatting issues
            test = test.strip()

            # Ensure it looks like a test assertion
            if not any(keyword in test.lower() for keyword in ["assert", "test", "check", "=="]):
                logger.debug(f"Suspicious unit test: {test}")

            # Basic syntax validation (very permissive)
            if test and len(test) < 500:  # Reasonable length
                validated.append(test)

        return validated[:10]  # Limit number of tests

    def _validate_problem_quality(self, problem: Problem) -> bool:
        """Basic quality validation for generated problems."""

        # Check required fields
        if not all(
            [
                problem.id,
                problem.topic,
                problem.statement,
                problem.canonical_answer,
                problem.rubric,
            ]
        ):
            return False

        # Check reasonable lengths
        if len(problem.statement) < 20 or len(problem.statement) > 2000:
            return False

        if len(problem.canonical_answer) < 10 or len(problem.canonical_answer) > 5000:
            return False

        # Check difficulty is in reasonable range
        if not (0.0 <= problem.difficulty <= 1.0):
            return False

        # Check for unit tests
        if not problem.unit_tests:
            logger.debug(f"Problem {problem.id} has no unit tests")

        return True

    async def generate_problems_with_feedback(
        self,
        domain: str,
        edge: EdgeWindow,
        topic_mix: list[TopicMix],
        n: int,
        quality_threshold: float = 0.8,
        max_iterations: int = 3,
    ) -> ProblemGenerationResponse:
        """Generate problems with iterative quality improvement.

        Args:
            domain: Problem domain
            edge: Difficulty window
            topic_mix: Topic distribution
            n: Number of problems
            quality_threshold: Minimum quality score (0-1)
            max_iterations: Maximum refinement iterations

        Returns:
            ProblemGenerationResponse with high-quality problems
        """

        best_response = None
        best_quality = 0.0

        for iteration in range(max_iterations):
            logger.info(f"Problem generation iteration {iteration + 1}/{max_iterations}")

            try:
                response = await self.generate_problems(domain, edge, topic_mix, n)
                quality_score = self._assess_batch_quality(response.problems)

                logger.info(f"Iteration {iteration + 1} quality: {quality_score:.2f}")

                if quality_score > best_quality:
                    best_response = response
                    best_quality = quality_score

                if quality_score >= quality_threshold:
                    logger.info(f"Quality threshold reached: {quality_score:.2f}")
                    break

            except Exception as e:
                logger.error(f"Generation iteration {iteration + 1} failed: {e}")
                continue

        if best_response is None:
            raise RuntimeError("All generation iterations failed")

        logger.info(f"Final quality score: {best_quality:.2f}")
        return best_response

    def _assess_batch_quality(self, problems: list[Problem]) -> float:
        """Assess overall quality of a problem batch (0-1 score)."""

        if not problems:
            return 0.0

        scores = []

        for problem in problems:
            score = 0.0

            # Statement quality (clarity, length)
            if 50 <= len(problem.statement) <= 500:
                score += 0.2

            # Canonical answer quality
            if 20 <= len(problem.canonical_answer) <= 1000:
                score += 0.2
            if "def " in problem.canonical_answer:  # Has function definition
                score += 0.1

            # Rubric quality
            if 20 <= len(problem.rubric) <= 300:
                score += 0.2

            # Unit tests quality
            if problem.unit_tests and len(problem.unit_tests) >= 2:
                score += 0.2
            if any("assert" in test for test in problem.unit_tests):
                score += 0.1

            scores.append(min(1.0, score))

        return sum(scores) / len(scores)


async def generate_coding_problems(
    api_key: str,
    domain: str,
    edge: EdgeWindow,
    topic_mix: list[TopicMix],
    n: int = 10,
    model: str = "anthropic/claude-3-5-sonnet-20241022",
    **kwargs,
) -> ProblemGenerationResponse:
    """Convenience function to generate problems with minimal setup.

    Args:
        api_key: OpenRouter API key
        domain: Problem domain
        edge: Difficulty window
        topic_mix: Topic distribution
        n: Number of problems
        model: Model to use for generation
        **kwargs: Additional arguments for ProblemGenerator

    Returns:
        ProblemGenerationResponse with generated problems
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        generator = ProblemGenerator(client, model=model)
        return await generator.generate_problems(domain, edge, topic_mix, n, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os

    async def demo():
        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        if api_key == "demo-key":
            print("🔧 Demo mode: Set OPENROUTER_API_KEY for live testing")
            return

        # Create sample edge and topic mix
        edge = EdgeWindow(low=0.55, high=0.75)
        topic_mix = [
            TopicMix(topic="string_manipulation", weight=0.4),
            TopicMix(topic="list_operations", weight=0.3),
            TopicMix(topic="basic_algorithms", weight=0.3),
        ]

        try:
            result = await generate_coding_problems(
                api_key=api_key,
                domain="coding-python",
                edge=edge,
                topic_mix=topic_mix,
                n=5,
            )

            print(f"✅ Generated {len(result.problems)} problems")
            for problem in result.problems[:2]:  # Show first 2
                print(f"\n📝 {problem.id} (difficulty: {problem.difficulty:.2f})")
                print(f"   Topic: {problem.topic}")
                print(f"   Statement: {problem.statement[:100]}...")
                print(f"   Tests: {len(problem.unit_tests)} unit tests")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    asyncio.run(demo())
