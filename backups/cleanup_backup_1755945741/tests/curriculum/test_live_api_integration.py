"""Live API Integration Tests for Frontier Curriculum Engine.

Tests the complete system with real OpenRouter API calls to validate
production readiness and API contract compliance.

These tests require OPENROUTER_API_KEY environment variable.
"""

import asyncio
import logging
import os
from pathlib import Path

# Test imports
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_forge.curriculum import (
    CurriculumOrchestrator,
    EdgeConstraints,
    EdgeFinder,
    EdgeWindow,
    Grader,
    HintGenerator,
    MasteryTracker,
    NumericJitterPolicy,
    OpenRouterLLM,
    Problem,
    ProblemGenerator,
    TelemetryEntry,
    TopicMix,
    VariantMaker,
    VariantPolicy,
    run_full_curriculum_pipeline,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
SKIP_LIVE_TESTS = API_KEY is None

# Test models - use cheaper models for testing
TEST_MODELS = {
    "fast": "openai/gpt-4o-mini",
    "smart": "anthropic/claude-3-5-haiku-20241022",
    "creative": "openai/gpt-4o-mini",
}


def requires_api_key(func):
    """Decorator to skip tests if API key not available."""
    return pytest.mark.skipif(SKIP_LIVE_TESTS, reason="OPENROUTER_API_KEY not set")(func)


@pytest.fixture
def live_llm():
    """Fixture providing live OpenRouter client."""
    if not API_KEY:
        pytest.skip("OPENROUTER_API_KEY not available")
    return OpenRouterLLM(api_key=API_KEY, model=TEST_MODELS["fast"])


@pytest.fixture
def sample_telemetry_realistic():
    """Fixture providing realistic telemetry data."""
    import random

    telemetry = []
    for i in range(50):
        difficulty = random.uniform(0.2, 0.9)
        # Simulate realistic accuracy curve
        if difficulty < 0.4:
            correct_prob = 0.85  # Easy problems
        elif 0.4 <= difficulty < 0.6:
            correct_prob = 0.65  # Medium problems
        elif 0.6 <= difficulty < 0.8:
            correct_prob = 0.45  # Hard problems
        else:
            correct_prob = 0.25  # Very hard problems

        correct = random.random() < correct_prob
        telemetry.append(
            TelemetryEntry(
                task_id=f"realistic_task_{i:03d}",
                difficulty=round(difficulty, 2),
                correct=correct,
            )
        )

    return telemetry


class TestLiveAPIIntegration:
    """Test live API integration with real OpenRouter calls."""

    @requires_api_key
    @pytest.mark.asyncio
    async def test_live_edge_detection(self, live_llm, sample_telemetry_realistic):
        """Test live edge detection with real API."""
        edge_finder = EdgeFinder(live_llm, model=TEST_MODELS["smart"])

        start_time = time.time()

        result = await edge_finder.find_edge(
            domain="coding-python",
            telemetry=sample_telemetry_realistic,
            constraints=EdgeConstraints(target_low=0.55, target_high=0.75, problem_budget=100),
        )

        duration = time.time() - start_time

        # Validate response
        assert result.ok is True
        assert result.edge.low < result.edge.high
        assert 0.1 <= result.edge.low <= 0.9
        assert 0.1 <= result.edge.high <= 0.9
        assert len(result.topic_mix) > 0
        assert result.generation_plan.n_total == 100

        # Check performance
        assert duration < 60, f"Edge detection took {duration:.1f}s, should be under 60s"

        # Validate topic mix weights
        total_weight = sum(topic.weight for topic in result.topic_mix)
        assert 0.8 <= total_weight <= 1.2, f"Topic weights sum to {total_weight}, should be ~1.0"

        logger.info(f"Edge detection: {result.edge.low:.2f}-{result.edge.high:.2f} in {duration:.1f}s")

    @requires_api_key
    @pytest.mark.asyncio
    async def test_live_problem_generation(self, live_llm):
        """Test live problem generation with real API."""
        problem_gen = ProblemGenerator(live_llm, model=TEST_MODELS["creative"])

        edge = EdgeWindow(low=0.55, high=0.75)
        topic_mix = [
            TopicMix(topic="string_manipulation", weight=0.6),
            TopicMix(topic="list_operations", weight=0.4),
        ]

        start_time = time.time()

        result = await problem_gen.generate_problems(
            domain="coding-python",
            edge=edge,
            topic_mix=topic_mix,
            n=3,  # Small number for testing
            batch_size=2,
        )

        duration = time.time() - start_time

        # Validate response
        assert result.ok is True
        assert len(result.problems) > 0

        # Check problem quality
        for problem in result.problems:
            assert problem.id
            assert len(problem.statement) > 20
            assert len(problem.canonical_answer) > 10
            assert problem.rubric
            assert 0.0 <= problem.difficulty <= 1.0
            assert problem.topic in ["string_manipulation", "list_operations"]

            # Should have unit tests
            assert len(problem.unit_tests) > 0
            for test in problem.unit_tests:
                assert "assert" in test.lower() or "test" in test.lower()

        logger.info(f"Generated {len(result.problems)} problems in {duration:.1f}s")

    @requires_api_key
    @pytest.mark.asyncio
    async def test_live_variant_generation(self, live_llm):
        """Test live variant generation with real API."""
        variant_maker = VariantMaker(live_llm, model=TEST_MODELS["fast"])

        base_problem = Problem(
            id="test_live_001",
            topic="string_manipulation",
            difficulty=0.6,
            statement="Write a function that counts the number of vowels in a string.",
            canonical_answer="def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')",
            rubric="Function correctly counts vowels (a, e, i, o, u) case-insensitively",
            unit_tests=[
                "assert count_vowels('hello') == 2",
                "assert count_vowels('PYTHON') == 1",
                "assert count_vowels('xyz') == 0",
            ],
        )

        variant_policy = VariantPolicy(paraphrase=True, numeric_jitter=NumericJitterPolicy(enabled=True, pct=15))

        start_time = time.time()

        result = await variant_maker.create_variants(
            base_problem=base_problem, variant_policy=variant_policy, n_variants=2
        )

        duration = time.time() - start_time

        # Validate response
        assert result.ok is True
        assert len(result.variants) > 0

        # Check variant quality
        for variant in result.variants:
            assert variant.id != base_problem.id
            assert variant.statement != base_problem.statement
            assert len(variant.statement) > 20
            assert variant.canonical_answer
            assert variant.rubric

            # Should be similar but different
            assert "vowel" in variant.statement.lower() or "count" in variant.statement.lower()

        logger.info(f"Generated {len(result.variants)} variants in {duration:.1f}s")

    @requires_api_key
    @pytest.mark.asyncio
    async def test_live_grading_system(self, live_llm):
        """Test live grading with real API."""
        grader = Grader(live_llm, model=TEST_MODELS["smart"])

        test_problem = Problem(
            id="grading_test_001",
            topic="algorithms",
            difficulty=0.6,
            statement="Write a function that returns the maximum value in a list of integers.",
            canonical_answer="def find_max(lst): return max(lst) if lst else None",
            rubric="Function finds maximum correctly and handles empty list",
            unit_tests=[
                "assert find_max([1, 3, 2]) == 3",
                "assert find_max([5]) == 5",
                "assert find_max([]) is None",
            ],
        )

        test_cases = [
            ("def find_max(lst): return max(lst) if lst else None", True),  # Correct
            ("def find_max(lst): return max(lst)", False),  # Missing empty case
            ("def find_max(lst): return min(lst)", False),  # Wrong function
            ("print('hello')", False),  # Completely wrong
        ]

        for model_answer, expected_correct in test_cases:
            start_time = time.time()

            result = await grader.grade_solution(
                problem=test_problem,
                model_answer=model_answer,
                use_code_execution=False,  # Safer for testing
            )

            duration = time.time() - start_time

            # Validate response
            assert result.ok is True
            assert isinstance(result.correct, bool)
            assert isinstance(result.error_tags, list)

            # Check reasonable grading time
            assert duration < 30, f"Grading took {duration:.1f}s, should be under 30s"

            logger.info(f"Grading {'CORRECT' if result.correct else 'INCORRECT'} in {duration:.1f}s")

    @requires_api_key
    @pytest.mark.asyncio
    async def test_live_hint_generation(self, live_llm):
        """Test live hint generation with real API."""
        hint_gen = HintGenerator(live_llm, model=TEST_MODELS["fast"])

        problem = Problem(
            id="hint_test_001",
            topic="string_manipulation",
            difficulty=0.6,
            statement="Write a function that reverses a string without using built-in reverse methods.",
            canonical_answer="def reverse_string(s): return s[::-1]",
            rubric="Function reverses string correctly without using reverse() method",
            unit_tests=["assert reverse_string('hello') == 'olleh'"],
        )

        wrong_answers = [
            "def reverse_string(s): return s.reverse()",  # Using forbidden method
            "def reverse_string(s): return s",  # No reversal
            "def reverse_string(s): print(s[::-1])",  # Print instead of return
        ]

        for wrong_answer in wrong_answers:
            start_time = time.time()

            result = await hint_gen.generate_hint(problem=problem, wrong_answer=wrong_answer)

            duration = time.time() - start_time

            # Validate response
            assert result.ok is True
            assert result.hint
            assert result.hint_type

            # Check token constraint
            word_count = len(result.hint.split())
            assert word_count <= 25, f"Hint has {word_count} words, should be ≤25"

            # Should be helpful
            assert len(result.hint.strip()) > 5

            logger.info(f"Generated hint: '{result.hint}' ({word_count} words) in {duration:.1f}s")

    @requires_api_key
    @pytest.mark.asyncio
    async def test_live_mastery_tracking(self, live_llm):
        """Test live mastery tracking with real API."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            tracker = MasteryTracker(live_llm, model=TEST_MODELS["fast"], storage_path=temp_db)

            # Simulate learning progression
            student_id = "live_test_student"
            problem_id = "mastery_test_001"

            attempts = [
                ("variant_1", False),  # First attempt fails
                ("variant_1", True),  # Second attempt succeeds
                ("variant_2", True),  # Different variant succeeds
                ("variant_3", True),  # Third variant succeeds (should achieve mastery)
            ]

            for variant_id, correct in attempts:
                start_time = time.time()

                result = await tracker.evaluate_mastery(
                    student_id=student_id,
                    problem_id=problem_id,
                    variant_id=variant_id,
                    correct=correct,
                    use_llm_policy=True,  # Test real LLM policy
                )

                duration = time.time() - start_time

                # Validate response
                assert result.ok is True
                assert result.status
                assert result.next_action

                logger.info(
                    f"Mastery evaluation: {result.status.value} -> {result.next_action.value} in {duration:.1f}s"
                )

            # Check final summary
            summary = tracker.get_student_mastery_summary(student_id)
            assert summary["student_id"] == student_id
            assert summary["total_problems"] == 1

        finally:
            os.unlink(temp_db)


class TestEndToEndPipelineLive:
    """Test complete end-to-end pipeline with live API."""

    @requires_api_key
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, sample_telemetry_realistic):
        """Test complete pipeline with live API integration."""
        if not API_KEY:
            pytest.skip("OPENROUTER_API_KEY not available")

        logger.info("Starting full pipeline integration test...")

        start_time = time.time()

        # Run complete pipeline with smaller parameters for testing
        result = await run_full_curriculum_pipeline(
            api_key=API_KEY,
            domain="coding-python",
            initial_telemetry=sample_telemetry_realistic,
            num_cycles=2,  # Reduced for testing
            model=TEST_MODELS["smart"],
        )

        duration = time.time() - start_time

        # Validate pipeline execution
        assert result["pipeline_success"] is True
        assert result["initialization"]["success"] is True
        assert result["cycles"]["total_cycles"] == 2

        # Check initialization results
        init = result["initialization"]
        assert "initial_edge" in init
        assert init["initial_problems"] > 0
        assert len(init["topic_mix"]) > 0

        # Check cycle execution
        cycles = result["cycles"]
        assert cycles["successful_cycles"] >= 1
        assert cycles["total_operations_completed"] >= 0

        # Check final status
        status = result["final_status"]
        assert "current_edge" in status
        assert "queues" in status
        assert "system_health" in status

        logger.info(f"Full pipeline completed in {duration:.1f}s")
        logger.info(f"Initialization: {init['initial_problems']} problems, {len(init['topic_mix'])} topics")
        logger.info(f"Cycles: {cycles['successful_cycles']}/{cycles['total_cycles']} successful")
        logger.info(f"Final queues: {status['queues']}")
        logger.info(f"System health: {status['system_health']}")

    @requires_api_key
    @pytest.mark.asyncio
    async def test_orchestrator_stress_test(self, sample_telemetry_realistic):
        """Test orchestrator under moderate stress."""
        if not API_KEY:
            pytest.skip("OPENROUTER_API_KEY not available")

        async with OpenRouterLLM(api_key=API_KEY, model=TEST_MODELS["fast"]) as client:
            orchestrator = CurriculumOrchestrator(client)

            # Initialize with larger telemetry set
            init_result = await orchestrator.initialize_curriculum(
                domain="coding-python",
                initial_telemetry=sample_telemetry_realistic,
                constraints=EdgeConstraints(target_low=0.55, target_high=0.75, problem_budget=50),
            )

            assert init_result["success"] is True

            # Run multiple cycles with increasing load
            for cycle_num in range(3):
                capacity = 5 + cycle_num * 2  # 5, 7, 9

                start_time = time.time()

                cycle_result = await orchestrator.run_curriculum_cycle(
                    domain="coding-python", num_cycles=1, cycle_capacity=capacity
                )

                duration = time.time() - start_time

                assert cycle_result["successful_cycles"] >= 0
                logger.info(f"Cycle {cycle_num + 1} with capacity {capacity} completed in {duration:.1f}s")

            # Check final system status
            final_status = await orchestrator.get_curriculum_status()
            assert final_status["system_health"] != "critical - no problems queued"


class TestAPIContractCompliance:
    """Test API contract compliance and error handling."""

    @requires_api_key
    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self):
        """Test rate limiting compliance with OpenRouter."""
        if not API_KEY:
            pytest.skip("OPENROUTER_API_KEY not available")

        async with OpenRouterLLM(api_key=API_KEY, rpm_limit=10) as client:  # Very low limit
            start_time = time.time()

            # Make multiple rapid requests
            tasks = []
            for i in range(5):
                task = client.invoke(f"Say 'test {i}'", max_tokens=10)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time

            # Should have taken time due to rate limiting
            assert duration >= 4, f"5 requests with 10 RPM limit should take ≥4s, took {duration:.1f}s"

            # All requests should eventually succeed (no permanent failures)
            successful = sum(1 for r in results if isinstance(r, str))
            assert successful >= 3, f"Expected ≥3 successful requests, got {successful}"

    @requires_api_key
    @pytest.mark.asyncio
    async def test_cost_tracking_accuracy(self):
        """Test cost tracking accuracy."""
        if not API_KEY:
            pytest.skip("OPENROUTER_API_KEY not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            async with OpenRouterLLM(api_key=API_KEY, cache_dir=temp_dir) as client:
                # Make a few requests
                await client.invoke("Count to 3", max_tokens=20)
                await client.invoke("Say hello", max_tokens=10)

                # Check cost tracking
                stats = client.get_cache_stats()

                assert "total_entries" in stats
                assert stats["total_entries"] >= 0

                if stats["total_entries"] > 0:
                    assert "by_model" in stats
                    assert len(stats["by_model"]) > 0

    @requires_api_key
    @pytest.mark.asyncio
    async def test_caching_effectiveness(self):
        """Test caching effectiveness."""
        if not API_KEY:
            pytest.skip("OPENROUTER_API_KEY not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            async with OpenRouterLLM(api_key=API_KEY, cache_dir=temp_dir) as client:
                prompt = "What is 2+2?"

                # First request (should miss cache)
                start_time = time.time()
                result1 = await client.invoke(prompt, max_tokens=10)
                first_duration = time.time() - start_time

                # Second request (should hit cache)
                start_time = time.time()
                result2 = await client.invoke(prompt, max_tokens=10)
                second_duration = time.time() - start_time

                # Results should be identical
                assert result1 == result2

                # Second request should be much faster (cached)
                assert (
                    second_duration < first_duration * 0.5
                ), f"Cached request ({second_duration:.1f}s) should be much faster than first ({first_duration:.1f}s)"


if __name__ == "__main__":
    # Run tests with proper async support
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
