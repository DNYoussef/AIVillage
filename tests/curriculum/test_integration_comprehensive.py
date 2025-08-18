"""Comprehensive integration tests for Frontier Curriculum Engine.

Tests all components working together in realistic scenarios with
proper error handling, performance validation, and data flow verification.
"""

import json
import logging
import os

# Test imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_forge.curriculum import (
    CurriculumOrchestrator,
    EdgeConstraints,
    EdgeController,
    EdgeFinder,
    EdgeWindow,
    Grader,
    HintGenerator,
    MasteryStatus,
    MasteryTracker,
    NumericJitterPolicy,
    Problem,
    ProblemGenerator,
    TelemetryEntry,
    VariantMaker,
    VariantPolicy,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockOpenRouterLLM:
    """Mock OpenRouter client for testing without API calls."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.call_count = 0
        self.responses = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def render_template(self, template: str, **kwargs) -> str:
        """Mock template rendering."""
        return f"Rendered template with {len(kwargs)} parameters"

    async def invoke(self, prompt: str, **kwargs) -> str:
        """Mock LLM invocation."""
        self.call_count += 1

        # Return different responses based on prompt content
        if "edge" in prompt.lower():
            return self._mock_edge_response()
        elif "problem" in prompt.lower():
            return self._mock_problem_response()
        elif "variant" in prompt.lower():
            return self._mock_variant_response()
        elif "grade" in prompt.lower():
            return self._mock_grading_response()
        elif "hint" in prompt.lower():
            return self._mock_hint_response()
        elif "mastery" in prompt.lower():
            return self._mock_mastery_response()
        elif "control" in prompt.lower():
            return self._mock_controller_response()
        elif "conduct" in prompt.lower():
            return self._mock_conductor_response()
        else:
            return '{"ok": true, "msg": "mock response"}'

    async def invoke_with_schema(self, prompt: str, schema_class: type, **kwargs):
        """Mock schema-validated invocation."""
        response_text = await self.invoke(prompt, **kwargs)

        # Parse and return as schema object
        try:
            import json

            data = json.loads(response_text)
            return schema_class(**data)
        except Exception as e:
            logger.warning(f"Mock schema parsing failed: {e}")
            # Return a basic mock object
            return MagicMock()

    def _mock_edge_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "edge selected",
                "edge": {"low": 0.55, "high": 0.75},
                "topic_mix": [
                    {"topic": "string_manipulation", "weight": 0.4},
                    {"topic": "list_operations", "weight": 0.6},
                ],
                "distribution": [
                    {"difficulty": 0.6, "count": 500},
                    {"difficulty": 0.7, "count": 500},
                ],
                "generation_plan": {
                    "n_total": 1000,
                    "per_topic_min": 50,
                    "variant_rate": 0.3,
                },
            }
        )

    def _mock_problem_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "generated",
                "problems": [
                    {
                        "id": "mock_problem_001",
                        "topic": "string_manipulation",
                        "difficulty": 0.6,
                        "statement": "Write a function to reverse a string",
                        "canonical_answer": "def reverse_string(s): return s[::-1]",
                        "rubric": "Function correctly reverses string",
                        "unit_tests": ["assert reverse_string('hello') == 'olleh'"],
                    }
                ],
            }
        )

    def _mock_variant_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "variants",
                "variants": [
                    {
                        "id": "mock_variant_001",
                        "statement": "Create a function to invert a text string",
                        "canonical_answer": "def invert_string(text): return text[::-1]",
                        "rubric": "Function correctly inverts string",
                        "unit_tests": ["assert invert_string('world') == 'dlrow'"],
                    }
                ],
            }
        )

    def _mock_grading_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "graded",
                "correct": True,
                "error_tags": [],
                "normalizer_notes": "Solution is correct",
            }
        )

    def _mock_hint_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "hint",
                "hint": "Consider string slicing with [::-1]",
                "hint_type": "procedure",
            }
        )

    def _mock_mastery_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "updated",
                "status": "learning",
                "next_action": "reshuffle",
                "needs_hint": False,
            }
        )

    def _mock_controller_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "nudged",
                "new_edge": {"low": 0.52, "high": 0.72},
                "delta": {"low": -0.03, "high": -0.03},
            }
        )

    def _mock_conductor_response(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "msg": "batch plan",
                "queue": [
                    {
                        "op": "generate",
                        "n": 10,
                        "params": {"edge_low": 0.55, "edge_high": 0.75},
                    }
                ],
            }
        )


@pytest.fixture
def mock_llm():
    """Fixture providing mock OpenRouter client."""
    return MockOpenRouterLLM("mock-api-key")


@pytest.fixture
def sample_telemetry():
    """Fixture providing sample telemetry data."""
    return [
        TelemetryEntry(task_id="task_001", difficulty=0.4, correct=True),
        TelemetryEntry(task_id="task_002", difficulty=0.6, correct=True),
        TelemetryEntry(task_id="task_003", difficulty=0.65, correct=False),
        TelemetryEntry(task_id="task_004", difficulty=0.7, correct=True),
        TelemetryEntry(task_id="task_005", difficulty=0.8, correct=False),
    ]


@pytest.fixture
def sample_problem():
    """Fixture providing sample problem."""
    return Problem(
        id="test_problem_001",
        topic="string_manipulation",
        difficulty=0.6,
        statement="Write a function that removes vowels from a string",
        canonical_answer="def remove_vowels(s): return ''.join(c for c in s if c not in 'aeiou')",
        rubric="Function removes all vowels correctly",
        unit_tests=[
            "assert remove_vowels('hello') == 'hll'",
            "assert remove_vowels('world') == 'wrld'",
        ],
    )


class TestComponentIntegration:
    """Test integration between individual components."""

    @pytest.mark.asyncio
    async def test_edge_finder_integration(self, mock_llm, sample_telemetry):
        """Test EdgeFinder component integration."""
        edge_finder = EdgeFinder(mock_llm)

        result = await edge_finder.find_edge(
            domain="coding-python",
            telemetry=sample_telemetry,
            constraints=EdgeConstraints(target_low=0.55, target_high=0.75),
        )

        assert result.ok is True
        assert 0.0 <= result.edge.low <= result.edge.high <= 1.0
        assert len(result.topic_mix) > 0
        assert result.generation_plan.n_total > 0
        assert mock_llm.call_count > 0

    @pytest.mark.asyncio
    async def test_problem_generator_integration(self, mock_llm):
        """Test ProblemGenerator component integration."""
        from agent_forge.curriculum import TopicMix

        problem_gen = ProblemGenerator(mock_llm)
        edge = EdgeWindow(low=0.55, high=0.75)
        topic_mix = [TopicMix(topic="string_ops", weight=1.0)]

        result = await problem_gen.generate_problems(domain="coding-python", edge=edge, topic_mix=topic_mix, n=3)

        assert result.ok is True
        assert len(result.problems) > 0

        for problem in result.problems:
            assert problem.id
            assert problem.statement
            assert problem.canonical_answer
            assert 0.0 <= problem.difficulty <= 1.0

    @pytest.mark.asyncio
    async def test_variant_maker_integration(self, mock_llm, sample_problem):
        """Test VariantMaker component integration."""
        variant_maker = VariantMaker(mock_llm)

        variant_policy = VariantPolicy(paraphrase=True, numeric_jitter=NumericJitterPolicy(enabled=True, pct=10))

        result = await variant_maker.create_variants(
            base_problem=sample_problem, variant_policy=variant_policy, n_variants=2
        )

        assert result.ok is True
        assert len(result.variants) > 0

        for variant in result.variants:
            assert variant.id != sample_problem.id
            assert variant.statement
            assert variant.canonical_answer

    @pytest.mark.asyncio
    async def test_grader_integration(self, mock_llm, sample_problem):
        """Test Grader component integration."""
        grader = Grader(mock_llm, enable_code_execution=False)

        model_answer = "def remove_vowels(s): return ''.join(c for c in s if c not in 'aeiou')"

        result = await grader.grade_solution(problem=sample_problem, model_answer=model_answer)

        assert result.ok is True
        assert isinstance(result.correct, bool)
        assert isinstance(result.error_tags, list)

    @pytest.mark.asyncio
    async def test_hint_generator_integration(self, mock_llm, sample_problem):
        """Test HintGenerator component integration."""
        hint_gen = HintGenerator(mock_llm)

        wrong_answer = "def remove_vowels(s): return s.replace('a', '')"

        result = await hint_gen.generate_hint(problem=sample_problem, wrong_answer=wrong_answer)

        assert result.ok is True
        assert result.hint
        assert len(result.hint.split()) <= 25  # Token limit
        assert result.hint_type

    @pytest.mark.asyncio
    async def test_mastery_tracker_integration(self, mock_llm):
        """Test MasteryTracker component integration."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            tracker = MasteryTracker(mock_llm, storage_path=temp_db)

            # Test mastery evaluation
            result = await tracker.evaluate_mastery(
                student_id="test_student",
                problem_id="test_problem",
                variant_id="variant_1",
                correct=True,
                use_llm_policy=False,  # Use local evaluation for testing
            )

            assert result.ok is True
            assert result.status in [
                MasteryStatus.LEARNING,
                MasteryStatus.UNDERSTOOD,
                MasteryStatus.STALLED,
            ]
            assert result.next_action

            # Test student summary
            summary = tracker.get_student_mastery_summary("test_student")
            assert summary["student_id"] == "test_student"
            assert "total_problems" in summary

        finally:
            os.unlink(temp_db)

    @pytest.mark.asyncio
    async def test_edge_controller_integration(self, mock_llm):
        """Test EdgeController component integration."""
        controller = EdgeController(mock_llm)

        current_edge = EdgeWindow(low=0.55, high=0.75)
        constraints = EdgeConstraints(target_low=0.55, target_high=0.75)

        result = await controller.nudge_edge(
            window_accuracy=0.45,  # Too hard
            current_edge=current_edge,
            constraints=constraints,
            use_llm_control=False,  # Use local control for testing
        )

        assert result.ok is True
        assert result.new_edge
        assert result.delta

        # Should nudge to make easier when accuracy too low
        assert result.new_edge.low <= current_edge.low or result.new_edge.high <= current_edge.high


class TestEndToEndPipeline:
    """Test complete end-to-end curriculum pipeline."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_llm, sample_telemetry):
        """Test CurriculumOrchestrator initialization."""
        orchestrator = CurriculumOrchestrator(mock_llm)

        result = await orchestrator.initialize_curriculum(domain="coding-python", initial_telemetry=sample_telemetry)

        assert result["success"] is True
        assert result["domain"] == "coding-python"
        assert "initial_edge" in result
        assert "topic_mix" in result
        assert result["initial_problems"] > 0

    @pytest.mark.asyncio
    async def test_orchestrator_batch_planning(self, mock_llm):
        """Test orchestrator batch planning."""
        orchestrator = CurriculumOrchestrator(mock_llm)
        orchestrator.current_edge = EdgeWindow(low=0.55, high=0.75)

        result = await orchestrator.conduct_batch_planning(
            capacity=20,
            use_llm_conductor=False,  # Use local planning for testing
        )

        assert result.ok is True
        assert isinstance(result.queue, list)

        # Should have planned some operations
        total_items = sum(item.n for item in result.queue)
        assert total_items <= 20  # Respects capacity

    @pytest.mark.asyncio
    async def test_curriculum_cycle_execution(self, mock_llm, sample_telemetry):
        """Test complete curriculum cycle execution."""
        orchestrator = CurriculumOrchestrator(mock_llm)

        # Initialize
        init_result = await orchestrator.initialize_curriculum(
            domain="coding-python", initial_telemetry=sample_telemetry
        )
        assert init_result["success"] is True

        # Run cycles
        cycle_result = await orchestrator.run_curriculum_cycle(domain="coding-python", num_cycles=2, cycle_capacity=10)

        assert cycle_result["total_cycles"] == 2
        assert cycle_result["successful_cycles"] >= 0
        assert "cycle_details" in cycle_result

    @pytest.mark.asyncio
    async def test_curriculum_status_monitoring(self, mock_llm):
        """Test curriculum status monitoring."""
        orchestrator = CurriculumOrchestrator(mock_llm)
        orchestrator.current_edge = EdgeWindow(low=0.55, high=0.75)

        status = await orchestrator.get_curriculum_status()

        assert "current_edge" in status
        assert "queues" in status
        assert "student_distribution" in status
        assert "system_health" in status
        assert "timestamp" in status

        # Should have reasonable values
        assert status["queues"]["total_queued"] >= 0
        assert status["system_health"] in [
            "critical - no problems queued",
            "good",
            "fair",
            "poor",
            "warning - low fresh problem supply",
            "warning - stalled students need hint variants",
        ]


class TestDataFlowValidation:
    """Test data flow and consistency across components."""

    @pytest.mark.asyncio
    async def test_edge_to_problem_generation_flow(self, mock_llm, sample_telemetry):
        """Test data flow from edge detection to problem generation."""
        edge_finder = EdgeFinder(mock_llm)
        problem_gen = ProblemGenerator(mock_llm)

        # Find edge
        edge_result = await edge_finder.find_edge(domain="coding-python", telemetry=sample_telemetry)

        # Use edge result for problem generation
        problem_result = await problem_gen.generate_problems(
            domain="coding-python",
            edge=edge_result.edge,
            topic_mix=edge_result.topic_mix,
            n=5,
        )

        assert problem_result.ok is True
        assert len(problem_result.problems) > 0

        # Verify problems respect edge bounds
        for problem in problem_result.problems:
            # Allow some tolerance for edge bounds
            assert edge_result.edge.low - 0.1 <= problem.difficulty <= edge_result.edge.high + 0.1

    @pytest.mark.asyncio
    async def test_problem_to_variant_flow(self, mock_llm, sample_problem):
        """Test data flow from problem to variant generation."""
        variant_maker = VariantMaker(mock_llm)

        variant_policy = VariantPolicy(paraphrase=True)

        variant_result = await variant_maker.create_variants(
            base_problem=sample_problem, variant_policy=variant_policy, n_variants=2
        )

        assert variant_result.ok is True
        assert len(variant_result.variants) > 0

        # Verify variants maintain base problem characteristics
        for variant in variant_result.variants:
            assert variant.statement != sample_problem.statement  # Should be different
            assert len(variant.statement) > 10  # Should be substantial
            assert variant.canonical_answer  # Should have solution

    @pytest.mark.asyncio
    async def test_grading_to_mastery_flow(self, mock_llm, sample_problem):
        """Test data flow from grading to mastery tracking."""
        grader = Grader(mock_llm, enable_code_execution=False)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            mastery_tracker = MasteryTracker(mock_llm, storage_path=temp_db)

            # Grade a solution
            grade_result = await grader.grade_solution(
                problem=sample_problem,
                model_answer="def remove_vowels(s): return ''.join(c for c in s if c not in 'aeiou')",
            )

            # Use grading result for mastery tracking
            mastery_result = await mastery_tracker.evaluate_mastery(
                student_id="test_student",
                problem_id=sample_problem.id,
                variant_id="variant_1",
                correct=grade_result.correct,
                use_llm_policy=False,
            )

            assert mastery_result.ok is True
            assert mastery_result.status
            assert mastery_result.next_action

        finally:
            os.unlink(temp_db)


class TestErrorHandlingAndRobustness:
    """Test error handling and system robustness."""

    @pytest.mark.asyncio
    async def test_invalid_telemetry_handling(self, mock_llm):
        """Test handling of invalid telemetry data."""
        edge_finder = EdgeFinder(mock_llm)

        # Test empty telemetry
        with pytest.raises(ValueError):
            await edge_finder.find_edge(domain="coding-python", telemetry=[])

        # Test insufficient telemetry
        with pytest.raises(ValueError):
            await edge_finder.find_edge(
                domain="coding-python",
                telemetry=[TelemetryEntry(task_id="1", difficulty=0.5, correct=True)],
            )

    @pytest.mark.asyncio
    async def test_invalid_problem_generation_params(self, mock_llm):
        """Test handling of invalid problem generation parameters."""
        from agent_forge.curriculum import TopicMix

        problem_gen = ProblemGenerator(mock_llm)
        edge = EdgeWindow(low=0.55, high=0.75)
        topic_mix = [TopicMix(topic="test", weight=1.0)]

        # Test negative problem count
        with pytest.raises(ValueError):
            await problem_gen.generate_problems(domain="coding-python", edge=edge, topic_mix=topic_mix, n=-1)

        # Test empty domain
        with pytest.raises(ValueError):
            await problem_gen.generate_problems(domain="", edge=edge, topic_mix=topic_mix, n=5)

    @pytest.mark.asyncio
    async def test_component_fallback_mechanisms(self, mock_llm):
        """Test component fallback mechanisms."""
        # Test local hint generation fallback
        hint_gen = HintGenerator(mock_llm)

        problem = Problem(
            id="test",
            topic="coding",
            difficulty=0.6,
            statement="Test problem",
            canonical_answer="test()",
            rubric="Test rubric",
            unit_tests=[],
        )

        # Should work even with LLM failure by falling back to patterns
        result = await hint_gen.generate_hint(problem=problem, wrong_answer="print('wrong')", use_local_fallback=True)

        assert result.ok is True
        assert result.hint


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, mock_llm):
        """Test batch processing performance."""
        import time

        problem_gen = ProblemGenerator(mock_llm)
        edge = EdgeWindow(low=0.55, high=0.75)

        from agent_forge.curriculum import TopicMix

        topic_mix = [TopicMix(topic="test", weight=1.0)]

        start_time = time.time()

        # Generate problems in batch
        result = await problem_gen.generate_problems(
            domain="coding-python", edge=edge, topic_mix=topic_mix, n=10, batch_size=5
        )

        end_time = time.time()
        duration = end_time - start_time

        assert result.ok is True
        assert len(result.problems) > 0
        assert duration < 30  # Should complete within reasonable time with mocks

        # Should batch requests efficiently
        assert mock_llm.call_count >= 2  # At least 2 batches for 10 problems with batch_size=5

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, mock_llm, sample_telemetry):
        """Test memory usage doesn't grow excessively."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        orchestrator = CurriculumOrchestrator(mock_llm)

        # Run multiple initialization cycles
        for i in range(5):
            await orchestrator.initialize_curriculum(domain=f"test-domain-{i}", initial_telemetry=sample_telemetry)
            gc.collect()  # Force garbage collection

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 50MB for mock operations)
        assert memory_growth < 50 * 1024 * 1024, f"Memory grew by {memory_growth / (1024 * 1024):.1f}MB"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
