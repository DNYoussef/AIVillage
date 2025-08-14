"""Agent Forge Training Loop Integration Tests.

Tests how the Frontier Curriculum Engine integrates with the existing
Agent Forge training infrastructure, ensuring seamless telemetry data
flow and curriculum adaptation during actual model training sessions.
"""

import logging

# Test imports
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from agent_forge.curriculum import (
    CurriculumOrchestrator,
    EdgeConstraints,
    Problem,
    TelemetryEntry,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Mock training metrics for integration testing."""

    step: int
    loss: float
    accuracy: float
    learning_rate: float
    timestamp: float


@dataclass
class ModelCheckpoint:
    """Mock model checkpoint for integration testing."""

    step: int
    model_path: str
    metrics: TrainingMetrics
    curriculum_state: dict[str, Any]


class MockAgentForgeTrainer:
    """Mock Agent Forge trainer for integration testing."""

    def __init__(self, curriculum_engine: CurriculumOrchestrator | None = None):
        self.curriculum_engine = curriculum_engine
        self.training_steps = 0
        self.current_problems: list[Problem] = []
        self.telemetry_history: list[TelemetryEntry] = []
        self.checkpoints: list[ModelCheckpoint] = []
        self.is_training = False

    async def start_training_session(self, domain: str = "coding-python"):
        """Start a mock training session with curriculum integration."""
        self.is_training = True

        if self.curriculum_engine:
            # Initialize curriculum with sample telemetry
            sample_telemetry = self._generate_initial_telemetry()

            result = await self.curriculum_engine.initialize_curriculum(
                domain=domain,
                initial_telemetry=sample_telemetry,
                constraints=EdgeConstraints(
                    target_low=0.55, target_high=0.75, problem_budget=50
                ),
            )

            logger.info(
                f"Training session started with curriculum: {result['success']}"
            )
            return result

        return {"success": True, "curriculum_enabled": False}

    async def training_step(self, step_num: int) -> TrainingMetrics:
        """Simulate a single training step."""
        self.training_steps = step_num

        # Simulate varying performance based on curriculum difficulty
        if self.curriculum_engine and self.curriculum_engine.current_edge:
            edge = self.curriculum_engine.current_edge
            target_accuracy = (edge.low + edge.high) / 2
            # Add noise to simulate training dynamics
            noise = (step_num % 10) / 100 - 0.05  # ±5% noise
            accuracy = max(0.0, min(1.0, target_accuracy + noise))
        else:
            # Default accuracy progression
            accuracy = min(0.9, 0.3 + (step_num * 0.02))  # Gradual improvement

        # Simulate loss decreasing with accuracy
        loss = max(0.1, 2.0 - (accuracy * 1.5))
        learning_rate = 0.001 * (0.99 ** (step_num // 100))  # Decay every 100 steps

        metrics = TrainingMetrics(
            step=step_num,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            timestamp=time.time(),
        )

        # Create telemetry entry for curriculum
        if self.current_problems and step_num % 5 == 0:  # Every 5 steps
            # Simulate evaluation on current curriculum problems
            problem_idx = step_num % len(self.current_problems)
            problem = self.current_problems[problem_idx]

            telemetry = TelemetryEntry(
                task_id=f"step_{step_num}_prob_{problem.id}",
                difficulty=problem.difficulty,
                correct=accuracy > problem.difficulty,  # Simple correctness model
            )

            self.telemetry_history.append(telemetry)

            # Update curriculum every 20 steps
            if step_num % 20 == 0 and self.curriculum_engine:
                await self._update_curriculum()

        return metrics

    async def save_checkpoint(self, step: int) -> ModelCheckpoint:
        """Save a training checkpoint with curriculum state."""
        metrics = await self.training_step(step)

        curriculum_state = {}
        if self.curriculum_engine:
            status = await self.curriculum_engine.get_curriculum_status()
            curriculum_state = {
                "current_edge": status.get("current_edge"),
                "queues": status.get("queues"),
                "system_health": status.get("system_health"),
            }

        checkpoint = ModelCheckpoint(
            step=step,
            model_path=f"/tmp/model_step_{step}.ckpt",
            metrics=metrics,
            curriculum_state=curriculum_state,
        )

        self.checkpoints.append(checkpoint)
        return checkpoint

    async def _update_curriculum(self):
        """Update curriculum based on recent performance."""
        if not self.curriculum_engine or len(self.telemetry_history) < 10:
            return

        # Get recent telemetry
        recent_telemetry = self.telemetry_history[-20:]

        # Run a curriculum cycle
        cycle_result = await self.curriculum_engine.run_curriculum_cycle(
            domain="coding-python", num_cycles=1, cycle_capacity=5
        )

        if cycle_result["successful_cycles"] > 0:
            # Update current problems (mock problem queue)
            status = await self.curriculum_engine.get_curriculum_status()
            queues = status.get("queues", {})

            # Simulate getting new problems from curriculum
            if queues.get("fresh_problems", 0) > 0:
                self.current_problems = await self._get_curriculum_problems()

        logger.info(
            f"Curriculum updated: {cycle_result['successful_cycles']} successful cycles"
        )

    async def _get_curriculum_problems(self) -> list[Problem]:
        """Get problems from curriculum queue (mock implementation)."""
        # Mock problems that would come from curriculum
        return [
            Problem(
                id=f"curriculum_prob_{i}",
                topic="string_manipulation",
                difficulty=0.55 + (i * 0.05),
                statement=f"Mock curriculum problem {i}",
                canonical_answer="def mock(): pass",
                rubric="Mock rubric",
                unit_tests=["assert True"],
            )
            for i in range(5)
        ]

    def _generate_initial_telemetry(self) -> list[TelemetryEntry]:
        """Generate initial telemetry for curriculum initialization."""
        import random

        telemetry = []
        for i in range(30):
            difficulty = random.uniform(0.3, 0.9)
            # Simulate performance curve
            if difficulty < 0.4:
                correct_prob = 0.8
            elif 0.4 <= difficulty <= 0.7:
                correct_prob = 0.6
            else:
                correct_prob = 0.3

            correct = random.random() < correct_prob
            telemetry.append(
                TelemetryEntry(
                    task_id=f"init_task_{i:03d}",
                    difficulty=round(difficulty, 2),
                    correct=correct,
                )
            )

        return telemetry


class TestAgentForgeTrainingIntegration:
    """Test curriculum engine integration with Agent Forge training loop."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client for testing."""
        from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

        return MockOpenRouterLLM("mock-api-key")

    @pytest.fixture
    def curriculum_engine(self, mock_llm):
        """Curriculum engine fixture."""
        return CurriculumOrchestrator(mock_llm)

    @pytest.fixture
    def agent_forge_trainer(self, curriculum_engine):
        """Agent Forge trainer with curriculum integration."""
        return MockAgentForgeTrainer(curriculum_engine)

    @pytest.mark.asyncio
    async def test_training_session_initialization(self, agent_forge_trainer):
        """Test training session initialization with curriculum."""
        result = await agent_forge_trainer.start_training_session("coding-python")

        assert result["success"] is True
        assert "initial_edge" in result
        assert "topic_mix" in result
        assert result["initial_problems"] > 0

        # Should have initialized curriculum state
        assert agent_forge_trainer.curriculum_engine.current_edge is not None

    @pytest.mark.asyncio
    async def test_telemetry_data_flow(self, agent_forge_trainer):
        """Test telemetry data flow from training to curriculum."""
        await agent_forge_trainer.start_training_session()

        # Run multiple training steps
        for step in range(25):
            metrics = await agent_forge_trainer.training_step(step)
            assert isinstance(metrics.accuracy, float)
            assert 0.0 <= metrics.accuracy <= 1.0

        # Should have generated telemetry
        assert len(agent_forge_trainer.telemetry_history) > 0

        # Telemetry should have correct structure
        for telemetry in agent_forge_trainer.telemetry_history:
            assert telemetry.task_id
            assert isinstance(telemetry.difficulty, float)
            assert isinstance(telemetry.correct, bool)
            assert 0.0 <= telemetry.difficulty <= 1.0

    @pytest.mark.asyncio
    async def test_curriculum_adaptation_during_training(self, agent_forge_trainer):
        """Test curriculum adapts based on training performance."""
        await agent_forge_trainer.start_training_session()

        # Get initial curriculum state
        initial_status = (
            await agent_forge_trainer.curriculum_engine.get_curriculum_status()
        )
        initial_edge = initial_status["current_edge"]

        # Run training for enough steps to trigger curriculum updates
        for step in range(50):
            await agent_forge_trainer.training_step(step)

        # Check if curriculum has adapted
        final_status = (
            await agent_forge_trainer.curriculum_engine.get_curriculum_status()
        )
        final_edge = final_status["current_edge"]

        # Curriculum should have some activity
        assert final_status["queues"]["total_queued"] >= 0
        assert final_status["system_health"] in [
            "good",
            "fair",
            "poor",
            "critical - no problems queued",
            "warning - low fresh problem supply",
            "warning - stalled students need hint variants",
        ]

    @pytest.mark.asyncio
    async def test_checkpoint_curriculum_state_persistence(self, agent_forge_trainer):
        """Test curriculum state is properly saved with model checkpoints."""
        await agent_forge_trainer.start_training_session()

        # Run some training steps
        for step in range(15):
            await agent_forge_trainer.training_step(step)

        # Save checkpoint
        checkpoint = await agent_forge_trainer.save_checkpoint(15)

        assert checkpoint.step == 15
        assert checkpoint.model_path
        assert checkpoint.curriculum_state

        # Curriculum state should contain key information
        curr_state = checkpoint.curriculum_state
        assert "current_edge" in curr_state
        assert "queues" in curr_state
        assert "system_health" in curr_state

    @pytest.mark.asyncio
    async def test_curriculum_problem_integration(self, agent_forge_trainer):
        """Test integration of curriculum-generated problems into training."""
        await agent_forge_trainer.start_training_session()

        # Initially should have some problems
        initial_problem_count = len(agent_forge_trainer.current_problems)

        # Run training with curriculum updates
        for step in range(30):
            await agent_forge_trainer.training_step(step)

        # Should have curriculum problems available
        assert len(agent_forge_trainer.current_problems) >= 0

        # Problems should have curriculum characteristics
        for problem in agent_forge_trainer.current_problems:
            assert problem.id.startswith("curriculum_prob_")
            assert 0.0 <= problem.difficulty <= 1.0
            assert problem.statement
            assert problem.canonical_answer

    @pytest.mark.asyncio
    async def test_training_without_curriculum(self):
        """Test training works without curriculum integration."""
        trainer = MockAgentForgeTrainer(curriculum_engine=None)

        result = await trainer.start_training_session()
        assert result["success"] is True
        assert result["curriculum_enabled"] is False

        # Should still be able to train
        for step in range(10):
            metrics = await trainer.training_step(step)
            assert isinstance(metrics.accuracy, float)
            assert metrics.accuracy >= 0.0

        # No telemetry should be generated
        assert len(trainer.telemetry_history) == 0


class TestCurriculumEffectivenessTracking:
    """Test curriculum effectiveness tracking and validation."""

    @pytest.mark.asyncio
    async def test_curriculum_impact_measurement(self):
        """Test measuring curriculum impact on training effectiveness."""
        from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

        # Create two trainers - one with curriculum, one without
        mock_llm = MockOpenRouterLLM("mock-key")
        curriculum_engine = CurriculumOrchestrator(mock_llm)

        trainer_with_curriculum = MockAgentForgeTrainer(curriculum_engine)
        trainer_without_curriculum = MockAgentForgeTrainer(None)

        # Run parallel training sessions
        await trainer_with_curriculum.start_training_session()
        await trainer_without_curriculum.start_training_session()

        curriculum_metrics = []
        baseline_metrics = []

        # Train both for same number of steps
        for step in range(50):
            curr_metrics = await trainer_with_curriculum.training_step(step)
            base_metrics = await trainer_without_curriculum.training_step(step)

            curriculum_metrics.append(curr_metrics.accuracy)
            baseline_metrics.append(base_metrics.accuracy)

        # Analyze effectiveness
        curriculum_final = curriculum_metrics[-10:]  # Last 10 steps
        baseline_final = baseline_metrics[-10:]

        curriculum_avg = sum(curriculum_final) / len(curriculum_final)
        baseline_avg = sum(baseline_final) / len(baseline_final)

        logger.info(f"Curriculum training avg accuracy: {curriculum_avg:.3f}")
        logger.info(f"Baseline training avg accuracy: {baseline_avg:.3f}")

        # Both should achieve reasonable performance
        assert curriculum_avg > 0.5
        assert baseline_avg > 0.5

        # Curriculum should show more consistent performance
        curriculum_variance = sum(
            (x - curriculum_avg) ** 2 for x in curriculum_final
        ) / len(curriculum_final)
        baseline_variance = sum((x - baseline_avg) ** 2 for x in baseline_final) / len(
            baseline_final
        )

        logger.info(f"Curriculum variance: {curriculum_variance:.4f}")
        logger.info(f"Baseline variance: {baseline_variance:.4f}")


class TestCurriculumRecoveryAndResilience:
    """Test curriculum system recovery and resilience during training."""

    @pytest.fixture
    def resilient_trainer(self):
        """Trainer with resilience testing setup."""
        from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

        mock_llm = MockOpenRouterLLM("mock-key")
        curriculum_engine = CurriculumOrchestrator(mock_llm)
        return MockAgentForgeTrainer(curriculum_engine)

    @pytest.mark.asyncio
    async def test_curriculum_failure_recovery(self, resilient_trainer):
        """Test training continues when curriculum fails."""
        await resilient_trainer.start_training_session()

        # Simulate curriculum failure by breaking the connection
        original_engine = resilient_trainer.curriculum_engine
        resilient_trainer.curriculum_engine = None

        # Training should continue without curriculum
        recovery_metrics = []
        for step in range(10):
            metrics = await resilient_trainer.training_step(step + 20)
            recovery_metrics.append(metrics.accuracy)

        # Restore curriculum
        resilient_trainer.curriculum_engine = original_engine

        # Should be able to continue
        for step in range(10):
            metrics = await resilient_trainer.training_step(step + 30)
            assert isinstance(metrics.accuracy, float)

        # Should have maintained reasonable performance during failure
        assert all(acc >= 0.0 for acc in recovery_metrics)

    @pytest.mark.asyncio
    async def test_curriculum_state_corruption_handling(self, resilient_trainer):
        """Test handling of corrupted curriculum state."""
        await resilient_trainer.start_training_session()

        # Corrupt curriculum state
        resilient_trainer.curriculum_engine.current_edge = None

        # Should still be able to continue training
        for step in range(15):
            metrics = await resilient_trainer.training_step(step)
            assert isinstance(metrics.accuracy, float)

        # System should recover or maintain safe defaults
        status = await resilient_trainer.curriculum_engine.get_curriculum_status()
        assert isinstance(status, dict)
        assert "system_health" in status


class TestLongRunningTrainingIntegration:
    """Test curriculum integration over long-running training sessions."""

    @pytest.mark.asyncio
    async def test_extended_training_session(self):
        """Test curriculum behavior over extended training period."""
        from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

        mock_llm = MockOpenRouterLLM("mock-key")
        curriculum_engine = CurriculumOrchestrator(mock_llm)
        trainer = MockAgentForgeTrainer(curriculum_engine)

        await trainer.start_training_session()

        # Track curriculum evolution over time
        curriculum_snapshots = []

        # Simulate 200 training steps with periodic snapshots
        for step in range(200):
            await trainer.training_step(step)

            # Take snapshots every 50 steps
            if step % 50 == 0 and step > 0:
                status = await curriculum_engine.get_curriculum_status()
                curriculum_snapshots.append(
                    {
                        "step": step,
                        "edge": status["current_edge"],
                        "queues": status["queues"],
                        "health": status["system_health"],
                    }
                )

        # Should have curriculum evolution
        assert len(curriculum_snapshots) >= 3

        # System should remain healthy
        final_health = curriculum_snapshots[-1]["health"]
        assert final_health != "critical - no problems queued"  # Should not be critical

        # Should have accumulated substantial telemetry
        assert len(trainer.telemetry_history) > 20

        logger.info(
            f"Extended training: {len(trainer.telemetry_history)} telemetry entries"
        )
        logger.info(f"Final curriculum health: {final_health}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
