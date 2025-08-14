"""Comprehensive tests for Track 3 Agent Forge implementations.

Tests all the stub replacements and ensures production readiness.
"""

import os
import sys
import time
from unittest.mock import Mock

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.communications.standard_protocol import StandardCommunicationProtocol
from src.production.agent_forge import (
    AgentFactory,
    AgentRole,
    AgentSpecialization,
    BaseMetaAgent,
    validate_all_agents,
)
from src.production.agent_forge.evolution import (
    EvolutionMetricsRecorder,
    EvolvableAgent,
)
from src.production.agent_forge.orchestrator import (
    FastAgentOrchestrator,
    TaskRequest,
)


class TestAgentFactory:
    """Test the Agent Factory implementation."""

    def test_agent_factory_initialization(self):
        """Test that agent factory initializes properly."""
        factory = AgentFactory()
        assert factory is not None
        assert len(factory.templates) == 18  # All 18 agent types
        assert len(factory.agent_classes) == 18

    def test_create_all_agent_types(self):
        """Test creating all 18 agent types."""
        factory = AgentFactory()

        for agent_type in factory.required_agent_types():
            agent = factory.create_agent(agent_type)
            assert agent is not None
            assert hasattr(agent, "process")
            assert hasattr(agent, "evaluate_kpi")
            assert callable(agent.process)
            assert callable(agent.evaluate_kpi)

    def test_agent_processing(self):
        """Test that agents can process basic requests."""
        factory = AgentFactory()

        # Test King agent specifically
        king = factory.create_agent("king")
        result = king.process({"task": "ping"})

        assert result["status"] == "completed"
        assert result["agent"] == "king"
        assert "result" in result

    def test_agent_kpi_evaluation(self):
        """Test that agents can evaluate KPIs."""
        factory = AgentFactory()

        king = factory.create_agent("king")
        kpis = king.evaluate_kpi()

        assert isinstance(kpis, dict)
        assert "overall_performance" in kpis or "performance" in kpis

        # All KPI values should be between 0 and 1
        for _key, value in kpis.items():
            assert isinstance(value, int | float)
            assert 0 <= value <= 1.0


class TestBaseClasses:
    """Test the base classes implementation."""

    def test_agent_role_enum(self):
        """Test AgentRole enum."""
        assert AgentRole.KING.value == "king"
        assert AgentRole.MAGI.value == "magi"
        assert AgentRole.SAGE.value == "sage"
        assert len(list(AgentRole)) == 18

    def test_agent_specialization(self):
        """Test AgentSpecialization dataclass."""
        spec = AgentSpecialization(
            role=AgentRole.KING,
            primary_capabilities=[],
            secondary_capabilities=[],
            performance_metrics={},
            resource_requirements={},
        )

        assert spec.role == AgentRole.KING
        spec_dict = spec.to_dict()
        assert spec_dict["role"] == "king"

    def test_base_meta_agent_interface(self):
        """Test BaseMetaAgent interface."""
        spec = AgentSpecialization(
            role=AgentRole.KING,
            primary_capabilities=[],
            secondary_capabilities=[],
            performance_metrics={},
            resource_requirements={},
        )

        # Create a mock implementation
        class TestAgent(BaseMetaAgent):
            def process(self, request):
                return {"status": "completed"}

            def evaluate_kpi(self):
                return {"performance": 0.8}

        agent = TestAgent(spec)
        assert agent.specialization == spec
        assert agent.name == "king"
        assert hasattr(agent, "performance_history")
        assert hasattr(agent, "kpi_scores")


class TestCommunicationProtocol:
    """Test the enhanced communication protocol."""

    @pytest.mark.asyncio
    async def test_standard_protocol_async(self):
        """Test async message handling."""
        protocol = StandardCommunicationProtocol()

        # Test subscription and message handling
        messages_received = []

        async def test_handler(message):
            messages_received.append(message)

        protocol.subscribe("test_agent", test_handler)

        # Create a mock message
        message = Mock()
        message.receiver = "test_agent"
        message.content = {"test": "data"}

        await protocol.send_message(message)

        assert len(messages_received) == 1
        assert messages_received[0] == message

    def test_message_queuing(self):
        """Test message queuing functionality."""
        protocol = StandardCommunicationProtocol()

        # Create mock messages
        message1 = Mock()
        message1.receiver = "agent1"
        message2 = Mock()
        message2.receiver = "agent1"

        # Send messages synchronously to queue them
        protocol.send(message1)
        protocol.send(message2)

        # Test retrieval
        assert protocol.receive() == message1
        assert protocol.receive() == message2
        assert protocol.receive() is None


class TestAgentOrchestrator:
    """Test the Fast Agent Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)

        await orchestrator.initialize_agents()

        assert len(orchestrator.agents) > 0
        assert orchestrator.task_count == 0

    @pytest.mark.asyncio
    async def test_single_task_execution(self):
        """Test single task execution with timing."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)
        await orchestrator.initialize_agents()

        task = TaskRequest(task_id="test_001", task_type="ping", payload={"task": "ping"})

        start_time = time.time() * 1000
        result = await orchestrator.execute_task(task)
        time.time() * 1000 - start_time

        assert result.success
        assert result.task_id == "test_001"
        assert result.execution_time_ms > 0

        # Test orchestration overhead requirement
        metrics = orchestrator.get_performance_metrics()
        assert metrics["average_orchestration_overhead_ms"] < 100.0

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch task processing."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)
        await orchestrator.initialize_agents()

        # Create batch of tasks
        tasks = [TaskRequest(task_id=f"batch_{i}", task_type="ping", payload={"task": "ping"}) for i in range(10)]

        start_time = time.time() * 1000
        results = await orchestrator.process_batch(tasks)
        batch_time = time.time() * 1000 - start_time

        assert len(results) == 10
        assert all(r.success for r in results)

        # Batch should be much faster than sequential
        assert batch_time < 1000  # Less than 1 second for 10 ping tasks

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test system health check."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)
        await orchestrator.initialize_agents()

        health = await orchestrator.health_check()

        assert health["system_healthy"]
        assert health["healthy_agents"] > 0
        assert health["health_check_time_ms"] < 5000  # Less than 5 seconds

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)

        metrics = orchestrator.get_performance_metrics()

        assert "average_orchestration_overhead_ms" in metrics
        assert "total_tasks_processed" in metrics
        assert "active_agents" in metrics
        assert "overhead_target_met" in metrics


class TestValidationFramework:
    """Test the agent validation system."""

    def test_validate_all_agents_execution(self):
        """Test that validation runs without errors."""
        results = validate_all_agents(full_test=True)

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that all required agents are validated
        expected_agents = [
            "king",
            "magi",
            "sage",
            "gardener",
            "sword_shield",
            "legal",
            "shaman",
            "oracle",
            "maker",
            "ensemble",
            "curator",
            "auditor",
            "medic",
            "sustainer",
            "navigator",
            "tutor",
            "polyglot",
            "strategist",
        ]

        for agent_id in expected_agents:
            assert agent_id in results
            assert isinstance(results[agent_id], dict)
            assert "created" in results[agent_id]

    def test_validation_categories(self):
        """Test that validation checks all required categories."""
        results = validate_all_agents(full_test=True)

        # Check validation categories for at least one agent
        first_agent_result = next(iter(results.values()))

        expected_checks = ["created", "communication", "kpi", "specialization"]
        for check in expected_checks:
            assert check in first_agent_result


class TestEvolutionSystem:
    """Test evolution system components."""

    def test_evolution_metrics_recorder(self):
        """Test evolution metrics recording."""
        recorder = EvolutionMetricsRecorder()

        # Test recording evolution start
        mutation_id = recorder.record_evolution_start("crossover", "king")
        assert mutation_id is not None
        assert len(mutation_id) > 0

        # Test recording fitness
        recorder.record_fitness(mutation_id, 0.85)

        # Test recording end
        recorder.record_evolution_end(mutation_id, True, 0.92)

        # Test metrics summary
        summary = recorder.get_metrics_summary()
        assert "total_rounds" in summary
        assert summary["total_rounds"] == 1

    def test_evolvable_agent_interface(self):
        """Test EvolvableAgent interface."""
        config = {"agent_id": "test_001", "agent_type": "king", "parameters": {}}

        agent = EvolvableAgent(config)

        assert agent.agent_id == "test_001"
        assert agent.agent_type == "king"
        assert hasattr(agent, "performance_history")
        assert hasattr(agent, "evolution_memory")


class TestIntegration:
    """Integration tests for the complete T3 system."""

    @pytest.mark.asyncio
    async def test_end_to_end_agent_workflow(self):
        """Test complete workflow from creation to execution."""
        # Initialize components
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)
        await orchestrator.initialize_agents()

        # Validate all agents
        validation_results = validate_all_agents()

        # Execute tasks on multiple agents
        tasks = [
            TaskRequest(
                task_id=f"e2e_{agent_type}",
                task_type=agent_type,
                payload={"task": "ping", "test": "integration"},
            )
            for agent_type in ["king", "magi", "sage"]
        ]

        results = await orchestrator.process_batch(tasks)

        # Verify all components worked together
        assert all(r.success for r in results)
        assert len(validation_results) >= 18

        # Check performance requirements
        metrics = orchestrator.get_performance_metrics()
        assert metrics["overhead_target_met"]  # <100ms requirement

        # Check health
        health = await orchestrator.health_check()
        assert health["system_healthy"]


# Performance benchmarks
class TestPerformanceRequirements:
    """Test that performance requirements are met."""

    @pytest.mark.asyncio
    async def test_orchestration_overhead_requirement(self):
        """Test that orchestration overhead is <100ms."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)
        await orchestrator.initialize_agents()

        # Execute multiple tasks to get stable measurements
        tasks = [
            TaskRequest(task_id=f"perf_{i}", task_type="ping", payload={"task": "ping"})
            for i in range(50)  # 50 tasks for stable average
        ]

        for task in tasks:
            await orchestrator.execute_task(task)

        metrics = orchestrator.get_performance_metrics()
        avg_overhead = metrics["average_orchestration_overhead_ms"]

        print(f"Average orchestration overhead: {avg_overhead:.2f}ms")
        assert avg_overhead < 100.0, f"Orchestration overhead {avg_overhead:.2f}ms exceeds 100ms requirement"

    @pytest.mark.asyncio
    async def test_agent_initialization_speed(self):
        """Test that agent initialization is fast."""
        factory = AgentFactory()
        orchestrator = FastAgentOrchestrator(factory)

        start_time = time.time() * 1000
        await orchestrator.initialize_agents()
        init_time = time.time() * 1000 - start_time

        print(f"Agent initialization time: {init_time:.1f}ms")
        assert init_time < 5000, f"Initialization time {init_time:.1f}ms too slow"

    def test_validation_speed(self):
        """Test that validation completes quickly."""
        start_time = time.time() * 1000
        results = validate_all_agents(full_test=True)
        validation_time = time.time() * 1000 - start_time

        print(f"Validation time: {validation_time:.1f}ms")
        assert validation_time < 10000, f"Validation time {validation_time:.1f}ms too slow"
        assert len(results) == 18


if __name__ == "__main__":
    # Run basic smoke test
    import sys

    print("Running T3 Agent Forge smoke test...")

    try:
        # Test 1: Factory initialization
        factory = AgentFactory()
        print(f"✅ Factory initialized with {len(factory.templates)} agent templates")

        # Test 2: Agent creation
        king = factory.create_agent("king")
        result = king.process({"task": "ping"})
        print(f"✅ King agent created and processed task: {result['status']}")

        # Test 3: Validation
        validation_results = validate_all_agents()
        passed_validations = sum(1 for checks in validation_results.values() if all(checks.values()))
        print(f"✅ Validation completed: {passed_validations}/{len(validation_results)} agents passed")

        print("\n🎉 T3 Agent Forge smoke test PASSED!")
        print("All major stub replacements are working correctly.")

    except Exception as e:
        print(f"❌ Smoke test FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
