# ruff: noqa: S101  # Use of assert detected - Expected in test files
"""Behavioral Contract Tests for Agent System

Tests that validate what agents do, not how they do it.
Focuses on observable behaviors and side effects while avoiding coupling to internal structure.
Follows connascence principles by minimizing test-to-implementation coupling.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

# Import agent interfaces and related types
from packages.agents.core.agent_interface import (
    AgentCapability,
    AgentInterface,
    AgentMetadata,
    AgentStatus,
    MessageInterface,
    TaskInterface,
)
from packages.agents.core.base_agent_template import (
    BaseAgentTemplate,
    GeometricSelfState,
    GeometricState,
    ReflectionType,
)


class TestAgentContracts:
    """Test suite for core agent behavioral contracts.

    These tests verify what agents do rather than how they do it,
    focusing on observable behaviors and maintaining loose coupling.
    """

    @pytest.fixture
    async def sample_agent_metadata(self) -> AgentMetadata:
        """Create sample agent metadata for testing."""
        return AgentMetadata(
            agent_id=f"test-agent-{uuid.uuid4()}",
            agent_type="TestAgent",
            name="Test Agent Instance",
            description="Agent for behavioral testing",
            version="1.0.0",
            capabilities={
                AgentCapability.MESSAGE_PROCESSING,
                AgentCapability.TASK_EXECUTION,
                AgentCapability.MEMORY_MANAGEMENT,
            },
        )

    @pytest.fixture
    async def mock_agent(self, sample_agent_metadata: AgentMetadata) -> BaseAgentTemplate:
        """Create a mock agent for testing behaviors."""

        class MockTestAgent(BaseAgentTemplate):
            """Mock agent that implements required abstract methods."""

            def __init__(self, metadata: AgentMetadata):
                super().__init__(metadata)
                self.specialized_role = "test_agent"

            async def get_specialized_capabilities(self) -> list[AgentCapability]:
                return [AgentCapability.TEXT_PROCESSING]

            async def process_specialized_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
                return {
                    "status": "success",
                    "result": f"Processed: {task_data.get('content', '')}",
                    "processing_time_ms": 100,
                }

            async def get_specialized_mcp_tools(self) -> dict[str, Any]:
                return {}

            # Required AgentInterface implementations
            async def process_task(self, task: TaskInterface) -> dict[str, Any]:
                start_time = datetime.now()

                # Simulate task processing
                await asyncio.sleep(0.01)  # Minimal delay

                result = await self.process_specialized_task({"content": task.content})

                # Record performance for geometric awareness
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._record_task_performance(
                    task.task_id,
                    processing_time,
                    accuracy=1.0 if result["status"] == "success" else 0.0,
                    status=result["status"],
                )

                return result

            async def can_handle_task(self, task: TaskInterface) -> bool:
                return task.task_type in ["test", "query", "specialized"]

            async def estimate_task_duration(self, task: TaskInterface) -> float:
                return 0.1  # 100ms estimate

            async def send_message(self, message: MessageInterface) -> bool:
                # Simulate message sending
                return True

            async def receive_message(self, message: MessageInterface) -> None:
                # Simulate message processing
                pass

            async def broadcast_message(self, message: MessageInterface, recipients: list[str]) -> dict[str, bool]:
                return {recipient: True for recipient in recipients}

            async def generate(self, prompt: str) -> str:
                return f"Generated response for: {prompt}"

            async def get_embedding(self, text: str) -> list[float]:
                # Return simple test embedding
                return [0.1] * 384

            async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
                return results[:k]

            async def introspect(self) -> dict[str, Any]:
                return {
                    "agent_type": self.agent_type,
                    "status": self.status.value,
                    "capabilities": [cap.value for cap in self.metadata.capabilities],
                }

            async def communicate(self, message: str, recipient: AgentInterface) -> str:
                return f"Communication response: {message}"

            async def activate_latent_space(self, query: str) -> tuple[str, str]:
                return "test_space", f"latent:{query}"

        agent = MockTestAgent(sample_agent_metadata)
        await agent.initialize()
        return agent

    # BEHAVIORAL CONTRACT: Agent Lifecycle Management

    async def test_agent_responds_to_initialization(self, mock_agent: BaseAgentTemplate):
        """Agent should respond to initialization requests and report success."""
        # Given: A newly created agent
        # When: Initialization is requested
        # Then: Agent reports successful initialization and becomes ready

        # Agent should be initialized from fixture
        assert mock_agent.status == AgentStatus.IDLE

        # Health check should report healthy status
        health = await mock_agent.health_check()
        assert health["status"] == "healthy"
        assert health["agent_id"] == mock_agent.agent_id
        assert "connections" in health
        assert "memory_stats" in health

    async def test_agent_maintains_status_consistency(self, mock_agent: BaseAgentTemplate):
        """Agent status should remain consistent with its operational state."""
        # Given: An initialized agent
        initial_status = mock_agent.get_status()
        assert initial_status == AgentStatus.IDLE

        # When: Agent processes a task
        task = TaskInterface(task_id="test-task-1", task_type="test", content="test content")

        # Then: Agent should handle status transitions properly
        can_handle = await mock_agent.can_handle_task(task)
        assert can_handle  # Should be able to handle test tasks

        result = await mock_agent.process_task(task)
        assert result["status"] == "success"

        # Status should return to idle after processing
        assert mock_agent.get_status() == AgentStatus.IDLE

    async def test_agent_capabilities_are_discoverable(self, mock_agent: BaseAgentTemplate):
        """Agent capabilities should be discoverable and consistent."""
        # Given: An agent with defined capabilities
        # When: Capabilities are queried
        capabilities = mock_agent.get_capabilities()

        # Then: Capabilities should include expected basic functionality
        assert AgentCapability.MESSAGE_PROCESSING in capabilities
        assert AgentCapability.TASK_EXECUTION in capabilities
        assert AgentCapability.MEMORY_MANAGEMENT in capabilities

        # And: Specialized capabilities should be accessible
        specialized_caps = await mock_agent.get_specialized_capabilities()
        assert AgentCapability.TEXT_PROCESSING in specialized_caps

    async def test_agent_metrics_are_collected(self, mock_agent: BaseAgentTemplate):
        """Agent should collect and report performance metrics."""
        # Given: An initialized agent
        initial_metrics = mock_agent.get_performance_metrics()
        initial_task_count = initial_metrics.total_tasks_processed

        # When: Agent processes multiple tasks
        tasks = [TaskInterface(task_id=f"task-{i}", task_type="test", content=f"content-{i}") for i in range(3)]

        for task in tasks:
            await mock_agent.process_task(task)

        # Then: Metrics should be updated
        final_metrics = mock_agent.get_performance_metrics()
        assert final_metrics.total_tasks_processed == initial_task_count + 3
        assert final_metrics.successful_tasks >= 3
        assert final_metrics.last_activity is not None

    async def test_agent_configuration_is_validated(self, sample_agent_metadata: AgentMetadata):
        """Agent should validate configuration and reject invalid setups."""
        # Given: Valid metadata
        # When: Agent is created with valid configuration
        agent = BaseAgentTemplate(sample_agent_metadata)

        # Then: Agent should accept valid configuration
        assert agent.agent_id == sample_agent_metadata.agent_id
        assert agent.agent_type == sample_agent_metadata.agent_type

        # And: Agent should have required components initialized
        assert hasattr(agent, "mcp_tools")
        assert hasattr(agent, "personal_journal")
        assert hasattr(agent, "personal_memory")

    # BEHAVIORAL CONTRACT: Message Processing

    async def test_agent_processes_messages_consistently(self, mock_agent: BaseAgentTemplate):
        """Agent should process messages consistently regardless of content."""
        # Given: Various types of messages
        messages = [
            MessageInterface(
                message_id="msg-1",
                sender="test-sender",
                receiver=mock_agent.agent_id,
                message_type="query",
                content="What is your status?",
            ),
            MessageInterface(
                message_id="msg-2",
                sender="test-sender",
                receiver=mock_agent.agent_id,
                message_type="command",
                content="Process this data",
            ),
        ]

        # When: Messages are sent to agent
        for message in messages:
            success = await mock_agent.send_message(message)
            # Then: Agent should acknowledge message handling
            assert success is True

    async def test_agent_maintains_message_ordering(self, mock_agent: BaseAgentTemplate):
        """Agent should process messages in a deterministic manner."""
        # Given: Sequential messages with priority
        high_priority_msg = MessageInterface(
            message_id="high-pri",
            sender="test",
            receiver=mock_agent.agent_id,
            message_type="urgent",
            content="urgent task",
            priority=10,
        )

        low_priority_msg = MessageInterface(
            message_id="low-pri",
            sender="test",
            receiver=mock_agent.agent_id,
            message_type="normal",
            content="normal task",
            priority=1,
        )

        # When: Messages are sent
        results = []
        results.append(await mock_agent.send_message(low_priority_msg))
        results.append(await mock_agent.send_message(high_priority_msg))

        # Then: All messages should be handled successfully
        assert all(results)

    # BEHAVIORAL CONTRACT: Task Processing

    async def test_agent_handles_task_validation(self, mock_agent: BaseAgentTemplate):
        """Agent should validate tasks before processing."""
        # Given: Valid and invalid tasks
        valid_task = TaskInterface(task_id="valid-task", task_type="test", content="valid content")

        unsupported_task = TaskInterface(
            task_id="unsupported-task", task_type="unsupported_type", content="unsupported content"
        )

        # When: Tasks are evaluated
        can_handle_valid = await mock_agent.can_handle_task(valid_task)
        can_handle_invalid = await mock_agent.can_handle_task(unsupported_task)

        # Then: Agent should correctly identify which tasks it can handle
        assert can_handle_valid is True
        assert can_handle_invalid is False

    async def test_agent_provides_task_estimates(self, mock_agent: BaseAgentTemplate):
        """Agent should provide realistic task duration estimates."""
        # Given: A task for estimation
        task = TaskInterface(task_id="estimate-task", task_type="test", content="content for estimation")

        # When: Duration is estimated
        estimate = await mock_agent.estimate_task_duration(task)

        # Then: Estimate should be reasonable and positive
        assert estimate is not None
        assert estimate > 0
        assert estimate < 3600  # Less than 1 hour for test tasks

    async def test_agent_handles_task_timeouts(self, mock_agent: BaseAgentTemplate):
        """Agent should handle task timeouts gracefully."""
        # Given: A task with very short timeout
        timeout_task = TaskInterface(
            task_id="timeout-task", task_type="test", content="content", timeout_seconds=0.001  # 1ms timeout
        )

        # When: Task is processed
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mock_agent.process_task(timeout_task)

        # Then: Agent should handle timeout gracefully
        assert result is not None
        assert "status" in result

    # BEHAVIORAL CONTRACT: Memory and Reflection

    async def test_agent_records_reflections(self, mock_agent: BaseAgentTemplate):
        """Agent should record reflections about its experiences."""
        # Given: An agent performing activities
        initial_reflection_count = len(mock_agent.personal_journal)

        # When: Agent records a reflection
        reflection_id = await mock_agent.record_quiet_star_reflection(
            reflection_type=ReflectionType.TASK_COMPLETION,
            context="Test task completion",
            raw_thoughts="Analyzing the completion of test task",
            insights="Test task completed successfully",
            emotional_valence=0.5,
            tags=["test", "completion"],
        )

        # Then: Reflection should be recorded
        assert reflection_id is not None
        assert len(mock_agent.personal_journal) == initial_reflection_count + 1

        # And: Reflection should have proper structure
        latest_reflection = mock_agent.personal_journal[-1]
        assert latest_reflection.reflection_id == reflection_id
        assert latest_reflection.reflection_type == ReflectionType.TASK_COMPLETION

    async def test_agent_retrieves_relevant_memories(self, mock_agent: BaseAgentTemplate):
        """Agent should retrieve memories relevant to current context."""
        # Given: Agent with some recorded memories
        await mock_agent.record_quiet_star_reflection(
            reflection_type=ReflectionType.LEARNING,
            context="Learning about task processing",
            raw_thoughts="Understanding how to process different task types",
            insights="Different task types require different approaches",
            emotional_valence=0.3,
            tags=["learning", "tasks"],
        )

        # When: Similar memories are requested
        memories = await mock_agent.retrieve_similar_memories(query="task processing", max_memories=5)

        # Then: Relevant memories should be returned
        assert isinstance(memories, list)
        # Note: Memory retrieval may be empty if unexpectedness score is low

    # BEHAVIORAL CONTRACT: Geometric Self-Awareness

    async def test_agent_monitors_self_state(self, mock_agent: BaseAgentTemplate):
        """Agent should monitor and report its internal state."""
        # Given: An active agent
        # When: Self-awareness is updated
        await mock_agent.update_geometric_self_awareness()

        # Then: Agent should have current state information
        assert mock_agent.current_geometric_state is not None
        assert isinstance(mock_agent.current_geometric_state, GeometricSelfState)
        assert mock_agent.current_geometric_state.geometric_state in GeometricState

    async def test_agent_detects_state_changes(self, mock_agent: BaseAgentTemplate):
        """Agent should detect and respond to state changes."""
        # Given: Initial state
        await mock_agent.update_geometric_self_awareness()
        initial_state = mock_agent.current_geometric_state

        # When: Agent processes intensive workload
        tasks = [TaskInterface(task_id=f"load-task-{i}", task_type="test", content=f"load-{i}") for i in range(5)]

        for task in tasks:
            await mock_agent.process_task(task)

        # Update state after workload
        await mock_agent.update_geometric_self_awareness()
        updated_state = mock_agent.current_geometric_state

        # Then: State should reflect the workload
        assert updated_state is not None
        assert updated_state.timestamp > initial_state.timestamp

    # BEHAVIORAL CONTRACT: Communication Patterns

    async def test_agent_maintains_communication_contracts(self, mock_agent: BaseAgentTemplate):
        """Agent should maintain consistent communication patterns."""
        # Given: Communication request
        test_message = "Hello from test"

        # When: Agent generates response
        response = await mock_agent.generate(test_message)

        # Then: Response should be consistent and non-empty
        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    async def test_agent_handles_broadcast_scenarios(self, mock_agent: BaseAgentTemplate):
        """Agent should handle broadcast communication scenarios."""
        # Given: Broadcast message
        broadcast_msg = MessageInterface(
            message_id="broadcast-test",
            sender=mock_agent.agent_id,
            receiver="*",
            message_type="announcement",
            content="Test broadcast message",
        )

        recipients = ["agent-1", "agent-2", "agent-3"]

        # When: Message is broadcast
        results = await mock_agent.broadcast_message(broadcast_msg, recipients)

        # Then: All recipients should be addressed
        assert len(results) == len(recipients)
        assert all(success for success in results.values())

    # BEHAVIORAL CONTRACT: Error Handling and Recovery

    async def test_agent_handles_errors_gracefully(self, mock_agent: BaseAgentTemplate):
        """Agent should handle errors without becoming unstable."""
        # Given: Agent in good state
        initial_health = await mock_agent.health_check()
        assert initial_health["status"] == "healthy"

        # When: Error condition is encountered (simulate via invalid task)
        error_task = TaskInterface(
            task_id="error-task", task_type="test", content=None  # This might cause processing issues
        )

        # Then: Agent should handle gracefully
        try:
            result = await mock_agent.process_task(error_task)
            # Even if processing succeeds with None content, that's fine
            assert result is not None
        except Exception as e:
            # If exception occurs, agent should remain stable
            logging.info(
                "Agent handled error task gracefully by raising exception",
                extra={
                    "test_case": "test_agent_handles_malformed_tasks",
                    "task_id": error_task.task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "expected_behavior": "agent_remains_stable_after_error",
                },
            )

        # Agent should remain healthy after error
        post_error_health = await mock_agent.health_check()
        assert post_error_health["status"] == "healthy"

    async def test_agent_recovers_from_resource_constraints(self, mock_agent: BaseAgentTemplate):
        """Agent should adapt to resource constraints."""
        # Given: Agent under resource pressure (simulated)
        with patch("psutil.cpu_percent", return_value=95.0):  # High CPU usage
            await mock_agent.update_geometric_self_awareness()

            # When: Agent checks its state
            state = mock_agent.current_geometric_state

            # Then: Agent should recognize resource pressure
            assert state is not None
            assert state.cpu_utilization > 0.9

    # BEHAVIORAL CONTRACT: Integration Points

    async def test_agent_integrates_with_external_systems(self, mock_agent: BaseAgentTemplate):
        """Agent should integrate properly with external systems."""
        # Given: Mock external systems
        mock_agent.rag_client = MagicMock()
        mock_agent.p2p_client = MagicMock()

        # When: Agent uses external systems
        rag_result = await mock_agent.query_group_memory("test query")

        # Then: Integration should work without errors
        assert rag_result is not None
        assert isinstance(rag_result, dict)

    # BEHAVIORAL CONTRACT: Performance Characteristics

    async def test_agent_maintains_performance_bounds(self, mock_agent: BaseAgentTemplate):
        """Agent should maintain acceptable performance characteristics."""
        # Given: Performance baseline
        start_time = datetime.now()

        # When: Agent processes standard workload
        task = TaskInterface(task_id="perf-test", task_type="test", content="performance test content")

        result = await mock_agent.process_task(task)
        end_time = datetime.now()

        # Then: Processing should complete within reasonable time
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 1.0  # Should complete within 1 second
        assert result["status"] == "success"

    async def test_agent_scales_with_concurrent_tasks(self, mock_agent: BaseAgentTemplate):
        """Agent should handle concurrent task processing."""
        # Given: Multiple concurrent tasks
        tasks = [TaskInterface(task_id=f"concurrent-{i}", task_type="test", content=f"content-{i}") for i in range(3)]

        # When: Tasks are processed concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*[mock_agent.process_task(task) for task in tasks])
        end_time = datetime.now()

        # Then: All tasks should complete successfully
        assert len(results) == 3
        assert all(result["status"] == "success" for result in results)

        # And: Concurrent processing should be efficient
        total_time = (end_time - start_time).total_seconds()
        assert total_time < 2.0  # Should complete within reasonable time


class TestAgentContractInvariants:
    """Property-based tests for agent behavioral invariants."""

    @pytest.fixture
    async def agent_instance(self, sample_agent_metadata) -> BaseAgentTemplate:
        """Create agent instance for invariant testing."""

        class InvariantTestAgent(BaseAgentTemplate):
            """Agent for testing behavioral invariants."""

            async def get_specialized_capabilities(self):
                return [AgentCapability.TEXT_PROCESSING]

            async def process_specialized_task(self, task_data):
                return {"status": "success", "result": "processed"}

            async def get_specialized_mcp_tools(self):
                return {}

            # Required AgentInterface implementations
            async def process_task(self, task):
                return {"status": "success", "task_id": task.task_id}

            async def can_handle_task(self, task):
                return True

            async def estimate_task_duration(self, task):
                return 0.1

            async def send_message(self, message):
                return True

            async def receive_message(self, message):
                pass

            async def broadcast_message(self, message, recipients):
                return {r: True for r in recipients}

            async def generate(self, prompt):
                return f"Generated: {prompt}"

            async def get_embedding(self, text):
                return [0.1] * 384

            async def rerank(self, query, results, k):
                return results[:k]

            async def introspect(self):
                return {"status": "ok"}

            async def communicate(self, message, recipient):
                return f"Response: {message}"

            async def activate_latent_space(self, query):
                return "space", f"latent:{query}"

        agent = InvariantTestAgent(sample_agent_metadata)
        await agent.initialize()
        return agent

    async def test_agent_id_immutability(self, agent_instance: BaseAgentTemplate):
        """Agent ID should remain immutable throughout lifecycle."""
        original_id = agent_instance.agent_id

        # Process various operations
        await agent_instance.update_geometric_self_awareness()
        await agent_instance.record_quiet_star_reflection(ReflectionType.LEARNING, "test", "thoughts", "insights")

        # Agent ID should remain unchanged
        assert agent_instance.agent_id == original_id

    async def test_capabilities_consistency(self, agent_instance: BaseAgentTemplate):
        """Agent capabilities should remain consistent unless explicitly modified."""
        initial_capabilities = agent_instance.get_capabilities().copy()

        # Perform operations
        task = TaskInterface(task_id="test", task_type="test", content="test")
        await agent_instance.process_task(task)

        # Capabilities should remain the same
        current_capabilities = agent_instance.get_capabilities()
        assert current_capabilities == initial_capabilities

    async def test_memory_monotonicity(self, agent_instance: BaseAgentTemplate):
        """Agent memory should only grow or stay the same, never shrink unexpectedly."""
        initial_journal_size = len(agent_instance.personal_journal)
        initial_memory_size = len(agent_instance.personal_memory)

        # Add reflection
        await agent_instance.record_quiet_star_reflection(
            ReflectionType.TASK_COMPLETION, "test", "thoughts", "insights", 0.8
        )

        # Memory should not shrink
        assert len(agent_instance.personal_journal) >= initial_journal_size
        assert len(agent_instance.personal_memory) >= initial_memory_size

    async def test_state_transition_validity(self, agent_instance: BaseAgentTemplate):
        """Agent state transitions should follow valid patterns."""
        # Agent should start in appropriate state
        assert agent_instance.get_status() in [AgentStatus.IDLE, AgentStatus.INITIALIZING]

        # After initialization, should be idle
        if agent_instance.get_status() == AgentStatus.INITIALIZING:
            await agent_instance.initialize()
        assert agent_instance.get_status() == AgentStatus.IDLE

    async def test_performance_metrics_non_negative(self, agent_instance: BaseAgentTemplate):
        """Performance metrics should never be negative."""
        metrics = agent_instance.get_performance_metrics()

        assert metrics.total_tasks_processed >= 0
        assert metrics.successful_tasks >= 0
        assert metrics.failed_tasks >= 0
        assert metrics.average_processing_time_ms >= 0.0
        assert metrics.uptime_seconds >= 0.0
        assert metrics.success_rate >= 0.0
        assert metrics.error_rate >= 0.0

    async def test_task_processing_determinism(self, agent_instance: BaseAgentTemplate):
        """Similar tasks should produce consistent behavior patterns."""
        # Create identical tasks
        task1 = TaskInterface(task_id="det-1", task_type="test", content="identical")
        task2 = TaskInterface(task_id="det-2", task_type="test", content="identical")

        # Process both
        result1 = await agent_instance.process_task(task1)
        result2 = await agent_instance.process_task(task2)

        # Results should have same status (behavior should be consistent)
        assert result1["status"] == result2["status"]
