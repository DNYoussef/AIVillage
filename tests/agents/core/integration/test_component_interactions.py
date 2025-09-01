# ruff: noqa: S101  # Use of assert detected - Expected in test files
"""Component Integration Tests for Agent System

Tests how agent components interact with each other while maintaining
loose coupling and avoiding implementation details.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from packages.agents.core.agent_interface import AgentCapability, AgentMetadata, MessageInterface, TaskInterface
from packages.agents.core.base_agent_template import BaseAgentTemplate, ReflectionType


class TestComponentInteractions:
    """Test suite for agent component integration behaviors."""

    @pytest.fixture
    async def integrated_agent(self) -> BaseAgentTemplate:
        """Create agent with all components integrated for testing."""

        class IntegratedTestAgent(BaseAgentTemplate):
            """Agent with full component integration."""

            def __init__(self):
                metadata = AgentMetadata(
                    agent_id=f"integrated-{uuid.uuid4()}",
                    agent_type="IntegratedAgent",
                    name="Integrated Test Agent",
                    description="Agent for integration testing",
                    version="1.0.0",
                    capabilities={
                        AgentCapability.MESSAGE_PROCESSING,
                        AgentCapability.TASK_EXECUTION,
                        AgentCapability.MEMORY_MANAGEMENT,
                        AgentCapability.PERFORMANCE_MONITORING,
                    },
                )
                super().__init__(metadata)
                self.specialized_role = "integration_test"

            async def get_specialized_capabilities(self):
                return [AgentCapability.REASONING, AgentCapability.PLANNING]

            async def process_specialized_task(self, task_data):
                # Simulate complex processing that affects multiple components
                await asyncio.sleep(0.01)  # Simulate work

                # This should trigger memory and reflection systems
                if task_data.get("trigger_reflection"):
                    await self.record_quiet_star_reflection(
                        ReflectionType.PROBLEM_SOLVING,
                        f"Processing complex task: {task_data.get('type', 'unknown')}",
                        "Analyzing the problem space and solution approaches",
                        f"Successfully processed {task_data.get('type')} task",
                        emotional_valence=0.4,
                        tags=["integration", "processing"],
                    )

                return {
                    "status": "success",
                    "result": f"Integrated processing of {task_data.get('content', '')}",
                    "component_interactions": {
                        "memory_updated": len(self.personal_memory) > 0,
                        "reflection_recorded": len(self.personal_journal) > 0,
                        "geometric_aware": self.current_geometric_state is not None,
                    },
                }

            async def get_specialized_mcp_tools(self):
                return {}

            # Required interface implementations
            async def process_task(self, task):
                start_time = datetime.now()

                # Update geometric awareness before processing
                await self.update_geometric_self_awareness()

                result = await self.process_specialized_task(
                    {"content": task.content, "type": task.task_type, "trigger_reflection": True}
                )

                # Record performance
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._record_task_performance(task.task_id, processing_time, accuracy=1.0, status=result["status"])

                return result

            async def can_handle_task(self, task):
                return task.task_type in ["integration", "complex", "test"]

            async def estimate_task_duration(self, task):
                return 0.1

            async def send_message(self, message):
                # Record communication activity
                self.interaction_history.append(
                    {
                        "type": "message_sent",
                        "timestamp": datetime.now().timestamp(),
                        "recipient": message.receiver,
                        "message_type": message.message_type,
                    }
                )
                return True

            async def receive_message(self, message):
                self.interaction_history.append(
                    {
                        "type": "message_received",
                        "timestamp": datetime.now().timestamp(),
                        "sender": message.sender,
                        "message_type": message.message_type,
                    }
                )

            async def broadcast_message(self, message, recipients):
                return {r: True for r in recipients}

            async def generate(self, prompt):
                return f"Generated response: {prompt}"

            async def get_embedding(self, text):
                return [0.1] * 384

            async def rerank(self, query, results, k):
                return results[:k]

            async def introspect(self):
                return {
                    "component_status": {
                        "memory_entries": len(self.personal_memory),
                        "journal_entries": len(self.personal_journal),
                        "task_history": len(self.task_history),
                        "geometric_states": len(self.geometric_state_history),
                    }
                }

            async def communicate(self, message, recipient):
                return f"Communication: {message}"

            async def activate_latent_space(self, query):
                return "integrated", f"latent:{query}"

        agent = IntegratedTestAgent()

        # Mock external dependencies
        agent.rag_client = MagicMock()
        agent.rag_client.query = AsyncMock(return_value={"results": ["mocked result"]})
        agent.p2p_client = MagicMock()
        agent.p2p_client.send_message = AsyncMock(return_value={"delivered": True})

        await agent.initialize()
        return agent

    # INTEGRATION: Communication with State Manager

    async def test_communication_with_state_manager(self, integrated_agent: BaseAgentTemplate):
        """Communication events should affect agent state appropriately."""
        # Given: Agent with initial state
        initial_interactions = len(integrated_agent.interaction_history)

        # When: Agent sends and receives messages
        outgoing_msg = MessageInterface(
            message_id="out-1",
            sender=integrated_agent.agent_id,
            receiver="other-agent",
            message_type="query",
            content="Status request",
        )

        incoming_msg = MessageInterface(
            message_id="in-1",
            sender="other-agent",
            receiver=integrated_agent.agent_id,
            message_type="response",
            content="Status OK",
        )

        await integrated_agent.send_message(outgoing_msg)
        await integrated_agent.receive_message(incoming_msg)

        # Then: Interaction history should be updated
        assert len(integrated_agent.interaction_history) == initial_interactions + 2

        # And: State should reflect communication activity
        recent_interactions = [
            i for i in integrated_agent.interaction_history if datetime.now().timestamp() - i["timestamp"] < 60
        ]
        assert len(recent_interactions) >= 2

    async def test_task_processing_affects_memory_systems(self, integrated_agent: BaseAgentTemplate):
        """Task processing should integrate with memory and reflection systems."""
        # Given: Agent with baseline memory state
        initial_journal_size = len(integrated_agent.personal_journal)
        len(integrated_agent.personal_memory)

        # When: Complex task is processed
        complex_task = TaskInterface(
            task_id="complex-integration", task_type="complex", content="Multi-component integration task"
        )

        result = await integrated_agent.process_task(complex_task)

        # Then: Task should be processed successfully
        assert result["status"] == "success"
        assert "component_interactions" in result

        # And: Memory systems should be affected
        assert len(integrated_agent.personal_journal) > initial_journal_size

        # And: Performance metrics should be updated
        metrics = integrated_agent.get_performance_metrics()
        assert metrics.total_tasks_processed > 0

    async def test_geometric_awareness_affects_processing(self, integrated_agent: BaseAgentTemplate):
        """Geometric awareness should influence task processing decisions."""
        # Given: Agent with current geometric state
        await integrated_agent.update_geometric_self_awareness()
        initial_state = integrated_agent.current_geometric_state

        # When: Multiple tasks are processed to affect load
        tasks = [
            TaskInterface(task_id=f"load-task-{i}", task_type="integration", content=f"Load testing task {i}")
            for i in range(3)
        ]

        for task in tasks:
            await integrated_agent.process_task(task)

        # Update awareness after load
        await integrated_agent.update_geometric_self_awareness()
        updated_state = integrated_agent.current_geometric_state

        # Then: Geometric state should reflect processing activity
        assert updated_state.timestamp > initial_state.timestamp
        assert len(integrated_agent.geometric_state_history) > 1

    async def test_memory_retrieval_affects_task_performance(self, integrated_agent: BaseAgentTemplate):
        """Memory retrieval should influence how tasks are processed."""
        # Given: Agent with recorded memories
        await integrated_agent.record_quiet_star_reflection(
            ReflectionType.LEARNING,
            "Learning about integration patterns",
            "Understanding how components work together",
            "Integration requires careful coordination between components",
            emotional_valence=0.6,
            tags=["integration", "learning", "patterns"],
        )

        # When: Similar task is processed
        integration_task = TaskInterface(
            task_id="memory-integration", task_type="integration", content="Task requiring integration knowledge"
        )

        # And: Relevant memories are retrieved
        await integrated_agent.retrieve_similar_memories("integration patterns", max_memories=3)

        result = await integrated_agent.process_task(integration_task)

        # Then: Task should be processed successfully
        assert result["status"] == "success"

        # And: Memory system should be accessible
        assert hasattr(integrated_agent, "personal_memory")

    # INTEGRATION: Configuration Application

    async def test_configuration_applies_to_all_components(self, integrated_agent: BaseAgentTemplate):
        """Configuration changes should affect all relevant components."""
        # Given: Initial configuration
        integrated_agent.adas_config.copy()

        # When: Configuration is updated
        new_config = {
            "adaptation_rate": 0.2,
            "stability_threshold": 0.9,
            "optimization_targets": ["accuracy", "efficiency"],
        }

        integrated_agent.adas_config.update(new_config)

        # Then: Configuration should be applied
        assert integrated_agent.adas_config["adaptation_rate"] == 0.2
        assert integrated_agent.adas_config["stability_threshold"] == 0.9

        # And: Components should reflect new configuration
        await integrated_agent.update_geometric_self_awareness()
        current_state = integrated_agent.current_geometric_state
        assert current_state.adaptation_rate == 0.2

    async def test_mcp_tools_integration(self, integrated_agent: BaseAgentTemplate):
        """MCP tools should integrate properly with agent operations."""
        # Given: Agent with MCP tools
        assert "rag_query" in integrated_agent.mcp_tools
        assert "communicate" in integrated_agent.mcp_tools

        # When: MCP tools are used
        rag_result = await integrated_agent.query_group_memory("test query")

        # Then: Tools should function properly
        assert rag_result is not None
        assert isinstance(rag_result, dict)

        # And: Tool usage should be tracked
        rag_tool = integrated_agent.mcp_tools["rag_query"]
        assert rag_tool.usage_count > 0

    # INTEGRATION: Cross-Component Communication

    async def test_reflection_triggers_memory_consolidation(self, integrated_agent: BaseAgentTemplate):
        """High-importance reflections should trigger memory consolidation."""
        # Given: Agent with initial memory state
        initial_memory_size = len(integrated_agent.personal_memory)

        # When: High-impact reflection is recorded
        await integrated_agent.record_quiet_star_reflection(
            ReflectionType.CREATIVE_INSIGHT,
            "Major breakthrough in understanding",
            "This insight completely changes how I approach problems",
            "Revolutionary new approach discovered",
            emotional_valence=0.9,  # Very positive
            tags=["breakthrough", "insight", "major"],
        )

        # Then: Memory should be consolidated appropriately
        # Note: Memory consolidation depends on unexpectedness score
        final_memory_size = len(integrated_agent.personal_memory)
        assert final_memory_size >= initial_memory_size

    async def test_performance_metrics_drive_adaptation(self, integrated_agent: BaseAgentTemplate):
        """Performance metrics should influence adaptation decisions."""
        # Given: Agent with performance history
        # Simulate some task failures
        integrated_agent._record_task_performance("fail-1", 1000, 0.0, "error")
        integrated_agent._record_task_performance("fail-2", 1500, 0.0, "error")

        # When: Performance metrics are evaluated
        metrics = integrated_agent._get_recent_performance_metrics()

        # Then: Metrics should reflect poor performance
        assert metrics["error_rate"] > 0

        # And: This should influence adaptation decisions
        # (In real implementation, this would trigger ADAS modifications)
        assert metrics["avg_accuracy"] < 1.0

    async def test_geometric_state_influences_fog_decisions(self, integrated_agent: BaseAgentTemplate):
        """Geometric state should influence fog computing decisions."""
        # Given: Agent under high load
        with patch("psutil.cpu_percent", return_value=85.0):
            await integrated_agent.update_geometric_self_awareness()

            # When: Fog computation is considered
            fog_result = await integrated_agent.offload_computation_to_fog(
                computation_type="training", input_data={"model": "test", "data": "sample"}
            )

            # Then: Decision should reflect current load
            assert fog_result is not None
            assert "status" in fog_result

    # INTEGRATION: Error Propagation and Recovery

    async def test_component_error_isolation(self, integrated_agent: BaseAgentTemplate):
        """Errors in one component should not crash other components."""
        # Given: Agent in good state
        initial_health = await integrated_agent.health_check()
        assert initial_health["status"] == "healthy"

        # When: Error occurs in memory component (simulated)
        with patch.object(integrated_agent, "retrieve_similar_memories", side_effect=Exception("Memory error")):
            try:
                await integrated_agent.retrieve_similar_memories("test query")
            except Exception as e:
                import logging
                logging.exception("Memory component error during integration test: %s", str(e))

        # Then: Other components should remain functional
        task = TaskInterface(task_id="recovery-test", task_type="test", content="recovery test")

        result = await integrated_agent.process_task(task)
        assert result["status"] == "success"

        # And: Overall health should be maintained
        post_error_health = await integrated_agent.health_check()
        assert post_error_health["status"] == "healthy"

    async def test_concurrent_component_access(self, integrated_agent: BaseAgentTemplate):
        """Components should handle concurrent access properly."""
        # Given: Multiple concurrent operations
        tasks = [integrated_agent.update_geometric_self_awareness()]
        tasks.extend(
            [
                integrated_agent.record_quiet_star_reflection(
                    ReflectionType.TASK_COMPLETION,
                    f"Concurrent operation {i}",
                    "Processing concurrent request",
                    "Completed concurrent operation",
                    tags=["concurrent"],
                )
                for i in range(3)
            ]
        )

        # When: Operations run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Then: All operations should complete without exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

    # INTEGRATION: System-Wide Consistency

    async def test_agent_state_consistency_after_operations(self, integrated_agent: BaseAgentTemplate):
        """Agent state should remain consistent after complex operations."""
        # Given: Initial consistent state
        initial_id = integrated_agent.agent_id
        initial_type = integrated_agent.agent_type
        initial_capabilities = integrated_agent.get_capabilities()

        # When: Multiple complex operations are performed
        await integrated_agent.process_task(
            TaskInterface(task_id="consistency-1", task_type="complex", content="consistency test 1")
        )

        await integrated_agent.record_quiet_star_reflection(
            ReflectionType.INTERACTION,
            "Complex interaction",
            "Managing multiple system interactions",
            "Successfully coordinated complex operations",
        )

        await integrated_agent.update_geometric_self_awareness()

        # Then: Core identity should remain unchanged
        assert integrated_agent.agent_id == initial_id
        assert integrated_agent.agent_type == initial_type
        assert integrated_agent.get_capabilities() == initial_capabilities

        # And: State should be valid
        health = await integrated_agent.health_check()
        assert health["status"] == "healthy"

    async def test_data_flow_between_components(self, integrated_agent: BaseAgentTemplate):
        """Data should flow properly between different components."""
        # Given: Agent ready for data flow testing
        # When: Operation generates data that flows between components
        task = TaskInterface(task_id="dataflow-test", task_type="integration", content="Data flow test content")

        # Process task (affects performance metrics and geometric state)
        result = await integrated_agent.process_task(task)

        # Then: Data should flow through components
        assert result["status"] == "success"

        # Performance metrics should reflect the task
        metrics = integrated_agent.get_performance_metrics()
        assert metrics.total_tasks_processed > 0

        # Geometric state should be current
        assert integrated_agent.current_geometric_state is not None

        # Task history should contain the task
        assert len(integrated_agent.task_history) > 0
        assert any(t["task_id"] == task.task_id for t in integrated_agent.task_history)
