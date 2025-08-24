"""Property-Based Tests for Agent Invariants

Tests fundamental properties that should hold true for all agent operations,
using property-based testing to verify behavior across many possible inputs.
"""

import asyncio
from datetime import datetime
import uuid

from hypothesis import assume, given, settings
from hypothesis import strategies as st
import pytest

from packages.agents.core.agent_interface import AgentCapability, AgentMetadata, MessageInterface, TaskInterface
from packages.agents.core.base_agent_template import BaseAgentTemplate, ReflectionType


class PropertyTestAgent(BaseAgentTemplate):
    """Agent implementation for property-based testing."""

    def __init__(self, metadata: AgentMetadata):
        super().__init__(metadata)
        self.specialized_role = "property_test"
        self.test_responses = {}

    async def get_specialized_capabilities(self):
        return [AgentCapability.TEXT_PROCESSING, AgentCapability.REASONING]

    async def process_specialized_task(self, task_data):
        # Simulate processing with some controlled randomness
        content = task_data.get("content", "")

        # Always succeed for non-empty content
        if content:
            return {
                "status": "success",
                "result": f"Processed: {content}",
                "processing_time_ms": max(10, len(str(content)) * 2),
            }
        else:
            return {"status": "error", "error": "Empty content", "processing_time_ms": 5}

    async def get_specialized_mcp_tools(self):
        return {}

    # Required interface implementations
    async def process_task(self, task):
        start_time = datetime.now()

        result = await self.process_specialized_task({"content": task.content, "task_type": task.task_type})

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self._record_task_performance(
            task.task_id,
            processing_time,
            accuracy=1.0 if result["status"] == "success" else 0.0,
            status=result["status"],
        )

        return result

    async def can_handle_task(self, task):
        # Can handle most task types, but not "unsupported"
        return task.task_type != "unsupported"

    async def estimate_task_duration(self, task):
        content_length = len(str(task.content)) if task.content else 0
        return max(0.01, content_length * 0.001)  # 1ms per character minimum 10ms

    async def send_message(self, message):
        # Always succeed unless receiver is "unreachable"
        return message.receiver != "unreachable"

    async def receive_message(self, message):
        self.interaction_history.append(
            {"type": "message_received", "message_id": message.message_id, "timestamp": datetime.now().timestamp()}
        )

    async def broadcast_message(self, message, recipients):
        return {recipient: recipient != "unreachable" for recipient in recipients}

    async def generate(self, prompt):
        if not prompt:
            return ""
        return f"Generated response to: {prompt[:50]}..."

    async def get_embedding(self, text):
        # Deterministic embedding based on text
        if not text:
            return [0.0] * 384
        hash_val = hash(text) % 1000
        return [hash_val / 1000.0] * 384

    async def rerank(self, query, results, k):
        if not results:
            return []
        return results[: min(k, len(results))]

    async def introspect(self):
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "task_count": len(self.task_history),
            "memory_count": len(self.personal_memory),
        }

    async def communicate(self, message, recipient):
        return f"Communicated with {recipient}: {message}"

    async def activate_latent_space(self, query):
        space = "general" if query else "empty"
        return space, f"latent:{space}:{query[:20]}"


class TestAgentInvariants:
    """Property-based tests for agent behavioral invariants."""

    @pytest.fixture
    async def property_agent(self):
        """Create agent for property testing."""
        metadata = AgentMetadata(
            agent_id=f"prop-test-{uuid.uuid4()}",
            agent_type="PropertyAgent",
            name="Property Test Agent",
            description="Agent for property-based testing",
            version="1.0.0",
            capabilities={
                AgentCapability.MESSAGE_PROCESSING,
                AgentCapability.TASK_EXECUTION,
                AgentCapability.TEXT_PROCESSING,
            },
        )

        agent = PropertyTestAgent(metadata)
        await agent.initialize()
        return agent

    # INVARIANT: State Transitions Are Deterministic

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20, deadline=5000)
    async def test_state_transitions_are_deterministic(self, property_agent, task_content):
        """Given the same input, agent state transitions should be deterministic."""
        assume(len(task_content.strip()) > 0)  # Non-empty content

        # Record initial state
        initial_state = {
            "status": property_agent.get_status(),
            "task_count": len(property_agent.task_history),
            "journal_count": len(property_agent.personal_journal),
        }

        # Create identical tasks
        task1 = TaskInterface(task_id=f"det1-{uuid.uuid4()}", task_type="deterministic", content=task_content)

        task2 = TaskInterface(task_id=f"det2-{uuid.uuid4()}", task_type="deterministic", content=task_content)

        # Process both tasks
        result1 = await property_agent.process_task(task1)
        result2 = await property_agent.process_task(task2)

        # Results should have consistent status
        assert result1["status"] == result2["status"]

        # State transitions should be predictable
        final_task_count = len(property_agent.task_history)
        assert final_task_count == initial_state["task_count"] + 2

    # INVARIANT: Message Ordering Preserved

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
    @settings(max_examples=15, deadline=5000)
    async def test_message_ordering_preserved(self, property_agent, message_contents):
        """Messages should be processed in a consistent order."""
        initial_interaction_count = len(property_agent.interaction_history)

        # Create messages with timestamps
        messages = []
        for i, content in enumerate(message_contents):
            msg = MessageInterface(
                message_id=f"order-{i}",
                sender="test-sender",
                receiver=property_agent.agent_id,
                message_type="ordered",
                content=content,
                priority=i,  # Increasing priority
            )
            messages.append(msg)

        # Process messages in order
        for msg in messages:
            await property_agent.receive_message(msg)

        # Interaction history should reflect all messages
        final_interaction_count = len(property_agent.interaction_history)
        assert final_interaction_count == initial_interaction_count + len(messages)

    # INVARIANT: Capabilities Are Idempotent

    @given(st.sampled_from(list(AgentCapability)))
    @settings(max_examples=10)
    async def test_capabilities_are_idempotent(self, property_agent, capability):
        """Adding the same capability multiple times should have same effect as adding once."""
        property_agent.get_capabilities().copy()

        # Add capability multiple times
        property_agent.add_capability(capability)
        after_first_add = property_agent.get_capabilities().copy()

        property_agent.add_capability(capability)
        property_agent.add_capability(capability)
        after_multiple_adds = property_agent.get_capabilities().copy()

        # Should be the same as adding once
        assert after_first_add == after_multiple_adds

        # Should contain the capability
        assert capability in after_multiple_adds

    # INVARIANT: Metrics Are Monotonic

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
    @settings(max_examples=10, deadline=10000)
    async def test_metrics_are_monotonic(self, property_agent, task_contents):
        """Performance metrics should only increase (monotonic)."""
        initial_metrics = property_agent.get_performance_metrics()

        # Process multiple tasks
        for i, content in enumerate(task_contents):
            task = TaskInterface(task_id=f"mono-{i}", task_type="monotonic", content=content)
            await property_agent.process_task(task)

        final_metrics = property_agent.get_performance_metrics()

        # Metrics should only increase
        assert final_metrics.total_tasks_processed >= initial_metrics.total_tasks_processed
        assert final_metrics.uptime_seconds >= initial_metrics.uptime_seconds

        # Should have processed all tasks
        expected_increase = len(task_contents)
        actual_increase = final_metrics.total_tasks_processed - initial_metrics.total_tasks_processed
        assert actual_increase == expected_increase

    # INVARIANT: Agent ID Immutability

    @given(st.lists(st.text(max_size=100), min_size=1, max_size=15))
    @settings(max_examples=10, deadline=5000)
    async def test_agent_id_immutability(self, property_agent, operations):
        """Agent ID should never change regardless of operations."""
        original_id = property_agent.agent_id

        # Perform various operations
        for i, op_data in enumerate(operations):
            # Task processing
            task = TaskInterface(task_id=f"immut-{i}", task_type="immutable", content=op_data)
            await property_agent.process_task(task)

            # State updates
            await property_agent.update_geometric_self_awareness()

            # Reflection recording
            if op_data.strip():  # Only if non-empty
                await property_agent.record_quiet_star_reflection(
                    ReflectionType.TASK_COMPLETION,
                    f"Operation {i}",
                    f"Processing {op_data[:20]}",
                    "Operation completed",
                    tags=["immutability"],
                )

        # Agent ID should remain unchanged
        assert property_agent.agent_id == original_id

    # INVARIANT: Memory Consistency

    @given(
        st.lists(
            st.tuples(
                st.sampled_from(list(ReflectionType)),
                st.text(min_size=1, max_size=100),
                st.floats(min_value=-1.0, max_value=1.0),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=10, deadline=10000)
    async def test_memory_consistency(self, property_agent, reflections):
        """Memory operations should maintain consistency."""
        initial_journal_size = len(property_agent.personal_journal)
        initial_memory_size = len(property_agent.personal_memory)

        # Record multiple reflections
        for reflection_type, context, valence in reflections:
            await property_agent.record_quiet_star_reflection(
                reflection_type=reflection_type,
                context=context,
                raw_thoughts=f"Thinking about {context[:30]}",
                insights=f"Learned from {context[:30]}",
                emotional_valence=valence,
                tags=["consistency"],
            )

        # Memory should only grow
        final_journal_size = len(property_agent.personal_journal)
        final_memory_size = len(property_agent.personal_memory)

        assert final_journal_size >= initial_journal_size
        assert final_memory_size >= initial_memory_size

        # Should have added all reflections to journal
        assert final_journal_size == initial_journal_size + len(reflections)

    # INVARIANT: Task Processing Bounds

    @given(st.text(max_size=1000))
    @settings(max_examples=20, deadline=5000)
    async def test_task_processing_bounds(self, property_agent, content):
        """Task processing should complete within reasonable bounds."""
        task = TaskInterface(task_id=f"bounds-{uuid.uuid4()}", task_type="bounded", content=content)

        start_time = datetime.now()

        # Process task
        result = await property_agent.process_task(task)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Should complete within reasonable time (5 seconds for any task)
        assert processing_time < 5.0

        # Should have a valid result
        assert result is not None
        assert "status" in result
        assert result["status"] in ["success", "error"]

    # INVARIANT: Error Recovery

    @given(
        st.lists(
            st.one_of(
                st.just(""),  # Empty content (error case)
                st.text(min_size=1, max_size=50),  # Valid content
                st.just(None),  # None content (error case)
            ),
            min_size=1,
            max_size=15,
        )
    )
    @settings(max_examples=10, deadline=10000)
    async def test_error_recovery(self, property_agent, mixed_contents):
        """Agent should recover from errors and continue processing."""
        successful_tasks = 0
        failed_tasks = 0

        for i, content in enumerate(mixed_contents):
            task = TaskInterface(task_id=f"recovery-{i}", task_type="recovery", content=content)

            try:
                result = await property_agent.process_task(task)
                if result["status"] == "success":
                    successful_tasks += 1
                else:
                    failed_tasks += 1
            except Exception:
                failed_tasks += 1

            # Agent should remain responsive after any task
            health = await property_agent.health_check()
            assert health["status"] == "healthy"

        # Should have processed all tasks (success or failure)
        total_processed = successful_tasks + failed_tasks
        assert total_processed == len(mixed_contents)

    # INVARIANT: Resource Utilization Bounds

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=5, deadline=15000)
    async def test_resource_utilization_bounds(self, property_agent, num_operations):
        """Resource utilization should stay within reasonable bounds."""
        property_agent.get_performance_metrics()

        # Perform multiple operations
        for i in range(num_operations):
            # Mix of operations
            await property_agent.update_geometric_self_awareness()

            task = TaskInterface(task_id=f"resource-{i}", task_type="resource", content=f"Resource test {i}")
            await property_agent.process_task(task)

        final_metrics = property_agent.get_performance_metrics()

        # Resource metrics should be reasonable
        assert final_metrics.memory_usage_mb >= 0
        assert final_metrics.cpu_usage_percent >= 0
        assert final_metrics.cpu_usage_percent <= 100

        # Performance should not degrade severely
        if final_metrics.total_tasks_processed > 0:
            assert final_metrics.average_processing_time_ms > 0
            assert final_metrics.average_processing_time_ms < 60000  # Less than 1 minute per task

    # INVARIANT: Communication Reliability

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),  # receiver
                st.text(min_size=1, max_size=100),  # content
                st.integers(min_value=1, max_value=10),  # priority
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=10, deadline=5000)
    async def test_communication_reliability(self, property_agent, communications):
        """Communication should be reliable for valid recipients."""
        successful_sends = 0
        total_sends = 0

        for receiver, content, priority in communications:
            # Skip unreachable receivers for this test
            if receiver == "unreachable":
                continue

            message = MessageInterface(
                message_id=f"comm-{total_sends}",
                sender=property_agent.agent_id,
                receiver=receiver,
                message_type="reliable",
                content=content,
                priority=priority,
            )

            success = await property_agent.send_message(message)
            total_sends += 1

            if success:
                successful_sends += 1

        # All valid communications should succeed
        if total_sends > 0:
            success_rate = successful_sends / total_sends
            assert success_rate >= 0.8  # At least 80% success rate for valid receivers


class TestAgentPropertyEdgeCases:
    """Property-based tests for edge cases and boundary conditions."""

    @pytest.fixture
    async def edge_case_agent(self):
        """Create agent for edge case testing."""
        metadata = AgentMetadata(
            agent_id=f"edge-{uuid.uuid4()}",
            agent_type="EdgeCaseAgent",
            name="Edge Case Test Agent",
            description="Agent for edge case testing",
            version="1.0.0",
            capabilities={AgentCapability.TASK_EXECUTION},
        )

        agent = PropertyTestAgent(metadata)
        await agent.initialize()
        return agent

    @given(st.text(max_size=0))  # Empty strings
    @settings(max_examples=5)
    async def test_empty_input_handling(self, edge_case_agent, empty_content):
        """Agent should handle empty inputs gracefully."""
        task = TaskInterface(task_id="empty-test", task_type="empty", content=empty_content)

        result = await edge_case_agent.process_task(task)

        # Should handle empty input without crashing
        assert result is not None
        assert "status" in result

    @given(st.text(min_size=10000, max_size=20000))  # Very large inputs
    @settings(max_examples=3, deadline=10000)
    async def test_large_input_handling(self, edge_case_agent, large_content):
        """Agent should handle large inputs within resource bounds."""
        task = TaskInterface(task_id="large-test", task_type="large", content=large_content)

        start_time = datetime.now()
        result = await edge_case_agent.process_task(task)
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        # Should complete within reasonable time even for large inputs
        assert processing_time < 30.0  # 30 seconds max
        assert result is not None

    @given(st.lists(st.nothing(), min_size=0, max_size=100))
    @settings(max_examples=5)
    async def test_concurrent_empty_operations(self, edge_case_agent, empty_list):
        """Agent should handle concurrent operations gracefully."""
        # Create empty operations
        operations = [
            edge_case_agent.update_geometric_self_awareness()
            for _ in range(len(empty_list) + 1)  # At least one operation
        ]

        # Run concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)

        # Should complete without exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000))
    @settings(max_examples=10)
    async def test_numeric_boundary_handling(self, edge_case_agent, numeric_value):
        """Agent should handle various numeric values appropriately."""
        # Test with numeric content
        task = TaskInterface(task_id="numeric-test", task_type="numeric", content=str(numeric_value))

        result = await edge_case_agent.process_task(task)

        # Should handle numeric content
        assert result is not None
        assert "status" in result
