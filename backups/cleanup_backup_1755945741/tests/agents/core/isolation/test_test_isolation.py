"""Test Isolation Validation

Tests that ensure test isolation and prevent interference between tests.
Validates that tests follow connascence principles and maintain proper boundaries.
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

from packages.agents.core.base_agent_template import ReflectionType
from tests.agents.core.fixtures.conftest import MockTestAgent
from tests.agents.core.fixtures.test_builders import TaskInterfaceBuilder


class TestIsolationValidation:
    """Tests that validate proper test isolation."""

    async def test_agent_instances_are_isolated(self, agent_factory):
        """Each test should get an isolated agent instance."""
        # Given: Multiple agent instances
        agent1 = await agent_factory("IsolationAgent1")
        agent2 = await agent_factory("IsolationAgent2")

        # When: Modifying state in one agent
        task = TaskInterfaceBuilder().with_content("Isolation test").build()
        await agent1.process_task(task)

        await agent1.record_quiet_star_reflection(
            ReflectionType.LEARNING,
            "Learning from isolation test",
            "Understanding isolation",
            "Isolation is important",
            tags=["isolation"],
        )

        # Then: Other agent should be unaffected
        assert len(agent1.personal_journal) > 0
        assert len(agent2.personal_journal) == 0

        assert len(agent1.task_history) > 0
        assert len(agent2.task_history) == 0

        # And: Agents should have different identities
        assert agent1.agent_id != agent2.agent_id
        assert agent1.metadata.name != agent2.metadata.name

    async def test_external_service_mocks_are_isolated(self, agent_factory):
        """External service mocks should be isolated between agents."""
        # Given: Two agents with different mock configurations
        agent1 = await agent_factory("MockAgent1")
        agent2 = await agent_factory("MockAgent2")

        # Configure different mock behaviors
        agent1.rag_client.query.return_value = {"results": ["agent1 result"]}
        agent2.rag_client.query.return_value = {"results": ["agent2 result"]}

        # When: Agents use their RAG clients
        result1 = await agent1.query_group_memory("test query")
        result2 = await agent2.query_group_memory("test query")

        # Then: Each agent should get its own mock response
        assert "agent1 result" in str(result1)
        assert "agent2 result" in str(result2)
        assert result1 != result2

    async def test_global_state_isolation(self, mock_agent: MockTestAgent):
        """Tests should not affect global state."""
        # Given: Global state that could be modified
        original_env = {}
        import os

        test_env_vars = ["TEST_VAR_1", "TEST_VAR_2"]

        for var in test_env_vars:
            original_env[var] = os.environ.get(var)

        try:
            # When: Test modifies environment
            os.environ["TEST_VAR_1"] = "test_value_1"
            os.environ["TEST_VAR_2"] = "test_value_2"

            # Process some tasks
            task = TaskInterfaceBuilder().with_content("Global state test").build()
            result = await mock_agent.process_task(task)

            # Then: Task should complete successfully
            assert result["status"] == "success"

            # And: Environment changes should be contained to this test
            assert os.environ.get("TEST_VAR_1") == "test_value_1"
            assert os.environ.get("TEST_VAR_2") == "test_value_2"

        finally:
            # Cleanup: Restore original environment
            for var in test_env_vars:
                if original_env[var] is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original_env[var]

    async def test_filesystem_isolation(self, mock_agent: MockTestAgent, tmp_path):
        """Tests should not interfere with filesystem state."""
        # Given: Isolated temporary directory
        test_file = tmp_path / "agent_test_file.txt"
        test_content = f"Test content for {mock_agent.agent_id}"

        # When: Agent performs operations that might affect filesystem
        test_file.write_text(test_content)

        # Simulate agent operations that might involve file I/O
        for i in range(5):
            task = TaskInterfaceBuilder().with_content(f"File test {i}").build()
            await mock_agent.process_task(task)

        # Then: File should remain unchanged
        assert test_file.exists()
        assert test_file.read_text() == test_content

        # And: No additional files should be created in temp dir
        created_files = list(tmp_path.iterdir())
        assert len(created_files) == 1
        assert created_files[0] == test_file

    async def test_time_isolation(self, mock_agent: MockTestAgent):
        """Tests should handle time mocking without interference."""
        # Given: Fixed time for deterministic testing
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)

        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # When: Agent records reflection with mocked time
            reflection_id = await mock_agent.record_quiet_star_reflection(
                ReflectionType.TASK_COMPLETION,
                "Time isolation test",
                "Testing time isolation",
                "Time is properly mocked",
                tags=["time_test"],
            )

            # Then: Reflection should use mocked time
            reflection = next(r for r in mock_agent.personal_journal if r.reflection_id == reflection_id)
            assert reflection.timestamp == fixed_time

    async def test_concurrent_test_isolation(self, agent_factory):
        """Concurrent tests should not interfere with each other."""

        # Given: Multiple concurrent test scenarios
        async def isolated_test_scenario(test_id: int):
            agent = await agent_factory(f"ConcurrentAgent{test_id}")

            # Perform test-specific operations
            results = []
            for i in range(5):
                task = TaskInterfaceBuilder().with_content(f"Concurrent test {test_id}-{i}").build()
                result = await agent.process_task(task)
                results.append(result)

            # Return test-specific data
            return {
                "test_id": test_id,
                "agent_id": agent.agent_id,
                "results": results,
                "journal_size": len(agent.personal_journal),
                "task_count": len(agent.task_history),
            }

        # When: Running multiple test scenarios concurrently
        num_concurrent_tests = 5
        test_results = await asyncio.gather(*[isolated_test_scenario(i) for i in range(num_concurrent_tests)])

        # Then: Each test should have succeeded independently
        assert len(test_results) == num_concurrent_tests

        for result in test_results:
            # All tasks should have succeeded
            assert all(r["status"] == "success" for r in result["results"])
            # Each agent should have unique ID
            assert result["agent_id"] is not None

        # Verify agent IDs are unique
        agent_ids = [result["agent_id"] for result in test_results]
        assert len(set(agent_ids)) == num_concurrent_tests, "Agent IDs should be unique"

    async def test_mock_isolation_between_tests(self, agent_factory):
        """Mock configurations should be isolated between different tests."""
        # Given: First agent with specific mock configuration
        agent1 = await agent_factory("MockIsolationAgent1")
        agent1.rag_client.query.return_value = {"status": "success", "results": ["test1"]}

        # Verify first configuration works
        result1 = await agent1.query_group_memory("test query")
        assert "test1" in str(result1)

        # When: Creating second agent
        agent2 = await agent_factory("MockIsolationAgent2")

        # Then: Second agent should have independent mock configuration
        # (Should not inherit the configuration from agent1)
        agent2.rag_client.query.return_value = {"status": "success", "results": ["test2"]}
        result2 = await agent2.query_group_memory("test query")

        assert "test2" in str(result2)
        assert "test1" not in str(result2)

        # And: Original agent should still work with its configuration
        result1_again = await agent1.query_group_memory("test query")
        assert "test1" in str(result1_again)


class TestConnascenceViolationPrevention:
    """Tests that prevent connascence violations in test suites."""

    async def test_no_positional_parameter_connascence(self, mock_agent: MockTestAgent):
        """Tests should avoid positional parameter connascence."""
        # Given: Task creation using named parameters (good)
        good_task = TaskInterfaceBuilder().with_type("good").with_content("good content").build()

        # When: Processing task
        result = await mock_agent.process_task(good_task)

        # Then: Should work correctly
        assert result["status"] == "success"

        # Note: This test demonstrates proper usage avoiding:
        # TaskInterface("task-id", "type", "content", 0, None, {}, {}, datetime.now())
        # which would create positional parameter connascence

    async def test_no_magic_number_connascence(self, mock_agent: MockTestAgent):
        """Tests should avoid magic numbers and use named constants."""
        # Given: Named constants instead of magic numbers
        HIGH_PRIORITY = 10

        PERFORMANCE_THRESHOLD_MS = 100

        # When: Using named constants in tests
        high_priority_task = (
            TaskInterfaceBuilder().with_priority(HIGH_PRIORITY).with_content("High priority task").build()
        )

        start_time = datetime.now()
        result = await mock_agent.process_task(high_priority_task)
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds() * 1000

        # Then: Assertions use named constants
        assert result["status"] == "success"
        assert processing_time < PERFORMANCE_THRESHOLD_MS

        # Note: This avoids magic numbers like:
        # assert processing_time < 100  # What does 100 mean?
        # task.priority = 10           # What does 10 mean?

    async def test_no_algorithm_duplication_connascence(self, mock_agent: MockTestAgent):
        """Tests should not duplicate business logic algorithms."""
        # Given: Test uses agent's own validation logic
        task = TaskInterfaceBuilder().with_type("validation_test").build()

        # When: Checking if agent can handle task
        can_handle = await mock_agent.can_handle_task(task)

        # Then: Use agent's own logic, don't duplicate it
        assert can_handle is True

        # Note: This avoids duplicating validation logic like:
        # assert task.task_type in ["test", "query", "process"]  # Duplicating agent logic
        # Instead, we trust the agent's can_handle_task method

    async def test_no_temporal_connascence(self, mock_agent: MockTestAgent):
        """Tests should avoid temporal coupling between operations."""
        # Given: Operations that could be order-dependent
        task1 = TaskInterfaceBuilder().with_content("First task").build()
        task2 = TaskInterfaceBuilder().with_content("Second task").build()

        # When: Processing in different orders
        # Order 1: task1 then task2
        result1a = await mock_agent.process_task(task1)
        result2a = await mock_agent.process_task(task2)

        # Create fresh agent for different order
        fresh_agent = MockTestAgent(mock_agent.metadata)
        await fresh_agent.initialize()

        # Order 2: task2 then task1
        result2b = await fresh_agent.process_task(task2)
        result1b = await fresh_agent.process_task(task1)

        # Then: Results should be independent of order
        assert result1a["status"] == result1b["status"]
        assert result2a["status"] == result2b["status"]

        # Note: This ensures tests don't have temporal coupling like:
        # "Must call setup_method() before test_method()"

    async def test_no_identity_connascence(self, agent_factory):
        """Tests should not depend on specific object identities."""
        # Given: Multiple agents of the same type
        agent1 = await agent_factory("IdentityTestAgent")
        agent2 = await agent_factory("IdentityTestAgent")

        # When: Performing identical operations
        task_content = "Identity test content"

        task1 = TaskInterfaceBuilder().with_content(task_content).build()
        task2 = TaskInterfaceBuilder().with_content(task_content).build()

        result1 = await agent1.process_task(task1)
        result2 = await agent2.process_task(task2)

        # Then: Results should be equivalent regardless of agent identity
        assert result1["status"] == result2["status"]
        assert "processed" in result1["result"].lower()
        assert "processed" in result2["result"].lower()

        # Note: This avoids identity connascence like:
        # assert result1 is result2  # Testing object identity
        # assert agent1 == agent2    # Comparing object identity


class TestTestDataConsistency:
    """Tests that ensure test data consistency and prevent coupling."""

    async def test_builder_pattern_consistency(self, mock_agent: MockTestAgent):
        """Builders should create consistent test data."""
        # Given: Multiple instances created with same builder pattern
        tasks = [TaskInterfaceBuilder().with_type("consistency").with_content("test").build() for _ in range(5)]

        # When: Processing all tasks
        results = []
        for task in tasks:
            result = await mock_agent.process_task(task)
            results.append(result)

        # Then: All results should be consistent
        statuses = [r["status"] for r in results]
        assert all(status == "success" for status in statuses)

        # And: Task IDs should be unique (proper builder behavior)
        task_ids = [task.task_id for task in tasks]
        assert len(set(task_ids)) == len(task_ids), "Task IDs should be unique"

    async def test_fixture_independence(self, agent_factory):
        """Fixtures should be independent and not share state."""
        # Given: Multiple agents from fixture
        agents = [await agent_factory(f"FixtureAgent{i}") for i in range(3)]

        # When: Modifying state in one agent
        task = TaskInterfaceBuilder().with_content("Fixture test").build()
        await agents[0].process_task(task)

        # Modify first agent's configuration
        agents[0].adas_config["adaptation_rate"] = 0.9

        # Then: Other agents should be unaffected
        for i, agent in enumerate(agents[1:], 1):
            assert len(agent.task_history) == 0, f"Agent {i} task history should be empty"
            assert agent.adas_config["adaptation_rate"] != 0.9, f"Agent {i} config should be independent"

    async def test_no_shared_mutable_state(self, agent_factory):
        """Test data should not share mutable state between instances."""
        # Given: Two agents that might share state
        agent1 = await agent_factory("StateAgent1")
        agent2 = await agent_factory("StateAgent2")

        # When: Modifying lists/dicts in one agent
        agent1.personal_journal.append("test_entry")
        agent1.adas_config["new_key"] = "test_value"
        agent1.metadata.tags.append("shared_test")

        # Then: Other agent should not be affected
        assert "test_entry" not in agent2.personal_journal
        assert "new_key" not in agent2.adas_config
        assert "shared_test" not in agent2.metadata.tags

        # And: Original modifications should persist
        assert "test_entry" in agent1.personal_journal
        assert agent1.adas_config["new_key"] == "test_value"
        assert "shared_test" in agent1.metadata.tags
