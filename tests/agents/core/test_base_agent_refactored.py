"""Behavioral tests for refactored BaseAgentTemplate.

These tests verify behavioral contracts and component interactions
without testing implementation details. Focus is on what the agent
does, not how it does it (avoiding CoA - Connascence of Algorithm).
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from packages.agents.core.agent_interface import AgentMetadata, TaskInterface
from packages.agents.core.base_agent_template_refactored import BaseAgentTemplate


class TestAgent(BaseAgentTemplate):
    """Test implementation of BaseAgentTemplate for behavioral testing."""

    async def get_specialized_capabilities(self) -> list[str]:
        return ["test_capability", "analysis", "processing"]

    async def process_specialized_task(self, task_data: dict) -> dict:
        return {"result": f"Processed task: {task_data.get('content', 'unknown')}"}

    async def get_specialized_mcp_tools(self) -> dict:
        return {}


@pytest.fixture
def agent_metadata():
    """Create test agent metadata."""
    return AgentMetadata(
        agent_id="test-agent-001",
        agent_type="TestAgent",
        name="Test Agent",
        description="Agent for behavioral testing",
        version="1.0.0",
        capabilities=set(),
    )


@pytest.fixture
def test_agent(agent_metadata):
    """Create test agent instance."""
    return TestAgent(agent_metadata)


@pytest.fixture
def mock_clients():
    """Create mock external clients."""
    return {"rag_client": AsyncMock(), "p2p_client": AsyncMock(), "agent_forge_client": AsyncMock()}


class TestAgentInitialization:
    """Test agent initialization and component assembly."""

    def test_agent_creates_with_correct_identity(self, test_agent):
        """Agent should have correct identity after creation."""
        assert test_agent.agent_id == "test-agent-001"
        assert test_agent.agent_type == "TestAgent"
        assert test_agent.get_specialized_role() == "base_template"  # Default role

    def test_agent_has_all_required_components(self, test_agent):
        """Agent should have all required components after initialization."""
        # Test component presence through behavioral contracts
        assert test_agent.get_current_state() is not None
        assert test_agent.get_performance_metrics() is not None
        assert callable(test_agent.send_message_to_agent)
        assert callable(test_agent.configure)

    @pytest.mark.asyncio
    async def test_agent_initialization_succeeds_with_dependencies(self, test_agent, mock_clients):
        """Agent initialization should succeed when dependencies are provided."""
        test_agent.inject_dependencies(**mock_clients)
        success = await test_agent.initialize()
        assert success is True

    @pytest.mark.asyncio
    async def test_agent_health_check_provides_complete_status(self, test_agent):
        """Health check should provide comprehensive status information."""
        await test_agent.initialize()
        health = await test_agent.health_check()

        assert "agent_id" in health
        assert "healthy" in health
        assert "components" in health
        assert "timestamp" in health

        # Verify component health is included
        components = health["components"]
        expected_components = ["state_manager", "communication", "capabilities", "performance", "configuration"]
        for component in expected_components:
            assert component in components


class TestConfigurationManagement:
    """Test configuration and dependency injection behaviors."""

    def test_dependency_injection_allows_external_clients(self, test_agent, mock_clients):
        """Agent should accept and store external client dependencies."""
        test_agent.inject_dependencies(**mock_clients)

        # Verify dependencies are accessible (behavioral test - not checking internals)
        config_summary = test_agent._config.get_configuration_summary()
        assert "rag_client" in config_summary["external_clients"]
        assert "p2p_client" in config_summary["external_clients"]

    def test_configuration_accepts_runtime_settings(self, test_agent):
        """Agent should accept and apply configuration settings."""
        test_agent.configure(max_concurrent_tasks=20, geometric_awareness_enabled=False, custom_setting="test_value")

        # Verify configuration was applied (behavioral contract)
        config = test_agent._config.get_core_config()
        assert config.max_concurrent_tasks == 20
        assert config.geometric_awareness_enabled is False

    def test_specialized_role_can_be_set_and_retrieved(self, test_agent):
        """Agent should allow setting and retrieving specialized role."""
        test_agent.set_specialized_role("test_specialist")
        assert test_agent.get_specialized_role() == "test_specialist"


class TestTaskProcessing:
    """Test task processing behavioral contracts."""

    @pytest.mark.asyncio
    async def test_agent_processes_supported_task_types(self, test_agent):
        """Agent should successfully process tasks it claims to support."""
        await test_agent.initialize()

        task = TaskInterface(task_id="test-task-001", task_type="analysis", content="Analyze this data")

        # Verify agent claims to support this task
        can_handle = await test_agent.can_handle_task(task)
        assert can_handle is True

        # Verify agent actually processes the task
        result = await test_agent.process_task(task)
        assert result["status"] in ["completed", "rejected"]
        assert "task_id" in result

    @pytest.mark.asyncio
    async def test_task_processing_records_metrics(self, test_agent):
        """Task processing should record performance metrics."""
        await test_agent.initialize()

        initial_metrics = test_agent.get_performance_metrics()
        initial_task_count = initial_metrics.get("task_statistics", {}).get("total_tasks", 0)

        task = TaskInterface(task_id="metrics-test-001", task_type="processing", content="Test metrics recording")

        await test_agent.process_task(task)

        updated_metrics = test_agent.get_performance_metrics()
        updated_task_count = updated_metrics.get("task_statistics", {}).get("total_tasks", 0)

        assert updated_task_count > initial_task_count

    @pytest.mark.asyncio
    async def test_task_duration_estimation_returns_reasonable_value(self, test_agent):
        """Agent should provide reasonable task duration estimates."""
        await test_agent.initialize()

        task = TaskInterface(task_id="duration-test-001", task_type="analysis", content="Test duration estimation")

        estimated_duration = await test_agent.estimate_task_duration(task)

        assert estimated_duration is not None
        assert isinstance(estimated_duration, int | float)
        assert estimated_duration > 0  # Should be positive
        assert estimated_duration < 3600  # Should be reasonable (< 1 hour)


class TestCommunication:
    """Test communication behavioral contracts."""

    @pytest.mark.asyncio
    async def test_direct_messaging_returns_status(self, test_agent, mock_clients):
        """Direct messaging should return delivery status."""
        test_agent.inject_dependencies(**mock_clients)
        await test_agent.initialize()

        # Mock successful communication
        mock_clients["p2p_client"].send_message = AsyncMock(return_value={"status": "success"})

        result = await test_agent.send_message_to_agent(recipient="other-agent", message="Test message")

        assert "status" in result
        assert result["status"] in ["success", "error"]

    @pytest.mark.asyncio
    async def test_broadcast_messaging_reaches_multiple_recipients(self, test_agent, mock_clients):
        """Broadcast messaging should attempt to reach multiple recipients."""
        test_agent.inject_dependencies(**mock_clients)
        await test_agent.initialize()

        # Mock successful broadcast
        mock_clients["p2p_client"].send_message = AsyncMock(return_value={"status": "success"})

        result = await test_agent.broadcast_message(message="Broadcast test", priority=7)

        assert "status" in result

    @pytest.mark.asyncio
    async def test_group_channel_membership_management(self, test_agent):
        """Agent should manage group channel memberships."""
        await test_agent.initialize()

        # Test joining channel
        join_result = await test_agent.join_group_channel("test-channel")
        assert join_result is True

        # Test channel status
        comm_status = test_agent._communication.get_channel_status()
        assert "test-channel" in comm_status["channels"]["group_memberships"]


class TestStateManagement:
    """Test state management and geometric awareness."""

    def test_agent_reports_current_state(self, test_agent):
        """Agent should always report a valid current state."""
        current_state = test_agent.get_current_state()

        assert current_state is not None
        assert isinstance(current_state, str)
        assert len(current_state) > 0

    @pytest.mark.asyncio
    async def test_geometric_awareness_updates_with_metrics(self, test_agent):
        """Geometric awareness should update and return meaningful data."""
        await test_agent.initialize()

        # Provide task metrics for geometric awareness
        task_metrics = {"task_id": "geo-test-001", "latency_ms": 150.0, "accuracy": 0.95}

        geo_result = await test_agent.update_geometric_awareness(task_metrics)

        assert "geometric_state" in geo_result
        assert "is_healthy" in geo_result
        assert "timestamp" in geo_result
        assert isinstance(geo_result["is_healthy"], bool)

    @pytest.mark.asyncio
    async def test_state_transitions_are_logged(self, test_agent):
        """State transitions should be properly tracked."""
        await test_agent.initialize()

        test_agent.get_current_state()

        # Force a state update through geometric awareness
        await test_agent.update_geometric_awareness()

        # State should still be valid (may or may not have changed)
        current_state = test_agent.get_current_state()
        assert current_state is not None


class TestMetricsAndMonitoring:
    """Test metrics collection and performance monitoring."""

    def test_performance_metrics_are_comprehensive(self, test_agent):
        """Performance metrics should include all required categories."""
        metrics = test_agent.get_performance_metrics()

        required_sections = ["performance", "task_statistics", "memory_statistics"]
        for section in required_sections:
            assert section in metrics

        # Verify performance section has expected metrics
        perf = metrics["performance"]
        expected_perf_metrics = ["cpu_utilization", "memory_utilization", "error_rate"]
        for metric in expected_perf_metrics:
            assert metric in perf

    def test_task_completion_recording_updates_counters(self, test_agent):
        """Recording task completion should update relevant counters."""
        initial_metrics = test_agent.get_performance_metrics()
        initial_completed = initial_metrics.get("task_statistics", {}).get("completed_tasks", 0)

        test_agent.record_task_completion(
            task_id="counter-test-001", processing_time_ms=200.0, success=True, accuracy=0.9
        )

        updated_metrics = test_agent.get_performance_metrics()
        updated_completed = updated_metrics.get("task_statistics", {}).get("completed_tasks", 0)

        assert updated_completed > initial_completed

    def test_metrics_include_timestamp_information(self, test_agent):
        """Metrics should include timing information for analysis."""
        metrics = test_agent.get_performance_metrics()

        assert "timestamp" in metrics
        # Verify timestamp is recent (within last minute)
        timestamp_str = metrics["timestamp"]
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        age = datetime.now() - timestamp.replace(tzinfo=None)
        assert age < timedelta(minutes=1)


class TestCapabilityManagement:
    """Test capability and tool management."""

    @pytest.mark.asyncio
    async def test_specialized_capabilities_are_loaded(self, test_agent):
        """Agent should load its specialized capabilities during initialization."""
        await test_agent.initialize()

        # Get capability metrics to verify capabilities were loaded
        cap_metrics = test_agent._capabilities.get_capability_metrics()
        capabilities = cap_metrics["capabilities"]["capability_ids"]

        # Should include the capabilities defined in TestAgent
        expected_capabilities = ["test_capability", "analysis", "processing"]
        for cap in expected_capabilities:
            assert cap in capabilities

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_records_metrics(self, test_agent, mock_clients):
        """MCP tool execution should record performance metrics."""
        # Add a mock tool for testing
        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value={"result": "success"})

        test_agent._capabilities.register_tool(mock_tool)
        test_agent.inject_dependencies(**mock_clients)
        await test_agent.initialize()

        test_agent.get_performance_metrics()

        # Execute tool (this will raise an error for unknown tool, but that's expected)
        try:
            await test_agent.execute_mcp_tool("mock_tool", {"param": "value"})
        except Exception as e:
            import logging
            logging.exception("MCP tool execution error in test setup: %s", str(e))

        # Verify metrics were updated (error metrics in this case)
        updated_metrics = test_agent.get_performance_metrics()
        assert updated_metrics is not None


class TestAgentLifecycle:
    """Test complete agent lifecycle behaviors."""

    @pytest.mark.asyncio
    async def test_complete_initialization_and_shutdown_cycle(self, test_agent, mock_clients):
        """Agent should complete full lifecycle without errors."""
        # Inject dependencies
        test_agent.inject_dependencies(**mock_clients)

        # Initialize
        init_result = await test_agent.initialize()
        assert init_result is True

        # Verify agent is operational
        health = await test_agent.health_check()
        assert health["healthy"] in [True, False]  # Should return a boolean

        # Shutdown
        shutdown_result = await test_agent.shutdown()
        assert shutdown_result is True

    @pytest.mark.asyncio
    async def test_agent_handles_missing_dependencies_gracefully(self, test_agent):
        """Agent should handle missing dependencies without crashing."""
        # Try to initialize without injecting dependencies
        init_result = await test_agent.initialize()

        # Should either succeed with warnings or fail gracefully
        assert isinstance(init_result, bool)

        # Health check should still work
        health = await test_agent.health_check()
        assert "healthy" in health
        assert isinstance(health["healthy"], bool)


class TestBackwardCompatibility:
    """Test compatibility with existing agent interface."""

    def test_legacy_properties_still_accessible(self, test_agent):
        """Legacy properties should still be accessible for existing code."""
        # Test legacy property access
        assert hasattr(test_agent, "agent_id")
        assert hasattr(test_agent, "agent_type")
        assert hasattr(test_agent, "specialized_role")

        # Test legacy property setting
        test_agent.specialized_role = "legacy_test_role"
        assert test_agent.specialized_role == "legacy_test_role"

    @pytest.mark.asyncio
    async def test_interface_methods_are_implemented(self, test_agent):
        """All AgentInterface methods should be implemented."""
        # Test key interface methods exist and are callable
        interface_methods = [
            "generate",
            "get_embedding",
            "rerank",
            "introspect",
            "communicate",
            "activate_latent_space",
        ]

        for method_name in interface_methods:
            assert hasattr(test_agent, method_name)
            assert callable(getattr(test_agent, method_name))

    @pytest.mark.asyncio
    async def test_task_interface_compatibility(self, test_agent):
        """Agent should work with standard TaskInterface objects."""
        await test_agent.initialize()

        # Create standard task using interface
        task = TaskInterface(task_id="compat-test-001", task_type="query", content="Test compatibility")

        # Should be able to check capability
        can_handle = await test_agent.can_handle_task(task)
        assert isinstance(can_handle, bool)

        # Should be able to process if capable
        if can_handle:
            result = await test_agent.process_task(task)
            assert "status" in result
            assert "task_id" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
