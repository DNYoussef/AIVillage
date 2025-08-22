"""Component isolation tests for agent refactoring.

These tests verify that each component can work independently and
maintains clean boundaries (low coupling, high cohesion).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.agents.core.components import (
    AgentCapabilities,
    AgentCommunication,
    AgentConfiguration,
    AgentMetrics,
    AgentStateManager,
)
from packages.agents.core.components.capabilities import CapabilityLevel, MCPTool
from packages.agents.core.components.communication import CommunicationConfig
from packages.agents.core.components.configuration import ConfigurationLevel
from packages.agents.core.components.metrics import MetricsConfig, MetricType
from packages.agents.core.components.state_manager import AgentState, StateConfig


class MockMCPTool(MCPTool):
    """Mock MCP tool for testing."""

    def __init__(self, name: str = "mock_tool"):
        super().__init__(name, f"Mock tool: {name}")
        self.execution_count = 0

    async def execute(self, parameters: dict) -> dict:
        self.execution_count += 1
        return {"status": "success", "result": f"Executed with {parameters}"}


class TestAgentCommunication:
    """Test AgentCommunication component in isolation."""

    @pytest.fixture
    def communication(self):
        """Create communication component."""
        return AgentCommunication("test-agent", CommunicationConfig())

    def test_communication_initializes_with_correct_identity(self, communication):
        """Communication should initialize with correct agent identity."""
        assert communication.agent_id == "test-agent"
        assert communication.config is not None
        assert communication.metrics is not None

    @pytest.mark.asyncio
    async def test_direct_messaging_without_client_returns_error(self, communication):
        """Direct messaging without P2P client should return error response."""
        result = await communication.send_direct_message("recipient", "test message")

        assert result["status"] == "error"
        assert "P2P client not available" in result["message"]

    @pytest.mark.asyncio
    async def test_direct_messaging_with_mock_client_succeeds(self, communication):
        """Direct messaging with mock client should succeed."""
        mock_p2p = AsyncMock()
        mock_tool = AsyncMock()
        mock_tool.execute = AsyncMock(return_value={"status": "success", "message_id": "123"})

        communication.inject_dependencies(mock_p2p, {"communicate": mock_tool})

        result = await communication.send_direct_message("recipient", "test message")

        assert result["status"] == "success"
        mock_tool.execute.assert_called_once()

    def test_group_channel_management(self, communication):
        """Group channel management should work independently."""
        # Join channel
        join_result = asyncio.run(communication.join_group_channel("test-channel"))
        assert join_result is True

        # Verify channel status
        status = communication.get_channel_status()
        assert "test-channel" in status["channels"]["group_memberships"]

        # Leave channel
        leave_result = asyncio.run(communication.leave_group_channel("test-channel"))
        assert leave_result is True

        # Verify channel removed
        status = communication.get_channel_status()
        assert "test-channel" not in status["channels"]["group_memberships"]

    def test_metrics_tracking_updates_counters(self, communication):
        """Metrics tracking should update internal counters."""
        initial_metrics = communication.get_communication_metrics()
        initial_sent = initial_metrics["messages_sent"]

        # Simulate message sending (internal metric update)
        communication._update_metrics(sent=True, latency_ms=100)

        updated_metrics = communication.get_communication_metrics()
        assert updated_metrics["messages_sent"] == initial_sent + 1
        assert updated_metrics["average_latency_ms"] > 0


class TestAgentConfiguration:
    """Test AgentConfiguration component in isolation."""

    @pytest.fixture
    def config(self):
        """Create configuration component."""
        return AgentConfiguration("test-agent", "TestAgent")

    def test_configuration_initializes_with_defaults(self, config):
        """Configuration should initialize with sensible defaults."""
        core_config = config.get_core_config()

        assert core_config.agent_id == "test-agent"
        assert core_config.agent_type == "TestAgent"
        assert core_config.max_concurrent_tasks > 0
        assert core_config.geometric_awareness_enabled is not None

    def test_configuration_precedence_levels_work(self, config):
        """Configuration precedence levels should be respected."""
        # Set with default level
        config.set_configuration("test_key", "default_value", ConfigurationLevel.DEFAULT)

        # Override with higher precedence
        config.set_configuration("test_key", "override_value", ConfigurationLevel.OVERRIDE)

        assert config.get_configuration("test_key") == "override_value"

        # Try to set with lower precedence - should not change
        config.set_configuration("test_key", "lower_value", ConfigurationLevel.FILE)

        assert config.get_configuration("test_key") == "override_value"

    def test_dependency_injection_works(self, config):
        """Dependency injection should store and retrieve clients."""
        mock_client = MagicMock()

        config.inject_client("test_client", mock_client)

        assert config.has_client("test_client")
        assert config.get_client("test_client") is mock_client

    def test_configuration_validation_catches_errors(self, config):
        """Configuration validation should catch invalid values."""
        # Add validation rule
        config.add_validation_rule("positive", lambda x: x > 0)

        # Set valid value
        result = config.set_configuration("valid_key", 10, validation_rule="positive")
        assert result is True

        # Try invalid value
        result = config.set_configuration("invalid_key", -5, validation_rule="positive")
        assert result is False

    def test_configuration_export_import_cycle(self, config):
        """Configuration should export and import correctly."""
        # Set some configuration
        config.set_configuration("export_test", "test_value")
        config.set_configuration("export_number", 42)

        # Export
        exported = config.export_configuration()

        # Create new config and import
        new_config = AgentConfiguration("new-agent", "NewAgent")
        import_result = new_config.import_configuration(exported)

        assert import_result["imported"] > 0
        assert new_config.get_configuration("export_test") == "test_value"
        assert new_config.get_configuration("export_number") == 42


class TestAgentCapabilities:
    """Test AgentCapabilities component in isolation."""

    @pytest.fixture
    def capabilities(self):
        """Create capabilities component."""
        return AgentCapabilities("test-agent", "TestAgent")

    def test_capabilities_initializes_correctly(self, capabilities):
        """Capabilities should initialize with correct identity."""
        assert capabilities.agent_id == "test-agent"
        assert capabilities.agent_type == "TestAgent"
        assert capabilities.get_specialized_role() == "base_template"

    def test_capability_management_lifecycle(self, capabilities):
        """Complete capability management lifecycle should work."""
        # Add capability
        result = capabilities.add_capability(
            "test_cap", "Test Capability", "Test description", CapabilityLevel.INTERMEDIATE
        )
        assert result is True

        # Check capability exists
        assert capabilities.has_capability("test_cap")
        assert capabilities.has_capability("test_cap", CapabilityLevel.BASIC)
        assert not capabilities.has_capability("test_cap", CapabilityLevel.EXPERT)

        # Upgrade capability
        upgrade_result = capabilities.upgrade_capability("test_cap", CapabilityLevel.ADVANCED)
        assert upgrade_result is True
        assert capabilities.has_capability("test_cap", CapabilityLevel.ADVANCED)

        # Remove capability
        remove_result = capabilities.remove_capability("test_cap")
        assert remove_result is True
        assert not capabilities.has_capability("test_cap")

    def test_capability_dependency_checking(self, capabilities):
        """Capability dependencies should be enforced."""
        # Add base capability
        capabilities.add_capability("base_cap", "Base", "Base capability")

        # Add dependent capability
        result = capabilities.add_capability("dependent_cap", "Dependent", "Depends on base", dependencies=["base_cap"])
        assert result is True

        # Try to add capability with missing dependency
        result = capabilities.add_capability("broken_cap", "Broken", "Missing dependency", dependencies=["missing_cap"])
        assert result is False

    @pytest.mark.asyncio
    async def test_tool_registration_and_execution(self, capabilities):
        """Tool registration and execution should work independently."""
        mock_tool = MockMCPTool("test_tool")

        # Register tool
        result = capabilities.register_tool(mock_tool)
        assert result is True

        # Verify tool is available
        available_tools = capabilities.get_available_tools()
        assert "test_tool" in available_tools

        # Execute tool
        execution_result = await capabilities.execute_tool("test_tool", {"param": "value"})
        assert execution_result["status"] == "success"
        assert mock_tool.execution_count == 1

    def test_task_type_capability_mapping(self, capabilities):
        """Task type to capability mapping should work correctly."""
        # Add relevant capabilities
        capabilities.add_capability("query_processing", "Query Processing", "Process queries")
        capabilities.add_capability("reasoning", "Reasoning", "Logical reasoning")

        # Test task type checking
        assert capabilities.can_handle_task_type("query")  # Has query_processing
        assert capabilities.can_handle_task_type("analysis")  # Has reasoning
        assert capabilities.can_handle_task_type("unknown_type")  # Unknown types allowed by default

    def test_skill_proficiency_management(self, capabilities):
        """Skill proficiency management should work correctly."""
        # Set skill proficiencies
        capabilities.set_skill_proficiency("python", 0.9)
        capabilities.set_skill_proficiency("communication", 0.7)

        # Retrieve proficiencies
        assert capabilities.get_skill_proficiency("python") == 0.9
        assert capabilities.get_skill_proficiency("communication") == 0.7
        assert capabilities.get_skill_proficiency("unknown_skill") == 0.0


class TestAgentMetrics:
    """Test AgentMetrics component in isolation."""

    @pytest.fixture
    def metrics(self):
        """Create metrics component."""
        return AgentMetrics("test-agent", MetricsConfig())

    def test_metrics_initializes_correctly(self, metrics):
        """Metrics should initialize with correct configuration."""
        assert metrics.agent_id == "test-agent"
        assert metrics.config is not None

        # Should have initial metrics
        current_metrics = metrics.get_current_metrics()
        assert current_metrics["agent_id"] == "test-agent"
        assert "timestamp" in current_metrics

    def test_metric_recording_and_retrieval(self, metrics):
        """Metric recording and retrieval should work correctly."""
        # Record a metric
        metrics.record_metric(MetricType.PERFORMANCE, "test_metric", 42.0, "units", ["tag1", "tag2"])

        # Retrieve metric history
        history = metrics.get_metric_history(metric_name="test_metric")
        assert len(history) == 1
        assert history[0].value == 42.0
        assert history[0].unit == "units"
        assert "tag1" in history[0].tags

    def test_task_completion_recording(self, metrics):
        """Task completion recording should update all relevant metrics."""
        initial_metrics = metrics.get_current_metrics()
        initial_completed = initial_metrics["task_statistics"]["completed_tasks"]

        # Record successful task
        metrics.record_task_completion("task-001", 150.0, True, 0.95)

        updated_metrics = metrics.get_current_metrics()
        assert updated_metrics["task_statistics"]["completed_tasks"] == initial_completed + 1

        # Record failed task
        metrics.record_task_completion("task-002", 200.0, False, 0.0)

        final_metrics = metrics.get_current_metrics()
        assert final_metrics["task_statistics"]["failed_tasks"] > 0
        assert final_metrics["task_statistics"]["success_rate"] < 1.0

    def test_performance_trend_analysis(self, metrics):
        """Performance trend analysis should provide insights."""
        # Record several performance snapshots
        for i in range(5):
            metrics.update_performance_snapshot(
                cpu_utilization=0.5 + i * 0.1, memory_utilization=0.3  # Increasing trend
            )

        # Analyze trends
        trends = metrics.get_performance_trends(hours=1)

        assert "cpu_utilization" in trends
        assert trends["cpu_utilization"]["trend"] in ["increasing", "decreasing", "stable"]
        assert trends["cpu_utilization"]["max"] >= trends["cpu_utilization"]["min"]

    def test_metric_cleanup_removes_old_entries(self, metrics):
        """Metric cleanup should remove entries older than retention period."""
        # Record metrics with short retention
        metrics.config.metric_retention_hours = 0  # Immediate cleanup

        metrics.record_metric(MetricType.CUSTOM, "old_metric", 1.0)
        initial_count = len(metrics._metrics)

        # Cleanup should remove all metrics
        removed_count = metrics.cleanup_old_metrics()

        assert removed_count == initial_count
        assert len(metrics._metrics) == 0

    def test_metrics_export_formats(self, metrics):
        """Metrics export should support different formats."""
        # Record some test data
        metrics.record_metric(MetricType.PERFORMANCE, "export_test", 123.0)
        metrics.record_task_completion("export-task", 100.0, True)

        # Test summary export
        summary = metrics.export_metrics("summary")
        assert "summary" in summary
        assert "trends" in summary
        assert "export_timestamp" in summary

        # Test JSON export
        json_export = metrics.export_metrics("json")
        assert "metrics" in json_export
        assert "performance_history" in json_export

        # Test invalid format
        with pytest.raises(ValueError):
            metrics.export_metrics("invalid_format")


class TestAgentStateManager:
    """Test AgentStateManager component in isolation."""

    @pytest.fixture
    def state_manager(self):
        """Create state manager component."""
        return AgentStateManager("test-agent", StateConfig())

    def test_state_manager_initializes_correctly(self, state_manager):
        """State manager should initialize with correct defaults."""
        assert state_manager.agent_id == "test-agent"
        assert state_manager.get_current_state() == AgentState.INITIALIZING

    def test_state_transitions_are_tracked(self, state_manager):
        """State transitions should be properly tracked."""
        initial_state = state_manager.get_current_state()

        # Transition to new state
        state_manager.set_state(AgentState.IDLE, "test transition")

        new_state = state_manager.get_current_state()
        assert new_state == AgentState.IDLE
        assert new_state != initial_state

        # State history should be updated
        metrics = state_manager.get_state_metrics()
        assert metrics["state_transitions"] > 0

    @pytest.mark.asyncio
    async def test_geometric_awareness_updates(self, state_manager):
        """Geometric awareness should update with task metrics."""
        # Update without task metrics
        geo_state = await state_manager.update_geometric_awareness()
        assert geo_state is not None
        assert geo_state.timestamp is not None

        # Update with task metrics
        task_metrics = {"task_id": "geo-test", "latency_ms": 200.0, "accuracy": 0.85}

        updated_geo_state = await state_manager.update_geometric_awareness(task_metrics)
        assert updated_geo_state.resource_metrics.response_latency_ms > 0
        assert updated_geo_state.resource_metrics.accuracy_score > 0

    def test_task_performance_recording(self, state_manager):
        """Task performance recording should maintain history."""
        initial_metrics = state_manager.get_state_metrics()
        initial_task_count = initial_metrics["recent_performance"]["tasks_completed"]

        # Record task performance
        state_manager.record_task_performance("perf-test", 150.0, 0.9, "success")

        updated_metrics = state_manager.get_state_metrics()
        assert updated_metrics["recent_performance"]["tasks_completed"] > initial_task_count

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, state_manager):
        """Monitoring start and stop should work correctly."""
        # Start monitoring
        await state_manager.start_monitoring()

        metrics = state_manager.get_state_metrics()
        assert metrics["monitoring_active"] is True

        # Stop monitoring
        await state_manager.stop_monitoring()

        # Give it a moment to stop
        await asyncio.sleep(0.1)

        metrics = state_manager.get_state_metrics()
        assert metrics["monitoring_active"] is False

    def test_health_status_reflects_state(self, state_manager):
        """Health status should reflect current state accurately."""
        # Healthy state
        state_manager.set_state(AgentState.IDLE)
        health = state_manager.get_health_status()
        assert health["healthy"] is True

        # Unhealthy state
        state_manager.set_state(AgentState.ERROR)
        health = state_manager.get_health_status()
        assert health["healthy"] is False
        assert "Agent in error state" in health["issues"]


class TestComponentInteraction:
    """Test that components can interact without tight coupling."""

    def test_components_can_share_data_without_coupling(self):
        """Components should be able to share data through clean interfaces."""
        # Create components
        config = AgentConfiguration("interaction-test", "TestAgent")
        metrics = AgentMetrics("interaction-test")
        capabilities = AgentCapabilities("interaction-test", "TestAgent")

        # Configuration provides settings that metrics can use
        config.set_configuration("metric_retention_hours", 48)
        retention_hours = config.get_configuration("metric_retention_hours", 24)

        # Metrics can use this configuration without being coupled to config component
        assert retention_hours == 48

        # Capabilities can report metrics without being coupled to metrics component
        cap_summary = capabilities.get_capability_summary()
        metrics.record_metric(MetricType.CUSTOM, "capability_count", len(cap_summary["capabilities"]))

        # Verify data flow worked
        metric_history = metrics.get_metric_history("capability_count")
        assert len(metric_history) == 1

    def test_components_maintain_independent_state(self):
        """Each component should maintain its own state independently."""
        # Create multiple instances of same component type
        comm1 = AgentCommunication("agent-1")
        comm2 = AgentCommunication("agent-2")

        # Modify state in one component
        asyncio.run(comm1.join_group_channel("channel-1"))

        # Other component should be unaffected
        status1 = comm1.get_channel_status()
        status2 = comm2.get_channel_status()

        assert "channel-1" in status1["channels"]["group_memberships"]
        assert "channel-1" not in status2["channels"]["group_memberships"]

    def test_component_failure_isolation(self):
        """Failure in one component should not affect others."""
        config = AgentConfiguration("failure-test", "TestAgent")
        metrics = AgentMetrics("failure-test")

        # Cause an error in one component
        try:
            config.set_configuration("invalid", None, "invalid_level")  # This should fail
        except:
            pass

        # Other component should still work
        metrics.record_metric(MetricType.PERFORMANCE, "test_metric", 1.0)
        current_metrics = metrics.get_current_metrics()

        assert current_metrics is not None
        assert current_metrics["agent_id"] == "failure-test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
