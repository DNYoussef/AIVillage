"""Pytest Configuration and Shared Fixtures

Provides shared fixtures and configuration for agent testing.
Follows connascence principles by providing single sources of truth for test setup.
"""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
import pytest_asyncio
from test_builders import AgentMetadataBuilder, complex_agent, quick_agent

from packages.agents.core.agent_interface import AgentCapability, AgentMetadata
from packages.agents.core.base_agent_template import BaseAgentTemplate

# Configure pytest-asyncio
pytest_asyncio.auto_mode = True


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_agent_metadata() -> AgentMetadata:
    """Provide sample agent metadata for basic testing."""
    return quick_agent().build()


@pytest.fixture
def complex_agent_metadata() -> AgentMetadata:
    """Provide complex agent metadata for integration testing."""
    return complex_agent().build()


@pytest.fixture
def specialized_agent_metadata() -> AgentMetadata:
    """Provide specialized agent metadata with specific capabilities."""
    return (
        AgentMetadataBuilder()
        .with_type("SpecializedTestAgent")
        .with_name("Specialized Test Agent")
        .with_capabilities(
            AgentCapability.REASONING,
            AgentCapability.PLANNING,
            AgentCapability.KNOWLEDGE_SYNTHESIS,
            AgentCapability.PERFORMANCE_MONITORING,
        )
        .add_tag("specialized")
        .add_tag("test")
        .build()
    )


class MockTestAgent(BaseAgentTemplate):
    """Reusable mock agent for testing."""

    def __init__(self, metadata: AgentMetadata):
        super().__init__(metadata)
        self.specialized_role = "mock_test_agent"
        self.test_responses = {}
        self.call_history = []

    def _record_call(self, method_name: str, *args, **kwargs):
        """Record method calls for verification."""
        self.call_history.append({"method": method_name, "args": args, "kwargs": kwargs, "timestamp": datetime.now()})

    async def get_specialized_capabilities(self) -> list[AgentCapability]:
        self._record_call("get_specialized_capabilities")
        return [AgentCapability.TEXT_PROCESSING, AgentCapability.REASONING]

    async def process_specialized_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        self._record_call("process_specialized_task", task_data)

        # Simulate realistic processing
        content = task_data.get("content", "")
        if not content:
            return {"status": "error", "error": "Empty content"}

        return {
            "status": "success",
            "result": f"Mock processed: {content}",
            "processing_time_ms": 50,
            "task_data": task_data,
        }

    async def get_specialized_mcp_tools(self) -> dict[str, Any]:
        self._record_call("get_specialized_mcp_tools")
        return {}

    # Required AgentInterface implementations
    async def process_task(self, task):
        self._record_call("process_task", task)
        start_time = datetime.now()

        result = await self.process_specialized_task(
            {"content": task.content, "task_type": task.task_type, "task_id": task.task_id}
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self._record_task_performance(
            task.task_id,
            processing_time,
            accuracy=1.0 if result["status"] == "success" else 0.0,
            status=result["status"],
        )

        return result

    async def can_handle_task(self, task):
        self._record_call("can_handle_task", task)
        # Can handle most tasks except "unsupported"
        return task.task_type != "unsupported"

    async def estimate_task_duration(self, task):
        self._record_call("estimate_task_duration", task)
        content_length = len(str(task.content)) if task.content else 0
        return max(0.01, content_length * 0.001)

    async def send_message(self, message):
        self._record_call("send_message", message)
        return message.receiver != "unreachable"

    async def receive_message(self, message):
        self._record_call("receive_message", message)
        self.interaction_history.append(
            {
                "type": "message_received",
                "message_id": message.message_id,
                "sender": message.sender,
                "timestamp": datetime.now().timestamp(),
            }
        )

    async def broadcast_message(self, message, recipients):
        self._record_call("broadcast_message", message, recipients)
        return {recipient: recipient != "unreachable" for recipient in recipients}

    async def generate(self, prompt):
        self._record_call("generate", prompt)
        if not prompt:
            return ""
        return f"Mock generated: {prompt[:50]}..."

    async def get_embedding(self, text):
        self._record_call("get_embedding", text)
        if not text:
            return [0.0] * 384
        # Simple deterministic embedding
        hash_val = hash(text) % 1000
        return [hash_val / 1000.0] * 384

    async def rerank(self, query, results, k):
        self._record_call("rerank", query, results, k)
        return results[: min(k, len(results))]

    async def introspect(self):
        self._record_call("introspect")
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "call_count": len(self.call_history),
            "memory_entries": len(self.personal_memory),
            "journal_entries": len(self.personal_journal),
        }

    async def communicate(self, message, recipient):
        self._record_call("communicate", message, recipient)
        return f"Mock communication: {message}"

    async def activate_latent_space(self, query):
        self._record_call("activate_latent_space", query)
        return "mock_space", f"latent:mock:{query[:20]}"


@pytest.fixture
async def mock_agent(sample_agent_metadata) -> MockTestAgent:
    """Provide initialized mock agent for testing."""
    agent = MockTestAgent(sample_agent_metadata)

    # Mock external dependencies
    agent.rag_client = MagicMock()
    agent.rag_client.query = AsyncMock(
        return_value={"status": "success", "results": ["mock rag result 1", "mock rag result 2"]}
    )

    agent.p2p_client = MagicMock()
    agent.p2p_client.send_message = AsyncMock(return_value={"delivered": True, "message_id": "mock-msg-id"})

    agent.agent_forge_client = MagicMock()
    agent.agent_forge_client.execute_adas_phase = AsyncMock(
        return_value={
            "status": "success",
            "new_architecture": "optimized",
            "modification_summary": "Mock optimization applied",
        }
    )

    await agent.initialize()
    return agent


@pytest.fixture
async def complex_mock_agent(complex_agent_metadata) -> MockTestAgent:
    """Provide complex mock agent with full capabilities."""
    agent = MockTestAgent(complex_agent_metadata)

    # Enhanced mocking for complex scenarios
    agent.rag_client = MagicMock()
    agent.rag_client.query = AsyncMock(
        side_effect=lambda **kwargs: {
            "status": "success",
            "query": kwargs.get("query", ""),
            "mode": kwargs.get("mode", "balanced"),
            "results": [
                {"content": f"Result for: {kwargs.get('query', '')}", "score": 0.9},
                {"content": "Additional context", "score": 0.7},
            ],
        }
    )

    agent.p2p_client = MagicMock()
    agent.p2p_client.send_message = AsyncMock(
        side_effect=lambda **kwargs: {
            "delivered": kwargs.get("recipient") != "unreachable",
            "message_id": f"msg-{uuid.uuid4()}",
            "timestamp": datetime.now().isoformat(),
        }
    )

    await agent.initialize()
    return agent


@pytest.fixture
def mock_external_services():
    """Provide mock external services for testing."""
    return {
        "rag_client": MagicMock(),
        "p2p_client": MagicMock(),
        "agent_forge_client": MagicMock(),
        "database_client": MagicMock(),
    }


@pytest.fixture
async def agent_factory():
    """Provide factory function for creating test agents."""
    created_agents = []

    async def create_agent(
        agent_type: str = "TestAgent", capabilities: list[AgentCapability] = None, **metadata_overrides
    ) -> MockTestAgent:
        """Create and initialize a test agent."""
        if capabilities is None:
            capabilities = [AgentCapability.MESSAGE_PROCESSING, AgentCapability.TASK_EXECUTION]

        metadata = (
            AgentMetadataBuilder()
            .with_type(agent_type)
            .with_name(f"{agent_type} Instance")
            .with_capabilities(*capabilities)
            .add_tag("factory_created")
        )

        # Apply any overrides
        for key, value in metadata_overrides.items():
            if hasattr(metadata, f"with_{key}"):
                getattr(metadata, f"with_{key}")(value)

        agent = MockTestAgent(metadata.build())
        await agent.initialize()
        created_agents.append(agent)
        return agent

    yield create_agent

    # Cleanup: shutdown all created agents
    for agent in created_agents:
        try:
            await agent.shutdown()
        except Exception as e:
            import logging

            logging.exception("Agent shutdown error in tests: %s", str(e))


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities for tests."""

    class PerformanceMonitor:
        def __init__(self):
            self.measurements = []

        def measure_duration(self, operation_name: str):
            """Context manager for measuring operation duration."""
            from contextlib import contextmanager
            import time

            @contextmanager
            def timer():
                start_time = time.perf_counter()
                yield
                end_time = time.perf_counter()
                duration = end_time - start_time
                self.measurements.append(
                    {"operation": operation_name, "duration_seconds": duration, "timestamp": datetime.now()}
                )

            return timer()

        def get_average_duration(self, operation_name: str) -> float:
            """Get average duration for an operation."""
            relevant_measurements = [
                m["duration_seconds"] for m in self.measurements if m["operation"] == operation_name
            ]
            return sum(relevant_measurements) / len(relevant_measurements) if relevant_measurements else 0.0

        def get_measurements(self) -> list[dict[str, Any]]:
            """Get all measurements."""
            return self.measurements.copy()

        def reset(self):
            """Reset all measurements."""
            self.measurements.clear()

    return PerformanceMonitor()


@pytest.fixture
def test_isolation():
    """Provide test isolation utilities."""

    class TestIsolation:
        def __init__(self):
            self.cleanup_functions = []

        def add_cleanup(self, cleanup_func):
            """Add cleanup function to be called after test."""
            self.cleanup_functions.append(cleanup_func)

        def isolate_filesystem(self, temp_dir):
            """Isolate filesystem operations to temp directory."""
            import os

            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            self.add_cleanup(lambda: os.chdir(original_cwd))

        def mock_time(self, fixed_time: datetime):
            """Mock time to fixed value for deterministic tests."""
            from unittest.mock import patch

            patcher = patch("datetime.datetime.now", return_value=fixed_time)
            patcher.start()
            self.add_cleanup(patcher.stop)

        def cleanup(self):
            """Execute all cleanup functions."""
            for cleanup_func in reversed(self.cleanup_functions):
                try:
                    cleanup_func()
                except Exception as e:
                    import logging

                    logging.exception("Cleanup error in tests: %s", str(e))
            self.cleanup_functions.clear()

    isolation = TestIsolation()
    yield isolation
    isolation.cleanup()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "behavioral: marks tests as behavioral contract tests")
    config.addinivalue_line("markers", "property: marks tests as property-based tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add markers based on test location
        if "behavioral" in str(item.fspath):
            item.add_marker(pytest.mark.behavioral)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "properties" in str(item.fspath):
            item.add_marker(pytest.mark.property)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
