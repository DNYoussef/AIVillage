"""Test Data Builders and Fixtures

Provides reusable test data builders following the Builder pattern to reduce
code duplication and maintain consistency across test suites.
Follows connascence principles by providing single sources of truth for test data.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, TypeVar

from packages.agents.core.agent_interface import AgentCapability, AgentMetadata, MessageInterface, TaskInterface
from packages.agents.core.base_agent_template import (
    GeometricSelfState,
    GeometricState,
    QuietStarReflection,
    ReflectionType,
)

T = TypeVar("T")


class TestDataBuilder:
    """Base class for test data builders with fluent interface."""

    def __init__(self):
        self._data = {}
        self._callbacks = []

    def with_callback(self, callback):
        """Add a callback to be executed on build."""
        self._callbacks.append(callback)
        return self

    def _execute_callbacks(self, instance):
        """Execute all registered callbacks with the built instance."""
        for callback in self._callbacks:
            callback(instance)
        return instance


class AgentMetadataBuilder(TestDataBuilder):
    """Builder for creating AgentMetadata test instances."""

    def __init__(self):
        super().__init__()
        self._agent_id = f"test-agent-{uuid.uuid4()}"
        self._agent_type = "TestAgent"
        self._name = "Test Agent"
        self._description = "Test agent for behavioral testing"
        self._version = "1.0.0"
        self._capabilities = {AgentCapability.MESSAGE_PROCESSING, AgentCapability.TASK_EXECUTION}
        self._created_at = datetime.now()
        self._last_updated = datetime.now()
        self._tags = ["test"]
        self._configuration = {}

    def with_id(self, agent_id: str) -> "AgentMetadataBuilder":
        """Set agent ID."""
        self._agent_id = agent_id
        return self

    def with_type(self, agent_type: str) -> "AgentMetadataBuilder":
        """Set agent type."""
        self._agent_type = agent_type
        return self

    def with_name(self, name: str) -> "AgentMetadataBuilder":
        """Set agent name."""
        self._name = name
        return self

    def with_description(self, description: str) -> "AgentMetadataBuilder":
        """Set agent description."""
        self._description = description
        return self

    def with_version(self, version: str) -> "AgentMetadataBuilder":
        """Set agent version."""
        self._version = version
        return self

    def with_capabilities(self, *capabilities: AgentCapability) -> "AgentMetadataBuilder":
        """Set agent capabilities."""
        self._capabilities = set(capabilities)
        return self

    def add_capability(self, capability: AgentCapability) -> "AgentMetadataBuilder":
        """Add a single capability."""
        self._capabilities.add(capability)
        return self

    def with_tags(self, *tags: str) -> "AgentMetadataBuilder":
        """Set agent tags."""
        self._tags = list(tags)
        return self

    def add_tag(self, tag: str) -> "AgentMetadataBuilder":
        """Add a single tag."""
        self._tags.append(tag)
        return self

    def with_configuration(self, **config) -> "AgentMetadataBuilder":
        """Set agent configuration."""
        self._configuration = config
        return self

    def with_timestamps(self, created_at: datetime, last_updated: datetime) -> "AgentMetadataBuilder":
        """Set creation and update timestamps."""
        self._created_at = created_at
        self._last_updated = last_updated
        return self

    def build(self) -> AgentMetadata:
        """Build the AgentMetadata instance."""
        metadata = AgentMetadata(
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            name=self._name,
            description=self._description,
            version=self._version,
            capabilities=self._capabilities.copy(),
            created_at=self._created_at,
            last_updated=self._last_updated,
            tags=self._tags.copy(),
            configuration=self._configuration.copy(),
        )
        return self._execute_callbacks(metadata)


class TaskInterfaceBuilder(TestDataBuilder):
    """Builder for creating TaskInterface test instances."""

    def __init__(self):
        super().__init__()
        self._task_id = f"test-task-{uuid.uuid4()}"
        self._task_type = "test"
        self._content = "Test task content"
        self._priority = 0
        self._timeout_seconds = None
        self._context = {}
        self._metadata = {}
        self._created_at = datetime.now()

    def with_id(self, task_id: str) -> "TaskInterfaceBuilder":
        """Set task ID."""
        self._task_id = task_id
        return self

    def with_type(self, task_type: str) -> "TaskInterfaceBuilder":
        """Set task type."""
        self._task_type = task_type
        return self

    def with_content(self, content: Any) -> "TaskInterfaceBuilder":
        """Set task content."""
        self._content = content
        return self

    def with_priority(self, priority: int) -> "TaskInterfaceBuilder":
        """Set task priority."""
        self._priority = priority
        return self

    def with_timeout(self, timeout_seconds: float) -> "TaskInterfaceBuilder":
        """Set task timeout."""
        self._timeout_seconds = timeout_seconds
        return self

    def with_context(self, **context) -> "TaskInterfaceBuilder":
        """Set task context."""
        self._context = context
        return self

    def add_context(self, key: str, value: Any) -> "TaskInterfaceBuilder":
        """Add single context item."""
        self._context[key] = value
        return self

    def with_metadata(self, **metadata) -> "TaskInterfaceBuilder":
        """Set task metadata."""
        self._metadata = metadata
        return self

    def add_metadata(self, key: str, value: Any) -> "TaskInterfaceBuilder":
        """Add single metadata item."""
        self._metadata[key] = value
        return self

    def with_created_at(self, created_at: datetime) -> "TaskInterfaceBuilder":
        """Set creation timestamp."""
        self._created_at = created_at
        return self

    def build(self) -> TaskInterface:
        """Build the TaskInterface instance."""
        task = TaskInterface(
            task_id=self._task_id,
            task_type=self._task_type,
            content=self._content,
            priority=self._priority,
            timeout_seconds=self._timeout_seconds,
            context=self._context.copy(),
            metadata=self._metadata.copy(),
            created_at=self._created_at,
        )
        return self._execute_callbacks(task)


class MessageInterfaceBuilder(TestDataBuilder):
    """Builder for creating MessageInterface test instances."""

    def __init__(self):
        super().__init__()
        self._message_id = f"test-msg-{uuid.uuid4()}"
        self._sender = "test-sender"
        self._receiver = "test-receiver"
        self._message_type = "test"
        self._content = "Test message content"
        self._priority = 0
        self._expires_at = None
        self._context = {}
        self._metadata = {}
        self._created_at = datetime.now()

    def with_id(self, message_id: str) -> "MessageInterfaceBuilder":
        """Set message ID."""
        self._message_id = message_id
        return self

    def with_sender(self, sender: str) -> "MessageInterfaceBuilder":
        """Set message sender."""
        self._sender = sender
        return self

    def with_receiver(self, receiver: str) -> "MessageInterfaceBuilder":
        """Set message receiver."""
        self._receiver = receiver
        return self

    def with_type(self, message_type: str) -> "MessageInterfaceBuilder":
        """Set message type."""
        self._message_type = message_type
        return self

    def with_content(self, content: Any) -> "MessageInterfaceBuilder":
        """Set message content."""
        self._content = content
        return self

    def with_priority(self, priority: int) -> "MessageInterfaceBuilder":
        """Set message priority."""
        self._priority = priority
        return self

    def with_expiration(self, expires_at: datetime) -> "MessageInterfaceBuilder":
        """Set message expiration."""
        self._expires_at = expires_at
        return self

    def expires_in(self, seconds: int) -> "MessageInterfaceBuilder":
        """Set message to expire in specified seconds."""
        self._expires_at = datetime.now() + timedelta(seconds=seconds)
        return self

    def with_context(self, **context) -> "MessageInterfaceBuilder":
        """Set message context."""
        self._context = context
        return self

    def with_metadata(self, **metadata) -> "MessageInterfaceBuilder":
        """Set message metadata."""
        self._metadata = metadata
        return self

    def build(self) -> MessageInterface:
        """Build the MessageInterface instance."""
        message = MessageInterface(
            message_id=self._message_id,
            sender=self._sender,
            receiver=self._receiver,
            message_type=self._message_type,
            content=self._content,
            priority=self._priority,
            expires_at=self._expires_at,
            context=self._context.copy(),
            metadata=self._metadata.copy(),
            created_at=self._created_at,
        )
        return self._execute_callbacks(message)


class QuietStarReflectionBuilder(TestDataBuilder):
    """Builder for creating QuietStarReflection test instances."""

    def __init__(self):
        super().__init__()
        self._reflection_id = f"reflection-{uuid.uuid4()}"
        self._timestamp = datetime.now()
        self._reflection_type = ReflectionType.TASK_COMPLETION
        self._context = "Test reflection context"
        self._thoughts = "<|startofthought|>Test thoughts<|endofthought|>"
        self._insights = "Test insights gained"
        self._emotional_valence = 0.0
        self._unexpectedness_score = 0.5
        self._tags = ["test"]

    def with_id(self, reflection_id: str) -> "QuietStarReflectionBuilder":
        """Set reflection ID."""
        self._reflection_id = reflection_id
        return self

    def with_type(self, reflection_type: ReflectionType) -> "QuietStarReflectionBuilder":
        """Set reflection type."""
        self._reflection_type = reflection_type
        return self

    def with_context(self, context: str) -> "QuietStarReflectionBuilder":
        """Set reflection context."""
        self._context = context
        return self

    def with_thoughts(self, thoughts: str) -> "QuietStarReflectionBuilder":
        """Set reflection thoughts (will be wrapped in thought tags)."""
        self._thoughts = f"<|startofthought|>{thoughts}<|endofthought|>"
        return self

    def with_raw_thoughts(self, thoughts: str) -> "QuietStarReflectionBuilder":
        """Set raw thoughts without wrapping."""
        self._thoughts = thoughts
        return self

    def with_insights(self, insights: str) -> "QuietStarReflectionBuilder":
        """Set reflection insights."""
        self._insights = insights
        return self

    def with_emotional_valence(self, valence: float) -> "QuietStarReflectionBuilder":
        """Set emotional valence (-1.0 to 1.0)."""
        self._emotional_valence = max(-1.0, min(1.0, valence))
        return self

    def positive_emotion(self) -> "QuietStarReflectionBuilder":
        """Set positive emotional valence."""
        self._emotional_valence = random.uniform(0.3, 1.0)
        return self

    def negative_emotion(self) -> "QuietStarReflectionBuilder":
        """Set negative emotional valence."""
        self._emotional_valence = random.uniform(-1.0, -0.3)
        return self

    def neutral_emotion(self) -> "QuietStarReflectionBuilder":
        """Set neutral emotional valence."""
        self._emotional_valence = random.uniform(-0.2, 0.2)
        return self

    def with_unexpectedness(self, score: float) -> "QuietStarReflectionBuilder":
        """Set unexpectedness score (0.0 to 1.0)."""
        self._unexpectedness_score = max(0.0, min(1.0, score))
        return self

    def highly_unexpected(self) -> "QuietStarReflectionBuilder":
        """Set high unexpectedness score."""
        self._unexpectedness_score = random.uniform(0.8, 1.0)
        return self

    def routine(self) -> "QuietStarReflectionBuilder":
        """Set low unexpectedness score for routine events."""
        self._unexpectedness_score = random.uniform(0.0, 0.2)
        return self

    def with_tags(self, *tags: str) -> "QuietStarReflectionBuilder":
        """Set reflection tags."""
        self._tags = list(tags)
        return self

    def add_tag(self, tag: str) -> "QuietStarReflectionBuilder":
        """Add a single tag."""
        self._tags.append(tag)
        return self

    def with_timestamp(self, timestamp: datetime) -> "QuietStarReflectionBuilder":
        """Set reflection timestamp."""
        self._timestamp = timestamp
        return self

    def build(self) -> QuietStarReflection:
        """Build the QuietStarReflection instance."""
        reflection = QuietStarReflection(
            reflection_id=self._reflection_id,
            timestamp=self._timestamp,
            reflection_type=self._reflection_type,
            context=self._context,
            thoughts=self._thoughts,
            insights=self._insights,
            emotional_valence=self._emotional_valence,
            unexpectedness_score=self._unexpectedness_score,
            tags=self._tags.copy(),
        )
        return self._execute_callbacks(reflection)


class GeometricSelfStateBuilder(TestDataBuilder):
    """Builder for creating GeometricSelfState test instances."""

    def __init__(self):
        super().__init__()
        self._timestamp = datetime.now()
        self._geometric_state = GeometricState.BALANCED
        self._cpu_utilization = 0.5
        self._memory_utilization = 0.5
        self._network_activity = 0.3
        self._task_queue_depth = 2
        self._response_latency_ms = 100.0
        self._accuracy_score = 0.95
        self._energy_efficiency = 0.8
        self._adaptation_rate = 0.1
        self._stability_score = 0.9
        self._optimization_direction = "balanced"

    def with_state(self, state: GeometricState) -> "GeometricSelfStateBuilder":
        """Set geometric state."""
        self._geometric_state = state
        return self

    def balanced(self) -> "GeometricSelfStateBuilder":
        """Configure as balanced state."""
        self._geometric_state = GeometricState.BALANCED
        self._cpu_utilization = random.uniform(0.3, 0.7)
        self._memory_utilization = random.uniform(0.3, 0.7)
        self._stability_score = random.uniform(0.7, 1.0)
        return self

    def overloaded(self) -> "GeometricSelfStateBuilder":
        """Configure as overloaded state."""
        self._geometric_state = GeometricState.OVERLOADED
        self._cpu_utilization = random.uniform(0.8, 1.0)
        self._memory_utilization = random.uniform(0.8, 1.0)
        self._response_latency_ms = random.uniform(2000, 8000)
        self._stability_score = random.uniform(0.2, 0.6)
        return self

    def underutilized(self) -> "GeometricSelfStateBuilder":
        """Configure as underutilized state."""
        self._geometric_state = GeometricState.UNDERUTILIZED
        self._cpu_utilization = random.uniform(0.0, 0.2)
        self._memory_utilization = random.uniform(0.0, 0.3)
        self._task_queue_depth = 0
        return self

    def adapting(self) -> "GeometricSelfStateBuilder":
        """Configure as adapting state."""
        self._geometric_state = GeometricState.ADAPTING
        self._adaptation_rate = random.uniform(0.3, 0.8)
        self._stability_score = random.uniform(0.3, 0.7)
        self._optimization_direction = random.choice(["accuracy", "efficiency", "responsiveness"])
        return self

    def with_resource_utilization(
        self, cpu: float, memory: float, network: float = None
    ) -> "GeometricSelfStateBuilder":
        """Set resource utilization values."""
        self._cpu_utilization = max(0.0, min(1.0, cpu))
        self._memory_utilization = max(0.0, min(1.0, memory))
        if network is not None:
            self._network_activity = max(0.0, min(1.0, network))
        return self

    def with_performance_metrics(
        self, latency: float, accuracy: float, efficiency: float = None
    ) -> "GeometricSelfStateBuilder":
        """Set performance metrics."""
        self._response_latency_ms = max(0.0, latency)
        self._accuracy_score = max(0.0, min(1.0, accuracy))
        if efficiency is not None:
            self._energy_efficiency = max(0.0, min(1.0, efficiency))
        return self

    def with_adaptation_metrics(
        self, rate: float, stability: float, direction: str = None
    ) -> "GeometricSelfStateBuilder":
        """Set adaptation metrics."""
        self._adaptation_rate = max(0.0, min(1.0, rate))
        self._stability_score = max(0.0, min(1.0, stability))
        if direction:
            self._optimization_direction = direction
        return self

    def with_queue_depth(self, depth: int) -> "GeometricSelfStateBuilder":
        """Set task queue depth."""
        self._task_queue_depth = max(0, depth)
        return self

    def healthy(self) -> "GeometricSelfStateBuilder":
        """Configure as healthy state that passes health check."""
        self._cpu_utilization = random.uniform(0.2, 0.8)
        self._memory_utilization = random.uniform(0.2, 0.8)
        self._response_latency_ms = random.uniform(50, 4000)
        self._accuracy_score = random.uniform(0.75, 1.0)
        self._stability_score = random.uniform(0.6, 1.0)
        return self

    def unhealthy(self) -> "GeometricSelfStateBuilder":
        """Configure as unhealthy state that fails health check."""
        unhealthy_aspect = random.choice(["cpu", "memory", "latency", "accuracy", "stability"])

        if unhealthy_aspect == "cpu":
            self._cpu_utilization = random.uniform(0.9, 1.0)
        elif unhealthy_aspect == "memory":
            self._memory_utilization = random.uniform(0.9, 1.0)
        elif unhealthy_aspect == "latency":
            self._response_latency_ms = random.uniform(5000, 15000)
        elif unhealthy_aspect == "accuracy":
            self._accuracy_score = random.uniform(0.0, 0.6)
        elif unhealthy_aspect == "stability":
            self._stability_score = random.uniform(0.0, 0.4)

        return self

    def build(self) -> GeometricSelfState:
        """Build the GeometricSelfState instance."""
        state = GeometricSelfState(
            timestamp=self._timestamp,
            geometric_state=self._geometric_state,
            cpu_utilization=self._cpu_utilization,
            memory_utilization=self._memory_utilization,
            network_activity=self._network_activity,
            task_queue_depth=self._task_queue_depth,
            response_latency_ms=self._response_latency_ms,
            accuracy_score=self._accuracy_score,
            energy_efficiency=self._energy_efficiency,
            adaptation_rate=self._adaptation_rate,
            stability_score=self._stability_score,
            optimization_direction=self._optimization_direction,
        )
        return self._execute_callbacks(state)


class TestScenarioBuilder:
    """Builder for creating complete test scenarios with multiple related objects."""

    def __init__(self):
        self._scenario_name = "test_scenario"
        self._agent_count = 1
        self._task_count = 3
        self._message_count = 2
        self._reflection_count = 2
        self._agent_types = ["TestAgent"]
        self._task_types = ["test", "query", "process"]
        self._message_types = ["request", "response", "notification"]

    def named(self, name: str) -> "TestScenarioBuilder":
        """Set scenario name."""
        self._scenario_name = name
        return self

    def with_agents(self, count: int, agent_types: list[str] = None) -> "TestScenarioBuilder":
        """Set number of agents and types."""
        self._agent_count = count
        if agent_types:
            self._agent_types = agent_types
        return self

    def with_tasks(self, count: int, task_types: list[str] = None) -> "TestScenarioBuilder":
        """Set number of tasks and types."""
        self._task_count = count
        if task_types:
            self._task_types = task_types
        return self

    def with_messages(self, count: int, message_types: list[str] = None) -> "TestScenarioBuilder":
        """Set number of messages and types."""
        self._message_count = count
        if message_types:
            self._message_types = message_types
        return self

    def with_reflections(self, count: int) -> "TestScenarioBuilder":
        """Set number of reflections."""
        self._reflection_count = count
        return self

    def build(self) -> dict[str, Any]:
        """Build complete test scenario."""
        scenario = {
            "name": self._scenario_name,
            "agents": [],
            "tasks": [],
            "messages": [],
            "reflections": [],
        }

        # Build agents
        for i in range(self._agent_count):
            agent_type = self._agent_types[i % len(self._agent_types)]
            metadata = (
                AgentMetadataBuilder()
                .with_type(agent_type)
                .with_name(f"{agent_type}_{i}")
                .add_tag(self._scenario_name)
                .build()
            )
            scenario["agents"].append(metadata)

        # Build tasks
        for i in range(self._task_count):
            task_type = self._task_types[i % len(self._task_types)]
            task = (
                TaskInterfaceBuilder()
                .with_type(task_type)
                .with_content(f"{self._scenario_name} task {i}")
                .add_context("scenario", self._scenario_name)
                .build()
            )
            scenario["tasks"].append(task)

        # Build messages
        if self._agent_count > 1:
            for i in range(self._message_count):
                message_type = self._message_types[i % len(self._message_types)]
                sender_idx = i % self._agent_count
                receiver_idx = (i + 1) % self._agent_count

                message = (
                    MessageInterfaceBuilder()
                    .with_type(message_type)
                    .with_sender(scenario["agents"][sender_idx].agent_id)
                    .with_receiver(scenario["agents"][receiver_idx].agent_id)
                    .with_content(f"{self._scenario_name} message {i}")
                    .add_context("scenario", self._scenario_name)
                    .build()
                )
                scenario["messages"].append(message)

        # Build reflections
        reflection_types = list(ReflectionType)
        for i in range(self._reflection_count):
            reflection_type = reflection_types[i % len(reflection_types)]
            reflection = (
                QuietStarReflectionBuilder()
                .with_type(reflection_type)
                .with_context(f"{self._scenario_name} reflection context {i}")
                .with_thoughts(f"Thinking about {self._scenario_name} scenario")
                .with_insights(f"Learned from {self._scenario_name} step {i}")
                .add_tag(self._scenario_name)
                .build()
            )
            scenario["reflections"].append(reflection)

        return scenario


# Convenience factory functions following single responsibility principle


def create_test_agent_metadata(**overrides) -> AgentMetadata:
    """Create test agent metadata with optional overrides."""
    builder = AgentMetadataBuilder()
    for key, value in overrides.items():
        if hasattr(builder, f"with_{key}"):
            getattr(builder, f"with_{key}")(value)
    return builder.build()


def create_test_task(**overrides) -> TaskInterface:
    """Create test task with optional overrides."""
    builder = TaskInterfaceBuilder()
    for key, value in overrides.items():
        if hasattr(builder, f"with_{key}"):
            getattr(builder, f"with_{key}")(value)
    return builder.build()


def create_test_message(**overrides) -> MessageInterface:
    """Create test message with optional overrides."""
    builder = MessageInterfaceBuilder()
    for key, value in overrides.items():
        if hasattr(builder, f"with_{key}"):
            getattr(builder, f"with_{key}")(value)
    return builder.build()


def create_test_reflection(**overrides) -> QuietStarReflection:
    """Create test reflection with optional overrides."""
    builder = QuietStarReflectionBuilder()
    for key, value in overrides.items():
        if hasattr(builder, f"with_{key}"):
            getattr(builder, f"with_{key}")(value)
    return builder.build()


def create_geometric_state(**overrides) -> GeometricSelfState:
    """Create test geometric state with optional overrides."""
    builder = GeometricSelfStateBuilder()
    for key, value in overrides.items():
        if hasattr(builder, f"with_{key}"):
            getattr(builder, f"with_{key}")(value)
    return builder.build()


def create_test_scenario(name: str, **config) -> dict[str, Any]:
    """Create test scenario with configuration."""
    builder = TestScenarioBuilder().named(name)

    if "agent_count" in config:
        builder.with_agents(config["agent_count"], config.get("agent_types"))
    if "task_count" in config:
        builder.with_tasks(config["task_count"], config.get("task_types"))
    if "message_count" in config:
        builder.with_messages(config["message_count"], config.get("message_types"))
    if "reflection_count" in config:
        builder.with_reflections(config["reflection_count"])

    return builder.build()


# Pre-configured builders for common test patterns


def quick_agent() -> AgentMetadataBuilder:
    """Quick agent builder with sensible defaults for rapid testing."""
    return (
        AgentMetadataBuilder()
        .with_capabilities(
            AgentCapability.MESSAGE_PROCESSING, AgentCapability.TASK_EXECUTION, AgentCapability.TEXT_PROCESSING
        )
        .add_tag("quick_test")
    )


def complex_agent() -> AgentMetadataBuilder:
    """Complex agent builder with full capabilities for integration testing."""
    return (
        AgentMetadataBuilder().with_capabilities(*list(AgentCapability)).add_tag("integration_test").add_tag("complex")
    )


def priority_task(priority: int = 5) -> TaskInterfaceBuilder:
    """Task builder with specified priority."""
    return TaskInterfaceBuilder().with_priority(priority)


def urgent_message() -> MessageInterfaceBuilder:
    """Message builder for urgent communications."""
    return MessageInterfaceBuilder().with_type("urgent").with_priority(10).expires_in(60)  # Expires in 1 minute


def learning_reflection() -> QuietStarReflectionBuilder:
    """Reflection builder for learning scenarios."""
    return (
        QuietStarReflectionBuilder()
        .with_type(ReflectionType.LEARNING)
        .positive_emotion()
        .highly_unexpected()
        .add_tag("learning")
        .add_tag("growth")
    )


def error_reflection() -> QuietStarReflectionBuilder:
    """Reflection builder for error scenarios."""
    return (
        QuietStarReflectionBuilder()
        .with_type(ReflectionType.ERROR_ANALYSIS)
        .negative_emotion()
        .highly_unexpected()
        .add_tag("error")
        .add_tag("analysis")
    )


def healthy_state() -> GeometricSelfStateBuilder:
    """Geometric state builder for healthy system state."""
    return GeometricSelfStateBuilder().healthy().balanced()


def stressed_state() -> GeometricSelfStateBuilder:
    """Geometric state builder for stressed system state."""
    return GeometricSelfStateBuilder().unhealthy().overloaded()


# Export all builders and factory functions
__all__ = [
    "TestDataBuilder",
    "AgentMetadataBuilder",
    "TaskInterfaceBuilder",
    "MessageInterfaceBuilder",
    "QuietStarReflectionBuilder",
    "GeometricSelfStateBuilder",
    "TestScenarioBuilder",
    "create_test_agent_metadata",
    "create_test_task",
    "create_test_message",
    "create_test_reflection",
    "create_geometric_state",
    "create_test_scenario",
    "quick_agent",
    "complex_agent",
    "priority_task",
    "urgent_message",
    "learning_reflection",
    "error_reflection",
    "healthy_state",
    "stressed_state",
]
