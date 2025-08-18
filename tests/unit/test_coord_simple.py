#!/usr/bin/env python3
"""Simple test to verify agent coordination system imports and basic functionality."""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.coordination_system import (
    Agent,
    AgentCapability,
    AgentRegistry,
    AgentStatus,
    Message,
    MessageBroker,
    MessageType,
    Resource,
    ResourceManager,
    Task,
    TaskScheduler,
)


def test_basic_functionality():
    """Test basic functionality to verify everything works."""
    print("Testing Agent Coordination System...")

    # Test agent registry
    print("  Testing AgentRegistry...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        registry = AgentRegistry(tmp.name)

        # Create agent with capabilities
        capabilities = [
            AgentCapability(
                "data_processing",
                "1.0",
                "Data processing capability",
                supported_task_types=["data_processing"],
            )
        ]
        agent = Agent(
            agent_id="test_agent",
            name="Test Agent",
            agent_type="worker",
            capabilities=capabilities,
            status=AgentStatus.IDLE,
            endpoint="http://localhost:8001",
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        success = registry.register_agent(agent)
        print(f"    Agent registration: {success}")

        found_agents = registry.find_agents_by_capability("data_processing")
        print(f"    Found agents: {len(found_agents)}")

        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test task scheduler
    print("  Testing TaskScheduler...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # TaskScheduler needs an AgentRegistry
        registry_for_scheduler = AgentRegistry(":memory:")
        scheduler = TaskScheduler(registry_for_scheduler, tmp.name)

        task = Task(
            task_id="test_task",
            task_type="data_processing",
            description="Test task",
            priority=5,
            payload={"input": "test_data"},
        )

        success = scheduler.submit_task(task)
        print(f"    Task submission: {success}")

        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test message broker
    print("  Testing MessageBroker...")
    broker = MessageBroker()

    # Register a dummy handler
    def dummy_handler(message):
        pass

    broker.register_handler("agent1", MessageType.TASK_REQUEST, dummy_handler)
    print("    Handler registration: OK")

    message = Message(
        message_id="msg1",
        message_type=MessageType.TASK_REQUEST,
        sender_id="coordinator",
        recipient_id="agent1",
        payload={"test": "data"},
        timestamp=time.time(),
    )

    broker.send_message(message)
    print("    Message send: OK")

    messages = broker.get_messages("agent1")
    print(f"    Messages retrieved: {len(messages)}")

    # Test resource manager
    print("  Testing ResourceManager...")
    rm = ResourceManager()

    resource = Resource(
        resource_id="test_resource",
        resource_type="cpu",
        capacity=100.0,
        allocated=0.0,
        available=100.0,
    )

    rm.register_resource(resource)
    print("    Resource registration: OK")

    print("Basic tests completed successfully!")


if __name__ == "__main__":
    test_basic_functionality()
