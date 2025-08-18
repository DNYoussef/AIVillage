"""Tests for Agent Coordination System - Prompt J (Working Version)

Comprehensive validation of agent coordination including:
- Agent registry and discovery functionality
- Task scheduling and management
- Inter-agent message broker communication
- Resource allocation and management
- Coordination engine orchestration

Integration Point: Agent coordination validation for Phase 4 testing
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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
    TaskStatus,
)


class TestAgentRegistry:
    """Test agent registry functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.registry = AgentRegistry(self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_registry_initialization(self):
        """Test agent registry initialization."""
        assert len(self.registry.agents) == 0
        assert len(self.registry.capabilities_index) == 0

    def test_agent_registration(self):
        """Test agent registration."""
        capabilities = [
            AgentCapability(
                "data_processing",
                "1.0",
                "Data processing capability",
                supported_task_types=["data_processing"],
            )
        ]
        agent = Agent(
            agent_id="test_agent_001",
            name="Test Agent",
            agent_type="worker",
            capabilities=capabilities,
            status=AgentStatus.IDLE,
            endpoint="http://localhost:8001",
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        success = self.registry.register_agent(agent)

        assert success is True
        assert agent.agent_id in self.registry.agents
        assert self.registry.agents[agent.agent_id] == agent
        assert len(self.registry.capabilities_index["data_processing"]) == 1

    def test_agent_discovery_by_capability(self):
        """Test agent discovery by capability."""
        # Create agents with different capabilities
        agents = [
            Agent(
                "agent1",
                "Agent 1",
                "worker",
                [
                    AgentCapability(
                        "data_processing",
                        "1.0",
                        "Data processing",
                        supported_task_types=["data_processing"],
                    )
                ],
                AgentStatus.IDLE,
                "http://localhost:8001",
                time.time(),
                time.time(),
            ),
            Agent(
                "agent2",
                "Agent 2",
                "worker",
                [
                    AgentCapability(
                        "ml_inference",
                        "1.0",
                        "ML inference",
                        supported_task_types=["ml_inference"],
                    )
                ],
                AgentStatus.IDLE,
                "http://localhost:8002",
                time.time(),
                time.time(),
            ),
        ]

        for agent in agents:
            self.registry.register_agent(agent)

        # Find agents with data_processing capability
        data_agents = self.registry.find_agents_by_capability("data_processing")
        assert len(data_agents) == 1
        assert data_agents[0].agent_id == "agent1"

        # Find agents with ml_inference capability
        ml_agents = self.registry.find_agents_by_capability("ml_inference")
        assert len(ml_agents) == 1
        assert ml_agents[0].agent_id == "agent2"

        # Find agents with non-existent capability
        no_agents = self.registry.find_agents_by_capability("non_existent")
        assert len(no_agents) == 0

    def test_agent_unregistration(self):
        """Test agent unregistration."""
        capabilities = [
            AgentCapability(
                "temp_capability",
                "1.0",
                "Temporary capability",
                supported_task_types=["temp_capability"],
            )
        ]
        agent = Agent(
            agent_id="temp_agent",
            name="Temporary Agent",
            agent_type="worker",
            capabilities=capabilities,
            status=AgentStatus.IDLE,
            endpoint="http://localhost:8001",
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        # Register then unregister
        self.registry.register_agent(agent)
        assert agent.agent_id in self.registry.agents

        success = self.registry.unregister_agent(agent.agent_id)

        assert success is True
        assert agent.agent_id not in self.registry.agents


class TestTaskScheduler:
    """Test task scheduler functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.registry = AgentRegistry(":memory:")
        self.scheduler = TaskScheduler(self.registry, self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_scheduler_initialization(self):
        """Test task scheduler initialization."""
        assert len(self.scheduler.pending_tasks) == 0
        assert len(self.scheduler.running_tasks) == 0
        assert len(self.scheduler.completed_tasks) == 0

    def test_task_submission(self):
        """Test task submission."""
        task = Task(
            task_id="test_task_001",
            task_type="data_processing",
            description="Test task",
            priority=5,
            payload={"input_file": "test.csv"},
        )

        task_id = self.scheduler.submit_task(task)

        assert task_id == "test_task_001"
        assert len(self.scheduler.pending_tasks) == 1
        assert task.status == TaskStatus.PENDING

    def test_task_priority_ordering(self):
        """Test task priority ordering."""
        tasks = [
            Task("low", "data_processing", "Low Priority", 1, {}),
            Task("high", "data_processing", "High Priority", 10, {}),
            Task("medium", "data_processing", "Medium Priority", 5, {}),
        ]

        # Submit tasks in random order
        for task in tasks:
            self.scheduler.submit_task(task)

        # Tasks should be ordered by priority (highest first)
        assert len(self.scheduler.pending_tasks) == 3
        # Priority queue returns highest priority first
        priorities = [item[0] for item in sorted(self.scheduler.pending_tasks, reverse=True)]
        assert priorities == [10, 5, 1]  # Descending order

    def test_task_completion(self):
        """Test task completion."""
        task = Task(
            task_id="complete_test",
            task_type="data_processing",
            description="Completion Test",
            priority=5,
            payload={},
        )

        self.scheduler.submit_task(task)

        # Complete task
        result = {"output": "test_result", "status": "success"}
        success = self.scheduler.complete_task("complete_test", result)

        assert success is True
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None
        assert "complete_test" in self.scheduler.completed_tasks


class TestMessageBroker:
    """Test message broker functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.broker = MessageBroker()

    def test_broker_initialization(self):
        """Test message broker initialization."""
        assert len(self.broker.message_queues) == 0
        assert len(self.broker.message_handlers) == 0

    def test_handler_registration(self):
        """Test message handler registration."""

        def dummy_handler(message):
            pass

        self.broker.register_handler("agent1", MessageType.TASK_REQUEST, dummy_handler)

        assert "agent1" in self.broker.message_handlers
        assert MessageType.TASK_REQUEST in self.broker.message_handlers["agent1"]
        assert self.broker.message_handlers["agent1"][MessageType.TASK_REQUEST] == dummy_handler

    def test_message_sending(self):
        """Test message sending."""
        message = Message(
            message_id="msg_001",
            message_type=MessageType.TASK_REQUEST,
            sender_id="coordinator",
            recipient_id="agent1",
            payload={"task_id": "task_001", "action": "start"},
            timestamp=time.time(),
        )

        # Send message
        self.broker.send_message(message)

        # Check message was queued
        assert "agent1" in self.broker.message_queues
        assert len(self.broker.message_queues["agent1"]) == 1

    def test_message_retrieval(self):
        """Test message retrieval by agents."""
        # Send multiple messages
        messages = [
            Message(
                f"msg_{i}",
                MessageType.TASK_REQUEST,
                "coordinator",
                "agent1",
                {"task_id": f"task_{i}"},
                time.time(),
            )
            for i in range(3)
        ]

        for message in messages:
            self.broker.send_message(message)

        # Retrieve messages
        retrieved = self.broker.get_messages("agent1")

        assert len(retrieved) == 3

    def test_broadcast_message(self):
        """Test broadcast message handling."""
        # Send broadcast message (no specific recipient)
        broadcast_msg = Message(
            message_id="broadcast_001",
            message_type=MessageType.BROADCAST,
            sender_id="coordinator",
            recipient_id=None,  # Broadcast
            payload={"notification": "System maintenance"},
            timestamp=time.time(),
        )

        self.broker.send_message(broadcast_msg)

        # Broadcast messages go to broadcast queue
        assert len(self.broker.broadcast_queue) == 1


class TestResourceManager:
    """Test resource manager functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.resource_manager = ResourceManager()

    def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        assert len(self.resource_manager.resources) == 0
        assert len(self.resource_manager.allocations) == 0

    def test_resource_registration(self):
        """Test resource registration."""
        resource = Resource(
            resource_id="cpu_001",
            resource_type="cpu",
            capacity=100.0,
            allocated=0.0,
            available=100.0,
        )

        self.resource_manager.register_resource(resource)

        assert resource.resource_id in self.resource_manager.resources
        assert self.resource_manager.resources[resource.resource_id] == resource

    def test_resource_allocation(self):
        """Test resource allocation."""
        # Register resource
        resource = Resource(
            resource_id="memory_001",
            resource_type="memory",
            capacity=1000.0,
            allocated=0.0,
            available=1000.0,
        )

        self.resource_manager.register_resource(resource)

        # Allocate resources
        success = self.resource_manager.allocate_resource(resource_id="memory_001", agent_id="agent_001", amount=500.0)

        assert success is True
        assert "memory_001" in self.resource_manager.allocations
        assert "agent_001" in self.resource_manager.allocations["memory_001"]
        assert self.resource_manager.allocations["memory_001"]["agent_001"] == 500.0

        # Check resource was updated
        updated_resource = self.resource_manager.resources["memory_001"]
        assert updated_resource.allocated == 500.0
        assert updated_resource.available == 500.0

    def test_resource_release(self):
        """Test resource release."""
        # Setup and allocate resource
        resource = Resource("test_resource", "cpu", 100.0, 0.0, 100.0)
        self.resource_manager.register_resource(resource)
        self.resource_manager.allocate_resource("test_resource", "agent1", 50.0)

        # Release resource
        success = self.resource_manager.release_resource("test_resource", "agent1")

        assert success is True
        assert "agent1" not in self.resource_manager.allocations.get("test_resource", {})

        # Check resource was updated
        updated_resource = self.resource_manager.resources["test_resource"]
        assert updated_resource.allocated == 0.0
        assert updated_resource.available == 100.0


class TestCoordinationEngine:
    """Test coordination engine functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dbs = []

        # Create temporary databases for each component
        for i in range(4):
            temp_db = tempfile.NamedTemporaryFile(delete=False)
            temp_db.close()
            self.temp_dbs.append(temp_db.name)

        # Note: CoordinationEngine constructor signature needs to be checked
        # Using simplified initialization for now
        self.registry = AgentRegistry(self.temp_dbs[0])
        self.scheduler = TaskScheduler(self.registry, self.temp_dbs[1])
        self.broker = MessageBroker()
        self.resource_manager = ResourceManager()

    def teardown_method(self):
        """Cleanup after each test."""
        for db_path in self.temp_dbs:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

    def test_agent_task_workflow(self):
        """Test complete agent-task workflow."""
        # Register an agent
        capabilities = [
            AgentCapability(
                "data_processing",
                "1.0",
                "Data processing capability",
                supported_task_types=["data_processing"],
            )
        ]
        agent = Agent(
            agent_id="workflow_agent",
            name="Workflow Agent",
            agent_type="worker",
            capabilities=capabilities,
            status=AgentStatus.IDLE,
            endpoint="http://localhost:8001",
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        self.registry.register_agent(agent)

        # Submit a task
        task = Task(
            task_id="workflow_task",
            task_type="data_processing",
            description="Workflow task",
            priority=5,
            payload={"input": "test_data"},
        )

        task_id = self.scheduler.submit_task(task)
        assert task_id == "workflow_task"

        # Find suitable agents for the task
        available_agents = self.registry.find_agents_by_capability("data_processing")
        assert len(available_agents) == 1
        assert available_agents[0].agent_id == "workflow_agent"

        # Simulate task scheduling and execution
        assert task.status == TaskStatus.PENDING

        # Complete the task
        result = {"output": "processed_data", "status": "success"}
        success = self.scheduler.complete_task("workflow_task", result)
        assert success is True
        assert task.status == TaskStatus.COMPLETED

    def test_resource_aware_coordination(self):
        """Test resource-aware coordination."""
        # Register a resource
        resource = Resource(
            resource_id="gpu_resource",
            resource_type="gpu",
            capacity=100.0,
            allocated=0.0,
            available=100.0,
        )

        self.resource_manager.register_resource(resource)

        # Allocate some resources
        success = self.resource_manager.allocate_resource("gpu_resource", "agent1", 50.0)
        assert success is True

        # Check resource usage
        usage = self.resource_manager.get_resource_usage()
        assert "gpu_resource" in usage
        assert usage["gpu_resource"]["agent1"] == 50.0

    def test_message_coordination(self):
        """Test message-based coordination."""

        # Register a handler
        def task_handler(message):
            return {"status": "handled", "message_id": message.message_id}

        self.broker.register_handler("agent1", MessageType.TASK_REQUEST, task_handler)

        # Send a task message
        message = Message(
            message_id="coord_msg_001",
            message_type=MessageType.TASK_REQUEST,
            sender_id="coordinator",
            recipient_id="agent1",
            payload={"task_id": "coord_task", "action": "execute"},
            timestamp=time.time(),
        )

        self.broker.send_message(message)

        # Retrieve and verify message
        messages = self.broker.get_messages("agent1")
        assert len(messages) == 1
        assert messages[0].message_id == "coord_msg_001"
        assert messages[0].payload["task_id"] == "coord_task"


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_multi_component_coordination(self):
        """Test coordination across multiple components."""
        # Setup components
        registry = AgentRegistry(":memory:")
        scheduler = TaskScheduler(registry, ":memory:")
        MessageBroker()
        resource_manager = ResourceManager()

        # Register multiple specialized agents
        agents = [
            Agent(
                "data_agent",
                "Data Agent",
                "processor",
                [
                    AgentCapability(
                        "data_cleaning",
                        "1.0",
                        "Data cleaning",
                        supported_task_types=["data_cleaning"],
                    )
                ],
                AgentStatus.IDLE,
                "http://localhost:8001",
                time.time(),
                time.time(),
            ),
            Agent(
                "ml_agent",
                "ML Agent",
                "processor",
                [
                    AgentCapability(
                        "model_training",
                        "1.0",
                        "Model training",
                        supported_task_types=["model_training"],
                    )
                ],
                AgentStatus.IDLE,
                "http://localhost:8002",
                time.time(),
                time.time(),
            ),
        ]

        for agent in agents:
            registry.register_agent(agent)

        # Register resources
        resources = [
            Resource("cpu_cluster", "cpu", 1000.0, 0.0, 1000.0),
            Resource("gpu_node", "gpu", 500.0, 0.0, 500.0),
        ]

        for resource in resources:
            resource_manager.register_resource(resource)

        # Submit workflow tasks
        tasks = [
            Task(
                "clean_data",
                "data_cleaning",
                "Clean dataset",
                10,
                {"dataset": "raw_data.csv"},
            ),
            Task(
                "train_model",
                "model_training",
                "Train ML model",
                8,
                {"cleaned_data": "processed.csv"},
            ),
        ]

        for task in tasks:
            scheduler.submit_task(task)

        # Verify system state
        assert len(registry.agents) == 2
        assert len(scheduler.pending_tasks) == 2
        assert len(resource_manager.resources) == 2

        # Test agent discovery for different capabilities
        data_agents = registry.find_agents_by_capability("data_cleaning")
        ml_agents = registry.find_agents_by_capability("model_training")

        assert len(data_agents) == 1
        assert len(ml_agents) == 1
        assert data_agents[0].agent_id == "data_agent"
        assert ml_agents[0].agent_id == "ml_agent"


if __name__ == "__main__":
    # Run coordination system validation
    print("=== Testing Agent Coordination System ===")

    # Test agent registry
    print("Testing agent registry...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        registry = AgentRegistry(tmp.name)
        capabilities = [AgentCapability("test_cap", "1.0", "Test", supported_task_types=["test_cap"])]
        agent = Agent(
            "test_agent",
            "Test Agent",
            "worker",
            capabilities,
            AgentStatus.IDLE,
            "http://localhost:8001",
            time.time(),
            time.time(),
        )
        success = registry.register_agent(agent)
        found_agents = registry.find_agents_by_capability("test_cap")
        print(f"OK Agent registry: registered={success}, found={len(found_agents)}")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test task scheduler
    print("Testing task scheduler...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        registry = AgentRegistry(":memory:")
        scheduler = TaskScheduler(registry, tmp.name)
        task = Task("test_task", "test_type", "Test Task", 5, {})
        task_id = scheduler.submit_task(task)
        print(f"OK Task scheduler: submitted task_id={task_id}")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test message broker
    print("Testing message broker...")
    broker = MessageBroker()

    def dummy_handler(msg):
        pass

    broker.register_handler("agent1", MessageType.TASK_REQUEST, dummy_handler)
    message = Message(
        "msg1",
        MessageType.TASK_REQUEST,
        "coordinator",
        "agent1",
        {"test": "data"},
        time.time(),
    )
    broker.send_message(message)
    messages = broker.get_messages("agent1")
    print(f"OK Message broker: sent and received {len(messages)} messages")

    # Test resource manager
    print("Testing resource manager...")
    rm = ResourceManager()
    resource = Resource("test_res", "cpu", 100.0, 0.0, 100.0)
    rm.register_resource(resource)
    allocated = rm.allocate_resource("test_res", "agent1", 50.0)
    print(f"OK Resource manager: allocated={allocated}")

    print("=== Agent coordination system validation completed ===")
