"""Tests for Agent Coordination System - Prompt J

Comprehensive validation of agent coordination including:
- Agent registry and discovery functionality
- Task scheduling and priority management
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
    CoordinationEngine,
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
            ),
            AgentCapability(
                "file_operations",
                "1.0",
                "File operations capability",
                supported_task_types=["file_operations"],
            ),
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
        assert len(self.registry.capabilities_index["file_operations"]) == 1

    def test_agent_registration_duplicate(self):
        """Test registration of duplicate agent."""
        agent1 = Agent(
            agent_id="duplicate_agent",
            name="First Agent",
            agent_type="worker",
            capabilities=[
                AgentCapability(
                    "capability1",
                    "1.0",
                    "Test capability",
                    supported_task_types=["capability1"],
                )
            ],
            endpoint="http://localhost:8001",
            status=AgentStatus.IDLE,
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        agent2 = Agent(
            agent_id="duplicate_agent",
            name="Second Agent",
            agent_type="worker",
            capabilities=[
                AgentCapability(
                    "capability2",
                    "1.0",
                    "Test capability",
                    supported_task_types=["capability2"],
                )
            ],
            endpoint="http://localhost:8002",
            status=AgentStatus.IDLE,
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        # First registration should succeed
        success1 = self.registry.register_agent(agent1)
        assert success1 is True

        # Second registration with same ID should fail
        success2 = self.registry.register_agent(agent2)
        assert success2 is False

        # Original agent should remain
        assert self.registry.agents["duplicate_agent"].name == "First Agent"

    def test_agent_deregistration(self):
        """Test agent deregistration."""
        agent = Agent(
            agent_id="temp_agent",
            name="Temporary Agent",
            agent_type="worker",
            capabilities=[
                AgentCapability(
                    "temp_capability",
                    "1.0",
                    "Temporary capability",
                    supported_task_types=["temp_capability"],
                )
            ],
            endpoint="http://localhost:8001",
            status=AgentStatus.IDLE,
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        # Register then deregister
        self.registry.register_agent(agent)
        assert agent.agent_id in self.registry.agents

        success = self.registry.unregister_agent(agent.agent_id)

        assert success is True
        assert agent.agent_id not in self.registry.agents
        assert len(self.registry.capabilities_index["temp_capability"]) == 0

    def test_agent_discovery_by_capability(self):
        """Test agent discovery by capability."""
        # Register multiple agents with different capabilities
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
                        "data_processing",
                        "1.0",
                        "Data processing",
                        supported_task_types=["data_processing"],
                    ),
                    AgentCapability(
                        "ml_inference",
                        "1.0",
                        "ML inference",
                        supported_task_types=["ml_inference"],
                    ),
                ],
                AgentStatus.IDLE,
                "http://localhost:8002",
                time.time(),
                time.time(),
            ),
            Agent(
                "agent3",
                "Agent 3",
                "coordinator",
                [
                    AgentCapability(
                        "task_coordination",
                        "1.0",
                        "Task coordination",
                        supported_task_types=["task_coordination"],
                    )
                ],
                AgentStatus.IDLE,
                "http://localhost:8003",
                time.time(),
                time.time(),
            ),
        ]

        for agent in agents:
            self.registry.register_agent(agent)

        # Find agents with data_processing capability
        data_agents = self.registry.find_agents_by_capability("data_processing")
        assert len(data_agents) == 2
        assert "agent1" in [agent.agent_id for agent in data_agents]
        assert "agent2" in [agent.agent_id for agent in data_agents]

        # Find agents with ml_inference capability
        ml_agents = self.registry.find_agents_by_capability("ml_inference")
        assert len(ml_agents) == 1
        assert ml_agents[0].agent_id == "agent2"

        # Find agents with non-existent capability
        no_agents = self.registry.find_agents_by_capability("non_existent")
        assert len(no_agents) == 0

    def test_agent_status_update(self):
        """Test agent status updates."""
        agent = Agent(
            agent_id="status_agent",
            name="Status Agent",
            agent_type="worker",
            capabilities=["status_test"],
            endpoint="http://localhost:8001",
            status=AgentStatus.IDLE,
        )

        self.registry.register_agent(agent)

        # Update status
        success = self.registry.update_agent_status("status_agent", AgentStatus.BUSY)

        assert success is True
        assert self.registry.agents["status_agent"].status == AgentStatus.BUSY
        assert self.registry.agents["status_agent"].last_seen > 0

        # Update non-existent agent
        success = self.registry.update_agent_status("non_existent", AgentStatus.BUSY)
        assert success is False

    def test_agent_heartbeat(self):
        """Test agent heartbeat functionality."""
        agent = Agent(
            agent_id="heartbeat_agent",
            name="Heartbeat Agent",
            agent_type="worker",
            capabilities=["heartbeat_test"],
            endpoint="http://localhost:8001",
        )

        self.registry.register_agent(agent)
        original_heartbeat = self.registry.agents["heartbeat_agent"].last_seen

        # Wait a moment then send heartbeat
        time.sleep(0.01)
        success = self.registry.agent_heartbeat("heartbeat_agent")

        assert success is True
        assert self.registry.agents["heartbeat_agent"].last_seen > original_heartbeat

    def test_get_agent_stats(self):
        """Test agent statistics retrieval."""
        # Register agents with different statuses
        agents = [
            Agent(
                "idle1",
                "Idle 1",
                "worker",
                ["cap1"],
                "http://localhost:8001",
                status=AgentStatus.IDLE,
            ),
            Agent(
                "idle2",
                "Idle 2",
                "worker",
                ["cap2"],
                "http://localhost:8002",
                status=AgentStatus.IDLE,
            ),
            Agent(
                "busy1",
                "Busy 1",
                "worker",
                ["cap3"],
                "http://localhost:8003",
                status=AgentStatus.BUSY,
            ),
            Agent(
                "error1",
                "Error 1",
                "worker",
                ["cap4"],
                "http://localhost:8004",
                status=AgentStatus.ERROR,
            ),
        ]

        for agent in agents:
            self.registry.register_agent(agent)

        stats = self.registry.get_agent_stats()

        assert stats["total"] == 4
        assert stats["idle"] == 2
        assert stats["busy"] == 1
        assert stats["error"] == 1
        assert stats["offline"] == 0


class TestTaskScheduler:
    """Test task scheduler functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.scheduler = TaskScheduler(self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_scheduler_initialization(self):
        """Test task scheduler initialization."""
        assert len(self.scheduler.task_queue) == 0
        assert len(self.scheduler.running_tasks) == 0
        assert len(self.scheduler.completed_tasks) == 0

    def test_task_submission(self):
        """Test task submission."""
        task = Task(
            task_id="test_task_001",
            name="Test Task",
            task_type="data_processing",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["data_processing"],
            parameters={"input_file": "test.csv"},
            created_by="user123",
        )

        success = self.scheduler.submit_task(task)

        assert success is True
        assert task.task_id in self.scheduler.task_queue
        assert self.scheduler.task_queue[task.task_id] == task
        assert task.status == TaskStatus.QUEUED

    def test_task_priority_ordering(self):
        """Test task priority ordering."""
        tasks = [
            Task(
                "low", "Low Priority", "test", TaskPriority.LOW, ["cap1"], {}, "user1"
            ),
            Task(
                "high",
                "High Priority",
                "test",
                TaskPriority.HIGH,
                ["cap1"],
                {},
                "user1",
            ),
            Task(
                "medium",
                "Medium Priority",
                "test",
                TaskPriority.MEDIUM,
                ["cap1"],
                {},
                "user1",
            ),
            Task(
                "critical",
                "Critical Priority",
                "test",
                TaskPriority.CRITICAL,
                ["cap1"],
                {},
                "user1",
            ),
        ]

        # Submit tasks in random order
        for task in tasks:
            self.scheduler.submit_task(task)

        # Get next tasks - should be ordered by priority
        next_tasks = self.scheduler.get_next_tasks(capability="cap1", limit=4)

        assert len(next_tasks) == 4
        assert next_tasks[0].task_id == "critical"  # CRITICAL first
        assert next_tasks[1].task_id == "high"  # HIGH second
        assert next_tasks[2].task_id == "medium"  # MEDIUM third
        assert next_tasks[3].task_id == "low"  # LOW last

    def test_task_assignment(self):
        """Test task assignment to agents."""
        task = Task(
            task_id="assign_test",
            name="Assignment Test",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["test_cap"],
            parameters={},
            created_by="user1",
        )

        self.scheduler.submit_task(task)

        # Assign task to agent
        success = self.scheduler.assign_task("assign_test", "test_agent_001")

        assert success is True
        assert task.assigned_agent == "test_agent_001"
        assert task.status == TaskStatus.RUNNING
        assert task.task_id in self.scheduler.running_tasks
        assert task.task_id not in self.scheduler.task_queue

    def test_task_completion(self):
        """Test task completion."""
        task = Task(
            task_id="complete_test",
            name="Completion Test",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["test_cap"],
            parameters={},
            created_by="user1",
        )

        self.scheduler.submit_task(task)
        self.scheduler.assign_task("complete_test", "test_agent_001")

        # Complete task
        result = {"output": "test_result", "status": "success"}
        success = self.scheduler.complete_task("complete_test", result)

        assert success is True
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None
        assert task.task_id in self.scheduler.completed_tasks
        assert task.task_id not in self.scheduler.running_tasks

    def test_task_failure(self):
        """Test task failure handling."""
        task = Task(
            task_id="fail_test",
            name="Failure Test",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["test_cap"],
            parameters={},
            created_by="user1",
        )

        self.scheduler.submit_task(task)
        self.scheduler.assign_task("fail_test", "test_agent_001")

        # Fail task
        error_msg = "Task failed due to invalid input"
        success = self.scheduler.fail_task("fail_test", error_msg)

        assert success is True
        assert task.status == TaskStatus.FAILED
        assert task.error_message == error_msg
        assert task.completed_at is not None
        assert task.task_id in self.scheduler.completed_tasks
        assert task.task_id not in self.scheduler.running_tasks

    def test_task_retry_mechanism(self):
        """Test task retry mechanism."""
        task = Task(
            task_id="retry_test",
            name="Retry Test",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["test_cap"],
            parameters={},
            created_by="user1",
            max_retries=2,
        )

        self.scheduler.submit_task(task)
        self.scheduler.assign_task("retry_test", "test_agent_001")

        # First failure - should retry
        success = self.scheduler.fail_task("retry_test", "First failure")

        assert success is True
        assert task.status == TaskStatus.QUEUED  # Should be back in queue
        assert task.retry_count == 1
        assert task.task_id in self.scheduler.task_queue

        # Assign and fail again
        self.scheduler.assign_task("retry_test", "test_agent_002")
        success = self.scheduler.fail_task("retry_test", "Second failure")

        assert success is True
        assert task.status == TaskStatus.QUEUED  # Should retry again
        assert task.retry_count == 2

        # Assign and fail third time - should not retry
        self.scheduler.assign_task("retry_test", "test_agent_003")
        success = self.scheduler.fail_task("retry_test", "Final failure")

        assert success is True
        assert task.status == TaskStatus.FAILED  # Should be permanently failed
        assert task.retry_count == 2
        assert task.task_id in self.scheduler.completed_tasks

    def test_get_task_status(self):
        """Test task status retrieval."""
        task = Task(
            task_id="status_test",
            name="Status Test",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["test_cap"],
            parameters={},
            created_by="user1",
        )

        self.scheduler.submit_task(task)

        # Get task status
        status = self.scheduler.get_task_status("status_test")

        assert status is not None
        assert status["task_id"] == "status_test"
        assert status["status"] == TaskStatus.QUEUED.value
        assert status["priority"] == TaskPriority.MEDIUM.value
        assert "created_at" in status

        # Non-existent task
        status = self.scheduler.get_task_status("non_existent")
        assert status is None


class TestMessageBroker:
    """Test message broker functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.broker = MessageBroker()

    def test_broker_initialization(self):
        """Test message broker initialization."""
        assert len(self.broker.subscriptions) == 0
        assert len(self.broker.message_queue) == 0

    def test_agent_subscription(self):
        """Test agent subscription to message types."""
        # Subscribe agent to specific message types
        success = self.broker.subscribe("agent1", MessageType.TASK_ASSIGNMENT)
        assert success is True

        success = self.broker.subscribe("agent1", MessageType.TASK_RESULT)
        assert success is True

        # Check subscriptions
        assert MessageType.TASK_ASSIGNMENT in self.broker.subscriptions
        assert MessageType.TASK_RESULT in self.broker.subscriptions
        assert "agent1" in self.broker.subscriptions[MessageType.TASK_ASSIGNMENT]
        assert "agent1" in self.broker.subscriptions[MessageType.TASK_RESULT]

    def test_agent_unsubscription(self):
        """Test agent unsubscription from message types."""
        # Subscribe then unsubscribe
        self.broker.subscribe("agent1", MessageType.TASK_ASSIGNMENT)
        self.broker.subscribe("agent1", MessageType.TASK_RESULT)

        success = self.broker.unsubscribe("agent1", MessageType.TASK_ASSIGNMENT)
        assert success is True

        # Check that agent is removed from task assignment but not task result
        assert "agent1" not in self.broker.subscriptions[MessageType.TASK_ASSIGNMENT]
        assert "agent1" in self.broker.subscriptions[MessageType.TASK_RESULT]

    def test_message_publishing(self):
        """Test message publishing."""
        # Subscribe agent
        self.broker.subscribe("agent1", MessageType.TASK_ASSIGNMENT)
        self.broker.subscribe("agent2", MessageType.TASK_ASSIGNMENT)

        message = Message(
            message_id="msg_001",
            message_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipients=["agent1"],
            payload={"task_id": "task_001", "action": "start"},
            timestamp=time.time(),
        )

        # Publish message
        success = self.broker.publish_message(message)

        assert success is True
        assert "agent1" in self.broker.message_queue
        assert len(self.broker.message_queue["agent1"]) == 1
        assert self.broker.message_queue["agent1"][0] == message

        # agent2 should not receive the message (not in recipients)
        assert (
            "agent2" not in self.broker.message_queue
            or len(self.broker.message_queue["agent2"]) == 0
        )

    def test_broadcast_message(self):
        """Test message broadcasting."""
        # Subscribe multiple agents
        agents = ["agent1", "agent2", "agent3"]
        for agent in agents:
            self.broker.subscribe(agent, MessageType.SYSTEM_NOTIFICATION)

        message = Message(
            message_id="broadcast_001",
            message_type=MessageType.SYSTEM_NOTIFICATION,
            sender="coordinator",
            recipients=[],  # Empty recipients for broadcast
            payload={"notification": "System maintenance in 10 minutes"},
            timestamp=time.time(),
        )

        # Broadcast message
        success = self.broker.broadcast_message(
            message, MessageType.SYSTEM_NOTIFICATION
        )

        assert success is True

        # All subscribed agents should receive the message
        for agent in agents:
            assert agent in self.broker.message_queue
            assert len(self.broker.message_queue[agent]) == 1
            assert self.broker.message_queue[agent][0] == message

    def test_message_retrieval(self):
        """Test message retrieval by agents."""
        # Subscribe agent and send messages
        self.broker.subscribe("agent1", MessageType.TASK_ASSIGNMENT)

        messages = [
            Message(
                f"msg_{i}",
                MessageType.TASK_ASSIGNMENT,
                "coordinator",
                ["agent1"],
                {"task_id": f"task_{i}"},
                time.time(),
            )
            for i in range(3)
        ]

        for message in messages:
            self.broker.publish_message(message)

        # Retrieve messages
        retrieved = self.broker.get_messages("agent1")

        assert len(retrieved) == 3
        assert all(msg in messages for msg in retrieved)

        # Message queue should be cleared after retrieval
        assert len(self.broker.message_queue.get("agent1", [])) == 0

    def test_message_filtering(self):
        """Test message filtering by type."""
        # Subscribe agent to multiple message types
        self.broker.subscribe("agent1", MessageType.TASK_ASSIGNMENT)
        self.broker.subscribe("agent1", MessageType.TASK_RESULT)

        # Send different types of messages
        task_msg = Message(
            "task_msg",
            MessageType.TASK_ASSIGNMENT,
            "coordinator",
            ["agent1"],
            {"task_id": "task_001"},
            time.time(),
        )
        result_msg = Message(
            "result_msg",
            MessageType.TASK_RESULT,
            "agent2",
            ["agent1"],
            {"result": "success"},
            time.time(),
        )

        self.broker.publish_message(task_msg)
        self.broker.publish_message(result_msg)

        # Get only task assignment messages
        task_messages = self.broker.get_messages(
            "agent1", message_type=MessageType.TASK_ASSIGNMENT
        )

        assert len(task_messages) == 1
        assert task_messages[0].message_type == MessageType.TASK_ASSIGNMENT

        # Get remaining messages (should be task result)
        remaining = self.broker.get_messages("agent1")
        assert len(remaining) == 1
        assert remaining[0].message_type == MessageType.TASK_RESULT


class TestResourceManager:
    """Test resource manager functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.resource_manager = ResourceManager(self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        assert len(self.resource_manager.resources) == 0
        assert len(self.resource_manager.allocations) == 0

    def test_resource_registration(self):
        """Test resource registration."""
        resource = Resource(
            resource_id="gpu_001",
            name="GPU Node 1",
            resource_type=ResourceType.GPU,
            capacity={"memory_gb": 16, "cores": 2048},
            available={"memory_gb": 16, "cores": 2048},
            metadata={"model": "RTX 4090", "location": "rack_01"},
        )

        success = self.resource_manager.register_resource(resource)

        assert success is True
        assert resource.resource_id in self.resource_manager.resources
        assert self.resource_manager.resources[resource.resource_id] == resource

    def test_resource_allocation(self):
        """Test resource allocation."""
        # Register resource
        resource = Resource(
            resource_id="cpu_001",
            name="CPU Node 1",
            resource_type=ResourceType.CPU,
            capacity={"cores": 8, "memory_gb": 32},
            available={"cores": 8, "memory_gb": 32},
        )

        self.resource_manager.register_resource(resource)

        # Allocate resources
        allocation_id = self.resource_manager.allocate_resource(
            resource_id="cpu_001",
            requester="task_001",
            requirements={"cores": 4, "memory_gb": 16},
        )

        assert allocation_id is not None
        assert allocation_id in self.resource_manager.allocations

        # Check resource availability updated
        updated_resource = self.resource_manager.resources["cpu_001"]
        assert updated_resource.available["cores"] == 4
        assert updated_resource.available["memory_gb"] == 16

    def test_insufficient_resources(self):
        """Test allocation with insufficient resources."""
        resource = Resource(
            resource_id="small_cpu",
            name="Small CPU",
            resource_type=ResourceType.CPU,
            capacity={"cores": 2, "memory_gb": 4},
            available={"cores": 2, "memory_gb": 4},
        )

        self.resource_manager.register_resource(resource)

        # Try to allocate more than available
        allocation_id = self.resource_manager.allocate_resource(
            resource_id="small_cpu",
            requester="greedy_task",
            requirements={"cores": 4, "memory_gb": 8},
        )

        assert allocation_id is None

        # Resource should remain unchanged
        assert self.resource_manager.resources["small_cpu"].available["cores"] == 2
        assert self.resource_manager.resources["small_cpu"].available["memory_gb"] == 4

    def test_resource_deallocation(self):
        """Test resource deallocation."""
        # Register and allocate resource
        resource = Resource(
            resource_id="memory_001",
            name="Memory Node 1",
            resource_type=ResourceType.MEMORY,
            capacity={"memory_gb": 64},
            available={"memory_gb": 64},
        )

        self.resource_manager.register_resource(resource)

        allocation_id = self.resource_manager.allocate_resource(
            resource_id="memory_001",
            requester="task_001",
            requirements={"memory_gb": 32},
        )

        assert allocation_id is not None
        assert (
            self.resource_manager.resources["memory_001"].available["memory_gb"] == 32
        )

        # Deallocate
        success = self.resource_manager.deallocate_resource(allocation_id)

        assert success is True
        assert allocation_id not in self.resource_manager.allocations
        assert (
            self.resource_manager.resources["memory_001"].available["memory_gb"] == 64
        )

    def test_find_available_resources(self):
        """Test finding available resources."""
        # Register multiple resources
        resources = [
            Resource(
                "gpu_1", "GPU 1", ResourceType.GPU, {"memory_gb": 16}, {"memory_gb": 16}
            ),
            Resource(
                "gpu_2", "GPU 2", ResourceType.GPU, {"memory_gb": 8}, {"memory_gb": 8}
            ),
            Resource("cpu_1", "CPU 1", ResourceType.CPU, {"cores": 16}, {"cores": 16}),
        ]

        for resource in resources:
            self.resource_manager.register_resource(resource)

        # Find GPUs with at least 10GB memory
        gpu_resources = self.resource_manager.find_available_resources(
            resource_type=ResourceType.GPU, requirements={"memory_gb": 10}
        )

        assert len(gpu_resources) == 1
        assert gpu_resources[0].resource_id == "gpu_1"

        # Find any resource type with specific requirements
        any_resources = self.resource_manager.find_available_resources(
            requirements={"memory_gb": 5}
        )

        assert len(any_resources) == 2  # Both GPUs meet the requirement

    def test_resource_usage_stats(self):
        """Test resource usage statistics."""
        # Register and partially allocate resources
        resource = Resource(
            resource_id="stats_test",
            name="Stats Test Resource",
            resource_type=ResourceType.CPU,
            capacity={"cores": 10, "memory_gb": 20},
            available={"cores": 10, "memory_gb": 20},
        )

        self.resource_manager.register_resource(resource)

        # Allocate some resources
        self.resource_manager.allocate_resource(
            resource_id="stats_test",
            requester="task_1",
            requirements={"cores": 3, "memory_gb": 8},
        )

        # Get usage stats
        stats = self.resource_manager.get_resource_usage_stats()

        assert "stats_test" in stats
        resource_stats = stats["stats_test"]

        assert resource_stats["capacity"]["cores"] == 10
        assert resource_stats["available"]["cores"] == 7
        assert resource_stats["usage_percent"]["cores"] == 30.0
        assert resource_stats["usage_percent"]["memory_gb"] == 40.0


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

        self.engine = CoordinationEngine(
            registry_db=self.temp_dbs[0],
            scheduler_db=self.temp_dbs[1],
            resource_db=self.temp_dbs[2],
            coordination_db=self.temp_dbs[3],
        )

    def teardown_method(self):
        """Cleanup after each test."""
        for db_path in self.temp_dbs:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

    def test_engine_initialization(self):
        """Test coordination engine initialization."""
        assert isinstance(self.engine.registry, AgentRegistry)
        assert isinstance(self.engine.scheduler, TaskScheduler)
        assert isinstance(self.engine.broker, MessageBroker)
        assert isinstance(self.engine.resource_manager, ResourceManager)
        assert self.engine.running is False

    def test_agent_onboarding(self):
        """Test complete agent onboarding process."""
        agent = Agent(
            agent_id="onboard_test",
            name="Onboarding Test Agent",
            agent_type="worker",
            capabilities=["data_processing", "file_operations"],
            endpoint="http://localhost:8001",
        )

        # Onboard agent
        success = self.engine.onboard_agent(agent)

        assert success is True

        # Agent should be registered
        assert agent.agent_id in self.engine.registry.agents

        # Agent should be subscribed to relevant message types
        assert agent.agent_id in self.engine.broker.subscriptions.get(
            MessageType.TASK_ASSIGNMENT, set()
        )
        assert agent.agent_id in self.engine.broker.subscriptions.get(
            MessageType.SYSTEM_NOTIFICATION, set()
        )

    def test_task_lifecycle_management(self):
        """Test complete task lifecycle management."""
        # First onboard an agent
        agent = Agent(
            agent_id="lifecycle_agent",
            name="Lifecycle Agent",
            agent_type="worker",
            capabilities=["data_processing"],
            endpoint="http://localhost:8001",
        )

        self.engine.onboard_agent(agent)

        # Submit a task
        task = Task(
            task_id="lifecycle_task",
            name="Lifecycle Task",
            task_type="data_processing",
            priority=TaskPriority.HIGH,
            required_capabilities=["data_processing"],
            parameters={"input": "test_data"},
            created_by="user123",
        )

        success = self.engine.submit_task(task)
        assert success is True

        # Task should be in scheduler
        assert task.task_id in self.engine.scheduler.task_queue

        # Process coordination step (would normally assign task to agent)
        available_agents = self.engine.registry.find_agents_by_capability(
            "data_processing"
        )
        assert len(available_agents) == 1

        next_tasks = self.engine.scheduler.get_next_tasks("data_processing", limit=1)
        assert len(next_tasks) == 1
        assert next_tasks[0].task_id == "lifecycle_task"

    def test_resource_aware_scheduling(self):
        """Test resource-aware task scheduling."""
        # Register a resource
        resource = Resource(
            resource_id="gpu_resource",
            name="GPU Resource",
            resource_type=ResourceType.GPU,
            capacity={"memory_gb": 16, "cores": 2048},
            available={"memory_gb": 16, "cores": 2048},
        )

        self.engine.resource_manager.register_resource(resource)

        # Register an agent
        agent = Agent(
            agent_id="gpu_agent",
            name="GPU Agent",
            agent_type="gpu_worker",
            capabilities=["ml_training"],
            endpoint="http://localhost:8001",
        )

        self.engine.onboard_agent(agent)

        # Submit a resource-intensive task
        task = Task(
            task_id="gpu_task",
            name="GPU Task",
            task_type="ml_training",
            priority=TaskPriority.HIGH,
            required_capabilities=["ml_training"],
            required_resources={"gpu_memory_gb": 8},
            parameters={"model": "large_transformer"},
            created_by="ml_engineer",
        )

        success = self.engine.submit_task(task)
        assert success is True

        # Check that resource requirements are tracked
        assert task.required_resources is not None
        assert task.required_resources["gpu_memory_gb"] == 8

        # Verify resource availability for scheduling
        available_resources = self.engine.resource_manager.find_available_resources(
            resource_type=ResourceType.GPU, requirements={"memory_gb": 8}
        )

        assert len(available_resources) == 1
        assert available_resources[0].resource_id == "gpu_resource"

    def test_agent_failure_handling(self):
        """Test agent failure detection and handling."""
        # Register an agent
        agent = Agent(
            agent_id="failure_agent",
            name="Failure Agent",
            agent_type="worker",
            capabilities=["test_capability"],
            endpoint="http://localhost:8001",
            status=AgentStatus.BUSY,
        )

        self.engine.onboard_agent(agent)

        # Simulate agent failure
        success = self.engine.handle_agent_failure(
            "failure_agent", "Connection timeout"
        )

        assert success is True

        # Agent status should be updated to ERROR
        failed_agent = self.engine.registry.agents["failure_agent"]
        assert failed_agent.status == AgentStatus.ERROR

        # Any running tasks should be rescheduled (in a real implementation)
        # This would involve checking running_tasks and moving them back to queue

    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        # Add some agents and tasks for monitoring
        agent = Agent(
            agent_id="health_agent",
            name="Health Agent",
            agent_type="worker",
            capabilities=["monitoring"],
            endpoint="http://localhost:8001",
            status=AgentStatus.IDLE,
        )

        self.engine.onboard_agent(agent)

        task = Task(
            task_id="health_task",
            name="Health Task",
            task_type="monitoring",
            priority=TaskPriority.LOW,
            required_capabilities=["monitoring"],
            parameters={},
            created_by="system",
        )

        self.engine.submit_task(task)

        # Get health status
        health_status = self.engine.get_system_health()

        assert "agents" in health_status
        assert "tasks" in health_status
        assert "resources" in health_status
        assert "timestamp" in health_status

        assert health_status["agents"]["total"] == 1
        assert health_status["agents"]["idle"] == 1
        assert health_status["tasks"]["queued"] == 1
        assert health_status["tasks"]["running"] == 0

    def test_coordination_statistics(self):
        """Test coordination statistics collection."""
        # Create some sample data
        agents = [
            Agent(
                f"agent_{i}",
                f"Agent {i}",
                "worker",
                ["cap1"],
                f"http://localhost:800{i}",
            )
            for i in range(3)
        ]

        for agent in agents:
            self.engine.onboard_agent(agent)

        tasks = [
            Task(
                f"task_{i}",
                f"Task {i}",
                "test",
                TaskPriority.MEDIUM,
                ["cap1"],
                {},
                "user1",
            )
            for i in range(5)
        ]

        for task in tasks:
            self.engine.submit_task(task)

        # Get statistics
        stats = self.engine.get_coordination_stats()

        assert "agent_stats" in stats
        assert "task_stats" in stats
        assert "resource_stats" in stats
        assert "system_uptime" in stats

        assert stats["agent_stats"]["total"] == 3
        assert stats["task_stats"]["queued"] == 5
        assert stats["task_stats"]["completed"] == 0


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_multi_agent_collaboration_simulation(self):
        """Test multi-agent collaboration simulation."""
        # Create coordination engine
        temp_dbs = []
        for i in range(4):
            temp_db = tempfile.NamedTemporaryFile(delete=False)
            temp_db.close()
            temp_dbs.append(temp_db.name)

        engine = CoordinationEngine(
            registry_db=temp_dbs[0],
            scheduler_db=temp_dbs[1],
            resource_db=temp_dbs[2],
            coordination_db=temp_dbs[3],
        )

        try:
            # Register multiple specialized agents
            agents = [
                Agent(
                    "data_agent",
                    "Data Agent",
                    "data_processor",
                    ["data_cleaning", "data_validation"],
                    "http://localhost:8001",
                ),
                Agent(
                    "ml_agent",
                    "ML Agent",
                    "ml_processor",
                    ["model_training", "inference"],
                    "http://localhost:8002",
                ),
                Agent(
                    "viz_agent",
                    "Visualization Agent",
                    "visualizer",
                    ["chart_generation", "dashboard_creation"],
                    "http://localhost:8003",
                ),
            ]

            for agent in agents:
                engine.onboard_agent(agent)

            # Register resources
            resources = [
                Resource(
                    "cpu_cluster",
                    "CPU Cluster",
                    ResourceType.CPU,
                    {"cores": 32, "memory_gb": 128},
                    {"cores": 32, "memory_gb": 128},
                ),
                Resource(
                    "gpu_node",
                    "GPU Node",
                    ResourceType.GPU,
                    {"memory_gb": 32, "cores": 5120},
                    {"memory_gb": 32, "cores": 5120},
                ),
            ]

            for resource in resources:
                engine.resource_manager.register_resource(resource)

            # Submit a complex workflow of tasks
            tasks = [
                Task(
                    "data_clean",
                    "Clean Dataset",
                    "data_processing",
                    TaskPriority.HIGH,
                    ["data_cleaning"],
                    {"dataset": "raw_data.csv"},
                    "analyst",
                ),
                Task(
                    "data_validate",
                    "Validate Dataset",
                    "data_processing",
                    TaskPriority.HIGH,
                    ["data_validation"],
                    {"dataset": "cleaned_data.csv"},
                    "analyst",
                ),
                Task(
                    "train_model",
                    "Train ML Model",
                    "ml_processing",
                    TaskPriority.MEDIUM,
                    ["model_training"],
                    {"dataset": "validated_data.csv", "model_type": "xgboost"},
                    "ml_engineer",
                    required_resources={"gpu_memory_gb": 16},
                ),
                Task(
                    "create_viz",
                    "Create Visualizations",
                    "visualization",
                    TaskPriority.LOW,
                    ["chart_generation"],
                    {"data": "model_results.json"},
                    "analyst",
                ),
            ]

            for task in tasks:
                success = engine.submit_task(task)
                assert success is True

            # Verify all components working together
            agent_stats = engine.registry.get_agent_stats()
            assert agent_stats["total"] == 3
            assert agent_stats["idle"] == 3

            # Check task scheduling by capability
            data_tasks = engine.scheduler.get_next_tasks("data_cleaning", limit=10)
            ml_tasks = engine.scheduler.get_next_tasks("model_training", limit=10)
            viz_tasks = engine.scheduler.get_next_tasks("chart_generation", limit=10)

            assert len(data_tasks) == 1
            assert len(ml_tasks) == 1
            assert len(viz_tasks) == 1

            # Check resource availability for ML task
            gpu_resources = engine.resource_manager.find_available_resources(
                ResourceType.GPU, {"memory_gb": 16}
            )
            assert len(gpu_resources) == 1

            # Verify message broker is ready for coordination
            assert engine.broker.subscriptions[MessageType.TASK_ASSIGNMENT]
            assert len(engine.broker.subscriptions[MessageType.TASK_ASSIGNMENT]) == 3

            # Get overall system health
            health = engine.get_system_health()
            assert health["agents"]["total"] == 3
            assert health["tasks"]["queued"] == 4
            assert health["resources"]["total"] == 2

        finally:
            # Cleanup
            for db_path in temp_dbs:
                try:
                    os.unlink(db_path)
                except PermissionError:
                    pass


if __name__ == "__main__":
    # Run agent coordination system validation
    print("=== Testing Agent Coordination System ===")

    # Test agent registry
    print("Testing agent registry...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        registry = AgentRegistry(tmp.name)
        agent = Agent(
            "test_agent", "Test Agent", "worker", ["test_cap"], "http://localhost:8001"
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
        scheduler = TaskScheduler(tmp.name)
        task = Task(
            "test_task",
            "Test Task",
            "test",
            TaskPriority.HIGH,
            ["test_cap"],
            {},
            "user1",
        )
        success = scheduler.submit_task(task)
        next_tasks = scheduler.get_next_tasks("test_cap", limit=1)
        print(f"OK Task scheduler: submitted={success}, next_tasks={len(next_tasks)}")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test message broker
    print("Testing message broker...")
    broker = MessageBroker()
    broker.subscribe("agent1", MessageType.TASK_ASSIGNMENT)
    message = Message(
        "msg1",
        MessageType.TASK_ASSIGNMENT,
        "coordinator",
        ["agent1"],
        {"test": "data"},
        time.time(),
    )
    pub_success = broker.publish_message(message)
    messages = broker.get_messages("agent1")
    print(f"OK Message broker: published={pub_success}, received={len(messages)}")

    # Test resource manager
    print("Testing resource manager...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        rm = ResourceManager(tmp.name)
        resource = Resource(
            "test_res", "Test Resource", ResourceType.CPU, {"cores": 8}, {"cores": 8}
        )
        reg_success = rm.register_resource(resource)
        alloc_id = rm.allocate_resource("test_res", "task1", {"cores": 4})
        print(
            f"OK Resource manager: registered={reg_success}, allocated={alloc_id is not None}"
        )
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test coordination engine
    print("Testing coordination engine...")
    temp_dbs = []
    for i in range(4):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        temp_db.close()
        temp_dbs.append(temp_db.name)

    engine = CoordinationEngine(temp_dbs[0], temp_dbs[1], temp_dbs[2], temp_dbs[3])

    agent = Agent(
        "coord_agent",
        "Coordination Agent",
        "worker",
        ["coordination"],
        "http://localhost:8001",
    )
    onboard_success = engine.onboard_agent(agent)

    task = Task(
        "coord_task",
        "Coordination Task",
        "test",
        TaskPriority.MEDIUM,
        ["coordination"],
        {},
        "user1",
    )
    submit_success = engine.submit_task(task)

    health = engine.get_system_health()

    print(
        f"OK Coordination engine: onboard={onboard_success}, submit={submit_success}, health_keys={len(health)}"
    )

    # Cleanup
    for db_path in temp_dbs:
        try:
            os.unlink(db_path)
        except PermissionError:
            pass

    print("=== Agent coordination system validation completed ===")
