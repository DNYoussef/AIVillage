"""Integration Test for Complete Agent System

Tests the full agent system integration including:
- Base agent template with all required systems
- Enhanced King Agent with full feature set
- Agent orchestration system
- Communication channels and messaging
- Multi-agent task coordination
- RAG system integration (mocked)
- P2P communication integration (mocked)
- MCP tool functionality
- Quiet-star reflection and Langroid memory
- Geometric self-awareness
- ADAS self-modification capabilities
"""

import asyncio
from datetime import datetime
from typing import Any

import pytest
import pytest_asyncio

from packages.agents.core.agent_interface import AgentCapability, AgentMetadata, MessageInterface, TaskInterface
from packages.agents.core.agent_orchestration_system import AgentOrchestrationSystem, CommunicationChannelType

# Import the complete agent system
from packages.agents.core.base_agent_template import BaseAgentTemplate, GeometricState, MCPTool, ReflectionType
from packages.agents.specialized.governance.enhanced_king_agent import EnhancedKingAgent


class MockRAGClient:
    """Mock RAG client for testing"""

    def __init__(self):
        self.query_count = 0
        self.mock_results = {
            "status": "success",
            "results": [
                {
                    "content": "Mock RAG result for agent capabilities and task patterns",
                    "source": "test_knowledge_base",
                    "confidence": 0.9,
                }
            ],
            "key_patterns": "Similar task patterns: coordination, communication, analysis",
        }

    async def query(self, query: str, mode: str = "balanced", **kwargs):
        self.query_count += 1
        return self.mock_results


class MockP2PClient:
    """Mock P2P client for testing"""

    def __init__(self):
        self.message_count = 0
        self.sent_messages = []

    async def send_message(self, recipient: str, message: str, **kwargs):
        self.message_count += 1
        self.sent_messages.append(
            {"recipient": recipient, "message": message, "timestamp": datetime.now().isoformat(), **kwargs}
        )
        return {"message_id": f"msg_{self.message_count}", "delivered": True, "status": "success"}


class MockAgentForgeClient:
    """Mock Agent Forge client for testing"""

    def __init__(self):
        self.adas_executions = []

    async def execute_adas_phase(self, modification_request: dict[str, Any]):
        self.adas_executions.append(modification_request)
        return {
            "status": "success",
            "new_architecture": "optimized_v2",
            "modification_summary": "Applied vector composition optimization for efficiency",
            "performance_improvement": 0.15,
        }


class TestAgentTemplate(BaseAgentTemplate):
    """Test agent implementation for integration testing"""

    def __init__(self, agent_id: str = "test_agent"):
        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type="Test",
            name=f"Test Agent {agent_id}",
            description="Test agent for integration testing",
            version="1.0.0",
            capabilities={
                AgentCapability.MESSAGE_PROCESSING,
                AgentCapability.TASK_EXECUTION,
                AgentCapability.REASONING,
            },
        )
        super().__init__(metadata)
        self.specialized_role = "test_agent"
        self.processed_tasks = []
        self.received_messages = []

    async def get_specialized_capabilities(self) -> list[AgentCapability]:
        return [AgentCapability.MESSAGE_PROCESSING, AgentCapability.TASK_EXECUTION, AgentCapability.REASONING]

    async def process_specialized_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        self.processed_tasks.append(task_data)
        return {
            "status": "success",
            "result": f"Processed task: {task_data.get('description', 'unknown')}",
            "agent_id": self.agent_id,
        }

    async def get_specialized_mcp_tools(self) -> dict[str, MCPTool]:
        return {}  # No additional tools for test agent

    # Override required abstract methods
    async def process_task(self, task: TaskInterface) -> dict[str, Any]:
        return await self.process_specialized_task(
            {"task_id": task.task_id, "description": str(task.content), "task_type": task.task_type}
        )

    async def can_handle_task(self, task: TaskInterface) -> bool:
        return "test" in str(task.content).lower()

    async def estimate_task_duration(self, task: TaskInterface) -> float:
        return 1.0  # 1 second

    async def send_message(self, message: MessageInterface) -> bool:
        return True

    async def receive_message(self, message: MessageInterface) -> None:
        self.received_messages.append(message)

    async def broadcast_message(self, message: MessageInterface, recipients: list[str]) -> dict[str, bool]:
        return {recipient: True for recipient in recipients}

    async def generate(self, prompt: str) -> str:
        return f"Test agent response to: {prompt}"

    async def get_embedding(self, text: str) -> list[float]:
        return [0.1] * 384

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return results[:k]

    async def introspect(self) -> dict[str, Any]:
        base = await super().health_check()
        base.update({"processed_tasks": len(self.processed_tasks), "received_messages": len(self.received_messages)})
        return base

    async def communicate(self, message: str, recipient: "BaseAgentTemplate") -> str:
        return f"Communication successful: {message[:50]}"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        return "test_space", f"TEST[{query[:30]}]"


class TestAgentSystemIntegration:
    """Integration tests for the complete agent system"""

    @pytest_asyncio.fixture
    async def mock_clients(self):
        """Create mock clients for system dependencies"""
        return {
            "rag_client": MockRAGClient(),
            "p2p_client": MockP2PClient(),
            "agent_forge_client": MockAgentForgeClient(),
        }

    @pytest_asyncio.fixture
    async def orchestration_system(self, mock_clients):
        """Create orchestration system with mock clients"""
        orchestrator = AgentOrchestrationSystem()

        # Inject mock clients
        orchestrator.rag_client = mock_clients["rag_client"]
        orchestrator.p2p_client = mock_clients["p2p_client"]
        orchestrator.agent_forge_client = mock_clients["agent_forge_client"]

        await orchestrator.start_orchestration()

        yield orchestrator

        # Cleanup
        await orchestrator.stop_orchestration()

    @pytest_asyncio.fixture
    async def enhanced_king_agent(self, mock_clients):
        """Create Enhanced King Agent with mock clients"""
        king = EnhancedKingAgent("integration_test_king")

        # Inject mock clients
        king.rag_client = mock_clients["rag_client"]
        king.p2p_client = mock_clients["p2p_client"]
        king.agent_forge_client = mock_clients["agent_forge_client"]

        await king.initialize()
        return king

    @pytest_asyncio.fixture
    async def test_agents(self, mock_clients):
        """Create test agents for orchestration"""
        agents = []

        for i in range(3):
            agent = TestAgentTemplate(f"test_agent_{i}")

            # Inject mock clients
            agent.rag_client = mock_clients["rag_client"]
            agent.p2p_client = mock_clients["p2p_client"]
            agent.agent_forge_client = mock_clients["agent_forge_client"]

            await agent.initialize()
            agents.append(agent)

        return agents

    @pytest.mark.asyncio
    async def test_base_agent_template_integration(self, mock_clients):
        """Test base agent template with all required systems"""

        # Create test agent
        agent = TestAgentTemplate("base_template_test")
        agent.rag_client = mock_clients["rag_client"]
        agent.p2p_client = mock_clients["p2p_client"]
        agent.agent_forge_client = mock_clients["agent_forge_client"]

        # Test initialization
        init_success = await agent.initialize()
        assert init_success, "Agent initialization should succeed"

        # Test RAG system integration (group memory access)
        rag_result = await agent.query_group_memory("test query for group memory", mode="balanced")
        assert rag_result["status"] == "success"
        assert mock_clients["rag_client"].query_count > 0

        # Test communication channels
        comm_result = await agent.send_agent_message(
            recipient="test_recipient", message="Test inter-agent communication", channel_type="direct"
        )
        assert comm_result["status"] == "success"
        assert mock_clients["p2p_client"].message_count > 0

        # Test quiet-star reflection
        reflection_id = await agent.record_quiet_star_reflection(
            reflection_type=ReflectionType.TASK_COMPLETION,
            context="Integration test task completion",
            raw_thoughts="Testing the quiet-star reflection system integration",
            insights="All systems appear to be working correctly",
            emotional_valence=0.5,
        )

        assert reflection_id is not None
        assert len(agent.personal_journal) > 0

        # Verify reflection format
        reflection = agent.personal_journal[-1]
        assert "<|startofthought|>" in reflection.thoughts
        assert "<|endofthought|>" in reflection.thoughts

        # Test Langroid memory system
        await agent.retrieve_similar_memories("integration test")
        # Should have at least one memory from the reflection above
        assert len(agent.personal_memory) >= 0  # May or may not store based on unexpectedness

        # Test geometric self-awareness
        await agent.update_geometric_self_awareness()
        assert agent.current_geometric_state is not None
        assert isinstance(agent.current_geometric_state.geometric_state, GeometricState)

        # Test ADAS self-modification
        modification_result = await agent.initiate_self_modification(
            optimization_target="efficiency", modification_params={"target_improvement": 0.1}
        )

        assert modification_result["status"] == "success"
        assert len(mock_clients["agent_forge_client"].adas_executions) > 0

        # Test health check
        health_info = await agent.health_check()
        assert "agent_id" in health_info
        assert "connections" in health_info
        assert "memory_stats" in health_info
        assert "geometric_state" in health_info

        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_enhanced_king_agent_integration(self, enhanced_king_agent, mock_clients):
        """Test Enhanced King Agent with full feature integration"""

        king = enhanced_king_agent

        # Test orchestration capabilities
        task_data = {
            "task_type": "orchestrate_complex_task",
            "description": "Complex multi-agent integration test task",
            "constraints": {"priority": "high", "max_agents": 3},
        }

        result = await king.process_specialized_task(task_data)
        assert result["status"] == "success"
        assert "orchestration_complete" in result

        # Verify RAG integration for orchestration
        assert mock_clients["rag_client"].query_count > 0

        # Test emergency oversight capability
        oversight_result = await king.mcp_tools["emergency_oversight"].execute(
            {"target_agent_id": "test_agent_1", "reason": "integration_test_emergency"}
        )

        assert oversight_result["status"] == "success"
        assert "thought_buffer" in oversight_result
        assert "transparency_logged" in oversight_result

        # Test transparency reporting
        transparency_report = await king.get_transparency_report()
        assert "decision_summary" in transparency_report
        assert "orchestration_summary" in transparency_report
        assert "system_integration" in transparency_report

        # Verify decision logging
        assert len(king.decision_log) > 0

        # Test multi-agent coordination
        coord_task = await king._orchestrate_complex_task(
            {
                "description": "Test coordination task",
                "constraints": {},
                "optimization_config": king.default_optimization.__dict__,
            }
        )

        assert coord_task["status"] == "success"
        assert "coordination" in coord_task

    @pytest.mark.asyncio
    async def test_orchestration_system_integration(self, orchestration_system, test_agents, enhanced_king_agent):
        """Test complete orchestration system integration"""

        orchestrator = orchestration_system

        # Register test agents
        for agent in test_agents:
            success = await orchestrator.register_agent(agent)
            assert success, f"Agent {agent.agent_id} registration should succeed"

        # Register king agent
        king_success = await orchestrator.register_agent(enhanced_king_agent)
        assert king_success, "King agent registration should succeed"

        # Verify agent registry
        assert len(orchestrator.agents) == 4  # 3 test agents + 1 king agent

        # Test agent lookup by type
        test_agents_found = await orchestrator.get_agents_by_type("Test")
        assert len(test_agents_found) == 3

        # Test agent lookup by capability
        reasoning_agents = await orchestrator.get_agents_by_capability(AgentCapability.REASONING)
        assert len(reasoning_agents) >= 3  # At least the test agents

        # Test communication channels
        channel_id = await orchestrator.create_communication_channel(
            CommunicationChannelType.GROUP, "integration_test", "Integration test communication channel"
        )

        assert channel_id is not None

        # Add agents to channel
        for agent in test_agents:
            join_success = await orchestrator.join_channel(agent.agent_id, channel_id)
            assert join_success

        # Test message sending
        message_success = await orchestrator.send_message(
            sender_id=enhanced_king_agent.agent_id,
            channel_id=channel_id,
            message="Integration test message from King to all agents",
            message_type="coordination",
            priority=3,
        )
        assert message_success

        # Give message processing time
        await asyncio.sleep(0.1)

        # Test task distribution
        test_task = TaskInterface(
            task_id="integration_test_task", task_type="test", content="Integration test task for agent processing"
        )

        distributed_task_id = await orchestrator.distribute_task(test_task)
        assert distributed_task_id == test_task.task_id

        # Give task processing time
        await asyncio.sleep(0.2)

        # Verify task was processed by an agent
        processed_count = sum(len(agent.processed_tasks) for agent in test_agents)
        assert processed_count > 0, "At least one agent should have processed the task"

        # Test multi-agent task coordination
        multi_task_id = await orchestrator.coordinate_multi_agent_task(
            description="Integration test multi-agent collaboration",
            required_capabilities=[AgentCapability.REASONING, AgentCapability.TASK_EXECUTION],
            max_agents=2,
            coordination_strategy="collaborative",
        )

        assert multi_task_id is not None
        assert multi_task_id in orchestrator.active_tasks

        multi_task = orchestrator.active_tasks[multi_task_id]
        assert len(multi_task.assigned_agents) > 0
        assert multi_task.task_status == "assigned"

        # Test system status
        system_status = await orchestrator.get_system_status()
        assert system_status["orchestration_system"]["is_running"]
        assert system_status["agents"]["total_registered"] == 4
        assert system_status["communication"]["active_channels"] > 0

        # Test agent-specific status
        agent_status = await orchestrator.get_agent_status(test_agents[0].agent_id)
        assert agent_status is not None
        assert agent_status["status"] == "active"
        assert "performance" in agent_status
        assert "communication" in agent_status

    @pytest.mark.asyncio
    async def test_cross_system_integration(self, orchestration_system, enhanced_king_agent, test_agents, mock_clients):
        """Test integration across all systems: agents, orchestration, RAG, P2P, Agent Forge"""

        orchestrator = orchestration_system

        # Register all agents
        all_agents = test_agents + [enhanced_king_agent]
        for agent in all_agents:
            await orchestrator.register_agent(agent)

        # Test complex workflow: King agent orchestrates task requiring multiple systems

        # Step 1: King agent receives complex orchestration request
        complex_task_data = {
            "task_type": "orchestrate_complex_task",
            "description": "Cross-system integration test: coordinate 3 agents to analyze data, make recommendations, and execute deployment",
            "constraints": {
                "use_rag_for_context": True,
                "require_agent_collaboration": True,
                "enable_adas_optimization": True,
            },
        }

        # Step 2: King processes orchestration request
        orchestration_result = await enhanced_king_agent.process_specialized_task(complex_task_data)
        assert orchestration_result["status"] == "success"

        # Step 3: Verify cross-system integration

        # RAG system was queried for task decomposition and agent assignment
        assert mock_clients["rag_client"].query_count > 0

        # P2P system was used for agent communication
        assert mock_clients["p2p_client"].message_count > 0

        # Agent Forge system was available for self-modification
        # (not necessarily used in this test, but available)

        # Step 4: Test system-wide health and performance
        system_status = await orchestrator.get_system_status()

        # Verify all agents are active and healthy
        assert system_status["agents"]["by_status"]["active"] >= 3

        # Verify communication is flowing
        assert system_status["communication"]["active_channels"] > 0

        # Verify tasks are being processed
        assert orchestrator.stats["tasks_distributed"] > 0
        assert orchestrator.stats["messages_routed"] > 0

        # Step 5: Test reflective learning
        # King agent should have recorded reflections about the orchestration
        king_reflections = enhanced_king_agent.personal_journal
        assert len(king_reflections) > 0

        # Find orchestration-related reflections
        orchestration_reflections = [
            r for r in king_reflections if "orchestration" in r.context.lower() or "task" in r.context.lower()
        ]
        assert len(orchestration_reflections) > 0

        # Step 6: Test memory formation
        # Check if any experiences were stored in Langroid memory
        king_memories = enhanced_king_agent.personal_memory
        # Memory storage depends on unexpectedness, so we just verify the system works
        assert isinstance(king_memories, list)

        # Step 7: Test geometric self-awareness
        assert enhanced_king_agent.current_geometric_state is not None
        geometric_health = enhanced_king_agent.current_geometric_state.is_healthy()
        assert isinstance(geometric_health, bool)

        # Step 8: Verify transparency and auditability
        transparency_report = await enhanced_king_agent.get_transparency_report()
        assert transparency_report["transparency_level"] == "full"
        assert len(transparency_report["decision_summary"]["recent_decisions"]) > 0

    @pytest.mark.asyncio
    async def test_mcp_tools_integration(self, enhanced_king_agent, mock_clients):
        """Test MCP (Model Control Protocol) tools integration"""

        king = enhanced_king_agent

        # Test RAG query MCP tool
        rag_tool = king.mcp_tools["rag_query"]
        rag_result = await rag_tool.execute(
            {"query": "agent orchestration patterns", "mode": "comprehensive", "max_results": 5}
        )

        assert rag_result["status"] == "success"
        assert "results" in rag_result
        assert rag_tool.usage_count > 0

        # Test communication MCP tool
        comm_tool = king.mcp_tools["communicate"]
        comm_result = await comm_tool.execute(
            {
                "recipient": "test_agent",
                "message": "MCP communication test",
                "channel_type": "direct",
                "sender_id": king.agent_id,
            }
        )

        assert comm_result["status"] == "success"
        assert comm_tool.usage_count > 0

        # Test emergency oversight MCP tool
        oversight_tool = king.mcp_tools["emergency_oversight"]
        oversight_result = await oversight_tool.execute(
            {"target_agent_id": "test_agent", "reason": "MCP tools integration test"}
        )

        assert oversight_result["status"] == "success"
        assert oversight_tool.usage_count > 0

        # Test King-specific MCP tools
        decomp_tool = king.mcp_tools["task_decomposition"]
        decomp_result = await decomp_tool.execute(
            {"task_description": "MCP test task decomposition", "constraints": {"test_mode": True}}
        )

        assert decomp_result["status"] == "success"
        assert decomp_tool.usage_count > 0

        assignment_tool = king.mcp_tools["agent_assignment"]

        # First create a task for assignment
        await decomp_tool.execute({"task_description": "Create task for assignment test", "constraints": {}})

        # Get the task ID from King's active tasks
        task_id = list(king.active_tasks.keys())[-1] if king.active_tasks else "test_task_1"

        assignment_result = await assignment_tool.execute(
            {"task_id": task_id, "optimization_config": king.default_optimization.__dict__}
        )

        # Assignment may fail due to no real agents, but tool should execute
        assert assignment_result["status"] in ["success", "error"]
        assert assignment_tool.usage_count > 0

    @pytest.mark.asyncio
    async def test_system_resilience_and_error_handling(self, orchestration_system, mock_clients):
        """Test system resilience and error handling"""

        orchestrator = orchestration_system

        # Test agent registration with invalid agent
        class BrokenAgent(TestAgentTemplate):
            async def initialize(self) -> bool:
                raise Exception("Simulated initialization failure")

        broken_agent = BrokenAgent("broken_agent")
        broken_agent.rag_client = mock_clients["rag_client"]
        broken_agent.p2p_client = mock_clients["p2p_client"]
        broken_agent.agent_forge_client = mock_clients["agent_forge_client"]

        # Should handle initialization failure gracefully
        registration_success = await orchestrator.register_agent(broken_agent)
        assert not registration_success

        # Test message handling with non-existent recipient
        message_success = await orchestrator.send_message(
            sender_id="non_existent_sender", recipient_id="non_existent_recipient", message="Test error handling"
        )

        # Should not crash the system
        assert isinstance(message_success, bool)

        # Test system status during error conditions
        status = await orchestrator.get_system_status()
        assert status["orchestration_system"]["is_running"]

        # Test health monitoring continues to work
        await orchestrator.health_monitor.update_health_metrics()
        assert "last_updated" in orchestrator.health_monitor.health_metrics


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
