#!/usr/bin/env python3
"""
Unified Agent Test Suite
Combines best practices from multiple agent test files into single comprehensive suite
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project roots to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))


class TestAgentSuite(unittest.TestCase):
    """Comprehensive agent test suite combining all agent test functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent_config = {
            "agent_type": "test",
            "capabilities": ["reasoning", "planning"],
            "memory_size": 1000,
            "rag_enabled": True,
        }
        self.mock_rag = MagicMock()
        self.mock_p2p = MagicMock()

    # Agent Specialization Tests (from test_agent_specialization.py)
    def test_all_23_agents_exist(self):
        """Test that all 23 specialized agents are implemented"""
        from packages.agents.core.agent_registry import AgentRegistry

        registry = AgentRegistry()
        agent_types = registry.get_all_agent_types()

        expected_agents = [
            "King",
            "Auditor",
            "Legal",
            "Shield",
            "Sword",
            "Coordinator",
            "Gardener",
            "Magi",
            "Navigator",
            "Sustainer",
            "Curator",
            "Oracle",
            "Sage",
            "Shaman",
            "Strategist",
            "Ensemble",
            "Horticulturist",
            "Maker",
            "Banker-Economist",
            "Merchant",
            "Medic",
            "Polyglot",
            "Tutor",
        ]

        for agent_type in expected_agents:
            self.assertIn(agent_type, agent_types)

    def test_agent_capabilities(self):
        """Test that agents have correct capabilities"""
        from packages.agents.specialized.governance.king_agent import KingAgent

        king = KingAgent(self.agent_config)

        # Test King has public thought bubbles
        self.assertTrue(king.has_public_thoughts)
        self.assertFalse(king.has_encrypted_thoughts)

        # Test other capabilities
        self.assertTrue(king.can_coordinate)
        self.assertTrue(king.can_make_decisions)

    # King Agent Tests (from test_king_agent.py)
    def test_king_agent_coordination(self):
        """Test King agent coordination capabilities"""
        from packages.agents.specialized.governance.enhanced_king_agent import EnhancedKingAgent

        king = EnhancedKingAgent(self.agent_config)

        # Test task delegation
        task = {"type": "research", "priority": "high"}
        delegated_agent = king.delegate_task(task)

        self.assertIsNotNone(delegated_agent)
        self.assertIn(delegated_agent, ["Magi", "Sage", "Oracle"])

    def test_king_public_thoughts(self):
        """Test King agent's unique public thought bubbles"""
        from packages.agents.specialized.governance.enhanced_king_agent import EnhancedKingAgent

        king = EnhancedKingAgent(self.agent_config)

        thought = king.think("What should we prioritize?")

        # King's thoughts are public (unencrypted)
        self.assertFalse(thought.encrypted)
        self.assertTrue(thought.public)
        self.assertIn("priority", thought.content.lower())

    # Coordination System Tests (from test_coordination_system.py)
    @patch("packages.agents.core.agent_orchestration_system.AgentOrchestrationSystem")
    def test_multi_agent_coordination(self, mock_orchestration):
        """Test multi-agent coordination system"""
        orchestrator = mock_orchestration.return_value

        # Create multiple agents
        agents = {
            "king": MagicMock(agent_type="King"),
            "magi": MagicMock(agent_type="Magi"),
            "sage": MagicMock(agent_type="Sage"),
        }

        orchestrator.register_agents(agents)

        # Test task distribution
        task = {"type": "complex_research", "requires": ["research", "analysis"]}
        orchestrator.execute_task(task)

        orchestrator.execute_task.assert_called_once_with(task)

    # Agent Communication Tests
    def test_inter_agent_communication(self):
        """Test P2P communication between agents"""
        from packages.agents.core.base_agent_template import BaseAgent

        agent1 = BaseAgent({"agent_type": "Agent1", "p2p_enabled": True})
        agent2 = BaseAgent({"agent_type": "Agent2", "p2p_enabled": True})

        # Mock P2P network
        agent1.p2p_network = self.mock_p2p
        agent2.p2p_network = self.mock_p2p

        message = {"content": "Hello", "from": "Agent1", "to": "Agent2"}
        agent1.send_message(message)

        self.mock_p2p.send.assert_called_once_with(message)

    # RAG Integration Tests
    def test_agent_rag_integration(self):
        """Test agent integration with RAG system"""
        from packages.agents.core.base_agent_template import BaseAgent

        agent = BaseAgent(self.agent_config)
        agent.rag_system = self.mock_rag

        query = "What is the current status?"
        self.mock_rag.query.return_value = {"answer": "All systems operational"}

        response = agent.query_memory(query)

        self.mock_rag.query.assert_called_once_with(query)
        self.assertEqual(response["answer"], "All systems operational")

    # Quiet-STaR Reflection Tests
    def test_agent_reflection_with_quietstar(self):
        """Test agent personal reflection with Quiet-STaR"""
        from packages.agents.core.base_agent_template import BaseAgent

        agent = BaseAgent(self.agent_config)

        reflection = agent.reflect("Today's performance")

        # All agents except King have encrypted thoughts
        self.assertTrue(reflection.encrypted)
        self.assertIn("<|startofthought|>", reflection.raw_content)
        self.assertIn("<|endofthought|>", reflection.raw_content)

    # ADAS Self-Modification Tests
    def test_agent_self_modification(self):
        """Test agent ADAS self-modification capabilities"""
        from packages.agents.core.base_agent_template import BaseAgent

        agent = BaseAgent(self.agent_config)

        # Test architecture discovery
        agent.get_architecture()
        proposed_modifications = agent.discover_improvements()

        self.assertIsNotNone(proposed_modifications)
        self.assertTrue(len(proposed_modifications) > 0)

        # Test self-modification (in test mode, doesn't actually modify)
        agent.test_mode = True
        success = agent.apply_modifications(proposed_modifications[0])

        self.assertTrue(success)


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agent system"""

    @pytest.mark.asyncio
    async def test_agent_system_full_integration(self):
        """Test full agent system integration"""
        from packages.agents.core.agent_orchestration_system import AgentOrchestrationSystem
        from packages.agents.specialized.governance.enhanced_king_agent import EnhancedKingAgent

        # Create orchestration system
        orchestrator = AgentOrchestrationSystem()

        # Create King agent
        king = EnhancedKingAgent({"agent_type": "King"})

        # Register King as coordinator
        orchestrator.register_coordinator(king)

        # Create task
        task = {"type": "research", "query": "Analyze system performance", "priority": "high"}

        # Execute task
        result = await orchestrator.execute_async_task(task)

        self.assertIsNotNone(result)
        self.assertIn("analysis", result)

    def test_agent_p2p_mesh_integration(self):
        """Test agent integration with P2P mesh network"""
        from packages.agents.core.base_agent_template import BaseAgent

        # Create agents
        agents = [BaseAgent({"agent_type": f"Agent{i}", "p2p_enabled": True}) for i in range(5)]

        # Mock P2P mesh
        mock_mesh = MagicMock()
        for agent in agents:
            agent.p2p_network = mock_mesh

        # Test broadcast
        agents[0].broadcast_message({"content": "System update"})

        mock_mesh.broadcast.assert_called_once()

    def test_agent_mobile_optimization(self):
        """Test agent optimization for mobile deployment"""
        from packages.agents.core.base_agent_template import BaseAgent

        mobile_config = {"agent_type": "MobileAgent", "mobile_optimized": True, "max_memory_mb": 50, "cpu_only": True}

        agent = BaseAgent(mobile_config)

        # Verify mobile optimizations
        self.assertTrue(agent.mobile_optimized)
        self.assertLess(agent.memory_usage_mb, 50)
        self.assertTrue(agent.cpu_only)


if __name__ == "__main__":
    unittest.main()
