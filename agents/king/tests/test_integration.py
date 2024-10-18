import unittest
from unittest.mock import Mock, patch, AsyncMock
from agents.unified_base_agent import UnifiedBaseAgent as Agent
from agents.king.king_agent import KingAgent, KingAgentConfig
from agents.king.coordinator import KingCoordinator
from agents.king.decision_maker import DecisionMaker
from agents.king.problem_analyzer import ProblemAnalyzer
from agents.king.unified_task_manager import UnifiedTaskManager
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from agents.utils.exceptions import AIVillageException
from rag_system.core.pipeline import EnhancedRAGPipeline as RAGSystem

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_communication_protocol = AsyncMock(spec=StandardCommunicationProtocol)
        self.mock_rag_system = AsyncMock(spec=RAGSystem)
        self.config = KingAgentConfig(name="TestKing", description="Test King Agent", model="gpt-4")
        self.king_agent = KingAgent(self.config, self.mock_communication_protocol, self.mock_rag_system)

    @patch('agents.king.coordinator.KingCoordinator.handle_task_message')
    async def test_execute_task_routing(self, mock_handle_task_message):
        mock_handle_task_message.return_value = {"result": "Task routed"}
        task = Mock()
        task.type = "route_task"
        task.content = {"description": "Test task"}
        result = await self.king_agent.execute_task(task)
        self.assertEqual(result, {"result": "Task routed"})
        mock_handle_task_message.assert_called_once_with(task)

    async def test_communication_protocol(self):
        test_message = Message(type=MessageType.TASK, sender="TestSender", receiver="TestReceiver", content={"description": "Test message"})
        await self.mock_communication_protocol.send_message(test_message)
        self.mock_communication_protocol.send_message.assert_called_once_with(test_message)

    def test_ai_village_exception(self):
        with self.assertRaises(AIVillageException):
            raise AIVillageException("Test exception")

    @patch('agents.king.decision_maker.DecisionMaker.make_decision')
    async def test_decision_maker_integration(self, mock_make_decision):
        mock_make_decision.return_value = {"decision": "Test decision"}
        task = Mock()
        task.type = "make_decision"
        task.content = "Test decision task"
        result = await self.king_agent.execute_task(task)
        self.assertEqual(result, {"decision": "Test decision"})
        mock_make_decision.assert_called_once_with("Test decision task")

    @patch('agents.king.problem_analyzer.ProblemAnalyzer.analyze')
    async def test_problem_analyzer_integration(self, mock_analyze):
        mock_analyze.return_value = {"analysis": "Test analysis"}
        task = Mock()
        task.type = "analyze_problem"
        task.content = "Test problem"
        result = await self.king_agent.execute_task(task)
        self.assertEqual(result, {"analysis": "Test analysis"})
        mock_analyze.assert_called_once_with("Test problem", {})

    @patch('agents.king.unified_task_manager.UnifiedTaskManager.create_task')
    async def test_task_manager_integration(self, mock_create_task):
        mock_create_task.return_value = Mock(id="test_task_id")
        task = Mock()
        task.type = "manage_task"
        task.content = {"action": "create", "description": "Test task", "agent": "test_agent"}
        result = await self.king_agent.execute_task(task)
        self.assertEqual(result, {"task_id": "test_task_id"})
        mock_create_task.assert_called_once_with("Test task", "test_agent")

    @patch('agents.king.coordinator.KingCoordinator.add_agent')
    async def test_add_agent(self, mock_add_agent):
        mock_agent = Mock()
        await self.king_agent.coordinator.add_agent("test_agent", mock_agent)
        mock_add_agent.assert_called_once_with("test_agent", mock_agent)

    @patch('rag_system.core.pipeline.EnhancedRAGPipeline.process_query')
    async def test_rag_system_integration(self, mock_process_query):
        mock_process_query.return_value = {"rag_result": "Test RAG result"}
        task = Mock()
        task.type = "analyze_problem"
        task.content = "Test problem"
        await self.king_agent.execute_task(task)
        mock_process_query.assert_called_once_with("Test problem")

    async def test_introspection(self):
        introspection_result = await self.king_agent.introspect()
        self.assertIn("coordinator_capabilities", introspection_result)
        self.assertIn("coordinator_info", introspection_result)
        self.assertIn("decision_maker_info", introspection_result)
        self.assertIn("problem_analyzer_info", introspection_result)
        self.assertIn("task_manager_info", introspection_result)

if __name__ == '__main__':
    unittest.main()

