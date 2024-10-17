import unittest
from unittest.mock import MagicMock, patch
from agents.king.king_agent import KingAgent
from agents.agent import Agent, AgentConfig  # Updated import
from langroid.agent.task import Task
from agents.utils.exceptions import AIVillageException
from agents.communication.protocol import StandardCommunicationProtocol

class TestKingAgentIntegration(unittest.TestCase):
    def setUp(self):
        config = AgentConfig(  # Changed from BaseAgentConfig to AgentConfig
            name="TestKing",
            description="Test King Agent",
            capabilities=["test"],
            vector_store=None,
            model="gpt-4",  # Added model parameter
            instructions="You are a test King Agent"  # Added instructions parameter
        )
        self.king_agent = KingAgent(config)
        self.communication_protocol = StandardCommunicationProtocol()

    @patch('agents.king.king_agent.KingAgent.route_task')
    async def test_execute_task_routing(self, mock_route_task):
        mock_route_task.return_value = {"result": "Task routed"}
        task = Task(self.king_agent, "Test task")
        task.type = "route_task"
        result = await self.king_agent.execute_task(task)
        self.assertEqual(result, {"result": "Task routed"})
        mock_route_task.assert_called_once_with(task)

    async def test_communication_protocol(self):
        test_message = {"content": "Test message"}
        received_message = None

        async def receive_message(task):
            nonlocal received_message
            received_message = task.content

        self.communication_protocol.subscribe("TestReceiver", receive_message)
        await self.communication_protocol.send_message("TestSender", "TestReceiver", test_message)

        self.assertEqual(received_message, test_message)

    def test_ai_village_exception(self):
        with self.assertRaises(AIVillageException):
            raise AIVillageException("Test exception")

    # Add more integration tests as needed

if __name__ == '__main__':
    unittest.main()
