import unittest
import asyncio
from unittest.mock import Mock, patch
from agents.king.quality_assurance_layer import QualityAssuranceLayer
from agents.king.planning_and_task_management.unified_decision_maker import UnifiedDecisionMaker
from agents.king.problem_analyzer import ProblemAnalyzer
from agents.king.continuous_learner import ContinuousLearner
from agents.king.king_agent import KingAgent, KingAgentConfig
from agents.utils.task import Task as LangroidTask
from rag_system.core.pipeline import EnhancedRAGPipeline
from communications.protocol import StandardCommunicationProtocol
from langroid.vector_store.base import VectorStore

class TestIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.communication_protocol = Mock(spec=StandardCommunicationProtocol)
        self.rag_system = Mock(spec=EnhancedRAGPipeline)
        self.vector_store = Mock(spec=VectorStore)
        self.rag_config = Mock()
        
        self.king_config = KingAgentConfig(
            name="TestKingAgent",
            description="Test King Agent",
            capabilities=["task_routing", "decision_making", "agent_management", "problem_analysis", "task_management"],
            vector_store=self.vector_store,
            model="gpt-4",
            instructions="You are a test King agent."
        )
        
        self.king_agent = KingAgent(self.king_config, self.communication_protocol, self.rag_config, self.vector_store)

    @patch('agents.king.quality_assurance_layer.EudaimoniaTriangulator.get_embedding')
    async def test_end_to_end_decision_making(self, mock_get_embedding):
        # Mock the embedding function to return a fixed vector
        mock_get_embedding.return_value = [0.1] * 768

        # Set up the mocks
        self.rag_system.process_query.return_value = {"rag_info": "Test RAG info"}
        self.king_agent.agent.generate_structured_response.return_value = ["Alternative 1", "Alternative 2"]
        self.communication_protocol.send_and_wait.return_value.content = {"analysis": "Test analysis"}
        self.king_agent.llm.complete.return_value.text = "Test decision"

        # Create a test task
        task_content = "Make a decision about AI safety measures"
        task = LangroidTask(None, task_content)

        # Execute the task
        result = await self.king_agent.execute_task(task)

        # Assertions
        self.assertIn('decision', result)
        self.assertIn('eudaimonia_score', result)
        self.assertIn('rule_compliance', result)
        self.assertIn('rag_info', result)
        self.assertIn('best_alternative', result)
        self.assertIn('implementation_plan', result)

    @patch('agents.king.quality_assurance_layer.EudaimoniaTriangulator.get_embedding')
    async def test_continuous_learning(self, mock_get_embedding):
        # Mock the embedding function to return a fixed vector
        mock_get_embedding.return_value = [0.1] * 768

        # Create a test task
        task_content = "Implement a new AI feature"
        task = LangroidTask(None, task_content)

        # Execute the task multiple times
        for _ in range(5):
            result = await self.king_agent.execute_task(task)
            self.assertIn('decision', result)

        # Provide feedback
        feedback = [
            {"task_content": "Implement a new AI feature", "performance": 0.8},
            {"task_content": "Optimize AI algorithm", "performance": 0.9},
        ]
        await self.king_agent.learn_from_feedback(feedback)

        # Check if the continuous learner's learning rate has been adjusted
        self.assertNotEqual(self.king_agent.continuous_learner.learning_rate, 0.01)  # 0.01 is the default value

    @patch('agents.king.quality_assurance_layer.EudaimoniaTriangulator.get_embedding')
    async def test_evolve(self, mock_get_embedding):
        # Mock the embedding function to return a fixed vector
        mock_get_embedding.return_value = [0.1] * 768

        # Evolve the agent
        await self.king_agent.evolve()

        # Check if the evolution process has occurred
        self.assertGreater(len(self.king_agent.task_manager.get_performance_history()), 0)

    async def test_save_and_load_models(self):
        # Mock the save and load methods
        self.king_agent.coordinator.save_models = Mock()
        self.king_agent.problem_analyzer.save_models = Mock()
        self.king_agent.task_manager.save_models = Mock()

        self.king_agent.coordinator.load_models = Mock()
        self.king_agent.problem_analyzer.load_models = Mock()
        self.king_agent.task_manager.load_models = Mock()

        # Save models
        path = "/test/path"
        self.king_agent.save_models(path)

        # Assert that save methods were called
        self.king_agent.coordinator.save_models.assert_called_once_with(f"{path}/coordinator")
        self.king_agent.problem_analyzer.save_models.assert_called_once_with(f"{path}/problem_analyzer")
        self.king_agent.task_manager.save_models.assert_called_once_with(f"{path}/task_manager")

        # Load models
        self.king_agent.load_models(path)

        # Assert that load methods were called
        self.king_agent.coordinator.load_models.assert_called_once_with(f"{path}/coordinator")
        self.king_agent.problem_analyzer.load_models.assert_called_once_with(f"{path}/problem_analyzer")
        self.king_agent.task_manager.load_models.assert_called_once_with(f"{path}/task_manager")

if __name__ == '__main__':
    unittest.main()
