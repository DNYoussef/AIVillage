import importlib.util
import unittest
from unittest.mock import Mock, patch

# Skip heavy integration tests if torch is missing since they rely on the
# quality assurance layer's transformer models.
try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from agents.king.king_agent import KingAgent
from agents.unified_base_agent import UnifiedAgentConfig as KingAgentConfig
from agents.utils.task import Task as LangroidTask
from core.error_handling import StandardCommunicationProtocol
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.retrieval.vector_store import VectorStore


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.communication_protocol = Mock(spec=StandardCommunicationProtocol)
        self.rag_system = Mock(spec=EnhancedRAGPipeline)
        self.vector_store = Mock(spec=VectorStore)
        self.rag_config = Mock()

        self.king_config = KingAgentConfig(
            name="TestKingAgent",
            description="Test King Agent",
            capabilities=[
                "task_routing",
                "decision_making",
                "agent_management",
                "problem_analysis",
                "task_management",
            ],
            vector_store=self.vector_store,
            model="gpt-4",
            instructions="You are a test King agent.",
        )

        self.king_agent = KingAgent(
            self.king_config,
            self.communication_protocol,
            self.rag_config,
            self.vector_store,
        )

    @patch("agents.king.quality_assurance_layer.EudaimoniaTriangulator.get_embedding")
    async def test_end_to_end_decision_making(self, mock_get_embedding) -> None:
        # Mock the embedding function to return a fixed vector
        mock_get_embedding.return_value = [0.1] * 768

        # Set up the mocks
        self.rag_system.process_query.return_value = {"rag_info": "Test RAG info"}
        self.king_agent.agent.generate_structured_response.return_value = [
            "Alternative 1",
            "Alternative 2",
        ]
        self.communication_protocol.send_and_wait.return_value.content = {
            "analysis": "Test analysis"
        }
        self.king_agent.llm.complete.return_value.text = "Test decision"

        # Create a test task
        task_content = "Make a decision about AI safety measures"
        task = LangroidTask(None, task_content)

        # Execute the task
        result = await self.king_agent.execute_task(task)

        # Assertions
        assert "decision" in result
        assert "eudaimonia_score" in result
        assert "rule_compliance" in result
        assert "rag_info" in result
        assert "best_alternative" in result
        assert "implementation_plan" in result

    @patch("agents.king.quality_assurance_layer.EudaimoniaTriangulator.get_embedding")
    async def test_continuous_learning(self, mock_get_embedding) -> None:
        # Mock the embedding function to return a fixed vector
        mock_get_embedding.return_value = [0.1] * 768

        # Create a test task
        task_content = "Implement a new AI feature"
        task = LangroidTask(None, task_content)

        # Execute the task multiple times
        for _ in range(5):
            result = await self.king_agent.execute_task(task)
            assert "decision" in result

        # Provide feedback
        feedback = [
            {"task_content": "Implement a new AI feature", "performance": 0.8},
            {"task_content": "Optimize AI algorithm", "performance": 0.9},
        ]
        await self.king_agent.learn_from_feedback(feedback)

        # Check if the continuous learner's learning rate has been adjusted
        assert (
            self.king_agent.continuous_learner.learning_rate != 0.01
        )  # 0.01 is the default value

    @patch("agents.king.quality_assurance_layer.EudaimoniaTriangulator.get_embedding")
    async def test_evolve(self, mock_get_embedding) -> None:
        # Mock the embedding function to return a fixed vector
        mock_get_embedding.return_value = [0.1] * 768

        # Evolve the agent
        await self.king_agent.evolve()

        # Check if the evolution process has occurred
        assert len(self.king_agent.task_manager.get_performance_history()) > 0

    async def test_save_and_load_models(self) -> None:
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
        self.king_agent.coordinator.save_models.assert_called_once_with(
            f"{path}/coordinator"
        )
        self.king_agent.problem_analyzer.save_models.assert_called_once_with(
            f"{path}/problem_analyzer"
        )
        self.king_agent.task_manager.save_models.assert_called_once_with(
            f"{path}/task_manager"
        )

        # Load models
        self.king_agent.load_models(path)

        # Assert that load methods were called
        self.king_agent.coordinator.load_models.assert_called_once_with(
            f"{path}/coordinator"
        )
        self.king_agent.problem_analyzer.load_models.assert_called_once_with(
            f"{path}/problem_analyzer"
        )
        self.king_agent.task_manager.load_models.assert_called_once_with(
            f"{path}/task_manager"
        )


if __name__ == "__main__":
    unittest.main()
