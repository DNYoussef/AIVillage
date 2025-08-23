import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.king.king_agent import KingAgent
from agents.unified_base_agent import UnifiedAgentConfig as KingAgentConfig
from agents.utils.task import Task as LangroidTask
from rag_system.core.config import RAGConfig
from rag_system.retrieval.vector_store import VectorStore

from core.error_handling import StandardCommunicationProtocol

# Skip these tests if PyTorch isn't installed since KingAgent relies on
# transformer models.
try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

# Ensure the repository root is on the Python path so that the ``agents``
# package imports correctly when running this test in isolation.
sys.path.append(str(Path(__file__).resolve().parents[1]))


class TestKingAgent(unittest.TestCase):
    def setUp(self):
        self.config = KingAgentConfig(name="TestKingAgent", description="Test Agent")
        self.communication_protocol = StandardCommunicationProtocol()
        self.rag_config = RAGConfig()
        self.vector_store = MagicMock(spec=VectorStore)
        self.king_agent = KingAgent(self.config, self.communication_protocol, self.vector_store)

    @patch("agents.unified_base_agent.DecisionMakingLayer._get_preferences")
    @patch("agents.utils.mcts.MonteCarloTreeSearch.search")
    async def test_decision_layer(self, mock_search, mock_prefs):
        mock_search.return_value = "Option B"
        mock_prefs.return_value = {"Approach X": 0.1, "Approach Y": 0.9}
        dm = self.king_agent.decision_making_layer

        async def fake_complete(prompt):
            return type("Resp", (object,), {"text": "final"})

        dm.llm.complete = fake_complete
        task = LangroidTask(self.king_agent, "demo", "1", 1)
        decision = await dm.make_decision(task, "ctx")
        assert decision == "final"
        mock_search.assert_called_once()
        mock_prefs.assert_called_once()

    @patch("agents.king.user_intent_interpreter.UserIntentInterpreter.interpret_intent")
    @patch("agents.king.key_concept_extractor.KeyConceptExtractor.extract_key_concepts")
    @patch("agents.king.task_planning_agent.TaskPlanningAgent.generate_task_plan")
    @patch("agents.king.knowledge_graph_agent.KnowledgeGraphAgent.query_graph")
    @patch("agents.king.reasoning_agent.ReasoningAgent.perform_reasoning")
    @patch("agents.king.response_generation_agent.ResponseGenerationAgent.generate_response")
    @patch("agents.king.dynamic_knowledge_integration_agent.DynamicKnowledgeIntegrationAgent.integrate_new_knowledge")
    async def test_process_user_input(
        self,
        mock_integrate,
        mock_generate_response,
        mock_reasoning,
        mock_query_graph,
        mock_task_plan,
        mock_extract_concepts,
        mock_interpret_intent,
    ):
        # Set up mock return values
        mock_interpret_intent.return_value = {"primary_intent": "test_intent"}
        mock_extract_concepts.return_value = {"key_concept": "test_concept"}
        mock_task_plan.return_value = {"task": "test_task"}
        mock_query_graph.return_value = {"graph_data": "test_data"}
        mock_reasoning.return_value = {"reasoning_result": "test_reasoning"}
        mock_generate_response.return_value = "Test response"
        mock_integrate.return_value = None

        # Call the method
        result = await self.king_agent.process_user_input("Test input")

        # Assert the result
        assert "interpreted_intent" in result
        assert "key_concepts" in result
        assert "task_plan" in result
        assert "reasoning_result" in result
        assert "response" in result

        # Assert that all mocked methods were called
        mock_interpret_intent.assert_called_once()
        mock_extract_concepts.assert_called_once()
        mock_task_plan.assert_called_once()
        mock_query_graph.assert_called_once()
        mock_reasoning.assert_called_once()
        mock_generate_response.assert_called_once()
        mock_integrate.assert_called_once()

    # Add more test methods for other KingAgent functionalities


def run_async_test(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()
