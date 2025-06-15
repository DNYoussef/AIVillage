import unittest
import asyncio
from unittest.mock import Mock, patch
import pytest

pytest.skip("Skipping King agent tests due to missing dependencies", allow_module_level=True)
from agents.king.planning_and_task_management.unified_decision_maker import UnifiedDecisionMaker
from agents.king.quality_assurance_layer import QualityAssuranceLayer
from agents.utils.task import Task as LangroidTask

class TestUnifiedDecisionMaker(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.communication_protocol = Mock()
        self.rag_system = Mock()
        self.agent = Mock()
        self.quality_assurance_layer = Mock(spec=QualityAssuranceLayer)
        self.decision_maker = UnifiedDecisionMaker(self.communication_protocol, self.rag_system, self.agent, self.quality_assurance_layer)

    async def test_make_decision(self):
        content = "Test decision content"
        eudaimonia_score = 0.8
        self.rag_system.process_query.return_value = {"rag_info": "Test RAG info"}
        self.quality_assurance_layer.eudaimonia_triangulator.get_embedding.return_value = [0.1] * 768
        self.quality_assurance_layer.evaluate_rule_compliance.return_value = 0.9
        self.agent.generate_structured_response.return_value = ["Alternative 1", "Alternative 2"]
        self.decision_maker.llm.complete.return_value.text = "Test decision"

        result = await self.decision_maker.make_decision(content, eudaimonia_score)

        self.assertIn("decision", result)
        self.assertIn("eudaimonia_score", result)
        self.assertIn("rule_compliance", result)
        self.assertIn("rag_info", result)
        self.assertIn("best_alternative", result)
        self.assertIn("implementation_plan", result)

    async def test_update_model(self):
        task = {"content": "Test task"}
        result = {"performance": 0.7, "uncertainty": 0.3}
        await self.decision_maker.update_model(task, result)
        self.decision_maker.mcts.update.assert_called_once_with(task, result)
        self.quality_assurance_layer.update_task_history.assert_called_once_with(task, 0.7, 0.3)

    async def test_save_models(self):
        path = "test_path"
        await self.decision_maker.save_models(path)
        self.decision_maker.mcts.save.assert_called_once_with(f"{path}/mcts_model.pt")

    async def test_load_models(self):
        path = "test_path"
        await self.decision_maker.load_models(path)
        self.decision_maker.mcts.load.assert_called_once_with(f"{path}/mcts_model.pt")

    def test_update_agent_list(self):
        agent_list = ["Agent1", "Agent2"]
        self.decision_maker.update_agent_list(agent_list)
        self.assertEqual(self.decision_maker.available_agents, agent_list)

    async def test_generate_alternatives(self):
        problem_analysis = {"content": "Test problem"}
        self.agent.generate_structured_response.return_value = ["Alt1", "Alt2"]
        self.communication_protocol.send_and_wait.return_value.content = {"alternatives": ["Alt3"]}

        alternatives = await self.decision_maker._generate_alternatives(problem_analysis)

        self.assertIn("Alt1", alternatives)
        self.assertIn("Alt2", alternatives)
        self.assertIn("Alt3", alternatives)

    async def test_evaluate_alternatives(self):
        alternatives = ["Alt1", "Alt2"]
        ranked_criteria = [{"criterion": "eudaimonia", "weight": 0.5}, {"criterion": "curiosity", "weight": 0.5}]
        self.quality_assurance_layer.eudaimonia_triangulator.get_embedding.return_value = [0.1] * 768
        self.quality_assurance_layer.eudaimonia_triangulator.triangulate.return_value = 0.8
        self.quality_assurance_layer.evaluate_rule_compliance.return_value = 0.9

        evaluated = await self.decision_maker._evaluate_alternatives(alternatives, ranked_criteria)

        self.assertEqual(len(evaluated), 2)
        self.assertIn("alternative", evaluated[0])
        self.assertIn("score", evaluated[0])

    async def test_create_implementation_plan(self):
        plan = {"decision": "Test decision", "best_alternative": "Test alternative"}
        self.agent.generate_structured_response.return_value = {"monitoring": ["Step1"], "feedback_analysis": ["Step2"]}

        implementation_plan = await self.decision_maker._create_implementation_plan(plan)

        self.assertIn("monitoring", implementation_plan)
        self.assertIn("feedback_analysis", implementation_plan)

if __name__ == '__main__':
    unittest.main()
