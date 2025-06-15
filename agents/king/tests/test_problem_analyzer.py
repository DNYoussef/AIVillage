import unittest
import asyncio
from unittest.mock import Mock, patch
import pytest

pytest.skip("Skipping King agent tests due to missing dependencies", allow_module_level=True)
from agents.king.planning.problem_analyzer import ProblemAnalyzer
from agents.king.quality_assurance_layer import QualityAssuranceLayer
from agents.utils.task import Task as LangroidTask

class TestProblemAnalyzer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.communication_protocol = Mock()
        self.agent = Mock()
        self.quality_assurance_layer = Mock(spec=QualityAssuranceLayer)
        self.problem_analyzer = ProblemAnalyzer(self.communication_protocol, self.agent, self.quality_assurance_layer)

    async def test_analyze(self):
        content = "Test problem content"
        rag_info = {"rag_data": "Test RAG info"}
        rule_compliance = 0.9

        self.quality_assurance_layer.eudaimonia_triangulator.get_embedding.return_value = [0.1] * 768
        self.quality_assurance_layer.eudaimonia_triangulator.triangulate.return_value = 0.8
        self.problem_analyzer.llm.complete.return_value.text = "Test analysis"

        result = await self.problem_analyzer.analyze(content, rag_info, rule_compliance)

        self.assertIn("analysis", result)
        self.assertIn("rule_compliance", result)
        self.assertIn("eudaimonia_score", result)
        self.assertIn("rag_info", result)

    async def test_collect_agent_analyses(self):
        task = "Test task"
        self.communication_protocol.get_all_agents.return_value = ["Agent1", "Agent2"]
        self.communication_protocol.send_and_wait.return_value.content = {"analysis": "Test analysis"}

        analyses = await self.problem_analyzer._collect_agent_analyses(task)

        self.assertEqual(len(analyses), 2)
        self.assertIn("agent", analyses[0])
        self.assertIn("analysis", analyses[0])

    async def test_collect_critiqued_analyses(self):
        initial_analyses = [
            {"agent": "Agent1", "analysis": "Analysis 1"},
            {"agent": "Agent2", "analysis": "Analysis 2"}
        ]
        self.communication_protocol.send_and_wait.return_value.content = {"critique": "Test critique"}

        critiqued_analyses = await self.problem_analyzer._collect_critiqued_analyses(initial_analyses)

        self.assertEqual(len(critiqued_analyses), 2)
        self.assertIn("critiques", critiqued_analyses[0])

    async def test_collect_revised_analyses(self):
        critiqued_analyses = [
            {"agent": "Agent1", "analysis": "Analysis 1", "critiques": ["Critique 1"]},
            {"agent": "Agent2", "analysis": "Analysis 2", "critiques": ["Critique 2"]}
        ]
        self.communication_protocol.send_and_wait.return_value.content = {"revised_analysis": "Revised analysis"}

        revised_analyses = await self.problem_analyzer._collect_revised_analyses(critiqued_analyses)

        self.assertEqual(len(revised_analyses), 2)
        self.assertIn("revised_analysis", revised_analyses[0])

    async def test_consolidate_analyses(self):
        revised_analyses = [
            {"agent": "Agent1", "revised_analysis": "Revised analysis 1"},
            {"agent": "Agent2", "revised_analysis": "Revised analysis 2"}
        ]
        king_analysis = "King's analysis"
        self.problem_analyzer.llm.complete.return_value.text = "Consolidated analysis"

        consolidated_analysis = await self.problem_analyzer._consolidate_analyses(revised_analyses, king_analysis)

        self.assertEqual(consolidated_analysis, "Consolidated analysis")

    async def test_update_models(self):
        task = {"content": "Test task"}
        result = {"performance": 0.7, "uncertainty": 0.3}
        await self.problem_analyzer.update_models(task, result)
        self.problem_analyzer.enhanced_plan_generator.update.assert_called_once_with(task, result)
        self.quality_assurance_layer.update_task_history.assert_called_once_with(task, 0.7, 0.3)

    async def test_save_models(self):
        path = "test_path"
        await self.problem_analyzer.save_models(path)
        self.problem_analyzer.enhanced_plan_generator.save.assert_called_once_with(path)

    async def test_load_models(self):
        path = "test_path"
        await self.problem_analyzer.load_models(path)
        self.problem_analyzer.enhanced_plan_generator.load.assert_called_once_with(path)

    async def test_introspect(self):
        self.quality_assurance_layer.get_info.return_value = {"qa_info": "Test QA info"}
        introspection = await self.problem_analyzer.introspect()
        self.assertIn("type", introspection)
        self.assertIn("description", introspection)
        self.assertIn("quality_assurance_info", introspection)

if __name__ == '__main__':
    unittest.main()
