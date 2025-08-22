import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch


class DummyTok:
    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]])}


class DummyModel:
    def __call__(self, *args, **kwargs):
        class Output:
            def __init__(self) -> None:
                self.last_hidden_state = torch.zeros((1, 1, 768))

        return Output()


# Skip if PyTorch is not installed since the quality assurance layer imports
# transformer models that require it.
try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from agents.king.planning.problem_analyzer import ProblemAnalyzer
from agents.king.quality_assurance_layer import QualityAssuranceLayer


class TestProblemAnalyzer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tok_patch = patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTok())
        model_patch = patch("transformers.AutoModel.from_pretrained", return_value=DummyModel())
        self.addCleanup(tok_patch.stop)
        self.addCleanup(model_patch.stop)
        tok_patch.start()
        model_patch.start()
        self.communication_protocol = Mock()
        self.agent = Mock()
        self.quality_assurance_layer = Mock(spec=QualityAssuranceLayer)
        self.problem_analyzer = ProblemAnalyzer(self.communication_protocol, self.agent, self.quality_assurance_layer)

    async def test_analyze(self) -> None:
        content = "Test problem content"
        rag_info = {"rag_data": "Test RAG info"}
        rule_compliance = 0.9

        self.quality_assurance_layer.eudaimonia_triangulator.get_embedding.return_value = [0.1] * 768
        self.quality_assurance_layer.eudaimonia_triangulator.triangulate.return_value = 0.8
        self.problem_analyzer.llm.complete.return_value.text = "Test analysis"

        result = await self.problem_analyzer.analyze(content, rag_info, rule_compliance)

        assert "analysis" in result
        assert "rule_compliance" in result
        assert "eudaimonia_score" in result
        assert "rag_info" in result

    async def test_collect_agent_analyses(self) -> None:
        task = "Test task"
        self.communication_protocol.get_all_agents.return_value = ["Agent1", "Agent2"]
        self.communication_protocol.send_and_wait.return_value.content = {"analysis": "Test analysis"}

        analyses = await self.problem_analyzer._collect_agent_analyses(task)

        assert len(analyses) == 2
        assert "agent" in analyses[0]
        assert "analysis" in analyses[0]

    async def test_collect_critiqued_analyses(self) -> None:
        initial_analyses = [
            {"agent": "Agent1", "analysis": "Analysis 1"},
            {"agent": "Agent2", "analysis": "Analysis 2"},
        ]
        self.communication_protocol.send_and_wait.return_value.content = {"critique": "Test critique"}

        critiqued_analyses = await self.problem_analyzer._collect_critiqued_analyses(initial_analyses)

        assert len(critiqued_analyses) == 2
        assert "critiques" in critiqued_analyses[0]

    async def test_collect_revised_analyses(self) -> None:
        critiqued_analyses = [
            {"agent": "Agent1", "analysis": "Analysis 1", "critiques": ["Critique 1"]},
            {"agent": "Agent2", "analysis": "Analysis 2", "critiques": ["Critique 2"]},
        ]
        self.communication_protocol.send_and_wait.return_value.content = {"revised_analysis": "Revised analysis"}

        revised_analyses = await self.problem_analyzer._collect_revised_analyses(critiqued_analyses)

        assert len(revised_analyses) == 2
        assert "revised_analysis" in revised_analyses[0]

    async def test_consolidate_analyses(self) -> None:
        revised_analyses = [
            {"agent": "Agent1", "revised_analysis": "Revised analysis 1"},
            {"agent": "Agent2", "revised_analysis": "Revised analysis 2"},
        ]
        king_analysis = "King's analysis"
        self.problem_analyzer.llm.complete.return_value.text = "Consolidated analysis"

        consolidated_analysis = await self.problem_analyzer._consolidate_analyses(revised_analyses, king_analysis)

        assert consolidated_analysis == "Consolidated analysis"

    async def test_update_models(self) -> None:
        task = {"content": "Test task"}
        result = {"performance": 0.7, "uncertainty": 0.3}
        await self.problem_analyzer.update_models(task, result)
        self.problem_analyzer.enhanced_plan_generator.update.assert_called_once_with(task, result)
        self.quality_assurance_layer.update_task_history.assert_called_once_with(task, 0.7, 0.3)

    async def test_save_models(self) -> None:
        path = "test_path"
        await self.problem_analyzer.save_models(path)
        self.problem_analyzer.enhanced_plan_generator.save.assert_called_once_with(path)

    async def test_load_models(self) -> None:
        path = "test_path"
        await self.problem_analyzer.load_models(path)
        self.problem_analyzer.enhanced_plan_generator.load.assert_called_once_with(path)

    async def test_introspect(self) -> None:
        self.quality_assurance_layer.get_info.return_value = {"qa_info": "Test QA info"}
        introspection = await self.problem_analyzer.introspect()
        assert "type" in introspection
        assert "description" in introspection
        assert "quality_assurance_info" in introspection


if __name__ == "__main__":
    unittest.main()
