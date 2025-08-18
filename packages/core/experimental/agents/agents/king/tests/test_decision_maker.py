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


# These decision maker tests rely on the quality assurance layer which pulls in
# transformer models requiring PyTorch. Skip the tests entirely if torch isn't
# available.
try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from agents.king.planning.unified_decision_maker import UnifiedDecisionMaker
from agents.king.quality_assurance_layer import QualityAssuranceLayer


class TestUnifiedDecisionMaker(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tok_patch = patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTok())
        model_patch = patch("transformers.AutoModel.from_pretrained", return_value=DummyModel())
        self.addCleanup(tok_patch.stop)
        self.addCleanup(model_patch.stop)
        tok_patch.start()
        model_patch.start()
        self.communication_protocol = Mock()
        self.rag_system = Mock()
        self.agent = Mock()
        self.quality_assurance_layer = Mock(spec=QualityAssuranceLayer)
        self.decision_maker = UnifiedDecisionMaker(
            self.communication_protocol,
            self.rag_system,
            self.agent,
            self.quality_assurance_layer,
        )

    async def test_make_decision(self) -> None:
        content = "Test decision content"
        eudaimonia_score = 0.8
        self.rag_system.process_query.return_value = {"rag_info": "Test RAG info"}
        self.quality_assurance_layer.eudaimonia_triangulator.get_embedding.return_value = [0.1] * 768
        self.quality_assurance_layer.evaluate_rule_compliance.return_value = 0.9
        self.agent.generate_structured_response.return_value = [
            "Alternative 1",
            "Alternative 2",
        ]
        self.decision_maker.llm.complete.return_value.text = "Test decision"

        result = await self.decision_maker.make_decision(content, eudaimonia_score)

        assert "decision" in result
        assert "eudaimonia_score" in result
        assert "rule_compliance" in result
        assert "rag_info" in result
        assert "best_alternative" in result
        assert "implementation_plan" in result

    async def test_update_model(self) -> None:
        task = {"content": "Test task"}
        result = {"performance": 0.7, "uncertainty": 0.3}
        await self.decision_maker.update_model(task, result)
        self.decision_maker.mcts.update.assert_called_once_with(task, result)
        self.quality_assurance_layer.update_task_history.assert_called_once_with(task, 0.7, 0.3)

    async def test_save_models(self) -> None:
        path = "test_path"
        await self.decision_maker.save_models(path)
        self.decision_maker.mcts.save.assert_called_once_with(f"{path}/mcts_model.pt")

    async def test_load_models(self) -> None:
        path = "test_path"
        await self.decision_maker.load_models(path)
        self.decision_maker.mcts.load.assert_called_once_with(f"{path}/mcts_model.pt")

    def test_update_agent_list(self) -> None:
        agent_list = ["Agent1", "Agent2"]
        self.decision_maker.update_agent_list(agent_list)
        assert self.decision_maker.available_agents == agent_list

    async def test_generate_alternatives(self) -> None:
        problem_analysis = {"content": "Test problem"}
        self.agent.generate_structured_response.return_value = ["Alt1", "Alt2"]
        self.communication_protocol.send_and_wait.return_value.content = {"alternatives": ["Alt3"]}

        alternatives = await self.decision_maker._generate_alternatives(problem_analysis)

        assert "Alt1" in alternatives
        assert "Alt2" in alternatives
        assert "Alt3" in alternatives

    async def test_evaluate_alternatives(self) -> None:
        alternatives = ["Alt1", "Alt2"]
        ranked_criteria = [
            {"criterion": "eudaimonia", "weight": 0.5},
            {"criterion": "curiosity", "weight": 0.5},
        ]
        self.quality_assurance_layer.eudaimonia_triangulator.get_embedding.return_value = [0.1] * 768
        self.quality_assurance_layer.eudaimonia_triangulator.triangulate.return_value = 0.8
        self.quality_assurance_layer.evaluate_rule_compliance.return_value = 0.9

        evaluated = await self.decision_maker._evaluate_alternatives(alternatives, ranked_criteria)

        assert len(evaluated) == 2
        assert "alternative" in evaluated[0]
        assert "score" in evaluated[0]

    async def test_create_implementation_plan(self) -> None:
        plan = {"decision": "Test decision", "best_alternative": "Test alternative"}
        self.agent.generate_structured_response.return_value = {
            "monitoring": ["Step1"],
            "feedback_analysis": ["Step2"],
        }

        implementation_plan = await self.decision_maker._create_implementation_plan(plan)

        assert "monitoring" in implementation_plan
        assert "feedback_analysis" in implementation_plan


if __name__ == "__main__":
    unittest.main()
