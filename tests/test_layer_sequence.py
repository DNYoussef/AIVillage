import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.unified_base_agent import (
    AgentArchitectureLayer,
    UnifiedAgentConfig,
    UnifiedBaseAgent,
)
from agents.utils.task import Task as LangroidTask
from core.error_handling import StandardCommunicationProtocol

# Skip if torch is unavailable since underlying agents rely on transformer models.
if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

sys.path.append(str(Path(__file__).resolve().parents[1]))


class DummyAgent(UnifiedBaseAgent):
    async def _process_task(self, task: LangroidTask):
        return {"step": "processed"}


def build_agent():
    config = UnifiedAgentConfig(
        name="dummy",
        description="dummy",
        capabilities=["general"],
        rag_config=MagicMock(),
        vector_store=MagicMock(),
        model="gpt-4",
        instructions="",
    )
    protocol = MagicMock(spec=StandardCommunicationProtocol)
    return DummyAgent(config, protocol, None)


class TestLayerSequence(unittest.IsolatedAsyncioTestCase):
    async def test_layers_run(self):
        agent = build_agent()
        task = LangroidTask(agent, "do something", "1", 1)
        task.type = "general"
        with (
            patch.object(agent.quality_assurance_layer, "check_task_safety", return_value=True) as qa_mock,
            patch.object(agent.foundational_layer, "process_task", AsyncMock(return_value=task)) as f_mock,
            patch.object(
                agent.agent_architecture_layer,
                "process_result",
                AsyncMock(return_value={"ok": True}),
            ) as arch_mock,
            patch.object(
                agent.decision_making_layer,
                "make_decision",
                AsyncMock(return_value="done"),
            ) as dm_mock,
            patch.object(agent.continuous_learning_layer, "update", AsyncMock()) as cl_mock,
        ):
            result = await agent.execute_task(task)
        qa_mock.assert_called_once_with(task)
        f_mock.assert_called_once_with(task)
        arch_mock.assert_called_once()
        dm_mock.assert_called_once_with(task, {"ok": True})
        cl_mock.assert_called_once_with(task, "done")
        assert result == {"result": "done"}


class TestArchitectureLayerCycle(unittest.IsolatedAsyncioTestCase):
    async def test_revision_cycle(self):
        layer = AgentArchitectureLayer()
        layer.quality_threshold = 0.9
        layer.max_revisions = 3
        layer.assistant = AsyncMock(return_value="draft1")
        layer.checker = AsyncMock(side_effect=[{"quality": 0.4}, {"quality": 0.95}])
        layer.reviser = AsyncMock(return_value="draft2")

        result = await layer.process_result("input")

        layer.assistant.assert_called_once_with("input")
        assert layer.checker.call_count == 2
        layer.reviser.assert_called_once_with("draft1", {"quality": 0.4})
        assert result == "draft2"
        assert layer.evaluation_history == [0.4, 0.95]

    async def test_no_revision_needed(self):
        layer = AgentArchitectureLayer()
        layer.quality_threshold = 0.5
        layer.assistant = AsyncMock(return_value="good")
        layer.checker = AsyncMock(return_value={"quality": 0.8})
        layer.reviser = AsyncMock()

        result = await layer.process_result("something")

        layer.assistant.assert_called_once_with("something")
        layer.checker.assert_called_once_with("good")
        layer.reviser.assert_not_called()
        assert result == "good"
        assert layer.evaluation_history == [0.8]


if __name__ == "__main__":
    unittest.main()
