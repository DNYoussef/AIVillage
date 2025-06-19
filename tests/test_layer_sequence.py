import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path
import importlib.util
import pytest

pytestmark = pytest.mark.requires_gpu
if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from communications.protocol import StandardCommunicationProtocol
from agents.utils.task import Task as LangroidTask

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
        instructions=""
    )
    protocol = MagicMock(spec=StandardCommunicationProtocol)
    return DummyAgent(config, protocol, None)

class TestLayerSequence(unittest.IsolatedAsyncioTestCase):
    async def test_layers_run(self):
        agent = build_agent()
        task = LangroidTask(agent, "do something", "1", 1)
        task.type = "general"
        with patch.object(agent.quality_assurance_layer, 'check_task_safety', return_value=True) as qa_mock, \
             patch.object(agent.foundational_layer, 'process_task', AsyncMock(return_value=task)) as f_mock, \
             patch.object(agent.agent_architecture_layer, 'process_result', AsyncMock(return_value={"ok": True})) as arch_mock, \
             patch.object(agent.decision_making_layer, 'make_decision', AsyncMock(return_value="done")) as dm_mock, \
             patch.object(agent.continuous_learning_layer, 'update', AsyncMock()) as cl_mock:
            result = await agent.execute_task(task)
        qa_mock.assert_called_once_with(task)
        f_mock.assert_called_once_with(task)
        arch_mock.assert_called_once()
        dm_mock.assert_called_once_with(task, {"ok": True})
        cl_mock.assert_called_once_with(task, "done")
        self.assertEqual(result, {"result": "done"})

if __name__ == '__main__':
    unittest.main()
