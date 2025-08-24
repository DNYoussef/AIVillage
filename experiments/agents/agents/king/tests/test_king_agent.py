import importlib.util
import unittest
from unittest.mock import Mock, patch

# Skip these tests when PyTorch is not installed since KingAgent depends on
# transformer models.
if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from rag_system.retrieval.vector_store import VectorStore

from agents.king.king_agent import KingAgent, UnifiedAgentConfig
from agents.utils.task import Task as LangroidTask
from core.error_handling import StandardCommunicationProtocol


class TestKingAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.communication_protocol = Mock(spec=StandardCommunicationProtocol)
        self.vector_store = Mock(spec=VectorStore)
        config = UnifiedAgentConfig(name="TestKingAgent", description="Test King Agent", model="gpt-4")
        self.king_agent = KingAgent(config, self.communication_protocol, self.vector_store)

    @patch("agents.king.unified_task_manager.UnifiedTaskManager.create_task")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.assign_task")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.get_task_status")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.get_task_result")
    async def test_execute_task(self, mock_get_result, mock_get_status, mock_assign_task, mock_create_task) -> None:
        mock_create_task.return_value = Mock(id="task_1")
        mock_get_status.side_effect = ["IN_PROGRESS", "IN_PROGRESS", "COMPLETED"]
        mock_get_result.return_value = {"success": True, "result": "Task completed"}

        task = LangroidTask(None, "Test task")
        result = await self.king_agent.execute_task(task)

        assert result["success"]
        assert result["result"] == "Task completed"
        mock_create_task.assert_called_once()
        mock_assign_task.assert_called_once()

    @patch("agents.king.unified_task_manager.UnifiedTaskManager.create_task")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.create_workflow")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.execute_workflow")
    async def test_create_and_execute_workflow(
        self, mock_execute_workflow, mock_create_workflow, mock_create_task
    ) -> None:
        mock_create_task.side_effect = [Mock(id="task_1"), Mock(id="task_2")]
        mock_create_workflow.return_value = Mock(id="workflow_1")

        tasks = [{"description": "Task 1"}, {"description": "Task 2"}]
        dependencies = {"task_2": ["task_1"]}

        workflow = await self.king_agent.create_workflow("Test Workflow", tasks, dependencies)
        assert workflow["workflow_id"] == "workflow_1"

        result = await self.king_agent.execute_workflow("workflow_1")
        assert result["status"] == "Workflow execution started"

        mock_create_task.assert_called()
        mock_create_workflow.assert_called_once()
        mock_execute_workflow.assert_called_once()

    @patch("agents.king.unified_task_manager.UnifiedTaskManager.create_complex_task")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.create_workflow")
    @patch("agents.king.unified_task_manager.UnifiedTaskManager.execute_workflow")
    async def test_execute_complex_task(
        self, mock_execute_workflow, mock_create_workflow, mock_create_complex_task
    ) -> None:
        mock_create_complex_task.return_value = [
            Mock(id="subtask_1"),
            Mock(id="subtask_2"),
        ]
        mock_create_workflow.return_value = Mock(id="workflow_1")

        task = {
            "description": "Complex task",
            "is_complex": True,
            "context": {"some": "context"},
        }

        await self.king_agent.execute_task(task)

        mock_create_complex_task.assert_called_once_with("Complex task", {"some": "context"})
        mock_create_workflow.assert_called_once()
        mock_execute_workflow.assert_called_once()

    @patch("asyncio.create_task")
    async def test_start_background_tasks(self, mock_create_task) -> None:
        await self.king_agent.start_background_tasks()
        assert mock_create_task.call_count == 3

    async def test_update_config(self) -> None:
        new_config = {"batch_size": 10}
        with patch.object(self.king_agent.task_manager, "set_batch_size") as mock_set_batch_size:
            await self.king_agent.update_config(new_config)
            mock_set_batch_size.assert_called_once_with(10)


if __name__ == "__main__":
    unittest.main()
