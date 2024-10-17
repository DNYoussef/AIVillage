import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.king.coordinator import KingCoordinator
from agents.king.unified_task_manager import UnifiedTaskManager
from agents.king.decision_maker import DecisionMaker
from agents.king.route_llm import AgentRouter
from agents.utils.exceptions import AIVillageException
from agents.communication.protocol import Message, MessageType

@pytest.fixture
def mock_communication_protocol():
    return AsyncMock()

@pytest.fixture
def mock_rag_system():
    return AsyncMock()

@pytest.fixture
def mock_ai_provider():
    return AsyncMock()

@pytest.fixture
def king_coordinator(mock_communication_protocol, mock_rag_system, mock_ai_provider):
    coordinator = KingCoordinator(mock_communication_protocol, mock_rag_system, mock_ai_provider)
    coordinator.router = MagicMock(spec=AgentRouter)
    coordinator.decision_maker = MagicMock(spec=DecisionMaker)
    coordinator.task_manager = MagicMock(spec=UnifiedTaskManager)
    return coordinator

@pytest.mark.asyncio
async def test_handle_task_message_routing_success(king_coordinator):
    mock_message = AsyncMock()
    mock_message.content = {'description': 'test task'}
    king_coordinator.router.route.return_value = [('sage', 0.8)]
    
    await king_coordinator.handle_task_message(mock_message)
    
    king_coordinator.router.route.assert_called_once_with(['test task'])
    king_coordinator.task_manager.create_task.assert_called_once()
    king_coordinator.task_manager.assign_task.assert_called_once()

@pytest.mark.asyncio
async def test_handle_task_message_decision_maker_fallback(king_coordinator):
    mock_message = AsyncMock()
    mock_message.content = {'description': 'test task'}
    king_coordinator.router.route.return_value = [('undecided', 0.3)]
    king_coordinator.decision_maker.make_decision.return_value = {'chosen_alternative': 'alt1', 'suggested_agent': 'magi'}
    
    await king_coordinator.handle_task_message(mock_message)
    
    king_coordinator.router.route.assert_called_once_with(['test task'])
    king_coordinator.decision_maker.make_decision.assert_called_once_with('test task')
    king_coordinator.task_manager.create_task.assert_called_once()
    king_coordinator.task_manager.assign_task.assert_called_once()

@pytest.mark.asyncio
async def test_process_task_completion(king_coordinator):
    mock_task = {'id': 'task1', 'description': 'test task', 'assigned_agents': ['sage']}
    mock_result = {'success': True}
    
    await king_coordinator.process_task_completion(mock_task, mock_result)
    
    king_coordinator.router.train_model.assert_called_once()
    king_coordinator.task_manager.complete_task.assert_called_once_with('task1', mock_result)
    king_coordinator.decision_maker.update_model.assert_called_once_with(mock_task, mock_result)
    king_coordinator.decision_maker.update_mcts.assert_called_once_with(mock_task, mock_result)

@pytest.mark.asyncio
async def test_add_and_remove_agent(king_coordinator):
    mock_agent = AsyncMock()
    
    await king_coordinator.add_agent('new_agent', mock_agent)
    assert 'new_agent' in king_coordinator.agents
    king_coordinator.router.update_agent_list.assert_called()
    
    await king_coordinator.remove_agent('new_agent')
    assert 'new_agent' not in king_coordinator.agents
    king_coordinator.router.update_agent_list.assert_called()

@pytest.mark.asyncio
async def test_save_and_load_models(king_coordinator):
    await king_coordinator.save_models('test_path')
    king_coordinator.router.save.assert_called_once_with('test_path/agent_router.pt')
    king_coordinator.decision_maker.save_models.assert_called_once_with('test_path/decision_maker')
    king_coordinator.task_manager.save_models.assert_called_once_with('test_path/task_manager')
    
    await king_coordinator.load_models('test_path')
    king_coordinator.router.load.assert_called_once_with('test_path/agent_router.pt')
    king_coordinator.decision_maker.load_models.assert_called_once_with('test_path/decision_maker')
    king_coordinator.task_manager.load_models.assert_called_once_with('test_path/task_manager')

if __name__ == "__main__":
    pytest.main()


# ... (rest of the test implementation remains the same)
