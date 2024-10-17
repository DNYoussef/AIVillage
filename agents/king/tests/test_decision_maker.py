import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.king.decision_maker import DecisionMaker
from agents.king.mcts import MCTS
from agents.utils.exceptions import AIVillageException
from agents.communication.protocol import Message, MessageType
from agents.agent import Agent  # Add this import

# Rest of the file remains unchanged
import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.king.decision_maker import DecisionMaker
from agents.king.mcts import MCTS
from agents.utils.exceptions import AIVillageException
from agents.communication.protocol import Message, MessageType

@pytest.fixture
def decision_maker():
    communication_protocol = AsyncMock()
    rag_system = AsyncMock()
    ai_provider = AsyncMock()
    return DecisionMaker(communication_protocol, rag_system, ai_provider)

@pytest.mark.asyncio
async def test_make_decision_success(decision_maker):
    # Mock the necessary methods and their return values
    decision_maker.rag_system.query = AsyncMock(return_value={"key": "value"})
    decision_maker.problem_analyzer.analyze = AsyncMock(return_value={"analysis": "result"})
    decision_maker._determine_success_criteria = AsyncMock(return_value=["criterion1", "criterion2"])
    decision_maker._rank_criteria = AsyncMock(return_value=[{"criterion": "criterion1", "rank": 1}, {"criterion": "criterion2", "rank": 2}])
    decision_maker._generate_alternatives = AsyncMock(return_value=["alternative1", "alternative2"])
    decision_maker._evaluate_alternatives = AsyncMock(return_value=[{"alternative": "alternative1", "score": 0.8}, {"alternative": "alternative2", "score": 0.6}])
    decision_maker.mcts.parallel_search = AsyncMock(return_value="optimized_workflow")
    decision_maker.plan_generator.generate_plan = AsyncMock(return_value={"plan": "details"})
    decision_maker.suggest_best_agent = AsyncMock(return_value="agent1")

    result = await decision_maker.make_decision("test task")

    assert result["chosen_alternative"] == "alternative1"
    assert result["optimized_workflow"] == "optimized_workflow"
    assert result["plan"] == {"plan": "details"}
    assert result["suggested_agent"] == "agent1"
    assert "problem_analysis" in result
    assert "criteria" in result
    assert "alternatives" in result

@pytest.mark.asyncio
async def test_make_decision_failure(decision_maker):
    decision_maker.rag_system.query.side_effect = Exception("RAG system error")

    with pytest.raises(AIVillageException):
        await decision_maker.make_decision("test task")

@pytest.mark.asyncio
async def test_suggest_best_agent(decision_maker):
    decision_maker.ai_provider.generate_text = AsyncMock(return_value="agent1")
    decision_maker.available_agents = ["agent1", "agent2"]

    result = await decision_maker.suggest_best_agent("test task", {"analysis": "result"}, "alternative1")

    assert result == "agent1"
    decision_maker.ai_provider.generate_text.assert_called_once()

@pytest.mark.asyncio
async def test_update_model(decision_maker):
    await decision_maker.update_model({"task": "details"}, "result")
    # Add assertions based on the expected behavior of update_model

@pytest.mark.asyncio
async def test_update_mcts(decision_maker):
    decision_maker.mcts.update = AsyncMock()
    await decision_maker.update_mcts({"task": "details"}, "result")
    decision_maker.mcts.update.assert_called_once_with({"task": "details"}, "result")

@pytest.mark.asyncio
async def test_mcts_parallel_search(decision_maker):
    decision_maker.problem_analyzer.generate_possible_states = AsyncMock(return_value=["state1", "state2"])
    decision_maker.plan_generator.evaluate = AsyncMock(return_value=0.5)
    
    result = await decision_maker.mcts.parallel_search("initial_state", decision_maker.problem_analyzer, decision_maker.plan_generator)
    
    assert isinstance(result, str)
    assert decision_maker.problem_analyzer.generate_possible_states.called
    assert decision_maker.plan_generator.evaluate.called

def test_save_models(decision_maker):
    decision_maker.mcts.save = MagicMock()
    decision_maker.save_models("test_path")
    decision_maker.mcts.save.assert_called_once_with("test_path/mcts_model.pt")

def test_load_models(decision_maker):
    decision_maker.mcts.load = MagicMock()
    decision_maker.load_models("test_path")
    decision_maker.mcts.load.assert_called_once_with("test_path/mcts_model.pt")

def test_update_agent_list(decision_maker):
    agents = ["agent1", "agent2", "agent3"]
    decision_maker.update_agent_list(agents)
    assert decision_maker.available_agents == agents

@pytest.mark.asyncio
async def test_generate_alternatives(decision_maker):
    decision_maker.ai_provider.generate_structured_response = AsyncMock(return_value=["alt1", "alt2"])
    decision_maker.communication_protocol.get_all_agents = AsyncMock(return_value=["agent1", "agent2"])
    decision_maker.communication_protocol.send_and_wait = AsyncMock(return_value=AsyncMock(content={"alternatives": ["alt3"]}))

    result = await decision_maker._generate_alternatives({"analysis": "result"})

    assert result == ["alt1", "alt2", "alt3"]
    assert decision_maker.ai_provider.generate_structured_response.called
    assert decision_maker.communication_protocol.get_all_agents.called
    assert decision_maker.communication_protocol.send_and_wait.called

# Add more tests for the private methods if needed

if __name__ == "__main__":
    pytest.main()
