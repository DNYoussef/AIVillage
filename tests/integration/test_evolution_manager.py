from experimental.agents.agents.king.evolution_manager import EvolutionManager
from experimental.agents.agents.king.king_agent import KingAgent
import pytest


@pytest.mark.asyncio
async def test_evolution_manager_integration():
    # Create a King Agent
    king_agent = KingAgent()

    # Create an Evolution Manager
    evolution_manager = EvolutionManager()

    # Evolve the King Agent
    await evolution_manager.evolve(king_agent)

    # Check that the King Agent has been evolved
    assert king_agent.is_evolved()
