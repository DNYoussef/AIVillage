from src.production.agent_forge.agent_factory import AgentFactory


def test_required_agent_types_count():
    factory = AgentFactory()
    agent_types = factory.required_agent_types()
    assert len(agent_types) == 18
    assert len(set(agent_types)) == 18
