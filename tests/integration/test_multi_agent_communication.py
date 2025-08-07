from dataclasses import dataclass, field


@dataclass
class Agent:
    agent_id: str
    inbox: list[tuple[str, str]] = field(default_factory=list)

    def send(self, other: "Agent", message: str) -> None:
        other.inbox.append((self.agent_id, message))


def test_multi_agent_communication() -> None:
    # Spawn 18 agents with unique IDs
    agents = [Agent(f"agent-{i}") for i in range(18)]
    ids = [a.agent_id for a in agents]
    assert len(ids) == 18
    assert len(set(ids)) == 18  # no conflicts

    # Cross-agent messaging
    agents[0].send(agents[1], "ping")
    agents[1].send(agents[2], "pong")

    assert agents[1].inbox == [("agent-0", "ping")]
    assert agents[2].inbox == [("agent-1", "pong")]

    # All other agents should remain isolated
    for idx, agent in enumerate(agents):
        if idx not in {1, 2}:
            assert agent.inbox == []
