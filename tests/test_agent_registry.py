from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path("src/production/distributed_agents/agent_registry.py")
spec = importlib.util.spec_from_file_location("agent_registry", MODULE_PATH)
agent_registry = importlib.util.module_from_spec(spec)
sys.modules["agent_registry"] = agent_registry
spec.loader.exec_module(agent_registry)  # type: ignore

AgentLocation = agent_registry.AgentLocation
DistributedAgentRegistry = agent_registry.DistributedAgentRegistry


def test_agent_registry_persistence(tmp_path: Path) -> None:
    cache_file = tmp_path / "agents.json"
    registry = DistributedAgentRegistry(cache_path=cache_file)

    registry.register("alpha", "local-endpoint", local=True)
    registry.register("beta", "remote-endpoint", local=False)

    assert registry.resolve("alpha") == AgentLocation("alpha", "local-endpoint", True)
    assert registry.resolve("beta") == AgentLocation("beta", "remote-endpoint", False)
    assert {a.name for a in registry.list()} == {"alpha", "beta"}

    registry2 = DistributedAgentRegistry(cache_path=cache_file)
    assert registry2.resolve("beta") == AgentLocation("beta", "remote-endpoint", False)
    assert len(registry2.list()) == 2
