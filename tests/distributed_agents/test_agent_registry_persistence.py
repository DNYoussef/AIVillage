
from src.production.distributed_agents.agent_registry import DistributedAgentRegistry


def test_register_and_persist(tmp_path, monkeypatch):
    # Use a temporary working directory so we don't affect real cache
    monkeypatch.chdir(tmp_path)

    registry = DistributedAgentRegistry()
    registry.register("alpha", "ipc://alpha")
    registry.register("beta", "tcp://beta", location="remote")

    # New instance should load persisted data
    registry2 = DistributedAgentRegistry()
    alpha = registry2.resolve("alpha")
    beta = registry2.resolve("beta")

    assert alpha is not None and alpha.endpoint == "ipc://alpha"
    assert alpha.location == "local"
    assert beta is not None and beta.location == "remote"
    assert sorted([a.agent_id for a in registry2.list()]) == ["alpha", "beta"]
