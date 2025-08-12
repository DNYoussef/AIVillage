import importlib

def test_agent_migration_manager_import():
    module = importlib.import_module(
        "AIVillage.production.distributed_agents.agent_migration_manager"
    )
    assert hasattr(module, "AgentMigrationManager")

