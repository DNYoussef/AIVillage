from pathlib import Path
import sys
import tempfile
from unittest import mock

# Ensure the top-level package is preferred over the similarly named one under src/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_forge.core import AgentForge  # noqa: E402


def test_agent_forge_manifest_roundtrip():
    forge = AgentForge(enable_evolution=False, enable_compression=False)
    spec = {"agent_type": "navigator", "name": "dummy"}
    agent = forge.create_agent(spec)
    assert agent is not None

    manifest = forge.create_manifest()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "manifest.json"
        forge.save_manifest(path, manifest)
        loaded = forge.load_manifest(path)
        assert loaded is not None
        assert loaded.to_dict() == manifest.to_dict()

        # Recreate agent from loaded manifest to ensure parity
        forge2 = AgentForge(enable_evolution=False, enable_compression=False)
        agent_spec = loaded.agents[0]
        agent2 = forge2.create_agent(
            {"agent_type": agent_spec["type"], "config": agent_spec["config"]}
        )
        assert agent2 is not None
        assert getattr(agent2, "config", {}) == getattr(agent, "config", {})


class DummyKPIEngine:
    def register_agent(self, *a, **k):
        return None

    def update_agent_kpi(self, *a, **k):
        return None

    def evaluate_population(self):
        return {}

    def start_evolution_scheduler(self):
        return None

    def stop_evolution_scheduler(self):
        return None


def test_hook_invocations():
    forge = AgentForge(enable_evolution=False, enable_compression=False)

    with mock.patch(
        "src.production.agent_forge.core.forge.on_agent_created"
    ) as created_hook:
        agent = forge.create_agent({"agent_type": "navigator"})
        assert agent is not None
        created_hook.assert_called_once()

    # Evolution hook
    forge.kpi_engine = DummyKPIEngine()
    manifest = forge.create_manifest()
    with mock.patch(
        "src.production.agent_forge.core.forge.evolution_step"
    ) as evo_hook:
        forge.run_kpi_cycle(manifest)
        assert evo_hook.called

    # Compression hook
    forge.compression_engines["dummy"] = object()
    agent_id = next(iter(forge.created_agents))
    with mock.patch(
        "src.production.agent_forge.core.forge.apply_compression"
    ) as comp_hook:
        forge.compress_agent(agent_id, "dummy")
        comp_hook.assert_called_once()
