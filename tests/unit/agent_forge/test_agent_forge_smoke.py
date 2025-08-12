import sys
import tempfile
from pathlib import Path

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
