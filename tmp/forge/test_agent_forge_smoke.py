"""Smoke tests for Agent Forge system.

These tests verify basic functionality and import capabilities.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_agent_forge_import():
    """Test that AgentForge can be imported successfully."""
    try:
        from src.production.agent_forge.core.forge import AgentForge

        assert AgentForge is not None, "AgentForge class should be importable"
    except ImportError as e:
        pytest.fail(f"Failed to import AgentForge: {e}")


def test_agent_forge_initialization():
    """Test that AgentForge can be initialized."""
    from src.production.agent_forge.core.forge import AgentForge

    # Test basic initialization
    forge = AgentForge()
    assert forge is not None, "AgentForge should initialize successfully"

    # Test string representation
    repr_str = repr(forge)
    assert "AgentForge" in repr_str, "String representation should contain class name"
    assert "agents=" in repr_str, "String representation should show agent count"


def test_agent_spec_creation():
    """Test AgentSpec creation and serialization."""
    from src.production.agent_forge.core.forge import AgentSpec

    spec = AgentSpec(
        agent_type="test_agent",
        name="Test Agent",
        config={"param1": "value1"},
        capabilities=["test_capability"],
    )

    assert spec.agent_type == "test_agent"
    assert spec.name == "Test Agent"
    assert spec.config == {"param1": "value1"}
    assert spec.capabilities == ["test_capability"]

    # Test serialization
    spec_dict = spec.to_dict()
    assert isinstance(spec_dict, dict)
    assert spec_dict["agent_type"] == "test_agent"


def test_agent_manifest_creation():
    """Test AgentManifest creation and serialization."""
    from src.production.agent_forge.core.forge import AgentManifest

    manifest = AgentManifest(
        agents=[{"id": "agent1", "type": "test_agent"}],
        evolution_config={"enabled": True},
        compression_config={"method": "bitnet"},
    )

    assert len(manifest.agents) == 1
    assert manifest.evolution_config["enabled"] is True
    assert manifest.compression_config["method"] == "bitnet"

    # Test serialization and deserialization
    manifest_dict = manifest.to_dict()
    assert isinstance(manifest_dict, dict)

    restored_manifest = AgentManifest.from_dict(manifest_dict)
    assert len(restored_manifest.agents) == 1
    assert restored_manifest.evolution_config["enabled"] is True


def test_system_status():
    """Test system status reporting."""
    from src.production.agent_forge.core.forge import AgentForge

    forge = AgentForge()
    status = forge.get_system_status()

    assert isinstance(status, dict)
    assert "agent_factory_available" in status
    assert "evolution_enabled" in status
    assert "compression_enabled" in status
    assert "created_agents" in status
    assert "available_agent_types" in status
    assert "compression_engines" in status
    assert "components" in status

    # Check components status
    components = status["components"]
    assert "kpi_engine" in components
    assert "dual_evolution" in components
    assert "resource_evolution" in components


def test_available_agent_types():
    """Test getting available agent types."""
    from src.production.agent_forge.core.forge import AgentForge

    forge = AgentForge()
    agent_types = forge.get_available_agent_types()

    assert isinstance(agent_types, list)
    # Should be empty or contain actual agent types
    # (depends on whether AgentFactory is available)


def test_compression_engines():
    """Test getting compression engines."""
    from src.production.agent_forge.core.forge import AgentForge

    forge = AgentForge()
    engines = forge.get_compression_engines()

    assert isinstance(engines, list)
    # May be empty if compression modules not available


def test_manifest_save_load(tmp_path):
    """Test manifest save and load functionality."""
    from src.production.agent_forge.core.forge import AgentForge, AgentManifest

    forge = AgentForge()

    # Create test manifest
    test_manifest = AgentManifest(
        agents=[{"id": "test1", "type": "sage"}], evolution_config={"test": True}
    )

    # Test save
    manifest_path = tmp_path / "test_manifest.json"
    success = forge.save_manifest(manifest_path, test_manifest)
    assert success, "Manifest should save successfully"
    assert manifest_path.exists(), "Manifest file should exist"

    # Test load
    loaded_manifest = forge.load_manifest(manifest_path)
    assert loaded_manifest is not None, "Manifest should load successfully"
    assert len(loaded_manifest.agents) == 1
    assert loaded_manifest.agents[0]["id"] == "test1"
    assert loaded_manifest.evolution_config["test"] is True


def test_kpi_cycle_without_agents():
    """Test KPI cycle with no agents."""
    from src.production.agent_forge.core.forge import AgentForge

    forge = AgentForge()
    result = forge.run_kpi_cycle()

    assert isinstance(result, dict)
    assert "success" in result
    # Should handle gracefully whether KPI engine is available or not


def test_evolution_start_stop():
    """Test evolution system start/stop."""
    from src.production.agent_forge.core.forge import AgentForge

    forge = AgentForge()

    # Test start
    start_result = forge.start_evolution()
    assert isinstance(start_result, bool)

    # Test stop
    stop_result = forge.stop_evolution()
    assert isinstance(stop_result, bool)


if __name__ == "__main__":
    # Run basic smoke test
    print("Running Agent Forge smoke tests...")

    try:
        test_agent_forge_import()
        print("[PASS] Import test passed")

        test_agent_forge_initialization()
        print("[PASS] Initialization test passed")

        test_agent_spec_creation()
        print("[PASS] AgentSpec test passed")

        test_agent_manifest_creation()
        print("[PASS] AgentManifest test passed")

        test_system_status()
        print("[PASS] System status test passed")

        print("[SUCCESS] All smoke tests passed!")

    except Exception as e:
        print(f"[FAIL] Smoke test failed: {e}")
        sys.exit(1)
