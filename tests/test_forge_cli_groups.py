from agent_forge.cli import forge


def test_forge_cli_groups_registered():
    """Ensure core subcommand groups are registered under forge."""
    assert "training" in forge.commands
    assert "curriculum" in forge.commands
    assert "compression" in forge.commands
