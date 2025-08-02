import subprocess
import sys

from agent_forge.validate_all_agents import validate_all_agents


def test_all_agent_templates_valid():
    results = validate_all_agents(full_test=True)
    assert results  # ensure templates were discovered
    for agent, checks in results.items():
        assert all(checks.values()), f"{agent} failed {checks}"


def test_validate_all_agents_cli():
    cmd = [sys.executable, "-m", "agent_forge.validate_all_agents", "--full-test"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "All agents validated successfully" in completed.stdout
