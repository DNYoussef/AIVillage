import subprocess

import pytest
from packages.core.deployment.continuous_deployment_automation import (
    ContinuousDeploymentAutomation,
    DeploymentConfig,
    DeploymentEnvironment,
)


@pytest.mark.asyncio
async def test_run_command_executes_without_shell():
    automation = ContinuousDeploymentAutomation(DeploymentConfig(environment=DeploymentEnvironment.LOCAL))
    result = await automation._run_command(["python", "-c", "print('hello')"])
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


@pytest.mark.asyncio
async def test_run_automated_tests_no_shell(monkeypatch):
    executed = []

    async def fake_run_command(cmd, timeout=30):
        executed.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    automation = ContinuousDeploymentAutomation(DeploymentConfig(environment=DeploymentEnvironment.LOCAL))
    monkeypatch.setattr(automation, "_run_command", fake_run_command)

    result = await automation._run_automated_tests()
    assert result is True
    assert all(isinstance(c, list) for c in executed)
    assert executed and executed[0][:3] == ["python", "-m", "pytest"]


@pytest.mark.asyncio
async def test_build_and_validate_no_shell(monkeypatch, tmp_path):
    pkg_dir = tmp_path / "packages" / "sample"
    pkg_dir.mkdir(parents=True)
    py_file = pkg_dir / "mod.py"
    py_file.write_text("x=1\n")

    (tmp_path / "tests").mkdir()

    automation = ContinuousDeploymentAutomation(DeploymentConfig(environment=DeploymentEnvironment.LOCAL))
    automation.project_root = tmp_path

    captured = []

    async def fake_run_command(cmd, timeout=30):
        captured.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(automation, "_run_command", fake_run_command)

    result = await automation._build_and_validate()
    assert result is True
    assert captured and captured[0][:3] == ["python", "-m", "py_compile"]
    assert str(py_file) in captured[0]
