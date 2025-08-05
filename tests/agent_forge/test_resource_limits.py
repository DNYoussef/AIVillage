from pathlib import Path
import platform
import sys
import types

import pytest

# Ensure the src directory is on the path so that the 'agent_forge' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Stub out heavy optional dependencies required by adas.py
for name in [
    "langroid",
    "langroid.agent",
    "langroid.agent.chat_agent",
    "langroid.agent.task",
    "langroid.agent.tool_message",
    "langroid.language_models",
    "langroid.language_models.openai_gpt",
    "rag_system",
    "rag_system.utils",
    "rag_system.utils.logging",
]:
    sys.modules.setdefault(name, types.ModuleType(name))

sys.modules["langroid.agent.chat_agent"].ChatAgent = object
sys.modules["langroid.agent.chat_agent"].ChatAgentConfig = object
sys.modules["langroid.agent.task"].Task = object
sys.modules["langroid.agent.tool_message"].ToolMessage = object
sys.modules["langroid.language_models.openai_gpt"].OpenAIGPTConfig = object
sys.modules["rag_system.utils.logging"].setup_logger = lambda name: None

from agent_forge.adas import adas


def test_setrlimit_success():
    """Setting limits succeeds on Unix platforms."""
    if platform.system() == "Windows":
        pytest.skip("Unix-only test")
    resource = adas.resource
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    new_soft = 1 if soft in (-1, 0) else max(1, soft - 1)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (new_soft, hard))
        current = resource.getrlimit(resource.RLIMIT_CPU)
        assert current[0] == new_soft
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))


def test_setrlimit_failure(monkeypatch):
    """Windows environments raise a clear OSError."""
    monkeypatch.setattr(adas.platform, "system", lambda: "Windows")
    with pytest.raises(OSError):
        adas.resource.setrlimit(adas.resource.RLIMIT_CPU, (1, 1))
