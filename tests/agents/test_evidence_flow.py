# ruff: noqa
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub heavy dependencies before importing SageAgent
tok_mod = types.ModuleType("tiktoken")
load_mod = types.ModuleType("tiktoken.load")


def _dummy_loader(path: str):
    return {}


load_mod.load_tiktoken_bpe = _dummy_loader
tok_mod.Encoding = object
sys.modules.setdefault("openai", types.ModuleType("openai"))
torch_mod = types.ModuleType("torch")
torch_mod.save = lambda *a, **k: None
torch_mod.nn = types.ModuleType("torch.nn")
torch_mod.optim = types.ModuleType("torch.optim")
torch_mod.nn.functional = types.ModuleType("torch.nn.functional")
torch_mod.nn.Module = object
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.nn", torch_mod.nn)
sys.modules.setdefault("torch.optim", torch_mod.optim)
sys.modules.setdefault("torch.nn.functional", torch_mod.nn.functional)
sys.modules.setdefault("tiktoken", tok_mod)
sys.modules.setdefault("tiktoken.load", load_mod)
sys.modules.setdefault(
    "agent_forge.adas.technique_archive",
    types.SimpleNamespace(ChainOfThought=object, TreeOfThoughts=object),
)
orch_mod = types.ModuleType("agents.orchestration")
orch_mod.main = lambda: None
sys.modules.setdefault("agents.orchestration", orch_mod)
sys.modules.setdefault("agents.king.king_agent", types.ModuleType("agents.king.king_agent"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

import importlib  # noqa: E402

from core.error_handling import (  # noqa: E402
    Message,
    MessageType,
    StandardCommunicationProtocol,
)

SageAgent = importlib.import_module("agents.sage.sage_agent").SageAgent  # noqa: E402
UnifiedConfig = importlib.import_module("rag_system.core.config").UnifiedConfig  # noqa: E402
EvidencePack = importlib.import_module("core.evidence").EvidencePack  # noqa: E402


@pytest.mark.asyncio
async def test_sage_emits_evidence_message():
    protocol = StandardCommunicationProtocol()
    received = []

    async def cb(msg: Message):
        received.append(msg)

    protocol.subscribe("tester", cb)

    with patch.object(SageAgent, "__init__", return_value=None):
        agent = SageAgent()
    agent.communication_protocol = protocol
    agent.name = "sage"
    agent.performance_metrics = {}
    agent.rag_system = MagicMock()
    agent.response_generator = MagicMock()
    agent.post_process_result = AsyncMock(side_effect=lambda *a, **k: {"rag_result": a[0]})
    agent.user_intent_interpreter = MagicMock()
    agent.user_intent_interpreter.interpret_intent = AsyncMock(return_value={})
    agent.pre_process_query = AsyncMock(side_effect=lambda q: q)
    agent.rag_system.process_query = AsyncMock(
        return_value={"retrieved_info": [{"id": "1", "content": "c", "score": 0.8}]}
    )
    agent.response_generator.generate_response = AsyncMock(return_value="resp")

    async def _exec(task):
        content = getattr(task, "content", getattr(task, "name", ""))
        return await agent.process_user_query(content)

    agent.execute_task = AsyncMock(side_effect=_exec)

    msg = Message(
        type=MessageType.TASK,
        sender="tester",
        receiver=agent.name,
        content={"content": "hi", "is_user_query": True},
    )
    await agent.handle_message(msg)

    evidence_msgs = [m for m in received if m.type == MessageType.EVIDENCE]
    assert len(evidence_msgs) == 1
    pack = EvidencePack(**evidence_msgs[0].content)
    assert pack.query == "hi"
