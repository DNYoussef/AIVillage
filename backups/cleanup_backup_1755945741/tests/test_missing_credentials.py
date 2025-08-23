import importlib.util
import pathlib
import sys
import types

import pytest
from pydantic import ValidationError


def _load_module(relative_path: str, name: str):
    path = pathlib.Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_settings_missing_env(monkeypatch):
    for var in ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)
    module = _load_module("packages/core/experimental/agents/agents/utils/configuration.py", "configuration")
    Settings = module.Settings
    with pytest.raises(ValidationError):
        Settings()


def test_credits_config_missing_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    module = _load_module("packages/p2p/communications/credits_ledger.py", "credits_ledger")
    CreditsConfig = module.CreditsConfig
    with pytest.raises(RuntimeError):
        CreditsConfig()


@pytest.mark.asyncio
async def test_mcp_server_missing_secret(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_SECRET", raising=False)

    software = types.ModuleType("software")
    hyper_rag = types.ModuleType("software.hyper_rag")
    hyper_pipeline = types.ModuleType("software.hyper_rag.hyper_rag_pipeline")

    class HyperRAGPipeline:  # pragma: no cover - simple stub
        pass

    hyper_pipeline.HyperRAGPipeline = HyperRAGPipeline
    sys.modules["software"] = software
    sys.modules["software.hyper_rag"] = hyper_rag
    sys.modules["software.hyper_rag.hyper_rag_pipeline"] = hyper_pipeline

    module = _load_module("packages/rag/mcp_servers/hyperag/mcp_server.py", "mcp_server")
    HypeRAGMCPServer = module.HypeRAGMCPServer
    server = HypeRAGMCPServer()
    with pytest.raises(RuntimeError):
        await server.initialize()
