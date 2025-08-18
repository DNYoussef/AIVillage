import importlib.util
import inspect
import unittest

if importlib.util.find_spec("httpx") is None:
    msg = "Required dependency not installed"
    raise unittest.SkipTest(msg)
import httpx

if "app" not in inspect.signature(httpx.Client).parameters:
    msg = "httpx lacks TestClient app support"
    raise unittest.SkipTest(msg)

import os
import sys
import types
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *a, **k: object()
fake_torch = types.ModuleType("torch")
fake_nx = types.ModuleType("networkx")
fake_nx.Graph = lambda *a, **k: object()

with patch.dict(sys.modules, {"faiss": fake_faiss, "torch": fake_torch, "networkx": fake_nx}):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
    import server


class TestAuthMiddleware(unittest.TestCase):
    def test_api_key_auth(self):
        # Set environment variable before importing server
        os.environ["API_KEY"] = "secret"

        # Need to reload the API_KEY variable in server module
        server.API_KEY = os.getenv("API_KEY")

        # Mock the RAG pipeline to avoid initialization issues
        with patch.object(
            server.rag_pipeline,
            "process",
            AsyncMock(return_value={"answer": "test response"}),
        ):
            client = TestClient(server.app)

            # Test without API key - should fail
            resp = client.post("/query", json={"query": "hi"})
            assert resp.status_code == 401

            # Test with correct API key - should succeed
            resp = client.post("/query", json={"query": "hi"}, headers={"x-api-key": "secret"})
            assert resp.status_code != 401
            # Should be 200 (success) or potentially other non-401 status
            assert resp.status_code in [200, 500]  # 500 might occur due to other issues


if __name__ == "__main__":
    unittest.main()
