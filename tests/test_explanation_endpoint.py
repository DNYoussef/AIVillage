import importlib.util
import unittest

if importlib.util.find_spec("httpx") is None:
    msg = "Required dependency not installed"
    raise unittest.SkipTest(msg)

import inspect

import httpx

if "app" not in inspect.signature(httpx.Client).parameters:
    msg = "httpx lacks TestClient app support"
    raise unittest.SkipTest(msg)

import sys
import types
from unittest.mock import patch

from fastapi.testclient import TestClient

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *a, **k: object()
fake_torch = types.ModuleType("torch")
fake_nx = types.ModuleType("networkx")
fake_nx.Graph = lambda *a, **k: object()

with patch.dict(
    sys.modules, {"faiss": fake_faiss, "torch": fake_torch, "networkx": fake_nx}
):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
    import server


class TestExplanationEndpoint(unittest.TestCase):
    def test_returns_evidence_list(self):
        client = TestClient(server.app)
        resp = client.get("/v1/explanation", params={"chat_id": "1"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "confidence_tier" in data[0]


if __name__ == "__main__":
    unittest.main()
