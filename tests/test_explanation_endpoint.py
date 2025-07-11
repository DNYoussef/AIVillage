import importlib.util
import unittest

if importlib.util.find_spec("httpx") is None:
    raise unittest.SkipTest("Required dependency not installed")

import httpx
import inspect

if "app" not in inspect.signature(httpx.Client).parameters:
    raise unittest.SkipTest("httpx lacks TestClient app support")

from fastapi.testclient import TestClient
from unittest.mock import patch
import types
import sys

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *a, **k: object()
fake_torch = types.ModuleType("torch")
fake_nx = types.ModuleType("networkx")
fake_nx.Graph = lambda *a, **k: object()

with patch.dict(
    sys.modules, {"faiss": fake_faiss, "torch": fake_torch, "networkx": fake_nx}
):
    import server


class TestExplanationEndpoint(unittest.TestCase):
    def test_returns_evidence_list(self):
        client = TestClient(server.app)
        resp = client.get("/v1/explanation", params={"chat_id": "1"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("confidence_tier", data[0])


if __name__ == "__main__":
    unittest.main()
