import importlib.util, unittest
if importlib.util.find_spec("httpx") is None:
    raise unittest.SkipTest("Required dependency not installed")

import os
from unittest.mock import patch
from fastapi.testclient import TestClient
import types, sys

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *a, **k: object()
fake_torch = types.ModuleType("torch")
fake_nx = types.ModuleType("networkx")
fake_nx.Graph = lambda *a, **k: object()

with patch.dict(sys.modules, {"faiss": fake_faiss, "torch": fake_torch, "networkx": fake_nx}):
    import server

class TestAuthMiddleware(unittest.TestCase):
    def test_api_key_auth(self):
        os.environ["API_KEY"] = "secret"
        client = TestClient(server.app)
        resp = client.post("/query", json={"query": "hi"})
        self.assertEqual(resp.status_code, 401)
        resp = client.post("/query", json={"query": "hi"}, headers={"X-API-Key": "secret"})
        self.assertNotEqual(resp.status_code, 401)

if __name__ == "__main__":
    unittest.main()
