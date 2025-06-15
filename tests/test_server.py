import unittest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import types

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *args, **kwargs: object()

with patch.dict(sys.modules, {"faiss": fake_faiss}):
    import server


class TestServer(unittest.TestCase):
    def test_query_and_upload(self):
        async_mock_response = {"answer": "ok"}
        with patch.object(server.rag_pipeline, "initialize", AsyncMock()) as mock_init, \
             patch.object(server.rag_pipeline, "shutdown", AsyncMock()) as mock_shutdown, \
             patch.object(server.rag_pipeline, "process", AsyncMock(return_value=async_mock_response)) as mock_process, \
             patch.object(server.vector_store, "add_texts", AsyncMock()) as mock_add_texts:
            with TestClient(server.app) as client:
                # Startup should have been awaited
                mock_init.assert_awaited_once()

                resp = client.post("/query", json={"query": "hello"})
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json(), async_mock_response)
                mock_process.assert_awaited_once_with("hello")

                resp = client.post("/upload", files={"file": ("test.txt", b"hello", "text/plain")})
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json(), {"status": "uploaded"})
                mock_add_texts.assert_awaited_once_with(["hello"])

            # After exiting the context manager shutdown event should have run
            mock_shutdown.assert_awaited_once()
            mock_init.assert_awaited_once()  # ensure still only once


if __name__ == "__main__":
    unittest.main()
