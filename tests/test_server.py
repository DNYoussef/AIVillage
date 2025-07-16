import importlib.util, unittest
if importlib.util.find_spec("httpx") is None:
    raise unittest.SkipTest("Required dependency not installed")

from fastapi import UploadFile
from io import BytesIO
import asyncio
from unittest.mock import AsyncMock, patch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import types

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *args, **kwargs: object()
fake_torch = types.ModuleType("torch")

with patch.dict(sys.modules, {"faiss": fake_faiss, "torch": fake_torch}):
    import server


class TestServer(unittest.TestCase):
    def test_query_and_upload(self):
        async_mock_response = {"answer": "ok"}
        class DummyIndex:
            def add(self, x):
                pass
            def search(self, x, k):
                return (np.zeros((1, k), dtype="float32"), np.zeros((1, k), dtype=int))
            def remove_ids(self, x):
                pass

        class DummyEmbeddingModel:
            def __init__(self, size=8) -> None:
                self.hidden_size = size
            def encode(self, text: str):
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                return [], rng.random(self.hidden_size).astype("float32")

        class DummyVectorStore:
            def __init__(self):
                self.documents = []
                self.index = DummyIndex()
                self.embedding_model = DummyEmbeddingModel(4)
            
            async def add_texts(self, texts):
                for text in texts:
                    _, embedding = self.embedding_model.encode(text)
                    self.documents.append({"text": text, "embedding": embedding})

        with patch.object(server.rag_pipeline, "initialize", AsyncMock()) as mock_init, \
             patch.object(server.rag_pipeline, "shutdown", AsyncMock()) as mock_shutdown, \
             patch.object(server.rag_pipeline, "process", AsyncMock(return_value=async_mock_response)) as mock_process:

            # Replace vector store with dummy implementation
            dummy_vector_store = DummyVectorStore()
            server.vector_store = dummy_vector_store

            async def run_flow():
                await server.startup_event()
                file = UploadFile(filename="test.txt", file=BytesIO(b"hello"))
                resp1 = await server.upload_endpoint(file)
                file = UploadFile(filename="test.txt", file=BytesIO(b"hello"))
                resp2 = await server.upload_endpoint(file)
                query_resp = await server.query_endpoint(server.SecureQueryRequest(query="hi"))
                await server.shutdown_event()
                return resp1, resp2, query_resp

            resp1, resp2, query_resp = asyncio.run(run_flow())

            mock_init.assert_awaited_once()
            mock_shutdown.assert_awaited_once()
            self.assertEqual(resp1["status"], "uploaded")
            self.assertEqual(resp1["filename"], "test.txt")
            self.assertEqual(resp1["size"], 5)
            self.assertEqual(resp1["message"], "File processed successfully")
            
            self.assertEqual(resp2["status"], "uploaded")
            self.assertEqual(resp2["filename"], "test.txt")
            self.assertEqual(resp2["size"], 5)
            self.assertEqual(resp2["message"], "File processed successfully")
            self.assertEqual(len(dummy_vector_store.documents), 2)
            emb1 = dummy_vector_store.documents[0]["embedding"]
            emb2 = dummy_vector_store.documents[1]["embedding"]
            self.assertTrue(np.array_equal(emb1, emb2))
            self.assertEqual(query_resp, async_mock_response)


if __name__ == "__main__":
    unittest.main()
