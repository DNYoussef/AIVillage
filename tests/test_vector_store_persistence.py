import importlib.util
import unittest
import json
import tempfile
import sys
from pathlib import Path
from datetime import datetime
import types

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("Required dependency not installed")

import numpy as np

# Provide a lightweight faiss stub with serialization support
fake_faiss = types.ModuleType("faiss")

class DummyIndex:
    def __init__(self):
        self.vectors = []

    def add(self, vecs):
        self.vectors.extend(vecs.tolist())

    def search(self, x, k):
        return np.zeros((1, k), dtype="float32"), np.zeros((1, k), dtype=int)

    def remove_ids(self, x):
        pass

fake_faiss.IndexFlatL2 = lambda dim: DummyIndex()
fake_faiss.serialize_index = lambda idx: json.dumps(idx.vectors).encode()

def _deserialize(data: bytes):
    idx = DummyIndex()
    idx.vectors = json.loads(data.decode())
    return idx

fake_faiss.deserialize_index = _deserialize
sys.modules["faiss"] = fake_faiss

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_system.retrieval.vector_store import VectorStore


class TestVectorStorePersistence(unittest.TestCase):
    def test_save_and_load(self):
        store = VectorStore(dimension=2)
        doc = {
            "id": "1",
            "content": "a",
            "embedding": np.zeros(2, dtype="float32"),
            "timestamp": datetime.now(),
        }
        store.add_documents([doc])

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "store.json"
            store.save(str(path))
            loaded = VectorStore.load(str(path), store.config)
            self.assertEqual(loaded.get_size(), 1)
            self.assertEqual(loaded.dimension, store.dimension)


if __name__ == "__main__":
    unittest.main()
