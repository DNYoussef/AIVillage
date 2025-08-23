import importlib
import os
from unittest.mock import patch

import pytest

MODULE_PATH = "src.production.rag.rag_system.vector_store"


def import_vector_store():
    if MODULE_PATH in os.sys.modules:
        del os.sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


def test_https_endpoint_allowed(monkeypatch):
    monkeypatch.setenv("AIVILLAGE_ENV", "production")
    monkeypatch.setenv("RAG_USE_QDRANT", "1")
    monkeypatch.setenv("QDRANT_URL", "https://example.com")
    with patch("qdrant_client.QdrantClient.get_collections", return_value={}):
        module = import_vector_store()
        store = module.VectorStore()
        assert module.QDRANT_URL.startswith("https://")
        assert store.backend is not store.faiss


def test_http_endpoint_disallowed(monkeypatch):
    monkeypatch.setenv("AIVILLAGE_ENV", "production")
    monkeypatch.setenv("QDRANT_URL", "http://example.com")
    with pytest.raises(ValueError):
        import_vector_store()
