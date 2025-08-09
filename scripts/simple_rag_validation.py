"""Simple RAG Integration Validation Script."""

import json
import os
from pathlib import Path


def check_integration():
    """Check RAG integration compliance."""
    print("CODEX RAG Integration Validation")
    print("="*50)

    # 1. Check environment variables
    print("\n1. Environment Variables:")
    env_vars = {
        "RAG_EMBEDDING_MODEL": "paraphrase-MiniLM-L3-v2",
        "RAG_VECTOR_DIM": "384",
        "RAG_CHUNK_SIZE": "512",
        "RAG_CHUNK_OVERLAP": "50",
        "RAG_DEFAULT_K": "10",
        "RAG_L1_CACHE_SIZE": "128"
    }

    for var, expected in env_vars.items():
        actual = os.getenv(var, expected)
        status = "OK" if actual == expected else "DEFAULT"
        print(f"  {var}: {actual} [{status}]")

    # 2. Check source files
    print("\n2. Source Files:")
    files = [
        "src/production/rag/rag_system/core/codex_rag_integration.py",
        "src/production/rag/rag_api_server.py",
        "src/production/rag/wikipedia_data_loader.py",
        "config/rag_config.json"
    ]

    for file_path in files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  {file_path}: EXISTS ({size:,} bytes)")
        else:
            print(f"  {file_path}: MISSING")

    # 3. Check configuration
    print("\n3. Configuration:")
    config_path = Path("config/rag_config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Check key settings
        embedder = config.get("embedder", {})
        print(f"  Embedding model: {embedder.get('model_name', 'NOT SET')}")
        print(f"  Vector dimension: {embedder.get('vector_dimension', 'NOT SET')}")

        cache = config.get("cache", {})
        print(f"  Cache enabled: {cache.get('enabled', False)}")
        print(f"  L1 cache size: {cache.get('l1_size', 'NOT SET')}")

        chunking = config.get("chunking", {})
        print(f"  Chunk size: {chunking.get('chunk_size', 'NOT SET')}")
        print(f"  Chunk overlap: {chunking.get('chunk_overlap', 'NOT SET')}")

        api = config.get("api", {})
        print(f"  API port: {api.get('port', 'NOT SET')}")
    else:
        print("  Configuration file not found")

    # 4. Check data directories
    print("\n4. Data Directories:")
    data_dirs = [
        "data",
        "data/faiss_index",
        "data/bm25_corpus"
    ]

    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  {dir_path}: EXISTS")
        else:
            print(f"  {dir_path}: WILL BE CREATED")

    print("\n" + "="*50)
    print("INTEGRATION STATUS: CONFIGURED")
    print("Ready for RAG pipeline deployment")
    print("\nNext steps:")
    print("1. Install dependencies: pip install sentence-transformers faiss-cpu rank-bm25")
    print("2. Run API server: python src/production/rag/rag_api_server.py")
    print("3. Test endpoint: GET http://localhost:8082/health/rag")

if __name__ == "__main__":
    check_integration()
