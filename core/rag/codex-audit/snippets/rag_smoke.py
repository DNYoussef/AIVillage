"""
Test C3: RAG Pipeline - Verify default instantiation and basic functionality
"""

import json
import os
import sys
import time
from pathlib import Path

# Set RAG_LOCAL_MODE for testing
os.environ["RAG_LOCAL_MODE"] = "1"

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))


def test_rag_pipeline():
    """Test RAG pipeline instantiation and basic operations"""
    results = {
        "import": False,
        "instantiation": False,
        "add_documents": False,
        "retrieve": False,
        "latency_ms": None,
    }

    # Try to import RAGPipeline
    try:
        from production.rag.rag_system.core.pipeline import RAGPipeline

        results["import"] = True
        print("[PASS] RAGPipeline imported successfully")
    except ImportError as e:
        print(f"[FAIL] RAGPipeline import failed: {e}")
        return results

    # Try to instantiate with defaults
    try:
        pipeline = RAGPipeline()
        results["instantiation"] = True
        print("[PASS] RAGPipeline instantiated with defaults")
    except Exception as e:
        print(f"[FAIL] RAGPipeline instantiation failed: {e}")
        # Try with minimal config
        try:
            pipeline = RAGPipeline(config={"cache_enabled": False})
            results["instantiation"] = True
            print("[PASS] RAGPipeline instantiated with minimal config")
        except Exception as e2:
            print(f"[FAIL] RAGPipeline minimal instantiation failed: {e2}")
            return results

    # Add test documents
    test_docs = [
        {
            "id": "doc1",
            "content": "The AIVillage project is a multi-agent AI system with self-evolution capabilities.",
            "metadata": {"source": "test", "type": "overview"},
        },
        {
            "id": "doc2",
            "content": "The compression pipeline achieves 98% size reduction while maintaining performance.",
            "metadata": {"source": "test", "type": "technical"},
        },
        {
            "id": "doc3",
            "content": "The P2P network uses LibP2P for reliable distributed communication.",
            "metadata": {"source": "test", "type": "networking"},
        },
        {
            "id": "doc4",
            "content": "The compression pipeline does NOT achieve 98% reduction. This is a contradiction.",
            "metadata": {"source": "test", "type": "contradiction"},
        },
    ]

    try:
        # Try different methods to add documents
        if hasattr(pipeline, "add_documents"):
            pipeline.add_documents(test_docs)
            results["add_documents"] = True
        elif hasattr(pipeline, "index"):
            for doc in test_docs:
                pipeline.index(doc["content"], doc["metadata"])
            results["add_documents"] = True
        elif hasattr(pipeline, "ingest"):
            pipeline.ingest(test_docs)
            results["add_documents"] = True
        else:
            print("[WARNING] No document ingestion method found")
            results["add_documents"] = False

        if results["add_documents"]:
            print("[PASS] Documents added to pipeline")
        else:
            print("[FAIL] Could not add documents")
    except Exception as e:
        print(f"[FAIL] Document addition failed: {e}")
        results["add_documents"] = False

    # Test retrieval
    try:
        query = "What is the compression ratio achieved?"

        start_time = time.time()

        if hasattr(pipeline, "retrieve"):
            retrieved = pipeline.retrieve(query)
        elif hasattr(pipeline, "query"):
            retrieved = pipeline.query(query)
        elif hasattr(pipeline, "search"):
            retrieved = pipeline.search(query)
        else:
            print("[FAIL] No retrieval method found")
            results["retrieve"] = False
            return results

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        results["latency_ms"] = latency_ms

        if retrieved:
            results["retrieve"] = True
            print(f"[PASS] Retrieved {len(retrieved) if hasattr(retrieved, '__len__') else '?'} results")
            print(f"[INFO] Query latency: {latency_ms:.2f}ms")
        else:
            results["retrieve"] = False
            print("[FAIL] No results retrieved")

    except Exception as e:
        print(f"[FAIL] Retrieval failed: {e}")
        results["retrieve"] = False

    return results


def main():
    """Run RAG smoke test"""
    print("=" * 60)
    print("C3: RAG Pipeline Smoke Test")
    print("=" * 60)

    results = test_rag_pipeline()

    # Calculate overall success
    overall_success = results["import"] and results["instantiation"] and results["retrieve"]

    # Save results
    output_path = Path(__file__).parent.parent / "artifacts" / "rag_latency.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "results": results,
                "overall_success": overall_success,
                "latency_target_ms": 100,
                "meets_target": results["latency_ms"] < 100 if results["latency_ms"] else False,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"Overall RAG Test Result: {'PASS' if overall_success else 'FAIL'}")
    if results["latency_ms"]:
        print(f"Latency: {results['latency_ms']:.2f}ms (Target: <100ms)")
    print(f"Results saved to: {output_path}")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
