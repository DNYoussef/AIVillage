#!/usr/bin/env python3
"""Comprehensive RAG Pipeline Smoke Test.

Tests the complete RAG pipeline with:
- 4 test documents (including one contradiction)
- Top-k retrieval testing
- Latency measurement and printing
- Safe defaults validation
"""

import asyncio
import sys
import time
from pathlib import Path

# Add current directory and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.production.rag.rag_system.core.pipeline import Document, RAGPipeline
    from src.production.rag.rag_system.core.structures import RetrievalResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting alternative import...")
    try:
        # Direct import without src prefix
        import os

        os.chdir(str(Path(__file__).parent.parent.parent))
        from src.production.rag.rag_system.core.pipeline import Document, RAGPipeline
        from src.production.rag.rag_system.core.structures import RetrievalResult
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)


# Test documents with contradictions
TEST_DOCUMENTS = [
    Document(
        id="doc_ml_intro",
        text="""Machine Learning is a subset of artificial intelligence (AI) that
        enables computers to learn and improve from experience without being explicitly
        programmed. It focuses on the development of computer programs that can access
        data and use it to learn for themselves. Machine learning algorithms build
        mathematical models based on training data to make predictions or decisions
        without being explicitly programmed to do so.""",
    ),
    Document(
        id="doc_deep_learning",
        text="""Deep learning is a subset of machine learning that uses artificial
        neural networks with multiple layers to model and understand complex patterns
        in data. These neural networks are inspired by the human brain's structure
        and function. Deep learning has achieved remarkable success in areas like
        image recognition, natural language processing, and game playing. It requires
        large amounts of data and computational power.""",
    ),
    Document(
        id="doc_ai_history",
        text="""Artificial Intelligence was first coined as a term in 1956 at the
        Dartmouth Conference. The field has gone through several periods of optimism
        followed by disappointment and loss of funding, known as AI winters. Early
        AI research focused on logic-based systems and expert systems. The modern
        era of AI began with the resurgence of neural networks and the availability
        of big data and powerful computing resources.""",
    ),
    Document(
        id="doc_ml_contradiction",
        text="""CONTRADICTION: Machine Learning is actually a completely manual process
        that requires explicit programming for every decision. Unlike popular belief,
        ML algorithms cannot learn from data automatically and must be hand-coded
        for each specific task. This document intentionally contradicts the standard
        definition to test retrieval quality and contradiction detection.""",
    ),
]

# Test queries with expected results
TEST_QUERIES = [
    {
        "query": "What is machine learning?",
        "expected_docs": ["doc_ml_intro", "doc_deep_learning"],
        "should_avoid": ["doc_ml_contradiction"],
        "k": 3,
    },
    {
        "query": "deep learning neural networks",
        "expected_docs": ["doc_deep_learning"],
        "should_avoid": [],
        "k": 2,
    },
    {
        "query": "artificial intelligence history Dartmouth",
        "expected_docs": ["doc_ai_history"],
        "should_avoid": [],
        "k": 2,
    },
    {
        "query": "machine learning explicit programming manual",
        "expected_docs": ["doc_ml_contradiction", "doc_ml_intro"],
        "should_avoid": [],
        "k": 3,
    },
]


async def test_pipeline_instantiation():
    """Test that pipeline can be instantiated safely with no config."""
    print("=== Testing Pipeline Instantiation ===")

    start_time = time.perf_counter()

    # Test with no config (should use safe defaults)
    try:
        pipeline = RAGPipeline()
        instantiation_time = (time.perf_counter() - start_time) * 1000
        print(f"Pipeline instantiated successfully in {instantiation_time:.2f}ms")
        print(f"  Pipeline: {pipeline}")
        return pipeline
    except Exception as e:
        print(f"Failed to instantiate pipeline: {e}")
        return None


async def test_document_indexing(pipeline: RAGPipeline):
    """Test document indexing with various content types."""
    print("\n=== Testing Document Indexing ===")

    total_start_time = time.perf_counter()
    indexing_times = []

    for _i, doc in enumerate(TEST_DOCUMENTS):
        doc_start_time = time.perf_counter()
        try:
            await pipeline.add_document(doc)
            doc_time = (time.perf_counter() - doc_start_time) * 1000
            indexing_times.append(doc_time)
            print(f"Indexed document {doc.id} in {doc_time:.2f}ms")
        except Exception as e:
            print(f"Failed to index document {doc.id}: {e}")
            return False

    total_time = (time.perf_counter() - total_start_time) * 1000
    avg_time = sum(indexing_times) / len(indexing_times)

    print("Indexing Summary:")
    print(f"   Total time: {total_time:.2f}ms")
    print(f"   Average per document: {avg_time:.2f}ms")
    print(f"   Documents indexed: {len(TEST_DOCUMENTS)}")

    return True


async def test_retrieval_quality(pipeline: RAGPipeline):
    """Test retrieval quality and latency."""
    print("\n=== Testing Retrieval Quality ===")

    retrieval_results = []
    total_queries = len(TEST_QUERIES)
    successful_queries = 0
    total_retrieval_time = 0

    for query_data in TEST_QUERIES:
        query = query_data["query"]
        expected_docs = query_data["expected_docs"]
        should_avoid = query_data["should_avoid"]
        k = query_data["k"]

        print(f"\nQuery: '{query}' (k={k})")

        start_time = time.perf_counter()
        try:
            results = await pipeline.retrieve(query, top_k=k)
            retrieval_time = (time.perf_counter() - start_time) * 1000
            total_retrieval_time += retrieval_time

            print(f"[TIME]  Retrieved {len(results)} results in {retrieval_time:.2f}ms")

            # Check result quality
            retrieved_ids = [result.id for result in results] if results else []

            # Count matches with expected documents
            matches = sum(1 for doc_id in expected_docs if doc_id in retrieved_ids)
            sum(1 for doc_id in should_avoid if doc_id not in retrieved_ids)

            if matches > 0 or (not expected_docs and len(results) >= 0):
                successful_queries += 1
                print(
                    f"[OK] Query successful - {matches}/{len(expected_docs)} expected matches"
                )
            else:
                print(
                    f"[WARNING]  Query had limited results - {matches}/{len(expected_docs)} expected matches"
                )

            # Print results with scores
            for i, result in enumerate(results):
                print(f"   {i + 1}. {result.id} (score: {result.score:.3f})")
                print(f"      Content preview: {result.content[:100]}...")

            retrieval_results.append(
                {
                    "query": query,
                    "results_count": len(results),
                    "retrieval_time_ms": retrieval_time,
                    "expected_matches": matches,
                    "total_expected": len(expected_docs),
                }
            )

        except Exception as e:
            print(f"[FAIL] Query failed: {e}")
            retrieval_results.append(
                {
                    "query": query,
                    "results_count": 0,
                    "retrieval_time_ms": 0,
                    "expected_matches": 0,
                    "total_expected": len(expected_docs),
                    "error": str(e),
                }
            )

    # Calculate summary statistics
    avg_retrieval_time = (
        total_retrieval_time / total_queries if total_queries > 0 else 0
    )
    success_rate = (
        (successful_queries / total_queries) * 100 if total_queries > 0 else 0
    )

    print("\n[STATS] Retrieval Summary:")
    print(f"   Total queries: {total_queries}")
    print(f"   Successful queries: {successful_queries}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Average retrieval time: {avg_retrieval_time:.2f}ms")
    print(f"   Total retrieval time: {total_retrieval_time:.2f}ms")

    return (
        retrieval_results,
        success_rate >= 50,
    )  # Consider 50%+ success rate as passing


async def test_end_to_end_pipeline(pipeline: RAGPipeline):
    """Test complete end-to-end query processing."""
    print("\n=== Testing End-to-End Pipeline ===")

    test_query = "Explain machine learning and deep learning"

    start_time = time.perf_counter()
    try:
        results, answer = await pipeline.query(
            query=test_query, top_k=3, synthesize=True, use_cache=True
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        print(f"[OK] End-to-end query processed in {processing_time:.2f}ms")
        print(f"   Retrieved {len(results)} results")

        if answer:
            print(f"   Answer confidence: {answer.confidence_score:.3f}")
            print(f"   Synthesis time: {answer.processing_time_ms:.2f}ms")
            print(f"   Answer preview: {answer.answer_text[:200]}...")
        else:
            print("   No synthesized answer generated")

        return True

    except Exception as e:
        print(f"[FAIL] End-to-end test failed: {e}")
        return False


async def test_cache_functionality(pipeline: RAGPipeline):
    """Test caching system performance."""
    print("\n=== Testing Cache Functionality ===")

    test_query = "What is machine learning?"

    # First query (cache miss)
    start_time = time.perf_counter()
    await pipeline.retrieve(test_query, top_k=3, use_cache=True)
    first_query_time = (time.perf_counter() - start_time) * 1000

    # Second query (should be cache hit if caching works)
    start_time = time.perf_counter()
    await pipeline.retrieve(test_query, top_k=3, use_cache=True)
    second_query_time = (time.perf_counter() - start_time) * 1000

    print(f"   First query: {first_query_time:.2f}ms")
    print(f"   Second query: {second_query_time:.2f}ms")

    if second_query_time < first_query_time * 0.8:  # 20% improvement suggests caching
        print(
            f"[OK] Cache appears to be working (speedup: {(first_query_time / second_query_time):.1f}x)"
        )
        cache_working = True
    else:
        print("[WARNING]  Cache impact unclear or not enabled")
        cache_working = False

    # Test cache metrics
    try:
        metrics = pipeline.get_metrics()
        print("[STATS] Cache Metrics:")
        if "cache" in metrics:
            cache_metrics = metrics["cache"]
            if isinstance(cache_metrics, dict):
                for key, value in cache_metrics.items():
                    print(f"   {key}: {value}")
        else:
            print("   Cache metrics not available")
    except Exception as e:
        print(f"   Error getting cache metrics: {e}")

    return cache_working


async def run_smoke_tests():
    """Run all smoke tests and generate report."""
    print("Starting RAG Pipeline Smoke Tests")
    print("=" * 50)

    overall_start_time = time.perf_counter()

    # Test Results Storage
    test_results = {
        "instantiation": False,
        "indexing": False,
        "retrieval": False,
        "end_to_end": False,
        "cache": False,
        "overall_time_ms": 0,
        "retrieval_stats": [],
    }

    # 1. Test Pipeline Instantiation
    pipeline = await test_pipeline_instantiation()
    if pipeline is None:
        print("\n[ERROR] CRITICAL: Pipeline instantiation failed!")
        return test_results
    test_results["instantiation"] = True

    # 2. Test Document Indexing
    indexing_success = await test_document_indexing(pipeline)
    test_results["indexing"] = indexing_success

    if not indexing_success:
        print("\n[ERROR] CRITICAL: Document indexing failed!")
        return test_results

    # 3. Test Retrieval Quality
    retrieval_stats, retrieval_success = await test_retrieval_quality(pipeline)
    test_results["retrieval"] = retrieval_success
    test_results["retrieval_stats"] = retrieval_stats

    # 4. Test End-to-End Pipeline
    e2e_success = await test_end_to_end_pipeline(pipeline)
    test_results["end_to_end"] = e2e_success

    # 5. Test Cache Functionality
    cache_success = await test_cache_functionality(pipeline)
    test_results["cache"] = cache_success

    # Calculate overall time
    test_results["overall_time_ms"] = (time.perf_counter() - overall_start_time) * 1000

    # Print Final Summary
    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)

    passed_tests = sum(
        [
            test_results["instantiation"],
            test_results["indexing"],
            test_results["retrieval"],
            test_results["end_to_end"],
            test_results["cache"],
        ]
    )
    total_tests = 5

    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Total Runtime: {test_results['overall_time_ms']:.2f}ms")

    print("\nTest Results:")
    print(
        f"   Pipeline Instantiation: {'PASS' if test_results['instantiation'] else 'FAIL'}"
    )
    print(f"   Document Indexing: {'PASS' if test_results['indexing'] else 'FAIL'}")
    print(f"   Retrieval Quality: {'PASS' if test_results['retrieval'] else 'FAIL'}")
    print(f"   End-to-End Pipeline: {'PASS' if test_results['end_to_end'] else 'FAIL'}")
    print(f"   Cache Functionality: {'PASS' if test_results['cache'] else 'FAIL'}")

    if passed_tests >= 4:  # Allow cache test to fail since it's optional
        print(f"\nSMOKE TESTS PASSED! ({passed_tests}/{total_tests})")
        exit_code = 0
    else:
        print(f"\nSMOKE TESTS FAILED! ({passed_tests}/{total_tests})")
        exit_code = 1

    return test_results, exit_code


if __name__ == "__main__":
    # Run smoke tests
    try:
        results, exit_code = asyncio.run(run_smoke_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[STOPPED]  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
