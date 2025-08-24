#!/usr/bin/env python3
"""
Fixed critical test of the RAG system to verify actual functionality.
"""

import asyncio
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path("src/production/rag/rag_system/core")))


def test_trust_scoring_actually_works():
    """Test if trust scoring produces meaningful results."""

    print("=== Testing Trust Scoring Reality Check ===")

    try:
        from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
        from codex_rag_integration import Document

        pipeline = BayesRAGEnhancedPipeline()

        # Create documents with different trust levels
        high_trust_doc = Document(
            id="high_trust",
            title="High Trust Document",
            content="This is a high trust document with lots of citations and quality content.",
            source_type="wikipedia",
            metadata={
                "parent_title": "High Trust Document",
                "trust_score": 0.95,
                "citation_count": 200,
                "source_quality": 0.9,
                "categories": ["Featured articles"],
            },
        )

        low_trust_doc = Document(
            id="low_trust",
            title="Low Trust Document",
            content="This is a low trust document with questionable content.",
            source_type="wikipedia",
            metadata={
                "parent_title": "Low Trust Document",
                "trust_score": 0.2,
                "citation_count": 5,
                "source_quality": 0.3,
                "categories": [],
            },
        )

        # Add trust scores to pipeline cache
        from bayesrag_codex_enhanced import TrustMetrics

        pipeline.trust_cache["High Trust Document"] = TrustMetrics(
            base_score=0.95, citation_count=200, source_quality=0.9
        )

        pipeline.trust_cache["Low Trust Document"] = TrustMetrics(base_score=0.2, citation_count=5, source_quality=0.3)

        pipeline.index_documents([high_trust_doc, low_trust_doc])

        # Test if trust weighting actually affects results
        async def test_trust_ranking():
            results, metrics = await pipeline.retrieve_with_trust(query="trust document content", k=2, trust_weight=0.5)

            if len(results) < 2:
                print("FAIL: Not enough results to compare trust ranking")
                return False

            # Check if results have trust scores
            first_result = results[0]
            second_result = results[1]

            if not hasattr(first_result, "trust_metrics") or first_result.trust_metrics is None:
                print("FAIL: First result missing trust metrics")
                return False

            if not hasattr(first_result, "bayesian_score"):
                print("FAIL: First result missing bayesian score")
                return False

            trust1 = first_result.trust_metrics.trust_score
            trust2 = second_result.trust_metrics.trust_score if second_result.trust_metrics else 0

            bayesian1 = first_result.bayesian_score
            bayesian2 = second_result.bayesian_score

            print(f"Result 1: Trust={trust1:.3f}, Bayesian={bayesian1:.3f}")
            print(f"Result 2: Trust={trust2:.3f}, Bayesian={bayesian2:.3f}")

            # High trust document should rank higher
            if trust1 <= trust2:
                print("WARN: Trust scoring may not be working properly")
                return False

            print("PASS: Trust scoring affects ranking")
            return True

        return asyncio.run(test_trust_ranking())

    except Exception as e:
        print(f"FAIL: Exception in trust scoring test: {e}")
        traceback.print_exc()
        return False


def test_cache_with_fixed_async():
    """Test semantic cache with proper async handling."""

    print("=== Testing Semantic Cache (Fixed) ===")

    async def async_cache_test():
        try:
            from semantic_cache_advanced import SemanticMultiTierCache

            # Create cache with prefetch disabled to avoid async issues
            cache = SemanticMultiTierCache(enable_prefetch=False)

            # Test basic cache operations
            test_query = "machine learning algorithms"
            test_results = [{"text": "ML algorithms include linear regression"}]

            # Should be miss initially
            result = await cache.get(test_query)
            if result is not None:
                print("FAIL: Should be cache miss initially")
                return False

            # Store in cache
            await cache.set(test_query, test_results, trust_score=0.8)

            # Should be hit now
            result = await cache.get(test_query)
            if result is None:
                print("FAIL: Should be cache hit after storing")
                return False

            print("PASS: Basic cache operations work")

            # Test semantic matching
            similar_query = "ML algorithm types"
            semantic_result = await cache.get(similar_query, semantic_threshold=0.7)

            if semantic_result is not None:
                print("PASS: Semantic matching works")
                if semantic_result[1].get("semantic_match"):
                    print(f"Similarity score: {semantic_result[1].get('similarity_score', 0):.3f}")
            else:
                print("INFO: No semantic match found (may be expected)")

            # Test cache metrics
            metrics = cache.get_metrics()
            print(f"Cache hit rate: {metrics['hit_rate']:.2%}")
            print(f"Cache sizes: {metrics['cache_sizes']}")

            return True

        except Exception as e:
            print(f"FAIL: Exception in cache test: {e}")
            traceback.print_exc()
            return False

    return asyncio.run(async_cache_test())


def test_monitoring_with_fixed_async():
    """Test production monitoring with proper async handling."""

    print("=== Testing Production Monitoring (Fixed) ===")

    async def async_monitoring_test():
        try:
            from codex_rag_integration import CODEXRAGPipeline
            from production_monitoring import HealthStatus, ProductionMonitor
            from semantic_cache_advanced import SemanticMultiTierCache

            # Create components
            pipeline = CODEXRAGPipeline()
            cache = SemanticMultiTierCache(enable_prefetch=False)

            # Create monitor
            monitor = ProductionMonitor(pipeline, cache)

            # Test health checks
            health_results = await monitor.run_health_checks()

            if "status" not in health_results:
                print("FAIL: Health check missing status")
                return False

            print(f"Health status: {health_results['status']}")
            print(f"Health checks: {len(health_results.get('checks', {}))}")

            # Test performance recording
            monitor.record_request(latency_ms=85.0, success=True)
            monitor.record_request(latency_ms=92.0, success=True)
            monitor.record_request(latency_ms=78.0, success=True)

            # Test circuit breakers
            breaker_name = "test_service"

            # Add a test circuit breaker
            from production_monitoring import CircuitBreaker

            monitor.circuit_breakers[breaker_name] = CircuitBreaker(name=breaker_name, failure_threshold=3)

            # Test circuit breaker functionality
            breaker = monitor.circuit_breakers[breaker_name]

            # Should start closed
            if not breaker.can_proceed():
                print("FAIL: Circuit breaker should start closed")
                return False

            # Record failures
            for _ in range(5):
                breaker.record_failure()

            # Should now be open
            if breaker.can_proceed():
                print("FAIL: Circuit breaker should be open after failures")
                return False

            print("PASS: Circuit breaker opens after failures")

            # Test graceful degradation
            monitor.health_status = HealthStatus.DEGRADED

            # This should work even in degraded state
            dashboard = monitor.get_dashboard_data()

            if "health" not in dashboard:
                print("FAIL: Dashboard missing health data")
                return False

            print("PASS: Dashboard data available")
            print(f"Dashboard health: {dashboard['health']['status']}")

            return True

        except Exception as e:
            print(f"FAIL: Exception in monitoring test: {e}")
            traceback.print_exc()
            return False

    return asyncio.run(async_monitoring_test())


def test_actual_performance():
    """Test if the system meets claimed performance targets."""

    print("=== Testing Actual Performance Claims ===")

    async def performance_test():
        try:
            from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
            from codex_rag_integration import Document

            pipeline = BayesRAGEnhancedPipeline()

            # Index several documents
            docs = []
            for i in range(10):
                doc = Document(
                    id=f"perf_test_{i}",
                    title=f"Performance Test Document {i}",
                    content=f"This is performance test document number {i}. " * 10
                    + "It contains content about artificial intelligence, machine learning, "
                    + "deep learning, neural networks, and natural language processing. " * 5,
                    source_type="test",
                    metadata={"trust_score": 0.5 + (i * 0.05)},
                )
                docs.append(doc)

            print("Indexing 10 documents...")
            start = time.perf_counter()
            stats = pipeline.index_documents(docs)
            index_time = (time.perf_counter() - start) * 1000

            print(f"Indexing took {index_time:.1f}ms for {stats['documents_processed']} docs")

            if index_time > 5000:  # 5 seconds
                print("WARN: Indexing is slow")

            # Test retrieval latency
            queries = [
                "artificial intelligence",
                "machine learning algorithms",
                "deep neural networks",
                "natural language processing",
                "performance test document",
            ]

            latencies = []

            print("Testing retrieval performance...")

            for query in queries:
                # Cold start
                start = time.perf_counter()
                results, metrics = await pipeline.retrieve_with_trust(query, k=5)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)

                print(f"Query: '{query}' -> {latency:.1f}ms, {len(results)} results")

                # Warm cache test
                start = time.perf_counter()
                cached_results, cached_metrics = await pipeline.retrieve_with_trust(query, k=5)
                cached_latency = (time.perf_counter() - start) * 1000

                cache_hit = cached_metrics.get("cache_hit", False)
                print(f"  Cached: {cached_latency:.1f}ms (hit: {cache_hit})")

            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            print("\nPerformance Summary:")
            print(f"Average latency: {avg_latency:.1f}ms")
            print(f"Max latency: {max_latency:.1f}ms")

            # Check against claimed targets
            target_uncached = 100  # <100ms claimed

            meets_uncached_target = avg_latency < target_uncached
            print(f"Meets <100ms target: {meets_uncached_target}")

            if not meets_uncached_target:
                print(f"FAIL: Average latency {avg_latency:.1f}ms exceeds target")
                return False
            print("PASS: Performance meets targets")

            return True

        except Exception as e:
            print(f"FAIL: Exception in performance test: {e}")
            traceback.print_exc()
            return False

    return asyncio.run(performance_test())


def main():
    """Run critical reality checks."""

    print("CRITICAL RAG SYSTEM REALITY CHECK")
    print("=" * 60)

    tests = [
        ("Trust Scoring Reality", test_trust_scoring_actually_works),
        ("Cache Functionality", test_cache_with_fixed_async),
        ("Monitoring System", test_monitoring_with_fixed_async),
        ("Performance Claims", test_actual_performance),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")

        try:
            result = test_func()
            results[test_name] = result

            if result:
                print(f"✓ PASS: {test_name}")
            else:
                print(f"✗ FAIL: {test_name}")

        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
            results[test_name] = False

    # Final verdict
    print("\n" + "=" * 60)
    print("REALITY CHECK VERDICT")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nScore: {passed}/{total} critical tests passed")

    if passed == total:
        print("\n🎯 VERDICT: RAG system is ACTUALLY FUNCTIONAL")
        print("   - Trust scoring works as claimed")
        print("   - Caching system functions properly")
        print("   - Monitoring provides real capabilities")
        print("   - Performance meets stated targets")
    elif passed >= total * 0.75:
        print("\n⚠️  VERDICT: RAG system is MOSTLY FUNCTIONAL")
        print("   - Core features work but some issues exist")
        print("   - May have implementation gaps in advanced features")
    else:
        print("\n❌ VERDICT: RAG system is BROKEN or MOSTLY STUBS")
        print("   - Critical functionality fails")
        print("   - Implementation appears incomplete")
        print("   - Claims not supported by actual performance")

    return passed >= total * 0.75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
