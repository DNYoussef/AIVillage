#!/usr/bin/env python3
"""RAG Pipeline Evaluation Benchmarks - Latency and Accuracy Testing."""

import asyncio
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "src" / "production" / "rag" / "rag_system" / "core"),
)

from codex_rag_integration import CODEXRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluationBenchmarks:
    """Comprehensive RAG pipeline evaluation and benchmarks."""

    def __init__(self) -> None:
        self.rag_pipeline = None

        # Evaluation queries with expected relevance
        self.test_queries = [
            {
                "query": "machine learning algorithms",
                "expected_keywords": [
                    "machine",
                    "learning",
                    "algorithm",
                    "neural",
                    "training",
                ],
            },
            {
                "query": "artificial intelligence applications",
                "expected_keywords": [
                    "artificial",
                    "intelligence",
                    "AI",
                    "application",
                    "computer",
                ],
            },
            {
                "query": "quantum physics principles",
                "expected_keywords": [
                    "quantum",
                    "physics",
                    "particle",
                    "mechanics",
                    "theory",
                ],
            },
            {
                "query": "climate change effects",
                "expected_keywords": [
                    "climate",
                    "change",
                    "global",
                    "warming",
                    "environment",
                ],
            },
            {
                "query": "renewable energy sources",
                "expected_keywords": [
                    "renewable",
                    "energy",
                    "solar",
                    "wind",
                    "sustainable",
                ],
            },
            {
                "query": "genetic engineering techniques",
                "expected_keywords": [
                    "genetic",
                    "engineering",
                    "DNA",
                    "gene",
                    "biotechnology",
                ],
            },
            {
                "query": "space exploration missions",
                "expected_keywords": [
                    "space",
                    "exploration",
                    "NASA",
                    "mission",
                    "astronaut",
                ],
            },
            {
                "query": "computer science fundamentals",
                "expected_keywords": [
                    "computer",
                    "science",
                    "programming",
                    "algorithm",
                    "data",
                ],
            },
            {
                "query": "medical breakthrough treatments",
                "expected_keywords": [
                    "medical",
                    "treatment",
                    "therapy",
                    "medicine",
                    "health",
                ],
            },
            {
                "query": "economic policy analysis",
                "expected_keywords": [
                    "economic",
                    "policy",
                    "finance",
                    "market",
                    "government",
                ],
            },
            # Speed test queries - simple and complex
            {
                "query": "What is science?",
                "expected_keywords": ["science", "research", "knowledge", "method"],
            },
            {
                "query": "How does photosynthesis work in plants and what are the molecular mechanisms involved?",
                "expected_keywords": [
                    "photosynthesis",
                    "plant",
                    "chlorophyll",
                    "carbon",
                    "oxygen",
                ],
            },
            # Cache test queries (duplicates)
            {
                "query": "machine learning algorithms",
                "expected_keywords": [
                    "machine",
                    "learning",
                    "algorithm",
                    "neural",
                    "training",
                ],
            },
            {
                "query": "artificial intelligence applications",
                "expected_keywords": [
                    "artificial",
                    "intelligence",
                    "AI",
                    "application",
                    "computer",
                ],
            },
        ]

        # Performance benchmarks
        self.latency_targets = {
            "max_latency_ms": 100,  # CODEX requirement: <100ms
            "p95_latency_ms": 80,
            "p99_latency_ms": 95,
            "avg_latency_ms": 50,
        }

    async def initialize_rag_pipeline(self) -> bool:
        """Initialize the RAG pipeline."""
        try:
            self.rag_pipeline = CODEXRAGPipeline()
            logger.info("RAG pipeline initialized for evaluation")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize RAG pipeline: {e}")
            return False

    def calculate_relevance_score(self, query: dict[str, Any], results: list[Any]) -> float:
        """Calculate relevance score based on keyword matching."""
        expected_keywords = [kw.lower() for kw in query["expected_keywords"]]
        total_score = 0.0

        for i, result in enumerate(results[:5]):  # Top 5 results
            text = result.text.lower()
            matched_keywords = sum(1 for kw in expected_keywords if kw in text)
            keyword_score = matched_keywords / len(expected_keywords)

            # Weight higher-ranked results more
            position_weight = 1.0 / (i + 1)
            total_score += keyword_score * position_weight

        return total_score / min(len(results), 5)

    async def run_latency_benchmark(self, queries: list[str], iterations: int = 3) -> dict[str, Any]:
        """Run latency benchmarks with multiple iterations."""
        all_latencies = []
        cache_hits = 0
        vector_searches = 0
        keyword_searches = 0

        logger.info(f"Running latency benchmark with {len(queries)} queries, {iterations} iterations each")

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            for i, query_data in enumerate(queries):
                query = query_data["query"]

                start_time = time.perf_counter()
                results, metrics = await self.rag_pipeline.retrieve(query, k=10, use_cache=True)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                all_latencies.append(latency_ms)

                # Track metrics
                if metrics.get("cache_hit", False):
                    cache_hits += 1
                if metrics.get("vector_search", False):
                    vector_searches += 1
                if metrics.get("keyword_search", False):
                    keyword_searches += 1

                logger.debug(f"Query {i + 1}: {latency_ms:.2f}ms (cache: {metrics.get('cache_hit', False)})")

        # Calculate statistics
        avg_latency = statistics.mean(all_latencies)
        p50_latency = statistics.median(all_latencies)
        p95_latency = statistics.quantiles(all_latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(all_latencies, n=100)[98]  # 99th percentile
        max_latency = max(all_latencies)
        min_latency = min(all_latencies)

        total_queries = len(queries) * iterations

        return {
            "total_queries": total_queries,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "max_latency_ms": max_latency,
            "min_latency_ms": min_latency,
            "cache_hit_rate": cache_hits / total_queries,
            "vector_search_rate": vector_searches / total_queries,
            "keyword_search_rate": keyword_searches / total_queries,
            "all_latencies": all_latencies,
            "meets_target": avg_latency < self.latency_targets["avg_latency_ms"],
        }

    async def run_accuracy_benchmark(self, queries: list[dict[str, Any]]) -> dict[str, Any]:
        """Run accuracy benchmarks based on relevance scoring."""
        relevance_scores = []
        retrieval_counts = []

        logger.info(f"Running accuracy benchmark with {len(queries)} test queries")

        for i, query_data in enumerate(queries):
            query = query_data["query"]

            results, metrics = await self.rag_pipeline.retrieve(query, k=10, use_cache=False)

            if results:
                relevance_score = self.calculate_relevance_score(query_data, results)
                relevance_scores.append(relevance_score)
                retrieval_counts.append(len(results))

                logger.debug(f"Query {i + 1}: relevance={relevance_score:.3f}, results={len(results)}")
            else:
                relevance_scores.append(0.0)
                retrieval_counts.append(0)
                logger.warning(f"Query {i + 1}: No results returned")

        # Calculate accuracy metrics
        avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
        avg_retrieval_count = statistics.mean(retrieval_counts) if retrieval_counts else 0.0
        successful_queries = sum(1 for score in relevance_scores if score > 0.1)  # 10% minimum relevance

        return {
            "total_test_queries": len(queries),
            "successful_queries": successful_queries,
            "success_rate": successful_queries / len(queries),
            "avg_relevance_score": avg_relevance,
            "avg_retrieval_count": avg_retrieval_count,
            "relevance_scores": relevance_scores,
            "retrieval_counts": retrieval_counts,
            "high_relevance_queries": sum(1 for score in relevance_scores if score > 0.5),  # 50% relevance
        }

    async def run_stress_test(self, concurrent_queries: int = 10, total_requests: int = 100) -> dict[str, Any]:
        """Run concurrent stress test to evaluate performance under load."""
        logger.info(f"Running stress test: {concurrent_queries} concurrent queries, {total_requests} total requests")

        query_pool = [q["query"] for q in self.test_queries[:5]]  # Use first 5 queries

        async def single_query(query_idx: int):
            query = query_pool[query_idx % len(query_pool)]
            start_time = time.perf_counter()

            try:
                results, metrics = await self.rag_pipeline.retrieve(query, k=5, use_cache=True)
                end_time = time.perf_counter()

                return {
                    "success": True,
                    "latency_ms": (end_time - start_time) * 1000,
                    "results_count": len(results),
                    "cache_hit": metrics.get("cache_hit", False),
                }
            except Exception as e:
                end_time = time.perf_counter()
                logger.exception(f"Query failed: {e}")
                return {
                    "success": False,
                    "latency_ms": (end_time - start_time) * 1000,
                    "error": str(e),
                }

        # Run concurrent queries in batches
        all_results = []
        batch_size = concurrent_queries

        start_time = time.perf_counter()

        for batch_start in range(0, total_requests, batch_size):
            batch_end = min(batch_start + batch_size, total_requests)
            batch_tasks = [single_query(i) for i in range(batch_start, batch_end)]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)

            logger.info(f"Completed batch {batch_start // batch_size + 1} ({len(batch_results)} queries)")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Analyze results
        successful_requests = [r for r in all_results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = len(all_results) - len(successful_requests)

        if successful_requests:
            latencies = [r["latency_ms"] for r in successful_requests]
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            cache_hits = sum(1 for r in successful_requests if r.get("cache_hit", False))
        else:
            avg_latency = max_latency = 0
            cache_hits = 0

        return {
            "total_requests": total_requests,
            "concurrent_queries": concurrent_queries,
            "successful_requests": len(successful_requests),
            "failed_requests": failed_requests,
            "success_rate": len(successful_requests) / total_requests,
            "total_time_s": total_time,
            "requests_per_second": total_requests / total_time,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "cache_hit_rate": (cache_hits / len(successful_requests) if successful_requests else 0),
        }

    async def run_comprehensive_evaluation(self) -> dict[str, Any]:
        """Run all evaluation benchmarks."""
        logger.info("Starting comprehensive RAG evaluation")

        # Initialize pipeline
        if not await self.initialize_rag_pipeline():
            msg = "Failed to initialize RAG pipeline"
            raise RuntimeError(msg)

        # Get initial pipeline metrics
        initial_metrics = self.rag_pipeline.get_performance_metrics()
        index_size = initial_metrics.get(
            "index_size",
            (self.rag_pipeline.index.ntotal if hasattr(self.rag_pipeline, "index") else 0),
        )
        logger.info(f"Initial pipeline state: {index_size} vectors indexed")

        evaluation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "initial_pipeline_metrics": initial_metrics,
        }

        # 1. Latency Benchmark
        logger.info("=== Running Latency Benchmark ===")
        latency_results = await self.run_latency_benchmark(self.test_queries)
        evaluation_results["latency_benchmark"] = latency_results

        # 2. Accuracy Benchmark
        logger.info("=== Running Accuracy Benchmark ===")
        accuracy_results = await self.run_accuracy_benchmark(self.test_queries)
        evaluation_results["accuracy_benchmark"] = accuracy_results

        # 3. Stress Test
        logger.info("=== Running Stress Test ===")
        stress_results = await self.run_stress_test(concurrent_queries=5, total_requests=50)
        evaluation_results["stress_test"] = stress_results

        # 4. Final pipeline metrics
        final_metrics = self.rag_pipeline.get_performance_metrics()
        evaluation_results["final_pipeline_metrics"] = final_metrics

        # 5. Overall Assessment
        overall_assessment = self.generate_assessment(evaluation_results)
        evaluation_results["overall_assessment"] = overall_assessment

        logger.info("Comprehensive evaluation completed")
        return evaluation_results

    def generate_assessment(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall assessment of RAG pipeline performance."""
        latency_results = results["latency_benchmark"]
        accuracy_results = results["accuracy_benchmark"]
        stress_results = results["stress_test"]

        # Performance scores (0-100)
        latency_score = 100 if latency_results["meets_target"] else max(0, 100 - latency_results["avg_latency_ms"])
        accuracy_score = accuracy_results["avg_relevance_score"] * 100
        reliability_score = stress_results["success_rate"] * 100

        overall_score = (latency_score + accuracy_score + reliability_score) / 3

        # Grade based on score
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": overall_score,
            "grade": grade,
            "latency_score": latency_score,
            "accuracy_score": accuracy_score,
            "reliability_score": reliability_score,
            "meets_codex_requirements": (
                latency_results["avg_latency_ms"] < 100
                and accuracy_results["success_rate"] > 0.8
                and stress_results["success_rate"] > 0.95
            ),
            "recommendations": self.generate_recommendations(results),
        }

    def generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        latency_results = results["latency_benchmark"]
        accuracy_results = results["accuracy_benchmark"]
        stress_results = results["stress_test"]

        # Latency recommendations
        if latency_results["avg_latency_ms"] > 100:
            recommendations.append("Optimize embedding model or use GPU acceleration to meet <100ms latency target")

        if latency_results["cache_hit_rate"] < 0.3:
            recommendations.append("Increase cache size or adjust caching strategy to improve cache hit rate")

        # Accuracy recommendations
        if accuracy_results["avg_relevance_score"] < 0.7:
            recommendations.append("Consider fine-tuning embedding model or adjusting chunking strategy")

        if accuracy_results["success_rate"] < 0.9:
            recommendations.append("Review query preprocessing and improve document quality filtering")

        # Reliability recommendations
        if stress_results["success_rate"] < 0.95:
            recommendations.append("Improve error handling and add connection pooling for better reliability")

        if stress_results["avg_latency_ms"] > latency_results["avg_latency_ms"] * 1.5:
            recommendations.append("Consider horizontal scaling or load balancing for concurrent queries")

        return recommendations


async def main():
    """Main function to run RAG evaluation benchmarks."""
    try:
        evaluation = RAGEvaluationBenchmarks()
        results = await evaluation.run_comprehensive_evaluation()

        # Save results
        results_file = Path("data/rag_evaluation_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        assessment = results["overall_assessment"]
        latency = results["latency_benchmark"]
        accuracy = results["accuracy_benchmark"]
        stress = results["stress_test"]

        print("\n=== RAG Pipeline Evaluation Results ===")
        print(f"Overall Score: {assessment['overall_score']:.1f}/100 (Grade: {assessment['grade']})")
        print(f"Meets CODEX Requirements: {'YES' if assessment['meets_codex_requirements'] else 'NO'}")
        print("\n--- Performance Metrics ---")
        print(f"Average Latency: {latency['avg_latency_ms']:.2f}ms (target: <100ms)")
        print(f"P95 Latency: {latency['p95_latency_ms']:.2f}ms")
        print(f"Cache Hit Rate: {latency['cache_hit_rate']:.1%}")
        print(f"Accuracy Score: {accuracy['avg_relevance_score']:.3f}")
        print(f"Success Rate: {accuracy['success_rate']:.1%}")
        print(f"Stress Test Success: {stress['success_rate']:.1%}")
        print(f"Requests/Second: {stress['requests_per_second']:.1f}")

        if assessment["recommendations"]:
            print("\n--- Recommendations ---")
            for i, rec in enumerate(assessment["recommendations"], 1):
                print(f"{i}. {rec}")

        print(f"\nDetailed results saved to: {results_file}")

        return results

    except Exception as e:
        logger.exception(f"RAG evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
