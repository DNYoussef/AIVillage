#!/usr/bin/env python3
"""Production System Benchmark Suite.

Post-cleanup performance validation for AIVillage systems.
Provides comprehensive benchmarking of compression, evolution, and RAG pipelines.

Usage:
    python production_benchmark_suite.py
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

try:
    import psutil
except ImportError:
    print("Error: 'psutil' package required. Install with: pip install psutil")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark_suite.log"),
    ],
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance during benchmarks.

    Tracks execution time, memory usage, and CPU utilization
    during benchmark execution.
    """

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.start_time: float | None = None
        self.start_memory: int | None = None
        self.peak_memory: int = 0

    def start(self) -> None:
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = self.start_memory

    def update_peak_memory(self) -> None:
        current = psutil.virtual_memory().used
        self.peak_memory = max(self.peak_memory, current)

    def get_metrics(self) -> dict[str, Any]:
        if self.start_time is None:
            return {}

        elapsed = time.time() - self.start_time
        memory_delta = self.peak_memory - self.start_memory

        return {
            "elapsed_time_seconds": round(elapsed, 3),
            "memory_delta_mb": round(memory_delta / (1024 * 1024), 2),
            "peak_memory_mb": round(self.peak_memory / (1024 * 1024), 2),
            "cpu_percent": psutil.cpu_percent(interval=1),
        }


class CompressionBenchmark:
    """Benchmark compression pipeline performance."""

    def __init__(self) -> None:
        self.results = {}

    def run_compression_test(self) -> dict[str, Any]:
        """Test compression pipeline with sample model."""
        logger.info("Starting compression pipeline benchmark...")
        monitor = PerformanceMonitor()
        monitor.start()

        try:
            # Import compression modules
            # Create test data (simulated model weights)
            import torch

            from agent_forge.compression import BitNetCompressor, SeedLMCompressor, VPTQCompressor

            test_tensor = torch.randn(1000, 1000)  # 4MB test tensor
            original_size = test_tensor.numel() * 4  # 4 bytes per float32

            results = {
                "original_size_mb": round(original_size / (1024 * 1024), 2),
                "compressions": {},
            }

            # Test each compression method
            compressors = [
                ("BitNet", BitNetCompressor()),
                ("SeedLM", SeedLMCompressor()),
                ("VPTQ", VPTQCompressor()),
            ]

            for name, compressor in compressors:
                try:
                    start_time = time.time()
                    compressed = compressor.compress(test_tensor)
                    compression_time = time.time() - start_time

                    # Calculate compression ratio
                    compressed_size = len(compressed) if isinstance(compressed, bytes) else compressed.numel() * 4
                    ratio = original_size / compressed_size

                    results["compressions"][name] = {
                        "compression_ratio": round(ratio, 2),
                        "compressed_size_mb": round(compressed_size / (1024 * 1024), 2),
                        "compression_time_seconds": round(compression_time, 3),
                        "status": "success",
                    }

                    logger.info(f"{name}: {ratio:.1f}x compression in {compression_time:.3f}s")

                except Exception as e:
                    results["compressions"][name] = {"status": "error", "error": str(e)}
                    logger.exception(f"{name} compression failed: {e}")

                monitor.update_peak_memory()

            results["performance"] = monitor.get_metrics()
            return results

        except ImportError as e:
            logger.exception(f"Compression modules not available: {e}")
            return {"status": "error", "error": f"Import failed: {e}"}
        except Exception as e:
            logger.exception(f"Compression benchmark failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class EvolutionBenchmark:
    """Benchmark evolution system performance."""

    def __init__(self) -> None:
        self.results = {}

    def run_evolution_test(self) -> dict[str, Any]:
        """Test evolution system with mini tournament."""
        logger.info("Starting evolution system benchmark...")
        monitor = PerformanceMonitor()
        monitor.start()

        try:
            # Import evolution modules
            from agent_forge.evolution import EvolutionOrchestrator

            # Create minimal evolution test
            orchestrator = EvolutionOrchestrator()

            # Test configuration
            config = {
                "population_size": 4,  # Small for quick test
                "generations": 2,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
            }

            start_time = time.time()

            # Run mini evolution
            result = orchestrator.run_mini_evolution(config)

            evolution_time = time.time() - start_time
            monitor.update_peak_memory()

            results = {
                "evolution_time_seconds": round(evolution_time, 3),
                "population_size": config["population_size"],
                "generations": config["generations"],
                "final_fitness": result.get("best_fitness", 0) if result else 0,
                "convergence_generation": (result.get("convergence_gen", -1) if result else -1),
                "performance": monitor.get_metrics(),
                "status": "success" if result else "failed",
            }

            logger.info(f"Evolution completed in {evolution_time:.3f}s")
            return results

        except ImportError as e:
            logger.exception(f"Evolution modules not available: {e}")
            return {"status": "error", "error": f"Import failed: {e}"}
        except Exception as e:
            logger.exception(f"Evolution benchmark failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class RAGBenchmark:
    """Benchmark RAG pipeline performance."""

    def __init__(self) -> None:
        self.results = {}

    def run_rag_test(self) -> dict[str, Any]:
        """Test RAG pipeline with sample documents and queries."""
        logger.info("Starting RAG pipeline benchmark...")
        monitor = PerformanceMonitor()
        monitor.start()

        try:
            # Import RAG modules
            from agent_forge.rag_integration import RAGManager

            # Create test documents
            test_docs = [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Deep learning uses neural networks with multiple layers to process data.",
                "Natural language processing enables computers to understand human language.",
                "Computer vision allows machines to interpret and understand visual information.",
                "Reinforcement learning teaches agents to make decisions through trial and error.",
            ]

            # Initialize RAG system
            rag_manager = RAGManager()

            # Test document indexing
            index_start = time.time()
            rag_manager.index_documents(test_docs)
            index_time = time.time() - index_start

            monitor.update_peak_memory()

            # Test queries
            test_queries = [
                "What is machine learning?",
                "How does deep learning work?",
                "What is computer vision?",
            ]

            query_results = []
            total_query_time = 0

            for query in test_queries:
                query_start = time.time()
                result = rag_manager.query(query, top_k=3)
                query_time = time.time() - query_start
                total_query_time += query_time

                query_results.append(
                    {
                        "query": query,
                        "response_time_seconds": round(query_time, 3),
                        "results_count": len(result.get("results", [])),
                        "confidence": result.get("confidence", 0),
                    }
                )

                monitor.update_peak_memory()

            avg_query_time = total_query_time / len(test_queries)

            results = {
                "indexing_time_seconds": round(index_time, 3),
                "documents_indexed": len(test_docs),
                "avg_query_time_seconds": round(avg_query_time, 3),
                "total_queries": len(test_queries),
                "query_results": query_results,
                "performance": monitor.get_metrics(),
                "status": "success",
            }

            logger.info(f"RAG indexing: {index_time:.3f}s, avg query: {avg_query_time:.3f}s")
            return results

        except ImportError as e:
            logger.exception(f"RAG modules not available: {e}")
            return {"status": "error", "error": f"Import failed: {e}"}
        except Exception as e:
            logger.exception(f"RAG benchmark failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class ProductionBenchmarkSuite:
    """Main benchmark suite orchestrator."""

    def __init__(self) -> None:
        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all production system benchmarks."""
        logger.info("Starting production benchmark suite...")

        overall_start = time.time()

        # System info
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "timestamp": self.timestamp,
        }

        results = {"system_info": system_info, "benchmarks": {}}

        # Run compression benchmark
        compression_bench = CompressionBenchmark()
        results["benchmarks"]["compression"] = compression_bench.run_compression_test()

        # Run evolution benchmark
        evolution_bench = EvolutionBenchmark()
        results["benchmarks"]["evolution"] = evolution_bench.run_evolution_test()

        # Run RAG benchmark
        rag_bench = RAGBenchmark()
        results["benchmarks"]["rag"] = rag_bench.run_rag_test()

        # Overall timing
        total_time = time.time() - overall_start
        results["total_benchmark_time_seconds"] = round(total_time, 3)

        logger.info(f"All benchmarks completed in {total_time:.3f}s")
        return results

    def save_results(self, results: dict[str, Any], filename: str | None = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        results_dir = Path(__file__).parent / "benchmark_results"
        results_dir.mkdir(exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return str(filepath)

    def compare_with_baseline(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compare current results with baseline if available."""
        try:
            baseline_path = Path(__file__).parent / "benchmark_results" / "baseline.json"

            if not baseline_path.exists():
                logger.info("No baseline found, current results will serve as baseline")
                return {
                    "status": "no_baseline",
                    "message": "First run - establishing baseline",
                }

            with open(baseline_path) as f:
                baseline = json.load(f)

            comparison = {"status": "comparison_available", "changes": {}}

            # Compare compression ratios
            if "compression" in results["benchmarks"] and "compression" in baseline.get("benchmarks", {}):
                current_comp = results["benchmarks"]["compression"]
                baseline_comp = baseline["benchmarks"]["compression"]

                if "compressions" in current_comp and "compressions" in baseline_comp:
                    for method in current_comp["compressions"]:
                        if method in baseline_comp["compressions"]:
                            current_ratio = current_comp["compressions"][method].get("compression_ratio", 0)
                            baseline_ratio = baseline_comp["compressions"][method].get("compression_ratio", 0)

                            if baseline_ratio > 0:
                                change_pct = ((current_ratio - baseline_ratio) / baseline_ratio) * 100
                                comparison["changes"][f"compression_{method}_ratio"] = {
                                    "current": current_ratio,
                                    "baseline": baseline_ratio,
                                    "change_percent": round(change_pct, 2),
                                }

            return comparison

        except Exception as e:
            logger.exception(f"Baseline comparison failed: {e}")
            return {"status": "comparison_failed", "error": str(e)}


def main() -> int | None:
    """Run the production benchmark suite."""
    print("=" * 60)
    print("AIVillage Production System Benchmark Suite")
    print("Post-cleanup performance validation")
    print("=" * 60)

    suite = ProductionBenchmarkSuite()

    try:
        # Run all benchmarks
        results = suite.run_all_benchmarks()

        # Save results
        results_file = suite.save_results(results)

        # Compare with baseline
        comparison = suite.compare_with_baseline(results)
        results["baseline_comparison"] = comparison

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)

        # Compression results
        comp_results = results["benchmarks"].get("compression", {})
        if comp_results.get("status") != "error":
            print("\n📦 COMPRESSION PIPELINE:")
            if "compressions" in comp_results:
                for method, data in comp_results["compressions"].items():
                    if data.get("status") == "success":
                        ratio = data.get("compression_ratio", 0)
                        time_taken = data.get("compression_time_seconds", 0)
                        status = "✅ PASS" if ratio >= 4.0 else "⚠️  BELOW TARGET"
                        print(f"  {method}: {ratio:.1f}x compression ({time_taken:.3f}s) {status}")
                    else:
                        print(f"  {method}: ❌ FAILED - {data.get('error', 'Unknown error')}")
        else:
            print(f"\n📦 COMPRESSION PIPELINE: ❌ FAILED - {comp_results.get('error', 'Unknown error')}")

        # Evolution results
        evo_results = results["benchmarks"].get("evolution", {})
        if evo_results.get("status") != "error":
            print("\n🧬 EVOLUTION SYSTEM:")
            evo_time = evo_results.get("evolution_time_seconds", 0)
            fitness = evo_results.get("final_fitness", 0)
            status = "✅ PASS" if evo_time < 60.0 else "⚠️  SLOW"
            print(f"  Evolution time: {evo_time:.3f}s {status}")
            print(f"  Final fitness: {fitness:.3f}")
        else:
            print(f"\n🧬 EVOLUTION SYSTEM: ❌ FAILED - {evo_results.get('error', 'Unknown error')}")

        # RAG results
        rag_results = results["benchmarks"].get("rag", {})
        if rag_results.get("status") != "error":
            print("\n🔍 RAG PIPELINE:")
            index_time = rag_results.get("indexing_time_seconds", 0)
            query_time = rag_results.get("avg_query_time_seconds", 0)
            query_status = "✅ PASS" if query_time < 2.0 else "⚠️  SLOW"
            print(f"  Indexing time: {index_time:.3f}s")
            print(f"  Avg query time: {query_time:.3f}s {query_status}")
        else:
            print(f"\n🔍 RAG PIPELINE: ❌ FAILED - {rag_results.get('error', 'Unknown error')}")

        # Overall performance
        total_time = results.get("total_benchmark_time_seconds", 0)
        print(f"\n⏱️  TOTAL BENCHMARK TIME: {total_time:.3f}s")

        # Baseline comparison
        if comparison.get("status") == "comparison_available":
            print("\n📊 BASELINE COMPARISON:")
            changes = comparison.get("changes", {})
            if changes:
                for metric, data in changes.items():
                    change_pct = data.get("change_percent", 0)
                    direction = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                    print(
                        f"  {metric}: {data['current']:.2f} vs {data['baseline']:.2f} ({change_pct:+.1f}%) {direction}"
                    )
            else:
                print("  No significant changes detected")

        print(f"\n📄 Full results saved to: {results_file}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.exception(f"Benchmark suite failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
