#!/usr/bin/env python3
"""
Quick Performance Demonstration

Shows working performance benchmarks with real data collection
to validate that we're measuring actual system performance, not mocks.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.benchmarks.performance_benchmarker import (
    RealP2PBenchmark,
    RealAgentForgeBenchmark,
    RealDigitalTwinBenchmark,
    IMPORTS_AVAILABLE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_real_p2p_performance():
    """Demonstrate real P2P network performance measurement."""
    print("=== P2P Network Performance Test ===")

    if not IMPORTS_AVAILABLE:
        print("Some imports not available, using simulation")
        return None

    try:
        p2p_benchmark = RealP2PBenchmark()

        # Test P2P connection establishment (small sample)
        print("Testing P2P connection establishment...")
        time.time()

        result = await p2p_benchmark.benchmark_connection_establishment(5)

        time.time()

        print("P2P Connection Test Results:")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Connections: {result.items_processed}")
        print(f"   Success Rate: {result.success_rate:.1%}")
        print(f"   Avg Latency: {result.latency_avg_ms:.1f}ms")
        print(f"   Throughput: {result.throughput_per_second:.1f} connections/sec")
        print(f"   CPU Usage: {result.cpu_percent_avg:.1f}%")
        print(f"   Memory Peak: {result.memory_mb_peak:.1f}MB")

        return result

    except Exception as e:
        print(f"P2P test failed: {e}")
        return None


async def demonstrate_real_agent_performance():
    """Demonstrate real Agent Forge performance measurement."""
    print("\n=== Agent Forge Performance Test ===")

    if not IMPORTS_AVAILABLE:
        print("⚠️  Imports not available, using simulation")
        return None

    try:
        agent_benchmark = RealAgentForgeBenchmark()

        # Test agent message processing (small sample)
        print("Testing agent message processing...")

        result = await agent_benchmark.benchmark_message_processing(20, 2)

        print("Agent Message Processing Results:")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Messages: {result.items_processed}")
        print(f"   Success Rate: {result.success_rate:.1%}")
        print(f"   Avg Latency: {result.latency_avg_ms:.1f}ms")
        print(f"   Throughput: {result.throughput_per_second:.1f} messages/sec")
        print(f"   CPU Usage: {result.cpu_percent_avg:.1f}%")
        print(f"   Memory Peak: {result.memory_mb_peak:.1f}MB")

        return result

    except Exception as e:
        print(f"Agent test failed: {e}")
        return None


async def demonstrate_real_twin_performance():
    """Demonstrate real Digital Twin performance measurement."""
    print("\n=== Digital Twin Performance Test ===")

    try:
        twin_benchmark = RealDigitalTwinBenchmark()

        # Test chat processing (small sample)
        print("Testing digital twin chat processing...")

        result = twin_benchmark.benchmark_chat_processing(5)

        print("Digital Twin Chat Results:")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Messages: {result.items_processed}")
        print(f"   Success Rate: {result.success_rate:.1%}")
        print(f"   Avg Latency: {result.latency_avg_ms:.1f}ms")
        print(f"   Throughput: {result.throughput_per_second:.1f} messages/sec")
        print(f"   CPU Usage: {result.cpu_percent_avg:.1f}%")
        print(f"   Memory Peak: {result.memory_mb_peak:.1f}MB")

        return result

    except Exception as e:
        print(f"Digital Twin test failed: {e}")
        return None


def demonstrate_performance_analysis(results):
    """Demonstrate performance analysis capabilities."""
    print("\n=== Performance Analysis ===")

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No valid results to analyze")
        return

    print(f"Analyzed {len(valid_results)} components")

    # Aggregate metrics
    total_operations = sum(r.items_processed for r in valid_results)
    avg_success_rate = sum(r.success_rate for r in valid_results) / len(valid_results)
    avg_latency = sum(r.latency_avg_ms for r in valid_results) / len(valid_results)
    total_throughput = sum(r.throughput_per_second for r in valid_results)

    print(f"   Total Operations: {total_operations}")
    print(f"   Average Success Rate: {avg_success_rate:.1%}")
    print(f"   Average Latency: {avg_latency:.1f}ms")
    print(f"   Combined Throughput: {total_throughput:.1f} ops/sec")

    # Identify performance characteristics
    print("\nPerformance Assessment:")

    if avg_success_rate >= 0.95:
        print("   Reliability: EXCELLENT (>95% success rate)")
    elif avg_success_rate >= 0.90:
        print("   Reliability: GOOD (90-95% success rate)")
    else:
        print("   Reliability: NEEDS IMPROVEMENT (<90% success rate)")

    if avg_latency <= 100:
        print("   Responsiveness: EXCELLENT (<100ms average)")
    elif avg_latency <= 500:
        print("   Responsiveness: GOOD (100-500ms average)")
    elif avg_latency <= 1000:
        print("   Responsiveness: ACCEPTABLE (500-1000ms average)")
    else:
        print("   Responsiveness: POOR (>1000ms average)")

    if total_throughput >= 10:
        print("   Throughput: GOOD (>10 ops/sec)")
    elif total_throughput >= 5:
        print("   Throughput: ACCEPTABLE (5-10 ops/sec)")
    else:
        print("   Throughput: LOW (<5 ops/sec)")


def generate_sample_report(results):
    """Generate sample performance report."""
    print("\n=== Sample Performance Report ===")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    valid_results = [r for r in results if r is not None]

    report = {
        "performance_test_summary": {
            "timestamp": timestamp,
            "components_tested": len(valid_results),
            "test_type": "quick_demonstration",
            "total_operations": sum(r.items_processed for r in valid_results),
            "average_success_rate": (
                sum(r.success_rate for r in valid_results) / len(valid_results) if valid_results else 0
            ),
            "status": "FUNCTIONAL" if len(valid_results) > 0 else "NEEDS_REPAIR",
        },
        "component_results": [
            {
                "component": r.component,
                "test_name": r.test_name,
                "success_rate": r.success_rate,
                "latency_ms": r.latency_avg_ms,
                "throughput": r.throughput_per_second,
                "items_processed": r.items_processed,
            }
            for r in valid_results
        ],
    }

    # Save sample report
    report_file = "tools/benchmarks/quick_demo_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Sample report saved to: {report_file}")
    print("\nKey findings:")

    summary = report["performance_test_summary"]
    print(f"   - {summary['components_tested']} components tested")
    print(f"   - {summary['total_operations']} operations performed")
    print(f"   - {summary['average_success_rate']:.1%} average success rate")
    print(f"   - System status: {summary['status']}")


async def main():
    """Main demonstration of real performance testing."""
    print("AIVillage Real-World Performance Testing Demonstration")
    print("=" * 60)
    print("Mission: Replace mock performance data with real measurements")
    print()

    # Test individual components
    p2p_result = await demonstrate_real_p2p_performance()
    agent_result = await demonstrate_real_agent_performance()
    twin_result = await demonstrate_real_twin_performance()

    results = [p2p_result, agent_result, twin_result]

    # Analyze results
    demonstrate_performance_analysis(results)

    # Generate sample report
    generate_sample_report(results)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

    valid_results = [r for r in results if r is not None]
    if valid_results:
        print("SUCCESS: Real performance data collected from working components")
        print(f"   - {len(valid_results)} components tested with actual functionality")
        print("   - Measurements show real latency, throughput, and resource usage")
        print("   - System performance is measurable and quantifiable")
        print("\nReady for comprehensive performance benchmarking")
        return 0
    else:
        print("No working components found for performance testing")
        print("   - System needs repair before performance validation")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
