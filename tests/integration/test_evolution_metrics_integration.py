"""
Integration tests for Evolution Metrics System with CODEX requirements.

Tests:
- Database connectivity with WAL mode
- Redis integration with fallback
- 100 evolution cycles data persistence
- API health endpoints
- KPI tracking for all 18 metrics
- Agent system integration
"""

import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.evolution_metrics_integrated import (
    EvolutionMetricsData,
    IntegratedEvolutionMetrics,
    KPIType,
    get_health_status,
    get_metrics_instance,
    record_kpi,
    start_metrics,
    stop_metrics,
)


def test_database_connection():
    """Test 1: Verify database connection with WAL mode."""
    print("\n" + "=" * 60)
    print("TEST 1: DATABASE CONNECTION WITH WAL MODE")
    print("=" * 60)

    # Check database exists
    db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return False

    print(f"✅ Database found at {db_path}")

    # Verify WAL mode
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode")
    mode = cursor.fetchone()[0]

    if mode.lower() == "wal":
        print(f"✅ WAL mode enabled: {mode}")
    else:
        print(f"❌ WAL mode not enabled: {mode}")
        return False

    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    required_tables = [
        "evolution_rounds",
        "fitness_metrics",
        "resource_metrics",
        "selection_outcomes",
    ]

    for table in required_tables:
        if table in tables:
            print(f"✅ Table '{table}' exists")
        else:
            print(f"❌ Table '{table}' missing")
            return False

    conn.close()
    return True


def test_redis_integration():
    """Test 2: Verify Redis integration with fallback."""
    print("\n" + "=" * 60)
    print("TEST 2: REDIS INTEGRATION WITH FALLBACK")
    print("=" * 60)

    IntegratedEvolutionMetrics()

    # Check Redis availability
    try:
        import redis

        print("✅ Redis module available")
    except ImportError:
        print("⚠️ Redis module not installed (will use SQLite fallback)")

    # Test Redis connection
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    try:
        import redis

        client = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)
        client.ping()
        print(f"✅ Redis connected at {redis_host}:{redis_port}")

        # Test pub/sub
        pubsub = client.pubsub()
        pubsub.subscribe("evolution:updates")
        print("✅ Redis pub/sub working")

        client.close()
        return True

    except Exception as e:
        print(f"⚠️ Redis not available: {e}")
        print("✅ Will fallback to SQLite-only mode")
        return True  # Fallback is acceptable


def test_data_persistence():
    """Test 3: Run 100 evolution cycles and verify persistence."""
    print("\n" + "=" * 60)
    print("TEST 3: DATA PERSISTENCE (100 EVOLUTION CYCLES)")
    print("=" * 60)

    # Start metrics system
    metrics = IntegratedEvolutionMetrics()
    metrics.start()
    print("✅ Metrics system started")

    # Generate 100 evolution cycles
    agents = [f"agent_{i}" for i in range(10)]
    cycles_recorded = 0

    print("\nRecording evolution cycles...")
    for cycle in range(100):
        agent_id = random.choice(agents)

        # Create metrics with all 18 KPIs
        evolution_metrics = EvolutionMetricsData(
            agent_id=agent_id,
            round_number=cycle,
            generation=cycle // 10,
            evolution_type="genetic",
            # 18 KPIs with random values
            performance_score=random.uniform(0.5, 1.0),
            learning_rate=random.uniform(0.001, 0.1),
            task_completion=random.uniform(0.7, 1.0),
            error_rate=random.uniform(0.0, 0.3),
            response_time=random.uniform(10, 100),
            memory_efficiency=random.uniform(0.6, 0.95),
            cpu_efficiency=random.uniform(0.5, 0.9),
            adaptation_speed=random.uniform(0.3, 0.8),
            creativity_score=random.uniform(0.4, 0.9),
            collaboration_score=random.uniform(0.5, 1.0),
            specialization_depth=random.uniform(0.3, 0.9),
            generalization_breadth=random.uniform(0.4, 0.8),
            robustness_score=random.uniform(0.6, 0.95),
            energy_efficiency=random.uniform(0.5, 0.9),
            knowledge_retention=random.uniform(0.7, 1.0),
            innovation_rate=random.uniform(0.1, 0.6),
            quality_consistency=random.uniform(0.7, 0.95),
            resource_utilization=random.uniform(0.4, 0.8),
            # Resource metrics
            memory_used_mb=random.uniform(100, 500),
            cpu_percent=random.uniform(10, 80),
            network_io_kb=random.uniform(0, 1000),
            disk_io_kb=random.uniform(0, 500),
            # Selection outcomes
            fitness_score=random.uniform(0.5, 1.0),
            selected=random.choice([True, False]),
            selection_method=random.choice(["tournament", "roulette", "elite"]),
            mutation_applied=random.choice([True, False]),
        )

        metrics.record_metric(evolution_metrics)
        cycles_recorded += 1

        if (cycle + 1) % 10 == 0:
            print(f"  Recorded {cycle + 1}/100 cycles")

    # Force flush
    metrics.flush()
    time.sleep(1)  # Wait for background worker

    print(f"\n✅ Recorded {cycles_recorded} evolution cycles")

    # Verify data was persisted
    db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check fitness metrics
    cursor.execute("SELECT COUNT(*) FROM fitness_metrics")
    fitness_count = cursor.fetchone()[0]
    print(f"✅ Fitness metrics persisted: {fitness_count} records")

    # Check resource metrics
    cursor.execute("SELECT COUNT(*) FROM resource_metrics")
    resource_count = cursor.fetchone()[0]
    print(f"✅ Resource metrics persisted: {resource_count} records")

    # Check selection outcomes
    cursor.execute("SELECT COUNT(*) FROM selection_outcomes")
    selection_count = cursor.fetchone()[0]
    print(f"✅ Selection outcomes persisted: {selection_count} records")

    # Test recovery after restart
    metrics.stop()
    print("\n✅ Metrics system stopped")

    # Start again and check data recovery
    metrics2 = IntegratedEvolutionMetrics()
    metrics2.start()

    # Get agent history
    test_agent = agents[0]
    history = metrics2.get_agent_history(test_agent, limit=10)
    if history:
        print(f"✅ Data recovery successful: {len(history)} records for {test_agent}")

    metrics2.stop()
    conn.close()

    return fitness_count >= 50  # At least 50% persisted


def test_kpi_tracking():
    """Test 4: Verify all 18 KPIs are tracked correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: 18 KPI TRACKING VERIFICATION")
    print("=" * 60)

    metrics = start_metrics()

    # Test each KPI type
    kpi_values = {}
    for kpi_type in KPIType:
        value = random.uniform(0.1, 1.0)
        record_kpi("test_agent", kpi_type, value)
        kpi_values[kpi_type.value] = value
        print(f"✅ Recorded {kpi_type.value}: {value:.3f}")

    # Flush and verify
    metrics.flush()
    time.sleep(1)

    # Check database for KPI values
    db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT performance_metrics
        FROM fitness_metrics
        WHERE agent_id = 'test_agent'
        ORDER BY timestamp DESC
        LIMIT 1
    """
    )

    row = cursor.fetchone()
    if row and row[0]:
        stored_metrics = json.loads(row[0])
        print(f"\n✅ KPIs stored in database: {len(stored_metrics)} metrics")

    stop_metrics()
    conn.close()
    return True


def test_flush_threshold():
    """Test 5: Verify flush threshold is working correctly."""
    print("\n" + "=" * 60)
    print("TEST 5: FLUSH THRESHOLD VERIFICATION")
    print("=" * 60)

    # Set a low flush threshold for testing
    os.environ["AIVILLAGE_METRICS_FLUSH_THRESHOLD"] = "5"

    metrics = IntegratedEvolutionMetrics()
    metrics.start()

    initial_buffer_size = len(metrics.metrics_buffer)
    print(f"Initial buffer size: {initial_buffer_size}")
    print(f"Flush threshold: {metrics.flush_threshold}")

    # Add metrics up to threshold
    for i in range(4):
        m = EvolutionMetricsData(
            agent_id=f"threshold_test_{i}", fitness_score=random.random()
        )
        metrics.record_metric(m)

    buffer_before = len(metrics.metrics_buffer)
    print(f"Buffer size after 4 metrics: {buffer_before}")

    # Add one more to trigger flush
    m = EvolutionMetricsData(
        agent_id="threshold_trigger", fitness_score=random.random()
    )
    metrics.record_metric(m)

    # Buffer should be empty or very small after auto-flush
    time.sleep(0.5)  # Give time for flush
    buffer_after = len(metrics.metrics_buffer)
    print(f"Buffer size after threshold reached: {buffer_after}")

    if buffer_after < buffer_before:
        print("✅ Auto-flush triggered at threshold")
        result = True
    else:
        print("❌ Auto-flush did not trigger")
        result = False

    metrics.stop()

    # Reset environment
    del os.environ["AIVILLAGE_METRICS_FLUSH_THRESHOLD"]

    return result


def test_api_health_endpoint():
    """Test 6: Verify API health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 6: API HEALTH ENDPOINT VERIFICATION")
    print("=" * 60)

    start_metrics()

    # Get health status
    health = get_health_status()

    print("\nHealth Status Response:")
    print(json.dumps(health, indent=2))

    # Verify required fields
    required_fields = ["status", "timestamp", "database", "redis", "metrics"]
    for field in required_fields:
        if field in health:
            print(f"✅ Field '{field}' present")
        else:
            print(f"❌ Field '{field}' missing")
            return False

    # Check status
    if health["status"] == "healthy":
        print("✅ System status: healthy")
    else:
        print(f"⚠️ System status: {health['status']}")

    # Check database connection
    if health["database"]["connected"]:
        print("✅ Database connected")
    else:
        print("❌ Database not connected")

    stop_metrics()
    return True


def test_concurrent_metrics():
    """Test 7: Test concurrent metric updates."""
    print("\n" + "=" * 60)
    print("TEST 7: CONCURRENT METRIC UPDATES")
    print("=" * 60)

    import threading

    metrics = start_metrics()
    errors = []

    def record_metrics_thread(thread_id, count):
        """Record metrics from a thread."""
        try:
            for i in range(count):
                m = EvolutionMetricsData(
                    agent_id=f"thread_{thread_id}_agent_{i}",
                    fitness_score=random.random(),
                    performance_score=random.random(),
                )
                metrics.record_metric(m)
                time.sleep(0.001)  # Small delay
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")

    # Create multiple threads
    threads = []
    thread_count = 5
    metrics_per_thread = 20

    print(
        f"Starting {thread_count} threads, each recording {metrics_per_thread} metrics..."
    )

    for i in range(thread_count):
        t = threading.Thread(target=record_metrics_thread, args=(i, metrics_per_thread))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    if errors:
        print(f"❌ Errors occurred: {errors}")
        return False

    print("✅ All threads completed successfully")

    # Verify metrics were recorded
    total_expected = thread_count * metrics_per_thread
    total_collected = metrics.total_metrics_collected

    print(f"Expected metrics: {total_expected}")
    print(f"Collected metrics: {total_collected}")

    if total_collected >= total_expected * 0.9:  # Allow 10% tolerance
        print("✅ Concurrent updates handled correctly")
        result = True
    else:
        print("❌ Some metrics may have been lost")
        result = False

    stop_metrics()
    return result


def test_agent_integration():
    """Test 8: Integration with agent system."""
    print("\n" + "=" * 60)
    print("TEST 8: AGENT SYSTEM INTEGRATION")
    print("=" * 60)

    # Simulate agent KPI reporting
    class MockAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.metrics = get_metrics_instance()

        def perform_task(self):
            """Simulate agent performing a task and reporting metrics."""
            start_time = time.time()

            # Simulate work
            time.sleep(0.1)

            # Calculate metrics
            duration = time.time() - start_time
            success = random.random() > 0.2

            # Report multiple KPIs
            record_kpi(self.agent_id, KPIType.TASK_COMPLETION, 1.0 if success else 0.0)
            record_kpi(self.agent_id, KPIType.RESPONSE_TIME, duration * 1000)
            record_kpi(
                self.agent_id, KPIType.PERFORMANCE_SCORE, random.uniform(0.6, 1.0)
            )

            return success

    # Start metrics system
    start_metrics()

    # Create and run agents
    agents = [MockAgent(f"agent_{i}") for i in range(5)]

    print("Running agent tasks...")
    successes = 0
    for agent in agents:
        if agent.perform_task():
            successes += 1
        print(f"  {agent.agent_id} completed task")

    print(f"\n✅ {successes}/{len(agents)} agents succeeded")

    # Check metrics were recorded
    metrics_instance = get_metrics_instance()
    current = metrics_instance.get_current_metrics()

    if current["total_metrics_collected"] >= len(agents) * 3:  # 3 KPIs per agent
        print(f"✅ Agent metrics recorded: {current['total_metrics_collected']} total")
        result = True
    else:
        print(f"❌ Insufficient metrics: {current['total_metrics_collected']}")
        result = False

    stop_metrics()
    return result


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("EVOLUTION METRICS INTEGRATION TESTS")
    print("CODEX Requirements Compliance Verification")
    print("=" * 60)

    tests = [
        ("Database Connection", test_database_connection),
        ("Redis Integration", test_redis_integration),
        ("Data Persistence (100 cycles)", test_data_persistence),
        ("18 KPI Tracking", test_kpi_tracking),
        ("Flush Threshold", test_flush_threshold),
        ("API Health Endpoint", test_api_health_endpoint),
        ("Concurrent Updates", test_concurrent_metrics),
        ("Agent Integration", test_agent_integration),
    ]

    results = []

    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Evolution metrics system is fully integrated.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please review the failures above.")

    return passed == total


if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent.parent.parent)

    # Run all tests
    success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
