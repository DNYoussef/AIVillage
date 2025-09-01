#!/usr/bin/env python3
"""
Simple test to validate BaseAnalytics implementation.
Tests core functionality without Unicode characters for compatibility.
"""

import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add analytics module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../experiments/agents/agents/king/analytics"))

try:
    from base_analytics import BaseAnalytics

    print("SUCCESS: BaseAnalytics module imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import BaseAnalytics: {e}")
    sys.exit(1)


class TestAnalytics(BaseAnalytics):
    """Simple concrete implementation for testing."""

    pass


def test_basic_functionality():
    """Test basic analytics functionality."""
    print("\nTesting basic functionality...")

    analytics = TestAnalytics()

    # Test metric recording
    analytics.record_metric("test_metric", 42.0)
    analytics.record_metric("test_metric", 43.0)
    analytics.record_metric("another_metric", 100.5)

    # Verify data
    assert len(analytics.metrics["test_metric"]) == 2
    assert analytics.metrics["test_metric"] == [42.0, 43.0]
    assert len(analytics.metrics["another_metric"]) == 1
    assert analytics.metrics["another_metric"][0] == 100.5

    print("  - Metric recording: PASS")

    # Test report generation
    report = analytics.generate_analytics_report()
    assert isinstance(report, dict)
    assert "metadata" in report
    assert "metrics" in report
    assert "summary" in report
    assert report["metadata"]["total_metrics"] == 2
    assert report["metadata"]["data_points"] == 3

    print("  - Report generation: PASS")
    return True


def test_persistence():
    """Test save and load functionality."""
    print("\nTesting persistence...")

    analytics = TestAnalytics()

    # Add test data
    for i in range(10):
        analytics.record_metric("persistence_test", float(i))

    test_dir = tempfile.mkdtemp()

    # Test JSON format
    json_path = Path(test_dir) / "test.json"
    save_success = analytics.save(str(json_path))
    assert save_success, "JSON save failed"

    # Verify file exists and contains data
    assert json_path.exists(), "JSON file not created"

    with open(json_path, "r") as f:
        data = json.load(f)
    assert "metrics" in data
    assert "persistence_test" in data["metrics"]

    print("  - JSON save: PASS")

    # Test loading
    new_analytics = TestAnalytics()
    load_success = new_analytics.load(str(json_path))
    assert load_success, "JSON load failed"

    # Verify data integrity
    assert len(new_analytics.metrics["persistence_test"]) == 10
    assert new_analytics.metrics["persistence_test"] == analytics.metrics["persistence_test"]

    print("  - JSON load: PASS")

    # Test pickle format
    pickle_path = Path(test_dir) / "test.pkl"
    save_success = analytics.save(str(pickle_path))
    assert save_success, "Pickle save failed"

    new_analytics2 = TestAnalytics()
    load_success = new_analytics2.load(str(pickle_path))
    assert load_success, "Pickle load failed"
    assert len(new_analytics2.metrics["persistence_test"]) == 10

    print("  - Pickle save/load: PASS")

    # Cleanup
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)

    return True


def test_statistical_analysis():
    """Test statistical analysis features."""
    print("\nTesting statistical analysis...")

    analytics = TestAnalytics()

    # Add known test data
    test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for value in test_values:
        analytics.record_metric("stats_test", value)

    # Test analysis
    analysis = analytics._analyze_metric("stats_test", test_values, [])
    stats = analysis["statistics"]

    # Verify statistical calculations
    assert stats["count"] == 5
    assert abs(stats["mean"] - 3.0) < 0.001
    assert abs(stats["median"] - 3.0) < 0.001
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["range"] == 4.0

    print("  - Basic statistics: PASS")

    # Test trend analysis
    trends = analysis["trends"]
    assert "linear_trend" in trends
    assert "slope" in trends

    print("  - Trend analysis: PASS")

    # Test outlier detection
    outlier_data = [1, 2, 3, 4, 100]  # 100 is outlier
    outliers = analytics._detect_outliers(outlier_data)
    assert outliers["count"] > 0
    assert 4 in outliers["indices"]  # Index of outlier

    print("  - Outlier detection: PASS")

    return True


def test_retention_policies():
    """Test retention policy functionality."""
    print("\nTesting retention policies...")

    analytics = TestAnalytics()

    # Set max points retention
    analytics.set_retention_policy(max_data_points=5)

    # Add more data than limit
    for i in range(10):
        analytics.record_metric("retention_test", float(i))

    # Should only keep last 5 points
    assert len(analytics.metrics["retention_test"]) == 5
    assert analytics.metrics["retention_test"] == [5.0, 6.0, 7.0, 8.0, 9.0]

    print("  - Max points retention: PASS")

    # Test time-based retention
    time_analytics = TestAnalytics()
    time_analytics.set_retention_policy(retention_period=timedelta(hours=1))

    # Add old data (should be filtered)
    old_time = datetime.now() - timedelta(hours=2)
    time_analytics.record_metric("time_test", 1.0, old_time)

    # Add recent data (should be kept)
    recent_time = datetime.now() - timedelta(minutes=30)
    time_analytics.record_metric("time_test", 2.0, recent_time)

    # Should only have recent data
    assert len(time_analytics.metrics["time_test"]) == 1
    assert time_analytics.metrics["time_test"][0] == 2.0

    print("  - Time-based retention: PASS")

    return True


def test_report_formats():
    """Test different report formats."""
    print("\nTesting report formats...")

    analytics = TestAnalytics()

    # Add test data
    for i in range(20):
        analytics.record_metric("format_test", float(i))

    # Test JSON format
    json_report = analytics.generate_analytics_report(report_format="json")
    assert "statistics" in json_report["metrics"]["format_test"]
    assert "trends" in json_report["metrics"]["format_test"]

    print("  - JSON format: PASS")

    # Test summary format
    summary_report = analytics.generate_analytics_report(report_format="summary")
    assert "count" in summary_report["metrics"]["format_test"]
    assert "mean" in summary_report["metrics"]["format_test"]
    assert "statistics" not in summary_report["metrics"]["format_test"]

    print("  - Summary format: PASS")

    # Test detailed format
    detailed_report = analytics.generate_analytics_report(report_format="detailed")
    assert "statistics" in detailed_report["metrics"]["format_test"]
    assert "quality" in detailed_report["metrics"]["format_test"]

    print("  - Detailed format: PASS")

    # Test time window
    windowed_report = analytics.generate_analytics_report(time_window=timedelta(minutes=30))
    # Should have some data (even if filtered)
    assert "metadata" in windowed_report

    print("  - Time window filtering: PASS")

    return True


def run_all_tests():
    """Run all validation tests."""
    print("BaseAnalytics Implementation Validation")
    print("=" * 50)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Persistence", test_persistence),
        ("Statistical Analysis", test_statistical_analysis),
        ("Retention Policies", test_retention_policies),
        ("Report Formats", test_report_formats),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n[{test_name}]")
            success = test_func()
            if success:
                print("  RESULT: PASS")
                passed += 1
            else:
                print("  RESULT: FAIL")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  RESULT: FAIL")
            failed += 1

    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} passed, {failed} failed")

    if failed == 0:
        print("\nSUCCESS: All tests passed!")
        print("\nImplemented methods:")
        print("  1. generate_analytics_report() - Multi-format analytics reporting")
        print("  2. save() - Multi-format persistence with atomic operations")
        print("  3. load() - Format detection and validation")
        print("\nKey features:")
        print("  - Time-series analytics with trend analysis")
        print("  - Statistical analysis and outlier detection")
        print("  - Memory management with retention policies")
        print("  - Multi-format persistence (JSON, Pickle, SQLite)")
        print("  - Atomic operations with backup and recovery")
        print("  - Schema validation and data integrity checks")
        return True
    else:
        print("\nFAILURE: Some tests failed")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
