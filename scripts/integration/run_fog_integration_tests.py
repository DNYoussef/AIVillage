#!/usr/bin/env python3
"""
Fog Computing Integration Test Runner

Execute this script to run comprehensive integration tests for the
fog computing infrastructure.
"""
import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("fog_integration_tests.log")],
)

logger = logging.getLogger(__name__)


async def main():
    """Run fog computing integration tests."""
    print("STARTING FOG COMPUTING INTEGRATION TESTS")
    print("=" * 60)

    try:
        # Import and run integration test suite
        from infrastructure.fog.integration.integration_test_suite import run_fog_integration_tests

        # Run the complete test suite
        test_suite = await run_fog_integration_tests()

        # Display final results
        print("\nINTEGRATION TEST SUMMARY")
        print("=" * 60)

        success_rate = (test_suite.passed_tests / test_suite.total_tests * 100) if test_suite.total_tests > 0 else 0

        # Color coding based on success rate
        if success_rate >= 90:
            status_emoji = "[EXCELLENT]"
            status_text = "EXCELLENT"
        elif success_rate >= 75:
            status_emoji = "[GOOD]"
            status_text = "GOOD"
        elif success_rate >= 50:
            status_emoji = "[NEEDS_IMPROVEMENT]"
            status_text = "NEEDS IMPROVEMENT"
        else:
            status_emoji = "[CRITICAL]"
            status_text = "CRITICAL ISSUES"

        print(f"{status_emoji} Overall Status: {status_text}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Tests Passed: {test_suite.passed_tests}")
        print(f"Tests Failed: {test_suite.failed_tests}")
        print(f"Tests Skipped: {test_suite.skipped_tests}")
        print(f"Total Tests: {test_suite.total_tests}")

        if test_suite.overall_metrics:
            print(f"\nTotal Duration: {test_suite.overall_metrics.get('total_duration', 0):.2f} seconds")
            print(f"Average Test Duration: {test_suite.overall_metrics.get('avg_test_duration', 0):.2f} seconds")

            # Category breakdown
            category_breakdown = test_suite.overall_metrics.get("category_breakdown", {})
            if category_breakdown:
                print("\nCategory Breakdown:")
                for category, stats in category_breakdown.items():
                    category_success = stats["success_rate"]
                    category_emoji = (
                        "[PASS]" if category_success >= 80 else "[WARN]" if category_success >= 50 else "[FAIL]"
                    )
                    print(
                        f"   {category_emoji} {category}: {stats['passed']}/{stats['total']} ({category_success:.1f}%)"
                    )

        # Show failed tests if any
        failed_tests = [r for r in test_suite.test_results if r.status.value == "failed"]
        if failed_tests:
            print(f"\nFAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test.test_name}")
                if test.error_message:
                    print(f"     Error: {test.error_message}")

        # Show critical recommendations
        print("\nRECOMMENDATIONS:")
        if success_rate >= 90:
            print("   * System is ready for production deployment!")
            print("   * Consider running chaos experiments in staging")
            print("   * Monitor SLOs in production environment")
        elif success_rate >= 75:
            print("   * Address failed tests before production deployment")
            print("   * Run additional performance testing under load")
            print("   * Review component interactions for edge cases")
        else:
            print("   * CRITICAL: System not ready for production")
            print("   * Address all failed tests immediately")
            print("   * Review architecture and component implementations")
            print("   * Focus on security and resilience improvements")

        print("\n" + "=" * 60)
        print("FOG COMPUTING INTEGRATION TESTS COMPLETE")

        # Exit with appropriate code
        if success_rate >= 75:
            print("[PASS] Tests passed - system ready for next phase")
            return 0
        else:
            print("[FAIL] Tests failed - address issues before proceeding")
            return 1

    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        print(f"\nCRITICAL ERROR: {e}")
        print("[ERROR] Integration tests could not complete")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
