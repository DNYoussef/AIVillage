"""Agent Core Test Suite

Comprehensive behavioral test suite for the AIVillage agent system.
Validates agent contracts, integration patterns, and system properties
while maintaining loose coupling through connascence principles.

Test Organization:
- behavioral/: Contract tests for observable behaviors
- integration/: Component interaction tests
- properties/: Property-based invariant tests
- performance/: Performance validation tests
- isolation/: Test isolation validation
- validation/: Connascence compliance validation
- fixtures/: Reusable test data builders and fixtures

Key Principles:
1. Test behaviors, not implementations
2. Minimize connascence between tests and code
3. Use builders for consistent test data
4. Maintain test isolation
5. Focus on observable contracts
"""

__version__ = "1.0.0"
__author__ = "AIVillage Test Team"

# Test suite metadata
TEST_SUITE_INFO = {
    "name": "Agent Core Test Suite",
    "version": __version__,
    "description": "Behavioral testing suite for AIVillage agents",
    "principles": [
        "Behavior-driven testing",
        "Loose coupling via connascence management",
        "Property-based validation",
        "Performance isolation",
        "Contract verification",
    ],
    "test_categories": [
        "behavioral_contracts",
        "component_integration",
        "system_properties",
        "performance_validation",
        "isolation_verification",
        "connascence_compliance",
    ],
}
