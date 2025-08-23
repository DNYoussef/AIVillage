"""
Consolidated HyperRAG Test Suite

Complete test coverage for the unified HyperRAG system:
- Unit tests for individual components
- Integration tests for multi-system coordination
- Production tests for deployment validation

Test Structure:
- unit/: Component-specific unit tests
- integration/: End-to-end integration tests
- production/: Production health and validation tests
- archive/: Legacy test files (pre-consolidation)
"""

# Test utilities and helpers
from .conftest import assert_health_check_valid, assert_performance_acceptable, assert_valid_answer

__all__ = ["assert_valid_answer", "assert_performance_acceptable", "assert_health_check_valid"]
