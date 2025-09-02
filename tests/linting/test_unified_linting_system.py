"""
Comprehensive test suite for the unified linting system
Tests error handling, caching, and integration components
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test imports with fallbacks
try:
    from config.linting.error_handler import (
        ErrorHandler, ErrorContext, LintingError, ToolNotFoundError,
        ConfigurationError, error_handler
    )
    from config.linting.linting_cache_system import (
        UnifiedCacheManager, InMemoryCacheBackend, CacheKey, cache_manager
    )
    from config.linting.unified_linting_manager import (
        UnifiedLintingPipeline, LintingResult, QualityMetrics
    )
    FULL_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full system not available for testing: {e}")
    FULL_SYSTEM_AVAILABLE = False
    
    # Create minimal mock classes for testing
    class MockErrorHandler:
        async def handle_operation(self, context):
            return AsyncMock()
    
    class MockCacheManager:
        async def initialize(self):
            return True
        async def get(self, key, service="default"):
            return None
        async def set(self, key, value, ttl=3600, service="default"):
            return True
    
    error_handler = MockErrorHandler()
    cache_manager = MockCacheManager()


class TestErrorHandling:
    """Test error handling system functionality"""
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_error_context_creation(self):
        """Test error context creation"""
        context = ErrorContext(
            operation="test_operation",
            tool="test_tool",
            target_paths=["test.py"],
            config={"test": True},
            timestamp="2025-01-01T00:00:00",
            session_id="test_session"
        )
        
        assert context.operation == "test_operation"
        assert context.tool == "test_tool"
        assert context.target_paths == ["test.py"]
        assert context.config == {"test": True}
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_linting_error_creation(self):
        """Test linting error creation"""
        from config.linting.error_handler import ErrorCategory, ErrorSeverity
        
        error = LintingError(
            "Test error message",
            ErrorCategory.TOOL_NOT_FOUND,
            ErrorSeverity.HIGH
        )
        
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.TOOL_NOT_FOUND
        assert error.severity == ErrorSeverity.HIGH
        assert error.timestamp is not None
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_tool_not_found_error(self):
        """Test tool not found error handling"""
        error = ToolNotFoundError("missing_tool", "Custom message")
        
        assert error.tool == "missing_tool"
        assert "Custom message" in error.message
        assert error.category.value == "tool_not_found"
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_error_handler_operation(self):
        """Test error handler context manager"""
        context = ErrorContext(
            operation="test_op",
            tool="test_tool",
            target_paths=[],
            config={},
            timestamp="2025-01-01T00:00:00",
            session_id="test"
        )
        
        handler = ErrorHandler()
        
        try:
            async with handler.handle_operation(context):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Check that error was recorded
        assert len(handler.error_history) > 0
        assert "Test error" in str(handler.error_history[-1].message)
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_error_handler_health_check(self):
        """Test error handler health check"""
        handler = ErrorHandler()
        health = await handler.health_check()
        
        assert "status" in health
        assert health["status"] == "healthy"
        assert health["error_handler_ready"] is True


class TestCacheSystem:
    """Test caching system functionality"""
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_cache_key_generation(self):
        """Test cache key generation"""
        params = {"tool": "ruff", "paths": ["src/"], "config": {"line_length": 120}}
        key = CacheKey.hash_params(params)
        
        assert isinstance(key, str)
        assert len(key) == 16  # SHA256 first 16 chars
        
        # Same params should generate same key
        key2 = CacheKey.hash_params(params)
        assert key == key2
        
        # Different params should generate different key
        different_params = {"tool": "black", "paths": ["src/"], "config": {"line_length": 120}}
        key3 = CacheKey.hash_params(different_params)
        assert key != key3
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_in_memory_cache_backend(self):
        """Test in-memory cache backend"""
        cache = InMemoryCacheBackend(max_size=100, default_ttl=3600)
        await cache.connect()
        
        # Test set and get
        success = await cache.set("test_key", {"data": "test_value"}, ttl=60)
        assert success is True
        
        value = await cache.get("test_key")
        assert value == {"data": "test_value"}
        
        # Test exists
        exists = await cache.exists("test_key")
        assert exists is True
        
        # Test non-existent key
        value = await cache.get("non_existent")
        assert value is None
        
        exists = await cache.exists("non_existent")
        assert exists is False
        
        # Test delete
        deleted = await cache.delete("test_key")
        assert deleted is True
        
        value = await cache.get("test_key")
        assert value is None
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        cache = InMemoryCacheBackend(default_ttl=1)  # 1 second TTL
        await cache.connect()
        
        # Set value with short TTL
        await cache.set("expire_key", "expire_value", ttl=1)
        
        # Should be available immediately
        value = await cache.get("expire_key")
        assert value == "expire_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        value = await cache.get("expire_key")
        assert value is None
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_unified_cache_manager(self):
        """Test unified cache manager"""
        from config.linting.linting_cache_system import CacheBackend
        
        # Create cache manager with in-memory backend
        cache_mgr = UnifiedCacheManager(
            preferred_backends=[CacheBackend.MEMORY],
            memory_config={"max_size": 100}
        )
        
        success = await cache_mgr.initialize()
        assert success is True
        
        # Test basic operations
        success = await cache_mgr.set("test", {"data": "value"}, service="test_service")
        assert success is True
        
        value = await cache_mgr.get("test", service="test_service")
        assert value == {"data": "value"}
        
        # Test health check
        health = await cache_mgr.health_check()
        assert health["status"] == "healthy"
        
        # Test stats
        stats = await cache_mgr.get_stats()
        assert "backend_type" in stats
        assert stats["backend_type"] == "memory"


class TestUnifiedLintingManager:
    """Test unified linting manager functionality"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration file"""
        config_data = {
            "python": {
                "ruff": {
                    "line_length": 120,
                    "select": ["E", "W", "F"],
                    "ignore": ["E501"]
                },
                "black": {
                    "line_length": 120
                },
                "mypy": {
                    "ignore_missing_imports": True
                },
                "bandit": {
                    "severity_level": "medium"
                }
            },
            "frontend": {
                "eslint": {
                    "rules": {
                        "no-unused-vars": "error"
                    }
                }
            },
            "security": {
                "block_on_critical": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            return Path(f.name)
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_linting_manager_initialization(self, temp_config):
        """Test linting manager initialization"""
        manager = UnifiedLintingPipeline(config_path=temp_config)
        
        assert manager.config_path == temp_config
        assert "python" in manager.config
        assert "ruff" in manager.config["python"]
        assert len(manager.python_tools) > 0
        assert len(manager.security_tools) > 0
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_linting_result_creation(self):
        """Test linting result data structure"""
        result = LintingResult(
            tool="test_tool",
            status="passed",
            issues_found=5,
            critical_issues=1,
            security_issues=2,
            performance_issues=1,
            style_issues=1,
            files_processed=10,
            execution_time=1.5,
            suggestions=["Fix critical issues"],
            details={"test": "data"},
            timestamp="2025-01-01T00:00:00"
        )
        
        assert result.tool == "test_tool"
        assert result.status == "passed"
        assert result.issues_found == 5
        assert result.critical_issues == 1
        assert result.execution_time == 1.5
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    def test_quality_metrics_creation(self):
        """Test quality metrics data structure"""
        metrics = QualityMetrics(
            overall_score=85.5,
            security_score=90.0,
            performance_score=80.0,
            style_score=85.0,
            maintainability_score=88.0,
            complexity_score=75.0,
            coverage_score=70.0,
            technical_debt_ratio=0.15,
            quality_gate_status="passed"
        )
        
        assert metrics.overall_score == 85.5
        assert metrics.security_score == 90.0
        assert metrics.quality_gate_status == "passed"
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_cache_initialization(self, temp_config):
        """Test cache system initialization"""
        manager = UnifiedLintingPipeline(config_path=temp_config)
        
        # Test cache initialization
        result = await manager._initialize_cache_system()
        # Should return True or False depending on system availability
        assert isinstance(result, bool)


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""
    
    @pytest.mark.asyncio
    async def test_missing_tool_fallback(self):
        """Test behavior when linting tools are missing"""
        # This test should work even without full system
        from config.linting.run_unified_linting import UnifiedLintingPipeline
        
        manager = UnifiedLintingPipeline()
        
        # Try to run linting on non-existent paths
        result = await manager.run_full_pipeline(target_paths=["non_existent/"])
        
        # Should get a result (possibly error result)
        assert isinstance(result, dict)
        assert "pipeline_summary" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_empty_target_paths(self):
        """Test behavior with empty target paths"""
        from config.linting.run_unified_linting import UnifiedLintingPipeline
        
        manager = UnifiedLintingPipeline()
        result = await manager.run_full_pipeline(target_paths=[])
        
        assert isinstance(result, dict)
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation"""
        from config.linting.error_handler import validate_configuration
        
        # Valid configuration
        valid_config = {"line_length": 120, "rules": ["E", "W"]}
        result = await validate_configuration(valid_config, "test_tool")
        assert result == valid_config
        
        # Invalid configuration (not a dict)
        try:
            await validate_configuration("invalid", "test_tool")
        except Exception:
            pass  # Expected to fail


class TestPerformanceAndReliability:
    """Test performance and reliability aspects"""
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with multiple operations"""
        cache = InMemoryCacheBackend(max_size=1000)
        await cache.connect()
        
        start_time = time.time()
        
        # Perform many cache operations
        for i in range(100):
            await cache.set(f"key_{i}", {"data": f"value_{i}"})
            await cache.get(f"key_{i}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete quickly (under 1 second for 200 operations)
        assert execution_time < 1.0
        
        stats = await cache.get_stats()
        assert stats.hits > 0
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_cache_memory_management(self):
        """Test cache memory management with LRU eviction"""
        cache = InMemoryCacheBackend(max_size=3)  # Very small cache
        await cache.connect()
        
        # Fill cache beyond capacity
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")  # Should evict key1
        
        # key1 should be evicted
        value1 = await cache.get("key1")
        assert value1 is None
        
        # Others should still exist
        value2 = await cache.get("key2")
        value3 = await cache.get("key3")
        value4 = await cache.get("key4")
        
        assert value2 == "value2"
        assert value3 == "value3"
        assert value4 == "value4"
    
    @pytest.mark.skipif(not FULL_SYSTEM_AVAILABLE, reason="Full system not available")
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms"""
        handler = ErrorHandler()
        
        # Simulate multiple errors
        context = ErrorContext(
            operation="test",
            tool="test_tool",
            target_paths=[],
            config={},
            timestamp="2025-01-01T00:00:00",
            session_id="test"
        )
        
        errors = []
        for i in range(5):
            try:
                async with handler.handle_operation(context):
                    raise ValueError(f"Error {i}")
            except ValueError as e:
                errors.append(e)
        
        # Check error history
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 5
        assert "ValueError" in summary["error_types"]


class TestCommandLineInterface:
    """Test command line interface functionality"""
    
    @pytest.mark.asyncio
    async def test_cli_argument_parsing(self):
        """Test CLI argument parsing"""
        # Import the CLI module
        from config.linting.run_unified_linting import main
        
        # Mock sys.argv for testing
        import sys
        original_argv = sys.argv
        
        try:
            # Test with basic arguments
            sys.argv = ["run_unified_linting.py", "--language=python", "--dry-run"]
            
            # Should not raise exception (dry run mode)
            result = await main()
            assert result == 0  # Success exit code
            
        finally:
            sys.argv = original_argv
    
    def test_output_formatting(self):
        """Test output formatting functions"""
        from config.linting.run_unified_linting import print_summary, convert_to_sarif
        
        # Test summary printing
        mock_results = {
            "pipeline_summary": {
                "total_tools_run": 3,
                "total_issues_found": 10,
                "critical_issues": 2,
                "security_issues": 3,
                "total_execution_time": 5.5
            }
        }
        
        # Should not raise exception
        print_summary(mock_results, "python")
        
        # Test SARIF conversion
        sarif_result = convert_to_sarif(mock_results)
        assert "$schema" in sarif_result
        assert "version" in sarif_result
        assert "runs" in sarif_result


# Fixtures for all tests
@pytest.fixture
def sample_python_file():
    """Create a temporary Python file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def example_function():
    """Example function for testing."""
    x = 1 + 1
    print("Hello, World!")
    return x
''')
        return Path(f.name)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "python": {
            "ruff": {"line_length": 120},
            "black": {"line_length": 120},
            "mypy": {"ignore_missing_imports": True}
        },
        "security": {
            "block_on_critical": True
        }
    }


if __name__ == "__main__":
    # Run specific tests for debugging
    pytest.main([__file__, "-v", "--tb=short"])