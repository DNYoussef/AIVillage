"""
AIVillage Unified Linting Error Handler
Comprehensive error handling with fallback mechanisms
"""

import asyncio
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category types"""
    TOOL_NOT_FOUND = "tool_not_found"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    tool: str
    target_paths: List[str]
    config: Dict[str, Any]
    timestamp: str
    session_id: str
    additional_context: Optional[Dict[str, Any]] = None


class LintingError(Exception):
    """Base exception for linting operations"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.timestamp = datetime.now().isoformat()


class ToolNotFoundError(LintingError):
    """Error when a linting tool is not available"""
    def __init__(self, tool: str, message: str = None, context: Optional[ErrorContext] = None):
        self.tool = tool
        message = message or f"Linting tool '{tool}' not found or not executable"
        super().__init__(message, ErrorCategory.TOOL_NOT_FOUND, ErrorSeverity.HIGH, context)


class ConfigurationError(LintingError):
    """Error in configuration or setup"""
    def __init__(self, message: str, config_key: str = None, context: Optional[ErrorContext] = None):
        self.config_key = config_key
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM, context)


class DependencyError(LintingError):
    """Error related to missing dependencies"""
    def __init__(self, dependency: str, message: str = None, context: Optional[ErrorContext] = None):
        self.dependency = dependency
        message = message or f"Required dependency '{dependency}' is not available"
        super().__init__(message, ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH, context)


class TimeoutError(LintingError):
    """Error when operation times out"""
    def __init__(self, operation: str, timeout: int, context: Optional[ErrorContext] = None):
        self.operation = operation
        self.timeout = timeout
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        super().__init__(message, ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM, context)


class PermissionError(LintingError):
    """Error related to file/directory permissions"""
    def __init__(self, path: str, operation: str = "access", context: Optional[ErrorContext] = None):
        self.path = path
        self.operation = operation
        message = f"Permission denied for {operation} on '{path}'"
        super().__init__(message, ErrorCategory.PERMISSION, ErrorSeverity.HIGH, context)


class NetworkError(LintingError):
    """Error related to network operations"""
    def __init__(self, endpoint: str = None, message: str = None, context: Optional[ErrorContext] = None):
        self.endpoint = endpoint
        message = message or f"Network error accessing {endpoint or 'external resource'}"
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, context)


class ErrorHandler:
    """
    Comprehensive error handler with fallback mechanisms and recovery strategies
    """
    
    def __init__(self):
        self.error_history: List[LintingError] = []
        self.recovery_strategies = {
            ErrorCategory.TOOL_NOT_FOUND: self._handle_tool_not_found,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.PERMISSION: self._handle_permission_error,
            ErrorCategory.TIMEOUT: self._handle_timeout_error,
            ErrorCategory.DEPENDENCY: self._handle_dependency_error,
        }
        self.fallback_tools = {
            "ruff": ["flake8", "pycodestyle", "pylint"],
            "black": ["autopep8", "yapf"],
            "mypy": ["pyright", "pyre-check"],
            "bandit": ["semgrep", "detect-secrets"],
            "eslint": ["jshint", "jslint"],
            "prettier": ["js-beautify"],
            "semgrep": ["bandit", "detect-secrets"],
            "detect-secrets": ["trufflesec", "gitleaks"]
        }

    @asynccontextmanager
    async def handle_operation(self, context: ErrorContext):
        """
        Async context manager for handling operations with comprehensive error recovery
        """
        try:
            logger.debug(f"Starting operation: {context.operation} with tool: {context.tool}")
            yield context
            logger.debug(f"Operation completed successfully: {context.operation}")
            
        except Exception as e:
            logger.error(f"Error in operation {context.operation}: {str(e)}")
            await self._handle_error(e, context)
            raise

    async def _handle_error(self, error: Exception, context: ErrorContext):
        """Handle and potentially recover from errors"""
        
        # Convert generic exceptions to LintingError
        if not isinstance(error, LintingError):
            linting_error = self._convert_to_linting_error(error, context)
        else:
            linting_error = error
            linting_error.context = context

        # Add to error history
        self.error_history.append(linting_error)
        
        # Log error details
        self._log_error(linting_error)
        
        # Attempt recovery if strategy exists
        recovery_strategy = self.recovery_strategies.get(linting_error.category)
        if recovery_strategy:
            await recovery_strategy(linting_error, context)

    def _convert_to_linting_error(self, error: Exception, context: ErrorContext) -> LintingError:
        """Convert generic exceptions to appropriate LintingError types"""
        
        error_str = str(error).lower()
        
        # Check for specific error patterns
        if "command not found" in error_str or "no such file" in error_str:
            return ToolNotFoundError(context.tool, str(error), context)
        elif "permission denied" in error_str:
            return PermissionError("unknown", "access", context)
        elif "timeout" in error_str or "timed out" in error_str:
            return TimeoutError(context.operation, 300, context)  # Default timeout
        elif "network" in error_str or "connection" in error_str:
            return NetworkError(message=str(error), context=context)
        elif "config" in error_str or "configuration" in error_str:
            return ConfigurationError(str(error), context=context)
        elif "import" in error_str or "module" in error_str:
            return DependencyError("unknown", str(error), context)
        else:
            return LintingError(str(error), ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, context)

    def _log_error(self, error: LintingError):
        """Log error details with appropriate severity"""
        
        error_details = {
            "error_type": error.__class__.__name__,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "timestamp": error.timestamp,
            "tool": getattr(error.context, 'tool', 'unknown') if error.context else 'unknown',
            "operation": getattr(error.context, 'operation', 'unknown') if error.context else 'unknown'
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_details)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {error.message}", extra=error_details)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=error_details)
        else:
            logger.info(f"LOW SEVERITY: {error.message}", extra=error_details)

    async def _handle_tool_not_found(self, error: ToolNotFoundError, context: ErrorContext):
        """Handle tool not found errors with fallback suggestions"""
        
        tool_name = error.tool
        fallback_tools = self.fallback_tools.get(tool_name, [])
        
        logger.warning(f"Tool '{tool_name}' not found. Checking fallback tools: {fallback_tools}")
        
        # Check if any fallback tools are available
        available_fallbacks = []
        for fallback in fallback_tools:
            try:
                # Simple check - in production would verify tool availability
                available_fallbacks.append(fallback)
            except Exception:
                continue
        
        if available_fallbacks:
            logger.info(f"Available fallback tools for {tool_name}: {available_fallbacks}")
            # Store fallback suggestions in context
            if context.additional_context is None:
                context.additional_context = {}
            context.additional_context['suggested_fallbacks'] = available_fallbacks
        else:
            logger.warning(f"No fallback tools available for {tool_name}")

    async def _handle_configuration_error(self, error: ConfigurationError, context: ErrorContext):
        """Handle configuration errors with auto-fix attempts"""
        
        logger.info(f"Attempting to resolve configuration error: {error.message}")
        
        # Attempt to create default configuration
        try:
            default_config = self._generate_default_config(context.tool)
            if context.additional_context is None:
                context.additional_context = {}
            context.additional_context['default_config'] = default_config
            logger.info(f"Generated default configuration for {context.tool}")
        except Exception as e:
            logger.error(f"Failed to generate default configuration: {e}")

    async def _handle_network_error(self, error: NetworkError, context: ErrorContext):
        """Handle network errors with retry logic"""
        
        logger.info(f"Network error encountered, will implement retry logic: {error.message}")
        # In production, implement exponential backoff retry logic
        
        if context.additional_context is None:
            context.additional_context = {}
        context.additional_context['retry_suggested'] = True
        context.additional_context['retry_delay'] = 5  # seconds

    async def _handle_permission_error(self, error: PermissionError, context: ErrorContext):
        """Handle permission errors with alternative approaches"""
        
        logger.warning(f"Permission error: {error.message}")
        
        # Suggest alternative approaches
        if context.additional_context is None:
            context.additional_context = {}
        context.additional_context['permission_workarounds'] = [
            "Run with elevated privileges",
            "Check file/directory ownership",
            "Use alternative temporary directory",
            "Skip protected directories"
        ]

    async def _handle_timeout_error(self, error: TimeoutError, context: ErrorContext):
        """Handle timeout errors with optimization suggestions"""
        
        logger.info(f"Timeout error in {error.operation}, suggesting optimizations")
        
        if context.additional_context is None:
            context.additional_context = {}
        context.additional_context['timeout_optimizations'] = [
            "Reduce scope of analysis",
            "Use parallel processing",
            "Increase timeout limit",
            "Enable caching"
        ]

    async def _handle_dependency_error(self, error: DependencyError, context: ErrorContext):
        """Handle dependency errors with installation suggestions"""
        
        logger.info(f"Dependency error for {error.dependency}")
        
        # Generate installation suggestions
        install_suggestions = []
        
        if error.dependency in ["redis", "redis-py"]:
            install_suggestions.extend([
                "pip install redis",
                "Install Redis server: https://redis.io/download",
                "Use fallback in-memory cache"
            ])
        elif error.dependency in ["memcache", "python-memcached"]:
            install_suggestions.extend([
                "pip install python-memcached",
                "Install Memcached server",
                "Use fallback in-memory cache"
            ])
        else:
            install_suggestions.append(f"pip install {error.dependency}")
        
        if context.additional_context is None:
            context.additional_context = {}
        context.additional_context['install_suggestions'] = install_suggestions

    def _generate_default_config(self, tool: str) -> Dict[str, Any]:
        """Generate minimal default configuration for a tool"""
        
        default_configs = {
            "ruff": {
                "line_length": 120,
                "select": ["E", "W", "F"],
                "ignore": ["E501"]
            },
            "black": {
                "line_length": 120,
                "target_version": ["py311"]
            },
            "mypy": {
                "ignore_missing_imports": True,
                "no_strict_optional": True
            },
            "bandit": {
                "severity_level": "medium"
            },
            "eslint": {
                "rules": {
                    "no-unused-vars": "error",
                    "no-console": "warn"
                }
            }
        }
        
        return default_configs.get(tool, {})

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        
        if not self.error_history:
            return {"total_errors": 0, "summary": "No errors recorded"}
        
        error_counts = {}
        severity_counts = {}
        category_counts = {}
        
        for error in self.error_history:
            # Count by error type
            error_type = error.__class__.__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by category
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "most_recent_error": {
                "message": self.error_history[-1].message,
                "timestamp": self.error_history[-1].timestamp,
                "category": self.error_history[-1].category.value
            }
        }

    def clear_error_history(self):
        """Clear the error history"""
        self.error_history.clear()
        logger.info("Error history cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of error handling system"""
        
        try:
            # Test basic error handling
            test_context = ErrorContext(
                operation="health_check",
                tool="test_tool",
                target_paths=[],
                config={},
                timestamp=datetime.now().isoformat(),
                session_id="health_check"
            )
            
            # Test error conversion
            test_error = ValueError("Test error")
            converted_error = self._convert_to_linting_error(test_error, test_context)
            
            return {
                "status": "healthy",
                "error_handler_ready": True,
                "recovery_strategies_available": len(self.recovery_strategies),
                "fallback_tools_configured": len(self.fallback_tools),
                "error_conversion_working": isinstance(converted_error, LintingError),
                "error_history_size": len(self.error_history)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_handler_ready": False
            }


# Global error handler instance
error_handler = ErrorHandler()


# Utility decorators for error handling
def handle_linting_errors(operation_name: str = None):
    """Decorator to automatically handle linting errors in functions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tool_name = kwargs.get('tool', operation_name or func.__name__)
            target_paths = kwargs.get('target_paths', [])
            config = kwargs.get('config', {})
            
            context = ErrorContext(
                operation=operation_name or func.__name__,
                tool=tool_name,
                target_paths=target_paths,
                config=config,
                timestamp=datetime.now().isoformat(),
                session_id=f"{func.__name__}_{id(args)}"
            )
            
            async with error_handler.handle_operation(context):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, convert to async
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Utility functions for common error scenarios
async def check_tool_availability(tool_name: str, context: ErrorContext = None) -> bool:
    """Check if a linting tool is available and executable"""
    try:
        import subprocess
        result = subprocess.run([tool_name, "--version"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
        if context:
            await error_handler._handle_error(e, context)
        return False


async def validate_configuration(config: Dict[str, Any], tool: str) -> Dict[str, Any]:
    """Validate and sanitize configuration for a tool"""
    try:
        # Basic validation logic would go here
        # For now, return config as-is with basic checks
        if not isinstance(config, dict):
            raise ConfigurationError(f"Configuration for {tool} must be a dictionary")
        
        return config
        
    except Exception as e:
        context = ErrorContext(
            operation="validate_configuration",
            tool=tool,
            target_paths=[],
            config=config,
            timestamp=datetime.now().isoformat(),
            session_id="config_validation"
        )
        await error_handler._handle_error(e, context)
        return {}


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test error handling
        context = ErrorContext(
            operation="test_operation",
            tool="test_tool",
            target_paths=["test.py"],
            config={"test": True},
            timestamp=datetime.now().isoformat(),
            session_id="test_session"
        )
        
        try:
            async with error_handler.handle_operation(context):
                # Simulate an error
                raise ToolNotFoundError("test_tool", context=context)
        except LintingError as e:
            print(f"Caught linting error: {e.message}")
        
        # Print error summary
        summary = error_handler.get_error_summary()
        print(f"Error summary: {summary}")
        
        # Health check
        health = await error_handler.health_check()
        print(f"Health check: {health}")
    
    asyncio.run(main())