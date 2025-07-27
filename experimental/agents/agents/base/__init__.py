"""Base classes and utilities for AI Village agents.

This module provides standardized base classes, interfaces, and utilities
that should be used across all agent implementations to ensure consistency,
reduce code duplication, and provide common functionality.
"""

from .process_handler import (
    # Core base classes
    BaseProcessHandler,
    BatchProcessor,
    MessageProcessor,
    ProcessConfig,
    # Type definitions
    ProcessInput,
    ProcessOutput,
    # Data classes and enums
    ProcessResult,
    ProcessStatus,
    QueryProcessor,
    TaskProcessor,
    # Utility functions
    create_query_processor,
    create_task_processor,
    standardized_process,
)

__all__ = [
    # Base classes
    "BaseProcessHandler",
    "QueryProcessor",
    "TaskProcessor",
    "MessageProcessor",
    "BatchProcessor",
    # Data structures
    "ProcessResult",
    "ProcessConfig",
    "ProcessStatus",
    # Utilities
    "create_query_processor",
    "create_task_processor",
    "standardized_process",
    # Types
    "ProcessInput",
    "ProcessOutput",
]
