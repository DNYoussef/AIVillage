"""Base classes and utilities for AI Village agents.

This module provides standardized base classes, interfaces, and utilities
that should be used across all agent implementations to ensure consistency,
reduce code duplication, and provide common functionality.
"""

from .process_handler import (  # Core base classes; Type definitions; Data classes and enums; Utility functions
    BaseProcessHandler,
    BatchProcessor,
    MessageProcessor,
    ProcessConfig,
    ProcessInput,
    ProcessOutput,
    ProcessResult,
    ProcessStatus,
    QueryProcessor,
    TaskProcessor,
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
