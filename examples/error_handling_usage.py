#!/usr/bin/env python3
"""Comprehensive examples demonstrating the new unified error handling system.

This file shows how to use the AIVillageException hierarchy, decorators,
and context managers in various scenarios.
"""

import asyncio
from typing import Any

from core import (
    AIVillageException,
    ErrorCategory,
    ErrorContext,
    ErrorContextManager,
    ErrorSeverity,
    get_component_logger,
    migrate_from_legacy_exception,
    with_error_handling,
)


# Example 1: Basic Exception Usage
def example_basic_exception():
    """Demonstrate basic exception creation and usage."""
    print("=== Example 1: Basic Exception Usage ===")

    try:
        raise AIVillageException(
            "Invalid configuration provided",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            context=ErrorContext(
                component="ConfigService",
                operation="validate_config",
                details={"config_file": "settings.json", "line": 42},
            ),
        )
    except AIVillageException as e:
        print(f"Caught exception: {e}")
        print(f"Details: {e.message}")
        print(f"Category: {e.category.value}")
        print(f"Severity: {e.severity.value}")


# Example 2: Using the Error Handling Decorator
@with_error_handling(
    context={"component": "DataProcessor", "method": "process_data"},
    category=ErrorCategory.PROCESSING,
    severity=ErrorSeverity.ERROR,
)
def process_data(data: dict[str, Any]) -> dict[str, Any]:
    """Example function with error handling decorator."""
    if not data:
        raise ValueError("Empty data provided")

    if "critical_field" not in data:
        raise KeyError("Missing required field: critical_field")

    # Simulate processing
    return {"processed": True, "data": data}


@with_error_handling(
    context={"component": "AsyncService", "method": "fetch_data"},
    category=ErrorCategory.NETWORK,
    severity=ErrorSeverity.WARNING,
)
async def fetch_data_async(url: str) -> str:
    """Async example with error handling."""
    if not url.startswith("http"):
        raise ValueError("Invalid URL format")

    # Simulate async operation
    await asyncio.sleep(0.1)
    return f"Data from {url}"


# Example 3: Using Context Manager
def example_context_manager():
    """Demonstrate using ErrorContextManager."""
    print("\n=== Example 3: Context Manager Usage ===")

    try:
        with ErrorContextManager(
            component="DatabaseService",
            operation="save_record",
            details={"table": "users", "record_id": 123},
        ):
            # Simulate database operation
            raise ValueError("Connection timeout")

    except AIVillageException as e:
        print(f"Context manager caught: {e}")
        if e.context:
            print(f"Component: {e.context.component}")
            print(f"Operation: {e.context.operation}")
            print(f"Details: {e.context.details}")


# Example 4: Component Logger
def example_component_logger():
    """Demonstrate component-specific logging."""
    print("\n=== Example 4: Component Logging ===")

    logger = get_component_logger("MyService")

    try:
        raise AIVillageException(
            "Service initialization failed",
            category=ErrorCategory.INITIALIZATION,
            severity=ErrorSeverity.CRITICAL,
            context=ErrorContext(
                component="MyService",
                operation="initialize",
                details={"config_file": "service.yaml"},
            ),
        )
    except AIVillageException as e:
        logger.error(f"Service error: {e.message}")


# Example 5: Migrating from Legacy Exceptions
def example_legacy_migration():
    """Demonstrate migrating from legacy exceptions."""
    print("\n=== Example 5: Legacy Migration ===")

    try:
        # Simulate legacy exception
        legacy_error = ValueError("Legacy validation error")

        # Migrate to new exception
        new_exception = migrate_from_legacy_exception(legacy_error)

        # Enhance with additional context
        enhanced_exception = AIVillageException(
            f"Migrated error: {new_exception.message}",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            context=ErrorContext(
                component="LegacyService",
                operation="validate_input",
                details={"input": "user_data", "original_error": str(legacy_error)},
            ),
            original_exception=legacy_error,
        )

        raise enhanced_exception

    except AIVillageException as e:
        print(f"Migrated exception: {e}")
        print(f"Original exception: {e.original_exception}")


# Example 6: Complex Error Handling Scenario
class DataProcessingService:
    """Example service demonstrating comprehensive error handling."""

    def __init__(self):
        self.logger = get_component_logger("DataProcessingService")

    @with_error_handling(
        context={"component": "DataProcessingService", "method": "process_batch"},
        category=ErrorCategory.PROCESSING,
        severity=ErrorSeverity.ERROR,
    )
    def process_batch(self, batch_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a batch of data with comprehensive error handling."""
        if not batch_data:
            raise AIVillageException(
                "Empty batch provided",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.WARNING,
                context=ErrorContext(
                    component="DataProcessingService",
                    operation="process_batch",
                    details={"batch_size": 0},
                ),
            )

        results = []
        for idx, item in enumerate(batch_data):
            try:
                with ErrorContextManager(
                    component="DataProcessingService",
                    operation="process_item",
                    details={"item_index": idx, "item_id": item.get("id")},
                ):
                    # Simulate processing
                    if item.get("id") == "error_item":
                        raise ValueError("Simulated processing error")

                    results.append({"id": item.get("id"), "status": "processed"})

            except AIVillageException as e:
                self.logger.error(f"Failed to process item {idx}: {e.message}")
                continue

        return {
            "processed_count": len(results),
            "failed_count": len(batch_data) - len(results),
            "results": results,
        }


# Example 7: Async Error Handling
class AsyncDataService:
    """Example async service with error handling."""

    def __init__(self):
        self.logger = get_component_logger("AsyncDataService")

    async def fetch_multiple(self, urls: list[str]) -> dict[str, str]:
        """Fetch data from multiple URLs with error handling."""
        results = {}

        for url in urls:
            try:
                data = await fetch_data_async(url)
                results[url] = data
            except AIVillageException as e:
                self.logger.error(f"Failed to fetch {url}: {e.message}")
                results[url] = f"ERROR: {e.message}"

        return results


# Main demonstration
async def main():
    """Run all examples."""
    print("AIVillage Error Handling Examples")
    print("=" * 50)

    # Example 1: Basic exception
    example_basic_exception()

    # Example 2: Decorator usage
    print("\n=== Example 2: Decorator Usage ===")
    try:
        result = process_data({"name": "test", "critical_field": "value"})
        print(f"Processing result: {result}")
    except AIVillageException as e:
        print(f"Processing failed: {e.message}")

    # Example 3: Context manager
    example_context_manager()

    # Example 4: Component logger
    example_component_logger()

    # Example 5: Legacy migration
    example_legacy_migration()

    # Example 6: Complex service
    print("\n=== Example 6: Complex Service ===")
    service = DataProcessingService()

    batch_data = [
        {"id": "item1", "data": "value1"},
        {"id": "error_item", "data": "will_fail"},
        {"id": "item3", "data": "value3"},
    ]

    try:
        result = service.process_batch(batch_data)
        print(f"Batch processing result: {result}")
    except AIVillageException as e:
        print(f"Batch processing failed: {e.message}")

    # Example 7: Async service
    print("\n=== Example 7: Async Service ===")
    async_service = AsyncDataService()

    urls = ["http://example.com", "invalid_url", "http://another.com"]
    try:
        results = await async_service.fetch_multiple(urls)
        print(f"Async fetch results: {results}")
    except AIVillageException as e:
        print(f"Async fetch failed: {e.message}")


if __name__ == "__main__":
    asyncio.run(main())
