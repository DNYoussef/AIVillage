"""
Compatibility Bridge System

Provides compatibility bridges between the new architectural layers and existing code.
Enables gradual migration to the new hardware/software architecture while maintaining
backward compatibility with legacy components.

Features:
- Legacy API compatibility wrappers
- Protocol adaptation layers
- Import path redirection
- Gradual migration utilities
- Deprecation warnings and guidance
"""

import importlib
import logging
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityMapping:
    """Defines a compatibility mapping between old and new APIs."""

    old_module: str
    new_module: str
    old_class: str | None = None
    new_class: str | None = None
    deprecated_in: str | None = None
    removed_in: str | None = None
    migration_guide: str | None = None


@dataclass
class DeprecationInfo:
    """Information about deprecated functionality."""

    item_name: str
    deprecated_since: str
    removal_version: str
    replacement: str
    migration_notes: str
    warning_issued: bool = False


class CompatibilityBridge:
    """
    Main compatibility bridge system for managing API transitions.

    Handles:
    - Import redirection
    - API compatibility wrappers
    - Deprecation warnings
    - Migration tracking
    """

    def __init__(self):
        """Initialize compatibility bridge system."""
        self.mappings: dict[str, CompatibilityMapping] = {}
        self.deprecations: dict[str, DeprecationInfo] = {}
        self.active_bridges: dict[str, Any] = {}
        self.allowed_modules: set[str] = set()
        self.migration_stats = {
            "total_redirects": 0,
            "active_deprecations": 0,
            "completed_migrations": 0,
        }

        # Load built-in compatibility mappings
        self._load_builtin_mappings()

        logger.info("Compatibility bridge system initialized")

    def register_mapping(self, mapping: CompatibilityMapping) -> bool:
        """
        Register a compatibility mapping between old and new APIs.

        Args:
            mapping: CompatibilityMapping defining the transition

        Returns:
            bool: True if successfully registered
        """
        try:
            key = f"{mapping.old_module}.{mapping.old_class or ''}"
            self.mappings[key] = mapping
            self.allowed_modules.update({mapping.old_module, mapping.new_module})

            # Register deprecation if specified
            if mapping.deprecated_in:
                self.register_deprecation(
                    item_name=key,
                    deprecated_since=mapping.deprecated_in,
                    removal_version=mapping.removed_in or "2.0.0",
                    replacement=f"{mapping.new_module}.{mapping.new_class or ''}",
                    migration_notes=mapping.migration_guide or "Update import path",
                )

            logger.debug(f"Registered compatibility mapping: {mapping.old_module} -> {mapping.new_module}")
            return True

        except Exception as e:
            logger.error(f"Failed to register mapping: {e}")
            return False

    def register_deprecation(
        self,
        item_name: str,
        deprecated_since: str,
        removal_version: str,
        replacement: str,
        migration_notes: str,
    ) -> bool:
        """
        Register a deprecated API item.

        Args:
            item_name: Name of deprecated item
            deprecated_since: Version when deprecated
            removal_version: Version when will be removed
            replacement: Replacement API
            migration_notes: Migration guidance

        Returns:
            bool: True if successfully registered
        """
        try:
            self.deprecations[item_name] = DeprecationInfo(
                item_name=item_name,
                deprecated_since=deprecated_since,
                removal_version=removal_version,
                replacement=replacement,
                migration_notes=migration_notes,
            )

            self.migration_stats["active_deprecations"] += 1
            logger.debug(f"Registered deprecation: {item_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register deprecation: {e}")
            return False

    def create_import_bridge(self, old_module: str, new_module: str) -> bool:
        """Create an import bridge that redirects old imports to new modules.

        Args:
            old_module: Old module path
            new_module: New module path

        Returns:
            bool: True if bridge created successfully
        """
        try:
            if old_module not in self.allowed_modules or new_module not in self.allowed_modules:
                raise ValueError(
                    f"Unauthorized module redirection attempted: {old_module} -> {new_module}"
                )

            target_module = importlib.import_module(new_module)

            import types

            def __getattr__(name: str):
                warnings.warn(
                    f"Module {old_module} is deprecated. Use {new_module} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(target_module, name)

            bridge_module = types.ModuleType(old_module)
            bridge_module.__getattr__ = __getattr__
            bridge_module.__doc__ = (
                f"Compatibility bridge for {old_module} -> {new_module}. This module is "
                "deprecated and will be removed in a future version."
            )

            sys.modules[old_module] = bridge_module
            self.active_bridges[old_module] = bridge_module

            self.migration_stats["total_redirects"] += 1
            logger.info(f"Created import bridge: {old_module} -> {new_module}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to create import bridge {old_module} -> {new_module}: {e}"
            )
            return False

    def create_class_wrapper(self, old_class: type, new_class: type, deprecated_since: str = "1.0.0") -> type:
        """
        Create a wrapper class that provides backward compatibility.

        Args:
            old_class: Original class type
            new_class: New class type
            deprecated_since: Version when deprecated

        Returns:
            Type: Wrapper class that forwards to new implementation
        """

        class CompatibilityWrapper(new_class):
            """Compatibility wrapper for deprecated class."""

            def __init__(self, *args, **kwargs):
                # Issue deprecation warning
                self._issue_deprecation_warning(
                    f"{old_class.__name__} is deprecated since v{deprecated_since}. "
                    f"Use {new_class.__name__} instead."
                )

                # Initialize with new class
                super().__init__(*args, **kwargs)

            def _issue_deprecation_warning(self, message: str):
                warnings.warn(message, DeprecationWarning, stacklevel=3)

        # Copy metadata from old class
        CompatibilityWrapper.__name__ = old_class.__name__
        CompatibilityWrapper.__qualname__ = old_class.__qualname__
        CompatibilityWrapper.__module__ = old_class.__module__

        return CompatibilityWrapper

    def create_function_wrapper(
        self, old_func: Callable, new_func: Callable, deprecated_since: str = "1.0.0"
    ) -> Callable:
        """
        Create a wrapper function that provides backward compatibility.

        Args:
            old_func: Original function
            new_func: New function
            deprecated_since: Version when deprecated

        Returns:
            Callable: Wrapper function that forwards to new implementation
        """

        @wraps(old_func)
        def wrapper(*args, **kwargs):
            # Issue deprecation warning
            warnings.warn(
                f"{old_func.__name__} is deprecated since v{deprecated_since}. " f"Use {new_func.__name__} instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Call new function
            return new_func(*args, **kwargs)

        return wrapper

    def get_migration_guidance(self, old_api: str) -> str | None:
        """
        Get migration guidance for a deprecated API.

        Args:
            old_api: Old API identifier

        Returns:
            Optional[str]: Migration guidance or None if not found
        """
        if old_api in self.deprecations:
            deprecation = self.deprecations[old_api]
            return f"""
Migration Guide for {deprecation.item_name}:

DEPRECATED: {deprecation.deprecated_since}
REMOVAL: {deprecation.removal_version}
REPLACEMENT: {deprecation.replacement}

Notes: {deprecation.migration_notes}

Example migration:
  Old: from {old_api.split(".")[0]} import {old_api.split(".")[-1]}
  New: from {deprecation.replacement.split(".")[0]} import {deprecation.replacement.split(".")[-1]}
"""

        return None

    def check_deprecated_usage(self, module_name: str) -> list[str]:
        """
        Check a module for usage of deprecated APIs.

        Args:
            module_name: Name of module to check

        Returns:
            List[str]: List of deprecated APIs found
        """
        deprecated_usage = []

        try:
            # This would require AST parsing to find all imports and usage
            # For now, return placeholder functionality
            if module_name in sys.modules:
                module = sys.modules[module_name]

                # Check module attributes against known deprecations
                for attr_name in dir(module):
                    full_name = f"{module_name}.{attr_name}"
                    if full_name in self.deprecations:
                        deprecated_usage.append(full_name)

        except Exception as e:
            logger.error(f"Error checking deprecated usage in {module_name}: {e}")

        return deprecated_usage

    def generate_migration_report(self) -> str:
        """
        Generate a comprehensive migration report.

        Returns:
            str: Formatted migration report
        """
        report = []
        report.append("=" * 60)
        report.append("AIVillage Compatibility Bridge Migration Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Statistics
        report.append("Migration Statistics:")
        report.append(f"  Total Import Redirects: {self.migration_stats['total_redirects']}")
        report.append(f"  Active Deprecations: {self.migration_stats['active_deprecations']}")
        report.append(f"  Completed Migrations: {self.migration_stats['completed_migrations']}")
        report.append("")

        # Active deprecations
        if self.deprecations:
            report.append("Active Deprecations:")
            report.append("-" * 40)

            for name, info in self.deprecations.items():
                report.append(f"  {name}")
                report.append(f"    Deprecated: {info.deprecated_since}")
                report.append(f"    Removal: {info.removal_version}")
                report.append(f"    Replacement: {info.replacement}")
                report.append("")

        # Active bridges
        if self.active_bridges:
            report.append("Active Import Bridges:")
            report.append("-" * 40)

            for old_module in self.active_bridges:
                if old_module in self.mappings:
                    mapping = self.mappings[old_module]
                    report.append(f"  {old_module} -> {mapping.new_module}")
                else:
                    report.append(f"  {old_module} -> (bridge active)")
            report.append("")

        # Migration recommendations
        report.append("Recommended Actions:")
        report.append("-" * 40)

        if self.deprecations:
            report.append("1. Update imports to use new module paths")
            report.append("2. Replace deprecated classes with new implementations")
            report.append("3. Update function calls to use new API signatures")
            report.append("4. Test thoroughly after migration")
        else:
            report.append("No active deprecations - migration is complete!")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def _load_builtin_mappings(self):
        """Load built-in compatibility mappings for AIVillage architecture."""
        builtin_mappings = [
            # Agent Forge migrations
            CompatibilityMapping(
                old_module="agent_forge",
                new_module="src.software.agent_forge",
                deprecated_in="1.0.0",
                removed_in="2.0.0",
                migration_guide="Update imports to use src.software.agent_forge",
            ),
            # Production migrations
            CompatibilityMapping(
                old_module="production",
                new_module="src.production",
                deprecated_in="1.0.0",
                removed_in="2.0.0",
                migration_guide="Update imports to use src.production",
            ),
            # Communication system migrations
            CompatibilityMapping(
                old_module="communications",
                new_module="src.communications",
                deprecated_in="1.0.0",
                removed_in="2.0.0",
                migration_guide="Update imports to use src.communications",
            ),
            # Core system migrations
            CompatibilityMapping(
                old_module="core",
                new_module="src.core",
                deprecated_in="1.0.0",
                removed_in="2.0.0",
                migration_guide="Update imports to use src.core",
            ),
            # Hardware layer migrations
            CompatibilityMapping(
                old_module="mobile",
                new_module="src.hardware.edge",
                deprecated_in="1.0.0",
                removed_in="2.0.0",
                migration_guide="Mobile components moved to hardware edge layer",
            ),
            # Meta-agents migrations
            CompatibilityMapping(
                old_module="agents",
                new_module="src.software.meta_agents",
                deprecated_in="1.0.0",
                removed_in="2.0.0",
                migration_guide="Agents moved to software meta-agents architecture",
            ),
        ]

        # Register all built-in mappings
        for mapping in builtin_mappings:
            self.register_mapping(mapping)

    def get_bridge_stats(self) -> dict[str, Any]:
        """Get compatibility bridge statistics."""
        return {
            "total_mappings": len(self.mappings),
            "active_bridges": len(self.active_bridges),
            "active_deprecations": len(self.deprecations),
            "migration_stats": self.migration_stats,
            "bridge_health": "healthy" if len(self.active_bridges) > 0 else "inactive",
        }


# Global compatibility bridge instance
compatibility_bridge = CompatibilityBridge()


# Convenience functions for common operations
def create_legacy_import_bridge(old_module: str, new_module: str) -> bool:
    """
    Create a legacy import bridge for backward compatibility.

    Args:
        old_module: Old module import path
        new_module: New module import path

    Returns:
        bool: True if bridge created successfully
    """
    return compatibility_bridge.create_import_bridge(old_module, new_module)


def register_deprecated_api(
    item_name: str,
    deprecated_since: str,
    replacement: str,
    removal_version: str = "2.0.0",
    notes: str = "Update to new API",
) -> bool:
    """
    Register a deprecated API for tracking and warnings.

    Args:
        item_name: Name of deprecated API item
        deprecated_since: Version when deprecated
        replacement: Replacement API
        removal_version: Version when will be removed
        notes: Migration notes

    Returns:
        bool: True if successfully registered
    """
    return compatibility_bridge.register_deprecation(
        item_name=item_name,
        deprecated_since=deprecated_since,
        removal_version=removal_version,
        replacement=replacement,
        migration_notes=notes,
    )


def get_migration_help(api_name: str) -> str | None:
    """
    Get migration help for a deprecated API.

    Args:
        api_name: Name of the API to get help for

    Returns:
        Optional[str]: Migration guidance or None if not found
    """
    return compatibility_bridge.get_migration_guidance(api_name)


def deprecated_api(deprecated_since: str, replacement: str, removal_version: str = "2.0.0"):
    """
    Decorator to mark APIs as deprecated.

    Args:
        deprecated_since: Version when deprecated
        replacement: Replacement API
        removal_version: Version when will be removed
    """

    def decorator(func_or_class):
        # Register the deprecation
        full_name = f"{func_or_class.__module__}.{func_or_class.__name__}"
        register_deprecated_api(
            item_name=full_name,
            deprecated_since=deprecated_since,
            replacement=replacement,
            removal_version=removal_version,
            notes=f"Use {replacement} instead of {func_or_class.__name__}",
        )

        # Add deprecation wrapper
        if isinstance(func_or_class, type):
            # Class decorator
            original_init = func_or_class.__init__

            def wrapped_init(self, *args, **kwargs):
                warnings.warn(
                    f"{func_or_class.__name__} is deprecated since v{deprecated_since}. "
                    f"Use {replacement} instead. Will be removed in v{removal_version}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                original_init(self, *args, **kwargs)

            func_or_class.__init__ = wrapped_init
            return func_or_class
        else:
            # Function decorator
            @wraps(func_or_class)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"{func_or_class.__name__} is deprecated since v{deprecated_since}. "
                    f"Use {replacement} instead. Will be removed in v{removal_version}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func_or_class(*args, **kwargs)

            return wrapper

    return decorator


# Auto-setup common bridges on import
def setup_aivillage_bridges():
    """Set up common AIVillage compatibility bridges."""
    try:
        # Create bridges for moved modules
        bridges_to_create = [
            ("agent_forge", "src.software.agent_forge"),
            ("production", "src.production"),
            ("communications", "src.communications"),
            ("mobile", "src.hardware.edge"),
            ("agents", "src.software.meta_agents"),
        ]

        bridges_created = 0
        for old_module, new_module in bridges_to_create:
            try:
                if create_legacy_import_bridge(old_module, new_module):
                    bridges_created += 1
            except Exception as e:
                logger.debug(f"Could not create bridge {old_module} -> {new_module}: {e}")

        if bridges_created > 0:
            logger.info(f"Created {bridges_created} compatibility bridges")

    except Exception as e:
        logger.warning(f"Error setting up compatibility bridges: {e}")


# Auto-setup bridges when module is imported
setup_aivillage_bridges()
