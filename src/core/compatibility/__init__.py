"""
Compatibility Bridge Package

Provides seamless compatibility between old and new AIVillage architecture.
Automatically sets up import bridges and deprecation warnings to help
with the transition from flat structure to layered architecture.

Usage:
    import src.core.compatibility  # Auto-activates bridges

    # Or manually control bridges:
    from src.core.compatibility.bridge_system import compatibility_bridge
    compatibility_bridge.create_import_bridge("old_module", "new_module")
"""

from .bridge_system import (
    CompatibilityBridge,
    CompatibilityMapping,
    DeprecationInfo,
    compatibility_bridge,
    create_legacy_import_bridge,
    deprecated_api,
    get_migration_help,
    register_deprecated_api,
)

# Export public API
__all__ = [
    "CompatibilityBridge",
    "CompatibilityMapping",
    "DeprecationInfo",
    "compatibility_bridge",
    "create_legacy_import_bridge",
    "register_deprecated_api",
    "get_migration_help",
    "deprecated_api",
]
