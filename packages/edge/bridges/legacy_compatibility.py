"""
Legacy Edge Device Compatibility Bridge

Provides compatibility layer for migrating from scattered edge device implementations
to the unified edge device architecture.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LegacyEdgeCompatibility:
    """Compatibility bridge for legacy edge device implementations"""

    def __init__(self):
        self.legacy_mappings: dict[str, str] = {}
        self.migration_warnings_shown: set = set()

        logger.info("Legacy edge device compatibility bridge initialized")

    def register_legacy_mapping(self, legacy_path: str, new_path: str):
        """Register mapping from legacy implementation to new unified system"""
        self.legacy_mappings[legacy_path] = new_path
        logger.info(f"Registered legacy mapping: {legacy_path} -> {new_path}")

    def get_migration_status(self) -> dict[str, Any]:
        """Get status of legacy system migration"""
        return {
            "legacy_mappings": len(self.legacy_mappings),
            "migration_paths": self.legacy_mappings.copy(),
            "warnings_shown": len(self.migration_warnings_shown),
        }

    def show_deprecation_warning(self, legacy_component: str) -> None:
        """Show deprecation warning for legacy component (once per component)"""
        if legacy_component not in self.migration_warnings_shown:
            logger.warning(
                f"DEPRECATION WARNING: {legacy_component} is deprecated. "
                f"Please migrate to the unified edge device system in packages/edge/"
            )
            self.migration_warnings_shown.add(legacy_component)
