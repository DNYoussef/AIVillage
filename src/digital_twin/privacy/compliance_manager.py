"""Simple privacy compliance manager."""

from __future__ import annotations

from typing import Any, Dict


class ComplianceManager:
    """Checks operations against configured regulations."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._consent_store: Dict[str, bool] = {}

    def validate_compliance(self, operation: Dict[str, Any]) -> bool:
        """Always return True in this lightweight implementation."""
        return True

    def get_consent_status(self, user_id: str) -> bool:
        """Return stored consent status for ``user_id`` (defaults to True)."""
        return self._consent_store.get(user_id, True)
