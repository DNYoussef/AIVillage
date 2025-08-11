"""Minimal vault manager used for tests."""

from __future__ import annotations

from typing import Any, Dict


class VaultManager:
    """Store and retrieve key/value pairs in an in-memory vault."""

    def __init__(self) -> None:
        self.vault_path = "./vault"
        self._storage: Dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key``."""
        self._storage[key] = value

    def retrieve(self, key: str) -> Any:
        """Return previously stored value or ``None``."""
        return self._storage.get(key)
