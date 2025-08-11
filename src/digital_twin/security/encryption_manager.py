"""Minimal encryption manager used for import validation."""

from __future__ import annotations

from typing import Any


class EncryptionManager:
    """Provides stubbed encryption/decryption helpers."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def encrypt_data(self, data: Any) -> Any:
        """Return data unchanged; placeholder for real encryption."""
        return data

    def decrypt_data(self, data: Any) -> Any:
        """Return data unchanged; placeholder for real decryption."""
        return data
