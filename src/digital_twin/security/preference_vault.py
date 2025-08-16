"""Encrypted preference loader for the digital twin.

This module provides a tiny preference vault used by the twin runtime.  It
loads a single encrypted JSON file from the user's home directory and decrypts
it using a per-user key supplied via the ``TWIN_PREF_KEY`` environment
variable.  The encrypted file lives at ``~/.aivillage/prefs.json.enc`` and is
protected using :class:`cryptography.fernet.Fernet`, which performs both
decryption and authentication.

The implementation intentionally keeps no network or filesystem side effects
beyond reading the preference file and therefore remains suitable for
restricted environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


class PreferenceVault:
    """Load encrypted user preferences.

    Parameters are intentionally minimal; the vault only knows how to locate a
    single encrypted JSON file and decrypt it with a key from the environment.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".aivillage" / "prefs.json.enc"
        self._cache: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Return decrypted preference data.

        Returns cached preferences when available.  Raises ``RuntimeError`` if
        the key is missing, the file does not exist or decryption fails.
        """

        if self._cache is not None:
            return self._cache

        key = os.getenv("TWIN_PREF_KEY")
        if not key:
            raise RuntimeError("TWIN_PREF_KEY not set")

        try:
            encrypted = self.path.read_bytes()
        except FileNotFoundError as exc:  # pragma: no cover - simple error path
            raise RuntimeError("Preferences file not found") from exc

        try:
            decrypted = Fernet(key).decrypt(encrypted)
        except Exception as exc:  # pragma: no cover - rare crypto error
            raise RuntimeError("Failed to decrypt preferences") from exc

        try:
            prefs: dict[str, Any] = json.loads(decrypted.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - invalid JSON
            raise RuntimeError("Invalid preference data") from exc

        self._cache = prefs
        return prefs


# Backwards compatibility for existing imports that expect the previous heavy
# ``SecurePreferenceVault`` implementation.
from .secure_preference_vault import (  # noqa: E402 - imported for compat
    SecurePreferenceVault,
)

__all__ = ["PreferenceVault", "SecurePreferenceVault"]
