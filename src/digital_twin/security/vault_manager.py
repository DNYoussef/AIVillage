"""High level manager for the preference vault."""

from __future__ import annotations

from typing import Any

from .preference_vault import PreferenceVault


class VaultManager:
    """Thin wrapper providing a simpler API around :class:`PreferenceVault`."""

    def __init__(self, vault: PreferenceVault | None = None) -> None:
        self.vault = vault or PreferenceVault({})

    async def store(self, *args: Any, **kwargs: Any) -> Any:
        return await self.vault.store_preference(*args, **kwargs)

    async def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        return await self.vault.get_preference(*args, **kwargs)
