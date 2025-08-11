"""Compatibility wrapper exposing :class:`PreferenceVault`."""

from .secure_preference_vault import SecurePreferenceVault


class PreferenceVault(SecurePreferenceVault):
    """Backward-compatible alias for :class:`SecurePreferenceVault`."""


__all__ = ["PreferenceVault", "SecurePreferenceVault"]
