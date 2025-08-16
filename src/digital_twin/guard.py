"""Safety guard for the digital twin runtime.

The guard evaluates user messages and decides whether an action is allowed,
requires confirmation (``ask``) or must be denied.  Decisions are based on a
few simple heuristics combined with user preferences loaded from the encrypted
preference vault.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from .security.preference_vault import PreferenceVault

# Patterns that indicate potentially unsafe shell or network usage.  The guard
# blocks these when the user has not explicitly enabled shell access via their
# preferences.
_SHELL_PATTERN = re.compile(
    r"\b(rm|ls|cat|wget|curl|scp|ssh|chmod|chown|kill|ps|systemctl)\b",
    re.IGNORECASE,
)
# Detect URLs or other network indicators without embedding insecure
# URL patterns directly in the source.
_NETWORK_PATTERN = re.compile(r"https?://", re.IGNORECASE)

# Secrets or credential patterns which always result in a denial.
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|password|private[_-]?key)"
)


def risk_gate(
    message: dict[str, Any], risk: float | None = None
) -> Literal["allow", "ask", "deny"]:
    """Evaluate ``message`` and return the guard decision.

    Parameters
    ----------
    message:
        Dictionary containing at least ``content`` and optionally ``tool``.
    risk:
        Optional numeric risk estimate from an external model.  Values above
        ``0.8`` trigger a denial and above ``0.5`` require confirmation.
    """

    content = str(message.get("content", ""))

    # Load user preferences (best effort; failures fall back to safe defaults).
    prefs: dict[str, Any]
    try:
        prefs = PreferenceVault().load()
    except Exception:
        prefs = {}

    allow_shell = bool(prefs.get("allow_shell", False))

    if not allow_shell and (
        _SHELL_PATTERN.search(content) or _NETWORK_PATTERN.search(content)
    ):
        return "deny"

    if _SECRET_PATTERN.search(content):
        return "deny"

    tool = message.get("tool")
    allowed_tools = prefs.get("tools", [])
    if tool and tool not in allowed_tools:
        return "ask"

    if risk is not None:
        if risk >= 0.8:
            return "deny"
        if risk >= 0.5:
            return "ask"

    return "allow"


__all__ = ["risk_gate"]
