"""Security Risk Gate - Critical for safety
Currently returns "allow" for everything - DANGEROUS!
"""

import logging
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)


class SecurityRiskGate:
    """Multi-stage validation pipeline with semantic-utility scoring."""

    def __init__(self) -> None:
        # Risk thresholds
        self.low_risk_threshold = 0.3
        self.high_risk_threshold = 0.8

        # Dangerous patterns with names for logging
        self.dangerous_patterns: list[tuple[str, str]] = [
            (r"rm\s+-rf\s+/", "filesystem_wipe"),
            (r"format\s+c:", "disk_format"),
            (r"delete\s+from", "sql_delete"),
            (r"drop\s+table", "sql_drop"),
            (r"insert\s+into", "sql_insert"),
            (r"select\s+.*from.*where\s+1=1", "sql_always_true"),
            (r"<script.*?>", "script_tag"),
            (r";\s*shutdown", "command_shutdown"),
            (r"eval\(", "eval_call"),
            (r"exec\(", "exec_call"),
            (r"__import__", "python_import"),
            (r"os\.system", "os_system"),
            (r"subprocess", "subprocess"),
            (r"\.\./", "path_traversal"),
        ]

        # Sensitive data patterns
        self.sensitive_patterns: list[tuple[str, str]] = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
            (r"\b(?:\d[ -]?){13,16}\b", "credit_card"),
            (r"password\s*=", "password"),
            (r"api[_-]?key", "api_key"),
            (r"secret[_-]?key", "secret_key"),
            (r"private[_-]?key", "private_key"),
        ]

    def risk_gate(self, msg: dict[str, Any], risk: float | None = None) -> Literal["allow", "ask", "deny"]:
        """Assess risk and determine action.

        Args:
            msg: Message dictionary with 'content' and optional 'type'
            risk: Pre-calculated risk score (0-1)

        Returns:
            "allow": Safe to proceed
            "ask": Requires user confirmation
            "deny": Block immediately
        """
        content = msg.get("content", "")
        msg_type = msg.get("type", "text")

        # Calculate risk if not provided
        if risk is None:
            risk = self._calculate_risk(content, msg_type)

        # Log for audit
        logger.info(f"Risk assessment: {risk:.2f} for message type: {msg_type}")

        # Check dangerous patterns
        danger = self._contains_dangerous_patterns(content)
        if danger:
            logger.warning("Dangerous pattern detected: %s in %s", danger, content[:100])
            return "deny"

        # Check sensitive data
        sensitive = self._contains_sensitive_data(content)
        if sensitive:
            logger.warning("Sensitive data detected: %s", sensitive)
            return "ask"

        # Risk-based decision
        if risk < self.low_risk_threshold:
            logger.info("Decision: allow")
            return "allow"
        if risk < self.high_risk_threshold:
            logger.info("Decision: ask")
            return "ask"
        logger.warning("Decision: deny due to high risk %.2f", risk)
        return "deny"

    def _calculate_risk(self, content: str, msg_type: str) -> float:
        """Calculate risk score."""
        risk = 0.0

        # Type-based risk
        type_risks = {
            "command": 0.5,
            "code": 0.4,
            "file": 0.3,
            "text": 0.1,
            "query": 0.1,
        }
        risk += type_risks.get(msg_type, 0.2)

        # Length-based risk (very long content is suspicious)
        if len(content) > 10000:
            risk += 0.2
        elif len(content) > 50000:
            risk += 0.4

        # Special character density
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        special_density = special_chars / max(len(content), 1)
        risk += special_density * 0.3

        # Keyword-based risk
        high_risk_keywords = ["delete", "drop", "execute", "system", "admin", "root"]
        for keyword in high_risk_keywords:
            if keyword in content.lower():
                risk += 0.1

        # Cap at 1.0
        return min(risk, 1.0)

    def _contains_dangerous_patterns(self, content: str) -> str | None:
        """Return name of dangerous pattern if found."""
        for pattern, name in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return name
        return None

    def _contains_sensitive_data(self, content: str) -> str | None:
        """Return name of sensitive data pattern if found."""
        for pattern, name in self.sensitive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return name
        return None


# Global instance
_gate_instance = None


def get_gate_instance():
    """Get or create gate instance."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = SecurityRiskGate()
    return _gate_instance


def risk_gate(msg: dict[str, Any], risk: float | None = None) -> Literal["allow", "ask", "deny"]:
    """Main risk gate function - NO LONGER ALWAYS "allow"!"""
    instance = get_gate_instance()
    return instance.risk_gate(msg, risk)
