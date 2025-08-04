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
        self.high_risk_threshold = 0.7

        # Dangerous patterns
        self.dangerous_patterns = [
            r"rm\s+-rf",
            r"format\s+c:",
            r"delete\s+from",
            r"drop\s+table",
            r"<script",
            r"eval\(",
            r"exec\(",
            r"__import__",
            r"os\.system",
            r"subprocess",
        ]

        # Sensitive data patterns
        self.sensitive_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"password\s*=",
            r"api[_-]?key",
            r"secret[_-]?key",
            r"private[_-]?key",
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
        if self._contains_dangerous_patterns(content):
            logger.warning(f"Dangerous pattern detected in: {content[:100]}...")
            return "deny"

        # Check sensitive data
        if self._contains_sensitive_data(content):
            logger.warning("Sensitive data detected")
            return "ask"

        # Risk-based decision
        if risk < self.low_risk_threshold:
            return "allow"
        if risk < self.high_risk_threshold:
            logger.info("Medium risk - requiring confirmation")
            return "ask"
        logger.warning(f"High risk {risk:.2f} - denying")
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

    def _contains_dangerous_patterns(self, content: str) -> bool:
        """Check for dangerous patterns."""
        for pattern in self.dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _contains_sensitive_data(self, content: str) -> bool:
        """Check for sensitive data patterns."""
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False


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
