#!/usr/bin/env python3
"""
Core Domain Constants Package

This package provides type-safe constants to eliminate magic literals
throughout the AIVillage codebase. Import these constants instead of
using hardcoded values to improve maintainability and reduce coupling.

Usage:
    from core.domain.security_constants import SecurityLevel, UserRole
    from core.domain.system_constants import SystemLimits, MobileProfile

    # Instead of: if level == "error"
    if level == SecurityLevel.ERROR:

    # Instead of: if timeout > 30.0
    if timeout > SystemLimits.DEFAULT_TIMEOUT:
"""

# Re-export commonly used constants for convenience
from .security_constants import SecurityLevel, SecurityLimits, ThreatLevel, TransportSecurity, UserRole
from .system_constants import (
    MessageStatus,
    MobileProfile,
    NetworkConstants,
    OperationMode,
    ProcessingLimits,
    QualityThresholds,
    SystemLimits,
    TransportType,
)

__all__ = [
    # Security Constants
    "SecurityLevel",
    "UserRole",
    "SecurityLimits",
    "TransportSecurity",
    "ThreatLevel",
    # System Constants
    "SystemLimits",
    "MobileProfile",
    "TransportType",
    "MessageStatus",
    "OperationMode",
    "ProcessingLimits",
    "NetworkConstants",
    "QualityThresholds",
]
