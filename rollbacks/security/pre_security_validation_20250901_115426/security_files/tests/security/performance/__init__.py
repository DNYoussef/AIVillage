"""
Security Performance Tests

Tests that validate security controls maintain acceptable performance overhead.
Focuses on performance impact of security mechanisms and scalability under load.
"""

from .test_security_overhead import *

__all__ = [
    "SecurityOverheadTest",
]
