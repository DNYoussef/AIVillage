"""
Negative Security Testing

Tests that validate security controls prevent malicious actions and attacks.
Focuses on attack prevention and security boundary enforcement under adversarial conditions.
"""

from .test_attack_prevention import *

__all__ = [
    'AttackPreventionTest',
]