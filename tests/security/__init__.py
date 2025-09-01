"""
Security Test Suite

Comprehensive security testing framework for AIVillage security implementations.
Validates all security components with behavioral testing approach and connascence compliance.

Test Categories:
- Unit tests: Individual security components
- Integration tests: Security workflows and processes  
- Performance tests: Security overhead validation
- Compliance tests: Governance framework validation
- Negative tests: Attack prevention validation

Testing Principles:
- Behavioral testing: Test security contracts, not internal implementation
- Connascence awareness: Ensure tests don't create coupling violations
- Security-focused assertions: Validate security properties and guarantees
"""

__version__ = "1.0.0"

from .unit import *
from .integration import *
from .performance import *
from .compliance import *
from .negative import *

__all__ = [
    # Unit tests
    "test_vulnerability_reporting",
    "test_security_templates",
    "test_dependency_auditing",
    "test_sbom_generation",
    "test_admin_security",
    "test_boundary_security",
    "test_grokfast_security",
    # Integration tests
    "test_security_workflows",
    "test_end_to_end_security",
    # Performance tests
    "test_security_overhead",
    # Compliance tests
    "test_governance_framework",
    # Negative tests
    "test_attack_prevention",
]
