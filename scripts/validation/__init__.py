"""Integrated validation framework for AIVillage.

This module consolidates validation functionality from:
- validate_dependencies.py
- check_quality_gates.py
- verify_docs.py

Provides comprehensive validation including:
- Dependency validation
- Code quality checks
- Documentation verification
- Production readiness assessment
"""

from .dependency_validator import DependencyValidator
from .quality_checker import QualityChecker, QualityGate
from .validation_suite import ValidationConfig, ValidationResult, ValidationSuite

__all__ = [
    "DependencyValidator",
    "QualityChecker",
    "QualityGate",
    "ValidationConfig",
    "ValidationResult",
    "ValidationSuite",
]
