"""
PII/PHI Compliance Management Package.

This package provides comprehensive privacy and healthcare data compliance
management for AIVillage including automated discovery, retention management,
and regulatory compliance monitoring.

Main Components:
- PIIPHIManager: Core compliance management system
- ComplianceCLI: Command-line interface for compliance operations
- PIIDetectionEngine: Automated PII/PHI discovery in data sources
- RetentionManager: Automated data retention and deletion
"""

from .compliance_cli import ComplianceCLI
from .pii_phi_manager import (
    ComplianceRegulation,
    DataClassification,
    DataLocation,
    PIIDetectionEngine,
    PIIDetectionRule,
    PIIPHIManager,
    RetentionJob,
    RetentionPolicy,
)

__version__ = "1.0.0"

__all__ = [
    "PIIPHIManager",
    "PIIDetectionEngine",
    "ComplianceCLI",
    "DataClassification",
    "RetentionPolicy",
    "ComplianceRegulation",
    "DataLocation",
    "RetentionJob",
    "PIIDetectionRule",
]
