#!/usr/bin/env python3
"""
Phase 7 ADAS Production Validation Framework

Comprehensive validation system for production deployment of ADAS Phase 7 including:
- Automotive certification (ISO 26262, SOTIF)
- Integration validation (Phase 6-7-8)
- Deployment readiness assessment
- Safety-critical quality gates
- Certification evidence generation
"""

from .automotive_certification import (
    AutomotiveCertificationFramework,
    ISO26262Validator,
    SOTIFValidator,
    ASILLevel,
    CertificationStatus
)
from .integration_validation import (
    IntegrationValidationFramework,
    Phase6ToPhase7Validator,
    Phase7ToPhase8Validator,
    EndToEndPipelineValidator,
    IntegrationStatus
)
from .deployment_readiness import (
    DeploymentReadinessFramework,
    HardwareCompatibilityValidator,
    SoftwareDependencyValidator,
    DeploymentPackageCreator,
    ReadinessStatus,
    DeploymentTarget
)
from .quality_gates import (
    QualityGateFramework,
    SafetyCriticalQualityGates,
    GateStatus,
    GateSeverity,
    GateCategory
)
from .validation_framework import (
    Phase7ProductionValidator,
    ValidationSummary,
    CertificationPackage
)

__all__ = [
    # Automotive Certification
    'AutomotiveCertificationFramework',
    'ISO26262Validator',
    'SOTIFValidator',
    'ASILLevel',
    'CertificationStatus',

    # Integration Validation
    'IntegrationValidationFramework',
    'Phase6ToPhase7Validator',
    'Phase7ToPhase8Validator',
    'EndToEndPipelineValidator',
    'IntegrationStatus',

    # Deployment Readiness
    'DeploymentReadinessFramework',
    'HardwareCompatibilityValidator',
    'SoftwareDependencyValidator',
    'DeploymentPackageCreator',
    'ReadinessStatus',
    'DeploymentTarget',

    # Quality Gates
    'QualityGateFramework',
    'SafetyCriticalQualityGates',
    'GateStatus',
    'GateSeverity',
    'GateCategory',

    # Main Framework
    'Phase7ProductionValidator',
    'ValidationSummary',
    'CertificationPackage'
]

__version__ = '1.0.0'
__author__ = 'ADAS Validation Team'
__description__ = 'Production validation framework for Phase 7 ADAS deployment'