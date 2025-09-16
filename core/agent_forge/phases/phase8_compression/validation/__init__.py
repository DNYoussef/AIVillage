"""
Phase 8 Compression Validation Package

Production validation framework for AI model compression with comprehensive
theater detection and deployment readiness verification.

This package provides:
- Compression quality validation with accuracy retention checks
- Deployment readiness validation across platforms and hardware
- Theater detection to identify fake or misleading compression claims
- Comprehensive production readiness assessment

Key Components:
- CompressionQualityValidator: Validates compression quality metrics
- DeploymentReadinessValidator: Validates production deployment readiness
- CompressionTheaterDetector: Detects compression theater patterns
- Phase8ProductionValidator: Orchestrates comprehensive validation

Usage:
    from phase8_compression.validation import Phase8ProductionValidator
    from phase8_compression.validation import ProductionValidationConfig

    config = ProductionValidationConfig(...)
    validator = Phase8ProductionValidator(config)
    result = validator.validate_production_readiness(...)
"""

from .compression_quality_validator import (
    CompressionQualityValidator,
    CompressionMetrics,
    QualityThresholds
)

from .deployment_readiness_validator import (
    DeploymentReadinessValidator,
    DeploymentMetrics,
    HardwareRequirements,
    PlatformCompatibility
)

from .compression_theater_detector import (
    CompressionTheaterDetector,
    TheaterDetectionResult,
    TheaterPattern,
    TheaterEvidence
)

from .phase8_production_validator import (
    Phase8ProductionValidator,
    ProductionValidationConfig,
    ProductionValidationResult
)

__all__ = [
    # Quality Validation
    'CompressionQualityValidator',
    'CompressionMetrics',
    'QualityThresholds',

    # Deployment Validation
    'DeploymentReadinessValidator',
    'DeploymentMetrics',
    'HardwareRequirements',
    'PlatformCompatibility',

    # Theater Detection
    'CompressionTheaterDetector',
    'TheaterDetectionResult',
    'TheaterPattern',
    'TheaterEvidence',

    # Production Validation
    'Phase8ProductionValidator',
    'ProductionValidationConfig',
    'ProductionValidationResult'
]

__version__ = "1.0.0"
__author__ = "AI Village Phase 8 Team"
__description__ = "Production validation framework for AI model compression"