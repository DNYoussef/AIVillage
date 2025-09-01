"""
Fog Security Infrastructure

This module provides comprehensive security capabilities for federated learning systems,
including consensus security, authentication, threat modeling, audit trails, and 
key management. All components integrate with BetaNet infrastructure for enhanced
security and blockchain anchoring.

Key Components:
- ConsensusSecurityManager: Byzantine fault-tolerant security for distributed consensus
- FederatedAuthenticationSystem: Multi-factor authentication and RBAC
- ThreatModelingEngine: Comprehensive threat analysis and risk assessment
- AuditTrailSystem: Immutable audit logging with blockchain anchoring
- SecureKeyManager: Distributed key management and rotation

Features:
- Threshold cryptography for secure aggregation
- Zero-knowledge proofs for privacy protection
- Attack detection and mitigation
- Compliance reporting (GDPR, HIPAA, SOC2, ISO27001)
- Forensic analysis capabilities
- Automated key rotation and backup
- Hardware security module (HSM) integration
- Multi-party computation protocols
"""

from .consensus_security_manager import (
    ConsensusSecurityManager,
    ThresholdCryptographySystem,
    ZeroKnowledgeProofSystem,
    AttackDetectionSystem,
    SecureFederatedAggregation,
    FederatedSecurityCoordinator,
    create_federated_security_coordinator,
    AttackType,
    SecurityLevel,
    SecurityAlert,
    ThresholdSignature,
    ZKProof,
)

from .federated_auth_system import (
    FederatedAuthenticationSystem,
    CryptographicKeyManager,
    MultiFactorAuthenticator,
    RoleBasedAccessControl,
    FederatedSessionManager,
    create_federated_auth_system,
    NodeRole,
    AuthenticationMethod,
    SessionStatus,
    NodeIdentity,
    AuthenticationChallenge,
    AuthenticationResponse,
    SessionToken,
    AccessRequest,
)

from .threat_modeling_system import (
    ThreatModelingEngine,
    FederatedThreatDatabase,
    create_threat_modeling_system,
    ThreatCategory,
    AttackVector,
    RiskLevel,
    ThreatActor,
    Asset,
    Vulnerability,
    ThreatScenario,
    SecurityControl,
    MitigationStatus,
)

from .audit_trail_system import (
    AuditTrailSystem,
    EventBuffer,
    ComplianceEngine,
    AnomalyDetectionEngine,
    ForensicAnalysisEngine,
    create_audit_trail_system,
    EventType,
    EventSeverity,
    ComplianceFramework,
    AuditEvent,
    ComplianceRule,
    ForensicQuery,
    AnomalyPattern,
)

from .secure_key_manager import (
    SecureKeyManager,
    SecureKeyStore,
    DistributedKeyGenerator,
    KeyRotationManager,
    KeyBackupManager,
    create_secure_key_manager,
    KeyType,
    KeyStatus,
    RotationTrigger,
    CryptographicKey,
    KeyShare,
    KeyBackup,
    RotationEvent,
)

# Version information
__version__ = "1.0.0"
__author__ = "AI Village Security Team"
__email__ = "security@aivillage.dev"

# Module exports
__all__ = [
    # Consensus Security
    "ConsensusSecurityManager",
    "ThresholdCryptographySystem",
    "ZeroKnowledgeProofSystem",
    "AttackDetectionSystem",
    "SecureFederatedAggregation",
    "FederatedSecurityCoordinator",
    "create_federated_security_coordinator",
    "AttackType",
    "SecurityLevel",
    "SecurityAlert",
    "ThresholdSignature",
    "ZKProof",
    # Authentication & Authorization
    "FederatedAuthenticationSystem",
    "CryptographicKeyManager",
    "MultiFactorAuthenticator",
    "RoleBasedAccessControl",
    "FederatedSessionManager",
    "create_federated_auth_system",
    "NodeRole",
    "AuthenticationMethod",
    "SessionStatus",
    "NodeIdentity",
    "AuthenticationChallenge",
    "AuthenticationResponse",
    "SessionToken",
    "AccessRequest",
    # Threat Modeling
    "ThreatModelingEngine",
    "FederatedThreatDatabase",
    "create_threat_modeling_system",
    "ThreatCategory",
    "AttackVector",
    "RiskLevel",
    "ThreatActor",
    "Asset",
    "Vulnerability",
    "ThreatScenario",
    "SecurityControl",
    "MitigationStatus",
    # Audit & Compliance
    "AuditTrailSystem",
    "EventBuffer",
    "ComplianceEngine",
    "AnomalyDetectionEngine",
    "ForensicAnalysisEngine",
    "create_audit_trail_system",
    "EventType",
    "EventSeverity",
    "ComplianceFramework",
    "AuditEvent",
    "ComplianceRule",
    "ForensicQuery",
    "AnomalyPattern",
    # Key Management
    "SecureKeyManager",
    "SecureKeyStore",
    "DistributedKeyGenerator",
    "KeyRotationManager",
    "KeyBackupManager",
    "create_secure_key_manager",
    "KeyType",
    "KeyStatus",
    "RotationTrigger",
    "CryptographicKey",
    "KeyShare",
    "KeyBackup",
    "RotationEvent",
]

# Configuration defaults
DEFAULT_SECURITY_CONFIG = {
    "consensus": {
        "threshold_crypto_enabled": True,
        "zk_proofs_enabled": True,
        "byzantine_tolerance": 1,
        "attack_detection_enabled": True,
        "reputation_threshold": 0.5,
    },
    "authentication": {
        "require_mfa": True,
        "session_timeout": 3600,
        "max_login_attempts": 3,
        "password_complexity": "high",
    },
    "audit": {
        "blockchain_anchoring": True,
        "anomaly_detection": True,
        "compliance_frameworks": ["SOC2", "ISO27001"],
        "retention_period_days": 2555,  # 7 years
    },
    "key_management": {
        "auto_rotation": True,
        "backup_required": True,
        "rotation_interval_days": 90,
        "key_strength": 2048,
    },
}

# Compliance mappings
COMPLIANCE_MAPPINGS = {
    "GDPR": {
        "data_access_logging": True,
        "consent_tracking": True,
        "retention_limits": True,
        "right_to_be_forgotten": True,
    },
    "HIPAA": {
        "phi_access_logging": True,
        "minimum_necessary": True,
        "audit_controls": True,
        "encryption_required": True,
    },
    "SOC2": {"security_monitoring": True, "access_controls": True, "system_availability": True, "data_integrity": True},
    "ISO27001": {
        "risk_assessment": True,
        "security_controls": True,
        "incident_management": True,
        "continuous_monitoring": True,
    },
}

# Security event severity levels
SEVERITY_LEVELS = {"INFO": 0, "NOTICE": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

# Attack detection thresholds
ATTACK_THRESHOLDS = {
    "byzantine": 0.7,
    "sybil": 0.8,
    "eclipse": 0.6,
    "dos": 0.9,
    "gradient_inversion": 0.8,
    "model_poisoning": 0.7,
}


def get_security_config(component: str) -> dict:
    """Get default security configuration for component"""
    return DEFAULT_SECURITY_CONFIG.get(component, {})


def get_compliance_requirements(framework: str) -> dict:
    """Get compliance requirements for framework"""
    return COMPLIANCE_MAPPINGS.get(framework, {})


def validate_security_config(config: dict) -> bool:
    """Validate security configuration"""
    required_components = ["consensus", "authentication", "audit", "key_management"]

    for component in required_components:
        if component not in config:
            return False

    return True


# Module initialization
import logging

logger = logging.getLogger(__name__)
logger.info(f"Initialized fog security infrastructure v{__version__}")
