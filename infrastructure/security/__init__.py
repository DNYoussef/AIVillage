"""
AIVillage Security Infrastructure
=================================

Comprehensive security implementation for federated learning and distributed consensus.

This module provides:
- Federated Authentication System with RBAC
- Secure Multi-Party Aggregation with Homomorphic Encryption
- BetaNet Transport Security Manager
- Byzantine Fault-Tolerant Consensus Security
- Advanced Threat Detection and Mitigation
- Reputation-Based Trust Management
- Unified Security Integration Manager

Key Components:
- FederatedAuthenticationSystem: Multi-factor auth for federated nodes
- SecureAggregationProtocol: Privacy-preserving gradient aggregation
- BetaNetSecurityManager: Secure transport with threat protection
- ConsensusSecurityManager: Byzantine-resilient consensus security
- ThreatDetectionSystem: ML-based threat detection and response
- ReputationTrustSystem: Dynamic trust scoring and management
- SecurityIntegrationManager: Unified security orchestration

Usage:
    from infrastructure.security import SecurityIntegrationManager
    
    security_manager = SecurityIntegrationManager("node_id")
    await security_manager.initialize()
    
    # Register federated node
    success, totp = await security_manager.register_federated_node(
        "participant_1", NodeRole.PARTICIPANT, "secure_password"
    )
    
    # Authenticate participant
    auth_success, session_id, metadata = await security_manager.authenticate_federated_participant(
        "participant_1", "secure_password"
    )
    
    # Perform secure aggregation
    success, result, metadata = await security_manager.secure_federated_aggregation(
        "aggregation_1", participant_gradients
    )
"""

from .federated_auth_system import (
    FederatedAuthenticationSystem,
    NodeRole,
    AuthenticationMethod,
    AuthenticationStatus,
    NodeIdentity,
    AuthenticationChallenge,
    AuthenticationSession
)

from .secure_aggregation import (
    SecureAggregationProtocol,
    AggregationMethod,
    PrivacyLevel,
    SecureGradient,
    AggregationShare,
    HomomorphicKey,
    PrivacyBudget
)

from .betanet_security_manager import (
    BetaNetSecurityManager,
    SecurityLevel,
    ThreatType,
    ChannelType,
    SecurityCredential,
    SecureChannel,
    ThreatEvent as BetaNetThreatEvent,
    SecurityPolicy
)

from .consensus_security_manager import (
    ConsensusSecurityManager,
    ConsensusProtocol,
    AttackType,
    NodeRole as ConsensusNodeRole,
    ThresholdKey,
    ConsensusMessage,
    AttackEvidence,
    ConsensusRound
)

from .threat_detection_system import (
    ThreatDetectionSystem,
    ThreatLevel,
    ThreatCategory,
    AttackVector,
    ThreatIndicator,
    ThreatEvent,
    BehaviorProfile,
    MitigationStrategy
)

from .reputation_trust_system import (
    ReputationTrustSystem,
    TrustMetric,
    ReputationEvent,
    TrustLevel,
    ReputationScore,
    TrustProfile,
    TrustTransaction,
    RecommendationCredential
)

from .security_integration_manager import (
    SecurityIntegrationManager,
    SecurityEvent,
    SecurityConfiguration,
    SecurityMetrics
)

__all__ = [
    # Authentication
    'FederatedAuthenticationSystem',
    'NodeRole',
    'AuthenticationMethod',
    'AuthenticationStatus',
    'NodeIdentity',
    'AuthenticationChallenge',
    'AuthenticationSession',
    
    # Secure Aggregation
    'SecureAggregationProtocol',
    'AggregationMethod',
    'PrivacyLevel',
    'SecureGradient',
    'AggregationShare',
    'HomomorphicKey',
    'PrivacyBudget',
    
    # Transport Security
    'BetaNetSecurityManager',
    'SecurityLevel',
    'ThreatType',
    'ChannelType',
    'SecurityCredential',
    'SecureChannel',
    'SecurityPolicy',
    
    # Consensus Security
    'ConsensusSecurityManager',
    'ConsensusProtocol',
    'AttackType',
    'ThresholdKey',
    'ConsensusMessage',
    'AttackEvidence',
    'ConsensusRound',
    
    # Threat Detection
    'ThreatDetectionSystem',
    'ThreatLevel',
    'ThreatCategory',
    'AttackVector',
    'ThreatIndicator',
    'ThreatEvent',
    'BehaviorProfile',
    'MitigationStrategy',
    
    # Trust Management
    'ReputationTrustSystem',
    'TrustMetric',
    'ReputationEvent',
    'TrustLevel',
    'ReputationScore',
    'TrustProfile',
    'TrustTransaction',
    'RecommendationCredential',
    
    # Integration
    'SecurityIntegrationManager',
    'SecurityEvent',
    'SecurityConfiguration',
    'SecurityMetrics'
]

# Version information
__version__ = "1.0.0"
__author__ = "AIVillage Security Team"
__description__ = "Comprehensive security infrastructure for federated learning and distributed consensus"

# Security configuration constants
DEFAULT_SECURITY_CONFIG = SecurityConfiguration(
    enable_mfa=True,
    password_policy_enabled=True,
    session_timeout=3600,
    default_aggregation_method=AggregationMethod.HOMOMORPHIC,
    privacy_level=PrivacyLevel.HIGH,
    differential_privacy_epsilon=1.0,
    transport_security_level=SecurityLevel.HIGH,
    enable_forward_secrecy=True,
    channel_rotation_interval=3600,
    consensus_protocol=ConsensusProtocol.BYZANTINE,
    byzantine_threshold=0.33,
    enable_threshold_signatures=True,
    threat_detection_enabled=True,
    auto_mitigation_enabled=True,
    detection_sensitivity=0.7,
    initial_trust_score=0.5,
    trust_decay_enabled=True,
    min_trust_for_participation=0.4
)

def create_security_manager(node_id: str, config=None):
    """
    Factory function to create a configured SecurityIntegrationManager.
    
    Args:
        node_id: Unique identifier for this node
        config: Optional SecurityConfiguration (uses defaults if None)
        
    Returns:
        SecurityIntegrationManager instance
    """
    return SecurityIntegrationManager(
        node_id=node_id,
        config=config or DEFAULT_SECURITY_CONFIG
    )