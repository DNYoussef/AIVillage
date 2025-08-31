"""
TEE (Trusted Execution Environment) Security Framework

Comprehensive TEE integration for constitutional fog computing with:
- Hardware attestation (Intel SGX, AMD SEV-SNP, ARM TrustZone)
- Secure enclave management and lifecycle
- Constitutional policy enforcement
- Hardware-backed trust verification
- Multi-tier security (Bronze/Silver/Gold)

This package provides the critical security foundation for constitutional AI
workload execution in distributed fog computing environments.
"""

from .attestation import (
    TEEType,
    ConstitutionalTier,
    AttestationStatus,
    HardwareCapability,
    TEEQuote,
    AttestationResult,
    ConstitutionalPolicy as AttestationPolicy,
    TEEAttestationManager,
    get_attestation_manager,
    attest_fog_node,
    validate_constitutional_deployment,
    get_trusted_nodes_for_tier
)

from .enclave_manager import (
    EnclaveStatus,
    WorkloadType,
    SecurityLevel,
    EnclaveConfiguration,
    WorkloadManifest,
    EnclaveInstance,
    TEEEnclaveManager,
    get_enclave_manager,
    execute_constitutional_inference,
    create_constitutional_training_enclave
)

from ..constitutional.security_policy import (
    HarmCategory,
    RiskLevel,
    ResponseAction,
    ConstitutionalPrinciple,
    SafetyConstraint,
    PolicyViolation,
    ConstitutionalPolicy as SecurityPolicy,
    ConstitutionalPolicyEngine,
    get_policy_engine,
    evaluate_constitutional_content,
    validate_constitutional_workload
)

__version__ = "1.0.0"

__all__ = [
    # Attestation
    "TEEType",
    "ConstitutionalTier", 
    "AttestationStatus",
    "HardwareCapability",
    "TEEQuote",
    "AttestationResult",
    "AttestationPolicy",
    "TEEAttestationManager",
    "get_attestation_manager",
    "attest_fog_node",
    "validate_constitutional_deployment",
    "get_trusted_nodes_for_tier",
    
    # Enclave Management
    "EnclaveStatus",
    "WorkloadType",
    "SecurityLevel",
    "EnclaveConfiguration",
    "WorkloadManifest", 
    "EnclaveInstance",
    "TEEEnclaveManager",
    "get_enclave_manager",
    "execute_constitutional_inference",
    "create_constitutional_training_enclave",
    
    # Constitutional Policy
    "HarmCategory",
    "RiskLevel",
    "ResponseAction",
    "ConstitutionalPrinciple",
    "SafetyConstraint",
    "PolicyViolation",
    "SecurityPolicy",
    "ConstitutionalPolicyEngine",
    "get_policy_engine",
    "evaluate_constitutional_content",
    "validate_constitutional_workload"
]