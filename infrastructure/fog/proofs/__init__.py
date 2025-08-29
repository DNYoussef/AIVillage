"""
Fog Computing Cryptographic Proof System

Comprehensive proof system for verifiable fog computing:
- Proof-of-Execution (PoE): Task completion verification
- Proof-of-Audit (PoA): AI auditor consensus verification
- Proof-of-SLA (PoSLA): Performance compliance verification
- Merkle tree aggregation for batch verification
- Cryptographic signature validation
- Tokenomics integration for proof rewards
- REST API for proof management

Key Features:
- Tamper-proof cryptographic proofs
- Efficient Merkle tree batch verification
- Automated reward distribution
- RESTful API interface
- Integration with fog infrastructure
"""

# Core proof system components
from .merkle_tree import MerkleProof, MerkleTree
from .proof_api import (
    AuditConsensusRequest,
    ProofAPIManager,
    ProofResponse,
    RewardResponse,
    SLAComplianceRequest,
    TaskExecutionRequest,
    VerificationResponse,
    create_proof_api,
)
from .proof_generator import (
    AuditEvidence,
    CryptographicProof,
    ProofGenerator,
    ProofOfAudit,
    ProofOfExecution,
    ProofOfSLA,
    ProofType,
    SLAMeasurement,
    TaskExecution,
)

# Integration components
from .proof_integration import (
    AuditProofRequest,
    ProofIntegrationError,
    ProofSystemIntegration,
    SLAProofRequest,
    TaskProofRequest,
)
from .proof_verifier import ProofVerifier, VerificationReport, VerificationResult
from .tokenomics_integration import ProofReward, ProofTokenomicsIntegration

# Version info
__version__ = "1.0.0"
__author__ = "AIVillage Fog Computing Team"

# Export all components
__all__ = [
    # Core proof system
    "ProofGenerator",
    "ProofVerifier",
    "MerkleTree",
    "MerkleProof",
    # Proof types and data structures
    "ProofType",
    "CryptographicProof",
    "ProofOfExecution",
    "ProofOfAudit",
    "ProofOfSLA",
    "TaskExecution",
    "AuditEvidence",
    "SLAMeasurement",
    "VerificationResult",
    "VerificationReport",
    # Integration
    "ProofSystemIntegration",
    "ProofTokenomicsIntegration",
    "ProofAPIManager",
    # Request/Response models
    "TaskProofRequest",
    "AuditProofRequest",
    "SLAProofRequest",
    "TaskExecutionRequest",
    "AuditConsensusRequest",
    "SLAComplianceRequest",
    "ProofResponse",
    "VerificationResponse",
    "RewardResponse",
    "ProofReward",
    # Factory functions
    "create_proof_api",
    # Exceptions
    "ProofIntegrationError",
    # Constants
    "__version__",
    "__author__",
]


def get_proof_system_info():
    """Get information about the proof system"""
    return {
        "version": __version__,
        "author": __author__,
        "components": {
            "proof_generator": "Generates PoE, PoA, and PoSLA proofs",
            "proof_verifier": "Verifies cryptographic proofs and signatures",
            "merkle_tree": "Efficient Merkle tree construction and verification",
            "integration": "Seamless integration with fog infrastructure",
            "tokenomics": "Automated reward calculation and distribution",
            "api": "RESTful API for proof management",
        },
        "proof_types": {
            "PoE": "Proof-of-Execution for task completion",
            "PoA": "Proof-of-Audit for AI auditor consensus",
            "PoSLA": "Proof-of-SLA for performance compliance",
            "Merkle Batch": "Batch aggregation using Merkle trees",
        },
        "features": [
            "Tamper-proof cryptographic proofs",
            "Efficient batch verification",
            "Automated tokenomics integration",
            "RESTful API interface",
            "Real-time monitoring",
            "Quality-based reward system",
        ],
    }
