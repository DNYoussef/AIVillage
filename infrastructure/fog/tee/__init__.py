"""
Trusted Execution Environment (TEE) Module

Provides comprehensive TEE support for confidential computing in fog infrastructure:
- Hardware-based security enclaves (AMD SEV-SNP, Intel TDX)
- Remote attestation and measurement verification
- Enclave lifecycle management
- Software isolation fallback (gVisor/Firecracker)
"""

from .attestation_service import AttestationService
from .enclave_executor import EnclaveExecutor
from .tee_runtime_manager import TEERuntimeManager
from .tee_types import (
    AttestationReport,
    EnclaveMetrics,
    EnclaveState,
    TEECapability,
    TEEConfiguration,
    TEEType,
)

__all__ = [
    "TEERuntimeManager",
    "AttestationService",
    "EnclaveExecutor",
    "TEEType",
    "TEECapability",
    "EnclaveState",
    "AttestationReport",
    "TEEConfiguration",
    "EnclaveMetrics",
]
