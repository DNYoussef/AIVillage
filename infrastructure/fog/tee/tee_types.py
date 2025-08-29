"""
TEE Types and Data Structures

Defines core types and data structures for the TEE runtime system.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Union, Callable
from uuid import uuid4


class TEEType(Enum):
    """Supported TEE technologies"""

    AMD_SEV_SNP = "amd_sev_snp"
    INTEL_TDX = "intel_tdx"
    INTEL_SGX = "intel_sgx"
    ARM_TRUSTZONE = "arm_trustzone"
    SOFTWARE_ISOLATION = "software_isolation"  # gVisor/Firecracker fallback


class EnclaveState(Enum):
    """Enclave lifecycle states"""

    INITIALIZING = "initializing"
    CREATED = "created"
    MEASURED = "measured"
    ATTESTED = "attested"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


class AttestationStatus(Enum):
    """Attestation verification status"""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class TEECapability:
    """TEE hardware capabilities"""

    tee_type: TEEType
    available: bool
    version: str
    max_memory_mb: int
    max_enclaves: int
    supports_migration: bool = False
    supports_remote_attestation: bool = True
    supports_sealed_storage: bool = False
    vendor_extensions: dict[str, Any] = field(default_factory=dict)

    # Hardware-specific features
    secure_boot: bool = False
    memory_encryption: bool = False
    io_protection: bool = False
    debug_disabled: bool = True

    @property
    def is_hardware_tee(self) -> bool:
        """Check if this is a hardware-based TEE"""
        return self.tee_type != TEEType.SOFTWARE_ISOLATION


@dataclass
class TEEConfiguration:
    """TEE configuration parameters"""

    preferred_tee_type: TEEType | None = None
    memory_mb: int = 512
    cpu_cores: int = 1
    allow_debug: bool = False
    require_attestation: bool = True
    attestation_timeout_seconds: int = 300

    # Security policies
    measurement_policy: str = "strict"  # strict, permissive
    allowed_measurements: list[str] = field(default_factory=list)
    required_features: list[str] = field(default_factory=list)

    # Networking configuration
    network_isolation: bool = True
    allowed_outbound_hosts: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate configuration and return any errors"""
        errors = []

        if self.memory_mb < 64:
            errors.append("Memory must be at least 64MB")
        if self.memory_mb > 16384:
            errors.append("Memory cannot exceed 16GB")
        if self.cpu_cores < 1 or self.cpu_cores > 32:
            errors.append("CPU cores must be between 1 and 32")
        if self.attestation_timeout_seconds < 30:
            errors.append("Attestation timeout must be at least 30 seconds")

        return errors


@dataclass
class Measurement:
    """TEE measurement information"""

    measurement_type: str  # "pcr", "mrenclave", "mrtd", etc.
    index: int
    value: str  # hex-encoded hash
    algorithm: str = "sha256"
    description: str = ""

    def __post_init__(self):
        # Validate hex encoding
        try:
            bytes.fromhex(self.value)
        except ValueError:
            raise ValueError(f"Invalid hex value for measurement: {self.value}")


@dataclass
class AttestationReport:
    """TEE attestation report"""

    report_id: str = field(default_factory=lambda: f"att_{uuid4().hex[:16]}")
    enclave_id: str = ""
    tee_type: TEEType = TEEType.SOFTWARE_ISOLATION
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Core attestation data
    measurements: list[Measurement] = field(default_factory=list)
    quote: bytes = b""  # Hardware quote/signature
    certificate_chain: list[bytes] = field(default_factory=list)

    # Report metadata
    report_version: str = "1.0"
    nonce: bytes = field(default_factory=lambda: uuid4().bytes)

    # Verification status
    status: AttestationStatus = AttestationStatus.PENDING
    verification_errors: list[str] = field(default_factory=list)
    verified_at: datetime | None = None
    expires_at: datetime | None = None

    def add_measurement(
        self, measurement_type: str, index: int, value: str, algorithm: str = "sha256", description: str = ""
    ):
        """Add a measurement to the report"""
        measurement = Measurement(
            measurement_type=measurement_type, index=index, value=value, algorithm=algorithm, description=description
        )
        self.measurements.append(measurement)

    def get_measurement(self, measurement_type: str, index: int) -> Measurement | None:
        """Get a specific measurement by type and index"""
        for measurement in self.measurements:
            if measurement.measurement_type == measurement_type and measurement.index == index:
                return measurement
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "report_id": self.report_id,
            "enclave_id": self.enclave_id,
            "tee_type": self.tee_type.value,
            "timestamp": self.timestamp.isoformat(),
            "measurements": [
                {
                    "type": m.measurement_type,
                    "index": m.index,
                    "value": m.value,
                    "algorithm": m.algorithm,
                    "description": m.description,
                }
                for m in self.measurements
            ],
            "quote": self.quote.hex() if self.quote else "",
            "certificate_chain": [cert.hex() for cert in self.certificate_chain],
            "report_version": self.report_version,
            "nonce": self.nonce.hex(),
            "status": self.status.value,
            "verification_errors": self.verification_errors,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class EnclaveMetrics:
    """Enclave performance and security metrics"""

    enclave_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Performance metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: int = 0
    memory_limit_mb: int = 512
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0

    # Security metrics
    measurement_violations: int = 0
    attestation_failures: int = 0
    policy_violations: int = 0
    debug_events: int = 0

    # Execution metrics
    tasks_executed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time_ms: float = 0.0
    uptime_seconds: int = 0

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage"""
        if self.memory_limit_mb == 0:
            return 0.0
        return min(100.0, (self.memory_usage_mb / self.memory_limit_mb) * 100.0)

    @property
    def task_success_rate(self) -> float:
        """Calculate task success rate"""
        if self.tasks_executed == 0:
            return 100.0
        return (self.tasks_completed / self.tasks_executed) * 100.0


@dataclass
class EnclaveSpec:
    """Enclave specification for creation"""

    enclave_id: str = field(default_factory=lambda: f"enclave_{uuid4().hex[:12]}")
    name: str = ""
    description: str = ""

    # Resource requirements
    memory_mb: int = 512
    cpu_cores: int = 1
    storage_mb: int = 0

    # Code and data
    code_hash: str = ""  # Hash of code to be loaded
    initial_data: bytes = b""
    environment_variables: dict[str, str] = field(default_factory=dict)

    # Security configuration
    config: TEEConfiguration = field(default_factory=TEEConfiguration)

    # Expected measurements (for attestation validation)
    expected_measurements: list[Measurement] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate enclave specification"""
        errors = []

        if not self.name:
            errors.append("Enclave name is required")
        if not self.code_hash:
            errors.append("Code hash is required")
        if self.memory_mb < 64:
            errors.append("Memory must be at least 64MB")
        if self.cpu_cores < 1:
            errors.append("At least 1 CPU core required")

        # Validate configuration
        errors.extend(self.config.validate())

        return errors


@dataclass
class EnclaveContext:
    """Enclave execution context and state"""

    spec: EnclaveSpec
    state: EnclaveState = EnclaveState.INITIALIZING
    tee_type: TEEType = TEEType.SOFTWARE_ISOLATION

    # Runtime information
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    terminated_at: datetime | None = None

    # Attestation information
    attestation_report: AttestationReport | None = None
    last_attestation: datetime | None = None

    # Metrics and monitoring
    metrics: EnclaveMetrics = field(init=False)

    # Runtime handles (platform-specific)
    runtime_handles: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metrics = EnclaveMetrics(enclave_id=self.spec.enclave_id)

    @property
    def is_running(self) -> bool:
        """Check if enclave is in running state"""
        return self.state == EnclaveState.RUNNING

    @property
    def is_attested(self) -> bool:
        """Check if enclave has valid attestation"""
        return (
            self.attestation_report is not None
            and self.attestation_report.status == AttestationStatus.VERIFIED
            and self.state in [EnclaveState.ATTESTED, EnclaveState.RUNNING]
        )

    @property
    def uptime_seconds(self) -> int:
        """Get enclave uptime in seconds"""
        if not self.started_at:
            return 0

        end_time = self.terminated_at or datetime.now(UTC)
        return int((end_time - self.started_at).total_seconds())


# Type aliases for convenience
TEEHandle = Union[str, int, Any]  # Platform-specific TEE handle
AttestationCallback = Callable[[AttestationReport], None]
