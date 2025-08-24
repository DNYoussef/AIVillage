"""
Shared Types for Edge Computing Infrastructure

Common data types, enums, and data structures used across the edge computing ecosystem
including Digital Twin, Device Management, Knowledge Systems, and Communication layers.

This module provides the foundational types that enable seamless integration between
all edge computing components while maintaining type safety and clear contracts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

# ============================================================================
# CORE ENUMS
# ============================================================================

class DataSource(Enum):
    """Sources of data in the edge computing ecosystem"""
    CONVERSATION = "conversation"
    PURCHASE = "purchase"
    LOCATION = "location"
    APP_USAGE = "app_usage"
    CALENDAR = "calendar"
    VOICE = "voice"
    SENSOR_DATA = "sensor_data"
    SYSTEM_METRICS = "system_metrics"
    NETWORK_ACTIVITY = "network_activity"
    USER_INTERACTION = "user_interaction"


class PrivacyLevel(Enum):
    """Privacy levels for data classification"""
    PUBLIC = "public"                    # Safe for global sharing
    PRIVATE = "private"                  # Personal but shareable anonymously
    PERSONAL = "personal"                # User-specific, keep local
    SENSITIVE = "sensitive"              # Highly sensitive, never share
    CONFIDENTIAL = "confidential"        # Top secret, encrypted storage only


class ProcessingMode(Enum):
    """Processing modes for edge computing tasks"""
    LOCAL_ONLY = "local_only"            # Process entirely on device
    REMOTE_PREFERRED = "remote_preferred" # Try remote first, fallback local
    REMOTE_ONLY = "remote_only"          # Must process remotely
    HYBRID = "hybrid"                    # Intelligent distribution
    DISTRIBUTED = "distributed"          # Spread across multiple nodes


class DeviceType(Enum):
    """Types of edge computing devices"""
    MOBILE_PHONE = "mobile_phone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    IOT_DEVICE = "iot_device"
    EDGE_SERVER = "edge_server"
    RASPBERRY_PI = "raspberry_pi"
    EMBEDDED_SYSTEM = "embedded_system"
    SMART_WATCH = "smart_watch"
    AR_VR_HEADSET = "ar_vr_headset"
    SMART_SPEAKER = "smart_speaker"
    VEHICLE_SYSTEM = "vehicle_system"


class TaskPriority(Enum):
    """Priority levels for edge computing tasks"""
    CRITICAL = "critical"     # Emergency/safety critical
    HIGH = "high"            # Important but not critical
    NORMAL = "normal"        # Standard priority
    LOW = "low"             # Low priority, can be delayed
    BACKGROUND = "background" # Run when resources available


class ResourceConstraint(Enum):
    """Resource constraint levels for adaptive processing"""
    SEVERE = "severe"         # Very limited resources
    MODERATE = "moderate"     # Some limitations
    MINIMAL = "minimal"       # Few limitations
    UNCONSTRAINED = "unconstrained"  # No significant limitations


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class EdgeResourceMetrics:
    """Comprehensive resource utilization metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_cores: int = 1
    cpu_frequency_mhz: float = 1000.0
    cpu_temperature_celsius: float | None = None
    
    # Memory metrics
    ram_used_mb: float = 0.0
    ram_total_mb: float = 1024.0
    ram_available_mb: float = 1024.0
    swap_used_mb: float = 0.0
    
    # Storage metrics
    storage_used_gb: float = 0.0
    storage_total_gb: float = 16.0
    storage_io_read_mbps: float = 0.0
    storage_io_write_mbps: float = 0.0
    
    # Network metrics
    network_rx_mbps: float = 0.0
    network_tx_mbps: float = 0.0
    network_latency_ms: float = 0.0
    network_packet_loss: float = 0.0
    
    # Power metrics
    battery_percent: float | None = None
    battery_charging: bool = False
    battery_time_remaining_minutes: int | None = None
    power_consumption_watts: float = 5.0
    
    # GPU metrics (if available)
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature_celsius: float | None = None
    
    @property
    def ram_percent(self) -> float:
        """Calculate RAM usage percentage"""
        if self.ram_total_mb > 0:
            return (self.ram_used_mb / self.ram_total_mb) * 100
        return 0.0
    
    @property
    def storage_percent(self) -> float:
        """Calculate storage usage percentage"""
        if self.storage_total_gb > 0:
            return (self.storage_used_gb / self.storage_total_gb) * 100
        return 0.0
    
    @property
    def is_resource_constrained(self) -> ResourceConstraint:
        """Determine current resource constraint level"""
        cpu_stressed = self.cpu_percent > 80
        memory_stressed = self.ram_percent > 80
        storage_stressed = self.storage_percent > 90
        battery_low = self.battery_percent is not None and self.battery_percent < 20
        
        stress_count = sum([cpu_stressed, memory_stressed, storage_stressed, battery_low])
        
        if stress_count >= 3:
            return ResourceConstraint.SEVERE
        elif stress_count >= 2:
            return ResourceConstraint.MODERATE
        elif stress_count >= 1:
            return ResourceConstraint.MINIMAL
        else:
            return ResourceConstraint.UNCONSTRAINED


@dataclass
class EdgeCapabilities:
    """Capabilities available on an edge device"""
    device_id: str
    device_type: DeviceType
    
    # Computing capabilities
    supports_ml_inference: bool = False
    supports_ml_training: bool = False
    supports_parallel_processing: bool = False
    supports_gpu_acceleration: bool = False
    supports_vector_operations: bool = False
    
    # Communication capabilities
    supports_ble: bool = False
    supports_wifi: bool = True
    supports_cellular: bool = False
    supports_ethernet: bool = False
    supports_p2p_mesh: bool = False
    
    # Storage capabilities
    supports_local_storage: bool = True
    supports_encrypted_storage: bool = False
    supports_database: bool = False
    max_storage_gb: float = 16.0
    
    # Sensor capabilities
    has_camera: bool = False
    has_microphone: bool = False
    has_gps: bool = False
    has_accelerometer: bool = False
    has_gyroscope: bool = False
    has_magnetometer: bool = False
    has_proximity_sensor: bool = False
    has_light_sensor: bool = False
    has_pressure_sensor: bool = False
    
    # Platform capabilities
    supports_background_processing: bool = False
    supports_push_notifications: bool = False
    supports_biometric_auth: bool = False
    supports_secure_enclave: bool = False
    
    # Performance characteristics
    max_concurrent_tasks: int = 2
    preferred_batch_size: int = 1
    thermal_throttling_available: bool = False
    
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class MobileDeviceProfile:
    """Comprehensive mobile device profile for optimization and adaptation"""
    device_id: str = "unknown"
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    # Battery metrics
    battery_percent: float = 100.0
    battery_charging: bool = False
    battery_health: float | None = None
    
    # Performance metrics
    cpu_temp_celsius: float = 25.0
    cpu_percent: float = 10.0
    cpu_cores: int = 4
    
    # Memory metrics
    ram_used_mb: float = 1000.0
    ram_available_mb: float = 3000.0
    ram_total_mb: float = 4000.0
    
    # Storage metrics
    storage_used_gb: float = 32.0
    storage_available_gb: float = 96.0
    storage_total_gb: float = 128.0
    
    # Network connectivity
    network_type: str = "wifi"
    network_quality: str = "good"
    is_connected: bool = True
    
    # Platform information
    platform: str = "android"
    os_version: str = "unknown"
    device_model: str = "unknown"
    
    # Performance constraints
    thermal_throttling: bool = False
    low_power_mode: bool = False
    background_processing_allowed: bool = True
    
    # App-specific metrics
    app_memory_limit_mb: float = 500.0
    app_cpu_limit_percent: float = 50.0
    
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EdgeTask:
    """Task definition for edge computing"""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_name: str = "unnamed_task"
    task_type: str = "generic"
    
    # Task requirements
    required_capabilities: list[str] = field(default_factory=list)
    min_memory_mb: float = 100.0
    min_cpu_cores: int = 1
    min_storage_mb: float = 10.0
    requires_gpu: bool = False
    requires_network: bool = False
    
    # Task configuration
    priority: TaskPriority = TaskPriority.NORMAL
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    max_execution_time_seconds: int = 300
    max_retries: int = 3
    
    # Task data
    input_data: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    
    # Privacy and security
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    data_sources: list[DataSource] = field(default_factory=list)
    requires_encryption: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    submitted_by: str = "system"
    estimated_completion_time: datetime | None = None
    
    # State tracking
    assigned_device_id: str | None = None
    execution_start_time: datetime | None = None
    execution_end_time: datetime | None = None
    current_status: str = "pending"
    
    def is_complete(self) -> bool:
        """Check if task execution is complete"""
        return self.current_status in ["completed", "failed", "cancelled"]
    
    def get_execution_duration(self) -> float | None:
        """Get task execution duration in seconds"""
        if self.execution_start_time and self.execution_end_time:
            return (self.execution_end_time - self.execution_start_time).total_seconds()
        return None


@dataclass
class EdgeTaskResult:
    """Result from edge task execution"""
    task_id: str
    success: bool
    
    # Execution details
    execution_time_seconds: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    executed_on_device: str | None = None
    processing_mode_used: ProcessingMode = ProcessingMode.LOCAL_ONLY
    
    # Results
    result_data: dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    error_message: str | None = None
    error_code: str | None = None
    
    # Resource usage during execution
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    total_storage_mb: float = 0.0
    network_usage_mb: float = 0.0
    energy_consumed_joules: float = 0.0
    
    # Quality metrics
    accuracy: float | None = None
    confidence: float | None = None
    throughput: float | None = None
    latency_ms: float | None = None
    
    # Privacy and compliance
    privacy_preserved: bool = True
    data_anonymized: bool = False
    compliance_verified: bool = False


@dataclass
class EdgeMessage:
    """Message structure for edge device communication"""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    message_type: str = "generic"
    
    # Message routing
    sender_id: str = "unknown"
    recipient_id: str | None = None
    broadcast: bool = False
    
    # Message content
    payload: dict[str, Any] = field(default_factory=dict)
    binary_data: bytes | None = None
    
    # Message properties
    priority: TaskPriority = TaskPriority.NORMAL
    requires_response: bool = False
    response_timeout_seconds: int = 30
    ttl_seconds: int = 3600
    
    # Privacy and security
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    encrypted: bool = False
    signed: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    protocol_version: str = "1.0"
    content_type: str = "application/json"
    
    # Delivery tracking
    attempts: int = 0
    max_attempts: int = 3
    last_attempt: datetime | None = None
    delivery_confirmed: bool = False


@dataclass
class EdgeDeviceProfile:
    """Comprehensive edge device profile"""
    device_id: str
    device_name: str
    device_type: DeviceType
    
    # Hardware information
    capabilities: EdgeCapabilities
    current_metrics: EdgeResourceMetrics
    
    # Network information
    ip_address: str | None = None
    mac_address: str | None = None
    network_interfaces: list[str] = field(default_factory=list)
    
    # Software information
    os_name: str = "unknown"
    os_version: str = "unknown"
    platform_architecture: str = "unknown"
    app_version: str = "1.0.0"
    
    # Operational state
    online: bool = True
    last_seen: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    
    # Performance history
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    
    # Preferences and configuration
    power_management_enabled: bool = True
    background_processing_allowed: bool = True
    data_collection_consent: bool = False
    privacy_mode: str = "balanced"
    
    # Security information
    last_security_check: datetime | None = None
    security_compliance_level: str = "basic"
    encryption_supported: bool = False
    
    updated_at: datetime = field(default_factory=datetime.now)
    
    def is_healthy(self) -> bool:
        """Check if device is healthy and responsive"""
        return (
            self.online and 
            self.current_metrics.cpu_percent < 90 and
            self.current_metrics.ram_percent < 85 and
            self.success_rate > 0.8
        )
    
    def get_availability_score(self) -> float:
        """Calculate device availability score (0.0 to 1.0)"""
        if not self.online:
            return 0.0
        
        # Factor in resource usage, success rate, and responsiveness
        resource_score = 1.0 - (self.current_metrics.cpu_percent / 100)
        memory_score = 1.0 - (self.current_metrics.ram_percent / 100)
        performance_score = self.success_rate
        
        return (resource_score + memory_score + performance_score) / 3


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_edge_task(
    name: str,
    task_type: str = "generic",
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs
) -> EdgeTask:
    """Create a new edge task with sensible defaults"""
    return EdgeTask(
        task_name=name,
        task_type=task_type,
        priority=priority,
        **kwargs
    )


def create_edge_message(
    sender: str,
    recipient: str | None = None,
    message_type: str = "generic",
    payload: dict[str, Any] | None = None,
    **kwargs
) -> EdgeMessage:
    """Create a new edge message with sensible defaults"""
    return EdgeMessage(
        sender_id=sender,
        recipient_id=recipient,
        message_type=message_type,
        payload=payload or {},
        **kwargs
    )


def assess_privacy_level(content: str, context: dict[str, Any] = None) -> PrivacyLevel:
    """Assess privacy level of content"""
    context = context or {}
    content_lower = content.lower()
    
    # Check for sensitive indicators
    sensitive_patterns = [
        "password", "ssn", "social security", "credit card", "bank account",
        "medical", "health record", "diagnosis", "prescription", "biometric"
    ]
    
    personal_patterns = [
        "my name", "i am", "my address", "my phone", "my email",
        "personal", "private", "family", "home"
    ]
    
    public_patterns = [
        "public", "general", "common", "tutorial", "guide", "instruction"
    ]
    
    if any(pattern in content_lower for pattern in sensitive_patterns):
        return PrivacyLevel.SENSITIVE
    elif any(pattern in content_lower for pattern in personal_patterns):
        return PrivacyLevel.PERSONAL
    elif any(pattern in content_lower for pattern in public_patterns):
        return PrivacyLevel.PUBLIC
    else:
        return PrivacyLevel.PRIVATE


# Export all public types and functions
__all__ = [
    # Enums
    "DataSource", "PrivacyLevel", "ProcessingMode", "DeviceType", 
    "TaskPriority", "ResourceConstraint",
    
    # Data classes
    "EdgeResourceMetrics", "EdgeCapabilities", "EdgeTask", "EdgeTaskResult",
    "EdgeMessage", "EdgeDeviceProfile",
    
    # Utility functions
    "create_edge_task", "create_edge_message", "assess_privacy_level",
]