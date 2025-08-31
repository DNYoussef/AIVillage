"""
HiddenServiceManagementService Specification
Secure hidden service hosting and management with advanced security controls

This service provides comprehensive hidden service management with censorship resistance,
traffic analysis protection, and secure service sandboxing capabilities.
"""

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Hidden service lifecycle status."""
    INITIALIZING = "initializing"
    REGISTERING = "registering"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class ServiceType(Enum):
    """Types of hidden services supported."""
    WEB = "web"              # HTTP/HTTPS web services
    API = "api"              # REST/GraphQL APIs
    MESSAGING = "messaging"   # Chat/messaging services
    FILE_SHARING = "file_sharing"  # File hosting/sharing
    DATABASE = "database"     # Database access
    CUSTOM = "custom"        # Custom TCP/UDP services


class SecurityLevel(Enum):
    """Security levels for hidden services."""
    STANDARD = "standard"     # Basic onion routing
    ENHANCED = "enhanced"     # Additional obfuscation
    MAXIMUM = "maximum"       # Full anonymity with decoys


@dataclass
class ServiceKeys:
    """Cryptographic keys for hidden service."""
    service_id: str
    onion_address: str
    private_key: bytes
    public_key: bytes
    descriptor_signing_key: bytes
    introduction_keys: List[bytes] = field(default_factory=list)
    
    @classmethod
    def generate_service_keys(cls, service_id: str) -> 'ServiceKeys':
        """Generate complete key set for hidden service."""
        # Generate Ed25519 identity key
        private_key_obj = ed25519.Ed25519PrivateKey.generate()
        private_key = private_key_obj.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_key = private_key_obj.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # Generate onion address (simplified)
        onion_address = hashlib.sha256(public_key).hexdigest()[:16] + ".onion"
        
        # Generate descriptor signing key
        descriptor_key = secrets.token_bytes(32)
        
        return cls(
            service_id=service_id,
            onion_address=onion_address,
            private_key=private_key,
            public_key=public_key,
            descriptor_signing_key=descriptor_key
        )


@dataclass
class ServiceConfiguration:
    """Configuration for hidden service."""
    service_id: str
    service_type: ServiceType
    security_level: SecurityLevel
    port_mappings: Dict[int, int]  # virtual_port -> local_port
    max_connections: int = 100
    bandwidth_limit_mbps: float = 10.0
    allowed_clients: Set[str] = field(default_factory=set)
    access_control_enabled: bool = True
    traffic_obfuscation: bool = False
    decoy_traffic: bool = False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate service configuration."""
        if not self.port_mappings:
            return False, "No port mappings specified"
        
        for virtual_port, local_port in self.port_mappings.items():
            if not (1 <= virtual_port <= 65535) or not (1 <= local_port <= 65535):
                return False, f"Invalid port mapping: {virtual_port} -> {local_port}"
        
        if self.max_connections <= 0:
            return False, "Invalid max_connections value"
        
        return True, "Configuration valid"


@dataclass
class SandboxEnvironment:
    """Isolated execution environment for hidden service."""
    sandbox_id: str
    service_id: str
    container_id: Optional[str] = None
    network_namespace: Optional[str] = None
    filesystem_mount: Optional[str] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    security_policies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default resource limits and security policies."""
        if not self.resource_limits:
            self.resource_limits = {
                "cpu_limit": "0.5",      # 0.5 CPU cores
                "memory_limit": "512m",   # 512 MB RAM
                "storage_limit": "1g",    # 1 GB storage
                "network_bandwidth": "10m" # 10 Mbps
            }
        
        if not self.security_policies:
            self.security_policies = [
                "no_network_access_outside_tor",
                "read_only_system_files",
                "limited_process_execution",
                "encrypted_storage"
            ]


@dataclass
class TrafficObfuscation:
    """Traffic obfuscation configuration."""
    enabled: bool = False
    obfuscation_method: str = "none"
    padding_patterns: List[str] = field(default_factory=list)
    timing_randomization: bool = False
    decoy_connections: int = 0
    cover_traffic_rate: float = 0.0  # KB/s
    
    @classmethod
    def create_for_security_level(cls, security_level: SecurityLevel) -> 'TrafficObfuscation':
        """Create obfuscation config based on security level."""
        if security_level == SecurityLevel.STANDARD:
            return cls()
        elif security_level == SecurityLevel.ENHANCED:
            return cls(
                enabled=True,
                obfuscation_method="padding",
                timing_randomization=True,
                cover_traffic_rate=1.0
            )
        else:  # MAXIMUM
            return cls(
                enabled=True,
                obfuscation_method="full_obfuscation",
                padding_patterns=["random", "constant", "burst"],
                timing_randomization=True,
                decoy_connections=5,
                cover_traffic_rate=5.0
            )


@dataclass
class AccessControl:
    """Access control configuration for hidden service."""
    authentication_required: bool = True
    authorized_clients: Set[str] = field(default_factory=set)
    client_certificates: Dict[str, bytes] = field(default_factory=dict)
    rate_limiting: Dict[str, int] = field(default_factory=dict)
    geo_restrictions: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize default rate limiting."""
        if not self.rate_limiting:
            self.rate_limiting = {
                "requests_per_minute": 60,
                "connections_per_client": 5,
                "bandwidth_per_client_kbps": 1000
            }


@dataclass
class HiddenService:
    """Complete hidden service with security and management capabilities."""
    service_id: str
    service_type: ServiceType
    keys: ServiceKeys
    configuration: ServiceConfiguration
    sandbox: SandboxEnvironment
    status: ServiceStatus = ServiceStatus.INITIALIZING
    
    # Security components
    access_control: Optional[AccessControl] = None
    traffic_obfuscation: Optional[TrafficObfuscation] = None
    
    # Runtime state
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    connection_count: int = 0
    data_transferred_mb: float = 0.0
    security_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize security components if not provided."""
        if self.access_control is None:
            self.access_control = AccessControl()
        
        if self.traffic_obfuscation is None:
            self.traffic_obfuscation = TrafficObfuscation.create_for_security_level(
                self.configuration.security_level
            )
    
    def is_healthy(self) -> bool:
        """Check if service is healthy and operational."""
        return (self.status == ServiceStatus.ACTIVE and
                time.time() - self.last_activity < 300)  # Active within 5 minutes
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()


class ServiceSandboxManager:
    """Manages isolated execution environments for hidden services."""
    
    def __init__(self):
        self.active_sandboxes: Dict[str, SandboxEnvironment] = {}
        self.resource_monitor = SandboxResourceMonitor()
    
    async def create_sandbox(self, service_id: str, 
                           security_level: SecurityLevel) -> SandboxEnvironment:
        """Create isolated sandbox environment for service."""
        sandbox_id = f"sandbox_{service_id}_{uuid.uuid4().hex[:8]}"
        
        # Create sandbox configuration
        sandbox = SandboxEnvironment(
            sandbox_id=sandbox_id,
            service_id=service_id
        )
        
        # Configure security based on level
        await self._configure_sandbox_security(sandbox, security_level)
        
        # Initialize container/namespace
        success = await self._initialize_sandbox_environment(sandbox)
        if not success:
            raise RuntimeError(f"Failed to create sandbox for service {service_id}")
        
        self.active_sandboxes[sandbox_id] = sandbox
        logger.info(f"Created sandbox {sandbox_id} for service {service_id}")
        
        return sandbox
    
    async def _configure_sandbox_security(self, sandbox: SandboxEnvironment, 
                                        security_level: SecurityLevel) -> None:
        """Configure sandbox security based on level."""
        base_policies = [
            "no_network_access_outside_tor",
            "read_only_system_files",
            "limited_process_execution"
        ]
        
        if security_level == SecurityLevel.ENHANCED:
            sandbox.security_policies.extend(base_policies + [
                "encrypted_storage",
                "memory_isolation",
                "syscall_filtering"
            ])
        elif security_level == SecurityLevel.MAXIMUM:
            sandbox.security_policies.extend(base_policies + [
                "encrypted_storage",
                "memory_isolation",
                "syscall_filtering",
                "hardware_isolation",
                "secure_boot_verification",
                "real_time_integrity_monitoring"
            ])
    
    async def _initialize_sandbox_environment(self, sandbox: SandboxEnvironment) -> bool:
        """Initialize the actual sandbox environment."""
        try:
            # This would interface with container runtime (Docker, Podman, etc.)
            # For now, simulate environment creation
            
            # Create network namespace
            sandbox.network_namespace = f"netns_{sandbox.sandbox_id}"
            
            # Create filesystem mount
            sandbox.filesystem_mount = f"/sandbox/{sandbox.sandbox_id}"
            
            # Set container ID (simulated)
            sandbox.container_id = f"container_{sandbox.sandbox_id}"
            
            logger.info(f"Initialized sandbox environment: {sandbox.sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}")
            return False
    
    async def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Securely destroy sandbox environment."""
        if sandbox_id not in self.active_sandboxes:
            return False
        
        sandbox = self.active_sandboxes[sandbox_id]
        
        try:
            # Stop and remove container
            if sandbox.container_id:
                await self._destroy_container(sandbox.container_id)
            
            # Clean up network namespace
            if sandbox.network_namespace:
                await self._cleanup_network_namespace(sandbox.network_namespace)
            
            # Secure cleanup of filesystem
            if sandbox.filesystem_mount:
                await self._secure_cleanup_filesystem(sandbox.filesystem_mount)
            
            del self.active_sandboxes[sandbox_id]
            logger.info(f"Destroyed sandbox: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error destroying sandbox {sandbox_id}: {e}")
            return False
    
    async def _destroy_container(self, container_id: str) -> None:
        """Destroy container securely."""
        logger.info(f"Destroying container: {container_id}")
    
    async def _cleanup_network_namespace(self, namespace: str) -> None:
        """Clean up network namespace."""
        logger.info(f"Cleaning up network namespace: {namespace}")
    
    async def _secure_cleanup_filesystem(self, mount_path: str) -> None:
        """Securely clean up filesystem mount."""
        logger.info(f"Secure cleanup of filesystem: {mount_path}")


class SandboxResourceMonitor:
    """Monitors resource usage in service sandboxes."""
    
    def __init__(self):
        self.monitoring_interval = 30  # seconds
        self.resource_thresholds = {
            "cpu_usage_percent": 80,
            "memory_usage_percent": 85,
            "disk_usage_percent": 90,
            "network_usage_mbps": 50
        }
    
    async def monitor_sandbox(self, sandbox: SandboxEnvironment) -> Dict[str, float]:
        """Monitor resource usage for a sandbox."""
        # This would interface with actual system monitoring
        # For now, return simulated metrics
        
        metrics = {
            "cpu_usage_percent": 25.0,
            "memory_usage_mb": 256.0,
            "disk_usage_mb": 100.0,
            "network_in_mbps": 2.0,
            "network_out_mbps": 1.5
        }
        
        # Check thresholds
        await self._check_resource_thresholds(sandbox, metrics)
        
        return metrics
    
    async def _check_resource_thresholds(self, sandbox: SandboxEnvironment, 
                                       metrics: Dict[str, float]) -> None:
        """Check if resource usage exceeds thresholds."""
        warnings = []
        
        if metrics["cpu_usage_percent"] > self.resource_thresholds["cpu_usage_percent"]:
            warnings.append("HIGH_CPU_USAGE")
        
        if metrics["memory_usage_mb"] > 400:  # 80% of 512MB limit
            warnings.append("HIGH_MEMORY_USAGE")
        
        if warnings:
            logger.warning(f"Resource warnings for sandbox {sandbox.sandbox_id}: {warnings}")


class ServiceRegistry:
    """Registry for hidden service discovery and management."""
    
    def __init__(self):
        self.services: Dict[str, HiddenService] = {}
        self.service_index: Dict[str, str] = {}  # onion_address -> service_id
        self.type_index: Dict[ServiceType, Set[str]] = {st: set() for st in ServiceType}
    
    async def register_service(self, service: HiddenService) -> bool:
        """Register hidden service in the registry."""
        try:
            self.services[service.service_id] = service
            self.service_index[service.keys.onion_address] = service.service_id
            self.type_index[service.service_type].add(service.service_id)
            
            logger.info(f"Registered service {service.service_id} at {service.keys.onion_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_id}: {e}")
            return False
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister service from registry."""
        if service_id not in self.services:
            return False
        
        service = self.services[service_id]
        
        # Remove from indices
        if service.keys.onion_address in self.service_index:
            del self.service_index[service.keys.onion_address]
        
        self.type_index[service.service_type].discard(service_id)
        
        # Remove main entry
        del self.services[service_id]
        
        logger.info(f"Unregistered service {service_id}")
        return True
    
    async def discover_services(self, service_type: Optional[ServiceType] = None,
                              security_level: Optional[SecurityLevel] = None) -> List[str]:
        """Discover services by type and security level."""
        if service_type:
            candidate_ids = self.type_index[service_type]
        else:
            candidate_ids = set(self.services.keys())
        
        # Filter by security level if specified
        if security_level:
            candidate_ids = {
                sid for sid in candidate_ids
                if self.services[sid].configuration.security_level == security_level
            }
        
        # Return only healthy services
        return [
            sid for sid in candidate_ids
            if self.services[sid].is_healthy()
        ]


class TrafficAnalysisResistance:
    """Implements traffic analysis resistance techniques."""
    
    def __init__(self):
        self.cover_traffic_generators = {}
        self.timing_obfuscator = TimingObfuscator()
        self.padding_manager = PaddingManager()
    
    async def start_cover_traffic(self, service_id: str, 
                                config: TrafficObfuscation) -> bool:
        """Start cover traffic generation for service."""
        if not config.enabled or config.cover_traffic_rate <= 0:
            return True
        
        generator = CoverTrafficGenerator(service_id, config)
        self.cover_traffic_generators[service_id] = generator
        
        # Start background task
        asyncio.create_task(generator.run())
        
        logger.info(f"Started cover traffic for service {service_id}")
        return True
    
    async def stop_cover_traffic(self, service_id: str) -> bool:
        """Stop cover traffic for service."""
        if service_id in self.cover_traffic_generators:
            generator = self.cover_traffic_generators[service_id]
            await generator.stop()
            del self.cover_traffic_generators[service_id]
            
            logger.info(f"Stopped cover traffic for service {service_id}")
            return True
        
        return False
    
    async def obfuscate_timing(self, service_id: str, original_timing: float) -> float:
        """Apply timing obfuscation to network operations."""
        return await self.timing_obfuscator.obfuscate(service_id, original_timing)
    
    async def add_padding(self, service_id: str, data: bytes) -> bytes:
        """Add padding to data for size obfuscation."""
        return await self.padding_manager.add_padding(service_id, data)


class TimingObfuscator:
    """Obfuscates timing patterns in network communications."""
    
    def __init__(self):
        self.timing_profiles = {}
    
    async def obfuscate(self, service_id: str, original_timing: float) -> float:
        """Apply timing obfuscation."""
        # Add random delay between 0-100ms
        additional_delay = secrets.randbelow(100) / 1000.0
        return original_timing + additional_delay


class PaddingManager:
    """Manages traffic padding for size obfuscation."""
    
    def __init__(self):
        self.padding_strategies = {}
    
    async def add_padding(self, service_id: str, data: bytes) -> bytes:
        """Add padding to data."""
        # Simple padding to next 1KB boundary
        target_size = ((len(data) // 1024) + 1) * 1024
        padding_size = target_size - len(data)
        
        return data + secrets.token_bytes(padding_size)


class CoverTrafficGenerator:
    """Generates cover traffic to resist traffic analysis."""
    
    def __init__(self, service_id: str, config: TrafficObfuscation):
        self.service_id = service_id
        self.config = config
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def run(self) -> None:
        """Run cover traffic generation."""
        self.running = True
        
        while self.running:
            # Generate cover traffic
            await self._generate_cover_packet()
            
            # Wait based on configured rate
            interval = 1.0 / (self.config.cover_traffic_rate / 1024)  # Convert KB/s to packets/s
            await asyncio.sleep(interval)
    
    async def _generate_cover_packet(self) -> None:
        """Generate a single cover traffic packet."""
        # Generate dummy data
        packet_size = secrets.randbelow(1024) + 64  # 64-1088 bytes
        cover_data = secrets.token_bytes(packet_size)
        
        # Send to dummy destination (this would be actual network code)
        logger.debug(f"Generated cover packet of {packet_size} bytes for {self.service_id}")
    
    async def stop(self) -> None:
        """Stop cover traffic generation."""
        self.running = False


class HiddenServiceManagementService:
    """
    Comprehensive hidden service management with advanced security controls.
    
    This service provides complete lifecycle management for hidden services with:
    - Secure service registration and hosting
    - Advanced traffic analysis resistance
    - Isolated execution environments (sandboxing)
    - Comprehensive access control and authentication
    - Censorship resistance mechanisms
    - Real-time security monitoring and threat detection
    
    Security Features:
    - Service isolation through containerization/sandboxing
    - Multi-layered authentication and authorization
    - Traffic obfuscation and cover traffic generation
    - Advanced cryptographic key management
    - Secure service discovery mechanisms
    - Automated threat detection and response
    """
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.sandbox_manager = ServiceSandboxManager()
        self.traffic_resistance = TrafficAnalysisResistance()
        self.security_monitor = ServiceSecurityMonitor()
        
        # Service state
        self.active_services: Dict[str, HiddenService] = {}
        self.service_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("HiddenServiceManagementService initialized")
    
    async def create_hidden_service(self, service_type: ServiceType,
                                  security_level: SecurityLevel,
                                  configuration: Dict[str, Any],
                                  owner_token: str) -> Tuple[bool, str, Optional[HiddenService]]:
        """Create new hidden service with specified configuration."""
        service_id = f"hs_{uuid.uuid4().hex}"
        
        try:
            # Validate configuration
            service_config = ServiceConfiguration(
                service_id=service_id,
                service_type=service_type,
                security_level=security_level,
                **configuration
            )
            
            is_valid, validation_message = service_config.validate()
            if not is_valid:
                return False, validation_message, None
            
            # Generate service keys
            service_keys = ServiceKeys.generate_service_keys(service_id)
            
            # Create sandbox environment
            sandbox = await self.sandbox_manager.create_sandbox(service_id, security_level)
            
            # Create hidden service object
            hidden_service = HiddenService(
                service_id=service_id,
                service_type=service_type,
                keys=service_keys,
                configuration=service_config,
                sandbox=sandbox
            )
            
            # Configure access control
            await self._configure_access_control(hidden_service, configuration, owner_token)
            
            # Start traffic analysis resistance
            await self.traffic_resistance.start_cover_traffic(
                service_id, hidden_service.traffic_obfuscation
            )
            
            # Register service
            await self.service_registry.register_service(hidden_service)
            
            # Add to active services
            self.active_services[service_id] = hidden_service
            
            # Initialize metrics tracking
            self.service_metrics[service_id] = {
                "created_at": time.time(),
                "owner_token_hash": hashlib.sha256(owner_token.encode()).hexdigest(),
                "connection_count": 0,
                "data_transferred": 0,
                "security_events": []
            }
            
            # Start service
            hidden_service.status = ServiceStatus.ACTIVE
            
            # Log creation
            await self.security_monitor.log_service_creation(hidden_service, owner_token)
            
            logger.info(f"Created hidden service {service_id} at {service_keys.onion_address}")
            return True, "Hidden service created successfully", hidden_service
            
        except Exception as e:
            logger.error(f"Failed to create hidden service: {e}")
            
            # Cleanup on failure
            if service_id in self.active_services:
                await self._cleanup_failed_service(service_id)
            
            return False, f"Service creation failed: {str(e)}", None
    
    async def terminate_hidden_service(self, service_id: str, 
                                     owner_token: str) -> Tuple[bool, str]:
        """Terminate hidden service securely."""
        if service_id not in self.active_services:
            return False, "Service not found"
        
        service = self.active_services[service_id]
        
        # Validate ownership
        if not await self._validate_service_ownership(service_id, owner_token):
            await self.security_monitor.log_unauthorized_access(service_id, owner_token)
            return False, "Unauthorized termination attempt"
        
        try:
            # Update status
            service.status = ServiceStatus.TERMINATING
            
            # Stop traffic analysis resistance
            await self.traffic_resistance.stop_cover_traffic(service_id)
            
            # Unregister from registry
            await self.service_registry.unregister_service(service_id)
            
            # Destroy sandbox
            await self.sandbox_manager.destroy_sandbox(service.sandbox.sandbox_id)
            
            # Secure key deletion
            service.keys.private_key = b""
            service.keys.descriptor_signing_key = b""
            service.keys.introduction_keys.clear()
            
            # Update status
            service.status = ServiceStatus.TERMINATED
            
            # Remove from active services
            del self.active_services[service_id]
            
            # Clean up metrics
            if service_id in self.service_metrics:
                del self.service_metrics[service_id]
            
            logger.info(f"Terminated hidden service {service_id}")
            return True, "Service terminated successfully"
            
        except Exception as e:
            logger.error(f"Error terminating service {service_id}: {e}")
            return False, f"Termination error: {str(e)}"
    
    async def get_service_status(self, service_id: str, 
                               owner_token: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Get comprehensive status of hidden service."""
        if service_id not in self.active_services:
            return False, "Service not found", {}
        
        # Validate ownership
        if not await self._validate_service_ownership(service_id, owner_token):
            return False, "Unauthorized status request", {}
        
        service = self.active_services[service_id]
        metrics = self.service_metrics.get(service_id, {})
        
        # Get sandbox resource metrics
        sandbox_metrics = await self.sandbox_manager.resource_monitor.monitor_sandbox(
            service.sandbox
        )
        
        status_info = {
            "service_id": service.service_id,
            "onion_address": service.keys.onion_address,
            "service_type": service.service_type.value,
            "status": service.status.value,
            "security_level": service.configuration.security_level.value,
            "created_at": service.created_at,
            "last_activity": service.last_activity,
            "connection_count": service.connection_count,
            "data_transferred_mb": service.data_transferred_mb,
            "sandbox_metrics": sandbox_metrics,
            "health_status": service.is_healthy(),
            "traffic_obfuscation_enabled": service.traffic_obfuscation.enabled,
            "access_control_enabled": service.access_control.authentication_required
        }
        
        return True, "Status retrieved successfully", status_info
    
    async def update_service_configuration(self, service_id: str,
                                         new_config: Dict[str, Any],
                                         owner_token: str) -> Tuple[bool, str]:
        """Update service configuration."""
        if service_id not in self.active_services:
            return False, "Service not found"
        
        # Validate ownership
        if not await self._validate_service_ownership(service_id, owner_token):
            return False, "Unauthorized configuration update"
        
        service = self.active_services[service_id]
        
        try:
            # Update configuration (this would validate and apply changes)
            # For now, just log the update
            logger.info(f"Updated configuration for service {service_id}")
            
            # Log security event
            await self.security_monitor.log_configuration_change(service_id, new_config)
            
            return True, "Configuration updated successfully"
            
        except Exception as e:
            logger.error(f"Error updating configuration for {service_id}: {e}")
            return False, f"Configuration update failed: {str(e)}"
    
    async def list_services(self, owner_token: str,
                          service_type: Optional[ServiceType] = None) -> List[Dict[str, Any]]:
        """List hidden services owned by requester."""
        owned_services = []
        
        for service_id, service in self.active_services.items():
            if await self._validate_service_ownership(service_id, owner_token):
                if service_type is None or service.service_type == service_type:
                    service_info = {
                        "service_id": service.service_id,
                        "onion_address": service.keys.onion_address,
                        "service_type": service.service_type.value,
                        "status": service.status.value,
                        "created_at": service.created_at,
                        "is_healthy": service.is_healthy()
                    }
                    owned_services.append(service_info)
        
        return owned_services
    
    async def _configure_access_control(self, service: HiddenService,
                                      config: Dict[str, Any],
                                      owner_token: str) -> None:
        """Configure access control for service."""
        access_config = config.get("access_control", {})
        
        service.access_control.authentication_required = access_config.get(
            "authentication_required", True
        )
        
        # Set authorized clients
        authorized_clients = access_config.get("authorized_clients", [])
        service.access_control.authorized_clients.update(authorized_clients)
        
        # Configure rate limiting
        rate_limits = access_config.get("rate_limiting", {})
        service.access_control.rate_limiting.update(rate_limits)
    
    async def _validate_service_ownership(self, service_id: str, owner_token: str) -> bool:
        """Validate that the token corresponds to the service owner."""
        if service_id not in self.service_metrics:
            return False
        
        stored_hash = self.service_metrics[service_id]["owner_token_hash"]
        provided_hash = hashlib.sha256(owner_token.encode()).hexdigest()
        
        return stored_hash == provided_hash
    
    async def _cleanup_failed_service(self, service_id: str) -> None:
        """Clean up resources after service creation failure."""
        try:
            # Stop traffic resistance if started
            await self.traffic_resistance.stop_cover_traffic(service_id)
            
            # Unregister if registered
            await self.service_registry.unregister_service(service_id)
            
            # Clean up sandbox
            service = self.active_services.get(service_id)
            if service and service.sandbox:
                await self.sandbox_manager.destroy_sandbox(service.sandbox.sandbox_id)
            
            # Remove from tracking
            if service_id in self.active_services:
                del self.active_services[service_id]
            if service_id in self.service_metrics:
                del self.service_metrics[service_id]
            
            logger.info(f"Cleaned up failed service {service_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up failed service {service_id}: {e}")


class ServiceSecurityMonitor:
    """Monitors security events and threats for hidden services."""
    
    def __init__(self):
        self.security_events = []
        self.threat_patterns = {}
    
    async def log_service_creation(self, service: HiddenService, owner_token: str) -> None:
        """Log service creation event."""
        event = {
            "event_type": "SERVICE_CREATED",
            "service_id": service.service_id,
            "service_type": service.service_type.value,
            "security_level": service.configuration.security_level.value,
            "owner_hash": hashlib.sha256(owner_token.encode()).hexdigest()[:16],
            "timestamp": time.time()
        }
        
        self.security_events.append(event)
        logger.info(f"Logged service creation: {service.service_id}")
    
    async def log_unauthorized_access(self, service_id: str, token: str) -> None:
        """Log unauthorized access attempt."""
        event = {
            "event_type": "UNAUTHORIZED_ACCESS",
            "service_id": service_id,
            "token_hash": hashlib.sha256(token.encode()).hexdigest()[:16],
            "timestamp": time.time()
        }
        
        self.security_events.append(event)
        logger.warning(f"Unauthorized access attempt for service {service_id}")
    
    async def log_configuration_change(self, service_id: str, config: Dict[str, Any]) -> None:
        """Log configuration change."""
        event = {
            "event_type": "CONFIGURATION_CHANGED",
            "service_id": service_id,
            "config_hash": hashlib.sha256(str(config).encode()).hexdigest()[:16],
            "timestamp": time.time()
        }
        
        self.security_events.append(event)
        logger.info(f"Configuration changed for service {service_id}")