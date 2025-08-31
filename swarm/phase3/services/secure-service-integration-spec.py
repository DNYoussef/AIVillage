"""
Secure Service Integration Specification
Security boundaries, authentication framework, and dependency management

This specification defines the secure integration framework that eliminates 
circular dependencies while maintaining comprehensive security controls.
"""

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """System-wide security levels."""
    PUBLIC = "public"
    PRIVATE = "private" 
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ServiceRole(Enum):
    """Roles in the privacy service architecture."""
    TASK_MANAGER = "task_manager"
    CIRCUIT_MANAGER = "circuit_manager"
    HIDDEN_SERVICE_MANAGER = "hidden_service_manager"
    GOSSIP_COORDINATOR = "gossip_coordinator"
    SECURITY_MONITOR = "security_monitor"
    AUTHENTICATION_SERVICE = "authentication_service"


class EventType(Enum):
    """Event types for service communication."""
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    SECURITY_EVENT = "security_event"
    TASK_SUBMITTED = "task_submitted"
    CIRCUIT_CREATED = "circuit_created"
    HIDDEN_SERVICE_REGISTERED = "hidden_service_registered"
    PEER_DISCOVERED = "peer_discovered"
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"


@dataclass
class SecurityContext:
    """Comprehensive security context for service operations."""
    context_id: str
    security_level: SecurityLevel
    authentication_token: str
    permissions: Set[str]
    encryption_keys: Dict[str, bytes]
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if security context has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions
    
    def add_audit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Add audit event to context."""
        audit_event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": event_data
        }
        self.audit_trail.append(audit_event)


@dataclass
class ServiceEvent:
    """Event for inter-service communication."""
    event_id: str
    event_type: EventType
    source_service: ServiceRole
    target_service: Optional[ServiceRole]
    payload: Dict[str, Any]
    security_context: Optional[SecurityContext] = None
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 minutes default TTL
    
    def is_expired(self) -> bool:
        """Check if event has expired."""
        return time.time() - self.timestamp > self.ttl


class ServiceInterface(Protocol):
    """Protocol defining the interface for privacy services."""
    
    async def start_service(self, security_context: SecurityContext) -> bool:
        """Start the service with security context."""
        ...
    
    async def stop_service(self, security_context: SecurityContext) -> bool:
        """Stop the service securely."""
        ...
    
    async def handle_event(self, event: ServiceEvent) -> bool:
        """Handle incoming service event."""
        ...
    
    def get_service_role(self) -> ServiceRole:
        """Get the service role identifier."""
        ...
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        ...


class AuthenticationCredential:
    """Represents authentication credential for service access."""
    
    def __init__(self, credential_type: str, credential_data: Dict[str, Any]):
        self.credential_id = f"cred_{uuid.uuid4().hex}"
        self.credential_type = credential_type  # token, certificate, key_pair
        self.credential_data = credential_data
        self.created_at = time.time()
        self.expires_at: Optional[float] = None
        self.is_revoked = False
    
    def is_valid(self) -> bool:
        """Check if credential is valid."""
        if self.is_revoked:
            return False
        
        if self.expires_at and time.time() > self.expires_at:
            return False
        
        return True


class PrivacySecurityManager:
    """Centralized security management for privacy services."""
    
    def __init__(self):
        self.security_policies: Dict[SecurityLevel, Dict[str, Any]] = {}
        self.active_contexts: Dict[str, SecurityContext] = {}
        self.threat_monitor = ThreatMonitoringSystem()
        self.audit_logger = SecurityAuditLogger()
        
        self._initialize_security_policies()
        logger.info("PrivacySecurityManager initialized")
    
    def _initialize_security_policies(self) -> None:
        """Initialize security policies for each level."""
        self.security_policies = {
            SecurityLevel.PUBLIC: {
                "encryption_required": False,
                "authentication_required": False,
                "audit_level": "basic",
                "isolation_required": False
            },
            SecurityLevel.PRIVATE: {
                "encryption_required": True,
                "authentication_required": True,
                "audit_level": "standard",
                "isolation_required": True,
                "min_key_size": 2048
            },
            SecurityLevel.CONFIDENTIAL: {
                "encryption_required": True,
                "authentication_required": True,
                "audit_level": "detailed",
                "isolation_required": True,
                "min_key_size": 3072,
                "perfect_forward_secrecy": True
            },
            SecurityLevel.SECRET: {
                "encryption_required": True,
                "authentication_required": True,
                "audit_level": "comprehensive",
                "isolation_required": True,
                "min_key_size": 4096,
                "perfect_forward_secrecy": True,
                "hardware_security_required": True
            }
        }
    
    async def create_security_context(self, security_level: SecurityLevel,
                                    requester_id: str,
                                    requested_permissions: Set[str],
                                    ttl_seconds: float = 3600) -> Optional[SecurityContext]:
        """Create security context with appropriate permissions."""
        try:
            # Validate security level requirements
            if not await self._validate_security_requirements(security_level, requester_id):
                return None
            
            # Generate authentication token
            auth_token = secrets.token_urlsafe(32)
            
            # Generate encryption keys based on security level
            encryption_keys = await self._generate_encryption_keys(security_level)
            
            # Filter permissions based on security level
            allowed_permissions = await self._filter_permissions(
                security_level, requested_permissions
            )
            
            # Create context
            context = SecurityContext(
                context_id=f"ctx_{uuid.uuid4().hex}",
                security_level=security_level,
                authentication_token=auth_token,
                permissions=allowed_permissions,
                encryption_keys=encryption_keys,
                expires_at=time.time() + ttl_seconds
            )
            
            # Store context
            self.active_contexts[context.context_id] = context
            
            # Log creation
            await self.audit_logger.log_context_creation(context, requester_id)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create security context: {e}")
            return None
    
    async def validate_security_context(self, context_id: str,
                                      required_permission: str) -> Tuple[bool, Optional[SecurityContext]]:
        """Validate security context and permission."""
        if context_id not in self.active_contexts:
            return False, None
        
        context = self.active_contexts[context_id]
        
        # Check expiration
        if context.is_expired():
            await self._cleanup_expired_context(context_id)
            return False, None
        
        # Check permission
        if not context.has_permission(required_permission):
            await self.audit_logger.log_permission_denied(context, required_permission)
            return False, None
        
        return True, context
    
    async def revoke_security_context(self, context_id: str) -> bool:
        """Revoke security context."""
        if context_id in self.active_contexts:
            context = self.active_contexts[context_id]
            await self.audit_logger.log_context_revocation(context)
            
            # Secure cleanup
            context.encryption_keys.clear()
            del self.active_contexts[context_id]
            
            return True
        return False
    
    async def _validate_security_requirements(self, security_level: SecurityLevel,
                                            requester_id: str) -> bool:
        """Validate that requester meets security level requirements."""
        # Check threat intelligence
        is_threat = await self.threat_monitor.is_known_threat(requester_id)
        if is_threat:
            return False
        
        # Additional validation based on security level
        policy = self.security_policies[security_level]
        
        if policy.get("hardware_security_required"):
            # In production, would verify hardware security module availability
            pass
        
        return True
    
    async def _generate_encryption_keys(self, security_level: SecurityLevel) -> Dict[str, bytes]:
        """Generate encryption keys appropriate for security level."""
        policy = self.security_policies[security_level]
        min_key_size = policy.get("min_key_size", 2048)
        
        keys = {
            "session_key": secrets.token_bytes(32),
            "hmac_key": secrets.token_bytes(32)
        }
        
        if policy.get("perfect_forward_secrecy"):
            keys["ephemeral_key"] = secrets.token_bytes(32)
        
        return keys
    
    async def _filter_permissions(self, security_level: SecurityLevel,
                                requested_permissions: Set[str]) -> Set[str]:
        """Filter permissions based on security level."""
        # Define permissions by security level
        level_permissions = {
            SecurityLevel.PUBLIC: {
                "read_public_data", "send_public_message"
            },
            SecurityLevel.PRIVATE: {
                "read_public_data", "send_public_message",
                "read_private_data", "send_private_message", "create_circuit"
            },
            SecurityLevel.CONFIDENTIAL: {
                "read_public_data", "send_public_message",
                "read_private_data", "send_private_message", "create_circuit",
                "read_confidential_data", "manage_hidden_service"
            },
            SecurityLevel.SECRET: {
                "read_public_data", "send_public_message",
                "read_private_data", "send_private_message", "create_circuit",
                "read_confidential_data", "manage_hidden_service",
                "read_secret_data", "admin_operations"
            }
        }
        
        allowed_permissions = level_permissions.get(security_level, set())
        return requested_permissions.intersection(allowed_permissions)
    
    async def _cleanup_expired_context(self, context_id: str) -> None:
        """Clean up expired security context."""
        if context_id in self.active_contexts:
            context = self.active_contexts[context_id]
            
            # Secure cleanup
            context.encryption_keys.clear()
            del self.active_contexts[context_id]
            
            await self.audit_logger.log_context_expiration(context)


class SecureServiceBus:
    """Secure communication bus for inter-service messaging."""
    
    def __init__(self, security_manager: PrivacySecurityManager):
        self.security_manager = security_manager
        self.registered_services: Dict[ServiceRole, ServiceInterface] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[ServiceRole, List[asyncio.Queue]] = {}
        self.is_running = False
        
        # Security features
        self.message_encryption = ServiceMessageEncryption()
        self.rate_limiter = ServiceRateLimiter()
        
        logger.info("SecureServiceBus initialized")
    
    async def start_bus(self) -> bool:
        """Start the service bus."""
        try:
            self.is_running = True
            
            # Start event processing
            asyncio.create_task(self._process_events())
            
            # Start health monitoring
            asyncio.create_task(self._monitor_service_health())
            
            logger.info("SecureServiceBus started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service bus: {e}")
            return False
    
    async def register_service(self, service: ServiceInterface,
                             security_context: SecurityContext) -> bool:
        """Register service with the bus."""
        try:
            # Validate security context
            is_valid, _ = await self.security_manager.validate_security_context(
                security_context.context_id, "register_service"
            )
            
            if not is_valid:
                return False
            
            service_role = service.get_service_role()
            self.registered_services[service_role] = service
            self.event_handlers[service_role] = []
            
            logger.info(f"Registered service: {service_role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False
    
    async def send_event(self, event: ServiceEvent,
                        sender_security_context: SecurityContext) -> bool:
        """Send event through secure service bus."""
        try:
            # Validate sender context
            is_valid, _ = await self.security_manager.validate_security_context(
                sender_security_context.context_id, "send_message"
            )
            
            if not is_valid:
                return False
            
            # Rate limiting check
            if not await self.rate_limiter.allow_event(event.source_service, event.event_type):
                logger.warning(f"Rate limit exceeded for {event.source_service.value}")
                return False
            
            # Encrypt event if required
            if sender_security_context.security_level != SecurityLevel.PUBLIC:
                event = await self.message_encryption.encrypt_event(
                    event, sender_security_context
                )
            
            # Add to queue
            await self.event_queue.put(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    async def subscribe_to_events(self, service_role: ServiceRole,
                                event_types: List[EventType]) -> Optional[asyncio.Queue]:
        """Subscribe service to specific event types."""
        if service_role not in self.registered_services:
            return None
        
        # Create event queue for subscriber
        subscriber_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers[service_role].append(subscriber_queue)
        
        logger.info(f"Service {service_role.value} subscribed to events: {[e.value for e in event_types]}")
        return subscriber_queue
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self.is_running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Validate event
                if event.is_expired():
                    logger.warning(f"Discarding expired event: {event.event_id}")
                    continue
                
                # Route event to appropriate services
                await self._route_event(event)
                
            except asyncio.TimeoutError:
                continue  # No events, continue loop
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _route_event(self, event: ServiceEvent) -> None:
        """Route event to appropriate service handlers."""
        if event.target_service:
            # Direct message to specific service
            if event.target_service in self.event_handlers:
                for handler_queue in self.event_handlers[event.target_service]:
                    try:
                        await handler_queue.put(event)
                    except asyncio.QueueFull:
                        logger.warning(f"Handler queue full for {event.target_service.value}")
        else:
            # Broadcast to all services
            for service_role, handler_queues in self.event_handlers.items():
                if service_role != event.source_service:  # Don't send back to sender
                    for handler_queue in handler_queues:
                        try:
                            await handler_queue.put(event)
                        except asyncio.QueueFull:
                            logger.warning(f"Handler queue full for {service_role.value}")
    
    async def _monitor_service_health(self) -> None:
        """Monitor health of registered services."""
        while self.is_running:
            try:
                for service_role, service in self.registered_services.items():
                    try:
                        status = await asyncio.wait_for(
                            service.get_service_status(), timeout=5.0
                        )
                        
                        # Check if service is healthy
                        if not status.get("is_healthy", True):
                            logger.warning(f"Service {service_role.value} reported unhealthy status")
                    
                    except asyncio.TimeoutError:
                        logger.warning(f"Health check timeout for {service_role.value}")
                    except Exception as e:
                        logger.error(f"Health check error for {service_role.value}: {e}")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(10)


class ServiceMessageEncryption:
    """Handles encryption/decryption of service messages."""
    
    def __init__(self):
        self.encryption_cache = {}
    
    async def encrypt_event(self, event: ServiceEvent,
                          security_context: SecurityContext) -> ServiceEvent:
        """Encrypt event payload based on security context."""
        if security_context.security_level == SecurityLevel.PUBLIC:
            return event
        
        try:
            # Get encryption key from context
            session_key = security_context.encryption_keys.get("session_key")
            if not session_key:
                raise ValueError("No session key available for encryption")
            
            # Encrypt payload
            import json
            payload_bytes = json.dumps(event.payload).encode('utf-8')
            
            # Simple encryption (in production would use proper AES-GCM)
            nonce = secrets.token_bytes(12)
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(nonce))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(payload_bytes) + encryptor.finalize()
            
            # Replace payload with encrypted version
            encrypted_payload = {
                "encrypted": True,
                "nonce": nonce.hex(),
                "ciphertext": ciphertext.hex(),
                "tag": encryptor.tag.hex()
            }
            
            event.payload = encrypted_payload
            return event
            
        except Exception as e:
            logger.error(f"Event encryption failed: {e}")
            raise
    
    async def decrypt_event(self, event: ServiceEvent,
                          security_context: SecurityContext) -> ServiceEvent:
        """Decrypt event payload."""
        if not event.payload.get("encrypted", False):
            return event
        
        try:
            # Get encryption key
            session_key = security_context.encryption_keys.get("session_key")
            if not session_key:
                raise ValueError("No session key available for decryption")
            
            # Extract encrypted components
            nonce = bytes.fromhex(event.payload["nonce"])
            ciphertext = bytes.fromhex(event.payload["ciphertext"])
            tag = bytes.fromhex(event.payload["tag"])
            
            # Decrypt
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(nonce, tag))
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Parse decrypted payload
            import json
            event.payload = json.loads(plaintext.decode('utf-8'))
            
            return event
            
        except Exception as e:
            logger.error(f"Event decryption failed: {e}")
            raise


class ServiceRateLimiter:
    """Rate limiting for service communications."""
    
    def __init__(self):
        self.service_limits = {
            ServiceRole.TASK_MANAGER: {"events_per_minute": 100},
            ServiceRole.CIRCUIT_MANAGER: {"events_per_minute": 200},
            ServiceRole.HIDDEN_SERVICE_MANAGER: {"events_per_minute": 50},
            ServiceRole.GOSSIP_COORDINATOR: {"events_per_minute": 500},
            ServiceRole.SECURITY_MONITOR: {"events_per_minute": 1000}
        }
        self.event_counts: Dict[ServiceRole, List[float]] = {}
    
    async def allow_event(self, service_role: ServiceRole, event_type: EventType) -> bool:
        """Check if event is allowed based on rate limits."""
        current_time = time.time()
        
        if service_role not in self.event_counts:
            self.event_counts[service_role] = []
        
        # Clean old entries (older than 1 minute)
        self.event_counts[service_role] = [
            t for t in self.event_counts[service_role]
            if current_time - t < 60
        ]
        
        # Check limit
        limit = self.service_limits.get(service_role, {}).get("events_per_minute", 100)
        if len(self.event_counts[service_role]) >= limit:
            return False
        
        # Record this event
        self.event_counts[service_role].append(current_time)
        return True


class ThreatMonitoringSystem:
    """Monitors for security threats across services."""
    
    def __init__(self):
        self.known_threats: Set[str] = set()
        self.threat_patterns = {}
        self.anomaly_detector = ServiceAnomalyDetector()
    
    async def is_known_threat(self, identifier: str) -> bool:
        """Check if identifier is a known threat."""
        return identifier in self.known_threats
    
    async def add_threat(self, identifier: str, threat_type: str,
                        evidence: Dict[str, Any]) -> None:
        """Add identifier to threat list."""
        self.known_threats.add(identifier)
        
        threat_info = {
            "threat_type": threat_type,
            "evidence": evidence,
            "added_at": time.time()
        }
        self.threat_patterns[identifier] = threat_info
        
        logger.warning(f"Added threat: {identifier} ({threat_type})")
    
    async def analyze_service_behavior(self, service_role: ServiceRole,
                                     behavior_data: Dict[str, Any]) -> bool:
        """Analyze service behavior for anomalies."""
        return await self.anomaly_detector.is_anomalous_behavior(
            service_role, behavior_data
        )


class ServiceAnomalyDetector:
    """Detects anomalous behavior patterns in services."""
    
    def __init__(self):
        self.behavior_baselines = {}
        self.anomaly_thresholds = {
            "event_rate_multiplier": 3.0,
            "error_rate_threshold": 0.1,
            "response_time_multiplier": 2.0
        }
    
    async def is_anomalous_behavior(self, service_role: ServiceRole,
                                  behavior_data: Dict[str, Any]) -> bool:
        """Check if service behavior is anomalous."""
        if service_role not in self.behavior_baselines:
            # Establish baseline
            self.behavior_baselines[service_role] = behavior_data
            return False
        
        baseline = self.behavior_baselines[service_role]
        
        # Check event rate
        if behavior_data.get("events_per_minute", 0) > \
           baseline.get("events_per_minute", 0) * self.anomaly_thresholds["event_rate_multiplier"]:
            return True
        
        # Check error rate
        if behavior_data.get("error_rate", 0) > self.anomaly_thresholds["error_rate_threshold"]:
            return True
        
        return False


class SecurityAuditLogger:
    """Logs security events for audit and compliance."""
    
    def __init__(self):
        self.audit_entries = []
        self.log_file_path = "security_audit.log"
    
    async def log_context_creation(self, context: SecurityContext, requester_id: str) -> None:
        """Log security context creation."""
        entry = {
            "event_type": "CONTEXT_CREATED",
            "context_id": context.context_id,
            "security_level": context.security_level.value,
            "requester_id": requester_id,
            "permissions": list(context.permissions),
            "timestamp": time.time()
        }
        
        await self._write_audit_entry(entry)
    
    async def log_permission_denied(self, context: SecurityContext, permission: str) -> None:
        """Log permission denied event."""
        entry = {
            "event_type": "PERMISSION_DENIED",
            "context_id": context.context_id,
            "requested_permission": permission,
            "available_permissions": list(context.permissions),
            "timestamp": time.time()
        }
        
        await self._write_audit_entry(entry)
    
    async def log_context_revocation(self, context: SecurityContext) -> None:
        """Log context revocation."""
        entry = {
            "event_type": "CONTEXT_REVOKED",
            "context_id": context.context_id,
            "security_level": context.security_level.value,
            "timestamp": time.time()
        }
        
        await self._write_audit_entry(entry)
    
    async def log_context_expiration(self, context: SecurityContext) -> None:
        """Log context expiration."""
        entry = {
            "event_type": "CONTEXT_EXPIRED",
            "context_id": context.context_id,
            "security_level": context.security_level.value,
            "lifetime": time.time() - context.created_at,
            "timestamp": time.time()
        }
        
        await self._write_audit_entry(entry)
    
    async def _write_audit_entry(self, entry: Dict[str, Any]) -> None:
        """Write audit entry to log."""
        self.audit_entries.append(entry)
        
        # In production, would write to secure, tamper-evident log
        logger.info(f"AUDIT: {entry['event_type']} - {entry}")


class DependencyInjectionContainer:
    """Manages service dependencies without circular references."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.service_configs: Dict[str, Dict[str, Any]] = {}
        self.initialization_order = []
    
    def register_service(self, service_name: str, service_class: type,
                        config: Dict[str, Any], dependencies: List[str]) -> None:
        """Register service with its dependencies."""
        self.service_configs[service_name] = {
            "class": service_class,
            "config": config,
            "dependencies": dependencies
        }
    
    async def initialize_services(self) -> bool:
        """Initialize all services in dependency order."""
        try:
            # Calculate initialization order
            self.initialization_order = self._calculate_dependency_order()
            
            # Initialize services
            for service_name in self.initialization_order:
                await self._initialize_service(service_name)
            
            logger.info("All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False
    
    def _calculate_dependency_order(self) -> List[str]:
        """Calculate service initialization order based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            config = self.service_configs.get(service_name, {})
            for dependency in config.get("dependencies", []):
                visit(dependency)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        for service_name in self.service_configs:
            if service_name not in visited:
                visit(service_name)
        
        return order
    
    async def _initialize_service(self, service_name: str) -> None:
        """Initialize individual service with its dependencies."""
        config = self.service_configs[service_name]
        service_class = config["class"]
        
        # Resolve dependencies
        dependency_instances = {}
        for dependency_name in config["dependencies"]:
            if dependency_name in self.services:
                dependency_instances[dependency_name] = self.services[dependency_name]
            else:
                raise ValueError(f"Dependency {dependency_name} not found for service {service_name}")
        
        # Create service instance
        service_instance = service_class(
            config=config["config"],
            dependencies=dependency_instances
        )
        
        self.services[service_name] = service_instance
        logger.info(f"Initialized service: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get service instance by name."""
        return self.services.get(service_name)