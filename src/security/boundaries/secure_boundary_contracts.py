#!/usr/bin/env python3
"""
Secure Module Boundary Contracts for AIVillage
Implements security boundaries using connascence principles and dependency injection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Protocol, TypeVar, Optional
import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
R = TypeVar('R')

class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"

class SecurityViolationType(Enum):
    """Types of security violations"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INVALID_AUTHENTICATION = "invalid_authentication"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_LEAKAGE = "data_leakage"
    NON_LOCALHOST_ADMIN = "non_localhost_admin"
    WEAK_COUPLING_VIOLATION = "weak_coupling_violation"

@dataclass(frozen=True)
class SecurityContext:
    """Immutable security context - weak connascence interface"""
    user_id: str
    session_id: str
    roles: frozenset[str]
    permissions: frozenset[str]
    security_level: SecurityLevel
    source_ip: str
    user_agent: str
    timestamp: datetime
    node_id: Optional[str] = None
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if context has specific role"""
        return role in self.roles
    
    def is_localhost(self) -> bool:
        """Check if request originates from localhost"""
        return self.source_ip in ['127.0.0.1', '::1', 'localhost']

class SecurityException(Exception):
    """Base security exception"""
    
    def __init__(self, message: str, violation_type: SecurityViolationType, context: Optional[SecurityContext] = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.context = context
        self.timestamp = datetime.utcnow()

# Security service interfaces (weak connascence through protocols)
class AuthenticationService(Protocol):
    """Authentication service interface"""
    
    async def validate_token(self, token: str) -> bool: ...
    async def validate_mfa(self, context: SecurityContext) -> bool: ...
    async def get_node_certificate(self, node_id: str) -> Any: ...

class AuthorizationService(Protocol):
    """Authorization service interface"""
    
    async def check_permission(self, context: SecurityContext, permission: str) -> bool: ...
    async def check_admin_permission(self, context: SecurityContext) -> bool: ...
    async def check_p2p_permission(self, context: SecurityContext) -> bool: ...

class AuditService(Protocol):
    """Audit service interface"""
    
    async def log_security_event(self, event_type: str, context: SecurityContext, details: dict = None) -> None: ...
    async def log_security_violation(self, violation_type: str, context: SecurityContext, details: dict = None) -> None: ...
    async def get_recent_logs(self, limit: int, user_id: str) -> list[dict]: ...

class ThreatDetectionService(Protocol):
    """Threat detection service interface"""
    
    async def analyze_request_pattern(self, context: SecurityContext) -> dict: ...
    async def detect_anomaly(self, context: SecurityContext, action: str) -> bool: ...
    async def trigger_comprehensive_scan(self, user_id: str) -> dict: ...

# Abstract security boundary base class
class SecureBoundary(Generic[T], ABC):
    """
    Abstract security boundary with dependency injection.
    
    Follows connascence principles:
    - Weak connascence through dependency injection
    - Strong connascence kept local within implementations
    """
    
    def __init__(self, 
                 auth_service: AuthenticationService,
                 authz_service: AuthorizationService, 
                 audit_service: AuditService,
                 threat_service: ThreatDetectionService):
        # Weak connascence - injected dependencies
        self._auth = auth_service
        self._authz = authz_service
        self._audit = audit_service
        self._threat = threat_service
        
        # Local state - strong connascence contained
        self._access_count = 0
        self._last_access = None
    
    @abstractmethod
    async def validate_access(self, context: SecurityContext) -> bool:
        """Validate access - implementations keep logic local"""
        pass
    
    @abstractmethod
    async def get_required_permissions(self) -> set[str]:
        """Get permissions required for this boundary"""
        pass
    
    async def execute_secured(self, 
                             context: SecurityContext, 
                             operation: Callable[[], T]) -> T:
        """Execute operation with security wrapper"""
        
        # Validate access first
        if not await self.validate_access(context):
            await self._audit.log_security_violation(
                "access_denied", 
                context,
                {"boundary": self.__class__.__name__}
            )
            raise SecurityException(
                f"Access denied to {self.__class__.__name__}",
                SecurityViolationType.UNAUTHORIZED_ACCESS,
                context
            )
        
        # Check for threat patterns
        threat_analysis = await self._threat.analyze_request_pattern(context)
        if threat_analysis.get("risk_level") == "high":
            await self._audit.log_security_violation(
                "high_risk_pattern",
                context, 
                threat_analysis
            )
            raise SecurityException(
                "Request blocked due to threat pattern",
                SecurityViolationType.UNAUTHORIZED_ACCESS,
                context
            )
        
        # Execute with audit logging
        start_time = datetime.utcnow()
        try:
            # Update local state (strong connascence within class)
            self._access_count += 1
            self._last_access = start_time
            
            result = await self._execute_with_monitoring(operation, context)
            
            await self._audit.log_security_event(
                "secure_operation_success",
                context,
                {
                    "boundary": self.__class__.__name__,
                    "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    "access_count": self._access_count
                }
            )
            
            return result
            
        except Exception as e:
            await self._audit.log_security_violation(
                "secure_operation_failure",
                context,
                {
                    "boundary": self.__class__.__name__, 
                    "error": str(e),
                    "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000)
                }
            )
            raise
    
    async def _execute_with_monitoring(self, 
                                      operation: Callable[[], T], 
                                      context: SecurityContext) -> T:
        """Execute operation with monitoring - can be overridden by subclasses"""
        
        if asyncio.iscoroutinefunction(operation):
            return await operation()
        else:
            return operation()

class AdminBoundary(SecureBoundary[dict]):
    """
    Admin interface security boundary - localhost only access.
    
    Strong connascence for admin-specific logic kept within this class.
    """
    
    def __init__(self,
                 auth_service: AuthenticationService,
                 authz_service: AuthorizationService,
                 audit_service: AuditService,
                 threat_service: ThreatDetectionService):
        super().__init__(auth_service, authz_service, audit_service, threat_service)
        
        # Admin-specific configuration (strong connascence local to this class)
        self._max_failed_attempts = 3
        self._failed_attempts: dict[str, int] = {}
        self._lockout_duration_minutes = 15
        self._locked_out_users: dict[str, datetime] = {}
    
    async def validate_access(self, context: SecurityContext) -> bool:
        """Validate admin access with comprehensive checks"""
        
        # Check if user is locked out
        if self._is_user_locked_out(context.user_id):
            await self._audit.log_security_violation(
                "locked_out_user_attempt",
                context,
                {"lockout_remaining_minutes": self._get_lockout_remaining_minutes(context.user_id)}
            )
            return False
        
        # CRITICAL: Localhost only check
        if not self._is_localhost_request(context):
            await self._audit.log_security_violation(
                "non_localhost_admin_access",
                context,
                {"attempted_from": context.source_ip}
            )
            self._record_failed_attempt(context.user_id)
            return False
        
        # Multi-factor authentication required
        if not await self._auth.validate_mfa(context):
            await self._audit.log_security_violation(
                "mfa_failure",
                context
            )
            self._record_failed_attempt(context.user_id)
            return False
        
        # Admin permission check
        if not await self._authz.check_admin_permission(context):
            await self._audit.log_security_violation(
                "insufficient_admin_privileges", 
                context,
                {"user_roles": list(context.roles)}
            )
            self._record_failed_attempt(context.user_id)
            return False
        
        # Reset failed attempts on successful validation
        self._reset_failed_attempts(context.user_id)
        return True
    
    async def get_required_permissions(self) -> set[str]:
        """Admin boundary requires admin permissions"""
        return {"admin:read", "admin:write", "admin:system"}
    
    def _is_localhost_request(self, context: SecurityContext) -> bool:
        """
        Check if request comes from localhost.
        Strong connascence - local implementation detail.
        """
        localhost_addresses = ['127.0.0.1', '::1', 'localhost']
        return context.source_ip in localhost_addresses
    
    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is currently locked out"""
        if user_id not in self._locked_out_users:
            return False
        
        lockout_time = self._locked_out_users[user_id]
        lockout_duration = datetime.utcnow() - lockout_time
        
        if lockout_duration.total_seconds() > (self._lockout_duration_minutes * 60):
            # Lockout expired
            del self._locked_out_users[user_id]
            return False
        
        return True
    
    def _get_lockout_remaining_minutes(self, user_id: str) -> int:
        """Get remaining lockout time in minutes"""
        if user_id not in self._locked_out_users:
            return 0
        
        lockout_time = self._locked_out_users[user_id]
        elapsed_seconds = (datetime.utcnow() - lockout_time).total_seconds()
        remaining_seconds = (self._lockout_duration_minutes * 60) - elapsed_seconds
        
        return max(0, int(remaining_seconds / 60))
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt"""
        self._failed_attempts[user_id] = self._failed_attempts.get(user_id, 0) + 1
        
        if self._failed_attempts[user_id] >= self._max_failed_attempts:
            self._locked_out_users[user_id] = datetime.utcnow()
            logger.warning(f"User {user_id} locked out due to {self._max_failed_attempts} failed attempts")
    
    def _reset_failed_attempts(self, user_id: str):
        """Reset failed attempts for user"""
        if user_id in self._failed_attempts:
            del self._failed_attempts[user_id]
        if user_id in self._locked_out_users:
            del self._locked_out_users[user_id]

class P2PBoundary(SecureBoundary[bytes]):
    """
    P2P network security boundary with node authentication and trust scoring.
    """
    
    def __init__(self,
                 auth_service: AuthenticationService,
                 authz_service: AuthorizationService,
                 audit_service: AuditService,
                 threat_service: ThreatDetectionService):
        super().__init__(auth_service, authz_service, audit_service, threat_service)
        
        # P2P-specific configuration
        self._min_trust_score = 0.7
        self._trust_scores: dict[str, float] = {}
        self._node_reputations: dict[str, dict] = {}
    
    async def validate_access(self, context: SecurityContext) -> bool:
        """Validate P2P network access"""
        
        if not context.node_id:
            await self._audit.log_security_violation(
                "p2p_missing_node_id",
                context
            )
            return False
        
        # Node certificate validation
        node_cert = await self._auth.get_node_certificate(context.node_id)
        if not node_cert or not node_cert.is_valid():
            await self._audit.log_security_violation(
                "p2p_invalid_certificate",
                context,
                {"node_id": context.node_id}
            )
            return False
        
        # Trust score validation
        trust_score = await self._get_trust_score(context.node_id)
        if trust_score < self._min_trust_score:
            await self._audit.log_security_violation(
                "p2p_insufficient_trust",
                context,
                {"node_id": context.node_id, "trust_score": trust_score}
            )
            return False
        
        # P2P permission check
        if not await self._authz.check_p2p_permission(context):
            await self._audit.log_security_violation(
                "p2p_insufficient_permissions",
                context,
                {"node_id": context.node_id}
            )
            return False
        
        return True
    
    async def get_required_permissions(self) -> set[str]:
        """P2P boundary requires network permissions"""
        return {"p2p:connect", "p2p:message", "p2p:discovery"}
    
    async def _get_trust_score(self, node_id: str) -> float:
        """
        Get trust score for node.
        This would integrate with reputation system in production.
        """
        if node_id not in self._trust_scores:
            # New nodes start with neutral trust
            self._trust_scores[node_id] = 0.5
        
        return self._trust_scores[node_id]
    
    def update_trust_score(self, node_id: str, delta: float):
        """Update trust score based on behavior"""
        current_score = self._trust_scores.get(node_id, 0.5)
        new_score = max(0.0, min(1.0, current_score + delta))
        self._trust_scores[node_id] = new_score
        
        logger.info(f"Updated trust score for node {node_id}: {current_score} -> {new_score}")

class RAGBoundary(SecureBoundary[dict]):
    """
    RAG system security boundary with PII protection and query validation.
    """
    
    def __init__(self,
                 auth_service: AuthenticationService,
                 authz_service: AuthorizationService,
                 audit_service: AuditService,
                 threat_service: ThreatDetectionService):
        super().__init__(auth_service, authz_service, audit_service, threat_service)
        
        # RAG-specific configuration
        self._max_query_length = 1000
        self._blocked_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        ]
        self._query_rate_limits: dict[str, list] = {}
    
    async def validate_access(self, context: SecurityContext) -> bool:
        """Validate RAG system access"""
        
        # Rate limiting check
        if not self._check_rate_limit(context.user_id):
            await self._audit.log_security_violation(
                "rag_rate_limit_exceeded",
                context
            )
            return False
        
        # Basic permission check
        if not context.has_permission("rag:query"):
            await self._audit.log_security_violation(
                "rag_insufficient_permissions",
                context
            )
            return False
        
        return True
    
    async def get_required_permissions(self) -> set[str]:
        """RAG boundary requires query permissions"""
        return {"rag:query", "rag:read"}
    
    async def validate_query(self, query: str, context: SecurityContext) -> bool:
        """Validate RAG query for safety"""
        
        # Length check
        if len(query) > self._max_query_length:
            await self._audit.log_security_violation(
                "rag_query_too_long",
                context,
                {"query_length": len(query)}
            )
            return False
        
        # Pattern matching for sensitive data
        import re
        for pattern in self._blocked_patterns:
            if re.search(pattern, query):
                await self._audit.log_security_violation(
                    "rag_query_contains_sensitive_data",
                    context,
                    {"pattern_matched": pattern}
                )
                return False
        
        return True
    
    def _check_rate_limit(self, user_id: str, limit: int = 100, window_minutes: int = 60) -> bool:
        """Check if user is within rate limits"""
        now = datetime.utcnow()
        window_start = now.timestamp() - (window_minutes * 60)
        
        if user_id not in self._query_rate_limits:
            self._query_rate_limits[user_id] = []
        
        # Clean old entries
        self._query_rate_limits[user_id] = [
            timestamp for timestamp in self._query_rate_limits[user_id]
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self._query_rate_limits[user_id]) >= limit:
            return False
        
        # Record new query
        self._query_rate_limits[user_id].append(now.timestamp())
        return True

# Factory for creating security boundaries
class SecurityBoundaryFactory:
    """Factory for creating security boundaries with proper dependency injection"""
    
    def __init__(self, 
                 auth_service: AuthenticationService,
                 authz_service: AuthorizationService,
                 audit_service: AuditService,
                 threat_service: ThreatDetectionService):
        self._auth = auth_service
        self._authz = authz_service
        self._audit = audit_service
        self._threat = threat_service
    
    def create_admin_boundary(self) -> AdminBoundary:
        """Create admin boundary with injected dependencies"""
        return AdminBoundary(
            self._auth,
            self._authz, 
            self._audit,
            self._threat
        )
    
    def create_p2p_boundary(self) -> P2PBoundary:
        """Create P2P boundary with injected dependencies"""
        return P2PBoundary(
            self._auth,
            self._authz,
            self._audit,
            self._threat
        )
    
    def create_rag_boundary(self) -> RAGBoundary:
        """Create RAG boundary with injected dependencies"""
        return RAGBoundary(
            self._auth,
            self._authz,
            self._audit,
            self._threat
        )

# Context manager for security boundaries
@asynccontextmanager
async def security_boundary_context(boundary: SecureBoundary, context: SecurityContext):
    """Context manager for security boundary operations"""
    
    # Pre-execution validation
    if not await boundary.validate_access(context):
        raise SecurityException(
            f"Access denied to {boundary.__class__.__name__}",
            SecurityViolationType.UNAUTHORIZED_ACCESS,
            context
        )
    
    try:
        yield boundary
    except Exception as e:
        # Log security-related exceptions
        if isinstance(e, SecurityException):
            logger.error(f"Security violation in {boundary.__class__.__name__}: {e}")
        raise
    finally:
        # Cleanup if needed
        pass

# Example usage and integration patterns
if __name__ == "__main__":
    # This would typically be done through a proper dependency injection container
    from unittest.mock import AsyncMock
    
    async def example_usage():
        """Example of how to use security boundaries"""
        
        # Mock services for example
        auth_service = AsyncMock(spec=AuthenticationService)
        authz_service = AsyncMock(spec=AuthorizationService)
        audit_service = AsyncMock(spec=AuditService)
        threat_service = AsyncMock(spec=ThreatDetectionService)
        
        # Configure mocks
        auth_service.validate_mfa.return_value = True
        authz_service.check_admin_permission.return_value = True
        audit_service.log_security_event.return_value = None
        threat_service.analyze_request_pattern.return_value = {"risk_level": "low"}
        
        # Create factory
        factory = SecurityBoundaryFactory(
            auth_service, authz_service, audit_service, threat_service
        )
        
        # Create security context
        context = SecurityContext(
            user_id="admin_user",
            session_id="session_123",
            roles=frozenset(["admin"]),
            permissions=frozenset(["admin:read", "admin:write"]),
            security_level=SecurityLevel.RESTRICTED,
            source_ip="127.0.0.1",
            user_agent="AdminClient/1.0",
            timestamp=datetime.utcnow()
        )
        
        # Use admin boundary
        admin_boundary = factory.create_admin_boundary()
        
        async with security_boundary_context(admin_boundary, context) as boundary:
            result = await boundary.execute_secured(
                context,
                lambda: {"status": "Admin operation successful", "timestamp": datetime.utcnow()}
            )
            print(f"Admin operation result: {result}")
    
    # Run example
    asyncio.run(example_usage())