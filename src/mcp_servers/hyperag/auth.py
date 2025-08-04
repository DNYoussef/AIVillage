"""HypeRAG MCP Server Authentication and Authorization.

Implements role-based access control with granular permissions for different agent types.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import jwt

logger = logging.getLogger(__name__)


class HypeRAGPermissions:
    """Permission constants for HypeRAG operations."""

    # Read Operations
    READ = "hyperag:read"
    READ_LIMITED = "hyperag:read_limited"

    # Write Operations
    WRITE = "hyperag:write"
    WRITE_CODE_DOCS = "hyperag:write_code_docs"

    # Graph Operations
    GRAPH_MODIFY = "hyperag:graph_modify"

    # Repair Operations
    REPAIR_PROPOSE = "hyperag:repair_propose"
    REPAIR_APPROVE = "hyperag:repair_approve"

    # Adapter Operations
    ADAPTER_USE = "hyperag:adapter_use"
    ADAPTER_MANAGE = "hyperag:adapter_manage"

    # System Operations
    GATE_OVERRIDE = "hyperag:gate_override"
    POLICY_MANAGE = "hyperag:policy_manage"
    MONITOR = "hyperag:monitor"

    # Admin Operations
    ADMIN = "hyperag:admin"

    # Agent-specific permission sets
    AGENT_PERMISSIONS = {
        "king": [READ, WRITE, GRAPH_MODIFY, REPAIR_APPROVE, ADAPTER_MANAGE, MONITOR],
        "sage": [READ, WRITE, GRAPH_MODIFY, ADAPTER_USE, MONITOR],
        "magi": [READ, WRITE_CODE_DOCS, ADAPTER_USE, MONITOR],
        "watcher": [READ, MONITOR],
        "external": [READ_LIMITED],
        "guardian": [READ, GATE_OVERRIDE, REPAIR_APPROVE, POLICY_MANAGE, MONITOR],
        "innovator": [READ, REPAIR_PROPOSE, MONITOR],
        "admin": [
            READ,
            WRITE,
            GRAPH_MODIFY,
            REPAIR_PROPOSE,
            REPAIR_APPROVE,
            ADAPTER_USE,
            ADAPTER_MANAGE,
            GATE_OVERRIDE,
            POLICY_MANAGE,
            MONITOR,
            ADMIN,
        ],
    }


@dataclass
class AuthContext:
    """Authentication context for a user/agent."""

    user_id: str
    agent_id: str
    role: str
    permissions: set[str]
    session_id: str
    expires_at: datetime
    ip_address: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AuditLogEntry:
    """Audit log entry for permission decisions."""

    timestamp: datetime
    user_id: str
    agent_id: str
    operation: str
    resource: str
    permission_required: str
    granted: bool
    reason: str
    ip_address: str | None = None
    session_id: str | None = None


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class AuthorizationError(Exception):
    """Raised when authorization fails."""


class PermissionChecker(ABC):
    """Abstract base class for permission checking strategies."""

    @abstractmethod
    async def check_permission(self, context: AuthContext, permission: str, resource: str | None = None) -> bool:
        """Check if the context has the required permission."""


class RoleBasedPermissionChecker(PermissionChecker):
    """Role-based permission checker."""

    def __init__(self, permissions_config: dict[str, list[str]]) -> None:
        self.permissions_config = permissions_config

    async def check_permission(self, context: AuthContext, permission: str, resource: str | None = None) -> bool:
        """Check if the role has the required permission."""
        return permission in context.permissions


class ResourceBasedPermissionChecker(PermissionChecker):
    """Resource-based permission checker with ownership rules."""

    def __init__(self, base_checker: PermissionChecker) -> None:
        self.base_checker = base_checker

    async def check_permission(self, context: AuthContext, permission: str, resource: str | None = None) -> bool:
        """Check permission with resource ownership rules."""
        # Check base permission first
        if not await self.base_checker.check_permission(context, permission, resource):
            return False

        # Additional resource-based checks
        if resource and permission == HypeRAGPermissions.WRITE:
            # Users can always write to their own namespace
            if resource.startswith(f"user:{context.user_id}"):
                return True

            # Code docs can only be written by magi+ roles
            if resource.startswith("docs:code:") and context.role not in [
                "magi",
                "sage",
                "king",
                "admin",
            ]:
                return False

        return True


class TimeBasedPermissionChecker(PermissionChecker):
    """Time-based permission checker with business hours restrictions."""

    def __init__(self, base_checker: PermissionChecker, business_hours: tuple | None = None) -> None:
        self.base_checker = base_checker
        self.business_hours = business_hours or (9, 17)  # 9 AM to 5 PM

    async def check_permission(self, context: AuthContext, permission: str, resource: str | None = None) -> bool:
        """Check permission with time-based restrictions."""
        # Check base permission first
        if not await self.base_checker.check_permission(context, permission, resource):
            return False

        # Time-based restrictions for write operations
        if permission in [HypeRAGPermissions.WRITE, HypeRAGPermissions.GRAPH_MODIFY]:
            current_hour = datetime.now().hour
            if not (self.business_hours[0] <= current_hour <= self.business_hours[1]):
                # Allow admin and emergency roles outside business hours
                if context.role not in ["admin", "guardian"]:
                    return False

        return True


class PermissionManager:
    """Manages authentication and authorization for HypeRAG MCP server."""

    def __init__(
        self,
        jwt_secret: str,
        permissions_config: dict[str, list[str]] | None = None,
        enable_audit: bool = True,
    ) -> None:
        self.jwt_secret = jwt_secret
        self.permissions_config = permissions_config or HypeRAGPermissions.AGENT_PERMISSIONS
        self.enable_audit = enable_audit
        self.audit_log: list[AuditLogEntry] = []
        self.active_sessions: dict[str, AuthContext] = {}

        # Set up permission checker chain
        base_checker = RoleBasedPermissionChecker(self.permissions_config)
        resource_checker = ResourceBasedPermissionChecker(base_checker)
        self.permission_checker = TimeBasedPermissionChecker(resource_checker)

        # Rate limiting
        self.rate_limits: dict[str, list[float]] = {}
        self.max_requests_per_minute = 1000

    async def authenticate_jwt(self, token: str, ip_address: str | None = None) -> AuthContext:
        """Authenticate using JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            user_id = payload.get("sub")
            agent_id = payload.get("agent_id", user_id)
            role = payload.get("role", "external")
            session_id = payload.get("jti", f"session_{int(time.time())}")

            if not user_id:
                msg = "Invalid token: missing user ID"
                raise AuthenticationError(msg)

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                msg = "Token expired"
                raise AuthenticationError(msg)

            # Get permissions for role
            permissions = set(self.permissions_config.get(role, []))

            # Create auth context
            context = AuthContext(
                user_id=user_id,
                agent_id=agent_id,
                role=role,
                permissions=permissions,
                session_id=session_id,
                expires_at=datetime.fromtimestamp(exp) if exp else datetime.now() + timedelta(hours=24),
                ip_address=ip_address,
                metadata=payload.get("metadata", {}),
            )

            # Store active session
            self.active_sessions[session_id] = context

            logger.info(f"Authenticated user {user_id} as {role} with session {session_id}")
            return context

        except jwt.InvalidTokenError as e:
            msg = f"Invalid token: {e!s}"
            raise AuthenticationError(msg)

    async def authenticate_api_key(self, api_key: str, ip_address: str | None = None) -> AuthContext:
        """Authenticate using API key (simplified for demo)."""
        # In production, this would lookup the API key in a secure store
        key_mapping = {
            "hrag_dev_test123": {"user_id": "dev_user", "role": "external"},
            "hrag_prod_king456": {"user_id": "king_agent", "role": "king"},
            "hrag_prod_sage789": {"user_id": "sage_agent", "role": "sage"},
        }

        if api_key not in key_mapping:
            msg = "Invalid API key"
            raise AuthenticationError(msg)

        key_info = key_mapping[api_key]
        user_id = key_info["user_id"]
        role = key_info["role"]

        permissions = set(self.permissions_config.get(role, []))
        session_id = f"apikey_{user_id}_{int(time.time())}"

        context = AuthContext(
            user_id=user_id,
            agent_id=user_id,
            role=role,
            permissions=permissions,
            session_id=session_id,
            expires_at=datetime.now() + timedelta(hours=1),
            ip_address=ip_address,
        )

        self.active_sessions[session_id] = context
        logger.info(f"API key authenticated user {user_id} as {role}")
        return context

    async def check_permission(self, context: AuthContext, permission: str, resource: str | None = None) -> bool:
        """Check if the context has the required permission."""
        # Check rate limiting
        if not await self._check_rate_limit(context.user_id):
            await self._audit_log(
                context,
                "rate_limit",
                resource or "system",
                permission,
                False,
                "Rate limit exceeded",
            )
            return False

        # Check if session is still valid
        if context.expires_at < datetime.now():
            await self._audit_log(
                context,
                "expired_session",
                resource or "system",
                permission,
                False,
                "Session expired",
            )
            return False

        # Use permission checker
        granted = await self.permission_checker.check_permission(context, permission, resource)

        # Audit log
        await self._audit_log(
            context,
            "permission_check",
            resource or "system",
            permission,
            granted,
            "Granted" if granted else "Denied",
        )

        return granted

    async def require_permission(self, context: AuthContext, permission: str, resource: str | None = None) -> None:
        """Require permission or raise AuthorizationError."""
        if not await self.check_permission(context, permission, resource):
            msg = f"Permission denied: {permission} on {resource or 'system'} for role {context.role}"
            raise AuthorizationError(msg)

    async def invalidate_session(self, session_id: str) -> None:
        """Invalidate a session."""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            await self._audit_log(context, "logout", "system", "session", True, "Session invalidated")
            logger.info(f"Invalidated session {session_id}")

    async def get_active_sessions(self) -> list[AuthContext]:
        """Get list of active sessions."""
        now = datetime.now()
        # Clean up expired sessions
        expired_sessions = [sid for sid, ctx in self.active_sessions.items() if ctx.expires_at < now]
        for sid in expired_sessions:
            await self.invalidate_session(sid)

        return list(self.active_sessions.values())

    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check rate limiting for user."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        if user_id in self.rate_limits:
            self.rate_limits[user_id] = [timestamp for timestamp in self.rate_limits[user_id] if timestamp > minute_ago]
        else:
            self.rate_limits[user_id] = []

        # Check limit
        if len(self.rate_limits[user_id]) >= self.max_requests_per_minute:
            return False

        # Add current request
        self.rate_limits[user_id].append(now)
        return True

    async def _audit_log(
        self,
        context: AuthContext,
        operation: str,
        resource: str,
        permission: str,
        granted: bool,
        reason: str,
    ) -> None:
        """Add entry to audit log."""
        if not self.enable_audit:
            return

        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=context.user_id,
            agent_id=context.agent_id,
            operation=operation,
            resource=resource,
            permission_required=permission,
            granted=granted,
            reason=reason,
            ip_address=context.ip_address,
            session_id=context.session_id,
        )

        self.audit_log.append(entry)

        # Keep only last 10000 entries to prevent memory bloat
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]

        # Log security events
        if not granted and operation != "rate_limit":
            logger.warning(
                f"Permission denied: {context.user_id} ({context.role}) "
                f"attempted {permission} on {resource}: {reason}"
            )

    async def get_audit_log(self, user_id: str | None = None, limit: int = 100) -> list[AuditLogEntry]:
        """Get audit log entries."""
        entries = self.audit_log

        if user_id:
            entries = [entry for entry in entries if entry.user_id == user_id]

        return entries[-limit:]

    async def update_permissions(self, role: str, permissions: list[str]) -> None:
        """Update permissions for a role (admin only)."""
        self.permissions_config[role] = permissions

        # Update active sessions with the role
        for context in self.active_sessions.values():
            if context.role == role:
                context.permissions = set(permissions)

        logger.info(f"Updated permissions for role {role}")


# Decorators for easy permission checking
def require_permission(permission: str, resource_param: str | None = None):
    """Decorator to require permission for a method."""

    def decorator(func):
        async def wrapper(self, context: AuthContext, *args, **kwargs):
            resource = kwargs.get(resource_param) if resource_param else None
            await self.permission_manager.require_permission(context, permission, resource)
            return await func(self, context, *args, **kwargs)

        return wrapper

    return decorator


def audit_operation(operation: str):
    """Decorator to audit operations."""

    def decorator(func):
        async def wrapper(self, context: AuthContext, *args, **kwargs):
            start_time = time.time()
            try:
                result = await func(self, context, *args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Operation {operation} completed for {context.user_id} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.exception(f"Operation {operation} failed for {context.user_id} in {duration:.3f}s: {e!s}")
                raise

        return wrapper

    return decorator
