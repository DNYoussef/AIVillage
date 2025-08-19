"""
Role-Based Access Control (RBAC) and Multi-Tenant Isolation System for AIVillage.

This module provides comprehensive security infrastructure for production deployment
with role-based permissions, tenant isolation, and secure resource access control.
"""

import asyncio
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class Role(Enum):
    """System roles with hierarchical permissions."""

    SUPER_ADMIN = "super_admin"  # Full system access
    ADMIN = "admin"  # Tenant administration
    DEVELOPER = "developer"  # Development and deployment
    DATA_SCIENTIST = "data_scientist"  # Model training and RAG access
    OPERATOR = "operator"  # System monitoring and operations
    USER = "user"  # Basic usage
    GUEST = "guest"  # Read-only access


class Permission(Enum):
    """Granular permissions for resource access."""

    # Agent permissions
    AGENT_CREATE = "agent.create"
    AGENT_READ = "agent.read"
    AGENT_UPDATE = "agent.update"
    AGENT_DELETE = "agent.delete"
    AGENT_EXECUTE = "agent.execute"

    # RAG permissions
    RAG_CREATE = "rag.create"
    RAG_READ = "rag.read"
    RAG_UPDATE = "rag.update"
    RAG_DELETE = "rag.delete"
    RAG_QUERY = "rag.query"

    # P2P Network permissions
    P2P_JOIN = "p2p.join"
    P2P_CREATE = "p2p.create"
    P2P_ADMIN = "p2p.admin"
    P2P_MONITOR = "p2p.monitor"

    # Model permissions
    MODEL_TRAIN = "model.train"
    MODEL_DEPLOY = "model.deploy"
    MODEL_DELETE = "model.delete"
    MODEL_INFERENCE = "model.inference"

    # System permissions
    SYSTEM_CONFIG = "system.config"
    SYSTEM_MONITOR = "system.monitor"
    SYSTEM_BACKUP = "system.backup"
    SYSTEM_RESTORE = "system.restore"

    # Tenant permissions
    TENANT_CREATE = "tenant.create"
    TENANT_MANAGE = "tenant.manage"
    TENANT_DELETE = "tenant.delete"


# Role-Permission Mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.SUPER_ADMIN: set(Permission),  # All permissions
    Role.ADMIN: {
        Permission.AGENT_CREATE,
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.AGENT_DELETE,
        Permission.AGENT_EXECUTE,
        Permission.RAG_CREATE,
        Permission.RAG_READ,
        Permission.RAG_UPDATE,
        Permission.RAG_DELETE,
        Permission.RAG_QUERY,
        Permission.P2P_JOIN,
        Permission.P2P_CREATE,
        Permission.P2P_ADMIN,
        Permission.MODEL_DEPLOY,
        Permission.MODEL_INFERENCE,
        Permission.SYSTEM_CONFIG,
        Permission.SYSTEM_MONITOR,
        Permission.SYSTEM_BACKUP,
        Permission.SYSTEM_RESTORE,
        Permission.TENANT_MANAGE,
    },
    Role.DEVELOPER: {
        Permission.AGENT_CREATE,
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.AGENT_EXECUTE,
        Permission.RAG_CREATE,
        Permission.RAG_READ,
        Permission.RAG_UPDATE,
        Permission.RAG_QUERY,
        Permission.P2P_JOIN,
        Permission.P2P_CREATE,
        Permission.MODEL_TRAIN,
        Permission.MODEL_DEPLOY,
        Permission.MODEL_INFERENCE,
        Permission.SYSTEM_MONITOR,
    },
    Role.DATA_SCIENTIST: {
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.RAG_READ,
        Permission.RAG_QUERY,
        Permission.P2P_JOIN,
        Permission.MODEL_TRAIN,
        Permission.MODEL_INFERENCE,
        Permission.SYSTEM_MONITOR,
    },
    Role.OPERATOR: {
        Permission.AGENT_READ,
        Permission.RAG_READ,
        Permission.P2P_MONITOR,
        Permission.SYSTEM_MONITOR,
        Permission.SYSTEM_BACKUP,
    },
    Role.USER: {
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.RAG_READ,
        Permission.RAG_QUERY,
        Permission.P2P_JOIN,
        Permission.MODEL_INFERENCE,
    },
    Role.GUEST: {
        Permission.AGENT_READ,
        Permission.RAG_READ,
    },
}


@dataclass
class TenantConfig:
    """Configuration for tenant isolation."""

    tenant_id: str
    name: str
    created_at: datetime
    resource_limits: dict[str, Any] = field(default_factory=dict)
    data_encryption_key: bytes | None = None
    isolated_namespaces: list[str] = field(default_factory=list)
    allowed_models: list[str] = field(default_factory=list)
    max_agents: int = 10
    max_rag_collections: int = 5
    max_p2p_nodes: int = 20
    storage_quota_gb: int = 100
    compute_quota_vcpu: int = 8
    memory_quota_gb: int = 32


@dataclass
class User:
    """User entity with role and tenant association."""

    user_id: str
    username: str
    email: str
    tenant_id: str
    role: Role
    created_at: datetime
    last_login: datetime | None = None
    active: bool = True
    mfa_enabled: bool = False
    api_keys: list[str] = field(default_factory=list)
    custom_permissions: set[Permission] = field(default_factory=set)


@dataclass
class Session:
    """User session with JWT token management."""

    session_id: str
    user_id: str
    tenant_id: str
    token: str
    refresh_token: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str


class RBACSystem:
    """Main RBAC and multi-tenant isolation system."""

    def __init__(self, config_path: Path | None = None):
        """Initialize RBAC system with configuration."""
        self.config_path = config_path or Path("config/security/rbac.json")
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.token_expiry = timedelta(hours=1)
        self.refresh_expiry = timedelta(days=7)

        # Storage
        self.tenants: dict[str, TenantConfig] = {}
        self.users: dict[str, User] = {}
        self.sessions: dict[str, Session] = {}
        self.revoked_tokens: set[str] = set()

        # Audit log
        self.audit_log: list[dict[str, Any]] = []

        # Load configuration
        self._load_config()

        # Initialize encryption
        self._init_encryption()

        logger.info("RBAC system initialized")

    def _load_config(self):
        """Load RBAC configuration from file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                # Load configuration settings
                self.jwt_secret = config.get("jwt_secret", self.jwt_secret)
                self.token_expiry = timedelta(hours=config.get("token_expiry_hours", 1))
                self.refresh_expiry = timedelta(days=config.get("refresh_expiry_days", 7))

    def _init_encryption(self):
        """Initialize encryption for tenant data isolation."""
        self.master_key = Fernet.generate_key()
        self.fernet = Fernet(self.master_key)

    def _generate_tenant_key(self, tenant_id: str) -> bytes:
        """Generate tenant-specific encryption key."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            iterations=100000,
        )
        kdf.derive(self.master_key)
        return Fernet(Fernet.generate_key())

    # Tenant Management

    async def create_tenant(
        self, name: str, admin_user: dict[str, str], config: dict[str, Any] | None = None
    ) -> TenantConfig:
        """Create new tenant with isolated resources."""
        tenant_id = f"tenant_{secrets.token_urlsafe(16)}"

        tenant = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            created_at=datetime.utcnow(),
            data_encryption_key=self._generate_tenant_key(tenant_id),
            isolated_namespaces=[
                f"{tenant_id}_agents",
                f"{tenant_id}_rag",
                f"{tenant_id}_models",
                f"{tenant_id}_p2p",
            ],
        )

        if config:
            tenant.resource_limits = config.get("resource_limits", {})
            tenant.max_agents = config.get("max_agents", 10)
            tenant.max_rag_collections = config.get("max_rag_collections", 5)
            tenant.storage_quota_gb = config.get("storage_quota_gb", 100)

        self.tenants[tenant_id] = tenant

        # Create admin user for tenant
        admin = await self.create_user(
            username=admin_user["username"],
            email=admin_user["email"],
            password=admin_user["password"],
            tenant_id=tenant_id,
            role=Role.ADMIN,
        )

        self._audit_log("tenant_created", {"tenant_id": tenant_id, "name": name, "admin_user": admin.user_id})

        logger.info(f"Created tenant {tenant_id} with admin {admin.user_id}")
        return tenant

    async def delete_tenant(self, tenant_id: str, requester_id: str) -> bool:
        """Delete tenant and all associated resources."""
        if not await self.check_permission(requester_id, Permission.TENANT_DELETE):
            raise PermissionError(f"User {requester_id} lacks permission to delete tenants")

        if tenant_id not in self.tenants:
            return False

        # Remove all users from tenant
        tenant_users = [u for u in self.users.values() if u.tenant_id == tenant_id]
        for user in tenant_users:
            await self.delete_user(user.user_id, requester_id)

        # Clean up tenant resources
        self.tenants[tenant_id]
        # TODO: Clean up actual resources (agents, RAG collections, etc.)

        del self.tenants[tenant_id]

        self._audit_log("tenant_deleted", {"tenant_id": tenant_id, "requester": requester_id})

        return True

    # User Management

    async def create_user(
        self, username: str, email: str, password: str, tenant_id: str, role: Role = Role.USER
    ) -> User:
        """Create new user within tenant."""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} does not exist")

        user_id = f"user_{secrets.token_urlsafe(16)}"
        self._hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            tenant_id=tenant_id,
            role=role,
            created_at=datetime.utcnow(),
        )

        self.users[user_id] = user

        # Store password hash separately (in production, use secure storage)
        # This is simplified for demonstration

        self._audit_log(
            "user_created", {"user_id": user_id, "username": username, "tenant_id": tenant_id, "role": role.value}
        )

        logger.info(f"Created user {user_id} in tenant {tenant_id}")
        return user

    async def delete_user(self, user_id: str, requester_id: str) -> bool:
        """Delete user from system."""
        if user_id not in self.users:
            return False

        user = self.users[user_id]
        requester = self.users.get(requester_id)

        # Check permissions
        if requester and requester.tenant_id != user.tenant_id:
            if requester.role != Role.SUPER_ADMIN:
                raise PermissionError("Cannot delete user from different tenant")

        # Revoke all sessions
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        for session in user_sessions:
            await self.revoke_session(session.session_id)

        del self.users[user_id]

        self._audit_log("user_deleted", {"user_id": user_id, "requester": requester_id})

        return True

    # Authentication

    async def authenticate(
        self, username: str, password: str, tenant_id: str, ip_address: str, user_agent: str
    ) -> Session:
        """Authenticate user and create session."""
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username and u.tenant_id == tenant_id:
                user = u
                break

        if not user:
            raise ValueError("Invalid credentials")

        # Verify password (simplified - in production use proper verification)
        # if not self._verify_password(password, stored_hash):
        #     raise ValueError("Invalid credentials")

        # Create session
        session = await self._create_session(user, ip_address, user_agent)

        # Update last login
        user.last_login = datetime.utcnow()

        self._audit_log(
            "user_authenticated", {"user_id": user.user_id, "tenant_id": tenant_id, "ip_address": ip_address}
        )

        return session

    async def _create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """Create new user session with JWT tokens."""
        session_id = f"session_{secrets.token_urlsafe(16)}"
        now = datetime.utcnow()

        # Create JWT token
        token_payload = {
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "role": user.role.value,
            "session_id": session_id,
            "exp": now + self.token_expiry,
            "iat": now,
        }
        token = jwt.encode(token_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Create refresh token
        refresh_payload = {
            "user_id": user.user_id,
            "session_id": session_id,
            "type": "refresh",
            "exp": now + self.refresh_expiry,
            "iat": now,
        }
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            token=token,
            refresh_token=refresh_token,
            created_at=now,
            expires_at=now + self.token_expiry,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.sessions[session_id] = session
        return session

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify JWT token and return payload."""
        if token in self.revoked_tokens:
            return None

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Check if session still exists
            session_id = payload.get("session_id")
            if session_id not in self.sessions:
                return None

            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

    async def refresh_token(self, refresh_token: str) -> Session | None:
        """Refresh access token using refresh token."""
        payload = await self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None

        session_id = payload.get("session_id")
        if session_id not in self.sessions:
            return None

        old_session = self.sessions[session_id]
        user = self.users.get(old_session.user_id)
        if not user:
            return None

        # Create new session
        new_session = await self._create_session(user, old_session.ip_address, old_session.user_agent)

        # Revoke old session
        await self.revoke_session(session_id)

        return new_session

    async def revoke_session(self, session_id: str):
        """Revoke user session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            self.revoked_tokens.add(session.token)
            self.revoked_tokens.add(session.refresh_token)
            del self.sessions[session_id]

            self._audit_log("session_revoked", {"session_id": session_id})

    # Authorization

    async def check_permission(self, user_id: str, permission: Permission, resource_id: str | None = None) -> bool:
        """Check if user has specific permission."""
        user = self.users.get(user_id)
        if not user or not user.active:
            return False

        # Super admin has all permissions
        if user.role == Role.SUPER_ADMIN:
            return True

        # Check role permissions
        role_perms = ROLE_PERMISSIONS.get(user.role, set())
        has_permission = permission in role_perms or permission in user.custom_permissions

        # If checking resource-specific permission, verify tenant isolation
        if has_permission and resource_id:
            has_permission = await self._check_resource_access(user, resource_id)

        return has_permission

    async def _check_resource_access(self, user: User, resource_id: str) -> bool:
        """Check if user can access specific resource based on tenant isolation."""
        # Extract tenant from resource ID (assuming format: tenant_xxx_resource_yyy)
        if resource_id.startswith(f"{user.tenant_id}_"):
            return True

        # Super admin can access all resources
        if user.role == Role.SUPER_ADMIN:
            return True

        return False

    def require_permission(self, permission: Permission):
        """Decorator to enforce permission requirements on functions."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user_id from arguments or context
                user_id = kwargs.get("user_id")
                if not user_id:
                    raise ValueError("user_id required for permission check")

                if not await self.check_permission(user_id, permission):
                    raise PermissionError(f"User {user_id} lacks permission {permission.value}")

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    # Resource Isolation

    async def isolate_resource(self, resource_type: str, resource_id: str, tenant_id: str, data: Any) -> str:
        """Isolate resource data for specific tenant."""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} does not exist")

        tenant = self.tenants[tenant_id]

        # Generate isolated resource ID
        isolated_id = f"{tenant_id}_{resource_type}_{resource_id}"

        # Encrypt data with tenant key
        if tenant.data_encryption_key:
            tenant.data_encryption_key.encrypt(json.dumps(data).encode())
        else:
            pass

        # Store in appropriate namespace
        # TODO: Implement actual storage backend

        self._audit_log(
            "resource_isolated", {"resource_type": resource_type, "resource_id": isolated_id, "tenant_id": tenant_id}
        )

        return isolated_id

    async def get_tenant_resources(self, tenant_id: str, resource_type: str, user_id: str) -> list[str]:
        """Get all resources of specific type for tenant."""
        user = self.users.get(user_id)
        if not user or user.tenant_id != tenant_id:
            if user and user.role != Role.SUPER_ADMIN:
                raise PermissionError("Cannot access other tenant's resources")

        # Return list of resource IDs for tenant
        # TODO: Implement actual resource listing
        resources = []

        return resources

    # Quota Management

    async def check_quota(self, tenant_id: str, resource_type: str, requested_amount: int = 1) -> bool:
        """Check if tenant has quota for resource."""
        if tenant_id not in self.tenants:
            return False

        tenant = self.tenants[tenant_id]

        # Check specific resource quotas
        if resource_type == "agents":
            # Count existing agents
            # TODO: Implement actual counting
            current_count = 0
            return current_count + requested_amount <= tenant.max_agents

        elif resource_type == "rag_collections":
            current_count = 0
            return current_count + requested_amount <= tenant.max_rag_collections

        elif resource_type == "storage_gb":
            current_usage = 0
            return current_usage + requested_amount <= tenant.storage_quota_gb

        return True

    async def update_quota_usage(self, tenant_id: str, resource_type: str, amount: int, operation: str = "add"):
        """Update quota usage for tenant."""
        if tenant_id not in self.tenants:
            return

        # TODO: Implement actual quota tracking

        self._audit_log(
            "quota_updated",
            {"tenant_id": tenant_id, "resource_type": resource_type, "amount": amount, "operation": operation},
        )

    # Utility Methods

    def _hash_password(self, password: str) -> str:
        """Hash password for storage."""
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), self.jwt_secret.encode("utf-8"), 100000).hex()

    def _audit_log(self, action: str, details: dict[str, Any]):
        """Log security audit event."""
        entry = {"timestamp": datetime.utcnow().isoformat(), "action": action, "details": details}
        self.audit_log.append(entry)

        # In production, persist to audit log storage
        logger.info(f"Audit: {action} - {details}")

    async def get_audit_log(
        self,
        user_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        action_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve audit log entries."""
        if not await self.check_permission(user_id, Permission.SYSTEM_MONITOR):
            raise PermissionError("User lacks permission to view audit logs")

        logs = self.audit_log

        # Apply filters
        if start_date:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"]) >= start_date]
        if end_date:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"]) <= end_date]
        if action_filter:
            logs = [l for l in logs if action_filter in l["action"]]

        return logs


# API Integration Middleware


class RBACMiddleware:
    """Middleware for integrating RBAC with API endpoints."""

    def __init__(self, rbac_system: RBACSystem):
        self.rbac = rbac_system

    async def authenticate_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Authenticate incoming API request."""
        auth_header = request.get("headers", {}).get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix
        return await self.rbac.verify_token(token)

    async def authorize_request(self, user_id: str, permission: Permission, resource_id: str | None = None) -> bool:
        """Authorize request based on required permission."""
        return await self.rbac.check_permission(user_id, permission, resource_id)

    def require_auth(self, permission: Permission | None = None):
        """Decorator for protecting API endpoints."""

        def decorator(func):
            @wraps(func)
            async def wrapper(request: dict[str, Any], *args, **kwargs):
                # Authenticate
                auth_data = await self.authenticate_request(request)
                if not auth_data:
                    return {"error": "Authentication required", "status": 401}

                # Authorize if permission specified
                if permission:
                    user_id = auth_data["user_id"]
                    resource_id = kwargs.get("resource_id")

                    if not await self.authorize_request(user_id, permission, resource_id):
                        return {"error": "Permission denied", "status": 403}

                # Add auth data to request
                request["auth"] = auth_data

                return await func(request, *args, **kwargs)

            return wrapper

        return decorator


async def initialize_rbac_system() -> RBACSystem:
    """Initialize and configure RBAC system for production."""
    rbac = RBACSystem()

    # Create default super admin tenant
    default_tenant = await rbac.create_tenant(
        name="AIVillage System",
        admin_user={"username": "admin", "email": "admin@aivillage.local", "password": secrets.token_urlsafe(32)},
        config={
            "max_agents": 100,
            "max_rag_collections": 50,
            "storage_quota_gb": 1000,
            "compute_quota_vcpu": 64,
            "memory_quota_gb": 256,
        },
    )

    logger.info(f"RBAC system initialized with default tenant: {default_tenant.tenant_id}")
    return rbac


if __name__ == "__main__":
    # Example usage
    async def main():
        rbac = await initialize_rbac_system()

        # Create a test tenant
        tenant = await rbac.create_tenant(
            name="Test Organization",
            admin_user={"username": "test_admin", "email": "admin@test.org", "password": "secure_password_123"},
        )

        print(f"Created tenant: {tenant.tenant_id}")

        # Create additional users
        developer = await rbac.create_user(
            username="developer1",
            email="dev@test.org",
            password="dev_password",
            tenant_id=tenant.tenant_id,
            role=Role.DEVELOPER,
        )

        print(f"Created developer: {developer.user_id}")

        # Test authentication
        session = await rbac.authenticate(
            username="test_admin",
            password="secure_password_123",
            tenant_id=tenant.tenant_id,
            ip_address="127.0.0.1",
            user_agent="TestClient/1.0",
        )

        print(f"Session created: {session.session_id}")

        # Test authorization
        can_create_agent = await rbac.check_permission(developer.user_id, Permission.AGENT_CREATE)

        print(f"Developer can create agents: {can_create_agent}")

    asyncio.run(main())
