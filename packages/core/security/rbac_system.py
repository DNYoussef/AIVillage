# Role-Based Access Control (RBAC) Security System
# Production-ready RBAC implementation with fine-grained permissions

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from collections import defaultdict
import json
import hashlib


logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """Types of permissions in the system."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"
    LIST = "list"
    MANAGE = "manage"
    CONFIGURE = "configure"


class ResourceType(Enum):
    """Types of resources that can be protected."""
    AGENT = "agent"
    SWARM = "swarm"
    NETWORK = "network"
    DATA = "data"
    CONFIG = "config"
    SYSTEM = "system"
    API = "api"
    FILE = "file"
    SERVICE = "service"
    DATABASE = "database"


@dataclass
class Permission:
    """Individual permission definition."""
    
    permission_type: PermissionType
    resource_type: ResourceType
    resource_id: Optional[str] = None  # Specific resource ID, None for all
    conditions: Dict[str, Any] = field(default_factory=dict)  # Additional conditions
    
    def __post_init__(self):
        if not self.conditions:
            self.conditions = {}
    
    def matches(self, permission_type: PermissionType, resource_type: ResourceType, resource_id: str = None) -> bool:
        """Check if this permission matches the requested access."""
        # Check permission type match
        if self.permission_type != permission_type:
            # Special case: ADMIN permission grants all other permissions
            if self.permission_type != PermissionType.ADMIN:
                return False
        
        # Check resource type match
        if self.resource_type != resource_type:
            return False
        
        # Check resource ID match (None means all resources)
        if self.resource_id is not None and self.resource_id != resource_id:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize permission to dictionary."""
        return {
            "permission_type": self.permission_type.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "conditions": self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Deserialize permission from dictionary."""
        return cls(
            permission_type=PermissionType(data["permission_type"]),
            resource_type=ResourceType(data["resource_type"]),
            resource_id=data.get("resource_id"),
            conditions=data.get("conditions", {})
        )
    
    def __str__(self) -> str:
        resource_part = f"{self.resource_type.value}"
        if self.resource_id:
            resource_part += f":{self.resource_id}"
        return f"{self.permission_type.value}:{resource_part}"


@dataclass
class Role:
    """Role definition with permissions and metadata."""
    
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_system_role: bool = False  # System roles cannot be deleted
    parent_roles: Set[str] = field(default_factory=set)  # Role inheritance
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        self.permissions.add(permission)
        self.updated_at = time.time()
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.updated_at = time.time()
    
    def has_permission(self, permission_type: PermissionType, resource_type: ResourceType, resource_id: str = None) -> bool:
        """Check if role has specific permission."""
        for permission in self.permissions:
            if permission.matches(permission_type, resource_type, resource_id):
                return True
        return False
    
    def get_effective_permissions(self, rbac_system: 'RBACSystem') -> Set[Permission]:
        """Get all permissions including inherited from parent roles."""
        effective_permissions = self.permissions.copy()
        
        # Add permissions from parent roles
        for parent_role_name in self.parent_roles:
            parent_role = rbac_system.get_role(parent_role_name)
            if parent_role:
                parent_permissions = parent_role.get_effective_permissions(rbac_system)
                effective_permissions.update(parent_permissions)
        
        return effective_permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize role to dictionary."""
        return {
            "name": self.name,
            "permissions": [p.to_dict() for p in self.permissions],
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_system_role": self.is_system_role,
            "parent_roles": list(self.parent_roles)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Deserialize role from dictionary."""
        permissions = {Permission.from_dict(p) for p in data.get("permissions", [])}
        return cls(
            name=data["name"],
            permissions=permissions,
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            is_system_role=data.get("is_system_role", False),
            parent_roles=set(data.get("parent_roles", []))
        )


@dataclass
class User:
    """User definition with roles and metadata."""
    
    user_id: str
    username: str
    roles: Set[str] = field(default_factory=set)  # Role names
    attributes: Dict[str, Any] = field(default_factory=dict)  # Custom user attributes
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    is_active: bool = True
    is_system_user: bool = False
    
    def add_role(self, role_name: str):
        """Add role to user."""
        self.roles.add(role_name)
    
    def remove_role(self, role_name: str):
        """Remove role from user."""
        if role_name in self.roles:
            self.roles.remove(role_name)
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role."""
        return role_name in self.roles
    
    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize user to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "roles": list(self.roles),
            "attributes": self.attributes,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "is_system_user": self.is_system_user
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Deserialize user from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            roles=set(data.get("roles", [])),
            attributes=data.get("attributes", {}),
            created_at=data.get("created_at", time.time()),
            last_login=data.get("last_login"),
            is_active=data.get("is_active", True),
            is_system_user=data.get("is_system_user", False)
        )


class AccessDecision(Enum):
    """Access control decision types."""
    ALLOW = "allow"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class AccessContext:
    """Context for access control decisions."""
    
    user_id: str
    permission_type: PermissionType
    resource_type: ResourceType
    resource_id: Optional[str] = None
    request_time: float = field(default_factory=time.time)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize access context to dictionary."""
        return {
            "user_id": self.user_id,
            "permission_type": self.permission_type.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "request_time": self.request_time,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "additional_context": self.additional_context
        }


@dataclass
class AccessResult:
    """Result of access control evaluation."""
    
    decision: AccessDecision
    user_id: str
    context: AccessContext
    matched_permissions: List[Permission] = field(default_factory=list)
    reason: str = ""
    evaluation_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize access result to dictionary."""
        return {
            "decision": self.decision.value,
            "user_id": self.user_id,
            "context": self.context.to_dict(),
            "matched_permissions": [p.to_dict() for p in self.matched_permissions],
            "reason": self.reason,
            "evaluation_time_ms": self.evaluation_time_ms
        }


class PolicyEvaluator(ABC):
    """Abstract base class for policy evaluators."""
    
    @abstractmethod
    def evaluate(self, context: AccessContext, user: User, rbac_system: 'RBACSystem') -> AccessResult:
        """Evaluate access policy for given context."""
        pass


class DefaultPolicyEvaluator(PolicyEvaluator):
    """Default RBAC policy evaluator."""
    
    def evaluate(self, context: AccessContext, user: User, rbac_system: 'RBACSystem') -> AccessResult:
        """Evaluate access based on user roles and permissions."""
        start_time = time.time()
        
        # Check if user is active
        if not user.is_active:
            return AccessResult(
                decision=AccessDecision.DENY,
                user_id=context.user_id,
                context=context,
                reason="User account is inactive",
                evaluation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Collect all permissions from user roles
        all_permissions = set()
        matched_permissions = []
        
        for role_name in user.roles:
            role = rbac_system.get_role(role_name)
            if role:
                effective_permissions = role.get_effective_permissions(rbac_system)
                all_permissions.update(effective_permissions)
        
        # Check if any permission matches the request
        for permission in all_permissions:
            if permission.matches(context.permission_type, context.resource_type, context.resource_id):
                matched_permissions.append(permission)
        
        if matched_permissions:
            decision = AccessDecision.ALLOW
            reason = f"Access granted via {len(matched_permissions)} matching permission(s)"
        else:
            decision = AccessDecision.DENY
            reason = "No matching permissions found"
        
        return AccessResult(
            decision=decision,
            user_id=context.user_id,
            context=context,
            matched_permissions=matched_permissions,
            reason=reason,
            evaluation_time_ms=(time.time() - start_time) * 1000
        )


class RBACSystem:
    """Complete Role-Based Access Control system."""
    
    def __init__(self, policy_evaluator: PolicyEvaluator = None):
        """
        Initialize RBAC system.
        
        Args:
            policy_evaluator: Custom policy evaluator (defaults to DefaultPolicyEvaluator)
        """
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.policy_evaluator = policy_evaluator or DefaultPolicyEvaluator()
        self.access_log: List[AccessResult] = []
        self.max_log_entries = 10000
        
        # Create default system roles
        self._create_system_roles()
        
        logger.info("Initialized RBAC system with default roles")
    
    def _create_system_roles(self):
        """Create default system roles."""
        # Administrator role - all permissions
        admin_permissions = set()
        for resource_type in ResourceType:
            for permission_type in PermissionType:
                admin_permissions.add(Permission(permission_type, resource_type))
        
        admin_role = Role(
            name="administrator",
            permissions=admin_permissions,
            description="Full system administrator with all permissions",
            is_system_role=True
        )
        self.roles["administrator"] = admin_role
        
        # Agent operator role - basic agent operations
        agent_operator_permissions = {
            Permission(PermissionType.READ, ResourceType.AGENT),
            Permission(PermissionType.WRITE, ResourceType.AGENT),
            Permission(PermissionType.EXECUTE, ResourceType.AGENT),
            Permission(PermissionType.READ, ResourceType.SWARM),
            Permission(PermissionType.LIST, ResourceType.AGENT),
            Permission(PermissionType.LIST, ResourceType.SWARM)
        }
        
        agent_role = Role(
            name="agent_operator",
            permissions=agent_operator_permissions,
            description="Basic agent operations and monitoring",
            is_system_role=True
        )
        self.roles["agent_operator"] = agent_role
        
        # Read-only role
        readonly_permissions = set()
        for resource_type in ResourceType:
            readonly_permissions.add(Permission(PermissionType.READ, resource_type))
            readonly_permissions.add(Permission(PermissionType.LIST, resource_type))
        
        readonly_role = Role(
            name="readonly",
            permissions=readonly_permissions,
            description="Read-only access to all resources",
            is_system_role=True
        )
        self.roles["readonly"] = readonly_role
    
    def create_user(self, user_id: str, username: str, roles: List[str] = None, attributes: Dict[str, Any] = None) -> User:
        """
        Create a new user.
        
        Args:
            user_id: Unique user identifier
            username: Human-readable username
            roles: List of role names to assign
            attributes: Additional user attributes
        
        Returns:
            Created User object
        """
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        
        user = User(
            user_id=user_id,
            username=username,
            roles=set(roles or []),
            attributes=attributes or {}
        )
        
        self.users[user_id] = user
        logger.info(f"Created user {username} ({user_id}) with roles: {roles or []}")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user attributes."""
        user = self.get_user(user_id)
        if not user:
            return None
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        logger.info(f"Updated user {user_id}")
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        user = self.get_user(user_id)
        if not user:
            return False
        
        if user.is_system_user:
            raise ValueError("Cannot delete system user")
        
        del self.users[user_id]
        logger.info(f"Deleted user {user_id}")
        return True
    
    def create_role(self, name: str, permissions: List[Permission] = None, description: str = "", parent_roles: List[str] = None) -> Role:
        """
        Create a new role.
        
        Args:
            name: Role name
            permissions: List of permissions
            description: Role description
            parent_roles: List of parent role names for inheritance
        
        Returns:
            Created Role object
        """
        if name in self.roles:
            raise ValueError(f"Role {name} already exists")
        
        role = Role(
            name=name,
            permissions=set(permissions or []),
            description=description,
            parent_roles=set(parent_roles or [])
        )
        
        self.roles[name] = role
        logger.info(f"Created role {name} with {len(permissions or [])} permissions")
        return role
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self.roles.get(name)
    
    def update_role(self, name: str, **kwargs) -> Optional[Role]:
        """Update role attributes."""
        role = self.get_role(name)
        if not role:
            return None
        
        if role.is_system_role and name in ["administrator", "agent_operator", "readonly"]:
            raise ValueError("Cannot modify system role")
        
        for key, value in kwargs.items():
            if hasattr(role, key):
                setattr(role, key, value)
        
        role.updated_at = time.time()
        logger.info(f"Updated role {name}")
        return role
    
    def delete_role(self, name: str) -> bool:
        """Delete role."""
        role = self.get_role(name)
        if not role:
            return False
        
        if role.is_system_role:
            raise ValueError("Cannot delete system role")
        
        # Remove role from all users
        for user in self.users.values():
            user.remove_role(name)
        
        del self.roles[name]
        logger.info(f"Deleted role {name}")
        return True
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign role to user."""
        user = self.get_user(user_id)
        role = self.get_role(role_name)
        
        if not user or not role:
            return False
        
        user.add_role(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")
        return True
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke role from user."""
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.remove_role(role_name)
        logger.info(f"Revoked role {role_name} from user {user_id}")
        return True
    
    def check_access(self, user_id: str, permission_type: PermissionType, resource_type: ResourceType, resource_id: str = None, **context_kwargs) -> AccessResult:
        """
        Check if user has access to perform operation on resource.
        
        Args:
            user_id: User identifier
            permission_type: Type of permission required
            resource_type: Type of resource being accessed
            resource_id: Specific resource identifier (optional)
            **context_kwargs: Additional context parameters
        
        Returns:
            AccessResult with decision and details
        """
        start_time = time.time()
        
        user = self.get_user(user_id)
        if not user:
            result = AccessResult(
                decision=AccessDecision.DENY,
                user_id=user_id,
                context=AccessContext(
                    user_id=user_id,
                    permission_type=permission_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    **context_kwargs
                ),
                reason="User not found",
                evaluation_time_ms=(time.time() - start_time) * 1000
            )
        else:
            context = AccessContext(
                user_id=user_id,
                permission_type=permission_type,
                resource_type=resource_type,
                resource_id=resource_id,
                **context_kwargs
            )
            
            result = self.policy_evaluator.evaluate(context, user, self)
        
        # Log access attempt
        self._log_access(result)
        
        return result
    
    def _log_access(self, result: AccessResult):
        """Log access attempt."""
        self.access_log.append(result)
        
        # Trim log if too large
        if len(self.access_log) > self.max_log_entries:
            self.access_log = self.access_log[-self.max_log_entries//2:]
        
        # Log to system logger
        log_level = logging.INFO if result.decision == AccessDecision.ALLOW else logging.WARNING
        logger.log(log_level, f"Access {result.decision.value}: {result.user_id} -> {result.context.permission_type.value}:{result.context.resource_type.value} ({result.reason})")
    
    def get_access_log(self, user_id: str = None, limit: int = 100) -> List[AccessResult]:
        """Get access log entries."""
        log_entries = self.access_log
        
        if user_id:
            log_entries = [entry for entry in log_entries if entry.user_id == user_id]
        
        return log_entries[-limit:]
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user."""
        user = self.get_user(user_id)
        if not user:
            return set()
        
        all_permissions = set()
        for role_name in user.roles:
            role = self.get_role(role_name)
            if role:
                effective_permissions = role.get_effective_permissions(self)
                all_permissions.update(effective_permissions)
        
        return all_permissions
    
    def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())
    
    def list_roles(self) -> List[Role]:
        """List all roles."""
        return list(self.roles.values())
    
    def export_config(self) -> Dict[str, Any]:
        """Export RBAC configuration to dictionary."""
        return {
            "users": {uid: user.to_dict() for uid, user in self.users.items()},
            "roles": {name: role.to_dict() for name, role in self.roles.items()},
            "export_timestamp": time.time()
        }
    
    def import_config(self, config: Dict[str, Any]):
        """Import RBAC configuration from dictionary."""
        # Import roles first (for dependencies)
        roles_data = config.get("roles", {})
        for role_name, role_data in roles_data.items():
            if role_name not in self.roles or not self.roles[role_name].is_system_role:
                self.roles[role_name] = Role.from_dict(role_data)
        
        # Import users
        users_data = config.get("users", {})
        for user_id, user_data in users_data.items():
            if user_id not in self.users or not self.users[user_id].is_system_user:
                self.users[user_id] = User.from_dict(user_data)
        
        logger.info(f"Imported RBAC configuration: {len(users_data)} users, {len(roles_data)} roles")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get RBAC system statistics."""
        active_users = sum(1 for user in self.users.values() if user.is_active)
        system_roles = sum(1 for role in self.roles.values() if role.is_system_role)
        
        recent_access_attempts = len([entry for entry in self.access_log if time.time() - entry.context.request_time < 3600])  # Last hour
        successful_access = len([entry for entry in self.access_log if entry.decision == AccessDecision.ALLOW and time.time() - entry.context.request_time < 3600])
        
        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "total_roles": len(self.roles),
            "system_roles": system_roles,
            "access_log_entries": len(self.access_log),
            "recent_access_attempts": recent_access_attempts,
            "recent_successful_access": successful_access,
            "access_success_rate": successful_access / recent_access_attempts if recent_access_attempts > 0 else 0
        }


# Backward compatibility - try to import from actual infrastructure locations first
try:
    from core.security.rbac_system import *
except ImportError:
    try:
        from infrastructure.shared.security.rbac_system import *
    except ImportError:
        # Use the implementations defined above
        pass
