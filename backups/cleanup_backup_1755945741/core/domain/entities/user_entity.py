"""
User Domain Entity

Represents users who interact with the AIVillage system,
with roles, permissions, and activity tracking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class UserRole(Enum):
    """User roles in the system"""

    GUEST = "guest"
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class UserId:
    """User identifier value object"""

    value: str

    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("UserId must be a non-empty string")

    @classmethod
    def generate(cls) -> UserId:
        """Generate new unique user ID"""
        return cls(str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass
class User:
    """
    Core User domain entity

    Represents users who interact with agents and the system,
    with role-based permissions and activity tracking.
    """

    # Identity
    id: UserId
    username: str
    email: str | None = None

    # Access control
    role: UserRole = UserRole.USER
    active: bool = True

    # Profile information
    display_name: str | None = None
    preferences: dict[str, Any] = field(default_factory=dict)

    # Activity tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_login: datetime | None = None
    login_count: int = 0

    # Session and interaction history
    current_session_id: str | None = None
    total_interactions: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate user invariants"""
        if not self.username.strip():
            raise ValueError("Username cannot be empty")

        if len(self.username) < 3:
            raise ValueError("Username must be at least 3 characters")

        if self.email and "@" not in self.email:
            raise ValueError("Invalid email format")

    def login(self, session_id: str) -> None:
        """Record user login"""
        if not self.active:
            raise ValueError("Cannot login inactive user")

        self.last_login = datetime.now()
        self.login_count += 1
        self.current_session_id = session_id

    def logout(self) -> None:
        """Record user logout"""
        self.current_session_id = None

    def deactivate(self) -> None:
        """Deactivate user account"""
        self.active = False
        self.current_session_id = None

    def reactivate(self) -> None:
        """Reactivate user account"""
        self.active = True

    def update_role(self, new_role: UserRole) -> None:
        """Update user role"""
        self.role = new_role

    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference"""
        self.preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        return self.preferences.get(key, default)

    def record_interaction(self) -> None:
        """Record user interaction with system"""
        self.total_interactions += 1

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        role_permissions = {
            UserRole.GUEST: ["read"],
            UserRole.USER: ["read", "interact", "create_session"],
            UserRole.MODERATOR: ["read", "interact", "create_session", "moderate", "manage_users"],
            UserRole.ADMIN: ["read", "interact", "create_session", "moderate", "manage_users", "admin"],
            UserRole.SYSTEM: ["all"],
        }

        permissions = role_permissions.get(self.role, [])
        return permission in permissions or "all" in permissions

    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.role in [UserRole.ADMIN, UserRole.SYSTEM]

    def is_moderator_or_above(self) -> bool:
        """Check if user has moderator or higher privileges"""
        return self.role in [UserRole.MODERATOR, UserRole.ADMIN, UserRole.SYSTEM]

    def get_display_name(self) -> str:
        """Get user's display name"""
        return self.display_name or self.username

    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary representation"""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "active": self.active,
            "display_name": self.display_name,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "login_count": self.login_count,
            "current_session_id": self.current_session_id,
            "total_interactions": self.total_interactions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> User:
        """Create user from dictionary representation"""
        return cls(
            id=UserId(data["id"]),
            username=data["username"],
            email=data.get("email"),
            role=UserRole(data["role"]),
            active=data.get("active", True),
            display_name=data.get("display_name"),
            preferences=data.get("preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None,
            login_count=data.get("login_count", 0),
            current_session_id=data.get("current_session_id"),
            total_interactions=data.get("total_interactions", 0),
            metadata=data.get("metadata", {}),
        )
