#!/usr/bin/env python3
"""Role-Based Access Control (RBAC) System.

Provides comprehensive RBAC with hierarchical roles, permissions, and
CODEX-compliant security controls for the AIVillage system.
"""

from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""

    # Digital Twin permissions
    DIGITAL_TWIN_READ = "digital_twin:read"
    DIGITAL_TWIN_WRITE = "digital_twin:write"
    DIGITAL_TWIN_DELETE = "digital_twin:delete"
    DIGITAL_TWIN_ADMIN = "digital_twin:admin"

    # Evolution Metrics permissions
    EVOLUTION_METRICS_READ = "evolution_metrics:read"
    EVOLUTION_METRICS_WRITE = "evolution_metrics:write"
    EVOLUTION_METRICS_ADMIN = "evolution_metrics:admin"

    # RAG Pipeline permissions
    RAG_PIPELINE_QUERY = "rag_pipeline:query"
    RAG_PIPELINE_WRITE = "rag_pipeline:write"
    RAG_PIPELINE_ADMIN = "rag_pipeline:admin"

    # P2P Network permissions
    P2P_NETWORK_CONNECT = "p2p_network:connect"
    P2P_NETWORK_ADMIN = "p2p_network:admin"

    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITORING = "system:monitoring"
    SYSTEM_BACKUP = "system:backup"

    # Compliance permissions
    COMPLIANCE_VIEW = "compliance:view"
    COMPLIANCE_MANAGE = "compliance:manage"
    COMPLIANCE_AUDIT = "compliance:audit"

    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"


class Role(Enum):
    """System roles with hierarchical structure."""

    # Basic roles
    GUEST = "guest"
    USER = "user"

    # Specialized roles
    RESEARCHER = "researcher"
    EDUCATOR = "educator"
    STUDENT = "student"

    # Administrative roles
    MODERATOR = "moderator"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

    # Compliance roles
    COMPLIANCE_OFFICER = "compliance_officer"
    DATA_PROTECTION_OFFICER = "data_protection_officer"

    # System roles
    SERVICE_ACCOUNT = "service_account"
    API_CLIENT = "api_client"


class AccessDeniedException(Exception):
    """Raised when access is denied."""


class RBACSystem:
    """Role-Based Access Control system."""

    def __init__(self, db_path: str | None = None):
        """Initialize RBAC system.

        Args:
            db_path: Path to RBAC database
        """
        self.db_path = db_path or os.getenv("RBAC_DB_PATH", "./data/rbac.db")

        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Load default role hierarchy and permissions
        self._init_default_roles()

        logger.info(f"RBAC system initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize RBAC database schema."""
        with self._get_connection() as conn:
            # Roles table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    parent_role TEXT,
                    is_system_role BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Permissions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    resource TEXT,
                    action TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Role-Permission assignments
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS role_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role_name TEXT NOT NULL,
                    permission_name TEXT NOT NULL,
                    granted_by TEXT,
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,

                    FOREIGN KEY (role_name) REFERENCES roles (name)
                        ON DELETE CASCADE ON UPDATE CASCADE,
                    FOREIGN KEY (permission_name) REFERENCES permissions (name)
                        ON DELETE CASCADE ON UPDATE CASCADE,

                    UNIQUE(role_name, permission_name)
                )
            """
            )

            # Users table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    password_hash TEXT,
                    password_salt TEXT,

                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    last_login TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Compliance fields
                    requires_coppa_consent BOOLEAN DEFAULT 0,
                    coppa_consent_date TIMESTAMP,
                    gdpr_consent BOOLEAN DEFAULT 0,
                    gdpr_consent_date TIMESTAMP,
                    data_retention_expires TIMESTAMP
                )
            """
            )

            # User-Role assignments
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role_name TEXT NOT NULL,
                    assigned_by TEXT,
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,

                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                        ON DELETE CASCADE ON UPDATE CASCADE,
                    FOREIGN KEY (role_name) REFERENCES roles (name)
                        ON DELETE CASCADE ON UPDATE CASCADE,

                    UNIQUE(user_id, role_name)
                )
            """
            )

            # Access log
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    permission_required TEXT,
                    access_granted BOOLEAN,
                    denial_reason TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    session_id TEXT
                )
            """
            )

            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions (role_name)",
                "CREATE INDEX IF NOT EXISTS idx_role_permissions_permission ON role_permissions (permission_name)",
                "CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles (user_id)",
                "CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles (role_name)",
                "CREATE INDEX IF NOT EXISTS idx_access_log_user ON access_log (user_id)",
                "CREATE INDEX IF NOT EXISTS idx_access_log_timestamp ON access_log (timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)",
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)",
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

    def _init_default_roles(self):
        """Initialize default roles and permissions."""
        # Define default role hierarchy and permissions
        role_configs = {
            Role.GUEST: {
                "display_name": "Guest User",
                "description": "Limited read-only access",
                "parent_role": None,
                "permissions": [],
            },
            Role.USER: {
                "display_name": "Regular User",
                "description": "Standard user with basic access",
                "parent_role": Role.GUEST,
                "permissions": [
                    Permission.DIGITAL_TWIN_READ,
                    Permission.EVOLUTION_METRICS_READ,
                    Permission.RAG_PIPELINE_QUERY,
                ],
            },
            Role.STUDENT: {
                "display_name": "Student",
                "description": "Student with learning access",
                "parent_role": Role.USER,
                "permissions": [
                    Permission.DIGITAL_TWIN_WRITE,  # Can update own profile
                ],
            },
            Role.EDUCATOR: {
                "display_name": "Educator",
                "description": "Educator with teaching access",
                "parent_role": Role.USER,
                "permissions": [
                    Permission.DIGITAL_TWIN_WRITE,
                    Permission.EVOLUTION_METRICS_WRITE,
                    Permission.RAG_PIPELINE_WRITE,
                    Permission.COMPLIANCE_VIEW,
                ],
            },
            Role.RESEARCHER: {
                "display_name": "Researcher",
                "description": "Researcher with analysis access",
                "parent_role": Role.USER,
                "permissions": [
                    Permission.EVOLUTION_METRICS_WRITE,
                    Permission.EVOLUTION_METRICS_ADMIN,
                    Permission.RAG_PIPELINE_WRITE,
                    Permission.SYSTEM_MONITORING,
                ],
            },
            Role.MODERATOR: {
                "display_name": "Moderator",
                "description": "Content moderator",
                "parent_role": Role.USER,
                "permissions": [
                    Permission.DIGITAL_TWIN_WRITE,
                    Permission.DIGITAL_TWIN_DELETE,
                    Permission.USER_READ,
                    Permission.COMPLIANCE_VIEW,
                ],
            },
            Role.OPERATOR: {
                "display_name": "System Operator",
                "description": "System operation and maintenance",
                "parent_role": Role.MODERATOR,
                "permissions": [
                    Permission.DIGITAL_TWIN_ADMIN,
                    Permission.EVOLUTION_METRICS_ADMIN,
                    Permission.RAG_PIPELINE_ADMIN,
                    Permission.P2P_NETWORK_CONNECT,
                    Permission.SYSTEM_MONITORING,
                    Permission.SYSTEM_BACKUP,
                    Permission.USER_WRITE,
                ],
            },
            Role.COMPLIANCE_OFFICER: {
                "display_name": "Compliance Officer",
                "description": "Compliance management and auditing",
                "parent_role": Role.USER,
                "permissions": [
                    Permission.COMPLIANCE_VIEW,
                    Permission.COMPLIANCE_MANAGE,
                    Permission.COMPLIANCE_AUDIT,
                    Permission.DIGITAL_TWIN_READ,
                    Permission.USER_READ,
                ],
            },
            Role.DATA_PROTECTION_OFFICER: {
                "display_name": "Data Protection Officer",
                "description": "GDPR and data protection oversight",
                "parent_role": Role.COMPLIANCE_OFFICER,
                "permissions": [
                    Permission.DIGITAL_TWIN_DELETE,  # For GDPR deletion
                    Permission.USER_DELETE,
                    Permission.SYSTEM_ADMIN,
                ],
            },
            Role.ADMIN: {
                "display_name": "Administrator",
                "description": "System administrator",
                "parent_role": Role.OPERATOR,
                "permissions": [
                    Permission.P2P_NETWORK_ADMIN,
                    Permission.SYSTEM_ADMIN,
                    Permission.USER_ADMIN,
                    Permission.COMPLIANCE_MANAGE,
                ],
            },
            Role.SUPER_ADMIN: {
                "display_name": "Super Administrator",
                "description": "Full system access",
                "parent_role": Role.ADMIN,
                "permissions": [
                    # Super admin gets all permissions automatically
                ],
            },
            Role.SERVICE_ACCOUNT: {
                "display_name": "Service Account",
                "description": "Automated service access",
                "parent_role": None,
                "permissions": [
                    Permission.DIGITAL_TWIN_WRITE,
                    Permission.EVOLUTION_METRICS_WRITE,
                    Permission.RAG_PIPELINE_WRITE,
                    Permission.P2P_NETWORK_CONNECT,
                ],
            },
            Role.API_CLIENT: {
                "display_name": "API Client",
                "description": "External API access",
                "parent_role": Role.USER,
                "permissions": [
                    Permission.RAG_PIPELINE_QUERY,
                ],
            },
        }

        with self._get_connection() as conn:
            # Insert roles
            for role, config in role_configs.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO roles (name, display_name, description, parent_role, is_system_role)
                    VALUES (?, ?, ?, ?, 1)
                """,
                    (
                        role.value,
                        config["display_name"],
                        config["description"],
                        config["parent_role"].value if config["parent_role"] else None,
                    ),
                )

            # Insert permissions
            for permission in Permission:
                resource, action = permission.value.split(":")
                conn.execute(
                    """
                    INSERT OR IGNORE INTO permissions (name, display_name, description, resource, action)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        permission.value,
                        permission.name.replace("_", " ").title(),
                        f"Permission to {action} {resource}",
                        resource,
                        action,
                    ),
                )

            # Assign permissions to roles
            for role, config in role_configs.items():
                for permission in config["permissions"]:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO role_permissions (role_name, permission_name, granted_by)
                        VALUES (?, ?, 'SYSTEM_INIT')
                    """,
                        (role.value, permission.value),
                    )

    def create_user(
        self,
        user_id: str,
        username: str,
        email: str = None,
        password_hash: str = None,
        password_salt: str = None,
        roles: list[Role] = None,
        **kwargs,
    ) -> bool:
        """Create new user with roles.

        Args:
            user_id: Unique user identifier
            username: Username
            email: Email address
            password_hash: Hashed password
            password_salt: Password salt
            roles: Initial roles to assign
            **kwargs: Additional user attributes

        Returns:
            True if user created successfully
        """
        with self._get_connection() as conn:
            try:
                # Create user
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, email, password_hash, password_salt,
                        requires_coppa_consent, gdpr_consent, data_retention_expires
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        username,
                        email,
                        password_hash,
                        password_salt,
                        kwargs.get("requires_coppa_consent", False),
                        kwargs.get("gdpr_consent", False),
                        kwargs.get("data_retention_expires"),
                    ),
                )

                # Assign roles
                roles = roles or [Role.USER]  # Default to USER role
                for role in roles:
                    self.assign_role_to_user(user_id, role)

                logger.info(
                    f"Created user: {username} ({user_id}) with roles: {[r.value for r in roles]}"
                )
                return True

            except sqlite3.IntegrityError as e:
                logger.error(f"Failed to create user {username}: {e}")
                return False

    def assign_role_to_user(
        self,
        user_id: str,
        role: Role,
        assigned_by: str = "SYSTEM",
        expires_at: datetime | None = None,
    ) -> bool:
        """Assign role to user.

        Args:
            user_id: User identifier
            role: Role to assign
            assigned_by: Who assigned the role
            expires_at: Optional expiration time

        Returns:
            True if role assigned successfully
        """
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO user_roles (user_id, role_name, assigned_by, expires_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (user_id, role.value, assigned_by, expires_at),
                )

                logger.info(f"Assigned role {role.value} to user {user_id}")
                return True

            except Exception as e:
                logger.error(
                    f"Failed to assign role {role.value} to user {user_id}: {e}"
                )
                return False

    def revoke_role_from_user(self, user_id: str, role: Role) -> bool:
        """Revoke role from user.

        Args:
            user_id: User identifier
            role: Role to revoke

        Returns:
            True if role revoked successfully
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                DELETE FROM user_roles WHERE user_id = ? AND role_name = ?
            """,
                (user_id, role.value),
            )

            logger.info(f"Revoked role {role.value} from user {user_id}")
            return True

    def get_user_roles(self, user_id: str) -> list[Role]:
        """Get all active roles for user.

        Args:
            user_id: User identifier

        Returns:
            List of active roles
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role_name FROM user_roles
                WHERE user_id = ?
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
                (user_id,),
            )

            role_names = [row[0] for row in cursor.fetchall()]
            return [
                Role(name) for name in role_names if name in [r.value for r in Role]
            ]

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """Get all permissions for user based on their roles.

        Args:
            user_id: User identifier

        Returns:
            Set of permissions
        """
        user_roles = self.get_user_roles(user_id)
        all_permissions = set()

        for role in user_roles:
            role_permissions = self.get_role_permissions(role)
            all_permissions.update(role_permissions)

            # Add inherited permissions from parent roles
            parent_permissions = self._get_inherited_permissions(role)
            all_permissions.update(parent_permissions)

        # Super admin gets all permissions
        if Role.SUPER_ADMIN in user_roles:
            all_permissions = set(Permission)

        return all_permissions

    def get_role_permissions(self, role: Role) -> set[Permission]:
        """Get permissions directly assigned to role.

        Args:
            role: Role to check

        Returns:
            Set of permissions
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT permission_name FROM role_permissions
                WHERE role_name = ?
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
                (role.value,),
            )

            permission_names = [row[0] for row in cursor.fetchall()]
            return {
                Permission(name)
                for name in permission_names
                if name in [p.value for p in Permission]
            }

    def _get_inherited_permissions(self, role: Role) -> set[Permission]:
        """Get permissions inherited from parent roles.

        Args:
            role: Role to check

        Returns:
            Set of inherited permissions
        """
        inherited_permissions = set()

        with self._get_connection() as conn:
            # Find parent role
            cursor = conn.execute(
                """
                SELECT parent_role FROM roles WHERE name = ?
            """,
                (role.value,),
            )
            row = cursor.fetchone()

            if row and row[0]:
                try:
                    parent_role = Role(row[0])
                    # Get parent role's permissions
                    parent_permissions = self.get_role_permissions(parent_role)
                    inherited_permissions.update(parent_permissions)

                    # Recursively get grandparent permissions
                    grandparent_permissions = self._get_inherited_permissions(
                        parent_role
                    )
                    inherited_permissions.update(grandparent_permissions)
                except ValueError:
                    pass  # Invalid parent role

        return inherited_permissions

    def check_permission(
        self, user_id: str, permission: Permission, resource_id: str = None
    ) -> bool:
        """Check if user has specific permission.

        Args:
            user_id: User identifier
            permission: Permission to check
            resource_id: Optional resource identifier

        Returns:
            True if user has permission
        """
        user_permissions = self.get_user_permissions(user_id)
        has_permission = permission in user_permissions

        # Log access attempt
        self._log_access(user_id, permission.value, resource_id, has_permission)

        return has_permission

    def require_permission(
        self, user_id: str, permission: Permission, resource_id: str = None
    ):
        """Require user to have permission, raise exception if not.

        Args:
            user_id: User identifier
            permission: Required permission
            resource_id: Optional resource identifier

        Raises:
            AccessDeniedException: If user lacks permission
        """
        if not self.check_permission(user_id, permission, resource_id):
            raise AccessDeniedException(
                f"User {user_id} does not have permission {permission.value}"
            )

    def _log_access(
        self,
        user_id: str,
        action: str,
        resource: str = None,
        access_granted: bool = True,
        denial_reason: str = None,
        ip_address: str = None,
        user_agent: str = None,
        session_id: str = None,
    ):
        """Log access attempt.

        Args:
            user_id: User identifier
            action: Action attempted
            resource: Resource accessed
            access_granted: Whether access was granted
            denial_reason: Reason for denial if applicable
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session identifier
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO access_log (
                    user_id, action, resource, permission_required,
                    access_granted, denial_reason, ip_address, user_agent, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    action,
                    resource,
                    action,
                    access_granted,
                    denial_reason,
                    ip_address,
                    user_agent,
                    session_id,
                ),
            )

    def get_access_log(
        self, user_id: str = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get access log entries.

        Args:
            user_id: Optional user filter
            limit: Maximum entries to return

        Returns:
            List of access log entries
        """
        with self._get_connection() as conn:
            if user_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM access_log WHERE user_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                """,
                    (user_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM access_log
                    ORDER BY timestamp DESC LIMIT ?
                """,
                    (limit,),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_rbac_stats(self) -> dict[str, Any]:
        """Get RBAC system statistics.

        Returns:
            Statistics about roles, users, and permissions
        """
        with self._get_connection() as conn:
            stats = {}

            # User statistics
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            stats["total_users"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            stats["active_users"] = cursor.fetchone()[0]

            # Role statistics
            cursor = conn.execute("SELECT COUNT(*) FROM roles")
            stats["total_roles"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM permissions")
            stats["total_permissions"] = cursor.fetchone()[0]

            # Assignment statistics
            cursor = conn.execute("SELECT COUNT(*) FROM user_roles")
            stats["role_assignments"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM role_permissions")
            stats["permission_assignments"] = cursor.fetchone()[0]

            # Access log statistics
            cursor = conn.execute("SELECT COUNT(*) FROM access_log")
            stats["total_access_attempts"] = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM access_log
                WHERE access_granted = 0 AND timestamp > datetime('now', '-24 hours')
            """
            )
            stats["denied_access_24h"] = cursor.fetchone()[0]

            # Most active users
            cursor = conn.execute(
                """
                SELECT user_id, COUNT(*) as access_count
                FROM access_log
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY user_id
                ORDER BY access_count DESC
                LIMIT 5
            """
            )
            stats["most_active_users"] = [dict(row) for row in cursor.fetchall()]

            return stats


# Decorator for permission checking
def require_permission(permission: Permission):
    """Decorator to require permission for function access."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get user_id from function arguments or context
            user_id = kwargs.get("user_id") or getattr(args[0], "user_id", None)
            if not user_id:
                raise AccessDeniedException("No user context for permission check")

            # Get RBAC instance (assume global instance)
            rbac = get_rbac_instance()
            rbac.require_permission(user_id, permission)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Global RBAC instance
_rbac_instance = None


def get_rbac_instance() -> RBACSystem:
    """Get global RBAC instance."""
    global _rbac_instance
    if _rbac_instance is None:
        _rbac_instance = RBACSystem()
    return _rbac_instance


# Example usage
if __name__ == "__main__":
    import tempfile

    # Test with temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    rbac = RBACSystem(db_path)

    # Create test user
    rbac.create_user(
        user_id="test_user_1",
        username="testuser",
        email="test@example.com",
        roles=[Role.STUDENT],
    )

    # Test permissions
    has_read = rbac.check_permission("test_user_1", Permission.DIGITAL_TWIN_READ)
    has_admin = rbac.check_permission("test_user_1", Permission.SYSTEM_ADMIN)

    print(f"User has DIGITAL_TWIN_READ: {has_read}")
    print(f"User has SYSTEM_ADMIN: {has_admin}")

    # Get user permissions
    permissions = rbac.get_user_permissions("test_user_1")
    print(f"User permissions: {[p.value for p in permissions]}")

    # Get RBAC stats
    stats = rbac.get_rbac_stats()
    print(f"RBAC Stats: {stats}")

    # Cleanup
    os.unlink(db_path)
