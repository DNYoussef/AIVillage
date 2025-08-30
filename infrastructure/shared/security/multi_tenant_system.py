#!/usr/bin/env python3
"""Multi-Tenant Isolation System for AIVillage.

Provides comprehensive multi-tenant isolation with organizations, workspaces,
and tenant-aware RBAC for complete data and resource separation.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class TenantType(Enum):
    """Types of tenants in the system."""

    PERSONAL = "personal"  # Individual user workspace
    TEAM = "team"  # Small team collaboration
    ORGANIZATION = "organization"  # Enterprise organization
    ENTERPRISE = "enterprise"  # Multi-org enterprise
    PLATFORM = "platform"  # Platform-level (AIVillage itself)


class IsolationLevel(Enum):
    """Data isolation levels."""

    NONE = "none"  # No isolation (public data)
    LOGICAL = "logical"  # Logical separation (same DB, filtered queries)
    PHYSICAL = "physical"  # Physical separation (separate DBs)
    CRYPTOGRAPHIC = "cryptographic"  # Encrypted with tenant-specific keys


class ResourceQuota:
    """Resource quotas for tenants."""

    def __init__(
        self,
        max_users: int = 10,
        max_storage_gb: int = 10,
        max_api_calls_per_day: int = 10000,
        max_models: int = 5,
        max_agents: int = 10,
        max_rag_documents: int = 1000,
        max_p2p_connections: int = 50,
        max_compute_hours: int = 100,
    ):
        self.max_users = max_users
        self.max_storage_gb = max_storage_gb
        self.max_api_calls_per_day = max_api_calls_per_day
        self.max_models = max_models
        self.max_agents = max_agents
        self.max_rag_documents = max_rag_documents
        self.max_p2p_connections = max_p2p_connections
        self.max_compute_hours = max_compute_hours


# Default quotas by tenant type
DEFAULT_QUOTAS = {
    TenantType.PERSONAL: ResourceQuota(
        max_users=1,
        max_storage_gb=5,
        max_api_calls_per_day=1000,
        max_models=2,
        max_agents=3,
        max_rag_documents=100,
        max_p2p_connections=10,
        max_compute_hours=10,
    ),
    TenantType.TEAM: ResourceQuota(
        max_users=10,
        max_storage_gb=50,
        max_api_calls_per_day=10000,
        max_models=5,
        max_agents=10,
        max_rag_documents=1000,
        max_p2p_connections=50,
        max_compute_hours=100,
    ),
    TenantType.ORGANIZATION: ResourceQuota(
        max_users=100,
        max_storage_gb=500,
        max_api_calls_per_day=100000,
        max_models=20,
        max_agents=50,
        max_rag_documents=10000,
        max_p2p_connections=200,
        max_compute_hours=1000,
    ),
    TenantType.ENTERPRISE: ResourceQuota(
        max_users=1000,
        max_storage_gb=5000,
        max_api_calls_per_day=1000000,
        max_models=100,
        max_agents=200,
        max_rag_documents=100000,
        max_p2p_connections=1000,
        max_compute_hours=10000,
    ),
    TenantType.PLATFORM: ResourceQuota(
        max_users=999999,
        max_storage_gb=999999,
        max_api_calls_per_day=999999999,
        max_models=999999,
        max_agents=999999,
        max_rag_documents=999999999,
        max_p2p_connections=999999,
        max_compute_hours=999999,
    ),
}


@dataclass
class TenantContext:
    """Context for current tenant operation."""

    tenant_id: str
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    isolation_level: IsolationLevel = IsolationLevel.LOGICAL
    encryption_key: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiTenantSystem:
    """Multi-tenant isolation and management system."""

    def __init__(self, db_path: str | None = None):
        """Initialize multi-tenant system.

        Args:
            db_path: Path to multi-tenant database
        """
        self.db_path = db_path or os.getenv("MULTI_TENANT_DB_PATH", "./data/multi_tenant.db")

        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Initialize platform tenant if not exists
        self._init_platform_tenant()

        # Cache for tenant contexts
        self._context_cache: dict[str, TenantContext] = {}

        # Encryption keys cache (in production, use KMS)
        self._encryption_keys: dict[str, bytes] = {}

        logger.info(f"Multi-tenant system initialized: {self.db_path}")

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
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Organizations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    tenant_type TEXT NOT NULL,
                    isolation_level TEXT NOT NULL,

                    -- Contact information
                    admin_email TEXT,
                    billing_email TEXT,
                    support_email TEXT,

                    -- Settings
                    settings TEXT,  -- JSON
                    features TEXT,  -- JSON list of enabled features

                    -- Status
                    is_active BOOLEAN DEFAULT 1,
                    is_suspended BOOLEAN DEFAULT 0,
                    suspension_reason TEXT,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,

                    -- Compliance
                    compliance_level TEXT,  -- GDPR, HIPAA, SOC2, etc.
                    data_residency TEXT,  -- Geographic restrictions

                    CHECK (tenant_type IN ('personal', 'team', 'organization', 'enterprise', 'platform'))
                )
            """
            )

            # Workspaces table (sub-tenants within organizations)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT,

                    -- Workspace settings
                    settings TEXT,  -- JSON
                    isolation_level TEXT NOT NULL,

                    -- Access control
                    is_public BOOLEAN DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (organization_id) REFERENCES organizations (id)
                        ON DELETE CASCADE ON UPDATE CASCADE,
                    UNIQUE (organization_id, name)
                )
            """
            )

            # Tenant memberships
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_memberships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    workspace_id TEXT,

                    -- Role within this tenant
                    role TEXT NOT NULL,
                    permissions TEXT,  -- JSON list of additional permissions

                    -- Status
                    is_active BOOLEAN DEFAULT 1,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    invited_by TEXT,
                    invitation_token TEXT,
                    invitation_expires_at TIMESTAMP,

                    FOREIGN KEY (organization_id) REFERENCES organizations (id)
                        ON DELETE CASCADE ON UPDATE CASCADE,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces (id)
                        ON DELETE CASCADE ON UPDATE CASCADE,
                    UNIQUE (user_id, organization_id, workspace_id)
                )
            """
            )

            # Resource quotas
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_quotas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    tenant_type TEXT NOT NULL,  -- 'organization' or 'workspace'

                    -- Quotas
                    max_users INTEGER DEFAULT 10,
                    max_storage_gb INTEGER DEFAULT 10,
                    max_api_calls_per_day INTEGER DEFAULT 10000,
                    max_models INTEGER DEFAULT 5,
                    max_agents INTEGER DEFAULT 10,
                    max_rag_documents INTEGER DEFAULT 1000,
                    max_p2p_connections INTEGER DEFAULT 50,
                    max_compute_hours INTEGER DEFAULT 100,

                    -- Custom quotas JSON
                    custom_quotas TEXT,

                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE (tenant_id, tenant_type)
                )
            """
            )

            # Resource usage tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    tenant_type TEXT NOT NULL,

                    -- Current usage
                    current_users INTEGER DEFAULT 0,
                    current_storage_gb REAL DEFAULT 0,
                    api_calls_today INTEGER DEFAULT 0,
                    current_models INTEGER DEFAULT 0,
                    current_agents INTEGER DEFAULT 0,
                    current_rag_documents INTEGER DEFAULT 0,
                    current_p2p_connections INTEGER DEFAULT 0,
                    compute_hours_this_month REAL DEFAULT 0,

                    -- Historical tracking
                    last_reset_date DATE,
                    total_api_calls_lifetime INTEGER DEFAULT 0,
                    total_compute_hours_lifetime REAL DEFAULT 0,

                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE (tenant_id, tenant_type)
                )
            """
            )

            # Tenant data isolation mapping
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_data_isolation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,  -- 'database', 'storage', 'models', etc.
                    resource_id TEXT NOT NULL,

                    -- Isolation configuration
                    isolation_level TEXT NOT NULL,
                    encryption_enabled BOOLEAN DEFAULT 0,
                    encryption_key_id TEXT,

                    -- Physical separation
                    database_name TEXT,
                    storage_bucket TEXT,
                    namespace TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE (tenant_id, resource_type, resource_id)
                )
            """
            )

            # Audit log for tenant operations
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    details TEXT,  -- JSON
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT
                )
            """
            )

            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_workspaces_org ON workspaces (organization_id)",
                "CREATE INDEX IF NOT EXISTS idx_memberships_user ON tenant_memberships (user_id)",
                "CREATE INDEX IF NOT EXISTS idx_memberships_org ON tenant_memberships (organization_id)",
                "CREATE INDEX IF NOT EXISTS idx_quotas_tenant ON resource_quotas (tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_usage_tenant ON resource_usage (tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_isolation_tenant ON tenant_data_isolation (tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_tenant ON tenant_audit_log (tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON tenant_audit_log (timestamp)",
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

    def _init_platform_tenant(self):
        """Initialize the platform-level tenant."""
        with self._get_connection() as conn:
            # Check if platform tenant exists
            cursor = conn.execute("SELECT id FROM organizations WHERE id = 'platform' AND tenant_type = 'platform'")
            if not cursor.fetchone():
                # Create platform tenant
                self.create_organization(
                    org_id="platform",
                    name="aivillage-platform",
                    display_name="AIVillage Platform",
                    description="Platform-level tenant for system operations",
                    tenant_type=TenantType.PLATFORM,
                    isolation_level=IsolationLevel.PHYSICAL,
                    admin_email="admin@aivillage.local",
                )
                logger.info("Created platform tenant")

    def create_organization(
        self,
        org_id: str | None = None,
        name: str = None,
        display_name: str = None,
        description: str = None,
        tenant_type: TenantType = TenantType.TEAM,
        isolation_level: IsolationLevel = IsolationLevel.LOGICAL,
        admin_email: str | None = None,
        compliance_level: str | None = None,
        data_residency: str | None = None,
        **kwargs,
    ) -> str:
        """Create a new organization.

        Args:
            org_id: Organization ID (generated if not provided)
            name: Unique organization name
            display_name: Display name
            description: Organization description
            tenant_type: Type of tenant
            isolation_level: Data isolation level
            admin_email: Administrator email
            compliance_level: Compliance requirements (GDPR, HIPAA, etc.)
            data_residency: Geographic data residency requirements
            **kwargs: Additional organization settings

        Returns:
            Organization ID
        """
        org_id = org_id or f"org_{uuid4().hex[:12]}"

        with self._get_connection() as conn:
            try:
                # Create organization
                conn.execute(
                    """
                    INSERT INTO organizations (
                        id, name, display_name, description,
                        tenant_type, isolation_level,
                        admin_email, compliance_level, data_residency,
                        settings, features
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        org_id,
                        name,
                        display_name,
                        description,
                        tenant_type.value,
                        isolation_level.value,
                        admin_email,
                        compliance_level,
                        data_residency,
                        json.dumps(kwargs.get("settings", {})),
                        json.dumps(kwargs.get("features", [])),
                    ),
                )

                # Set default quotas based on tenant type
                quota = DEFAULT_QUOTAS[tenant_type]
                conn.execute(
                    """
                    INSERT INTO resource_quotas (
                        tenant_id, tenant_type,
                        max_users, max_storage_gb, max_api_calls_per_day,
                        max_models, max_agents, max_rag_documents,
                        max_p2p_connections, max_compute_hours
                    ) VALUES (?, 'organization', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        org_id,
                        quota.max_users,
                        quota.max_storage_gb,
                        quota.max_api_calls_per_day,
                        quota.max_models,
                        quota.max_agents,
                        quota.max_rag_documents,
                        quota.max_p2p_connections,
                        quota.max_compute_hours,
                    ),
                )

                # Initialize usage tracking
                conn.execute(
                    """
                    INSERT INTO resource_usage (tenant_id, tenant_type)
                    VALUES (?, 'organization')
                """,
                    (org_id,),
                )

                # Create default workspace
                self.create_workspace(
                    organization_id=org_id,
                    name="default",
                    display_name="Default Workspace",
                    description="Default workspace for organization",
                )

                # Log creation
                self._log_audit(
                    tenant_id=org_id,
                    action="organization_created",
                    resource_type="organization",
                    resource_id=org_id,
                    details={"tenant_type": tenant_type.value, "isolation_level": isolation_level.value},
                )

                logger.info(f"Created organization: {org_id} ({name})")
                return org_id

            except sqlite3.IntegrityError as e:
                logger.error(f"Failed to create organization: {e}")
                raise ValueError(f"Organization with name '{name}' already exists")

    def create_workspace(
        self,
        organization_id: str,
        name: str,
        display_name: str,
        description: str | None = None,
        isolation_level: IsolationLevel | None = None,
        is_public: bool = False,
        **kwargs,
    ) -> str:
        """Create a workspace within an organization.

        Args:
            organization_id: Parent organization ID
            name: Unique workspace name within org
            display_name: Display name
            description: Workspace description
            isolation_level: Override organization's isolation level
            is_public: Whether workspace is publicly accessible
            **kwargs: Additional workspace settings

        Returns:
            Workspace ID
        """
        workspace_id = f"ws_{uuid4().hex[:12]}"

        # Get organization's isolation level if not specified
        if isolation_level is None:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT isolation_level FROM organizations WHERE id = ?", (organization_id,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Organization {organization_id} not found")
                isolation_level = IsolationLevel(row["isolation_level"])

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO workspaces (
                        id, organization_id, name, display_name,
                        description, settings, isolation_level, is_public
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        workspace_id,
                        organization_id,
                        name,
                        display_name,
                        description,
                        json.dumps(kwargs.get("settings", {})),
                        isolation_level.value,
                        is_public,
                    ),
                )

                # Log creation
                self._log_audit(
                    tenant_id=organization_id,
                    action="workspace_created",
                    resource_type="workspace",
                    resource_id=workspace_id,
                    details={"name": name, "is_public": is_public},
                )

                logger.info(f"Created workspace: {workspace_id} ({name}) in org {organization_id}")
                return workspace_id

            except sqlite3.IntegrityError:
                raise ValueError(f"Workspace '{name}' already exists in organization")

    def add_user_to_tenant(
        self,
        user_id: str,
        organization_id: str,
        workspace_id: str | None = None,
        role: str = "member",
        permissions: list[str] | None = None,
        invited_by: str | None = None,
    ) -> bool:
        """Add user to a tenant (organization or workspace).

        Args:
            user_id: User ID to add
            organization_id: Organization ID
            workspace_id: Optional workspace ID
            role: Role within the tenant
            permissions: Additional permissions
            invited_by: User who invited this user

        Returns:
            Success status
        """
        with self._get_connection() as conn:
            try:
                # Check resource quota
                if not self._check_user_quota(organization_id):
                    raise ValueError("User quota exceeded for organization")

                conn.execute(
                    """
                    INSERT OR REPLACE INTO tenant_memberships (
                        user_id, organization_id, workspace_id,
                        role, permissions, invited_by
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        organization_id,
                        workspace_id,
                        role,
                        json.dumps(permissions or []),
                        invited_by,
                    ),
                )

                # Update usage
                self._update_resource_usage(organization_id, "users", 1)

                # Log addition
                self._log_audit(
                    tenant_id=organization_id,
                    user_id=invited_by,
                    action="user_added_to_tenant",
                    resource_type="user",
                    resource_id=user_id,
                    details={
                        "workspace_id": workspace_id,
                        "role": role,
                    },
                )

                logger.info(f"Added user {user_id} to tenant {organization_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to add user to tenant: {e}")
                return False

    def remove_user_from_tenant(
        self,
        user_id: str,
        organization_id: str,
        workspace_id: str | None = None,
    ) -> bool:
        """Remove user from a tenant.

        Args:
            user_id: User ID to remove
            organization_id: Organization ID
            workspace_id: Optional workspace ID

        Returns:
            Success status
        """
        with self._get_connection() as conn:
            if workspace_id:
                conn.execute(
                    """
                    DELETE FROM tenant_memberships
                    WHERE user_id = ? AND organization_id = ? AND workspace_id = ?
                """,
                    (user_id, organization_id, workspace_id),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM tenant_memberships
                    WHERE user_id = ? AND organization_id = ?
                """,
                    (user_id, organization_id),
                )

            # Update usage
            self._update_resource_usage(organization_id, "users", -1)

            # Log removal
            self._log_audit(
                tenant_id=organization_id,
                action="user_removed_from_tenant",
                resource_type="user",
                resource_id=user_id,
                details={"workspace_id": workspace_id},
            )

            return True

    def get_user_tenants(self, user_id: str) -> list[dict[str, Any]]:
        """Get all tenants a user belongs to.

        Args:
            user_id: User ID

        Returns:
            List of tenant memberships
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    m.organization_id,
                    m.workspace_id,
                    m.role,
                    m.permissions,
                    o.name as org_name,
                    o.display_name as org_display_name,
                    o.tenant_type,
                    w.name as workspace_name,
                    w.display_name as workspace_display_name
                FROM tenant_memberships m
                JOIN organizations o ON m.organization_id = o.id
                LEFT JOIN workspaces w ON m.workspace_id = w.id
                WHERE m.user_id = ? AND m.is_active = 1
            """,
                (user_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_tenant_context(
        self,
        tenant_id: str,
        user_id: str | None = None,
    ) -> TenantContext:
        """Get tenant context for isolation.

        Args:
            tenant_id: Tenant ID (org or workspace)
            user_id: Optional user ID for permission checking

        Returns:
            TenantContext object
        """
        # Check cache
        cache_key = f"{tenant_id}:{user_id}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        with self._get_connection() as conn:
            # Try as organization first
            cursor = conn.execute(
                """
                SELECT id, isolation_level, settings
                FROM organizations
                WHERE id = ?
            """,
                (tenant_id,),
            )
            row = cursor.fetchone()

            if row:
                context = TenantContext(
                    tenant_id=tenant_id,
                    organization_id=tenant_id,
                    user_id=user_id,
                    isolation_level=IsolationLevel(row["isolation_level"]),
                    metadata=json.loads(row["settings"] or "{}"),
                )
            else:
                # Try as workspace
                cursor = conn.execute(
                    """
                    SELECT w.id, w.organization_id, w.isolation_level, w.settings
                    FROM workspaces w
                    WHERE w.id = ?
                """,
                    (tenant_id,),
                )
                row = cursor.fetchone()

                if row:
                    context = TenantContext(
                        tenant_id=tenant_id,
                        organization_id=row["organization_id"],
                        workspace_id=tenant_id,
                        user_id=user_id,
                        isolation_level=IsolationLevel(row["isolation_level"]),
                        metadata=json.loads(row["settings"] or "{}"),
                    )
                else:
                    raise ValueError(f"Tenant {tenant_id} not found")

            # Get encryption key if needed
            if context.isolation_level == IsolationLevel.CRYPTOGRAPHIC:
                context.encryption_key = self._get_encryption_key(tenant_id)

            # Cache context
            self._context_cache[cache_key] = context

            return context

    def check_tenant_access(
        self,
        user_id: str,
        tenant_id: str,
        required_role: str | None = None,
        required_permission: str | None = None,
    ) -> bool:
        """Check if user has access to tenant.

        Args:
            user_id: User ID
            tenant_id: Tenant ID (org or workspace)
            required_role: Optional required role
            required_permission: Optional required permission

        Returns:
            Access granted status
        """
        with self._get_connection() as conn:
            # Check organization membership
            cursor = conn.execute(
                """
                SELECT role, permissions
                FROM tenant_memberships
                WHERE user_id = ? AND organization_id = ? AND is_active = 1
            """,
                (user_id, tenant_id),
            )
            row = cursor.fetchone()

            if not row:
                # Check workspace membership
                cursor = conn.execute(
                    """
                    SELECT m.role, m.permissions
                    FROM tenant_memberships m
                    JOIN workspaces w ON m.workspace_id = w.id
                    WHERE m.user_id = ? AND w.id = ? AND m.is_active = 1
                """,
                    (user_id, tenant_id),
                )
                row = cursor.fetchone()

            if not row:
                return False

            # Check role if required
            if required_role and row["role"] != required_role:
                # Check role hierarchy (simplified)
                role_hierarchy = ["member", "contributor", "admin", "owner"]
                if row["role"] in role_hierarchy:
                    user_level = role_hierarchy.index(row["role"])
                    required_level = role_hierarchy.index(required_role)
                    if user_level < required_level:
                        return False
                else:
                    return False

            # Check permission if required
            if required_permission:
                permissions = json.loads(row["permissions"] or "[]")
                if required_permission not in permissions and row["role"] not in ["admin", "owner"]:
                    return False

            return True

    def isolate_query(
        self,
        query: str,
        tenant_context: TenantContext,
        table_name: str,
        params: list | None = None,
    ) -> tuple[str, list]:
        """Add tenant isolation to a SQL query with parameterized filtering.

        Args:
            query: Original SQL query
            tenant_context: Tenant context
            table_name: Table being queried
            params: Existing query parameters

        Returns:
            Tuple of (modified query, updated parameters) with tenant isolation
        """
        if tenant_context.isolation_level == IsolationLevel.NONE:
            return query, params or []

        # Sanitize table name to prevent injection
        if not table_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid table name: {table_name}")

        # Add tenant filtering with parameterized query
        tenant_filter = f" {table_name}.tenant_id = ?"
        updated_params = params or []
        updated_params.append(tenant_context.tenant_id)

        # Simple query modification (in production, use proper SQL parser)
        if "WHERE" in query.upper():
            query = query.replace("WHERE", f"WHERE {tenant_filter} AND", 1)
        else:
            query += f" WHERE {tenant_filter}"

        return query, updated_params

    def get_tenant_database(
        self,
        tenant_context: TenantContext,
    ) -> str:
        """Get database connection string for tenant.

        Args:
            tenant_context: Tenant context

        Returns:
            Database connection string
        """
        if tenant_context.isolation_level == IsolationLevel.PHYSICAL:
            # Return tenant-specific database
            return f"./data/tenants/{tenant_context.tenant_id}/database.db"
        else:
            # Return shared database
            return self.db_path

    def get_tenant_storage_path(
        self,
        tenant_context: TenantContext,
        resource_type: str = "general",
    ) -> Path:
        """Get storage path for tenant resources.

        Args:
            tenant_context: Tenant context
            resource_type: Type of resource (models, documents, etc.)

        Returns:
            Storage path
        """
        base_path = Path("./data/tenants")

        if tenant_context.isolation_level in [IsolationLevel.PHYSICAL, IsolationLevel.CRYPTOGRAPHIC]:
            # Tenant-specific storage
            path = base_path / tenant_context.tenant_id / resource_type
        else:
            # Shared storage with logical separation
            path = base_path / "shared" / resource_type / tenant_context.tenant_id

        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_encryption_key(self, tenant_id: str) -> bytes:
        """Get or generate encryption key for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Encryption key
        """
        if tenant_id not in self._encryption_keys:
            # In production, use KMS or HSM
            # This is a simplified version
            key_material = f"{tenant_id}:{os.getenv('TENANT_ENCRYPTION_SALT', 'default-salt')}"
            self._encryption_keys[tenant_id] = hashlib.pbkdf2_hmac(
                "sha256", key_material.encode(), b"aivillage-tenant-encryption", 100000, dklen=32
            )

        return self._encryption_keys[tenant_id]

    def _check_user_quota(self, organization_id: str) -> bool:
        """Check if organization can add more users.

        Args:
            organization_id: Organization ID

        Returns:
            Whether quota allows more users
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT q.max_users, u.current_users
                FROM resource_quotas q
                JOIN resource_usage u ON q.tenant_id = u.tenant_id
                WHERE q.tenant_id = ? AND q.tenant_type = 'organization'
            """,
                (organization_id,),
            )

            row = cursor.fetchone()
            if row:
                return row["current_users"] < row["max_users"]
            return True

    def check_resource_quota(
        self,
        tenant_id: str,
        resource_type: str,
        requested_amount: int = 1,
    ) -> bool:
        """Check if tenant has quota for resource.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource
            requested_amount: Amount requested

        Returns:
            Whether quota allows the request
        """
        resource_mapping = {
            "storage": ("current_storage_gb", "max_storage_gb"),
            "api_calls": ("api_calls_today", "max_api_calls_per_day"),
            "models": ("current_models", "max_models"),
            "agents": ("current_agents", "max_agents"),
            "documents": ("current_rag_documents", "max_rag_documents"),
            "connections": ("current_p2p_connections", "max_p2p_connections"),
            "compute": ("compute_hours_this_month", "max_compute_hours"),
        }

        if resource_type not in resource_mapping:
            return True

        usage_field, quota_field = resource_mapping[resource_type]

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT q.{quota_field} as quota, u.{usage_field} as usage
                FROM resource_quotas q
                JOIN resource_usage u ON q.tenant_id = u.tenant_id
                WHERE q.tenant_id = ?
            """,
                (tenant_id,),
            )

            row = cursor.fetchone()
            if row:
                return row["usage"] + requested_amount <= row["quota"]
            return True

    def _update_resource_usage(
        self,
        tenant_id: str,
        resource_type: str,
        delta: int,
    ):
        """Update resource usage for tenant.

        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource
            delta: Change in usage (positive or negative)
        """
        field_mapping = {
            "users": "current_users",
            "storage": "current_storage_gb",
            "api_calls": "api_calls_today",
            "models": "current_models",
            "agents": "current_agents",
            "documents": "current_rag_documents",
            "connections": "current_p2p_connections",
            "compute": "compute_hours_this_month",
        }

        if resource_type not in field_mapping:
            return

        field = field_mapping[resource_type]

        with self._get_connection() as conn:
            conn.execute(
                f"""
                UPDATE resource_usage
                SET {field} = MAX(0, {field} + ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE tenant_id = ?
            """,
                (delta, tenant_id),
            )

    def _log_audit(
        self,
        tenant_id: str,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
    ):
        """Log audit event.

        Args:
            tenant_id: Tenant ID
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource ID
            user_id: User who performed action
            details: Additional details
            success: Whether action succeeded
            error_message: Error message if failed
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO tenant_audit_log (
                    tenant_id, user_id, action,
                    resource_type, resource_id,
                    details, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tenant_id,
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(details or {}),
                    success,
                    error_message,
                ),
            )

    def get_tenant_stats(self, tenant_id: str) -> dict[str, Any]:
        """Get statistics for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tenant statistics
        """
        with self._get_connection() as conn:
            # Get basic info
            cursor = conn.execute(
                """
                SELECT * FROM organizations WHERE id = ?
            """,
                (tenant_id,),
            )
            org = dict(cursor.fetchone() or {})

            # Get quotas and usage
            cursor = conn.execute(
                """
                SELECT * FROM resource_quotas WHERE tenant_id = ?
            """,
                (tenant_id,),
            )
            quotas = dict(cursor.fetchone() or {})

            cursor = conn.execute(
                """
                SELECT * FROM resource_usage WHERE tenant_id = ?
            """,
                (tenant_id,),
            )
            usage = dict(cursor.fetchone() or {})

            # Get member count
            cursor = conn.execute(
                """
                SELECT COUNT(*) as member_count
                FROM tenant_memberships
                WHERE organization_id = ? AND is_active = 1
            """,
                (tenant_id,),
            )
            members = cursor.fetchone()["member_count"]

            # Get workspace count
            cursor = conn.execute(
                """
                SELECT COUNT(*) as workspace_count
                FROM workspaces
                WHERE organization_id = ? AND is_active = 1
            """,
                (tenant_id,),
            )
            workspaces = cursor.fetchone()["workspace_count"]

            return {
                "organization": org,
                "quotas": quotas,
                "usage": usage,
                "members": members,
                "workspaces": workspaces,
            }


# SQL Security utilities
class SQLSecurityManager:
    """Advanced SQL injection prevention and query validation."""

    @staticmethod
    def validate_identifier(identifier: str, max_length: int = 64) -> str:
        """Validate and sanitize SQL identifiers (table/column names).
        
        Args:
            identifier: SQL identifier to validate
            max_length: Maximum allowed length
            
        Returns:
            Validated identifier
            
        Raises:
            ValueError: If identifier is invalid or potentially malicious
        """
        if not isinstance(identifier, str):
            raise ValueError("Identifier must be a string")
            
        if not identifier:
            raise ValueError("Identifier cannot be empty")
            
        if len(identifier) > max_length:
            raise ValueError(f"Identifier too long: {len(identifier)} > {max_length}")
            
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not identifier.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid characters in identifier: {identifier}")
            
        # Check for SQL injection patterns
        dangerous_patterns = [
            '--', '/*', '*/', ';', 'union', 'select', 'insert', 'update', 
            'delete', 'drop', 'create', 'alter', 'exec', 'execute'
        ]
        
        identifier_lower = identifier.lower()
        for pattern in dangerous_patterns:
            if pattern in identifier_lower:
                raise ValueError(f"Potentially dangerous pattern in identifier: {pattern}")
                
        return identifier

    @staticmethod
    def build_parameterized_query(
        base_query: str,
        where_conditions: dict[str, Any],
        allowed_columns: set[str] | None = None
    ) -> tuple[str, list]:
        """Build parameterized query with dynamic WHERE conditions.
        
        Args:
            base_query: Base SQL query
            where_conditions: Dictionary of column->value conditions
            allowed_columns: Set of allowed column names for filtering
            
        Returns:
            Tuple of (query, parameters)
        """
        if not where_conditions:
            return base_query, []
            
        conditions = []
        params = []
        
        for column, value in where_conditions.items():
            # Validate column name
            SQLSecurityManager.validate_identifier(column)
            
            # Check against allowlist if provided
            if allowed_columns and column not in allowed_columns:
                raise ValueError(f"Column not allowed: {column}")
                
            conditions.append(f"{column} = ?")
            params.append(value)
            
        where_clause = " AND ".join(conditions)
        
        if "WHERE" in base_query.upper():
            query = base_query.replace("WHERE", f"WHERE {where_clause} AND", 1)
        else:
            query = f"{base_query} WHERE {where_clause}"
            
        return query, params

    @staticmethod  
    def validate_query_structure(query: str) -> None:
        """Validate SQL query structure for basic safety.
        
        Args:
            query: SQL query to validate
            
        Raises:
            ValueError: If query structure is potentially unsafe
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
            
        query_lower = query.lower().strip()
        
        # Check for multiple statements (basic check)
        if query_lower.count(';') > 1:
            raise ValueError("Multiple statements not allowed")
            
        # Check for dangerous SQL functions/keywords
        dangerous_keywords = [
            'xp_cmdshell', 'sp_oacreate', 'sp_oamethod', 'sp_oagetproperty',
            'dbms_java', 'dbms_lob.fileopen', 'utl_file', 'load_file',
            'into outfile', 'into dumpfile', 'load data'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                raise ValueError(f"Dangerous keyword not allowed: {keyword}")
                
        # Basic structure validation
        if not any(keyword in query_lower for keyword in ['select', 'insert', 'update', 'delete', 'create', 'alter', 'drop']):
            raise ValueError("Query must contain a valid SQL command")


class SecureInputValidator:
    """Enhanced input validation to prevent injection attacks."""

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            msg = "Input must be a string"
            raise ValueError(msg)

        # Remove null bytes and control characters
        sanitized = "".join(char for char in input_str if ord(char) >= 32 or char in "\n\r\t")

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    @staticmethod
    def validate_tenant_id(tenant_id: str) -> str:
        """Validate tenant ID format."""
        if not isinstance(tenant_id, str):
            raise ValueError("Tenant ID must be a string")
            
        # Remove any potentially dangerous characters
        sanitized = ''.join(c for c in tenant_id if c.isalnum() or c in '_-')
        
        if len(sanitized) != len(tenant_id):
            raise ValueError("Tenant ID contains invalid characters")
            
        if len(sanitized) > 64:
            raise ValueError("Tenant ID too long")
            
        if not sanitized:
            raise ValueError("Tenant ID cannot be empty")
            
        return sanitized

    @staticmethod
    def validate_sql_limit(limit: int, max_limit: int = 10000) -> int:
        """Validate SQL LIMIT values."""
        if not isinstance(limit, int):
            raise ValueError("Limit must be an integer")
            
        if limit < 0:
            raise ValueError("Limit cannot be negative")
            
        if limit > max_limit:
            raise ValueError(f"Limit too large: {limit} > {max_limit}")
            
        return limit


# Example usage
if __name__ == "__main__":
    import tempfile

    # Test with temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    system = MultiTenantSystem(db_path)

    # Create an organization
    org_id = system.create_organization(
        name="acme-corp",
        display_name="ACME Corporation",
        description="Test organization",
        tenant_type=TenantType.ORGANIZATION,
        isolation_level=IsolationLevel.LOGICAL,
        admin_email="admin@acme.com",
        compliance_level="GDPR",
    )

    print(f"Created organization: {org_id}")

    # Create a workspace
    ws_id = system.create_workspace(
        organization_id=org_id,
        name="research",
        display_name="Research Team",
        description="Research workspace",
    )

    print(f"Created workspace: {ws_id}")

    # Add a user
    system.add_user_to_tenant(
        user_id="user123",
        organization_id=org_id,
        workspace_id=ws_id,
        role="contributor",
    )

    # Check access
    has_access = system.check_tenant_access(
        user_id="user123",
        tenant_id=ws_id,
    )

    print(f"User has access to workspace: {has_access}")

    # Get tenant context
    context = system.get_tenant_context(ws_id, "user123")
    print(f"Tenant context: {context}")

    # Get stats
    stats = system.get_tenant_stats(org_id)
    print(f"Organization stats: {stats}")

    # Cleanup
    os.unlink(db_path)
