"""
Multi-Tenant Resource Manager for AIVillage.

Provides complete resource isolation and management for multi-tenant deployments,
integrating with RBAC system for secure access control.
"""

import asyncio
import json
import logging
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .rbac_system import Permission, RBACSystem, Role

logger = logging.getLogger(__name__)


@dataclass
class TenantResource:
    """Represents an isolated tenant resource."""

    resource_id: str
    tenant_id: str
    resource_type: str
    name: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    data_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    size_bytes: int = 0


class ResourceType:
    """Resource types in AIVillage."""

    AGENT = "agent"
    RAG_COLLECTION = "rag_collection"
    MODEL = "model"
    P2P_NETWORK = "p2p_network"
    VECTOR_DB = "vector_db"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    COMPUTE_JOB = "compute_job"
    STORAGE_BUCKET = "storage_bucket"


class MultiTenantManager:
    """Manages multi-tenant resource isolation and lifecycle."""

    def __init__(
        self, rbac_system: RBACSystem, base_path: Path = Path("data/tenants"), db_path: Path = Path("data/tenants.db")
    ):
        """Initialize multi-tenant manager."""
        self.rbac = rbac_system
        self.base_path = base_path
        self.db_path = db_path

        # Create base directories
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Resource registries per tenant
        self.tenant_resources: dict[str, dict[str, list[TenantResource]]] = defaultdict(lambda: defaultdict(list))

        # Resource locks for concurrent access
        self.resource_locks: dict[str, asyncio.Lock] = {}

        # Initialize database
        self._init_database()

        # Load existing resources
        self._load_resources()

        logger.info(f"Multi-tenant manager initialized at {self.base_path}")

    def _init_database(self):
        """Initialize SQLite database for resource tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS resources (
                resource_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT,
                data_path TEXT,
                config TEXT,
                status TEXT DEFAULT 'active',
                size_bytes INTEGER DEFAULT 0,
                UNIQUE(tenant_id, resource_type, name)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tenant_resources
            ON resources(tenant_id, resource_type)
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                cpu_percent REAL,
                memory_mb INTEGER,
                storage_gb REAL,
                network_mb INTEGER
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_resources(self):
        """Load existing resources from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM resources WHERE status = 'active'")
        rows = cursor.fetchall()

        for row in rows:
            resource = TenantResource(
                resource_id=row[0],
                tenant_id=row[1],
                resource_type=row[2],
                name=row[3],
                created_at=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5]) if row[5] else {},
                data_path=Path(row[6]) if row[6] else None,
                config=json.loads(row[7]) if row[7] else {},
                status=row[8],
                size_bytes=row[9],
            )

            self.tenant_resources[resource.tenant_id][resource.resource_type].append(resource)

        conn.close()
        logger.info(f"Loaded {len(rows)} existing resources")

    # Agent Management

    async def create_agent(
        self, tenant_id: str, user_id: str, agent_name: str, agent_type: str, config: dict[str, Any]
    ) -> TenantResource:
        """Create isolated agent for tenant."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.AGENT_CREATE):
            raise PermissionError(f"User {user_id} cannot create agents")

        # Check quota
        if not await self.rbac.check_quota(tenant_id, "agents"):
            raise ValueError(f"Tenant {tenant_id} has reached agent quota")

        # Create isolated agent directory
        agent_id = f"{tenant_id}_agent_{agent_name}_{datetime.utcnow().timestamp()}"
        agent_path = self.base_path / tenant_id / "agents" / agent_id
        agent_path.mkdir(parents=True, exist_ok=True)

        # Initialize agent configuration
        agent_config = {
            "type": agent_type,
            "tenant_id": tenant_id,
            "isolation": {
                "network_namespace": f"{tenant_id}_network",
                "storage_path": str(agent_path),
                "max_memory_mb": config.get("max_memory_mb", 512),
                "max_cpu_percent": config.get("max_cpu_percent", 25),
            },
            **config,
        }

        # Save agent configuration
        config_path = agent_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(agent_config, f, indent=2)

        # Create resource record
        resource = TenantResource(
            resource_id=agent_id,
            tenant_id=tenant_id,
            resource_type=ResourceType.AGENT,
            name=agent_name,
            created_at=datetime.utcnow(),
            metadata={"agent_type": agent_type, "created_by": user_id},
            data_path=agent_path,
            config=agent_config,
        )

        # Store in database
        self._save_resource(resource)

        # Update tenant resources
        self.tenant_resources[tenant_id][ResourceType.AGENT].append(resource)

        # Update quota
        await self.rbac.update_quota_usage(tenant_id, "agents", 1, "add")

        logger.info(f"Created agent {agent_id} for tenant {tenant_id}")
        return resource

    async def get_agent(self, tenant_id: str, user_id: str, agent_name: str) -> TenantResource | None:
        """Get agent with tenant isolation."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.AGENT_READ):
            raise PermissionError(f"User {user_id} cannot read agents")

        # Find agent
        agents = self.tenant_resources[tenant_id][ResourceType.AGENT]
        for agent in agents:
            if agent.name == agent_name:
                return agent

        return None

    async def list_agents(self, tenant_id: str, user_id: str) -> list[TenantResource]:
        """List all agents for tenant."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.AGENT_READ):
            raise PermissionError(f"User {user_id} cannot list agents")

        return self.tenant_resources[tenant_id][ResourceType.AGENT]

    async def delete_agent(self, tenant_id: str, user_id: str, agent_name: str) -> bool:
        """Delete agent and clean up resources."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.AGENT_DELETE):
            raise PermissionError(f"User {user_id} cannot delete agents")

        # Find agent
        agents = self.tenant_resources[tenant_id][ResourceType.AGENT]
        agent = None
        for a in agents:
            if a.name == agent_name:
                agent = a
                break

        if not agent:
            return False

        # Clean up agent resources
        if agent.data_path and agent.data_path.exists():
            shutil.rmtree(agent.data_path)

        # Update database
        self._delete_resource(agent.resource_id)

        # Remove from registry
        agents.remove(agent)

        # Update quota
        await self.rbac.update_quota_usage(tenant_id, "agents", 1, "remove")

        logger.info(f"Deleted agent {agent.resource_id}")
        return True

    # RAG Collection Management

    async def create_rag_collection(
        self, tenant_id: str, user_id: str, collection_name: str, config: dict[str, Any]
    ) -> TenantResource:
        """Create isolated RAG collection for tenant."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.RAG_CREATE):
            raise PermissionError(f"User {user_id} cannot create RAG collections")

        # Check quota
        if not await self.rbac.check_quota(tenant_id, "rag_collections"):
            raise ValueError(f"Tenant {tenant_id} has reached RAG collection quota")

        # Create isolated collection directory
        collection_id = f"{tenant_id}_rag_{collection_name}_{datetime.utcnow().timestamp()}"
        collection_path = self.base_path / tenant_id / "rag" / collection_id
        collection_path.mkdir(parents=True, exist_ok=True)

        # Initialize collection with tenant isolation
        collection_config = {
            "name": collection_name,
            "tenant_id": tenant_id,
            "vector_db": {
                "type": config.get("vector_db_type", "faiss"),
                "namespace": f"{tenant_id}_{collection_name}",
                "path": str(collection_path / "vectors"),
            },
            "knowledge_graph": {
                "enabled": config.get("use_knowledge_graph", False),
                "namespace": f"{tenant_id}_{collection_name}_graph",
            },
            "encryption": {"enabled": True, "key_id": f"{tenant_id}_rag_key"},
            **config,
        }

        # Save configuration
        config_path = collection_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(collection_config, f, indent=2)

        # Create resource record
        resource = TenantResource(
            resource_id=collection_id,
            tenant_id=tenant_id,
            resource_type=ResourceType.RAG_COLLECTION,
            name=collection_name,
            created_at=datetime.utcnow(),
            metadata={"created_by": user_id},
            data_path=collection_path,
            config=collection_config,
        )

        # Store in database
        self._save_resource(resource)

        # Update tenant resources
        self.tenant_resources[tenant_id][ResourceType.RAG_COLLECTION].append(resource)

        # Update quota
        await self.rbac.update_quota_usage(tenant_id, "rag_collections", 1, "add")

        logger.info(f"Created RAG collection {collection_id} for tenant {tenant_id}")
        return resource

    async def add_documents_to_rag(
        self, tenant_id: str, user_id: str, collection_name: str, documents: list[dict[str, Any]]
    ) -> bool:
        """Add documents to RAG collection with isolation."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.RAG_UPDATE):
            raise PermissionError(f"User {user_id} cannot update RAG collections")

        # Find collection
        collections = self.tenant_resources[tenant_id][ResourceType.RAG_COLLECTION]
        collection = None
        for c in collections:
            if c.name == collection_name:
                collection = c
                break

        if not collection:
            raise ValueError(f"Collection {collection_name} not found")

        # Store documents in isolated location
        docs_path = collection.data_path / "documents"
        docs_path.mkdir(exist_ok=True)

        for i, doc in enumerate(documents):
            doc_path = docs_path / f"doc_{datetime.utcnow().timestamp()}_{i}.json"
            with open(doc_path, "w") as f:
                json.dump(doc, f)

        # Update collection size
        collection.size_bytes += sum(len(json.dumps(d)) for d in documents)
        self._update_resource_size(collection.resource_id, collection.size_bytes)

        logger.info(f"Added {len(documents)} documents to collection {collection_name}")
        return True

    # P2P Network Management

    async def create_p2p_network(
        self, tenant_id: str, user_id: str, network_name: str, config: dict[str, Any]
    ) -> TenantResource:
        """Create isolated P2P network for tenant."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.P2P_CREATE):
            raise PermissionError(f"User {user_id} cannot create P2P networks")

        # Create isolated network configuration
        network_id = f"{tenant_id}_p2p_{network_name}_{datetime.utcnow().timestamp()}"
        network_path = self.base_path / tenant_id / "p2p" / network_id
        network_path.mkdir(parents=True, exist_ok=True)

        # Configure isolated network
        network_config = {
            "name": network_name,
            "tenant_id": tenant_id,
            "isolation": {
                "network_id": f"{tenant_id}_{network_name}",
                "bootstrap_nodes": [],  # Tenant-specific bootstrap nodes
                "allowed_peers": [],  # Only allow tenant's peers
                "encryption_key": f"{tenant_id}_p2p_key",
            },
            "limits": {
                "max_peers": config.get("max_peers", 50),
                "max_bandwidth_mbps": config.get("max_bandwidth_mbps", 100),
                "max_connections": config.get("max_connections", 100),
            },
            **config,
        }

        # Save configuration
        config_path = network_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(network_config, f, indent=2)

        # Create resource record
        resource = TenantResource(
            resource_id=network_id,
            tenant_id=tenant_id,
            resource_type=ResourceType.P2P_NETWORK,
            name=network_name,
            created_at=datetime.utcnow(),
            metadata={"created_by": user_id},
            data_path=network_path,
            config=network_config,
        )

        # Store in database
        self._save_resource(resource)

        # Update tenant resources
        self.tenant_resources[tenant_id][ResourceType.P2P_NETWORK].append(resource)

        logger.info(f"Created P2P network {network_id} for tenant {tenant_id}")
        return resource

    # Model Management

    async def deploy_model(
        self, tenant_id: str, user_id: str, model_name: str, model_path: Path, config: dict[str, Any]
    ) -> TenantResource:
        """Deploy model with tenant isolation."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.MODEL_DEPLOY):
            raise PermissionError(f"User {user_id} cannot deploy models")

        # Check storage quota
        model_size_gb = model_path.stat().st_size / (1024**3)
        if not await self.rbac.check_quota(tenant_id, "storage_gb", int(model_size_gb)):
            raise ValueError(f"Tenant {tenant_id} has insufficient storage quota")

        # Create isolated model directory
        model_id = f"{tenant_id}_model_{model_name}_{datetime.utcnow().timestamp()}"
        isolated_model_path = self.base_path / tenant_id / "models" / model_id
        isolated_model_path.mkdir(parents=True, exist_ok=True)

        # Copy model to isolated location
        shutil.copy2(model_path, isolated_model_path / "model.bin")

        # Configure model isolation
        model_config = {
            "name": model_name,
            "tenant_id": tenant_id,
            "isolation": {
                "inference_namespace": f"{tenant_id}_inference",
                "max_instances": config.get("max_instances", 1),
                "max_memory_gb": config.get("max_memory_gb", 4),
                "max_batch_size": config.get("max_batch_size", 32),
            },
            **config,
        }

        # Save configuration
        config_path = isolated_model_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Create resource record
        resource = TenantResource(
            resource_id=model_id,
            tenant_id=tenant_id,
            resource_type=ResourceType.MODEL,
            name=model_name,
            created_at=datetime.utcnow(),
            metadata={"created_by": user_id, "original_path": str(model_path)},
            data_path=isolated_model_path,
            config=model_config,
            size_bytes=model_path.stat().st_size,
        )

        # Store in database
        self._save_resource(resource)

        # Update tenant resources
        self.tenant_resources[tenant_id][ResourceType.MODEL].append(resource)

        # Update storage quota
        await self.rbac.update_quota_usage(tenant_id, "storage_gb", int(model_size_gb), "add")

        logger.info(f"Deployed model {model_id} for tenant {tenant_id}")
        return resource

    # Cross-Tenant Resource Sharing

    async def share_resource(
        self, owner_tenant_id: str, target_tenant_id: str, resource_id: str, user_id: str, permissions: list[Permission]
    ) -> bool:
        """Share resource with another tenant (controlled)."""
        # Only super admin can enable cross-tenant sharing
        user = self.rbac.users.get(user_id)
        if not user or user.role != Role.SUPER_ADMIN:
            raise PermissionError("Only super admin can enable cross-tenant sharing")

        # Find resource
        resource = await self._get_resource_by_id(resource_id)
        if not resource or resource.tenant_id != owner_tenant_id:
            return False

        # Create sharing record
        {
            "resource_id": resource_id,
            "owner_tenant": owner_tenant_id,
            "shared_with": target_tenant_id,
            "permissions": [p.value for p in permissions],
            "shared_at": datetime.utcnow().isoformat(),
            "shared_by": user_id,
        }

        # Store sharing configuration
        # TODO: Implement sharing mechanism

        logger.info(f"Shared resource {resource_id} from {owner_tenant_id} to {target_tenant_id}")
        return True

    # Resource Monitoring

    async def get_tenant_usage(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        """Get resource usage for tenant."""
        # Check permissions
        user = self.rbac.users.get(user_id)
        if not user:
            raise ValueError("Invalid user")

        if user.tenant_id != tenant_id and user.role != Role.SUPER_ADMIN:
            raise PermissionError("Cannot view other tenant's usage")

        # Calculate usage
        resources = self.tenant_resources[tenant_id]

        usage = {
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "resources": {
                "agents": len(resources[ResourceType.AGENT]),
                "rag_collections": len(resources[ResourceType.RAG_COLLECTION]),
                "models": len(resources[ResourceType.MODEL]),
                "p2p_networks": len(resources[ResourceType.P2P_NETWORK]),
            },
            "storage": {
                "total_bytes": sum(r.size_bytes for r_list in resources.values() for r in r_list),
                "by_type": {},
            },
        }

        # Calculate storage by type
        for resource_type, resource_list in resources.items():
            usage["storage"]["by_type"][resource_type] = sum(r.size_bytes for r in resource_list)

        return usage

    async def monitor_resource(self, resource_id: str, metrics: dict[str, Any]):
        """Record resource usage metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get resource
        resource = await self._get_resource_by_id(resource_id)
        if not resource:
            return

        cursor.execute(
            """
            INSERT INTO resource_usage
            (tenant_id, resource_type, timestamp, cpu_percent, memory_mb, storage_gb, network_mb)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                resource.tenant_id,
                resource.resource_type,
                datetime.utcnow(),
                metrics.get("cpu_percent"),
                metrics.get("memory_mb"),
                metrics.get("storage_gb"),
                metrics.get("network_mb"),
            ),
        )

        conn.commit()
        conn.close()

    # Cleanup and Maintenance

    async def cleanup_tenant(self, tenant_id: str, user_id: str) -> bool:
        """Clean up all resources for tenant."""
        # Check permissions
        if not await self.rbac.check_permission(user_id, Permission.TENANT_DELETE):
            raise PermissionError(f"User {user_id} cannot delete tenant resources")

        # Delete all tenant resources
        for resource_type, resources in self.tenant_resources[tenant_id].items():
            for resource in resources:
                # Clean up physical resources
                if resource.data_path and resource.data_path.exists():
                    shutil.rmtree(resource.data_path)

                # Update database
                self._delete_resource(resource.resource_id)

        # Remove tenant directory
        tenant_path = self.base_path / tenant_id
        if tenant_path.exists():
            shutil.rmtree(tenant_path)

        # Clear from registry
        del self.tenant_resources[tenant_id]

        logger.info(f"Cleaned up all resources for tenant {tenant_id}")
        return True

    # Database Operations

    def _save_resource(self, resource: TenantResource):
        """Save resource to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO resources
            (resource_id, tenant_id, resource_type, name, created_at,
             metadata, data_path, config, status, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                resource.resource_id,
                resource.tenant_id,
                resource.resource_type,
                resource.name,
                resource.created_at.isoformat(),
                json.dumps(resource.metadata),
                str(resource.data_path) if resource.data_path else None,
                json.dumps(resource.config),
                resource.status,
                resource.size_bytes,
            ),
        )

        conn.commit()
        conn.close()

    def _delete_resource(self, resource_id: str):
        """Delete resource from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE resources SET status = 'deleted' WHERE resource_id = ?", (resource_id,))

        conn.commit()
        conn.close()

    def _update_resource_size(self, resource_id: str, size_bytes: int):
        """Update resource size in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE resources SET size_bytes = ? WHERE resource_id = ?", (size_bytes, resource_id))

        conn.commit()
        conn.close()

    async def _get_resource_by_id(self, resource_id: str) -> TenantResource | None:
        """Get resource by ID from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM resources WHERE resource_id = ? AND status = 'active'", (resource_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return TenantResource(
            resource_id=row[0],
            tenant_id=row[1],
            resource_type=row[2],
            name=row[3],
            created_at=datetime.fromisoformat(row[4]),
            metadata=json.loads(row[5]) if row[5] else {},
            data_path=Path(row[6]) if row[6] else None,
            config=json.loads(row[7]) if row[7] else {},
            status=row[8],
            size_bytes=row[9],
        )


async def initialize_multi_tenant_manager(rbac_system: RBACSystem) -> MultiTenantManager:
    """Initialize multi-tenant manager with RBAC integration."""
    manager = MultiTenantManager(rbac_system)
    logger.info("Multi-tenant manager initialized")
    return manager


if __name__ == "__main__":
    # Example usage
    async def main():
        from .rbac_system import initialize_rbac_system

        # Initialize systems
        rbac = await initialize_rbac_system()
        manager = await initialize_multi_tenant_manager(rbac)

        # Create test tenant and user
        tenant = await rbac.create_tenant(
            name="Test Company",
            admin_user={"username": "company_admin", "email": "admin@company.com", "password": "secure_password"},
        )

        # Get admin user
        admin_user = None
        for user in rbac.users.values():
            if user.username == "company_admin":
                admin_user = user
                break

        if admin_user:
            # Create agent
            agent = await manager.create_agent(
                tenant_id=tenant.tenant_id,
                user_id=admin_user.user_id,
                agent_name="test_agent",
                agent_type="king_agent",
                config={"max_memory_mb": 1024},
            )
            print(f"Created agent: {agent.resource_id}")

            # Create RAG collection
            rag = await manager.create_rag_collection(
                tenant_id=tenant.tenant_id,
                user_id=admin_user.user_id,
                collection_name="knowledge_base",
                config={"vector_db_type": "faiss"},
            )
            print(f"Created RAG collection: {rag.resource_id}")

            # Get usage
            usage = await manager.get_tenant_usage(tenant_id=tenant.tenant_id, user_id=admin_user.user_id)
            print(f"Tenant usage: {usage}")

    asyncio.run(main())
