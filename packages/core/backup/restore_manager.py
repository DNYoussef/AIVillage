"""
Comprehensive Restore Manager for AIVillage.

Provides secure restore capabilities for all AIVillage components with
validation, rollback protection, and point-in-time recovery.
"""

import asyncio
import json
import logging
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .backup_manager import BackupManager, BackupStatus

logger = logging.getLogger(__name__)


class RestoreType(Enum):
    """Types of restore operations."""

    FULL_SYSTEM = "full_system"  # Complete system restore
    TENANT_ONLY = "tenant_only"  # Single tenant restore
    COMPONENT_ONLY = "component_only"  # Specific component restore
    POINT_IN_TIME = "point_in_time"  # Point-in-time recovery
    DISASTER_RECOVERY = "disaster_recovery"  # Complete disaster recovery


class RestoreStrategy(Enum):
    """Restore execution strategies."""

    REPLACE = "replace"  # Replace existing data
    MERGE = "merge"  # Merge with existing data
    SIDE_BY_SIDE = "side_by_side"  # Restore alongside existing
    TEST_RESTORE = "test_restore"  # Test restore only


class RestoreStatus(Enum):
    """Restore operation status."""

    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    VERIFIED = "verified"


@dataclass
class RestoreMetadata:
    """Metadata for restore operations."""

    restore_id: str
    backup_id: str
    restore_type: RestoreType
    strategy: RestoreStrategy
    created_at: datetime
    completed_at: datetime | None = None
    status: RestoreStatus = RestoreStatus.PENDING

    # Content control
    components_to_restore: list[str] = field(default_factory=list)
    tenants_to_restore: list[str] = field(default_factory=list)

    # Validation
    pre_restore_validation: bool = True
    post_restore_validation: bool = True

    # Rollback
    rollback_point_created: bool = False
    rollback_data_path: Path | None = None

    # Progress tracking
    total_files: int = 0
    processed_files: int = 0
    progress_percent: float = 0.0

    # Results
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    restored_components: list[str] = field(default_factory=list)
    verification_results: dict[str, bool] = field(default_factory=dict)


class RestoreManager:
    """Main restore management system."""

    def __init__(self, backup_manager: BackupManager):
        """Initialize restore manager."""
        self.backup_manager = backup_manager
        self.restore_root = backup_manager.backup_root / "restores"
        self.restore_root.mkdir(exist_ok=True)

        # Temporary restore area
        self.temp_restore_dir = self.restore_root / "temp"
        self.temp_restore_dir.mkdir(exist_ok=True)

        # Rollback storage
        self.rollback_dir = self.restore_root / "rollback"
        self.rollback_dir.mkdir(exist_ok=True)

        # Active restores tracking
        self.active_restores: dict[str, RestoreMetadata] = {}

        # Restore history database
        self.restore_db = self.restore_root / "restore_history.db"
        self._init_restore_db()

        logger.info("Restore manager initialized")

    def _init_restore_db(self):
        """Initialize restore history database."""
        conn = sqlite3.connect(self.restore_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS restores (
                restore_id TEXT PRIMARY KEY,
                backup_id TEXT NOT NULL,
                restore_type TEXT NOT NULL,
                strategy TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                status TEXT NOT NULL,
                components_to_restore TEXT,
                tenants_to_restore TEXT,
                rollback_point_created BOOLEAN DEFAULT 0,
                rollback_data_path TEXT,
                total_files INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                progress_percent REAL DEFAULT 0.0,
                errors TEXT,
                warnings TEXT,
                restored_components TEXT,
                verification_results TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_restore_created
            ON restores(created_at)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_restore_backup
            ON restores(backup_id)
        """
        )

        conn.commit()
        conn.close()

    # Main restore operations

    async def restore_full_system(
        self, backup_id: str, strategy: RestoreStrategy = RestoreStrategy.REPLACE, create_rollback: bool = True
    ) -> str:
        """Restore complete system from backup."""
        restore_id = f"full_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Validate backup
        backup_metadata = await self.backup_manager.get_backup_info(backup_id)
        if not backup_metadata:
            raise ValueError(f"Backup {backup_id} not found")

        if backup_metadata.status != BackupStatus.COMPLETED:
            raise ValueError(f"Backup {backup_id} is not in completed state")

        metadata = RestoreMetadata(
            restore_id=restore_id,
            backup_id=backup_id,
            restore_type=RestoreType.FULL_SYSTEM,
            strategy=strategy,
            created_at=datetime.utcnow(),
            components_to_restore=backup_metadata.components_included,
            tenants_to_restore=backup_metadata.tenants_included,
        )

        self.active_restores[restore_id] = metadata

        try:
            logger.info(f"Starting full system restore {restore_id} from backup {backup_id}")

            # Create rollback point
            if create_rollback:
                await self._create_rollback_point(metadata)

            # Pre-restore validation
            if metadata.pre_restore_validation:
                metadata.status = RestoreStatus.VALIDATING
                await self._save_restore_metadata(metadata)

                if not await self._validate_restore_prerequisites(metadata):
                    metadata.status = RestoreStatus.FAILED
                    metadata.errors.append("Pre-restore validation failed")
                    await self._save_restore_metadata(metadata)
                    raise Exception("Pre-restore validation failed")

            # Extract backup
            metadata.status = RestoreStatus.IN_PROGRESS
            await self._save_restore_metadata(metadata)

            extracted_path = await self._extract_backup(backup_id)

            # Restore components
            restore_tasks = []

            if "rbac_system" in metadata.components_to_restore:
                restore_tasks.append(self._restore_rbac_system(extracted_path, metadata))

            if "tenant_data" in metadata.components_to_restore:
                restore_tasks.append(self._restore_tenant_data(extracted_path, metadata))

            if "agents" in metadata.components_to_restore:
                restore_tasks.append(self._restore_agents(extracted_path, metadata))

            if "rag_collections" in metadata.components_to_restore:
                restore_tasks.append(self._restore_rag_collections(extracted_path, metadata))

            if "p2p_networks" in metadata.components_to_restore:
                restore_tasks.append(self._restore_p2p_networks(extracted_path, metadata))

            if "models" in metadata.components_to_restore:
                restore_tasks.append(self._restore_models(extracted_path, metadata))

            if "configurations" in metadata.components_to_restore:
                restore_tasks.append(self._restore_configurations(extracted_path, metadata))

            if "logs" in metadata.components_to_restore:
                restore_tasks.append(self._restore_logs(extracted_path, metadata))

            # Execute restore tasks
            results = await asyncio.gather(*restore_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component = metadata.components_to_restore[i]
                    metadata.errors.append(f"Failed to restore {component}: {result}")
                elif isinstance(result, dict):
                    if result.get("component"):
                        metadata.restored_components.append(result["component"])
                    if result.get("errors"):
                        metadata.errors.extend(result["errors"])
                    if result.get("warnings"):
                        metadata.warnings.extend(result["warnings"])

            # Cleanup temporary extraction
            if extracted_path.exists():
                shutil.rmtree(extracted_path)

            # Post-restore validation
            if metadata.post_restore_validation:
                verification_results = await self._verify_restore(metadata)
                metadata.verification_results = verification_results

                all_verified = all(verification_results.values())
                if not all_verified:
                    metadata.warnings.append("Some components failed post-restore verification")

            # Update status
            if metadata.errors:
                metadata.status = RestoreStatus.FAILED
            else:
                metadata.status = RestoreStatus.COMPLETED

            metadata.completed_at = datetime.utcnow()
            metadata.progress_percent = 100.0
            await self._save_restore_metadata(metadata)

            logger.info(f"Full system restore {restore_id} completed")

            return restore_id

        except Exception as e:
            metadata.status = RestoreStatus.FAILED
            metadata.errors.append(str(e))
            await self._save_restore_metadata(metadata)
            logger.error(f"Full system restore {restore_id} failed: {e}")

            # Attempt rollback if needed
            if create_rollback and metadata.rollback_point_created:
                logger.info("Attempting automatic rollback...")
                try:
                    await self._rollback_restore(metadata)
                    metadata.status = RestoreStatus.ROLLED_BACK
                    await self._save_restore_metadata(metadata)
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
                    metadata.errors.append(f"Rollback failed: {rollback_error}")
                    await self._save_restore_metadata(metadata)

            raise

    async def restore_tenant(
        self, backup_id: str, tenant_id: str, strategy: RestoreStrategy = RestoreStrategy.REPLACE
    ) -> str:
        """Restore specific tenant from backup."""
        restore_id = f"tenant_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Validate backup contains tenant
        backup_metadata = await self.backup_manager.get_backup_info(backup_id)
        if not backup_metadata:
            raise ValueError(f"Backup {backup_id} not found")

        if backup_metadata.tenants_included and tenant_id not in backup_metadata.tenants_included:
            raise ValueError(f"Tenant {tenant_id} not included in backup {backup_id}")

        metadata = RestoreMetadata(
            restore_id=restore_id,
            backup_id=backup_id,
            restore_type=RestoreType.TENANT_ONLY,
            strategy=strategy,
            created_at=datetime.utcnow(),
            tenants_to_restore=[tenant_id],
            components_to_restore=["tenant_data", "agents", "rag_collections", "p2p_networks"],
        )

        self.active_restores[restore_id] = metadata

        try:
            logger.info(f"Starting tenant restore {restore_id} for tenant {tenant_id}")

            metadata.status = RestoreStatus.IN_PROGRESS
            await self._save_restore_metadata(metadata)

            # Extract backup
            extracted_path = await self._extract_backup(backup_id)

            # Restore tenant-specific components
            restore_tasks = [
                self._restore_tenant_data(extracted_path, metadata, tenant_filter=[tenant_id]),
                self._restore_agents(extracted_path, metadata, tenant_filter=[tenant_id]),
                self._restore_rag_collections(extracted_path, metadata, tenant_filter=[tenant_id]),
                self._restore_p2p_networks(extracted_path, metadata, tenant_filter=[tenant_id]),
            ]

            results = await asyncio.gather(*restore_tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    metadata.errors.append(str(result))
                elif isinstance(result, dict):
                    if result.get("component"):
                        metadata.restored_components.append(result["component"])
                    if result.get("errors"):
                        metadata.errors.extend(result["errors"])
                    if result.get("warnings"):
                        metadata.warnings.extend(result["warnings"])

            # Cleanup
            if extracted_path.exists():
                shutil.rmtree(extracted_path)

            # Update status
            metadata.status = RestoreStatus.COMPLETED if not metadata.errors else RestoreStatus.FAILED
            metadata.completed_at = datetime.utcnow()
            metadata.progress_percent = 100.0
            await self._save_restore_metadata(metadata)

            logger.info(f"Tenant restore {restore_id} completed for {tenant_id}")

            return restore_id

        except Exception as e:
            metadata.status = RestoreStatus.FAILED
            metadata.errors.append(str(e))
            await self._save_restore_metadata(metadata)
            logger.error(f"Tenant restore {restore_id} failed: {e}")
            raise

    async def restore_component(
        self, backup_id: str, component: str, strategy: RestoreStrategy = RestoreStrategy.REPLACE
    ) -> str:
        """Restore specific component from backup."""
        restore_id = f"comp_{component}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Validate backup contains component
        backup_metadata = await self.backup_manager.get_backup_info(backup_id)
        if not backup_metadata:
            raise ValueError(f"Backup {backup_id} not found")

        if component not in backup_metadata.components_included:
            raise ValueError(f"Component {component} not included in backup {backup_id}")

        metadata = RestoreMetadata(
            restore_id=restore_id,
            backup_id=backup_id,
            restore_type=RestoreType.COMPONENT_ONLY,
            strategy=strategy,
            created_at=datetime.utcnow(),
            components_to_restore=[component],
        )

        self.active_restores[restore_id] = metadata

        try:
            logger.info(f"Starting component restore {restore_id} for {component}")

            metadata.status = RestoreStatus.IN_PROGRESS
            await self._save_restore_metadata(metadata)

            # Extract backup
            extracted_path = await self._extract_backup(backup_id)

            # Restore specific component
            result = None
            if component == "rbac_system":
                result = await self._restore_rbac_system(extracted_path, metadata)
            elif component == "tenant_data":
                result = await self._restore_tenant_data(extracted_path, metadata)
            elif component == "agents":
                result = await self._restore_agents(extracted_path, metadata)
            elif component == "rag_collections":
                result = await self._restore_rag_collections(extracted_path, metadata)
            elif component == "p2p_networks":
                result = await self._restore_p2p_networks(extracted_path, metadata)
            elif component == "models":
                result = await self._restore_models(extracted_path, metadata)
            elif component == "configurations":
                result = await self._restore_configurations(extracted_path, metadata)
            elif component == "logs":
                result = await self._restore_logs(extracted_path, metadata)

            # Process result
            if isinstance(result, dict):
                if result.get("component"):
                    metadata.restored_components.append(result["component"])
                if result.get("errors"):
                    metadata.errors.extend(result["errors"])
                if result.get("warnings"):
                    metadata.warnings.extend(result["warnings"])

            # Cleanup
            if extracted_path.exists():
                shutil.rmtree(extracted_path)

            # Update status
            metadata.status = RestoreStatus.COMPLETED if not metadata.errors else RestoreStatus.FAILED
            metadata.completed_at = datetime.utcnow()
            metadata.progress_percent = 100.0
            await self._save_restore_metadata(metadata)

            logger.info(f"Component restore {restore_id} completed for {component}")

            return restore_id

        except Exception as e:
            metadata.status = RestoreStatus.FAILED
            metadata.errors.append(str(e))
            await self._save_restore_metadata(metadata)
            logger.error(f"Component restore {restore_id} failed: {e}")
            raise

    # Component-specific restore methods

    async def _restore_rbac_system(self, extracted_path: Path, metadata: RestoreMetadata) -> dict[str, Any]:
        """Restore RBAC system data."""
        logger.info("Restoring RBAC system")

        try:
            rbac_source = extracted_path / "rbac"
            if not rbac_source.exists():
                return {"component": "rbac_system", "errors": ["RBAC data not found in backup"], "warnings": []}

            errors = []
            warnings = []

            # Restore RBAC database
            if (rbac_source / "rbac.db").exists():
                rbac_db_dest = Path("data/rbac.db")
                rbac_db_dest.parent.mkdir(parents=True, exist_ok=True)

                if metadata.strategy == RestoreStrategy.REPLACE:
                    if rbac_db_dest.exists():
                        rbac_db_dest.rename(rbac_db_dest.with_suffix(".db.backup"))
                    shutil.copy2(rbac_source / "rbac.db", rbac_db_dest)
                else:
                    warnings.append("RBAC database merge not implemented, using replace strategy")
                    shutil.copy2(rbac_source / "rbac.db", rbac_db_dest)

            # Restore tenant database
            if (rbac_source / "tenants.db").exists():
                tenant_db_dest = Path("data/tenants.db")
                tenant_db_dest.parent.mkdir(parents=True, exist_ok=True)

                if metadata.strategy == RestoreStrategy.REPLACE:
                    if tenant_db_dest.exists():
                        tenant_db_dest.rename(tenant_db_dest.with_suffix(".db.backup"))
                    shutil.copy2(rbac_source / "tenants.db", tenant_db_dest)
                else:
                    warnings.append("Tenant database merge not implemented, using replace strategy")
                    shutil.copy2(rbac_source / "tenants.db", tenant_db_dest)

            # Restore security configuration
            if (rbac_source / "security_config").exists():
                security_dest = Path("config/security")
                security_dest.parent.mkdir(parents=True, exist_ok=True)

                if metadata.strategy == RestoreStrategy.REPLACE:
                    if security_dest.exists():
                        shutil.move(security_dest, security_dest.with_suffix(".backup"))
                    shutil.copytree(rbac_source / "security_config", security_dest)
                else:
                    # Merge security configs
                    for config_file in (rbac_source / "security_config").rglob("*"):
                        if config_file.is_file():
                            relative_path = config_file.relative_to(rbac_source / "security_config")
                            dest_file = security_dest / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(config_file, dest_file)

            return {"component": "rbac_system", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore RBAC system: {e}")
            return {"component": "rbac_system", "errors": [str(e)], "warnings": []}

    async def _restore_tenant_data(
        self, extracted_path: Path, metadata: RestoreMetadata, tenant_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """Restore tenant data."""
        logger.info(f"Restoring tenant data (filter: {tenant_filter})")

        try:
            tenants_source = extracted_path / "tenants"
            if not tenants_source.exists():
                return {"component": "tenant_data", "errors": ["Tenant data not found in backup"], "warnings": []}

            errors = []
            warnings = []

            # Restore tenant directories
            for tenant_dir in tenants_source.iterdir():
                if not tenant_dir.is_dir():
                    continue

                tenant_id = tenant_dir.name

                # Apply tenant filter
                if tenant_filter and tenant_id not in tenant_filter:
                    continue

                dest_path = Path(f"data/tenants/{tenant_id}")

                try:
                    if metadata.strategy == RestoreStrategy.REPLACE:
                        if dest_path.exists():
                            shutil.move(dest_path, dest_path.with_suffix(".backup"))
                        shutil.copytree(tenant_dir, dest_path)
                    else:
                        # Merge tenant data
                        dest_path.mkdir(parents=True, exist_ok=True)
                        for item in tenant_dir.rglob("*"):
                            if item.is_file():
                                relative_path = item.relative_to(tenant_dir)
                                dest_file = dest_path / relative_path
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, dest_file)
                except Exception as e:
                    errors.append(f"Failed to restore tenant {tenant_id}: {e}")

            return {"component": "tenant_data", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore tenant data: {e}")
            return {"component": "tenant_data", "errors": [str(e)], "warnings": []}

    async def _restore_agents(
        self, extracted_path: Path, metadata: RestoreMetadata, tenant_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """Restore agent configurations."""
        logger.info("Restoring agents")

        try:
            agents_source = extracted_path / "agents"
            if not agents_source.exists():
                return {"component": "agents", "errors": ["Agent data not found in backup"], "warnings": []}

            errors = []
            warnings = []

            # Restore agent configurations
            if (agents_source / "configs").exists():
                configs_dest = Path("packages/agents")
                for config_file in (agents_source / "configs").rglob("*.json"):
                    relative_path = config_file.relative_to(agents_source / "configs")
                    dest_file = configs_dest / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    if metadata.strategy == RestoreStrategy.REPLACE or not dest_file.exists():
                        shutil.copy2(config_file, dest_file)
                    else:
                        warnings.append(f"Agent config {relative_path} already exists, skipping")

            # Restore tenant-specific agent data
            if (agents_source / "tenant_data").exists():
                for tenant_dir in (agents_source / "tenant_data").iterdir():
                    if not tenant_dir.is_dir():
                        continue

                    tenant_id = tenant_dir.name

                    # Apply tenant filter
                    if tenant_filter and tenant_id not in tenant_filter:
                        continue

                    dest_path = Path(f"data/tenants/{tenant_id}/agents")

                    if metadata.strategy == RestoreStrategy.REPLACE:
                        if dest_path.exists():
                            shutil.move(dest_path, dest_path.with_suffix(".backup"))
                        shutil.copytree(tenant_dir, dest_path)
                    else:
                        dest_path.mkdir(parents=True, exist_ok=True)
                        for item in tenant_dir.rglob("*"):
                            if item.is_file():
                                relative_path = item.relative_to(tenant_dir)
                                dest_file = dest_path / relative_path
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, dest_file)

            return {"component": "agents", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore agents: {e}")
            return {"component": "agents", "errors": [str(e)], "warnings": []}

    async def _restore_rag_collections(
        self, extracted_path: Path, metadata: RestoreMetadata, tenant_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """Restore RAG collections."""
        logger.info("Restoring RAG collections")

        try:
            rag_source = extracted_path / "rag"
            if not rag_source.exists():
                return {"component": "rag_collections", "errors": ["RAG data not found in backup"], "warnings": []}

            errors = []
            warnings = []

            # Restore RAG configurations
            if (rag_source / "configs").exists():
                configs_dest = Path("packages/rag")
                for config_file in (rag_source / "configs").rglob("*.json"):
                    relative_path = config_file.relative_to(rag_source / "configs")
                    dest_file = configs_dest / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, dest_file)

            # Restore vector databases
            if (rag_source / "vector_db").exists():
                vector_db_dest = Path("data/vector_db")
                if metadata.strategy == RestoreStrategy.REPLACE:
                    if vector_db_dest.exists():
                        shutil.move(vector_db_dest, vector_db_dest.with_suffix(".backup"))
                    shutil.copytree(rag_source / "vector_db", vector_db_dest)
                else:
                    warnings.append("Vector DB merge not fully implemented, using copy strategy")
                    vector_db_dest.mkdir(parents=True, exist_ok=True)
                    for item in (rag_source / "vector_db").rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(rag_source / "vector_db")
                            dest_file = vector_db_dest / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_file)

            # Restore tenant RAG data
            if (rag_source / "tenant_data").exists():
                for tenant_dir in (rag_source / "tenant_data").iterdir():
                    if not tenant_dir.is_dir():
                        continue

                    tenant_id = tenant_dir.name

                    if tenant_filter and tenant_id not in tenant_filter:
                        continue

                    dest_path = Path(f"data/tenants/{tenant_id}/rag")

                    if metadata.strategy == RestoreStrategy.REPLACE:
                        if dest_path.exists():
                            shutil.move(dest_path, dest_path.with_suffix(".backup"))
                        shutil.copytree(tenant_dir, dest_path)
                    else:
                        dest_path.mkdir(parents=True, exist_ok=True)
                        for item in tenant_dir.rglob("*"):
                            if item.is_file():
                                relative_path = item.relative_to(tenant_dir)
                                dest_file = dest_path / relative_path
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, dest_file)

            return {"component": "rag_collections", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore RAG collections: {e}")
            return {"component": "rag_collections", "errors": [str(e)], "warnings": []}

    async def _restore_p2p_networks(
        self, extracted_path: Path, metadata: RestoreMetadata, tenant_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """Restore P2P network configurations."""
        logger.info("Restoring P2P networks")

        try:
            p2p_source = extracted_path / "p2p"
            if not p2p_source.exists():
                return {"component": "p2p_networks", "warnings": ["P2P data not found in backup"]}

            errors = []
            warnings = []

            # Restore P2P configurations
            if (p2p_source / "configs").exists():
                configs_dest = Path("packages/p2p")
                for config_file in (p2p_source / "configs").rglob("*.json"):
                    relative_path = config_file.relative_to(p2p_source / "configs")
                    dest_file = configs_dest / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, dest_file)

            # Restore tenant P2P data
            if (p2p_source / "tenant_data").exists():
                for tenant_dir in (p2p_source / "tenant_data").iterdir():
                    if not tenant_dir.is_dir():
                        continue

                    tenant_id = tenant_dir.name

                    if tenant_filter and tenant_id not in tenant_filter:
                        continue

                    dest_path = Path(f"data/tenants/{tenant_id}/p2p")
                    dest_path.mkdir(parents=True, exist_ok=True)

                    for item in tenant_dir.rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(tenant_dir)
                            dest_file = dest_path / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_file)

            return {"component": "p2p_networks", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore P2P networks: {e}")
            return {"component": "p2p_networks", "errors": [str(e)], "warnings": []}

    async def _restore_models(
        self, extracted_path: Path, metadata: RestoreMetadata, tenant_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """Restore AI models."""
        logger.info("Restoring models")

        try:
            models_source = extracted_path / "models"
            if not models_source.exists():
                return {"component": "models", "warnings": ["Model data not found in backup"]}

            errors = []
            warnings = []

            # Restore Agent Forge models
            if (models_source / "agent_forge").exists():
                agent_forge_dest = Path("packages/agent_forge")
                for model_file in (models_source / "agent_forge").rglob("*"):
                    if model_file.is_file() and model_file.suffix in [".bin", ".safetensors"]:
                        relative_path = model_file.relative_to(models_source / "agent_forge")
                        dest_file = agent_forge_dest / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        if metadata.strategy == RestoreStrategy.REPLACE or not dest_file.exists():
                            shutil.copy2(model_file, dest_file)
                        else:
                            warnings.append(f"Model {relative_path} already exists, skipping")

            # Restore tenant models
            if (models_source / "tenant_models").exists():
                for tenant_dir in (models_source / "tenant_models").iterdir():
                    if not tenant_dir.is_dir():
                        continue

                    tenant_id = tenant_dir.name

                    if tenant_filter and tenant_id not in tenant_filter:
                        continue

                    dest_path = Path(f"data/tenants/{tenant_id}/models")
                    dest_path.mkdir(parents=True, exist_ok=True)

                    for item in tenant_dir.rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(tenant_dir)
                            dest_file = dest_path / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_file)

            return {"component": "models", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore models: {e}")
            return {"component": "models", "errors": [str(e)], "warnings": []}

    async def _restore_configurations(self, extracted_path: Path, metadata: RestoreMetadata) -> dict[str, Any]:
        """Restore system configurations."""
        logger.info("Restoring configurations")

        try:
            config_source = extracted_path / "configurations"
            if not config_source.exists():
                return {"component": "configurations", "warnings": ["Configuration data not found in backup"]}

            errors = []
            warnings = []

            # Restore system config
            if (config_source / "system_config").exists():
                config_dest = Path("config")

                if metadata.strategy == RestoreStrategy.REPLACE:
                    if config_dest.exists():
                        shutil.move(config_dest, config_dest.with_suffix(".backup"))
                    shutil.copytree(config_source / "system_config", config_dest)
                else:
                    config_dest.mkdir(parents=True, exist_ok=True)
                    for item in (config_source / "system_config").rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(config_source / "system_config")
                            dest_file = config_dest / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_file)

            # Restore deployment config
            if (config_source / "deployment_config").exists():
                deploy_dest = Path("deploy")

                if metadata.strategy == RestoreStrategy.REPLACE:
                    if deploy_dest.exists():
                        shutil.move(deploy_dest, deploy_dest.with_suffix(".backup"))
                    shutil.copytree(config_source / "deployment_config", deploy_dest)
                else:
                    deploy_dest.mkdir(parents=True, exist_ok=True)
                    for item in (config_source / "deployment_config").rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(config_source / "deployment_config")
                            dest_file = deploy_dest / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_file)

            # Restore environment files
            for env_file in [".env", ".env.production", ".env.local"]:
                env_source = config_source / env_file
                if env_source.exists():
                    env_dest = Path(env_file)
                    if metadata.strategy == RestoreStrategy.REPLACE or not env_dest.exists():
                        shutil.copy2(env_source, env_dest)
                    else:
                        warnings.append(f"Environment file {env_file} already exists, skipping")

            return {"component": "configurations", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore configurations: {e}")
            return {"component": "configurations", "errors": [str(e)], "warnings": []}

    async def _restore_logs(self, extracted_path: Path, metadata: RestoreMetadata) -> dict[str, Any]:
        """Restore system logs."""
        logger.info("Restoring logs")

        try:
            logs_source = extracted_path / "logs"
            if not logs_source.exists():
                return {"component": "logs", "warnings": ["Log data not found in backup"]}

            errors = []
            warnings = []

            # Restore logs to separate restore directory to avoid conflicts
            logs_dest = Path("data/restored_logs")
            logs_dest.mkdir(parents=True, exist_ok=True)

            for log_dir in logs_source.iterdir():
                if log_dir.is_dir():
                    dest_dir = logs_dest / log_dir.name
                    shutil.copytree(log_dir, dest_dir, dirs_exist_ok=True)

            warnings.append("Logs restored to data/restored_logs to avoid conflicts with current logs")

            return {"component": "logs", "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to restore logs: {e}")
            return {"component": "logs", "errors": [str(e)], "warnings": []}

    # Support methods

    async def _extract_backup(self, backup_id: str) -> Path:
        """Extract backup archive to temporary directory."""
        # Find backup file
        backup_file = None
        for backup_dir in [
            self.backup_manager.full_backups_dir,
            self.backup_manager.incremental_backups_dir,
            self.backup_manager.tenant_backups_dir,
            self.backup_manager.config_backups_dir,
        ]:
            potential_files = [backup_dir / f"{backup_id}.tar.gz", backup_dir / f"{backup_id}.tar.gz.enc"]

            for potential_file in potential_files:
                if potential_file.exists():
                    backup_file = potential_file
                    break

            if backup_file:
                break

        if not backup_file:
            raise FileNotFoundError(f"Backup file not found for {backup_id}")

        # Create temporary extraction directory
        extract_dir = self.temp_restore_dir / f"extract_{backup_id}_{int(time.time())}"
        extract_dir.mkdir()

        try:
            # Decrypt if needed
            if backup_file.suffix == ".enc":
                decrypted_file = extract_dir / f"{backup_id}.tar.gz"
                await self.backup_manager._decrypt_file(backup_file, decrypted_file)
                archive_file = decrypted_file
            else:
                archive_file = backup_file

            # Extract archive
            with tarfile.open(archive_file, "r:gz") as tar:
                tar.extractall(extract_dir)

            # Clean up decrypted file if created
            if backup_file.suffix == ".enc":
                decrypted_file.unlink()

            # Return path to extracted content
            extracted_content = extract_dir / backup_id
            if not extracted_content.exists():
                # Try to find the actual content directory
                subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
                if subdirs:
                    extracted_content = subdirs[0]
                else:
                    extracted_content = extract_dir

            return extracted_content

        except Exception:
            # Cleanup on failure
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            raise

    async def _create_rollback_point(self, metadata: RestoreMetadata):
        """Create rollback point before restore."""
        logger.info("Creating rollback point")

        try:
            # Create rollback directory
            rollback_id = f"rollback_{metadata.restore_id}_{int(time.time())}"
            rollback_path = self.rollback_dir / rollback_id
            rollback_path.mkdir()

            # Backup current state of components being restored
            for component in metadata.components_to_restore:
                if component == "rbac_system":
                    # Backup RBAC databases
                    rbac_rollback = rollback_path / "rbac"
                    rbac_rollback.mkdir()

                    for db_file in ["data/rbac.db", "data/tenants.db"]:
                        db_path = Path(db_file)
                        if db_path.exists():
                            shutil.copy2(db_path, rbac_rollback / db_path.name)

                    # Backup security config
                    security_config = Path("config/security")
                    if security_config.exists():
                        shutil.copytree(security_config, rbac_rollback / "security")

                elif component == "configurations":
                    # Backup config directory
                    config_path = Path("config")
                    if config_path.exists():
                        shutil.copytree(config_path, rollback_path / "config")

                # Add other components as needed

            metadata.rollback_point_created = True
            metadata.rollback_data_path = rollback_path

            logger.info(f"Rollback point created at {rollback_path}")

        except Exception as e:
            logger.warning(f"Failed to create rollback point: {e}")
            metadata.rollback_point_created = False

    async def _rollback_restore(self, metadata: RestoreMetadata):
        """Rollback restore operation."""
        if not metadata.rollback_point_created or not metadata.rollback_data_path:
            raise Exception("No rollback point available")

        rollback_path = metadata.rollback_data_path
        if not rollback_path.exists():
            raise Exception("Rollback data not found")

        logger.info(f"Rolling back restore {metadata.restore_id}")

        # Restore from rollback point
        for component_dir in rollback_path.iterdir():
            if component_dir.name == "rbac":
                # Restore RBAC system
                for db_file in component_dir.glob("*.db"):
                    dest_path = Path("data") / db_file.name
                    shutil.copy2(db_file, dest_path)

                if (component_dir / "security").exists():
                    security_dest = Path("config/security")
                    if security_dest.exists():
                        shutil.rmtree(security_dest)
                    shutil.copytree(component_dir / "security", security_dest)

            elif component_dir.name == "config":
                # Restore configuration
                config_dest = Path("config")
                if config_dest.exists():
                    shutil.rmtree(config_dest)
                shutil.copytree(component_dir, config_dest)

        logger.info("Rollback completed")

    async def _validate_restore_prerequisites(self, metadata: RestoreMetadata) -> bool:
        """Validate prerequisites for restore operation."""
        logger.info("Validating restore prerequisites")

        try:
            # Check backup exists and is valid
            backup_metadata = await self.backup_manager.get_backup_info(metadata.backup_id)
            if not backup_metadata or backup_metadata.status != BackupStatus.COMPLETED:
                logger.error("Backup is not in completed state")
                return False

            # Verify backup integrity
            if not await self.backup_manager._verify_backup(metadata.backup_id):
                logger.error("Backup integrity verification failed")
                return False

            # Check disk space
            backup_size = backup_metadata.size_bytes
            available_space = shutil.disk_usage(Path.cwd()).free

            if backup_size * 2 > available_space:  # Need 2x space for extraction
                logger.error(f"Insufficient disk space. Need {backup_size * 2}, have {available_space}")
                return False

            # Check if system is in suitable state for restore
            # (e.g., no active operations, databases accessible)

            return True

        except Exception as e:
            logger.error(f"Prerequisites validation failed: {e}")
            return False

    async def _verify_restore(self, metadata: RestoreMetadata) -> dict[str, bool]:
        """Verify restore operation results."""
        logger.info("Verifying restore results")

        verification_results = {}

        for component in metadata.restored_components:
            try:
                if component == "rbac_system":
                    # Verify RBAC databases exist and are accessible
                    rbac_db = Path("data/rbac.db")
                    tenant_db = Path("data/tenants.db")

                    rbac_ok = rbac_db.exists() and rbac_db.stat().st_size > 0
                    tenant_ok = tenant_db.exists() and tenant_db.stat().st_size > 0

                    verification_results[component] = rbac_ok and tenant_ok

                elif component == "configurations":
                    # Verify config directory exists
                    config_dir = Path("config")
                    verification_results[component] = config_dir.exists()

                elif component == "tenant_data":
                    # Verify tenant directories exist
                    tenants_dir = Path("data/tenants")
                    has_tenants = tenants_dir.exists() and any(tenants_dir.iterdir())
                    verification_results[component] = has_tenants

                else:
                    # Default verification - just check for existence
                    verification_results[component] = True

            except Exception as e:
                logger.warning(f"Verification failed for {component}: {e}")
                verification_results[component] = False

        return verification_results

    async def _save_restore_metadata(self, metadata: RestoreMetadata):
        """Save restore metadata to database."""
        conn = sqlite3.connect(self.restore_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO restores
            (restore_id, backup_id, restore_type, strategy, created_at, completed_at,
             status, components_to_restore, tenants_to_restore, rollback_point_created,
             rollback_data_path, total_files, processed_files, progress_percent,
             errors, warnings, restored_components, verification_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metadata.restore_id,
                metadata.backup_id,
                metadata.restore_type.value,
                metadata.strategy.value,
                metadata.created_at.isoformat(),
                metadata.completed_at.isoformat() if metadata.completed_at else None,
                metadata.status.value,
                json.dumps(metadata.components_to_restore),
                json.dumps(metadata.tenants_to_restore),
                metadata.rollback_point_created,
                str(metadata.rollback_data_path) if metadata.rollback_data_path else None,
                metadata.total_files,
                metadata.processed_files,
                metadata.progress_percent,
                json.dumps(metadata.errors),
                json.dumps(metadata.warnings),
                json.dumps(metadata.restored_components),
                json.dumps(metadata.verification_results),
            ),
        )

        conn.commit()
        conn.close()

    # Public query methods

    async def list_restores(self, status: RestoreStatus | None = None, limit: int = 50) -> list[RestoreMetadata]:
        """List restore operations."""
        conn = sqlite3.connect(self.restore_db)
        cursor = conn.cursor()

        query = "SELECT * FROM restores"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        restores = []
        for row in rows:
            metadata = RestoreMetadata(
                restore_id=row[0],
                backup_id=row[1],
                restore_type=RestoreType(row[2]),
                strategy=RestoreStrategy(row[3]),
                created_at=datetime.fromisoformat(row[4]),
                completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                status=RestoreStatus(row[6]),
                components_to_restore=json.loads(row[7]) if row[7] else [],
                tenants_to_restore=json.loads(row[8]) if row[8] else [],
                rollback_point_created=bool(row[9]),
                rollback_data_path=Path(row[10]) if row[10] else None,
                total_files=row[11],
                processed_files=row[12],
                progress_percent=row[13],
                errors=json.loads(row[14]) if row[14] else [],
                warnings=json.loads(row[15]) if row[15] else [],
                restored_components=json.loads(row[16]) if row[16] else [],
                verification_results=json.loads(row[17]) if row[17] else {},
            )
            restores.append(metadata)

        return restores

    async def get_restore_status(self, restore_id: str) -> RestoreMetadata | None:
        """Get detailed restore status."""
        # Check active restores first
        if restore_id in self.active_restores:
            return self.active_restores[restore_id]

        # Query database
        conn = sqlite3.connect(self.restore_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM restores WHERE restore_id = ?", (restore_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return RestoreMetadata(
            restore_id=row[0],
            backup_id=row[1],
            restore_type=RestoreType(row[2]),
            strategy=RestoreStrategy(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
            status=RestoreStatus(row[6]),
            components_to_restore=json.loads(row[7]) if row[7] else [],
            tenants_to_restore=json.loads(row[8]) if row[8] else [],
            rollback_point_created=bool(row[9]),
            rollback_data_path=Path(row[10]) if row[10] else None,
            total_files=row[11],
            processed_files=row[12],
            progress_percent=row[13],
            errors=json.loads(row[14]) if row[14] else [],
            warnings=json.loads(row[15]) if row[15] else [],
            restored_components=json.loads(row[16]) if row[16] else [],
            verification_results=json.loads(row[17]) if row[17] else {},
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        from .backup_manager import BackupManager

        # Initialize managers
        backup_manager = BackupManager()
        restore_manager = RestoreManager(backup_manager)

        # List available backups
        backups = await backup_manager.list_backups()
        if backups:
            latest_backup = backups[0]
            print(f"Latest backup: {latest_backup.backup_id}")

            # Test restore (component only for safety)
            restore_id = await restore_manager.restore_component(
                backup_id=latest_backup.backup_id, component="configurations", strategy=RestoreStrategy.SIDE_BY_SIDE
            )

            print(f"Restore completed: {restore_id}")

            # Check restore status
            restore_status = await restore_manager.get_restore_status(restore_id)
            print(f"Restore status: {restore_status.status.value}")
        else:
            print("No backups available")

    asyncio.run(main())
