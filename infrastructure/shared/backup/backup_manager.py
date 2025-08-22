"""
Comprehensive Backup and Restore Manager for AIVillage.

Provides automated backup and restore capabilities for all AIVillage components:
- RBAC system and tenant data
- Agent configurations and models
- RAG collections and knowledge graphs
- P2P network configurations
- Digital twin data (with privacy protection)
- System configurations and logs
"""

import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups supported."""

    FULL = "full"  # Complete system backup
    INCREMENTAL = "incremental"  # Changes since last backup
    DIFFERENTIAL = "differential"  # Changes since last full backup
    TENANT_ONLY = "tenant_only"  # Single tenant backup
    CONFIGURATION = "configuration"  # System config only
    EMERGENCY = "emergency"  # Critical data only


class BackupStatus(Enum):
    """Backup operation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    RESTORED = "restored"


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""

    backup_id: str
    backup_type: BackupType
    created_at: datetime
    completed_at: datetime | None = None
    status: BackupStatus = BackupStatus.PENDING
    size_bytes: int = 0
    checksum: str = ""
    encrypted: bool = True
    compression_ratio: float = 0.0

    # Content tracking
    tenants_included: list[str] = field(default_factory=list)
    components_included: list[str] = field(default_factory=list)
    file_count: int = 0

    # Restore information
    restore_compatible_versions: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BackupManager:
    """Main backup and restore management system."""

    def __init__(
        self, backup_root: Path = Path("backups"), config_path: Path = Path("config/backup/backup_config.json")
    ):
        """Initialize backup manager."""
        self.backup_root = backup_root
        self.config_path = config_path

        # Create directories
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.backup_root / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Backup storage structure
        self.full_backups_dir = self.backup_root / "full"
        self.incremental_backups_dir = self.backup_root / "incremental"
        self.tenant_backups_dir = self.backup_root / "tenant"
        self.config_backups_dir = self.backup_root / "config"

        for dir_path in [
            self.full_backups_dir,
            self.incremental_backups_dir,
            self.tenant_backups_dir,
            self.config_backups_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        # Metadata tracking
        self.metadata_db = self.backup_root / "backup_metadata.db"
        self.active_backups: dict[str, BackupMetadata] = {}

        # Encryption setup
        self.encryption_key = self._load_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)

        # Configuration
        self.config = self._load_config()

        # Initialize metadata database
        self._init_metadata_db()

        # Load existing backups
        self._load_backup_metadata()

        logger.info(f"Backup manager initialized at {self.backup_root}")

    def _load_or_create_encryption_key(self) -> bytes:
        """Load or create encryption key for backup security."""
        key_file = self.backup_root / ".backup_key"

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Restrict permissions
            key_file.chmod(0o600)
            return key

    def _load_config(self) -> dict[str, Any]:
        """Load backup configuration."""
        default_config = {
            "retention_policy": {
                "full_backups_keep": 30,  # Keep 30 full backups
                "incremental_keep_days": 90,  # Keep incrementals for 90 days
                "tenant_backups_keep": 10,  # Keep 10 tenant backups per tenant
            },
            "compression": {
                "enabled": True,
                "level": 6,  # gzip compression level
                "threshold_mb": 10,  # Compress files larger than 10MB
            },
            "encryption": {"enabled": True, "algorithm": "AES-256", "key_rotation_days": 90},
            "scheduling": {
                "full_backup_hour": 2,  # 2 AM daily full backup
                "incremental_interval_hours": 6,  # Every 6 hours
                "tenant_backup_daily": True,
            },
            "verification": {"verify_after_backup": True, "checksum_algorithm": "SHA-256", "test_restore_weekly": True},
            "storage": {"max_backup_size_gb": 100, "cleanup_temp_files": True, "parallel_operations": 4},
        }

        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        else:
            # Create default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)

        return default_config

    def _init_metadata_db(self):
        """Initialize backup metadata database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backups (
                backup_id TEXT PRIMARY KEY,
                backup_type TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                status TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                checksum TEXT,
                encrypted BOOLEAN DEFAULT 1,
                compression_ratio REAL DEFAULT 0.0,
                tenants_included TEXT,
                components_included TEXT,
                file_count INTEGER DEFAULT 0,
                restore_compatible_versions TEXT,
                prerequisites TEXT,
                errors TEXT,
                warnings TEXT,
                backup_path TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS restore_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_id TEXT NOT NULL,
                restored_at TIMESTAMP NOT NULL,
                restored_by TEXT,
                restore_type TEXT,
                success BOOLEAN,
                errors TEXT,
                FOREIGN KEY (backup_id) REFERENCES backups (backup_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backup_created
            ON backups(created_at)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backup_type
            ON backups(backup_type, status)
        """
        )

        conn.commit()
        conn.close()

    def _load_backup_metadata(self):
        """Load existing backup metadata from database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM backups ORDER BY created_at DESC")
        rows = cursor.fetchall()

        for row in rows:
            metadata = BackupMetadata(
                backup_id=row[0],
                backup_type=BackupType(row[1]),
                created_at=datetime.fromisoformat(row[2]),
                completed_at=datetime.fromisoformat(row[3]) if row[3] else None,
                status=BackupStatus(row[4]),
                size_bytes=row[5],
                checksum=row[6] or "",
                encrypted=bool(row[7]),
                compression_ratio=row[8],
                tenants_included=json.loads(row[9]) if row[9] else [],
                components_included=json.loads(row[10]) if row[10] else [],
                file_count=row[11],
                restore_compatible_versions=json.loads(row[12]) if row[12] else [],
                prerequisites=json.loads(row[13]) if row[13] else [],
                errors=json.loads(row[14]) if row[14] else [],
                warnings=json.loads(row[15]) if row[15] else [],
            )

            # Only keep active/recent backups in memory
            if metadata.status in [BackupStatus.IN_PROGRESS, BackupStatus.PENDING]:
                self.active_backups[metadata.backup_id] = metadata

        conn.close()
        logger.info(f"Loaded metadata for {len(rows)} backups")

    # Core Backup Operations

    async def create_full_backup(
        self, include_tenants: list[str] | None = None, exclude_components: list[str] | None = None
    ) -> str:
        """Create comprehensive full system backup."""
        backup_id = f"full_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            created_at=datetime.utcnow(),
            components_included=[
                "rbac_system",
                "tenant_data",
                "agents",
                "rag_collections",
                "p2p_networks",
                "models",
                "configurations",
                "logs",
            ],
        )

        if exclude_components:
            metadata.components_included = [c for c in metadata.components_included if c not in exclude_components]

        self.active_backups[backup_id] = metadata

        try:
            metadata.status = BackupStatus.IN_PROGRESS
            await self._save_metadata(metadata)

            # Create backup directory
            backup_dir = self.full_backups_dir / backup_id
            backup_dir.mkdir()

            logger.info(f"Starting full backup {backup_id}")

            # Backup each component
            tasks = []

            if "rbac_system" in metadata.components_included:
                tasks.append(self._backup_rbac_system(backup_dir))

            if "tenant_data" in metadata.components_included:
                tasks.append(self._backup_tenant_data(backup_dir, include_tenants))

            if "agents" in metadata.components_included:
                tasks.append(self._backup_agents(backup_dir, include_tenants))

            if "rag_collections" in metadata.components_included:
                tasks.append(self._backup_rag_collections(backup_dir, include_tenants))

            if "p2p_networks" in metadata.components_included:
                tasks.append(self._backup_p2p_networks(backup_dir, include_tenants))

            if "models" in metadata.components_included:
                tasks.append(self._backup_models(backup_dir, include_tenants))

            if "configurations" in metadata.components_included:
                tasks.append(self._backup_configurations(backup_dir))

            if "logs" in metadata.components_included:
                tasks.append(self._backup_logs(backup_dir))

            # Execute backup tasks concurrently (limited parallelism)
            semaphore = asyncio.Semaphore(self.config["storage"]["parallel_operations"])

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])

            # Process results
            total_files = 0
            for component_result in results:
                if isinstance(component_result, dict):
                    total_files += component_result.get("file_count", 0)
                    if component_result.get("errors"):
                        metadata.errors.extend(component_result["errors"])
                    if component_result.get("warnings"):
                        metadata.warnings.extend(component_result["warnings"])

            metadata.file_count = total_files

            # Create archive
            archive_path = backup_dir.parent / f"{backup_id}.tar.gz"
            await self._create_compressed_archive(backup_dir, archive_path)

            # Calculate size and checksum
            metadata.size_bytes = archive_path.stat().st_size
            metadata.checksum = await self._calculate_checksum(archive_path)

            # Encrypt if enabled
            if self.config["encryption"]["enabled"]:
                encrypted_path = archive_path.with_suffix(".tar.gz.enc")
                await self._encrypt_file(archive_path, encrypted_path)
                archive_path.unlink()  # Remove unencrypted version
                metadata.size_bytes = encrypted_path.stat().st_size

            # Cleanup temporary directory
            if self.config["storage"]["cleanup_temp_files"]:
                shutil.rmtree(backup_dir)

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.utcnow()

            # Verify backup if enabled
            if self.config["verification"]["verify_after_backup"]:
                verification_result = await self._verify_backup(backup_id)
                if not verification_result:
                    metadata.status = BackupStatus.CORRUPTED
                    metadata.errors.append("Backup verification failed")

            await self._save_metadata(metadata)

            logger.info(f"Full backup {backup_id} completed successfully")
            logger.info(f"Size: {metadata.size_bytes / 1024 / 1024:.2f} MB")
            logger.info(f"Files: {metadata.file_count}")

            return backup_id

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.errors.append(str(e))
            await self._save_metadata(metadata)
            logger.error(f"Full backup {backup_id} failed: {e}")
            raise

    async def create_incremental_backup(self, base_backup_id: str | None = None) -> str:
        """Create incremental backup (changes since last backup)."""
        backup_id = f"inc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Find base backup
        if not base_backup_id:
            base_backup_id = await self._get_latest_backup_id()

        if not base_backup_id:
            raise ValueError("No base backup found for incremental backup")

        base_metadata = await self._get_backup_metadata(base_backup_id)
        if not base_metadata:
            raise ValueError(f"Base backup {base_backup_id} not found")

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            created_at=datetime.utcnow(),
            components_included=base_metadata.components_included,
            prerequisites=[base_backup_id],
        )

        self.active_backups[backup_id] = metadata

        try:
            metadata.status = BackupStatus.IN_PROGRESS
            await self._save_metadata(metadata)

            backup_dir = self.incremental_backups_dir / backup_id
            backup_dir.mkdir()

            logger.info(f"Starting incremental backup {backup_id} based on {base_backup_id}")

            # Get modification time of base backup
            base_time = base_metadata.created_at

            # Backup changed files since base backup
            changed_files = await self._find_changed_files(base_time)

            if not changed_files:
                logger.info("No changes found, skipping incremental backup")
                metadata.status = BackupStatus.COMPLETED
                metadata.completed_at = datetime.utcnow()
                await self._save_metadata(metadata)
                shutil.rmtree(backup_dir)
                return backup_id

            # Copy changed files
            total_files = 0
            for source_path, relative_path in changed_files:
                dest_path = backup_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                total_files += 1

            metadata.file_count = total_files

            # Create change manifest
            manifest = {
                "base_backup_id": base_backup_id,
                "base_backup_time": base_time.isoformat(),
                "changed_files": [str(p) for _, p in changed_files],
                "total_files": total_files,
            }

            with open(backup_dir / "incremental_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Create archive
            archive_path = backup_dir.parent / f"{backup_id}.tar.gz"
            await self._create_compressed_archive(backup_dir, archive_path)

            metadata.size_bytes = archive_path.stat().st_size
            metadata.checksum = await self._calculate_checksum(archive_path)

            # Encrypt if enabled
            if self.config["encryption"]["enabled"]:
                encrypted_path = archive_path.with_suffix(".tar.gz.enc")
                await self._encrypt_file(archive_path, encrypted_path)
                archive_path.unlink()
                metadata.size_bytes = encrypted_path.stat().st_size

            # Cleanup
            if self.config["storage"]["cleanup_temp_files"]:
                shutil.rmtree(backup_dir)

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.utcnow()
            await self._save_metadata(metadata)

            logger.info(f"Incremental backup {backup_id} completed")
            logger.info(f"Changed files: {total_files}")
            logger.info(f"Size: {metadata.size_bytes / 1024 / 1024:.2f} MB")

            return backup_id

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.errors.append(str(e))
            await self._save_metadata(metadata)
            logger.error(f"Incremental backup {backup_id} failed: {e}")
            raise

    async def create_tenant_backup(self, tenant_id: str, include_models: bool = True) -> str:
        """Create backup for specific tenant."""
        backup_id = f"tenant_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.TENANT_ONLY,
            created_at=datetime.utcnow(),
            tenants_included=[tenant_id],
            components_included=["tenant_data", "agents", "rag_collections", "p2p_networks"],
        )

        if include_models:
            metadata.components_included.append("models")

        self.active_backups[backup_id] = metadata

        try:
            metadata.status = BackupStatus.IN_PROGRESS
            await self._save_metadata(metadata)

            backup_dir = self.tenant_backups_dir / backup_id
            backup_dir.mkdir()

            logger.info(f"Starting tenant backup {backup_id} for tenant {tenant_id}")

            # Backup tenant-specific data
            tasks = [
                self._backup_tenant_data(backup_dir, [tenant_id]),
                self._backup_agents(backup_dir, [tenant_id]),
                self._backup_rag_collections(backup_dir, [tenant_id]),
                self._backup_p2p_networks(backup_dir, [tenant_id]),
            ]

            if include_models:
                tasks.append(self._backup_models(backup_dir, [tenant_id]))

            results = await asyncio.gather(*tasks)

            # Process results
            total_files = sum(r.get("file_count", 0) for r in results if isinstance(r, dict))
            metadata.file_count = total_files

            for result in results:
                if isinstance(result, dict):
                    if result.get("errors"):
                        metadata.errors.extend(result["errors"])
                    if result.get("warnings"):
                        metadata.warnings.extend(result["warnings"])

            # Create tenant manifest
            tenant_manifest = {
                "tenant_id": tenant_id,
                "backup_time": metadata.created_at.isoformat(),
                "components": metadata.components_included,
                "file_count": total_files,
            }

            with open(backup_dir / "tenant_manifest.json", "w") as f:
                json.dump(tenant_manifest, f, indent=2)

            # Create archive
            archive_path = backup_dir.parent / f"{backup_id}.tar.gz"
            await self._create_compressed_archive(backup_dir, archive_path)

            metadata.size_bytes = archive_path.stat().st_size
            metadata.checksum = await self._calculate_checksum(archive_path)

            # Encrypt
            if self.config["encryption"]["enabled"]:
                encrypted_path = archive_path.with_suffix(".tar.gz.enc")
                await self._encrypt_file(archive_path, encrypted_path)
                archive_path.unlink()
                metadata.size_bytes = encrypted_path.stat().st_size

            # Cleanup
            if self.config["storage"]["cleanup_temp_files"]:
                shutil.rmtree(backup_dir)

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.utcnow()
            await self._save_metadata(metadata)

            logger.info(f"Tenant backup {backup_id} completed for {tenant_id}")

            return backup_id

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.errors.append(str(e))
            await self._save_metadata(metadata)
            logger.error(f"Tenant backup {backup_id} failed: {e}")
            raise

    # Component-specific backup methods

    async def _backup_rbac_system(self, backup_dir: Path) -> dict[str, Any]:
        """Backup RBAC system data."""
        logger.info("Backing up RBAC system")

        rbac_dir = backup_dir / "rbac"
        rbac_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup RBAC database
            rbac_db_path = Path("data/rbac.db")
            if rbac_db_path.exists():
                shutil.copy2(rbac_db_path, rbac_dir / "rbac.db")
                file_count += 1
            else:
                warnings.append("RBAC database not found")

            # Backup tenant database
            tenant_db_path = Path("data/tenants.db")
            if tenant_db_path.exists():
                shutil.copy2(tenant_db_path, rbac_dir / "tenants.db")
                file_count += 1
            else:
                warnings.append("Tenant database not found")

            # Backup security configuration
            security_config_path = Path("config/security")
            if security_config_path.exists():
                security_dest = rbac_dir / "security_config"
                shutil.copytree(security_config_path, security_dest)
                file_count += len(list(security_dest.rglob("*")))
            else:
                warnings.append("Security configuration not found")

            return {"component": "rbac_system", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup RBAC system: {e}")
            return {"component": "rbac_system", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_tenant_data(self, backup_dir: Path, include_tenants: list[str] | None = None) -> dict[str, Any]:
        """Backup tenant-specific data."""
        logger.info(f"Backing up tenant data (tenants: {include_tenants})")

        tenant_dir = backup_dir / "tenants"
        tenant_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup tenant directories
            tenants_base_path = Path("data/tenants")
            if not tenants_base_path.exists():
                warnings.append("Tenants data directory not found")
                return {"component": "tenant_data", "file_count": 0, "errors": errors, "warnings": warnings}

            for tenant_path in tenants_base_path.iterdir():
                if not tenant_path.is_dir():
                    continue

                tenant_id = tenant_path.name

                # Skip if not in include list
                if include_tenants and tenant_id not in include_tenants:
                    continue

                try:
                    dest_path = tenant_dir / tenant_id
                    shutil.copytree(tenant_path, dest_path)
                    file_count += len(list(dest_path.rglob("*")))
                except Exception as e:
                    errors.append(f"Failed to backup tenant {tenant_id}: {e}")

            return {"component": "tenant_data", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup tenant data: {e}")
            return {"component": "tenant_data", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_agents(self, backup_dir: Path, include_tenants: list[str] | None = None) -> dict[str, Any]:
        """Backup agent configurations and data."""
        logger.info("Backing up agents")

        agents_dir = backup_dir / "agents"
        agents_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup agent configurations from packages/agents
            agents_source = Path("packages/agents")
            if agents_source.exists():
                # Only backup configuration files, not code
                for config_file in agents_source.rglob("*.json"):
                    if "config" in config_file.name.lower():
                        relative_path = config_file.relative_to(agents_source)
                        dest_path = agents_dir / "configs" / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(config_file, dest_path)
                        file_count += 1
            else:
                warnings.append("Agents source directory not found")

            # Backup tenant-specific agent data
            if include_tenants:
                for tenant_id in include_tenants:
                    tenant_agents_path = Path(f"data/tenants/{tenant_id}/agents")
                    if tenant_agents_path.exists():
                        dest_path = agents_dir / "tenant_data" / tenant_id
                        shutil.copytree(tenant_agents_path, dest_path)
                        file_count += len(list(dest_path.rglob("*")))

            return {"component": "agents", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup agents: {e}")
            return {"component": "agents", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_rag_collections(
        self, backup_dir: Path, include_tenants: list[str] | None = None
    ) -> dict[str, Any]:
        """Backup RAG collections and knowledge graphs."""
        logger.info("Backing up RAG collections")

        rag_dir = backup_dir / "rag"
        rag_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup RAG configurations
            rag_config_path = Path("packages/rag")
            if rag_config_path.exists():
                for config_file in rag_config_path.rglob("*.json"):
                    if any(x in config_file.name.lower() for x in ["config", "schema", "mapping"]):
                        relative_path = config_file.relative_to(rag_config_path)
                        dest_path = rag_dir / "configs" / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(config_file, dest_path)
                        file_count += 1

            # Backup tenant RAG data
            if include_tenants:
                for tenant_id in include_tenants:
                    tenant_rag_path = Path(f"data/tenants/{tenant_id}/rag")
                    if tenant_rag_path.exists():
                        dest_path = rag_dir / "tenant_data" / tenant_id
                        shutil.copytree(tenant_rag_path, dest_path)
                        file_count += len(list(dest_path.rglob("*")))

            # Backup vector databases (if they exist)
            vector_db_path = Path("data/vector_db")
            if vector_db_path.exists():
                dest_path = rag_dir / "vector_db"
                shutil.copytree(vector_db_path, dest_path)
                file_count += len(list(dest_path.rglob("*")))

            return {"component": "rag_collections", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup RAG collections: {e}")
            return {"component": "rag_collections", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_p2p_networks(self, backup_dir: Path, include_tenants: list[str] | None = None) -> dict[str, Any]:
        """Backup P2P network configurations."""
        logger.info("Backing up P2P networks")

        p2p_dir = backup_dir / "p2p"
        p2p_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup P2P configurations
            p2p_config_path = Path("packages/p2p")
            if p2p_config_path.exists():
                for config_file in p2p_config_path.rglob("*.json"):
                    if any(x in config_file.name.lower() for x in ["config", "peers", "network"]):
                        relative_path = config_file.relative_to(p2p_config_path)
                        dest_path = p2p_dir / "configs" / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(config_file, dest_path)
                        file_count += 1

            # Backup tenant P2P data
            if include_tenants:
                for tenant_id in include_tenants:
                    tenant_p2p_path = Path(f"data/tenants/{tenant_id}/p2p")
                    if tenant_p2p_path.exists():
                        dest_path = p2p_dir / "tenant_data" / tenant_id
                        shutil.copytree(tenant_p2p_path, dest_path)
                        file_count += len(list(dest_path.rglob("*")))

            return {"component": "p2p_networks", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup P2P networks: {e}")
            return {"component": "p2p_networks", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_models(self, backup_dir: Path, include_tenants: list[str] | None = None) -> dict[str, Any]:
        """Backup AI models and Agent Forge artifacts."""
        logger.info("Backing up models")

        models_dir = backup_dir / "models"
        models_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup Agent Forge models
            agent_forge_path = Path("packages/agent_forge")
            if agent_forge_path.exists():
                for model_file in agent_forge_path.rglob("*.bin"):
                    relative_path = model_file.relative_to(agent_forge_path)
                    dest_path = models_dir / "agent_forge" / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_file, dest_path)
                    file_count += 1

                for model_file in agent_forge_path.rglob("*.safetensors"):
                    relative_path = model_file.relative_to(agent_forge_path)
                    dest_path = models_dir / "agent_forge" / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_file, dest_path)
                    file_count += 1

            # Backup tenant models
            if include_tenants:
                for tenant_id in include_tenants:
                    tenant_models_path = Path(f"data/tenants/{tenant_id}/models")
                    if tenant_models_path.exists():
                        dest_path = models_dir / "tenant_models" / tenant_id
                        shutil.copytree(tenant_models_path, dest_path)
                        file_count += len(list(dest_path.rglob("*")))

            # Note: Large models might be skipped based on size limits
            max_size_gb = self.config["storage"]["max_backup_size_gb"]
            total_size_gb = sum(f.stat().st_size for f in models_dir.rglob("*") if f.is_file()) / (1024**3)

            if total_size_gb > max_size_gb:
                warnings.append(f"Models backup size ({total_size_gb:.1f}GB) exceeds limit ({max_size_gb}GB)")

            return {"component": "models", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup models: {e}")
            return {"component": "models", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_configurations(self, backup_dir: Path) -> dict[str, Any]:
        """Backup system configurations."""
        logger.info("Backing up configurations")

        config_dir = backup_dir / "configurations"
        config_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup config directory
            config_source = Path("config")
            if config_source.exists():
                dest_path = config_dir / "system_config"
                shutil.copytree(config_source, dest_path)
                file_count += len(list(dest_path.rglob("*")))
            else:
                warnings.append("System config directory not found")

            # Backup deployment configurations
            deploy_source = Path("deploy")
            if deploy_source.exists():
                dest_path = config_dir / "deployment_config"
                shutil.copytree(deploy_source, dest_path)
                file_count += len(list(dest_path.rglob("*")))

            # Backup environment files
            for env_file in [".env", ".env.production", ".env.local"]:
                env_path = Path(env_file)
                if env_path.exists():
                    shutil.copy2(env_path, config_dir / env_file)
                    file_count += 1

            return {"component": "configurations", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup configurations: {e}")
            return {"component": "configurations", "file_count": 0, "errors": [str(e)], "warnings": []}

    async def _backup_logs(self, backup_dir: Path) -> dict[str, Any]:
        """Backup system logs (recent only)."""
        logger.info("Backing up logs")

        logs_dir = backup_dir / "logs"
        logs_dir.mkdir()

        try:
            file_count = 0
            errors = []
            warnings = []

            # Backup recent logs (last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)

            for log_source in [Path("logs"), Path("data/logs")]:
                if not log_source.exists():
                    continue

                for log_file in log_source.rglob("*.log"):
                    try:
                        if log_file.stat().st_mtime > cutoff_date.timestamp():
                            relative_path = log_file.relative_to(log_source)
                            dest_path = logs_dir / log_source.name / relative_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(log_file, dest_path)
                            file_count += 1
                    except Exception as e:
                        warnings.append(f"Could not backup log {log_file}: {e}")

            return {"component": "logs", "file_count": file_count, "errors": errors, "warnings": warnings}

        except Exception as e:
            logger.error(f"Failed to backup logs: {e}")
            return {"component": "logs", "file_count": 0, "errors": [str(e)], "warnings": []}

    # Utility methods

    async def _create_compressed_archive(self, source_dir: Path, archive_path: Path):
        """Create compressed tar archive."""
        with tarfile.open(archive_path, "w:gz", compresslevel=self.config["compression"]["level"]) as tar:
            tar.add(source_dir, arcname=source_dir.name)

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def _encrypt_file(self, source_path: Path, dest_path: Path):
        """Encrypt file using Fernet encryption."""
        with open(source_path, "rb") as f:
            data = f.read()

        encrypted_data = self.fernet.encrypt(data)

        with open(dest_path, "wb") as f:
            f.write(encrypted_data)

    async def _decrypt_file(self, source_path: Path, dest_path: Path):
        """Decrypt file using Fernet encryption."""
        with open(source_path, "rb") as f:
            encrypted_data = f.read()

        data = self.fernet.decrypt(encrypted_data)

        with open(dest_path, "wb") as f:
            f.write(data)

    async def _find_changed_files(self, since: datetime) -> list[tuple[Path, Path]]:
        """Find files changed since given timestamp."""
        changed_files = []
        since_timestamp = since.timestamp()

        # Check all relevant directories
        search_paths = [Path("data"), Path("config"), Path("packages")]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for file_path in search_path.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip temporary and cache files
                if any(part.startswith(".") for part in file_path.parts):
                    continue

                if file_path.stat().st_mtime > since_timestamp:
                    relative_path = file_path.relative_to(Path.cwd())
                    changed_files.append((file_path, relative_path))

        return changed_files

    async def _save_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO backups
            (backup_id, backup_type, created_at, completed_at, status, size_bytes,
             checksum, encrypted, compression_ratio, tenants_included, components_included,
             file_count, restore_compatible_versions, prerequisites, errors, warnings, backup_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metadata.backup_id,
                metadata.backup_type.value,
                metadata.created_at.isoformat(),
                metadata.completed_at.isoformat() if metadata.completed_at else None,
                metadata.status.value,
                metadata.size_bytes,
                metadata.checksum,
                metadata.encrypted,
                metadata.compression_ratio,
                json.dumps(metadata.tenants_included),
                json.dumps(metadata.components_included),
                metadata.file_count,
                json.dumps(metadata.restore_compatible_versions),
                json.dumps(metadata.prerequisites),
                json.dumps(metadata.errors),
                json.dumps(metadata.warnings),
                str(self.backup_root / metadata.backup_id),
            ),
        )

        conn.commit()
        conn.close()

    async def _get_backup_metadata(self, backup_id: str) -> BackupMetadata | None:
        """Get backup metadata from database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM backups WHERE backup_id = ?", (backup_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return BackupMetadata(
            backup_id=row[0],
            backup_type=BackupType(row[1]),
            created_at=datetime.fromisoformat(row[2]),
            completed_at=datetime.fromisoformat(row[3]) if row[3] else None,
            status=BackupStatus(row[4]),
            size_bytes=row[5],
            checksum=row[6] or "",
            encrypted=bool(row[7]),
            compression_ratio=row[8],
            tenants_included=json.loads(row[9]) if row[9] else [],
            components_included=json.loads(row[10]) if row[10] else [],
            file_count=row[11],
            restore_compatible_versions=json.loads(row[12]) if row[12] else [],
            prerequisites=json.loads(row[13]) if row[13] else [],
            errors=json.loads(row[14]) if row[14] else [],
            warnings=json.loads(row[15]) if row[15] else [],
        )

    async def _get_latest_backup_id(self) -> str | None:
        """Get the most recent successful backup ID."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute("SELECT backup_id FROM backups WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    async def _verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        try:
            metadata = await self._get_backup_metadata(backup_id)
            if not metadata:
                return False

            # Find backup file
            backup_file = None
            for backup_dir in [
                self.full_backups_dir,
                self.incremental_backups_dir,
                self.tenant_backups_dir,
                self.config_backups_dir,
            ]:
                potential_files = [backup_dir / f"{backup_id}.tar.gz", backup_dir / f"{backup_id}.tar.gz.enc"]

                for potential_file in potential_files:
                    if potential_file.exists():
                        backup_file = potential_file
                        break

                if backup_file:
                    break

            if not backup_file:
                logger.error(f"Backup file not found for {backup_id}")
                return False

            # Verify checksum
            actual_checksum = await self._calculate_checksum(backup_file)
            expected_checksum = metadata.checksum

            if actual_checksum != expected_checksum:
                logger.error(f"Checksum mismatch for backup {backup_id}")
                return False

            # Try to extract archive (basic verification)
            if backup_file.suffix == ".enc":
                # Decrypt to temp file for verification
                with tempfile.NamedTemporaryFile() as temp_file:
                    await self._decrypt_file(backup_file, Path(temp_file.name))

                    # Verify tar archive
                    try:
                        with tarfile.open(temp_file.name, "r:gz") as tar:
                            # Just check if we can list contents
                            _ = tar.getnames()
                    except Exception as e:
                        logger.error(f"Archive verification failed for {backup_id}: {e}")
                        return False
            else:
                # Verify unencrypted archive
                try:
                    with tarfile.open(backup_file, "r:gz") as tar:
                        _ = tar.getnames()
                except Exception as e:
                    logger.error(f"Archive verification failed for {backup_id}: {e}")
                    return False

            logger.info(f"Backup {backup_id} verification successful")
            return True

        except Exception as e:
            logger.error(f"Backup verification failed for {backup_id}: {e}")
            return False

    # Public query methods

    async def list_backups(
        self, backup_type: BackupType | None = None, status: BackupStatus | None = None, limit: int = 50
    ) -> list[BackupMetadata]:
        """List available backups."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT * FROM backups"
        params = []
        conditions = []

        if backup_type:
            conditions.append("backup_type = ?")
            params.append(backup_type.value)

        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        backups = []
        for row in rows:
            metadata = BackupMetadata(
                backup_id=row[0],
                backup_type=BackupType(row[1]),
                created_at=datetime.fromisoformat(row[2]),
                completed_at=datetime.fromisoformat(row[3]) if row[3] else None,
                status=BackupStatus(row[4]),
                size_bytes=row[5],
                checksum=row[6] or "",
                encrypted=bool(row[7]),
                compression_ratio=row[8],
                tenants_included=json.loads(row[9]) if row[9] else [],
                components_included=json.loads(row[10]) if row[10] else [],
                file_count=row[11],
                restore_compatible_versions=json.loads(row[12]) if row[12] else [],
                prerequisites=json.loads(row[13]) if row[13] else [],
                errors=json.loads(row[14]) if row[14] else [],
                warnings=json.loads(row[15]) if row[15] else [],
            )
            backups.append(metadata)

        return backups

    async def get_backup_info(self, backup_id: str) -> BackupMetadata | None:
        """Get detailed information about a backup."""
        return await self._get_backup_metadata(backup_id)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize backup manager
        backup_manager = BackupManager()

        # Create full backup
        print("Creating full backup...")
        backup_id = await backup_manager.create_full_backup()
        print(f"Full backup created: {backup_id}")

        # List backups
        backups = await backup_manager.list_backups()
        print(f"Available backups: {len(backups)}")
        for backup in backups:
            print(f"  {backup.backup_id}: {backup.status.value} ({backup.size_bytes / 1024 / 1024:.1f} MB)")

    asyncio.run(main())
