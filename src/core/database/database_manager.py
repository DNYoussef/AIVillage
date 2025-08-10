"""Database manager for AIVillage CODEX integration.

This module provides centralized database management for all CODEX components:
- Evolution metrics SQLite database
- Digital Twin SQLite database
- RAG index SQLite database
- Redis connections with fallbacks
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    aioredis = None
    REDIS_AVAILABLE = False

try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    Fernet = None
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str
    wal_mode: bool = True
    timeout: int = 30
    max_connections: int = 10
    backup_enabled: bool = True
    encryption_enabled: bool = False
    encryption_key: str | None = None


@dataclass
class RedisConfig:
    """Redis configuration."""

    url: str
    db: int = 0
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


class DatabaseManager:
    """Centralized database manager for all CODEX components."""

    # Database schemas
    EVOLUTION_METRICS_SCHEMA = """
    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT
    );

    -- Evolution rounds tracking
    CREATE TABLE IF NOT EXISTS evolution_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time REAL NOT NULL,
        end_time REAL,
        status TEXT NOT NULL DEFAULT 'running',
        agent_count INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 0.0,
        metadata TEXT, -- JSON blob
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Fitness metrics for each agent evolution
    CREATE TABLE IF NOT EXISTS fitness_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id INTEGER NOT NULL,
        agent_id TEXT NOT NULL,
        evolution_id TEXT NOT NULL,
        fitness_score REAL NOT NULL,
        improvement_delta REAL DEFAULT 0.0,
        timestamp REAL NOT NULL,
        metadata TEXT, -- JSON blob
        FOREIGN KEY(round_id) REFERENCES evolution_rounds(id)
    );

    -- Resource usage metrics
    CREATE TABLE IF NOT EXISTS resource_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id INTEGER NOT NULL,
        cpu_usage REAL NOT NULL,
        memory_mb REAL NOT NULL,
        energy_estimate REAL DEFAULT 0.0,
        duration_seconds REAL DEFAULT 0.0,
        timestamp REAL NOT NULL,
        FOREIGN KEY(round_id) REFERENCES evolution_rounds(id)
    );

    -- Selection outcomes for mutations
    CREATE TABLE IF NOT EXISTS selection_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id INTEGER NOT NULL,
        mutation_id TEXT NOT NULL,
        selected INTEGER NOT NULL, -- 0 or 1
        reason TEXT,
        confidence_score REAL DEFAULT 0.0,
        timestamp REAL NOT NULL,
        FOREIGN KEY(round_id) REFERENCES evolution_rounds(id)
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_fitness_metrics_round_agent ON fitness_metrics(round_id, agent_id);
    CREATE INDEX IF NOT EXISTS idx_fitness_metrics_timestamp ON fitness_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_resource_metrics_round ON resource_metrics(round_id);
    CREATE INDEX IF NOT EXISTS idx_selection_outcomes_round ON selection_outcomes(round_id);
    """

    DIGITAL_TWIN_SCHEMA = """
    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT
    );

    -- Learning profiles for students
    CREATE TABLE IF NOT EXISTS learning_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        grade_level INTEGER NOT NULL,
        language TEXT NOT NULL DEFAULT 'en',
        region TEXT NOT NULL DEFAULT 'US',
        learning_style TEXT NOT NULL DEFAULT 'visual',
        strengths TEXT, -- JSON array
        challenges TEXT, -- JSON array
        interests TEXT, -- JSON array
        attention_span_minutes INTEGER DEFAULT 15,
        preferred_session_times TEXT, -- JSON array
        parent_constraints TEXT, -- JSON object (encrypted)
        accessibility_needs TEXT, -- JSON array
        motivation_triggers TEXT, -- JSON array
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Learning sessions
    CREATE TABLE IF NOT EXISTS learning_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        student_id TEXT NOT NULL,
        tutor_model_id TEXT,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        duration_minutes INTEGER DEFAULT 0,
        concepts_covered TEXT, -- JSON array
        questions_asked INTEGER DEFAULT 0,
        questions_correct INTEGER DEFAULT 0,
        engagement_score REAL DEFAULT 0.0,
        difficulty_level REAL DEFAULT 0.5,
        adaptations_made TEXT, -- JSON array
        parent_feedback TEXT,
        student_mood TEXT DEFAULT 'neutral',
        session_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES learning_profiles(student_id)
    );

    -- Knowledge state tracking
    CREATE TABLE IF NOT EXISTS knowledge_states (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        subject TEXT NOT NULL,
        concept TEXT NOT NULL,
        mastery_level REAL NOT NULL DEFAULT 0.0,
        confidence_score REAL NOT NULL DEFAULT 0.0,
        last_practiced TIMESTAMP,
        practice_count INTEGER DEFAULT 0,
        mistake_patterns TEXT, -- JSON array
        prerequisite_gaps TEXT, -- JSON array
        next_recommended TEXT, -- JSON array
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES learning_profiles(student_id),
        UNIQUE(student_id, subject, concept)
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_learning_profiles_student_id ON learning_profiles(student_id);
    CREATE INDEX IF NOT EXISTS idx_learning_sessions_student_id ON learning_sessions(student_id);
    CREATE INDEX IF NOT EXISTS idx_learning_sessions_start_time ON learning_sessions(start_time);
    CREATE INDEX IF NOT EXISTS idx_knowledge_states_student ON knowledge_states(student_id, subject);
    CREATE INDEX IF NOT EXISTS idx_knowledge_states_mastery ON knowledge_states(mastery_level);
    """

    RAG_INDEX_SCHEMA = """
    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT
    );

    -- Document metadata
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT UNIQUE NOT NULL,
        title TEXT,
        content_hash TEXT NOT NULL,
        source_type TEXT NOT NULL DEFAULT 'text',
        file_path TEXT,
        url TEXT,
        language TEXT DEFAULT 'en',
        word_count INTEGER DEFAULT 0,
        chunk_count INTEGER DEFAULT 0,
        embedding_model TEXT,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT -- JSON blob
    );

    -- Document chunks for retrieval
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id TEXT UNIQUE NOT NULL,
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        token_count INTEGER DEFAULT 0,
        overlap_start INTEGER DEFAULT 0,
        overlap_end INTEGER DEFAULT 0,
        embedding_vector BLOB, -- Serialized numpy array
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(document_id) REFERENCES documents(document_id)
    );

    -- Embeddings metadata for FAISS index
    CREATE TABLE IF NOT EXISTS embeddings_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faiss_index_id INTEGER NOT NULL, -- FAISS internal ID
        chunk_id TEXT NOT NULL,
        embedding_model TEXT NOT NULL,
        vector_dimension INTEGER NOT NULL,
        norm REAL, -- Vector L2 norm
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    -- Query cache for performance
    CREATE TABLE IF NOT EXISTS query_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_hash TEXT UNIQUE NOT NULL,
        query_text TEXT NOT NULL,
        result_chunk_ids TEXT NOT NULL, -- JSON array of chunk IDs
        similarity_scores TEXT NOT NULL, -- JSON array of scores
        cache_hit_count INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
    CREATE INDEX IF NOT EXISTS idx_embeddings_faiss_id ON embeddings_metadata(faiss_index_id);
    CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings_metadata(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON query_cache(query_hash);
    CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at);
    """

    def __init__(self, config_manager=None):
        """Initialize database manager."""
        self.config_manager = config_manager
        self.connections: dict[str, sqlite3.Connection] = {}
        self.redis_pools: dict[str, Any] = {}
        self.encryption_keys: dict[str, Fernet] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all databases and connections."""
        if self._initialized:
            return

        logger.info("Initializing database manager...")

        # Create data directories
        await self._create_data_directories()

        # Initialize SQLite databases
        await self._initialize_sqlite_databases()

        # Initialize Redis connections
        await self._initialize_redis_connections()

        # Set up encryption
        await self._setup_encryption()

        self._initialized = True
        logger.info("Database manager initialized successfully")

    async def _create_data_directories(self) -> None:
        """Create necessary data directories."""
        if self.config_manager:
            dirs = [
                self.config_manager.get("AIVILLAGE_LOG_DIR", "./logs"),
                self.config_manager.get("RAG_FAISS_INDEX_PATH", "./data/faiss_index"),
                self.config_manager.get("DIGITAL_TWIN_VAULT_PATH", "./data/vault"),
                self.config_manager.get("BACKUP_STORAGE_PATH", "./backups"),
            ]
        else:
            dirs = ["./data", "./logs", "./backups"]

        for dir_path in dirs:
            if dir_path and dir_path != ":memory:":
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")

    async def _initialize_sqlite_databases(self) -> None:
        """Initialize all SQLite databases with schemas."""
        databases = {
            "evolution_metrics": {
                "path": (
                    self.config_manager.get(
                        "AIVILLAGE_DB_PATH", "./data/evolution_metrics.db"
                    )
                    if self.config_manager
                    else "./data/evolution_metrics.db"
                ),
                "schema": self.EVOLUTION_METRICS_SCHEMA,
                "description": "Evolution metrics database v1.0",
            },
            "digital_twin": {
                "path": (
                    self.config_manager.get(
                        "DIGITAL_TWIN_DB_PATH", "./data/digital_twin.db"
                    )
                    if self.config_manager
                    else "./data/digital_twin.db"
                ),
                "schema": self.DIGITAL_TWIN_SCHEMA,
                "description": "Digital Twin database v1.0",
            },
            "rag_index": {
                "path": (
                    self.config_manager.get("RAG_INDEX_DB_PATH", "./data/rag_index.db")
                    if self.config_manager
                    else "./data/rag_index.db"
                ),
                "schema": self.RAG_INDEX_SCHEMA,
                "description": "RAG index database v1.0",
            },
        }

        for db_name, db_config in databases.items():
            await self._create_sqlite_database(db_name, db_config)

    async def _create_sqlite_database(self, name: str, config: dict[str, str]) -> None:
        """Create and initialize a SQLite database."""
        db_path = config["path"]

        if db_path == ":memory:":
            logger.info(f"Creating in-memory database: {name}")
        else:
            # Create parent directory
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating SQLite database: {name} at {db_path}")

        # Create connection
        conn = sqlite3.connect(db_path, timeout=30)

        try:
            # Enable WAL mode for concurrent access
            if db_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=memory")
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            # Execute schema
            conn.executescript(config["schema"])

            # Insert schema version
            schema_version = 1
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
                (schema_version, config["description"]),
            )

            conn.commit()

            # Store connection
            self.connections[name] = conn

            logger.info(
                f"Database {name} initialized successfully with schema version {schema_version}"
            )

        except Exception as e:
            conn.close()
            logger.error(f"Failed to initialize database {name}: {e}")
            raise

    async def _initialize_redis_connections(self) -> None:
        """Initialize Redis connections with fallbacks."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using SQLite fallbacks")
            return

        if not self.config_manager:
            return

        redis_configs = {
            "evolution_metrics": {
                "url": self.config_manager.get(
                    "AIVILLAGE_REDIS_URL", "redis://localhost:6379/0"
                ),
                "db": 0,
                "description": "Evolution metrics real-time data",
            },
            "rag_cache": {
                "url": self.config_manager.get(
                    "RAG_REDIS_URL", "redis://localhost:6379/1"
                ),
                "db": 1,
                "description": "RAG pipeline caching",
            },
            "p2p_discovery": {
                "url": "redis://localhost:6379/2",
                "db": 2,
                "description": "P2P peer discovery cache",
            },
        }

        for pool_name, config in redis_configs.items():
            try:
                # Parse URL to get database number
                redis_url = config["url"]
                if redis_url.endswith(f"/{config['db']}"):
                    # URL already has database number
                    pass
                else:
                    # Append database number
                    redis_url = f"{redis_url.rstrip('/')}/{config['db']}"

                # Create connection pool
                pool = aioredis.ConnectionPool.from_url(
                    redis_url,
                    max_connections=10,
                    retry_on_timeout=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )

                # Test connection
                redis_client = aioredis.Redis(connection_pool=pool)
                await redis_client.ping()
                await redis_client.close()

                self.redis_pools[pool_name] = pool
                logger.info(
                    f"Redis pool {pool_name} initialized: {config['description']}"
                )

            except Exception as e:
                logger.warning(f"Failed to initialize Redis pool {pool_name}: {e}")
                # Continue without Redis - will use SQLite fallbacks

    async def _setup_encryption(self) -> None:
        """Set up encryption for sensitive data."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available, encryption disabled")
            return

        if not self.config_manager:
            return

        # Digital Twin encryption
        encryption_key = self.config_manager.get("DIGITAL_TWIN_ENCRYPTION_KEY")
        if (
            encryption_key
            and encryption_key != "REPLACE_WITH_BASE64_ENCODED_32_BYTE_KEY"
        ):
            try:
                self.encryption_keys["digital_twin"] = Fernet(encryption_key.encode())
                logger.info("Digital Twin encryption enabled")
            except Exception as e:
                logger.error(f"Failed to setup Digital Twin encryption: {e}")
        else:
            logger.warning("Digital Twin encryption key not configured")

    @contextmanager
    def get_connection(self, database: str):
        """Get SQLite database connection (context manager)."""
        if database not in self.connections:
            raise ValueError(f"Database {database} not initialized")

        conn = self.connections[database]
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    @asynccontextmanager
    async def get_redis_connection(self, pool_name: str):
        """Get Redis connection from pool (async context manager)."""
        if pool_name not in self.redis_pools:
            # Fallback to None - caller should handle
            yield None
            return

        pool = self.redis_pools[pool_name]
        redis_client = aioredis.Redis(connection_pool=pool)
        try:
            yield redis_client
        finally:
            await redis_client.close()

    def encrypt_data(self, data: str, encryption_type: str = "digital_twin") -> str:
        """Encrypt sensitive data."""
        if encryption_type not in self.encryption_keys:
            logger.warning(f"Encryption key {encryption_type} not available")
            return data

        fernet = self.encryption_keys[encryption_type]
        return fernet.encrypt(data.encode()).decode()

    def decrypt_data(
        self, encrypted_data: str, encryption_type: str = "digital_twin"
    ) -> str:
        """Decrypt sensitive data."""
        if encryption_type not in self.encryption_keys:
            logger.warning(f"Encryption key {encryption_type} not available")
            return encrypted_data

        try:
            fernet = self.encryption_keys[encryption_type]
            return fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return encrypted_data

    async def check_database_integrity(self) -> dict[str, bool]:
        """Check integrity of all databases."""
        results = {}

        # Check SQLite databases
        for db_name, conn in self.connections.items():
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                results[f"sqlite_{db_name}"] = result[0] == "ok" if result else False
                logger.info(
                    f"Database {db_name} integrity: {results[f'sqlite_{db_name}']}"
                )
            except Exception as e:
                logger.error(f"Integrity check failed for {db_name}: {e}")
                results[f"sqlite_{db_name}"] = False

        # Check Redis connections
        for pool_name in self.redis_pools:
            try:
                async with self.get_redis_connection(pool_name) as redis_client:
                    if redis_client:
                        await redis_client.ping()
                        results[f"redis_{pool_name}"] = True
                    else:
                        results[f"redis_{pool_name}"] = False
            except Exception as e:
                logger.error(f"Redis connection check failed for {pool_name}: {e}")
                results[f"redis_{pool_name}"] = False

        return results

    async def create_backup(self, database: str, backup_path: str | None = None) -> str:
        """Create database backup."""
        if database not in self.connections:
            raise ValueError(f"Database {database} not found")

        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("./backups")
            backup_dir.mkdir(exist_ok=True)
            backup_path = str(backup_dir / f"{database}_backup_{timestamp}.db")

        source_conn = self.connections[database]
        backup_conn = sqlite3.connect(backup_path)

        try:
            # Use SQLite backup API
            source_conn.backup(backup_conn)
            backup_conn.close()

            logger.info(f"Database {database} backed up to {backup_path}")
            return backup_path

        except Exception as e:
            backup_conn.close()
            if Path(backup_path).exists():
                Path(backup_path).unlink()
            logger.error(f"Backup failed for {database}: {e}")
            raise

    async def optimize_databases(self) -> None:
        """Optimize all databases for performance."""
        for db_name, conn in self.connections.items():
            try:
                logger.info(f"Optimizing database {db_name}")

                # Analyze tables for query optimization
                conn.execute("ANALYZE")

                # Vacuum if not in WAL mode or if database is small
                cursor = conn.cursor()
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]

                if page_count < 10000:  # Small database, safe to vacuum
                    conn.execute("VACUUM")
                    logger.info(f"Vacuumed database {db_name}")

                conn.commit()

            except Exception as e:
                logger.error(f"Optimization failed for {db_name}: {e}")

    async def get_database_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all databases."""
        stats = {}

        for db_name, conn in self.connections.items():
            try:
                cursor = conn.cursor()

                # Get table information
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                table_stats = {}
                total_rows = 0

                for table in tables:
                    if table != "sqlite_sequence":
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_stats[table] = count
                        total_rows += count

                # Get database size
                if (
                    self.connections[db_name]
                    .execute("PRAGMA database_list")
                    .fetchone()[2]
                    != ":memory:"
                ):
                    cursor.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]
                    cursor.execute("PRAGMA page_size")
                    page_size = cursor.fetchone()[0]
                    size_bytes = page_count * page_size
                else:
                    size_bytes = 0

                stats[db_name] = {
                    "tables": table_stats,
                    "total_rows": total_rows,
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024 * 1024),
                }

            except Exception as e:
                logger.error(f"Failed to get stats for {db_name}: {e}")
                stats[db_name] = {"error": str(e)}

        return stats

    async def close(self) -> None:
        """Close all database connections."""
        logger.info("Closing database connections...")

        # Close SQLite connections
        for db_name, conn in self.connections.items():
            try:
                conn.close()
                logger.debug(f"Closed SQLite connection: {db_name}")
            except Exception as e:
                logger.error(f"Error closing SQLite connection {db_name}: {e}")

        self.connections.clear()

        # Close Redis connection pools
        for pool_name, pool in self.redis_pools.items():
            try:
                await pool.disconnect()
                logger.debug(f"Closed Redis pool: {pool_name}")
            except Exception as e:
                logger.error(f"Error closing Redis pool {pool_name}: {e}")

        self.redis_pools.clear()

        self._initialized = False
        logger.info("Database manager closed")


# Global database manager instance
_database_manager: DatabaseManager | None = None


async def get_database_manager(config_manager=None) -> DatabaseManager:
    """Get global database manager instance."""
    global _database_manager

    if _database_manager is None:
        _database_manager = DatabaseManager(config_manager)
        await _database_manager.initialize()

    return _database_manager


async def initialize_databases(config_manager=None) -> DatabaseManager:
    """Initialize database manager with configuration."""
    db_manager = DatabaseManager(config_manager)
    await db_manager.initialize()
    return db_manager


if __name__ == "__main__":
    import asyncio

    async def main():
        """Test database manager."""
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Check integrity
        integrity = await db_manager.check_database_integrity()
        print("Database integrity:", integrity)

        # Get stats
        stats = await db_manager.get_database_stats()
        print("Database stats:", stats)

        await db_manager.close()

    asyncio.run(main())
