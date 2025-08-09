#!/usr/bin/env python3
"""Schema migration system for CODEX Integration databases.

Implements a version-based migration system that tracks schema changes
and can upgrade/rollback database schemas safely.
"""

import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles schema migrations for CODEX databases."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.connection = None
        self.current_version = 0

    def connect(self) -> sqlite3.Connection:
        """Connect to database with proper settings."""
        self.connection = sqlite3.connect(
            str(self.db_path),
            isolation_level=None  # autocommit mode
        )

        # Enable WAL mode and optimizations
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=10000")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA mmap_size=268435456")

        return self.connection

    def get_current_version(self) -> int:
        """Get current schema version from database."""
        try:
            cursor = self.connection.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            result = cursor.fetchone()[0]
            self.current_version = result if result is not None else 0
        except sqlite3.OperationalError:
            # Table doesn't exist, version is 0
            self.current_version = 0

        return self.current_version

    def create_schema_version_table(self):
        """Create schema_version table if it doesn't exist."""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)

    def record_migration(self, version: int, description: str):
        """Record a successful migration."""
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
            (version, description)
        )

    def rollback_migration(self, version: int):
        """Remove a migration record."""
        self.connection.execute(
            "DELETE FROM schema_version WHERE version = ?",
            (version,)
        )

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None


class EvolutionMetricsMigrator(DatabaseMigrator):
    """Migration handler for evolution_metrics.db"""

    def __init__(self):
        super().__init__("data/evolution_metrics.db")

    def migrate_to_v1(self):
        """Initial schema creation."""
        logger.info("Creating evolution metrics schema v1...")

        # Evolution rounds table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS evolution_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                population_size INTEGER,
                mutation_rate REAL,
                selection_pressure REAL,
                status TEXT DEFAULT 'running',
                metadata TEXT,
                UNIQUE(round_number, generation)
            )
        """)

        # Fitness metrics table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS fitness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                fitness_score REAL NOT NULL,
                performance_metrics TEXT,
                resource_usage TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (round_id) REFERENCES evolution_rounds (id)
            )
        """)

        # Resource metrics table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                cpu_usage REAL,
                memory_usage_mb REAL,
                network_io_kb REAL,
                disk_io_kb REAL,
                gpu_usage REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (round_id) REFERENCES evolution_rounds (id)
            )
        """)

        # Selection outcomes table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS selection_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                parent_agent_id TEXT NOT NULL,
                child_agent_id TEXT,
                selection_method TEXT,
                crossover_points TEXT,
                mutation_applied BOOLEAN DEFAULT FALSE,
                survival_reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (round_id) REFERENCES evolution_rounds (id)
            )
        """)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_evolution_rounds_number ON evolution_rounds(round_number)",
            "CREATE INDEX IF NOT EXISTS idx_fitness_agent ON fitness_metrics(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_fitness_score ON fitness_metrics(fitness_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_resource_timestamp ON resource_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_selection_parent ON selection_outcomes(parent_agent_id)"
        ]

        for index_sql in indexes:
            self.connection.execute(index_sql)

        self.record_migration(1, "Initial evolution metrics schema")
        logger.info("Evolution metrics schema v1 created successfully")

    def run_migrations(self):
        """Run all necessary migrations."""
        self.connect()
        self.create_schema_version_table()
        current_version = self.get_current_version()

        if current_version < 1:
            self.migrate_to_v1()

        self.close()


class DigitalTwinMigrator(DatabaseMigrator):
    """Migration handler for digital_twin.db"""

    def __init__(self):
        super().__init__("data/digital_twin.db")

    def migrate_to_v1(self):
        """Initial schema creation."""
        logger.info("Creating digital twin schema v1...")

        # Learning profiles table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS learning_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT UNIQUE NOT NULL,
                user_id_hash TEXT NOT NULL,
                learning_style TEXT,
                preferred_difficulty TEXT DEFAULT 'medium',
                knowledge_domains TEXT,
                learning_goals TEXT,
                privacy_settings TEXT,
                encrypted_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ttl_expires_at TIMESTAMP
            )
        """)

        # Learning sessions table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                profile_id TEXT NOT NULL,
                session_type TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                duration_minutes REAL,
                topics_covered TEXT,
                performance_metrics TEXT,
                engagement_score REAL,
                completion_status TEXT DEFAULT 'in_progress',
                FOREIGN KEY (profile_id) REFERENCES learning_profiles (profile_id)
            )
        """)

        # Knowledge states table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                knowledge_domain TEXT NOT NULL,
                topic TEXT NOT NULL,
                mastery_level REAL DEFAULT 0.0,
                confidence_score REAL DEFAULT 0.0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                learning_trajectory TEXT,
                prerequisites_met BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (profile_id) REFERENCES learning_profiles (profile_id),
                UNIQUE(profile_id, knowledge_domain, topic)
            )
        """)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_profiles_user_hash ON learning_profiles(user_id_hash)",
            "CREATE INDEX IF NOT EXISTS idx_profiles_updated ON learning_profiles(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_profile ON learning_sessions(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start ON learning_sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_profile ON knowledge_states(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_states(knowledge_domain)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_mastery ON knowledge_states(mastery_level DESC)"
        ]

        for index_sql in indexes:
            self.connection.execute(index_sql)

        self.record_migration(1, "Initial digital twin schema")
        logger.info("Digital twin schema v1 created successfully")

    def run_migrations(self):
        """Run all necessary migrations."""
        self.connect()
        self.create_schema_version_table()
        current_version = self.get_current_version()

        if current_version < 1:
            self.migrate_to_v1()

        self.close()


class RAGIndexMigrator(DatabaseMigrator):
    """Migration handler for rag_index.db"""

    def __init__(self):
        super().__init__("data/rag_index.db")

    def migrate_to_v1(self):
        """Initial schema creation."""
        logger.info("Creating RAG index schema v1...")

        # Documents table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                document_type TEXT DEFAULT 'text',
                source_path TEXT,
                source_url TEXT,
                file_hash TEXT,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Chunks table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_size INTEGER,
                overlap_size INTEGER DEFAULT 50,
                embedding_vector BLOB,
                embedding_model TEXT DEFAULT 'paraphrase-MiniLM-L3-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (document_id),
                UNIQUE(document_id, chunk_index)
            )
        """)

        # Embeddings metadata table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                vector_dimension INTEGER DEFAULT 384,
                faiss_index_id INTEGER,
                bm25_doc_id INTEGER,
                similarity_scores TEXT,
                last_queried TIMESTAMP,
                query_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
                UNIQUE(chunk_id)
            )
        """)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON embeddings_metadata(faiss_index_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_queries ON embeddings_metadata(query_count DESC)"
        ]

        for index_sql in indexes:
            self.connection.execute(index_sql)

        self.record_migration(1, "Initial RAG index schema")
        logger.info("RAG index schema v1 created successfully")

    def run_migrations(self):
        """Run all necessary migrations."""
        self.connect()
        self.create_schema_version_table()
        current_version = self.get_current_version()

        if current_version < 1:
            self.migrate_to_v1()

        self.close()


def main():
    """Run all database migrations."""
    logging.basicConfig(level=logging.INFO)

    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    print("Starting CODEX database migrations...")

    # Run migrations for all databases
    migrators = [
        EvolutionMetricsMigrator(),
        DigitalTwinMigrator(),
        RAGIndexMigrator()
    ]

    success_count = 0

    for migrator in migrators:
        try:
            migrator.run_migrations()
            success_count += 1
        except Exception as e:
            logger.error(f"Migration failed for {migrator.__class__.__name__}: {e}")

    print(f"\nMigration complete: {success_count}/{len(migrators)} databases migrated successfully")

    # Verify databases exist
    data_dir = Path("data")
    expected_dbs = ["evolution_metrics.db", "digital_twin.db", "rag_index.db"]

    print("\nVerifying databases:")
    for db_name in expected_dbs:
        db_path = data_dir / db_name
        if db_path.exists():
            size = db_path.stat().st_size
            print(f"  ✅ {db_name}: {size:,} bytes")
        else:
            print(f"  ❌ {db_name}: Missing")

    return success_count == len(migrators)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
