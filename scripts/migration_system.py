#!/usr/bin/env python3
"""Schema migration system for CODEX Integration databases.

Implements a version-based migration system that tracks schema changes
and can upgrade/rollback database schemas safely.
"""

import base64
import logging
import os
from pathlib import Path
import sqlite3
import sys

# Ensure repository root and src are on path for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from core.security.digital_twin_encryption import (
    DigitalTwinEncryption,
    DigitalTwinEncryptionError,
)

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
            isolation_level=None,  # autocommit mode
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
            cursor = self.connection.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()[0]
            self.current_version = result if result is not None else 0
        except sqlite3.OperationalError:
            # Table doesn't exist, version is 0
            self.current_version = 0

        return self.current_version

    def create_schema_version_table(self):
        """Create schema_version table if it doesn't exist."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """
        )

    def record_migration(self, version: int, description: str):
        """Record a successful migration."""
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
            (version, description),
        )

    def rollback_migration(self, version: int):
        """Remove a migration record."""
        self.connection.execute(
            "DELETE FROM schema_version WHERE version = ?", (version,)
        )

    def rename_column_if_exists(self, table: str, old: str, new: str) -> None:
        """Rename a column if the old name exists and new one is missing."""
        cursor = self.connection.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if old in columns and new not in columns:
            self.connection.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")

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
        """Create schema and seed data for evolution metrics."""
        logger.info("Creating evolution metrics schema v1...")

        # Evolution rounds table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                population_size INTEGER,
                avg_fitness REAL,
                best_fitness REAL,
                worst_fitness REAL,
                generation_time_ms REAL,
                metadata TEXT
            )
            """
        )

        # Fitness metrics table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fitness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER REFERENCES evolution_rounds(id),
                agent_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Resource metrics table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER REFERENCES evolution_rounds(id),
                cpu_usage REAL,
                memory_usage_mb REAL,
                gpu_usage REAL,
                network_io_kb REAL,
                disk_io_kb REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Selection outcomes table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS selection_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER REFERENCES evolution_rounds(id),
                parent_agent_id TEXT,
                offspring_agent_id TEXT,
                mutation_type TEXT,
                mutation_strength REAL,
                fitness_improvement REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # KPI tracking table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS kpi_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kpi_name TEXT NOT NULL,
                kpi_value REAL,
                target_value REAL,
                achievement_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_round_number ON evolution_rounds(round_number)",
            "CREATE INDEX IF NOT EXISTS idx_fitness_round ON fitness_metrics(round_id)",
            "CREATE INDEX IF NOT EXISTS idx_fitness_metric ON fitness_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_resource_round ON resource_metrics(round_id)",
            "CREATE INDEX IF NOT EXISTS idx_selection_round ON selection_outcomes(round_id)",
            "CREATE INDEX IF NOT EXISTS idx_kpi_name ON kpi_tracking(kpi_name)",
        ]
        for sql in indexes:
            self.connection.execute(sql)

        # Seed data
        self.connection.execute(
            "INSERT INTO evolution_rounds (round_number, population_size, avg_fitness) VALUES (1, 10, 0.5)"
        )
        round_id = self.connection.execute(
            "SELECT id FROM evolution_rounds WHERE round_number=1"
        ).fetchone()[0]
        self.connection.execute(
            "INSERT INTO fitness_metrics (round_id, agent_id, metric_name, metric_value) VALUES (?, ?, ?, ?)",
            (round_id, "agent_1", "accuracy", 0.8),
        )
        self.connection.execute(
            "INSERT INTO resource_metrics (round_id, cpu_usage, memory_usage_mb) VALUES (?, ?, ?)",
            (round_id, 50.0, 128.0),
        )
        self.connection.execute(
            "INSERT INTO selection_outcomes (round_id, parent_agent_id, offspring_agent_id, mutation_type) VALUES (?, ?, ?, ?)",
            (round_id, "agent_parent", "agent_child", "mutation"),
        )
        self.connection.execute(
            "INSERT INTO kpi_tracking (kpi_name, kpi_value, target_value, achievement_rate) VALUES ('throughput', 10, 20, 0.5)"
        )

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
        """Create schema and seed data for digital twin."""
        logger.info("Creating digital twin schema v1...")

        # Learning profiles table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                encrypted_name BLOB,
                age_group TEXT,
                learning_style TEXT,
                knowledge_level TEXT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                coppa_compliant BOOLEAN DEFAULT 1,
                ferpa_compliant BOOLEAN DEFAULT 1,
                gdpr_compliant BOOLEAN DEFAULT 1
            )
            """
        )

        # Learning sessions table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER REFERENCES learning_profiles(id),
                session_id TEXT UNIQUE NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds INTEGER,
                topics_covered TEXT,
                performance_metrics TEXT,
                engagement_score REAL,
                completion_rate REAL
            )
            """
        )

        # Knowledge states table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER REFERENCES learning_profiles(id),
                topic TEXT NOT NULL,
                mastery_level REAL,
                confidence_score REAL,
                last_assessed TIMESTAMP,
                assessment_count INTEGER DEFAULT 0,
                encrypted_details BLOB
            )
            """
        )

        # Privacy settings table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS privacy_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER REFERENCES learning_profiles(id),
                data_collection_consent BOOLEAN DEFAULT 0,
                analytics_consent BOOLEAN DEFAULT 0,
                personalization_consent BOOLEAN DEFAULT 1,
                data_retention_days INTEGER DEFAULT 365,
                deletion_requested BOOLEAN DEFAULT 0,
                deletion_date TIMESTAMP
            )
            """
        )

        # Indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_profiles_user ON learning_profiles(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_profile ON learning_sessions(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_profile ON knowledge_states(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_privacy_profile ON privacy_settings(profile_id)",
        ]
        for sql in indexes:
            self.connection.execute(sql)

        # Ensure encryption key and encrypt seed data
        try:
            encryption = DigitalTwinEncryption()
        except DigitalTwinEncryptionError:
            raw_key = os.urandom(32)
            os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = base64.b64encode(
                raw_key
            ).decode()
            encryption = DigitalTwinEncryption()

        enc_name = encryption.encrypt_sensitive_field("Seed Learner", "name")
        self.connection.execute(
            "INSERT INTO learning_profiles (user_id, encrypted_name, age_group, learning_style, knowledge_level, preferences) VALUES (?, ?, ?, ?, ?, ?)",
            ("user_1", enc_name, "adult", "visual", "beginner", "none"),
        )
        profile_id = self.connection.execute(
            "SELECT id FROM learning_profiles WHERE user_id=?",
            ("user_1",),
        ).fetchone()[0]

        self.connection.execute(
            "INSERT INTO learning_sessions (profile_id, session_id, duration_seconds) VALUES (?, ?, ?)",
            (profile_id, "session_1", 0),
        )
        enc_details = encryption.encrypt_sensitive_field("details", "details")
        self.connection.execute(
            "INSERT INTO knowledge_states (profile_id, topic, mastery_level, confidence_score, encrypted_details) VALUES (?, ?, ?, ?, ?)",
            (profile_id, "math", 0.1, 0.9, enc_details),
        )
        self.connection.execute(
            "INSERT INTO privacy_settings (profile_id) VALUES (?)",
            (profile_id,),
        )

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
        """Create schema and seed data for RAG index."""
        logger.info("Creating RAG index schema v1...")

        # Documents table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                title TEXT,
                source TEXT,
                content_hash TEXT,
                chunk_count INTEGER,
                total_tokens INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                indexed_at TIMESTAMP,
                metadata TEXT
            )
            """
        )

        # Chunks table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT REFERENCES documents(doc_id),
                chunk_id TEXT UNIQUE NOT NULL,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                token_count INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                embedding_id TEXT
            )
            """
        )

        # Embeddings metadata table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_id TEXT UNIQUE NOT NULL,
                chunk_id TEXT REFERENCES chunks(chunk_id),
                model_name TEXT,
                dimension INTEGER,
                vector_index INTEGER,
                l2_norm REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Search cache table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT,
                result_ids TEXT,
                scores TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
            """
        )

        # Indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings_metadata(chunk_id)",
            "CREATE INDEX IF NOT EXISTS idx_search_query ON search_cache(query_hash)",
        ]
        for sql in indexes:
            self.connection.execute(sql)

        # Seed data
        self.connection.execute(
            "INSERT INTO documents (doc_id, title, source, content_hash, chunk_count, total_tokens) VALUES ('doc1', 'Seed Doc', 'unit_test', 'hash1', 1, 10)"
        )
        self.connection.execute(
            "INSERT INTO chunks (doc_id, chunk_id, chunk_index, content, token_count, start_char, end_char, embedding_id) VALUES ('doc1', 'chunk1', 0, 'sample text', 10, 0, 11, 'embed1')"
        )
        self.connection.execute(
            "INSERT INTO embeddings_metadata (embedding_id, chunk_id, model_name, dimension, vector_index, l2_norm) VALUES ('embed1', 'chunk1', 'tfidf', 100, 0, 1.0)"
        )
        self.connection.execute(
            "INSERT INTO search_cache (query_hash, query_text, result_ids, scores, hit_count) VALUES ('hashq', 'seed query', 'doc1', '1.0', 1)"
        )

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
    migrators = [EvolutionMetricsMigrator(), DigitalTwinMigrator(), RAGIndexMigrator()]

    success_count = 0

    for migrator in migrators:
        try:
            migrator.run_migrations()
            success_count += 1
        except Exception as e:
            logger.error(f"Migration failed for {migrator.__class__.__name__}: {e}")

    print(
        f"\nMigration complete: {success_count}/{len(migrators)} databases migrated successfully"
    )

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
