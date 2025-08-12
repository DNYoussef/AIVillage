#!/usr/bin/env python3
"""Database setup script for CODEX Integration Requirements.

Initializes all required SQLite databases with proper schemas, WAL mode,
and performance optimizations according to CODEX Integration Requirements.
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configurations from CODEX Integration Requirements
DATABASE_CONFIGS = {
    "evolution_metrics": {
        "path": "./data/evolution_metrics.db",
        "tables": [
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS evolution_rounds (
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
            )""",
            """CREATE TABLE IF NOT EXISTS fitness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                fitness_score REAL NOT NULL,
                performance_metrics TEXT,
                resource_usage TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (round_id) REFERENCES evolution_rounds (id)
            )""",
            """CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                cpu_usage REAL,
                memory_usage_mb REAL,
                network_io_kb REAL,
                disk_io_kb REAL,
                gpu_usage REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (round_id) REFERENCES evolution_rounds (id)
            )""",
            """CREATE TABLE IF NOT EXISTS selection_outcomes (
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
            )""",
        ],
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_evolution_rounds_number ON evolution_rounds(round_number)",
            "CREATE INDEX IF NOT EXISTS idx_fitness_agent ON fitness_metrics(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_fitness_score ON fitness_metrics(fitness_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_resource_timestamp ON resource_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_selection_parent ON selection_outcomes(parent_agent_id)",
        ],
    },
    "digital_twin": {
        "path": "./data/digital_twin.db",
        "tables": [
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS learning_profiles (
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
            )""",
            """CREATE TABLE IF NOT EXISTS learning_sessions (
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
            )""",
            """CREATE TABLE IF NOT EXISTS knowledge_states (
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
            )""",
        ],
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_profiles_user_hash ON learning_profiles(user_id_hash)",
            "CREATE INDEX IF NOT EXISTS idx_profiles_updated ON learning_profiles(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_profile ON learning_sessions(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start ON learning_sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_profile ON knowledge_states(profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_states(knowledge_domain)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_mastery ON knowledge_states(mastery_level DESC)",
        ],
    },
    "rag_index": {
        "path": "./data/rag_index.db",
        "tables": [
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS documents (
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
            )""",
            """CREATE TABLE IF NOT EXISTS chunks (
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
            )""",
            """CREATE TABLE IF NOT EXISTS embeddings_metadata (
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
            )""",
        ],
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON embeddings_metadata(faiss_index_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_queries ON embeddings_metadata(query_count DESC)",
        ],
    },
}


class DatabaseSetup:
    """Database setup and initialization class."""

    def __init__(self, base_path: str = ".") -> None:
        self.base_path = Path(base_path)
        self.data_dir = self.base_path / "data"
        self.connections: dict[str, sqlite3.Connection] = {}

    def ensure_data_directory(self) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(exist_ok=True)
        logger.info(f"Data directory ready: {self.data_dir}")

    def initialize_database(self, db_name: str, config: dict) -> sqlite3.Connection:
        """Initialize a single database with schema and optimizations."""
        db_path = self.data_dir / Path(config["path"]).name
        logger.info(f"Initializing database: {db_name} at {db_path}")

        # Connect to database
        conn = sqlite3.connect(str(db_path), isolation_level=None)

        # Enable WAL mode for concurrent access
        conn.execute("PRAGMA journal_mode=WAL")

        # Performance optimizations
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB

        # Create tables
        for table_sql in config["tables"]:
            conn.execute(table_sql)
            logger.debug(f"Created table in {db_name}")

        # Create indexes
        for index_sql in config["indexes"]:
            conn.execute(index_sql)
            logger.debug(f"Created index in {db_name}")

        # Insert initial schema version
        conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (1)")

        conn.commit()
        logger.info(f"Database {db_name} initialized successfully")

        return conn

    def setup_all_databases(self) -> dict[str, sqlite3.Connection]:
        """Set up all databases according to CODEX requirements."""
        self.ensure_data_directory()

        for db_name, config in DATABASE_CONFIGS.items():
            try:
                conn = self.initialize_database(db_name, config)
                self.connections[db_name] = conn
            except Exception as e:
                logger.exception(f"Failed to initialize {db_name}: {e}")
                raise

        logger.info("All databases initialized successfully")
        return self.connections

    def verify_database_integrity(self) -> dict[str, bool]:
        """Verify integrity of all databases."""
        results = {}

        for db_name, conn in self.connections.items():
            try:
                # Run integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]

                if result == "ok":
                    results[db_name] = True
                    logger.info(f"Database {db_name} integrity check: PASSED")
                else:
                    results[db_name] = False
                    logger.error(
                        f"Database {db_name} integrity check: FAILED - {result}"
                    )

            except Exception as e:
                results[db_name] = False
                logger.exception(f"Database {db_name} integrity check error: {e}")

        return results

    def get_database_info(self) -> dict[str, dict]:
        """Get information about all databases."""
        info = {}

        for db_name, conn in self.connections.items():
            try:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]

                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]

                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]

                size_bytes = page_count * page_size

                info[db_name] = {
                    "path": str(
                        self.data_dir / Path(DATABASE_CONFIGS[db_name]["path"]).name
                    ),
                    "tables": tables,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                }

            except Exception as e:
                logger.exception(f"Error getting info for {db_name}: {e}")
                info[db_name] = {"error": str(e)}

        return info

    def close_connections(self) -> None:
        """Close all database connections."""
        for db_name, conn in self.connections.items():
            conn.close()
            logger.info(f"Closed connection to {db_name}")

        self.connections.clear()


def main() -> bool | None:
    """Main database setup function."""
    logger.info("Starting CODEX database setup...")

    try:
        setup = DatabaseSetup()

        # Initialize all databases
        setup.setup_all_databases()

        # Verify integrity
        integrity_results = setup.verify_database_integrity()

        # Get database information
        db_info = setup.get_database_info()

        # Print summary
        print("\n" + "=" * 60)
        print("CODEX DATABASE SETUP COMPLETE")
        print("=" * 60)

        for db_name, info in db_info.items():
            if "error" not in info:
                print(f"\n{db_name.upper()} DATABASE:")
                print(f"  Path: {info['path']}")
                print(f"  Tables: {len(info['tables'])}")
                print(f"  Size: {info['size_mb']} MB")
                print(f"  Tables: {', '.join(info['tables'])}")
                print(
                    f"  Integrity: {'✅ PASSED' if integrity_results.get(db_name) else '❌ FAILED'}"
                )
            else:
                print(f"\n{db_name.upper()} DATABASE: ❌ ERROR - {info['error']}")

        print("\n" + "=" * 60)
        print("Database setup completed successfully!")
        print("All databases are ready for CODEX integration.")
        print("=" * 60)

        # Close connections
        setup.close_connections()

        return True

    except Exception as e:
        logger.exception(f"Database setup failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
