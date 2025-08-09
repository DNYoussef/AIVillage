from pathlib import Path
import sqlite3

# Ensure data directory exists
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Database initialization
databases = {
    "evolution_metrics.db": [
        "PRAGMA journal_mode=WAL;",
        "PRAGMA synchronous=NORMAL;",
        "PRAGMA cache_size=10000;",
        "PRAGMA temp_store=MEMORY;",
        "PRAGMA mmap_size=268435456;",
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
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
        );""",
        """CREATE TABLE IF NOT EXISTS fitness_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            fitness_score REAL NOT NULL,
            performance_metrics TEXT,
            resource_usage TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (round_id) REFERENCES evolution_rounds (id)
        );""",
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
        );""",
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
        );""",
        "CREATE INDEX IF NOT EXISTS idx_evolution_rounds_number ON evolution_rounds(round_number);",
        "CREATE INDEX IF NOT EXISTS idx_fitness_agent ON fitness_metrics(agent_id);",
        "CREATE INDEX IF NOT EXISTS idx_fitness_score ON fitness_metrics(fitness_score DESC);",
        "CREATE INDEX IF NOT EXISTS idx_resource_timestamp ON resource_metrics(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_selection_parent ON selection_outcomes(parent_agent_id);",
        "INSERT OR IGNORE INTO schema_version (version) VALUES (1);"
    ],

    "digital_twin.db": [
        "PRAGMA journal_mode=WAL;",
        "PRAGMA synchronous=NORMAL;",
        "PRAGMA cache_size=10000;",
        "PRAGMA temp_store=MEMORY;",
        "PRAGMA mmap_size=268435456;",
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
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
        );""",
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
        );""",
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
        );""",
        "CREATE INDEX IF NOT EXISTS idx_profiles_user_hash ON learning_profiles(user_id_hash);",
        "CREATE INDEX IF NOT EXISTS idx_profiles_updated ON learning_profiles(updated_at);",
        "CREATE INDEX IF NOT EXISTS idx_sessions_profile ON learning_sessions(profile_id);",
        "CREATE INDEX IF NOT EXISTS idx_sessions_start ON learning_sessions(start_time);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_profile ON knowledge_states(profile_id);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_states(knowledge_domain);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_mastery ON knowledge_states(mastery_level DESC);",
        "INSERT OR IGNORE INTO schema_version (version) VALUES (1);"
    ],

    "rag_index.db": [
        "PRAGMA journal_mode=WAL;",
        "PRAGMA synchronous=NORMAL;",
        "PRAGMA cache_size=10000;",
        "PRAGMA temp_store=MEMORY;",
        "PRAGMA mmap_size=268435456;",
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
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
        );""",
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
        );""",
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
        );""",
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);",
        "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index);",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_faiss ON embeddings_metadata(faiss_index_id);",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_queries ON embeddings_metadata(query_count DESC);",
        "INSERT OR IGNORE INTO schema_version (version) VALUES (1);"
    ]
}

print("Creating CODEX Integration databases...")

for db_name, commands in databases.items():
    db_path = data_dir / db_name
    print(f"\nInitializing {db_name}...")

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("BEGIN")

        for cmd in commands:
            conn.execute(cmd)

        conn.commit()

        # Verify creation
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"  Created tables: {', '.join(tables)}")

        cursor = conn.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        print(f"  Integrity check: {integrity}")

        cursor = conn.execute("PRAGMA journal_mode")
        wal_mode = cursor.fetchone()[0]
        print(f"  WAL mode: {wal_mode}")

        conn.close()

    except Exception as e:
        print(f"  Error creating {db_name}: {e}")
        if "conn" in locals():
            conn.close()

print("\nDatabase initialization complete!")
print("\nCreated databases:")
for db_name in databases:
    db_path = data_dir / db_name
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"  {db_name}: {size} bytes")
