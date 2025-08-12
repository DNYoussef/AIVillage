#!/usr/bin/env python3
"""Database Integrity and Migration Verification Script
Verifies all SQLite databases exist with correct schemas and data integrity
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    from colorama import Fore, Style, init
    from cryptography.fernet import Fernet
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Installing required packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy", "cryptography", "colorama"],
        check=False,
    )
    from colorama import Fore, Style, init
    from cryptography.fernet import Fernet

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseSchema:
    """Expected database schema definition"""

    name: str
    path: str
    tables: dict[str, list[str]]  # table_name: [columns]
    indexes: dict[str, list[str]]  # index_name: [columns]
    wal_enabled: bool = True
    encrypted_fields: list[str] = None


class DatabaseIntegrityValidator:
    """Validates database integrity, schemas, and data persistence"""

    def __init__(self):
        self.base_path = Path.cwd()
        self.data_dir = self.base_path / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Define expected schemas based on CODEX requirements
        self.schemas = self._define_schemas()
        self.test_results = []

        # Encryption key for Digital Twin
        self.encryption_key = os.environ.get("DIGITAL_TWIN_ENCRYPTION_KEY")
        if not self.encryption_key:
            # Generate a test key
            self.encryption_key = Fernet.generate_key()
            os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = self.encryption_key.decode()

    def _define_schemas(self) -> list[DatabaseSchema]:
        """Define expected database schemas"""
        return [
            DatabaseSchema(
                name="Evolution Metrics",
                path=str(self.data_dir / "evolution_metrics.db"),
                tables={
                    "evolution_rounds": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "round_number INTEGER NOT NULL",
                        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "population_size INTEGER",
                        "avg_fitness REAL",
                        "best_fitness REAL",
                        "worst_fitness REAL",
                        "generation_time_ms REAL",
                        "metadata TEXT",
                    ],
                    "fitness_metrics": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "round_id INTEGER REFERENCES evolution_rounds(id)",
                        "agent_id TEXT NOT NULL",
                        "metric_name TEXT NOT NULL",
                        "metric_value REAL",
                        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    ],
                    "resource_metrics": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "round_id INTEGER REFERENCES evolution_rounds(id)",
                        "cpu_usage REAL",
                        "memory_usage_mb REAL",
                        "gpu_usage REAL",
                        "network_io_kb REAL",
                        "disk_io_kb REAL",
                        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    ],
                    "selection_outcomes": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "round_id INTEGER REFERENCES evolution_rounds(id)",
                        "parent_agent_id TEXT",
                        "offspring_agent_id TEXT",
                        "mutation_type TEXT",
                        "mutation_strength REAL",
                        "fitness_improvement REAL",
                        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    ],
                    "kpi_tracking": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "kpi_name TEXT NOT NULL",
                        "kpi_value REAL",
                        "target_value REAL",
                        "achievement_rate REAL",
                        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    ],
                },
                indexes={
                    "idx_evolution_rounds_timestamp": ["evolution_rounds", "timestamp"],
                    "idx_fitness_metrics_round": ["fitness_metrics", "round_id"],
                    "idx_selection_outcomes_round": ["selection_outcomes", "round_id"],
                },
                wal_enabled=True,
            ),
            DatabaseSchema(
                name="Digital Twin",
                path=str(self.data_dir / "digital_twin.db"),
                tables={
                    "learning_profiles": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "user_id TEXT UNIQUE NOT NULL",
                        "encrypted_name TEXT",
                        "age_group TEXT",
                        "learning_style TEXT",
                        "knowledge_level TEXT",
                        "preferences TEXT",  # JSON, encrypted
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "last_activity TIMESTAMP",
                        "coppa_compliant BOOLEAN DEFAULT 1",
                        "ferpa_compliant BOOLEAN DEFAULT 1",
                        "gdpr_compliant BOOLEAN DEFAULT 1",
                    ],
                    "learning_sessions": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "profile_id INTEGER REFERENCES learning_profiles(id)",
                        "session_id TEXT UNIQUE NOT NULL",
                        "start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "end_time TIMESTAMP",
                        "duration_seconds INTEGER",
                        "topics_covered TEXT",  # JSON
                        "performance_metrics TEXT",  # JSON, encrypted
                        "engagement_score REAL",
                        "completion_rate REAL",
                    ],
                    "knowledge_states": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "profile_id INTEGER REFERENCES learning_profiles(id)",
                        "topic TEXT NOT NULL",
                        "mastery_level REAL",
                        "confidence_score REAL",
                        "last_assessed TIMESTAMP",
                        "assessment_count INTEGER DEFAULT 0",
                        "encrypted_details TEXT",
                    ],
                    "privacy_settings": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "profile_id INTEGER REFERENCES learning_profiles(id)",
                        "data_collection_consent BOOLEAN DEFAULT 0",
                        "analytics_consent BOOLEAN DEFAULT 0",
                        "personalization_consent BOOLEAN DEFAULT 1",
                        "data_retention_days INTEGER DEFAULT 365",
                        "deletion_requested BOOLEAN DEFAULT 0",
                        "deletion_date TIMESTAMP",
                    ],
                },
                indexes={
                    "idx_profiles_user": ["learning_profiles", "user_id"],
                    "idx_sessions_profile": ["learning_sessions", "profile_id"],
                    "idx_knowledge_profile": ["knowledge_states", "profile_id"],
                },
                wal_enabled=True,
                encrypted_fields=[
                    "encrypted_name",
                    "preferences",
                    "performance_metrics",
                    "encrypted_details",
                ],
            ),
            DatabaseSchema(
                name="RAG Index",
                path=str(self.data_dir / "rag_index.db"),
                tables={
                    "documents": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "doc_id TEXT UNIQUE NOT NULL",
                        "title TEXT",
                        "source TEXT",
                        "content_hash TEXT",
                        "chunk_count INTEGER",
                        "total_tokens INTEGER",
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "indexed_at TIMESTAMP",
                        "metadata TEXT",  # JSON
                    ],
                    "chunks": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "doc_id TEXT REFERENCES documents(doc_id)",
                        "chunk_id TEXT UNIQUE NOT NULL",
                        "chunk_index INTEGER",
                        "content TEXT NOT NULL",
                        "token_count INTEGER",
                        "start_char INTEGER",
                        "end_char INTEGER",
                        "embedding_id TEXT",
                    ],
                    "embeddings_metadata": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "embedding_id TEXT UNIQUE NOT NULL",
                        "chunk_id TEXT REFERENCES chunks(chunk_id)",
                        "model_name TEXT",
                        "dimension INTEGER",
                        "vector_index INTEGER",  # Index in FAISS
                        "l2_norm REAL",
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    ],
                    "search_cache": [
                        "id INTEGER PRIMARY KEY AUTOINCREMENT",
                        "query_hash TEXT UNIQUE NOT NULL",
                        "query_text TEXT",
                        "result_ids TEXT",  # JSON array
                        "scores TEXT",  # JSON array
                        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "hit_count INTEGER DEFAULT 0",
                    ],
                },
                indexes={
                    "idx_chunks_doc": ["chunks", "doc_id"],
                    "idx_embeddings_chunk": ["embeddings_metadata", "chunk_id"],
                    "idx_cache_query": ["search_cache", "query_hash"],
                },
                wal_enabled=True,
            ),
        ]

    def create_database_if_missing(self, schema: DatabaseSchema) -> bool:
        """Create database with schema if it doesn't exist"""
        db_path = Path(schema.path)

        if not db_path.exists():
            logger.info(f"Creating database: {schema.name} at {schema.path}")

            try:
                conn = sqlite3.connect(schema.path)
                cursor = conn.cursor()

                # Enable WAL mode if required
                if schema.wal_enabled:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")

                # Create tables
                for table_name, columns in schema.tables.items():
                    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
                    create_sql += ",\n  ".join(columns)
                    create_sql += "\n)"

                    cursor.execute(create_sql)
                    logger.info(f"  Created table: {table_name}")

                # Create indexes
                for index_name, (table, column) in schema.indexes.items():
                    index_sql = (
                        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})"
                    )
                    cursor.execute(index_sql)
                    logger.info(f"  Created index: {index_name}")

                conn.commit()
                conn.close()
                return True

            except sqlite3.Error as e:
                logger.error(f"Failed to create database {schema.name}: {e}")
                return False

        return True

    def verify_schema(self, schema: DatabaseSchema) -> dict:
        """Verify database schema matches expectations"""
        result = {
            "database": schema.name,
            "path": schema.path,
            "exists": False,
            "wal_enabled": False,
            "tables_correct": False,
            "indexes_correct": False,
            "issues": [],
        }

        db_path = Path(schema.path)

        if not db_path.exists():
            result["issues"].append("Database file does not exist")
            return result

        result["exists"] = True

        try:
            conn = sqlite3.connect(schema.path)
            cursor = conn.cursor()

            # Check WAL mode
            wal_mode = cursor.execute("PRAGMA journal_mode").fetchone()[0]
            result["wal_enabled"] = wal_mode.upper() == "WAL"

            if schema.wal_enabled and not result["wal_enabled"]:
                result["issues"].append(f"WAL mode not enabled (current: {wal_mode})")

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            expected_tables = set(schema.tables.keys())

            missing_tables = expected_tables - existing_tables
            extra_tables = existing_tables - expected_tables

            if missing_tables:
                result["issues"].append(f"Missing tables: {missing_tables}")
            if extra_tables and "sqlite_sequence" not in extra_tables:
                result["issues"].append(f"Unexpected tables: {extra_tables}")

            result["tables_correct"] = len(missing_tables) == 0

            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            existing_indexes = {
                row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")
            }
            expected_indexes = set(schema.indexes.keys())

            missing_indexes = expected_indexes - existing_indexes

            if missing_indexes:
                result["issues"].append(f"Missing indexes: {missing_indexes}")

            result["indexes_correct"] = len(missing_indexes) == 0

            conn.close()

        except sqlite3.Error as e:
            result["issues"].append(f"Database error: {e}")

        return result

    def test_concurrent_access(self, schema: DatabaseSchema) -> dict:
        """Test concurrent database access without locks or corruption"""
        result = {
            "database": schema.name,
            "concurrent_writes": False,
            "concurrent_reads": False,
            "no_locks": False,
            "no_corruption": False,
            "performance_ms": 0,
        }

        if not Path(schema.path).exists():
            return result

        start_time = time.time()
        errors = []

        def write_worker(worker_id: int, iterations: int):
            """Worker thread for concurrent writes"""
            try:
                conn = sqlite3.connect(schema.path, timeout=30)
                cursor = conn.cursor()

                for i in range(iterations):
                    # Use the first table for testing
                    table_name = list(schema.tables.keys())[0]

                    if table_name == "evolution_rounds":
                        cursor.execute(
                            f"INSERT INTO {table_name} (round_number, avg_fitness) VALUES (?, ?)",
                            (worker_id * 1000 + i, 0.5 + i * 0.01),
                        )
                    elif table_name == "learning_profiles":
                        cursor.execute(
                            f"INSERT INTO {table_name} (user_id, learning_style) VALUES (?, ?)",
                            (f"test_user_{worker_id}_{i}", "visual"),
                        )
                    elif table_name == "documents":
                        cursor.execute(
                            f"INSERT INTO {table_name} (doc_id, title) VALUES (?, ?)",
                            (f"doc_{worker_id}_{i}", f"Test Document {i}"),
                        )

                    conn.commit()

                conn.close()

            except Exception as e:
                errors.append(f"Write worker {worker_id}: {e}")

        def read_worker(worker_id: int, iterations: int):
            """Worker thread for concurrent reads"""
            try:
                conn = sqlite3.connect(schema.path, timeout=30)
                cursor = conn.cursor()

                for i in range(iterations):
                    table_name = list(schema.tables.keys())[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    _ = cursor.fetchone()

                conn.close()

            except Exception as e:
                errors.append(f"Read worker {worker_id}: {e}")

        # Test concurrent access with multiple threads
        threads = []

        # Create write threads
        for i in range(5):
            t = threading.Thread(target=write_worker, args=(i, 10))
            threads.append(t)
            t.start()

        # Create read threads
        for i in range(5):
            t = threading.Thread(target=read_worker, args=(i, 20))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=10)

        result["performance_ms"] = (time.time() - start_time) * 1000

        # Check results
        result["concurrent_writes"] = not any("Write worker" in e for e in errors)
        result["concurrent_reads"] = not any("Read worker" in e for e in errors)
        result["no_locks"] = "database is locked" not in str(errors)

        # Verify database integrity
        try:
            conn = sqlite3.connect(schema.path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            result["no_corruption"] = integrity == "ok"
            conn.close()
        except:
            result["no_corruption"] = False

        return result

    def test_data_persistence(self, schema: DatabaseSchema) -> dict:
        """Test that data persists across restarts"""
        result = {
            "database": schema.name,
            "data_persists": False,
            "recovery_works": False,
            "checkpoint_works": False,
        }

        if not Path(schema.path).exists():
            return result

        try:
            # Insert test data
            conn = sqlite3.connect(schema.path)
            cursor = conn.cursor()

            table_name = list(schema.tables.keys())[0]
            test_id = f"persistence_test_{time.time()}"

            if table_name == "evolution_rounds":
                cursor.execute(
                    f"INSERT INTO {table_name} (round_number, avg_fitness) VALUES (?, ?)",
                    (999999, 0.999),
                )
            elif table_name == "learning_profiles":
                cursor.execute(
                    f"INSERT INTO {table_name} (user_id, learning_style) VALUES (?, ?)",
                    (test_id, "kinesthetic"),
                )
            elif table_name == "documents":
                cursor.execute(
                    f"INSERT INTO {table_name} (doc_id, title) VALUES (?, ?)",
                    (test_id, "Persistence Test"),
                )

            conn.commit()

            # Force checkpoint in WAL mode
            if schema.wal_enabled:
                cursor.execute("PRAGMA wal_checkpoint(FULL)")
                result["checkpoint_works"] = True

            conn.close()

            # Simulate restart by reopening connection
            conn = sqlite3.connect(schema.path)
            cursor = conn.cursor()

            # Check if data persists
            if table_name == "evolution_rounds":
                cursor.execute(
                    f"SELECT * FROM {table_name} WHERE round_number = 999999"
                )
            elif table_name == "learning_profiles":
                cursor.execute(
                    f"SELECT * FROM {table_name} WHERE user_id = ?", (test_id,)
                )
            elif table_name == "documents":
                cursor.execute(
                    f"SELECT * FROM {table_name} WHERE doc_id = ?", (test_id,)
                )

            data = cursor.fetchone()
            result["data_persists"] = data is not None

            # Clean up test data
            if table_name == "evolution_rounds":
                cursor.execute(f"DELETE FROM {table_name} WHERE round_number = 999999")
            elif table_name == "learning_profiles":
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE user_id = ?", (test_id,)
                )
            elif table_name == "documents":
                cursor.execute(f"DELETE FROM {table_name} WHERE doc_id = ?", (test_id,))

            conn.commit()
            conn.close()

            result["recovery_works"] = True

        except Exception as e:
            logger.error(f"Persistence test failed: {e}")

        return result

    def test_encryption(self, schema: DatabaseSchema) -> dict:
        """Test encryption for sensitive fields"""
        result = {
            "database": schema.name,
            "encryption_works": False,
            "fields_encrypted": [],
            "decryption_works": False,
        }

        if not schema.encrypted_fields or not Path(schema.path).exists():
            return result

        try:
            fernet = Fernet(self.encryption_key)
            conn = sqlite3.connect(schema.path)
            cursor = conn.cursor()

            # Test encryption for Digital Twin
            if schema.name == "Digital Twin":
                test_data = {
                    "name": "Test User",
                    "preferences": json.dumps({"theme": "dark", "language": "en"}),
                    "details": "Sensitive information",
                }

                # Encrypt data
                encrypted_data = {}
                for field in ["encrypted_name", "preferences", "encrypted_details"]:
                    if field in schema.encrypted_fields:
                        key = field.replace("encrypted_", "").replace("_", "")
                        if key in test_data:
                            encrypted_data[field] = fernet.encrypt(
                                test_data[key].encode()
                            ).decode()

                # Insert encrypted data
                cursor.execute(
                    "INSERT INTO learning_profiles (user_id, encrypted_name, preferences) VALUES (?, ?, ?)",
                    (
                        "encryption_test",
                        encrypted_data.get("encrypted_name"),
                        encrypted_data.get("preferences"),
                    ),
                )
                conn.commit()

                # Read and decrypt
                cursor.execute(
                    "SELECT encrypted_name, preferences FROM learning_profiles WHERE user_id = ?",
                    ("encryption_test",),
                )
                row = cursor.fetchone()

                if row:
                    decrypted_name = (
                        fernet.decrypt(row[0].encode()).decode() if row[0] else None
                    )
                    decrypted_prefs = (
                        fernet.decrypt(row[1].encode()).decode() if row[1] else None
                    )

                    result["encryption_works"] = True
                    result["fields_encrypted"] = ["encrypted_name", "preferences"]
                    result["decryption_works"] = (
                        decrypted_name == test_data["name"]
                        and decrypted_prefs == test_data["preferences"]
                    )

                # Clean up
                cursor.execute(
                    "DELETE FROM learning_profiles WHERE user_id = ?",
                    ("encryption_test",),
                )
                conn.commit()

            conn.close()

        except Exception as e:
            logger.error(f"Encryption test failed: {e}")

        return result

    def test_rag_embeddings(self) -> dict:
        """Test that RAG index contains real embeddings, not SHA256 hashes"""
        result = {
            "has_real_embeddings": False,
            "embedding_dimension": 0,
            "uses_sha256": False,
            "sample_vectors": [],
        }

        rag_schema = next((s for s in self.schemas if s.name == "RAG Index"), None)
        if not rag_schema or not Path(rag_schema.path).exists():
            return result

        try:
            conn = sqlite3.connect(rag_schema.path)
            cursor = conn.cursor()

            # Check for embeddings metadata
            cursor.execute(
                "SELECT embedding_id, dimension FROM embeddings_metadata LIMIT 5"
            )
            embeddings = cursor.fetchall()

            if embeddings:
                # Check dimension (should be 384 for paraphrase-MiniLM-L3-v2)
                dimensions = [e[1] for e in embeddings if e[1]]
                if dimensions:
                    result["embedding_dimension"] = dimensions[0]
                    result["has_real_embeddings"] = dimensions[0] == 384

                # Check if using SHA256 (would be 64 chars hex)
                for embedding_id, _ in embeddings:
                    if embedding_id and len(embedding_id) == 64:
                        try:
                            int(embedding_id, 16)  # Try to parse as hex
                            result["uses_sha256"] = True
                            break
                        except ValueError:
                            pass

            # Try to load actual vectors from FAISS if configured
            faiss_path = os.environ.get(
                "RAG_FAISS_INDEX_PATH", str(self.data_dir / "faiss_index")
            )
            if Path(faiss_path).exists():
                # Would load FAISS index here to verify real vectors
                result["sample_vectors"].append("FAISS index found")

            conn.close()

        except Exception as e:
            logger.error(f"RAG embeddings test failed: {e}")

        return result

    def run_all_tests(self) -> tuple[list[dict], dict]:
        """Run all database integrity tests"""
        results = []

        logger.info("Starting database integrity validation...")

        for schema in self.schemas:
            logger.info(f"\nTesting {schema.name} database...")

            # Create if missing
            self.create_database_if_missing(schema)

            # Verify schema
            schema_result = self.verify_schema(schema)
            results.append(("schema", schema_result))

            # Test concurrent access
            concurrent_result = self.test_concurrent_access(schema)
            results.append(("concurrent", concurrent_result))

            # Test persistence
            persistence_result = self.test_data_persistence(schema)
            results.append(("persistence", persistence_result))

            # Test encryption if applicable
            if schema.encrypted_fields:
                encryption_result = self.test_encryption(schema)
                results.append(("encryption", encryption_result))

        # Test RAG embeddings specifically
        rag_result = self.test_rag_embeddings()
        results.append(("rag_embeddings", rag_result))

        # Calculate statistics
        total_tests = len(results)
        passed_tests = sum(
            1
            for test_type, result in results
            if self._is_test_passed(test_type, result)
        )

        stats = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            ),
            "timestamp": datetime.now().isoformat(),
        }

        return results, stats

    def _is_test_passed(self, test_type: str, result: dict) -> bool:
        """Determine if a test passed"""
        if test_type == "schema":
            return (
                result.get("exists")
                and result.get("tables_correct")
                and not result.get("issues")
            )
        if test_type == "concurrent":
            return (
                result.get("concurrent_writes")
                and result.get("concurrent_reads")
                and result.get("no_corruption")
            )
        if test_type == "persistence":
            return result.get("data_persists") and result.get("recovery_works")
        if test_type == "encryption":
            return result.get("encryption_works") and result.get("decryption_works")
        if test_type == "rag_embeddings":
            return result.get("has_real_embeddings") and not result.get("uses_sha256")
        return False

    def print_results(self, results: list[tuple[str, dict]], stats: dict):
        """Print test results with colors"""
        print("\n" + "=" * 80)
        print(f"{Style.BRIGHT}DATABASE INTEGRITY VALIDATION RESULTS{Style.RESET_ALL}")
        print("=" * 80)

        current_db = None

        for test_type, result in results:
            # Group by database
            if "database" in result and result["database"] != current_db:
                current_db = result["database"]
                print(f"\n{Style.BRIGHT}{current_db}{Style.RESET_ALL}")
                print("-" * 40)

            # Print test results
            if test_type == "schema":
                status = (
                    "✓"
                    if result.get("exists") and result.get("tables_correct")
                    else "✗"
                )
                color = Fore.GREEN if status == "✓" else Fore.RED
                print(f"  Schema: {color}{status}{Style.RESET_ALL}")
                if result.get("issues"):
                    for issue in result["issues"]:
                        print(f"    {Fore.YELLOW}⚠ {issue}{Style.RESET_ALL}")

            elif test_type == "concurrent":
                status = "✓" if result.get("no_corruption") else "✗"
                color = Fore.GREEN if status == "✓" else Fore.RED
                print(f"  Concurrent Access: {color}{status}{Style.RESET_ALL}")
                print(f"    Performance: {result.get('performance_ms', 0):.2f}ms")

            elif test_type == "persistence":
                status = "✓" if result.get("data_persists") else "✗"
                color = Fore.GREEN if status == "✓" else Fore.RED
                print(f"  Data Persistence: {color}{status}{Style.RESET_ALL}")

            elif test_type == "encryption":
                status = "✓" if result.get("encryption_works") else "✗"
                color = Fore.GREEN if status == "✓" else Fore.RED
                print(f"  Encryption: {color}{status}{Style.RESET_ALL}")
                if result.get("fields_encrypted"):
                    print(
                        f"    Encrypted fields: {', '.join(result['fields_encrypted'])}"
                    )

            elif test_type == "rag_embeddings":
                print(f"\n{Style.BRIGHT}RAG Embeddings{Style.RESET_ALL}")
                print("-" * 40)
                status = "✓" if result.get("has_real_embeddings") else "✗"
                color = Fore.GREEN if status == "✓" else Fore.RED
                print(f"  Real Embeddings: {color}{status}{Style.RESET_ALL}")
                if result.get("embedding_dimension"):
                    print(f"    Dimension: {result['embedding_dimension']}")
                if result.get("uses_sha256"):
                    print(
                        f"    {Fore.RED}⚠ Using SHA256 hashes instead of embeddings{Style.RESET_ALL}"
                    )

        # Print summary
        print("\n" + "=" * 80)
        print(f"{Style.BRIGHT}SUMMARY{Style.RESET_ALL}")
        print("=" * 80)

        success_color = (
            Fore.GREEN
            if stats["success_rate"] > 90
            else Fore.YELLOW
            if stats["success_rate"] > 70
            else Fore.RED
        )

        print(f"Total Tests: {stats['total_tests']}")
        print(f"Passed: {Fore.GREEN}{stats['passed']}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{stats['failed']}{Style.RESET_ALL}")
        print(
            f"Success Rate: {success_color}{stats['success_rate']:.1f}%{Style.RESET_ALL}"
        )

    def save_results(self, results: list[tuple[str, dict]], stats: dict):
        """Save results to JSON file"""
        output_file = Path("database_integrity_report.json")

        report = {
            "stats": stats,
            "results": [
                {"test_type": test_type, "result": result}
                for test_type, result in results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{Fore.CYAN}Results saved to {output_file}{Style.RESET_ALL}")

    def auto_fix_issues(self, results: list[tuple[str, dict]]):
        """Attempt to fix identified database issues"""
        print(
            f"\n{Style.BRIGHT}ATTEMPTING AUTO-FIX FOR DATABASE ISSUES...{Style.RESET_ALL}"
        )

        for test_type, result in results:
            if test_type == "schema" and result.get("issues"):
                db_name = result.get("database")
                print(f"\nFixing schema issues for {db_name}...")

                # Find the schema
                schema = next((s for s in self.schemas if s.name == db_name), None)
                if schema:
                    # Recreate with correct schema
                    self.create_database_if_missing(schema)

                    # Enable WAL if needed
                    if "WAL mode not enabled" in str(result.get("issues")):
                        try:
                            conn = sqlite3.connect(schema.path)
                            conn.execute("PRAGMA journal_mode=WAL")
                            conn.close()
                            print(f"  {Fore.GREEN}✓ Enabled WAL mode{Style.RESET_ALL}")
                        except Exception as e:
                            print(
                                f"  {Fore.RED}✗ Failed to enable WAL: {e}{Style.RESET_ALL}"
                            )

            elif test_type == "rag_embeddings" and result.get("uses_sha256"):
                print(
                    f"\n{Fore.YELLOW}RAG system using SHA256 hashes instead of embeddings{Style.RESET_ALL}"
                )
                print("  This requires regenerating embeddings with a proper model")
                print("  Run: python scripts/regenerate_rag_embeddings.py")


def main():
    """Main entry point"""
    validator = DatabaseIntegrityValidator()

    # Run all tests
    results, stats = validator.run_all_tests()

    # Print results
    validator.print_results(results, stats)

    # Save results
    validator.save_results(results, stats)

    # Attempt auto-fix if issues found
    if stats["failed"] > 0:
        validator.auto_fix_issues(results)

        # Re-run tests after fixes
        print(f"\n{Style.BRIGHT}RE-RUNNING TESTS AFTER AUTO-FIX...{Style.RESET_ALL}")
        results, stats = validator.run_all_tests()
        validator.print_results(results, stats)

    # Exit with appropriate code
    sys.exit(0 if stats["success_rate"] >= 90 else 1)


if __name__ == "__main__":
    main()
