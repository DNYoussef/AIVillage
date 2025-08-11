#!/usr/bin/env python3
"""Simplified Database Validation - No external dependencies
Checks database existence, schemas, and basic integrity
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys


class DatabaseValidator:
    """Simple database validator"""

    def __init__(self):
        self.base_path = Path.cwd()
        self.data_dir = self.base_path / "data"
        self.results = []

    def check_database(self, name, path, expected_tables):
        """Check if database exists and has expected tables"""
        result = {
            "name": name,
            "path": str(path),
            "exists": False,
            "tables": [],
            "missing_tables": [],
            "wal_enabled": False,
            "row_counts": {},
            "issues": [],
        }

        if not Path(path).exists():
            result["issues"].append("Database file does not exist")
            return result

        result["exists"] = True

        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()

            # Check WAL mode
            wal_mode = cursor.execute("PRAGMA journal_mode").fetchone()
            if wal_mode:
                result["wal_enabled"] = wal_mode[0].upper() == "WAL"

            # Get existing tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            result["tables"] = existing_tables

            # Check for missing tables
            missing = set(expected_tables) - set(existing_tables)
            if missing:
                result["missing_tables"] = list(missing)
                result["issues"].append(f"Missing tables: {', '.join(missing)}")

            # Get row counts
            for table in existing_tables:
                if table != "sqlite_sequence":
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        result["row_counts"][table] = count
                    except:
                        result["row_counts"][table] = "error"

            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            if integrity != "ok":
                result["issues"].append(f"Integrity check failed: {integrity}")

            conn.close()

        except sqlite3.Error as e:
            result["issues"].append(f"Database error: {e}")

        return result

    def create_missing_databases(self):
        """Create any missing databases with proper schemas"""
        databases = [
            {
                "name": "Evolution Metrics",
                "path": self.data_dir / "evolution_metrics.db",
                "tables": {
                    "evolution_rounds": """
                        CREATE TABLE evolution_rounds (
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
                    """,
                    "fitness_metrics": """
                        CREATE TABLE fitness_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            round_id INTEGER REFERENCES evolution_rounds(id),
                            agent_id TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value REAL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """,
                    "resource_metrics": """
                        CREATE TABLE resource_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            round_id INTEGER REFERENCES evolution_rounds(id),
                            cpu_usage REAL,
                            memory_usage_mb REAL,
                            gpu_usage REAL,
                            network_io_kb REAL,
                            disk_io_kb REAL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """,
                    "selection_outcomes": """
                        CREATE TABLE selection_outcomes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            round_id INTEGER REFERENCES evolution_rounds(id),
                            parent_agent_id TEXT,
                            offspring_agent_id TEXT,
                            mutation_type TEXT,
                            mutation_strength REAL,
                            fitness_improvement REAL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """,
                    "kpi_tracking": """
                        CREATE TABLE kpi_tracking (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            kpi_name TEXT NOT NULL,
                            kpi_value REAL,
                            target_value REAL,
                            achievement_rate REAL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """,
                },
            },
            {
                "name": "Digital Twin",
                "path": self.data_dir / "digital_twin.db",
                "tables": {
                    "learning_profiles": """
                        CREATE TABLE learning_profiles (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT UNIQUE NOT NULL,
                            encrypted_name TEXT,
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
                    """,
                    "learning_sessions": """
                        CREATE TABLE learning_sessions (
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
                    """,
                    "knowledge_states": """
                        CREATE TABLE knowledge_states (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            profile_id INTEGER REFERENCES learning_profiles(id),
                            topic TEXT NOT NULL,
                            mastery_level REAL,
                            confidence_score REAL,
                            last_assessed TIMESTAMP,
                            assessment_count INTEGER DEFAULT 0,
                            encrypted_details TEXT
                        )
                    """,
                    "privacy_settings": """
                        CREATE TABLE privacy_settings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            profile_id INTEGER REFERENCES learning_profiles(id),
                            data_collection_consent BOOLEAN DEFAULT 0,
                            analytics_consent BOOLEAN DEFAULT 0,
                            personalization_consent BOOLEAN DEFAULT 1,
                            data_retention_days INTEGER DEFAULT 365,
                            deletion_requested BOOLEAN DEFAULT 0,
                            deletion_date TIMESTAMP
                        )
                    """,
                },
            },
            {
                "name": "RAG Index",
                "path": self.data_dir / "rag_index.db",
                "tables": {
                    "documents": """
                        CREATE TABLE documents (
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
                    """,
                    "chunks": """
                        CREATE TABLE chunks (
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
                    """,
                    "embeddings_metadata": """
                        CREATE TABLE embeddings_metadata (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            embedding_id TEXT UNIQUE NOT NULL,
                            chunk_id TEXT REFERENCES chunks(chunk_id),
                            model_name TEXT,
                            dimension INTEGER,
                            vector_index INTEGER,
                            l2_norm REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """,
                    "search_cache": """
                        CREATE TABLE search_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query_hash TEXT UNIQUE NOT NULL,
                            query_text TEXT,
                            result_ids TEXT,
                            scores TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            hit_count INTEGER DEFAULT 0
                        )
                    """,
                },
            },
        ]

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

        created = []

        for db_config in databases:
            db_path = db_config["path"]

            if not db_path.exists():
                print(f"\nCreating database: {db_config['name']}")

                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()

                    # Enable WAL mode
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")

                    # Create tables
                    for table_name, create_sql in db_config["tables"].items():
                        cursor.execute(create_sql)
                        print(f"  Created table: {table_name}")

                    conn.commit()
                    conn.close()

                    created.append(db_config["name"])

                except sqlite3.Error as e:
                    print(f"  ERROR: Failed to create database: {e}")

        return created

    def run_validation(self):
        """Run database validation"""
        print("=" * 80)
        print("DATABASE VALIDATION")
        print("=" * 80)

        # Define expected databases
        databases = [
            {
                "name": "Evolution Metrics",
                "path": self.data_dir / "evolution_metrics.db",
                "tables": [
                    "evolution_rounds",
                    "fitness_metrics",
                    "resource_metrics",
                    "selection_outcomes",
                    "kpi_tracking",
                ],
            },
            {
                "name": "Digital Twin",
                "path": self.data_dir / "digital_twin.db",
                "tables": [
                    "learning_profiles",
                    "learning_sessions",
                    "knowledge_states",
                    "privacy_settings",
                ],
            },
            {
                "name": "RAG Index",
                "path": self.data_dir / "rag_index.db",
                "tables": [
                    "documents",
                    "chunks",
                    "embeddings_metadata",
                    "search_cache",
                ],
            },
        ]

        # Check each database
        for db_config in databases:
            print(f"\nChecking {db_config['name']}...")
            result = self.check_database(db_config["name"], db_config["path"], db_config["tables"])
            self.results.append(result)

            if result["exists"]:
                print(f"  [EXISTS] at {result['path']}")
                print(f"  Tables: {len(result['tables'])}")
                print(f"  WAL Mode: {'Enabled' if result['wal_enabled'] else 'Disabled'}")

                if result["row_counts"]:
                    print("  Row counts:")
                    for table, count in result["row_counts"].items():
                        print(f"    - {table}: {count}")

                if result["issues"]:
                    print("  ISSUES:")
                    for issue in result["issues"]:
                        print(f"    - {issue}")
            else:
                print("  [MISSING] Database does not exist")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total = len(self.results)
        existing = sum(1 for r in self.results if r["exists"])
        healthy = sum(1 for r in self.results if r["exists"] and not r["issues"])

        print(f"Total Databases: {total}")
        print(f"Existing: {existing}")
        print(f"Healthy: {healthy}")
        print(f"Missing: {total - existing}")

        # Save results
        output_file = Path("database_validation.json")
        with open(output_file, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "existing": existing,
                        "healthy": healthy,
                        "missing": total - existing,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {output_file}")

        return existing == total and healthy == total

    def fix_issues(self):
        """Attempt to fix database issues"""
        print("\n" + "=" * 80)
        print("ATTEMPTING TO FIX DATABASE ISSUES")
        print("=" * 80)

        # Create missing databases
        created = self.create_missing_databases()

        if created:
            print(f"\nCreated {len(created)} missing database(s)")

        # Fix WAL mode for existing databases
        for result in self.results:
            if result["exists"] and not result["wal_enabled"]:
                print(f"\nEnabling WAL mode for {result['name']}...")
                try:
                    conn = sqlite3.connect(result["path"])
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.close()
                    print("  [SUCCESS] WAL mode enabled")
                except Exception as e:
                    print(f"  [ERROR] Failed to enable WAL: {e}")

        return len(created) > 0


def main():
    """Main entry point"""
    validator = DatabaseValidator()

    # Run validation
    all_healthy = validator.run_validation()

    if not all_healthy:
        # Attempt fixes
        validator.fix_issues()

        # Re-run validation
        print("\n" + "=" * 80)
        print("RE-RUNNING VALIDATION AFTER FIXES")
        print("=" * 80)

        validator.results = []
        all_healthy = validator.run_validation()

    return 0 if all_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
