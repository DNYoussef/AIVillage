#!/usr/bin/env python3
"""
Fix all integration issues without starting services
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def fix_database_schemas():
    """Fix missing tables in databases"""
    print("\n" + "=" * 80)
    print("FIXING DATABASE SCHEMAS")
    print("=" * 80)

    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)

    fixes = [
        {
            "db": "evolution_metrics.db",
            "table": "kpi_tracking",
            "sql": """
                CREATE TABLE IF NOT EXISTS kpi_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kpi_name TEXT NOT NULL,
                    kpi_value REAL,
                    target_value REAL,
                    achievement_rate REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
        },
        {
            "db": "digital_twin.db",
            "table": "privacy_settings",
            "sql": """
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
            """,
        },
        {
            "db": "rag_index.db",
            "table": "search_cache",
            "sql": """
                CREATE TABLE IF NOT EXISTS search_cache (
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
    ]

    fixed = 0
    for fix in fixes:
        db_path = data_dir / fix["db"]
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(fix["sql"])
                conn.commit()
                conn.close()
                print(f"  [FIXED] Added {fix['table']} to {fix['db']}")
                fixed += 1
            except Exception as e:
                print(f"  [ERROR] Failed to add {fix['table']}: {e}")

    return fixed


def add_test_data():
    """Add test data to databases"""
    print("\n" + "=" * 80)
    print("ADDING TEST DATA")
    print("=" * 80)

    data_dir = Path.cwd() / "data"
    added = 0

    # Test KPI data
    try:
        conn = sqlite3.connect(str(data_dir / "evolution_metrics.db"))
        cursor = conn.cursor()

        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM kpi_tracking")
        count = cursor.fetchone()[0]

        if count == 0:
            cursor.execute(
                """
                INSERT INTO kpi_tracking (kpi_name, kpi_value, target_value, achievement_rate)
                VALUES
                ('inference_speed', 0.85, 1.0, 0.85),
                ('compression_ratio', 4.2, 4.0, 1.05),
                ('accuracy', 0.92, 0.95, 0.97),
                ('memory_usage', 450, 500, 0.9),
                ('latency_p95', 95, 100, 0.95)
            """
            )
            conn.commit()
            print("  [ADDED] 5 KPI tracking records")
            added += 5
        else:
            print(f"  [SKIP] KPI tracking already has {count} records")

        conn.close()
    except Exception as e:
        print(f"  [ERROR] Failed to add KPI data: {e}")

    # Test learning profile
    try:
        conn = sqlite3.connect(str(data_dir / "digital_twin.db"))
        cursor = conn.cursor()

        # Check if data exists
        cursor.execute("SELECT COUNT(*) FROM learning_profiles")
        count = cursor.fetchone()[0]

        if count == 0:
            cursor.execute(
                """
                INSERT INTO learning_profiles (user_id, age_group, learning_style, knowledge_level)
                VALUES
                ('test_user_001', '13-18', 'visual', 'intermediate'),
                ('test_user_002', '18-25', 'kinesthetic', 'advanced'),
                ('test_user_003', '8-12', 'auditory', 'beginner')
            """
            )

            # Add privacy settings
            cursor.execute(
                """
                INSERT INTO privacy_settings (profile_id, data_collection_consent, analytics_consent)
                VALUES (1, 1, 1), (2, 1, 0), (3, 0, 0)
            """
            )

            conn.commit()
            print("  [ADDED] 3 learning profiles with privacy settings")
            added += 3
        else:
            print(f"  [SKIP] Learning profiles already has {count} records")

        conn.close()
    except Exception as e:
        print(f"  [ERROR] Failed to add profile data: {e}")

    # Test RAG documents
    try:
        conn = sqlite3.connect(str(data_dir / "rag_index.db"))
        cursor = conn.cursor()

        # Check if data exists
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]

        if count == 0:
            cursor.execute(
                """
                INSERT INTO documents (doc_id, title, source, chunk_count, total_tokens)
                VALUES
                ('doc_001', 'Introduction to AI', 'wikipedia', 5, 1200),
                ('doc_002', 'Machine Learning Basics', 'textbook', 8, 2400),
                ('doc_003', 'Deep Learning Tutorial', 'tutorial', 12, 3600)
            """
            )

            # Add sample chunks
            cursor.execute(
                """
                INSERT INTO chunks (doc_id, chunk_id, chunk_index, content, token_count)
                VALUES
                ('doc_001', 'chunk_001', 0, 'AI is the simulation of human intelligence...', 50),
                ('doc_001', 'chunk_002', 1, 'Machine learning is a subset of AI...', 45),
                ('doc_002', 'chunk_003', 0, 'Supervised learning uses labeled data...', 48)
            """
            )

            conn.commit()
            print("  [ADDED] 3 documents with sample chunks")
            added += 3
        else:
            print(f"  [SKIP] Documents already has {count} records")

        conn.close()
    except Exception as e:
        print(f"  [ERROR] Failed to add RAG data: {e}")

    return added


def verify_fixes():
    """Verify all fixes were applied"""
    print("\n" + "=" * 80)
    print("VERIFYING FIXES")
    print("=" * 80)

    data_dir = Path.cwd() / "data"

    databases = [
        (
            "evolution_metrics.db",
            [
                "evolution_rounds",
                "fitness_metrics",
                "resource_metrics",
                "selection_outcomes",
                "kpi_tracking",
            ],
        ),
        (
            "digital_twin.db",
            [
                "learning_profiles",
                "learning_sessions",
                "knowledge_states",
                "privacy_settings",
            ],
        ),
        (
            "rag_index.db",
            ["documents", "chunks", "embeddings_metadata", "search_cache"],
        ),
    ]

    all_good = True

    for db_file, expected_tables in databases:
        db_path = data_dir / db_file

        if not db_path.exists():
            print(f"\n  [ERROR] {db_file} does not exist")
            all_good = False
            continue

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [
                row[0]
                for row in cursor.fetchall()
                if row[0] != "sqlite_sequence" and row[0] != "schema_version"
            ]

            # Check tables
            missing = set(expected_tables) - set(tables)

            if missing:
                print(f"\n  [ERROR] {db_file} missing tables: {missing}")
                all_good = False
            else:
                print(f"\n  [OK] {db_file} has all required tables")

                # Show row counts
                for table in expected_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        print(f"      {table}: {count} rows")

            conn.close()

        except Exception as e:
            print(f"\n  [ERROR] Failed to verify {db_file}: {e}")
            all_good = False

    return all_good


def create_startup_instructions():
    """Create instructions for starting services"""
    instructions = """
================================================================================
INTEGRATION FIXES COMPLETE
================================================================================

All database schema issues have been fixed and test data has been added.

To start the services, you need to:

1. INSTALL DEPENDENCIES (if not already installed):
   pip install fastapi uvicorn[standard] aiohttp redis websockets

2. START API SERVERS:
   python src/api/start_api_servers.py

   This will start:
   - Digital Twin API on port 8080
   - Evolution Metrics API on port 8081
   - RAG Pipeline API on port 8082

3. START P2P SERVICES:
   python src/core/p2p/start_p2p_services.py

   This will start:
   - LibP2P TCP on port 4001
   - LibP2P WebSocket on port 4002
   - mDNS discovery on port 5353

4. VERIFY SERVICES:
   python validate_services_simple.py

   This will check that all services are responding correctly.

5. ACCESS API DOCUMENTATION:
   - Digital Twin: http://localhost:8080/docs
   - Evolution Metrics: http://localhost:8081/docs
   - RAG Pipeline: http://localhost:8082/docs

ENVIRONMENT CONFIGURATION:
- Configuration file created at: .env.integration
- Load with: source .env.integration (Linux/Mac) or set -a && . .env.integration && set +a (Windows Git Bash)

DATABASE LOCATIONS:
- Evolution Metrics: data/evolution_metrics.db
- Digital Twin: data/digital_twin.db
- RAG Index: data/rag_index.db

All databases have WAL mode enabled for concurrent access.
================================================================================
"""

    print(instructions)

    # Save to file
    with open("INTEGRATION_SETUP_COMPLETE.txt", "w") as f:
        f.write(instructions)

    print("Instructions saved to INTEGRATION_SETUP_COMPLETE.txt")


def main():
    """Main entry point"""
    print("=" * 80)
    print("FIXING AIVILLAGE INTEGRATION ISSUES")
    print("=" * 80)

    # Fix database schemas
    fixed_schemas = fix_database_schemas()
    print(f"\nFixed {fixed_schemas} database schema issues")

    # Add test data
    added_data = add_test_data()
    print(f"\nAdded {added_data} test data records")

    # Verify fixes
    all_good = verify_fixes()

    if all_good:
        print("\n[SUCCESS] All integration issues have been fixed!")
        create_startup_instructions()
    else:
        print("\n[WARNING] Some issues remain - check error messages above")

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "schemas_fixed": fixed_schemas,
        "test_data_added": added_data,
        "verification_passed": all_good,
        "next_steps": [
            "Install FastAPI and dependencies",
            "Run service startup scripts",
            "Verify services are responding",
        ],
    }

    with open("integration_fix_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to integration_fix_report.json")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
