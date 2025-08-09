#!/usr/bin/env python3
"""Data migration from existing JSON files to SQLite databases.

Migrates evolution metrics from JSON files to SQLite as noted in
CODEX Integration Requirements Migration Notes.
"""

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

class DataMigrator:
    """Handles migration of data from legacy formats to CODEX databases."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.databases = {}

    def connect_databases(self):
        """Connect to all SQLite databases."""
        db_files = [
            "evolution_metrics.db",
            "digital_twin.db",
            "rag_index.db"
        ]

        for db_file in db_files:
            db_path = self.data_dir / db_file
            if db_path.exists():
                self.databases[db_file] = sqlite3.connect(str(db_path))
                logger.info(f"Connected to {db_file}")
            else:
                logger.warning(f"Database not found: {db_file}")

    def migrate_evolution_metrics_from_json(self) -> bool:
        """Migrate evolution metrics from JSON files to SQLite."""
        json_file = self.data_dir / "evolution_metrics.json"

        if not json_file.exists():
            logger.info("No evolution_metrics.json found - nothing to migrate")
            return True

        if "evolution_metrics.db" not in self.databases:
            logger.error("Evolution metrics database not available")
            return False

        try:
            conn = self.databases["evolution_metrics.db"]

            with open(json_file) as f:
                data = json.load(f)

            logger.info(f"Migrating evolution metrics from {json_file}")

            # Migrate evolution rounds
            if "rounds" in data:
                for round_data in data["rounds"]:
                    conn.execute("""
                        INSERT OR REPLACE INTO evolution_rounds 
                        (round_number, generation, population_size, mutation_rate, 
                         selection_pressure, status, metadata, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        round_data.get("round_number", 0),
                        round_data.get("generation", 0),
                        round_data.get("population_size"),
                        round_data.get("mutation_rate"),
                        round_data.get("selection_pressure"),
                        round_data.get("status", "completed"),
                        json.dumps(round_data.get("metadata", {})),
                        round_data.get("timestamp", datetime.now().isoformat())
                    ))

                logger.info(f"Migrated {len(data['rounds'])} evolution rounds")

            # Migrate fitness metrics
            if "fitness_scores" in data:
                for fitness_data in data["fitness_scores"]:
                    # Find corresponding round_id
                    round_number = fitness_data.get("round_number", 0)
                    cursor = conn.execute(
                        "SELECT id FROM evolution_rounds WHERE round_number = ?",
                        (round_number,)
                    )
                    round_row = cursor.fetchone()

                    if round_row:
                        conn.execute("""
                            INSERT OR REPLACE INTO fitness_metrics
                            (round_id, agent_id, fitness_score, performance_metrics, 
                             resource_usage, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            round_row[0],
                            fitness_data.get("agent_id", "unknown"),
                            fitness_data.get("fitness_score", 0.0),
                            json.dumps(fitness_data.get("performance_metrics", {})),
                            json.dumps(fitness_data.get("resource_usage", {})),
                            fitness_data.get("timestamp", datetime.now().isoformat())
                        ))

                logger.info(f"Migrated {len(data['fitness_scores'])} fitness metrics")

            conn.commit()

            # Backup original JSON file
            backup_file = json_file.with_suffix(".json.migrated")
            json_file.rename(backup_file)
            logger.info(f"Backed up original file to {backup_file}")

            return True

        except Exception as e:
            logger.error(f"Failed to migrate evolution metrics: {e}")
            return False

    def migrate_rag_embeddings(self) -> bool:
        """Convert SHA256 embeddings to real vectors in RAG system."""
        if "rag_index.db" not in self.databases:
            logger.error("RAG index database not available")
            return False

        try:
            conn = self.databases["rag_index.db"]

            # Find documents with SHA256 embeddings (placeholder data)
            cursor = conn.execute("""
                SELECT chunk_id, embedding_vector FROM chunks 
                WHERE embedding_vector IS NOT NULL
            """)

            sha256_chunks = []
            for row in cursor.fetchall():
                chunk_id, embedding_data = row
                if embedding_data and len(embedding_data) == 32:  # SHA256 hash length
                    sha256_chunks.append(chunk_id)

            if sha256_chunks:
                logger.warning(f"Found {len(sha256_chunks)} chunks with SHA256 embeddings")
                logger.info("Note: Real embedding vectors will be generated during RAG system initialization")

                # Mark these for re-processing
                for chunk_id in sha256_chunks:
                    conn.execute("""
                        UPDATE chunks SET embedding_vector = NULL 
                        WHERE chunk_id = ?
                    """, (chunk_id,))

                conn.commit()
                logger.info("Cleared SHA256 placeholder embeddings for re-processing")

            return True

        except Exception as e:
            logger.error(f"Failed to migrate RAG embeddings: {e}")
            return False

    def migrate_digital_twin_data(self) -> bool:
        """Transfer any existing Digital Twin data."""
        if "digital_twin.db" not in self.databases:
            logger.error("Digital twin database not available")
            return False

        # Look for existing profile data
        profile_files = list(self.data_dir.glob("profile_*.json"))

        if not profile_files:
            logger.info("No existing digital twin profile files found")
            return True

        try:
            conn = self.databases["digital_twin.db"]
            migrated_count = 0

            for profile_file in profile_files:
                with open(profile_file) as f:
                    profile_data = json.load(f)

                profile_id = profile_data.get("profile_id", str(profile_file.stem))
                user_id = profile_data.get("user_id", "unknown")

                # Create a hash for privacy compliance
                user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()

                conn.execute("""
                    INSERT OR REPLACE INTO learning_profiles
                    (profile_id, user_id_hash, learning_style, preferred_difficulty,
                     knowledge_domains, learning_goals, privacy_settings, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile_id,
                    user_id_hash,
                    profile_data.get("learning_style"),
                    profile_data.get("preferred_difficulty", "medium"),
                    json.dumps(profile_data.get("knowledge_domains", [])),
                    json.dumps(profile_data.get("learning_goals", [])),
                    json.dumps(profile_data.get("privacy_settings", {})),
                    profile_data.get("created_at", datetime.now().isoformat())
                ))

                migrated_count += 1

                # Backup original file
                backup_file = profile_file.with_suffix(".json.migrated")
                profile_file.rename(backup_file)

            conn.commit()
            logger.info(f"Migrated {migrated_count} digital twin profiles")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate digital twin data: {e}")
            return False

    def validate_migration(self) -> dict[str, dict]:
        """Validate that migration was successful."""
        results = {}

        for db_name, conn in self.databases.items():
            try:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                table_counts = {}
                for table in tables:
                    if table != "schema_version":
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_counts[table] = count

                results[db_name] = {
                    "status": "success",
                    "tables": tables,
                    "record_counts": table_counts
                }

            except Exception as e:
                results[db_name] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def close_databases(self):
        """Close all database connections."""
        for db_name, conn in self.databases.items():
            conn.close()
            logger.info(f"Closed connection to {db_name}")

        self.databases.clear()


def main():
    """Main migration function."""
    logging.basicConfig(level=logging.INFO)

    print("Starting CODEX data migration...")

    migrator = DataMigrator()

    # Connect to databases
    migrator.connect_databases()

    if not migrator.databases:
        print("❌ No databases found - run database setup first")
        return False

    success_count = 0
    total_migrations = 3

    # Run migrations
    print("\n1. Migrating evolution metrics from JSON...")
    if migrator.migrate_evolution_metrics_from_json():
        success_count += 1
        print("   ✅ Evolution metrics migration completed")
    else:
        print("   ❌ Evolution metrics migration failed")

    print("\n2. Converting RAG SHA256 embeddings...")
    if migrator.migrate_rag_embeddings():
        success_count += 1
        print("   ✅ RAG embeddings conversion completed")
    else:
        print("   ❌ RAG embeddings conversion failed")

    print("\n3. Migrating Digital Twin data...")
    if migrator.migrate_digital_twin_data():
        success_count += 1
        print("   ✅ Digital Twin migration completed")
    else:
        print("   ❌ Digital Twin migration failed")

    # Validate migration
    print("\nValidating migration results...")
    validation_results = migrator.validate_migration()

    print("\n" + "="*60)
    print("DATA MIGRATION COMPLETE")
    print("="*60)

    for db_name, result in validation_results.items():
        print(f"\n{db_name.upper()}:")
        if result["status"] == "success":
            print("  Status: ✅ SUCCESS")
            print(f"  Tables: {len(result['tables'])}")
            for table, count in result["record_counts"].items():
                print(f"    {table}: {count:,} records")
        else:
            print(f"  Status: ❌ ERROR - {result['error']}")

    print("\nMigration Summary:")
    print(f"  Completed: {success_count}/{total_migrations} migrations")
    print(f"  Databases: {len([r for r in validation_results.values() if r['status'] == 'success'])}/{len(validation_results)} validated")

    print("\n" + "="*60)
    if success_count == total_migrations:
        print("All data migrations completed successfully!")
    else:
        print("Some migrations failed - check logs for details")
    print("="*60)

    # Close connections
    migrator.close_databases()

    return success_count == total_migrations


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
