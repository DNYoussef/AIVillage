"""Database migration system for AIVillage CODEX components.

This module provides comprehensive database schema migration functionality with:
- Version tracking and history
- Forward and backward migrations  
- Data preservation during schema changes
- Migration validation and rollback
- Cross-database migration support
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""

    def __init__(
        self,
        version: int,
        name: str,
        description: str,
        up_sql: str,
        down_sql: str = "",
        data_migration: callable | None = None,
        requires: list[int] | None = None
    ):
        self.version = version
        self.name = name
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.data_migration = data_migration
        self.requires = requires or []

    def __repr__(self):
        return f"Migration(v{self.version}: {self.name})"


class MigrationManager:
    """Manages database schema migrations with version tracking."""

    def __init__(self, database_manager):
        self.database_manager = database_manager
        self.migrations: dict[str, list[Migration]] = {
            "evolution_metrics": [],
            "digital_twin": [],
            "rag_index": []
        }
        self._load_migrations()

    def _load_migrations(self):
        """Load all migration definitions."""
        # Evolution Metrics migrations
        self.migrations["evolution_metrics"] = [
            Migration(
                version=1,
                name="initial_schema",
                description="Create initial evolution metrics tables",
                up_sql=self.database_manager.EVOLUTION_METRICS_SCHEMA,
                down_sql="""
                DROP TABLE IF EXISTS selection_outcomes;
                DROP TABLE IF EXISTS resource_metrics;
                DROP TABLE IF EXISTS fitness_metrics;
                DROP TABLE IF EXISTS evolution_rounds;
                DROP TABLE IF EXISTS schema_version;
                """
            ),
            Migration(
                version=2,
                name="add_agent_metadata",
                description="Add metadata columns for enhanced agent tracking",
                up_sql="""
                ALTER TABLE fitness_metrics ADD COLUMN agent_type TEXT;
                ALTER TABLE fitness_metrics ADD COLUMN generation INTEGER DEFAULT 0;
                ALTER TABLE evolution_rounds ADD COLUMN configuration TEXT;
                
                CREATE INDEX IF NOT EXISTS idx_fitness_metrics_agent_type ON fitness_metrics(agent_type);
                CREATE INDEX IF NOT EXISTS idx_fitness_metrics_generation ON fitness_metrics(generation);
                """,
                down_sql="""
                DROP INDEX IF EXISTS idx_fitness_metrics_generation;
                DROP INDEX IF EXISTS idx_fitness_metrics_agent_type;
                -- Note: SQLite doesn't support DROP COLUMN, would need table recreation
                """
            )
        ]

        # Digital Twin migrations
        self.migrations["digital_twin"] = [
            Migration(
                version=1,
                name="initial_schema",
                description="Create initial digital twin tables",
                up_sql=self.database_manager.DIGITAL_TWIN_SCHEMA,
                down_sql="""
                DROP TABLE IF EXISTS knowledge_states;
                DROP TABLE IF EXISTS learning_sessions;
                DROP TABLE IF EXISTS learning_profiles;
                DROP TABLE IF EXISTS schema_version;
                """
            ),
            Migration(
                version=2,
                name="add_privacy_fields",
                description="Add privacy and compliance tracking fields",
                up_sql="""
                ALTER TABLE learning_profiles ADD COLUMN privacy_consent TEXT;
                ALTER TABLE learning_profiles ADD COLUMN data_retention_until TIMESTAMP;
                ALTER TABLE learning_sessions ADD COLUMN privacy_level TEXT DEFAULT 'standard';
                
                CREATE INDEX IF NOT EXISTS idx_learning_profiles_retention ON learning_profiles(data_retention_until);
                """,
                down_sql="""
                DROP INDEX IF EXISTS idx_learning_profiles_retention;
                -- Note: SQLite doesn't support DROP COLUMN, would need table recreation
                """
            )
        ]

        # RAG Index migrations
        self.migrations["rag_index"] = [
            Migration(
                version=1,
                name="initial_schema",
                description="Create initial RAG index tables",
                up_sql=self.database_manager.RAG_INDEX_SCHEMA,
                down_sql="""
                DROP TABLE IF EXISTS query_cache;
                DROP TABLE IF EXISTS embeddings_metadata;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS documents;
                DROP TABLE IF EXISTS schema_version;
                """
            ),
            Migration(
                version=2,
                name="add_semantic_search",
                description="Add semantic search enhancements",
                up_sql="""
                ALTER TABLE chunks ADD COLUMN semantic_keywords TEXT;
                ALTER TABLE documents ADD COLUMN document_category TEXT;
                ALTER TABLE query_cache ADD COLUMN query_intent TEXT;
                
                CREATE TABLE IF NOT EXISTS semantic_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id TEXT UNIQUE NOT NULL,
                    center_vector BLOB,
                    document_count INTEGER DEFAULT 0,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(document_category);
                CREATE INDEX IF NOT EXISTS idx_semantic_clusters_keywords ON semantic_clusters(keywords);
                """,
                down_sql="""
                DROP INDEX IF EXISTS idx_semantic_clusters_keywords;
                DROP INDEX IF EXISTS idx_documents_category;
                DROP TABLE IF EXISTS semantic_clusters;
                -- Note: SQLite doesn't support DROP COLUMN, would need table recreation
                """
            )
        ]

    async def get_current_version(self, database: str) -> int:
        """Get current schema version for database."""
        if database not in self.database_manager.connections:
            raise ValueError(f"Database {database} not initialized")

        with self.database_manager.get_connection(database) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0
            except sqlite3.OperationalError:
                # schema_version table doesn't exist
                return 0

    async def get_migration_history(self, database: str) -> list[dict[str, Any]]:
        """Get migration history for database."""
        if database not in self.database_manager.connections:
            raise ValueError(f"Database {database} not initialized")

        with self.database_manager.get_connection(database) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT version, applied_at, description 
                FROM schema_version 
                ORDER BY version ASC
                """)

                history = []
                for row in cursor.fetchall():
                    history.append({
                        "version": row[0],
                        "applied_at": row[1],
                        "description": row[2]
                    })

                return history

            except sqlite3.OperationalError:
                # schema_version table doesn't exist
                return []

    async def migrate_to_latest(self, database: str) -> bool:
        """Migrate database to latest schema version."""
        if database not in self.migrations:
            raise ValueError(f"No migrations defined for database {database}")

        current_version = await self.get_current_version(database)
        available_migrations = self.migrations[database]

        if not available_migrations:
            logger.info(f"No migrations available for {database}")
            return True

        latest_version = max(m.version for m in available_migrations)

        if current_version >= latest_version:
            logger.info(f"Database {database} is already at latest version {latest_version}")
            return True

        logger.info(f"Migrating {database} from version {current_version} to {latest_version}")

        # Get migrations to apply
        migrations_to_apply = [
            m for m in available_migrations
            if m.version > current_version
        ]
        migrations_to_apply.sort(key=lambda m: m.version)

        # Apply migrations in order
        for migration in migrations_to_apply:
            success = await self._apply_migration(database, migration)
            if not success:
                logger.error(f"Failed to apply migration {migration.version} to {database}")
                return False
            logger.info(f"Applied migration {migration.version}: {migration.name}")

        logger.info(f"Successfully migrated {database} to version {latest_version}")
        return True

    async def migrate_to_version(self, database: str, target_version: int) -> bool:
        """Migrate database to specific version."""
        current_version = await self.get_current_version(database)

        if current_version == target_version:
            logger.info(f"Database {database} is already at version {target_version}")
            return True

        if current_version < target_version:
            # Forward migration
            migrations_to_apply = [
                m for m in self.migrations[database]
                if current_version < m.version <= target_version
            ]
            migrations_to_apply.sort(key=lambda m: m.version)

            for migration in migrations_to_apply:
                success = await self._apply_migration(database, migration)
                if not success:
                    return False

        else:
            # Backward migration (rollback)
            migrations_to_rollback = [
                m for m in self.migrations[database]
                if target_version < m.version <= current_version
            ]
            migrations_to_rollback.sort(key=lambda m: m.version, reverse=True)

            for migration in migrations_to_rollback:
                success = await self._rollback_migration(database, migration)
                if not success:
                    return False

        return True

    async def _apply_migration(self, database: str, migration: Migration) -> bool:
        """Apply a single migration."""
        logger.info(f"Applying migration {migration.version}: {migration.name}")

        with self.database_manager.get_connection(database) as conn:
            try:
                # Start transaction
                conn.execute("BEGIN")

                # Check if migration already applied
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM schema_version WHERE version = ?", (migration.version,))
                if cursor.fetchone():
                    logger.warning(f"Migration {migration.version} already applied")
                    conn.rollback()
                    return True

                # Execute migration SQL
                if migration.up_sql.strip():
                    conn.executescript(migration.up_sql)

                # Execute data migration if provided
                if migration.data_migration:
                    await migration.data_migration(conn)

                # Record migration
                conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (migration.version, migration.description)
                )

                # Commit transaction
                conn.commit()

                logger.info(f"Successfully applied migration {migration.version}")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to apply migration {migration.version}: {e}")
                return False

    async def _rollback_migration(self, database: str, migration: Migration) -> bool:
        """Rollback a single migration."""
        logger.info(f"Rolling back migration {migration.version}: {migration.name}")

        if not migration.down_sql.strip():
            logger.error(f"No rollback SQL provided for migration {migration.version}")
            return False

        with self.database_manager.get_connection(database) as conn:
            try:
                # Start transaction
                conn.execute("BEGIN")

                # Execute rollback SQL
                conn.executescript(migration.down_sql)

                # Remove migration record
                conn.execute("DELETE FROM schema_version WHERE version = ?", (migration.version,))

                # Commit transaction
                conn.commit()

                logger.info(f"Successfully rolled back migration {migration.version}")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to rollback migration {migration.version}: {e}")
                return False

    async def validate_migrations(self, database: str) -> dict[str, Any]:
        """Validate migration consistency and dependencies."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        migrations = self.migrations.get(database, [])
        if not migrations:
            validation_results["warnings"].append(f"No migrations defined for {database}")
            return validation_results

        # Check version sequences
        versions = sorted([m.version for m in migrations])
        for i, version in enumerate(versions[1:], 1):
            if version != versions[i-1] + 1:
                validation_results["errors"].append(
                    f"Missing migration version {versions[i-1] + 1} (gap between {versions[i-1]} and {version})"
                )
                validation_results["valid"] = False

        # Check for duplicate versions
        version_counts = {}
        for migration in migrations:
            version_counts[migration.version] = version_counts.get(migration.version, 0) + 1

        for version, count in version_counts.items():
            if count > 1:
                validation_results["errors"].append(f"Duplicate migration version {version}")
                validation_results["valid"] = False

        # Check dependencies
        for migration in migrations:
            for required_version in migration.requires:
                if required_version not in versions:
                    validation_results["errors"].append(
                        f"Migration {migration.version} requires version {required_version} which doesn't exist"
                    )
                    validation_results["valid"] = False

        return validation_results

    async def create_migration_template(self, database: str, name: str) -> str:
        """Create a template for a new migration."""
        if database not in self.migrations:
            raise ValueError(f"Unknown database {database}")

        # Get next version number
        current_migrations = self.migrations[database]
        next_version = max([m.version for m in current_migrations], default=0) + 1

        template = f'''"""
Migration {next_version}: {name}
Database: {database}
Created: {datetime.now().isoformat()}
"""

from core.database.migrations import Migration

migration_{next_version} = Migration(
    version={next_version},
    name="{name}",
    description="TODO: Add description",
    up_sql="""
    -- TODO: Add forward migration SQL
    """,
    down_sql="""
    -- TODO: Add rollback migration SQL
    """,
    # data_migration=custom_data_migration_function,  # Optional
    # requires=[{next_version-1}]  # Optional dependencies
)

# Optional: Custom data migration function
async def custom_data_migration_function(conn):
    """Custom data migration logic."""
    # TODO: Add custom data migration code if needed
    pass
'''

        return template


class DataMigrator:
    """Handles data migration from existing systems."""

    def __init__(self, database_manager):
        self.database_manager = database_manager

    async def migrate_evolution_metrics_from_json(self, json_file_path: str) -> bool:
        """Migrate evolution metrics from JSON file to SQLite database."""
        json_path = Path(json_file_path)
        if not json_path.exists():
            logger.warning(f"JSON file {json_file_path} not found, skipping migration")
            return True

        logger.info(f"Migrating evolution metrics from {json_file_path}")

        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {json_file_path}: {e}")
            return False

        with self.database_manager.get_connection("evolution_metrics") as conn:
            try:
                conn.execute("BEGIN")

                # Migrate evolution rounds
                if "rounds" in data:
                    for round_data in data["rounds"]:
                        conn.execute("""
                        INSERT OR REPLACE INTO evolution_rounds 
                        (start_time, end_time, status, agent_count, success_rate, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            round_data.get("start_time", time.time()),
                            round_data.get("end_time"),
                            round_data.get("status", "completed"),
                            round_data.get("agent_count", 0),
                            round_data.get("success_rate", 0.0),
                            json.dumps(round_data.get("metadata", {}))
                        ))
                        round_id = conn.lastrowid

                        # Migrate fitness metrics for this round
                        if "fitness_metrics" in round_data:
                            for metric in round_data["fitness_metrics"]:
                                conn.execute("""
                                INSERT INTO fitness_metrics
                                (round_id, agent_id, evolution_id, fitness_score, improvement_delta, timestamp, metadata)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    round_id,
                                    metric.get("agent_id", "unknown"),
                                    metric.get("evolution_id", "unknown"),
                                    metric.get("fitness_score", 0.0),
                                    metric.get("improvement_delta", 0.0),
                                    metric.get("timestamp", time.time()),
                                    json.dumps(metric.get("metadata", {}))
                                ))

                conn.commit()
                logger.info(f"Successfully migrated evolution metrics from {json_file_path}")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to migrate evolution metrics: {e}")
                return False

    async def migrate_digital_twin_profiles(self, profiles_dir: str) -> bool:
        """Migrate digital twin profiles from directory structure to database."""
        profiles_path = Path(profiles_dir)
        if not profiles_path.exists():
            logger.warning(f"Profiles directory {profiles_dir} not found, skipping migration")
            return True

        logger.info(f"Migrating digital twin profiles from {profiles_dir}")

        with self.database_manager.get_connection("digital_twin") as conn:
            try:
                conn.execute("BEGIN")

                # Find all profile JSON files
                for profile_file in profiles_path.glob("*.json"):
                    try:
                        with open(profile_file) as f:
                            profile_data = json.load(f)

                        # Insert learning profile
                        conn.execute("""
                        INSERT OR REPLACE INTO learning_profiles
                        (student_id, name, age, grade_level, language, region, learning_style,
                         strengths, challenges, interests, attention_span_minutes,
                         preferred_session_times, accessibility_needs, motivation_triggers)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            profile_data.get("student_id", profile_file.stem),
                            profile_data.get("name", "Unknown"),
                            profile_data.get("age", 10),
                            profile_data.get("grade_level", 5),
                            profile_data.get("language", "en"),
                            profile_data.get("region", "US"),
                            profile_data.get("learning_style", "visual"),
                            json.dumps(profile_data.get("strengths", [])),
                            json.dumps(profile_data.get("challenges", [])),
                            json.dumps(profile_data.get("interests", [])),
                            profile_data.get("attention_span_minutes", 15),
                            json.dumps(profile_data.get("preferred_session_times", [])),
                            json.dumps(profile_data.get("accessibility_needs", [])),
                            json.dumps(profile_data.get("motivation_triggers", []))
                        ))

                        # Migrate learning sessions if present
                        if "sessions" in profile_data:
                            student_id = profile_data.get("student_id", profile_file.stem)
                            for session in profile_data["sessions"]:
                                conn.execute("""
                                INSERT INTO learning_sessions
                                (session_id, student_id, start_time, end_time, duration_minutes,
                                 concepts_covered, questions_asked, questions_correct,
                                 engagement_score, difficulty_level, session_notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    session.get("session_id", f"{student_id}_{time.time()}"),
                                    student_id,
                                    session.get("start_time", datetime.now().isoformat()),
                                    session.get("end_time"),
                                    session.get("duration_minutes", 0),
                                    json.dumps(session.get("concepts_covered", [])),
                                    session.get("questions_asked", 0),
                                    session.get("questions_correct", 0),
                                    session.get("engagement_score", 0.0),
                                    session.get("difficulty_level", 0.5),
                                    session.get("session_notes", "")
                                ))

                    except Exception as e:
                        logger.warning(f"Failed to migrate profile {profile_file}: {e}")
                        continue

                conn.commit()
                logger.info(f"Successfully migrated digital twin profiles from {profiles_dir}")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to migrate digital twin profiles: {e}")
                return False


# Global migration manager instance
_migration_manager: MigrationManager | None = None


async def get_migration_manager(database_manager) -> MigrationManager:
    """Get global migration manager instance."""
    global _migration_manager

    if _migration_manager is None:
        _migration_manager = MigrationManager(database_manager)

    return _migration_manager


if __name__ == "__main__":
    import asyncio

    from database_manager import DatabaseManager

    async def main():
        """Test migration system."""
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Get migration manager
        migration_manager = await get_migration_manager(db_manager)

        # Validate all migrations
        for database in ["evolution_metrics", "digital_twin", "rag_index"]:
            validation = await migration_manager.validate_migrations(database)
            print(f"Migration validation for {database}: {validation}")

            # Show current version
            current_version = await migration_manager.get_current_version(database)
            print(f"Current version for {database}: {current_version}")

            # Migrate to latest
            success = await migration_manager.migrate_to_latest(database)
            print(f"Migration to latest for {database}: {'success' if success else 'failed'}")

        # Test data migration
        data_migrator = DataMigrator(db_manager)

        # Try to migrate from common file locations
        await data_migrator.migrate_evolution_metrics_from_json("./evolution_metrics.json")
        await data_migrator.migrate_digital_twin_profiles("./profiles")

        await db_manager.close()

    asyncio.run(main())
