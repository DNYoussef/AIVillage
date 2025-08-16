"""Evolution Metrics Migration Script.

Migrates evolution metrics from JSON files to CODEX-compliant SQLite database.
"""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CODEX environment variables
AIVILLAGE_DB_PATH = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")
AIVILLAGE_STORAGE_BACKEND = os.getenv("AIVILLAGE_STORAGE_BACKEND", "sqlite")
AIVILLAGE_METRICS_FILE = os.getenv("AIVILLAGE_METRICS_FILE", "evolution_metrics.json")
AIVILLAGE_LOG_DIR = os.getenv("AIVILLAGE_LOG_DIR", "./evolution_logs")


class EvolutionMetricsMigrator:
    """Handles migration from JSON files to SQLite database."""

    def __init__(self) -> None:
        self.db_path = Path(AIVILLAGE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.metrics_file = Path(AIVILLAGE_METRICS_FILE)
        self.log_dir = Path(AIVILLAGE_LOG_DIR)

        self.migration_log = []

    def create_database_schema(self) -> None:
        """Create CODEX-compliant database schema."""
        logger.info("Creating evolution metrics database schema...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode for concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=268435456")

        # Evolution rounds tracking
        cursor.execute(
            """
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
        """
        )

        # Agent fitness metrics
        cursor.execute(
            """
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
        """
        )

        # Resource utilization metrics
        cursor.execute(
            """
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
        """
        )

        # Selection and reproduction outcomes
        cursor.execute(
            """
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
        """
        )

        # Create performance indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_evolution_rounds_number ON evolution_rounds(round_number)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fitness_agent ON fitness_metrics(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fitness_score ON fitness_metrics(fitness_score)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_resource_timestamp ON resource_metrics(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_selection_parent ON selection_outcomes(parent_agent_id)"
        )

        # Schema version tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """
        )

        cursor.execute(
            """
            INSERT OR IGNORE INTO schema_version (version, description)
            VALUES (1, 'Initial CODEX-compliant schema')
        """
        )

        conn.commit()
        conn.close()

        logger.info(f"Database schema created at {self.db_path}")

    def find_legacy_files(self) -> list[Path]:
        """Find legacy JSON files to migrate."""
        logger.info("Scanning for legacy evolution metrics files...")

        legacy_files = []

        # Search patterns for legacy files
        search_patterns = [
            "evolution_metrics*.json",
            "*evolution*.json",
            "metrics*.json",
            "agent_metrics*.json",
        ]

        # Search in multiple directories
        search_dirs = [
            Path(),
            Path("./data"),
            Path("./logs"),
            self.log_dir,
            Path("./evolution_logs"),
            Path("./metrics"),
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in search_patterns:
                    for file_path in search_dir.glob(pattern):
                        if file_path.is_file():
                            legacy_files.append(file_path)
                            logger.info(f"Found legacy file: {file_path}")

        # Remove duplicates
        legacy_files = list(set(legacy_files))

        logger.info(f"Found {len(legacy_files)} legacy files to migrate")
        return legacy_files

    def validate_json_file(self, file_path: Path) -> bool:
        """Validate JSON file format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check if it looks like evolution metrics
            if isinstance(data, dict):
                # Look for evolution-related keys
                evo_keys = [
                    "rounds",
                    "generations",
                    "fitness",
                    "agents",
                    "metrics",
                    "evolution",
                ]
                if any(key in str(data).lower() for key in evo_keys):
                    return True
            elif isinstance(data, list):
                # Check if it's a list of metrics
                if data and isinstance(data[0], dict):
                    first_item = data[0]
                    if any(
                        key in str(first_item).lower()
                        for key in ["fitness", "agent", "round", "generation"]
                    ):
                        return True

            return False

        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Cannot validate {file_path}: {e}")
            return False

    def migrate_json_file(self, file_path: Path) -> dict[str, Any]:
        """Migrate a single JSON file to database."""
        logger.info(f"Migrating {file_path}...")

        migration_stats = {
            "file": str(file_path),
            "rounds_migrated": 0,
            "agents_migrated": 0,
            "metrics_migrated": 0,
            "errors": [],
        }

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Parse different JSON formats
            if isinstance(data, dict):
                self._migrate_dict_format(cursor, data, migration_stats)
            elif isinstance(data, list):
                self._migrate_list_format(cursor, data, migration_stats)

            conn.commit()
            conn.close()

        except Exception as e:
            error_msg = f"Error migrating {file_path}: {e}"
            logger.exception(error_msg)
            migration_stats["errors"].append(error_msg)

        return migration_stats

    def _migrate_dict_format(
        self, cursor: sqlite3.Cursor, data: dict, stats: dict[str, Any]
    ) -> None:
        """Migrate dictionary format data."""
        # Handle nested evolution data
        if "evolution_rounds" in data:
            rounds_data = data["evolution_rounds"]
        elif "rounds" in data:
            rounds_data = data["rounds"]
        elif "generations" in data:
            rounds_data = data["generations"]
        else:
            # Treat the whole dict as a single round
            rounds_data = [data]

        if not isinstance(rounds_data, list):
            rounds_data = [rounds_data]

        for round_data in rounds_data:
            self._migrate_round(cursor, round_data, stats)

    def _migrate_list_format(
        self, cursor: sqlite3.Cursor, data: list, stats: dict[str, Any]
    ) -> None:
        """Migrate list format data."""
        for item in data:
            if isinstance(item, dict):
                self._migrate_round(cursor, item, stats)

    def _migrate_round(
        self, cursor: sqlite3.Cursor, round_data: dict, stats: dict[str, Any]
    ) -> None:
        """Migrate a single evolution round."""
        try:
            # Extract round information
            round_number = round_data.get("round", round_data.get("round_number", 0))
            generation = round_data.get("generation", round_data.get("gen", 0))
            timestamp = round_data.get(
                "timestamp", datetime.now(UTC).isoformat()
            )

            # Parse timestamp if it's a string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now(UTC)

            # Insert evolution round
            cursor.execute(
                """
                INSERT OR IGNORE INTO evolution_rounds
                (round_number, generation, timestamp, population_size, mutation_rate, selection_pressure, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    round_number,
                    generation,
                    timestamp,
                    round_data.get("population_size"),
                    round_data.get("mutation_rate"),
                    round_data.get("selection_pressure"),
                    round_data.get("status", "completed"),
                    json.dumps(
                        {
                            k: v
                            for k, v in round_data.items()
                            if k
                            not in ["agents", "fitness_metrics", "resource_metrics"]
                        }
                    ),
                ),
            )

            # Get round ID
            cursor.execute(
                "SELECT id FROM evolution_rounds WHERE round_number = ? AND generation = ?",
                (round_number, generation),
            )
            round_id = cursor.fetchone()[0]

            stats["rounds_migrated"] += 1

            # Migrate agent fitness metrics
            agents_data = round_data.get(
                "agents", round_data.get("fitness_metrics", [])
            )
            if isinstance(agents_data, dict):
                # Convert dict to list format
                agents_data = [{"agent_id": k, **v} for k, v in agents_data.items()]

            for agent_data in agents_data:
                if isinstance(agent_data, dict):
                    self._migrate_agent_fitness(cursor, round_id, agent_data, stats)

            # Migrate resource metrics
            resource_data = round_data.get(
                "resource_metrics", round_data.get("resources", {})
            )
            if resource_data:
                self._migrate_resource_metrics(cursor, round_id, resource_data, stats)

            # Migrate selection outcomes
            selection_data = round_data.get(
                "selection_outcomes", round_data.get("selections", [])
            )
            for selection in selection_data:
                if isinstance(selection, dict):
                    self._migrate_selection_outcome(cursor, round_id, selection, stats)

        except Exception as e:
            error_msg = f"Error migrating round {round_data}: {e}"
            logger.warning(error_msg)
            stats["errors"].append(error_msg)

    def _migrate_agent_fitness(
        self,
        cursor: sqlite3.Cursor,
        round_id: int,
        agent_data: dict,
        stats: dict[str, Any],
    ) -> None:
        """Migrate agent fitness data."""
        try:
            agent_id = agent_data.get(
                "agent_id", agent_data.get("id", f"agent_{stats['agents_migrated']}")
            )
            fitness_score = agent_data.get(
                "fitness", agent_data.get("fitness_score", 0.0)
            )

            performance_metrics = {
                k: v
                for k, v in agent_data.items()
                if k not in ["agent_id", "id", "fitness", "fitness_score", "resources"]
            }

            resource_usage = agent_data.get(
                "resources", agent_data.get("resource_usage", {})
            )

            cursor.execute(
                """
                INSERT INTO fitness_metrics
                (round_id, agent_id, fitness_score, performance_metrics, resource_usage)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    round_id,
                    agent_id,
                    float(fitness_score),
                    json.dumps(performance_metrics),
                    json.dumps(resource_usage),
                ),
            )

            stats["agents_migrated"] += 1

        except Exception as e:
            stats["errors"].append(f"Error migrating agent {agent_data}: {e}")

    def _migrate_resource_metrics(
        self,
        cursor: sqlite3.Cursor,
        round_id: int,
        resource_data: dict,
        stats: dict[str, Any],
    ) -> None:
        """Migrate resource utilization metrics."""
        try:
            cursor.execute(
                """
                INSERT INTO resource_metrics
                (round_id, cpu_usage, memory_usage_mb, network_io_kb, disk_io_kb, gpu_usage)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    round_id,
                    resource_data.get("cpu_usage", resource_data.get("cpu")),
                    resource_data.get("memory_usage_mb", resource_data.get("memory")),
                    resource_data.get("network_io_kb", resource_data.get("network")),
                    resource_data.get("disk_io_kb", resource_data.get("disk")),
                    resource_data.get("gpu_usage", resource_data.get("gpu")),
                ),
            )

            stats["metrics_migrated"] += 1

        except Exception as e:
            stats["errors"].append(f"Error migrating resources {resource_data}: {e}")

    def _migrate_selection_outcome(
        self,
        cursor: sqlite3.Cursor,
        round_id: int,
        selection_data: dict,
        stats: dict[str, Any],
    ) -> None:
        """Migrate selection outcome data."""
        try:
            cursor.execute(
                """
                INSERT INTO selection_outcomes
                (round_id, parent_agent_id, child_agent_id, selection_method, crossover_points, mutation_applied, survival_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    round_id,
                    selection_data.get("parent_agent_id", selection_data.get("parent")),
                    selection_data.get("child_agent_id", selection_data.get("child")),
                    selection_data.get(
                        "selection_method", selection_data.get("method")
                    ),
                    json.dumps(selection_data.get("crossover_points", [])),
                    bool(selection_data.get("mutation_applied", False)),
                    selection_data.get("survival_reason", selection_data.get("reason")),
                ),
            )

        except Exception as e:
            stats["errors"].append(f"Error migrating selection {selection_data}: {e}")

    def archive_legacy_files(self, migrated_files: list[Path]) -> None:
        """Archive migrated JSON files."""
        logger.info("Archiving legacy files...")

        archive_dir = Path("./data/archive/legacy_evolution_metrics")
        archive_dir.mkdir(parents=True, exist_ok=True)

        for file_path in migrated_files:
            try:
                # Create archive filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                archive_path = archive_dir / archive_name

                # Copy to archive
                shutil.copy2(file_path, archive_path)
                logger.info(f"Archived {file_path} -> {archive_path}")

                # Create backup info
                info_file = archive_path.with_suffix(archive_path.suffix + ".info")
                with open(info_file, "w") as f:
                    json.dump(
                        {
                            "original_path": str(file_path),
                            "archived_at": datetime.now(UTC).isoformat(),
                            "file_size": file_path.stat().st_size,
                            "file_hash": hashlib.md5(
                                file_path.read_bytes()
                            ).hexdigest(),
                        },
                        f,
                        indent=2,
                    )

            except Exception as e:
                logger.exception(f"Error archiving {file_path}: {e}")

    def validate_migration(self) -> dict[str, Any]:
        """Validate migration integrity."""
        logger.info("Validating migration integrity...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        validation_results = {
            "database_exists": self.db_path.exists(),
            "tables_created": [],
            "data_integrity": {},
            "indexes_created": [],
            "foreign_keys_valid": True,
        }

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            "evolution_rounds",
            "fitness_metrics",
            "resource_metrics",
            "selection_outcomes",
            "schema_version",
        ]
        for table in expected_tables:
            if table in tables:
                validation_results["tables_created"].append(table)
            else:
                logger.error(f"Missing table: {table}")

        # Check data integrity
        for table in [
            "evolution_rounds",
            "fitness_metrics",
            "resource_metrics",
            "selection_outcomes",
        ]:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                validation_results["data_integrity"][table] = count
                logger.info(f"{table}: {count} records")

        # Check indexes
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        validation_results["indexes_created"] = indexes

        # Check foreign key constraints
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()
        if fk_violations:
            validation_results["foreign_keys_valid"] = False
            logger.error(f"Foreign key violations: {fk_violations}")

        conn.close()

        return validation_results

    def run_migration(self) -> dict[str, Any]:
        """Execute complete migration process."""
        logger.info("Starting evolution metrics migration...")

        start_time = datetime.now()

        # Create database schema
        self.create_database_schema()

        # Find legacy files
        legacy_files = self.find_legacy_files()

        if not legacy_files:
            logger.info("No legacy files found - creating empty database")
            return {
                "status": "completed",
                "legacy_files_found": 0,
                "files_migrated": 0,
                "total_rounds": 0,
                "total_agents": 0,
                "validation": self.validate_migration(),
                "duration": (datetime.now() - start_time).total_seconds(),
            }

        # Migrate each file
        migration_results = []
        total_rounds = 0
        total_agents = 0

        for file_path in legacy_files:
            if self.validate_json_file(file_path):
                result = self.migrate_json_file(file_path)
                migration_results.append(result)
                total_rounds += result["rounds_migrated"]
                total_agents += result["agents_migrated"]
            else:
                logger.warning(f"Skipping {file_path} - not evolution metrics format")

        # Archive legacy files
        migrated_files = [
            Path(r["file"]) for r in migration_results if r["rounds_migrated"] > 0
        ]
        if migrated_files:
            self.archive_legacy_files(migrated_files)

        # Validate migration
        validation = self.validate_migration()

        # Generate final report
        report = {
            "status": "completed",
            "legacy_files_found": len(legacy_files),
            "files_migrated": len(migration_results),
            "total_rounds": total_rounds,
            "total_agents": total_agents,
            "migration_results": migration_results,
            "validation": validation,
            "duration": (datetime.now() - start_time).total_seconds(),
        }

        logger.info(
            f"Migration completed: {total_rounds} rounds, {total_agents} agents migrated"
        )

        return report


def main() -> None:
    """Main migration function."""
    migrator = EvolutionMetricsMigrator()
    report = migrator.run_migration()

    # Save migration report
    report_path = Path("./data/evolution_metrics_migration_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 50}")
    print("EVOLUTION METRICS MIGRATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Status: {report['status']}")
    print(f"Files found: {report['legacy_files_found']}")
    print(f"Files migrated: {report['files_migrated']}")
    print(f"Rounds migrated: {report['total_rounds']}")
    print(f"Agents migrated: {report['total_agents']}")
    print(f"Duration: {report['duration']:.2f} seconds")
    print(f"Database: {AIVILLAGE_DB_PATH}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
