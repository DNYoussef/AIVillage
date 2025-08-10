"""Legacy data migration script for AIVillage CODEX integration.

This script migrates data from various legacy formats to the new SQLite databases:
- JSON evolution metrics → evolution_metrics.db
- Profile directories → digital_twin.db
- File-based document storage → rag_index.db
- Configuration files → unified configuration system
"""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config.configuration_manager import get_config
from core.database.database_manager import initialize_databases
from core.database.migrations import DataMigrator, get_migration_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LegacyDataMigrator:
    """Comprehensive legacy data migration system."""

    def __init__(self, database_manager, config_manager=None):
        self.database_manager = database_manager
        self.config_manager = config_manager
        self.data_migrator = DataMigrator(database_manager)
        self.migration_stats = {
            "evolution_metrics": {"processed": 0, "migrated": 0, "errors": 0},
            "digital_twin": {"processed": 0, "migrated": 0, "errors": 0},
            "rag_index": {"processed": 0, "migrated": 0, "errors": 0},
            "configuration": {"processed": 0, "migrated": 0, "errors": 0},
        }

    async def migrate_all_legacy_data(self) -> dict[str, Any]:
        """Run complete legacy data migration."""
        logger.info("Starting comprehensive legacy data migration")
        start_time = time.time()

        # Migration phases
        migration_tasks = [
            ("Evolution Metrics", self._migrate_evolution_metrics),
            ("Digital Twin Data", self._migrate_digital_twin_data),
            ("RAG Document Data", self._migrate_rag_documents),
            ("Configuration Files", self._migrate_configuration_data),
        ]

        results = {}

        for phase_name, migration_func in migration_tasks:
            logger.info(f"Starting migration phase: {phase_name}")
            phase_start = time.time()

            try:
                phase_result = await migration_func()
                phase_duration = time.time() - phase_start

                results[phase_name.lower().replace(" ", "_")] = {
                    "success": True,
                    "duration_seconds": phase_duration,
                    "details": phase_result,
                }

                logger.info(f"Completed {phase_name} in {phase_duration:.2f}s")

            except Exception as e:
                phase_duration = time.time() - phase_start
                logger.error(f"Failed {phase_name}: {e}")

                results[phase_name.lower().replace(" ", "_")] = {
                    "success": False,
                    "duration_seconds": phase_duration,
                    "error": str(e),
                }

        total_duration = time.time() - start_time

        # Generate migration summary
        summary = {
            "migration_timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "phases_completed": len(results),
            "phases_successful": sum(1 for r in results.values() if r["success"]),
            "migration_stats": self.migration_stats,
            "phase_results": results,
        }

        logger.info(f"Legacy data migration completed in {total_duration:.2f}s")
        return summary

    async def _migrate_evolution_metrics(self) -> dict[str, Any]:
        """Migrate evolution metrics from various legacy formats."""
        results = {
            "json_files_processed": 0,
            "csv_files_processed": 0,
            "records_migrated": 0,
            "errors": [],
        }

        # Common legacy file locations
        search_paths = [
            "./evolution_metrics.json",
            "./data/evolution_metrics.json",
            "./logs/evolution_metrics.json",
            "./evolution_logs/metrics.json",
            "./data/metrics/evolution_*.json",
            "./evolution_history.json",
        ]

        # Search for JSON files
        json_files = []
        for pattern in search_paths:
            if "*" in pattern:
                # Glob pattern
                for path in Path().glob(pattern):
                    if path.is_file():
                        json_files.append(path)
            else:
                # Direct path
                path = Path(pattern)
                if path.exists():
                    json_files.append(path)

        logger.info(f"Found {len(json_files)} potential evolution metrics files")

        # Migrate each JSON file
        for json_file in json_files:
            try:
                logger.info(f"Processing evolution metrics file: {json_file}")

                success = await self.data_migrator.migrate_evolution_metrics_from_json(
                    str(json_file)
                )

                if success:
                    results["json_files_processed"] += 1
                    self.migration_stats["evolution_metrics"]["processed"] += 1
                    self.migration_stats["evolution_metrics"]["migrated"] += 1

                    # Count records in file
                    try:
                        with open(json_file) as f:
                            data = json.load(f)

                        record_count = 0
                        if "rounds" in data:
                            record_count = len(data["rounds"])
                        elif "metrics" in data:
                            record_count = len(data["metrics"])
                        elif isinstance(data, list):
                            record_count = len(data)

                        results["records_migrated"] += record_count
                        logger.info(f"Migrated {record_count} records from {json_file}")

                    except Exception as count_error:
                        logger.warning(
                            f"Could not count records in {json_file}: {count_error}"
                        )

                else:
                    results["errors"].append(f"Failed to migrate {json_file}")
                    self.migration_stats["evolution_metrics"]["errors"] += 1

            except Exception as e:
                error_msg = f"Error processing {json_file}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                self.migration_stats["evolution_metrics"]["errors"] += 1

        # Look for CSV files (alternative format)
        csv_patterns = [
            "./data/evolution_metrics.csv",
            "./logs/evolution_*.csv",
            "./data/metrics/*.csv",
        ]

        csv_files = []
        for pattern in csv_patterns:
            if "*" in pattern:
                for path in Path().glob(pattern):
                    if path.is_file():
                        csv_files.append(path)
            else:
                path = Path(pattern)
                if path.exists():
                    csv_files.append(path)

        # Migrate CSV files
        for csv_file in csv_files:
            try:
                records_migrated = await self._migrate_csv_evolution_metrics(csv_file)
                results["csv_files_processed"] += 1
                results["records_migrated"] += records_migrated
                self.migration_stats["evolution_metrics"]["migrated"] += 1

                logger.info(f"Migrated {records_migrated} records from CSV: {csv_file}")

            except Exception as e:
                error_msg = f"Error processing CSV {csv_file}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                self.migration_stats["evolution_metrics"]["errors"] += 1

        return results

    async def _migrate_csv_evolution_metrics(self, csv_file: Path) -> int:
        """Migrate evolution metrics from CSV format."""
        records_migrated = 0

        try:
            import csv

            with open(csv_file) as f:
                reader = csv.DictReader(f)

                # Convert CSV data to JSON format for migration
                rounds_data = []
                current_round = None

                for row in reader:
                    # Detect if this is a new round
                    if row.get("round_id") != getattr(current_round, "id", None):
                        if current_round:
                            rounds_data.append(current_round)

                        current_round = {
                            "id": row.get("round_id", f"csv_round_{len(rounds_data)}"),
                            "start_time": float(row.get("timestamp", time.time())),
                            "status": row.get("status", "completed"),
                            "agent_count": int(row.get("agent_count", 0)),
                            "fitness_metrics": [],
                        }

                    # Add fitness metric
                    if row.get("fitness_score"):
                        current_round["fitness_metrics"].append(
                            {
                                "agent_id": row.get("agent_id", "unknown"),
                                "evolution_id": row.get("evolution_id", "unknown"),
                                "fitness_score": float(row.get("fitness_score", 0.0)),
                                "timestamp": float(row.get("timestamp", time.time())),
                            }
                        )

                # Add last round
                if current_round:
                    rounds_data.append(current_round)

                # Create temporary JSON file
                temp_json = {"rounds": rounds_data}
                temp_file = Path(f"/tmp/temp_evolution_{int(time.time())}.json")

                with open(temp_file, "w") as f:
                    json.dump(temp_json, f)

                # Migrate using existing JSON migrator
                success = await self.data_migrator.migrate_evolution_metrics_from_json(
                    str(temp_file)
                )

                # Cleanup temp file
                if temp_file.exists():
                    temp_file.unlink()

                if success:
                    records_migrated = len(rounds_data)

        except ImportError:
            logger.warning("CSV module not available, skipping CSV migration")
        except Exception as e:
            logger.error(f"CSV migration failed: {e}")

        return records_migrated

    async def _migrate_digital_twin_data(self) -> dict[str, Any]:
        """Migrate digital twin data from various legacy formats."""
        results = {
            "profile_directories_processed": 0,
            "profile_files_processed": 0,
            "profiles_migrated": 0,
            "sessions_migrated": 0,
            "errors": [],
        }

        # Common profile storage locations
        profile_paths = [
            "./profiles",
            "./data/profiles",
            "./student_profiles",
            "./digital_twin/profiles",
            "./data/digital_twin/profiles",
        ]

        for profile_dir in profile_paths:
            profile_path = Path(profile_dir)
            if not profile_path.exists():
                continue

            logger.info(f"Processing profile directory: {profile_dir}")

            try:
                success = await self.data_migrator.migrate_digital_twin_profiles(
                    str(profile_path)
                )

                if success:
                    results["profile_directories_processed"] += 1
                    self.migration_stats["digital_twin"]["processed"] += 1

                    # Count migrated files
                    json_files = list(profile_path.glob("*.json"))
                    results["profile_files_processed"] += len(json_files)

                    # Estimate migrated profiles and sessions
                    for json_file in json_files:
                        try:
                            with open(json_file) as f:
                                data = json.load(f)

                            results["profiles_migrated"] += 1

                            # Count sessions if present
                            if "sessions" in data:
                                results["sessions_migrated"] += len(data["sessions"])

                        except Exception as e:
                            logger.warning(f"Could not parse {json_file}: {e}")

                    self.migration_stats["digital_twin"]["migrated"] += 1
                    logger.info(
                        f"Migrated {len(json_files)} profiles from {profile_dir}"
                    )

                else:
                    results["errors"].append(
                        f"Failed to migrate profiles from {profile_dir}"
                    )
                    self.migration_stats["digital_twin"]["errors"] += 1

            except Exception as e:
                error_msg = f"Error processing profile directory {profile_dir}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                self.migration_stats["digital_twin"]["errors"] += 1

        # Look for individual profile files
        individual_profile_patterns = [
            "./profile_*.json",
            "./data/profile_*.json",
            "./student_*.json",
        ]

        for pattern in individual_profile_patterns:
            for profile_file in Path().glob(pattern):
                try:
                    # Create temporary directory and copy file
                    temp_dir = Path(f"/tmp/temp_profiles_{int(time.time())}")
                    temp_dir.mkdir(exist_ok=True)

                    import shutil

                    temp_file = temp_dir / profile_file.name
                    shutil.copy2(profile_file, temp_file)

                    # Migrate
                    success = await self.data_migrator.migrate_digital_twin_profiles(
                        str(temp_dir)
                    )

                    if success:
                        results["profile_files_processed"] += 1
                        results["profiles_migrated"] += 1
                        logger.info(f"Migrated individual profile: {profile_file}")

                    # Cleanup
                    shutil.rmtree(temp_dir)

                except Exception as e:
                    error_msg = (
                        f"Error migrating individual profile {profile_file}: {e!s}"
                    )
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

        return results

    async def _migrate_rag_documents(self) -> dict[str, Any]:
        """Migrate RAG document data from legacy storage."""
        results = {
            "document_directories_processed": 0,
            "documents_migrated": 0,
            "chunks_generated": 0,
            "errors": [],
        }

        # Common document storage locations
        document_paths = [
            "./documents",
            "./data/documents",
            "./rag_documents",
            "./data/rag",
            "./corpus",
            "./data/corpus",
        ]

        for doc_dir in document_paths:
            doc_path = Path(doc_dir)
            if not doc_path.exists():
                continue

            logger.info(f"Processing document directory: {doc_dir}")

            try:
                # Process text files
                text_files = list(doc_path.glob("*.txt")) + list(doc_path.glob("*.md"))

                for text_file in text_files:
                    try:
                        await self._migrate_document_file(text_file)
                        results["documents_migrated"] += 1

                    except Exception as e:
                        error_msg = f"Error migrating document {text_file}: {e!s}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

                if text_files:
                    results["document_directories_processed"] += 1
                    self.migration_stats["rag_index"]["processed"] += 1
                    self.migration_stats["rag_index"]["migrated"] += 1

                logger.info(f"Migrated {len(text_files)} documents from {doc_dir}")

            except Exception as e:
                error_msg = f"Error processing document directory {doc_dir}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                self.migration_stats["rag_index"]["errors"] += 1

        return results

    async def _migrate_document_file(self, file_path: Path):
        """Migrate a single document file to RAG database."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Generate document metadata
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        word_count = len(content.split())

        # Insert document
        with self.database_manager.get_connection("rag_index") as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
            INSERT OR REPLACE INTO documents
            (document_id, title, content_hash, source_type, file_path, word_count)
            VALUES (?, ?, ?, 'text', ?, ?)
            """,
                (
                    str(file_path),
                    file_path.stem,
                    content_hash,
                    str(file_path),
                    word_count,
                ),
            )

            # Generate chunks (simple sentence-based chunking)
            sentences = content.split(". ")
            chunk_size = 3  # 3 sentences per chunk

            for i in range(0, len(sentences), chunk_size):
                chunk_content = ". ".join(sentences[i : i + chunk_size])
                if chunk_content:
                    chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()

                    cursor.execute(
                        """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, document_id, chunk_index, content, content_hash, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            f"{file_path}_{i // chunk_size}",
                            str(file_path),
                            i // chunk_size,
                            chunk_content,
                            chunk_hash,
                            len(chunk_content.split()),
                        ),
                    )

            conn.commit()

    async def _migrate_configuration_data(self) -> dict[str, Any]:
        """Migrate configuration from legacy formats."""
        results = {
            "config_files_processed": 0,
            "env_files_processed": 0,
            "variables_migrated": 0,
            "errors": [],
        }

        # Look for legacy configuration files
        config_patterns = [
            "./.env.local",
            "./.env.dev",
            "./.env.prod",
            "./config.json",
            "./aivillage.conf",
            "./settings.ini",
        ]

        for config_pattern in config_patterns:
            config_path = Path(config_pattern)
            if not config_path.exists():
                continue

            try:
                if config_path.suffix == ".json":
                    await self._migrate_json_config(config_path)
                    results["config_files_processed"] += 1
                elif config_path.name.startswith(".env"):
                    await self._migrate_env_config(config_path)
                    results["env_files_processed"] += 1
                elif config_path.suffix == ".ini":
                    await self._migrate_ini_config(config_path)
                    results["config_files_processed"] += 1

                self.migration_stats["configuration"]["processed"] += 1
                self.migration_stats["configuration"]["migrated"] += 1

                logger.info(f"Processed configuration file: {config_path}")

            except Exception as e:
                error_msg = f"Error migrating config {config_path}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                self.migration_stats["configuration"]["errors"] += 1

        return results

    async def _migrate_json_config(self, config_path: Path):
        """Migrate JSON configuration to new format."""
        with open(config_path) as f:
            config_data = json.load(f)

        # Create migration mapping
        mapping_file = Path("config") / "legacy_config_mapping.json"

        # Flatten nested config
        flat_config = {}
        self._flatten_dict(config_data, flat_config)

        # Save migration record
        migration_record = {
            "source_file": str(config_path),
            "migrated_at": datetime.now().isoformat(),
            "variables": flat_config,
        }

        mapping_file.parent.mkdir(exist_ok=True)
        with open(mapping_file, "w") as f:
            json.dump(migration_record, f, indent=2)

    async def _migrate_env_config(self, env_path: Path):
        """Migrate .env configuration to new format."""
        env_vars = {}

        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        # Create migration record
        migration_file = Path("config") / f"{env_path.stem}_migration.json"
        migration_record = {
            "source_file": str(env_path),
            "migrated_at": datetime.now().isoformat(),
            "variables": env_vars,
        }

        migration_file.parent.mkdir(exist_ok=True)
        with open(migration_file, "w") as f:
            json.dump(migration_record, f, indent=2)

    async def _migrate_ini_config(self, ini_path: Path):
        """Migrate INI configuration to new format."""
        try:
            import configparser

            config = configparser.ConfigParser()
            config.read(ini_path)

            # Convert to flat dictionary
            flat_config = {}
            for section in config.sections():
                for key, value in config[section].items():
                    flat_key = f"{section.upper()}_{key.upper()}"
                    flat_config[flat_key] = value

            # Save migration record
            migration_file = Path("config") / f"{ini_path.stem}_ini_migration.json"
            migration_record = {
                "source_file": str(ini_path),
                "migrated_at": datetime.now().isoformat(),
                "variables": flat_config,
            }

            migration_file.parent.mkdir(exist_ok=True)
            with open(migration_file, "w") as f:
                json.dump(migration_record, f, indent=2)

        except ImportError:
            logger.warning("configparser not available, skipping INI migration")

    def _flatten_dict(
        self, d: dict[str, Any], parent: dict[str, Any], prefix: str = ""
    ):
        """Flatten nested dictionary for configuration migration."""
        for key, value in d.items():
            new_key = f"{prefix}_{key}".upper() if prefix else key.upper()

            if isinstance(value, dict):
                self._flatten_dict(value, parent, new_key)
            else:
                parent[new_key] = value


async def main():
    """Main migration script."""
    print("🔄 AIVillage Legacy Data Migration")
    print("=" * 60)

    try:
        # Initialize configuration and database systems
        config = get_config("development")
        db_manager = await initialize_databases(config)

        # Run database migrations first
        migration_manager = await get_migration_manager(db_manager)

        print("📋 Running schema migrations...")
        for database in ["evolution_metrics", "digital_twin", "rag_index"]:
            success = await migration_manager.migrate_to_latest(database)
            print(f"  {'✅' if success else '❌'} {database}")

        # Run legacy data migration
        print("\n📦 Migrating legacy data...")
        migrator = LegacyDataMigrator(db_manager, config)

        migration_results = await migrator.migrate_all_legacy_data()

        # Print results summary
        print("\n" + "=" * 60)
        print("MIGRATION RESULTS")
        print("=" * 60)

        print(f"Total Duration: {migration_results['total_duration_seconds']:.2f}s")
        print(f"Phases Completed: {migration_results['phases_completed']}")
        print(f"Phases Successful: {migration_results['phases_successful']}")

        print("\nPhase Results:")
        for phase, result in migration_results["phase_results"].items():
            status = "✅" if result["success"] else "❌"
            print(
                f"  {status} {phase.replace('_', ' ').title()}: {result['duration_seconds']:.2f}s"
            )

            if not result["success"]:
                print(f"    Error: {result.get('error', 'Unknown error')}")

        print("\nMigration Statistics:")
        for component, stats in migration_results["migration_stats"].items():
            if stats["processed"] > 0:
                print(
                    f"  {component}: {stats['migrated']}/{stats['processed']} successful, {stats['errors']} errors"
                )

        # Save detailed results
        results_file = Path("migration_results.json")
        with open(results_file, "w") as f:
            json.dump(migration_results, f, indent=2)

        print(f"\n📄 Detailed results saved to: {results_file}")

        # Validate migrated data
        print("\n🔍 Validating migrated data...")
        from core.database.database_validator import validate_aivillage_databases

        validation_results = await validate_aivillage_databases(db_manager)

        print(
            f"Database Health Score: {validation_results['health_score']}/100 ({validation_results['health_status']})"
        )
        print(
            f"Tests: {validation_results['passed_tests']}/{validation_results['total_tests']} passed"
        )

        if validation_results["recommendations"]:
            print("\nRecommendations:")
            for rec in validation_results["recommendations"]:
                print(f"  - {rec}")

        await db_manager.close()

        success = (
            migration_results["phases_successful"]
            == migration_results["phases_completed"]
            and validation_results["health_score"] > 80
        )

        if success:
            print("\n🎉 Legacy data migration completed successfully!")
            print("All databases are ready for AIVillage CODEX integration.")
        else:
            print("\n⚠️ Migration completed with issues.")
            print("Review the results and fix any errors before proceeding.")

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
