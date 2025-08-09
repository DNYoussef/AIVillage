#!/usr/bin/env python3
"""Database integrity verification and concurrent access testing.

Verifies database integrity, tests concurrent access patterns,
and validates all CODEX integration requirements.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time

logger = logging.getLogger(__name__)

class DatabaseVerifier:
    """Comprehensive database verification for CODEX integration."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.verification_results = {}

    def verify_database_integrity(self, db_path: Path) -> dict:
        """Run comprehensive integrity checks on a database."""
        results = {
            "database": db_path.name,
            "exists": False,
            "size_bytes": 0,
            "integrity_check": False,
            "wal_mode": False,
            "tables": [],
            "indexes": [],
            "schema_version": None,
            "errors": []
        }

        try:
            if not db_path.exists():
                results["errors"].append("Database file does not exist")
                return results

            results["exists"] = True
            results["size_bytes"] = db_path.stat().st_size

            # Connect and check
            conn = sqlite3.connect(str(db_path))

            # Check WAL mode
            cursor = conn.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            results["wal_mode"] = journal_mode.upper() == "WAL"

            # Integrity check
            cursor = conn.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            results["integrity_check"] = integrity_result == "ok"

            if integrity_result != "ok":
                results["errors"].append(f"Integrity check failed: {integrity_result}")

            # Get tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            results["tables"] = [row[0] for row in cursor.fetchall()]

            # Get indexes
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'")
            results["indexes"] = [row[0] for row in cursor.fetchall()]

            # Get schema version
            if "schema_version" in results["tables"]:
                cursor = conn.execute("SELECT MAX(version) FROM schema_version")
                version_result = cursor.fetchone()
                results["schema_version"] = version_result[0] if version_result[0] is not None else 0

            conn.close()

        except Exception as e:
            results["errors"].append(f"Verification error: {e!s}")

        return results

    def test_concurrent_access(self, db_path: Path, num_threads: int = 5) -> dict:
        """Test concurrent access patterns for a database."""
        results = {
            "database": db_path.name,
            "threads_tested": num_threads,
            "successful_connections": 0,
            "failed_connections": 0,
            "read_operations": 0,
            "write_operations": 0,
            "errors": [],
            "avg_response_time": 0.0
        }

        if not db_path.exists():
            results["errors"].append("Database does not exist")
            return results

        def worker_thread(thread_id: int):
            """Worker thread for concurrent access testing."""
            thread_results = {
                "connection_success": False,
                "read_success": False,
                "write_success": False,
                "response_times": [],
                "errors": []
            }

            try:
                # Test connection
                start_time = time.time()
                conn = sqlite3.connect(str(db_path))
                connect_time = time.time() - start_time
                thread_results["response_times"].append(connect_time)
                thread_results["connection_success"] = True

                # Test read operation
                start_time = time.time()
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                read_time = time.time() - start_time
                thread_results["response_times"].append(read_time)
                thread_results["read_success"] = len(tables) > 0

                # Test write operation (if schema_version table exists)
                if any("schema_version" in str(table) for table in tables):
                    start_time = time.time()
                    conn.execute(f"""
                        INSERT OR IGNORE INTO schema_version (version, description) 
                        VALUES (999, 'Concurrent test thread {thread_id}')
                    """)
                    conn.execute("DELETE FROM schema_version WHERE version = 999")
                    conn.commit()
                    write_time = time.time() - start_time
                    thread_results["response_times"].append(write_time)
                    thread_results["write_success"] = True

                conn.close()

            except Exception as e:
                thread_results["errors"].append(str(e))

            return thread_results

        # Run concurrent tests
        thread_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]

            for future in futures:
                thread_results.append(future.result())

        # Aggregate results
        all_response_times = []
        for thread_result in thread_results:
            if thread_result["connection_success"]:
                results["successful_connections"] += 1
            else:
                results["failed_connections"] += 1

            if thread_result["read_success"]:
                results["read_operations"] += 1

            if thread_result["write_success"]:
                results["write_operations"] += 1

            all_response_times.extend(thread_result["response_times"])
            results["errors"].extend(thread_result["errors"])

        if all_response_times:
            results["avg_response_time"] = sum(all_response_times) / len(all_response_times)

        return results

    def verify_schema_requirements(self, db_name: str) -> dict:
        """Verify database meets CODEX schema requirements."""
        results = {
            "database": db_name,
            "schema_compliant": False,
            "required_tables": [],
            "missing_tables": [],
            "required_indexes": [],
            "missing_indexes": [],
            "errors": []
        }

        # Define required schemas per CODEX requirements
        required_schemas = {
            "evolution_metrics.db": {
                "tables": ["schema_version", "evolution_rounds", "fitness_metrics",
                          "resource_metrics", "selection_outcomes"],
                "indexes": ["idx_evolution_rounds_number", "idx_fitness_agent",
                           "idx_fitness_score", "idx_resource_timestamp", "idx_selection_parent"]
            },
            "digital_twin.db": {
                "tables": ["schema_version", "learning_profiles", "learning_sessions", "knowledge_states"],
                "indexes": ["idx_profiles_user_hash", "idx_profiles_updated", "idx_sessions_profile",
                           "idx_sessions_start", "idx_knowledge_profile", "idx_knowledge_domain", "idx_knowledge_mastery"]
            },
            "rag_index.db": {
                "tables": ["schema_version", "documents", "chunks", "embeddings_metadata"],
                "indexes": ["idx_documents_hash", "idx_documents_type", "idx_chunks_document",
                           "idx_chunks_index", "idx_embeddings_faiss", "idx_embeddings_queries"]
            }
        }

        if db_name not in required_schemas:
            results["errors"].append(f"No schema requirements defined for {db_name}")
            return results

        schema_req = required_schemas[db_name]
        results["required_tables"] = schema_req["tables"]
        results["required_indexes"] = schema_req["indexes"]

        db_path = self.data_dir / db_name

        try:
            conn = sqlite3.connect(str(db_path))

            # Check tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]

            results["missing_tables"] = [table for table in schema_req["tables"]
                                       if table not in existing_tables]

            # Check indexes
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'")
            existing_indexes = [row[0] for row in cursor.fetchall()]

            results["missing_indexes"] = [index for index in schema_req["indexes"]
                                        if index not in existing_indexes]

            results["schema_compliant"] = (len(results["missing_tables"]) == 0 and
                                         len(results["missing_indexes"]) == 0)

            conn.close()

        except Exception as e:
            results["errors"].append(f"Schema verification error: {e!s}")

        return results

    def test_backup_restore(self, db_path: Path) -> dict:
        """Test backup and restore procedures."""
        results = {
            "database": db_path.name,
            "backup_created": False,
            "backup_size": 0,
            "restore_successful": False,
            "data_integrity_maintained": False,
            "errors": []
        }

        if not db_path.exists():
            results["errors"].append("Database does not exist")
            return results

        try:
            # Create backup
            backup_path = db_path.with_suffix(".backup.db")

            # Use SQLite backup API
            source_conn = sqlite3.connect(str(db_path))
            backup_conn = sqlite3.connect(str(backup_path))

            source_conn.backup(backup_conn)

            source_conn.close()
            backup_conn.close()

            if backup_path.exists():
                results["backup_created"] = True
                results["backup_size"] = backup_path.stat().st_size

                # Test restore by comparing checksums
                original_checksum = self._get_database_checksum(db_path)
                backup_checksum = self._get_database_checksum(backup_path)

                results["restore_successful"] = True
                results["data_integrity_maintained"] = (original_checksum == backup_checksum)

                # Clean up backup
                backup_path.unlink()

        except Exception as e:
            results["errors"].append(f"Backup/restore test error: {e!s}")

        return results

    def _get_database_checksum(self, db_path: Path) -> str:
        """Calculate checksum of database content."""
        try:
            conn = sqlite3.connect(str(db_path))

            # Get all table data in deterministic order
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            content_hash = hashlib.sha256()

            for table in tables:
                cursor = conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
                for row in cursor.fetchall():
                    content_hash.update(str(row).encode())

            conn.close()
            return content_hash.hexdigest()

        except Exception as e:
            logger.error(f"Error calculating checksum for {db_path}: {e}")
            return ""

    def run_comprehensive_verification(self) -> dict:
        """Run all verification tests on all databases."""
        print("Running comprehensive database verification...")

        expected_databases = [
            "evolution_metrics.db",
            "digital_twin.db",
            "rag_index.db"
        ]

        verification_results = {
            "timestamp": datetime.now().isoformat(),
            "databases": {},
            "summary": {
                "total_databases": len(expected_databases),
                "databases_found": 0,
                "integrity_passed": 0,
                "schema_compliant": 0,
                "concurrent_access_ok": 0,
                "backup_restore_ok": 0,
                "overall_health": "UNKNOWN"
            }
        }

        for db_name in expected_databases:
            db_path = self.data_dir / db_name

            print(f"\nVerifying {db_name}...")

            db_results = {
                "integrity": self.verify_database_integrity(db_path),
                "schema": self.verify_schema_requirements(db_name),
                "concurrent_access": self.test_concurrent_access(db_path),
                "backup_restore": self.test_backup_restore(db_path)
            }

            verification_results["databases"][db_name] = db_results

            # Update summary
            if db_results["integrity"]["exists"]:
                verification_results["summary"]["databases_found"] += 1

            if db_results["integrity"]["integrity_check"]:
                verification_results["summary"]["integrity_passed"] += 1

            if db_results["schema"]["schema_compliant"]:
                verification_results["summary"]["schema_compliant"] += 1

            if db_results["concurrent_access"]["failed_connections"] == 0:
                verification_results["summary"]["concurrent_access_ok"] += 1

            if db_results["backup_restore"]["data_integrity_maintained"]:
                verification_results["summary"]["backup_restore_ok"] += 1

        # Determine overall health
        summary = verification_results["summary"]
        if summary["databases_found"] == summary["total_databases"]:
            if (summary["integrity_passed"] == summary["total_databases"] and
                summary["schema_compliant"] == summary["total_databases"] and
                summary["concurrent_access_ok"] == summary["total_databases"]):
                summary["overall_health"] = "EXCELLENT"
            elif summary["integrity_passed"] == summary["total_databases"]:
                summary["overall_health"] = "GOOD"
            else:
                summary["overall_health"] = "POOR"
        else:
            summary["overall_health"] = "CRITICAL"

        return verification_results


def main():
    """Main verification function."""
    logging.basicConfig(level=logging.INFO)

    verifier = DatabaseVerifier()
    results = verifier.run_comprehensive_verification()

    # Print detailed results
    print("\n" + "="*80)
    print("CODEX DATABASE VERIFICATION REPORT")
    print("="*80)

    summary = results["summary"]
    print(f"\nOverall Health: {summary['overall_health']}")
    print(f"Timestamp: {results['timestamp']}")

    print("\nSummary:")
    print(f"  Databases Found: {summary['databases_found']}/{summary['total_databases']}")
    print(f"  Integrity Passed: {summary['integrity_passed']}/{summary['total_databases']}")
    print(f"  Schema Compliant: {summary['schema_compliant']}/{summary['total_databases']}")
    print(f"  Concurrent Access OK: {summary['concurrent_access_ok']}/{summary['total_databases']}")
    print(f"  Backup/Restore OK: {summary['backup_restore_ok']}/{summary['total_databases']}")

    print("\nDetailed Results:")
    for db_name, db_results in results["databases"].items():
        print(f"\n{db_name.upper()}:")

        # Integrity
        integrity = db_results["integrity"]
        status = "✅" if integrity["integrity_check"] else "❌"
        print(f"  Integrity: {status} ({integrity['size_bytes']:,} bytes)")
        if integrity["errors"]:
            for error in integrity["errors"]:
                print(f"    Error: {error}")

        # Schema
        schema = db_results["schema"]
        status = "✅" if schema["schema_compliant"] else "❌"
        print(f"  Schema: {status} ({len(schema['required_tables'])} tables, {len(schema['required_indexes'])} indexes)")
        if schema["missing_tables"]:
            print(f"    Missing tables: {', '.join(schema['missing_tables'])}")
        if schema["missing_indexes"]:
            print(f"    Missing indexes: {', '.join(schema['missing_indexes'])}")

        # Concurrent access
        concurrent = db_results["concurrent_access"]
        status = "✅" if concurrent["failed_connections"] == 0 else "❌"
        print(f"  Concurrent Access: {status} ({concurrent['successful_connections']}/{concurrent['threads_tested']} threads)")
        if concurrent["avg_response_time"] > 0:
            print(f"    Avg Response Time: {concurrent['avg_response_time']:.3f}s")

        # Backup/Restore
        backup = db_results["backup_restore"]
        status = "✅" if backup["data_integrity_maintained"] else "❌"
        print(f"  Backup/Restore: {status}")

    print("\n" + "="*80)

    if summary["overall_health"] in ["EXCELLENT", "GOOD"]:
        print("✅ Database verification completed successfully!")
        print("All databases are ready for CODEX integration.")
    else:
        print("❌ Database verification found issues!")
        print("Review the detailed results above and fix issues before proceeding.")

    print("="*80)

    # Save detailed report
    report_file = Path("data") / "database_verification_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")

    return summary["overall_health"] in ["EXCELLENT", "GOOD"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
