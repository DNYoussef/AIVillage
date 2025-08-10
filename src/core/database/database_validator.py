"""Database integrity and performance validation for AIVillage CODEX.

This module provides comprehensive database validation including:
- Schema integrity checks
- Performance benchmarking
- Data consistency validation
- Connection pool health monitoring
- Optimization recommendations
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Database validation result."""

    database: str
    test_name: str
    passed: bool
    value: Any
    expected: Any | None = None
    message: str = ""
    duration_ms: float = 0.0
    severity: str = "info"  # info, warning, error


@dataclass
class PerformanceMetrics:
    """Database performance metrics."""

    database: str
    operation: str
    duration_ms: float
    rows_affected: int = 0
    rows_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_time_ms: float = 0.0


class DatabaseValidator:
    """Comprehensive database validation and performance testing."""

    def __init__(self, database_manager, redis_manager=None):
        self.database_manager = database_manager
        self.redis_manager = redis_manager
        self.results: list[ValidationResult] = []
        self.performance_metrics: list[PerformanceMetrics] = []

    async def validate_all_databases(self) -> dict[str, Any]:
        """Run comprehensive validation on all databases."""
        logger.info("Starting comprehensive database validation")
        self.results.clear()
        self.performance_metrics.clear()

        start_time = time.time()

        # Validate each database
        databases = ["evolution_metrics", "digital_twin", "rag_index"]

        for database in databases:
            await self._validate_database(database)

        # Validate Redis connections if available
        if self.redis_manager:
            await self._validate_redis_connections()

        # Generate validation summary
        total_duration = (time.time() - start_time) * 1000
        summary = self._generate_validation_summary(total_duration)

        logger.info(f"Database validation completed in {total_duration:.2f}ms")
        return summary

    async def _validate_database(self, database: str):
        """Validate a single SQLite database."""
        logger.info(f"Validating database: {database}")

        # Test database connection
        await self._test_database_connection(database)

        # Validate schema integrity
        await self._validate_schema_integrity(database)

        # Test basic operations
        await self._test_basic_operations(database)

        # Performance benchmarks
        await self._benchmark_database_performance(database)

        # Check database size and optimization
        await self._analyze_database_optimization(database)

        # Validate data consistency
        await self._validate_data_consistency(database)

    async def _test_database_connection(self, database: str):
        """Test database connection and basic queries."""
        start_time = time.time()

        try:
            with self.database_manager.get_connection(database) as conn:
                cursor = conn.cursor()

                # Test basic query
                cursor.execute("SELECT sqlite_version()")
                version = cursor.fetchone()[0]

                # Test write capability
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]

                duration_ms = (time.time() - start_time) * 1000

                self.results.append(
                    ValidationResult(
                        database=database,
                        test_name="connection_test",
                        passed=True,
                        value={"sqlite_version": version, "table_count": table_count},
                        message=f"Connection successful, SQLite {version}, {table_count} tables",
                        duration_ms=duration_ms,
                    )
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database=database,
                    test_name="connection_test",
                    passed=False,
                    value=None,
                    message=f"Connection failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="error",
                )
            )

    async def _validate_schema_integrity(self, database: str):
        """Validate database schema integrity."""
        start_time = time.time()

        try:
            with self.database_manager.get_connection(database) as conn:
                cursor = conn.cursor()

                # Check schema version
                cursor.execute(
                    """
                SELECT version, description FROM schema_version
                ORDER BY version DESC LIMIT 1
                """
                )
                version_info = cursor.fetchone()

                if not version_info:
                    self.results.append(
                        ValidationResult(
                            database=database,
                            test_name="schema_version",
                            passed=False,
                            value=None,
                            message="No schema version found",
                            duration_ms=(time.time() - start_time) * 1000,
                            severity="error",
                        )
                    )
                    return

                current_version, description = version_info

                # Validate required tables exist
                expected_tables = self._get_expected_tables(database)
                cursor.execute(
                    """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
                )
                actual_tables = set(row[0] for row in cursor.fetchall())

                missing_tables = expected_tables - actual_tables
                extra_tables = actual_tables - expected_tables

                # Run integrity check
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]

                duration_ms = (time.time() - start_time) * 1000

                passed = len(missing_tables) == 0 and integrity_result == "ok"

                self.results.append(
                    ValidationResult(
                        database=database,
                        test_name="schema_integrity",
                        passed=passed,
                        value={
                            "current_version": current_version,
                            "description": description,
                            "missing_tables": list(missing_tables),
                            "extra_tables": list(extra_tables),
                            "integrity_check": integrity_result,
                        },
                        message=f"Schema v{current_version}, integrity: {integrity_result}",
                        duration_ms=duration_ms,
                        severity="error" if not passed else "info",
                    )
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database=database,
                    test_name="schema_integrity",
                    passed=False,
                    value=None,
                    message=f"Schema validation failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="error",
                )
            )

    def _get_expected_tables(self, database: str) -> set:
        """Get expected tables for each database."""
        table_map = {
            "evolution_metrics": {
                "schema_version",
                "evolution_rounds",
                "fitness_metrics",
                "resource_metrics",
                "selection_outcomes",
            },
            "digital_twin": {
                "schema_version",
                "learning_profiles",
                "learning_sessions",
                "knowledge_states",
            },
            "rag_index": {
                "schema_version",
                "documents",
                "chunks",
                "embeddings_metadata",
                "query_cache",
            },
        }
        return table_map.get(database, set())

    async def _test_basic_operations(self, database: str):
        """Test basic CRUD operations."""
        start_time = time.time()

        try:
            with self.database_manager.get_connection(database) as conn:
                cursor = conn.cursor()

                # Test operations based on database type
                if database == "evolution_metrics":
                    # Test evolution round creation
                    test_round_id = f"test_round_{int(time.time())}"
                    cursor.execute(
                        """
                    INSERT INTO evolution_rounds (start_time, status, agent_count)
                    VALUES (?, 'testing', 1)
                    """,
                        (time.time(),),
                    )

                    round_id = cursor.lastrowid

                    # Test fitness metric creation
                    cursor.execute(
                        """
                    INSERT INTO fitness_metrics
                    (round_id, agent_id, evolution_id, fitness_score, timestamp)
                    VALUES (?, 'test_agent', 'test_evolution', 0.85, ?)
                    """,
                        (round_id, time.time()),
                    )

                    # Test query
                    cursor.execute(
                        """
                    SELECT COUNT(*) FROM fitness_metrics WHERE round_id = ?
                    """,
                        (round_id,),
                    )
                    count = cursor.fetchone()[0]

                    # Cleanup test data
                    cursor.execute(
                        "DELETE FROM fitness_metrics WHERE round_id = ?", (round_id,)
                    )
                    cursor.execute(
                        "DELETE FROM evolution_rounds WHERE id = ?", (round_id,)
                    )

                    conn.commit()

                    passed = count == 1
                    message = f"CRUD operations successful, inserted and queried {count} record(s)"

                elif database == "digital_twin":
                    # Test profile creation
                    test_student_id = f"test_student_{int(time.time())}"
                    cursor.execute(
                        """
                    INSERT INTO learning_profiles
                    (student_id, name, age, grade_level)
                    VALUES (?, 'Test Student', 12, 7)
                    """,
                        (test_student_id,),
                    )

                    # Test query
                    cursor.execute(
                        """
                    SELECT name FROM learning_profiles WHERE student_id = ?
                    """,
                        (test_student_id,),
                    )
                    result = cursor.fetchone()

                    # Cleanup
                    cursor.execute(
                        "DELETE FROM learning_profiles WHERE student_id = ?",
                        (test_student_id,),
                    )
                    conn.commit()

                    passed = result and result[0] == "Test Student"
                    message = (
                        f"CRUD operations successful, profile test passed: {passed}"
                    )

                elif database == "rag_index":
                    # Test document creation
                    test_doc_id = f"test_doc_{int(time.time())}"
                    cursor.execute(
                        """
                    INSERT INTO documents
                    (document_id, title, content_hash, word_count)
                    VALUES (?, 'Test Document', 'test_hash_123', 100)
                    """,
                        (test_doc_id,),
                    )

                    # Test query
                    cursor.execute(
                        """
                    SELECT title FROM documents WHERE document_id = ?
                    """,
                        (test_doc_id,),
                    )
                    result = cursor.fetchone()

                    # Cleanup
                    cursor.execute(
                        "DELETE FROM documents WHERE document_id = ?", (test_doc_id,)
                    )
                    conn.commit()

                    passed = result and result[0] == "Test Document"
                    message = (
                        f"CRUD operations successful, document test passed: {passed}"
                    )

                else:
                    passed = True
                    message = "Basic operations test skipped for unknown database"

                duration_ms = (time.time() - start_time) * 1000

                self.results.append(
                    ValidationResult(
                        database=database,
                        test_name="basic_operations",
                        passed=passed,
                        value={"operations_tested": "insert, select, delete"},
                        message=message,
                        duration_ms=duration_ms,
                    )
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database=database,
                    test_name="basic_operations",
                    passed=False,
                    value=None,
                    message=f"Basic operations failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="error",
                )
            )

    async def _benchmark_database_performance(self, database: str):
        """Benchmark database performance."""
        start_time = time.time()

        try:
            with self.database_manager.get_connection(database) as conn:
                cursor = conn.cursor()

                # Benchmark SELECT performance
                select_start = time.time()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master")
                select_duration = (time.time() - select_start) * 1000

                self.performance_metrics.append(
                    PerformanceMetrics(
                        database=database,
                        operation="select_count",
                        duration_ms=select_duration,
                        rows_affected=1,
                    )
                )

                # Benchmark INSERT performance (batch)
                insert_start = time.time()
                test_table = self._get_test_table_name(database)

                if test_table:
                    # Create temporary test data
                    test_data = self._generate_test_data(database, 100)

                    for data in test_data:
                        cursor.execute(data["sql"], data["params"])

                    conn.commit()
                    insert_duration = (time.time() - insert_start) * 1000

                    self.performance_metrics.append(
                        PerformanceMetrics(
                            database=database,
                            operation="batch_insert",
                            duration_ms=insert_duration,
                            rows_affected=len(test_data),
                            rows_per_second=len(test_data) / (insert_duration / 1000),
                        )
                    )

                    # Cleanup test data
                    cleanup_sql = self._get_cleanup_sql(database)
                    if cleanup_sql:
                        cursor.execute(cleanup_sql)
                        conn.commit()

                # Test query performance on existing data
                query_start = time.time()
                tables = self._get_expected_tables(database)
                for table in tables:
                    if table != "schema_version":
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        cursor.fetchone()

                query_duration = (time.time() - query_start) * 1000

                self.performance_metrics.append(
                    PerformanceMetrics(
                        database=database,
                        operation="multi_table_count",
                        duration_ms=query_duration,
                        rows_affected=len(tables),
                    )
                )

                duration_ms = (time.time() - start_time) * 1000

                self.results.append(
                    ValidationResult(
                        database=database,
                        test_name="performance_benchmark",
                        passed=True,
                        value={
                            "select_duration_ms": select_duration,
                            "insert_duration_ms": insert_duration if test_table else 0,
                            "query_duration_ms": query_duration,
                        },
                        message=f"Performance benchmark completed in {duration_ms:.2f}ms",
                        duration_ms=duration_ms,
                    )
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database=database,
                    test_name="performance_benchmark",
                    passed=False,
                    value=None,
                    message=f"Performance benchmark failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="warning",
                )
            )

    def _get_test_table_name(self, database: str) -> str | None:
        """Get appropriate test table name for each database."""
        table_map = {
            "evolution_metrics": "evolution_rounds",
            "digital_twin": "learning_profiles",
            "rag_index": "documents",
        }
        return table_map.get(database)

    def _generate_test_data(self, database: str, count: int) -> list[dict[str, Any]]:
        """Generate test data for performance benchmarking."""
        test_data = []

        for i in range(count):
            if database == "evolution_metrics":
                test_data.append(
                    {
                        "sql": "INSERT INTO evolution_rounds (start_time, status) VALUES (?, 'test')",
                        "params": (time.time() + i,),
                    }
                )
            elif database == "digital_twin":
                test_data.append(
                    {
                        "sql": """INSERT INTO learning_profiles
                             (student_id, name, age, grade_level)
                             VALUES (?, ?, 10, 5)""",
                        "params": (f"perf_test_student_{i}", f"Test Student {i}"),
                    }
                )
            elif database == "rag_index":
                test_data.append(
                    {
                        "sql": """INSERT INTO documents
                             (document_id, title, content_hash, word_count)
                             VALUES (?, ?, ?, 100)""",
                        "params": (
                            f"perf_test_doc_{i}",
                            f"Test Document {i}",
                            f"hash_{i}",
                        ),
                    }
                )

        return test_data

    def _get_cleanup_sql(self, database: str) -> str | None:
        """Get cleanup SQL for test data."""
        cleanup_map = {
            "evolution_metrics": "DELETE FROM evolution_rounds WHERE status = 'test'",
            "digital_twin": "DELETE FROM learning_profiles WHERE student_id LIKE 'perf_test_%'",
            "rag_index": "DELETE FROM documents WHERE document_id LIKE 'perf_test_%'",
        }
        return cleanup_map.get(database)

    async def _analyze_database_optimization(self, database: str):
        """Analyze database optimization opportunities."""
        start_time = time.time()

        try:
            with self.database_manager.get_connection(database) as conn:
                cursor = conn.cursor()

                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]

                db_size_bytes = page_count * page_size
                db_size_mb = db_size_bytes / (1024 * 1024)

                # Check for unused space
                cursor.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]

                # Analyze table sizes
                table_stats = {}
                for table in self._get_expected_tables(database):
                    if table != "schema_version":
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        row_count = cursor.fetchone()[0]
                        table_stats[table] = row_count

                # Check index usage
                cursor.execute(
                    """
                SELECT name, sql FROM sqlite_master
                WHERE type='index' AND sql IS NOT NULL
                """
                )
                indexes = cursor.fetchall()

                duration_ms = (time.time() - start_time) * 1000

                # Generate optimization recommendations
                recommendations = []
                if freelist_count > page_count * 0.1:
                    recommendations.append("Consider running VACUUM to reclaim space")

                if db_size_mb > 100:
                    recommendations.append("Large database - consider partitioning")

                total_rows = sum(table_stats.values())
                if total_rows > 10000 and len(indexes) < 3:
                    recommendations.append(
                        "Consider adding more indexes for better query performance"
                    )

                self.results.append(
                    ValidationResult(
                        database=database,
                        test_name="optimization_analysis",
                        passed=True,
                        value={
                            "size_mb": db_size_mb,
                            "page_count": page_count,
                            "freelist_count": freelist_count,
                            "table_stats": table_stats,
                            "index_count": len(indexes),
                            "recommendations": recommendations,
                        },
                        message=f"Database size: {db_size_mb:.2f}MB, {len(recommendations)} recommendations",
                        duration_ms=duration_ms,
                    )
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database=database,
                    test_name="optimization_analysis",
                    passed=False,
                    value=None,
                    message=f"Optimization analysis failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="warning",
                )
            )

    async def _validate_data_consistency(self, database: str):
        """Validate data consistency and referential integrity."""
        start_time = time.time()

        try:
            with self.database_manager.get_connection(database) as conn:
                cursor = conn.cursor()

                # Check foreign key consistency
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()

                # Database-specific consistency checks
                consistency_issues = []

                if database == "evolution_metrics":
                    # Check that all fitness_metrics have valid round_ids
                    cursor.execute(
                        """
                    SELECT COUNT(*) FROM fitness_metrics fm
                    LEFT JOIN evolution_rounds er ON fm.round_id = er.id
                    WHERE er.id IS NULL
                    """
                    )
                    orphaned_metrics = cursor.fetchone()[0]
                    if orphaned_metrics > 0:
                        consistency_issues.append(
                            f"{orphaned_metrics} orphaned fitness metrics"
                        )

                elif database == "digital_twin":
                    # Check that all learning_sessions have valid student_ids
                    cursor.execute(
                        """
                    SELECT COUNT(*) FROM learning_sessions ls
                    LEFT JOIN learning_profiles lp ON ls.student_id = lp.student_id
                    WHERE lp.student_id IS NULL
                    """
                    )
                    orphaned_sessions = cursor.fetchone()[0]
                    if orphaned_sessions > 0:
                        consistency_issues.append(
                            f"{orphaned_sessions} orphaned learning sessions"
                        )

                elif database == "rag_index":
                    # Check that all chunks have valid document_ids
                    cursor.execute(
                        """
                    SELECT COUNT(*) FROM chunks c
                    LEFT JOIN documents d ON c.document_id = d.document_id
                    WHERE d.document_id IS NULL
                    """
                    )
                    orphaned_chunks = cursor.fetchone()[0]
                    if orphaned_chunks > 0:
                        consistency_issues.append(f"{orphaned_chunks} orphaned chunks")

                duration_ms = (time.time() - start_time) * 1000

                passed = len(fk_violations) == 0 and len(consistency_issues) == 0

                self.results.append(
                    ValidationResult(
                        database=database,
                        test_name="data_consistency",
                        passed=passed,
                        value={
                            "foreign_key_violations": len(fk_violations),
                            "consistency_issues": consistency_issues,
                        },
                        message=f"Consistency check: {len(fk_violations)} FK violations, {len(consistency_issues)} issues",
                        duration_ms=duration_ms,
                        severity="error" if not passed else "info",
                    )
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database=database,
                    test_name="data_consistency",
                    passed=False,
                    value=None,
                    message=f"Data consistency check failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="error",
                )
            )

    async def _validate_redis_connections(self):
        """Validate Redis connections and fallback systems."""
        start_time = time.time()

        try:
            # Check connection status
            connection_status = await self.redis_manager.check_connections()

            # Test each connection pool
            for pool_name in [
                "evolution_metrics",
                "rag_cache",
                "p2p_discovery",
                "session_store",
            ]:
                pool_start = time.time()

                async with self.redis_manager.get_connection(pool_name) as conn:
                    # Test basic operations
                    test_key = f"validation_test_{int(time.time())}"
                    test_value = {"test": "data", "timestamp": time.time()}

                    # Set/Get test
                    set_success = await conn.set(test_key, test_value, ex=10)
                    get_value = await conn.get(test_key)

                    # Cleanup
                    await conn.delete(test_key)

                    pool_duration = (time.time() - pool_start) * 1000

                    passed = set_success and get_value is not None
                    connection_type = conn.connection_type

                    self.results.append(
                        ValidationResult(
                            database=f"redis_{pool_name}",
                            test_name="redis_operations",
                            passed=passed,
                            value={
                                "connection_type": connection_type,
                                "set_success": set_success,
                                "get_success": get_value is not None,
                            },
                            message=f"Redis {pool_name} ({connection_type}): Operations {'successful' if passed else 'failed'}",
                            duration_ms=pool_duration,
                        )
                    )

            duration_ms = (time.time() - start_time) * 1000

            self.results.append(
                ValidationResult(
                    database="redis_manager",
                    test_name="redis_validation",
                    passed=True,
                    value=connection_status,
                    message=f"Redis validation completed for {len(connection_status)} pools",
                    duration_ms=duration_ms,
                )
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.append(
                ValidationResult(
                    database="redis_manager",
                    test_name="redis_validation",
                    passed=False,
                    value=None,
                    message=f"Redis validation failed: {e!s}",
                    duration_ms=duration_ms,
                    severity="warning",
                )
            )

    def _generate_validation_summary(self, total_duration_ms: float) -> dict[str, Any]:
        """Generate comprehensive validation summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_ms": total_duration_ms,
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "failed_tests": sum(1 for r in self.results if not r.passed),
            "warnings": sum(1 for r in self.results if r.severity == "warning"),
            "errors": sum(1 for r in self.results if r.severity == "error"),
            "databases_tested": len(set(r.database for r in self.results)),
            "performance_metrics_count": len(self.performance_metrics),
            "results": [],
            "performance_metrics": [],
            "recommendations": [],
        }

        # Add detailed results
        for result in self.results:
            summary["results"].append(
                {
                    "database": result.database,
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "value": result.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "severity": result.severity,
                }
            )

        # Add performance metrics
        for metric in self.performance_metrics:
            summary["performance_metrics"].append(
                {
                    "database": metric.database,
                    "operation": metric.operation,
                    "duration_ms": metric.duration_ms,
                    "rows_affected": metric.rows_affected,
                    "rows_per_second": metric.rows_per_second,
                }
            )

        # Generate recommendations
        recommendations = []

        # Check for slow operations
        slow_operations = [m for m in self.performance_metrics if m.duration_ms > 1000]
        if slow_operations:
            recommendations.append(
                f"Found {len(slow_operations)} slow operations (>1s) - consider optimization"
            )

        # Check for failed tests
        failed_critical = [
            r for r in self.results if not r.passed and r.severity == "error"
        ]
        if failed_critical:
            recommendations.append(
                f"{len(failed_critical)} critical tests failed - immediate attention required"
            )

        # Check database sizes
        large_dbs = [
            r
            for r in self.results
            if r.test_name == "optimization_analysis"
            and r.value
            and r.value.get("size_mb", 0) > 100
        ]
        if large_dbs:
            recommendations.append(
                "Large databases detected - consider maintenance and optimization"
            )

        summary["recommendations"] = recommendations

        # Overall health score (0-100)
        health_score = (summary["passed_tests"] / summary["total_tests"]) * 100
        if summary["errors"] > 0:
            health_score *= 0.7  # Penalize errors
        if summary["warnings"] > 5:
            health_score *= 0.9  # Penalize many warnings

        summary["health_score"] = round(health_score, 2)
        summary["health_status"] = (
            "excellent"
            if health_score >= 95
            else (
                "good"
                if health_score >= 85
                else "fair" if health_score >= 70 else "poor"
            )
        )

        return summary


async def validate_aivillage_databases(
    database_manager, redis_manager=None
) -> dict[str, Any]:
    """Run comprehensive validation on AIVillage databases."""
    validator = DatabaseValidator(database_manager, redis_manager)
    return await validator.validate_all_databases()


if __name__ == "__main__":

    async def main():
        """Test database validation."""
        from database_manager import initialize_databases
        from redis_manager import initialize_redis

        # Initialize systems
        db_manager = await initialize_databases()
        redis_manager = await initialize_redis()

        # Run validation
        results = await validate_aivillage_databases(db_manager, redis_manager)

        # Print summary
        print("Database Validation Results")
        print(
            f"Health Score: {results['health_score']}/100 ({results['health_status']})"
        )
        print(f"Tests: {results['passed_tests']}/{results['total_tests']} passed")
        print(f"Duration: {results['total_duration_ms']:.2f}ms")

        if results["recommendations"]:
            print("\nRecommendations:")
            for rec in results["recommendations"]:
                print(f"  - {rec}")

        # Close connections
        await db_manager.close()
        await redis_manager.close()

    asyncio.run(main())
