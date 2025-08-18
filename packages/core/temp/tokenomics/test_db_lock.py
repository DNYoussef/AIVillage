"""Test to validate database locking fixes for Windows compatibility.

This test validates that the SQLite database improvements resolve the
WinError 32 file locking issues and provide cross-platform reliability.
"""

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class TestDatabaseLocking(unittest.TestCase):
    """Test cases to verify database locking fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent
        self.temp_dir = Path(tempfile.gettempdir())

        # Import the fixed credit system
        import sys

        sys.path.append(str(self.project_root / "src"))

        try:
            from token_economy.credit_system import EarningRule, SQLiteDatabase, VILLAGECreditSystem

            self.VILLAGECreditSystem = VILLAGECreditSystem
            self.EarningRule = EarningRule
            self.SQLiteDatabase = SQLiteDatabase
        except ImportError as e:
            self.skipTest(f"Could not import credit system: {e}")

    def test_sqlite_wal_mode_configured(self):
        """Test that SQLite is configured with WAL mode."""
        test_db_path = self.temp_dir / "test_wal_mode.db"

        try:
            db = self.SQLiteDatabase(str(test_db_path))

            # Check WAL mode is enabled
            cursor = db.conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.upper(), "WAL", "Database should be in WAL mode")

            # Check busy timeout is configured
            cursor.execute("PRAGMA busy_timeout")
            timeout = cursor.fetchone()[0]
            self.assertGreaterEqual(timeout, 5000, "Busy timeout should be at least 5 seconds")

            cursor.close()
            db.close()

        finally:
            # Clean up
            self._cleanup_test_files(test_db_path)

    def test_concurrent_database_access(self):
        """Test that multiple threads can access the database concurrently."""
        test_db_path = self.temp_dir / "test_concurrent.db"

        def worker_thread(thread_id: int, results: list):
            """Worker thread that performs database operations."""
            try:
                credit_system = self.VILLAGECreditSystem(str(test_db_path))

                # Add earning rule (if not already exists)
                try:
                    rule = self.EarningRule("TEST_ACTION", 10, {}, {})
                    credit_system.add_earning_rule(rule)
                except:
                    pass  # Rule may already exist

                # Perform some operations
                user_id = f"user_{thread_id}"
                credit_system.earn_credits(user_id, "TEST_ACTION", {"thread": str(thread_id)})
                balance = credit_system.get_balance(user_id)

                results.append({"thread_id": thread_id, "balance": balance, "success": True})

                credit_system.close()

            except Exception as e:
                results.append({"thread_id": thread_id, "error": str(e), "success": False})

        # Run multiple threads concurrently
        results = []
        num_threads = 5

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i, results) for i in range(num_threads)]

                # Wait for all threads to complete
                for future in futures:
                    future.result(timeout=30)  # 30 second timeout

            # Verify all threads succeeded
            self.assertEqual(len(results), num_threads, "All threads should complete")

            successful_threads = [r for r in results if r.get("success", False)]
            failed_threads = [r for r in results if not r.get("success", False)]

            if failed_threads:
                error_messages = [r.get("error", "Unknown error") for r in failed_threads]
                self.fail(f"Threads failed with errors: {error_messages}")

            self.assertEqual(
                len(successful_threads),
                num_threads,
                f"All {num_threads} threads should succeed, got {len(successful_threads)}",
            )

        finally:
            # Clean up
            self._cleanup_test_files(test_db_path)

    def test_per_process_temp_databases(self):
        """Test that per-process database files avoid conflicts."""
        base_test_path = self.temp_dir / "test_per_process.db"

        try:
            # Create database instances
            db1 = self.SQLiteDatabase(str(base_test_path))
            db2 = self.SQLiteDatabase(str(base_test_path))

            # Both should work without conflicts
            db1.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, value TEXT)")
            db2.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, value TEXT)")

            # Insert data from both
            db1.execute("INSERT INTO test_table (value) VALUES (?)", ("from_db1",))
            db2.execute("INSERT INTO test_table (value) VALUES (?)", ("from_db2",))

            # Both should be able to read
            cursor1 = db1.execute("SELECT COUNT(*) FROM test_table")
            row1 = cursor1.fetchone()
            count1 = row1[0] if row1 else 0

            cursor2 = db2.execute("SELECT COUNT(*) FROM test_table")
            row2 = cursor2.fetchone()
            count2 = row2[0] if row2 else 0

            self.assertGreaterEqual(count1, 1, "DB1 should have at least 1 record")
            self.assertGreaterEqual(count2, 1, "DB2 should have at least 1 record")

            db1.close()
            db2.close()

        finally:
            # Clean up all possible temp files
            self._cleanup_test_files(base_test_path)

    def test_memory_database_for_tests(self):
        """Test that in-memory databases work for tests."""
        try:
            # Create in-memory credit system
            credit_system = self.VILLAGECreditSystem(":memory:")

            # Should be able to add rules and perform operations
            rule = self.EarningRule("MEMORY_TEST", 50, {}, {})
            credit_system.add_earning_rule(rule)

            # Earn some credits
            credit_system.earn_credits("test_user", "MEMORY_TEST", {"test": "memory"})
            balance = credit_system.get_balance("test_user")

            self.assertGreater(balance, 0, "Should earn credits in memory database")

            credit_system.close()

        except Exception as e:
            self.fail(f"In-memory database test failed: {e}")

    def test_database_recovery_from_busy_state(self):
        """Test that database can recover from busy/locked state."""
        test_db_path = self.temp_dir / "test_recovery.db"

        try:
            credit_system = self.VILLAGECreditSystem(str(test_db_path))

            # Add earning rule
            rule = self.EarningRule("RECOVERY_TEST", 25, {}, {})
            credit_system.add_earning_rule(rule)

            # Simulate busy database by creating a long-running transaction
            def busy_worker():
                try:
                    busy_system = self.VILLAGECreditSystem(str(test_db_path))
                    # Start a transaction and hold it
                    busy_system.db.conn.execute("BEGIN IMMEDIATE")
                    time.sleep(2)  # Hold for 2 seconds
                    busy_system.db.conn.rollback()
                    busy_system.close()
                except:
                    pass

            # Start busy worker in background
            busy_thread = threading.Thread(target=busy_worker)
            busy_thread.start()

            time.sleep(0.5)  # Let busy worker start

            # Now try to perform operation (should succeed with retry logic)
            start_time = time.time()
            credit_system.earn_credits("recovery_user", "RECOVERY_TEST", {"test": "recovery"})
            end_time = time.time()

            # Should eventually succeed
            balance = credit_system.get_balance("recovery_user")
            self.assertGreater(balance, 0, "Should earn credits despite busy database")

            # Should not take too long (busy_timeout + retry)
            self.assertLess(end_time - start_time, 10, "Recovery should not take too long")

            busy_thread.join()  # Wait for busy worker to complete
            credit_system.close()

        finally:
            # Clean up
            self._cleanup_test_files(test_db_path)

    def test_cross_platform_file_paths(self):
        """Test that database works with cross-platform file paths."""
        # Test with various path formats
        test_cases = [
            "test_cross_platform.db",
            str(self.temp_dir / "test_cross_platform2.db"),
        ]

        if os.name == "nt":  # Windows
            test_cases.append(r"C:\temp\test_windows_path.db")
        else:  # Unix-like
            test_cases.append("/tmp/test_unix_path.db")

        for db_path in test_cases:
            try:
                # Skip invalid paths on current platform
                path_obj = Path(db_path)
                if not path_obj.parent.exists():
                    continue

                credit_system = self.VILLAGECreditSystem(db_path)

                # Basic functionality test
                rule = self.EarningRule("CROSS_PLATFORM", 15, {}, {})
                credit_system.add_earning_rule(rule)

                credit_system.earn_credits("path_user", "CROSS_PLATFORM", {"path": db_path})
                balance = credit_system.get_balance("path_user")

                self.assertGreater(balance, 0, f"Should work with path: {db_path}")

                credit_system.close()

            except Exception as e:
                self.fail(f"Cross-platform path test failed for {db_path}: {e}")
            finally:
                # Clean up
                self._cleanup_test_files(Path(db_path))

    def test_portalocker_availability(self):
        """Test that portalocker is available for cross-platform locking."""
        try:
            import portalocker

            # Test basic locking functionality
            test_file = self.temp_dir / "test_lock_file.txt"

            try:
                with open(test_file, "w") as f:
                    portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    f.write("test")
                    portalocker.unlock(f)

                self.assertTrue(test_file.exists(), "Portalocker should create and lock file")

            finally:
                if test_file.exists():
                    test_file.unlink()

        except ImportError:
            self.skipTest("Portalocker not available - install with: pip install portalocker")

    def test_windows_temp_file_cleanup(self):
        """Test that temporary database files are cleaned up properly on Windows."""
        if os.name != "nt":
            self.skipTest("Windows-specific test")

        test_db_base = self.temp_dir / "test_windows_cleanup"

        try:
            # Create multiple database instances to generate temp files
            databases = []
            for i in range(3):
                db_path = f"{test_db_base}_{i}.db"
                credit_system = self.VILLAGECreditSystem(db_path)
                databases.append((credit_system, Path(db_path)))

            # Close all databases
            for credit_system, db_path in databases:
                credit_system.close()

                # Verify file can be deleted (not locked)
                if db_path.exists():
                    try:
                        db_path.unlink()
                    except PermissionError:
                        self.fail(f"Database file {db_path} is still locked after close")

        finally:
            # Clean up any remaining files
            for i in range(3):
                self._cleanup_test_files(Path(f"{test_db_base}_{i}.db"))

    def _cleanup_test_files(self, base_path: Path):
        """Clean up test database files including WAL and SHM files."""
        cleanup_paths = [
            base_path,
            Path(str(base_path) + "-wal"),
            Path(str(base_path) + "-shm"),
        ]

        # Also clean up per-process files
        for i in range(10):  # Clean up process IDs 0-9
            cleanup_paths.append(Path(f"{base_path}.{i}"))
            cleanup_paths.append(Path(f"{base_path}.{i}-wal"))
            cleanup_paths.append(Path(f"{base_path}.{i}-shm"))

        for path in cleanup_paths:
            try:
                if path.exists():
                    path.unlink()
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    unittest.main(verbosity=2)
