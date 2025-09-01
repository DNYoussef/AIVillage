#!/usr/bin/env python3
"""Comprehensive SQL Injection Prevention Tests.

Tests to verify that all database operations are secure against SQL injection attacks.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path

# Import the classes we're testing
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.rag.mcp_servers.hyperag.secure_database import SecureDatabaseManager
from infrastructure.shared.security.multi_tenant_system import (
    MultiTenantSystem,
    TenantType,
    SQLSecurityManager,
    SecureInputValidator as MTSecureInputValidator,
)


class TestSQLInjectionPrevention:
    """Test SQL injection prevention mechanisms."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def secure_db(self, temp_db_path):
        """Create secure database instance."""
        db = SecureDatabaseManager(temp_db_path)
        yield db
        db.close()

    @pytest.fixture
    def multi_tenant_system(self, temp_db_path):
        """Create multi-tenant system instance."""
        return MultiTenantSystem(temp_db_path)

    @pytest.mark.asyncio
    async def test_memory_search_sql_injection_attempts(self, secure_db):
        """Test that memory search is protected against SQL injection."""
        # Store some test data first
        await secure_db.store_memory("Test content 1", ["tag1"], 0.5)
        await secure_db.store_memory("Test content 2", ["tag2"], 0.8)

        # SQL injection attempts that should be handled safely
        injection_attempts = [
            "' OR '1'='1",
            "'; DROP TABLE memories; --",
            "' UNION SELECT * FROM sqlite_master --",
            "admin'--",
            "admin' /*",
            "' OR 1=1#",
            "' OR 'x'='x",
            "'; INSERT INTO memories VALUES ('malicious'); --",
            "' AND (SELECT COUNT(*) FROM memories) > 0 --",
            "1' AND SLEEP(5) AND '1'='1",
        ]

        for injection_query in injection_attempts:
            # These should not cause SQL errors or return unauthorized data
            results = await secure_db.search_memories(injection_query)

            # Results should be empty or contain only legitimate matches
            # The injection should be treated as literal text, not SQL
            assert isinstance(results, list)

            # Verify the database is still intact
            stats = await secure_db.get_database_stats()
            assert stats["memory_count"] >= 2  # Our test data should still exist

    @pytest.mark.asyncio
    async def test_tag_based_sql_injection_attempts(self, secure_db):
        """Test tag filtering against SQL injection."""
        # Store test data with tags
        await secure_db.store_memory("Content with tag", ["safe_tag"], 0.5)

        malicious_tags = [
            "'; DROP TABLE memories; --",
            "' OR '1'='1",
            "tag'; DELETE FROM memories; --",
            "tag' UNION SELECT * FROM sqlite_master --",
        ]

        # Test with malicious tags
        results = await secure_db.search_memories("Content", tags=malicious_tags)

        # Should handle gracefully without SQL errors
        assert isinstance(results, list)

        # Verify database integrity
        stats = await secure_db.get_database_stats()
        assert stats["memory_count"] >= 1

    @pytest.mark.asyncio
    async def test_knowledge_graph_sql_injection(self, secure_db):
        """Test knowledge graph operations against SQL injection."""
        # Store legitimate data first
        await secure_db.store_knowledge_triple("Python", "is_a", "language")

        # SQL injection attempts in knowledge graph fields
        injection_subjects = [
            "'; DROP TABLE knowledge_graph; --",
            "Python' OR '1'='1",
            "subject'; DELETE FROM knowledge_graph; --",
        ]

        for malicious_subject in injection_subjects:
            # Should handle injection attempts safely
            results = await secure_db.query_knowledge_graph(subject=malicious_subject)
            assert isinstance(results, list)

            # Verify the legitimate data is still there
            legitimate_results = await secure_db.query_knowledge_graph(subject="Python")
            assert len(legitimate_results) >= 1

    def test_multi_tenant_sql_injection_protection(self, multi_tenant_system):
        """Test multi-tenant system SQL injection protection."""
        # Create organization with potentially malicious data
        malicious_names = [
            "org'; DROP TABLE organizations; --",
            "test' OR '1'='1",
            "org'; DELETE FROM organizations; --",
        ]

        for malicious_name in malicious_names:
            try:
                # This should either fail safely or sanitize the input
                org_id = multi_tenant_system.create_organization(
                    name=malicious_name,
                    display_name="Test Organization",
                    tenant_type=TenantType.TEAM,
                )

                # If it succeeds, verify the data was sanitized
                stats = multi_tenant_system.get_tenant_stats(org_id)
                assert stats is not None

            except Exception as e:
                # If it fails, it should be a validation error, not SQL error
                assert "sql" not in str(e).lower()

    def test_tenant_access_sql_injection(self, multi_tenant_system):
        """Test tenant access checking against SQL injection."""
        # Create a legitimate organization first
        org_id = multi_tenant_system.create_organization(
            name="legitimate_org",
            display_name="Legitimate Organization",
            tenant_type=TenantType.TEAM,
        )

        # Add a user
        multi_tenant_system.add_user_to_tenant(
            user_id="user123",
            organization_id=org_id,
            role="member",
        )

        # Try SQL injection in user_id parameter
        malicious_user_ids = [
            "'; DROP TABLE tenant_memberships; --",
            "user123' OR '1'='1",
            "user'; DELETE FROM tenant_memberships; --",
        ]

        for malicious_user_id in malicious_user_ids:
            # Should handle safely without SQL errors
            has_access = multi_tenant_system.check_tenant_access(
                user_id=malicious_user_id,
                tenant_id=org_id,
            )
            # Should return False for malicious input
            assert has_access is False

    def test_sql_security_manager_validation(self):
        """Test SQLSecurityManager input validation."""
        # Test identifier validation
        with pytest.raises(ValueError):
            SQLSecurityManager.validate_identifier("'; DROP TABLE test; --")

        with pytest.raises(ValueError):
            SQLSecurityManager.validate_identifier("table/*malicious*/")

        with pytest.raises(ValueError):
            SQLSecurityManager.validate_identifier("test; DELETE FROM users;")

        # Test legitimate identifiers
        assert SQLSecurityManager.validate_identifier("valid_table_name") == "valid_table_name"
        assert SQLSecurityManager.validate_identifier("user-data") == "user-data"

    def test_parameterized_query_builder(self):
        """Test secure parameterized query building."""
        base_query = "SELECT * FROM users"

        # Test with safe conditions
        conditions = {"name": "John", "age": 30}
        query, params = SQLSecurityManager.build_parameterized_query(base_query, conditions)

        assert "?" in query  # Should have parameter placeholders
        assert len(params) == 2
        assert "John" in params
        assert 30 in params

        # Test with malicious column names should raise error
        malicious_conditions = {"name'; DROP TABLE users; --": "value"}

        with pytest.raises(ValueError):
            SQLSecurityManager.build_parameterized_query(base_query, malicious_conditions)

    def test_query_structure_validation(self):
        """Test SQL query structure validation."""
        # Test legitimate queries
        SQLSecurityManager.validate_query_structure("SELECT * FROM users")
        SQLSecurityManager.validate_query_structure("INSERT INTO users VALUES (?, ?)")

        # Test dangerous queries
        with pytest.raises(ValueError):
            SQLSecurityManager.validate_query_structure("SELECT * FROM users; DROP TABLE users;")

        with pytest.raises(ValueError):
            SQLSecurityManager.validate_query_structure("SELECT xp_cmdshell('dir')")

        with pytest.raises(ValueError):
            SQLSecurityManager.validate_query_structure("SELECT * FROM users INTO OUTFILE '/tmp/users.txt'")

    def test_input_validation(self):
        """Test secure input validation."""
        # Test tenant ID validation
        valid_tenant_id = MTSecureInputValidator.validate_tenant_id("org_12345")
        assert valid_tenant_id == "org_12345"

        # Test malicious tenant IDs
        with pytest.raises(ValueError):
            MTSecureInputValidator.validate_tenant_id("'; DROP TABLE users; --")

        with pytest.raises(ValueError):
            MTSecureInputValidator.validate_tenant_id("org/malicious")

        # Test SQL limit validation
        assert MTSecureInputValidator.validate_sql_limit(100) == 100

        with pytest.raises(ValueError):
            MTSecureInputValidator.validate_sql_limit(-1)

        with pytest.raises(ValueError):
            MTSecureInputValidator.validate_sql_limit(99999)

    @pytest.mark.asyncio
    async def test_database_integrity_after_attacks(self, secure_db):
        """Verify database integrity is maintained after injection attempts."""
        # Store initial data
        hash1 = await secure_db.store_memory("Initial content", ["initial"], 1.0)
        await secure_db.store_knowledge_triple("Initial", "relates_to", "test")

        initial_stats = await secure_db.get_database_stats()

        # Attempt various injection attacks
        attack_queries = [
            "'; DROP TABLE memories; --",
            "' OR 1=1; DELETE FROM memories; --",
            "'; UPDATE memories SET content='hacked'; --",
            "' UNION SELECT * FROM sqlite_master; --",
        ]

        for attack in attack_queries:
            await secure_db.search_memories(attack)
            await secure_db.query_knowledge_graph(subject=attack)

        # Verify database integrity maintained
        final_stats = await secure_db.get_database_stats()
        assert final_stats["memory_count"] >= initial_stats["memory_count"]
        assert final_stats["knowledge_triple_count"] >= initial_stats["knowledge_triple_count"]

        # Verify original data is intact
        retrieved_memory = await secure_db.retrieve_memory(hash1)
        assert retrieved_memory is not None
        assert retrieved_memory["content"] == "Initial content"

    def test_connection_security(self, temp_db_path):
        """Test database connection security settings."""
        db = SecureDatabaseManager(temp_db_path)

        # Verify secure connection settings are applied
        cursor = db.connection.execute("PRAGMA foreign_keys")
        foreign_keys_enabled = cursor.fetchone()[0]
        assert foreign_keys_enabled == 1

        cursor = db.connection.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode == "wal"

        db.close()


if __name__ == "__main__":
    # Run basic tests
    async def run_basic_tests():
        """Run basic injection tests."""
        print("Starting SQL injection prevention tests...")

        # Create temporary database
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            # Test SecureDatabaseManager
            print("Testing SecureDatabaseManager...")
            db = SecureDatabaseManager(db_path)

            # Test basic functionality with clean data
            hash1 = await db.store_memory("Test content", ["test"], 0.5)
            memory = await db.retrieve_memory(hash1)
            assert memory is not None
            print("✓ Basic functionality works")

            # Test injection attempts
            results = await db.search_memories("'; DROP TABLE memories; --")
            assert isinstance(results, list)
            print("✓ Search injection attempt handled safely")

            # Verify database is still functional
            stats = await db.get_database_stats()
            assert stats["memory_count"] >= 1
            print("✓ Database integrity maintained")

            db.close()

            # Test MultiTenantSystem
            print("Testing MultiTenantSystem...")
            mt_system = MultiTenantSystem(db_path)

            org_id = mt_system.create_organization(
                name="test_org",
                display_name="Test Organization",
                tenant_type=TenantType.TEAM,
            )
            print("✓ Organization creation works")

            # Test access with malicious input
            has_access = mt_system.check_tenant_access(
                user_id="'; DROP TABLE tenant_memberships; --",
                tenant_id=org_id,
            )
            assert has_access is False
            print("✓ Access check injection attempt handled safely")

            print("\n🛡️  All SQL injection prevention tests passed!")

        finally:
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass

    # Run the tests
    asyncio.run(run_basic_tests())
