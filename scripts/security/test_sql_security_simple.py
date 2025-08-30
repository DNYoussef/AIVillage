#!/usr/bin/env python3
"""Simple SQL Security Test.

Quick test to verify SQL injection prevention in target files.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.rag.mcp_servers.hyperag.secure_database import SecureDatabaseManager
from infrastructure.shared.security.multi_tenant_system import MultiTenantSystem, TenantType


async def test_secure_database():
    """Test SecureDatabaseManager SQL injection prevention."""
    print("Testing SecureDatabaseManager...")
    
    # Create temporary database
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    try:
        db = SecureDatabaseManager(db_path)
        
        # Test 1: Store legitimate data
        print("  1. Testing legitimate operations...")
        hash1 = await db.store_memory("Test content", ["test"], 0.5)
        memory = await db.retrieve_memory(hash1)
        assert memory is not None, "Failed to store/retrieve legitimate data"
        print("     PASS: Basic operations work")

        # Test 2: SQL injection in search
        print("  2. Testing SQL injection in search...")
        injection_queries = [
            "'; DROP TABLE memories; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM sqlite_master --",
        ]
        
        for query in injection_queries:
            results = await db.search_memories(query)
            assert isinstance(results, list), f"Search failed for: {query}"
        print("     PASS: Search injection attempts handled safely")

        # Test 3: Verify database integrity
        print("  3. Testing database integrity...")
        stats = await db.get_database_stats()
        assert stats["memory_count"] >= 1, "Database corruption detected"
        
        # Verify original data is still there
        memory_check = await db.retrieve_memory(hash1)
        assert memory_check is not None, "Original data was corrupted"
        assert memory_check["content"] == "Test content", "Data integrity compromised"
        print("     PASS: Database integrity maintained")

        # Test 4: Knowledge graph injection
        print("  4. Testing knowledge graph injection...")
        await db.store_knowledge_triple("Python", "is_a", "language")
        
        malicious_subjects = [
            "'; DROP TABLE knowledge_graph; --",
            "Python' OR '1'='1",
        ]
        
        for subject in malicious_subjects:
            results = await db.query_knowledge_graph(subject=subject)
            assert isinstance(results, list), f"Knowledge graph query failed for: {subject}"
        
        # Verify legitimate data still exists
        results = await db.query_knowledge_graph(subject="Python")
        assert len(results) >= 1, "Knowledge graph data was corrupted"
        print("     PASS: Knowledge graph injection handled safely")

        db.close()
        print("  SUCCESS: SecureDatabaseManager is secure!")
        
    finally:
        try:
            os.unlink(db_path)
        except (OSError, PermissionError):
            pass  # File might be locked on Windows


def test_multi_tenant_system():
    """Test MultiTenantSystem SQL injection prevention."""
    print("Testing MultiTenantSystem...")
    
    # Create temporary database
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    try:
        system = MultiTenantSystem(db_path)
        
        # Test 1: Create legitimate organization
        print("  1. Testing legitimate organization creation...")
        org_id = system.create_organization(
            name="test_org",
            display_name="Test Organization",
            tenant_type=TenantType.TEAM,
        )
        assert org_id is not None, "Failed to create organization"
        print("     PASS: Organization creation works")

        # Test 2: Add user
        print("  2. Testing user addition...")
        success = system.add_user_to_tenant(
            user_id="user123",
            organization_id=org_id,
            role="member",
        )
        assert success, "Failed to add user to tenant"
        print("     PASS: User addition works")

        # Test 3: Test access check with injection
        print("  3. Testing access check injection...")
        malicious_user_ids = [
            "'; DROP TABLE tenant_memberships; --",
            "user123' OR '1'='1",
        ]
        
        for malicious_id in malicious_user_ids:
            has_access = system.check_tenant_access(
                user_id=malicious_id,
                tenant_id=org_id,
            )
            assert has_access is False, f"Access granted for malicious ID: {malicious_id}"
        print("     PASS: Access check injection handled safely")

        # Test 4: Verify legitimate user still has access
        print("  4. Testing legitimate access...")
        has_access = system.check_tenant_access(
            user_id="user123",
            tenant_id=org_id,
        )
        assert has_access is True, "Legitimate user lost access"
        print("     PASS: Legitimate access maintained")

        # Test 5: Test malicious organization names
        print("  5. Testing organization name injection...")
        malicious_names = [
            "org'; DROP TABLE organizations; --",
            "test' OR '1'='1",
        ]
        
        for malicious_name in malicious_names:
            try:
                system.create_organization(
                    name=malicious_name,
                    display_name="Test",
                    tenant_type=TenantType.TEAM,
                )
                # If it succeeds, that's okay as long as it doesn't cause SQL errors
            except Exception as e:
                # Should be validation error, not SQL error
                assert "sql" not in str(e).lower(), f"SQL error occurred: {e}"
        print("     PASS: Organization name injection handled safely")

        print("  SUCCESS: MultiTenantSystem is secure!")
        
    finally:
        try:
            os.unlink(db_path)
        except (OSError, PermissionError):
            pass


async def main():
    """Run all security tests."""
    print("SQL INJECTION SECURITY TEST")
    print("=" * 40)
    print()
    
    try:
        # Test SecureDatabaseManager
        await test_secure_database()
        print()
        
        # Test MultiTenantSystem
        test_multi_tenant_system()
        print()
        
        print("=" * 40)
        print("ALL TESTS PASSED - SQL INJECTION PREVENTION IS WORKING!")
        print("=" * 40)
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)