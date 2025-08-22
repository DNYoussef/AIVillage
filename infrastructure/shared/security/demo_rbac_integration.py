"""
AIVillage RBAC Integration Demo.

Demonstrates the complete RBAC and multi-tenant isolation system
integrated with all major AIVillage components.
"""

import asyncio
import logging

from . import Permission, Role, initialize_aivillage_rbac

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_rbac_system():
    """Comprehensive demo of RBAC system with AIVillage integration."""
    print("🚀 AIVillage RBAC Integration Demo")
    print("=" * 50)

    # Initialize RBAC system
    print("\n1. Initializing RBAC system...")
    integration = await initialize_aivillage_rbac()
    rbac = integration.rbac
    tenant_manager = integration.tenant_manager

    # Create demo tenant
    print("\n2. Creating demo tenant...")
    tenant = await rbac.create_tenant(
        name="Demo Corporation",
        admin_user={"username": "demo_admin", "email": "admin@demo.corp", "password": "SecurePassword123!"},
        config={"max_agents": 20, "max_rag_collections": 10, "storage_quota_gb": 500},
    )
    print(f"   Created tenant: {tenant.name} ({tenant.tenant_id})")

    # Create additional users with different roles
    print("\n3. Creating users with different roles...")

    developer = await rbac.create_user(
        username="alice_dev",
        email="alice@demo.corp",
        password="DevPassword123!",
        tenant_id=tenant.tenant_id,
        role=Role.DEVELOPER,
    )
    print(f"   Created developer: {developer.username}")

    data_scientist = await rbac.create_user(
        username="bob_scientist",
        email="bob@demo.corp",
        password="DataPassword123!",
        tenant_id=tenant.tenant_id,
        role=Role.DATA_SCIENTIST,
    )
    print(f"   Created data scientist: {data_scientist.username}")

    regular_user = await rbac.create_user(
        username="charlie_user",
        email="charlie@demo.corp",
        password="UserPassword123!",
        tenant_id=tenant.tenant_id,
        role=Role.USER,
    )
    print(f"   Created regular user: {regular_user.username}")

    # Test authentication
    print("\n4. Testing authentication...")

    admin_session = await rbac.authenticate(
        username="demo_admin",
        password="SecurePassword123!",
        tenant_id=tenant.tenant_id,
        ip_address="127.0.0.1",
        user_agent="Demo/1.0",
    )
    print(f"   Admin authenticated: {admin_session.user_id}")

    dev_session = await rbac.authenticate(
        username="alice_dev",
        password="DevPassword123!",
        tenant_id=tenant.tenant_id,
        ip_address="127.0.0.1",
        user_agent="Demo/1.0",
    )
    print(f"   Developer authenticated: {dev_session.user_id}")

    # Test permission system
    print("\n5. Testing permission system...")

    permissions_to_test = [
        (developer.user_id, Permission.AGENT_CREATE, "Can create agents"),
        (developer.user_id, Permission.SYSTEM_CONFIG, "Can configure system"),
        (data_scientist.user_id, Permission.MODEL_TRAIN, "Can train models"),
        (data_scientist.user_id, Permission.AGENT_DELETE, "Can delete agents"),
        (regular_user.user_id, Permission.AGENT_EXECUTE, "Can execute agents"),
        (regular_user.user_id, Permission.TENANT_MANAGE, "Can manage tenant"),
    ]

    for user_id, permission, description in permissions_to_test:
        has_permission = await rbac.check_permission(user_id, permission)
        status = "✅ ALLOWED" if has_permission else "❌ DENIED"
        print(f"   {status}: {description}")

    # Test agent system integration
    print("\n6. Testing Agent System Integration...")

    # Create agent (as developer - should work)
    agent_result = await integration.secure_api_call(
        system="agents",
        action="create_agent",
        user_id=developer.user_id,
        tenant_id=tenant.tenant_id,
        params={"name": "demo_agent", "type": "king_agent", "config": {"max_memory_mb": 1024}},
    )

    if agent_result.get("status") == 201:
        print(f"   ✅ Agent created: {agent_result['data']['agent_id']}")
        agent_id = agent_result["data"]["agent_id"]

        # Try to execute agent (as regular user - should work)
        execute_result = await integration.secure_api_call(
            system="agents",
            action="execute_agent",
            user_id=regular_user.user_id,
            tenant_id=tenant.tenant_id,
            resource_id=agent_id,
            params={"task": "Hello, world!"},
        )

        if execute_result.get("status") == 200:
            print(f"   ✅ Agent executed: {execute_result['data']['task_id']}")
        else:
            print(f"   ❌ Agent execution failed: {execute_result.get('error')}")

        # Try to delete agent (as regular user - should fail)
        delete_result = await integration.secure_api_call(
            system="agents",
            action="delete_agent",
            user_id=regular_user.user_id,
            tenant_id=tenant.tenant_id,
            resource_id=agent_id,
        )

        if delete_result.get("status") == 403:
            print("   ✅ Agent deletion properly denied for regular user")
        else:
            print("   ❌ Agent deletion should have been denied")
    else:
        print(f"   ❌ Agent creation failed: {agent_result.get('error')}")

    # Test RAG system integration
    print("\n7. Testing RAG System Integration...")

    # Create RAG collection (as data scientist - should work)
    rag_result = await integration.secure_api_call(
        system="rag",
        action="create_collection",
        user_id=data_scientist.user_id,
        tenant_id=tenant.tenant_id,
        params={"name": "demo_knowledge", "config": {"vector_db_type": "faiss"}},
    )

    if rag_result.get("status") == 201:
        print(f"   ✅ RAG collection created: {rag_result['data']['collection_id']}")
        collection_id = rag_result["data"]["collection_id"]

        # Query RAG (as regular user - should work)
        query_result = await integration.secure_api_call(
            system="rag",
            action="query_rag",
            user_id=regular_user.user_id,
            tenant_id=tenant.tenant_id,
            resource_id=collection_id,
            params={"query": "What is AIVillage?"},
        )

        if query_result.get("status") == 200:
            print("   ✅ RAG query executed successfully")
        else:
            print(f"   ❌ RAG query failed: {query_result.get('error')}")
    else:
        print(f"   ❌ RAG collection creation failed: {rag_result.get('error')}")

    # Test P2P network integration
    print("\n8. Testing P2P Network Integration...")

    # Create P2P network (as developer - should work)
    p2p_result = await integration.secure_api_call(
        system="p2p",
        action="create_network",
        user_id=developer.user_id,
        tenant_id=tenant.tenant_id,
        params={"name": "demo_network", "config": {"max_peers": 50}},
    )

    if p2p_result.get("status") == 201:
        print(f"   ✅ P2P network created: {p2p_result['data']['network_id']}")
        network_id = p2p_result["data"]["network_id"]

        # Join network (as regular user - should work)
        join_result = await integration.secure_api_call(
            system="p2p",
            action="join_network",
            user_id=regular_user.user_id,
            tenant_id=tenant.tenant_id,
            resource_id=network_id,
            params={},
        )

        if join_result.get("status") == 200:
            print(f"   ✅ P2P network joined: {join_result['data']['peer_id']}")
        else:
            print(f"   ❌ P2P network join failed: {join_result.get('error')}")
    else:
        print(f"   ❌ P2P network creation failed: {p2p_result.get('error')}")

    # Test tenant isolation
    print("\n9. Testing Tenant Isolation...")

    # Create second tenant
    tenant2 = await rbac.create_tenant(
        name="Another Corp",
        admin_user={"username": "other_admin", "email": "admin@other.corp", "password": "OtherPassword123!"},
    )

    other_user = await rbac.create_user(
        username="eve_other",
        email="eve@other.corp",
        password="EvePassword123!",
        tenant_id=tenant2.tenant_id,
        role=Role.USER,
    )

    # Try to access first tenant's resources from second tenant (should fail)
    cross_tenant_result = await integration.secure_api_call(
        system="agents", action="list_agents", user_id=other_user.user_id, tenant_id=tenant.tenant_id  # Wrong tenant!
    )

    if cross_tenant_result.get("status") == 403:
        print("   ✅ Cross-tenant access properly denied")
    else:
        print("   ❌ Cross-tenant access should have been denied")

    # Test resource usage monitoring
    print("\n10. Testing Resource Usage Monitoring...")

    usage = await tenant_manager.get_tenant_usage(tenant_id=tenant.tenant_id, user_id=developer.user_id)

    print("   Tenant usage report:")
    print(f"   - Agents: {usage['resources']['agents']}")
    print(f"   - RAG collections: {usage['resources']['rag_collections']}")
    print(f"   - P2P networks: {usage['resources']['p2p_networks']}")
    print(f"   - Total storage: {usage['storage']['total_bytes']} bytes")

    # Test audit logging
    print("\n11. Testing Audit Logging...")

    audit_logs = await rbac.get_audit_log(user_id=admin_session.user_id, action_filter="created")

    print(f"   Found {len(audit_logs)} audit log entries:")
    for log in audit_logs[-5:]:  # Show last 5 entries
        print(f"   - {log['timestamp']}: {log['action']} - {log['details']}")

    print("\n🎉 RBAC Demo Complete!")
    print("Successfully demonstrated:")
    print("✅ Multi-tenant isolation")
    print("✅ Role-based permissions")
    print("✅ Agent system integration")
    print("✅ RAG system integration")
    print("✅ P2P network integration")
    print("✅ Resource usage monitoring")
    print("✅ Comprehensive audit logging")
    print("✅ Cross-tenant security enforcement")

    return integration


async def demo_api_server():
    """Demo the API server functionality."""
    print("\n🌐 Starting API Server Demo...")

    try:
        from .rbac_api_server import create_rbac_server

        server = await create_rbac_server()

        print("📡 RBAC API Server Features:")
        print("✅ FastAPI-based REST API")
        print("✅ JWT token authentication")
        print("✅ Role-based authorization")
        print("✅ Multi-tenant resource isolation")
        print("✅ Comprehensive audit logging")
        print("✅ Integration with all AIVillage systems")

        print("\n📋 Available API Endpoints:")
        endpoints = [
            "POST /auth/login - User authentication",
            "POST /auth/refresh - Token refresh",
            "POST /tenants - Create tenant",
            "POST /users - Create user",
            "GET  /agents - List agents",
            "POST /agents - Create agent",
            "POST /agents/{id}/execute - Execute agent",
            "GET  /rag/collections - List RAG collections",
            "POST /rag/collections - Create RAG collection",
            "POST /rag/collections/{id}/query - Query RAG",
            "POST /p2p/networks - Create P2P network",
            "GET  /p2p/networks - List P2P networks",
            "POST /agent-forge/training - Start training",
            "POST /digital-twin - Create digital twin",
            "POST /mobile/devices - Register device",
            "GET  /audit/logs - Get audit logs",
        ]

        for endpoint in endpoints:
            print(f"   {endpoint}")

        print("\n🚀 To start the API server:")
        print("   python -m packages.core.security.rbac_api_server")
        print("   Then visit: http://localhost:8000/docs")

        return server

    except ImportError as e:
        print("⚠️  FastAPI not installed. Install with: pip install fastapi uvicorn")
        print(f"   Error: {e}")
        return None


if __name__ == "__main__":

    async def main():
        # Run RBAC system demo
        await demo_rbac_system()

        # Demo API server
        await demo_api_server()

        print("\n🎯 Production Deployment Notes:")
        print("1. Change JWT secret in config/security/rbac.json")
        print("2. Configure CORS origins for your domain")
        print("3. Set up HTTPS/TLS certificates")
        print("4. Configure database backup procedures")
        print("5. Set up monitoring and alerting")
        print("6. Review and adjust resource quotas")
        print("7. Enable MFA for admin users")
        print("8. Configure IP restrictions if needed")

    asyncio.run(main())
