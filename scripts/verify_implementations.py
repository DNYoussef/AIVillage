#!/usr/bin/env python3
"""Comprehensive verification of stub elimination sprint
Tests all implemented components to ensure they actually work
"""
import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Track test results
test_results = {"passed": 0, "failed": 0, "errors": []}


def test_passed(test_name):
    test_results["passed"] += 1
    print(f"[PASS] {test_name}")


def test_failed(test_name, error):
    test_results["failed"] += 1
    test_results["errors"].append(f"{test_name}: {error}")
    print(f"[FAIL] {test_name}: {error}")


async def test_communications_protocol():
    """Test that communications protocol actually works"""
    print("\n=== TESTING COMMUNICATIONS PROTOCOL ===")

    try:
        from communications.protocol import CommunicationsProtocol

        # Test 1: Can create protocol instances
        agent1 = CommunicationsProtocol("test_agent_1", port=9001)
        agent2 = CommunicationsProtocol("test_agent_2", port=9002)
        test_passed("Create protocol instances")

        # Test 2: Can start servers
        await agent1.start_server()
        await agent2.start_server()
        await asyncio.sleep(0.5)  # Give servers time to start
        test_passed("Start servers")

        # Test 3: Can connect agents
        connected = await agent1.connect("ws://localhost:9002", "test_agent_2")
        if connected:
            test_passed("Connect agents")
        else:
            test_failed("Connect agents", "Connection failed")

        # Test 4: Can send messages
        if connected:
            message_sent = await agent1.send_message("test_agent_2", {"type": "test", "content": "Hello from agent 1"})
            if message_sent:
                test_passed("Send message")
            else:
                test_failed("Send message", "Message send failed")

        # Test 5: Check encryption is working
        test_msg = {"secret": "data"}
        encrypted = agent1._encrypt_message("test_agent_2", test_msg)
        if "secret" not in encrypted and "data" not in encrypted:
            test_passed("Message encryption")
        else:
            test_failed("Message encryption", "Message not encrypted")

        # Cleanup
        await agent1.stop_server()
        await agent2.stop_server()

    except Exception as e:
        test_failed("Communications protocol", str(e))


async def test_whatsapp_connector():
    """Test WhatsApp connector functionality"""
    print("\n=== TESTING WHATSAPP CONNECTOR ===")

    try:
        from ingestion.connectors.whatsapp import WhatsAppConnector

        # Test 1: Can create connector
        connector = WhatsAppConnector()
        test_passed("Create WhatsApp connector")

        # Test 2: Get auth URL (should not be empty)
        auth_url = connector.get_auth_url()
        if auth_url and len(auth_url) > 10:
            test_passed("Get auth URL")
        else:
            test_failed("Get auth URL", "Empty or invalid URL")

        # Test 3: Get message count (should not be 0)
        count = connector.get_message_count()
        if count > 0:
            test_passed(f"Get message count: {count}")
        else:
            test_failed("Get message count", "Count is 0")

        # Test 4: Get messages (should not be empty)
        messages = connector.get_messages(5)
        if len(messages) > 0:
            test_passed(f"Get messages: {len(messages)} messages")

            # Test 5: Verify message structure
            msg = messages[0]
            required_fields = ["id", "from", "text", "timestamp", "type"]
            if all(field in msg for field in required_fields):
                test_passed("Message structure valid")
            else:
                test_failed("Message structure", f"Missing fields in {msg}")
        else:
            test_failed("Get messages", "No messages returned")

        # Test 6: Analyze conversation patterns
        analysis = connector.analyze_conversation_patterns()
        if "total_messages" in analysis and analysis["total_messages"] > 0:
            test_passed(f"Analyze patterns: {analysis['total_messages']} messages analyzed")
        else:
            test_failed("Analyze patterns", "No analysis data")

    except Exception as e:
        test_failed("WhatsApp connector", str(e))


async def test_amazon_connector():
    """Test Amazon orders connector functionality"""
    print("\n=== TESTING AMAZON ORDERS CONNECTOR ===")

    try:
        from ingestion.connectors.amazon_orders import AmazonOrdersConnector

        # Test 1: Can create connector
        connector = AmazonOrdersConnector()
        test_passed("Create Amazon connector")

        # Test 2: Get order count (should not be 0)
        count = connector.get_order_count()
        if count > 0:
            test_passed(f"Get order count: {count}")
        else:
            test_failed("Get order count", "Count is 0")

        # Test 3: Get orders (should not be empty)
        orders = connector.get_orders(5)
        if len(orders) > 0:
            test_passed(f"Get orders: {len(orders)} orders")

            # Test 4: Verify order structure
            order = orders[0]
            required_fields = ["order_id", "title", "price", "category", "order_date"]
            if all(field in order for field in required_fields):
                test_passed("Order structure valid")
            else:
                test_failed("Order structure", f"Missing fields in {order}")

            # Test 5: Verify order has realistic data
            if order["price"] > 0 and len(order["title"]) > 0:
                test_passed(f"Order data realistic: {order['title']} - ${order['price']}")
            else:
                test_failed("Order data", "Invalid price or title")
        else:
            test_failed("Get orders", "No orders returned")

        # Test 6: Analyze purchase patterns
        analysis = connector.analyze_purchase_patterns()
        if "total_orders" in analysis and analysis["total_orders"] > 0:
            test_passed(f"Analyze patterns: ${analysis.get('total_spent', 0):.2f} spent")
        else:
            test_failed("Analyze patterns", "No analysis data")

    except Exception as e:
        test_failed("Amazon connector", str(e))


async def test_ppr_retriever():
    """Test PPR retriever functionality"""
    print("\n=== TESTING PPR RETRIEVER ===")

    try:
        from mcp_servers.hyperag.retrieval.ppr_retriever import PersonalizedPageRank, PPRResults

        # Test 1: Can import and create PPR retriever
        # Create mock dependencies
        class MockHippoIndex:
            async def get_recent_nodes(self, hours, user_id, limit):
                return []

        class MockHypergraphKG:
            pass

        class MockQueryPlan:
            user_id = "test_user"

        hippo = MockHippoIndex()
        hypergraph = MockHypergraphKG()

        ppr = PersonalizedPageRank(hippo_index=hippo, hypergraph=hypergraph, damping=0.85)
        test_passed("Create PPR retriever")

        # Test 2: Can run retrieval
        query_seeds = ["node1", "node2", "node3"]
        plan = MockQueryPlan()

        results = await ppr.retrieve(query_seeds=query_seeds, user_id="test_user", plan=plan, creative_mode=False)

        if isinstance(results, PPRResults):
            test_passed("PPR retrieval returns results")
        else:
            test_failed("PPR retrieval", "Invalid result type")

        # Test 3: Check results structure
        if hasattr(results, "nodes") and hasattr(results, "reasoning_trace"):
            test_passed("PPR results have correct structure")
        else:
            test_failed("PPR results structure", "Missing attributes")

        # Test 4: Check reasoning trace
        if len(results.reasoning_trace) > 0:
            test_passed(f"PPR reasoning trace: {len(results.reasoning_trace)} steps")
        else:
            test_failed("PPR reasoning trace", "Empty trace")

    except Exception as e:
        test_failed("PPR retriever", str(e))


async def test_divergent_retriever():
    """Test divergent retriever functionality"""
    print("\n=== TESTING DIVERGENT RETRIEVER ===")

    try:
        from mcp_servers.hyperag.retrieval.divergent_retriever import DivergentRetriever

        # Test 1: Can import and create divergent retriever
        class MockHypergraphKG:
            pass

        class MockQueryPlan:
            user_id = "test_user"

        hypergraph = MockHypergraphKG()
        divergent = DivergentRetriever(hypergraph=hypergraph)
        test_passed("Create divergent retriever")

        # Test 2: Can run creative retrieval
        query_seeds = ["creativity", "innovation", "art"]
        plan = MockQueryPlan()

        results = await divergent.retrieve_creative(query_seeds=query_seeds, user_id="test_user", plan=plan)

        if results and hasattr(results, "creativity_score"):
            test_passed(f"Divergent retrieval: creativity={results.creativity_score:.2f}")
        else:
            test_failed("Divergent retrieval", "No creativity score")

        # Test 3: Check creative nodes generated
        if len(results.nodes) > 0:
            test_passed(f"Generated {len(results.nodes)} creative nodes")

            # Test 4: Verify node types
            node = results.nodes[0]
            if "type" in node and "creativity_factor" in node:
                test_passed(f"Creative node type: {node['type']}")
            else:
                test_failed("Creative node structure", "Missing type or creativity_factor")
        else:
            test_failed("Creative nodes", "No nodes generated")

        # Test 5: Check surprise factor
        if results.surprise_factor > 0:
            test_passed(f"Surprise factor: {results.surprise_factor:.2f}")
        else:
            test_failed("Surprise factor", "No surprise calculated")

    except Exception as e:
        test_failed("Divergent retriever", str(e))


async def test_system_health_dashboard():
    """Test system health dashboard functionality"""
    print("\n=== TESTING SYSTEM HEALTH DASHBOARD ===")

    try:
        from monitoring.system_health_dashboard import ComponentHealthChecker, SystemHealthDashboard

        # Test 1: Can create dashboard
        dashboard = SystemHealthDashboard()
        test_passed("Create system health dashboard")

        # Test 2: Can create health checker
        checker = ComponentHealthChecker()
        test_passed("Create component health checker")

        # Test 3: Can check a component
        test_file = Path(__file__).parent / "src" / "communications" / "protocol.py"
        if test_file.exists():
            health = checker.check_component_health(test_file, "test_component")
            if "health_score" in health and health["health_score"] > 0:
                test_passed(f"Check component health: {health['health_score']:.1%}")
            else:
                test_failed("Check component health", "Invalid health score")

        # Test 4: Can generate dashboard data
        dashboard_data = await dashboard.generate_dashboard()
        if "overall_health" in dashboard_data:
            overall = dashboard_data["overall_health"]
            test_passed(f"Generate dashboard: {overall['completion_percentage']:.1f}% complete")
        else:
            test_failed("Generate dashboard", "No overall health data")

        # Test 5: Verify sprint success
        if dashboard_data.get("sprint_success", False):
            test_passed("Sprint success verified (>60% completion)")
        else:
            test_failed("Sprint success", "Completion below 60%")

    except Exception as e:
        test_failed("System health dashboard", str(e))


async def main():
    """Run all verification tests"""
    print("=" * 60)
    print("VERIFYING STUB ELIMINATION SPRINT IMPLEMENTATIONS")
    print("=" * 60)

    start_time = time.time()

    # Run all tests
    await test_communications_protocol()
    await test_whatsapp_connector()
    await test_amazon_connector()
    await test_ppr_retriever()
    await test_divergent_retriever()
    await test_system_health_dashboard()

    # Summary
    elapsed = time.time() - start_time
    total_tests = test_results["passed"] + test_results["failed"]

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {test_results['passed']} ({test_results['passed']/total_tests*100:.1f}%)")
    print(f"Failed: {test_results['failed']} ({test_results['failed']/total_tests*100:.1f}%)")
    print(f"Time: {elapsed:.2f} seconds")

    if test_results["failed"] > 0:
        print("\nFailed Tests:")
        for error in test_results["errors"]:
            print(f"  - {error}")
    else:
        print("\nALL TESTS PASSED! The implementations are working correctly.")

    # Save results
    results_file = Path(__file__).parent / "verification_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": total_tests,
                "passed": test_results["passed"],
                "failed": test_results["failed"],
                "success_rate": test_results["passed"] / total_tests * 100,
                "errors": test_results["errors"],
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")

    return test_results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
