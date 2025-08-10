#!/usr/bin/env python3
"""Test script to evaluate agent capabilities and inter-agent communication."""

import asyncio
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.communications.message import Message, MessageType
    from src.communications.protocol import CommunicationsProtocol
    from src.production.agent_forge.agent_factory import AgentFactory
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative imports...")
    try:
        # Try importing from agent_forge directly
        from agent_forge.validate_all_agents import validate_all_agents
    except ImportError as e2:
        print(f"Alternative import failed: {e2}")
        sys.exit(1)


def test_agent_creation():
    """Test if agents can be created successfully."""
    print("=" * 60)
    print("TESTING AGENT CREATION")
    print("=" * 60)

    try:
        factory = AgentFactory()
        available_agents = factory.list_available_agents()

        print(f"Found {len(available_agents)} agent templates:")
        for agent in available_agents:
            print(f"  - {agent['id']}: {agent['name']} - {agent['role']}")

        print("\nTesting agent instantiation...")
        results = {}

        for agent_info in available_agents:
            agent_id = agent_info["id"]
            try:
                agent = factory.create_agent(agent_id)
                print(f"  ✓ {agent_id}: Created successfully")
                results[agent_id] = {
                    "created": True,
                    "has_config": hasattr(agent, "config"),
                    "has_name": hasattr(agent, "name"),
                    "has_role": hasattr(agent, "role_description"),
                    "has_process": hasattr(agent, "process"),
                    "agent_class": agent.__class__.__name__,
                }
            except Exception as e:
                print(f"  ✗ {agent_id}: Failed - {e}")
                results[agent_id] = {"created": False, "error": str(e)}

        return results

    except Exception as e:
        print(f"Failed to initialize AgentFactory: {e}")
        return {}


def test_agent_basic_functionality():
    """Test basic agent functionality."""
    print("\n" + "=" * 60)
    print("TESTING AGENT BASIC FUNCTIONALITY")
    print("=" * 60)

    try:
        factory = AgentFactory()

        # Test a few key agents
        test_agents = ["king", "magi", "sage"]

        for agent_id in test_agents:
            try:
                print(f"\nTesting {agent_id} agent...")
                agent = factory.create_agent(agent_id)

                # Test basic process function
                test_task = {"task": "ping", "data": "test"}
                result = agent.process(test_task)
                print(f"  Process result: {result}")

                # Test KPI functionality
                kpi_result = agent.evaluate_kpi()
                print(f"  KPI evaluation: {kpi_result}")

                # Test performance update
                agent.update_performance(
                    {
                        "timestamp": "2025-01-01T12:00:00",
                        "success": True,
                        "metrics": {"test": 1},
                    }
                )
                print(f"  Performance history length: {len(agent.performance_history)}")

            except Exception as e:
                print(f"  ✗ Failed to test {agent_id}: {e}")

    except Exception as e:
        print(f"Failed to test agent functionality: {e}")


async def test_agent_communication():
    """Test inter-agent communication capabilities."""
    print("\n" + "=" * 60)
    print("TESTING INTER-AGENT COMMUNICATION")
    print("=" * 60)

    try:
        # Create a simple communication protocol
        protocol = StandardCommunicationProtocol()

        factory = AgentFactory()

        # Create two agents for communication test
        king_agent = factory.create_agent("king")
        sage_agent = factory.create_agent("sage")

        print("Created agents for communication test")

        # Set up message handlers
        king_responses = []
        sage_responses = []

        async def king_handler(msg):
            result = king_agent.process(msg.content)
            king_responses.append(result)
            print(f"King received and processed: {result}")

        async def sage_handler(msg):
            result = sage_agent.process(msg.content)
            sage_responses.append(result)
            print(f"Sage received and processed: {result}")

        # Subscribe agents to protocol
        protocol.subscribe("king", king_handler)
        protocol.subscribe("sage", sage_handler)

        # Test message passing
        test_message = Message(
            type=MessageType.TASK,
            sender="tester",
            receiver="king",
            content={"task": "coordinate", "details": "Test coordination task"},
        )

        print("Sending test message to King agent...")
        await protocol.send_message(test_message)

        # Give some time for message processing
        await asyncio.sleep(0.1)

        print(f"King responses: {len(king_responses)}")
        print(f"Sage responses: {len(sage_responses)}")

        return len(king_responses) > 0

    except Exception as e:
        print(f"Communication test failed: {e}")
        return False


def main():
    """Run all agent tests."""
    print("AIVillage Agent Capability Analysis")
    print("=" * 60)

    # Test 1: Agent Creation
    creation_results = test_agent_creation()

    # Test 2: Basic Functionality
    test_agent_basic_functionality()

    # Test 3: Communication (async)
    communication_success = asyncio.run(test_agent_communication())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    created_count = sum(1 for r in creation_results.values() if r.get("created", False))
    total_count = len(creation_results)

    print(f"Agent Creation: {created_count}/{total_count} agents created successfully")
    print(f"Communication Test: {'PASSED' if communication_success else 'FAILED'}")

    # Print detailed results
    print("\nDetailed Agent Analysis:")
    for agent_id, result in creation_results.items():
        if result.get("created"):
            status = (
                "✓ REAL IMPLEMENTATION"
                if result.get("agent_class") != "GenericAgent"
                else "○ GENERIC STUB"
            )
            print(f"  {agent_id}: {status} ({result.get('agent_class', 'Unknown')})")
        else:
            print(f"  {agent_id}: ✗ FAILED - {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
