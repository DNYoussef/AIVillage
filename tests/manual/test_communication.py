#!/usr/bin/env python3
"""Test inter-agent communication system."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))


def test_basic_communication():
    """Test the basic communication system."""
    print("=" * 50)
    print("TESTING COMMUNICATION PROTOCOL")
    print("=" * 50)

    try:
        from src.communications.message import Message, MessageType
        from src.communications.protocol import CommunicationsProtocol

        # Create a protocol instance
        CommunicationsProtocol("test_agent", port=8889)
        print("+ Communication protocol created successfully")

        # Test message creation
        message = Message(
            type=MessageType.TASK,
            sender="test_sender",
            receiver="test_receiver",
            content={"task": "ping", "data": "test"},
        )
        print("+ Message created successfully")
        print(f"  Message type: {message.type}")
        print(f"  Sender: {message.sender}")
        print(f"  Receiver: {message.receiver}")

        return True

    except Exception as e:
        print(f"- Communication test failed: {e}")
        return False


async def test_agent_messaging():
    """Test actual agent to agent messaging."""
    print("\n" + "=" * 50)
    print("TESTING AGENT MESSAGING")
    print("=" * 50)

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        factory = AgentFactory()

        # Create two agents
        king_agent = factory.create_agent("king")
        sage_agent = factory.create_agent("sage")

        print("+ Created two agents for messaging test")

        # Test direct processing
        test_message = {"task": "test_coordination", "data": "ping"}

        king_result = king_agent.process(test_message)
        sage_result = sage_agent.process(test_message)

        print(f"King response: {king_result}")
        print(f"Sage response: {sage_result}")

        # Test if they have different behaviors
        if king_result != sage_result:
            print("+ Agents show different behaviors")
            return True
        print("o Agents show identical behaviors (likely generic)")
        return False

    except Exception as e:
        print(f"- Agent messaging test failed: {e}")
        return False


def test_agent_specialization():
    """Test if agents have different specializations."""
    print("\n" + "=" * 50)
    print("TESTING AGENT SPECIALIZATION")
    print("=" * 50)

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        factory = AgentFactory()

        # Test different agent types
        agent_types = ["king", "magi", "sage", "oracle", "maker"]

        specializations = {}

        for agent_type in agent_types:
            try:
                agent = factory.create_agent(agent_type)

                # Check specialization
                if hasattr(agent, "specialization"):
                    spec = agent.specialization
                    specializations[agent_type] = {
                        "role": getattr(spec, "role", "unknown"),
                        "class": agent.__class__.__name__,
                    }
                else:
                    specializations[agent_type] = {
                        "role": "no specialization",
                        "class": agent.__class__.__name__,
                    }

                print(f"{agent_type}: {specializations[agent_type]}")

            except Exception as e:
                print(f"{agent_type}: ERROR - {e}")
                specializations[agent_type] = {"error": str(e)}

        # Check if specializations are different
        unique_roles = set()
        unique_classes = set()

        for spec in specializations.values():
            if "role" in spec:
                unique_roles.add(str(spec["role"]))
            if "class" in spec:
                unique_classes.add(spec["class"])

        print(f"\nUnique roles: {len(unique_roles)}")
        print(f"Unique classes: {len(unique_classes)}")

        return len(unique_roles) > 1 or len(unique_classes) > 1

    except Exception as e:
        print(f"- Specialization test failed: {e}")
        return False


def test_kpi_system():
    """Test the KPI tracking system."""
    print("\n" + "=" * 50)
    print("TESTING KPI SYSTEM")
    print("=" * 50)

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create_agent("king")

        print("+ Agent created for KPI testing")

        # Test initial KPI
        initial_kpi = agent.evaluate_kpi()
        print(f"Initial KPI: {initial_kpi}")

        # Update performance
        agent.update_performance(
            {
                "timestamp": "2024-01-01T12:00:00",
                "success": True,
                "metrics": {"test_score": 0.8},
            }
        )

        # Test updated KPI
        updated_kpi = agent.evaluate_kpi()
        print(f"Updated KPI: {updated_kpi}")

        # Check if KPI system is working
        has_kpi = bool(updated_kpi)
        has_history = bool(agent.performance_history)

        print(f"Has KPI scores: {has_kpi}")
        print(f"Has performance history: {has_history}")

        return has_kpi and has_history

    except Exception as e:
        print(f"- KPI test failed: {e}")
        return False


async def main():
    """Run all communication tests."""
    print("AIVillage Agent Communication Test Suite")

    # Test 1: Basic Communication
    comm_basic = test_basic_communication()

    # Test 2: Agent Messaging
    agent_msg = await test_agent_messaging()

    # Test 3: Specialization
    specialization = test_agent_specialization()

    # Test 4: KPI System
    kpi_system = test_kpi_system()

    # Summary
    print("\n" + "=" * 50)
    print("COMMUNICATION TEST SUMMARY")
    print("=" * 50)

    tests = [
        ("Basic Communication", comm_basic),
        ("Agent Messaging", agent_msg),
        ("Agent Specialization", specialization),
        ("KPI System", kpi_system),
    ]

    passed = sum(1 for _, result in tests if result)

    print(f"Tests passed: {passed}/{len(tests)}")

    for test_name, result in tests:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    # Final assessment
    if passed >= 3:
        print("\n++ Agent communication system is functional!")
    elif passed >= 2:
        print("\n+- Agent system partially functional")
    else:
        print("\n-- Agent system has significant issues")


if __name__ == "__main__":
    asyncio.run(main())
