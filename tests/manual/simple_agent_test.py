#!/usr/bin/env python3
"""Simple script to test agent implementations."""

import os
import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYTHONIOENCODING"] = "utf-8"


def test_agent_templates():
    """Test if agent templates exist and can be loaded."""
    print("=" * 50)
    print("AGENT TEMPLATE ANALYSIS")
    print("=" * 50)

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        # Initialize factory
        factory = AgentFactory()

        # Get available agents
        agents = factory.list_available_agents()
        print(f"Found {len(agents)} agent templates:")

        for agent in agents:
            print(f"  {agent['id']}: {agent['name']}")
            print(f"    Role: {agent['role']}")

        return agents

    except Exception as e:
        print(f"Error loading agent factory: {e}")
        return []


def test_agent_instantiation(agents):
    """Test creating agents from templates."""
    print("\n" + "=" * 50)
    print("AGENT INSTANTIATION TEST")
    print("=" * 50)

    if not agents:
        print("No agents to test")
        return {}

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        factory = AgentFactory()

        results = {}

        for agent_info in agents:
            agent_id = agent_info["id"]
            try:
                agent = factory.create_agent(agent_id)
                agent_class = agent.__class__.__name__

                # Determine if it's a real implementation or generic
                is_real = agent_class != "GenericAgent"
                status = "REAL" if is_real else "STUB"

                print(f"  {agent_id}: {status} ({agent_class})")

                results[agent_id] = {
                    "created": True,
                    "class": agent_class,
                    "is_real": is_real,
                    "has_config": hasattr(agent, "config"),
                    "has_name": hasattr(agent, "name"),
                }

            except Exception as e:
                print(f"  {agent_id}: FAILED - {str(e)[:60]}")
                results[agent_id] = {"created": False, "error": str(e)}

        return results

    except Exception as e:
        print(f"Error during instantiation: {e}")
        return {}


def test_agent_basic_ops(results):
    """Test basic agent operations."""
    print("\n" + "=" * 50)
    print("BASIC OPERATIONS TEST")
    print("=" * 50)

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        factory = AgentFactory()

        # Test a few key agents
        test_agents = ["king", "magi", "sage"]

        for agent_id in test_agents:
            if agent_id in results and results[agent_id].get("created"):
                try:
                    print(f"\nTesting {agent_id}...")
                    agent = factory.create_agent(agent_id)

                    # Test process method
                    test_task = {"task": "ping", "data": "test"}
                    result = agent.process(test_task)
                    print(f"  Process result: {result.get('status', 'no status')}")

                    # Test KPI method if available
                    if hasattr(agent, "evaluate_kpi"):
                        kpi = agent.evaluate_kpi()
                        print(f"  KPI keys: {list(kpi.keys()) if kpi else 'None'}")

                except Exception as e:
                    print(f"  Error testing {agent_id}: {str(e)[:60]}")

    except Exception as e:
        print(f"Error in basic operations test: {e}")


def check_experimental_agents():
    """Check what's in the experimental agents directory."""
    print("\n" + "=" * 50)
    print("EXPERIMENTAL AGENTS CHECK")
    print("=" * 50)

    exp_path = Path("experimental/agents/agents")
    if exp_path.exists():
        agent_dirs = [
            d for d in exp_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        print(f"Found {len(agent_dirs)} experimental agent directories:")

        for agent_dir in sorted(agent_dirs):
            # Check for main agent file
            agent_files = list(agent_dir.glob("*_agent.py"))
            if agent_files:
                print(f"  {agent_dir.name}: HAS IMPLEMENTATION")
                # Check file size to estimate complexity
                main_file = agent_files[0]
                file_size = main_file.stat().st_size
                print(f"    Main file: {main_file.name} ({file_size} bytes)")
            else:
                print(f"  {agent_dir.name}: NO MAIN FILE")
    else:
        print("Experimental agents directory not found")


def main():
    """Run the comprehensive agent analysis."""
    print("AIVillage Agent System Analysis")

    # Test 1: Templates
    agents = test_agent_templates()

    # Test 2: Instantiation
    results = test_agent_instantiation(agents)

    # Test 3: Basic Operations
    test_agent_basic_ops(results)

    # Test 4: Experimental Agents
    check_experimental_agents()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if results:
        created = sum(1 for r in results.values() if r.get("created", False))
        real_agents = sum(1 for r in results.values() if r.get("is_real", False))
        stub_agents = created - real_agents

        print(f"Total templates: {len(agents)}")
        print(f"Successfully created: {created}")
        print(f"Real implementations: {real_agents}")
        print(f"Generic stubs: {stub_agents}")

        print("\nAgent Status:")
        for agent_id, result in results.items():
            if result.get("created"):
                agent_type = "REAL" if result.get("is_real") else "STUB"
                print(f"  {agent_id}: {agent_type}")
            else:
                print(f"  {agent_id}: FAILED")
    else:
        print("No agents could be tested")


if __name__ == "__main__":
    main()
