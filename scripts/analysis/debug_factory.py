#!/usr/bin/env python3
"""Debug the AgentFactory template loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))


def debug_agent_factory():
    """Debug why AgentFactory isn't finding templates."""
    print("=" * 50)
    print("DEBUGGING AGENT FACTORY")
    print("=" * 50)

    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        # Try with default path first
        print("1. Testing with default path...")
        factory1 = AgentFactory()
        print(f"   Template dir: {factory1.template_dir}")
        print(f"   Templates found: {len(factory1.templates)}")
        print(f"   Template keys: {list(factory1.templates.keys())}")

        # Try with absolute path
        print("\n2. Testing with absolute path...")
        abs_template_path = Path.cwd() / "src" / "production" / "agent_forge" / "templates"
        print(f"   Trying absolute path: {abs_template_path}")
        print(f"   Path exists: {abs_template_path.exists()}")

        if abs_template_path.exists():
            factory2 = AgentFactory(str(abs_template_path))
            print(f"   Templates found: {len(factory2.templates)}")
            print(f"   Template keys: {list(factory2.templates.keys())}")

        # Check what's in the agents directory
        agents_dir = abs_template_path / "agents"
        print(f"\n3. Checking agents directory: {agents_dir}")
        print(f"   Agents dir exists: {agents_dir.exists()}")

        if agents_dir.exists():
            json_files = list(agents_dir.glob("*.json"))
            print(f"   JSON files found: {len(json_files)}")
            for json_file in json_files[:5]:  # Show first 5
                print(f"     - {json_file.name}")
            if len(json_files) > 5:
                print(f"     ... and {len(json_files) - 5} more")

        return factory2 if abs_template_path.exists() else factory1

    except Exception as e:
        print(f"Error debugging factory: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_working_factory(factory):
    """Test a working factory."""
    if not factory:
        print("No factory to test")
        return

    print("\n" + "=" * 50)
    print("TESTING WORKING FACTORY")
    print("=" * 50)

    if not factory.templates:
        print("No templates loaded")
        return

    # Test creating a few agents
    test_agents = list(factory.templates.keys())[:3]

    for agent_id in test_agents:
        try:
            print(f"\nTesting {agent_id}...")
            agent = factory.create_agent(agent_id)
            print(f"  Class: {agent.__class__.__name__}")
            print(f"  Name: {getattr(agent, 'name', 'no name')}")

            # Test basic functionality
            result = agent.process({"task": "test", "data": "ping"})
            print(f"  Process result: {result}")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run factory debugging."""
    print("AgentFactory Debug Analysis")

    factory = debug_agent_factory()
    test_working_factory(factory)


if __name__ == "__main__":
    main()
