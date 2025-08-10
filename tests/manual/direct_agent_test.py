#!/usr/bin/env python3
"""Direct test of agent templates and implementations."""

import json
import sys
from pathlib import Path


def check_agent_templates():
    """Check what agent templates exist."""
    print("=" * 60)
    print("CHECKING AGENT TEMPLATES")
    print("=" * 60)

    template_path = Path("src/production/agent_forge/templates/agents")
    if not template_path.exists():
        print(f"Template directory not found: {template_path}")
        return []

    templates = []
    for template_file in template_path.glob("*.json"):
        try:
            with open(template_file) as f:
                data = json.load(f)

            agent_id = template_file.stem
            templates.append(
                {
                    "id": agent_id,
                    "name": data.get("name", agent_id),
                    "role": data.get("role", ""),
                    "file": str(template_file),
                }
            )
            print(f"  {agent_id}: {data.get('name', 'Unnamed')}")
            print(f"    Role: {data.get('role', 'No role defined')}")

        except Exception as e:
            print(f"  ERROR loading {template_file}: {e}")

    return templates


def check_experimental_implementations():
    """Check experimental agent implementations."""
    print("\n" + "=" * 60)
    print("CHECKING EXPERIMENTAL IMPLEMENTATIONS")
    print("=" * 60)

    exp_path = Path("experimental/agents/agents")
    if not exp_path.exists():
        print("Experimental agents directory not found")
        return {}

    implementations = {}

    # Check specific agent directories
    agent_dirs = ["king", "magi", "sage"]

    for agent_dir in agent_dirs:
        agent_path = exp_path / agent_dir
        if agent_path.exists():
            print(f"\n{agent_dir.upper()} AGENT:")

            # Look for main agent file
            main_files = list(agent_path.glob("*_agent.py"))
            if main_files:
                main_file = main_files[0]
                size = main_file.stat().st_size
                print(f"  Main file: {main_file.name} ({size} bytes)")

                # Try to read first few lines to check complexity
                try:
                    with open(main_file, encoding="utf-8") as f:
                        lines = f.readlines()[:10]

                    # Count imports and classes
                    imports = sum(
                        1
                        for line in lines
                        if line.strip().startswith("import")
                        or line.strip().startswith("from")
                    )
                    classes = sum(
                        1 for line in lines if line.strip().startswith("class")
                    )

                    print(f"  Initial imports: {imports}")
                    print(f"  Classes defined: {classes}")

                    implementations[agent_dir] = {
                        "has_file": True,
                        "size": size,
                        "imports": imports,
                        "classes": classes,
                    }

                except Exception as e:
                    print(f"  Error reading file: {e}")
                    implementations[agent_dir] = {"has_file": True, "error": str(e)}
            else:
                print("  No main agent file found")
                implementations[agent_dir] = {"has_file": False}

            # Check for supporting files
            py_files = list(agent_path.glob("*.py"))
            if py_files:
                print(f"  Total Python files: {len(py_files)}")
                for py_file in py_files[:5]:  # Show first 5
                    print(f"    - {py_file.name}")
                if len(py_files) > 5:
                    print(f"    ... and {len(py_files) - 5} more")

    return implementations


def test_direct_import():
    """Try to import agent classes directly."""
    print("\n" + "=" * 60)
    print("TESTING DIRECT IMPORTS")
    print("=" * 60)

    sys.path.insert(0, str(Path.cwd()))

    # Try importing from experimental
    test_imports = [
        ("experimental.agents.agents.king.king_agent", "KingAgent"),
        ("experimental.agents.agents.magi.magi_agent", "MagiAgent"),
        ("experimental.agents.agents.sage.sage_agent", "SageAgent"),
    ]

    results = {}

    for module_path, class_name in test_imports:
        try:
            print(f"Importing {class_name} from {module_path}...")
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"  SUCCESS: {agent_class}")
            results[class_name] = {"imported": True, "class": agent_class}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[class_name] = {"imported": False, "error": str(e)}

    return results


def validate_agent_system():
    """Use the existing validation system."""
    print("\n" + "=" * 60)
    print("USING EXISTING VALIDATION SYSTEM")
    print("=" * 60)

    try:
        # Try to use the existing validation
        sys.path.insert(0, str(Path.cwd()))
        from agent_forge.validate_all_agents import validate_all_agents

        results = validate_all_agents(full_test=True)

        if results:
            print(f"Validation found {len(results)} agents:")
            for agent_id, checks in results.items():
                status = "PASS" if all(checks.values()) else "FAIL"
                print(f"  {agent_id}: {status}")
                for check, passed in checks.items():
                    symbol = "+" if passed else "-"
                    print(f"    {symbol} {check}")
        else:
            print("No agents found by validation system")

        return results

    except Exception as e:
        print(f"Validation system failed: {e}")
        return {}


def main():
    """Run comprehensive analysis."""
    print("AIVillage Agent System Deep Analysis")

    # Check templates
    templates = check_agent_templates()

    # Check experimental implementations
    implementations = check_experimental_implementations()

    # Test direct imports
    imports = test_direct_import()

    # Use validation system
    validation = validate_agent_system()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"Agent Templates Found: {len(templates)}")
    if templates:
        print("Templates:")
        for t in templates:
            print(f"  - {t['id']}: {t['name']}")

    print(
        f"\nExperimental Implementations: {len([i for i in implementations.values() if i.get('has_file')])}"
    )
    for agent, impl in implementations.items():
        status = "HAS FILE" if impl.get("has_file") else "NO FILE"
        print(f"  - {agent}: {status}")

    print(
        f"\nDirect Imports: {len([i for i in imports.values() if i.get('imported')])}"
    )
    for class_name, result in imports.items():
        status = "SUCCESS" if result.get("imported") else "FAILED"
        print(f"  - {class_name}: {status}")

    print(f"\nValidation Results: {len(validation)} agents")
    if validation:
        working = sum(1 for v in validation.values() if all(v.values()))
        print(f"  - Fully working: {working}")
        print(f"  - With issues: {len(validation) - working}")


if __name__ == "__main__":
    main()
