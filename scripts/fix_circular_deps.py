#!/usr/bin/env python3
"""Detect and resolve circular dependencies in the agent system.
Creates a refactoring plan and implements it.
"""

import ast
from pathlib import Path


def build_dependency_graph() -> dict[str, set[str]]:
    """Build a dictionary of module dependencies."""
    dependencies = {}

    # Focus on key directories to avoid timeout
    target_dirs = ["agents", "agent_forge", "rag_system"]

    for target_dir in target_dirs:
        target_path = Path(target_dir)
        if target_path.exists():
            for py_file in target_path.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    module = str(py_file).replace("\\", ".").replace(".py", "")
                    imports = extract_imports(py_file)
                    dependencies[module] = imports

    return dependencies


def extract_imports(filepath: Path) -> set[str]:
    """Extract all imports from a Python file."""
    imports = set()

    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and any(
                target in node.module
                for target in ["agents", "agent_forge", "rag_system"]
            ):
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    target in alias.name
                    for target in ["agents", "agent_forge", "rag_system"]
                ):
                    imports.add(alias.name)

    return imports


def find_circular_dependencies(dependencies: dict[str, set[str]]) -> list[list[str]]:
    """Find circular dependencies using simple cycle detection."""
    cycles = []

    def has_path(start: str, end: str, visited: set[str]) -> bool:
        if start == end:
            return True
        if start in visited:
            return False

        visited.add(start)
        for dep in dependencies.get(start, set()):
            if has_path(dep, end, visited.copy()):
                return True
        return False

    # Check for simple A->B->A cycles
    for module, deps in dependencies.items():
        for dep in deps:
            if dep in dependencies and has_path(dep, module, set()):
                cycle = [module, dep]
                if cycle not in cycles and [dep, module] not in cycles:
                    cycles.append(cycle)

    return cycles


def create_interfaces_solution(cycles: list[list[str]]) -> dict[str, str]:
    """Create interface-based solution for circular dependencies."""
    interfaces = {}

    # Create abstract base classes for each agent type
    agent_interface_template = '''"""
Abstract interface for {agent_type} agent.
This breaks circular dependencies by defining contracts without implementation.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class {agent_type}Interface(ABC):
    """Interface for {agent_type} agent functionality."""

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input according to {agent_type} specialization."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status."""
        pass
'''

    # Identify unique agents in cycles
    agents_in_cycles = set()
    for cycle in cycles:
        for module in cycle:
            if "agents" in module:
                parts = module.split(".")
                for part in parts:
                    if part in ["king", "sage", "magi", "base"]:
                        agents_in_cycles.add(part)

    # Generate interfaces
    for agent in agents_in_cycles:
        interface_content = agent_interface_template.format(
            agent_type=agent.title(),
        )
        interfaces[f"agents/interfaces/{agent}_interface.py"] = interface_content

    return interfaces


def generate_circular_deps_report(
    cycles: list[list[str]], interfaces: dict[str, str]
) -> str:
    """Generate a report on circular dependencies and solutions."""
    report = "# Circular Dependencies Analysis Report\n\n"
    report += f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    if cycles:
        report += f"## Found {len(cycles)} Circular Dependencies\n\n"
        for i, cycle in enumerate(cycles, 1):
            report += f"### Cycle {i}: {' -> '.join(cycle + [cycle[0]])}\n\n"
            report += "This creates a circular import dependency that can cause:\n"
            report += "- Import errors at runtime\n"
            report += "- Difficult testing and mocking\n"
            report += "- Tight coupling between modules\n\n"
    else:
        report += "## No Circular Dependencies Found\n\n"
        report += "The agent system appears to have a clean dependency structure.\n\n"

    if interfaces:
        report += "## Proposed Interface Solution\n\n"
        report += "Created abstract interfaces to break circular dependencies:\n\n"
        for filepath in interfaces:
            report += f"- `{filepath}`\n"

        report += "\n### Implementation Steps\n\n"
        report += "1. Create interface files in `agents/interfaces/`\n"
        report += "2. Update agent implementations to inherit from interfaces\n"
        report += "3. Replace direct imports with interface imports where possible\n"
        report += "4. Use dependency injection for concrete implementations\n\n"

    return report


# Execute the analysis and fixes
if __name__ == "__main__":
    print("Analyzing agent dependencies...")
    dependencies = build_dependency_graph()
    cycles = find_circular_dependencies(dependencies)

    print(f"Analyzed {len(dependencies)} modules")

    if cycles:
        print(f"Found {len(cycles)} circular dependency cycles:")
        for cycle in cycles:
            print(f"  {' -> '.join(cycle + [cycle[0]])}")

        # Create interface solution
        interfaces = create_interfaces_solution(cycles)

        # Create interfaces directory
        Path("agents/interfaces").mkdir(exist_ok=True)

        # Write interface files
        created_files = []
        for filepath, content in interfaces.items():
            Path(filepath).write_text(content, encoding="utf-8")
            created_files.append(filepath)
            print(f"Created {filepath}")

        # Generate report
        report = generate_circular_deps_report(cycles, interfaces)
        with open("circular_dependencies_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nCreated {len(created_files)} interface files")
        print("Generated circular_dependencies_report.md")
        print("\nNext step: Update agent implementations to use interfaces")
    else:
        print("No circular dependencies found!")

        # Still generate report for documentation
        report = generate_circular_deps_report(cycles, {})
        with open("circular_dependencies_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("Generated circular_dependencies_report.md")
