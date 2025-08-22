#!/usr/bin/env python3
"""
Circular Dependency Checker
Detects circular dependencies in Python modules.
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque


class CircularDependencyDetector:
    """Detects circular dependencies in Python modules."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_graph = defaultdict(set)
        self.file_imports = defaultdict(set)
        self.circular_deps = []
        
    def analyze_project(self) -> List[Dict]:
        """Analyze the project for circular dependencies."""
        # Build dependency graph
        self._build_dependency_graph()
        
        # Detect circular dependencies
        self._detect_circular_dependencies()
        
        return self.circular_deps
    
    def _build_dependency_graph(self):
        """Build the dependency graph from Python files."""
        python_files = list(self.project_root.rglob("*.py"))
        
        # Exclude common directories
        excluded_dirs = {'__pycache__', '.git', 'venv', 'env', 'node_modules', 
                        'deprecated', 'archive', '.pytest_cache', 'build', 'dist'}
        
        python_files = [
            f for f in python_files 
            if not any(part in excluded_dirs for part in f.parts)
        ]
        
        for py_file in python_files:
            try:
                imports = self._extract_imports(py_file)
                module_name = self._get_module_name(py_file)
                
                self.file_imports[module_name] = imports
                
                for imported_module in imports:
                    # Only track internal dependencies
                    if self._is_internal_module(imported_module):
                        self.dependency_graph[module_name].add(imported_module)
                        
            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}", file=sys.stderr)
    
    def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                        
        except Exception:
            # Skip files that can't be parsed
            pass
            
        return imports
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get the module name from a file path."""
        relative_path = file_path.relative_to(self.project_root)
        
        # Remove .py extension
        if relative_path.suffix == '.py':
            relative_path = relative_path.with_suffix('')
        
        # Convert path to module name
        parts = relative_path.parts
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts) if parts else ''
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Check if the module exists as a file or directory in the project
        module_path = self.project_root / module_name.replace('.', os.sep)
        
        return (
            (module_path.with_suffix('.py')).exists() or
            (module_path / '__init__.py').exists() or
            module_path.is_dir()
        )
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            """Depth-first search to detect cycles."""
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self._add_circular_dependency(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph[node]:
                if dfs(neighbor, path + [node]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Run DFS from each node
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, [])
    
    def _add_circular_dependency(self, cycle: List[str]):
        """Add a circular dependency to the results."""
        # Normalize the cycle (start from the lexicographically smallest)
        min_idx = cycle.index(min(cycle[:-1]))  # Exclude the last duplicate
        normalized_cycle = cycle[min_idx:-1] + cycle[:min_idx]
        
        # Check if we already have this cycle
        cycle_key = tuple(normalized_cycle)
        if not any(tuple(cd['modules']) == cycle_key for cd in self.circular_deps):
            
            # Calculate severity based on cycle length and import types
            severity = self._calculate_cycle_severity(normalized_cycle)
            
            self.circular_deps.append({
                'modules': normalized_cycle,
                'cycle_length': len(normalized_cycle),
                'severity': severity,
                'description': self._describe_cycle(normalized_cycle),
                'suggestions': self._suggest_solutions(normalized_cycle)
            })
    
    def _calculate_cycle_severity(self, cycle: List[str]) -> str:
        """Calculate the severity of a circular dependency."""
        cycle_length = len(cycle)
        
        # Shorter cycles are generally more problematic
        if cycle_length == 2:
            return "high"  # Direct circular dependency
        elif cycle_length <= 4:
            return "medium"  # Short indirect cycle
        else:
            return "low"  # Long indirect cycle
    
    def _describe_cycle(self, cycle: List[str]) -> str:
        """Generate a human-readable description of the cycle."""
        if len(cycle) == 2:
            return f"Direct circular dependency between {cycle[0]} and {cycle[1]}"
        else:
            cycle_str = " → ".join(cycle + [cycle[0]])
            return f"Circular dependency chain: {cycle_str}"
    
    def _suggest_solutions(self, cycle: List[str]) -> List[str]:
        """Suggest solutions for breaking the circular dependency."""
        suggestions = []
        
        if len(cycle) == 2:
            suggestions.extend([
                "Extract common functionality into a separate module",
                "Use dependency injection to break the direct coupling",
                "Implement one module as an interface/protocol",
                "Move shared constants or types to a common module"
            ])
        else:
            suggestions.extend([
                "Identify the least essential dependency in the cycle",
                "Extract common abstractions to break the chain",
                "Use event-driven architecture to decouple modules",
                "Implement dependency inversion principle",
                "Consider restructuring the module hierarchy"
            ])
        
        # Add specific suggestions based on module names
        for module in cycle:
            if 'test' in module.lower():
                suggestions.append("Move test utilities to a separate test helper module")
            elif 'config' in module.lower():
                suggestions.append("Extract configuration to a dedicated config module")
            elif 'util' in module.lower() or 'helper' in module.lower():
                suggestions.append("Split utility modules by functional domain")
        
        return suggestions
    
    def find_dependency_path(self, from_module: str, to_module: str) -> Optional[List[str]]:
        """Find the shortest dependency path between two modules."""
        if from_module not in self.dependency_graph:
            return None
        
        queue = deque([(from_module, [from_module])])
        visited = {from_module}
        
        while queue:
            current, path = queue.popleft()
            
            if current == to_module:
                return path
            
            for neighbor in self.dependency_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_dependency_statistics(self) -> Dict:
        """Get statistics about the dependency graph."""
        total_modules = len(self.dependency_graph)
        total_dependencies = sum(len(deps) for deps in self.dependency_graph.values())
        
        # Calculate in-degree and out-degree for each module
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        
        for module, deps in self.dependency_graph.items():
            out_degrees[module] = len(deps)
            for dep in deps:
                in_degrees[dep] += 1
        
        # Find modules with high fan-in/fan-out
        high_fan_in = [mod for mod, degree in in_degrees.items() if degree > 5]
        high_fan_out = [mod for mod, degree in out_degrees.items() if degree > 5]
        
        return {
            'total_modules': total_modules,
            'total_dependencies': total_dependencies,
            'circular_dependencies': len(self.circular_deps),
            'average_dependencies_per_module': total_dependencies / max(total_modules, 1),
            'modules_with_high_fan_in': high_fan_in,
            'modules_with_high_fan_out': high_fan_out,
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'max_out_degree': max(out_degrees.values()) if out_degrees else 0
        }
    
    def format_results(self, show_details: bool = True) -> str:
        """Format the circular dependency detection results."""
        if not self.circular_deps:
            return "✅ No circular dependencies detected!"
        
        output = []
        output.append(f"❌ Found {len(self.circular_deps)} circular dependency issue(s):")
        output.append("")
        
        # Sort by severity
        severity_order = {"high": 3, "medium": 2, "low": 1}
        sorted_deps = sorted(
            self.circular_deps,
            key=lambda x: (severity_order.get(x['severity'], 0), x['cycle_length']),
            reverse=True
        )
        
        severity_emojis = {
            "high": "🚨",
            "medium": "⚠️",
            "low": "📝"
        }
        
        for i, cycle_info in enumerate(sorted_deps, 1):
            severity = cycle_info['severity']
            emoji = severity_emojis.get(severity, "📝")
            
            output.append(f"{emoji} **Issue #{i} - {severity.upper()} Severity**")
            output.append(f"   {cycle_info['description']}")
            
            if show_details:
                output.append("   Modules involved:")
                for module in cycle_info['modules']:
                    output.append(f"     - {module}")
                
                output.append("   Suggested solutions:")
                for suggestion in cycle_info['suggestions'][:3]:  # Show first 3 suggestions
                    output.append(f"     • {suggestion}")
            
            output.append("")
        
        # Add dependency statistics
        stats = self.get_dependency_statistics()
        output.append("📊 **Dependency Statistics:**")
        output.append(f"   - Total modules analyzed: {stats['total_modules']}")
        output.append(f"   - Total dependencies: {stats['total_dependencies']}")
        output.append(f"   - Average dependencies per module: {stats['average_dependencies_per_module']:.1f}")
        
        if stats['modules_with_high_fan_in']:
            output.append(f"   - Modules with high fan-in (>5): {len(stats['modules_with_high_fan_in'])}")
        
        if stats['modules_with_high_fan_out']:
            output.append(f"   - Modules with high fan-out (>5): {len(stats['modules_with_high_fan_out'])}")
        
        return "\n".join(output)


def main():
    """Main entry point for circular dependency detection."""
    parser = argparse.ArgumentParser(description="Detect circular dependencies in Python project")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Root directory of the project to analyze")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--count", action="store_true",
                       help="Output only the count of circular dependencies")
    parser.add_argument("--details", action="store_true", default=True,
                       help="Show detailed information about cycles")
    parser.add_argument("--find-path", nargs=2, metavar=("FROM", "TO"),
                       help="Find dependency path between two modules")
    
    args = parser.parse_args()
    
    # Initialize detector
    project_root = Path(args.project_root).resolve()
    detector = CircularDependencyDetector(project_root)
    
    # Analyze project
    circular_deps = detector.analyze_project()
    
    # Handle specific operations
    if args.find_path:
        from_module, to_module = args.find_path
        path = detector.find_dependency_path(from_module, to_module)
        if path:
            print(f"Dependency path: {' → '.join(path)}")
        else:
            print(f"No dependency path found from {from_module} to {to_module}")
        return
    
    # Output results
    if args.count:
        print(len(circular_deps))
    elif args.json:
        import json
        results = {
            'circular_dependencies': circular_deps,
            'statistics': detector.get_dependency_statistics()
        }
        print(json.dumps(results, indent=2))
    else:
        print(detector.format_results(show_details=args.details))
    
    # Exit with error code if circular dependencies found
    if circular_deps:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()