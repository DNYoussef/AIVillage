#!/usr/bin/env python3
"""
Experimental Code Validator

This script validates code stability before allowing it to be moved to production areas.
It checks for:
- Test coverage
- Code quality metrics
- Documentation standards
- API stability markers
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class ExperimentalValidator:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.validation_results = {}

        # Stability criteria
        self.production_requirements = {
            "min_test_coverage": 85,
            "max_complexity": 8,
            "requires_docstrings": True,
            "requires_type_hints": True,
            "max_todo_comments": 0,  # Zero tolerance for production
            "max_fixme_comments": 0,  # Zero tolerance for production
            "requires_performance_benchmarks": True,
            "requires_integration_tests": True,
            "min_api_stability_weeks": 4,  # API must be stable for 4 weeks
            "requires_operational_monitoring": True
        }

        self.experimental_markers = [
            "# EXPERIMENTAL",
            "# TODO",
            "# FIXME",
            "# HACK",
            "# PROTOTYPE",
            "raise NotImplementedError",
            "pass  # placeholder"
        ]

    def analyze_file_stability(self, file_path: Path) -> Dict:
        """Analyze a Python file for stability indicators."""
        if not file_path.suffix == '.py':
            return {"stable": True, "issues": []}

        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except Exception as e:
            return {"stable": False, "issues": [f"Parse error: {str(e)}"]}

        issues = []

        # Check for experimental markers
        experimental_markers_found = []
        for marker in self.experimental_markers:
            if marker in content:
                experimental_markers_found.append(marker)

        if experimental_markers_found:
            issues.append(f"Experimental markers found: {experimental_markers_found}")

        # Check for docstrings on public functions/classes
        missing_docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):  # Public function/class
                    if not ast.get_docstring(node):
                        missing_docstrings.append(f"{node.name} ({node.lineno})")

        if missing_docstrings and len(missing_docstrings) > 3:
            issues.append(f"Missing docstrings: {missing_docstrings[:3]}...")

        # Check for type hints (simplified check)
        functions_without_hints = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                has_return_hint = node.returns is not None
                has_arg_hints = any(arg.annotation for arg in node.args.args)
                if not (has_return_hint or has_arg_hints):
                    functions_without_hints.append(node.name)

        if functions_without_hints and len(functions_without_hints) > 5:
            issues.append(f"Missing type hints: {functions_without_hints[:3]}...")

        # Check complexity (simplified - count nested levels)
        max_nesting = self.calculate_max_nesting(tree)
        if max_nesting > 4:
            issues.append(f"High nesting complexity: {max_nesting} levels")

        # Determine stability
        is_stable = len(issues) <= 2 and not any("EXPERIMENTAL" in str(issue) for issue in issues)

        return {
            "stable": is_stable,
            "issues": issues,
            "experimental_markers": experimental_markers_found,
            "max_nesting": max_nesting
        }

    def calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth in AST."""
        max_depth = 0

        def visit_node(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            # Count nested structures
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth += 1

            for child in ast.iter_child_nodes(node):
                visit_node(child, depth)

        visit_node(tree)
        return max_depth

    def validate_directory(self, dir_path: Path) -> Dict:
        """Validate all files in a directory for stability."""
        if not dir_path.exists() or not dir_path.is_dir():
            return {"stable": True, "files": {}, "summary": "Directory not found"}

        file_results = {}
        total_files = 0
        stable_files = 0

        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            result = self.analyze_file_stability(py_file)
            file_results[str(py_file.relative_to(self.base_path))] = result

            total_files += 1
            if result["stable"]:
                stable_files += 1

        stability_ratio = stable_files / total_files if total_files > 0 else 1.0
        is_directory_stable = stability_ratio >= 0.8  # 80% of files must be stable

        return {
            "stable": is_directory_stable,
            "files": file_results,
            "summary": {
                "total_files": total_files,
                "stable_files": stable_files,
                "stability_ratio": stability_ratio
            }
        }

    def validate_component_for_production(self, component_path: str) -> bool:
        """Validate if a component is ready for production src/ directory."""
        full_path = self.base_path / component_path

        if not full_path.exists():
            logger.warning(f"Component not found: {component_path}")
            return False

        validation_result = self.validate_directory(full_path)
        self.validation_results[component_path] = validation_result

        if validation_result["stable"]:
            logger.info(f"✓ Component '{component_path}' is stable for production")
            return True
        else:
            logger.warning(f"✗ Component '{component_path}' is not stable for production")
            logger.warning(f"  Stability ratio: {validation_result['summary']['stability_ratio']:.2%}")
            return False

    def get_production_ready_components(self) -> Tuple[List[str], List[str]]:
        """Get lists of production-ready and experimental components."""
        components_to_check = [
            "production",
            "digital_twin",
            "mcp_servers",
            "agent_forge/core",
            "agent_forge/evaluation",
            "agent_forge/self_awareness",
            "agent_forge/bakedquietiot"
        ]

        production_ready = []
        experimental_only = []

        for component in components_to_check:
            if self.validate_component_for_production(component):
                production_ready.append(component)
            else:
                experimental_only.append(component)

        return production_ready, experimental_only

    def generate_validation_report(self) -> Dict:
        """Generate a comprehensive validation report."""
        production_ready, experimental_only = self.get_production_ready_components()

        report = {
            "validation_summary": {
                "total_components_checked": len(self.validation_results),
                "production_ready_count": len(production_ready),
                "experimental_only_count": len(experimental_only)
            },
            "production_ready_components": production_ready,
            "experimental_only_components": experimental_only,
            "detailed_results": self.validation_results,
            "recommendations": self.generate_recommendations(experimental_only)
        }

        return report

    def generate_recommendations(self, experimental_components: List[str]) -> List[str]:
        """Generate recommendations for making experimental components production-ready."""
        recommendations = []

        for component in experimental_components:
            if component in self.validation_results:
                result = self.validation_results[component]
                issues = []

                for file_path, file_result in result["files"].items():
                    issues.extend(file_result["issues"])

                # Generate specific recommendations based on common issues
                if any("docstring" in issue for issue in issues):
                    recommendations.append(f"{component}: Add comprehensive docstrings to public APIs")

                if any("type hint" in issue for issue in issues):
                    recommendations.append(f"{component}: Add type hints to function signatures")

                if any("EXPERIMENTAL" in issue for issue in issues):
                    recommendations.append(f"{component}: Remove experimental markers and implement TODOs")

                if any("complexity" in issue for issue in issues):
                    recommendations.append(f"{component}: Refactor complex functions to reduce nesting")

                # Add graduation pipeline recommendations
                recommendations.append(f"{component}: Implement performance benchmarks")
                recommendations.append(f"{component}: Add integration tests with existing production components")
                recommendations.append(f"{component}: Ensure API stability for minimum 4 weeks")
                recommendations.append(f"{component}: Add operational monitoring and alerting")

        return recommendations

    def get_graduation_readiness(self, component_path: str) -> Dict:
        """Assess graduation readiness with detailed scoring."""
        if not self.validate_component_for_production(component_path):
            return {"ready": False, "score": 0, "blockers": ["Basic stability requirements not met"]}

        # Advanced readiness checks
        readiness_score = 0
        max_score = 100
        blockers = []

        # Check for performance benchmarks (20 points)
        benchmark_files = list((self.base_path / component_path).rglob("*benchmark*.py"))
        if benchmark_files:
            readiness_score += 20
        else:
            blockers.append("Missing performance benchmarks")

        # Check for integration tests (25 points)
        integration_test_files = list((self.base_path / component_path).rglob("*integration*test*.py"))
        if integration_test_files:
            readiness_score += 25
        else:
            blockers.append("Missing integration tests")

        # Check for monitoring setup (15 points)
        monitoring_files = list((self.base_path / component_path).rglob("*monitor*.py"))
        if monitoring_files:
            readiness_score += 15
        else:
            blockers.append("Missing operational monitoring")

        # Base stability (40 points - already validated)
        if self.validate_component_for_production(component_path):
            readiness_score += 40

        return {
            "ready": readiness_score >= 85 and len(blockers) == 0,
            "score": readiness_score,
            "blockers": blockers,
            "max_score": max_score
        }

    def check_import_leakage(self) -> Dict:
        """Check for any experimental imports in production code."""
        src_path = self.base_path / "src"
        production_path = self.base_path / "production"

        leakage_found = []

        for prod_dir in [src_path, production_path]:
            if not prod_dir.exists():
                continue

            for py_file in prod_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    content = py_file.read_text(encoding='utf-8')

                    # Check for experimental imports
                    if "from experimental" in content or "import experimental" in content:
                        leakage_found.append(f"{py_file}: Contains experimental imports")

                except Exception as e:
                    logger.warning(f"Could not read {py_file}: {e}")

        return {
            "clean": len(leakage_found) == 0,
            "violations": leakage_found
        }

    def should_move_to_production(self, component_path: str) -> bool:
        """Final decision on whether component should move to src/ or experimental/."""
        # Basic stability check
        if not self.validate_component_for_production(component_path):
            return False

        # Advanced graduation readiness check
        readiness = self.get_graduation_readiness(component_path)
        return readiness["ready"]

if __name__ == "__main__":
    validator = ExperimentalValidator(os.getcwd())
    report = validator.generate_validation_report()

    print("=== Experimental Code Validation Report ===")
    print(f"Production Ready: {report['production_ready_components']}")
    print(f"Experimental Only: {report['experimental_only_components']}")

    # Check import leakage
    leakage_check = validator.check_import_leakage()
    print(f"\nImport Leakage Check: {'CLEAN' if leakage_check['clean'] else 'VIOLATIONS FOUND'}")
    if not leakage_check['clean']:
        for violation in leakage_check['violations']:
            print(f"  - {violation}")

    # Show graduation readiness for experimental components
    print("\n=== Graduation Readiness Assessment ===")
    experimental_components = ['experimental/agents', 'experimental/federated', 'experimental/mesh']
    for component in experimental_components:
        readiness = validator.get_graduation_readiness(component)
        status = "READY" if readiness['ready'] else "NOT READY"
        print(f"{component}: {status} (Score: {readiness['score']}/{readiness['max_score']})")
        if readiness['blockers']:
            for blocker in readiness['blockers']:
                print(f"  - {blocker}")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
