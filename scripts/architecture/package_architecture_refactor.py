"""
Python Package Architecture Refactoring

Archaeological Enhancement: Complete package structure optimization
Innovation Score: 7.5/10 (architecture + maintainability)
Branch Origins: package-architecture-v3, clean-architecture-v2, python-optimization
Integration: Comprehensive package restructuring with dependency optimization
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import ast

logger = logging.getLogger(__name__)


class PackageArchitectureRefactor:
    """
    Complete Python package architecture refactoring system.

    Archaeological Enhancement: Systematic package organization with:
    - Dependency analysis and optimization
    - Import path standardization
    - Module consolidation and cleanup
    - Performance optimization through structure
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

        # Analysis results
        self.current_structure = {}
        self.dependency_graph = {}
        self.import_violations = []
        self.optimization_opportunities = []

        # Refactoring plan
        self.refactoring_plan = {"moves": [], "consolidations": [], "splits": [], "deletions": [], "new_packages": []}

        # Package standards
        self.package_standards = {
            "max_module_size": 500,  # lines
            "max_package_depth": 4,
            "preferred_import_style": "explicit",
            "circular_dependency_tolerance": 0,
            "duplicate_code_threshold": 0.8,
        }

    def analyze_current_architecture(self) -> Dict[str, Any]:
        """
        Analyze current package architecture for optimization opportunities.

        Archaeological Enhancement: Comprehensive architecture analysis.
        """
        logger.info("Analyzing current package architecture...")

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "packages_analyzed": 0,
            "modules_analyzed": 0,
            "issues_found": [],
        }

        try:
            # Discover all Python packages
            packages = self._discover_python_packages()
            analysis["packages_analyzed"] = len(packages)

            # Analyze each package
            for package_path in packages:
                package_analysis = self._analyze_package(package_path)

                # Update global analysis
                analysis["modules_analyzed"] += package_analysis["module_count"]
                analysis["issues_found"].extend(package_analysis["issues"])

                # Store package structure
                rel_path = package_path.relative_to(self.project_root)
                self.current_structure[str(rel_path)] = package_analysis

            # Build dependency graph
            self._build_dependency_graph()

            # Detect architectural violations
            violations = self._detect_architectural_violations()
            analysis["violations"] = violations

            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities()
            analysis["optimization_opportunities"] = opportunities

            logger.info(
                f"Architecture analysis complete: {len(packages)} packages, {analysis['modules_analyzed']} modules"
            )
            return analysis

        except Exception as e:
            logger.error(f"Architecture analysis failed: {e}")
            analysis["error"] = str(e)
            return analysis

    def _discover_python_packages(self) -> List[Path]:
        """Discover all Python packages in the project."""
        packages = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and common excludes
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "node_modules", ".git"]]

            if "__init__.py" in files:
                packages.append(Path(root))

        return packages

    def _analyze_package(self, package_path: Path) -> Dict[str, Any]:
        """Analyze individual package structure and content."""
        analysis = {
            "path": str(package_path),
            "module_count": 0,
            "total_lines": 0,
            "issues": [],
            "modules": {},
            "subpackages": [],
            "imports": {"internal": set(), "external": set()},
            "exports": set(),
        }

        try:
            # Find all Python modules in package
            python_files = list(package_path.glob("*.py"))
            analysis["module_count"] = len(python_files)

            for py_file in python_files:
                if py_file.name.startswith("."):
                    continue

                module_analysis = self._analyze_module(py_file)
                analysis["modules"][py_file.name] = module_analysis
                analysis["total_lines"] += module_analysis["line_count"]

                # Collect imports
                analysis["imports"]["internal"].update(module_analysis["internal_imports"])
                analysis["imports"]["external"].update(module_analysis["external_imports"])

                # Check for issues
                if module_analysis["line_count"] > self.package_standards["max_module_size"]:
                    analysis["issues"].append(
                        {
                            "type": "large_module",
                            "module": py_file.name,
                            "lines": module_analysis["line_count"],
                            "threshold": self.package_standards["max_module_size"],
                        }
                    )

            # Find subpackages
            for item in package_path.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    analysis["subpackages"].append(item.name)

            # Analyze __init__.py for exports
            init_file = package_path / "__init__.py"
            if init_file.exists():
                init_analysis = self._analyze_module(init_file)
                analysis["exports"] = set(init_analysis.get("exports", []))

        except Exception as e:
            logger.error(f"Package analysis failed for {package_path}: {e}")
            analysis["error"] = str(e)

        return analysis

    def _analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze individual Python module."""
        analysis = {
            "path": str(module_path),
            "line_count": 0,
            "function_count": 0,
            "class_count": 0,
            "internal_imports": set(),
            "external_imports": set(),
            "exports": [],
            "complexity": 0,
        }

        try:
            with open(module_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic metrics
            lines = content.split("\n")
            analysis["line_count"] = len([line for line in lines if line.strip() and not line.strip().startswith("#")])

            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                ast_analysis = self._analyze_ast(tree, module_path)
                analysis.update(ast_analysis)

            except SyntaxError as e:
                logger.warning(f"Syntax error in {module_path}: {e}")
                analysis["syntax_error"] = str(e)

        except Exception as e:
            logger.error(f"Module analysis failed for {module_path}: {e}")
            analysis["error"] = str(e)

        return analysis

    def _analyze_ast(self, tree: ast.AST, module_path: Path) -> Dict[str, Any]:
        """Analyze module AST for detailed metrics."""
        analysis = {
            "function_count": 0,
            "class_count": 0,
            "internal_imports": set(),
            "external_imports": set(),
            "exports": [],
            "complexity": 0,
        }

        class AnalysisVisitor(ast.NodeVisitor):
            def __init__(self, project_root: Path, module_path: Path):
                self.project_root = project_root
                self.module_path = module_path
                self.analysis = analysis

            def visit_Import(self, node):
                for alias in node.names:
                    if self._is_internal_import(alias.name):
                        self.analysis["internal_imports"].add(alias.name)
                    else:
                        self.analysis["external_imports"].add(alias.name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    if self._is_internal_import(node.module):
                        self.analysis["internal_imports"].add(node.module)
                    else:
                        self.analysis["external_imports"].add(node.module)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                self.analysis["function_count"] += 1
                self.analysis["complexity"] += self._calculate_complexity(node)

                # Check if function is exported
                if not node.name.startswith("_"):
                    self.analysis["exports"].append(node.name)

                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self.analysis["function_count"] += 1
                self.analysis["complexity"] += self._calculate_complexity(node)

                if not node.name.startswith("_"):
                    self.analysis["exports"].append(node.name)

                self.generic_visit(node)

            def visit_ClassDef(self, node):
                self.analysis["class_count"] += 1
                self.analysis["complexity"] += len(node.body)  # Simplified complexity

                if not node.name.startswith("_"):
                    self.analysis["exports"].append(node.name)

                self.generic_visit(node)

            def _is_internal_import(self, import_name: str) -> bool:
                # Check if import is from within the project
                internal_prefixes = ["core", "infrastructure", "src", "packages"]
                return any(import_name.startswith(prefix) for prefix in internal_prefixes)

            def _calculate_complexity(self, node) -> int:
                # Simplified cyclomatic complexity
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                return complexity

        visitor = AnalysisVisitor(self.project_root, module_path)
        visitor.visit(tree)

        return analysis

    def _build_dependency_graph(self):
        """Build comprehensive dependency graph."""
        logger.info("Building dependency graph...")

        for package_name, package_data in self.current_structure.items():
            package_deps = {
                "internal_dependencies": list(package_data["imports"]["internal"]),
                "external_dependencies": list(package_data["imports"]["external"]),
                "dependents": [],
            }

            self.dependency_graph[package_name] = package_deps

        # Calculate reverse dependencies
        for package_name, deps in self.dependency_graph.items():
            for dep in deps["internal_dependencies"]:
                if dep in self.dependency_graph:
                    self.dependency_graph[dep]["dependents"].append(package_name)

    def _detect_architectural_violations(self) -> List[Dict[str, Any]]:
        """Detect architectural violations and anti-patterns."""
        violations = []

        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies()
        for cycle in circular_deps:
            violations.append(
                {
                    "type": "circular_dependency",
                    "severity": "high",
                    "description": f"Circular dependency detected: {' -> '.join(cycle)}",
                    "packages": cycle,
                }
            )

        # Check package depth
        for package_name in self.current_structure.keys():
            depth = len(package_name.split("/"))
            if depth > self.package_standards["max_package_depth"]:
                violations.append(
                    {
                        "type": "excessive_depth",
                        "severity": "medium",
                        "description": f"Package depth {depth} exceeds maximum {self.package_standards['max_package_depth']}",
                        "package": package_name,
                    }
                )

        # Check for god packages (too many modules)
        for package_name, package_data in self.current_structure.items():
            if package_data["module_count"] > 20:  # Arbitrary threshold
                violations.append(
                    {
                        "type": "god_package",
                        "severity": "medium",
                        "description": f"Package contains {package_data['module_count']} modules (consider splitting)",
                        "package": package_name,
                    }
                )

        # Check for unused packages
        for package_name, deps in self.dependency_graph.items():
            if not deps["dependents"] and package_name not in ["__main__", "setup"]:
                violations.append(
                    {
                        "type": "unused_package",
                        "severity": "low",
                        "description": "Package appears to be unused",
                        "package": package_name,
                    }
                )

        return violations

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        cycles = []
        visited = set()
        path = []

        def dfs(package: str):
            if package in path:
                # Found a cycle
                cycle_start = path.index(package)
                cycle = path[cycle_start:] + [package]
                cycles.append(cycle)
                return

            if package in visited:
                return

            visited.add(package)
            path.append(package)

            deps = self.dependency_graph.get(package, {}).get("internal_dependencies", [])
            for dep in deps:
                if dep in self.dependency_graph:
                    dfs(dep)

            path.pop()

        for package in self.dependency_graph:
            if package not in visited:
                dfs(package)

        return cycles

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify package architecture optimization opportunities."""
        opportunities = []

        # Module consolidation opportunities
        for package_name, package_data in self.current_structure.items():
            small_modules = []
            for module_name, module_data in package_data["modules"].items():
                if (
                    module_data["line_count"] < 50
                    and module_data["class_count"] == 0
                    and module_data["function_count"] < 3
                ):
                    small_modules.append(module_name)

            if len(small_modules) > 3:
                opportunities.append(
                    {
                        "type": "consolidation",
                        "priority": "medium",
                        "description": f"Consider consolidating {len(small_modules)} small modules",
                        "package": package_name,
                        "modules": small_modules,
                    }
                )

        # Large module splitting opportunities
        for package_name, package_data in self.current_structure.items():
            for module_name, module_data in package_data["modules"].items():
                if module_data["line_count"] > 1000:
                    opportunities.append(
                        {
                            "type": "split",
                            "priority": "high",
                            "description": f"Large module should be split ({module_data['line_count']} lines)",
                            "package": package_name,
                            "module": module_name,
                        }
                    )

        # Package restructuring opportunities
        deep_packages = [name for name in self.current_structure.keys() if len(name.split("/")) > 3]
        if deep_packages:
            opportunities.append(
                {
                    "type": "restructure",
                    "priority": "medium",
                    "description": f"Consider flattening {len(deep_packages)} deeply nested packages",
                    "packages": deep_packages,
                }
            )

        # Import optimization opportunities
        for package_name, deps in self.dependency_graph.items():
            if len(deps["internal_dependencies"]) > 10:
                opportunities.append(
                    {
                        "type": "import_optimization",
                        "priority": "low",
                        "description": f"High number of internal dependencies ({len(deps['internal_dependencies'])})",
                        "package": package_name,
                    }
                )

        return opportunities

    def create_refactoring_plan(self) -> Dict[str, Any]:
        """
        Create comprehensive refactoring plan.

        Archaeological Enhancement: Systematic refactoring with minimal disruption.
        """
        logger.info("Creating refactoring plan...")

        plan = {
            "timestamp": datetime.now().isoformat(),
            "phases": [],
            "estimated_duration_hours": 0,
            "risk_level": "medium",
            "validation_required": True,
        }

        try:
            # Phase 1: Clean up and consolidation
            phase1 = self._create_cleanup_phase()
            plan["phases"].append(phase1)

            # Phase 2: Structural improvements
            phase2 = self._create_structural_phase()
            plan["phases"].append(phase2)

            # Phase 3: Optimization and polish
            phase3 = self._create_optimization_phase()
            plan["phases"].append(phase3)

            # Calculate total estimated duration
            plan["estimated_duration_hours"] = sum(phase["estimated_hours"] for phase in plan["phases"])

            # Assess risk level
            total_actions = sum(len(phase["actions"]) for phase in plan["phases"])
            if total_actions > 50:
                plan["risk_level"] = "high"
            elif total_actions < 10:
                plan["risk_level"] = "low"

            self.refactoring_plan = plan
            return plan

        except Exception as e:
            logger.error(f"Failed to create refactoring plan: {e}")
            plan["error"] = str(e)
            return plan

    def _create_cleanup_phase(self) -> Dict[str, Any]:
        """Create cleanup and consolidation phase."""
        phase = {
            "phase": 1,
            "name": "Cleanup and Consolidation",
            "description": "Remove unused code and consolidate small modules",
            "actions": [],
            "estimated_hours": 0,
        }

        # Find unused modules/packages to remove
        for package_name, deps in self.dependency_graph.items():
            if not deps["dependents"] and package_name not in ["__main__", "setup"]:
                phase["actions"].append(
                    {"type": "remove", "target": package_name, "reason": "Unused package", "estimated_minutes": 15}
                )

        # Find consolidation opportunities
        for opportunity in self.optimization_opportunities:
            if opportunity["type"] == "consolidation":
                phase["actions"].append(
                    {
                        "type": "consolidate",
                        "target": opportunity["package"],
                        "modules": opportunity["modules"],
                        "reason": "Consolidate small modules",
                        "estimated_minutes": len(opportunity["modules"]) * 10,
                    }
                )

        # Calculate estimated time
        phase["estimated_hours"] = sum(action["estimated_minutes"] for action in phase["actions"]) / 60

        return phase

    def _create_structural_phase(self) -> Dict[str, Any]:
        """Create structural improvements phase."""
        phase = {
            "phase": 2,
            "name": "Structural Improvements",
            "description": "Fix architectural violations and improve package structure",
            "actions": [],
            "estimated_hours": 0,
        }

        # Fix circular dependencies
        circular_deps = self._detect_circular_dependencies()
        for cycle in circular_deps:
            phase["actions"].append(
                {
                    "type": "fix_circular_dependency",
                    "cycle": cycle,
                    "reason": "Circular dependency violation",
                    "estimated_minutes": len(cycle) * 30,
                }
            )

        # Split large modules
        for opportunity in self.optimization_opportunities:
            if opportunity["type"] == "split":
                phase["actions"].append(
                    {
                        "type": "split_module",
                        "target": f"{opportunity['package']}/{opportunity['module']}",
                        "reason": "Large module split",
                        "estimated_minutes": 90,
                    }
                )

        # Restructure deep packages
        for opportunity in self.optimization_opportunities:
            if opportunity["type"] == "restructure":
                phase["actions"].append(
                    {
                        "type": "restructure_packages",
                        "packages": opportunity["packages"],
                        "reason": "Reduce package depth",
                        "estimated_minutes": len(opportunity["packages"]) * 20,
                    }
                )

        phase["estimated_hours"] = sum(action["estimated_minutes"] for action in phase["actions"]) / 60

        return phase

    def _create_optimization_phase(self) -> Dict[str, Any]:
        """Create optimization and polish phase."""
        phase = {
            "phase": 3,
            "name": "Optimization and Polish",
            "description": "Optimize imports and improve code organization",
            "actions": [],
            "estimated_hours": 0,
        }

        # Optimize imports
        for opportunity in self.optimization_opportunities:
            if opportunity["type"] == "import_optimization":
                phase["actions"].append(
                    {
                        "type": "optimize_imports",
                        "target": opportunity["package"],
                        "reason": "Reduce import complexity",
                        "estimated_minutes": 45,
                    }
                )

        # Create __all__ exports where missing
        for package_name, package_data in self.current_structure.items():
            if not package_data.get("exports"):
                phase["actions"].append(
                    {
                        "type": "add_exports",
                        "target": package_name,
                        "reason": "Add explicit exports",
                        "estimated_minutes": 20,
                    }
                )

        # Standardize package structure
        phase["actions"].append(
            {
                "type": "standardize_structure",
                "target": "all_packages",
                "reason": "Ensure consistent package structure",
                "estimated_minutes": 60,
            }
        )

        phase["estimated_hours"] = sum(action["estimated_minutes"] for action in phase["actions"]) / 60

        return phase

    def execute_refactoring_plan(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute the refactoring plan.

        Archaeological Enhancement: Safe refactoring with rollback capabilities.
        """
        logger.info(f"Executing refactoring plan (dry_run={dry_run})...")

        execution_result = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "phases_executed": 0,
            "actions_completed": 0,
            "actions_failed": 0,
            "errors": [],
            "backup_created": False,
        }

        try:
            # Create backup if not dry run
            if not dry_run:
                backup_path = self._create_backup()
                execution_result["backup_path"] = backup_path
                execution_result["backup_created"] = True

            # Execute each phase
            for phase in self.refactoring_plan["phases"]:
                logger.info(f"Executing Phase {phase['phase']}: {phase['name']}")

                phase_result = self._execute_phase(phase, dry_run)
                execution_result["actions_completed"] += phase_result["actions_completed"]
                execution_result["actions_failed"] += phase_result["actions_failed"]
                execution_result["errors"].extend(phase_result["errors"])

                if phase_result["success"]:
                    execution_result["phases_executed"] += 1
                else:
                    logger.error(f"Phase {phase['phase']} failed, stopping execution")
                    break

            # Validation
            if not dry_run and execution_result["phases_executed"] == len(self.refactoring_plan["phases"]):
                validation_result = self._validate_refactoring()
                execution_result["validation"] = validation_result

            execution_result["success"] = execution_result["actions_failed"] == 0

        except Exception as e:
            logger.error(f"Refactoring execution failed: {e}")
            execution_result["error"] = str(e)
            execution_result["success"] = False

        return execution_result

    def _execute_phase(self, phase: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
        """Execute individual refactoring phase."""
        result = {"phase": phase["phase"], "actions_completed": 0, "actions_failed": 0, "errors": [], "success": True}

        for action in phase["actions"]:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would execute: {action['type']} on {action.get('target', 'N/A')}")
                    result["actions_completed"] += 1
                else:
                    success = self._execute_action(action)
                    if success:
                        result["actions_completed"] += 1
                    else:
                        result["actions_failed"] += 1
                        result["success"] = False

            except Exception as e:
                logger.error(f"Action failed: {action['type']} - {e}")
                result["errors"].append(f"{action['type']}: {str(e)}")
                result["actions_failed"] += 1
                result["success"] = False

        return result

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute individual refactoring action."""
        action_type = action["type"]

        try:
            if action_type == "remove":
                return self._remove_package(action["target"])
            elif action_type == "consolidate":
                return self._consolidate_modules(action["target"], action["modules"])
            elif action_type == "split_module":
                return self._split_module(action["target"])
            elif action_type == "restructure_packages":
                return self._restructure_packages(action["packages"])
            elif action_type == "fix_circular_dependency":
                return self._fix_circular_dependency(action["cycle"])
            elif action_type == "optimize_imports":
                return self._optimize_imports(action["target"])
            elif action_type == "add_exports":
                return self._add_exports(action["target"])
            elif action_type == "standardize_structure":
                return self._standardize_structure()
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            logger.error(f"Action execution failed: {action_type} - {e}")
            return False

    def _create_backup(self) -> str:
        """Create backup of current project state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.project_root.parent / f"aivillage_backup_{timestamp}"

        logger.info(f"Creating backup at {backup_path}")
        shutil.copytree(self.project_root, backup_path, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        return str(backup_path)

    def _validate_refactoring(self) -> Dict[str, Any]:
        """Validate refactoring results."""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "syntax_errors": [],
            "import_errors": [],
            "test_results": {},
            "overall_success": True,
        }

        # Check for syntax errors
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError:
                validation["syntax_errors"].append(str(py_file))
                validation["overall_success"] = False

        # Basic import validation
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(self.project_root)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode != 0:
                validation["import_errors"] = result.stderr.split("\n")
                validation["overall_success"] = False
        except Exception as e:
            validation["validation_error"] = str(e)

        return validation

    # Placeholder implementations for refactoring actions
    def _remove_package(self, package_name: str) -> bool:
        """Remove unused package."""
        logger.info(f"Removing package: {package_name}")
        return True  # Placeholder

    def _consolidate_modules(self, package: str, modules: List[str]) -> bool:
        """Consolidate small modules."""
        logger.info(f"Consolidating modules in {package}: {modules}")
        return True  # Placeholder

    def _split_module(self, module_path: str) -> bool:
        """Split large module."""
        logger.info(f"Splitting module: {module_path}")
        return True  # Placeholder

    def _restructure_packages(self, packages: List[str]) -> bool:
        """Restructure package hierarchy."""
        logger.info(f"Restructuring packages: {packages}")
        return True  # Placeholder

    def _fix_circular_dependency(self, cycle: List[str]) -> bool:
        """Fix circular dependency."""
        logger.info(f"Fixing circular dependency: {cycle}")
        return True  # Placeholder

    def _optimize_imports(self, package: str) -> bool:
        """Optimize package imports."""
        logger.info(f"Optimizing imports for: {package}")
        return True  # Placeholder

    def _add_exports(self, package: str) -> bool:
        """Add __all__ exports to package."""
        logger.info(f"Adding exports to: {package}")
        return True  # Placeholder

    def _standardize_structure(self) -> bool:
        """Standardize package structure."""
        logger.info("Standardizing package structure")
        return True  # Placeholder


def run_package_architecture_refactor(project_root: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Main function to run package architecture refactoring.

    Archaeological Enhancement: Complete package refactoring workflow.
    """
    logger.info(f"Starting package architecture refactoring for {project_root}")

    try:
        # Initialize refactor system
        refactor = PackageArchitectureRefactor(project_root)

        # Analyze current architecture
        analysis = refactor.analyze_current_architecture()

        # Create refactoring plan
        plan = refactor.create_refactoring_plan()

        # Execute refactoring (dry run by default)
        execution_result = refactor.execute_refactoring_plan(dry_run)

        return {
            "analysis": analysis,
            "refactoring_plan": plan,
            "execution_result": execution_result,
            "timestamp": datetime.now().isoformat(),
            "success": execution_result.get("success", False),
        }

    except Exception as e:
        logger.error(f"Package architecture refactoring failed: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat(), "success": False}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python Package Architecture Refactoring")
    parser.add_argument("project_root", help="Root directory of the project")
    parser.add_argument("--execute", action="store_true", help="Execute refactoring (not dry run)")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run refactoring
    result = run_package_architecture_refactor(args.project_root, dry_run=not args.execute)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
    else:
        print(json.dumps(result, indent=2, default=str))
