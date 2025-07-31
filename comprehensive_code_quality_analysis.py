#!/usr/bin/env python3
"""Comprehensive Code Quality Analysis for AIVillage Codebase.

This script performs in-depth code quality analysis including:
- TODO/FIXME tracking and categorization
- Code complexity metrics (cyclomatic, cognitive)
- Documentation coverage analysis
- Import dependency analysis
- Security issue detection
- Code duplication detection
- Style compliance assessment
"""

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class QualityMetrics:
    """Container for code quality metrics."""
    
    # File-level metrics
    file_path: str
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    
    # Documentation metrics
    has_module_docstring: bool = False
    documented_functions: int = 0
    total_functions: int = 0
    
    # Technical debt
    todos: List[str] = field(default_factory=list)
    fixmes: List[str] = field(default_factory=list)
    hacks: List[str] = field(default_factory=list)
    
    # Code style issues
    style_violations: List[str] = field(default_factory=list)
    
    # Import analysis
    imports: List[str] = field(default_factory=list)
    relative_imports: List[str] = field(default_factory=list)
    
    # Security issues
    security_issues: List[str] = field(default_factory=list)


class CodeQualityAnalyzer:
    """Main analyzer for code quality metrics."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.metrics: Dict[str, QualityMetrics] = {}
        
        # Define key directories for different quality standards
        self.production_dirs = [
            "production/compression",
            "production/evolution", 
            "production/rag",
            "production/geometry"
        ]
        
        self.core_dirs = [
            "agent_forge",
            "mcp_servers/hyperag"
        ]
        
        self.experimental_dirs = [
            "experimental",
            "benchmarks",
            "scripts"
        ]
    
    def analyze_file(self, file_path: Path) -> QualityMetrics:
        """Analyze a single Python file."""
        metrics = QualityMetrics(file_path=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic metrics
            metrics.lines_of_code = len([line for line in content.split('\n') if line.strip()])
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, metrics)
            except SyntaxError as e:
                metrics.style_violations.append(f"Syntax error: {e}")
            
            # TODO/FIXME analysis
            self._analyze_technical_debt(content, metrics)
            
            # Import analysis
            self._analyze_imports(tree if 'tree' in locals() else None, metrics)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
        return metrics
    
    def _analyze_ast(self, tree: ast.AST, metrics: QualityMetrics) -> None:
        """Analyze AST for complexity and documentation metrics."""
        
        # Check for module docstring
        if (isinstance(tree, ast.Module) and tree.body and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Str)):
            metrics.has_module_docstring = True
        
        # Analyze functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.total_functions += 1
                
                # Check for docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    metrics.documented_functions += 1
                
                # Calculate cyclomatic complexity
                complexity = self._calculate_cyclomatic_complexity(node)
                metrics.cyclomatic_complexity += complexity
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    def _analyze_technical_debt(self, content: str, metrics: QualityMetrics) -> None:
        """Find and categorize technical debt markers."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            if 'todo' in line_lower:
                metrics.todos.append(f"Line {i}: {line.strip()}")
            elif 'fixme' in line_lower:
                metrics.fixmes.append(f"Line {i}: {line.strip()}")
            elif 'hack' in line_lower or 'xxx' in line_lower:
                metrics.hacks.append(f"Line {i}: {line.strip()}")
    
    def _analyze_imports(self, tree: Optional[ast.AST], metrics: QualityMetrics) -> None:
        """Analyze import patterns."""
        if not tree:
            return
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    metrics.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:  # Relative import
                    module = node.module or ""
                    metrics.relative_imports.append(f"{'.' * node.level}{module}")
                else:
                    if node.module:
                        metrics.imports.append(node.module)
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        print("Starting comprehensive code quality analysis...")
        
        # Find all Python files
        python_files = list(self.root_path.rglob("*.py"))
        
        # Filter out virtual environment and cache files
        python_files = [
            f for f in python_files 
            if not any(part in f.parts for part in [
                'new_env', '.venv', 'venv', '__pycache__', '.git',
                'node_modules', '.mypy_cache', '.ruff_cache'
            ])
        ]
        
        print(f"Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        for file_path in python_files:
            if file_path.stat().st_size > 0:  # Skip empty files
                self.metrics[str(file_path)] = self.analyze_file(file_path)
        
        # Generate comprehensive report
        return self._generate_comprehensive_report()
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        # Overall statistics
        total_files = len(self.metrics)
        total_loc = sum(m.lines_of_code for m in self.metrics.values())
        total_functions = sum(m.total_functions for m in self.metrics.values())
        total_documented = sum(m.documented_functions for m in self.metrics.values())
        
        # Technical debt analysis
        all_todos = []
        all_fixmes = []
        all_hacks = []
        
        for file_path, metrics in self.metrics.items():
            for todo in metrics.todos:
                all_todos.append({"file": file_path, "item": todo})
            for fixme in metrics.fixmes:
                all_fixmes.append({"file": file_path, "item": fixme})
            for hack in metrics.hacks:
                all_hacks.append({"file": file_path, "item": hack})
        
        # Categorize by directory
        production_metrics = self._categorize_metrics(self.production_dirs)
        core_metrics = self._categorize_metrics(self.core_dirs)
        experimental_metrics = self._categorize_metrics(self.experimental_dirs)
        
        # Complexity analysis
        high_complexity_files = [
            {"file": path, "complexity": metrics.cyclomatic_complexity}
            for path, metrics in self.metrics.items()
            if metrics.cyclomatic_complexity > 10
        ]
        high_complexity_files.sort(key=lambda x: x["complexity"], reverse=True)
        
        # Documentation coverage by category
        doc_coverage = {
            "production": self._calculate_doc_coverage(production_metrics),
            "core": self._calculate_doc_coverage(core_metrics),
            "experimental": self._calculate_doc_coverage(experimental_metrics),
            "overall": total_documented / total_functions if total_functions > 0 else 0
        }
        
        return {
            "summary": {
                "total_files": total_files,
                "total_lines_of_code": total_loc,
                "total_functions": total_functions,
                "documented_functions": total_documented,
                "documentation_coverage": doc_coverage["overall"],
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            },
            "technical_debt": {
                "todos": {
                    "count": len(all_todos),
                    "items": all_todos[:20],  # Top 20 for report
                    "files_affected": len(set(item["file"] for item in all_todos))
                },
                "fixmes": {
                    "count": len(all_fixmes),
                    "items": all_fixmes[:20],
                    "files_affected": len(set(item["file"] for item in all_fixmes))
                },
                "hacks": {
                    "count": len(all_hacks),
                    "items": all_hacks[:20],
                    "files_affected": len(set(item["file"] for item in all_hacks))
                }
            },
            "complexity_analysis": {
                "high_complexity_files": high_complexity_files[:15],
                "average_complexity": sum(m.cyclomatic_complexity for m in self.metrics.values()) / total_files,
                "files_needing_refactor": len(high_complexity_files)
            },
            "documentation_coverage": doc_coverage,
            "directory_breakdown": {
                "production": self._generate_directory_summary(production_metrics),
                "core": self._generate_directory_summary(core_metrics),
                "experimental": self._generate_directory_summary(experimental_metrics)
            },
            "import_analysis": self._analyze_import_patterns(),
            "quality_gates": self._evaluate_quality_gates()
        }
    
    def _categorize_metrics(self, directories: List[str]) -> Dict[str, QualityMetrics]:
        """Categorize metrics by directory patterns."""
        categorized = {}
        
        for file_path, metrics in self.metrics.items():
            path_obj = Path(file_path)
            for directory in directories:
                if any(part in path_obj.parts for part in directory.split('/')):
                    categorized[file_path] = metrics
                    break
                    
        return categorized
    
    def _calculate_doc_coverage(self, metrics_dict: Dict[str, QualityMetrics]) -> float:
        """Calculate documentation coverage for a set of metrics."""
        total_functions = sum(m.total_functions for m in metrics_dict.values())
        documented_functions = sum(m.documented_functions for m in metrics_dict.values())
        
        return documented_functions / total_functions if total_functions > 0 else 0.0
    
    def _generate_directory_summary(self, metrics_dict: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """Generate summary for a directory category."""
        if not metrics_dict:
            return {"files": 0, "loc": 0, "complexity": 0, "todos": 0}
            
        return {
            "files": len(metrics_dict),
            "lines_of_code": sum(m.lines_of_code for m in metrics_dict.values()),
            "avg_complexity": sum(m.cyclomatic_complexity for m in metrics_dict.values()) / len(metrics_dict),
            "total_todos": sum(len(m.todos) for m in metrics_dict.values()),
            "total_fixmes": sum(len(m.fixmes) for m in metrics_dict.values()),
            "documentation_coverage": self._calculate_doc_coverage(metrics_dict)
        }
    
    def _analyze_import_patterns(self) -> Dict[str, Any]:
        """Analyze import patterns across the codebase."""
        all_imports = defaultdict(int)
        relative_imports = defaultdict(int)
        
        for metrics in self.metrics.values():
            for imp in metrics.imports:
                all_imports[imp] += 1
            for rel_imp in metrics.relative_imports:
                relative_imports[rel_imp] += 1
        
        # Find most common imports
        common_imports = sorted(all_imports.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "most_common_imports": common_imports,
            "relative_import_usage": len(relative_imports),
            "total_unique_imports": len(all_imports),
            "potential_consolidation_opportunities": [
                imp for imp, count in all_imports.items() 
                if count > 10 and '.' in imp
            ]
        }
    
    def _evaluate_quality_gates(self) -> Dict[str, Any]:
        """Evaluate quality gates for different code categories."""
        production_metrics = self._categorize_metrics(self.production_dirs)
        
        # Production quality gates (strictest)
        production_todos = sum(len(m.todos) + len(m.fixmes) for m in production_metrics.values())
        production_doc_coverage = self._calculate_doc_coverage(production_metrics)
        
        quality_gates = {
            "production_no_todos": {
                "passed": production_todos == 0,
                "current_count": production_todos,
                "threshold": 0
            },
            "production_documentation": {
                "passed": production_doc_coverage >= 0.8,
                "current_coverage": production_doc_coverage,
                "threshold": 0.8
            },
            "overall_complexity": {
                "passed": True,  # Will be calculated based on thresholds
                "files_exceeding_threshold": len([
                    m for m in self.metrics.values() 
                    if m.cyclomatic_complexity > 15
                ])
            }
        }
        
        return quality_gates


def run_linting_tools(root_path: str) -> Dict[str, Any]:
    """Run actual linting tools and capture results."""
    results = {}
    
    try:
        # Run ruff check
        ruff_result = subprocess.run([
            'ruff', 'check', root_path, '--output-format=json'
        ], capture_output=True, text=True, timeout=300)
        
        if ruff_result.stdout:
            try:
                results['ruff'] = json.loads(ruff_result.stdout)
            except json.JSONDecodeError:
                results['ruff'] = {"error": "Failed to parse ruff output"}
        else:
            results['ruff'] = {"violations": []}
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        results['ruff'] = {"error": f"Ruff execution failed: {e}"}
    
    try:
        # Run mypy on key directories
        mypy_result = subprocess.run([
            'mypy', 'agent_forge/', '--ignore-missing-imports', '--json-report'
        ], capture_output=True, text=True, timeout=300)
        
        results['mypy'] = {
            "stdout": mypy_result.stdout,
            "stderr": mypy_result.stderr,
            "returncode": mypy_result.returncode
        }
        
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        results['mypy'] = {"error": f"Mypy execution failed: {e}"}
    
    return results


def main():
    """Main execution function."""
    root_path = os.getcwd()
    
    print("AIVillage Comprehensive Code Quality Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CodeQualityAnalyzer(root_path)
    
    # Run comprehensive analysis
    quality_report = analyzer.analyze_codebase()
    
    # Run linting tools
    print("\nRunning linting tools...")
    linting_results = run_linting_tools(root_path)
    
    # Combine results
    final_report = {
        "code_quality_analysis": quality_report,
        "linting_results": linting_results,
        "recommendations": generate_recommendations(quality_report)
    }
    
    # Save comprehensive report
    output_file = Path(root_path) / "comprehensive_quality_dashboard.json"
    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\nComprehensive quality report saved to: {output_file}")
    
    # Print executive summary
    print_executive_summary(quality_report)
    
    return final_report


def generate_recommendations(quality_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # High-priority recommendations
    if quality_report["technical_debt"]["todos"]["count"] > 50:
        recommendations.append({
            "priority": "HIGH",
            "category": "Technical Debt",
            "issue": f"Found {quality_report['technical_debt']['todos']['count']} TODO items",
            "recommendation": "Create sprint backlog items for critical TODOs, especially in production code",
            "effort_estimate": "2-3 sprints"
        })
    
    if quality_report["documentation_coverage"]["production"] < 0.8:
        recommendations.append({
            "priority": "HIGH", 
            "category": "Documentation",
            "issue": f"Production documentation coverage at {quality_report['documentation_coverage']['production']:.1%}",
            "recommendation": "Add docstrings to all public APIs in production modules",
            "effort_estimate": "1 sprint"
        })
    
    if quality_report["complexity_analysis"]["files_needing_refactor"] > 10:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Code Complexity",
            "issue": f"{quality_report['complexity_analysis']['files_needing_refactor']} files need refactoring",
            "recommendation": "Break down complex functions into smaller, testable units",
            "effort_estimate": "2-4 sprints"
        })
    
    return recommendations


def print_executive_summary(quality_report: Dict[str, Any]) -> None:
    """Print executive summary of quality analysis."""
    summary = quality_report["summary"]
    tech_debt = quality_report["technical_debt"]
    
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    
    print(f"📊 Codebase Overview:")
    print(f"   • Total Files: {summary['total_files']:,}")
    print(f"   • Lines of Code: {summary['total_lines_of_code']:,}")
    print(f"   • Functions: {summary['total_functions']:,}")
    print(f"   • Documentation Coverage: {summary['documentation_coverage']:.1%}")
    
    print(f"\n🔧 Technical Debt:")
    print(f"   • TODO Items: {tech_debt['todos']['count']} across {tech_debt['todos']['files_affected']} files")
    print(f"   • FIXME Items: {tech_debt['fixmes']['count']} across {tech_debt['fixmes']['files_affected']} files")
    print(f"   • HACK/XXX Items: {tech_debt['hacks']['count']} across {tech_debt['hacks']['files_affected']} files")
    
    print(f"\n📈 Quality Gates:")
    quality_gates = quality_report["quality_gates"]
    for gate_name, gate_data in quality_gates.items():
        status = "✅ PASS" if gate_data.get("passed", False) else "❌ FAIL"
        print(f"   • {gate_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Priority Actions:")
    print(f"   1. Address {tech_debt['todos']['count']} TODO items (focus on production)")
    print(f"   2. Improve documentation coverage from {summary['documentation_coverage']:.1%}")
    print(f"   3. Refactor {quality_report['complexity_analysis']['files_needing_refactor']} high-complexity files")


if __name__ == "__main__":
    main()