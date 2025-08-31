"""
Coupling Analyzer for Phase 4 Validation

Automated coupling analysis to ensure architectural improvements meet targets.
"""

import ast
import asyncio
import logging
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json


@dataclass
class CouplingMetrics:
    """Coupling metrics for a component"""
    
    component_name: str
    coupling_score: float
    incoming_dependencies: int
    outgoing_dependencies: int
    dependency_list: List[str]
    circular_dependencies: List[Tuple[str, str]]
    stability_metric: float  # (outgoing) / (incoming + outgoing)
    abstractness_metric: float  # abstract classes / total classes
    distance_from_main: float  # Distance from main sequence


class CouplingAnalyzer:
    """
    Analyzes coupling scores for Phase 4 architectural improvements
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        
        # Target components for Phase 4
        self.target_components = {
            'UnifiedManagement': 'swarm/agents/unified_management.py',
            'SageAgent': 'swarm/agents/sage_agent.py',
            'TaskManager': 'swarm/core/task_manager.py',
            'WorkflowEngine': 'swarm/core/workflow_engine.py',
            'ExecutionManager': 'swarm/core/execution_manager.py',
            'ResourceManager': 'swarm/core/resource_manager.py'
        }
        
        # Baseline coupling scores from coupling_metrics.py analysis
        self.baseline_scores = {
            'UnifiedManagement': 21.6,
            'SageAgent': 47.46,
            'task_management_average': 9.56
        }
    
    async def analyze_all_components(self) -> Dict[str, Any]:
        """
        Analyze coupling for all target components
        
        Returns:
            Dictionary containing coupling metrics for all components
        """
        self.logger.info("Starting coupling analysis for all components...")
        
        results = {}
        
        # Run coupling analysis in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            
            for component_name, file_path in self.target_components.items():
                full_path = self.project_root / file_path
                if full_path.exists():
                    task = asyncio.get_event_loop().run_in_executor(
                        executor, self._analyze_component_coupling, component_name, full_path
                    )
                    tasks.append((component_name, task))
                else:
                    self.logger.warning(f"Component file not found: {full_path}")
            
            # Wait for all analyses to complete
            for component_name, task in tasks:
                try:
                    coupling_metrics = await task
                    results[component_name] = coupling_metrics.__dict__
                except Exception as e:
                    self.logger.error(f"Failed to analyze {component_name}: {e}")
                    results[component_name] = {'error': str(e)}
        
        # Calculate task management average
        task_components = ['TaskManager', 'WorkflowEngine', 'ExecutionManager', 'ResourceManager']
        task_scores = [results.get(comp, {}).get('coupling_score', 0) for comp in task_components if comp in results]
        
        if task_scores:
            results['task_management_average'] = sum(task_scores) / len(task_scores)
        
        # Add improvement percentages
        results['improvements'] = self._calculate_improvements(results)
        
        self.logger.info("Coupling analysis completed")
        return results
    
    def _analyze_component_coupling(self, component_name: str, file_path: Path) -> CouplingMetrics:
        """
        Analyze coupling metrics for a single component
        
        Args:
            component_name: Name of the component
            file_path: Path to the component file
            
        Returns:
            CouplingMetrics object with detailed analysis
        """
        self.logger.debug(f"Analyzing coupling for {component_name}")
        
        try:
            # Parse the Python file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(tree, content)
            
            # Calculate coupling metrics
            incoming_deps = self._count_incoming_dependencies(component_name)
            outgoing_deps = len(dependencies)
            coupling_score = self._calculate_coupling_score(incoming_deps, outgoing_deps, content)
            
            # Calculate additional metrics
            stability = outgoing_deps / (incoming_deps + outgoing_deps) if (incoming_deps + outgoing_deps) > 0 else 0
            abstractness = self._calculate_abstractness(tree)
            distance = abs(stability + abstractness - 1)
            
            # Find circular dependencies
            circular_deps = self._find_circular_dependencies(component_name, dependencies)
            
            return CouplingMetrics(
                component_name=component_name,
                coupling_score=coupling_score,
                incoming_dependencies=incoming_deps,
                outgoing_dependencies=outgoing_deps,
                dependency_list=list(dependencies),
                circular_dependencies=circular_deps,
                stability_metric=stability,
                abstractness_metric=abstractness,
                distance_from_main=distance
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {component_name}: {e}")
            return CouplingMetrics(
                component_name=component_name,
                coupling_score=float('inf'),
                incoming_dependencies=0,
                outgoing_dependencies=0,
                dependency_list=[],
                circular_dependencies=[],
                stability_metric=0,
                abstractness_metric=0,
                distance_from_main=1
            )
    
    def _extract_dependencies(self, tree: ast.AST, content: str) -> Set[str]:
        """Extract all dependencies from the AST"""
        dependencies = set()
        
        class DependencyVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Look for dynamic imports
                if (isinstance(node.func, ast.Name) and node.func.id == '__import__') or \
                   (isinstance(node.func, ast.Attribute) and node.func.attr == '__import__'):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        dependencies.add(node.args[0].value.split('.')[0])
                self.generic_visit(node)
        
        visitor = DependencyVisitor()
        visitor.visit(tree)
        
        # Also scan for string-based imports (like in factories)
        import_patterns = [
            r'importlib\.import_module\(["\']([^"\']+)["\']',
            r'__import__\(["\']([^"\']+)["\']'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                dependencies.add(match.split('.')[0])
        
        # Filter out standard library modules
        external_deps = set()
        for dep in dependencies:
            if not self._is_stdlib_module(dep) and dep != '__main__':
                external_deps.add(dep)
        
        return external_deps
    
    def _count_incoming_dependencies(self, component_name: str) -> int:
        """Count how many other components depend on this one"""
        count = 0
        
        # Search for imports of this component across the codebase
        search_paths = [
            self.project_root / "swarm",
            self.project_root / "core",
            self.project_root / "infrastructure"
        ]
        
        component_file = self.target_components.get(component_name, '').replace('.py', '').replace('/', '.')
        
        for search_path in search_paths:
            if search_path.exists():
                for python_file in search_path.rglob("*.py"):
                    try:
                        with open(python_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Look for imports of this component
                        import_patterns = [
                            f'from.*{component_name}.*import',
                            f'import.*{component_name}',
                            f'from.*{component_file}.*import',
                            f'import.*{component_file}'
                        ]
                        
                        for pattern in import_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                count += 1
                                break
                    except Exception:
                        continue
        
        return count
    
    def _calculate_coupling_score(self, incoming: int, outgoing: int, content: str) -> float:
        """
        Calculate coupling score using multiple factors
        
        Formula considers:
        - Fan-in and fan-out
        - Lines of code
        - Number of classes/methods
        - Complexity indicators
        """
        lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        # Count classes and methods
        tree = ast.parse(content)
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        methods = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Base coupling score: weighted fan-in and fan-out
        base_score = (incoming * 2) + (outgoing * 1.5)
        
        # Adjust for size and complexity
        size_factor = lines_of_code / 100  # Normalize to per-100-lines
        complexity_factor = (classes + methods) / 10  # Normalize complexity
        
        # Final score
        coupling_score = base_score + (size_factor * 0.5) + (complexity_factor * 0.3)
        
        return round(coupling_score, 2)
    
    def _calculate_abstractness(self, tree: ast.AST) -> float:
        """Calculate abstractness metric (abstract classes / total classes)"""
        total_classes = 0
        abstract_classes = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                total_classes += 1
                
                # Check if class is abstract (has @abstractmethod or ABC base)
                for decorator in getattr(node, 'decorator_list', []):
                    if isinstance(decorator, ast.Name) and decorator.id in ['abstractmethod', 'ABC']:
                        abstract_classes += 1
                        break
                    elif isinstance(decorator, ast.Attribute) and decorator.attr in ['abstractmethod']:
                        abstract_classes += 1
                        break
                
                # Check for ABC in base classes
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'ABC':
                        abstract_classes += 1
                        break
        
        return abstract_classes / total_classes if total_classes > 0 else 0
    
    def _find_circular_dependencies(self, component_name: str, dependencies: Set[str]) -> List[Tuple[str, str]]:
        """Find circular dependencies involving this component"""
        circular = []
        
        # Simple circular dependency detection
        # This would need to be enhanced for complex dependency graphs
        for dep in dependencies:
            if dep in self.target_components:
                # Check if the dependency also depends on this component
                dep_path = self.project_root / self.target_components[dep]
                if dep_path.exists():
                    try:
                        with open(dep_path, 'r', encoding='utf-8') as f:
                            dep_content = f.read()
                        
                        if component_name.lower() in dep_content.lower():
                            circular.append((component_name, dep))
                    except Exception:
                        continue
        
        return circular
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the Python standard library"""
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'logging', 'pathlib', 'typing',
            'asyncio', 'concurrent', 'threading', 'multiprocessing', 're', 'collections',
            'itertools', 'functools', 'operator', 'copy', 'pickle', 'csv', 'xml',
            'http', 'urllib', 'socket', 'ssl', 'email', 'mimetypes', 'base64',
            'hashlib', 'hmac', 'uuid', 'random', 'math', 'statistics', 'decimal',
            'fractions', 'string', 'textwrap', 'unicodedata', 'locale', 'calendar',
            'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile'
        }
        
        return module_name in stdlib_modules
    
    def _calculate_improvements(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate improvement percentages compared to baseline"""
        improvements = {}
        
        for component, baseline in self.baseline_scores.items():
            if component in results and 'coupling_score' in results[component]:
                current = results[component]['coupling_score']
                improvement_percent = ((baseline - current) / baseline) * 100
                
                improvements[component] = {
                    'baseline': baseline,
                    'current': current,
                    'improvement_percent': round(improvement_percent, 1),
                    'target_met': current <= self._get_target_for_component(component)
                }
        
        return improvements
    
    def _get_target_for_component(self, component: str) -> float:
        """Get coupling target for a specific component"""
        targets = {
            'UnifiedManagement': 8.0,
            'SageAgent': 25.0,
            'task_management_average': 6.0
        }
        return targets.get(component, 0)
    
    async def run_legacy_coupling_analysis(self) -> Dict[str, Any]:
        """
        Run the existing coupling_metrics.py script for comparison
        
        Returns:
            Results from the legacy coupling analysis
        """
        coupling_script = self.project_root / "scripts/coupling_metrics.py"
        
        if not coupling_script.exists():
            self.logger.warning("Legacy coupling_metrics.py not found")
            return {}
        
        try:
            result = subprocess.run(
                ["python", str(coupling_script)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # Parse output for coupling scores
                output_lines = result.stdout.split('\n')
                scores = {}
                
                for line in output_lines:
                    if 'coupling score' in line.lower():
                        # Extract component name and score
                        parts = line.split(':')
                        if len(parts) >= 2:
                            component = parts[0].strip()
                            score_str = parts[1].strip().split()[0]
                            try:
                                scores[component] = float(score_str)
                            except ValueError:
                                continue
                
                return {'legacy_scores': scores, 'output': result.stdout}
            else:
                self.logger.error(f"Legacy coupling analysis failed: {result.stderr}")
                return {'error': result.stderr}
                
        except Exception as e:
            self.logger.error(f"Failed to run legacy coupling analysis: {e}")
            return {'error': str(e)}