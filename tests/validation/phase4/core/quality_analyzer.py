"""
Code Quality Analyzer for Phase 4 Validation

Analyzes code quality metrics to ensure Phase 4 improvements meet quality targets.
"""

import ast
import asyncio
import logging
import re
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json
import statistics


@dataclass
class QualityMetrics:
    """Code quality metrics for a component"""
    
    file_path: str
    lines_of_code: int
    lines_per_class: List[int]
    magic_literals: List[str]
    cyclomatic_complexity: int
    test_coverage: float
    maintainability_index: float
    code_duplication: float


class QualityAnalyzer:
    """
    Analyzes code quality metrics for Phase 4 validation
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        
        # Files to analyze for Phase 4
        self.target_files = [
            'swarm/agents/unified_management.py',
            'swarm/agents/sage_agent.py',
            'swarm/core/task_manager.py',
            'swarm/core/workflow_engine.py',
            'swarm/core/execution_manager.py',
            'swarm/core/resource_manager.py'
        ]
        
        # Quality targets
        self.targets = {
            'max_lines_per_class': 150,
            'magic_literals_target': 0,
            'min_test_coverage': 90.0,
            'max_cyclomatic_complexity': 10,
            'min_maintainability_index': 70.0,
            'max_code_duplication': 5.0
        }
    
    async def analyze_quality_metrics(self) -> Dict[str, Any]:
        """
        Analyze code quality metrics for all target files
        
        Returns:
            Comprehensive quality analysis results
        """
        self.logger.info("Starting code quality analysis...")
        
        results = {
            'files': [],
            'summary': {
                'total_files': 0,
                'analyzed_files': 0,
                'failed_files': 0
            },
            'aggregated_metrics': {},
            'targets_met': {}
        }
        
        # Analyze files concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            
            for file_path in self.target_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    task = asyncio.get_event_loop().run_in_executor(
                        executor, self._analyze_file_quality, full_path
                    )
                    tasks.append((file_path, task))
                    results['summary']['total_files'] += 1
                else:
                    self.logger.warning(f"File not found: {full_path}")
            
            # Wait for all analyses to complete
            for file_path, task in tasks:
                try:
                    quality_metrics = await task
                    results['files'].append({
                        'file_path': file_path,
                        'metrics': quality_metrics.__dict__,
                        'analysis_successful': True
                    })
                    results['summary']['analyzed_files'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to analyze {file_path}: {e}")
                    results['files'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'analysis_successful': False
                    })
                    results['summary']['failed_files'] += 1
        
        # Calculate aggregated metrics
        results['aggregated_metrics'] = self._calculate_aggregated_metrics(results['files'])
        
        # Check if targets are met
        results['targets_met'] = self._check_quality_targets(results['aggregated_metrics'])
        
        # Add overall quality score
        results['overall_quality_score'] = self._calculate_overall_quality_score(results['aggregated_metrics'])
        
        self.logger.info("Code quality analysis completed")
        return results
    
    def _analyze_file_quality(self, file_path: Path) -> QualityMetrics:
        """
        Analyze quality metrics for a single file
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            QualityMetrics object with detailed analysis
        """
        self.logger.debug(f"Analyzing quality for {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Calculate various metrics
            lines_of_code = self._count_lines_of_code(content)
            lines_per_class = self._calculate_lines_per_class(tree, content)
            magic_literals = self._find_magic_literals(tree, content)
            cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
            test_coverage = self._estimate_test_coverage(file_path)
            maintainability_index = self._calculate_maintainability_index(content, cyclomatic_complexity)
            code_duplication = self._detect_code_duplication(content)
            
            return QualityMetrics(
                file_path=str(file_path),
                lines_of_code=lines_of_code,
                lines_per_class=lines_per_class,
                magic_literals=magic_literals,
                cyclomatic_complexity=cyclomatic_complexity,
                test_coverage=test_coverage,
                maintainability_index=maintainability_index,
                code_duplication=code_duplication
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            raise
    
    def _count_lines_of_code(self, content: str) -> int:
        """Count non-blank, non-comment lines of code"""
        lines = content.split('\\n')
        loc = 0
        
        in_multiline_string = False
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle multiline strings
            if '"""' in line or "'''" in line:
                quote_count = line.count('"""') + line.count("'''")
                if quote_count % 2 == 1:
                    in_multiline_string = not in_multiline_string
                continue
            
            if in_multiline_string:
                continue
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Count as line of code
            loc += 1
        
        return loc
    
    def _calculate_lines_per_class(self, tree: ast.AST, content: str) -> List[int]:
        """Calculate lines per class"""
        lines = content.split('\\n')
        lines_per_class = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                
                # If end_lineno is not available, estimate based on next class/function
                if end_line == start_line:
                    # Find next class or function at same indentation level
                    for other_node in ast.walk(tree):
                        if (isinstance(other_node, (ast.ClassDef, ast.FunctionDef)) and 
                            other_node != node and 
                            other_node.lineno > start_line):
                            end_line = other_node.lineno - 1
                            break
                    else:
                        end_line = len(lines)
                
                class_lines = end_line - start_line + 1
                lines_per_class.append(class_lines)
        
        return lines_per_class
    
    def _find_magic_literals(self, tree: ast.AST, content: str) -> List[str]:
        """Find magic numbers and strings in the code"""
        magic_literals = []
        
        # Common non-magic numbers/strings to exclude
        excluded_numbers = {0, 1, 2, -1, 100, 1000}
        excluded_strings = {'', ' ', '\\n', '\\t'}
        
        class MagicLiteralVisitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    if node.value not in excluded_numbers and abs(node.value) > 1:
                        magic_literals.append(f"Number: {node.value} (line {node.lineno})")
                elif isinstance(node.value, str):
                    if (len(node.value) > 1 and 
                        node.value not in excluded_strings and
                        not node.value.isspace() and
                        not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', node.value)):  # Not just identifier-like
                        # Only report if it's not a docstring
                        parent = getattr(node, 'parent', None)
                        if not (isinstance(parent, ast.Expr) and isinstance(parent.value, ast.Constant)):
                            magic_literals.append(f"String: '{node.value[:50]}...' (line {node.lineno})")
                self.generic_visit(node)
            
            # Handle older Python versions
            def visit_Num(self, node):
                if node.n not in excluded_numbers and abs(node.n) > 1:
                    magic_literals.append(f"Number: {node.n} (line {node.lineno})")
                self.generic_visit(node)
            
            def visit_Str(self, node):
                if (len(node.s) > 1 and 
                    node.s not in excluded_strings and
                    not node.s.isspace()):
                    magic_literals.append(f"String: '{node.s[:50]}...' (line {node.lineno})")
                self.generic_visit(node)
        
        # Add parent references
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        
        visitor = MagicLiteralVisitor()
        visitor.visit(tree)
        
        return magic_literals
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points that increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # And/Or operations
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ListComp):
                # List comprehensions with conditions
                for generator in node.generators:
                    complexity += len(generator.ifs)
                    complexity += 1
            elif isinstance(node, (ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Other comprehensions
                for generator in node.generators:
                    complexity += len(generator.ifs)
                    complexity += 1
        
        return complexity
    
    def _estimate_test_coverage(self, file_path: Path) -> float:
        """Estimate test coverage for the file"""
        # Look for corresponding test file
        test_patterns = [
            f"test_{file_path.stem}.py",
            f"{file_path.stem}_test.py",
            f"test_{file_path.stem}_*.py"
        ]
        
        test_dirs = [
            self.project_root / "tests",
            self.project_root / "test", 
            file_path.parent / "tests",
            file_path.parent / "test"
        ]
        
        # Check if test files exist
        test_files_found = []
        for test_dir in test_dirs:
            if test_dir.exists():
                for pattern in test_patterns:
                    test_files = list(test_dir.glob(pattern))
                    test_files_found.extend(test_files)
        
        if not test_files_found:
            return 0.0  # No tests found
        
        # Simple heuristic: estimate coverage based on test file size vs source file size
        try:
            source_size = file_path.stat().st_size
            total_test_size = sum(tf.stat().st_size for tf in test_files_found)
            
            # Very rough estimation: coverage = min(100%, test_size / source_size * 80%)
            estimated_coverage = min(100.0, (total_test_size / source_size) * 80.0)
            
            # If we have test files but they're small, assume at least 30% coverage
            if total_test_size > 0 and estimated_coverage < 30.0:
                estimated_coverage = 30.0
            
            return estimated_coverage
            
        except Exception as e:
            self.logger.debug(f"Error estimating test coverage for {file_path}: {e}")
            return 50.0 if test_files_found else 0.0  # Default if tests exist
    
    def _calculate_maintainability_index(self, content: str, cyclomatic_complexity: int) -> float:
        """Calculate maintainability index (simplified version)"""
        loc = self._count_lines_of_code(content)
        
        # Simplified maintainability index calculation
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
        # We'll use a simplified version since calculating Halstead metrics is complex
        
        import math
        
        # Approximate Halstead volume (very rough estimate)
        unique_operators = len(re.findall(r'[+\-*/%=<>!&|^~]|and|or|not|in|is', content))
        unique_operands = len(set(re.findall(r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b', content)))
        halstead_volume = max(1, (unique_operators + unique_operands) * math.log2(max(1, unique_operators + unique_operands)))
        
        try:
            mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(max(1, loc))
            # Normalize to 0-100 scale
            mi = max(0, min(100, mi))
        except (ValueError, ZeroDivisionError):
            mi = 50.0  # Default value if calculation fails
        
        return mi
    
    def _detect_code_duplication(self, content: str) -> float:
        """Detect code duplication percentage"""
        lines = [line.strip() for line in content.split('\\n') if line.strip() and not line.strip().startswith('#')]
        
        if len(lines) < 10:
            return 0.0  # Too few lines to have meaningful duplication
        
        # Look for exact line duplicates
        line_counts = {}
        for line in lines:
            if len(line) > 10:  # Only consider substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1
        
        # Count duplicated lines
        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        # Calculate duplication percentage
        duplication_percentage = (duplicated_lines / len(lines)) * 100 if lines else 0
        
        return duplication_percentage
    
    def _calculate_aggregated_metrics(self, file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics across all files"""
        successful_results = [fr for fr in file_results if fr.get('analysis_successful', False)]
        
        if not successful_results:
            return {}
        
        # Aggregate metrics
        all_lines_per_class = []
        all_magic_literals = []
        all_complexities = []
        all_coverages = []
        all_maintainability = []
        all_duplications = []
        total_loc = 0
        
        for result in successful_results:
            metrics = result['metrics']
            
            all_lines_per_class.extend(metrics.get('lines_per_class', []))
            all_magic_literals.extend(metrics.get('magic_literals', []))
            all_complexities.append(metrics.get('cyclomatic_complexity', 0))
            all_coverages.append(metrics.get('test_coverage', 0))
            all_maintainability.append(metrics.get('maintainability_index', 0))
            all_duplications.append(metrics.get('code_duplication', 0))
            total_loc += metrics.get('lines_of_code', 0)
        
        return {
            'total_lines_of_code': total_loc,
            'max_lines_per_class': max(all_lines_per_class) if all_lines_per_class else 0,
            'avg_lines_per_class': statistics.mean(all_lines_per_class) if all_lines_per_class else 0,
            'magic_literals_count': len(all_magic_literals),
            'magic_literals_list': all_magic_literals,
            'max_cyclomatic_complexity': max(all_complexities) if all_complexities else 0,
            'avg_cyclomatic_complexity': statistics.mean(all_complexities) if all_complexities else 0,
            'avg_test_coverage': statistics.mean(all_coverages) if all_coverages else 0,
            'min_test_coverage': min(all_coverages) if all_coverages else 0,
            'avg_maintainability_index': statistics.mean(all_maintainability) if all_maintainability else 0,
            'max_code_duplication': max(all_duplications) if all_duplications else 0,
            'files_analyzed': len(successful_results)
        }
    
    def _check_quality_targets(self, aggregated_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check if quality targets are met"""
        if not aggregated_metrics:
            return {}
        
        return {
            'lines_per_class_target_met': aggregated_metrics.get('max_lines_per_class', float('inf')) <= self.targets['max_lines_per_class'],
            'magic_literals_target_met': aggregated_metrics.get('magic_literals_count', float('inf')) <= self.targets['magic_literals_target'],
            'test_coverage_target_met': aggregated_metrics.get('avg_test_coverage', 0) >= self.targets['min_test_coverage'],
            'cyclomatic_complexity_target_met': aggregated_metrics.get('max_cyclomatic_complexity', float('inf')) <= self.targets['max_cyclomatic_complexity'],
            'maintainability_target_met': aggregated_metrics.get('avg_maintainability_index', 0) >= self.targets['min_maintainability_index'],
            'code_duplication_target_met': aggregated_metrics.get('max_code_duplication', float('inf')) <= self.targets['max_code_duplication']
        }
    
    def _calculate_overall_quality_score(self, aggregated_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        if not aggregated_metrics:
            return 0.0
        
        scores = []
        
        # Lines per class score (0-100)
        max_lines = aggregated_metrics.get('max_lines_per_class', 0)
        lines_score = max(0, 100 - (max_lines - self.targets['max_lines_per_class']) * 2) if max_lines > 0 else 100
        scores.append(lines_score)
        
        # Magic literals score (0-100)
        magic_count = aggregated_metrics.get('magic_literals_count', 0)
        magic_score = max(0, 100 - magic_count * 5)  # Penalty of 5 points per magic literal
        scores.append(magic_score)
        
        # Test coverage score (direct percentage)
        coverage_score = aggregated_metrics.get('avg_test_coverage', 0)
        scores.append(coverage_score)
        
        # Cyclomatic complexity score (0-100)
        max_complexity = aggregated_metrics.get('max_cyclomatic_complexity', 0)
        complexity_score = max(0, 100 - (max_complexity - self.targets['max_cyclomatic_complexity']) * 10) if max_complexity > 0 else 100
        scores.append(complexity_score)
        
        # Maintainability score (already 0-100)
        maintainability_score = aggregated_metrics.get('avg_maintainability_index', 0)
        scores.append(maintainability_score)
        
        # Code duplication score (0-100)
        duplication = aggregated_metrics.get('max_code_duplication', 0)
        duplication_score = max(0, 100 - duplication * 2)  # Penalty of 2 points per percent duplication
        scores.append(duplication_score)
        
        # Calculate weighted average
        weights = [0.15, 0.20, 0.25, 0.15, 0.15, 0.10]  # Coverage gets highest weight
        overall_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return overall_score
    
    async def run_external_quality_tools(self) -> Dict[str, Any]:
        """Run external code quality tools (flake8, pylint, etc.)"""
        external_results = {}
        
        # List of external tools to run
        tools = [
            ('flake8', ['flake8', '--statistics', '--count']),
            ('pylint', ['pylint', '--output-format=json']),
            ('mypy', ['mypy', '--json-report'])
        ]
        
        for tool_name, command in tools:
            try:
                # Add target files to command
                full_command = command + [str(self.project_root / fp) for fp in self.target_files 
                                         if (self.project_root / fp).exists()]
                
                result = subprocess.run(
                    full_command,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=60  # 1 minute timeout
                )
                
                external_results[tool_name] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'success': result.returncode == 0
                }
                
            except subprocess.TimeoutExpired:
                external_results[tool_name] = {
                    'error': 'Timeout',
                    'success': False
                }
            except FileNotFoundError:
                external_results[tool_name] = {
                    'error': f'{tool_name} not installed',
                    'success': False
                }
            except Exception as e:
                external_results[tool_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return external_results