"""
Production-ready development quality validation system.
Provides production-safe code quality validation for development workflows.
"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum


logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of development issues that can be detected."""
    CODE_SMELL = "code_smell"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"


class SeverityLevel(Enum):
    """Severity levels for detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    file_path: str
    line_number: int
    issue_type: IssueType
    severity: SeverityLevel
    message: str
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class QualityReport:
    """Quality analysis report."""
    file_path: str
    total_lines: int
    issues: List[QualityIssue]
    complexity_score: float
    maintainability_index: float
    timestamp: str


class DevelopmentQualityValidator:
    """Production-ready development quality validation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with configuration."""
        self.config = config or self._default_config()
        self.patterns = self._compile_patterns()
        self.excluded_paths = set(self.config.get('excluded_paths', []))
    
    def _default_config(self) -> Dict:
        """Default configuration for quality validation."""
        return {
            'max_line_length': 120,
            'max_function_length': 50,
            'max_complexity': 10,
            'excluded_paths': [
                'tests/',
                '__pycache__/',
                '.git/',
                'node_modules/',
                'venv/',
                'archive/',
                'deprecated/',
            ],
            'file_extensions': ['.py', '.js', '.ts', '.rs', '.go'],
            'severity_thresholds': {
                'complexity': {'medium': 5, 'high': 8, 'critical': 12},
                'function_length': {'medium': 30, 'high': 50, 'critical': 100},
                'line_length': {'medium': 100, 'high': 120, 'critical': 150},
            }
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for code analysis."""
        return {
            'long_line': re.compile(r'.{' + str(self.config['max_line_length'] + 1) + ',}'),
            'complex_condition': re.compile(r'\s+(if|elif|while|for).+\s+(and|or)\s+.+\s+(and|or)\s+'),
            'nested_loops': re.compile(r'(\s+)(for|while).+:\s*\n(\s+)+.*\n(\1\s+)(for|while)'),
            'magic_numbers': re.compile(r'\b(?<![\.\w])(?!0)[0-9]{2,}\b(?![\.\w])'),
            'empty_except': re.compile(r'except.*:\s*(?:\n\s*pass\s*)?$', re.MULTILINE),
            'broad_except': re.compile(r'except\s*:\s*'),
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> QualityReport:
        """Validate a single file for quality issues."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return self._empty_report(str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return self._empty_report(str(file_path))
        
        issues = []
        
        # Line-based analysis
        issues.extend(self._analyze_lines(lines, str(file_path)))
        
        # AST-based analysis for Python files
        if file_path.suffix == '.py':
            issues.extend(self._analyze_python_ast(content, str(file_path)))
        
        # Calculate metrics
        complexity_score = self._calculate_complexity(content, file_path.suffix)
        maintainability_index = self._calculate_maintainability_index(
            len(lines), len(issues), complexity_score
        )
        
        return QualityReport(
            file_path=str(file_path),
            total_lines=len(lines),
            issues=issues,
            complexity_score=complexity_score,
            maintainability_index=maintainability_index,
            timestamp=self._current_timestamp()
        )
    
    def validate_directory(self, directory: Union[str, Path], 
                         recursive: bool = True) -> List[QualityReport]:
        """Validate all files in a directory."""
        directory = Path(directory)
        reports = []
        
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return reports
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if self._should_validate_file(file_path):
                report = self.validate_file(file_path)
                reports.append(report)
        
        return reports
    
    def _should_validate_file(self, file_path: Path) -> bool:
        """Check if a file should be validated."""
        if not file_path.is_file():
            return False
        
        # Check extension
        if file_path.suffix not in self.config['file_extensions']:
            return False
        
        # Check excluded paths
        path_str = str(file_path)
        for excluded in self.excluded_paths:
            if excluded in path_str:
                return False
        
        return True
    
    def _analyze_lines(self, lines: List[str], file_path: str) -> List[QualityIssue]:
        """Analyze individual lines for quality issues."""
        issues = []
        
        for line_num, line in enumerate(lines, 1):
            # Long line detection
            if len(line) > self.config['max_line_length']:
                severity = self._determine_line_length_severity(len(line))
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type=IssueType.CODE_SMELL,
                    severity=severity,
                    message=f"Line too long ({len(line)} characters)",
                    suggestion="Consider breaking long lines for better readability"
                ))
            
            # Magic numbers
            if self.patterns['magic_numbers'].search(line):
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type=IssueType.MAINTAINABILITY,
                    severity=SeverityLevel.MEDIUM,
                    message="Magic number detected",
                    suggestion="Consider using named constants"
                ))
            
            # Empty except blocks
            if self.patterns['empty_except'].search(line):
                issues.append(QualityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type=IssueType.SECURITY,
                    severity=SeverityLevel.HIGH,
                    message="Empty except block",
                    suggestion="Handle exceptions appropriately or use specific exception types"
                ))
        
        return issues
    
    def _analyze_python_ast(self, content: str, file_path: str) -> List[QualityIssue]:
        """Analyze Python AST for quality issues."""
        issues = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append(QualityIssue(
                file_path=file_path,
                line_number=e.lineno or 1,
                issue_type=IssueType.CODE_SMELL,
                severity=SeverityLevel.CRITICAL,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix syntax error"
            ))
            return issues
        
        # Function complexity analysis
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_function_complexity(node)
                severity = self._determine_complexity_severity(complexity)
                
                if severity != SeverityLevel.LOW:
                    issues.append(QualityIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type=IssueType.COMPLEXITY,
                        severity=severity,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        suggestion="Consider breaking down into smaller functions"
                    ))
        
        return issues
    
    def _calculate_complexity(self, content: str, file_extension: str) -> float:
        """Calculate cyclomatic complexity score."""
        if file_extension == '.py':
            return self._calculate_python_complexity(content)
        else:
            # Basic complexity for other languages
            complexity_indicators = [
                'if', 'elif', 'else', 'for', 'while', 'try', 'catch',
                'switch', 'case', '&&', '||', '?', ':'
            ]
            return sum(content.count(indicator) for indicator in complexity_indicators)
    
    def _calculate_python_complexity(self, content: str) -> float:
        """Calculate cyclomatic complexity for Python code."""
        try:
            tree = ast.parse(content)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, 
                                   ast.ExceptHandler, ast.With, ast.Assert)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except SyntaxError:
            return 0
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate complexity for a specific function."""
        complexity = 1
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, 
                               ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_maintainability_index(self, lines_count: int, 
                                       issues_count: int, complexity: float) -> float:
        """Calculate maintainability index (0-100 scale)."""
        # Calculate maintainability index using standard metrics
        base_score = 100
        
        # Penalties for issues and complexity
        issue_penalty = min(issues_count * 2, 30)
        complexity_penalty = min(complexity * 1.5, 25)
        size_penalty = min(lines_count / 50, 15)
        
        score = base_score - issue_penalty - complexity_penalty - size_penalty
        return max(0, min(100, score))
    
    def _determine_line_length_severity(self, length: int) -> SeverityLevel:
        """Determine severity based on line length."""
        thresholds = self.config['severity_thresholds']['line_length']
        
        if length >= thresholds['critical']:
            return SeverityLevel.CRITICAL
        elif length >= thresholds['high']:
            return SeverityLevel.HIGH
        elif length >= thresholds['medium']:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _determine_complexity_severity(self, complexity: int) -> SeverityLevel:
        """Determine severity based on complexity score."""
        thresholds = self.config['severity_thresholds']['complexity']
        
        if complexity >= thresholds['critical']:
            return SeverityLevel.CRITICAL
        elif complexity >= thresholds['high']:
            return SeverityLevel.HIGH
        elif complexity >= thresholds['medium']:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _empty_report(self, file_path: str) -> QualityReport:
        """Create an empty quality report."""
        return QualityReport(
            file_path=file_path,
            total_lines=0,
            issues=[],
            complexity_score=0.0,
            maintainability_index=100.0,
            timestamp=self._current_timestamp()
        )
    
    def _current_timestamp(self) -> str:
        """Get current timestamp string."""
        import datetime
        return datetime.datetime.now().isoformat()


def create_quality_validator(config: Optional[Dict] = None) -> DevelopmentQualityValidator:
    """Factory function to create a quality validator instance."""
    return DevelopmentQualityValidator(config)


def validate_codebase_quality(directory: Union[str, Path], 
                            config: Optional[Dict] = None) -> List[QualityReport]:
    """Convenience function to validate entire codebase quality."""
    validator = create_quality_validator(config)
    return validator.validate_directory(directory, recursive=True)