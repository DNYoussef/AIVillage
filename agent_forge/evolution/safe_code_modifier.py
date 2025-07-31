"""
Safe Code Modification System - Autonomous Agent Self-Improvement

Implements safe, sandboxed code modification capabilities for agent self-evolution.
Includes validation, rollback, and security measures to prevent harmful modifications.
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
import re

import black
import isort
from pylint import epylint as lint

logger = logging.getLogger(__name__)


@dataclass
class CodeModification:
    """Represents a code modification operation"""
    modification_id: str
    agent_id: str
    file_path: str
    original_code: str
    modified_code: str
    modification_type: str
    description: str
    safety_score: float
    validation_results: Dict[str, Any]
    timestamp: datetime
    applied: bool = False
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class SafetyPolicy:
    """Safety policies for code modifications"""
    allowed_imports: Set[str]
    forbidden_patterns: List[str]
    max_file_size: int
    max_complexity: int
    require_tests: bool
    sandbox_timeout: int


class CodeValidator:
    """Validates code modifications for safety and correctness"""
    
    def __init__(self, safety_policy: SafetyPolicy):
        self.safety_policy = safety_policy
        
        # Default forbidden patterns for security
        self.default_forbidden = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'open\s*\([^)]*[\'"]w[\'"]',  # Write file operations
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            r'rm\s+-rf',
            r'shutil\.rmtree',
            r'socket\.',
            r'urllib\.request',
            r'requests\.',
            r'http\.',
            r'pickle\.load',
            r'marshal\.load'
        ]
        
        self.safety_policy.forbidden_patterns.extend(self.default_forbidden)
    
    async def validate_modification(self, modification: CodeModification) -> Dict[str, Any]:
        """Comprehensive validation of code modification"""
        
        validation_results = {
            'syntax_valid': False,
            'security_safe': False,
            'complexity_acceptable': False,
            'imports_allowed': False,
            'tests_present': False,
            'formatting_correct': False,
            'overall_safe': False,
            'safety_score': 0.0,
            'warnings': [],
            'errors': []
        }
        
        try:
            # 1. Syntax validation
            validation_results['syntax_valid'] = await self._validate_syntax(modification.modified_code)
            
            # 2. Security validation
            validation_results['security_safe'] = await self._validate_security(modification.modified_code)
            
            # 3. Complexity validation
            validation_results['complexity_acceptable'] = await self._validate_complexity(modification.modified_code)
            
            # 4. Import validation
            validation_results['imports_allowed'] = await self._validate_imports(modification.modified_code)
            
            # 5. Test presence validation (if required)
            if self.safety_policy.require_tests:
                validation_results['tests_present'] = await self._validate_tests(modification)
            else:
                validation_results['tests_present'] = True
            
            # 6. Formatting validation
            validation_results['formatting_correct'] = await self._validate_formatting(modification.modified_code)
            
            # Calculate overall safety score
            safety_score = self._calculate_safety_score(validation_results)
            validation_results['safety_score'] = safety_score
            validation_results['overall_safe'] = safety_score >= 0.8
            
        except Exception as e:
            logger.error(f"Validation failed for modification {modification.modification_id}: {e}")
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    async def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            return False
    
    async def _validate_security(self, code: str) -> bool:
        """Validate code security against forbidden patterns"""
        
        for pattern in self.safety_policy.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Security violation: Found forbidden pattern '{pattern}'")
                return False
        
        # Additional AST-based security checks
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['exec', 'eval', '__import__']:
                            return False
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['system', 'popen', 'rmtree']:
                            return False
                
                # Check for file operations
                if isinstance(node, ast.With):
                    if isinstance(node.items[0].context_expr, ast.Call):
                        if isinstance(node.items[0].context_expr.func, ast.Name):
                            if node.items[0].context_expr.func.id == 'open':
                                # Check for write mode
                                if len(node.items[0].context_expr.args) > 1:
                                    mode_arg = node.items[0].context_expr.args[1]
                                    if isinstance(mode_arg, ast.Str) and 'w' in mode_arg.s:
                                        return False
        
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return False
        
        return True
    
    async def _validate_complexity(self, code: str) -> bool:
        """Validate code complexity"""
        try:
            tree = ast.parse(code)
            complexity = self._calculate_cyclomatic_complexity(tree)
            return complexity <= self.safety_policy.max_complexity
        except Exception as e:
            logger.error(f"Complexity validation error: {e}")
            return False
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of AST"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    async def _validate_imports(self, code: str) -> bool:
        """Validate that only allowed imports are used"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.safety_policy.allowed_imports:
                            logger.warning(f"Forbidden import: {alias.name}")
                            return False
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.safety_policy.allowed_imports:
                        # Check if parent module is allowed
                        parent_allowed = any(
                            node.module.startswith(allowed) 
                            for allowed in self.safety_policy.allowed_imports
                        )
                        if not parent_allowed:
                            logger.warning(f"Forbidden import: {node.module}")
                            return False
        
        except Exception as e:
            logger.error(f"Import validation error: {e}")
            return False
        
        return True
    
    async def _validate_tests(self, modification: CodeModification) -> bool:
        """Validate that tests exist for the modification"""
        # Simple heuristic: check if test functions are present
        code = modification.modified_code
        
        # Look for test functions or test classes
        test_patterns = [
            r'def\s+test_',
            r'class\s+Test\w+',
            r'import\s+unittest',
            r'import\s+pytest',
            r'from\s+unittest',
            r'from\s+pytest'
        ]
        
        for pattern in test_patterns:
            if re.search(pattern, code):
                return True
        
        return False
    
    async def _validate_formatting(self, code: str) -> bool:
        """Validate code formatting"""
        try:
            # Check if black would modify the code
            formatted = black.format_str(code, mode=black.FileMode())
            return formatted == code
        except Exception:
            return False
    
    def _calculate_safety_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall safety score"""
        weights = {
            'syntax_valid': 0.25,
            'security_safe': 0.30,
            'complexity_acceptable': 0.15,
            'imports_allowed': 0.15,
            'tests_present': 0.10,
            'formatting_correct': 0.05
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if validation_results.get(metric, False):
                score += weight
        
        return score


class SandboxEnvironment:
    """Isolated environment for safe code execution and testing"""
    
    def __init__(self, base_path: str = "evolution_data/sandbox"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.active_sandboxes = {}
    
    async def create_sandbox(self, modification: CodeModification) -> str:
        """Create isolated sandbox for testing modification"""
        
        sandbox_id = f"sandbox_{modification.modification_id}_{int(time.time())}"
        sandbox_path = self.base_path / sandbox_id
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        # Copy original code to sandbox
        original_file = sandbox_path / "original.py"
        with open(original_file, 'w') as f:
            f.write(modification.original_code)
        
        # Write modified code to sandbox
        modified_file = sandbox_path / "modified.py"
        with open(modified_file, 'w') as f:
            f.write(modification.modified_code)
        
        # Create test runner
        test_runner = sandbox_path / "test_runner.py"
        with open(test_runner, 'w') as f:
            f.write(self._generate_test_runner(modification))
        
        self.active_sandboxes[sandbox_id] = {
            'path': sandbox_path,
            'modification': modification,
            'created_at': datetime.now()
        }
        
        return sandbox_id
    
    def _generate_test_runner(self, modification: CodeModification) -> str:
        """Generate test runner for sandbox"""
        
        return f'''
import sys
import traceback
import time
import resource
from pathlib import Path

def set_resource_limits():
    """Set resource limits for safety"""
    # Limit CPU time to 30 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    
    # Limit memory to 100MB
    resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))

def test_modification():
    """Test the modified code"""
    results = {{
        'success': False,
        'error': None,
        'output': '',
        'execution_time': 0,
        'memory_usage': 0
    }}
    
    try:
        set_resource_limits()
        
        start_time = time.time()
        
        # Import and test the modified code
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Basic import test
        import modified
        
        # If it's a class, try to instantiate
        for name in dir(modified):
            obj = getattr(modified, name)
            if isinstance(obj, type) and not name.startswith('_'):
                try:
                    instance = obj()
                    results['output'] += f"Successfully instantiated {{name}}\\n"
                except Exception as e:
                    results['output'] += f"Failed to instantiate {{name}}: {{e}}\\n"
        
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        results['output'] = traceback.format_exc()
    
    return results

if __name__ == "__main__":
    import json
    results = test_modification()
    print(json.dumps(results, indent=2))
'''
    
    async def run_sandbox_test(self, sandbox_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Run test in sandbox environment"""
        
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        sandbox_path = sandbox_info['path']
        test_runner = sandbox_path / "test_runner.py"
        
        try:
            # Run test in subprocess with timeout
            result = subprocess.run(
                [sys.executable, str(test_runner)],
                cwd=sandbox_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                try:
                    test_results = json.loads(result.stdout)
                except json.JSONDecodeError:
                    test_results = {
                        'success': False,
                        'error': 'Failed to parse test results',
                        'output': result.stdout
                    }
            else:
                test_results = {
                    'success': False,
                    'error': f'Test runner failed with code {result.returncode}',
                    'output': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            test_results = {
                'success': False,
                'error': 'Test execution timed out',
                'output': f'Execution exceeded {timeout} seconds'
            }
        
        except Exception as e:
            test_results = {
                'success': False,
                'error': f'Sandbox execution failed: {str(e)}',
                'output': ''
            }
        
        return test_results
    
    async def cleanup_sandbox(self, sandbox_id: str):
        """Clean up sandbox environment"""
        
        if sandbox_id in self.active_sandboxes:
            sandbox_path = self.active_sandboxes[sandbox_id]['path']
            
            try:
                shutil.rmtree(sandbox_path)
                del self.active_sandboxes[sandbox_id]
                logger.info(f"Cleaned up sandbox {sandbox_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup sandbox {sandbox_id}: {e}")
    
    async def cleanup_old_sandboxes(self, max_age_hours: int = 24):
        """Clean up old sandbox environments"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_cleanup = []
        
        for sandbox_id, info in self.active_sandboxes.items():
            if info['created_at'] < cutoff_time:
                to_cleanup.append(sandbox_id)
        
        for sandbox_id in to_cleanup:
            await self.cleanup_sandbox(sandbox_id)


class SafeCodeModifier:
    """Main class for safe code modification operations"""
    
    def __init__(self, 
                 safety_policy: Optional[SafetyPolicy] = None,
                 backup_path: str = "evolution_data/backups"):
        
        # Default safety policy
        if safety_policy is None:
            safety_policy = SafetyPolicy(
                allowed_imports={
                    'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
                    'torch', 'transformers', 'datasets', 'tokenizers',
                    'json', 'csv', 'pickle', 'datetime', 'time', 'math',
                    'random', 'collections', 'itertools', 'functools',
                    'typing', 'dataclasses', 'pathlib', 'logging',
                    'asyncio', 'concurrent', 'threading'
                },
                forbidden_patterns=[],
                max_file_size=100000,  # 100KB
                max_complexity=20,
                require_tests=False,  # Relaxed for initial implementation
                sandbox_timeout=30
            )
        
        self.safety_policy = safety_policy
        self.validator = CodeValidator(safety_policy)
        self.sandbox = SandboxEnvironment()
        
        # Backup system
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Modification tracking
        self.modifications = {}
        self.modification_history = []
    
    async def propose_modification(self, 
                                 agent_id: str,
                                 file_path: str,
                                 modification_type: str,
                                 description: str,
                                 code_transformer: Callable[[str], str]) -> CodeModification:
        """Propose a code modification"""
        
        modification_id = self._generate_modification_id(agent_id, file_path)
        
        # Read original code
        try:
            with open(file_path, 'r') as f:
                original_code = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")
        
        # Apply transformation
        try:
            modified_code = code_transformer(original_code)
        except Exception as e:
            raise ValueError(f"Code transformation failed: {e}")
        
        # Create modification object
        modification = CodeModification(
            modification_id=modification_id,
            agent_id=agent_id,
            file_path=file_path,
            original_code=original_code,
            modified_code=modified_code,
            modification_type=modification_type,
            description=description,
            safety_score=0.0,
            validation_results={},
            timestamp=datetime.now()
        )
        
        # Validate modification
        validation_results = await self.validator.validate_modification(modification)
        modification.validation_results = validation_results
        modification.safety_score = validation_results['safety_score']
        
        # Store modification
        self.modifications[modification_id] = modification
        
        logger.info(f"Proposed modification {modification_id} with safety score {modification.safety_score:.3f}")
        
        return modification
    
    async def test_modification(self, modification_id: str) -> Dict[str, Any]:
        """Test modification in sandbox environment"""
        
        if modification_id not in self.modifications:
            raise ValueError(f"Modification {modification_id} not found")
        
        modification = self.modifications[modification_id]
        
        # Create sandbox
        sandbox_id = await self.sandbox.create_sandbox(modification)
        
        try:
            # Run tests
            test_results = await self.sandbox.run_sandbox_test(
                sandbox_id, 
                timeout=self.safety_policy.sandbox_timeout
            )
            
            # Update modification with test results
            if 'test_results' not in modification.validation_results:
                modification.validation_results['test_results'] = {}
            
            modification.validation_results['test_results'].update(test_results)
            
            return test_results
        
        finally:
            # Cleanup sandbox
            await self.sandbox.cleanup_sandbox(sandbox_id)
    
    async def apply_modification(self, modification_id: str, force: bool = False) -> bool:
        """Apply validated modification to actual code"""
        
        if modification_id not in self.modifications:
            raise ValueError(f"Modification {modification_id} not found")
        
        modification = self.modifications[modification_id]
        
        # Safety checks
        if not force:
            if modification.safety_score < 0.8:
                logger.warning(f"Modification {modification_id} has low safety score: {modification.safety_score}")
                return False
            
            if not modification.validation_results.get('overall_safe', False):
                logger.warning(f"Modification {modification_id} failed safety validation")
                return False
        
        try:
            # Create backup
            backup_file = await self._create_backup(modification.file_path)
            modification.rollback_data = {'backup_file': str(backup_file)}
            
            # Apply modification
            with open(modification.file_path, 'w') as f:
                f.write(modification.modified_code)
            
            # Format code
            await self._format_code(modification.file_path)
            
            modification.applied = True
            self.modification_history.append(modification)
            
            logger.info(f"Successfully applied modification {modification_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to apply modification {modification_id}: {e}")
            
            # Attempt rollback if backup was created
            if modification.rollback_data and 'backup_file' in modification.rollback_data:
                await self._rollback_modification(modification)
            
            return False
    
    async def rollback_modification(self, modification_id: str) -> bool:
        """Rollback a previously applied modification"""
        
        if modification_id not in self.modifications:
            raise ValueError(f"Modification {modification_id} not found")
        
        modification = self.modifications[modification_id]
        
        if not modification.applied:
            logger.warning(f"Modification {modification_id} was not applied")
            return False
        
        return await self._rollback_modification(modification)
    
    async def _rollback_modification(self, modification: CodeModification) -> bool:
        """Internal rollback implementation"""
        
        try:
            if modification.rollback_data and 'backup_file' in modification.rollback_data:
                backup_file = modification.rollback_data['backup_file']
                
                # Restore from backup
                shutil.copy2(backup_file, modification.file_path)
                
                modification.applied = False
                logger.info(f"Successfully rolled back modification {modification.modification_id}")
                return True
            else:
                # Restore original code
                with open(modification.file_path, 'w') as f:
                    f.write(modification.original_code)
                
                await self._format_code(modification.file_path)
                
                modification.applied = False
                logger.info(f"Restored original code for modification {modification.modification_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to rollback modification {modification.modification_id}: {e}")
            return False
    
    async def _create_backup(self, file_path: str) -> Path:
        """Create backup of file before modification"""
        
        file_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}.backup"
        backup_file = self.backup_path / backup_name
        
        shutil.copy2(file_path, backup_file)
        
        return backup_file
    
    async def _format_code(self, file_path: str):
        """Format code using black and isort"""
        
        try:
            # Format with black
            with open(file_path, 'r') as f:
                code = f.read()
            
            formatted_code = black.format_str(code, mode=black.FileMode())
            
            with open(file_path, 'w') as f:
                f.write(formatted_code)
            
            # Sort imports with isort
            isort.file(file_path)
            
        except Exception as e:
            logger.warning(f"Failed to format code in {file_path}: {e}")
    
    def _generate_modification_id(self, agent_id: str, file_path: str) -> str:
        """Generate unique modification ID"""
        
        timestamp = int(time.time())
        content = f"{agent_id}_{file_path}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        
        return f"mod_{hash_obj.hexdigest()[:8]}_{timestamp}"
    
    async def get_modification_status(self, modification_id: str) -> Dict[str, Any]:
        """Get detailed status of a modification"""
        
        if modification_id not in self.modifications:
            raise ValueError(f"Modification {modification_id} not found")
        
        modification = self.modifications[modification_id]
        
        return {
            'modification_id': modification.modification_id,
            'agent_id': modification.agent_id,
            'file_path': modification.file_path,
            'modification_type': modification.modification_type,
            'description': modification.description,
            'safety_score': modification.safety_score,
            'applied': modification.applied,
            'timestamp': modification.timestamp.isoformat(),
            'validation_results': modification.validation_results
        }
    
    async def list_modifications(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all modifications, optionally filtered by agent"""
        
        modifications = []
        
        for modification in self.modifications.values():
            if agent_id is None or modification.agent_id == agent_id:
                modifications.append({
                    'modification_id': modification.modification_id,
                    'agent_id': modification.agent_id,
                    'file_path': modification.file_path,
                    'modification_type': modification.modification_type,
                    'safety_score': modification.safety_score,
                    'applied': modification.applied,
                    'timestamp': modification.timestamp.isoformat()
                })
        
        return sorted(modifications, key=lambda x: x['timestamp'], reverse=True)
    
    async def cleanup_old_backups(self, max_age_days: int = 30):
        """Clean up old backup files"""
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for backup_file in self.backup_path.glob("*.backup"):
            try:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_time:
                    backup_file.unlink()
                    logger.info(f"Cleaned up old backup: {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup backup {backup_file}: {e}")


# Common code transformations for agent self-improvement
class CodeTransformations:
    """Collection of safe code transformations for agent improvement"""
    
    @staticmethod
    def optimize_hyperparameters(code: str, param_adjustments: Dict[str, Any]) -> str:
        """Optimize hyperparameters in code"""
        
        modified_code = code
        
        for param_name, new_value in param_adjustments.items():
            # Simple regex-based replacement for common parameter patterns
            patterns = [
                rf'{param_name}\s*=\s*[\d\.]+',
                rf'"{param_name}":\s*[\d\.]+',
                rf"'{param_name}':\s*[\d\.]+"
            ]
            
            for pattern in patterns:
                if isinstance(new_value, (int, float)):
                    replacement = f'{param_name} = {new_value}'
                else:
                    replacement = f'{param_name} = "{new_value}"'
                
                modified_code = re.sub(pattern, replacement, modified_code)
        
        return modified_code
    
    @staticmethod
    def add_error_handling(code: str, function_names: List[str]) -> str:
        """Add error handling to specified functions"""
        
        # Parse AST and add try-except blocks
        try:
            tree = ast.parse(code)
            
            class ErrorHandlingTransformer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name in function_names:
                        # Wrap function body in try-except
                        try_node = ast.Try(
                            body=node.body,
                            handlers=[
                                ast.ExceptHandler(
                                    type=ast.Name(id='Exception', ctx=ast.Load()),
                                    name='e',
                                    body=[
                                        ast.Expr(
                                            value=ast.Call(
                                                func=ast.Attribute(
                                                    value=ast.Name(id='logger', ctx=ast.Load()),
                                                    attr='error',
                                                    ctx=ast.Load()
                                                ),
                                                args=[
                                                    ast.BinOp(
                                                        left=ast.Str(s=f'Error in {node.name}: '),
                                                        op=ast.Add(),
                                                        right=ast.Name(id='e', ctx=ast.Load())
                                                    )
                                                ],
                                                keywords=[]
                                            )
                                        ),
                                        ast.Return(value=ast.Constant(value=None))
                                    ]
                                )
                            ],
                            orelse=[],
                            finalbody=[]
                        )
                        node.body = [try_node]
                    
                    return self.generic_visit(node)
            
            transformer = ErrorHandlingTransformer()
            modified_tree = transformer.visit(tree)
            
            return ast.unparse(modified_tree)
        
        except Exception as e:
            logger.error(f"Failed to add error handling: {e}")
            return code
    
    @staticmethod
    def improve_documentation(code: str) -> str:
        """Add or improve docstrings and comments"""
        
        try:
            tree = ast.parse(code)
            
            class DocstringAdder(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # Add basic docstring if missing
                    if not (node.body and isinstance(node.body[0], ast.Expr) and 
                           isinstance(node.body[0].value, ast.Str)):
                        
                        # Generate basic docstring
                        args_str = ', '.join([arg.arg for arg in node.args.args])
                        docstring = f'"""{node.name} function.\n\nArgs:\n    {args_str}\n\nReturns:\n    Result of {node.name}\n"""'
                        
                        docstring_node = ast.Expr(value=ast.Str(s=docstring))
                        node.body.insert(0, docstring_node)
                    
                    return self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    # Add basic class docstring if missing
                    if not (node.body and isinstance(node.body[0], ast.Expr) and 
                           isinstance(node.body[0].value, ast.Str)):
                        
                        docstring = f'"""{node.name} class.\n\nA class for {node.name.lower()} operations.\n"""'
                        docstring_node = ast.Expr(value=ast.Str(s=docstring))
                        node.body.insert(0, docstring_node)
                    
                    return self.generic_visit(node)
            
            transformer = DocstringAdder()
            modified_tree = transformer.visit(tree)
            
            return ast.unparse(modified_tree)
        
        except Exception as e:
            logger.error(f"Failed to improve documentation: {e}")
            return code
    
    @staticmethod
    def optimize_imports(code: str) -> str:
        """Optimize and clean up imports"""
        
        try:
            # Use isort to optimize imports
            from isort import code as isort_code
            return isort_code(code)
        except Exception as e:
            logger.error(f"Failed to optimize imports: {e}")
            return code


if __name__ == "__main__":
    async def example_usage():
        # Initialize safe code modifier
        modifier = SafeCodeModifier()
        
        # Example: optimize hyperparameters
        def hyperparameter_transformer(code: str) -> str:
            return CodeTransformations.optimize_hyperparameters(
                code, 
                {'learning_rate': 0.001, 'batch_size': 32}
            )
        
        # Propose modification
        modification = await modifier.propose_modification(
            agent_id="test_agent",
            file_path="example.py",
            modification_type="hyperparameter_optimization",
            description="Optimize learning rate and batch size",
            code_transformer=hyperparameter_transformer
        )
        
        print(f"Proposed modification: {modification.modification_id}")
        print(f"Safety score: {modification.safety_score:.3f}")
        
        # Test modification
        if modification.safety_score >= 0.8:
            test_results = await modifier.test_modification(modification.modification_id)
            print(f"Test results: {test_results}")
            
            # Apply if tests pass
            if test_results.get('success', False):
                success = await modifier.apply_modification(modification.modification_id)
                print(f"Applied successfully: {success}")
    
    asyncio.run(example_usage())