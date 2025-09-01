#!/usr/bin/env python3
"""
Python Syntax and Exception Handling Validator
Comprehensive syntax analysis tool for Python codebase
"""
import os
import py_compile
import ast
import tempfile
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import re

class PythonSyntaxValidator:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.errors = []
        self.warnings = []
        self.files_checked = 0
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the directory tree."""
        python_files = []
        for file_path in self.root_path.rglob("*.py"):
            if not any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pytest_cache']):
                python_files.append(file_path)
        return python_files
    
    def compile_check(self, file_path: Path) -> Tuple[bool, str]:
        """Check if a Python file compiles without syntax errors."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                tmp_file.write(content)
                tmp_file.flush()
                
                py_compile.compile(tmp_file.name, doraise=True)
                os.unlink(tmp_file.name)
                return True, "OK"
        except py_compile.PyCompileError as e:
            os.unlink(tmp_file.name) if 'tmp_file' in locals() else None
            return False, str(e)
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def ast_parse_check(self, file_path: Path) -> Tuple[bool, str, ast.AST]:
        """Parse file with AST to get detailed syntax information."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            return True, "OK", tree
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}", None
        except Exception as e:
            return False, f"Parse error: {e}", None
    
    def check_try_except_blocks(self, file_path: Path, content: str) -> List[Dict]:
        """Check for malformed try-except blocks."""
        issues = []
        lines = content.split('\n')
        
        # Look for try blocks around line 299-300 pattern
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for try without except
            if line_stripped == 'try:':
                # Look ahead for except, finally, or another try
                found_except_or_finally = False
                j = i
                while j < len(lines) and j < i + 10:  # Look ahead max 10 lines
                    next_line = lines[j].strip()
                    if next_line.startswith('except') or next_line.startswith('finally'):
                        found_except_or_finally = True
                        break
                    elif next_line.startswith('try:') and j != i - 1:
                        break  # Found another try block
                    j += 1
                
                if not found_except_or_finally:
                    issues.append({
                        'line': i,
                        'type': 'malformed_try',
                        'message': 'try block without except or finally clause',
                        'code': line.rstrip()
                    })
            
            # Look for specific pattern mentioned in CI
            if (i == 299 or i == 300) and 'SecureQueryRequest' in line:
                # Check if this is inside a proper try-except
                in_try_block = False
                for k in range(max(0, i-10), i):
                    if lines[k].strip() == 'try:':
                        in_try_block = True
                        break
                
                if in_try_block:
                    # Look for except clause
                    found_except = False
                    for k in range(i, min(len(lines), i+10)):
                        if lines[k].strip().startswith('except'):
                            found_except = True
                            break
                    
                    if not found_except:
                        issues.append({
                            'line': i,
                            'type': 'missing_except',
                            'message': 'Code in try block but no except clause found',
                            'code': line.rstrip()
                        })
        
        return issues
    
    def check_async_await_syntax(self, file_path: Path, tree: ast.AST) -> List[Dict]:
        """Check for async/await syntax issues."""
        issues = []
        
        class AsyncChecker(ast.NodeVisitor):
            def __init__(self):
                self.async_functions = set()
                self.current_function = None
                self.issues = []
            
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_AsyncFunctionDef(self, node):
                self.async_functions.add(node.name)
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
            
            def visit_Await(self, node):
                # Check if await is used outside async function
                if self.current_function not in self.async_functions:
                    self.issues.append({
                        'line': node.lineno,
                        'type': 'await_outside_async',
                        'message': f'await used outside async function (in {self.current_function or "global scope"})',
                        'code': f'Line {node.lineno}: await expression'
                    })
                self.generic_visit(node)
        
        if tree:
            checker = AsyncChecker()
            checker.visit(tree)
            issues.extend(checker.issues)
        
        return issues
    
    def check_indentation_issues(self, file_path: Path, content: str) -> List[Dict]:
        """Check for indentation inconsistencies."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if line.strip():  # Skip empty lines
                # Check for mixed tabs and spaces
                if '\t' in line and ' ' in line[:len(line) - len(line.lstrip())]:
                    issues.append({
                        'line': i,
                        'type': 'mixed_indentation',
                        'message': 'Mixed tabs and spaces in indentation',
                        'code': repr(line[:20]) + "..."
                    })
        
        return issues
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive validation of a single Python file."""
        self.files_checked += 1
        result = {
            'file': str(file_path),
            'compile_ok': False,
            'ast_ok': False,
            'issues': []
        }
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            result['issues'].append({
                'line': 0,
                'type': 'file_read_error',
                'message': f'Could not read file: {e}',
                'code': ''
            })
            return result
        
        # Compile check
        compile_ok, compile_msg = self.compile_check(file_path)
        result['compile_ok'] = compile_ok
        if not compile_ok:
            # Parse compile error for line number
            line_match = re.search(r'line (\d+)', compile_msg)
            line_num = int(line_match.group(1)) if line_match else 0
            result['issues'].append({
                'line': line_num,
                'type': 'compile_error',
                'message': compile_msg,
                'code': ''
            })
        
        # AST parse check
        ast_ok, ast_msg, tree = self.ast_parse_check(file_path)
        result['ast_ok'] = ast_ok
        if not ast_ok:
            # Parse AST error for details
            line_match = re.search(r'line (\d+)', ast_msg)
            line_num = int(line_match.group(1)) if line_match else 0
            result['issues'].append({
                'line': line_num,
                'type': 'syntax_error',
                'message': ast_msg,
                'code': ''
            })
        
        # Additional checks if parsing succeeded
        if compile_ok and ast_ok:
            # Check try-except blocks
            try_except_issues = self.check_try_except_blocks(file_path, content)
            result['issues'].extend(try_except_issues)
            
            # Check async/await syntax
            async_issues = self.check_async_await_syntax(file_path, tree)
            result['issues'].extend(async_issues)
            
            # Check indentation
            indent_issues = self.check_indentation_issues(file_path, content)
            result['issues'].extend(indent_issues)
        
        return result
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate comprehensive syntax error report."""
        report_lines = [
            "PYTHON SYNTAX ERRORS FOUND:",
            "=" * 50,
            f"Files checked: {self.files_checked}",
            ""
        ]
        
        # Summary statistics
        files_with_errors = [r for r in results if r['issues']]
        compile_errors = sum(1 for r in results if not r['compile_ok'])
        syntax_errors = sum(1 for r in results if not r['ast_ok'])
        
        report_lines.extend([
            f"Files with errors: {len(files_with_errors)}",
            f"Compile errors: {compile_errors}",
            f"Syntax errors: {syntax_errors}",
            "",
            "DETAILED ERRORS:",
            "-" * 30
        ])
        
        # Detailed error listing
        for result in results:
            if result['issues']:
                report_lines.append(f"\nFile: {result['file']}")
                for issue in result['issues']:
                    report_lines.extend([
                        f"Line: {issue['line']}",
                        f"Error: {issue['type']}",
                        f"Issue: {issue['message']}",
                        f"Code: {issue['code']}" if issue['code'] else "",
                        ""
                    ])
        
        return "\n".join(report_lines)
    
    def run_full_audit(self) -> str:
        """Run complete Python syntax audit."""
        print(f"Starting Python syntax validation for: {self.root_path}")
        
        # Find all Python files
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files to check")
        
        # Validate each file
        results = []
        for i, file_path in enumerate(python_files, 1):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(python_files)} files processed")
            
            try:
                result = self.validate_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({
                    'file': str(file_path),
                    'compile_ok': False,
                    'ast_ok': False,
                    'issues': [{
                        'line': 0,
                        'type': 'processing_error',
                        'message': f'Error during validation: {e}',
                        'code': ''
                    }]
                })
        
        # Generate and return report
        report = self.generate_report(results)
        return report

if __name__ == "__main__":
    root_path = sys.argv[1] if len(sys.argv) > 1 else "."
    validator = PythonSyntaxValidator(root_path)
    report = validator.run_full_audit()
    print("\n" + report)