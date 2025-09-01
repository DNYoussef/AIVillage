#!/usr/bin/env python3
"""
Focused Python Syntax Checker
Target specific syntax patterns and issues
"""
import os
import py_compile
import tempfile
import re
from pathlib import Path

def check_file_syntax(file_path):
    """Check single file for syntax issues"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return [{'line': 0, 'error': f'Read error: {e}', 'code': ''}]
    
    # Quick compile check
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            py_compile.compile(tmp_file.name, doraise=True)
            os.unlink(tmp_file.name)
    except py_compile.PyCompileError as e:
        os.unlink(tmp_file.name) if 'tmp_file' in locals() else None
        error_msg = str(e)
        line_match = re.search(r'line (\d+)', error_msg)
        line_num = int(line_match.group(1)) if line_match else 0
        issues.append({'line': line_num, 'error': error_msg, 'code': ''})
        return issues
    except Exception as e:
        issues.append({'line': 0, 'error': f'Compile error: {e}', 'code': ''})
        return issues
    
    # Check specific patterns
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Check for try without except around lines 299-300
        if (i >= 295 and i <= 305) or line_stripped == 'try:':
            if line_stripped == 'try:':
                # Look ahead for except or finally
                found_handler = False
                for j in range(i, min(len(lines) + 1, i + 15)):
                    if j < len(lines):
                        next_line = lines[j].strip()
                        if next_line.startswith('except') or next_line.startswith('finally'):
                            found_handler = True
                            break
                        elif next_line.startswith('def ') or next_line.startswith('class ') or (next_line.startswith('try:') and j != i):
                            break
                
                if not found_handler:
                    issues.append({
                        'line': i,
                        'error': 'try block without except or finally clause',
                        'code': line.rstrip()
                    })
        
        # Check for SecureQueryRequest usage
        if 'SecureQueryRequest' in line and (290 <= i <= 310):
            issues.append({
                'line': i,
                'error': 'SecureQueryRequest usage around critical line range',
                'code': line.rstrip()
            })
    
    return issues

def main():
    # First, check the most likely problematic files
    critical_patterns = [
        "**/server.py", "**/gateway.py", "**/secure*.py", "**/auth*.py",
        "**/api*.py", "**/request*.py", "**/query*.py"
    ]
    
    print("FOCUSED PYTHON SYNTAX CHECK")
    print("=" * 40)
    
    total_issues = 0
    files_checked = 0
    
    for pattern in critical_patterns:
        for file_path in Path('.').glob(pattern):
            if file_path.suffix == '.py':
                files_checked += 1
                issues = check_file_syntax(file_path)
                if issues:
                    total_issues += len(issues)
                    print(f"\nFile: {file_path}")
                    for issue in issues:
                        print(f"Line {issue['line']}: {issue['error']}")
                        if issue['code']:
                            print(f"Code: {issue['code']}")
    
    # Also check files with recent modifications (from git status)
    print("\nChecking modified files from git status...")
    try:
        import subprocess
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip() and line.endswith('.py'):
                    file_path = line[3:].strip()  # Remove git status prefix
                    if os.path.exists(file_path):
                        files_checked += 1
                        issues = check_file_syntax(Path(file_path))
                        if issues:
                            total_issues += len(issues)
                            print(f"\nModified file: {file_path}")
                            for issue in issues:
                                print(f"Line {issue['line']}: {issue['error']}")
                                if issue['code']:
                                    print(f"Code: {issue['code']}")
    except:
        print("Could not check git modified files")
    
    print(f"\nSUMMARY:")
    print(f"Files checked: {files_checked}")
    print(f"Total syntax issues found: {total_issues}")

if __name__ == "__main__":
    main()