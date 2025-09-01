#!/usr/bin/env python3
"""
Critical Syntax Issues Detector
Focused analysis on the most critical syntax problems
"""
import os
import re
from pathlib import Path

def analyze_critical_files():
    """Analyze the most critical files for syntax issues"""
    
    critical_files = [
        "core/gateway/server.py",
        "infrastructure/gateway/server.py", 
        "src/security/admin/secure_admin_server_refactored.py"
    ]
    
    issues = []
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                issues.append({
                    'file': file_path,
                    'line': 0,
                    'error': f'File read error: {e}',
                    'type': 'file_access'
                })
                continue
            
            # Check for Unicode issues
            unicode_chars = []
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Look for non-ASCII characters that might cause encoding issues
                for char in line:
                    if ord(char) > 127:
                        if char not in ['\u2192', '\u2713', '\u26a0']:  # Known problematic chars
                            continue
                        unicode_chars.append({
                            'line': i,
                            'char': char,
                            'ord': ord(char),
                            'context': line.strip()[:50]
                        })
            
            if unicode_chars:
                for uc in unicode_chars:
                    issues.append({
                        'file': file_path,
                        'line': uc['line'],
                        'error': f'Unicode character {uc["char"]} (ord {uc["ord"]}) causes encoding issues',
                        'type': 'unicode_encoding',
                        'context': uc['context']
                    })
            
            # Check for try-except patterns around lines 299-300
            for i, line in enumerate(lines, 1):
                if 295 <= i <= 305:
                    line_stripped = line.strip()
                    
                    # Look for try without proper except
                    if line_stripped == 'try:':
                        # Check next few lines for except
                        found_except = False
                        for j in range(i, min(len(lines), i + 10)):
                            if j < len(lines) and lines[j].strip().startswith('except'):
                                found_except = True
                                break
                        
                        if not found_except:
                            issues.append({
                                'file': file_path,
                                'line': i,
                                'error': 'try block without except clause in critical range (295-305)',
                                'type': 'missing_except',
                                'context': line.strip()
                            })
                    
                    # Look for SecureQueryRequest usage
                    if 'SecureQueryRequest' in line:
                        issues.append({
                            'file': file_path,
                            'line': i,
                            'error': 'SecureQueryRequest usage in critical line range',
                            'type': 'critical_usage',
                            'context': line.strip()
                        })
    
    return issues

def generate_report(issues):
    """Generate formatted syntax error report"""
    
    report = [
        "PYTHON SYNTAX ERRORS FOUND:",
        "=" * 50,
        f"Critical Issues Detected: {len(issues)}",
        ""
    ]
    
    # Group by file
    by_file = {}
    for issue in issues:
        if issue['file'] not in by_file:
            by_file[issue['file']] = []
        by_file[issue['file']].append(issue)
    
    for file_path, file_issues in by_file.items():
        report.append(f"\nFile: {file_path}")
        report.append("-" * len(f"File: {file_path}"))
        
        for issue in file_issues:
            report.extend([
                f"Line: {issue['line']}",
                f"Error: {issue['type']}",
                f"Issue: {issue['error']}",
                f"Context: {issue.get('context', '')}",
                ""
            ])
    
    # Summary by type
    report.extend([
        "\nSUMMARY BY ERROR TYPE:",
        "-" * 30
    ])
    
    type_counts = {}
    for issue in issues:
        issue_type = issue['type']
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
    
    for error_type, count in sorted(type_counts.items()):
        report.append(f"{error_type}: {count} occurrences")
    
    return "\n".join(report)

if __name__ == "__main__":
    print("Critical Syntax Issues Analysis")
    print("=" * 40)
    
    issues = analyze_critical_files()
    
    if issues:
        report = generate_report(issues)
        print("\n" + report)
        
        # Save report
        with open('scripts/critical_syntax_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: scripts/critical_syntax_report.txt")
    else:
        print("No critical syntax issues found in analyzed files.")