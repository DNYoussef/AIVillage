#!/usr/bin/env python3
"""SQL Security Validation Script.

Comprehensive validation of SQL injection prevention measures across the codebase.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class SQLSecurityValidator:
    """Validate SQL security across the codebase."""

    def __init__(self):
        self.vulnerabilities = []
        self.secure_patterns = []
        self.files_checked = []

    def check_file_for_sql_vulnerabilities(self, file_path: Path) -> List[Dict]:
        """Check a Python file for SQL vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for common SQL injection patterns
            sql_patterns = [
                # String concatenation in SQL (dangerous)
                (r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*\+.*["\']', 
                 "String concatenation in SQL query"),
                
                # f-string with variables in SQL (potentially dangerous)
                (r'f["\'].*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*\{[^}]+\}', 
                 "f-string with variables in SQL query"),
                
                # .format() in SQL (potentially dangerous)
                (r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*\.format\(', 
                 ".format() used in SQL query"),
                
                # % formatting in SQL (potentially dangerous)
                (r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*%[sd]', 
                 "% formatting in SQL query"),
            ]
            
            secure_patterns = [
                # Parameterized queries (good)
                (r'execute\([^)]*,\s*\([^)]*\)\)', "Parameterized query found"),
                (r'execute\([^)]*,\s*\[[^\]]*\]', "Parameterized query with list found"),
                (r'\?.*execute', "Parameter placeholder found"),
            ]
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # Check for vulnerabilities
                for pattern, description in sql_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        vulnerabilities.append({
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip(),
                            'type': 'vulnerability',
                            'description': description,
                            'severity': 'HIGH'
                        })
                
                # Check for secure patterns
                for pattern, description in secure_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        self.secure_patterns.append({
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip(),
                            'description': description
                        })
                        
        except Exception as e:
            print(f"Error checking file {file_path}: {e}")
            
        return vulnerabilities

    def validate_codebase(self, root_path: Path) -> Dict:
        """Validate entire codebase for SQL security."""
        results = {
            'vulnerabilities': [],
            'secure_patterns': 0,
            'files_checked': 0,
            'summary': {}
        }
        
        # Find Python files that might contain database code
        db_related_patterns = [
            '**/database*.py',
            '**/db*.py', 
            '**/sql*.py',
            '**/models*.py',
            '**/repositories*.py',
            '**/*tenant*.py',
            '**/*security*.py'
        ]
        
        python_files = set()
        for pattern in db_related_patterns:
            python_files.update(root_path.glob(pattern))
            
        # Also check specific files mentioned in the task
        specific_files = [
            'core/rag/mcp_servers/hyperag/secure_database.py',
            'infrastructure/shared/security/multi_tenant_system.py'
        ]
        
        for file_path in specific_files:
            full_path = root_path / file_path
            if full_path.exists():
                python_files.add(full_path)
                
        for file_path in python_files:
            if file_path.suffix == '.py':
                self.files_checked.append(str(file_path))
                vulnerabilities = self.check_file_for_sql_vulnerabilities(file_path)
                results['vulnerabilities'].extend(vulnerabilities)
                results['files_checked'] += 1
                
        results['secure_patterns'] = len(self.secure_patterns)
        results['summary'] = self.generate_summary(results)
        
        return results

    def generate_summary(self, results: Dict) -> Dict:
        """Generate security summary."""
        total_vulnerabilities = len(results['vulnerabilities'])
        high_severity = sum(1 for v in results['vulnerabilities'] if v.get('severity') == 'HIGH')
        
        return {
            'total_files_checked': results['files_checked'],
            'total_vulnerabilities': total_vulnerabilities,
            'high_severity_vulnerabilities': high_severity,
            'secure_patterns_found': results['secure_patterns'],
            'risk_level': 'HIGH' if high_severity > 0 else 'LOW' if total_vulnerabilities > 0 else 'MINIMAL'
        }

    def generate_report(self, results: Dict) -> str:
        """Generate a detailed security report."""
        summary = results['summary']
        
        report = f"""
🛡️  SQL INJECTION SECURITY AUDIT REPORT
{'='*50}

📊 SUMMARY
----------
Files Checked: {summary['total_files_checked']}
Total Vulnerabilities: {summary['total_vulnerabilities']}
High Severity: {summary['high_severity_vulnerabilities']}
Secure Patterns Found: {summary['secure_patterns_found']}
Overall Risk Level: {summary['risk_level']}

"""
        
        if results['vulnerabilities']:
            report += "\n🚨 VULNERABILITIES FOUND\n"
            report += "-" * 25 + "\n"
            
            for vuln in results['vulnerabilities']:
                report += f"\n📍 {vuln['file']}:{vuln['line']}\n"
                report += f"   Severity: {vuln['severity']}\n"
                report += f"   Issue: {vuln['description']}\n"
                report += f"   Code: {vuln['content']}\n"
        else:
            report += "\n✅ NO VULNERABILITIES FOUND\n"
            report += "All SQL queries appear to use parameterized statements.\n"
            
        if self.secure_patterns:
            report += f"\n🔒 SECURE PATTERNS DETECTED ({len(self.secure_patterns)})\n"
            report += "-" * 35 + "\n"
            
            # Group by file
            patterns_by_file = {}
            for pattern in self.secure_patterns:
                file_name = Path(pattern['file']).name
                if file_name not in patterns_by_file:
                    patterns_by_file[file_name] = []
                patterns_by_file[file_name].append(pattern)
                
            for file_name, patterns in patterns_by_file.items():
                report += f"\n📁 {file_name}: {len(patterns)} secure patterns\n"
                
        report += "\n🎯 RECOMMENDATIONS\n"
        report += "-" * 17 + "\n"
        
        if results['vulnerabilities']:
            report += "• Fix identified vulnerabilities by using parameterized queries\n"
            report += "• Replace string concatenation with parameter placeholders (?)\n"
            report += "• Validate and sanitize all user inputs\n"
            report += "• Use allowlists for dynamic column/table names\n"
        else:
            report += "✅ Current implementation appears secure\n"
            
        report += "• Continue using parameterized queries for all database operations\n"
        report += "• Implement input validation for all user-provided data\n"
        report += "• Regular security audits and code reviews\n"
        report += "• Consider using an ORM for additional abstraction\n"
        
        return report


def main():
    """Run SQL security validation."""
    print("🔍 Starting SQL Security Validation...")
    
    validator = SQLSecurityValidator()
    results = validator.validate_codebase(project_root)
    
    # Generate and print report
    report = validator.generate_report(results)
    print(report)
    
    # Save report to file
    report_path = project_root / "reports" / "sql_security_audit.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Return appropriate exit code
    if results['summary']['high_severity_vulnerabilities'] > 0:
        print("\n❌ HIGH SEVERITY VULNERABILITIES FOUND - IMMEDIATE ACTION REQUIRED")
        return 1
    elif results['summary']['total_vulnerabilities'] > 0:
        print("\n⚠️  POTENTIAL VULNERABILITIES FOUND - REVIEW RECOMMENDED")
        return 2
    else:
        print("\n✅ NO VULNERABILITIES DETECTED - SECURITY LOOKS GOOD")
        return 0


if __name__ == "__main__":
    sys.exit(main())