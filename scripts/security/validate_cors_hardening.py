#!/usr/bin/env python3
"""
CORS Security Validation Script
==============================

Validates that all CORS configurations have been hardened from wildcards to secure policies.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def find_cors_wildcards(directory: Path) -> List[Tuple[str, int, str]]:
    """
    Find all remaining CORS wildcard configurations.
    
    Returns:
        List of (filename, line_number, line_content) tuples
    """
    wildcards = []
    
    for py_file in directory.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if re.search(r'allow_origins.*\["?\*"?\]', line):
                        wildcards.append((str(py_file), line_num, line.strip()))
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return wildcards


def validate_secure_cors(directory: Path) -> Dict[str, List[str]]:
    """
    Validate secure CORS implementations.
    
    Returns:
        Dictionary with 'secure' and 'insecure' file lists
    """
    results = {"secure": [], "insecure": [], "mixed": []}
    
    for py_file in directory.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "allow_origins" not in content:
                    continue  # No CORS configuration
                
                has_wildcard = bool(re.search(r'allow_origins.*\["?\*"?\]', content))
                has_secure = bool(re.search(r'SECURE_CORS_CONFIG|ADMIN_CORS_CONFIG|WEBSOCKET_CORS_CONFIG', content))
                has_env_config = bool(re.search(r'os\.getenv.*CORS_ORIGINS', content))
                
                if has_wildcard and (has_secure or has_env_config):
                    results["mixed"].append(str(py_file))
                elif has_wildcard:
                    results["insecure"].append(str(py_file))
                elif has_secure or has_env_config:
                    results["secure"].append(str(py_file))
                else:
                    results["insecure"].append(str(py_file))
                    
        except Exception as e:
            print(f"Warning: Could not analyze {py_file}: {e}")
    
    return results


def generate_security_report() -> str:
    """Generate comprehensive security validation report."""
    
    project_root = Path(__file__).parent.parent.parent
    gateway_dir = project_root / "infrastructure" / "gateway"
    
    print("🔍 CORS Security Validation Report")
    print("=" * 50)
    
    # Find remaining wildcards
    wildcards = find_cors_wildcards(gateway_dir)
    print(f"\n📊 Wildcard Analysis:")
    print(f"   Remaining CORS wildcards: {len(wildcards)}")
    
    if wildcards:
        print(f"\n⚠️  REMAINING WILDCARDS TO FIX:")
        for filename, line_num, line_content in wildcards:
            rel_path = Path(filename).relative_to(project_root)
            print(f"   {rel_path}:{line_num} - {line_content}")
    
    # Validate secure implementations
    validation_results = validate_secure_cors(gateway_dir)
    print(f"\n🛡️  Security Implementation Status:")
    print(f"   Secure implementations: {len(validation_results['secure'])}")
    print(f"   Insecure implementations: {len(validation_results['insecure'])}")
    print(f"   Mixed implementations: {len(validation_results['mixed'])}")
    
    # Calculate security score
    total_files = len(validation_results['secure']) + len(validation_results['insecure']) + len(validation_results['mixed'])
    if total_files > 0:
        security_score = (len(validation_results['secure']) / total_files) * 100
        print(f"   Security Score: {security_score:.1f}%")
    else:
        security_score = 100.0
        print(f"   Security Score: No CORS configurations found")
    
    # List secure implementations
    if validation_results['secure']:
        print(f"\n✅ SECURE IMPLEMENTATIONS:")
        for filename in validation_results['secure']:
            rel_path = Path(filename).relative_to(project_root)
            print(f"   ✓ {rel_path}")
    
    # Generate recommendations
    print(f"\n📋 SECURITY RECOMMENDATIONS:")
    
    if wildcards:
        print(f"   🚨 HIGH PRIORITY: Fix {len(wildcards)} remaining wildcard configurations")
        print(f"      Use: from src.security.cors_config import SECURE_CORS_CONFIG")
    else:
        print(f"   ✅ No wildcard CORS configurations found")
    
    if validation_results['insecure']:
        print(f"   ⚠️  MEDIUM PRIORITY: Secure {len(validation_results['insecure'])} insecure implementations")
    
    if security_score < 90:
        print(f"   📊 Security score below 90% - recommend additional hardening")
    elif security_score >= 95:
        print(f"   🎯 Excellent security score - CORS hardening successful!")
    
    # Environment configuration check
    print(f"\n🔧 ENVIRONMENT CONFIGURATION:")
    cors_env = os.getenv("CORS_ORIGINS")
    if cors_env:
        print(f"   CORS_ORIGINS: {cors_env}")
    else:
        print(f"   CORS_ORIGINS: Not set (using defaults)")
    
    aivillage_env = os.getenv("AIVILLAGE_ENV", "development")
    print(f"   AIVILLAGE_ENV: {aivillage_env}")
    
    return f"Security Score: {security_score:.1f}%, Wildcards Remaining: {len(wildcards)}"


if __name__ == "__main__":
    try:
        summary = generate_security_report()
        print(f"\n📝 SUMMARY: {summary}")
        
        # Exit with appropriate code
        wildcards = find_cors_wildcards(Path(__file__).parent.parent.parent / "infrastructure" / "gateway")
        sys.exit(1 if wildcards else 0)
        
    except Exception as e:
        print(f"❌ Error generating security report: {e}")
        sys.exit(1)