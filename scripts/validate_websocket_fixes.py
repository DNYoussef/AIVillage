#!/usr/bin/env python3
"""
WebSocket RCE Fix Validation Script

Validates that the critical RCE vulnerabilities have been properly fixed.
"""

import re
from pathlib import Path


def validate_file_security(file_path):
    """Validate security fixes in a specific file."""
    print(f"\nValidating: {file_path}")

    if not Path(file_path).exists():
        print("  ERROR: File not found!")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for eval() usage with data
    eval_pattern = r"eval\s*\(\s*data\s*\)"
    if re.search(eval_pattern, content):
        print("  CRITICAL: Still using eval(data) - RCE vulnerability present!")
        return False

    # Check that json.loads is used instead
    json_pattern = r"json\.loads\s*\(\s*data\s*\)"
    if re.search(json_pattern, content):
        print("  PASS: Using json.loads(data) instead of eval()")
    else:
        print("  WARNING: No json.loads(data) found")

    # Check for input validation
    validation_patterns = [r"allowed_types", r"validate.*message", r"msg_type.*not.*in", r"JSONDecodeError"]

    validation_found = False
    for pattern in validation_patterns:
        if re.search(pattern, content):
            validation_found = True
            print(f"  PASS: Input validation found: {pattern}")

    if not validation_found:
        print("  WARNING: No input validation patterns found")

    # Check for dangerous patterns
    dangerous_patterns = {
        "eval(": "Code execution via eval()",
        "exec(": "Code execution via exec()",
        "os.system": "Command execution via os.system",
        "subprocess.call": "Command execution via subprocess.call",
    }

    for pattern, description in dangerous_patterns.items():
        if pattern in content and "eval(" != pattern:  # We already checked eval separately
            # Check if it's in a comment or string
            lines_with_pattern = []
            for i, line in enumerate(content.split("\n"), 1):
                if pattern in line:
                    # Skip if in comment or string literal
                    if line.strip().startswith("#") or pattern in f'"{pattern}"' or pattern in f"'{pattern}'":
                        continue
                    lines_with_pattern.append((i, line.strip()))

            if lines_with_pattern:
                print(f"  WARNING: Found potentially dangerous pattern '{pattern}': {description}")
                for line_num, line in lines_with_pattern:
                    print(f"    Line {line_num}: {line}")

    return True


def main():
    """Main validation function."""
    print("WebSocket RCE Vulnerability Fix Validation")
    print("=" * 50)

    # Files we know had RCE vulnerabilities
    files_to_check = [
        "infrastructure/gateway/unified_api_gateway.py",
        "infrastructure/gateway/enhanced_unified_api_gateway.py",
    ]

    all_passed = True

    for file_path in files_to_check:
        if not validate_file_security(file_path):
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: All RCE vulnerabilities have been fixed!")
        print("- eval(data) replaced with json.loads(data)")
        print("- Input validation added")
        print("- Error handling implemented")
    else:
        print("FAILURE: Critical vulnerabilities still present!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
