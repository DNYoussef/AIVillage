#!/usr/bin/env python3
"""
God Object Detector for Pre-commit Hook
Detects classes that exceed size thresholds (God Objects).
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class GodObjectDetector:
    """Detects God Objects in Python code."""
    
    def __init__(self, line_threshold: int = 500, method_threshold: int = 20):
        self.line_threshold = line_threshold
        self.method_threshold = method_threshold
        self.violations = []
    
    def check_file(self, filepath: Path) -> List[Dict]:
        """Check a single file for God Objects."""
        violations = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    violation = self._check_class(node, filepath)
                    if violation:
                        violations.append(violation)
                        
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        
        return violations
    
    def _check_class(self, node: ast.ClassDef, filepath: Path) -> Dict:
        """Check if a class is a God Object."""
        # Count lines of code
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            lines_of_code = node.end_lineno - node.lineno + 1
        else:
            lines_of_code = 0
        
        # Count methods
        method_count = sum(1 for child in node.body if isinstance(child, ast.FunctionDef))
        
        # Count attributes/properties
        attribute_count = sum(1 for child in node.body 
                            if isinstance(child, ast.Assign) or isinstance(child, ast.AnnAssign))
        
        # Check thresholds
        violations = []
        
        if lines_of_code > self.line_threshold:
            violations.append(f"Exceeds line threshold: {lines_of_code} > {self.line_threshold}")
        
        if method_count > self.method_threshold:
            violations.append(f"Exceeds method threshold: {method_count} > {self.method_threshold}")
        
        # Additional heuristics for God Objects
        if method_count > 15 and lines_of_code > 300:
            violations.append("High method count combined with large size")
        
        if attribute_count > 20:
            violations.append(f"Too many attributes: {attribute_count}")
        
        # Check for diverse responsibilities (method name analysis)
        method_names = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
        responsibility_indicators = self._analyze_method_responsibilities(method_names)
        
        if len(responsibility_indicators) > 3:
            violations.append(f"Multiple responsibilities detected: {', '.join(responsibility_indicators)}")
        
        if violations:
            return {
                "file": str(filepath),
                "class": node.name,
                "line": node.lineno,
                "lines_of_code": lines_of_code,
                "method_count": method_count,
                "attribute_count": attribute_count,
                "violations": violations,
                "severity": self._calculate_severity(lines_of_code, method_count, len(violations))
            }
        
        return None
    
    def _analyze_method_responsibilities(self, method_names: List[str]) -> List[str]:
        """Analyze method names to detect multiple responsibilities."""
        responsibility_patterns = {
            "data": ["get_", "set_", "load_", "save_", "read_", "write_"],
            "validation": ["validate_", "check_", "verify_", "is_valid"],
            "conversion": ["to_", "from_", "convert_", "transform_"],
            "networking": ["send_", "receive_", "connect_", "disconnect_"],
            "ui": ["show_", "hide_", "display_", "render_", "draw_"],
            "calculation": ["calculate_", "compute_", "sum_", "average_"],
            "formatting": ["format_", "print_", "stringify_", "serialize_"],
            "business": ["process_", "handle_", "execute_", "perform_"]
        }
        
        detected_responsibilities = set()
        
        for method_name in method_names:
            for responsibility, patterns in responsibility_patterns.items():
                if any(method_name.startswith(pattern) for pattern in patterns):
                    detected_responsibilities.add(responsibility)
        
        return list(detected_responsibilities)
    
    def _calculate_severity(self, lines: int, methods: int, violation_count: int) -> str:
        """Calculate severity based on violations."""
        score = 0
        
        # Line penalty
        if lines > self.line_threshold * 2:
            score += 3
        elif lines > self.line_threshold * 1.5:
            score += 2
        elif lines > self.line_threshold:
            score += 1
        
        # Method penalty
        if methods > self.method_threshold * 2:
            score += 3
        elif methods > self.method_threshold * 1.5:
            score += 2
        elif methods > self.method_threshold:
            score += 1
        
        # Violation count penalty
        score += violation_count
        
        if score >= 6:
            return "critical"
        elif score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
    
    def check_files(self, filepaths: List[Path]) -> List[Dict]:
        """Check multiple files for God Objects."""
        all_violations = []
        
        for filepath in filepaths:
            violations = self.check_file(filepath)
            all_violations.extend(violations)
        
        return all_violations
    
    def format_violations(self, violations: List[Dict], show_details: bool = True) -> str:
        """Format violations for output."""
        if not violations:
            return "✅ No God Objects detected!"
        
        output = []
        output.append(f"❌ Found {len(violations)} God Object(s):")
        output.append("")
        
        # Group by severity
        by_severity = {}
        for violation in violations:
            severity = violation["severity"]
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(violation)
        
        # Output by severity (critical first)
        severity_order = ["critical", "high", "medium", "low"]
        severity_emojis = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "📝"
        }
        
        for severity in severity_order:
            if severity in by_severity:
                output.append(f"{severity_emojis[severity]} {severity.upper()} SEVERITY:")
                
                for violation in by_severity[severity]:
                    output.append(f"  📁 {violation['file']}:{violation['line']}")
                    output.append(f"     Class: {violation['class']}")
                    output.append(f"     Lines: {violation['lines_of_code']}, Methods: {violation['method_count']}")
                    
                    if show_details:
                        for v in violation['violations']:
                            output.append(f"     - {v}")
                    
                    output.append("")
        
        # Add refactoring suggestions
        output.append("🛠️  REFACTORING SUGGESTIONS:")
        output.append("   1. Extract related methods into separate classes")
        output.append("   2. Use composition instead of inheritance")
        output.append("   3. Apply Single Responsibility Principle")
        output.append("   4. Consider using Strategy or Command patterns")
        output.append("   5. Move data-related methods to data classes")
        
        return "\n".join(output)


def main():
    """Main entry point for God Object detection."""
    parser = argparse.ArgumentParser(description="Detect God Objects in Python code")
    parser.add_argument("files", nargs="*", help="Python files to check")
    parser.add_argument("--threshold", type=int, default=500,
                       help="Line count threshold for God Objects")
    parser.add_argument("--method-threshold", type=int, default=20,
                       help="Method count threshold")
    parser.add_argument("--severity", choices=["low", "medium", "high", "critical"],
                       default="medium", help="Minimum severity to report")
    parser.add_argument("--json", action="store_true",
                       help="Output in JSON format")
    parser.add_argument("--count", action="store_true",
                       help="Output only the count of violations")
    
    args = parser.parse_args()
    
    # Get files to check
    if args.files:
        filepaths = [Path(f) for f in args.files if f.endswith('.py')]
    else:
        # Find all Python files in current directory
        filepaths = list(Path('.').rglob('*.py'))
        # Exclude common directories
        excluded_dirs = {'__pycache__', '.git', 'venv', 'env', 'node_modules', 'deprecated', 'archive'}
        filepaths = [f for f in filepaths if not any(part in excluded_dirs for part in f.parts)]
    
    # Initialize detector
    detector = GodObjectDetector(
        line_threshold=args.threshold,
        method_threshold=args.method_threshold
    )
    
    # Check files
    violations = detector.check_files(filepaths)
    
    # Filter by severity
    severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    min_level = severity_levels[args.severity]
    
    filtered_violations = [
        v for v in violations
        if severity_levels.get(v["severity"], 1) >= min_level
    ]
    
    # Output results
    if args.count:
        print(len(filtered_violations))
    elif args.json:
        import json
        print(json.dumps(filtered_violations, indent=2))
    else:
        print(detector.format_violations(filtered_violations))
    
    # Exit with error code if violations found
    if filtered_violations:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()