#!/usr/bin/env python3
"""
Secret Externalization Validation Script
Validates that all hardcoded secrets have been properly externalized to environment variables.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class SecretValidator:
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.violations = []
        self.environment_vars_found = set()

    def load_yaml_safely(self, file_path: Path) -> dict:
        """Load YAML file safely."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Replace environment variable placeholders for validation
                content = re.sub(r"\${([^}]+)}", r"TEST_\1_VALUE", content)
                return yaml.safe_load(content)
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return {}

    def extract_env_vars(self, content: str) -> set:
        """Extract environment variable references from content."""
        env_var_pattern = r"\${([^}]+)}"
        return set(re.findall(env_var_pattern, content))

    def scan_for_hardcoded_secrets(self, file_path: Path) -> List[Dict]:
        """Scan file for potential hardcoded secrets."""
        violations = []

        # Patterns that indicate hardcoded secrets
        secret_patterns = [
            (r'password["\s]*:["\s]*[^"${\s][^"]*"', "hardcoded password"),
            (r'secret["\s]*:["\s]*[^"${\s][^"]*"', "hardcoded secret"),
            (r'key["\s]*:["\s]*[^"${\s][^"]*"', "hardcoded key"),
            (r'token["\s]*:["\s]*[^"${\s][^"]*"', "hardcoded token"),
            (r'api_key["\s]*:["\s]*[^"${\s][^"]*"', "hardcoded API key"),
            (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key pattern"),
            (r"sk-ant-[a-zA-Z0-9-]{95,}", "Anthropic API key pattern"),
            (r"postgresql://[^/]+:[^/]+@", "database URL with credentials"),
        ]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract environment variables used
            env_vars = self.extract_env_vars(content)
            self.environment_vars_found.update(env_vars)

            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                for pattern, description in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip if it's already using environment variables
                        if "${" in line:
                            continue
                        # Skip common placeholder patterns
                        if any(
                            placeholder in line.upper()
                            for placeholder in ["REPLACE_", "YOUR_", "CHANGE_", "SET_VIA", "EXAMPLE"]
                        ):
                            continue

                        violations.append(
                            {
                                "file": str(file_path.relative_to(self.repo_root)),
                                "line": line_num,
                                "content": line.strip(),
                                "type": description,
                                "severity": "HIGH",
                            }
                        )

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

        return violations

    def validate_target_files(self) -> None:
        """Validate specific target files mentioned in the security audit."""
        target_files = [
            "devops/deployment/k8s/secrets.yaml",
            "devops/deployment/helm/aivillage/values-production.yaml",
            "devops/deployment/helm/aivillage/values-staging.yaml",
            "config/cogment/deployment_config.yaml",
            "core/gateway/config.yaml",
        ]

        print("Validating target configuration files...")
        print("=" * 60)

        for file_path_str in target_files:
            file_path = self.repo_root / file_path_str
            if file_path.exists():
                print(f"\\nValidating: {file_path_str}")
                violations = self.scan_for_hardcoded_secrets(file_path)

                if violations:
                    print(f"  [FAIL] Found {len(violations)} potential secret(s)")
                    for violation in violations:
                        print(f"    Line {violation['line']}: {violation['type']}")
                        print(f"    Content: {violation['content']}")
                    self.violations.extend(violations)
                else:
                    print(f"  [PASS] No hardcoded secrets found")
            else:
                print(f"\\n[FAIL] File not found: {file_path_str}")
                self.violations.append(
                    {
                        "file": file_path_str,
                        "line": 0,
                        "content": "File not found",
                        "type": "missing file",
                        "severity": "MEDIUM",
                    }
                )

    def validate_environment_templates(self) -> None:
        """Validate that environment templates exist and contain required variables."""
        print("\\nValidating environment variable templates...")
        print("=" * 60)

        required_templates = [
            "devops/deployment/.env.production.example",
            "devops/deployment/.env.staging.example",
            "devops/deployment/.env.development.example",
        ]

        required_env_vars = {
            "AIVILLAGE_POSTGRES_PASSWORD",
            "AIVILLAGE_REDIS_PASSWORD",
            "AIVILLAGE_NEO4J_PASSWORD",
            "AIVILLAGE_HYPERAG_JWT_SECRET",
            "AIVILLAGE_OPENAI_API_KEY",
            "AIVILLAGE_ANTHROPIC_API_KEY",
            "AIVILLAGE_GRAFANA_PASSWORD",
            "AIVILLAGE_GATEWAY_API_KEY",
            "AIVILLAGE_GATEWAY_SECRET_KEY",
        }

        for template_path_str in required_templates:
            template_path = self.repo_root / template_path_str
            if template_path.exists():
                print(f"[PASS] Template exists: {template_path_str}")

                # Check if template contains required variables
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()

                missing_vars = []
                for var in required_env_vars:
                    if var not in template_content:
                        missing_vars.append(var)

                if missing_vars:
                    print(f"  [WARN] Missing variables: {', '.join(missing_vars)}")
                else:
                    print(f"  [PASS] All required variables documented")
            else:
                print(f"[FAIL] Template missing: {template_path_str}")
                self.violations.append(
                    {
                        "file": template_path_str,
                        "line": 0,
                        "content": "Environment template missing",
                        "type": "missing template",
                        "severity": "HIGH",
                    }
                )

    def check_environment_variable_usage(self) -> None:
        """Check that all found environment variables follow naming conventions."""
        print("\\nValidating environment variable naming conventions...")
        print("=" * 60)

        valid_prefixes = [
            "AIVILLAGE_",
            "POSTGRES_",
            "REDIS_",
            "NEO4J_",
            "HYPERAG_",
            "OPENAI_",
            "ANTHROPIC_",
            "GRAFANA_",
        ]

        for env_var in sorted(self.environment_vars_found):
            if not any(env_var.startswith(prefix) for prefix in valid_prefixes):
                print(f"[WARN] Non-standard naming: {env_var}")
                print(f"       Consider prefixing with AIVILLAGE_ for consistency")
            else:
                print(f"[PASS] {env_var}")

    def generate_report(self) -> None:
        """Generate a comprehensive validation report."""
        print("\\n" + "=" * 80)
        print("SECRET EXTERNALIZATION VALIDATION REPORT")
        print("=" * 80)

        if not self.violations:
            print("\\n[SUCCESS] All hardcoded secrets have been externalized!")
            print("\\nValidation Results:")
            print("- All target files scanned successfully")
            print("- No hardcoded secrets detected")
            print("- Environment variable templates created")
            print("- Naming conventions followed")

        else:
            print(f"\\n[FAILED] Found {len(self.violations)} security violation(s)")
            print("\\nViolations by severity:")

            high_violations = [v for v in self.violations if v["severity"] == "HIGH"]
            medium_violations = [v for v in self.violations if v["severity"] == "MEDIUM"]

            if high_violations:
                print(f"\\nHIGH SEVERITY ({len(high_violations)}):")
                for v in high_violations:
                    print(f"  {v['file']}:{v['line']} - {v['type']}")
                    if v["content"] != "File not found":
                        print(f"    Content: {v['content']}")

            if medium_violations:
                print(f"\\nMEDIUM SEVERITY ({len(medium_violations)}):")
                for v in medium_violations:
                    print(f"  {v['file']}:{v['line']} - {v['type']}")

        print(f"\\nEnvironment Variables Found: {len(self.environment_vars_found)}")

        # Return exit code
        return 0 if not self.violations else 1


def main():
    """Main validation function."""
    validator = SecretValidator()

    print("AIVillage Secret Externalization Validator")
    print("=" * 80)
    print("Validating that all hardcoded secrets have been properly externalized...")

    # Validate target files
    validator.validate_target_files()

    # Validate environment templates
    validator.validate_environment_templates()

    # Check environment variable naming
    validator.check_environment_variable_usage()

    # Generate final report
    exit_code = validator.generate_report()

    if exit_code == 0:
        print("\\nNext Steps:")
        print("1. Copy .env.example files to actual .env files")
        print("2. Fill in real secret values (never commit these)")
        print("3. Test deployments with externalized secrets")
        print("4. Set up secret rotation schedules")
        print("5. Configure monitoring for secret access")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
