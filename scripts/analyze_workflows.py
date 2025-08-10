#!/usr/bin/env python3
"""Analyze GitHub Actions workflows for issues."""

import sys
from pathlib import Path

import yaml


def analyze_workflow(workflow_path) -> bool | None:
    """Analyze a single workflow file."""
    print(f"Analyzing {workflow_path}")

    try:
        with open(workflow_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        issues = []

        # Check required fields
        if "name" not in workflow:
            issues.append("Missing 'name' field")

        if "on" not in workflow:
            issues.append("Missing 'on' trigger field")

        if "jobs" not in workflow:
            issues.append("Missing 'jobs' field")
        elif not workflow["jobs"]:
            issues.append("No jobs defined")

        # Check each job
        if "jobs" in workflow:
            for job_name, job in workflow["jobs"].items():
                if "runs-on" not in job:
                    issues.append(f"Job '{job_name}' missing 'runs-on'")

                if "steps" not in job:
                    issues.append(f"Job '{job_name}' missing 'steps'")

        # Report results
        if issues:
            print("  ISSUES FOUND:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        print(
            f"  VALID: {workflow.get('name', 'Unnamed')} - {len(workflow.get('jobs', {}))} jobs"
        )
        return True

    except yaml.YAMLError as e:
        print(f"  YAML SYNTAX ERROR: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> int:
    """Analyze all workflows."""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        print("No .github/workflows directory found!")
        return 1

    workflows = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))

    if not workflows:
        print("No workflow files found!")
        return 1

    print(f"Found {len(workflows)} workflow files\n")

    all_valid = True
    for workflow in workflows:
        if not analyze_workflow(workflow):
            all_valid = False
        print()

    if all_valid:
        print("All workflows have valid syntax!")
        return 0
    print("Some workflows have issues that need fixing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
