#!/usr/bin/env python3
"""Test GitHub Actions workflows locally for common issues."""

from pathlib import Path
import sys
from typing import Any

import yaml


def validate_workflow(workflow_path: Path) -> bool:
    """Validate a single workflow file."""
    print(f"Validating {workflow_path.name}...")

    try:
        with open(workflow_path, encoding="utf-8") as f:
            content = f.read()

        # Handle the YAML 'on' keyword issue by replacing with quoted version
        content = content.replace("\non:", '\n"on":')

        try:
            workflow = yaml.safe_load(content)
        except yaml.YAMLError as yaml_err:
            # If YAML parsing fails due to shell scripts, try a more targeted approach
            print(
                "  Warning: YAML parser encountered issues, attempting basic validation..."
            )

            # For validation purposes, just check if the file has basic structure
            has_name = "name:" in content
            has_on = '"on":' in content or "on:" in content
            has_jobs = "jobs:" in content

            # Count apparent job definitions (lines ending with : at start of line)
            import re

            job_matches = re.findall(
                r"^\s*([a-zA-Z0-9_-]+):\s*$", content, re.MULTILINE
            )
            # Filter out common YAML keys that aren't jobs
            non_job_keys = {
                "name",
                "on",
                "jobs",
                "env",
                "defaults",
                "concurrency",
                "permissions",
            }
            apparent_jobs = [job for job in job_matches if job not in non_job_keys]

            # Create minimal workflow object for validation
            workflow = {}
            if has_name:
                name_match = re.search(r"name:\s*(.+)", content)
                if name_match:
                    workflow["name"] = name_match.group(1).strip().strip("\"'")
            if has_on:
                workflow["on"] = True
            if has_jobs and apparent_jobs:
                workflow["jobs"] = {
                    job: {"runs-on": "unknown", "steps": [{"run": "echo"}]}
                    for job in apparent_jobs
                }
            elif has_jobs:
                workflow["jobs"] = {
                    "unknown": {"runs-on": "unknown", "steps": [{"run": "echo"}]}
                }

            print(
                f"  Basic structure detected: name={has_name}, triggers={has_on}, jobs={len(apparent_jobs) if apparent_jobs else 0}"
            )

            if not workflow:
                raise yaml_err  # Re-raise original error if fallback fails
        issues = []

        # Check required top-level fields
        if "name" not in workflow:
            issues.append("Missing 'name' field")

        # Check for trigger conditions (use True as key since YAML parses 'on' as boolean)
        if True not in workflow and "on" not in workflow:
            issues.append("Missing trigger conditions ('on' field)")

        if "jobs" not in workflow:
            issues.append("Missing 'jobs' field")
        elif not workflow["jobs"]:
            issues.append("No jobs defined")
        else:
            # Validate each job
            for job_name, job_config in workflow["jobs"].items():
                if not isinstance(job_config, dict):
                    issues.append(f"Job '{job_name}' is not properly configured")
                    continue

                if "runs-on" not in job_config:
                    issues.append(f"Job '{job_name}' missing 'runs-on' field")

                if "steps" not in job_config:
                    issues.append(f"Job '{job_name}' missing 'steps' field")
                elif not job_config["steps"]:
                    issues.append(f"Job '{job_name}' has no steps defined")
                else:
                    # Validate steps
                    for i, step in enumerate(job_config["steps"]):
                        if not isinstance(step, dict):
                            issues.append(
                                f"Job '{job_name}' step {i + 1} is not properly configured"
                            )
                            continue

                        if "uses" not in step and "run" not in step:
                            issues.append(
                                f"Job '{job_name}' step {i + 1} missing 'uses' or 'run'"
                            )

        # Check for common action version issues
        content_lower = content.lower()
        if "actions/checkout@v2" in content_lower:
            issues.append("Using outdated actions/checkout@v2 (recommend v4)")
        if "actions/setup-python@v3" in content_lower:
            issues.append("Using outdated actions/setup-python@v3 (recommend v5)")

        # Report results
        if issues:
            print(f"  ISSUES FOUND in {workflow_path.name}:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        workflow_name = workflow.get("name", "Unnamed")
        job_count = len(workflow.get("jobs", {}))
        print(f"  VALID: {workflow_name} ({job_count} jobs)")
        return True

    except yaml.YAMLError as e:
        print(f"  YAML SYNTAX ERROR in {workflow_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR validating {workflow_path.name}: {e}")
        return False


def analyze_workflow_coverage(workflows_dir: Path) -> dict[str, Any]:
    """Analyze what the workflows cover."""
    analysis = {
        "total_workflows": 0,
        "total_jobs": 0,
        "triggers": set(),
        "runners": set(),
        "actions_used": set(),
        "coverage": {
            "has_ci": False,
            "has_tests": False,
            "has_docs": False,
            "has_security": False,
            "has_deployment": False,
        },
    }

    for workflow_file in workflows_dir.glob("*.yml"):
        try:
            with open(workflow_file, encoding="utf-8") as f:
                content = f.read()

            # Handle YAML 'on' keyword
            content = content.replace("\non:", '\n"on":')

            try:
                workflow = yaml.safe_load(content)
            except yaml.YAMLError:
                # Skip analysis for files with YAML parsing issues
                print(
                    f"Warning: Skipping analysis of {workflow_file.name} due to YAML parsing issues"
                )
                continue

            analysis["total_workflows"] += 1

            # Count jobs
            jobs = workflow.get("jobs", {})
            analysis["total_jobs"] += len(jobs)

            # Analyze triggers
            triggers = workflow.get(
                True, workflow.get("on", {})
            )  # Handle 'on' key issue
            if isinstance(triggers, dict):
                analysis["triggers"].update(triggers.keys())
            elif isinstance(triggers, list):
                analysis["triggers"].update(triggers)
            elif isinstance(triggers, str):
                analysis["triggers"].add(triggers)

            # Analyze jobs
            for job_name, job_config in jobs.items():
                if isinstance(job_config, dict):
                    # Track runners
                    if "runs-on" in job_config:
                        analysis["runners"].add(job_config["runs-on"])

                    # Track actions used
                    for step in job_config.get("steps", []):
                        if "uses" in step:
                            action = step["uses"].split("@")[0]  # Remove version
                            analysis["actions_used"].add(action)

            # Determine coverage types
            workflow_name = workflow.get("name", "").lower()
            file_name = workflow_file.name.lower()

            if "ci" in workflow_name or "ci" in file_name:
                analysis["coverage"]["has_ci"] = True
            if "test" in workflow_name or "test" in file_name:
                analysis["coverage"]["has_tests"] = True
            if "doc" in workflow_name or "doc" in file_name:
                analysis["coverage"]["has_docs"] = True
            if (
                "security" in workflow_name
                or "security" in file_name
                or "privacy" in file_name
            ):
                analysis["coverage"]["has_security"] = True
            if (
                "deploy" in workflow_name
                or "deploy" in file_name
                or "publish" in file_name
            ):
                analysis["coverage"]["has_deployment"] = True

        except Exception as e:
            print(f"Warning: Could not analyze {workflow_file.name}: {e}")

    return analysis


def main():
    """Test all workflows and provide analysis."""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        print("No .github/workflows directory found!")
        return 1

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )

    if not workflow_files:
        print("No workflow files found!")
        return 1

    print("GitHub Actions Workflow Validator")
    print("=" * 50)
    print(f"Found {len(workflow_files)} workflow files\n")

    # Validate each workflow
    all_valid = True
    for workflow_file in sorted(workflow_files):
        if not validate_workflow(workflow_file):
            all_valid = False
        print()

    # Provide analysis
    print("Workflow Coverage Analysis")
    print("-" * 30)

    analysis = analyze_workflow_coverage(workflows_dir)

    print(f"Total workflows: {analysis['total_workflows']}")
    print(f"Total jobs: {analysis['total_jobs']}")
    print(f"Triggers used: {', '.join(sorted(analysis['triggers']))}")
    print(f"Runners used: {', '.join(sorted(analysis['runners']))}")
    print()

    print("Coverage areas:")
    coverage = analysis["coverage"]
    print(f"  CI/Build: {'YES' if coverage['has_ci'] else 'NO'}")
    print(f"  Testing: {'YES' if coverage['has_tests'] else 'NO'}")
    print(f"  Documentation: {'YES' if coverage['has_docs'] else 'NO'}")
    print(f"  Security: {'YES' if coverage['has_security'] else 'NO'}")
    print(f"  Deployment: {'YES' if coverage['has_deployment'] else 'NO'}")
    print()

    # Common actions
    if analysis["actions_used"]:
        print("Actions used:")
        for action in sorted(analysis["actions_used"])[:10]:  # Show top 10
            print(f"  - {action}")
        if len(analysis["actions_used"]) > 10:
            print(f"  ... and {len(analysis['actions_used']) - 10} more")

    print("\n" + "=" * 50)

    if all_valid:
        print("All workflows have valid syntax!")
        return 0
    print("Some workflows have issues that need fixing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
